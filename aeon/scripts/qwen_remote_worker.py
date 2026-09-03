#!/usr/bin/env python3
"""Exact, non-daemon worker-side adapter for Aeon's remote Qwen container.

The orchestrating Aeon process remains on .177 and owns coordinator admission,
heartbeats, retry, and release.  This program is copied as part of the immutable
Aeon source closure and is invoked only through fixed BatchMode SSH.  It manages
one exact Docker receipt on the already-selected worker and never contacts or
reimplements the fleet coordinator.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from aeon.core.qwen_capabilities import (
    COMPACT_REMOTE_DOCKER_CAPABILITY_KEYS,
    RTX5000_RELEASE_CANDIDATE_KEY,
    QwenCapabilityError,
    qwen_release_candidate_capability,
    qwen_runtime_capability,
    validate_qwen_capability_manifest_identity,
)
from aeon.core.qwen_fleet_runtime import (
    FLEET_WORKER_CACHE_ROOT,
    WORKER_STATE_ROOT,
    _validated_artifact_cache_request,
    fleet_remote_runtime_resources,
)
from aeon.core.qwen_runtime import (
    _CLAIM_RE,
    _CONTAINER_ID_RE,
    _OWNER_RE,
    _SHA256_RE,
    _UUID_RE,
    RUNTIME_ROOT,
    RUNTIME_STATE_FILE,
    ArtifactIdentity,
    QwenRuntimeError,
    _docker_cli_environment,
    _docker_command,
    _image_config,
    _private_json_read,
    _private_json_write,
    _resolve_container,
    _source_identity,
    _validate_run_dir,
    _validate_token,
    clear_runtime_state,
    current_runtime_state,
    load_artifact_identity,
    local_container_pid,
    local_image_id,
    local_image_size,
    qwen_runtime_liveness,
    reuse_qwen_runtime,
    start_local_runtime,
    stop_qwen_runtime,
)
from aeon.core.utils.io import read_bounded_fd

MAX_REQUEST_BYTES = 1024 * 1024
MODEL_CACHE_ROOT = RUNTIME_ROOT / "models"
RELEASE_ROOT = RUNTIME_ROOT / "releases"
WORKER_CONTROLLER_ROOT = RUNTIME_ROOT / "worker-controllers"


def _runtime_binding(request: dict[str, Any]) -> tuple[Path, Path]:
    """Validate and select exactly one worker lifecycle journal."""

    run_dir = _validate_run_dir(request.get("run_dir"))
    mode = request.get("worker_receipt_mode")
    runtime_id = request.get("runtime_id")
    physical_gpu = request.get("physical_gpu")
    if mode == "per-runtime":
        resources = fleet_remote_runtime_resources(run_dir, physical_gpu)
        if any(
            request.get(field) != expected
            for field, expected in (
                ("runtime_id", resources["runtime_id"]),
                ("run_dir", resources["run_dir"]),
                ("worker_state_path", str(resources["worker_state_path"])),
                ("container_name", resources["container_name"]),
                ("port", resources["remote_port"]),
            )
        ):
            raise QwenRuntimeError("remote worker runtime binding changed")
        return run_dir, resources["worker_state_path"]
    if mode == "legacy":
        fleet_runtime_id = (
            run_dir.name
            if run_dir.parent
            == Path("/home/aday/.local/state/fleet-compute/runs")
            else None
        )
        if (
            runtime_id != fleet_runtime_id
            or request.get("worker_state_path") != str(RUNTIME_STATE_FILE)
        ):
            raise QwenRuntimeError("legacy worker runtime binding changed")
        return run_dir, RUNTIME_STATE_FILE
    raise QwenRuntimeError("remote worker receipt mode is malformed")


def _controller_lock_path(request: dict[str, Any]) -> Path:
    _run_dir, state_path = _runtime_binding(request)
    if state_path == RUNTIME_STATE_FILE:
        return WORKER_CONTROLLER_ROOT / "legacy.lock"
    if state_path.parent != WORKER_STATE_ROOT:
        raise QwenRuntimeError("remote worker controller binding changed")
    return WORKER_CONTROLLER_ROOT / f"{state_path.stem}.lock"


def _open_controller_lock(request: dict[str, Any], *, create: bool) -> int:
    root = WORKER_CONTROLLER_ROOT
    if create:
        root.mkdir(mode=0o700, exist_ok=True)
    try:
        root_metadata = root.lstat()
    except FileNotFoundError as exc:
        raise QwenRuntimeError("remote worker controller receipt is absent") from exc
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or root_metadata.st_mode & 0o077
    ):
        raise QwenRuntimeError("remote worker controller directory is unsafe")
    path = _controller_lock_path(request)
    flags = os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileNotFoundError as exc:
        raise QwenRuntimeError("remote worker controller receipt is absent") from exc
    try:
        metadata = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino)
            != (current.st_dev, current.st_ino)
        ):
            raise QwenRuntimeError("remote worker controller receipt is unsafe")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


_CONTROLLER_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "action",
        "runtime_id",
        "run_dir",
        "worker_state_path",
        "capability_key",
        "capability_manifest_sha256",
        "source_manifest_sha256",
        "physical_gpu",
        "gpu_uuid",
        "claim_id",
        "owner",
        "updated_at",
    }
)


def _controller_identity(request: dict[str, Any]) -> dict[str, Any]:
    run_dir, state_path = _runtime_binding(request)
    capability, _manifest_sha256 = _capability(request, require_enabled=False)
    runtime_id = request.get("runtime_id")
    if runtime_id is not None and (
        not isinstance(runtime_id, str)
        or len(runtime_id) != 35
        or not runtime_id.startswith("fr-")
        or any(ch not in "0123456789abcdef" for ch in runtime_id[3:])
    ):
        raise QwenRuntimeError("remote worker controller runtime ID is malformed")
    physical_gpu = request.get("physical_gpu")
    if isinstance(physical_gpu, bool) or not isinstance(physical_gpu, int):
        raise QwenRuntimeError("remote worker controller GPU is malformed")
    return {
        "runtime_id": runtime_id,
        "run_dir": str(run_dir),
        "worker_state_path": str(state_path),
        "capability_key": capability.key,
        "capability_manifest_sha256": _validate_token(
            request.get("capability_manifest_sha256"),
            _SHA256_RE,
            "capability manifest",
        ),
        "source_manifest_sha256": _validate_token(
            request.get("source_manifest_sha256"),
            _SHA256_RE,
            "source manifest",
        ),
        "physical_gpu": physical_gpu,
        "gpu_uuid": _validate_token(
            request.get("gpu_uuid"), _UUID_RE, "runtime GPU UUID"
        ),
        "claim_id": _validate_token(
            request.get("claim_id"), _CLAIM_RE, "runtime claim"
        ),
        "owner": _validate_token(
            request.get("owner"), _OWNER_RE, "runtime owner"
        ),
    }


def _read_controller_receipt(descriptor: int) -> dict[str, Any]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    payload = read_bounded_fd(descriptor, 8192)
    if not payload or len(payload) > 8192:
        raise QwenRuntimeError("remote worker controller receipt is incomplete")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("remote worker controller receipt is malformed") from exc
    if (
        not isinstance(value, dict)
        or set(value) != _CONTROLLER_RECEIPT_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("status") not in {"active", "quiescent", "recovered"}
        or value.get("action") not in _CONTROLLER_ACTIONS
        or isinstance(value.get("updated_at"), bool)
        or not isinstance(value.get("updated_at"), (int, float))
        or not 0 < float(value["updated_at"]) < time.time() + 300
    ):
        raise QwenRuntimeError("remote worker controller receipt is malformed")
    return value


def _write_controller_receipt(
    descriptor: int,
    identity: dict[str, Any],
    *,
    status: str,
    action: str,
) -> None:
    payload = json.dumps(
        {
            "schema_version": 1,
            "status": status,
            "action": action,
            **identity,
            "updated_at": time.time(),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count <= 0:
            raise QwenRuntimeError("remote worker controller receipt write failed")
        written += count
    os.fsync(descriptor)


@contextmanager
def _hold_controller_lock(
    request: dict[str, Any], *, action: str, create: bool
):
    descriptor = _open_controller_lock(request, create=create)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QwenRuntimeError(
                "remote worker lifecycle controller is already active"
            ) from exc
        identity = _controller_identity(request)
        if create:
            if os.fstat(descriptor).st_size != 0:
                raise QwenRuntimeError(
                    "new remote worker controller receipt is not empty"
                )
        else:
            saved = _read_controller_receipt(descriptor)
            if any(saved.get(key) != value for key, value in identity.items()):
                raise QwenRuntimeError(
                    "remote worker controller receipt identity changed"
                )
        _write_controller_receipt(
            descriptor, identity, status="active", action=action
        )
        completed = False
        try:
            yield
            completed = True
        finally:
            _write_controller_receipt(
                descriptor,
                identity,
                status=(
                    "recovered"
                    if action
                    in {"recover-precontainer", "recover-uncommitted"}
                    and completed
                    else "quiescent"
                ),
                action=action,
            )
    finally:
        os.close(descriptor)


def _per_runtime_worker_state_paths() -> tuple[Path, ...]:
    try:
        root = WORKER_STATE_ROOT.lstat()
    except FileNotFoundError:
        return ()
    if (
        not stat.S_ISDIR(root.st_mode)
        or root.st_uid != os.geteuid()
        or root.st_mode & 0o077
    ):
        raise QwenRuntimeError("remote worker receipt directory is unsafe")
    paths: list[Path] = []
    for path in sorted(WORKER_STATE_ROOT.iterdir()):
        metadata = path.lstat()
        runtime_id = path.name.removesuffix(".json")
        if (
            path.suffix != ".json"
            or len(runtime_id) != 35
            or not runtime_id.startswith("fr-")
            or any(ch not in "0123456789abcdef" for ch in runtime_id[3:])
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink != 1
        ):
            raise QwenRuntimeError("remote worker receipt entry is unsafe")
        paths.append(path)
    return tuple(paths)


def _assert_worker_gpu_slot_available(
    request: dict[str, Any], selected_state_path: Path
) -> None:
    paths = (RUNTIME_STATE_FILE, *_per_runtime_worker_state_paths())
    for path in paths:
        state = current_runtime_state(path)
        if state is None:
            continue
        if path == selected_state_path:
            raise QwenRuntimeError("selected remote worker receipt already exists")
        if state.get("physical_gpu") == request.get("physical_gpu"):
            raise QwenRuntimeError(
                "another exact worker receipt already occupies this GPU slot"
            )


def _request() -> dict[str, Any]:
    payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if not payload or len(payload) > MAX_REQUEST_BYTES:
        raise QwenRuntimeError("remote runtime request size is invalid")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("remote runtime request is malformed") from exc
    if not isinstance(value, dict):
        raise QwenRuntimeError("remote runtime request is not an object")
    return value


def _capability(request: dict[str, Any], *, require_enabled: bool = True):
    try:
        if request.get("release_gate") is True:
            if require_enabled:
                capability, manifest_sha256 = qwen_release_candidate_capability(
                    request.get("capability_key")
                )
            else:
                capability, manifest_sha256 = qwen_runtime_capability(
                    request.get("capability_key"), require_enabled=False
                )
        elif request.get("release_gate") is False:
            capability, manifest_sha256 = qwen_runtime_capability(
                request.get("capability_key"), require_enabled=require_enabled
            )
        else:
            raise QwenCapabilityError("remote release-gate marker is malformed")
    except QwenCapabilityError as exc:
        raise QwenRuntimeError("remote runtime capability is unavailable") from exc
    if (
        capability.runtime_adapter != "remote-docker"
        or socket.gethostname() != capability.hostname
    ):
        raise QwenRuntimeError("remote runtime host/capability identity changed")
    try:
        validate_qwen_capability_manifest_identity(
            key=capability.key,
            manifest_sha256=request.get("capability_manifest_sha256"),
            current_manifest_sha256=manifest_sha256,
            allow_retired_manifest=not require_enabled,
        )
    except QwenCapabilityError as exc:
        raise QwenRuntimeError(
            "remote runtime capability manifest changed"
        ) from exc
    if request.get("worker_receipt_mode") == "per-runtime" and (
        capability.key not in COMPACT_REMOTE_DOCKER_CAPABILITY_KEYS
        and capability.key != RTX5000_RELEASE_CANDIDATE_KEY
        or capability.max_num_seqs != 8
    ):
        raise QwenRuntimeError("remote multi-instance capability is not released")
    return capability, manifest_sha256


def _release_paths(request: dict[str, Any], capability) -> tuple[Path, Path, Path]:
    source_sha256 = request.get("source_manifest_sha256")
    model_sha256 = request.get("model_sha256s_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in source_sha256)
        or not isinstance(model_sha256, str)
        or len(model_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in model_sha256)
    ):
        raise QwenRuntimeError("remote release identity is malformed")
    cache = request.get("artifact_cache")
    if cache is None:
        package_root = RELEASE_ROOT / source_sha256
        model_dir = MODEL_CACHE_ROOT / model_sha256
    else:
        cache = _validated_artifact_cache_request(cache)
        if (
            cache["source"]["digest_sha256"] != source_sha256
            or cache["model"]["digest_sha256"] != model_sha256
            or cache["image"]["digest_sha256"]
            != str(capability.image_id).removeprefix("sha256:")
        ):
            raise QwenRuntimeError("remote cache binding differs from its capability")
        package_root = Path(cache["source"]["worker_path"])
        model_dir = Path(cache["model"]["worker_path"])
    preflight = RUNTIME_ROOT / "preflights" / (
        f"{capability.key}-{source_sha256}.json"
    )
    if request.get("package_root") != str(package_root) or request.get(
        "model_dir"
    ) != str(model_dir):
        raise QwenRuntimeError("remote release path is not content-addressed")
    return package_root, model_dir, preflight


def _verify_cache_filesystem(path: Path, binding: dict[str, Any], *, directory: bool) -> None:
    try:
        relative = PurePosixPath(str(path)).relative_to(
            PurePosixPath(str(FLEET_WORKER_CACHE_ROOT))
        )
    except ValueError as exc:
        raise QwenRuntimeError("remote cache entry escaped its Fleet root") from exc
    if not relative.parts or ".." in relative.parts:
        raise QwenRuntimeError("remote cache entry escaped its Fleet root")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    def mount_id(fd: int) -> str:
        with open(f"/proc/self/fdinfo/{fd}", encoding="ascii") as info:
            values = [
                line.split(":", 1)[1].strip()
                for line in info
                if line.startswith("mnt_id:")
            ]
        if len(values) != 1 or not values[0].isdecimal():
            raise QwenRuntimeError("remote cache mount identity is unavailable")
        return values[0]

    descriptor = os.open("/", flags)
    current = PurePosixPath("/")
    try:
        for part in PurePosixPath(str(FLEET_WORKER_CACHE_ROOT)).parts[1:]:
            current = current / part
            child = os.open(part, flags, dir_fd=descriptor)
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                raise QwenRuntimeError("remote cache ancestor is not a directory")
            if current.is_relative_to(PurePosixPath("/home/aday")) and (
                metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o022
                or os.path.ismount(str(current))
            ):
                raise QwenRuntimeError("remote cache ancestor identity changed")
            os.close(descriptor)
            descriptor = child
        root_meta = os.fstat(descriptor)
        if str(root_meta.st_dev) != binding["filesystem_id"]:
            raise QwenRuntimeError("remote cache filesystem changed")
        root_mount_id = mount_id(descriptor)
        for index, part in enumerate(relative.parts):
            final = index + 1 == len(relative.parts)
            open_flags = (
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
                if final and not directory
                else flags
            )
            child = os.open(part, open_flags, dir_fd=descriptor)
            metadata = os.fstat(child)
            if (
                metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o022
                or str(metadata.st_dev) != binding["filesystem_id"]
                or mount_id(child) != root_mount_id
                or os.path.ismount(str(FLEET_WORKER_CACHE_ROOT / Path(*relative.parts[: index + 1])))
            ):
                raise QwenRuntimeError("remote cache entry identity changed")
            if final:
                if directory and not stat.S_ISDIR(metadata.st_mode):
                    raise QwenRuntimeError("remote cache entry type changed")
                if not directory and not stat.S_ISREG(metadata.st_mode):
                    raise QwenRuntimeError("remote cache entry type changed")
                if not directory and metadata.st_nlink != 1:
                    raise QwenRuntimeError("remote cache entry link count changed")
            elif not stat.S_ISDIR(metadata.st_mode):
                raise QwenRuntimeError("remote cache ancestor is not a directory")
            os.close(descriptor)
            descriptor = child
        marker = {
            "schema_version": 1,
            "kind": binding["kind"],
            "digest_sha256": binding["digest_sha256"],
            "cache_root": str(FLEET_WORKER_CACHE_ROOT),
        }
        expected_marker = (
            json.dumps(marker, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        try:
            observed_marker = os.getxattr(
                descriptor, "user.fleet_compute_cache"
            )
        except OSError as exc:
            raise QwenRuntimeError("remote cache ownership proof is absent") from exc
        if observed_marker != expected_marker:
            raise QwenRuntimeError("remote cache ownership proof changed")
    except OSError as exc:
        raise QwenRuntimeError("remote cache no-follow proof failed") from exc
    finally:
        os.close(descriptor)


def _cached_image_identity(request: dict[str, Any], capability) -> tuple[str, int]:
    """Require the exact image installed by the Fleet cache backend.

    The backend owns cold archive transfer/load and leaves only a small receipt
    in Fleet's cache. This worker never removes images from Docker's global store.
    """

    cache = _validated_artifact_cache_request(request.get("artifact_cache"))
    binding = cache["image"]
    receipt_path = Path(binding["worker_path"])
    _verify_cache_filesystem(receipt_path, binding, directory=False)
    descriptor = os.open(
        receipt_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_size != binding["size_bytes"]
            or not 0 < before.st_size <= 65_536
        ):
            raise QwenRuntimeError("remote OCI cache receipt size changed")
        payload = read_bounded_fd(descriptor, 65_536)
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise QwenRuntimeError("remote OCI cache receipt changed while read")
        receipt = json.loads(payload.decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("remote OCI cache receipt is malformed") from exc
    finally:
        os.close(descriptor)
    _verify_cache_filesystem(receipt_path, binding, directory=False)
    expected_image_id = str(capability.image_id)
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {
            "schema_version",
            "image_id",
            "image_size_bytes",
            "archive_payload_sha256",
        }
        or receipt.get("schema_version") != 1
        or receipt.get("image_id") != expected_image_id
        or not isinstance(receipt.get("archive_payload_sha256"), str)
        or _SHA256_RE.fullmatch(receipt["archive_payload_sha256"]) is None
        or receipt.get("archive_payload_sha256") != binding["payload_sha256"]
    ):
        raise QwenRuntimeError("remote OCI cache receipt identity changed")
    image_id = local_image_id(expected_image_id)
    image_size = local_image_size(image_id)
    if image_id != expected_image_id or receipt.get("image_size_bytes") != image_size:
        raise QwenRuntimeError("remote cached image release changed")
    _image_config(image_id)
    return image_id, image_size


def _artifact_payload(identity: ArtifactIdentity) -> dict[str, Any]:
    return {
        "model_dir": str(identity.model_dir),
        "manifest_sha256": identity.manifest_sha256,
        "sha256s_sha256": identity.sha256s_sha256,
        "files": list(identity.files),
        "total_bytes": identity.total_bytes,
        "root_device": identity.root_device,
        "root_inode": identity.root_inode,
        "file_stats": [list(item) for item in identity.file_stats],
    }


def _artifact_from_payload(value: Any) -> ArtifactIdentity:
    if not isinstance(value, dict):
        raise QwenRuntimeError("remote model preflight receipt is malformed")
    try:
        return ArtifactIdentity(
            model_dir=Path(value["model_dir"]),
            manifest_sha256=str(value["manifest_sha256"]),
            sha256s_sha256=str(value["sha256s_sha256"]),
            files=tuple(str(item) for item in value["files"]),
            total_bytes=int(value["total_bytes"]),
            root_device=int(value["root_device"]),
            root_inode=int(value["root_inode"]),
            file_stats=tuple(tuple(item) for item in value["file_stats"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QwenRuntimeError("remote model preflight receipt changed") from exc


def _preflight(request: dict[str, Any]) -> dict[str, Any]:
    capability, manifest_sha256 = _capability(request)
    package_root, model_dir, preflight_path = _release_paths(request, capability)
    cache = request.get("artifact_cache")
    if cache is not None:
        cache = _validated_artifact_cache_request(cache)
        _verify_cache_filesystem(
            package_root,
            cache["source"],
            directory=True,
        )
        _verify_cache_filesystem(
            model_dir,
            cache["model"],
            directory=True,
        )
    source = _source_identity(package_root, RUNTIME_ROOT / "preflight")
    if source.manifest_sha256 != request["source_manifest_sha256"]:
        raise QwenRuntimeError("remote source release changed")
    artifact = load_artifact_identity(model_dir, verify_payload=True)
    if (
        artifact.manifest_sha256 != capability.model_manifest_sha256
        or artifact.manifest_sha256 != request.get("model_manifest_sha256")
        or artifact.sha256s_sha256 != request["model_sha256s_sha256"]
    ):
        raise QwenRuntimeError("remote model release changed")
    if cache is not None and artifact.total_bytes != cache["model"]["size_bytes"]:
        raise QwenRuntimeError("remote model cache size receipt changed")
    if cache is None:
        image_id = local_image_id(str(capability.image_id))
        image_size = local_image_size(image_id)
        receipt_schema = 1
    else:
        image_id, image_size = _cached_image_identity(request, capability)
        receipt_schema = 2
    if image_id != capability.image_id:
        raise QwenRuntimeError("remote image release changed")
    receipt = {
        "schema_version": receipt_schema,
        "capability_key": capability.key,
        "capability_manifest_sha256": manifest_sha256,
        "source_manifest_sha256": source.manifest_sha256,
        "model_sha256s_sha256": artifact.sha256s_sha256,
        "model_manifest_sha256": artifact.manifest_sha256,
        "artifact": _artifact_payload(artifact),
        "image_id": image_id,
        "image_size_bytes": image_size,
        "verified_at": time.time(),
        **({} if cache is None else {"artifact_cache": cache}),
    }
    _private_json_write(preflight_path, receipt)
    return {
        "state": "preflight_ready",
        "capability_key": capability.key,
        "source_manifest_sha256": source.manifest_sha256,
        "model_manifest_sha256": artifact.manifest_sha256,
        "model_sha256s_sha256": artifact.sha256s_sha256,
        "image_id": image_id,
        "image_size_bytes": image_size,
    }


def _load_preflight(request: dict[str, Any], capability) -> tuple[Path, Path, dict[str, Any]]:
    package_root, model_dir, preflight_path = _release_paths(request, capability)
    receipt = _private_json_read(preflight_path)
    if receipt is None:
        receipt = _private_json_read(
            RUNTIME_ROOT / f"preflight-{capability.key}.json"
        )
    cache = request.get("artifact_cache")
    expected_schema = 1 if cache is None else 2
    if cache is not None:
        cache = _validated_artifact_cache_request(cache)
    if (
        receipt is None
        or receipt.get("schema_version") != expected_schema
        or receipt.get("capability_key") != capability.key
        or receipt.get("capability_manifest_sha256")
        != request.get("capability_manifest_sha256")
        or receipt.get("source_manifest_sha256")
        != request.get("source_manifest_sha256")
        or receipt.get("model_sha256s_sha256")
        != request.get("model_sha256s_sha256")
        or receipt.get("model_manifest_sha256")
        != request.get("model_manifest_sha256")
        or receipt.get("image_id") != capability.image_id
        or (
            cache is not None
            and receipt.get("artifact_cache") != cache
        )
    ):
        raise QwenRuntimeError("remote runtime preflight is absent or stale")
    return package_root, model_dir, receipt


def _state_for_request(request: dict[str, Any], capability) -> dict[str, Any] | None:
    package_root, model_dir, _preflight_path = _release_paths(request, capability)
    run_dir, state_path = _runtime_binding(request)
    state = current_runtime_state(state_path)
    if state is None:
        return None
    if any(
        state.get(key) != expected
        for key, expected in (
            ("runtime_capability_key", capability.key),
            (
                "runtime_capability_manifest_sha256",
                request.get("capability_manifest_sha256"),
            ),
            ("source_manifest_sha256", request.get("source_manifest_sha256")),
            ("model_sha256s_sha256", request.get("model_sha256s_sha256")),
            ("model_manifest_sha256", request.get("model_manifest_sha256")),
            ("run_dir", str(run_dir)),
            ("physical_gpu", request.get("physical_gpu")),
            ("gpu_uuid", request.get("gpu_uuid")),
            ("claim_id", request.get("claim_id")),
            ("owner", request.get("owner")),
            ("container_name", request.get("container_name")),
            ("local_port", request.get("port")),
            (
                "source_dir",
                str(
                    run_dir
                    / f"local-source-{request.get('source_manifest_sha256')}"
                ),
            ),
            ("model_dir", str(model_dir)),
        )
    ):
        raise QwenRuntimeError("remote saved runtime differs from its request")
    if package_root.name != request.get("source_manifest_sha256"):
        raise QwenRuntimeError("remote source path identity changed")
    return state


def _runtime_identity_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        field: state[field]
        for field in ("run_dir", "physical_gpu", "gpu_uuid", "claim_id", "owner")
    }


def _start(request: dict[str, Any]) -> dict[str, Any]:
    capability, _manifest_sha256 = _capability(request)
    package_root, model_dir, preflight = _load_preflight(request, capability)
    run_dir, state_path = _runtime_binding(request)
    _assert_worker_gpu_slot_available(request, state_path)
    lease = request.get("lease")
    deploy_environment = request.get("deploy_environment")
    if not isinstance(lease, dict) or not isinstance(deploy_environment, dict):
        raise QwenRuntimeError("remote start request lacks its exact lease/plan")
    if any(
        lease.get(field) != expected
        for field, expected in (
            ("run_dir", str(run_dir)),
            ("physical_gpu", request.get("physical_gpu")),
        )
    ):
        raise QwenRuntimeError("remote start lease differs from its runtime binding")
    artifact = _artifact_from_payload(preflight.get("artifact"))
    if artifact.model_dir != model_dir:
        raise QwenRuntimeError("remote model preflight path changed")
    state = start_local_runtime(
        lease,
        deploy_environment,
        package_root=package_root,
        model_dir=model_dir,
        container_name=str(request.get("container_name") or ""),
        image=str(capability.image_id),
        port=int(request.get("port") or 0),
        artifact_identity=artifact,
        image_identity=str(preflight["image_id"]),
        image_size_bytes=int(preflight["image_size_bytes"]),
        state_path=state_path,
        coordinator_verify_func=False,
        final_heartbeat_func=lambda *_args, **_kwargs: None,
        heartbeat_promoter=lambda: int(
            local_container_pid(state_path=state_path) or 0
        ),
    )
    return {
        "state": "ready",
        "container_id": state["container_id"],
        "container_pid": state["container_pid"],
        **_runtime_identity_payload(state),
    }


def _status(request: dict[str, Any]) -> dict[str, Any]:
    capability, _manifest_sha256 = _capability(request, require_enabled=False)
    _run_dir, state_path = _runtime_binding(request)
    state = _state_for_request(request, capability)
    if state is None:
        return {"state": "gone", "container_pid": None, "container_id": None}
    if state.get("runtime_capability_key") != capability.key:
        raise QwenRuntimeError("remote saved runtime belongs to another capability")
    liveness, container_pid, resolved = _resolve_container(
        state,
        adopt=False,
        state_path=state_path,
    )
    return {
        "state": liveness,
        "phase": resolved.get("phase"),
        "container_pid": container_pid,
        "container_id": resolved.get("container_id"),
        **_runtime_identity_payload(resolved),
        "scratch_cleaned": resolved.get("scratch_cleaned"),
    }


def _claim_container_candidates(
    request: dict[str, Any], capability
) -> tuple[str, ...]:
    """Find only containers bearing this exact Aeon claim/capability pair."""

    claim_id = _validate_token(request.get("claim_id"), _CLAIM_RE, "runtime claim")
    try:
        result = subprocess.run(
            [
                *_docker_command("ps"),
                "-aq",
                "--no-trunc",
                "--filter",
                "label=com.bc_aeon.component=qwen38-vllm",
                "--filter",
                f"label=com.bc_aeon.claim={claim_id}",
                "--filter",
                f"label=com.bc_aeon.runtime-capability={capability.key}",
            ],
            env=_docker_cli_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        raise QwenRuntimeError("remote exact-claim Docker proof is unavailable") from exc
    if (
        result.returncode != 0
        or result.stderr != ""
        or len(result.stdout or "") > 8192
    ):
        raise QwenRuntimeError("remote exact-claim Docker proof is unavailable")
    candidates = tuple(
        _validate_token(value, _CONTAINER_ID_RE, "container candidate")
        for line in result.stdout.splitlines()
        if (value := line.strip())
    )
    if len(candidates) != len(set(candidates)):
        raise QwenRuntimeError("remote exact-claim Docker proof is ambiguous")
    return candidates


def _recover_precontainer(request: dict[str, Any]) -> dict[str, Any]:
    """Atomically prove and clear one exact, never-launched worker lifecycle."""

    capability, _manifest_sha256 = _capability(request, require_enabled=False)
    _run_dir, state_path = _runtime_binding(request)
    before = _state_for_request(request, capability)
    if before is None:
        if _claim_container_candidates(request, capability):
            raise QwenRuntimeError(
                "remote missing receipt retains an exact-claim container"
            )
        return {
            "state": "recovered",
            "controller_protocol": 1,
            "process_absent": True,
            "worker_receipt_absent": True,
            **{
                field: request[field]
                for field in (
                    "run_dir",
                    "physical_gpu",
                    "gpu_uuid",
                    "claim_id",
                    "owner",
                )
            },
        }
    if (
        before.get("phase") not in {"preparing", "releasing"}
        or before.get("container_pid") is not None
        or type(before.get("scratch_cleaned")) is not bool
        or (
            before.get("phase") == "preparing"
            and before.get("scratch_cleaned") is not False
        )
    ):
        raise QwenRuntimeError("remote worker is not an unlaunched lifecycle")
    # Docker may atomically write the cidfile and create the exact labelled
    # container even when its CLI call returns a failure.  The ordinary
    # liveness proof then adopts that immutable ID into this receipt.  A
    # verified exited container is still safe for the same receipt-owned stop
    # transaction; only an active or ambiguous process remains protected.
    if qwen_runtime_liveness(state_path=state_path) not in {"gone", "exited"}:
        raise QwenRuntimeError("remote pre-container process absence is unproven")
    if before.get("scratch_cleaned") is True:
        releasing = before
    else:
        if not stop_qwen_runtime(state_path=state_path, allow_lost_lease=True):
            raise QwenRuntimeError("remote pre-container cleanup is incomplete")
        releasing = current_runtime_state(state_path)
    if (
        releasing is None
        or releasing.get("phase") != "releasing"
        or releasing.get("container_pid") is not None
        or releasing.get("scratch_cleaned") is not True
        or qwen_runtime_liveness(state_path=state_path) != "gone"
    ):
        raise QwenRuntimeError("remote pre-container cleanup proof changed")
    clear_runtime_state(state_path)
    if current_runtime_state(state_path) is not None:
        raise QwenRuntimeError("remote pre-container receipt was not cleared")
    return {
        "state": "recovered",
        "controller_protocol": 1,
        "process_absent": True,
        "worker_receipt_absent": False,
        **_runtime_identity_payload(before),
    }


def _recover_uncommitted(request: dict[str, Any]) -> dict[str, Any]:
    """Atomically stop and clear one exact container Fleet never committed."""

    capability, _manifest_sha256 = _capability(request, require_enabled=False)
    _run_dir, state_path = _runtime_binding(request)
    if request.get("worker_receipt_mode") != "per-runtime":
        raise QwenRuntimeError("uncommitted recovery requires a Fleet runtime")
    expected_container_id = _validate_token(
        request.get("expected_container_id"),
        _CONTAINER_ID_RE,
        "expected uncommitted container",
    )
    expected_container_pid = request.get("expected_container_pid")
    if (
        type(expected_container_pid) is not int
        or expected_container_pid <= 1
    ):
        raise QwenRuntimeError("expected uncommitted container PID is malformed")

    before = _state_for_request(request, capability)
    if before is None:
        if _claim_container_candidates(request, capability):
            raise QwenRuntimeError(
                "remote cleared receipt retains an exact-claim container"
            )
        return {
            "state": "recovered",
            "controller_protocol": 2,
            "process_absent": True,
            "worker_receipt_absent": True,
            "container_id": expected_container_id,
            "container_pid": expected_container_pid,
            **{
                field: request[field]
                for field in (
                    "run_dir",
                    "physical_gpu",
                    "gpu_uuid",
                    "claim_id",
                    "owner",
                )
            },
        }

    if (
        before.get("phase") not in {"launching", "ready", "releasing"}
        or before.get("container_id") != expected_container_id
        or type(before.get("scratch_cleaned")) is not bool
    ):
        raise QwenRuntimeError("remote uncommitted runtime identity changed")
    clean_releasing = (
        before.get("phase") == "releasing"
        and before.get("scratch_cleaned") is True
    )
    exited_pid_cleared_candidate = (
        before.get("phase") == "launching"
        and before.get("container_pid") is None
        and before.get("scratch_cleaned") is False
    )
    dirty_releasing_pid_cleared_candidate = (
        before.get("phase") == "releasing"
        and before.get("container_pid") is None
        and before.get("scratch_cleaned") is False
    )
    if (
        not clean_releasing
        and before.get("container_pid") != expected_container_pid
        and not exited_pid_cleared_candidate
        and not dirty_releasing_pid_cleared_candidate
    ):
        raise QwenRuntimeError("remote uncommitted runtime PID changed")

    liveness = qwen_runtime_liveness(state_path=state_path)
    if exited_pid_cleared_candidate and liveness != "exited":
        raise QwenRuntimeError("remote uncommitted cleared PID is not exited")
    if (
        dirty_releasing_pid_cleared_candidate
        and liveness not in {"exited", "gone"}
    ):
        raise QwenRuntimeError(
            "remote uncommitted dirty release absence is unproven"
        )
    if clean_releasing:
        if liveness != "gone" or _claim_container_candidates(request, capability):
            raise QwenRuntimeError("remote uncommitted cleanup proof changed")
    else:
        if liveness not in {"active", "exited", "gone"}:
            raise QwenRuntimeError("remote uncommitted runtime is ambiguous")
        candidates = _claim_container_candidates(request, capability)
        if (
            liveness in {"active", "exited"}
            and candidates != (expected_container_id,)
        ) or (liveness == "gone" and candidates):
            raise QwenRuntimeError("remote uncommitted container proof changed")
        if not stop_qwen_runtime(state_path=state_path, allow_lost_lease=True):
            raise QwenRuntimeError("remote uncommitted cleanup is incomplete")

    releasing = _state_for_request(request, capability)
    if (
        releasing is None
        or releasing.get("phase") != "releasing"
        or releasing.get("container_id") != expected_container_id
        or releasing.get("container_pid") is not None
        or releasing.get("scratch_cleaned") is not True
        or qwen_runtime_liveness(state_path=state_path) != "gone"
        or _claim_container_candidates(request, capability)
    ):
        raise QwenRuntimeError("remote uncommitted cleanup proof changed")
    clear_runtime_state(state_path)
    if (
        current_runtime_state(state_path) is not None
        or _claim_container_candidates(request, capability)
    ):
        raise QwenRuntimeError("remote uncommitted receipt was not cleared")
    return {
        "state": "recovered",
        "controller_protocol": 2,
        "process_absent": True,
        "worker_receipt_absent": False,
        "container_id": expected_container_id,
        "container_pid": expected_container_pid,
        **_runtime_identity_payload(before),
    }


def _reuse(request: dict[str, Any]) -> dict[str, Any]:
    capability, _manifest_sha256 = _capability(request)
    package_root, _model_dir, _preflight = _load_preflight(request, capability)
    run_dir, state_path = _runtime_binding(request)
    lease = request.get("lease")
    config = request.get("config")
    if not isinstance(lease, dict) or not isinstance(config, dict):
        raise QwenRuntimeError("remote reuse request is malformed")
    if any(
        lease.get(field) != expected
        for field, expected in (
            ("run_dir", str(run_dir)),
            ("physical_gpu", request.get("physical_gpu")),
        )
    ):
        raise QwenRuntimeError("remote reuse lease differs from its runtime binding")
    pid = reuse_qwen_runtime(
        config=config,
        package_root=package_root,
        state_path=state_path,
        lease_override=lease,
        coordinator_verify_func=False,
    )
    if pid is None:
        state = _state_for_request(request, capability)
        return {
            "state": "gone",
            "container_pid": None,
            **({} if state is None else _runtime_identity_payload(state)),
        }
    state = _state_for_request(request, capability)
    if state is None:
        raise QwenRuntimeError("remote reuse journal disappeared")
    return {
        "state": "active",
        "container_pid": pid,
        "container_id": state.get("container_id"),
        **_runtime_identity_payload(state),
    }


def _stop(request: dict[str, Any]) -> dict[str, Any]:
    capability, _manifest_sha256 = _capability(request, require_enabled=False)
    _run_dir, state_path = _runtime_binding(request)
    before = _state_for_request(request, capability)
    stopped = stop_qwen_runtime(state_path=state_path, allow_lost_lease=True)
    state = current_runtime_state(state_path)
    return {
        "state": "stopped" if stopped else "ambiguous",
        "scratch_cleaned": None if state is None else state.get("scratch_cleaned"),
        **({} if before is None else _runtime_identity_payload(before)),
    }


def _clear(request: dict[str, Any]) -> dict[str, Any]:
    capability, _manifest_sha256 = _capability(request, require_enabled=False)
    _run_dir, state_path = _runtime_binding(request)
    state = _state_for_request(request, capability)
    if state is None:
        return {"state": "cleared", "receipt_absent": True}
    if (
        state.get("phase") != "releasing"
        or state.get("scratch_cleaned") is not True
        or qwen_runtime_liveness(state_path=state_path) != "gone"
    ):
        raise QwenRuntimeError("remote runtime is not safe to clear")
    clear_runtime_state(state_path)
    return {
        "state": "cleared",
        "receipt_absent": False,
        **_runtime_identity_payload(state),
    }


_ACTIONS = {
    "preflight": _preflight,
    "start": _start,
    "status": _status,
    "reuse": _reuse,
    "stop": _stop,
    "clear": _clear,
    "recover-precontainer": _recover_precontainer,
    "recover-uncommitted": _recover_uncommitted,
}

_CONTROLLER_ACTIONS = frozenset(
    {
        "start",
        "reuse",
        "stop",
        "clear",
        "recover-precontainer",
        "recover-uncommitted",
    }
)


def _dispatch(action: str, request: dict[str, Any]) -> dict[str, Any]:
    handler = _ACTIONS[action]
    if action not in _CONTROLLER_ACTIONS:
        return handler(request)
    with _hold_controller_lock(
        request, action=action, create=action == "start"
    ):
        return handler(request)


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in _ACTIONS:
        print(json.dumps({"ok": False, "error": "invalid_action"}))
        return 64
    try:
        result = _dispatch(sys.argv[1], _request())
    except (QwenRuntimeError, QwenCapabilityError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": type(exc).__name__, "detail": str(exc)},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
