"""Coordinator-owned placement and transport for release-compatible Qwen workers.

Local Docker remains implemented by :mod:`aeon.core.qwen_runtime`.  This module
adds the .177-side half of the fixed SSH remote-Docker adapter: immutable source
staging, worker preflight/start/reuse/stop calls, exact-PID heartbeats, and a
receipted loopback tunnel.  It is invoked only by a foreground Aeon session.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .gpu_queue import QWEN_LEASE_FILE
from .fleet_hosts import network_address
from .qwen_capabilities import (
    COMPACT_REMOTE_DOCKER_CAPABILITY_KEYS,
    RTX5000_180_RELEASE_CAPABILITY_KEY,
    RTX5000_RELEASE_CANDIDATE_KEY,
    QwenCapabilityError,
    QwenRuntimeCapability,
    qwen_runtime_capability,
    validate_qwen_capability_manifest_identity,
)
from .qwen_runtime import (
    RUNTIME_ROOT,
    RUNTIME_STATE_FILE,
    SOURCE_FILES,
    SOURCE_MANIFEST_FILE,
    QwenRuntimeError,
    SourceIdentity,
    _private_json_read,
    _private_json_write,
    _source_identity,
    _validate_run_dir as _validate_qwen_run_dir,
)


REMOTE_STATE_FILE = RUNTIME_ROOT / "remote-runtime.json"
REMOTE_STATE_ROOT = RUNTIME_ROOT / "remote-runtimes"
WORKER_STATE_ROOT = RUNTIME_ROOT / "worker-runtimes"
REMOTE_RELEASE_ROOT = Path("/home/aday/.aeon/runtime/qwen38/releases")
# Recovery-only location used by source closures released before Fleet's shared
# artifact cache existed. New Fleet runtimes receive exact cache bindings.
REMOTE_MODEL_ROOT = Path("/home/aday/.aeon/runtime/qwen38/models")
FLEET_WORKER_CACHE_ROOT = Path(
    "/home/aday/.local/state/fleet-compute/cache/aeon-qwen38"
)
QWEN_MODEL_CACHE_ARTIFACT_ID = "aeon-qwen38-model"
QWEN_IMAGE_CACHE_ARTIFACT_ID = "aeon-qwen38-image"
QWEN_SOURCE_CACHE_ARTIFACT_ID = "aeon-qwen38-source"
QWEN_STANDARD_IMAGE_CONFIG_SHA256 = (
    "d57400972ab0ae46baac64d4bfcc49cb136c07d8b0c50a76c7e2d81bd8a9fe47"
)
QWEN_SOURCE_CACHE_MAX_BYTES = 50_000_000
QWEN_SOURCE_CACHE_MAX_INODES = 1_000
QWEN_SOURCE_CACHE_TRANSFER_MAX_BYTES = 50_000_000
QWEN_SOURCE_CACHE_COLD_PEAK_BYTES = 50_000_000
QWEN_MODEL_CACHE_MAX_BYTES = 20_600_000_000
QWEN_MODEL_CACHE_MAX_INODES = 100_000
QWEN_MODEL_CACHE_TRANSFER_MAX_BYTES = 20_600_000_000
QWEN_MODEL_CACHE_COLD_PEAK_BYTES = 20_600_000_000
QWEN_IMAGE_CACHE_RECEIPT_MAX_BYTES = 65_536
QWEN_IMAGE_CACHE_MAX_INODES = 1
QWEN_IMAGE_ARCHIVE_MAX_BYTES = 16_000_000_000
QWEN_IMAGE_CACHE_COLD_PEAK_BYTES = 26_318_824_199
QWEN_ARTIFACT_CACHE_SCHEMA_VERSION = 1
REMOTE_PYTHON = Path(
    "/home/aday/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12"
)
REMOTE_WRAPPER = Path("/home/aday/bin/fleet-low-priority")
LOCAL_PORT = 8033
FLEET_REMOTE_PORT_BASE = 18035
FLEET_LOCAL_PORT_BASE = 18035
FLEET_LOCAL_HOST_PORT_OFFSETS = {
    "192.168.0.180": 0,
    "192.168.0.178": 2,
    "192.168.0.179": 4,
}
REMOTE_STARTUP_TIMEOUT_SECONDS = 2100
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_CONTAINER_ID_RE = re.compile(r"^[a-f0-9]{64}$")
_CLAIM_RE = re.compile(r"^gc-[A-Za-z0-9._:-]{1,196}$")
_OWNER_RE = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]{8,120}$")
_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REMOTE_PHASES = frozenset({"starting", "ready", "releasing"})
_FLEET_RUNTIME_ID_RE = re.compile(r"^fr-[0-9a-f]{32}$")
_FLEET_RUN_ROOT = Path("/home/aday/.local/state/fleet-compute/runs")
_FLEET_REMOTE_GPUS = frozenset({0, 1})
_REMOTE_CACHE_BOOTSTRAP = r'''
import hashlib, json, os, pathlib, re, runpy, stat, sys
worker_raw, action, source_raw, digest, fsid, manifest_rel = sys.argv[1:7]
root = pathlib.Path("/home/aday/.local/state/fleet-compute/cache/aeon-qwen38")
source, worker = pathlib.Path(source_raw), pathlib.Path(worker_raw)
assert source == root / "sha256" / digest[:2] / digest
assert worker == source / "aeon/scripts/qwen_remote_worker.py"
assert re.fullmatch(r"[a-f0-9]{64}", digest)
relative = source.relative_to(root)
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
descriptor = os.open("/", flags)
current = pathlib.PurePosixPath("/")
try:
    for part in pathlib.PurePosixPath(str(root)).parts[1:]:
        current = current / part
        child = os.open(part, flags, dir_fd=descriptor)
        metadata = os.fstat(child)
        assert stat.S_ISDIR(metadata.st_mode)
        if current.is_relative_to(pathlib.PurePosixPath("/home/aday")):
            assert metadata.st_uid == os.geteuid() and not metadata.st_mode & 0o022
            assert not os.path.ismount(str(current))
        os.close(descriptor); descriptor = child
    root_meta = os.fstat(descriptor)
    assert str(root_meta.st_dev) == fsid
    for index, part in enumerate(relative.parts):
        child = os.open(part, flags, dir_fd=descriptor)
        metadata = os.fstat(child)
        assert stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.geteuid()
        assert not metadata.st_mode & 0o022 and str(metadata.st_dev) == fsid
        assert not os.path.ismount(str(root / pathlib.Path(*relative.parts[:index+1])))
        os.close(descriptor); descriptor = child
    marker = {"schema_version":1,"kind":"manifested_tree",
              "digest_sha256":digest,"cache_root":str(root)}
    expected_marker = (json.dumps(marker,sort_keys=True,separators=(",",":"),
                                  allow_nan=False)+"\n").encode("utf-8")
    assert os.getxattr(descriptor,"user.fleet_compute_cache") == expected_marker
finally:
    os.close(descriptor)
manifest = source / manifest_rel
manifest_meta = manifest.lstat()
assert stat.S_ISREG(manifest_meta.st_mode) and not stat.S_ISLNK(manifest_meta.st_mode)
assert manifest_meta.st_uid == os.geteuid() and not manifest_meta.st_mode & 0o022
assert manifest_meta.st_nlink == 1 and str(manifest_meta.st_dev) == fsid
assert not os.path.ismount(manifest)
assert 0 < manifest_meta.st_size <= 16*1024*1024
manifest_payload = manifest.read_bytes()
assert hashlib.sha256(manifest_payload).hexdigest() == digest
expected_files = {manifest_rel}; checks = []
for line in manifest_payload.decode("utf-8").splitlines():
    match = re.fullmatch(r"([a-f0-9]{64}) [ *](.+)",line)
    assert match is not None and not match.group(2).startswith("/")
    assert ".." not in pathlib.PurePosixPath(match.group(2)).parts
    assert match.group(2) not in expected_files
    expected_files.add(match.group(2)); checks.append((match.group(1),match.group(2)))
actual_files = set()
for item in source.rglob("*"):
    item_meta = item.lstat(); relative_item = item.relative_to(source).as_posix()
    assert item_meta.st_uid == os.geteuid() and not item_meta.st_mode & 0o022
    assert str(item_meta.st_dev) == fsid and not stat.S_ISLNK(item_meta.st_mode)
    if stat.S_ISDIR(item_meta.st_mode):
        assert not os.path.ismount(item)
    else:
        assert stat.S_ISREG(item_meta.st_mode) and item_meta.st_nlink == 1
        assert not os.path.ismount(item); actual_files.add(relative_item)
assert actual_files == expected_files
for wanted, relative_item in checks:
    item = source / relative_item
    hasher = hashlib.sha256()
    with item.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b""):
            hasher.update(chunk)
    assert hasher.hexdigest() == wanted
sys.path.insert(0,str(source)); sys.argv = [str(worker),action]
runpy.run_path(str(worker),run_name="__main__")
'''
# This one immutable staged release predates the PID-cleared exited-container
# recovery protocol.  Its controller-locked ``stop``/``clear`` actions already
# provide the exact-ID cleanup proof, so only this source may use that narrow
# compatibility path after a fresh exact status proof.
_PID_CLEARED_STOP_CLEAR_SOURCE_SHA256 = (
    "82106323334eb495613d51002f977c42d9457a4efea98c0b820d9ae5449c302c"
)


@dataclass(frozen=True)
class QwenRemoteArtifactCache:
    """Exact, serializable Fleet cache bindings for one remote Qwen launch."""

    source: dict[str, Any]
    model: dict[str, Any]
    image: dict[str, Any]

    def to_request(self) -> dict[str, Any]:
        return {
            "schema_version": QWEN_ARTIFACT_CACHE_SCHEMA_VERSION,
            "source": dict(self.source),
            "model": dict(self.model),
            "image": dict(self.image),
        }


def _binding_field(binding: Any, name: str) -> Any:
    if isinstance(binding, Mapping):
        return binding.get(name)
    return getattr(binding, name, None)


def _cache_entry_path(digest: str) -> Path:
    return FLEET_WORKER_CACHE_ROOT / "sha256" / digest[:2] / digest


def _normalize_cache_binding(
    binding: Any,
    *,
    artifact_id: str,
    kind: str,
    digest: str,
    maximum_bytes: int,
    maximum_inodes: int,
) -> dict[str, Any]:
    observed_artifact_id = _binding_field(binding, "artifact_id")
    observed_kind = _binding_field(binding, "kind")
    observed_digest = _binding_field(binding, "digest_sha256")
    raw_path = _binding_field(binding, "worker_path")
    size_bytes = _binding_field(binding, "size_bytes")
    inode_count = _binding_field(binding, "inode_count")
    filesystem_id = _binding_field(binding, "filesystem_id")
    payload_sha256 = _binding_field(binding, "payload_sha256")
    expected_path = _cache_entry_path(digest)
    try:
        path = Path(str(raw_path))
    except (TypeError, ValueError) as exc:
        raise QwenRuntimeError("Fleet Qwen cache path is malformed") from exc
    if (
        observed_artifact_id != artifact_id
        or observed_kind != kind
        or observed_digest != digest
        or path != expected_path
        or ".." in PurePosixPath(str(path)).parts
        or type(size_bytes) is not int
        or not 0 < size_bytes <= maximum_bytes
        or type(inode_count) is not int
        or not 0 < inode_count <= maximum_inodes
        or not isinstance(filesystem_id, str)
        or not filesystem_id.isdecimal()
        or len(filesystem_id) > 32
        or (
            kind == "oci_archive"
            and (
                not isinstance(payload_sha256, str)
                or _SHA256_RE.fullmatch(payload_sha256) is None
            )
        )
        or (
            kind != "oci_archive"
            and payload_sha256 is not None
            and payload_sha256 != digest
        )
    ):
        raise QwenRuntimeError("Fleet Qwen cache binding changed")
    result = {
        "artifact_id": artifact_id,
        "kind": kind,
        "worker_path": str(path),
        "digest_sha256": digest,
        "size_bytes": size_bytes,
        "inode_count": inode_count,
        "filesystem_id": filesystem_id,
    }
    if kind == "oci_archive":
        result["payload_sha256"] = payload_sha256
    return result


def qwen_remote_artifact_cache(
    capability: QwenRuntimeCapability,
    source: SourceIdentity,
    cached_artifacts: Mapping[str, Any],
) -> QwenRemoteArtifactCache:
    """Bind only the two exact cache entries authorized by Qwen's capability."""

    if not isinstance(cached_artifacts, Mapping) or set(cached_artifacts) != {
        QWEN_MODEL_CACHE_ARTIFACT_ID,
        QWEN_IMAGE_CACHE_ARTIFACT_ID,
        QWEN_SOURCE_CACHE_ARTIFACT_ID,
    }:
        raise QwenRuntimeError("Fleet did not acquire the exact Qwen cache entries")
    image_id = capability.image_id
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise QwenRuntimeError("Qwen capability has no exact image identity")
    source_binding = _normalize_cache_binding(
        cached_artifacts[QWEN_SOURCE_CACHE_ARTIFACT_ID],
        artifact_id=QWEN_SOURCE_CACHE_ARTIFACT_ID,
        kind="manifested_tree",
        digest=_source_sha256(source),
        maximum_bytes=QWEN_SOURCE_CACHE_MAX_BYTES,
        maximum_inodes=QWEN_SOURCE_CACHE_MAX_INODES,
    )
    model = _normalize_cache_binding(
        cached_artifacts[QWEN_MODEL_CACHE_ARTIFACT_ID],
        artifact_id=QWEN_MODEL_CACHE_ARTIFACT_ID,
        kind="manifested_tree",
        digest=capability.model_sha256s_sha256,
        maximum_bytes=QWEN_MODEL_CACHE_MAX_BYTES,
        maximum_inodes=QWEN_MODEL_CACHE_MAX_INODES,
    )
    image = _normalize_cache_binding(
        cached_artifacts[QWEN_IMAGE_CACHE_ARTIFACT_ID],
        artifact_id=QWEN_IMAGE_CACHE_ARTIFACT_ID,
        kind="oci_archive",
        digest=image_id.removeprefix("sha256:"),
        maximum_bytes=QWEN_IMAGE_CACHE_RECEIPT_MAX_BYTES,
        maximum_inodes=QWEN_IMAGE_CACHE_MAX_INODES,
    )
    if image["inode_count"] != 1:
        raise QwenRuntimeError("Fleet Qwen OCI cache binding is not one archive")
    return QwenRemoteArtifactCache(
        source=source_binding,
        model=model,
        image=image,
    )


def _validated_artifact_cache_request(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "source", "model", "image"}
        or value.get("schema_version") != QWEN_ARTIFACT_CACHE_SCHEMA_VERSION
    ):
        raise QwenRuntimeError("remote Qwen artifact-cache receipt is malformed")
    # The capability-specific digests are validated where the capability is in
    # hand. This validator still enforces the closed, serialized wire shape.
    result: dict[str, Any] = {
        "schema_version": QWEN_ARTIFACT_CACHE_SCHEMA_VERSION
    }
    for label, expected_kind, maximum_bytes, maximum_inodes in (
        (
            "source",
            "manifested_tree",
            QWEN_SOURCE_CACHE_MAX_BYTES,
            QWEN_SOURCE_CACHE_MAX_INODES,
        ),
        (
            "model",
            "manifested_tree",
            QWEN_MODEL_CACHE_MAX_BYTES,
            QWEN_MODEL_CACHE_MAX_INODES,
        ),
        (
            "image",
            "oci_archive",
            QWEN_IMAGE_CACHE_RECEIPT_MAX_BYTES,
            QWEN_IMAGE_CACHE_MAX_INODES,
        ),
    ):
        entry = value.get(label)
        expected_fields = {
            "artifact_id",
            "kind",
            "worker_path",
            "digest_sha256",
            "size_bytes",
            "inode_count",
            "filesystem_id",
        }
        if label == "image":
            expected_fields.add("payload_sha256")
        if not isinstance(entry, Mapping) or set(entry) != expected_fields:
            raise QwenRuntimeError("remote Qwen artifact-cache receipt is malformed")
        digest = entry.get("digest_sha256")
        path = entry.get("worker_path")
        size = entry.get("size_bytes")
        inodes = entry.get("inode_count")
        filesystem_id = entry.get("filesystem_id")
        payload_sha256 = entry.get("payload_sha256")
        if (
            entry.get("kind") != expected_kind
            or not isinstance(entry.get("artifact_id"), str)
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or not isinstance(path, str)
            or Path(path) != _cache_entry_path(digest)
            or ".." in PurePosixPath(path).parts
            or type(size) is not int
            or not 0 < size <= maximum_bytes
            or type(inodes) is not int
            or not 0 < inodes <= maximum_inodes
            or not isinstance(filesystem_id, str)
            or not filesystem_id.isdecimal()
            or len(filesystem_id) > 32
            or (
                label == "image"
                and (
                    not isinstance(payload_sha256, str)
                    or _SHA256_RE.fullmatch(payload_sha256) is None
                )
            )
        ):
            raise QwenRuntimeError("remote Qwen artifact-cache receipt is malformed")
        result[label] = dict(entry)
    if (
        result["source"]["artifact_id"] != QWEN_SOURCE_CACHE_ARTIFACT_ID
        or result["model"]["artifact_id"] != QWEN_MODEL_CACHE_ARTIFACT_ID
        or result["image"]["artifact_id"] != QWEN_IMAGE_CACHE_ARTIFACT_ID
    ):
        raise QwenRuntimeError("remote Qwen artifact-cache identity changed")
    return result


def _uses_pid_cleared_stop_clear_compatibility(
    state: Mapping[str, Any],
) -> bool:
    """Identify only the reviewed old compact-worker protocol closure."""

    return (
        state.get("runtime_capability_key")
        == RTX5000_180_RELEASE_CAPABILITY_KEY
        and state.get("source_manifest_sha256")
        == _PID_CLEARED_STOP_CLEAR_SOURCE_SHA256
    )


def _fleet_runtime_id(run_dir: str | Path) -> str | None:
    """Return the exact Fleet runtime ID, or None for a valid legacy run dir."""

    path = _validate_qwen_run_dir(run_dir)
    if path.parent != _FLEET_RUN_ROOT:
        return None
    if _FLEET_RUNTIME_ID_RE.fullmatch(path.name) is None:
        raise QwenRuntimeError("Fleet remote runtime identity is malformed")
    return path.name


def fleet_remote_runtime_resources(
    run_dir: str | Path,
    physical_gpu: int,
    *,
    host: str = "192.168.0.180",
) -> dict[str, Any]:
    """Derive all mutable-name/port/receipt bindings from one Fleet lease."""

    runtime_id = _fleet_runtime_id(run_dir)
    if runtime_id is None:
        raise QwenRuntimeError("remote multi-instance runtime requires a Fleet run directory")
    if type(physical_gpu) is not int or physical_gpu not in _FLEET_REMOTE_GPUS:
        raise QwenRuntimeError("remote multi-instance GPU slot is invalid")
    host_offset = FLEET_LOCAL_HOST_PORT_OFFSETS.get(host)
    if host_offset is None:
        raise QwenRuntimeError("remote multi-instance host is invalid")
    return {
        "runtime_id": runtime_id,
        "run_dir": str(_FLEET_RUN_ROOT / runtime_id),
        "orchestrator_state_path": REMOTE_STATE_ROOT / f"{runtime_id}.json",
        "worker_state_path": WORKER_STATE_ROOT / f"{runtime_id}.json",
        "container_name": f"aeon-qwen38-standard-{runtime_id}-gpu{physical_gpu}",
        "remote_port": FLEET_REMOTE_PORT_BASE + physical_gpu,
        "local_port": FLEET_LOCAL_PORT_BASE + host_offset + physical_gpu,
    }


def _source_sha256(source: SourceIdentity | str) -> str:
    value = source.manifest_sha256 if isinstance(source, SourceIdentity) else source
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise QwenRuntimeError("remote source manifest is malformed")
    return value


def _ssh_base(capability: QwenRuntimeCapability) -> list[str]:
    if capability.runtime_adapter != "remote-docker":
        raise QwenRuntimeError("capability is not a remote Docker release")
    return [
        "/usr/bin/ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ControlPersist=no",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=6",
        f"aday@{network_address(capability.host)}",
    ]


def _release_path(source: SourceIdentity | str) -> Path:
    return REMOTE_RELEASE_ROOT / _source_sha256(source)


def _request_base(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    source: SourceIdentity | str,
    artifact_cache: Mapping[str, Any] | QwenRemoteArtifactCache | None = None,
) -> dict[str, Any]:
    if _SHA256_RE.fullmatch(manifest_sha256) is None:
        raise QwenRuntimeError("capability manifest identity is malformed")
    cache = (
        None
        if artifact_cache is None
        else _validated_artifact_cache_request(
            artifact_cache.to_request()
            if isinstance(artifact_cache, QwenRemoteArtifactCache)
            else artifact_cache
        )
    )
    model_dir = (
        REMOTE_MODEL_ROOT / capability.model_sha256s_sha256
        if cache is None
        else Path(cache["model"]["worker_path"])
    )
    package_root = (
        _release_path(source)
        if cache is None
        else Path(cache["source"]["worker_path"])
    )
    request = {
        "capability_key": capability.key,
        "capability_manifest_sha256": manifest_sha256,
        "source_manifest_sha256": _source_sha256(source),
        "model_manifest_sha256": capability.model_manifest_sha256,
        "model_sha256s_sha256": capability.model_sha256s_sha256,
        "package_root": str(package_root),
        "model_dir": str(model_dir),
        "release_gate": capability.key == RTX5000_RELEASE_CANDIDATE_KEY,
    }
    if cache is not None:
        request["artifact_cache"] = cache
    return request


def stage_remote_source(
    capability: QwenRuntimeCapability,
    package_root: Path,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> SourceIdentity:
    """Copy only the immutable source closure to its content-addressed worker root."""

    source = _source_identity(package_root, RUNTIME_ROOT / "remote-preflight")
    destination = _release_path(source)
    make_command = [
        *_ssh_base(capability),
        "/usr/bin/env",
        "-i",
        "PATH=/usr/bin:/bin",
        "HOME=/home/aday",
        "LANG=C",
        "LC_ALL=C",
        "/usr/bin/mkdir",
        "-p",
        "-m",
        "0700",
        str(destination),
    ]
    make = None
    for attempt in range(3):
        make = command_runner(
            make_command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if make.returncode == 0:
            break
        if attempt < 2:
            time.sleep(2)
    assert make is not None
    if make.returncode != 0:
        raise QwenRuntimeError("remote release root could not be prepared")
    ssh_transport = " ".join(_ssh_base(capability)[:-1])
    transfer_command = [
        "/usr/bin/bash",
        str(REMOTE_WRAPPER),
        "/usr/bin/rsync",
        "-aR",
        "--checksum",
        "--protect-args",
        "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
        "-e",
        ssh_transport,
        "--",
        *SOURCE_FILES,
        f"aday@{network_address(capability.host)}:{destination}/",
    ]
    transfer = None
    for attempt in range(3):
        transfer = command_runner(
            transfer_command,
            cwd=str(package_root),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if transfer.returncode == 0:
            break
        if attempt < 2:
            time.sleep(2)
    assert transfer is not None
    if transfer.returncode != 0:
        raise QwenRuntimeError("remote immutable source staging failed")
    return source


def _remote_command(
    capability: QwenRuntimeCapability,
    source: SourceIdentity | str,
    action: str,
    artifact_cache: Mapping[str, Any] | QwenRemoteArtifactCache | None = None,
) -> list[str]:
    if action not in {
        "preflight",
        "start",
        "status",
        "reuse",
        "stop",
        "clear",
        "recover-precontainer",
        "recover-uncommitted",
    }:
        raise QwenRuntimeError("invalid remote runtime action")
    cache = (
        None
        if artifact_cache is None
        else _validated_artifact_cache_request(
            artifact_cache.to_request()
            if isinstance(artifact_cache, QwenRemoteArtifactCache)
            else artifact_cache
        )
    )
    release_root = (
        _release_path(source)
        if cache is None
        else Path(cache["source"]["worker_path"])
    )
    worker = release_root / "aeon/scripts/qwen_remote_worker.py"
    command = [
        *_ssh_base(capability),
        "/usr/bin/env",
        "-i",
        "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME=/home/aday",
        "LANG=C",
        "LC_ALL=C",
        "USE_TF=0",
        "USE_FLAX=0",
        *([] if cache is not None else [f"PYTHONPATH={release_root}"]),
        "PYTHONDONTWRITEBYTECODE=1",
        "/usr/bin/bash",
        str(REMOTE_WRAPPER),
        str(REMOTE_PYTHON),
    ]
    if cache is None:
        return [
            *command,
            str(worker),
            action,
        ]
    return [
        *command,
        "-I",
        "-B",
        "-c",
        shlex.quote(_REMOTE_CACHE_BOOTSTRAP),
        shlex.quote(str(worker)),
        action,
        shlex.quote(str(release_root)),
        cache["source"]["digest_sha256"],
        cache["source"]["filesystem_id"],
        SOURCE_MANIFEST_FILE,
    ]


def _parse_response(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if len(result.stdout or "") > 262144:
        raise QwenRuntimeError("remote runtime response is unbounded")
    try:
        value = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("remote runtime response is malformed") from exc
    if not isinstance(value, dict) or value.get("ok") is not True:
        detail = value.get("detail") if isinstance(value, dict) else None
        if not isinstance(detail, str) or len(detail) > 500:
            detail = "remote runtime refused the request"
        raise QwenRuntimeError(detail)
    if result.returncode != 0:
        raise QwenRuntimeError("remote runtime returned contradictory success")
    return value


def remote_call(
    capability: QwenRuntimeCapability,
    source: SourceIdentity | str,
    action: str,
    request: Mapping[str, Any],
    *,
    timeout: float,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    startup_check: Callable[[], None] | None = None,
    popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
) -> dict[str, Any]:
    payload = json.dumps(
        dict(request), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    # Mutating lifecycle calls, including atomic pre-container recovery, are
    # issued once. Their exact receipts make a later broker reconciliation
    # idempotent without risking an in-call replay after a transport ambiguity.
    attempts = 3 if action in {"preflight", "status", "reuse"} else 1
    result = None
    for attempt in range(attempts):
        command = _remote_command(
            capability,
            source,
            action,
            request.get("artifact_cache"),
        )
        if startup_check is None:
            result = command_runner(
                command,
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        else:
            startup_check()
            process = popen_factory(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + timeout
            try:
                if process.stdin is None:
                    raise QwenRuntimeError("remote runtime input pipe is unavailable")
                process.stdin.write(payload)
                process.stdin.close()
                process.stdin = None
                while True:
                    try:
                        stdout, stderr = process.communicate(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        startup_check()
                        if time.monotonic() >= deadline:
                            raise QwenRuntimeError("remote runtime call timed out")
                startup_check()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                raise
            result = subprocess.CompletedProcess(
                command,
                process.returncode,
                stdout,
                stderr,
            )
        if result.stdout or result.returncode not in {0, 255}:
            break
        if attempt + 1 < attempts:
            time.sleep(2)
    assert result is not None
    return _parse_response(result)


def remote_preflight(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    package_root: Path,
    *,
    artifact_cache: Mapping[str, Any] | QwenRemoteArtifactCache | None = None,
    startup_check: Callable[[], None] | None = None,
) -> tuple[SourceIdentity, dict[str, Any]]:
    source = (
        stage_remote_source(capability, package_root)
        if artifact_cache is None
        else _source_identity(package_root, RUNTIME_ROOT / "remote-preflight")
    )
    if isinstance(artifact_cache, QwenRemoteArtifactCache) and (
        artifact_cache.source["digest_sha256"] != source.manifest_sha256
    ):
        raise QwenRuntimeError("Fleet Qwen source cache differs from local source")
    request = _request_base(
        capability, manifest_sha256, source, artifact_cache
    )
    result = remote_call(
        capability,
        source,
        "preflight",
        request,
        timeout=1800,
        startup_check=startup_check,
    )
    return source, result


def capability_deploy_environment(
    capability: QwenRuntimeCapability,
    base_environment: Mapping[str, Any],
    lease: Mapping[str, Any],
) -> dict[str, str]:
    if (
        capability.vram_budget_gb is None
        or capability.gpu_memory_utilization is None
        or capability.max_num_seqs is None
        or capability.max_batched_tokens is None
    ):
        raise QwenRuntimeError("remote capability lacks its release plan")
    environment = {str(key): str(value) for key, value in base_environment.items()}
    try:
        plan = json.loads(environment["AEON_DEPLOY_PLAN"])
        nodes = plan["nodes"]
        if plan.get("tier") != "solo" or not isinstance(nodes, list) or len(nodes) != 1:
            raise ValueError
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("base Qwen deployment plan is malformed") from exc
    nodes[0]["ctx"] = capability.context_tokens
    nodes[0]["devices"] = str(lease["gpu_uuid"])
    plan["context_limit"] = capability.context_tokens
    plan["image"] = str(capability.image_id)
    environment.update(
        {
            "AEON_DEPLOY_PLAN": json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ),
            "AEON_GPU_MEM_UTIL": f"{capability.gpu_memory_utilization:g}",
            "AEON_LLM_VRAM_BUDGET_GB": f"{capability.vram_budget_gb:g}",
            "AEON_MAX_NUM_SEQS": str(capability.max_num_seqs),
            "AEON_MAX_NUM_BATCHED": str(capability.max_batched_tokens),
            "GPU_AGENT_CLAIM_ID": str(lease["claim_id"]),
            "GPU_LEASE_OWNER": str(lease["owner"]),
            "GPU_LEASE_RUN_DIR": str(lease["run_dir"]),
            "CUDA_VISIBLE_DEVICES": str(lease["gpu_uuid"]),
            "GPU_PLANNED_VRAM_GB": f"{capability.vram_budget_gb:g}",
            "GPU_RESERVE_GB": "6",
        }
    )
    return environment


def _bind_fleet_runtime_deploy_environment(
    deploy_environment: Mapping[str, Any],
    *,
    container_name: str,
    port: int,
) -> dict[str, Any]:
    """Copy and bind one coherent solo plan to an exact Fleet runtime."""

    if (
        not isinstance(container_name, str)
        or _CONTAINER_RE.fullmatch(container_name) is None
        or type(port) is not int
        or not 1024 <= port <= 65535
    ):
        raise QwenRuntimeError("Fleet remote runtime resources are malformed")
    environment = dict(deploy_environment)
    try:
        plan = json.loads(environment["AEON_DEPLOY_PLAN"])
        nodes = plan["nodes"]
        node = nodes[0]
        base_container_name = plan["container_name"]
        base_port = plan["health_port"]
        if (
            not isinstance(plan, dict)
            or plan.get("tier") != "solo"
            or not isinstance(nodes, list)
            or len(nodes) != 1
            or not isinstance(node, dict)
            or not isinstance(base_container_name, str)
            or _CONTAINER_RE.fullmatch(base_container_name) is None
            or plan.get("all_containers") != [base_container_name]
            or node.get("container") != base_container_name
            or type(base_port) is not int
            or not 1024 <= base_port <= 65535
            or type(plan.get("lb_port")) is not int
            or plan["lb_port"] != base_port
            or type(node.get("port")) is not int
            or node["port"] != base_port
        ):
            raise ValueError
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError(
            "Fleet Qwen deployment plan is not a coherent one-node release"
        ) from exc

    plan["container_name"] = container_name
    plan["all_containers"] = [container_name]
    plan["health_port"] = port
    plan["lb_port"] = port
    node["container"] = container_name
    node["port"] = port
    environment["AEON_DEPLOY_PLAN"] = json.dumps(
        plan, sort_keys=True, separators=(",", ":")
    )
    return environment


def _capability_for_state(
    state: Mapping[str, Any], *, require_enabled: bool
) -> tuple[QwenRuntimeCapability, str]:
    key = state.get("runtime_capability_key")
    try:
        capability, current_manifest_sha256 = qwen_runtime_capability(
            key, require_enabled=require_enabled
        )
    except QwenCapabilityError as exc:
        raise QwenRuntimeError("remote runtime capability is unavailable") from exc
    if any(
        state.get(field) != expected
        for field, expected in (
            ("runtime_adapter", capability.runtime_adapter),
            ("host", capability.host),
            ("expected_hostname", capability.hostname),
            ("model_manifest_sha256", capability.model_manifest_sha256),
            ("model_sha256s_sha256", capability.model_sha256s_sha256),
        )
    ):
        raise QwenRuntimeError("remote runtime capability receipt changed")
    if state.get("physical_gpu") not in capability.allowed_physical_gpus:
        raise QwenRuntimeError("remote runtime GPU is outside its capability")
    try:
        validate_qwen_capability_manifest_identity(
            key=capability.key,
            manifest_sha256=state.get("runtime_capability_manifest_sha256"),
            current_manifest_sha256=current_manifest_sha256,
            allow_retired_manifest=not require_enabled,
        )
    except QwenCapabilityError as exc:
        raise QwenRuntimeError("remote runtime capability manifest changed") from exc
    return capability, current_manifest_sha256


def _validate_remote_state(
    value: Any,
    *,
    require_enabled: bool = False,
    legacy_binding: bool | None = None,
) -> dict[str, Any]:
    base_fields = {
        "schema_version",
        "phase",
        "runtime_capability_key",
        "runtime_capability_manifest_sha256",
        "runtime_adapter",
        "host",
        "expected_hostname",
        "physical_gpu",
        "gpu_uuid",
        "claim_id",
        "owner",
        "run_dir",
        "source_manifest_sha256",
        "model_manifest_sha256",
        "model_sha256s_sha256",
        "container_name",
        "container_id",
        "container_pid",
        "remote_port",
        "local_port",
        "deploy_environment",
        "tunnel_nonce",
        "tunnel_pid",
        "tunnel_create_time",
        "updated_at",
    }
    if not isinstance(value, dict) or (
        value.get("schema_version") == 1
        and set(value) != base_fields
        or value.get("schema_version") == 2
        and set(value) != {*base_fields, "artifact_cache"}
        or value.get("schema_version") not in {1, 2}
    ):
        raise QwenRuntimeError("remote runtime receipt fields changed")
    capability, _manifest_sha256 = _capability_for_state(
        value, require_enabled=require_enabled
    )
    manifest = value.get("runtime_capability_manifest_sha256")
    physical_gpu = value.get("physical_gpu")
    remote_port = value.get("remote_port")
    local_port = value.get("local_port")
    updated_at = value.get("updated_at")
    container_id = value.get("container_id")
    container_pid = value.get("container_pid")
    tunnel_nonce = value.get("tunnel_nonce")
    tunnel_pid = value.get("tunnel_pid")
    tunnel_create_time = value.get("tunnel_create_time")
    environment = value.get("deploy_environment")
    run_dir = value.get("run_dir")
    artifact_cache = (
        None
        if value.get("schema_version") == 1
        else _validated_artifact_cache_request(value.get("artifact_cache"))
    )
    try:
        validated_run_dir = _validate_qwen_run_dir(run_dir)
        runtime_id = _fleet_runtime_id(validated_run_dir)
        fleet_resources = (
            fleet_remote_runtime_resources(
                validated_run_dir, physical_gpu, host=capability.host
            )
            if runtime_id is not None
            else None
        )
    except (TypeError, ValueError, QwenRuntimeError) as exc:
        raise QwenRuntimeError("remote runtime receipt is malformed") from exc
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] not in {1, 2}
        or value.get("phase") not in _REMOTE_PHASES
        or not isinstance(manifest, str)
        or _SHA256_RE.fullmatch(manifest) is None
        or type(physical_gpu) is not int
        or physical_gpu not in capability.allowed_physical_gpus
        or not isinstance(value.get("gpu_uuid"), str)
        or _UUID_RE.fullmatch(value["gpu_uuid"]) is None
        or not isinstance(value.get("claim_id"), str)
        or _CLAIM_RE.fullmatch(value["claim_id"]) is None
        or not isinstance(value.get("owner"), str)
        or _OWNER_RE.fullmatch(value["owner"]) is None
        or not isinstance(run_dir, str)
        or str(validated_run_dir) != run_dir
        or not isinstance(value.get("source_manifest_sha256"), str)
        or _SHA256_RE.fullmatch(value["source_manifest_sha256"]) is None
        or value.get("model_manifest_sha256") != capability.model_manifest_sha256
        or (
            artifact_cache is not None
            and (
                artifact_cache["source"]["digest_sha256"]
                != value.get("source_manifest_sha256")
                or artifact_cache["model"]["digest_sha256"]
                != capability.model_sha256s_sha256
                or artifact_cache["image"]["digest_sha256"]
                != str(capability.image_id).removeprefix("sha256:")
            )
        )
        or not isinstance(value.get("container_name"), str)
        or _CONTAINER_RE.fullmatch(value["container_name"]) is None
        or (container_id is not None and (
            not isinstance(container_id, str)
            or _CONTAINER_ID_RE.fullmatch(container_id) is None
        ))
        or (container_pid is not None and (type(container_pid) is not int or container_pid <= 1))
        or type(remote_port) is not int
        or not 1024 <= remote_port <= 65535
        or (
            (fleet_resources is None or legacy_binding is True)
            and local_port != LOCAL_PORT
        )
        or (
            fleet_resources is not None
            and legacy_binding is not True
            and capability.key not in COMPACT_REMOTE_DOCKER_CAPABILITY_KEYS
            and capability.key != RTX5000_RELEASE_CANDIDATE_KEY
        )
        or (
            fleet_resources is not None
            and legacy_binding is not True
            and any(
                value.get(field) != fleet_resources[field]
                for field in ("run_dir", "container_name", "remote_port", "local_port")
            )
        )
        or not isinstance(environment, dict)
        or len(environment) > 128
        or any(
            not isinstance(key, str)
            or not isinstance(item, str)
            or len(key) > 128
            or len(item) > 262144
            for key, item in environment.items()
        )
        or (tunnel_nonce is not None and (
            not isinstance(tunnel_nonce, str)
            or _SHA256_RE.fullmatch(tunnel_nonce) is None
        ))
        or (tunnel_pid is not None and (type(tunnel_pid) is not int or tunnel_pid <= 1))
        or (
            tunnel_create_time is not None
            and (type(tunnel_create_time) is not int or tunnel_create_time <= 0)
        )
        or not (
            (tunnel_nonce is None and tunnel_pid is None and tunnel_create_time is None)
            or (tunnel_nonce is not None and tunnel_pid is None and tunnel_create_time is None)
            or (tunnel_nonce is not None and tunnel_pid is not None and tunnel_create_time is not None)
        )
        or isinstance(updated_at, bool)
        or not isinstance(updated_at, (int, float))
        or not 0 < float(updated_at) < time.time() + 300
    ):
        raise QwenRuntimeError("remote runtime receipt is malformed")
    return dict(value)


def _validated_remote_state_at(
    path: Path, *, require_enabled: bool = False
) -> dict[str, Any] | None:
    value = _private_json_read(path)
    if value is None:
        return None
    return _validate_remote_state(
        value,
        require_enabled=require_enabled,
        legacy_binding=path == REMOTE_STATE_FILE,
    )


def _remote_state_entry(
    run_dir: str | Path, *, require_enabled: bool = False
) -> tuple[dict[str, Any] | None, Path, bool]:
    """Resolve one exact receipt, including a matching legacy singleton."""

    checked_run_dir = _validate_qwen_run_dir(run_dir)
    runtime_id = _fleet_runtime_id(checked_run_dir)
    if runtime_id is None:
        legacy = _validated_remote_state_at(
            REMOTE_STATE_FILE, require_enabled=require_enabled
        )
        if legacy is None or legacy.get("run_dir") != str(checked_run_dir):
            return None, REMOTE_STATE_FILE, True
        return legacy, REMOTE_STATE_FILE, True

    per_runtime_path = REMOTE_STATE_ROOT / f"{runtime_id}.json"
    per_runtime = _validated_remote_state_at(
        per_runtime_path, require_enabled=require_enabled
    )
    legacy = _validated_remote_state_at(
        REMOTE_STATE_FILE, require_enabled=require_enabled
    )
    if per_runtime is not None and per_runtime.get("run_dir") != str(checked_run_dir):
        raise QwenRuntimeError("per-runtime remote receipt path identity changed")
    matching_legacy = (
        legacy is not None and legacy.get("run_dir") == str(checked_run_dir)
    )
    if per_runtime is not None and matching_legacy:
        raise QwenRuntimeError("remote runtime has duplicate lifecycle receipts")
    if per_runtime is not None:
        return per_runtime, per_runtime_path, False
    if matching_legacy:
        return legacy, REMOTE_STATE_FILE, True
    return None, per_runtime_path, False


def _per_runtime_remote_states() -> tuple[dict[str, Any], ...]:
    try:
        root = REMOTE_STATE_ROOT.lstat()
    except FileNotFoundError:
        return ()
    if (
        not stat.S_ISDIR(root.st_mode)
        or root.st_uid != os.geteuid()
        or root.st_mode & 0o077
    ):
        raise QwenRuntimeError("per-runtime remote receipt directory is unsafe")
    states: list[dict[str, Any]] = []
    for path in sorted(REMOTE_STATE_ROOT.iterdir()):
        metadata = path.lstat()
        runtime_id = path.name.removesuffix(".json")
        if (
            path.suffix != ".json"
            or _FLEET_RUNTIME_ID_RE.fullmatch(runtime_id) is None
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink != 1
        ):
            raise QwenRuntimeError("per-runtime remote receipt entry is unsafe")
        state = _validated_remote_state_at(path)
        if state is None or Path(state["run_dir"]).name != runtime_id:
            raise QwenRuntimeError("per-runtime remote receipt entry changed")
        states.append(state)
    return tuple(states)


def _assert_remote_gpu_slot_available(
    *, host: str, physical_gpu: int, run_dir: str
) -> None:
    for state in _per_runtime_remote_states():
        if (
            state["run_dir"] != run_dir
            and state["host"] == host
            and state["physical_gpu"] == physical_gpu
        ):
            raise QwenRuntimeError(
                "another exact remote receipt already occupies this GPU slot"
            )


def remote_state(
    run_dir: str | Path | None = None, *, require_enabled: bool = False
) -> dict[str, Any] | None:
    """Read the exact run receipt; no run_dir intentionally means legacy only."""

    if run_dir is None:
        return _validated_remote_state_at(
            REMOTE_STATE_FILE, require_enabled=require_enabled
        )
    state, _path, _legacy = _remote_state_entry(
        run_dir, require_enabled=require_enabled
    )
    return state


def _worker_request_binding(
    state: Mapping[str, Any],
    receipt_path: Path,
    *,
    host: str = "192.168.0.180",
) -> dict[str, Any]:
    run_dir = _validate_qwen_run_dir(state.get("run_dir"))
    runtime_id = _fleet_runtime_id(run_dir)
    legacy = receipt_path == REMOTE_STATE_FILE
    if legacy:
        worker_path = RUNTIME_STATE_FILE
    else:
        if runtime_id is None:
            raise QwenRuntimeError("per-runtime receipt lacks a Fleet runtime identity")
        resources = fleet_remote_runtime_resources(
            run_dir,
            int(state.get("physical_gpu")),
            host=host,
        )
        if receipt_path != resources["orchestrator_state_path"]:
            raise QwenRuntimeError("orchestrator receipt path identity changed")
        worker_path = resources["worker_state_path"]
    return {
        "runtime_id": runtime_id,
        "run_dir": str(run_dir),
        "worker_receipt_mode": "legacy" if legacy else "per-runtime",
        "worker_state_path": str(worker_path),
        "container_name": state.get("container_name"),
        "port": state.get("remote_port"),
        "physical_gpu": state.get("physical_gpu"),
        "gpu_uuid": state.get("gpu_uuid"),
        "claim_id": state.get("claim_id"),
        "owner": state.get("owner"),
    }


def _worker_response_matches_state(
    response: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    allow_legacy_fields: bool = False,
    require_legacy_claim: bool = True,
) -> bool:
    fields = ("run_dir", "physical_gpu", "gpu_uuid", "claim_id", "owner")
    if all(
        response.get(field) == state.get(field)
        for field in fields
    ):
        return True
    if not allow_legacy_fields:
        return False
    if any(
        field in response and response[field] != state.get(field)
        for field in fields
    ):
        return False
    return not require_legacy_claim or response.get("claim_id") == state.get("claim_id")


def _runtime_request_base(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    source: SourceIdentity | str,
    state: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    return {
        **_request_base(
            capability,
            manifest_sha256,
            source,
            state.get("artifact_cache"),
        ),
        **_worker_request_binding(state, receipt_path, host=capability.host),
    }


def _remote_state_matches_lease(
    state: Mapping[str, Any], lease: Mapping[str, Any]
) -> bool:
    keys = (
        "runtime_capability_key",
        "runtime_capability_manifest_sha256",
        "runtime_adapter",
        "host",
        "physical_gpu",
        "gpu_uuid",
        "claim_id",
        "owner",
        "run_dir",
    )
    return all(state.get(key) == lease.get(key) for key in keys)


def _terminate_remote_start_controller(process: subprocess.Popen[str]) -> None:
    """Terminate and reap only the exact SSH child created for this start."""

    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
        process.stdin = None
    if process.poll() is None:
        process.terminate()
    try:
        process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired as exc:
            raise QwenRuntimeError(
                "remote Qwen start SSH controller could not be reaped"
            ) from exc
    if process.poll() is None:
        raise QwenRuntimeError("remote Qwen start SSH controller remains active")


def start_remote_runtime(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    source: SourceIdentity,
    lease: Mapping[str, Any],
    deploy_environment: Mapping[str, Any],
    *,
    receipt_path: Path,
    container_name: str,
    port: int,
    heartbeat_pid: Callable[[int], None],
    artifact_cache: Mapping[str, Any] | QwenRemoteArtifactCache | None = None,
    progress_check: Callable[[], None] | None = None,
    timeout: float = REMOTE_STARTUP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    cache = (
        None
        if artifact_cache is None
        else _validated_artifact_cache_request(
            artifact_cache.to_request()
            if isinstance(artifact_cache, QwenRemoteArtifactCache)
            else artifact_cache
        )
    )
    request_state = {
        "run_dir": lease.get("run_dir"),
        "physical_gpu": lease.get("physical_gpu"),
        "gpu_uuid": lease.get("gpu_uuid"),
        "claim_id": lease.get("claim_id"),
        "owner": lease.get("owner"),
        "container_name": container_name,
        "remote_port": int(port),
        **({} if cache is None else {"artifact_cache": cache}),
    }
    request = {
        **_runtime_request_base(
            capability,
            manifest_sha256,
            source,
            request_state,
            receipt_path,
        ),
        "lease": dict(lease),
        "deploy_environment": dict(deploy_environment),
        "container_name": container_name,
        "port": int(port),
    }
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
    process = subprocess.Popen(
        _remote_command(capability, source, "start", cache),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdin is not None
        process.stdin.write(payload)
        process.stdin.close()
        process.stdin = None
        deadline = time.monotonic() + float(timeout)
        bound_pid: int | None = None
        while process.poll() is None:
            if progress_check is not None:
                progress_check()
            try:
                status = remote_call(
                    capability,
                    source,
                    "status",
                    _runtime_request_base(
                        capability,
                        manifest_sha256,
                        source,
                        request_state,
                        receipt_path,
                    ),
                    timeout=20,
                )
                pid = status.get("container_pid")
                if not isinstance(pid, bool) and isinstance(pid, int) and pid > 1:
                    if bound_pid is not None and pid != bound_pid:
                        raise QwenRuntimeError(
                            "remote container PID changed during startup"
                        )
                    if bound_pid is None:
                        heartbeat_pid(pid)
                        bound_pid = pid
            except QwenRuntimeError:
                if bound_pid is not None:
                    raise
            if time.monotonic() >= deadline:
                raise QwenRuntimeError(
                    "remote Qwen startup exceeded its bounded timeout"
                )
            time.sleep(2)
        stdout, _stderr = process.communicate(timeout=5)
        result = _parse_response(
            subprocess.CompletedProcess(process.args, process.returncode, stdout, "")
        )
        pid = result.get("container_pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
            raise QwenRuntimeError("remote Qwen start has no exact PID")
        if bound_pid is None:
            heartbeat_pid(pid)
        elif pid != bound_pid:
            raise QwenRuntimeError("remote Qwen ready PID changed")
        return result
    except BaseException:
        _terminate_remote_start_controller(process)
        raise


def _process_create_time(pid: int) -> int:
    payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    end = payload.rfind(")")
    if end < 0:
        raise QwenRuntimeError("tunnel process stat is malformed")
    fields = payload[end + 2 :].split()
    return int(fields[19])


def _tunnel_argv(
    capability: QwenRuntimeCapability,
    local_port: int,
    remote_port: int,
    nonce: str,
) -> list[str]:
    if _SHA256_RE.fullmatch(nonce) is None:
        raise QwenRuntimeError("remote tunnel nonce is malformed")
    return [
        *_ssh_base(capability)[:-1],
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        f"ControlPath=/home/aday/.aeon/runtime/qwen38/tunnel-{nonce}.sock",
        "-L",
        f"127.0.0.1:{int(local_port)}:127.0.0.1:{int(remote_port)}",
        _ssh_base(capability)[-1],
    ]


def _process_argv(pid: int) -> list[str]:
    try:
        metadata = Path(f"/proc/{pid}").stat()
        if metadata.st_uid != os.geteuid():
            raise QwenRuntimeError("remote tunnel process owner changed")
        payload = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if payload and payload[-1] == b"":
            payload.pop()
        return [item.decode("utf-8") for item in payload]
    except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
        raise QwenRuntimeError("remote tunnel process identity is unavailable") from exc


def _assert_loopback_port_available(port: int) -> None:
    """Refuse an occupied port instead of discovering/adopting a process.

    The durable runtime receipt is the only tunnel ownership source.  A process
    scan can accidentally classify another same-UID SSH process as ours, so a
    PID-less receipt is launchable only while its exact loopback port can be
    reserved by this lifecycle transaction.
    """

    if type(port) is not int or not 1024 <= port <= 65535:
        raise QwenRuntimeError("remote tunnel loopback port is invalid")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        probe.bind(("127.0.0.1", port))
    except OSError as exc:
        raise QwenRuntimeError(
            "remote tunnel loopback port is occupied without a usable receipt"
        ) from exc
    finally:
        probe.close()


def _bounded_loopback_body(response: Any, maximum: int) -> bytes:
    """Consume and close one streamed loopback response within an exact bound."""

    if type(maximum) is not int or maximum <= 0:
        raise QwenRuntimeError("remote tunnel response bound is invalid")
    payload = bytearray()
    try:
        advertised = response.headers.get("content-length")
        if advertised is not None:
            try:
                advertised_size = int(advertised)
            except (TypeError, ValueError) as exc:
                raise QwenRuntimeError(
                    "remote tunnel response Content-Length is malformed"
                ) from exc
            if advertised_size < 0 or advertised_size > maximum:
                raise QwenRuntimeError("remote tunnel response exceeded its bound")
        for chunk in response.iter_content(chunk_size=min(64 * 1024, maximum + 1)):
            payload.extend(chunk)
            if len(payload) > maximum:
                raise QwenRuntimeError("remote tunnel response exceeded its bound")
    finally:
        response.close()
    return bytes(payload)


def start_tunnel(
    capability: QwenRuntimeCapability,
    state: Mapping[str, Any],
    *,
    receipt_path: Path | None = None,
    health_timeout: float = 30,
) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise QwenRuntimeError(
            "local remote-tunnel HTTP dependency is unavailable"
        ) from exc
    raw = dict(state)
    current, resolved_path, legacy = _remote_state_entry(raw.get("run_dir"))
    checked = _validate_remote_state(
        raw,
        require_enabled=capability.key != RTX5000_RELEASE_CANDIDATE_KEY,
        legacy_binding=legacy,
    )
    if (
        current is None
        or current.get("claim_id") != checked["claim_id"]
        or (receipt_path is not None and receipt_path != resolved_path)
    ):
        raise QwenRuntimeError("remote tunnel receipt binding changed")
    receipt_path = resolved_path
    if checked.get("tunnel_pid") is not None:
        if tunnel_is_exact(checked):
            return checked
        raise QwenRuntimeError("remote tunnel receipt is ambiguous")
    if checked.get("tunnel_nonce") is not None:
        # A nonce without a PID is a previously published launch intent.  The
        # process may have escaped before its PID receipt was committed, so an
        # automatic retry would race/adopt unknown port ownership.
        raise QwenRuntimeError("remote tunnel launch intent is ambiguous")
    nonce = os.urandom(32).hex()
    intent = {
        **checked,
        "tunnel_nonce": nonce,
        "tunnel_pid": None,
        "tunnel_create_time": None,
        "updated_at": time.time(),
    }
    _private_json_write(receipt_path, intent)
    argv = _tunnel_argv(
        capability,
        int(checked["local_port"]),
        int(checked["remote_port"]),
        nonce,
    )
    _assert_loopback_port_available(int(checked["local_port"]))
    process = None
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid = process.pid
        receipt = {
            **intent,
            "tunnel_pid": pid,
            "tunnel_create_time": _process_create_time(pid),
            "updated_at": time.time(),
        }
        _private_json_write(receipt_path, receipt)
        deadline = time.monotonic() + health_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise QwenRuntimeError("remote Qwen tunnel exited before health")
            if not tunnel_is_exact(receipt):
                raise QwenRuntimeError("remote Qwen tunnel identity changed before health")
            try:
                response = requests.get(
                    f"http://127.0.0.1:{checked['local_port']}/health",
                    timeout=2,
                    allow_redirects=False,
                    proxies={"http": "", "https": ""},
                    stream=True,
                )
                status_code = response.status_code
                _bounded_loopback_body(response, 64 * 1024)
                if status_code == 200:
                    return receipt
            except (requests.RequestException, QwenRuntimeError):
                pass
            time.sleep(0.5)
        raise QwenRuntimeError("remote Qwen tunnel did not become healthy")
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # The durable receipt (or occupied port, if receipt publication
                # failed) keeps this ambiguous tunnel unavailable for recovery.
                pass
        raise


def tunnel_is_exact(state: Mapping[str, Any]) -> bool:
    pid = state.get("tunnel_pid")
    create_time = state.get("tunnel_create_time")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 1
        or isinstance(create_time, bool)
        or not isinstance(create_time, int)
    ):
        return False
    try:
        if _process_create_time(pid) != create_time:
            return False
        capability, _manifest = _capability_for_state(state, require_enabled=False)
        return _process_argv(pid) == _tunnel_argv(
            capability,
            int(state["local_port"]),
            int(state["remote_port"]),
            str(state["tunnel_nonce"]),
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, QwenRuntimeError):
        return False


def tunnel_liveness(state: Mapping[str, Any]) -> str:
    pid = state.get("tunnel_pid")
    if pid is None:
        return "gone"
    if type(pid) is not int or pid <= 1:
        return "ambiguous"
    try:
        Path(f"/proc/{pid}").stat()
    except FileNotFoundError:
        return "gone"
    except OSError:
        return "ambiguous"
    return "active" if tunnel_is_exact(state) else "ambiguous"


def stop_tunnel(state: Mapping[str, Any]) -> bool:
    if state.get("tunnel_pid") is None:
        return True
    liveness = tunnel_liveness(state)
    if liveness == "gone":
        return True
    if liveness != "active":
        return False
    pid = int(state["tunnel_pid"])
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        # If this Aeon process is still the tunnel's parent, reap an exited
        # child so procfs can prove the PID number is truly absent.  A tunnel
        # inherited from an earlier session simply raises ChildProcessError.
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            pass
        liveness = tunnel_liveness(state)
        if liveness == "gone":
            return True
        if liveness != "active":
            # Identity drift or PID reuse is not proof that our exact process
            # exited, and must never authorize remote runtime cleanup.
            return False
        time.sleep(0.1)
    return False


def remote_runtime_liveness(run_dir: str | Path | None = None) -> str:
    """Return active/exited/gone/ambiguous for the exact saved worker runtime."""

    try:
        if run_dir is None:
            state = remote_state()
            receipt_path = REMOTE_STATE_FILE
            legacy = True
        else:
            state, receipt_path, legacy = _remote_state_entry(run_dir)
        if state is None:
            return "gone"
        capability, _current_manifest = _capability_for_state(
            state, require_enabled=False
        )
        result = remote_call(
            capability,
            str(state["source_manifest_sha256"]),
            "status",
            _runtime_request_base(
                capability,
                str(state["runtime_capability_manifest_sha256"]),
                str(state["source_manifest_sha256"]),
                state,
                receipt_path,
            ),
            timeout=30,
        )
        status = result.get("state")
        if status not in {"active", "exited", "gone"}:
            return "ambiguous"
        if status == "gone":
            return "gone"
        if not _worker_response_matches_state(
            result, state, allow_legacy_fields=legacy
        ):
            return "ambiguous"
        container_id = result.get("container_id")
        if (
            not isinstance(container_id, str)
            or _CONTAINER_ID_RE.fullmatch(container_id) is None
            or (
                state.get("container_id") is not None
                and container_id != state["container_id"]
            )
        ):
            return "ambiguous"
        pid = result.get("container_pid")
        if status == "active" and (
            type(pid) is not int
            or pid <= 1
            or (
                state.get("container_pid") is not None
                and pid != state["container_pid"]
            )
        ):
            return "ambiguous"
        return str(status)
    except Exception:
        return "ambiguous"


_RECOVERED_PRECONTAINER_FIELDS = frozenset(
    {
        "ok",
        "state",
        "controller_protocol",
        "process_absent",
        "worker_receipt_absent",
        "run_dir",
        "physical_gpu",
        "gpu_uuid",
        "claim_id",
        "owner",
    }
)


def _recover_remote_precontainer_intent(
    state: Mapping[str, Any], receipt_path: Path, *, legacy: bool
) -> bool:
    """Atomically settle one lock-aware worker pre-container lifecycle."""

    if (
        legacy
        or state.get("phase") not in {"starting", "releasing"}
        or any(
            state.get(field) is not None
            for field in (
                "container_id",
                "container_pid",
                "tunnel_nonce",
                "tunnel_pid",
                "tunnel_create_time",
            )
        )
    ):
        return False
    try:
        capability, _current_manifest = _capability_for_state(
            state, require_enabled=False
        )
        result = remote_call(
            capability,
            str(state["source_manifest_sha256"]),
            "recover-precontainer",
            _runtime_request_base(
                capability,
                str(state["runtime_capability_manifest_sha256"]),
                str(state["source_manifest_sha256"]),
                state,
                receipt_path,
            ),
            timeout=120,
        )
    except Exception:
        return False
    return (
        frozenset(result) == _RECOVERED_PRECONTAINER_FIELDS
        and result.get("ok") is True
        and result.get("state") == "recovered"
        and type(result.get("controller_protocol")) is int
        and result.get("controller_protocol") == 1
        and result.get("process_absent") is True
        and type(result.get("worker_receipt_absent")) is bool
        and _worker_response_matches_state(result, state)
    )


def recover_remote_precontainer_intent(run_dir: str | Path) -> bool:
    """Recover only through the worker's exact lock-held atomic action."""

    try:
        state, receipt_path, legacy = _remote_state_entry(run_dir)
    except Exception:
        return False
    if state is None:
        return False
    return _recover_remote_precontainer_intent(
        state, receipt_path, legacy=legacy
    )


_REMOTE_STATUS_FIELDS = frozenset(
    {
        "ok",
        "state",
        "phase",
        "container_pid",
        "container_id",
        "run_dir",
        "physical_gpu",
        "gpu_uuid",
        "claim_id",
        "owner",
        "scratch_cleaned",
    }
)
_RECOVERED_UNCOMMITTED_FIELDS = frozenset(
    {
        "ok",
        "state",
        "controller_protocol",
        "process_absent",
        "worker_receipt_absent",
        "container_id",
        "container_pid",
        "run_dir",
        "physical_gpu",
        "gpu_uuid",
        "claim_id",
        "owner",
    }
)


def _remote_uncommitted_worker_status(
    state: Mapping[str, Any], receipt_path: Path, *, legacy: bool
) -> tuple[dict[str, Any], str] | None:
    """Read one exact startup status, including the reviewed old retry state."""

    if (
        legacy
        or state.get("phase") != "starting"
        or type(state.get("container_pid")) is not int
        or state["container_pid"] <= 1
    ):
        return None
    capability, _current_manifest = _capability_for_state(
        state, require_enabled=False
    )
    result = remote_call(
        capability,
        str(state["source_manifest_sha256"]),
        "status",
        _runtime_request_base(
            capability,
            str(state["runtime_capability_manifest_sha256"]),
            str(state["source_manifest_sha256"]),
            state,
            receipt_path,
        ),
        timeout=30,
    )
    container_id = result.get("container_id")
    active = (
        result.get("state") == "active"
        and result.get("phase") in {"launching", "ready"}
        and type(result.get("container_pid")) is int
        and result.get("container_pid") == state["container_pid"]
    )
    exited_pid_cleared = (
        result.get("state") == "exited"
        and result.get("phase") == "launching"
        and result.get("container_pid") is None
        and result.get("scratch_cleaned") is False
    )
    compatibility_dirty_releasing = (
        _uses_pid_cleared_stop_clear_compatibility(state)
        and isinstance(state.get("container_id"), str)
        and result.get("container_id") == state["container_id"]
        and result.get("state") in {"exited", "gone"}
        and result.get("phase") == "releasing"
        and result.get("container_pid") is None
        and result.get("scratch_cleaned") is False
    )
    if (
        frozenset(result) != _REMOTE_STATUS_FIELDS
        or result.get("ok") is not True
        or not (active or exited_pid_cleared or compatibility_dirty_releasing)
        or type(result.get("scratch_cleaned")) is not bool
        or not isinstance(container_id, str)
        or _CONTAINER_ID_RE.fullmatch(container_id) is None
        or (
            state.get("container_id") is not None
            and state.get("container_id") != container_id
        )
        or not _worker_response_matches_state(result, state)
    ):
        return None
    if compatibility_dirty_releasing:
        status = "compatibility_dirty_releasing"
    elif exited_pid_cleared:
        status = "exited_pid_cleared"
    else:
        status = "active"
    return dict(result), status


def _bind_remote_uncommitted_container_id(
    state: Mapping[str, Any], receipt_path: Path, *, legacy: bool
) -> tuple[dict[str, Any], str] | None:
    """Read and bind the worker's immutable ID before atomic recovery."""

    if state.get("container_id") is not None:
        return None
    observed = _remote_uncommitted_worker_status(
        state, receipt_path, legacy=legacy
    )
    if observed is None:
        return None
    result, status = observed
    container_id = str(result["container_id"])
    current, current_path, current_legacy = _remote_state_entry(state["run_dir"])
    if current != state or current_path != receipt_path or current_legacy:
        return None
    bound = {**state, "container_id": container_id, "updated_at": time.time()}
    _private_json_write(receipt_path, bound)
    return bound, status


def recover_remote_uncommitted_intent(run_dir: str | Path) -> bool:
    """Atomically recover one exact PID-bound launch Fleet never committed."""

    try:
        state, receipt_path, legacy = _remote_state_entry(run_dir)
        if (
            state is None
            or legacy
            or state.get("phase") not in {"starting", "releasing"}
            or type(state.get("container_pid")) is not int
            or state["container_pid"] <= 1
            or any(
                state.get(field) is not None
                for field in (
                    "tunnel_nonce",
                    "tunnel_pid",
                    "tunnel_create_time",
                )
            )
        ):
            return False
        worker_status: str | None = None
        if state.get("container_id") is None:
            bound = _bind_remote_uncommitted_container_id(
                state, receipt_path, legacy=legacy
            )
            if bound is None:
                return False
            state, worker_status = bound
        elif (
            state.get("phase") == "starting"
            and _uses_pid_cleared_stop_clear_compatibility(state)
        ):
            # Re-prove the narrow legacy condition after a crash that may have
            # persisted the immutable ID before its controller-locked cleanup.
            observed = _remote_uncommitted_worker_status(
                state, receipt_path, legacy=legacy
            )
            if observed is not None:
                current, current_path, current_legacy = _remote_state_entry(
                    state["run_dir"]
                )
                if (
                    current != state
                    or current_path != receipt_path
                    or current_legacy
                ):
                    return False
                _result, worker_status = observed
        container_id = state.get("container_id")
        container_pid = state.get("container_pid")
        if (
            not isinstance(container_id, str)
            or _CONTAINER_ID_RE.fullmatch(container_id) is None
            or type(container_pid) is not int
            or container_pid <= 1
        ):
            return False
        capability, _current_manifest = _capability_for_state(
            state, require_enabled=False
        )
        if (
            worker_status
            in {"exited_pid_cleared", "compatibility_dirty_releasing"}
            and _uses_pid_cleared_stop_clear_compatibility(state)
        ):
            if not stop_managed_remote_runtime(
                capability,
                str(state["runtime_capability_manifest_sha256"]),
                str(state["source_manifest_sha256"]),
                release_reason=(
                    "recover exact exited uncommitted startup with cleared PID"
                ),
                release_claim=False,
                run_dir=str(state["run_dir"]),
            ):
                return False
            current, current_path, current_legacy = _remote_state_entry(run_dir)
            return (
                current is None
                and current_path == receipt_path
                and not current_legacy
                and remote_runtime_liveness(run_dir) == "gone"
            )
        request = {
            **_runtime_request_base(
                capability,
                str(state["runtime_capability_manifest_sha256"]),
                str(state["source_manifest_sha256"]),
                state,
                receipt_path,
            ),
            "expected_container_id": container_id,
            "expected_container_pid": container_pid,
        }
        result = remote_call(
            capability,
            str(state["source_manifest_sha256"]),
            "recover-uncommitted",
            request,
            timeout=120,
        )
        if (
            frozenset(result) != _RECOVERED_UNCOMMITTED_FIELDS
            or result.get("state") != "recovered"
            or type(result.get("controller_protocol")) is not int
            or result.get("controller_protocol") != 2
            or result.get("process_absent") is not True
            or type(result.get("worker_receipt_absent")) is not bool
            or result.get("container_id") != container_id
            or type(result.get("container_pid")) is not int
            or result.get("container_pid") != container_pid
            or not _worker_response_matches_state(result, state)
            or not stop_tunnel(state)
            or remote_runtime_liveness(run_dir) != "gone"
        ):
            return False
        current, current_path, current_legacy = _remote_state_entry(run_dir)
        if (
            current is None
            or current_path != receipt_path
            or current_legacy
            or current != state
        ):
            return False
        releasing = {
            **current,
            "phase": "releasing",
            "tunnel_nonce": None,
            "tunnel_pid": None,
            "tunnel_create_time": None,
            "updated_at": time.time(),
        }
        _private_json_write(receipt_path, releasing)
        settled, settled_path, settled_legacy = _remote_state_entry(run_dir)
        if settled != releasing or settled_path != receipt_path or settled_legacy:
            return False
        metadata = receipt_path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
        ):
            return False
        receipt_path.unlink()
        return True
    except Exception:
        return False


def restore_managed_remote_tunnel(run_dir: str | Path) -> dict[str, Any]:
    """Restore only a provably-gone tunnel for one exact active Fleet runtime."""

    state, receipt_path, _legacy = _remote_state_entry(
        run_dir, require_enabled=True
    )
    if state is None:
        raise QwenRuntimeError("remote Qwen runtime receipt is absent")
    if state.get("phase") != "ready":
        raise QwenRuntimeError("remote Qwen runtime is not ready for tunnel recovery")
    capability, _current_manifest = _capability_for_state(
        state, require_enabled=True
    )
    if remote_runtime_liveness(run_dir) != "active":
        raise QwenRuntimeError(
            "remote Qwen runtime is not exactly active for tunnel recovery"
        )
    tunnel_status = tunnel_liveness(state)
    if tunnel_status == "active":
        return state
    if tunnel_status != "gone":
        raise QwenRuntimeError("remote Qwen tunnel identity is ambiguous")
    without_tunnel = {
        **state,
        "tunnel_nonce": None,
        "tunnel_pid": None,
        "tunnel_create_time": None,
        "updated_at": time.time(),
    }
    _private_json_write(receipt_path, without_tunnel)
    return start_tunnel(
        capability,
        without_tunnel,
        receipt_path=receipt_path,
    )


def remote_container_pid(run_dir: str | Path | None = None) -> int | None:
    state = remote_state(run_dir, require_enabled=True)
    if state is None:
        return None
    if remote_runtime_liveness(run_dir) != "active":
        if state.get("phase") == "starting" and state.get("container_pid") is None:
            return None
        raise QwenRuntimeError("remote Qwen container PID is not exactly active")
    pid = state.get("container_pid")
    if type(pid) is not int or pid <= 1:
        raise QwenRuntimeError("remote Qwen receipt has no exact active PID")
    return pid


def source_receipt_hash(source: SourceIdentity) -> str:
    return hashlib.sha256(source.manifest_bytes).hexdigest()


def start_managed_remote_runtime(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    source: SourceIdentity,
    lease: Mapping[str, Any],
    deploy_environment: Mapping[str, Any],
    *,
    container_name: str,
    port: int,
    heartbeat_pid: Callable[[int], None],
    artifact_cache: Mapping[str, Any] | QwenRemoteArtifactCache | None = None,
    progress_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from .qwen_runtime import verify_coordinator_lease

    checked = verify_coordinator_lease(lease)
    runtime_id = _fleet_runtime_id(checked["run_dir"])
    if runtime_id is None:
        receipt_path = REMOTE_STATE_FILE
        effective_container_name = container_name
        effective_remote_port = int(port)
        effective_local_port = LOCAL_PORT
        effective_deploy_environment = dict(deploy_environment)
        if _validated_remote_state_at(REMOTE_STATE_FILE) is not None:
            raise QwenRuntimeError("a legacy remote Qwen lifecycle receipt already exists")
    else:
        if (
            capability.key not in COMPACT_REMOTE_DOCKER_CAPABILITY_KEYS
            and capability.key != RTX5000_RELEASE_CANDIDATE_KEY
            or capability.max_num_seqs != 8
        ):
            raise QwenRuntimeError(
                "Fleet multi-instance runtime is not the released batched adapter"
            )
        resources = fleet_remote_runtime_resources(
            checked["run_dir"],
            checked["physical_gpu"],
            host=capability.host,
        )
        receipt_path = resources["orchestrator_state_path"]
        effective_container_name = str(resources["container_name"])
        effective_remote_port = int(resources["remote_port"])
        effective_local_port = int(resources["local_port"])
        if (
            container_name != effective_container_name
            or type(port) is not int
            or port != effective_remote_port
        ):
            raise QwenRuntimeError("Fleet remote runtime resources were not exactly derived")
        effective_deploy_environment = _bind_fleet_runtime_deploy_environment(
            deploy_environment,
            container_name=effective_container_name,
            port=effective_remote_port,
        )
        existing, _existing_path, _legacy = _remote_state_entry(checked["run_dir"])
        if existing is not None:
            raise QwenRuntimeError("this Fleet remote lifecycle receipt already exists")
        _assert_remote_gpu_slot_available(
            host=capability.host,
            physical_gpu=checked["physical_gpu"],
            run_dir=checked["run_dir"],
        )
        legacy = _validated_remote_state_at(REMOTE_STATE_FILE)
        if (
            legacy is not None
            and legacy.get("host") == capability.host
            and legacy.get("physical_gpu") == checked["physical_gpu"]
        ):
            raise QwenRuntimeError("legacy remote receipt already occupies this GPU slot")
    cache = (
        None
        if artifact_cache is None
        else _validated_artifact_cache_request(
            artifact_cache.to_request()
            if isinstance(artifact_cache, QwenRemoteArtifactCache)
            else artifact_cache
        )
    )
    intent = {
        "schema_version": 1 if cache is None else 2,
        "phase": "starting",
        "runtime_capability_key": capability.key,
        "runtime_capability_manifest_sha256": manifest_sha256,
        "runtime_adapter": capability.runtime_adapter,
        "host": capability.host,
        "expected_hostname": capability.hostname,
        "physical_gpu": checked["physical_gpu"],
        "gpu_uuid": checked["gpu_uuid"],
        "claim_id": checked["claim_id"],
        "owner": checked["owner"],
        "run_dir": checked["run_dir"],
        "source_manifest_sha256": source.manifest_sha256,
        "model_manifest_sha256": capability.model_manifest_sha256,
        "model_sha256s_sha256": capability.model_sha256s_sha256,
        "container_name": effective_container_name,
        "container_id": None,
        "container_pid": None,
        "remote_port": effective_remote_port,
        "local_port": effective_local_port,
        "deploy_environment": dict(effective_deploy_environment),
        "tunnel_nonce": None,
        "tunnel_pid": None,
        "tunnel_create_time": None,
        "updated_at": time.time(),
        **({} if cache is None else {"artifact_cache": cache}),
    }
    _private_json_write(receipt_path, intent)

    def bind_remote_pid(pid: int) -> None:
        current, current_path, _legacy = _remote_state_entry(checked["run_dir"])
        if current is None or current.get("claim_id") != intent["claim_id"]:
            raise QwenRuntimeError("remote startup receipt changed before PID binding")
        if current_path != receipt_path:
            raise QwenRuntimeError("remote startup receipt path changed before PID binding")
        saved_pid = current.get("container_pid")
        if saved_pid is not None and saved_pid != pid:
            raise QwenRuntimeError("remote startup PID identity changed")
        _private_json_write(
            receipt_path,
            {**current, "container_pid": pid, "updated_at": time.time()},
        )
        heartbeat_pid(pid)

    result = start_remote_runtime(
        capability,
        manifest_sha256,
        source,
        checked,
        effective_deploy_environment,
        receipt_path=receipt_path,
        container_name=effective_container_name,
        port=effective_remote_port,
        heartbeat_pid=bind_remote_pid,
        artifact_cache=cache,
        progress_check=progress_check,
    )
    container_id = result.get("container_id")
    container_pid = result.get("container_pid")
    if (
        not _worker_response_matches_state(result, intent)
        or
        not isinstance(container_id, str)
        or _CONTAINER_ID_RE.fullmatch(container_id) is None
        or type(container_pid) is not int
        or container_pid <= 1
    ):
        raise QwenRuntimeError("remote Qwen ready identity is malformed")
    current, current_path, _legacy = _remote_state_entry(
        checked["run_dir"],
        require_enabled=capability.key != RTX5000_RELEASE_CANDIDATE_KEY,
    )
    if (
        current is None
        or current_path != receipt_path
        or not _remote_state_matches_lease(current, checked)
    ):
        raise QwenRuntimeError("remote Qwen receipt changed before readiness")
    ready = {
        **current,
        "phase": "ready",
        "container_id": container_id,
        "container_pid": container_pid,
        "updated_at": time.time(),
    }
    _private_json_write(receipt_path, ready)
    return start_tunnel(capability, ready, receipt_path=receipt_path)


def reuse_managed_remote_runtime(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    source: SourceIdentity,
    lease: Mapping[str, Any],
    *,
    container_name: str,
    port: int,
) -> int | None:
    from .qwen_runtime import verify_coordinator_lease

    checked = verify_coordinator_lease(lease)
    state, receipt_path, legacy = _remote_state_entry(
        checked["run_dir"], require_enabled=True
    )
    if state is None:
        return None
    if not legacy:
        resources = fleet_remote_runtime_resources(
            checked["run_dir"],
            checked["physical_gpu"],
            host=capability.host,
        )
        if (
            container_name != resources["container_name"]
            or type(port) is not int
            or port != resources["remote_port"]
        ):
            raise QwenRuntimeError("Fleet remote reuse resources were not exactly derived")
    if any(
        state.get(key) != expected
        for key, expected in (
            ("runtime_capability_key", capability.key),
            ("runtime_capability_manifest_sha256", manifest_sha256),
            ("host", capability.host),
            ("physical_gpu", checked["physical_gpu"]),
            ("gpu_uuid", checked["gpu_uuid"]),
            ("claim_id", checked["claim_id"]),
            ("owner", checked["owner"]),
            ("run_dir", checked["run_dir"]),
            ("source_manifest_sha256", source.manifest_sha256),
            ("container_name", container_name),
            ("remote_port", int(port)),
        )
    ):
        raise QwenRuntimeError("remote runtime receipt differs from its exact lease")
    request = {
        **_runtime_request_base(
            capability, manifest_sha256, source, state, receipt_path
        ),
        "lease": dict(checked),
        "config": {
            "container_name": container_name,
            "health_port": int(port),
            "_deploy_env": dict(state["deploy_environment"]),
        },
    }
    result = remote_call(capability, source, "reuse", request, timeout=60)
    if not _worker_response_matches_state(
        result,
        state,
        allow_legacy_fields=legacy,
        require_legacy_claim=False,
    ):
        raise QwenRuntimeError("remote reuse lease identity changed")
    pid = result.get("container_pid")
    if result.get("state") == "gone":
        return None
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise QwenRuntimeError("remote reuse has no exact container PID")
    if state.get("container_pid") is not None and state["container_pid"] != pid:
        raise QwenRuntimeError("remote reuse container PID changed")
    if (
        not legacy
        and result.get("container_id") != state.get("container_id")
    ):
        raise QwenRuntimeError("remote reuse container identity changed")
    state = {
        **state,
        "phase": "ready",
        "container_pid": pid,
        "updated_at": time.time(),
    }
    _private_json_write(receipt_path, state)
    restore_managed_remote_tunnel(checked["run_dir"])
    return pid


def stop_managed_remote_runtime(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    source: SourceIdentity,
    *,
    release_reason: str,
    release_claim: bool = True,
    run_dir: str | Path | None = None,
    require_unlaunched: bool = False,
) -> bool:
    from .gpu_queue import release_vram

    if run_dir is None:
        state = remote_state()
        receipt_path = REMOTE_STATE_FILE
        legacy = True
    else:
        state, receipt_path, legacy = _remote_state_entry(run_dir)
    if state is None:
        return True
    saved_capability, _current_manifest = _capability_for_state(
        state, require_enabled=False
    )
    if (
        capability.key != saved_capability.key
        or manifest_sha256 != state["runtime_capability_manifest_sha256"]
        or _source_sha256(source) != state["source_manifest_sha256"]
    ):
        raise QwenRuntimeError("remote stop identity differs from its saved receipt")
    request = _runtime_request_base(
        saved_capability,
        str(state["runtime_capability_manifest_sha256"]),
        str(state["source_manifest_sha256"]),
        state,
        receipt_path,
    )
    unlaunched = (
        state.get("phase") in {"starting", "releasing"}
        and state.get("container_id") is None
        and state.get("container_pid") is None
    )
    if require_unlaunched and not unlaunched:
        return False
    worker_precontainer_recovered = False
    if unlaunched and not legacy:
        worker_precontainer_recovered = _recover_remote_precontainer_intent(
            state, receipt_path, legacy=legacy
        )
        if not worker_precontainer_recovered:
            # Older reviewed workers could adopt an exact exited container
            # after Docker wrote its cidfile but returned a failed create
            # result, then refuse the narrower pre-container recovery action.
            # Their ordinary receipt-bound stop action already proves and
            # removes that exact immutable container.  Use it only after the
            # old status path proves an exited (never live) process.
            if remote_runtime_liveness(str(state["run_dir"])) != "exited":
                return False
            stopped = remote_call(
                saved_capability,
                str(state["source_manifest_sha256"]),
                "stop",
                request,
                timeout=90,
            )
            if (
                stopped.get("state") != "stopped"
                or stopped.get("scratch_cleaned") is not True
                or not _worker_response_matches_state(stopped, state)
            ):
                return False
        if not stop_tunnel(state):
            return False
        state = {
            **state,
            "phase": "releasing",
            "tunnel_nonce": None,
            "tunnel_pid": None,
            "tunnel_create_time": None,
            "updated_at": time.time(),
        }
        _private_json_write(receipt_path, state)
    elif state.get("phase") != "releasing":
        if legacy and not unlaunched:
            observed = remote_call(
                saved_capability,
                str(state["source_manifest_sha256"]),
                "status",
                request,
                timeout=30,
            )
            if (
                observed.get("state") not in {"active", "exited"}
                or observed.get("claim_id") != state["claim_id"]
                or observed.get("container_id") != state["container_id"]
                or (
                    observed.get("state") == "active"
                    and observed.get("container_pid") != state["container_pid"]
                )
            ):
                return False
        result = remote_call(
            saved_capability,
            str(state["source_manifest_sha256"]),
            "stop",
            request,
            timeout=90,
        )
        if (
            result.get("state") != "stopped"
            or (
                not unlaunched
                and not _worker_response_matches_state(
                    result,
                    state,
                    allow_legacy_fields=legacy,
                    require_legacy_claim=False,
                )
            )
            or (result.get("scratch_cleaned") is not True and not unlaunched)
        ):
            return False
        if not stop_tunnel(state):
            return False
        state = {
            **state,
            "phase": "releasing",
            "tunnel_nonce": None,
            "tunnel_pid": None,
            "tunnel_create_time": None,
            "updated_at": time.time(),
        }
        _private_json_write(receipt_path, state)
    if release_claim:
        release_vram(
            release_reason,
            QWEN_LEASE_FILE,
            expected_claim_id=str(state["claim_id"]),
        )
    if not worker_precontainer_recovered:
        cleared = remote_call(
            saved_capability,
            str(state["source_manifest_sha256"]),
            "clear",
            request,
            timeout=30,
        )
        if cleared.get("state") != "cleared":
            return False
        if (
            cleared.get("receipt_absent") is not True
            and not _worker_response_matches_state(
                cleared,
                state,
                allow_legacy_fields=legacy,
                require_legacy_claim=False,
            )
        ):
            return False
    if run_dir is None:
        current = remote_state()
        current_path = REMOTE_STATE_FILE
    else:
        current, current_path, _legacy = _remote_state_entry(run_dir)
    if (
        current is None
        or current_path != receipt_path
        or current.get("claim_id") != state["claim_id"]
    ):
        raise QwenRuntimeError("remote release receipt changed before clear")
    metadata = receipt_path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        raise QwenRuntimeError("remote release receipt is unsafe")
    receipt_path.unlink()
    return True
