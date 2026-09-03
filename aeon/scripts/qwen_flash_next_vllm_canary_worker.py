#!/usr/bin/env python3
"""Fleet-owned .177 worker for the disabled Flash-Next vLLM canary.

Every Docker operation is bound to a unique Fleet runtime, claim hash, leased
GPU UUID, immutable image digest, and private evidence directory.  The module
has no import-time side effects and never uses host GPU discovery interfaces.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tarfile
import time
from typing import Any, Mapping, Sequence

import requests

from aeon.core import qwen_flash_next_vllm_contract as contract
from aeon.scripts import qualify_qwen38_flash_next_vllm as qualify


SCHEMA = "aeon-qwen38-flash-next-vllm-canary-job-v1"
STATUS_SCHEMA = "aeon-qwen38-flash-next-vllm-canary-status-v1"
HOST = contract.HOST
HOSTNAME = "DAY2RTX6000PRO"
PHYSICAL_GPU = contract.PHYSICAL_GPU
VRAM_CAP_GIB = contract.VRAM_CAP_GIB
TASK_MEMORY_GIB = 220
DOCKER = "/usr/bin/docker"
HOST_PORT = 18049
CONTAINER_PORT = 8000
CANONICAL_OUTPUT_ROOT = PurePosixPath(
    "/home/aday/.local/state/fleet-compute/artifacts/"
    "aeon-qwen38-flash-next-vllm-canary"
)
RUN_ROOT = PurePosixPath("/home/aday/.local/state/fleet-compute/runs")
IMAGE_ARCHIVE_ROOT = PurePosixPath(
    "/home/aday/.local/state/aeon-flash-next/runtime-images"
)
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_REQUEST_BYTES = 2 * 1024 * 1024
READINESS_TIMEOUT_SECONDS = 2400
_RUNTIME = re.compile(r"^fr-[0-9a-f]{32}$")
_JOB = re.compile(r"^fj-[0-9a-f]{32}$")
_GPU = re.compile(r"^GPU-[0-9A-Fa-f-]{32,64}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
_IMAGE_REFERENCE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,200}:[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_HF_TOKEN = re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{12,}")
MAX_DOCKER_LOG_BYTES = 512 * 1024
_active_container: tuple[str, str, str] | None = None


class CanaryWorkerError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _oci_identity(path: Path) -> tuple[str, str]:
    """Read one exact OCI manifest/config identity without extracting layers."""

    def _blob(
        archive: tarfile.TarFile,
        descriptor: Mapping[str, Any],
        *,
        maximum: int,
        label: str,
    ) -> tuple[str, bytes]:
        digest = str(descriptor.get("digest") or "")
        if not digest.startswith("sha256:") or _SHA.fullmatch(digest[7:]) is None:
            raise CanaryWorkerError(f"OCI {label} digest changed")
        member = archive.getmember(f"blobs/sha256/{digest[7:]}")
        if (
            not member.isfile()
            or member.size != descriptor.get("size")
            or not 0 < member.size <= maximum
        ):
            raise CanaryWorkerError(f"OCI {label} blob is malformed")
        handle = archive.extractfile(member)
        if handle is None:
            raise CanaryWorkerError(f"OCI {label} blob is unreadable")
        raw = handle.read(member.size + 1)
        if len(raw) != member.size or hashlib.sha256(raw).hexdigest() != digest[7:]:
            raise CanaryWorkerError(f"OCI {label} blob digest changed")
        return digest, raw

    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
            or not 0 < metadata.st_size <= 20_000_000_000
        ):
            raise CanaryWorkerError("OCI archive is not private and bounded")
        with tarfile.open(path, mode="r:") as archive:
            index_member = archive.getmember("index.json")
            if not index_member.isfile() or not 0 < index_member.size <= 1024 * 1024:
                raise CanaryWorkerError("OCI index is not one bounded file")
            index_file = archive.extractfile(index_member)
            if index_file is None:
                raise CanaryWorkerError("OCI index is unreadable")
            index = json.loads(index_file.read(index_member.size + 1))
            if not isinstance(index, Mapping):
                raise CanaryWorkerError("OCI index is not an object")
            manifests = index.get("manifests")
            if (
                index.get("schemaVersion") != 2
                or not isinstance(manifests, list)
                or len(manifests) != 1
                or not isinstance(manifests[0], Mapping)
            ):
                raise CanaryWorkerError("OCI index closure is not single-platform")
            descriptor = manifests[0]
            # ``docker image save`` may wrap a single platform image and its
            # provenance attestation in one named OCI index.  Traverse exactly
            # that bounded shape and select the sole linux/amd64 image; never
            # accept a second runnable platform or an unverified descriptor.
            if descriptor.get("mediaType") == "application/vnd.oci.image.index.v1+json":
                _index_digest, nested_raw = _blob(
                    archive, descriptor, maximum=2 * 1024 * 1024, label="nested index"
                )
                nested = json.loads(nested_raw)
                nested_manifests = nested.get("manifests") if isinstance(nested, Mapping) else None
                if (
                    nested.get("schemaVersion") != 2
                    or nested.get("mediaType") != "application/vnd.oci.image.index.v1+json"
                    or not isinstance(nested_manifests, list)
                ):
                    raise CanaryWorkerError("OCI nested index changed")
                candidates = [
                    item
                    for item in nested_manifests
                    if isinstance(item, Mapping)
                    and item.get("mediaType") == "application/vnd.oci.image.manifest.v1+json"
                    and item.get("platform") == {"architecture": "amd64", "os": "linux"}
                ]
                if len(candidates) != 1:
                    raise CanaryWorkerError("OCI nested index is not single-platform amd64")
                descriptor = candidates[0]
            manifest_digest = str(descriptor.get("digest") or "")
            if (
                descriptor.get("mediaType")
                != "application/vnd.oci.image.manifest.v1+json"
                or descriptor.get("platform") != {"architecture": "amd64", "os": "linux"}
            ):
                raise CanaryWorkerError("OCI manifest descriptor changed")
            manifest_digest, manifest_raw = _blob(
                archive, descriptor, maximum=2 * 1024 * 1024, label="manifest"
            )
            manifest = json.loads(manifest_raw)
            config = manifest.get("config") if isinstance(manifest, Mapping) else None
            config_digest = str(config.get("digest") or "") if isinstance(config, Mapping) else ""
            if (
                manifest.get("schemaVersion") != 2
                or not config_digest.startswith("sha256:")
                or _SHA.fullmatch(config_digest[7:]) is None
                or config.get("mediaType")
                != "application/vnd.oci.image.config.v1+json"
            ):
                raise CanaryWorkerError("OCI config descriptor changed")
            _config_digest, _config_raw = _blob(
                archive, config, maximum=2 * 1024 * 1024, label="config"
            )
    except (KeyError, OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise CanaryWorkerError("OCI archive identity is malformed") from exc
    return manifest_digest, config_digest[7:]


def _oci_load_digest(path: Path) -> str:
    """Return Docker's exact imported image/index ID after OCI validation."""

    _oci_identity(path)
    try:
        with tarfile.open(path, mode="r:") as archive:
            member = archive.getmember("index.json")
            handle = archive.extractfile(member)
            if handle is None or not member.isfile() or not 0 < member.size <= 1024 * 1024:
                raise CanaryWorkerError("OCI load index is unreadable")
            value = json.loads(handle.read(member.size + 1))
    except (KeyError, OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise CanaryWorkerError("OCI load identity is malformed") from exc
    manifests = value.get("manifests") if isinstance(value, Mapping) else None
    if not isinstance(manifests, list) or len(manifests) != 1 or not isinstance(manifests[0], Mapping):
        raise CanaryWorkerError("OCI load identity is ambiguous")
    digest = str(manifests[0].get("digest") or "")
    if not digest.startswith("sha256:") or _SHA.fullmatch(digest[7:]) is None:
        raise CanaryWorkerError("OCI load identity changed")
    return digest


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CanaryWorkerError(f"refusing to overwrite evidence: {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _replace_status(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _sanitize_log(value: str, request: Mapping[str, Any]) -> tuple[str, bool]:
    sanitized = _ANSI.sub("", value)
    sanitized = sanitized.replace(str(request["claim_id"]), "<fleet-claim-redacted>")
    sanitized = sanitized.replace(str(request["gpu_uuid"]), "<leased-gpu-redacted>")
    sanitized = _HF_TOKEN.sub("<hf-token-redacted>", sanitized)
    raw = sanitized.encode("utf-8", errors="replace")
    truncated = len(raw) > MAX_DOCKER_LOG_BYTES
    if truncated:
        raw = raw[-MAX_DOCKER_LOG_BYTES:]
        # A byte tail may begin inside one UTF-8 sequence.
        sanitized = raw.decode("utf-8", errors="replace")
    return sanitized, truncated


def _capture_container_logs(
    request: Mapping[str, Any], arm: str, container_id: str, evidence: Path
) -> None:
    item = _verify_container(request, arm, container_id)
    if item.get("Id") != container_id:
        raise CanaryWorkerError("refusing logs from changed container identity")
    result = _docker(
        [
            "container", "logs", "--timestamps", "--tail", "512",
            container_id,
        ],
        timeout=30,
    )
    stdout, stdout_truncated = _sanitize_log(result.stdout, request)
    stderr, stderr_truncated = _sanitize_log(result.stderr, request)
    if result.returncode != 0:
        raise CanaryWorkerError("task-owned Docker log capture failed")
    _atomic_json(
        evidence / "docker-logs.json",
        {
            "schema_version": "aeon-qwen38-flash-next-vllm-docker-logs-v1",
            "runtime_id": request["runtime_id"],
            "arm": arm,
            "container_id": container_id,
            "captured_at": _now(),
            "tail_lines_requested": 512,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        },
    )


def _private_file(path: Path, *, maximum: int) -> Mapping[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or metadata.st_size <= 0
        or metadata.st_size > maximum
    ):
        raise CanaryWorkerError("request/evidence file is not private and bounded")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanaryWorkerError("request/evidence JSON is malformed") from exc
    if not isinstance(value, Mapping):
        raise CanaryWorkerError("request/evidence JSON is not an object")
    return value


def _paths(request: Mapping[str, Any]) -> Mapping[str, Path]:
    root = Path(str(request["canonical_output_path"]))
    runtime_id = str(request["runtime_id"])
    if root != Path(CANONICAL_OUTPUT_ROOT) / runtime_id:
        raise CanaryWorkerError("canonical output path changed")
    return {
        "root": root,
        "source": root / "source",
        "output": root / "output",
        "status": root / "status.json",
        "supervisor_pid": root / "supervisor.pid",
        "request": root / "canary-request.json",
        "image": root / "assets" / "candy.JPG",
        "behavior_eval": root / "source" / "aeon/behavioral_sft/data/eval.jsonl",
    }


def validate_request(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if contract.unresolved_release_fields():
        raise CanaryWorkerError("v20 canary release identities are unresolved")
    required = {
        "schema_version", "runtime_id", "job_id", "host", "hostname",
        "physical_gpu", "gpu_uuid", "claim_id", "owner", "exclusive",
        "vram_cap_gib", "canonical_output_path", "checkpoint_path",
        "checkpoint_manifest_path", "checkpoint_manifest_sha256",
        "derived_image_digest",
        "derived_image_config_digest", "derived_image_archive_path",
        "derived_image_archive_sha256", "served_model", "runtime",
        "source_files", "asset_files",
    }
    if set(value) != required or value.get("schema_version") != SCHEMA:
        raise CanaryWorkerError("canary request fields changed")
    runtime_id = str(value["runtime_id"])
    if _RUNTIME.fullmatch(runtime_id) is None or _JOB.fullmatch(str(value["job_id"])) is None:
        raise CanaryWorkerError("Fleet runtime/job identity is malformed")
    if (
        value["host"] != HOST
        or value["hostname"] != HOSTNAME
        or value["physical_gpu"] != PHYSICAL_GPU
        or value["exclusive"] is not True
        or value["vram_cap_gib"] != VRAM_CAP_GIB
        or _GPU.fullmatch(str(value["gpu_uuid"])) is None
        or not isinstance(value["claim_id"], str)
        or not value["claim_id"]
        or value["served_model"] != contract.SERVED_MODEL
        or value["runtime"] != contract.expected_runtime()
    ):
        raise CanaryWorkerError("Fleet lease/runtime contract changed")
    for key in (
        "checkpoint_manifest_sha256", "derived_image_config_digest",
        "derived_image_archive_sha256",
    ):
        if _SHA.fullmatch(str(value[key])) is None:
            raise CanaryWorkerError(f"{key} is malformed")
    image = str(value["derived_image_digest"])
    if not image.startswith("sha256:") or _SHA.fullmatch(image[7:]) is None:
        raise CanaryWorkerError("derived image digest is malformed")
    checkpoint = Path(str(value["checkpoint_path"]))
    checkpoint_manifest = Path(str(value["checkpoint_manifest_path"]))
    archive = Path(str(value["derived_image_archive_path"]))
    try:
        metadata = checkpoint.resolve(strict=True).lstat()
    except OSError as exc:
        raise CanaryWorkerError("checkpoint is absent") from exc
    if not stat.S_ISDIR(metadata.st_mode) or checkpoint.is_symlink():
        raise CanaryWorkerError("checkpoint root is unsafe")
    try:
        if (
            checkpoint_manifest.resolve(strict=True)
            != checkpoint.resolve(strict=True) / "SHA256SUMS"
            or _sha256(checkpoint_manifest) != value["checkpoint_manifest_sha256"]
        ):
            raise CanaryWorkerError("checkpoint manifest binding changed")
    except OSError as exc:
        raise CanaryWorkerError("checkpoint manifest is absent") from exc
    try:
        archive_metadata = archive.resolve(strict=True).lstat()
        archive.resolve(strict=True).relative_to(
            Path(IMAGE_ARCHIVE_ROOT).resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise CanaryWorkerError("derived image archive is outside its canonical root") from exc
    if (
        not stat.S_ISREG(archive_metadata.st_mode)
        or archive.is_symlink()
        or archive_metadata.st_uid != os.geteuid()
        or archive_metadata.st_mode & 0o022
    ):
        raise CanaryWorkerError("derived image archive is unsafe")
    paths = _paths(value)
    for group in ("source_files", "asset_files"):
        receipts = value[group]
        if not isinstance(receipts, Mapping) or not receipts:
            raise CanaryWorkerError(f"{group} receipt is absent")
        for name, receipt in receipts.items():
            if not isinstance(name, str) or not isinstance(receipt, Mapping):
                raise CanaryWorkerError(f"{group} receipt is malformed")
            target = (paths["source"] if group == "source_files" else paths["root"] / "assets") / name
            if _SHA.fullmatch(str(receipt.get("sha256") or "")) is None or _sha256(target) != receipt["sha256"]:
                raise CanaryWorkerError(f"{group} identity changed")
    return value


def _sha256_before(path: Path, deadline: float) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            if time.monotonic() > deadline:
                raise CanaryWorkerError("checkpoint verification exceeded its bound")
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checkpoint_manifest(request: Mapping[str, Any]) -> None:
    root = Path(str(request["checkpoint_path"])).resolve(strict=True)
    manifest = Path(str(request["checkpoint_manifest_path"])).resolve(strict=True)
    metadata = manifest.lstat()
    if (
        manifest != root / "SHA256SUMS"
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= 2 * 1024 * 1024
        or _sha256(manifest) != request["checkpoint_manifest_sha256"]
    ):
        raise CanaryWorkerError("checkpoint manifest is not exact and private")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if (
            match is None
            or _SAFE_RELATIVE.fullmatch(match.group(2)) is None
            or match.group(2) == "SHA256SUMS"
            or match.group(2) in expected
        ):
            raise CanaryWorkerError("checkpoint manifest closure is malformed")
        expected[match.group(2)] = match.group(1)
    if contract.CHECKPOINT_FILE_COUNT is None:
        raise CanaryWorkerError("v20 checkpoint file count is unresolved")
    if len(expected) != contract.CHECKPOINT_FILE_COUNT:
        raise CanaryWorkerError("checkpoint manifest file count changed")
    actual: set[str] = set()
    for item in root.rglob("*"):
        item_metadata = item.lstat()
        relative = item.relative_to(root).as_posix()
        if (
            stat.S_ISLNK(item_metadata.st_mode)
            or item_metadata.st_uid != os.geteuid()
            or item_metadata.st_mode & 0o022
            or not stat.S_ISREG(item_metadata.st_mode)
        ):
            raise CanaryWorkerError("checkpoint tree contains an unsafe inode")
        if relative != "SHA256SUMS":
            actual.add(relative)
    if actual != set(expected):
        raise CanaryWorkerError("checkpoint tree closure differs from its manifest")
    deadline = time.monotonic() + 1500.0
    for name, digest in sorted(expected.items()):
        if _sha256_before(root / name, deadline) != digest:
            raise CanaryWorkerError("checkpoint file digest changed")


def load_request(path: Path, digest: str) -> Mapping[str, Any]:
    if _SHA.fullmatch(digest) is None or _sha256(path) != digest:
        raise CanaryWorkerError("request digest changed")
    return validate_request(_private_file(path, maximum=MAX_REQUEST_BYTES))


def _docker(arguments: Sequence[str], *, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [DOCKER, *arguments], stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=timeout, env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CanaryWorkerError("Docker transport failed") from exc


def _labels(request: Mapping[str, Any], arm: str) -> Mapping[str, str]:
    return {
        "aeon.fleet.profile": contract.PROFILE_ID,
        "aeon.fleet.runtime": str(request["runtime_id"]),
        "aeon.fleet.claim-sha256": hashlib.sha256(str(request["claim_id"]).encode()).hexdigest(),
        "aeon.fleet.arm": arm,
        "aeon.fleet.image": str(request["derived_image_digest"]),
    }


def _inspect(reference: str) -> Mapping[str, Any] | None:
    result = _docker(["container", "inspect", reference])
    if result.returncode != 0:
        if "No such" in result.stderr:
            return None
        raise CanaryWorkerError("exact container inspection failed")
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CanaryWorkerError("Docker inspection is malformed") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], Mapping):
        raise CanaryWorkerError("Docker inspection is ambiguous")
    return values[0]


def server_command(request: Mapping[str, Any], *, mtp_enabled: bool) -> list[str]:
    runtime = request["runtime"]
    command = [
        "vllm", "serve", "/model", "--served-model-name", str(request["served_model"]),
        "--host", "0.0.0.0", "--port", str(CONTAINER_PORT),
        "--tensor-parallel-size", str(runtime["tensor_parallel_size"]),
        "--distributed-executor-backend", str(runtime["distributed_executor_backend"]),
        "--gpu-memory-utilization", str(runtime["gpu_memory_utilization"]),
        "--kv-cache-memory-bytes", str(runtime["kv_cache_memory_bytes"]),
        "--max-model-len", str(runtime["max_model_len"]),
        "--max-num-seqs", str(runtime["max_num_seqs"]),
        "--max-num-batched-tokens", str(runtime["max_num_batched_tokens"]),
        "--kv-cache-dtype", str(runtime["kv_cache_dtype"]),
        "--quantization", str(runtime["quantization"]),
        "--moe-backend", str(runtime["moe_backend"]),
        "--enable-prefix-caching", "--enable-chunked-prefill",
        "--no-enable-flashinfer-autotune",
        "--compilation-config", json.dumps(
            {"cudagraph_capture_sizes": runtime["cudagraph_capture_sizes"]},
            sort_keys=True, separators=(",", ":"),
        ),
        "--enable-auto-tool-choice", "--tool-call-parser", str(runtime["tool_call_parser"]),
        "--reasoning-parser", str(runtime["reasoning_parser"]),
    ]
    if mtp_enabled:
        command.extend(("--speculative-config", json.dumps(runtime["speculative_config"], sort_keys=True, separators=(",", ":"))))
    return command


def docker_create_command(
    request: Mapping[str, Any], arm: str, evidence: Path, *,
    mtp_enabled: bool, image_reference: str | None = None,
) -> list[str]:
    name = f"aeon-vllm-{request['runtime_id']}-{arm}"
    claim_hash = hashlib.sha256(str(request["claim_id"]).encode()).hexdigest()
    gpu_hash = hashlib.sha256(str(request["gpu_uuid"]).encode()).hexdigest()
    source = _paths(request)["source"]
    runnable_image = image_reference or "sha256:" + str(
        request["derived_image_config_digest"]
    )
    if _IMAGE_REFERENCE.fullmatch(runnable_image) is None:
        raise CanaryWorkerError("runnable derived image reference is unsafe")
    command = [
        "container", "create", "--name", name, "--pull=never",
        "--user", f"{os.geteuid()}:{os.getegid()}",
        "--entrypoint", "python3", "--init=false", "--restart=no",
        "--memory", f"{TASK_MEMORY_GIB}g", "--memory-swap", f"{TASK_MEMORY_GIB}g",
        "--shm-size", "64g", "--pids-limit", "4096", "--ipc", "private",
        "--ulimit", "memlock=-1:-1",
        "--network", "bridge", "--publish", f"127.0.0.1:{HOST_PORT}:{CONTAINER_PORT}",
        "--gpus", f"device={request['gpu_uuid']}",
        "--env", f"CUDA_VISIBLE_DEVICES={request['gpu_uuid']}",
        "--env", f"GPU_AGENT_CLAIM_ID={request['claim_id']}",
        "--env", f"GPU_MEM_LIMIT_GB={VRAM_CAP_GIB}",
        "--env", f"AEON_RUNTIME_ID={request['runtime_id']}",
        "--env", "AEON_ENGINE_ATTESTATION=1",
        "--env", f"AEON_CANARY_ARM={arm}",
        "--env", f"AEON_MTP_ENABLED={1 if mtp_enabled else 0}",
        "--env", f"AEON_CHECKPOINT_MANIFEST_SHA256={request['checkpoint_manifest_sha256']}",
        "--env", f"AEON_DERIVED_IMAGE_CONFIG_DIGEST={request['derived_image_config_digest']}",
        "--env", f"AEON_LEASE_CLAIM_SHA256={claim_hash}",
        "--env", f"AEON_LEASE_GPU_UUID_SHA256={gpu_hash}",
        "--env", f"AEON_CANARY_HOST={HOST}",
        "--env", f"AEON_CANARY_PHYSICAL_GPU={PHYSICAL_GPU}",
        "--env", "AEON_CANARY_EXCLUSIVE=1",
        "--env", f"AEON_CHECKPOINT_REPOSITORY={contract.CHECKPOINT_REPOSITORY}",
        "--env", f"AEON_CHECKPOINT_REVISION={contract.CHECKPOINT_REVISION}",
        "--env", f"AEON_BASE_IMAGE_AMD64_DIGEST={contract.BASE_IMAGE_AMD64_DIGEST}",
        "--env", f"AEON_SERVED_MODEL={contract.SERVED_MODEL}",
        "--env", "VLLM_PLE_CPU_OFFLOAD=1",
        "--env", "VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800",
        "--env", "VLLM_USE_V2_MODEL_RUNNER=0",
        "--env", "OMP_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1",
        "--env", "OPENBLAS_NUM_THREADS=1", "--env", "OMP_WAIT_POLICY=PASSIVE",
        "--env", "TORCH_CUDA_ARCH_LIST=12.0f",
        "--mount", f"type=bind,src={request['checkpoint_path']},dst=/model,readonly",
        "--mount", f"type=bind,src={source},dst=/aeon-source,readonly",
        "--mount", f"type=bind,src={evidence},dst=/evidence",
    ]
    for key, value in _labels(request, arm).items():
        command.extend(("--label", f"{key}={value}"))
    command.extend((
        runnable_image,
        "/aeon-source/aeon/scripts/qwen_flash_next_container_supervisor.py",
        "--output", "/evidence/cuda-memory.json", "--freeze", "/evidence/freeze",
        "--context", "/evidence/runtime-context.json", "--runtime-id", str(request["runtime_id"]),
        "--arm", "tuned_mtp_on_winner" if mtp_enabled else "tuned_mtp_off",
        "--claim-sha256", claim_hash, "--gpu-uuid", str(request["gpu_uuid"]),
        "--checkpoint-tree-sha256", str(request["checkpoint_manifest_sha256"]), "--",
        *server_command(request, mtp_enabled=mtp_enabled),
    ))
    return command


def _verify_container(request: Mapping[str, Any], arm: str, container_id: str) -> Mapping[str, Any]:
    item = _inspect(container_id)
    if item is None or item.get("Id") != container_id:
        raise CanaryWorkerError("canary container disappeared")
    labels = item.get("Config", {}).get("Labels")
    if not isinstance(labels, Mapping) or any(labels.get(k) != v for k, v in _labels(request, arm).items()):
        raise CanaryWorkerError("canary container identity changed")
    env = item.get("Config", {}).get("Env")
    host_config = item.get("HostConfig")
    if (
        not isinstance(env, list)
        or not isinstance(host_config, Mapping)
        or host_config.get("CapAdd") not in (None, [])
        or host_config.get("CapDrop") not in (None, [])
        or host_config.get("SecurityOpt") not in (None, [])
    ):
        raise CanaryWorkerError("canary container environment is malformed")
    expected = {
        f"CUDA_VISIBLE_DEVICES={request['gpu_uuid']}",
        f"GPU_AGENT_CLAIM_ID={request['claim_id']}",
        f"GPU_MEM_LIMIT_GB={VRAM_CAP_GIB}",
        f"AEON_RUNTIME_ID={request['runtime_id']}",
        "AEON_ENGINE_ATTESTATION=1",
        f"AEON_CANARY_ARM={arm}",
        f"AEON_CHECKPOINT_MANIFEST_SHA256={request['checkpoint_manifest_sha256']}",
        f"AEON_DERIVED_IMAGE_CONFIG_DIGEST={request['derived_image_config_digest']}",
        f"AEON_MTP_ENABLED={1 if arm == 'mtp_on' else 0}",
        "AEON_LEASE_CLAIM_SHA256="
        + hashlib.sha256(str(request["claim_id"]).encode()).hexdigest(),
        "AEON_LEASE_GPU_UUID_SHA256="
        + hashlib.sha256(str(request["gpu_uuid"]).encode()).hexdigest(),
        f"AEON_CANARY_HOST={HOST}",
        f"AEON_CANARY_PHYSICAL_GPU={PHYSICAL_GPU}",
        "AEON_CANARY_EXCLUSIVE=1",
        "VLLM_USE_V2_MODEL_RUNNER=0",
        f"AEON_CHECKPOINT_REPOSITORY={contract.CHECKPOINT_REPOSITORY}",
        f"AEON_CHECKPOINT_REVISION={contract.CHECKPOINT_REVISION}",
        f"AEON_BASE_IMAGE_AMD64_DIGEST={contract.BASE_IMAGE_AMD64_DIGEST}",
        f"AEON_SERVED_MODEL={contract.SERVED_MODEL}",
    }
    if not expected.issubset(set(env)):
        raise CanaryWorkerError("canary lease environment changed")
    return item


def _task_cgroup(pid: int, container_id: str) -> Path:
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CanaryWorkerError("container cgroup identity is unreadable") from exc
    unified = [line.split(":", 2)[2] for line in lines if line.startswith("0::")]
    if len(unified) != 1:
        raise CanaryWorkerError("container has no exact cgroup-v2 path")
    relative = PurePosixPath(unified[0])
    if (
        not relative.is_absolute()
        or ".." in relative.parts
        or not any(container_id in part or container_id[:12] in part for part in relative.parts)
    ):
        raise CanaryWorkerError("container task cgroup identity changed")
    try:
        path = Path("/sys/fs/cgroup").joinpath(*relative.parts[1:]).resolve(strict=True)
        if int((path / "memory.max").read_text(encoding="ascii")) != TASK_MEMORY_GIB * 1024**3:
            raise CanaryWorkerError("container task memory cap changed")
        processes = {int(value) for value in (path / "cgroup.procs").read_text(encoding="ascii").split()}
        events = {}
        for line in (path / "memory.events").read_text(encoding="ascii").splitlines():
            fields = line.split()
            if len(fields) != 2 or not fields[1].isdigit() or fields[0] in events:
                raise CanaryWorkerError("task cgroup memory events are malformed")
            events[fields[0]] = int(fields[1])
    except (OSError, ValueError) as exc:
        raise CanaryWorkerError("container task cgroup attestation failed") from exc
    if pid not in processes or any(events.get(name) != 0 for name in ("max", "oom", "oom_kill")):
        raise CanaryWorkerError("task cgroup is not fresh and event-free")
    return path


def _wait_ready(request: Mapping[str, Any], arm: str, container_id: str) -> None:
    deadline = time.monotonic() + READINESS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        item = _verify_container(request, arm, container_id)
        if item is None or item.get("State", {}).get("Running") is not True:
            raise CanaryWorkerError("vLLM container exited during bounded readiness")
        try:
            qualify.semantic_ready(f"http://127.0.0.1:{HOST_PORT}", contract.SERVED_MODEL)
            return
        except (requests.RequestException, qualify.VllmQualificationError):
            time.sleep(2)
    raise CanaryWorkerError("vLLM did not become semantically ready before deadline")


def _runtime_receipt(
    request: Mapping[str, Any], evidence: Path, *, container_id: str, pid: int,
    mtp_enabled: bool,
) -> Mapping[str, Any]:
    # The immutable image emits independent GPU-runner and PLE-worker fragments.
    # The host validates both exact closures and atomically merges them; Docker
    # flags or one process's self-report alone cannot establish placement.
    gpu = _private_file(
        evidence / "engine-gpu-fragment.json", maximum=2 * 1024 * 1024
    )
    ple = _private_file(
        evidence / "engine-ple-fragment.json", maximum=2 * 1024 * 1024
    )
    common = {
        "schema_version", "runtime_id", "arm", "mtp_enabled", "container_id",
        "container_pid", "checkpoint_manifest_sha256", "lease_claim_id_sha256",
        "leased_gpu_uuid_sha256", "derived_image_config_digest", "emitter_pid",
        "emitted_after_model_load", "placement",
    }
    if set(gpu) != common | {"runtime", "runtime_provenance"} or set(ple) != common:
        raise CanaryWorkerError("derived-image fragment closure changed")
    if (
        gpu.get("schema_version") != "aeon-qwen38-flash-next-vllm-gpu-fragment-v1"
        or ple.get("schema_version") != "aeon-qwen38-flash-next-vllm-ple-fragment-v1"
    ):
        raise CanaryWorkerError("derived-image fragment schema changed")
    expected_common = {
        "runtime_id": request["runtime_id"],
        "arm": "mtp_on" if mtp_enabled else "mtp_off",
        "mtp_enabled": mtp_enabled,
        "container_id": container_id,
        "container_pid": pid,
        "checkpoint_manifest_sha256": request["checkpoint_manifest_sha256"],
        "lease_claim_id_sha256": hashlib.sha256(
            str(request["claim_id"]).encode()
        ).hexdigest(),
        "leased_gpu_uuid_sha256": hashlib.sha256(
            str(request["gpu_uuid"]).encode()
        ).hexdigest(),
        "derived_image_config_digest": request["derived_image_config_digest"],
        "emitted_after_model_load": True,
    }
    for fragment in (gpu, ple):
        if any(fragment.get(key) != value for key, value in expected_common.items()):
            raise CanaryWorkerError("derived-image fragment binding changed")
        emitter_pid = fragment.get("emitter_pid")
        if type(emitter_pid) is not int or emitter_pid <= 0:
            raise CanaryWorkerError("derived-image fragment emitter PID is malformed")
    runtime = gpu.get("runtime")
    if not isinstance(runtime, Mapping):
        raise CanaryWorkerError("derived-image runtime receipt is absent")
    projected_runtime = dict(runtime)
    if mtp_enabled:
        if runtime.get("speculative_config") != contract.EXPECTED_RUNTIME["speculative_config"]:
            raise CanaryWorkerError("MTP-on engine receipt changed")
    else:
        if runtime.get("speculative_config") is not None:
            raise CanaryWorkerError("MTP-off control unexpectedly enabled speculation")
        projected_runtime["speculative_config"] = contract.EXPECTED_RUNTIME["speculative_config"]
    failures = contract.validate_runtime_receipt(projected_runtime)
    provenance = gpu.get("runtime_provenance")
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != set(runtime)
        or any(
            value not in {"engine_native", "verified_env", "verified_docker"}
            for value in provenance.values()
        )
    ):
        raise CanaryWorkerError("runtime field provenance is incomplete")
    gpu_placement = gpu.get("placement")
    ple_placement = ple.get("placement")
    if (
        failures
        or not isinstance(gpu_placement, Mapping)
        or not isinstance(ple_placement, Mapping)
    ):
        raise CanaryWorkerError("derived-image runtime/placement attestation failed")
    gpu_keys = {
        "categories", "ple_placeholder_layer_count", "ple_placeholder_names_sha256",
        "unexpected_cpu_parameters", "unexpected_meta_parameters",
        "unexpected_non_cuda_parameters", "unexpected_cpu_persistent_buffers",
        "unexpected_meta_persistent_buffers", "unexpected_non_cuda_persistent_buffers",
    }
    categories = gpu_placement.get("categories")
    category_keys = {
        "parameter_references", "persistent_buffer_references", "numel_references",
        "devices", "names_sha256",
    }
    if (
        set(gpu_placement) != gpu_keys
        or not isinstance(categories, Mapping)
        or set(categories) != {"transformer", "mtp", "lm_head", "vision"}
        or type(gpu_placement.get("ple_placeholder_layer_count")) is not int
        or gpu_placement["ple_placeholder_layer_count"] <= 0
        or _SHA.fullmatch(str(gpu_placement.get("ple_placeholder_names_sha256") or "")) is None
        or any(gpu_placement.get(key) != [] for key in gpu_keys if key.startswith("unexpected_"))
    ):
        raise CanaryWorkerError("GPU model placement evidence is malformed")
    for name, category in categories.items():
        if not isinstance(category, Mapping) or set(category) != category_keys:
            raise CanaryWorkerError("GPU category placement closure changed")
        parameters = category.get("parameter_references")
        buffers = category.get("persistent_buffer_references")
        numel = category.get("numel_references")
        allow_empty_mtp = name == "mtp" and not mtp_enabled
        if (
            type(parameters) is not int or parameters < (0 if allow_empty_mtp else 1)
            or type(buffers) is not int or buffers < 0
            or type(numel) is not int or numel < (0 if allow_empty_mtp else 1)
            or category.get("devices") != ([] if allow_empty_mtp and parameters == 0 and buffers == 0 else ["cuda:0"])
            or _SHA.fullmatch(str(category.get("names_sha256") or "")) is None
        ):
            raise CanaryWorkerError("GPU category is not entirely CUDA resident")
    ple_keys = {
        "ple_layer_count", "ple_layer_names_sha256", "parameter_references",
        "persistent_buffer_references", "numel_references", "devices",
        "bf16_table_references", "bf16_table_numel", "bf16_table_names_sha256",
        "fp8_table_references", "fp8_table_numel", "fp8_table_names_sha256",
        "scale_references", "scale_names_sha256", "pinned_h2d_buffer_count",
        "pinned_h2d_bytes", "pinned_h2d_devices",
        "registered_cuda_output_target_count", "registered_cuda_output_target_devices",
        "non_ple_retained_modules", "unexpected_non_cpu_model_tensors",
        "unexpected_unpinned_h2d_buffers", "unexpected_non_cuda_output_targets",
    }
    positive_ple = (
        "ple_layer_count", "parameter_references", "numel_references",
        "bf16_table_references", "bf16_table_numel",
        "pinned_h2d_buffer_count", "pinned_h2d_bytes",
        "registered_cuda_output_target_count",
    )
    if (
        set(ple_placement) != ple_keys
        or any(type(ple_placement.get(key)) is not int or ple_placement[key] <= 0 for key in positive_ple)
        or type(ple_placement.get("persistent_buffer_references")) is not int
        or ple_placement["persistent_buffer_references"] < 0
        or ple_placement.get("fp8_table_references") != 0
        or ple_placement.get("fp8_table_numel") != 0
        or ple_placement.get("scale_references") != 0
        or ple_placement.get("devices") != ["cpu"]
        or ple_placement.get("pinned_h2d_devices") != ["cpu"]
        or ple_placement.get("registered_cuda_output_target_devices") != ["cuda:0"]
        or any(
            ple_placement.get(key) != []
            for key in (
                "non_ple_retained_modules", "unexpected_non_cpu_model_tensors",
                "unexpected_unpinned_h2d_buffers", "unexpected_non_cuda_output_targets",
            )
        )
        or any(
            _SHA.fullmatch(str(ple_placement.get(key) or "")) is None
            for key in (
                "ple_layer_names_sha256", "bf16_table_names_sha256",
                "fp8_table_names_sha256", "scale_names_sha256"
            )
        )
    ):
        raise CanaryWorkerError("PLE worker placement/H2D evidence is malformed")
    placement = {
        "transformer_weights": "cuda",
        "mtp_weights": (
            "cuda"
            if categories["mtp"]["parameter_references"] > 0
            or categories["mtp"]["persistent_buffer_references"] > 0
            else "not_loaded_mtp_off_control"
        ),
        "lm_head": "cuda",
        "vision_weights": "cuda", "ple_table": "cpu_worker_pinned_h2d", "other_cpu_model_components": [],
    }
    merged = {
        "schema_version": "aeon-qwen38-flash-next-vllm-engine-runtime-v1",
        "runtime": dict(runtime), "runtime_provenance": dict(provenance),
        "placement": placement,
        "gpu_fragment_sha256": _sha256(evidence / "engine-gpu-fragment.json"),
        "ple_fragment_sha256": _sha256(evidence / "engine-ple-fragment.json"),
        "gpu_placement": dict(gpu_placement), "ple_placement": dict(ple_placement),
        **expected_common,
    }
    _atomic_json(evidence / "engine-runtime.json", merged)
    return merged


def _remove_container(request: Mapping[str, Any], arm: str, container_id: str) -> None:
    item = _verify_container(request, arm, container_id)
    if item.get("State", {}).get("Running") is True:
        stopped = _docker(["container", "stop", "--time", "30", container_id], timeout=60)
        if stopped.returncode != 0:
            raise CanaryWorkerError("task-owned vLLM container did not stop")
    removed = _docker(["container", "rm", container_id])
    if removed.returncode != 0 or _inspect(container_id) is not None:
        raise CanaryWorkerError("task-owned vLLM container absence is unproven")


def _validate_cuda_attestation(
    value: Mapping[str, Any], request: Mapping[str, Any], *, container_id: str,
    pid: int, cgroup: Path, mtp_enabled: bool,
) -> None:
    expected_arm = "tuned_mtp_on_winner" if mtp_enabled else "tuned_mtp_off"
    try:
        started = datetime.fromisoformat(str(value.get("started_at") or ""))
        completed = datetime.fromisoformat(str(value.get("completed_at") or ""))
        max_used_at = datetime.fromisoformat(str(value.get("max_used_at") or ""))
        min_reserve_at = datetime.fromisoformat(
            str(value.get("min_reserve_at") or "")
        )
    except ValueError as exc:
        raise CanaryWorkerError("CUDA sampler extrema timestamps are malformed") from exc
    if (
        value.get("schema_version") != "aeon-qwen38-flash-next-cuda-memory-v1"
        or value.get("complete") is not True
        or value.get("runtime_id") != request["runtime_id"]
        or value.get("arm") != expected_arm
        or value.get("lease_claim_id_sha256")
        != hashlib.sha256(str(request["claim_id"]).encode()).hexdigest()
        or value.get("leased_gpu_uuid_sha256")
        != hashlib.sha256(str(request["gpu_uuid"]).encode()).hexdigest()
        or value.get("container_id") != container_id
        or value.get("container_pid") != pid
        or value.get("cgroup_path") != str(cgroup)
        or value.get("reserve_passed") is not True
        or value.get("sample_interval_seconds") != 0.1
        or not started <= max_used_at <= completed
        or not started <= min_reserve_at <= completed
    ):
        raise CanaryWorkerError("CUDA sampler identity/completeness changed")


def _run_arm(request: Mapping[str, Any], *, mtp_enabled: bool) -> Mapping[str, Any]:
    global _active_container
    arm = "mtp_on" if mtp_enabled else "mtp_off"
    paths = _paths(request)
    evidence = paths["root"] / arm
    evidence.mkdir(mode=0o700)
    # Docker/containerd can retain image metadata after the runnable manifest
    # blob has vanished.  The first arm's exact container cleanup exposed that
    # state before the second arm.  Rehydrate the already hash/OCI-validated
    # archive immediately before every create, not only once per job spawn.
    image_reference = _ensure_image_loaded(request)
    command = docker_create_command(
        request, arm, evidence, mtp_enabled=mtp_enabled,
        image_reference=image_reference,
    )
    name = f"aeon-vllm-{request['runtime_id']}-{arm}"
    if _inspect(name) is not None:
        raise CanaryWorkerError("canary container name already exists")
    created = _docker(command, timeout=120)
    container_id = created.stdout.strip()
    if created.returncode != 0 or _CONTAINER.fullmatch(container_id) is None:
        exact_absence = _inspect(name) is None
        _record_create_failure(
            request, arm, evidence, created, exact_absence=exact_absence
        )
        if exact_absence:
            raise CanaryWorkerError("Docker create failed with exact absence")
        raise CanaryWorkerError("Docker create result is ambiguous")
    _active_container = (arm, container_id, str(request["runtime_id"]))
    try:
        _verify_container(request, arm, container_id)
        started = _docker(["container", "start", container_id], timeout=120)
        if started.returncode != 0 or started.stdout.strip() != container_id:
            raise CanaryWorkerError("task-owned vLLM container failed to start")
        item = _verify_container(request, arm, container_id)
        pid = item.get("State", {}).get("Pid")
        if type(pid) is not int or pid <= 1:
            raise CanaryWorkerError("container PID identity is malformed")
        cgroup = _task_cgroup(pid, container_id)
        _atomic_json(evidence / "runtime-context.json", {
            "container_id": container_id, "container_pid": pid,
            "cgroup_path": str(cgroup),
            "container_pid_in_cgroup": True,
        })
        _wait_ready(request, arm, container_id)
        engine = _runtime_receipt(
            request, evidence, container_id=container_id, pid=pid,
            mtp_enabled=mtp_enabled,
        )
        report = qualify.probe_arm(
            f"http://127.0.0.1:{HOST_PORT}", contract.SERVED_MODEL,
            paths["image"], paths["behavior_eval"], mtp_enabled=mtp_enabled,
            benchmark_callback=lambda value: _atomic_json(
                evidence / "benchmark-report.json", value
            ),
        )
        _atomic_json(evidence / "arm-report.json", report)
        _task_cgroup(pid, container_id)
        (evidence / "freeze").write_text("freeze\n", encoding="ascii")
        time.sleep(1)
        cuda = _private_file(evidence / "cuda-memory.json", maximum=2 * 1024 * 1024)
        _validate_cuda_attestation(
            cuda, request, container_id=container_id, pid=pid, cgroup=cgroup,
            mtp_enabled=mtp_enabled,
        )
        return {"report": report, "engine": engine, "cuda": cuda}
    finally:
        active_exception = sys.exc_info()[0] is not None
        log_error: BaseException | None = None
        try:
            _capture_container_logs(request, arm, container_id, evidence)
        except BaseException as exc:
            log_error = exc
            try:
                _atomic_json(
                    evidence / "docker-log-capture-failure.json",
                    {
                        "schema_version": (
                            "aeon-qwen38-flash-next-vllm-docker-log-failure-v1"
                        ),
                        "runtime_id": request["runtime_id"],
                        "arm": arm,
                        "container_id": container_id,
                        "failure_type": type(exc).__name__,
                        "failure": str(exc)[:400],
                    },
                )
            except BaseException:
                pass
        try:
            _remove_container(request, arm, container_id)
        finally:
            _active_container = None
        if log_error is not None and not active_exception:
            raise log_error


def _qualification_receipt(request: Mapping[str, Any]) -> Mapping[str, Any]:
    off = _run_arm(request, mtp_enabled=False)
    on = _run_arm(request, mtp_enabled=True)
    comparison = qualify.compare(off["report"], on["report"])
    cuda = on["cuda"]
    total = float(cuda.get("total_bytes", 0))
    samples = float(cuda.get("sample_count", 0))
    interval = float(cuda.get("sample_interval_seconds", 0))
    started = str(cuda.get("started_at", ""))
    completed = str(cuda.get("completed_at", ""))
    # Sampler enforces density before its PID1 exits; preserve the exact result
    # and independently project cap/reserve/gap for the promotion contract.
    peak = float(cuda.get("max_used_bytes", 0)) / 1024**3
    reserve = float(cuda.get("min_reserve_bytes", 0)) / 1024**3
    try:
        duration = (
            datetime.fromisoformat(completed) - datetime.fromisoformat(started)
        ).total_seconds()
        expected_samples = duration / interval + 1
        density = samples / expected_samples
    except (ValueError, ZeroDivisionError):
        density = 0.0
    comparison.update({
        "runtime": on["engine"]["runtime"],
        "placement": on["engine"]["placement"],
        "cuda_sampling": {
            "peak_used_gib": peak, "minimum_reserve_gib": reserve,
            "cadence_density": density if samples >= 10 else 0.0,
            "maximum_gap_seconds": cuda.get("max_sample_gap_seconds"),
            "total_gib": total / 1024**3,
        },
        "capacity": {
            "max_model_len": on["engine"]["runtime"]["max_model_len"],
            "kv_cache_memory_bytes": on["engine"]["runtime"]["kv_cache_memory_bytes"],
            # vLLM refuses semantic readiness when its initialized KV cache
            # cannot serve the configured maximum model length.
            "startup_max_model_len_validated": True,
        },
        "process_identity_verified": True,
        "semantic_readiness_verified": True,
    })
    failures = contract.validate_qualification_receipt(comparison)
    if failures:
        raise CanaryWorkerError("qualification gates failed: " + "; ".join(failures))
    return comparison


def _manifest(output: Path) -> str:
    lines: list[str] = []
    total = 0
    for item in sorted(output.rglob("*")):
        metadata = item.lstat()
        if stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise CanaryWorkerError("output contains an unsafe inode")
        if stat.S_ISREG(metadata.st_mode):
            relative = item.relative_to(output).as_posix()
            if relative == "MANIFEST.sha256":
                continue
            total += metadata.st_size
            lines.append(f"{_sha256(item)}  {relative}")
    if total > MAX_OUTPUT_BYTES:
        raise CanaryWorkerError("canary evidence exceeds output bound")
    manifest = output / "MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
    return _sha256(manifest)


def _supervise(request_path: Path, digest: str) -> int:
    request = load_request(request_path, digest)
    paths = _paths(request)
    output = paths["output"]
    output.mkdir(mode=0o700)
    _replace_status(paths["status"], {"schema_version": STATUS_SCHEMA, "state": "running", "pid": os.getpid(), "started_at": _now()})

    def stop(signum: int, _frame: Any) -> None:
        if _active_container is not None:
            arm, container_id, runtime_id = _active_container
            if runtime_id == request["runtime_id"]:
                _remove_container(request, arm, container_id)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        receipt = _qualification_receipt(request)
        _atomic_json(output / "qualification.json", receipt)
        manifest = _manifest(output)
        _replace_status(paths["status"], {"schema_version": STATUS_SCHEMA, "state": "completed", "pid": None, "completed_at": _now(), "manifest_sha256": manifest})
        return 0
    except BaseException as exc:
        _replace_status(paths["status"], {"schema_version": STATUS_SCHEMA, "state": "failed", "pid": None, "completed_at": _now(), "failure": f"{type(exc).__name__}: {str(exc)[:400]}"})
        return 1


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    return True


def _supervisor_matches(pid: int, request_path: Path, digest: str) -> bool:
    if not _pid_alive(pid):
        return False
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    fields = [item.decode("utf-8") for item in raw.split(b"\0") if item]
    expected_tail = ["supervise", str(request_path), digest]
    return (
        len(fields) >= 4
        and Path(fields[1]).resolve(strict=True) == Path(__file__).resolve(strict=True)
        and fields[-3:] == expected_tail
    )


def _ensure_image_loaded(request: Mapping[str, Any]) -> str:
    # containerd metadata can survive after one or more referenced blobs have
    # vanished.  Therefore image-inspect is not content readiness: rehydrate
    # the already hash/OCI-validated single-image archive immediately before
    # every spawn, then prove the runnable manifest identity.
    loaded = _docker(
        ["image", "load", "--input", str(request["derived_image_archive_path"])],
        timeout=1800,
    )
    loaded_lines = [
        line for line in loaded.stdout.splitlines()
        if line.startswith("Loaded image: ") or line.startswith("Loaded image ID: ")
    ]
    if loaded.returncode != 0 or len(loaded_lines) != 1:
        raise CanaryWorkerError("exact derived image archive load was ambiguous")
    prefix, loaded_reference = loaded_lines[0].split(": ", 1)
    expected_config_id = "sha256:" + str(request["derived_image_config_digest"])
    expected_load_id = _oci_load_digest(Path(str(request["derived_image_archive_path"])))
    if prefix == "Loaded image ID":
        if loaded_reference not in {expected_config_id, expected_load_id}:
            raise CanaryWorkerError("loaded derived image config identity changed")
        inspect_reference = loaded_reference
    else:
        if _IMAGE_REFERENCE.fullmatch(loaded_reference) is None:
            raise CanaryWorkerError("loaded derived image reference is unsafe")
        inspect_reference = loaded_reference
    image = _docker(["image", "inspect", inspect_reference])
    if image.returncode != 0:
        raise CanaryWorkerError("loaded derived image identity is absent")
    try:
        values = json.loads(image.stdout)
    except json.JSONDecodeError as exc:
        raise CanaryWorkerError("derived image inspection is malformed") from exc
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], Mapping)
        or values[0].get("Id") not in {expected_config_id, expected_load_id}
        or (
            prefix == "Loaded image"
            and loaded_reference not in (values[0].get("RepoTags") or [])
        )
    ):
        raise CanaryWorkerError("derived OCI manifest image identity changed")
    return inspect_reference


def _record_create_failure(
    request: Mapping[str, Any], arm: str, evidence: Path,
    result: subprocess.CompletedProcess[str], *, exact_absence: bool,
) -> None:
    stdout, stdout_truncated = _sanitize_log(result.stdout, request)
    stderr, stderr_truncated = _sanitize_log(result.stderr, request)
    _atomic_json(
        evidence / "docker-create-failure.json",
        {
            "schema_version": "aeon-qwen38-flash-next-vllm-create-failure-v1",
            "runtime_id": request["runtime_id"],
            "arm": arm,
            "derived_image_digest": request["derived_image_digest"],
            "returncode": result.returncode,
            "exact_container_absence": exact_absence,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "captured_at": _now(),
        },
    )


def action_preflight(request_path: Path, digest: str) -> Mapping[str, Any]:
    request = load_request(request_path, digest)
    _verify_checkpoint_manifest(request)
    if _sha256(Path(str(request["derived_image_archive_path"]))) != request["derived_image_archive_sha256"]:
        raise CanaryWorkerError("pinned derived image archive changed")
    archive_manifest, archive_config = _oci_identity(
        Path(str(request["derived_image_archive_path"]))
    )
    if (
        archive_manifest != request["derived_image_digest"]
        or archive_config != request["derived_image_config_digest"]
    ):
        raise CanaryWorkerError("derived image/archive identity binding changed")
    _ensure_image_loaded(request)
    return {
        "request_sha256": digest,
        "checkpoint_manifest_sha256": request["checkpoint_manifest_sha256"],
        "derived_image_digest": request["derived_image_digest"],
        "derived_image_archive_sha256": request["derived_image_archive_sha256"],
        "vram_cap_gib": VRAM_CAP_GIB,
    }


def action_spawn(request_path: Path, digest: str) -> Mapping[str, Any]:
    request = load_request(request_path, digest)
    paths = _paths(request)
    action_preflight(request_path, digest)
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "supervise", str(request_path), digest],
        stdin=subprocess.DEVNULL, stdout=(paths["root"] / "supervisor.stdout").open("xb"),
        stderr=(paths["root"] / "supervisor.stderr").open("xb"), start_new_session=True,
        close_fds=True, env={"HOME": "/home/aday", "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "PYTHONPATH": str(paths["source"]), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    paths["supervisor_pid"].write_text(f"{process.pid}\n", encoding="ascii")
    return {"pid": process.pid}


def action_status(request_path: Path, digest: str) -> Mapping[str, Any]:
    request = load_request(request_path, digest)
    paths = _paths(request)
    if not paths["status"].exists():
        return {"state": "absent", "pid": None}
    status = dict(_private_file(paths["status"], maximum=64 * 1024))
    pid = status.get("pid")
    if status.get("state") == "running" and (
        type(pid) is not int or not _supervisor_matches(pid, request_path, digest)
    ):
        return {"state": "failed", "pid": None, "failure": "supervisor identity vanished"}
    return status


def action_stop(request_path: Path, digest: str) -> Mapping[str, Any]:
    status = action_status(request_path, digest)
    pid = status.get("pid")
    if status.get("state") == "running" and type(pid) is int:
        if not _supervisor_matches(pid, request_path, digest):
            raise CanaryWorkerError("refusing to signal changed supervisor identity")
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 90
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        if _pid_alive(pid):
            return {"process_absent": False}
    return {"process_absent": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "spawn", "status", "stop", "supervise"))
    parser.add_argument("request", type=Path)
    parser.add_argument("digest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "supervise":
        return _supervise(args.request, args.digest)
    actions = {"preflight": action_preflight, "spawn": action_spawn, "status": action_status, "stop": action_stop}
    try:
        result = actions[args.action](args.request, args.digest)
        print(json.dumps({"ok": True, "result": result}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "detail": f"{type(exc).__name__}: {str(exc)[:400]}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
