#!/usr/bin/env python3
"""Worker-side lifecycle for one Fleet-leased Flash-Next service container.

This program is copied into one Fleet run directory and is invoked only through
BatchMode SSH.  It never contacts the coordinator, chooses a GPU, scans Docker,
pulls an image, or removes shared cache/image data.  Authority is the exact
private request plus the exact container receipt created for that request.
"""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request


SCHEMA = "aeon-qwen38-flash-next-remote-worker-v1"
PROFILE_ID = "aeon-qwen38-flash-next-179"
HOST = "192.168.0.179"
HOSTNAME = "DAY2XRTX6000-2"
RUN_ROOT = Path("/home/aday/.local/state/fleet-compute/runs")
CACHE_ROOT = Path(
    "/home/aday/.local/state/fleet-compute/cache/aeon-qwen38-flash-next"
)
LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
DOCKER = "/home/aday/bin/docker"
CONTAINER_PORT = 30000
VRAM_BUDGET_GB = 88.0
MIN_PHYSICAL_VRAM_GB = 94.0
MAX_TASK_MEMORY_BYTES = 200 * 1024**3
MAX_RELEASE_BYTES = 150 * 1024**3
MAX_RELEASE_INODES = 10_000
MAX_MODEL_BYTES = 150 * 1024**3
MAX_MODEL_INODES = 10_000
MAX_IMAGE_BYTES = 64 * 1024**3
SERVED_ALIAS = "Qwen3.8-27B-ARA-NVFP4-MTP"
DISPLAY_NAME = "Aeon Qwen3.8-Flash-Next 125B-A6B NVFP4+MTP"
ARTIFACT_NAME = "Aeon-Qwen3.8-Flash-Next-NVFP4-MTP"
MODEL_ARCHITECTURE = "qwen4_exp"
SGLANG_SOURCE_STACK_SHA256 = (
    "f9087c7d56219f49fb575c8b1008e923ddeea1ea878e46b20f8e5585317136ed"
)
SGLANG_IMAGE_DIGEST = (
    "sha256:067473b3134f933ebc04a3c4774b16bd400a15afcaf9eec8230c57205f7e7719"
)
SGLANG_IMAGE_CONFIG_DIGEST = (
    "sha256:ac23f9a937f1e82cc1bade15079a568a73e68b1cecbe4d4f326ba330418e0a36"
)
SGLANG_IMAGE_ID = SGLANG_IMAGE_DIGEST
SGLANG_IMAGE_ARCHIVE_SHA256 = (
    "f25ab76b3f48b55e1632e020e9fc4709766bae447c42564d2058f16a4bc13374"
)
SGLANG_IMAGE_REFERENCE = (
    "aeon/sglang:qwen38-flash-next-sm120-headroom-a6c61-424e@"
    + SGLANG_IMAGE_DIGEST
)
SGLANG_IMAGE_REPO_DIGEST = "aeon/sglang@" + SGLANG_IMAGE_DIGEST
REQUIRED_MOE_RUNNER_BACKEND = "flashinfer_cutlass"
QUALIFIED_MOE_RUNNER_BACKENDS = frozenset({REQUIRED_MOE_RUNNER_BACKEND})
EXPECTED_IMAGE_LABELS = {
    "com.bc-aeon.qwen38-flash-next.artifact": ARTIFACT_NAME,
    "com.bc-aeon.qwen38-flash-next.display-name": DISPLAY_NAME,
    "com.bc-aeon.qwen38-flash-next.model-architecture": MODEL_ARCHITECTURE,
    "com.bc-aeon.qwen38-flash-next.base-image": (
        "lmsysorg/sglang:qwen38flashnext@"
        "sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae"
    ),
    "com.bc-aeon.qwen38-flash-next.base-sglang-commit": (
        "d91c3682b0b429e4c70df63cd57f819588ce29b0"
    ),
    "com.bc-aeon.qwen38-flash-next.qwen-overlay-pr": (
        "Qiaolin-Yu/sglang-qwen-next#38"
    ),
    "com.bc-aeon.qwen38-flash-next.qwen-overlay-commits": (
        "3ea3a37a1,12070370f"
    ),
    "com.bc-aeon.qwen38-flash-next.qwen-reference-commit": (
        "73a255206f916366c8d26d4022f82ddfb0ab558d"
    ),
    "com.bc-aeon.qwen38-flash-next.sm120-fix-commit": (
        "dac5523d1e5d2f4297fec40ef02fc76fb0f662d1"
    ),
    "com.bc-aeon.qwen38-flash-next.sm120-patch-sha256": (
        "eba9b1b2c07f6bdfe42502ffc50667f7e1e5467dc1ee96f0a8e791562e1c9679"
    ),
    "com.bc-aeon.qwen38-flash-next.sm120-fp4-backend-selection-commit": (
        "3836cba9eed2cc0db093e58ca839215609a44c31"
    ),
    "com.bc-aeon.qwen38-flash-next.fused-shared-expert-commit": (
        "cdb7ac8f4740f0baf5d01d673fd0fb671a14ebdf"
    ),
    "com.bc-aeon.qwen38-flash-next.fused-shared-expert-patch-sha256": (
        "9c3d91412bd3599ccfb5a8879448423fbc34cc24659593933dabe22858ce7338"
    ),
    "com.bc-aeon.qwen38-flash-next.mtp-shared-expert-commit": (
        "7db597910dab20741770862d328c1399be0e6ab8"
    ),
    "com.bc-aeon.qwen38-flash-next.mtp-shared-expert-patch-sha256": (
        "e9f26827b1c0da319c1116caea575b89a794c983ed35671331d421d40137b7fb"
    ),
    "com.bc-aeon.qwen38-flash-next.cutlass-scale-headroom-patch-sha256": (
        "a6c61ef9eaa1153551506b26aca7627f7ecc98851f6cd7e7038cd6d0a25b5c6a"
    ),
    "com.bc-aeon.qwen38-flash-next.mtp-share-before-pool-issue": "36452",
    "com.bc-aeon.qwen38-flash-next.mtp-share-before-pool-patch-sha256": (
        "424eb761834646089437f7e2d16694ab06f03e102f045da07f4a35aa3c83b607"
    ),
    "com.bc-aeon.qwen38-flash-next.source-stack-sha256": (
        SGLANG_SOURCE_STACK_SHA256
    ),
}
MAX_JSON_BYTES = 2 * 1024 * 1024
RECEIPT_NAME = "flash-next-remote-container.json"
WORKER_NAME = "qwen_flash_next_remote_service_worker.py"

_RUNTIME = re.compile(r"^fr-[0-9a-f]{32}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(r"^GPU-[0-9A-Fa-f-]{16,64}$")
_CLAIM = re.compile(r"^gc-[A-Za-z0-9_.:-]{8,128}$")
_OWNER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class RemoteWorkerError(RuntimeError):
    """The exact remote lifecycle or artifact identity failed closed."""


def _sha256(path: Path, *, maximum: int | None = None) -> str:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
        or (maximum is not None and not 0 < metadata.st_size <= maximum)
    ):
        raise RemoteWorkerError(f"unsafe checksum input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _private_json(path: Path, *, maximum: int = MAX_JSON_BYTES) -> dict[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= maximum
    ):
        raise RemoteWorkerError(f"private receipt is unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteWorkerError(f"private receipt is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise RemoteWorkerError(f"private receipt is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or parent.st_mode & 0o077
    ):
        raise RemoteWorkerError("worker receipt parent is unsafe")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        payload = (
            json.dumps(
                dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RemoteWorkerError("worker receipt write was incomplete")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise RemoteWorkerError(f"{label} is not a SHA-256 digest")
    return value


def _require_cache_path(value: Any, label: str) -> Path:
    path = Path(str(value))
    try:
        relative = PurePosixPath(str(path)).relative_to(PurePosixPath(str(CACHE_ROOT)))
    except ValueError as exc:
        raise RemoteWorkerError(f"{label} escaped Fleet cache") from exc
    if (
        not relative.parts
        or ".." in relative.parts
        or str(path) != str(PurePosixPath(str(path)))
        or path.resolve(strict=True) != path
    ):
        raise RemoteWorkerError(f"{label} is not a normalized Fleet cache path")
    return path


def _safe_run_dir(value: Any) -> Path:
    path = Path(str(value))
    if path.parent != RUN_ROOT or _RUNTIME.fullmatch(path.name) is None:
        raise RemoteWorkerError("worker run directory is outside its Fleet root")
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise RemoteWorkerError("worker run directory is unsafe")
    return path


def _request(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require_sha(expected_sha256, "request digest")
    request = _private_json(path)
    if _sha256(path, maximum=MAX_JSON_BYTES) != expected_sha256:
        raise RemoteWorkerError("worker request digest changed")
    expected_fields = {
        "schema_version",
        "contract_sha256",
        "runtime_id",
        "profile_id",
        "host",
        "hostname",
        "run_dir",
        "source_path",
        "source_sha256",
        "binding_sha256",
        "release_manifest_sha256",
        "release_tree_sha256",
        "checkpoint_tree_sha256",
        "materialized_checkpoint_tree_sha256",
        "ple_materialization_manifest_sha256",
        "ple_materializer_sha256",
        "materialization_receipt_sha256",
        "runtime_config_sha256",
        "qualification_sha256",
        "qualification_mtp_off_sha256",
        "qualification_mtp_on_sha256",
        "release",
        "materialized_model",
        "image",
        "lease",
        "container",
    }
    if set(request) != expected_fields or request.get("schema_version") != SCHEMA:
        raise RemoteWorkerError("worker request fields changed")
    run_dir = _safe_run_dir(request.get("run_dir"))
    if (
        path != run_dir / "flash-next-remote-request.json"
        or request.get("runtime_id") != run_dir.name
        or request.get("profile_id") != PROFILE_ID
        or request.get("host") != HOST
        or request.get("hostname") != HOSTNAME
        or socket.gethostname() != HOSTNAME
        or request.get("source_path") != str(run_dir / "source" / WORKER_NAME)
        or _sha256(Path(str(request.get("source_path"))), maximum=2 * 1024 * 1024)
        != _require_sha(request.get("source_sha256"), "worker source")
    ):
        raise RemoteWorkerError("worker source/host/runtime binding changed")
    for field in (
        "contract_sha256",
        "binding_sha256",
        "release_manifest_sha256",
        "release_tree_sha256",
        "checkpoint_tree_sha256",
        "materialized_checkpoint_tree_sha256",
        "ple_materialization_manifest_sha256",
        "ple_materializer_sha256",
        "materialization_receipt_sha256",
        "runtime_config_sha256",
        "qualification_sha256",
        "qualification_mtp_off_sha256",
        "qualification_mtp_on_sha256",
    ):
        _require_sha(request.get(field), field)
    if request.get("materialized_checkpoint_tree_sha256") != request.get(
        "checkpoint_tree_sha256"
    ):
        raise RemoteWorkerError("materialized model is not the qualified checkpoint")
    _validate_lease(request)
    _validate_container_request(request)
    return request


def _validate_lease(request: Mapping[str, Any]) -> None:
    lease = request.get("lease")
    if not isinstance(lease, Mapping) or set(lease) != {
        "claim_id",
        "owner",
        "physical_gpu",
        "gpu_uuid",
        "vram_budget_gb",
        "exclusive",
        "model",
        "memory_total_mib",
    }:
        raise RemoteWorkerError("worker lease fields changed")
    model = str(lease.get("model") or "").casefold()
    memory = lease.get("memory_total_mib")
    if (
        not isinstance(lease.get("claim_id"), str)
        or _CLAIM.fullmatch(str(lease["claim_id"])) is None
        or not isinstance(lease.get("owner"), str)
        or _OWNER.fullmatch(str(lease["owner"])) is None
        or lease.get("physical_gpu") not in {0, 1}
        or not isinstance(lease.get("gpu_uuid"), str)
        or _UUID.fullmatch(str(lease["gpu_uuid"])) is None
        or lease.get("vram_budget_gb") != VRAM_BUDGET_GB
        or lease.get("exclusive") is not True
        or type(memory) is not int
        or memory < int(MIN_PHYSICAL_VRAM_GB * 1024)
        or memory / 1024 - VRAM_BUDGET_GB < 6
        or "rtx pro 6000" not in model
        or "blackwell" not in model
    ):
        raise RemoteWorkerError("worker lease is not an exclusive 88 GiB PRO 6000 lease")


def _validate_container_request(request: Mapping[str, Any]) -> None:
    container = request.get("container")
    lease = request["lease"]
    runtime_id = request["runtime_id"]
    if not isinstance(container, Mapping) or set(container) != {
        "name",
        "host_port",
        "container_port",
        "image_reference",
        "image_id",
        "task_memory_bytes",
        "shm_bytes",
        "command",
        "command_sha256",
        "environment",
        "environment_sha256",
    }:
        raise RemoteWorkerError("worker container request fields changed")
    command = container.get("command")
    environment = container.get("environment")
    expected_env = {
        "CUDA_VISIBLE_DEVICES": lease["gpu_uuid"],
        "GPU_AGENT_CLAIM_ID": lease["claim_id"],
        "GPU_MEM_LIMIT_GB": "88",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "USE_TF": "0",
        "USE_FLAX": "0",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
    }
    task_memory = container.get("task_memory_bytes")
    port = container.get("host_port")
    if (
        container.get("name") != f"aeon-qwen38-flash-next-179-{runtime_id}"
        or type(port) is not int
        or port != 18140 + int(lease["physical_gpu"])
        or container.get("container_port") != CONTAINER_PORT
        or container.get("image_id") != SGLANG_IMAGE_ID
        or request.get("image", {}).get("digest_sha256")
        != SGLANG_IMAGE_ID.removeprefix("sha256:")
        or container.get("image_reference") != SGLANG_IMAGE_REFERENCE
        or type(task_memory) is not int
        or not 1 <= task_memory <= MAX_TASK_MEMORY_BYTES
        or container.get("shm_bytes") != 32 * 1024**3
        or not isinstance(command, list)
        or not 2 <= len(command) <= 256
        or not all(isinstance(item, str) and item and "\x00" not in item for item in command)
        or _canonical_sha(
            {
                "command": command,
                "environment": {
                    key: expected_env[key]
                    for key in (
                        "HF_HUB_OFFLINE",
                        "SGLANG_RAGGED_VERIFY_MODE",
                        "TOKENIZERS_PARALLELISM",
                        "TRANSFORMERS_OFFLINE",
                        "USE_FLAX",
                        "USE_TF",
                    )
                },
            }
        )
        != container.get("command_sha256")
        or environment != expected_env
        or _canonical_sha(environment) != container.get("environment_sha256")
    ):
        raise RemoteWorkerError("worker container identity changed")
    required_flags = {
        "--model-path": "/model",
        "--served-model-name": SERVED_ALIAS,
        "--host": "0.0.0.0",
        "--port": str(CONTAINER_PORT),
        "--tp-size": "1",
        "--dtype": "bfloat16",
        "--quantization": "modelopt_fp4",
        "--reasoning-parser": "qwen3",
        "--prefill-attention-backend": "triton",
        "--decode-attention-backend": "trtllm_mha",
        "--speculative-draft-model-quantization": "unquant",
        "--speculative-algorithm": "NEXTN",
        "--speculative-eagle-topk": "1",
        "--max-running-requests": "4",
        "--linear-attn-backend": "triton",
        "--moe-a2a-backend": "none",
        "--fp4-gemm-backend": "flashinfer_cutlass",
        "--speculative-moe-a2a-backend": "none",
    }
    for flag, expected in required_flags.items():
        indexes = [index for index, item in enumerate(command) if item == flag]
        if len(indexes) != 1 or indexes[0] + 1 >= len(command) or command[indexes[0] + 1] != expected:
            raise RemoteWorkerError(f"qualified command changed {flag}")
    moe_backends: dict[str, str] = {}
    for flag in (
        "--moe-runner-backend",
        "--speculative-moe-runner-backend",
    ):
        indexes = [index for index, item in enumerate(command) if item == flag]
        if len(indexes) != 1 or indexes[0] + 1 >= len(command):
            raise RemoteWorkerError(f"qualified command changed {flag}")
        moe_backends[flag] = command[indexes[0] + 1]
    if (
        moe_backends["--moe-runner-backend"]
        != REQUIRED_MOE_RUNNER_BACKEND
        or moe_backends["--speculative-moe-runner-backend"]
        != REQUIRED_MOE_RUNNER_BACKEND
    ):
        raise RemoteWorkerError(
            "qualified main/speculative MoE backend pair is not reviewed"
        )
    if command.count("--ple-offload-embedding") != 1:
        raise RemoteWorkerError("qualified command lost PLE host offload")
    values: dict[str, list[str | None]] = {}
    for flag in (
        "--mamba-ssm-dtype",
        "--linear-attn-decode-backend",
        "--linear-attn-prefill-backend",
        "--linear-attn-verify-backend",
        "--cuda-graph-config",
        "--chunked-prefill-size",
        "--speculative-num-steps",
        "--speculative-num-draft-tokens",
        "--mamba-radix-cache-strategy",
        "--mem-fraction-static",
        "--cpu-offload-gb",
        "--offload-group-size",
        "--offload-num-in-group",
    ):
        found: list[str | None] = []
        for index, item in enumerate(command):
            if item == flag:
                found.append(command[index + 1] if index + 1 < len(command) else None)
            elif item.startswith(flag + "="):
                found.append(item.split("=", 1)[1])
        values[flag] = found
    allowed_graphs = {
        '{"decode":{"backend":"full","max_bs":4,"bs":[1,2,4]},'
        '"prefill":{"backend":"disabled"}}',
        '{"decode":{"backend":"disabled"},'
        '"prefill":{"backend":"disabled"}}',
    }
    try:
        geometry = (
            int(str(values["--speculative-num-steps"][0])),
            int(str(values["--speculative-num-draft-tokens"][0])),
        )
    except (IndexError, TypeError, ValueError) as exc:
        raise RemoteWorkerError("qualified NEXTN geometry is malformed") from exc
    if (
        values["--mamba-ssm-dtype"] not in (["float32"], ["bfloat16"])
        or values["--linear-attn-decode-backend"]
        not in (["triton"], ["cutedsl"], ["flashinfer"])
        or values["--linear-attn-prefill-backend"] not in (["triton"], ["cutedsl"])
        or values["--linear-attn-verify-backend"]
        != (
            ["flashinfer"]
            if values["--linear-attn-decode-backend"] == ["flashinfer"]
            else ["triton"]
        )
        or values["--cuda-graph-config"] not in ([item] for item in allowed_graphs)
        or values["--chunked-prefill-size"] not in (["4096"], ["8192"])
        or values["--mem-fraction-static"] not in (["0.84"], ["0.86"], ["0.88"])
        or len(values["--speculative-num-steps"]) != 1
        or len(values["--speculative-num-draft-tokens"]) != 1
        or geometry not in {(1, 2), (2, 3), (3, 4)}
        or values["--linear-attn-decode-backend"] == ["flashinfer"]
        and values["--mamba-ssm-dtype"] != ["bfloat16"]
    ):
        raise RemoteWorkerError("qualified dynamic SM120 winner is outside its bounds")
    replay = command.count("--enable-linear-replayssm-spec")
    if replay not in {0, 1} or (
        replay == 1
        and (
            values["--mamba-radix-cache-strategy"] != ["extra_buffer"]
            or values["--mamba-ssm-dtype"] != ["float32"]
            or values["--linear-attn-decode-backend"] != ["triton"]
        )
    ) or (replay == 0 and values["--mamba-radix-cache-strategy"]):
        raise RemoteWorkerError("qualified ReplaySSM winner settings are inconsistent")
    if values["--cpu-offload-gb"] not in ([], ["0"], ["0.0"]):
        raise RemoteWorkerError("qualified command enables transformer CPU offload")
    if values["--offload-group-size"] not in ([], ["0"]):
        raise RemoteWorkerError("qualified command enables layer-group CPU offload")
    if values["--offload-num-in-group"] not in ([], ["0"]):
        raise RemoteWorkerError("qualified command enables grouped CPU offload")
    forbidden = {
        "--enable-hierarchical-cache",
        "--enable-hicache",
        "--no-ple-offload-embedding",
        "--offload-prefetch-step",
    }
    if any(
        item in forbidden or any(item.startswith(flag + "=") for flag in forbidden)
        for item in command
    ):
        raise RemoteWorkerError("qualified command enables an unqualified offload")


def _parse_sha256sums(
    root: Path,
    expected_digest: str,
    *,
    maximum_bytes: int,
    maximum_inodes: int,
    label: str,
) -> tuple[dict[str, tuple[str, int]], int, int]:
    sums = root / "SHA256SUMS"
    sums_metadata = sums.lstat()
    if (
        not stat.S_ISREG(sums_metadata.st_mode)
        or sums_metadata.st_uid != os.geteuid()
        or sums_metadata.st_nlink != 1
        or sums_metadata.st_mode & 0o077
    ):
        raise RemoteWorkerError(f"{label} SHA256SUMS is unsafe")
    payload = sums.read_bytes()
    if len(payload) > 16 * 1024 * 1024 or not payload.endswith(b"\n"):
        raise RemoteWorkerError("release SHA256SUMS is unsafe")
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise RemoteWorkerError("release tree digest changed")
    records: dict[str, tuple[str, int]] = {}
    previous = ""
    total = 0
    for raw in payload.decode("ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)", raw)
        if match is None or match.group(2) <= previous or match.group(2) == "SHA256SUMS":
            raise RemoteWorkerError("release SHA256SUMS is malformed")
        digest, name = match.groups()
        path = root / name
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
            or _sha256(path) != digest
        ):
            raise RemoteWorkerError(f"release member changed: {name}")
        previous = name
        total += metadata.st_size
        if total > maximum_bytes or len(records) + 2 > maximum_inodes:
            raise RemoteWorkerError(f"{label} exceeds its staged resource bound")
        records[name] = (digest, metadata.st_size)
    actual = {path.name for path in root.iterdir() if path.name != "SHA256SUMS"}
    if actual != set(records):
        raise RemoteWorkerError(f"remote {label} tree has unreceipted members")
    total += sums.lstat().st_size
    inodes = len(records) + 2
    if total > maximum_bytes or inodes > maximum_inodes:
        raise RemoteWorkerError(f"{label} exceeds its staged resource bound")
    return records, total, inodes


def _verify_release(request: Mapping[str, Any]) -> Path:
    binding = request.get("release")
    if not isinstance(binding, Mapping) or set(binding) != {
        "artifact_id",
        "worker_path",
        "digest_sha256",
        "filesystem_id",
        "size_bytes",
        "inode_count",
    }:
        raise RemoteWorkerError("release cache binding changed")
    root = _require_cache_path(binding.get("worker_path"), "release cache path")
    metadata = root.lstat()
    if (
        binding.get("artifact_id") != "aeon-qwen38-flash-next-release"
        or binding.get("digest_sha256") != request.get("release_tree_sha256")
        or str(metadata.st_dev) != binding.get("filesystem_id")
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or type(binding.get("size_bytes")) is not int
        or not 1 <= int(binding["size_bytes"]) <= MAX_RELEASE_BYTES
        or type(binding.get("inode_count")) is not int
        or not 1 <= int(binding["inode_count"]) <= MAX_RELEASE_INODES
    ):
        raise RemoteWorkerError("release cache entry identity changed")
    records, total, inodes = _parse_sha256sums(
        root,
        str(request["release_tree_sha256"]),
        maximum_bytes=MAX_RELEASE_BYTES,
        maximum_inodes=MAX_RELEASE_INODES,
        label="release",
    )
    if total != binding["size_bytes"] or inodes != binding["inode_count"]:
        raise RemoteWorkerError("release cache inventory changed")
    expected = {
        "RELEASE_MANIFEST.json": request["release_manifest_sha256"],
        "RUNTIME_CONFIG.json": request["runtime_config_sha256"],
        "QUALIFICATION_REPORT.json": request["qualification_sha256"],
        "QUALIFICATION_TUNED_MTP_OFF.json": request[
            "qualification_mtp_off_sha256"
        ],
        "QUALIFICATION_TUNED_MTP_ON_WINNER.json": request[
            "qualification_mtp_on_sha256"
        ],
        "PLE_MATERIALIZATION.json": request[
            "ple_materialization_manifest_sha256"
        ],
        "materialize_ple.py": request["ple_materializer_sha256"],
    }
    if any(records.get(name, (None,))[0] != digest for name, digest in expected.items()):
        raise RemoteWorkerError("release qualification/runtime receipts changed")
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    preservation = manifest.get("preservation") if isinstance(manifest, Mapping) else None
    manifest_runtime = manifest.get("runtime") if isinstance(manifest, Mapping) else None
    if not isinstance(preservation, Mapping) or preservation != {
        **dict(preservation),
        "vision_image_video_bf16": True,
        "mtp_bf16": True,
        "ple_fp8_host_offload_contract": True,
        "ordinary_transformer_weight_cpu_offload": False,
    }:
        raise RemoteWorkerError("release preservation contract changed")
    if (
        not isinstance(manifest_runtime, Mapping)
        or manifest_runtime.get("sglang_source_stack_sha256")
        != SGLANG_SOURCE_STACK_SHA256
        or manifest_runtime.get("oci_image") != SGLANG_IMAGE_REFERENCE
        or manifest_runtime.get("oci_manifest_digest")
        != SGLANG_IMAGE_DIGEST
        or manifest_runtime.get("oci_config_digest")
        != SGLANG_IMAGE_CONFIG_DIGEST
        or manifest_runtime.get("oci_archive_sha256")
        != SGLANG_IMAGE_ARCHIVE_SHA256
        or manifest_runtime.get("local_docker_image_id") != SGLANG_IMAGE_ID
        or manifest_runtime.get("required_image_labels") != EXPECTED_IMAGE_LABELS
        or manifest_runtime.get("wire_served_alias") != SERVED_ALIAS
        or manifest_runtime.get("display_name") != DISPLAY_NAME
        or manifest_runtime.get("artifact_name") != ARTIFACT_NAME
    ):
        raise RemoteWorkerError("release runtime image/model identity changed")
    packaging = manifest.get("packaging") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(packaging, Mapping)
        or packaging.get("kind")
        != "thin-private-official-ple-materialized-offline"
        or packaging.get("canonical_checkpoint_tree_sha256")
        != request["materialized_checkpoint_tree_sha256"]
        or packaging.get("ple_materialization_manifest_sha256")
        != request["ple_materialization_manifest_sha256"]
        or packaging.get("ple_materializer_sha256")
        != request["ple_materializer_sha256"]
        or packaging.get("omitted_ple_shard_count") != 33
        or packaging.get("published_ple_payload_bytes") != 0
    ):
        raise RemoteWorkerError("thin release packaging contract changed")
    materialization = json.loads(
        (root / "PLE_MATERIALIZATION.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(materialization, Mapping)
        or materialization.get("checkpoint_tree_sha256")
        != request["materialized_checkpoint_tree_sha256"]
        or materialization.get("materializer_sha256")
        != request["ple_materializer_sha256"]
        or not isinstance(materialization.get("canonical_files"), Mapping)
        or len(materialization["canonical_files"]) < 1
    ):
        raise RemoteWorkerError("release materialization closure changed")
    runtime = json.loads((root / "RUNTIME_CONFIG.json").read_text(encoding="utf-8"))
    if (
        runtime.get("served_alias") != SERVED_ALIAS
        or runtime.get("display_name") != DISPLAY_NAME
        or runtime.get("artifact_name") != ARTIFACT_NAME
        or runtime.get("model_architecture") != MODEL_ARCHITECTURE
        or runtime.get("toolchain", {}).get("sglang", {}).get(
            "source_stack_sha256"
        )
        != SGLANG_SOURCE_STACK_SHA256
        or runtime.get("toolchain", {}).get("sglang", {}).get(
            "oci_image_digest"
        )
        != SGLANG_IMAGE_DIGEST
        or runtime.get("toolchain", {}).get("sglang", {}).get(
            "oci_config_digest"
        )
        != SGLANG_IMAGE_CONFIG_DIGEST
        or runtime.get("toolchain", {}).get("sglang", {}).get(
            "oci_archive_sha256"
        )
        != SGLANG_IMAGE_ARCHIVE_SHA256
        or runtime.get("toolchain", {}).get("sglang", {}).get(
            "local_docker_image_id"
        )
        != SGLANG_IMAGE_ID
        or runtime.get("toolchain", {}).get("sglang", {}).get(
            "required_image_labels"
        )
        != EXPECTED_IMAGE_LABELS
        or runtime.get("model_path_contract") != {
        "checkpoint_tree_sha256": request["materialized_checkpoint_tree_sha256"],
        "host_path_placeholder": "@AEON_MATERIALIZED_MODEL_PATH@",
        "container_path": "/model",
        "mount_read_only": True,
        "source_role": "offline-materialized-canonical-checkpoint",
        }
    ):
        raise RemoteWorkerError("runtime materialized-model path contract changed")
    arm = runtime.get("arms", {}).get("tuned_mtp_on_winner", {})
    mtp = arm.get("runtime_config", {}) if isinstance(arm, Mapping) else {}
    expected_environment = {
        "HF_HUB_OFFLINE": "1",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "USE_FLAX": "0",
        "USE_TF": "0",
    }
    if (
        arm.get("environment") != expected_environment
        or mtp.get("runtime_environment") != expected_environment
        or mtp.get("ragged_verify_mode") != "static"
        or mtp.get("ple_offload_embedding") is not True
        or mtp.get("cpu_offload_gb") != 0
        or (
            mtp.get("speculative_num_steps"),
            mtp.get("speculative_num_draft_tokens"),
        )
        not in {(1, 2), (2, 3), (3, 4)}
        or mtp.get("speculative_eagle_topk") != 1
        or mtp.get("requested_speculative_algorithm") != "NEXTN"
        or mtp.get("requested_speculative_draft_model_quantization") != "unquant"
        or mtp.get("speculative_draft_model_quantization") is not None
        or mtp.get("reasoning_parser") != "qwen3"
        or mtp.get("prefill_attention_backend") != "triton"
        or mtp.get("decode_attention_backend") != "trtllm_mha"
        or mtp.get("moe_a2a_backend") != "none"
        or mtp.get("moe_runner_backend") != REQUIRED_MOE_RUNNER_BACKEND
        or mtp.get("fp4_gemm_backend") != "flashinfer_cutlass"
        or mtp.get("speculative_moe_a2a_backend") != "none"
        or mtp.get("speculative_moe_runner_backend")
        != REQUIRED_MOE_RUNNER_BACKEND
    ):
        raise RemoteWorkerError("qualified MTP-on runtime contract changed")
    return root


def _verify_materialized_model(request: Mapping[str, Any]) -> Path:
    binding = request.get("materialized_model")
    if not isinstance(binding, Mapping) or set(binding) != {
        "artifact_id",
        "worker_path",
        "digest_sha256",
        "filesystem_id",
        "size_bytes",
        "inode_count",
    }:
        raise RemoteWorkerError("materialized model cache binding changed")
    root = _require_cache_path(binding.get("worker_path"), "model cache path")
    metadata = root.lstat()
    if (
        binding.get("artifact_id")
        != "aeon-qwen38-flash-next-materialized-model"
        or binding.get("digest_sha256")
        != request.get("materialized_checkpoint_tree_sha256")
        or str(metadata.st_dev) != binding.get("filesystem_id")
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or type(binding.get("size_bytes")) is not int
        or not 1 <= int(binding["size_bytes"]) <= MAX_MODEL_BYTES
        or type(binding.get("inode_count")) is not int
        or not 1 <= int(binding["inode_count"]) <= MAX_MODEL_INODES
    ):
        raise RemoteWorkerError("materialized model cache identity changed")
    _records, total, inodes = _parse_sha256sums(
        root,
        str(request["materialized_checkpoint_tree_sha256"]),
        maximum_bytes=MAX_MODEL_BYTES,
        maximum_inodes=MAX_MODEL_INODES,
        label="materialized model",
    )
    if total != binding["size_bytes"] or inodes != binding["inode_count"]:
        raise RemoteWorkerError("materialized model cache inventory changed")
    return root


def _docker(arguments: Sequence[str], *, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [LOW_PRIORITY, DOCKER, *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            "HOME": "/home/aday",
            "PATH": "/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
        },
    )


def _docker_absent(result: subprocess.CompletedProcess[str], identity: str) -> bool:
    return result.returncode == 1 and re.search(
        rf"(?:No such object|No such container):\s*{re.escape(identity)}(?:\s|$)",
        result.stderr,
    ) is not None


def _inspect_container(identity: str) -> dict[str, Any] | None:
    result = _docker(("container", "inspect", identity), timeout=30)
    if _docker_absent(result, identity):
        return None
    if result.returncode != 0:
        raise RemoteWorkerError("exact Docker container inspection failed")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RemoteWorkerError("Docker container inspection is malformed") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise RemoteWorkerError("Docker did not return one exact container")
    return value[0]


def _verify_image(request: Mapping[str, Any]) -> None:
    binding = request.get("image")
    if not isinstance(binding, Mapping) or set(binding) != {
        "artifact_id",
        "worker_path",
        "digest_sha256",
        "filesystem_id",
        "size_bytes",
        "inode_count",
        "payload_sha256",
    }:
        raise RemoteWorkerError("image cache binding changed")
    receipt_path = _require_cache_path(binding.get("worker_path"), "image cache path")
    metadata = receipt_path.lstat()
    image_id = "sha256:" + str(binding.get("digest_sha256"))
    if any(
        value == "sha256:" + ("0" * 64)
        for value in (
            SGLANG_IMAGE_DIGEST,
            SGLANG_IMAGE_CONFIG_DIGEST,
            SGLANG_IMAGE_ID,
        )
    ):
        raise RemoteWorkerError(
            "patched SM120 image manifest/config identities are not settled"
        )
    if (
        binding.get("artifact_id") != "aeon-qwen38-flash-next-image"
        or binding.get("digest_sha256") != request.get("container", {}).get("image_id", "").removeprefix("sha256:")
        or str(metadata.st_dev) != binding.get("filesystem_id")
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= 64 * 1024
    ):
        raise RemoteWorkerError("image cache receipt identity changed")
    receipt = _private_json(receipt_path, maximum=64 * 1024)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("image_id") != image_id
        or type(receipt.get("image_size_bytes")) is not int
        or not 0 < receipt["image_size_bytes"] <= MAX_IMAGE_BYTES
        or receipt.get("archive_payload_sha256") != binding.get("payload_sha256")
    ):
        raise RemoteWorkerError("image cache semantic receipt changed")
    result = _docker(("image", "inspect", SGLANG_IMAGE_REFERENCE), timeout=30)
    if result.returncode != 0:
        raise RemoteWorkerError("pinned SGLang image is absent; worker never pulls")
    try:
        image = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RemoteWorkerError("pinned SGLang image inspection is malformed") from exc
    inspected = image[0] if isinstance(image, list) and len(image) == 1 else None
    config = inspected.get("Config") if isinstance(inspected, Mapping) else None
    labels = config.get("Labels") if isinstance(config, Mapping) else None
    if (
        not isinstance(inspected, Mapping)
        or inspected.get("Id") != image_id
        or not isinstance(inspected.get("Descriptor"), Mapping)
        or inspected["Descriptor"].get("digest") != SGLANG_IMAGE_DIGEST
        or not isinstance(inspected.get("RepoDigests"), list)
        or SGLANG_IMAGE_REPO_DIGEST not in inspected["RepoDigests"]
        or inspected.get("Size") != receipt["image_size_bytes"]
        or not isinstance(labels, Mapping)
        or any(labels.get(key) != expected for key, expected in EXPECTED_IMAGE_LABELS.items())
    ):
        raise RemoteWorkerError("pinned SGLang image identity changed")


def _labels(request: Mapping[str, Any]) -> dict[str, str]:
    return {
        "aeon.fleet.profile": PROFILE_ID,
        "aeon.fleet.runtime": str(request["runtime_id"]),
        "aeon.fleet.claim_sha256": hashlib.sha256(
            str(request["lease"]["claim_id"]).encode("utf-8")
        ).hexdigest(),
        "aeon.fleet.binding": str(request["binding_sha256"]),
        "aeon.fleet.release": str(request["release_tree_sha256"]),
        "aeon.fleet.checkpoint": str(
            request["materialized_checkpoint_tree_sha256"]
        ),
        "aeon.fleet.materialization": str(
            request["materialization_receipt_sha256"]
        ),
        "aeon.fleet.command": str(request["container"]["command_sha256"]),
        "aeon.fleet.remote_contract": str(request["contract_sha256"]),
        "aeon.model.artifact": ARTIFACT_NAME,
        "aeon.model.display_name": DISPLAY_NAME,
        "aeon.model.wire_alias": SERVED_ALIAS,
    }


def _create_argv(request: Mapping[str, Any], model: Path) -> tuple[str, ...]:
    container = request["container"]
    lease = request["lease"]
    arguments = [
        "container",
        "create",
        "--pull=never",
        "--name",
        str(container["name"]),
        "--user",
        f"{os.geteuid()}:{os.getegid()}",
        "--gpus",
        f"device={lease['gpu_uuid']}",
        "--memory",
        f"{container['task_memory_bytes']}b",
        "--memory-swap",
        f"{container['task_memory_bytes']}b",
        "--shm-size",
        f"{container['shm_bytes']}b",
        "--pids-limit",
        "4096",
        "--ulimit",
        "memlock=-1:-1",
        "--security-opt",
        "no-new-privileges=true",
        "--publish",
        f"127.0.0.1:{container['host_port']}:{CONTAINER_PORT}",
        "--mount",
        f"type=bind,src={model},dst=/model,readonly",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,exec,size=8g",
    ]
    for key, value in sorted(_labels(request).items()):
        arguments.extend(("--label", f"{key}={value}"))
    for key, value in sorted(container["environment"].items()):
        arguments.extend(("--env", f"{key}={value}"))
    arguments.extend((container["image_reference"], *container["command"]))
    return tuple(arguments)


def _process_start_ticks(pid: int) -> int:
    payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    end = payload.rfind(")")
    if end < 0:
        raise RemoteWorkerError("container process stat is malformed")
    return int(payload[end + 2 :].split()[19])


def _cgroup_exact(pid: int, container_id: str, memory_bytes: int) -> bool:
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii").splitlines()
        unified = [line.split(":", 2)[2] for line in lines if line.startswith("0::")]
        if len(unified) != 1:
            return False
        relative = PurePosixPath(unified[0])
        if not relative.is_absolute() or ".." in relative.parts:
            return False
        if not any(container_id in item or container_id[:12] in item for item in relative.parts):
            return False
        maximum = Path("/sys/fs/cgroup").joinpath(*relative.parts[1:], "memory.max")
        return int(maximum.read_text(encoding="ascii").strip()) == memory_bytes
    except (OSError, ValueError, IndexError):
        return False


def _container_identity(
    item: Mapping[str, Any], request: Mapping[str, Any], *, running: bool
) -> tuple[str, int]:
    container = request["container"]
    lease = request["lease"]
    config = item.get("Config")
    host_config = item.get("HostConfig")
    state = item.get("State")
    mounts = item.get("Mounts")
    container_id = str(item.get("Id") or "")
    if (
        _CONTAINER.fullmatch(container_id) is None
        or item.get("Name") != "/" + container["name"]
        or not isinstance(config, Mapping)
        or not isinstance(host_config, Mapping)
        or not isinstance(state, Mapping)
        or not isinstance(mounts, list)
        or config.get("Image") != container["image_reference"]
        or config.get("User") != f"{os.geteuid()}:{os.getegid()}"
        or config.get("Cmd") != container["command"]
        or host_config.get("Memory") != container["task_memory_bytes"]
        or host_config.get("MemorySwap") != container["task_memory_bytes"]
        or host_config.get("ShmSize") != container["shm_bytes"]
        or host_config.get("PidsLimit") != 4096
        or host_config.get("Ulimits")
        != [{"Name": "memlock", "Hard": -1, "Soft": -1}]
        or host_config.get("SecurityOpt") != ["no-new-privileges=true"]
        or host_config.get("PortBindings")
        != {
            f"{CONTAINER_PORT}/tcp": [
                {
                    "HostIp": "127.0.0.1",
                    "HostPort": str(container["host_port"]),
                }
            ]
        }
    ):
        raise RemoteWorkerError("exact remote container configuration changed")
    labels = config.get("Labels")
    if not isinstance(labels, Mapping) or any(
        labels.get(key) != value for key, value in _labels(request).items()
    ):
        raise RemoteWorkerError("exact remote container labels changed")
    parsed: dict[str, list[str]] = {}
    for raw in config.get("Env") or []:
        if isinstance(raw, str) and "=" in raw:
            key, value = raw.split("=", 1)
            parsed.setdefault(key, []).append(value)
    if any(parsed.get(key) != [value] for key, value in container["environment"].items()):
        raise RemoteWorkerError("exact remote container environment changed")
    model_mounts = [item for item in mounts if item.get("Destination") == "/model"]
    if len(model_mounts) != 1 or (
        model_mounts[0].get("Source")
        != request["materialized_model"]["worker_path"]
        or model_mounts[0].get("RW") is not False
    ):
        raise RemoteWorkerError("exact remote materialized-model mount changed")
    device_requests = host_config.get("DeviceRequests") or []
    if (
        len(device_requests) != 1
        or device_requests[0].get("DeviceIDs") != [lease["gpu_uuid"]]
        or device_requests[0].get("Capabilities") != [["gpu"]]
    ):
        raise RemoteWorkerError("exact remote GPU UUID binding changed")
    pid = state.get("Pid")
    if type(pid) is not int or pid < 0:
        raise RemoteWorkerError("remote container PID is malformed")
    is_running = state.get("Running") is True and state.get("Status") == "running"
    if running and (not is_running or pid <= 1):
        raise RemoteWorkerError("exact remote container is not running")
    return container_id, pid


def _health(port: int) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/health", timeout=10) as response:
            return response.status == 200 and len(response.read(64 * 1024 + 1)) <= 64 * 1024
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _preflight(request: Mapping[str, Any], request_sha256: str) -> dict[str, Any]:
    _verify_release(request)
    _verify_materialized_model(request)
    _verify_image(request)
    if _inspect_container(str(request["container"]["name"])) is not None:
        raise RemoteWorkerError("exact remote container name already exists")
    return {
        "state": "verified",
        "request_sha256": request_sha256,
        "contract_sha256": request["contract_sha256"],
        "binding_sha256": request["binding_sha256"],
        "release_tree_sha256": request["release_tree_sha256"],
        "materialized_checkpoint_tree_sha256": request[
            "materialized_checkpoint_tree_sha256"
        ],
        "materialization_receipt_sha256": request[
            "materialization_receipt_sha256"
        ],
        "image_digest_sha256": request["image"]["digest_sha256"],
        "command_sha256": request["container"]["command_sha256"],
        "environment_sha256": request["container"]["environment_sha256"],
    }


def _start(request: Mapping[str, Any], request_sha256: str) -> dict[str, Any]:
    _verify_release(request)
    model = _verify_materialized_model(request)
    _verify_image(request)
    run_dir = Path(str(request["run_dir"]))
    receipt_path = run_dir / RECEIPT_NAME
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RemoteWorkerError("remote container receipt already exists")
    if _inspect_container(str(request["container"]["name"])) is not None:
        raise RemoteWorkerError("exact remote container name already exists")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "runtime_id": request["runtime_id"],
        "profile_id": PROFILE_ID,
        "request_sha256": request_sha256,
        "contract_sha256": request["contract_sha256"],
        "binding_sha256": request["binding_sha256"],
        "release_tree_sha256": request["release_tree_sha256"],
        "materialized_checkpoint_tree_sha256": request[
            "materialized_checkpoint_tree_sha256"
        ],
        "materialization_receipt_sha256": request[
            "materialization_receipt_sha256"
        ],
        "command_sha256": request["container"]["command_sha256"],
        "environment_sha256": request["container"]["environment_sha256"],
        "container_name": request["container"]["name"],
        "container_id": None,
        "state": "creating",
        "pid": None,
        "start_ticks": None,
    }
    _atomic_json(receipt_path, receipt)
    created = _docker(_create_argv(request, model), timeout=180)
    container_id = created.stdout.strip()
    if created.returncode != 0 or _CONTAINER.fullmatch(container_id) is None:
        if _inspect_container(str(request["container"]["name"])) is None:
            return {"state": "absent", "process_absent": True}
        raise RemoteWorkerError("remote Docker create result is ambiguous")
    receipt.update(container_id=container_id, state="created")
    _atomic_json(receipt_path, receipt)
    item = _inspect_container(container_id)
    if item is None:
        return {"state": "absent", "process_absent": True}
    _container_identity(item, request, running=False)
    started = _docker(("container", "start", container_id), timeout=120)
    if started.returncode != 0 or started.stdout.strip() != container_id:
        raise RemoteWorkerError("exact remote container did not start")
    item = _inspect_container(container_id)
    if item is None:
        return {"state": "absent", "process_absent": True}
    _identity, pid = _container_identity(item, request, running=True)
    ticks = _process_start_ticks(pid)
    if not _cgroup_exact(pid, container_id, int(request["container"]["task_memory_bytes"])):
        raise RemoteWorkerError("remote task cgroup identity changed")
    receipt.update(state="running", pid=pid, start_ticks=ticks)
    _atomic_json(receipt_path, receipt)
    return {
        "state": "starting",
        "process_absent": False,
        "display_name": DISPLAY_NAME,
        "artifact_name": ARTIFACT_NAME,
        "served_alias": SERVED_ALIAS,
        "container_id": container_id,
        "pid": pid,
        "start_ticks": ticks,
    }


def _saved_receipt(request: Mapping[str, Any], request_sha256: str) -> dict[str, Any]:
    receipt = _private_json(Path(str(request["run_dir"])) / RECEIPT_NAME, maximum=64 * 1024)
    expected = {
        "runtime_id": request["runtime_id"],
        "profile_id": PROFILE_ID,
        "request_sha256": request_sha256,
        "contract_sha256": request["contract_sha256"],
        "binding_sha256": request["binding_sha256"],
        "release_tree_sha256": request["release_tree_sha256"],
        "materialized_checkpoint_tree_sha256": request[
            "materialized_checkpoint_tree_sha256"
        ],
        "materialization_receipt_sha256": request[
            "materialization_receipt_sha256"
        ],
        "command_sha256": request["container"]["command_sha256"],
        "environment_sha256": request["container"]["environment_sha256"],
        "container_name": request["container"]["name"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RemoteWorkerError("remote container receipt identity changed")
    return receipt


def _status(request: Mapping[str, Any], request_sha256: str) -> dict[str, Any]:
    try:
        receipt = _saved_receipt(request, request_sha256)
    except FileNotFoundError:
        if _inspect_container(str(request["container"]["name"])) is None:
            return {"state": "absent", "process_absent": True}
        raise RemoteWorkerError("unreceipted exact container name exists")
    container_id = receipt.get("container_id")
    if not isinstance(container_id, str) or _CONTAINER.fullmatch(container_id) is None:
        item = _inspect_container(str(request["container"]["name"]))
        if receipt.get("state") == "creating" and item is None:
            return {"state": "absent", "process_absent": True}
        if receipt.get("state") != "creating" or item is None:
            raise RemoteWorkerError("remote container receipt is incomplete")
        container_id, _unused_pid = _container_identity(item, request, running=False)
        receipt.update(container_id=container_id, state="created")
        _atomic_json(Path(str(request["run_dir"])) / RECEIPT_NAME, receipt)
    else:
        item = _inspect_container(container_id)
    if item is None:
        return {"state": "absent", "process_absent": True, "container_id": container_id}
    _identity, pid = _container_identity(item, request, running=True)
    ticks = _process_start_ticks(pid)
    if not _cgroup_exact(
        pid, container_id, int(request["container"]["task_memory_bytes"])
    ):
        raise RemoteWorkerError("remote live process/cgroup identity changed")
    if receipt.get("state") in {"creating", "created"}:
        receipt.update(state="running", pid=pid, start_ticks=ticks)
        _atomic_json(Path(str(request["run_dir"])) / RECEIPT_NAME, receipt)
    elif (
        receipt.get("state") != "running"
        or receipt.get("pid") != pid
        or receipt.get("start_ticks") != ticks
    ):
        raise RemoteWorkerError("remote live process receipt changed")
    return {
        "state": "ready" if _health(int(request["container"]["host_port"])) else "starting",
        "process_absent": False,
        "display_name": DISPLAY_NAME,
        "artifact_name": ARTIFACT_NAME,
        "served_alias": SERVED_ALIAS,
        "container_id": container_id,
        "pid": pid,
        "start_ticks": ticks,
    }


def _stop(request: Mapping[str, Any], request_sha256: str) -> dict[str, Any]:
    try:
        receipt = _saved_receipt(request, request_sha256)
    except FileNotFoundError:
        if _inspect_container(str(request["container"]["name"])) is None:
            return {"state": "stopped", "process_absent": True}
        raise RemoteWorkerError("unreceipted exact container name exists")
    container_id = receipt.get("container_id")
    if not isinstance(container_id, str) or _CONTAINER.fullmatch(container_id) is None:
        item = _inspect_container(str(request["container"]["name"]))
        if item is None:
            receipt.update(state="stopped", pid=None, start_ticks=None)
            _atomic_json(Path(str(request["run_dir"])) / RECEIPT_NAME, receipt)
            return {"state": "stopped", "process_absent": True}
        if receipt.get("state") != "creating":
            raise RemoteWorkerError("remote receipt lacks exact container authority")
        container_id, _unused_pid = _container_identity(item, request, running=False)
        receipt.update(container_id=container_id, state="created")
        _atomic_json(Path(str(request["run_dir"])) / RECEIPT_NAME, receipt)
    else:
        item = _inspect_container(container_id)
    if item is not None:
        _container_identity(item, request, running=False)
        if item.get("State", {}).get("Running") is True:
            stopped = _docker(("container", "stop", "--time", "30", container_id), timeout=60)
            if stopped.returncode != 0 or stopped.stdout.strip() != container_id:
                return {"state": "stopping", "process_absent": False}
        item = _inspect_container(container_id)
        if item is not None:
            _container_identity(item, request, running=False)
            if item.get("State", {}).get("Running") is True:
                return {"state": "stopping", "process_absent": False}
            removed = _docker(("container", "rm", container_id), timeout=30)
            if removed.returncode != 0 or removed.stdout.strip() != container_id:
                return {"state": "stopping", "process_absent": False}
    absent = _inspect_container(container_id) is None
    if absent:
        receipt.update(state="stopped", pid=None, start_ticks=None)
        _atomic_json(Path(str(request["run_dir"])) / RECEIPT_NAME, receipt)
    return {"state": "stopped" if absent else "stopping", "process_absent": absent}


def _cleanup(request: Mapping[str, Any], request_sha256: str) -> dict[str, Any]:
    stopped = _stop(request, request_sha256)
    if stopped.get("process_absent") is not True:
        raise RemoteWorkerError("remote container absence is not proven")
    run_dir = Path(str(request["run_dir"]))
    source_dir = run_dir / "source"
    allowed = {
        "flash-next-remote-request.json",
        RECEIPT_NAME,
        "source",
    }
    members = {item.name for item in run_dir.iterdir()}
    if not members <= allowed:
        raise RemoteWorkerError("unknown data blocks exact remote scratch cleanup")
    source_members = {item.name for item in source_dir.iterdir()} if source_dir.exists() else set()
    if not source_members <= {WORKER_NAME}:
        raise RemoteWorkerError("unknown source data blocks exact remote scratch cleanup")
    reclaimed = 0
    for path in (
        source_dir / WORKER_NAME,
        run_dir / RECEIPT_NAME,
        run_dir / "flash-next-remote-request.json",
    ):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
        ):
            raise RemoteWorkerError("remote scratch cleanup target changed")
        reclaimed += metadata.st_size
        path.unlink()
    if source_dir.exists():
        source_dir.rmdir()
    run_dir.rmdir()
    return {
        "state": "cleaned",
        "process_absent": True,
        "cache_entries_removed": 0,
        "docker_images_removed": 0,
        "reclaimed_bytes": reclaimed,
    }


_ACTIONS = {
    "preflight": _preflight,
    "start": _start,
    "status": _status,
    "stop": _stop,
    "cleanup": _cleanup,
}


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in _ACTIONS:
        print(json.dumps({"ok": False, "error": "invalid_action"}, sort_keys=True))
        return 64
    try:
        path = Path(sys.argv[2])
        request_sha256 = _require_sha(sys.argv[3], "request digest")
        # One exclusive directory lock serializes every request-bound lifecycle
        # mutation.  If an SSH response is lost, a later recovery waits for the
        # original action to quiesce before it may inspect, stop, or clean the
        # exact container.  The open directory remains a valid lock anchor even
        # when the cleanup action removes the now-empty pathname.
        run_dir = _safe_run_dir(path.parent)
        lock = os.open(
            run_dir,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            request = _request(path, request_sha256)
            result = _ACTIONS[sys.argv[1]](request, request_sha256)
        finally:
            os.close(lock)
    except (RemoteWorkerError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:500]},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"ok": True, **result}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
