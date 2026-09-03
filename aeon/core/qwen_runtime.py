"""Fail-closed host-side lifecycle for one coordinator-leased Qwen runtime.

The same exact Docker lifecycle runs locally on `.177` or inside a separately
released worker adapter. Fleet placement and SSH transport remain centralized on
`.177`; this module never selects a host or contacts another worker.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import secrets
import shutil
import socket
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .compute_profile import QWEN38_VLLM_PROFILE, ComputeProfile
from .gpu_queue import (
    QWEN_LEASE_FILE,
    _coord,
    clear_reconciled_lease_state,
    current_lease,
    release_vram,
)
from .qwen_capabilities import (
    RTX5000_RELEASE_CANDIDATE_KEY,
    STANDARD_CONTEXT_TOKENS,
    STANDARD_MIN_PHYSICAL_VRAM_GB,
    STANDARD_VRAM_BUDGET_GB,
    QwenCapabilityError,
    QwenRuntimeCapability,
    active_qwen_runtime_capability,
    validate_qwen_capability_receipt,
)
from .utils.io import read_bounded_fd

LEGACY_SCHEMA_VERSION = 6
SCHEMA_VERSION = 7
MIN_QWEN_PHYSICAL_VRAM_GB = STANDARD_MIN_PHYSICAL_VRAM_GB
QWEN_PLANNED_VRAM_GB = STANDARD_VRAM_BUDGET_GB
QWEN_CONTEXT_TOKENS = STANDARD_CONTEXT_TOKENS
QWEN_GPU_MEMORY_UTILIZATION = 0.415
QWEN_MAX_NUM_SEQS = 1
QWEN_MAX_BATCHED_TOKENS = 32768
QWEN_RELEASE_ATTENTION_BACKEND = "TRITON_ATTN"
QWEN_CONTAINER_SHM_BYTES = 8 * 1024**3
QWEN_RUNTIME_CACHE_BYTES = 8 * 1024**3
QWEN_CONTAINER_TMPDIR = "/workspace/cache"
QWEN_IMAGE_EXPOSED_PORTS = frozenset({"8000/tcp"})
MAX_IMAGE_LOGICAL_BYTES = 64 * 1024**3
MAX_MODEL_ARTIFACT_BYTES = 128 * 1024**3
_MODEL_SHA256_VERIFICATION_SIDECAR = ".podcast-sha256-verified"
RUNTIME_ROOT = Path("/home/aday/.aeon/runtime/qwen38")
RUNTIME_STATE_FILE = RUNTIME_ROOT / "runtime.json"
FLEET_LOW_PRIORITY = Path("/home/aday/bin/fleet-low-priority")
DOCKER = Path("/home/aday/bin/docker")
REAL_DOCKER = Path("/usr/bin/docker")
HOST_BASH = Path("/usr/bin/bash")
HOST_NICE = Path("/usr/bin/nice")
HOST_IONICE = Path("/usr/bin/ionice")
HOST_SHA256SUM = Path("/usr/bin/sha256sum")
HOST_PYTHON = Path("/usr/bin/python3")
SYSTEM_EXECUTABLE_UIDS = frozenset({0})
HOST_LAUNCH_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/aday",
    "LANG": "C",
    "LC_ALL": "C",
}
DOCKER_HOST = "unix:///var/run/docker.sock"
DOCKER_CONFIG_DIRNAME = "docker-cli-empty"

_ACTIVE_CAPABILITY, _ACTIVE_CAPABILITY_MANIFEST_SHA256 = (
    active_qwen_runtime_capability()
)
LOCAL_COORD_HOST = _ACTIVE_CAPABILITY.host
LOCAL_COORD_HOSTNAME = _ACTIVE_CAPABILITY.hostname
APPROVED_HOSTS = {_ACTIVE_CAPABILITY.host: _ACTIVE_CAPABILITY.hostname}


class _LoopbackResponse:
    """Small streaming facade over one exact stdlib loopback connection."""

    def __init__(
        self,
        connection: http.client.HTTPConnection,
        response: http.client.HTTPResponse,
    ) -> None:
        self._connection = connection
        self._response = response
        self.status_code = response.status
        self.headers = response.headers

    def iter_content(self, *, chunk_size: int):
        while True:
            chunk = self._response.read(chunk_size)
            if not chunk:
                return
            yield chunk

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()


def _loopback_get(url: str, *, timeout: float) -> _LoopbackResponse:
    """Stream one GET that cannot use proxies, redirects, DNS, or another host."""

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise QwenRuntimeError("Qwen loopback URL is malformed") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or port is None
        or not 1024 <= port <= 65535
        or parsed.fragment
    ):
        raise QwenRuntimeError("Qwen endpoint is not an exact loopback URL")
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(
            "GET",
            target,
            headers={"Accept": "application/json", "Connection": "close"},
        )
        response = connection.getresponse()
    except BaseException:
        connection.close()
        raise
    return _LoopbackResponse(connection, response)


def _bounded_loopback_body(response: Any, maximum: int) -> bytes:
    """Consume and close a streamed local runtime response within ``maximum``."""

    if type(maximum) is not int or maximum <= 0:
        raise QwenRuntimeError("Qwen endpoint response bound is invalid")
    payload = bytearray()
    try:
        advertised = response.headers.get("content-length")
        if advertised is not None:
            try:
                advertised_size = int(advertised)
            except (TypeError, ValueError) as exc:
                raise QwenRuntimeError(
                    "Qwen endpoint response Content-Length is malformed"
                ) from exc
            if advertised_size < 0 or advertised_size > maximum:
                raise QwenRuntimeError("Qwen endpoint response exceeded its bound")
        for chunk in response.iter_content(chunk_size=min(64 * 1024, maximum + 1)):
            payload.extend(chunk)
            if len(payload) > maximum:
                raise QwenRuntimeError("Qwen endpoint response exceeded its bound")
    finally:
        response.close()
    return bytes(payload)


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _container_tmpfs_options(*, executable: bool) -> str:
    if type(executable) is not bool:
        raise QwenRuntimeError("container tmpfs execution policy is malformed")
    execution = "exec," if executable else ""
    return (
        f"rw,{execution}nosuid,nodev,size={QWEN_RUNTIME_CACHE_BYTES},"
        f"uid={os.geteuid()},gid={os.getegid()},mode=0700"
    )


def _coordinator_status_claim_is_exclusive(value: Any) -> bool:
    """Require the coordinator status wire's exact SQLite INTEGER form."""

    return type(value) is int and value == 1


SOURCE_FILES = (
    "aeon/__init__.py",
    "aeon/core/__init__.py",
    "aeon/core/action_schema.py",
    "aeon/core/compute_profile.py",
    "aeon/core/deploy_planner.py",
    "aeon/core/fleet_hosts.py",
    "aeon/core/gpu.py",
    "aeon/core/gpu_queue.py",
    "aeon/core/model_catalog.py",
    "aeon/core/mtp_tuning.py",
    "aeon/core/qwen_artifact_cache.py",
    "aeon/core/qwen_capabilities.py",
    "aeon/core/qwen_fleet_runtime.py",
    "aeon/core/qwen_runtime.py",
    "aeon/core/sampling.py",
    "aeon/core/data/qwen38_mtp_selection.json",
    "aeon/core/data/qwen38_rtx5000_178_128k_release_receipt.json",
    "aeon/core/data/qwen38_rtx5000_128k_release_receipt.json",
    "aeon/core/data/qwen_runtime_capabilities.json",
    "aeon/scripts/vllm_uuid_sitecustomize.py",
    "aeon/scripts/warmup_qwen38_vllm.py",
    "aeon/scripts/qwen_remote_worker.py",
)

# Schema-6 teardown receipts used this exact ordered closure.  Keep it frozen so
# an old runtime can still be proved and stopped safely; the interactive agent
# entrypoint is deliberately not part of new Qwen serving releases.
_LEGACY_SOURCE_FILES = (
    "aeon/__init__.py",
    "aeon/main.py",
    "aeon/core/__init__.py",
    "aeon/core/action_schema.py",
    "aeon/core/compute_profile.py",
    "aeon/core/deploy_planner.py",
    "aeon/core/gpu.py",
    "aeon/core/gpu_queue.py",
    "aeon/core/model_catalog.py",
    "aeon/core/mtp_tuning.py",
    "aeon/core/qwen_artifact_cache.py",
    "aeon/core/qwen_capabilities.py",
    "aeon/core/qwen_fleet_runtime.py",
    "aeon/core/qwen_runtime.py",
    "aeon/core/sampling.py",
    "aeon/core/data/qwen38_mtp_selection.json",
    "aeon/core/data/qwen38_rtx5000_178_128k_release_receipt.json",
    "aeon/core/data/qwen38_rtx5000_128k_release_receipt.json",
    "aeon/core/data/qwen_runtime_capabilities.json",
    "aeon/scripts/vllm_uuid_sitecustomize.py",
    "aeon/scripts/warmup_qwen38_vllm.py",
    "aeon/scripts/qwen_remote_worker.py",
)
SOURCE_MANIFEST_FILE = "aeon/core/data/qwen_runtime_source.SHA256SUMS"

# These are the only schema-6 source closures reviewed for one-way teardown
# migration. The first already requested executable tmpfs; the second relied
# on Docker's default noexec behavior. Neither predecessor may be reused.
_LEGACY_TMPFS_PREDECESSORS = (
    (
        "f5e7a0722dceeb4c45558ad1cf5390278db4324a7d36b003077551dc7fe6c67a",
        "b319ebcd59aebd8bc74fd5a82e9d2d7b2575ab85ec3430e408167c3b0a9b4857",
        True,
    ),
    (
        "cac5152b23a87e9a406e3b12f60aa8d304e545d23d5fd9e0ff02468c8c8288e6",
        "283692460d93de68ba933ad82cd1d214265a06435a936f83c1ce328ddf454786",
        False,
    ),
)

_CLAIM_RE = re.compile(r"^gc-[A-Za-z0-9T-]{8,96}$")
_OWNER_RE = re.compile(r"^[A-Za-z0-9_.-]{8,160}$")
_UUID_RE = re.compile(r"^GPU-[A-Fa-f0-9-]{8,96}$")
_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@-]{0,255}$")
_IMAGE_ID_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_CONTAINER_ID_RE = re.compile(r"^[a-f0-9]{64}$")
_DOCKER_CPU_THERMAL_MASK_RE = re.compile(
    r"^/sys/devices/system/cpu/cpu[0-9]+/thermal_throttle$"
)
_BASELINE_MASKED_PATHS = (
    "/proc/asound",
    "/proc/acpi",
    "/proc/kcore",
    "/proc/keys",
    "/proc/latency_stats",
    "/proc/timer_list",
    "/proc/timer_stats",
    "/proc/sched_debug",
    "/proc/scsi",
    "/sys/firmware",
    "/sys/devices/virtual/powercap",
)


class QwenRuntimeError(RuntimeError):
    """A Qwen runtime failed an exact identity or lifecycle invariant."""


class QwenLeaseLostError(QwenRuntimeError):
    """The exact coordinator claim is no longer active/admissible."""


class QwenRuntimeLoadingError(QwenRuntimeError):
    """The exact container exists but its verified endpoint is still loading."""


class _ContainerIdReceiptError(QwenRuntimeError):
    """The optional Docker cidfile cannot be trusted as process identity."""


@dataclass(frozen=True)
class ArtifactIdentity:
    model_dir: Path
    manifest_sha256: str
    sha256s_sha256: str
    files: tuple[str, ...]
    total_bytes: int
    root_device: int
    root_inode: int
    file_stats: tuple[tuple[str, int, int, int, int, int, int], ...]


@dataclass(frozen=True)
class SourceIdentity:
    package_root: Path
    stage_dir: Path
    manifest_sha256: str
    manifest_bytes: bytes
    file_sha256: tuple[tuple[str, str], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validate_token(value: Any, pattern: re.Pattern[str], label: str) -> str:
    text = str(value or "")
    if not pattern.fullmatch(text):
        raise QwenRuntimeError(f"invalid {label}")
    return text


def _validate_relative_path(value: Any, label: str) -> str:
    relative = str(value or "")
    parts = PurePosixPath(relative).parts
    if (
        not parts
        or relative.startswith("/")
        or ".." in parts
        or "\x00" in relative
    ):
        raise QwenRuntimeError(f"invalid {label}")
    return relative


def _validate_run_dir(value: Any) -> Path:
    raw = str(value or "")
    prefix = f"{RUNTIME_ROOT}/aeon-qwen38-vllm-"
    path = Path(raw)
    if ".." in PurePosixPath(raw).parts:
        raise QwenRuntimeError("unexpected Qwen run directory")
    suffix = path.name.removeprefix("aeon-qwen38-vllm-")
    legacy = (
        raw.startswith(prefix)
        and path.parent == RUNTIME_ROOT
        and _OWNER_RE.fullmatch(suffix) is not None
    )
    fleet = (
        path.parent == Path("/home/aday/.local/state/fleet-compute/runs")
        and re.fullmatch(r"fr-[0-9a-f]{32}", path.name) is not None
    )
    if not legacy and not fleet:
        raise QwenRuntimeError("invalid Qwen run-directory owner")
    return path


def _validate_absolute_path(value: Any, label: str) -> Path:
    raw = str(value or "")
    path = PurePosixPath(raw)
    if (
        not raw.startswith("/")
        or not path.parts
        or ".." in path.parts
        or "\x00" in raw
        or len(raw) > 4096
    ):
        raise QwenRuntimeError(f"invalid {label}")
    return Path(raw)


def _capability_from_receipt(
    receipt: Mapping[str, Any],
    *,
    allow_retired_manifest: bool = False,
) -> QwenRuntimeCapability:
    release_gate = receipt.get("release_gate")
    if (
        release_gate is None
        and receipt.get("runtime_capability_key") == RTX5000_RELEASE_CANDIDATE_KEY
    ):
        # The first candidate launcher reached Docker create before schema 7
        # persisted this marker. The exact disabled key is sufficient only for
        # recovery/teardown of that one release-gate receipt.
        release_gate = True
    try:
        return validate_qwen_capability_receipt(
            key=receipt.get("runtime_capability_key"),
            manifest_sha256=receipt.get("runtime_capability_manifest_sha256"),
            runtime_adapter=receipt.get("runtime_adapter"),
            host=receipt.get("host"),
            physical_gpu=receipt.get("physical_gpu"),
            release_gate=False if release_gate is None else release_gate,
            allow_retired_manifest=allow_retired_manifest,
        )
    except QwenCapabilityError as exc:
        raise QwenRuntimeError("Qwen runtime capability receipt changed") from exc


def _validate_lease(
    lease: Mapping[str, Any], *, allow_retired_manifest: bool = False
) -> dict[str, Any]:
    capability = _capability_from_receipt(
        lease, allow_retired_manifest=allow_retired_manifest
    )
    host = str(lease.get("host") or "")
    if host != capability.host or capability.runtime_adapter not in {
        "local-docker",
        "remote-docker",
    }:
        raise QwenRuntimeError("Qwen lease has no reviewed runtime adapter")
    physical = lease.get("physical_gpu")
    total_mib = lease.get("memory_total_mib")
    budget = lease.get("vram_budget_gb")
    budget_mib = lease.get("vram_budget_mib")
    if (
        isinstance(physical, bool)
        or not isinstance(physical, int)
        or physical not in capability.allowed_physical_gpus
        or isinstance(total_mib, bool)
        or not isinstance(total_mib, int)
        or total_mib < capability.min_physical_vram_gb * 1024
        or not _is_finite_number(budget)
        or isinstance(budget_mib, bool)
        or not isinstance(budget_mib, int)
        or budget_mib != round(float(budget) * 1024)
        or capability.vram_budget_gb is None
        or abs(float(budget) - capability.vram_budget_gb) > 1e-9
        or float(budget) > float(total_mib) / 1024.0 - 6.0
        or lease.get("exclusive") is not True
        or lease.get("compute_profile") != capability.compute_profile
    ):
        raise QwenRuntimeError("coordinator Qwen receipt is not exact/exclusive")
    checked = {
        **dict(lease),
        "host": host,
        "expected_hostname": capability.hostname,
        "physical_gpu": physical,
        "memory_total_mib": total_mib,
        "claim_id": _validate_token(lease.get("claim_id"), _CLAIM_RE, "claim ID"),
        "owner": _validate_token(lease.get("owner"), _OWNER_RE, "lease owner"),
        "gpu_uuid": _validate_token(lease.get("gpu_uuid"), _UUID_RE, "GPU UUID"),
        "run_dir": str(_validate_run_dir(lease.get("run_dir"))),
        "vram_budget_gb": float(budget),
    }
    profile = lease_admission_profile(checked, validate_lease=False)
    if (
        profile.key != QWEN38_VLLM_PROFILE.key
        or any(
            getattr(profile, field) < getattr(QWEN38_VLLM_PROFILE, field)
            for field in (
                "min_host_memory_gb",
                "min_host_commit_gb",
                "min_disk_free_gb",
                "min_shm_free_gb",
            )
        )
        or capability.compute_profile != QWEN38_VLLM_PROFILE.key
        or capability.context_tokens < 65536
        or capability.gpu_memory_utilization is None
        or capability.max_num_seqs is None
        or capability.max_batched_tokens is None
    ):
        raise QwenRuntimeError("local Qwen lease profile changed")
    checked.update(
        {
            "vram_budget_mib": budget_mib,
            "exclusive": True,
            "compute_profile": capability.compute_profile,
            "min_host_memory_gb": float(profile.min_host_memory_gb),
            "min_host_commit_gb": float(profile.min_host_commit_gb),
            "min_disk_free_gb": float(profile.min_disk_free_gb),
            "min_shm_free_gb": float(profile.min_shm_free_gb),
        }
    )
    return checked


def lease_admission_profile(
    lease: Mapping[str, Any], *, validate_lease: bool = True
) -> ComputeProfile:
    if validate_lease:
        _validate_lease(lease)
    values: dict[str, float] = {}
    for key in (
        "min_host_memory_gb",
        "min_host_commit_gb",
        "min_disk_free_gb",
        "min_shm_free_gb",
    ):
        raw = lease.get(key)
        if not _is_finite_number(raw):
            raise QwenRuntimeError("Qwen lease lacks its durable resource profile")
        values[key] = float(raw)
    return ComputeProfile(key=QWEN38_VLLM_PROFILE.key, **values)


def runtime_state_matches_lease(
    state: Mapping[str, Any], lease: Mapping[str, Any]
) -> bool:
    try:
        if (
            type(state.get("schema_version")) is not int
            or state["schema_version"] != SCHEMA_VERSION
        ):
            return False
        checked = _validate_lease(lease, allow_retired_manifest=True)
        checked_state = _validate_lease(
            state, allow_retired_manifest=True
        )
        return (
            state.get("expected_hostname") == checked["expected_hostname"]
            and all(
                checked_state.get(key) == checked.get(key)
                for key in (
                    "host",
                    "physical_gpu",
                    "gpu_uuid",
                    "claim_id",
                    "owner",
                    "run_dir",
                    "memory_total_mib",
                    "vram_budget_mib",
                    "exclusive",
                    "compute_profile",
                    "min_host_memory_gb",
                    "min_host_commit_gb",
                    "min_disk_free_gb",
                    "min_shm_free_gb",
                    "runtime_capability_key",
                    "runtime_capability_manifest_sha256",
                    "runtime_adapter",
                    "release_gate",
                )
            )
            and abs(
                checked_state["vram_budget_gb"] - checked["vram_budget_gb"]
            )
            <= 1e-6
        )
    except (TypeError, ValueError, QwenRuntimeError):
        return False


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise QwenRuntimeError("Qwen runtime directory is not private/owned")
    os.chmod(path, 0o700)


def _private_json_read(path: Path) -> dict[str, Any] | None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink != 1
            or metadata.st_size > 262144
        ):
            raise QwenRuntimeError("runtime receipt is not a private owned file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            value = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("runtime receipt is unreadable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise QwenRuntimeError("runtime receipt is not an object")
    return value


def _private_json_write(path: Path, value: Mapping[str, Any]) -> None:
    _ensure_private_directory(path.parent)
    temp_path: str | None = None
    try:
        descriptor, temp_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(value),
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
        os.chmod(path, 0o600)
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _legacy_tmpfs_predecessor(
    source_manifest_sha256: Any,
) -> tuple[str, bool] | None:
    if not isinstance(source_manifest_sha256, str):
        return None
    matches = [
        (runtime_sha256, executable)
        for manifest_sha256, runtime_sha256, executable in _LEGACY_TMPFS_PREDECESSORS
        if source_manifest_sha256 == manifest_sha256
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _validated_tmpfs_options(state: Mapping[str, Any]) -> str:
    teardown_only = state.get("teardown_only")
    if type(teardown_only) is not bool:
        raise QwenRuntimeError("runtime teardown-only policy is invalid")
    tmpfs_options = state.get("container_tmpfs_options")
    if not isinstance(tmpfs_options, str):
        raise QwenRuntimeError("runtime tmpfs policy is invalid")
    if teardown_only:
        predecessor = _legacy_tmpfs_predecessor(state.get("source_manifest_sha256"))
        if predecessor is None:
            raise QwenRuntimeError("teardown-only source identity is unsupported")
        runtime_sha256, executable = predecessor
        if (
            type(state.get("migrated_from_schema")) is not int
            or state["migrated_from_schema"] != LEGACY_SCHEMA_VERSION
            or state.get("legacy_qwen_runtime_sha256") != runtime_sha256
            or tmpfs_options
            != _container_tmpfs_options(executable=executable)
        ):
            raise QwenRuntimeError("teardown-only migration receipt changed")
        return tmpfs_options
    if (
        "migrated_from_schema" in state
        or "legacy_qwen_runtime_sha256" in state
        or tmpfs_options != _container_tmpfs_options(executable=True)
    ):
        raise QwenRuntimeError("runtime tmpfs policy changed")
    return tmpfs_options


def _validate_legacy_source_stage(
    state: Mapping[str, Any],
    *,
    expected_runtime_sha256: str,
) -> None:
    manifest_sha256 = _validate_token(
        state.get("source_manifest_sha256"), _SHA256_RE, "legacy source identity"
    )
    run_dir = _validate_run_dir(state.get("run_dir"))
    source_dir = _validate_absolute_path(state.get("source_dir"), "legacy source path")
    if source_dir != run_dir / f"local-source-{manifest_sha256}":
        raise QwenRuntimeError("legacy source path is not run/hash bound")
    try:
        source_dir.lstat()
    except FileNotFoundError:
        if state.get("phase") == "preparing" and state.get("container_id") is None:
            return
        if state.get("phase") == "releasing":
            return
        raise QwenRuntimeError("legacy source stage is unavailable")

    manifest_path = source_dir / "SOURCE_SHA256SUMS"
    try:
        descriptor = os.open(
            manifest_path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise QwenRuntimeError("legacy source manifest is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= 65536
        ):
            raise QwenRuntimeError("legacy source manifest is unsafe")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            manifest_bytes = handle.read(65537)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise QwenRuntimeError("legacy source manifest identity changed")
    try:
        lines = manifest_bytes.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise QwenRuntimeError("legacy source manifest is malformed") from exc
    if len(lines) != len(_LEGACY_SOURCE_FILES):
        raise QwenRuntimeError("legacy source manifest file set changed")
    files: list[tuple[str, str]] = []
    for expected_relative, line in zip(_LEGACY_SOURCE_FILES, lines, strict=True):
        match = re.fullmatch(r"([a-f0-9]{64})  ([^\n]+)\n", line)
        if match is None or match.group(2) != expected_relative:
            raise QwenRuntimeError("legacy source manifest is malformed")
        files.append((expected_relative, match.group(1)))
    runtime_hashes = [
        digest
        for relative, digest in files
        if relative == "aeon/core/qwen_runtime.py"
    ]
    if runtime_hashes != [expected_runtime_sha256]:
        raise QwenRuntimeError("legacy Qwen runtime source identity changed")
    source_files = state.get("source_files")
    if (
        not isinstance(source_files, list)
        or len(source_files) != len(set(source_files))
        or set(source_files) != {*_LEGACY_SOURCE_FILES, "SOURCE_SHA256SUMS"}
    ):
        raise QwenRuntimeError("legacy runtime source file receipt changed")
    _validate_source_stage(
        SourceIdentity(
            package_root=Path("/"),
            stage_dir=source_dir,
            manifest_sha256=manifest_sha256,
            manifest_bytes=manifest_bytes,
            file_sha256=tuple(files),
        )
    )


def _migrate_legacy_runtime_state(state: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = state.get("schema_version")
    if type(schema_version) is not int:
        raise QwenRuntimeError("unsupported Qwen runtime receipt schema")
    if schema_version != LEGACY_SCHEMA_VERSION:
        return dict(state)
    if any(
        key in state
        for key in (
            "container_tmpfs_options",
            "teardown_only",
            "migrated_from_schema",
            "legacy_qwen_runtime_sha256",
        )
    ):
        raise QwenRuntimeError("legacy runtime receipt contains migration fields")
    predecessor = _legacy_tmpfs_predecessor(state.get("source_manifest_sha256"))
    if predecessor is None:
        raise QwenRuntimeError("unsupported Qwen runtime receipt schema")
    runtime_sha256, executable = predecessor
    _validate_legacy_source_stage(
        state,
        expected_runtime_sha256=runtime_sha256,
    )
    return {
        **dict(state),
        "schema_version": SCHEMA_VERSION,
        "container_tmpfs_options": _container_tmpfs_options(
            executable=executable
        ),
        "teardown_only": True,
        "migrated_from_schema": LEGACY_SCHEMA_VERSION,
        "legacy_qwen_runtime_sha256": runtime_sha256,
    }


def current_runtime_state(path: Path = RUNTIME_STATE_FILE) -> dict[str, Any] | None:
    state = _private_json_read(path)
    if state is None:
        return None
    state = _migrate_legacy_runtime_state(state)
    if (
        type(state.get("schema_version")) is not int
        or state["schema_version"] != SCHEMA_VERSION
    ):
        raise QwenRuntimeError("unsupported Qwen runtime receipt schema")
    capability = _capability_from_receipt(
        state, allow_retired_manifest=True
    )
    if (
        state.get("host") != capability.host
        or state.get("expected_hostname") != capability.hostname
        or state.get("runtime_adapter") != capability.runtime_adapter
    ):
        raise QwenRuntimeError("runtime receipt differs from its host capability")
    for key, pattern, label in (
        ("claim_id", _CLAIM_RE, "runtime claim"),
        ("owner", _OWNER_RE, "runtime owner"),
        ("gpu_uuid", _UUID_RE, "runtime GPU UUID"),
        ("container_name", _CONTAINER_RE, "container name"),
        ("image", _IMAGE_RE, "runtime image"),
        ("image_id", _IMAGE_ID_RE, "runtime image ID"),
        ("model_manifest_sha256", _SHA256_RE, "model manifest identity"),
        ("model_sha256s_sha256", _SHA256_RE, "model sums identity"),
        ("source_manifest_sha256", _SHA256_RE, "source identity"),
        ("launch_nonce", _SHA256_RE, "launch nonce"),
        ("launch_spec_sha256", _SHA256_RE, "launch specification"),
        ("wrapper_sha256", _SHA256_RE, "wrapper identity"),
        ("docker_sha256", _SHA256_RE, "Docker client identity"),
        (
            "runtime_capability_manifest_sha256",
            _SHA256_RE,
            "runtime capability manifest identity",
        ),
    ):
        _validate_token(state.get(key), pattern, label)
    run_dir = _validate_run_dir(state.get("run_dir"))
    _validate_absolute_path(state.get("model_dir"), "model path")
    source_dir = _validate_absolute_path(state.get("source_dir"), "source path")
    source_hash = str(state["source_manifest_sha256"])
    if source_dir != run_dir / f"local-source-{source_hash}":
        raise QwenRuntimeError("source path is not run/hash bound")
    if state.get("phase") not in {
        "preparing",
        "preflight",
        "launching",
        "ready",
        "releasing",
    }:
        raise QwenRuntimeError("runtime receipt phase is invalid")
    if "warmup_failure" in state:
        warmup_failure = state["warmup_failure"]
        if (
            state.get("phase") not in {"launching", "releasing"}
            or not isinstance(warmup_failure, dict)
            or type(warmup_failure.get("schema_version")) is not int
            or warmup_failure != _validated_warmup_failure(warmup_failure)
        ):
            raise QwenRuntimeError("runtime warmup failure receipt is invalid")
    # The runtime receipt is also a durable copy of the exact lease accounting.
    # Validate it through the same release-bound contract before trusting any
    # saved process/container identity.
    _validate_lease(state, allow_retired_manifest=True)
    container_id = state.get("container_id")
    if container_id is not None:
        _validate_token(container_id, _CONTAINER_ID_RE, "container ID")
    if state.get("image_id") != capability.image_id:
        raise QwenRuntimeError("runtime image is outside its capability receipt")
    for key, maximum in (
        ("model_bytes", MAX_MODEL_ARTIFACT_BYTES),
        ("image_size_bytes", MAX_IMAGE_LOGICAL_BYTES),
    ):
        value = state.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
            raise QwenRuntimeError(f"runtime receipt has invalid {key}")
    for key in ("local_port", "remote_port"):
        value = state.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 1024 <= value <= 65535:
            raise QwenRuntimeError("runtime receipt has invalid port")
    model_files = state.get("model_files")
    source_files = state.get("source_files")
    if not isinstance(model_files, list) or not model_files:
        raise QwenRuntimeError("runtime receipt has no model file set")
    if not isinstance(source_files, list) or not source_files:
        raise QwenRuntimeError("runtime receipt has no source file set")
    if len({_validate_relative_path(v, "model file") for v in model_files}) != len(model_files):
        raise QwenRuntimeError("runtime model file set has duplicates")
    if len({_validate_relative_path(v, "source file") for v in source_files}) != len(source_files):
        raise QwenRuntimeError("runtime source file set has duplicates")
    model_stats = state.get("model_file_stats")
    if not isinstance(model_stats, list) or not model_stats:
        raise QwenRuntimeError("runtime receipt has no model stat identity")
    if not isinstance(state.get("scratch_cleaned"), bool):
        raise QwenRuntimeError("runtime scratch-cleanup journal is invalid")
    _validated_tmpfs_options(state)
    if "cidfile_recovery_authorized" in state and not isinstance(
        state["cidfile_recovery_authorized"], bool
    ):
        raise QwenRuntimeError("runtime cidfile-recovery journal is invalid")
    for key, expected_type in (
        ("container_command", list),
        ("container_environment", dict),
        ("container_labels", dict),
        ("container_mounts", dict),
        ("image_base_environment", dict),
        ("image_base_labels", dict),
        ("image_base_exposed_ports", dict),
    ):
        if not isinstance(state.get(key), expected_type):
            raise QwenRuntimeError(f"runtime receipt has invalid {key}")
    image_ports = state["image_base_exposed_ports"]
    if state["phase"] == "preparing":
        if image_ports:
            raise QwenRuntimeError("preparing runtime has an image port receipt")
    elif _normalise_image_exposed_ports(image_ports) != image_ports:
        raise QwenRuntimeError("runtime image port receipt is not canonical")
    served = state.get("served_name")
    if not isinstance(served, str) or not served or len(served) > 200 or "\x00" in served:
        raise QwenRuntimeError("runtime served-model name is invalid")
    return state


def clear_runtime_state(path: Path = RUNTIME_STATE_FILE) -> None:
    state = current_runtime_state(path)
    if state is not None:
        path.unlink()


def _expected_directories(files: set[str]) -> set[str]:
    result: set[str] = set()
    for relative in files:
        parts = PurePosixPath(relative).parts[:-1]
        result.update(
            PurePosixPath(*parts[:depth]).as_posix()
            for depth in range(1, len(parts) + 1)
        )
    return result


def _exact_owned_tree(root: Path, expected_files: set[str]) -> int:
    root_metadata = root.lstat()
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or root_metadata.st_mode & 0o022
    ):
        raise QwenRuntimeError("identity-tree root is mutable or unowned")
    expected_dirs = _expected_directories(expected_files)
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    total = 0
    for path in root.rglob("*"):
        metadata = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
                raise QwenRuntimeError("identity tree contains a mutable/unowned directory")
            actual_dirs.add(relative)
        elif stat.S_ISREG(metadata.st_mode):
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
                raise QwenRuntimeError("identity tree contains a mutable/unowned file")
            actual_files.add(relative)
            total += metadata.st_size
        else:
            raise QwenRuntimeError("identity tree contains a symlink/special inode")
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise QwenRuntimeError("identity tree file set changed")
    return total


def _model_verification_sidecar(
    root: Path,
    *,
    checksummed_files: set[str],
    sha256s_sha256: str,
) -> set[str]:
    """Validate the one permitted non-payload model-tree attestation."""

    name = _MODEL_SHA256_VERIFICATION_SIDECAR
    if name in checksummed_files:
        raise QwenRuntimeError("model verification marker is part of the payload")
    path = root / name
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return set()
    except OSError as exc:
        raise QwenRuntimeError("model verification marker is unreadable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or metadata.st_size != 65
    ):
        raise QwenRuntimeError("model verification marker is unsafe")
    expected = f"{sha256s_sha256}\n".encode("ascii")
    try:
        observed = path.read_bytes()
    except OSError as exc:
        raise QwenRuntimeError("model verification marker is unreadable") from exc
    if observed != expected:
        raise QwenRuntimeError("model verification marker identity changed")
    return {name}


def load_artifact_identity(
    model_dir: Path,
    *,
    verify_payload: bool = True,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    progress_check: Callable[[], None] | None = None,
) -> ArtifactIdentity:
    expanded = model_dir.expanduser()
    metadata = expanded.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise QwenRuntimeError("canonical model is not an owned real directory")
    root = expanded.resolve(strict=True)
    manifest_path = root / "BUILD_MANIFEST.json"
    sums_path = root / "SHA256SUMS"
    for path in (
        manifest_path,
        sums_path,
        root / "config.json",
        root / "model.safetensors.index.json",
    ):
        item = path.lstat()
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_uid != os.geteuid()
            or item.st_mode & 0o022
        ):
            raise QwenRuntimeError("canonical model artifact is incomplete")
    if manifest_path.stat().st_size > 4 * 1024**2 or sums_path.stat().st_size > 16 * 1024**2:
        raise QwenRuntimeError("model release manifests are unbounded")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("model build manifest is invalid") from exc
    if manifest.get("complete") is not True or manifest.get("status") != "validated":
        raise QwenRuntimeError("model artifact is not release-validated")
    files: list[str] = []
    seen: set[str] = set()
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        match = re.fullmatch(r"([a-f0-9]{64}) [ *](.+)", line)
        if match is None:
            raise QwenRuntimeError("model checksum manifest has an invalid line")
        relative = _validate_relative_path(match.group(2), "model checksum path")
        if relative in seen:
            raise QwenRuntimeError("model checksum manifest has a duplicate path")
        candidate = root / relative
        item = candidate.lstat()
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_uid != os.geteuid()
            or item.st_mode & 0o022
        ):
            raise QwenRuntimeError("model checksum path is not an owned regular file")
        seen.add(relative)
        files.append(relative)
    if not {"BUILD_MANIFEST.json", "config.json", "model.safetensors.index.json"}.issubset(seen):
        raise QwenRuntimeError("model checksum manifest is incomplete")
    sha256s_sha256 = _sha256(sums_path)
    expected_files = {*seen, "SHA256SUMS"}
    expected_files.update(
        _model_verification_sidecar(
            root,
            checksummed_files=seen,
            sha256s_sha256=sha256s_sha256,
        )
    )
    total = _exact_owned_tree(root, expected_files)
    if not 0 < total <= MAX_MODEL_ARTIFACT_BYTES:
        raise QwenRuntimeError("model artifact exceeds its size bound")
    if verify_payload:
        command = [
            str(HOST_BASH),
            str(FLEET_LOW_PRIORITY),
            str(HOST_SHA256SUM),
            "--check",
            "--strict",
            "SHA256SUMS",
        ]
        if progress_check is None:
            result = command_runner(
                command,
                cwd=str(root),
                env=dict(HOST_LAUNCH_ENV),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1800,
            )
        else:
            progress_check()
            process = subprocess.Popen(
                command,
                cwd=str(root),
                env=dict(HOST_LAUNCH_ENV),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                while True:
                    try:
                        _output, error = process.communicate(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        progress_check()
            except BaseException:
                if process.returncode is None:
                    try:
                        process.terminate()
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate(timeout=5)
                raise
            result = subprocess.CompletedProcess(
                command, process.returncode, stdout=None, stderr=error
            )
        if result.returncode != 0:
            raise QwenRuntimeError("canonical model payload verification failed")
    return ArtifactIdentity(
        model_dir=root,
        manifest_sha256=_sha256(manifest_path),
        sha256s_sha256=sha256s_sha256,
        files=tuple(files),
        total_bytes=total,
        root_device=root.lstat().st_dev,
        root_inode=root.lstat().st_ino,
        file_stats=tuple(
            (
                relative,
                (item := (root / relative).lstat()).st_size,
                item.st_dev,
                item.st_ino,
                item.st_mtime_ns,
                item.st_ctime_ns,
                stat.S_IMODE(item.st_mode),
            )
            for relative in sorted(expected_files)
        ),
    )


def revalidate_artifact_identity(identity: ArtifactIdentity) -> None:
    """Fast post-reserve proof of the fully hashed pre-reserve model identity.

    The expensive checksum pass happens before a claim is requested.  Once the
    claim exists, exact no-follow inode, ctime, mtime, size, owner, mode and file
    set receipts prove that none of those already-hashed bytes changed while we
    move promptly to container creation.
    """

    root = identity.model_dir
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
        or metadata.st_dev != identity.root_device
        or metadata.st_ino != identity.root_inode
    ):
        raise QwenRuntimeError("canonical model root changed after verification")
    expected = {relative for relative, *_rest in identity.file_stats}
    payload_files = {*identity.files, "SHA256SUMS"}
    if expected != payload_files and expected != {
        *payload_files,
        _MODEL_SHA256_VERIFICATION_SIDECAR,
    }:
        raise QwenRuntimeError("canonical model receipt is internally inconsistent")
    _exact_owned_tree(root, expected)
    for receipt in identity.file_stats:
        relative, size, device, inode, mtime_ns, ctime_ns, mode = receipt
        item = (root / relative).lstat()
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_uid != os.geteuid()
            or item.st_mode & 0o022
            or (
                item.st_size,
                item.st_dev,
                item.st_ino,
                item.st_mtime_ns,
                item.st_ctime_ns,
                stat.S_IMODE(item.st_mode),
            )
            != (size, device, inode, mtime_ns, ctime_ns, mode)
        ):
            raise QwenRuntimeError("canonical model bytes changed after verification")
    if (
        _sha256(root / "BUILD_MANIFEST.json") != identity.manifest_sha256
        or _sha256(root / "SHA256SUMS") != identity.sha256s_sha256
    ):
        raise QwenRuntimeError("canonical model manifest identity changed")
    _model_verification_sidecar(
        root,
        checksummed_files=set(identity.files),
        sha256s_sha256=identity.sha256s_sha256,
    )


def local_image_id(
    image: str,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    image = _validate_token(image, _IMAGE_RE, "runtime image")
    result = command_runner(
        _docker_command("image", "inspect", "--format", "{{.Id}}", image),
        env=_docker_cli_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        raise QwenRuntimeError("release-bound local Qwen image is unavailable")
    return _validate_token(result.stdout.strip(), _IMAGE_ID_RE, "runtime image ID")


def local_image_size(
    image_id: str,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    image_id = _validate_token(image_id, _IMAGE_ID_RE, "runtime image ID")
    result = command_runner(
        _docker_command("image", "inspect", "--format", "{{.Size}}", image_id),
        env=_docker_cli_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
    )
    try:
        size = int(result.stdout.strip()) if result.returncode == 0 else 0
    except ValueError:
        size = 0
    if not 0 < size <= MAX_IMAGE_LOGICAL_BYTES:
        raise QwenRuntimeError("release-bound image has an invalid size")
    return size


def _image_config(
    image_id: str,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    result = command_runner(
        _docker_command("image", "inspect", image_id),
        env=_docker_cli_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
    )
    try:
        payload = json.loads(result.stdout)
        item = payload[0]
        config = item["Config"]
    except (ValueError, json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        raise QwenRuntimeError("runtime image configuration is unreadable") from exc
    if result.returncode != 0 or item.get("Id") != image_id or not isinstance(config, dict):
        raise QwenRuntimeError("runtime image configuration changed")
    return config


def _executable_receipt(
    path: Path, *, allowed_uids: set[int], label: str
) -> dict[str, Any]:
    """Hash one exact no-follow executable plus release-critical stat identity."""

    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid not in allowed_uids
        or metadata.st_nlink != 1
        or not metadata.st_mode & stat.S_IXUSR
        or metadata.st_mode & 0o022
    ):
        raise QwenRuntimeError(f"{label} is mutable, linked, or unowned")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "size": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode": stat.S_IMODE(metadata.st_mode),
        "nlink": metadata.st_nlink,
    }


def _system_executable_receipt(path: Path, *, label: str) -> dict[str, Any]:
    """Attest a root/system executable, including distro multicall symlinks.

    Ubuntu's Rust coreutils package exposes commands such as ``/usr/bin/nice``
    as root-owned symlinks into one hard-linked multicall binary. That layout is
    immutable to ``aday`` and is safe when both the link and resolved target are
    bound. User-owned launch wrappers continue to require one regular link.
    """

    allowed_uids = set(SYSTEM_EXECUTABLE_UIDS)
    metadata = path.lstat()
    if stat.S_ISREG(metadata.st_mode):
        if (
            metadata.st_uid not in allowed_uids
            or metadata.st_nlink < 1
            or not metadata.st_mode & stat.S_IXUSR
            or metadata.st_mode & 0o022
        ):
            raise QwenRuntimeError(f"{label} is mutable or unowned")
        return {
            "kind": "regular",
            "path": str(path),
            "sha256": _sha256(path),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "mode": stat.S_IMODE(metadata.st_mode),
            "nlink": metadata.st_nlink,
        }
    if (
        not stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid not in allowed_uids
        or metadata.st_nlink != 1
    ):
        raise QwenRuntimeError(f"{label} is mutable, linked, or unowned")
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid not in allowed_uids
        or parent.st_mode & 0o022
    ):
        raise QwenRuntimeError(f"{label} parent is mutable or unowned")
    link_target = os.readlink(path)
    if not link_target or "\x00" in link_target or len(link_target) > 4096:
        raise QwenRuntimeError(f"{label} symlink target is malformed")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise QwenRuntimeError(f"{label} symlink target is unavailable") from exc
    target = resolved.lstat()
    if (
        not stat.S_ISREG(target.st_mode)
        or target.st_uid not in allowed_uids
        or target.st_nlink < 1
        or not target.st_mode & stat.S_IXUSR
        or target.st_mode & 0o022
    ):
        raise QwenRuntimeError(f"{label} target is mutable or unowned")
    return {
        "kind": "symlink",
        "path": str(path),
        "link_target": link_target,
        "link_device": metadata.st_dev,
        "link_inode": metadata.st_ino,
        "link_mtime_ns": metadata.st_mtime_ns,
        "link_ctime_ns": metadata.st_ctime_ns,
        "link_uid": metadata.st_uid,
        "link_gid": metadata.st_gid,
        "link_nlink": metadata.st_nlink,
        "parent_device": parent.st_dev,
        "parent_inode": parent.st_ino,
        "parent_mode": stat.S_IMODE(parent.st_mode),
        "resolved_path": str(resolved),
        "sha256": _sha256(resolved),
        "target_device": target.st_dev,
        "target_inode": target.st_ino,
        "target_size": target.st_size,
        "target_mtime_ns": target.st_mtime_ns,
        "target_ctime_ns": target.st_ctime_ns,
        "target_uid": target.st_uid,
        "target_gid": target.st_gid,
        "target_mode": stat.S_IMODE(target.st_mode),
        "target_nlink": target.st_nlink,
    }


def _docker_command(*arguments: str) -> list[str]:
    """Invoke the exact policy wrapper through the exact host shell."""

    return [str(HOST_BASH), str(DOCKER), *arguments]


def _docker_cli_environment() -> dict[str, str]:
    """Return one fail-closed Docker CLI routing environment for every phase."""

    _ensure_private_directory(RUNTIME_ROOT)
    config_dir = RUNTIME_ROOT / DOCKER_CONFIG_DIRNAME
    _ensure_private_directory(config_dir)
    try:
        with os.scandir(config_dir) as entries:
            if next(entries, None) is not None:
                raise QwenRuntimeError("private Docker CLI configuration is not empty")
    except OSError as exc:
        raise QwenRuntimeError("private Docker CLI configuration is unreadable") from exc
    return {
        **HOST_LAUNCH_ENV,
        # Pin the only reviewed local daemon. Supplying a complete environment
        # also drops inherited DOCKER_CONTEXT/TLS variables, while the exact
        # empty config directory prevents a mutable currentContext/proxy config.
        "DOCKER_HOST": DOCKER_HOST,
        "DOCKER_CONFIG": str(config_dir),
    }


def low_priority_wrapper_sha256() -> str:
    return _canonical_sha256(
        {
            "wrapper": _executable_receipt(
                FLEET_LOW_PRIORITY,
                allowed_uids={os.geteuid()},
                label="fleet-low-priority wrapper",
            ),
            # The wrapper is invoked through HOST_BASH and receives a fixed
            # PATH, so these are the only host scheduling executables it can
            # resolve before exec. Bind every transitive executable identity.
            "bash": _system_executable_receipt(HOST_BASH, label="host Bash"),
            "nice": _system_executable_receipt(HOST_NICE, label="host nice"),
            "ionice": _system_executable_receipt(HOST_IONICE, label="host ionice"),
            "sha256sum": _system_executable_receipt(
                HOST_SHA256SUM, label="host sha256sum"
            ),
            "python": _system_executable_receipt(HOST_PYTHON, label="host Python"),
            "path": HOST_LAUNCH_ENV["PATH"],
        }
    )


def docker_client_sha256() -> str:
    parent = DOCKER.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or parent.st_mode & 0o022
    ):
        raise QwenRuntimeError("Docker client wrapper parent is mutable or unsafe")
    return _canonical_sha256(
        {
            "wrapper": _executable_receipt(
                DOCKER,
                allowed_uids={os.geteuid()},
                label="Docker client wrapper",
            ),
            "real_client": _executable_receipt(
                REAL_DOCKER,
                allowed_uids=set(SYSTEM_EXECUTABLE_UIDS),
                label="real Docker client",
            ),
            "bash": _executable_receipt(
                HOST_BASH, allowed_uids=set(SYSTEM_EXECUTABLE_UIDS), label="host Bash"
            ),
            "daemon_endpoint": DOCKER_HOST,
            "config_policy": {
                "path": str(RUNTIME_ROOT / DOCKER_CONFIG_DIRNAME),
                "required_empty": True,
            },
        }
    )


def _source_identity(package_root: Path, run_dir: Path) -> SourceIdentity:
    root_metadata = package_root.lstat()
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or root_metadata.st_mode & 0o022
    ):
        raise QwenRuntimeError("Aeon source root is mutable, linked, or unowned")
    root = package_root.resolve(strict=True)
    expected_dirs = _expected_directories(set(SOURCE_FILES))
    for relative in expected_dirs:
        metadata = (root / relative).lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            raise QwenRuntimeError("runtime source directory is mutable or unowned")
    files: list[tuple[str, str]] = []
    for relative in SOURCE_FILES:
        candidate = root / relative
        metadata = candidate.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            raise QwenRuntimeError("runtime source is not immutable/owned")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise QwenRuntimeError("runtime source escaped its package root") from exc
        files.append((relative, _sha256(resolved)))
    generated_manifest = "".join(
        f"{digest}  {relative}\n" for relative, digest in files
    ).encode("utf-8")
    manifest_path = root / SOURCE_MANIFEST_FILE
    manifest_metadata = manifest_path.lstat()
    if (
        not stat.S_ISREG(manifest_metadata.st_mode)
        or manifest_metadata.st_uid != os.geteuid()
        or manifest_metadata.st_mode & 0o022
        or manifest_metadata.st_size > 64 * 1024
    ):
        raise QwenRuntimeError("runtime source manifest is mutable or unowned")
    manifest = manifest_path.read_bytes()
    if manifest != generated_manifest:
        raise QwenRuntimeError("runtime source manifest is stale")
    identity = hashlib.sha256(manifest).hexdigest()
    return SourceIdentity(
        package_root=root,
        stage_dir=run_dir / f"local-source-{identity}",
        manifest_sha256=identity,
        manifest_bytes=manifest,
        file_sha256=tuple(files),
    )


def _write_new_private_file(path: Path, source: Path | None, payload: bytes | None) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            if source is not None:
                with source.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, handle, length=1024 * 1024)
            else:
                handle.write(payload or b"")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_source_stage(identity: SourceIdentity) -> None:
    metadata = identity.stage_dir.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise QwenRuntimeError("source stage is not private/owned")
    expected = {relative for relative, _digest in identity.file_sha256}
    expected.add("SOURCE_SHA256SUMS")
    _exact_owned_tree(identity.stage_dir, expected)
    if (identity.stage_dir / "SOURCE_SHA256SUMS").read_bytes() != identity.manifest_bytes:
        raise QwenRuntimeError("source-stage manifest changed")
    for relative, digest in identity.file_sha256:
        if _sha256(identity.stage_dir / relative) != digest:
            raise QwenRuntimeError("source-stage payload changed")


def _prepare_source_stage(
    package_root: Path,
    run_dir: Path,
    *,
    expected_identity: SourceIdentity | None = None,
) -> SourceIdentity:
    run_dir = _validate_run_dir(run_dir)
    _ensure_private_directory(RUNTIME_ROOT)
    _ensure_private_directory(run_dir)
    identity = _source_identity(package_root, run_dir)
    if expected_identity is not None and identity != expected_identity:
        raise QwenRuntimeError("runtime source changed after receipt creation")
    if identity.stage_dir.exists() or identity.stage_dir.is_symlink():
        _validate_source_stage(identity)
        return identity
    os.mkdir(identity.stage_dir, mode=0o700)
    for relative, _digest in identity.file_sha256:
        destination = identity.stage_dir / relative
        parent = destination.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        current = parent
        while current != identity.stage_dir:
            metadata = current.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o077
            ):
                raise QwenRuntimeError("source-stage parent is unsafe")
            current = current.parent
        _write_new_private_file(destination, identity.package_root / relative, None)
    _write_new_private_file(
        identity.stage_dir / "SOURCE_SHA256SUMS", None, identity.manifest_bytes
    )
    _validate_source_stage(identity)
    return identity


def cleanup_local_source_stage(identity: Mapping[str, Any]) -> bool:
    run_dir = _validate_run_dir(identity.get("run_dir"))
    source_hash = _validate_token(
        identity.get("source_manifest_sha256"), _SHA256_RE, "source identity"
    )
    stage = run_dir / f"local-source-{source_hash}"
    allowed_raw = identity.get("source_files")
    if not isinstance(allowed_raw, list) or not allowed_raw:
        return False
    allowed = {_validate_relative_path(value, "source cleanup path") for value in allowed_raw}
    try:
        metadata = stage.lstat()
    except FileNotFoundError:
        return True
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        return False
    files: list[Path] = []
    directories: list[Path] = []

    def inspect(directory: Path, relative: PurePosixPath) -> bool:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    child = Path(entry.path)
                    child_relative = (relative / entry.name).as_posix()
                    item = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(item.st_mode):
                        if item.st_uid != os.geteuid() or item.st_mode & 0o077:
                            return False
                        if not inspect(child, relative / entry.name):
                            return False
                        directories.append(child)
                    elif stat.S_ISREG(item.st_mode):
                        if (
                            item.st_uid != os.geteuid()
                            or item.st_mode & 0o077
                            or child_relative not in allowed
                        ):
                            return False
                        files.append(child)
                    else:
                        return False
        except OSError:
            return False
        return True

    if not inspect(stage, PurePosixPath()):
        return False
    try:
        for path in files:
            path.unlink()
        for path in directories:
            path.rmdir()
        stage.rmdir()
    except OSError:
        return False
    return True


def _unique_environment(values: Any, label: str) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, list):
        raise QwenRuntimeError(f"{label} environment is not a list")
    result: dict[str, str] = {}
    for raw in values:
        if not isinstance(raw, str) or "=" not in raw or "\x00" in raw:
            raise QwenRuntimeError(f"{label} environment contains an invalid item")
        key, value = raw.split("=", 1)
        if not key or key in result:
            raise QwenRuntimeError(f"{label} environment contains a duplicate key")
        result[key] = value
    return result


def _normalise_image_exposed_ports(value: Any) -> dict[str, dict[str, Any]]:
    if (
        not isinstance(value, dict)
        or set(value) != QWEN_IMAGE_EXPOSED_PORTS
        or any(value[port] != {} for port in QWEN_IMAGE_EXPOSED_PORTS)
    ):
        raise QwenRuntimeError("release image exposed-port receipt changed")
    return {port: {} for port in sorted(QWEN_IMAGE_EXPOSED_PORTS)}


def _normalise_image_config(
    config: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, Any]]]:
    if config.get("Volumes") not in (None, {}):
        # Any Dockerfile VOLUME would create writable anonymous storage outside
        # the exact tmpfs/log bounds even with a read-only root filesystem.
        raise QwenRuntimeError("release image declares unmanaged volumes")
    exposed_ports = _normalise_image_exposed_ports(config.get("ExposedPorts"))
    if (
        config.get("WorkingDir") not in (None, "")
        or config.get("Healthcheck") is not None
        or config.get("StopSignal") not in (None, "")
        or config.get("Shell") is not None
        or config.get("OnBuild") is not None
    ):
        raise QwenRuntimeError("release image has unreceipted runtime defaults")
    environment = _unique_environment(config.get("Env"), "image")
    labels_raw = config.get("Labels") or {}
    if not isinstance(labels_raw, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in labels_raw.items()
    ):
        raise QwenRuntimeError("image labels are malformed")
    return environment, dict(labels_raw), exposed_ports


def verify_coordinator_lease(
    lease: Mapping[str, Any],
    *,
    coord_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Revalidate exact coordinator claim and host admission on verified .177."""

    checked = _validate_lease(lease)
    capability = _capability_from_receipt(checked)
    coord_runner = coord_runner or _coord
    try:
        result = coord_runner("status", "--json", check=False)
    except Exception as exc:
        raise QwenRuntimeError("coordinator status transport is unavailable") from exc
    if result.returncode != 0:
        raise QwenRuntimeError("coordinator status recheck failed")
    try:
        inventory = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("coordinator status recheck was malformed") from exc
    if not isinstance(inventory, list):
        raise QwenRuntimeError("coordinator status recheck is not an inventory")
    targets = [
        item
        for item in inventory
        if isinstance(item, dict)
        and item.get("host") == checked["host"]
        and not isinstance(item.get("physical_gpu"), bool)
        and isinstance(item.get("physical_gpu"), int)
        and item.get("physical_gpu") == checked["physical_gpu"]
    ]
    if len(targets) != 1:
        raise QwenLeaseLostError("coordinator no longer exposes the exact Qwen GPU")
    target = targets[0]
    target_total_mib = target.get("memory_total_mib")
    if (
        target.get("uuid") != checked["gpu_uuid"]
        or target.get("acl") != "OPEN"
        or target.get("state") not in {"RESERVED", "RESERVED_RUNNING"}
        or target.get("vast_watchdog_active") is not True
        or target.get("physical_gpu") not in capability.allowed_physical_gpus
        or isinstance(target_total_mib, bool)
        or not isinstance(target_total_mib, int)
        or target_total_mib != checked["memory_total_mib"]
    ):
        raise QwenLeaseLostError("Qwen GPU is no longer exactly leased and admissible")
    claims = target.get("claims")
    if not isinstance(claims, list):
        raise QwenRuntimeError("coordinator target claim list is malformed")
    matches = [
        claim
        for claim in claims
        if isinstance(claim, dict) and claim.get("claim_id") == checked["claim_id"]
    ]
    if len(matches) != 1:
        raise QwenLeaseLostError("the exact Qwen claim is no longer active")
    claim = matches[0]
    claim_budget_mib = claim.get("vram_budget_mib")
    if (
        claim.get("owner") != checked["owner"]
        or claim.get("run_dir") != checked["run_dir"]
        or claim.get("gpu_uuid") != checked["gpu_uuid"]
        or isinstance(claim_budget_mib, bool)
        or not isinstance(claim_budget_mib, int)
        or claim_budget_mib != round(checked["vram_budget_gb"] * 1024)
        or not _coordinator_status_claim_is_exclusive(claim.get("exclusive"))
    ):
        raise QwenLeaseLostError("coordinator Qwen claim identity changed")
    floors = {
        "host_memory_available_mib": QWEN38_VLLM_PROFILE.min_host_memory_gb * 1024,
        "host_commit_headroom_mib": QWEN38_VLLM_PROFILE.min_host_commit_gb * 1024,
        "host_disk_available_mib": QWEN38_VLLM_PROFILE.min_disk_free_gb * 1024,
        "host_shm_available_mib": QWEN38_VLLM_PROFILE.min_shm_free_gb * 1024,
    }
    for key, minimum in floors.items():
        value = target.get(key)
        if not _is_finite_number(value):
            raise QwenLeaseLostError("coordinator Qwen resource receipt is incomplete")
        if float(value) + 1e-6 < minimum:
            raise QwenLeaseLostError("coordinator Qwen host resources fell below profile")
    return checked


def _planner_contract(
    deploy_environment: Mapping[str, Any],
    lease: Mapping[str, Any],
    artifact: ArtifactIdentity,
    image_id: str,
    package_root: Path,
    *,
    container_name: str,
    port: int,
) -> tuple[list[str], str]:
    """Build the release-bound vLLM argv and served name from one solo plan."""

    try:
        plan = json.loads(str(deploy_environment["AEON_DEPLOY_PLAN"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("Qwen deployment plan is malformed") from exc
    if (
        not isinstance(plan, dict)
        or plan.get("tier") != "solo"
        or plan.get("image") not in {image_id, deploy_environment.get("AEON_RUNTIME_IMAGE_REF")}
        and plan.get("image") != "aeon_vllm:latest"
        or plan.get("container_name") != container_name
        or plan.get("health_port") != port
        or not isinstance(plan.get("nodes"), list)
        or len(plan["nodes"]) != 1
    ):
        raise QwenRuntimeError("Qwen deployment plan is not the one-node release")
    node = plan["nodes"][0]
    if not isinstance(node, dict):
        raise QwenRuntimeError("Qwen deployment node is malformed")
    devices = str(node.get("devices") or "")
    if devices not in {"0", lease["gpu_uuid"]}:
        raise QwenRuntimeError("deployment plan is not bound to the leased UUID template")
    try:
        node_port = int(node["port"])
        context = int(node["ctx"])
        cpu_offload = float(node.get("cpu_offload_gib") or 0)
        utility = float(deploy_environment["AEON_GPU_MEM_UTIL"])
        max_sequences = int(deploy_environment["AEON_MAX_NUM_SEQS"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QwenRuntimeError("Qwen deployment values are malformed") from exc
    capability = _capability_from_receipt(lease)
    if (
        node_port != port
        or context != capability.context_tokens
        or not math.isfinite(cpu_offload)
        or cpu_offload != 0
        or not math.isfinite(utility)
        or capability.gpu_memory_utilization is None
        or abs(utility - capability.gpu_memory_utilization) > 1e-9
        or capability.max_num_seqs is None
        or max_sequences != capability.max_num_seqs
        or not _is_finite_number(lease.get("vram_budget_gb"))
        or capability.vram_budget_gb is None
        or abs(float(lease["vram_budget_gb"]) - capability.vram_budget_gb) > 1e-9
        or str(deploy_environment.get("AEON_LLM_VRAM_BUDGET_GB") or "")
        != f"{capability.vram_budget_gb:g}"
    ):
        raise QwenRuntimeError("Qwen deployment values are outside the release profile")
    served = str(deploy_environment.get("AEON_SERVED_NAME") or "")
    if not served or len(served) > 200 or "\x00" in served:
        raise QwenRuntimeError("Qwen served-model name is invalid")
    attention = str(deploy_environment.get("AEON_VLLM_ATTENTION_BACKEND") or "")
    kv_dtype = str(deploy_environment.get("AEON_KV_QUANT") or "")
    method = str(deploy_environment.get("AEON_MTP_METHOD") or "")
    if (
        served != "Qwen3.8-27B-ARA-NVFP4-MTP"
        or attention != QWEN_RELEASE_ATTENTION_BACKEND
        or kv_dtype != "fp8_per_token_head"
        or str(plan.get("entry_name") or "") != "Qwen3.8-27B-ARA-NVFP4-MTP"
    ):
        raise QwenRuntimeError("Qwen release identity/options changed")
    manifest_rel = _validate_relative_path(
        deploy_environment.get("AEON_MTP_SELECTION_MANIFEST"), "MTP manifest"
    )
    selection_path = package_root / "aeon/core" / manifest_rel
    try:
        from .mtp_tuning import (
            PACKAGED_SELECTION_BENCHMARK_SCRIPT_SHA256,
            PACKAGED_SELECTION_SUITE_SHA256,
            PACKAGED_SELECTION_SUITE_VERSION,
            load_selection,
        )

        selected_k, _selection = load_selection(
            selection_path,
            expected_entry=str(plan.get("entry_name") or ""),
            expected_model_build_sha256=artifact.manifest_sha256,
            expected_sha256s_sha256=artifact.sha256s_sha256,
            expected_image_id=image_id,
            expected_attention_backend=attention,
            expected_kv_cache_dtype=kv_dtype,
            expected_suite_version=PACKAGED_SELECTION_SUITE_VERSION,
            expected_suite_sha256=PACKAGED_SELECTION_SUITE_SHA256,
            expected_benchmark_script_sha256=(
                PACKAGED_SELECTION_BENCHMARK_SCRIPT_SHA256
            ),
        )
        declared_k = int(deploy_environment.get("AEON_MTP_NMAX") or 0)
    except Exception as exc:
        raise QwenRuntimeError("Qwen MTP release selection is invalid") from exc
    if method != "mtp" or selected_k != declared_k or selected_k != 3:
        raise QwenRuntimeError("Qwen MTP selection differs from the release")
    command = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "/models",
        "--served-model-name",
        served,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        f"{utility:g}",
        "--attention-backend",
        attention,
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-model-len",
        str(context),
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
        "--reasoning-parser",
        "qwen3",
        "--structured-outputs-config.enable_in_reasoning=False",
        "--max-num-seqs",
        str(max_sequences),
    ]
    batched = str(deploy_environment.get("AEON_MAX_NUM_BATCHED") or "")
    if capability.max_batched_tokens is None or batched != str(
        capability.max_batched_tokens
    ):
        raise QwenRuntimeError("Qwen batched-token limit differs from the release")
    command += ["--max-num-batched-tokens", batched]
    command += [
        "--speculative-config",
        json.dumps(
            {"method": method, "num_speculative_tokens": selected_k},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--kv-cache-dtype",
        kv_dtype,
    ]
    return command, served


def _mount_receipt(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise QwenRuntimeError("container bind source is mutable or unowned")
    if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
        raise QwenRuntimeError("container bind source is not a real inode")
    return {
        "source": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
    }


def _container_contract(
    *,
    lease: Mapping[str, Any],
    deploy_environment: Mapping[str, Any],
    artifact: ArtifactIdentity,
    source: SourceIdentity,
    image_id: str,
    image_config: Mapping[str, Any],
    package_root: Path,
    container_name: str,
    port: int,
    launch_nonce: str,
) -> dict[str, Any]:
    checked = _validate_lease(lease)
    capability = _capability_from_receipt(checked)
    if image_id != capability.image_id:
        raise QwenRuntimeError("container image is outside its runtime capability")
    command, served = _planner_contract(
        deploy_environment,
        checked,
        artifact,
        image_id,
        package_root,
        container_name=container_name,
        port=port,
    )
    base_environment, base_labels, base_exposed_ports = _normalise_image_config(
        image_config
    )
    runtime_port = f"{port}/tcp"
    if runtime_port in base_exposed_ports:
        raise QwenRuntimeError("runtime API port overlaps the inherited image port")
    planned_budget = float(checked["vram_budget_gb"])
    overrides = {
        "GPU_AGENT_CLAIM_ID": str(checked["claim_id"]),
        "GPU_LEASE_OWNER": str(checked["owner"]),
        "GPU_LEASE_RUN_DIR": str(checked["run_dir"]),
        "CUDA_VISIBLE_DEVICES": str(checked["gpu_uuid"]),
        "GPU_PLANNED_VRAM_GB": f"{planned_budget:g}",
        "GPU_RESERVE_GB": "6",
        "GPU_LEASE_EXCLUSIVE": "1",
        "SPT_NOENV": "1",
        "PYTHONPATH": "/workspace/aeon_runtime",
        "HOME": "/workspace/cache/home",
        "HF_HOME": "/workspace/cache/huggingface",
        "TRANSFORMERS_CACHE": "/workspace/cache/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/workspace/cache/home/.cache",
        "TRITON_CACHE_DIR": "/workspace/cache/triton",
        "VLLM_CACHE_ROOT": "/workspace/cache/vllm",
        "FLASHINFER_WORKSPACE_DIR": "/workspace/cache/home/.cache/flashinfer",
        "TORCHINDUCTOR_CACHE_DIR": "/workspace/cache/torchinductor",
        # The cache root is itself the private owner-only bounded tmpfs.  A
        # nested path would not exist on a fresh mount and Python imports use
        # tempfile before any application bootstrap can create it.
        "TMPDIR": QWEN_CONTAINER_TMPDIR,
        "MAX_JOBS": "4",
        "NVCC_THREADS": "1",
        "AEON_MODEL_MANIFEST_SHA256": artifact.manifest_sha256,
        "AEON_MODEL_SHA256S_SHA256": artifact.sha256s_sha256,
        "AEON_RUNTIME_CAPABILITY_KEY": capability.key,
        "AEON_RUNTIME_CAPABILITY_MANIFEST_SHA256": str(
            checked["runtime_capability_manifest_sha256"]
        ),
    }
    environment = {**base_environment, **overrides}
    labels = {
        **base_labels,
        "owner": "aday",
        "com.bc_aeon.component": "qwen38-vllm",
        "com.bc_aeon.claim": str(checked["claim_id"]),
        "com.bc_aeon.launch-nonce": launch_nonce,
        "com.bc_aeon.runtime-capability": capability.key,
        "com.bc_aeon.runtime-capability-manifest": str(
            checked["runtime_capability_manifest_sha256"]
        ),
    }
    sitecustomize = source.stage_dir / "aeon/scripts/vllm_uuid_sitecustomize.py"
    mounts = {
        "/usr/local/bin/fleet-low-priority": _mount_receipt(FLEET_LOW_PRIORITY),
        "/workspace/aeon_runtime/sitecustomize.py": _mount_receipt(sitecustomize),
        "/models": _mount_receipt(artifact.model_dir),
    }
    receipt_core = {
        "image_id": image_id,
        "user": f"{os.geteuid()}:{os.getegid()}",
        "entrypoint": ["/usr/local/bin/fleet-low-priority"],
        "command": command,
        "environment": environment,
        "labels": labels,
        "runtime_capability_key": capability.key,
        "runtime_capability_manifest_sha256": str(
            checked["runtime_capability_manifest_sha256"]
        ),
        "runtime_adapter": capability.runtime_adapter,
        "image_base_exposed_ports": base_exposed_ports,
        "mounts": mounts,
        "gpu_uuid": str(checked["gpu_uuid"]),
        "port": int(port),
        "shm_bytes": QWEN_CONTAINER_SHM_BYTES,
        "cache_bytes": QWEN_RUNTIME_CACHE_BYTES,
        "tmpfs_options": _container_tmpfs_options(executable=True),
        "network_mode": "bridge",
        "log_driver": "local",
        "log_options": {"max-file": "3", "max-size": "10m"},
    }
    digest = _canonical_sha256(receipt_core)
    labels["com.bc_aeon.launch-spec"] = digest
    receipt_core["labels"] = labels
    receipt_core["launch_spec_sha256"] = digest
    receipt_core["served_name"] = served
    receipt_core["image_base_environment"] = base_environment
    receipt_core["image_base_labels"] = base_labels
    return receipt_core


def _base_runtime_state(
    lease: Mapping[str, Any],
    *,
    container_name: str,
    image: str,
    image_id: str,
    image_size_bytes: int,
    artifact: ArtifactIdentity,
    source: SourceIdentity,
    port: int,
    launch_nonce: str,
    served_name: str,
    phase: str = "preparing",
) -> dict[str, Any]:
    checked = _validate_lease(lease)
    capability = _capability_from_receipt(checked)
    if image_id != capability.image_id:
        raise QwenRuntimeError("runtime image is outside its capability receipt")
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "host": capability.host,
        "expected_hostname": capability.hostname,
        "physical_gpu": checked["physical_gpu"],
        "gpu_uuid": checked["gpu_uuid"],
        "claim_id": checked["claim_id"],
        "owner": checked["owner"],
        "run_dir": checked["run_dir"],
        "memory_total_mib": checked["memory_total_mib"],
        "vram_budget_gb": checked["vram_budget_gb"],
        "vram_budget_mib": checked["vram_budget_mib"],
        "exclusive": True,
        "compute_profile": checked["compute_profile"],
        "min_host_memory_gb": checked["min_host_memory_gb"],
        "min_host_commit_gb": checked["min_host_commit_gb"],
        "min_disk_free_gb": checked["min_disk_free_gb"],
        "min_shm_free_gb": checked["min_shm_free_gb"],
        "runtime_capability_key": capability.key,
        "runtime_capability_manifest_sha256": checked[
            "runtime_capability_manifest_sha256"
        ],
        "runtime_adapter": capability.runtime_adapter,
        "release_gate": checked.get("release_gate", False),
        "container_name": _validate_token(container_name, _CONTAINER_RE, "container name"),
        "container_id": None,
        "container_pid": None,
        "image": _validate_token(image, _IMAGE_RE, "runtime image"),
        "image_id": _validate_token(image_id, _IMAGE_ID_RE, "runtime image ID"),
        "image_size_bytes": int(image_size_bytes),
        "model_dir": str(artifact.model_dir),
        "model_manifest_sha256": artifact.manifest_sha256,
        "model_sha256s_sha256": artifact.sha256s_sha256,
        "model_files": list(artifact.files),
        "model_bytes": artifact.total_bytes,
        "model_root_device": artifact.root_device,
        "model_root_inode": artifact.root_inode,
        "model_file_stats": [list(item) for item in artifact.file_stats],
        "source_dir": str(source.stage_dir),
        "source_manifest_sha256": source.manifest_sha256,
        "source_files": [
            *(relative for relative, _digest in source.file_sha256),
            "SOURCE_SHA256SUMS",
        ],
        "launch_nonce": launch_nonce,
        "launch_spec_sha256": _canonical_sha256(
            {"claim": checked["claim_id"], "nonce": launch_nonce, "phase": "preparing"}
        ),
        "wrapper_sha256": low_priority_wrapper_sha256(),
        "docker_sha256": docker_client_sha256(),
        "local_port": int(port),
        "remote_port": int(port),
        "served_name": served_name,
        "container_command": [],
        "container_environment": {},
        "container_labels": {},
        "container_mounts": {},
        "image_base_environment": {},
        "image_base_labels": {},
        "image_base_exposed_ports": {},
        "container_tmpfs_options": _container_tmpfs_options(executable=True),
        "teardown_only": False,
        "scratch_cleaned": False,
        "cidfile_recovery_authorized": False,
        "updated_at": time.time(),
    }


def _apply_contract(state: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(state),
        "container_command": list(contract["command"]),
        "container_environment": dict(contract["environment"]),
        "container_labels": dict(contract["labels"]),
        "container_mounts": dict(contract["mounts"]),
        "image_base_environment": dict(contract["image_base_environment"]),
        "image_base_labels": dict(contract["image_base_labels"]),
        "image_base_exposed_ports": dict(contract["image_base_exposed_ports"]),
        "container_tmpfs_options": str(contract["tmpfs_options"]),
        "teardown_only": False,
        "runtime_capability_key": str(contract["runtime_capability_key"]),
        "runtime_capability_manifest_sha256": str(
            contract["runtime_capability_manifest_sha256"]
        ),
        "runtime_adapter": str(contract["runtime_adapter"]),
        "launch_spec_sha256": str(contract["launch_spec_sha256"]),
        "served_name": str(contract["served_name"]),
        "phase": "preflight",
        "updated_at": time.time(),
    }


def build_local_state(
    lease: Mapping[str, Any],
    *,
    container_name: str,
    image: str,
    image_id: str,
    artifact: ArtifactIdentity,
    source: SourceIdentity,
    port: int,
    served_name: str,
    image_size_bytes: int | None = None,
    launch_nonce: str | None = None,
    phase: str = "preparing",
) -> dict[str, Any]:
    """Build a receipt skeleton; launch code adds the canonical Docker contract."""

    return _base_runtime_state(
        lease,
        container_name=container_name,
        image=image,
        image_id=image_id,
        image_size_bytes=image_size_bytes or local_image_size(image_id),
        artifact=artifact,
        source=source,
        port=port,
        launch_nonce=launch_nonce or secrets.token_hex(32),
        served_name=served_name,
        phase=phase,
    )


def _read_meminfo(path: Path = Path("/proc/meminfo")) -> tuple[int, int]:
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            key, raw = line.split(":", 1)
            parts = raw.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) * 1024
    except (OSError, ValueError) as exc:
        raise QwenLeaseLostError("host memory admission is unreadable") from exc
    available = values.get("MemAvailable", 0)
    commit = max(0, values.get("CommitLimit", 0) - values.get("Committed_AS", 0))
    return available, commit


def _disk_free(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError as exc:
        raise QwenLeaseLostError("host disk admission is unreadable") from exc


def _docker_root(
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    result = command_runner(
        _docker_command("info", "--format", "{{json .DockerRootDir}}"),
        env=_docker_cli_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        raise QwenRuntimeError("Docker daemon/root storage is unavailable")
    raw = result.stdout.strip()
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        value = raw
    root = _validate_absolute_path(value, "Docker root")
    metadata = root.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise QwenRuntimeError("Docker root is not a real directory")
    return root


def final_launch_admission_gate(
    lease: Mapping[str, Any],
    *,
    expected_wrapper_sha256: str,
    expected_docker_sha256: str,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    coordinator_verify_func: Callable[[Mapping[str, Any]], Any] | bool | None = None,
) -> None:
    """Last pre-create gate.  No substantive work may follow before Docker run."""

    checked = _validate_lease(lease)
    capability = _capability_from_receipt(checked)
    # Client-side operations are also part of the immutable release; the daemon
    # is authoritative only after this exact safe wrapper invokes it.
    if docker_client_sha256() != _validate_token(
        expected_docker_sha256, _SHA256_RE, "expected Docker client identity"
    ):
        raise QwenRuntimeError("Docker client wrapper changed before create")
    if low_priority_wrapper_sha256() != _validate_token(
        expected_wrapper_sha256, _SHA256_RE, "expected low-priority identity"
    ):
        raise QwenRuntimeError("fleet-low-priority changed before create")
    if (
        capability.runtime_adapter not in {"local-docker", "remote-docker"}
        or socket.gethostname() != capability.hostname
    ):
        raise QwenRuntimeError("Qwen launch is not on its capability host")
    device = Path(f"/dev/nvidia{checked['physical_gpu']}")
    try:
        metadata = device.lstat()
    except OSError as exc:
        raise QwenLeaseLostError("leased physical GPU node is unavailable") from exc
    if not stat.S_ISCHR(metadata.st_mode):
        raise QwenLeaseLostError("leased physical GPU node is not a character device")
    acl = command_runner(
        ["/usr/bin/getfacl", "-cp", "--", str(device)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if acl.returncode != 0 or "user:aday:---" in acl.stdout.splitlines():
        raise QwenLeaseLostError("leased physical GPU is renter-blocked or ambiguous")
    available, commit = _read_meminfo()
    gib = 1024**3
    if (
        available < QWEN38_VLLM_PROFILE.min_host_memory_gb * gib
        or commit < QWEN38_VLLM_PROFILE.min_host_commit_gb * gib
        or _disk_free(Path("/home/aday")) < QWEN38_VLLM_PROFILE.min_disk_free_gb * gib
        or _disk_free(Path("/dev/shm")) < QWEN38_VLLM_PROFILE.min_shm_free_gb * gib
    ):
        raise QwenLeaseLostError("local host resources fell below the Qwen profile")
    docker_root = _docker_root(command_runner)
    if _disk_free(docker_root) < QWEN38_VLLM_PROFILE.min_disk_free_gb * gib:
        raise QwenLeaseLostError("Docker-root storage fell below the Qwen profile")
    # Coordinator truth is deliberately the last potentially slow operation.
    # The caller immediately invokes Docker with the UUID-only selector.
    if coordinator_verify_func is not False:
        (coordinator_verify_func or verify_coordinator_lease)(checked)


def _docker_run_command(state: Mapping[str, Any]) -> list[str]:
    run_dir = _validate_run_dir(state["run_dir"])
    cidfile = run_dir / "container.cid"
    contract_env = state["container_environment"]
    labels = state["container_labels"]
    mounts = state["container_mounts"]
    command = [
        str(HOST_BASH),
        str(FLEET_LOW_PRIORITY),
        str(HOST_BASH),
        str(DOCKER),
        "run",
        "-d",
        "--cidfile",
        str(cidfile),
        "--name",
        str(state["container_name"]),
        "--hostname",
        str(state["container_name"]),
        "--interactive=false",
        "--tty=false",
        "--network",
        "bridge",
        "--cgroupns",
        "private",
        "--runtime",
        "runc",
        "--gpus",
        f"device={state['gpu_uuid']}",
        "--shm-size",
        str(QWEN_CONTAINER_SHM_BYTES),
        "--ipc",
        "private",
        "--publish",
        f"127.0.0.1:{state['local_port']}:{state['remote_port']}",
        "--oom-score-adj",
        "1000",
        "--cpu-shares",
        "2",
        "--blkio-weight",
        "10",
        "--pids-limit",
        "1024",
        "--user",
        f"{os.geteuid()}:{os.getegid()}",
        "--read-only",
        "--privileged=false",
        "--publish-all=false",
        "--init=false",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        f"/workspace/cache:{_validated_tmpfs_options(state)}",
        "--log-driver",
        "local",
        "--log-opt",
        "max-size=10m",
        "--log-opt",
        "max-file=3",
        "--restart",
        "no",
    ]
    for key in sorted(labels):
        command += ["--label", f"{key}={labels[key]}"]
    for destination in sorted(mounts):
        command += [
            "--mount",
            f"type=bind,src={mounts[destination]['source']},dst={destination},readonly",
        ]
    for key in sorted(contract_env):
        command += ["--env", f"{key}={contract_env[key]}"]
    command += [
        "--entrypoint",
        "/usr/local/bin/fleet-low-priority",
        str(state["image_id"]),
        *state["container_command"],
    ]
    return command


def _read_cidfile(run_dir: Path) -> str | None:
    path = run_dir / "container.cid"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _ContainerIdReceiptError(
            "container ID receipt cannot be opened safely"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size > 128
            or metadata.st_mode & 0o022
        ):
            raise _ContainerIdReceiptError("container ID receipt is unsafe")
        value = read_bounded_fd(descriptor, 128).decode("ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise _ContainerIdReceiptError("container ID receipt is unreadable") from exc
    finally:
        os.close(descriptor)
    try:
        return _validate_token(value, _CONTAINER_ID_RE, "container ID")
    except QwenRuntimeError as exc:
        raise _ContainerIdReceiptError("container ID receipt is malformed") from exc


def _docker_inspect(
    container_id: str,
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any] | None:
    container_id = _validate_token(container_id, _CONTAINER_ID_RE, "container ID")
    try:
        result = command_runner(
            _docker_command("inspect", container_id),
            env=_docker_cli_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        raise QwenRuntimeError("Docker container presence is ambiguous") from exc
    if result.returncode != 0:
        missing = {
            f"Error: No such object: {container_id}",
            f"Error response from daemon: No such container: {container_id}",
        }
        # Do not strip arbitrary whitespace: only the two Docker not-found
        # diagnostics, anchored to this exact full immutable ID, prove absence.
        diagnostic = result.stderr
        if diagnostic.endswith("\r\n"):
            diagnostic = diagnostic[:-2]
        elif diagnostic.endswith("\n"):
            diagnostic = diagnostic[:-1]
        # The pinned Docker CLI serializes its empty successful-inspect result
        # before reporting a per-reference failure, so one canonical missing
        # object produces exactly ``[]\n`` on stdout plus the anchored stderr.
        # Keep the legacy empty stdout form deliberately supported; every other
        # byte remains ambiguity.
        canonical_missing = (
            result.returncode == 1
            and result.stdout in {"", "[]\n"}
            and diagnostic in missing
        )
        # The fleet's pinned wrapper lowercases only this one diagnostic while
        # retaining Docker inspect's exact ``[]\n`` serialization.  Bind the
        # exception to every byte of that observed shape; do not case-fold or
        # admit its empty-stdout/CRLF/prefix/suffix variants.
        pinned_wrapper_missing = (
            result.returncode == 1
            and result.stdout == "[]\n"
            and result.stderr == f"error: no such object: {container_id}\n"
        )
        if not canonical_missing and not pinned_wrapper_missing:
            raise QwenRuntimeError("Docker container presence is ambiguous")
        try:
            daemon = command_runner(
                _docker_command("info", "--format", "{{.ServerVersion}}"),
                env=_docker_cli_environment(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception as exc:
            raise QwenRuntimeError("Docker container absence is ambiguous") from exc
        server_version = daemon.stdout
        if server_version.endswith("\r\n"):
            server_version = server_version[:-2]
        elif server_version.endswith("\n"):
            server_version = server_version[:-1]
        if (
            daemon.returncode != 0
            or daemon.stderr != ""
            or re.fullmatch(r"[0-9][A-Za-z0-9.+_-]{0,127}", server_version) is None
        ):
            raise QwenRuntimeError("Docker container absence is ambiguous")
        return None
    if result.stderr != "":
        raise QwenRuntimeError("Docker inspect receipt is ambiguous")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("Docker inspect receipt is malformed") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise QwenRuntimeError("Docker inspect receipt is not exact")
    if payload[0].get("Id") != container_id:
        raise QwenRuntimeError("Docker inspect returned the wrong immutable ID")
    return payload[0]


def _label_candidates(
    state: Mapping[str, Any],
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> list[str]:
    try:
        result = command_runner(
            [
                *_docker_command("ps"),
                "-aq",
                "--no-trunc",
                "--filter",
                f"label=com.bc_aeon.claim={state['claim_id']}",
                "--filter",
                f"label=com.bc_aeon.launch-nonce={state['launch_nonce']}",
            ],
            env=_docker_cli_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        raise QwenRuntimeError("Docker nonce recovery is unavailable") from exc
    if result.returncode != 0 or result.stderr != "":
        raise QwenRuntimeError("Docker nonce recovery is unavailable")
    candidates = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(candidates) != len(set(candidates)):
        raise QwenRuntimeError("Docker nonce recovery returned duplicate IDs")
    return [_validate_token(value, _CONTAINER_ID_RE, "container candidate") for value in candidates]


def _mounts_match_live_pid(state: Mapping[str, Any], pid: int) -> bool:
    if pid <= 1:
        return False
    for destination, receipt in state["container_mounts"].items():
        source = Path(str(receipt["source"]))
        mounted = Path(f"/proc/{pid}/root") / destination.lstrip("/")
        try:
            source_metadata = source.lstat()
            mount_metadata = mounted.lstat()
        except OSError:
            return False
        expected = (int(receipt["device"]), int(receipt["inode"]))
        if (
            (source_metadata.st_dev, source_metadata.st_ino) != expected
            or (mount_metadata.st_dev, mount_metadata.st_ino) != expected
        ):
            return False
    return True


def _require_exact_fields(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    error: str,
) -> None:
    for key, value in expected.items():
        # Docker omits some nil pointer fields from older API versions.  Missing
        # and JSON null have the same no-capability meaning; every non-null value
        # is required to be present and byte-for-byte equivalent after decoding.
        if value is None:
            if observed.get(key) is not None:
                raise QwenRuntimeError(error)
        elif key not in observed or not _exact_json_value(observed[key], value):
            raise QwenRuntimeError(error)


def _exact_json_value(observed: Any, expected: Any) -> bool:
    """Compare decoded JSON without Python's bool/int equality coercions."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _exact_json_value(observed[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _exact_json_value(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _normalise_docker_config_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize only Docker API fields omitted at their inert defaults."""

    result = dict(config)
    for key, default in {
        "Healthcheck": None,
        "ArgsEscaped": False,
        "OnBuild": None,
        "StopSignal": "",
        "StopTimeout": None,
        "Shell": None,
    }.items():
        if key not in result:
            result[key] = default
    return result


def _normalise_docker_host_defaults(host: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize daemon encodings that preserve the exact create contract."""

    result = dict(host)
    if result.get("Dns") is None:
        result["Dns"] = []
    if result.get("Ulimits") == []:
        result["Ulimits"] = None
    masked = result.get("MaskedPaths")
    if (
        not isinstance(masked, list)
        or any(not isinstance(path, str) for path in masked)
        or len(masked) != len(set(masked))
    ):
        raise QwenRuntimeError("container masked-path receipt is malformed")
    observed = set(masked)
    baseline = set(_BASELINE_MASKED_PATHS)
    extras = observed - baseline
    if not baseline <= observed or any(
        path != "/proc/interrupts"
        and _DOCKER_CPU_THERMAL_MASK_RE.fullmatch(path) is None
        for path in extras
    ):
        raise QwenRuntimeError("container masked-path security receipt changed")
    # Additional whitelisted masks only remove access.  Canonicalize their
    # order/count after proving every required baseline restriction remains.
    result["MaskedPaths"] = list(_BASELINE_MASKED_PATHS)
    return result


def _validate_host_mount_receipt(
    mounts: Any,
    expected_mounts: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(mounts, list) or len(mounts) != len(expected_mounts):
        raise QwenRuntimeError("container HostConfig mount set changed")
    allowed = {
        "Type",
        "Source",
        "Target",
        "ReadOnly",
        "Consistency",
        "BindOptions",
        "VolumeOptions",
        "TmpfsOptions",
        "ImageOptions",
        "ClusterOptions",
    }
    observed: set[str] = set()
    for mount in mounts:
        if not isinstance(mount, dict) or set(mount) - allowed:
            raise QwenRuntimeError("container HostConfig mount is malformed")
        target = mount.get("Target")
        if target not in expected_mounts or target in observed:
            raise QwenRuntimeError("container HostConfig mount set changed")
        expected = expected_mounts[target]
        if (
            mount.get("Type") != "bind"
            or mount.get("Source") != expected["source"]
            or mount.get("ReadOnly") is not True
            or mount.get("Consistency", "") != ""
            or any(
                mount.get(key) not in (None, {})
                for key in (
                    "BindOptions",
                    "VolumeOptions",
                    "TmpfsOptions",
                    "ImageOptions",
                    "ClusterOptions",
                )
            )
        ):
            raise QwenRuntimeError("container HostConfig bind mount changed")
        observed.add(str(target))
    if observed != set(expected_mounts):
        raise QwenRuntimeError("container HostConfig mount set is incomplete")


def _validate_top_level_mount_receipt(
    mounts: Any,
    expected_mounts: Mapping[str, Mapping[str, Any]],
    *,
    running: bool,
) -> None:
    if not isinstance(mounts, list) or len(mounts) not in {
        len(expected_mounts),
        len(expected_mounts) + 1,
    }:
        raise QwenRuntimeError("container top-level mount set changed")
    observed: set[str] = set()
    cache_seen = False
    for mount in mounts:
        if not isinstance(mount, dict):
            raise QwenRuntimeError("container mount receipt is malformed")
        destination = mount.get("Destination")
        if destination == "/workspace/cache":
            if cache_seen or not _exact_json_value(mount, {
                "Type": "tmpfs",
                "Source": "",
                "Destination": "/workspace/cache",
                "Mode": "",
                "RW": True,
                "Propagation": "",
            }):
                raise QwenRuntimeError("container tmpfs mount receipt changed")
            cache_seen = True
            continue
        if destination not in expected_mounts or destination in observed:
            raise QwenRuntimeError("container has an unreceipted mount")
        expected = expected_mounts[destination]
        if not _exact_json_value(mount, {
            "Type": "bind",
            "Source": expected["source"],
            "Destination": destination,
            "Mode": "",
            "RW": False,
            "Propagation": "rprivate",
        }):
            raise QwenRuntimeError("container bind mount identity changed")
        observed.add(str(destination))
    # Docker 29 can omit the tmpfs projection from top-level Mounts for both an
    # active and an exited container.  HostConfig.Tmpfs remains the exact
    # immutable create receipt, the root filesystem remains read-only, and
    # every bind mount remains fully inode-bound below.  When Docker does
    # project the tmpfs here, the exact receipt above is still mandatory.
    if observed != set(expected_mounts):
        raise QwenRuntimeError("container mount set is incomplete")


def _validate_network_receipt(
    settings: Any,
    expected_binding: Mapping[str, Any],
    *,
    container_name: str,
    container_id: str,
    running: bool,
) -> None:
    if not isinstance(settings, dict):
        raise QwenRuntimeError("container network receipt is malformed")
    allowed_settings = {
        "Bridge",
        "SandboxID",
        "SandboxKey",
        "Ports",
        "HairpinMode",
        "LinkLocalIPv6Address",
        "LinkLocalIPv6PrefixLen",
        "SecondaryIPAddresses",
        "SecondaryIPv6Addresses",
        "EndpointID",
        "Gateway",
        "GlobalIPv6Address",
        "GlobalIPv6PrefixLen",
        "IPAddress",
        "IPPrefixLen",
        "IPv6Gateway",
        "MacAddress",
        "Networks",
    }
    if set(settings) - allowed_settings:
        raise QwenRuntimeError("container network has an unreceipted field")
    observed_ports = settings.get("Ports")
    if running:
        if not _exact_json_value(observed_ports, expected_binding):
            raise QwenRuntimeError("container network/port receipt changed")
    elif not (
        _exact_json_value(observed_ports, expected_binding)
        or _exact_json_value(observed_ports, {})
    ):
        # Docker detaches an exited container and reports an empty runtime port
        # map.  The immutable HostConfig.PortBindings receipt remains exact.
        raise QwenRuntimeError("container network/port receipt changed")
    for key in ("SandboxID", "SandboxKey"):
        if key in settings and not isinstance(settings[key], str):
            raise QwenRuntimeError("container network identity is malformed")
    # API v1.52 (Docker 29) removed these long-deprecated top-level bridge
    # mirrors.  Accept exactly either the modern omission or the complete
    # legacy group; a partial/mixed response is ambiguous.
    legacy_static = {
        "Bridge": "",
        "HairpinMode": False,
        "LinkLocalIPv6Address": "",
        "LinkLocalIPv6PrefixLen": 0,
        "SecondaryIPAddresses": None,
        "SecondaryIPv6Addresses": None,
    }
    legacy_dynamic = {
        "EndpointID",
        "Gateway",
        "GlobalIPv6Address",
        "GlobalIPv6PrefixLen",
        "IPAddress",
        "IPPrefixLen",
        "IPv6Gateway",
        "MacAddress",
    }
    legacy_fields = set(legacy_static) | legacy_dynamic
    present_legacy = set(settings) & legacy_fields
    if present_legacy and present_legacy != legacy_fields:
        raise QwenRuntimeError("container network legacy receipt is incomplete")
    if present_legacy:
        _require_exact_fields(
            settings,
            legacy_static,
            "container network legacy defaults changed",
        )
    networks = settings.get("Networks")
    if not isinstance(networks, dict) or set(networks) != {"bridge"}:
        raise QwenRuntimeError("container is attached to an unexpected network")
    bridge = networks["bridge"]
    if not isinstance(bridge, dict):
        raise QwenRuntimeError("container bridge receipt is malformed")
    allowed_bridge = {
        "IPAMConfig",
        "Links",
        "Aliases",
        "MacAddress",
        "DriverOpts",
        "GwPriority",
        "NetworkID",
        "EndpointID",
        "Gateway",
        "IPAddress",
        "IPPrefixLen",
        "IPv6Gateway",
        "GlobalIPv6Address",
        "GlobalIPv6PrefixLen",
        "DNSNames",
    }
    if set(bridge) - allowed_bridge:
        raise QwenRuntimeError("container bridge has an unreceipted field")
    for key in ("IPAMConfig", "Links", "Aliases", "DriverOpts"):
        if bridge.get(key) not in (None, [], {}):
            raise QwenRuntimeError("container bridge has an unreceipted option")
    for key in (
        "MacAddress",
        "NetworkID",
        "EndpointID",
        "Gateway",
        "IPAddress",
        "IPv6Gateway",
        "GlobalIPv6Address",
    ):
        if key in bridge and not isinstance(bridge[key], str):
            raise QwenRuntimeError("container bridge identity is malformed")
    for key in ("GwPriority", "IPPrefixLen", "GlobalIPv6PrefixLen"):
        if key in bridge and (
            isinstance(bridge[key], bool) or not isinstance(bridge[key], int)
        ):
            raise QwenRuntimeError("container bridge prefix/priority is malformed")
    dns_names_present = "DNSNames" in bridge
    dns_names = bridge.get("DNSNames")
    expected_dns_names = {container_name, container_id[:12]}
    if dns_names is None and running:
        # Linux's default bridge has no embedded service discovery. Docker 29
        # therefore reports an explicit JSON null rather than endpoint names.
        # Missing is not accepted as the same form: this field is present in
        # the pinned API response and remains part of the exact receipt.
        if not dns_names_present:
            raise QwenRuntimeError("container bridge DNS identity changed")
    elif dns_names is None and not running:
        # The bridge endpoint is detached on exit.  Require Docker's exact
        # no-endpoint defaults rather than accepting arbitrary stale topology.
        detached = {
            "EndpointID": "",
            "Gateway": "",
            "IPAddress": "",
            "IPPrefixLen": 0,
            "IPv6Gateway": "",
            "GlobalIPv6Address": "",
            "GlobalIPv6PrefixLen": 0,
            "MacAddress": "",
        }
        _require_exact_fields(
            bridge,
            detached,
            "exited container bridge endpoint is not detached",
        )
    elif (
        not isinstance(dns_names, list)
        or any(not isinstance(value, str) for value in dns_names)
        or len(dns_names) != len(set(dns_names))
        or set(dns_names) != expected_dns_names
    ):
        raise QwenRuntimeError("container bridge DNS identity changed")
    if present_legacy:
        for key in legacy_dynamic:
            if not _exact_json_value(
                settings[key],
                bridge.get(key, "" if "PrefixLen" not in key else 0),
            ):
                raise QwenRuntimeError("container legacy bridge identity diverged")


def _inspect_identity(item: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[str, int | None]:
    """Validate the full create receipt; return active/exited and exact host PID."""

    capability = _capability_from_receipt(
        state, allow_retired_manifest=True
    )
    if state.get("image_id") != capability.image_id:
        raise QwenRuntimeError("container state is outside its runtime capability")
    try:
        config = item["Config"]
        host = item["HostConfig"]
        run_state = item["State"]
        mounts = item["Mounts"]
        network = item["NetworkSettings"]
    except (KeyError, TypeError) as exc:
        raise QwenRuntimeError("container inspect lacks release-critical fields") from exc
    if not all(isinstance(value, dict) for value in (config, host, run_state)):
        raise QwenRuntimeError("container inspect fields are malformed")
    running = run_state.get("Running")
    if type(running) is not bool:
        raise QwenRuntimeError("container run state is ambiguous")
    expected_id = state.get("container_id")
    if expected_id is not None and item.get("Id") != expected_id:
        raise QwenRuntimeError("container immutable ID changed")
    expected_port = f"{state['remote_port']}/tcp"
    expected_binding = {
        expected_port: [
            {"HostIp": "127.0.0.1", "HostPort": str(state["local_port"])}
        ]
    }
    image_exposed_ports = _normalise_image_exposed_ports(
        state.get("image_base_exposed_ports")
    )
    if expected_port in image_exposed_ports:
        raise QwenRuntimeError("runtime API port overlaps the inherited image port")
    expected_exposed_ports = {**image_exposed_ports, expected_port: {}}
    expected_network_ports = {
        **{port: None for port in image_exposed_ports},
        **expected_binding,
    }
    expected_config = {
            "Hostname": state["container_name"],
            "Domainname": "",
            "User": f"{os.geteuid()}:{os.getegid()}",
            "AttachStdin": False,
            "AttachStdout": False,
            "AttachStderr": False,
            "ExposedPorts": expected_exposed_ports,
            "Tty": False,
            "OpenStdin": False,
            "StdinOnce": False,
            "Cmd": state["container_command"],
            "Healthcheck": None,
            "ArgsEscaped": False,
            "Image": state["image_id"],
            "Volumes": None,
            "WorkingDir": "",
            "Entrypoint": ["/usr/local/bin/fleet-low-priority"],
            "OnBuild": None,
            "Labels": state["container_labels"],
            "StopSignal": "",
            "StopTimeout": None,
            "Shell": None,
    }
    if set(config) - (set(expected_config) | {"Env"}):
        raise QwenRuntimeError("container Config has an unreceipted field")
    _require_exact_fields(
        _normalise_docker_config_defaults(config),
        expected_config,
        "container Config receipt changed",
    )
    if (
        item.get("Image") != state["image_id"]
        or item.get("Name") != f"/{state['container_name']}"
        or item.get("Path") != "/usr/local/bin/fleet-low-priority"
        or item.get("Args") != state["container_command"]
        or item.get("Platform") != "linux"
        or _unique_environment(config.get("Env"), "container") != state["container_environment"]
    ):
        raise QwenRuntimeError("container command/environment/image identity changed")
    device_requests = host.get("DeviceRequests")
    if not isinstance(device_requests, list) or len(device_requests) != 1:
        raise QwenRuntimeError("container has an unexpected GPU request")
    request = device_requests[0]
    if not _exact_json_value(request, {
        "Driver": "",
        # Docker's explicit DeviceIDs form canonicalizes Count to zero.  -1 is
        # the semantically different all-devices request and is never accepted.
        "Count": 0,
        "DeviceIDs": [state["gpu_uuid"]],
        "Capabilities": [["gpu"]],
        "Options": {},
    }):
        raise QwenRuntimeError("container GPU UUID request changed")
    expected_tmpfs = _validated_tmpfs_options(state)
    cidfile = str(_validate_run_dir(state["run_dir"]) / "container.cid")
    expected_host = {
            "Binds": None,
            "ContainerIDFile": cidfile,
            "LogConfig": {
                "Type": "local",
                "Config": {"max-file": "3", "max-size": "10m"},
            },
            "NetworkMode": "bridge",
            "PortBindings": expected_binding,
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "AutoRemove": False,
            "VolumeDriver": "",
            "VolumesFrom": None,
            "ConsoleSize": [0, 0],
            "CapAdd": None,
            "CapDrop": ["ALL"],
            "CgroupnsMode": "private",
            "Dns": [],
            "DnsOptions": [],
            "DnsSearch": [],
            "ExtraHosts": None,
            "GroupAdd": None,
            "IpcMode": "private",
            "Cgroup": "",
            "Links": None,
            "OomScoreAdj": 1000,
            "PidMode": "",
            "Privileged": False,
            "PublishAllPorts": False,
            "ReadonlyRootfs": True,
            "SecurityOpt": ["no-new-privileges"],
            "UTSMode": "",
            "UsernsMode": "",
            "ShmSize": QWEN_CONTAINER_SHM_BYTES,
            "Runtime": "runc",
            "Isolation": "",
            "CpuShares": 2,
            "Memory": 0,
            "NanoCpus": 0,
            "CgroupParent": "",
            "BlkioWeight": 10,
            "BlkioWeightDevice": [],
            "BlkioDeviceReadBps": [],
            "BlkioDeviceWriteBps": [],
            "BlkioDeviceReadIOps": [],
            "BlkioDeviceWriteIOps": [],
            "CpuPeriod": 0,
            "CpuQuota": 0,
            "CpuRealtimePeriod": 0,
            "CpuRealtimeRuntime": 0,
            "CpusetCpus": "",
            "CpusetMems": "",
            "Devices": [],
            "DeviceCgroupRules": None,
            "MemoryReservation": 0,
            "MemorySwap": 0,
            "MemorySwappiness": None,
            "OomKillDisable": None,
            "PidsLimit": 1024,
            "Ulimits": None,
            "CpuCount": 0,
            "CpuPercent": 0,
            "IOMaximumIOps": 0,
            "IOMaximumBandwidth": 0,
            "MaskedPaths": list(_BASELINE_MASKED_PATHS),
            "ReadonlyPaths": [
                "/proc/bus",
                "/proc/fs",
                "/proc/irq",
                "/proc/sys",
                "/proc/sysrq-trigger",
            ],
            "Tmpfs": {"/workspace/cache": expected_tmpfs},
            "Init": False,
    }
    host_extensions = {"DeviceRequests", "Mounts", "UseApiSocket", "Annotations"}
    if set(host) - (set(expected_host) | host_extensions):
        raise QwenRuntimeError("container HostConfig has an unreceipted field")
    _require_exact_fields(
        _normalise_docker_host_defaults(host),
        expected_host,
        "container host/security/resource receipt changed",
    )
    if host.get("UseApiSocket", False) is not False or host.get("Annotations") not in (
        None,
        {},
    ):
        raise QwenRuntimeError("container has an unreceipted Docker capability")
    expected_mounts = state["container_mounts"]
    _validate_host_mount_receipt(host.get("Mounts"), expected_mounts)
    _validate_top_level_mount_receipt(mounts, expected_mounts, running=running)
    container_id = _validate_token(item.get("Id"), _CONTAINER_ID_RE, "container ID")
    _validate_network_receipt(
        network,
        expected_network_ports,
        container_name=str(state["container_name"]),
        container_id=container_id,
        running=running,
    )
    pid_raw = run_state.get("Pid")
    if running is True:
        if isinstance(pid_raw, bool) or not isinstance(pid_raw, int) or pid_raw <= 1:
            raise QwenRuntimeError("running container has no exact host PID")
        if not _mounts_match_live_pid(state, pid_raw):
            raise QwenRuntimeError("live bind-mount inode identity changed")
        return "active", pid_raw
    if isinstance(pid_raw, bool) or not isinstance(pid_raw, int) or pid_raw != 0:
        raise QwenRuntimeError("exited container retains an ambiguous host PID")
    # Even an exited container is removable only while the current source paths
    # still name the exact receipted inodes.
    for receipt in expected_mounts.values():
        try:
            metadata = Path(receipt["source"]).lstat()
        except OSError as exc:
            raise QwenRuntimeError("exited container bind source is unavailable") from exc
        if (metadata.st_dev, metadata.st_ino) != (
            int(receipt["device"]),
            int(receipt["inode"]),
        ):
            raise QwenRuntimeError("exited container bind source identity changed")
    return "exited", None


def _resolve_container(
    state: Mapping[str, Any],
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    adopt: bool = False,
    state_path: Path = RUNTIME_STATE_FILE,
) -> tuple[str, int | None, dict[str, Any]]:
    candidate = state.get("container_id")
    cidfile_recovery = state.get("cidfile_recovery_authorized") is True
    cidfile_recovery_needs_nonce_proof = False
    run_dir = _validate_run_dir(state["run_dir"])
    if candidate is None:
        try:
            candidate = _read_cidfile(run_dir)
        except _ContainerIdReceiptError:
            # A cidfile is only an optimization, never lifecycle authority.  An
            # interrupted Docker create can leave it with umask-inherited or
            # partially written metadata.  Ignore its contents and recover only
            # through the two exact immutable labels below.
            cidfile_recovery = True
            candidate = None
    elif not cidfile_recovery:
        try:
            cidfile_candidate = _read_cidfile(run_dir)
        except _ContainerIdReceiptError:
            cidfile_recovery = True
            cidfile_recovery_needs_nonce_proof = True
        else:
            if cidfile_candidate is not None and cidfile_candidate != candidate:
                # Never let the mutable cidfile override a durable immutable ID.
                # Its eventual cleanup requires an independent nonce proof.
                cidfile_recovery = True
                cidfile_recovery_needs_nonce_proof = True
    if candidate is None:
        candidates = _label_candidates(state, command_runner)
        if not candidates:
            recovered = dict(state)
            if cidfile_recovery:
                recovered["cidfile_recovery_authorized"] = True
            if adopt and recovered != dict(state):
                recovered["updated_at"] = time.time()
                _private_json_write(state_path, recovered)
            return "gone", None, recovered
        if len(candidates) != 1:
            raise QwenRuntimeError("launch nonce resolves to multiple containers")
        candidate = candidates[0]
    item = _docker_inspect(str(candidate), command_runner)
    if item is None:
        candidates = _label_candidates(state, command_runner)
        if candidates:
            raise QwenRuntimeError("saved container ID disappeared but nonce remains live")
        recovered = dict(state)
        if cidfile_recovery:
            recovered["cidfile_recovery_authorized"] = True
        if adopt and recovered != dict(state):
            recovered["updated_at"] = time.time()
            _private_json_write(state_path, recovered)
        return "gone", None, recovered
    adopted = {**dict(state), "container_id": str(candidate)}
    status, pid = _inspect_identity(item, adopted)
    if cidfile_recovery_needs_nonce_proof:
        candidates = _label_candidates(state, command_runner)
        if candidates != [str(candidate)]:
            raise QwenRuntimeError(
                "saved container ID and launch nonce no longer identify one container"
            )
    if cidfile_recovery:
        adopted["cidfile_recovery_authorized"] = True
    adopted["container_pid"] = pid
    if adopt and adopted != dict(state):
        adopted["updated_at"] = time.time()
        _private_json_write(state_path, adopted)
    return status, pid, adopted


def local_container_pid(
    *,
    state_path: Path = RUNTIME_STATE_FILE,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int | None:
    state = current_runtime_state(state_path)
    if state is None or state.get("phase") in {"preparing", "preflight"}:
        return None
    status, pid, _state = _resolve_container(
        state, command_runner=command_runner, adopt=True, state_path=state_path
    )
    return pid if status == "active" else None


def _endpoint_identity(
    state: Mapping[str, Any],
    *,
    request_get: Callable[..., Any] = _loopback_get,
) -> None:
    base = f"http://127.0.0.1:{state['local_port']}"
    try:
        health = request_get(f"{base}/health", timeout=3)
        health_status = health.status_code
        _bounded_loopback_body(health, 64 * 1024)
        models = request_get(f"{base}/v1/models", timeout=5)
        models_status = models.status_code
        payload = json.loads(_bounded_loopback_body(models, 256 * 1024))
    except Exception as exc:
        raise QwenRuntimeLoadingError("exact Qwen endpoint is not ready") from exc
    identifiers = {
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict)
    } if isinstance(payload, dict) else set()
    if (
        health_status != 200
        or models_status != 200
        or identifiers != {state["served_name"]}
    ):
        raise QwenRuntimeLoadingError("Qwen endpoint model identity is not exact")


def _wait_for_endpoint(
    state: Mapping[str, Any],
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
    request_get: Callable[..., Any],
    sleep_func: Callable[[float], None],
    timeout_seconds: float,
    progress_check: Callable[[], None] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if progress_check is not None:
            progress_check()
        status, _pid, _ = _resolve_container(state, command_runner=command_runner)
        if status != "active":
            raise QwenRuntimeError("exact Qwen container exited during model load")
        try:
            _endpoint_identity(state, request_get=request_get)
            return
        except QwenRuntimeLoadingError:
            if time.monotonic() >= deadline:
                raise
            sleep_func(min(5.0, max(0.0, deadline - time.monotonic())))


def _artifact_from_state(state: Mapping[str, Any]) -> ArtifactIdentity:
    stats: list[tuple[str, int, int, int, int, int, int]] = []
    raw_stats = state.get("model_file_stats")
    if not isinstance(raw_stats, list):
        raise QwenRuntimeError("saved model stat receipt is missing")
    for raw in raw_stats:
        if not isinstance(raw, list) or len(raw) != 7:
            raise QwenRuntimeError("saved model stat receipt is malformed")
        relative = _validate_relative_path(raw[0], "saved model path")
        numbers = raw[1:]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in numbers):
            raise QwenRuntimeError("saved model stat receipt is malformed")
        stats.append((relative, *numbers))
    return ArtifactIdentity(
        model_dir=_validate_absolute_path(state["model_dir"], "saved model path"),
        manifest_sha256=str(state["model_manifest_sha256"]),
        sha256s_sha256=str(state["model_sha256s_sha256"]),
        files=tuple(str(value) for value in state["model_files"]),
        total_bytes=int(state["model_bytes"]),
        root_device=int(state["model_root_device"]),
        root_inode=int(state["model_root_inode"]),
        file_stats=tuple(stats),
    )


def _source_from_state(state: Mapping[str, Any], package_root: Path) -> SourceIdentity:
    identity = _source_identity(package_root, _validate_run_dir(state["run_dir"]))
    if (
        identity.manifest_sha256 != state["source_manifest_sha256"]
        or identity.stage_dir != Path(state["source_dir"])
        or {relative for relative, _digest in identity.file_sha256}
        != set(state["source_files"]) - {"SOURCE_SHA256SUMS"}
    ):
        raise QwenRuntimeError("runtime source receipt changed")
    _validate_source_stage(identity)
    return identity


_WARMUP_FAILURE_SCHEMA_VERSION = 1
_WARMUP_FAILURE_MAX_BYTES = 256
_WARMUP_TURN_FAILURE_CODES = frozenset(
    {
        "input_build",
        "http_timeout",
        "http_request",
        "http_status",
        "response_json",
        "completion_count",
        "completion_content",
        "turn_json",
        "turn_not_object",
        "turn_missing_required",
        "turn_unexpected_fields",
        "turn_action",
        "internal",
    }
)
_WARMUP_FAILURE_CODES_BY_STAGE = {
    "preflight": frozenset({"staged_imports", "internal"}),
    "text": _WARMUP_TURN_FAILURE_CODES,
    "vision": _WARMUP_TURN_FAILURE_CODES,
    "runner": frozenset(
        {
            "exec_error",
            "invalid_diagnostic",
            "result_mismatch",
            "timeout",
        }
    ),
}


def _validated_warmup_failure(value: Any) -> dict[str, Any]:
    """Return only the bounded v1 warmup failure contract or a safe fallback."""

    fallback = {
        "schema_version": _WARMUP_FAILURE_SCHEMA_VERSION,
        "stage": "runner",
        "code": "invalid_diagnostic",
    }
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "stage", "code"}
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != _WARMUP_FAILURE_SCHEMA_VERSION
    ):
        return fallback
    stage = value.get("stage")
    code = value.get("code")
    if (
        not isinstance(stage, str)
        or not isinstance(code, str)
        or code not in _WARMUP_FAILURE_CODES_BY_STAGE.get(stage, frozenset())
    ):
        return fallback
    return {
        "schema_version": _WARMUP_FAILURE_SCHEMA_VERSION,
        "stage": stage,
        "code": code,
    }


def _warmup_runner_failure(code: str) -> dict[str, Any]:
    return _validated_warmup_failure(
        {
            "schema_version": _WARMUP_FAILURE_SCHEMA_VERSION,
            "stage": "runner",
            "code": code,
        }
    )


def _read_warmup_failure(descriptor: int) -> dict[str, Any]:
    """Read at most one exact private failure envelope from an inherited file."""

    fallback = _warmup_runner_failure("invalid_diagnostic")
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink > 1
            or not 0 < metadata.st_size <= _WARMUP_FAILURE_MAX_BYTES
        ):
            return fallback
        os.lseek(descriptor, 0, os.SEEK_SET)
        payload = read_bounded_fd(descriptor, _WARMUP_FAILURE_MAX_BYTES)
        if len(payload) != metadata.st_size:
            return fallback
    except OSError:
        return fallback
    try:
        raw = payload.decode("ascii")
        value = json.loads(raw)
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, UnicodeError, ValueError, json.JSONDecodeError):
        return fallback
    if raw not in {canonical, f"{canonical}\n"}:
        return fallback
    return _validated_warmup_failure(value)


def _warmup_receipt_size(descriptor: int) -> int | None:
    try:
        metadata = os.fstat(descriptor)
    except OSError:
        return None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or metadata.st_nlink > 1
        or metadata.st_size < 0
    ):
        return None
    return metadata.st_size


def _record_warmup_failure(
    state: Mapping[str, Any],
    state_path: Path,
    value: Any,
) -> None:
    """Durably record and raise one sanitized warmup failure diagnosis."""

    failure = _validated_warmup_failure(value)
    try:
        _private_json_write(
            state_path,
            {
                **dict(state),
                "warmup_failure": failure,
                "updated_at": time.time(),
            },
        )
    except Exception:
        raise QwenRuntimeError(
            "Qwen structured release warmup diagnostic write failed"
        ) from None
    raise QwenRuntimeError(
        "Qwen structured release warmup failed "
        f"[v{failure['schema_version']}:{failure['stage']}:{failure['code']}]"
    ) from None


def _run_structured_warmup(
    command: list[str],
    *,
    cwd: str | Path,
    environment: Mapping[str, str],
    receipt_dir: Path,
    state: Mapping[str, Any],
    state_path: Path,
    command_runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> None:
    """Run the warmup with no output pipes and one bounded diagnostic receipt."""

    try:
        receipt = tempfile.TemporaryFile(mode="w+b", dir=receipt_dir)
    except Exception:
        _record_warmup_failure(
            state,
            state_path,
            _warmup_runner_failure("exec_error"),
        )
    try:
        with receipt:
            try:
                descriptor = receipt.fileno()
                os.fchmod(descriptor, 0o600)
            except Exception:
                _record_warmup_failure(
                    state,
                    state_path,
                    _warmup_runner_failure("exec_error"),
                )
            try:
                result = command_runner(
                    [*command, "--failure-fd", str(descriptor)],
                    cwd=str(cwd),
                    env=dict(environment),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    pass_fds=(descriptor,),
                    timeout=600,
                )
                returncode = result.returncode
                if isinstance(returncode, bool) or not isinstance(returncode, int):
                    raise QwenRuntimeError("warmup runner result is malformed")
            except subprocess.TimeoutExpired:
                _record_warmup_failure(
                    state,
                    state_path,
                    _warmup_runner_failure("timeout"),
                )
            except Exception:
                _record_warmup_failure(
                    state,
                    state_path,
                    _warmup_runner_failure("exec_error"),
                )
            if returncode != 0:
                _record_warmup_failure(
                    state,
                    state_path,
                    _read_warmup_failure(descriptor),
                )
            if _warmup_receipt_size(descriptor) != 0:
                _record_warmup_failure(
                    state,
                    state_path,
                    _warmup_runner_failure("result_mismatch"),
                )
    except QwenRuntimeError:
        raise
    except Exception:
        _record_warmup_failure(
            state,
            state_path,
            _warmup_runner_failure("exec_error"),
        )


def start_local_runtime(
    lease: Mapping[str, Any],
    deploy_environment: Mapping[str, Any],
    *,
    package_root: Path,
    model_dir: Path,
    container_name: str,
    image: str,
    port: int,
    artifact_identity: ArtifactIdentity | None = None,
    image_identity: str | None = None,
    image_size_bytes: int | None = None,
    state_path: Path = RUNTIME_STATE_FILE,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    request_get: Callable[..., Any] = _loopback_get,
    sleep_func: Callable[[float], None] = time.sleep,
    readiness_timeout: float = 2100,
    progress_check: Callable[[], None] | None = None,
    heartbeat_promoter: Callable[[], int] | None = None,
    coordinator_verify_func: Callable[[Mapping[str, Any]], Any] | bool | None = None,
    final_heartbeat_func: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Create one exact local container after a full pre-reserve artifact proof."""

    checked = _validate_lease(lease)
    if current_runtime_state(state_path) is not None:
        raise QwenRuntimeError("a Qwen runtime receipt already exists")
    verify_func = (
        None
        if coordinator_verify_func is False
        else coordinator_verify_func or verify_coordinator_lease
    )
    if verify_func is not None:
        verify_func(checked)
    artifact = artifact_identity or load_artifact_identity(model_dir)
    if artifact.model_dir != model_dir.expanduser().resolve(strict=True):
        raise QwenRuntimeError("preverified model path differs from launch path")
    revalidate_artifact_identity(artifact)
    image_id = image_identity or local_image_id(image, command_runner=command_runner)
    if local_image_id(image, command_runner=command_runner) != image_id:
        raise QwenRuntimeError("preverified image ID changed before launch")
    size = image_size_bytes or local_image_size(image_id, command_runner=command_runner)
    if local_image_size(image_id, command_runner=command_runner) != size:
        raise QwenRuntimeError("preverified image size changed before launch")
    image_config = _image_config(image_id, command_runner=command_runner)
    run_dir = _validate_run_dir(checked["run_dir"])
    source = _source_identity(package_root, run_dir)
    try:
        plan = json.loads(str(deploy_environment["AEON_DEPLOY_PLAN"]))
        served_name = str(deploy_environment["AEON_SERVED_NAME"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("deployment environment is incomplete") from exc
    if not isinstance(plan, dict):
        raise QwenRuntimeError("deployment plan is malformed")
    nonce = secrets.token_hex(32)
    state = _base_runtime_state(
        checked,
        container_name=container_name,
        image=image,
        image_id=image_id,
        image_size_bytes=size,
        artifact=artifact,
        source=source,
        port=port,
        launch_nonce=nonce,
        served_name=served_name,
        phase="preparing",
    )
    # This durable receipt precedes all claim-owned scratch creation.
    _private_json_write(state_path, state)
    try:
        source = _prepare_source_stage(package_root, run_dir, expected_identity=source)
        contract = _container_contract(
            lease=checked,
            deploy_environment=deploy_environment,
            artifact=artifact,
            source=source,
            image_id=image_id,
            image_config=image_config,
            package_root=package_root,
            container_name=container_name,
            port=port,
            launch_nonce=nonce,
        )
        state = _apply_contract(state, contract)
        _private_json_write(state_path, state)
        cidfile = run_dir / "container.cid"
        if cidfile.exists() or cidfile.is_symlink():
            raise QwenRuntimeError("container ID receipt already exists")
        docker_command = _docker_run_command(state)
        # All hashing, MTP parsing, source staging and command construction are
        # complete. The gate's coordinator recheck is immediately adjacent to
        # the UUID-only Docker create operation.
        if _source_identity(package_root, run_dir) != source:
            raise QwenRuntimeError("host Qwen launch source changed before create")
        _validate_source_stage(source)
        revalidate_artifact_identity(artifact)
        final_launch_admission_gate(
            checked,
            expected_wrapper_sha256=str(state["wrapper_sha256"]),
            expected_docker_sha256=str(state["docker_sha256"]),
            command_runner=command_runner,
            coordinator_verify_func=verify_func if verify_func is not None else False,
        )
        result = command_runner(
            docker_command,
            env=_docker_cli_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            # Docker creates --cidfile itself, so it cannot be safely
            # precreated.  A private child umask makes the receipt mode 0600
            # without changing the long-running Aeon process's global umask.
            umask=0o077,
        )
        if result.returncode != 0:
            raise QwenRuntimeError("Docker refused the exact Qwen create receipt")
        status, pid, state = _resolve_container(
            state,
            command_runner=command_runner,
            adopt=True,
            state_path=state_path,
        )
        if status != "active" or pid is None:
            raise QwenRuntimeError("new exact Qwen container is not running")
        state = {
            **state,
            "phase": "launching",
            "container_pid": pid,
            "updated_at": time.time(),
        }
        _private_json_write(state_path, state)
        # Atomically switch the already-running startup heartbeat from its
        # bounded pre-container PID-less phase to this exact container PID
        # before the first endpoint probe. No later beat may omit the PID.
        if heartbeat_promoter is None or int(heartbeat_promoter()) != pid:
            raise QwenRuntimeError("startup heartbeat did not bind the exact container PID")
        if progress_check is not None:
            progress_check()
        _wait_for_endpoint(
            state,
            command_runner=command_runner,
            request_get=request_get,
            sleep_func=sleep_func,
            timeout_seconds=readiness_timeout,
            progress_check=progress_check,
        )
        if progress_check is not None:
            progress_check()
        warmup = source.stage_dir / "aeon/scripts/warmup_qwen38_vllm.py"
        _run_structured_warmup(
            [
                str(HOST_BASH),
                str(FLEET_LOW_PRIORITY),
                str(HOST_PYTHON),
                str(warmup),
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--model",
                served_name,
            ],
            cwd=str(source.stage_dir),
            environment={
                **HOST_LAUNCH_ENV,
                "PYTHONPATH": str(source.stage_dir),
                "PYTHONDONTWRITEBYTECODE": "1",
                "AEON_STAGED_SOURCE_ROOT": str(source.stage_dir),
            },
            receipt_dir=run_dir,
            state=state,
            state_path=state_path,
            command_runner=command_runner,
        )
        if verify_func is not None:
            verify_func(checked)
        status, pid, state = _resolve_container(
            state,
            command_runner=command_runner,
            adopt=True,
            state_path=state_path,
        )
        _endpoint_identity(state, request_get=request_get)
        if status != "active" or pid is None:
            raise QwenRuntimeError("Qwen container exited after warmup")
        state = {**state, "phase": "ready", "container_pid": pid, "updated_at": time.time()}
        _private_json_write(state_path, state)
        if final_heartbeat_func is None and verify_func is not None:
            from .gpu_queue import heartbeat_vram

            final_heartbeat_func = heartbeat_vram
        if final_heartbeat_func is not None:
            final_heartbeat_func(
                pid,
                "Aeon Qwen exact runtime passed identity and warmup",
                QWEN_LEASE_FILE,
            )
        return state
    except BaseException:
        # State remains durable for exact foreground reconciliation; never rm by
        # name and never release an identity-ambiguous claim here.
        raise


def reuse_qwen_runtime(
    *,
    config: Mapping[str, Any],
    package_root: Path,
    state_path: Path = RUNTIME_STATE_FILE,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    request_get: Callable[..., Any] = _loopback_get,
    lease_override: Mapping[str, Any] | None = None,
    coordinator_verify_func: Callable[[Mapping[str, Any]], Any] | bool | None = None,
) -> int | None:
    state = current_runtime_state(state_path)
    if state is None:
        return None
    if state["teardown_only"] is True:
        raise QwenRuntimeError("legacy Qwen runtime is teardown-only")
    lease = dict(lease_override) if lease_override is not None else current_lease(QWEN_LEASE_FILE)
    if lease is None or not runtime_state_matches_lease(state, lease):
        raise QwenLeaseLostError("saved Qwen runtime has no exact lease receipt")
    verify_func = (
        None
        if coordinator_verify_func is False
        else coordinator_verify_func or verify_coordinator_lease
    )
    checked = dict(verify_func(lease)) if verify_func is not None else _validate_lease(lease)
    if (
        config.get("container_name") != state["container_name"]
        or int(config.get("health_port") or 0) != state["local_port"]
    ):
        raise QwenRuntimeError("selected model config differs from saved Qwen runtime")
    artifact = _artifact_from_state(state)
    revalidate_artifact_identity(artifact)
    source = _source_from_state(state, package_root)
    if local_image_id(state["image"], command_runner=command_runner) != state["image_id"]:
        raise QwenRuntimeError("saved Qwen image tag changed")
    if local_image_size(state["image_id"], command_runner=command_runner) != state["image_size_bytes"]:
        raise QwenRuntimeError("saved Qwen image size changed")
    if low_priority_wrapper_sha256() != state["wrapper_sha256"]:
        raise QwenRuntimeError("renter-yielding wrapper changed")
    if docker_client_sha256() != state["docker_sha256"]:
        raise QwenRuntimeError("Docker client wrapper changed")
    contract = _container_contract(
        lease=checked,
        deploy_environment=config.get("_deploy_env") or {},
        artifact=artifact,
        source=source,
        image_id=state["image_id"],
        image_config=_image_config(state["image_id"], command_runner=command_runner),
        package_root=package_root,
        container_name=state["container_name"],
        port=state["local_port"],
        launch_nonce=state["launch_nonce"],
    )
    expected = _apply_contract(state, contract)
    for key in (
        "container_command",
        "container_environment",
        "container_labels",
        "container_mounts",
        "image_base_environment",
        "image_base_labels",
        "image_base_exposed_ports",
        "container_tmpfs_options",
        "teardown_only",
        "runtime_capability_key",
        "runtime_capability_manifest_sha256",
        "runtime_adapter",
        "launch_spec_sha256",
        "served_name",
    ):
        if expected[key] != state[key]:
            raise QwenRuntimeError("saved Qwen canonical launch receipt changed")
    status, pid, state = _resolve_container(
        state, command_runner=command_runner, adopt=True, state_path=state_path
    )
    if status == "gone":
        return None
    if status == "exited":
        raise QwenRuntimeError("exact Qwen container exited")
    _endpoint_identity(state, request_get=request_get)
    return pid


reuse_local_runtime = reuse_qwen_runtime


def qwen_runtime_liveness(
    *,
    state_path: Path = RUNTIME_STATE_FILE,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    try:
        state = current_runtime_state(state_path)
        if state is None:
            return "gone"
        status, _pid, _ = _resolve_container(
            state, command_runner=command_runner, adopt=True, state_path=state_path
        )
        return status
    except Exception:
        return "ambiguous"


def _remove_cidfile(
    state: Mapping[str, Any],
    *,
    recovery_authorized: bool = False,
) -> bool:
    path = _validate_run_dir(state["run_dir"]) / "container.cid"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size > 128
            or (metadata.st_mode & 0o022 and not recovery_authorized)
        ):
            return False
        if not recovery_authorized:
            try:
                value = read_bounded_fd(descriptor, 128).decode("ascii").strip()
                value = _validate_token(value, _CONTAINER_ID_RE, "container ID")
            except (OSError, UnicodeDecodeError, QwenRuntimeError):
                return False
            if state.get("container_id") is not None and value != state["container_id"]:
                return False
        try:
            current = path.lstat()
        except OSError:
            return False
        if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
            return False
        path.unlink()
    except OSError:
        return False
    finally:
        os.close(descriptor)
    return True


def _cleanup_run_directory(state: Mapping[str, Any]) -> bool:
    if not cleanup_local_source_stage(state) or not _remove_cidfile(
        state,
        recovery_authorized=state.get("cidfile_recovery_authorized") is True,
    ):
        return False
    run_dir = _validate_run_dir(state["run_dir"])
    try:
        metadata = run_dir.lstat()
    except FileNotFoundError:
        return True
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        return False
    try:
        with os.scandir(run_dir) as entries:
            if next(entries, None) is not None:
                return False
        run_dir.rmdir()
    except OSError:
        return False
    return True


def stop_qwen_runtime(
    *,
    state_path: Path = RUNTIME_STATE_FILE,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    allow_lost_lease: bool = False,
) -> bool:
    """Stop/remove only the immutable-ID container after full receipt validation."""

    state = current_runtime_state(state_path)
    if state is None:
        return True
    if state["teardown_only"] is True:
        # Persist the one-way schema migration before any Docker or scratch
        # mutation. A crash therefore leaves either the exact schema-6 source
        # receipt or a complete schema-7 teardown journal, never a hybrid.
        _private_json_write(state_path, state)
        state = current_runtime_state(state_path)
        if state is None or state["teardown_only"] is not True:
            return False
    try:
        if (
            docker_client_sha256() != state["docker_sha256"]
            or low_priority_wrapper_sha256() != state["wrapper_sha256"]
        ):
            return False
    except (KeyError, OSError, QwenRuntimeError):
        return False
    lease = current_lease(QWEN_LEASE_FILE)
    if state.get("phase") != "releasing" and (
        lease is None or not runtime_state_matches_lease(state, lease)
    ) and not allow_lost_lease:
        return False
    status, _pid, state = _resolve_container(
        state, command_runner=command_runner, adopt=True, state_path=state_path
    )
    if status == "gone":
        # A pre-create crash is safely absent only after exact nonce search and
        # a healthy daemon.  Journal the same cleanup/release transaction.
        state = {**state, "phase": "releasing", "updated_at": time.time()}
        _private_json_write(state_path, state)
    else:
        container_id = state.get("container_id")
        if container_id is None:
            return False
        state = {**state, "phase": "releasing", "updated_at": time.time()}
        _private_json_write(state_path, state)
        if status == "active":
            stopped = command_runner(
                _docker_command("stop", "--time", "30", str(container_id)),
                env=_docker_cli_environment(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=45,
            )
            if stopped.returncode != 0:
                return False
            item = _docker_inspect(str(container_id), command_runner)
            if item is None:
                return False
            exact_status, _ = _inspect_identity(item, state)
            if exact_status != "exited":
                return False
        # Re-inspect exact immutable ID immediately before removal.  Never use
        # the mutable name as an authorization target.
        item = _docker_inspect(str(container_id), command_runner)
        if item is None:
            return False
        exact_status, _ = _inspect_identity(item, state)
        if exact_status != "exited":
            return False
        removed = command_runner(
            _docker_command("rm", "-v", str(container_id)),
            env=_docker_cli_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if removed.returncode != 0:
            return False
        if _docker_inspect(str(container_id), command_runner) is not None:
            return False
        if _label_candidates(state, command_runner):
            return False
    if not _cleanup_run_directory(state):
        return False
    state = {
        **state,
        "phase": "releasing",
        "container_pid": None,
        "scratch_cleaned": True,
        "updated_at": time.time(),
    }
    _private_json_write(state_path, state)
    return True


def _coordinator_claim_matches(state: Mapping[str, Any]) -> tuple[int, dict[str, Any] | None]:
    capability = _capability_from_receipt(
        state, allow_retired_manifest=True
    )
    try:
        result = _coord("status", "--json", check=False)
    except Exception as exc:
        raise QwenRuntimeError("coordinator release reconciliation is unavailable") from exc
    if result.returncode != 0:
        raise QwenRuntimeError("coordinator release reconciliation failed")
    try:
        inventory = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("coordinator release reconciliation is malformed") from exc
    if not isinstance(inventory, list):
        raise QwenRuntimeError("coordinator release inventory is malformed")
    exact_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    claim_id_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    owner_run_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    exact_targets: list[dict[str, Any]] = []
    for target in inventory:
        if not isinstance(target, dict):
            raise QwenRuntimeError("coordinator release target is malformed")
        target_physical = target.get("physical_gpu")
        saved_target = (
            target.get("host") == state["host"]
            and not isinstance(target_physical, bool)
            and isinstance(target_physical, int)
            and target_physical == state["physical_gpu"]
        )
        if saved_target:
            exact_targets.append(target)
        if "claims" not in target or not isinstance(target["claims"], list):
            if not saved_target:
                # An unrelated worker's unavailable placeholder says nothing
                # about the pinned local Qwen claim and must not wedge release.
                continue
            raise QwenRuntimeError("coordinator release claims are malformed")
        claims = target["claims"]
        for claim in claims:
            if not isinstance(claim, dict):
                raise QwenRuntimeError("coordinator release claim is malformed")
            same_claim = claim.get("claim_id") == state["claim_id"]
            same_owner_run = (
                claim.get("owner") == state["owner"]
                and claim.get("run_dir") == state["run_dir"]
            )
            if same_claim:
                claim_id_matches.append((target, claim))
            if same_owner_run:
                owner_run_matches.append((target, claim))
            if same_claim and same_owner_run:
                exact_matches.append((target, claim))
    if (
        len(exact_targets) != 1
        or exact_targets[0].get("uuid") != state["gpu_uuid"]
    ):
        raise QwenRuntimeError(
            "coordinator inventory lacks the exact saved Qwen GPU target"
        )
    if len(claim_id_matches) > 1:
        raise QwenRuntimeError("saved coordinator claim ID is duplicated globally")
    if len(owner_run_matches) > 1:
        raise QwenRuntimeError("saved coordinator owner/run identity is duplicated")
    if claim_id_matches != exact_matches:
        raise QwenRuntimeError("saved coordinator claim ID moved to a foreign owner/run")
    if owner_run_matches != exact_matches:
        raise QwenRuntimeError("a foreign claim occupies the saved owner/run identity")
    if exact_matches:
        target, claim = exact_matches[0]
        target_total_mib = target.get("memory_total_mib")
        claim_budget_mib = claim.get("vram_budget_mib")
        if (
            target.get("host") != state["host"]
            or isinstance(target.get("physical_gpu"), bool)
            or not isinstance(target.get("physical_gpu"), int)
            or target.get("physical_gpu") != state["physical_gpu"]
            or target.get("physical_gpu") not in capability.allowed_physical_gpus
            or target.get("uuid") != state["gpu_uuid"]
            or isinstance(target_total_mib, bool)
            or not isinstance(target_total_mib, int)
            or target_total_mib != state["memory_total_mib"]
            or claim.get("gpu_uuid") != state["gpu_uuid"]
            or isinstance(claim_budget_mib, bool)
            or not isinstance(claim_budget_mib, int)
            or claim_budget_mib != state["vram_budget_mib"]
            or not _coordinator_status_claim_is_exclusive(
                claim.get("exclusive")
            )
        ):
            raise QwenRuntimeError("coordinator release claim identity changed")
        return 1, claim
    return 0, None


def finalize_releasing_qwen_runtime(
    reason: str = "Aeon Qwen exact runtime stopped",
    *,
    state_path: Path = RUNTIME_STATE_FILE,
) -> bool:
    """Idempotently finish the journaled stop→cleanup→release transaction."""

    state = current_runtime_state(state_path)
    if state is None:
        return current_lease(QWEN_LEASE_FILE) is None
    if state.get("phase") != "releasing" or state.get("scratch_cleaned") is not True:
        return False
    count, _claim = _coordinator_claim_matches(state)
    lease = current_lease(QWEN_LEASE_FILE)
    if lease is not None and not runtime_state_matches_lease(state, lease):
        raise QwenRuntimeError("release journal no longer matches its local lease")
    if count == 1:
        if lease is not None:
            release_vram(
                reason,
                QWEN_LEASE_FILE,
                expected_claim_id=str(state["claim_id"]),
            )
        else:
            result = _coord(
                "release",
                "--claim",
                str(state["claim_id"]),
                "--owner",
                str(state["owner"]),
                "--reason",
                reason,
                check=False,
            )
            detail = ((result.stdout or "") + (result.stderr or "")).lower()
            if result.returncode != 0 and "already released" not in detail:
                raise QwenRuntimeError("journaled coordinator release was not verified")
    elif lease is not None:
        # Coordinator absence plus a still-present local lease is exactly the
        # release→state-unlink crash window.  Remove only the same private lease.
        clear_reconciled_lease_state(
            QWEN_LEASE_FILE,
            expected_claim_id=str(state["claim_id"]),
            expected_owner=str(state["owner"]),
            expected_run_dir=str(state["run_dir"]),
        )
    if _coordinator_claim_matches(state)[0] != 0:
        raise QwenRuntimeError("coordinator claim remains after release")
    clear_runtime_state(state_path)
    return True


def reconcile_gone_qwen_runtime(
    *,
    state_path: Path = RUNTIME_STATE_FILE,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    state = current_runtime_state(state_path)
    if state is None:
        return "cleared" if current_lease(QWEN_LEASE_FILE) is None else "ambiguous"
    if state.get("phase") == "releasing" and state.get("scratch_cleaned") is True:
        return "cleared" if finalize_releasing_qwen_runtime(state_path=state_path) else "ambiguous"
    liveness = qwen_runtime_liveness(state_path=state_path, command_runner=command_runner)
    if liveness == "active":
        return "active"
    if liveness == "ambiguous":
        return "ambiguous"
    if liveness in {"gone", "exited"}:
        if not stop_qwen_runtime(state_path=state_path, command_runner=command_runner):
            return "ambiguous"
        return "cleared" if finalize_releasing_qwen_runtime(state_path=state_path) else "ambiguous"
    return "ambiguous"
