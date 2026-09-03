"""Read-only recovery proof for one audited legacy Flash-Next build attempt.

This module deliberately recognizes exactly one historical Fleet runtime.  It is
not a general PID-less recovery mechanism and it never starts, stops, signals
with a nonzero signal, rewrites, or removes anything.  A caller may accept the
typed absence result only through Fleet's existing quarantine/storage workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping

from fleet_compute.models import ProbeResult, ProbeState


class LegacyFlashNextRecoveryError(RuntimeError):
    """The exact historical recovery proof was absent or ambiguous."""


@dataclass(frozen=True)
class LegacyBuildRecoveryContract:
    """Immutable identity and receipts for one audited legacy attempt."""

    runtime_id: str
    profile_id: str
    deployment_revision: str
    adapter: str
    job_id: str
    owner: str
    claim_id: str
    host: str
    hostname: str
    physical_gpu: int
    gpu_uuid: str
    vram_budget_gb: float
    run_dir: Path
    artifact_dir: Path
    payload_json: str
    historical_pid: int
    request_sha256: str
    preflight_sha256: str
    spawn_sha256: str
    result_sha256: str
    source_manifest_sha256: str
    adapter_source_sha256: str
    worker_source_sha256: str
    sglang_commit: str
    sglang_image_digest: str


LEGACY_RECOVERY_CONTRACT = LegacyBuildRecoveryContract(
    runtime_id="fr-f92bf14ea335404987052f2fbf6c8235",
    profile_id="aeon-qwen38-flash-next-build",
    deployment_revision=(
        "2e79363db10facae2415f2e22ab1b605efd6ea4a0e8d08d6f4c129e00eee405c"
    ),
    adapter="aeon-qwen38-flash-next-build-v1",
    job_id="fj-1ee326ce540446098eaf89655b9ba5f9",
    owner=(
        "aeon-qwen38-flash-next-build-day2rtx6000pro-"
        "20260827T031735Z-1064fb"
    ),
    claim_id="gc-20260827T031735Z-95281428",
    host="192.168.0.177",
    hostname="DAY2RTX6000PRO",
    physical_gpu=0,
    gpu_uuid="GPU-2fbc2113-cbb0-c835-8c29-5bf04b8a69be",
    vram_budget_gb=88.0,
    run_dir=Path(
        "/home/aday/.local/state/fleet-compute/runs/"
        "fr-f92bf14ea335404987052f2fbf6c8235"
    ),
    artifact_dir=Path(
        "/home/aday/.local/state/fleet-compute/artifacts/"
        "aeon-qwen38-flash-next-build/"
        "fr-f92bf14ea335404987052f2fbf6c8235"
    ),
    payload_json='{"recipe": "behavior-r4-expert-nvfp4-v1"}',
    historical_pid=1_873_217,
    request_sha256=(
        "add327b84d7a7284a34e8d51cd6006223d87482c0b7831f587cbfd35451bfd40"
    ),
    preflight_sha256=(
        "e6322a702f04927c3ee436c2b4558c2de7334b00e6074a375b09183c855d34eb"
    ),
    spawn_sha256=(
        "6b829fa351f1e18c9294f1097f931ce9c4431e9d6782fcd20c5b53eef2b9e0e4"
    ),
    result_sha256=(
        "070ca195d08b047d8af5c8a3d7a6032f1dde6884753b8073f715f7771f2fa568"
    ),
    source_manifest_sha256=(
        "c447659c92938c913aaca8f37c847a1bad44c070cd76e211056cde1dba9d5670"
    ),
    adapter_source_sha256=(
        "e8bf13db09b360e03746091e94eb03fa3cb47e91e47e4370bc57bad601d877ac"
    ),
    worker_source_sha256=(
        "c2965a4992bb415aadf12c938a0f655c8f15d450e21e2f658fff159700a11cbe"
    ),
    sglang_commit="73a255206f916366c8d26d4022f82ddfb0ab558d",
    sglang_image_digest=(
        "12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1"
    ),
)

_REQUEST_NAME = "qwen-flash-next-build-request.json"
_PREFLIGHT_RELATIVE = Path("output/preflight.json")
_SPAWN_NAME = "spawn.json"
_RESULT_RELATIVE = Path("output/result.json")
_WORKER_SCHEMA = "aeon-qwen38-flash-next-build-worker-v1"
_RESULT_SCHEMA = "aeon-qwen38-flash-next-build-result-v1"
_SOURCE_SCHEMA = "aeon-qwen38-flash-next-trainer-source-v1"
_ALLOWED_STATES = frozenset({"starting", "quarantined"})


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _private_directory(path: Path) -> None:
    """Require an exact owner-private, non-symlink directory."""

    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise LegacyFlashNextRecoveryError(
            f"legacy receipt directory is unavailable: {path}"
        ) from exc
    if (
        resolved != path
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise LegacyFlashNextRecoveryError(
            f"legacy receipt directory is unsafe: {path}"
        )


def _exact_private_json(
    path: Path, expected_sha256: str, *, maximum_bytes: int
) -> tuple[bytes, dict[str, Any]]:
    """Read one exact private receipt without following a final symlink."""

    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LegacyFlashNextRecoveryError(
            f"legacy receipt is unavailable: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_mode & 0o077
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise LegacyFlashNextRecoveryError(
                f"legacy receipt is unsafe: {path}"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise LegacyFlashNextRecoveryError(
                    f"legacy receipt was truncated: {path}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise LegacyFlashNextRecoveryError(
                f"legacy receipt grew while read: {path}"
            )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_identity = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in stable_identity):
        raise LegacyFlashNextRecoveryError(
            f"legacy receipt changed while read: {path}"
        )
    raw = b"".join(chunks)
    if _sha256(raw) != expected_sha256:
        raise LegacyFlashNextRecoveryError(
            f"legacy receipt digest changed: {path}"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LegacyFlashNextRecoveryError(
            f"legacy receipt is malformed: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise LegacyFlashNextRecoveryError(
            f"legacy receipt is not an object: {path}"
        )
    return raw, value


def _require_exact(value: Any, expected: Any, detail: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise LegacyFlashNextRecoveryError(detail)


def _validate_runtime(
    runtime: Mapping[str, Any], contract: LegacyBuildRecoveryContract
) -> None:
    expected = {
        "runtime_id": contract.runtime_id,
        "profile_id": contract.profile_id,
        "deployment_revision": contract.deployment_revision,
        "mode": "batch",
        "adapter": contract.adapter,
        "job_id": contract.job_id,
        "owner": contract.owner,
        "claim_id": contract.claim_id,
        "host": contract.host,
        "physical_gpu": contract.physical_gpu,
        "gpu_uuid": contract.gpu_uuid,
        "vram_budget_gb": contract.vram_budget_gb,
        "exclusive": 1,
        "run_dir": str(contract.run_dir),
        "payload_json": contract.payload_json,
    }
    for name, expected_value in expected.items():
        _require_exact(
            runtime.get(name),
            expected_value,
            f"legacy runtime {name} changed",
        )
    if runtime.get("state") not in _ALLOWED_STATES:
        raise LegacyFlashNextRecoveryError("legacy runtime state is not recoverable")
    for name in ("pid", "process_identity", "endpoint"):
        if runtime.get(name) is not None:
            raise LegacyFlashNextRecoveryError(
                f"legacy runtime {name} was published"
            )


def _validate_request(
    request: Mapping[str, Any], contract: LegacyBuildRecoveryContract
) -> None:
    expected = {
        "schema_version": _WORKER_SCHEMA,
        "runtime_id": contract.runtime_id,
        "job_id": contract.job_id,
        "claim_id": contract.claim_id,
        "owner": contract.owner,
        "host": contract.host,
        "hostname": contract.hostname,
        "physical_gpu": contract.physical_gpu,
        "gpu_uuid": contract.gpu_uuid,
        "vram_budget_gb": contract.vram_budget_gb,
        "exclusive": True,
        "scratch_path": str(contract.artifact_dir),
        "source_root": "/home/aday/NexusAgentDashboard/bc_aeon",
        "sglang_commit": contract.sglang_commit,
        "sglang_image_digest": contract.sglang_image_digest,
    }
    for name, expected_value in expected.items():
        _require_exact(
            request.get(name),
            expected_value,
            f"legacy request {name} changed",
        )
    source_files = request.get("source_files")
    if not isinstance(source_files, dict):
        raise LegacyFlashNextRecoveryError("legacy source closure is malformed")
    expected_sources = {
        "aeon/core/qwen_flash_next_build_adapter.py": (
            contract.adapter_source_sha256
        ),
        "aeon/scripts/qwen_flash_next_build_worker.py": (
            contract.worker_source_sha256
        ),
    }
    for name, expected_sha256 in expected_sources.items():
        receipt = source_files.get(name)
        if not isinstance(receipt, dict):
            raise LegacyFlashNextRecoveryError(
                f"legacy source receipt is absent: {name}"
            )
        _require_exact(
            receipt.get("sha256"),
            expected_sha256,
            f"legacy source digest changed: {name}",
        )


def _validate_receipts(
    contract: LegacyBuildRecoveryContract,
) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    _private_directory(contract.run_dir)
    _private_directory(contract.artifact_dir)
    _private_directory(contract.artifact_dir / "output")
    run_request_raw, run_request = _exact_private_json(
        contract.run_dir / _REQUEST_NAME,
        contract.request_sha256,
        maximum_bytes=128 * 1024,
    )
    artifact_request_raw, artifact_request = _exact_private_json(
        contract.artifact_dir / _REQUEST_NAME,
        contract.request_sha256,
        maximum_bytes=128 * 1024,
    )
    if run_request_raw != artifact_request_raw or run_request != artifact_request:
        raise LegacyFlashNextRecoveryError(
            "legacy durable and canonical requests differ"
        )
    _validate_request(run_request, contract)

    preflight_raw, preflight = _exact_private_json(
        contract.artifact_dir / _PREFLIGHT_RELATIVE,
        contract.preflight_sha256,
        maximum_bytes=64 * 1024,
    )
    _require_exact(
        preflight.get("schema_version"),
        _WORKER_SCHEMA,
        "legacy preflight schema changed",
    )
    _require_exact(
        preflight.get("request_sha256"),
        contract.request_sha256,
        "legacy preflight request identity changed",
    )
    source_stage = preflight.get("source_stage")
    if not isinstance(source_stage, dict):
        raise LegacyFlashNextRecoveryError("legacy source-stage receipt is malformed")
    _require_exact(
        source_stage.get("schema_version"),
        _SOURCE_SCHEMA,
        "legacy source-stage schema changed",
    )
    _require_exact(
        source_stage.get("source_manifest_sha256"),
        contract.source_manifest_sha256,
        "legacy source-stage manifest changed",
    )

    spawn_raw, spawn = _exact_private_json(
        contract.artifact_dir / _SPAWN_NAME,
        contract.spawn_sha256,
        maximum_bytes=16 * 1024,
    )
    expected_spawn = {
        "schema_version": _WORKER_SCHEMA,
        "runtime_id": contract.runtime_id,
        "request_sha256": contract.request_sha256,
        "pid": contract.historical_pid,
    }
    for name, expected_value in expected_spawn.items():
        _require_exact(
            spawn.get(name),
            expected_value,
            f"legacy spawn {name} changed",
        )

    result_raw, result = _exact_private_json(
        contract.artifact_dir / _RESULT_RELATIVE,
        contract.result_sha256,
        maximum_bytes=16 * 1024,
    )
    _require_exact(
        result.get("schema_version"),
        _RESULT_SCHEMA,
        "legacy result schema changed",
    )
    _require_exact(
        result.get("success"),
        False,
        "legacy result is not the audited terminal failure",
    )
    return (
        run_request_raw,
        artifact_request_raw,
        preflight_raw,
        spawn_raw,
        result_raw,
    )


def historical_pid_and_group_absent(
    pid: int, *, proc_root: Path = Path("/proc")
) -> bool:
    """Prove the exact historical PID and process group are both absent.

    Signal zero performs only the kernel liveness/permission check.  A live or
    recycled PID/group, EPERM, or any result other than ESRCH remains ambiguous.
    No nonzero signal is ever delivered.
    """

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise LegacyFlashNextRecoveryError("historical PID is malformed")
    historical_proc = proc_root / str(pid)
    try:
        historical_proc.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise LegacyFlashNextRecoveryError(
            "historical PID visibility is ambiguous"
        ) from exc
    else:
        return False
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise LegacyFlashNextRecoveryError(
            "historical process-group visibility is ambiguous"
        ) from exc
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise LegacyFlashNextRecoveryError(
                "historical process-group liveness check failed"
            ) from exc
    else:
        return False
    try:
        historical_proc.lstat()
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise LegacyFlashNextRecoveryError(
            "historical PID visibility changed during recovery"
        ) from exc
    return False


def probe_legacy_pidless_build(
    runtime: Mapping[str, Any],
    *,
    _contract: LegacyBuildRecoveryContract = LEGACY_RECOVERY_CONTRACT,
    _absence_check: Callable[[int], bool] = historical_pid_and_group_absent,
) -> ProbeResult | None:
    """Return typed absence only for the single exact audited legacy attempt.

    ``None`` means the runtime belongs to another profile and this helper has no
    authority over it.  Every mismatch for the legacy profile is ``UNKNOWN`` so
    Fleet retains quarantine and never releases or replaces ambiguous work.
    """

    if runtime.get("profile_id") != _contract.profile_id:
        return None
    try:
        _validate_runtime(runtime, _contract)
        before = _validate_receipts(_contract)
        try:
            process_absent = _absence_check(_contract.historical_pid)
        except LegacyFlashNextRecoveryError:
            raise
        except Exception as exc:
            raise LegacyFlashNextRecoveryError(
                "historical process-group check failed"
            ) from exc
        if process_absent is not True:
            raise LegacyFlashNextRecoveryError(
                "historical PID or process group is live or recycled"
            )
        after = _validate_receipts(_contract)
        if after != before:
            raise LegacyFlashNextRecoveryError(
                "legacy receipts changed during the absence audit"
            )
    except (LegacyFlashNextRecoveryError, OSError, ValueError, TypeError) as exc:
        return ProbeResult(
            ProbeState.UNKNOWN,
            process_identity_verified=False,
            process_absent=False,
            note=f"legacy Flash-Next recovery refused: {exc}"[:500],
        )
    return ProbeResult(
        ProbeState.ABSENT,
        process_identity_verified=False,
        process_absent=True,
        note=(
            "exact receipt-bound legacy Flash-Next process group is absent; "
            "canonical .177 artifacts retained"
        ),
        prelaunch_cleanup_verified=True,
    )


__all__ = [
    "LEGACY_RECOVERY_CONTRACT",
    "LegacyBuildRecoveryContract",
    "LegacyFlashNextRecoveryError",
    "historical_pid_and_group_absent",
    "probe_legacy_pidless_build",
]
