"""Cooperative GPU leases for Aeon's local ComfyUI runtime.

Fleet GPU identity and availability must come from the coordinator, never from
NVML's visible-device numbering.  This module deliberately keeps the old public
function names so callers outside Aeon fail safe after upgrading.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import socket
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .compute_profile import COMFYUI_PROFILE, QWEN38_VLLM_PROFILE, ComputeProfile
from .qwen_capabilities import (
    QwenCapabilityError,
    QwenRuntimeCapability,
    active_qwen_runtime_capability,
    require_enabled_qwen_target,
    require_qwen_release_candidate_target,
    validate_qwen_capability_receipt,
)


COORD_DIR = Path("/home/aday/website_hosting/ads")
COORD = Path("/home/aday/website_hosting/gpu_coord.py")
LEASE_FILE = Path("/tmp/aeon_comfyui_lease.json")
LEASE_LOCK = Path("/tmp/aeon_comfyui_lease.lock")
QWEN_STATE_ROOT = Path("/home/aday/.aeon/runtime/qwen38")
QWEN_LEASE_FILE = QWEN_STATE_ROOT / "lease.json"
LOCAL_COORD_HOSTNAME = "DAY2RTX6000PRO"
LOCAL_COORD_HOST = "192.168.0.177"
PROJECT = "bc-aeon"
LEASE_RECORD = "lease"
RESERVATION_INTENT_RECORD = "reservation_intent"
STATE_SCHEMA_VERSION = 3
# A coordinator census may include several bounded SSH worker probes.  Keep this
# below the operator-update interval while allowing one slow/unreachable worker
# to be represented as unavailable instead of turning a successful reservation
# into an ambiguous transport timeout.
COORDINATOR_COMMAND_TIMEOUT_SECONDS = 45


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _coordinator_status_exclusive_matches(value: Any, expected: bool) -> bool:
    """Match the coordinator's SQLite-backed status wire exactly.

    Reserve responses expose argparse's JSON boolean, while status claim rows
    expose the canonical SQLite INTEGER 0/1. Keep those schemas distinct so
    truthy coercion cannot change a claim's sharing contract.
    """

    return isinstance(expected, bool) and type(value) is int and value == int(expected)


def _intent_resources_are_positive_and_finite(intent: dict[str, Any]) -> bool:
    return all(
        _is_finite_number(intent.get(key)) and float(intent[key]) > 0
        for key in (
            "vram_budget_gb",
            "min_vram_gb",
            "min_host_memory_gb",
            "min_host_commit_gb",
            "min_disk_free_gb",
            "min_shm_free_gb",
        )
    )


class ReservationQuarantinedError(RuntimeError):
    """Coordinator intent cannot yet be proven claim-free or safely released."""


def _active_qwen_capability() -> tuple[QwenRuntimeCapability, str]:
    try:
        return active_qwen_runtime_capability()
    except QwenCapabilityError as exc:
        raise ReservationQuarantinedError(
            "Qwen runtime capability authorization is unavailable"
        ) from exc


def _qwen_capability_from_record(record: dict[str, Any]) -> QwenRuntimeCapability:
    if record.get("compute_profile") != "qwen38-vllm":
        raise ReservationQuarantinedError("record is not a Qwen reservation")
    is_lease = record.get("record_type") in {None, LEASE_RECORD} and bool(
        record.get("claim_id")
    )
    host = record.get("host") if is_lease else record.get("requested_host")
    physical_gpu = (
        record.get("physical_gpu") if is_lease else record.get("requested_gpu")
    )
    try:
        return validate_qwen_capability_receipt(
            key=record.get("runtime_capability_key"),
            manifest_sha256=record.get("runtime_capability_manifest_sha256"),
            runtime_adapter=record.get("runtime_adapter"),
            host=host,
            physical_gpu=physical_gpu,
            release_gate=record.get("release_gate", False),
        )
    except QwenCapabilityError as exc:
        raise ReservationQuarantinedError(
            "Qwen reservation capability receipt is invalid"
        ) from exc


def _assert_coordinator_host() -> None:
    if socket.gethostname() != LOCAL_COORD_HOSTNAME:
        raise RuntimeError(
            "Aeon GPU coordination must run on DAY2RTX6000PRO (.177); "
            f"this host is {socket.gethostname()!r}."
        )
    if not COORD.is_file():
        raise RuntimeError(f"GPU coordinator is missing: {COORD}")


def _coord(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    _assert_coordinator_host()
    result = subprocess.run(
        ["python3", str(COORD), *args],
        cwd=str(COORD_DIR),
        capture_output=True,
        text=True,
        timeout=COORDINATOR_COMMAND_TIMEOUT_SECONDS,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"GPU coordinator exited {result.returncode}")
    return result


def _lock_path(state_file: Path) -> Path:
    return state_file.with_suffix(state_file.suffix + ".lock")


def _locked_state(state_file: Path = LEASE_FILE) -> tuple[Any, dict[str, Any]]:
    state_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_metadata = state_file.parent.lstat()
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise RuntimeError("Aeon lease-state parent is not a real directory")
    if state_file == QWEN_LEASE_FILE and (
        parent_metadata.st_uid != os.geteuid()
        or parent_metadata.st_mode & 0o077
    ):
        raise RuntimeError("Qwen lease-state parent is not private and owned")
    lock_path = _lock_path(state_file)
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as exc:
        raise RuntimeError("Aeon lease lock is unavailable or unsafe") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
        os.close(descriptor)
        raise RuntimeError("Aeon lease lock is not an owned regular file")
    os.fchmod(descriptor, 0o600)
    lock_fd = os.fdopen(descriptor, "r+")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    state_descriptor = -1
    try:
        try:
            state_descriptor = os.open(
                state_file, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
        except FileNotFoundError:
            return lock_fd, {}
        state_metadata = os.fstat(state_descriptor)
        if (
            not stat.S_ISREG(state_metadata.st_mode)
            or state_metadata.st_uid != os.geteuid()
            or state_metadata.st_mode & 0o077
            or state_metadata.st_size > 1024 * 1024
        ):
            raise RuntimeError("Aeon lease state is not a private owned file")
        with os.fdopen(state_descriptor, "r", encoding="utf-8") as handle:
            state_descriptor = -1
            state = json.load(handle)
        if not isinstance(state, dict):
            raise RuntimeError("Aeon lease state is not an object")
        return lock_fd, state
    except Exception:
        _unlock(lock_fd)
        raise
    finally:
        if state_descriptor >= 0:
            os.close(state_descriptor)


def _unlock(lock_fd: Any) -> None:
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()


def current_lease(state_file: Path = LEASE_FILE) -> dict[str, Any] | None:
    lock_fd, state = _locked_state(state_file)
    try:
        record_type = state.get("record_type")
        return dict(state) if (
            state.get("claim_id")
            and record_type in {None, LEASE_RECORD}
        ) else None
    finally:
        _unlock(lock_fd)


def _save_state(state: dict[str, Any], state_file: Path = LEASE_FILE) -> None:
    """Atomically persist private lease identity without following a symlink."""

    state_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{state_file.name}.", suffix=".tmp", dir=str(state_file.parent)
        )
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                state,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, state_file)
        temp_path = None
        os.chmod(state_file, 0o600)
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _current_record(state_file: Path) -> dict[str, Any]:
    lock_fd, state = _locked_state(state_file)
    try:
        return dict(state)
    finally:
        _unlock(lock_fd)


def _replace_record(
    state_file: Path,
    expected: dict[str, Any],
    replacement: dict[str, Any] | None,
) -> None:
    """Replace one exact private record while holding its cross-process lock."""

    lock_fd, live = _locked_state(state_file)
    try:
        identity_keys = (
            "record_type",
            "owner",
            "run_dir",
            "claim_id",
            "runtime_capability_key",
            "runtime_capability_manifest_sha256",
            "runtime_adapter",
        )
        if any(live.get(key) != expected.get(key) for key in identity_keys):
            raise ReservationQuarantinedError(
                "Aeon reservation state changed during reconciliation"
            )
        if replacement is None:
            state_file.unlink(missing_ok=True)
        else:
            _save_state(replacement, state_file)
    finally:
        _unlock(lock_fd)


def _reservation_intent(
    *,
    owner: str,
    run_dir: str,
    purpose: str,
    profile: ComputeProfile,
    required_gb: float,
    physical_floor_gb: float,
    exclusive: bool,
    host: str | None,
    gpu_id: int | None,
    release_gate_capability_key: str | None = None,
) -> dict[str, Any]:
    if (
        not _is_finite_number(required_gb)
        or float(required_gb) <= 0
        or not _is_finite_number(physical_floor_gb)
        or float(physical_floor_gb) <= 0
    ):
        raise ReservationQuarantinedError(
            "reservation intent has non-finite resource requirements"
        )
    now = time.time()
    capability_receipt: dict[str, Any] = {}
    if profile.key == "qwen38-vllm":
        if host is None or gpu_id is None:
            raise ReservationQuarantinedError("Qwen reservation target is incomplete")
        try:
            if release_gate_capability_key is None:
                capability, manifest_sha256 = require_enabled_qwen_target(
                    host, gpu_id
                )
            else:
                capability, manifest_sha256 = require_qwen_release_candidate_target(
                    release_gate_capability_key, host, gpu_id
                )
        except QwenCapabilityError as exc:
            raise ReservationQuarantinedError(
                "Qwen reservation selector is outside an enabled capability"
            ) from exc
        capability_receipt = {
            "runtime_capability_key": capability.key,
            "runtime_capability_manifest_sha256": manifest_sha256,
            "runtime_adapter": capability.runtime_adapter,
            "release_gate": release_gate_capability_key is not None,
        }
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "record_type": RESERVATION_INTENT_RECORD,
        "owner": owner,
        "project": PROJECT,
        "purpose": purpose,
        "run_dir": run_dir,
        "compute_profile": profile.key,
        "min_host_memory_gb": float(profile.min_host_memory_gb),
        "min_host_commit_gb": float(profile.min_host_commit_gb),
        "min_disk_free_gb": float(profile.min_disk_free_gb),
        "min_shm_free_gb": float(profile.min_shm_free_gb),
        "vram_budget_gb": required_gb,
        "vram_budget_mib": round(required_gb * 1024),
        "min_vram_gb": physical_floor_gb,
        "exclusive": bool(exclusive),
        "requested_host": host,
        "requested_gpu": gpu_id,
        "created_at": now,
        "updated_at": now,
        **capability_receipt,
    }


def _reservation_matches(
    inventory: Any, intent: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not isinstance(inventory, list):
        raise ReservationQuarantinedError("Coordinator inventory is not a list")
    owner_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    known_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    known_claim = intent.get("claim_id")
    qwen_targets: list[dict[str, Any]] = []
    qwen_intent = intent.get("compute_profile") == "qwen38-vllm"
    if qwen_intent:
        _qwen_capability_from_record(intent)
    for target in inventory:
        if not isinstance(target, dict):
            raise ReservationQuarantinedError("Coordinator inventory is malformed")
        requested_qwen_target = (
            qwen_intent
            and target.get("host") == intent.get("requested_host")
            and target.get("physical_gpu") == intent.get("requested_gpu")
        )
        if requested_qwen_target:
            qwen_targets.append(target)
        if "claims" not in target or not isinstance(target["claims"], list):
            if qwen_intent and not requested_qwen_target:
                # Worker probe placeholders are unrelated to this pinned local
                # claim. They are not evidence about it, but must not couple
                # local teardown to every worker's availability either.
                continue
            raise ReservationQuarantinedError("Coordinator claim inventory is malformed")
        claims = target["claims"]
        for claim in claims:
            if not isinstance(claim, dict):
                raise ReservationQuarantinedError("Coordinator claim receipt is malformed")
            if (
                claim.get("owner") == intent.get("owner")
                and claim.get("run_dir") == intent.get("run_dir")
            ):
                owner_matches.append((target, claim))
            if known_claim is not None and claim.get("claim_id") == known_claim:
                known_matches.append((target, claim))
    if qwen_intent:
        if len(qwen_targets) != 1:
            raise ReservationQuarantinedError(
                "Coordinator inventory does not contain one exact requested Qwen GPU"
            )
        target_uuid = qwen_targets[0].get("uuid")
        if not isinstance(target_uuid, str) or not target_uuid.startswith("GPU-"):
            raise ReservationQuarantinedError(
                "Coordinator Qwen target UUID is malformed"
            )
        recovered = intent.get("recovered_lease")
        if known_claim is not None:
            if not isinstance(recovered, dict):
                raise ReservationQuarantinedError(
                    "Known Qwen claim has no durable recovered lease identity"
                )
            if recovered.get("gpu_uuid") != target_uuid:
                raise ReservationQuarantinedError(
                    "Coordinator Qwen target UUID changed from the durable receipt"
                )
    if known_claim is not None:
        if len(known_matches) > 1:
            raise ReservationQuarantinedError(
                "Coordinator duplicated the quarantined claim ID globally"
            )
        if any(claim.get("claim_id") != known_claim for _target, claim in owner_matches):
            raise ReservationQuarantinedError(
                "Coordinator has a different claim for the quarantined owner/run directory"
            )
        if known_matches and (
            not owner_matches
            or known_matches[0] != owner_matches[0]
            or len(owner_matches) != 1
        ):
            raise ReservationQuarantinedError(
                "Quarantined claim ID moved to a different owner/run identity"
            )
        if not known_matches and owner_matches:
            raise ReservationQuarantinedError(
                "Quarantined claim ID is absent but its owner/run identity is occupied"
            )
        return known_matches
    return owner_matches


def _recovered_lease(
    intent: dict[str, Any], target: dict[str, Any], claim: dict[str, Any]
) -> dict[str, Any]:
    capability = (
        _qwen_capability_from_record(intent)
        if intent.get("compute_profile") == "qwen38-vllm"
        else None
    )
    claim_id = str(claim.get("claim_id") or "")
    host = str(target.get("host") or "")
    target_uuid = str(target.get("uuid") or "")
    claim_uuid = str(claim.get("gpu_uuid") or "")
    gpu_uuid = claim_uuid
    physical_gpu = target.get("physical_gpu")
    budget_mib = claim.get("vram_budget_mib")
    intent_budget = intent.get("vram_budget_gb")
    total_mib = target.get("memory_total_mib")
    intent_exclusive = intent.get("exclusive")
    if (
        not claim_id.startswith("gc-")
        or not host
        or (capability is not None and host != capability.host)
        or not target_uuid.startswith("GPU-")
        or not claim_uuid.startswith("GPU-")
        or target_uuid != claim_uuid
        or isinstance(physical_gpu, bool)
        or not isinstance(physical_gpu, int)
        or isinstance(budget_mib, bool)
        or not isinstance(budget_mib, int)
        or not _intent_resources_are_positive_and_finite(intent)
        or isinstance(total_mib, bool)
        or not isinstance(total_mib, int)
        or total_mib < float(intent["min_vram_gb"]) * 1024
        or budget_mib != round(float(intent_budget) * 1024)
        or not isinstance(intent_exclusive, bool)
        or not _coordinator_status_exclusive_matches(
            claim.get("exclusive"), intent_exclusive
        )
        or (
            capability is not None
            and physical_gpu not in capability.allowed_physical_gpus
        )
    ):
        raise ReservationQuarantinedError(
            "Coordinator claim cannot be reconstructed exactly from reservation intent"
        )
    requested_host = intent.get("requested_host")
    requested_gpu = intent.get("requested_gpu")
    if (
        (requested_host is not None and host != requested_host)
        or (requested_gpu is not None and physical_gpu != requested_gpu)
    ):
        raise ReservationQuarantinedError(
            "Recovered coordinator claim violates its persisted selectors"
        )
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "record_type": LEASE_RECORD,
        "claim_id": claim_id,
        "owner": intent["owner"],
        "project": PROJECT,
        "purpose": intent["purpose"],
        "host": host,
        "physical_gpu": physical_gpu,
        "gpu_uuid": gpu_uuid,
        "model": target.get("model"),
        "memory_total_mib": total_mib,
        "vram_budget_mib": budget_mib,
        "vram_budget_gb": float(intent_budget),
        "exclusive": intent_exclusive,
        "run_dir": intent["run_dir"],
        "compute_profile": intent["compute_profile"],
        "min_host_memory_gb": float(intent["min_host_memory_gb"]),
        "min_host_commit_gb": float(intent["min_host_commit_gb"]),
        "min_disk_free_gb": float(intent["min_disk_free_gb"]),
        "min_shm_free_gb": float(intent["min_shm_free_gb"]),
        **(
            {
                "runtime_capability_key": intent["runtime_capability_key"],
                "runtime_capability_manifest_sha256": intent[
                    "runtime_capability_manifest_sha256"
                ],
                "runtime_adapter": intent["runtime_adapter"],
                "release_gate": intent.get("release_gate", False),
            }
            if capability is not None
            else {}
        ),
        "reserved_at": intent.get("created_at") or time.time(),
    }


def _claim_is_proven_unlaunched(
    target: dict[str, Any], claim: dict[str, Any]
) -> bool:
    claim_id = claim.get("claim_id")
    pid = claim.get("pid")
    intents = target.get("intent_processes") or []
    compute = target.get("compute_processes") or []
    if not isinstance(intents, list) or not isinstance(compute, list):
        return False
    if any(
        isinstance(item, dict) and item.get("claim_id") == claim_id
        for item in intents
    ):
        return False
    if pid not in {None, ""} and any(
        isinstance(item, dict) and item.get("pid") == pid
        for item in compute
    ):
        return False
    return target.get("state") in {
        "RESERVED", "SHARED_AVAILABLE", "AVAILABLE"
    }


def reconcile_reservation_intent(
    state_file: Path,
    *,
    reason: str = "Aeon reconciled an interrupted unlaunched reservation",
) -> str:
    """Reconcile one durable intent; never reserve while its outcome is unknown.

    Returns ``clear`` when no claim exists and ``released`` when exactly one
    proven-unlaunched claim was recovered and released. Any ambiguity leaves a
    private quarantine record in place and raises.
    """

    intent = _current_record(state_file)
    if intent.get("record_type") != RESERVATION_INTENT_RECORD:
        raise ReservationQuarantinedError("No reservation intent is available to reconcile")
    try:
        result = _coord("status", "--json", check=False)
    except Exception as exc:
        raise ReservationQuarantinedError(
            "Coordinator status is unavailable; reservation intent remains quarantined"
        ) from exc
    if result.returncode != 0:
        raise ReservationQuarantinedError(
            "Coordinator status failed; reservation intent remains quarantined"
        )
    try:
        inventory = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ReservationQuarantinedError(
            "Coordinator status was malformed; reservation intent remains quarantined"
        ) from exc
    matches = _reservation_matches(inventory, intent)
    if not matches:
        _replace_record(state_file, intent, None)
        return "clear"
    if len(matches) != 1:
        raise ReservationQuarantinedError(
            "Multiple coordinator claims match one reservation intent"
        )
    target, claim = matches[0]
    recovered = _recovered_lease(intent, target, claim)
    claimed_intent = {
        **intent,
        "claim_id": recovered["claim_id"],
        "recovered_lease": recovered,
        "updated_at": time.time(),
    }
    if intent.get("claim_id") is None:
        _replace_record(state_file, intent, claimed_intent)
        intent = claimed_intent
    elif intent.get("claim_id") != recovered["claim_id"]:
        raise ReservationQuarantinedError("Recovered claim identity changed")
    if not _claim_is_proven_unlaunched(target, claim):
        raise ReservationQuarantinedError(
            "Matching reservation has process evidence; refusing release or duplicate reserve"
        )
    release = _coord(
        "release", "--claim", recovered["claim_id"],
        "--owner", recovered["owner"], "--reason", reason,
        check=False,
    )
    if release.returncode != 0 and "already released" not in (
        (release.stdout or "") + (release.stderr or "")
    ):
        raise ReservationQuarantinedError(
            "Recovered claim release was not verified; intent remains quarantined"
        )
    # Do not publish an interrupted reservation as a launchable lease even
    # briefly.  Its exact recovered receipt remains nested in the quarantine
    # record until coordinator release is verified.
    _replace_record(state_file, intent, None)
    return "released"


def cancel_pending_reservation(state_file: Path) -> str:
    """Cancel/reconcile one durable pre-lease intent after foreground unwind.

    A published lease is deliberately left to the exact runtime teardown path.
    Unknown/malformed state remains quarantined rather than being unlinked.
    """

    record = _current_record(state_file)
    if not record:
        return "clear"
    if record.get("record_type") == RESERVATION_INTENT_RECORD:
        return reconcile_reservation_intent(
            state_file,
            reason="Aeon canceled an interrupted unlaunched reservation",
        )
    if record.get("record_type") in {None, LEASE_RECORD} and record.get("claim_id"):
        return "lease"
    raise ReservationQuarantinedError(
        "Canceled admission left unknown reservation state quarantined"
    )


def _update_compute_presence(
    state: str, profile: ComputeProfile | str, summary: str
) -> None:
    """Best-effort safe status; observability can never block coordination."""

    try:
        from .presence import get_active_presence

        presence = get_active_presence()
        if presence is not None:
            presence.update_compute(
                state=state,
                profile=profile.key if isinstance(profile, ComputeProfile) else str(profile),
                summary=summary,
            )
    except Exception:
        pass


def get_real_vram() -> dict[int, float]:
    """Return coordinator-reported free VRAM for ACL-open local devices.

    Kept for a legacy recovery caller.  Numeric keys are diagnostics only; no
    launch path may use them as CUDA selectors.
    """
    result = _coord("status", "--json")
    inventory = json.loads(result.stdout)
    allowed = {"AVAILABLE", "SHARED_AVAILABLE", "RESERVED", "RESERVED_RUNNING"}
    return {
        int(item["physical_gpu"]): float(item["memory_free_mib"]) / 1024.0
        for item in inventory
        if item.get("host") == LOCAL_COORD_HOST
        and item.get("acl") == "OPEN"
        and item.get("state") in allowed
        and item.get("memory_free_mib") is not None
    }


def select_tool_gpu(inventory: list[dict[str, Any]], required_gb: float,
                    qwen_lease: dict[str, Any] | None = None) -> int | None:
    """Choose a physical tool GPU without assuming GPU0/GPU1 exists.

    An exclusive Qwen lease is never a tool co-location candidate.  Select a
    different coordinator-safe local card or wait.

    The returned integer is used only as the coordinator's ``--gpu`` selector;
    the launched process is always pinned to the reservation's stable UUID.
    """
    if not _is_finite_number(required_gb) or float(required_gb) <= 0:
        return None
    required_mib = float(required_gb) * 1024.0
    allowed = {
        "AVAILABLE", "SHARED_AVAILABLE", "RESERVED", "RESERVED_RUNNING",
        "RESERVED_STALE",
    }
    local = [
        item for item in inventory
        if isinstance(item, dict)
        and item.get("host") == LOCAL_COORD_HOST
        and item.get("acl") == "OPEN"
        and item.get("state") in allowed
        and not isinstance(item.get("physical_gpu"), bool)
        and isinstance(item.get("physical_gpu"), int)
    ]
    qwen_gpu = None
    if qwen_lease is not None:
        if not isinstance(qwen_lease, dict):
            return None
        qwen_gpu = qwen_lease.get("physical_gpu")
        try:
            capability = _qwen_capability_from_record(qwen_lease)
            saved_profile = ComputeProfile(
                key=QWEN38_VLLM_PROFILE.key,
                min_host_memory_gb=qwen_lease.get("min_host_memory_gb"),
                min_host_commit_gb=qwen_lease.get("min_host_commit_gb"),
                min_disk_free_gb=qwen_lease.get("min_disk_free_gb"),
                min_shm_free_gb=qwen_lease.get("min_shm_free_gb"),
            )
        except (QwenCapabilityError, ReservationQuarantinedError, ValueError):
            return None
        total_mib = qwen_lease.get("memory_total_mib")
        budget_gb = qwen_lease.get("vram_budget_gb")
        budget_mib = qwen_lease.get("vram_budget_mib")
        if (
            qwen_lease.get("record_type") not in {None, LEASE_RECORD}
            or not qwen_lease.get("claim_id")
            or qwen_lease.get("exclusive") is not True
            or isinstance(qwen_gpu, bool)
            or not isinstance(qwen_gpu, int)
            or qwen_gpu not in capability.allowed_physical_gpus
            or saved_profile != QWEN38_VLLM_PROFILE
            or isinstance(total_mib, bool)
            or not isinstance(total_mib, int)
            or total_mib < capability.min_physical_vram_gb * 1024
            or not _is_finite_number(budget_gb)
            or capability.vram_budget_gb is None
            or abs(float(budget_gb) - capability.vram_budget_gb) > 1e-9
            or isinstance(budget_mib, bool)
            or not isinstance(budget_mib, int)
            or budget_mib != round(float(budget_gb) * 1024)
        ):
            return None
        local = [item for item in local if item.get("physical_gpu") != qwen_gpu]
    if not local:
        return None

    def capacity_mib(item: dict[str, Any]) -> float:
        value = item.get("vram_share_capacity_mib")
        if value is None:
            value = item.get("memory_free_mib")
        if not _is_finite_number(value) or float(value) < 0:
            return 0.0
        return float(value)

    fitting = [item for item in local if capacity_mib(item) >= required_mib]
    if len(fitting) == 1:
        return fitting[0]["physical_gpu"]
    separate = [item for item in fitting if item["physical_gpu"] != qwen_gpu]
    if separate:
        return max(separate, key=capacity_mib)["physical_gpu"]
    # Let the coordinator choose/wait rather than pinning to a currently
    # incapable device on a multi-GPU host.
    return None


def preferred_tool_gpu(required_gb: float) -> int | None:
    result = _coord("status", "--json")
    inventory = json.loads(result.stdout)
    qwen = current_lease(QWEN_LEASE_FILE)
    return select_tool_gpu(inventory, required_gb, qwen)


def _clear_releasable_stale_state(state_file: Path = LEASE_FILE) -> None:
    """Release only our recorded claim, and only if the coordinator verifies exit."""
    state = current_lease(state_file)
    if not state:
        return
    result = _coord(
        "release",
        "--claim", state["claim_id"],
        "--owner", state["owner"],
        "--reason", "Aeon recovered lease state with no ComfyUI container",
        check=False,
    )
    if result.returncode != 0 and "already released" not in (result.stderr + result.stdout):
        raise RuntimeError(
            "A previous Aeon GPU lease still has live or ambiguous process evidence; "
            "refusing to reserve another GPU. " + (result.stderr or result.stdout).strip()
        )
    lock_fd, live = _locked_state(state_file)
    try:
        if live.get("claim_id") == state.get("claim_id"):
            state_file.unlink(missing_ok=True)
    finally:
        _unlock(lock_fd)


def _validate_reservation_receipt(
    payload: Any,
    *,
    intent: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ReservationQuarantinedError("Coordinator reservation receipt is not an object")
    claim_id = str(payload.get("claim_id") or "")
    host = str(payload.get("host") or "")
    gpu_uuid = str(payload.get("gpu_uuid") or "")
    physical_gpu = payload.get("physical_gpu")
    total_mib = payload.get("memory_total_mib")
    budget_mib = payload.get("vram_budget_mib")
    intent_budget = intent.get("vram_budget_gb")
    minimum_vram = intent.get("min_vram_gb")
    intent_exclusive = intent.get("exclusive")
    capability = (
        _qwen_capability_from_record(intent)
        if intent.get("compute_profile") == "qwen38-vllm"
        else None
    )
    if (
        not claim_id.startswith("gc-")
        or payload.get("owner") != intent.get("owner")
        or payload.get("project") != PROJECT
        or payload.get("purpose") != intent.get("purpose")
        or not host
        or (capability is not None and host != capability.host)
        or not gpu_uuid.startswith("GPU-")
        or isinstance(physical_gpu, bool)
        or not isinstance(physical_gpu, int)
        or isinstance(total_mib, bool)
        or not isinstance(total_mib, int)
        or not _intent_resources_are_positive_and_finite(intent)
        or total_mib < float(minimum_vram) * 1024
        or isinstance(budget_mib, bool)
        or not isinstance(budget_mib, int)
        or not _is_finite_number(intent_budget)
        or float(intent_budget) <= 0
        or budget_mib != round(float(intent_budget) * 1024)
        or not isinstance(intent_exclusive, bool)
        or not isinstance(payload.get("exclusive"), bool)
        or payload.get("exclusive") is not intent_exclusive
        or (
            intent.get("requested_host") is not None
            and host != intent.get("requested_host")
        )
        or (
            intent.get("requested_gpu") is not None
            and physical_gpu != intent.get("requested_gpu")
        )
        or (
            capability is not None
            and physical_gpu not in capability.allowed_physical_gpus
        )
    ):
        raise ReservationQuarantinedError(
            "Coordinator reservation receipt does not match its durable intent"
        )
    return {
        **dict(payload),
        "schema_version": STATE_SCHEMA_VERSION,
        "record_type": LEASE_RECORD,
        "owner": intent["owner"],
        "run_dir": intent["run_dir"],
        "vram_budget_gb": float(intent_budget),
        "compute_profile": intent["compute_profile"],
        "min_host_memory_gb": float(intent["min_host_memory_gb"]),
        "min_host_commit_gb": float(intent["min_host_commit_gb"]),
        "min_disk_free_gb": float(intent["min_disk_free_gb"]),
        "min_shm_free_gb": float(intent["min_shm_free_gb"]),
        **(
            {
                "runtime_capability_key": intent["runtime_capability_key"],
                "runtime_capability_manifest_sha256": intent[
                    "runtime_capability_manifest_sha256"
                ],
                "runtime_adapter": intent["runtime_adapter"],
                "release_gate": intent.get("release_gate", False),
            }
            if capability is not None
            else {}
        ),
        "reserved_at": time.time(),
    }


def _reconcile_reservation_attempt(
    state_file: Path,
    intent: dict[str, Any],
    known_lease: dict[str, Any] | None = None,
) -> str:
    """Force an interrupted attempt back into intent form, then reconcile it."""

    live = _current_record(state_file)
    quarantine = {
        **intent,
        "updated_at": time.time(),
    }
    if known_lease is not None:
        quarantine.update({
            "claim_id": known_lease["claim_id"],
            "recovered_lease": known_lease,
        })
    if not live:
        _replace_record(state_file, {}, quarantine)
    elif live.get("record_type") == LEASE_RECORD:
        if known_lease is None or live.get("claim_id") != known_lease.get("claim_id"):
            raise ReservationQuarantinedError(
                "A different lease appeared during reservation reconciliation"
            )
        _replace_record(state_file, live, quarantine)
    elif live.get("record_type") == RESERVATION_INTENT_RECORD:
        if (
            live.get("owner") != intent.get("owner")
            or live.get("run_dir") != intent.get("run_dir")
            or (
                live.get("claim_id") is not None
                and known_lease is not None
                and live.get("claim_id") != known_lease.get("claim_id")
            )
        ):
            raise ReservationQuarantinedError(
                "Reservation intent identity changed during reconciliation"
            )
        if known_lease is not None and live.get("claim_id") is None:
            _replace_record(state_file, live, quarantine)
    else:
        raise ReservationQuarantinedError(
            "Unknown reservation record prevents safe reconciliation"
        )
    return reconcile_reservation_intent(state_file)


def reserve_named_lease(*, required_gb: float, purpose: str, state_file: Path,
                        profile: ComputeProfile = COMFYUI_PROFILE,
                        timeout: int = 600, gpu_id: int | None = None,
                        exclusive: bool = False,
                        host: str | None = LOCAL_COORD_HOST,
                        min_vram_gb: float | None = None,
                        run_dir_root: Path | None = None,
                        durable_wait: bool = False,
                        release_gate_capability_key: str | None = None,
                        sleep_func: Callable[[float], None] | None = None) -> dict[str, Any]:
    """Reserve a lease and persist its exact UUID binding.

    ComfyUI callers keep the local-host default. Qwen callers provide one exact
    enabled capability host/GPU; the higher-level placement loop tries those
    capabilities in preference order. ``durable_wait`` keeps the owning
    foreground Aeon process in a cancelable, bounded-backoff admission loop; it
    creates no daemon and holds no lease between unsuccessful reserve calls.
    """
    if isinstance(required_gb, bool):
        raise ValueError("required_gb must be positive")
    required_gb = float(required_gb)
    if not math.isfinite(required_gb) or required_gb <= 0:
        raise ValueError("required_gb must be positive")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout < 0
    ):
        raise ValueError("timeout must be non-negative")
    if isinstance(min_vram_gb, bool):
        raise ValueError("min_vram_gb must be positive")
    requested_floor = required_gb if min_vram_gb is None else float(min_vram_gb)
    if not math.isfinite(requested_floor) or requested_floor <= 0:
        raise ValueError("min_vram_gb must be positive")
    physical_floor_gb = max(1.0, requested_floor)
    if not isinstance(exclusive, bool):
        raise ValueError("exclusive must be a boolean")
    if gpu_id is not None and (
        isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0
    ):
        raise ValueError("gpu_id must be a physical GPU integer")
    if profile.key == "qwen38-vllm":
        if host is None or gpu_id is None:
            raise ValueError("Qwen requires an exact enabled host/GPU target")
        try:
            if release_gate_capability_key is None:
                capability, _manifest_sha256 = require_enabled_qwen_target(
                    host, gpu_id
                )
            else:
                capability, _manifest_sha256 = require_qwen_release_candidate_target(
                    release_gate_capability_key, host, gpu_id
                )
        except QwenCapabilityError as exc:
            raise ValueError("Qwen target has no enabled release capability") from exc
        if exclusive is not capability.exclusive:
            raise ValueError("Qwen requires an exclusive coordinator lease")
        if (
            capability.vram_budget_gb is None
            or abs(required_gb - capability.vram_budget_gb) > 1e-9
            or physical_floor_gb < capability.min_physical_vram_gb
        ):
            raise ValueError(
                "Qwen request differs from the enabled capability release profile"
            )
        if gpu_id not in capability.allowed_physical_gpus:
            raise ValueError("Qwen GPU selector is outside the enabled capability")
    elif release_gate_capability_key is not None:
        raise ValueError("release gating is only valid for the Qwen profile")
    sleeper = sleep_func or time.sleep
    record = _current_record(state_file)
    if record.get("record_type") == RESERVATION_INTENT_RECORD:
        reconcile_reservation_intent(state_file)
        record = {}
    elif record and not current_lease(state_file):
        raise ReservationQuarantinedError(
            "Unknown Aeon reservation state must be reviewed before a new reserve"
        )
    if current_lease(state_file):
        _clear_releasable_stale_state(state_file)

    owner = _coord("new-owner", "--project", PROJECT).stdout.strip()
    if not owner or len(owner) > 200 or any(ch.isspace() for ch in owner):
        raise RuntimeError("GPU coordinator returned an invalid owner identity")
    run_dir = str(
        (run_dir_root or Path("/tmp")) / f"aeon-{profile.key}-{owner}"
    )
    intent = _reservation_intent(
        owner=owner,
        run_dir=run_dir,
        purpose=purpose,
        profile=profile,
        required_gb=required_gb,
        physical_floor_gb=physical_floor_gb,
        exclusive=exclusive,
        host=host,
        gpu_id=gpu_id,
        release_gate_capability_key=release_gate_capability_key,
    )
    _replace_record(state_file, {}, intent)
    deadline = None if durable_wait else time.monotonic() + timeout

    def waiting_summary(delay: float | None = None) -> str:
        if profile.key == "qwen38-vllm":
            base = (
                f"Waiting for Qwen-compatible compute: {required_gb:g} GiB "
                f"planned peak on {capability.host} GPU "
                f"{gpu_id} (>={physical_floor_gb:g} GiB class) plus host floors."
            )
        else:
            base = (
                f"Waiting for admissible {profile.key} compute on the approved host; "
                "fleet-visible GPUs are not necessarily compatible or allocatable."
            )
        if delay is not None:
            base += f" Retrying in {int(delay)} seconds."
        return base

    _update_compute_presence(
        "waiting_for_compute",
        profile,
        waiting_summary(),
    )
    backoff_seconds = 15.0
    while True:
        args = [
            "reserve", "--owner", owner, "--project", PROJECT,
            "--purpose", purpose,
            "--min-vram-gb", f"{physical_floor_gb:g}",
            "--vram-budget-gb", f"{required_gb:g}",
            "--run-dir", run_dir,
            "--note", (
                "runtime reservation; UUID-pinned and hard-capped"
                if profile.key != "qwen38-vllm"
                else (
                    "exclusive Qwen runtime; UUID-pinned, capability-bound measured "
                    "plan, 6 GiB reserve"
                )
            ),
            "--json",
        ]
        if host is not None:
            args[args.index("--min-vram-gb"):args.index("--min-vram-gb")] = [
                "--host", str(host)
            ]
        args.extend(profile.coordinator_args())
        if exclusive:
            args.append("--exclusive")
        if gpu_id is not None:
            args[args.index("--min-vram-gb"):args.index("--min-vram-gb")] = ["--gpu", str(gpu_id)]
        try:
            result = _coord(*args, check=False)
        except BaseException as exc:
            _update_compute_presence(
                "unavailable",
                profile,
                "Coordinator reserve was interrupted; its durable intent is being reconciled.",
            )
            try:
                _reconcile_reservation_attempt(state_file, intent)
            except Exception as reconcile_exc:
                raise ReservationQuarantinedError(
                    "Coordinator reserve outcome is ambiguous; intent remains quarantined"
                ) from reconcile_exc
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError(
                "Coordinator reserve failed after a claim-free reconciliation"
            ) from exc
        if result.returncode == 0:
            parsed: Any = None
            try:
                parsed = json.loads(result.stdout)
                lease = _validate_reservation_receipt(parsed, intent=intent)
            except (
                TypeError, ValueError, json.JSONDecodeError,
                ReservationQuarantinedError,
            ) as exc:
                _update_compute_presence(
                    "unavailable",
                    profile,
                    "Coordinator returned an invalid receipt; its durable intent is quarantined.",
                )
                try:
                    _reconcile_reservation_attempt(state_file, intent)
                except Exception as reconcile_exc:
                    raise ReservationQuarantinedError(
                        "Invalid successful receipt could not be reconciled safely"
                    ) from reconcile_exc
                raise RuntimeError(
                    "GPU coordinator returned an invalid successful reservation receipt"
                ) from exc
            try:
                claimed_intent = {
                    **intent,
                    "claim_id": lease["claim_id"],
                    "recovered_lease": lease,
                    "updated_at": time.time(),
                }
                # The claim identity reaches durable storage before the record
                # is promoted to a launchable lease. A crash at either atomic
                # replace is therefore reconciled by owner/run-dir/claim.
                _replace_record(state_file, intent, claimed_intent)
                _replace_record(state_file, claimed_intent, lease)
            except BaseException as exc:
                _update_compute_presence(
                    "unavailable",
                    profile,
                    "Reservation persistence was interrupted; exact reconciliation is required.",
                )
                try:
                    _reconcile_reservation_attempt(state_file, intent, lease)
                except Exception as reconcile_exc:
                    raise ReservationQuarantinedError(
                        "Persisted reservation could not be safely reconciled"
                    ) from reconcile_exc
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                raise RuntimeError(
                    "Reservation persistence failed after safe claim reconciliation"
                ) from exc
            _update_compute_presence(
                "allocated",
                profile,
                (
                    f"{profile.key} compute allocated on "
                    f"{str(lease.get('host') or 'an approved host')} after coordinator "
                    f"admission ({required_gb:g} GiB "
                    + ("planned peak" if profile.key == "qwen38-vllm" else "cap")
                    + f", >={physical_floor_gb:g} GiB class)."
                ),
            )
            return lease
        if result.returncode not in {2, 3}:
            _update_compute_presence(
                "unavailable",
                profile,
                "Coordinator admission returned an ambiguous result; intent is being reconciled.",
            )
            try:
                _reconcile_reservation_attempt(state_file, intent)
            except Exception as reconcile_exc:
                raise ReservationQuarantinedError(
                    "Ambiguous coordinator result remains quarantined"
                ) from reconcile_exc
            raise RuntimeError(
                "GPU coordinator reservation failed after claim-free reconciliation"
            )
        if deadline is not None and time.monotonic() >= deadline:
            _replace_record(state_file, intent, None)
            _update_compute_presence(
                "unavailable",
                profile,
                "Compute admission timed out; resume the Aeon session to retry",
            )
            raise TimeoutError(
                f"Timed out waiting for a coordinator-approved {required_gb:g}GB "
                + (
                    f"lease on {host}; " if host is not None
                    else "lease on a compatible host; "
                )
                + "rented, reserved, and ambiguous GPUs were left untouched."
            )
        delay = backoff_seconds
        if deadline is not None:
            delay = min(delay, max(0.1, deadline - time.monotonic()))
        _update_compute_presence(
            "waiting_for_compute", profile, waiting_summary(delay)
        )
        try:
            sleeper(delay)
        except (KeyboardInterrupt, SystemExit):
            _replace_record(state_file, intent, None)
            _update_compute_presence(
                "unavailable",
                profile,
                "Compute wait was canceled; no coordinator lease is held.",
            )
            raise
        backoff_seconds = min(120.0, backoff_seconds * 2.0)


def wait_for_vram(required_gb: float, timeout: int = 600,
                  gpu_id: int | None = None) -> dict[str, Any]:
    if gpu_id is None:
        gpu_id = preferred_tool_gpu(required_gb)
    lease = reserve_named_lease(
        required_gb=required_gb,
        purpose="Aeon ComfyUI image, edit, or video tool",
        state_file=LEASE_FILE,
        profile=COMFYUI_PROFILE,
        timeout=timeout,
        gpu_id=gpu_id,
    )
    return lease


def heartbeat_vram(pid: int | None = None, note: str = "Aeon ComfyUI is healthy",
                   state_file: Path = LEASE_FILE) -> None:
    state = current_lease(state_file)
    if not state:
        raise RuntimeError("Healthy ComfyUI has no recorded coordinator lease.")
    profile_key = str(state.get("compute_profile") or "aeon-runtime")
    capability = (
        _qwen_capability_from_record(state)
        if profile_key == "qwen38-vllm"
        else None
    )
    args = [
        "heartbeat", "--claim", state["claim_id"], "--owner", state["owner"],
        "--run-dir", state["run_dir"], "--note", note,
    ]
    if pid:
        args += ["--pid", str(int(pid))]
    _coord(*args)
    host = str(state.get("host") or "an approved host")
    if profile_key == "qwen38-vllm":
        if capability is None:
            raise ReservationQuarantinedError(
                "Qwen heartbeat has no runtime capability"
            )
        summary = (
            f"Qwen compute remains allocated on {host} after coordinator heartbeat "
            f"({float(state.get('vram_budget_gb') or 0):g} GiB planned peak, "
            f">={capability.min_physical_vram_gb:g} GiB class, exclusive)."
        )
    else:
        summary = f"{profile_key} compute remains allocated on {host} after coordinator heartbeat."
    _update_compute_presence(
        "allocated",
        profile_key,
        summary,
    )


class PeriodicLeaseHeartbeat:
    """Heartbeat one recorded claim for the lifetime of an owning process.

    This is an in-process helper, not a daemon or allocator. Multiple Aeon
    sessions may heartbeat the same exclusive Qwen runtime claim; this is owner
    reference sharing, never GPU co-location. Each stops its own thread before
    unregistering from the model registry.
    """

    def __init__(
        self,
        *,
        state_file: Path,
        note: str,
        pid_provider: Callable[[], int | None] | None = None,
        interval_seconds: float = 300,
        require_pid: bool = False,
        promote_when_pid_available: bool = False,
        heartbeat_func: Callable[..., None] = heartbeat_vram,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        if not 1 <= float(interval_seconds) <= 600:
            raise ValueError("lease heartbeat interval must be between 1 and 600 seconds")
        self.state_file = Path(state_file)
        self.note = str(note)
        self.pid_provider = pid_provider
        self.interval_seconds = float(interval_seconds)
        self.require_pid = bool(require_pid)
        self.promote_when_pid_available = bool(promote_when_pid_available)
        self._heartbeat_func = heartbeat_func
        self._on_error = on_error
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mode_lock = threading.RLock()
        self._failure = threading.Event()
        self._last_exception: Exception | None = None
        self.last_error: str = ""

    def beat_once(self) -> None:
        try:
            with self._mode_lock:
                pid = self.pid_provider() if self.pid_provider is not None else None
                if self.require_pid and pid is None:
                    raise RuntimeError("active Qwen heartbeat has no verified container PID")
                # Always bind a heartbeat to a verified PID when the provider
                # has one.  The pre-container staging phase is PID-less only
                # while no exact container exists yet.
                heartbeat_pid = pid if pid is not None else None
                self._heartbeat_func(heartbeat_pid, self.note, self.state_file)
                if self.promote_when_pid_available and pid is not None:
                    # No pid-less beat can pass this lock after the first exact
                    # PID heartbeat succeeds.
                    self.require_pid = True
                    self.promote_when_pid_available = False
                self.last_error = ""
        except Exception as exc:
            self._record_failure(exc)
            raise

    def promote_to_exact_pid(
        self, pid_provider: Callable[[], int | None] | None = None
    ) -> int:
        """Atomically heartbeat and enter the exact-container-PID phase."""

        if pid_provider is not None:
            self.pid_provider = pid_provider
        with self._mode_lock:
            pid = self.pid_provider() if self.pid_provider is not None else None
            if pid is None:
                exc = RuntimeError("Qwen runtime has no exact container PID to heartbeat")
                self._record_failure(exc)
                raise exc
            try:
                self._heartbeat_func(pid, self.note, self.state_file)
            except Exception as exc:
                self._record_failure(exc)
                raise
            self.require_pid = True
            self.promote_when_pid_available = False
            self.last_error = ""
            return int(pid)

    def _record_failure(self, exc: Exception) -> None:
        self._last_exception = exc
        self._failure.set()
        self.last_error = type(exc).__name__
        try:
            state = current_lease(self.state_file) or {}
            profile = str(state.get("compute_profile") or "aeon-runtime")
            _update_compute_presence(
                "unavailable",
                profile,
                "Coordinator heartbeat failed; runtime identity requires foreground reconciliation.",
            )
        except Exception:
            pass
        try:
            if self._on_error is not None:
                self._on_error(type(exc).__name__)
        except Exception:
            pass

    def raise_if_failed(self) -> None:
        if not self._failure.is_set():
            return
        error = RuntimeError(
            f"lease heartbeat failed ({self.last_error or 'unknown error'})"
        )
        raise error from self._last_exception

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.beat_once()
            except Exception:
                # A failed claim heartbeat is a foreground reconciliation event,
                # not permission for a background retry loop.
                return

    def start(self, *, immediate: bool = False) -> "PeriodicLeaseHeartbeat":
        if self._thread is not None and self._thread.is_alive():
            return self
        if immediate:
            self.beat_once()
        self._stop.clear()
        self._failure.clear()
        self._last_exception = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"aeon-lease-heartbeat-{self.state_file.stem}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
        self._thread = None


def release_vram(
    reason: str = "Aeon ComfyUI stopped and exited",
    state_file: Path = LEASE_FILE,
    *,
    expected_claim_id: str | None = None,
) -> None:
    """Release the recorded claim after the coordinator verifies process exit."""
    state = current_lease(state_file)
    if not state:
        return
    if expected_claim_id is not None and state.get("claim_id") != expected_claim_id:
        raise RuntimeError("Aeon lease identity changed before release")
    result = _coord(
        "release", "--claim", state["claim_id"], "--owner", state["owner"],
        "--reason", reason,
        check=False,
    )
    detail = ((result.stdout or "") + (result.stderr or "")).lower()
    if result.returncode != 0 and "already released" not in detail:
        raise RuntimeError(
            "Coordinator did not verify exact claim release; preserving lease state"
        )
    lock_fd, live = _locked_state(state_file)
    try:
        if not live:
            return
        identity_keys = ("claim_id", "owner", "run_dir")
        if (
            live.get("record_type") not in {None, LEASE_RECORD}
            or any(live.get(key) != state.get(key) for key in identity_keys)
        ):
            raise ReservationQuarantinedError(
                "Aeon lease identity changed after coordinator release"
            )
        state_file.unlink(missing_ok=True)
    finally:
        _unlock(lock_fd)
    # Keep the process-level dashboard summary truthful when one of the two
    # local runtimes is released but the other remains allocated.
    other_file = LEASE_FILE if state_file == QWEN_LEASE_FILE else QWEN_LEASE_FILE
    try:
        remaining = current_lease(other_file)
    except Exception:
        # Presence is observational and must not turn a coordinator-confirmed
        # release into an apparent lifecycle failure after this exact state was
        # already removed. Preserve uncertainty in the summary and let the
        # caller finish clearing its verified-stopped runtime receipt.
        _update_compute_presence(
            "unavailable",
            str(state.get("compute_profile") or "aeon-runtime"),
            "Exact compute claim was released; another runtime state is unreadable",
        )
        return
    if remaining:
        remaining_profile = str(
            remaining.get("compute_profile") or "aeon-runtime"
        )
        _update_compute_presence(
            "allocated",
            remaining_profile,
            (
                f"{remaining_profile} compute remains allocated on "
                f"{str(remaining.get('host') or 'an approved host')}."
            ),
        )
    else:
        _update_compute_presence(
            "idle",
            str(state.get("compute_profile") or "aeon-runtime"),
            "No active Aeon compute allocation",
        )


def clear_reconciled_lease_state(
    state_file: Path,
    *,
    expected_claim_id: str,
    expected_owner: str,
    expected_run_dir: str,
) -> None:
    """Clear one private receipt after exact coordinator absence is proven.

    This is deliberately separate from :func:`release_vram`: callers may use it
    only after coordinator status proves that the saved claim is already absent.
    The cross-process lease lock and all release identity fields prevent a stale
    recovery process from deleting a newer or foreign receipt.
    """

    lock_fd, live = _locked_state(state_file)
    try:
        if not live:
            return
        if live.get("record_type") not in {None, LEASE_RECORD}:
            raise ReservationQuarantinedError(
                "reconciled lease state is not an exact lease receipt"
            )
        expected = {
            "claim_id": expected_claim_id,
            "owner": expected_owner,
            "run_dir": expected_run_dir,
        }
        if any(live.get(key) != value for key, value in expected.items()):
            raise ReservationQuarantinedError(
                "reconciled lease identity changed before local receipt cleanup"
            )
        state_file.unlink(missing_ok=True)
    finally:
        _unlock(lock_fd)
