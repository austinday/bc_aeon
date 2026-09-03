from __future__ import annotations

from dataclasses import replace
import errno
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from aeon.core import qwen_flash_next_legacy_recovery as recovery
from fleet_compute.models import ProbeState


def _private(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True)
    path.chmod(0o700)
    return path


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_private(path: Path, value: dict[str, Any]) -> str:
    raw = _json_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    request_overrides: dict[str, Any] | None = None,
    preflight_overrides: dict[str, Any] | None = None,
    source_stage_overrides: dict[str, Any] | None = None,
    spawn_overrides: dict[str, Any] | None = None,
    result_overrides: dict[str, Any] | None = None,
) -> tuple[
    recovery.LegacyBuildRecoveryContract,
    dict[str, Any],
    dict[str, Path],
]:
    base = recovery.LEGACY_RECOVERY_CONTRACT
    root = _private(tmp_path / "private")
    run_dir = _private(root / "runs" / base.runtime_id)
    artifact_dir = _private(root / "artifacts" / base.runtime_id)
    output = _private(artifact_dir / "output")
    request = {
        "schema_version": "aeon-qwen38-flash-next-build-worker-v1",
        "runtime_id": base.runtime_id,
        "job_id": base.job_id,
        "claim_id": base.claim_id,
        "owner": base.owner,
        "host": base.host,
        "hostname": base.hostname,
        "physical_gpu": base.physical_gpu,
        "gpu_uuid": base.gpu_uuid,
        "vram_budget_gb": base.vram_budget_gb,
        "exclusive": True,
        "scratch_path": str(artifact_dir),
        "source_root": "/home/aday/NexusAgentDashboard/bc_aeon",
        "source_files": {
            "aeon/core/qwen_flash_next_build_adapter.py": {
                "sha256": base.adapter_source_sha256,
                "size": 39_451,
            },
            "aeon/scripts/qwen_flash_next_build_worker.py": {
                "sha256": base.worker_source_sha256,
                "size": 49_454,
            },
        },
        "sglang_commit": base.sglang_commit,
        "sglang_image_digest": base.sglang_image_digest,
    }
    if request_overrides:
        request.update(request_overrides)
    request_sha256 = _write_private(run_dir / "qwen-flash-next-build-request.json", request)
    assert (
        _write_private(artifact_dir / "qwen-flash-next-build-request.json", request)
        == request_sha256
    )
    source_stage = {
        "schema_version": "aeon-qwen38-flash-next-trainer-source-v1",
        "source_manifest_sha256": base.source_manifest_sha256,
    }
    if source_stage_overrides:
        source_stage.update(source_stage_overrides)
    preflight = {
        "schema_version": "aeon-qwen38-flash-next-build-worker-v1",
        "request_sha256": request_sha256,
        "source_stage": source_stage,
    }
    if preflight_overrides:
        preflight.update(preflight_overrides)
    preflight_sha256 = _write_private(output / "preflight.json", preflight)
    spawn = {
        "schema_version": "aeon-qwen38-flash-next-build-worker-v1",
        "runtime_id": base.runtime_id,
        "request_sha256": request_sha256,
        "pid": base.historical_pid,
        "created_at": 1.0,
    }
    if spawn_overrides:
        spawn.update(spawn_overrides)
    spawn_sha256 = _write_private(artifact_dir / "spawn.json", spawn)
    result = {
        "schema_version": "aeon-qwen38-flash-next-build-result-v1",
        "success": False,
        "failure_type": "FlashBuildWorkerError",
        "failure": "behavior trainer failed",
        "completed_at": 2.0,
    }
    if result_overrides:
        result.update(result_overrides)
    result_sha256 = _write_private(output / "result.json", result)
    contract = replace(
        base,
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        request_sha256=request_sha256,
        preflight_sha256=preflight_sha256,
        spawn_sha256=spawn_sha256,
        result_sha256=result_sha256,
    )
    runtime = {
        "runtime_id": contract.runtime_id,
        "profile_id": contract.profile_id,
        "deployment_revision": contract.deployment_revision,
        "mode": "batch",
        "state": "quarantined",
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
        "pid": None,
        "process_identity": None,
        "endpoint": None,
        "payload_json": contract.payload_json,
        "process_absent": 0,
    }
    paths = {
        "run_request": run_dir / "qwen-flash-next-build-request.json",
        "artifact_request": artifact_dir / "qwen-flash-next-build-request.json",
        "preflight": output / "preflight.json",
        "spawn": artifact_dir / "spawn.json",
        "result": output / "result.json",
    }
    return contract, runtime, paths


def _tree_snapshot(root: Path) -> tuple[tuple[Any, ...], ...]:
    result: list[tuple[Any, ...]] = []
    for path in sorted((root, *root.rglob("*"))):
        metadata = path.lstat()
        result.append(
            (
                str(path.relative_to(root)),
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                path.read_bytes() if path.is_file() else None,
            )
        )
    return tuple(result)


def test_production_contract_matches_exact_disabled_tombstone() -> None:
    contract = recovery.LEGACY_RECOVERY_CONTRACT
    workspace = Path(__file__).resolve().parents[3]
    tombstone_path = (
        workspace
        / "fleet_compute/profiles.d/"
        "aeon-qwen38-flash-next-build-legacy-tombstone.json"
    )
    raw = tombstone_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "60d142c094920ea3c7eccc2c0c48304c6db2488df424d10e0fe02bf33edfc5ca"
    )
    profile = json.loads(raw)
    identities = profile["artifact_identity"]

    assert profile["profile_id"] == contract.profile_id
    assert profile["enabled"] is False
    assert profile["adapter"] == contract.adapter
    assert identities["legacy_deployment_revision"] == contract.deployment_revision
    assert identities["legacy_request"] == contract.request_sha256
    assert identities["legacy_preflight_receipt"] == contract.preflight_sha256
    assert identities["legacy_spawn_receipt"] == contract.spawn_sha256
    assert identities["legacy_result_receipt"] == contract.result_sha256
    assert identities["source_manifest"] == contract.source_manifest_sha256
    assert identities["adapter_source_at_attempt"] == (
        contract.adapter_source_sha256
    )
    assert identities["worker_source_at_attempt"] == contract.worker_source_sha256


def test_exact_legacy_receipts_prove_typed_absence_without_mutation(
    tmp_path: Path,
) -> None:
    contract, runtime, _paths = _fixture(tmp_path)
    before = _tree_snapshot(tmp_path)

    first = recovery.probe_legacy_pidless_build(
        runtime,
        _contract=contract,
        _absence_check=lambda pid: pid == contract.historical_pid,
    )
    second = recovery.probe_legacy_pidless_build(
        runtime,
        _contract=contract,
        _absence_check=lambda pid: pid == contract.historical_pid,
    )

    assert first == second
    assert first is not None
    assert first.state is ProbeState.ABSENT
    assert first.process_absent is True
    assert first.process_identity_verified is False
    assert first.prelaunch_cleanup_verified is True
    assert "artifacts retained" in first.note
    assert _tree_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("runtime_id", "fr-" + "0" * 32),
        ("deployment_revision", "0" * 64),
        ("mode", "service"),
        ("adapter", "another-adapter"),
        ("job_id", "fj-another"),
        ("owner", "another-owner"),
        ("claim_id", "gc-another"),
        ("host", "192.168.0.178"),
        ("physical_gpu", 1),
        ("gpu_uuid", "GPU-00000000-0000-0000-0000-000000000000"),
        ("vram_budget_gb", 87.0),
        ("exclusive", True),
        ("run_dir", "/home/aday/changed"),
        ("payload_json", "{}"),
        ("state", "running"),
        ("pid", 1_873_217),
        ("process_identity", "published"),
        ("endpoint", "http://127.0.0.1:1"),
    ),
)
def test_every_legacy_runtime_identity_mismatch_is_unknown(
    tmp_path: Path, field: str, changed: Any
) -> None:
    contract, runtime, _paths = _fixture(tmp_path)
    runtime[field] = changed

    result = recovery.probe_legacy_pidless_build(
        runtime,
        _contract=contract,
        _absence_check=lambda _pid: True,
    )

    assert result is not None
    assert result.state is ProbeState.UNKNOWN
    assert result.process_absent is False
    assert result.prelaunch_cleanup_verified is False


def test_nonlegacy_profile_receives_no_recovery_authority(tmp_path: Path) -> None:
    contract, runtime, _paths = _fixture(tmp_path)
    runtime["profile_id"] = "another-profile"

    assert (
        recovery.probe_legacy_pidless_build(
            runtime,
            _contract=contract,
            _absence_check=lambda _pid: True,
        )
        is None
    )


@pytest.mark.parametrize(
    "request_overrides",
    (
        {"schema_version": "changed"},
        {"runtime_id": "fr-" + "1" * 32},
        {"job_id": "fj-changed"},
        {"claim_id": "gc-changed"},
        {"owner": "changed-owner"},
        {"host": "192.168.0.178"},
        {"hostname": "CHANGED"},
        {"physical_gpu": 1},
        {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000"},
        {"vram_budget_gb": 87.0},
        {"exclusive": False},
        {"scratch_path": "/home/aday/changed"},
        {"source_root": "/home/aday/changed"},
        {"sglang_commit": "0" * 40},
        {"sglang_image_digest": "0" * 64},
        {
            "source_files": {
                "aeon/core/qwen_flash_next_build_adapter.py": {
                    "sha256": "0" * 64,
                    "size": 39_451,
                },
                "aeon/scripts/qwen_flash_next_build_worker.py": {
                    "sha256": recovery.LEGACY_RECOVERY_CONTRACT.worker_source_sha256,
                    "size": 49_454,
                },
            }
        },
        {
            "source_files": {
                "aeon/core/qwen_flash_next_build_adapter.py": {
                    "sha256": recovery.LEGACY_RECOVERY_CONTRACT.adapter_source_sha256,
                    "size": 39_451,
                },
                "aeon/scripts/qwen_flash_next_build_worker.py": {
                    "sha256": "0" * 64,
                    "size": 49_454,
                },
            }
        },
    ),
)
def test_rebased_request_receipt_semantic_mismatch_is_unknown(
    tmp_path: Path, request_overrides: dict[str, Any]
) -> None:
    contract, runtime, _paths = _fixture(
        tmp_path, request_overrides=request_overrides
    )

    result = recovery.probe_legacy_pidless_build(
        runtime,
        _contract=contract,
        _absence_check=lambda _pid: True,
    )

    assert result is not None
    assert result.state is ProbeState.UNKNOWN


@pytest.mark.parametrize(
    "fixture_options",
    (
        {"preflight_overrides": {"schema_version": "changed"}},
        {"preflight_overrides": {"request_sha256": "0" * 64}},
        {"source_stage_overrides": {"schema_version": "changed"}},
        {"source_stage_overrides": {"source_manifest_sha256": "0" * 64}},
        {"spawn_overrides": {"schema_version": "changed"}},
        {"spawn_overrides": {"runtime_id": "fr-" + "2" * 32}},
        {"spawn_overrides": {"request_sha256": "0" * 64}},
        {"spawn_overrides": {"pid": 1_873_218}},
        {"result_overrides": {"schema_version": "changed"}},
        {"result_overrides": {"success": True}},
    ),
)
def test_rebased_lifecycle_receipt_semantic_mismatch_is_unknown(
    tmp_path: Path, fixture_options: dict[str, Any]
) -> None:
    contract, runtime, _paths = _fixture(tmp_path, **fixture_options)

    result = recovery.probe_legacy_pidless_build(
        runtime,
        _contract=contract,
        _absence_check=lambda _pid: True,
    )

    assert result is not None
    assert result.state is ProbeState.UNKNOWN


@pytest.mark.parametrize("receipt_name", ("run_request", "preflight", "spawn", "result"))
def test_unrebased_receipt_digest_mismatch_is_unknown(
    tmp_path: Path, receipt_name: str
) -> None:
    contract, runtime, paths = _fixture(tmp_path)
    path = paths[receipt_name]
    path.write_bytes(path.read_bytes() + b" ")
    path.chmod(0o600)

    result = recovery.probe_legacy_pidless_build(
        runtime,
        _contract=contract,
        _absence_check=lambda _pid: True,
    )

    assert result is not None
    assert result.state is ProbeState.UNKNOWN


@pytest.mark.parametrize("unsafe", ("writable", "hardlink", "symlink"))
def test_unsafe_receipt_inode_is_unknown_and_preserved(
    tmp_path: Path, unsafe: str
) -> None:
    contract, runtime, paths = _fixture(tmp_path)
    path = paths["spawn"]
    if unsafe == "writable":
        path.chmod(0o660)
    elif unsafe == "hardlink":
        os.link(path, path.with_suffix(".linked"))
    else:
        preserved = path.with_suffix(".preserved")
        path.rename(preserved)
        path.symlink_to(preserved)
    before = _tree_snapshot(tmp_path)

    result = recovery.probe_legacy_pidless_build(
        runtime,
        _contract=contract,
        _absence_check=lambda _pid: True,
    )

    assert result is not None
    assert result.state is ProbeState.UNKNOWN
    assert _tree_snapshot(tmp_path) == before


def test_live_or_recycled_process_group_is_unknown(tmp_path: Path) -> None:
    contract, runtime, _paths = _fixture(tmp_path)

    result = recovery.probe_legacy_pidless_build(
        runtime,
        _contract=contract,
        _absence_check=lambda _pid: False,
    )

    assert result is not None
    assert result.state is ProbeState.UNKNOWN
    assert "live or recycled" in result.note


def test_historical_pid_and_group_absence_requires_both_proofs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = 123_456
    proc_root = _private(tmp_path / "proc")
    calls: list[tuple[int, int]] = []

    def absent_group(pgid: int, signal_number: int) -> None:
        calls.append((pgid, signal_number))
        raise ProcessLookupError(errno.ESRCH, "absent")

    monkeypatch.setattr(recovery.os, "killpg", absent_group)
    assert recovery.historical_pid_and_group_absent(pid, proc_root=proc_root)
    assert calls == [(pid, 0)]

    _private(proc_root / str(pid))
    calls.clear()
    assert not recovery.historical_pid_and_group_absent(pid, proc_root=proc_root)
    assert calls == []


def test_surviving_group_and_permission_ambiguity_are_not_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = 123_457
    proc_root = _private(tmp_path / "proc")
    monkeypatch.setattr(recovery.os, "killpg", lambda _pgid, _signal: None)
    assert not recovery.historical_pid_and_group_absent(pid, proc_root=proc_root)

    def refused(_pgid: int, _signal_number: int) -> None:
        raise PermissionError(errno.EPERM, "ambiguous")

    monkeypatch.setattr(recovery.os, "killpg", refused)
    with pytest.raises(
        recovery.LegacyFlashNextRecoveryError,
        match="process-group visibility is ambiguous",
    ):
        recovery.historical_pid_and_group_absent(pid, proc_root=proc_root)


def test_receipt_change_during_absence_check_is_unknown(tmp_path: Path) -> None:
    contract, runtime, paths = _fixture(tmp_path)

    def mutate_after_first_audit(_pid: int) -> bool:
        path = paths["result"]
        path.write_bytes(path.read_bytes() + b" ")
        path.chmod(0o600)
        return True

    result = recovery.probe_legacy_pidless_build(
        runtime,
        _contract=contract,
        _absence_check=mutate_after_first_audit,
    )

    assert result is not None
    assert result.state is ProbeState.UNKNOWN
