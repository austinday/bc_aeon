from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aeon.core import qwen_flash_next_mtp_quant_adapter as adapter


PROFILE = (
    Path(__file__).resolve().parents[3]
    / "fleet_compute/profiles.d/aeon-qwen38-flash-next-mtp-nvfp4-build.json"
)


def test_payload_binds_one_clean_source_child(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    closures = tmp_path / "closures"
    derivatives = tmp_path / "derivatives"
    monkeypatch.setattr(adapter, "CLOSURE_ROOT", closures)
    monkeypatch.setattr(adapter, "DERIVATIVE_ROOT", derivatives)
    revision = "a" * 40
    source, manifest, returned_revision, destination = (
        adapter.AeonQwenFlashNextMTPNVFP4Adapter._payload(
            {
                "source_path": str(closures / "closed-source"),
                "source_manifest_sha256": "b" * 64,
                "source_revision": revision,
            }
        )
    )
    assert source == closures / "closed-source"
    assert manifest == "b" * 64
    assert returned_revision == revision
    assert destination.parent == derivatives
    with pytest.raises(adapter.MTPNVFP4FleetError):
        adapter.AeonQwenFlashNextMTPNVFP4Adapter._payload(
            {
                "source_path": str(tmp_path / "raw-hf-staging"),
                "source_manifest_sha256": "b" * 64,
                "source_revision": revision,
            }
        )


def test_profile_is_enabled_closed_and_truthfully_bounded() -> None:
    raw = json.loads(PROFILE.read_text(encoding="utf-8"))
    manifest = raw.pop("manifest_sha256")
    assert manifest == hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert raw["enabled"] is True
    assert raw["adapter"] == adapter.ADAPTER_ID
    assert raw["vram_budget_gb"] <= 88.5
    assert raw["exclusive"] is True
    assert raw["runtime_growth_bytes_max"] == 7_500_000_000
    assert raw["worker_free_reserve_bytes"] == 20_000_000_000
    assert raw["min_shm_free_gb"] == 8.0
    assert raw["artifact_identity"] == adapter.expected_artifact_identity()
    assert raw["placements"] == [
        {"host": adapter.HOST, "physical_gpu": 0, "enabled": True}
    ]


def test_local_contract_uses_run_dir_without_worker_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter.socket, "gethostname", lambda: adapter.HOSTNAME)
    run_dir = Path(adapter.RUN_ROOT) / ("fr-" + "a" * 32)
    sources = adapter._source_manifest()
    profile = SimpleNamespace(
        profile_id=adapter.PROFILE_ID,
        project=adapter.PROJECT,
        adapter=adapter.ADAPTER_ID,
        enabled=True,
        artifact_identity=adapter.expected_artifact_identity(sources),
    )
    lease = SimpleNamespace(
        host=adapter.HOST,
        physical_gpu=0,
        exclusive=True,
        memory_total_mib=94 * 1024,
        vram_budget_gb=adapter.worker.VRAM_CAP_GIB,
        run_dir=str(run_dir),
    )
    context = SimpleNamespace(
        runtime_id=run_dir.name,
        job_id="fj-" + "b" * 32,
        profile=profile,
        lease=lease,
        run_dir=run_dir,
        scratch_path=None,
        canonical_output_path=adapter.ARTIFACT_ROOT / "result",
    )
    adapter.AeonQwenFlashNextMTPNVFP4Adapter._contract(context, sources)
    context.scratch_path = str(run_dir)
    with pytest.raises(adapter.MTPNVFP4FleetError, match="contract changed"):
        adapter.AeonQwenFlashNextMTPNVFP4Adapter._contract(context, sources)


def test_probe_runtime_reconstructs_canonical_path_from_runtime_id() -> None:
    runtime_id = "fr-" + "a" * 32
    pid = 1234
    digest = "b" * 64
    source, request, returned_digest, returned_pid = (
        adapter.AeonQwenFlashNextMTPNVFP4Adapter._runtime(
            {
                "runtime_id": runtime_id,
                "process_identity": f"aeon-mtp-nvfp4:{runtime_id}:{digest}:{pid}",
                "pid": pid,
                "host": adapter.HOST,
                "run_dir": str(Path(adapter.RUN_ROOT) / runtime_id),
            }
        )
    )
    assert source == adapter.ARTIFACT_ROOT / runtime_id / "source"
    assert request == adapter.ARTIFACT_ROOT / runtime_id / "mtp-quant-request.json"
    assert returned_digest == digest
    assert returned_pid == pid


def test_adapter_entry_point_is_registered() -> None:
    setup = (Path(__file__).resolve().parents[2] / "setup.py").read_text(encoding="utf-8")
    assert (
        f'"{adapter.ADAPTER_ID} = '
        'aeon.core.qwen_flash_next_mtp_quant_adapter:create_fleet_adapter"'
    ) in setup


def test_profile_identity_changes_with_any_source_receipt() -> None:
    baseline = adapter.expected_artifact_identity({"one": "a" * 64})
    changed = adapter.expected_artifact_identity({"one": "b" * 64})
    assert baseline["source_manifest"] != changed["source_manifest"]
    assert baseline["modelopt_wheel"] == adapter.converter.base.MODELOPT_WHEEL_SHA256
