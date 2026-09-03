from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aeon.core import qwen_dflash_training_adapter as adapter
from aeon.scripts import qwen_dflash_training_worker as worker


def test_training_dataset_modes_are_exact_subsets() -> None:
    full, full_rows = adapter.AeonQwenDFlashTrainingAdapter._dataset("adapt-v1")
    dpace, dpace_rows = adapter.AeonQwenDFlashTrainingAdapter._dataset(
        "adapt-full-dpace-v2"
    )
    smoke, smoke_rows = adapter.AeonQwenDFlashTrainingAdapter._dataset("smoke")
    assert full_rows == 64
    assert dpace_rows == 256
    assert smoke_rows == 1
    assert full.startswith(smoke)
    assert dpace.startswith(full)
    assert hashlib.sha256(dpace).hexdigest() == adapter.CANONICAL_DATASET_SHA256


def test_training_config_is_exact_target_single_process() -> None:
    scratch = "/home/aday/.local/state/fleet-compute/runs/fr-" + "a" * 32
    config = adapter.AeonQwenDFlashTrainingAdapter._training_config(scratch, "smoke")
    assert "distributed" not in config
    assert config["recipe_args"]["target_model_name_or_path"] == str(
        adapter.worker.TARGET_DIR
    )
    assert config["recipe_args"]["train_data_path"] == f"{scratch}/train.jsonl"
    assert config["recipe_args"]["num_anchors"] == 8
    assert config["recipe_args"]["num_epochs"] == 1
    assert config["recipe_args"]["adaptation_mode"] == (
        "projection-selector-conv-norm-v1"
    )
    calibrated = adapter.AeonQwenDFlashTrainingAdapter._training_config(
        scratch, "calibrate"
    )
    assert calibrated["recipe_args"]["num_anchors"] == 64
    assert calibrated["recipe_args"]["num_epochs"] == 1
    full = adapter.AeonQwenDFlashTrainingAdapter._training_config(
        scratch, "calibrate-full"
    )
    assert full["recipe_args"]["adaptation_mode"] == "all-draft-v1"
    assert full["recipe_args"]["num_anchors"] == 64
    dpace = adapter.AeonQwenDFlashTrainingAdapter._training_config(
        scratch, "adapt-full-dpace-v2"
    )
    assert dpace["recipe_args"]["adaptation_mode"] == "all-draft-v1"
    assert dpace["recipe_args"]["training_objective"] == "dpace-v1"
    assert dpace["recipe_args"]["dpace_alpha"] == 0.5
    assert dpace["recipe_args"]["grad_accumulation_steps"] == 8
    assert dpace["recipe_args"]["num_epochs"] == 2
    assert dpace["recipe_args"]["ckpt_every_steps"] == 4
    assert dpace["optimizer"]["lr"] == 0.0006
    assert dpace["optimizer"]["warmup_ratio"] == 0.04
    assert dpace["checkpoint"]["max_recent_checkpoints"] == 1


def test_training_payload_is_closed() -> None:
    assert adapter.AeonQwenDFlashTrainingAdapter._payload({}) == {
        "run_mode": "smoke"
    }
    with pytest.raises(adapter.QwenDFlashTrainingError):
        adapter.AeonQwenDFlashTrainingAdapter._payload({"run_mode": "unreviewed"})
    with pytest.raises(adapter.QwenDFlashTrainingError):
        adapter.AeonQwenDFlashTrainingAdapter._payload({"run_mode": "smoke", "x": 1})


def test_profile_artifact_identity_is_reproducible() -> None:
    sources = adapter._source_manifest()
    identity = adapter._expected_artifact_identity(sources)
    assert identity["source_manifest"] == adapter._canonical_sha256(sources)
    assert identity["target_tree"] == adapter._canonical_sha256(
        adapter.worker.TARGET_FILES
    )


def test_probe_retries_transport_failure_without_declaring_identity_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = adapter.AeonQwenDFlashTrainingAdapter()
    monkeypatch.setattr(
        instance,
        "_runtime_identity",
        lambda _runtime: ("fr-" + "a" * 32, "b" * 64, 12345),
    )

    def unavailable(*_args, **_kwargs):
        raise adapter.QwenDFlashTrainingTransportError("link unavailable")

    monkeypatch.setattr(instance, "_runtime_action", unavailable)
    with pytest.raises(
        adapter.QwenDFlashTrainingTransportError, match="link unavailable"
    ):
        instance.probe({})


def test_process_exit_between_proc_reads_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def raced_read(_path: Path) -> bytes:
        raise OSError("procfs entry disappeared during read")

    monkeypatch.setattr(Path, "read_bytes", raced_read)
    monkeypatch.setattr(Path, "exists", lambda _path: False)
    assert worker._process_alive({}, 12345) is False


def test_unreadable_existing_process_remains_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable(_path: Path) -> bytes:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    monkeypatch.setattr(Path, "exists", lambda _path: True)
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: None)
    with pytest.raises(worker.TrainingWorkerError, match="identity is unreadable"):
        worker._process_alive({}, 12345)
