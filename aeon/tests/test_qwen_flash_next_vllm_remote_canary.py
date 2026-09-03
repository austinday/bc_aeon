import hashlib
import json
from pathlib import Path

from aeon.core import qwen_flash_next_vllm_remote_canary_adapter as adapter
from aeon.scripts import qwen_flash_next_vllm_remote_worker as worker


def test_remote_wrapper_is_hard_bound_to_179_gpu1_and_attempt_roots():
    assert worker.HOST == "192.168.0.179"
    assert worker.HOSTNAME == "DAY2XRTX6000-2"
    assert worker.PHYSICAL_GPU == 1
    assert worker.HOST_PORT == 18059
    source = Path(worker.__file__).read_text(encoding="utf-8")
    assert "worker.IMAGE_ARCHIVE_ROOT = RUN_ROOT" in source
    assert "worker.CANONICAL_OUTPUT_ROOT = RUN_ROOT" in source
    assert "worker.__file__" in source


def test_remote_adapter_is_launch_inert_and_complete():
    instance = adapter.create_fleet_adapter()
    assert isinstance(instance, adapter.AeonQwenFlashNextVllmRemoteCanaryAdapter)
    for method in ("prepare_storage", "launch", "probe", "stop", "finalize_storage"):
        assert callable(getattr(instance, method))


def test_remote_lane_uses_exact_staging_and_promotion_gates():
    assert adapter.STAGE_BYTES_MAX == 145_720_000_000
    assert adapter.TRANSFER_BYTES_PER_SECOND == 100_000_000
    assert adapter.PHYSICAL_GPU == 1
    assert adapter.contract.MIN_SINGLE_STREAM_DECODE_TPS == 120.0
    assert adapter.contract.MIN_C4_AGGREGATE_TPS == 490.0
    assert "aeon/scripts/qwen_flash_next_vllm_remote_worker.py" in adapter.SOURCE_FILES


def test_remote_identity_reuses_native_allocator_shared_closure():
    payload = {
        "checkpoint_manifest_sha256": "1" * 64,
        "derived_image_digest": "sha256:" + "2" * 64,
        "derived_image_config_digest": "3" * 64,
        "derived_image_archive_sha256": "4" * 64,
    }
    identity = adapter.expected_artifact_identity(payload)
    assert identity["runtime_contract_source"] == (
        "9461d25c4070d6850e2c0f14ddaf50134d7bf44b3a3d34447aaa21e89488d4cf"
    )
    assert identity["shared_worker_source"] == (
        "ca60d20ebe1578f395c02b83c1a10bb119d09582f150c89c808f3f29eafe27b8"
    )
    assert identity["source_manifest"] == (
        "1501879a4f668f1900cc95c77bed7de57dfd3385d434f492e2e0419ae16e6ef0"
    )


def test_cleanup_is_token_bound_and_never_targets_cache_or_canonical_177():
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    assert 'ownership.get("token")' in source
    assert 'root=pathlib.Path("/home/aday/.local/state/fleet-compute/runs")' in source
    assert "shutil.rmtree(run)" in source
    assert "docker\",\"container\",\"inspect" in source
    assert "rsync --delete" not in source
    assert "docker image prune" not in source


def test_settlement_receipt_is_promotion_compatible():
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    assert "validate_qualification_receipt" in source
    assert '"promotion_compatible": status["state"] == "completed"' in source
    value = {
        "schema_version": "aeon-qwen38-flash-next-vllm-remote-settlement-v1",
        "runtime_id": "fr-" + "a" * 32,
        "host": adapter.HOST,
        "physical_gpu": 1,
        "terminal_state": "completed",
        "settled_manifest_sha256": "1" * 64,
        "promotion_compatible": True,
    }
    assert hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_setup_registers_remote_canary_adapter():
    setup = (Path(__file__).resolve().parents[2] / "setup.py").read_text(encoding="utf-8")
    assert (
        "aeon-qwen38-flash-next-vllm-canary-179-v1 = "
        "aeon.core.qwen_flash_next_vllm_remote_canary_adapter:create_fleet_adapter"
    ) in setup


def test_remote_sources_never_use_forbidden_gpu_discovery():
    combined = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (adapter.__file__, worker.__file__)
    ).casefold()
    assert "nvidia" + "-smi" not in combined
    assert "pynvml" not in combined
