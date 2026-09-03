from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aeon.core import qwen_flash_next_vllm_contract as contract
from aeon.core import qwen_flash_next_vllm_canary_adapter as canary_adapter
from aeon.core import qwen_flash_next_vllm_service_binding as binding


_TEST_BASE_IMAGE = "sha256:" + "e" * 64


def _write_private(path: Path, value: object) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification() -> dict:
    return {
        "schema_version": contract.QUALIFICATION_SCHEMA,
        "runtime": {
            **contract.expected_runtime(),
            "checkpoint_repository": contract.CHECKPOINT_REPOSITORY,
            "checkpoint_revision": contract.CHECKPOINT_REVISION,
            "base_image_amd64_digest": _TEST_BASE_IMAGE,
            "served_model": contract.SERVED_MODEL,
            "host": contract.HOST,
            "physical_gpu": contract.PHYSICAL_GPU,
            "exclusive_lease": True,
            "vram_cap_gib": contract.VRAM_CAP_GIB,
        },
        "placement": {
            "transformer_weights": "cuda",
            "mtp_weights": "cuda",
            "lm_head": "cuda",
            "vision_weights": "cuda",
            "ple_table": "cpu_worker_pinned_h2d",
            "other_cpu_model_components": [],
        },
        "performance": {
            "single_stream_decode_tps": 120.0,
            "single_stream_measured_after_prefill": True,
            "c2_aggregate_completion_tps": 1.0,
        },
        "mtp": {
            "draft_tokens": 1,
            "accepted_tokens": 1,
            "causal_equivalence_passed": True,
        },
        "semantic": {
            "behavior_low_refusal_passed": True,
            "behavior_non_harmful_coherence_passed": True,
            "behavior_boundary_caution_passed": True,
            "harmful_prompt_diagnostic": {
                "schema_version": contract.HARMFUL_DIAGNOSTIC_SCHEMA,
                "prompt_count": 4,
                "completed_count": 4,
                "refusal_or_safe_redirect_count": 0,
                "operational_detail_count": 4,
                "response_sha256": ["c" * 64] * 4,
                "response_utf8_bytes": [128] * 4,
            },
            "multimodal_passed": True,
            "tool_call_passed": True,
            "reasoning_parser_passed": True,
            "long_context_passed": True,
            "long_context_prompt_tokens": 120_000,
        },
        "capacity": {
            "max_model_len": contract.expected_runtime()["max_model_len"],
            "kv_cache_memory_bytes": contract.KV_CACHE_MEMORY_BYTES,
            "startup_max_model_len_validated": True,
        },
        "cuda_sampling": {
            "peak_used_gib": 88.0,
            "minimum_reserve_gib": 6.0,
            "cadence_density": 0.9,
            "maximum_gap_seconds": 2.0,
        },
        "process_identity_verified": True,
        "semantic_readiness_verified": True,
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict, dict[str, Path]]:
    roots = {
        "canary": tmp_path / "canary",
        "checkpoint": tmp_path / "models",
        "images": tmp_path / "images",
    }
    qualification = roots["canary"] / "fr-" / "output" / "qualification.json"
    qualification_sha = _write_private(qualification, _qualification())
    checkpoint = roots["checkpoint"] / "candidate"
    manifest = checkpoint / "SHA256SUMS"
    manifest_sha = _write_private(manifest, {"checkpoint": "exact"})
    archive = roots["images"] / "candidate.tar"
    archive_sha = _write_private(archive, {"image": "exact"})
    image = "a" * 64
    canary_identity = {name: "b" * 64 for name in binding.CANARY_IDENTITY_FIELDS}
    canary_identity.update(
        checkpoint_manifest=manifest_sha,
        derived_image=image,
        derived_image_config=image,
        derived_image_archive=archive_sha,
    )
    value = {
        "schema_version": binding.BINDING_SCHEMA,
        "complete": True,
        "profile_id": binding.PROFILE_ID,
        "service_id": binding.SERVICE_ID,
        "host": contract.HOST,
        "physical_gpu": contract.PHYSICAL_GPU,
        "vram_cap_gib": contract.VRAM_CAP_GIB,
        "runtime": contract.expected_runtime(),
        "qualification_receipt": str(qualification),
        "qualification_receipt_sha256": qualification_sha,
        "checkpoint_path": str(checkpoint),
        "checkpoint_manifest_path": str(manifest),
        "checkpoint_manifest_sha256": manifest_sha,
        "derived_image_digest": f"sha256:{image}",
        "derived_image_config_digest": image,
        "derived_image_archive_path": str(archive),
        "derived_image_archive_sha256": archive_sha,
        "canary_artifact_identity": canary_identity,
    }
    path = tmp_path / "binding.json"
    _write_private(path, value)
    return path, value, roots


def _load(path: Path, roots: dict[str, Path]) -> binding.VllmServiceBinding:
    with (
        patch.object(binding, "CANARY_OUTPUT_ROOT", roots["canary"]),
        patch.object(binding, "CHECKPOINT_ROOT", roots["checkpoint"]),
        patch.object(binding, "IMAGE_ARCHIVE_ROOT", roots["images"]),
        patch.object(contract, "BASE_IMAGE_AMD64_DIGEST", _TEST_BASE_IMAGE),
        patch.object(contract, "CHECKPOINT_FILE_COUNT", 1),
        patch.object(binding.canary_worker, "_verify_checkpoint_manifest"),
        patch.object(
            canary_adapter,
            "expected_artifact_identity",
            return_value=json.loads(path.read_text())["canary_artifact_identity"],
        ),
        patch.object(
            binding.canary_worker,
            "_oci_identity",
            return_value=("sha256:" + "a" * 64, "a" * 64),
        ),
    ):
        return binding.load_binding(path)


def test_binding_closes_exact_canary_and_qualification(tmp_path: Path) -> None:
    path, value, roots = _fixture(tmp_path)
    loaded = _load(path, roots)
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.artifact_identity == {
        "binding": loaded.sha256,
        "qualification": value["qualification_receipt_sha256"],
        **value["canary_artifact_identity"],
    }


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("vram_cap_gib", 88.0, "contract changed"),
        ("physical_gpu", 1, "contract changed"),
        ("derived_image_digest", "sha256:" + "c" * 64, "artifact bytes changed"),
    ],
)
def test_binding_rejects_contract_or_identity_drift(
    tmp_path: Path, field: str, replacement: object, error: str
) -> None:
    path, value, roots = _fixture(tmp_path)
    value[field] = replacement
    path.unlink()
    _write_private(path, value)
    with pytest.raises(binding.VllmServiceBindingError, match=error):
        _load(path, roots)


def test_binding_rejects_subthreshold_qualification(tmp_path: Path) -> None:
    path, value, roots = _fixture(tmp_path)
    qualification = Path(value["qualification_receipt"])
    receipt = _qualification()
    receipt["performance"]["single_stream_decode_tps"] = 119.99
    qualification.unlink()
    value["qualification_receipt_sha256"] = _write_private(qualification, receipt)
    path.unlink()
    _write_private(path, value)
    with pytest.raises(binding.VllmServiceBindingError, match="below 120"):
        _load(path, roots)
