from copy import deepcopy

from aeon.core import qwen_flash_next_vllm_contract as contract


def _valid_runtime():
    return {
        **contract.expected_runtime(),
        "checkpoint_repository": contract.CHECKPOINT_REPOSITORY,
        "checkpoint_revision": contract.CHECKPOINT_REVISION,
        "base_image_amd64_digest": contract.BASE_IMAGE_AMD64_DIGEST,
        "served_model": contract.SERVED_MODEL,
        "host": contract.HOST,
        "physical_gpu": contract.PHYSICAL_GPU,
        "exclusive_lease": True,
        "vram_cap_gib": contract.VRAM_CAP_GIB,
    }


def _valid_qualification():
    return {
        "schema_version": contract.QUALIFICATION_SCHEMA,
        "runtime": _valid_runtime(),
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
            "c2_aggregate_completion_tps": 1.0,
            "single_stream_measured_after_prefill": True,
        },
        "mtp": {
            "draft_tokens": 3_000,
            "accepted_tokens": 2_000,
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
                "response_sha256": ["a" * 64] * 4,
                "response_utf8_bytes": [128] * 4,
            },
            "multimodal_passed": True,
            "tool_call_passed": True,
            "reasoning_parser_passed": True,
            "long_context_passed": True,
            "long_context_prompt_tokens": 120_000,
        },
        "capacity": {
            "max_model_len": 131_072,
            "kv_cache_memory_bytes": 7_623_566_950,
            "startup_max_model_len_validated": True,
        },
        "cuda_sampling": {
            "peak_used_gib": 88.0,
            "minimum_reserve_gib": 6.0,
            "cadence_density": 0.90,
            "maximum_gap_seconds": 2.0,
        },
        "process_identity_verified": True,
        "semantic_readiness_verified": True,
    }


def test_reviewed_runtime_is_tp1_mtp3_and_128k_capable():
    runtime = contract.expected_runtime()
    assert runtime["tensor_parallel_size"] == 1
    assert runtime["distributed_executor_backend"] == "mp"
    assert runtime["max_num_seqs"] == 2
    assert runtime["max_model_len"] == 131_072
    assert runtime["max_num_batched_tokens"] == 2_048
    assert runtime["kv_cache_memory_bytes"] == 7_623_566_950
    assert runtime["cudagraph_capture_sizes"] == [1, 2, 4]
    assert runtime["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "quantization": "modelopt_fp4",
        "moe_backend": "flashinfer_cutlass",
    }
    assert runtime["moe_backend"] == "auto"
    assert runtime["ple_cpu_offload"] is True
    assert runtime["ple_fp8_checkpoint"] is False
    assert runtime["ple_offload_ready_timeout_seconds"] == 1_800


def test_release_identities_are_resolved_and_missing_values_still_fail_closed(
    monkeypatch,
):
    assert contract.unresolved_release_fields() == ()
    assert contract.BASE_IMAGE_AMD64_DIGEST == (
        "sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3"
    )
    assert contract.CHECKPOINT_FILE_COUNT == 112
    monkeypatch.setattr(contract, "BASE_IMAGE_AMD64_DIGEST", None)
    assert "v20 release identities are unresolved" in (
        contract.validate_qualification_receipt(_valid_qualification())
    )


def test_valid_qualification_receipt_passes_exact_floors(monkeypatch):
    monkeypatch.setattr(contract, "BASE_IMAGE_AMD64_DIGEST", "sha256:" + "a" * 64)
    monkeypatch.setattr(contract, "CHECKPOINT_FILE_COUNT", 100)
    receipt = _valid_qualification()
    receipt["runtime"]["base_image_amd64_digest"] = contract.BASE_IMAGE_AMD64_DIGEST
    assert contract.validate_qualification_receipt(receipt) == ()


def test_runtime_receipt_rejects_more_gpu_utilization_than_reviewed():
    receipt = _valid_runtime()
    receipt["gpu_memory_utilization"] = 0.96
    assert "runtime.gpu_memory_utilization: expected 0.88, got 0.96" in (
        contract.validate_runtime_receipt(receipt)
    )


def test_runtime_receipt_rejects_target_or_draft_moe_backend_drift():
    receipt = _valid_runtime()
    receipt["moe_backend"] = "flashinfer_b12x"
    receipt["speculative_config"]["moe_backend"] = "flashinfer_b12x"
    failures = contract.validate_runtime_receipt(receipt)
    assert "runtime.moe_backend: expected 'auto', got 'flashinfer_b12x'" in failures
    assert any(value.startswith("runtime.speculative_config:") for value in failures)


def test_qualification_rejects_transformer_cpu_offload_and_ple_in_vram_claim():
    receipt = _valid_qualification()
    receipt["placement"]["transformer_weights"] = "cpu"
    receipt["placement"]["ple_table"] = "cuda"
    failures = contract.validate_qualification_receipt(receipt)
    assert "transformer weights are not entirely CUDA resident" in failures
    assert "PLE table is not the sole CPU-resident model component" in failures


def test_qualification_rejects_below_floor_or_end_to_end_single_stream_metric():
    receipt = _valid_qualification()
    receipt["performance"]["single_stream_decode_tps"] = 119.999
    receipt["performance"]["c2_aggregate_completion_tps"] = 0.0
    receipt["performance"]["single_stream_measured_after_prefill"] = False
    failures = contract.validate_qualification_receipt(receipt)
    assert "single-stream decode throughput is below 120 tok/s" in failures
    assert "C2 aggregate throughput was not measured" in failures
    assert "single-stream result was not decode-after-prefill" in failures


def test_qualification_requires_real_mtp_activity_and_semantics():
    receipt = _valid_qualification()
    receipt["mtp"]["accepted_tokens"] = 0
    receipt["semantic"]["tool_call_passed"] = False
    failures = contract.validate_qualification_receipt(receipt)
    assert "MTP produced no measured accepted tokens" in failures
    assert "semantic.tool_call_passed did not pass" in failures


def test_uncensored_harmful_prompt_outcomes_are_diagnostic_not_refusal_gate(
    monkeypatch,
):
    monkeypatch.setattr(contract, "BASE_IMAGE_AMD64_DIGEST", "sha256:" + "a" * 64)
    monkeypatch.setattr(contract, "CHECKPOINT_FILE_COUNT", 100)
    receipt = _valid_qualification()
    receipt["runtime"]["base_image_amd64_digest"] = contract.BASE_IMAGE_AMD64_DIGEST
    diagnostic = receipt["semantic"]["harmful_prompt_diagnostic"]
    diagnostic["refusal_or_safe_redirect_count"] = 0
    diagnostic["operational_detail_count"] = 4
    assert contract.validate_qualification_receipt(receipt) == ()


def test_harmful_prompt_diagnostic_must_be_complete_and_bounded(monkeypatch):
    monkeypatch.setattr(contract, "BASE_IMAGE_AMD64_DIGEST", "sha256:" + "a" * 64)
    monkeypatch.setattr(contract, "CHECKPOINT_FILE_COUNT", 100)
    receipt = _valid_qualification()
    receipt["runtime"]["base_image_amd64_digest"] = contract.BASE_IMAGE_AMD64_DIGEST
    receipt["semantic"]["harmful_prompt_diagnostic"]["response_utf8_bytes"][0] = (
        contract.MAX_HARMFUL_DIAGNOSTIC_RESPONSE_BYTES + 1
    )
    assert "harmful-prompt diagnostic evidence is malformed" in (
        contract.validate_qualification_receipt(receipt)
    )


def test_qualification_rejects_cuda_cap_reserve_and_sampling_drift():
    receipt = _valid_qualification()
    receipt["cuda_sampling"].update(
        peak_used_gib=88.901,
        minimum_reserve_gib=5.99,
        cadence_density=0.899,
        maximum_gap_seconds=2.01,
    )
    failures = contract.validate_qualification_receipt(receipt)
    assert "physical CUDA usage exceeded the 88.9 GiB cap" in failures
    assert "physical CUDA reserve fell below 6 GiB" in failures
    assert "physical CUDA sampling density is below 90%" in failures
    assert "physical CUDA sampling gap exceeded 2 seconds" in failures


def test_expected_runtime_returns_deep_enough_copy():
    first = contract.expected_runtime()
    second = deepcopy(first)
    first["speculative_config"]["num_speculative_tokens"] = 2
    first["cudagraph_capture_sizes"].append(8)
    assert contract.expected_runtime() == second
