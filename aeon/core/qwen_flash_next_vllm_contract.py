"""Fail-closed contract for the Qwen3.8 Flash-Next vLLM canary.

This module is deliberately launch-free.  It defines the runtime and evidence
which a Fleet-owned canary worker must prove before the candidate can be made a
service lane.  In particular, a healthy HTTP port is not qualification.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any, Final


PROFILE_ID: Final = "aeon-qwen38-flash-next-vllm-canary"
ADAPTER_NAME: Final = "aeon-qwen38-flash-next-vllm-canary-v1"
HOST: Final = "192.168.0.177"
PHYSICAL_GPU: Final = 0
VRAM_CAP_GIB: Final = 88.9
MIN_PHYSICAL_VRAM_GIB: Final = 94.0
PHYSICAL_RESERVE_GIB: Final = 6.0

CHECKPOINT_REPOSITORY: Final = "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4"
CHECKPOINT_REVISION: Final = "f2c21eb3d2ff5f24c208ea7e3afba65e2e70f83f"
# Reviewed immutable identities for the converted MTP-NVFP4 checkpoint and the
# linux/amd64 child of the pinned upstream vLLM image.  The derived overlay and
# full checkpoint closure remain independently hash-bound by the Fleet profile.
BASE_IMAGE_AMD64_DIGEST: Final[str | None] = (
    "sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3"
)
CHECKPOINT_FILE_COUNT: Final[int | None] = 112
SERVED_MODEL: Final = "Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP"
KV_CACHE_MEMORY_BYTES: Final = 7_623_566_950  # floor(7.1 * 2**30)
CUDA_GRAPH_CAPTURE_SIZES: Final = (1, 2, 4)
MIN_LONG_CONTEXT_PROMPT_TOKENS: Final = 120_000
QUALIFICATION_SCHEMA: Final = "aeon-qwen38-flash-next-vllm-qualification-v2"
HARMFUL_DIAGNOSTIC_SCHEMA: Final = (
    "aeon-qwen38-flash-next-harmful-prompt-diagnostic-v1"
)
HARMFUL_DIAGNOSTIC_PROMPT_COUNT: Final = 4
MAX_HARMFUL_DIAGNOSTIC_RESPONSE_BYTES: Final = 16 * 1024
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")

# v20 fixes the KV allocation instead of relying on utilization-derived memory.
# Fleet's independent 88.9 GiB cap and the sampled 6 GiB reserve remain
# authoritative; successful startup must also validate the full 131072 context.
EXPECTED_RUNTIME: Final[dict[str, Any]] = {
    "provider": "vllm",
    "tensor_parallel_size": 1,
    "distributed_executor_backend": "mp",
    "gpu_memory_utilization": 0.88,
    "kv_cache_memory_bytes": KV_CACHE_MEMORY_BYTES,
    "max_model_len": 131_072,
    "max_num_seqs": 2,
    "max_num_batched_tokens": 2_048,
    "kv_cache_dtype": "auto",
    "quantization": "modelopt_fp4",
    # B12x failed on the target.  Auto resolves to the known-good
    # FLASHINFER_CUTLASS path in the pinned image and is attested at runtime.
    "moe_backend": "auto",
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "enable_flashinfer_autotune": False,
    "cudagraph_capture_sizes": list(CUDA_GRAPH_CAPTURE_SIZES),
    "speculative_config": {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "quantization": "modelopt_fp4",
        "moe_backend": "flashinfer_cutlass",
    },
    "enable_auto_tool_choice": True,
    "tool_call_parser": "qwen3_coder",
    "reasoning_parser": "qwen3",
    "ple_cpu_offload": True,
    "ple_fp8_checkpoint": False,
    "ple_offload_ready_timeout_seconds": 1_800,
    "torch_cuda_arch_list": "12.0f",
    # PyTorch's expandable-segment CUDA IPC path reconstructs shared storage
    # with pidfd_getfd.  The exact vLLM PLE worker topology remains denied by
    # this Docker/kernel boundary even with SYS_PTRACE, while the native CUDA
    # allocator uses the legacy CUDA IPC handle path.  Keep the variable unset.
    "pytorch_alloc_conf": None,
}

MIN_SINGLE_STREAM_DECODE_TPS: Final = 120.0
MIN_CUDA_SAMPLE_DENSITY: Final = 0.90
MAX_CUDA_SAMPLE_GAP_SECONDS: Final = 2.0


def expected_runtime() -> dict[str, Any]:
    """Return an independent copy of the reviewed runtime configuration."""

    result = dict(EXPECTED_RUNTIME)
    result["speculative_config"] = dict(EXPECTED_RUNTIME["speculative_config"])
    result["cudagraph_capture_sizes"] = list(
        EXPECTED_RUNTIME["cudagraph_capture_sizes"]
    )
    return result


def unresolved_release_fields() -> tuple[str, ...]:
    """Return identities which deliberately keep the v20 canary inert."""

    pending: list[str] = []
    if BASE_IMAGE_AMD64_DIGEST is None:
        pending.append("base_image_amd64_digest")
    if CHECKPOINT_FILE_COUNT is None:
        pending.append("checkpoint_file_count")
    return tuple(pending)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def validate_runtime_receipt(receipt: object) -> tuple[str, ...]:
    """Validate launch readback, not merely the requested command line."""

    if not isinstance(receipt, Mapping):
        return ("runtime receipt is not an object",)
    failures: list[str] = []
    if unresolved_release_fields():
        failures.append("v20 release identities are unresolved")
    for key, expected in EXPECTED_RUNTIME.items():
        if receipt.get(key) != expected:
            failures.append(
                f"runtime.{key}: expected {expected!r}, got {receipt.get(key)!r}"
            )
    if receipt.get("checkpoint_repository") != CHECKPOINT_REPOSITORY:
        failures.append("runtime checkpoint repository changed")
    if receipt.get("checkpoint_revision") != CHECKPOINT_REVISION:
        failures.append("runtime checkpoint revision changed")
    if (
        BASE_IMAGE_AMD64_DIGEST is None
        or receipt.get("base_image_amd64_digest") != BASE_IMAGE_AMD64_DIGEST
    ):
        failures.append("runtime image digest changed")
    if receipt.get("served_model") != SERVED_MODEL:
        failures.append("runtime served-model alias changed")
    if receipt.get("host") != HOST or receipt.get("physical_gpu") != PHYSICAL_GPU:
        failures.append("runtime placement is not canonical .177 GPU0")
    if receipt.get("exclusive_lease") is not True:
        failures.append("runtime lease is not exclusive")
    if _finite_number(receipt.get("vram_cap_gib")) != VRAM_CAP_GIB:
        failures.append("runtime VRAM cap is not exactly 88.9 GiB")
    return tuple(failures)


def validate_qualification_receipt(receipt: object) -> tuple[str, ...]:
    """Require identity, placement, semantic, MTP, and hard speed evidence."""

    if not isinstance(receipt, Mapping):
        return ("qualification receipt is not an object",)
    failures: list[str] = []
    if receipt.get("schema_version") != QUALIFICATION_SCHEMA:
        failures.append("qualification receipt schema changed")
    failures.extend(validate_runtime_receipt(receipt.get("runtime")))

    placement = receipt.get("placement")
    if not isinstance(placement, Mapping):
        failures.append("placement receipt is absent")
    else:
        if placement.get("transformer_weights") != "cuda":
            failures.append("transformer weights are not entirely CUDA resident")
        if placement.get("mtp_weights") != "cuda":
            failures.append("MTP weights are not entirely CUDA resident")
        if placement.get("lm_head") != "cuda":
            failures.append("LM head is not entirely CUDA resident")
        if placement.get("vision_weights") != "cuda":
            failures.append("vision weights are not entirely CUDA resident")
        if placement.get("ple_table") != "cpu_worker_pinned_h2d":
            failures.append("PLE table is not the sole CPU-resident model component")
        if placement.get("other_cpu_model_components") != []:
            failures.append("unexpected model components are CPU resident")

    performance = receipt.get("performance")
    if not isinstance(performance, Mapping):
        failures.append("performance receipt is absent")
    else:
        single = _finite_number(performance.get("single_stream_decode_tps"))
        aggregate = _finite_number(performance.get("c2_aggregate_completion_tps"))
        if single is None or single < MIN_SINGLE_STREAM_DECODE_TPS:
            failures.append("single-stream decode throughput is below 120 tok/s")
        if aggregate is None or aggregate <= 0:
            failures.append("C2 aggregate throughput was not measured")
        if performance.get("single_stream_measured_after_prefill") is not True:
            failures.append("single-stream result was not decode-after-prefill")

    mtp = receipt.get("mtp")
    if not isinstance(mtp, Mapping):
        failures.append("MTP receipt is absent")
    else:
        drafted = _finite_number(mtp.get("draft_tokens"))
        accepted = _finite_number(mtp.get("accepted_tokens"))
        if drafted is None or drafted <= 0:
            failures.append("MTP produced no measured draft tokens")
        if accepted is None or accepted <= 0:
            failures.append("MTP produced no measured accepted tokens")
        if mtp.get("causal_equivalence_passed") is not True:
            failures.append("MTP-on/off causal equivalence did not pass")

    semantic = receipt.get("semantic")
    required_semantics = (
        "behavior_low_refusal_passed",
        "behavior_non_harmful_coherence_passed",
        "behavior_boundary_caution_passed",
        "multimodal_passed",
        "tool_call_passed",
        "reasoning_parser_passed",
        "long_context_passed",
    )
    if not isinstance(semantic, Mapping):
        failures.append("semantic receipt is absent")
    else:
        for key in required_semantics:
            if semantic.get(key) is not True:
                failures.append(f"semantic.{key} did not pass")
        long_tokens = _finite_number(semantic.get("long_context_prompt_tokens"))
        if long_tokens is None or long_tokens < MIN_LONG_CONTEXT_PROMPT_TOKENS:
            failures.append("long-context prompt evidence is below 120000 tokens")
        diagnostic = semantic.get("harmful_prompt_diagnostic")
        if not isinstance(diagnostic, Mapping):
            failures.append("harmful-prompt diagnostic evidence is absent")
        else:
            expected_fields = {
                "schema_version",
                "prompt_count",
                "completed_count",
                "refusal_or_safe_redirect_count",
                "operational_detail_count",
                "response_sha256",
                "response_utf8_bytes",
            }
            counts = (
                diagnostic.get("refusal_or_safe_redirect_count"),
                diagnostic.get("operational_detail_count"),
            )
            hashes = diagnostic.get("response_sha256")
            sizes = diagnostic.get("response_utf8_bytes")
            if (
                set(diagnostic) != expected_fields
                or diagnostic.get("schema_version") != HARMFUL_DIAGNOSTIC_SCHEMA
                or diagnostic.get("prompt_count") != HARMFUL_DIAGNOSTIC_PROMPT_COUNT
                or diagnostic.get("completed_count") != HARMFUL_DIAGNOSTIC_PROMPT_COUNT
                or any(type(value) is not int for value in counts)
                or any(
                    value < 0 or value > HARMFUL_DIAGNOSTIC_PROMPT_COUNT
                    for value in counts
                )
                or not isinstance(hashes, list)
                or len(hashes) != HARMFUL_DIAGNOSTIC_PROMPT_COUNT
                or any(
                    not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
                    for value in hashes
                )
                or not isinstance(sizes, list)
                or len(sizes) != HARMFUL_DIAGNOSTIC_PROMPT_COUNT
                or any(
                    type(value) is not int
                    or value <= 0
                    or value > MAX_HARMFUL_DIAGNOSTIC_RESPONSE_BYTES
                    for value in sizes
                )
            ):
                failures.append("harmful-prompt diagnostic evidence is malformed")

    capacity = receipt.get("capacity")
    if not isinstance(capacity, Mapping):
        failures.append("KV capacity receipt is absent")
    elif (
        capacity.get("max_model_len") != EXPECTED_RUNTIME["max_model_len"]
        or capacity.get("kv_cache_memory_bytes") != KV_CACHE_MEMORY_BYTES
        or capacity.get("startup_max_model_len_validated") is not True
    ):
        failures.append("131072-token BF16 KV startup capacity was not proven")

    sampling = receipt.get("cuda_sampling")
    if not isinstance(sampling, Mapping):
        failures.append("physical CUDA sampling receipt is absent")
    else:
        peak = _finite_number(sampling.get("peak_used_gib"))
        reserve = _finite_number(sampling.get("minimum_reserve_gib"))
        density = _finite_number(sampling.get("cadence_density"))
        gap = _finite_number(sampling.get("maximum_gap_seconds"))
        if peak is None or peak > VRAM_CAP_GIB:
            failures.append("physical CUDA usage exceeded the 88.9 GiB cap")
        if reserve is None or reserve < PHYSICAL_RESERVE_GIB:
            failures.append("physical CUDA reserve fell below 6 GiB")
        if density is None or density < MIN_CUDA_SAMPLE_DENSITY:
            failures.append("physical CUDA sampling density is below 90%")
        if gap is None or gap > MAX_CUDA_SAMPLE_GAP_SECONDS:
            failures.append("physical CUDA sampling gap exceeded 2 seconds")

    if receipt.get("process_identity_verified") is not True:
        failures.append("process identity was not verified")
    if receipt.get("semantic_readiness_verified") is not True:
        failures.append("semantic readiness was not verified")
    return tuple(failures)
