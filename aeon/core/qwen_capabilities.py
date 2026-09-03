"""Immutable, release-bound placement capabilities for Aeon's Qwen runtime.

This is an authorization registry, not an inventory. A host appearing here as
disabled is deliberately not compute capacity. Live availability still comes
only from the fleet coordinator after one enabled capability is selected.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CAPABILITY_SCHEMA_VERSION = "aeon-qwen-runtime-capabilities-v1"
CAPABILITY_MANIFEST_FILE = (
    Path(__file__).resolve().parent / "data/qwen_runtime_capabilities.json"
)
STANDARD_RELEASE_PROFILE = "qwen38-standard-114688-k3"
STANDARD_CONTEXT_TOKENS = 114688
STANDARD_VRAM_BUDGET_GB = 48.7
STANDARD_MIN_PHYSICAL_VRAM_GB = 90.0
STANDARD_IMAGE_ID = (
    "sha256:d57400972ab0ae46baac64d4bfcc49cb136c07d8b0c50a76c7e2d81bd8a9fe47"
)
STANDARD_MODEL_MANIFEST_SHA256 = (
    "1a3ba1eb88d0507bdef3798a6db59830dc076199b7db7d111201f6997588220e"
)
STANDARD_MODEL_SHA256S_SHA256 = (
    "e7eca7ebee03c4f27482d4fe421ca1fac9f1d9986663a51fd7614361010c1237"
)
LOCAL_DOCKER_CAPABILITY_KEY = "qwen38-standard-177-local-docker"
RTX5000_RELEASE_CANDIDATE_KEY = "qwen38-compact-178-disabled"
RTX5000_178_RELEASE_CAPABILITY_KEY = "qwen38-compact-178-128k"
RTX5000_180_RELEASE_CAPABILITY_KEY = "qwen38-compact-180-128k"
# Transitional name retained for exact receipts created by the first reviewed
# compact lane. New code should use the host-qualified constants or the shared
# compact remote-Docker key set below.
RTX5000_RELEASE_CAPABILITY_KEY = RTX5000_180_RELEASE_CAPABILITY_KEY
COMPACT_REMOTE_DOCKER_CAPABILITY_KEYS = frozenset(
    {
        RTX5000_178_RELEASE_CAPABILITY_KEY,
        RTX5000_180_RELEASE_CAPABILITY_KEY,
    }
)
RTX5000_178_RELEASE_RECEIPT_FILE = (
    Path(__file__).resolve().parent
    / "data/qwen38_rtx5000_178_128k_release_receipt.json"
)
RTX5000_180_RELEASE_RECEIPT_FILE = (
    Path(__file__).resolve().parent
    / "data/qwen38_rtx5000_128k_release_receipt.json"
)
RTX5000_RELEASE_RECEIPT_FILE = RTX5000_180_RELEASE_RECEIPT_FILE
_PACKAGED_REMOTE_RELEASE_RECEIPTS = {
    RTX5000_178_RELEASE_CAPABILITY_KEY: RTX5000_178_RELEASE_RECEIPT_FILE,
    RTX5000_180_RELEASE_CAPABILITY_KEY: RTX5000_180_RELEASE_RECEIPT_FILE,
}
RTX5000_178_CANDIDATE_EVIDENCE_SHA256 = (
    "43ebe495ce0a2ede53a0c9463a79f56ea8f7baa8d281a2c60931ce108f4e75e1"
)
RTX5000_178_MTP_REPORT_SHA256 = (
    "62f98e6a056fd0355dc1ce3d5d35c7bdd8729768c656ce32d91933f8764abc5c"
)
RTX5000_178_LONG_BATCH_REPORT_SHA256 = (
    "f5d84bb3bdabfea9ca50417c6345b7397b3f49accfa7f9c118285826f79ad7c4"
)
RTX5000_178_LONG_BATCH_SCRIPT_SHA256 = (
    "2db7ab3998d9aba14c43fb82d03647ac9c9b8901282f54b01d86cbf0389d38d3"
)
RTX5000_178_NORMAL_AEON_REPORT_SHA256 = (
    "b370182a891c944fa9eac602abcdd602ee6219c768d7d8a097cb0860a29e025e"
)
RTX5000_178_CANDIDATE_MANIFEST_SHA256 = (
    "d36efd8a0b7b6c22bc10803b11bbc48ee61e9cc4893fef04aa230cb0ce223f96"
)
RTX5000_178_PRIVATE_LIFECYCLE_EVIDENCE_SHA256 = (
    "82f873dbf106c903ecfa45b0848e5bdca01fa982f3423d4fe69a522cd0ba3a9f"
)
_RTX5000_178_MISSING_RELEASE_GATES = (
    "exact_teardown_and_release",
)

# Existing exact runtimes may outlive a capability-manifest rollout.  Recovery
# accepts only these explicitly retired manifest/key pairs; new leases and
# launches must always carry the current manifest hash.
_RETIRED_ENABLED_MANIFEST_KEYS = {
    "52e2d54b70c14eefac3d5cae796b1f1ce40ececb95961a42d1c8ec6457254b6a": frozenset(
        {LOCAL_DOCKER_CAPABILITY_KEY, RTX5000_RELEASE_CAPABILITY_KEY}
    ),
    # Exact pre-promotion two-lane manifest retained only for recovery of a
    # runtime that was already launched under it. Normal validation still
    # requires the current manifest hash and an enabled current capability.
    "d36efd8a0b7b6c22bc10803b11bbc48ee61e9cc4893fef04aa230cb0ce223f96": frozenset(
        {LOCAL_DOCKER_CAPABILITY_KEY, RTX5000_180_RELEASE_CAPABILITY_KEY}
    ),
}

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
_HOST_RE = re.compile(r"^192[.]168[.]0[.][0-9]{1,3}$")
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,62}$")
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_COMPLETED_AT_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_CAPABILITY_FIELDS = {
    "allowed_physical_gpus",
    "compute_profile",
    "context_tokens",
    "disabled_reason",
    "enabled",
    "exclusive",
    "host",
    "hostname",
    "image_id",
    "gpu_memory_utilization",
    "key",
    "max_batched_tokens",
    "max_num_seqs",
    "model_manifest_sha256",
    "model_sha256s_sha256",
    "min_physical_vram_gb",
    "release_profile",
    "runtime_adapter",
    "release_receipt_sha256",
    "vram_budget_gb",
}


class QwenCapabilityError(ValueError):
    """The static capability authorization is missing, changed, or disabled."""


@dataclass(frozen=True)
class QwenRuntimeCapability:
    key: str
    enabled: bool
    release_profile: str
    host: str
    hostname: str
    runtime_adapter: str
    allowed_physical_gpus: tuple[int, ...]
    min_physical_vram_gb: float
    vram_budget_gb: float | None
    exclusive: bool
    context_tokens: int
    image_id: str | None
    gpu_memory_utilization: float | None
    max_num_seqs: int | None
    max_batched_tokens: int | None
    model_manifest_sha256: str
    model_sha256s_sha256: str
    release_receipt_sha256: str | None
    compute_profile: str
    disabled_reason: str | None

    @property
    def coordinator_gpu(self) -> int:
        if not self.enabled or len(self.allowed_physical_gpus) != 1:
            raise QwenCapabilityError(
                "the active runtime capability has no single safe GPU selector"
            )
        return self.allowed_physical_gpus[0]


@dataclass(frozen=True)
class QwenRuntimeCapabilityRegistry:
    schema_version: str
    manifest_sha256: str
    capabilities: tuple[QwenRuntimeCapability, ...]

    @property
    def active(self) -> QwenRuntimeCapability:
        """Return the preferred enabled capability for legacy local callers."""

        enabled = self.enabled
        if not enabled:
            raise QwenCapabilityError("no Qwen runtime capability is enabled")
        return enabled[0]

    @property
    def enabled(self) -> tuple[QwenRuntimeCapability, ...]:
        """Enabled releases in deterministic placement-preference order."""

        return tuple(item for item in self.capabilities if item.enabled)


def _capability(
    *,
    key: str,
    enabled: bool,
    release_profile: str,
    host: str,
    hostname: str,
    runtime_adapter: str,
    allowed_physical_gpus: tuple[int, ...],
    min_physical_vram_gb: float,
    vram_budget_gb: float | None,
    exclusive: bool,
    context_tokens: int,
    image_id: str | None,
    gpu_memory_utilization: float | None,
    max_num_seqs: int | None,
    max_batched_tokens: int | None,
    model_manifest_sha256: str,
    model_sha256s_sha256: str,
    release_receipt_sha256: str | None,
    compute_profile: str,
    disabled_reason: str | None,
) -> QwenRuntimeCapability:
    return QwenRuntimeCapability(
        key=key,
        enabled=enabled,
        release_profile=release_profile,
        host=host,
        hostname=hostname,
        runtime_adapter=runtime_adapter,
        allowed_physical_gpus=allowed_physical_gpus,
        min_physical_vram_gb=min_physical_vram_gb,
        vram_budget_gb=vram_budget_gb,
        exclusive=exclusive,
        context_tokens=context_tokens,
        image_id=image_id,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_batched_tokens=max_batched_tokens,
        model_manifest_sha256=model_manifest_sha256,
        model_sha256s_sha256=model_sha256s_sha256,
        release_receipt_sha256=release_receipt_sha256,
        compute_profile=compute_profile,
        disabled_reason=disabled_reason,
    )


_EXPECTED_CAPABILITIES = (
    _capability(
        key=LOCAL_DOCKER_CAPABILITY_KEY,
        enabled=True,
        release_profile=STANDARD_RELEASE_PROFILE,
        host="192.168.0.177",
        hostname="DAY2RTX6000PRO",
        runtime_adapter="local-docker",
        allowed_physical_gpus=(0,),
        min_physical_vram_gb=STANDARD_MIN_PHYSICAL_VRAM_GB,
        vram_budget_gb=STANDARD_VRAM_BUDGET_GB,
        exclusive=True,
        context_tokens=STANDARD_CONTEXT_TOKENS,
        image_id=STANDARD_IMAGE_ID,
        gpu_memory_utilization=0.415,
        max_num_seqs=1,
        max_batched_tokens=32768,
        model_manifest_sha256=STANDARD_MODEL_MANIFEST_SHA256,
        model_sha256s_sha256=STANDARD_MODEL_SHA256S_SHA256,
        release_receipt_sha256=None,
        compute_profile="qwen38-vllm",
        disabled_reason=None,
    ),
    _capability(
        key="qwen38-standard-179-awaiting-runtime",
        enabled=False,
        release_profile=STANDARD_RELEASE_PROFILE,
        host="192.168.0.179",
        hostname="DAY2XRTX6000-2",
        runtime_adapter="remote-bare-unreleased",
        allowed_physical_gpus=(0, 1),
        min_physical_vram_gb=STANDARD_MIN_PHYSICAL_VRAM_GB,
        vram_budget_gb=STANDARD_VRAM_BUDGET_GB,
        exclusive=True,
        context_tokens=STANDARD_CONTEXT_TOKENS,
        image_id=None,
        gpu_memory_utilization=0.415,
        max_num_seqs=1,
        max_batched_tokens=32768,
        model_manifest_sha256=STANDARD_MODEL_MANIFEST_SHA256,
        model_sha256s_sha256=STANDARD_MODEL_SHA256S_SHA256,
        release_receipt_sha256=None,
        compute_profile="qwen38-vllm",
        disabled_reason=(
            "No reviewed aday-owned runtime, transport, or host release receipt "
            "exists on .179."
        ),
    ),
    _capability(
        key=RTX5000_178_RELEASE_CAPABILITY_KEY,
        enabled=True,
        release_profile="qwen38-compact-128k-k3",
        host="192.168.0.178",
        hostname="DAY2XRTX5000",
        runtime_adapter="remote-docker",
        allowed_physical_gpus=(0, 1),
        min_physical_vram_gb=47.0,
        vram_budget_gb=41.25,
        exclusive=True,
        context_tokens=131072,
        image_id=STANDARD_IMAGE_ID,
        gpu_memory_utilization=0.70,
        max_num_seqs=8,
        max_batched_tokens=8192,
        model_manifest_sha256=STANDARD_MODEL_MANIFEST_SHA256,
        model_sha256s_sha256=STANDARD_MODEL_SHA256S_SHA256,
        release_receipt_sha256=(
            "fef559cd0b88506b7b0b29f12cd6c1fdee8b525fa2962358c16048529804f13d"
        ),
        compute_profile="qwen38-vllm",
        disabled_reason=None,
    ),
    _capability(
        key=RTX5000_RELEASE_CAPABILITY_KEY,
        enabled=True,
        release_profile="qwen38-compact-128k-k3",
        host="192.168.0.180",
        hostname="DAY2XRTX5000PRO-2",
        runtime_adapter="remote-docker",
        allowed_physical_gpus=(0, 1),
        min_physical_vram_gb=47.0,
        vram_budget_gb=41.25,
        exclusive=True,
        context_tokens=131072,
        image_id=STANDARD_IMAGE_ID,
        gpu_memory_utilization=0.70,
        max_num_seqs=8,
        max_batched_tokens=8192,
        model_manifest_sha256=STANDARD_MODEL_MANIFEST_SHA256,
        model_sha256s_sha256=STANDARD_MODEL_SHA256S_SHA256,
        release_receipt_sha256=(
            "5eda720a4e168733fd0881cedb144a69f892c707439a8e64304e24ec6a04a91a"
        ),
        compute_profile="qwen38-vllm",
        disabled_reason=None,
    ),
)


def _validate_packaged_remote_release_receipt(
    capability: QwenRuntimeCapability, payload: bytes
) -> None:
    """Validate one host qualification without making it a placement source."""

    try:
        receipt = json.loads(payload)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenCapabilityError("compact Qwen release receipt is malformed") from exc
    if not isinstance(receipt, dict):
        raise QwenCapabilityError("compact Qwen release receipt is malformed")
    runtime = receipt.get("runtime")
    gates = receipt.get("gates")
    raw_reports = receipt.get("raw_reports")
    required_true_gates = (
        "structured_text_and_vision_warmup",
        "long_context_exact_recall",
        "normal_aeon_spill_start",
        "normal_aeon_vision_selftest",
        "exact_teardown_and_release",
    )
    if capability.key == RTX5000_178_RELEASE_CAPABILITY_KEY:
        required_report_hashes = (
            "mtp_k3_sha256",
            "long_batch_sha256",
            "normal_aeon_sha256",
            "lifecycle_state_sha256",
            "exact_teardown_sha256",
            "release_evidence_validator_sha256",
        )
    else:
        required_report_hashes = (
            "mtp_k3_sha256",
            "long_batch_sha256",
            "full_release_receipt_sha256",
        )
    if (
        receipt.get("schema_version") != "aeon-qwen38-rtx5000-release-v1"
        or receipt.get("status") != "passed"
        or not isinstance(receipt.get("completed_at"), str)
        or _RELEASE_COMPLETED_AT_RE.fullmatch(receipt["completed_at"]) is None
        or not isinstance(receipt.get("capability_candidate_manifest_sha256"), str)
        or _SHA256_RE.fullmatch(receipt["capability_candidate_manifest_sha256"])
        is None
        or receipt.get("host") != capability.host
        or receipt.get("hostname") != capability.hostname
        or not isinstance(receipt.get("gpu_model"), str)
        or not receipt["gpu_model"]
        or isinstance(receipt.get("memory_total_mib"), bool)
        or not isinstance(receipt.get("memory_total_mib"), int)
        or receipt["memory_total_mib"]
        < math.ceil(capability.min_physical_vram_gb * 1024)
        or receipt.get("vram_budget_gib") != capability.vram_budget_gb
        or receipt.get("context_tokens") != capability.context_tokens
        or not isinstance(runtime, dict)
        or runtime.get("image_id") != capability.image_id
        or runtime.get("model_manifest_sha256")
        != capability.model_manifest_sha256
        or runtime.get("model_sha256s_sha256")
        != capability.model_sha256s_sha256
        or runtime.get("max_num_seqs") != capability.max_num_seqs
        or runtime.get("max_num_batched_tokens") != capability.max_batched_tokens
        or runtime.get("mtp_k") != 3
        or runtime.get("attention_backend") != "TRITON_ATTN"
        or runtime.get("kv_cache_dtype") != "fp8_per_token_head"
        or not isinstance(gates, dict)
        or any(gates.get(name) is not True for name in required_true_gates)
        or isinstance(gates.get("semantic_mtp_requests_passed"), bool)
        or not isinstance(gates.get("semantic_mtp_requests_passed"), int)
        or gates["semantic_mtp_requests_passed"] < 1
        or isinstance(gates.get("long_prompt_tokens"), bool)
        or not isinstance(gates.get("long_prompt_tokens"), int)
        or gates["long_prompt_tokens"] <= 0
        or not isinstance(raw_reports, dict)
        or any(
            not isinstance(raw_reports.get(name), str)
            or _SHA256_RE.fullmatch(raw_reports[name]) is None
            for name in required_report_hashes
        )
    ):
        raise QwenCapabilityError("compact Qwen release receipt is not qualified")
    if capability.key == RTX5000_178_RELEASE_CAPABILITY_KEY:
        _validate_packaged_178_release_receipt(receipt)


def _validate_packaged_178_release_receipt(receipt: dict[str, Any]) -> None:
    """Bind every sanitized `.178` qualification fact to its exact evidence."""

    if (
        set(receipt)
        != {
            "capability_candidate_manifest_sha256",
            "completed_at",
            "context_tokens",
            "gates",
            "gpu_model",
            "host",
            "hostname",
            "memory_total_mib",
            "raw_reports",
            "runtime",
            "schema_version",
            "status",
            "vram_budget_gib",
        }
        or receipt.get("capability_candidate_manifest_sha256")
        != RTX5000_178_CANDIDATE_MANIFEST_SHA256
        or receipt.get("gpu_model") != "NVIDIA RTX PRO 5000 Blackwell"
        or receipt.get("memory_total_mib") != 48935
        or receipt.get("runtime")
        != {
            "attention_backend": "TRITON_ATTN",
            "image_id": STANDARD_IMAGE_ID,
            "kv_cache_dtype": "fp8_per_token_head",
            "largest_sampled_memory_used_mib": 35376,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 8,
            "model_manifest_sha256": STANDARD_MODEL_MANIFEST_SHA256,
            "model_sha256s_sha256": STANDARD_MODEL_SHA256S_SHA256,
            "mtp_k": 3,
            "sampled_ready_memory_free_mib": 13028,
            "vllm_version": "0.23.0",
            "vram_budget_capacity_mib": 42791,
            "vram_budget_committed_mib": 42240,
        }
        or receipt.get("gates")
        != {
            "batch_8_aggregate_decode_tps": 483.8710217019877,
            "batch_scale_vs_suite_serial": 5.834420636815434,
            "exact_teardown_and_release": True,
            "long_context_exact_recall": True,
            "long_prompt_tokens": 125985,
            "normal_aeon_spill_start": True,
            "normal_aeon_vision_selftest": True,
            "normal_aeon_workspace_pwd": (
                "/home/aday/NexusAgentDashboard/bc_aeon"
            ),
            "ready_state_memory_sample_under_hard_cap": True,
            "routed_semantic_transport": True,
            "semantic_mtp_requests_passed": 15,
            "serial_median_decode_tps": 115.3366489005115,
            "structured_text_and_vision_warmup": True,
        }
        or receipt.get("raw_reports")
        != {
            "exact_teardown_sha256": (
                "af5d320b6db7629f4ca2b505f08ee16a6cde2e903cfb62b53ff9fa1426f53417"
            ),
            "lifecycle_state_sha256": RTX5000_178_PRIVATE_LIFECYCLE_EVIDENCE_SHA256,
            "long_batch_sha256": RTX5000_178_LONG_BATCH_REPORT_SHA256,
            "mtp_k3_sha256": RTX5000_178_MTP_REPORT_SHA256,
            "normal_aeon_sha256": RTX5000_178_NORMAL_AEON_REPORT_SHA256,
            "release_evidence_validator_sha256": (
                "dd8f636f7c8d6f6608f36725e6598de26d778223a5c880c20ad6d3295b38683b"
            ),
        }
    ):
        raise QwenCapabilityError(".178 Qwen release evidence is not exact")


def _verify_packaged_remote_release_receipt(
    capability: QwenRuntimeCapability,
) -> None:
    """Bind an enabled compact lane to its distinct immutable host receipt."""

    receipt_path = _PACKAGED_REMOTE_RELEASE_RECEIPTS.get(capability.key)
    if receipt_path is None or capability.release_receipt_sha256 is None:
        raise QwenCapabilityError(
            "compact Qwen capability has no packaged host release receipt"
        )
    try:
        receipt_payload = receipt_path.read_bytes()
    except OSError as exc:
        raise QwenCapabilityError(
            "compact Qwen host release receipt is unavailable"
        ) from exc
    if (
        not receipt_payload
        or len(receipt_payload) > 64 * 1024
        or hashlib.sha256(receipt_payload).hexdigest()
        != capability.release_receipt_sha256
    ):
        raise QwenCapabilityError(
            "compact Qwen host release receipt identity changed"
        )
    _validate_packaged_remote_release_receipt(capability, receipt_payload)


def _validate_packaged_178_candidate_evidence(payload: bytes) -> None:
    """Validate sanitized benchmark evidence without treating it as a release.

    The bound reports establish semantic, long-context, batching, normal-agent,
    routed-transport, and sampled READY-state facts. They do not attest final
    backend teardown and coordinator release. Keeping a distinct schema and a
    literal ``authorizes_runtime: false`` prevents this otherwise-complete bundle
    from being mistaken for a host qualification receipt.
    """

    try:
        receipt = json.loads(payload)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenCapabilityError(".178 candidate evidence is malformed") from exc
    if not isinstance(receipt, dict) or set(receipt) != {
        "authorizes_runtime",
        "candidate",
        "capability_candidate_manifest_sha256",
        "evidence_created_at",
        "evidence_validator",
        "missing_required_gates",
        "private_evidence_sha256",
        "raw_report_sha256",
        "reported_runtime",
        "schema_version",
        "status",
        "verified_benchmark_gates",
        "verified_normal_aeon_gate",
        "verified_ready_state_sample",
        "verified_runtime_attestation",
    }:
        raise QwenCapabilityError(".178 candidate evidence fields changed")

    candidate = receipt.get("candidate")
    evidence_validator = receipt.get("evidence_validator")
    reports = receipt.get("raw_report_sha256")
    private_evidence = receipt.get("private_evidence_sha256")
    reported_runtime = receipt.get("reported_runtime")
    verified = receipt.get("verified_benchmark_gates")
    normal_aeon = receipt.get("verified_normal_aeon_gate")
    ready_sample = receipt.get("verified_ready_state_sample")
    runtime_attestation = receipt.get("verified_runtime_attestation")
    if (
        receipt.get("schema_version")
        != "aeon-qwen38-rtx5000-candidate-evidence-v1"
        or receipt.get("status") != "incomplete"
        or receipt.get("authorizes_runtime") is not False
        or receipt.get("evidence_created_at")
        != "2026-08-25T21:06:12.349140+00:00"
        or receipt.get("capability_candidate_manifest_sha256")
        != RTX5000_178_CANDIDATE_MANIFEST_SHA256
        or evidence_validator
        != {
            "schema_version": "aeon-qwen38-release-evidence-validation-v1",
            "script_sha256": (
                "dd8f636f7c8d6f6608f36725e6598de26d778223a5c880c20ad6d3295b38683b"
            ),
        }
        or receipt.get("missing_required_gates")
        != list(_RTX5000_178_MISSING_RELEASE_GATES)
        or candidate
        != {
            "host": "192.168.0.178",
            "hostname": "DAY2XRTX5000",
            "key": RTX5000_178_RELEASE_CAPABILITY_KEY,
            "release_profile": "qwen38-compact-128k-k3",
        }
        or reports
        != {
            "long_batch": RTX5000_178_LONG_BATCH_REPORT_SHA256,
            "mtp_k3": RTX5000_178_MTP_REPORT_SHA256,
            "normal_aeon": RTX5000_178_NORMAL_AEON_REPORT_SHA256,
        }
        or private_evidence
        != {
            "lifecycle_state": RTX5000_178_PRIVATE_LIFECYCLE_EVIDENCE_SHA256,
        }
        or reported_runtime
        != {
            "attention_backend": "TRITON_ATTN",
            "image_id": STANDARD_IMAGE_ID,
            "kv_cache_dtype": "fp8_per_token_head",
            "model": "Qwen3.8-27B-ARA-NVFP4-MTP",
            "mtp_k": 3,
            "vllm_version": "0.23.0",
        }
        or not isinstance(verified, dict)
        or set(verified) != {"batch", "long_context", "mtp_k3"}
        or normal_aeon
        != {
            "canonical_workspace_pwd": True,
            "completed_at": "2026-08-25T21:03:24.561060+00:00",
            "gates": {
                "exact_pwd_receipt": True,
                "process_exit_zero": True,
                "session_cleanup_complete": True,
                "single_exact_pwd_action": True,
                "startup_vision_selftest": True,
                "ticket_release_verified": True,
                "truthful_final": True,
            },
            "profile": "aeon-qwen38-compact-178-release-gate",
            "schema_version": "aeon-qwen38-normal-agent-gate-v1",
            "status": "passed",
        }
        or ready_sample
        != {
            "acl": "OPEN",
            "ambiguous_intent_count": 0,
            "claim_count": 1,
            "lease_violation_count": 0,
            "memory_free_mib": 13028,
            "memory_total_mib": 48935,
            "memory_used_mib": 35376,
            "sample_kind": "ready_state_observation",
            "state": "RESERVED_RUNNING",
            "vram_budget_capacity_mib": 42791,
            "vram_budget_committed_mib": 42240,
            "watchdog_active": True,
        }
        or runtime_attestation
        != {
            "exclusive": True,
            "host": "192.168.0.178",
            "hostname": "DAY2XRTX5000",
            "image_id": STANDARD_IMAGE_ID,
            "memory_total_mib": 48935,
            "model_manifest_sha256": STANDARD_MODEL_MANIFEST_SHA256,
            "model_sha256s_sha256": STANDARD_MODEL_SHA256S_SHA256,
            "phase": "releasing",
            "release_gate": True,
            "source_manifest_sha256": (
                "fd0adafe722ee34dfad917883423267be47fee6378964b0c0ce1bc09c539eb1a"
            ),
            "state_schema_version": 7,
            "vram_budget_gib": 41.25,
        }
    ):
        raise QwenCapabilityError(".178 candidate evidence is not exact")

    mtp = verified["mtp_k3"]
    action_hashes = mtp.get("deterministic_action_sha256_by_case", {}) if isinstance(mtp, dict) else {}
    if (
        not isinstance(mtp, dict)
        or set(mtp)
        != {
            "accepted_tokens",
            "accepted_tokens_by_position",
            "benchmark_script_sha256",
            "deterministic_action_sha256_by_case",
            "deterministic_actions",
            "draft_tokens",
            "median_decode_tps",
            "passed",
            "repeats",
            "request_count",
            "schema_valid",
            "semantic_valid",
            "successful_requests",
            "suite_sha256",
            "suite_version",
        }
        or mtp.get("passed") is not True
        or mtp.get("schema_valid") is not True
        or mtp.get("semantic_valid") is not True
        or mtp.get("deterministic_actions") is not True
        or mtp.get("repeats") != 3
        or mtp.get("request_count") != 15
        or mtp.get("successful_requests") != 15
        or mtp.get("request_count") != mtp.get("repeats") * 5
        or mtp.get("benchmark_script_sha256")
        != "a38cba76d5ffe73e9200b748311aaaa2f14593f0758ebf99f9191296672e0a1a"
        or mtp.get("suite_version")
        != "aeon-agent-mtp-suite-v6-long-context-control"
        or mtp.get("suite_sha256")
        != "b4148783023ad5bf95c174c5af2a6b0c2059d52183f33811cfaad91b98e22e5e"
        or not isinstance(mtp.get("median_decode_tps"), (int, float))
        or isinstance(mtp.get("median_decode_tps"), bool)
        or not math.isfinite(float(mtp["median_decode_tps"]))
        or float(mtp["median_decode_tps"]) < 100.0
        or set(action_hashes)
        != {
            "browser_grounding",
            "code_failure_replan",
            "long_context_recall",
            "safe_system_diagnosis",
            "verified_completion",
        }
        or any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in action_hashes.values()
        )
    ):
        raise QwenCapabilityError(".178 MTP candidate evidence is not qualified")

    accepted = mtp.get("accepted_tokens")
    drafted = mtp.get("draft_tokens")
    accepted_by_position = mtp.get("accepted_tokens_by_position")
    if (
        isinstance(accepted, bool)
        or not isinstance(accepted, int)
        or isinstance(drafted, bool)
        or not isinstance(drafted, int)
        or not 0 < accepted <= drafted
        or not isinstance(accepted_by_position, list)
        or len(accepted_by_position) != reported_runtime["mtp_k"]
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in accepted_by_position)
        or sum(accepted_by_position) != accepted
    ):
        raise QwenCapabilityError(".178 native MTP evidence is not qualified")

    long_context = verified["long_context"]
    if (
        not isinstance(long_context, dict)
        or set(long_context)
        != {
            "exact_answer",
            "passed",
            "prompt_tokens_measured",
            "prompt_tokens_reported",
        }
        or long_context.get("passed") is not True
        or long_context.get("exact_answer") is not True
        or isinstance(long_context.get("prompt_tokens_measured"), bool)
        or not isinstance(long_context.get("prompt_tokens_measured"), int)
        or long_context["prompt_tokens_measured"] < 120000
        or long_context.get("prompt_tokens_reported")
        != long_context["prompt_tokens_measured"]
    ):
        raise QwenCapabilityError(".178 long-context candidate evidence is not qualified")

    batch = verified["batch"]
    if not isinstance(batch, dict) or set(batch) != {
        "benchmark_script_sha256",
        "concurrency_8_aggregate_decode_tps",
        "concurrency_8_scale_vs_serial",
        "levels",
        "passed",
        "serial_aggregate_decode_tps",
    }:
        raise QwenCapabilityError(".178 batch candidate evidence is malformed")
    serial = batch.get("serial_aggregate_decode_tps")
    concurrency_8 = batch.get("concurrency_8_aggregate_decode_tps")
    scale = batch.get("concurrency_8_scale_vs_serial")
    if (
        batch.get("passed") is not True
        or batch.get("benchmark_script_sha256")
        != RTX5000_178_LONG_BATCH_SCRIPT_SHA256
        or batch.get("levels") != [1, 2, 4, 8]
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (serial, concurrency_8, scale)
        )
        or not 0 < float(serial) < float(concurrency_8)
        or not math.isclose(
            float(scale),
            float(concurrency_8) / float(serial),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise QwenCapabilityError(".178 batch candidate evidence is not qualified")

    serialized = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    if any(marker in serialized for marker in ("://", "/home/", "fd-", "fr-", "gc-")):
        raise QwenCapabilityError(".178 candidate evidence exposes a transient identifier")


def _verify_packaged_178_candidate_evidence() -> None:
    """Bind release-gate use to the exact non-authorizing evidence package."""

    try:
        payload = RTX5000_178_RELEASE_RECEIPT_FILE.read_bytes()
    except OSError as exc:
        raise QwenCapabilityError(".178 candidate evidence is unavailable") from exc
    if (
        not payload
        or len(payload) > 64 * 1024
        or hashlib.sha256(payload).hexdigest()
        != RTX5000_178_CANDIDATE_EVIDENCE_SHA256
    ):
        raise QwenCapabilityError(".178 candidate evidence identity changed")
    _validate_packaged_178_candidate_evidence(payload)


def _parse_capability(raw: Any) -> QwenRuntimeCapability:
    if not isinstance(raw, dict) or set(raw) != _CAPABILITY_FIELDS:
        raise QwenCapabilityError("Qwen runtime capability fields changed")
    key = raw["key"]
    host = raw["host"]
    hostname = raw["hostname"]
    enabled = raw["enabled"]
    exclusive = raw["exclusive"]
    gpus = raw["allowed_physical_gpus"]
    minimum = raw["min_physical_vram_gb"]
    budget = raw["vram_budget_gb"]
    context = raw["context_tokens"]
    image_id = raw["image_id"]
    utility = raw["gpu_memory_utilization"]
    max_num_seqs = raw["max_num_seqs"]
    max_batched_tokens = raw["max_batched_tokens"]
    model_manifest_sha256 = raw["model_manifest_sha256"]
    model_sha256s_sha256 = raw["model_sha256s_sha256"]
    release_receipt_sha256 = raw["release_receipt_sha256"]
    disabled_reason = raw["disabled_reason"]
    if (
        not isinstance(key, str)
        or _KEY_RE.fullmatch(key) is None
        or not isinstance(host, str)
        or _HOST_RE.fullmatch(host) is None
        or not isinstance(hostname, str)
        or _HOSTNAME_RE.fullmatch(hostname) is None
        or not isinstance(enabled, bool)
        or not isinstance(exclusive, bool)
        or not isinstance(gpus, list)
        or not gpus
        or any(isinstance(value, bool) or not isinstance(value, int) for value in gpus)
        or len(gpus) != len(set(gpus))
        or any(value not in {0, 1} for value in gpus)
        or isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or float(minimum) <= 0
        or (
            budget is not None
            and (
                isinstance(budget, bool)
                or not isinstance(budget, (int, float))
                or not math.isfinite(float(budget))
                or float(budget) <= 0
            )
        )
        or isinstance(context, bool)
        or not isinstance(context, int)
        or context <= 0
        or (image_id is not None and (not isinstance(image_id, str) or _IMAGE_ID_RE.fullmatch(image_id) is None))
        or (
            utility is not None
            and (
                isinstance(utility, bool)
                or not isinstance(utility, (int, float))
                or not math.isfinite(float(utility))
                or not 0 < float(utility) < 1
            )
        )
        or (
            max_num_seqs is not None
            and (isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int) or max_num_seqs <= 0)
        )
        or (
            max_batched_tokens is not None
            and (
                isinstance(max_batched_tokens, bool)
                or not isinstance(max_batched_tokens, int)
                or max_batched_tokens <= 0
            )
        )
        or (
            release_receipt_sha256 is not None
            and (
                not isinstance(release_receipt_sha256, str)
                or _SHA256_RE.fullmatch(release_receipt_sha256) is None
            )
        )
        or not isinstance(model_manifest_sha256, str)
        or _SHA256_RE.fullmatch(model_manifest_sha256) is None
        or not isinstance(model_sha256s_sha256, str)
        or _SHA256_RE.fullmatch(model_sha256s_sha256) is None
        or not isinstance(raw["release_profile"], str)
        or not raw["release_profile"]
        or not isinstance(raw["runtime_adapter"], str)
        or not raw["runtime_adapter"]
        or not isinstance(raw["compute_profile"], str)
        or not raw["compute_profile"]
        or (disabled_reason is not None and not isinstance(disabled_reason, str))
    ):
        raise QwenCapabilityError("Qwen runtime capability is malformed")
    return _capability(
        key=key,
        enabled=enabled,
        release_profile=raw["release_profile"],
        host=host,
        hostname=hostname,
        runtime_adapter=raw["runtime_adapter"],
        allowed_physical_gpus=tuple(gpus),
        min_physical_vram_gb=float(minimum),
        vram_budget_gb=None if budget is None else float(budget),
        exclusive=exclusive,
        context_tokens=context,
        image_id=image_id,
        gpu_memory_utilization=None if utility is None else float(utility),
        max_num_seqs=max_num_seqs,
        max_batched_tokens=max_batched_tokens,
        model_manifest_sha256=model_manifest_sha256,
        model_sha256s_sha256=model_sha256s_sha256,
        release_receipt_sha256=release_receipt_sha256,
        compute_profile=raw["compute_profile"],
        disabled_reason=disabled_reason,
    )


def load_qwen_runtime_capabilities(
    path: Path = CAPABILITY_MANIFEST_FILE,
) -> QwenRuntimeCapabilityRegistry:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise QwenCapabilityError("Qwen runtime capability manifest is unavailable") from exc
    if not payload or len(payload) > 64 * 1024:
        raise QwenCapabilityError("Qwen runtime capability manifest size is invalid")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenCapabilityError("Qwen runtime capability manifest is malformed") from exc
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema_version", "capabilities"}
        or raw.get("schema_version") != CAPABILITY_SCHEMA_VERSION
        or not isinstance(raw.get("capabilities"), list)
    ):
        raise QwenCapabilityError("Qwen runtime capability manifest schema changed")
    capabilities = tuple(_parse_capability(item) for item in raw["capabilities"])
    if capabilities != _EXPECTED_CAPABILITIES:
        raise QwenCapabilityError(
            "Qwen runtime targets changed without a reviewed capability code update"
        )
    for capability in capabilities:
        if capability.enabled and capability.runtime_adapter == "remote-docker":
            _verify_packaged_remote_release_receipt(capability)
    if len({item.key for item in capabilities}) != len(capabilities) or len(
        {item.host for item in capabilities}
    ) != len(capabilities):
        raise QwenCapabilityError("Qwen runtime capabilities are duplicated")
    manifest_sha256 = hashlib.sha256(payload).hexdigest()
    return QwenRuntimeCapabilityRegistry(
        schema_version=CAPABILITY_SCHEMA_VERSION,
        manifest_sha256=manifest_sha256,
        capabilities=capabilities,
    )


def active_qwen_runtime_capability() -> tuple[QwenRuntimeCapability, str]:
    registry = load_qwen_runtime_capabilities()
    return registry.active, registry.manifest_sha256


def enabled_qwen_runtime_capabilities() -> tuple[
    tuple[QwenRuntimeCapability, ...], str
]:
    registry = load_qwen_runtime_capabilities()
    if not registry.enabled:
        raise QwenCapabilityError("no Qwen runtime capability is enabled")
    return registry.enabled, registry.manifest_sha256


def qwen_runtime_capability(
    key: str, *, require_enabled: bool = True
) -> tuple[QwenRuntimeCapability, str]:
    if not isinstance(key, str) or _KEY_RE.fullmatch(key) is None:
        raise QwenCapabilityError("Qwen capability key is malformed")
    registry = load_qwen_runtime_capabilities()
    matches = tuple(item for item in registry.capabilities if item.key == key)
    if len(matches) != 1:
        raise QwenCapabilityError("Qwen capability key is unknown")
    capability = matches[0]
    if require_enabled and not capability.enabled:
        raise QwenCapabilityError(
            capability.disabled_reason or "Qwen target capability is disabled"
        )
    return capability, registry.manifest_sha256


def qwen_release_candidate_capability(
    key: str,
) -> tuple[QwenRuntimeCapability, str]:
    """Return the one exact disabled target authorized only for release gating.

    Normal placement deliberately uses :func:`enabled_qwen_runtime_capabilities`
    and can never see this entry. Keeping this accessor exact prevents a generic
    ``allow_disabled`` switch from turning future placeholder hosts into compute.
    """

    if key != RTX5000_RELEASE_CANDIDATE_KEY:
        raise QwenCapabilityError("Qwen release-candidate key is not authorized")
    capability, manifest_sha256 = qwen_runtime_capability(
        key, require_enabled=False
    )
    if (
        capability.enabled
        or capability.runtime_adapter != "remote-docker"
        or capability.host != "192.168.0.178"
        or capability.release_receipt_sha256 is not None
        or not capability.disabled_reason
    ):
        raise QwenCapabilityError("Qwen release-candidate contract changed")
    _verify_packaged_178_candidate_evidence()
    return capability, manifest_sha256


def require_qwen_release_candidate_target(
    key: str, host: str, physical_gpu: int
) -> tuple[QwenRuntimeCapability, str]:
    capability, manifest_sha256 = qwen_release_candidate_capability(key)
    if host != capability.host or physical_gpu not in capability.allowed_physical_gpus:
        raise QwenCapabilityError("Qwen release-candidate selector is unauthorized")
    return capability, manifest_sha256


def require_enabled_qwen_target(
    host: str, physical_gpu: int
) -> tuple[QwenRuntimeCapability, str]:
    if (
        not isinstance(host, str)
        or isinstance(physical_gpu, bool)
        or not isinstance(physical_gpu, int)
    ):
        raise QwenCapabilityError("Qwen target selector is malformed")
    registry = load_qwen_runtime_capabilities()
    matches = tuple(item for item in registry.capabilities if item.host == host)
    if len(matches) != 1:
        raise QwenCapabilityError("Qwen target has no exact capability entry")
    capability = matches[0]
    if not capability.enabled:
        raise QwenCapabilityError(
            capability.disabled_reason or "Qwen target capability is disabled"
        )
    if physical_gpu not in capability.allowed_physical_gpus:
        raise QwenCapabilityError("Qwen target GPU is outside the capability receipt")
    return capability, registry.manifest_sha256


def validate_qwen_capability_manifest_identity(
    *,
    key: Any,
    manifest_sha256: Any,
    current_manifest_sha256: Any,
    allow_retired_manifest: bool = False,
) -> None:
    """Accept the current manifest, or one exact key-scoped recovery identity."""

    if (
        not isinstance(key, str)
        or not isinstance(manifest_sha256, str)
        or _SHA256_RE.fullmatch(manifest_sha256) is None
        or not isinstance(current_manifest_sha256, str)
        or _SHA256_RE.fullmatch(current_manifest_sha256) is None
        or type(allow_retired_manifest) is not bool
    ):
        raise QwenCapabilityError("Qwen capability manifest identity is malformed")
    if manifest_sha256 == current_manifest_sha256:
        return
    if (
        allow_retired_manifest
        and key in _RETIRED_ENABLED_MANIFEST_KEYS.get(manifest_sha256, ())
    ):
        return
    raise QwenCapabilityError("Qwen capability manifest identity changed")


def validate_qwen_capability_receipt(
    *,
    key: Any,
    manifest_sha256: Any,
    runtime_adapter: Any,
    host: Any,
    physical_gpu: Any,
    release_gate: Any = False,
    allow_retired_manifest: bool = False,
) -> QwenRuntimeCapability:
    if (
        not isinstance(key, str)
        or not isinstance(manifest_sha256, str)
        or _SHA256_RE.fullmatch(manifest_sha256) is None
        or not isinstance(runtime_adapter, str)
        or not isinstance(host, str)
        or isinstance(physical_gpu, bool)
        or not isinstance(physical_gpu, int)
    ):
        raise QwenCapabilityError("Qwen capability receipt is malformed")
    if release_gate is True:
        capability, current_hash = require_qwen_release_candidate_target(
            key, host, physical_gpu
        )
    elif release_gate is False:
        capability, current_hash = require_enabled_qwen_target(host, physical_gpu)
    else:
        raise QwenCapabilityError("Qwen release-gate receipt is malformed")
    validate_qwen_capability_manifest_identity(
        key=key,
        manifest_sha256=manifest_sha256,
        current_manifest_sha256=current_hash,
        allow_retired_manifest=allow_retired_manifest,
    )
    if (
        key != capability.key
        or runtime_adapter != capability.runtime_adapter
    ):
        raise QwenCapabilityError("Qwen capability receipt changed")
    return capability
