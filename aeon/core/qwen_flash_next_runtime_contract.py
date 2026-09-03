"""Immutable SGLang/SM120 runtime contract for Qwen3.8-Flash-Next.

The public day-zero image is useful source material, but is not an eligible
runtime on RTX PRO 6000 (SM120) by itself.  The production image must carry the
exact labels below after applying the narrow upstream SM120 QSA, Qwen3.8
shared-expert/MTP loader, and memory-headroom fixes.  The qualified addresses
below bind the independently inspected OCI build to that exact source stack.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final, Mapping


# Kept for API compatibility during the rolling replacement.  This is not the
# artifact or display identity of the Flash-Next checkpoint.
WIRE_SERVED_ALIAS: Final = "Qwen3.8-27B-ARA-NVFP4-MTP"
DISPLAY_NAME: Final = "Aeon Qwen3.8-Flash-Next 125B-A6B NVFP4+MTP"
ARTIFACT_NAME: Final = "Aeon-Qwen3.8-Flash-Next-NVFP4-MTP"
MODEL_FAMILY: Final = "qwen3.8-flash-next"
MODEL_ARCHITECTURE: Final = "qwen4_exp"

PUBLIC_DAY0_IMAGE: Final = "lmsysorg/sglang:qwen38flashnext"
PUBLIC_DAY0_INDEX_DIGEST: Final = (
    "sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1"
)
PUBLIC_DAY0_IMAGE_REFERENCE: Final = (
    f"{PUBLIC_DAY0_IMAGE}@{PUBLIC_DAY0_INDEX_DIGEST}"
)
PUBLIC_DAY0_AMD64_DIGEST: Final = (
    "sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae"
)
PUBLIC_DAY0_AMD64_REFERENCE: Final = (
    f"{PUBLIC_DAY0_IMAGE}@{PUBLIC_DAY0_AMD64_DIGEST}"
)
BASE_SGLANG_COMMIT: Final = "d91c3682b0b429e4c70df63cd57f819588ce29b0"
BASE_SOURCE_IMAGE: Final = (
    "lmsysorg/sglang:nightly-dev-cu13-20260817-d91c3682@"
    "sha256:fa8774dd128600a09fd6d46670b06fb69a55dac8a3881e50ccf0916a45eb39af"
)
QWEN_OVERLAY_PR: Final = "Qiaolin-Yu/sglang-qwen-next#38"
QWEN_OVERLAY_COMMITS: Final = ("3ea3a37a1", "12070370f")

# PR 36497 is the public, auditable qwen4_exp reference implementation.  PR
# 36556 is based directly on it and supplies the required SM120 QSA correction.
QWEN_REFERENCE_PR: Final = 36497
QWEN_REFERENCE_COMMIT: Final = "73a255206f916366c8d26d4022f82ddfb0ab558d"
SM120_FIX_PR: Final = 36556
SM120_FIX_COMMIT: Final = "dac5523d1e5d2f4297fec40ef02fc76fb0f662d1"
SM120_PATCH_SHA256: Final = (
    "eba9b1b2c07f6bdfe42502ffc50667f7e1e5467dc1ee96f0a8e791562e1c9679"
)
SM120_FP4_BACKEND_SELECTION_COMMIT: Final = (
    "3836cba9eed2cc0db093e58ca839215609a44c31"
)
FUSED_SHARED_EXPERT_COMMIT: Final = (
    "cdb7ac8f4740f0baf5d01d673fd0fb671a14ebdf"
)
FUSED_SHARED_EXPERT_PATCH_SHA256: Final = (
    "9c3d91412bd3599ccfb5a8879448423fbc34cc24659593933dabe22858ce7338"
)
MTP_SHARED_EXPERT_COMMIT: Final = (
    "7db597910dab20741770862d328c1399be0e6ab8"
)
MTP_SHARED_EXPERT_PATCH_SHA256: Final = (
    "e9f26827b1c0da319c1116caea575b89a794c983ed35671331d421d40137b7fb"
)
CUTLASS_SCALE_HEADROOM_PATCH_SHA256: Final = (
    "a6c61ef9eaa1153551506b26aca7627f7ecc98851f6cd7e7038cd6d0a25b5c6a"
)
MTP_SHARE_BEFORE_POOL_ISSUE: Final = 36452
MTP_SHARE_BEFORE_POOL_PATCH_SHA256: Final = (
    "424eb761834646089437f7e2d16694ab06f03e102f045da07f4a35aa3c83b607"
)

SOURCE_STACK: Final[Mapping[str, object]] = {
    "base_image_index": PUBLIC_DAY0_IMAGE_REFERENCE,
    "base_image_linux_amd64": PUBLIC_DAY0_AMD64_REFERENCE,
    "base_sglang_commit": BASE_SGLANG_COMMIT,
    "qwen_overlay_pr": QWEN_OVERLAY_PR,
    "qwen_overlay_commits": list(QWEN_OVERLAY_COMMITS),
    "qwen_reference_commit": QWEN_REFERENCE_COMMIT,
    "sm120_fix_commit": SM120_FIX_COMMIT,
    "sm120_patch_sha256": SM120_PATCH_SHA256,
    "sm120_fp4_backend_selection_commit": SM120_FP4_BACKEND_SELECTION_COMMIT,
    "fused_shared_expert_commit": FUSED_SHARED_EXPERT_COMMIT,
    "fused_shared_expert_patch_sha256": FUSED_SHARED_EXPERT_PATCH_SHA256,
    "mtp_shared_expert_commit": MTP_SHARED_EXPERT_COMMIT,
    "mtp_shared_expert_patch_sha256": MTP_SHARED_EXPERT_PATCH_SHA256,
    "cutlass_scale_headroom_patch_sha256": CUTLASS_SCALE_HEADROOM_PATCH_SHA256,
    "mtp_share_before_pool_issue": MTP_SHARE_BEFORE_POOL_ISSUE,
    "mtp_share_before_pool_patch_sha256": MTP_SHARE_BEFORE_POOL_PATCH_SHA256,
}
SOURCE_STACK_SHA256: Final = hashlib.sha256(
    json.dumps(
        SOURCE_STACK,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
assert SOURCE_STACK_SHA256 == (
    "f9087c7d56219f49fb575c8b1008e923ddeea1ea878e46b20f8e5585317136ed"
)

# The repository is deliberately owned by Aeon: the patched image must never be
# represented as the unmodified lmsysorg artifact.  These four addresses were
# read independently from the reproducible amd64 OCI build and Docker 29.2's
# containerd-backed image store.  On this daemon ``docker image inspect .Id`` is
# the OCI manifest digest; the raw OCI config digest is provenance evidence but
# is not a launchable Docker reference.
QUALIFIED_IMAGE: Final = (
    "aeon/sglang:qwen38-flash-next-sm120-headroom-a6c61-424e"
)
UNSET_DIGEST: Final = "sha256:" + ("0" * 64)
QUALIFIED_IMAGE_MANIFEST_DIGEST: Final = (
    "sha256:067473b3134f933ebc04a3c4774b16bd400a15afcaf9eec8230c57205f7e7719"
)
QUALIFIED_IMAGE_CONFIG_DIGEST: Final = (
    "sha256:ac23f9a937f1e82cc1bade15079a568a73e68b1cecbe4d4f326ba330418e0a36"
)
QUALIFIED_LOCAL_DOCKER_IMAGE_ID: Final = QUALIFIED_IMAGE_MANIFEST_DIGEST
QUALIFIED_IMAGE_ARCHIVE_SHA256: Final = (
    "f25ab76b3f48b55e1632e020e9fc4709766bae447c42564d2058f16a4bc13374"
)
QUALIFIED_IMAGE_SOURCE_STACK_SHA256: Final = (
    "f9087c7d56219f49fb575c8b1008e923ddeea1ea878e46b20f8e5585317136ed"
)
# Compatibility spelling for callers that already understand this as the OCI
# manifest digest.  New code should use the explicit name above.
QUALIFIED_IMAGE_DIGEST: Final = QUALIFIED_IMAGE_MANIFEST_DIGEST
QUALIFIED_IMAGE_REFERENCE: Final = (
    f"{QUALIFIED_IMAGE}@{QUALIFIED_IMAGE_MANIFEST_DIGEST}"
)
QUALIFIED_IMAGE_REPO_DIGEST: Final = (
    f"{QUALIFIED_IMAGE.split(':', 1)[0]}@{QUALIFIED_IMAGE_MANIFEST_DIGEST}"
)

IMAGE_LABEL_PREFIX: Final = "com.bc-aeon.qwen38-flash-next"
EXPECTED_IMAGE_LABELS: Final[Mapping[str, str]] = {
    f"{IMAGE_LABEL_PREFIX}.artifact": ARTIFACT_NAME,
    f"{IMAGE_LABEL_PREFIX}.display-name": DISPLAY_NAME,
    f"{IMAGE_LABEL_PREFIX}.model-architecture": MODEL_ARCHITECTURE,
    f"{IMAGE_LABEL_PREFIX}.base-image": PUBLIC_DAY0_AMD64_REFERENCE,
    f"{IMAGE_LABEL_PREFIX}.base-sglang-commit": BASE_SGLANG_COMMIT,
    f"{IMAGE_LABEL_PREFIX}.qwen-overlay-pr": QWEN_OVERLAY_PR,
    f"{IMAGE_LABEL_PREFIX}.qwen-overlay-commits": ",".join(
        QWEN_OVERLAY_COMMITS
    ),
    f"{IMAGE_LABEL_PREFIX}.qwen-reference-commit": QWEN_REFERENCE_COMMIT,
    f"{IMAGE_LABEL_PREFIX}.sm120-fix-commit": SM120_FIX_COMMIT,
    f"{IMAGE_LABEL_PREFIX}.sm120-patch-sha256": SM120_PATCH_SHA256,
    f"{IMAGE_LABEL_PREFIX}.sm120-fp4-backend-selection-commit": (
        SM120_FP4_BACKEND_SELECTION_COMMIT
    ),
    f"{IMAGE_LABEL_PREFIX}.fused-shared-expert-commit": (
        FUSED_SHARED_EXPERT_COMMIT
    ),
    f"{IMAGE_LABEL_PREFIX}.fused-shared-expert-patch-sha256": (
        FUSED_SHARED_EXPERT_PATCH_SHA256
    ),
    f"{IMAGE_LABEL_PREFIX}.mtp-shared-expert-commit": MTP_SHARED_EXPERT_COMMIT,
    f"{IMAGE_LABEL_PREFIX}.mtp-shared-expert-patch-sha256": (
        MTP_SHARED_EXPERT_PATCH_SHA256
    ),
    f"{IMAGE_LABEL_PREFIX}.cutlass-scale-headroom-patch-sha256": (
        CUTLASS_SCALE_HEADROOM_PATCH_SHA256
    ),
    f"{IMAGE_LABEL_PREFIX}.mtp-share-before-pool-issue": str(
        MTP_SHARE_BEFORE_POOL_ISSUE
    ),
    f"{IMAGE_LABEL_PREFIX}.mtp-share-before-pool-patch-sha256": (
        MTP_SHARE_BEFORE_POOL_PATCH_SHA256
    ),
    f"{IMAGE_LABEL_PREFIX}.source-stack-sha256": SOURCE_STACK_SHA256,
}

QUANTIZATION: Final = "modelopt_fp4"
MTP_DRAFT_QUANTIZATION: Final = "unquant"
REASONING_PARSER: Final = "qwen3"
PREFILL_ATTENTION_BACKEND: Final = "triton"
DECODE_ATTENTION_BACKEND: Final = "trtllm_mha"
# Before the headroom patch, ModelOpt's CUTLASS create_weights path eagerly
# materialized an uninitialized swizzled placeholder for every NVFP4 MoE scale.
# Post-load alias_or_bind_derived_param already overwrote the source scale in
# place and aliased the derived name, so the placeholders were not live model
# parameters; their released 7,549,747,200 bytes nevertheless remained in
# fragmented CUDA allocator segments.  The patch defers each swizzle until its
# loaded scale is processed and keeps the existing in-place alias contract.
CUTLASS_MOE_RUNNER_BACKEND: Final = "flashinfer_cutlass"
PREFERRED_MOE_RUNNER_BACKEND: Final = CUTLASS_MOE_RUNNER_BACKEND
QUALIFICATION_MOE_RUNNER_BACKENDS: Final = (CUTLASS_MOE_RUNNER_BACKEND,)
CUTLASS_NVFP4_EAGER_SCALE_PLACEHOLDER_BYTES: Final = 7_549_747_200
# Wire-compatible spelling retained until the qualification schema is revised.
CUTLASS_NVFP4_SCALE_DUPLICATION_BYTES: Final = (
    CUTLASS_NVFP4_EAGER_SCALE_PLACEHOLDER_BYTES
)
MTP_BF16_PAYLOAD_BYTES: Final = 5_214_301_696
MTP_STALE_EMBED_HEAD_BYTES: Final = 2_542_796_800
CUTLASS_MIN_CUDA_RESERVE_BYTES: Final = 8 * 1024**3
CUTLASS_MIN_GEOMETRIC_MEAN_SPEEDUP: Final = 1.03
FP4_GEMM_BACKEND: Final = "flashinfer_cutlass"
LINEAR_ATTN_PREFILL_BACKEND: Final = "triton"
LINEAR_ATTN_DECODE_BACKEND: Final = "flashinfer"
MAMBA_SSM_DTYPE: Final = "bfloat16"
CPU_OFFLOAD_GB: Final = "0"
SM120_VALIDATED_CONTEXT_LENGTH: Final = 65_536


def image_digest_is_settled(
    digest: str = QUALIFIED_IMAGE_DIGEST,
    source_stack_sha256: str = QUALIFIED_IMAGE_SOURCE_STACK_SHA256,
) -> bool:
    """Return whether a non-placeholder derived-image digest is configured."""

    return (
        digest.startswith("sha256:")
        and digest != UNSET_DIGEST
        and source_stack_sha256 == SOURCE_STACK_SHA256
    )


def local_docker_image_id_is_settled(
    image_id: str = QUALIFIED_LOCAL_DOCKER_IMAGE_ID,
    source_stack_sha256: str = QUALIFIED_IMAGE_SOURCE_STACK_SHA256,
) -> bool:
    """Return whether a non-placeholder daemon launch ID is configured."""

    return (
        image_id.startswith("sha256:")
        and image_id != UNSET_DIGEST
        and source_stack_sha256 == SOURCE_STACK_SHA256
    )


def image_config_digest_is_settled(
    digest: str = QUALIFIED_IMAGE_CONFIG_DIGEST,
    source_stack_sha256: str = QUALIFIED_IMAGE_SOURCE_STACK_SHA256,
) -> bool:
    """Return whether independent raw OCI config provenance is configured."""

    return (
        digest.startswith("sha256:")
        and digest != UNSET_DIGEST
        and source_stack_sha256 == SOURCE_STACK_SHA256
    )


def validate_image_labels(labels: object) -> tuple[str, ...]:
    """Return deterministic mismatches for a Docker Config.Labels object."""

    if not isinstance(labels, Mapping):
        return ("Config.Labels is not an object",)
    mismatches: list[str] = []
    for key, expected in sorted(EXPECTED_IMAGE_LABELS.items()):
        actual = labels.get(key)
        if actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
    return tuple(mismatches)
