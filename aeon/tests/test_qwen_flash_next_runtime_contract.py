from __future__ import annotations

import hashlib
from pathlib import Path

from aeon.core import qwen_flash_next_runtime_contract as contract


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "services/sglang/Dockerfile.qwen38-flash-next-sm120"
QSA_PATCH = (
    ROOT
    / "services/sglang/patches/qwen38-flash-next-sm120-qsa-dac5523.patch"
)
FUSED_SHARED_PATCH = (
    ROOT
    / "services/sglang/patches/qwen38-flash-next-fused-shared-cdb7ac8.patch"
)
MTP_SHARED_PATCH = (
    ROOT
    / "services/sglang/patches/qwen38-flash-next-mtp-shared-day0-cdb7-7db5.patch"
)
CUTLASS_SCALE_HEADROOM_PATCH = (
    ROOT
    / "services/sglang/patches/qwen38-flash-next-cutlass-scale-headroom.patch"
)
MTP_SHARE_BEFORE_POOL_PATCH = (
    ROOT
    / "services/sglang/patches/qwen38-flash-next-mtp-share-before-pool.patch"
)


def test_day0_image_is_not_the_qualified_sm120_image() -> None:
    assert contract.QUALIFIED_IMAGE != contract.PUBLIC_DAY0_IMAGE
    assert contract.QUALIFIED_IMAGE == (
        "aeon/sglang:qwen38-flash-next-sm120-headroom-a6c61-424e"
    )
    assert contract.image_digest_is_settled()
    assert contract.image_config_digest_is_settled()
    assert contract.local_docker_image_id_is_settled()
    assert contract.image_digest_is_settled(
        contract.QUALIFIED_IMAGE_MANIFEST_DIGEST,
        contract.SOURCE_STACK_SHA256,
    )
    assert contract.QUALIFIED_IMAGE_MANIFEST_DIGEST == (
        "sha256:067473b3134f933ebc04a3c4774b16bd400a15afcaf9eec8230c57205f7e7719"
    )
    assert contract.QUALIFIED_IMAGE_CONFIG_DIGEST == (
        "sha256:ac23f9a937f1e82cc1bade15079a568a73e68b1cecbe4d4f326ba330418e0a36"
    )
    assert contract.QUALIFIED_IMAGE_MANIFEST_DIGEST != (
        contract.QUALIFIED_IMAGE_CONFIG_DIGEST
    )
    assert contract.QUALIFIED_LOCAL_DOCKER_IMAGE_ID == (
        contract.QUALIFIED_IMAGE_MANIFEST_DIGEST
    )
    assert contract.QUALIFIED_IMAGE_ARCHIVE_SHA256 == (
        "f25ab76b3f48b55e1632e020e9fc4709766bae447c42564d2058f16a4bc13374"
    )
    assert contract.SM120_FIX_COMMIT != contract.QWEN_REFERENCE_COMMIT
    assert contract.SOURCE_STACK_SHA256 == (
        "f9087c7d56219f49fb575c8b1008e923ddeea1ea878e46b20f8e5585317136ed"
    )


def test_sm120_patch_and_docker_provenance_are_exact() -> None:
    assert hashlib.sha256(QSA_PATCH.read_bytes()).hexdigest() == (
        contract.SM120_PATCH_SHA256
    )
    assert hashlib.sha256(FUSED_SHARED_PATCH.read_bytes()).hexdigest() == (
        contract.FUSED_SHARED_EXPERT_PATCH_SHA256
    )
    assert hashlib.sha256(MTP_SHARED_PATCH.read_bytes()).hexdigest() == (
        contract.MTP_SHARED_EXPERT_PATCH_SHA256
    )
    assert hashlib.sha256(CUTLASS_SCALE_HEADROOM_PATCH.read_bytes()).hexdigest() == (
        contract.CUTLASS_SCALE_HEADROOM_PATCH_SHA256
    )
    assert hashlib.sha256(MTP_SHARE_BEFORE_POOL_PATCH.read_bytes()).hexdigest() == (
        contract.MTP_SHARE_BEFORE_POOL_PATCH_SHA256
    )
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert f"FROM {contract.PUBLIC_DAY0_AMD64_REFERENCE}" in dockerfile
    assert contract.SM120_PATCH_SHA256 in dockerfile
    assert contract.SOURCE_STACK_SHA256 in dockerfile
    assert "--include=python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py" in (
        dockerfile
    )
    for key, value in contract.EXPECTED_IMAGE_LABELS.items():
        assert f'{key}="{value}"' in dockerfile


def test_image_label_contract_fails_closed() -> None:
    labels = dict(contract.EXPECTED_IMAGE_LABELS)
    assert contract.validate_image_labels(labels) == ()
    labels.pop(next(iter(labels)))
    assert contract.validate_image_labels(labels)
    assert contract.validate_image_labels(None) == (
        "Config.Labels is not an object",
    )


def test_wire_alias_is_explicitly_not_the_flash_display_identity() -> None:
    assert "27B" in contract.WIRE_SERVED_ALIAS
    assert "Flash-Next" in contract.DISPLAY_NAME
    assert contract.WIRE_SERVED_ALIAS != contract.DISPLAY_NAME
    assert contract.MODEL_ARCHITECTURE == "qwen4_exp"


def test_moe_backend_memory_selector_contract_is_exact() -> None:
    assert contract.QUALIFICATION_MOE_RUNNER_BACKENDS == ("flashinfer_cutlass",)
    assert contract.PREFERRED_MOE_RUNNER_BACKEND == "flashinfer_cutlass"
    assert contract.CUTLASS_NVFP4_EAGER_SCALE_PLACEHOLDER_BYTES == 7_549_747_200
    assert contract.CUTLASS_NVFP4_SCALE_DUPLICATION_BYTES == 7_549_747_200
    assert contract.CUTLASS_NVFP4_SCALE_DUPLICATION_BYTES / 1024**3 == 7.03125
    assert contract.MTP_BF16_PAYLOAD_BYTES == 5_214_301_696
    assert contract.MTP_STALE_EMBED_HEAD_BYTES == 2_542_796_800
    assert contract.CUTLASS_MIN_CUDA_RESERVE_BYTES == 8 * 1024**3
    assert contract.CUTLASS_MIN_GEOMETRIC_MEAN_SPEEDUP == 1.03
    # FP4 GEMM remains a separate SGLang choice; the avoided allocation was
    # the ModelOpt MoE runner's eager block-scale placeholder.
    assert contract.FP4_GEMM_BACKEND == "flashinfer_cutlass"
