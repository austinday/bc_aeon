"""Logical default and concrete runtime identity regressions."""

from aeon.core.model_identity import (
    AEON_DEFAULT_MODEL_NAME,
    QWEN38_LEGACY_WIRE_ALIAS,
    runtime_pool_summary,
    wire_model_for_runtime_profiles,
)
from aeon.core.qwen_flash_next_vllm_contract import SERVED_MODEL
def test_flash_is_default_while_legacy_wire_alias_stays_distinct() -> None:
    assert AEON_DEFAULT_MODEL_NAME == (
        "Aeon Qwen3.8-Flash-Next 125B-A6B NVFP4+MTP"
    )
    assert QWEN38_LEGACY_WIRE_ALIAS == "Qwen3.8-27B-ARA-NVFP4-MTP"
    assert AEON_DEFAULT_MODEL_NAME != QWEN38_LEGACY_WIRE_ALIAS


def test_runtime_pool_summary_never_mislabels_fallback_as_flash() -> None:
    assert runtime_pool_summary(["aeon-qwen38-flash-next-177"]) == (
        "Ready pool: Qwen3.8-Flash-Next NVFP4+MTP via "
        "aeon-qwen38-flash-next-177"
    )
    assert runtime_pool_summary(["aeon-qwen38-compact-workers"]) == (
        "Ready pool: Qwen3.8-27B RTX 5000 fallback via "
        "aeon-qwen38-compact-workers"
    )
    assert runtime_pool_summary(["aeon-qwen38-standard"]) == (
        "Ready pool: Qwen3.8-27B retained local runtime via "
        "aeon-qwen38-standard"
    )


def test_runtime_pool_summary_exposes_mixed_router_pool() -> None:
    summary = runtime_pool_summary(
        [
            "aeon-qwen38-compact-workers",
            "aeon-qwen38-flash-next-177",
        ]
    )
    assert "Qwen3.8-Flash-Next NVFP4+MTP" in summary
    assert "Qwen3.8-27B RTX 5000 fallback" in summary


def test_wire_model_supports_flash_bridge_and_compatible_pool_shapes() -> None:
    assert wire_model_for_runtime_profiles(
        ["aeon-qwen38-flash-next-vllm-177"]
    ) == SERVED_MODEL
    assert wire_model_for_runtime_profiles(
        ["aeon-qwen38-compact-workers"]
    ) == QWEN38_LEGACY_WIRE_ALIAS
    assert wire_model_for_runtime_profiles(
        ["aeon-qwen38-flash-next-vllm-177", "aeon-qwen38-compact-workers"]
    ) == QWEN38_LEGACY_WIRE_ALIAS


def test_wire_model_rejects_empty_or_unknown_runtime_identity() -> None:
    for profiles in (
        [],
        ["aeon-qwen38-unknown"],
    ):
        try:
            wire_model_for_runtime_profiles(profiles)
        except ValueError:
            continue
        raise AssertionError(f"accepted ambiguous runtime profiles: {profiles}")
