"""Stable logical-model and concrete Fleet-runtime display identities.

Aeon's user-facing model choice names the preferred logical service, not the
OpenAI-compatible ``model`` token used by an already-running server.  The latter
must remain stable while Fleet rolls from the 27B fallback to Flash-Next.  Keep
that compatibility alias out of UI/status text and derive concrete status from
the broker-proven runtime profile set instead.
"""

from __future__ import annotations

from collections.abc import Iterable

from aeon.core.qwen_flash_next_runtime_contract import DISPLAY_NAME
from aeon.core.qwen_flash_next_vllm_contract import SERVED_MODEL as FLASH_NEXT_SERVED_MODEL
AEON_DEFAULT_MODEL_NAME = DISPLAY_NAME
QWEN38_LEGACY_WIRE_ALIAS = "Qwen3.8-27B-ARA-NVFP4-MTP"
QWEN38_FALLBACK_MODEL_NAME = "Qwen3.8-27B-ARA-NVFP4-MTP"

FLASH_NEXT_RUNTIME_PROFILES = frozenset(
    {
        "aeon-qwen38-flash-next-177",
        "aeon-qwen38-flash-next-179",
        "aeon-qwen38-flash-next-vllm-177",
    }
)
QWEN38_FALLBACK_RUNTIME_PROFILES = frozenset(
    {
        "aeon-qwen38-compact-workers",
        "aeon-qwen38-standard",
    }
)


def runtime_pool_summary(profile_ids: Iterable[str]) -> str:
    """Describe the exact routed model mix without exposing lease identities."""

    profiles = tuple(sorted(set(profile_ids)))
    if not profiles:
        return "Qwen runtime ready; concrete profile unavailable from broker"
    flash = tuple(item for item in profiles if item in FLASH_NEXT_RUNTIME_PROFILES)
    compact_fallback = tuple(
        item for item in profiles if item == "aeon-qwen38-compact-workers"
    )
    retained_local = tuple(
        item for item in profiles if item == "aeon-qwen38-standard"
    )
    unknown = tuple(
        item
        for item in profiles
        if item not in set(flash) | set(compact_fallback) | set(retained_local)
    )
    parts: list[str] = []
    if flash:
        parts.append(
            "Qwen3.8-Flash-Next NVFP4+MTP via " + ", ".join(flash)
        )
    if compact_fallback:
        parts.append(
            "Qwen3.8-27B RTX 5000 fallback via "
            + ", ".join(compact_fallback)
        )
    if retained_local:
        parts.append(
            "Qwen3.8-27B retained local runtime via "
            + ", ".join(retained_local)
        )
    if unknown:
        parts.append("other reviewed Qwen runtime via " + ", ".join(unknown))
    return "Ready pool: " + "; ".join(parts)


def wire_model_for_runtime_profiles(profile_ids: Iterable[str]) -> str:
    """Return the shared OpenAI token for the reviewed routed pool.

    Fleet's least-busy endpoint can route one session across Flash and compact
    lanes. Concrete profiles prove the displayed artifact identity, but every
    compatible lane deliberately accepts the same legacy wire token.
    """

    profiles = frozenset(profile_ids)
    flash = profiles & FLASH_NEXT_RUNTIME_PROFILES
    fallback = profiles & QWEN38_FALLBACK_RUNTIME_PROFILES
    unknown = profiles - FLASH_NEXT_RUNTIME_PROFILES - QWEN38_FALLBACK_RUNTIME_PROFILES
    if unknown or not profiles:
        raise ValueError("Fleet runtime profiles do not identify the reviewed wire contract")
    # A Flash-only bridge from the immediately preceding deployment accepts its
    # artifact-specific name. Current Flash production accepts both names, while
    # mixed and fallback-only pools use the compatibility token shared with 27B.
    return (
        QWEN38_LEGACY_WIRE_ALIAS
        if fallback
        else FLASH_NEXT_SERVED_MODEL
    )


__all__ = (
    "AEON_DEFAULT_MODEL_NAME",
    "FLASH_NEXT_RUNTIME_PROFILES",
    "QWEN38_FALLBACK_MODEL_NAME",
    "QWEN38_FALLBACK_RUNTIME_PROFILES",
    "QWEN38_LEGACY_WIRE_ALIAS",
    "runtime_pool_summary",
    "wire_model_for_runtime_profiles",
)
