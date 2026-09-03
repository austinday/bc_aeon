"""Reviewed per-instance model and effort choices for managed agent tabs.

Browser input never becomes an arbitrary CLI option.  Every persisted value is
normalized against this server-owned catalog before a launcher can consume it.
An empty value means the provider's current default and is intentionally
different from guessing a model that may not be available to the subscription.
"""

from __future__ import annotations

from dataclasses import dataclass

from aeon.core.model_identity import (
    AEON_DEFAULT_MODEL_NAME,
    QWEN38_LEGACY_WIRE_ALIAS,
)
from aeon.harnesses.catalog import public_harness_catalog


class AgentSettingsError(ValueError):
    """Raised when a requested launch preference is not reviewed."""


@dataclass(frozen=True)
class AgentSettingCatalog:
    kind: str
    models: tuple[str, ...]
    efforts: tuple[str, ...]
    default_model: str
    default_effort: str
    model_editable: bool = True
    effort_editable: bool = True


_CATALOGS = {
    "aeon": AgentSettingCatalog(
        kind="aeon",
        models=(AEON_DEFAULT_MODEL_NAME,),
        efforts=("",),
        default_model=AEON_DEFAULT_MODEL_NAME,
        default_effort="",
        model_editable=False,
        effort_editable=False,
    ),
    "codex": AgentSettingCatalog(
        kind="codex",
        models=(
            "",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
        ),
        efforts=("", "minimal", "low", "medium", "high", "xhigh"),
        default_model="",
        default_effort="",
    ),
    "claude": AgentSettingCatalog(
        kind="claude",
        models=("", "sonnet", "opus", "haiku"),
        efforts=("", "low", "medium", "high", "xhigh", "max"),
        default_model="",
        default_effort="",
    ),
    "grok": AgentSettingCatalog(
        kind="grok",
        models=("", "grok-4.5"),
        efforts=("",),
        default_model="",
        default_effort="",
        effort_editable=False,
    ),
}


def catalog_for(kind: str) -> AgentSettingCatalog:
    if not isinstance(kind, str) or kind not in _CATALOGS:
        raise AgentSettingsError("Unsupported agent kind")
    return _CATALOGS[kind]


def normalize_settings(
    kind: str,
    *,
    model: str | None,
    effort: str | None,
) -> tuple[str, str]:
    catalog = catalog_for(kind)
    normalized_model = "" if model is None else str(model).strip()
    normalized_effort = "" if effort is None else str(effort).strip().lower()
    # The Flash-Next rollout changed Aeon's user-facing logical service name,
    # while existing durable tabs still carry the reviewed 27B wire alias.
    # Treat that one exact historical value as the current logical service so
    # restarts can reach Fleet and use either the preferred runtime or its 27B
    # fallback. Arbitrary or older unreviewed values remain rejected below.
    if kind == "aeon" and normalized_model == QWEN38_LEGACY_WIRE_ALIAS:
        normalized_model = AEON_DEFAULT_MODEL_NAME
    if normalized_model not in catalog.models:
        raise AgentSettingsError("That model is not available for this agent")
    if normalized_effort not in catalog.efforts:
        raise AgentSettingsError("That reasoning effort is not available for this agent")
    if not catalog.model_editable and normalized_model != catalog.default_model:
        raise AgentSettingsError("This agent's validated model is fixed")
    if not catalog.effort_editable and normalized_effort != catalog.default_effort:
        raise AgentSettingsError("This agent does not expose a reviewed effort override")
    return normalized_model, normalized_effort


def public_catalog(kind: str) -> dict[str, object]:
    catalog = catalog_for(kind)
    payload = {
        "kind": catalog.kind,
        "models": [
            {"id": value, "label": value or "Provider default"}
            for value in catalog.models
        ],
        "efforts": [
            {"id": value, "label": value or "Provider default"}
            for value in catalog.efforts
        ],
        "default_model": catalog.default_model,
        "default_effort": catalog.default_effort,
        "model_editable": catalog.model_editable,
        "effort_editable": catalog.effort_editable,
    }
    if kind == "aeon":
        payload["harnesses"] = public_harness_catalog()
    return payload


__all__ = (
    "AgentSettingCatalog",
    "AgentSettingsError",
    "catalog_for",
    "normalize_settings",
    "public_catalog",
)
