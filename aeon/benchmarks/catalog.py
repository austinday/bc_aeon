"""Immutable benchmark suites and reviewed harness/model/tool combinations.

Scenario definitions intentionally contain semantic IDs rather than prompts.
Prompt bodies, credentials, Fleet claims, routed endpoints, and visual fixtures
belong to a trusted executor and are never returned by the public catalog.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from aeon.harnesses.catalog import OPENCODE_VERSION, public_harness_catalog

from .protocol import (
    EXECUTOR_PROTOCOL_SHA256,
    EXECUTOR_PROTOCOL_VERSION,
    HARNESS_SOURCE_SHA256,
    RUNNER_SOURCE_SHA256,
    TOOL_SOURCE_SHA256,
)


BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_CATALOG_VERSION = "2026-09-02.7"
RUNNER_PROTOCOL_VERSION = "5"
DEFAULT_SUITE_ID = "smoke"
DEFAULT_MODEL_ID = "local/qwen"
DEFAULT_TOOL_PROFILE_ID = "fleet-local"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ScenarioSpec:
    case_id: str
    label: str
    category: str
    timeout_seconds: int
    required_capabilities: tuple[str, ...] = ()

    def provenance_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "category": self.category,
            "timeout_seconds": self.timeout_seconds,
            "required_capabilities": list(self.required_capabilities),
        }


@dataclass(frozen=True)
class SuiteSpec:
    suite_id: str
    label: str
    description: str
    version: str
    cases: tuple[ScenarioSpec, ...]
    default_tool_profile_id: str = DEFAULT_TOOL_PROFILE_ID

    @property
    def required_capabilities(self) -> tuple[str, ...]:
        """Return the deterministic union required by every case in the suite."""

        return tuple(
            sorted(
                {
                    capability
                    for case in self.cases
                    for capability in case.required_capabilities
                }
            )
        )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "suite_id": self.suite_id,
                "label": self.label,
                "description": self.description,
                "version": self.version,
                "default_tool_profile_id": self.default_tool_profile_id,
                "cases": [case.provenance_record() for case in self.cases],
            }
        )


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    revision: str
    harnesses: tuple[str, ...]
    identity_scope: str
    service_id: str
    selection_semantics: str
    default: bool = False

    def public_record(self) -> dict[str, object]:
        return {
            "id": self.model_id,
            "label": self.label,
            "revision": self.revision,
            "harnesses": list(self.harnesses),
            "identity_scope": self.identity_scope,
            "service_id": self.service_id,
            "selection_semantics": self.selection_semantics,
            "default": self.default,
        }


@dataclass(frozen=True)
class ToolProfileSpec:
    profile_id: str
    label: str
    version: str
    capabilities: tuple[str, ...]
    default: bool = False

    def supports(self, required_capabilities: tuple[str, ...]) -> bool:
        return set(required_capabilities).issubset(self.capabilities)

    def public_record(self) -> dict[str, object]:
        return {
            "id": self.profile_id,
            "label": self.label,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "default": self.default,
        }


@dataclass(frozen=True)
class CombinationSpec:
    combination_id: str
    harness_id: str
    harness_version: str
    model_id: str
    model_revision: str
    tool_profile_id: str
    tool_profile_version: str

    def public_record(self) -> dict[str, object]:
        return {
            "id": self.combination_id,
            "harness_id": self.harness_id,
            "harness_version": self.harness_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tool_profile_id": self.tool_profile_id,
            "tool_profile_version": self.tool_profile_version,
        }


_SMOKE_CASES = (
    ScenarioSpec("smoke.direct", "Direct completion", "smoke", 45),
    ScenarioSpec("smoke.bounded", "Bounded completion", "smoke", 60),
)
_TOOL_CASES = (
    ScenarioSpec(
        "tools.local_read", "Select and read a local artifact", "tools", 90, ("local-tools",)
    ),
    ScenarioSpec(
        "tools.mutate_verify",
        "Perform and verify a sandboxed mutation",
        "tools",
        120,
        ("local-tools",),
    ),
    ScenarioSpec(
        "tools.fleet_wait",
        "Respect durable Fleet wait semantics",
        "tools",
        120,
        ("fleet-tools",),
    ),
)
_BROWSER_CASES = (
    ScenarioSpec(
        "browser.observe", "Observe a controlled dynamic page", "browser", 120, ("browser",)
    ),
    ScenarioSpec(
        "browser.form", "Complete a controlled multi-field form", "browser", 180, ("browser",)
    ),
    ScenarioSpec(
        "browser.session", "Preserve a controlled authenticated session", "browser", 180, ("browser",)
    ),
)
_VISION_CASES = (
    ScenarioSpec(
        "vision.image", "Ground an answer in a fixture image", "vision", 120, ("vision",)
    ),
    ScenarioSpec(
        "vision.browser", "Ground browser action in a screenshot", "vision", 150, ("browser", "vision")
    ),
)
_CONTEXT_CASES = (
    ScenarioSpec("context.recall", "Recall an early bounded fact", "context", 180),
    ScenarioSpec("context.compaction", "Recover after context compaction", "context", 240),
    ScenarioSpec("context.loop", "Avoid repeating an unchanged failed action", "context", 180),
)
_COMPREHENSIVE_CASES = (
    *_SMOKE_CASES,
    *_TOOL_CASES,
    *_BROWSER_CASES,
    *_VISION_CASES,
    *_CONTEXT_CASES,
)

SUITES: Mapping[str, SuiteSpec] = MappingProxyType(
    {
        suite.suite_id: suite
        for suite in (
            SuiteSpec(
                "smoke",
                "Smoke",
                "Fast completion and lifecycle checks.",
                "1",
                _SMOKE_CASES,
            ),
            SuiteSpec(
                "tools",
                "Local and Fleet tools",
                "Selection, validation, mutation, and Fleet-wait behavior.",
                "2",
                _TOOL_CASES,
            ),
            SuiteSpec(
                "browser",
                "Browser workflows",
                "Controlled observation, form, and session workflows.",
                "2",
                _BROWSER_CASES,
            ),
            SuiteSpec(
                "vision",
                "Vision",
                "Image and browser-screenshot grounding.",
                "1",
                _VISION_CASES,
            ),
            SuiteSpec(
                "context",
                "Context durability",
                "Recall, compaction recovery, and loop avoidance.",
                "1",
                _CONTEXT_CASES,
            ),
            SuiteSpec(
                "comprehensive",
                "Comprehensive",
                "All deterministic benchmark scenarios.",
                "2",
                _COMPREHENSIVE_CASES,
            ),
        )
    }
)

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id=DEFAULT_MODEL_ID,
        label="Fleet-routed Qwen service",
        revision="fleet-logical-aeon-qwen38-standard",
        harnesses=("opencode", "legacy-aeon"),
        identity_scope="logical_service",
        service_id="aeon-qwen38-standard",
        selection_semantics="fleet_policy_routed",
        default=True,
    ),
)

TOOL_PROFILES: tuple[ToolProfileSpec, ...] = (
    ToolProfileSpec(
        profile_id=DEFAULT_TOOL_PROFILE_ID,
        label="Fleet-aware local tools",
        version="2",
        capabilities=("local-tools", "fleet-tools", "browser", "vision"),
        default=True,
    ),
)

HARNESS_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "opencode": OPENCODE_VERSION,
        "legacy-aeon": "0.2.0",
    }
)

COMBINATIONS: tuple[CombinationSpec, ...] = tuple(
    CombinationSpec(
        combination_id=f"{harness_id}:local-qwen:fleet-local",
        harness_id=harness_id,
        harness_version=HARNESS_VERSIONS[harness_id],
        model_id=DEFAULT_MODEL_ID,
        model_revision=MODELS[0].revision,
        tool_profile_id=DEFAULT_TOOL_PROFILE_ID,
        tool_profile_version=TOOL_PROFILES[0].version,
    )
    for harness_id in ("opencode", "legacy-aeon")
)


def combination_for(
    harness_id: str,
    model_id: str,
    tool_profile_id: str,
) -> dict[str, object] | None:
    for item in COMBINATIONS:
        if (
            item.harness_id == harness_id
            and item.model_id == model_id
            and item.tool_profile_id == tool_profile_id
        ):
            return item.public_record()
    return None


def combination_sha256(combination: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
            "executor_protocol_version": EXECUTOR_PROTOCOL_VERSION,
            "executor_protocol_sha256": EXECUTOR_PROTOCOL_SHA256,
            "runner_source_sha256": RUNNER_SOURCE_SHA256,
            "harness_source_sha256": HARNESS_SOURCE_SHA256,
            "tool_source_sha256": TOOL_SOURCE_SHA256,
            **dict(combination),
        }
    )


def _catalog_provenance() -> dict[str, object]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "catalog_version": BENCHMARK_CATALOG_VERSION,
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "executor_protocol_version": EXECUTOR_PROTOCOL_VERSION,
        "executor_protocol_sha256": EXECUTOR_PROTOCOL_SHA256,
        "runner_source_sha256": RUNNER_SOURCE_SHA256,
        "harness_source_sha256": HARNESS_SOURCE_SHA256,
        "tool_source_sha256": TOOL_SOURCE_SHA256,
        "suites": [
            {
                "id": suite.suite_id,
                "version": suite.version,
                "sha256": suite.sha256,
            }
            for suite in SUITES.values()
        ],
        "harness_versions": dict(HARNESS_VERSIONS),
        "models": [item.public_record() for item in MODELS],
        "tool_profiles": [item.public_record() for item in TOOL_PROFILES],
        "combinations": [item.public_record() for item in COMBINATIONS],
    }


BENCHMARK_CATALOG_SHA256 = _canonical_sha256(_catalog_provenance())
RUNNER_PROTOCOL_SHA256 = _canonical_sha256(
    {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "executor_protocol_version": EXECUTOR_PROTOCOL_VERSION,
        "executor_protocol_sha256": EXECUTOR_PROTOCOL_SHA256,
        "runner_source_sha256": RUNNER_SOURCE_SHA256,
        "harness_source_sha256": HARNESS_SOURCE_SHA256,
        "tool_source_sha256": TOOL_SOURCE_SHA256,
        "result_fields": [
            "case_id",
            "label",
            "category",
            "repetition",
            "status",
            "score",
            "wall_ms",
            "active_wall_ms",
            "compute_wait_ms",
            "tool_success",
            "browser_success",
            "vision_score",
            "error_code",
        ],
    }
)


def public_catalog() -> dict[str, object]:
    """Return a fresh JSON-safe copy of the immutable reviewed catalog."""

    harnesses = []
    for item in public_harness_catalog():
        record = dict(item)
        record["version"] = HARNESS_VERSIONS[str(record["id"])]
        harnesses.append(record)
    suites = [
        {
            "id": suite.suite_id,
            "label": suite.label,
            "description": suite.description,
            "version": suite.version,
            "sha256": suite.sha256,
            "case_count": len(suite.cases),
            "categories": sorted({case.category for case in suite.cases}),
            "required_capabilities": list(suite.required_capabilities),
            "default_tool_profile_id": suite.default_tool_profile_id,
            "default": suite.suite_id == DEFAULT_SUITE_ID,
        }
        for suite in SUITES.values()
    ]
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "catalog_version": BENCHMARK_CATALOG_VERSION,
        "catalog_sha256": BENCHMARK_CATALOG_SHA256,
        "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
        "runner_protocol_sha256": RUNNER_PROTOCOL_SHA256,
        "executor_protocol_version": EXECUTOR_PROTOCOL_VERSION,
        "executor_protocol_sha256": EXECUTOR_PROTOCOL_SHA256,
        "runner_source_sha256": RUNNER_SOURCE_SHA256,
        "harness_source_sha256": HARNESS_SOURCE_SHA256,
        "tool_source_sha256": TOOL_SOURCE_SHA256,
        "suites": suites,
        "harnesses": harnesses,
        "models": [item.public_record() for item in MODELS],
        "tool_profiles": [item.public_record() for item in TOOL_PROFILES],
        "combinations": [item.public_record() for item in COMBINATIONS],
    }


__all__ = (
    "BENCHMARK_CATALOG_SHA256",
    "BENCHMARK_CATALOG_VERSION",
    "BENCHMARK_SCHEMA_VERSION",
    "COMBINATIONS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_SUITE_ID",
    "DEFAULT_TOOL_PROFILE_ID",
    "EXECUTOR_PROTOCOL_SHA256",
    "EXECUTOR_PROTOCOL_VERSION",
    "HARNESS_VERSIONS",
    "HARNESS_SOURCE_SHA256",
    "MODELS",
    "ModelSpec",
    "RUNNER_PROTOCOL_SHA256",
    "RUNNER_PROTOCOL_VERSION",
    "RUNNER_SOURCE_SHA256",
    "SUITES",
    "TOOL_PROFILES",
    "TOOL_SOURCE_SHA256",
    "ToolProfileSpec",
    "CombinationSpec",
    "ScenarioSpec",
    "SuiteSpec",
    "combination_for",
    "combination_sha256",
    "public_catalog",
)
