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
from typing import Iterable, Mapping

from aeon.harnesses.catalog import OPENCODE_VERSION, public_harness_catalog

from .protocol import (
    EXECUTOR_PROTOCOL_SHA256,
    EXECUTOR_PROTOCOL_VERSION,
    HARNESS_SOURCE_SHA256,
    RUNNER_SOURCE_SHA256,
    TOOL_SOURCE_SHA256,
)


BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_CATALOG_VERSION = "2026-09-03.5"
RUNNER_PROTOCOL_VERSION = "9"
# ``suite_id`` remains in durable rows and requests so existing records and API
# clients keep a stable shape.  There is now exactly one benchmark a user can
# start, however, and callers may omit the field entirely.
DEFAULT_SUITE_ID = "comprehensive"
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
    component_id: str = "instruction_following"
    case_weight: float = 1.0
    target_active_seconds: int = 30

    def provenance_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "category": self.category,
            "timeout_seconds": self.timeout_seconds,
            "required_capabilities": list(self.required_capabilities),
            "component_id": self.component_id,
            "case_weight": self.case_weight,
            "target_active_seconds": self.target_active_seconds,
        }


@dataclass(frozen=True)
class SuiteSpec:
    suite_id: str
    label: str
    description: str
    version: str
    cases: tuple[ScenarioSpec, ...]
    default_tool_profile_id: str = DEFAULT_TOOL_PROFILE_ID
    accepts_new_submissions: bool = False

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
                "accepts_new_submissions": self.accepts_new_submissions,
                "cases": [case.provenance_record() for case in self.cases],
            }
        )


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    label: str
    description: str
    weight: float

    def public_record(self) -> dict[str, object]:
        return {
            "id": self.component_id,
            "label": self.label,
            "description": self.description,
            "weight": self.weight,
        }


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


COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        "instruction_following",
        "Instruction following",
        "Follows exact output and constraint instructions without unnecessary actions.",
        0.20,
    ),
    ComponentSpec(
        "memory_context",
        "Memory and context",
        "Retains, updates, and applies information across bounded turns.",
        0.20,
    ),
    ComponentSpec(
        "tool_judgment",
        "Tool-call judgment",
        "Chooses the right reviewed tool, supplies task-relevant inputs, and avoids redundant calls.",
        0.20,
    ),
    ComponentSpec(
        "web_vision",
        "Web and visual reasoning",
        "Selects browser or vision capabilities when needed and grounds actions in observations.",
        0.10,
    ),
    ComponentSpec(
        "fleet_resilience",
        "Fleet resilience",
        "Uses durable demand, useful wait work, checkpoints, and same-job recovery without polling or warm reservation.",
        0.10,
    ),
    ComponentSpec(
        "parallel_execution",
        "Parallel orchestration",
        "Schedules independent and dependent work with useful principal overlap and verifies integrated results.",
        0.10,
    ),
    ComponentSpec(
        "reliability_efficiency",
        "Reliability and efficiency",
        "Completes cases without stalls while using a bounded amount of active execution time.",
        0.10,
    ),
)
COMPONENT_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {component.component_id: component.weight for component in COMPONENTS}
)


_SMOKE_CASES = (
    ScenarioSpec(
        "smoke.direct",
        "Follow an exact no-tool response instruction",
        "instruction",
        45,
        component_id="instruction_following",
        target_active_seconds=15,
    ),
    ScenarioSpec(
        "smoke.bounded",
        "Follow a bounded reasoning-and-format instruction",
        "instruction",
        60,
        component_id="instruction_following",
        target_active_seconds=20,
    ),
    ScenarioSpec(
        "instruction.ambiguity",
        "Ask for missing mutation details without acting",
        "instruction",
        60,
        component_id="instruction_following",
        target_active_seconds=20,
    ),
    ScenarioSpec(
        "instruction.unknown",
        "Report an unknowable value without fabrication",
        "instruction",
        60,
        component_id="instruction_following",
        target_active_seconds=20,
    ),
)
_TOOL_CASES = (
    ScenarioSpec(
        "tools.local_read",
        "Choose the local read tool when evidence is required",
        "tool_judgment",
        90,
        ("local-tools",),
        "tool_judgment",
        1.0,
        35,
    ),
    ScenarioSpec(
        "tools.mutate_verify",
        "Choose the mutation tool with exact task inputs",
        "tool_judgment",
        120,
        ("local-tools",),
        "tool_judgment",
        1.0,
        45,
    ),
    ScenarioSpec(
        "tools.fleet_wait",
        "Choose the Fleet capability tool exactly once",
        "tool_judgment",
        120,
        ("fleet-tools",),
        "tool_judgment",
        1.0,
        35,
    ),
    ScenarioSpec(
        "context.loop",
        "Stop after one unchanged failed tool call",
        "tool_judgment",
        180,
        ("local-tools",),
        "tool_judgment",
        1.0,
        45,
    ),
)
_BROWSER_CASES = (
    ScenarioSpec(
        "browser.observe",
        "Choose browser observation for page-only evidence",
        "web_vision",
        120,
        ("browser",),
        "web_vision",
        1.0,
        50,
    ),
    ScenarioSpec(
        "browser.form",
        "Translate instructions into a grounded browser workflow",
        "web_vision",
        180,
        ("browser",),
        "web_vision",
        1.0,
        75,
    ),
    ScenarioSpec(
        "browser.session",
        "Use remembered browser state across turns",
        "memory_context",
        180,
        ("browser",),
        "memory_context",
        1.0,
        100,
    ),
)
_VISION_CASES = (
    ScenarioSpec(
        "vision.image",
        "Choose visual inspection and ground the answer",
        "web_vision",
        120,
        ("vision",),
        "web_vision",
        1.0,
        55,
    ),
    ScenarioSpec(
        "vision.browser",
        "Choose screenshot-aware browser reasoning",
        "web_vision",
        150,
        ("browser", "vision"),
        "web_vision",
        1.0,
        65,
    ),
)
_CONTEXT_CASES = (
    ScenarioSpec(
        "context.recall",
        "Recall information across bounded turns",
        "memory_context",
        180,
        component_id="memory_context",
        target_active_seconds=80,
    ),
    ScenarioSpec(
        "context.update",
        "Prefer a later correction over stale context",
        "memory_context",
        240,
        component_id="memory_context",
        target_active_seconds=110,
    ),
    ScenarioSpec(
        "context.pressure",
        "Retain and transform implicit facts under bounded context pressure",
        "memory_context",
        600,
        component_id="memory_context",
        target_active_seconds=300,
    ),
)
_ORCHESTRATION_CASES = (
    ScenarioSpec(
        "fleet.resilience",
        "Preserve useful progress through queued and preempted compute",
        "fleet_resilience",
        180,
        ("local-tools",),
        "fleet_resilience",
        1.0,
        65,
    ),
    ScenarioSpec(
        "parallel.orchestration",
        "Orchestrate useful parallel work with a dependent branch",
        "parallel_execution",
        180,
        ("local-tools",),
        "parallel_execution",
        1.0,
        65,
    ),
)
_COMPREHENSIVE_CASES = (
    *_SMOKE_CASES,
    *_TOOL_CASES,
    *_BROWSER_CASES,
    *_VISION_CASES,
    *_CONTEXT_CASES,
    *_ORCHESTRATION_CASES,
)

SUITES: Mapping[str, SuiteSpec] = MappingProxyType(
    {
        suite.suite_id: suite
        for suite in (
            SuiteSpec(
                "smoke",
                "Legacy smoke",
                "Historical partial benchmark; retained only to read old records.",
                "1",
                _SMOKE_CASES,
            ),
            SuiteSpec(
                "tools",
                "Legacy tools",
                "Historical partial benchmark; retained only to read old records.",
                "2",
                _TOOL_CASES[:3],
            ),
            SuiteSpec(
                "browser",
                "Legacy browser",
                "Historical partial benchmark; retained only to read old records.",
                "2",
                _BROWSER_CASES,
            ),
            SuiteSpec(
                "vision",
                "Legacy vision",
                "Historical partial benchmark; retained only to read old records.",
                "1",
                _VISION_CASES,
            ),
            SuiteSpec(
                "context",
                "Legacy context",
                "Historical partial benchmark; retained only to read old records.",
                "2",
                _CONTEXT_CASES,
            ),
            SuiteSpec(
                "comprehensive",
                "Agent capability benchmark",
                "One complete behavioral evaluation of instruction following, memory, tool judgment, web and visual reasoning, reliability, and efficiency.",
                "5",
                _COMPREHENSIVE_CASES,
                accepts_new_submissions=True,
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


def valid_combinations(
    *,
    harness_ids: Iterable[str] | None = None,
    model_ids: Iterable[str] | None = None,
    tool_profile_ids: Iterable[str] | None = None,
) -> tuple[dict[str, object], ...]:
    """Enumerate reviewed rows without manufacturing a Cartesian product.

    Nexus uses this primitive for each ``All`` selector.  Filters intersect the
    explicit allowlist, so a future model/tool addition cannot create an invalid
    harness combination merely because its ID appears in another selector.
    """

    selected_harnesses = None if harness_ids is None else frozenset(harness_ids)
    selected_models = None if model_ids is None else frozenset(model_ids)
    selected_tools = (
        None if tool_profile_ids is None else frozenset(tool_profile_ids)
    )
    return tuple(
        item.public_record()
        for item in COMBINATIONS
        if (selected_harnesses is None or item.harness_id in selected_harnesses)
        and (selected_models is None or item.model_id in selected_models)
        and (
            selected_tools is None
            or item.tool_profile_id in selected_tools
        )
    )


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
        "components": [item.public_record() for item in COMPONENTS],
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
            "component_id",
            "overall_score",
            "quality_score",
            "component_scores",
            "total_wall_ms",
            "end_to_end_wall_ms",
            "total_active_wall_ms",
            "total_compute_wait_ms",
            "model_turn_count",
            "model_call_count",
            "tool_call_count",
            "prompt_tokens",
            "peak_prompt_tokens",
            "context_tokens",
            "completion_tokens",
            "context_pressure_bytes",
            "context_pressure_turns",
            "highest_verified_context_pressure_bytes",
        ],
    }
)


def _suite_public_record(suite: SuiteSpec) -> dict[str, object]:
    return {
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


def public_catalog() -> dict[str, object]:
    """Return a fresh JSON-safe copy of the immutable reviewed catalog."""

    harnesses = []
    for item in public_harness_catalog():
        record = dict(item)
        record["version"] = HARNESS_VERSIONS[str(record["id"])]
        harnesses.append(record)
    benchmark = _suite_public_record(SUITES[DEFAULT_SUITE_ID])
    components = [item.public_record() for item in COMPONENTS]
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
        # ``suites`` and ``default_suite_id`` are compatibility fields.  The
        # single record is also exposed with domain language so new clients do
        # not need to present a suite selector.
        "default_suite_id": DEFAULT_SUITE_ID,
        "benchmark": dict(benchmark),
        "suites": [dict(benchmark)],
        "scoring": {
            "overall_field": "overall_score",
            "scale_min": 0,
            "scale_max": 100,
            "higher_is_better": True,
            "components": components,
            "quality_field": "quality_score",
            "efficiency": {
                "basis": "whole_run_total_active_wall_ms",
                "fleet_wait_excluded": True,
                "missing_or_stuck_charged_as_deadline": True,
                "targets_are_case_bound": True,
            },
            "observable_metrics": {
                "total_time_fields": [
                    "end_to_end_wall_ms",
                    "total_wall_ms",
                    "total_active_wall_ms",
                    "total_compute_wait_ms",
                ],
                "count_fields": [
                    "model_turn_count",
                    "model_call_count",
                    "tool_call_count",
                ],
                "token_fields": [
                    "prompt_tokens",
                    "peak_prompt_tokens",
                    "context_tokens",
                    "completion_tokens",
                ],
                "unknown_value": None,
            },
            "pareto_axes": [
                {
                    "field": "quality_score",
                    "label": "Behavioral quality",
                    "direction": "maximize",
                },
                {
                    "field": "total_active_wall_ms",
                    "label": "Total active time to verified completion",
                    "direction": "minimize",
                },
            ],
        },
        "selection": {
            "all_value": "all",
            "enumerate_from": "combinations",
            "cartesian_product_allowed": False,
        },
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
    "COMPONENTS",
    "COMPONENT_WEIGHTS",
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
    "ComponentSpec",
    "ScenarioSpec",
    "SuiteSpec",
    "combination_for",
    "combination_sha256",
    "public_catalog",
    "valid_combinations",
)
