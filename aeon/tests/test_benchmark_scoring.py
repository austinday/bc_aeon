"""Deterministic tests for comprehensive benchmark aggregation semantics."""

from __future__ import annotations

import pytest

from aeon.benchmarks.catalog import COMPONENTS, SUITES
from aeon.benchmarks.runner import summarize_cases


def _case(
    scenario,
    *,
    active_ms: float,
    repetition: int = 1,
    model_turn_count: int = 1,
    model_call_count: int | None = None,
    tool_call_count: int = 0,
    with_tokens: bool = False,
) -> dict[str, object]:
    return {
        "case_id": scenario.case_id,
        "component_id": scenario.component_id,
        "repetition": repetition,
        "status": "passed",
        "score": 1.0,
        "wall_ms": active_ms,
        "active_wall_ms": active_ms,
        "compute_wait_ms": 0.0,
        "model_turn_count": model_turn_count,
        "model_call_count": model_call_count,
        "tool_call_count": tool_call_count,
        "prompt_tokens": 10 if with_tokens else None,
        "peak_prompt_tokens": 8 if with_tokens else None,
        "context_tokens": 20 if with_tokens else None,
        "completion_tokens": 5 if with_tokens else None,
    }


def test_whole_run_efficiency_uses_total_active_time_and_fixed_case_bounds() -> None:
    scenarios = SUITES["comprehensive"].cases
    at_targets = [
        _case(scenario, active_ms=scenario.target_active_seconds * 1000.0)
        for scenario in scenarios
    ]
    fast = summarize_cases(
        at_targets, planned_cases=len(scenarios), scenarios=scenarios
    )
    assert fast["quality_score"] == pytest.approx(100.0)
    assert fast["component_scores"]["reliability_efficiency"] == pytest.approx(
        100.0
    )
    assert fast["overall_score"] == pytest.approx(100.0)

    at_deadlines = [
        _case(scenario, active_ms=scenario.timeout_seconds * 1000.0)
        for scenario in scenarios
    ]
    slow = summarize_cases(
        at_deadlines, planned_cases=len(scenarios), scenarios=scenarios
    )
    reliability_weight = next(
        item.weight
        for item in COMPONENTS
        if item.component_id == "reliability_efficiency"
    )
    assert slow["quality_score"] == pytest.approx(100.0)
    assert slow["component_scores"]["reliability_efficiency"] == pytest.approx(
        50.0
    )
    assert slow["overall_score"] == pytest.approx(
        100.0 - 50.0 * reliability_weight
    )
    assert slow["total_active_wall_ms"] == pytest.approx(
        sum(scenario.timeout_seconds * 1000.0 for scenario in scenarios)
    )


def test_turn_count_does_not_reward_fragmenting_same_total_active_work() -> None:
    scenarios = SUITES["comprehensive"].cases
    one_long_call = [
        _case(scenario, active_ms=1000.0, model_turn_count=1)
        for scenario in scenarios
    ]
    many_short_calls = [
        _case(scenario, active_ms=1000.0, model_turn_count=9)
        for scenario in scenarios
    ]
    first = summarize_cases(
        one_long_call, planned_cases=len(scenarios), scenarios=scenarios
    )
    second = summarize_cases(
        many_short_calls, planned_cases=len(scenarios), scenarios=scenarios
    )
    assert first["overall_score"] == pytest.approx(second["overall_score"])
    assert first["total_active_wall_ms"] == second["total_active_wall_ms"]
    assert first["model_turn_count"] == len(scenarios)
    assert second["model_turn_count"] == 9 * len(scenarios)


def test_missing_cases_are_not_renormalized_and_consume_fixed_deadlines() -> None:
    scenarios = SUITES["comprehensive"].cases
    observed = [
        _case(
            scenarios[0],
            active_ms=scenarios[0].target_active_seconds * 1000.0,
        )
    ]
    partial = summarize_cases(
        observed, planned_cases=len(scenarios), scenarios=scenarios
    )
    assert partial["case_count"] == 1
    assert partial["completion_rate"] == pytest.approx(1.0 / len(scenarios))
    assert partial["overall_score"] < 20.0
    assert any(score == 0.0 for score in partial["component_scores"].values())


def test_counts_and_tokens_are_reported_only_when_complete() -> None:
    scenarios = SUITES["comprehensive"].cases
    incomplete = [
        _case(scenario, active_ms=1.0, model_call_count=None)
        for scenario in scenarios
    ]
    result = summarize_cases(
        incomplete, planned_cases=len(scenarios), scenarios=scenarios
    )
    assert result["model_call_count"] is None
    assert result["prompt_tokens"] is None
    assert result["peak_prompt_tokens"] is None
    assert result["context_tokens"] is None
    assert result["completion_tokens"] is None
    assert result["token_metrics_complete"] is False

    complete = [
        _case(
            scenario,
            active_ms=1.0,
            model_call_count=1,
            tool_call_count=2,
            with_tokens=True,
        )
        for scenario in scenarios
    ]
    result = summarize_cases(
        complete, planned_cases=len(scenarios), scenarios=scenarios
    )
    assert result["model_call_count"] == len(scenarios)
    assert result["tool_call_count"] == 2 * len(scenarios)
    assert result["prompt_tokens"] == 10 * len(scenarios)
    assert result["peak_prompt_tokens"] == 8
    assert result["context_tokens"] == 20 * len(scenarios)
    assert result["completion_tokens"] == 5 * len(scenarios)
    assert result["token_metrics_complete"] is True
