"""Deterministic benchmark runner with an injectable harness executor."""

from __future__ import annotations

import math
import queue
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Mapping, Protocol

from .catalog import (
    BENCHMARK_CATALOG_SHA256,
    BENCHMARK_CATALOG_VERSION,
    EXECUTOR_PROTOCOL_SHA256,
    EXECUTOR_PROTOCOL_VERSION,
    HARNESS_SOURCE_SHA256,
    RUNNER_PROTOCOL_SHA256,
    RUNNER_PROTOCOL_VERSION,
    RUNNER_SOURCE_SHA256,
    SUITES,
    TOOL_SOURCE_SHA256,
    ScenarioSpec,
    combination_for,
    combination_sha256,
)
from .service import BenchmarkService


CASE_STATUSES = frozenset({"passed", "failed", "timeout", "stuck", "unsupported"})
EXECUTOR_CANCEL_GRACE_SECONDS = 12.0


class ExecutorUnavailable(RuntimeError):
    """No reviewed real harness executor is configured."""


class ExecutionCancelled(RuntimeError):
    """The owner cancelled a benchmark while its exact harness was running."""


class ExecutorUnresolved(RuntimeError):
    """A supposedly cancellable executor did not prove that it stopped."""


@dataclass(frozen=True)
class ExecutionRequest:
    """Prompt-free semantic request passed to a trusted benchmark executor."""

    run_id: str
    suite_id: str
    scenario: ScenarioSpec
    repetition: int
    harness_id: str
    model_id: str
    tool_profile_id: str
    timeout_seconds: int


class HarnessExecutor(Protocol):
    def execute(self, request: ExecutionRequest) -> Mapping[str, object]: ...


class UnavailableHarnessExecutor:
    """Truthful default until a reviewed real-harness bridge is supplied."""

    def execute(self, request: ExecutionRequest) -> Mapping[str, object]:
        del request
        raise ExecutorUnavailable("no reviewed harness executor is configured")


def _bounded_number(value: object, *, low: float, high: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        return None
    return result


def _safe_case_result(
    scenario: ScenarioSpec,
    repetition: int,
    raw: object,
    *,
    measured_wall_ms: float,
) -> dict[str, object]:
    mapping = raw if isinstance(raw, Mapping) else {}
    status = mapping.get("status")
    if status not in CASE_STATUSES:
        status = "failed"
    score = _bounded_number(mapping.get("score"), low=0.0, high=1.0)
    if score is None:
        score = 1.0 if status == "passed" else 0.0
    wall_ms = _bounded_number(mapping.get("wall_ms"), low=0.0, high=86_400_000.0)
    if wall_ms is None:
        wall_ms = max(0.0, measured_wall_ms)
    compute_wait_ms = _bounded_number(
        mapping.get("compute_wait_ms"), low=0.0, high=wall_ms
    )
    if compute_wait_ms is None:
        compute_wait_ms = 0.0
    derived_active_wall_ms = max(0.0, wall_ms - compute_wait_ms)
    active_wall_ms = _bounded_number(
        mapping.get("active_wall_ms"), low=0.0, high=wall_ms
    )
    # Keep all three clocks arithmetically bound. An injected executor cannot
    # skew comparisons by publishing mutually inconsistent timing fields.
    if (
        active_wall_ms is None
        or abs((active_wall_ms + compute_wait_ms) - wall_ms) > 1.0
    ):
        active_wall_ms = derived_active_wall_ms
    result: dict[str, object] = {
        "id": f"{scenario.case_id}:{repetition}",
        "case_id": scenario.case_id,
        "label": scenario.label,
        "category": scenario.category,
        "repetition": repetition,
        "status": status,
        "score": score,
        "wall_ms": wall_ms,
        "active_wall_ms": active_wall_ms,
        "compute_wait_ms": compute_wait_ms,
    }
    for field in ("tool_success", "browser_success"):
        value = mapping.get(field)
        if isinstance(value, bool):
            result[field] = value
    vision_score = _bounded_number(mapping.get("vision_score"), low=0.0, high=1.0)
    if vision_score is not None:
        result["vision_score"] = vision_score
    if scenario.category == "tools" and "tool_success" not in result:
        result["tool_success"] = status == "passed"
    if scenario.category == "browser" and "browser_success" not in result:
        result["browser_success"] = status == "passed"
    if scenario.category == "vision" and "vision_score" not in result:
        result["vision_score"] = score
    if status != "passed":
        result["error_code"] = {
            "timeout": "case_timeout",
            "stuck": "case_stuck",
            "unsupported": "case_unsupported",
        }.get(str(status), "case_failed")
    return result


def _rate(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def summarize_cases(
    cases: list[Mapping[str, object]],
    *,
    planned_cases: int,
) -> dict[str, float | int]:
    scores = [float(item["score"]) for item in cases]
    wall = [float(item["wall_ms"]) for item in cases]
    active_wall = [float(item["active_wall_ms"]) for item in cases]
    compute_wait = [float(item["compute_wait_ms"]) for item in cases]
    passed = [item.get("status") == "passed" for item in cases]
    stuck = [item.get("status") in {"stuck", "timeout"} for item in cases]
    unsupported = [item.get("status") == "unsupported" for item in cases]
    tools = [bool(item["tool_success"]) for item in cases if "tool_success" in item]
    browser = [
        bool(item["browser_success"])
        for item in cases
        if "browser_success" in item
    ]
    vision = [float(item["vision_score"]) for item in cases if "vision_score" in item]
    denominator = max(1, planned_cases)
    return {
        "score": statistics.fmean(scores) if scores else 0.0,
        "completion_rate": sum(passed) / denominator,
        "median_wall_ms": statistics.median(wall) if wall else 0.0,
        "median_active_wall_ms": (
            statistics.median(active_wall) if active_wall else 0.0
        ),
        "median_compute_wait_ms": (
            statistics.median(compute_wait) if compute_wait else 0.0
        ),
        "stuck_rate": sum(stuck) / denominator,
        "unsupported_rate": sum(unsupported) / denominator,
        "tool_success_rate": _rate(tools),
        "browser_success_rate": _rate(browser),
        "vision_score": statistics.fmean(vision) if vision else 0.0,
        "case_count": len(cases),
        "passed_cases": sum(passed),
    }


def _request_executor_cancel(
    executor: HarnessExecutor,
    request: ExecutionRequest,
) -> bool:
    cancel = getattr(executor, "cancel", None)
    if callable(cancel):
        try:
            cancel(request)
            return True
        except Exception:
            pass
    return False


def _execute_with_deadline(
    service: BenchmarkService,
    executor: HarnessExecutor,
    request: ExecutionRequest,
) -> tuple[object, float, bool]:
    """Bound even an injected synchronous executor and poll durable cancellation.

    The production executor exposes ``cancel`` and proves termination of its
    exact child process group.  A non-cooperative injected executor is isolated
    in a daemon thread so it cannot hold the benchmark worker past the deadline;
    such a case is recorded as stuck rather than successful.
    """

    prepare = getattr(executor, "prepare", None)
    if callable(prepare):
        prepare(request)
    result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result_queue.put_nowait((True, executor.execute(request)))
        except BaseException as exc:
            try:
                result_queue.put_nowait((False, exc))
            except queue.Full:
                pass

    started = time.monotonic()
    worker = threading.Thread(
        target=invoke,
        name=f"benchmark-case-{request.scenario.case_id}",
        daemon=True,
    )
    worker.start()
    remaining_budget = max(0.001, float(request.timeout_seconds))
    last_tick = time.monotonic()
    previously_paused = False
    paused_elapsed = 0.0
    cancelled = False
    while worker.is_alive():
        now = time.monotonic()
        deadline_paused = getattr(executor, "deadline_paused", None)
        paused = callable(deadline_paused) and deadline_paused(request) is True
        # Do not charge the polling interval in which a pause transition was
        # observed. That interval can contain an arbitrary amount of proven
        # Fleet wait and cannot be split safely from this supervising thread.
        elapsed = max(0.0, now - last_tick)
        if paused or previously_paused:
            paused_elapsed += elapsed
        else:
            remaining_budget -= elapsed
        last_tick = now
        previously_paused = paused
        if service._cancel_requested(request.run_id):
            # An executor may atomically finish its case and request that the
            # remaining cases be cancelled. Preserve that completed evidence.
            worker.join(timeout=0.05)
            if not worker.is_alive():
                break
            cancelled = True
            cooperative = _request_executor_cancel(executor, request)
            worker.join(
                timeout=EXECUTOR_CANCEL_GRACE_SECONDS if cooperative else 0.05
            )
            if worker.is_alive():
                raise ExecutorUnresolved()
            break
        if remaining_budget <= 0:
            cooperative = _request_executor_cancel(executor, request)
            worker.join(
                timeout=EXECUTOR_CANCEL_GRACE_SECONDS if cooperative else 0.05
            )
            measured = (time.monotonic() - started) * 1000.0
            return (
                {
                    "status": "timeout" if not worker.is_alive() else "stuck",
                    "score": 0.0,
                    "wall_ms": measured,
                    "active_wall_ms": max(0.0, measured - paused_elapsed * 1000.0),
                    "compute_wait_ms": min(measured, paused_elapsed * 1000.0),
                },
                measured,
                False,
            )
        worker.join(
            timeout=0.1 if paused else min(0.1, remaining_budget)
        )
    finished = time.monotonic()
    final_elapsed = max(0.0, finished - last_tick)
    final_pause = getattr(executor, "deadline_paused", None)
    if previously_paused or (callable(final_pause) and final_pause(request) is True):
        paused_elapsed += final_elapsed
    measured = (finished - started) * 1000.0
    if cancelled:
        return (
            {
                "status": "failed",
                "score": 0.0,
                "wall_ms": measured,
                "active_wall_ms": max(0.0, measured - paused_elapsed * 1000.0),
                "compute_wait_ms": min(measured, paused_elapsed * 1000.0),
            },
            measured,
            True,
        )
    try:
        ok, value = result_queue.get_nowait()
    except queue.Empty:
        return ({"status": "stuck", "score": 0.0}, measured, False)
    if ok:
        if isinstance(value, Mapping):
            timed = dict(value)
            wait_ms = min(measured, max(0.0, paused_elapsed * 1000.0))
            timed.setdefault("wall_ms", measured)
            timed.setdefault("compute_wait_ms", wait_ms)
            timed.setdefault("active_wall_ms", max(0.0, measured - wait_ms))
            return timed, measured, False
        return value, measured, False
    if isinstance(value, BaseException):
        raise value
    return ({"status": "failed", "score": 0.0}, measured, False)


def run_benchmark(
    service: BenchmarkService,
    run_id: str,
    *,
    executor: HarnessExecutor | None = None,
) -> dict[str, object]:
    """Claim and run one queued record; never manufacture a successful result."""

    if not service._register_worker(run_id):
        return service.get_run(run_id)
    claimed = service._claim_run(run_id)
    if claimed is None:
        return service.get_run(run_id)
    suite = SUITES.get(str(claimed["suite_id"]))
    combination = combination_for(
        str(claimed["harness_id"]),
        str(claimed["model_id"]),
        str(claimed["tool_profile_id"]),
    )
    if (
        suite is None
        or claimed["suite_version"] != suite.version
        or claimed["suite_sha256"] != suite.sha256
        or claimed["catalog_version"] != BENCHMARK_CATALOG_VERSION
        or claimed["catalog_sha256"] != BENCHMARK_CATALOG_SHA256
        or claimed["runner_protocol_version"] != RUNNER_PROTOCOL_VERSION
        or claimed["runner_protocol_sha256"] != RUNNER_PROTOCOL_SHA256
        or claimed["executor_protocol_version"] != EXECUTOR_PROTOCOL_VERSION
        or claimed["executor_protocol_sha256"] != EXECUTOR_PROTOCOL_SHA256
        or claimed["runner_source_sha256"] != RUNNER_SOURCE_SHA256
        or claimed["harness_source_sha256"] != HARNESS_SOURCE_SHA256
        or claimed["tool_source_sha256"] != TOOL_SOURCE_SHA256
        or combination is None
        or claimed["combination_sha256"] != combination_sha256(combination)
        or any(
            claimed[key] != value
            for key, value in combination.items()
            if key != "id"
        )
    ):
        service._mark_failed(run_id, error_code="runner_failed")
        return service.get_run(run_id)
    selected_executor = executor or UnavailableHarnessExecutor()
    repetitions = int(claimed["repetitions"])
    planned = len(suite.cases) * repetitions
    cases: list[dict[str, object]] = []
    for repetition in range(1, repetitions + 1):
        for scenario in suite.cases:
            if service._cancel_requested(run_id):
                summary = summarize_cases(cases, planned_cases=planned)
                return service._finish_run(
                    run_id,
                    status_value="cancelled",
                    summary=summary,
                    cases=cases,
                )
            request = ExecutionRequest(
                run_id=run_id,
                suite_id=suite.suite_id,
                scenario=scenario,
                repetition=repetition,
                harness_id=str(claimed["harness_id"]),
                model_id=str(claimed["model_id"]),
                tool_profile_id=str(claimed["tool_profile_id"]),
                timeout_seconds=scenario.timeout_seconds,
            )
            try:
                raw, measured, cancelled = _execute_with_deadline(
                    service, selected_executor, request
                )
                if cancelled:
                    return service._finish_run(
                        run_id,
                        status_value="cancelled",
                        summary=summarize_cases(cases, planned_cases=planned),
                        cases=cases,
                    )
            except ExecutionCancelled:
                return service._finish_run(
                    run_id,
                    status_value="cancelled",
                    summary=summarize_cases(cases, planned_cases=planned),
                    cases=cases,
                )
            except ExecutorUnresolved:
                service._mark_failed(run_id, error_code="executor_stuck")
                return service.get_run(run_id)
            except ExecutorUnavailable:
                measured = 0.0
                cases.append(
                    _safe_case_result(
                        scenario,
                        repetition,
                        {"status": "failed", "score": 0.0},
                        measured_wall_ms=measured,
                    )
                )
                return service._finish_run(
                    run_id,
                    status_value="failed",
                    summary=summarize_cases(cases, planned_cases=planned),
                    cases=cases,
                    error_code="executor_unavailable",
                )
            except Exception:
                raw = {"status": "failed", "score": 0.0}
                measured = 0.0
            safe = _safe_case_result(
                scenario,
                repetition,
                raw,
                measured_wall_ms=measured,
            )
            cases.append(safe)
            if safe["status"] == "stuck":
                return service._finish_run(
                    run_id,
                    status_value="failed",
                    summary=summarize_cases(cases, planned_cases=planned),
                    cases=cases,
                    error_code="executor_stuck",
                )
    return service._finish_run(
        run_id,
        status_value="succeeded",
        summary=summarize_cases(cases, planned_cases=planned),
        cases=cases,
    )


__all__ = (
    "CASE_STATUSES",
    "ExecutionRequest",
    "ExecutionCancelled",
    "ExecutorUnresolved",
    "ExecutorUnavailable",
    "HarnessExecutor",
    "UnavailableHarnessExecutor",
    "run_benchmark",
    "summarize_cases",
)
