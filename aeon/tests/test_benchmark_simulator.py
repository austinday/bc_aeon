"""Hermetic adversarial tests for the benchmark-only Fleet/DAG simulator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from aeon.benchmarks.catalog import COMPONENTS, SUITES
from aeon.benchmarks.executor import (
    CONTEXT_PRESSURE_TURN_BYTES,
    FleetHarnessExecutor,
    ProcessResult,
    _context_pressure_filler,
)
from aeon.benchmarks.runner import ExecutionRequest, ExecutorUnavailable
from aeon.benchmarks.service import BenchmarkService
from aeon.core.benchmark_receipt import (
    CAPABILITY_RECEIPT_KEY_ENV,
    CAPABILITY_RECEIPT_PATH_ENV,
    TRACE_CASE_ID_ENV,
    TRACE_NONCE_ENV,
    TRACE_REPETITION_ENV,
    TRACE_RUN_ID_ENV,
    ScenarioEffectReceipt,
    ToolCallReceipt,
    decode_capability_receipts,
    emit_tool_call_receipt,
)
from aeon.core.benchmark_simulator import (
    SCENARIO_CAPABILITY_ENV,
    SCENARIO_TOOL_NAME,
    ScenarioSession,
    load_scenario_capability,
    mint_scenario_capability,
    score_scenario_effects,
    simulator_preflight,
)
from aeon.core.sub_agent_environment import bounded_sub_agent_environment
from aeon.core.worker import Worker
from aeon.tools.benchmark_scenarios import BenchmarkWorkflowTool
from aeon.tools.command_fleet_guard import scrubbed_payload_environment


RUN_ID = "run-" + "1" * 32
TRACE_NONCE = "2" * 64
KEY = "3" * 64


def _capability_environment(path: Path, case_id: str) -> dict[str, str]:
    metadata = path.stat()
    raw = mint_scenario_capability(
        run_id=RUN_ID,
        case_id=case_id,
        repetition=1,
        trace_nonce=TRACE_NONCE,
        receipt_path=path,
        receipt_device=metadata.st_dev,
        receipt_inode=metadata.st_ino,
        receipt_key=KEY,
    )
    assert raw is not None
    return {
        CAPABILITY_RECEIPT_PATH_ENV: str(path),
        CAPABILITY_RECEIPT_KEY_ENV: KEY,
        TRACE_RUN_ID_ENV: RUN_ID,
        TRACE_CASE_ID_ENV: case_id,
        TRACE_REPETITION_ENV: "1",
        TRACE_NONCE_ENV: TRACE_NONCE,
        SCENARIO_CAPABILITY_ENV: raw,
    }


def _effects(path: Path, case_id: str) -> tuple[ScenarioEffectReceipt, ...]:
    records = decode_capability_receipts(
        path.read_bytes(),
        key=KEY,
        run_id=RUN_ID,
        case_id=case_id,
        repetition=1,
        trace_nonce=TRACE_NONCE,
    )
    return tuple(item for item in records if isinstance(item, ScenarioEffectReceipt))


def _request(
    case_id: str,
    *,
    harness_id: str = "opencode",
    run_id: str = RUN_ID,
) -> ExecutionRequest:
    scenario = next(
        item for item in SUITES["comprehensive"].cases if item.case_id == case_id
    )
    return ExecutionRequest(
        run_id=run_id,
        suite_id="comprehensive",
        scenario=scenario,
        repetition=1,
        harness_id=harness_id,
        model_id="local/qwen",
        tool_profile_id="fleet-local",
        timeout_seconds=scenario.timeout_seconds,
    )


def _call(session: ScenarioSession, operation: str, refs=(), branch: str = "") -> dict:
    arguments = {"operation": operation}
    if refs:
        arguments["reference_ids"] = list(refs)
    if branch:
        arguments["branch"] = branch
    emit_tool_call_receipt(SCENARIO_TOOL_NAME, arguments)
    return json.loads(session.execute(operation, list(refs), branch))


def _optimal_fleet(session: ScenarioSession) -> str:
    job = _call(session, "submit_gpu_job")["job_id"]
    cpu = _call(session, "perform_cpu_manifest")["artifact_id"]
    _call(session, "check_job", [job])
    checkpoint = _call(session, "run_gpu_stage", [job])["checkpoint_id"]
    _call(session, "continue_manager", [job])
    gpu = _call(session, "resume_gpu_job", [job, checkpoint])["artifact_id"]
    return _call(session, "finalize", [cpu, gpu])["completion_code"]


def _optimal_parallel(session: ScenarioSession) -> str:
    task_a = _call(session, "delegate", branch="a")["task_id"]
    task_b = _call(session, "delegate", branch="b")["task_id"]
    principal = _call(session, "principal_work")["report_id"]
    report_a = _call(session, "collect", [task_a])["report_id"]
    task_c = _call(session, "delegate", [report_a], "c")["task_id"]
    prep = _call(session, "integration_prep")["report_id"]
    report_b = _call(session, "collect", [task_b])["report_id"]
    report_c = _call(session, "collect", [task_c])["report_id"]
    return _call(
        session,
        "integrate",
        [report_a, report_b, report_c, principal, prep],
    )["completion_code"]


def test_catalog_has_one_weighted_case_for_each_new_component() -> None:
    assert {item.component_id: item.weight for item in COMPONENTS} == {
        "instruction_following": 0.20,
        "memory_context": 0.20,
        "tool_judgment": 0.20,
        "web_vision": 0.10,
        "fleet_resilience": 0.10,
        "parallel_execution": 0.10,
        "reliability_efficiency": 0.10,
    }
    cases = SUITES["comprehensive"].cases
    assert len(cases) == 18
    assert {item.case_id for item in cases} >= {
        "fleet.resilience",
        "parallel.orchestration",
    }


def test_tool_is_absent_without_exact_capability_and_all_benchmark_env_is_scrubbed(
    tmp_path: Path,
) -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert BenchmarkWorkflowTool().is_internal is True

    path = tmp_path / "receipt"
    path.touch(mode=0o600)
    environment = _capability_environment(path, "fleet.resilience")
    assert SCENARIO_CAPABILITY_ENV not in scrubbed_payload_environment(environment)
    assert SCENARIO_CAPABILITY_ENV not in bounded_sub_agent_environment(environment)
    assert CAPABILITY_RECEIPT_KEY_ENV not in scrubbed_payload_environment(environment)
    assert CAPABILITY_RECEIPT_KEY_ENV not in bounded_sub_agent_environment(environment)


def test_valid_capability_clamps_worker_to_synthetic_tool_only(tmp_path: Path) -> None:
    path = tmp_path / "receipt"
    path.touch(mode=0o600)
    environment = _capability_environment(path, "parallel.orchestration")
    with patch.dict(os.environ, environment, clear=True):
        tool = BenchmarkWorkflowTool()
        assert tool.is_internal is False
        worker = object.__new__(Worker)
        worker.tools = {
            SCENARIO_TOOL_NAME: tool,
            "run_command": object(),
            "spawn_sub_agent": object(),
            "fleet_submit_batch_job": object(),
        }
        worker.collaborator_mode_state = None
        assert worker._active_tool_names() == {SCENARIO_TOOL_NAME}


def test_capability_rejects_tamper_wrong_context_and_replaced_inode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt"
    path.touch(mode=0o600)
    environment = _capability_environment(path, "fleet.resilience")
    with patch.dict(os.environ, environment, clear=True):
        assert load_scenario_capability() is not None

    changed = json.loads(environment[SCENARIO_CAPABILITY_ENV])
    changed["payload"]["case_id"] = "parallel.orchestration"
    tampered = dict(environment)
    tampered[SCENARIO_CAPABILITY_ENV] = json.dumps(changed)
    with patch.dict(os.environ, tampered, clear=True):
        assert load_scenario_capability() is None

    wrong_context = dict(environment)
    wrong_context[TRACE_REPETITION_ENV] = "2"
    with patch.dict(os.environ, wrong_context, clear=True):
        assert load_scenario_capability() is None

    replacement = tmp_path / "replacement"
    replacement.touch(mode=0o600)
    assert replacement.stat().st_ino != path.stat().st_ino
    path.unlink()
    replacement.rename(path)
    with patch.dict(os.environ, environment, clear=True):
        assert load_scenario_capability() is None


def test_capability_rejects_hardlinked_receipt(tmp_path: Path) -> None:
    path = tmp_path / "receipt"
    path.touch(mode=0o600)
    environment = _capability_environment(path, "fleet.resilience")
    os.link(path, tmp_path / "second-name")
    with patch.dict(os.environ, environment, clear=True):
        assert load_scenario_capability() is None


def test_receipts_reject_duplicate_reorder_and_cross_process_sequence_gap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt"
    path.touch(mode=0o600)
    environment = _capability_environment(path, "fleet.resilience")
    with patch.dict(os.environ, environment, clear=True):
        emit_tool_call_receipt("open_file", {"file_path": "/a"})
        emit_tool_call_receipt("open_file", {"file_path": "/b"})
    lines = path.read_bytes().splitlines(keepends=True)
    records = decode_capability_receipts(
        b"".join(lines),
        key=KEY,
        run_id=RUN_ID,
        case_id="fleet.resilience",
        repetition=1,
        trace_nonce=TRACE_NONCE,
    )
    assert [item.sequence for item in records if isinstance(item, ToolCallReceipt)] == [1, 2]
    assert decode_capability_receipts(
        lines[0] + lines[0] + lines[1], key=KEY
    ) == ()
    assert decode_capability_receipts(lines[1] + lines[0], key=KEY) == ()
    assert decode_capability_receipts(lines[1], key=KEY) == ()


def test_effect_stream_rejects_duplicate_gap_and_mixed_case(tmp_path: Path) -> None:
    fleet_path = tmp_path / "fleet"
    fleet_path.touch(mode=0o600)
    fleet_environment = _capability_environment(fleet_path, "fleet.resilience")
    with patch.dict(os.environ, fleet_environment, clear=True):
        fleet_capability = load_scenario_capability()
        assert fleet_capability is not None
        session = ScenarioSession(fleet_capability)
        _call(session, "submit_gpu_job")
    fleet_lines = fleet_path.read_bytes().splitlines(keepends=True)
    effect_lines = [
        line
        for line in fleet_lines
        if json.loads(line)["payload"]["receipt_type"] == "scenario_effect"
    ]
    assert len(effect_lines) == 2
    assert decode_capability_receipts(
        b"".join(fleet_lines + [effect_lines[-1]]), key=KEY
    ) == ()
    without_ready = [line for line in fleet_lines if line != effect_lines[0]]
    assert decode_capability_receipts(b"".join(without_ready), key=KEY) == ()

    parallel_path = tmp_path / "parallel"
    parallel_path.touch(mode=0o600)
    parallel_environment = _capability_environment(
        parallel_path, "parallel.orchestration"
    )
    with patch.dict(os.environ, parallel_environment, clear=True):
        parallel_capability = load_scenario_capability()
        assert parallel_capability is not None
        ScenarioSession(parallel_capability)
    assert decode_capability_receipts(
        fleet_path.read_bytes() + parallel_path.read_bytes(), key=KEY
    ) == ()


def test_session_restart_replays_private_ledger_and_keeps_same_job(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt"
    path.touch(mode=0o600)
    environment = _capability_environment(path, "fleet.resilience")
    with patch.dict(os.environ, environment, clear=True):
        capability = load_scenario_capability()
        assert capability is not None
        first = ScenarioSession(capability)
        job = _call(first, "submit_gpu_job")["job_id"]
        cpu = _call(first, "perform_cpu_manifest")["artifact_id"]

        # A fresh facade models an MCP process restart. State must be recovered
        # only from the signed inode-bound effect stream.
        resumed = ScenarioSession(capability)
        _call(resumed, "check_job", [job])
        checkpoint = _call(resumed, "run_gpu_stage", [job])["checkpoint_id"]
        _call(resumed, "continue_manager", [job])
        gpu = _call(resumed, "resume_gpu_job", [job, checkpoint])["artifact_id"]
        completion = _call(resumed, "finalize", [cpu, gpu])["completion_code"]
        effects = _effects(path, "fleet.resilience")
        scored = score_scenario_effects(capability, effects)
    assert completion
    assert [event.operation for event in effects].count("fixture_reopened") == 1
    assert scored["score"] == 1.0


def test_preflight_exercises_both_optimal_state_machines() -> None:
    assert simulator_preflight("fleet.resilience") is True
    assert simulator_preflight("parallel.orchestration") is True
    assert simulator_preflight("not-a-scenario") is False


@pytest.mark.parametrize(
    ("case_id", "runner"),
    [
        ("fleet.resilience", _optimal_fleet),
        ("parallel.orchestration", _optimal_parallel),
    ],
)
def test_optimal_simulated_workflows_have_effect_backed_full_scores(
    tmp_path: Path, case_id: str, runner
) -> None:
    path = tmp_path / "receipt"
    path.touch(mode=0o600)
    environment = _capability_environment(path, case_id)
    with patch.dict(os.environ, environment, clear=True):
        capability = load_scenario_capability()
        assert capability is not None
        session = ScenarioSession(capability)
        completion = runner(session)
        scored = score_scenario_effects(capability, _effects(path, case_id))
    assert completion
    assert scored["status"] == "passed"
    assert scored["score"] == 1.0
    if case_id == "parallel.orchestration":
        assert scored["max_parallelism"] == 2
        assert scored["useful_overlap_ratio"] == 1.0
        assert scored["idle_wait_ratio"] == 0.0


def test_fleet_duplicate_poll_and_restart_cannot_game_lifecycle_score(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt"
    path.touch(mode=0o600)
    environment = _capability_environment(path, "fleet.resilience")
    with patch.dict(os.environ, environment, clear=True):
        capability = load_scenario_capability()
        assert capability is not None
        session = ScenarioSession(capability)
        job = _call(session, "submit_gpu_job")["job_id"]
        assert _call(session, "submit_gpu_job")["status"] == "rejected"
        _call(session, "poll_job", [job])
        assert _call(session, "restart_job", [job])["status"] == "rejected"
        scored = score_scenario_effects(
            capability, _effects(path, "fleet.resilience")
        )
    assert scored["score"] < 1.0
    assert scored["duplicate_submission_count"] == 1
    assert scored["fleet_compute_judgment_score"] == 0.0


def test_parallel_order_only_recitation_has_no_concurrency_or_integration_credit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt"
    path.touch(mode=0o600)
    environment = _capability_environment(path, "parallel.orchestration")
    with patch.dict(os.environ, environment, clear=True):
        capability = load_scenario_capability()
        assert capability is not None
        session = ScenarioSession(capability)
        assert _call(session, "delegate", branch="c")["status"] == "rejected"
        assert _call(session, "delegate_principal")["status"] == "rejected"
        assert _call(session, "integrate", ["made-up-report"])["status"] == "rejected"
        scored = score_scenario_effects(
            capability, _effects(path, "parallel.orchestration")
        )
    assert scored["score"] < 1.0
    assert scored["max_parallelism"] == 0
    assert scored["integration_score"] == 0.0


@pytest.mark.parametrize(
    ("case_id", "runner"),
    [
        ("fleet.resilience", _optimal_fleet),
        ("parallel.orchestration", _optimal_parallel),
    ],
)
def test_executor_scores_private_effect_ledger_not_model_action_words(
    tmp_path: Path, case_id: str, runner
) -> None:
    service = BenchmarkService(
        tmp_path / "benchmarks", launcher=lambda _argv: None
    )
    scenario = next(
        item for item in SUITES["comprehensive"].cases if item.case_id == case_id
    )
    request = ExecutionRequest(
        run_id=RUN_ID,
        suite_id="comprehensive",
        scenario=scenario,
        repetition=1,
        harness_id="opencode",
        model_id="local/qwen",
        tool_profile_id="fleet-local",
        timeout_seconds=scenario.timeout_seconds,
    )

    def fake_process(_argv, _cwd, environment, _timeout, _cancel, _transition):
        with patch.dict(os.environ, dict(environment), clear=True):
            capability = load_scenario_capability()
            assert capability is not None
            completion = runner(ScenarioSession(capability))
        return ProcessResult("exited", 0, (completion + "\n").encode("ascii"), 4.0)

    with FleetHarnessExecutor(
        service,
        RUN_ID,
        process_runner=fake_process,
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        result = executor.execute(request)
        assert executor._receipt_root.parent == executor.harness_state
        assert not executor._receipt_root.is_relative_to(executor.workspace)
    assert result["status"] == "passed"
    assert result["score"] == 1.0


def test_executor_treats_recited_sequence_without_ready_effect_as_invalid(
    tmp_path: Path,
) -> None:
    service = BenchmarkService(
        tmp_path / "benchmarks", launcher=lambda _argv: None
    )
    scenario = next(
        item
        for item in SUITES["comprehensive"].cases
        if item.case_id == "fleet.resilience"
    )
    request = ExecutionRequest(
        run_id=RUN_ID,
        suite_id="comprehensive",
        scenario=scenario,
        repetition=1,
        harness_id="opencode",
        model_id="local/qwen",
        tool_profile_id="fleet-local",
        timeout_seconds=scenario.timeout_seconds,
    )

    def prose_only(_argv, _cwd, _environment, _timeout, _cancel, _transition):
        return ProcessResult(
            "exited",
            0,
            b"submit cpu checkpoint reacquire resume complete\n",
            1.0,
        )

    with FleetHarnessExecutor(
        service,
        RUN_ID,
        process_runner=prose_only,
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        with pytest.raises(ExecutorUnavailable, match="evidence is incomplete"):
            executor.execute(request)


def test_unexecuted_extra_workflow_proposal_is_behavioral_not_infrastructure(
    tmp_path: Path,
) -> None:
    service = BenchmarkService(tmp_path / "benchmarks", launcher=lambda _argv: None)
    request = _request("fleet.resilience")

    def extra_proposal(_argv, _cwd, environment, _timeout, _cancel, _transition):
        with patch.dict(os.environ, dict(environment), clear=True):
            capability = load_scenario_capability()
            assert capability is not None
            emit_tool_call_receipt(
                SCENARIO_TOOL_NAME, {"operation": "idle_wait"}
            )
            completion = _optimal_fleet(ScenarioSession(capability))
        return ProcessResult("exited", 0, (completion + "\n").encode(), 4.0)

    with FleetHarnessExecutor(
        service,
        RUN_ID,
        process_runner=extra_proposal,
        readiness_checker=lambda _harness: {"supported": True},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        result = executor.execute(request)
    assert result["status"] == "failed"
    assert 0.0 < result["score"] < 1.0


@pytest.mark.parametrize(
    "case_id",
    ("browser.observe", "browser.form", "browser.session", "vision.browser"),
)
def test_controlled_browser_seed_failure_is_infrastructure_invalid(
    tmp_path: Path, case_id: str
) -> None:
    service = BenchmarkService(tmp_path / "benchmarks", launcher=lambda _argv: None)

    def should_not_run(*_args):
        raise AssertionError("harness must not run without a seeded fixture")

    with FleetHarnessExecutor(
        service,
        RUN_ID,
        process_runner=should_not_run,
        readiness_checker=lambda _harness: {"supported": True},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        with pytest.raises(ExecutorUnavailable, match="fixture .* seed failed"):
            executor.execute(_request(case_id))


def test_fixture_negative_verify_is_model_failure_but_transport_error_is_invalid(
    tmp_path: Path,
) -> None:
    service = BenchmarkService(tmp_path / "benchmarks", launcher=lambda _argv: None)
    with FleetHarnessExecutor(
        service,
        RUN_ID,
        process_runner=lambda *_args: ProcessResult("exited", 0, b"", 1.0),
        readiness_checker=lambda _harness: {"supported": True},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        assert executor._verify_fixture_operation("form-v1") is False

        def unavailable(_fixture, _operation):
            raise ConnectionError("fixture transport unavailable")

        executor._fixture_client = unavailable
        with pytest.raises(ExecutorUnavailable, match="verify was unavailable"):
            executor._verify_fixture_operation("form-v1")
        with pytest.raises(ExecutorUnavailable, match="reopen failed"):
            executor._require_fixture_operation("session-v1", "reopen")


@pytest.mark.parametrize("harness_id", ("opencode", "legacy-aeon"))
def test_context_pressure_uses_resumed_sub_40kb_turns_and_exact_final_tool(
    tmp_path: Path, harness_id: str
) -> None:
    run_id = "run-" + ("4" if harness_id == "opencode" else "5") * 32
    service = BenchmarkService(tmp_path / harness_id, launcher=lambda _argv: None)
    prompts: list[str] = []

    def fake_process(argv, _cwd, environment, _timeout, _cancel, _transition):
        prompt = argv[argv.index("--start") + 1]
        prompts.append(prompt)
        assert max(len(str(argument).encode("utf-8")) for argument in argv) < 40_000
        if "planning note says Alder" in prompt:
            output = "ACK"
        elif "PRESSURE_" in prompt:
            output = next(
                word.rstrip(".\n")
                for word in prompt.split()
                if word.startswith("PRESSURE_")
            )
        elif "sum of Alder and Cedar" in prompt:
            output = "60"
        elif "Birch minus Alder" in prompt:
            output = "12"
        elif "descending order" in prompt:
            output = "43,29,17"
        elif "final audit needs the exact token" in prompt:
            with patch.dict(os.environ, dict(environment), clear=True):
                evidence_path = _cwd / "fixtures" / "read-token.txt"
                emit_tool_call_receipt(
                    "open_file", {"file_path": str(evidence_path)}
                )
            output = "BENCH_READ_LARK_7319"
        else:  # pragma: no cover - protects the closed prompt catalog
            raise AssertionError(f"unexpected pressure prompt: {prompt[:100]}")
        return ProcessResult("exited", 0, (output + "\n").encode(), 1.0)

    with FleetHarnessExecutor(
        service,
        run_id,
        process_runner=fake_process,
        readiness_checker=lambda _harness: {"supported": True},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        result = executor.execute(
            _request("context.pressure", harness_id=harness_id, run_id=run_id)
        )
    assert result["status"] == "passed"
    assert result["score"] == 1.0
    assert result["context_pressure_bytes"] == 224_000
    assert result["context_pressure_turns"] == 7
    assert result["highest_verified_context_pressure_bytes"] == 224_000
    assert len(prompts) == 12
    pressure_prompts = [prompt for prompt in prompts if "PRESSURE_" in prompt]
    assert len(pressure_prompts) == 7
    assert all(len(prompt.encode("utf-8")) < 40_000 for prompt in pressure_prompts)
    first = _context_pressure_filler(1, CONTEXT_PRESSURE_TURN_BYTES)
    assert len(first.encode("ascii")) == CONTEXT_PRESSURE_TURN_BYTES
    assert first == _context_pressure_filler(1, CONTEXT_PRESSURE_TURN_BYTES)
    assert first != _context_pressure_filler(2, CONTEXT_PRESSURE_TURN_BYTES)
