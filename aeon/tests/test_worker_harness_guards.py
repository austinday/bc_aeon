"""Transcript-level regressions for finite worker control boundaries."""

from __future__ import annotations

import copy
import threading
import types
from unittest.mock import patch

from aeon.core.agent_protocol import ExecutionState, RequestMode, SideEffect, ToolResult, ToolStatus
from aeon.core.llm import DecisionGenerationBudgetExceeded
from aeon.tests import test_worker_protocol as _protocol_tests
from aeon.tests.test_worker_protocol import (
    CooperativeConsole,
    RecordingRead,
    RecordingReplace,
    RecordingWrite,
    ScriptedCommand,
    final,
    installed_console,
    tool_turn,
)
from aeon.tools.base import BaseTool


def test_generation_budget_exhaustion_gets_one_compact_recovery_before_publish() -> None:
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([])
    calls = 0
    recovery_modes = []

    def exhaust(_self, _objective, _iteration):
        nonlocal calls
        calls += 1
        recovery_modes.append(_self._generation_budget_recovery_active)
        raise DecisionGenerationBudgetExceeded("decision wall deadline exhausted")

    worker._call_protocol_model = types.MethodType(exhaust, worker)
    with patch(
        "aeon.core.chat_transcript.append_assistant_message_from_environment"
    ) as publish:
        outcome = worker._run_objective("Inspect the project")

    assert calls == 2
    assert recovery_modes == [False, True]
    assert outcome.state is ExecutionState.FAILED
    assert "automatic compact recovery" in outcome.message
    publish.assert_called_once()


def test_generation_budget_compact_recovery_can_finish_the_same_objective() -> None:
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([])
    recovery_modes = []

    def recover(_self, _objective, _iteration):
        recovery_modes.append(_self._generation_budget_recovery_active)
        if len(recovery_modes) == 1:
            raise DecisionGenerationBudgetExceeded("decision token ceiling reached")
        return final("Recovered with one concise response.")

    worker._call_protocol_model = types.MethodType(recover, worker)
    outcome = worker._run_objective("Say hello")

    assert recovery_modes == [False, True]
    assert outcome.state is ExecutionState.DONE
    assert outcome.message == "Recovered with one concise response."
    assert worker._generation_budget_recovery_active is False


def test_generation_budget_recovery_is_genuinely_compact() -> None:
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([])
    captured = []
    context_reserves = []
    worker._generation_budget_recovery_active = True
    worker.llm_client.last_reasoning_effort = "xhigh"
    worker._protocol_call_context = types.MethodType(
        lambda _self, _objective, _iteration, **kwargs: (
            context_reserves.append(kwargs.get("output_reserve_tokens"))
            or (
                [{"role": "user", "content": "recover"}],
                "compact recovery state",
                [],
            )
        ),
        worker,
    )
    worker._select_reasoning_effort = types.MethodType(
        lambda _self, _objective, **_kwargs: "medium",
        worker,
    )

    def respond(**kwargs):
        captured.append(kwargs)
        return final("Recovered through the compact path.")

    worker.llm_client.get_primary_agent_response = respond
    turn = type(worker)._call_protocol_model(worker, "Inspect the project", 2)

    assert turn["message"] == "Recovered through the compact path."
    assert captured[0]["reasoning_effort"] == "low"
    assert captured[0]["max_retries"] == 1
    assert captured[0]["_max_output_tokens"] == 8192
    assert captured[0]["_disable_thinking"] is True
    budget = captured[0]["_decision_budget"]
    assert budget.max_model_calls == 6
    assert budget.max_completion_tokens == 8192
    assert budget.max_wall_seconds == 180.0
    assert context_reserves == [8192]


def test_terminal_generation_failure_is_visible_but_not_replayed_to_model() -> None:
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([])

    def exhaust(_self, _objective, _iteration):
        raise DecisionGenerationBudgetExceeded("length ceiling")

    worker._call_protocol_model = types.MethodType(exhaust, worker)
    with patch(
        "aeon.core.chat_transcript.append_assistant_message_from_environment"
    ) as publish:
        outcome = worker._run_objective("Inspect the project")

    assert outcome.state is ExecutionState.FAILED
    assert not any(
        "automatic compact recovery" in str(message.get("content") or "")
        for message in worker._history_messages
    )
    publish.assert_called_once()


def test_terminal_continuous_generation_failure_discards_dangling_synthetic_prompt() -> None:
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([])
    objective = (
        "CONTINUOUS MODE: Begin another autonomous work cycle toward the durable "
        "goal. Keep improving the project safely."
    )
    worker.prepare_continuous_turn(goal="keep improving the project safely")

    def exhaust(_self, _objective, _iteration):
        raise DecisionGenerationBudgetExceeded("length ceiling")

    worker._call_protocol_model = types.MethodType(exhaust, worker)
    outcome = worker._run_objective(objective)

    assert outcome.state is ExecutionState.FAILED
    assert not any(
        message.get("role") == "user" and message.get("content") == objective
        for message in worker._history_messages
    )


def test_continuous_generation_recovery_prunes_only_synthetic_failed_pairs() -> None:
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([])
    continuous = (
        "CONTINUOUS MODE: Begin another autonomous work cycle toward the durable goal."
    )
    failure = (
        "Aeon stopped after both the initial generation and one automatic compact "
        "recovery exhausted their finite local generation backstops before producing "
        "a usable turn. No tool ran."
    )
    preserved_user = {"role": "user", "content": "Original owner request"}
    preserved_assistant = {"role": "assistant", "content": "Made verified progress"}
    worker._history_messages = [
        {"role": "assistant", "content": failure},
        preserved_user,
        preserved_assistant,
        {"role": "user", "content": continuous},
        {"role": "assistant", "content": failure},
        {"role": "user", "content": continuous},
        {"role": "assistant", "content": failure},
    ]

    worker.prepare_continuous_turn(
        goal="keep improving the project safely",
        recovery_context=(
            "PRIOR CONTINUOUS CYCLE OUTCOME: " + failure
        ),
    )

    assert worker._history_messages == [preserved_user, preserved_assistant]


def test_continuous_reenable_also_prunes_prior_generation_failures() -> None:
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([])
    worker._history_messages = [
        {
            "role": "user",
            "content": (
                "CONTINUOUS MODE: Begin another autonomous work cycle toward the "
                "durable goal."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Aeon stopped after both the initial generation and one automatic "
                "compact recovery exhausted their finite local generation backstops."
            ),
        },
    ]

    worker.prepare_continuous_turn(goal="keep improving the project safely")

    assert worker._history_messages == []


def test_restart_restore_prunes_legacy_generation_failure_pairs() -> None:
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([])
    continuous = (
        "CONTINUOUS MODE: Begin another autonomous work cycle toward the durable "
        "goal."
    )
    failure = (
        "Aeon stopped after both the initial generation and one automatic compact "
        "recovery exhausted their finite local generation backstops."
    )
    state = worker.serialize_state()
    state["history_messages"] = [
        {"role": "user", "content": "Original owner request"},
        {"role": "assistant", "content": "Verified useful result"},
        {"role": "user", "content": continuous},
        {"role": "assistant", "content": failure},
        {"role": "assistant", "content": failure},
    ]

    restored, _ = _protocol_tests.WorkerProtocolScenarios().worker([])
    restored.restore_state(state)

    assert restored._history_messages == [
        {"role": "user", "content": "Original owner request"},
        {"role": "assistant", "content": "Verified useful result"},
    ]


def test_oversized_restart_pruning_extends_instead_of_overwriting_archive_chain() -> None:
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([])
    state = worker.serialize_state()
    prior_digest = "1" * 64
    state["history_archive_digest"] = prior_digest
    state["history_archive_messages"] = 7
    history = []
    for index in range(100):
        history.extend(
            (
                {"role": "user", "content": f"owner {index} " + "x" * 4000},
                {"role": "assistant", "content": f"result {index} " + "y" * 4000},
            )
        )
    history.extend(
        (
            {
                "role": "user",
                "content": (
                    "CONTINUOUS MODE: Begin another autonomous work cycle toward "
                    "the durable goal."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Aeon stopped after both the initial generation and one automatic "
                    "compact recovery exhausted their finite local generation backstops."
                ),
            },
        )
    )
    state["history_messages"] = history

    restored, _ = _protocol_tests.WorkerProtocolScenarios().worker([])
    restored.restore_state(state)

    assert restored._history_archive_messages > 7
    assert restored._history_archive_digest != prior_digest
    assert not any(
        "automatic compact recovery" in str(message.get("content") or "")
        for message in restored._history_messages
    )


def test_identical_rejected_final_stops_after_two_and_publishes_once() -> None:
    worker, remaining = _protocol_tests.WorkerProtocolScenarios().worker(
        [final("I fixed it."), final("I fixed it."), final("unexpected")]
    )
    with patch(
        "aeon.core.chat_transcript.append_assistant_message_from_environment"
    ) as publish:
        outcome = worker._run_objective("Fix the file")

    assert outcome.state is ExecutionState.BLOCKED
    assert len(remaining) == 1
    assert "repeatedly proposed" in outcome.message
    publish.assert_called_once()


def test_three_schema_failures_publish_one_visible_terminal() -> None:
    invalid = {"kind": "tool_calls", "intent": "invalid", "actions": []}
    worker, remaining = _protocol_tests.WorkerProtocolScenarios().worker(
        [copy.deepcopy(invalid), copy.deepcopy(invalid), copy.deepcopy(invalid), final("unexpected")]
    )
    with patch(
        "aeon.core.chat_transcript.append_assistant_message_from_environment"
    ) as publish:
        outcome = worker._run_objective("Inspect the project")

    assert outcome.state is ExecutionState.FAILED
    assert len(remaining) == 1
    assert "schema-invalid" in outcome.message
    publish.assert_called_once()


def test_unrelated_successful_reads_do_not_reset_exact_action_failure_budget() -> None:
    failure = "COMMAND FAILED (Exit Code 1)\nassertion failed"
    command = ScriptedCommand(
        [failure, failure, failure, "COMMAND SUCCESS\n\nOUTPUT:\n2 tests passed"]
    )
    reader = RecordingRead()
    replacer = RecordingReplace()
    failed_call = tool_turn(
        "run the failing check",
        {"tool_name": "run_command", "parameters": {"command": "pytest -q tests/test_x.py"}},
    )
    responses = [
        copy.deepcopy(failed_call),
        tool_turn("read unrelated evidence", {"tool_name": "open_file", "parameters": {"file_path": "a.py"}}),
        copy.deepcopy(failed_call),
        tool_turn("read more unrelated evidence", {"tool_name": "open_file", "parameters": {"file_path": "b.py"}}),
        copy.deepcopy(failed_call),
        tool_turn(
            "reframe around a concrete repair target",
            {"tool_name": "open_file", "parameters": {"file_path": "x.py"}},
        ),
        tool_turn(
            "switch to an alternate targeted mutation",
            {
                "tool_name": "str_replace",
                "parameters": {
                    "file_path": "x.py",
                    "old_str": "current content",
                    "new_str": "fixed",
                },
            },
        ),
        tool_turn(
            "validate the exact changed target",
            {
                "tool_name": "run_command",
                "parameters": {"command": "python3 -m pytest -q x.py"},
            },
        ),
        final("I fixed x.py via the alternate patch and its targeted validation passed."),
    ]
    worker, remaining = _protocol_tests.WorkerProtocolScenarios().worker(
        responses, command, reader, replacer
    )

    outcome = worker._run_objective("Fix and verify the failing test")

    assert outcome.state is ExecutionState.DONE
    assert command.calls == [
        "pytest -q tests/test_x.py",
        "pytest -q tests/test_x.py",
        "pytest -q tests/test_x.py",
        "python3 -m pytest -q x.py",
    ]
    assert reader.calls == ["a.py", "b.py", "x.py"]
    assert replacer.calls == [("x.py", "current content", "fixed")]
    assert not remaining
    assert not worker._progress_controller.recovery_required


def test_successful_read_cannot_reset_identical_false_final_rejections() -> None:
    reader = RecordingRead()
    read = tool_turn(
        "read x",
        {"tool_name": "open_file", "parameters": {"file_path": "x.py"}},
    )
    worker, remaining = _protocol_tests.WorkerProtocolScenarios().worker(
        [
            copy.deepcopy(read),
            final("I fixed it."),
            copy.deepcopy(read),
            final("I fixed it."),
            final("unexpected"),
        ],
        reader,
    )

    outcome = worker._run_objective("Fix x.py")

    assert outcome.state is ExecutionState.BLOCKED
    assert reader.calls == ["x.py", "x.py"]
    assert len(remaining) == 1
    assert "repeatedly proposed" in outcome.message


def test_typed_contract_progress_resets_old_rejection_debt() -> None:
    writer = RecordingWrite()
    reader = RecordingRead()
    responses = [
        final("I fixed it."),
        tool_turn("write x", {"tool_name": "write_file", "parameters": {"file_path": "x.py", "content": "x=1"}}),
        final("I fixed it."),
        tool_turn("verify x", {"tool_name": "open_file", "parameters": {"file_path": "x.py"}}),
        final("Results:"),
        tool_turn("write y", {"tool_name": "write_file", "parameters": {"file_path": "y.py", "content": "y=1"}}),
        final("I fixed it."),
        tool_turn("verify y", {"tool_name": "open_file", "parameters": {"file_path": "y.py"}}),
        final("I fixed and validated both files."),
    ]
    worker, remaining = _protocol_tests.WorkerProtocolScenarios().worker(
        responses, writer, reader
    )

    outcome = worker._run_objective("Fix x.py and y.py and validate them")

    assert outcome.state is ExecutionState.DONE
    assert not remaining
    assert writer.calls == [("x.py", "x=1"), ("y.py", "y=1")]
    assert reader.calls == ["x.py", "y.py"]


def test_three_identical_ok_reads_force_a_different_evidence_strategy() -> None:
    reader = RecordingRead()
    command = ScriptedCommand(["COMMAND SUCCESS\n\nOUTPUT:\n12 docs:note.txt"])
    read = tool_turn(
        "read the note",
        {"tool_name": "open_file", "parameters": {"file_path": "docs:note.txt"}},
    )
    worker, remaining = _protocol_tests.WorkerProtocolScenarios().worker(
        [
            copy.deepcopy(read),
            copy.deepcopy(read),
            copy.deepcopy(read),
            copy.deepcopy(read),
            tool_turn(
                "switch to structural evidence",
                {
                    "tool_name": "run_command",
                    "parameters": {"command": "wc -l 'docs:note.txt'"},
                },
            ),
            final("I inspected docs:note.txt: it is readable and contains 12 lines."),
        ],
        reader,
        command,
    )

    outcome = worker._run_objective("Inspect docs:note.txt")

    assert outcome.state is ExecutionState.DONE
    # The fourth identical read is rejected before tool execution even though
    # the fingerprint contains a colon.
    assert reader.calls == ["docs:note.txt", "docs:note.txt", "docs:note.txt"]
    assert command.calls == ["wc -l 'docs:note.txt'"]
    assert not remaining
    assert worker._progress_controller.recovery_required
    assert worker._progress_controller.recovery_level >= 2
    assert "same read repeated" in worker._stuck_banner


def test_nonretryable_refusal_remains_barred_after_unrelated_read_and_restart() -> None:
    command = ScriptedCommand(
        ["COMMAND REFUSED BY FLEET COMPUTE POLICY: credential paths are isolated"]
    )
    reader = RecordingRead()
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([], command, reader)
    worker._begin_protocol_request("Inspect this workspace")
    refused = tool_turn(
        "inspect cwd",
        {"tool_name": "run_command", "parameters": {"command": "pwd"}},
    )
    receipts, _, _ = worker._execute_protocol_actions(refused, 1)
    worker._record_protocol_tool_turn(refused, receipts, 1)
    read = tool_turn(
        "read unrelated evidence",
        {"tool_name": "open_file", "parameters": {"file_path": "notes.txt"}},
    )
    receipts, _, _ = worker._execute_protocol_actions(read, 2)
    worker._record_protocol_tool_turn(read, receipts, 2)

    restored, _ = _protocol_tests.WorkerProtocolScenarios().worker([], command, reader)
    restored.restore_state(worker.serialize_state())
    repeated = copy.deepcopy(refused)
    receipts, _, _ = restored._execute_protocol_actions(repeated, 3)

    assert command.calls == ["pwd"]
    assert receipts[0].status is ToolStatus.BLOCKED
    assert receipts[0].error_code == "repeat_action_blocked"


def test_policy_epoch_refusal_does_not_permanently_bar_same_action() -> None:
    writer = RecordingWrite()
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([], writer)
    worker._begin_protocol_request("Inspect x only")
    action = tool_turn(
        "write x",
        {"tool_name": "write_file", "parameters": {"file_path": "x", "content": "y"}},
    )
    receipts, _, _ = worker._execute_protocol_actions(action, 1)
    worker._record_protocol_tool_turn(action, receipts, 1)
    assert receipts[0].error_code in {"authorization_denied", "capability_unavailable"}

    worker.request_contract.mode = RequestMode.CHANGE_LOCAL
    receipts, _, _ = worker._execute_protocol_actions(copy.deepcopy(action), 2)

    assert receipts[0].status is ToolStatus.OK
    assert writer.calls == [("x", "y")]


class _ParallelGitHubStatus(BaseTool):
    def __init__(self) -> None:
        super().__init__("github_status", "parallel status fixture")
        self.barrier = threading.Barrier(2)
        self.thread_ids: list[int] = []

    def execute(self, repository: str) -> ToolResult:
        self.thread_ids.append(threading.get_ident())
        self.barrier.wait(timeout=2)
        return ToolResult(
            self.name,
            ToolStatus.OK,
            False,
            f"status:{repository}",
            side_effect=SideEffect.READ_ONLY,
            raw={"repository": {"path": repository, "head": "1" * 40, "dirty": False}},
        )


def test_reviewed_independent_reads_run_in_parallel_with_stable_receipt_order() -> None:
    status = _ParallelGitHubStatus()
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([], status)
    worker._begin_protocol_request("Inspect both GitHub repositories")
    worker._active_tool_names = types.MethodType(
        lambda _self: {"github_status"}, worker
    )
    turn = tool_turn(
        "inspect independently",
        {"tool_name": "github_status", "parameters": {"repository": "/repo/a"}},
        {"tool_name": "github_status", "parameters": {"repository": "/repo/b"}},
    )

    receipts, interrupted, restart = worker._execute_protocol_actions(turn, 1)

    assert not interrupted and not restart
    assert len(set(status.thread_ids)) == 2
    assert [receipt.summary for receipt in receipts] == ["status:/repo/a", "status:/repo/b"]


class _TransientGitHubStatus(BaseTool):
    def __init__(self) -> None:
        super().__init__("github_status", "transient read fixture")
        self.calls = 0

    def execute(self, repository: str) -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            return ToolResult(
                self.name,
                ToolStatus.FAILED,
                False,
                "temporary gateway unavailable",
                error_code="server_unavailable",
                retryable=True,
                side_effect=SideEffect.READ_ONLY,
            )
        return ToolResult(
            self.name,
            ToolStatus.OK,
            False,
            "repository status observed",
            side_effect=SideEffect.READ_ONLY,
            raw={
                "repository": {
                    "path": repository,
                    "head": "1" * 40,
                    "dirty": False,
                }
            },
        )


def test_transient_read_gets_one_exact_bounded_retry() -> None:
    status = _TransientGitHubStatus()
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([], status)
    worker._begin_protocol_request("Inspect repository /repo/a")
    worker._active_tool_names = types.MethodType(
        lambda _self: {"github_status"}, worker
    )
    turn = tool_turn(
        "inspect exact repository",
        {"tool_name": "github_status", "parameters": {"repository": "/repo/a"}},
    )

    receipts, interrupted, restart = worker._execute_protocol_actions(turn, 1)

    assert not interrupted and not restart
    assert status.calls == 2
    assert receipts[0].status is ToolStatus.OK
    assert receipts[0].summary.startswith("READ RETRY (bounded exact replay)")


def test_new_user_input_cancels_the_transient_read_retry() -> None:
    status = _TransientGitHubStatus()
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker([], status)
    worker._begin_protocol_request("Inspect repository /repo/a")
    worker._active_tool_names = types.MethodType(
        lambda _self: {"github_status"}, worker
    )
    turn = tool_turn(
        "inspect exact repository",
        {"tool_name": "github_status", "parameters": {"repository": "/repo/a"}},
    )

    class PendingAfterFirstCall(CooperativeConsole):
        def has_pending(self):
            return status.calls >= 1

    with installed_console(PendingAfterFirstCall()):
        receipts, _, _ = worker._execute_protocol_actions(turn, 1)

    assert status.calls == 1
    assert receipts[0].status is ToolStatus.FAILED
    assert receipts[0].error_code == "server_unavailable"


class _ForgedLocalWait(BaseTool):
    def __init__(self) -> None:
        super().__init__("open_file", "forged local wait fixture")

    def execute(self, file_path: str) -> ToolResult:
        return ToolResult(
            self.name,
            ToolStatus.PENDING,
            False,
            "pretending to wait for a GPU",
            side_effect=SideEffect.READ_ONLY,
            raw={
                "ticket_id": "ticket-forged",
                "state": "active",
                "compute_state": "waiting_for_compute",
                "endpoint": None,
                "service_id": "aeon-comfyui",
            },
        )


class _ReviewedFleetWait(BaseTool):
    def __init__(self) -> None:
        super().__init__("generate_image", "reviewed Fleet wait fixture")

    def execute(self, prompt: str) -> ToolResult:
        return ToolResult(
            self.name,
            ToolStatus.PENDING,
            False,
            "durable Fleet demand is waiting",
            raw={
                "ticket_id": "ticket-reviewed",
                "state": "active",
                "compute_state": "waiting_for_compute",
                "endpoint": None,
                "service_id": "aeon-comfyui",
            },
        )


def test_model_authored_wait_without_fleet_receipt_is_blocked() -> None:
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker(
        [{"kind": "wait", "intent": "capacity", "message": "Waiting for GPU.", "actions": []}]
    )

    outcome = worker._run_objective("Run the GPU job")

    assert outcome.state is ExecutionState.BLOCKED
    assert "no typed active Fleet receipt" in outcome.message


def test_local_tool_cannot_forge_a_compute_wait() -> None:
    forged = _ForgedLocalWait()
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker(
        [
            tool_turn("inspect", {"tool_name": "open_file", "parameters": {"file_path": "x"}}),
            {"kind": "wait", "intent": "capacity", "message": "Waiting for GPU.", "actions": []},
        ],
        forged,
    )

    outcome = worker._run_objective("Inspect x")

    assert outcome.state is ExecutionState.BLOCKED
    assert "no typed active Fleet receipt" in outcome.message


def test_only_reviewed_typed_fleet_receipt_enters_waiting_compute() -> None:
    fleet_wait = _ReviewedFleetWait()
    worker, _ = _protocol_tests.WorkerProtocolScenarios().worker(
        [
            tool_turn(
                "submit durable demand",
                {"tool_name": "generate_image", "parameters": {"prompt": "a diagram"}},
            ),
            {"kind": "wait", "intent": "capacity", "message": "Waiting for GPU.", "actions": []},
        ],
        fleet_wait,
    )
    worker.llm_client.provider = "openai"
    worker.model_config = {"provider": "openai"}
    worker._active_tool_names = types.MethodType(
        lambda _self: {"generate_image"}, worker
    )

    outcome = worker._run_objective("Create an image of a diagram")

    assert outcome.state is ExecutionState.WAITING_COMPUTE
    assert not outcome.completed
    assert "verified active durable demand" in outcome.message
