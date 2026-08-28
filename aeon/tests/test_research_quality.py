"""Hermetic evidence-gate regressions for continuous Hub scouting."""

from __future__ import annotations

import json
import types
from unittest.mock import patch

from aeon.core.agent_protocol import ExecutionState, SideEffect, ToolResult, ToolStatus
from aeon.core.continuous_mode import ContinuousModeState
from aeon.core.research_quality import ResearchQualityGuard
from aeon.core.worker import Worker
from aeon.tools.base import BaseTool


GOAL = (
    "Continuously scout Hugging Face model candidate opportunities and "
    "derivative gaps."
)
SHA = "a" * 40
UPLOAD_GOAL = (
    "Get eyes on a donation address through making useful uploads, while only "
    "working on projects feasible with the compute available."
)


def _result(name: str, call_id: str, payload: dict) -> ToolResult:
    return ToolResult(
        tool_name=name,
        status=ToolStatus.OK,
        changed=False,
        summary="fixture Hub receipt",
        side_effect=SideEffect.READ_ONLY,
        call_id=call_id,
        raw=json.dumps(payload, sort_keys=True),
    )


def _model_info(
    repo_id: str,
    *,
    sha: str,
    base_model: str | None = None,
    parameter_total: int | None = 3_000_000_000,
) -> dict:
    safetensors = {"total": parameter_total} if parameter_total is not None else {}
    tags = [f"base_model:{base_model}"] if base_model else []
    return {
        "metadata": {
            "id": repo_id,
            "sha": sha,
            "createdAt": "2026-08-20T01:02:03.000Z",
            "lastModified": "2026-08-28T01:02:03.000Z",
            "tags": tags,
        },
        "config": {"architectures": ["FixtureModel"]},
        "card_data": {"license": "apache-2.0", "base_model": base_model},
        "safetensors": safetensors,
    }


def _observe(guard: ResearchQualityGuard, calls: list[tuple[str, dict, dict]]) -> None:
    actions = []
    results = []
    for index, (name, parameters, payload) in enumerate(calls):
        call_id = f"call-{index}"
        actions.append(
            {
                "_call_id": call_id,
                "tool_name": name,
                "parameters": parameters,
            }
        )
        results.append(_result(name, call_id, payload))
    guard.observe_turn({"actions": actions}, results)


def _complete_receipts(guard: ResearchQualityGuard) -> None:
    _observe(
        guard,
        [
            (
                "huggingface_model_info",
                {"repo_id": "owner/model"},
                _model_info("owner/model", sha=SHA),
            ),
            (
                "huggingface_model_info",
                {"repo_id": "other/model-gguf"},
                _model_info(
                    "other/model-gguf", sha="b" * 40, base_model="owner/model"
                ),
            ),
            (
                "huggingface_repo_file",
                {
                    "repo_id": "owner/model",
                    "path": "LICENSE",
                    "revision": SHA,
                },
                {
                    "repo_commit": SHA,
                    "content": "Apache License Version 2.0 fixture text",
                    "truncated": False,
                },
            ),
            (
                "huggingface_model_search",
                {"query": "owner/model", "filter_tag": "gguf"},
                {
                    "result_count": 1,
                    "results": [
                        {
                            "id": "other/model-gguf",
                            "tags": ["base_model:owner/model"],
                        }
                    ],
                },
            ),
            (
                "huggingface_model_search",
                {"query": "owner/model derivative", "filter_tag": "fp8"},
                {"result_count": 0, "results": []},
            ),
        ],
    )


def _scoped_receipt_claim() -> str:
    return (
        "For owner/model, the exact Hub identity and creation/modified timestamps, "
        "architecture and parameter count metadata, license-tag and license-file "
        "retrieval, and bounded competition search sample are confirmed for this "
        "current receipt set."
    )


def test_provisional_and_explicitly_negated_reports_need_no_receipts() -> None:
    guard = ResearchQualityGuard()
    guard.begin_cycle(GOAL)

    assert guard.completion_error(
        "Owner/model remains a provisional lead; it is not validated or decision-ready."
    ) == ""


def test_unrelated_confirmed_and_covered_clauses_are_outside_candidate_gate() -> None:
    guard = ResearchQualityGuard()
    guard.begin_cycle(GOAL)

    for message in (
        "Credential access is confirmed.",
        "The targets.md SHA-256 checksum is validated.",
        "All requested files are covered by the backup.",
        "The regression test result is confirmed.",
    ):
        assert guard.completion_error(message) == ""

    assert "RESEARCH CLAIM BLOCKED" in guard.completion_error(
        "The video modality is closed."
    )


def test_useful_upload_campaign_is_guarded_even_when_goal_omits_hub_name() -> None:
    guard = ResearchQualityGuard()
    guard.begin_cycle(UPLOAD_GOAL)

    assert guard.active
    assert "RESEARCH CLAIM BLOCKED" in guard.completion_error(
        "Owner/model is a validated candidate."
    )
    assert guard.completion_error(
        "I cannot call owner/model a winner with the current evidence."
    ) == ""


def test_premature_candidate_promotion_is_blocked_without_current_receipts() -> None:
    guard = ResearchQualityGuard()
    guard.begin_cycle(GOAL)

    error = guard.completion_error("Owner/model is a validated candidate.")

    assert "RESEARCH CLAIM BLOCKED" in error
    assert "current-cycle" in error
    assert "provisional lead" in error


def test_scoped_hub_receipt_validation_requires_every_typed_gate() -> None:
    guard = ResearchQualityGuard()
    guard.begin_cycle(GOAL)
    _complete_receipts(guard)

    assert guard.completion_error(_scoped_receipt_claim()) == ""

    incomplete = ResearchQualityGuard()
    incomplete.begin_cycle(GOAL)
    _observe(
        incomplete,
        [
            (
                "huggingface_model_info",
                {"repo_id": "owner/model"},
                _model_info("owner/model", sha=SHA, parameter_total=None),
            )
        ],
    )
    error = incomplete.completion_error(_scoped_receipt_claim())
    assert "numeric safetensors parameter total" in error
    assert "same-revision license text" in error
    assert "competition searches" in error


def test_winner_and_absence_claims_remain_blocked_after_hub_receipts() -> None:
    guard = ResearchQualityGuard()
    guard.begin_cycle(GOAL)
    _complete_receipts(guard)

    winner_error = guard.completion_error(
        "Owner/model is the clear winner and decision-ready."
    )
    assert "hardware/toolchain feasibility" in winner_error
    assert "differentiated user value" in winner_error
    assert "reproducible benchmark" in winner_error

    gap_error = guard.completion_error(
        "The missing derivative gap for owner/model is confirmed."
    )
    assert "cannot confirm absence" in gap_error


def test_global_closure_is_never_inferred_and_finite_coverage_is_scoped() -> None:
    guard = ResearchQualityGuard()
    guard.begin_cycle(GOAL)
    _complete_receipts(guard)

    closed_error = guard.completion_error(
        "Within the sampled queries, the model opportunity lane is closed."
    )
    assert "cannot establish" in closed_error

    insufficient_error = guard.completion_error(
        "Within the four sampled Hub queries and two repositories inspected, "
        "this sample is covered."
    )
    assert "at least four distinct" in insufficient_error

    _observe(
        guard,
        [
            (
                "huggingface_model_search",
                {"query": "owner model", "filter_tag": "awq"},
                {"result_count": 0, "results": []},
            ),
            (
                "huggingface_model_search",
                {"author": "owner", "pipeline_tag": "text-generation"},
                {"result_count": 1, "results": [{"id": "owner/model"}]},
            ),
        ],
    )
    assert guard.completion_error(
        "Within the four sampled Hub queries and two repositories inspected, "
        "this sample is covered."
    ) == ""


def test_campaign_state_keeps_branches_but_never_prior_cycle_receipts() -> None:
    guard = ResearchQualityGuard()
    guard.begin_cycle(GOAL)
    _complete_receipts(guard)
    assert guard.completion_error(_scoped_receipt_claim()) == ""
    guard.completion_error("Owner/model is the clear winner.")

    restored = ResearchQualityGuard()
    restored.restore_state_dict(guard.to_state_dict())
    restored.begin_cycle(GOAL)

    summary = restored.campaign_summary()
    assert "strategy history only" in summary
    assert "exact model metadata: owner/model" in summary
    assert "demoted unsupported claim" in summary
    assert "current-cycle" in restored.completion_error(_scoped_receipt_claim())

    restored.begin_cycle(
        "Scout Hugging Face dataset compatibility opportunities instead."
    )
    assert restored.campaign_summary() == ""


class _ScriptedLLM:
    context_limit = 100_000
    last_reasoning_content = ""
    last_generation_performance = None

    def set_action_schema(self, schema) -> None:
        self.action_schema = schema

    def set_iteration(self, iteration: int) -> None:
        self.iteration = iteration


def _final(message: str) -> dict:
    return {"kind": "final", "intent": "report", "message": message, "actions": []}


def _tool_turn(*actions: dict) -> dict:
    return {
        "kind": "tool_calls",
        "intent": "collect current Hub receipts",
        "message": "",
        "actions": list(actions),
    }


class _InfoTool(BaseTool):
    def __init__(self) -> None:
        super().__init__("huggingface_model_info", "fixture model info", directives=[])

    def execute(self, repo_id: str) -> str:
        if repo_id == "owner/model":
            payload = _model_info(repo_id, sha=SHA)
        else:
            payload = _model_info(repo_id, sha="b" * 40, base_model="owner/model")
        return json.dumps(payload)


class _LicenseTool(BaseTool):
    def __init__(self) -> None:
        super().__init__("huggingface_repo_file", "fixture repo file", directives=[])

    def execute(self, repo_id: str, path: str, revision: str) -> str:
        assert (repo_id, path, revision) == ("owner/model", "LICENSE", SHA)
        return json.dumps(
            {
                "repo_commit": SHA,
                "content": "Apache License Version 2.0 fixture text",
                "truncated": False,
            }
        )


class _SearchTool(BaseTool):
    def __init__(self) -> None:
        super().__init__("huggingface_model_search", "fixture search", directives=[])

    def execute(self, query: str, filter_tag: str) -> str:
        results = (
            [{"id": "other/model-gguf", "tags": ["base_model:owner/model"]}]
            if filter_tag == "gguf"
            else []
        )
        return json.dumps({"result_count": len(results), "results": results})


def test_worker_rejects_premature_final_then_accepts_truthful_provisional() -> None:
    worker = Worker(_ScriptedLLM(), tools=[], print_func=lambda *_: None)
    worker.persist_session = False
    queue = [
        _final("Owner/model is a validated candidate and clear winner."),
        _final(
            "Owner/model is only a provisional lead; it is not validated or "
            "decision-ready because no current Hub receipts were collected."
        ),
    ]

    def scripted_call(_self, _objective, _iteration):
        return queue.pop(0)

    worker._call_protocol_model = types.MethodType(scripted_call, worker)
    with patch(
        "aeon.core.chat_transcript.append_assistant_message_from_environment"
    ):
        outcome = worker._run_objective(GOAL)

    assert outcome.state is ExecutionState.DONE
    assert "provisional lead" in outcome.message
    assert queue == []


def test_worker_records_typed_hub_tools_before_accepting_scoped_claim() -> None:
    worker = Worker(
        _ScriptedLLM(),
        tools=[_InfoTool(), _LicenseTool(), _SearchTool()],
        print_func=lambda *_: None,
    )
    worker.persist_session = False
    queue = [
        _tool_turn(
            {
                "tool_name": "huggingface_model_info",
                "parameters": {"repo_id": "owner/model"},
            },
            {
                "tool_name": "huggingface_model_info",
                "parameters": {"repo_id": "other/model-gguf"},
            },
            {
                "tool_name": "huggingface_repo_file",
                "parameters": {
                    "repo_id": "owner/model",
                    "path": "LICENSE",
                    "revision": SHA,
                },
            },
            {
                "tool_name": "huggingface_model_search",
                "parameters": {"query": "owner/model", "filter_tag": "gguf"},
            },
            {
                "tool_name": "huggingface_model_search",
                "parameters": {
                    "query": "owner/model derivative",
                    "filter_tag": "fp8",
                },
            },
        ),
        _final(_scoped_receipt_claim()),
    ]

    def scripted_call(_self, _objective, _iteration):
        return queue.pop(0)

    worker._call_protocol_model = types.MethodType(scripted_call, worker)
    with patch(
        "aeon.core.chat_transcript.append_assistant_message_from_environment"
    ):
        outcome = worker._run_objective(GOAL)

    assert outcome.state is ExecutionState.DONE
    assert outcome.message == _scoped_receipt_claim()
    assert queue == []


def test_worker_projects_branch_history_but_resets_receipts_each_continuous_cycle() -> None:
    worker = Worker(_ScriptedLLM(), tools=[], print_func=lambda *_: None)
    worker.persist_session = False
    prompt = ContinuousModeState(enabled=True, goal=GOAL).prompt()

    worker.prepare_continuous_turn(goal=GOAL)
    worker._begin_protocol_request(prompt)
    _complete_receipts(worker._research_quality_guard)
    assert worker._research_quality_guard.completion_error(_scoped_receipt_claim()) == ""

    worker.execution_state = ExecutionState.DONE
    worker.request_contract.state = ExecutionState.DONE
    worker.prepare_continuous_turn(goal=GOAL)
    worker._begin_protocol_request(prompt)

    assert "RESEARCH CAMPAIGN LEDGER" in worker.last_observation
    assert "does not satisfy this cycle's evidence gate" in worker.last_observation
    error = worker._research_quality_guard.completion_error(_scoped_receipt_claim())
    assert "current-cycle" in error
