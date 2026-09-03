"""Closed, deterministic behavioral simulator for two benchmark cases.

The simulator never contacts Fleet Compute, launches a process, sleeps, or
spawns an agent.  A case-scoped HMAC capability binds it to the executor's
private receipt inode.  That signed append-only stream is also the durable state
machine, so an OpenCode MCP restart can replay it without trusting model prose or
a model-writable state file.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from aeon.core.benchmark_receipt import (
    CAPABILITY_RECEIPT_KEY_ENV,
    CAPABILITY_RECEIPT_PATH_ENV,
    CAPABILITY_RECEIPT_SCHEMA_VERSION,
    SCENARIO_EFFECT_RECEIPT_TYPE,
    TRACE_CASE_ID_ENV,
    TRACE_NONCE_ENV,
    TRACE_REPETITION_ENV,
    TRACE_RUN_ID_ENV,
    ScenarioEffectReceipt,
    append_scenario_effect_receipt,
    tool_arguments_sha256,
)


SCENARIO_CAPABILITY_ENV = "AEON_BENCHMARK_SCENARIO_CAPABILITY"
SCENARIO_TOOL_NAME = "benchmark_workflow"
SUPPORTED_SCENARIO_CASES = frozenset(
    {"fleet.resilience", "parallel.orchestration"}
)
_RUN_ID_RE = re.compile(r"^run-[0-9a-f]{32}$")
_CASE_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,99}$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_CAPABILITY_BYTES = 4096

FLEET_OPERATIONS = (
    "submit_gpu_job",
    "perform_cpu_manifest",
    "check_job",
    "run_gpu_stage",
    "continue_manager",
    "resume_gpu_job",
    "finalize",
    # Plausible but inappropriate choices are intentional distractors.
    "poll_job",
    "restart_job",
    "reserve_warm_capacity",
    "submit_cpu_job",
)
PARALLEL_OPERATIONS = (
    "delegate",
    "principal_work",
    "integration_prep",
    "collect",
    "integrate",
    # Explicitly measurable idle/over-delegation distractors.
    "idle_wait",
    "delegate_principal",
)


class ScenarioInfrastructureError(RuntimeError):
    """The fixture or its authenticated evidence is not trustworthy."""


@dataclass(frozen=True)
class ScenarioCapability:
    schema_version: int
    run_id: str
    case_id: str
    repetition: int
    trace_nonce: str
    capability_nonce: str
    receipt_path: str
    receipt_device: int
    receipt_inode: int
    receipt_key: str

    @property
    def scenario(self) -> str:
        return "fleet" if self.case_id == "fleet.resilience" else "parallel"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _token(capability: ScenarioCapability, label: str) -> str:
    suffix = hashlib.sha256(
        f"{capability.capability_nonce}:{label}".encode("ascii")
    ).hexdigest()[:20]
    return f"{label}_{suffix}"


def mint_scenario_capability(
    *,
    run_id: str,
    case_id: str,
    repetition: int,
    trace_nonce: str,
    receipt_path: Path,
    receipt_device: int,
    receipt_inode: int,
    receipt_key: str,
) -> str | None:
    """Return a signed capability only for the two simulator-backed cases."""

    if case_id not in SUPPORTED_SCENARIO_CASES:
        return None
    payload = {
        "schema_version": CAPABILITY_RECEIPT_SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": case_id,
        "repetition": repetition,
        "trace_nonce": trace_nonce,
        "capability_nonce": secrets.token_hex(32),
        "receipt_path": str(receipt_path),
        "receipt_device": receipt_device,
        "receipt_inode": receipt_inode,
    }
    encoded = _canonical(payload)
    envelope = {
        "payload": payload,
        "hmac_sha256": hmac.new(
            bytes.fromhex(receipt_key), encoded, hashlib.sha256
        ).hexdigest(),
    }
    return _canonical(envelope).decode("ascii")


def decode_scenario_capability(
    raw: str,
    key: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> ScenarioCapability | None:
    """Validate a signed case capability and its fixed receipt-file identity."""

    if not raw or len(raw.encode("utf-8", errors="ignore")) > _MAX_CAPABILITY_BYTES:
        return None
    if _HEX_RE.fullmatch(key) is None:
        return None
    try:
        envelope = json.loads(raw)
        if not isinstance(envelope, dict) or set(envelope) != {
            "payload",
            "hmac_sha256",
        }:
            return None
        payload = envelope["payload"]
        supplied = envelope["hmac_sha256"]
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "run_id",
            "case_id",
            "repetition",
            "trace_nonce",
            "capability_nonce",
            "receipt_path",
            "receipt_device",
            "receipt_inode",
        }:
            return None
        expected = hmac.new(
            bytes.fromhex(key), _canonical(payload), hashlib.sha256
        ).hexdigest()
        if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
            return None
        run_id = payload["run_id"]
        case_id = payload["case_id"]
        repetition = payload["repetition"]
        trace_nonce = payload["trace_nonce"]
        capability_nonce = payload["capability_nonce"]
        receipt_path = payload["receipt_path"]
        receipt_device = payload["receipt_device"]
        receipt_inode = payload["receipt_inode"]
        if (
            payload["schema_version"] != CAPABILITY_RECEIPT_SCHEMA_VERSION
            or not isinstance(run_id, str)
            or _RUN_ID_RE.fullmatch(run_id) is None
            or case_id not in SUPPORTED_SCENARIO_CASES
            or not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or not 1 <= repetition <= 20
            or not isinstance(trace_nonce, str)
            or _HEX_RE.fullmatch(trace_nonce) is None
            or not isinstance(capability_nonce, str)
            or _HEX_RE.fullmatch(capability_nonce) is None
            or not isinstance(receipt_path, str)
            or not Path(receipt_path).is_absolute()
            or not isinstance(receipt_device, int)
            or isinstance(receipt_device, bool)
            or receipt_device < 0
            or not isinstance(receipt_inode, int)
            or isinstance(receipt_inode, bool)
            or receipt_inode <= 0
            or (
                environment is not None
                and (
                    environment.get(CAPABILITY_RECEIPT_PATH_ENV, "") != receipt_path
                    or environment.get(TRACE_RUN_ID_ENV, "") != run_id
                    or environment.get(TRACE_CASE_ID_ENV, "") != case_id
                    or environment.get(TRACE_REPETITION_ENV, "") != str(repetition)
                    or environment.get(TRACE_NONCE_ENV, "") != trace_nonce
                )
            )
        ):
            return None
        descriptor = os.open(
            receipt_path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_dev != receipt_device
                or metadata.st_ino != receipt_inode
            ):
                return None
        finally:
            os.close(descriptor)
    except (OSError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return None
    return ScenarioCapability(
        schema_version=CAPABILITY_RECEIPT_SCHEMA_VERSION,
        run_id=run_id,
        case_id=case_id,
        repetition=repetition,
        trace_nonce=trace_nonce,
        capability_nonce=capability_nonce,
        receipt_path=receipt_path,
        receipt_device=receipt_device,
        receipt_inode=receipt_inode,
        receipt_key=key,
    )


def load_scenario_capability() -> ScenarioCapability | None:
    """Load the exact environment capability without raising in normal runs."""

    return decode_scenario_capability(
        os.environ.get(SCENARIO_CAPABILITY_ENV, ""),
        os.environ.get(CAPABILITY_RECEIPT_KEY_ENV, ""),
        environment=os.environ,
    )


def _initial_state(capability: ScenarioCapability) -> dict[str, object]:
    common: dict[str, object] = {
        "scenario": capability.scenario,
        "clock": 0,
        "bad_attempts": 0,
        "phase": "ready",
    }
    if capability.scenario == "fleet":
        common.update(
            {
                "job_id": "",
                "cpu_artifact": "",
                "checkpoint_id": "",
                "gpu_artifact": "",
                "completion_code": "",
                "submissions": 0,
                "duplicate_submissions": 0,
                "polls": 0,
                "warm_attempts": 0,
                "restart_attempts": 0,
                "manager_continues": 0,
                "intervals": [],
            }
        )
    else:
        common.update(
            {
                "tasks": {},
                "collected": [],
                "principal_report": "",
                "prep_report": "",
                "completion_code": "",
                "duplicate_delegations": 0,
                "idle_ms": 0,
                "blocked_wait_ms": 0,
                "intervals": [],
            }
        )
    return common


def _response(code: str, state: Mapping[str, object], **values: object) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "ok" if code not in {
            "duplicate_submission",
            "cpu_work_mistimed",
            "duplicate_cpu_work",
            "wrong_job",
            "stage_wrong_state",
            "manager_wrong_state",
            "wrong_resume_reference",
            "resume_wrong_state",
            "incomplete_artifacts",
            "finalize_wrong_state",
            "warm_capacity_refused",
            "inappropriate_cpu_submission",
            "restart_refused",
            "duplicate_delegation",
            "dependency_not_ready",
            "wrong_dependency",
            "overspawn_refused",
            "duplicate_principal",
            "duplicate_prep",
            "duplicate_collect",
            "unknown_task",
            "integration_missing_reports",
            "principal_delegation_refused",
        } else "rejected",
        "effect": code,
        "workflow_state": str(state["phase"]),
        "virtual_time_ms": int(state["clock"]),
    }
    result.update(values)
    return result


def _append_interval(
    state: dict[str, object], kind: str, start: int, end: int
) -> None:
    intervals = list(state["intervals"])
    intervals.append({"kind": kind, "start": start, "end": end})
    state["intervals"] = intervals


def _fleet_code(
    capability: ScenarioCapability,
    state: Mapping[str, object],
    operation: str,
    references: tuple[str, ...],
    _branch: str,
) -> str:
    job = str(state["job_id"])
    phase = str(state["phase"])
    if operation == "submit_gpu_job":
        return "job_submitted" if not job else "duplicate_submission"
    if operation == "perform_cpu_manifest":
        if state["cpu_artifact"]:
            return "duplicate_cpu_work"
        return "cpu_manifest_complete" if phase == "waiting" else "cpu_work_mistimed"
    if operation == "check_job":
        if references != (job,) or not job:
            return "wrong_job"
        if phase == "waiting" and state["cpu_artifact"]:
            return "compute_allocated"
        return "still_waiting" if phase == "waiting" else "redundant_status"
    if operation == "run_gpu_stage":
        if phase != "allocated":
            return "stage_wrong_state"
        return "preempted_checkpointed" if references == (job,) else "wrong_job"
    if operation == "continue_manager":
        if phase != "preempted":
            return "manager_wrong_state"
        return "compute_reacquired" if references == (job,) else "wrong_job"
    if operation == "resume_gpu_job":
        if phase != "reacquired":
            return "resume_wrong_state"
        expected = (job, str(state["checkpoint_id"]))
        return "gpu_resumed" if references == expected else "wrong_resume_reference"
    if operation == "finalize":
        if phase != "resumed":
            return "finalize_wrong_state"
        expected = {str(state["cpu_artifact"]), str(state["gpu_artifact"])}
        return (
            "fleet_complete"
            if len(references) == 2 and set(references) == expected
            else "incomplete_artifacts"
        )
    return {
        "poll_job": "passive_poll",
        "restart_job": "restart_refused",
        "reserve_warm_capacity": "warm_capacity_refused",
        "submit_cpu_job": "inappropriate_cpu_submission",
    }.get(operation, "invalid_operation")


def _apply_fleet(
    capability: ScenarioCapability,
    state: Mapping[str, object],
    operation: str,
    code: str,
) -> tuple[dict[str, object], bool, dict[str, object], int, int]:
    new = deepcopy(dict(state))
    start = int(new["clock"])
    end = start
    accepted = True
    values: dict[str, object] = {}
    if operation == "submit_gpu_job" and code == "job_submitted" and not new["job_id"]:
        end += 1
        new["clock"] = end
        new["phase"] = "waiting"
        new["job_id"] = _token(capability, "job")
        new["submissions"] = 1
        values["job_id"] = new["job_id"]
    elif operation == "submit_gpu_job" and code == "duplicate_submission" and new["job_id"]:
        accepted = False
        new["duplicate_submissions"] = int(new["duplicate_submissions"]) + 1
    elif operation == "perform_cpu_manifest" and code == "cpu_manifest_complete" and new["phase"] == "waiting" and not new["cpu_artifact"]:
        end += 4
        new["clock"] = end
        new["cpu_artifact"] = _token(capability, "cpu_manifest")
        _append_interval(new, "cpu_manifest", start, end)
        values["artifact_id"] = new["cpu_artifact"]
    elif operation == "perform_cpu_manifest" and code in {"cpu_work_mistimed", "duplicate_cpu_work"}:
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    elif operation == "check_job" and code == "compute_allocated" and new["phase"] == "waiting" and new["cpu_artifact"]:
        new["phase"] = "allocated"
        values["job_id"] = new["job_id"]
    elif operation == "check_job" and code in {"still_waiting", "redundant_status"}:
        end += 1
        new["clock"] = end
        new["polls"] = int(new["polls"]) + 1
    elif operation in {"check_job", "run_gpu_stage", "continue_manager"} and code == "wrong_job":
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    elif operation == "run_gpu_stage" and code == "preempted_checkpointed" and new["phase"] == "allocated":
        end += 3
        new["clock"] = end
        new["phase"] = "preempted"
        new["checkpoint_id"] = _token(capability, "checkpoint")
        _append_interval(new, "gpu_before_preemption", start, end)
        values.update(
            {
                "job_id": new["job_id"],
                "checkpoint_id": new["checkpoint_id"],
                "preempted": True,
            }
        )
    elif operation == "run_gpu_stage" and code == "stage_wrong_state":
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    elif operation == "continue_manager" and code == "compute_reacquired" and new["phase"] == "preempted":
        end += 2
        new["clock"] = end
        new["phase"] = "reacquired"
        new["manager_continues"] = int(new["manager_continues"]) + 1
        _append_interval(new, "durable_reacquire", start, end)
        values["job_id"] = new["job_id"]
    elif operation == "continue_manager" and code == "manager_wrong_state":
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    elif operation == "resume_gpu_job" and code == "gpu_resumed" and new["phase"] == "reacquired":
        end += 4
        new["clock"] = end
        new["phase"] = "resumed"
        new["gpu_artifact"] = _token(capability, "gpu_result")
        _append_interval(new, "gpu_after_reacquire", start, end)
        values["artifact_id"] = new["gpu_artifact"]
    elif operation == "resume_gpu_job" and code in {"wrong_resume_reference", "resume_wrong_state"}:
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    elif operation == "finalize" and code == "fleet_complete" and new["phase"] == "resumed":
        new["phase"] = "complete"
        new["completion_code"] = _token(capability, "fleet_complete").upper()
        values["completion_code"] = new["completion_code"]
    elif operation == "finalize" and code in {"incomplete_artifacts", "finalize_wrong_state"}:
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    elif operation == "poll_job" and code == "passive_poll":
        end += 1
        new["clock"] = end
        new["polls"] = int(new["polls"]) + 1
    elif operation == "restart_job" and code == "restart_refused":
        accepted = False
        new["restart_attempts"] = int(new["restart_attempts"]) + 1
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    elif operation == "reserve_warm_capacity" and code == "warm_capacity_refused":
        accepted = False
        new["warm_attempts"] = int(new["warm_attempts"]) + 1
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    elif operation == "submit_cpu_job" and code == "inappropriate_cpu_submission":
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    else:
        raise ScenarioInfrastructureError("invalid Fleet simulator transition")
    return new, accepted, _response(code, new, **values), start, end


def _parallel_code(
    capability: ScenarioCapability,
    state: Mapping[str, object],
    operation: str,
    references: tuple[str, ...],
    branch: str,
) -> str:
    tasks = dict(state["tasks"])
    if operation == "delegate":
        if branch in tasks:
            return "duplicate_delegation"
        if branch in {"a", "b"}:
            return f"delegated_{branch}"
        if branch == "c":
            if "a" not in state["collected"]:
                return "dependency_not_ready"
            return (
                "delegated_c"
                if references == (_token(capability, "report_a"),)
                else "wrong_dependency"
            )
        return "overspawn_refused"
    if operation == "principal_work":
        return "duplicate_principal" if state["principal_report"] else "principal_complete"
    if operation == "integration_prep":
        return "duplicate_prep" if state["prep_report"] else "prep_complete"
    if operation == "collect":
        if len(references) != 1:
            return "unknown_task"
        lookup = {
            str(value["task_id"]): name for name, value in tasks.items()
        }
        selected = lookup.get(references[0])
        if selected is None:
            return "unknown_task"
        return "duplicate_collect" if selected in state["collected"] else f"collected_{selected}"
    if operation == "integrate":
        required = {
            _token(capability, "report_a"),
            _token(capability, "report_b"),
            _token(capability, "report_c"),
            _token(capability, "principal_report"),
            _token(capability, "prep_report"),
        }
        return (
            "parallel_complete"
            if set(references) == required
            and len(references) == len(required)
            and set(state["collected"]) == {"a", "b", "c"}
            and bool(state["principal_report"])
            and bool(state["prep_report"])
            else "integration_missing_reports"
        )
    return {
        "idle_wait": "idle_waited",
        "delegate_principal": "principal_delegation_refused",
    }.get(operation, "invalid_operation")


def _apply_parallel(
    capability: ScenarioCapability,
    state: Mapping[str, object],
    operation: str,
    code: str,
) -> tuple[dict[str, object], bool, dict[str, object], int, int]:
    new = deepcopy(dict(state))
    start = int(new["clock"])
    end = start
    accepted = True
    values: dict[str, object] = {}
    if operation == "delegate" and code in {"delegated_a", "delegated_b", "delegated_c"}:
        branch = code[-1]
        if branch in new["tasks"]:
            raise ScenarioInfrastructureError("duplicate accepted task")
        duration = {"a": 2, "b": 4, "c": 2}[branch]
        task = {
            "task_id": _token(capability, f"task_{branch}"),
            "report_id": _token(capability, f"report_{branch}"),
            "start": start,
            "end": start + duration,
        }
        tasks = dict(new["tasks"])
        tasks[branch] = task
        new["tasks"] = tasks
        _append_interval(new, f"child_{branch}", start, start + duration)
        values["task_id"] = task["task_id"]
    elif operation == "delegate" and code in {"duplicate_delegation", "dependency_not_ready", "wrong_dependency", "overspawn_refused"}:
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
        if code == "duplicate_delegation":
            new["duplicate_delegations"] = int(new["duplicate_delegations"]) + 1
    elif operation == "principal_work" and code == "principal_complete" and not new["principal_report"]:
        end += 2
        new["clock"] = end
        new["principal_report"] = _token(capability, "principal_report")
        _append_interval(new, "principal_work", start, end)
        values["report_id"] = new["principal_report"]
    elif operation == "principal_work" and code == "duplicate_principal":
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    elif operation == "integration_prep" and code == "prep_complete" and not new["prep_report"]:
        end += 2
        new["clock"] = end
        new["prep_report"] = _token(capability, "prep_report")
        _append_interval(new, "integration_prep", start, end)
        values["report_id"] = new["prep_report"]
    elif operation == "integration_prep" and code == "duplicate_prep":
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    elif operation == "collect" and code in {"collected_a", "collected_b", "collected_c"}:
        branch = code[-1]
        task = dict(dict(new["tasks"])[branch])
        end = max(start, int(task["end"]))
        new["clock"] = end
        new["blocked_wait_ms"] = int(new["blocked_wait_ms"]) + (end - start)
        collected = list(new["collected"])
        collected.append(branch)
        new["collected"] = collected
        values["report_id"] = task["report_id"]
    elif operation == "collect" and code in {"duplicate_collect", "unknown_task"}:
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    elif operation == "integrate" and code == "parallel_complete":
        new["phase"] = "complete"
        new["completion_code"] = _token(capability, "parallel_complete").upper()
        values["completion_code"] = new["completion_code"]
    elif operation == "integrate" and code == "integration_missing_reports":
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    elif operation == "idle_wait" and code == "idle_waited":
        end += 2
        new["clock"] = end
        new["idle_ms"] = int(new["idle_ms"]) + 2
    elif operation == "delegate_principal" and code == "principal_delegation_refused":
        accepted = False
        new["bad_attempts"] = int(new["bad_attempts"]) + 1
    else:
        raise ScenarioInfrastructureError("invalid parallel simulator transition")
    return new, accepted, _response(code, new, **values), start, end


def _apply_code(
    capability: ScenarioCapability,
    state: Mapping[str, object],
    operation: str,
    code: str,
) -> tuple[dict[str, object], bool, dict[str, object], int, int]:
    if operation in {"fixture_ready", "fixture_reopened"}:
        expected = operation
        if code != expected:
            raise ScenarioInfrastructureError("invalid fixture lifecycle event")
        current = deepcopy(dict(state))
        response = _response(code, current, workflow=capability.scenario)
        return current, True, response, int(current["clock"]), int(current["clock"])
    return (
        _apply_fleet(capability, state, operation, code)
        if capability.scenario == "fleet"
        else _apply_parallel(capability, state, operation, code)
    )


def replay_scenario_effects(
    capability: ScenarioCapability,
    effects: Sequence[ScenarioEffectReceipt],
    *,
    require_ready: bool = True,
) -> dict[str, object]:
    state = _initial_state(capability)
    saw_ready = False
    for expected_sequence, event in enumerate(effects, start=1):
        if (
            event.run_id != capability.run_id
            or event.case_id != capability.case_id
            or event.repetition != capability.repetition
            or event.trace_nonce != capability.trace_nonce
            or event.effect_sequence != expected_sequence
            or event.state_before_sha256 != _digest(state)
        ):
            raise ScenarioInfrastructureError("scenario evidence context or order is invalid")
        new, accepted, response, start, end = _apply_code(
            capability, state, event.operation, event.effect_code
        )
        if (
            event.accepted is not accepted
            or event.virtual_start_ms != start
            or event.virtual_end_ms != end
            or event.state_after_sha256 != _digest(new)
            or event.effect_sha256 != _digest(response)
        ):
            raise ScenarioInfrastructureError("scenario effect evidence is inconsistent")
        if event.operation == "fixture_ready":
            if saw_ready:
                raise ScenarioInfrastructureError("scenario fixture initialized twice")
            saw_ready = True
        elif event.operation == "fixture_reopened" and not saw_ready:
            raise ScenarioInfrastructureError("scenario fixture reopened before initialization")
        state = new
    if require_ready and not saw_ready:
        raise ScenarioInfrastructureError("scenario fixture did not attest readiness")
    return state


class ScenarioSession:
    """One capability-bound facade used by the benchmark-only tool."""

    def __init__(self, capability: ScenarioCapability) -> None:
        self.capability = capability
        if not self._record_lifecycle():
            raise ScenarioInfrastructureError("scenario readiness could not be recorded")

    def _record_lifecycle(self) -> bool:
        capability = self.capability

        def build(effects: tuple[ScenarioEffectReceipt, ...]) -> ScenarioEffectReceipt:
            state = replay_scenario_effects(
                capability, effects, require_ready=bool(effects)
            )
            operation = "fixture_reopened" if effects else "fixture_ready"
            new, accepted, response, start, end = _apply_code(
                capability, state, operation, operation
            )
            return ScenarioEffectReceipt(
                schema_version=CAPABILITY_RECEIPT_SCHEMA_VERSION,
                receipt_type=SCENARIO_EFFECT_RECEIPT_TYPE,
                run_id=capability.run_id,
                case_id=capability.case_id,
                repetition=capability.repetition,
                trace_nonce=capability.trace_nonce,
                effect_sequence=len(effects) + 1,
                operation=operation,
                accepted=accepted,
                effect_code=operation,
                arguments_sha256=tool_arguments_sha256({}),
                state_before_sha256=_digest(state),
                state_after_sha256=_digest(new),
                effect_sha256=_digest(response),
                virtual_start_ms=start,
                virtual_end_ms=end,
            )

        return append_scenario_effect_receipt(
            build,
            expected_device=capability.receipt_device,
            expected_inode=capability.receipt_inode,
        ) is not None

    def execute(
        self,
        operation: str,
        reference_ids: Sequence[str] | None = None,
        branch: str = "",
    ) -> str:
        capability = self.capability
        references = tuple(str(item) for item in (reference_ids or ()))
        arguments = {
            "operation": operation,
            "reference_ids": list(references),
            "branch": branch,
        }
        response_box: list[dict[str, object]] = []

        def build(effects: tuple[ScenarioEffectReceipt, ...]) -> ScenarioEffectReceipt:
            state = replay_scenario_effects(capability, effects)
            allowed = FLEET_OPERATIONS if capability.scenario == "fleet" else PARALLEL_OPERATIONS
            if operation not in allowed:
                raise ScenarioInfrastructureError("operation is outside this fixture")
            code = (
                _fleet_code(capability, state, operation, references, branch)
                if capability.scenario == "fleet"
                else _parallel_code(capability, state, operation, references, branch)
            )
            new, accepted, response, start, end = _apply_code(
                capability, state, operation, code
            )
            response_box.append(response)
            return ScenarioEffectReceipt(
                schema_version=CAPABILITY_RECEIPT_SCHEMA_VERSION,
                receipt_type=SCENARIO_EFFECT_RECEIPT_TYPE,
                run_id=capability.run_id,
                case_id=capability.case_id,
                repetition=capability.repetition,
                trace_nonce=capability.trace_nonce,
                effect_sequence=len(effects) + 1,
                operation=operation,
                accepted=accepted,
                effect_code=code,
                arguments_sha256=tool_arguments_sha256(arguments),
                state_before_sha256=_digest(state),
                state_after_sha256=_digest(new),
                effect_sha256=_digest(response),
                virtual_start_ms=start,
                virtual_end_ms=end,
            )

        receipt = append_scenario_effect_receipt(
            build,
            expected_device=capability.receipt_device,
            expected_inode=capability.receipt_inode,
        )
        if receipt is None or len(response_box) != 1:
            raise ScenarioInfrastructureError("scenario effect could not be recorded")
        return _canonical(response_box[0]).decode("ascii")


def _max_concurrency(intervals: Sequence[Mapping[str, object]]) -> int:
    points: list[tuple[int, int]] = []
    for interval in intervals:
        kind = str(interval["kind"])
        if not kind.startswith("child_"):
            continue
        points.append((int(interval["start"]), 1))
        points.append((int(interval["end"]), -1))
    # End before start at an equal timestamp: touching intervals do not overlap.
    active = maximum = 0
    for _time, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _overlap_ratio(intervals: Sequence[Mapping[str, object]]) -> float:
    principal = [
        item
        for item in intervals
        if item["kind"] in {"principal_work", "integration_prep"}
    ]
    if len(principal) != 2:
        return 0.0
    useful = total = 0
    for own in principal:
        start, end = int(own["start"]), int(own["end"])
        total += end - start
        covered: set[int] = set()
        for item in intervals:
            if not str(item["kind"]).startswith("child_"):
                continue
            for tick in range(
                max(start, int(item["start"])), min(end, int(item["end"]))
            ):
                covered.add(tick)
        useful += len(covered)
    return useful / max(1, total)


def score_scenario_effects(
    capability: ScenarioCapability,
    effects: Sequence[ScenarioEffectReceipt],
) -> dict[str, object]:
    """Replay trusted effects and derive behavioral metrics, never tool prose."""

    state = replay_scenario_effects(capability, effects)
    operations = [
        event for event in effects if not event.operation.startswith("fixture_")
    ]
    if capability.scenario == "fleet":
        intervals = list(state["intervals"])
        cpu = next((item for item in intervals if item["kind"] == "cpu_manifest"), None)
        submitted = next(
            (item for item in operations if item.effect_code == "job_submitted"), None
        )
        allocated = next(
            (item for item in operations if item.effect_code == "compute_allocated"), None
        )
        preempted = next(
            (item for item in operations if item.effect_code == "preempted_checkpointed"), None
        )
        reacquired = next(
            (item for item in operations if item.effect_code == "compute_reacquired"), None
        )
        resumed = next(
            (item for item in operations if item.effect_code == "gpu_resumed"), None
        )
        clean_submission = (
            int(state["submissions"]) == 1
            and int(state["duplicate_submissions"]) == 0
            and int(state["warm_attempts"]) == 0
        )
        useful_wait = bool(
            cpu is not None
            and submitted is not None
            and allocated is not None
            and int(cpu["start"]) >= submitted.virtual_end_ms
            and int(cpu["end"]) <= allocated.virtual_start_ms
            and int(state["polls"]) == 0
        )
        recovered = bool(
            preempted is not None
            and reacquired is not None
            and resumed is not None
            and preempted.effect_sequence < reacquired.effect_sequence < resumed.effect_sequence
            and int(state["restart_attempts"]) == 0
        )
        checkpoint = recovered and bool(state["checkpoint_id"] and state["gpu_artifact"])
        durable = (
            state["phase"] == "complete"
            and int(state["manager_continues"]) == 1
            and int(state["bad_attempts"]) == 0
        )
        dimensions = (clean_submission, useful_wait, recovered, checkpoint, durable)
        score = sum(bool(value) for value in dimensions) / len(dimensions)
        return {
            "status": "passed" if score == 1.0 else "failed",
            "score": score,
            "tool_success": score == 1.0,
            "fleet_compute_judgment_score": 100.0 * clean_submission,
            "preemption_recovery_score": 100.0 * recovered,
            "useful_wait_work_score": 100.0 * useful_wait,
            "checkpoint_reacquire_score": 100.0 * checkpoint,
            "duplicate_submission_count": int(state["duplicate_submissions"]),
            "completion_code": str(state["completion_code"]),
        }

    intervals = list(state["intervals"])
    maximum = _max_concurrency(intervals)
    overlap = _overlap_ratio(intervals)
    tasks = dict(state["tasks"])
    dependency_ok = bool(
        set(tasks) == {"a", "b", "c"}
        and int(tasks["c"]["start"]) >= int(tasks["a"]["end"])
    )
    integration = state["phase"] == "complete"
    clean = (
        int(state["bad_attempts"]) == 0
        and int(state["duplicate_delegations"]) == 0
        and int(state["idle_ms"]) == 0
        and int(state["blocked_wait_ms"]) == 0
    )
    # The deterministic optimal schedule finishes at virtual t=4. Slower valid
    # schedules retain partial credit but cannot claim the critical-path point.
    critical_path = int(state["clock"]) <= 4 and integration
    dimensions = (
        maximum >= 2,
        overlap >= 0.99,
        dependency_ok,
        integration,
        clean and critical_path,
    )
    score = sum(bool(value) for value in dimensions) / len(dimensions)
    return {
        "status": "passed" if score == 1.0 else "failed",
        "score": score,
        "tool_success": score == 1.0,
        "useful_overlap_ratio": overlap,
        "idle_wait_ratio": (
            int(state["idle_ms"]) + int(state["blocked_wait_ms"])
        ) / max(1, int(state["clock"])),
        "max_parallelism": maximum,
        "integration_score": 100.0 * (dependency_ok and integration),
        "completion_code": str(state["completion_code"]),
    }


def simulator_preflight(case_id: str) -> bool:
    """Replay the optimal hidden workflow without files, time, or live services."""

    if case_id not in SUPPORTED_SCENARIO_CASES:
        return False
    capability = ScenarioCapability(
        schema_version=CAPABILITY_RECEIPT_SCHEMA_VERSION,
        run_id="run-" + "0" * 32,
        case_id=case_id,
        repetition=1,
        trace_nonce="1" * 64,
        capability_nonce="2" * 64,
        receipt_path="/nonexistent-benchmark-preflight",
        receipt_device=0,
        receipt_inode=1,
        receipt_key="3" * 64,
    )
    effects: list[ScenarioEffectReceipt] = []
    state = _initial_state(capability)

    def step(
        operation: str,
        references: Sequence[str] = (),
        branch: str = "",
    ) -> dict[str, object]:
        nonlocal state
        reference_tuple = tuple(references)
        if operation == "fixture_ready":
            code = operation
            arguments: dict[str, object] = {}
        else:
            code = (
                _fleet_code(
                    capability, state, operation, reference_tuple, branch
                )
                if capability.scenario == "fleet"
                else _parallel_code(
                    capability, state, operation, reference_tuple, branch
                )
            )
            arguments = {
                "operation": operation,
                "reference_ids": list(reference_tuple),
                "branch": branch,
            }
        new, accepted, response, start, end = _apply_code(
            capability, state, operation, code
        )
        effects.append(
            ScenarioEffectReceipt(
                schema_version=CAPABILITY_RECEIPT_SCHEMA_VERSION,
                receipt_type=SCENARIO_EFFECT_RECEIPT_TYPE,
                run_id=capability.run_id,
                case_id=capability.case_id,
                repetition=capability.repetition,
                trace_nonce=capability.trace_nonce,
                effect_sequence=len(effects) + 1,
                operation=operation,
                accepted=accepted,
                effect_code=code,
                arguments_sha256=tool_arguments_sha256(arguments),
                state_before_sha256=_digest(state),
                state_after_sha256=_digest(new),
                effect_sha256=_digest(response),
                virtual_start_ms=start,
                virtual_end_ms=end,
            )
        )
        state = replay_scenario_effects(capability, effects)
        return response

    try:
        step("fixture_ready")
        if capability.scenario == "fleet":
            job = str(step("submit_gpu_job")["job_id"])
            cpu = str(step("perform_cpu_manifest")["artifact_id"])
            step("check_job", (job,))
            checkpoint = str(step("run_gpu_stage", (job,))["checkpoint_id"])
            step("continue_manager", (job,))
            gpu = str(
                step("resume_gpu_job", (job, checkpoint))["artifact_id"]
            )
            completion = step("finalize", (cpu, gpu))["completion_code"]
        else:
            task_a = str(step("delegate", branch="a")["task_id"])
            task_b = str(step("delegate", branch="b")["task_id"])
            principal = str(step("principal_work")["report_id"])
            report_a = str(step("collect", (task_a,))["report_id"])
            task_c = str(
                step("delegate", (report_a,), branch="c")["task_id"]
            )
            prep = str(step("integration_prep")["report_id"])
            report_b = str(step("collect", (task_b,))["report_id"])
            report_c = str(step("collect", (task_c,))["report_id"])
            completion = step(
                "integrate",
                (report_a, report_b, report_c, principal, prep),
            )["completion_code"]
        scored = score_scenario_effects(capability, effects)
        return bool(
            completion
            and scored.get("status") == "passed"
            and scored.get("score") == 1.0
        )
    except (KeyError, ScenarioInfrastructureError, TypeError, ValueError):
        return False


__all__ = (
    "FLEET_OPERATIONS",
    "PARALLEL_OPERATIONS",
    "SCENARIO_CAPABILITY_ENV",
    "SCENARIO_TOOL_NAME",
    "SUPPORTED_SCENARIO_CASES",
    "ScenarioCapability",
    "ScenarioInfrastructureError",
    "ScenarioSession",
    "decode_scenario_capability",
    "load_scenario_capability",
    "mint_scenario_capability",
    "replay_scenario_effects",
    "score_scenario_effects",
    "simulator_preflight",
)
