"""Typed, recipe-bound access to durable Fleet Compute batch work.

This module intentionally does not expose Fleet's generic profile/payload API to
the model.  Each callable recipe binds one already-reviewed profile, project,
and closed payload.  Adding a new recipe is therefore a release review, not a
prompt or agent decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from aeon.core.benchmark_receipt import (
    emit_fleet_wait_capability_receipt,
)
from aeon.core.agent_protocol import SideEffect, ToolResult, ToolStatus
from aeon.core.fleet_backend import (
    FleetBackendError,
    FleetBrokerClient,
    FleetBrokerUnavailable,
)
from aeon.core.utils.io import read_bounded_fd, write_all_fd
from aeon.tools.base import BaseTool

_JOB_ID = re.compile(r"^fj-[0-9a-f]{32}$")
_REQUEST_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_ACTIVE_STATES = frozenset(
    {
        "queued",
        "waiting_for_compute",
        "starting",
        "running",
        "settling_output",
        "cleanup_pending",
    }
)
_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})


@dataclass(frozen=True)
class _BatchRecipe:
    recipe_id: str
    profile_id: str
    project: str
    purpose: str
    payload: Mapping[str, Any]
    required_goal_terms: tuple[str, ...]

    def authorized_for(self, objective: str) -> bool:
        value = " ".join(str(objective or "").casefold().split())
        return all(term in value for term in self.required_goal_terms)


# These are deliberately exact internal release recipes.  They prove the typed
# path is usable without turning it into a generic GPU shell.  In particular,
# neither recipe is a general Hugging Face model builder.
_RECIPES = {
    "qwen38-dflash-adapt-v1": _BatchRecipe(
        recipe_id="qwen38-dflash-adapt-v1",
        profile_id="aeon-qwen38-dflash-adapt",
        project="aeon-dflash-adapt",
        purpose="Exact Qwen3.8 DFlash2 adapt-v1 training release",
        payload={"run_mode": "adapt-v1"},
        required_goal_terms=("qwen3.8", "dflash"),
    ),
    "qwen38-full-gdn-nvfp4-v1": _BatchRecipe(
        recipe_id="qwen38-full-gdn-nvfp4-v1",
        profile_id="aeon-qwen38-full-gdn-quant",
        project="aeon-qwen38-full-gdn-quant",
        purpose="Exact Qwen3.8 full-GDN ModelOpt NVFP4 conversion",
        payload={"recipe": "full-gdn-max-v1"},
        required_goal_terms=("qwen3.8", "full-gdn"),
    ),
}


class FleetBatchToolError(RuntimeError):
    pass


def _client() -> FleetBrokerClient:
    return FleetBrokerClient(timeout=15)


def _objective(worker: Any) -> str:
    return str(getattr(worker, "current_objective", "") or "")


def _agent_id(worker: Any) -> str:
    value = str(getattr(worker, "instance_id", "") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise FleetBatchToolError("the managed agent identity is unavailable")
    return value


def _ledger_root(worker: Any) -> Path:
    resolver = getattr(worker, "_instance_state_dir", None)
    if not callable(resolver):
        raise FleetBatchToolError("the private agent state directory is unavailable")
    parent = Path(resolver())
    root = parent / "fleet-batch-jobs"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise FleetBatchToolError("the Fleet job ledger is not owner-private")
    return root


def _ledger_path(worker: Any, job_id: str) -> Path:
    if not _JOB_ID.fullmatch(str(job_id or "")):
        raise FleetBatchToolError("the Fleet batch job ID is invalid")
    return _ledger_root(worker) / f"{job_id}.json"


def _write_ledger(worker: Any, document: Mapping[str, Any]) -> None:
    job_id = str(document.get("job_id") or "")
    destination = _ledger_path(worker, job_id)
    payload = (
        json.dumps(dict(document), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    temporary = destination.parent / f".{job_id}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        write_all_fd(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)


def _read_ledger(worker: Any, job_id: str) -> dict[str, Any]:
    path = _ledger_path(worker, job_id)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise FleetBatchToolError(
            "that Fleet job is not owned by this managed agent"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
            or metadata.st_size < 2
            or metadata.st_size > 16_384
        ):
            raise FleetBatchToolError("the Fleet job ownership receipt is unsafe")
        raw = read_bounded_fd(descriptor, 16_384)
    finally:
        os.close(descriptor)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FleetBatchToolError("the Fleet job ownership receipt is invalid") from exc
    if (
        not isinstance(document, dict)
        or document.get("job_id") != job_id
        or document.get("agent_id") != _agent_id(worker)
        or document.get("recipe_id") not in _RECIPES
    ):
        raise FleetBatchToolError("the Fleet job ownership receipt does not match")
    return document


def _profile_registry(status: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    profiles = status.get("profiles")
    if not isinstance(profiles, list):
        raise FleetBatchToolError("Fleet omitted its reviewed profile registry")
    result: dict[str, Mapping[str, Any]] = {}
    for item in profiles:
        if not isinstance(item, Mapping):
            raise FleetBatchToolError("Fleet returned a malformed profile registry")
        profile_id = item.get("profile_id")
        if not isinstance(profile_id, str) or profile_id in result:
            raise FleetBatchToolError("Fleet returned an ambiguous profile registry")
        result[profile_id] = item
    return result


def _validate_recipe_profile(
    recipe: _BatchRecipe, profiles: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    profile = profiles.get(recipe.profile_id)
    if profile is None:
        raise FleetBatchToolError("the recipe's reviewed Fleet profile is unavailable")
    if (
        profile.get("enabled") is not True
        or profile.get("mode") != "batch"
        or profile.get("project") != recipe.project
    ):
        raise FleetBatchToolError("the recipe's reviewed Fleet profile is not enabled")
    return profile


def _validate_job_receipt(
    raw: Mapping[str, Any], recipe: _BatchRecipe, *, job_id: str | None = None
) -> dict[str, Any]:
    expected_id = job_id or raw.get("job_id")
    if not isinstance(expected_id, str) or not _JOB_ID.fullmatch(expected_id):
        raise FleetBatchToolError("Fleet returned an invalid batch job identity")
    state = raw.get("state")
    if (
        raw.get("job_id") != expected_id
        or raw.get("profile_id") != recipe.profile_id
        or raw.get("project") != recipe.project
        or raw.get("demand_class") != "standard"
        or state not in _ACTIVE_STATES | _TERMINAL_STATES
    ):
        raise FleetBatchToolError("Fleet returned a mismatched batch job receipt")
    return {
        "job_id": expected_id,
        "recipe_id": recipe.recipe_id,
        "profile_id": recipe.profile_id,
        "project": recipe.project,
        "state": state,
        "attempts": raw.get("attempts"),
        "runtime_state": raw.get("runtime_state"),
        "wait_reason": raw.get("wait_reason"),
        "retry_at": raw.get("retry_at"),
        "result": raw.get("result"),
        "last_error": raw.get("last_error"),
        "owned_by_agent": True,
    }


def _result(tool: BaseTool, receipt: Mapping[str, Any]) -> ToolResult:
    state = str(receipt.get("state") or "")
    if state in _ACTIVE_STATES:
        status = ToolStatus.PENDING
        error_code = ""
    elif state == "succeeded":
        status = ToolStatus.OK
        error_code = ""
    elif state == "cancelled":
        status = ToolStatus.BLOCKED
        error_code = "fleet_job_cancelled"
    else:
        status = ToolStatus.FAILED
        error_code = "fleet_job_failed"
    return ToolResult(
        tool_name=tool.name,
        status=status,
        changed=tool.name == "fleet_submit_batch_job",
        summary=json.dumps(dict(receipt), ensure_ascii=False, indent=2)[:12_000],
        evidence=[
            f"job_id={receipt.get('job_id')}",
            f"state={state}",
            f"profile_id={receipt.get('profile_id')}",
        ],
        artifacts=[str(receipt.get("job_id"))],
        error_code=error_code,
        retryable=False,
        side_effect=tool.policy.side_effect,
        raw=dict(receipt),
    )


def _failure(tool: BaseTool, error: Exception) -> ToolResult:
    temporary = isinstance(error, FleetBrokerUnavailable)
    blocked = isinstance(error, FleetBatchToolError)
    return ToolResult(
        tool_name=tool.name,
        status=ToolStatus.BLOCKED if blocked else ToolStatus.FAILED,
        changed=False,
        summary=str(error)[:1600],
        error_code=(
            "fleet_batch_capability_unavailable"
            if blocked
            else "fleet_broker_unavailable" if temporary else "fleet_batch_failed"
        ),
        retryable=temporary,
        side_effect=tool.policy.side_effect,
    )


class FleetBatchCapabilitiesTool(BaseTool):
    def __init__(self, worker=None) -> None:
        super().__init__(
            name="fleet_batch_capabilities",
            description=(
                "List the exact reviewed Fleet Compute batch recipes eligible for the "
                "current owner goal. Use this before auditing local GPU/toolchain access. "
                "It is the authoritative answer to whether this agent can submit a GPU "
                "build; AGENTS.md policy is not an executable capability. An empty recipe "
                "list means there is no general model-build lane: do not probe GPUs, SSH, "
                "Docker, the broker socket, or run_command for another route. Continue with "
                "a genuinely CPU-safe contribution or report the single stable limitation."
            ),
        )
        self.worker = worker

    def execute(self) -> ToolResult:
        try:
            profiles = _profile_registry(_client().status())
            eligible = []
            for recipe in _RECIPES.values():
                if not recipe.authorized_for(_objective(self.worker)):
                    continue
                try:
                    _validate_recipe_profile(recipe, profiles)
                except FleetBatchToolError:
                    continue
                eligible.append(
                    {
                        "recipe_id": recipe.recipe_id,
                        "profile_id": recipe.profile_id,
                        "purpose": recipe.purpose,
                        "durable_wait": True,
                    }
                )
            document = {
                "status": "ok",
                "recipes": eligible,
                "general_model_build_available": False,
                "submission_boundary": "reviewed_recipe_only",
                "unavailable_compute_is_durable_wait": True,
                "guidance": (
                    "Submit only a listed recipe. No listed recipe means this goal has "
                    "no reviewed GPU batch executor; policy text cannot create one."
                ),
            }
            # Benchmark scoring consumes this authenticated typed receipt from
            # a private side channel.  Model prose and echoed prompt markers do
            # not prove that this tool returned the reviewed boundary.
            emit_fleet_wait_capability_receipt(document)
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.OK,
                changed=False,
                summary=json.dumps(document, ensure_ascii=False, indent=2),
                evidence=[f"eligible_recipes={len(eligible)}"],
                side_effect=SideEffect.READ_ONLY,
                raw=document,
            )
        except (FleetBackendError, FleetBatchToolError) as exc:
            return _failure(self, exc)


class FleetSubmitBatchJobTool(BaseTool):
    def __init__(self, worker=None) -> None:
        super().__init__(
            name="fleet_submit_batch_job",
            description=(
                "Submit one exact recipe returned by fleet_batch_capabilities as an "
                "ordinary durable Fleet job. The tool chooses the reviewed profile, "
                "project, and closed payload; you cannot choose a host, GPU, command, or "
                "arbitrary payload. Reuse request_key only for retries of the same logical "
                "build. A pending receipt is real durable work: yield with kind=wait instead "
                "of polling, resubmitting, or repeating preflight."
            ),
        )
        self.worker = worker

    def execute(self, recipe_id: str, request_key: str) -> ToolResult:
        try:
            recipe = _RECIPES.get(str(recipe_id or ""))
            if recipe is None or not recipe.authorized_for(_objective(self.worker)):
                raise FleetBatchToolError(
                    "that reviewed Fleet recipe is not eligible for the current owner goal"
                )
            if not _REQUEST_KEY.fullmatch(str(request_key or "")):
                raise FleetBatchToolError(
                    "request_key must be 1-80 safe letters, numbers, dots, dashes, or underscores"
                )
            _validate_recipe_profile(recipe, _profile_registry(_client().status()))
            agent_id = _agent_id(self.worker)
            digest = hashlib.sha256(
                f"{agent_id}\0{recipe.recipe_id}\0{request_key}".encode("utf-8")
            ).hexdigest()
            idempotency_key = f"aeon-{digest}"
            raw = _client().submit_job(
                profile=recipe.profile_id,
                project=recipe.project,
                idempotency_key=idempotency_key,
                payload=recipe.payload,
            )
            receipt = _validate_job_receipt(raw, recipe)
            _write_ledger(
                self.worker,
                {
                    "version": 1,
                    "agent_id": agent_id,
                    "job_id": receipt["job_id"],
                    "recipe_id": recipe.recipe_id,
                    "profile_id": recipe.profile_id,
                    "project": recipe.project,
                    "request_key_sha256": hashlib.sha256(
                        request_key.encode("utf-8")
                    ).hexdigest(),
                },
            )
            return _result(self, receipt)
        except (FleetBackendError, FleetBatchToolError) as exc:
            return _failure(self, exc)


class FleetBatchJobStatusTool(BaseTool):
    def __init__(self, worker=None) -> None:
        super().__init__(
            name="fleet_batch_job_status",
            description=(
                "Read the durable status of a Fleet batch job previously submitted by "
                "this exact managed agent. It cannot inspect another agent's jobs. In "
                "continuous mode, check the owned job after the harness's wait interval; "
                "do not restart candidate discovery while it is pending."
            ),
        )
        self.worker = worker

    def execute(self, job_id: str) -> ToolResult:
        try:
            ownership = _read_ledger(self.worker, job_id)
            recipe = _RECIPES[str(ownership["recipe_id"])]
            raw = _client().job_status(job_id)
            receipt = _validate_job_receipt(raw, recipe, job_id=job_id)
            return _result(self, receipt)
        except (FleetBackendError, FleetBatchToolError) as exc:
            return _failure(self, exc)
