"""Signed legacy evidence gate for one OpenCode turn.

OpenCode owns reasoning and context management, but it is not the authority on
whether an Aeon tool effect completed.  The MCP process publishes the reviewed
Worker's exact request-contract state after each call.  The supervisor verifies
that snapshot before making model text visible.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from aeon.core.agent_protocol import ExecutionState, RequestContract
from aeon.core.durable_agent_guard import DurableAgentTurnGuard
from aeon.core.research_quality import ResearchQualityGuard, _ModelEvidence, _SearchEvidence

from .opencode_config import _atomic_private_bytes, _private_directory


COMPLETION_STATE_ENV = "AEON_OPENCODE_COMPLETION_STATE"
COMPLETION_KEY_FILE_ENV = "AEON_OPENCODE_COMPLETION_KEY_FILE"
COMPLETION_NONCE_ENV = "AEON_OPENCODE_COMPLETION_NONCE"
COMPLETION_AUTHORITY_SHA256_ENV = "AEON_OPENCODE_AUTHORITY_SHA256"
MAX_COMPLETION_STATE_BYTES = 8 * 1024 * 1024
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE_RE = re.compile(r"^[0-9a-f]{32}$")


class OpenCodeCompletionError(RuntimeError):
    """OpenCode final text lacks an exact, intact legacy evidence record."""


def authority_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def authority_sha256(value: Any) -> str:
    return hashlib.sha256(authority_text(value).encode("utf-8")).hexdigest()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _research_snapshot(guard: ResearchQualityGuard) -> dict[str, Any]:
    return {
        "active": bool(guard._active),
        "campaign_goal_sha256": str(guard._campaign_goal_sha256),
        "branches": [dict(item) for item in guard._branches[-guard.MAX_BRANCHES :]],
        "infos": {
            key: asdict(value) for key, value in list(guard._infos.items())[:100]
        },
        "license_commits": {
            key: sorted(values)[:100]
            for key, values in list(guard._license_commits.items())[:100]
        },
        "searches": {
            key: asdict(value) for key, value in list(guard._searches.items())[:100]
        },
    }


def _restore_research(value: Any) -> ResearchQualityGuard:
    guard = ResearchQualityGuard()
    if not isinstance(value, Mapping):
        raise OpenCodeCompletionError("OpenCode completion research state is invalid")
    digest = str(value.get("campaign_goal_sha256") or "")
    if digest and not _HEX_64_RE.fullmatch(digest):
        raise OpenCodeCompletionError("OpenCode completion research identity is invalid")
    branches = value.get("branches")
    infos = value.get("infos")
    commits = value.get("license_commits")
    searches = value.get("searches")
    if not isinstance(branches, list) or not isinstance(infos, Mapping):
        raise OpenCodeCompletionError("OpenCode completion research evidence is invalid")
    if not isinstance(commits, Mapping) or not isinstance(searches, Mapping):
        raise OpenCodeCompletionError("OpenCode completion research evidence is invalid")
    if len(branches) > guard.MAX_BRANCHES or len(infos) > 100 or len(commits) > 100 or len(searches) > 100:
        raise OpenCodeCompletionError("OpenCode completion research evidence is oversized")
    guard._campaign_goal_sha256 = digest
    guard._branches = [dict(item) for item in branches if isinstance(item, Mapping)]
    if len(guard._branches) != len(branches):
        raise OpenCodeCompletionError("OpenCode completion research ledger is invalid")
    try:
        guard._infos = {
            str(key): _ModelEvidence(**dict(item))
            for key, item in infos.items()
            if isinstance(key, str) and isinstance(item, Mapping)
        }
        guard._license_commits = {
            str(key): {str(commit) for commit in values}
            for key, values in commits.items()
            if isinstance(key, str) and isinstance(values, list) and len(values) <= 100
        }
        guard._searches = {
            str(key): _SearchEvidence(**dict(item))
            for key, item in searches.items()
            if isinstance(key, str) and isinstance(item, Mapping)
        }
    except (TypeError, ValueError) as exc:
        raise OpenCodeCompletionError("OpenCode completion research evidence is invalid") from exc
    if len(guard._infos) != len(infos) or len(guard._license_commits) != len(commits) or len(guard._searches) != len(searches):
        raise OpenCodeCompletionError("OpenCode completion research evidence is invalid")
    guard._active = value.get("active") is True
    return guard


class CompletionStateWriter:
    """Publish one authenticated snapshot from the MCP Worker process."""

    def __init__(
        self,
        *,
        path: Path,
        key: bytes,
        nonce: str,
        authority_digest: str,
        instance_id: str,
        workspace: str,
    ) -> None:
        self.path = path
        self.key = key
        self.nonce = nonce
        self.authority_digest = authority_digest
        self.instance_id = instance_id
        self.workspace = workspace

    @classmethod
    def from_environment(cls) -> "CompletionStateWriter":
        path_value = os.environ.get(COMPLETION_STATE_ENV, "")
        key_path_value = os.environ.get(COMPLETION_KEY_FILE_ENV, "")
        nonce = os.environ.get(COMPLETION_NONCE_ENV, "")
        digest = os.environ.get(COMPLETION_AUTHORITY_SHA256_ENV, "")
        instance_id = os.environ.get("AEON_OPENCODE_INSTANCE_ID", "")
        path = Path(path_value)
        key_path = Path(key_path_value)
        if not path_value or not path.is_absolute() or path.name != "completion-state.json":
            raise OpenCodeCompletionError("OpenCode completion state path is invalid")
        if (
            not key_path_value
            or not key_path.is_absolute()
            or key_path.name != "completion-key.bin"
            or key_path.parent != path.parent
            or not _HEX_64_RE.fullmatch(nonce)
        ):
            raise OpenCodeCompletionError("OpenCode completion capability is invalid")
        if not _HEX_64_RE.fullmatch(digest) or not _INSTANCE_RE.fullmatch(instance_id):
            raise OpenCodeCompletionError("OpenCode completion binding is invalid")
        _private_directory(path.parent, create=False)
        descriptor: int | None = None
        try:
            descriptor = os.open(
                key_path,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size != 32
            ):
                raise OpenCodeCompletionError(
                    "OpenCode completion capability is not owner-private"
                )
            key = os.read(descriptor, 33)
            if len(key) != 32:
                raise OpenCodeCompletionError("OpenCode completion capability is invalid")
        except OSError as exc:
            raise OpenCodeCompletionError(
                "OpenCode completion capability is unavailable"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return cls(
            path=path,
            key=key,
            nonce=nonce,
            authority_digest=digest,
            instance_id=instance_id,
            workspace=str(Path.cwd().resolve(strict=True)),
        )

    def publish(self, worker: Any, *, tool_invocations: int) -> None:
        contract = getattr(worker, "request_contract", None)
        if not isinstance(contract, RequestContract):
            raise OpenCodeCompletionError("OpenCode Worker has no request contract")
        payload = {
            "schema": 1,
            "nonce": self.nonce,
            "authority_sha256": self.authority_digest,
            "instance_id": self.instance_id,
            "workspace": self.workspace,
            "tool_invocations": int(tool_invocations),
            "worker": {
                "instance_id": str(getattr(worker, "instance_id", "")),
                "request_contract": contract.to_state_dict(),
                "durable_project_manager": bool(worker._durable_agent_guard.project_manager),
                "durable_agent_guard": worker._durable_agent_guard.to_state_dict(),
                "research_quality_guard": _research_snapshot(worker._research_quality_guard),
                # This is conservative if a child finishes between the last tool
                # receipt and final publication; it can never manufacture success.
                "unresolved_sub_agent_error": str(worker._unresolved_sub_agent_error() or "")[:4000],
            },
        }
        document = dict(payload)
        document["hmac_sha256"] = hmac.new(self.key, _canonical(payload), hashlib.sha256).hexdigest()
        encoded = _canonical(document) + b"\n"
        if len(encoded) > MAX_COMPLETION_STATE_BYTES:
            raise OpenCodeCompletionError("OpenCode completion evidence is oversized")
        _atomic_private_bytes(self.path.parent, self.path.name, encoded)


def _read_document(path: Path) -> dict[str, Any]:
    descriptor: int | None = None
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or not 1 <= metadata.st_size <= MAX_COMPLETION_STATE_BYTES
        ):
            raise OpenCodeCompletionError("OpenCode completion state is not owner-private")
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns
        ):
            raise OpenCodeCompletionError("OpenCode completion state changed before read")
        raw = os.read(descriptor, MAX_COMPLETION_STATE_BYTES + 1)
        final = os.fstat(descriptor)
        if (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) != (
            opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns
        ):
            raise OpenCodeCompletionError("OpenCode completion state changed during read")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpenCodeCompletionError("OpenCode completion state is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OpenCodeCompletionError("OpenCode completion state is invalid") from exc
    if not isinstance(value, dict):
        raise OpenCodeCompletionError("OpenCode completion state is invalid")
    return value


def _completion_error_from_worker_state(
    state: Any,
    *,
    final_text: str,
    expected_authority: str,
    expected_instance_id: str,
    expected_workspace: str,
    expected_project_manager: bool,
) -> str:
    if not isinstance(state, Mapping):
        raise OpenCodeCompletionError("OpenCode Worker completion state is invalid")
    if state.get("instance_id") != expected_instance_id:
        raise OpenCodeCompletionError("OpenCode Worker completion state belongs to another instance")
    contract_state = state.get("request_contract")
    if not isinstance(contract_state, Mapping):
        raise OpenCodeCompletionError("OpenCode Worker request state is missing")
    contract = RequestContract.from_state_dict(contract_state)
    if (
        contract.raw_request != expected_authority
        or contract.authority_request != expected_authority
        or contract.workspace_root != expected_workspace
        or contract.state != ExecutionState.RUNNING
    ):
        raise OpenCodeCompletionError("OpenCode Worker request state has the wrong authority")
    project_manager_value = state.get("durable_project_manager")
    if not isinstance(project_manager_value, bool) or (
        project_manager_value != expected_project_manager
    ):
        raise OpenCodeCompletionError(
            "OpenCode Worker completion state has the wrong authority class"
        )
    project_manager = project_manager_value
    durable = DurableAgentTurnGuard(project_manager=project_manager)
    durable.restore_state_dict(state.get("durable_agent_guard"))
    research = _restore_research(state.get("research_quality_guard"))
    unresolved = state.get("unresolved_sub_agent_error")
    if not isinstance(unresolved, str) or len(unresolved) > 4000:
        raise OpenCodeCompletionError("OpenCode Worker sub-agent state is invalid")
    return (
        durable.visible_claim_error(final_text)
        or contract.goal_completion_error()
        or durable.completion_error(final_text)
        or research.completion_error(final_text)
        or contract.completion_error(final_text)
        or unresolved
    )


def validate_completion(
    *,
    path: Path,
    key: bytes,
    nonce: str,
    authority: str,
    instance_id: str,
    workspace: str,
    final_text: str,
    tool_calls: int,
    project_manager: bool = False,
) -> None:
    """Raise unless final text is supported by exact current-turn evidence."""

    exact_authority = authority_text(authority)
    if tool_calls <= 0:
        # OpenCode may answer without ever starting MCP. Rebuild the empty legacy
        # contract locally: ordinary knowledge answers pass, while inspect/change
        # claims still require receipts.
        contract = RequestContract.from_request(exact_authority)
        durable = DurableAgentTurnGuard(project_manager=project_manager)
        durable.begin_user_turn(exact_authority)
        research = ResearchQualityGuard()
        research.begin_cycle(exact_authority)
        error = (
            durable.visible_claim_error(final_text)
            or contract.goal_completion_error()
            or durable.completion_error(final_text)
            or research.completion_error(final_text)
            or contract.completion_error(final_text)
        )
        if error:
            raise OpenCodeCompletionError(error)
        return

    document = _read_document(path)
    mac = document.pop("hmac_sha256", None)
    if not isinstance(mac, str) or not _HEX_64_RE.fullmatch(mac):
        raise OpenCodeCompletionError("OpenCode completion state has no valid authenticator")
    expected_mac = hmac.new(key, _canonical(document), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, expected_mac):
        raise OpenCodeCompletionError("OpenCode completion state failed integrity verification")
    if (
        document.get("schema") != 1
        or document.get("nonce") != nonce
        or document.get("authority_sha256") != authority_sha256(exact_authority)
        or document.get("instance_id") != instance_id
        or document.get("workspace") != workspace
    ):
        raise OpenCodeCompletionError("OpenCode completion state is bound to another turn")
    invocations = document.get("tool_invocations")
    if (
        not isinstance(invocations, int)
        or isinstance(invocations, bool)
        or invocations < 1
        or invocations != tool_calls
    ):
        raise OpenCodeCompletionError(
            "OpenCode completion state does not match the observed tool calls"
        )
    worker_state = document.get("worker")
    contract_state = worker_state.get("request_contract") if isinstance(worker_state, Mapping) else None
    results = contract_state.get("results") if isinstance(contract_state, Mapping) else None
    if not isinstance(results, list) or not results:
        raise OpenCodeCompletionError("OpenCode completion state has no typed tool receipt")
    error = _completion_error_from_worker_state(
        worker_state,
        final_text=final_text,
        expected_authority=exact_authority,
        expected_instance_id=instance_id,
        expected_workspace=workspace,
        expected_project_manager=project_manager,
    )
    if error:
        raise OpenCodeCompletionError(error)


__all__ = (
    "COMPLETION_AUTHORITY_SHA256_ENV",
    "COMPLETION_KEY_FILE_ENV",
    "COMPLETION_NONCE_ENV",
    "COMPLETION_STATE_ENV",
    "CompletionStateWriter",
    "OpenCodeCompletionError",
    "authority_sha256",
    "validate_completion",
)
