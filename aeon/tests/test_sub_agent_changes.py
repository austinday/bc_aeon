"""Hermetic behavior tests for mutable sub-agent change integration."""

from __future__ import annotations

import os
import subprocess
import tempfile
import types
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from aeon.core import runtime_signals as rt
from aeon.core.agent_protocol import (
    RequestContract,
    RequestMode,
    ToolStatus,
    infer_tool_policy,
)
from aeon.core.sub_agent_changes import (
    MUTABLE_CHANGE_RECEIPT,
    MUTABLE_INTEGRATION_RECEIPT,
    MUTABLE_PATCH_FILE,
    MUTABLE_WORKSPACE_RECEIPT,
    SubAgentChangeError,
    read_owned_json,
    snapshot_mutable_changes,
)
from aeon.core.tool_resources import ToolComputeRoute, tool_resource_policy
from aeon.tools.sub_agent import (
    GetSubAgentReport,
    IntegrateSubAgentChanges,
    SpawnSubAgent,
)


def _git(
    repository: Path,
    *arguments: str,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"},
    )


def _checked_git(repository: Path, *arguments: str) -> str:
    result = _git(repository, *arguments)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _fixture(base: Path, *, terminal_status: str = "COMPLETED") -> dict:
    repository = base / "principal"
    child = base / "child"
    agents = base / "agent-state"
    repository.mkdir()
    agents.mkdir()
    _checked_git(repository, "init")
    _checked_git(repository, "config", "user.name", "Aeon Test")
    _checked_git(repository, "config", "user.email", "aeon@example.invalid")
    (repository / "shared.txt").write_text("base\n", encoding="utf-8")
    (repository / "untouched.txt").write_text("keep\n", encoding="utf-8")
    _checked_git(repository, "add", "shared.txt", "untouched.txt")
    _checked_git(repository, "commit", "-m", "base")
    base_commit = _checked_git(repository, "rev-parse", "HEAD")
    _checked_git(repository, "worktree", "add", "--detach", str(child), base_commit)

    agent_id = str(uuid.uuid4())
    agent_dir = agents / agent_id
    agent_dir.mkdir(mode=0o700)
    rt.atomic_write_json(
        agent_dir / MUTABLE_WORKSPACE_RECEIPT,
        {
            "schema": 1,
            "agent_id": agent_id,
            "read_only": False,
            "base_commit": base_commit,
            "parent_repository": str(repository.resolve()),
            "worktree_repository": str(child.resolve()),
            "relative_workspace": ".",
        },
    )
    os.chmod(agent_dir / MUTABLE_WORKSPACE_RECEIPT, 0o600)

    (child / "shared.txt").write_text("from child\n", encoding="utf-8")
    (child / "created.txt").write_text("new artifact\n", encoding="utf-8")
    snapshot = snapshot_mutable_changes(child, agent_dir, agent_id)
    rt.atomic_write_json(
        agent_dir / "output.json",
        {
            "agent_id": agent_id,
            "status": terminal_status,
            "result": "Implemented the isolated change and checked its local behavior.",
            "workspace_changes": snapshot,
        },
    )
    rt.atomic_write_text(agent_dir / "status.txt", terminal_status)
    worker = types.SimpleNamespace(
        instance_id="principal",
        notified_sub_agents=set(),
        sub_agent_output_dir=lambda: agents,
    )
    return {
        "repository": repository,
        "child": child,
        "agents": agents,
        "agent_id": agent_id,
        "agent_dir": agent_dir,
        "worker": worker,
        "snapshot": snapshot,
    }


def _collect_report(fixture: dict) -> str:
    return GetSubAgentReport(worker=fixture["worker"]).execute(
        fixture["agent_id"][:8]
    )


def _integration_arguments(fixture: dict) -> dict:
    return {
        "agent_id": fixture["agent_id"][:8],
        "changeset_id": fixture["snapshot"]["patch_sha256"],
    }


def test_mutable_snapshot_captures_tracked_and_untracked_paths():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        snapshot = fixture["snapshot"]

        assert snapshot["empty"] is False
        assert snapshot["changed_paths"] == ["created.txt", "shared.txt"]
        assert [item["status"] for item in snapshot["path_changes"]] == ["A", "M"]
        assert snapshot["patch_bytes"] > 0
        assert len(snapshot["patch_sha256"]) == 64
        assert (fixture["agent_dir"] / MUTABLE_PATCH_FILE).is_file()
        assert read_owned_json(fixture["agent_dir"] / MUTABLE_CHANGE_RECEIPT) == snapshot


def test_mutable_snapshot_rejects_symlink_changes():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        os.symlink("shared.txt", fixture["child"] / "linked.txt")

        with pytest.raises(SubAgentChangeError, match="symlink|non-regular"):
            snapshot_mutable_changes(
                fixture["child"],
                fixture["agent_dir"],
                fixture["agent_id"],
            )


def test_mutable_spawn_binds_detached_worktree_to_exact_clean_base():
    class _Process:
        pid = 4242

        @staticmethod
        def wait():
            raise RuntimeError("fixture launcher is not real")

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        repository = base / "repository"
        agents = base / "agents"
        request_state = base / "request-state"
        repository.mkdir()
        agents.mkdir()
        _checked_git(repository, "init")
        _checked_git(repository, "config", "user.name", "Aeon Test")
        _checked_git(repository, "config", "user.email", "aeon@example.invalid")
        (repository / "source.txt").write_text("base\n", encoding="utf-8")
        _checked_git(repository, "add", "source.txt")
        _checked_git(repository, "commit", "-m", "base")
        base_commit = _checked_git(repository, "rev-parse", "HEAD")
        worker = types.SimpleNamespace(
            instance_id="principal",
            request_id="request",
            model_config={"model": "fixture", "provider": "openai"},
            debug_mode=False,
            sub_agent_output_dir=lambda: agents,
            _request_state_dir=lambda: request_state,
            blackboard_path=lambda: base / "blackboard.jsonl",
        )

        def spawn_git(_workspace: Path, *arguments: str, timeout: int = 30):
            del timeout
            if arguments == ("rev-parse", "--show-toplevel"):
                return subprocess.CompletedProcess(arguments, 0, str(repository) + "\n", "")
            if arguments[:2] == ("status", "--porcelain"):
                return subprocess.CompletedProcess(arguments, 0, "", "")
            if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
                return subprocess.CompletedProcess(arguments, 0, base_commit + "\n", "")
            if arguments[:3] == ("worktree", "add", "--detach"):
                Path(arguments[3]).mkdir(parents=True)
                return subprocess.CompletedProcess(arguments, 0, "", "")
            raise AssertionError(f"unexpected Git call: {arguments!r}")

        with _working_directory(repository), patch(
            "aeon.tools.sub_agent._run_git", side_effect=spawn_git
        ), patch(
            "aeon.tools.sub_agent.assert_sub_agent_systemd_units_available"
        ), patch(
            "aeon.tools.sub_agent.subprocess.Popen", return_value=_Process()
        ), patch(
            "aeon.tools.sub_agent.capture_sub_agent_process",
            return_value={"schema": 2, "agent_id": "fixture", "pid": 4242},
        ):
            result = SpawnSubAgent(worker=worker).execute(
                "Implement one isolated source change and report it.",
                read_only=False,
                time_budget_minutes=1,
                max_iterations=1,
                stall_timeout_seconds=60,
            )

        assert "Sub-agent spawned" in result
        agent_dirs = [path for path in agents.iterdir() if path.is_dir()]
        assert len(agent_dirs) == 1
        receipt = read_owned_json(agent_dirs[0] / MUTABLE_WORKSPACE_RECEIPT)
        assert receipt["agent_id"] == agent_dirs[0].name
        assert receipt["base_commit"] == base_commit
        assert receipt["parent_repository"] == str(repository.resolve())
        assert Path(receipt["worktree_repository"]).is_dir()


def test_principal_integrates_exact_patch_once_and_preserves_unrelated_edit():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        repository = fixture["repository"]
        (repository / "untouched.txt").write_text("principal concurrent edit\n", encoding="utf-8")
        tool = IntegrateSubAgentChanges(worker=fixture["worker"])
        _collect_report(fixture)

        with _working_directory(repository), patch(
            "aeon.tools.sub_agent._run_git", side_effect=_git
        ), patch(
            "aeon.tools.sub_agent.pid_alive", return_value=False
        ):
            result = tool.execute(**_integration_arguments(fixture))
            repeated = tool.execute(**_integration_arguments(fixture))

        assert result.status == ToolStatus.OK
        assert result.changed is True
        assert (repository / "shared.txt").read_text(encoding="utf-8") == "from child\n"
        assert (repository / "created.txt").read_text(encoding="utf-8") == "new artifact\n"
        assert (
            repository / "untouched.txt"
        ).read_text(encoding="utf-8") == "principal concurrent edit\n"
        assert repeated.status == ToolStatus.NO_CHANGE
        assert (fixture["agent_dir"] / MUTABLE_INTEGRATION_RECEIPT).is_file()


def test_conflicting_principal_edit_blocks_without_writing_any_child_path():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        repository = fixture["repository"]
        (repository / "shared.txt").write_text("principal wins\n", encoding="utf-8")
        tool = IntegrateSubAgentChanges(worker=fixture["worker"])
        _collect_report(fixture)

        with _working_directory(repository), patch(
            "aeon.tools.sub_agent._run_git", side_effect=_git
        ), patch(
            "aeon.tools.sub_agent.pid_alive", return_value=False
        ):
            result = tool.execute(**_integration_arguments(fixture))

        assert result.status == ToolStatus.BLOCKED
        assert result.error_code == "sub_agent_patch_conflict"
        assert (repository / "shared.txt").read_text(encoding="utf-8") == "principal wins\n"
        assert not (repository / "created.txt").exists()
        assert not (fixture["agent_dir"] / MUTABLE_INTEGRATION_RECEIPT).exists()


def test_partial_child_requires_deliberate_acceptance_and_report_points_to_integration():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary), terminal_status="BLOCKED")
        repository = fixture["repository"]
        tool = IntegrateSubAgentChanges(worker=fixture["worker"])

        report = GetSubAgentReport(worker=fixture["worker"]).execute(
            fixture["agent_id"][:8]
        )
        assert "integrate_sub_agent_changes" in report
        assert "patch receipt" in report

        with _working_directory(repository), patch(
            "aeon.tools.sub_agent._run_git", side_effect=_git
        ), patch(
            "aeon.tools.sub_agent.pid_alive", return_value=False
        ):
            refused = tool.execute(**_integration_arguments(fixture))
            accepted = tool.execute(
                **_integration_arguments(fixture),
                accept_partial=True,
            )

        assert refused.status == ToolStatus.BLOCKED
        assert refused.error_code == "partial_sub_agent_changes_require_acceptance"
        assert accepted.status == ToolStatus.OK


def test_tampered_patch_digest_fails_closed():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        repository = fixture["repository"]
        patch_path = fixture["agent_dir"] / MUTABLE_PATCH_FILE
        patch_path.write_bytes(patch_path.read_bytes() + b"\n")
        os.chmod(patch_path, 0o600)
        _collect_report(fixture)

        with _working_directory(repository), patch(
            "aeon.tools.sub_agent._run_git", side_effect=_git
        ), patch(
            "aeon.tools.sub_agent.pid_alive", return_value=False
        ):
            result = IntegrateSubAgentChanges(worker=fixture["worker"]).execute(
                **_integration_arguments(fixture)
            )

        assert result.status == ToolStatus.BLOCKED
        assert result.error_code == "invalid_sub_agent_change_receipt"
        assert (repository / "shared.txt").read_text(encoding="utf-8") == "base\n"


def test_terminal_report_eof_and_exact_process_absence_are_required():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        repository = fixture["repository"]
        tool = IntegrateSubAgentChanges(worker=fixture["worker"])

        with _working_directory(repository), patch(
            "aeon.tools.sub_agent.pid_alive", return_value=False
        ):
            uncollected = tool.execute(**_integration_arguments(fixture))
        assert uncollected.status == ToolStatus.BLOCKED
        assert uncollected.error_code == "sub_agent_report_uncollected"

        _collect_report(fixture)
        with _working_directory(repository), patch(
            "aeon.tools.sub_agent.pid_alive", return_value=True
        ):
            exiting = tool.execute(**_integration_arguments(fixture))
        assert exiting.status == ToolStatus.PENDING
        assert exiting.error_code == "sub_agent_still_exiting"

        with _working_directory(repository), patch(
            "aeon.tools.sub_agent.pid_alive", return_value=None
        ):
            ambiguous = tool.execute(**_integration_arguments(fixture))
        assert ambiguous.status == ToolStatus.BLOCKED
        assert ambiguous.error_code == "sub_agent_liveness_ambiguous"
        assert (repository / "shared.txt").read_text(encoding="utf-8") == "base\n"


def test_terminal_report_pages_cannot_skip_directly_to_eof():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        long_report = "A" * 9000
        rt.atomic_write_json(
            fixture["agent_dir"] / "output.json",
            {
                "agent_id": fixture["agent_id"],
                "status": "COMPLETED",
                "result": long_report,
                "workspace_changes": fixture["snapshot"],
            },
        )
        tool = GetSubAgentReport(worker=fixture["worker"])

        skipped = tool.execute(fixture["agent_id"], offset=8000)
        first = tool.execute(fixture["agent_id"], offset=0)
        final = tool.execute(fixture["agent_id"], offset=8000)

        assert "expected offset 0" in skipped
        assert "next_offset=8000" in first
        assert "EOF (report collected)" in final


def test_protected_path_policy_is_rechecked_before_integration():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        repository = fixture["repository"]
        _collect_report(fixture)

        with _working_directory(repository), patch(
            "aeon.tools.sub_agent.pid_alive", return_value=False
        ), patch(
            "aeon.core.protected.guard",
            side_effect=lambda path: "BLOCKED: protected fixture" if path.endswith("shared.txt") else None,
        ):
            result = IntegrateSubAgentChanges(worker=fixture["worker"]).execute(
                **_integration_arguments(fixture)
            )

        assert result.status == ToolStatus.BLOCKED
        assert result.error_code == "invalid_sub_agent_change_receipt"
        assert (repository / "shared.txt").read_text(encoding="utf-8") == "base\n"
        assert not (repository / "created.txt").exists()


def test_receipt_owned_paths_are_checked_against_owner_invariants():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        repository = fixture["repository"]
        fixture["worker"].request_contract = RequestContract.from_request(
            "Update shared.txt, but do not modify created.txt.",
            forced_mode=RequestMode.CHANGE_LOCAL,
            workspace_root=str(repository),
        )
        _collect_report(fixture)

        with _working_directory(repository), patch(
            "aeon.tools.sub_agent.pid_alive", return_value=False
        ):
            result = IntegrateSubAgentChanges(worker=fixture["worker"]).execute(
                **_integration_arguments(fixture)
            )

        assert result.status == ToolStatus.BLOCKED
        assert result.error_code == "sub_agent_changes_violate_invariant"
        assert not (repository / "created.txt").exists()


def test_prepared_journal_recovers_by_hash_without_replaying_patch():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        repository = fixture["repository"]
        tool = IntegrateSubAgentChanges(worker=fixture["worker"])
        _collect_report(fixture)

        with _working_directory(repository), patch(
            "aeon.tools.sub_agent._run_git", side_effect=_git
        ), patch(
            "aeon.tools.sub_agent.pid_alive", return_value=False
        ):
            applied = tool.execute(**_integration_arguments(fixture))
            journal_path = fixture["agent_dir"] / MUTABLE_INTEGRATION_RECEIPT
            journal = read_owned_json(journal_path)
            journal["status"] = "PREPARED"
            rt.atomic_write_json(journal_path, journal)
            os.chmod(journal_path, 0o600)
            recovered = tool.execute(**_integration_arguments(fixture))

        assert applied.status == ToolStatus.OK
        assert recovered.status == ToolStatus.OK
        assert recovered.changed is True
        assert read_owned_json(journal_path)["status"] == "APPLIED"


def test_integration_receipt_opens_principal_validation_debt():
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _fixture(Path(temporary))
        repository = fixture["repository"]
        contract = RequestContract.from_request(
            "Update shared.txt and validate the result.",
            forced_mode=RequestMode.CHANGE_LOCAL,
            workspace_root=str(repository),
        )
        fixture["worker"].request_contract = contract
        tool = IntegrateSubAgentChanges(worker=fixture["worker"])
        _collect_report(fixture)
        parameters = _integration_arguments(fixture)

        with _working_directory(repository), patch(
            "aeon.tools.sub_agent._run_git", side_effect=_git
        ), patch(
            "aeon.tools.sub_agent.pid_alive", return_value=False
        ):
            result = tool.execute(**parameters)

        contract.observe(
            result,
            policy=tool.policy,
            parameters=parameters,
            goal_refs=[],
        )
        assert result.status == ToolStatus.OK
        assert contract.needs_verification is True
        assert "later exact readback" in contract.completion_error("Done.")
        assert "shared.txt" in contract.pending_validation_targets


def test_integration_tool_has_local_mutation_and_cpu_route_contracts():
    policy = infer_tool_policy("integrate_sub_agent_changes")
    assert policy.side_effect.value == "local_mutation"
    assert tool_resource_policy("integrate_sub_agent_changes").route == ToolComputeRoute.LOCAL_CPU
