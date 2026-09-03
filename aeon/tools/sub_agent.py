import os
import sys
import json
import hashlib
import uuid
import time
import signal
import ctypes
import re
import stat
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

from aeon.tools.base import BaseTool
from aeon.core.agent_protocol import SideEffect, ToolResult, ToolStatus
from aeon.tools.command_fleet_guard import (
    require_fleet_low_priority_wrapper,
    scrubbed_fleet_command_environment,
)
from aeon.core import runtime_signals as rt
from aeon.core.sub_agent_changes import (
    MAX_CHANGED_PATHS,
    MAX_PATCH_BYTES,
    MUTABLE_CHANGE_RECEIPT,
    MUTABLE_INTEGRATION_RECEIPT,
    MUTABLE_PATCH_FILE,
    MUTABLE_WORKSPACE_RECEIPT,
    SUB_AGENT_REPORT_COLLECTION_RECEIPT,
    SUB_AGENT_REPORT_PROGRESS_RECEIPT,
    SubAgentChangeError,
    read_owned_json,
    validate_patch_file,
    validate_relative_path,
)
from aeon.core.workspace_files import WorkspaceFileBoundary, WorkspacePathError
from aeon.core.sub_agent_environment import bounded_sub_agent_environment
from aeon.remote.mcp_capability import (
    MCP_DELEGATION_ID_ENV,
    MCP_DELEGATION_TOKEN_FILE_ENV,
    MCP_URL_ENV,
    mcp_action_endpoint,
)
from aeon.remote.self_settings import (
    SELF_SETTINGS_TOKEN_FILE_ENV,
    SelfSettingsCapabilityError,
    read_self_settings_token,
    validate_managed_instance_id,
)
from aeon.core.sub_agent_state import (
    CPU_SANDBOX_SLICE_ENV,
    ProcessIdentityError,
    assert_sub_agent_systemd_units_available,
    capture_sub_agent_process,
    norm_status,
    pid_alive,
    resolve,
    sub_agent_systemd_command,
    sub_agent_systemd_units,
    terminate_sub_agent,
)


def _resolve_agent_dir(base_dir, agent_id):
    """Resolve an agent_id (full UUID, an unambiguous prefix, or a full directory
    name) to its actual sub-agent directory. gather_sub_agents shows operators a
    SHORT id, so the model frequently passes a prefix back to report/kill/steer;
    an exact-match lookup then fails with 'not found'. Matches:
      1. exact directory name (fast path)
      2. unique prefix match
      3. unique substring match (covers labelled dirs like 'verify_<uuid>')
    Returns (path, error_string). Exactly one of the two is None.
    """
    base_dir = Path(base_dir)
    if not agent_id:
        return None, "No agent_id provided."
    if not base_dir.exists():
        return None, "No sub-agents have been spawned in this session."

    agent_id = str(agent_id).strip()
    exact = base_dir / agent_id
    if exact.exists() and exact.is_dir():
        return exact, None

    dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    prefix = [d for d in dirs if d.name.startswith(agent_id)]
    if len(prefix) == 1:
        return prefix[0], None
    if len(prefix) > 1:
        opts = ", ".join(sorted(d.name[:12] for d in prefix))
        return None, (f"Ambiguous agent id '{agent_id}' matches multiple sub-agents ({opts}). "
                      f"Use more characters of the id.")

    sub = [d for d in dirs if agent_id in d.name]
    if len(sub) == 1:
        return sub[0], None
    if len(sub) > 1:
        opts = ", ".join(sorted(d.name[:12] for d in sub))
        return None, (f"Ambiguous agent id '{agent_id}' matches multiple sub-agents ({opts}). "
                      f"Use more characters of the id.")

    available = sorted(d.name[:12] for d in dirs if (d / "pid.txt").exists())
    hint = f" Known sub-agents: {', '.join(available)}." if available else ""
    return None, f"Agent '{agent_id}' not found.{hint}"


def _output_dir_for_worker(worker):
    """Use request-scoped private state, with a legacy test/session fallback."""

    resolver = getattr(worker, "sub_agent_output_dir", None)
    if callable(resolver):
        return resolver()
    instance_id = getattr(worker, "instance_id", "default")
    return Path(os.getcwd()) / "aeon_output" / instance_id / "sub_agents"


def _run_git(workspace: Path, *arguments: str, timeout: int):
    """Run fixed lifecycle-only Git without hooks or inherited Fleet authority."""

    environment = scrubbed_fleet_command_environment()
    for key in tuple(environment):
        if key.startswith("GIT_"):
            environment.pop(key, None)
    return subprocess.run(
        [
            require_fleet_low_priority_wrapper(),
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(workspace),
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=environment,
    )


def _git_failure_detail(result: subprocess.CompletedProcess) -> str:
    detail = str(result.stderr or result.stdout or "").strip().splitlines()
    return detail[-1][:500] if detail else "unknown git error"


def _change_result(
    status: ToolStatus,
    summary: str,
    *,
    changed: bool = False,
    error_code: str = "",
    evidence: list[str] | None = None,
    artifacts: list[str] | None = None,
) -> ToolResult:
    return ToolResult(
        tool_name="integrate_sub_agent_changes",
        status=status,
        changed=changed,
        summary=summary,
        evidence=list(evidence or []),
        artifacts=list(artifacts or []),
        error_code=error_code,
        retryable=False,
        side_effect=SideEffect.LOCAL_MUTATION,
    )


def _admit_principal_change_paths(
    worker,
    repository: Path,
    relative_workspace: str,
    path_changes: list[dict],
) -> list[str]:
    """Apply the ordinary file-tool boundary to every receipt-owned path."""

    boundary = WorkspaceFileBoundary.from_worker(worker)
    expected_workspace = (
        repository
        if relative_workspace == "."
        else repository / relative_workspace
    ).resolve(strict=True)
    if boundary.root != expected_workspace:
        raise SubAgentChangeError(
            "principal launch-workspace identity no longer matches the delegated workspace"
        )
    try:
        from aeon.core.protected import guard as protected_guard
    except Exception as exc:
        raise SubAgentChangeError("protected-path policy is unavailable") from exc

    artifacts = []
    for item in path_changes:
        target = repository / item["path"]
        try:
            bound = boundary.bind(str(target))
        except WorkspacePathError as exc:
            raise SubAgentChangeError(str(exc)) from exc
        blocked = protected_guard(str(bound.absolute))
        if blocked:
            raise SubAgentChangeError(blocked)

        # Reject every existing symlink/non-directory ancestor and every
        # non-regular/multiply-linked leaf. Git's path checks are useful defense
        # in depth, but they do not replace Aeon's descriptor-oriented boundary.
        cursor = boundary.root
        missing_parent = False
        for component in bound.parts[:-1]:
            cursor = cursor / component
            if missing_parent:
                continue
            try:
                metadata = cursor.lstat()
            except FileNotFoundError:
                missing_parent = True
                continue
            except OSError as exc:
                raise SubAgentChangeError(
                    f"cannot validate parent of integration path {item['path']}"
                ) from exc
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise SubAgentChangeError(
                    f"integration path has a symlink or non-directory ancestor: {item['path']}"
                )
        if not missing_parent:
            try:
                leaf = bound.absolute.lstat()
            except FileNotFoundError:
                leaf = None
            except OSError as exc:
                raise SubAgentChangeError(
                    f"cannot validate integration path {item['path']}"
                ) from exc
            if leaf is not None and (
                not stat.S_ISREG(leaf.st_mode)
                or stat.S_ISLNK(leaf.st_mode)
                or leaf.st_nlink != 1
            ):
                raise SubAgentChangeError(
                    f"integration target is a symlink, linked file, or non-regular file: {item['path']}"
                )
        artifacts.append(str(bound.absolute))
    return artifacts


def _principal_paths_match_base(
    repository: Path,
    base_commit: str,
    path_changes: list[dict],
) -> str:
    """Return a conflict reason if any affected principal path changed."""

    new_paths = [
        item["path"] for item in path_changes if item.get("old_mode") == "000000"
    ]
    for path in new_paths:
        if os.path.lexists(repository / path):
            return f"new child path already exists in the principal worktree: {path}"
    tracked_paths = [
        item["path"] for item in path_changes if item.get("old_mode") != "000000"
    ]
    for offset in range(0, len(tracked_paths), 128):
        batch = tracked_paths[offset : offset + 128]
        comparison = _run_git(
            repository,
            "diff",
            "--quiet",
            "--no-ext-diff",
            "--no-textconv",
            base_commit,
            "--",
            *batch,
            timeout=30,
        )
        if comparison.returncode == 1:
            return "an affected principal path changed after the child was dispatched"
        if comparison.returncode != 0:
            return "Git could not prove affected principal paths still match the child base"
    return ""


def _final_change_receipt_matches(repository: Path, path_changes: list[dict]) -> bool:
    """Verify the exact filesystem state described by the child snapshot."""

    for item in path_changes:
        target = repository / item["path"]
        if item["status"] == "D":
            if os.path.lexists(target):
                return False
            continue
        try:
            metadata = target.lstat()
        except OSError:
            return False
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            return False
        mode = "100755" if metadata.st_mode & 0o111 else "100644"
        if (
            mode != item["new_mode"]
            or metadata.st_size != item["final_size"]
        ):
            return False
        digest = hashlib.sha256()
        try:
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return False
        if digest.hexdigest() != item["final_sha256"]:
            return False
    return True


def _reap_sub_agent_launcher(process, agent_dir):
    """Reap systemd-run and retire its exact leaf slice after natural exit.

    The durable receipt remains the authority. If a nested command scope is
    unexpectedly still populated, ``terminate_sub_agent`` gives the whole exact
    slice the same 30-second cleanup grace before escalation. Ambiguity is
    recorded for an operator and is never converted into a broader signal.
    """

    try:
        process.wait()
    except Exception:
        return
    try:
        terminate_sub_agent(agent_dir)
    except ProcessIdentityError as exc:
        try:
            rt.atomic_write_text(
                Path(agent_dir) / "lifecycle_error.txt",
                f"Exact sub-agent slice cleanup was refused: {str(exc)[:500]}",
            )
        except Exception:
            pass


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _create_mcp_delegation(
    *,
    agent_id: str,
    credential_ids: list[str],
    expires_in: int,
    agent_dir: Path,
) -> tuple[str, Path]:
    """Mint one expiring Nexus proxy capability without exposing OAuth secrets."""

    try:
        endpoint = mcp_action_endpoint(os.environ.get(MCP_URL_ENV, ""), "delegations")
        parent_id = validate_managed_instance_id(
            os.environ.get("AEON_REMOTE_INSTANCE_ID")
        )
        parent_token = read_self_settings_token(
            os.environ.get(SELF_SETTINGS_TOKEN_FILE_ENV, "")
        )
    except SelfSettingsCapabilityError as exc:
        raise RuntimeError(f"MCP delegation is unavailable: {exc}") from exc
    request = urllib.request.Request(
        endpoint,
        method="POST",
        headers={
            "Authorization": f"Bearer {parent_token}",
            "Content-Type": "application/json",
            "X-Nexus-Agent-Instance": parent_id,
        },
        data=json.dumps(
            {
                "delegation_id": agent_id,
                "credential_ids": credential_ids,
                "expires_in": expires_in,
            },
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirectHandler(),
    )
    try:
        with opener.open(request, timeout=15) as response:
            raw = response.read(16385)
            if len(raw) > 16384:
                raise RuntimeError("Nexus returned an oversized delegation response")
            document = json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read(8192)
        try:
            detail = json.loads(raw.decode("utf-8")).get("detail")
        except (UnicodeError, json.JSONDecodeError, AttributeError):
            detail = None
        raise RuntimeError(detail or f"Nexus returned HTTP {exc.code}") from None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Nexus MCP delegation failed: {exc}") from exc
    if (
        not isinstance(document, dict)
        or document.get("delegation_id") != agent_id
        or not isinstance(document.get("token"), str)
    ):
        raise RuntimeError("Nexus returned an invalid delegation capability")
    token_path = agent_dir / "mcp-delegation.token"
    rt.atomic_write_text(token_path, str(document["token"]))
    os.chmod(token_path, 0o600)
    return os.environ[MCP_URL_ENV], token_path


def uncollected_sub_agents(base_dir, notified_set):
    """Return short ids of sub-agents that have a terminal result which was never
    surfaced to the principal (i.e. never gathered/reported). Used to stop the
    primary from abandoning a dispatched researcher at task_complete."""
    base_dir = Path(base_dir)
    out = []
    if not base_dir.exists():
        return out
    for d in base_dir.iterdir():
        if not (d.is_dir() and (d / "pid.txt").exists()):
            continue
        is_term, status, _ = resolve(d)
        if not is_term:
            # Still running but spawned this session and never harvested -> also worth flagging.
            out.append((d.name.split("-")[0], "RUNNING"))
            continue
        key = f"{d.name}_{norm_status(status)}"
        if key not in (notified_set or set()):
            out.append((d.name.split("-")[0], norm_status(status)))
    return out


class SpawnSubAgent(BaseTool):
    MAX_CONCURRENT = 5
    DEFAULT_BUDGET_MIN = 40
    DEFAULT_STALL = 600
    HARD_WALLCLOCK_CEILING = 7200
    HARD_STALL_CEILING = 1800

    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="spawn_sub_agent",
            description=(
                "Dispatch one bounded sub-agent for a genuinely independent thread. Default read_only=true "
                "shares the current workspace through an enforced read-only request mode. read_only=false "
                "is allowed only for a clean Git repository and creates a detached isolated worktree; it "
                "never permits concurrent edits in the principal's tree. A finished mutable child emits "
                "an immutable patch receipt which the principal must review and apply with "
                "integrate_sub_agent_changes. Sub-agents cannot recursively "
                "delegate, resume the principal, or use principal-only Nexus capabilities. Give a complete "
                "objective and expected report. Collect the terminal report before final completion.\n"
                "Parameters: objective (str, required); time_budget_minutes (int, optional, default 40); "
                "max_iterations (int, optional, default 20); stall_timeout_seconds (int, optional, default "
                "600); read_only (bool, optional, default true)."
                " allowed_credentials (list of exact credential IDs, optional) grants "
                "only those accounts for this bounded run and defaults to none."
            ),
        )
        self.worker = worker
        self.llm_client = llm_client

    @property
    def output_dir(self):
        return _output_dir_for_worker(self.worker)

    def _running_count(self):
        if not self.output_dir.exists():
            return 0
        n = 0
        for d in self.output_dir.iterdir():
            if d.is_dir() and (d / "pid.txt").exists() and not resolve(d)[0]:
                n += 1
        return n

    def _running_objective_match(self, norm_objective):
        """Return the short id of a still-running sub-agent whose objective
        normalizes to the same text, or None. Guards against duplicate spawns."""
        if not self.output_dir.exists():
            return None
        for d in self.output_dir.iterdir():
            if not (d.is_dir() and (d / "pid.txt").exists() and not resolve(d)[0]):
                continue
            obj_file = d / "objective.txt"
            if not obj_file.exists():
                continue
            try:
                existing = " ".join(obj_file.read_text(encoding="utf-8").split()).lower()
            except OSError:
                continue
            if existing == norm_objective:
                return d.name[:8]
        return None

    def execute(
        self,
        objective: str,
        time_budget_minutes: int = None,
        max_iterations: int = None,
        stall_timeout_seconds: int = None,
        read_only: bool = True,
        allowed_credentials: list[str] | None = None,
    ):
        if not self.worker:
            return "COMMAND FAILED: Worker context missing."

        if not objective or not str(objective).strip():
            return ("COMMAND FAILED: 'objective' is empty. A sub-agent needs a complete, self-contained "
                    "task description (what to do, where, and what 'done' looks like).")
        objective = str(objective)

        running = self._running_count()
        if running >= self.MAX_CONCURRENT:
            return (f"COMMAND FAILED: Maximum concurrent sub-agents ({self.MAX_CONCURRENT}) reached. "
                    f"Wait/collect with gather_sub_agents or free one with kill_sub_agent.")

        # Prevent accidentally spawning a duplicate of work already in flight.
        norm = " ".join(objective.split()).lower()
        dup = self._running_objective_match(norm)
        if dup:
            return (f"COMMAND FAILED: a sub-agent ('{dup}') is already running this exact objective. "
                    f"Watch it in the SUB-AGENTS section, steer_sub_agent it, or advance other work — "
                    f"do not spawn a duplicate. Use a DIFFERENT, orthogonal objective if you meant to "
                    f"parallelize further.")

        model_cfg = getattr(self.worker, "model_config", None)
        if not model_cfg:
            return ("COMMAND FAILED: No model_config on the primary worker, so a sub-agent cannot be "
                    "configured with the active model.")

        try:
            budget_min = int(time_budget_minutes) if time_budget_minutes else self.DEFAULT_BUDGET_MIN
        except (TypeError, ValueError):
            budget_min = self.DEFAULT_BUDGET_MIN
        max_wallclock = max(60, min(budget_min * 60, self.HARD_WALLCLOCK_CEILING))

        try:
            stall = int(stall_timeout_seconds) if stall_timeout_seconds else self.DEFAULT_STALL
        except (TypeError, ValueError):
            stall = self.DEFAULT_STALL
        stall = max(60, min(stall, self.HARD_STALL_CEILING))

        try:
            iters = int(max_iterations) if max_iterations else 20
        except (TypeError, ValueError):
            iters = 20
        iters = max(1, min(iters, 100))

        if not isinstance(read_only, bool):
            return "COMMAND FAILED: read_only must be a JSON boolean."
        if allowed_credentials is None:
            allowed_credentials = []
        if (
            not isinstance(allowed_credentials, list)
            or any(
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 128
                for value in allowed_credentials
            )
        ):
            return "COMMAND FAILED: allowed_credentials must be a list of credential IDs."
        allowed_credentials = sorted({value.strip() for value in allowed_credentials})

        agent_id = str(uuid.uuid4())
        try:
            scope_unit, slice_unit = sub_agent_systemd_units(agent_id)
            assert_sub_agent_systemd_units_available(agent_id)
        except ProcessIdentityError as exc:
            return f"COMMAND FAILED: could not reserve unique sub-agent units: {exc}"
        agent_dir = self.output_dir / agent_id
        agent_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(agent_dir, 0o700)
        except OSError:
            pass
        # Persist the objective so duplicate-spawn detection (and humans reading
        # the output dir) can see what each sub-agent was tasked with.
        try:
            rt.atomic_write_text(agent_dir / "objective.txt", objective)
        except Exception:
            pass

        workspace_path = Path(os.getcwd()).resolve()
        child_workspace = agent_dir / "workspace"
        isolation_note = "read-only shared workspace"
        mutable_repo_root = None
        mutable_worktree_root = None

        def rollback_unlaunched_worktree() -> str:
            """Remove only this task-created, never-launched clean worktree."""

            nonlocal mutable_worktree_root
            if mutable_repo_root is None or mutable_worktree_root is None:
                return ""
            try:
                removed = _run_git(
                    mutable_repo_root,
                    "worktree",
                    "remove",
                    "--force",
                    str(mutable_worktree_root),
                    timeout=60,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return f" Exact task worktree cleanup also failed: {exc}"
            if removed.returncode != 0:
                return (
                    " Exact task worktree cleanup also failed: "
                    + _git_failure_detail(removed)
                )
            mutable_worktree_root = None
            return ""

        if read_only:
            if child_workspace.exists() or child_workspace.is_symlink():
                child_workspace.unlink()
            child_workspace.symlink_to(workspace_path)
        else:
            # Mutable agents never share a writable tree. A detached worktree is
            # created only from a clean Git snapshot so it cannot silently omit
            # the principal's uncommitted work or race those edits.
            try:
                root_result = _run_git(
                    workspace_path, "rev-parse", "--show-toplevel", timeout=10,
                )
                if root_result.returncode != 0:
                    return (
                        "COMMAND BLOCKED: mutable sub-agent requested, but the workspace is not "
                        "inside a Git repository. Use read_only=true or prepare explicit isolation."
                    )
                repo_root = Path(root_result.stdout.strip()).resolve()
                relative_workspace = workspace_path.relative_to(repo_root)
                dirty = _run_git(
                    repo_root,
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    timeout=15,
                )
                if dirty.returncode != 0 or dirty.stdout.strip():
                    return (
                        "COMMAND BLOCKED: mutable sub-agent isolation requires a clean Git worktree. "
                        "Current tracked/untracked changes would be omitted from a detached snapshot. "
                        "Use a read-only agent or let the principal finish the edits."
                    )
                head = _run_git(
                    repo_root,
                    "rev-parse",
                    "--verify",
                    "HEAD^{commit}",
                    timeout=10,
                )
                base_commit = str(head.stdout or "").strip().lower()
                if head.returncode != 0 or not re.fullmatch(
                    r"[0-9a-f]{40,64}", base_commit
                ):
                    return (
                        "COMMAND BLOCKED: mutable sub-agent isolation could not bind "
                        "the clean repository to an exact base commit."
                    )
                worktree_root = self.worker._request_state_dir() / "worktrees" / agent_id
                worktree_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                add = _run_git(
                    repo_root,
                    "worktree",
                    "add",
                    "--detach",
                    str(worktree_root),
                    "HEAD",
                    timeout=60,
                )
                if add.returncode != 0:
                    detail = (add.stderr or add.stdout).strip().splitlines()
                    return "COMMAND FAILED: could not create isolated mutable worktree: " + (
                        detail[-1][:500] if detail else "unknown git error"
                    )
                mutable_repo_root = repo_root
                mutable_worktree_root = worktree_root
                child_workspace = worktree_root / relative_workspace
                if not child_workspace.is_dir():
                    return (
                        "COMMAND FAILED: isolated worktree does not contain the requested workspace path."
                        + rollback_unlaunched_worktree()
                    )
                rt.atomic_write_json(
                    agent_dir / MUTABLE_WORKSPACE_RECEIPT,
                    {
                        "schema": 1,
                        "agent_id": agent_id,
                        "read_only": False,
                        "base_commit": base_commit,
                        "parent_repository": str(repo_root),
                        "worktree_repository": str(worktree_root.resolve()),
                        "relative_workspace": relative_workspace.as_posix() or ".",
                    },
                )
                os.chmod(agent_dir / MUTABLE_WORKSPACE_RECEIPT, 0o600)
                isolation_note = f"isolated detached worktree at {worktree_root}"
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                return (
                    f"COMMAND FAILED: mutable sub-agent isolation failed: {type(exc).__name__}: {exc}"
                    + rollback_unlaunched_worktree()
                )

        coordinated_objective = (
            f"{objective}\n\n"
            f"[CAPABILITY] This sub-agent is {'read-only' if read_only else 'mutable but isolated'}. "
            f"Stay within that boundary. "
            f"{'Do not commit, checkout, or change Git HEAD/refs; the terminal transfer accepts only uncommitted worktree edits.' if not read_only else ''}"
            f"\n\n"
            f"[COORDINATION] For a genuinely shared finding, check/post the run-scoped blackboard; "
            f"do not duplicate a sibling's work.\n\n"
            f"[REPORTING] End with one complete `final` report for the principal: findings, evidence, "
            f"actions actually taken, artifacts, and any blocker. Do not claim completion from intent alone. "
            f"For mutable work, the harness separately captures your exact repository patch."
        )

        cmd = [
            sys.executable, "-m", "aeon.scripts.sub_agent_wrapper",
            "--agent_id", agent_id,
            "--objective", coordinated_objective,
            "--model_config", json.dumps(model_cfg),
            "--workspace", str(child_workspace),
            "--output_dir", str(agent_dir),
            "--max_iterations", str(iters),
            "--stall_timeout", str(stall),
            "--max_wallclock", str(max_wallclock),
        ]
        if read_only:
            cmd.append("--read_only")
        if getattr(self.worker, "debug_mode", False):
            cmd.append("--debug")

        try:
            scoped_cmd = sub_agent_systemd_command(agent_id, cmd)
        except ProcessIdentityError as exc:
            return (
                f"COMMAND FAILED: could not construct exact sub-agent scope: {exc}"
                + rollback_unlaunched_worktree()
            )

        def set_pdeathsig():
            try:
                ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)
            except Exception:
                pass

        request_id = str(getattr(self.worker, "request_id", "") or "unscoped")
        blackboard_resolver = getattr(self.worker, "blackboard_path", None)
        blackboard_path = (
            blackboard_resolver()
            if callable(blackboard_resolver)
            else self.output_dir.parent / "blackboard.jsonl"
        )
        child_env = bounded_sub_agent_environment()
        if allowed_credentials:
            try:
                mcp_url, delegation_token_path = _create_mcp_delegation(
                    agent_id=agent_id,
                    credential_ids=allowed_credentials,
                    expires_in=max_wallclock,
                    agent_dir=agent_dir,
                )
            except RuntimeError as exc:
                return f"COMMAND FAILED: {exc}" + rollback_unlaunched_worktree()
            child_env.update(
                {
                    MCP_URL_ENV: mcp_url,
                    MCP_DELEGATION_ID_ENV: agent_id,
                    MCP_DELEGATION_TOKEN_FILE_ENV: str(delegation_token_path),
                }
            )
        child_env.update({
            "AEON_PARENT_INSTANCE_ID": str(self.worker.instance_id),
            "AEON_PARENT_REQUEST_ID": request_id,
            "AEON_BLACKBOARD_PATH": str(blackboard_path),
            "AEON_READ_ONLY": "1" if read_only else "0",
            # Generated here only. Nested generic-shell scopes may inherit this
            # exact leaf slice, but neither the model nor inherited state chooses
            # or overrides the lifecycle boundary.
            CPU_SANDBOX_SLICE_ENV: slice_unit,
        })
        try:
            log_fd = open(agent_dir / "agent.log", "a")
        except OSError as exc:
            return (
                f"COMMAND FAILED: could not open sub-agent log: {exc}"
                + rollback_unlaunched_worktree()
            )
        try:
            process = subprocess.Popen(
                scoped_cmd,
                # Detach stdin: inheriting the principal's TTY makes the
                # sub-agent's console reader contend with the principal's for
                # the same terminal (background-session reads -> SIGTTIN).
                stdin=subprocess.DEVNULL,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                # A bounded child must not inherit the principal Project Manager's
                # identity or owner-only Nexus mutation capabilities.
                env=child_env,
                preexec_fn=set_pdeathsig,
                start_new_session=True,
            )
        except Exception as e:
            return (
                f"COMMAND FAILED: could not launch sub-agent process: {e}"
                + rollback_unlaunched_worktree()
            )
        finally:
            # The child inherited the fd; keeping it open in the parent leaks one
            # fd per spawn for the life of the session.
            log_fd.close()

        try:
            process_ref = capture_sub_agent_process(
                agent_dir,
                process.pid,
                scope_unit=scope_unit,
                slice_unit=slice_unit,
            )
            rt.atomic_write_json(agent_dir / "process.json", process_ref)
        except Exception as exc:
            # Popen returned this exact systemd-run launcher. Kill only that
            # pinned child; the wrapper's parent-death signal handles a scope we
            # could not safely commit to a durable identity receipt.
            try:
                process.kill()
                process.wait(timeout=5)
            except Exception:
                pass
            return f"COMMAND FAILED: could not record sub-agent process identity: {exc}"

        rt.atomic_write_text(agent_dir / "pid.txt", str(process.pid))
        rt.atomic_write_text(agent_dir / "status.txt", "RUNNING")
        threading.Thread(
            target=_reap_sub_agent_launcher,
            args=(process, agent_dir),
            daemon=True,
            name=f"aeon-subagent-reaper-{agent_id[:8]}",
        ).start()

        short_id = agent_id[:8]
        msg = (f"Sub-agent spawned. Agent ID: {agent_id} (refer to it as '{short_id}' in steer/report/kill). "
               f"Budget: {max_wallclock // 60} min wall-clock, {stall}s stall, {iters} max iterations. "
               f"Capability: {isolation_note}. It will appear in the run-scoped SUB-AGENTS section. "
               f"A mutable child's edits remain isolated until integrate_sub_agent_changes succeeds. "
               f"Collect its report before final completion, or stop it explicitly if no longer needed.")
        return msg


class IntegrateSubAgentChanges(BaseTool):
    """Apply one immutable detached-worktree patch after conflict checks."""

    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="integrate_sub_agent_changes",
            description=(
                "Integrate the exact patch produced by a finished mutable sub-agent into the "
                "principal's Git worktree. The harness verifies the child/base/repository identity, "
                "patch digest, bounded changed paths, exact unchanged base, and a no-write `git apply "
                "--check` before applying. It never merges commits, changes refs, stages files, or "
                "overwrites a conflicting principal edit. Read the terminal report first. "
                "By default only a COMPLETED child can be integrated; set accept_partial=true only "
                "after reviewing a BLOCKED/FAILED child's report and deliberately accepting its "
                "partial patch. Validate the integrated result in a later action.\n"
                "Parameters: agent_id (str, required); changeset_id (str, required, exact sha256 "
                "shown by get_sub_agent_report); accept_partial (bool, optional, default false)."
            ),
        )
        self.worker = worker
        self.llm_client = llm_client

    @property
    def output_dir(self):
        return _output_dir_for_worker(self.worker)

    @staticmethod
    def _blocked(message: str, code: str = "sub_agent_changes_blocked") -> ToolResult:
        return _change_result(
            ToolStatus.BLOCKED,
            f"COMMAND BLOCKED: {message}",
            error_code=code,
        )

    @staticmethod
    def _failed(message: str) -> ToolResult:
        return _change_result(
            ToolStatus.FAILED,
            f"COMMAND FAILED: {message}",
            error_code="sub_agent_changes_failed",
        )

    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "minLength": 4, "maxLength": 128},
                "changeset_id": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
                "accept_partial": {"type": "boolean"},
            },
            "required": ["agent_id", "changeset_id"],
            "additionalProperties": False,
        }

    def execute(
        self,
        agent_id: str,
        changeset_id: str,
        accept_partial: bool = False,
    ) -> ToolResult:
        if not self.worker:
            return self._failed("Worker context missing.")
        if not isinstance(accept_partial, bool):
            return self._failed("accept_partial must be a JSON boolean.")
        changeset_id = str(changeset_id or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", changeset_id):
            return self._failed("changeset_id must be the exact 64-character sha256 from the report.")
        agent_dir, error = _resolve_agent_dir(self.output_dir, agent_id)
        if error:
            return self._failed(error)

        terminal, status, terminal_report = resolve(agent_dir)
        normalized_status = norm_status(status)
        if not terminal:
            return _change_result(
                ToolStatus.PENDING,
                f"Sub-agent {agent_dir.name[:8]} is still running; no patch was applied.",
                error_code="sub_agent_running",
            )
        try:
            collection = read_owned_json(
                agent_dir / SUB_AGENT_REPORT_COLLECTION_RECEIPT
            )
            report_text = str(terminal_report or "N/A")
            collected = bool(
                collection.get("schema") == 1
                and collection.get("agent_id") == agent_dir.name
                and collection.get("status") == normalized_status
                and collection.get("report_chars") == len(report_text)
                and collection.get("report_sha256")
                == hashlib.sha256(report_text.encode("utf-8")).hexdigest()
            )
        except SubAgentChangeError:
            collected = False
        if not collected:
            return self._blocked(
                f"read sub-agent {agent_dir.name[:8]}'s terminal report through EOF before "
                "integrating its changes.",
                "sub_agent_report_uncollected",
            )
        lifecycle = pid_alive(agent_dir)
        if lifecycle is True:
            return _change_result(
                ToolStatus.PENDING,
                f"Sub-agent {agent_dir.name[:8]} published a terminal report but its exact "
                "process scope has not exited; no patch was applied.",
                error_code="sub_agent_still_exiting",
            )
        if lifecycle is None:
            return self._blocked(
                "the exact sub-agent process/scope absence cannot be proven.",
                "sub_agent_liveness_ambiguous",
            )
        if normalized_status != "COMPLETED" and not accept_partial:
            return self._blocked(
                f"sub-agent {agent_dir.name[:8]} ended {normalized_status}; read its report and "
                "set accept_partial=true only if its incomplete changes are intentionally wanted.",
                "partial_sub_agent_changes_require_acceptance",
            )

        try:
            binding = read_owned_json(agent_dir / MUTABLE_WORKSPACE_RECEIPT)
            changes = read_owned_json(agent_dir / MUTABLE_CHANGE_RECEIPT)
            if (
                binding.get("schema") != 1
                or binding.get("agent_id") != agent_dir.name
                or binding.get("read_only") is not False
                or changes.get("schema") != 1
                or changes.get("agent_id") != agent_dir.name
            ):
                raise SubAgentChangeError("mutable change receipt identity is invalid")

            base_commit = str(changes.get("base_commit") or "").strip().lower()
            if (
                base_commit != str(binding.get("base_commit") or "").strip().lower()
                or not re.fullmatch(r"[0-9a-f]{40,64}", base_commit)
                or changes.get("child_head") != base_commit
            ):
                raise SubAgentChangeError("mutable change receipt base/child HEAD is invalid")
            relative_workspace = validate_relative_path(
                changes.get("relative_workspace") or "."
            )
            if relative_workspace != str(binding.get("relative_workspace") or "."):
                raise SubAgentChangeError("mutable change receipt workspace changed")

            patch_name = str(changes.get("patch_file") or "")
            if patch_name != MUTABLE_PATCH_FILE:
                raise SubAgentChangeError("mutable change receipt names an unexpected patch")
            patch_size = changes.get("patch_bytes")
            patch_sha256 = str(changes.get("patch_sha256") or "").strip().lower()
            if (
                not isinstance(patch_size, int)
                or isinstance(patch_size, bool)
                or patch_size < 0
                or patch_size > MAX_PATCH_BYTES
                or not re.fullmatch(r"[0-9a-f]{64}", patch_sha256)
            ):
                raise SubAgentChangeError("mutable patch metadata is invalid")
            if changeset_id != patch_sha256:
                raise SubAgentChangeError(
                    "changeset_id does not match the child's exact immutable patch receipt"
                )
            raw_paths = changes.get("changed_paths")
            raw_changes = changes.get("path_changes")
            if (
                not isinstance(raw_paths, list)
                or not isinstance(raw_changes, list)
                or len(raw_paths) > MAX_CHANGED_PATHS
                or len(raw_changes) != len(raw_paths)
            ):
                raise SubAgentChangeError("mutable changed-path manifest is invalid")
            changed_paths = [validate_relative_path(item) for item in raw_paths]
            if len(set(changed_paths)) != len(changed_paths):
                raise SubAgentChangeError("mutable changed-path manifest contains duplicates")
            path_changes = []
            for index, raw_change in enumerate(raw_changes):
                if not isinstance(raw_change, dict):
                    raise SubAgentChangeError("mutable path-change entry is invalid")
                path = validate_relative_path(raw_change.get("path"))
                status_code = str(raw_change.get("status") or "")
                old_mode = str(raw_change.get("old_mode") or "")
                new_mode = str(raw_change.get("new_mode") or "")
                final_size = raw_change.get("final_size")
                final_sha256 = str(raw_change.get("final_sha256") or "").lower()
                if (
                    path != changed_paths[index]
                    or status_code not in {"A", "M", "D"}
                    or old_mode not in {"000000", "100644", "100755"}
                    or new_mode not in {"000000", "100644", "100755"}
                    or not isinstance(final_size, int)
                    or isinstance(final_size, bool)
                    or final_size < 0
                    or final_size > 1024 * 1024 * 1024
                ):
                    raise SubAgentChangeError("mutable path-change metadata is invalid")
                expected_modes = {
                    "A": old_mode == "000000" and new_mode in {"100644", "100755"},
                    "M": old_mode in {"100644", "100755"} and new_mode in {"100644", "100755"},
                    "D": old_mode in {"100644", "100755"} and new_mode == "000000",
                }
                if not expected_modes[status_code]:
                    raise SubAgentChangeError("mutable path status/mode transition is invalid")
                if status_code == "D":
                    if final_size != 0 or final_sha256:
                        raise SubAgentChangeError("deleted-path receipt has final content")
                elif not re.fullmatch(r"[0-9a-f]{64}", final_sha256):
                    raise SubAgentChangeError("mutable final file digest is invalid")
                path_changes.append(
                    {
                        "path": path,
                        "status": status_code,
                        "old_mode": old_mode,
                        "new_mode": new_mode,
                        "final_size": final_size,
                        "final_sha256": final_sha256,
                    }
                )
            empty = changes.get("empty")
            if (
                not isinstance(empty, bool)
                or empty != (patch_size == 0)
                or empty != (not path_changes)
            ):
                raise SubAgentChangeError("mutable empty-patch receipt is inconsistent")

            current_workspace = Path(os.getcwd()).resolve(strict=True)
            root = _run_git(
                current_workspace,
                "rev-parse",
                "--show-toplevel",
                timeout=10,
            )
            if root.returncode != 0:
                return self._blocked(
                    "the principal workspace is no longer inside the bound Git repository.",
                    "principal_repository_unavailable",
                )
            repository = Path(str(root.stdout or "").strip()).resolve(strict=True)
            expected_repository = Path(
                str(binding.get("parent_repository") or "")
            ).resolve()
            if repository != expected_repository:
                raise SubAgentChangeError("principal repository identity changed")

            principal_head = _run_git(
                repository,
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
                timeout=10,
            )
            if (
                principal_head.returncode != 0
                or str(principal_head.stdout or "").strip().lower() != base_commit
            ):
                return self._blocked(
                    "the principal HEAD no longer equals the mutable child's exact base; "
                    "inspect and port the report manually.",
                    "sub_agent_base_diverged",
                )

            admitted_artifacts = _admit_principal_change_paths(
                self.worker,
                repository,
                relative_workspace,
                path_changes,
            )
            # Preserve the same request-relative identity used by ordinary file
            # tools while also carrying canonical absolute receipts.
            absolute_artifacts = changed_paths + admitted_artifacts
            contract = getattr(self.worker, "request_contract", None)
            invariant_check = getattr(contract, "invariant_mutation_error", None)
            if callable(invariant_check):
                invariant_error = invariant_check(
                    self.policy,
                    {
                        "agent_id": agent_dir.name,
                        "changeset_id": changeset_id,
                        "accept_partial": accept_partial,
                    },
                    artifacts=absolute_artifacts,
                )
                if invariant_error:
                    return self._blocked(
                        invariant_error,
                        "sub_agent_changes_violate_invariant",
                    )
            integration_path = agent_dir / MUTABLE_INTEGRATION_RECEIPT
            if integration_path.exists():
                integrated = read_owned_json(integration_path)
                if (
                    integrated.get("schema") == 1
                    and integrated.get("agent_id") == agent_dir.name
                    and integrated.get("patch_sha256") == patch_sha256
                    and integrated.get("parent_repository") == str(repository)
                ):
                    journal_status = integrated.get("status")
                    if journal_status == "PREPARED":
                        if not _final_change_receipt_matches(repository, path_changes):
                            return self._blocked(
                                "a prior integration stopped after PREPARED and the principal "
                                "files do not match the exact final receipt; refusing replay.",
                                "sub_agent_integration_recovery_required",
                            )
                        integrated["status"] = "APPLIED"
                        integrated["recovered"] = True
                        rt.atomic_write_json(integration_path, integrated)
                        os.chmod(integration_path, 0o600)
                        return _change_result(
                            ToolStatus.OK,
                            f"Recovered sub-agent {agent_dir.name[:8]}'s exact completed "
                            "integration journal. Validate the principal worktree.",
                            changed=True,
                            evidence=[f"sha256:{patch_sha256}", f"base:{base_commit}"],
                            artifacts=absolute_artifacts,
                        )
                    if journal_status != "APPLIED":
                        raise SubAgentChangeError("integration journal status is invalid")
                    if not _final_change_receipt_matches(repository, path_changes):
                        return self._blocked(
                            "the already-integrated paths no longer match their exact receipt; "
                            "inspect current principal changes instead of replaying the patch.",
                            "integrated_sub_agent_state_changed",
                        )
                    return _change_result(
                        ToolStatus.NO_CHANGE,
                        f"NO CHANGE: sub-agent {agent_dir.name[:8]}'s exact patch already has an "
                        "integration receipt. Validate the principal worktree instead of applying it twice.",
                        error_code="already_integrated",
                        artifacts=absolute_artifacts,
                    )
                raise SubAgentChangeError("an incompatible integration receipt already exists")

            patch_path = agent_dir / MUTABLE_PATCH_FILE
            validate_patch_file(
                patch_path,
                expected_size=patch_size,
                expected_sha256=patch_sha256,
            )
            if empty:
                rt.atomic_write_json(
                    integration_path,
                    {
                        "schema": 1,
                        "agent_id": agent_dir.name,
                        "patch_sha256": patch_sha256,
                        "parent_repository": str(repository),
                        "status": "APPLIED",
                        "changed": False,
                    },
                )
                os.chmod(integration_path, 0o600)
                return _change_result(
                    ToolStatus.NO_CHANGE,
                    f"NO CHANGE: mutable sub-agent {agent_dir.name[:8]} produced an empty patch.",
                    error_code="empty_sub_agent_patch",
                )

            conflict = _principal_paths_match_base(
                repository,
                base_commit,
                path_changes,
            )
            if conflict:
                return self._blocked(conflict, "sub_agent_patch_conflict")

            check = _run_git(
                repository,
                "apply",
                "--check",
                "--binary",
                "--whitespace=nowarn",
                str(patch_path),
                timeout=60,
            )
            if check.returncode != 0:
                return self._blocked(
                    "the isolated patch conflicts with current principal files; no file was written. "
                    f"Git check: {_git_failure_detail(check)}",
                    "sub_agent_patch_conflict",
                )

            # Re-evaluate file boundaries immediately before the write, then
            # journal PREPARED. A crash after apply is recovered by comparing
            # exact final hashes; the patch is never blindly replayed.
            _admit_principal_change_paths(
                self.worker,
                repository,
                relative_workspace,
                path_changes,
            )
            rt.atomic_write_json(
                integration_path,
                {
                    "schema": 1,
                    "agent_id": agent_dir.name,
                    "patch_sha256": patch_sha256,
                    "parent_repository": str(repository),
                    "base_commit": base_commit,
                    "status": "PREPARED",
                    "changed_paths": changed_paths,
                },
            )
            os.chmod(integration_path, 0o600)

            applied = _run_git(
                repository,
                "apply",
                "--binary",
                "--whitespace=nowarn",
                str(patch_path),
                timeout=60,
            )
            if applied.returncode != 0:
                return self._failed(
                    "the patch changed between preflight and apply or Git refused it; "
                    f"the PREPARED journal prevents unsafe replay. {_git_failure_detail(applied)}"
                )

            if not _final_change_receipt_matches(repository, path_changes):
                return self._blocked(
                    "Git returned success but the exact final file hashes/modes do not match "
                    "the immutable changeset; the PREPARED journal prevents replay.",
                    "sub_agent_post_apply_verification_failed",
                )

            rt.atomic_write_json(
                integration_path,
                {
                    "schema": 1,
                    "agent_id": agent_dir.name,
                    "patch_sha256": patch_sha256,
                    "parent_repository": str(repository),
                    "base_commit": base_commit,
                    "status": "APPLIED",
                    "changed": True,
                    "changed_paths": changed_paths,
                },
            )
            os.chmod(integration_path, 0o600)
            return _change_result(
                ToolStatus.OK,
                f"Integrated sub-agent {agent_dir.name[:8]}'s verified patch across "
                f"{len(changed_paths)} path(s). Validate the result before completion.",
                changed=True,
                evidence=[f"sha256:{patch_sha256}", f"base:{base_commit}"],
                artifacts=absolute_artifacts,
            )
        except (OSError, ValueError, SubAgentChangeError) as exc:
            return self._blocked(str(exc), "invalid_sub_agent_change_receipt")


class GatherSubAgents(BaseTool):
    DEFAULT_TIMEOUT = 0
    HARD_MAX_TIMEOUT = 120
    STALL_FLAG_SECONDS = 120
    FREEZE_SECONDS = 60
    POLL_INTERVAL = 3

    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="gather_sub_agents",
            description=(
                "Snapshot check-in on your sub-agents (your graduate students). For each you get its short id, "
                "status, time since last progress, current step, and any stall/loop/freeze flag, plus a "
                "recommended action. NOTE: you ALSO see this automatically in the SUB-AGENTS section of your "
                "context every turn -- so as an advisor you should mostly be ACTING on that (steer_sub_agent, "
                "get_sub_agent_report, kill_sub_agent) and doing your own orthogonal work, not repeatedly "
                "polling here.\n"
                "By default this returns an INSTANT snapshot and does NOT block. Only pass a non-zero timeout "
                "when you have genuinely nothing else to do and want to pause until something changes; even "
                "then it returns the moment any agent finishes/freezes (capped at 120s). Do NOT use it as an "
                "idle wait loop -- supervising and doing parallel work is the whole point of dispatching them.\n"
                "Schema:\n"
                "  agent_ids (list[str], optional): specific ids; omit for all running sub-agents.\n"
                "  timeout (int, optional, default=0): 0 = instant snapshot (recommended). Non-zero = wait up "
                "to this many seconds (capped 120) for a change.\n"
                "  stall_threshold (int, optional, default=120): flag an agent showing no progress this long.\n"
                "Example: {\"tool_name\": \"gather_sub_agents\", \"parameters\": {}}"
            ),
        )
        self.worker = worker
        self.llm_client = llm_client

    @property
    def output_dir(self):
        return _output_dir_for_worker(self.worker)

    def _short_id(self, dir_name):
        return dir_name.split("-")[0]

    def _progress(self, agent_dir):
        """Delegate to the shared reader so gather and the principal's always-on
        digest never disagree about what a student is doing."""
        from aeon.core.sub_agent_state import read_progress
        return read_progress(agent_dir, freeze_seconds=self.FREEZE_SECONDS)

    def execute(self, agent_ids=None, timeout=None, stall_threshold=None):
        if not self.worker:
            return "Error: Worker context missing."
        try:
            timeout = self.DEFAULT_TIMEOUT if timeout is None else int(timeout)
        except (TypeError, ValueError):
            timeout = self.DEFAULT_TIMEOUT
        timeout = max(0, min(self.HARD_MAX_TIMEOUT, timeout))
        try:
            stall_threshold = self.STALL_FLAG_SECONDS if stall_threshold is None else int(stall_threshold)
        except (TypeError, ValueError):
            stall_threshold = self.STALL_FLAG_SECONDS

        base = self.output_dir
        if not base.exists():
            return "No sub-agents have been spawned in this session."

        missing = []
        if agent_ids:
            if isinstance(agent_ids, str):
                agent_ids = [agent_ids]
            targets = []
            for aid in agent_ids:
                d, err = _resolve_agent_dir(base, aid)
                if d:
                    targets.append(d)
                else:
                    missing.append(str(aid))
            if not targets:
                return f"None of the requested sub-agents were found: {agent_ids}"
        else:
            targets = [d for d in base.iterdir()
                       if d.is_dir() and ((d / "status.txt").exists()
                                          or (d / "output.json").exists()
                                          or (d / "pid.txt").exists())]
            if not targets:
                return "No sub-agents found to gather."

        initially_running = {d.name for d in targets if not resolve(d)[0]}
        start = time.time()
        while (time.time() - start) < timeout:
            running_now = {d.name for d in targets if not resolve(d)[0]}
            if running_now != initially_running:
                break
            if not running_now:
                break
            if any(self._progress(d)["frozen"] for d in targets if d.name in running_now):
                break
            time.sleep(self.POLL_INTERVAL)

        completed = failed = killed = stalled = frozen = looping = healthy = 0
        lines = []
        for d in targets:
            is_term, status, report = resolve(d)
            sid = self._short_id(d.name)
            if is_term:
                base_status = norm_status(status)
                if base_status == "COMPLETED":
                    completed += 1
                    lines.append(f"[{sid}] COMPLETED\n  {(report or '')[:800]}\n"
                                 f"  (full findings: get_sub_agent_report(agent_id='{sid}'))")
                elif base_status == "KILLED":
                    killed += 1
                    lines.append(f"[{sid}] KILLED")
                else:
                    failed += 1
                    lines.append(f"[{sid}] {status}\n  {(report or '')[:600]}")
                continue
            pr = self._progress(d)
            age, step, it, is_frozen, stuck = pr["age"], pr["step"], pr["iteration"], pr["frozen"], pr["stuck_reason"]
            age_str = f"{age:.0f}s ago" if age is not None else "unknown"
            sfx = (f" on '{step}'" if step else "") + (f" (iter {it})" if it else "")
            if is_frozen:
                frozen += 1
                lines.append(f"[{sid}] FROZEN - watchdog stopped responding (whole-process freeze). "
                             f"It cannot self-recover; kill_sub_agent(agent_id='{sid}') and proceed.")
            elif stuck:
                looping += 1
                lines.append(f"[{sid}] LOOPING - {stuck} It is burning budget without progress; "
                             f"steer_sub_agent(agent_id='{sid}') with a new approach, or kill_sub_agent.")
            elif age is not None and age > stall_threshold:
                stalled += 1
                lines.append(f"[{sid}] POSSIBLY STALLED - no progress for {age:.0f}s{sfx}. "
                             f"Confirm with get_sub_agent_report(agent_id='{sid}'), then steer or kill.")
            else:
                healthy += 1
                lines.append(f"[{sid}] RUNNING (healthy) - last progress {age_str}{sfx}. "
                             f"Do other orthogonal work, or gather_sub_agents again with a non-zero timeout to wait.")

        header = (f"Check-in: {completed} completed, {failed} failed, {killed} killed, "
                  f"{stalled} possibly stalled, {looping} looping, {frozen} frozen, "
                  f"{healthy} healthy & running.")
        if missing:
            header += f" (Requested but not found: {missing})"
        if frozen:
            footer = "\n\nAction: kill the FROZEN agent(s) - they cannot recover - then continue."
        elif looping:
            footer = ("\n\nAction: a LOOPING agent self-reported it is repeating itself. Steer it with a "
                      "concretely different approach, or kill it if its work is no longer needed.")
        elif stalled:
            footer = ("\n\nAction: confirm stalls with get_sub_agent_report before acting (an agent may be on "
                      "a long legitimate step). If truly stuck, steer with a corrected approach or kill.")
        elif healthy:
            footer = ("\n\nAction: agents are still running and you can see them live in your SUB-AGENTS "
                      "section each turn. Advance your OWN orthogonal work and steer them as needed; collect "
                      "each report (get_sub_agent_report) before you finish the task. Don't idle-poll.")
        else:
            footer = ""
        return header + "\n\n" + "\n\n".join(lines) + footer


class GetSubAgentReport(BaseTool):
    MAX_RESULT_CHARS = 8000

    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="get_sub_agent_report",
            description=(
                "Read a sub-agent in depth. If finished, returns its FULL findings (fold these into your "
                "synthesis and spawn follow-ups for leads it surfaced). If still running, returns a live "
                "analysis of its recent activity. Accepts the short id shown by gather_sub_agents or a full "
                "UUID. Don't call this every turn for a running agent - prefer gather_sub_agents for batch "
                "check-ins.\n"
                "Schema:\n"
                "  agent_id (str, required): short id or full UUID.\n"
                "  specific_question (str, optional): a targeted question about a running agent's progress.\n"
                "  offset (int, optional): terminal-report character offset; follow next_offset until EOF.\n"
                "Example: {\"tool_name\": \"get_sub_agent_report\", \"parameters\": {\"agent_id\": \"a44fa909\"}}"
            ),
            underlying_model=llm_client.model if llm_client else None,
        )
        self.worker = worker
        self.llm_client = llm_client

    @property
    def output_dir(self):
        return _output_dir_for_worker(self.worker)

    def execute(self, agent_id, specific_question=None, offset=0):
        agent_dir, err = _resolve_agent_dir(self.output_dir, agent_id)
        if err:
            return err

        is_term, status, report = resolve(agent_dir)
        base_status = norm_status(status)

        if is_term:
            try:
                start = int(offset)
            except (TypeError, ValueError):
                return "COMMAND FAILED: offset must be a non-negative integer."
            if start < 0:
                return "COMMAND FAILED: offset must be a non-negative integer."
            result = report or "N/A"
            if start > len(result):
                return (
                    f"COMMAND FAILED: offset {start} exceeds terminal report length "
                    f"{len(result)}."
                )
            report_sha256 = hashlib.sha256(result.encode("utf-8")).hexdigest()
            already_collected = False
            try:
                collection = read_owned_json(
                    agent_dir / SUB_AGENT_REPORT_COLLECTION_RECEIPT
                )
                already_collected = bool(
                    collection.get("schema") == 1
                    and collection.get("agent_id") == agent_dir.name
                    and collection.get("status") == base_status
                    and collection.get("report_chars") == len(result)
                    and collection.get("report_sha256") == report_sha256
                )
            except SubAgentChangeError:
                pass
            if not already_collected:
                expected_offset = 0
                try:
                    progress = read_owned_json(
                        agent_dir / SUB_AGENT_REPORT_PROGRESS_RECEIPT
                    )
                    if (
                        progress.get("schema") == 1
                        and progress.get("agent_id") == agent_dir.name
                        and progress.get("status") == base_status
                        and progress.get("report_chars") == len(result)
                        and progress.get("report_sha256") == report_sha256
                        and isinstance(progress.get("next_offset"), int)
                    ):
                        expected_offset = max(
                            0,
                            min(len(result), progress["next_offset"]),
                        )
                except SubAgentChangeError:
                    pass
                if start != expected_offset:
                    return (
                        f"COMMAND FAILED: terminal report pages must be consumed in order; "
                        f"expected offset {expected_offset}, received {start}."
                    )
            end = min(len(result), start + self.MAX_RESULT_CHARS)
            chunk = result[start:end]
            eof = end >= len(result)
            if not already_collected:
                rt.atomic_write_json(
                    agent_dir / SUB_AGENT_REPORT_PROGRESS_RECEIPT,
                    {
                        "schema": 1,
                        "agent_id": agent_dir.name,
                        "status": base_status,
                        "report_chars": len(result),
                        "report_sha256": report_sha256,
                        "next_offset": end,
                    },
                )
                os.chmod(
                    agent_dir / SUB_AGENT_REPORT_PROGRESS_RECEIPT,
                    0o600,
                )
            if eof:
                # A terminal child is resolved only after the principal has
                # consumed every bounded page; a status preview is insufficient.
                self.worker.notified_sub_agents.add(
                    f"{agent_dir.name}_{base_status}"
                )
                rt.atomic_write_json(
                    agent_dir / SUB_AGENT_REPORT_COLLECTION_RECEIPT,
                    {
                        "schema": 1,
                        "agent_id": agent_dir.name,
                        "status": base_status,
                        "report_chars": len(result),
                        "report_sha256": report_sha256,
                    },
                )
                os.chmod(
                    agent_dir / SUB_AGENT_REPORT_COLLECTION_RECEIPT,
                    0o600,
                )
            page = (
                f"report_chars={start}:{end}/{len(result)} · "
                + ("EOF (report collected)" if eof else f"next_offset={end}")
            )
            heading = "--- FINDINGS ---\n" if base_status == "COMPLETED" else ""
            change_note = ""
            if (agent_dir / MUTABLE_WORKSPACE_RECEIPT).exists():
                try:
                    changes = read_owned_json(agent_dir / MUTABLE_CHANGE_RECEIPT)
                    paths = changes.get("changed_paths")
                    if not isinstance(paths, list):
                        raise SubAgentChangeError("changed-path manifest is invalid")
                    digest = str(changes.get("patch_sha256") or "")
                    if changes.get("empty") is True:
                        change_note = (
                            "\n[MUTABLE WORKTREE] The exact terminal patch is empty; there are no "
                            "isolated edits to integrate.\n"
                        )
                    else:
                        change_note = (
                            f"\n[MUTABLE WORKTREE] Verified patch receipt: {len(paths)} path(s), "
                            f"changeset_id={digest}. After reviewing this report, apply it with "
                            "integrate_sub_agent_changes("
                            f"agent_id='{agent_dir.name[:8]}', changeset_id='{digest}') and then "
                            "validate the principal worktree.\n"
                        )
                except SubAgentChangeError as exc:
                    change_note = (
                        "\n[MUTABLE WORKTREE] No valid transferable patch receipt is available "
                        f"({exc}). Do not claim these isolated edits were integrated.\n"
                    )
            return (
                f"Agent {agent_dir.name[:8]} Status: {status}\n{page}\n\n"
                f"{change_note}{heading}{chunk}"
            )

        report_str = f"Agent {agent_dir.name[:8]} Status: RUNNING"
        log_path = agent_dir / "agent.log"
        log_tail = ""
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_tail = "".join(f.readlines()[-150:])
            except Exception as e:
                log_tail = f"(Could not read log: {e})"

        if self.llm_client and log_tail:
            prompt = (
                f"You are a principal agent monitoring a research sub-agent's progress.\n"
                f"Analyze this recent log tail from sub-agent '{agent_dir.name[:8]}'.\n"
                f"1. What concrete progress has it made recently?\n"
                f"2. Is it stuck, looping, or blocked?\n"
                f"3. Any critical errors?\n"
                f"4. Recommendation: keep waiting, steer it, or kill it?\n"
            )
            if specific_question:
                prompt += f"\nAlso answer this specific question: {specific_question}\n"
            prompt += f"\n--- RECENT LOG TAIL ---\n{log_tail}\n--- END LOG ---"
            try:
                task_create = getattr(self.llm_client, "task_completion_create", None)
                create = (
                    task_create
                    if callable(task_create)
                    else self.llm_client.client.chat.completions.create
                )
                resp = create(
                    model=self.llm_client.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                report_str += f"\n\n[LIVE PROGRESS ANALYSIS]\n{resp.choices[0].message.content}"
            except Exception as e:
                report_str += f"\n\n[LIVE PROGRESS ANALYSIS FAILED]: {e}\nRaw log tail:\n{log_tail[-1000:]}"
        elif log_tail:
            report_str += f"\n\n[RECENT LOG TAIL]\n{log_tail[-1500:]}"
        else:
            report_str += "\n\n[No log data found yet.]"

        report_str += ("\n\n[GUIDANCE] Still running. You see its live status every turn in your SUB-AGENTS "
                       "section, so don't re-poll here each turn - advance your own orthogonal work. If it is "
                       "drifting, steer_sub_agent; if its work is no longer needed, kill_sub_agent. Do not "
                       "finish the task with this agent's report uncollected.")
        return report_str


class KillSubAgent(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="kill_sub_agent",
            description=(
                "Terminate a sub-agent and its child processes when it is stuck, frozen, or no longer needed. "
                "Kills the whole process group so nothing leaks. Accepts the short id shown by "
                "gather_sub_agents or a full UUID.\n"
                "Schema:\n  agent_id (str, required): short id or full UUID.\n"
                "Example: {\"tool_name\": \"kill_sub_agent\", \"parameters\": {\"agent_id\": \"a44fa909\"}}"
            ),
        )
        self.worker = worker
        self.llm_client = llm_client

    @property
    def output_dir(self):
        return _output_dir_for_worker(self.worker)

    def execute(self, agent_id):
        agent_dir, err = _resolve_agent_dir(self.output_dir, agent_id)
        if err:
            return err

        # Already terminal? Do NOT overwrite output.json — that would destroy a
        # completed agent's findings. A status check is not report collection;
        # leave the completion guard armed and point at the report tool.
        is_term, status, _ = resolve(agent_dir)
        if is_term:
            base_status = norm_status(status)
            return (f"Sub-agent {agent_dir.name[:8]} already finished ({base_status}); nothing to kill. "
                    f"Its report is preserved — read it with get_sub_agent_report(agent_id='{agent_dir.name[:8]}').")

        try:
            signalled = terminate_sub_agent(agent_dir)
        except ProcessIdentityError as exc:
            return (f"REFUSED: could not prove that the recorded PID still belongs to sub-agent "
                    f"{agent_dir.name[:8]} ({exc}). Its state was not overwritten.")

        rt.atomic_write_json(agent_dir / "output.json", {
            "agent_id": agent_dir.name,
            "status": "KILLED",
            "result": "Terminated by the principal agent before completion.",
        })
        rt.atomic_write_text(agent_dir / "status.txt", "KILLED")
        self.worker.notified_sub_agents.add(f"{agent_dir.name}_KILLED")
        outcome = "terminated" if signalled else "had already exited"
        return f"Sub-agent {agent_dir.name[:8]} {outcome} and was marked KILLED."
