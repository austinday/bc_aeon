import argparse
import ctypes
import json
import os
import re
import signal
import stat
import sys
import threading
import time
from pathlib import Path, PurePosixPath

from aeon.core import runtime_signals as rt
from aeon.core.agent_protocol import ExecutionState, RunOutcome, SideEffect
from aeon.core.fleet_backend import FleetBackendError, validate_loopback_endpoint
from aeon.core.llm import LLMClient
from aeon.core.skills.manager import INSTANCE_SKILLS_DIR_ENV
from aeon.core.sub_agent_changes import (
    SubAgentChangeError,
    snapshot_mutable_changes,
)
from aeon.core.sub_agent_environment import (
    CHILD_FLEET_CONFIGURATION_KEYS,
    VERIFICATION_PREBOUND_NONCE_ENV,
    VERIFICATION_PREBOUND_RECEIPT,
    SubAgentFleetCompute,
    scrub_principal_capabilities,
)
from aeon.core.sub_agent_state import (
    CPU_SANDBOX_SLICE_ENV,
    ProcessIdentityError,
    sub_agent_systemd_units,
)
from aeon.core.utils.io import read_bounded_fd
from aeon.core.worker import Worker
from aeon.remote.mcp_capability import (
    MCP_DELEGATION_ID_ENV,
    MCP_DELEGATION_TOKEN_FILE_ENV,
    MCP_URL_ENV,
    validate_mcp_base_endpoint,
)
from aeon.tools.loader import load_tools_from_directory

# Tools a sub-agent is NOT allowed to have: no recursive spawning (runaway GPU
# oversubscription) and no self-modification/restart of the framework.
SUB_AGENT_FORBIDDEN_TOOLS = {
    "spawn_sub_agent",
    "gather_sub_agents",
    "get_sub_agent_report",
    "integrate_sub_agent_changes",
    "kill_sub_agent",
    "steer_sub_agent",
    "get_sub_agent_status",
    "verify_self_modification",
    "restart_aeon",
    # A sub-agent has its own short-lived objective; it must not resume the
    # principal's interrupted session.
    "resume_previous_session",
    # No detached background jobs either: a sub-agent is short-lived and exists to
    # return a report, so a job that would outlive it (and whose workload group
    # would orphan when the wrapper exits) has no owner. Use run_command instead.
    "run_command_async",
    "job_output",
    "kill_job",
    # No interactive user: stdin is detached (DEVNULL) and the terminal belongs
    # to the principal. A sub-agent must report blockers in its final report,
    # not sit waiting for input that can never arrive.
    "get_user_input",
    # Durable standalone agents are owned by the primary Nexus Project Manager,
    # never by one of its bounded nested sub-agents.
    "start_agent_instance",
    "create_collaboration_portal",
    "send_collaborator_handoff",
    # This capability belongs to the durable principal Nexus tab. A nested,
    # short-lived sub-agent must never rewrite its principal's Job Role.
    "set_job_role",
    # Bounded children may use explicitly delegated MCP accounts, but they cannot
    # create a durable shared connection or recursively delegate it.
    "connect_mcp_account",
}

_VERIFICATION_NONCE_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_SELF_CGROUP = Path("/proc/self/cgroup")


def _enter_sub_agent_workspace(value: str) -> tuple[Path, tuple[int, int]]:
    """Bind every child capability to one canonical workspace before init."""

    try:
        workspace = Path(value).resolve(strict=True)
        metadata = workspace.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError("sub-agent workspace is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("sub-agent workspace is not a directory")
    os.chdir(workspace)
    try:
        entered = Path.cwd().resolve(strict=True)
        entered_metadata = entered.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError("sub-agent workspace could not be entered") from exc
    expected_identity = (int(metadata.st_dev), int(metadata.st_ino))
    entered_identity = (int(entered_metadata.st_dev), int(entered_metadata.st_ino))
    if entered != workspace or entered_identity != expected_identity:
        raise RuntimeError("sub-agent workspace identity changed during entry")
    return workspace, expected_identity


def _bind_private_skill_overlay(agent_id: str, output_dir: Path) -> Path:
    """Bind runtime learning to this child's exact durable output directory."""

    directory = Path(output_dir).expanduser().absolute()
    if directory.name != str(agent_id):
        raise RuntimeError("sub-agent output directory does not match its UUID")
    try:
        metadata = directory.lstat()
    except OSError as exc:
        raise RuntimeError("sub-agent output directory is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or directory.resolve(strict=True) != directory
    ):
        raise RuntimeError("sub-agent output directory is not owner-private")

    overlay = directory / "skills"
    try:
        overlay.mkdir(mode=0o700, exist_ok=True)
        overlay_metadata = overlay.lstat()
        if (
            not stat.S_ISDIR(overlay_metadata.st_mode)
            or overlay_metadata.st_uid != os.geteuid()
            or overlay.resolve(strict=True) != overlay.absolute()
        ):
            raise RuntimeError("sub-agent skill overlay is not owner-private")
        os.chmod(overlay, 0o700, follow_symlinks=False)
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError("sub-agent skill overlay is unavailable") from exc
    os.environ[INSTANCE_SKILLS_DIR_ENV] = str(overlay)
    return overlay


def _scrub_inherited_principal_environment(
    agent_id: str | None = None, output_dir: str | None = None
) -> None:
    """Apply the capability scrub while preserving only our generated slice.

    The generic scrub always removes inherited launcher capabilities. A schema-2
    launcher may restore its slice only when it is exactly the leaf name derived
    from this wrapper's canonical agent UUID. A verification nonce is restored
    only alongside that exact slice and still requires its one-use receipt and
    live cgroup membership below.
    """

    supplied_slice = os.environ.get(CPU_SANDBOX_SLICE_ENV)
    supplied_verification_nonce = os.environ.get(VERIFICATION_PREBOUND_NONCE_ENV)
    supplied_mcp_url = os.environ.get(MCP_URL_ENV)
    supplied_delegation_id = os.environ.get(MCP_DELEGATION_ID_ENV)
    supplied_delegation_token = os.environ.get(MCP_DELEGATION_TOKEN_FILE_ENV)
    scrub_principal_capabilities(os.environ)
    supplied_mcp_values = (
        supplied_mcp_url,
        supplied_delegation_id,
        supplied_delegation_token,
    )
    if any(value is not None for value in supplied_mcp_values) and (
        agent_id is not None or output_dir is not None
    ):
        if (
            agent_id is None
            or output_dir is None
            or any(not value for value in supplied_mcp_values)
            or supplied_delegation_id != agent_id
        ):
            raise RuntimeError("sub-agent MCP delegation is incomplete or mismatched")
        approved_url = validate_mcp_base_endpoint(str(supplied_mcp_url))
        expected_token = (Path(output_dir).resolve() / "mcp-delegation.token")
        supplied_token = Path(str(supplied_delegation_token))
        try:
            if supplied_token.resolve(strict=True) != expected_token:
                raise RuntimeError("sub-agent MCP delegation token path is mismatched")
            metadata = supplied_token.lstat()
        except OSError as exc:
            raise RuntimeError("sub-agent MCP delegation token is unavailable") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            raise RuntimeError("sub-agent MCP delegation token is not owner-safe")
        os.environ.update(
            {
                MCP_URL_ENV: approved_url,
                MCP_DELEGATION_ID_ENV: agent_id,
                MCP_DELEGATION_TOKEN_FILE_ENV: str(expected_token),
            }
        )
    if supplied_slice is None:
        return
    if agent_id is None:
        return
    try:
        _scope_unit, expected_slice = sub_agent_systemd_units(agent_id)
    except ProcessIdentityError as exc:
        raise RuntimeError("legacy wrapper received a schema-2 slice capability") from exc
    if supplied_slice != expected_slice:
        raise RuntimeError("sub-agent CPU sandbox slice does not match its UUID")
    os.environ[CPU_SANDBOX_SLICE_ENV] = expected_slice
    if supplied_verification_nonce is not None:
        os.environ[VERIFICATION_PREBOUND_NONCE_ENV] = supplied_verification_nonce


def _current_unified_cgroup() -> PurePosixPath:
    """Read this process's one cgroup-v2 membership with a fixed size bound."""

    try:
        raw = _SELF_CGROUP.read_bytes()
    except OSError as exc:
        raise RuntimeError("verification scope membership is unreadable") from exc
    if len(raw) > 16 * 1024:
        raise RuntimeError("verification scope membership exceeded its read bound")
    memberships = []
    for line in raw.decode("utf-8", "strict").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            memberships.append(parts[2])
    if len(memberships) != 1:
        raise RuntimeError("verification child lacks one cgroup-v2 membership")
    path = PurePosixPath(memberships[0])
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise RuntimeError("verification child cgroup membership is unsafe")
    return path


def _consume_prebound_verification_capability(args, config) -> str | None:
    """Consume a one-launch verifier receipt and return its pinned endpoint.

    This is deliberately not an argparse/model-facing flag.  The parent creates
    a fresh UUID scope plus a 256-bit nonce receipt after it owns the Fleet
    ticket.  The child gets only that nonce and the exact loopback endpoint; no
    broker socket, ticket, claim, or device authority crosses this boundary.
    """

    nonce = os.environ.pop(VERIFICATION_PREBOUND_NONCE_ENV, None)
    receipt_path = Path(args.output_dir) / VERIFICATION_PREBOUND_RECEIPT
    if nonce is None:
        if receipt_path.exists() or receipt_path.is_symlink():
            raise RuntimeError("verification Fleet receipt lacks its launch nonce")
        return None
    if not _VERIFICATION_NONCE_RE.fullmatch(nonce):
        raise RuntimeError("verification Fleet launch nonce is invalid")

    scope_unit, slice_unit = sub_agent_systemd_units(args.agent_id)
    if os.environ.get(CPU_SANDBOX_SLICE_ENV) != slice_unit:
        raise RuntimeError("prebound verification escaped its exact leaf slice")
    membership = _current_unified_cgroup()
    if tuple(membership.parts[-2:]) != (slice_unit, scope_unit):
        raise RuntimeError("prebound verification escaped its exact systemd scope")

    descriptor = -1
    try:
        descriptor = os.open(
            receipt_path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        raw = read_bounded_fd(descriptor, 4096)
    except OSError as exc:
        raise RuntimeError("verification Fleet receipt is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or len(raw) > 4096
    ):
        raise RuntimeError("verification Fleet receipt failed identity validation")
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("verification Fleet receipt is malformed") from exc
    expected_keys = {
        "schema", "kind", "agent_id", "scope_unit", "slice_unit",
        "endpoint", "nonce",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise RuntimeError("verification Fleet receipt has an unexpected schema")
    if (
        receipt.get("schema") != 1
        or receipt.get("kind") != "aeon-verification-prebound-fleet"
        or receipt.get("agent_id") != args.agent_id
        or receipt.get("scope_unit") != scope_unit
        or receipt.get("slice_unit") != slice_unit
        or receipt.get("nonce") != nonce
    ):
        raise RuntimeError("verification Fleet receipt does not match this launch")
    try:
        endpoint = validate_loopback_endpoint(receipt.get("endpoint"))
        configured_endpoint = validate_loopback_endpoint(config.get("base_url"))
    except FleetBackendError as exc:
        raise RuntimeError("verification Fleet endpoint is not exact loopback") from exc
    if (
        config.get("provider") != "vllm"
        or endpoint != configured_endpoint
    ):
        raise RuntimeError("verification Fleet endpoint/config identity mismatch")

    try:
        current = receipt_path.lstat()
        if (
            current.st_dev != metadata.st_dev
            or current.st_ino != metadata.st_ino
            or not stat.S_ISREG(current.st_mode)
        ):
            raise RuntimeError("verification Fleet receipt changed before consumption")
        receipt_path.unlink()
    except OSError as exc:
        raise RuntimeError("verification Fleet receipt could not be consumed") from exc
    for key in CHILD_FLEET_CONFIGURATION_KEYS:
        os.environ.pop(key, None)
    return endpoint


def _install_launcher_parent_death_signal() -> None:
    """Make a receipted wrapper die if its exact systemd-run parent disappears."""

    if CPU_SANDBOX_SLICE_ENV not in os.environ:
        return  # compatibility path for synchronous legacy verification wrappers
    parent_pid = os.getppid()
    if parent_pid <= 1:
        raise RuntimeError("systemd-run launcher disappeared before wrapper startup")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "could not install wrapper parent-death signal")
    if os.getppid() != parent_pid:
        os._exit(1)


def _release_browser_profile(agent_id):
    """Best-effort: tell the browser service to close this sub-agent's isolated
    context so its Chrome doesn't linger after the task ends. No-op if the sub
    agent never browsed (service not running / profile never created)."""
    try:
        import requests

        from aeon.tools.browser import BROWSER_API_URL, browser_auth_headers
        with requests.Session() as browser_http:
            browser_http.trust_env = False
            browser_http.post(
                f"{BROWSER_API_URL}/release_profile",
                json={"profile": f"agent-{agent_id}"},
                headers=browser_auth_headers(),
                timeout=5,
                allow_redirects=False,
            )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Aeon Sub-Agent Wrapper")
    parser.add_argument("--agent_id", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--stall_timeout", type=int, default=600,
                        help="Hard-terminate if no liveness signal for this many seconds.")
    parser.add_argument("--max_wallclock", type=int, default=2400,
                        help="Absolute wall-clock cap in seconds.")
    parser.add_argument("--read_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _scrub_inherited_principal_environment(args.agent_id, args.output_dir)
    _install_launcher_parent_death_signal()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _bind_private_skill_overlay(args.agent_id, output_dir)
    output_path = output_dir / "output.json"
    status_path = output_dir / "status.txt"
    log_path = output_dir / "agent.log"
    telemetry_path = output_dir / "telemetry.json"
    progress_path = output_dir / "progress.json"
    steering_path = output_dir / "steering.jsonl"
    steering_offset_path = output_dir / ".steering_offset"

    def log(message):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line, flush=True)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    # ----- liveness / watchdog state -----
    done_event = threading.Event()
    watchdog_terminating = threading.Event()
    rt.reset()
    started_at = time.time()
    current_step = {"iteration": 0, "step": "initializing", "stuck_reason": None}
    fleet_compute = None

    # Only ever group-kill if WE are the group leader. verify_self_modification
    # launches this wrapper inside the PRIMARY's process group; a killpg there
    # would kill the primary. In that (non-leader) case fall back to exiting self.
    try:
        is_group_leader = (os.getpgrp() == os.getpid())
    except Exception:
        is_group_leader = False

    def write_terminal(status_value, payload):
        # Durable record FIRST (output.json is never deleted and is read first by
        # the principal via resolve()), THEN the status file. Invariant: a terminal
        # status always implies the result/error is already on disk.
        try:
            rt.atomic_write_json(output_path, payload)
        except Exception as e:
            log(f"failed to write output.json: {e}")
        try:
            rt.atomic_write_text(status_path, status_value)
        except Exception as e:
            log(f"failed to write status.txt: {e}")

    def publish_progress():
        try:
            rt.atomic_write_json(progress_path, {
                "agent_id": args.agent_id,
                "alive": True,
                "started_at": started_at,
                "updated_at": time.time(),
                "activity_age": round(rt.activity_age(), 1),
                "wallclock": round(time.time() - started_at, 1),
                "iteration": current_step["iteration"],
                "step": current_step["step"],
                "stuck_reason": current_step.get("stuck_reason"),
            })
        except Exception:
            pass

    def release_fleet_compute(reason):
        """Best-effort exact child-ticket release; safe to call repeatedly."""

        nonlocal fleet_compute
        compute = fleet_compute
        if compute is None:
            return True
        try:
            proof = compute.close()
            if compute.required:
                state = proof.get("state") if isinstance(proof, dict) else "released"
                log(f"Fleet compute demand {state} during {reason}.")
            fleet_compute = None
            return True
        except Exception as exc:
            # Broker tickets are expiring durable demand. If exact release cannot
            # be confirmed during a forced exit, its TTL remains the fail-safe;
            # never guess at, or release, another consumer's ticket.
            log(f"Fleet compute release deferred to exact ticket TTL during {reason}: {exc}")
            return False

    def handle_termination(signum, _frame):
        """Convert catchable termination into durable state and exact cleanup."""

        done_event.set()
        current_step["step"] = "terminating"
        reason = signal.Signals(signum).name
        log(f"Received {reason}; stopping sub-agent with resource cleanup.")
        write_terminal("CANCELLED", {
            "agent_id": args.agent_id,
            "status": "CANCELLED",
            "error": f"Sub-agent stopped by {reason}.",
            "last_step": dict(current_step),
        })
        # Signal handlers only request cancellation; the outer finally performs
        # broker I/O after Python has unwound the interrupted operation.
        if fleet_compute is not None:
            fleet_compute.request_stop()
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGINT, handle_termination)

    def watchdog():
        # Daemon thread. Heartbeats progress.json so the principal can tell this
        # agent is alive (and detect a whole-process freeze via the file mtime),
        # and HARD-terminates on stall or wall-clock breach so the principal is
        # never blocked indefinitely. os._exit/killpg work even if the main thread
        # is wedged in a C call -- the canonical hang case.
        publish_progress()
        while not done_event.wait(timeout=5):
            age = rt.activity_age()
            wall = time.time() - started_at
            stalled = age > args.stall_timeout
            expired = wall > args.max_wallclock
            if not (stalled or expired):
                publish_progress()
                continue
            reason = (f"no progress for {age:.0f}s (stall_timeout={args.stall_timeout}s)"
                      if stalled else
                      f"exceeded wall-clock budget {wall:.0f}s (max_wallclock={args.max_wallclock}s)")
            log(f"[WATCHDOG] Hard-terminating sub-agent: {reason}")
            watchdog_terminating.set()
            write_terminal(f"FAILED: watchdog - {reason}", {
                "agent_id": args.agent_id,
                "status": "FAILED",
                "error": f"Watchdog terminated the sub-agent: {reason}",
                "note": ("Stalled or over budget; force-terminated so the principal agent is "
                         "never blocked. Re-spawn with a larger time_budget_minutes if the task "
                         "is legitimately long, or refine the objective if it got stuck."),
                "last_step": dict(current_step),
            })
            # Cancel broker admission waits or release the active exact ticket
            # before the non-catchable group kill. A broker/network failure still
            # leaves only this ticket's bounded TTL; it never widens authority.
            release_fleet_compute("watchdog termination")
            _release_browser_profile(args.agent_id)
            try:
                sys.stderr.flush()
            except Exception:
                pass
            if is_group_leader:
                try:
                    os.killpg(os.getpgrp(), signal.SIGKILL)  # self + all command children
                except Exception:
                    os._exit(1)
            else:
                os._exit(1)

    threading.Thread(target=watchdog, daemon=True, name="aeon-subagent-watchdog").start()

    def consume_steering():
        # Queued, ordered, never-lost/duplicated steering: read only the bytes
        # appended since our last offset, then advance the offset.
        if not steering_path.exists():
            return []
        try:
            offset = 0
            if steering_offset_path.exists():
                try:
                    offset = int(steering_offset_path.read_text().strip())
                except Exception:
                    offset = 0
            with open(steering_path, "r", encoding="utf-8") as f:
                f.seek(offset)
                chunk = f.read()
                new_offset = f.tell()
            rt.atomic_write_text(steering_offset_path, str(new_offset))
            messages = []
            for line in chunk.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line).get("guidance", line))
                except Exception:
                    messages.append(line)
            return messages
        except Exception as e:
            log(f"steering read failed: {e}")
            return []

    worker = None  # bound below; update_telemetry reads it as a free variable

    def update_telemetry(iteration, display_max, step_description):
        # Heartbeat + progress checkpoint + steering injection point, fired by the
        # worker at every iteration.
        rt.touch()
        current_step["iteration"] = iteration
        current_step["step"] = step_description
        # Surface the sub-agent's OWN loop-detector to the principal: a student
        # spinning on an identical command keeps touching the heartbeat, so it
        # never looks "stalled" -- this is how the principal learns it's looping.
        current_step["stuck_reason"] = getattr(worker, "stuck_reason", None)
        try:
            rt.atomic_write_json(telemetry_path, {
                "agent_id": args.agent_id,
                "iteration": iteration,
                "current_step": step_description,
                "timestamp": time.time(),
            })
        except Exception as e:
            log(f"telemetry write failed: {e}")
        publish_progress()
        if worker is not None:
            for guidance in consume_steering():
                worker.last_observation = (
                    f"[STEERING GUIDANCE FROM PRINCIPAL AGENT] {guidance}\n\n"
                    f"{worker.last_observation}"
                )
                log(f"applied steering guidance: {guidance[:120]}")

    try:
        # Enter before constructing LLMClient, Worker, loggers, or tools. Worker
        # and file capabilities intentionally capture cwd once; a late chdir made
        # mutable children run shell commands in the detached worktree while file
        # tools silently edited the principal tree.
        workspace, _workspace_identity = _enter_sub_agent_workspace(args.workspace)
        log(f"Bound sub-agent capabilities to workspace: {workspace}")
        log(f"Initializing sub-agent {args.agent_id}...")
        config = json.loads(args.model_config)
        if not isinstance(config, dict):
            raise ValueError("model_config must decode to an object")
        prebound_verification_endpoint = _consume_prebound_verification_capability(
            args, config
        )

        def fleet_wait_heartbeat():
            rt.touch()
            current_step["step"] = "waiting for Fleet Compute"
            publish_progress()

        if prebound_verification_endpoint is None:
            fleet_compute = SubAgentFleetCompute(
                agent_id=args.agent_id,
                model_config=config,
                wait_callback=fleet_wait_heartbeat,
            )
            if fleet_compute.required:
                current_step["step"] = "requesting Fleet Compute"
                publish_progress()
                log(
                    "Validating broker-only compute and requesting an independent "
                    f"Fleet demand as {fleet_compute.consumer}."
                )
                fleet_compute.start()
        else:
            # The trusted verifier parent owns/renews/releases this demand and
            # supervises our exact scope.  This process receives only its pinned
            # endpoint and can neither inspect nor mutate broker state.
            os.environ["AEON_VISION_BASE_URL"] = prebound_verification_endpoint
            log("Using the verifier parent's exact prebound Fleet endpoint.")
        llm_client = LLMClient(config)

        worker = Worker(llm_client=llm_client, debug_mode=args.debug)
        if fleet_compute is not None:
            fleet_compute.bind(llm_client=llm_client, worker=worker)
        elif prebound_verification_endpoint is not None:
            worker.compute_guard = lambda: validate_loopback_endpoint(
                prebound_verification_endpoint
            )
        worker.model_name = config.get("model", "unknown")
        worker.model_config = config
        # Read-only children may resolve to the principal workspace; mutable
        # children use a detached worktree. In both cases their short-lived task
        # state must not share the principal's session checkpoint namespace.
        worker.persist_session = False
        worker.read_only = bool(args.read_only)
        if args.read_only:
            worker.forced_request_mode = "inspect"
        # Browse as an ISOLATED identity: each sub-agent gets its own browser
        # context (own cookies/session/fingerprint) instead of sharing the
        # principal's profile, so parallel agents don't collide.
        worker.browser_profile = f"agent-{args.agent_id}"

        tools = load_tools_from_directory(
            "aeon.tools", dependencies={"llm_client": llm_client, "worker": worker}
        )
        tools = [t for t in tools if getattr(t, "name", None) not in SUB_AGENT_FORBIDDEN_TOOLS]
        if args.read_only:
            # Defense in depth beyond the Worker contract: do not even register
            # fixed project/external/destructive mutation capabilities. Keep
            # run_command because each concrete command is classified at runtime.
            tools = [
                tool for tool in tools
                if getattr(tool, "name", "") == "run_command"
                or getattr(getattr(tool, "policy", None), "side_effect", None)
                in {SideEffect.READ_ONLY, SideEffect.AGENT_STATE, SideEffect.CONTROL}
            ]
        worker.register_tools(tools)

        log(f"Starting execution of objective: {args.objective}")

        rt.atomic_write_text(status_path, "RUNNING")
        publish_progress()

        default_instruction = (
            "When you finish, provide a detailed, informative report of your findings, actions "
            "taken, and final result. This report will be read by the principal agent."
        )
        objective = f"{default_instruction}\n\n{args.objective}"

        outcome = worker.run(
            objective,
            max_iterations=args.max_iterations,
            step_callback=update_telemetry,
        )

        done_event.set()  # stand the watchdog down BEFORE the final writes
        workspace_changes = None
        workspace_change_error = ""
        if not args.read_only:
            current_step["step"] = "capturing isolated workspace changes"
            try:
                workspace_changes = snapshot_mutable_changes(
                    workspace,
                    output_dir,
                    args.agent_id,
                )
                log(
                    "Captured immutable mutable-worktree receipt: "
                    f"{len(workspace_changes.get('changed_paths') or [])} path(s), "
                    f"{workspace_changes.get('patch_bytes', 0)} patch bytes."
                )
            except (OSError, SubAgentChangeError) as exc:
                workspace_change_error = str(exc)[:1000]
                log(f"Failed to capture mutable-worktree receipt: {workspace_change_error}")
        # The report the principal reads: prefer the sub-agent's final say_to_user
        # message (the instructed deliverable). last_observation is the fallback,
        # but on the terminal turn it still holds the PREVIOUS turn's output — the
        # say_to_user text itself is never echoed into it.
        final_report = (
            getattr(outcome, "message", "")
            or getattr(worker, "last_say_to_user", None)
            or worker.last_observation
        )
        truthful_status = "COMPLETED"
        if isinstance(outcome, RunOutcome) and not outcome.completed:
            truthful_status = (
                "FAILED" if outcome.state == ExecutionState.FAILED else "BLOCKED"
            )
        if workspace_change_error and truthful_status == "COMPLETED":
            # A mutable child is not deliverable until the principal has an exact,
            # bounded patch receipt. Preserve its prose report, but never call an
            # unexportable detached-worktree result complete.
            truthful_status = "BLOCKED"
        write_terminal(truthful_status, {
            "agent_id": args.agent_id,
            "status": truthful_status,
            "execution_state": (
                outcome.state.value if isinstance(outcome, RunOutcome) else "done"
            ),
            "result": final_report,
            "plan": worker.current_plan,
            "memories": worker.memories,
            "workspace_changes": workspace_changes,
            "workspace_change_error": workspace_change_error,
        })
        log(
            "Task completed with verified terminal state."
            if truthful_status == "COMPLETED"
            else f"Task ended without completion: {truthful_status}."
        )

    except Exception as e:
        done_event.set()
        log(f"CRITICAL ERROR: {e}")
        if not watchdog_terminating.is_set():
            write_terminal(f"FAILED: {e}", {
                "agent_id": args.agent_id,
                "status": "FAILED",
                "error": str(e),
            })
        sys.exit(1)
    finally:
        done_event.set()
        release_fleet_compute("wrapper finalization")
        _release_browser_profile(args.agent_id)


if __name__ == "__main__":
    main()
