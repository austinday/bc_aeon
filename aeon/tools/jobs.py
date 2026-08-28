"""
Background command jobs: a lightweight, detached counterpart to run_command.

Where spawn_sub_agent dispatches a whole LLM-driven Worker process (heavy; on a
local model the agents serialize on the GPU), a *job* is just a tracked shell
command that runs detached so the agent never blocks on it. It mirrors the
sub-agent state machine -- a per-job state directory, atomic terminal records,
systemd service lifecycle -- but with NO model loop, so jobs are cheap and genuinely
parallel (a build, a test run, a server, a long download).

Lifecycle / IPC (all filesystem, crash-safe):
  command.txt   the exact command (for the digest and duplicate inspection)
  request.json  fixed controller input, protected read-only from the payload
  pid.txt       informational controller PID (never used to signal the workload)
  service_receipt.json
                exact unit, InvocationID, cgroup, slice, and command digest
  status.txt    RUNNING -> (KILLED on explicit kill)
  exit_code.txt the AUTHORITATIVE terminal signal, written atomically by the
                runner the instant the command exits. Its presence == terminal.
  output.log    combined stdout+stderr stream.

resolve_job() is the single source of truth for "is this job done, and how",
read both by job_output/kill_job and by the principal's always-on BACKGROUND
JOBS digest so they never disagree. Priority: an explicit KILLED status, then
exit_code.txt, then "runner process is gone but never wrote an exit code" ->
FAILED (crashed/killed externally) so a dead job never lingers as RUNNING.
"""

import os
import uuid
import time
import subprocess
from pathlib import Path

from aeon.tools.base import BaseTool
from aeon.tools.command_fleet_guard import (
    FleetCommandGuardError,
    SERVICE_CONTROLLER,
    SYSTEM_PYTHON,
    SERVICE_STOP_TIMEOUT,
    discard_prepared_sandbox_boundary,
    guard_fleet_shell_command,
    prepare_fleet_shell_boundary,
    resolve_command_cwd,
    read_sandbox_receipt,
    reconcile_sandbox_service,
    sandbox_log_text,
    scrubbed_service_controller_environment,
    stop_sandbox_service,
)
from aeon.core import runtime_signals as rt
from aeon.tools.sub_agent import _resolve_agent_dir  # generic id/prefix/substring resolver


SERVICE_RECEIPT_REF = "service_receipt.json"


def jobs_base(worker):
    instance_id = getattr(worker, "instance_id", "default")
    return Path(os.getcwd()) / "aeon_output" / instance_id / "jobs"


def read_command(job_dir):
    try:
        return (Path(job_dir) / "command.txt").read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def resolve_job(job_dir):
    """Returns (is_terminal: bool, status: str, exit_code: Optional[int])."""
    job_dir = Path(job_dir)
    status_path = job_dir / "status.txt"
    ec_path = job_dir / "exit_code.txt"

    if status_path.exists():
        try:
            recorded_status = status_path.read_text(encoding="utf-8").strip().upper()
            if recorded_status == "KILLED":
                return True, "KILLED", None
            if recorded_status == "FAILED" and (job_dir / "startup_error.txt").exists():
                return True, "FAILED (sandbox startup)", 125
        except Exception:
            pass

    if ec_path.exists():
        try:
            code = int(ec_path.read_text(encoding="utf-8").strip())
        except Exception:
            return True, "FAILED", None
        # The controller drops a marker before stopping the exact receipted
        # service on timeout, so the wrapper's eventual exit code is secondary.
        if (job_dir / "timed_out").exists():
            return True, f"TIMED OUT (exit {code})", code
        if code == 0:
            return True, "COMPLETED", 0
        return True, f"FAILED (exit {code})", code

    receipt_path = job_dir / SERVICE_RECEIPT_REF
    if receipt_path.exists():
        try:
            receipt = read_sandbox_receipt(receipt_path)
            state = reconcile_sandbox_service(receipt)
        except FleetCommandGuardError:
            return True, "FAILED (invalid service receipt)", None
        if state == "mismatch":
            return True, "FAILED (service identity mismatch)", None
        if state in {"running", "terminal"}:
            return False, "RUNNING", None
        # Unit collection can win the small race before the durable controller
        # writes exit_code. Keep it live only while that exact controller still
        # has a proc entry; its PID is informational and is never signaled.
        try:
            controller_pid = int((job_dir / "pid.txt").read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            controller_pid = 0
        if controller_pid > 1 and Path(f"/proc/{controller_pid}").exists():
            return False, "FINALIZING", None
        return True, "FAILED (controller lost terminal status)", None

    if status_path.exists():
        try:
            if status_path.read_text(encoding="utf-8").strip().upper() == "STARTING":
                return False, "STARTING", None
        except OSError:
            pass

    return False, "RUNNING", None


def status_keyword(status):
    """Collapse 'FAILED (exit 1)' / 'TIMED OUT (exit 124)' -> a single token used
    as the notified-set key and digest classifier."""
    if not status:
        return ""
    return status.replace("TIMED OUT", "TIMEOUT").split()[0].split(":")[0].strip().upper()


class RunCommandAsync(BaseTool):
    MAX_CONCURRENT = 8
    HARD_MAX_TIMEOUT = 86400  # 24h ceiling for an explicit timeout

    def __init__(self, worker=None):
        super().__init__(
            name="run_command_async",
            description=(
                "Launch a shell command in the BACKGROUND and return immediately with a job id — the "
                "lightweight way to parallelize. Use it for anything you'd otherwise block on while having "
                "other work to do: a CPU build, a test suite, or a long download. Unlike spawn_sub_agent "
                "this starts NO model loop (it's just a tracked "
                "process), so it's cheap and you can run several at once. Commands run through "
                "/home/aday/bin/fleet-low-priority inside a gated, receipt-bound user-systemd service with "
                "DevicePolicy=closed, an exact private temp directory, and the Fleet/delegation/Comfy/source "
                "guardrails read-only. Landlock independently denies non-standard devices, coordinator/control/"
                "credential reads, and out-of-grant writes. GPU visibility and inherited Fleet lease authority "
                "are removed. All socket creation is denied, including AF_UNIX, DNS, loopback, and public Internet; "
                "use reviewed browser/search/provider tools for network effects. Direct GPU/coordinator/device/lease "
                "access, every container/runtime client, "
                "privilege/scope/namespace launchers, remote execution, service/scheduler mutation, generic "
                "process signaling, shell background escapes, and recognized GPU/distributed launch forms "
                "are rejected. Opaque descendants remain service-cgroup-owned, device-blocked, unable to create "
                "sockets or non-standard devices, and unable to rewrite the guardrails. In the Aeon source cwd, "
                "only `$AEON_COMMAND_SCRATCH_DIR`/`$TMPDIR` is writable and it is removed on collection; make "
                "durable edits with the receipt-bound file tools. Submit GPU work through a "
                "reviewed Fleet Compute service or batch profile "
                "instead.\n"
                "The job appears LIVE in the BACKGROUND JOBS section of your context every turn; when it "
                "finishes or fails you see it there ONCE — then read its output with job_output. Do your own "
                "work meanwhile; never idle-poll. For a quick command whose output you need right now, use "
                "run_command instead.\n"
                "Schema:\n"
                "  command (str, required): the shell command to run detached.\n"
                "  cwd (str, optional): exact existing project directory beneath this agent's launch workspace.\n"
                "  timeout (int, optional): seconds before the exact receipted service is stopped. "
                "Omit for no time limit (e.g. a server).\n"
                "Example: {\"tool_name\": \"run_command_async\", \"parameters\": {\"command\": \"pytest -q > /dev/null\", "
                "\"timeout\": 1200}}"
            ),
        )
        self.worker = worker

    def _running_count(self, base):
        if not base.exists():
            return 0
        return sum(1 for d in base.iterdir()
                   if d.is_dir() and (d / "pid.txt").exists() and not resolve_job(d)[0])

    def execute(
        self,
        command: str = None,
        timeout: int = 0,
        cwd: str | None = None,
    ) -> str:
        if not self.worker:
            return "COMMAND FAILED: Worker context missing."
        if not command or not str(command).strip():
            return "COMMAND FAILED: 'command' is required."
        command = str(command)

        try:
            timeout = max(0, min(int(timeout), self.HARD_MAX_TIMEOUT))
        except (TypeError, ValueError):
            timeout = 0

        # Admit and verify every fixed executable/guardrail before creating
        # durable job state. The controller later verifies the actual gated unit
        # before it permits the model-requested shell to execute.
        try:
            session_root = Path.cwd().resolve(strict=True)
            command_cwd = resolve_command_cwd(cwd, session_root=session_root)
            cwd_metadata = command_cwd.stat(follow_symlinks=False)
            cwd_identity = (int(cwd_metadata.st_dev), int(cwd_metadata.st_ino))
            command = guard_fleet_shell_command(command)
            boundary, _manager_environment = prepare_fleet_shell_boundary(
                cwd=command_cwd,
                session_root=session_root,
                expected_cwd_identity=cwd_identity,
                runtime_max_seconds=(timeout + 15 if timeout > 0 else None),
            )
        except FleetCommandGuardError as exc:
            return str(exc)
        # This call is preflight-only; the durable controller creates its own
        # cryptographic service identity. Remove the unused owner-only control
        # directory without broad cleanup.
        discard_prepared_sandbox_boundary(boundary)

        base = jobs_base(self.worker)
        if self._running_count(base) >= self.MAX_CONCURRENT:
            return (f"COMMAND FAILED: {self.MAX_CONCURRENT} background jobs already running. "
                    f"Wait for some to finish (watch the BACKGROUND JOBS section), read them with "
                    f"job_output, or free one with kill_job.")

        job_id = str(uuid.uuid4())
        job_dir = base / job_id
        job_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        job_dir.chmod(0o700)

        rt.atomic_write_text(job_dir / "command.txt", command)
        rt.atomic_write_json(
            job_dir / "request.json",
            {
                "schema": 1,
                "command": command,
                "cwd": str(command_cwd),
                "timeout": timeout,
            },
        )
        rt.atomic_write_text(job_dir / "status.txt", "STARTING")
        try:
            process = subprocess.Popen(
                [
                    boundary.low_priority,
                    str(SYSTEM_PYTHON),
                    "-I",
                    str(SERVICE_CONTROLLER),
                    str(job_dir.resolve()),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=scrubbed_service_controller_environment(os.environ),
                start_new_session=True,
            )
        except Exception as e:
            rt.atomic_write_text(job_dir / "status.txt", "FAILED")
            rt.atomic_write_text(job_dir / "startup_error.txt", str(e))
            return f"COMMAND FAILED: could not launch background job: {e}"

        # Do not claim the job started until the controller has written the exact
        # unit/InvocationID receipt and opened the service gate. Fast commands may
        # also reach a terminal record within this bounded startup window.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            status = ""
            try:
                status = (job_dir / "status.txt").read_text(encoding="utf-8").strip().upper()
            except OSError:
                pass
            if (job_dir / SERVICE_RECEIPT_REF).exists() and status == "RUNNING":
                break
            if status == "FAILED" or (job_dir / "exit_code.txt").exists():
                error = ""
                try:
                    error = (job_dir / "startup_error.txt").read_text(encoding="utf-8").strip()
                except OSError:
                    pass
                return f"COMMAND FAILED: transient-service startup failed{': ' + error if error else '.'}"
            if process.poll() is not None:
                return "COMMAND FAILED: transient-service controller exited before a verified receipt."
            time.sleep(0.02)
        else:
            # A timed-out startup is a cancellation, not permission for a late
            # controller to open the payload gate after this tool has returned.
            # The controller checks this protected marker before and immediately
            # after launch; if a receipt already exists, stop only that exact
            # unit/InvocationID. Never signal the informational controller PID.
            rt.atomic_write_text(job_dir / "cancel_startup", "cancel")
            cleanup_deadline = time.monotonic() + 25.0
            cleanup_error = ""
            while time.monotonic() < cleanup_deadline:
                receipt_path = job_dir / SERVICE_RECEIPT_REF
                if receipt_path.exists():
                    try:
                        receipt = read_sandbox_receipt(receipt_path)
                        stop_sandbox_service(receipt)
                    except FleetCommandGuardError as exc:
                        cleanup_error = str(exc)
                if process.poll() is not None or (job_dir / "controller_done").exists():
                    break
                time.sleep(0.05)
            if process.poll() is None and not (job_dir / "controller_done").exists():
                return (
                    "COMMAND FAILED: startup cancellation could not prove the trusted "
                    "controller had exited; the payload gate remains fail-closed."
                )
            if cleanup_error:
                return (
                    "COMMAND FAILED: startup was cancelled, but exact service "
                    f"reconciliation failed ({cleanup_error})."
                )
            return "COMMAND FAILED: timed out; startup was cancelled and reconciled."

        short = job_id[:8]
        tmo = f"{timeout}s timeout" if timeout > 0 else "no timeout"
        return (f"Background job started ({tmo}). Job ID: {job_id} (refer to it as '{short}'). "
                f"It now appears LIVE in the BACKGROUND JOBS section of your context every turn; when it "
                f"finishes or fails you'll see it there — read its output with job_output(job_id='{short}'). "
                f"Advance other work meanwhile; kill_job(job_id='{short}') to stop it.")


class JobOutput(BaseTool):
    MAX_RETURN_CHARS = 12000

    def __init__(self, worker=None):
        super().__init__(
            name="job_output",
            description=(
                "Read a background job's captured output and status. If the job has finished you get its "
                "exit status plus the full (head+tail-truncated) output; if it's still running you get the "
                "output so far. Accepts the short id shown in the BACKGROUND JOBS section or a full id. "
                "Don't poll a running job every turn — you can already see its live status in that section.\n"
                "Schema:\n  job_id (str, required): short id or full UUID.\n"
                "Example: {\"tool_name\": \"job_output\", \"parameters\": {\"job_id\": \"a44fa909\"}}"
            ),
        )
        self.worker = worker

    def execute(self, job_id: str = None) -> str:
        if not job_id:
            return "Error: 'job_id' is required."
        job_dir, err = _resolve_agent_dir(jobs_base(self.worker), job_id)
        if err:
            return err

        is_term, status, _ = resolve_job(job_dir)
        short = job_dir.name[:8]
        cmd = read_command(job_dir)

        log_path = job_dir / "output.log"
        output = ""
        if log_path.exists():
            try:
                output = sandbox_log_text(log_path)
            except Exception as e:
                output = f"(could not read output.log: {e})"
        if len(output) > self.MAX_RETURN_CHARS:
            from aeon.core.worker_utils import truncate_output
            output = truncate_output(output, max_chars=self.MAX_RETURN_CHARS)

        if is_term:
            # Mark collected so the always-on digest stops re-flagging it.
            if self.worker is not None:
                self.worker.notified_jobs.add(f"{job_dir.name}_{status_keyword(status)}")
            body = output if output.strip() else "(no output)"
            return f"Job {short} [{status}]  `{cmd}`\n\n--- OUTPUT ---\n{body}"

        body = output if output.strip() else "(no output yet)"
        return (f"Job {short} [RUNNING]  `{cmd}`\n\n--- OUTPUT SO FAR ---\n{body}\n\n"
                f"[Still running — you see its live status each turn in BACKGROUND JOBS, so advance other "
                f"work rather than re-polling. kill_job(job_id='{short}') to stop it.]")


class KillJob(BaseTool):
    def __init__(self, worker=None):
        super().__init__(
            name="kill_job",
            description=(
                "Stop the exact receipt-validated transient service for a background job. systemd applies "
                "TERM then bounded KILL to the entire service cgroup; no numeric workload PID is signaled. "
                "Accepts the short id shown in BACKGROUND JOBS or a full id.\n"
                "Schema:\n  job_id (str, required): short id or full UUID.\n"
                "Example: {\"tool_name\": \"kill_job\", \"parameters\": {\"job_id\": \"a44fa909\"}}"
            ),
        )
        self.worker = worker

    def execute(self, job_id: str = None) -> str:
        if not job_id:
            return "Error: 'job_id' is required."
        job_dir, err = _resolve_agent_dir(jobs_base(self.worker), job_id)
        if err:
            return err

        short = job_dir.name[:8]
        if resolve_job(job_dir)[0]:
            return f"Job {short} is already finished. Read it with job_output(job_id='{short}')."

        try:
            receipt = read_sandbox_receipt(job_dir / SERVICE_RECEIPT_REF)
            stop_sandbox_service(receipt)
        except FleetCommandGuardError as exc:
            return (
                f"REFUSED: could not prove and stop the exact transient service for "
                f"job {short} ({exc}). Its state was not overwritten."
            )
        cleanup_deadline = time.monotonic() + SERVICE_STOP_TIMEOUT
        while time.monotonic() < cleanup_deadline:
            if (
                (job_dir / "controller_done").exists()
                and not Path(receipt.control_dir).exists()
                and not Path(receipt.scratch_dir).exists()
            ):
                break
            time.sleep(0.02)
        rt.atomic_write_text(job_dir / "status.txt", "KILLED")
        if self.worker is not None:
            self.worker.notified_jobs.add(f"{job_dir.name}_KILLED")
        return f"Job {short} stopped using its exact unit/InvocationID receipt and marked KILLED."
