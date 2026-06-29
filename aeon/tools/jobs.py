"""
Background command jobs: a lightweight, detached counterpart to run_command.

Where spawn_sub_agent dispatches a whole LLM-driven Worker process (heavy; on a
local model the agents serialize on the GPU), a *job* is just a tracked shell
command that runs detached so the agent never blocks on it. It mirrors the
sub-agent state machine -- a per-job state directory, atomic terminal records,
process-group kill -- but with NO model loop, so jobs are cheap and genuinely
parallel (a build, a test run, a server, a long download).

Lifecycle / IPC (all filesystem, crash-safe):
  command.txt   the exact command (for the digest and duplicate inspection)
  cmd.sh        the command body the runner executes
  pid.txt       runner PID (group leader; killable as a tree)
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
import sys
import uuid
import time
import signal
import ctypes
import subprocess
from pathlib import Path

from aeon.tools.base import BaseTool
from aeon.core import runtime_signals as rt
from aeon.core.sub_agent_state import group_kill
from aeon.tools.sub_agent import _resolve_agent_dir  # generic id/prefix/substring resolver


def jobs_base(worker):
    instance_id = getattr(worker, "instance_id", "default")
    return Path(os.getcwd()) / "aeon_output" / instance_id / "jobs"


def read_command(job_dir):
    try:
        return (Path(job_dir) / "command.txt").read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _job_pid_alive(job_dir):
    """True / False / None(unknown) for the runner process. Plain liveness — the
    exit_code file is the normal terminal signal, so this only backstops crashes."""
    p = Path(job_dir) / "pid.txt"
    try:
        pid = int(p.read_text().strip())
    except Exception:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def resolve_job(job_dir):
    """Returns (is_terminal: bool, status: str, exit_code: Optional[int])."""
    job_dir = Path(job_dir)
    status_path = job_dir / "status.txt"
    ec_path = job_dir / "exit_code.txt"

    if status_path.exists():
        try:
            if status_path.read_text(encoding="utf-8").strip().upper() == "KILLED":
                return True, "KILLED", None
        except Exception:
            pass

    if ec_path.exists():
        try:
            code = int(ec_path.read_text(encoding="utf-8").strip())
        except Exception:
            return True, "FAILED", None
        # The watcher drops a marker before it group-kills on timeout, so we
        # report TIMED OUT regardless of the exact signal-derived exit code.
        if (job_dir / "timed_out").exists():
            return True, f"TIMED OUT (exit {code})", code
        if code == 0:
            return True, "COMPLETED", 0
        return True, f"FAILED (exit {code})", code

    # No exit code recorded. If the runner is gone, it died before writing one.
    if _job_pid_alive(job_dir) is False:
        return True, "FAILED", None

    return False, "RUNNING", None


def status_keyword(status):
    """Collapse 'FAILED (exit 1)' / 'TIMED OUT (exit 124)' -> a single token used
    as the notified-set key and digest classifier."""
    if not status:
        return ""
    return status.replace("TIMED OUT", "TIMEOUT").split()[0].split(":")[0].strip().upper()


def _set_pdeathsig():
    # Die if the launching aeon process dies (no orphaned bookkeeping runner).
    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)
    except Exception:
        pass


class RunCommandAsync(BaseTool):
    MAX_CONCURRENT = 8
    HARD_MAX_TIMEOUT = 86400  # 24h ceiling for an explicit timeout

    def __init__(self, worker=None):
        super().__init__(
            name="run_command_async",
            description=(
                "Launch a shell command in the BACKGROUND and return immediately with a job id — the "
                "lightweight way to parallelize. Use it for anything you'd otherwise block on while having "
                "other work to do: a build, a test suite, a long download, a training run, or a server you "
                "want left running. Unlike spawn_sub_agent this starts NO model loop (it's just a tracked "
                "process), so it's cheap and you can run several at once.\n"
                "The job appears LIVE in the BACKGROUND JOBS section of your context every turn; when it "
                "finishes or fails you see it there ONCE — then read its output with job_output. Do your own "
                "work meanwhile; never idle-poll. For a quick command whose output you need right now, use "
                "run_command instead.\n"
                "Schema:\n"
                "  command (str, required): the shell command to run detached.\n"
                "  timeout (int, optional): seconds before the job is force-killed (records exit 124/137). "
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

    def execute(self, command: str = None, timeout: int = 0) -> str:
        if not self.worker:
            return "COMMAND FAILED: Worker context missing."
        if not command or not str(command).strip():
            return "COMMAND FAILED: 'command' is required."
        command = str(command)

        try:
            timeout = max(0, min(int(timeout), self.HARD_MAX_TIMEOUT))
        except (TypeError, ValueError):
            timeout = 0

        base = jobs_base(self.worker)
        if self._running_count(base) >= self.MAX_CONCURRENT:
            return (f"COMMAND FAILED: {self.MAX_CONCURRENT} background jobs already running. "
                    f"Wait for some to finish (watch the BACKGROUND JOBS section), read them with "
                    f"job_output, or free one with kill_job.")

        job_id = str(uuid.uuid4())
        job_dir = base / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        cmd_sh = job_dir / "cmd.sh"
        log_path = job_dir / "output.log"
        ec_path = job_dir / "exit_code.txt"
        wl_path = job_dir / "workload_pid.txt"
        to_path = job_dir / "timed_out"

        rt.atomic_write_text(job_dir / "command.txt", command)
        rt.atomic_write_text(cmd_sh, f"source ~/.bashrc 2>/dev/null\n{command}\n")

        # The workload runs in its OWN session (setsid, no --wait so setsid execs
        # in place and $! IS the new group leader). That lets us TERM/KILL the
        # whole workload TREE -- a plain `timeout bash cmd.sh` only signals the
        # bash layer and orphans its children. The runner stays in the agent's
        # process group (so kill_job's group_kill reaches it) and records the exit
        # code atomically the instant the workload exits. The EXIT trap cleans the
        # workload group on any normal/term exit of the runner; only an uncatchable
        # SIGKILL of the runner itself could leave the workload briefly orphaned.
        watcher = ""
        if timeout > 0:
            watcher = (f"( sleep {timeout} ; touch '{to_path}' ; "
                       f"kill -TERM -$wpid 2>/dev/null ; sleep 10 ; "
                       f"kill -KILL -$wpid 2>/dev/null ) & watcher=$! ; ")
        kill_watcher = "kill $watcher 2>/dev/null ; " if timeout > 0 else ""
        runner = (
            f"trap 'kill -KILL -$wpid 2>/dev/null' EXIT ; "
            f"setsid bash '{cmd_sh}' > '{log_path}' 2>&1 & wpid=$! ; "
            f"echo $wpid > '{wl_path}.tmp' && mv '{wl_path}.tmp' '{wl_path}' ; "
            f"{watcher}"
            f"wait $wpid ; rc=$? ; "
            f"{kill_watcher}"
            f"echo $rc > '{ec_path}.tmp' && mv '{ec_path}.tmp' '{ec_path}'"
        )

        try:
            process = subprocess.Popen(
                ["bash", "-c", runner],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,   # own group -> killable as a tree
                preexec_fn=_set_pdeathsig,
            )
        except Exception as e:
            return f"COMMAND FAILED: could not launch background job: {e}"

        rt.atomic_write_text(job_dir / "pid.txt", str(process.pid))
        rt.atomic_write_text(job_dir / "status.txt", "RUNNING")

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
                output = log_path.read_text(encoding="utf-8", errors="replace")
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
                "Stop a background job and its child processes (kills the whole process group, so a server "
                "or build tree leaves nothing behind). Accepts the short id shown in BACKGROUND JOBS or a "
                "full id.\n"
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

        rt.atomic_write_text(job_dir / "status.txt", "KILLED")
        if self.worker is not None:
            self.worker.notified_jobs.add(f"{job_dir.name}_KILLED")

        # Kill the workload's own session first (it's a separate process group led
        # by the setsid'd shell), then the runner's group. group_kill only ever
        # killpg's a target that is its own group leader, so each call is scoped.
        wl_path = job_dir / "workload_pid.txt"
        if wl_path.exists():
            try:
                group_kill(int(wl_path.read_text().strip()))
            except Exception:
                pass

        pid_path = job_dir / "pid.txt"
        if not pid_path.exists():
            return f"Job {short} marked KILLED (no PID file; process may have already exited)."
        try:
            pid = int(pid_path.read_text().strip())
        except Exception:
            return f"Job {short} marked KILLED (PID file unreadable)."
        group_kill(pid)
        return f"Job {short} (PID {pid}) terminated (process group killed) and marked KILLED."
