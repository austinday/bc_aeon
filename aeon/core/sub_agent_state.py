"""
Crash-safe terminal-state resolution and safe process-group termination for
sub-agents. Pure logic, stdlib-only (no aeon imports), so worker.py and the
sub-agent tools can all use it without import cycles.

resolve() is the single source of truth for "is this sub-agent done, and if so
what happened". It reads, in priority order:
  1. output.json  -- the durable terminal record (always written terminal-FIRST,
     atomically, and NEVER deleted). If present, the agent is terminal.
  2. status.txt   -- an explicit terminal token (COMPLETED/FAILED/KILLED).
  3. PID liveness -- if neither terminal record exists but the wrapper PROCESS is
     gone, the agent crashed/was-killed before writing a result. This is the case
     that previously made gather() block for its full timeout and leaked a
     concurrency slot forever; here it resolves immediately as FAILED.

group_kill() only ever uses killpg when the target is its OWN group leader (i.e.
it was spawned with start_new_session=True). Otherwise it kills just that PID.
This makes it structurally impossible to take out a shared group -- critically,
the primary agent's group, which verify_self_modification's test sub-agent shares.
"""

import os
import json
import signal
from pathlib import Path

TERMINAL = {"COMPLETED", "FAILED", "KILLED"}


def norm_status(status):
    """Collapse 'FAILED: <detail>' -> 'FAILED' for matching."""
    if not status:
        return ""
    return str(status).split(":", 1)[0].strip().upper()


def pid_alive(agent_dir):
    """True / False / None(unknown). Guards against PID recycling via /proc cmdline."""
    pid_path = Path(agent_dir) / "pid.txt"
    try:
        pid = int(pid_path.read_text().strip())
    except Exception:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "r") as f:
            cmdline = f.read().replace("\x00", " ")
        if "sub_agent_wrapper" not in cmdline:
            return False
    except FileNotFoundError:
        return False
    except Exception:
        pass
    return True


def resolve(agent_dir):
    """Returns (is_terminal: bool, status: str, report: Optional[str])."""
    agent_dir = Path(agent_dir)
    output_path = agent_dir / "output.json"
    status_path = agent_dir / "status.txt"

    if output_path.exists():
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
            st = data.get("status", "COMPLETED")
            if "error" in data and norm_status(st) != "COMPLETED":
                return True, st, f"Error: {data['error']}"
            return True, st, str(data.get("result", "N/A"))
        except Exception:
            pass  # atomic writes make a torn read impossible; fall through if genuinely corrupt

    status_text = None
    if status_path.exists():
        try:
            status_text = status_path.read_text(encoding="utf-8").strip()
        except Exception:
            status_text = None

    if status_text and norm_status(status_text) in TERMINAL:
        return True, status_text, "(terminal status reported; no output.json found)"

    if pid_alive(agent_dir) is False:
        return True, "FAILED", ("Process exited without writing a result "
                                "(crashed during startup or was killed externally).")

    return False, (status_text or "RUNNING"), None


def group_kill(pid):
    """SIGKILL a sub-agent. Group-kills (process + command children) only when the
    target is its own group leader; otherwise kills the single PID to avoid collateral."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    except OSError:
        pgid = None
    try:
        if pgid is not None and pgid == pid:
            os.killpg(pgid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
