"""Linux process-group receipts that fail closed when a PID is reused."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
from typing import Any, Mapping


class ProcessIdentityError(RuntimeError):
    """A process cannot be proven to match its durable receipt."""


class ProcessNotReady(ProcessIdentityError):
    """A newly launched process has not reached its recordable stopped state."""


def _proc_fields(pid: int) -> tuple[str, int, int, int]:
    """Return state, PPID, process group, and start ticks from Linux ``/proc``."""
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = raw[raw.rfind(")") + 2:].split()
    return fields[0], int(fields[1]), int(fields[2]), int(fields[19])


def capture_process_group(
    pid: int,
    identity: str,
    *,
    expected_parent_pid: int | None = None,
    require_stopped: bool = False,
) -> dict[str, Any]:
    """Capture a newly-created session leader before its PID can be trusted later."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise ProcessIdentityError("process id must be greater than one")
    if not isinstance(identity, str) or not identity or len(identity) > 256:
        raise ProcessIdentityError("process identity label is invalid")
    try:
        process_state, parent_pid, process_group, start_ticks = _proc_fields(pid)
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise ProcessLookupError(pid) from exc
    if process_group != pid or os.getpgid(pid) != pid:
        raise ProcessIdentityError("process is not its own group leader")
    if expected_parent_pid is not None and parent_pid != expected_parent_pid:
        raise ProcessIdentityError("process parent does not match its launcher")
    if require_stopped and process_state not in {"T", "t"}:
        raise ProcessNotReady("process has not reached its stopped identity barrier")
    return {
        "schema": 1,
        "identity": identity,
        "pid": pid,
        "pgid": process_group,
        "start_ticks": start_ticks,
    }


def read_process_group(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProcessIdentityError("missing or unreadable process-group receipt") from exc
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ProcessIdentityError("unsupported process-group receipt")
    return value


def validate_process_group(
    reference: Mapping[str, Any], expected_identity: str,
) -> tuple[int, int]:
    try:
        pid = int(reference["pid"])
        pgid = int(reference["pgid"])
        start_ticks = int(reference["start_ticks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProcessIdentityError("invalid process-group receipt") from exc
    if (
        reference.get("identity") != expected_identity
        or pid <= 1
        or pgid != pid
    ):
        raise ProcessIdentityError("process-group receipt identity does not match")
    try:
        _state, _parent_pid, current_group, current_start_ticks = _proc_fields(pid)
        observed_group = os.getpgid(pid)
    except (FileNotFoundError, ProcessLookupError) as exc:
        raise ProcessLookupError(pid) from exc
    if (
        current_group != pgid
        or observed_group != pgid
        or current_start_ticks != start_ticks
    ):
        raise ProcessIdentityError("PID or process group was reused")
    return pid, pgid


def process_group_alive(
    reference: Mapping[str, Any], expected_identity: str,
) -> bool | None:
    """Return true, false, or unknown when identity cannot be proven."""
    try:
        pid, _pgid = validate_process_group(reference, expected_identity)
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except ProcessIdentityError:
        return None
    except PermissionError:
        return True
    except OSError:
        return False


def signal_process_group(
    reference: Mapping[str, Any], expected_identity: str, sig: signal.Signals,
) -> bool:
    """Signal only an exactly matching group, pinned with a pidfd while checked."""
    if sig not in {signal.SIGCONT, signal.SIGTERM, signal.SIGKILL}:
        raise ProcessIdentityError("unsupported process-group signal")
    pid, pgid = validate_process_group(reference, expected_identity)
    if not hasattr(os, "pidfd_open"):
        raise ProcessIdentityError("pidfd_open is unavailable")
    try:
        pidfd = os.pidfd_open(pid, 0)
    except ProcessLookupError:
        return False
    try:
        validate_process_group(reference, expected_identity)
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(pidfd)
