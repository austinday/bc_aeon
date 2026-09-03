"""Stopped bootstrap for an exactly-recorded Aeon background workload."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import signal
import sys


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        return 2
    command_path = Path(arguments[0])
    if not command_path.is_absolute() or not command_path.is_file():
        return 2

    parent_pid = os.getppid()
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(1, signal.SIGKILL) != 0:
            return 1
    except Exception:
        return 1
    if os.getppid() != parent_pid:
        return 1

    # The parent records PID, PGID, and start ticks while this process cannot
    # execute user code. It resumes the exact recorded group with SIGCONT.
    os.kill(os.getpid(), signal.SIGSTOP)
    os.execv("/bin/bash", ["bash", str(command_path)])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
