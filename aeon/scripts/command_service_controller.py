#!/usr/bin/python3
"""Durable controller for one asynchronous Aeon command service.

This process executes no model-supplied shell. It owns only the exact transient
unit launch/readback/gate, output descriptor, timeout, and terminal record.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aeon.core import runtime_signals as rt  # noqa: E402
from aeon.tools.command_fleet_guard import (  # noqa: E402
    SERVICE_GATE_TIMEOUT,
    discard_prepared_sandbox_boundary,
    finalize_sandbox_service,
    launch_sandbox_service,
    prepare_fleet_shell_boundary,
    stop_sandbox_service,
)


def _read_request(job_dir: Path) -> tuple[str, str, int]:
    value = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    command = value.get("command")
    cwd = value.get("cwd")
    timeout = value.get("timeout")
    if (
        value.get("schema") != 1
        or not isinstance(command, str)
        or not command.strip()
        or "\x00" in command
        or not isinstance(cwd, str)
        or not Path(cwd).is_absolute()
        or isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or timeout < 0
    ):
        raise ValueError("invalid asynchronous command request")
    canonical_cwd = Path(cwd).resolve(strict=True)
    try:
        relative = job_dir.relative_to(canonical_cwd / "aeon_output")
        uuid.UUID(job_dir.name)
    except (OSError, ValueError) as exc:
        raise ValueError("invalid asynchronous job-state identity") from exc
    metadata = job_dir.lstat()
    if (
        len(relative.parts) < 3
        or relative.parts[-2] != "jobs"
        or not job_dir.is_dir()
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise ValueError("invalid asynchronous job-state identity")
    return command, cwd, timeout


def _cancelled(job_dir: Path) -> bool:
    return (job_dir / "cancel_startup").is_file()


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        return 2
    job_dir = Path(arguments[0])
    if not job_dir.is_absolute() or not job_dir.is_dir():
        return 2
    handle = None
    boundary = None
    validated_job_dir = False
    try:
        command, cwd, timeout = _read_request(job_dir)
        if job_dir != job_dir.resolve(strict=True):
            raise ValueError("invalid asynchronous job-state identity")
        validated_job_dir = True
        rt.atomic_write_text(job_dir / "pid.txt", str(os.getpid()))
        if _cancelled(job_dir):
            raise RuntimeError("asynchronous command startup was cancelled")
        # RuntimeMaxSec remains an independent, manager-owned backstop if this
        # controller crashes. The controller enforces the requested payload
        # timeout from gate release; the extra interval covers gated startup and
        # systemd's bounded TERM-to-KILL cleanup.
        runtime_limit = (
            timeout + int(SERVICE_GATE_TIMEOUT) + 5 if timeout > 0 else None
        )
        boundary, manager_environment = prepare_fleet_shell_boundary(
            source_environment=os.environ,
            cwd=cwd,
            runtime_max_seconds=runtime_limit,
            internal_state_path=job_dir,
        )
        if _cancelled(job_dir):
            discard_prepared_sandbox_boundary(boundary)
            boundary = None
            raise RuntimeError("asynchronous command startup was cancelled")
        handle = launch_sandbox_service(
            command,
            boundary,
            manager_environment,
            receipt_path=job_dir / "service_receipt.json",
            output_path=job_dir / "output.log",
            payload_environment=os.environ,
        )
        boundary = None
        if _cancelled(job_dir):
            stop_sandbox_service(handle.receipt)
            raise RuntimeError("asynchronous command startup was cancelled")
        rt.atomic_write_text(job_dir / "status.txt", "RUNNING")
        started = time.monotonic()
        if timeout > 0:
            try:
                return_code = handle.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                rt.atomic_write_text(job_dir / "timed_out", "timeout")
                stop_sandbox_service(handle.receipt)
                return_code = handle.process.wait(timeout=10)
        else:
            return_code = handle.process.wait()
        # If systemd's RuntimeMaxSec backstop won the deadline race, report it
        # truthfully even if the local wait returned just before TimeoutExpired.
        if timeout > 0 and time.monotonic() - started >= timeout - 0.05 and return_code != 0:
            rt.atomic_write_text(job_dir / "timed_out", "timeout")
        rt.atomic_write_text(job_dir / "exit_code.txt", str(int(return_code)))
        rt.atomic_write_text(job_dir / "controller_done", "done")
        return 0
    except Exception as exc:
        if not validated_job_dir:
            return 2
        rt.atomic_write_text(
            job_dir / "startup_error.txt",
            f"{type(exc).__name__}: {exc}",
        )
        rt.atomic_write_text(job_dir / "status.txt", "FAILED")
        rt.atomic_write_text(job_dir / "exit_code.txt", "125")
        rt.atomic_write_text(job_dir / "controller_done", "done")
        return 1
    finally:
        if boundary is not None:
            discard_prepared_sandbox_boundary(boundary)
        if handle is not None:
            finalize_sandbox_service(handle)


if __name__ == "__main__":
    raise SystemExit(main())
