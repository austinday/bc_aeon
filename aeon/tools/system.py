import queue
import subprocess
import time
import shutil
import getpass
import threading
from pathlib import Path
from .base import BaseTool
from ..core import runtime_signals as rt
from ..core.process_resources import (
    register_receipted_command,
    unregister_receipted_command,
)
from ..core.prompts import (
    TOOL_DESC_RUN_COMMAND,
    TOOL_DESC_TASK_COMPLETE,
    TOOL_DESC_GET_USER_INPUT,
)
from .command_fleet_guard import (
    FleetCommandGuardError,
    SERVICE_GATE_TIMEOUT,
    finalize_sandbox_service,
    guard_fleet_shell_command,
    launch_sandbox_service,
    prepare_fleet_shell_boundary,
    resolve_command_cwd,
    stop_sandbox_service,
)


def _diagnose_failure(command: str, output: str) -> str:
    """Produce safe, additive hints about why a command failed."""
    hints = []
    out_lower = output.lower()
    first_token = command.strip().split()[0] if command.strip() else ''

    if 'command not found' in out_lower:
        if first_token and shutil.which(first_token) is None:
            hints.append(f"The executable '{first_token}' is not installed or not on PATH.")
        else:
            hints.append("A command in the pipeline was not found. Check spelling and PATH.")
    if 'permission denied' in out_lower:
        hints.append(
            f"Permission denied (current user: '{getpass.getuser()}'). "
            f"For Docker volume mounts, fix permissions on the HOST, not inside the container."
        )
    if 'no such file or directory' in out_lower:
        hints.append("A referenced file or directory does not exist. Verify paths against the Project Tree.")
    if 'is a directory' in out_lower:
        hints.append("The target path is a directory, not a file.")
    if 'address already in use' in out_lower or 'port is already allocated' in out_lower:
        hints.append("A required port is already in use (possibly a leftover container or process).")
    if 'no space left on device' in out_lower:
        hints.append("The disk is full.")

    if not hints:
        return ''
    return "\n\nHINTS:\n- " + "\n- ".join(hints)


class RunCommandTool(BaseTool):
    """Executes a shell command on the HOST, streaming output to the terminal."""

    MAX_RETURN_CHARS = 30000
    HARD_MAX_TIMEOUT = 1800

    def __init__(self):
        super().__init__(
            name="run_command",
            description=TOOL_DESC_RUN_COMMAND
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.MAX_RETURN_CHARS:
            return text
        omitted = len(text) - self.MAX_RETURN_CHARS
        head = self.MAX_RETURN_CHARS // 4
        tail = self.MAX_RETURN_CHARS - head
        # The agent cannot scroll the terminal in later turns, so point it at a
        # retrievable way to get the omitted middle instead of a dead reference.
        return (
            text[:head]
            + f"\n\n... [{omitted:,} characters truncated to protect context — showing head+tail. "
              f"To inspect the omitted middle, re-run capturing to a file "
              f"(`<cmd> > /tmp/out.log 2>&1`) then read specific parts with grep -n / sed -n / tail.] ...\n\n"
            + text[len(text) - tail:]
        )

    def execute(
        self,
        command: str,
        timeout: int = 300,
        cwd: str | None = None,
    ) -> str:
        if not command:
            return "Error: command parameter is required."

        cleanup_token: str | None = None

        def _register_active_receipt(receipt) -> None:
            nonlocal cleanup_token
            cleanup_token = register_receipted_command(
                lambda: stop_sandbox_service(receipt)
            )

        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 300
        # A non-positive timeout previously meant "no limit", which could hang an
        # entire sub-agent iteration with no per-command bound. Always bound it,
        # and never exceed the ceiling (which is >= the watchdog's stall window so
        # the command's own timeout fires first, not a sub-agent kill).
        effective_timeout = 300 if timeout <= 0 else min(timeout, self.HARD_MAX_TIMEOUT)

        # Admission and fixed-file validation happen before any process. The
        # service bootstrap then remains behind its gate until the actual unit's
        # sandbox, MainPID/cgroup, optional parent slice, and InvocationID are
        # externally read back and the receipt is durably stored.
        try:
            session_root = Path.cwd().resolve(strict=True)
            command_cwd = resolve_command_cwd(cwd, session_root=session_root)
            cwd_metadata = command_cwd.stat(follow_symlinks=False)
            cwd_identity = (int(cwd_metadata.st_dev), int(cwd_metadata.st_ino))
            command = guard_fleet_shell_command(command)
            boundary, manager_environment = prepare_fleet_shell_boundary(
                cwd=command_cwd,
                session_root=session_root,
                expected_cwd_identity=cwd_identity,
                runtime_max_seconds=effective_timeout + int(SERVICE_GATE_TIMEOUT) + 5
            )
            handle = launch_sandbox_service(
                command,
                boundary,
                manager_environment,
                on_receipt=_register_active_receipt,
            )
        except FleetCommandGuardError as exc:
            unregister_receipted_command(cleanup_token)
            return str(exc)
        except Exception as exc:
            unregister_receipted_command(cleanup_token)
            return (
                "COMMAND REFUSED: transient-service sandbox startup failed "
                f"({type(exc).__name__}). The requested command was not executed."
            )

        output_lines = [handle.initial_output] if handle.initial_output else []
        # Test doubles and compatible downstream launchers may not implement the
        # pre-gate callback yet. Keep the post-return registration as a bounded
        # compatibility fallback; the production launcher always registers
        # before opening the payload gate.
        if cleanup_token is None:
            _register_active_receipt(handle.receipt)
        start_time = time.time()
        process = handle.process
        # Keep the liveness heartbeat fresh for the whole command so a long-but-alive
        # command (e.g. a slow build) is never mistaken for a hang by the watchdog.
        # Harmless no-op in the primary agent.
        stop_toucher = threading.Event()

        def _toucher():
            while not stop_toucher.wait(timeout=5):
                rt.touch()

        threading.Thread(target=_toucher, daemon=True).start()

        try:
            # Read stdout on a dedicated thread so the wall-clock timeout is
            # enforced even when the command produces NO output (a silent hang
            # used to block readline() past the timeout indefinitely). The
            # reader pushes lines onto a queue and a None sentinel at EOF.
            line_q: "queue.Queue" = queue.Queue()

            def _reader():
                try:
                    for ln in iter(process.stdout.readline, ''):
                        line_q.put(ln)
                finally:
                    line_q.put(None)

            reader_thread = threading.Thread(target=_reader, daemon=True)
            reader_thread.start()

            while True:
                if time.time() - start_time > effective_timeout:
                    stop_sandbox_service(handle.receipt)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                    partial = self._truncate("".join(output_lines))
                    return (
                        f"COMMAND TIMED OUT after {effective_timeout}s "
                        "(exact transient service stopped).\n\n"
                        f"PARTIAL OUTPUT:\n{partial}\n\n"
                        f"HINT: If this is expected to run long, raise 'timeout' (capped at "
                        f"{self.HARD_MAX_TIMEOUT}s), run it in the background, or bound its runtime."
                    )

                try:
                    line = line_q.get(timeout=0.5)
                except queue.Empty:
                    continue  # no output yet — loop back and re-check the timeout
                if line is None:
                    break  # EOF: command finished and stdout closed
                print(line, end='', flush=True)
                output_lines.append(line)
                rt.touch()

            # stdout is closed; give the process a moment to be reaped so we read
            # a real exit code rather than None.
            try:
                return_code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return_code = process.poll()
            output = "".join(output_lines)

            if return_code != 0:
                diag = _diagnose_failure(command, output)
                body = self._truncate(output) if output.strip() else "(no output)"
                return (
                    f"COMMAND FAILED (Exit Code {return_code})\n"
                    f"WORKING DIRECTORY: {boundary.cwd}\n\nOUTPUT:\n{body}{diag}"
                )

            if not output.strip():
                return (
                    "COMMAND SUCCESS (no output).\n"
                    f"WORKING DIRECTORY: {boundary.cwd}"
                )
            return (
                f"COMMAND SUCCESS\nWORKING DIRECTORY: {boundary.cwd}\n\n"
                f"OUTPUT:\n{self._truncate(output)}"
            )

        except KeyboardInterrupt:
            print("\n[RunCommand] Interrupted! Stopping exact transient service...", flush=True)
            stop_sandbox_service(handle.receipt)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise

        except Exception as e:
            try:
                stop_sandbox_service(handle.receipt)
            except Exception:
                pass
            return f"An error occurred while running the command: {type(e).__name__}: {e}"

        finally:
            stop_toucher.set()
            unregister_receipted_command(cleanup_token)
            finalize_sandbox_service(handle)


class TaskCompleteTool(BaseTool):
    def __init__(self):
        super().__init__(name="task_complete", description=TOOL_DESC_TASK_COMPLETE)

    def execute(self, reason: str = "Task completed.") -> str:
        return f"Task marked as complete with reason: {reason}"


class GetUserInputTool(BaseTool):
    def __init__(self):
        super().__init__(name="get_user_input", description=TOOL_DESC_GET_USER_INPUT)

    def execute(self, prompt: str = "Please provide input:") -> str:
        return f"Awaiting user input with prompt: {prompt}"
