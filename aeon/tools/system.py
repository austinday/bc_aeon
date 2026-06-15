import subprocess
import time
import shutil
import getpass
from .base import BaseTool
from ..core.prompts import (
    TOOL_DESC_RUN_COMMAND,
    TOOL_DESC_TASK_COMPLETE,
    TOOL_DESC_GET_USER_INPUT,
)


def _diagnose_failure(command: str, output: str) -> str:
    """Produce safe, additive hints about why a command failed.

    Scans the command string and its output for common, recognizable failure
    signatures. References no undefined names and never raises. This replaces the
    previous broken branch that referenced an undefined `path` / `getpass` /
    `_is_executable` and crashed with NameError on every non-zero exit.
    """
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

    def __init__(self):
        super().__init__(
            name="run_command",
            description=TOOL_DESC_RUN_COMMAND
        )

    def _truncate(self, text: str) -> str:
        """Head+tail truncation of the text RETURNED to the LLM. The full output
        is always streamed to the terminal, so the user sees everything; only the
        model's context is bounded to prevent context blowups from huge outputs."""
        if len(text) <= self.MAX_RETURN_CHARS:
            return text
        omitted = len(text) - self.MAX_RETURN_CHARS
        head = self.MAX_RETURN_CHARS // 4
        tail = self.MAX_RETURN_CHARS - head
        return (
            text[:head]
            + f"\n\n... [{omitted:,} characters truncated to protect context; "
              f"the full output was streamed to the terminal above] ...\n\n"
            + text[len(text) - tail:]
        )

    def execute(self, command: str, timeout: int = 300) -> str:
        if not command:
            return "Error: command parameter is required."

        effective_timeout = timeout if timeout > 0 else None
        output_lines = []
        start_time = time.time()
        wrapped_command = f"source ~/.bashrc 2>/dev/null; {command}"

        process = None
        try:
            process = subprocess.Popen(
                wrapped_command,
                shell=True,
                executable="/bin/bash",
                stdin=subprocess.DEVNULL,  # instantly fail interactive prompts instead of hanging
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
            )

            while True:
                if effective_timeout and (time.time() - start_time > effective_timeout):
                    process.kill()
                    partial = self._truncate("".join(output_lines))
                    return (
                        f"COMMAND TIMED OUT after {timeout}s (process killed).\n\n"
                        f"PARTIAL OUTPUT:\n{partial}\n\n"
                        f"HINT: If this command is expected to run long, raise 'timeout', "
                        f"run it in the background, or bound its runtime."
                    )

                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line, end='', flush=True)
                    output_lines.append(line)

            return_code = process.poll()
            output = "".join(output_lines)

            if return_code != 0:
                diag = _diagnose_failure(command, output)
                body = self._truncate(output) if output.strip() else "(no output)"
                return f"COMMAND FAILED (Exit Code {return_code})\n\nOUTPUT:\n{body}{diag}"

            if not output.strip():
                return "COMMAND SUCCESS (no output)."
            return f"COMMAND SUCCESS\n\nOUTPUT:\n{self._truncate(output)}"

        except KeyboardInterrupt:
            print("\n[RunCommand] Interrupted! Stopping subprocess...", flush=True)
            if process:
                try:
                    process.kill()
                    process.wait(timeout=1)
                except Exception:
                    pass
            raise

        except Exception as e:
            return f"An error occurred while running the command: {type(e).__name__}: {e}"


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
