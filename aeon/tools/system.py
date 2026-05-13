import os
import subprocess
import sys
import time
import signal
from .base import BaseTool
from ..core.prompts import (
    TOOL_DESC_RUN_COMMAND,
    TOOL_DESC_TASK_COMPLETE,
    TOOL_DESC_GET_USER_INPUT,
)

class RunCommandTool(BaseTool):
    """A tool to execute a command on the command line."""
    def __init__(self):
        super().__init__(
            name="run_command",
            description=TOOL_DESC_RUN_COMMAND
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
                stdin=subprocess.DEVNULL,  # Instantly crash interactive prompts
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )            
            while True:
                if effective_timeout and (time.time() - start_time > effective_timeout):
                    process.kill()
                    return f"Error: Command timed out after {timeout} seconds.\nPartial Output:\n{''.join(output_lines)}"

                line = process.stdout.readline()
                
                if not line and process.poll() is not None:
                    break
                
                if line:
                    print(line, end='', flush=True)
                    output_lines.append(line)
            
            return_code = process.poll()
            output = "".join(output_lines)
            
            if return_code != 0:
                debug_info = ""
                out_lower = output.lower()
                if "is a directory" in out_lower:
                    debug_info += "\n[SYSTEM AUTO-DEBUG] Fact: The target path is a directory, not a file."
                if "no such file or directory" in out_lower:
                    debug_info += "\n[SYSTEM AUTO-DEBUG] Fact: The specified path does not exist on this filesystem."
                if "permission denied" in out_lower:
                    import getpass
                    debug_info += f"\n[SYSTEM AUTO-DEBUG] Fact: The current user '{getpass.getuser()}' lacks permissions for this operation."
                if "command not found" in out_lower:
                    debug_info += "\n[SYSTEM AUTO-DEBUG] Fact: The executable is not installed or not in the system PATH."
                if "file exists" in out_lower:
                    debug_info += "\n[SYSTEM AUTO-DEBUG] Fact: The target path already exists."

                return f"COMMAND FAILED (Exit Code {return_code})\n\nOUTPUT:\n{output}{debug_info}"
            
            if not output.strip():
                return "Command executed successfully with no output."
                
            return f"COMMAND SUCCESS\n\nOUTPUT:\n{output}"

        except KeyboardInterrupt:
            # Kill the subprocess but let the exception propagate to worker loop
            # The worker loop has an interactive dialog for user guidance
            print("\n[RunCommand] Interrupted! Stopping subprocess...", flush=True)
            if process:
                try:
                    process.kill()  # Send SIGKILL (cannot be ignored)
                    process.wait(timeout=1)  # Briefly wait to reap zombie
                except: pass
            # Re-raise to worker loop which handles user interaction
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
