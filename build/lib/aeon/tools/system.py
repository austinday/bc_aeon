import os
import subprocess
import sys
import time
import signal
from .base import BaseTool

class RunCommandTool(BaseTool):
    """A tool to execute a command on the command line."""
    def __init__(self):
        super().__init__(
            name="run_command",
            description='Executes shell commands on the HOST machine. Params: `command` (str), `timeout` (int).'
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
                return f"COMMAND FAILED (Exit Code {return_code})\n\nOUTPUT:\n{output}"
            
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
        super().__init__(name="task_complete", description='End task. Params: `reason` (str).')
    def execute(self, reason: str = "Task completed.") -> str:
        return f"Task marked as complete with reason: {reason}"

class GetUserInputTool(BaseTool):
    def __init__(self):
        super().__init__(name="get_user_input", description='Ask user for input. Params: `prompt` (str).')
    def execute(self, prompt: str = "Please provide input:") -> str:
        return f"Awaiting user input with prompt: {prompt}"