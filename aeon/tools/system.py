import os
import subprocess
import sys
import time
from .base import BaseTool

class RunCommandTool(BaseTool):
    """A tool to execute a command on the command line."""
    def __init__(self):
        super().__init__(
            name="run_command",
            description='Executes shell commands. Returns FULL output for analysis. NEVER use run_command for opening files, use open_file instead. Params: `command` (str), optional `timeout` (int, default 300). Example: `{"tool_name": "run_command", "parameters": {"command": "ls -la"}}`'
        )

    def execute(self, command: str, timeout: int = 300):
        effective_timeout = timeout if timeout > 0 else None
        output_lines = []
        start_time = time.time()
        
        try:
            # Use Popen to stream output
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr into stdout
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1 # Line buffered
            )
            
            while True:
                # Check timeout (Note: readline may block if no output, delaying this check)
                if effective_timeout and (time.time() - start_time > effective_timeout):
                    process.kill()
                    output = "".join(output_lines)
                    return f"Error: Command timed out after {timeout} seconds.\nPartial Output:\n{output}"

                line = process.stdout.readline()
                
                if not line and process.poll() is not None:
                    break
                
                if line:
                    # Print to real terminal for user visibility
                    print(line, end='', flush=True)
                    output_lines.append(line)
            
            return_code = process.poll()
            output = "".join(output_lines)
            
            if return_code != 0:
                return f"COMMAND FAILED (Exit Code {return_code})\n\nOUTPUT:\n{output}"
            
            if not output.strip():
                return "Command executed successfully with no output."
                
            return f"COMMAND SUCCESS\n\nOUTPUT:\n{output}"

        except Exception as e:
            return f"An error occurred while running the command: {e}"

class TaskCompleteTool(BaseTool):
    """A tool to signal that the task is complete."""
    def __init__(self):
        super().__init__(
            name="task_complete",
            description='End task. Use when objective is met. Params: `reason` (str). Example: `{"tool_name": "task_complete", "parameters": {"reason": "All tests passed."}}`'
        )
    def execute(self, reason: str):
        return f"Task marked as complete with reason: {reason}"

class GetUserInputTool(BaseTool):
    """A tool to signal a request for user input."""
    def __init__(self):
        super().__init__(
            name="get_user_input",
            description='Ask user for input. Use to pause for clarification. Params: `prompt` (str). Example: `{\"tool_name\": \"get_user_input\", \"parameters\": {\"prompt\": \"Confirm deletion?\"}}`'
        )
    def execute(self, prompt: str):
        return f"Awaiting user input with prompt: {prompt}"