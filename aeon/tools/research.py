import os
import time
import json
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from .base import BaseTool
from ..core.worker import Worker
from ..core.llm import LLMClient

# --- Docker Tools for Sub-Agents ---

class DockerExecTool(BaseTool):
    def __init__(self, container_name: str):
        super().__init__(
            name="run_command",
            description='Executes shell commands INSIDE the isolated research container. Params: `command` (str). '
        )
        self.container_name = container_name
        self.is_internal = True

    def execute(self, command: str):
        # Best approach: docker exec -w /workspace CONTAINER bash -c "COMMAND"
        try:
            cmd = ["docker", "exec", "-w", "/workspace", self.container_name, "bash", "-c", command]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
                
            if result.returncode != 0:
                return f"COMMAND FAILED (Code {result.returncode}):\n{output}"
            return f"COMMAND SUCCESS:\n{output}"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out."
        except Exception as e:
            return f"Error executing docker command: {e}"

class DockerWriteFileTool(BaseTool):
    def __init__(self, container_name: str):
        super().__init__(
            name="write_file",
            description='Writes a file inside the container. Params: `file_path`, `content`.'
        )
        self.container_name = container_name
        self.is_internal = True

    def execute(self, file_path: str, content: str):
        # Use docker exec -i with stdin to write files safely
        try:
            cmd = ["docker", "exec", "-i", "-w", "/workspace", self.container_name, "bash", "-c", f"cat > {file_path}"]
            subprocess.run(cmd, input=content, text=True, check=True)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"

class DockerReadFileTool(BaseTool):
    def __init__(self, container_name: str):
        super().__init__(
            name="open_file",
            description='Reads a file from the container. Params: `file_path`.'
        )
        self.container_name = container_name
        self.is_internal = True

    def execute(self, file_path: str):
        try:
            cmd = ["docker", "exec", "-w", "/workspace", self.container_name, "cat", file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return f"Error reading file: {result.stderr}"
            return result.stdout
        except Exception as e:
            return f"Error reading file: {e}"

class SubmitFindingsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="submit_findings",
            description='Submit your final research findings and EXIT. Params: `summary` (str), `details` (str).'
        )
        self.is_internal = True
        self.findings = None

    def execute(self, summary: str, details: str):
        self.findings = {"summary": summary, "details": details}
        return "Findings submitted. Terminating."

# --- Main Research Tool ---

class ConductResearchTool(BaseTool):
    def __init__(self, llm_client: LLMClient, worker: Worker):
        super().__init__(
            name="conduct_research",
            description='Spawns parallel research agents in isolated Docker containers. Params: `topic` (str), `hypotheses` (List[Dict{"name": str, "description": str}]).'
        )
        self.llm_client = llm_client
        self.main_worker = worker # Parent worker context

    def _run_researcher(self, idx: int, topic: str, hypothesis: Dict, image_tag: str, iterations: int) -> Dict:
        container_name = f"aeon_research_{uuid.uuid4().hex[:8]}"
        h_name = hypothesis.get("name", f"Hypothesis {idx}")
        h_desc = hypothesis.get("description", "No description")
        
        print(f"[Research] Launching Agent {idx} for '{h_name}'...")
        
        # 1. Start Container
        try:
            subprocess.run(
                ["docker", "run", "-d", "--rm", "--name", container_name, image_tag, "sleep", "infinity"],
                check=True, capture_output=True
            )
        except subprocess.CalledProcessError as e:
            return {"error": f"Failed to start container: {e}"}

        # 2. Setup Tools
        submit_tool = SubmitFindingsTool()
        tools = [
            DockerExecTool(container_name),
            DockerWriteFileTool(container_name),
            DockerReadFileTool(container_name),
            submit_tool
        ]
        
        # 3. Create Sub-Worker
        # We use a simple prefix printer to avoid console chaos
        def prefix_print(msg):
            if "Objective:" in msg or "Result Summary:" in msg:
                # ANSI Cyan for sub-agent distinction
                C_CYAN = '\033[96m'
                C_RESET = '\033[0m'
                print(f"{C_CYAN}[Agent {idx}]{C_RESET} {msg}")

        sub_worker = Worker(self.llm_client, tools=tools, print_func=prefix_print)
        
        objective = f"""
RESEARCH ASSIGNMENT: {topic}
HYPOTHESIS TO TEST: {h_name}
DETAILS: {h_desc}

Your goal is to prove or disprove this hypothesis experimentally.
1. Write scripts/tests to verify the hypothesis.
2. Execute them in the environment.
3. Analyze the output.
4. Use 'submit_findings' to report your conclusion.
"""
        
        # 4. Run
        try:
            # We treat 'submit_findings' as the terminal tool
            sub_worker.run(objective, max_iterations=iterations, terminal_tools=['submit_findings'])
        except Exception as e:
            print(f"[Agent {idx}] Crashed: {e}")
        finally:
            # 5. Cleanup
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

        return {
            "hypothesis": h_name,
            "findings": submit_tool.findings or {"summary": "Agent timed out or failed to report.", "details": "No data."}
        }

    def execute(self, topic: str, hypotheses: List[Dict[str, str]], image_tag: str = "aeon_base:py3.10-cuda12.1", iterations: int = 15):
        if not hypotheses:
            return "Error: No hypotheses provided."

        results = []
        # Limit concurrency to 4 to save VRAM/CPU
        with ThreadPoolExecutor(max_workers=min(len(hypotheses), 4)) as executor:
            futures = [
                executor.submit(self._run_researcher, i, topic, h, image_tag, iterations) 
                for i, h in enumerate(hypotheses)
            ]
            
            for future in as_completed(futures):
                results.append(future.result())

        # Compile Final Report
        report = f"RESEARCH REPORT: {topic}\n" + "="*40 + "\n\n"
        for res in results:
            findings = res.get("findings", {})
            report += f"HYPOTHESIS: {res.get('hypothesis')}\n"
            report += f"SUMMARY: {findings.get('summary')}\n"
            report += f"DETAILS: {findings.get('details')}\n"
            report += "-"*40 + "\n"

        return report
