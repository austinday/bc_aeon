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
from ..core.prompts import (
    TOOL_DESC_DOCKER_EXEC,
    TOOL_DESC_DOCKER_WRITE_FILE,
    TOOL_DESC_DOCKER_READ_FILE,
    TOOL_DESC_SUBMIT_FINDINGS,
    TOOL_DESC_CONDUCT_RESEARCH,
    RESEARCH_OBJECTIVE_TEMPLATE,
)

# --- Docker Tools for Sub-Agents ---

class DockerExecTool(BaseTool):
    def __init__(self, container_name: str):
        super().__init__(
            name="run_command",
            description=TOOL_DESC_DOCKER_EXEC
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
            description=TOOL_DESC_DOCKER_WRITE_FILE
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
            description=TOOL_DESC_DOCKER_READ_FILE
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
            description=TOOL_DESC_SUBMIT_FINDINGS
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
            description=TOOL_DESC_CONDUCT_RESEARCH
        )
        self.is_internal = True  # DISABLED: Sub-agents need Docker directive improvements
        self.llm_client = llm_client
        self.main_worker = worker  # Parent worker context

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
            return {
                "hypothesis": h_name,
                "findings": {"summary": "Startup Failed", "details": f"Failed to start Docker container: {e.stderr.decode('utf-8', errors='replace')}"}
            }

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
        
        objective = RESEARCH_OBJECTIVE_TEMPLATE.format(topic=topic, h_name=h_name, h_desc=h_desc)
        
        crash_error = None
        
        # 4. Run
        try:
            # We treat 'submit_findings' as the terminal tool
            sub_worker.run(objective, max_iterations=iterations, terminal_tools=['submit_findings'])
        except Exception as e:
            crash_error = str(e)
            print(f"[Agent {idx}] Crashed: {e}")
        finally:
            # 5. Cleanup
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

        # Construct findings safely
        if submit_tool.findings:
            return {"hypothesis": h_name, "findings": submit_tool.findings}
        
        # Fallback if no findings were submitted
        fail_reason = "Agent timed out or failed to report."
        fail_details = "No data returned."
        if crash_error:
            fail_reason = "Agent Crashed"
            fail_details = f"Exception encountered: {crash_error}"
            
        return {
            "hypothesis": h_name,
            "findings": {"summary": fail_reason, "details": fail_details}
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
