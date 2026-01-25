import os
import shutil
import time
import json
import subprocess
import uuid
import sys
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from .base import BaseTool
from ..core.worker import Worker
from ..core.llm import LLMClient

class DockerExecTool(BaseTool):
    def __init__(self, container_name: str):
        super().__init__(
            name="run_command",
            description='Executes shell commands INSIDE the isolated research container.'
        )
        self.container_name = container_name
        self.is_internal = True # Internal to sub-agents

    def execute(self, command: str, timeout: int = 300):
        docker_cmd = f"docker exec -w /workspace {self.container_name} bash -c {json.dumps(command)}"
        try:
            process = subprocess.run(docker_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            output = process.stdout + process.stderr
            return f"COMMAND {'SUCCESS' if process.returncode == 0 else 'FAILED'}\n\nOUTPUT:\n{output}"
        except Exception as e: return f"Error: {e}"

class SubmitFindingsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="submit_findings",
            description='Submit your research results and EXIT.'
        )
        self.is_internal = True # Internal to sub-agents

    def execute(self, findings: str, summary: str):
        return f"Findings Submitted.\nSUMMARY: {summary}\nDETAILS:\n{findings}"

class ConductResearchTool(BaseTool):
    def __init__(self, llm_client: LLMClient, worker: Worker):
        super().__init__(
            name="conduct_research",
            description='Spawns parallel research agents in isolated Docker containers to test distinct hypotheses. Returns a comprehensive report.'
        )
        self.llm_client = llm_client
        self.main_worker = worker

    def execute(self, topic: str, hypotheses: List[Dict[str, str]], image_tag: str = "aeon_base:py3.10-cuda12.1", iterations: int = 30):
        # Implementation logic remains same as research.py source
        return "Conducting research... (Implementation hidden for brevity)"