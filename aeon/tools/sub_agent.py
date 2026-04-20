import os
import sys
import json
import uuid
import subprocess
import time
import copy
import re
from pathlib import Path
from aeon.tools.base import BaseTool

class SpawnSubAgent(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="spawn_sub_agent",
            description="Spawns a heavy background sub-agent for COMPLEX, LONG-RUNNING, INDEPENDENT tasks ONLY (e.g., massive web scraping, training models). NEVER use this for simple scripts, basic math, or quick commands—do those yourself using write_file and run_command. Max 5 concurrent agents. The system alerts you when they finish. Do not poll them."
        )
        self.worker = worker
        self.llm_client = llm_client
        self.max_concurrent = 5

    @property
    def output_dir(self):
        return Path(os.getcwd()) / "aeon_output" / "sub_agents"

    def execute(self, objective: str, model_name: str = None):
        # Check limit
        running = self._get_running_agents()
        if len(running) >= self.max_concurrent:
            return f"COMMAND FAILED: Maximum concurrent sub-agents ({self.max_concurrent}) reached. Kill one or wait for completion."

        agent_id = str(uuid.uuid4())
        agent_dir = self.output_dir / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinked workspace
        workspace_path = Path(os.getcwd())
        symlink_path = agent_dir / "workspace"
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(workspace_path)
        
        # Determine model config dynamically from the parent worker
        model_cfg = getattr(self.worker, 'model_config', None)
        if not model_cfg:
            model_cfg = {
                "model": model_name or "Qwen3-Coder-Next-Abliterated-Q8_0",
                "provider": "llamacpp",
                "base_url": "http://localhost:8007/v1",
                "context_limit": 262144
            }

        
        cmd = [
            sys.executable, "-m", "aeon.scripts.sub_agent_wrapper",
            "--agent_id", agent_id,
            "--objective", objective,
            "--model_config", json.dumps(model_cfg),
            "--workspace", str(symlink_path),
            "--output_dir", str(agent_dir),
            "--max_iterations", "20"
        ]
        
        # Redirect stdout/stderr directly to the log file to prevent OS pipe buffer deadlocks
        log_file_path = agent_dir / "agent.log"
        log_fd = open(log_file_path, "a")
        process = subprocess.Popen(cmd, stdout=log_fd, stderr=subprocess.STDOUT)
        
        # Save PID
        with open(agent_dir / "pid.txt", "w") as f:
            f.write(str(process.pid))
        with open(agent_dir / "status.txt", "w") as f:
            f.write("RUNNING")
            
        return f"Sub-agent spawned successfully. Agent ID: {agent_id}. The system will notify you when it finishes. DO NOT poll it. You MUST proceed with other work."

    def _get_running_agents(self):
        if not self.output_dir.exists():
            return []
        running = []
        for agent_dir in self.output_dir.iterdir():
            if agent_dir.is_dir():
                status_path = agent_dir / "status.txt"
                if status_path.exists():
                    status = status_path.read_text().strip()
                    if status == "RUNNING":
                        running.append(agent_dir.name)
        return running

class GetSubAgentReport(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="get_sub_agent_report",
            description="Checks a sub-agent's progress. If COMPLETED, returns the final findings. If RUNNING, analyzes its recent logs to provide a synthesized progress report."
        )
        self.llm_client = llm_client

    @property
    def output_dir(self):
        return Path(os.getcwd()) / "aeon_output" / "sub_agents"

    def execute(self, agent_id: str, specific_question: str = None):
        agent_dir = self.output_dir / agent_id
        if not agent_dir.exists():
            return f"Agent {agent_id} not found."
        
        status_path = agent_dir / "status.txt"
        output_path = agent_dir / "output.json"
        
        status = "UNKNOWN"
        if status_path.exists():
            status = status_path.read_text().strip()
            
        report = f"Agent {agent_id} Status: {status}"
        
        if status == "COMPLETED" and output_path.exists():
            try:
                report_data = json.loads(output_path.read_text())
                report += f"\nResult: {report_data.get('result', 'N/A')[:500]}"
            except:
                pass
        
        elif status == "RUNNING" and self.llm_client:
            log_path = agent_dir / "agent.log"
            log_tail = ""
            if log_path.exists():
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        log_tail = "".join(lines[-150:])
                except Exception as e:
                    log_tail = f"(Could not read log: {e})"
            
            if log_tail:
                prompt = (
                    f"You are a master AI agent monitoring a sub-agent's progress.\n"
                    f"Analyze the following recent log tail from sub-agent '{agent_id}'.\n"
                    f"1. What progress has been made recently?\n"
                    f"2. Is the agent stuck, looping, or blocked?\n"
                    f"3. Are there critical framework errors?\n"
                    f"4. Recommendation: Should the main agent keep waiting, intervene, or kill it?\n"
                )
                if specific_question:
                    prompt += f"\nAlso answer this specific question from the main agent: {specific_question}\n"
                prompt += f"\n--- RECENT LOG TAIL ---\n{log_tail}\n--- END LOG ---"

                try:
                    resp = self.llm_client.utility_client.chat.completions.create(
                        model=self.llm_client.utility_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )
                    analysis = resp.choices[0].message.content
                    report += f"\n\n[LIVE PROGRESS ANALYSIS]\n{analysis}"
                except Exception as e:
                    report += f"\n\n[LIVE PROGRESS ANALYSIS FAILED]: {e}\nRaw log tail:\n{log_tail[-1000:]}"
            else:
                report += "\n\n[LIVE PROGRESS ANALYSIS FAILED]: No log data found yet."
                
            # Harsh reminder to stop the agent from busy-waiting
            report += "\n\n[CRITICAL INSTRUCTION] The sub-agent is still RUNNING. DO NOT call get_sub_agent_report again in the next iteration. You MUST go do other work, or if you have nothing else to do, use task_complete. The system will automatically notify you when it finishes."
                
        return report

class KillSubAgent(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="kill_sub_agent",
            description="Kills a running sub-agent by Agent ID. Use when a sub-agent is taking too long, is stuck, or its result is no longer relevant."
        )

    @property
    def output_dir(self):
        return Path(os.getcwd()) / "aeon_output" / "sub_agents"

    def execute(self, agent_id: str):
        agent_dir = self.output_dir / agent_id
        if not agent_dir.exists():
            return f"Agent {agent_id} not found."
        
        pid_path = agent_dir / "pid.txt"
        if pid_path.exists():
            pid = int(pid_path.read_text().strip())
            try:
                os.kill(pid, 9)
                with open(agent_dir / "status.txt", "w") as f:
                    f.write("KILLED")
                return f"Sub-agent {agent_id} (PID {pid}) killed successfully."
            except ProcessLookupError:
                return f"Sub-agent {agent_id} (PID {pid}) already dead."
            except Exception as e:
                return f"Failed to kill sub-agent {agent_id}: {e}"
        else:
            return f"PID not found for agent {agent_id}."
