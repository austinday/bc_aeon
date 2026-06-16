import os
import sys
import json
import uuid
import subprocess
import time
import copy
import re
import ctypes
import signal
from pathlib import Path
from aeon.tools.base import BaseTool
from ..core.prompts import (
    TOOL_DESC_SPAWN_SUB_AGENT,
    TOOL_DESC_GET_SUB_AGENT_REPORT,
    TOOL_DESC_KILL_SUB_AGENT
)

class SpawnSubAgent(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="spawn_sub_agent",
            description=TOOL_DESC_SPAWN_SUB_AGENT
        )
        self.worker = worker
        self.llm_client = llm_client
        self.max_concurrent = 5

    @property
    def output_dir(self):
        return Path(os.getcwd()) / "aeon_output" / self.worker.instance_id / "sub_agents"

    def execute(self, objective: str):
        # Enforce concurrency limit
        running = self._get_running_agents()
        if len(running) >= self.max_concurrent:
            return (f"COMMAND FAILED: Maximum concurrent sub-agents ({self.max_concurrent}) reached. "
                    f"Kill one (kill_sub_agent) or wait for completion (gather_sub_agents).")

        # Sub-agents ALWAYS run the same model as the primary agent. Heterogeneous
        # models are not supported: exactly one model is served per session, and the
        # sub-agent inherits the primary worker's model_config. There is no fallback
        # to a different model.
        model_cfg = getattr(self.worker, 'model_config', None)
        if not model_cfg:
            return ("COMMAND FAILED: No model_config is set on the primary worker, so a sub-agent "
                    "cannot be configured with the active model. Cannot spawn.")

        agent_id = str(uuid.uuid4())
        agent_dir = self.output_dir / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Create symlinked workspace (points at the shared task workspace)
        workspace_path = Path(os.getcwd())
        symlink_path = agent_dir / "workspace"
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(workspace_path)

        # Tell the sub-agent to coordinate through the shared blackboard.
        coordinated_objective = (
            f"{objective}\n\n"
            f"[COORDINATION] You are one of several parallel sub-agents sharing this task. "
            f"BEFORE starting a self-contained chunk of work, call blackboard_read to check whether "
            f"a sibling has already produced that result or already hit that dead end. When you "
            f"produce something reusable (a working approach, a confirmed fact, an artifact path, or "
            f"a dead end), call blackboard_post so the others can use it."
        )

        cmd = [
            sys.executable, "-m", "aeon.scripts.sub_agent_wrapper",
            "--agent_id", agent_id,
            "--objective", coordinated_objective,
            "--model_config", json.dumps(model_cfg),
            "--workspace", str(symlink_path),
            "--output_dir", str(agent_dir),
            "--max_iterations", "20"
        ]
        if getattr(self.worker, 'debug_mode', False):
            cmd.append("--debug")

        def set_pdeathsig():
            try:
                ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)
            except Exception:
                pass

        # Redirect stdout/stderr directly to the log file to prevent OS pipe buffer deadlocks
        log_file_path = agent_dir / "agent.log"
        log_fd = open(log_file_path, "a")
        process = subprocess.Popen(cmd, stdout=log_fd, stderr=subprocess.STDOUT, preexec_fn=set_pdeathsig)

        with open(agent_dir / "pid.txt", "w") as f:
            f.write(str(process.pid))
        with open(agent_dir / "status.txt", "w") as f:
            f.write("RUNNING")

        return (f"Sub-agent spawned successfully. Agent ID: {agent_id}. It runs the same model as you. "
                f"The system will notify you when it finishes. DO NOT poll it one-by-one; continue with "
                f"other orthogonal work, then call gather_sub_agents to collect the whole batch at once.")

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


class GatherSubAgents(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="gather_sub_agents",
            description=(
                "Fan-in primitive: blocks until the specified (or all currently running) sub-agents "
                "reach a terminal state (COMPLETED / FAILED / KILLED), then returns ALL of their final "
                "reports together in one result. After spawning a batch of orthogonal sub-agents, do "
                "your own orthogonal work first, then call gather_sub_agents ONCE to wait for the batch "
                "and collect every result in a single step instead of polling get_sub_agent_report.\n"
                "Schema:\n"
                "  agent_ids (list[str], optional): specific agent IDs to wait for. Omit to wait for ALL sub-agents that are currently running.\n"
                "  timeout (int, optional, default=1200): max seconds to block before returning with whatever has finished so far.\n"
                "Example: {\"tool_name\": \"gather_sub_agents\", \"parameters\": {\"timeout\": 900}}"
            )
        )
        self.worker = worker
        self.llm_client = llm_client

    @property
    def output_dir(self):
        return Path(os.getcwd()) / "aeon_output" / self.worker.instance_id / "sub_agents"

    def _resolve(self, agent_dir):
        """Return (is_terminal, status, report). Treats a present output.json OR an
        explicit terminal status.txt as terminal, because the primary loop's notifier
        may have already consumed (deleted) the status file after alerting."""
        status_path = agent_dir / "status.txt"
        output_path = agent_dir / "output.json"

        status = None
        if status_path.exists():
            try:
                status = status_path.read_text().strip()
            except Exception:
                status = None

        if output_path.exists():
            try:
                data = json.loads(output_path.read_text())
                st = data.get("status", "COMPLETED")
                if "error" in data and st != "COMPLETED":
                    return True, st, f"Error: {data['error']}"
                return True, st, str(data.get("result", "N/A"))
            except Exception as e:
                return True, (status or "COMPLETED"), f"(output.json present but unreadable: {e})"

        if status and (status in ("COMPLETED", "KILLED") or status.startswith("FAILED")):
            return True, status, "(terminal status reported, no output.json found)"

        return False, (status or "RUNNING"), None

    def execute(self, agent_ids=None, timeout: int = 1200):
        if not self.worker:
            return "Error: Worker context missing."

        base = self.output_dir
        if not base.exists():
            return "No sub-agents have been spawned in this session."

        missing = []
        if agent_ids:
            if isinstance(agent_ids, str):
                agent_ids = [agent_ids]
            targets = []
            for aid in agent_ids:
                p = base / str(aid)
                if p.exists():
                    targets.append(p)
                else:
                    missing.append(str(aid))
            if not targets:
                return f"None of the requested sub-agents were found: {agent_ids}"
        else:
            targets = [
                d for d in base.iterdir()
                if d.is_dir() and (
                    (d / "status.txt").exists() or (d / "output.json").exists() or (d / "pid.txt").exists()
                )
            ]
            if not targets:
                return "No sub-agents found to gather."

        start = time.time()
        pending = list(targets)
        while pending and (time.time() - start) < timeout:
            pending = [d for d in pending if not self._resolve(d)[0]]
            if not pending:
                break
            time.sleep(3)

        completed = failed = killed = timed_out = 0
        lines = []
        for d in targets:
            is_term, status, report = self._resolve(d)
            aid = d.name[:8]
            if not is_term:
                timed_out += 1
                lines.append(f"[{aid}] STILL RUNNING after {timeout}s (timed out of the wait).")
                continue
            if status == "COMPLETED":
                completed += 1
            elif status and status.startswith("FAILED"):
                failed += 1
            elif status == "KILLED":
                killed += 1
            snippet = (report or "")[:800]
            lines.append(f"[{aid}] {status}\n  {snippet}")

        header = (
            f"Gathered {len(targets)} sub-agent(s): {completed} completed, "
            f"{failed} failed, {killed} killed, {timed_out} still running."
        )
        if missing:
            header += f" (Requested but not found: {missing})"
        return header + "\n\n" + "\n\n".join(lines)


class GetSubAgentReport(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="get_sub_agent_report",
            description=TOOL_DESC_GET_SUB_AGENT_REPORT,
            underlying_model=llm_client.utility_model if llm_client else None
        )
        self.worker = worker
        self.llm_client = llm_client

    @property
    def output_dir(self):
        return Path(os.getcwd()) / "aeon_output" / self.worker.instance_id / "sub_agents"

    def execute(self, agent_id: str, specific_question: str = None):
        agent_dir = self.output_dir / agent_id
        if not agent_dir.exists():
            return f"Agent {agent_id} not found."

        status_path = agent_dir / "status.txt"
        output_path = agent_dir / "output.json"

        status = "UNKNOWN"
        if status_path.exists():
            status = status_path.read_text().strip()
        elif output_path.exists():
            status = "COMPLETED"

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

            report += ("\n\n[CRITICAL INSTRUCTION] The sub-agent is still RUNNING. DO NOT call "
                       "get_sub_agent_report again next iteration. Go do other work, or call "
                       "gather_sub_agents to block until the batch finishes. The system auto-notifies "
                       "you on completion.")

        return report


class KillSubAgent(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="kill_sub_agent",
            description=TOOL_DESC_KILL_SUB_AGENT
        )
        self.worker = worker

    @property
    def output_dir(self):
        return Path(os.getcwd()) / "aeon_output" / self.worker.instance_id / "sub_agents"

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
