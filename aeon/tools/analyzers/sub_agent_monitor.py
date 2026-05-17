import json
import os
from pathlib import Path
from aeon.tools.base import BaseTool
from aeon.core.logger import get_logger

class SubAgentMonitor(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="sub_agent_monitor",
            description="Monitors the progress of a running sub-agent by reading its telemetry and logs."
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, agent_id: str) -> str:
        logger = get_logger()
        # Path aligned with sub_agent_wrapper.py
        agent_path = Path("aeon_output") / "sub_agents" / agent_id
        telemetry_file = agent_path / "telemetry.json"
        log_file = agent_path / "agent.log"
        status_file = agent_path / "status.txt"

        if not agent_path.exists():
            return f"Error: Sub-agent directory not found at {agent_path}"

        status = "Unknown"
        if status_file.exists():
            status = status_file.read_text().strip()

        telemetry_data = {}
        if telemetry_file.exists():
            try:
                with open(telemetry_file, 'r') as f:
                    telemetry_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read telemetry for {agent_id}: {e}")

        log_tail = ""
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    log_tail = "".join(lines[-20:])
            except Exception as e:
                logger.warning(f"Failed to read log for {agent_id}: {e}")

        return json.dumps({
            "agent_id": agent_id,
            "status": status,
            "telemetry": telemetry_data,
            "recent_logs": log_tail
        }, indent=2)