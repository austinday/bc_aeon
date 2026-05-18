import os
from pathlib import Path
from typing import Dict, Any
from aeon.tools.base import BaseTool
from aeon.core.logger import get_logger

logger = get_logger()

class SubAgentSteering(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="steer_sub_agent",
            description="Provides mid-execution guidance (steering) to a running sub-agent by writing to its steering.txt file."
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, sub_agent_id: str, guidance: str) -> str:
        if not sub_agent_id or not guidance:
            return "Error: Missing 'sub_agent_id' or 'guidance' parameters."
        
        # Path aligned with sub_agent_wrapper.py
        steering_file = Path("aeon_output") / "sub_agents" / sub_agent_id / "steering.txt"
        
        try:
            steering_file.parent.mkdir(parents=True, exist_ok=True)
            steering_file.write_text(guidance, encoding='utf-8')
            logger.info(f"Steering guidance sent to sub-agent {sub_agent_id}: {guidance[:100]}...")
            return f"Successfully sent steering guidance to sub-agent {sub_agent_id}."
        except Exception as e:
            logger.error(f"Failed to write steering file for {sub_agent_id}: {e}")
            return f"Error sending steering guidance: {e}"

class GetSubAgentStatus(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="get_sub_agent_status",
            description="Checks the current status and last observation of a sub-agent."
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, sub_agent_id: str) -> str:
        if not sub_agent_id:
            return "Error: Missing 'sub_agent_id' parameter."
        
        status_file = Path("aeon_output") / "sub_agents" / sub_agent_id / "status.txt"
        
        status = "Unknown"
        if status_file.exists():
            status = status_file.read_text().strip()
            
        return f"Sub-Agent {sub_agent_id} Status: {status}"