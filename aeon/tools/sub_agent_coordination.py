import os
from pathlib import Path
from typing import Dict, Any
from aeon.tools.base import BaseTool
from aeon.core.logger import get_logger

logger = get_logger()

class SubAgentSharedWrite(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="sub_agent_shared_write",
            description="Writes data to the shared space of a specific sub-agent."
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, agent_id: str, key: str, value: str) -> str:
        if not agent_id or not key:
            return "Error: agent_id and key are required."
        
        try:
            # Path aligned with sub_agent_wrapper.py: aeon_output/sub_agents/<id>/shared_space/
            shared_dir = Path("aeon_output/sub_agents") / agent_id / "shared_space"
            shared_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = shared_dir / f"{key}.txt"
            file_path.write_text(value)
            
            return f"Successfully wrote to shared space for agent {agent_id}: {key} = {value[:50]}..."
        except Exception as e:
            logger.error(f"Failed to write to shared space: {e}")
            return f"Error writing to shared space: {e}"

class SubAgentSharedRead(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="sub_agent_shared_read",
            description="Reads data from the shared space of a specific sub-agent."
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, agent_id: str, key: str) -> str:
        if not agent_id or not key:
            return "Error: agent_id and key are required."
        
        try:
            shared_dir = Path("aeon_output/sub_agents") / agent_id / "shared_space"
            file_path = shared_dir / f"{key}.txt"
            
            if not file_path.exists():
                return f"Key '{key}' not found in shared space for agent {agent_id}."
                
            content = file_path.read_text()
            return content
        except Exception as e:
            logger.error(f"Failed to read from shared space: {e}")
            return f"Error reading from shared space: {e}"