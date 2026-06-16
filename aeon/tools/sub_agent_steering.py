import os
from pathlib import Path
from aeon.tools.base import BaseTool
from aeon.core.logger import get_logger

logger = get_logger()


def _agent_dir(worker, sub_agent_id: str) -> Path:
    """Resolve a sub-agent's directory.

    Sub-agents are created by SpawnSubAgent under the PARENT worker's
    instance_id: <cwd>/aeon_output/<instance_id>/sub_agents/<agent_id>.
    These tools are called by the primary agent (which owns `worker`), so its
    instance_id is the correct scope. The previous implementation omitted
    instance_id entirely and silently pointed at a directory that never
    contained the targeted agent.
    """
    instance_id = getattr(worker, "instance_id", "default")
    return Path(os.getcwd()) / "aeon_output" / instance_id / "sub_agents" / sub_agent_id


class SubAgentSteering(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="steer_sub_agent",
            description=(
                "Writes mid-execution guidance to a running sub-agent's steering.txt file. "
                "Use to course-correct a sub-agent without killing it."
            )
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, sub_agent_id: str = None, guidance: str = None) -> str:
        if not sub_agent_id or not guidance:
            return "Error: Missing 'sub_agent_id' or 'guidance' parameters."

        agent_dir = _agent_dir(self.worker, sub_agent_id)
        if not agent_dir.exists():
            return f"Error: Sub-agent '{sub_agent_id}' not found at {agent_dir}."

        steering_file = agent_dir / "steering.txt"
        try:
            steering_file.write_text(guidance, encoding="utf-8")
            logger.info(f"Steering guidance written for sub-agent {sub_agent_id}: {guidance[:100]}...")
            return f"Steering guidance written for sub-agent {sub_agent_id}."
        except Exception as e:
            logger.error(f"Failed to write steering file for {sub_agent_id}: {e}")
            return f"Error sending steering guidance: {e}"


class GetSubAgentStatus(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="get_sub_agent_status",
            description=(
                "Returns the current status (RUNNING / COMPLETED / FAILED / KILLED) of a sub-agent. "
                "Lightweight status-only check; use get_sub_agent_report for the full findings + live analysis."
            )
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, sub_agent_id: str = None) -> str:
        if not sub_agent_id:
            return "Error: Missing 'sub_agent_id' parameter."

        status_file = _agent_dir(self.worker, sub_agent_id) / "status.txt"
        if status_file.exists():
            status = status_file.read_text().strip()
        else:
            status = "Unknown (sub-agent not found or not yet started)"
        return f"Sub-Agent {sub_agent_id} Status: {status}"
