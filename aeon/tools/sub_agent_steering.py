import os
import json
import time
import fcntl
from pathlib import Path
from aeon.tools.base import BaseTool
from aeon.core.logger import get_logger
from aeon.core.sub_agent_state import resolve
from aeon.tools.sub_agent import _output_dir_for_worker, _resolve_agent_dir

logger = get_logger()


def _sub_agents_base(worker):
    return _output_dir_for_worker(worker)


class SubAgentSteering(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="steer_sub_agent",
            description=(
                "Send mid-execution guidance to a RUNNING sub-agent without killing it. The sub-agent folds "
                "your guidance into its context at the start of its next iteration. Guidance is QUEUED and "
                "ordered, so you can fire several course-corrections and none are lost. Accepts the short id "
                "shown by gather_sub_agents or a full UUID. Use it as a PI would: redirect an approach, "
                "inject a hypothesis to test, narrow/widen scope, or relay a fact another sub-agent found.\n"
                "Schema:\n"
                "  sub_agent_id (str, required): short id or full UUID.\n"
                "  guidance (str, required)\n"
                "Example: {\"tool_name\": \"steer_sub_agent\", \"parameters\": {\"sub_agent_id\": \"9e8d4039\", "
                "\"guidance\": \"Focus on post-2023 sources; drop the pricing angle.\"}}"
            ),
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, sub_agent_id=None, guidance=None):
        if not sub_agent_id or not guidance:
            return "Error: both 'sub_agent_id' and 'guidance' are required."
        agent_dir, err = _resolve_agent_dir(_sub_agents_base(self.worker), sub_agent_id)
        if err:
            return err
        if resolve(agent_dir)[0]:
            return (f"Sub-agent '{agent_dir.name[:8]}' is already terminal; steering has no effect. "
                    f"Use get_sub_agent_report to read its result.")
        steering_path = agent_dir / "steering.jsonl"
        entry = json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "guidance": str(guidance)})
        try:
            with open(steering_path, "a", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(entry + "\n")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            logger.info(f"Queued steering for {agent_dir.name[:8]}: {str(guidance)[:100]}")
            return (f"Steering queued for sub-agent {agent_dir.name[:8]}; it will be applied at the start of "
                    f"its next iteration.")
        except Exception as e:
            logger.error(f"Failed to queue steering for {agent_dir.name[:8]}: {e}")
            return f"Error queuing steering guidance: {e}"


class GetSubAgentStatus(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="get_sub_agent_status",
            description=(
                "Lightweight status of a single sub-agent: terminal (COMPLETED/FAILED/KILLED) or RUNNING. "
                "Accepts the short id shown by gather_sub_agents or a full UUID. Use get_sub_agent_report "
                "for full findings or live analysis, and gather_sub_agents for a whole-batch check-in.\n"
                "Schema:\n  sub_agent_id (str, required): short id or full UUID.\n"
                "Example: {\"tool_name\": \"get_sub_agent_status\", \"parameters\": {\"sub_agent_id\": \"9e8d4039\"}}"
            ),
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, sub_agent_id=None):
        if not sub_agent_id:
            return "Error: missing 'sub_agent_id' parameter."
        agent_dir, err = _resolve_agent_dir(_sub_agents_base(self.worker), sub_agent_id)
        if err:
            return err
        _, status, _ = resolve(agent_dir)
        return f"Sub-Agent {agent_dir.name[:8]} Status: {status}"
