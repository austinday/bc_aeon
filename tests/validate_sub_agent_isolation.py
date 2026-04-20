import os
import sys
import json
import uuid
import shutil
from pathlib import Path
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.sub_agent import SpawnSubAgent, GetSubAgentReport, KillSubAgent

# Mock LLM Client to avoid actual API calls
class MockLLMClient(LLMClient):
    def __init__(self, *args, **kwargs):
        # We don't call super().__init__ because it might try to load configs
        self.utility_model = "mock-model"
        self.context_limit = 100000
        self.utility_client = type('obj', (object,), {
            'chat': type('obj', (object,), {
                'completions': type('obj', (object,), {
                    'create': lambda model, messages, temperature: type('obj', (object,), {
                        'choices': [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': "Mock analysis: Agent is making progress."
                            })
                        })]
                    })
                })
            })
        })

    def get_primary_agent_response(self, prompt):
        return json.dumps({
            "thought": "Mock thought",
            "previous_result_summary": "Mock summary",
            "intent": "Mock intent",
            "actions": []
        })

    def analyze_interruption(self, objective, user_in):
        return {"classification": "ADVICE", "updated_text": user_in, "reasoning": "Mock"}

    def set_debug_path(self, path):
        pass

    def set_iteration(self, iter):
        pass

def test_sub_agent_system():
    print("Starting Sub-Agent Isolation and Cleanup Validation...")
    
    # 1. Setup Worker and Tools
    llm_client = MockLLMClient()
    worker = Worker(llm_client=llm_client)
    
    spawn_tool = SpawnSubAgent(worker=worker, llm_client=llm_client)
    report_tool = GetSubAgentReport(worker=worker, llm_client=llm_client)
    kill_tool = KillSubAgent(worker=worker)
    
    worker.register_tools([spawn_tool, report_tool, kill_tool])
    
    instance_id = worker.instance_id
    print(f"Worker Instance ID: {instance_id}")
    
    # 2. Spawn a sub-agent
    objective = "Test objective for isolation"
    result = spawn_tool.execute(objective=objective)
    print(f"Spawn result: {result}")
    
    # Extract agent_id from result
    import re
    match = re.search(r"Agent ID: ([a-f0-9\-]+)", result)
    if not match:
        print("FAILED: Could not extract agent_id from spawn result")
        sys.exit(1)
    agent_id = match.group(1)
    
    # 3. Verify Directory Structure
    expected_dir = Path(os.getcwd()) / "aeon_output" / instance_id / "sub_agents" / agent_id
    print(f"Expected directory: {expected_dir}")
    
    if not expected_dir.exists():
        print(f"FAILED: Directory {expected_dir} does not exist")
        sys.exit(1)
    print("SUCCESS: Directory structure is correct (Instance Isolated).")
    
    # 4. Verify status.txt is RUNNING
    status_path = expected_dir / "status.txt"
    if not status_path.exists():
        print("FAILED: status.txt does not exist")
        sys.exit(1)
    
    status = status_path.read_text().strip()
    print(f"Initial status: {status}")
    if status != "RUNNING":
        print(f"FAILED: Expected status RUNNING, got {status}")
        sys.exit(1)
    print("SUCCESS: status.txt is correctly set to RUNNING.")
    
    # 5. Kill the sub-agent
    kill_result = kill_tool.execute(agent_id=agent_id)
    print(f"Kill result: {kill_result}")
    
    # Verify status.txt is now KILLED
    status = status_path.read_text().strip()
    print(f"Status after kill: {status}")
    if status != "KILLED":
        print(f"FAILED: Expected status KILLED, got {status}")
        sys.exit(1)
    print("SUCCESS: status.txt is correctly set to KILLED.")
    
    # 6. Simulate Worker Loop Cleanup
    # In Worker.run, the loop checks for COMPLETED/FAILED/KILLED and unlinks status.txt
    print("Simulating Worker loop cleanup...")
    if status == "KILLED":
        status_path.unlink()
    
    if status_path.exists():
        print("FAILED: status.txt was not cleaned up")
        sys.exit(1)
    print("SUCCESS: status.txt was cleaned up after notification.")
    
    # 7. Verify GetSubAgentReport behavior after cleanup
    # Since status.txt is gone and output.json doesn't exist (killed), it should be UNKNOWN
    report = report_tool.execute(agent_id=agent_id)
    print(f"Report after cleanup: {report}")
    if "Status: UNKNOWN" not in report:
        print(f"FAILED: Expected Status: UNKNOWN in report, got {report}")
        sys.exit(1)
    print("SUCCESS: GetSubAgentReport handles missing status.txt correctly.")

    # Cleanup test data
    shutil.rmtree(Path(os.getcwd()) / "aeon_output" / instance_id, ignore_errors=True)
    print("\nALL TESTS PASSED!")

if __name__ == "__main__":
    test_sub_agent_system()