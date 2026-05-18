import sys
import os
from pathlib import Path

# Ensure we are using the local source code, not the installed package
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from aeon.core.worker import Worker

class MockLLM:
    """A stateful LLM that simulates a multi-turn interaction."""
    def __init__(self):
        self.turn = 0

    def get_primary_agent_response(self, prompt):
        self.turn += 1
        # Turn 1: Decide to use a tool
        if self.turn == 1:
            return '{"thought": "I will use the tool", "actions": [{"tool_name": "test_tool", "parameters": {"x": 1}}]}'
        # Turn 2: Decide to use the tool again
        elif self.turn == 2:
            return '{"thought": "I will use it again", "actions": [{"tool_name": "test_tool", "parameters": {"x": 2}}]}'
        # Turn 3: Complete the task
        else:
            return '{"thought": "Done", "actions": [{"tool_name": "task_complete", "parameters": {"reason": "verified"}}]}'

class MockTool:
    """A proper tool class to avoid AttributeError."""
    def __init__(self, name):
        self.name = name
    def __call__(self, args):
        return f"Tool {self.name} executed with {args}"
    def execute(self, args):
        return self.__call__(args)

def test_telemetry_flow():
    print("Starting Telemetry Flow Validation...")
    
    telemetry_logs = []
    def mock_telemetry_callback(iteration, intent, observation):
        telemetry_logs.append({
            "iteration": iteration,
            "intent": intent,
            "observation": observation
        })
        print(f"[CALLBACK] Iter: {iteration}, Intent: {intent}")

    # Setup
    llm = MockLLM()
    worker = Worker(llm_client=llm, telemetry_callback=mock_telemetry_callback)
    
    # Register a proper MockTool object
    tools = {"test_tool": MockTool("test_tool")}
    worker.register_tools(tools)

    # Run the worker
    worker.run("Test objective", max_iterations=5)

    print("\nTelemetry Logs Collected:")
    for log in telemetry_logs:
        print(log)

    # Validation
    # We expect at least 3 iterations. 
    # Each iteration should have:
    # 1. A 'Thinking...' heartbeat at the start.
    # 2. An 'Executing' or 'test_tool' heartbeat after the LLM response.
    
    if len(telemetry_logs) < 3:
        print("FAIL: Not enough telemetry events captured.")
        return False
    
    # Check if we have the initial 'Thinking...' heartbeats
    thinking_count = sum(1 for log in telemetry_logs if log['intent'] == "Thinking...")
    if thinking_count < 3:
        print(f"FAIL: Expected at least 3 'Thinking...' heartbeats, found {thinking_count}")
        return False

    # Check if we have the tool-specific heartbeats
    tool_count = sum(1 for log in telemetry_logs if log['intent'] == "test_tool")
    if tool_count < 2:
        print(f"FAIL: Expected at least 2 'test_tool' heartbeats, found {tool_count}")
        return False

    print("\nSUCCESS: Telemetry flow verified across multiple iterations!")
    return True

if __name__ == "__main__":
    try:
        if test_telemetry_flow():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)