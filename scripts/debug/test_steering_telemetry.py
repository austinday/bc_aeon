import os
import json
import time
from pathlib import Path
from typing import Dict, Any

# Import the actual Worker and LLMClient from the local package
try:
    from aeon.core.worker import Worker
    from aeon.core.llm import LLMClient
except ImportError:
    print("Error: Could not import aeon. Please ensure 'pip install -e .' was run.")
    exit(1)

class MockLLM:
    """A mock LLM that returns a sequence of predefined responses."""
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def get_primary_agent_response(self, prompt: str) -> str:
        if self.call_count >= len(self.responses):
            # Return a default 'task_complete' response if we run out of predefined ones
            return json.dumps({
                "thought": "All tasks finished.",
                "actions": [{"tool_name": "task_complete", "parameters": {"reason": "Mock finished"}}]
            })
        
        resp = self.responses[self.call_count]
        self.call_count += 1
        return json.dumps(resp)

class MockTool:
    """A mock tool that records calls and allows attribute assignment (like .worker)."""
    def __init__(self, name):
        self.name = name
        self.calls = []
        self.worker = None # Explicitly allow this for compatibility with some Worker versions

    def __call__(self, args):
        self.calls.append(args)
        return f"Tool {self.name} executed with {args}"

    def execute(self, args):
        return self.__call__(args)

def test_steering_and_telemetry():
    print("--- Starting Steering & Telemetry Validation ---")
    
    # 1. Setup Mock LLM to run for 3 iterations
    # Iter 1: Call test_tool, Iter 2: Call test_tool, Iter 3: task_complete
    mock_responses = [
        {"thought": "I will call the tool first", "actions": [{"tool_name": "test_tool", "parameters": {"x": 1}}]},
        {"thought": "I will call the tool again", "actions": [{"tool_name": "test_tool", "parameters": {"x": 2}}]},
        {"thought": "Now I am done", "actions": [{"tool_name": "task_complete", "parameters": {"reason": "done"}}]}
    ]
    llm = MockLLM(mock_responses)
    
    # 2. Setup Telemetry Callback
    telemetry_log = []
    def telemetry_callback(iteration, intent, observation):
        print(f"[TELEMETRY] Iter {iteration} | Intent: {intent} | Obs: {observation[:50]}...")
        telemetry_log.append((iteration, intent))

    # 3. Initialize Worker
    worker = Worker(llm_client=llm, telemetry_callback=telemetry_callback)
    
    # 4. Register Mock Tool
    test_tool = MockTool("test_tool")
    worker.register_tools({"test_tool": test_tool})
    
    # 5. Run the worker
    print("\nRunning worker loop...")
    worker.run("Test objective", max_iterations=5)
    
    # 6. Validations
    print("\n--- Validation Results ---")
    
    # Check if telemetry was called
    # Expected: 
    # Iter 1: Thinking...
    # Iter 1: test_tool
    # Iter 2: Thinking...
    # Iter 2: test_tool
    # Iter 3: Thinking...
    # Iter 3: task_complete
    print(f"Telemetry calls recorded: {len(telemetry_log)}")
    assert len(telemetry_log) >= 3, "Telemetry should have been called at least 3 times"
    
    # Check if tool was actually called
    print(f"Tool calls recorded: {len(test_tool.calls)}")
    assert len(test_tool.calls) == 2, "test_tool should have been called exactly 2 times"
    
    print("\n[SUCCESS] Steering and Telemetry validation passed!")

if __name__ == "__main__":
    try:
        test_steering_and_telemetry()
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)