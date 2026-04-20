import json
from unittest.mock import MagicMock
from aeon.core.worker import Worker

class MockLLMClient:
    def __init__(self):
        self.context_limit = 100000
        self.iteration = 0
    def set_iteration(self, i): 
        self.iteration = i
    def get_primary_agent_response(self, prompt):
        # Iteration 1: Call tool with long path 1
        if self.iteration == 1:
            return json.dumps({
                "thought": "Iteration 1",
                "previous_result_summary": "N/A",
                "intent": "Call tool 1",
                "actions": [
                    {"tool_name": "test_tool", "parameters": {"path": "/home/user/project/very/long/path/to/file_1.txt"}}
                ]
            })
        # Iteration 2: Call tool with long path 2 (differs only at the end)
        elif self.iteration == 2:
            return json.dumps({
                "thought": "Iteration 2",
                "previous_result_summary": "Success",
                "intent": "Call tool 2",
                "actions": [
                    {"tool_name": "test_tool", "parameters": {"path": "/home/user/project/very/long/path/to/file_2.txt"}}
                ]
            })
        # Iteration 3: Trigger completion
        else:
            return json.dumps({
                "thought": "Done",
                "previous_result_summary": "Success",
                "intent": "Complete",
                "actions": [
                    {"tool_name": "task_complete", "parameters": {"reason": "Test finished"}}
                ]
            })

class MockTool:
    def __init__(self, name):
        self.name = name
        self.description = "Test tool"
    def execute(self, **params):
        return "Fixed output for loop detection"

def test_loop_detection_fix():
    print("Testing loop detection with DIFFERENT long arguments...")
    llm = MockLLMClient()
    tool = MockTool("test_tool")
    # We need to provide the terminal tool 'task_complete' as well
    class TerminalTool:
        def __init__(self, name): self.name = name; self.description = "Terminal"
        def execute(self, **params): return "Completed"
    
    worker = Worker(llm, tools=[tool, TerminalTool("task_complete")], debug_mode=False)
    
    # Run the worker
    worker.run(objective="Test loop detection", max_iterations=5)
    
    # Check if loop warning is in the last observation or action log
    # The loop warning is appended to self.last_observation
    if "LOOP DETECTED" in worker.last_observation:
        print("\n❌ BUG REPRODUCED: Loop detected despite different arguments!")
        return False
    else:
        print("\n✅ SUCCESS: No false positive loop detected for different arguments.")
        return True

def test_loop_detection_still_works():
    print("\nTesting loop detection with IDENTICAL arguments...")
    class LoopLLMClient(MockLLMClient):
        def get_primary_agent_response(self, prompt):
            if self.iteration <= 3:
                return json.dumps({
                    "thought": "Looping",
                    "previous_result_summary": "Success",
                    "intent": "Loop",
                    "actions": [
                        {"tool_name": "test_tool", "parameters": {"path": "/home/user/project/same_path.txt"}}
                    ]
                })
            return json.dumps({
                "thought": "Done",
                "previous_result_summary": "Success",
                "intent": "Complete",
                "actions": [{"tool_name": "task_complete", "parameters": {"reason": "Test finished"}}]
            })

    llm = LoopLLMClient()
    tool = MockTool("test_tool")
    class TerminalTool:
        def __init__(self, name): self.name = name; self.description = "Terminal"
        def execute(self, **params): return "Completed"
    
    worker = Worker(llm, tools=[tool, TerminalTool("task_complete")], debug_mode=False)
    worker.run(objective="Test loop detection", max_iterations=5)
    
    if "LOOP DETECTED" in worker.last_observation:
        print("✅ SUCCESS: Loop correctly detected for identical arguments.")
        return True
    else:
        print("❌ FAILURE: Loop NOT detected for identical arguments!")
        return False

if __name__ == "__main__":
    res1 = test_loop_detection_fix()
    res2 = test_loop_detection_still_works()
    if res1 and res2:
        print("\nALL TESTS PASSED: Loop detection is now robust.")
    else:
        print("\nSOME TESTS FAILED.")
        exit(1)