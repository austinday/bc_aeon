import os
import json
import time
import sys
from pathlib import Path
from aeon.core.worker import Worker

class MockLLM:
    def __init__(self):
        self.context_limit = 128000

    def get_primary_agent_response(self, prompt, diagnostic_str=""):
        # Return a valid JSON response that tells the worker to take an action and then complete
        # We use a simple state machine in the mock to simulate a few iterations
        if not hasattr(self, 'call_count'):
            self.call_count = 0
        
        self.call_count += 1
        
        if self.call_count < 3:
            return json.dumps({
                "thought": f"Iteration {self.call_count}: I am testing telemetry.",
                "previous_result_summary": f"Iteration {self.call_count-1} complete.",
                "intent": f"Testing iteration {self.call_count}",
                "updated_plan": "Continue testing",
                "actions": [{"tool_name": "mock_tool", "parameters": {"val": self.call_count}}]
            })
        else:
            return json.dumps({
                "thought": "Finished testing.",
                "previous_result_summary": "All iterations complete.",
                "intent": "Complete task",
                "updated_plan": "Done",
                "actions": [{"tool_name": "task_complete", "parameters": {"reason": "Telemetry test finished"}}]
            })

    def analyze_interruption(self, objective, user_in):
        return {"classification": "ADVICE", "updated_text": user_in, "reasoning": "mock"}

    def compress_action_log(self, text):
        return "Compressed log"

    def compress_memories(self, text):
        return {}

class MockTool:
    def __init__(self, name):
        self.name = name
        self.description = "A mock tool for testing"
    def execute(self, **params):
        return f"Mock tool {self.name} executed with {params}"

def test_telemetry():
    print("Starting Telemetry Live Test...")
    
    # Setup paths
    output_dir = Path("aeon_output/test_telemetry")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock LLM and Worker
    llm = MockLLM()
    worker = Worker(llm_client=llm)
    
    # Register mock tool
    mock_tool = MockTool("mock_tool")
    worker.register_tools([mock_tool])
    
    # Define telemetry callback
    telemetry_path = output_dir / "telemetry.json"
    def update_telemetry(iteration, max_iters, step_desc):
        print(f"Telemetry Callback: Iter {iteration}, Step: {step_desc}")
        telemetry = {
            "iteration": iteration,
            "max_iterations": max_iters,
            "current_step": step_desc,
            "timestamp": time.time()
        }
        with open(telemetry_path, "w") as f:
            json.dump(telemetry, f, indent=2)

    try:
        # Run the worker for a few iterations
        # We use a small max_iterations to keep the test fast
        worker.run(
            objective="Test telemetry updates", 
            max_iterations=5, 
            step_callback=update_telemetry
        )
    except Exception as e:
        print(f"Worker run encountered error: {e}")

    # Verify telemetry file
    if telemetry_path.exists():
        with open(telemetry_path, "r") as f:
            data = json.load(f)
            print(f"Final Telemetry: {data}")
            if data['iteration'] >= 3:
                print("SUCCESS: Telemetry updated across multiple iterations.")
                return True
    
    print("FAILURE: Telemetry not updated as expected.")
    return False

if __name__ == "__main__":
    success = test_telemetry()
    sys.exit(0 if success else 1)