import json
import os
from pathlib import Path
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient

class MockLLMClient(LLMClient):
    def __init__(self, *args, **kwargs):
        # Bypass initialization that might require API keys or servers
        self.context_limit = 128000
    def get_primary_agent_response(self, prompt, diagnostic_str=""):
        return json.dumps({
            "thought": "I am simulating a thought process for verification.",
            "previous_result_summary": "The previous step was successful.",
            "intent": "I will now verify the logging mechanism.",
            "updated_plan": "1. Test log\n2. Verify output",
            "actions": [{"tool_name": "say_to_user", "parameters": {"message": "Hello world"}}]
        })
    def analyze_interruption(self, objective, user_in):
        return {"classification": "ADVICE", "updated_text": user_in, "reasoning": "ok"}
    def compress_action_log(self, text):
        return "compressed log"
    def compress_memories(self, text):
        return {"mem": "compressed"}

def test_logging():
    log_file = "test_reasoning_trace.jsonl"
    if Path(log_file).exists():
        Path(log_file).unlink()
    
    # Initialize Worker with debug mode and a specific log path
    llm = MockLLMClient()
    # We pass a dummy list of tools to avoid prompt loading errors in the mock
    worker = Worker(llm_client=llm, tools=[], debug_mode=True, debug_log_path=log_file)
    
    # Test data that mimics what is captured in the Worker.run loop
    trace_data = {
        "iteration": 1,
        "previous_result_summary": "Summary of previous result",
        "thought": "Detailed reasoning thought",
        "intent": "Intent to perform action",
        "updated_plan": "Step 1: Action A\nStep 2: Action B",
        "actions": [{"tool_name": "run_command", "parameters": {"command": "ls"}}],
        "result": "Output of the command"
    }
    
    print(f"Testing _log_reasoning_trace with log file: {log_file}...")
    worker._log_reasoning_trace(1, trace_data)
    
    if not Path(log_file).exists():
        print("FAILED: Log file was not created.")
        exit(1)
        
    with open(log_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        if not line:
            print("FAILED: Log file is empty.")
            exit(1)
        
        try:
            data = json.loads(line)
            print("Successfully parsed JSONL entry.")
            
            # Verify all required fields are present
            required_fields = ["iteration", "previous_result_summary", "thought", "intent", "updated_plan", "actions", "result"]
            for field in required_fields:
                if field not in data:
                    print(f"FAILED: Missing field {field} in log entry.")
                    exit(1)
                if data[field] != trace_data[field]:
                    print(f"FAILED: Field {field} value mismatch. Expected {trace_data[field]}, got {data[field]}")
                    exit(1)
            
            print("All fields verified correctly!")
        except json.JSONDecodeError as e:
            print(f"FAILED: Log entry is not valid JSON: {e}")
            exit(1)

    print("\nVERIFICATION SUCCESSFUL: Reasoning trace logging is implemented and producing correct JSONL output.")

if __name__ == "__main__":
    test_logging()