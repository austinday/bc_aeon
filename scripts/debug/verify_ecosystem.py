import os
import json
import time
import subprocess
from pathlib import Path

def run_sub_agent():
    print("--- Starting Sub-Agent Ecosystem Validation ---")
    
    # Setup paths
    workspace = "test_ecosystem_val/workspace"
    output_dir = "test_ecosystem_val/agent_1"
    os.makedirs(workspace, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Mock LLM config (pointing to a non-existent local server is fine because we mock the client)
    model_config = json.dumps({
        "model": "test-model",
        "provider": "local",
        "base_url": "http://localhost:8000/v1",
        "context_limit": 128000
    })

    # We need to mock the LLMClient.get_primary_agent_response to return a task_complete action immediately.
    # Since we can't easily mock the class inside the wrapper without modifying the wrapper,
    # we will create a tiny mock_llm.py and use it if possible, or just rely on the fact that 
    # we can't easily inject a mock into the wrapper process.
    # ACTUALLY: The best way to test the WRAPPER is to let it run and see if it crashes.
    # To make it finish, we can't easily mock the LLM without changing the code.
    # I will modify the wrapper to use a mock if a certain flag is passed, or just use a very simple mock server.
    
    # For this specific test, I'll just run the wrapper and check if it starts and logs.
    # Since I can't easily provide a running LLM server, I'll check for the 'TypeError' fix.
    
    cmd = [
        "python3", "aeon/scripts/sub_agent_wrapper.py",
        "--agent_id", "val_agent",
        "--objective", "Verify startup",
        "--model_config", model_config,
        "--workspace", workspace,
        "--output_dir", output_dir,
        "--max_iterations", "1",
        "--debug"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    try:
        # We expect this to fail eventually because there's no real LLM server, 
        # but we want to see if it gets PAST the register_tools and startup phase.
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(process.stdout)
        print(process.stderr)
    except subprocess.TimeoutExpired:
        print("Process timed out as expected (waiting for LLM), but let's check logs.")
    except Exception as e:
        print(f"Execution failed: {e}")

    # Validation
    log_file = Path(output_dir) / "agent.log"
    if log_file.exists():
        content = log_file.read_text()
        if "worker.register_tools completed successfully" in content:
            print("\n[SUCCESS] Sub-agent startup and tool registration verified.")
        else:
            print("\n[FAILURE] Startup sequence did not complete.")
    else:
        print("\n[FAILURE] Log file not created.")

if __name__ == "__main__":
    run_sub_agent()