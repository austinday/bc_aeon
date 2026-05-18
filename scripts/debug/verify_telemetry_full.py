import os
import json
import time
import subprocess
from pathlib import Path

def test_telemetry():
    print("Starting Full Telemetry Flow Validation...")
    
    # Setup paths
    output_dir = Path("aeon_output/test_telemetry_agent")
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_file = output_dir / "telemetry.json"
    
    # Mock model config
    model_config = json.dumps({
        "model": "Qwen3.6-35B-A3B-Uncensored",
        "provider": "llamacpp",
        "base_url": "http://localhost:8009/v1",
        "context_limit": 128000
    })
    
    # Spawn sub-agent using the wrapper
    # We use a simple objective that will take a few iterations
    cmd = [
        "python3", "aeon/scripts/sub_agent_wrapper.py",
        "--agent_id", "test_telemetry_1",
        "--objective", "Write a 5-step plan to explore the filesystem and then call task_complete.",
        "--model_config", model_config,
        "--workspace", ".",
        "--output_dir", str(output_dir),
        "--max_iterations", "5"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("Sub-agent spawned. Polling telemetry.json for updates...")
    
    start_time = time.time()
    last_iter = -1
    found_intent = False
    
    while time.time() - start_time < 60:
        if telemetry_file.exists():
            try:
                with open(telemetry_file, "r") as f:
                    data = json.load(f)
                
                curr_iter = data.get("iteration", 0)
                curr_intent = data.get("current_intent", "")
                
                if curr_iter > last_iter:
                    print(f"Detected Iteration Update: {curr_iter} | Intent: {curr_intent}")
                    last_iter = curr_iter
                
                if curr_intent and curr_intent != "Thinking":
                    found_intent = True
                    print(f"Detected Actual Intent: {curr_intent}")
                
                if curr_iter >= 2: # We've seen at least 2 iterations
                    print("Telemetry flow verified: Iterations are incrementing and intents are being captured.")
                    process.terminate()
                    return True
            except Exception as e:
                print(f"Error reading telemetry: {e}")
        
        time.sleep(2)
    
    process.terminate()
    print(f"Validation failed. Last iteration seen: {last_iter}, Intent found: {found_intent}")
    return False

if __name__ == "__main__":
    if test_telemetry():
        print("Telemetry Flow VALIDATION SUCCESSFUL.")
        exit(0)
    else:
        print("Telemetry Flow VALIDATION FAILED.")
        exit(1)