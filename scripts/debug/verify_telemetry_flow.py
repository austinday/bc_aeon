import os
import json
import time
import subprocess
from pathlib import Path

def test_telemetry():
    print("Starting Telemetry Flow Validation...")
    
    # Setup paths
    output_dir = Path("aeon_output/telemetry_test").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock model config
    model_config = json.dumps({
        "model": "test-model",
        "provider": "local",
        "context_limit": 128000
    })
    
    # We use a simple objective that will take a few turns
    objective = "Count from 1 to 5 using the run_command tool to echo the number, then call task_complete."
    
    # Spawn the sub-agent using the wrapper
    # Note: We use a mock-like environment or a real one if available. 
    # Since we are testing the WRAPPER and WORKER, we call the wrapper script.
    cmd = [
        "python3", "aeon/scripts/sub_agent_wrapper.py",
        "--agent_id", "telemetry_test_1",
        "--objective", objective,
        "--model_config", model_config,
        "--workspace", os.getcwd(),
        "--output_dir", str(output_dir),
        "--max_iterations", "5"
    ]
    
    print(f"Spawning sub-agent: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    telemetry_file = output_dir / "telemetry.json"
    start_time = time.time()
    found_updates = 0
    last_iteration = -1
    
    try:
        while time.time() - start_time < 60:
            if telemetry_file.exists():
                with open(telemetry_file, "r") as f:
                    data = json.load(f)
                    curr_iter = data.get("iteration", 0)
                    if curr_iter > last_iteration:
                        print(f"Telemetry Update Detected: Iteration {curr_iter}, Intent: {data.get('current_intent')}")
                        last_iteration = curr_iter
                        found_updates += 1
            
            # Check if process finished
            if process.poll() is not None:
                print("Sub-agent process finished.")
                break
                
            time.sleep(1)
            
    finally:
        process.terminate()

    if found_updates > 0:
        print(f"SUCCESS: Detected {found_updates} telemetry updates.")
        return True
    else:
        print("FAILURE: No telemetry updates detected in telemetry.json")
        return False

if __name__ == "__main__":
    if test_telemetry():
        exit(0)
    else:
        exit(1)