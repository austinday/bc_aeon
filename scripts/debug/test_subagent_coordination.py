import os
import json
import time
import subprocess
from pathlib import Path

def test_subagent_flow():
    print("--- Starting Sub-Agent Coordination Test ---")
    
    # 1. Setup paths
    output_dir = Path("aeon_output/test_run/sub_agents/test_agent_1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock model config
    model_config = json.dumps({
        "model": "Qwen3.6-35B-A3B-Uncensored",
        "provider": "llamacpp",
        "base_url": "http://localhost:8009/v1",
        "context_limit": 128000
    })
    
    # 2. Spawn sub-agent via the wrapper
    # We use a simple objective that takes a few iterations
    cmd = [
        "python3", "aeon/scripts/sub_agent_wrapper.py",
        "--agent_id", "test_agent_1",
        "--objective", "Count from 1 to 5, then summarize the result.",
        "--model_config", model_config,
        "--workspace", os.getcwd(),
        "--output_dir", str(output_dir),
        "--max_iterations", "10"
    ]
    
    print(f"Spawning sub-agent: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    
    # 3. Monitor the agent in real-time
    try:
        while process.poll() is None:
            status_file = output_dir / "status.txt"
            telemetry_file = output_dir / "telemetry.json"
            
            status = "UNKNOWN"
            if status_file.exists():
                status = status_file.read_text().strip()
            
            telemetry = "N/A"
            if telemetry_file.exists():
                with open(telemetry_file, "r") as f:
                    telemetry = f.read()
            
            print(f"[MONITOR] Status: {status} | Telemetry: {telemetry[:100]}...")
            time.sleep(2)
            
    except KeyboardInterrupt:
        process.terminate()
    
    print("\n--- Sub-Agent Finished ---")
    
    # 4. Verify final output
    output_file = output_dir / "output.json"
    if output_file.exists():
        print(f"Final Result: {output_file.read_text()}")
    else:
        print("Error: output.json not found!")

if __name__ == "__main__":
    test_subagent_flow()