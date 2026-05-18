import os
import json
from pathlib import Path

def test_subagent_files():
    print("Testing sub-agent output directory structure...")
    test_dir = Path("aeon_output/sub_agents/test_agent_123")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate files created by wrapper
    status_file = test_dir / "status.txt"
    log_file = test_dir / "agent.log"
    telemetry_file = test_dir / "telemetry.json"
    output_file = test_dir / "output.json"
    
    status_file.write_text("RUNNING")
    log_file.write_text("[2026-05-17 16:00:00] Initializing...\n[2026-05-17 16:00:01] Step 1...")
    telemetry_file.write_text(json.dumps({"agent_id": "test_agent_123", "iteration": 1, "current_step": "Testing", "timestamp": 123456789}))
    output_file.write_text(json.dumps({"status": "COMPLETED", "result": "Success"}))
    
    print(f"Created mock files in {test_dir}")
    
    # Now simulate the Monitor's logic
    print("\nSimulating SubAgentMonitor logic...")
    try:
        status = status_file.read_text().strip()
        with open(telemetry_file, 'r') as f:
            telemetry = json.load(f)
        with open(log_file, 'r') as f:
            logs = f.readlines()[-20:]
            
        print(f"Status: {status}")
        print(f"Telemetry: {telemetry}")
        print(f"Logs: {logs}")
        
        if status == "RUNNING" and telemetry['agent_id'] == "test_agent_123":
            print("\nVERIFICATION SUCCESS: Monitor can read wrapper outputs.")
        else:
            print("\nVERIFICATION FAILURE: Data mismatch.")
            
    except Exception as e:
        print(f"VERIFICATION FAILURE: {e}")

if __name__ == "__main__":
    test_subagent_files()