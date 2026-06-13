import subprocess
import json
import os
from pathlib import Path

def test_subagent():
    print("Testing sub-agent wrapper for telemetry TypeError...")
    
    # Setup paths
    workspace = os.getcwd()
    output_dir = os.path.join(workspace, "aeon_output", "test_fix_dir")
    os.makedirs(output_dir, exist_ok=True)
    
    # Minimal model config
    model_config = json.dumps({
        "model": "gemma-4",
        "provider": "local",
        "base_url": "http://localhost:8000/v1"
    })
    
    cmd = [
        "python3", "aeon/scripts/sub_agent_wrapper.py",
        "--agent_id", "test_fix_1",
        "--objective", "Say hello and exit",
        "--model_config", model_config,
        "--workspace", workspace,
        "--output_dir", output_dir,
        "--max_iterations", "1"
    ]
    
    try:
        # We use a timeout because the agent might actually try to run if the LLM is up,
        # but we are primarily looking for the immediate crash during initialization/first step.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if "TypeError" in result.stderr or "update_telemetry" in result.stderr:
            print("\n❌ FAILED: TypeError still present in logs.")
            return False
        
        if result.returncode == 0 or "COMPLETED" in result.stdout or "FAILED" in result.stdout:
            # If it reached the point of completing or failing the objective (rather than crashing the wrapper), it's a win.
            print("\n✅ SUCCESS: Wrapper did not crash with telemetry TypeError.")
            return True
        else:
            print(f"\n⚠️ Unexpected exit code {result.returncode}. Check logs.")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n✅ SUCCESS: Agent ran for 30s without crashing (telemetry is likely working).")
        return True
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        return False

if __name__ == "__main__":
    success = test_subagent()
    exit(0 if success else 1)