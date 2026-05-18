import os
import time
import json
import subprocess
import sys
from pathlib import Path

# Ensure we can import from the aeon package
sys.path.append(os.getcwd())
try:
    from aeon.main import register_models_for_agent, unregister_models_for_agent, MODEL_REGISTRY_PATH
except ImportError:
    print("Error: Could not import from aeon.main. Ensure you are running from the project root.")
    sys.exit(1)

def test_heartbeat():
    model = "test-heartbeat-model"
    print(f"Testing heartbeat for model: {model}")
    
    # 1. Start a "stuck" process
    # We'll use a separate python script that registers and then sleeps without a heartbeat thread
    stuck_script = "scripts/debug/stuck_agent.py"
    with open(stuck_script, "w") as f:
        f.write(f"""
import os, time, sys
sys.path.append(os.getcwd())
from aeon.main import register_models_for_agent
# Register but do NOT start a heartbeat thread
register_models_for_agent(['{model}'])
print(f"Stuck PID {{os.getpid()}} registered. Sleeping forever...")
time.sleep(1000)
""")
    
    try:
        proc = subprocess.Popen([sys.executable, stuck_script])
        print(f"Started stuck process PID: {proc.pid}")
        
        # Give it a moment to register
        time.sleep(2)
        
        # Verify it's in the registry
        with open(MODEL_REGISTRY_PATH, 'r') as f:
            reg = json.load(f)
        
        if model not in reg or str(proc.pid) not in reg[model]:
            print("Error: Stuck process not found in registry.")
            proc.terminate()
            return
        print("Stuck process verified in registry.")
        
        # 2. Wait for heartbeat to expire (60s)
        print("Waiting 65 seconds for heartbeat to expire...")
        time.sleep(65)
        
        # 3. Trigger cleanup by registering another agent
        print("Triggering cleanup via new registration...")
        register_models_for_agent([model])
        
        # 4. Check if stuck PID is gone
        with open(MODEL_REGISTRY_PATH, 'r') as f:
            reg = json.load(f)
        
        if model in reg and str(proc.pid) in reg[model]:
            print(f"FAILURE: Stuck PID {proc.pid} is still in registry!")
        else:
            print(f"SUCCESS: Stuck PID {proc.pid} was cleaned up.")
            
    finally:
        # Cleanup
        if 'proc' in locals():
            proc.terminate()
        unregister_models_for_agent([model])
        if os.path.exists(stuck_script):
            os.remove(stuck_script)

if __name__ == "__main__":
    test_heartbeat()