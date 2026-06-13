import subprocess
import json
import time
import os
from pathlib import Path

# This script simulates the main agent spawning sub-agents to see if they actually 
# hit the load balancer in parallel and utilize both GPUs.

def spawn_test_agent(agent_id, model_config):
    # Mocking the command that spawn_sub_agent would likely run
    # We use a simple objective that requires a few LLM calls
    cmd = [
        "python3", "-m", "aeon.scripts.sub_agent_wrapper",
        "--agent_id", agent_id,
        "--objective", "Count to 10 and then stop.",
        "--model_config", json.dumps(model_config),
        "--workspace", "/home/aday/bc_aeon",
        "--output_dir", f"/home/aday/bc_aeon/aeon_output/sub_agents/{agent_id}",
        "--debug"
    ]
    
    print(f"Spawning agent {agent_id}...")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

if __name__ == "__main__":
    # Use the vLLM LB config from main.py
    model_config = {
        'model': 'Gemma-4-31B-NVFP4',
        'provider': 'vllm',
        'base_url': 'http://localhost:8018/v1',
        'context_limit': 131072,
        'container_name': 'aeon_gemma_vllm_lb',
    }

    agents = []
    for i in range(4):
        agents.append(spawn_test_agent(f"test_parallel_{i}", model_config))

    print("Agents spawned. Monitoring for 30 seconds...")
    start_time = time.time()
    while time.time() - start_time < 30:
        # Check if any agents finished
        still_running = []
        for a in agents:
            poll = a.poll()
            if poll is None:
                still_running.append(a)
        
        agents = still_running
        if not agents:
            break
        time.sleep(2)

    print("Test complete. Check vLLM LB logs to see if requests were balanced.")