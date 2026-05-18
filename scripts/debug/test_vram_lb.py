import multiprocessing
import time
import os
import random
from aeon.core.gpu_queue import wait_for_vram, release_vram

def request_vram(agent_id, required_gb, target_gpu=None):
    """Simulates an agent requesting VRAM and holding it for a short period."""
    try:
        print(f"[Agent {agent_id}] Requesting {required_gb}GB VRAM (Target: {target_gpu})...")
        start_time = time.time()
        
        # This is the function we are testing
        gpu_id = wait_for_vram(required_gb, timeout=30, gpu_id=target_gpu)
        
        end_time = time.time()
        wait_duration = end_time - start_time
        
        print(f"[Agent {agent_id}] ACQUIRED GPU {gpu_id} after {wait_duration:.2f}s")
        
        # Simulate work (holding the VRAM)
        # In the real system, the 'pending' entry expires after 90s or is cleared.
        # Here we just sleep to simulate the model being loaded and used.
        time.sleep(random.uniform(2, 5))
        
        release_vram()
        print(f"[Agent {agent_id}] Released VRAM and finished work on GPU {gpu_id}")
        return True, gpu_id, wait_duration
    except Exception as e:
        print(f"[Agent {agent_id}] FAILED: {e}")
        return False, None, 0

if __name__ == "__main__":
    # Setup: Clear the state file to start fresh
    state_file = "/tmp/aeon_vram_state.lock"
    if os.path.exists(state_file):
        try:
            os.remove(state_file)
        except:
            pass

    # Test Scenario:
    # We have 2 GPUs. Let's assume each has ~80GB.
    # We will spawn 8 agents, each requesting 20GB.
    # Expected: 
    # - First 4 agents should get VRAM almost immediately (2 per GPU).
    # - Next 4 agents should block and only get VRAM as the first 4 finish (or as pending entries expire).
    # Note: Since we aren't actually loading models, the 'real' VRAM doesn't change.
    # The load balancer relies on the 'pending' list in the state file.
    
    num_agents = 8
    vram_per_agent = 20.0
    
    print(f"Starting VRAM LB Stress Test: {num_agents} agents requesting {vram_per_agent}GB each.")
    print("Expected behavior: Max 4 agents active at once (assuming 2x 80GB GPUs).")
    
    processes = []
    results_queue = multiprocessing.Queue()

    def wrapper(aid, req, target, q):
        res = request_vram(aid, req, target)
        q.put(res)

    for i in range(num_agents):
        # Mix of targeted and untargeted requests
        target = 0 if i % 3 == 0 else None 
        p = multiprocessing.Process(target=wrapper, args=(i, vram_per_agent, target, results_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Analyze results
    all_results = []
    while not results_queue.empty():
        all_results.append(results_queue.get())

    successes = [r for r in all_results if r[0]]
    print(f"\nTest Complete. Successes: {len(successes)}/{num_agents}")
    
    if len(successes) < num_agents:
        print("Warning: Some agents timed out or failed.")