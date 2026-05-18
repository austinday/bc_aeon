import os
import time
import subprocess
import requests
import json
import threading
from aeon.core.gpu_queue import wait_for_vram, release_vram

# Configuration
VLLM_LB_PORT = 8020
GEMMA_LB_PORT = 8013
VLLM_CONTAINER = "aeon_gemma4_vllm_node0"
MTP_CONTAINER = "aeon_gemma4_mtp_node0"

def check_lb_health(port):
    try:
        resp = requests.get(f"http://localhost:{port}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False

def simulate_vram_hog(gb, duration=15):
    print(f"[VRAM Hog] Attempting to reserve {gb}GB VRAM...")
    try:
        gpu_id = wait_for_vram(gb)
        print(f"[VRAM Hog] Successfully reserved {gb}GB on GPU {gpu_id}. Holding for {duration}s...")
        time.sleep(duration)
        release_vram()
        print(f"[VRAM Hog] Released VRAM on GPU {gpu_id}.")
    except Exception as e:
        print(f"[VRAM Hog] Error: {e}")

def test_llm_responsiveness(port, label):
    print(f"[{label}] Checking LLM responsiveness on port {port}...")
    try:
        # Simple minimal request to see if the LB/Node is alive
        resp = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": "gemma-4",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5
            },
            timeout=10
        )
        if resp.status_code == 200:
            print(f"[{label}] LLM is RESPONSIVE.")
            return True
        else:
            print(f"[{label}] LLM returned status {resp.status_code}.")
            return False
    except Exception as e:
        print(f"[{label}] LLM is UNRESPONSIVE: {e}")
        return False

def test_self_healing(container_name, lb_port):
    print(f"\n[Self-Healing] Testing recovery for {container_name}...")
    
    # 1. Ensure it's running
    subprocess.run(["docker", "start", container_name], capture_output=True)
    time.sleep(5)
    
    # 2. Kill it
    print(f"[Self-Healing] Killing container {container_name}...")
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    
    # 3. Verify it's stopped
    res = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container_name], capture_output=True, text=True)
    if "false" in res.stdout.lower():
        print(f"[Self-Healing] Confirmed: {container_name} is stopped.")
    else:
        print(f"[Self-Healing] Error: {container_name} is still running.")
        return False

    # 4. Trigger LB check by making a request
    print(f"[Self-Healing] Sending request to LB on port {lb_port} to trigger restart...")
    try:
        requests.post(
            f"http://localhost:{lb_port}/v1/chat/completions",
            json={"model": "gemma-4", "messages": [{"role": "user", "content": "hi"}]},
            timeout=15
        )
    except:
        pass

    # 5. Wait for LB to detect and restart
    print("[Self-Healing] Waiting 20s for LB to restart node...")
    time.sleep(20)
    
    # 6. Verify it's running again
    res = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container_name], capture_output=True, text=True)
    if "true" in res.stdout.lower():
        print(f"[Self-Healing] SUCCESS: {container_name} was automatically restarted!")
        return True
    else:
        print(f"[Self-Healing] FAILURE: {container_name} was NOT restarted.")
        return False

def main():
    print("=== AEON STABILITY & SELF-HEALING STRESS TEST ===\n")
    
    # --- Test 1: VRAM Protection ---
    print("--- Test 1: VRAM Protection ---")
    # Start a thread that hogs VRAM
    hog_thread = threading.Thread(target=simulate_vram_hog, args=(80, 15)) # Request almost all VRAM
    hog_thread.start()
    
    # Give the hog a moment to acquire
    time.sleep(2)
    
    # Try to call LLM while VRAM is hogged
    # The LLM nodes should be fine because the hog is just a lock in /tmp/aeon_vram_state.lock
    # and doesn't actually allocate VRAM unless the tool is running.
    # But we want to ensure that if a tool *did* request VRAM, it wouldn't crash the LLM.
    if test_llm_responsiveness(GEMMA_LB_PORT, "MTP-LB"):
        print("[VRAM Protection] LLM remained responsive during VRAM lock.")
    else:
        print("[VRAM Protection] LLM became unresponsive!")
    
    hog_thread.join()
    print("[VRAM Protection] Test complete.\n")

    # --- Test 2: MTP Self-Healing ---
    print("--- Test 2: MTP Self-Healing ---")
    if test_self_healing(MTP_CONTAINER, GEMMA_LB_PORT):
        print("[MTP-LB] Self-healing verified.")
    else:
        print("[MTP-LB] Self-healing FAILED.")

    # --- Test 3: vLLM Self-Healing ---
    print("\n--- Test 3: vLLM Self-Healing ---")
    if test_self_healing(VLLM_CONTAINER, VLLM_LB_PORT):
        print("[vLLM-LB] Self-healing verified.")
    else:
        print("[vLLM-LB] Self-healing FAILED.")

    print("\n=== STRESS TEST COMPLETE ===")

if __name__ == "__main__":
    main()