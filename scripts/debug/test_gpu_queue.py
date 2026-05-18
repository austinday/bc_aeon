import multiprocessing
import time
import os
from aeon.core.gpu_queue import wait_for_vram, STATE_FILE
import json

def request_vram(name, amount, duration):
    try:
        print(f"[{name}] Requesting {amount}GB VRAM...")
        gpu_id = wait_for_vram(amount)
        print(f"[{name}] ALLOCATED: GPU {gpu_id} for {amount}GB")
        
        # Simulate work
        time.sleep(duration)
        
        print(f"[{name}] RELEASED: GPU {gpu_id}")
    except Exception as e:
        print(f"[{name}] ERROR: {e}")

if __name__ == "__main__":
    # Clean up state file before starting
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    # We will simulate 3 processes requesting VRAM.
    # Assuming a system with 2 GPUs (as per user prompt).
    # If we request more than available, they should queue.
    
    # Process 1: Takes a large chunk of GPU 0
    # Process 2: Takes a large chunk of GPU 1
    # Process 3: Should have to wait until one of the above finishes
    
    # Note: We use a very high amount to force queuing regardless of actual hardware,
    # but wait_for_vram checks nvidia-smi. 
    # To make this test portable, we'll use a value that is likely to fit 
    # but we'll launch more requests than GPUs available.
    
    # We'll request 10GB. Most modern GPUs have at least 10GB.
    # If the system has 2 GPUs, the 3rd process MUST wait.
    
    processes = [
        multiprocessing.Process(target=request_vram, args=("Proc_1", 10, 5)),
        multiprocessing.Process(target=request_vram, args=("Proc_2", 10, 5)),
        multiprocessing.Process(target=request_vram, args=("Proc_3", 10, 2)),
    ]

    start_time = time.time()
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f}s")
    print("If the logic works and there are 2 GPUs, total time should be ~10s (5s for first batch, 5s for the queued one).")