import os
import time
import json
import fcntl
import subprocess

STATE_FILE = "/tmp/aeon_vram_state.lock"

def get_real_vram():
    """Queries nvidia-smi for current free VRAM in GB per GPU."""
    res = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    vram = {}
    for line in res.stdout.strip().split('\n'):
        if line:
            idx, free_mb = map(int, line.split(', '))
            vram[idx] = free_mb / 1024.0
    return vram

def wait_for_vram(required_gb: float, timeout: int = 600) -> int:
    """
    Blocks until a GPU has `required_gb` of free VRAM available.
    Uses a file lock and a JSON state file to track 'pending' allocations
    for containers that are booting up but haven't allocated memory yet.
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        with open(STATE_FILE, 'a+') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                lock_fd.seek(0)
                content = lock_fd.read()
                try:
                    state = json.loads(content) if content else {"pending": []}
                except (json.JSONDecodeError, ValueError):
                    state = {"pending": []}
                
                current_time = time.time()
                # Keep pending allocations that are less than 90 seconds old (gives models time to boot)
                state["pending"] = [p for p in state.get("pending", []) if current_time - p["timestamp"] < 90]
                
                real_vram = get_real_vram()
                effective_vram = dict(real_vram)
                
                # Subtract pending allocations to get the true safe available memory
                for p in state["pending"]:
                    if p["gpu_id"] in effective_vram:
                        effective_vram[p["gpu_id"]] -= p["requested_gb"]
                
                selected_gpu = None
                # Sort GPUs by available effective memory (descending) so we pack efficiently
                sorted_gpus = sorted(effective_vram.items(), key=lambda x: x[1], reverse=True)
                for gpu_id, free_gb in sorted_gpus:
                    if free_gb >= required_gb:
                        selected_gpu = gpu_id
                        break
                
                if selected_gpu is not None:
                    state["pending"].append({
                        "gpu_id": selected_gpu,
                        "requested_gb": required_gb,
                        "timestamp": current_time
                    })
                    lock_fd.seek(0)
                    lock_fd.truncate()
                    json.dump(state, lock_fd)
                    return selected_gpu
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        
        # If no GPU is available, sleep and try again
        time.sleep(5)
        
    raise TimeoutError(f"Timed out waiting for {required_gb}GB VRAM.")
