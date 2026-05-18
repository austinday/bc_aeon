import os
import subprocess
import time
import requests
import json

def run_command(cmd, shell=False):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    return result

def main():
    # 1. Start ComfyUI
    print("Starting ComfyUI server...")
    run_command("bash aeon/scripts/start_comfyui.sh", shell=True)
    
    # 2. Wait for server to be ready
    timeout = 120
    start_time = time.time()
    while True:
        try:
            response = requests.get("http://localhost:8188", timeout=2)
            if response.status_code == 200:
                print("ComfyUI server is UP!")
                break
        except:
            pass
        if time.time() - start_time > timeout:
            print("Timeout waiting for ComfyUI to start.")
            return
        print("Waiting for server...")
        time.sleep(5)

    # 3. Minimal Workflow for Testing
    # This is a stripped-down version of the LTX workflow to isolate the crash
    workflow = {
        "3": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "ltx-2.3-22b-dev-Q4_1.gguf"}},
        "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5xxl_fp8_e4m3fn.safetensors", "type": "ltxv"}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.3-22b-dev_video_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "a simple test", "clip": ["7", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality", "clip": ["7", 0]}},
        "6": {"class_type": "ModelSamplingLTXV", "inputs": {"model": ["3", 0], "max_shift": 2.05, "base_shift": 0.95}},
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42, "steps": 10, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                "denoise": 1.0, "model": ["6", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["11", 0]
            }
        },
        "11": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 256, "height": 256, "length": 9, "batch_size": 1}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["8", 0]}},
        "13": {"class_type": "VHS_VideoCombine", "inputs": {"frame_rate": 24, "loop_count": 0, "filename_prefix": "diag", "format": "video/h264-mp4", "save_output": True, "pingpong": False, "images": ["12", 0]}}
    }

    print("\nSending minimal generation request...")
    try:
        response = requests.post("http://localhost:8188/prompt", json={"prompt": workflow}, timeout=300)
        response.raise_for_status()
        print("Request accepted. Monitoring for crash...")
        
        # Poll for completion or crash
        prompt_id = response.json().get("prompt_id")
        while True:
            try:
                history = requests.get(f"http://localhost:8188/history/{prompt_id}", timeout=5).json()
                if prompt_id in history:
                    print("Success! Minimal video generated without crashing.")
                    break
            except requests.exceptions.ConnectionError:
                print("\n!!! CONNECTION RESET DETECTED !!!")
                print("The server has crashed. Capturing logs now...")
                break
            time.sleep(2)

    except Exception as e:
        print(f"\nRequest failed: {e}")

    # 4. Capture Logs
    print("\n--- DOCKER LOGS ---")
    logs = run_command("docker logs aeon_comfyui", shell=True)
    print(logs.stdout)
    if logs.stderr:
        print("--- STDERR ---")
        print(logs.stderr)

    # 5. Check Container Status
    print("\n--- CONTAINER STATUS ---")
    status = run_command("docker inspect aeon_comfyui --format='{{.State.Status}} {{.State.ExitCode}}'", shell=True)
    print(f"Status: {status.stdout.strip()}")

if __name__ == "__main__":
    main()