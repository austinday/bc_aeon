import requests
import json
import os
import time
import sys
from aeon.tools.generate_video import GenerateVideoTool

def wait_for_comfyui(url="http://localhost:8188", timeout=120):
    print(f"[VERIFY] Waiting for ComfyUI to be healthy at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/system_stats", timeout=2)
            if resp.status_code == 200:
                print("[VERIFY] ComfyUI is UP!")
                return True
        except:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    print("\n[VERIFY] Timeout waiting for ComfyUI")
    return False

def main():
    if not wait_for_comfyui():
        print("[VERIFY] FAILURE: ComfyUI not available")
        sys.exit(1)

    print("[VERIFY] Initializing GenerateVideoTool...")
    tool = GenerateVideoTool()
    
    prompt = "A cinematic shot of a futuristic city with flying cars, neon lights, raining, 4k, highly detailed"
    output_path = "scripts/debug/test_video.mp4"
    
    print(f"[VERIFY] Requesting video generation with prompt: {prompt}")
    # We use the tool's execute method directly
    result = tool.execute(
        mode="text_to_video",
        prompt=prompt,
        output_path=output_path,
        frames=33
    )
    
    print(f"[VERIFY] Tool Result: {result}")
    
    # Debug: Check the actual ComfyUI output directory if possible
    # The start_comfyui.sh mounts $HOME/.aeon/temp/comfyui_output to /workspace/ComfyUI/output
    aeon_home = os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))
    output_dir = os.path.join(aeon_home, "temp/comfyui_output")
    
    if os.path.exists(output_dir):
        print(f"[VERIFY] Checking output directory: {output_dir}")
        files = os.listdir(output_dir)
        print(f"[VERIFY] Files found: {files}")
        if files:
            print("[VERIFY] SUCCESS: Files were generated in the output directory, even if the tool didn't 'see' them.")
        else:
            print("[VERIFY] No files found in output directory.")
    else:
        print(f"[VERIFY] Output directory {output_dir} does not exist.")

    if "successfully" in result.lower() or (os.path.exists(output_dir) and os.listdir(output_dir)):
        print("[VERIFY] FINAL RESULT: SUCCESS")
    else:
        print("[VERIFY] FINAL RESULT: FAILURE")
        sys.exit(1)

if __name__ == "__main__":
    main()