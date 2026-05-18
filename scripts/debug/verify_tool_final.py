import sys
import os
import time

# Add the root directory to sys.path to allow importing from aeon.tools
sys.path.append(os.getcwd())

try:
    from aeon.tools.generate_video import GenerateVideoTool
    print("[VERIFY] Successfully imported GenerateVideoTool")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def main():
    tool = GenerateVideoTool()
    
    # Test parameters
    params = {
        "mode": "text_to_video",
        "prompt": "A cinematic shot of a futuristic city with flying cars, neon lights, 4k, highly detailed",
        "output_path": "results/verify_final.mp4",
        "width": 1920,
        "height": 1080,
        "frames": 33
    }
    
    print(f"[VERIFY] Calling tool.execute with params: {params}")
    start_time = time.time()
    result = tool.execute(**params)
    end_time = time.time()
    
    print(f"[VERIFY] Result: {result}")
    print(f"[VERIFY] Time taken: {end_time - start_time:.2f} seconds")
    
    if "successfully" in result.lower():
        print("[VERIFY] SUCCESS: Video generated.")
    else:
        print("[VERIFY] FAILURE: Video generation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()