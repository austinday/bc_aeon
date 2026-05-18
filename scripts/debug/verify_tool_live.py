import sys
import os
import time
from aeon.tools.generate_video import GenerateVideoTool

def verify():
    print("[VERIFY] Starting Live Tool Verification...")
    tool = GenerateVideoTool()
    
    params = {
        'mode': 'text_to_video',
        'prompt': 'A cinematic shot of a futuristic city with flying cars, neon lights, 4k, highly detailed',
        'output_path': 'results/verify_live.mp4',
        'width': 1920,
        'height': 1080,
        'frames': 33
    }
    
    print(f"[VERIFY] Calling tool.execute with params: {params}")
    start_time = time.time()
    try:
        result = tool.execute(**params)
        duration = time.time() - start_time
        print(f"[VERIFY] Result: {result}")
        print(f"[VERIFY] Time taken: {duration:.2f} seconds")
        
        if "successfully" in result.lower() or os.path.exists(params['output_path']):
            print("[VERIFY] SUCCESS: Video generated.")
            sys.exit(0)
        else:
            print(f"[VERIFY] FAILURE: Tool reported failure or file not found. Result: {result}")
            sys.exit(1)
    except Exception as e:
        print(f"[VERIFY] CRITICAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify()