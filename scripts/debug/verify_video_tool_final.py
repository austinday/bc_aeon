import os
import sys
from aeon.tools.generate_video import GenerateVideoTool

def test_video_generation():
    print("[TEST] Initializing GenerateVideoTool...")
    try:
        tool = GenerateVideoTool()
    except Exception as e:
        print(f"[TEST] Failed to initialize tool: {e}")
        return

    prompt = "A cinematic shot of a futuristic city with flying cars, neon lights, raining, 4k, highly detailed"
    output_path = "results/test_video_final.mp4"
    
    print(f"[TEST] Requesting video generation with prompt: {prompt}")
    # We use the execute method of the tool, which uses the updated _get_workflow logic
    result = tool.execute(
        mode="text_to_video",
        prompt=prompt,
        output_path=output_path
    )
    
    print(f"[TEST] Result: {result}")
    
    if "successfully" in result.lower():
        print("[TEST] SUCCESS: Tool reported success.")
    else:
        print(f"[TEST] FAILURE: Tool returned error: {result}")
        sys.exit(1)

if __name__ == "__main__":
    test_video_generation()