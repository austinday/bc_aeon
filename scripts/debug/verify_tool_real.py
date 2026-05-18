import sys
import os

# Add the project root to sys.path to allow importing from aeon.tools
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

try:
    from aeon.tools.generate_video import GenerateVideoTool
    print("[TEST] Successfully imported GenerateVideoTool")
except ImportError as e:
    print(f"[FAILURE] Failed to import GenerateVideoTool: {e}")
    sys.exit(1)

def test_video_generation():
    print("[TEST] Initializing GenerateVideoTool...")
    tool = GenerateVideoTool()
    
    prompt = "A cinematic shot of a futuristic city with flying cars, high detail, 4k"
    output_path = "aeon_output/test_video.mp4"
    
    print(f"[TEST] Calling tool.execute() with prompt: {prompt}")
    # We use the actual execute method of the tool to verify the logic implemented in the class
    result = tool.execute(
        mode="text_to_video",
        prompt=prompt,
        output_path=output_path
    )
    
    print(f"[RESULT] Tool returned: {result}")
    
    if "successfully" in result.lower() or "saved to" in result.lower():
        print("[SUCCESS] Video generation reported as successful.")
    else:
        print("[FAILURE] Tool reported an error or failed to generate video.")
        sys.exit(1)

if __name__ == "__main__":
    test_video_generation()