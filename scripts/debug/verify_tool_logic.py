import sys
import os

# Add the root directory to sys.path to allow importing from aeon.tools
sys.path.append(os.getcwd())

try:
    from aeon.tools.generate_video import GenerateVideoTool
    print("[TEST] Successfully imported GenerateVideoTool")
except ImportError as e:
    print(f"[FAILURE] Import failed: {e}")
    sys.exit(1)

def test_generation():
    print("[TEST] Initializing GenerateVideoTool...")
    tool = GenerateVideoTool()
    
    prompt = "A cinematic shot of a futuristic city with flying cars, high detail, 4k"
    output_path = "results/test_video.mp4"
    
    print(f"[TEST] Calling tool.execute() with prompt: {prompt}")
    # We use the actual execute method to test the logic implemented in the tool
    result = tool.execute(
        mode="text_to_video",
        prompt=prompt,
        output_path=output_path
    )
    
    print(f"[RESULT] Tool returned: {result}")
    
    if "successfully" in result.lower() or "saved to" in result.lower():
        print("[SUCCESS] Video generation appears to have worked!")
    else:
        print(f"[FAILURE] Tool reported an error: {result}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_generation()
    except Exception as e:
        print(f"[CRITICAL] Script crashed: {e}")
        sys.exit(1)