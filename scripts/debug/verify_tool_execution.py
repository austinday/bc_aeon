import sys
import os

# Add the project root to sys.path so we can import aeon
sys.path.append(os.getcwd())

try:
    from aeon.tools.generate_video import GenerateVideoTool
    print("[TEST] Successfully imported GenerateVideoTool")
except ImportError as e:
    print(f"[FAILURE] Import failed: {e}")
    sys.exit(1)

def test_generation():
    print("[TEST] Instantiating GenerateVideoTool...")
    tool = GenerateVideoTool()
    
    prompt = "A cinematic shot of a futuristic city with flying cars, neon lights, raining, 4k, highly detailed"
    output_path = "scripts/debug/test_video.mp4"
    
    print(f"[TEST] Calling tool.execute with prompt: {prompt}")
    # We call the actual execute method of the tool to test the logic implemented in the class
    result = tool.execute(
        mode="text_to_video",
        prompt=prompt,
        output_path=output_path
    )
    
    print(f"[RESULT] Tool returned: {result}")
    
    if "Video generated successfully" in result:
        print("[SUCCESS] Video generation verified!")
    else:
        print(f"[FAILURE] Tool reported an error: {result}")
        sys.exit(1)

if __name__ == "__main__":
    test_generation()