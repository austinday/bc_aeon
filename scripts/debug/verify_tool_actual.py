import sys
import os

# Add the project root to sys.path to allow importing aeon
sys.path.append(os.getcwd())

try:
    from aeon.tools.generate_video import GenerateVideoTool
    print("[TEST] Successfully imported GenerateVideoTool")
except Exception as e:
    print(f"[FAILURE] Import failed: {e}")
    sys.exit(1)

def test_generation():
    print("[TEST] Initializing GenerateVideoTool...")
    tool = GenerateVideoTool()
    
    prompt = "A cinematic shot of a futuristic city with flying cars, high detail, 4k"
    output_path = "results/verify_video.mp4"
    
    print(f"[TEST] Calling tool.execute() with prompt: {prompt}")
    # We call the actual execute method of the tool to test the logic implemented in the class
    result = tool.execute(
        mode="text_to_video",
        prompt=prompt,
        output_path=output_path,
        width=768,
        height=512,
        frames=33
    )
    
    print(f"[RESULT] Tool returned: {result}")
    
    if "Video generated successfully" in result:
        print("[SUCCESS] Video generation verified!")
    else:
        print("[FAILURE] Tool reported an error or failed to generate video.")
        sys.exit(1)

if __name__ == "__main__":
    test_generation()