import sys
import os
# Ensure the current directory is in path for aeon imports
sys.path.append(os.getcwd())

try:
    from aeon.tools.generate_video import GenerateVideoTool
    print("[TEST] Successfully imported GenerateVideoTool")
except Exception as e:
    print(f"[TEST] Import failed: {e}")
    sys.exit(1)

def test_generation():
    print("[TEST] Instantiating GenerateVideoTool...")
    tool = GenerateVideoTool()
    
    print("[TEST] Calling execute method...")
    # Using a simple prompt for testing
    result = tool.execute(
        mode='text_to_video',
        prompt='A cinematic shot of a futuristic city with flying cars, high detail, 4k',
        output_path='aeon_output/test_video.mp4',
        width=768,
        height=512,
        frames=33
    )
    print(f"[TEST] Result: {result}")

if __name__ == "__main__":
    test_generation()