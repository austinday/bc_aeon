import sys
import os
# Add current directory to path to import aeon tools
sys.path.append(os.getcwd())

try:
    from aeon.tools.generate_video import GenerateVideoTool
    print("[TEST] Successfully imported GenerateVideoTool")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def test_generation():
    tool = GenerateVideoTool()
    print("[TEST] Calling execute with test parameters...")
    # Using a simple prompt and default settings to verify connectivity and node validity
    result = tool.execute(
        mode="text_to_video",
        prompt="A cinematic shot of a futuristic city with flying cars, high detail, 4k",
        output_path="aeon_output/test_video.mp4",
        width=768,
        height=512,
        frames=33
    )
    print(f"[RESULT] Tool output: {result}")

if __name__ == "__main__":
    test_generation()