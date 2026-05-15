import os
import sys
from aeon.tools.generate_video import GenerateVideoTool

def test():
    print("Initializing GenerateVideoTool...")
    tool = GenerateVideoTool()
    
    # Test a simple text-to-video generation to trigger the 400 error
    print("\nAttempting to generate a short video to capture 400 error...")
    result = tool.execute(
        mode="text_to_video",
        prompt="A cinematic shot of a futuristic city",
        output_path="aeon_output/debug/debug_test.mp4",
        frames=33
    )
    print(f"\nTool Result: {result}")

if __name__ == "__main__":
    test()