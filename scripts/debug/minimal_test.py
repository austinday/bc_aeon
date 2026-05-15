import os
import requests
import json
from aeon.tools.generate_video import GenerateVideoTool

def test():
    tool = GenerateVideoTool()
    print("Testing minimal video generation...")
    # Use a very simple prompt and small frame count to minimize VRAM and time
    result = tool.execute(
        mode="text_to_video",
        prompt="A simple red ball bouncing",
        output_path="aeon_output/debug/minimal_test.mp4",
        width=512,
        height=512,
        frames=16
    )
    print(f"Result: {result}")

if __name__ == "__main__":
    test()