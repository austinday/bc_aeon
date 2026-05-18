import os
import sys
from aeon.tools.vision import AnalyzeImageTool

def test_vision():
    print("--- Vision Tool Validation ---")
    tool = AnalyzeImageTool()
    
    # Test images
    test_cases = [
        ("Small Image", "scripts/debug/small_verify.jpg"),
        ("Large Image", "scripts/debug/large_test_image.jpg"),
    ]
    
    for name, path in test_cases:
        print(f"\nTesting {name}: {path}")
        if not os.path.exists(path):
            print(f"Error: File {path} not found. Skipping.")
            continue
            
        try:
            # Use a simple prompt to verify basic functionality
            result = tool.execute(image_path=path, prompt="Describe this image in one sentence.")
            print(f"Result: {result}")
            print(f"Status: SUCCESS")
        except Exception as e:
            print(f"Status: FAILED")
            print(f"Error: {e}")

if __name__ == "__main__":
    test_vision()