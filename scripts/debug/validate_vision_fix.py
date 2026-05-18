import os
import sys
from aeon.tools.vision import AnalyzeImageTool

def main():
    print("[Validation] Testing AnalyzeImageTool with a large image...")
    
    # Path to the large test image
    image_path = "scripts/debug/large_test_image.jpg"
    if not os.path.exists(image_path):
        print(f"[Error] Test image not found at {image_path}")
        sys.exit(1)
    
    print(f"[Validation] Image path: {image_path}")
    print(f"[Validation] Image size: {os.path.getsize(image_path) / (1024*1024):.2f} MB")
    
    # Initialize the tool
    tool = AnalyzeImageTool()
    prompt = "Describe this image in detail."
    
    try:
        print("[Validation] Executing tool.execute()...")
        result = tool.execute(image_path, prompt)
        print("\n[Validation] SUCCESS! Tool returned response:")
        print("-" * 40)
        print(result)
        print("-" * 40)
    except Exception as e:
        print(f"\n[Validation] FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()