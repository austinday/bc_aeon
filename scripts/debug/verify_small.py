import os
import requests
from PIL import Image
import numpy as np
from aeon.tools.vision import AnalyzeImageTool

def create_small_image(path):
    print(f"[Test] Creating small image at {path}...")
    # Create a simple 100x100 image
    data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(data)
    img.save(path)
    print(f"[Test] Image created.")

def main():
    img_path = "scripts/debug/small_verify.jpg"
    create_small_image(img_path)
    
    print("[Test] Calling AnalyzeImageTool...")
    tool = AnalyzeImageTool()
    try:
        # Use a very simple prompt
        result = tool.execute(img_path, "What is in this image?")
        print(f"[Test] Success! Result: {result}")
    except Exception as e:
        print(f"[Test] Failed: {e}")

if __name__ == "__main__":
    main()