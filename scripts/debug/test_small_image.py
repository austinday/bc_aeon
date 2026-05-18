import os
import requests
from PIL import Image
import io
import base64
from aeon.tools.vision import AnalyzeImageTool

def test_small_image():
    print("[Test] Creating a small image...")
    img = Image.new('RGB', (100, 100), color='blue')
    img_path = 'scripts/debug/small_test.jpg'
    img.save(img_path)
    
    print(f"[Test] Testing AnalyzeImageTool with {img_path}...")
    tool = AnalyzeImageTool()
    try:
        result = tool.execute(img_path, "What color is this image?")
        print(f"[Test] SUCCESS: Tool returned: {result}")
    except Exception as e:
        print(f"[Test] FAILED: {e}")

if __name__ == "__main__":
    test_small_image()