import os
import sys
from PIL import Image

# Ensure the current directory is in the path so we can import aeon
sys.path.append(os.getcwd())

try:
    from aeon.tools.vision import AnalyzeImageTool
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_vision_tool():
    tool = AnalyzeImageTool()
    # Use the large image created in previous iterations or create one
    image_path = 'scripts/debug/large_test_image.jpg'
    if not os.path.exists(image_path):
        print(f"Large image not found at {image_path}, creating one...")
        img = Image.new('RGB', (5000, 5000), color='red')
        img.save(image_path)
    
    print(f"Testing AnalyzeImageTool with image: {image_path}")
    print("This will trigger the ephemeral server startup if not already running...")
    
    try:
        # The tool.execute method handles server startup and image preparation
        result = tool.execute(image_path, "What is in this image?")
        print("\n" + "="*20)
        print("SUCCESS!")
        print(f"Result: {result}")
        print("="*20)
    except Exception as e:
        print("\n" + "="*20)
        print(f"FAILED: {e}")
        print("="*20)
        raise e

if __name__ == "__main__":
    test_vision_tool()