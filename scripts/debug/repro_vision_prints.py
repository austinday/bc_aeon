import os
import sys

# Ensure the workspace is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from aeon.tools.vision import AnalyzeImageTool
    print("Successfully imported AnalyzeImageTool")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def main():
    tool = AnalyzeImageTool()
    
    # Use a small image for testing. 
    # I'll check if any image exists in the workspace first, or just create a dummy one.
    test_image = "scripts/debug/small_verify.jpg"
    if not os.path.exists(test_image):
        from PIL import Image
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(test_image)
        print(f"Created dummy image at {test_image}")

    print(f"Executing tool with image: {test_image}")
    # We just want to see the prints, so we don't care about the actual result
    result = tool.execute(image_path=test_image, prompt="What is in this image?")
    print(f"Result: {result}")

if __name__ == "__main__":
    main()