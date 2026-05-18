import os
from aeon.tools.vision import AnalyzeImageTool

def main():
    # Use a small image for fast testing
    test_image = "scripts/debug/small_verify.jpg"
    if not os.path.exists(test_image):
        # Create a dummy image if it doesn't exist
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        os.makedirs(os.path.dirname(test_image), exist_ok=True)
        img.save(test_image)

    print("--- START OF VISION TOOL EXECUTION ---")
    tool = AnalyzeImageTool()
    # We just want to see what is printed to stdout
    result = tool.execute(image_path=test_image, prompt="What color is this image?")
    print("--- END OF VISION TOOL EXECUTION ---")
    print(f"Result: {result}")

if __name__ == "__main__":
    main()