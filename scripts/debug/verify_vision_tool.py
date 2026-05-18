import os
import sys
from aeon.tools.vision import AnalyzeImageTool

def test_vision_tool(image_path, prompt):
    print(f"Testing image: {image_path}")
    print(f"Prompt: {prompt}")
    
    # Initialize the tool
    tool = AnalyzeImageTool()
    
    try:
        result = tool.execute(image_path, prompt)
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Paths to test images
    small_img = "scripts/debug/small_verify.jpg"
    large_img = "scripts/debug/large_test_image.jpg"
    
    # Ensure images exist
    for img in [small_img, large_img]:
        if not os.path.exists(img):
            print(f"Missing test image: {img}")
            # Create a dummy if missing for the sake of the test
            from PIL import Image
            import numpy as np
            if "small" in img:
                Image.new('RGB', (100, 100), color='red').save(img)
            else:
                # Create a truly large image to test the 413 fix
                # 4000x4000 random noise
                data = np.random.randint(0, 256, (4000, 4000, 3), dtype=np.uint8)
                Image.fromarray(data).save(img)
            print(f"Created dummy image: {img}")

    print("--- Testing Small Image ---")
    success_small = test_vision_tool(small_img, "What is in this image?")
    
    print("\n--- Testing Large Image ---")
    success_large = test_vision_tool(large_img, "What is in this image?")
    
    if success_small and success_large:
        print("\nSUCCESS: Both small and large images were processed successfully.")
        sys.exit(0)
    else:
        print("\nFAILURE: One or more tests failed.")
        sys.exit(1)