import os
import sys

# Use insert(0, ...) to ensure we use the local source code instead of any installed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from aeon.tools.vision import AnalyzeImageTool

def main():
    print("[Validation] Initializing AnalyzeImageTool...")
    tool = AnalyzeImageTool()
    
    # Use the large test image to specifically verify the 413 fix (resizing/compression)
    test_image = "scripts/debug/large_test_image.jpg"
    if not os.path.exists(test_image):
        print(f"[Error] Large test image not found at {test_image}")
        # Fallback to any image in the debug folder
        import glob
        images = glob.glob("aeon_output/debug/*.png") + glob.glob("aeon_output/debug/*.jpg")
        if images:
            test_image = images[0]
        else:
            print("[Error] No test images found anywhere.")
            return

    print(f"[Validation] Testing with image: {test_image} (Size: {os.path.getsize(test_image) / (1024*1024):.2f} MB)")
    prompt = "Describe this image in detail."
    
    try:
        print("[Validation] Executing tool.execute()...")
        result = tool.execute(test_image, prompt)
        print("\n[Success] Tool returned result:")
        print("-" * 40)
        print(result)
        print("-" * 40)
    except Exception as e:
        print(f"\n[Failure] Tool execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()