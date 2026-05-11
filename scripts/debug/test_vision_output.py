import sys
import os

# Add root to path so we can import aeon
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from aeon.tools.vision import AnalyzeImageTool

def main():
    tool = AnalyzeImageTool()
    # Use a known image in the root directory
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../screenshot_1.jpg'))
    prompt = 'Describe this image in detail.'
    
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        sys.exit(1)

    print(f"Testing vision tool with image: {image_path}")
    print(f"Prompt: {prompt}\n")
    
    try:
        result = tool.execute(image_path, prompt)
        print("\n--- Returned Result (from function return) ---")
        print(result)
        print("----------------------------------------------")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()