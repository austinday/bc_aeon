import os
import sys

# Add the root directory to sys.path to allow importing aeon
sys.path.append(os.getcwd())

from aeon.tools.generate_video import GenerateVideoTool

def test_fix():
    print("Initializing GenerateVideoTool...")
    tool = GenerateVideoTool()
    
    # We use a small number of frames for a quick check
    prompt = "A pretty Asian girl jumping up and down in a swimsuit"
    output_path = "aeon_output/debug/verify_fix.mp4"
    
    print(f"Testing single chunk generation with prompt: {prompt}")
    try:
        # Use _generate_single_chunk to bypass the recursive logic for a quick test
        result = tool._generate_single_chunk(
            mode="text_to_video",
            prompt=prompt,
            output_path=output_path,
            width=768,
            height=512,
            frames=16, # Small chunk for verification
            image_path=None
        )
        print(f"Result: {result}")
        if os.path.exists(output_path):
            print("SUCCESS: Video file created.")
        else:
            print("FAILURE: Video file not found.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_fix()