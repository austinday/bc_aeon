import os
import sys
import shutil

# Add the project root to sys.path to allow importing aeon
PROJECT_ROOT = "/home/aday/bc_aeon"
sys.path.append(PROJECT_ROOT)

from aeon.tools.generate_video import GenerateVideoTool

def test_portability():
    print("Testing GenerateVideoTool portability from /tmp/aeon_test...")
    
    # Initialize the tool
    tool = GenerateVideoTool()
    
    # Define output path in a temporary location to ensure it's not relying on workspace relative paths
    test_output_dir = "/tmp/aeon_test/output"
    os.makedirs(test_output_dir, exist_ok=True)
    output_video = os.path.join(test_output_dir, "portability_test.mp4")
    
    # Test Text-to-Video (Short)
    print("\n--- Testing Text-to-Video (Short) ---")
    try:
        result = tool.execute(
            mode="text_to_video",
            prompt="A cinematic shot of a futuristic city with flying cars, high detail, 4k",
            output_path=output_video,
            width=512,
            height=512,
            frames=16 # Small number for fast validation
        )
        print(f"Result: {result}")
        if os.path.exists(output_video):
            print("SUCCESS: Output video created.")
        else:
            print("FAILURE: Output video not found.")
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")

    # Test Image-to-Video (Short)
    print("\n--- Testing Image-to-Video (Short) ---")
    # Use a small dummy image for testing
    test_image = "/tmp/aeon_test/test_input.jpg"
    from PIL import Image
    img = Image.new('RGB', (512, 512), color = 'red')
    img.save(test_image)
    
    output_video_img = os.path.join(test_output_dir, "portability_test_img.mp4")
    try:
        result = tool.execute(
            mode="image_to_video",
            prompt="The red square starts to morph into a sphere",
            output_path=output_video_img,
            width=512,
            height=512,
            frames=16,
            input_path_1=test_image
        )
        print(f"Result: {result}")
        if os.path.exists(output_video_img):
            print("SUCCESS: Output video created.")
        else:
            print("FAILURE: Output video not found.")
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")

if __name__ == "__main__":
    test_portability()