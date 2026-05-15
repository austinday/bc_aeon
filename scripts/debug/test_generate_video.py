import os
import sys
import shutil
from pathlib import Path

# Ensure the aeon package is importable from anywhere
# We assume the project root is /home/aday/bc_aeon
PROJECT_ROOT = "/home/aday/bc_aeon"
sys.path.append(PROJECT_ROOT)

try:
    from aeon.tools.generate_video import GenerateVideoTool
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_portability():
    print("Starting generate_video portability test...")
    
    # 1. Setup test environment
    test_dir = Path("/tmp/aeon_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy input image for image-to-video test
    input_image = test_dir / "test_input.jpg"
    # Try to copy a small image from the workspace for realism
    workspace_img = Path(PROJECT_ROOT) / "scripts/debug/small_test.jpg"
    if workspace_img.exists():
        shutil.copy(workspace_img, input_image)
        print(f"Using workspace image: {workspace_img}")
    else:
        try:
            from PIL import Image
            img = Image.new('RGB', (512, 512), color = 'white')
            img.save(input_image)
            print("Created dummy white image for testing.")
        except ImportError:
            print("PIL not installed, cannot create dummy image. Image-to-video test may fail.")

    tool = GenerateVideoTool()
    
    # 2. Test Text-to-Video
    print("\n--- Testing Text-to-Video ---")
    t2v_output = test_dir / "t2v_result.mp4"
    try:
        result = tool.execute(
            mode="text_to_video",
            prompt="A cinematic shot of a futuristic city with flying cars, high detail, 4k",
            output_path=str(t2v_output),
            width=768,
            height=512,
            frames=33
        )
        print(f"Result: {result}")
        if t2v_output.exists():
            print(f"SUCCESS: Output file created at {t2v_output}")
        else:
            print("FAILURE: Output file not found")
    except Exception as e:
        print(f"EXCEPTION: {e}")

    # 3. Test Image-to-Video
    print("\n--- Testing Image-to-Video ---")
    i2v_output = test_dir / "i2v_result.mp4"
    try:
        if not input_image.exists():
            raise FileNotFoundError(f"Input image missing: {input_image}")
            
        result = tool.execute(
            mode="image_to_video",
            prompt="The image comes to life, subtle movement, cinematic",
            output_path=str(i2v_output),
            width=768,
            height=512,
            frames=33,
            input_path_1=str(input_image)
        )
        print(f"Result: {result}")
        if i2v_output.exists():
            print(f"SUCCESS: Output file created at {i2v_output}")
        else:
            print("FAILURE: Output file not found")
    except Exception as e:
        print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    test_portability()