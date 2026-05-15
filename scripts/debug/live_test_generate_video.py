import os
import sys
import shutil

# Add the project root to the front of sys.path to ensure the local version is imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from aeon.tools.generate_video import GenerateVideoTool

def live_test():
    print("Starting Live Test for GenerateVideoTool from CWD:", os.getcwd())
    
    # Instantiate the tool
    tool = GenerateVideoTool()
    
    # Define a test output path in /tmp to ensure we aren't relying on workspace
    test_output = "/tmp/aeon_live_test_video.mp4"
    if os.path.exists(test_output):
        os.remove(test_output)
        
    try:
        print("Generating a short test video (text-to-video)...")
        # Use a very short frame count for a quick test
        result = tool.execute(
            mode="text_to_video",
            prompt="A simple cinematic shot of a floating cube",
            output_path=test_output,
            frames=16 # Small number for speed
        )
        print(f"Tool result: {result}")
        
        if os.path.exists(test_output):
            size = os.path.getsize(test_output)
            print(f"SUCCESS: Video generated at {test_output} (Size: {size} bytes)")
            if size > 0:
                print("Video file is not empty.")
            else:
                print("ERROR: Video file is empty.")
                sys.exit(1)
        else:
            print(f"ERROR: Output file {test_output} was not created.")
            sys.exit(1)
            
    except Exception as e:
        print(f"EXCEPTION during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if os.path.exists(test_output):
            os.remove(test_output)
            print("Cleaned up test output file.")

    print("\nLive test PASSED: GenerateVideoTool works independently of the workspace directory.")

if __name__ == "__main__":
    live_test()