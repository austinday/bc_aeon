import os
import sys

# Ensure the local aeon package is imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from aeon.tools.generate_video import GenerateVideoTool

def live_test():
    print("Starting live test of GenerateVideoTool from /tmp...")
    
    # Instantiate the tool
    tool = GenerateVideoTool()
    
    # Define output path in /tmp to verify directory independence
    output_path = "/tmp/aeon_live_test_video.mp4"
    
    # Use a very short prompt and minimal frames for a fast test
    prompt = "A simple rotating cube, high quality, 4k"
    frames = 16 # Small number of frames for speed
    
    print(f"Generating video: {prompt} ({frames} frames)...")
    print(f"Expected output: {output_path}")
    
    try:
        result = tool.execute(
            mode="text_to_video",
            prompt=prompt,
            output_path=output_path,
            frames=frames,
            width=512,
            height=512
        )
        print(f"Tool result: {result}")
        
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            if size > 0:
                print(f"SUCCESS: Video created at {output_path} (Size: {size} bytes)")
            else:
                print(f"FAILURE: Video file created but is empty.")
                sys.exit(1)
        else:
            print(f"FAILURE: Video file was not created at {output_path}")
            sys.exit(1)
            
    except Exception as e:
        print(f"CRITICAL ERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    live_test()