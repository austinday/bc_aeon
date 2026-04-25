import os
import sys
import time
from aeon.tools.generate_video import GenerateVideoTool

def main():
    print("Initializing GenerateVideoTool in DEBUG mode...")
    tool = GenerateVideoTool()
    
    # Monkey-patch the registry manager to ALWAYS return 1 (simulating another active user).
    # This prevents the `finally` block from killing the aeon_comfyui container,
    # allowing us to read the docker logs and inspect the API response.
    tool._manage_registry = lambda action: 1 

    prompt = "A high-energy TikTok style advertisement for a t-shirt on a tropical beach."
    output_path = "aeon_output/debug/validation_test_video.mp4"
    
    print(f"Generating video with prompt: {prompt}")
    print(f"Output path: {output_path}")
    
    try:
        result = tool.execute(
            mode="text_to_video",
            prompt=prompt,
            output_path=output_path,
            width=512,
            height=768,
            frames=17  # Shortened to 17 frames (8*2 + 1) for a faster test
        )
        print(f"\n=== TOOL RESULT ===")
        print(result)
        
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"\nSuccess! Video created at {output_path} (Size: {size} bytes)")
        else:
            print("\nError: Output file was not created.")
            
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure we run from the project root so paths align
    if not os.path.exists("aeon"):
        print("Please run this script from the project root: python3 tests/validate_video_gen.py")
        sys.exit(1)
    main()
