import os
import sys

# Add the root directory to sys.path so we can import aeon
sys.path.append(os.getcwd())

try:
    from aeon.tools.generate_video import GenerateVideoTool
    print("[TEST] Successfully imported GenerateVideoTool")
except Exception as e:
    print(f"[FAILURE] Import failed: {e}")
    sys.exit(1)

def test_generation():
    print("[TEST] Initializing GenerateVideoTool...")
    tool = GenerateVideoTool()
    
    prompt = "A cinematic shot of a futuristic city with flying cars, neon lights, rain falling, 4k, highly detailed"
    output_path = "aeon_output/final_test_video.mp4"
    
    print(f"[TEST] Generating video with prompt: {prompt}")
    print(f"[TEST] Target output path: {output_path}")
    
    try:
        # Use the execute method which now uses LtxvApiTextToVideo
        result = tool.execute(
            mode="text_to_video",
            prompt=prompt,
            output_path=output_path,
            width=768,
            height=512,
            frames=33
        )
        print(f"[RESULT] Tool output: {result}")
        
        if "successfully" in result.lower():
            print("[SUCCESS] Video generation reported as successful.")
        else:
            print(f"[FAILURE] Tool reported an error: {result}")
            
    except Exception as e:
        print(f"[FAILURE] Exception during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generation()