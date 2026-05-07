import sys
import os

# Add the root directory to sys.path so we can import aeon
sys.path.append(os.getcwd())

from aeon.tools.generate_video import GenerateVideoTool

def test_gen():
    print("Initializing GenerateVideoTool...")
    tool = GenerateVideoTool()
    
    prompt = "A cinematic shot of a futuristic city with flying cars, high detail, 4k"
    output_path = "aeon_output/debug/test_video.mp4"
    
    print(f"Testing text_to_video with prompt: {prompt}")
    result = tool.execute(
        mode='text_to_video',
        prompt=prompt,
        output_path=output_path,
        width=768,
        height=512,
        frames=33
    )
    print(f"Result: {result}")

if __name__ == "__main__":
    try:
        test_gen()
    except Exception as e:
        print(f"Exception occurred: {e}")
        import traceback
        traceback.print_exc()