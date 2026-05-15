import os
import sys

# Ensure the root directory is in the path so we can import aeon
sys.path.append(os.getcwd())

from aeon.tools.generate_video import GenerateVideoTool

def main():
    print("Initializing GenerateVideoTool for 10-second video generation...")
    tool = GenerateVideoTool()
    
    output_path = "aeon_output/debug/final_10s_video.mp4"
    prompt = "A cinematic shot of a futuristic city with flying cars, neon lights, rainy streets, high detail, 4k, highly realistic"
    frames = 240  # 10 seconds * 24 fps
    
    print(f"Starting generation: {frames} frames, prompt: {prompt}")
    result = tool.execute(
        mode="text_to_video",
        prompt=prompt,
        output_path=output_path,
        width=768,
        height=512,
        frames=frames
    )
    
    print(f"\nTool Result: {result}")
    
    if os.path.exists(output_path):
        print(f"\nSUCCESS: Video generated at {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
    else:
        print(f"\nFAILURE: Video file not found at {output_path}")

if __name__ == "__main__":
    main()