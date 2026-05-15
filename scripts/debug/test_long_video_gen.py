import os
from aeon.tools.generate_video import GenerateVideoTool

def test_long_video():
    print("Initializing GenerateVideoTool...")
    tool = GenerateVideoTool()
    
    prompt = "A cinematic shot of a futuristic city with flying cars, neon lights, raining, high detail, 4k"
    output_path = "aeon_output/debug/test_10sec_video.mp4"
    frames = 240 # 10 seconds * 24 fps
    
    print(f"Generating long video: text_to_video, frames={frames}...")
    result = tool.execute(
        mode="text_to_video",
        prompt=prompt,
        output_path=output_path,
        frames=frames
    )
    
    print(f"Result: {result}")
    
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"SUCCESS: Video saved to {output_path}")
        print(f"File size: {size} bytes")
    else:
        print(f"FAILURE: Video file not found at {output_path}")
        exit(1)

if __name__ == "__main__":
    test_long_video()