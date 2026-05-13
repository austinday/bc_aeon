import os
import sys
from aeon.tools.generate_video import GenerateVideoTool

def main():
    print("Initializing GenerateVideoTool for Long Video Validation...")
    tool = GenerateVideoTool()
    
    # Prevent the tool from killing the container during tests
    tool._manage_registry = lambda action: 1 

    prompt = "A cinematic drone shot of a futuristic city with flying cars, highly detailed, smooth motion."
    output_path = "aeon_output/debug/long_video_test.mp4"
    
    # Request 100 frames (approx 3 chunks of 33 + 1 small chunk)
    frames = 100
    print(f"Generating long video with prompt: {prompt}")
    print(f"Target frames: {frames}")
    print(f"Output path: {output_path}")
    
    try:
        result = tool.execute(
            mode="text_to_video",
            prompt=prompt,
            output_path=output_path,
            width=512,
            height=432,
            frames=frames
        )
        print(f"\n=== TOOL RESULT ===")
        print(result)
        
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"\nSuccess! Long video created at {output_path} (Size: {size} bytes)")
            
            # Basic check: is it actually a video?
            # We can use ffprobe via docker to check the frame count
            import subprocess
            cwd = os.getcwd()
            rel_path = os.path.relpath(output_path, cwd)
            cmd = [
                "docker", "run", "--rm", 
                "-v", f"{cwd}:/app", 
                "-w", "/app",
                "mwader/static-ffmpeg", 
                "ffprobe", "-v", "error", "-select_streams", "v:0", 
                "-count_packets", "-show_entries", "stream=nb_read_packets", 
                f"/app/{rel_path}"
            ]
            probe = subprocess.run(cmd, capture_output=True, text=True)
            print(f"Probe output: {probe.stdout}")
        else:
            print("\nError: Output file was not created.")
            
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists("aeon"):
        print("Please run this script from the project root.")
        sys.exit(1)
    main()