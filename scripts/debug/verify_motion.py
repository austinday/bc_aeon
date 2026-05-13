import os
import subprocess
import shutil
import numpy as np
import cv2

def verify_motion(video_path):
    print(f"Analyzing video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    frames_dir = "aeon_output/debug/motion_analysis_frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # Use docker to run ffmpeg and extract frames
    # We mount the current directory to /app
    cwd = os.getcwd()
    rel_video_path = os.path.relpath(video_path, cwd)
    rel_frames_dir = os.path.relpath(frames_dir, cwd)
    
    print("Extracting frames using ffmpeg...")
    # Use mwader/static-ffmpeg for a reliable, lightweight ffmpeg binary
    cmd = [
        "docker", "run", "--rm", 
        "-v", f"{cwd}:/app", 
        "-w", "/app",
        "mwader/static-ffmpeg", 
        "-i", f"/app/{rel_video_path}", 
        f"/app/{rel_frames_dir}/frame_%03d.png"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg extraction failed: {e.stderr.decode()}")
        return

    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if len(frames) < 2:
        print("Error: Not enough frames extracted to analyze motion.")
        return

    print(f"Extracted {len(frames)} frames. Calculating pixel differences...")
    
    first_frame = cv2.imread(os.path.join(frames_dir, frames[0]), cv2.IMREAD_GRAYSCALE)
    if first_frame is None:
        print("Error: Could not read the first frame.")
        return

    diffs = []
    for i in range(1, len(frames)):
        current_frame = cv2.imread(os.path.join(frames_dir, frames[i]), cv2.IMREAD_GRAYSCALE)
        if current_frame is None:
            continue
        
        # Calculate Absolute Difference
        diff = cv2.absdiff(first_frame, current_frame)
        mean_diff = np.mean(diff)
        diffs.append(mean_diff)

    avg_diff = np.mean(diffs)
    max_diff = np.max(diffs)

    print(f"\n--- Motion Analysis Report ---")
    print(f"Average Pixel Difference: {avg_diff:.4f}")
    print(f"Max Pixel Difference: {max_diff:.4f}")
    
    # Threshold for "motion": 
    # 0.0 means identical. 
    # Very low (e.g. < 1.0) usually indicates a slideshow or static image.
    if avg_diff < 1.0:
        print("RESULT: Slideshow effect persists (frames are nearly identical).")
    elif avg_diff < 5.0:
        print("RESULT: Very subtle motion detected, but likely still too static.")
    else:
        print("RESULT: Significant motion detected!")

if __name__ == "__main__":
    video_to_test = "aeon_output/debug/motion_test.mp4"
    verify_motion(video_to_test)