import os
import subprocess
import json

def get_duration(file_path):
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        return f"Error: {e}"

videos = [
    "aeon_output/debug/repro_0_text_to_video_16:9.mp4",
    "aeon_output/debug/repro_1_text_to_video_9:16.mp4",
    "aeon_output/debug/repro_2_image_to_video_16:9.mp4",
    "aeon_output/debug/repro_3_image_to_video_9:16.mp4",
]

print(f"{'File':<50} | {'Duration (s)':<15}")
print("-" * 67)
for v in videos:
    if os.path.exists(v):
        dur = get_duration(v)
        print(f"{v:<50} | {dur:<15}")
    else:
        print(f"{v:<50} | NOT FOUND")