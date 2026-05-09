import os
import subprocess
import json

def get_video_info(path):
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,duration,avg_frame_rate",
            "-of", "json", path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data['streams'][0]
        
        duration = float(stream.get('duration', 0))
        frames = int(stream.get('nb_frames', 0))
        fps_str = stream.get('avg_frame_rate', '0/0')
        num, den = map(int, fps_str.split('/'))
        fps = num / den if den != 0 else 0
        
        return {"duration": duration, "frames": frames, "fps": fps}
    except Exception as e:
        return {"error": str(e)}

videos = [
    "aeon_output/debug/repro_0_text_to_video_16:9.mp4",
    "aeon_output/debug/repro_1_text_to_video_9:16.mp4",
    "aeon_output/debug/repro_2_image_to_video_16:9.mp4",
    "aeon_output/debug/repro_3_image_to_video_9:16.mp4",
]

print(f"{'File':<50} | {'Duration':<10} | {'Frames':<10} | {'FPS':<10}")
print("-" * 85)
for v in videos:
    if os.path.exists(v):
        info = get_video_info(v)
        if "error" in info:
            print(f"{v:<50} | Error: {info['error']}")
        else:
            print(f"{v:<50} | {info['duration']:<10.2f} | {info['frames']:<10} | {info['fps']:<10.2f}")
    else:
        print(f"{v:<50} | Not Found")