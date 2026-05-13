import os
import subprocess
import threading
import time
from aeon.tools.generate_video import GenerateVideoTool

def monitor_vram(stop_event, log_file):
    print(f"Monitoring VRAM... logging to {log_file}")
    with open(log_file, "w") as f:
        while not stop_event.is_set():
            try:
                res = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], 
                                   capture_output=True, text=True)
                f.write(f"{time.time()}: {res.stdout.strip()}\n")
                f.flush()
            except Exception as e:
                f.write(f"Error: {e}\n")
            time.sleep(0.5)

def main():
    log_file = "aeon_output/debug/vram_monitor.log"
    os.makedirs("aeon_output/debug", exist_ok=True)
    
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_vram, args=(stop_event, log_file))
    monitor_thread.start()

    try:
        print("Initializing GenerateVideoTool...")
        tool = GenerateVideoTool()
        
        # Minimal request to test stability
        print("Triggering minimal video generation...")
        result = tool.execute(
            mode="text_to_video",
            prompt="A simple red ball bouncing",
            output_path="aeon_output/debug/stability_test.mp4",
            width=256,
            height=256,
            frames=17
        )
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        stop_event.set()
        monitor_thread.join()
        print(f"VRAM log saved to {log_file}")

if __name__ == "__main__":
    main()