import os
import json
import time
import base64
import requests
import subprocess
from PIL import Image
import io

def start_server():
    print("[Repro] Starting Gemma 4 Vision Server...")
    script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'aeon', 'scripts', 'start_gemma4_vl.sh'))
    # We run this in the background or let the script handle the wait.
    # The script itself waits for health, but we'll double check.
    process = subprocess.Popen(['bash', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for health check
    url = "http://localhost:8020/health"
    for i in range(120): # 10 minutes
        try:
            if requests.get(url, timeout=2).status_code == 200:
                print("[Repro] Server is healthy!")
                return True
        except:
            pass
        if i % 10 == 0:
            print(f"[Repro] Waiting for server... ({i*5}s)")
        time.sleep(5)
    
    print("[Repro] Server failed to start in time.")
    return False

def test_image_size(size_mb):
    print(f"[Repro] Testing image size: {size_mb}MB")
    # Create a dummy large image
    img = Image.new('RGB', (2000, 2000), color='red')
    # To make it actually large in bytes, we can't just use a solid color (compression).
    # We'll create a large random array.
    import numpy as np
    random_data = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
    img = Image.fromarray(random_data)
    
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=100)
    img_bytes = buf.getvalue()
    
    # If it's not large enough, we'll just pad it or create a larger one.
    # For a 413, we usually need something in the range of 10MB-100MB depending on the limit.
    # Let's just make a really big one.
    if len(img_bytes) < size_mb * 1024 * 1024:
        # Create a massive image
        side = int((size_mb * 1024 * 1024)**0.5)
        random_data = np.random.randint(0, 256, (side, side, 3), dtype=np.uint8)
        img = Image.fromarray(random_data)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=100)
        img_bytes = buf.getvalue()

    print(f"[Repro] Actual image size: {len(img_bytes) / (1024*1024):.2f} MB")
    
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    url = "http://localhost:8020/v1/chat/completions"
    payload = {
        "model": "google/gemma-4-31b-it",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ]
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=180)
        print(f"[Repro] Response status: {resp.status_code}")
        if resp.status_code == 413:
            print("[Repro] SUCCESS: Caught 413 Request Entity Too Large!")
            return True
        elif resp.status_code == 200:
            print("[Repro] Server accepted the image.")
            return False
        else:
            print(f"[Repro] Server returned unexpected status: {resp.status_code}")
            print(resp.text)
            return False
    except Exception as e:
        print(f"[Repro] Request failed: {e}")
        return False

if __name__ == "__main__":
    if start_server():
        # Try a few sizes to find the threshold
        for size in [1, 10, 50, 100]:
            if test_image_size(size):
                print(f"[Repro] Confirmed 413 at {size}MB")
                break
        else:
            print("[Repro] Could not reproduce 413 error.")
    else:
        print("[Repro] Could not start server.")