import os
import base64
import requests
import time
from PIL import Image
import io

# Configuration
URL = "http://localhost:8020/v1/chat/completions"
MODEL = "google/gemma-4-31b-it"

def create_image(size_mb):
    """Creates a dummy image of approximately size_mb MB."""
    # 1 MB is roughly 1024 * 1024 pixels for a grayscale image, 
    # but we'll use a large RGB image and save as JPEG to control size.
    # A 2000x2000 RGB image is ~12MB raw.
    width = 2000
    height = int((size_mb * 1024 * 1024) / (width * 3)) 
    if height < 1: height = 1
    
    img = Image.new('RGB', (width, height), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()

def test_request(image_data, use_proxy_disable=False):
    b64 = base64.b64encode(image_data).decode('utf-8')
    payload = {
        "model": MODEL,
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
    
    proxies = {"http": None, "https": None} if use_proxy_disable else None
    
    try:
        print(f"Sending request (Size: {len(image_data)/1024/1024:.2f} MB, ProxyDisable: {use_proxy_disable})...")
        resp = requests.post(URL, json=payload, proxies=proxies, timeout=120)
        return resp.status_code, resp.text
    except Exception as e:
        return "ERROR", str(e)

def main():
    # 1. Ensure server is running (simple health check)
    print("Checking server health...")
    try:
        health = requests.get("http://localhost:8020/health", timeout=5)
        if health.status_code != 200:
            print(f"Server not healthy: {health.status_code}")
            return
        print("Server is healthy.")
    except Exception as e:
        print(f"Could not connect to server: {e}")
        print("Attempting to start server via script...")
        os.system("bash aeon/scripts/start_gemma4_vl.sh")
        time.sleep(30) # Give it some time to boot

    # Test cases: (Image Size MB, Proxy Disable)
    tests = [
        (0.1, False), # Small, Default
        (0.1, True),  # Small, No Proxy
        (10, False),  # Large, Default
        (10, True),   # Large, No Proxy
    ]

    results = []
    for size, disable in tests:
        img_data = create_image(size)
        status, text = test_request(img_data, use_proxy_disable=disable)
        results.append((size, disable, status, text))
        print(f"Result: Size={size}MB, DisableProxy={disable} -> Status={status}")

    print("\n--- FINAL REPORT ---")
    for size, disable, status, text in results:
        print(f"Size: {size}MB | ProxyDisable: {disable} | Status: {status} | Response: {text[:100]}")

if __name__ == "__main__":
    main()