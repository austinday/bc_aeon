import os
import json
import time
import base64
import requests
import subprocess
from PIL import Image
import io

def create_large_image(size_mb=15):
    """Creates a large dummy image to trigger 413."""
    print(f"Creating a ~{size_mb}MB dummy image...")
    # Create a large image (e.g., 5000x5000)
    img = Image.new('RGB', (5000, 5000), color='red')
    img_byte_arr = io.BytesIO()
    # Use a format that preserves size or just save as a large png
    img.save(img_byte_arr, format='PNG')
    data = img_byte_arr.getvalue()
    
    # If it's not large enough, we can just pad it or create a larger one
    # But for 413, we just need it to exceed the server limit (usually 1MB, 10MB, etc.)
    return data

def main():
    port = 8020
    url = f'http://localhost:{port}/v1'
    health_url = f'http://localhost:{port}/health'
    
    # 1. Start the server using the existing script
    print("Starting Gemma 4 Vision Server...")
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../aeon/scripts/start_gemma4_vl.sh'))
    subprocess.run(['bash', script_path], check=True)
    
    # 2. Wait for health check
    print("Waiting for server to be healthy...")
    max_retries = 60
    ready = False
    for i in range(max_retries):
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                print("Server is healthy!")
                ready = True
                break
        except Exception as e:
            pass
        print(f"Retry {i+1}/{max_retries}...", end='\r')
        time.sleep(5)
    
    if not ready:
        print("\nServer failed to become healthy in time.")
        return

    # 3. Send a large image
    image_data = create_large_image(size_mb=20)
    b64_image = base64.b64encode(image_data).decode('utf-8')
    
    payload = {
        "model": "google/gemma-4-31b-it",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }
        ],
        "max_tokens": 100
    }
    
    print(f"Sending request with payload size: {len(json.dumps(payload)) / (1024*1024):.2f} MB...")
    try:
        response = requests.post(f'{url}/chat/completions', json=payload, timeout=300)
        print(f"Response Status Code: {response.status_code}")
        if response.status_code == 413:
            print("SUCCESS: Reproduced 413 Request Entity Too Large!")
        elif response.status_code == 200:
            print("FAILURE: Request succeeded. No 413 error.")
        else:
            print(f"Unexpected status code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Request failed with exception: {e}")
    finally:
        # Cleanup
        print("Cleaning up container...")
        subprocess.run(['docker', 'rm', '-f', 'aeon_gemma4_vl'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    main()