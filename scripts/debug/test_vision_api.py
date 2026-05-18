import os
import base64
import requests
import json
from PIL import Image
import io

def test_vision():
    image_path = '/home/aday/bc_aeon/aeon_output/test_fix_image.png'
    prompt = "Describe this image in detail."
    url = 'http://localhost:8020/v1/chat/completions'
    model = 'Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P'

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Loading image: {image_path}")
    img = Image.open(image_path)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')

    # Use a smaller size for the first test to see if it's a payload issue
    MAX_DIM = 512 
    w, h = img.size
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                {'type': 'text', 'text': prompt}
            ]
        }
    ]

    payload = {
        'model': model,
        'messages': messages,
        'max_tokens': 512,
        'temperature': 0.3,
    }

    print("Sending request to vision server...")
    try:
        resp = requests.post(url, json=payload, timeout=120)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_vision()