import requests
import json

def test_api():
    url = "http://localhost:8188/prompt"
    
    # This is a highly simplified LTX-Video workflow attempt
    # We use a very basic structure to see if the server accepts it
    workflow = {
        "prompt": {
            "100": {
                "class_type": "CLIPLoaderGGUF",
                "inputs": {
                    "clip_name": "mistral_3_small_flux2_fp8.safetensors"
                }
            },
            "101": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "A cinematic video of a futuristic city",
                    "clip": ["100", 0]
                }
            },
            "102": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 20,
                    "cfg": 8,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["103", 0],
                    "positive": ["101", 0],
                    "negative": ["104", 0],
                    "latent_image": ["105", 0]
                }
            },
            "103": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "ltx_video_model.safetensors"
                }
            },
            "104": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "low quality, blurry",
                    "clip": ["100", 0]
                }
            },
            "105": {
                "class_type": "EmptyLTXVideoLatent",
                "inputs": {
                    "width": 768,
                    "height": 512,
                    "num_frames": 33
                }
            }
        }
    }

    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=workflow)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()