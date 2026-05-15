import os
import requests
from tqdm import tqdm

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}. Skipping.")
        return

    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)
    ) as bar:
        for data in response.iter_content(chunk_size=1024 * 1024):
            size = f.write(data)
            bar.update(size)

# Model definitions: (URL, destination_relative_path)
# Using 'resolve' instead of 'blob' for direct downloads
models = [
    (
        "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/ltx-2.3-22b-dev-Q4_1.gguf", 
        "aeon_output/comfyui/models/unet/ltx-2.3-22b-dev-Q4_1.gguf"
    ),
    (
        "https://huggingface.co/UmeAiRT/ComfyUI-Auto-Installer-Assets/resolve/88a47ea21217a605d54694d25e6d10cc2d757854/t5xxl_fp8_e4m3fn.safetensors", 
        "aeon_output/comfyui/models/clip/t5xxl_fp8_e4m3fn.safetensors"
    ),
    (
        "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-dev_video_vae.safetensors", 
        "aeon_output/comfyui/models/vae/ltx-2.3-22b-dev_video_vae.safetensors"
    ),
]

if __name__ == "__main__":
    try:
        for url, path in models:
            download_file(url, path)
        print("\nAll models downloaded successfully.")
    except Exception as e:
        print(f"\nError downloading models: {e}")
        exit(1)