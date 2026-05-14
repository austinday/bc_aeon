import os
import requests
from tqdm import tqdm

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}. Skipping.")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024*1024):
            size = f.write(data)
            bar.update(size)

def main():
    # Base paths
    base_model_dir = "/home/aday/.aeon/models/comfyui"
    
    # Unsloth LTX-2.3 GGUF URLs
    # Using Q4_K_M as a balanced choice
    models = {
        "unet/ltx-2.3-unsloth-q4_k_m.gguf": "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/LTX-2.3-22B-dev-Q4_K_M.gguf",
        "text_encoders/ltx-2.3-unsloth-connector.safetensors": "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/ltx-2.3-22b-dev_embeddings_connectors.safetensors"
    }

    try:
        for rel_path, url in models.items():
            full_path = os.path.join(base_model_dir, rel_path)
            download_file(url, full_path)
        print("\nAll downloads completed successfully.")
    except Exception as e:
        print(f"\nError during download: {e}")
        exit(1)

if __name__ == "__main__":
    main()