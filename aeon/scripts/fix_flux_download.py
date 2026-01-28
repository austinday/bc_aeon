import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

def main():
    # 1. Load Token
    token_path = Path.home() / 'huggingface_access_token.txt'
    if not token_path.exists():
        print("Error: No token file.")
        return
    with open(token_path, 'r') as f:
        token = f.read().strip()

    # 2. Set Env Var (Force Auth)
    os.environ["HF_TOKEN"] = token
    print(f"Loaded token (len={len(token)}).")

    repo_id = "black-forest-labs/FLUX.1-schnell"
    local_dir = Path.home() / 'bc_aeon' / 'aeon_models' / 'flux_schnell'
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n1. Testing connection to {repo_id}...")
    try:
        path = hf_hub_download(repo_id=repo_id, filename="README.md", local_dir=local_dir, token=token)
        print(f"Success! README downloaded to: {path}")
    except Exception as e:
        print(f"Failed to download README: {e}")
        return

    print("\n2. Downloading Model Weights (*.safetensors)...")
    try:
        # Download only the critical model weights first
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=["*.safetensors", "*.json"], # Get weights + config
            token=token
        )
        print("\nSUCCESS: Flux.1-Schnell downloaded successfully.")
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")

if __name__ == "__main__":
    main()