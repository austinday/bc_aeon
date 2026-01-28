import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, login
except ImportError:
    print('Error: huggingface_hub not found.')
    sys.exit(1)

# OPTIMIZED REGISTRY: Best-in-Class Models for Each Tier
# vision_small: Qwen2-VL-2B-Instruct (SOTA for <8GB VRAM)
# vision_med:   InternVL2-26B (SOTA for 24GB VRAM, requires 4-bit)
# vision_large: Qwen2-VL-72B-Instruct (SOTA for 80GB+ VRAM, requires 4-bit/8-bit)
MODELS = {
    'flux_schnell': 'black-forest-labs/FLUX.1-schnell',
    'dreamshaper_8': 'Lykon/dreamshaper-8',
    'vision_small': 'Qwen/Qwen2-VL-2B-Instruct',
    'vision_med': 'OpenGVLab/InternVL2-26B',
    'vision_large': 'Qwen/Qwen2-VL-72B-Instruct'
}

def get_token():
    token_path = Path.home() / 'huggingface_access_token.txt'
    if token_path.exists():
        try:
            with open(token_path, 'r') as f:
                token = f.read().strip()
            if token:
                return token
        except: pass
    return None

def main():
    default_path = Path.home() / 'bc_aeon' / 'aeon_models'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=str(default_path))
    parser.add_argument('--model', type=str, help='Specific model key to download')
    args = parser.parse_args()

    token = get_token()
    if token:
        print(f"Authentication token found (len={len(token)}).")
        # CRITICAL FIX: Set env var for underlying libraries to ensure gated access works
        os.environ["HF_TOKEN"] = token
        try:
            login(token=token, add_to_git_credential=False)
        except Exception as e:
            print(f"Login warning (non-fatal): {e}")
    else:
        print("Warning: No Hugging Face token found. Gated models (Flux) will fail.")

    base_dir = Path(args.dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Aeon Vision: Syncing models to {base_dir}...')
    
    targets = {args.model: MODELS[args.model]} if args.model in MODELS else MODELS

    # Files to grab: weights (safetensors/bin), config (json), code (py), metadata (txt/md/model)
    patterns = ["*.safetensors", "*.bin", "*.json", "*.txt", "*.py", "*.model", "*.md"]

    for key, repo_id in targets.items():
        print(f'Checking {key} ({repo_id})...')
        target_dir = base_dir / key
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=patterns,
                token=token
            )
        except Exception as e:
            print(f"Failed {key}: {e}")

if __name__ == '__main__':
    main()