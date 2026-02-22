#!/bin/bash
set -e

PROJECT_ROOT="$HOME/bc_aeon"
MODELS_DIR="$PROJECT_ROOT/aeon_models/comfyui"

# Load HF_TOKEN if not set
HF_TOKEN_FILE="$HOME/huggingface_access_token.txt"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '\n')
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN environment variable required for model downloads."
    echo "Please export HF_TOKEN or place it in ~/huggingface_access_token.txt"
    exit 1
fi

mkdir -p "$MODELS_DIR/unet"
mkdir -p "$MODELS_DIR/text_encoders"
mkdir -p "$MODELS_DIR/vae"

echo "Preparing to download FLUX GGUF models and encoders..."

DL_SCRIPT=$(mktemp /tmp/aeon_dl_flux_XXXXXX.py)
cat > "$DL_SCRIPT" << 'PYEOF'
import os
from huggingface_hub import hf_hub_download

# Flux 2 Dev UNet GGUF (no fallback — require FLUX.2)
print("Downloading Flux 2 Dev UNet GGUF...")
hf_hub_download(repo_id="unsloth/FLUX.2-dev-GGUF", filename="flux2-dev-Q8_0.gguf", local_dir="/models/unet")

# Flux 2 VAE
print("Downloading Flux 2 VAE...")
hf_hub_download(repo_id="Comfy-Org/flux2-dev", filename="vae/flux2-vae.safetensors", local_dir="/models")

# Text Encoder (Mistral)
print("Downloading FLUX.2 Mistral text encoder...")
hf_hub_download(repo_id="Comfy-Org/flux2-dev", filename="text_encoders/mistral_3_small_flux2_bf16.safetensors", local_dir="/models")

print("Downloads complete!")
PYEOF

# Run the download script inside the aeon_downloader container
docker run --rm -t \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$MODELS_DIR:/models" \
    -v "$DL_SCRIPT:/download.py:ro" \
    aeon_downloader:latest \
    python3 /download.py

rm -f "$DL_SCRIPT"
echo "FLUX models downloaded successfully to $MODELS_DIR"
