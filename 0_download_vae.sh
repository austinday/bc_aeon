#!/bin/bash
set -e

# Configuration
HF_TOKEN_FILE="$HOME/huggingface_access_token.txt"
VAE_DEST="/home/aday/bc_aeon/aeon_models/comfyui/split_files/vae/ae.safetensors"
# Using the official Black Forest Labs Flux.1-dev VAE
VAE_URL="https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors"

if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "ERROR: Hugging Face token file not found at $HF_TOKEN_FILE"
    exit 1
fi

TOKEN=$(cat "$HF_TOKEN_FILE")

if [ -f "$VAE_DEST" ]; then
    echo "VAE already exists at $VAE_DEST. Skipping download."
else
    echo "Downloading Flux VAE to $VAE_DEST..."
    mkdir -p "$(dirname "$VAE_DEST")"
    curl -L -H "Authorization: Bearer $TOKEN" "$VAE_URL" -o "$VAE_DEST"
    echo "Download complete."
fi