#!/bin/bash
set -e

# Directories
CLIP_DIR="aeon_output/comfyui/models/clip"
VAE_DIR="aeon_output/comfyui/models/vae"
UNET_DIR="aeon_output/comfyui/models/unet"

mkdir -p "$CLIP_DIR" "$VAE_DIR" "$UNET_DIR"

# URLs (Converted from blob to resolve)
CLIP_URL="https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
VAE_URL="https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-dev_video_vae.safetensors"

echo "Downloading CLIP model..."
wget -c "$CLIP_URL" -O "$CLIP_DIR/t5xxl_fp8_e4m3fn.safetensors"

echo "Downloading VAE model..."
wget -c "$VAE_URL" -O "$VAE_DIR/ltx-2.3-22b-dev_video_vae.safetensors"

echo "Models download complete."