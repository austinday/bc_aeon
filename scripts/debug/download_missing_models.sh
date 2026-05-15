#!/bin/bash
set -e

# Directories
CLIP_DIR="aeon_output/comfyui/models/clip"
VAE_DIR="aeon_output/comfyui/models/vae"
UNET_DIR="aeon_output/comfyui/models/unet"

mkdir -p "$CLIP_DIR" "$VAE_DIR" "$UNET_DIR"

echo "Downloading CLIP model..."
wget -c "https://huggingface.co/UmeAiRT/ComfyUI-Auto-Installer-Assets/resolve/main/t5xxl_fp8_e4m3fn.safetensors" -O "$CLIP_DIR/t5xxl_fp8_e4m3fn.safetensors"

echo "Downloading VAE model..."
wget -c "https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-dev_video_vae.safetensors" -O "$VAE_DIR/ltx-2.3-22b-dev_video_vae.safetensors"

echo "Models download process complete."