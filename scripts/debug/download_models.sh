#!/bin/bash

# Directories
UNET_DIR="aeon_output/comfyui/models/unet"
CLIP_DIR="aeon_output/comfyui/models/clip"
VAE_DIR="aeon_output/comfyui/models/vae"

mkdir -p $UNET_DIR $CLIP_DIR $VAE_DIR

# URLs
UNET_URL="https://huggingface.co/unsloth/LTX-2.3-GGUF/resolve/main/ltx-2.3-22b-dev-Q4_1.gguf"
CLIP_URL="https://huggingface.co/Lightricks/LTX-Video/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
VAE_URL="https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-2.3-22b-dev_video_vae.safetensors"

echo "Downloading UNet model (GGUF)..."
wget -c "$UNET_URL" -O "$UNET_DIR/ltx-2.3-22b-dev-Q4_1.gguf"

echo "Downloading CLIP model..."
wget -c "$CLIP_URL" -O "$CLIP_DIR/t5xxl_fp8_e4m3fn.safetensors"

echo "Downloading VAE model..."
wget -c "$VAE_URL" -O "$VAE_DIR/ltx-2.3-22b-dev_video_vae.safetensors"

echo "Downloads complete. Verifying files..."
ls -lh $UNET_DIR/ltx-2.3-22b-dev-Q4_1.gguf
ls -lh $CLIP_DIR/t5xxl_fp8_e4m3fn.safetensors
ls -lh $VAE_DIR/ltx-2.3-22b-dev_video_vae.safetensors