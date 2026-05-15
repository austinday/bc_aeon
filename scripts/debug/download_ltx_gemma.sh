#!/bin/bash
set -e

# Define paths
CLIP_DIR="aeon_output/comfyui/models/clip"
TEXT_ENC_DIR="aeon_output/comfyui/models/text_encoders"

mkdir -p "$CLIP_DIR"
mkdir -p "$TEXT_ENC_DIR"

# Gemma 3 Text Encoder (GGUF)
GEMMA_URL="https://huggingface.co/unsloth/gemma-3-12b-it-qat-GGUF/resolve/main/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"
GEMMA_FILE="$CLIP_DIR/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"

# Text Projection
PROJ_URL="https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors"
PROJ_FILE="$TEXT_ENC_DIR/ltx-2.3_text_projection_bf16.safetensors"

echo "Downloading Gemma 3 Text Encoder..."
if [ -f "$GEMMA_FILE" ]; then
    echo "Gemma 3 already exists, skipping."
else
    wget -c "$GEMMA_URL" -O "$GEMMA_FILE"
fi

echo "Downloading Text Projection..."
if [ -f "$PROJ_FILE" ]; then
    echo "Text Projection already exists, skipping."
else
    wget -c "$PROJ_URL" -O "$PROJ_FILE"
fi

echo "Downloads complete."
ls -lh "$GEMMA_FILE" "$PROJ_FILE"