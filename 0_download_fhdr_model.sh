#!/bin/bash
set -e

# Configuration
PROJECT_ROOT="/home/aday/bc_aeon"
MODEL_DIR="$PROJECT_ROOT/aeon_models/comfyui/unet"
MODEL_NAME="FHDR_ComfyUI-Q8_0.gguf"
REPO="kpsss34/FHDR_Uncensored"
TOKEN_FILE="$HOME/huggingface_access_token.txt"
URL="https://huggingface.co/$REPO/resolve/main/$MODEL_NAME?download=true"

echo "Checking for Hugging Face token..."
if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: Token file not found at $TOKEN_FILE"
    exit 1
fi

TOKEN=$(cat "$TOKEN_FILE")

echo "Ensuring model directory exists: $MODEL_DIR"
mkdir -p "$MODEL_DIR"

DEST_PATH="$MODEL_DIR/$MODEL_NAME"

if [ -f "$DEST_PATH" ]; then
    echo "Model $MODEL_NAME already exists at $DEST_PATH. Skipping download."
else
    echo "Downloading $MODEL_NAME from $REPO..."
    # Use curl with -L to follow redirects and -H for authorization
    # -# shows a simple progress bar
    curl -L -# -H "Authorization: Bearer $TOKEN" -o "$DEST_PATH" "$URL"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $MODEL_NAME to $DEST_PATH"
    else
        echo "ERROR: Failed to download model."
        exit 1
    fi
fi

echo "Verification: $(ls -lh $DEST_PATH)"