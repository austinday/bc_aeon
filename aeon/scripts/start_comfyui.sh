#!/bin/bash
set -e

AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
MODELS_DIR="$AEON_HOME/models/comfyui"
OUTPUT_DIR="$AEON_HOME/temp/comfyui_output"
mkdir -p "$OUTPUT_DIR"

# Pre-check: image must have been built during setup_environment.sh
if ! docker image inspect aeon_comfyui:latest >/dev/null 2>&1; then
    echo "ERROR: aeon_comfyui:latest image not found. Run setup_environment.sh first."
    exit 1
fi

echo "Starting ComfyUI container on GPU 1..."
docker rm -f aeon_comfyui >/dev/null 2>&1 || true

docker run -d \
    --name aeon_comfyui \
    --gpus device=1 \
    --shm-size=8gb \
    -p 8188:8188 \
    -v "$MODELS_DIR/unet:/workspace/ComfyUI/models/unet" \
    -v "$MODELS_DIR/text_encoders:/workspace/ComfyUI/models/text_encoders" \
    -v "$MODELS_DIR/vae:/workspace/ComfyUI/models/vae" \
    -v "$MODELS_DIR/pulid:/workspace/ComfyUI/models/pulid" \
    -v "$MODELS_DIR/clip:/workspace/ComfyUI/models/clip" \
    -v "$MODELS_DIR/insightface:/workspace/ComfyUI/models/insightface" \
    -v "$OUTPUT_DIR:/workspace/ComfyUI/output" \
    aeon_comfyui:latest

echo "ComfyUI is starting! It will be available at http://localhost:8188"
echo "Check logs with: docker logs -f aeon_comfyui"
