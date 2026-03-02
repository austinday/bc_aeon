#!/bin/bash
set -e

PROJECT_ROOT="$HOME/bc_aeon"
MODELS_DIR="$PROJECT_ROOT/aeon_models/comfyui"
OUTPUT_DIR="$PROJECT_ROOT/aeon_output/comfyui"
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
    -p 8188:8188 \
    -v "$PROJECT_ROOT/aeon_models/comfyui/unet:/workspace/ComfyUI/models/unet" \
    -v "$PROJECT_ROOT/aeon_models/comfyui/split_files/text_encoders:/workspace/ComfyUI/models/text_encoders" \
    -v "$PROJECT_ROOT/aeon_models/comfyui/split_files/vae:/workspace/ComfyUI/models/vae" \
    -v "$PROJECT_ROOT/aeon_models/comfyui/pulid:/workspace/ComfyUI/models/pulid" \
    -v "$PROJECT_ROOT/aeon_models/comfyui/clip:/workspace/ComfyUI/models/clip" \
    -v "$PROJECT_ROOT/aeon_models/comfyui/insightface:/workspace/ComfyUI/models/insightface" \
    -v "$OUTPUT_DIR:/workspace/ComfyUI/output" \
    aeon_comfyui:latest

echo "ComfyUI is starting! It will be available at http://localhost:8188"
echo "Check logs with: docker logs -f aeon_comfyui"
