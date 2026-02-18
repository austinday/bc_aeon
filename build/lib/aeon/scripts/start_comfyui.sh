#!/bin/bash
# =============================================================================
# Start ComfyUI container for Aeon Agent (manual/debug use)
# =============================================================================
# The generate_image tool starts the container automatically. This script
# is for manual debugging or running ComfyUI with the browser UI.
#
# Usage:
#   bash start_comfyui.sh          # API-only mode (what the agent uses)
#   bash start_comfyui.sh --ui     # With browser UI at http://localhost:8188
# =============================================================================
set -e

CONTAINER_NAME='aeon_comfyui'
IMAGE_NAME='aeon_comfyui:latest'
PORT=8188
MODELS_DIR="$HOME/bc_aeon/aeon_models/comfyui_models"
OUTPUT_DIR="$HOME/bc_aeon/comfyui_output"

echo "[ComfyUI] Checking for existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[ComfyUI] Removing existing container..."
    docker rm -f $CONTAINER_NAME >/dev/null 2>&1
fi

mkdir -p "$MODELS_DIR"
mkdir -p "$OUTPUT_DIR"

# Determine flags
EXTRA_FLAGS=""
if [ "$1" = "--ui" ]; then
    echo "[ComfyUI] Starting WITH browser UI (http://localhost:$PORT)..."
    # Override CMD to remove --disable-auto-launch
    EXTRA_FLAGS="python3 main.py --listen 0.0.0.0 --port 8188"
else
    echo "[ComfyUI] Starting in API-only mode on port $PORT..."
fi

if [ -n "$EXTRA_FLAGS" ]; then
     docker run -d \
         --name $CONTAINER_NAME \
         -u "$(id -u):$(id -g)" \
         --gpus '"device=1"' \
         -p ${PORT}:8188 \
         -v "${MODELS_DIR}:/opt/ComfyUI/models/diffusion_models" \
         -v "${MODELS_DIR}/text_encoders:/opt/ComfyUI/models/text_encoders" \
         -v "${MODELS_DIR}/vae:/opt/ComfyUI/models/vae" \
         -v "${MODELS_DIR}/loras:/opt/ComfyUI/models/loras" \
         -v "${OUTPUT_DIR}:/opt/ComfyUI/output" \
         $IMAGE_NAME \
        $EXTRA_FLAGS
else
     docker run -d \
         --name $CONTAINER_NAME \
         -u "$(id -u):$(id -g)" \
         --gpus '"device=1"' \
         -p ${PORT}:8188 \
         -v "${MODELS_DIR}:/opt/ComfyUI/models/diffusion_models" \
         -v "${MODELS_DIR}/text_encoders:/opt/ComfyUI/models/text_encoders" \
         -v "${MODELS_DIR}/vae:/opt/ComfyUI/models/vae" \
         -v "${MODELS_DIR}/loras:/opt/ComfyUI/models/loras" \
         -v "${OUTPUT_DIR}:/opt/ComfyUI/output" \
         $IMAGE_NAME
        $IMAGE_NAME
fi

echo "[ComfyUI] Waiting for API..."
count=0
while ! curl -s http://localhost:${PORT}/system_stats >/dev/null 2>&1; do
    sleep 2
    count=$((count+1))
    if [ $count -ge 60 ]; then
        echo "[ComfyUI] ERROR: API did not respond within 120s"
        docker logs $CONTAINER_NAME
        exit 1
    fi
    echo -n "."
done

echo ""
echo "[ComfyUI] Ready on port $PORT."
