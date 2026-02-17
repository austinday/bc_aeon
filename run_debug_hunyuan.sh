#!/bin/bash
set -e
# =============================================================================
# Run HunyuanImage debug inspector inside the ComfyUI container
# Outputs to ~/hunyuanDebug.txt
# =============================================================================

CONTAINER_NAME='aeon_comfyui'
IMAGE_NAME='aeon_comfyui:latest'
MODELS_DIR="$HOME/bc_aeon/aeon_models/comfyui_models"
VAE_DIR="$MODELS_DIR/vae"
OUTPUT_DIR="$HOME/bc_aeon/comfyui_output"
DEBUG_SCRIPT="$HOME/bc_aeon/debug_hunyuan.py"
DEBUG_OUTPUT="$HOME/hunyuanDebug.txt"

echo "[DEBUG] Starting HunyuanImage debug inspection..."
echo "[DEBUG] Models dir: $MODELS_DIR"
echo "[DEBUG] Debug script: $DEBUG_SCRIPT"

# Make sure output dir exists
mkdir -p "$OUTPUT_DIR"

# Stop any existing container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[DEBUG] Stopping existing container..."
    docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true
fi

# Start the container with all the same mounts as the real agent uses
# PLUS mount the debug script and an output volume
echo "[DEBUG] Starting container with full model mounts..."
docker run -d \
    --name $CONTAINER_NAME \
    --gpus 'device=1' \
    -p 8188:8188 \
    -v "${MODELS_DIR}:/opt/ComfyUI/models/checkpoints" \
    -v "${MODELS_DIR}:/opt/ComfyUI/models/diffusion_models" \
    -v "${MODELS_DIR}:/opt/ComfyUI/models/unet" \
    -v "${MODELS_DIR}/clip:/opt/ComfyUI/models/clip" \
    -v "${MODELS_DIR}/text_encoders:/opt/ComfyUI/models/text_encoders" \
    -v "${MODELS_DIR}/llm:/opt/ComfyUI/models/llm" \
    -v "${VAE_DIR}:/opt/ComfyUI/models/vae" \
    -v "${OUTPUT_DIR}:/opt/ComfyUI/output" \
    $IMAGE_NAME

# Wait for ComfyUI API to be ready (some inspections use it)
echo "[DEBUG] Waiting for ComfyUI API..."
count=0
while ! curl -s http://localhost:8188/system_stats >/dev/null 2>&1; do
    sleep 2
    count=$((count+1))
    if [ $count -ge 60 ]; then
        echo "[DEBUG] WARNING: ComfyUI API not available after 120s"
        echo "[DEBUG] Container logs:"
        docker logs --tail 30 $CONTAINER_NAME
        echo "[DEBUG] Continuing anyway (API-dependent sections will fail gracefully)..."
        break
    fi
    echo -n "."
done
echo ""

# Copy and run the debug script inside the container
echo "[DEBUG] Copying debug script into container..."
docker cp "$DEBUG_SCRIPT" $CONTAINER_NAME:/tmp/debug_hunyuan.py

# Create output mount point inside container
docker exec $CONTAINER_NAME mkdir -p /output

echo "[DEBUG] Running debug inspection (this may take a few minutes)..."
docker exec $CONTAINER_NAME python3 /tmp/debug_hunyuan.py

# Copy the output from the container
echo "[DEBUG] Extracting debug report..."
docker cp $CONTAINER_NAME:/output/hunyuanDebug.txt "$DEBUG_OUTPUT"

echo ""
echo "============================================================"
echo "  Debug report written to: $DEBUG_OUTPUT"
echo "  Size: $(wc -c < "$DEBUG_OUTPUT") bytes"
echo "  Lines: $(wc -l < "$DEBUG_OUTPUT")"
echo "============================================================"
echo ""
echo "Run: less $DEBUG_OUTPUT"
echo "Or:  grep -i 'img_in' $DEBUG_OUTPUT"

# Leave container running in case you want to exec into it
echo ""
echo "[DEBUG] Container '$CONTAINER_NAME' left running for further inspection."
echo "  docker exec -it $CONTAINER_NAME bash"
echo "  docker rm -f $CONTAINER_NAME   # when done"
