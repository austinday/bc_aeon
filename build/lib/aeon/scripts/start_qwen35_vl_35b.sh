#!/bin/bash
# =============================================================================
# Start llama.cpp server for Qwen3.5-35B-A3B-UD-Q8_K_XL
# Used by the analyze_image tool for on-demand image understanding.
# Runs on GPU1 by default.
# Uses aeon_llamacpp:latest (same image as text model serving).
# =============================================================================
set -e

CONTAINER_NAME='aeon_qwen35_vl'
IMAGE_NAME='aeon_llamacpp:latest'
PORT=8020
GPU_ID=${VISION_GPU:-1}
MODELS_DIR="$HOME/bc_aeon/aeon_models/vl_models/Qwen3.5-35B-A3B-GGUF"

echo "[Qwen3.5-VL] Checking for existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[Qwen3.5-VL] Container already running. Checking health..."
        count=0
        while true; do
            HC=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
            if [ "$HC" = "200" ]; then break; fi
            sleep 2
            count=$((count+1))
            if [ $count -ge 15 ]; then
                echo "[Qwen3.5-VL] Running but unhealthy (HTTP $HC). Restarting..."
                docker rm -f $CONTAINER_NAME >/dev/null 2>&1
                break
            fi
        done
        if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null)" = "200" ]; then
            echo "[Qwen3.5-VL] Already running and healthy on port $PORT."
            exit 0
        fi
    else
        echo "[Qwen3.5-VL] Removing stopped container..."
        docker rm -f $CONTAINER_NAME >/dev/null 2>&1
    fi
fi

# Verify model files exist
MODEL_FILE="$MODELS_DIR/Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf"
MMPROJ_FILE=$(find "$MODELS_DIR" -maxdepth 1 -name "mmproj*.gguf" | head -n 1)

if [ ! -f "$MODEL_FILE" ]; then
    echo "[Qwen3.5-VL] ERROR: Model file not found at $MODEL_FILE"
    echo "Please run setup_environment.sh first."
    exit 1
fi

echo "[Qwen3.5-VL] Starting llama.cpp server on GPU ${GPU_ID} (port ${PORT})..."

# Conditionally use mmproj if it exists
MMPROJ_FLAG=""
if [ -n "$MMPROJ_FILE" ] && [ -f "$MMPROJ_FILE" ]; then
    MMPROJ_BASENAME=$(basename "$MMPROJ_FILE")
    MMPROJ_FLAG="--mmproj /models/$MMPROJ_BASENAME"
fi

docker run -d \
    --name $CONTAINER_NAME \
    --gpus "device=${GPU_ID}" \
    -v "${MODELS_DIR}:/models:ro" \
    -p ${PORT}:8080 \
    --ipc=host \
    $IMAGE_NAME \
    --model /models/Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf \
    $MMPROJ_FLAG \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 999 \
    -c 8192 \
    --flash-attn on

echo "[Qwen3.5-VL] Waiting for server to load model..."
count=0
while true; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[Qwen3.5-VL] ERROR: Container crashed during model loading!"
        echo "--- Container Logs ---"
        docker logs --tail 50 $CONTAINER_NAME
        echo "---"
        exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then break; fi
    sleep 3
    count=$((count+1))
    if [ $count -ge 60 ]; then
        echo "[Qwen3.5-VL] ERROR: Server did not become healthy within 3 minutes."
        docker logs --tail 50 $CONTAINER_NAME
        exit 1
    fi
    if [ $((count % 5)) -eq 0 ]; then
        elapsed=$((count * 3))
        echo "[Qwen3.5-VL] Still loading... (${elapsed}s)"
    fi
done

echo "[Qwen3.5-VL] Server ready on port $PORT (GPU ${GPU_ID})."
