#!/bin/bash
# =============================================================================
# Start llama.cpp server for Qwen3.5-397B-A17B-MXFP4 GGUF
# =============================================================================
# Uses GPU0 only with system RAM offloading for layers that don't fit.
# Continuous batching + parallel slots for multi-agent sharing.
#
# Usage:
#   bash start_qwen397b.sh              # Default settings
#   NGL=40 bash start_qwen397b.sh       # Override GPU layers
#   PARALLEL=8 bash start_qwen397b.sh   # Override parallel slots
# =============================================================================
set -e

CONTAINER_NAME='aeon_qwen397b'
IMAGE_NAME='aeon_llamacpp:latest'
PORT=8001
MODELS_DIR="$HOME/bc_aeon/aeon_models/gguf_models/Qwen3.5-397B-A17B-MXFP4/MXFP4_MOE"

# Tunable parameters (override via environment variables)
N_GPU_LAYERS=${NGL:-0}           # 0 = let llama.cpp auto-fit to available VRAM. Override with NGL=N.
PARALLEL_SLOTS=${PARALLEL:-1}    # Single slot maximizes VRAM for model layers
CTX_SIZE=${CTX:-16384}           # Reduced context to free VRAM (override with CTX=32768)
BATCH_SIZE=${BATCH:-2048}        # Prompt processing batch size

# Auto-detect the first shard (the one llama.cpp needs to find all splits)
MODEL_FILE=$(ls -1 "${MODELS_DIR}"/*.gguf 2>/dev/null | sort | head -1 | xargs -r basename)
if [ -z "$MODEL_FILE" ]; then
    echo "[Qwen397B] ERROR: No .gguf files found in ${MODELS_DIR}"
    echo "[Qwen397B] Run setup_environment.sh to download the model."
    exit 1
fi

echo "[Qwen397B] Using model file: ${MODEL_FILE}"
echo "[Qwen397B] Checking for existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # If already running, just verify health
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[Qwen397B] Container already running. Checking health..."
        count=0
        while true; do
            HC=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
            if [ "$HC" = "200" ]; then break; fi
            sleep 2
            count=$((count+1))
            if [ $count -ge 10 ]; then
                echo "[Qwen397B] Running but unhealthy (HTTP $HC). Restarting..."
                docker rm -f $CONTAINER_NAME >/dev/null 2>&1
                break
            fi
        done
        if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null)" = "200" ]; then
            echo "[Qwen397B] Already running and healthy on port $PORT."
            exit 0
        fi
    else
        echo "[Qwen397B] Removing stopped container..."
        docker rm -f $CONTAINER_NAME >/dev/null 2>&1
    fi
fi

echo "[Qwen397B] Starting llama.cpp server..."
echo "[Qwen397B]   GPU layers: ${N_GPU_LAYERS}"
echo "[Qwen397B]   Parallel slots: ${PARALLEL_SLOTS}"
echo "[Qwen397B]   Context size: ${CTX_SIZE}"
echo "[Qwen397B]   Batch size: ${BATCH_SIZE}"
echo "[Qwen397B]   Port: ${PORT}"

docker run -d \
    --name $CONTAINER_NAME \
    --gpus '"device=0"' \
    -p ${PORT}:8001 \
    -v "${MODELS_DIR}:/models:ro" \
    --shm-size=1g \
    --ulimit memlock=-1 \
    $IMAGE_NAME \
    --model "/models/${MODEL_FILE}" \
    --no-mmap \
    $( [ "${N_GPU_LAYERS}" -gt 0 ] && echo "--n-gpu-layers ${N_GPU_LAYERS}" ) \
    --parallel ${PARALLEL_SLOTS} \
    --ctx-size ${CTX_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --flash-attn on \
    --host 0.0.0.0 \
    --port 8001 \
    --metrics

echo "[Qwen397B] Waiting for server to load model (this may take several minutes)..."
count=0
while true; do
    # Check HTTP status code, not just connectivity
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        break
    fi
    sleep 5
    count=$((count+1))
    if [ $count -ge 120 ]; then
        echo "[Qwen397B] ERROR: Server did not become healthy within 10 minutes."
        echo "[Qwen397B] Last health check returned HTTP $HTTP_CODE"
        echo "[Qwen397B] Container logs:"
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    # Print progress every 30 seconds
    if [ $((count % 6)) -eq 0 ]; then
        elapsed=$((count * 5))
        echo "[Qwen397B] Still loading... (${elapsed}s, last HTTP status: $HTTP_CODE)"
    fi
done

echo "[Qwen397B] Server ready on port $PORT."
