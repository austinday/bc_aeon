#!/bin/bash
# =============================================================================
# Start llama.cpp server for Qwen3.5-397B-A17B-UD-TQ1_0 GGUF on SINGLE GPU
# =============================================================================
set -e

CONTAINER_NAME='aeon_qwen397b_ud_tq1_single'
IMAGE_NAME='aeon_llamacpp:latest'
PORT=8006
MODELS_DIR="$HOME/bc_aeon/aeon_models/gguf_models/Qwen3.5-397B-A17B-MXFP4"

# Tunable parameters
N_GPU_LAYERS=${NGL:-61}          # Fits entirely in 96GB VRAM
PARALLEL_SLOTS=${PARALLEL:-1}    # Single slot maximizes VRAM for model layers
CTX_SIZE=${CTX:-131072}          # 128k context
BATCH_SIZE=${BATCH:-4096}        # Prompt processing batch size

PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)

MODEL_FILE=$(cd "${MODELS_DIR}" 2>/dev/null && find . -name "*.gguf" | grep -i "UD-TQ1_0" | sort | head -1 | sed 's|^\./||')
if [ -z "$MODEL_FILE" ]; then
    echo "[UD-TQ1-Single] ERROR: No .gguf files matching UD-TQ1_0 found in ${MODELS_DIR}"
    exit 1
fi

echo "[UD-TQ1-Single] Using model file: ${MODEL_FILE}"
echo "[UD-TQ1-Single] Checking for existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[UD-TQ1-Single] Container already running. Checking health..."
        count=0
        while true; do
            HC=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
            if [ "$HC" = "200" ]; then break; fi
            sleep 2
            count=$((count+1))
            if [ $count -ge 10 ]; then
                echo "[UD-TQ1-Single] Running but unhealthy (HTTP $HC). Restarting..."
                docker rm -f $CONTAINER_NAME >/dev/null 2>&1
                break
            fi
        done
        if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null)" = "200" ]; then
            echo "[UD-TQ1-Single] Already running and healthy on port $PORT."
            exit 0
        fi
    else
        echo "[UD-TQ1-Single] Removing stopped container..."
        docker rm -f $CONTAINER_NAME >/dev/null 2>&1
    fi
fi

echo "[UD-TQ1-Single] Starting llama.cpp server..."

docker run -d \
    --name $CONTAINER_NAME \
    --gpus '"device=0"' \
    -p ${PORT}:8001 \
    -v "${MODELS_DIR}:/models:ro" \
    --shm-size=16g \
    --ulimit memlock=-1 \
    $IMAGE_NAME \
    --model "/models/${MODEL_FILE}" \
    --n-gpu-layers ${N_GPU_LAYERS} \
    --parallel ${PARALLEL_SLOTS} \
    --ctx-size ${CTX_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --threads ${PHYSICAL_CORES} \
    --flash-attn on \
    --host 0.0.0.0 \
    --port 8001 \
    --metrics \
    --mlock \
    --no-mmap

echo "[UD-TQ1-Single] Waiting for server to load model (this may take several minutes)..."
count=0
while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then break; fi
    sleep 5
    count=$((count+1))
    if [ $count -ge 120 ]; then
        echo "[UD-TQ1-Single] ERROR: Server did not become healthy within 10 minutes."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    if [ $((count % 6)) -eq 0 ]; then
        elapsed=$((count * 5))
        echo "[UD-TQ1-Single] Still loading... (${elapsed}s)"
    fi
done

echo "[UD-TQ1-Single] Server ready on port $PORT."
