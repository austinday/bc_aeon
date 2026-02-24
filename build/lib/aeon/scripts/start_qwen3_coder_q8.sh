#!/bin/bash
# =============================================================================
# Start llama.cpp server for Qwen3-Coder-Next-Abliterated Q8_0 GGUF on SINGLE GPU
# =============================================================================
set -e

CONTAINER_NAME='aeon_qwen3_coder_q8'
IMAGE_NAME='aeon_llamacpp:latest'
PORT=8007
MODELS_DIR="$HOME/bc_aeon/aeon_models/gguf_models/Qwen3-Coder-Next-Abliterated"

# Tunable parameters
N_GPU_LAYERS=${NGL:-99}          # Fits entirely in VRAM
PARALLEL_SLOTS=${PARALLEL:-1}    # Single slot maximizes VRAM for model layers
CTX_SIZE=${CTX:-262144}          # 256k context
BATCH_SIZE=${BATCH:-4096}        # Prompt processing batch size

PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)

# Need the first file in the sequence for split models
MODEL_FILE=$(cd "${MODELS_DIR}" 2>/dev/null && find . -name "*.gguf" | grep -i "Q8_0" | sort | head -1 | sed 's|^\./||')
if [ -z "$MODEL_FILE" ]; then
    echo "[Qwen3-Coder-Q8] ERROR: No .gguf files matching Q8_0 found in ${MODELS_DIR}"
    exit 1
fi

echo "[Qwen3-Coder-Q8] Using model file: ${MODEL_FILE}"
echo "[Qwen3-Coder-Q8] Checking for existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[Qwen3-Coder-Q8] Container already running. Checking health..."
        count=0
        while true; do
            HC=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
            if [ "$HC" = "200" ]; then break; fi
            sleep 2
            count=$((count+1))
            if [ $count -ge 10 ]; then
                echo "[Qwen3-Coder-Q8] Running but unhealthy (HTTP $HC). Restarting..."
                docker rm -f $CONTAINER_NAME >/dev/null 2>&1
                break
            fi
        done
        if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null)" = "200" ]; then
            echo "[Qwen3-Coder-Q8] Already running and healthy on port $PORT."
            exit 0
        fi
    else
        echo "[Qwen3-Coder-Q8] Removing stopped container..."
        docker rm -f $CONTAINER_NAME >/dev/null 2>&1
    fi
fi

echo "[Qwen3-Coder-Q8] Starting llama.cpp server..."

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
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --host 0.0.0.0 \
    --port 8001 \
    --metrics \
    --mlock \
    --no-mmap

echo "[Qwen3-Coder-Q8] Waiting for server to load model (this may take several minutes)..."
count=0
while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then break; fi
    sleep 5
    count=$((count+1))
    if [ $count -ge 120 ]; then
        echo "[Qwen3-Coder-Q8] ERROR: Server did not become healthy within 10 minutes."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    if [ $((count % 6)) -eq 0 ]; then
        elapsed=$((count * 5))
        echo "[Qwen3-Coder-Q8] Still loading... (${elapsed}s)"
    fi
done

echo "[Qwen3-Coder-Q8] Server ready on port $PORT."
