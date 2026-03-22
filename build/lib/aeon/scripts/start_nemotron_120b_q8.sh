#!/bin/bash
# =============================================================================
# Start llama.cpp server for NVIDIA-Nemotron-3-Super-120B-A12B Q8_0
# GPU0 completely filled with model + context, remainder on GPU1
# =============================================================================
set -e

CONTAINER_NAME='aeon_nemotron_120b_q8'
IMAGE_NAME='aeon_llamacpp:latest'
PORT=8005
MODELS_DIR="$HOME/bc_aeon/aeon_models/gguf_models/NVIDIA-Nemotron-3-Super-120B-A12B"

# Tunable parameters
N_GPU_LAYERS=${NGL:-999}
PARALLEL_SLOTS=${PARALLEL:-1}
CTX_SIZE=${CTX:-262144}
BATCH_SIZE=${BATCH:-4096}
# tensor-split: Heavily bias towards GPU 0 (e.g., 75,25) to fill it up first
TENSOR_SPLIT=${TSPLIT:-75,25}
QUANT="Q8_0"

PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)

# Note: Nemotron-3-Super-120B Q8_0 is split into multiple files (00001-of-00004.gguf etc.)
# llama.cpp only needs the first file, it will automatically load the rest.
MODEL_FILE=$(cd "${MODELS_DIR}" 2>/dev/null && find . -name "*.gguf" | grep -i "${QUANT}" | grep -i "00001" | sort | head -1 | sed 's|^\./||')
if [ -z "$MODEL_FILE" ]; then
    echo "[Nemotron-120B] ERROR: No .gguf files matching ${QUANT} (part 1) found in ${MODELS_DIR}"
    exit 1
fi

echo "[Nemotron-120B] Using model file: ${MODEL_FILE}"
echo "[Nemotron-120B] Checking for existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[Nemotron-120B] Container already running. Checking health..."
        count=0
        while true; do
            HC=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
            if [ "$HC" = "200" ]; then break; fi
            sleep 2
            count=$((count+1))
            if [ $count -ge 10 ]; then
                echo "[Nemotron-120B] Running but unhealthy (HTTP $HC). Restarting..."
                docker rm -f $CONTAINER_NAME >/dev/null 2>&1
                break
            fi
        done
        if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null)" = "200" ]; then
            echo "[Nemotron-120B] Already running and healthy on port $PORT."
            exit 0
        fi
    else
        echo "[Nemotron-120B] Removing stopped container..."
        docker rm -f $CONTAINER_NAME >/dev/null 2>&1
    fi
fi

echo "[Nemotron-120B] Starting llama.cpp server..."

docker run -d \
    --name $CONTAINER_NAME \
    --gpus '"device=0,1"' \
    -p ${PORT}:8001 \
    -v "${MODELS_DIR}:/models:ro" \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --memory="200g" \
    --memory-swap="200g" \
    $IMAGE_NAME \
    --model "/models/${MODEL_FILE}" \
    --split-mode layer \
    --tensor-split ${TENSOR_SPLIT} \
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

echo "[Nemotron-120B] Waiting for server to load model (this may take several minutes)..."
count=0
while true; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[Nemotron-120B] ERROR: Container crashed during model loading!"
        echo "--- Container Logs ---"
        docker logs --tail 40 $CONTAINER_NAME
        echo "---"
        exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then break; fi
    sleep 5
    count=$((count+1))
    if [ $count -ge 120 ]; then
        echo "[Nemotron-120B] ERROR: Server did not become healthy within 10 minutes."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    if [ $((count % 6)) -eq 0 ]; then
        elapsed=$((count * 5))
        echo "[Nemotron-120B] Still loading... (${elapsed}s)"
    fi
done

echo "[Nemotron-120B] Server ready on port $PORT."
