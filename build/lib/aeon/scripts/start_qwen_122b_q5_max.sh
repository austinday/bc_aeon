#!/bin/bash
# =============================================================================
# Start llama.cpp server for Qwen3.5-122B-A10B Q5_K_S
# Fits completely on GPU0 with ~12GB left for context
# =============================================================================
set -e

CONTAINER_NAME='aeon_qwen122b_max'
IMAGE_NAME='aeon_llamacpp:latest'
PORT=8006
MODELS_DIR="$HOME/bc_aeon/aeon_models/gguf_models/Qwen3.5-122B-A10B"

# Tunable parameters
N_GPU_LAYERS=${NGL:-999}
PARALLEL_SLOTS=${PARALLEL:-1}
CTX_SIZE=${CTX:-262144}
BATCH_SIZE=${BATCH:-4096}
QUANT="Q5_K_S"

PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)

MODEL_FILE=$(cd "${MODELS_DIR}" 2>/dev/null && find . -name "*.gguf" | grep -i "${QUANT}" | sort | head -1 | sed 's|^\./||')
if [ -z "$MODEL_FILE" ]; then
    echo "[Qwen122B-Max] ERROR: No .gguf files matching ${QUANT} found in ${MODELS_DIR}"
    exit 1
fi

echo "[Qwen122B-Max] Using model file: ${MODEL_FILE}"
echo "[Qwen122B-Max] Checking for existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[Qwen122B-Max] Container already running. Checking health..."
        count=0
        while true; do
            HC=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
            if [ "$HC" = "200" ]; then break; fi
            sleep 2
            count=$((count+1))
            if [ $count -ge 10 ]; then
                echo "[Qwen122B-Max] Running but unhealthy (HTTP $HC). Restarting..."
                docker rm -f $CONTAINER_NAME >/dev/null 2>&1
                break
            fi
        done
        if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null)" = "200" ]; then
            echo "[Qwen122B-Max] Already running and healthy on port $PORT."
            exit 0
        fi
    else
        echo "[Qwen122B-Max] Removing stopped container..."
        docker rm -f $CONTAINER_NAME >/dev/null 2>&1
    fi
fi

echo "[Qwen122B-Max] Starting llama.cpp server..."

docker run -d \
    --name $CONTAINER_NAME \
    --gpus '"device=0"' \
    -p ${PORT}:8001 \
    -v "${MODELS_DIR}:/models:ro" \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --memory="200g" \
    --memory-swap="200g" \
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

echo "[Qwen122B-Max] Waiting for server to load model (this may take several minutes)..."
count=0
while true; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[Qwen122B-Max] ERROR: Container crashed during model loading!"
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
        echo "[Qwen122B-Max] ERROR: Server did not become healthy within 10 minutes."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    if [ $((count % 6)) -eq 0 ]; then
        elapsed=$((count * 5))
        echo "[Qwen122B-Max] Still loading... (${elapsed}s)"
    fi
done

echo "[Qwen122B-Max] Server ready on port $PORT."
