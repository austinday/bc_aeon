#!/bin/bash
# =============================================================================
# Start llama.cpp server for MiniMax-M2.5-Q5_K_S GGUF (LIGHT mode)
# Offloads all layers to GPU to prevent system RAM swapping. Fits on ~1.5 GPUs.
# GPU0 maxed, GPU1 partially loaded.
# =============================================================================
set -e

CONTAINER_NAME='aeon_minimax_m25_q5_light'
IMAGE_NAME='aeon_llamacpp:latest'
PORT=8014
MODELS_DIR="$HOME/bc_aeon/aeon_models/gguf_models/MiniMax-M2.5"

# Tunable parameters
N_GPU_LAYERS=${NGL:-999}         # Offload all to GPU
PARALLEL_SLOTS=${PARALLEL:-1}    # Single slot
CTX_SIZE=${CTX:-131072}          # 128k context
BATCH_SIZE=${BATCH:-4096}        # Prompt processing batch size
# tensor-split: 66,34 roughly puts 2/3 on GPU0 and 1/3 on GPU1.
TENSOR_SPLIT=${TSPLIT:-66,34}
QUANT="Q5_K_S"

PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)

MODEL_FILE=$(cd "${MODELS_DIR}" 2>/dev/null && find . -name "*.gguf" | grep -i "${QUANT}" | sort | head -1 | sed 's|^\./||')
if [ -z "$MODEL_FILE" ]; then
    echo "[MiniMax-M2.5-${QUANT}-Light] ERROR: No .gguf files matching ${QUANT} found in ${MODELS_DIR}"
    exit 1
fi

echo "[MiniMax-M2.5-${QUANT}-Light] Using model file: ${MODEL_FILE}"
echo "[MiniMax-M2.5-${QUANT}-Light] Checking for existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[MiniMax-M2.5-${QUANT}-Light] Container already running. Checking health..."
        count=0
        while true; do
            HC=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
            if [ "$HC" = "200" ]; then break; fi
            sleep 2
            count=$((count+1))
            if [ $count -ge 10 ]; then
                echo "[MiniMax-M2.5-${QUANT}-Light] Running but unhealthy (HTTP $HC). Restarting..."
                docker rm -f $CONTAINER_NAME >/dev/null 2>&1
                break
            fi
        done
        if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null)" = "200" ]; then
            echo "[MiniMax-M2.5-${QUANT}-Light] Already running and healthy on port $PORT."
            exit 0
        fi
    else
        echo "[MiniMax-M2.5-${QUANT}-Light] Removing stopped container..."
        docker rm -f $CONTAINER_NAME >/dev/null 2>&1
    fi
fi

echo "[MiniMax-M2.5-${QUANT}-Light] Starting llama.cpp server..."

docker run -d \
    --name $CONTAINER_NAME \
    --gpus '"device=0,1"' \
    -p ${PORT}:8001 \
    -v "${MODELS_DIR}:/models:ro" \
    --shm-size=16g \
    --ulimit memlock=-1 \
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

echo "[MiniMax-M2.5-${QUANT}-Light] Waiting for server to load model (this may take several minutes)..."
count=0
while true; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[MiniMax-M2.5-${QUANT}-Light] ERROR: Container crashed during model loading!"
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
        echo "[MiniMax-M2.5-${QUANT}-Light] ERROR: Server did not become healthy within 10 minutes."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    if [ $((count % 6)) -eq 0 ]; then
        elapsed=$((count * 5))
        echo "[MiniMax-M2.5-${QUANT}-Light] Still loading... (${elapsed}s)"
    fi
done

echo "[MiniMax-M2.5-${QUANT}-Light] Server ready on port $PORT."
