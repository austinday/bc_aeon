#!/bin/bash
# =============================================================================
# Start llama.cpp server for MiniMax-M2.5-Q5_K_M GGUF
# Fits entirely on dual GPU (no CPU spillover)
# GPU0 maxed (~92GB), GPU1 gets remainder (~70GB)
# =============================================================================
set -e

CONTAINER_NAME='aeon_minimax_m25'
IMAGE_NAME='aeon_llamacpp:latest'
PORT=8013
MODELS_DIR="$HOME/bc_aeon/aeon_models/gguf_models/MiniMax-M2.5"

# Tunable parameters
N_GPU_LAYERS=${NGL:-60}          # Most layers on GPU, slight CPU spillover for safety
PARALLEL_SLOTS=${PARALLEL:-1}    # Single slot maximizes VRAM for model layers
CTX_SIZE=${CTX:-131072}          # 128k context (q8 KV cache keeps VRAM manageable)
BATCH_SIZE=${BATCH:-4096}        # Prompt processing batch size
# tensor-split ratio: GPU0 maxed, GPU1 gets remainder
TENSOR_SPLIT=${TSPLIT:-92,70}

PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)

MODEL_FILE=$(cd "${MODELS_DIR}" 2>/dev/null && find . -name "*.gguf" | grep -i "Q5_K_M" | sort | head -1 | sed 's|^\./||')
if [ -z "$MODEL_FILE" ]; then
    echo "[MiniMax-M2.5-Q5KM] ERROR: No .gguf files matching Q5_K_M found in ${MODELS_DIR}"
    exit 1
fi

echo "[MiniMax-M2.5-Q5KM] Using model file: ${MODEL_FILE}"
echo "[MiniMax-M2.5-Q5KM] Checking for existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[MiniMax-M2.5-Q5KM] Container already running. Checking health..."
        count=0
        while true; do
            HC=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
            if [ "$HC" = "200" ]; then break; fi
            sleep 2
            count=$((count+1))
            if [ $count -ge 10 ]; then
                echo "[MiniMax-M2.5-Q5KM] Running but unhealthy (HTTP $HC). Restarting..."
                docker rm -f $CONTAINER_NAME >/dev/null 2>&1
                break
            fi
        done
        if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null)" = "200" ]; then
            echo "[MiniMax-M2.5-Q5KM] Already running and healthy on port $PORT."
            exit 0
        fi
    else
        echo "[MiniMax-M2.5-Q5KM] Removing stopped container..."
        docker rm -f $CONTAINER_NAME >/dev/null 2>&1
    fi
fi

echo "[MiniMax-M2.5-Q5KM] Starting llama.cpp server (all layers on GPU, GPU0 maxed)..."

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

echo "[MiniMax-M2.5-Q5KM] Waiting for server to load model (this may take several minutes)..."
count=0
while true; do
    # Check if container crashed
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[MiniMax-M2.5-Q5KM] ERROR: Container crashed during model loading!"
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
        echo "[MiniMax-M2.5-Q5KM] ERROR: Server did not become healthy within 10 minutes."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    if [ $((count % 6)) -eq 0 ]; then
        elapsed=$((count * 5))
        echo "[MiniMax-M2.5-Q5KM] Still loading... (${elapsed}s)"
    fi
done

echo "[MiniMax-M2.5-Q5KM] Server ready on port $PORT."
