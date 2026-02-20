#!/bin/bash
# =============================================================================
# Start llama.cpp server for Qwen3.5-397B-A17B IQ4_XS (or Q5_K_S fallback)
#
# GPU MODES (set via QWEN_MODE env var):
#   max    - Both GPUs full, ~87% on GPU, fastest (~5-8 t/s)
#   medium - GPU0 full + GPU1 leaves 24GB free (~75% on GPU, ~4-6 t/s)
#   light  - GPU0 full + GPU1 leaves 48GB free (~64% on GPU, ~3-5 t/s)
#
# Usage: QWEN_MODE=max bash start_qwen397b_q6k.sh
#        QWEN_MODE=medium bash start_qwen397b_q6k.sh  (default)
# =============================================================================
set -e

CONTAINER_NAME='aeon_qwen397b'
IMAGE_NAME='aeon_llamacpp:latest'
PORT=8005
MODELS_DIR="$HOME/bc_aeon/aeon_models/gguf_models/Qwen3.5-397B-A17B"

# GPU mode selection
MODE=${QWEN_MODE:-medium}
case "$MODE" in
    max)
        # Both GPUs full — GPU1 gets slightly more (GPU0 carries 4GB embed overhead)
        # Budget: GPU0 ~82GB layers+KV+embed, GPU1 ~85GB layers+KV
        # KV cache quantized to q8_0, so 128k ctx ≈ same VRAM as old 64k f16
        N_GPU_LAYERS=${NGL:-55}
        TENSOR_SPLIT=${TSPLIT:-48,52}
        CTX_SIZE=${CTX:-131072}
        echo "[Qwen397B] Mode: MAX (both GPUs full, 128k ctx, ~4-6 t/s)"
        ;;
    medium)
        # GPU0 full, GPU1 leaves ~24GB free for light tools
        # Budget: GPU0 ~82GB, GPU1 ~67GB (29GB free)
        N_GPU_LAYERS=${NGL:-48}
        TENSOR_SPLIT=${TSPLIT:-54,46}
        CTX_SIZE=${CTX:-131072}
        echo "[Qwen397B] Mode: MEDIUM (GPU1 leaves ~28GB free, 128k ctx, ~3-5 t/s)"
        ;;
    light)
        # GPU0 full, GPU1 leaves ~48GB free for ComfyUI/heavy tools
        # Budget: GPU0 ~84GB, GPU1 ~45GB (51GB free)
        N_GPU_LAYERS=${NGL:-40}
        TENSOR_SPLIT=${TSPLIT:-64,36}
        CTX_SIZE=${CTX:-131072}
        echo "[Qwen397B] Mode: LIGHT (GPU1 leaves ~48GB free, 128k ctx, ~2.5-4 t/s)"
        ;;
    *)
        echo "[Qwen397B] ERROR: Unknown mode '$MODE'. Use max, medium, or light."
        exit 1
        ;;
esac

PARALLEL_SLOTS=${PARALLEL:-1}
BATCH_SIZE=${BATCH:-4096}
PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)

# Try IQ4_XS first (faster, fits more on GPU), fall back to Q5_K_S
MODEL_FILE=$(cd "${MODELS_DIR}" 2>/dev/null && find . -name "*.gguf" | grep -i "IQ4_XS" | sort | head -1 | sed 's|^\./||')
if [ -z "$MODEL_FILE" ]; then
    MODEL_FILE=$(cd "${MODELS_DIR}" 2>/dev/null && find . -name "*.gguf" | grep -i "Q5_K_S" | sort | head -1 | sed 's|^\./||')
fi
if [ -z "$MODEL_FILE" ]; then
    echo "[Qwen397B] ERROR: No .gguf files matching IQ4_XS or Q5_K_S found in ${MODELS_DIR}"
    exit 1
fi

echo "[Qwen397B] Using model file: ${MODEL_FILE}"
echo "[Qwen397B] NGL=${N_GPU_LAYERS}, split=${TENSOR_SPLIT}, ctx=${CTX_SIZE}"
echo "[Qwen397B] Checking for existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
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

echo "[Qwen397B] Starting llama.cpp server (mode: $MODE)..."

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

echo "[Qwen397B] Waiting for server to load model (this may take several minutes)..."
count=0
while true; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[Qwen397B] ERROR: Container crashed during model loading!"
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
        echo "[Qwen397B] ERROR: Server did not become healthy within 10 minutes."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    if [ $((count % 6)) -eq 0 ]; then
        elapsed=$((count * 5))
        echo "[Qwen397B] Still loading... (${elapsed}s)"
    fi
done

echo "[Qwen397B] Server ready on port $PORT (mode: $MODE)."
