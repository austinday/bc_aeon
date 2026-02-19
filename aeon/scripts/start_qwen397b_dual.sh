#!/bin/bash
# =============================================================================
# Start llama.cpp server for Qwen3.5-397B-A17B-MXFP4 GGUF on DUAL GPUs
# =============================================================================
set -e

CONTAINER_NAME='aeon_qwen397b_dual'
IMAGE_NAME='aeon_llamacpp:latest'
PORT=8003
MODELS_DIR="$HOME/bc_aeon/aeon_models/gguf_models/Qwen3.5-397B-A17B-MXFP4/MXFP4_MOE"

# Tunable parameters
N_GPU_LAYERS=${NGL:-48}          # 48/61 layers (~170GB) to fit inside 192GB combined VRAM safely
PARALLEL_SLOTS=${PARALLEL:-1}    # Single slot maximizes VRAM for model layers
CTX_SIZE=${CTX:-16384}           # 16k context to leave VRAM for layers
BATCH_SIZE=${BATCH:-4096}        # Prompt processing batch size

PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)

MODEL_FILE=$(ls -1 "${MODELS_DIR}"/*.gguf 2>/dev/null | sort | head -1 | xargs -r basename)
if [ -z "$MODEL_FILE" ]; then
    echo "[Qwen397B-Dual] ERROR: No .gguf files found in ${MODELS_DIR}"
    exit 1
fi

echo "[Qwen397B-Dual] Using model file: ${MODEL_FILE}"
echo "[Qwen397B-Dual] Checking for existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[Qwen397B-Dual] Container already running. Checking health..."
        count=0
        while true; do
            HC=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
            if [ "$HC" = "200" ]; then break; fi
            sleep 2
            count=$((count+1))
            if [ $count -ge 10 ]; then
                echo "[Qwen397B-Dual] Running but unhealthy (HTTP $HC). Restarting..."
                docker rm -f $CONTAINER_NAME >/dev/null 2>&1
                break
            fi
        done
        if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null)" = "200" ]; then
            echo "[Qwen397B-Dual] Already running and healthy on port $PORT."
            exit 0
        fi
    else
        echo "[Qwen397B-Dual] Removing stopped container..."
        docker rm -f $CONTAINER_NAME >/dev/null 2>&1
    fi
fi

echo "[Qwen397B-Dual] Starting llama.cpp server..."

docker run -d \
    --name $CONTAINER_NAME \
    --gpus '"device=0,1"' \
    -p ${PORT}:8001 \
    -v "${MODELS_DIR}:/models:ro" \
    --shm-size=1g \
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
    --metrics

echo "[Qwen397B-Dual] Waiting for server to load model (this may take several minutes)..."
count=0
while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then break; fi
    sleep 5
    count=$((count+1))
    if [ $count -ge 120 ]; then
        echo "[Qwen397B-Dual] ERROR: Server did not become healthy within 10 minutes."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    if [ $((count % 6)) -eq 0 ]; then
        elapsed=$((count * 5))
        echo "[Qwen397B-Dual] Still loading... (${elapsed}s)"
    fi
done

echo "[Qwen397B-Dual] Server ready on port $PORT."
