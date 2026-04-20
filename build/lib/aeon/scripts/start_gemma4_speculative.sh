#!/bin/bash
# =============================================================================
# Start llama.cpp server for Gemma-4-31B with Speculative Decoding
# Target Model: gemma-4-31b-abliterated-Q8_0.gguf
# Draft Model: (Dynamic discovery for Q4_K_M)
# =============================================================================
set -e

CONTAINER_NAME='aeon_gemma4_speculative'
IMAGE_NAME='aeon_llamacpp:latest'
PORT=8008
AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
MODELS_DIR="$AEON_HOME/models/gguf_models/Gemma-4"

TARGET_MODEL="gemma-4-31b-abliterated-Q8_0.gguf"
# Find the draft model dynamically to handle any casing/naming variations
DRAFT_MODEL=$(cd "${MODELS_DIR}" 2>/dev/null && ls *Q4_K_M*.gguf 2>/dev/null | grep -i "heretic" | head -n 1 || true)

if [ -z "$DRAFT_MODEL" ]; then
    echo "[Gemma-4-Speculative] ERROR: Could not find draft model matching *Q4_K_M*heretic* in ${MODELS_DIR}"
    exit 1
fi

# Tunable parameters
N_GPU_LAYERS=${NGL:-99}          # Fit both models in VRAM
PARALLEL_SLOTS=${PARALLEL:-5}    # 5 slots for increased throughput
CTX_SIZE=${CTX:-262144}          # 256k context
BATCH_SIZE=${BATCH:-4096}
DRAFT_MAX=${DRAFT_MAX:-5}        # Number of tokens the draft model guesses at once

PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)

echo "[Gemma-4-Speculative] Checking for existing container..."
docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true

echo "[Gemma-4-Speculative] Starting llama.cpp server..."

docker run -d \
  --name $CONTAINER_NAME \
  --gpus '"device=0"' \
  -p ${PORT}:8001 \
  -v "${MODELS_DIR}:/models:ro" \
  --shm-size=16g \
  --ulimit memlock=-1 \
  $IMAGE_NAME \
  --model "/models/${TARGET_MODEL}" \
  --model-draft "/models/${DRAFT_MODEL}" \
  --draft-max ${DRAFT_MAX} \
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

echo "[Gemma-4-Speculative] Container launched!"
echo "[Gemma-4-Speculative] Target: ${TARGET_MODEL}"
echo "[Gemma-4-Speculative] Draft:  ${DRAFT_MODEL}"
echo "[Gemma-4-Speculative] Waiting for both models to load into VRAM..."

count=0
while true; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[Gemma-4-Speculative] ERROR: Container crashed! llama.cpp rejected the arguments or OOMed."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then break; fi
    sleep 2
    count=$((count+1))
    if [ $count -ge 60 ]; then
        echo "[Gemma-4-Speculative] ERROR: Server did not become healthy within 2 minutes."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    if [ $((count % 5)) -eq 0 ]; then
        elapsed=$((count * 2))
        echo "[Gemma-4-Speculative] Still loading weights... (${elapsed}s)"
    fi
done

echo "[Gemma-4-Speculative] Server ready and models loaded!"
