#!/bin/bash
# =============================================================================
# Start llama.cpp server for Qwen3.5-27B-Uncensored with Speculative Decoding
# Target Model: Qwen3.5-27B-Instruct-Uncensored (Q8_0)
# Draft Model: Huihui-Qwen3.5-2B-abliterated-i1 (Q4_K_M)
# =============================================================================
set -e

CONTAINER_NAME='aeon_qwen35_27b_speculative'
IMAGE_NAME='aeon_llamacpp:latest'
PORT=8009
MODELS_DIR="$HOME/bc_aeon/aeon_models/gguf_models/Qwen3.5-27B"

# Find the target model dynamically (case-insensitive to handle any HuggingFace naming)
TARGET_MODEL=$(cd "${MODELS_DIR}" 2>/dev/null && ls *[qQ]8_0*.gguf 2>/dev/null | sort | head -n 1 || true)
if [ -z "$TARGET_MODEL" ]; then
    echo "[Qwen3.5-27B-Spec] ERROR: Could not find target model matching *Q8_0*.gguf in ${MODELS_DIR}"
    exit 1
fi

# Find the draft model dynamically
DRAFT_MODEL=$(cd "${MODELS_DIR}" 2>/dev/null && ls *[qQ]4_[kK]_[mM]*.gguf 2>/dev/null | head -n 1 || true)
if [ -z "$DRAFT_MODEL" ]; then
    echo "[Qwen3.5-27B-Spec] ERROR: Could not find draft model matching *Q4_K_M*.gguf in ${MODELS_DIR}"
    exit 1
fi

# Tunable parameters
N_GPU_LAYERS=${NGL:-99}          # Fit both models in VRAM
PARALLEL_SLOTS=${PARALLEL:-1}    # Keep at 1 for max single-batch speed
CTX_SIZE=${CTX:-262144}          # 256k context
BATCH_SIZE=${BATCH:-4096}
DRAFT_MAX=${DRAFT_MAX:-5}        # Number of tokens the draft model guesses at once

PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)

echo "[Qwen3.5-27B-Spec] Checking for existing container..."
docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true

echo "[Qwen3.5-27B-Spec] Starting llama.cpp server..."

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

echo "[Qwen3.5-27B-Spec] Container launched!"
echo "[Qwen3.5-27B-Spec] Target: ${TARGET_MODEL}"
echo "[Qwen3.5-27B-Spec] Draft:  ${DRAFT_MODEL}"
echo "[Qwen3.5-27B-Spec] Waiting for both models to load into VRAM..."

count=0
while true; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[Qwen3.5-27B-Spec] ERROR: Container crashed! llama.cpp rejected the arguments or OOMed."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then break; fi
    sleep 2
    count=$((count+1))
    if [ $count -ge 60 ]; then
        echo "[Qwen3.5-27B-Spec] ERROR: Server did not become healthy within 2 minutes."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    if [ $((count % 5)) -eq 0 ]; then
        elapsed=$((count * 2))
        echo "[Qwen3.5-27B-Spec] Still loading weights... (${elapsed}s)"
    fi
done

echo "[Qwen3.5-27B-Spec] Server ready and models loaded!"
