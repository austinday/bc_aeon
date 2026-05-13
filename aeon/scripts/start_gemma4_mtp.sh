#!/bin/bash
# =============================================================================
# Start llama.cpp server for Gemma-4-31B-it-abliterated Q8_0 GGUF
# Uses mainline llama.cpp with Native MTP (Multi-Token Prediction) Speculative Decoding.
# =============================================================================
set -e

CONTAINER_NAME='aeon_gemma4_mtp'
IMAGE_NAME='aeon_gemma4_mtp:latest'
PORT=8013
AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
MODELS_DIR="$AEON_HOME/models/gguf_models/Gemma-4"

TARGET_MODEL="gemma-4-31b-abliterated-Q8_0.gguf"
ASSISTANT_MODEL="gemma-4-31B-it-assistant.Q4_K_M.gguf"

if [ ! -f "${MODELS_DIR}/${TARGET_MODEL}" ]; then
    echo "[Gemma-4-MTP] ERROR: Target model ${TARGET_MODEL} not found in ${MODELS_DIR}"
    exit 1
fi

if [ ! -f "${MODELS_DIR}/${ASSISTANT_MODEL}" ]; then
    echo "[Gemma-4-MTP] ERROR: Assistant model ${ASSISTANT_MODEL} not found in ${MODELS_DIR}"
    exit 1
fi

PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)

echo "[Gemma-4-MTP] Cleaning up existing instances..."
docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true

echo "[Gemma-4-MTP] Starting Speculative server on GPU 0 (Port $PORT)..."
docker run -d \
  --name $CONTAINER_NAME \
  --gpus '"device=0"' \
  -p ${PORT}:8080 \
  -v "${MODELS_DIR}:/models:ro" \
  --shm-size=16g \
  --ulimit memlock=-1 \
  $IMAGE_NAME \
  -m "/models/${TARGET_MODEL}" \
  -md "/models/${ASSISTANT_MODEL}" \
  --spec-type mtp \
  --draft-block-size 6 \
  -ngl 99 \
  -ngld 99 \
  --flash-attn on \
  --batch-size 4096 \
  -c 16384 \
  -ctk f16 \
  -ctv f16 \
  --threads 8 \
  --host 0.0.0.0 \
  --port 8080

echo "[Gemma-4-MTP] Waiting for server to load model (this may take several minutes)..."
count=0
while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then break; fi
    sleep 5
    count=$((count+1))
    if [ $count -ge 120 ]; then
        echo "[Gemma-4-MTP] ERROR: Server did not become healthy within 10 minutes."
        docker logs $CONTAINER_NAME --tail 30
        exit 1
    fi
    if [ $((count % 6)) -eq 0 ]; then
        elapsed=$((count * 5))
        echo "[Gemma-4-MTP] Still loading... (${elapsed}s)"
    fi
done

echo "[Gemma-4-MTP] Server ready on port $PORT."
