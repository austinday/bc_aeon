#!/bin/bash
# =============================================================================
# Start llama.cpp cluster for Gemma-4-31B using native MTP (Multi-Token Prediction)
# Deploys two asymmetric instances with Nginx content-length routing.
# =============================================================================
set -e

NODE0_NAME='aeon_gemma4_mtp_node0'
NODE1_NAME='aeon_gemma4_mtp_node1'
LB_NAME='aeon_gemma_mtp_lb'
IMAGE_NAME='aeon_gemma4_mtp:latest'

# Exposed API port for the agent framework
MAIN_PORT=8013
# Internal cluster ports
NODE0_PORT=8014
NODE1_PORT=8015

AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
MODELS_DIR="$AEON_HOME/models/gguf_models/Gemma-4"

TARGET_MODEL="gemma-4-31b-abliterated-Q8_0.gguf"
ASSISTANT_MODEL="gemma-4-31B-it-assistant.Q4_K_M.gguf"

if [ ! -f "${MODELS_DIR}/${TARGET_MODEL}" ]; then
    echo "[Gemma-4-MTP-Cluster] ERROR: Target model ${TARGET_MODEL} not found in ${MODELS_DIR}"
    exit 1
fi

if [ ! -f "${MODELS_DIR}/${ASSISTANT_MODEL}" ]; then
    echo "[Gemma-4-MTP-Cluster] ERROR: Assistant model ${ASSISTANT_MODEL} not found in ${MODELS_DIR}"
    exit 1
fi

PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)
CORES_PER_NODE=$((PHYSICAL_CORES / 2))

echo "[Gemma-4-MTP-Cluster] Cleaning up existing instances..."
docker rm -f $NODE0_NAME $NODE1_NAME $LB_NAME >/dev/null 2>&1 || true

# --- Function to launch a node ---
launch_node() {
    local NAME=$1
    local GPU=$2
    local PORT=$3
    local CTX=$4

    echo "[Gemma-4-MTP-Cluster] Starting $NAME on GPU $GPU (Port $PORT, Ctx: $CTX)..."
    docker run -d \
      --name $NAME \
      --gpus "\"device=$GPU\"" \
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
      -c ${CTX} \
      -ctk q4_0 \
      -ctv q4_0 \
      --threads ${CORES_PER_NODE} \
      --host 0.0.0.0 \
      --port 8080 > /tmp/aeon_${NAME}.log 2>&1
}

# --- NODE 0: High-Capacity ---
# ~33GB Target Model + ~3GB Assistant Model + ~6.4GB KV Cache
launch_node $NODE0_NAME 0 $NODE0_PORT 262144

# --- NODE 1: Medium-Capacity
# ~33GB Target Model + ~3GB Assistant Model + ~2.2GB KV Cache
launch_node $NODE1_NAME 1 $NODE1_PORT 89984
# --- Python Context-Aware Load Balancer ---
echo "[Gemma-4-MTP-Cluster] Starting Python Context-Aware Load Balancer on Port $MAIN_PORT..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
docker run -d \
    --name $LB_NAME \
    --network host \
    -v "${SCRIPT_DIR}/gemma_lb.py:/app/gemma_lb.py" \
    -w /app \
    python:3.11-slim \
    sh -c "pip install fastapi uvicorn httpx && python gemma_lb.py" > /dev/null

echo "[Gemma-4-MTP-Cluster] Waiting for nodes to load into VRAM (this may take several minutes)..."

# Wait for both nodes to become healthy
for PORT in $NODE0_PORT $NODE1_PORT; do
    count=0
    while true; do
        HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then 
            echo "[Gemma-4-MTP-Cluster] Node on port $PORT is READY."
            break
        fi
        sleep 5
        count=$((count+1))
        if [ $count -ge 120 ]; then
            echo "[Gemma-4-MTP-Cluster] ERROR: Node on port $PORT failed to start."
            exit 1
        fi
        if [ $((count % 6)) -eq 0 ]; then
            elapsed=$((count * 5))
            echo "[Gemma-4-MTP-Cluster] Still loading... (${elapsed}s)"
        fi
    done
done

echo "[Gemma-4-MTP-Cluster] Cluster is ONLINE and routing traffic via Port $MAIN_PORT!"
