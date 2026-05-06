#!/bin/bash
# =============================================================================
# Start llama.cpp cluster for Gemma-4-31B using native MTP (Self-Speculative)
# Deploys two asymmetric instances with Nginx content-length routing.
# =============================================================================
set -e

NODE0_NAME='aeon_gemma4_node0'
NODE1_NAME='aeon_gemma4_node1'
LB_NAME='aeon_gemma_lb'
IMAGE_NAME='aeon_llamacpp:latest'

# Exposed API port for the agent framework
MAIN_PORT=8008
# Internal cluster ports
NODE0_PORT=8011
NODE1_PORT=8012

AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
MODELS_DIR="$AEON_HOME/models/gguf_models/Gemma-4"

TARGET_MODEL="gemma-4-31b-abliterated-Q8_0.gguf"

if [ ! -f "${MODELS_DIR}/${TARGET_MODEL}" ]; then
    echo "[Gemma-4-Cluster] ERROR: Target model ${TARGET_MODEL} not found in ${MODELS_DIR}"
    exit 1
fi

# Find the draft model downloaded by setup_environment.sh
DRAFT_MODEL=$(cd "${MODELS_DIR}" 2>/dev/null && find . -maxdepth 1 -name "*Q4_K_M*.gguf" | head -1 | sed 's|^\./||')
if [ -n "$DRAFT_MODEL" ]; then
    echo "[Gemma-4-Cluster] Found draft model: ${DRAFT_MODEL}"
    DRAFT_ARGS="--model-draft /models/${DRAFT_MODEL}"
else
    echo "[Gemma-4-Cluster] WARNING: Draft model not found. Speculative decoding will run without an external draft model."
    DRAFT_ARGS=""
fi

PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)
CORES_PER_NODE=$((PHYSICAL_CORES / 2))

echo "[Gemma-4-Cluster] Cleaning up existing instances..."
docker rm -f $NODE0_NAME $NODE1_NAME $LB_NAME >/dev/null 2>&1 || true

# --- Function to launch a node ---
launch_node() {
    local NAME=$1
    local GPU=$2
    local PORT=$3
    local CTX=$4
    local PARALLEL=$5

    echo "[Gemma-4-Cluster] Starting $NAME on GPU $GPU (Port $PORT, Ctx: $CTX, Slots: $PARALLEL)..."
    docker run -d \
      --name $NAME \
      --gpus "\"device=$GPU\"" \
      -p ${PORT}:8001 \
      -v "${MODELS_DIR}:/models:ro" \
      --shm-size=16g \
      --ulimit memlock=-1 \
      $IMAGE_NAME \
      --model "/models/${TARGET_MODEL}" \
      $DRAFT_ARGS \
      --spec-draft-n-max 5 \
      --spec-draft-n-min 1 \
      --n-gpu-layers 99 \
      --parallel ${PARALLEL} \
      --ctx-size ${CTX} \
      --ctx-size-draft ${CTX} \
      --batch-size 4096 \
      --threads ${CORES_PER_NODE} \
      --flash-attn on \
      --cache-type-k q4_0 \
      --cache-type-v q4_0 \
      --defrag-thold 0.1 \
      --host 0.0.0.0 \
      --port 8001 \
      --metrics \
      --mlock \
      --no-mmap > /dev/null
}

# --- NODE 0: High-Capacity (Maxes out 96GB) ---
# ~31GB Model + ~60GB KV Cache
launch_node $NODE0_NAME 0 $NODE0_PORT 262144 1

# --- NODE 1: Low-Capacity (Restricts to ~48GB) ---
# ~31GB Model + ~15GB KV Cache (Leaves ~50GB free for other tools)
launch_node $NODE1_NAME 1 $NODE1_PORT 65536 1

# --- Nginx Intelligent Load Balancer ---
echo "[Gemma-4-Cluster] Configuring Nginx Content-Length Router..."
cat << 'EOF' > /tmp/aeon_gemma_lb.conf
events { worker_connections 1024; }
http {
    upstream cluster_all {
        least_conn;
        server 127.0.0.1:8011 max_fails=3 fail_timeout=10s;
        server 127.0.0.1:8012 max_fails=3 fail_timeout=10s;
    }
    upstream cluster_large {
        server 127.0.0.1:8011 max_fails=3 fail_timeout=10s;
    }
    server {
        listen 8008;
        location / {
            # Default: Route to either node based on current load
            set $backend http://cluster_all;
            
            # Context size routing: payload >= 200k bytes (~50k+ tokens) forced to GPU0 (256k ctx)
            if ($http_content_length ~ "^([2-9]\d{5}|\d{7,})$") {
                set $backend http://cluster_large;
            }
            
            proxy_pass $backend;
            proxy_read_timeout 1200s;
            proxy_set_header Host $host;
        }
    }
}
EOF

echo "[Gemma-4-Cluster] Starting Load Balancer on Port $MAIN_PORT..."
docker run -d \
    --name $LB_NAME \
    --network host \
    -v /tmp/aeon_gemma_lb.conf:/etc/nginx/nginx.conf:ro \
    nginx:alpine > /dev/null

echo "[Gemma-4-Cluster] Cluster deployed using native MTP! Target: ${TARGET_MODEL}"
echo "[Gemma-4-Cluster] Waiting for nodes to load into VRAM..."

# Wait for both nodes to become healthy
for PORT in $NODE0_PORT $NODE1_PORT; do
    count=0
    while true; do
        HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then 
            echo "[Gemma-4-Cluster] Node on port $PORT is READY."
            break
        fi
        sleep 2
        count=$((count+1))
        if [ $count -ge 60 ]; then
            echo "[Gemma-4-Cluster] ERROR: Node on port $PORT failed to start."
            exit 1
        fi
    done
done

echo "[Gemma-4-Cluster] Cluster is ONLINE and routing traffic via Port $MAIN_PORT!"
