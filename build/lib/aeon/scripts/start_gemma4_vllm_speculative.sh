#!/bin/bash
# =============================================================================
# Start vLLM cluster for Gemma-4-31B using Native MTP
# Deploys two asymmetric instances with Nginx content-length routing.
# =============================================================================
set -e

NODE0_NAME='aeon_gemma4_vllm_node0'
NODE1_NAME='aeon_gemma4_vllm_node1'
LB_NAME='aeon_gemma_vllm_lb'
IMAGE_NAME='aeon_vllm:latest'

# Exposed API port for the agent framework
MAIN_PORT=8010
# Internal cluster ports
NODE0_PORT=8013
NODE1_PORT=8014

# We mount the HuggingFace cache directly into the container so vLLM can pull/load
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

TARGET_MODEL="google/gemma-4-31b-it"

echo "[Gemma-4-vLLM] Cleaning up existing instances..."
docker rm -f $NODE0_NAME $NODE1_NAME $LB_NAME >/dev/null 2>&1 || true

# --- Function to launch a vLLM node ---
launch_node() {
    local NAME=$1
    local GPU=$2
    local PORT=$3
    local CTX=$4
    local MEM_UTIL=$5

    echo "[Gemma-4-vLLM] Starting $NAME on GPU $GPU (Port $PORT, Ctx: $CTX, Mem Util: $MEM_UTIL)..."
    docker run -d \
      --name $NAME \
      --gpus "\"device=$GPU\"" \
      -p ${PORT}:8000 \
      -v "${HF_CACHE}:/root/.cache/huggingface" \
      --shm-size=16g \
      --ulimit memlock=-1 \
      --ipc=host \
      -e HF_TOKEN="${HF_TOKEN}" \
      $IMAGE_NAME \
      --model $TARGET_MODEL \
      --max-model-len $CTX \
      --gpu-memory-utilization $MEM_UTIL \
      --trust-remote-code \
      --host 0.0.0.0 \
      --port 8000 > /dev/null
}

# --- NODE 0: High-Capacity ---
# MEM_UTIL=0.95 will use ~90GB of the 96GB VRAM.
launch_node $NODE0_NAME 0 $NODE0_PORT 262144 0.95

# --- NODE 1: Low-Capacity (Restricts to exactly 50%) ---
# MEM_UTIL=0.50 will use ~48GB of the 96GB VRAM (Leaving exactly half free for tool calls)
launch_node $NODE1_NAME 1 $NODE1_PORT 32768 0.50

# --- Nginx Intelligent Load Balancer ---
echo "[Gemma-4-vLLM] Configuring Nginx Content-Length Router..."
cat << 'EOF' > /tmp/aeon_gemma_vllm_lb.conf
events { worker_connections 1024; }
http {
    upstream cluster_all {
        least_conn;
        server 127.0.0.1:8013 max_fails=3 fail_timeout=10s;
        server 127.0.0.1:8014 max_fails=3 fail_timeout=10s;
    }
    upstream cluster_large {
        server 127.0.0.1:8013 max_fails=3 fail_timeout=10s;
    }
    server {
        listen 8010;
        location / {
            set $backend http://cluster_all;
            
            # Context size routing: 6+ digit payload (~25k+ tokens) forced to GPU0
            if ($http_content_length ~ "^\d{6,}$") {
                set $backend http://cluster_large;
            }
            
            proxy_pass $backend;
            proxy_read_timeout 1200s;
            proxy_set_header Host $host;
        }
    }
}
EOF

echo "[Gemma-4-vLLM] Starting Load Balancer on Port $MAIN_PORT..."
docker run -d \
    --name $LB_NAME \
    --network host \
    -v /tmp/aeon_gemma_vllm_lb.conf:/etc/nginx/nginx.conf:ro \
    nginx:alpine > /dev/null

echo "[Gemma-4-vLLM] Cluster deployed using vLLM Native MTP! Target: ${TARGET_MODEL}"
echo "[Gemma-4-vLLM] Waiting for nodes to load into VRAM..."

for PORT in $NODE0_PORT $NODE1_PORT; do
    count=0
    while true; do
        HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then 
            echo "[Gemma-4-vLLM] Node on port $PORT is READY."
            break
        fi
        sleep 5
        count=$((count+1))
        if [ $count -ge 120 ]; then
            echo "[Gemma-4-vLLM] ERROR: Node on port $PORT failed to start."
            exit 1
        fi
    done
done

echo "[Gemma-4-vLLM] Cluster is ONLINE and routing traffic via Port $MAIN_PORT!"
