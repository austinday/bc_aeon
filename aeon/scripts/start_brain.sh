#!/bin/bash
set -e

echo "=================================================="
echo "    STARTING AEON LOCAL BRAIN (SINGLE NODE)       "
echo "=================================================="

# Host Directory for Persistence (Unified Model Lake)
AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
HOST_OLLAMA_DIR="$AEON_HOME/models/ollama_home"

if [ ! -d "$HOST_OLLAMA_DIR" ]; then
    echo "Model directory not found at $HOST_OLLAMA_DIR. Creating it..."
    mkdir -p "$HOST_OLLAMA_DIR"
fi

# --- 1. CLEANUP ---
echo "[1/2] Checking for existing brain nodes..."

kill_container() {
  local NAME=$1
  if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "   >> Removing old container: $NAME"
    docker rm -f $NAME >/dev/null
  fi
}

kill_container "aeon_strong_node"
kill_container "aeon_weak_node"
kill_container "aeon_brain_node"

# Check port 8000 usage
CID=$(docker ps -q --filter "publish=8000")
if [ ! -z "$CID" ]; then
  echo "   >> Found container $CID holding port 8000. Killing..."
  docker rm -f $CID >/dev/null
fi

# --- 2. START BRAIN NODE (GPU 1 - Port 8000) ---
echo "[2/2] Launching Brain Node (GPU 1 -> :8000)..."

# The brain hosts the small "utility" model that handles the agent's high-frequency
# support tasks (skill routing, JSON repair, summarization, log/memory compression,
# interruption analysis). It runs on GPU 1 so those tasks stop contending with the
# strong LLM (+ sub-agents) on GPU 0. A small model + modest ctx keeps GPU 1 mostly
# free for the image/video/vision tools that also live there.
#   - device=1: pin to GPU 1.
#   - OLLAMA_NUM_PARALLEL=2: utility calls can overlap a little without thrashing.
#   - OLLAMA_NUM_CTX=65536: enough for long summaries/compressions; light KV on a 3B.
UTILITY_MODEL="${AEON_UTILITY_MODEL:-huihui_ai/qwen2.5-abliterate:3b}"

docker run -d \
    --name aeon_brain_node \
    --gpus '"device=1"' \
    -v "$HOST_OLLAMA_DIR:/root/.ollama" \
    -e OLLAMA_KEEP_ALIVE=-1 \
    -e OLLAMA_MAX_LOADED_MODELS=2 \
    -e OLLAMA_NUM_PARALLEL=2 \
    -e OLLAMA_NUM_CTX=65536 \
    -p 8000:11434 \
    ollama/ollama:latest

# Wait for the Ollama API, then ensure the utility model is present (one-time pull).
echo "   >> Waiting for Ollama API on :8000..."
for i in $(seq 1 30); do
    curl -s -o /dev/null http://localhost:8000/api/tags && break
    sleep 2
done
if ! curl -s http://localhost:8000/api/tags | grep -q "${UTILITY_MODEL%%:*}"; then
    echo "   >> Pulling utility model ${UTILITY_MODEL} (one-time)..."
    docker exec aeon_brain_node ollama pull "${UTILITY_MODEL}" || \
        echo "   >> WARNING: utility model pull failed; agent will fall back to the strong model."
fi

echo "=================================================="
echo "    BRAIN ONLINE (GPU 1). READY ON PORT 8000.     "
echo "=================================================="
