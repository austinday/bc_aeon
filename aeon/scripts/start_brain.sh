#!/bin/bash
set -e

echo "=================================================="
echo "    STARTING AEON LOCAL BRAIN (SINGLE NODE)       "
echo "=================================================="

# Host Directory for Persistence (Unified Model Lake)
HOST_OLLAMA_DIR="$HOME/bc_aeon/aeon_models/ollama_home"

if [ ! -d "$HOST_OLLAMA_DIR" ]; then
    echo "Error: Model directory not found at $HOST_OLLAMA_DIR"
    echo "Please run 'bash setup_environment.sh' first."
    exit 1
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

# --- 2. START BRAIN NODE (GPU 0 - Port 8000) ---
echo "[2/2] Launching Brain Node (GPU 0 -> :8000)..."

# Configuration:
# - OLLAMA_MAX_LOADED_MODELS=2: Allows Strong & Weak to coexist in VRAM if they fit.
# - OLLAMA_NUM_PARALLEL=4: Allows queueing and parallel requests.
# - device=0: Strictly use GPU 0, leaving GPU 1 free.

docker run -d \
    --name aeon_brain_node \
    --gpus '"device=0"' \
    -v "$HOST_OLLAMA_DIR:/root/.ollama" \
    -e OLLAMA_KEEP_ALIVE=-1 \
    -e OLLAMA_MAX_LOADED_MODELS=2 \
    -e OLLAMA_NUM_PARALLEL=4 \
    -e OLLAMA_NUM_CTX=131072 \
    -p 8000:11434 \
    ollama/ollama:latest

echo "=================================================="
echo "    BRAIN ONLINE (GPU 0). READY ON PORT 8000.     "
echo "=================================================="
