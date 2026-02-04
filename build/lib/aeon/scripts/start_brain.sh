#!/bin/bash
set -e

echo "=================================================="
echo "    STARTING AEON LOCAL BRAIN (OLLAMA ENGINE)      "
echo "=================================================="

# Host Directory for Persistence (Unified Model Lake)
HOST_OLLAMA_DIR="$HOME/bc_aeon/aeon_models/ollama_home"

if [ ! -d "$HOST_OLLAMA_DIR" ]; then
    echo "Error: Model directory not found at $HOST_OLLAMA_DIR"
    echo "Please run 'bash setup_environment.sh' first."
    exit 1
fi

# --- 1. AGGRESSIVE PORT CLEANUP ---
echo "[1/3] Checking for port conflicts..."

kill_container_on_port() {
  local PORT=$1
  local CID=$(docker ps -q --filter "publish=$PORT")
  if [ ! -z "$CID" ]; then
    echo "   >> Found container $CID holding port $PORT. Killing..."
    docker rm -f $CID >/dev/null
  fi
}

kill_container_on_port 8000
kill_container_on_port 8001

# Cleanup names
names=("aeon_strong_node" "aeon_weak_node")
for name in "${names[@]}"; do
  if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
    echo "   >> Removing old container: $name"
    docker rm -f $name >/dev/null
  fi
done

# --- 2. START STRONG NODE (GPU 0 - Port 8000) ---
echo "[2/3] Launching Strong Node (GPU 0 -> :8000)..."
# Mounts the HOST directory directly. No internal volume needed.
docker run -d \
    --name aeon_strong_node \
    --gpus '"device=0"' \
    -v "$HOST_OLLAMA_DIR:/root/.ollama" \
    -e OLLAMA_KEEP_ALIVE=-1 \
    -e OLLAMA_NUM_CTX=131072 \
    -p 8000:11434 \
    ollama/ollama:latest

# --- 3. START WEAK NODE (GPU 1 - Port 8001) ---
echo "[3/3] Launching Weak Node (GPU 1 -> :8001)..."
# Mounts the EXACT SAME host directory. Models are shared instantly.
docker run -d \
    --name aeon_weak_node \
    --gpus '"device=1"' \
    -v "$HOST_OLLAMA_DIR:/root/.ollama" \
    -e OLLAMA_KEEP_ALIVE=-1 \
    -e OLLAMA_NUM_CTX=131072 \
    -p 8001:11434 \
    ollama/ollama:latest

echo "=================================================="
echo "    BRAIN ONLINE. READY FOR CONNECTIONS.           "
echo "=================================================="
