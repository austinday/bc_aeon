#!/bin/bash
set -e

echo "=================================================="
echo "    STARTING AEON LOCAL BRAIN (OLLAMA ENGINE)       "
echo "=================================================="

# 1. CLEANUP OLD CONTAINERS
echo "[1/4] Cleaning up..."
docker stop aeon_planner aeon_executor 2>/dev/null || true
docker rm aeon_planner aeon_executor 2>/dev/null || true

# 2. START PLANNER (GPU 0 - DeepSeek R1)
# Added OLLAMA_KEEP_ALIVE=-1 to prevent unloading
# Updated OLLAMA_NUM_CTX=131072 (128k) per user request
echo "[2/4] Launching Planner (DeepSeek R1) on GPU 0..."
docker run -d \
    --name aeon_planner \
    --gpus '"device=0"' \
    -v aeon_ollama_planner:/root/.ollama \
    -e OLLAMA_KEEP_ALIVE=-1 \
    -e OLLAMA_NUM_CTX=131072 \
    -p 8000:11434 \
    ollama/ollama:latest

# 3. START EXECUTOR (GPU 1 - Qwen 2.5)
# Added OLLAMA_KEEP_ALIVE=-1 to prevent unloading
# Updated OLLAMA_NUM_CTX=131072 (128k) per user request
echo "[3/4] Launching Executor (Qwen 2.5) on GPU 1..."
docker run -d \
    --name aeon_executor \
    --gpus '"device=1"' \
    -v aeon_ollama_executor:/root/.ollama \
    -e OLLAMA_KEEP_ALIVE=-1 \
    -e OLLAMA_NUM_CTX=131072 \
    -p 8001:11434 \
    ollama/ollama:latest

# 4. PRE-WARM MODELS (Force Load into VRAM)
echo "[4/4] Pre-warming models..."

# Wait for Planner API to be live
echo -n "Waiting for Planner API..."
until curl -s -f http://localhost:8000/api/tags > /dev/null; do
    sleep 2
    echo -n "."
done
echo " OK"

# Wait for Executor API to be live
echo -n "Waiting for Executor API..."
until curl -s -f http://localhost:8001/api/tags > /dev/null; do
    sleep 2
    echo -n "."
done
echo " OK"

echo " >> Triggering VRAM Hydration (Sending 'warmup' query)..."
# We send a short prompt 'warmup' to force the engine to load weights.
# We set keep_alive to -1 to lock it in memory.
# We background (&) this so the script finishes, but the GPU will start filling up immediately.
curl -s http://localhost:8000/api/generate -d '{"model": "deepseek-r1:70b", "prompt": "warmup", "keep_alive": -1}' > /dev/null &
curl -s http://localhost:8001/api/generate -d '{"model": "qwen2.5:72b", "prompt": "warmup", "keep_alive": -1}' > /dev/null &

echo "=================================================="
echo "    BRAIN ONLINE. MODELS LOADING IN BACKGROUND.    "
echo "=================================================="
