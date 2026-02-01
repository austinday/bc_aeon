#!/bin/bash
set -e

echo "========================================"
echo "      AEON INFRASTRUCTURE SETUP         "
echo "========================================"

# 1. Create Model Cache Directory (For Tools)
echo "\n[1/6] Creating Tool Model Cache..."
mkdir -p ~/bc_aeon/aeon_models

# 2. Build Base Images
echo "\n[2/6] Building Project Base Images..."
docker build -t aeon_base:py3.10-cuda12.1 \
  --build-arg PYTHON_VERSION_MINOR=10 \
  --build-arg CUDA_VERSION=12.1.1 \
  --build-arg PYTORCH_CUDA_SUFFIX=cu121 \
  .

# 3. Pull Images
echo "\n[3/6] Pulling Docker Images..."
docker pull ollama/ollama:latest

# 4. Build Vision Image
echo "\n[4/6] Building Vision Image..."
docker build -t aeon_vision:latest -f Dockerfile.vision .

# 5. Download Tool Models (Flux/Vision)
echo "\n[5/6] Downloading Tool Models (Python)..."
pip install huggingface_hub
python3 -u aeon/scripts/download_models.py

# 6. Hydrate Brain Models (Ollama)
echo "\n[6/6] Hydrating Brain Models into Docker Volumes..."
echo "This ensures models are ready before runtime. This may take a while (70GB+ per model)."

# --- PLANNER (DeepSeek R1) ---
echo " >> Pre-loading Planner Volume (aeon_ollama_planner)..."
# We run a temporary container just to populate the volume
docker rm -f temp_planner_setup 2>/dev/null || true
docker run -d --name temp_planner_setup -v aeon_ollama_planner:/root/.ollama ollama/ollama:latest

echo "    Pulling deepseek-r1:70b..."
docker exec temp_planner_setup ollama pull deepseek-r1:70b

echo "    Planner hydration complete. Cleaning up temp container..."
docker stop temp_planner_setup && docker rm temp_planner_setup

# --- EXECUTOR (Qwen 2.5) ---
echo " >> Pre-loading Executor Volume (aeon_ollama_executor)..."
docker rm -f temp_executor_setup 2>/dev/null || true
docker run -d --name temp_executor_setup -v aeon_ollama_executor:/root/.ollama ollama/ollama:latest

echo "    Pulling qwen2.5:72b..."
docker exec temp_executor_setup ollama pull qwen2.5:72b

echo "    Executor hydration complete. Cleaning up temp container..."
docker stop temp_executor_setup && docker rm temp_executor_setup

echo "\n========================================"
echo "        SETUP COMPLETE                  "
echo "========================================"
