#!/bin/bash
set -e

echo "========================================"
echo "      AEON INFRASTRUCTURE SETUP         "
echo "========================================"

# 1. Build Base Image
echo "\n[1/3] Building Project Base Image..."
docker build -t aeon_base:py3.10-cuda12.1 \
  --build-arg PYTHON_VERSION_MINOR=10 \
  --build-arg CUDA_VERSION=12.1.1 \
  --build-arg PYTORCH_CUDA_SUFFIX=cu121 \
  .

# 2. Pull Ollama Engine
echo "\n[2/3] Pulling Ollama Engine..."
docker pull ollama/ollama:latest

# 3. Hydrate Brain Models (Ollama)
echo "\n[3/3] Hydrating Brain Models into Docker Volumes..."
echo "Models will be pulled into BOTH Planner and Executor volumes for maximum flexibility."

# The 2026 Model Registry
MODELS=(
  "qwen3-next:80b"
  "glm-4.7-flash:bf16"
  "nemotron-3-nano:30b"
  "qwen3:32b"
  "gpt-oss:120b"
  "deepseek-r1:70b"
  "gemma3:27b"
  "qwen3-coder:30b"
  "llama4:16x17b"
  "deepcoder:14b"
  "phi4:14b"
)

# Function to hydrate a specific volume
hydrate_volume() {
  local VOL_NAME=$1
  local TEMP_CONTAINER="temp_setup_${VOL_NAME}"
  
  echo " >> Hydrating Volume: ${VOL_NAME}..."
  
  # Cleanup potential leftovers
  docker rm -f ${TEMP_CONTAINER} 2>/dev/null || true
  
  # Start temp container with the volume mounted
  docker run -d --name ${TEMP_CONTAINER} -v ${VOL_NAME}:/root/.ollama ollama/ollama:latest
  
  # Pull each model
  for model in "${MODELS[@]}"; do
    echo "    ... pulling ${model}"
    docker exec ${TEMP_CONTAINER} ollama pull ${model}
  done
  
  # Stop container
  docker stop ${TEMP_CONTAINER} && docker rm ${TEMP_CONTAINER}
  echo "    Done."
}

# Execute Hydration
hydrate_volume "aeon_ollama_planner"
hydrate_volume "aeon_ollama_executor"

echo "\n========================================"
echo "        SETUP COMPLETE                  "
echo "========================================"
