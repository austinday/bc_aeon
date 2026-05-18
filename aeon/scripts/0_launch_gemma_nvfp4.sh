#!/bin/bash
set -e

echo "=== Gemma-4-31B NVFP4 Dual-Node Launcher ==="
echo "GPU0 -> 8016 (primary)"
echo "GPU1 -> 8017 (primary)"
echo "Load Balancer -> 8018"

# Kill ONLY this cluster's previous instances
docker rm -f aeon_gemma_vllm_lb gemma_node0 gemma_node1 >/dev/null 2>&1 || true
fuser -k 8018/tcp 2>/dev/null || true
sleep 2

# Node 0 (GPU 0) - port 8016
docker run -d --gpus '"device=0"' \
  --name gemma_node0 \
  --ipc=host \
  -p 8016:8016 \
  -v /home/aday/bc_aeon:/app \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/triton:/root/.triton" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -e TRITON_CACHE_DIR="/root/.triton" \
  -e VLLM_CACHE_ROOT="/root/.cache/vllm" \
  aeon_vllm:latest \
  --model LilaRest/gemma-4-31B-it-NVFP4-turbo \
  --served-model-name Gemma-4-31B-NVFP4 \
  --speculative-config '{"method": "draft_model", "model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 5}' \
  --port 8016 \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 262144

# Node 1 (GPU 1) - port 8017
docker run -d --gpus '"device=1"' \
  --name gemma_node1 \
  --ipc=host \
  -p 8017:8017 \
  -v /home/aday/bc_aeon:/app \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/triton:/root/.triton" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -e TRITON_CACHE_DIR="/root/.triton" \
  -e VLLM_CACHE_ROOT="/root/.cache/vllm" \
  aeon_vllm:latest \
  --model LilaRest/gemma-4-31B-it-NVFP4-turbo \
  --served-model-name Gemma-4-31B-NVFP4 \
  --speculative-config '{"method": "draft_model", "model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 5}' \
  --port 8017 \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 262144

echo "Waiting for nodes to start..."
sleep 25

# Start load balancer in a Docker container for proper lifecycle management
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
docker run -d \
    --name aeon_gemma_vllm_lb \
    --network host \
    -v "${SCRIPT_DIR}/vllm_lb.py:/app/vllm_lb.py" \
    -w /app \
    python:3.11-slim \
    sh -c "pip install fastapi uvicorn httpx && python vllm_lb.py" > /dev/null

echo "Load balancer started on port 8018"
echo "All services running. Use http://localhost:8018 as the OpenAI-compatible endpoint."
