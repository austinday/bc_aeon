#!/bin/bash
# Starts Gemma-4-31B as an ephemeral vision/multimodal server on GPU 0.
set -e

CONTAINER_NAME='aeon_gemma4_vl'
IMAGE_NAME='aeon_vllm:latest'
PORT=8020
GPU_ID=${VISION_GPU:-0}
AEON_HOME="${AEON_HOME:-$HOME/.aeon}"

echo "[Gemma4-VL] Checking for existing container..."
docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true

# We use vLLM for Gemma 4 vision as it handles video/multi-image payloads natively.
# Use TP=1 on GPU 0 to avoid RPC timeouts and communication overhead.
# Increase memory utilization to ensure the 31B model and KV cache fit comfortably.
docker run -d \
  --name $CONTAINER_NAME \
  --gpus "\"device=${GPU_ID}\"" \
  -e VLLM_USE_V1=0 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -p ${PORT}:8000 \
  --ipc=host \
  $IMAGE_NAME \
  --model ebircak/gemma-4-31B-it-4bit-W4A16-AWQ \
  --quantization compressed-tensors \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.3 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --limit-mm-per-prompt image=10 video=1 \
  --host 0.0.0.0 \
  --port 8000

echo "[Gemma4-VL] Waiting for server to load..."
for i in {1..120}; do
    if curl -s http://localhost:${PORT}/health >/dev/null; then
        echo "[Gemma4-VL] Server ready on port $PORT."
        exit 0
    fi
    sleep 5
done

echo "[Gemma4-VL] ERROR: Failed to start."
exit 1
