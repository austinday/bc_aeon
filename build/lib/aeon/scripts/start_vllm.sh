#!/bin/bash
# Usage: ./start_vllm.sh <model_folder_name> <gpu_device_id>
# Example: ./start_vllm.sh "Qwen2-VL-2B-Instruct" 1

MODEL_NAME=$1
GPU_ID=${2:-1}
PORT=8002

# Unified Model Lake Path
MODELS_DIR="$HOME/bc_aeon/aeon_models"

if [ -z "$MODEL_NAME" ]; then
  echo "Error: Model folder name required."
  echo "Available models in aeon_models:"
  ls -1 "$MODELS_DIR"
  exit 1
fi

MODEL_PATH="/models/$MODEL_NAME"

echo "Starting vLLM for $MODEL_NAME on GPU $GPU_ID (Port $PORT)..."

docker run -d \
  --name aeon_vllm \
  --gpus "device=$GPU_ID" \
  -v "$MODELS_DIR:/models" \
  -p $PORT:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code

echo "Container 'aeon_vllm' launched."
echo "Run 'docker logs -f aeon_vllm' to watch startup."
