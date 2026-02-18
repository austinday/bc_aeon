#!/bin/bash
set -e

log_step() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"; }

HF_TOKEN_FILE="/home/aday/huggingface_access_token.txt"
MODELS_BASE="/home/aday/bc_aeon/aeon_models/comfyui_models"
HOST_UID=$(id -u)
HOST_GID=$(id -g)

log_step "Checking HF_TOKEN..."
if [[ ! -f $HF_TOKEN_FILE ]]; then
  echo "ERROR: $HF_TOKEN_FILE missing."
  exit 1
fi
HF_TOKEN=$(cat $HF_TOKEN_FILE | tr -d '\n')

log_step "Creating models dirs..."
mkdir -p "$MODELS_BASE"/{clip,unet,vae,loras}

log_step "Current models:"
find "$MODELS_BASE" -name '*.safetensors' -exec ls -lh {} + 2>/dev/null || true

du -sh "$MODELS_BASE"/* 2>/dev/null || true

log_step "Downloading/verifying Flux.2 piFlow models via root Docker..."

docker run --rm -v "$PWD":/app -v "$HF_TOKEN_FILE":/token.txt:ro -e HF_TOKEN="$HF_TOKEN" \
  python:3.12-slim bash -c "
    apt-get update && apt-get install -y git curl && \
    pip install huggingface_hub && \
    python /app/download_flux2_models.py
  "

log_step "Chown outputs to host user..."
docker run --rm -v "$PWD/aeon_models":/models -u root busybox \
  chown -R $HOST_UID:$HOST_GID /models/comfyui_models

log_step "Final verification:"
find "$MODELS_BASE" -name '*.safetensors' -exec ls -lh {} + 2>/dev/null || true

du -sh "$MODELS_BASE"/* 2>/dev/null || true

log_step "Flux.2 piFlow models ready!"