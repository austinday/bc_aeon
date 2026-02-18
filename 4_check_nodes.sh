#!/bin/bash
set -e

CWD=/home/aday/bc_aeon
MODELS_DIR=${CWD}/aeon_models/comfyui_models
OUTPUT_DIR=${CWD}/comfyui_output

# Cleanup
docker rm -f aeon_comfyui 2>/dev/null || true

echo "Starting ComfyUI container for node diagnostic..."
docker run -d --name aeon_comfyui \
  --gpus 'device=1' \
  -p 8188:8188 \
  -v ${MODELS_DIR}:/opt/ComfyUI/models \
  -v ${OUTPUT_DIR}:/opt/ComfyUI/output \
  -u $(id -u):$(id -g) \
  aeon/comfyui:latest

sleep 30
echo "=== Available nodes (grep flux, unet, piflow, cliptextencodeflux):"
curl -s http://localhost:8188/object_info | grep -i 'flux\|unet\|piflow\|cliptextencode\|modelsamplingflux\|fluxguidance' | head -50
echo "=== ComfyUI logs (last 20 lines):"
docker logs aeon_comfyui | tail -20
echo "Diagnostic complete. Stopping container."
docker stop aeon_comfyui && docker rm aeon_comfyui