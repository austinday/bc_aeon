#!/bin/bash
set -e

echo 'Removing stale container...'
docker rm -f aeon_comfyui

echo 'Starting ComfyUI as root (no -u)...'
docker run -d --name aeon_comfyui --gpus 'device=1' -p 8188:8188 \
  -v /home/aday/bc_aeon/aeon_models/comfyui_models:/opt/ComfyUI/models \
  -v /home/aday/bc_aeon/comfyui_output:/opt/ComfyUI/output \
  aeon/comfyui:latest

sleep 30
echo 'Container logs:'
docker logs aeon_comfyui --tail 50
echo '---'

echo 'API healthy?'
curl -s http://localhost:8188/system_stats || echo 'API down'

echo 'Flux-related nodes:'
curl -s http://localhost:8188/object_info | grep -i 'flux\|unet\|cliptextencodeflux\|modelsamplingflux\|piflow\|pi-flow' | head -20 || echo 'No matches'

echo 'All UNet loaders:'
curl -s http://localhost:8188/object_info | grep -i 'unetloader' || echo 'No UNetLoader'

echo 'pi-Flow nodes:'
curl -s http://localhost:8188/object_info | grep -i 'piflow\|pi-flow' || echo 'No piFlow'

docker stop aeon_comfyui
docker rm aeon_comfyui
echo 'Diagnostic complete.'