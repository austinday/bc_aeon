#!/bin/bash
# Hard reset for GPU memory. 
# Restarts Ollama containers (instant VRAM flush) and kills vLLM.

echo "[1/4] Stopping transient tools (vLLM, ComfyUI)..."
docker rm -f aeon_vllm >/dev/null 2>&1 || true
docker rm -f aeon_comfyui >/dev/null 2>&1 || true

echo "[2/4] Restarting Brain Node (Ollama)..."
# We use restart because 'keep_alive=0' API calls can sometimes fail if the model is stuck generating.
# Restarting the container is the only 100% guarantee of zero VRAM usage.
docker restart aeon_brain_node

echo "[3/4] Resetting session locks..."
rm -f /tmp/aeon_runtime.lock
rm -f /tmp/aeon_brain_startup.lock

echo "[4/4] Done. GPU memory is empty."
