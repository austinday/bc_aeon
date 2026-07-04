#!/bin/bash
# Hard reset for GPU memory. 
# Restarts Ollama containers (instant VRAM flush) and kills vLLM / llama.cpp clusters.

echo "[1/4] Stopping model clusters (vLLM/llama.cpp)..."
# Match the deploy-planner container families by prefix (aeon_<slug>[_node0/_node1/_lb])
# so this doesn't drift as model names change. Covers the Gemma-4 primary and the
# DeepSeek secondary; leaves the brain (Ollama) and service containers alone.
docker ps -aq --filter "name=aeon_gemma" --filter "name=aeon_cyberneurova" --filter "name=aeon_vllm" \
  | xargs -r docker rm -f >/dev/null 2>&1 || true

echo "[2/4] Restarting Brain Node (Ollama)..."
# We use restart because 'keep_alive=0' API calls can sometimes fail if the model is stuck generating.
# Restarting the container is the only 100% guarantee of zero VRAM usage.
docker restart aeon_brain_node

echo "[3/4] Resetting session locks and registries..."
rm -f /tmp/aeon_runtime.lock
rm -f /tmp/aeon_brain_startup.lock
rm -f /tmp/aeon_model_registry.json
rm -f /tmp/aeon_model_registry.lock
rm -f /tmp/aeon_comfyui_registry.*
rm -f /tmp/aeon_browser_registry.*

echo "[4/4] Done. GPU memory is empty."
