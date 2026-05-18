#!/bin/bash
# Hard reset for GPU memory. 
# Restarts Ollama containers (instant VRAM flush) and kills vLLM / llama.cpp clusters.

echo "[1/4] Stopping transient tools and clusters (vLLM/llama.cpp)..."
docker rm -f aeon_vllm aeon_gemma4_node0 aeon_gemma4_node1 aeon_gemma_lb aeon_qwen36_vl aeon_gemma4_mtp aeon_gemma4_mtp_node0 aeon_gemma4_mtp_node1 aeon_gemma_mtp_lb >/dev/null 2>&1 || true

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
