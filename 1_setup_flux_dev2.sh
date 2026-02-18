#!/bin/bash
set -e

echo "PHASE 1: Flux Dev2 setup check (models already downloaded via setup_environment.sh)."

# Idempotent: Check key models exist
if [[ -f "aeon_models/comfyui_models/clip/t5xxl_fp8_e4m3fn.safetensors" && -f "aeon_models/comfyui_models/unet/flux1-dev-fp8.safetensors" && -f "aeon_models/comfyui_models/vae/ae.safetensors" ]]; then
    echo "✓ All Flux Dev FP8 models present. Ready to test."
else
    echo "✗ Missing models. Run setup_environment.sh first."
    exit 1
fi

echo "Setup complete."