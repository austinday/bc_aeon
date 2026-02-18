#!/bin/bash
set -euo pipefail
MODELS_BASE="/home/aday/bc_aeon/aeon_models/comfyui_models"
SPLIT_FILES="$MODELS_BASE/split_files"
mkdir -p "$SPLIT_FILES"/{diffusion_models,text_encoders,vae} "$MODELS_BASE/text_encoders"

# Flux.2 piFlow symlinks (idempotent)
ln -sf "../../unet/flux2_dev_fp8mixed.safetensors" "$SPLIT_FILES/diffusion_models/flux2_dev_fp8mixed.safetensors" 2>/dev/null || true
ln -sf "../../vae/flux2-vae.safetensors" "$SPLIT_FILES/vae/flux2-vae.safetensors" 2>/dev/null || true
ln -sf "../../clip/mistral_3_small_flux2_fp8.safetensors" "$SPLIT_FILES/text_encoders/mistral_3_small_flux2_fp8.safetensors" 2>/dev/null || true
ln -sf "../../clip/mistral_3_small_flux2_fp8.safetensors" "$MODELS_BASE/text_encoders/mistral_3_small_flux2_fp8.safetensors" 2>/dev/null || true

# LoRA already in loras/
echo 'Flux.2 symlinks created/verified.'
ls -lh "$SPLIT_FILES"/{"diffusion_models","vae","text_encoders"}/*.safetensors