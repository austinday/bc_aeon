#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/home/aday/bc_aeon"
MODEL_BASE="$PROJECT_ROOT/aeon_models/comfyui_models"
HF_TOKEN_FILE="/home/aday/huggingface_access_token.txt"

log_step() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Load HF_TOKEN if not set
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '\n')
    log_step "Loaded HF_TOKEN from $HF_TOKEN_FILE"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN environment variable required for model downloads."
    echo "Create $HF_TOKEN_FILE with your token or export HF_TOKEN."
    exit 1
fi

log_step "PHASE 1: Create model directories (idempotent)"
mkdir -p "$MODEL_BASE"/{clip,unet,vae,transformers}

log_step "PHASE 2: Check/Idempotent Docker build aeon_base:latest"
if ! docker images | grep -q 'aeon_base:latest'; then
    log_step "Building aeon_base:latest"
    docker build -t aeon_base:latest -f "$PROJECT_ROOT/aeon/comfyui/Dockerfile.aeon_base" "$PROJECT_ROOT/aeon/comfyui/"
else
    log_step "aeon_base:latest exists, skipping build"
fi

log_step "PHASE 3: Idempotent Docker build aeon/comfyui:latest"
if ! docker images | grep -q 'aeon/comfyui:latest'; then
    log_step "Building aeon/comfyui:latest"
    docker build -t aeon/comfyui:latest -f "$PROJECT_ROOT/aeon/comfyui/Dockerfile" "$PROJECT_ROOT/aeon/comfyui/"
else
    log_step "aeon/comfyui:latest exists, skipping build"
fi

log_step "PHASE 4: Download Flux Dev FP8 models (idempotent, requires HF_TOKEN)"

CUDA_IMAGE="nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04"

# Function to download model idempotently
download_model() {
    local model_id="$1"
    local local_path="$2"
    local filename="$3"

    if [[ -f "$MODEL_BASE/$local_path/$filename" ]]; then
        log_step "Model $filename already exists, skipping."
        return
    fi

    log_step "Downloading $filename from $model_id..."

    docker run --rm \
        -e HF_TOKEN="$HF_TOKEN" \
        -v "$MODEL_BASE:/models" \
        "$CUDA_IMAGE" \
        bash -c ": > /dev/tcp/huggingface.co/443 && \
        apt-get update && apt-get install -y python3 python3-pip git && \
        pip3 install --no-cache-dir huggingface_hub[cli] && \
        python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"$model_id\", local_dir=\"/models/$local_path\", local_dir_use_symlinks=False, resume_download=True)' && \
        echo 'Download complete'"

    chown -R $(id -u):$(id -g) "$MODEL_BASE/$local_path" || true
}

# clip_l
log_step "Downloading clip_l.safetensors..."
download_model "comfyanonymous/flux_text_encoders" "clip" "clip_l.safetensors"

# Mistral 7B instruct FP8 for clip_name2 (skipped: use t5xxl_fp8_e4m3fn.safetensors from clip/ instead)
# log_step "Downloading mistral text encoder..."
# download_model "comfyanonymous/flux_text_encoders" "transformers/mistral_instruct" "mistral_instruct_7b_flux.1_dev.safetensors"  # Wrong repo/filename

# Flux1-dev FP8 unet
log_step "Downloading flux1-dev unet FP8..."
download_model "Kijai/flux-fp8" "unet" "flux1-dev-fp8.safetensors"

# VAE ae.safetensors
log_step "Downloading ae.safetensors VAE..."
download_model "black-forest-labs/FLUX.1-dev" "vae" "ae.safetensors"

log_step "PHASE 4 complete: All Flux Dev FP8 models downloaded."

log_step "Setup complete. Models in $MODEL_BASE/{clip,unet,vae,transformers}"