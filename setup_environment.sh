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

log_step "PHASE 2: Build Docker images (layer cache makes unchanged builds fast)"

# Stop any running containers that depend on these images
for cname in aeon_comfyui aeon_qwen397b; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${cname}$"; then
        log_step "Stopping stale container: $cname"
        docker rm -f "$cname" >/dev/null 2>&1 || true
    fi
done

log_step "Building aeon/comfyui:latest..."
docker build -t aeon/comfyui:latest -f "$PROJECT_ROOT/aeon/comfyui/Dockerfile" "$PROJECT_ROOT/aeon/comfyui/"

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

log_step "PHASE 4 complete."

# =============================================================================
# PHASE 5: Qwen3.5-397B-A17B-MXFP4 GGUF (llama.cpp served)
# =============================================================================
QWEN_GGUF_DIR="$PROJECT_ROOT/aeon_models/gguf_models/Qwen3.5-397B-A17B-MXFP4"

log_step "PHASE 5: Download Qwen3.5-397B-A17B-MXFP4 GGUF model shards"
mkdir -p "$QWEN_GGUF_DIR"

# Write download script to temp file (avoids quoting hell in docker bash -c)
QWEN_DL_SCRIPT=$(mktemp /tmp/aeon_dl_qwen_XXXXXX.py)
cat > "$QWEN_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "unsloth/Qwen3.5-397B-A17B-GGUF"
PREFIX = "MXFP4_MOE/Qwen3.5-397B-A17B-MXFP4"
TARGET = "/models"

print(f"Listing files in {REPO} matching {PREFIX}...", flush=True)
all_files = list_repo_files(REPO)
shards = sorted([f for f in all_files if f.startswith(PREFIX) and f.endswith(".gguf")])
print(f"Found {len(shards)} shard(s):", flush=True)
for s in shards:
    print(f"  {s}", flush=True)

if not shards:
    print("ERROR: No matching GGUF shards found in repo!", flush=True)
    all_mxfp4 = [f for f in all_files if "MXFP4" in f]
    print(f"All MXFP4 files in repo: {all_mxfp4}", flush=True)
    sys.exit(1)

all_done = True
for i, shard in enumerate(shards, 1):
    dest = os.path.join(TARGET, shard)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000_000:
        sz = os.path.getsize(dest) / (1024**3)
        print(f"[{i}/{len(shards)}] {os.path.basename(shard)} already exists ({sz:.1f}GB), skipping.", flush=True)
        continue
    all_done = False
    print(f"[{i}/{len(shards)}] Downloading {shard}...", flush=True)
    hf_hub_download(
        repo_id=REPO,
        filename=shard,
        local_dir=TARGET,
    )
    sz = os.path.getsize(dest) / (1024**3)
    print(f"  Done: {sz:.1f}GB", flush=True)

if all_done:
    print("All shards already present and valid.", flush=True)
else:
    print("All shards downloaded successfully.", flush=True)

total = sum(
    os.path.getsize(os.path.join(TARGET, s))
    for s in shards
    if os.path.exists(os.path.join(TARGET, s))
)
print(f"Total model size: {total / (1024**3):.1f}GB", flush=True)
PYEOF

# Run download with live output (-t for progress bars)
TTY_FLAG=""
if [ -t 0 ]; then TTY_FLAG="-t"; fi
docker run --rm $TTY_FLAG \
    -e HF_TOKEN="$HF_TOKEN" \
    -e PYTHONUNBUFFERED=1 \
    -v "$QWEN_GGUF_DIR:/models" \
    -v "$QWEN_DL_SCRIPT:/download.py:ro" \
    python:3.12-slim \
    bash -c "pip install --no-cache-dir huggingface_hub && python3 /download.py"

DL_EXIT=$?
rm -f "$QWEN_DL_SCRIPT"
if [[ $DL_EXIT -ne 0 ]]; then
    log_step "ERROR: Qwen GGUF download failed (exit code $DL_EXIT)"
    exit 1
fi

chown -R $(id -u):$(id -g) "$QWEN_GGUF_DIR" 2>/dev/null || true
log_step "PHASE 5 complete."

# =============================================================================
# PHASE 6: Build llama.cpp server Docker image (for GGUF model serving)
# =============================================================================
log_step "PHASE 6: Build aeon_llamacpp:latest Docker image"
log_step "Building aeon_llamacpp:latest (compiling llama.cpp with CUDA, may take 5-10 min on first build)..."
docker build -t aeon_llamacpp:latest -f "$PROJECT_ROOT/aeon/llamacpp/Dockerfile" "$PROJECT_ROOT/aeon/llamacpp/"
log_step "aeon_llamacpp:latest built successfully."

log_step "Setup complete. Models in $MODEL_BASE/{clip,unet,vae,transformers} and $QWEN_GGUF_DIR"