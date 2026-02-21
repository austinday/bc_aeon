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

if ! docker image inspect aeon_downloader:latest >/dev/null 2>&1; then
    log_step "Building aeon_downloader:latest..."
    cat > "$PROJECT_ROOT/Dockerfile.downloader" << 'EOF'
FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "huggingface_hub[cli]"
EOF
    docker build -t aeon_downloader:latest -f "$PROJECT_ROOT/Dockerfile.downloader" "$PROJECT_ROOT"
    rm -f "$PROJECT_ROOT/Dockerfile.downloader"
else
    log_step "aeon_downloader:latest image already exists, skipping build."
fi

log_step "PHASE 4: Download Flux Dev FP8 models (idempotent, requires HF_TOKEN)"

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
        aeon_downloader:latest \
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"$model_id\", local_dir=\"/models/$local_path\", local_dir_use_symlinks=False, resume_download=True)"

    chown -R $(id -u):$(id -g) "$MODEL_BASE/$local_path" || true
}

# clip_l
log_step "Downloading clip_l.safetensors..."
download_model "comfyanonymous/flux_text_encoders" "clip" "clip_l.safetensors"

# Flux1-dev FP8 unet
log_step "Downloading flux1-dev unet FP8..."
download_model "Kijai/flux-fp8" "unet" "flux1-dev-fp8.safetensors"

# VAE ae.safetensors
log_step "Downloading ae.safetensors VAE..."
download_model "black-forest-labs/FLUX.1-dev" "vae" "ae.safetensors"

log_step "PHASE 4 complete: All Flux Dev FP8 models downloaded."

# =============================================================================
# PHASE 7: FLUX.2-dev FP8 models (for FLUX.2 image generation tools)
# =============================================================================
log_step "PHASE 7: Download FLUX.2-dev FP8 models"

# FLUX.2-dev FP8 diffusion model (pi-Flow LoRA enabled)
log_step "Downloading flux2_dev_fp8mixed.safetensors (FLUX.2-dev FP8)..."
download_model "Kijai/flux-fp8" "flux2" "flux2_dev_fp8mixed.safetensors"

# FLUX.2 VAE
log_step "Downloading flux2-vae.safetensors..."
download_model "black-forest-labs/FLUX.2-dev" "flux2/vae" "flux2-vae.safetensors"

# FLUX.2 text encoders (CLIP)
log_step "Downloading mistral_3_small_flux2_fp8.safetensors (CLIP)..."
download_model "Kijai/flux-fp8" "flux2/text_encoders" "mistral_3_small_flux2_fp8.safetensors"

# FLUX.2 pi-Flow LoRA adapter
log_step "Downloading gmflux2_k8_piid_4step.safetensors (pi-Flow LoRA)..."
download_model "Kijai/flux-fp8" "flux2/adapters" "gmflux2_k8_piid_4step.safetensors"

log_step "PHASE 7 complete: All FLUX.2-dev FP8 models downloaded."

log_step "PHASE 4 complete."

# =============================================================================
# PHASE 5: Qwen3.5-397B-A17B-Q6_K GGUF (llama.cpp served)
# =============================================================================
QWEN_GGUF_DIR="$PROJECT_ROOT/aeon_models/gguf_models/Qwen3.5-397B-A17B"

log_step "PHASE 5: Download Qwen3.5-397B-A17B-Q6_K GGUF model shards"
mkdir -p "$QWEN_GGUF_DIR"

if [[ -f "$QWEN_GGUF_DIR/.download_complete" ]]; then
    log_step "Qwen GGUF already downloaded, skipping."
else
    QWEN_DL_SCRIPT=$(mktemp /tmp/aeon_dl_qwen_XXXXXX.py)
    cat > "$QWEN_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "unsloth/Qwen3.5-397B-A17B-GGUF"
TARGET = "/models"
MODELS_TO_DOWNLOAD = ["Q5_K_S"]

print(f"Listing files in {REPO}...", flush=True)
all_files = list_repo_files(REPO)

for prefix in MODELS_TO_DOWNLOAD:
    print(f"\nProcessing {prefix}...", flush=True)
    shards = sorted([f for f in all_files if prefix in f and f.endswith(".gguf")])
    print(f"Found {len(shards)} shard(s):", flush=True)
    for s in shards:
        print(f"  {s}", flush=True)

    if not shards:
        print(f"ERROR: No matching GGUF shards found in repo for {prefix}!", flush=True)
        continue

    all_done = True
    for i, shard in enumerate(shards, 1):
        dest = os.path.join(TARGET, shard)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
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
        print(f"All {prefix} shards already present and valid.", flush=True)
    else:
        print(f"All {prefix} shards downloaded successfully.", flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$QWEN_GGUF_DIR:/models" \
        -v "$QWEN_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py

    DL_EXIT=$?
    rm -f "$QWEN_DL_SCRIPT"
    if [[ $DL_EXIT -ne 0 ]]; then
        log_step "ERROR: Qwen GGUF download failed (exit code $DL_EXIT)"
        exit 1
    fi

    touch "$QWEN_GGUF_DIR/.download_complete"
fi
chown -R $(id -u):$(id -g) "$QWEN_GGUF_DIR" 2>/dev/null || true
log_step "PHASE 5 complete."

# =============================================================================
# PHASE 5.5: Qwen3-Coder-Next-Abliterated-Q8_0 (llama.cpp served)
# =============================================================================
QWEN3_CODER_GGUF_DIR="$PROJECT_ROOT/aeon_models/gguf_models/Qwen3-Coder-Next-Abliterated"
log_step "PHASE 5.5: Download Qwen3-Coder-Next-Abliterated-Q8_0 GGUF model shards"
mkdir -p "$QWEN3_CODER_GGUF_DIR"

if [[ -f "$QWEN3_CODER_GGUF_DIR/.download_complete" ]]; then
    log_step "Qwen3 Coder GGUF already downloaded, skipping."
else
    QWEN3_CODER_DL_SCRIPT=$(mktemp /tmp/aeon_dl_qwen3_coder_XXXXXX.py)
    cat > "$QWEN3_CODER_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "bartowski/huihui-ai_Qwen3-Coder-Next-abliterated-GGUF"
TARGET = "/models"
PREFIX = "Q8_0"

print(f"Listing files in {REPO}...", flush=True)
try:
    all_files = list_repo_files(REPO)
except Exception as e:
    print(f"Failed to list repo: {e}")
    sys.exit(1)

print(f"\nProcessing {PREFIX}...", flush=True)
shards = sorted([f for f in all_files if PREFIX in f and f.endswith(".gguf")])
print(f"Found {len(shards)} shard(s):", flush=True)
for s in shards:
    print(f"  {s}", flush=True)

if not shards:
    print(f"ERROR: No matching GGUF shards found in repo for {PREFIX}!", flush=True)
    sys.exit(1)

all_done = True
for i, shard in enumerate(shards, 1):
    dest = os.path.join(TARGET, shard)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000_000:
        sz = os.path.getsize(dest) / (1024**3)
        print(f"[{i}/{len(shards)}] {os.path.basename(shard)} already exists ({sz:.1f}GB), skipping.", flush=True)
        continue
    all_done = False
    print(f"[{i}/{len(shards)}] Downloading {shard}...", flush=True)
    try:
        hf_hub_download(
            repo_id=REPO,
            filename=shard,
            local_dir=TARGET,
        )
        sz = os.path.getsize(dest) / (1024**3)
        print(f"  Done: {sz:.1f}GB", flush=True)
    except Exception as e:
        print(f"Failed to download {shard}: {e}")
        sys.exit(1)

if all_done:
    print(f"All {PREFIX} shards already present and valid.", flush=True)
else:
    print(f"All {PREFIX} shards downloaded successfully.", flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$QWEN3_CODER_GGUF_DIR:/models" \
        -v "$QWEN3_CODER_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py

    DL_EXIT=$?
    rm -f "$QWEN3_CODER_DL_SCRIPT"
    if [[ $DL_EXIT -ne 0 ]]; then
        log_step "ERROR: Qwen3 Coder GGUF download failed (exit code $DL_EXIT)"
        exit 1
    fi

    touch "$QWEN3_CODER_GGUF_DIR/.download_complete"
fi
chown -R $(id -u):$(id -g) "$QWEN3_CODER_GGUF_DIR" 2>/dev/null || true
log_step "PHASE 5.5 complete."

# =============================================================================
# PHASE 5.6: MiniMax-M2.5 GGUF — Q8_0 (llama.cpp served)
# =============================================================================
MINIMAX_GGUF_DIR="$PROJECT_ROOT/aeon_models/gguf_models/MiniMax-M2.5"
log_step "PHASE 5.6: Download MiniMax-M2.5-Q8_0 GGUF model shards"
mkdir -p "$MINIMAX_GGUF_DIR"

if [[ -f "$MINIMAX_GGUF_DIR/.download_complete" ]]; then
    log_step "MiniMax GGUF already downloaded, skipping."
else
    MINIMAX_DL_SCRIPT=$(mktemp /tmp/aeon_dl_minimax_XXXXXX.py)
    cat > "$MINIMAX_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "unsloth/MiniMax-M2.5-GGUF"
TARGET = "/models"
PREFIXES = ["Q5_K_M"]

print(f"Listing files in {REPO}...", flush=True)
try:
    all_files = list_repo_files(REPO)
except Exception as e:
    print(f"Failed to list repo: {e}")
    sys.exit(1)

for PREFIX in PREFIXES:
    print(f"\nProcessing {PREFIX}...", flush=True)
    shards = sorted([f for f in all_files if PREFIX in f and f.endswith(".gguf")])
    print(f"Found {len(shards)} shard(s):", flush=True)
    for s in shards:
        print(f"  {s}", flush=True)

    if not shards:
        print(f"ERROR: No matching GGUF shards found in repo for {PREFIX}!", flush=True)
        sys.exit(1)

    all_done = True
    for i, shard in enumerate(shards, 1):
        dest = os.path.join(TARGET, shard)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000_000:
            sz = os.path.getsize(dest) / (1024**3)
            print(f"[{i}/{len(shards)}] {os.path.basename(shard)} already exists ({sz:.1f}GB), skipping.", flush=True)
            continue
        all_done = False
        print(f"[{i}/{len(shards)}] Downloading {shard}...", flush=True)
        try:
            hf_hub_download(
                repo_id=REPO,
                filename=shard,
                local_dir=TARGET,
            )
            sz = os.path.getsize(dest) / (1024**3)
            print(f"  Done: {sz:.1f}GB", flush=True)
        except Exception as e:
            print(f"Failed to download {shard}: {e}")
            sys.exit(1)

    if all_done:
        print(f"All {PREFIX} shards already present and valid.", flush=True)
    else:
        print(f"All {PREFIX} shards downloaded successfully.", flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$MINIMAX_GGUF_DIR:/models" \
        -v "$MINIMAX_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py

    DL_EXIT=$?
    rm -f "$MINIMAX_DL_SCRIPT"
    if [[ $DL_EXIT -ne 0 ]]; then
        log_step "ERROR: MiniMax GGUF download failed (exit code $DL_EXIT)"
        exit 1
    fi

    touch "$MINIMAX_GGUF_DIR/.download_complete"
fi
chown -R $(id -u):$(id -g) "$MINIMAX_GGUF_DIR" 2>/dev/null || true
log_step "PHASE 5.6 complete."

# =============================================================================
# PHASE 6: Build llama.cpp server Docker image (for GGUF model serving)
# =============================================================================
log_step "PHASE 6: Build aeon_llamacpp:latest Docker image"
log_step "Building aeon_llamacpp:latest (compiling llama.cpp with CUDA, may take 5-10 min on first build)..."
docker build -t aeon_llamacpp:latest -f "$PROJECT_ROOT/aeon/llamacpp/Dockerfile" "$PROJECT_ROOT/aeon/llamacpp/"
log_step "aeon_llamacpp:latest built successfully."

log_step "Setup complete. Models in $MODEL_BASE/{clip,unet,vae,transformers}, $QWEN_GGUF_DIR, $MINIMAX_GGUF_DIR"
