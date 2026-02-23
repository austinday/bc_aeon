#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/home/aday/bc_aeon"
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

log_step "PHASE 1: Build Docker images (layer cache makes unchanged builds fast)"

# Stop any running containers that depend on these images
for cname in aeon_qwen397b; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${cname}$"; then
        log_step "Stopping stale container: $cname"
        docker rm -f "$cname" >/dev/null 2>&1 || true
    fi
done

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
# PHASE 5.6: MiniMax-M2.5 GGUF — Q5_K_S (llama.cpp served)
# =============================================================================
MINIMAX_GGUF_DIR="$PROJECT_ROOT/aeon_models/gguf_models/MiniMax-M2.5"
log_step "PHASE 5.6: Download MiniMax-M2.5 GGUF model shards (Q5_K_S, Q6_K, Q8_0)"
mkdir -p "$MINIMAX_GGUF_DIR"

if [[ -f "$MINIMAX_GGUF_DIR/.download_complete_v2" ]]; then
    log_step "MiniMax GGUF already downloaded, skipping."
else
    MINIMAX_DL_SCRIPT=$(mktemp /tmp/aeon_dl_minimax_XXXXXX.py)
    cat > "$MINIMAX_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "unsloth/MiniMax-M2.5-GGUF"
TARGET = "/models"
PREFIXES = ["Q5_K_S", "Q6_K", "Q8_0"]

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

    touch "$MINIMAX_GGUF_DIR/.download_complete_v2"
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

# Build ComfyUI Docker image (for FLUX image generation tool)
log_step "PHASE 6b: Build aeon_comfyui:latest Docker image"
log_step "Building aeon_comfyui:latest (installs PyTorch + ComfyUI + GGUF plugin, may take 5-10 min on first build)..."
docker build -t aeon_comfyui:latest -f "$PROJECT_ROOT/aeon/services/comfyui/Dockerfile" "$PROJECT_ROOT/aeon/services/comfyui/"
log_step "aeon_comfyui:latest built successfully."

# =============================================================================
# PHASE 7: ComfyUI Models (FLUX)
# =============================================================================
COMFY_MODELS_DIR="$PROJECT_ROOT/aeon_models/comfyui"
log_step "PHASE 7: Download FLUX GGUF models and encoders for ComfyUI"
mkdir -p "$COMFY_MODELS_DIR/unet"
mkdir -p "$COMFY_MODELS_DIR/text_encoders"
mkdir -p "$COMFY_MODELS_DIR/vae"

if [[ -f "$COMFY_MODELS_DIR/.download_complete" ]]; then
    log_step "FLUX models already downloaded, skipping."
else
    FLUX_DL_SCRIPT=$(mktemp /tmp/aeon_dl_flux_XXXXXX.py)
    cat > "$FLUX_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download

print('Downloading Flux 2 Dev UNet GGUF...', flush=True)
hf_hub_download(repo_id='unsloth/FLUX.2-dev-GGUF', filename='flux2-dev-Q4_K_S.gguf', local_dir='/models/unet')

print('Downloading Flux 2 VAE...', flush=True)
hf_hub_download(repo_id='Comfy-Org/flux2-dev', filename='split_files/vae/flux2-vae.safetensors', local_dir='/models')

print('Downloading FLUX.2 Mistral text encoder...', flush=True)
hf_hub_download(repo_id='Comfy-Org/flux2-dev', filename='split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors', local_dir='/models')

print('Downloads complete!', flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$COMFY_MODELS_DIR:/models" \
        -v "$FLUX_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py

    DL_EXIT=$?
    rm -f "$FLUX_DL_SCRIPT"
    if [[ $DL_EXIT -ne 0 ]]; then
        log_step "ERROR: FLUX download failed (exit code $DL_EXIT)"
        exit 1
    fi

    touch "$COMFY_MODELS_DIR/.download_complete"
fi
chown -R $(id -u):$(id -g) "$COMFY_MODELS_DIR" 2>/dev/null || true
log_step "PHASE 7 complete."

log_step "Setup complete. Models in $QWEN_GGUF_DIR, $MINIMAX_GGUF_DIR, $COMFY_MODELS_DIR"
