#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
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

if ! docker image inspect aeon_downloader:latest >/dev/null 2>&1; then
    log_step "Building aeon_downloader:latest..."
    cat > "$PROJECT_ROOT/Dockerfile.downloader" << 'EOF'
FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "huggingface_hub[cli]"
EOF
    docker build --network=host -t aeon_downloader:latest -f "$PROJECT_ROOT/Dockerfile.downloader" "$PROJECT_ROOT"
    rm -f "$PROJECT_ROOT/Dockerfile.downloader"
else
    log_step "aeon_downloader:latest image already exists, skipping build."
fi

# =============================================================================
# PHASE 5.5: Qwen3-Coder-Next-Abliterated-Q8_0 (llama.cpp served)
# =============================================================================
QWEN3_CODER_GGUF_DIR="$AEON_HOME/models/gguf_models/Qwen3-Coder-Next-Abliterated"
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
    docker run --network=host --rm $TTY_FLAG \
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

    docker run --rm -v "$QWEN3_CODER_GGUF_DIR:/models" aeon_downloader:latest chown -R $(id -u):$(id -g) /models || true
    touch "$QWEN3_CODER_GGUF_DIR/.download_complete"
fi
log_step "PHASE 5.5 complete."

# =============================================================================
# PHASE 5.6: Gemma-4-31B + E2B Draft Models
# =============================================================================
GEMMA4_GGUF_DIR="$AEON_HOME/models/gguf_models/Gemma-4"
log_step "PHASE 5.6: Download Gemma 4 31B and E2B Draft GGUFs"
mkdir -p "$GEMMA4_GGUF_DIR"

if [[ -f "$GEMMA4_GGUF_DIR/.download_complete" ]]; then
    log_step "Gemma 4 models already downloaded, skipping."
else
    GEMMA4_DL_SCRIPT=$(mktemp /tmp/aeon_dl_gemma4_XXXXXX.py)
    cat > "$GEMMA4_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files

TARGET = "/models"

# 1. Target Model: 31B Abliterated Q8_0
target_repo = "paperscarecrow/Gemma-4-31B-it-abliterated"
target_file = "gemma-4-31b-abliterated-Q8_0.gguf"

# 2. Draft Model: E2B Heretic i1-Q4_K_M
draft_repo = "mradermacher/gemma-4-E2B-it-heretic-i1-GGUF"
print(f"Listing files in {draft_repo}...", flush=True)
try:
    repo_files = list_repo_files(draft_repo)
except Exception as e:
    print(f"ERROR: Failed to list repo {draft_repo}: {e}")
    sys.exit(1)

draft_file = next((f for f in repo_files if "Q4_K_M" in f and f.endswith(".gguf")), None)
if not draft_file:
    print(f"ERROR: Could not find Q4_K_M file in {draft_repo}!", flush=True)
    sys.exit(1)

downloads = [
    (target_repo, target_file),
    (draft_repo, draft_file)
]

for repo, fname in downloads:
    dest = os.path.join(TARGET, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000_000:
        print(f"[{fname}] already exists, skipping.", flush=True)
        continue
        
    print(f"Downloading {fname} from {repo}...", flush=True)
    try:
        hf_hub_download(
            repo_id=repo,
            filename=fname,
            local_dir=TARGET,
        )
        print(f"  Done: {fname}", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to download {fname}: {e}")
        sys.exit(1)

print("All Gemma 4 files downloaded successfully.", flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --network=host --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$GEMMA4_GGUF_DIR:/models" \
        -v "$GEMMA4_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py

    DL_EXIT=$?
    rm -f "$GEMMA4_DL_SCRIPT"
    if [[ $DL_EXIT -ne 0 ]]; then
        log_step "ERROR: Gemma 4 download failed (exit code $DL_EXIT)"
        exit 1
    fi

    docker run --rm -v "$GEMMA4_GGUF_DIR:/models" aeon_downloader:latest chown -R $(id -u):$(id -g) /models || true
    touch "$GEMMA4_GGUF_DIR/.download_complete"
fi
log_step "PHASE 5.6 complete."

# =============================================================================
# PHASE 5.7: Qwen3.6-35B-A3B-Uncensored GGUF (Vision & Primary LLM for llama.cpp)
# =============================================================================
QWEN36_VL_DIR="$AEON_HOME/models/vl_models/Qwen3.6-35B-A3B-GGUF"
log_step "PHASE 5.7: Download Qwen3.6-35B-A3B GGUF for vision analysis tool and primary LLM"
mkdir -p "$QWEN36_VL_DIR"

if [[ -f "$QWEN36_VL_DIR/.download_complete" ]]; then
    log_step "Qwen3.6-35B-A3B GGUF already downloaded, skipping."
else
    QWEN36_VL_DL_SCRIPT=$(mktemp /tmp/aeon_dl_qwen3_vl_XXXXXX.py)
    cat > "$QWEN36_VL_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files

REPO = 'HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive'
TARGET = '/models'

print(f'Listing files in {REPO}...', flush=True)
try:
    repo_files = list_repo_files(REPO)
    mmproj_file = next((f for f in repo_files if f.startswith('mmproj') and f.endswith('.gguf')), None)
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)

FILES = [
    'Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf',
]
if mmproj_file:
    FILES.append(mmproj_file)

for fname in FILES:
    print(f'Downloading {fname} from {REPO}...', flush=True)
    try:
        hf_hub_download(
            repo_id=REPO,
            filename=fname,
            local_dir=TARGET,
        )
        print(f'  -> {fname} complete.', flush=True)
    except Exception as e:
        print(f'ERROR: Failed to download {fname}: {e}', flush=True)
        sys.exit(1)

print('All files downloaded successfully!', flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --network=host --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$QWEN36_VL_DIR:/models" \
        -v "$QWEN36_VL_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py

    DL_EXIT=$?
    rm -f "$QWEN36_VL_DL_SCRIPT"
    if [[ $DL_EXIT -ne 0 ]]; then
        log_step "ERROR: Qwen3.6 GGUF download failed (exit code $DL_EXIT)"
        exit 1
    fi

    docker run --rm -v "$QWEN36_VL_DIR:/models" aeon_downloader:latest chown -R $(id -u):$(id -g) /models || true
    touch "$QWEN36_VL_DIR/.download_complete"
fi
log_step "PHASE 5.7 complete."

log_step "PHASE 6: Build aeon_llamacpp:latest Docker image"
log_step "Building aeon_llamacpp:latest (compiling llama.cpp with CUDA, may take 5-10 min on first build)..."
docker build --network=host -t aeon_llamacpp:latest -f "$PROJECT_ROOT/aeon/llamacpp/Dockerfile" "$PROJECT_ROOT/aeon/llamacpp/"
log_step "aeon_llamacpp:latest built successfully."

# Build ComfyUI Docker image (for FLUX image generation tool)
log_step "PHASE 6b: Build aeon_comfyui:latest Docker image"
log_step "Building aeon_comfyui:latest (installs PyTorch + ComfyUI + GGUF plugin, may take 5-10 min on first build)..."
docker build --network=host --no-cache -t aeon_comfyui:latest -f "$PROJECT_ROOT/aeon/services/comfyui/Dockerfile" "$PROJECT_ROOT/aeon/services/comfyui/"
log_step "aeon_comfyui:latest built successfully."

# =============================================================================
# PHASE 7: ComfyUI Models (FLUX)
# =============================================================================
COMFY_MODELS_DIR="$AEON_HOME/models/comfyui"
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

print('Downloading FHDR UNet GGUF...', flush=True)
hf_hub_download(repo_id='kpsss34/FHDR_Uncensored', filename='FHDR_ComfyUI-Q8_0.gguf', local_dir='/models/unet')

print('Downloading FLUX.1 VAE...', flush=True)
hf_hub_download(repo_id='black-forest-labs/FLUX.1-schnell', filename='ae.safetensors', local_dir='/models/vae')

print('Downloading FLUX.1 CLIP-L...', flush=True)
hf_hub_download(repo_id='comfyanonymous/flux_text_encoders', filename='clip_l.safetensors', local_dir='/models/text_encoders')

print('Downloading FLUX.1 T5XXL FP8...', flush=True)
hf_hub_download(repo_id='comfyanonymous/flux_text_encoders', filename='t5xxl_fp8_e4m3fn.safetensors', local_dir='/models/text_encoders')

print('Downloads complete!', flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --network=host --rm $TTY_FLAG \
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

    docker run --rm -v "$COMFY_MODELS_DIR:/models" aeon_downloader:latest chown -R $(id -u):$(id -g) /models || true
    touch "$COMFY_MODELS_DIR/.download_complete"
fi
log_step "PHASE 7 complete."

# =============================================================================
# PHASE 7.5: Qwen-Image-Edit-2511 (ComfyUI Edit Models)
# =============================================================================
log_step "PHASE 7.5: Download Qwen-Image-Edit-2511 models"

# Explicitly re-declare the directory to prevent empty variable evaluation
COMFY_MODELS_DIR="${AEON_HOME:-$HOME/.aeon}/models/comfyui"
mkdir -p "$COMFY_MODELS_DIR/unet" "$COMFY_MODELS_DIR/text_encoders" "$COMFY_MODELS_DIR/vae"

if [ -f "$COMFY_MODELS_DIR/unet/.qwen_edit_download_complete" ]; then
    log_step "Qwen-Image-Edit models already downloaded, skipping."
else
    QWEN_EDIT_DL_SCRIPT=$(mktemp /tmp/aeon_dl_qwen_edit_XXXXXX.py)
    cat > "$QWEN_EDIT_DL_SCRIPT" << 'PYEOF'
import os, sys, shutil
from huggingface_hub import hf_hub_download

print('Downloading Qwen-Rapid-NSFW-v23_Q8_0.gguf...', flush=True)
hf_hub_download(repo_id='Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF', filename='v23/Qwen-Rapid-NSFW-v23_Q8_0.gguf', local_dir='/models/unet')

print('Downloading Qwen VAE...', flush=True)
vae_path = hf_hub_download(repo_id='Comfy-Org/Qwen-Image_ComfyUI', filename='split_files/vae/qwen_image_vae.safetensors', local_dir='/tmp/dl')
os.makedirs('/models/vae', exist_ok=True)
shutil.copy(vae_path, '/models/vae/qwen_image_vae.safetensors')

print('Downloading Qwen Text Encoder...', flush=True)
te_path = hf_hub_download(repo_id='Comfy-Org/Qwen-Image_ComfyUI', filename='split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors', local_dir='/tmp/dl')
os.makedirs('/models/text_encoders', exist_ok=True)
shutil.copy(te_path, '/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors')

print('Qwen Image Edit downloads complete!', flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --network=host --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$COMFY_MODELS_DIR:/models" \
        -v "$QWEN_EDIT_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py

    DL_EXIT=$?
    rm -f "$QWEN_EDIT_DL_SCRIPT"
    if [ $DL_EXIT -ne 0 ]; then
        log_step "ERROR: Qwen-Image-Edit download failed (exit code $DL_EXIT)"
        exit 1
    fi

    docker run --rm -v "$COMFY_MODELS_DIR:/models" aeon_downloader:latest chown -R $(id -u):$(id -g) /models || true
    touch "$COMFY_MODELS_DIR/unet/.qwen_edit_download_complete"
fi
log_step "PHASE 7.5 complete."

# =============================================================================
# PHASE 8: PuLID FLUX Models (Consistent Characters)
# =============================================================================
PULID_MODELS_DIR="$AEON_HOME/models/comfyui/pulid"
CLIP_DIR="$AEON_HOME/models/comfyui/clip"
INSIGHTFACE_DIR="$AEON_HOME/models/comfyui/insightface"

log_step "PHASE 8: Download PuLID Flux and Face models"
mkdir -p "$PULID_MODELS_DIR" "$CLIP_DIR" "$INSIGHTFACE_DIR"

if [[ -f "$PULID_MODELS_DIR/.download_complete" ]]; then
    log_step "PuLID models already downloaded, skipping."
else
    PULID_DL_SCRIPT=$(mktemp /tmp/aeon_dl_pulid_XXXXXX.py)
    cat > "$PULID_DL_SCRIPT" << 'PYEOF'
import os
from huggingface_hub import hf_hub_download, snapshot_download

print('Downloading PuLID Flux...', flush=True)
hf_hub_download(repo_id='guozinan/PuLID', filename='pulid_flux_v0.9.0.safetensors', local_dir='/models/pulid')

print('Downloading EvaCLIP...', flush=True)
hf_hub_download(repo_id='QuanSun/EVA-CLIP', filename='EVA02_CLIP_L_336_psz14_s6B.pt', local_dir='/models/clip')

print('Downloading AntelopeV2 (InsightFace)...', flush=True)
snapshot_download(repo_id='kidyu/antelopev2-for-InstantID-ComfyUI', local_dir='/models/insightface/models/antelopev2')
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --network=host --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$AEON_HOME/models/comfyui:/models" \
        -v "$PULID_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py
        
    rm -f "$PULID_DL_SCRIPT"
    docker run --rm -v "$AEON_HOME/models/comfyui:/models" aeon_downloader:latest chown -R $(id -u):$(id -g) /models/pulid /models/clip /models/insightface || true
    touch "$PULID_MODELS_DIR/.download_complete"
fi
log_step "PHASE 8 complete."

log_step "Setup complete. Models in $QWEN3_CODER_GGUF_DIR, $COMFY_MODELS_DIR, $QWEN36_VL_DIR, $GEMMA4_GGUF_DIR"
log_step "NOTE: To remove old models (if present), you may want to clean up $AEON_HOME/models/vl_models/Qwen3.5-35B-A3B-GGUF or $AEON_HOME/models/gguf_models/Qwen3.5-27B"
