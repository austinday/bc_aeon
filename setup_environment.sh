#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
HF_TOKEN_FILE="/home/aday/huggingface_access_token.txt"
SETUP_VERSION="v2"

DOCKER_CACHE_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--force" ]; then
        DOCKER_CACHE_FLAG="--no-cache"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FORCE MODE ENABLED: Docker builds will use --no-cache"
    fi
done

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

build_image() {
    local img_name=$1
    local dockerfile=$2
    local context=$3
    log_step "Building/Verifying $img_name (Docker cache will handle unchanged layers)..."
    docker build --network=host $DOCKER_CACHE_FLAG -t "$img_name" -f "$dockerfile" "$context"
}

run_downloader() {
    local state_file="$1"
    local state_val="$2"
    local vol_map="$3"
    local cmd="$4"

    if [[ -f "$state_file" ]] && [[ "$(cat "$state_file")" == "$state_val" ]]; then
        log_step "Skipping download (already up-to-date): $(basename "$state_file")"
        return 0
    fi

    log_step "Running downloader for $state_val..."

    local tty_flag=""
    if [ -t 0 ]; then tty_flag="-t"; fi

    docker run --network=host --rm $tty_flag \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$vol_map" \
        aeon_downloader:latest \
        bash -c "$cmd"

    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_step "ERROR: Download failed (exit code $exit_code)"
        exit 1
    fi

    # Fix permissions to match host user
    local vol_mount="${vol_map##*:}"
    docker run --rm -v "$vol_map" aeon_downloader:latest chown -R $(id -u):$(id -g) "$vol_mount" || true

    echo "$state_val" > "$state_file"
}

log_step "PHASE 1: Build aeon_downloader"
cat > "$PROJECT_ROOT/Dockerfile.downloader" << 'EOF'
FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "huggingface_hub[cli]"
EOF
build_image "aeon_downloader:latest" "$PROJECT_ROOT/Dockerfile.downloader" "$PROJECT_ROOT"
rm -f "$PROJECT_ROOT/Dockerfile.downloader"

log_step "PHASE 5.5: Qwen3-Coder-Next-Abliterated-Q8_0"
QWEN3_CODER_GGUF_DIR="$AEON_HOME/models/gguf_models/Qwen3-Coder-Next-Abliterated"
mkdir -p "$QWEN3_CODER_GGUF_DIR"
CMD="hf download bartowski/huihui-ai_Qwen3-Coder-Next-abliterated-GGUF --include '*Q8_0*gguf' --local-dir /models"
run_downloader "$QWEN3_CODER_GGUF_DIR/.setup_state" "$SETUP_VERSION:qwen3-coder-q8_0" "$QWEN3_CODER_GGUF_DIR:/models" "$CMD"

log_step "PHASE 5.6: Gemma-4-31B + E2B Draft Models"
GEMMA4_GGUF_DIR="$AEON_HOME/models/gguf_models/Gemma-4"
mkdir -p "$GEMMA4_GGUF_DIR"
CMD="hf download paperscarecrow/Gemma-4-31B-it-abliterated gemma-4-31b-abliterated-Q8_0.gguf --local-dir /models && \
     hf download mradermacher/gemma-4-E2B-it-heretic-i1-GGUF --include '*Q4_K_M*.gguf' --local-dir /models"
run_downloader "$GEMMA4_GGUF_DIR/.setup_state" "$SETUP_VERSION:gemma4-q8_0-e2b-draft" "$GEMMA4_GGUF_DIR:/models" "$CMD"

log_step "PHASE 5.7: Qwen3.6-35B-A3B-Uncensored GGUF"
QWEN36_VL_DIR="$AEON_HOME/models/vl_models/Qwen3.6-35B-A3B-GGUF"
mkdir -p "$QWEN36_VL_DIR"
CMD="hf download HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf --local-dir /models && \
     hf download HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive --include '*mmproj*.gguf' --local-dir /models"
run_downloader "$QWEN36_VL_DIR/.setup_state" "$SETUP_VERSION:qwen36-vl-q8_k_p" "$QWEN36_VL_DIR:/models" "$CMD"

log_step "PHASE 6: Build aeon_vllm:latest Docker image"
build_image "aeon_vllm:latest" "$PROJECT_ROOT/aeon/services/vllm/Dockerfile" "$PROJECT_ROOT/aeon/services/vllm/"

log_step "PHASE 6.5: Gemma-4-31B Native Download (vLLM MTP)"
GEMMA4_VLLM_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$GEMMA4_VLLM_DIR"
CMD="hf download google/gemma-4-31b-it --exclude '*.msgpack' '*.h5' '*.pt'"
run_downloader "$AEON_HOME/models/.vllm_gemma4_setup_state" "$SETUP_VERSION:vllm_gemma4" "$GEMMA4_VLLM_DIR:/root/.cache/huggingface" "$CMD"

log_step "PHASE 6.8: Build aeon_llamacpp:latest Docker image"
build_image "aeon_llamacpp:latest" "$PROJECT_ROOT/aeon/llamacpp/Dockerfile" "$PROJECT_ROOT/aeon/llamacpp/"

log_step "PHASE 6.9: Build aeon_comfyui:latest Docker image"
build_image "aeon_comfyui:latest" "$PROJECT_ROOT/aeon/services/comfyui/Dockerfile" "$PROJECT_ROOT/aeon/services/comfyui/"

log_step "PHASE 7: ComfyUI Models (FLUX)"
COMFY_MODELS_DIR="$AEON_HOME/models/comfyui"
mkdir -p "$COMFY_MODELS_DIR/unet" "$COMFY_MODELS_DIR/text_encoders" "$COMFY_MODELS_DIR/vae"
CMD="hf download kpsss34/FHDR_Uncensored FHDR_ComfyUI-Q8_0.gguf --local-dir /models/unet && \
     hf download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir /models/vae && \
     hf download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir /models/text_encoders && \
     hf download comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn.safetensors --local-dir /models/text_encoders"
run_downloader "$COMFY_MODELS_DIR/.flux_setup_state" "$SETUP_VERSION:flux_comfyui" "$COMFY_MODELS_DIR:/models" "$CMD"

log_step "PHASE 7.5: Qwen-Image-Edit-2511 (ComfyUI Edit Models)"
CMD="hf download Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF v23/Qwen-Rapid-NSFW-v23_Q8_0.gguf --local-dir /models/unet && \
     hf download Comfy-Org/Qwen-Image_ComfyUI split_files/vae/qwen_image_vae.safetensors --local-dir /models/tmp && \
     mv /models/tmp/split_files/vae/qwen_image_vae.safetensors /models/vae/ && \
     hf download Comfy-Org/Qwen-Image_ComfyUI split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors --local-dir /models/tmp && \
     mv /models/tmp/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors /models/text_encoders/ && \
     rm -rf /models/tmp"
run_downloader "$COMFY_MODELS_DIR/.qwen_edit_setup_state" "$SETUP_VERSION:qwen_edit_comfyui" "$COMFY_MODELS_DIR:/models" "$CMD"

log_step "PHASE 8: PuLID FLUX Models (Consistent Characters)"
PULID_MODELS_DIR="$COMFY_MODELS_DIR/pulid"
CLIP_DIR="$COMFY_MODELS_DIR/clip"
INSIGHTFACE_DIR="$COMFY_MODELS_DIR/insightface"
mkdir -p "$PULID_MODELS_DIR" "$CLIP_DIR" "$INSIGHTFACE_DIR"
CMD="hf download guozinan/PuLID pulid_flux_v0.9.0.safetensors --local-dir /models/pulid && \
     hf download QuanSun/EVA-CLIP EVA02_CLIP_L_336_psz14_s6B.pt --local-dir /models/clip && \
     hf download kidyu/antelopev2-for-InstantID-ComfyUI --local-dir /models/insightface/models/antelopev2"
run_downloader "$COMFY_MODELS_DIR/.pulid_setup_state" "$SETUP_VERSION:pulid_comfyui" "$COMFY_MODELS_DIR:/models" "$CMD"

log_step "PHASE 9: Build aeon_browser_service:latest Docker image"
build_image "aeon_browser_service:latest" "$PROJECT_ROOT/aeon/services/browser/Dockerfile" "$PROJECT_ROOT/aeon/services/browser/"

log_step "PHASE 10: LTX-2.3 Video Generation Models"
CMD="hf download unsloth/LTX-2.3-GGUF ltx-2.3-22b-dev-F16.gguf --local-dir /models/unet && \
     hf download unsloth/LTX-2.3-GGUF vae/ltx-2.3-22b-dev_video_vae.safetensors --local-dir /models/vae && \
     hf download unsloth/LTX-2.3-GGUF text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors --local-dir /models/text_encoders && \
     hf download unsloth/gemma-3-12b-it-qat-GGUF gemma-3-12b-it-qat-UD-Q4_K_XL.gguf --local-dir /models/text_encoders && \
     hf download unsloth/gemma-3-12b-it-qat-GGUF mmproj-BF16.gguf --local-dir /models/text_encoders && \
     hf download unsloth/gemma-3-12b-it-FP8-Dynamic tokenizer.model --local-dir /models/tmp && \
     mv /models/tmp/tokenizer.model /models/text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.model && \
     rm -rf /models/tmp"
run_downloader "$COMFY_MODELS_DIR/.ltx_setup_state" "$SETUP_VERSION:ltx_comfyui" "$COMFY_MODELS_DIR:/models" "$CMD"

log_step "Setup complete! All Dockerfiles will automatically rebuild if changed, and partial downloads will automatically resume."
