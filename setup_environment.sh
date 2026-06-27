#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
HF_TOKEN_FILE="/home/aday/huggingface_access_token.txt"
SETUP_VERSION="v2"

DOCKER_CACHE_FLAG=""
LITE_MODE="false"

for arg in "$@"; do
    if [ "$arg" == "--force" ]; then
        DOCKER_CACHE_FLAG="--no-cache"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FORCE MODE ENABLED: Docker builds will use --no-cache"
    elif [ "$arg" == "--lite" ]; then
        LITE_MODE="true"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] LITE MODE ENABLED: Skipping massive models and heavy containers."
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

log_step "PHASE 5.6: Local model catalog (GPU-adaptive, VRAM-gated downloads)"
# Detect this machine's GPUs and download only the catalog models that fit (see
# aeon/core/model_catalog.py). The same catalog drives runtime auto-deployment,
# so setup and the agent agree on what is available. Portable across the 48 GB
# (RTX 5000) and 96 GB (RTX 6000) Blackwell machines.
GEMMA4_GGUF_DIR="$AEON_HOME/models/gguf_models/Gemma-4"
mkdir -p "$GEMMA4_GGUF_DIR"
MIN_VRAM=$(python3 -m aeon.core.gpu 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin).get('min_total_gib',0))" 2>/dev/null || echo 0)
log_step "Detected min per-GPU VRAM: ${MIN_VRAM} GiB"
LITE_FLAG=""; [[ "$LITE_MODE" == "true" ]] && LITE_FLAG="--lite"
if python3 -c "import sys; sys.exit(0 if float('${MIN_VRAM}')>0 else 1)" 2>/dev/null; then
    while IFS=$'\t' read -r action a b c; do
        case "$action" in
            DOWNLOAD)
                DEST_DIR="$AEON_HOME/models/$a"
                mkdir -p "$DEST_DIR"
                log_step "  -> downloading into $a (state: $b)"
                run_downloader "$DEST_DIR/.setup_state" "$SETUP_VERSION:$b" "$DEST_DIR:/models" "$c"
                ;;
            SKIP) log_step "  -> skip $a ($b)" ;;
        esac
    done < <(python3 -m aeon.core.model_catalog --emit-downloads "$MIN_VRAM" $LITE_FLAG)
else
    log_step "WARNING: no GPU detected; skipping local model downloads."
fi

# The published MTP assistant GGUF uses a naming convention the fork's
# gemma4-assistant loader rejects. Normalize it once into *.aeon.* so the MTP
# cluster can load it. Idempotent; the adaptive launcher also self-heals at runtime.
RAW_MTP_ASSISTANT="$GEMMA4_GGUF_DIR/gemma-4-31B-it-assistant.Q4_K_M.gguf"
NORM_MTP_ASSISTANT="$GEMMA4_GGUF_DIR/gemma-4-31B-it-assistant.aeon.Q4_K_M.gguf"
if [[ -f "$RAW_MTP_ASSISTANT" && ! -f "$NORM_MTP_ASSISTANT" ]]; then
    log_step "PHASE 5.6b: Normalize Gemma-4 MTP assistant GGUF for fork compatibility"
    python3 "$PROJECT_ROOT/aeon/scripts/normalize_gemma4_assistant.py" \
        "$RAW_MTP_ASSISTANT" "$NORM_MTP_ASSISTANT" \
        || { rm -f "$NORM_MTP_ASSISTANT"; echo "WARNING: MTP assistant normalization failed (will retry at runtime)"; }
fi

if [[ "$LITE_MODE" != "true" ]]; then
    log_step "PHASE 5.7: Qwen3.6-35B-A3B-Uncensored GGUF (Q4_K_M for the dedicated GPU1 vision server)"
    QWEN36_VL_DIR="$AEON_HOME/models/vl_models/Qwen3.6-35B-A3B-GGUF"
    mkdir -p "$QWEN36_VL_DIR"
    CMD="hf download HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive --include '*Q4_K_M*.gguf' --local-dir /models && \
         hf download HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive --include '*mmproj*.gguf' --local-dir /models"
    run_downloader "$QWEN36_VL_DIR/.setup_state" "$SETUP_VERSION:qwen36-vl-q4_k_m" "$QWEN36_VL_DIR:/models" "$CMD"
fi

log_step "PHASE 6: Build aeon_vllm:latest Docker image"
build_image "aeon_vllm:latest" "$PROJECT_ROOT/aeon/services/vllm/Dockerfile" "$PROJECT_ROOT/aeon/services/vllm/"

log_step "PHASE 6.8: Build aeon_llamacpp:latest Docker image"
build_image "aeon_llamacpp:latest" "$PROJECT_ROOT/aeon/llamacpp/Dockerfile" "$PROJECT_ROOT/aeon/llamacpp/"

log_step "PHASE 6.8b: Build aeon_gemma4_mtp:latest Docker image"
build_image "aeon_gemma4_mtp:latest" "$PROJECT_ROOT/aeon/llamacpp/Dockerfile.mtp" "$PROJECT_ROOT/aeon/llamacpp/"

if [[ "$LITE_MODE" != "true" ]]; then
    log_step "PHASE 6.8c: Build aeon_ds4:latest (DeepSeek-V4-Flash fork) Docker image"
    build_image "aeon_ds4:latest" "$PROJECT_ROOT/aeon/llamacpp/Dockerfile.ds4" "$PROJECT_ROOT/aeon/llamacpp/"

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
fi

log_step "PHASE 9: Build aeon_browser_service:latest Docker image"
build_image "aeon_browser_service:latest" "$PROJECT_ROOT/aeon/services/browser/Dockerfile" "$PROJECT_ROOT/aeon/services/browser/"

if [[ "$LITE_MODE" != "true" ]]; then
    log_step "PHASE 10: LTX-2.3 Video Generation Models"
    # Quantized unet (Q4_K_M, ~14 GiB) so ComfyUI fits one Blackwell GPU alongside
    # the VAE + Gemma text encoder. F16 (~42 GiB) does not fit a 48 GB card.
    # generate_video.py auto-resolves whichever ltx unet quant is present.
    # VAE and text-encoder connectors live in repo subfolders; flatten them into
    # models/vae and models/text_encoders so ComfyUI lists them by basename
    # (matching the names generate_video.py resolves).
    CMD="hf download unsloth/LTX-2.3-GGUF ltx-2.3-22b-dev-Q4_K_M.gguf --local-dir /models/unet && \
         hf download unsloth/LTX-2.3-GGUF vae/ltx-2.3-22b-dev_video_vae.safetensors --local-dir /models/tmp_vae && \
         mv /models/tmp_vae/vae/*.safetensors /models/vae/ && rm -rf /models/tmp_vae && \
         hf download unsloth/LTX-2.3-GGUF text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors --local-dir /models/tmp_te && \
         mv /models/tmp_te/text_encoders/*.safetensors /models/text_encoders/ && rm -rf /models/tmp_te && \
         hf download unsloth/gemma-3-12b-it-qat-GGUF gemma-3-12b-it-qat-UD-Q4_K_XL.gguf --local-dir /models/text_encoders && \
         hf download unsloth/gemma-3-12b-it-qat-GGUF mmproj-BF16.gguf --local-dir /models/text_encoders && \
         hf download unsloth/gemma-3-12b-it-FP8-Dynamic tokenizer.model --local-dir /models/tmp && \
         mv /models/tmp/tokenizer.model /models/text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.model && \
         rm -rf /models/tmp"
    run_downloader "$COMFY_MODELS_DIR/.ltx_setup_state" "$SETUP_VERSION:ltx_comfyui_q4km" "$COMFY_MODELS_DIR:/models" "$CMD"

    # PHASE 11 (CyberNeurova DeepSeek V4) is now handled by the GPU-adaptive
    # catalog loop in PHASE 5.6: it downloads DeepSeek only on machines whose
    # VRAM can deploy it (e.g. 2x 96 GB), and skips it elsewhere.
    :
fi

log_step "Setup complete! All Dockerfiles will automatically rebuild if changed, and partial downloads will automatically resume."