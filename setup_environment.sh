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

    docker run --label owner=aday --network=host --rm $tty_flag \
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
    docker run --label owner=aday --rm -v "$vol_map" aeon_downloader:latest chown -R $(id -u):$(id -g) "$vol_mount" || true

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

# Pre-cache the abliterated Gemma-4 NVFP4 build (the vLLM catalog entry) and its MTP
# draft into the HF hub cache that the vLLM launcher mounts, so the first agent launch
# doesn't pay a ~20 GB download. VRAM-gated like the catalog loop above; if skipped, the
# vLLM launcher still fetches at runtime. The downloader's chown step also repairs the
# (root-owned) ~/.cache/huggingface so the host user can manage it afterwards.
# NOTE: google/gemma-4-31B-it-assistant may be gated -- accept its terms once with the
# account that owns your HF token before running setup, or this phase will fail.
if python3 -c "import sys; sys.exit(0 if float('${MIN_VRAM}')>0 else 1)" 2>/dev/null \
   && python3 -c "import sys; sys.exit(0 if 21.0 <= 1.5*float('${MIN_VRAM}') else 1)" 2>/dev/null; then
    log_step "PHASE 5.6c: Pre-cache Gemma-4-31B NVFP4 (abliterated) + assistant draft for vLLM"
    HF_HUB_CACHE_DIR="$HOME/.cache/huggingface"
    mkdir -p "$HF_HUB_CACHE_DIR"
    GEMMA4_NVFP4_CMD="hf download aday777/gemma-4-31B-it-abliterated-NVFP4 && \
         hf download google/gemma-4-31B-it-assistant"
    run_downloader "$HF_HUB_CACHE_DIR/.aeon_gemma4_nvfp4_state" "$SETUP_VERSION:gemma4-nvfp4-vllm" \
        "$HF_HUB_CACHE_DIR:/root/.cache/huggingface" "$GEMMA4_NVFP4_CMD"
fi

# Pre-cache the abliterated Qwen3.6-27B FP8 (8-bit) build for vLLM. NATIVE MTP: the
# MTP head is baked into this checkpoint, so there is NO separate draft repo to fetch
# (unlike Gemma-4's assistant). ~31 GiB; VRAM-gated at the same 1.5x-of-one-GPU rule as
# the catalog entry (weights_gib=32.0). Skipped on small GPUs; the launcher still
# fetches at runtime if this phase is skipped.
if python3 -c "import sys; sys.exit(0 if float('${MIN_VRAM}')>0 else 1)" 2>/dev/null \
   && python3 -c "import sys; sys.exit(0 if 32.0 <= 1.5*float('${MIN_VRAM}') else 1)" 2>/dev/null; then
    log_step "PHASE 5.6d: Pre-cache Qwen3.6-27B FP8 (abliterated, native MTP) for vLLM"
    HF_HUB_CACHE_DIR="$HOME/.cache/huggingface"
    mkdir -p "$HF_HUB_CACHE_DIR"
    QWEN36_FP8_CMD="hf download kasimat/Qwen3.6-27B-AEON-Ultimate-Uncensored-FP8-MTP"
    run_downloader "$HF_HUB_CACHE_DIR/.aeon_qwen36_fp8_state" "$SETUP_VERSION:qwen36-27b-fp8-vllm" \
        "$HF_HUB_CACHE_DIR:/root/.cache/huggingface" "$QWEN36_FP8_CMD"
fi

# (Removed) PHASE 5.7 used to download the Qwen3.6-35B-A3B GGUF for a dedicated GPU1
# vision server. Vision now runs on the multimodal Gemma-4 already loaded on GPU0
# (analyze_image reuses it), so that ~21 GB download is no longer needed.

log_step "PHASE 6: Build aeon_vllm:latest Docker image"
build_image "aeon_vllm:latest" "$PROJECT_ROOT/aeon/services/vllm/Dockerfile" "$PROJECT_ROOT/aeon/services/vllm/"

# (Removed) aeon_llamacpp + aeon_gemma4_mtp builds: the only catalog entries that used
# them (Qwen3.6-35B text and Gemma-4 Q8_0) were retired in favor of Gemma-4 NVFP4 (vLLM).
# DeepSeek-V4-Flash uses its own aeon_ds4 image, built below on 96 GB machines.

if [[ "$LITE_MODE" != "true" ]]; then
    log_step "PHASE 6.8c: Build aeon_ds4:latest (DeepSeek-V4-Flash fork) Docker image"
    build_image "aeon_ds4:latest" "$PROJECT_ROOT/aeon/llamacpp/Dockerfile.ds4" "$PROJECT_ROOT/aeon/llamacpp/"

    log_step "PHASE 6.9: Build aeon_comfyui:latest Docker image"
    build_image "aeon_comfyui:latest" "$PROJECT_ROOT/aeon/services/comfyui/Dockerfile" "$PROJECT_ROOT/aeon/services/comfyui/"

    log_step "PHASE 7: ComfyUI Image Models (FLUX.2-klein-9B, UNCENSORED)"
    # Best uncensored image model that fits one Blackwell GPU: FLUX.2-klein-9B (Q8_0, ~10 GiB)
    # paired with an UNCENSORED FLUX.2 text encoder (the Mistral encoder is where FLUX.2's
    # restrictions live; this abliterated replacement removes them). Newer/higher-quality than
    # FLUX.1; generate_image.py builds the FLUX.2 graph and auto-resolves these files.
    #
    # NOTE: the uncensored encoder repo is GATED. Accept it once (with the account that owns
    # your HF token) before running setup, or this phase will fail with "requires approval":
    #   https://huggingface.co/ponpoke/flux2-klein-9b-uncensored-text-encoder
    COMFY_MODELS_DIR="$AEON_HOME/models/comfyui"
    mkdir -p "$COMFY_MODELS_DIR/unet" "$COMFY_MODELS_DIR/text_encoders" "$COMFY_MODELS_DIR/vae"
    CMD="hf download unsloth/FLUX.2-klein-9B-GGUF flux-2-klein-9b-Q8_0.gguf --local-dir /models/unet && \
         hf download ponpoke/flux2-klein-9b-uncensored-text-encoder flux2-klein-9b-uncensored-q8_0.gguf --local-dir /models/text_encoders && \
         hf download Comfy-Org/flux2-dev split_files/vae/flux2-vae.safetensors --local-dir /models/tmp_fv && \
         mv /models/tmp_fv/split_files/vae/flux2-vae.safetensors /models/vae/ && rm -rf /models/tmp_fv"
    run_downloader "$COMFY_MODELS_DIR/.flux_setup_state" "$SETUP_VERSION:flux2_klein_uncensored" "$COMFY_MODELS_DIR:/models" "$CMD"

    log_step "PHASE 7.5: Qwen-Image-Edit-2511 (ComfyUI Edit Models)"
    # The Qwen-Image-Edit TEXT ENCODER is the ABLITERATED (uncensored) Qwen2.5-VL, not the
    # stock Comfy-Org fp8_scaled safetensors (which is where this pipeline's prompt-level
    # censorship lived). Edit reads the input image in vision-language mode, so we fetch the
    # encoder GGUF AND its matching mmproj (vision projector) -- city96's CLIPLoaderGGUF
    # auto-pairs the mmproj by basename. Phil2Sat ships both, pre-fixed for this exact
    # Rapid-AIO pipeline. generate_image.py's EditImageTool loads it via CLIPLoaderGGUF.
    CMD="hf download Arunk25/Qwen-Image-Edit-Rapid-AIO-GGUF v23/Qwen-Rapid-NSFW-v23_Q8_0.gguf --local-dir /models/unet && \
         hf download Comfy-Org/Qwen-Image_ComfyUI split_files/vae/qwen_image_vae.safetensors --local-dir /models/tmp && \
         mv /models/tmp/split_files/vae/qwen_image_vae.safetensors /models/vae/ && \
         hf download Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF Qwen2.5-VL-7B-Instruct-abliterated/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf --local-dir /models/tmp_qwen && \
         hf download Phil2Sat/Qwen-Image-Edit-Rapid-AIO-GGUF Qwen2.5-VL-7B-Instruct-abliterated/Qwen2.5-VL-7B-Instruct-abliterated.mmproj-f16.gguf --local-dir /models/tmp_qwen && \
         mv -f /models/tmp_qwen/Qwen2.5-VL-7B-Instruct-abliterated/*.gguf /models/text_encoders/ && \
         rm -rf /models/tmp /models/tmp_qwen"
    run_downloader "$COMFY_MODELS_DIR/.qwen_edit_setup_state" "$SETUP_VERSION:qwen_edit_ablit_comfyui" "$COMFY_MODELS_DIR:/models" "$CMD"

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

log_step "PHASE 9.5: Pull SearXNG image (local metasearch for the search_web tool)"
# Local-only web search: aeon queries this on-machine SearXNG (no Tavily/cloud search
# API). Pre-pull so the first search_web call isn't a cold image download. The tool's
# start_searxng.sh writes the settings (JSON output + SafeSearch off) and runs it.
docker pull searxng/searxng:latest || echo "[setup] WARNING: searxng image pull failed; search_web will pull on first use."

if [[ "$LITE_MODE" != "true" ]]; then
    log_step "PHASE 10: LTX-2.3 Video Generation Models (UNCENSORED 10Eros NSFW finetune)"
    # The video unet is the NSFW-finetuned LTX-2.3 "10Eros" (vantagewithai), which bakes
    # in explicit-motion ("sulphur") weights + tuned connectors -- the real diffusion-level
    # uncensoring (stock LTX base only lightly knows NSFW). It is the same LTX-2.3 arch, so
    # it still uses the standard LTX VAE + connectors + Gemma-3 encoder downloaded below and
    # drops into generate_video.py's UnetLoaderGGUF node.
    # The Gemma-3 TEXT ENCODER is the ABLITERATED (uncensored) build, not stock: stock Gemma-3
    # is where the prompt-level restrictions live, so an uncensored unet fed by a censored
    # encoder still under-represents explicit conditioning. Same QAT lineage/arch as the stock
    # encoder (abliteration only orthogonalizes the refusal direction in the weights), so the
    # stock mmproj + tokenizer.model sidecar remain compatible and the on-disk filename is kept
    # identical -- generate_video.py's DualCLIPLoaderGGUF resolves it with no workflow change.
    # GPU-adaptive quant: Q8_0 (~23 GiB) on big cards (>=80 GiB), else Q4_K_M (~14 GiB) so it
    # still fits one 48 GB Blackwell alongside the vision server.
    LTX_QUANT="Q4_K_M"
    if python3 -c "import sys; sys.exit(0 if float('${MIN_VRAM}')>=80 else 1)" 2>/dev/null; then LTX_QUANT="Q8_0"; fi
    log_step "  -> 10Eros video unet quant: ${LTX_QUANT} (min per-GPU VRAM ${MIN_VRAM} GiB)"
    CMD="hf download vantagewithai/LTX2.3-10Eros-1.2-GGUF --include '*${LTX_QUANT}*.gguf' --local-dir /models/unet && \
         hf download unsloth/LTX-2.3-GGUF vae/ltx-2.3-22b-dev_video_vae.safetensors --local-dir /models/tmp_vae && \
         mv /models/tmp_vae/vae/*.safetensors /models/vae/ && rm -rf /models/tmp_vae && \
         hf download unsloth/LTX-2.3-GGUF text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors --local-dir /models/tmp_te && \
         mv /models/tmp_te/text_encoders/*.safetensors /models/text_encoders/ && rm -rf /models/tmp_te && \
         hf download mradermacher/gemma-3-12b-it-qat-abliterated-GGUF gemma-3-12b-it-qat-abliterated.Q4_K_M.gguf --local-dir /models/text_encoders && \
         mv -f /models/text_encoders/gemma-3-12b-it-qat-abliterated.Q4_K_M.gguf /models/text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf && \
         hf download unsloth/gemma-3-12b-it-qat-GGUF mmproj-BF16.gguf --local-dir /models/text_encoders && \
         hf download unsloth/gemma-3-12b-it-FP8-Dynamic tokenizer.model --local-dir /models/tmp && \
         mv /models/tmp/tokenizer.model /models/text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.model && \
         rm -rf /models/tmp"
    run_downloader "$COMFY_MODELS_DIR/.ltx_setup_state" "$SETUP_VERSION:ltx_10eros_v12_ablit_${LTX_QUANT}" "$COMFY_MODELS_DIR:/models" "$CMD"

    # PHASE 11 (CyberNeurova DeepSeek V4) is now handled by the GPU-adaptive
    # catalog loop in PHASE 5.6: it downloads DeepSeek only on machines whose
    # VRAM can deploy it (e.g. 2x 96 GB), and skips it elsewhere.
    :
fi

log_step "Setup complete! All Dockerfiles will automatically rebuild if changed, and partial downloads will automatically resume."