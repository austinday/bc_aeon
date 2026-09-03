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

# Qwen3.8-27B is a locally built, checksum-validated NVFP4 artifact rather than
# a mutable Hub tag. setup never substitutes or downloads a different model.
# The build/install workflow places it at this exact path; the runtime catalog
# hides it until BUILD_MANIFEST.json says complete+validated.
QWEN38_LOCAL_DIR="$AEON_HOME/models/Qwen3.8-27B-ARA-abliterated-NVFP4-MTP"
if [ -f "$QWEN38_LOCAL_DIR/BUILD_MANIFEST.json" ]; then
    log_step "PHASE 5.6d: Found local Qwen3.8-27B ARA NVFP4 + native MTP artifact"
else
    log_step "PHASE 5.6d: Qwen3.8 artifact not installed at $QWEN38_LOCAL_DIR"
fi

# (Removed) PHASE 5.7 used to download a legacy 35B-A3B GGUF for a dedicated GPU1
# vision server. Vision now runs on the already-loaded multimodal Qwen3.8;
# topology-aware placement may share one sufficiently large GPU with media tools.
# (analyze_image reuses it), so that ~21 GB download is no longer needed.

log_step "PHASE 6: Build aeon_vllm:latest Docker image"
build_image "aeon_vllm:latest" "$PROJECT_ROOT/aeon/services/vllm/Dockerfile" "$PROJECT_ROOT/aeon/services/vllm/"

# Retired alternate language-model images are intentionally not built. Qwen3.8
# is Aeon's sole text and vision model and uses aeon_vllm.

if [[ "$LITE_MODE" != "true" ]]; then
    log_step "PHASE 6.9: Build aeon_comfyui:latest Docker image"
    build_image "aeon_comfyui:latest" "$PROJECT_ROOT/aeon/services/comfyui/Dockerfile" "$PROJECT_ROOT/aeon/services/comfyui/"

    # The enabled worker-video profile is a release manifest, not a mutable tag.
    # Materialize its exact, untagged amd64 transport only when this build still
    # matches the reviewed local image identity; a changed image needs a normal
    # profile/adapter release rather than silently replacing production bytes.
    VIDEO_LOCAL_IMAGE_ID="sha256:e87d7bcd4da3b5826e03740585ee22a5c78bf5f4468e881495375798f677ba8d"
    VIDEO_WORKER_IMAGE_CONFIG="75d861d5d12d2f27004d131568356d49952e24e2c500b668e771045bc20c9633"
    VIDEO_ARCHIVE_SHA256="7b2c2e156cdb70d8c75de1e3b6c6744e9fba6056a6dc0a9c966280a62a276091"
    VIDEO_OCI_DIR="/home/aday/.local/state/fleet-compute/artifacts/aeon-video-comfyui/oci/$VIDEO_WORKER_IMAGE_CONFIG"
    VIDEO_OCI_ARCHIVE="$VIDEO_OCI_DIR/image.tar"
    OBSERVED_COMFY_ID="$(docker image inspect --format '{{.Id}}' aeon_comfyui:latest)"
    if [[ "$OBSERVED_COMFY_ID" != "$VIDEO_LOCAL_IMAGE_ID" ]]; then
        echo "ERROR: aeon_comfyui changed; review and release new video image/profile identities."
        exit 1
    fi
    install -d -m 700 "$VIDEO_OCI_DIR"
    if [[ ! -f "$VIDEO_OCI_ARCHIVE" ]]; then
        VIDEO_OCI_TEMP="$(mktemp "$VIDEO_OCI_DIR/.image.tar.XXXXXXXX.partial")"
        if ! docker image save --platform linux/amd64 --output "$VIDEO_OCI_TEMP" "$VIDEO_LOCAL_IMAGE_ID"; then
            rm -f -- "$VIDEO_OCI_TEMP"
            exit 1
        fi
        chmod 600 "$VIDEO_OCI_TEMP"
        if [[ "$(sha256sum "$VIDEO_OCI_TEMP" | cut -d' ' -f1)" != "$VIDEO_ARCHIVE_SHA256" ]]; then
            rm -f -- "$VIDEO_OCI_TEMP"
            echo "ERROR: reviewed ComfyUI OCI transport checksum changed."
            exit 1
        fi
        mv -- "$VIDEO_OCI_TEMP" "$VIDEO_OCI_ARCHIVE"
    fi
    if [[ "$(sha256sum "$VIDEO_OCI_ARCHIVE" | cut -d' ' -f1)" != "$VIDEO_ARCHIVE_SHA256" ]]; then
        echo "ERROR: installed ComfyUI OCI transport checksum changed."
        exit 1
    fi

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

log_step "PHASE 9: Build and receipt the exact CPU-only browser service image"
BROWSER_BUILD_ARGS=(build-image)
if [[ "$DOCKER_CACHE_FLAG" == "--no-cache" ]]; then
    BROWSER_BUILD_ARGS+=(--no-cache)
fi
python3 -m aeon.scripts.browser_service "${BROWSER_BUILD_ARGS[@]}"

log_step "PHASE 9.5: Pull SearXNG image (local metasearch for the search_web tool)"
# Local-only web search: the model-facing tool has no Docker authority. Pre-pull
# the exact reviewed image used by the receipted, CPU-capped operator helper.
SEARXNG_IMAGE="searxng/searxng@sha256:892cf809341915a4b7710d3c9045005b4c377d51335a089b6d4da0b28750788d"
docker pull "$SEARXNG_IMAGE" || echo "[setup] WARNING: pinned SearXNG image pull failed; operator provisioning remains unavailable."

if [[ "$LITE_MODE" != "true" ]]; then
    log_step "PHASE 10: MiniMax H3 Audiovisual Models (10Eros-Max + Heretic NVFP4)"
    # Primary renderer for ordinary T2VA/I2VA/FL2VA: H3 generates synchronized
    # stereo audio, follows a timed multi-shot IR, and supports first/last-frame
    # conditioning. The diffusion checkpoint and Qwen3-VL conditioner are both
    # uncensored community variants. Both are NVFP4 and use ComfyUI's native H3
    # path; exact Hub revisions prevent a mutable tag from silently changing the
    # installed stack. The video/audio VAEs are the official matching artifacts.
    CMD="hf download sakamakismile/10Eros-Max-beta2-NVFP4 10Eros_Max_h3_fl2va_beta2_pruned_nvfp4.safetensors --revision 7c6e6d55251d0b8fa08a796b90391a7366c644cf --local-dir /models/unet && \
         hf download sakamakismile/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4 qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors --revision 2814607c9e6034e2cf2c76da82f996d179567551 --local-dir /models/text_encoders && \
         hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors --revision 4cc1d817b6184899b41293954329f576cb5ae86b --local-dir /models && \
         hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_audio_vae_fp32.safetensors --revision 4cc1d817b6184899b41293954329f576cb5ae86b --local-dir /models"
    run_downloader "$COMFY_MODELS_DIR/.h3_setup_state" "$SETUP_VERSION:minimax_h3_10eros_beta2_heretic_nvfp4" "$COMFY_MODELS_DIR:/models" "$CMD"

    log_step "PHASE 10.2: LTX-2.3 Video Specialist (UNCENSORED 10Eros 1.5 finetune)"
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
    # 10Eros 1.5 restores the explicit motion/anatomy access of 1.2 while retaining
    # the more stable structure and base-model prompting behavior of 1.4. GPU-adaptive
    # quant: Q8_0 (~23 GiB) on big cards (>=80 GiB), else Q4_K_M (~14 GiB).
    LTX_QUANT="Q4_K_M"
    if python3 -c "import sys; sys.exit(0 if float('${MIN_VRAM}')>=80 else 1)" 2>/dev/null; then LTX_QUANT="Q8_0"; fi
    log_step "  -> 10Eros video unet quant: ${LTX_QUANT} (min per-GPU VRAM ${MIN_VRAM} GiB)"
    CMD="hf download vantagewithai/LTX2.3-10Eros-1.5-GGUF --include '*${LTX_QUANT}*.gguf' --local-dir /models/unet && \
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
    run_downloader "$COMFY_MODELS_DIR/.ltx_setup_state" "$SETUP_VERSION:ltx_10eros_v15_ablit_${LTX_QUANT}" "$COMFY_MODELS_DIR:/models" "$CMD"

    :
fi

log_step "Setup complete! All Dockerfiles will automatically rebuild if changed, and partial downloads will automatically resume."
