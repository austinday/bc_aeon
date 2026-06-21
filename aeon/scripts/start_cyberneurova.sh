#!/bin/bash
# =============================================================================
# Start the DeepSeek-V4-Flash fork server (aeon_ds4:latest = Fringe210 CUDA fork)
# for the CyberNeurova-DeepSeek-V4-Flash-abliterated GGUF (deepseek4 arch).
#
# This GGUF is a 284B MoE at ~153 GiB of weights: it CANNOT fit one 96 GB card,
# so it always runs split across GPU0 + GPU1. The Fringe210 fork implements the
# V4 ops as CUDA kernels, so the graph stays on the GPU (graph splits ~171, not
# ~11901) -- measured ~104 tok/s prefill on a long prompt.
#
# DEFAULT SPLIT IS GPU0-WEIGHTED to maximize FREE VRAM on GPU1 (e.g. to co-host
# another model / ComfyUI there). This does NOT affect generation speed: a layer
# split only relocates weights between cards; V4 decode is bound by the
# indexer/sinkhorn/expert-routing kernels, not by which card holds which layer.
#
# CALIBRATION NOTE: 46,54 loaded fine; 60,40 tried to put ~97.6 GiB on GPU0 and
# OOM'd (card has only ~95 GiB usable). So the max safe GPU0 weighting is between
# those. Default is 54,46 (safe). To free MORE on GPU1, nudge the first number up
# by 1-2 at a time, watching `nvidia-smi -l 1` DURING a prompt; if GPU0 OOMs at
# load, you've gone one step too far -- back off.
#
# Tuning knobs (all env-overridable; watch utilization DURING a prompt, not at
# idle -- at idle both cards read P8 / 0% and tell you nothing):
#   TENSOR_SPLIT  weight ratio "GPU0,GPU1". RAISE first number => more free GPU1.
#                 Hard ceiling ~56,44 before GPU0 OOMs at this CTX/UBATCH.
#   UBATCH        physical microbatch = prefill speed vs compute-buffer size.
#                 The compute buffer sits on GPU0, so LOWERING it (e.g. 256) buys
#                 GPU0 headroom and lets you push TENSOR_SPLIT a notch higher.
#   CTX           total context. Compressed KV makes this cheap; 256k default.
# =============================================================================
set -e

CONTAINER_NAME='aeon_cyberneurova'
IMAGE_NAME='aeon_ds4:latest'
PORT=8021
AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
MODELS_DIR="$AEON_HOME/models/gguf_models/CyberNeurova"

# Tunable parameters (env-overridable)
N_GPU_LAYERS=${NGL:-99}              # offload all layers to GPU
PARALLEL_SLOTS=${PARALLEL:-1}        # agent issues one request at a time
CTX_SIZE=${CTX:-262144}              # 256k; compressed KV makes this ~free
BATCH_SIZE=${BATCH:-2048}            # logical batch (prompt scheduling)
UBATCH_SIZE=${UBATCH:-512}           # physical microbatch -> prefill speed
TENSOR_SPLIT=${TENSOR_SPLIT:-54,46}  # GPU0-weighted (safe); frees GPU1 vs 46,54
PHYSICAL_CORES=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l 2>/dev/null || nproc)

MODEL_FILE=$(cd "${MODELS_DIR}" 2>/dev/null && find . -name "*.gguf" | head -1 | sed 's|^\./||')
if [ -z "$MODEL_FILE" ]; then
    echo "[CyberNeurova-V4] ERROR: No .gguf files found in ${MODELS_DIR}"
    exit 1
fi
echo "[CyberNeurova-V4] Using model file: ${MODEL_FILE}"
echo "[CyberNeurova-V4] CTX=${CTX_SIZE} UBATCH=${UBATCH_SIZE} TENSOR_SPLIT=${TENSOR_SPLIT} (GPU0-weighted: frees GPU1)"

# Short-circuit: already running and healthy?
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || true)" = "200" ]; then
        echo "[CyberNeurova-V4] Already running and healthy on port $PORT."
        exit 0
    fi
fi
docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true

# Poll for health. 0 = healthy, 1 = container died (crash/OOM), 2 = timeout.
wait_for_health() {
    local count=0
    while true; do
        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            return 1
        fi
        if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health 2>/dev/null || true)" = "200" ]; then
            return 0
        fi
        sleep 5
        count=$((count+1))
        if [ $count -ge 180 ]; then
            return 2
        fi
        if [ $((count % 6)) -eq 0 ]; then
            echo "[CyberNeurova-V4] Still loading... ($((count*5))s)"
        fi
    done
}

# 284B / ~153 GiB cannot fit one card -> always split GPU0 + GPU1, GPU0-weighted.
echo "[CyberNeurova-V4] Starting dual-GPU server (GPU0-weighted split ${TENSOR_SPLIT}, ubatch ${UBATCH_SIZE}, ctx ${CTX_SIZE})..."
docker run -d \
    --name $CONTAINER_NAME \
    --gpus '"device=0,1"' \
    -p ${PORT}:8001 \
    -v "${MODELS_DIR}:/models:ro" \
    --shm-size=16g \
    --ulimit memlock=-1 \
    $IMAGE_NAME \
    --model "/models/${MODEL_FILE}" \
    --n-gpu-layers ${N_GPU_LAYERS} \
    --split-mode layer \
    --main-gpu 0 \
    --tensor-split ${TENSOR_SPLIT} \
    --parallel ${PARALLEL_SLOTS} \
    --ctx-size ${CTX_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --ubatch-size ${UBATCH_SIZE} \
    --threads ${PHYSICAL_CORES} \
    --flash-attn on \
    --jinja \
    --host 0.0.0.0 \
    --port 8001 \
    --metrics

echo "[CyberNeurova-V4] Waiting for server to load (this may take several minutes)..."
if wait_for_health; then
    echo "[CyberNeurova-V4] Server ready on port $PORT (GPU0+GPU1, split ${TENSOR_SPLIT}, ctx ${CTX_SIZE})."
    exit 0
fi
echo "[CyberNeurova-V4] ERROR: server failed to start. Diagnosis from the logs below:"
echo "[CyberNeurova-V4] - 'CUDA error: out of memory' on CUDA0 => GPU0 over-loaded; LOWER the first"
echo "[CyberNeurova-V4]   TENSOR_SPLIT number (e.g. 50,50) and/or LOWER UBATCH (e.g. 256)."
echo "[CyberNeurova-V4] - 'GGML_ASSERT ... .cu' => an unimplemented CUDA op on this fork (paste it)."
docker logs $CONTAINER_NAME --tail 40
exit 1
