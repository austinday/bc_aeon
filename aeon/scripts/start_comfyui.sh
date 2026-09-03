#!/bin/bash
set -euo pipefail

AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
MODELS_DIR="$AEON_HOME/models/comfyui"
OUTPUT_DIR="$AEON_HOME/temp/comfyui_output"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEET_LOW_PRIORITY="/home/aday/bin/fleet-low-priority"
mkdir -p "$OUTPUT_DIR"

[ -x "$FLEET_LOW_PRIORITY" ] || {
    echo "ERROR: renter-yielding launcher is unavailable." >&2
    exit 2
}

case "${GPU_AGENT_CLAIM_ID:-}" in
    gc-*) ;;
    *) echo "ERROR: GPU_AGENT_CLAIM_ID must be a coordinator claim." >&2; exit 2 ;;
esac
case "${CUDA_VISIBLE_DEVICES:-}" in
    GPU-*) ;;
    *) echo "ERROR: CUDA_VISIBLE_DEVICES must be the leased GPU UUID." >&2; exit 2 ;;
esac
if ! [[ "${GPU_MEM_LIMIT_GB:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: GPU_MEM_LIMIT_GB must be a positive numeric lease cap." >&2
    exit 2
fi
if ! [[ "${GPU_RESERVE_GB:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: GPU_RESERVE_GB must be numeric and at least 6." >&2
    exit 2
fi
python3 -c 'import os; assert float(os.environ["GPU_MEM_LIMIT_GB"]) > 0; assert float(os.environ["GPU_RESERVE_GB"]) >= 6'

# Pre-check: image must have been built during setup_environment.sh
if ! docker image inspect aeon_comfyui:latest >/dev/null 2>&1; then
    echo "ERROR: aeon_comfyui:latest image not found. Run setup_environment.sh first."
    exit 1
fi

echo "Starting ComfyUI container..."
if docker inspect aeon_comfyui >/dev/null 2>&1; then
    echo "ERROR: refusing to replace an existing container; Fleet must reconcile its exact runtime identity." >&2
    exit 3
fi

echo "Using leased GPU UUID: ${CUDA_VISIBLE_DEVICES} (cap ${GPU_MEM_LIMIT_GB}GB)"

"$FLEET_LOW_PRIORITY" docker run -d \
    --oom-score-adj 1000 --cpu-shares 2 --blkio-weight 10 \
    --label owner=aday \
    --label com.bc_aeon.component=comfyui \
    --label "com.bc_aeon.claim=${GPU_AGENT_CLAIM_ID}" \
    --name aeon_comfyui \
    --gpus "device=${CUDA_VISIBLE_DEVICES}" \
    -e "GPU_AGENT_CLAIM_ID=${GPU_AGENT_CLAIM_ID}" \
    -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    -e "GPU_MEM_LIMIT_GB=${GPU_MEM_LIMIT_GB}" \
    -e "GPU_RESERVE_GB=${GPU_RESERVE_GB}" \
    -e PYTHONPATH=/workspace/aeon_runtime \
    --shm-size=8gb \
    -p 127.0.0.1:8188:8188 \
    -v "$FLEET_LOW_PRIORITY:/usr/local/bin/fleet-low-priority:ro" \
    -v "$SCRIPT_DIR/comfyui_sitecustomize.py:/workspace/aeon_runtime/sitecustomize.py:ro" \
    -v "$MODELS_DIR/unet:/workspace/ComfyUI/models/unet" \
    -v "$MODELS_DIR/text_encoders:/workspace/ComfyUI/models/text_encoders" \
    -v "$MODELS_DIR/vae:/workspace/ComfyUI/models/vae" \
    -v "$MODELS_DIR/pulid:/workspace/ComfyUI/models/pulid" \
    -v "$MODELS_DIR/clip:/workspace/ComfyUI/models/clip" \
    -v "$MODELS_DIR/insightface:/workspace/ComfyUI/models/insightface" \
    -v "$OUTPUT_DIR:/workspace/ComfyUI/output" \
    --entrypoint /usr/local/bin/fleet-low-priority \
    aeon_comfyui:latest \
    python main.py --listen --port 8188 --reserve-vram "$GPU_RESERVE_GB"

echo "ComfyUI is starting! It will be available at http://localhost:8188"
echo "Check logs with: docker logs -f aeon_comfyui"
