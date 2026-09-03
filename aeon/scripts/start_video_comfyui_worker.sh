#!/bin/bash
set -euo pipefail

FLEET_LOW_PRIORITY=/home/aday/bin/fleet-low-priority
DOCKER=/usr/bin/docker
CACHE_ROOT=/home/aday/.local/state/fleet-compute/cache/aeon-video-comfyui
SCRATCH_ROOT=/home/aday/.local/state/fleet-compute/runs
IMAGE_ID=sha256:e87d7bcd4da3b5826e03740585ee22a5c78bf5f4468e881495375798f677ba8d

[ -x "$FLEET_LOW_PRIORITY" ] && [ -x "$DOCKER" ] || exit 2
case "${GPU_AGENT_CLAIM_ID:-}" in gc-*) ;; *) exit 2 ;; esac
case "${CUDA_VISIBLE_DEVICES:-}" in GPU-*) ;; *) exit 2 ;; esac
case "${AEON_VIDEO_RUNTIME_ID:-}" in fr-[0-9a-f][0-9a-f]*) ;; *) exit 2 ;; esac
case "${AEON_VIDEO_CONTAINER_NAME:-}" in aeon_video_comfyui_fr_[0-9a-f]*) ;; *) exit 2 ;; esac
case "${AEON_VIDEO_REMOTE_PORT:-}" in *[!0-9]*|'') exit 2 ;; esac

python3 - <<'PY'
import os
from pathlib import Path, PurePosixPath

cache = PurePosixPath('/home/aday/.local/state/fleet-compute/cache/aeon-video-comfyui')
scratch = PurePosixPath('/home/aday/.local/state/fleet-compute/runs')
for key in (
    'AEON_VIDEO_LAUNCHER', 'AEON_VIDEO_ALLOCATOR_CAP', 'AEON_VIDEO_H3_MODEL',
    'AEON_VIDEO_H3_ENCODER', 'AEON_VIDEO_H3_VIDEO_VAE',
    'AEON_VIDEO_H3_AUDIO_VAE', 'AEON_VIDEO_LTX_MODEL',
    'AEON_VIDEO_LTX_ENCODER', 'AEON_VIDEO_LTX_CONNECTORS',
    'AEON_VIDEO_LTX_VAE',
):
    value = PurePosixPath(os.environ[key])
    assert value.is_absolute() and '..' not in value.parts
    value.relative_to(cache)
    assert Path(value).is_file() and not Path(value).is_symlink()
output = PurePosixPath(os.environ['AEON_VIDEO_OUTPUT_DIR'])
assert output.is_absolute() and '..' not in output.parts
output.relative_to(scratch)
assert Path(output).is_dir() and not Path(output).is_symlink()
limit = float(os.environ['GPU_MEM_LIMIT_GB'])
reserve = float(os.environ.get('GPU_RESERVE_GB', '6'))
port = int(os.environ['AEON_VIDEO_REMOTE_PORT'])
assert 0 < limit <= 40 and reserve >= 6 and 1024 <= port <= 65535
PY

observed_image="$($DOCKER image inspect --format '{{.Id}}' "$IMAGE_ID" 2>/dev/null || true)"
[ "$observed_image" = "$IMAGE_ID" ] || exit 3
if $DOCKER container inspect "$AEON_VIDEO_CONTAINER_NAME" >/dev/null 2>&1; then
    exit 4
fi

exec "$FLEET_LOW_PRIORITY" "$DOCKER" run -d \
    --oom-score-adj 1000 --cpu-shares 2 --blkio-weight 10 \
    --label owner=aday \
    --label com.bc_aeon.component=video-comfyui \
    --label "com.bc_aeon.claim=${GPU_AGENT_CLAIM_ID}" \
    --label "com.bc_aeon.runtime=${AEON_VIDEO_RUNTIME_ID}" \
    --name "$AEON_VIDEO_CONTAINER_NAME" \
    --gpus "device=${CUDA_VISIBLE_DEVICES}" \
    -e "GPU_AGENT_CLAIM_ID=${GPU_AGENT_CLAIM_ID}" \
    -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
    -e "GPU_MEM_LIMIT_GB=${GPU_MEM_LIMIT_GB}" \
    -e "GPU_RESERVE_GB=${GPU_RESERVE_GB}" \
    -e PYTHONPATH=/workspace/aeon_runtime \
    --shm-size=8gb \
    -p "127.0.0.1:${AEON_VIDEO_REMOTE_PORT}:8188" \
    -v "$FLEET_LOW_PRIORITY:/usr/local/bin/fleet-low-priority:ro" \
    -v "$AEON_VIDEO_ALLOCATOR_CAP:/workspace/aeon_runtime/sitecustomize.py:ro" \
    -v "$AEON_VIDEO_H3_MODEL:/workspace/ComfyUI/models/unet/10Eros_Max_h3_fl2va_beta2_pruned_nvfp4.safetensors:ro" \
    -v "$AEON_VIDEO_H3_ENCODER:/workspace/ComfyUI/models/text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors:ro" \
    -v "$AEON_VIDEO_H3_VIDEO_VAE:/workspace/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors:ro" \
    -v "$AEON_VIDEO_H3_AUDIO_VAE:/workspace/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors:ro" \
    -v "$AEON_VIDEO_LTX_MODEL:/workspace/ComfyUI/models/unet/10Eros_v1.5-Q8_0.gguf:ro" \
    -v "$AEON_VIDEO_LTX_ENCODER:/workspace/ComfyUI/models/text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf:ro" \
    -v "$AEON_VIDEO_LTX_CONNECTORS:/workspace/ComfyUI/models/text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors:ro" \
    -v "$AEON_VIDEO_LTX_VAE:/workspace/ComfyUI/models/vae/ltx-2.3-22b-dev_video_vae.safetensors:ro" \
    -v "$AEON_VIDEO_OUTPUT_DIR:/workspace/ComfyUI/output" \
    --entrypoint /usr/local/bin/fleet-low-priority \
    "$IMAGE_ID" \
    python main.py --listen --port 8188 --reserve-vram "$GPU_RESERVE_GB"
