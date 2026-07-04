#!/bin/bash
set -e

CONTAINER_NAME="aeon_browser"
IMAGE_NAME="aeon_browser_service:latest"
PORT=8030
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[Browser] Checking if browser service is running..."
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[Browser] Service already running on port $PORT."
    exit 0
fi

if ! docker image inspect $IMAGE_NAME >/dev/null 2>&1; then
    echo "[Browser] ERROR: $IMAGE_NAME image not found. You must run setup_environment.sh first."
    exit 1
fi

echo "[Browser] Removing old container if exists..."
docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true

# Persistent browser profile on the HOST so logins/cookies survive container
# restarts (the headed Chromium runs as a persistent context in /profiles).
PROFILE_HOST_DIR="${AEON_HOME:-$HOME/.aeon}/browser_profiles"
mkdir -p "$PROFILE_HOST_DIR"

# GPU-accelerated WebGL when the host has an NVIDIA GPU: without it, Chrome under
# Xvfb uses the SwiftShader SOFTWARE renderer, which is a datacenter/headless
# fingerprint tell ("WebGL Renderer: SwiftShader"). With the GPU it reports the
# real NVIDIA renderer like a normal desktop. WebGL is lightweight, so sharing a
# GPU with the model is fine. Falls back to software when no GPU is present.
GPU_RUN_ARGS=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_RUN_ARGS="--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e AEON_BROWSER_GPU=1"
    echo "[Browser] NVIDIA GPU detected -> GPU-accelerated WebGL (real renderer)."
else
    echo "[Browser] No NVIDIA GPU -> software WebGL (SwiftShader)."
fi

echo "[Browser] Starting container (headed Chromium under Xvfb, persistent profile)..."
docker run -d --name $CONTAINER_NAME \
    $GPU_RUN_ARGS \
    -p $PORT:8030 \
    -e PORT=8030 \
    -e AEON_BROWSER_PROFILE=/profiles/default \
    -v "$PROFILE_HOST_DIR":/profiles \
    --shm-size=2g \
    $IMAGE_NAME

echo "[Browser] Waiting for service to become healthy (timeout 60s)..."
for i in {1..60}; do
    if curl -s http://localhost:$PORT/health >/dev/null; then
        echo "[Browser] Service is up and healthy!"
        exit 0
    fi
    sleep 1
done

echo "[Browser] ERROR: Service failed to start in time."
docker logs $CONTAINER_NAME
exit 1
