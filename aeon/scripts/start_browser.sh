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

echo "[Browser] Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    -v $PROJECT_ROOT:/app \
    -p $PORT:8030 \
    -e PORT=8030 \
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
