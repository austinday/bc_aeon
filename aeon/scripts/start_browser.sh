#!/bin/bash
set -e

CONTAINER_NAME="aeon_browser"
IMAGE_NAME="aeon_browser_service:latest"
PORT=8030
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AEON_STATE_DIR="${AEON_HOME:-$HOME/.aeon}"
TOKEN_HOST_FILE="${AEON_BROWSER_TOKEN_FILE:-$AEON_STATE_DIR/browser_api_token}"
TOKEN_CONTAINER_FILE="/run/secrets/aeon_browser_token"

# Create the service login credential without ever placing it in argv or logs.
# The browser API is unusable without this file, and both host and container
# validate that it is private.
umask 077
mkdir -p "$AEON_STATE_DIR"
chmod 700 "$AEON_STATE_DIR"
python3 - "$TOKEN_HOST_FILE" <<'PY'
import os
import secrets
import stat
import sys

path = sys.argv[1]
parent = os.path.dirname(path) or "."
os.makedirs(parent, mode=0o700, exist_ok=True)
try:
    info = os.stat(path, follow_symlinks=False)
except FileNotFoundError:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # another simultaneous launcher created it
        info = os.stat(path, follow_symlinks=False)
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(48) + "\n")
        info = os.stat(path, follow_symlinks=False)
if not stat.S_ISREG(info.st_mode):
    raise SystemExit("browser API token must be a regular file, not a symlink")
os.chmod(path, 0o600)
with open(path, "r", encoding="utf-8") as handle:
    if len(handle.read().strip().encode("utf-8")) < 32:
        raise SystemExit("browser API token must be at least 32 bytes")
PY

authenticated_healthcheck() {
    python3 - "$TOKEN_HOST_FILE" "http://127.0.0.1:$PORT/health" <<'PY'
import json
import sys
from urllib.request import Request, urlopen

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    token = handle.read().strip()
request = Request(sys.argv[2], headers={"Authorization": f"Bearer {token}"})
try:
    with urlopen(request, timeout=2) as response:
        body = json.load(response)
    if (response.status == 200 and body.get("auth_required") is True
            and body.get("api_version") == "human_v6"):
        raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PY
}

echo "[Browser] Checking if browser service is running..."
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if authenticated_healthcheck; then
        echo "[Browser] Authenticated service already running on localhost:$PORT."
        exit 0
    fi
    echo "[Browser] Existing service is not the current authenticated human-v6 API; replacing it securely."
fi

if ! docker image inspect $IMAGE_NAME >/dev/null 2>&1; then
    echo "[Browser] ERROR: $IMAGE_NAME image not found. You must run setup_environment.sh first."
    exit 1
fi
IMAGE_AUTH_LABEL="$(docker image inspect -f '{{ index .Config.Labels "com.bc_aeon.browser.auth" }}' "$IMAGE_NAME")"
if [ "$IMAGE_AUTH_LABEL" != "required-v1" ]; then
    echo "[Browser] ERROR: refusing to launch a stale unauthenticated image."
    echo "[Browser] Rebuild $IMAGE_NAME with setup_environment.sh first."
    exit 1
fi
IMAGE_API_LABEL="$(docker image inspect -f '{{ index .Config.Labels "com.bc_aeon.browser.api" }}' "$IMAGE_NAME")"
if [ "$IMAGE_API_LABEL" != "human-v6" ]; then
    echo "[Browser] ERROR: refusing to launch an image without the human-v6 capability marker."
    echo "[Browser] Rebuild $IMAGE_NAME with setup_environment.sh first."
    exit 1
fi

echo "[Browser] Removing old container if exists..."
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    OLD_OWNER="$(docker inspect -f '{{ index .Config.Labels "owner" }}' "$CONTAINER_NAME")"
    OLD_AUTH="$(docker inspect -f '{{ index .Config.Labels "com.bc_aeon.browser.auth" }}' "$CONTAINER_NAME")"
    OLD_API="$(docker inspect -f '{{ index .Config.Labels "com.bc_aeon.browser.api" }}' "$CONTAINER_NAME")"
    if [ "$OLD_OWNER" != "aday" ] || [ "$OLD_AUTH" != "required-v1" ] || \
            { [ "$OLD_API" != "human-v5" ] && [ "$OLD_API" != "human-v6" ]; }; then
        echo "[Browser] ERROR: refusing to replace an unrecognized container named $CONTAINER_NAME." >&2
        exit 1
    fi
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# Persistent browser profile on the HOST so logins/cookies survive container
# restarts (the headed Chromium runs as a persistent context in /profiles).
PROFILE_HOST_DIR="$AEON_STATE_DIR/browser_profiles"
mkdir -p "$PROFILE_HOST_DIR"

# Browser rendering is CPU-only. Screenshot understanding is performed by the
# coordinator-leased Qwen3.8 model; Chromium must never grab a renter GPU merely
# to alter its WebGL fingerprint.
echo "[Browser] Using software WebGL; no GPU device is exposed to Chromium."

echo "[Browser] Starting container (headed Chromium under Xvfb, persistent profile)..."
docker run -d --label owner=aday --label com.bc_aeon.component=browser --name $CONTAINER_NAME \
    -p 127.0.0.1:$PORT:8030 \
    -e PORT=8030 \
    -e AEON_BROWSER_PROFILE=/profiles/default \
    -e AEON_BROWSER_TOKEN_FILE="$TOKEN_CONTAINER_FILE" \
    -v "$PROFILE_HOST_DIR":/profiles \
    -v "$TOKEN_HOST_FILE":"$TOKEN_CONTAINER_FILE":ro \
    --shm-size=2g \
    $IMAGE_NAME

echo "[Browser] Waiting for service to become healthy (timeout 60s)..."
for i in {1..60}; do
    if authenticated_healthcheck; then
        echo "[Browser] Service is up, authenticated, and localhost-only!"
        exit 0
    fi
    sleep 1
done

echo "[Browser] ERROR: Service failed to start in time."
docker logs $CONTAINER_NAME
exit 1
