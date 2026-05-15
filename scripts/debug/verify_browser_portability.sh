#!/bin/bash
set -e

# Path to the start_browser.sh script
START_SCRIPT="/home/aday/bc_aeon/aeon/scripts/start_browser.sh"
HEALTH_URL="http://localhost:8030/health"

echo "[Verify] Testing browser portability..."
echo "[Verify] Moving to /tmp to simulate running from another directory..."
cd /tmp

echo "[Verify] Executing $START_SCRIPT..."
if bash "$START_SCRIPT"; then
    echo "[Verify] start_browser.sh executed successfully."
else
    echo "[Verify] ERROR: start_browser.sh failed."
    exit 1
fi

echo "[Verify] Checking health endpoint $HEALTH_URL..."
if curl -s "$HEALTH_URL" > /dev/null; then
    echo "[Verify] SUCCESS: Browser service is healthy and reachable!"
else
    echo "[Verify] ERROR: Browser service is NOT healthy."
    docker logs aeon_browser
    exit 1
fi