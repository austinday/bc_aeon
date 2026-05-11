#!/bin/bash
set -e

echo "Building browser service Docker image..."
# Match the image name expected by aeon/scripts/start_browser.sh
docker build -t aeon_browser_service:latest -f aeon/services/browser/Dockerfile aeon/services/browser/

echo "Browser service image built successfully."