#!/bin/bash
set -e

echo "Cleaning up old containers..."
docker rm -f aeon-browser-val 2>/dev/null || true

echo "Starting browser service container..."
# Run in background, map port 8030. 
# Removed -u $(id -u):$(id -g) because the image was built as root and Camoufox binaries are in /root/.camoufox
docker run -d --name aeon-browser-val \
    -p 8030:8030 \
    aeon_browser_service:latest

# Wait for the service to be healthy with a timeout
echo "Waiting for browser service to be ready..."
for i in {1..15}; do
    if curl --output /dev/null --silent --fail http://localhost:8030/health 2>/dev/null; then
        echo " Browser service is UP!"
        break
    fi
    printf '.'
    sleep 2
    if [ $i -eq 15 ]; then
        echo -e "\nError: Browser service failed to start. Container logs:"
        docker logs aeon-browser-val
        docker rm -f aeon-browser-val
        exit 1
    fi
done

echo "Running validation script..."
# Use aeon_downloader and ensure requests is installed
# Run as root to allow pip install
docker run --rm --network host \
    -v $(pwd):/app \
    -w /app \
    aeon_downloader:latest \
    sh -c "pip install requests && python3 scripts/debug/validate_browser.py"

echo "Cleaning up..."
docker stop aeon-browser-val
docker rm aeon-browser-val

echo "Validation process finished."
