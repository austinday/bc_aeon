#!/bin/bash
set -e

# 1. Build the image using the browser service directory as context
# This ensures that 'COPY . .' in the Dockerfile only copies the browser service files
docker build -t aeon-browser-test -f aeon/services/browser/Dockerfile aeon/services/browser

# 2. Run the container on port 8001
CONTAINER_NAME="aeon_browser_measure"
docker rm -f $CONTAINER_NAME || true

echo "Starting browser service on port 8001..."
# We run the image as-is. The Dockerfile already sets the WORKDIR and ENTRYPOINT.
# We map host 8001 to container 8000.
docker run -d \
  --name $CONTAINER_NAME \
  -p 8001:8000 \
  -e PORT=8000 \
  aeon-browser-test

# Wait for the service to be healthy
echo "Waiting for health check..."
until curl -s http://localhost:8001/health | grep -q "ok"; do
  echo "Still waiting for browser service..."
  sleep 2
done
echo "Service is healthy!"

# 3. Run the measurement script
python3 scripts/debug/measure_browser_resources.py

# 4. Cleanup
echo "Cleaning up..."
docker rm -f $CONTAINER_NAME