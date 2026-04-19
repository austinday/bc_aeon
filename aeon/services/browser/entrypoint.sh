#!/bin/bash
set -e

echo "Starting Xvfb on display :99..."
# Run Xvfb in the background. 
# -ac disables host-based access control (fixes permission denied errors in Docker)
Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99

# Wait a second to ensure the X server socket is fully established
sleep 2

echo "Xvfb is running. Starting Uvicorn..."
exec uvicorn server:app --host 0.0.0.0 --port 8030
