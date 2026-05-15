#!/bin/bash
set -e

echo "Starting Xvfb on display :99..."
# Run Xvfb in the background. 
# -ac disables host-based access control
Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99

# Wait for X server to be ready
sleep 2

echo "Xvfb is running. Moving to browser service directory..."
# Ensure we are in the directory where server.py is located
cd /app/browser_service

echo "Starting Uvicorn on port ${PORT:-8000}..."
exec python3 -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}