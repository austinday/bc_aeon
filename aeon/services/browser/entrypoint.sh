#!/bin/bash
set -e

echo "Starting Xvfb on display :99..."
Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99

# Wait for X server to be ready
sleep 2

echo "Starting Uvicorn on port ${PORT:-8000}..."
# Run from /app where server.py is located
exec python3 -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}