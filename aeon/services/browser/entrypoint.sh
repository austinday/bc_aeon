#!/bin/bash
set -e

echo "Starting Xvfb on display :99..."
# 1920x1080 is the single most common real-world screen resolution; the previous
# 1280x1024 (5:4) is a CRT-era size almost no modern device reports, which is
# itself a fingerprint. A common resolution + a larger viewport (more elements
# visible per observation, fewer scrolls) is both more human and more capable.
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99

# Wait for X server to be ready
sleep 2

echo "Starting Uvicorn on port ${PORT:-8000}..."
# Run from /app where server.py is located
exec python3 -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}