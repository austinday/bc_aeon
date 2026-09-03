#!/bin/bash
set -e

# The receipted service runs as the unprivileged owner with a read-only image.
# Its only runtime-writable locations are the bounded /tmp and /run tmpfs mounts
# plus the explicit persistent /profiles mount.
mkdir -p "${XDG_RUNTIME_DIR:-/run/user}" "${HOME:-/profiles/.browser-home}"
chmod 700 "${XDG_RUNTIME_DIR:-/run/user}" "${HOME:-/profiles/.browser-home}"

echo "Starting Xvfb on display :99..."
# 1920x1080 is the single most common real-world screen resolution; the previous
# 1280x1024 (5:4) is a CRT-era size almost no modern device reports, which is
# itself a fingerprint. A common resolution + a larger viewport (more elements
# visible per observation, fewer scrolls) is both more human and more capable.
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99

# Wait for X server to be ready
sleep 2

# Xvfb is only a display server; without a window manager Chrome ignores
# --start-maximized and opens at roughly 945x973. A tiny real WM gives Chrome the
# full common 1920x1080 desktop, fitting more controls into each observation and
# keeping screen/window geometry internally consistent like a normal desktop.
echo "Starting Openbox window manager..."
openbox >/tmp/openbox.log 2>&1 &
sleep 1

echo "Starting Uvicorn on port ${PORT:-8000}..."
# Run from /app where server.py is located
exec python3 -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
