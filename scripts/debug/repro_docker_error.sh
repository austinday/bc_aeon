#!/bin/bash
echo "Testing malformed docker run command..."

# This mimics the exact syntax in aeon/scripts/start_comfyui.sh
# Note the trailing spaces after the backslash on the --gpus line
docker run -d \
    --name aeon_comfyui_test \
    --gpus "\"device=1\"" \    --shm-size=8gb \
    aeon_comfyui:latest 2>&1

if [ $? -ne 0 ]; then
    echo "Caught expected error!"
else
    echo "Command unexpectedly succeeded."
    docker rm -f aeon_comfyui_test >/dev/null 2>&1 || true
fi