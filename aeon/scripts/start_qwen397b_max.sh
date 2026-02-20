#!/bin/bash
# Wrapper: Qwen3.5 IQ4_XS — MAX mode (both GPUs full)
# Kill existing container first to allow mode switching
docker rm -f aeon_qwen397b >/dev/null 2>&1
export QWEN_MODE=max
exec bash "$(dirname "$0")/start_qwen397b_q6k.sh"
