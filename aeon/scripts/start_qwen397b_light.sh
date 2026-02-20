#!/bin/bash
# Wrapper: Qwen3.5 IQ4_XS — LIGHT mode (GPU1 leaves 48GB free)
docker rm -f aeon_qwen397b >/dev/null 2>&1
export QWEN_MODE=light
exec bash "$(dirname "$0")/start_qwen397b_q6k.sh"
