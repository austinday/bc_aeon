#!/bin/bash
# Wrapper: Qwen3.5 IQ4_XS — MEDIUM mode (GPU1 leaves 24GB free)
docker rm -f aeon_qwen397b >/dev/null 2>&1
export QWEN_MODE=medium
exec bash "$(dirname "$0")/start_qwen397b_q6k.sh"
