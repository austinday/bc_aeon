#!/usr/bin/env bash
set -euo pipefail

# Qwen creation moved into aeon.core.qwen_runtime so the durable launch nonce,
# immutable container ID, exact Docker receipt, final coordinator/ACL gate and
# release journal are one crash-consistent transaction.  Keeping a second shell
# launcher would create an unreceipted lifecycle path.
echo "[adaptive-vllm] disabled: use Aeon's coordinator-managed local Qwen runtime" >&2
exit 64
