#!/usr/bin/env bash
# The historical direct-Docker MTP sweep is intentionally disabled.
#
# Benchmark GPU work must use the same audited local-only coordinator receipt,
# immutable artifact/image/source identity, exact container-ID journal,
# heartbeat, final ACL/resource/claim gate, and verified stop/release lifecycle
# as production Qwen. No benchmark-mode adapter currently implements that
# contract, so accepting caller-supplied claim/UUID strings would be unsafe.
set -euo pipefail

echo "[mtp-sweep] ERROR: direct Qwen benchmark launching is disabled." >&2
echo "[mtp-sweep] A reviewed benchmark mode must be added to aeon.core.qwen_runtime before GPU sweeps can run." >&2
exit 64
