#!/usr/bin/env bash
# Operator-only compatibility entry point. Container identity, caps, and
# receipts are enforced by the protected Python helper; model-facing tools never
# invoke this script or gain Docker authority.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -B -m aeon.scripts.searxng_service
