#!/bin/bash
set -euo pipefail

# Compatibility entrypoint for operators. The Python helper owns the entire
# lifecycle boundary: exact receipts, random identity, CPU-only limits, private
# authentication, and authenticated semantic health. This wrapper intentionally
# contains no Docker/container discovery or lifecycle commands.
echo "[Browser] Using software WebGL; no GPU device is exposed to Chromium."
exec /usr/bin/python3 -m aeon.scripts.browser_service ensure
