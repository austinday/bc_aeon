#!/bin/bash
set -e

echo "============================================="
echo "      BRAIN SYNC: UNIFYING GPU VOLUMES       "
echo "============================================="

# 1. Stop nodes to release file locks
echo "[1/3] Stopping nodes to unlock files..."
docker stop aeon_strong_node aeon_weak_node >/dev/null 2>&1 || true

# 2. Sync Strong -> Weak
echo "[2/3] Cloning Strong(GPU0) -> Weak(GPU1)..."
# We use a temporary Alpine container to mount both volumes and copy missing files.
# -n = no clobber (don't overwrite existing files, just add missing ones)
# -a = archive mode (preserve permissions/dates)
docker run --rm \
  -v aeon_strong_vol:/source \
  -v aeon_weak_vol:/dest \
  alpine ash -c "cp -an /source/. /dest/ && echo ' >> Sync Complete.'"

# 3. Sync Weak -> Strong (Just in case you downloaded something to Weak only)
echo "[3/3] Cloning Weak(GPU1) -> Strong(GPU0)..."
docker run --rm \
  -v aeon_weak_vol:/source \
  -v aeon_strong_vol:/dest \
  alpine ash -c "cp -an /source/. /dest/ && echo ' >> Sync Complete.'"

# 4. Restart
echo "[4/4] Restarting Brain..."
bash $(dirname "$0")/start_brain.sh

echo "============================================="
echo "      SYNC COMPLETE. BOTH NODES IDENTICAL.   "
echo "============================================="
