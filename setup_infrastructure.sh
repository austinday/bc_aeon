#!/bin/bash
set -e

echo "========================================"
echo "      AEON INFRASTRUCTURE SETUP         "
echo "========================================"

# 1. Create Model Cache Directory
echo "\n[1/4] Creating Model Cache Directory..."
mkdir -p ~/bc_aeon/aeon_models
echo "Dir: ~/bc_aeon/aeon_models"

# 2. Build Base Images (Project Environments)
echo "\n[2/4] Building Project Base Images (Cpu/Cuda)..."
docker build -t aeon_base:py3.10-cuda12.1 \
  --build-arg PYTHON_VERSION_MINOR=10 \
  --build-arg CUDA_VERSION=12.1.1 \
  --build-arg PYTORCH_CUDA_SUFFIX=cu121 \
  .

# 3. Build Vision Image (Multimedia Tools)
echo "\n[3/4] Building Vision Docker Image (aeon_vision)..."
docker build -t aeon_vision:latest -f Dockerfile.vision .

# 4. Download Models (Host Side)
echo "\n[4/4] Downloading Image Gen Models..."
echo "Targets: Flux (High VRAM), Pony (Creative), DreamShaper (Low VRAM)."
echo "Note: Interactive auth required for Flux/Pony if token not found."
pip install huggingface_hub
python3 aeon/scripts/download_models.py

echo "\n========================================"
echo "       SETUP COMPLETE                   "
echo "========================================"
