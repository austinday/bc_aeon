#!/bin/bash
set -e

echo "--- BUILDING py3.11-cuda12.1 ---"
docker build -t aeon_base:py3.11-cuda12.1 \
  --build-arg PYTHON_VERSION_MINOR=11 \
  --build-arg CUDA_VERSION=12.1.1 \
  --build-arg PYTORCH_CUDA_SUFFIX=cu121 \
  .

echo "--- BUILDING py3.11-cuda11.8 ---"
docker build -t aeon_base:py3.11-cuda11.8 \
  --build-arg PYTHON_VERSION_MINOR=11 \
  --build-arg CUDA_VERSION=11.8.0 \
  --build-arg PYTORCH_CUDA_SUFFIX=cu118 \
  .

echo "--- BUILDING py3.10-cuda12.1 ---"
docker build -t aeon_base:py3.10-cuda12.1 \
  --build-arg PYTHON_VERSION_MINOR=10 \
  --build-arg CUDA_VERSION=12.1.1 \
  --build-arg PYTORCH_CUDA_SUFFIX=cu121 \
  .

echo "--- BUILDING py3.10-cuda11.8 ---"
docker build -t aeon_base:py3.10-cuda11.8 \
  --build-arg PYTHON_VERSION_MINOR=10 \
  --build-arg CUDA_VERSION=11.8.0 \
  --build-arg PYTORCH_CUDA_SUFFIX=cu118 \
  .

echo "--- BUILDING py3.9-cuda12.1 ---"
docker build -t aeon_base:py3.9-cuda12.1 \
  --build-arg PYTHON_VERSION_MINOR=9 \
  --build-arg CUDA_VERSION=12.1.1 \
  --build-arg PYTORCH_CUDA_SUFFIX=cu121 \
  .

echo "--- BUILDING py3.9-cuda11.8 ---"
docker build -t aeon_base:py3.9-cuda11.8 \
  --build-arg PYTHON_VERSION_MINOR=9 \
  --build-arg CUDA_VERSION=11.8.0 \
  --build-arg PYTORCH_CUDA_SUFFIX=cu118 \
  .

echo "--- ALL 6 BASE IMAGES BUILT SUCCESSFULLY ---"
