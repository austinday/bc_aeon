# =============================================================================
# llama.cpp Server with CUDA - for serving large GGUF models
# =============================================================================
# Builds llama-server from source with CUDA 12.8 support.
# Supports: continuous batching, parallel slots, GPU/CPU hybrid inference.
#
# Build:  docker build -t aeon_llamacpp:latest -f this/Dockerfile .
# =============================================================================

# --- Stage 1: Build ---
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git cmake build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Clone latest llama.cpp (MXFP4 and split-GGUF support)
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /build/llama.cpp

WORKDIR /build/llama.cpp

# Build with CUDA support for common architectures
# 80=A100, 86=A6000/3090, 89=L40/4090, 90=H100, 120=B200
# Make CUDA stubs available for linking (real driver is injected at runtime via --gpus)
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo '/usr/local/cuda/lib64/stubs' > /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;120" \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc) --target llama-server

# --- Stage 2: Runtime ---
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /build/llama.cpp/build/bin/ /usr/local/lib/llama/
ENV LD_LIBRARY_PATH=/usr/local/lib/llama:${LD_LIBRARY_PATH}

# Create model mount point
RUN mkdir -p /models

EXPOSE 8001

ENTRYPOINT ["llama-server"]
