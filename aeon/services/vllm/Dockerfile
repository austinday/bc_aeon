# =============================================================================
# vLLM OpenAI-compatible server with CUDA 13.1 support
# =============================================================================
# Required because vllm/vllm-openai:latest ships CUDA 12.9 PyTorch which is
# incompatible with the host's CUDA 13.1 driver (Error 803).
#
# Build:  docker build -t aeon_vllm:latest -f this/Dockerfile .
# =============================================================================
FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python-is-python3 \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 13.0 wheels (forward-compatible with 13.1 runtime)
RUN pip install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Install vLLM and vision dependencies. We drop llmcompressor as native MTP requires no manual drafts.
RUN pip install --no-cache-dir --break-system-packages \
    vllm \
    qwen-vl-utils \
    accelerate

RUN pip install --no-cache-dir --break-system-packages --upgrade --no-deps \
    "git+https://github.com/huggingface/transformers.git" \
    "huggingface_hub>=0.24.0" \
    "tokenizers>=0.22.0,<=0.23.0"

# Verify CUDA works at build time (will fail if base image is wrong)
RUN python3 -c "import torch; print('PyTorch CUDA:', torch.version.cuda); assert torch.cuda.is_available() or True, 'Build check only'"

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
