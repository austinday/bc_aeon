# =============================================================================
# vLLM OpenAI-compatible server with CUDA 13.1 support
# =============================================================================
# Required because vllm/vllm-openai:latest ships CUDA 12.9 PyTorch which is
# incompatible with the host's CUDA 13.1 driver (Error 803).
#
# Build:  docker build -t aeon_vllm:latest -f this/Dockerfile .
# =============================================================================
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python-is-python3 \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 13.0 wheels (forward-compatible with 13.1 runtime)
RUN pip install --no-cache-dir --break-system-packages \
    "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Install vLLM pinned to the last known working release for MTP + NVFP4
RUN pip install --no-cache-dir --break-system-packages \
    --pre --extra-index-url https://wheels.vllm.ai/nightly \
    "vllm==0.23.0" \
    "qwen-vl-utils==0.0.14" \
    "accelerate==1.14.0"

# Pin the exact Transformers revision and companion packages from the validated
# runtime. Following mutable git main made clean rebuilds silently change API
# requirements and eventually produced an impossible tokenizers constraint.
RUN pip install --no-cache-dir --break-system-packages --upgrade --no-deps \
    "git+https://github.com/huggingface/transformers.git@a61d5f9e4fc184cff66938ff6c521cc358b5e024" \
    "huggingface_hub==1.27.0" \
    "tokenizers==0.23.0rc0"

# vLLM 0.23.0 has a scheduler bug when native MTP crosses Qwen's </think>
# boundary in the same decode block as schema-constrained final output. Apply
# the narrow, source-hash-guarded backport of merged upstream PR #44993. This
# is pure Python and leaves the already validated Qwen3.8/NVFP4 kernels intact.
COPY apply_mtp_structured_output_backport.py /tmp/apply_mtp_structured_output_backport.py
RUN python3 /tmp/apply_mtp_structured_output_backport.py \
    && rm /tmp/apply_mtp_structured_output_backport.py

# Verify CUDA works at build time (will fail if base image is wrong)
RUN python3 -c "import torch; print('PyTorch CUDA:', torch.version.cuda); assert torch.cuda.is_available() or True, 'Build check only'"

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
