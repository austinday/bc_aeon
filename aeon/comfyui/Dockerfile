# =============================================================================
# ComfyUI Runtime Image for Aeon Agent
# =============================================================================
# This image provides ComfyUI in API-only mode. Models are NOT baked in --
# they are mounted at runtime from the host model lake.
#
# === FOR FUTURE LLMs ===
# To add custom nodes for a new model:
#   1. Add a RUN block below to clone the custom node repo into
#      /opt/ComfyUI/custom_nodes/ and install its requirements.
#   2. Rebuild: docker build -t aeon_comfyui:latest -f this/Dockerfile .
#      (or just re-run setup_environment.sh)
#   3. The container will pick up the new nodes on next start.
# ========================

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-dev python3-pip \
    git wget curl libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.8 - required for Blackwell/sm_120 GPUs)
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# ComfyUI core
WORKDIR /opt
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI
WORKDIR /opt/ComfyUI
RUN pip3 install --no-cache-dir -r requirements.txt

RUN cd /opt/ComfyUI && git checkout master && git pull origin master
RUN pip3 install --no-cache-dir -r requirements.txt --upgrade

# Common dependencies many models share
RUN pip3 install --no-cache-dir \
    accelerate transformers sentencepiece protobuf \
    safetensors gguf bitsandbytes scipy \
    opencv-python-headless pillow

# ---------------------------------------------------------------------------
# CUSTOM NODES
# ---------------------------------------------------------------------------
# Each block installs one custom node pack. Add new ones here as needed.
# The || true ensures the build doesn't fail if a requirements.txt is missing.


# GGUF loader for quantized models
RUN cd /opt/ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF.git && \
    cd ComfyUI-GGUF && \
    pip3 install --no-cache-dir -r requirements.txt || true

# HunyuanImage 3.0 dedicated nodes (INT8/BF16/NF4 support)
RUN cd /opt/ComfyUI/custom_nodes && \
    git clone https://github.com/EricRollei/Comfy_HunyuanImage3.git && \
    cd Comfy_HunyuanImage3 && \
    pip3 install --no-cache-dir -r requirements.txt || true
#     cd ComfyUI-SomeVideoNodes && \
#     pip3 install --no-cache-dir -r requirements.txt || true
# =========================================================

 # pi-Flow nodes for FLUX.2 optimized sampling
 RUN cd /opt/ComfyUI/custom_nodes && \
     git clone https://github.com/Lakonik/ComfyUI-piFlow.git && \
     cd ComfyUI-piFlow && \
     pip3 install --no-cache-dir -r requirements.txt || true
RUN cd /opt/ComfyUI/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git ComfyUI-Manager && \
    cd ComfyUI-Manager && \
    pip3 install --no-cache-dir -r requirements.txt || true
# Create mount point directories
RUN mkdir -p /opt/ComfyUI/models/checkpoints \
    /opt/ComfyUI/models/clip \
    /opt/ComfyUI/models/vae \
    /opt/ComfyUI/models/unet \
    /opt/ComfyUI/models/diffusion_models \
    /opt/ComfyUI/models/text_encoders \
    /opt/ComfyUI/output \
    /opt/ComfyUI/input
RUN mkdir -p /opt/ComfyUI/user/default && \
    chown -R 1000:1000 /opt/ComfyUI || true

EXPOSE 8188

# Ensure main.py is executable (fixes Permission denied error)
RUN chmod +x /opt/ComfyUI/main.py

# API-only mode (no browser UI needed for agent usage)
CMD ["python3", "/opt/ComfyUI/main.py", "--listen", "0.0.0.0", "--port", "8188", "--disable-auto-launch"]
