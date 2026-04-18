FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git curl wget vim python3 python3-venv python-is-python3 python3-pip libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone https://github.com/comfyanonymous/ComfyUI.git

WORKDIR /workspace/ComfyUI
RUN pip install --no-cache-dir --break-system-packages torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

WORKDIR /workspace/ComfyUI/custom_nodes
RUN git clone https://github.com/city96/ComfyUI-GGUF.git
WORKDIR /workspace/ComfyUI/custom_nodes/ComfyUI-GGUF
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

RUN sed -i 's/except ImportError:/except Exception:/' __init__.py
RUN sed -i 's/from custom_nodes\.ComfyUI_GGUF\./from ./g' nodes.py

WORKDIR /workspace/ComfyUI
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
