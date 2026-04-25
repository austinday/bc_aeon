FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git curl wget vim python3 python3-venv python-is-python3 python3-pip libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*

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

# Add KJNodes (required for advanced LTX-2.3 workflows and text encoders)
WORKDIR /workspace/ComfyUI/custom_nodes
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git
WORKDIR /workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Add VideoHelperSuite (required for loading/saving mp4 videos)
WORKDIR /workspace/ComfyUI/custom_nodes
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
WORKDIR /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# --- PATCH COMFYUI SPIECE TOKENIZER FOR GGUF COMPATIBILITY ---
# When ComfyUI parses a GGUF text encoder, it extracts the SentencePiece vocab to a temp file
# and loads it. For Gemma-3, this metadata reconstruction results in a corrupt vocabulary file.
# This python snippet intercepts the tokenizer load and overwrites the corrupt temp file
# with the raw, correct tokenizer.model we downloaded in setup_environment.sh
WORKDIR /workspace/ComfyUI
RUN cat > patch_tokenizer.py << 'EOF'
import os
path = "/workspace/ComfyUI/comfy/text_encoders/spiece_tokenizer.py"
with open(path, "r") as f:
    content = f.read()

patch = """        real_model = "/workspace/ComfyUI/models/text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.model"
        import os, shutil
        if os.path.exists(real_model):
            try: shutil.copy(real_model, tokenizer_path)
            except: tokenizer_path = real_model
"""
content = content.replace("self.sp = sentencepiece.SentencePieceProcessor()", patch + "        self.sp = sentencepiece.SentencePieceProcessor()")
with open(path, "w") as f:
    f.write(content)
EOF
RUN python3 patch_tokenizer.py && rm patch_tokenizer.py

CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
