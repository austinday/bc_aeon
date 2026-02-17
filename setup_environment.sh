#!/bin/bash
set -e

# =================================================================================================
# AEON ENVIRONMENT SETUP (Unified Model Lake Architecture)
# =================================================================================================
# 1. Builds the core Docker images (Base & Vision).
# 2. Downloads AI Models (Ollama & HuggingFace) into a unified ./aeon_models directory.
# 3. Prepares the runtime (Ollama/vLLM) to use these persistent, host-mounted models.
# =================================================================================================

# --- CONFIGURATION ---
# Brain Models (Ollama Registry)
# 1. qwen3-coder-next:q8_0 (Coding Specialist)
# 2. llama4:16x17b (General Purpose)
# 3. qwen3:235b-iq4xs (Custom Manual Install below)
OLLAMA_MODELS=(
 "qwen3-coder-next:q8_0"
 "llama4:16x17b"
)

# Tool Models (Hugging Face Registry)
HF_MODELS=()

# Directories
MODELS_DIR="$HOME/bc_aeon/aeon_models"
OLLAMA_DIR="$MODELS_DIR/ollama_home"

# Colors
C_CYAN='\033[96m'
C_GREEN='\033[92m'
C_YELLOW='\033[93m'
C_RED='\033[91m'
C_RESET='\033[0m'

# TTY Detection
TTY_FLAG=""
if [ -t 1 ]; then TTY_FLAG="-it"; fi

print_banner() {
 echo ""
 echo -e "${C_CYAN}======================================================================${C_RESET}"
 echo -e "${C_CYAN} $1${C_RESET}"
 echo -e "${C_CYAN}======================================================================${C_RESET}"
}

log_step() {
 echo -e "${C_GREEN}[+] $1${C_RESET}"
}

# =================================================================================================
# PHASE 1: PREPARATION & CLEANUP
# =================================================================================================
print_banner "PHASE 1: PREPARATION"

log_step "Creating Unified Model Lake at: $MODELS_DIR"
mkdir -p "$OLLAMA_DIR"

log_step "Stopping any existing Aeon containers..."
docker stop aeon_brain_node aeon_strong_node aeon_weak_node aeon_vllm aeon_setup_provisioner >/dev/null 2>&1 || true
docker rm -f aeon_setup_provisioner >/dev/null 2>&1 || true

# =================================================================================================
# PHASE 2: THE FOUNDRY (Docker Builds)
# =================================================================================================
print_banner "PHASE 2: THE FOUNDRY (Building Images)"

log_step "Building 'aeon_base' (The Core Runtime)..."
docker build -t aeon_base:py3.10-cuda12.1 \
 --build-arg PYTHON_VERSION_MINOR=10 \
 --build-arg CUDA_VERSION=12.1.1 \
 --build-arg PYTORCH_CUDA_SUFFIX=cu121 \
 .

log_step "Tagging 'aeon_base:latest'..."
docker tag aeon_base:py3.10-cuda12.1 aeon_base:latest

log_step "Pulling Inference Engines..."
docker pull ollama/ollama:latest
docker pull vllm/vllm-openai:latest

# =================================================================================================
# PHASE 3: BRAIN TRANSPLANT (Ollama Models)
# =================================================================================================
print_banner "PHASE 3: BRAIN TRANSPLANT (Ollama Models)"

log_step "Starting Provisioner Container..."
docker run -d --rm \
 --name aeon_setup_provisioner \
 --gpus all \
 -v "$OLLAMA_DIR:/root/.ollama" \
 -p 11435:11434 \
 ollama/ollama:latest

log_step "Waiting for Provisioner API..."
count=0
while ! curl -s http://localhost:11435/api/tags >/dev/null; do
 sleep 1
 count=$((count+1))
 if [ $count -ge 30 ]; then
 echo -e "${C_RED}Error: Provisioner failed to start.${C_RESET}"
 docker logs aeon_setup_provisioner
 exit 1
 fi
done

log_step "Downloading Standard Brain Models..."
for model in "${OLLAMA_MODELS[@]}"; do
 echo -e "${C_YELLOW} >> Pulling $model...${C_RESET}"
 if docker exec aeon_setup_provisioner ollama list | grep -q "$model"; then
 echo " (Already present)"
 else
 docker exec $TTY_FLAG aeon_setup_provisioner ollama pull "$model"
 fi
done

# Custom GGUF models removed due to VRAM limitations.

log_step "Stopping Provisioner..."
docker stop aeon_setup_provisioner

# =================================================================================================
# PHASE 4: TOOL SHED (Hugging Face Models)
# =================================================================================================
print_banner "PHASE 4: TOOL SHED (Hugging Face Models)"

HF_TOKEN_VAL=""
if [ -f "$HOME/huggingface_access_token.txt" ]; then
    HF_TOKEN_VAL=$(cat "$HOME/huggingface_access_token.txt" | tr -d '\n')
    echo -e "${C_GREEN}[+] Loaded HuggingFace Token from host.${C_RESET}"
else
    echo -e "${C_YELLOW}[!] No HF Token found. Gated models may fail.${C_RESET}"
fi

for model in "${HF_MODELS[@]}"; do
 clean_name=$(basename "$model")
 target_dir="/models/$clean_name"
 
 echo -e "${C_YELLOW} >> Downloading Tool Model: $model${C_RESET}"
 
 host_dir="$MODELS_DIR/$clean_name"
 if [ -f "$host_dir/config.json" ] && { [ -f "$host_dir/tokenizer.json" ] || [ -f "$host_dir/tokenizer_config.json" ]; }; then
 echo " (Already downloaded and validated - Skipping)"
 else
 if [ -d "$host_dir" ]; then
 echo -e "${C_YELLOW} (Incomplete download detected - re-downloading)${C_RESET}"
 rm -rf "$host_dir"
 fi
 docker run --rm $TTY_FLAG \
 --gpus all \
 -v "$MODELS_DIR:/models" \
 -e HF_HOME=/tmp/cache \
 -e HF_TOKEN="$HF_TOKEN_VAL" \
 aeon_base:py3.10-cuda12.1 \
 bash -c "python3 -c 'import huggingface_hub' 2>/dev/null || uv pip install --system --no-cache-dir huggingface_hub; python3 -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='$model', local_dir='$target_dir', local_dir_use_symlinks=False)\""
 fi
done

# =================================================================================================
# PHASE 5: COMFYUI GENERATIVE ENGINE
# =================================================================================================
print_banner "PHASE 5: COMFYUI GENERATIVE ENGINE"

COMFYUI_MODELS_DIR="$MODELS_DIR/comfyui_models"
COMFYUI_OUTPUT_DIR="$HOME/bc_aeon/comfyui_output"
mkdir -p "$COMFYUI_MODELS_DIR"
mkdir -p "$COMFYUI_OUTPUT_DIR"

log_step "Building ComfyUI Docker image (aeon_comfyui:latest)..."
COMFYUI_DOCKERFILE_DIR="$(cd "$(dirname "$0")/aeon/comfyui" && pwd)"
if [ -f "$COMFYUI_DOCKERFILE_DIR/Dockerfile" ]; then
    docker build -t aeon_comfyui:latest -f "$COMFYUI_DOCKERFILE_DIR/Dockerfile" "$COMFYUI_DOCKERFILE_DIR"
else
    echo -e "${C_YELLOW}[!] ComfyUI Dockerfile not found, skipping image build.${C_RESET}"
fi

# --- Model: HunyuanImage 3.0 Instruct INT8 (text-to-image) ---


log_step "Downloading Flux.2-dev FP8 UNet to diffusion_models/..."
UNET_DIR="$COMFYUI_MODELS_DIR/diffusion_models"
mkdir -p "$UNET_DIR"
FLUX_UNET="flux2_dev_fp8mixed.safetensors"
if [ ! -f "$UNET_DIR/$(basename "$FLUX_UNET")" ]; then
  docker run --rm $TTY_FLAG \
    -v "$UNET_DIR:/diffusion_models" \
    -e HF_HOME=/tmp/cache \
    ${HF_TOKEN_VAL:+-e HF_TOKEN="$HF_TOKEN_VAL"} \
    aeon_base:py3.10-cuda12.1 \
    bash -c "python3 -c 'import huggingface_hub' 2>/dev/null || uv pip install --system --no-cache-dir huggingface_hub; python3 -c 'from huggingface_hub import hf_hub_download; import shutil; path = hf_hub_download(repo_id=\"Comfy-Org/flux2-dev\", filename=\"split_files/diffusion_models/flux2_dev_fp8mixed.safetensors\", local_dir=\"/tmp/hf\"); shutil.move(path, \"/diffusion_models/flux2_dev_fp8mixed.safetensors\")'"
  echo "  Downloaded $FLUX_UNET"
else
  echo "  $FLUX_UNET already exists"
fi

log_step "Downloading Flux.2 Mistral FP8 Text Encoder to text_encoders/..."
TEXT_ENCODERS_DIR="$COMFYUI_MODELS_DIR/text_encoders"
mkdir -p "$TEXT_ENCODERS_DIR"
 MISTRAL_MODEL="mistral_3_small_flux2_fp8.safetensors"
if [ ! -f "$TEXT_ENCODERS_DIR/$(basename "$MISTRAL_MODEL")" ]; then
  docker run --rm $TTY_FLAG \
    -v "$TEXT_ENCODERS_DIR:/text_encoders" \
    -e HF_HOME=/tmp/cache \
    ${HF_TOKEN_VAL:+-e HF_TOKEN="$HF_TOKEN_VAL"} \
    aeon_base:py3.10-cuda12.1 \
     bash -c "python3 -c 'import huggingface_hub' 2>/dev/null || uv pip install --system --no-cache-dir huggingface_hub; python3 -c 'from huggingface_hub import hf_hub_download; import shutil; path = hf_hub_download(repo_id=\"Comfy-Org/flux2-dev\", filename=\"split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors\", local_dir=\"/tmp/hf\"); shutil.move(path, \"/text_encoders/mistral_3_small_flux2_fp8.safetensors\")'"
  echo "  Downloaded $MISTRAL_MODEL"
else
  echo "  $MISTRAL_MODEL already exists"
fi

log_step "Downloading Flux.2-dev VAE to vae/..."
VAE_DIR="$COMFYUI_MODELS_DIR/vae"
mkdir -p "$VAE_DIR"
FLUX_VAE="flux2-vae.safetensors"
if [ ! -f "$VAE_DIR/$FLUX_VAE" ]; then
  docker run --rm $TTY_FLAG \
    -v "$VAE_DIR:/vae" \
    -e HF_HOME=/tmp/cache \
    ${HF_TOKEN_VAL:+-e HF_TOKEN="$HF_TOKEN_VAL"} \
    aeon_base:py3.10-cuda12.1 \
    bash -c "python3 -c 'import huggingface_hub' 2>/dev/null || uv pip install --system --no-cache-dir huggingface_hub; python3 -c 'from huggingface_hub import hf_hub_download; import shutil; path = hf_hub_download(repo_id=\"Comfy-Org/flux2-dev\", filename=\"split_files/vae/flux2-vae.safetensors\", local_dir=\"/tmp/hf\"); shutil.move(path, \"/vae/flux2-vae.safetensors\")'"
  echo "  Downloaded $FLUX_VAE"
else
  echo "  $FLUX_VAE already exists"
fi
 log_step "Downloading pi-Flow LoRA adapter to loras/..."
 LORAS_DIR="$COMFYUI_MODELS_DIR/loras"
 mkdir -p "$LORAS_DIR"
 PIFLOW_LORA="gmflux2_k8_piid_4step.safetensors"
 if [ ! -f "$LORAS_DIR/$PIFLOW_LORA" ]; then
   docker run --rm $TTY_FLAG \
     -v "$LORAS_DIR:/loras" \
     -e HF_HOME=/tmp/cache \
     ${HF_TOKEN_VAL:+-e HF_TOKEN="$HF_TOKEN_VAL"} \
     aeon_base:py3.10-cuda12.1 \
     bash -c "python3 -c 'import huggingface_hub' 2>/dev/null || uv pip install --system --no-cache-dir huggingface_hub; python3 -c 'from huggingface_hub import hf_hub_download; import shutil; path = hf_hub_download(repo_id=\"Lakonik/pi-FLUX.2\", filename=\"gmflux2_k8_piid_4step/diffusion_pytorch_model.safetensors\", local_dir=\"/tmp/hf\"); shutil.move(path, \"/loras/gmflux2_k8_piid_4step.safetensors\")'"
   echo "  Downloaded $PIFLOW_LORA"
 else
   echo "  $PIFLOW_LORA already exists"
 fi


log_step "Fixing ComfyUI permissions..."
sudo chown -R $(id -u):$(id -g) "$COMFYUI_MODELS_DIR" "$COMFYUI_OUTPUT_DIR" "$COMFYUI_MODELS_DIR/vae"

# =================================================================================================
# PHASE 6: FINALIZATION
# =================================================================================================
print_banner "PHASE 6: FINALIZATION"
log_step "Fixing Permissions..."
sudo chown -R $(id -u):$(id -g) "$MODELS_DIR"
log_step "Environment Ready."
