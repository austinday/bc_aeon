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
OLLAMA_MODELS=(
 "deepseek-r1:70b"
 "qwen3-next:80b"
 "gpt-oss:120b"
 "llama4:16x17b"
 "qwen3:32b"
 "qwen3-coder:30b"
 "nemotron-3-nano:30b"
 "glm-4.7-flash:bf16"
 "gemma3:27b"
 "phi4:14b"
 "deepcoder:14b"
 "hf.co/TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF:Q4_K_M"
 "hf.co/TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF:Q8_0"
 "hf.co/TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill-GGUF:F16"
)

# Tool Models (Hugging Face Registry)
# Switched to Tencent HunyuanImage-3.0 as requested
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

log_step "Tagging 'aeon_vision' (Aliasing Base)..."
# Vision alias removed

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

log_step "Downloading Brain Models..."
for model in "${OLLAMA_MODELS[@]}"; do
 echo -e "${C_YELLOW} >> Pulling $model...${C_RESET}"
 if docker exec aeon_setup_provisioner ollama list | grep -q "$model"; then
 echo " (Already present)"
 else
 docker exec $TTY_FLAG aeon_setup_provisioner ollama pull "$model"
 fi
done

log_step "Stopping Provisioner..."
docker stop aeon_setup_provisioner

# =================================================================================================
# PHASE 4: TOOL SHED (Hugging Face Models)
# =================================================================================================
print_banner "PHASE 4: TOOL SHED (Hugging Face Models)"

# Load HF Token from host for Gated Repos
HF_TOKEN_VAL=""
if [ -f "$HOME/huggingface_access_token.txt" ]; then
    HF_TOKEN_VAL=$(cat "$HOME/huggingface_access_token.txt" | tr -d '\n')
    echo -e "${C_GREEN}[+] Loaded HuggingFace Token from host.${C_RESET}"
else
    echo -e "${C_YELLOW}[!] No HF Token found. Gated models (like Hunyuan) may fail.${C_RESET}"
fi

for model in "${HF_MODELS[@]}"; do
 clean_name=$(basename "$model")
 target_dir="/models/$clean_name"
 
 echo -e "${C_YELLOW} >> Downloading Tool Model: $model${C_RESET}"
 
 if [ -d "$MODELS_DIR/$clean_name" ] && [ "$(ls -A $MODELS_DIR/$clean_name)" ]; then
 echo " (Directory exists - Skipping)"
 else
 # Using direct Python call to avoid CLI path issues
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
# PHASE 5: FINALIZATION
# =================================================================================================
print_banner "PHASE 5: FINALIZATION"
log_step "Fixing Permissions..."
sudo chown -R $(id -u):$(id -g) "$MODELS_DIR"
log_step "Environment Ready."
