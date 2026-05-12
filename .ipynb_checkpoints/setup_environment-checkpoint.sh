#!/bin/bash
set -euo pipefail

AEON_HOME="${AEON_HOME:-$HOME/.aeon}"
CACHE_DIR="$AEON_HOME/.setup_cache"
mkdir -p "$CACHE_DIR"

# =============================================================================
# IDEMPOTENCY HELPERS
# =============================================================================
_stamp_path() { echo "$CACHE_DIR/$(echo "$1" | sha256sum | cut -d' ' -f1).stamp"; }

docker_needs_rebuild() {
    local dockerfile="$1"
    local image_tag="$2"
    local context_dir="${3:-$(dirname "$dockerfile")}"
    
    if ! docker image inspect "$image_tag" >/dev/null 2>&1; then
        return 0  # missing
    fi
    
    local current_hash
    current_hash=$(find "$context_dir" -type f -not -path '*/\.*' -not -path '*/__pycache__/*' | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
    
    local stamp_file="$(_stamp_path "$image_tag")"
    if [[ -f "$stamp_file" ]] && [[ "$(cat "$stamp_file")" == "$current_hash" ]]; then
        return 1  # up-to-date
    fi
    return 0  # stale or never built
}

record_docker_build() {
    local dockerfile="$1"
    local image_tag="$2"
    local context_dir="${3:-$(dirname "$dockerfile")}"
    local current_hash
    current_hash=$(find "$context_dir" -type f -not -path '*/\.*' -not -path '*/__pycache__/*' | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
    echo "$current_hash" > "$(_stamp_path "$image_tag")"
}

model_needs_download() {
    local path="$1"
    local min_size="${2:-1048576}"  # 1 MB sanity floor
    [[ ! -f "$path" ]] && return 0
    local size
    size=$(stat -f%z "$path" 2>/dev/null || stat -c%s "$path" 2>/dev/null || echo 0)
    [[ "$size" -lt "$min_size" ]] && return 0
    return 1
}

record_download() {
    local path="$1"
    sha256sum "$path" 2>/dev/null | cut -d' ' -f1 > "$(_stamp_path "$path")" || touch "$(_stamp_path "$path")"
}

# =============================================================================
# BUILD WRAPPER
# =============================================================================
build_if_needed() {
    local name="$1"
    local dockerfile="$2"
    local tag="$3"
    local context="${4:-$(dirname "$dockerfile")}"
    
    if docker_needs_rebuild "$dockerfile" "$tag" "$context"; then
        echo "[BUILD] $name changed or missing. Building $tag ..."
        docker build -t "$tag" -f "$dockerfile" "$context"
        record_docker_build "$dockerfile" "$tag" "$context"
        echo "[BUILD] $name done."
    else
        echo "[BUILD] $name up-to-date (cached)."
    fi
}

# =============================================================================
# DOWNLOAD WRAPPER
# =============================================================================
download_if_needed() {
    local url="$1"
    local dest="$2"
    local min_size="${3:-1048576}"
    
    if model_needs_download "$dest" "$min_size"; then
        echo "[DOWNLOAD] Fetching $(basename "$dest") ..."
        mkdir -p "$(dirname "$dest")"
        if command -v aria2c >/dev/null 2>&1; then
            aria2c -x4 -s4 -c -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url"
        else
            wget -c -q --show-progress -O "$dest" "$url"
        fi
        record_download "$dest"
        echo "[DOWNLOAD] $(basename "$dest") ready."
    else
        echo "[DOWNLOAD] $(basename "$dest") already present. Skipping."
    fi
}

# =============================================================================
# EXAMPLE USAGE (populate with your real images / models)
# =============================================================================
echo "=================================================="
echo "    AEON ENVIRONMENT SETUP (idempotent)           "
echo "=================================================="

# Docker images
# build_if_needed "ComfyUI"   "aeon/services/comfyui/Dockerfile"   "aeon_comfyui:latest"   "aeon/services/comfyui"
# build_if_needed "vLLM"      "aeon/services/vllm/Dockerfile"      "aeon_vllm:latest"      "aeon/services/vllm"
# build_if_needed "Browser"   "aeon/services/browser/Dockerfile"   "aeon_browser_service:latest" "aeon/services/browser"
# build_if_needed "llama.cpp" "aeon/llamacpp/Dockerfile"           "aeon_llamacpp:latest"  "aeon/llamacpp"

# Models
# download_if_needed "<url>" "$AEON_HOME/models/gguf_models/Gemma-4/gemma-4-31b-abliterated-Q8_0.gguf"  30000000000

echo "=================================================="
echo "    SETUP COMPLETE                                "
echo "=================================================="