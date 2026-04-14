#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/home/aday/bc_aeon"
HF_TOKEN_FILE="/home/aday/huggingface_access_token.txt"

log_step() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Load HF_TOKEN if not set
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '\n')
    log_step "Loaded HF_TOKEN from $HF_TOKEN_FILE"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN environment variable required for model downloads."
    echo "Create $HF_TOKEN_FILE with your token or export HF_TOKEN."
    exit 1
fi

log_step "PHASE 1: Build Docker images (layer cache makes unchanged builds fast)"

# Stop any running containers that depend on these images
for cname in aeon_qwen35_27b_speculative aeon_gemma4_speculative; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${cname}$"; then
        log_step "Stopping stale container: $cname"
        docker rm -f "$cname" >/dev/null 2>&1 || true
    fi
done

if ! docker image inspect aeon_downloader:latest >/dev/null 2>&1; then
    log_step "Building aeon_downloader:latest..."
    cat > "$PROJECT_ROOT/Dockerfile.downloader" << 'EOF'
FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "huggingface_hub[cli]"
EOF
    docker build -t aeon_downloader:latest -f "$PROJECT_ROOT/Dockerfile.downloader" "$PROJECT_ROOT"
    rm -f "$PROJECT_ROOT/Dockerfile.downloader"
else
    log_step "aeon_downloader:latest image already exists, skipping build."
fi

# =============================================================================
# PHASE 5.1: Qwen3.5-27B-Uncensored + 2B Draft (llama.cpp served)
# =============================================================================
QWEN35_27B_DIR="$PROJECT_ROOT/aeon_models/gguf_models/Qwen3.5-27B"
log_step "PHASE 5.1: Download Qwen3.5-27B-Uncensored and 2B Draft GGUFs"
mkdir -p "$QWEN35_27B_DIR"

if [[ -f "$QWEN35_27B_DIR/.download_complete" ]]; then
    log_step "Qwen3.5 27B Speculative models already downloaded, skipping."
else
    QWEN35_27B_DL_SCRIPT=$(mktemp /tmp/aeon_dl_qwen35_27b_XXXXXX.py)
    cat > "$QWEN35_27B_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files

TARGET = "/models"
# 1. Target Model: Qwen3.5-27B-Instruct-Uncensored (Q8_0)
target_repo = "n0ctyx/Qwen3.5-27B-Instruct-Uncensored"
target_prefix = "Q8_0"

print(f"Listing files in {target_repo}...", flush=True)
try:
    all_target_files = list_repo_files(target_repo)
except Exception as e:
    print(f"Failed to list repo: {e}")
    sys.exit(1)

# Case insensitive search
target_shards = sorted([f for f in all_target_files if target_prefix.lower() in f.lower() and f.endswith(".gguf")])
if not target_shards:
    print(f"ERROR: Could not find target model matching {target_prefix} in {target_repo}!", flush=True)
    sys.exit(1)

for shard in target_shards:
    dest = os.path.join(TARGET, shard)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000_000:
        print(f"[{shard}] already exists, skipping.", flush=True)
        continue
    print(f"Downloading {shard} from {target_repo}...", flush=True)
    try:
        hf_hub_download(repo_id=target_repo, filename=shard, local_dir=TARGET)
    except Exception as e:
        print(f"ERROR: Failed to download {shard}: {e}")
        sys.exit(1)

# 2. Draft Model: Huihui-Qwen3.5-2B-abliterated-i1-GGUF (Q4_K_M)
draft_repo = "mradermacher/Huihui-Qwen3.5-2B-abliterated-i1-GGUF"
print(f"Listing files in {draft_repo}...", flush=True)
try:
    all_draft_files = list_repo_files(draft_repo)
except Exception as e:
    print(f"Failed to list repo: {e}")
    sys.exit(1)

draft_file = next((f for f in all_draft_files if "q4_k_m" in f.lower() and f.endswith(".gguf")), None)
if draft_file:
    dest = os.path.join(TARGET, draft_file)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000_000:
        print(f"[{draft_file}] already exists, skipping.", flush=True)
    else:
        print(f"Downloading {draft_file} from {draft_repo}...", flush=True)
        try:
            hf_hub_download(repo_id=draft_repo, filename=draft_file, local_dir=TARGET)
        except Exception as e:
            print(f"ERROR: Failed to download {draft_file}: {e}")
            sys.exit(1)

print("All Qwen3.5-27B Speculative files downloaded successfully.", flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$QWEN35_27B_DIR:/models" \
        -v "$QWEN35_27B_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py

    DL_EXIT=$?
    rm -f "$QWEN35_27B_DL_SCRIPT"
    if [[ $DL_EXIT -ne 0 ]]; then
        log_step "ERROR: Qwen3.5-27B Speculative download failed (exit code $DL_EXIT)"
        exit 1
    fi

    docker run --rm -v "$QWEN35_27B_DIR:/models" aeon_downloader:latest chown -R $(id -u):$(id -g) /models || true
    touch "$QWEN35_27B_DIR/.download_complete"
fi
log_step "PHASE 5.1 complete."

# =============================================================================
# PHASE 5.5: Qwen3-Coder-Next-Abliterated-Q8_0 (llama.cpp served)
# =============================================================================
QWEN3_CODER_GGUF_DIR="$PROJECT_ROOT/aeon_models/gguf_models/Qwen3-Coder-Next-Abliterated"
log_step "PHASE 5.5: Download Qwen3-Coder-Next-Abliterated-Q8_0 GGUF model shards"
mkdir -p "$QWEN3_CODER_GGUF_DIR"

if [[ -f "$QWEN3_CODER_GGUF_DIR/.download_complete" ]]; then
    log_step "Qwen3 Coder GGUF already downloaded, skipping."
else
    QWEN3_CODER_DL_SCRIPT=$(mktemp /tmp/aeon_dl_qwen3_coder_XXXXXX.py)
    cat > "$QWEN3_CODER_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "bartowski/huihui-ai_Qwen3-Coder-Next-abliterated-GGUF"
TARGET = "/models"
PREFIX = "Q8_0"

print(f"Listing files in {REPO}...", flush=True)
try:
    all_files = list_repo_files(REPO)
except Exception as e:
    print(f"Failed to list repo: {e}")
    sys.exit(1)

print(f"\nProcessing {PREFIX}...", flush=True)
shards = sorted([f for f in all_files if PREFIX in f and f.endswith(".gguf")])
print(f"Found {len(shards)} shard(s):", flush=True)
for s in shards:
    print(f"  {s}", flush=True)

if not shards:
    print(f"ERROR: No matching GGUF shards found in repo for {PREFIX}!", flush=True)
    sys.exit(1)

all_done = True
for i, shard in enumerate(shards, 1):
    dest = os.path.join(TARGET, shard)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000_000:
        sz = os.path.getsize(dest) / (1024**3)
        print(f"[{i}/{len(shards)}] {os.path.basename(shard)} already exists ({sz:.1f}GB), skipping.", flush=True)
        continue
    all_done = False
    print(f"[{i}/{len(shards)}] Downloading {shard}...", flush=True)
    try:
        hf_hub_download(
            repo_id=REPO,
            filename=shard,
            local_dir=TARGET,
        )
        sz = os.path.getsize(dest) / (1024**3)
        print(f"  Done: {sz:.1f}GB", flush=True)
    except Exception as e:
        print(f"Failed to download {shard}: {e}")
        sys.exit(1)

if all_done:
    print(f"All {PREFIX} shards already present and valid.", flush=True)
else:
    print(f"All {PREFIX} shards downloaded successfully.", flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$QWEN3_CODER_GGUF_DIR:/models" \
        -v "$QWEN3_CODER_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py

    DL_EXIT=$?
    rm -f "$QWEN3_CODER_DL_SCRIPT"
    if [[ $DL_EXIT -ne 0 ]]; then
        log_step "ERROR: Qwen3 Coder GGUF download failed (exit code $DL_EXIT)"
        exit 1
    fi

    docker run --rm -v "$QWEN3_CODER_GGUF_DIR:/models" aeon_downloader:latest chown -R $(id -u):$(id -g) /models || true
    touch "$QWEN3_CODER_GGUF_DIR/.download_complete"
fi
log_step "PHASE 5.5 complete."

# =============================================================================
# PHASE 5.6: Gemma-4-31B + E2B Draft Models
# =============================================================================
GEMMA4_GGUF_DIR="$PROJECT_ROOT/aeon_models/gguf_models/Gemma-4"
log_step "PHASE 5.6: Download Gemma 4 31B and E2B Draft GGUFs"
mkdir -p "$GEMMA4_GGUF_DIR"

if [[ -f "$GEMMA4_GGUF_DIR/.download_complete" ]]; then
    log_step "Gemma 4 models already downloaded, skipping."
else
    GEMMA4_DL_SCRIPT=$(mktemp /tmp/aeon_dl_gemma4_XXXXXX.py)
    cat > "$GEMMA4_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files

TARGET = "/models"

# 1. Target Model: 31B Abliterated Q8_0
target_repo = "paperscarecrow/Gemma-4-31B-it-abliterated"
target_file = "gemma-4-31b-abliterated-Q8_0.gguf"

# 2. Draft Model: E2B Heretic i1-Q4_K_M
draft_repo = "mradermacher/gemma-4-E2B-it-heretic-i1-GGUF"
print(f"Listing files in {draft_repo}...", flush=True)
try:
    repo_files = list_repo_files(draft_repo)
except Exception as e:
    print(f"ERROR: Failed to list repo {draft_repo}: {e}")
    sys.exit(1)

draft_file = next((f for f in repo_files if "Q4_K_M" in f and f.endswith(".gguf")), None)
if not draft_file:
    print(f"ERROR: Could not find Q4_K_M file in {draft_repo}!", flush=True)
    sys.exit(1)

downloads = [
    (target_repo, target_file),
    (draft_repo, draft_file)
]

for repo, fname in downloads:
    dest = os.path.join(TARGET, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000_000:
        print(f"[{fname}] already exists, skipping.", flush=True)
        continue
        
    print(f"Downloading {fname} from {repo}...", flush=True)
    try:
        hf_hub_download(
            repo_id=repo,
            filename=fname,
            local_dir=TARGET,
        )
        print(f"  Done: {fname}", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to download {fname}: {e}")
        sys.exit(1)

print("All Gemma 4 files downloaded successfully.", flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$GEMMA4_GGUF_DIR:/models" \
        -v "$GEMMA4_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py

    DL_EXIT=$?
    rm -f "$GEMMA4_DL_SCRIPT"
    if [[ $DL_EXIT -ne 0 ]]; then
        log_step "ERROR: Gemma 4 download failed (exit code $DL_EXIT)"
        exit 1
    fi

    docker run --rm -v "$GEMMA4_GGUF_DIR:/models" aeon_downloader:latest chown -R $(id -u):$(id -g) /models || true
    touch "$GEMMA4_GGUF_DIR/.download_complete"
fi
log_step "PHASE 5.6 complete."

# =============================================================================
# PHASE 5.7: Qwen3.5-35B-A3B GGUF (Vision-Language Model for llama.cpp)
# =============================================================================
QWEN3_VL_DIR="$PROJECT_ROOT/aeon_models/vl_models/Qwen3.5-35B-A3B-GGUF"
log_step "PHASE 5.7: Download Qwen3.5-35B-A3B GGUF for vision analysis tool"
mkdir -p "$QWEN3_VL_DIR"

if [[ -f "$QWEN3_VL_DIR/.download_complete" ]]; then
    log_step "Qwen3.5-35B-A3B GGUF already downloaded, skipping."
else
    QWEN3_VL_DL_SCRIPT=$(mktemp /tmp/aeon_dl_qwen3_vl_XXXXXX.py)
    cat > "$QWEN3_VL_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download, list_repo_files

REPO = 'unsloth/Qwen3.5-35B-A3B-GGUF'
TARGET = '/models'

print(f'Listing files in {REPO}...', flush=True)
try:
    repo_files = list_repo_files(REPO)
    mmproj_file = next((f for f in repo_files if f.startswith('mmproj') and f.endswith('.gguf')), None)
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)

FILES = [
    'Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf',
]
if mmproj_file:
    FILES.append(mmproj_file)

for fname in FILES:
    print(f'Downloading {fname} from {REPO}...', flush=True)
    try:
        hf_hub_download(
            repo_id=REPO,
            filename=fname,
            local_dir=TARGET,
        )
        print(f'  -> {fname} complete.', flush=True)
    except Exception as e:
        print(f'ERROR: Failed to download {fname}: {e}', flush=True)
        sys.exit(1)

print('All files downloaded successfully!', flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$QWEN3_VL_DIR:/models" \
        -v "$QWEN3_VL_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py

    DL_EXIT=$?
    rm -f "$QWEN3_VL_DL_SCRIPT"
    if [[ $DL_EXIT -ne 0 ]]; then
        log_step "ERROR: Qwen3-VL GGUF download failed (exit code $DL_EXIT)"
        exit 1
    fi

    docker run --rm -v "$QWEN3_VL_DIR:/models" aeon_downloader:latest chown -R $(id -u):$(id -g) /models || true
    touch "$QWEN3_VL_DIR/.download_complete"
fi
log_step "PHASE 5.7 complete."

# =============================================================================
# PHASE 5.9: Build prepare_for_printify Docker image (print-on-demand preprocessing)
# =============================================================================
log_step "PHASE 5.9: Build bananacoconut-preprocessor Docker image for printify preprocessing"

BANANA_COCONUT_DIR="$PROJECT_ROOT/../bananaCoconut"
mkdir -p "$BANANA_COCONUT_DIR"

# Write the preprocessing script
cat > "$BANANA_COCONUT_DIR/complete_printify_preprocess.py" << 'PYEOF'
import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import io

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("WARNING: rembg not available. Background removal will be skipped.")

def process_image(input_path: str, output_path: str, args):
    """Process a single image through the pipeline."""
    print(f"Processing: {input_path}")
    
    try:
        # Open image
        img = Image.open(input_path)
        
        # Convert to RGBA for transparency support
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Stage 1: Resize if specified
        if args.target_width or args.target_height:
            orig_w, orig_h = img.size
            if args.target_width and args.target_height:
                new_w, new_h = args.target_width, args.target_height
            elif args.target_width:
                scale = args.target_width / orig_w
                new_w = args.target_width
                new_h = int(orig_h * scale)
            else:
                scale = args.target_height / orig_h
                new_w = int(orig_w * scale)
                new_h = args.target_height
            
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f"  Resized to {new_w}x{new_h}")
        
        # Stage 2: Background removal
        if args.background_removal and REMBG_AVAILABLE:
            print("  Removing background...")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Try multiple models if first fails
            for model in ['u2net', 'u2netp', 'silueta']:
                try:
                    img = remove(img_bytes.read(), model_name=model)
                    img = Image.open(io.BytesIO(img))
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    print(f"  Background removed using {model}")
                    break
                except Exception as e:
                    print(f"    Model {model} failed: {e}")
                    continue
            else:
                print("  WARNING: All background removal models failed")
        
        # Stage 3: Trim transparent edges
        if args.trim_transparent_edges:
            print("  Trimming transparent edges...")
            img = trim_transparent_edges(img)
        
        # Stage 4: Add watermark if specified
        if args.watermark_text:
            print(f"  Adding watermark: {args.watermark_text}")
            img = add_watermark(img, args.watermark_text)
        
        # Stage 5: Save with correct DPI and format
        output_format = args.output_format.upper()
        if output_format == 'JPG' or output_format == 'JPEG':
            # Convert to RGB for JPG
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save with DPI
        save_kwargs = {'dpi': (args.dpi, args.dpi)}
        if output_format in ['PNG', 'TIFF']:
            save_kwargs['format'] = output_format
        elif output_format in ['JPG', 'JPEG']:
            save_kwargs['format'] = 'JPEG'
            save_kwargs['quality'] = 95
        
        img.save(output_path, **save_kwargs)
        print(f"  Saved to {output_path} ({img.size[0]}x{img.size[1]}, {args.dpi} DPI)")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return False


def trim_transparent_edges(img: Image.Image) -> Image.Image:
    """Trim transparent edges from an RGBA image."""
    if img.mode != 'RGBA':
        return img
    
    # Get alpha channel
    alpha = img.split()[3]
    
    # Find bounding box of non-transparent pixels
    bbox = alpha.getbbox()
    
    if bbox is None:
        # All transparent, return empty image
        return Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    
    # Crop to bounding box
    return img.crop(bbox)


def add_watermark(img: Image.Image, text: str) -> Image.Image:
    """Add a watermark text to the image."""
    from PIL import ImageDraw, ImageFont
    
    # Create a copy to draw on
    img = img.copy()
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(img)
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position in bottom-right corner with padding
    padding = 10
    x = img.size[0] - text_width - padding
    y = img.size[1] - text_height - padding
    
    # Draw semi-transparent background for text
    background_bbox = (x - 5, y - 5, x + text_width + 5, y + text_height + 5)
    draw.rectangle(background_bbox, fill=(0, 0, 0, 128))
    
    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    
    return img


def main():
    parser = argparse.ArgumentParser(description='Printify Image Preprocessing Pipeline')
    parser.add_argument('--input_dir', required=True, help='Input directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--background_removal', action='store_true', default=True, help='Remove background')
    parser.add_argument('--no_background_removal', action='store_true', help='Disable background removal')
    parser.add_argument('--target_width', type=int, help='Target width in pixels')
    parser.add_argument('--target_height', type=int, help='Target height in pixels')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--trim_transparent_edges', action='store_true', default=True, help='Trim transparent edges')
    parser.add_argument('--no_trim', action='store_true', help='Disable edge trimming')
    parser.add_argument('--output_format', default='PNG', help='Output format (PNG/JPG)')
    parser.add_argument('--watermark_text', type=str, help='Watermark text to add')
    parser.add_argument('--max_workers', type=int, default=2, help='Max parallel workers')
    
    args = parser.parse_args()
    
    # Override flags
    if args.no_background_removal:
        args.background_removal = False
    if args.no_trim:
        args.trim_transparent_edges = False
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(images)} images to process")
    
    # Process each image
    success_count = 0
    for img_path in images:
        output_path = output_dir / f"{img_path.stem}_processed.{args.output_format.lower()}"
        if process_image(str(img_path), str(output_path), args):
            success_count += 1
    
    print(f"\\nCompleted: {success_count}/{len(images)} images processed successfully")


if __name__ == '__main__':
    main()
PYEOF

# Write the Dockerfile
cat > "$BANANA_COCONUT_DIR/Dockerfile" << 'EOF'
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    Pillow \
    rembg[cpu] \
    onnxruntime

# Copy preprocessing script
COPY complete_printify_preprocess.py /app/

# Default command
CMD ["python", "complete_printify_preprocess.py"]
EOF

# Build the Docker image
if ! docker image inspect bananacoconut-preprocessor:latest >/dev/null 2>&1; then
    log_step "Building bananacoconut-preprocessor:latest..."
    docker build -t bananacoconut-preprocessor:latest "$BANANA_COCONUT_DIR"
    log_step "bananacoconut-preprocessor:latest built successfully."
else
    log_step "bananacoconut-preprocessor:latest already exists, skipping build."
fi

log_step "PHASE 5.9 complete."

log_step "PHASE 6: Build aeon_llamacpp:latest Docker image"
log_step "Building aeon_llamacpp:latest (compiling llama.cpp with CUDA, may take 5-10 min on first build)..."
docker build -t aeon_llamacpp:latest -f "$PROJECT_ROOT/aeon/llamacpp/Dockerfile" "$PROJECT_ROOT/aeon/llamacpp/"
log_step "aeon_llamacpp:latest built successfully."

# Build ComfyUI Docker image (for FLUX image generation tool)
log_step "PHASE 6b: Build aeon_comfyui:latest Docker image"
log_step "Building aeon_comfyui:latest (installs PyTorch + ComfyUI + GGUF plugin, may take 5-10 min on first build)..."
docker build -t aeon_comfyui:latest -f "$PROJECT_ROOT/aeon/services/comfyui/Dockerfile" "$PROJECT_ROOT/aeon/services/comfyui/"
log_step "aeon_comfyui:latest built successfully."

# =============================================================================
# PHASE 7: ComfyUI Models (FLUX)
# =============================================================================
COMFY_MODELS_DIR="$PROJECT_ROOT/aeon_models/comfyui"
log_step "PHASE 7: Download FLUX GGUF models and encoders for ComfyUI"
mkdir -p "$COMFY_MODELS_DIR/unet"
mkdir -p "$COMFY_MODELS_DIR/text_encoders"
mkdir -p "$COMFY_MODELS_DIR/vae"

if [[ -f "$COMFY_MODELS_DIR/.download_complete" ]]; then
    log_step "FLUX models already downloaded, skipping."
else
    FLUX_DL_SCRIPT=$(mktemp /tmp/aeon_dl_flux_XXXXXX.py)
    cat > "$FLUX_DL_SCRIPT" << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download

print('Downloading Flux 2 Dev UNet GGUF...', flush=True)
hf_hub_download(repo_id='unsloth/FLUX.2-dev-GGUF', filename='flux2-dev-Q4_K_S.gguf', local_dir='/models/unet')

print('Downloading Flux 2 VAE...', flush=True)
hf_hub_download(repo_id='Comfy-Org/flux2-dev', filename='split_files/vae/flux2-vae.safetensors', local_dir='/models')

print('Downloading FLUX.2 Mistral text encoder...', flush=True)
hf_hub_download(repo_id='Comfy-Org/flux2-dev', filename='split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors', local_dir='/models')

print('Downloads complete!', flush=True)
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$COMFY_MODELS_DIR:/models" \
        -v "$FLUX_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py

    DL_EXIT=$?
    rm -f "$FLUX_DL_SCRIPT"
    if [[ $DL_EXIT -ne 0 ]]; then
        log_step "ERROR: FLUX download failed (exit code $DL_EXIT)"
        exit 1
    fi

    docker run --rm -v "$COMFY_MODELS_DIR:/models" aeon_downloader:latest chown -R $(id -u):$(id -g) /models || true
    touch "$COMFY_MODELS_DIR/.download_complete"
fi
log_step "PHASE 7 complete."

# =============================================================================
# PHASE 8: PuLID FLUX Models (Consistent Characters)
# =============================================================================
PULID_MODELS_DIR="$PROJECT_ROOT/aeon_models/comfyui/pulid"
CLIP_DIR="$PROJECT_ROOT/aeon_models/comfyui/clip"
INSIGHTFACE_DIR="$PROJECT_ROOT/aeon_models/comfyui/insightface"

log_step "PHASE 8: Download PuLID Flux and Face models"
mkdir -p "$PULID_MODELS_DIR" "$CLIP_DIR" "$INSIGHTFACE_DIR"

if [[ -f "$PULID_MODELS_DIR/.download_complete" ]]; then
    log_step "PuLID models already downloaded, skipping."
else
    PULID_DL_SCRIPT=$(mktemp /tmp/aeon_dl_pulid_XXXXXX.py)
    cat > "$PULID_DL_SCRIPT" << 'PYEOF'
import os
from huggingface_hub import hf_hub_download, snapshot_download

print('Downloading PuLID Flux...', flush=True)
hf_hub_download(repo_id='guozinan/PuLID', filename='pulid_flux_v0.9.0.safetensors', local_dir='/models/pulid')

print('Downloading EvaCLIP...', flush=True)
hf_hub_download(repo_id='QuanSun/EVA-CLIP', filename='EVA02_CLIP_L_336_psz14_s6B.pt', local_dir='/models/clip')

print('Downloading AntelopeV2 (InsightFace)...', flush=True)
snapshot_download(repo_id='kidyu/antelopev2-for-InstantID-ComfyUI', local_dir='/models/insightface/models/antelopev2')
PYEOF

    TTY_FLAG=""
    if [ -t 0 ]; then TTY_FLAG="-t"; fi
    docker run --rm $TTY_FLAG \
        -e HF_TOKEN="$HF_TOKEN" \
        -e PYTHONUNBUFFERED=1 \
        -v "$PROJECT_ROOT/aeon_models/comfyui:/models" \
        -v "$PULID_DL_SCRIPT:/download.py:ro" \
        aeon_downloader:latest \
        python3 /download.py
        
    rm -f "$PULID_DL_SCRIPT"
    docker run --rm -v "$PROJECT_ROOT/aeon_models/comfyui:/models" aeon_downloader:latest chown -R $(id -u):$(id -g) /models/pulid /models/clip /models/insightface || true
    touch "$PULID_MODELS_DIR/.download_complete"
fi
log_step "PHASE 8 complete."

log_step "Setup complete. Models in $QWEN35_27B_DIR, $QWEN3_CODER_GGUF_DIR, $COMFY_MODELS_DIR, $QWEN3_VL_DIR, $GEMMA4_GGUF_DIR"
log_step "NOTE: To remove old BF16 Qwen3-VL model (if present): rm -rf $PROJECT_ROOT/aeon_models/vl_models/Qwen3-VL-32B-Instruct"
