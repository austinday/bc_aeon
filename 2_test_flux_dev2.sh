#!/bin/bash
set -e

# Pre-test GPU/container state
echo "Pre-test: GPU1 VRAM free:"
pre_vram=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$1==1 {print $2 " MiB"}')
echo "$pre_vram"
docker ps | grep aeon || echo "No aeon containers running."

# Run FluxDev2Tool test (generates test_flux_cat.png)
echo "Running FluxDev2Tool test..."
python test_flux.py

# Post-test validation
if [[ -f "comfyui_output/test_flux_cat.png" ]]; then
    size=$(du -h comfyui_output/test_flux_cat.png | cut -f1)
    echo "✓ Generated test_flux_cat.png (${size})."
else
    echo "✗ test_flux_cat.png not generated."
    exit 1
fi

docker ps | grep aeon || echo "✓ No aeon containers running (shutdown confirmed)."

post_vram=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, '$1==1 {print $2 " MiB"}')
echo "Post-test GPU1 VRAM free: $post_vram"

post_num=$(echo "$post_vram" | grep -o '[0-9]\+' | head -1)
if [[ $post_num -gt 90000 ]]; then
    echo "✓ GPU1 VRAM released (~95GB+ free)."
else
    echo "⚠ GPU1 VRAM not fully released ($post_num MiB)."
fi

echo "Test complete. FluxDev2Tool works!"