# Flux Dev2 Image Generation Tool

## Setup
Run `bash setup_environment.sh` (idempotent, downloads Flux.1-dev FP8 models: t5xxl_fp8 clip, flux1-dev-fp8 unet, ae vae).

## Test
bash
bash 1_setup_flux_dev2.sh  # Verify models
bash 2_test_flux_dev2.sh  # Generate test_flux_cat.png


## Usage
Import `FluxDev2Tool` from `aeon.tools.flux_dev2`:
python
from aeon.tools.flux_dev2 import FluxDev2Tool
tool = FluxDev2Tool()
image_path = tool.generate_image(
    prompt="photorealistic cat astronaut floating in space",
    width=1024,
    height=1024,
    steps=40,      # Best quality: 40-50
    guidance=3.5,  # Flux guidance
    shift=3.5,     # Max shift for quality
    seed=-1        # Random
)


- **Defaults**: Optimized for quality (steps=40, guidance=3.5, shift=3.5).
- **GPU**: Uses GPU1 (ComfyUI backend), auto-starts container, generates image, shuts down (releases VRAM).
- **Output**: `comfyui_output/<filename>.png`.
- **Workflow**: `aeon/comfyui/workflows/flux_image_api.json` (DualCLIP t5xxl+clip_l, FP8 UNet, ae VAE).