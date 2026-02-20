"""
FLUX.2-dev image generation tools for the Aeon agent.
Provides text-to-image, image-edit, and style-transfer capabilities.
"""

import json
import os
import subprocess
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List

from aeon.tools.base import BaseTool
from aeon.comfyui.backend import ComfyUIBackend


class FluxTextToImage(BaseTool):
    """Text-to-image generation using FLUX.2-dev FP8 with pi-Flow LoRA."""
    
    def __init__(self, backend: ComfyUIBackend = None):
        super().__init__(
            name="flux_text_to_image",
            description=(
                "Generate images from text prompts using FLUX.2-dev FP8 model. "
                "Best quality at 20 steps with pi-Flow LoRA enabled. "
                "Parameters: prompt (required), width (default 1024), height (default 1024), "
                "steps (default 20), guidance (default 4.0), shift (default 3.5), seed (default -1)."
            )
        )
        self.backend = backend or ComfyUIBackend()
    
    def execute(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        guidance: float = 4.0,
        shift: float = 3.5,
        seed: int = -1,
        negative_prompt: str = "",
        filename_prefix: str = "flux_t2i"
    ) -> Dict[str, Any]:
        """Generate an image from a text prompt."""
        params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "shift": shift,
            "seed": seed,
            "negative_prompt": negative_prompt,
            "filename_prefix": filename_prefix
        }
        return self.backend.run_model("flux_image", params)


class FluxImageEdit(BaseTool):
    """Image editing using FLUX.2-dev FP8 with pi-Flow LoRA (img2img)."""
    
    def __init__(self, backend: ComfyUIBackend = None):
        super().__init__(
            name="flux_image_edit",
            description=(
                "Edit an existing image using text instructions with FLUX.2-dev FP8. "
                "Uses img2img workflow with pi-Flow LoRA for high-fidelity edits. "
                "Parameters: prompt (required), input_image_path (required), "
                "width (default 1024), height (default 1024), steps (default 25), "
                "guidance (default 3.0), strength (default 0.7), seed (default -1)."
            )
        )
        self.backend = backend or ComfyUIBackend()
    
    def execute(
        self,
        prompt: str,
        input_image_path: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 25,
        guidance: float = 3.0,
        strength: float = 0.7,
        seed: int = -1,
        negative_prompt: str = "",
        filename_prefix: str = "flux_edit"
    ) -> Dict[str, Any]:
        """Edit an image using text instructions."""
        # First, ensure the input image is accessible to ComfyUI
        input_path = Path(input_image_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_image_path}")
        
        # Copy image to ComfyUI input directory if needed
        comfy_input_dir = Path("/home/aday/bc_aeon/comfyui_input")
        comfy_input_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = comfy_input_dir / input_path.name
        if dest_path != input_path:
            subprocess.run(
                ["cp", str(input_path), str(dest_path)],
                check=True
            )
        
        # Prepare workflow with img2img parameters
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "filename_prefix": filename_prefix,
            "input_image": input_path.name,
            "strength": strength
        }
        
        return self.backend.run_model("flux_image", params)


class FluxStyleTransfer(BaseTool):
    """Style transfer using FLUX.2-dev FP8 with pi-Flow LoRA."""
    
    def __init__(self, backend: ComfyUIBackend = None):
        super().__init__(
            name="flux_style_transfer",
            description=(
                "Transfer style from a reference image to a content image using FLUX.2-dev FP8. "
                "Parameters: prompt (required), content_image_path (required), "
                "style_image_path (required), width (default 1024), height (default 1024), "
                "steps (default 25), guidance (default 3.0), style_strength (default 0.8), "
                "seed (default -1)."
            )
        )
        self.backend = backend or ComfyUIBackend()
    
    def execute(
        self,
        prompt: str,
        content_image_path: str,
        style_image_path: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 25,
        guidance: float = 3.0,
        style_strength: float = 0.8,
        seed: int = -1,
        negative_prompt: str = "",
        filename_prefix: str = "flux_style"
    ) -> Dict[str, Any]:
        """Transfer style from one image to another."""
        # Ensure both images are accessible to ComfyUI
        content_path = Path(content_image_path)
        style_path = Path(style_image_path)
        
        if not content_path.exists():
            raise FileNotFoundError(f"Content image not found: {content_image_path}")
        if not style_path.exists():
            raise FileNotFoundError(f"Style image not found: {style_image_path}")
        
        # Copy images to ComfyUI input directory
        comfy_input_dir = Path("/home/aday/bc_aeon/comfyui_input")
        comfy_input_dir.mkdir(parents=True, exist_ok=True)
        
        content_dest = comfy_input_dir / content_path.name
        style_dest = comfy_input_dir / style_path.name
        
        if content_dest != content_path:
            subprocess.run(["cp", str(content_path), str(content_dest)], check=True)
        if style_dest != style_path:
            subprocess.run(["cp", str(style_path), str(style_dest)], check=True)
        
        # Prepare workflow with style transfer parameters
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "filename_prefix": filename_prefix,
            "content_image": content_path.name,
            "style_image": style_path.name,
            "style_strength": style_strength
        }
        
        return self.backend.run_model("flux_image", params)