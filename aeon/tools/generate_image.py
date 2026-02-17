"""
GenerateImageTool - Text-to-image generation for the Aeon agent.

This is a thin wrapper around the shared ComfyUI backend. The agent sees
this as a simple 'generate_image' tool and never needs to know about
ComfyUI, Docker containers, or workflow JSONs.

=== FOR FUTURE LLMs: HOW TO ADD A NEW GENERATIVE TOOL ===

Copy this file as a template. You need to change:
  1. The class name (e.g., GenerateVideoTool)
  2. The tool name string (e.g., 'generate_video')
  3. The description import (e.g., TOOL_DESC_GENERATE_VIDEO)
  4. The model_id passed to self.backend.run_model() (must match a profile JSON)
  5. The execute() parameters to match what that model accepts

Then create the matching:
  - Profile JSON in aeon/comfyui/profiles/
  - Workflow JSON in aeon/comfyui/workflows/
  - Tool description in aeon/core/prompts/tool_desc_generate_<modality>.txt
  - Load it in aeon/core/prompts/__init__.py

The tool loader in aeon/tools/loader.py discovers new tools automatically.
No registration needed.
==========================================================
"""

from .base import BaseTool
from ..comfyui.backend import ComfyUIBackend
from ..core.prompts import TOOL_DESC_GENERATE_IMAGE

import re
import shutil


class GenerateImageTool(BaseTool):
    """Agent-facing tool for text-to-image generation. Uses Flux.2-dev for unrestricted high-quality images."""

    def __init__(self):
        super().__init__(
            name='generate_image',
            description=TOOL_DESC_GENERATE_IMAGE
        )
        self.backend = ComfyUIBackend()

    def execute(self, prompt: str, 
                width: int = 1024, height: int = 1024,
                steps: int = 50, cfg_scale: float = 1.0,
                seed: int = -1, output_path: str = None) -> str:
        """
        Generate an image from a text prompt using Flux.2-dev (uncensored, high quality).

        Args:
            prompt: Detailed description of the image.
            width/height: Resolution (e.g., 1024).
            steps: Sampling steps (20-50 for quality/speed).
            cfg_scale: Guidance (1.0 recommended for Flux).
            seed: Random seed (-1 for random).
            output_path: Optional path to move the generated PNG.

        Returns:
            Success message with path, or error/debug info.
        """
        if not prompt:
            return 'Error: prompt parameter is required. Describe the image you want to generate.'

        params = {
            'prompt': prompt,
            'width': width,
            'height': height,
            'steps': steps,
            'cfg_scale': cfg_scale,
            'seed': seed,
        }

        print(f'{self.C_CYAN}[GenerateImage] Flux.2-dev T2I{self.C_RESET}')
        print(f'{self.C_CYAN}Prompt: {prompt[:120]}{ "..." if len(prompt) > 120 else "" }{self.C_RESET}')
        print(f'{self.C_CYAN}Params: {width}×{height}, steps={steps}, cfg={cfg_scale}, seed={seed}{self.C_RESET}')

        result = self.backend.run_model('flux_image', params)

        if output_path:
            image_match = re.search(r'^/home/aday/bc_aeon/comfyui_output/[^ \t\n\r]*\.png', result, re.MULTILINE)
            if image_match:
                full_path = image_match.group(0).strip()
                shutil.move(full_path, output_path)
                print(f'{self.C_CYAN}[GenerateImage] Moved output to: {output_path}{self.C_RESET}')
                return f"✅ Flux.2-dev image generated and saved to: {output_path}"
            else:
                print(f'{self.C_CYAN}[GenerateImage] Parse fail - no PNG path found.{self.C_RESET}')
                return f"⚠️ Generation complete but could not auto-move to {output_path}. Backend result:\n{result[:500]}..."

        print(f'{self.C_CYAN}[GenerateImage] Backend result:\n{result[:400]}...{self.C_RESET}')
        return result