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
from pathlib import Path
import os


class GenerateImageTool(BaseTool):
    """Agent-facing tool for text-to-image generation. Uses pi-Flux.2-dev-fp8 for unrestricted high-quality images."""

    def __init__(self):
        super().__init__(
            name='generate_image',
            description=TOOL_DESC_GENERATE_IMAGE
        )
        self.backend = ComfyUIBackend()

    def execute(self, prompt: str, negative_prompt: str = "", 
                width: int = 1024, height: int = 1024,
                steps: int = 20, guidance: float = 4.0, shift: float = 3.2,
                seed: int = -1, output_path: str = None) -> str:
        """
        Generate an image from a text prompt using pi-Flux.2-dev-fp8 (uncensored, high quality).

        Args:
            prompt: Detailed description of the image.
             negative_prompt: Negative prompt (ignored by piFlow sampler).
              width/height: Resolution (e.g., 1024 x 1024 recommended).
             steps: Sampling steps (20+ for piFlow quality/speed).
             guidance: Flux Guidance (4.0 recommended for piFlow).
             shift: Flow shift (3.2 recommended for piFlow).
             seed: Random seed (-1 for random).
            output_path: Optional path to move the generated PNG.

        Returns:
            Success message with path, or error/debug info.
        """
        if not prompt:
            return 'Error: prompt parameter is required. Describe the image you want to generate.'

        params = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'width': width,
            'height': height,
            'steps': steps,
            'guidance': guidance,
            'shift': shift,
            'seed': seed,
        }

        if output_path:
            output_prefix = Path(output_path).stem
            params['FILENAME_PREFIX'] = output_prefix
        print(f'{self.C_CYAN}[GenerateImage] pi-Flux.2 T2I{self.C_RESET}')
        print(f'{self.C_CYAN}Prompt: {prompt[:120]}{ "..." if len(prompt) > 120 else "" }{self.C_RESET}')
        print(f'{self.C_CYAN}Params: {width}×{height}, steps={steps}, guidance={guidance}, shift={shift}, seed={seed}{self.C_RESET}')

        result = self.backend.run_model('pi_flux2', params)

        output_dir = str(Path.home() / 'bc_aeon' / 'comfyui_output')
        output_dir_path = Path(output_dir)
        if output_path:
            png_files = list(output_dir_path.glob(f'{output_prefix}*.png'))
            if png_files:
                latest_png = max(png_files, key=lambda p: p.stat().st_mtime)
                full_path = latest_png
                if full_path.exists():
                    shutil.move(str(full_path), output_path)
                    print(f'{self.C_CYAN}[GenerateImage] Moved {full_path.name} to: {output_path}{self.C_RESET}')
                    return f"✅ Flux.2-dev image generated and saved to: {output_path}"
            print(f'{self.C_CYAN}[GenerateImage] No PNGs with prefix "{output_prefix}" found in {output_dir}.{self.C_RESET}')
            print(f'{self.C_CYAN}Recent PNGs:{self.C_RESET}')
            recent = sorted(output_dir_path.glob('*.png'), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
            for p in recent:
                print(f'  {p.name} (mtime: {p.stat().st_mtime:.0f})')
        return result
    def run(self, **kwargs):
        return self.execute(**kwargs)
