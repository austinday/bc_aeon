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


class GenerateImageTool(BaseTool):
    """Agent-facing tool for text-to-image generation."""

    def __init__(self):
        super().__init__(
            name='generate_image',
            description=TOOL_DESC_GENERATE_IMAGE
        )
        self.backend = ComfyUIBackend()

    def execute(self, prompt: str, negative_prompt: str = '',
                width: int = 1024, height: int = 1024,
                steps: int = 30, cfg_scale: float = 6.0,
                flow_shift: float = 7.0, seed: int = -1) -> str:
        """
        Generate an image from a text prompt.

        Returns a string describing the result: either the output file path(s)
        on success, or an error message with debugging hints on failure.
        """
        if not prompt:
            return 'Error: prompt parameter is required. Describe the image you want to generate.'

        params = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'width': width,
            'height': height,
            'steps': steps,
            'cfg_scale': cfg_scale,
            'flow_shift': flow_shift,
            'seed': seed,
        }

        print(f'{self.C_CYAN}[GenerateImage] Calling ComfyUI backend with model=hunyuan_image{self.C_RESET}')
        print(f'{self.C_CYAN}[GenerateImage] Prompt: {prompt[:100]}...{self.C_RESET}')
        print(f'{self.C_CYAN}[GenerateImage] Params: {width}x{height}, steps={steps}, cfg={cfg_scale}, flow_shift={flow_shift}, seed={seed}{self.C_RESET}')

        result = self.backend.run_model('hunyuan_image', params)

        print(f'{self.C_CYAN}[GenerateImage] Result length: {len(result)} chars{self.C_RESET}')
        print(f'{self.C_CYAN}[GenerateImage] === FULL DEBUG RESULT ==={self.C_RESET}')
        # Print full result in chunks to avoid terminal buffer issues
        for i in range(0, len(result), 1000):
            print(f'{self.C_CYAN}{result[i:i+1000]}{self.C_RESET}')
        print(f'{self.C_CYAN}[GenerateImage] === END DEBUG RESULT ==={self.C_RESET}')

        return result
