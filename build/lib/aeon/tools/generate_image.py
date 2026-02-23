import os
import json
import time
import random
import requests
from .base import BaseTool
from ..core.prompts import TOOL_DESC_GENERATE_IMAGE

class GenerateImageTool(BaseTool):
    """A tool to generate images using FLUX GGUF via a local ComfyUI instance."""
    def __init__(self):
        super().__init__(
            name="generate_image",
            description=TOOL_DESC_GENERATE_IMAGE
        )
        self.comfy_url = "http://localhost:8188"

    def _check_comfyui_health(self):
        try:
            res = requests.get(f"{self.comfy_url}/system_stats", timeout=2)
            return res.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def execute(self, prompt: str, output_path: str, width: int = 1024, height: int = 1024) -> str:
        if not prompt:
            return "Error: 'prompt' parameter is required."
        if not output_path:
            return "Error: 'output_path' parameter is required."

        if not self._check_comfyui_health():
            return (
                "Error: ComfyUI server is not running or not accessible at localhost:8188. "
                "Please run the startup script (e.g., `bash /home/aday/bc_aeon/aeon/scripts/start_comfyui.sh`) "
                "using the run_command tool before attempting to generate images."
            )

        abs_output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

        workflow = {
            "1": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {
                    "unet_name": "flux2-dev-Q4_K_S.gguf"
                }
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "mistral_3_small_flux2_fp8.safetensors",
                    "type": "flux2"
                }
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "flux2-vae.safetensors"
                }
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["2", 0]
                }
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "",
                    "clip": ["2", 0]
                }
            },
            "6": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": 1,
                    "width": width,
                    "height": height
                }
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": random.randint(1, 0xffffffffffffffff),
                    "steps": 25,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["6", 0]
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["3", 0]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "aeon_flux",
                    "images": ["8", 0]
                }
            }
        }

        try:
            print(f"{self.C_CYAN}Submitting image generation workflow to ComfyUI...{self.C_RESET}")
            req = requests.post(f"{self.comfy_url}/prompt", json={"prompt": workflow}, timeout=5)
            if req.status_code != 200:
                return f"Error submitting workflow to ComfyUI: {req.text}"
            
            prompt_id = req.json()["prompt_id"]
            
            print(f"{self.C_CYAN}Waiting for image generation to complete...{self.C_RESET}")
            max_retries = 120 # 6 minutes max wait
            for _ in range(max_retries):
                history_req = requests.get(f"{self.comfy_url}/history/{prompt_id}", timeout=5)
                history = history_req.json()
                
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    if "9" not in outputs:
                        return "Error: ComfyUI execution completed but SaveImage node did not produce output. Check server logs."
                    
                    image_info = outputs["9"]["images"][0]
                    filename = image_info["filename"]
                    subfolder = image_info["subfolder"]
                    folder_type = image_info["type"]
                    
                    img_req = requests.get(
                        f"{self.comfy_url}/view?filename={filename}&subfolder={subfolder}&type={folder_type}",
                        timeout=10
                    )
                    
                    if img_req.status_code == 200:
                        with open(abs_output_path, "wb") as f:
                            f.write(img_req.content)
                        return f"Successfully generated image and saved to: {abs_output_path}"
                    else:
                        return f"Error: Failed to download the generated image from ComfyUI (HTTP {img_req.status_code})"
                
                time.sleep(3)
                
            return "Error: Image generation timed out after 6 minutes."
            
        except Exception as e:
            return self.format_error_message(e, "generating image via ComfyUI", "checking if ComfyUI is running correctly")
