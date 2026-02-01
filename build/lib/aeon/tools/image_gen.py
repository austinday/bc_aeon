import os
import subprocess
import json
import uuid
from pathlib import Path
from .base import BaseTool

class GenerateImageTool(BaseTool):
    """Resource-aware tool for generating images with automatic model selection."""
    def __init__(self):
        super().__init__(
            name="generate_image",
            description='Generates images. Params: `prompt` (str), `name` (str), `count` (int), `aspect_ratio`. BATCH: `requests` list.'
        )
        self.models_base = Path.home() / "bc_aeon" / "aeon_models"
        self.scripts_dir = Path.home() / "bc_aeon" / "aeon" / "scripts"

    def execute(self, output_dir: str, requests: list = None, prompt: str = None, name: str = None, count: int = 1, aspect_ratio: str = "1:1"):
        if requests is None and prompt is not None:
            requests = [{"prompt": prompt, "name": name, "count": count, "aspect_ratio": aspect_ratio}]
        
        if not requests:
            return "Error: No prompts provided."

        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)

        req_filename = f"requests_{uuid.uuid4().hex[:8]}.json"
        req_host_path = os.path.join(abs_output_dir, req_filename)
        
        try:
            with open(req_host_path, 'w', encoding='utf-8') as f:
                json.dump(requests, f, indent=2)
        except Exception as e:
            return f"Error writing request file: {e}"

        # Use legacy runtime to avoid CDI errors on host
        cmd = [
            "docker", "run", "--rm", 
            "--runtime", "nvidia",
            "-e", "NVIDIA_VISIBLE_DEVICES=1",
            "-v", f"{self.models_base}:/models",
            "-v", f"{abs_output_dir}:/output",
            "-v", f"{self.scripts_dir}:/scripts",
            "aeon_vision:latest",
            "python", "/scripts/gen_img.py",
            "--models_dir", "/models",
            "--requests_file", f"/output/{req_filename}",
            "--output_dir", "/output"
        ]

        try:
            print(f"Aeon Vision: Processing {len(requests)} image request(s) on GPU 1...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return f"Success. Images in {abs_output_dir}.\nLogs:\n{result.stderr}\n{result.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Vision system failed:\n{e.stderr}\n{e.stdout}"
        finally:
            if os.path.exists(req_host_path): os.remove(req_host_path)
