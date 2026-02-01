import os
import subprocess
from pathlib import Path
from .base import BaseTool

class ImageViewerTool(BaseTool):
    """Agent-driven vision tool for targeted image analysis."""
    def __init__(self):
        super().__init__(
            name='image_viewer',
            description='Analyzes an image file using an AI vision model. Params: `image_path` (str), `prompt` (str).'
        )
        self.models_base = Path.home() / 'bc_aeon' / 'aeon_models'
        self.scripts_dir = Path.home() / 'bc_aeon' / 'aeon' / 'scripts'
        self.C_CYAN = '\033[96m'
        self.C_RESET = '\033[0m'

    def execute(self, image_path: str, prompt: str):
        if not image_path or not prompt:
            return 'Error: Both `image_path` and `prompt` are required.'

        abs_image_path = os.path.abspath(image_path)
        if not os.path.exists(abs_image_path):
            return f'Error: Image not found at {abs_image_path}'

        image_dir = os.path.dirname(abs_image_path)
        image_file = os.path.basename(abs_image_path)

        uid_gid = f'{os.getuid()}:{os.getgid()}'
        
        # Use legacy runtime to avoid CDI errors on host
        cmd = [
            'docker', 'run', '--rm',
            '--runtime', "nvidia",
            '-e', 'NVIDIA_VISIBLE_DEVICES=1',
            '-u', uid_gid,
            '-e', 'HF_HOME=/tmp/.cache',
            '-e', 'PYTHONPATH=/scripts',
            '-v', f'{self.models_base}:/models',
            '-v', f'{image_dir}:/data',
            '-v', f'{self.scripts_dir}:/scripts',
            'aeon_vision:latest',
            'python3', '/scripts/vision_inference.py',
            '--models_dir', '/models',
            '--image_path', f'/data/{image_file}',
            '--prompt', prompt
        ]

        try:
            print(f'Aeon Vision: Running query on {image_file} (GPU 1)...')
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            output = result.stdout
            if '--- VISION ANALYSIS START ---' in output:
                parts = output.split('--- VISION ANALYSIS START ---')
                if len(parts) > 1:
                    output = parts[1].split('--- VISION ANALYSIS END ---')[0].strip()
            
            print(f'\n{self.C_CYAN}Vision Result:{self.C_RESET}\n{output}\n')
            
            return output
        except subprocess.CalledProcessError as e:
            return f'Vision System Error:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}'
        except Exception as e:
            return f'Unexpected Error: {e}'
