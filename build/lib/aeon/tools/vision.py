import os
import subprocess
from pathlib import Path
from .base import BaseTool

class ImageViewerTool(BaseTool):
    """Agent-driven vision tool for targeted image analysis."""
    def __init__(self):
        super().__init__(
            name='image_viewer',
            description='Analyzes an image file using an AI vision model. You must provide a specific `prompt` describing what you want to extract or observe. Params: `image_path` (str), `prompt` (str). Example: `{"tool_name": "image_viewer", "parameters": {"image_path": "plot.png", "prompt": "What is the value of the highest bar?"}}`'
        )
        self.models_base = Path.home() / 'bc_aeon' / 'aeon_models'
        self.scripts_dir = Path.home() / 'bc_aeon' / 'aeon' / 'scripts'

    def execute(self, image_path: str, prompt: str):
        if not image_path or not prompt:
            return 'Error: Both `image_path` and `prompt` are required.'

        abs_image_path = os.path.abspath(image_path)
        if not os.path.exists(abs_image_path):
            return f'Error: Image not found at {abs_image_path}'

        image_dir = os.path.dirname(abs_image_path)
        image_file = os.path.basename(abs_image_path)

        uid_gid = f'{os.getuid()}:{os.getgid()}'
        
        cmd = [
            'docker', 'run', '--rm', '--gpus', 'all',
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
            print(f'Aeon Vision: Running agent-provided query on {image_file}...')
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            output = result.stdout
            if '--- VISION ANALYSIS START ---' in output:
                parts = output.split('--- VISION ANALYSIS START ---')
                if len(parts) > 1:
                    output = parts[1].split('--- VISION ANALYSIS END ---')[0].strip()
            
            # FIX: Print the result to the user's terminal for verbosity
            print(f'\n{self.C_CYAN}Vision Result:{self.C_RESET}\n{output}\n')
            
            return output
        except subprocess.CalledProcessError as e:
            return f'Vision System Error:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}'
        except Exception as e:
            return f'Unexpected Error: {e}'