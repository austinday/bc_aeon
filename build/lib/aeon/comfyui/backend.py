import json
import os
import re
import subprocess
import time
import requests
from pathlib import Path

class ComfyUIBackend:
    def __init__(self, output_dir='/host_output'):
        self.project_root = Path('/home/aday/bc_aeon')
        self.output_dir = self.project_root / 'comfyui_output'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.container_name = 'aeon_comfyui'
        self.host_port = 8188
        self.model_workflows = {
            'flux_image': {
                'workflow_path': self.project_root / 'aeon/comfyui/workflows/flux_image_api.json',
                'profile_path': self.project_root / 'aeon/comfyui/profiles/flux_image.json'
            }
        }

    def _load_json(self, path):
        with open(path) as f:
            return json.load(f)

    def _recursive_replace(self, obj, params):
        if isinstance(obj, dict):
            return {k: self._recursive_replace(v, params) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._recursive_replace(item, params) for item in obj]
        elif isinstance(obj, str):
            for key, value in params.items():
                placeholder = f'{{{{{key}}}}}'  # {{WIDTH}}
                obj = re.sub(re.escape(placeholder), str(value), obj)
            return obj
        else:
            return obj

    def _substitute_params(self, workflow, params):
        print('DEBUG: Pre-sub workflow JSON:')
        print(json.dumps(workflow, indent=2))
        substituted = self._recursive_replace(workflow, params)
        print('DEBUG: Post-sub workflow JSON:')
        print(json.dumps(substituted, indent=2))
        return substituted

    def start_container(self):
        subprocess.run(['docker', 'rm', '-f', self.container_name], capture_output=True)
        docker_cmd = [
            'docker', 'run', '-d',
            '--name', self.container_name,
            '--gpus', 'device=1',
            '-p', f'{self.host_port}:8188',
            '-v', f'{self.project_root.absolute()}:/app',
            '-v', f'{self.output_dir.absolute()}:/host_output',
            '-e', 'CUDA_VISIBLE_DEVICES=1',
            'aeon/comfyui:latest',
            '/opt/ComfyUI/main.py', '--listen', '0.0.0.0', '--port', '8188'
        ]
        print('Starting container:', ' '.join(docker_cmd))
        result = subprocess.run(docker_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print('Docker start stderr:', result.stderr)
            raise RuntimeError('Failed to start container')
        # Health check
        for _ in range(60):
            try:
                resp = requests.get(f'http://localhost:{self.host_port}/health', timeout=5)
                if resp.status_code == 200:
                    print('Container healthy')
                    return
            except:
                pass
            time.sleep(1)
        raise RuntimeError('Container health check failed')

    def stop_container(self):
        subprocess.run(['docker', 'stop', self.container_name], capture_output=True)
        subprocess.run(['docker', 'rm', self.container_name], capture_output=True)

    def run_model(self, model_id, params={}):
        config = self.model_workflows[model_id]
        workflow = self._load_json(config['workflow_path'])
        profile = self._load_json(config['profile_path'])
        full_params = {**profile.get('defaults', {}), **params}
        workflow = self._substitute_params(workflow, full_params)
        self.start_container()
        try:
            prompt_resp = requests.post(f'http://localhost:{self.host_port}/prompt', json={'prompt': workflow})
            print('POST /prompt status:', prompt_resp.status_code)
            if prompt_resp.status_code != 200:
                print('ComfyUI error response:', prompt_resp.text)
                print('Container logs:')
                logs = subprocess.run(['docker', 'logs', '--tail=50', self.container_name], capture_output=True, text=True)
                print(logs.stdout)
                print(logs.stderr)
                prompt_resp.raise_for_status()
            # Poll /history
            prompt_id = list(workflow['3'].keys())[0]  # Assume last node ID
            for _ in range(300):  # 5min timeout
                hist_resp = requests.get(f'http://localhost:{self.host_port}/history/{prompt_id}')
                if hist_resp.status_code == 200 and hist_resp.json():
                    history = hist_resp.json()[prompt_id]
                    outputs = history['outputs']
                    # Find SaveImage filename
                    for node_id, node_out in outputs.items():
                        if 'images' in node_out:
                            img_filenames = [img['filename'] for img in node_out['images']]
                            img_path = Path('/host_output') / img_filenames[0]
                            local_path = self.output_dir / img_filenames[0]
                            subprocess.run(['docker', 'cp', f'{self.container_name}:/host_output/{img_filenames[0]}', str(local_path)], check=True)
                            self.stop_container()
                            return {'image_path': str(local_path)}
                time.sleep(2)
            raise RuntimeError('Workflow timeout')
        finally:
            self.stop_container()
