import os
import json
import time
import base64
import requests
import subprocess
from PIL import Image
import io
from .base import BaseTool
from ..core.prompts import TOOL_DESC_ANALYZE_IMAGE


class AnalyzeImageTool(BaseTool):
    """A tool to analyze images using Qwen3.6-35B-A3B-Uncensored via a local vLLM server."""

    # Max image dimension before resizing (keeps VRAM usage reasonable)
    MAX_IMAGE_DIM = 640
    # Port for the dedicated vision llama.cpp server
    VLLM_PORT = 8020

    def __init__(self):
        super().__init__(
            name='analyze_image',
            description=TOOL_DESC_ANALYZE_IMAGE,
            underlying_model='Qwen3.6-35B-A3B-VL'
        )
        self.vllm_url = f'http://localhost:{self.VLLM_PORT}'

    def _check_health(self):
        """Check if the vLLM vision server is healthy."""
        try:
            resp = requests.get(f'{self.vllm_url}/health', timeout=3)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _manage_registry(self, action: str):
        """Manage active users of the vision server using a lockfile and JSON registry."""
        import fcntl
        registry_path = '/tmp/aeon_vision_vllm_registry.json'
        lock_path = '/tmp/aeon_vision_vllm_registry.lock'
        pid = os.getpid()
        active_pids = []

        with open(lock_path, 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                if os.path.exists(registry_path):
                    with open(registry_path, 'r') as f:
                        active_pids = json.load(f)
            except (json.JSONDecodeError, EOFError):
                pass

            # Clean up dead PIDs
            cleaned_pids = []
            for p in active_pids:
                try:
                    os.kill(p, 0)
                    cleaned_pids.append(p)
                except OSError:
                    pass

            if action == 'register':
                if pid not in cleaned_pids:
                    cleaned_pids.append(pid)
            elif action == 'unregister':
                if pid in cleaned_pids:
                    cleaned_pids.remove(pid)

            with open(registry_path, 'w') as f:
                json.dump(cleaned_pids, f)

            return len(cleaned_pids)

    def _load_and_encode_image(self, image_path: str) -> tuple:
        """Load an image, resize if needed, and return (base64_str, mime_type)."""
        img = Image.open(image_path)

        # Convert RGBA/palette to RGB for JPEG compatibility
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')

        # Resize if either dimension exceeds the max
        w, h = img.size
        if max(w, h) > self.MAX_IMAGE_DIM:
            scale = self.MAX_IMAGE_DIM / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Encode to JPEG for smaller payload
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return b64, 'image/jpeg'

    def execute(self, image_path: str, prompt: str) -> str:
        if not image_path:
            return "Error: 'image_path' parameter is required."
        if not prompt:
            return "Error: 'prompt' parameter is required."

        abs_image_path = os.path.abspath(image_path)
        if not os.path.exists(abs_image_path):
            return f'Error: Image not found at {abs_image_path}'

        try:
            self._manage_registry('register')

            # Start server if not running
            if not self._check_health():
                # Try package directory first, then fallback to local project root
                pkg_script_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '..', 'scripts', 'start_qwen36_vl_35b.sh')
                )
                if os.path.exists(pkg_script_path):
                    script_path = pkg_script_path
                else:
                    # Fallback to current working directory (assuming we are at project root)
                    local_script_path = os.path.abspath(
                        os.path.join(os.getcwd(), 'aeon', 'scripts', 'start_qwen36_vl_35b.sh')
                    )
                    if os.path.exists(local_script_path):
                        script_path = local_script_path
                    else:
                        return f'Error: Vision start script not found. Tried {pkg_script_path} and {local_script_path}'

                env = os.environ.copy()
                env["AEON_HOME"] = os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))
                res = subprocess.run(['bash', script_path], capture_output=True, text=True, env=env)
                if res.returncode != 0:
                    return f'Error starting vision server: {res.stderr}'

                for attempt in range(60):  # Up to 3 minutes for Q8 GGUF loading
                    if self._check_health():
                        break
                    time.sleep(3)
                else:
                    return 'Error: Vision server failed to become healthy after 3 minutes. Check: docker logs aeon_qwen36_vl'

            # Load and encode image
            try:
                b64_image, mime_type = self._load_and_encode_image(abs_image_path)
            except Exception as e:
                return f'Error loading image: {type(e).__name__}: {e}'

            # Build OpenAI-compatible vision request
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:{mime_type};base64,{b64_image}'
                            }
                        },
                        {
                            'type': 'text',
                            'text': prompt
                        }
                    ]
                }
            ]


            resp = requests.post(
                f'{self.vllm_url}/v1/chat/completions',
                json={
                    'model': 'Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P',
                    'messages': messages,
                    'max_tokens': 2048,
                    'temperature': 0.3,
                },
                timeout=120
            )

            if resp.status_code != 200:
                return f'Error from vision server (HTTP {resp.status_code}): {resp.text[:500]}'

            result = resp.json()
            try:
                answer = result['choices'][0]['message']['content']
            except (KeyError, IndexError):
                return f'Error: Unexpected response format from vision server: {json.dumps(result)[:500]}'

            # Keep the full output, including thinking tags, as requested by the user.
            answer = answer.strip()

            full_output = (
                f"Prompt: {prompt}\n"
                f"Image: {abs_image_path}\n"
                f"Analysis:\n{answer}"
            )
            print(f'{self.C_GREEN}--- Vision Analysis ---')
            print(full_output)
            print(f'-----------------------{self.C_RESET}')
            return full_output

        except Exception as e:
            return self.format_error_message(e, 'analyzing image via Qwen3.6-35B', 'checking vision server logs (docker logs aeon_qwen36_vl)')

        finally:
            remaining_users = self._manage_registry('unregister')
            if remaining_users == 0:
                subprocess.run(['docker', 'rm', '-f', 'aeon_qwen36_vl'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
