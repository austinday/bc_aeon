import os
import json
import time
import random
import requests
import subprocess
from .base import BaseTool
from ..core.prompts import TOOL_DESC_GENERATE_IMAGE, TOOL_DESC_EDIT_IMAGE

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

    def _manage_registry(self, action: str):
        """Manage active users of ComfyUI using a lockfile and JSON registry."""
        import fcntl
        registry_path = "/tmp/aeon_comfyui_registry.json"
        lock_path = "/tmp/aeon_comfyui_registry.lock"
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

    def execute(self, prompt: str, output_path: str, width: int = 1024, height: int = 1024) -> str:
        if not prompt:
            return "Error: 'prompt' parameter is required."
        if not output_path:
            return "Error: 'output_path' parameter is required."

        abs_output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

        try:
            # Register this agent as an active user of ComfyUI
            self._manage_registry('register')
            
            if not self._check_comfyui_health():
                print(f"{self.C_CYAN}Starting ComfyUI server (this takes a moment)...{self.C_RESET}")
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "start_comfyui.sh"))
                res = subprocess.run(["bash", script_path], capture_output=True, text=True)
                if res.returncode != 0:
                    return f"Error starting ComfyUI: {res.stderr}"
                
                print(f"{self.C_CYAN}Waiting for ComfyUI to become healthy...{self.C_RESET}")
                for _ in range(60):
                    if self._check_comfyui_health():
                        break
                    time.sleep(2)
                else:
                    return "Error: ComfyUI failed to become healthy after starting."

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
                    "class_type": "PreviewImage",
                    "inputs": {
                        "images": ["8", 0]
                    }
                }
            }

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
        
        finally:
            # Unregister and check if we are the last active user
            remaining_users = self._manage_registry('unregister')
            if remaining_users == 0:
                print(f"{self.C_CYAN}Last agent finished. Releasing GPU memory (stopping ComfyUI)...{self.C_RESET}")
                subprocess.run(["docker", "rm", "-f", "aeon_comfyui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print(f"{self.C_CYAN}Image complete. Leaving ComfyUI running for {remaining_users} other active agent(s)...{self.C_RESET}")


class EditImageTool(BaseTool):
    """A tool to edit images using FLUX GGUF via a local ComfyUI instance."""
    def __init__(self):
        super().__init__(
            name="edit_image",
            description=TOOL_DESC_EDIT_IMAGE
        )
        self.comfy_url = "http://localhost:8188"

    def _check_comfyui_health(self):
        try:
            res = requests.get(f"{self.comfy_url}/system_stats", timeout=2)
            return res.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _manage_registry(self, action: str):
        import fcntl
        registry_path = "/tmp/aeon_comfyui_registry.json"
        lock_path = "/tmp/aeon_comfyui_registry.lock"
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

    def execute(self, input_path: str, prompt: str, output_path: str, denoise: float = 0.75) -> str:
        if not input_path:
            return "Error: 'input_path' parameter is required."
        if not prompt:
            return "Error: 'prompt' parameter is required."
        if not output_path:
            return "Error: 'output_path' parameter is required."

        abs_input_path = os.path.abspath(input_path)
        abs_output_path = os.path.abspath(output_path)

        if not os.path.exists(abs_input_path):
            return f"Error: Input image not found at {abs_input_path}"

        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

        try:
            self._manage_registry('register')
            
            if not self._check_comfyui_health():
                print(f"{self.C_CYAN}Starting ComfyUI server (this takes a moment)...{self.C_RESET}")
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "start_comfyui.sh"))
                res = subprocess.run(["bash", script_path], capture_output=True, text=True)
                if res.returncode != 0:
                    return f"Error starting ComfyUI: {res.stderr}"
                
                print(f"{self.C_CYAN}Waiting for ComfyUI to become healthy...{self.C_RESET}")
                for _ in range(60):
                    if self._check_comfyui_health():
                        break
                    time.sleep(2)
                else:
                    return "Error: ComfyUI failed to become healthy after starting."

            print(f"{self.C_CYAN}Uploading input image to ComfyUI...{self.C_RESET}")
            with open(abs_input_path, 'rb') as f:
                upload_res = requests.post(f"{self.comfy_url}/upload/image", files={"image": f}, timeout=10)
            
            if upload_res.status_code != 200:
                return f"Error uploading image to ComfyUI: {upload_res.text}"
            
            uploaded_filename = upload_res.json()["name"]

            workflow = {
                "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux2-dev-Q4_K_S.gguf"}},
                "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2"}},
                "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
                "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
                "10": {"class_type": "LoadImage", "inputs": {"image": uploaded_filename}},
                "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["3", 0]}},
                "7": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": random.randint(1, 0xffffffffffffffff),
                        "steps": 25,
                        "cfg": 1.0,
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "denoise": denoise,
                        "model": ["1", 0],
                        "positive": ["4", 0],
                        "negative": ["5", 0],
                        "latent_image": ["11", 0]
                    }
                },
                "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
                "9": {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}}
            }

            print(f"{self.C_CYAN}Submitting image edit workflow to ComfyUI...{self.C_RESET}")
            req = requests.post(f"{self.comfy_url}/prompt", json={"prompt": workflow}, timeout=5)
            if req.status_code != 200:
                return f"Error submitting workflow to ComfyUI: {req.text}"
            
            prompt_id = req.json()["prompt_id"]
            
            print(f"{self.C_CYAN}Waiting for image editing to complete...{self.C_RESET}")
            max_retries = 120
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
                        return f"Successfully edited image and saved to: {abs_output_path}"
                    else:
                        return f"Error: Failed to download the edited image from ComfyUI (HTTP {img_req.status_code})"
                
                time.sleep(3)
                
            return "Error: Image editing timed out after 6 minutes."
            
        except Exception as e:
            return self.format_error_message(e, "editing image via ComfyUI", "checking if ComfyUI is running correctly")
        
        finally:
            remaining_users = self._manage_registry('unregister')
            if remaining_users == 0:
                print(f"{self.C_CYAN}Last agent finished. Releasing GPU memory (stopping ComfyUI)...{self.C_RESET}")
                subprocess.run(["docker", "rm", "-f", "aeon_comfyui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print(f"{self.C_CYAN}Image edit complete. Leaving ComfyUI running for {remaining_users} other active agent(s)...{self.C_RESET}")
