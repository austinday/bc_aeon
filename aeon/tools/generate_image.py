import os
import json
import time
import random
import requests
import subprocess
from .base import BaseTool
from ..core.prompts import TOOL_DESC_GENERATE_IMAGE, TOOL_DESC_EDIT_IMAGE
from ..core.gpu_queue import wait_for_vram, release_vram

class ComfyUITool(BaseTool):
    """Base class for tools using ComfyUI to handle VRAM and registry management."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comfy_url = "http://localhost:8188"

    def _check_comfyui_health(self):
        try:
            res = requests.get(f"{self.comfy_url}/system_stats", timeout=2)
            return res.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _manage_registry(self, action: str, gpu_id: int = None):
        """Manage active users of ComfyUI and track the assigned GPU."""
        import fcntl
        registry_path = "/tmp/aeon_comfyui_registry.json"
        lock_path = "/tmp/aeon_comfyui_registry.lock"
        pid = os.getpid()
        
        with open(lock_path, 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                if os.path.exists(registry_path):
                    with open(registry_path, 'r') as f:
                        state = json.load(f)
                else:
                    state = {"pids": [], "gpu_id": None}
            except (json.JSONDecodeError, EOFError):
                state = {"pids": [], "gpu_id": None}
                
            # Clean up dead PIDs
            cleaned_pids = []
            for p in state.get("pids", []):
                try:
                    os.kill(p, 0)
                    cleaned_pids.append(p)
                except OSError:
                    pass
            state["pids"] = cleaned_pids
                    
            if action == 'register':
                if pid not in state["pids"]:
                    state["pids"].append(pid)
                if gpu_id is not None:
                    state["gpu_id"] = gpu_id
            elif action == 'unregister':
                if pid in state["pids"]:
                    state["pids"].remove(pid)
            
            with open(registry_path, 'w') as f:
                json.dump(state, f)
                
            return len(state["pids"]), state.get("gpu_id")

    def _ensure_comfyui_running(self, required_vram: float = 20.0):
        """Ensures ComfyUI is healthy and running on a GPU with sufficient VRAM."""
        # 1. Determine which GPU ComfyUI is currently using
        _, current_gpu = self._manage_registry('get_info') if hasattr(self, '_manage_registry') else (None, None)
        # Note: _manage_registry('get_info') is a conceptual addition, let's use the existing logic
        # Since _manage_registry returns (count, gpu_id) on 'register' and 'unregister', 
        # we can just call it with a dummy action or modify it. 
        # For now, let's just call it with 'register' to get the current state.
        
        # To avoid double-registering in this call, we'll just peek at the registry.
        import fcntl
        registry_path = "/tmp/aeon_comfyui_registry.json"
        lock_path = "/tmp/aeon_comfyui_registry.lock"
        current_gpu = None
        try:
            with open(lock_path, 'w') as lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                if os.path.exists(registry_path):
                    with open(registry_path, 'r') as f:
                        current_gpu = json.load(f).get("gpu_id")
        except: pass

        # 2. Reserve VRAM. If server is already on a GPU, we MUST wait for that specific GPU.
        print(f"{self.C_CYAN}Reserving {required_vram}GB VRAM (Target GPU: {current_gpu if current_gpu is not None else 'Any'})...{self.C_RESET}")
        allocated_gpu = wait_for_vram(required_vram, gpu_id=current_gpu)
        print(f"{self.C_CYAN}VRAM reserved on GPU {allocated_gpu}.{self.C_RESET}")

        # 3. Check if server is healthy AND on the correct GPU
        if self._check_comfyui_health() and allocated_gpu == current_gpu:
            return True

        # 4. If not healthy or on wrong GPU, restart it on the allocated GPU
        print(f"{self.C_CYAN}Starting/Restarting ComfyUI on GPU {allocated_gpu}...{self.C_RESET}")
        self._manage_registry('register', gpu_id=allocated_gpu)
        
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "start_comfyui.sh"))
        env = os.environ.copy()
        env["AEON_HOME"] = os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))
        env["COMFYUI_GPU"] = str(allocated_gpu)
        res = subprocess.run(["bash", script_path], capture_output=True, text=True, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"Error starting ComfyUI: {res.stderr}")
        
        print(f"{self.C_CYAN}Waiting for ComfyUI to become healthy...{self.C_RESET}")
        for _ in range(60):
            if self._check_comfyui_health():
                return True
            time.sleep(2)
        
        raise RuntimeError("Error: ComfyUI failed to become healthy after starting.")

class GenerateImageTool(ComfyUITool):
    """A tool to generate images using FLUX GGUF via a local ComfyUI instance."""
    def __init__(self):
        super().__init__(
            name="generate_image",
            description=TOOL_DESC_GENERATE_IMAGE,
            underlying_model='FLUX.2 GGUF'
        )

    def execute(self, prompt: str, output_path: str, width: int = 1024, height: int = 1024) -> str:
        if not prompt:
            return "Error: 'prompt' parameter is required."
        if not output_path:
            return "Error: 'output_path' parameter is required."

        abs_output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

        try:
            # Register this agent as an active user and ensure server is running on allocated VRAM
            self._manage_registry('register')
            self._ensure_comfyui_running(required_vram=20.0)

            workflow = {
                "1": {
                    "class_type": "UnetLoaderGGUF",
                    "inputs": {
                        "unet_name": "FHDR_ComfyUI-Q8_0.gguf"
                    }
                },
                "2": {
                    "class_type": "DualCLIPLoader",
                    "inputs": {
                        "clip_name1": "clip_l.safetensors",
                        "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                        "type": "flux"
                    }
                },
                "3": {
                    "class_type": "VAELoader",
                    "inputs": {
                        "vae_name": "ae.safetensors"
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
                        "filename_prefix": "Aeon",
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
            release_vram()
            if remaining_users == 0:
                print(f"{self.C_CYAN}Last agent finished. (Container deletion disabled for debugging)...{self.C_RESET}")
                # subprocess.run(["docker", "rm", "-f", "aeon_comfyui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print(f"{self.C_CYAN}Image complete. Leaving ComfyUI running for {remaining_users} other active agent(s)...{self.C_RESET}")


class EditImageTool(ComfyUITool):
    """A tool to edit images using Qwen-Image-Edit GGUF via a local ComfyUI instance."""
    def __init__(self):
        super().__init__(
            name="edit_image",
            description=TOOL_DESC_EDIT_IMAGE,
            underlying_model='Qwen-Image-Edit-Rapid'
        )

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
            # Register this agent as an active user and ensure server is running on allocated VRAM
            self._manage_registry('register')
            self._ensure_comfyui_running(required_vram=20.0)

            print(f"{self.C_CYAN}Uploading input image to ComfyUI...{self.C_RESET}")
            with open(abs_input_path, 'rb') as f:
                upload_res = requests.post(f"{self.comfy_url}/upload/image", files={"image": f}, timeout=10)
            
            if upload_res.status_code != 200:
                return f"Error uploading image to ComfyUI: {upload_res.text}"
            
            uploaded_filename = upload_res.json()["name"]

            workflow = {
                "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "v23/Qwen-Rapid-NSFW-v23_Q8_0.gguf"}},
                "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
                "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
                "10": {"class_type": "LoadImage", "inputs": {"image": uploaded_filename}},
                "4": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": prompt, "clip": ["2", 0], "vae": ["3", 0], "image1": ["10", 0]}},
                "5": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": "", "clip": ["2", 0], "vae": ["3", 0], "image1": ["10", 0]}},
                "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["3", 0]}},
                "7": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": random.randint(1, 0xffffffffffffffff),
                        "steps": 8,
                        "cfg": 4.0,
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
                "9": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "filename_prefix": "Aeon_Edit",
                        "images": ["8", 0]
                    }
                }
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
            # Unregister and check if we are the last active user
            remaining_users = self._manage_registry('unregister')
            release_vram()
            if remaining_users == 0:
                print(f"{self.C_CYAN}Last agent finished. (Container deletion disabled for debugging)...{self.C_RESET}")
            else:
                print(f"{self.C_CYAN}Image edit complete. Leaving ComfyUI running for {remaining_users} other active agent(s)...{self.C_RESET}")
