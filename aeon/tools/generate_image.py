import os
import json
import time
import random
import requests
import subprocess
from .base import BaseTool
from ..core.prompts import TOOL_DESC_GENERATE_IMAGE, TOOL_DESC_EDIT_IMAGE
from ..core.gpu_queue import wait_for_vram, release_vram
from ..core.prompt_enhancer import enhance_prompt
from ..core.paths import resolve_output_path

class ComfyUITool(BaseTool):
    """Base class for tools using ComfyUI to handle VRAM and registry management."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comfy_url = "http://localhost:8188"

    @staticmethod
    def _norm_dim(value, default=1024, lo=256, hi=2048, multiple=16):
        """Coerce a model-supplied width/height into a valid int: tolerate string
        numbers, clamp to [lo, hi], and round to a multiple the model accepts.
        Falls back to `default` on garbage input."""
        try:
            v = int(round(float(value)))
        except (TypeError, ValueError):
            return default
        v = max(lo, min(hi, v))
        return max(lo, (v // multiple) * multiple)

    @staticmethod
    def _norm_unit(value, default=0.75):
        """Coerce a 0..1 strength/denoise value: tolerate strings, clamp to [0,1],
        fall back to default on garbage."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, v))

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
            
            count = len(state["pids"])
            pass
            return count, state.get("gpu_id")

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
    def __init__(self, llm_client=None):
        super().__init__(
            name="generate_image",
            description=TOOL_DESC_GENERATE_IMAGE,
            underlying_model='FLUX.2-klein-9B uncensored GGUF'
        )
        self.llm_client = llm_client

    def _resolve(self, subdir: str, patterns, default: str) -> str:
        """Basename of the first model in comfyui/<subdir> matching any pattern; else default."""
        import glob
        base = os.path.join(os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon")),
                            "models", "comfyui", subdir)
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(base, pat)))
            if hits:
                return os.path.basename(hits[0])
        return default

    def _flux2_models(self):
        """Resolve the uncensored FLUX.2 model set (auto-adapts to whichever quant is present)."""
        unet = self._resolve("unet", ("*flux*2*klein*.gguf", "*flux-2*.gguf", "*flux*klein*.gguf"),
                             "flux-2-klein-9b-Q8_0.gguf")
        clip = self._resolve("text_encoders", ("*flux2*uncensored*.gguf", "*flux*2*klein*uncensored*.gguf"),
                             "flux2-klein-9b-uncensored-q8_0.gguf")
        vae = self._resolve("vae", ("*flux2*vae*.safetensors", "flux2-vae*.safetensors"),
                            "flux2-vae.safetensors")
        return unet, clip, vae

    def execute(self, prompt: str, output_path: str = None, width: int = 1024, height: int = 1024, enhance: bool = None) -> str:
        if not prompt:
            return "Error: 'prompt' parameter is required."

        # Tolerate string/odd dimensions from the model (e.g. "1024" or 1000).
        width = self._norm_dim(width)
        height = self._norm_dim(height)

        prompt = enhance_prompt(self.llm_client, prompt, "image", force=enhance)
        # Resolve relative to the workspace (where aeon was launched), or
        # auto-name at the workspace base when no path is given.
        abs_output_path = str(resolve_output_path(output_path, time.strftime("aeon_image_%Y%m%d_%H%M%S.png")))
        os.makedirs(os.path.dirname(abs_output_path) or ".", exist_ok=True)

        try:
            # Register this agent as an active user and ensure server is running on allocated VRAM
            self._manage_registry('register')
            self._ensure_comfyui_running(required_vram=20.0)

            unet, clip, vae = self._flux2_models()
            seed = random.randint(1, 0xffffffffffffffff)
            # Uncensored FLUX.2-klein graph: GGUF model + flux2 (Mistral) uncensored text
            # encoder + flux2 VAE, sampled via the modern guider path (validated).
            workflow = {
                "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}},
                "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": clip, "type": "flux2"}},
                "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
                "5": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": 4.0}},
                "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
                "7": {"class_type": "Flux2Scheduler", "inputs": {"steps": 20, "width": width, "height": height}},
                "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
                "14": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
                "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
                "11": {"class_type": "SamplerCustomAdvanced",
                       "inputs": {"noise": ["10", 0], "guider": ["14", 0], "sampler": ["8", 0],
                                  "sigmas": ["7", 0], "latent_image": ["6", 0]}},
                "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
                "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Aeon", "images": ["12", 0]}},
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
            remaining_users, _ = self._manage_registry('unregister')
            release_vram()
            if remaining_users == 0:
                print(f"{self.C_CYAN}Last agent finished. Cleaning up container...{self.C_RESET}")
                subprocess.run(["docker", "rm", "-f", "aeon_comfyui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print(f"{self.C_CYAN}Image complete. Leaving ComfyUI running for {remaining_users} other active agent(s)...{self.C_RESET}")


class EditImageTool(ComfyUITool):
    """A tool to edit images using Qwen-Image-Edit GGUF via a local ComfyUI instance."""
    def __init__(self, llm_client=None):
        super().__init__(
            name="edit_image",
            description=TOOL_DESC_EDIT_IMAGE,
            underlying_model='Qwen-Image-Edit-Rapid'
        )
        self.llm_client = llm_client

    def execute(self, input_path: str, prompt: str, output_path: str = None, denoise: float = 0.75, enhance: bool = None) -> str:
        if not input_path:
            return "Error: 'input_path' parameter is required."
        if not prompt:
            return "Error: 'prompt' parameter is required."

        denoise = self._norm_unit(denoise, default=0.75)
        prompt = enhance_prompt(self.llm_client, prompt, "image_edit", force=enhance)
        abs_input_path = os.path.abspath(input_path)
        # Default: '<input-name>_edited.png' at the workspace base.
        default_name = os.path.splitext(os.path.basename(abs_input_path))[0] + "_edited.png"
        abs_output_path = str(resolve_output_path(output_path, default_name))

        if not os.path.exists(abs_input_path):
            return f"Error: Input image not found at {abs_input_path}"

        os.makedirs(os.path.dirname(abs_output_path) or ".", exist_ok=True)

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
                # ABLITERATED Qwen2.5-VL text encoder (was stock qwen_2.5_vl_7b_fp8_scaled,
                # the last censored component). CLIPLoaderGGUF auto-pairs the matching
                # mmproj-*.gguf alongside it (needed because Qwen-Image-Edit reads the input
                # image in vision-language mode), and ignores the unrelated gemma-3 mmproj.
                "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": "Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf", "type": "qwen_image"}},
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
            remaining_users, _ = self._manage_registry('unregister')
            release_vram()
            if remaining_users == 0:
                print(f"{self.C_CYAN}Last agent finished. Cleaning up container...{self.C_RESET}")
                subprocess.run(["docker", "rm", "-f", "aeon_comfyui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print(f"{self.C_CYAN}Image edit complete. Leaving ComfyUI running for {remaining_users} other active agent(s)...{self.C_RESET}")
