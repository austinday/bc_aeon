import os
import json
import time
import random
import requests
import subprocess
import shutil
from .base import BaseTool
from ..core.prompts import TOOL_DESC_GENERATE_VIDEO

class GenerateVideoTool(BaseTool):
    """A tool to generate and edit video using LTX-2.3 GGUF via a local ComfyUI instance."""
    
    def __init__(self):
        super().__init__(
            name="generate_video",
            description=TOOL_DESC_GENERATE_VIDEO,
            underlying_model='LTX-2.3-22B GGUF'
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

    def execute(self, mode: str, prompt: str, output_path: str, input_path_1: str = None, input_path_2: str = None, width: int = 768, height: int = 512, frames: int = 33) -> str:
        if not prompt:
            return "Error: 'prompt' parameter is required."
        if not output_path:
            return "Error: 'output_path' parameter is required."
        
        valid_modes = ['text_to_video', 'image_to_video', 'video_extension', 'interpolate']
        if mode not in valid_modes:
            return f"Error: Invalid mode. Must be one of {valid_modes}"
            
        if mode in ['image_to_video', 'video_extension', 'interpolate'] and not input_path_1:
            return f"Error: 'input_path_1' is required for mode '{mode}'."
            
        if mode == 'interpolate' and not input_path_2:
            return "Error: 'input_path_2' is required for mode 'interpolate'."

        abs_output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

        try:
            self._manage_registry('register')
            
            if not self._check_comfyui_health():
                print(f"{self.C_CYAN}Starting ComfyUI server (this takes a moment)...{self.C_RESET}")
                script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "start_comfyui.sh"))
                env = os.environ.copy()
                env["AEON_HOME"] = os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))
                res = subprocess.run(["bash", script_path], capture_output=True, text=True, env=env)
                if res.returncode != 0:
                    return f"Error starting ComfyUI: {res.stderr}"
                
                print(f"{self.C_CYAN}Waiting for ComfyUI to become healthy...{self.C_RESET}")
                for _ in range(60):
                    if self._check_comfyui_health():
                        break
                    time.sleep(2)
                else:
                    return "Error: ComfyUI failed to become healthy after starting."

            # Upload inputs if provided
            uploaded_file_1 = None
            uploaded_file_2 = None
            
            if input_path_1:
                abs_in1 = os.path.abspath(input_path_1)
                if not os.path.exists(abs_in1):
                    return f"Error: Input file not found: {abs_in1}"
                with open(abs_in1, 'rb') as f:
                    up_res = requests.post(f"{self.comfy_url}/upload/image", files={"image": f}, timeout=10)
                if up_res.status_code != 200:
                    return f"Error uploading input_path_1 to ComfyUI: {up_res.text}"
                uploaded_file_1 = up_res.json()["name"]
                
            if input_path_2:
                abs_in2 = os.path.abspath(input_path_2)
                if not os.path.exists(abs_in2):
                    return f"Error: Input file 2 not found: {abs_in2}"
                with open(abs_in2, 'rb') as f:
                    up_res = requests.post(f"{self.comfy_url}/upload/image", files={"image": f}, timeout=10)
                if up_res.status_code != 200:
                    return f"Error uploading input_path_2 to ComfyUI: {up_res.text}"
                uploaded_file_2 = up_res.json()["name"]

            # Construct LTX-2.3 Workflow 
            workflow = {
                "1": {
                    "class_type": "UnetLoaderGGUF",
                    "inputs": {"unet_name": "ltx-2.3-22b-dev-F16.gguf"}
                },
                "2": {
                    "class_type": "VAELoader",
                    "inputs": {"vae_name": "vae/ltx-2.3-22b-dev_video_vae.safetensors"}
                },
                "3": {
                    "class_type": "DualCLIPLoaderGGUF",
                    "inputs": {
                        "clip_name1": "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors",
                        "clip_name2": "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
                        "type": "ltxv"
                    }
                },
                "4": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "text": prompt,
                        "clip": ["3", 0]
                    }
                },
                "5": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "text": "worst quality, inconsistent, blurry, deformed, mutated",
                        "clip": ["3", 0]
                    }
                },
                "7": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": random.randint(1, 0xffffffffffffffff),
                        "steps": 25,
                        "cfg": 3.0,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "denoise": 1.0,
                        "model": ["1", 0],
                        "positive": ["4", 0],
                        "negative": ["5", 0],
                        "latent_image": ["6", 0]
                    }
                },
                "8": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["7", 0], "vae": ["2", 0]}
                },
                "9": {
                    "class_type": "VHS_VideoCombine",
                    "inputs": {
                        "frame_rate": 24,
                        "loop_count": 0,
                        "filename_prefix": "Aeon_Video",
                        "format": "video/h264-mp4",
                        "pingpong": False,
                        "save_output": True,
                        "images": ["8", 0]
                    }
                }
            }

            # Inject latents based on mode
            if mode == 'text_to_video':
                workflow["6"] = {
                    "class_type": "EmptyLatentImage",
                    "inputs": {"batch_size": frames, "width": width, "height": height}
                }
            
            elif mode == 'image_to_video':
                workflow["10"] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": uploaded_file_1}
                }
                workflow["11"] = {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": ["10", 0], "vae": ["2", 0]}
                }
                workflow["6"] = {
                    "class_type": "LatentComposite",
                    "inputs": {
                        "samples_from": ["11", 0],
                        "samples_to": {
                            "class_type": "EmptyLatentImage", 
                            "inputs": {"batch_size": frames, "width": width, "height": height}
                        },
                        "x": 0, "y": 0, "feather": 0
                    }
                }
                workflow["7"]["inputs"]["denoise"] = 0.95
                
            elif mode == 'video_extension':
                workflow["10"] = {
                    "class_type": "VHS_LoadVideo",
                    "inputs": {"video": uploaded_file_1, "force_rate": 0, "force_size": "Disabled", "custom_width": width, "custom_height": height, "frame_load_cap": frames}
                }
                workflow["11"] = {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": ["10", 0], "vae": ["2", 0]}
                }
                workflow["6"] = {"class_type": "LatentComposite", "inputs": {"samples_from": ["11", 0], "samples_to": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": frames * 2, "width": width, "height": height}}, "x": 0, "y": 0, "feather": 0}}
                workflow["7"]["inputs"]["denoise"] = 0.85
                
            elif mode == 'interpolate':
                workflow["10"] = {"class_type": "VHS_LoadVideo", "inputs": {"video": uploaded_file_1, "force_rate": 0, "force_size": "Disabled"}}
                workflow["11"] = {"class_type": "VHS_LoadVideo", "inputs": {"video": uploaded_file_2, "force_rate": 0, "force_size": "Disabled"}}
                workflow["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["2", 0]}}
                workflow["13"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["2", 0]}}
                workflow["6"] = {"class_type": "LatentComposite", "inputs": {"samples_from": ["12", 0], "samples_to": ["13", 0], "x": 0, "y": 0, "feather": 0}}
                workflow["7"]["inputs"]["denoise"] = 0.70

            print(f"{self.C_CYAN}Submitting video generation workflow ({mode}) to ComfyUI...{self.C_RESET}")
            req = requests.post(f"{self.comfy_url}/prompt", json={"prompt": workflow}, timeout=5)
            if req.status_code != 200:
                return f"Error submitting workflow to ComfyUI: {req.text}"
            
            prompt_id = req.json()["prompt_id"]
            
            print(f"{self.C_CYAN}Waiting for video generation to complete (this may take a long time)...{self.C_RESET}")
            max_retries = 360 # 18 minutes max wait for video
            for _ in range(max_retries):
                history_req = requests.get(f"{self.comfy_url}/history/{prompt_id}", timeout=5)
                history = history_req.json()
                
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    if "9" not in outputs:
                        return "Error: ComfyUI execution completed but VideoCombine node did not produce output. Check server logs."
                    
                    video_info = outputs["9"]["videos"][0]
                    filename = video_info["filename"]
                    subfolder = video_info["subfolder"]
                    folder_type = video_info["type"]
                    
                    vid_req = requests.get(
                        f"{self.comfy_url}/view?filename={filename}&subfolder={subfolder}&type={folder_type}",
                        timeout=30
                    )
                    
                    if vid_req.status_code == 200:
                        with open(abs_output_path, "wb") as f:
                            f.write(vid_req.content)
                        return f"Successfully generated video and saved to: {abs_output_path}"
                    else:
                        return f"Error: Failed to download the generated video from ComfyUI (HTTP {vid_req.status_code})"
                
                time.sleep(3)
                
            return "Error: Video generation timed out after 18 minutes."
            
        except Exception as e:
            return self.format_error_message(e, "generating video via ComfyUI", "checking if ComfyUI is running correctly and has required custom nodes.")
        
        finally:
            remaining_users = self._manage_registry('unregister')
            if remaining_users == 0:
                print(f"{self.C_CYAN}Last agent finished. Releasing GPU memory (stopping ComfyUI)...{self.C_RESET}")
                subprocess.run(["docker", "rm", "-f", "aeon_comfyui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print(f"{self.C_CYAN}Video generation complete. Leaving ComfyUI running for {remaining_users} other active agent(s)...{self.C_RESET}")
