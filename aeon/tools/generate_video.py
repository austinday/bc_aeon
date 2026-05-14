import os
import json
import time
import requests
import subprocess
import shutil
from typing import List, Optional, Dict, Any, Union

from aeon.tools.base import BaseTool

class GenerateVideoTool(BaseTool):
    """
    Tool for generating high-quality videos using LTX-Video via ComfyUI.
    Supports text-to-video, image-to-video, and recursive long-video generation
    with support for prompt scheduling.
    """

    def __init__(self):
        super().__init__(
            name="generate_video",
            description="Generates high-quality videos using LTX-Video via ComfyUI. Supports text-to-video and image-to-video."
        )
        self.comfy_url = "http://localhost:8188"
        self.output_dir = "aeon_output/comfyui"
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_chunk_frames = 33  # LTX-Video optimal chunk size

    def _manage_registry(self, action: str) -> int:
        """
        Simulates registry management for ComfyUI containers.
        """
        return 1

    def _upload_image(self, image_path: str) -> str:
        """Uploads an image to ComfyUI and returns the filename used by the server."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Uploading image {image_path} to ComfyUI...")
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f)}
            response = requests.post(f"{self.comfy_url}/upload/image", files=files, timeout=60)
            response.raise_for_status()
            return response.json().get("name")

    def _extract_last_frame(self, video_path: str, output_image_path: str):
        """Extracts the last frame of a video using ffmpeg via docker."""
        cwd = os.getcwd()
        rel_video_path = os.path.relpath(video_path, cwd)
        rel_image_path = os.path.relpath(output_image_path, cwd)
        
        cmd = [
            "docker", "run", "--rm", 
            "-v", f"{cwd}:/app", 
            "-w", "/app",
            "mwader/static-ffmpeg", 
            "-sseof", "-1", 
            "-i", f"/app/{rel_video_path}", 
            "-update", "1", 
            "-q:v", "2", 
            f"/app/{rel_image_path}"
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _concatenate_videos(self, video_paths: List[str], output_path: str):
        """Concatenates multiple mp4 files into one using ffmpeg via docker."""
        cwd = os.getcwd()
        list_file_path = "aeon_output/debug/concat_list.txt"
        os.makedirs(os.path.dirname(list_file_path), exist_ok=True)
        
        with open(list_file_path, "w") as f:
            for path in video_paths:
                rel_path = os.path.relpath(path, cwd)
                f.write(f"file '{rel_path}'\n")

        rel_list_file = os.path.relpath(list_file_path, cwd)
        rel_output_path = os.path.relpath(output_path, cwd)

        cmd = [
            "docker", "run", "--rm", 
            "-v", f"{cwd}:/app", 
            "-w", "/app",
            "mwader/static-ffmpeg", 
            "-f", "concat", 
            "-safe", "0", 
            "-i", f"/app/{rel_list_file}", 
            "-c", "copy", 
            f"/app/{rel_output_path}"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(list_file_path):
            os.remove(list_file_path)

    def _get_workflow(self, mode: str, prompt: str, width: int, height: int, frames: int, uploaded_image_name: Optional[str] = None) -> Dict[str, Any]:
        """Constructs the ComfyUI workflow JSON for LTX-Video."""
        
        workflow = {
            "3": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {
                    "unet_name": "ltx-2.3-22b-dev-Q4_1.gguf"
                }
            },
            "7": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "t5xxl_fp8_e4m3fn.safetensors",
                    "type": "ltxv"
                }
            },
            "8": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "ltx-2.3-22b-dev_video_vae.safetensors"
                }
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["7", 0]
                }
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "low quality, blurry, distorted, static, slideshow, flickering, watermark, text",
                    "clip": ["7", 0]
                }
            },
            "6": {
                "class_type": "ModelSamplingLTXV",
                "inputs": {
                    "model": ["3", 0],
                    "max_shift": 2.05,
                    "base_shift": 0.95
                }
            },
            "10": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 30,
                    "cfg": 5.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["6", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["11", 0]
                }
            },
            "12": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["10", 0],
                    "vae": ["8", 0]
                }
            },
            "13": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": 24,
                    "loop_count": 0,
                    "filename_prefix": "AeonVideo",
                    "format": "video/h264-mp4",
                    "save_output": True,
                    "pingpong": False,
                    "images": ["12", 0]
                }
            }
        }

        if mode == "text_to_video":
            workflow["11"] = {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {
                    "width": width,
                    "height": height,
                    "length": frames,
                    "batch_size": 1
                }
            }
        elif mode in ["image_to_video", "video_extension"]:
            workflow["11"] = {
                "class_type": "LTXVImgToVideo",
                "inputs": {
                    "image": ["14", 0],
                    "width": width,
                    "height": height,
                    "length": frames,
                    "strength": 1.0
                }
            }
            workflow["14"] = {
                "class_type": "LoadImage",
                "inputs": {
                    "image": uploaded_image_name if uploaded_image_name else "default.png"
                }
            }
        
        return workflow

    def execute(self, mode: str, prompt: Union[str, List[str]], output_path: str, width: int = 768, height: int = 512, frames: int = 33, input_path_1: Optional[str] = None, **kwargs) -> str:
        """
        Executes video generation. 
        If frames > max_chunk_frames, it uses recursive generation.
        'prompt' can be a single string or a list of strings for prompt scheduling.
        """
        self._manage_registry("start")
        try:
            prompts = [prompt] if isinstance(prompt, str) else prompt
            
            if frames <= self.max_chunk_frames and len(prompts) == 1:
                return self._generate_single_chunk(mode, prompts[0], output_path, width, height, frames, input_path_1)
            
            print(f"Generating long video ({frames} frames) in chunks...")
            chunks = []
            current_input_image = input_path_1
            remaining_frames = frames
            chunk_idx = 0
            
            while remaining_frames > 0:
                current_chunk_frames = min(remaining_frames, self.max_chunk_frames)
                chunk_output = f"aeon_output/debug/chunk_{chunk_idx}.mp4"
                
                prompt_idx = min(chunk_idx, len(prompts) - 1)
                current_prompt = prompts[prompt_idx]
                
                current_mode = mode if chunk_idx == 0 else "image_to_video"
                
                print(f"Generating chunk {chunk_idx+1}/{((frames + self.max_chunk_frames - 1) // self.max_chunk_frames)} "
                      f"({current_chunk_frames} frames) with prompt: {current_prompt[:50]}...")
                
                self._generate_single_chunk(current_mode, current_prompt, chunk_output, width, height, current_chunk_frames, current_input_image)
                
                chunks.append(chunk_output)
                
                next_input_image = f"aeon_output/debug/last_frame_{chunk_idx}.jpg"
                self._extract_last_frame(chunk_output, next_input_image)
                current_input_image = next_input_image
                
                remaining_frames -= current_chunk_frames
                chunk_idx += 1
            
            print("Concatenating chunks...")
            self._concatenate_videos(chunks, output_path)
            
            for f in chunks:
                if os.path.exists(f): os.remove(f)
            
            return f"Successfully generated long video at {output_path}"

        except Exception as e:
            return f"Error during video generation: {str(e)}"
        finally:
            self._manage_registry("end")

    def _generate_single_chunk(self, mode: str, prompt: str, output_path: str, width: int, height: int, frames: int, image_path: Optional[str]) -> str:
        """Handles the API call to ComfyUI for a single video segment."""
        uploaded_name = None
        if image_path:
            uploaded_name = self._upload_image(image_path)

        workflow = self._get_workflow(mode, prompt, width, height, frames, uploaded_name)
        
        payload = {"prompt": workflow}
        try:
            response = requests.post(f"{self.comfy_url}/prompt", json=payload, timeout=300)
            if response.status_code == 400:
                print(f"\n=== ComfyUI 400 Bad Request Debug ===")
                print(f"Payload: {json.dumps(payload, indent=2)}")
                print(f"Response: {response.text}")
                print(f"==================================\n")
            response.raise_for_status()
            prompt_id = response.json().get("prompt_id")
            
            while True:
                history = requests.get(f"{self.comfy_url}/history/{prompt_id}").json()
                if prompt_id in history:
                    break
                time.sleep(2)
            
            files = sorted([os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f.endswith(".mp4")], key=os.path.getmtime)
            if not files:
                raise FileNotFoundError("No output video found in ComfyUI output directory.")
            
            latest_video = files[-1]
            shutil.copy(latest_video, output_path)
            return f"Video generated and saved to {output_path}"
            
        except Exception as e:
            raise RuntimeError(f"ComfyUI API error: {str(e)}")