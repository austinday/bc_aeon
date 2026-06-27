import os
import time
import glob
import shutil
import subprocess
from typing import List, Optional, Dict, Any, Union

import requests

from aeon.tools.generate_image import ComfyUITool
from aeon.core.gpu_queue import release_vram


def _aeon_home() -> str:
    return os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))


class GenerateVideoTool(ComfyUITool):
    """
    Generate videos with LTX-Video via the local ComfyUI service.

    Portable by construction: it inherits ComfyUITool (so it auto-starts ComfyUI
    with VRAM/registry management regardless of where aeon is run), retrieves the
    rendered video through the ComfyUI HTTP API (/history -> /view) rather than
    reading any host directory, and resolves every path to an absolute path. All
    intermediate work happens in an absolute scratch dir under $AEON_HOME so the
    tool behaves identically from any working directory.
    """

    def __init__(self):
        super().__init__(
            name="generate_video",
            description="Generates videos using LTX-Video via ComfyUI. Supports text-to-video and image-to-video.",
        )
        self.max_chunk_frames = 33  # LTX-Video optimal chunk size
        self.comfy_models_dir = os.path.join(_aeon_home(), "models", "comfyui")

    # ---- model filename resolution (portable across downloaded quants) ----
    def _resolve_model(self, subdir: str, patterns: List[str], default: str) -> str:
        """Return the basename of the first model in comfyui/<subdir> matching any
        pattern; fall back to `default`. ComfyUI loads weights by basename, so we
        only need the filename, not the path."""
        base = os.path.join(self.comfy_models_dir, subdir)
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(base, pat)))
            if hits:
                return os.path.basename(hits[0])
        return default

    # ---- ffmpeg helpers (mount an absolute work dir; reference basenames) ----
    def _ffmpeg(self, work_dir: str, args: List[str]):
        cmd = ["docker", "run", "--rm", "-v", f"{work_dir}:/work", "-w", "/work",
               "mwader/static-ffmpeg", *args]
        subprocess.run(cmd, check=True, capture_output=True)

    def _extract_last_frame(self, work_dir: str, video_name: str, image_name: str):
        self._ffmpeg(work_dir, ["-sseof", "-1", "-i", video_name, "-update", "1", "-q:v", "2", image_name])

    def _concatenate_videos(self, work_dir: str, video_names: List[str], output_name: str):
        list_path = os.path.join(work_dir, "concat_list.txt")
        with open(list_path, "w") as f:
            for name in video_names:
                f.write(f"file '{name}'\n")
        self._ffmpeg(work_dir, ["-f", "concat", "-safe", "0", "-i", "concat_list.txt", "-c", "copy", output_name])
        os.remove(list_path)

    # ---- ComfyUI workflow ----
    def _upload_image(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f)}
            resp = requests.post(f"{self.comfy_url}/upload/image", files=files, timeout=60)
            resp.raise_for_status()
            return resp.json().get("name")

    @staticmethod
    def _round32(x: int) -> int:
        return max(32, int(round(x / 32.0)) * 32)

    @staticmethod
    def _valid_len(frames: int) -> int:
        # LTX-Video latents are temporally /8 + 1, so length must be 8n+1.
        return max(9, int(round((frames - 1) / 8.0)) * 8 + 1)

    def _get_workflow(self, mode: str, prompt: str, width: int, height: int, frames: int,
                      uploaded_image_name: Optional[str] = None) -> Dict[str, Any]:
        """Validated LTX-2.3 graph: GGUF unet + Gemma/connectors text encoder
        (DualCLIPLoaderGGUF, type 'ltxv') + LTXV custom-sampler path."""
        unet = self._resolve_model("unet", ["ltx*dev*.gguf", "ltx*.gguf"], "ltx-2.3-22b-dev-Q4_K_M.gguf")
        gemma = self._resolve_model("text_encoders", ["gemma-3*.gguf"], "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf")
        connectors = self._resolve_model(
            "text_encoders", ["*connectors*.safetensors", "*projection*.safetensors"],
            "ltx-2.3-22b-dev_embeddings_connectors.safetensors")
        vae = self._resolve_model("vae", ["ltx*video_vae*.safetensors", "ltx*vae*.safetensors"],
                                  "ltx-2.3-22b-dev_video_vae.safetensors")
        w, h, length = self._round32(width), self._round32(height), self._valid_len(frames)
        neg = "low quality, blurry, distorted, static, slideshow, flickering, watermark, text"

        wf: Dict[str, Any] = {
            "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}},
            "2": {"class_type": "DualCLIPLoaderGGUF",
                  "inputs": {"clip_name1": gemma, "clip_name2": connectors, "type": "ltxv"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["2", 0]}},
            "6": {"class_type": "LTXVConditioning",
                  "inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": 24.0}},
            "8": {"class_type": "ModelSamplingLTXV", "inputs": {"model": ["1", 0], "max_shift": 2.05, "base_shift": 0.95}},
            "9": {"class_type": "LTXVScheduler",
                  "inputs": {"steps": 25, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1}},
            "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
            "13": {"class_type": "VHS_VideoCombine",
                   "inputs": {"images": ["12", 0], "frame_rate": 24, "loop_count": 0, "filename_prefix": "AeonVideo",
                              "format": "video/h264-mp4", "pingpong": False, "save_output": True}},
        }
        if mode == "text_to_video":
            wf["7"] = {"class_type": "EmptyLTXVLatentVideo",
                       "inputs": {"width": w, "height": h, "length": length, "batch_size": 1}}
            sampler_pos, sampler_neg, sampler_lat = ["6", 0], ["6", 1], ["7", 0]
        else:  # image_to_video (video_extension is converted to this upstream)
            wf["14"] = {"class_type": "LoadImage", "inputs": {"image": uploaded_image_name or "default.png"}}
            wf["7"] = {"class_type": "LTXVImgToVideo",
                       "inputs": {"positive": ["6", 0], "negative": ["6", 1], "vae": ["3", 0], "image": ["14", 0],
                                  "width": w, "height": h, "length": length, "batch_size": 1, "strength": 1.0}}
            sampler_pos, sampler_neg, sampler_lat = ["7", 0], ["7", 1], ["7", 2]
        wf["11"] = {"class_type": "SamplerCustom",
                    "inputs": {"model": ["8", 0], "add_noise": True, "noise_seed": 42, "cfg": 3.0,
                               "positive": sampler_pos, "negative": sampler_neg,
                               "sampler": ["10", 0], "sigmas": ["9", 0], "latent_image": sampler_lat}}
        return wf

    def _render_chunk(self, mode: str, prompt: str, abs_output_path: str, width: int, height: int,
                      frames: int, image_path: Optional[str]) -> str:
        """Submit one chunk to ComfyUI and save the result to abs_output_path via the API."""
        uploaded = self._upload_image(image_path) if image_path else None
        workflow = self._get_workflow(mode, prompt, width, height, frames, uploaded)

        resp = requests.post(f"{self.comfy_url}/prompt", json={"prompt": workflow}, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected workflow (HTTP {resp.status_code}): {resp.text}")
        prompt_id = resp.json()["prompt_id"]

        for _ in range(300):  # up to ~15 min per chunk
            history = requests.get(f"{self.comfy_url}/history/{prompt_id}", timeout=10).json()
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {}).get("13", {})
                items = outputs.get("gifs") or outputs.get("videos") or outputs.get("images")
                if not items:
                    raise RuntimeError("ComfyUI finished but produced no video output (check VHS node / models).")
                info = items[0]
                view = requests.get(f"{self.comfy_url}/view", params={
                    "filename": info["filename"], "subfolder": info.get("subfolder", ""),
                    "type": info.get("type", "output")}, timeout=60)
                view.raise_for_status()
                with open(abs_output_path, "wb") as f:
                    f.write(view.content)
                return abs_output_path
            time.sleep(3)
        raise RuntimeError("Video chunk generation timed out.")

    def execute(self, mode: str, prompt: Union[str, List[str]], output_path: str, width: int = 768,
                height: int = 512, frames: int = 33, input_path_1: Optional[str] = None, **kwargs) -> str:
        if not output_path:
            return "Error: 'output_path' parameter is required."

        abs_output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output_path) or ".", exist_ok=True)
        work_dir = os.path.join(_aeon_home(), "temp", "video_work", str(os.getpid()))
        os.makedirs(work_dir, exist_ok=True)
        abs_input = os.path.abspath(input_path_1) if input_path_1 else None

        try:
            self._ensure_comfyui_running(required_vram=24.0)
            prompts = [prompt] if isinstance(prompt, str) else prompt

            # video_extension: seed from the last frame of the input *video*, then
            # continue as image_to_video (LoadImage cannot ingest an mp4 directly).
            if mode == "video_extension" and abs_input:
                local_vid = os.path.join(work_dir, os.path.basename(abs_input))
                shutil.copy(abs_input, local_vid)
                seed_frame = os.path.join(work_dir, "seed_frame.jpg")
                self._extract_last_frame(work_dir, os.path.basename(local_vid), "seed_frame.jpg")
                abs_input = seed_frame
                mode = "image_to_video"

            # Single short chunk: render straight to the output.
            if frames <= self.max_chunk_frames and len(prompts) == 1:
                self._render_chunk(mode, prompts[0], abs_output_path, width, height, frames, abs_input)
                return f"Successfully generated video at {abs_output_path}"

            # Long video: render chunks in the absolute work dir, chain via last frame, concat.
            chunks: List[str] = []
            current_input = abs_input
            remaining = frames
            idx = 0
            total = (frames + self.max_chunk_frames - 1) // self.max_chunk_frames
            while remaining > 0:
                n = min(remaining, self.max_chunk_frames)
                chunk_path = os.path.join(work_dir, f"chunk_{idx}.mp4")
                cur_prompt = prompts[min(idx, len(prompts) - 1)]
                cur_mode = mode if idx == 0 else "image_to_video"
                print(f"Generating chunk {idx + 1}/{total} ({n} frames): {cur_prompt[:50]}...")
                self._render_chunk(cur_mode, cur_prompt, chunk_path, width, height, n, current_input)
                chunks.append(chunk_path)

                last_frame = os.path.join(work_dir, f"last_frame_{idx}.jpg")
                self._extract_last_frame(work_dir, os.path.basename(chunk_path), os.path.basename(last_frame))
                current_input = last_frame
                remaining -= n
                idx += 1

            final_name = "final.mp4"
            self._concatenate_videos(work_dir, [os.path.basename(c) for c in chunks], final_name)
            shutil.copy(os.path.join(work_dir, final_name), abs_output_path)
            return f"Successfully generated long video at {abs_output_path}"

        except Exception as e:
            return f"Error during video generation: {e}"
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            try:
                remaining_users, _ = self._manage_registry('unregister')
                release_vram()
                if remaining_users == 0:
                    subprocess.run(["docker", "rm", "-f", "aeon_comfyui"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
