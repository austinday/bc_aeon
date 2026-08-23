import os
import time
import glob
import json
import shutil
import subprocess
from typing import List, Optional, Dict, Any, Union

import requests

from aeon.tools.generate_image import ComfyUITool
from aeon.core.prompt_enhancer import enhance_prompt
from aeon.core.prompts import TOOL_DESC_GENERATE_VIDEO
from aeon.core.paths import resolve_output_dir


def _aeon_home() -> str:
    return os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))


# ComfyUI's output dir is bind-mounted from here (see start_comfyui.sh), so files we
# drop here are visible inside the container at /workspace/ComfyUI/output/<name>.
def _comfy_output_host() -> str:
    return os.path.join(_aeon_home(), "temp", "comfyui_output")


CONTAINER_OUTPUT = "/workspace/ComfyUI/output"


class GenerateVideoTool(ComfyUITool):
    """
    Flexible video generation via LTX-2.3 in ComfyUI. One tool, many modes, so the
    agent can fulfil open-ended requests using any assets:

      text_to_video   prompt only.
      image_to_video  one start image (init_image) animated by the prompt.
      keyframes       a storyboard: images pinned at frame positions (interpolate/
                      sequence between them) -> consistent subjects, A->B morphs,
                      "bring these stills to life in order".
      extend_video    continue an existing video (seeds from its last frame).
      edit_video      restyle/edit an existing video (video->video at `denoise`).

    Portable: auto-starts ComfyUI, resolves models dynamically, retrieves output via
    the ComfyUI API, and works from any working directory.
    """

    SINGLE_PASS_MAX = 121          # frames renderable in one LTX pass on a 48 GB GPU
    DEFAULT_FRAMES = 97

    def __init__(self, llm_client=None):
        super().__init__(name="generate_video", description=TOOL_DESC_GENERATE_VIDEO)
        self.comfy_models_dir = os.path.join(_aeon_home(), "models", "comfyui")
        self.llm_client = llm_client

    # ---------- model + asset resolution ----------
    def _resolve_model(self, subdir: str, patterns: List[str], default: str) -> str:
        base = os.path.join(self.comfy_models_dir, subdir)
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(base, pat)))
            if hits:
                return os.path.basename(hits[0])
        return default

    def _upload_image(self, image_path: str) -> str:
        ap = os.path.abspath(image_path)
        if not os.path.exists(ap):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(ap, "rb") as f:
            r = requests.post(f"{self.comfy_url}/upload/image",
                              files={"image": (os.path.basename(ap), f)}, timeout=120)
            r.raise_for_status()
            return r.json().get("name")

    def _stage_video(self, video_path: str) -> str:
        """Copy a host video into the mounted ComfyUI output dir; return its in-container path."""
        ap = os.path.abspath(video_path)
        if not os.path.exists(ap):
            raise FileNotFoundError(f"Video not found: {video_path}")
        os.makedirs(_comfy_output_host(), exist_ok=True)
        name = f"_in_{os.getpid()}_{os.path.basename(ap)}"
        shutil.copy(ap, os.path.join(_comfy_output_host(), name))
        return f"{CONTAINER_OUTPUT}/{name}"

    # ---------- ffmpeg helpers (mount an absolute work dir; reference basenames) ----------
    def _ffmpeg(self, work_dir: str, args: List[str], timeout: int = 600):
        # Bound every ffmpeg call: a malformed/streaming input can make ffmpeg
        # block indefinitely, which would hang the whole agent loop. --rm cleans
        # the container up on both success and TimeoutExpired-triggered kill.
        try:
            subprocess.run(["docker", "run", "--rm", "-v", f"{work_dir}:/work", "-w", "/work",
                            "mwader/static-ffmpeg", *args], check=True, capture_output=True,
                           timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffmpeg timed out after {timeout}s (input may be malformed).")

    def _extract_last_frame(self, work_dir: str, video_name: str, image_name: str):
        self._ffmpeg(work_dir, ["-sseof", "-1", "-i", video_name, "-update", "1", "-q:v", "2", image_name])

    def _concatenate_videos(self, work_dir: str, names: List[str], out_name: str):
        lst = os.path.join(work_dir, "concat_list.txt")
        with open(lst, "w") as f:
            for n in names:
                f.write(f"file '{n}'\n")
        self._ffmpeg(work_dir, ["-f", "concat", "-safe", "0", "-i", "concat_list.txt", "-c", "copy", out_name])
        os.remove(lst)

    # ---------- workflow construction ----------
    @staticmethod
    def _round32(x: int) -> int:
        return max(32, int(round(x / 32.0)) * 32)

    @staticmethod
    def _valid_len(frames: int) -> int:
        return max(9, int(round((frames - 1) / 8.0)) * 8 + 1)

    def _loaders_and_cond(self, prompt: str, neg: str) -> Dict[str, Any]:
        # Prefer the uncensored 10Eros NSFW LTX-2.3 finetune; fall back to stock LTX if present.
        unet = self._resolve_model("unet", ["*10Eros*.gguf", "*[Ee]ros*.gguf", "ltx*dev*.gguf", "ltx*.gguf"],
                                   "10Eros_v1.210Eros_v1.2-Q4_K_M.gguf")
        gemma = self._resolve_model("text_encoders", ["gemma-3*.gguf"], "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf")
        connectors = self._resolve_model("text_encoders", ["*connectors*.safetensors", "*projection*.safetensors"],
                                         "ltx-2.3-22b-dev_embeddings_connectors.safetensors")
        vae = self._resolve_model("vae", ["ltx*video_vae*.safetensors", "ltx*vae*.safetensors"],
                                  "ltx-2.3-22b-dev_video_vae.safetensors")
        return {
            "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}},
            "2": {"class_type": "DualCLIPLoaderGGUF",
                  "inputs": {"clip_name1": gemma, "clip_name2": connectors, "type": "ltxv"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["2", 0]}},
            "6": {"class_type": "LTXVConditioning",
                  "inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": 24.0}},
        }

    def _build_workflow(self, mode: str, prompt: str, neg: str, w: int, h: int, length: int,
                        init_image_name: Optional[str], keyframes: Optional[List[Dict]],
                        staged_video: Optional[str], denoise: float, seed: int) -> Dict[str, Any]:
        wf = self._loaders_and_cond(prompt, neg)
        pos_in, neg_in = ["6", 0], ["6", 1]
        crop_guides = False
        sigmas = ["9", 0]

        wf["8"] = {"class_type": "ModelSamplingLTXV", "inputs": {"model": ["1", 0], "max_shift": 2.05, "base_shift": 0.95}}
        wf["9"] = {"class_type": "LTXVScheduler",
                   "inputs": {"steps": 25, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1}}

        if mode == "edit_video":
            # video -> video: decode the input clip, encode to latent, partially denoise.
            wf["20"] = {"class_type": "VHS_LoadVideoPath",
                        "inputs": {"video": staged_video, "force_rate": 24.0, "custom_width": w, "custom_height": h,
                                   "frame_load_cap": length, "skip_first_frames": 0, "select_every_nth": 1}}
            wf["21"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["20", 0], "vae": ["3", 0]}}
            wf["22"] = {"class_type": "SplitSigmasDenoise", "inputs": {"sigmas": ["9", 0], "denoise": denoise}}
            sigmas = ["22", 1]                 # low_sigmas (partial denoise tail)
            latent_in = ["21", 0]
        elif mode in ("image_to_video", "extend_video"):
            wf["7"] = {"class_type": "LTXVImgToVideo",
                       "inputs": {"positive": pos_in, "negative": neg_in, "vae": ["3", 0],
                                  "image": ["14", 0], "width": w, "height": h, "length": length,
                                  "batch_size": 1, "strength": 1.0}}
            wf["14"] = {"class_type": "LoadImage", "inputs": {"image": init_image_name or "default.png"}}
            pos_in, neg_in, latent_in = ["7", 0], ["7", 1], ["7", 2]
        elif mode == "keyframes":
            wf["7"] = {"class_type": "EmptyLTXVLatentVideo",
                       "inputs": {"width": w, "height": h, "length": length, "batch_size": 1}}
            latent_in = ["7", 0]
            node = 30
            for kf in (keyframes or []):
                wf[str(node)] = {"class_type": "LoadImage", "inputs": {"image": kf["_name"]}}
                wf[str(node + 1)] = {"class_type": "LTXVAddGuide",
                                     "inputs": {"positive": pos_in, "negative": neg_in, "vae": ["3", 0],
                                                "latent": latent_in, "image": [str(node), 0],
                                                "frame_idx": int(kf.get("at_frame", 0)),
                                                "strength": float(kf.get("strength", 1.0))}}
                pos_in, neg_in, latent_in = [str(node + 1), 0], [str(node + 1), 1], [str(node + 1), 2]
                node += 2
            crop_guides = True
        else:  # text_to_video
            wf["7"] = {"class_type": "EmptyLTXVLatentVideo",
                       "inputs": {"width": w, "height": h, "length": length, "batch_size": 1}}
            latent_in = ["7", 0]

        wf["10"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        wf["11"] = {"class_type": "SamplerCustom",
                    "inputs": {"model": ["8", 0], "add_noise": True, "noise_seed": seed, "cfg": 3.0,
                               "positive": pos_in, "negative": neg_in, "sampler": ["10", 0],
                               "sigmas": sigmas, "latent_image": latent_in}}
        decode_latent = ["11", 0]
        if crop_guides:
            wf["12c"] = {"class_type": "LTXVCropGuides",
                         "inputs": {"positive": pos_in, "negative": neg_in, "latent": ["11", 0]}}
            decode_latent = ["12c", 2]
        wf["12"] = {"class_type": "VAEDecode", "inputs": {"samples": decode_latent, "vae": ["3", 0]}}
        wf["13"] = {"class_type": "VHS_VideoCombine",
                    "inputs": {"images": ["12", 0], "frame_rate": 24, "loop_count": 0,
                               "filename_prefix": "AeonVideo", "format": "video/h264-mp4",
                               "pingpong": False, "save_output": True}}
        return wf

    def _render(self, workflow: Dict[str, Any], abs_output_path: str) -> str:
        r = requests.post(f"{self.comfy_url}/prompt", json={"prompt": workflow}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected workflow (HTTP {r.status_code}): {r.text[:600]}")
        pid = r.json()["prompt_id"]
        # Queue-aware wait (shared helper): tolerates time spent queued behind
        # other agents' jobs so concurrent callers don't false-timeout.
        out = self._await_comfy(pid, node="13", hard_timeout=1800)
        items = out.get("gifs") or out.get("videos") or out.get("images")
        if not items:
            raise RuntimeError("ComfyUI finished but produced no video output.")
        self._download_comfy_output(items[0], abs_output_path, timeout=120)
        return abs_output_path

    # ---------- public entrypoint ----------
    def execute(self, mode: str, output_dir: str = "", prompt: Union[str, List[str]] = "",
                width: int = 768, height: int = 512, frames: int = DEFAULT_FRAMES,
                init_image: Optional[str] = None, keyframes: Optional[List[Dict]] = None,
                init_video: Optional[str] = None, denoise: float = 0.6,
                negative_prompt: Optional[str] = None, seed: int = 42,
                input_path_1: Optional[str] = None, **kwargs) -> str:
        # Validate mode up front: an unrecognized mode previously fell through to
        # text_to_video, silently ignoring init_image/init_video/keyframes.
        valid_modes = {"text_to_video", "image_to_video", "extend_video", "edit_video", "keyframes"}
        if not mode or mode not in valid_modes:
            import difflib
            sugg = difflib.get_close_matches(str(mode), sorted(valid_modes), n=1, cutoff=0.3)
            hint = f" Did you mean '{sugg[0]}'?" if sugg else ""
            return (f"Error: invalid mode '{mode}'. Valid modes: {', '.join(sorted(valid_modes))}.{hint}")
        if not output_dir or not str(output_dir).strip():
            return "Error: 'output_dir' is required — the directory to save the generated video in."

        # Tolerate string/odd numeric params from the model.
        def _int(v, default):
            try:
                return int(round(float(v)))
            except (TypeError, ValueError):
                return default
        width, height = _int(width, 768), _int(height, 512)
        frames = _int(frames, self.DEFAULT_FRAMES)
        seed = _int(seed, 42)

        # Back-compat: older callers used input_path_1 for the image/video asset.
        if input_path_1 and not init_image and not init_video:
            (init_video, init_image) = (input_path_1, None) if mode in ("extend_video", "edit_video") else (None, input_path_1)

        # Auto-name the file inside the caller-provided output_dir (relative dirs
        # resolve against the workspace aeon was launched from).
        abs_output = str(resolve_output_dir(output_dir, time.strftime("aeon_video_%Y%m%d_%H%M%S.mp4")))
        os.makedirs(os.path.dirname(abs_output) or ".", exist_ok=True)
        work_dir = os.path.join(_aeon_home(), "temp", "video_work", str(os.getpid()))
        os.makedirs(work_dir, exist_ok=True)
        prompt_text = prompt[0] if isinstance(prompt, list) and prompt else (prompt or "")
        prompt_text = enhance_prompt(self.llm_client, prompt_text, "video", force=kwargs.get("enhance"))
        neg = negative_prompt or "low quality, blurry, distorted, static, flickering, watermark, text"
        w, h, length = self._round32(width), self._round32(height), self._valid_len(min(frames, self.SINGLE_PASS_MAX))
        init_image_name = None
        staged_video = None

        try:
            self._ensure_comfyui_running(required_vram=36.0)

            if mode == "extend_video":
                # Seed from the last frame of the input video, then image_to_video.
                src = init_video or init_image
                if not src:
                    return "Error: extend_video requires init_video (the clip to continue)."
                local = os.path.join(work_dir, os.path.basename(os.path.abspath(src)))
                shutil.copy(os.path.abspath(src), local)
                self._extract_last_frame(work_dir, os.path.basename(local), "seed.jpg")
                init_image_name = self._upload_image(os.path.join(work_dir, "seed.jpg"))
                mode = "image_to_video"
            elif mode == "edit_video":
                if not init_video:
                    return "Error: edit_video requires init_video (the clip to restyle)."
                staged_video = self._stage_video(init_video)
            elif mode == "image_to_video":
                if not init_image:
                    return "Error: image_to_video requires init_image."
                init_image_name = self._upload_image(init_image)
            elif mode == "keyframes":
                if not keyframes:
                    return "Error: keyframes mode requires a 'keyframes' list of {image, at_frame, strength}."
                for kf in keyframes:
                    kf["_name"] = self._upload_image(kf["image"])

            wf = self._build_workflow(mode, prompt_text, neg, w, h, length,
                                      init_image_name, keyframes, staged_video, denoise, seed)
            self._render(wf, abs_output)
            return f"Successfully generated video ({mode}) at {abs_output}"

        except Exception as e:
            return f"Error during video generation ({mode}): {e}"
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            if staged_video:
                try: os.remove(os.path.join(_comfy_output_host(), os.path.basename(staged_video)))
                except Exception: pass
            # Keep the shared ComfyUI warm across a burst of comfy ops, reap it once
            # idle (frees VRAM for other tools). Same debounced policy as image gen;
            # the per-call teardown here cold-started the model on every video.
            try:
                self._finish_comfy_session()
            except Exception:
                pass
