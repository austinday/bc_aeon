import os
import time
import glob
import math
import shutil
import subprocess
import uuid
from typing import List, Optional, Dict, Any, Union

import requests

from aeon.core.agent_protocol import SideEffect, ToolResult, ToolStatus
from aeon.tools.generate_image import ComfyUITool, _LOCAL_HTTP_KWARGS
from aeon.core.prompt_enhancer import enhance_prompt, wants_adult
from aeon.core.prompts import TOOL_DESC_GENERATE_VIDEO
from aeon.core.paths import resolve_output_dir
from aeon.tools.command_fleet_guard import (
    require_fleet_low_priority_wrapper,
    scrubbed_fleet_command_environment,
)


def _aeon_home() -> str:
    return os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))


# ComfyUI's output dir is bind-mounted from here (see start_comfyui.sh), so files we
# drop here are visible inside the container at /workspace/ComfyUI/output/<name>.
def _comfy_output_host() -> str:
    return os.path.join(_aeon_home(), "temp", "comfyui_output")


CONTAINER_OUTPUT = "/workspace/ComfyUI/output"


class GenerateVideoTool(ComfyUITool):
    COMFY_SERVICE_ID = "aeon-video-comfyui"
    # A first use can safely stage and verify the complete 74 GB H3/LTX bundle
    # over 1 GbE before Comfy starts. Keep the foreground demand renewable for
    # that reviewed cold path; warm starts still return in seconds.
    COMFY_WAIT_TIMEOUT_SECONDS = 7200

    """
    Flexible audiovisual generation via MiniMax H3 and LTX-2.3 in ComfyUI. One
    tool, many modes, so the agent can fulfil open-ended requests using any assets:

      text_to_video   prompt only.
      image_to_video  one start image (init_image) animated by the prompt.
      keyframes       a storyboard: images pinned at frame positions (interpolate/
                      sequence between them) -> consistent subjects, A->B morphs,
                      "bring these stills to life in order".
      extend_video    continue an existing video (seeds from its last frame).
      edit_video      restyle/edit an existing video (video->video at `denoise`).
      concatenate     assemble ordered generated clips into one final MP4.

    Portable: requests ComfyUI through Fleet Compute, resolves models dynamically,
    retrieves output via the loopback API, and works from any working directory.
    """

    LTX_SINGLE_PASS_MAX = 121
    H3_SINGLE_PASS_MAX = 362
    DEFAULT_FRAMES = 124
    LTX_MAX_PIXELS = 768 * 512
    H3_MAX_PIXELS = 1344 * 768
    H3_MODEL_RELEASE = "MiniMax H3 / 10Eros-Max beta2 NVFP4"
    LTX_MODEL_RELEASE = "LTX-2.3 / 10Eros 1.5 Q8"
    H3_MODEL_CANDIDATES = (
        "10Eros_Max_h3_fl2va_beta2_pruned_nvfp4.safetensors",
    )
    H3_TEXT_ENCODER_CANDIDATES = (
        "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
    )
    H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
    H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
    LTX_MODEL_CANDIDATES = (
        "10Eros_v1.5-Q8_0.gguf",
        "10Eros_v1.5-Q6_K.gguf",
        "10Eros_v1.5-Q5_K_M.gguf",
        "10Eros_v1.5-Q4_K_M.gguf",
    )
    # Compatibility names for older extensions/tests that directly exercise the
    # original LTX graph. The public execution path now routes automatically.
    SINGLE_PASS_MAX = LTX_SINGLE_PASS_MAX
    MAX_PIXELS = LTX_MAX_PIXELS
    VIDEO_MODEL_RELEASE = LTX_MODEL_RELEASE

    def __init__(self, llm_client=None):
        super().__init__(
            name="generate_video",
            description=TOOL_DESC_GENERATE_VIDEO,
            underlying_model=(
                "MiniMax H3 / 10Eros-Max NVFP4 + uncensored Qwen3-VL encoder "
                "(LTX-2.3 / 10Eros 1.5 Q8 specialist fallback)"
            ),
        )
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

    @staticmethod
    def _require_exact_file(base: str, candidates: tuple[str, ...], label: str) -> str:
        for filename in candidates:
            if os.path.isfile(os.path.join(base, filename)):
                return filename
        raise RuntimeError(f"The reviewed {label} is missing")

    def _resolve_h3_stack(self) -> tuple[str, str, str, str]:
        """Resolve the complete uncensored H3 stack; never mix silent fallbacks."""

        unet_dir = os.path.join(self.comfy_models_dir, "unet")
        encoder_dir = os.path.join(self.comfy_models_dir, "text_encoders")
        vae_dir = os.path.join(self.comfy_models_dir, "vae")
        model_override = os.environ.get("AEON_H3_VIDEO_MODEL", "").strip()
        if model_override:
            if os.path.basename(model_override) != model_override:
                raise RuntimeError("AEON_H3_VIDEO_MODEL must name one file in the ComfyUI unet directory")
            if not os.path.isfile(os.path.join(unet_dir, model_override)):
                raise RuntimeError(f"Configured H3 video model is not installed: {model_override}")
            model = model_override
        else:
            model = self._require_exact_file(
                unet_dir, self.H3_MODEL_CANDIDATES, "MiniMax H3 10Eros-Max NVFP4 model"
            )
        encoder = self._require_exact_file(
            encoder_dir,
            self.H3_TEXT_ENCODER_CANDIDATES,
            "uncensored MiniMax H3 Qwen3-VL NVFP4 text encoder",
        )
        video_vae = self._require_exact_file(vae_dir, (self.H3_VIDEO_VAE,), "MiniMax H3 video VAE")
        audio_vae = self._require_exact_file(vae_dir, (self.H3_AUDIO_VAE,), "MiniMax H3 audio VAE")
        return model, encoder, video_vae, audio_vae

    def _resolve_ltx_model(self) -> str:
        """Resolve the reviewed LTX specialist exactly; never select an old merge."""

        base = os.path.join(self.comfy_models_dir, "unet")
        override = os.environ.get("AEON_LTX_VIDEO_MODEL", os.environ.get("AEON_VIDEO_MODEL", "")).strip()
        if override:
            if os.path.basename(override) != override:
                raise RuntimeError("AEON_VIDEO_MODEL must name one file in the ComfyUI unet directory")
            if not os.path.isfile(os.path.join(base, override)):
                raise RuntimeError(f"Configured video model is not installed: {override}")
            return override
        for filename in self.LTX_MODEL_CANDIDATES:
            if os.path.isfile(os.path.join(base, filename)):
                return filename
        raise RuntimeError(
            "The reviewed uncensored video model is missing: install "
            "vantagewithai/LTX2.3-10Eros-1.5-GGUF (Q8_0 preferred)"
        )

    # Backwards-compatible private name retained for older local tests/extensions.
    def _resolve_video_model(self) -> str:
        return self._resolve_ltx_model()

    def _upload_image(self, image_path: str) -> str:
        ap = os.path.abspath(image_path)
        if not os.path.exists(ap):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(ap, "rb") as f:
            r = requests.post(f"{self.comfy_url}/upload/image",
                              files={"image": (os.path.basename(ap), f)}, timeout=120,
                              **_LOCAL_HTTP_KWARGS)
            r.raise_for_status()
            return r.json().get("name")

    def _stage_video(self, video_path: str) -> str:
        """Copy a host video into the mounted ComfyUI output dir; return its in-container path."""
        ap = os.path.abspath(video_path)
        self._validate_video(ap)
        os.makedirs(_comfy_output_host(), exist_ok=True)
        name = f"_in_{os.getpid()}_{uuid.uuid4().hex[:12]}{os.path.splitext(ap)[1].lower()}"
        shutil.copy(ap, os.path.join(_comfy_output_host(), name))
        return f"{CONTAINER_OUTPUT}/{name}"

    # ---------- bounded local ffmpeg helpers ----------
    @staticmethod
    def _ffmpeg_executable() -> str:
        try:
            import imageio_ffmpeg

            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError("The bundled ffmpeg runtime is unavailable") from exc
        if not os.path.isabs(executable) or not os.path.isfile(executable):
            raise RuntimeError("The bundled ffmpeg runtime is invalid")
        return executable

    def _ffmpeg(self, work_dir: str, args: List[str], timeout: int = 600):
        try:
            completed = subprocess.run(
                [
                    require_fleet_low_priority_wrapper(),
                    self._ffmpeg_executable(),
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-threads",
                    "2",
                    *args,
                ],
                cwd=work_dir,
                env=scrubbed_fleet_command_environment(),
                check=False,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffmpeg timed out after {timeout}s (input may be malformed).")
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()[-800:]
            raise RuntimeError(f"ffmpeg failed: {detail or 'unknown media error'}")

    def _extract_last_frame(self, work_dir: str, video_name: str, image_name: str):
        self._ffmpeg(
            work_dir,
            ["-sseof", "-0.08", "-i", video_name, "-frames:v", "1", "-y", image_name],
        )

    def _concatenate_videos(self, work_dir: str, names: List[str], out_name: str):
        lst = os.path.join(work_dir, "concat_list.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for n in names:
                f.write(f"file '{n}'\n")
        try:
            self._ffmpeg(
                work_dir,
                [
                    "-f", "concat", "-safe", "1", "-i", "concat_list.txt",
                    "-c", "copy", "-movflags", "+faststart", "-y", out_name,
                ],
            )
        finally:
            try:
                os.remove(lst)
            except FileNotFoundError:
                pass

    def _validate_video(self, path: str, *, require_audio: bool = False) -> int:
        """Prove that an owned, non-link artifact contains a decodable video stream."""

        ap = os.path.abspath(path)
        try:
            metadata = os.lstat(ap)
        except OSError as exc:
            raise RuntimeError(f"Video not found: {path}") from exc
        if (
            not os.path.isfile(ap)
            or os.path.islink(ap)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size < 32
        ):
            raise RuntimeError(f"Video is not a safe regular file: {path}")
        with open(ap, "rb") as stream:
            header = stream.read(12)
        if len(header) < 12 or header[4:8] != b"ftyp":
            raise RuntimeError(f"Video is not a valid MP4 container: {path}")
        self._ffmpeg(
            os.path.dirname(ap) or ".",
            ["-i", os.path.basename(ap), "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"],
            timeout=120,
        )
        if require_audio:
            self._ffmpeg(
                os.path.dirname(ap) or ".",
                ["-i", os.path.basename(ap), "-map", "0:a:0", "-t", "0.1", "-f", "null", "-"],
                timeout=120,
            )
        return metadata.st_size

    # ---------- workflow construction ----------
    @staticmethod
    def _round32(x: int) -> int:
        return max(32, int(round(x / 32.0)) * 32)

    def _dimensions(self, width: int, height: int, *, renderer: str = "ltx") -> tuple[int, int]:
        width = max(256, self._round32(width))
        height = max(256, self._round32(height))
        max_side = 1344 if renderer == "h3" else 1024
        max_pixels = self.H3_MAX_PIXELS if renderer == "h3" else self.LTX_MAX_PIXELS
        if width > max_side or height > max_side:
            scale = min(max_side / float(width), max_side / float(height))
            width = max(256, int(width * scale) // 32 * 32)
            height = max(256, int(height * scale) // 32 * 32)
        if width * height <= max_pixels:
            return width, height
        scale = (max_pixels / float(width * height)) ** 0.5
        width = max(256, int(width * scale) // 32 * 32)
        height = max(256, int(height * scale) // 32 * 32)
        return width, height

    @staticmethod
    def _valid_ltx_len(frames: int) -> int:
        return max(9, int(round((frames - 1) / 8.0)) * 8 + 1)

    @staticmethod
    def _valid_h3_len(frames: int) -> int:
        """Snap upward to H3's 17n+5 grid (2.3-15.1 seconds at 24 fps)."""

        return max(56, min(362, int(math.ceil((frames - 5) / 17.0)) * 17 + 5))

    # Backwards-compatible helper for existing callers that mean the LTX grid.
    @staticmethod
    def _valid_len(frames: int) -> int:
        return GenerateVideoTool._valid_ltx_len(frames)

    def _ltx_loaders_and_cond(self, prompt: str, neg: str) -> Dict[str, Any]:
        unet = self._resolve_ltx_model()
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

    def _build_ltx_workflow(self, mode: str, prompt: str, neg: str, w: int, h: int, length: int,
                            init_image_name: Optional[str], keyframes: Optional[List[Dict]],
                            staged_video: Optional[str], denoise: float, seed: int) -> Dict[str, Any]:
        wf = self._ltx_loaders_and_cond(prompt, neg)
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

    @staticmethod
    def _h3_boundary_keyframes(keyframes: Optional[List[Dict]], requested_frames: int) -> bool:
        """H3's FL2VA path accepts only a first frame, a last frame, or both."""

        if not keyframes or len(keyframes) > 2:
            return False
        positions: list[int] = []
        for keyframe in keyframes:
            if not isinstance(keyframe, dict):
                return False
            try:
                positions.append(int(round(float(keyframe.get("at_frame", 0)))))
            except (TypeError, ValueError):
                return False
        final = max(0, requested_frames - 1)
        return len(set(positions)) == len(positions) and all(
            position <= 0 or position >= final for position in positions
        )

    def _select_renderer(
        self,
        requested_mode: str,
        raw_prompt: str,
        keyframes: Optional[List[Dict]],
        requested_frames: int,
        renderer: str,
    ) -> str:
        renderer = str(renderer or "auto").strip().lower()
        if renderer not in {"auto", "h3", "ltx"}:
            raise ValueError("renderer must be auto, h3, or ltx")
        h3_supported = requested_mode in {"text_to_video", "image_to_video"} or (
            requested_mode == "keyframes"
            and self._h3_boundary_keyframes(keyframes, requested_frames)
        )
        if renderer == "h3":
            if not h3_supported:
                raise ValueError(
                    "H3 supports text/image generation and boundary-only keyframes; "
                    "use renderer=ltx for edit, continuation, or interior storyboard anchors"
                )
            return "h3"
        if renderer == "ltx":
            return "ltx"
        if not h3_supported or wants_adult(raw_prompt):
            # LTX 10Eros 1.5 currently has the more mature explicit-motion and
            # arbitrary-guide behavior. H3 remains available explicitly.
            return "ltx"
        return "h3"

    @staticmethod
    def _structured_h3_prompt(
        prompt: str,
        *,
        has_first: bool,
        has_last: bool,
        duration: float,
        negative_prompt: Optional[str],
    ) -> str:
        """Preserve official H3 IR or safely wrap a plain creative brief in it."""

        body = str(prompt or "").strip()
        required = (
            "integrated_multimodal_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
        )
        if not all(field in body for field in required):
            body = (
                f"integrated_multimodal_description: [Shot 1] {body}\n\n"
                "overall_soundscape: Natural ambience and physical sounds synchronized "
                "to the visible setting and action.\n\n"
                "non_diegetic_music: N/A"
            )

        exclusions = str(negative_prompt or "").strip()
        if exclusions:
            insertion = f" Visual exclusions: {exclusions}."
            marker = "\n\noverall_soundscape:"
            if marker in body:
                body = body.replace(marker, f"{insertion}{marker}", 1)
            else:
                body = f"{body}{insertion}"

        first_line = body.splitlines()[0] if body else ""
        if first_line.startswith(("For the target video,", "How the reference pictures align")):
            return body
        if has_first and has_last:
            instruction = (
                "How the reference pictures align with the target video — Picture 1 "
                "(from Shot 1) aligns with the 0.00-second mark of the target video; "
                f"Picture 2 (from Shot 1) aligns with the {duration:.2f}-second mark "
                "of the target video."
            )
        elif has_first:
            instruction = (
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced."
            )
        elif has_last:
            instruction = (
                "How the reference pictures align with the target video — <Picture 1> "
                f"(from [Shot 1]) aligns with the {duration:.2f}-second mark of the target video."
            )
        else:
            return body
        return f"{instruction}\n\n{body}"

    def _build_h3_workflow(
        self,
        prompt: str,
        w: int,
        h: int,
        length: int,
        first_image_name: Optional[str],
        last_image_name: Optional[str],
        seed: int,
    ) -> Dict[str, Any]:
        model, encoder, video_vae, audio_vae = self._resolve_h3_stack()
        conditioning_inputs: Dict[str, Any] = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": prompt,
            "width": w,
            "height": h,
            "length": length,
        }
        workflow: Dict[str, Any] = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": model, "weight_dtype": "default"},
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": encoder, "type": "minimax", "device": "default"},
            },
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae}},
            "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": conditioning_inputs},
            "6": {
                "class_type": "MiniMaxH3SigmaShift",
                "inputs": {"model": ["1", 0], "shift_video": 12.0, "shift_audio": 3.0},
            },
            "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
            "9": {
                "class_type": "BasicScheduler",
                "inputs": {"model": ["6", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0},
            },
            "10": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["6", 0], "conditioning": ["5", 0]},
            },
            "11": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["7", 0],
                    "guider": ["10", 0],
                    "sampler": ["8", 0],
                    "sigmas": ["9", 0],
                    "latent_image": ["5", 1],
                },
            },
            "12": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
            },
            "13": {
                "class_type": "VAEDecodeAudio",
                "inputs": {"samples": ["11", 0], "vae": ["4", 0]},
            },
            "14": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["12", 0],
                    "audio": ["13", 0],
                    "frame_rate": 24,
                    "loop_count": 0,
                    "filename_prefix": "AeonVideoH3",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                    "save_metadata": True,
                    "trim_to_audio": False,
                    "pingpong": False,
                    "save_output": True,
                },
            },
        }
        if first_image_name:
            workflow["15"] = {"class_type": "LoadImage", "inputs": {"image": first_image_name}}
            conditioning_inputs["first_frame"] = ["15", 0]
        if last_image_name:
            workflow["16"] = {"class_type": "LoadImage", "inputs": {"image": last_image_name}}
            conditioning_inputs["last_frame"] = ["16", 0]
        return workflow

    def _build_workflow(self, *args, **kwargs) -> Dict[str, Any]:
        """Backward-compatible private entry point for the LTX graph."""

        return self._build_ltx_workflow(*args, **kwargs)

    def _render(
        self,
        workflow: Dict[str, Any],
        abs_output_path: str,
        *,
        output_node: str = "13",
        require_audio: bool = False,
        timeout: int = 1800,
    ) -> str:
        r = requests.post(
            f"{self.comfy_url}/prompt",
            json={"prompt": workflow},
            timeout=30,
            **_LOCAL_HTTP_KWARGS,
        )
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected workflow (HTTP {r.status_code}): {r.text[:600]}")
        pid = r.json()["prompt_id"]
        # Queue-aware wait (shared helper): tolerates time spent queued behind
        # other agents' jobs so concurrent callers don't false-timeout.
        out = self._await_comfy(pid, node=output_node, hard_timeout=timeout)
        items = out.get("gifs") or out.get("videos") or out.get("images")
        if not items:
            raise RuntimeError("ComfyUI finished but produced no video output.")
        self._download_comfy_output(items[0], abs_output_path, timeout=120)
        self._validate_video(abs_output_path, require_audio=require_audio)
        return abs_output_path

    @staticmethod
    def _failure(mode: str, message: str, *, code: str = "video_generation_failed") -> ToolResult:
        return ToolResult(
            tool_name="generate_video",
            status=ToolStatus.FAILED,
            changed=False,
            summary=f"Video {mode} failed: {message}",
            error_code=code,
            retryable=False,
            side_effect=SideEffect.LOCAL_MUTATION,
        )

    def _assemble(self, input_videos: List[str], work_dir: str, abs_output: str) -> int:
        if len(input_videos) < 2 or len(input_videos) > 20:
            raise ValueError("concatenate requires 2-20 ordered input_videos")
        staged: list[str] = []
        for index, value in enumerate(input_videos):
            source = os.path.abspath(str(value))
            self._validate_video(source)
            name = f"clip_{index:03d}.mp4"
            shutil.copyfile(source, os.path.join(work_dir, name))
            staged.append(name)
        assembled = "assembled.mp4"
        self._concatenate_videos(work_dir, staged, assembled)
        self._validate_video(os.path.join(work_dir, assembled))
        partial = f"{abs_output}.{uuid.uuid4().hex}.part"
        try:
            shutil.copyfile(os.path.join(work_dir, assembled), partial)
            os.replace(partial, abs_output)
        finally:
            try:
                os.remove(partial)
            except FileNotFoundError:
                pass
        return self._validate_video(abs_output)

    # ---------- public entrypoint ----------
    def execute(self, mode: str, output_dir: str = "", prompt: Union[str, List[str]] = "",
                width: int = 864, height: int = 480, frames: int = DEFAULT_FRAMES,
                init_image: Optional[str] = None, keyframes: Optional[List[Dict]] = None,
                init_video: Optional[str] = None, denoise: float = 0.6,
                negative_prompt: Optional[str] = None, seed: int = 42,
                input_videos: Optional[List[str]] = None,
                renderer: str = "auto", enhance: Optional[bool] = None,
                input_path_1: Optional[str] = None) -> ToolResult:
        # Validate mode up front: an unrecognized mode previously fell through to
        # text_to_video, silently ignoring init_image/init_video/keyframes.
        valid_modes = {
            "text_to_video",
            "image_to_video",
            "extend_video",
            "edit_video",
            "keyframes",
            "concatenate",
        }
        if not mode or mode not in valid_modes:
            import difflib
            sugg = difflib.get_close_matches(str(mode), sorted(valid_modes), n=1, cutoff=0.3)
            hint = f" Did you mean '{sugg[0]}'?" if sugg else ""
            return self._failure(
                str(mode),
                f"invalid mode; valid modes are {', '.join(sorted(valid_modes))}.{hint}",
                code="invalid_mode",
            )
        if not output_dir or not str(output_dir).strip():
            return self._failure(mode, "output_dir is required", code="invalid_parameters")

        # Tolerate string/odd numeric params from the model.
        def _int(v, default):
            try:
                return int(round(float(v)))
            except (TypeError, ValueError):
                return default
        width, height = _int(width, 864), _int(height, 480)
        requested_frames = max(5, min(self.H3_SINGLE_PASS_MAX, _int(frames, self.DEFAULT_FRAMES)))
        seed = _int(seed, 42)
        try:
            denoise = max(0.05, min(1.0, float(denoise)))
        except (TypeError, ValueError):
            denoise = 0.6

        # Back-compat: older callers used input_path_1 for the image/video asset.
        if input_path_1 and not init_image and not init_video:
            (init_video, init_image) = (input_path_1, None) if mode in ("extend_video", "edit_video") else (None, input_path_1)

        # Auto-name the file inside the caller-provided output_dir (relative dirs
        # resolve against the workspace aeon was launched from).
        basename = f"aeon_video_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}.mp4"
        abs_output = str(resolve_output_dir(output_dir, basename))
        os.makedirs(os.path.dirname(abs_output) or ".", exist_ok=True)
        work_dir = os.path.join(
            _aeon_home(), "temp", "video_work", f"{os.getpid()}-{uuid.uuid4().hex}"
        )
        os.makedirs(work_dir, exist_ok=True)
        requested_mode = mode
        init_image_name = None
        staged_video = None
        selected_renderer = "none"

        try:
            if mode == "concatenate":
                size = self._assemble(list(input_videos or []), work_dir, abs_output)
                return ToolResult(
                    tool_name="generate_video",
                    status=ToolStatus.OK,
                    changed=True,
                    summary=(
                        f"Assembled {len(input_videos or [])} verified clips into {abs_output}"
                    ),
                    evidence=[f"decodable MP4; {size} bytes"],
                    artifacts=[abs_output],
                    side_effect=SideEffect.LOCAL_MUTATION,
                )

            prompt_text = prompt[0] if isinstance(prompt, list) and prompt else (prompt or "")
            prompt_text = str(prompt_text).strip()
            if not prompt_text:
                return self._failure(mode, "prompt is required", code="invalid_parameters")

            selected_renderer = self._select_renderer(
                requested_mode,
                prompt_text,
                keyframes,
                requested_frames,
                renderer,
            )
            if requested_mode == "image_to_video" and not init_image:
                return self._failure(mode, "init_image is required", code="invalid_parameters")
            if requested_mode in {"extend_video", "edit_video"} and not (init_video or (
                requested_mode == "extend_video" and init_image
            )):
                return self._failure(mode, "init_video is required", code="invalid_parameters")
            if requested_mode == "keyframes" and not keyframes:
                return self._failure(
                    mode,
                    "keyframes must contain at least one image anchor",
                    code="invalid_parameters",
                )

            length = (
                self._valid_h3_len(requested_frames)
                if selected_renderer == "h3"
                else self._valid_ltx_len(min(requested_frames, self.LTX_SINGLE_PASS_MAX))
            )
            prompt_text = enhance_prompt(
                self.llm_client,
                prompt_text,
                "video_h3" if selected_renderer == "h3" else "video",
                force=enhance,
                context=(
                    f"Target duration is {length / 24.0:.2f} seconds at 24 fps."
                    if selected_renderer == "h3"
                    else ""
                ),
            )
            if negative_prompt is not None:
                neg = negative_prompt
            elif selected_renderer == "h3":
                neg = (
                    "low quality, blurry, distorted anatomy, duplicate limbs, frozen motion, "
                    "flicker, unintended subtitles, watermark, logo"
                )
            else:
                neg = (
                    "low quality, blurry, distorted anatomy, duplicate limbs, frozen motion, "
                    "flicker, abrupt cut, scene transition, subtitles, watermark, logo, text"
                )
            w, h = self._dimensions(width, height, renderer=selected_renderer)

            # Fail before acquiring scarce compute if any exact model component is absent.
            if selected_renderer == "h3":
                selected_model = self._resolve_h3_stack()[0]
                release = self.H3_MODEL_RELEASE
            else:
                selected_model = self._resolve_ltx_model()
                release = self.LTX_MODEL_RELEASE

            self._ensure_comfyui_running(required_vram=40.0)

            if mode == "extend_video":
                # Seed from the last frame of the input video, then image_to_video.
                src = init_video or init_image
                if not src:
                    return self._failure(
                        mode, "init_video is required", code="invalid_parameters"
                    )
                self._validate_video(os.path.abspath(src))
                local_name = "source.mp4"
                shutil.copyfile(os.path.abspath(src), os.path.join(work_dir, local_name))
                self._extract_last_frame(work_dir, local_name, "seed.png")
                init_image_name = self._upload_image(os.path.join(work_dir, "seed.png"))
                mode = "image_to_video"
            elif mode == "edit_video":
                staged_video = self._stage_video(init_video)
            elif mode == "image_to_video":
                init_image_name = self._upload_image(init_image)
            elif mode == "keyframes":
                prepared: list[dict] = []
                occupied: set[int] = set()
                for raw_keyframe in keyframes:
                    if not isinstance(raw_keyframe, dict) or not raw_keyframe.get("image"):
                        return self._failure(
                            mode,
                            "each keyframe needs image, at_frame, and optional strength",
                            code="invalid_parameters",
                        )
                    raw_at_frame = _int(raw_keyframe.get("at_frame"), 0)
                    if selected_renderer == "h3":
                        at_frame = length - 1 if raw_at_frame >= requested_frames - 1 else 0
                    else:
                        at_frame = max(0, min(length - 1, raw_at_frame))
                    if at_frame in occupied:
                        return self._failure(
                            mode,
                            f"duplicate keyframe at frame {at_frame}",
                            code="invalid_parameters",
                        )
                    occupied.add(at_frame)
                    try:
                        strength = max(0.0, min(1.0, float(raw_keyframe.get("strength", 1.0))))
                    except (TypeError, ValueError):
                        strength = 1.0
                    prepared.append(
                        {
                            "image": str(raw_keyframe["image"]),
                            "at_frame": at_frame,
                            "strength": strength,
                            "_name": self._upload_image(str(raw_keyframe["image"])),
                        }
                    )
                keyframes = sorted(prepared, key=lambda item: item["at_frame"])

            if selected_renderer == "h3":
                first_image_name = init_image_name
                last_image_name = None
                if requested_mode == "keyframes":
                    first_image_name = next(
                        (item["_name"] for item in keyframes or [] if item["at_frame"] == 0),
                        None,
                    )
                    last_image_name = next(
                        (item["_name"] for item in keyframes or [] if item["at_frame"] == length - 1),
                        None,
                    )
                prompt_text = self._structured_h3_prompt(
                    prompt_text,
                    has_first=bool(first_image_name),
                    has_last=bool(last_image_name),
                    duration=length / 24.0,
                    negative_prompt=neg,
                )
                workflow = self._build_h3_workflow(
                    prompt_text,
                    w,
                    h,
                    length,
                    first_image_name,
                    last_image_name,
                    seed,
                )
                self._render(
                    workflow,
                    abs_output,
                    output_node="14",
                    require_audio=True,
                    timeout=3600,
                )
                size = self._validate_video(abs_output, require_audio=True)
                media_evidence = "native synchronized stereo audio"
            else:
                workflow = self._build_ltx_workflow(
                    mode,
                    prompt_text,
                    neg,
                    w,
                    h,
                    length,
                    init_image_name,
                    keyframes,
                    staged_video,
                    denoise,
                    seed,
                )
                self._render(workflow, abs_output)
                size = self._validate_video(abs_output)
                media_evidence = "silent H.264 MP4"
            return ToolResult(
                tool_name="generate_video",
                status=ToolStatus.OK,
                changed=True,
                summary=(
                    f"Generated a verified {requested_mode} video with {release} "
                    f"at {abs_output}"
                ),
                evidence=[
                    f"renderer={selected_renderer}; model={selected_model}",
                    f"video={w}x{h}, {length} frames at 24 fps, {size} bytes",
                    media_evidence,
                ],
                artifacts=[abs_output],
                side_effect=SideEffect.LOCAL_MUTATION,
            )

        except Exception as e:
            return self._failure(requested_mode, str(e))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            if staged_video:
                try: os.remove(os.path.join(_comfy_output_host(), os.path.basename(staged_video)))
                except Exception: pass
            # Release only this invocation's opaque demand. Fleet owns runtime
            # warmth, teardown, and all coordinator claims.
            self._finish_comfy_session()
