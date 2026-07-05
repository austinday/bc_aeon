import os
import json
import time
import random
import requests
import subprocess
import threading
from .base import BaseTool
from ..core.prompts import TOOL_DESC_GENERATE_IMAGE, TOOL_DESC_EDIT_IMAGE
from ..core.gpu_queue import wait_for_vram, release_vram
from ..core.prompt_enhancer import enhance_prompt
from ..core.paths import resolve_output_path

# ComfyUI is a SHARED, VRAM-heavy service (image gen, image edit, and video all
# hit one container). It must free its ~20GB for other tools when the agent moves
# on — but tearing it down after every single call cold-started the model on each
# image and caused an unreachable-server timeout loop. So we debounce: keep it
# warm across a burst of comfy ops, then reap it once none has run for a grace
# period. Tune with AEON_COMFYUI_IDLE_S (seconds).
_COMFY_IDLE_GRACE_S = float(os.environ.get("AEON_COMFYUI_IDLE_S", "90"))
_reaper_lock = threading.Lock()
_reaper_timer = None  # module-global debounced timer, re-armed on each op finish


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
            elif action == 'reap_if_idle':
                # Atomic under the registry lock: if NO comfy op is in flight
                # (across every agent/tool), free the shared container's VRAM.
                # Registering happens under this same lock, so a starting op
                # either bumps the count before we check (we skip) or starts a
                # fresh container after our teardown (clean) — no half-dead race.
                if not state["pids"]:
                    subprocess.run(["docker", "rm", "-f", "aeon_comfyui"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    state["gpu_id"] = None
                    state["reaped"] = True
                else:
                    state["reaped"] = False

            reaped = state.pop("reaped", False)
            with open(registry_path, 'w') as f:
                json.dump(state, f)

            count = len(state["pids"])
            if action == 'reap_if_idle':
                return count, reaped
            return count, state.get("gpu_id")

    def _finish_comfy_session(self):
        """Call from a tool's `finally`: drop our in-flight slot + VRAM reservation,
        then arm a debounced idle reaper. ComfyUI stays warm across rapid
        sequential ops (generate -> edit -> video reuse one loaded server) but is
        torn down — freeing its VRAM for other tools — once no comfy op has run for
        _COMFY_IDLE_GRACE_S. Restores the original bring-up/tear-down VRAM sharing
        without the per-call cold-start thrash. Session-exit cleanup (main.py's
        _safe_cleanup) remains the backstop."""
        try:
            self._manage_registry('unregister')
        finally:
            release_vram()
        global _reaper_timer
        with _reaper_lock:
            if _reaper_timer is not None:
                _reaper_timer.cancel()
            _reaper_timer = threading.Timer(_COMFY_IDLE_GRACE_S, self._reap_if_idle)
            _reaper_timer.daemon = True
            _reaper_timer.start()
        print(f"{self.C_CYAN}Comfy op done. ComfyUI stays warm for "
              f"{int(_COMFY_IDLE_GRACE_S)}s of idle, then its VRAM is released.{self.C_RESET}")

    def _reap_if_idle(self):
        """Debounced-timer callback: tear the shared ComfyUI container down iff no
        comfy op is in flight. A new op that arrived during the grace window keeps
        it alive (its finish re-arms this timer)."""
        try:
            _, reaped = self._manage_registry('reap_if_idle')
            if reaped:
                print(f"{self.C_CYAN}ComfyUI idle for {int(_COMFY_IDLE_GRACE_S)}s — "
                      f"container reaped, VRAM released for other tools.{self.C_RESET}")
        except Exception:
            pass

    def _registry_gpu(self):
        """Peek the GPU the shared ComfyUI was last started on (or None)."""
        import fcntl
        try:
            with open("/tmp/aeon_comfyui_registry.lock", 'w') as lock_fd:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                if os.path.exists("/tmp/aeon_comfyui_registry.json"):
                    with open("/tmp/aeon_comfyui_registry.json") as f:
                        return json.load(f).get("gpu_id")
        except Exception:
            pass
        return None

    def _ensure_comfyui_running(self, required_vram: float = 20.0):
        """Ensure the SHARED ComfyUI is up, safely under concurrent agents.

        Concurrency rules that keep multiple agents from crashing each other:
          - If it's already healthy, USE IT AS-IS. Never restart a running server
            (that would kill other agents' in-flight jobs) and never reserve VRAM
            again (its memory is already accounted for by the live container).
          - If it needs starting, do so under a cross-process START lock so only
            ONE agent runs start_comfyui.sh; the rest block, then find it healthy.
        """
        # Fast path: warm and healthy -> reuse, no reservation, no restart.
        if self._check_comfyui_health():
            return True

        import fcntl
        with open("/tmp/aeon_comfyui_start.lock", 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)  # serialize starts across all agents
            # Someone may have started it while we waited for the lock.
            if self._check_comfyui_health():
                return True

            # Reserve VRAM only now, for the actual model load. Prefer the GPU the
            # server last used; else any GPU with room. wait_for_vram blocks (with
            # its own timeout) if the GPU is full -> callers wait in line, not crash.
            current_gpu = self._registry_gpu()
            print(f"{self.C_CYAN}Reserving {required_vram}GB VRAM for ComfyUI "
                  f"(target GPU: {current_gpu if current_gpu is not None else 'any'})...{self.C_RESET}")
            allocated_gpu = wait_for_vram(required_vram, gpu_id=current_gpu)
            self._manage_registry('register', gpu_id=allocated_gpu)

            print(f"{self.C_CYAN}Starting ComfyUI on GPU {allocated_gpu}...{self.C_RESET}")
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "start_comfyui.sh"))
            env = os.environ.copy()
            env["AEON_HOME"] = os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))
            env["COMFYUI_GPU"] = str(allocated_gpu)
            res = subprocess.run(["bash", script_path], capture_output=True, text=True, env=env)
            if res.returncode != 0:
                raise RuntimeError(f"Error starting ComfyUI: {res.stderr}")

            print(f"{self.C_CYAN}Waiting for ComfyUI to become healthy...{self.C_RESET}")
            for _ in range(90):  # up to ~180s for a cold container + first boot
                if self._check_comfyui_health():
                    return True
                time.sleep(2)
            raise RuntimeError("Error: ComfyUI failed to become healthy after starting.")

    def _prompt_in_queue(self, prompt_id: str) -> bool:
        """Is our prompt still running or pending in ComfyUI's queue? Used to tell
        'legitimately waiting behind another agent's job' from 'lost'. On a
        transient error we assume still-queued (patience over a false failure)."""
        try:
            q = requests.get(f"{self.comfy_url}/queue", timeout=10).json()
        except requests.RequestException:
            return True
        for key in ("queue_running", "queue_pending"):
            for item in q.get(key, []) or []:
                if isinstance(item, (list, tuple)) and len(item) > 1 and item[1] == prompt_id:
                    return True
        return False

    def _await_comfy(self, prompt_id: str, node: str = "9", hard_timeout: int = 1800):
        """Wait for a submitted prompt's SaveImage output, tolerating time spent
        QUEUED behind other agents' jobs so concurrent callers don't false-timeout
        waiting their turn. Returns the node's output dict; raises on real failure
        or the (generous) hard cap. `hard_timeout` counts wall time, but a job that
        is still visibly in the queue never trips the 'lost' check."""
        start = time.time()
        missing = 0
        while time.time() - start < hard_timeout:
            try:
                hist = requests.get(f"{self.comfy_url}/history/{prompt_id}", timeout=10).json()
            except requests.RequestException:
                time.sleep(2)
                continue
            if prompt_id in hist:
                outputs = hist[prompt_id].get("outputs", {})
                if node in outputs:
                    return outputs[node]
                status = hist[prompt_id].get("status", {})
                raise RuntimeError(f"ComfyUI finished but produced no image (status={status}).")
            # Not done. Still queued/running = we are legitimately waiting our turn.
            if self._prompt_in_queue(prompt_id):
                missing = 0
            else:
                missing += 1
                if missing >= 5:  # ~10s absent from BOTH history and queue -> dropped
                    raise RuntimeError("ComfyUI job vanished from the queue without producing output.")
            time.sleep(2)
        raise RuntimeError(f"ComfyUI timed out after {hard_timeout // 60} min (job still not complete).")

    def _download_comfy_output(self, info: dict, dest: str, timeout: int = 30):
        """Download a produced file (image or video) from ComfyUI's /view to
        `dest`; raise on failure. Larger timeout for big video files."""
        r = requests.get(f"{self.comfy_url}/view", params={
            "filename": info["filename"],
            "subfolder": info.get("subfolder", ""),
            "type": info.get("type", "output"),
        }, timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download output from ComfyUI (HTTP {r.status_code}).")
        with open(dest, "wb") as f:
            f.write(r.content)

class GenerateImageTool(ComfyUITool):
    """Generate images with an abliterated/uncensored FLUX model via local ComfyUI.

    Default model is FLUX.1-dev-abliterated-V2 (fully abliterated, open/ungated),
    using the FLUX.1 dual-encoder graph (t5xxl + clip_l). FLUX.2-dev is a drop-in
    upgrade once its abliterated Mistral text encoder is available: drop a
    `mistral*flux2*.safetensors` into text_encoders and the resolver switches to
    the higher-quality FLUX.2 graph automatically.
    """
    def __init__(self, llm_client=None):
        super().__init__(
            name="generate_image",
            description=TOOL_DESC_GENERATE_IMAGE,
            underlying_model='FLUX.1-dev-abliterated-V2 (uncensored GGUF)'
        )
        self.llm_client = llm_client

    def _resolve(self, subdir: str, patterns, default: str = None) -> str:
        """Basename of the first model in comfyui/<subdir> matching any pattern; else default."""
        import glob
        base = os.path.join(os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon")),
                            "models", "comfyui", subdir)
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(base, pat)))
            if hits:
                return os.path.basename(hits[0])
        return default

    def _flux2_dev_te(self):
        """The FLUX.2-dev Mistral text encoder (safetensors), if present — its
        arrival is what upgrades generation from FLUX.1 to FLUX.2-dev. Prefers an
        abliterated build over a base one. Returns None when not installed yet."""
        return self._resolve("text_encoders",
                             ("*mistral*abliterated*flux2*.safetensors", "*flux2*abliterated*.safetensors",
                              "*mistral*flux2*.safetensors", "*flux2*dev*te*.safetensors"))

    def _flux1_models(self):
        """Resolve the abliterated FLUX.1-dev set (auto-adapts to whichever quant
        is present). Encoders/VAE are the standard FLUX.1 pieces already on disk."""
        unet = self._resolve("unet",
                             ("*flux*1*dev*abliterated*.gguf", "*flux*dev*abliterated*.gguf", "*abliterated*V2*.gguf"),
                             "T8-flux.1-dev-abliterated-V2-GGUF-Q8_0.gguf")
        clip_l = self._resolve("text_encoders", ("clip_l.safetensors", "*clip_l*.safetensors"), "clip_l.safetensors")
        t5 = self._resolve("text_encoders", ("t5xxl_fp8*.safetensors", "t5xxl*.safetensors"), "t5xxl_fp8_e4m3fn.safetensors")
        vae = self._resolve("vae", ("ae.safetensors", "*flux*ae*.safetensors"), "ae.safetensors")
        return unet, clip_l, t5, vae

    def _flux2_dev_models(self, te):
        """Resolve the FLUX.2-dev set (used only once the Mistral TE `te` is present)."""
        unet = self._resolve("unet", ("*flux2*dev*.gguf", "*flux-2-dev*.gguf"), "flux2-dev-Q8_0.gguf")
        vae = self._resolve("vae", ("*flux2*vae*.safetensors", "flux2-vae*.safetensors"), "flux2-vae.safetensors")
        return unet, te, vae

    @staticmethod
    def _flux_workflow(*, unet_node, clip_node, vae_name, latent_node, scheduler_node,
                       prompt, guidance, seed):
        """Assemble a FLUX text-to-image graph. The sampler chain (encode →
        guidance → custom-advanced sampler → decode → save) is identical across
        FLUX.1 and FLUX.2; only the loaders, latent, and scheduler differ, so those
        are passed in. Keeps the two model paths from duplicating ~15 nodes each."""
        return {
            "1": unet_node,
            "2": clip_node,
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            "5": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": guidance}},
            "6": latent_node,
            "7": scheduler_node,
            "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "14": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["5", 0]}},
            "11": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["10", 0], "guider": ["14", 0], "sampler": ["8", 0],
                              "sigmas": ["7", 0], "latent_image": ["6", 0]}},
            "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "Aeon", "images": ["12", 0]}},
        }

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

            seed = random.randint(1, 0xffffffffffffffff)
            flux2_te = self._flux2_dev_te()
            if flux2_te:
                # FLUX.2-dev (higher quality) — active once the abliterated Mistral
                # text encoder is installed: GGUF UNet + safetensors Mistral TE via
                # native CLIPLoader + flux2 VAE/latent/scheduler.
                unet, te, vae = self._flux2_dev_models(flux2_te)
                print(f"{self.C_CYAN}Model: FLUX.2-dev ({unet}) + {te}{self.C_RESET}")
                workflow = self._flux_workflow(
                    unet_node={"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}},
                    clip_node={"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "flux2"}},
                    vae_name=vae,
                    latent_node={"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
                    scheduler_node={"class_type": "Flux2Scheduler", "inputs": {"steps": 20, "width": width, "height": height}},
                    prompt=prompt, guidance=4.0, seed=seed)
            else:
                # FLUX.1-dev-abliterated (default) — fully abliterated, ungated:
                # GGUF UNet + dual encoder (t5xxl + clip_l) + flux `ae` VAE.
                unet, clip_l, t5, vae = self._flux1_models()
                print(f"{self.C_CYAN}Model: FLUX.1-dev-abliterated ({unet}){self.C_RESET}")
                workflow = self._flux_workflow(
                    unet_node={"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}},
                    clip_node={"class_type": "DualCLIPLoader", "inputs": {"clip_name1": t5, "clip_name2": clip_l, "type": "flux"}},
                    vae_name=vae,
                    latent_node={"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
                    scheduler_node={"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}},
                    prompt=prompt, guidance=3.5, seed=seed)

            print(f"{self.C_CYAN}Submitting image generation workflow to ComfyUI...{self.C_RESET}")
            req = requests.post(f"{self.comfy_url}/prompt", json={"prompt": workflow}, timeout=5)
            if req.status_code != 200:
                return f"Error submitting workflow to ComfyUI: {req.text}"
            
            prompt_id = req.json()["prompt_id"]

            print(f"{self.C_CYAN}Waiting for image generation to complete...{self.C_RESET}")
            node_out = self._await_comfy(prompt_id, node="9")  # queue-aware; raises on failure
            self._download_comfy_output(node_out["images"][0], abs_output_path)
            return f"Successfully generated image and saved to: {abs_output_path}"

        except Exception as e:
            return self.format_error_message(e, "generating image via ComfyUI", "checking if ComfyUI is running correctly")
        
        finally:
            # Keep ComfyUI warm across a burst of ops, reap it once idle (see
            # _finish_comfy_session). Fixes the per-call teardown that cold-started
            # the ~20GB model on every image, while still freeing VRAM for other
            # tools when image work stops.
            self._finish_comfy_session()


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
            node_out = self._await_comfy(prompt_id, node="9")  # queue-aware; raises on failure
            self._download_comfy_output(node_out["images"][0], abs_output_path)
            return f"Successfully edited image and saved to: {abs_output_path}"

        except Exception as e:
            return self.format_error_message(e, "editing image via ComfyUI", "checking if ComfyUI is running correctly")
        
        finally:
            # Warm across bursts, reap when idle (see _finish_comfy_session): the
            # per-call teardown cold-started the model on every edit and caused the
            # same "server unreachable" timeout loop.
            self._finish_comfy_session()
