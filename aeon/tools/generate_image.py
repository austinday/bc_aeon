import os
import json
import time
import random
import requests
import subprocess
from .base import BaseTool
from ..core.prompts import TOOL_DESC_GENERATE_IMAGE, TOOL_DESC_EDIT_IMAGE
from ..core.gpu_queue import (
    current_lease,
    heartbeat_vram,
    release_vram,
    wait_for_vram,
)
from ..core.prompt_enhancer import enhance_prompt
from ..core.paths import resolve_output_dir

FLEET_LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"

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

    def _manage_registry(self, action: str):
        """Manage active users and reap only Aeon's exact labeled container."""
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
                    state = {"pids": []}
            except (json.JSONDecodeError, EOFError):
                state = {"pids": []}
                
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
                    label = subprocess.run(
                        ["docker", "inspect", "-f",
                         "{{ index .Config.Labels \"com.bc_aeon.component\" }}",
                         "aeon_comfyui"], capture_output=True, text=True)
                    if label.returncode == 0 and label.stdout.strip() != "comfyui":
                        raise RuntimeError(
                            "Refusing to remove an aeon_comfyui container without "
                            "the bc_aeon ownership label."
                        )
                    if label.returncode == 0:
                        stopped = subprocess.run(
                            ["docker", "rm", "-f", "aeon_comfyui"],
                            capture_output=True, text=True)
                        if stopped.returncode != 0:
                            raise RuntimeError(stopped.stderr.strip() or "Could not stop ComfyUI")
                    remains = subprocess.run(
                        ["docker", "inspect", "aeon_comfyui"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    state["reaped"] = remains.returncode != 0
                else:
                    state["reaped"] = False

            reaped = state.pop("reaped", False)
            with open(registry_path, 'w') as f:
                json.dump(state, f)

            count = len(state["pids"])
            if action == 'reap_if_idle':
                return count, reaped
            return count, None

    def _finish_comfy_session(self):
        """Drop our slot, then stop and release only after the last caller exits."""
        import fcntl
        with open("/tmp/aeon_comfyui_start.lock", 'w') as start_fd:
            fcntl.flock(start_fd, fcntl.LOCK_EX)
            self._manage_registry('unregister')
            _, reaped = self._manage_registry('reap_if_idle')
            if reaped and current_lease():
                release_vram("Aeon ComfyUI container stopped after final tool call")
                print(f"{self.C_CYAN}ComfyUI stopped and its coordinator lease was released."
                      f"{self.C_RESET}")

    def _reap_if_idle(self):
        """Tear down the shared container iff no Comfy operation is in flight."""
        try:
            _, reaped = self._manage_registry('reap_if_idle')
            if reaped and current_lease():
                release_vram("Aeon ComfyUI container stopped while idle")
        except Exception:
            pass

    @staticmethod
    def _container_pid():
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Pid}}", "aeon_comfyui"],
            capture_output=True, text=True)
        if result.returncode != 0:
            return None
        try:
            return int(result.stdout.strip())
        except ValueError:
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
        self._manage_registry('register')

        # Fast path: reuse only when its live coordinator lease can be refreshed.
        if self._check_comfyui_health():
            heartbeat_vram(self._container_pid(), "Aeon reused healthy ComfyUI")
            return True

        import fcntl
        with open("/tmp/aeon_comfyui_start.lock", 'w') as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)  # serialize starts across all agents
            # Someone may have started it while we waited for the lock.
            if self._check_comfyui_health():
                heartbeat_vram(self._container_pid(), "Aeon reused healthy ComfyUI")
                return True

            print(f"{self.C_CYAN}Requesting a coordinator-approved, hard-capped "
                  f"{required_vram:g}GB ComfyUI lease on .177...{self.C_RESET}")
            if not os.path.isfile(FLEET_LOW_PRIORITY) or not os.access(
                FLEET_LOW_PRIORITY, os.X_OK
            ):
                raise RuntimeError(
                    f"Renter-yielding launcher is unavailable: {FLEET_LOW_PRIORITY}"
                )
            lease = wait_for_vram(required_vram)

            placement = ("shared with Qwen under independent hard caps"
                         if lease.get("shared_with_qwen") else "separate tool placement")
            print(f"{self.C_CYAN}Starting ComfyUI on leased UUID {lease['gpu_uuid']} "
                  f"({placement})...{self.C_RESET}")
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "start_comfyui.sh"))
            env = os.environ.copy()
            env["AEON_HOME"] = os.environ.get("AEON_HOME", os.path.expanduser("~/.aeon"))
            env["GPU_AGENT_CLAIM_ID"] = lease["claim_id"]
            env["CUDA_VISIBLE_DEVICES"] = lease["gpu_uuid"]
            env["GPU_MEM_LIMIT_GB"] = f"{lease['vram_budget_gb']:g}"
            env["GPU_RESERVE_GB"] = "6"
            res = subprocess.run(
                [FLEET_LOW_PRIORITY, "bash", script_path],
                capture_output=True,
                text=True,
                env=env,
            )
            if res.returncode != 0:
                release_vram("ComfyUI launch failed before a GPU process started")
                raise RuntimeError(f"Error starting ComfyUI: {res.stderr}")

            print(f"{self.C_CYAN}Waiting for ComfyUI to become healthy...{self.C_RESET}")
            for _ in range(90):  # up to ~180s for a cold container + first boot
                if self._check_comfyui_health():
                    heartbeat_vram(self._container_pid(), "Aeon ComfyUI started and is healthy")
                    return True
                time.sleep(2)
            subprocess.run(["docker", "rm", "-f", "aeon_comfyui"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            release_vram("ComfyUI failed its startup health check and was stopped")
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
        next_heartbeat = start + 300
        missing = 0
        while time.time() - start < hard_timeout:
            if time.time() >= next_heartbeat:
                heartbeat_vram(self._container_pid(), f"Aeon ComfyUI job {prompt_id} is active")
                next_heartbeat = time.time() + 300
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

    def _upload_image(self, abs_path: str) -> str:
        """Upload a local image to ComfyUI and return the server-side filename."""
        with open(abs_path, "rb") as f:
            r = requests.post(f"{self.comfy_url}/upload/image", files={"image": f}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to upload image to ComfyUI: {r.text[:200]}")
        return r.json()["name"]

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

    Prefers the installed uncensored FLUX.2-klein GGUF pair and falls back to the
    abliterated FLUX.1-dev GGUF plus its dual text encoders.
    """
    def __init__(self, llm_client=None):
        super().__init__(
            name="generate_image",
            description=TOOL_DESC_GENERATE_IMAGE,
            underlying_model='FLUX.2-klein uncensored GGUF (FLUX.1 abliterated fallback)'
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
        """Resolve an uncensored/abliterated FLUX.2 encoder, GGUF preferred."""
        return self._resolve("text_encoders",
                             ("*flux2*klein*uncensored*.gguf", "*flux2*uncensored*.gguf",
                              "*mistral*abliterated*flux2*.safetensors", "*flux2*abliterated*.safetensors",
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
        """Resolve the FLUX.2 set matching the selected text encoder."""
        unet = self._resolve("unet", ("*flux-2-klein*.gguf", "*flux2*klein*.gguf",
                                      "*flux2*dev*.gguf", "*flux-2-dev*.gguf"),
                             "flux-2-klein-9b-Q8_0.gguf")
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

    def execute(self, prompt: str, output_dir: str = None, width: int = 1024, height: int = 1024, enhance: bool = None) -> str:
        if not prompt:
            return "Error: 'prompt' parameter is required."
        if not output_dir or not str(output_dir).strip():
            return "Error: 'output_dir' is required — the directory to save the generated image in."

        # Tolerate string/odd dimensions from the model (e.g. "1024" or 1000).
        width = self._norm_dim(width)
        height = self._norm_dim(height)

        prompt = enhance_prompt(self.llm_client, prompt, "image", force=enhance)
        # Auto-name the file inside the caller-provided output_dir (relative dirs
        # resolve against the workspace aeon was launched from).
        abs_output_path = str(resolve_output_dir(output_dir, time.strftime("aeon_image_%Y%m%d_%H%M%S.png")))
        os.makedirs(os.path.dirname(abs_output_path) or ".", exist_ok=True)

        try:
            # Register this agent as an active user and ensure server is running on allocated VRAM
            self._manage_registry('register')
            self._ensure_comfyui_running(required_vram=24.0)

            seed = random.randint(1, 0xffffffffffffffff)
            flux2_te = self._flux2_dev_te()
            if flux2_te:
                # FLUX.2 with its uncensored encoder and matching VAE.
                unet, te, vae = self._flux2_dev_models(flux2_te)
                print(f"{self.C_CYAN}Model: FLUX.2 ({unet}) + {te}{self.C_RESET}")
                clip_loader = "CLIPLoaderGGUF" if te.lower().endswith(".gguf") else "CLIPLoader"
                workflow = self._flux_workflow(
                    unet_node={"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}},
                    clip_node={"class_type": clip_loader, "inputs": {"clip_name": te, "type": "flux2"}},
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

    def execute(self, input_path: str, prompt: str, output_dir: str = None,
                input_path_2: str = None, input_path_3: str = None,
                denoise: float = 0.75, enhance: bool = None) -> str:
        if not input_path:
            return "Error: 'input_path' parameter is required."
        if not prompt:
            return "Error: 'prompt' parameter is required."
        if not output_dir or not str(output_dir).strip():
            return "Error: 'output_dir' is required — the directory to save the edited image in."

        denoise = self._norm_unit(denoise, default=0.75)
        prompt = enhance_prompt(self.llm_client, prompt, "image_edit", force=enhance)
        abs_input_path = os.path.abspath(input_path)
        # Auto-name '<input-name>_edited.png' inside the caller-provided output_dir.
        default_name = os.path.splitext(os.path.basename(abs_input_path))[0] + "_edited.png"
        abs_output_path = str(resolve_output_dir(output_dir, default_name))

        if not os.path.exists(abs_input_path):
            return f"Error: Input image not found at {abs_input_path}"
        # Optional reference images (multi-image edit): image1 is the base that gets
        # edited (its latent seeds the result); image2/image3 are references the
        # model can pull content from — e.g. a brand logo or product to place into
        # the base scene. Qwen-Image-Edit-2509 (TextEncodeQwenImageEditPlus) reads
        # up to three images and follows a prompt like "add the logo from the second
        # image to the top-right of the first image".
        extra_paths = []
        for p in (input_path_2, input_path_3):
            if not p:
                continue
            ap = os.path.abspath(p)
            if not os.path.exists(ap):
                return f"Error: reference image not found at {ap}"
            extra_paths.append(ap)

        os.makedirs(os.path.dirname(abs_output_path) or ".", exist_ok=True)

        try:
            # Register this agent as an active user and ensure server is running on allocated VRAM
            self._manage_registry('register')
            self._ensure_comfyui_running(required_vram=40.0)

            n_imgs = 1 + len(extra_paths)
            print(f"{self.C_CYAN}Uploading {n_imgs} image(s) to ComfyUI...{self.C_RESET}")
            base_name = self._upload_image(abs_input_path)
            # LoadImage nodes: base is "10"; references get 15, 16. image_inputs maps
            # each to image1/image2/image3 on the TextEncode nodes.
            load_nodes = {"10": base_name}
            image_inputs = {"image1": ["10", 0]}
            for idx, ap in enumerate(extra_paths, start=2):
                nid = str(13 + idx)  # 15, 16
                load_nodes[nid] = self._upload_image(ap)
                image_inputs[f"image{idx}"] = [nid, 0]

            workflow = {
                "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "v23/Qwen-Rapid-NSFW-v23_Q8_0.gguf"}},
                # ABLITERATED Qwen2.5-VL text encoder (was stock qwen_2.5_vl_7b_fp8_scaled,
                # the last censored component). CLIPLoaderGGUF auto-pairs the matching
                # mmproj-*.gguf alongside it (needed because Qwen-Image-Edit reads the input
                # image in vision-language mode), and ignores the unrelated gemma-3 mmproj.
                "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": "Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf", "type": "qwen_image"}},
                "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
                "4": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": prompt, "clip": ["2", 0], "vae": ["3", 0], **image_inputs}},
                "5": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": "", "clip": ["2", 0], "vae": ["3", 0], **image_inputs}},
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
            # Attach the LoadImage node(s) for the base + any reference images.
            for nid, nm in load_nodes.items():
                workflow[nid] = {"class_type": "LoadImage", "inputs": {"image": nm}}

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
