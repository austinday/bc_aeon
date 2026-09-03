import os
import re
import time
import random
import uuid
from collections.abc import Mapping
from numbers import Real
from urllib.parse import urlparse

import requests
from .base import BaseTool
from ..core.fleet_backend import (
    DEFAULT_BROKER_SOCKET,
    FleetBackendError,
    FleetBrokerClient,
)
from ..core.prompts import TOOL_DESC_GENERATE_IMAGE, TOOL_DESC_EDIT_IMAGE
from ..core.prompt_enhancer import enhance_prompt
from ..core.paths import resolve_output_dir

_COMFY_SERVICE_ID = "aeon-comfyui"
_COMFY_TICKET_ID = re.compile(r"^fd-[0-9a-f]{32}$")
_COMFY_TICKET_TTL_SECONDS = 900
_COMFY_WAIT_TIMEOUT_SECONDS = 1800
_COMFY_RENEW_INTERVAL_SECONDS = 300
_LOCAL_HTTP_KWARGS = {
    "allow_redirects": False,
    "proxies": {"http": "", "https": ""},
}

class ComfyUITool(BaseTool):
    """Base class for one-call, broker-owned ComfyUI service sessions."""
    COMFY_SERVICE_ID = _COMFY_SERVICE_ID
    COMFY_TICKET_TTL_SECONDS = _COMFY_TICKET_TTL_SECONDS
    COMFY_WAIT_TIMEOUT_SECONDS = _COMFY_WAIT_TIMEOUT_SECONDS
    COMFY_RENEW_INTERVAL_SECONDS = _COMFY_RENEW_INTERVAL_SECONDS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comfy_url = "http://127.0.0.1:8188"
        self._fleet_ticket_id = None
        self._fleet_consumer_id = None

    @staticmethod
    def _new_fleet_consumer() -> str:
        """Return an invocation-unique consumer identity for one media demand."""

        return f"aeon/tool/comfy/{os.getpid()}/{uuid.uuid4().hex}"

    @staticmethod
    def _fleet_client() -> FleetBrokerClient:
        """Use Aeon's ACL-validating client for the owner-only Fleet socket."""

        socket_path = os.environ.get(
            "AEON_FLEET_SOCKET", str(DEFAULT_BROKER_SOCKET)
        )
        return FleetBrokerClient(socket_path, timeout=15)

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
            response = requests.get(
                f"{self.comfy_url}/system_stats",
                timeout=(2, 10),
                **_LOCAL_HTTP_KWARGS,
            )
            if response.status_code != 200 or len(response.content) > 2 * 1024 * 1024:
                return False
            return isinstance(response.json(), dict)
        except (ValueError, requests.exceptions.RequestException):
            return False

    @staticmethod
    def _validate_comfy_endpoint(value) -> str:
        """Accept only a credential-free loopback HTTP origin for ComfyUI."""

        if (
            not isinstance(value, str)
            or not value
            or len(value) > 512
            or any(
                character.isspace()
                or ord(character) < 0x20
                or ord(character) == 0x7F
                for character in value
            )
        ):
            raise RuntimeError("Fleet returned an invalid ComfyUI endpoint")
        parsed = urlparse(value)
        try:
            port = parsed.port
        except ValueError as exc:
            raise RuntimeError("Fleet returned an invalid ComfyUI endpoint") from exc
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.params
            or parsed.path.rstrip("/") != ""
            or port is None
            or port < 1024
            or port > 65535
        ):
            raise RuntimeError("Fleet returned a non-loopback ComfyUI endpoint")
        canonical = (
            f"http://[::1]:{port}"
            if parsed.hostname == "::1"
            else f"http://127.0.0.1:{port}"
        )
        if value not in {canonical, canonical + "/"}:
            raise RuntimeError("Fleet returned a non-canonical ComfyUI endpoint")
        return canonical

    @classmethod
    def _validate_active_ticket(
        cls, value, *, expected_ticket_id=None, expected_consumer=None
    ):
        """Validate the complete sanitized proof for one active demand ticket."""

        if not isinstance(value, Mapping):
            raise RuntimeError("Fleet returned a malformed ComfyUI ticket")
        ticket_id = value.get("ticket_id")
        if not isinstance(ticket_id, str) or _COMFY_TICKET_ID.fullmatch(ticket_id) is None:
            raise RuntimeError("Fleet returned an invalid ComfyUI ticket ID")
        if expected_ticket_id is not None and ticket_id != expected_ticket_id:
            raise RuntimeError("Fleet returned status for a different ComfyUI ticket")
        if expected_consumer is not None and value.get("consumer") != expected_consumer:
            raise RuntimeError("Fleet changed the ComfyUI demand consumer identity")
        if (
            value.get("profile_id") != cls.COMFY_SERVICE_ID
            or value.get("service_id") != cls.COMFY_SERVICE_ID
            or value.get("state") != "active"
        ):
            raise RuntimeError("Fleet returned an inconsistent ComfyUI demand")
        compute_state = value.get("compute_state")
        endpoint = value.get("endpoint")
        if compute_state == "waiting_for_compute":
            if endpoint is not None:
                raise RuntimeError("Fleet exposed an endpoint before ComfyUI was ready")
            return ticket_id, compute_state, None
        if compute_state == "ready":
            return ticket_id, compute_state, cls._validate_comfy_endpoint(endpoint)
        raise RuntimeError("Fleet returned an unknown ComfyUI compute state")

    @classmethod
    def _validate_release_proof(
        cls, value, *, expected_ticket_id: str, expected_consumer: str
    ):
        if (
            not isinstance(value, Mapping)
            or value.get("ticket_id") != expected_ticket_id
            or value.get("profile_id") != cls.COMFY_SERVICE_ID
            or value.get("service_id") != cls.COMFY_SERVICE_ID
            or value.get("consumer") != expected_consumer
            or value.get("state") != "released"
            or value.get("compute_state") != "inactive"
            or value.get("endpoint") is not None
        ):
            raise RuntimeError("Fleet did not prove exact ComfyUI ticket release")
        return {"state": "released", "compute_state": "inactive"}

    def _finish_comfy_session(self):
        """Release this call's opaque ticket and require exact terminal proof."""
        ticket_id = self._fleet_ticket_id
        if not ticket_id:
            # No ticket was ever bound, so any provisional consumer belongs to a
            # rejected/cross-wired acquisition and carries no release authority.
            self._fleet_consumer_id = None
            return None
        consumer = self._fleet_consumer_id
        if not isinstance(consumer, str) or not consumer:
            raise RuntimeError("ComfyUI Fleet consumer identity is unavailable")
        try:
            response = self._fleet_client().release_service(ticket_id)
        except FleetBackendError as exc:
            # Keep the exact ID so a later cleanup attempt cannot acquire a new
            # ticket while ownership of this one is unresolved.
            raise RuntimeError("Fleet could not release the ComfyUI ticket") from exc
        proof = self._validate_release_proof(
            response,
            expected_ticket_id=ticket_id,
            expected_consumer=consumer,
        )
        self._fleet_ticket_id = None
        self._fleet_consumer_id = None
        self.comfy_url = "http://127.0.0.1:8188"
        return proof

    def _bind_acquired_comfy_ticket(self, value, *, consumer: str) -> str:
        """Retain only an acquisition proven to belong to this consumer/service."""

        if not isinstance(value, Mapping):
            raise RuntimeError("Fleet returned a malformed ComfyUI ticket")
        ticket_id = value.get("ticket_id")
        if (
            not isinstance(ticket_id, str)
            or _COMFY_TICKET_ID.fullmatch(ticket_id) is None
        ):
            raise RuntimeError("Fleet returned an invalid ComfyUI ticket ID")
        if value.get("consumer") != consumer:
            raise RuntimeError("Fleet returned an unowned ComfyUI demand consumer")
        if (
            value.get("profile_id") != self.COMFY_SERVICE_ID
            or value.get("service_id") != self.COMFY_SERVICE_ID
        ):
            raise RuntimeError("Fleet returned an unowned ComfyUI service identity")
        self._fleet_ticket_id = ticket_id
        return ticket_id

    def _renew_comfy_ticket(self, *, require_ready: bool = False):
        if not self._fleet_ticket_id:
            raise RuntimeError("ComfyUI Fleet ticket is unavailable")
        ticket_id = self._fleet_ticket_id
        consumer = self._fleet_consumer_id
        if not isinstance(consumer, str) or not consumer:
            raise RuntimeError("ComfyUI Fleet consumer identity is unavailable")
        response = self._fleet_client().renew_service(
            ticket_id, ttl_seconds=self.COMFY_TICKET_TTL_SECONDS
        )
        _, compute_state, endpoint = self._validate_active_ticket(
            response,
            expected_ticket_id=ticket_id,
            expected_consumer=consumer,
        )
        if require_ready:
            if compute_state != "ready":
                raise RuntimeError("Fleet-managed ComfyUI capacity is no longer ready")
            if endpoint != self.comfy_url:
                raise RuntimeError("Fleet changed the ComfyUI endpoint during an active job")
        return response

    def _ensure_comfyui_running(self, required_vram: float = 20.0):
        """Acquire the broker-managed ComfyUI service and wait durably for compute."""
        if (
            isinstance(required_vram, bool)
            or not isinstance(required_vram, Real)
            or not 0 < float(required_vram) <= 40
        ):
            raise RuntimeError("requested ComfyUI workload exceeds the reviewed 40 GB profile")
        if self._fleet_ticket_id is not None or self._fleet_consumer_id is not None:
            raise RuntimeError("a previous ComfyUI Fleet ticket is still unresolved")
        try:
            consumer = self._new_fleet_consumer()
            self._fleet_consumer_id = consumer
            response = self._fleet_client().acquire_service(
                profile=self.COMFY_SERVICE_ID,
                consumer=consumer,
                idempotency_key=f"{self.COMFY_SERVICE_ID}/{uuid.uuid4().hex}",
                ttl_seconds=self.COMFY_TICKET_TTL_SECONDS,
            )
            # A valid-looking ID is not ownership proof. Bind only after the
            # exact consumer and reviewed service identities match; an ambiguous
            # cross-wired ID is left to its bounded Fleet TTL.
            raw_ticket_id = self._bind_acquired_comfy_ticket(
                response, consumer=consumer
            )
            self._validate_active_ticket(
                response,
                expected_ticket_id=raw_ticket_id,
                expected_consumer=consumer,
            )

            print(
                f"{self.C_CYAN}Waiting for Fleet-managed ComfyUI capacity "
                f"({float(required_vram):g}GB operation)...{self.C_RESET}"
            )
            deadline = time.monotonic() + self.COMFY_WAIT_TIMEOUT_SECONDS
            next_renewal = time.monotonic() + self.COMFY_RENEW_INTERVAL_SECONDS
            while time.monotonic() < deadline:
                status = self._fleet_client().service_status(raw_ticket_id)
                _, compute_state, endpoint = self._validate_active_ticket(
                    status,
                    expected_ticket_id=raw_ticket_id,
                    expected_consumer=consumer,
                )
                if compute_state == "ready":
                    self.comfy_url = endpoint
                    if self._check_comfyui_health():
                        return True
                if time.monotonic() >= next_renewal:
                    self._renew_comfy_ticket()
                    next_renewal = time.monotonic() + self.COMFY_RENEW_INTERVAL_SECONDS
                time.sleep(2)
            raise RuntimeError(
                "Fleet did not provide safe ComfyUI capacity within 30 minutes"
            )
        except BaseException:
            if self._fleet_ticket_id is not None:
                try:
                    self._finish_comfy_session()
                except BaseException as release_error:
                    raise RuntimeError(
                        "ComfyUI acquisition failed and Fleet did not prove exact ticket release"
                    ) from release_error
            else:
                # The response never proved that a ticket belonged to this
                # invocation. Leave any cross-wired ID to broker TTL, but clear
                # the provisional local identity so a future call can proceed.
                self._fleet_consumer_id = None
            raise

    def _prompt_in_queue(self, prompt_id: str) -> bool:
        """Is our prompt still running or pending in ComfyUI's queue? Used to tell
        'legitimately waiting behind another agent's job' from 'lost'. On a
        transient error we assume still-queued (patience over a false failure)."""
        try:
            q = requests.get(
                f"{self.comfy_url}/queue", timeout=10, **_LOCAL_HTTP_KWARGS
            ).json()
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
                self._renew_comfy_ticket(require_ready=True)
                next_heartbeat = time.time() + 300
            try:
                hist = requests.get(
                    f"{self.comfy_url}/history/{prompt_id}",
                    timeout=10,
                    **_LOCAL_HTTP_KWARGS,
                ).json()
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
            r = requests.post(
                f"{self.comfy_url}/upload/image",
                files={"image": f},
                timeout=30,
                **_LOCAL_HTTP_KWARGS,
            )
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
        }, timeout=timeout, **_LOCAL_HTTP_KWARGS)
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
            req = requests.post(
                f"{self.comfy_url}/prompt",
                json={"prompt": workflow},
                timeout=5,
                **_LOCAL_HTTP_KWARGS,
            )
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
            # Release only this invocation's opaque demand. Fleet decides whether
            # the shared service remains warm for other callers.
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
            req = requests.post(
                f"{self.comfy_url}/prompt",
                json={"prompt": workflow},
                timeout=5,
                **_LOCAL_HTTP_KWARGS,
            )
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
            # Release only this invocation's opaque demand. Fleet owns runtime
            # warmth, teardown, and all coordinator claims.
            self._finish_comfy_session()
