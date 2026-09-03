import os
import json
import time
import base64
import requests
import re
import warnings
from PIL import Image
import io
from .base import BaseTool
from ..core.prompts import TOOL_DESC_ANALYZE_IMAGE
from ..core.model_catalog import VISION_MODEL_NAME, VISION_MODEL_NAMES
from ..core.fleet_backend import FleetBackendError, validate_loopback_endpoint


_LOCAL_HTTP_KWARGS = {
    "allow_redirects": False,
    "proxies": {"http": "", "https": ""},
}
_MAX_VISION_RESPONSE_BYTES = 4 * 1024 * 1024


def _bounded_response_body(response) -> bytes:
    advertised = response.headers.get("content-length")
    if advertised is not None:
        try:
            advertised_size = int(advertised)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("vision endpoint returned an invalid Content-Length") from exc
        if not 0 <= advertised_size <= _MAX_VISION_RESPONSE_BYTES:
            raise RuntimeError("vision endpoint response exceeded the 4 MiB limit")
    payload = bytearray()
    for chunk in response.iter_content(chunk_size=64 * 1024):
        payload.extend(chunk)
        if len(payload) > _MAX_VISION_RESPONSE_BYTES:
            raise RuntimeError("vision endpoint response exceeded the 4 MiB limit")
    return bytes(payload)


class AnalyzeImageTool(BaseTool):
    """Standalone image-analysis tool. Runs on the selected multimodal primary
    model (Qwen3.8), whose OpenAI-compatible endpoint main.py exports as
    AEON_VISION_*. There is no separate vision server or GPU model — it reuses the
    already-loaded primary. (The main browser loop no longer routes through this
    tool; it attaches screenshots to the agent's own multimodal prompt directly.
    This tool remains for explicit, one-off image analysis of any file on disk.)"""

    # Max image dimension before resizing (keeps the request small for standalone
    # image Q&A). The browser path uses its own, larger cap for page legibility.
    MAX_IMAGE_DIM = 640
    MAX_INPUT_BYTES = 32 * 1024 * 1024
    MAX_INPUT_PIXELS = 40_000_000
    MAX_INPUT_SIDE = 16_384

    @classmethod
    def _dimension_error(cls, image) -> str | None:
        try:
            width, height = image.size
        except (TypeError, ValueError):
            return "image dimensions are unavailable"
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
            or width > cls.MAX_INPUT_SIDE
            or height > cls.MAX_INPUT_SIDE
            or width * height > cls.MAX_INPUT_PIXELS
        ):
            return (
                f"image dimensions {width}x{height} exceed the "
                f"{cls.MAX_INPUT_PIXELS:,}-pixel/{cls.MAX_INPUT_SIDE:,}-pixel-side limit"
            )
        return None

    def __init__(self):
        super().__init__(
            name='analyze_image',
            description=TOOL_DESC_ANALYZE_IMAGE,
            underlying_model='multimodal-primary'
        )

    @staticmethod
    def _validate_image(image_path: str):
        """Return None if the path is a decodable image, else an error string.
        Run BEFORE encoding/sending so a non-image or corrupt file fails fast."""
        try:
            input_bytes = os.path.getsize(image_path)
            if input_bytes == 0:
                return f'Error: Image file is empty (0 bytes): {image_path}'
            if input_bytes > AnalyzeImageTool.MAX_INPUT_BYTES:
                return (
                    f"Error: Image file exceeds the "
                    f"{AnalyzeImageTool.MAX_INPUT_BYTES // (1024 * 1024)} MiB input limit."
                )
        except OSError as e:
            return f'Error: cannot stat image {image_path}: {e}'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(image_path) as img:
                    dimension_error = AnalyzeImageTool._dimension_error(img)
                    if dimension_error:
                        return f"Error: {dimension_error}."
                    img.verify()  # cheap integrity check; does not decode pixels
            return None
        except Exception as e:
            return (f"Error: '{image_path}' is not a valid/decodable image "
                    f"({type(e).__name__}: {e}). Provide a real image file (png/jpg/webp/...).")

    def _load_and_encode_image(self, image_path: str) -> tuple:
        """Load an image, resize if needed, and return (base64_str, mime_type)."""
        if os.path.getsize(image_path) > self.MAX_INPUT_BYTES:
            raise ValueError("image changed and now exceeds the input byte limit")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(image_path) as img:
                dimension_error = self._dimension_error(img)
                if dimension_error:
                    raise ValueError(dimension_error)
                img.load()  # decode only after the repeated dimension check

        # Convert RGBA/palette to RGB for JPEG compatibility
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')

        # Resize if either dimension exceeds the max
        w, h = img.size
        if max(w, h) > self.MAX_IMAGE_DIM:
            scale = self.MAX_IMAGE_DIM / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Encode to JPEG for smaller payload
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return b64, 'image/jpeg'

    @staticmethod
    def _reasoning_effort(prompt: str) -> str:
        """Use low for perception and xhigh when the request needs visual reasoning."""
        if re.search(
                r"\b(reason|solve|diagnos|debug|compare|infer|prove|ambiguous|unclear|"
                r"diagram|chart|code|error|plan)\w*\b", prompt or "", re.IGNORECASE):
            return "xhigh"
        return "low"

    def execute(self, image_path: str, prompt: str, auto_cleanup: bool = True,
                max_tokens: int = 1024, temperature: float = 1.0) -> str:
        if not image_path:
            return "Error: 'image_path' parameter is required."
        if not prompt:
            return "Error: 'prompt' parameter is required."

        abs_image_path = os.path.abspath(image_path)
        if not os.path.exists(abs_image_path):
            return f'Error: Image not found at {abs_image_path}'

        # Fail fast on a bad image before encoding/sending.
        validation_error = self._validate_image(abs_image_path)
        if validation_error:
            return validation_error

        # Runs on the selected Qwen3.8 multimodal primary, whose endpoint main.py
        # exports as AEON_VISION_*. If a text-only model is selected there is no
        # local vision backend -> return a clear, actionable error.
        worker = getattr(self, "worker", None)
        model_config = getattr(worker, "model_config", None)
        if not isinstance(model_config, dict):
            model_config = {}
        primary_base = model_config.get("base_url")
        primary_model = model_config.get("api_model") or model_config.get("model")
        if not (primary_base and primary_model):
            return ("Error: image analysis requires Aeon's Qwen3.8 vision model. The current "
                    f"session does not serve vision. Restart Aeon and select {VISION_MODEL_NAME}.")
        if model_config.get("provider") != "vllm":
            return "Error: image analysis requires the Fleet-backed vLLM provider."
        if primary_model not in VISION_MODEL_NAMES:
            return (f"Error: refusing to send image data to '{primary_model}'. Aeon's only "
                    f"approved vision model is '{VISION_MODEL_NAME}'.")
        try:
            primary_base = validate_loopback_endpoint(primary_base)
        except FleetBackendError as exc:
            return (
                "Error: the active vision endpoint is not an exact Fleet-issued "
                f"loopback endpoint ({exc})."
            )
        vision_url = primary_base.rstrip('/') + '/chat/completions'  # base already ends in /v1
        vision_model = primary_model

        try:
            # Load and encode image
            try:
                b64_image, mime_type = self._load_and_encode_image(abs_image_path)
            except Exception as e:
                return f'Error loading image: {type(e).__name__}: {e}'

            # Build OpenAI-compatible vision request
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:{mime_type};base64,{b64_image}'
                            }
                        },
                        {
                            'type': 'text',
                            'text': prompt
                        }
                    ]
                }
            ]

            # Generation length dominates vision latency. Callers that already
            # have the page text (e.g. the browser's structured snapshot) pass a
            # small max_tokens for a fast, concise visual summary; standalone
            # image analysis keeps a larger budget.
            payload = {
                'model': vision_model,
                'messages': messages,
                'max_tokens': int(max_tokens),
                'temperature': float(temperature),
                'top_p': 0.95,
                'presence_penalty': 0.0,
                'top_k': 20,
                'min_p': 0.0,
                'repetition_penalty': 1.0,
                'reasoning_effort': self._reasoning_effort(prompt),
                'chat_template_kwargs': {
                    'enable_thinking': True,
                    'preserve_thinking': True,
                },
            }

            print(f"Sending request to vision endpoint ({vision_model})...")
            guard = getattr(worker, "compute_guard", None)
            if not callable(guard):
                return "Error: the active vision request has no Fleet ticket guard."
            try:
                guard()
            except Exception as exc:
                detail = str(exc).splitlines()[0][:300] if str(exc) else type(exc).__name__
                return (
                    "Error: Fleet compute changed before the vision request "
                    f"({detail})."
                )
            refreshed = getattr(worker, "model_config", None)
            if (
                not isinstance(refreshed, dict)
                or refreshed.get("provider") != "vllm"
                or (refreshed.get("api_model") or refreshed.get("model"))
                not in VISION_MODEL_NAMES
            ):
                return "Error: the active Fleet vision identity changed before transport."
            try:
                refreshed_base = validate_loopback_endpoint(refreshed.get("base_url"))
            except FleetBackendError as exc:
                return (
                    "Error: the active Fleet vision endpoint changed before transport "
                    f"({exc})."
                )
            vision_url = refreshed_base.rstrip('/') + '/chat/completions'
            resp = requests.post(
                vision_url,
                json=payload,
                headers={
                    "Authorization": "Bearer "
                    + str(refreshed.get("api_key") or "no-key-needed")
                },
                timeout=120,
                stream=True,
                **_LOCAL_HTTP_KWARGS,
            )
            try:
                raw_response = _bounded_response_body(resp)
            finally:
                resp.close()
            response_text = raw_response.decode("utf-8", errors="replace")
            if resp.status_code != 200:
                return (
                    f'Error from vision endpoint (HTTP {resp.status_code}): '
                    f'{response_text[:500]}'
                )

            result = json.loads(raw_response.decode("utf-8"))
            if not isinstance(result, dict):
                return 'Error: Unexpected non-object response from vision endpoint.'
            try:
                answer = result['choices'][0]['message']['content']
            except (KeyError, IndexError):
                return f'Error: Unexpected response format from vision endpoint: {json.dumps(result)[:500]}'

            # Keep the full output, including thinking tags, as requested by the user.
            answer = answer.strip()

            full_output = (
                f"Prompt: {prompt}\n"
                f"Image: {abs_image_path}\n"
                f"Analysis:\n{answer}"
            )
            print(f'{self.C_GREEN}--- Vision Analysis ---')
            print(full_output)
            print(f'-----------------------{self.C_RESET}')
            return full_output

        except Exception as e:
            return self.format_error_message(e, 'analyzing image on the multimodal model',
                                             f'confirming {VISION_MODEL_NAME} is selected')
        # No teardown: the multimodal primary is owned by the session, not this tool.
