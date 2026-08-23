import os
import json
import time
import base64
import requests
import subprocess
import re
from PIL import Image
import io
from .base import BaseTool
from ..core.prompts import TOOL_DESC_ANALYZE_IMAGE
from ..core.model_catalog import VISION_MODEL_NAME


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
            if os.path.getsize(image_path) == 0:
                return f'Error: Image file is empty (0 bytes): {image_path}'
        except OSError as e:
            return f'Error: cannot stat image {image_path}: {e}'
        try:
            with Image.open(image_path) as img:
                img.verify()  # cheap integrity check; does not decode pixels
            return None
        except Exception as e:
            return (f"Error: '{image_path}' is not a valid/decodable image "
                    f"({type(e).__name__}: {e}). Provide a real image file (png/jpg/webp/...).")

    def _load_and_encode_image(self, image_path: str) -> tuple:
        """Load an image, resize if needed, and return (base64_str, mime_type)."""
        with Image.open(image_path) as img:
            img.load()  # decode now so the file handle can be released

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
        primary_base = os.environ.get('AEON_VISION_BASE_URL')
        primary_model = os.environ.get('AEON_VISION_MODEL')
        if not (primary_base and primary_model):
            return ("Error: image analysis requires Aeon's Qwen3.8 vision model. The current "
                    f"session does not serve vision. Restart Aeon and select {VISION_MODEL_NAME}.")
        if primary_model != VISION_MODEL_NAME:
            return (f"Error: refusing to send image data to '{primary_model}'. Aeon's only "
                    f"approved vision model is '{VISION_MODEL_NAME}'.")
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
            resp = requests.post(
                vision_url,
                json=payload,
                timeout=120
            )

            if resp.status_code != 200:
                return f'Error from vision endpoint (HTTP {resp.status_code}): {resp.text[:500]}'

            result = resp.json()
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
