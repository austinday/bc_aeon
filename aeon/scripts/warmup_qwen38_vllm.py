#!/usr/bin/env python3
"""Warm the Qwen3.8 kernels and verify one real structured Aeon turn."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import requests

from aeon.core import action_schema as _action_schema
from aeon.core import sampling as _sampling
from aeon.core.action_schema import TURN_FIELDS_REQUIRED, build_turn_schema
from aeon.core.sampling import (
    QWEN_CONTROL_TEMPERATURE,
    QWEN_CONTROL_TOP_K,
    QWEN_CONTROL_TOP_P,
)


MARKER = "AEON_RUNTIME_WARM"
REASON = "Verified runtime warmup marker AEON_RUNTIME_WARM"
SCHEMA = build_turn_schema(["task_complete"])


def _assert_staged_imports() -> None:
    """Refuse warmup when release-critical helpers resolve outside the stage."""

    raw = os.environ.get("AEON_STAGED_SOURCE_ROOT", "")
    if not raw or not raw.startswith("/"):
        raise RuntimeError("warmup has no exact staged source root")
    root = Path(raw).resolve(strict=True)
    for module in (_action_schema, _sampling):
        module_path = Path(str(module.__file__)).resolve(strict=True)
        try:
            module_path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("warmup imported release code outside its stage") from exc


def _vision_data_url() -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (112, 112), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 88, 88), fill=(30, 120, 210))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def warm(base_url: str, model: str, *, include_image: bool = False) -> dict:
    user_content = (
        "Verify the runtime is ready. Return exactly one task_complete action "
        f"with reason '{REASON}', and set intent exactly to {MARKER}. "
        "Its parameters object must contain exactly that one reason key and no other keys. "
        "Keep every required metadata string concise."
    )
    if include_image:
        user_content = [
            {"type": "text", "text": user_content},
            {"type": "image_url", "image_url": {"url": _vision_data_url()}},
        ]
    response = requests.post(
        base_url.rstrip("/") + "/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Aeon's local reasoner. Think privately, then return only "
                        "the schema-constrained final object."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "temperature": QWEN_CONTROL_TEMPERATURE,
            "top_p": QWEN_CONTROL_TOP_P,
            "top_k": QWEN_CONTROL_TOP_K,
            "min_p": 0.0,
            "reasoning_effort": "medium",
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "aeon_runtime_warmup",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
            "seed": 1701,
            "max_tokens": 2048,
        },
        timeout=(15, 240),
    )
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    if len(choices) != 1:
        raise RuntimeError("warmup returned no unique completion")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("warmup returned no post-reasoning content")
    parsed = json.loads(content)
    required = set(TURN_FIELDS_REQUIRED)
    allowed = set(SCHEMA["properties"])
    if not isinstance(parsed, dict) or not required <= set(parsed) <= allowed:
        raise RuntimeError("warmup response violated the Aeon turn envelope")
    actions = parsed.get("actions")
    if (
        parsed.get("intent") != MARKER
        or not isinstance(actions, list)
        or len(actions) != 1
        or actions[0] != {"tool_name": "task_complete", "parameters": {"reason": REASON}}
    ):
        raise RuntimeError("warmup response chose the wrong Aeon action")
    return body.get("usage") or {}


def main(argv=None) -> int:
    _assert_staged_imports()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    text_usage = warm(args.base_url, args.model)
    vision_usage = warm(args.base_url, args.model, include_image=True)
    print(
        "QWEN38_WARMUP_OK "
        f"text_completion_tokens={int(text_usage.get('completion_tokens') or 0)} "
        f"vision_completion_tokens={int(vision_usage.get('completion_tokens') or 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
