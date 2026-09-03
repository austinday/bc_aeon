#!/usr/bin/env python3
"""Warm the Qwen3.8 kernels and verify one real structured Aeon turn."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import stat
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
TOOL_NAME = "task_complete"
CANONICAL_TURN_MAX_BYTES = 512


def _expected_turn() -> dict:
    return {
        "kind": "tool_calls",
        "intent": MARKER,
        "message": "",
        "actions": [
            {
                "tool_name": TOOL_NAME,
                "parameters": {"reason": REASON},
                "goal_refs": [],
            }
        ],
    }


CANONICAL_TURN_JSON = json.dumps(
    _expected_turn(),
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
)
if len(CANONICAL_TURN_JSON.encode("ascii")) > CANONICAL_TURN_MAX_BYTES:
    raise RuntimeError("canonical warmup turn exceeds its release bound")

SCHEMA = build_turn_schema([TOOL_NAME])
FAILURE_SCHEMA_VERSION = 1
FAILURE_MAX_BYTES = 256
_TURN_FAILURE_CODES = frozenset(
    {
        "input_build",
        "http_timeout",
        "http_request",
        "http_status",
        "response_json",
        "completion_count",
        "completion_content",
        "turn_json",
        "turn_not_object",
        "turn_missing_required",
        "turn_unexpected_fields",
        "turn_action",
        "internal",
    }
)
FAILURE_CODES_BY_STAGE = {
    "preflight": frozenset({"staged_imports", "internal"}),
    "text": _TURN_FAILURE_CODES,
    "vision": _TURN_FAILURE_CODES,
    # The caller uses this one code when the wrapper/interpreter did not return
    # a valid diagnostic. The warmup process itself never emits this stage.
    "runner": frozenset(
        {
            "exec_error",
            "invalid_diagnostic",
            "result_mismatch",
            "timeout",
        }
    ),
}


class WarmupFailure(RuntimeError):
    """A release-safe warmup failure with no response or identity payload."""

    def __init__(self, stage: str, code: str, message: str) -> None:
        if code not in FAILURE_CODES_BY_STAGE.get(stage, frozenset()):
            raise ValueError("warmup failure code is outside the release contract")
        self.stage = stage
        self.code = code
        super().__init__(message)


def _emit_failure(descriptor: int, stage: str, code: str) -> bool:
    """Write the complete bounded failure wire format to one inherited receipt."""

    if code not in FAILURE_CODES_BY_STAGE.get(stage, frozenset()):
        stage, code = "runner", "invalid_diagnostic"
    if isinstance(descriptor, bool) or not isinstance(descriptor, int) or descriptor < 3:
        return False
    try:
        metadata = os.fstat(descriptor)
    except OSError:
        return False
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or metadata.st_nlink > 1
        or metadata.st_size != 0
    ):
        return False
    payload = (
        json.dumps(
            {
                "schema_version": FAILURE_SCHEMA_VERSION,
                "stage": stage,
                "code": code,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")
    if len(payload) > FAILURE_MAX_BYTES:
        return False
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                return False
            written += count
        os.fsync(descriptor)
    except OSError:
        return False
    return True


def _assert_staged_imports() -> None:
    """Refuse warmup when release-critical helpers resolve outside the stage."""

    raw = os.environ.get("AEON_STAGED_SOURCE_ROOT", "")
    if not raw or not raw.startswith("/"):
        raise WarmupFailure(
            "preflight", "staged_imports", "warmup has no exact staged source root"
        )
    try:
        root = Path(raw).resolve(strict=True)
    except OSError as exc:
        raise WarmupFailure(
            "preflight", "staged_imports", "warmup staged source is unavailable"
        ) from exc
    for module in (_action_schema, _sampling):
        try:
            module_path = Path(str(module.__file__)).resolve(strict=True)
            module_path.relative_to(root)
        except (OSError, ValueError) as exc:
            raise WarmupFailure(
                "preflight",
                "staged_imports",
                "warmup imported release code outside its stage",
            ) from exc


def _vision_data_url() -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (112, 112), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 88, 88), fill=(30, 120, 210))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def warm(base_url: str, model: str, *, include_image: bool = False) -> dict:
    stage = "vision" if include_image else "text"
    user_content = (
        "Verify the runtime is ready. Return only this exact JSON object with "
        "no additional fields or prose:\n"
        f"{CANONICAL_TURN_JSON}"
    )
    if include_image:
        try:
            user_content = [
                {"type": "text", "text": user_content},
                {"type": "image_url", "image_url": {"url": _vision_data_url()}},
            ]
        except Exception as exc:
            raise WarmupFailure(
                stage, "input_build", "warmup input construction failed"
            ) from exc
    try:
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
            allow_redirects=False,
            proxies={"http": "", "https": ""},
        )
    except requests.Timeout as exc:
        raise WarmupFailure(stage, "http_timeout", "warmup request timed out") from exc
    except requests.RequestException as exc:
        raise WarmupFailure(stage, "http_request", "warmup request failed") from exc
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WarmupFailure(stage, "http_status", "warmup HTTP status failed") from exc
    try:
        body = response.json()
    except (TypeError, ValueError) as exc:
        raise WarmupFailure(stage, "response_json", "warmup response was not JSON") from exc
    if not isinstance(body, dict):
        raise WarmupFailure(stage, "response_json", "warmup response was not an object")
    choices = body.get("choices") or []
    if len(choices) != 1:
        raise WarmupFailure(
            stage, "completion_count", "warmup returned no unique completion"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise WarmupFailure(
            stage, "completion_content", "warmup returned no completion object"
        )
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        raise WarmupFailure(
            stage, "completion_content", "warmup returned no completion message"
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise WarmupFailure(
            stage,
            "completion_content",
            "warmup returned no post-reasoning content",
        )
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WarmupFailure(
            stage, "turn_json", "warmup turn content was not JSON"
        ) from exc
    required = set(TURN_FIELDS_REQUIRED)
    allowed = set(SCHEMA["properties"])
    if not isinstance(parsed, dict):
        raise WarmupFailure(
            stage,
            "turn_not_object",
            "warmup turn was not an object",
        )
    fields = set(parsed)
    if not required <= fields:
        raise WarmupFailure(
            stage,
            "turn_missing_required",
            "warmup turn omitted required fields",
        )
    if not fields <= allowed:
        raise WarmupFailure(
            stage,
            "turn_unexpected_fields",
            "warmup turn contained unexpected fields",
        )
    if parsed != _expected_turn():
        raise WarmupFailure(
            stage, "turn_action", "warmup response chose the wrong Aeon action"
        )
    usage = body.get("usage") or {}
    return usage if isinstance(usage, dict) else {}


def _completion_tokens(usage: dict) -> int:
    value = usage.get("completion_tokens")
    if type(value) is not int or not 0 <= value <= 10_000_000:
        return 0
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--failure-fd", required=True, type=int)
    args = parser.parse_args(argv)
    stage = "preflight"
    try:
        _assert_staged_imports()
        stage = "text"
        text_usage = warm(args.base_url, args.model)
        stage = "vision"
        vision_usage = warm(args.base_url, args.model, include_image=True)
    except WarmupFailure as exc:
        _emit_failure(args.failure_fd, exc.stage, exc.code)
        return 1
    except Exception:
        _emit_failure(args.failure_fd, stage, "internal")
        return 1
    print(
        "QWEN38_WARMUP_OK "
        f"text_completion_tokens={_completion_tokens(text_usage)} "
        f"vision_completion_tokens={_completion_tokens(vision_usage)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
