"""Conservative parsing for benchmark-only authoritative model usage.

This module never estimates token counts and never stores prompt or completion
text. It accepts only internally consistent counters returned by the model
transport itself. Missing, partial, conflicting, malformed, or oversized usage
evidence remains unknown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Mapping

from aeon.core.benchmark_receipt import ModelCallReceipt


MAX_TELEMETRY_JSON_BYTES = 8 * 1024 * 1024
MAX_TELEMETRY_SSE_EVENT_BYTES = 1024 * 1024
_MISSING = object()


@dataclass(frozen=True)
class AuthoritativeTokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name, _MISSING)
    try:
        return getattr(value, name, _MISSING)
    except Exception:
        return _MISSING


def _aliased_counter(value: object, primary: str, alias: str) -> int | None:
    primary_value = _field(value, primary)
    alias_value = _field(value, alias)
    present = tuple(item for item in (primary_value, alias_value) if item is not _MISSING)
    if not present:
        return None
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not 0 <= item <= 1_000_000_000
        for item in present
    ):
        return None
    if len(set(present)) != 1:
        return None
    return present[0]


def authoritative_token_usage(usage: object) -> AuthoritativeTokenUsage | None:
    """Return exact OpenAI-compatible usage or ``None``; never derive counters."""

    if usage is None:
        return None
    prompt = _aliased_counter(usage, "prompt_tokens", "input_tokens")
    completion = _aliased_counter(usage, "completion_tokens", "output_tokens")
    total = _field(usage, "total_tokens")
    if (
        prompt is None
        or completion is None
        or isinstance(total, bool)
        or not isinstance(total, int)
        or not 0 <= total <= 1_000_000_000
        or total != prompt + completion
    ):
        return None
    return AuthoritativeTokenUsage(prompt, completion, total)


class UsageAccumulator:
    """Accept exactly one consistent usage value across streamed chunks."""

    def __init__(self) -> None:
        self._usage: AuthoritativeTokenUsage | None = None
        self._invalid = False

    def observe(self, usage: object) -> None:
        if usage is None:
            return
        parsed = authoritative_token_usage(usage)
        if parsed is None:
            self._invalid = True
            return
        if self._usage is not None and parsed != self._usage:
            self._invalid = True
            return
        self._usage = parsed

    @property
    def result(self) -> AuthoritativeTokenUsage | None:
        return None if self._invalid else self._usage


class ResponseUsageCapture:
    """Incrementally inspect a proxied JSON or SSE body for exact usage only."""

    def __init__(self, content_type: str) -> None:
        self._is_sse = str(content_type).lower().startswith("text/event-stream")
        self._json_body = bytearray()
        self._line_buffer = bytearray()
        self._event_data: list[bytes] = []
        self._event_size = 0
        self._invalid = False
        self._usage = UsageAccumulator()

    def feed(self, chunk: bytes) -> None:
        if self._invalid or not chunk:
            return
        if not self._is_sse:
            if len(self._json_body) + len(chunk) > MAX_TELEMETRY_JSON_BYTES:
                self._invalid = True
                self._json_body.clear()
                return
            self._json_body.extend(chunk)
            return
        self._line_buffer.extend(chunk)
        if len(self._line_buffer) > MAX_TELEMETRY_SSE_EVENT_BYTES:
            self._invalid = True
            self._line_buffer.clear()
            return
        while b"\n" in self._line_buffer:
            raw_line, _, tail = self._line_buffer.partition(b"\n")
            self._line_buffer = bytearray(tail)
            self._consume_sse_line(raw_line.rstrip(b"\r"))
            if self._invalid:
                return

    def _consume_sse_line(self, line: bytes) -> None:
        if not line:
            self._consume_sse_event()
            return
        if not line.startswith(b"data:"):
            return
        data = line[5:]
        if data.startswith(b" "):
            data = data[1:]
        self._event_size += len(data)
        if self._event_size > MAX_TELEMETRY_SSE_EVENT_BYTES:
            self._invalid = True
            self._event_data.clear()
            return
        self._event_data.append(data)

    def _consume_sse_event(self) -> None:
        if not self._event_data:
            self._event_size = 0
            return
        payload = b"\n".join(self._event_data)
        self._event_data.clear()
        self._event_size = 0
        if payload.strip() == b"[DONE]":
            return
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            self._invalid = True
            return
        if not isinstance(document, Mapping):
            self._invalid = True
            return
        if document.get("usage") is not None:
            self._usage.observe(document.get("usage"))

    def finish(self) -> AuthoritativeTokenUsage | None:
        if self._invalid:
            return None
        if self._is_sse:
            if self._line_buffer:
                self._consume_sse_line(bytes(self._line_buffer).rstrip(b"\r"))
                self._line_buffer.clear()
            self._consume_sse_event()
            return None if self._invalid else self._usage.result
        try:
            document = json.loads(bytes(self._json_body).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(document, Mapping):
            return None
        self._usage.observe(document.get("usage"))
        return self._usage.result


def summarize_model_calls(
    receipts: Iterable[object], *, expected_sources: Iterable[str]
) -> dict[str, int | None]:
    """Aggregate one decoded trace without undercounting failed transport work."""

    model_receipts = tuple(
        item for item in receipts if isinstance(item, ModelCallReceipt)
    )
    starts = tuple(item for item in model_receipts if item.phase == "started")
    allowed_sources = frozenset(expected_sources)
    if (
        not starts
        or not allowed_sources
        or any(item.source not in allowed_sources for item in model_receipts)
    ):
        return {
            "model_call_count": None,
            "prompt_tokens": None,
            "peak_prompt_tokens": None,
            "context_tokens": None,
            "completion_tokens": None,
        }
    finishes = {
        item.call_sequence: item
        for item in model_receipts
        if item.phase == "finished"
    }
    complete = len(finishes) == len(starts)
    finished_calls = tuple(finishes.values())
    # A failed or interrupted transport may have consumed provider tokens before
    # its error surfaced. Since that usage is not authoritative, whole-task token
    # totals must remain unknown instead of summing only the successful retries.
    token_complete = complete and all(
        item.outcome == "succeeded" and item.usage_complete
        for item in finished_calls
    )
    return {
        "model_call_count": len(starts),
        "prompt_tokens": (
            sum(int(item.prompt_tokens) for item in finished_calls)
            if token_complete
            else None
        ),
        "peak_prompt_tokens": (
            max(int(item.prompt_tokens) for item in finished_calls)
            if token_complete
            else None
        ),
        # The public field uses the provider's authoritative total/context count.
        "context_tokens": (
            sum(int(item.total_tokens) for item in finished_calls)
            if token_complete
            else None
        ),
        "completion_tokens": (
            sum(int(item.completion_tokens) for item in finished_calls)
            if token_complete
            else None
        ),
    }


__all__ = (
    "AuthoritativeTokenUsage",
    "ResponseUsageCapture",
    "UsageAccumulator",
    "authoritative_token_usage",
    "summarize_model_calls",
)
