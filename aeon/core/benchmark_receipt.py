"""Authenticated, typed evidence emitted by benchmarked Aeon components.

The model's response and harness transcript are untrusted for scoring.  During
an owner benchmark only, a reviewed tool or model transport can append a small
HMAC-bound receipt to an executor-created private file.  The executor validates
that receipt independently after the harness exits.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import threading
import fcntl
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping


CAPABILITY_RECEIPT_PATH_ENV = "AEON_BENCHMARK_GPU_CAPABILITY_RECEIPT_PATH"
CAPABILITY_RECEIPT_KEY_ENV = "AEON_BENCHMARK_GPU_CAPABILITY_RECEIPT_KEY"
CAPABILITY_RECEIPT_SCHEMA_VERSION = 1
CAPABILITY_RECEIPT_TYPE = "fleet_wait_capability"
TOOL_CALL_RECEIPT_TYPE = "tool_call"
SCENARIO_EFFECT_RECEIPT_TYPE = "scenario_effect"
MODEL_CALL_RECEIPT_TYPE = "model_call"
TRACE_RUN_ID_ENV = "AEON_BENCHMARK_TRACE_RUN_ID"
TRACE_CASE_ID_ENV = "AEON_BENCHMARK_TRACE_CASE_ID"
TRACE_REPETITION_ENV = "AEON_BENCHMARK_TRACE_REPETITION"
TRACE_NONCE_ENV = "AEON_BENCHMARK_TRACE_NONCE"
# A bad model may use the full bounded 12-step/15-call protocol. Each simulator
# call has both a proposal and an effect, so retain enough private evidence to
# score that behavior instead of misclassifying a bounded call storm as broken
# infrastructure.
MAX_CAPABILITY_RECEIPT_BYTES = 128 * 1024
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_RUN_ID_RE = re.compile(r"^run-[0-9a-f]{32}$")
_CASE_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,99}$")
_NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TRACE_LOCK = threading.RLock()
_MODEL_CALL_OUTCOMES = frozenset(
    {"pending", "succeeded", "http_error", "transport_error", "cancelled", "failed"}
)


class BenchmarkTelemetryError(RuntimeError):
    """A requested benchmark trace could not durably bracket a model call."""


@dataclass(frozen=True)
class FleetWaitCapabilityReceipt:
    schema_version: int
    receipt_type: str
    tool_name: str
    status: str
    submission_boundary: str
    unavailable_compute_is_durable_wait: bool
    general_model_build_available: bool
    eligible_recipe_count: int
    capability_payload_sha256: str


@dataclass(frozen=True)
class ToolCallReceipt:
    """Content-minimal proof of a model-requested reviewed tool call."""

    schema_version: int
    receipt_type: str
    tool_name: str
    arguments_sha256: str
    run_id: str
    case_id: str
    repetition: int
    trace_nonce: str
    sequence: int
    phase: str


@dataclass(frozen=True)
class ScenarioEffectReceipt:
    """Post-effect evidence from the closed deterministic benchmark simulator."""

    schema_version: int
    receipt_type: str
    run_id: str
    case_id: str
    repetition: int
    trace_nonce: str
    effect_sequence: int
    operation: str
    accepted: bool
    effect_code: str
    arguments_sha256: str
    state_before_sha256: str
    state_after_sha256: str
    effect_sha256: str
    virtual_start_ms: int
    virtual_end_ms: int


@dataclass(frozen=True)
class ModelCallReceipt:
    """Content-free proof of one primary-agent model transport attempt.

    A ``started`` event is durable before the transport is invoked, so retries and
    interrupted calls are still counted. A matching ``finished`` event carries
    provider-reported token usage only when all three OpenAI-compatible counters
    were present and internally consistent. Prompts and completions are never
    written to benchmark evidence.
    """

    schema_version: int
    receipt_type: str
    run_id: str
    case_id: str
    repetition: int
    trace_nonce: str
    event_sequence: int
    call_sequence: int
    call_nonce: str
    source: str
    phase: str
    outcome: str
    usage_complete: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class ModelCallHandle:
    """Opaque in-process identity returned after a model-call start is durable."""

    run_id: str
    case_id: str
    repetition: int
    trace_nonce: str
    call_sequence: int
    call_nonce: str
    source: str


CapabilityReceipt = (
    FleetWaitCapabilityReceipt
    | ToolCallReceipt
    | ScenarioEffectReceipt
    | ModelCallReceipt
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _typed_receipt(document: Mapping[str, object]) -> FleetWaitCapabilityReceipt:
    if not isinstance(document, Mapping):
        raise ValueError("Fleet capability result is not structured")
    recipes = document.get("recipes")
    if not isinstance(recipes, list) or len(recipes) > 64:
        raise ValueError("Fleet capability recipes are malformed")
    for recipe in recipes:
        if not isinstance(recipe, Mapping) or set(recipe) != {
            "recipe_id",
            "profile_id",
            "purpose",
            "durable_wait",
        }:
            raise ValueError("Fleet capability recipe is malformed")
        if (
            not all(
                isinstance(recipe.get(field), str) and bool(recipe.get(field))
                for field in ("recipe_id", "profile_id", "purpose")
            )
            or recipe.get("durable_wait") is not True
        ):
            raise ValueError("Fleet capability recipe is not durable and typed")
    if (
        document.get("status") != "ok"
        or document.get("general_model_build_available") is not False
        or document.get("submission_boundary") != "reviewed_recipe_only"
        or document.get("unavailable_compute_is_durable_wait") is not True
    ):
        raise ValueError("Fleet capability boundary is not the reviewed durable path")
    return FleetWaitCapabilityReceipt(
        schema_version=CAPABILITY_RECEIPT_SCHEMA_VERSION,
        receipt_type=CAPABILITY_RECEIPT_TYPE,
        tool_name="fleet_batch_capabilities",
        status="ok",
        submission_boundary="reviewed_recipe_only",
        unavailable_compute_is_durable_wait=True,
        general_model_build_available=False,
        eligible_recipe_count=len(recipes),
        capability_payload_sha256=hashlib.sha256(
            _canonical_json(dict(document))
        ).hexdigest(),
    )


def tool_arguments_sha256(arguments: Mapping[str, object]) -> str:
    """Bind exact JSON-compatible arguments without exposing them in evidence."""

    if not isinstance(arguments, Mapping):
        raise ValueError("benchmark tool arguments are not structured")
    return hashlib.sha256(_canonical_json(dict(arguments))).hexdigest()


def _trace_context() -> tuple[str, str, int, str] | None:
    run_id = os.environ.get(TRACE_RUN_ID_ENV, "")
    case_id = os.environ.get(TRACE_CASE_ID_ENV, "")
    repetition_raw = os.environ.get(TRACE_REPETITION_ENV, "")
    trace_nonce = os.environ.get(TRACE_NONCE_ENV, "")
    try:
        repetition = int(repetition_raw)
    except (TypeError, ValueError):
        return None
    if (
        _RUN_ID_RE.fullmatch(run_id) is None
        or _CASE_ID_RE.fullmatch(case_id) is None
        or not 1 <= repetition <= 20
        or _NONCE_RE.fullmatch(trace_nonce) is None
    ):
        return None
    return run_id, case_id, repetition, trace_nonce


def benchmark_trace_requested() -> bool:
    """Return whether this process was launched with any benchmark trace state."""

    return any(
        name in os.environ
        for name in (
            CAPABILITY_RECEIPT_PATH_ENV,
            CAPABILITY_RECEIPT_KEY_ENV,
            TRACE_RUN_ID_ENV,
            TRACE_CASE_ID_ENV,
            TRACE_REPETITION_ENV,
            TRACE_NONCE_ENV,
        )
    )


def _receipt_envelope(payload: Mapping[str, object], key: str) -> bytes:
    encoded = _canonical_json(dict(payload))
    return _canonical_json(
        {
            "payload": dict(payload),
            "hmac_sha256": hmac.new(
                bytes.fromhex(key), encoded, hashlib.sha256
            ).hexdigest(),
        }
    ) + b"\n"


def _read_locked_receipts(descriptor: int) -> bytes | None:
    metadata = os.fstat(descriptor)
    if metadata.st_size > MAX_CAPABILITY_RECEIPT_BYTES:
        return None
    chunks: list[bytes] = []
    offset = 0
    remaining = metadata.st_size
    while remaining:
        chunk = os.pread(descriptor, min(remaining, 4096), offset)
        if not chunk:
            return None
        chunks.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _stream_transaction(
    builder: Callable[
        [tuple[CapabilityReceipt, ...]],
        Mapping[str, object],
    ],
    *,
    expected_device: int | None = None,
    expected_inode: int | None = None,
) -> bool:
    """Validate, extend, and fsync one receipt stream under an OS file lock.

    Deriving sequences from the durable stream keeps evidence valid if OpenCode
    restarts its MCP child during one harness invocation.  The optional identity
    check binds scenario state to the executor-created object, not merely a path.
    """

    raw_path = os.environ.get(CAPABILITY_RECEIPT_PATH_ENV, "")
    raw_key = os.environ.get(CAPABILITY_RECEIPT_KEY_ENV, "")
    path = Path(raw_path)
    if not raw_path or not path.is_absolute() or _KEY_RE.fullmatch(raw_key) is None:
        return False
    context = _trace_context()
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDWR
            | os.O_APPEND
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
        )
        with _TRACE_LOCK:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or (
                    expected_device is not None
                    and metadata.st_dev != expected_device
                )
                or (
                    expected_inode is not None
                    and metadata.st_ino != expected_inode
                )
            ):
                return False
            current = _read_locked_receipts(descriptor)
            if current is None:
                return False
            kwargs: dict[str, object] = {"key": raw_key}
            if context is not None:
                kwargs.update(
                    {
                        "run_id": context[0],
                        "case_id": context[1],
                        "repetition": context[2],
                        "trace_nonce": context[3],
                    }
                )
            records = (
                decode_capability_receipts(current, **kwargs) if current else ()
            )
            if current and not records:
                return False
            payload = dict(builder(records))
            envelope = _receipt_envelope(payload, raw_key)
            if (
                len(envelope) > 4096
                or len(current) + len(envelope) > MAX_CAPABILITY_RECEIPT_BYTES
            ):
                return False
            # Validate the complete candidate before publishing one byte.  This
            # catches a duplicate/gapped sequence or a malformed typed payload.
            if not decode_capability_receipts(current + envelope, **kwargs):
                return False
            written = 0
            while written < len(envelope):
                written += os.write(descriptor, envelope[written:])
            os.fsync(descriptor)
            return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(descriptor)


def _append_receipt(payload: Mapping[str, object]) -> bool:
    return _stream_transaction(lambda _records: payload)


def emit_tool_call_receipt(tool_name: str, arguments: Mapping[str, object]) -> None:
    """Record a proposed tool call only when benchmark instrumentation is active."""

    if not isinstance(tool_name, str) or not _TOOL_NAME_RE.fullmatch(tool_name):
        return
    try:
        context = _trace_context()
        if context is None:
            return
        run_id, case_id, repetition, trace_nonce = context
        recorded_arguments = dict(arguments)
        # The benchmark workflow's Python signature supplies these two defaults
        # before the post-effect receipt is built. Normalize the proposal the
        # same way so the executor can cryptographically correlate a requested
        # operation with its effect even when the model omitted optional fields.
        if tool_name == "benchmark_workflow":
            recorded_arguments.setdefault("reference_ids", [])
            recorded_arguments.setdefault("branch", "")

        def build(
            records: tuple[CapabilityReceipt, ...],
        ) -> Mapping[str, object]:
            sequence = 1 + sum(isinstance(item, ToolCallReceipt) for item in records)
            receipt = ToolCallReceipt(
                schema_version=CAPABILITY_RECEIPT_SCHEMA_VERSION,
                receipt_type=TOOL_CALL_RECEIPT_TYPE,
                tool_name=tool_name,
                arguments_sha256=tool_arguments_sha256(recorded_arguments),
                run_id=run_id,
                case_id=case_id,
                repetition=repetition,
                trace_nonce=trace_nonce,
                sequence=sequence,
                phase="proposed",
            )
            return asdict(receipt)

        _stream_transaction(build)
    except (OSError, TypeError, ValueError):
        return


def begin_model_call(source: str) -> ModelCallHandle | None:
    """Durably count one primary-agent transport invocation during a benchmark.

    Normal harness runs have no benchmark trace capability, so this is a cheap
    no-op. The returned handle exists only after the authenticated ``started``
    event has been appended and fsynced.
    """

    if not benchmark_trace_requested():
        return None
    if not isinstance(source, str) or _EVENT_NAME_RE.fullmatch(source) is None:
        raise BenchmarkTelemetryError("benchmark model-call source is invalid")
    context = _trace_context()
    if context is None:
        raise BenchmarkTelemetryError("benchmark model-call trace context is invalid")
    run_id, case_id, repetition, trace_nonce = context
    published: list[ModelCallReceipt] = []
    call_nonce = os.urandom(32).hex()

    def build(records: tuple[CapabilityReceipt, ...]) -> Mapping[str, object]:
        model_records = tuple(
            item for item in records if isinstance(item, ModelCallReceipt)
        )
        receipt = ModelCallReceipt(
            schema_version=CAPABILITY_RECEIPT_SCHEMA_VERSION,
            receipt_type=MODEL_CALL_RECEIPT_TYPE,
            run_id=run_id,
            case_id=case_id,
            repetition=repetition,
            trace_nonce=trace_nonce,
            event_sequence=len(model_records) + 1,
            call_sequence=(
                1 + sum(item.phase == "started" for item in model_records)
            ),
            call_nonce=call_nonce,
            source=source,
            phase="started",
            outcome="pending",
            usage_complete=False,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )
        published.append(receipt)
        return asdict(receipt)

    try:
        if not _stream_transaction(build) or len(published) != 1:
            raise BenchmarkTelemetryError(
                "benchmark model-call start could not be made durable"
            )
    except (OSError, TypeError, ValueError):
        raise BenchmarkTelemetryError(
            "benchmark model-call start could not be made durable"
        ) from None
    receipt = published[0]
    return ModelCallHandle(
        run_id=receipt.run_id,
        case_id=receipt.case_id,
        repetition=receipt.repetition,
        trace_nonce=receipt.trace_nonce,
        call_sequence=receipt.call_sequence,
        call_nonce=receipt.call_nonce,
        source=receipt.source,
    )


def _authoritative_token_usage(
    prompt_tokens: object,
    completion_tokens: object,
    total_tokens: object,
) -> tuple[int, int, int] | None:
    values = (prompt_tokens, completion_tokens, total_tokens)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 1_000_000_000
        for value in values
    ):
        return None
    prompt, completion, total = values
    if total != prompt + completion:
        return None
    return prompt, completion, total


def finish_model_call(
    handle: ModelCallHandle | None,
    *,
    outcome: str,
    prompt_tokens: object = None,
    completion_tokens: object = None,
    total_tokens: object = None,
) -> bool:
    """Append the terminal event for ``handle`` without affecting model behavior."""

    if handle is None or not isinstance(handle, ModelCallHandle):
        return False
    if outcome not in _MODEL_CALL_OUTCOMES - {"pending"}:
        return False
    context = _trace_context()
    expected_context = (
        handle.run_id,
        handle.case_id,
        handle.repetition,
        handle.trace_nonce,
    )
    if context != expected_context:
        return False
    usage = (
        _authoritative_token_usage(
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )
        if outcome == "succeeded"
        else None
    )

    def build(records: tuple[CapabilityReceipt, ...]) -> Mapping[str, object]:
        model_records = tuple(
            item for item in records if isinstance(item, ModelCallReceipt)
        )
        starts = tuple(
            item
            for item in model_records
            if item.phase == "started"
            and item.call_sequence == handle.call_sequence
        )
        finishes = tuple(
            item
            for item in model_records
            if item.phase == "finished"
            and item.call_sequence == handle.call_sequence
        )
        if (
            len(starts) != 1
            or finishes
            or starts[0].call_nonce != handle.call_nonce
            or starts[0].source != handle.source
        ):
            raise ValueError("model call handle does not name one unfinished call")
        receipt = ModelCallReceipt(
            schema_version=CAPABILITY_RECEIPT_SCHEMA_VERSION,
            receipt_type=MODEL_CALL_RECEIPT_TYPE,
            run_id=handle.run_id,
            case_id=handle.case_id,
            repetition=handle.repetition,
            trace_nonce=handle.trace_nonce,
            event_sequence=len(model_records) + 1,
            call_sequence=handle.call_sequence,
            call_nonce=handle.call_nonce,
            source=handle.source,
            phase="finished",
            outcome=outcome,
            usage_complete=usage is not None,
            prompt_tokens=usage[0] if usage is not None else None,
            completion_tokens=usage[1] if usage is not None else None,
            total_tokens=usage[2] if usage is not None else None,
        )
        return asdict(receipt)

    try:
        return _stream_transaction(build)
    except (OSError, TypeError, ValueError):
        return False


def append_scenario_effect_receipt(
    builder: Callable[[tuple[ScenarioEffectReceipt, ...]], ScenarioEffectReceipt],
    *,
    expected_device: int,
    expected_inode: int,
) -> ScenarioEffectReceipt | None:
    """Atomically replay and append one post-effect simulator receipt."""

    published: list[ScenarioEffectReceipt] = []

    def build(
        records: tuple[CapabilityReceipt, ...],
    ) -> Mapping[str, object]:
        effects = tuple(
            item for item in records if isinstance(item, ScenarioEffectReceipt)
        )
        receipt = builder(effects)
        if not isinstance(receipt, ScenarioEffectReceipt):
            raise TypeError("scenario receipt builder returned the wrong type")
        published.append(receipt)
        return asdict(receipt)

    if not _stream_transaction(
        build,
        expected_device=expected_device,
        expected_inode=expected_inode,
    ):
        return None
    return published[0] if len(published) == 1 else None


def emit_fleet_wait_capability_receipt(document: Mapping[str, object]) -> None:
    """Append one authenticated receipt when a benchmark executor requested it.

    This observational side channel grants no Fleet authority.  Missing or
    malformed benchmark configuration is ignored so normal tool behavior never
    depends on benchmark instrumentation.
    """

    raw_path = os.environ.get(CAPABILITY_RECEIPT_PATH_ENV, "")
    raw_key = os.environ.get(CAPABILITY_RECEIPT_KEY_ENV, "")
    if not raw_path or not _KEY_RE.fullmatch(raw_key):
        return
    path = Path(raw_path)
    if not path.is_absolute():
        return
    try:
        receipt = _typed_receipt(document)
        _append_receipt(asdict(receipt))
    except (OSError, TypeError, ValueError):
        return


def decode_capability_receipts(
    payload: bytes,
    *,
    key: str,
    run_id: str | None = None,
    case_id: str | None = None,
    repetition: int | None = None,
    trace_nonce: str | None = None,
) -> tuple[
    CapabilityReceipt, ...
]:
    """Validate a bounded receipt stream and return only typed receipts."""

    if not _KEY_RE.fullmatch(str(key or "")):
        return ()
    if not payload or len(payload) > MAX_CAPABILITY_RECEIPT_BYTES:
        return ()
    result: list[CapabilityReceipt] = []
    expected_sequence = 1
    expected_effect_sequence = 1
    expected_model_event_sequence = 1
    expected_model_call_sequence = 1
    seen_fleet_capability = False
    observed_context: tuple[str, str, int, str] | None = None
    for line in payload.splitlines():
        if not line or len(line) > 4096:
            return ()
        try:
            envelope = json.loads(line)
        except (UnicodeError, json.JSONDecodeError):
            return ()
        if not isinstance(envelope, dict) or set(envelope) != {
            "payload",
            "hmac_sha256",
        }:
            return ()
        item = envelope.get("payload")
        supplied_mac = envelope.get("hmac_sha256")
        if not isinstance(item, dict) or not isinstance(supplied_mac, str):
            return ()
        encoded = _canonical_json(item)
        expected_mac = hmac.new(
            bytes.fromhex(key), encoded, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(supplied_mac, expected_mac):
            return ()
        receipt_type = item.get("receipt_type")
        if receipt_type == CAPABILITY_RECEIPT_TYPE:
            if seen_fleet_capability:
                return ()
            seen_fleet_capability = True
            try:
                receipt = FleetWaitCapabilityReceipt(**item)
            except TypeError:
                return ()
            if (
                asdict(receipt) != item
                or isinstance(receipt.schema_version, bool)
                or not isinstance(receipt.schema_version, int)
                or receipt.schema_version != CAPABILITY_RECEIPT_SCHEMA_VERSION
                or receipt.tool_name != "fleet_batch_capabilities"
                or receipt.status != "ok"
                or receipt.submission_boundary != "reviewed_recipe_only"
                or receipt.unavailable_compute_is_durable_wait is not True
                or receipt.general_model_build_available is not False
                or isinstance(receipt.eligible_recipe_count, bool)
                or not isinstance(receipt.eligible_recipe_count, int)
                or not 0 <= receipt.eligible_recipe_count <= 64
                or not isinstance(receipt.capability_payload_sha256, str)
                or _DIGEST_RE.fullmatch(receipt.capability_payload_sha256) is None
            ):
                return ()
        elif receipt_type == TOOL_CALL_RECEIPT_TYPE:
            try:
                receipt = ToolCallReceipt(**item)
            except TypeError:
                return ()
            if (
                asdict(receipt) != item
                or isinstance(receipt.schema_version, bool)
                or receipt.schema_version != CAPABILITY_RECEIPT_SCHEMA_VERSION
                or not isinstance(receipt.tool_name, str)
                or _TOOL_NAME_RE.fullmatch(receipt.tool_name) is None
                or not isinstance(receipt.arguments_sha256, str)
                or _DIGEST_RE.fullmatch(receipt.arguments_sha256) is None
                or not isinstance(receipt.run_id, str)
                or _RUN_ID_RE.fullmatch(receipt.run_id) is None
                or not isinstance(receipt.case_id, str)
                or _CASE_ID_RE.fullmatch(receipt.case_id) is None
                or isinstance(receipt.repetition, bool)
                or not isinstance(receipt.repetition, int)
                or not 1 <= receipt.repetition <= 20
                or not isinstance(receipt.trace_nonce, str)
                or _NONCE_RE.fullmatch(receipt.trace_nonce) is None
                or isinstance(receipt.sequence, bool)
                or not isinstance(receipt.sequence, int)
                or receipt.sequence != expected_sequence
                or receipt.phase != "proposed"
                or (run_id is not None and receipt.run_id != run_id)
                or (case_id is not None and receipt.case_id != case_id)
                or (repetition is not None and receipt.repetition != repetition)
                or (trace_nonce is not None and receipt.trace_nonce != trace_nonce)
            ):
                return ()
            expected_sequence += 1
            context = (
                receipt.run_id,
                receipt.case_id,
                receipt.repetition,
                receipt.trace_nonce,
            )
            if observed_context is not None and context != observed_context:
                return ()
            observed_context = context
        elif receipt_type == SCENARIO_EFFECT_RECEIPT_TYPE:
            try:
                receipt = ScenarioEffectReceipt(**item)
            except TypeError:
                return ()
            if (
                asdict(receipt) != item
                or isinstance(receipt.schema_version, bool)
                or receipt.schema_version != CAPABILITY_RECEIPT_SCHEMA_VERSION
                or not isinstance(receipt.run_id, str)
                or _RUN_ID_RE.fullmatch(receipt.run_id) is None
                or not isinstance(receipt.case_id, str)
                or _CASE_ID_RE.fullmatch(receipt.case_id) is None
                or isinstance(receipt.repetition, bool)
                or not isinstance(receipt.repetition, int)
                or not 1 <= receipt.repetition <= 20
                or not isinstance(receipt.trace_nonce, str)
                or _NONCE_RE.fullmatch(receipt.trace_nonce) is None
                or isinstance(receipt.effect_sequence, bool)
                or not isinstance(receipt.effect_sequence, int)
                or receipt.effect_sequence != expected_effect_sequence
                or not isinstance(receipt.operation, str)
                or _EVENT_NAME_RE.fullmatch(receipt.operation) is None
                or not isinstance(receipt.accepted, bool)
                or not isinstance(receipt.effect_code, str)
                or _EVENT_NAME_RE.fullmatch(receipt.effect_code) is None
                or not isinstance(receipt.arguments_sha256, str)
                or _DIGEST_RE.fullmatch(receipt.arguments_sha256) is None
                or not isinstance(receipt.state_before_sha256, str)
                or _DIGEST_RE.fullmatch(receipt.state_before_sha256) is None
                or not isinstance(receipt.state_after_sha256, str)
                or _DIGEST_RE.fullmatch(receipt.state_after_sha256) is None
                or not isinstance(receipt.effect_sha256, str)
                or _DIGEST_RE.fullmatch(receipt.effect_sha256) is None
                or isinstance(receipt.virtual_start_ms, bool)
                or not isinstance(receipt.virtual_start_ms, int)
                or isinstance(receipt.virtual_end_ms, bool)
                or not isinstance(receipt.virtual_end_ms, int)
                or not 0 <= receipt.virtual_start_ms <= receipt.virtual_end_ms <= 1_000_000_000
                or (run_id is not None and receipt.run_id != run_id)
                or (case_id is not None and receipt.case_id != case_id)
                or (repetition is not None and receipt.repetition != repetition)
                or (trace_nonce is not None and receipt.trace_nonce != trace_nonce)
            ):
                return ()
            expected_effect_sequence += 1
            context = (
                receipt.run_id,
                receipt.case_id,
                receipt.repetition,
                receipt.trace_nonce,
            )
            if observed_context is not None and context != observed_context:
                return ()
            observed_context = context
        elif receipt_type == MODEL_CALL_RECEIPT_TYPE:
            try:
                receipt = ModelCallReceipt(**item)
            except TypeError:
                return ()
            if (
                asdict(receipt) != item
                or isinstance(receipt.schema_version, bool)
                or receipt.schema_version != CAPABILITY_RECEIPT_SCHEMA_VERSION
                or not isinstance(receipt.run_id, str)
                or _RUN_ID_RE.fullmatch(receipt.run_id) is None
                or not isinstance(receipt.case_id, str)
                or _CASE_ID_RE.fullmatch(receipt.case_id) is None
                or isinstance(receipt.repetition, bool)
                or not isinstance(receipt.repetition, int)
                or not 1 <= receipt.repetition <= 20
                or not isinstance(receipt.trace_nonce, str)
                or _NONCE_RE.fullmatch(receipt.trace_nonce) is None
                or isinstance(receipt.event_sequence, bool)
                or not isinstance(receipt.event_sequence, int)
                or receipt.event_sequence != expected_model_event_sequence
                or isinstance(receipt.call_sequence, bool)
                or not isinstance(receipt.call_sequence, int)
                or not isinstance(receipt.call_nonce, str)
                or _NONCE_RE.fullmatch(receipt.call_nonce) is None
                or not isinstance(receipt.source, str)
                or _EVENT_NAME_RE.fullmatch(receipt.source) is None
                or receipt.phase not in {"started", "finished"}
                or receipt.outcome not in _MODEL_CALL_OUTCOMES
                or not isinstance(receipt.usage_complete, bool)
                or (run_id is not None and receipt.run_id != run_id)
                or (case_id is not None and receipt.case_id != case_id)
                or (repetition is not None and receipt.repetition != repetition)
                or (trace_nonce is not None and receipt.trace_nonce != trace_nonce)
            ):
                return ()
            prior_starts = tuple(
                value
                for value in result
                if isinstance(value, ModelCallReceipt)
                and value.phase == "started"
                and value.call_sequence == receipt.call_sequence
            )
            prior_finishes = tuple(
                value
                for value in result
                if isinstance(value, ModelCallReceipt)
                and value.phase == "finished"
                and value.call_sequence == receipt.call_sequence
            )
            usage = _authoritative_token_usage(
                receipt.prompt_tokens,
                receipt.completion_tokens,
                receipt.total_tokens,
            )
            if receipt.phase == "started":
                if (
                    receipt.call_sequence != expected_model_call_sequence
                    or prior_starts
                    or receipt.outcome != "pending"
                    or receipt.usage_complete
                    or any(
                        value is not None
                        for value in (
                            receipt.prompt_tokens,
                            receipt.completion_tokens,
                            receipt.total_tokens,
                        )
                    )
                ):
                    return ()
                expected_model_call_sequence += 1
            elif (
                len(prior_starts) != 1
                or prior_finishes
                or receipt.outcome == "pending"
                or receipt.call_nonce != prior_starts[0].call_nonce
                or receipt.source != prior_starts[0].source
                or (
                    receipt.usage_complete
                    and (receipt.outcome != "succeeded" or usage is None)
                )
                or (
                    not receipt.usage_complete
                    and any(
                        value is not None
                        for value in (
                            receipt.prompt_tokens,
                            receipt.completion_tokens,
                            receipt.total_tokens,
                        )
                    )
                )
            ):
                return ()
            expected_model_event_sequence += 1
            context = (
                receipt.run_id,
                receipt.case_id,
                receipt.repetition,
                receipt.trace_nonce,
            )
            if observed_context is not None and context != observed_context:
                return ()
            observed_context = context
        else:
            return ()
        result.append(receipt)
    return tuple(result)


__all__ = (
    "CAPABILITY_RECEIPT_KEY_ENV",
    "CAPABILITY_RECEIPT_PATH_ENV",
    "BenchmarkTelemetryError",
    "FleetWaitCapabilityReceipt",
    "ModelCallHandle",
    "ModelCallReceipt",
    "ScenarioEffectReceipt",
    "ToolCallReceipt",
    "TRACE_CASE_ID_ENV",
    "TRACE_NONCE_ENV",
    "TRACE_REPETITION_ENV",
    "TRACE_RUN_ID_ENV",
    "append_scenario_effect_receipt",
    "benchmark_trace_requested",
    "begin_model_call",
    "decode_capability_receipts",
    "emit_fleet_wait_capability_receipt",
    "emit_tool_call_receipt",
    "finish_model_call",
    "tool_arguments_sha256",
)
