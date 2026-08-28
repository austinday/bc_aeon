"""Owner-private chat transcript shared by one Nexus-managed Aeon instance.

The transcript is deliberately separate from presence telemetry.  Presence stays
small and status-only, while this file is read only by Nexus's authenticated chat
endpoint.  Writers use one per-instance lock and append complete JSON records so
the web process and the foreground Aeon process cannot interleave messages.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

from .presence import sanitize_summary
from .collaborator_mode import COLLABORATOR_MODE_ENV


CHAT_TRANSCRIPT_ENV = "AEON_CHAT_TRANSCRIPT_PATH"
CHAT_WRITER_PID_ENV = "AEON_CHAT_WRITER_PID"
CHAT_TRANSCRIPT_FILENAME = "chat-transcript.jsonl"
CHAT_LOCK_FILENAME = "chat-transcript.lock"
CHAT_DELIVERY_STATE_FILENAME = "chat-delivery-state.json"
CHAT_DELIVERY_PREFIX = "/__nexus_chat_delivery_3d5eb9f4__"
CHAT_DELIVERY_COMMIT_WAIT_SECONDS = 10.0
# The deadline remains the primary bound.  This second independent bound keeps
# a monkeypatched/frozen clock or unexpectedly fast failing filesystem from
# turning the receiver into an unbounded loop, while still allowing the normal
# 10-second pending-commit window to elapse.
CHAT_DELIVERY_MAX_ATTEMPTS = 1024
CHAT_DELIVERY_QUARANTINE_LOCK_WAIT_SECONDS = 0.1
CHAT_DELIVERY_ACK_WAIT_SECONDS = 0.75
CHAT_DELIVERY_ACK_MAX_ATTEMPTS = 256
CHAT_DELIVERY_FAILURES_FILENAME = "chat-delivery-failures.jsonl"
MAX_CHAT_DELIVERY_FAILURE_BYTES = 256 * 1024
MAX_CHAT_DELIVERY_STATE_BYTES = 256 * 1024
MAX_CHAT_DELIVERY_ENTRIES = 512
MAX_CHAT_MESSAGE_BYTES = 20_000
MAX_CHAT_TRANSPORT_BYTES = 40_000
MAX_CHAT_TRANSCRIPT_BYTES = 8 * 1024 * 1024
# Public sibling transcripts are long-lived but contain no owner-only history.
# Compact them well before the generic hard stop, retaining a generous recent
# suffix and any currently unmatched user turn.
COLLABORATOR_CHAT_TRANSCRIPT_BYTES = 4 * 1024 * 1024
COLLABORATOR_CHAT_RETAIN_BYTES = 2 * 1024 * 1024
MAX_CHAT_MESSAGES = 500
_ROLES = frozenset({"user", "assistant", "progress", "plan"})
_ATTACHMENT_ID_RE = re.compile(r"^att-[0-9a-f]{32}$")
_MEDIA_TYPES = frozenset({"image", "video", "audio"})
_CLEAR_COMMAND = "/clear"


class ChatTranscriptError(RuntimeError):
    """The private transcript could not be safely read or updated."""


def _private_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ChatTranscriptError("Chat storage is unavailable") from exc
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise ChatTranscriptError("Chat storage is not owner-private")
    return descriptor


def _open_owned_file(
    directory_fd: int,
    name: str,
    *,
    flags: int,
    create: bool,
) -> int:
    safe_flags = flags | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    if create:
        safe_flags |= os.O_CREAT
    try:
        descriptor = os.open(name, safe_flags, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        raise ChatTranscriptError("Chat storage is unavailable") from exc
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise ChatTranscriptError("Chat storage is not an owner-private file")
    return descriptor


def _acquire_flock_until(
    descriptor: int,
    operation: int,
    *,
    deadline: float | None,
) -> None:
    """Acquire a lock, with a hard monotonic bound for receiver-side paths."""

    if deadline is None:
        fcntl.flock(descriptor, operation)
        return
    while True:
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ChatTranscriptError("Chat delivery lock remained busy") from exc
            time.sleep(min(0.01, remaining))


def _attachments(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or len(value) > 4:
        raise ChatTranscriptError("Chat attachments are invalid")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "id", "name", "media_type", "mime_type", "size_bytes"
        }:
            raise ChatTranscriptError("Chat attachment metadata is invalid")
        attachment_id = item.get("id")
        name = item.get("name")
        media_type = item.get("media_type")
        mime_type = item.get("mime_type")
        size_bytes = item.get("size_bytes")
        if (
            not isinstance(attachment_id, str)
            or not _ATTACHMENT_ID_RE.fullmatch(attachment_id)
            or attachment_id in seen
            or not isinstance(name, str)
            or not name
            or len(name.encode("utf-8")) > 180
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
            or media_type not in _MEDIA_TYPES
            or not isinstance(mime_type, str)
            or not mime_type.startswith(f"{media_type}/")
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or size_bytes > 128 * 1024 * 1024
        ):
            raise ChatTranscriptError("Chat attachment metadata is invalid")
        seen.add(attachment_id)
        normalized.append(dict(item))
    return normalized


def _message(
    role: str,
    content: str,
    *,
    message_id: str | None = None,
    attachments: object = None,
    performance: object = None,
) -> dict[str, Any]:
    normalized_role = str(role or "").strip().lower()
    if normalized_role not in _ROLES:
        raise ChatTranscriptError("Chat message role is invalid")
    if not isinstance(content, str):
        raise ChatTranscriptError("Chat message must be text")
    rendered = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if any(ord(character) < 32 and character not in {"\n", "\t"} for character in rendered):
        raise ChatTranscriptError("Chat message contains unsupported control characters")
    encoded = rendered.encode("utf-8")
    if (not rendered and normalized_role != "plan") or len(encoded) > MAX_CHAT_MESSAGE_BYTES:
        raise ChatTranscriptError("Chat message must contain 1-20,000 UTF-8 bytes")
    identifier = message_id or f"msg-{uuid.uuid4().hex}"
    if not isinstance(identifier, str) or not identifier.startswith("msg-") or len(identifier) != 36:
        raise ChatTranscriptError("Chat message identity is invalid")
    record = {
        "id": identifier,
        "role": normalized_role,
        "content": rendered,
        "created_at": time.time(),
    }
    normalized_attachments = _attachments(attachments)
    if normalized_attachments:
        if normalized_role not in {"user", "assistant"}:
            raise ChatTranscriptError("Progress messages may not contain attachments")
        record["attachments"] = normalized_attachments
    if performance is not None:
        if normalized_role != "assistant" or not isinstance(performance, dict):
            raise ChatTranscriptError("Chat response performance is invalid")
        tokens_per_second = performance.get("tokens_per_second")
        completion_tokens = performance.get("completion_tokens")
        if (
            isinstance(tokens_per_second, bool)
            or not isinstance(tokens_per_second, (int, float))
            or not 0 < float(tokens_per_second) < 100_000
            or isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or not 0 < completion_tokens <= 1_000_000
        ):
            raise ChatTranscriptError("Chat response performance is invalid")
        normalized_performance: dict[str, Any] = {
            "tokens_per_second": round(float(tokens_per_second), 2),
            "completion_tokens": completion_tokens,
        }
        numeric_fields = {
            "decode_tokens_per_second": (0.0, 100_000.0, 2),
            "end_to_end_tokens_per_second": (0.0, 100_000.0, 2),
            "inference_tokens_per_second": (0.0, 100_000.0, 2),
            "time_to_first_token_seconds": (0.0, 86_400.0, 3),
            "prefill_time_to_first_token_seconds": (0.0, 86_400.0, 3),
            "queue_seconds": (0.0, 86_400.0, 3),
            "mean_inter_token_seconds": (0.0, 86_400.0, 4),
            "decode_seconds": (0.0, 86_400.0, 3),
            "end_to_end_seconds": (0.0, 86_400.0, 3),
        }
        for key, (minimum, maximum, digits) in numeric_fields.items():
            value = performance.get(key)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not minimum <= float(value) < maximum
            ):
                raise ChatTranscriptError("Chat response performance is invalid")
            normalized_performance[key] = round(float(value), digits)
        integer_fields = {
            "prompt_tokens": 10_000_000,
            "cached_prompt_tokens": 10_000_000,
            "speculative_tokens": 64,
        }
        for key, maximum in integer_fields.items():
            value = performance.get(key)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= maximum
            ):
                raise ChatTranscriptError("Chat response performance is invalid")
            normalized_performance[key] = value
        for key, maximum in {
            "reasoning_effort": 16,
            "served_model": 160,
            "measurement": 80,
            "speculative_method": 16,
        }.items():
            value = performance.get(key)
            if value is None:
                continue
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value.encode("utf-8")) > maximum
                or any(ord(character) < 32 for character in value)
            ):
                raise ChatTranscriptError("Chat response performance is invalid")
            normalized_performance[key] = value.strip()
        if normalized_performance.get("cached_prompt_tokens", 0) > normalized_performance.get(
            "prompt_tokens", 0
        ):
            raise ChatTranscriptError("Chat response performance is invalid")
        record["performance"] = normalized_performance
    return record


def normalize_chat_message(content: str) -> str:
    """Validate browser-delivered text before it reaches the managed pane."""

    return str(_message("user", content)["content"])


def _transport_identity(content: str) -> tuple[str, str]:
    if not isinstance(content, str):
        raise ChatTranscriptError("Chat delivery transport must be text")
    rendered = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if (
        not rendered
        or len(rendered.encode("utf-8")) > MAX_CHAT_TRANSPORT_BYTES
        or any(
            ord(character) < 32 and character not in {"\n", "\t"}
            for character in rendered
        )
    ):
        raise ChatTranscriptError("Chat delivery transport is invalid")
    return rendered, hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _visible_identity(
    content: str,
    attachments: object = None,
) -> tuple[str, list[dict[str, Any]], str]:
    rendered = normalize_chat_message(content)
    normalized_attachments = _attachments(attachments)
    identity = json.dumps(
        {
            "role": "user",
            "content": rendered,
            "attachments": normalized_attachments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return rendered, normalized_attachments, hashlib.sha256(identity).hexdigest()


def chat_delivery_claim_sha256(
    transport_content: str,
    *,
    visible_content: str | None = None,
    attachments: object = None,
) -> str:
    """Bind the private PTY transport and public transcript record in one DB key."""

    _transport, transport_sha256 = _transport_identity(transport_content)
    _visible, _attachments_value, record_sha256 = _visible_identity(
        transport_content if visible_content is None else visible_content,
        attachments,
    )
    return hashlib.sha256(
        f"transport:{transport_sha256}\nrecord:{record_sha256}".encode("ascii")
    ).hexdigest()


def build_chat_delivery_envelope(
    message_id: str,
    content: str,
    *,
    visible_content: str | None = None,
    attachments: object = None,
) -> str:
    """Bind one private PTY line to its distinct future visible record."""

    if not re.fullmatch(r"msg-[A-Za-z0-9_-]{32}", str(message_id or "")):
        raise ChatTranscriptError("Chat delivery identity is invalid")
    rendered, transport_sha256 = _transport_identity(content)
    _visible, _attachments_value, record_sha256 = _visible_identity(
        content if visible_content is None else visible_content,
        attachments,
    )
    return (
        f"{CHAT_DELIVERY_PREFIX}:{message_id}:{transport_sha256}:{record_sha256}\n"
        f"{rendered}"
    )


def committed_chat_delivery_from_environment(line: str) -> str | None:
    """Unwrap a managed PTY line only after its exact user record is durable.

    The manager must paste before appending so an ambiguous tmux result cannot
    expose a visible phantom message.  This receiver-side commit barrier keeps
    the pasted envelope out of the model until the matching record exists.
    """

    if not isinstance(line, str) or not line.startswith(CHAT_DELIVERY_PREFIX):
        return line
    header, separator, content = line.partition("\n")
    match = re.fullmatch(
        re.escape(CHAT_DELIVERY_PREFIX)
        + r":(msg-[A-Za-z0-9_-]{32}):([0-9a-f]{64})(?::([0-9a-f]{64}))?",
        header,
    )
    if not separator or match is None:
        return None
    message_id, expected_transport_digest, expected_record_digest = match.groups()
    if (
        hashlib.sha256(content.encode("utf-8")).hexdigest()
        != expected_transport_digest
    ):
        return None
    # Version-1 envelopes used one digest because transport and transcript were
    # identical. Keep them consumable across a rolling upgrade.
    expected_record_digest = expected_record_digest or expected_transport_digest
    transcript = os.environ.get(CHAT_TRANSCRIPT_ENV, "")
    if not transcript:
        return None
    deadline = time.monotonic() + CHAT_DELIVERY_COMMIT_WAIT_SECONDS
    attempts = 0
    last_error = ""
    while attempts < CHAT_DELIVERY_MAX_ATTEMPTS:
        attempts += 1
        try:
            status = _consume_chat_delivery(
                transcript,
                message_id,
                content,
                record_sha256=expected_record_digest,
                abandon_pending=time.monotonic() >= deadline,
                lock_deadline=deadline,
            )
        except ChatTranscriptError as exc:
            last_error = str(exc)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
            continue
        if status == "accepted":
            return content
        if status == "rejected":
            return None
        if time.monotonic() >= deadline:
            # One final locked pass asks the state machine to abandon pending.
            continue
        time.sleep(0.01)
    _quarantine_failed_chat_delivery(
        transcript,
        message_id,
        expected_transport_digest,
        expected_record_digest,
        last_error or "Chat delivery did not reach a terminal state",
    )
    warnings.warn(
        "A managed chat envelope was dropped after its durable delivery state failed",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise ChatTranscriptError("Chat message could not be saved")
        offset += written


def _collaborator_suffix(existing: bytes, new_payload: bytes) -> bytes:
    """Build a recent complete-record suffix without losing an active user turn."""

    lines = [line + b"\n" for line in existing.splitlines() if line]
    selected_start = len(lines)
    retained_bytes = len(new_payload)
    for index in range(len(lines) - 1, -1, -1):
        if (
            selected_start < len(lines)
            and retained_bytes + len(lines[index]) > COLLABORATOR_CHAT_RETAIN_BYTES
        ):
            break
        selected_start = index
        retained_bytes += len(lines[index])

    latest_user = -1
    latest_assistant = -1
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except (UnicodeError, ValueError, TypeError):
            continue
        if not isinstance(value, dict):
            continue
        if value.get("role") == "user":
            latest_user = index
        elif value.get("role") == "assistant":
            latest_assistant = index
    if latest_user > latest_assistant:
        selected_start = min(selected_start, latest_user)

    compacted = b"".join(lines[selected_start:]) + new_payload
    if len(compacted) > COLLABORATOR_CHAT_TRANSCRIPT_BYTES:
        raise ChatTranscriptError(
            "The active collaborator turn cannot be retained within its storage bound"
        )
    return compacted


def _replace_private_file(
    directory_fd: int,
    target_name: str,
    payload: bytes,
    *,
    label: str,
) -> None:
    """Atomically replace one lock-protected owner-private state file."""

    temporary_name = f".{label}-{uuid.uuid4().hex}.tmp"
    temporary_fd = None
    try:
        temporary_fd = _open_owned_file(
            directory_fd,
            temporary_name,
            flags=os.O_WRONLY | os.O_EXCL,
            create=True,
        )
        _write_all(temporary_fd, payload)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        # The old descriptor remains open only so its already-validated inode
        # cannot be swapped before this exact same-directory replacement.
        os.replace(
            temporary_name,
            target_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except OSError as exc:
        raise ChatTranscriptError("Chat state could not be committed") from exc
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _read_owned_bytes(
    directory_fd: int,
    name: str,
    *,
    maximum: int,
) -> bytes | None:
    try:
        descriptor = _open_owned_file(
            directory_fd,
            name,
            flags=os.O_RDONLY,
            create=False,
        )
    except ChatTranscriptError as exc:
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError:
            pass
        raise exc
    try:
        size = os.fstat(descriptor).st_size
        if size > maximum:
            raise ChatTranscriptError("Chat state exceeded its storage limit")
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _delivery_entries_locked(directory_fd: int) -> dict[str, dict[str, Any]]:
    payload = _read_owned_bytes(
        directory_fd,
        CHAT_DELIVERY_STATE_FILENAME,
        maximum=MAX_CHAT_DELIVERY_STATE_BYTES,
    )
    if payload is None:
        return {}
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ChatTranscriptError("Chat delivery state is invalid") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"version", "deliveries"}
        or document.get("version") not in {1, 2}
        or not isinstance(document.get("deliveries"), dict)
        or len(document["deliveries"]) > MAX_CHAT_DELIVERY_ENTRIES
    ):
        raise ChatTranscriptError("Chat delivery state is invalid")
    version = int(document["version"])
    entries: dict[str, dict[str, Any]] = {}
    for message_id, value in document["deliveries"].items():
        expected_keys = (
            {"sha256", "state", "created_at", "updated_at"}
            if version == 1
            else {
                "transport_sha256",
                "record_sha256",
                "state",
                "created_at",
                "updated_at",
            }
        )
        if (
            not isinstance(message_id, str)
            or not re.fullmatch(r"msg-[A-Za-z0-9_-]{32}", message_id)
            or not isinstance(value, dict)
            or set(value) != expected_keys
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(
                    value.get("sha256")
                    if version == 1
                    else value.get("transport_sha256")
                    or ""
                ),
            )
            or (
                version == 2
                and not re.fullmatch(
                    r"[0-9a-f]{64}", str(value.get("record_sha256") or "")
                )
            )
            or value.get("state")
            not in {"pending", "committed", "consumed", "abandoned"}
            or isinstance(value.get("created_at"), bool)
            or not isinstance(value.get("created_at"), (int, float))
            or isinstance(value.get("updated_at"), bool)
            or not isinstance(value.get("updated_at"), (int, float))
        ):
            raise ChatTranscriptError("Chat delivery state is invalid")
        entry = dict(value)
        if version == 1:
            legacy_digest = str(entry.pop("sha256"))
            entry["transport_sha256"] = legacy_digest
            entry["record_sha256"] = legacy_digest
        entries[message_id] = entry
    return entries


def _write_delivery_entries_locked(
    directory_fd: int,
    entries: dict[str, dict[str, Any]],
) -> None:
    active = {
        key: value
        for key, value in entries.items()
        if value["state"] in {"pending", "committed"}
    }
    if len(active) > MAX_CHAT_DELIVERY_ENTRIES:
        raise ChatTranscriptError("Too many chat deliveries are unresolved")
    terminal = sorted(
        (
            (key, value)
            for key, value in entries.items()
            if value["state"] in {"consumed", "abandoned"}
        ),
        key=lambda item: float(item[1]["updated_at"]),
        reverse=True,
    )
    keep = dict(active)
    for key, value in terminal[: MAX_CHAT_DELIVERY_ENTRIES - len(active)]:
        keep[key] = value
    payload = (
        json.dumps(
            {"version": 2, "deliveries": keep},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_CHAT_DELIVERY_STATE_BYTES:
        raise ChatTranscriptError("Chat delivery state reached its storage limit")
    _replace_private_file(
        directory_fd,
        CHAT_DELIVERY_STATE_FILENAME,
        payload,
        label="chat-delivery-state",
    )


def _append_delivery_failure_locked(
    directory_fd: int,
    *,
    message_id: str,
    transport_sha256: str,
    record_sha256: str,
    detail: str,
) -> None:
    record = {
        "message_id": message_id,
        "transport_sha256": transport_sha256,
        "record_sha256": record_sha256,
        "status": "dropped",
        "error": sanitize_summary(detail, max_chars=200)
        or "delivery state unavailable",
        "created_at": time.time(),
    }
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    descriptor = _open_owned_file(
        directory_fd,
        CHAT_DELIVERY_FAILURES_FILENAME,
        flags=os.O_RDWR,
        create=True,
    )
    try:
        size = os.fstat(descriptor).st_size
        if size + len(payload) > MAX_CHAT_DELIVERY_FAILURE_BYTES:
            raise ChatTranscriptError("Chat delivery failure log is full")
        os.lseek(descriptor, 0, os.SEEK_END)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _quarantine_failed_chat_delivery(
    path: str | Path,
    message_id: str,
    transport_sha256: str,
    record_sha256: str,
    detail: str,
) -> None:
    """Durably drop one finite receiver failure and isolate corrupt JSON state."""

    transcript = Path(path)
    if (
        transcript.name != CHAT_TRANSCRIPT_FILENAME
        or not transcript.is_absolute()
        or not re.fullmatch(r"msg-[A-Za-z0-9_-]{32}", message_id)
        or not re.fullmatch(r"[0-9a-f]{64}", transport_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", record_sha256)
    ):
        return
    try:
        directory_fd = _private_directory(transcript.parent)
    except ChatTranscriptError:
        return
    try:
        lock_fd = _open_owned_file(
            directory_fd,
            CHAT_LOCK_FILENAME,
            flags=os.O_RDWR,
            create=True,
        )
        lock_acquired = False
        try:
            _acquire_flock_until(
                lock_fd,
                fcntl.LOCK_EX,
                deadline=(
                    time.monotonic()
                    + CHAT_DELIVERY_QUARANTINE_LOCK_WAIT_SECONDS
                ),
            )
            lock_acquired = True
            try:
                _append_delivery_failure_locked(
                    directory_fd,
                    message_id=message_id,
                    transport_sha256=transport_sha256,
                    record_sha256=record_sha256,
                    detail=detail,
                )
            except ChatTranscriptError:
                pass

            try:
                entries = _delivery_entries_locked(directory_fd)
            except ChatTranscriptError:
                # Quarantine only a regular owner-private state file. Unsafe
                # filesystem entries remain untouched and still fail closed.
                try:
                    _read_owned_bytes(
                        directory_fd,
                        CHAT_DELIVERY_STATE_FILENAME,
                        maximum=MAX_CHAT_DELIVERY_STATE_BYTES,
                    )
                    quarantine_name = (
                        f"chat-delivery-state.quarantine-{uuid.uuid4().hex}.json"
                    )
                    os.replace(
                        CHAT_DELIVERY_STATE_FILENAME,
                        quarantine_name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                    )
                    os.fsync(directory_fd)
                    entries = {}
                except (ChatTranscriptError, FileNotFoundError, OSError):
                    return
            now = time.time()
            existing = entries.get(message_id)
            if existing is None:
                entries[message_id] = {
                    "transport_sha256": transport_sha256,
                    "record_sha256": record_sha256,
                    "state": "abandoned",
                    "created_at": now,
                    "updated_at": now,
                }
            elif (
                existing["transport_sha256"] == transport_sha256
                and existing["record_sha256"] == record_sha256
                and existing["state"] != "consumed"
            ):
                existing["state"] = "abandoned"
                existing["updated_at"] = now
            _write_delivery_entries_locked(directory_fd, entries)
        finally:
            if lock_acquired:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    except ChatTranscriptError:
        pass
    finally:
        os.close(directory_fd)


def _exact_user_record_locked(
    directory_fd: int,
    message_id: str,
    record_sha256: str,
) -> dict[str, Any] | None:
    payload = _read_owned_bytes(
        directory_fd,
        CHAT_TRANSCRIPT_FILENAME,
        maximum=MAX_CHAT_TRANSCRIPT_BYTES,
    )
    if payload is None:
        return None
    for raw_line in reversed(payload.splitlines()):
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeError, ValueError, TypeError):
            continue
        if not isinstance(value, dict) or value.get("id") != message_id:
            continue
        try:
            _content, _public_attachments, observed_sha256 = _visible_identity(
                value.get("content"), value.get("attachments")
            )
        except ChatTranscriptError as exc:
            raise ChatTranscriptError(
                "Chat delivery identity conflicts with history"
            ) from exc
        if value.get("role") != "user" or observed_sha256 != record_sha256:
            raise ChatTranscriptError("Chat delivery identity conflicts with history")
        return value
    return None


def _append_record_locked(
    directory_fd: int,
    record: dict[str, Any],
    *,
    rolling: bool,
) -> None:
    payload = (
        json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    transcript_fd = _open_owned_file(
        directory_fd,
        CHAT_TRANSCRIPT_FILENAME,
        flags=os.O_RDWR,
        create=True,
    )
    try:
        metadata = os.fstat(transcript_fd)
        if metadata.st_size > MAX_CHAT_TRANSCRIPT_BYTES:
            raise ChatTranscriptError("Chat transcript exceeded its storage limit")
        if (
            rolling
            and metadata.st_size + len(payload) > COLLABORATOR_CHAT_TRANSCRIPT_BYTES
        ):
            os.lseek(transcript_fd, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                chunk = os.read(transcript_fd, min(remaining, 65_536))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            compacted = _collaborator_suffix(b"".join(chunks), payload)
            _replace_private_file(
                directory_fd,
                CHAT_TRANSCRIPT_FILENAME,
                compacted,
                label="chat-transcript",
            )
        elif metadata.st_size + len(payload) > MAX_CHAT_TRANSCRIPT_BYTES:
            raise ChatTranscriptError("Chat transcript reached its storage limit")
        else:
            os.lseek(transcript_fd, 0, os.SEEK_END)
            _write_all(transcript_fd, payload)
            os.fsync(transcript_fd)
    finally:
        os.close(transcript_fd)


def _delivery_identity(
    message_id: str,
    transport_content: str,
    *,
    visible_content: str | None = None,
    attachments: object = None,
) -> tuple[str, str, str, list[dict[str, Any]], str]:
    if not re.fullmatch(r"msg-[A-Za-z0-9_-]{32}", str(message_id or "")):
        raise ChatTranscriptError("Chat delivery identity is invalid")
    transport, transport_sha256 = _transport_identity(transport_content)
    visible, public_attachments, record_sha256 = _visible_identity(
        transport_content if visible_content is None else visible_content,
        attachments,
    )
    return (
        transport,
        transport_sha256,
        visible,
        public_attachments,
        record_sha256,
    )


def prepare_chat_delivery(
    path: str | Path,
    message_id: str,
    content: str,
    *,
    visible_content: str | None = None,
    attachments: object = None,
) -> str:
    """Durably register one exact managed line before it crosses tmux."""

    transcript = Path(path)
    if transcript.name != CHAT_TRANSCRIPT_FILENAME or not transcript.is_absolute():
        raise ChatTranscriptError("Chat transcript path is invalid")
    transport, transport_sha256, visible, public_attachments, record_sha256 = (
        _delivery_identity(
            message_id,
            content,
            visible_content=visible_content,
            attachments=attachments,
        )
    )
    directory_fd = _private_directory(transcript.parent)
    try:
        lock_fd = _open_owned_file(
            directory_fd,
            CHAT_LOCK_FILENAME,
            flags=os.O_RDWR,
            create=True,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            entries = _delivery_entries_locked(directory_fd)
            existing = entries.get(message_id)
            if existing is not None:
                if (
                    existing["transport_sha256"] != transport_sha256
                    or existing["record_sha256"] != record_sha256
                ):
                    raise ChatTranscriptError("Chat delivery identity conflicts")
                if existing["state"] == "abandoned":
                    raise ChatTranscriptError("Chat delivery was already abandoned")
                if existing["state"] in {"committed", "consumed"}:
                    raise ChatTranscriptError("Chat delivery was already committed")
            else:
                now = time.time()
                entries[message_id] = {
                    "transport_sha256": transport_sha256,
                    "record_sha256": record_sha256,
                    "state": "pending",
                    "created_at": now,
                    "updated_at": now,
                }
                _write_delivery_entries_locked(directory_fd, entries)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    finally:
        os.close(directory_fd)
    return build_chat_delivery_envelope(
        message_id,
        transport,
        visible_content=visible,
        attachments=public_attachments,
    )


def commit_chat_delivery(
    path: str | Path,
    message_id: str,
    content: str,
    *,
    visible_content: str | None = None,
    attachments: object = None,
    rolling: bool = False,
) -> dict[str, Any]:
    """Linearize transcript visibility against receiver abandonment."""

    transcript = Path(path)
    if transcript.name != CHAT_TRANSCRIPT_FILENAME or not transcript.is_absolute():
        raise ChatTranscriptError("Chat transcript path is invalid")
    transport, transport_sha256, visible, public_attachments, record_sha256 = (
        _delivery_identity(
            message_id,
            content,
            visible_content=visible_content,
            attachments=attachments,
        )
    )
    record = _message(
        "user",
        visible,
        message_id=message_id,
        attachments=public_attachments,
    )
    directory_fd = _private_directory(transcript.parent)
    try:
        lock_fd = _open_owned_file(
            directory_fd,
            CHAT_LOCK_FILENAME,
            flags=os.O_RDWR,
            create=True,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            entries = _delivery_entries_locked(directory_fd)
            entry = entries.get(message_id)
            if entry is None or (
                entry["transport_sha256"] != transport_sha256
                or entry["record_sha256"] != record_sha256
            ):
                raise ChatTranscriptError("Chat delivery was not prepared")
            if entry["state"] == "abandoned":
                raise ChatTranscriptError("Chat delivery was abandoned before commit")
            existing = _exact_user_record_locked(
                directory_fd, message_id, record_sha256
            )
            if existing is None:
                if entry["state"] not in {"pending", "committed"}:
                    raise ChatTranscriptError("Chat delivery state conflicts with history")
                _append_record_locked(directory_fd, record, rolling=rolling)
                existing = record
            entry["state"] = "committed"
            entry["updated_at"] = time.time()
            _write_delivery_entries_locked(directory_fd, entries)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    finally:
        os.close(directory_fd)
    return existing


def abandon_chat_delivery(
    path: str | Path,
    message_id: str,
    content: str,
    *,
    visible_content: str | None = None,
    attachments: object = None,
) -> bool:
    """Atomically abandon only a still-invisible prepared delivery."""

    return (
        _consume_chat_delivery(
            path,
            message_id,
            content,
            record_sha256=_delivery_identity(
                message_id,
                content,
                visible_content=visible_content,
                attachments=attachments,
            )[-1],
            abandon_pending=True,
        )
        == "rejected"
    )


def _consume_chat_delivery(
    path: str | Path,
    message_id: str,
    content: str,
    *,
    record_sha256: str | None = None,
    abandon_pending: bool,
    lock_deadline: float | None = None,
) -> str:
    """Return wait/accepted/rejected under the shared transcript lock."""

    transcript = Path(path)
    if transcript.name != CHAT_TRANSCRIPT_FILENAME or not transcript.is_absolute():
        raise ChatTranscriptError("Chat transcript path is invalid")
    transport, transport_sha256 = _transport_identity(content)
    expected_record_sha256 = record_sha256 or transport_sha256
    if not re.fullmatch(r"[0-9a-f]{64}", expected_record_sha256):
        raise ChatTranscriptError("Chat delivery record identity is invalid")
    directory_fd = _private_directory(transcript.parent)
    try:
        lock_fd = _open_owned_file(
            directory_fd,
            CHAT_LOCK_FILENAME,
            flags=os.O_RDWR,
            create=True,
        )
        lock_acquired = False
        try:
            _acquire_flock_until(
                lock_fd,
                fcntl.LOCK_EX,
                deadline=lock_deadline,
            )
            lock_acquired = True
            entries = _delivery_entries_locked(directory_fd)
            entry = entries.get(message_id)
            if entry is None:
                return "rejected"
            if (
                entry["transport_sha256"] != transport_sha256
                or entry["record_sha256"] != expected_record_sha256
            ):
                raise ChatTranscriptError("Chat delivery identity conflicts")
            if entry["state"] in {"consumed", "abandoned"}:
                return "rejected"
            existing = _exact_user_record_locked(
                directory_fd, message_id, expected_record_sha256
            )
            if existing is not None:
                entry["state"] = "consumed"
                entry["updated_at"] = time.time()
                _write_delivery_entries_locked(directory_fd, entries)
                return "accepted"
            if abandon_pending:
                entry["state"] = "abandoned"
                entry["updated_at"] = time.time()
                _write_delivery_entries_locked(directory_fd, entries)
                return "rejected"
            return "wait"
        finally:
            if lock_acquired:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    finally:
        os.close(directory_fd)


def wait_for_chat_delivery_consumed(
    path: str | Path,
    message_id: str,
    content: str,
    *,
    visible_content: str | None = None,
    attachments: object = None,
    timeout_seconds: float | None = None,
) -> bool:
    """Boundedly prove that the foreground receiver accepted one exact envelope.

    Transcript visibility is the crash-recovery commit point, but it is not an
    HTTP delivery receipt.  Only the receiver moves this state to ``consumed``
    after validating the exact envelope and visible record.
    """

    transcript = Path(path)
    if transcript.name != CHAT_TRANSCRIPT_FILENAME or not transcript.is_absolute():
        return False
    try:
        (
            _transport,
            transport_sha256,
            _visible,
            _public_attachments,
            record_sha256,
        ) = _delivery_identity(
            message_id,
            content,
            visible_content=visible_content,
            attachments=attachments,
        )
    except ChatTranscriptError:
        return False
    wait_seconds = (
        CHAT_DELIVERY_ACK_WAIT_SECONDS
        if timeout_seconds is None
        else max(0.0, min(float(timeout_seconds), CHAT_DELIVERY_ACK_WAIT_SECONDS))
    )
    deadline = time.monotonic() + wait_seconds
    try:
        directory_fd = _private_directory(transcript.parent)
        lock_fd = _open_owned_file(
            directory_fd,
            CHAT_LOCK_FILENAME,
            flags=os.O_RDWR,
            create=True,
        )
    except (ChatTranscriptError, TypeError, ValueError, OverflowError):
        try:
            os.close(directory_fd)
        except (NameError, OSError):
            pass
        return False
    try:
        for _attempt in range(CHAT_DELIVERY_ACK_MAX_ATTEMPTS):
            now = time.monotonic()
            lock_acquired = False
            try:
                _acquire_flock_until(
                    lock_fd,
                    fcntl.LOCK_SH,
                    deadline=min(deadline, now + 0.02),
                )
                lock_acquired = True
                entries = _delivery_entries_locked(directory_fd)
                entry = entries.get(message_id)
                if entry is not None:
                    if (
                        entry["transport_sha256"] != transport_sha256
                        or entry["record_sha256"] != record_sha256
                    ):
                        return False
                    if entry["state"] == "consumed":
                        return True
                    if entry["state"] == "abandoned":
                        return False
            except ChatTranscriptError:
                pass
            finally:
                if lock_acquired:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.005, remaining))
        return False
    finally:
        os.close(lock_fd)
        os.close(directory_fd)


def append_chat_message(
    path: str | Path,
    *,
    role: str,
    content: str,
    message_id: str | None = None,
    attachments: object = None,
    performance: object = None,
    rolling: bool = False,
) -> dict[str, Any]:
    """Append one complete record beneath an already-private instance directory."""

    transcript = Path(path)
    if transcript.name != CHAT_TRANSCRIPT_FILENAME or not transcript.is_absolute():
        raise ChatTranscriptError("Chat transcript path is invalid")
    record = _message(
        role,
        content,
        message_id=message_id,
        attachments=attachments,
        performance=performance,
    )
    directory_fd = _private_directory(transcript.parent)
    try:
        lock_fd = _open_owned_file(
            directory_fd,
            CHAT_LOCK_FILENAME,
            flags=os.O_RDWR,
            create=True,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            _append_record_locked(directory_fd, record, rolling=rolling)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    finally:
        os.close(directory_fd)
    return record


def read_chat_messages(path: str | Path) -> list[dict[str, Any]]:
    """Return a bounded, validated suffix of the private transcript."""

    transcript = Path(path)
    if transcript.name != CHAT_TRANSCRIPT_FILENAME or not transcript.is_absolute():
        raise ChatTranscriptError("Chat transcript path is invalid")
    directory_fd = _private_directory(transcript.parent)
    try:
        lock_fd = _open_owned_file(
            directory_fd,
            CHAT_LOCK_FILENAME,
            flags=os.O_RDWR,
            create=True,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH)
            try:
                transcript_fd = _open_owned_file(
                    directory_fd,
                    CHAT_TRANSCRIPT_FILENAME,
                    flags=os.O_RDONLY,
                    create=False,
                )
            except ChatTranscriptError as exc:
                try:
                    os.stat(CHAT_TRANSCRIPT_FILENAME, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return []
                except OSError:
                    pass
                raise exc
            try:
                size = os.fstat(transcript_fd).st_size
                if size > MAX_CHAT_TRANSCRIPT_BYTES:
                    raise ChatTranscriptError("Chat transcript exceeded its storage limit")
                chunks: list[bytes] = []
                remaining = size
                while remaining:
                    chunk = os.read(transcript_fd, min(remaining, 65_536))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
            finally:
                os.close(transcript_fd)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    finally:
        os.close(directory_fd)

    messages: list[dict[str, Any]] = []
    for raw_line in b"".join(chunks).splitlines():
        try:
            value = json.loads(raw_line.decode("utf-8"))
            if not isinstance(value, dict) or not set(value).issubset({
                "id", "role", "content", "created_at", "attachments", "performance"
            }) or not {"id", "role", "content", "created_at"}.issubset(value):
                continue
            validated = _message(
                value["role"],
                value["content"],
                message_id=value["id"],
                attachments=value.get("attachments"),
                performance=value.get("performance"),
            )
            created_at = value["created_at"]
            if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
                continue
            validated["created_at"] = float(created_at)
            # /clear is a control record, not conversation content.  Filtering it
            # also closes the tiny delivery/append race with the separate Aeon
            # process that truncates the transcript when it handles the command.
            if (
                validated["role"] == "user"
                and validated["content"].strip().lower() == _CLEAR_COMMAND
            ):
                continue
            messages.append(validated)
        except (ChatTranscriptError, UnicodeError, ValueError, TypeError):
            continue
    visible = messages[-MAX_CHAT_MESSAGES:]
    if not any(message["role"] == "plan" for message in visible):
        latest_plan = next(
            (message for message in reversed(messages) if message["role"] == "plan"),
            None,
        )
        if latest_plan is not None:
            visible = [latest_plan, *visible[-(MAX_CHAT_MESSAGES - 1):]]
    return visible


def clear_chat_messages(path: str | Path) -> None:
    """Atomically empty one owner-private transcript without replacing its inode."""

    transcript = Path(path)
    if transcript.name != CHAT_TRANSCRIPT_FILENAME or not transcript.is_absolute():
        raise ChatTranscriptError("Chat transcript path is invalid")
    directory_fd = _private_directory(transcript.parent)
    try:
        lock_fd = _open_owned_file(
            directory_fd,
            CHAT_LOCK_FILENAME,
            flags=os.O_RDWR,
            create=True,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            transcript_fd = _open_owned_file(
                directory_fd,
                CHAT_TRANSCRIPT_FILENAME,
                flags=os.O_WRONLY,
                create=True,
            )
            try:
                os.ftruncate(transcript_fd, 0)
                os.fsync(transcript_fd)
            finally:
                os.close(transcript_fd)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    finally:
        os.close(directory_fd)


def clear_chat_messages_from_environment() -> bool:
    """Clear chat only from the exact primary Aeon process for this transcript."""

    path = os.environ.get(CHAT_TRANSCRIPT_ENV, "")
    expected_pid = os.environ.get(CHAT_WRITER_PID_ENV, "")
    if not path or expected_pid != str(os.getpid()):
        return False
    clear_chat_messages(path)
    return True


def append_assistant_message_from_environment(
    content: str,
    *,
    performance: object = None,
    artifact_paths: object = None,
) -> dict[str, Any] | bool:
    """Append from the primary Aeon PID and privately stage its media artifacts."""

    path = os.environ.get(CHAT_TRANSCRIPT_ENV, "")
    expected_pid = os.environ.get(CHAT_WRITER_PID_ENV, "")
    if not path or expected_pid != str(os.getpid()):
        return False
    attachments = []
    if artifact_paths:
        from .chat_attachments import (
            remove_chat_attachments,
            store_generated_chat_attachments,
        )

        attachments = store_generated_chat_attachments(
            Path(path).parent,
            list(artifact_paths),
        )
    try:
        record = append_chat_message(
            path,
            role="assistant",
            content=content,
            attachments=[item.public() for item in attachments],
            performance=performance,
            rolling=bool(os.environ.get(COLLABORATOR_MODE_ENV)),
        )
    except BaseException:
        if attachments:
            remove_chat_attachments(attachments)
        raise
    return record


def append_progress_message_from_environment(content: str) -> bool:
    """Append one redacted, single-line progress update from the primary PID.

    Progress is intentionally presentation-only.  It never carries tool
    parameters, command lines, raw tool output, prompts, or model reasoning.
    """

    path = os.environ.get(CHAT_TRANSCRIPT_ENV, "")
    expected_pid = os.environ.get(CHAT_WRITER_PID_ENV, "")
    if not path or expected_pid != str(os.getpid()):
        return False
    rendered = sanitize_summary(content, max_chars=500)
    if not rendered:
        return False
    append_chat_message(
        path,
        role="progress",
        content=rendered,
        rolling=bool(os.environ.get(COLLABORATOR_MODE_ENV)),
    )
    return True


def append_plan_message_from_environment(content: str) -> bool:
    """Publish a bounded, redacted checklist from the exact primary PID.

    Plan records are user-visible coordination state, not model reasoning. Empty
    content is a deliberate marker that clears the checklist for a new request.
    Keeping that marker in the same append-only transcript preserves ordering
    with user messages and live tool progress.
    """

    path = os.environ.get(CHAT_TRANSCRIPT_ENV, "")
    expected_pid = os.environ.get(CHAT_WRITER_PID_ENV, "")
    if not path or expected_pid != str(os.getpid()):
        return False

    lines: list[str] = []
    normalized = str(content or "").replace("\r\n", "\n").replace("\r", "\n")
    for raw_line in normalized.splitlines():
        rendered = sanitize_summary(raw_line, max_chars=400)
        if rendered:
            lines.append(rendered)
        if len(lines) >= 32:
            break
    checklist = "\n".join(lines)
    if len(checklist) > 6_000:
        checklist = checklist[:5_999].rstrip() + "…"
    append_chat_message(
        path,
        role="plan",
        content=checklist,
        rolling=bool(os.environ.get(COLLABORATOR_MODE_ENV)),
    )
    return True
