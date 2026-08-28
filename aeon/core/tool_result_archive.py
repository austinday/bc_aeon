"""Owner-private, bounded archive for oversized tool results.

The archive is evidence storage, not model context.  Model-visible receipts carry
only a small preview plus an opaque request-scoped reference.  Reads are bounded,
integrity checked, and never accept filesystem paths from the model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from typing import Any


MAX_ARCHIVED_RESULT_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_REQUEST_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_REQUEST_FILES = 32
MAX_ARCHIVE_INSTANCE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_INSTANCE_FILES = 256
MAX_INSPECTION_CHARS = 3_000
MIN_INSPECTION_CHARS = 256
MAX_SEARCH_QUERY_CHARS = 256
MAX_SEARCH_MATCHES = 10

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_REFERENCE_RE = re.compile(r"^tr_[0-9a-f]{32}_[0-9a-f]{16}$")
_ARCHIVE_FILE_RE = re.compile(
    r"^(?P<request>[A-Za-z0-9_-]{1,64})--"
    r"(?P<reference>tr_[0-9a-f]{32}_[0-9a-f]{16})\.txt$"
)


class ToolResultArchiveError(RuntimeError):
    """The archive could not safely preserve or retrieve a result."""


class ToolResultArchiveCapacityError(ToolResultArchiveError):
    """A bounded archive quota refused additional evidence."""


@dataclass(frozen=True)
class ArchivedToolResult:
    reference: str
    sha256: str
    chars: int
    bytes: int


def render_tool_result_content(value: Any) -> str:
    """Render a tool's raw value deterministically without executing it."""

    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        )
    except (TypeError, ValueError):
        return str(value)


class ToolResultArchive:
    """Append-only result storage beneath one Aeon instance state directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._refs_by_digest: dict[tuple[str, str], ArchivedToolResult] = {}

    @staticmethod
    def _validate_request_id(request_id: str) -> str:
        value = str(request_id or "")
        if not _REQUEST_ID_RE.fullmatch(value):
            raise ToolResultArchiveError("tool-result request identity is invalid")
        return value

    @staticmethod
    def _validate_reference(reference: str) -> str:
        value = str(reference or "")
        if not _REFERENCE_RE.fullmatch(value):
            raise ToolResultArchiveError("tool-result reference is invalid")
        return value

    def _open_directory(self, *, create: bool) -> int:
        if not self.root.is_absolute():
            raise ToolResultArchiveError("tool-result archive root must be absolute")
        if create:
            try:
                self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
            except FileExistsError:
                pass
        try:
            metadata = self.root.lstat()
        except OSError as exc:
            raise ToolResultArchiveError("tool-result archive is unavailable") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ToolResultArchiveError("tool-result archive is not owner-private")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_CLOEXEC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.root, flags)
        except OSError as exc:
            raise ToolResultArchiveError("tool-result archive is unavailable") from exc
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o700
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            os.close(descriptor)
            raise ToolResultArchiveError("tool-result archive identity changed")
        return descriptor

    @staticmethod
    def _inventory(directory_fd: int, request_id: str) -> tuple[int, int, int, int]:
        total_files = 0
        total_bytes = 0
        request_files = 0
        request_bytes = 0
        try:
            names = os.listdir(directory_fd)
        except OSError as exc:
            raise ToolResultArchiveError("tool-result archive cannot be inventoried") from exc
        for name in names:
            match = _ARCHIVE_FILE_RE.fullmatch(name)
            if match is None:
                raise ToolResultArchiveError(
                    "tool-result archive contains an unrecognized entry"
                )
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise ToolResultArchiveError(
                    "tool-result archive entry cannot be validated"
                ) from exc
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size > MAX_ARCHIVED_RESULT_BYTES
            ):
                raise ToolResultArchiveError(
                    "tool-result archive entry is not an owner-private bounded file"
                )
            total_files += 1
            total_bytes += int(metadata.st_size)
            if match.group("request") == request_id:
                request_files += 1
                request_bytes += int(metadata.st_size)
        return total_files, total_bytes, request_files, request_bytes

    @staticmethod
    def _filename(request_id: str, reference: str) -> str:
        return f"{request_id}--{reference}.txt"

    def persist(self, *, request_id: str, content: str) -> ArchivedToolResult:
        """Persist one complete UTF-8 result, refusing rather than exceeding quotas."""

        request_id = self._validate_request_id(request_id)
        text = str(content)
        payload = text.encode("utf-8")
        if len(payload) > MAX_ARCHIVED_RESULT_BYTES:
            raise ToolResultArchiveCapacityError(
                "tool result exceeds the per-result archive limit"
            )
        digest = hashlib.sha256(payload).hexdigest()
        key = (request_id, digest)
        with self._lock:
            existing = self._refs_by_digest.get(key)
            if existing is not None:
                try:
                    if self._read_content(
                        request_id, existing.reference, existing.sha256
                    ) == text:
                        return existing
                except ToolResultArchiveError:
                    self._refs_by_digest.pop(key, None)

            directory_fd = self._open_directory(create=True)
            try:
                total_files, total_bytes, request_files, request_bytes = self._inventory(
                    directory_fd, request_id
                )
                if total_files >= MAX_ARCHIVE_INSTANCE_FILES:
                    raise ToolResultArchiveCapacityError(
                        "tool-result instance file quota is exhausted"
                    )
                if request_files >= MAX_ARCHIVE_REQUEST_FILES:
                    raise ToolResultArchiveCapacityError(
                        "tool-result request file quota is exhausted"
                    )
                if total_bytes + len(payload) > MAX_ARCHIVE_INSTANCE_BYTES:
                    raise ToolResultArchiveCapacityError(
                        "tool-result instance byte quota is exhausted"
                    )
                if request_bytes + len(payload) > MAX_ARCHIVE_REQUEST_BYTES:
                    raise ToolResultArchiveCapacityError(
                        "tool-result request byte quota is exhausted"
                    )

                reference = f"tr_{secrets.token_hex(16)}_{digest[:16]}"
                filename = self._filename(request_id, reference)
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
                flags |= getattr(os, "O_NOFOLLOW", 0)
                try:
                    descriptor = os.open(filename, flags, 0o600, dir_fd=directory_fd)
                except OSError as exc:
                    raise ToolResultArchiveError(
                        "tool-result archive file could not be created"
                    ) from exc
                try:
                    view = memoryview(payload)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise ToolResultArchiveError(
                                "tool-result archive write made no progress"
                            )
                        view = view[written:]
                    os.fsync(descriptor)
                    metadata = os.fstat(descriptor)
                    if (
                        metadata.st_uid != os.geteuid()
                        or metadata.st_nlink != 1
                        or stat.S_IMODE(metadata.st_mode) != 0o600
                        or metadata.st_size != len(payload)
                    ):
                        raise ToolResultArchiveError(
                            "tool-result archive write could not be verified"
                        )
                finally:
                    os.close(descriptor)
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

            receipt = ArchivedToolResult(
                reference=reference,
                sha256=digest,
                chars=len(text),
                bytes=len(payload),
            )
            self._refs_by_digest[key] = receipt
            return receipt

    def _read_content(
        self, request_id: str, reference: str, expected_sha256: str
    ) -> str:
        request_id = self._validate_request_id(request_id)
        reference = self._validate_reference(reference)
        expected_sha256 = str(expected_sha256 or "")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ToolResultArchiveError("tool-result digest is invalid")
        directory_fd = self._open_directory(create=False)
        try:
            filename = self._filename(request_id, reference)
            flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(filename, flags, dir_fd=directory_fd)
            except OSError as exc:
                raise ToolResultArchiveError(
                    "tool-result reference is unavailable for this request"
                ) from exc
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_size > MAX_ARCHIVED_RESULT_BYTES
                ):
                    raise ToolResultArchiveError(
                        "tool-result archive file failed validation"
                    )
                chunks: list[bytes] = []
                remaining = MAX_ARCHIVED_RESULT_BYTES + 1
                while remaining > 0:
                    chunk = os.read(descriptor, min(65_536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                payload = b"".join(chunks)
                if len(payload) > MAX_ARCHIVED_RESULT_BYTES:
                    raise ToolResultArchiveError(
                        "tool-result archive file exceeds its read bound"
                    )
                final = os.fstat(descriptor)
                if (
                    final.st_dev,
                    final.st_ino,
                    final.st_size,
                    final.st_mtime_ns,
                ) != (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                ):
                    raise ToolResultArchiveError(
                        "tool-result archive file changed during read"
                    )
            finally:
                os.close(descriptor)
        finally:
            os.close(directory_fd)
        digest = hashlib.sha256(payload).hexdigest()
        if (
            digest != expected_sha256
            or digest[:16] != reference.rsplit("_", 1)[-1]
        ):
            raise ToolResultArchiveError("tool-result archive integrity check failed")
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolResultArchiveError(
                "tool-result archive is not valid UTF-8"
            ) from exc

    def inspect(
        self,
        *,
        request_id: str,
        reference: str,
        expected_sha256: str,
        query: str = "",
        offset: int = 0,
        limit: int = 2_000,
    ) -> dict[str, Any]:
        """Return a bounded page or literal-search window from one archived result."""

        reference = self._validate_reference(reference)
        try:
            offset = int(offset)
            limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ToolResultArchiveError("offset and limit must be integers") from exc
        if offset < 0:
            raise ToolResultArchiveError("offset must be non-negative")
        limit = max(MIN_INSPECTION_CHARS, min(MAX_INSPECTION_CHARS, limit))
        content = self._read_content(request_id, reference, expected_sha256)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        needle = str(query or "").strip()
        if not needle:
            start = min(offset, len(content))
            end = min(len(content), start + limit)
            return {
                "reference": reference,
                "sha256": digest,
                "mode": "page",
                "offset": start,
                "next_offset": end if end < len(content) else None,
                "total_chars": len(content),
                "eof": end >= len(content),
                "content": content[start:end],
            }

        if len(needle) > MAX_SEARCH_QUERY_CHARS:
            raise ToolResultArchiveError(
                f"query must be at most {MAX_SEARCH_QUERY_CHARS} characters"
            )
        pattern = re.compile(re.escape(needle), re.IGNORECASE)
        matches: list[dict[str, Any]] = []
        more_matches_possible = False
        context_chars = max(80, min(240, limit // 4))
        for match in pattern.finditer(content):
            start = max(0, match.start() - context_chars)
            end = min(len(content), match.end() + context_chars)
            candidate = {
                "start": match.start(),
                "end": match.end(),
                "snippet": content[start:end],
            }
            projected = {
                "reference": reference,
                "sha256": digest,
                "mode": "search",
                "query": needle,
                "total_chars": len(content),
                "matches": [*matches, candidate],
            }
            if len(json.dumps(projected, ensure_ascii=False)) > limit:
                more_matches_possible = True
                break
            matches.append(candidate)
            if len(matches) >= MAX_SEARCH_MATCHES:
                more_matches_possible = True
                break
        return {
            "reference": reference,
            "sha256": digest,
            "mode": "search",
            "query": needle,
            "total_chars": len(content),
            "matches": matches,
            "more_matches_possible": more_matches_possible,
        }
