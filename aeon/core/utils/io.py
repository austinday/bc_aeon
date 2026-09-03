"""Small, dependency-free helpers for bounded low-level I/O."""

from __future__ import annotations

import os


def read_bounded_fd(descriptor: int, maximum_bytes: int) -> bytes:
    """Read through EOF, returning at most one byte beyond the stated limit.

    ``os.read`` may legally return fewer bytes than requested before EOF.  A
    one-shot read can therefore reject a complete control file or receipt even
    though the file itself is valid.
    """

    limit = int(maximum_bytes)
    if limit < 0:
        raise ValueError("maximum_bytes must be non-negative")
    chunks: list[bytes] = []
    remaining = limit + 1
    while remaining > 0:
        chunk = os.read(descriptor, min(65_536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def write_all_fd(descriptor: int, payload: bytes | bytearray | memoryview) -> int:
    """Write an entire bytes-like payload or fail if the descriptor stalls."""

    view = memoryview(payload)
    total = len(view)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("file descriptor made no write progress")
        view = view[written:]
    return total


__all__ = ("read_bounded_fd", "write_all_fd")
