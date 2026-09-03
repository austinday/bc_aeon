from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from aeon.core.utils.io import read_bounded_fd, write_all_fd


def test_bounded_fd_reader_handles_legal_short_reads() -> None:
    payload = b"complete owner control document"
    with tempfile.TemporaryFile() as stream:
        stream.write(payload)
        stream.flush()
        stream.seek(0)
        real_read = os.read

        def short_read(descriptor: int, size: int) -> bytes:
            return real_read(descriptor, min(size, 3))

        with patch("aeon.core.utils.io.os.read", side_effect=short_read):
            assert read_bounded_fd(stream.fileno(), len(payload)) == payload


def test_bounded_fd_reader_returns_only_one_oversize_byte() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "oversized"
        path.write_bytes(b"x" * 100)
        descriptor = os.open(path, os.O_RDONLY)
        try:
            assert read_bounded_fd(descriptor, 12) == b"x" * 13
        finally:
            os.close(descriptor)


def test_write_all_retries_partial_writes() -> None:
    target = bytearray()

    def partial_write(_descriptor: int, payload: memoryview) -> int:
        chunk = bytes(payload[:2])
        target.extend(chunk)
        return len(chunk)

    with patch("aeon.core.utils.io.os.write", side_effect=partial_write):
        assert write_all_fd(12, b"abcdef") == 6
    assert bytes(target) == b"abcdef"


def test_write_all_rejects_zero_progress() -> None:
    with patch("aeon.core.utils.io.os.write", return_value=0):
        try:
            write_all_fd(12, b"x")
        except OSError:
            pass
        else:
            raise AssertionError("a stalled descriptor must fail")
