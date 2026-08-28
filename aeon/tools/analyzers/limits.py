"""Small, deterministic resource guards for local file inspection.

The analyzer runs in the agent process, so parser convenience APIs that read an
entire file, line, archive, or decoded object are unsafe without an explicit
contract.  Helpers here validate regular-file identity and enforce byte/row/line
bounds before returning data to a parser.  They intentionally do not launch any
subprocess or inspect fleet/GPU state.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import stat
from typing import BinaryIO


class ResourceLimitError(ValueError):
    """A local input exceeded its reviewed in-process parsing contract."""


def regular_file_stat(path: str) -> os.stat_result:
    try:
        result = os.stat(path)
    except OSError as exc:
        raise ResourceLimitError(f"could not inspect input file: {exc}") from exc
    if not stat.S_ISREG(result.st_mode):
        raise ResourceLimitError("input must be a regular file")
    return result


def require_max_file_bytes(path: str, max_bytes: int, *, label: str) -> int:
    result = regular_file_stat(path)
    if result.st_size > max_bytes:
        raise ResourceLimitError(
            f"{label} is {result.st_size:,} bytes; limit is {max_bytes:,} bytes"
        )
    return int(result.st_size)


def read_bounded_bytes(path: str, max_bytes: int, *, label: str) -> bytes:
    """Read a regular file only when its current size fits ``max_bytes``.

    The second length check closes the common stat/open growth race.  At most one
    sentinel byte beyond the declared limit is read.
    """

    require_max_file_bytes(path, max_bytes, label=label)
    with open(path, "rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ResourceLimitError(
            f"{label} grew beyond the {max_bytes:,}-byte limit while reading"
        )
    return data


def read_text_prefix(path: str, max_bytes: int) -> tuple[str, bool]:
    """Return a UTF-8-replaced byte-bounded prefix and whether data was omitted."""

    result = regular_file_stat(path)
    to_read = min(int(result.st_size), max_bytes)
    with open(path, "rb") as handle:
        data = handle.read(to_read)
    return data.decode("utf-8", errors="replace"), result.st_size > len(data)


def bounded_binary_readline(handle: BinaryIO, max_line_bytes: int) -> bytes:
    """Read one line without ever materializing an attacker-sized row."""

    line = handle.readline(max_line_bytes + 1)
    if len(line) > max_line_bytes:
        raise ResourceLimitError(
            f"input row exceeds the {max_line_bytes:,}-byte line limit"
        )
    return line


@dataclass(frozen=True)
class LineScan:
    row_count: int
    sampled_lines: tuple[str, ...]
    truncated: bool
    bytes_scanned: int


def scan_text_lines(
    path: str,
    *,
    max_rows: int,
    max_bytes: int,
    max_line_bytes: int,
    sample_rows: int = 0,
) -> LineScan:
    """Scan text with exact row, byte, and single-line ceilings.

    ``truncated`` is truthful whenever a byte or row ceiling prevents reaching
    EOF.  A single overlong row is refused instead of being treated as multiple
    records, which is important before handing samples to CSV/JSON parsers.
    """

    file_stat = regular_file_stat(path)
    rows = 0
    scanned = 0
    samples: list[str] = []
    truncated = False
    with open(path, "rb") as handle:
        while rows < max_rows and scanned < max_bytes:
            remaining = max_bytes - scanned
            if remaining < max_line_bytes:
                line = handle.readline(remaining + 1)
                if len(line) > remaining:
                    truncated = True
                    break
            else:
                line = bounded_binary_readline(handle, max_line_bytes)
            if not line:
                break
            scanned += len(line)
            if not line.endswith((b"\n", b"\r")) and handle.tell() < file_stat.st_size:
                # Byte budget ended inside a row; omit the fragment entirely.
                truncated = True
                break
            rows += 1
            if len(samples) < sample_rows:
                samples.append(line.decode("utf-8", errors="replace"))
        if handle.tell() < file_stat.st_size:
            truncated = True

    return LineScan(
        row_count=rows,
        sampled_lines=tuple(samples),
        truncated=truncated,
        bytes_scanned=scanned,
    )


def limit_error(exc: BaseException) -> dict[str, str]:
    return {
        "summary_type": "error",
        "error_message": f"Resource limit: {exc}",
    }
