import bz2
import gzip
import struct
import tarfile
from typing import Any, BinaryIO, Dict
import zipfile

from .utility import summarize_opaque
from ..limits import (
    ResourceLimitError,
    limit_error,
    require_max_file_bytes,
)


_ZIP_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP_EOCD_SIZE = 22
_ZIP_MAX_COMMENT = 65_535


class _LimitedReader:
    """A read-only wrapper that caps bytes emitted by a decompressor."""

    def __init__(self, source: BinaryIO, limit: int):
        self._source = source
        self._limit = limit
        self._read = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self._limit - self._read
        if remaining < 0:
            raise ResourceLimitError(
                f"archive decompression exceeded {self._limit:,} bytes"
            )
        request = remaining + 1 if size < 0 else min(size, remaining + 1)
        data = self._source.read(request)
        self._read += len(data)
        if self._read > self._limit:
            raise ResourceLimitError(
                f"archive decompression exceeded {self._limit:,} bytes"
            )
        return data


def _validate_name(analyzer, name: str) -> None:
    name_bytes = len(str(name).encode("utf-8", errors="replace"))
    if name_bytes > analyzer.MAX_ARCHIVE_NAME_BYTES:
        raise ResourceLimitError(
            f"archive member name is {name_bytes:,} bytes; limit is "
            f"{analyzer.MAX_ARCHIVE_NAME_BYTES:,} bytes"
        )


def _validate_expansion(analyzer, *, compressed_bytes: int, expanded_bytes: int) -> None:
    if expanded_bytes > analyzer.MAX_ARCHIVE_EXPANDED_BYTES:
        raise ResourceLimitError(
            f"archive advertises {expanded_bytes:,} expanded bytes; limit is "
            f"{analyzer.MAX_ARCHIVE_EXPANDED_BYTES:,} bytes"
        )
    ratio = expanded_bytes / max(1, compressed_bytes)
    if ratio > analyzer.MAX_ARCHIVE_EXPANSION_RATIO:
        raise ResourceLimitError(
            f"archive expansion ratio {ratio:.1f}:1 exceeds the "
            f"{analyzer.MAX_ARCHIVE_EXPANSION_RATIO:,}:1 limit"
        )


def _zip_preflight(analyzer, file_size: int) -> int:
    """Read the bounded EOCD and reject member/central-dir bombs before ZipFile."""

    tail_size = min(file_size, _ZIP_EOCD_SIZE + _ZIP_MAX_COMMENT)
    with open(analyzer.file_path, "rb") as handle:
        handle.seek(file_size - tail_size)
        tail = handle.read(tail_size)
    offset = tail.rfind(_ZIP_EOCD_SIGNATURE)
    if offset < 0 or len(tail) - offset < _ZIP_EOCD_SIZE:
        raise zipfile.BadZipFile("end-of-central-directory record not found")
    (
        _signature,
        disk_number,
        directory_disk,
        disk_entries,
        total_entries,
        directory_size,
        _directory_offset,
        comment_length,
    ) = struct.unpack_from("<4s4H2LH", tail, offset)
    if offset + _ZIP_EOCD_SIZE + comment_length > len(tail):
        raise zipfile.BadZipFile("truncated end-of-central-directory record")
    if disk_number or directory_disk or disk_entries != total_entries:
        raise ResourceLimitError("multi-disk ZIP archives are not supported")
    if total_entries == 0xFFFF or directory_size == 0xFFFFFFFF:
        raise ResourceLimitError("ZIP64 metadata is outside the bounded archive inspector")
    if total_entries > analyzer.MAX_ARCHIVE_MEMBERS:
        raise ResourceLimitError(
            f"archive has {total_entries:,} members; limit is "
            f"{analyzer.MAX_ARCHIVE_MEMBERS:,} members"
        )
    if directory_size > analyzer.MAX_ARCHIVE_CENTRAL_DIRECTORY_BYTES:
        raise ResourceLimitError(
            f"ZIP central directory is {directory_size:,} bytes; limit is "
            f"{analyzer.MAX_ARCHIVE_CENTRAL_DIRECTORY_BYTES:,} bytes"
        )
    return int(total_entries)


def _summarize_zip(analyzer, file_size: int) -> Dict[str, Any]:
    expected_count = _zip_preflight(analyzer, file_size)
    with zipfile.ZipFile(analyzer.file_path, "r") as archive:
        members = archive.infolist()
        if len(members) != expected_count:
            raise ResourceLimitError("ZIP member count changed during inspection")
        expanded = 0
        compressed = 0
        names: list[str] = []
        for member in members:
            _validate_name(analyzer, member.filename)
            expanded += int(member.file_size)
            compressed += int(member.compress_size)
            _validate_expansion(
                analyzer,
                compressed_bytes=max(compressed, 1),
                expanded_bytes=expanded,
            )
            if len(names) < analyzer.MAX_ARCHIVE_LIST_FILES:
                names.append(member.filename)
    return _archive_summary(analyzer, len(members), names)


def _tar_stream(analyzer, raw: BinaryIO) -> BinaryIO:
    if analyzer.file_extension in {".gz", ".tgz"}:
        decoded: BinaryIO = gzip.GzipFile(fileobj=raw, mode="rb")
    elif analyzer.file_extension == ".bz2":
        decoded = bz2.BZ2File(raw, mode="rb")
    else:
        decoded = raw
    return _LimitedReader(decoded, analyzer.MAX_ARCHIVE_STREAM_BYTES)


def _summarize_tar(analyzer, file_size: int) -> Dict[str, Any]:
    names: list[str] = []
    member_count = 0
    expanded = 0
    with open(analyzer.file_path, "rb") as raw:
        stream = _tar_stream(analyzer, raw)
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                member_count += 1
                if member_count > analyzer.MAX_ARCHIVE_MEMBERS:
                    raise ResourceLimitError(
                        f"archive has more than {analyzer.MAX_ARCHIVE_MEMBERS:,} members"
                    )
                _validate_name(analyzer, member.name)
                expanded += max(0, int(member.size))
                _validate_expansion(
                    analyzer,
                    compressed_bytes=file_size,
                    expanded_bytes=expanded,
                )
                if len(names) < analyzer.MAX_ARCHIVE_LIST_FILES:
                    names.append(member.name)
    return _archive_summary(analyzer, member_count, names)


def _archive_summary(analyzer, count: int, names: list[str]) -> Dict[str, Any]:
    file_list = list(names)
    if count > len(file_list):
        file_list.append(f"... ({count - len(file_list)} more files)")
    return {
        "summary_type": "archive_contents",
        "file_format": analyzer.file_extension.lstrip('.'),
        "file_count": count,
        "file_list": file_list,
    }

def summarize_archive(analyzer) -> Dict[str, Any]:
    try:
        file_size = require_max_file_bytes(
            analyzer.file_path,
            analyzer.MAX_ARCHIVE_INPUT_BYTES,
            label="archive",
        )
        if analyzer.file_extension == ".zip":
            return _summarize_zip(analyzer, file_size)
        if analyzer.file_extension in {".tar", ".tgz", ".gz", ".bz2"}:
            try:
                return _summarize_tar(analyzer, file_size)
            except tarfile.ReadError:
                # Plain .gz/.bz2 files are streams, not necessarily tar archives.
                return summarize_opaque(analyzer)
        # RAR/7z need optional external decoders; do not invoke them from the
        # agent process or pretend their output is bounded.
        return summarize_opaque(analyzer)
    except ResourceLimitError as e:
        return limit_error(e)
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as e:
        return {
            "summary_type": "error",
            "error_message": f"Could not inspect archive: {e}",
        }
