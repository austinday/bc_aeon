#!/usr/bin/env python3
"""Extract a hash-audited tensor subset from pinned Hugging Face shards.

This utility reads only safetensors headers and the exact byte ranges for the
selected tensors.  It is used by the Qwen3.8-Flash-Next build to retain the
official BF16 ``mtp.*`` tensors without downloading unrelated 360 GB source
weights, and to retain the tiny ModelOpt calibration tensors from a pinned
reference export.  It never accepts a moving revision and never reads a Hub
credential: both source repositories used by the release pipeline are public.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import time
from typing import Any, Iterable
from urllib.parse import quote, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SCHEMA_VERSION = "aeon-hf-safetensors-subset-v1"
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REVISION_RE = re.compile(r"^[a-f0-9]{40}$")
_SAFE_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class SubsetError(RuntimeError):
    """Pinned remote tensor data or the local destination failed validation."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_index(path: Path) -> dict[str, str]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
        or metadata.st_nlink != 1
        or not 0 < metadata.st_size <= 64 * 1024 * 1024
    ):
        raise SubsetError("local safetensors index is unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubsetError("local safetensors index is malformed") from exc
    weight_map = value.get("weight_map") if isinstance(value, dict) else None
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or not all(
            isinstance(name, str)
            and name
            and len(name) <= 1024
            and isinstance(shard, str)
            and _SAFE_FILE_RE.fullmatch(shard)
            for name, shard in weight_map.items()
        )
    ):
        raise SubsetError("local safetensors weight map is malformed")
    return dict(weight_map)


def _session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    result = requests.Session()
    result.trust_env = False
    result.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))
    result.headers.update(
        {
            "Accept-Encoding": "identity",
            "User-Agent": "aeon-qwen38-flash-next-subset/1",
        }
    )
    return result


def _source_url(repo: str, revision: str, shard: str) -> str:
    return (
        f"https://huggingface.co/{quote(repo, safe='/')}/resolve/"
        f"{revision}/{quote(shard, safe='')}"
    )


def _validated_range_response(
    session: requests.Session,
    url: str,
    start: int,
    end: int,
    *,
    timeout: tuple[float, float] = (20.0, 120.0),
) -> requests.Response:
    if start < 0 or end < start:
        raise SubsetError("remote byte range is invalid")
    response = session.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        stream=True,
        timeout=timeout,
        allow_redirects=True,
    )
    final = urlparse(response.url)
    expected_length = end - start + 1
    expected_prefix = f"bytes {start}-{end}/"
    if (
        response.status_code != 206
        or final.scheme != "https"
        or not final.hostname
        or response.headers.get("Content-Range", "").startswith(expected_prefix) is False
        or response.headers.get("Content-Length") != str(expected_length)
    ):
        response.close()
        raise SubsetError("Hub did not honor the exact bounded byte range")
    return response


def _read_exact_range(
    session: requests.Session, url: str, start: int, end: int, maximum: int
) -> bytes:
    expected = end - start + 1
    if expected <= 0 or expected > maximum:
        raise SubsetError("remote metadata range exceeds its bound")
    response = _validated_range_response(session, url, start, end)
    try:
        value = response.content
    finally:
        response.close()
    if len(value) != expected:
        raise SubsetError("Hub returned a truncated metadata range")
    return value


def _remote_header(
    session: requests.Session, repo: str, revision: str, shard: str
) -> tuple[int, dict[str, Any], int]:
    url = _source_url(repo, revision, shard)
    raw_length = _read_exact_range(session, url, 0, 7, 8)
    header_length = struct.unpack("<Q", raw_length)[0]
    if not 2 <= header_length <= 256 * 1024 * 1024 or header_length % 8:
        raise SubsetError(f"remote safetensors header is invalid: {shard}")
    raw_header = _read_exact_range(
        session, url, 8, 8 + header_length - 1, 256 * 1024 * 1024
    )
    try:
        header = json.loads(raw_header.rstrip(b" "))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubsetError(f"remote safetensors header is malformed: {shard}") from exc
    if not isinstance(header, dict):
        raise SubsetError(f"remote safetensors header is not an object: {shard}")
    return 8 + header_length, header, header_length


def _tensor_entry(name: str, value: Any) -> tuple[str, tuple[int, ...], int, int]:
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
        raise SubsetError(f"remote tensor metadata is malformed: {name}")
    dtype = value.get("dtype")
    shape = value.get("shape")
    offsets = value.get("data_offsets")
    if (
        dtype not in _DTYPE_BYTES
        or not isinstance(shape, list)
        or not all(type(item) is int and item >= 0 for item in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(type(item) is int and item >= 0 for item in offsets)
        or offsets[1] < offsets[0]
    ):
        raise SubsetError(f"remote tensor metadata is malformed: {name}")
    expected = math.prod(shape) * _DTYPE_BYTES[dtype]
    if offsets[1] - offsets[0] != expected:
        raise SubsetError(f"remote tensor byte length is inconsistent: {name}")
    return dtype, tuple(shape), offsets[0], offsets[1]


def _selected_names(weight_map: dict[str, str], patterns: Iterable[str]) -> list[str]:
    try:
        compiled = [re.compile(item) for item in patterns]
    except re.error as exc:
        raise SubsetError("tensor selection regex is invalid") from exc
    if not compiled:
        raise SubsetError("at least one tensor selection regex is required")
    selected = sorted(
        name for name in weight_map if any(pattern.search(name) for pattern in compiled)
    )
    if not selected:
        raise SubsetError("tensor selection matched nothing")
    return selected


def _output_header(entries: list[dict[str, Any]], metadata: dict[str, str]) -> bytes:
    offset = 0
    value: dict[str, Any] = {"__metadata__": metadata}
    for entry in entries:
        size = entry["source_end"] - entry["source_start"]
        value[entry["name"]] = {
            "dtype": entry["dtype"],
            "shape": list(entry["shape"]),
            "data_offsets": [offset, offset + size],
        }
        entry["output_start"] = offset
        entry["output_end"] = offset + size
        offset += size
    raw = json.dumps(
        value, sort_keys=False, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    padding = (-len(raw)) % 8
    return raw + b" " * padding


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        payload = json.dumps(
            value, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8") + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def extract(
    *,
    repo: str,
    revision: str,
    index_path: Path,
    patterns: list[str],
    output: Path,
    manifest: Path,
    maximum_output_bytes: int,
) -> dict[str, Any]:
    if _REPO_RE.fullmatch(repo) is None or _REVISION_RE.fullmatch(revision) is None:
        raise SubsetError("source repository or immutable revision is invalid")
    if maximum_output_bytes <= 0:
        raise SubsetError("output byte bound must be positive")
    if output.exists() or output.is_symlink() or manifest.exists() or manifest.is_symlink():
        raise SubsetError("output or manifest already exists")
    if output.parent != manifest.parent:
        raise SubsetError("output and manifest must share one private directory")
    parent = output.parent
    metadata = parent.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise SubsetError("destination directory is not private and owned")

    weight_map = _read_index(index_path)
    selected = _selected_names(weight_map, patterns)
    by_shard: dict[str, list[str]] = {}
    for name in selected:
        by_shard.setdefault(weight_map[name], []).append(name)

    session = _session()
    entries: list[dict[str, Any]] = []
    shard_headers: dict[str, int] = {}
    try:
        for shard in sorted(by_shard):
            data_start, header, header_length = _remote_header(
                session, repo, revision, shard
            )
            shard_headers[shard] = header_length
            for name in by_shard[shard]:
                if name not in header:
                    raise SubsetError(f"index tensor is absent from remote shard: {name}")
                dtype, shape, start, end = _tensor_entry(name, header[name])
                entries.append(
                    {
                        "name": name,
                        "shard": shard,
                        "dtype": dtype,
                        "shape": shape,
                        "source_start": data_start + start,
                        "source_end": data_start + end,
                    }
                )
    except BaseException:
        session.close()
        raise

    entries.sort(key=lambda item: (item["shard"], item["source_start"], item["name"]))
    tensor_bytes = sum(item["source_end"] - item["source_start"] for item in entries)
    if tensor_bytes <= 0 or tensor_bytes > maximum_output_bytes:
        session.close()
        raise SubsetError("selected tensor bytes exceed the declared output bound")
    header = _output_header(
        entries,
        {
            "schema_version": SCHEMA_VERSION,
            "source_repo": repo,
            "source_revision": revision,
        },
    )
    total_bytes = 8 + len(header) + tensor_bytes
    if total_bytes > maximum_output_bytes:
        session.close()
        raise SubsetError("safetensors output exceeds the declared byte bound")

    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    descriptor = os.open(
        partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    file_hash = hashlib.sha256()
    tensor_hashes: dict[str, str] = {}
    started = time.time()
    try:
        prefix = struct.pack("<Q", len(header)) + header
        prefix_view = memoryview(prefix)
        while prefix_view:
            written = os.write(descriptor, prefix_view)
            if written <= 0:
                raise SubsetError("local safetensors header write was incomplete")
            prefix_view = prefix_view[written:]
        file_hash.update(prefix)
        index = 0
        while index < len(entries):
            first = entries[index]
            span_entries = [first]
            span_end = first["source_end"]
            index += 1
            while (
                index < len(entries)
                and entries[index]["shard"] == first["shard"]
                and entries[index]["source_start"] == span_end
            ):
                span_entries.append(entries[index])
                span_end = entries[index]["source_end"]
                index += 1
            response = _validated_range_response(
                session,
                _source_url(repo, revision, first["shard"]),
                first["source_start"],
                span_end - 1,
            )
            span_written = 0
            entry_index = 0
            entry_hash = hashlib.sha256()
            entry_remaining = (
                span_entries[0]["source_end"] - span_entries[0]["source_start"]
            )
            try:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    view = memoryview(chunk)
                    while view:
                        if entry_index >= len(span_entries):
                            raise SubsetError("Hub returned bytes beyond the requested span")
                        take = min(len(view), entry_remaining)
                        piece = view[:take]
                        written = os.write(descriptor, piece)
                        if written != take:
                            raise SubsetError("local subset write was incomplete")
                        file_hash.update(piece)
                        entry_hash.update(piece)
                        span_written += take
                        entry_remaining -= take
                        view = view[take:]
                        if entry_remaining == 0:
                            tensor_hashes[span_entries[entry_index]["name"]] = entry_hash.hexdigest()
                            entry_index += 1
                            if entry_index < len(span_entries):
                                entry_hash = hashlib.sha256()
                                entry_remaining = (
                                    span_entries[entry_index]["source_end"]
                                    - span_entries[entry_index]["source_start"]
                                )
            finally:
                response.close()
            if span_written != span_end - first["source_start"] or entry_index != len(span_entries):
                raise SubsetError("Hub returned a truncated tensor span")
            print(
                json.dumps(
                    {
                        "event": "remote_span_copied",
                        "shard": first["shard"],
                        "bytes": span_written,
                        "tensors": len(span_entries),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
        session.close()

    if partial.stat().st_size != total_bytes or len(tensor_hashes) != len(entries):
        raise SubsetError("completed subset size or tensor count is inconsistent")
    os.replace(partial, output)
    result = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "source": {"repo": repo, "revision": revision},
        "index_path": str(index_path),
        "selection_regex": patterns,
        "shards": sorted(by_shard),
        "shard_header_lengths": shard_headers,
        "tensor_count": len(entries),
        "tensor_bytes": tensor_bytes,
        "output_bytes": total_bytes,
        "output_file": output.name,
        "output_sha256": file_hash.hexdigest(),
        "tensor_sha256": tensor_hashes,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(manifest, result)
    parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--include-regex", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maximum-output-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    result = extract(
        repo=args.repo,
        revision=args.revision,
        index_path=args.index.resolve(),
        patterns=list(args.include_regex),
        output=args.output.resolve(),
        manifest=args.manifest.resolve(),
        maximum_output_bytes=args.maximum_output_bytes,
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output.resolve()),
                "sha256": result["output_sha256"],
                "tensor_count": result["tensor_count"],
                "bytes": result["output_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SubsetError as exc:
        print(f"subset extraction refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
