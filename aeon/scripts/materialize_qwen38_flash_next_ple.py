#!/usr/bin/env python3
"""Resolve and fail-closed materialize the omitted official FP8 PLE shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import struct
from typing import Any, Iterable, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request


SCHEMA_VERSION = "aeon-qwen38-flash-next-thin-ple-materialization-v1"
COMPLETION_SCHEMA_VERSION = "aeon-qwen38-flash-next-ple-materialization-completion-v1"
MANIFEST_NAME = "PLE_MATERIALIZATION.json"
OFFICIAL_REPO = "Qwen/Qwen3.8-Flash-Next-FP8"
OFFICIAL_REVISION = "bcd9f01ddc9cff2316eb84281bebcd5b058bddce"
OFFICIAL_FILES_MANIFEST_SHA256 = (
    "9252137500962bd9d639f66316d8f22e1005f45e65065e5fc15efe9924d45e3a"
)
OFFICIAL_INDEX_SHA256 = (
    "0419e2c2dfbb925257d7409405433a793cf7ff7d96f3eba882a815ec6d9fe7a6"
)
OFFICIAL_INDEX_SIZE = 17_410_140
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_INDEX_BYTES = 64 * 1024 * 1024
_SHA256_LENGTH = 64
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


class MaterializationError(RuntimeError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MaterializationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, *, maximum: int) -> dict[str, Any]:
    metadata = _regular(path, maximum=maximum)
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                MaterializationError(f"non-finite JSON number: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict) or metadata.st_size <= 0:
        raise MaterializationError(f"{path.name} is not a JSON object")
    return value


def _regular(path: Path, *, maximum: int | None = None) -> os.stat_result:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink < 1
        or metadata.st_mode & 0o022
        or (maximum is not None and not 0 < metadata.st_size <= maximum)
    ):
        raise MaterializationError(f"unsafe regular file: {path}")
    return metadata


def _directory(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise MaterializationError(f"unsafe directory: {path}")
    return metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or PurePosixPath(value).name != value
        or value in {".", ".."}
        or any(character in value for character in "\x00/\\")
    ):
        raise MaterializationError(f"unsafe {label}")
    return value


def _receipt(value: Any, label: str) -> tuple[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"sha256", "size"}:
        raise MaterializationError(f"{label} receipt changed")
    digest = value.get("sha256")
    size = value.get("size")
    if (
        not isinstance(digest, str)
        or len(digest) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise MaterializationError(f"{label} receipt is malformed")
    return digest, size


def _verify(path: Path, receipt: Any, label: str) -> None:
    digest, size = _receipt(receipt, label)
    metadata = _regular(path)
    if metadata.st_size != size or _sha256(path) != digest:
        raise MaterializationError(f"{label} identity changed")


def _read_header(path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]], int]:
    metadata = _regular(path)
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise MaterializationError("safetensors header is truncated")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= min(metadata.st_size - 8, 256 * 1024 * 1024):
            raise MaterializationError("safetensors header length changed")
        raw = handle.read(header_size)
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError("safetensors header is malformed") from exc
    if not isinstance(header, dict):
        raise MaterializationError("safetensors header is not an object")
    raw_metadata = header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_metadata.items()
    ):
        raise MaterializationError("safetensors metadata changed")
    tensors: dict[str, dict[str, Any]] = {}
    cursor = 0
    for name, descriptor in sorted(
        header.items(), key=lambda item: item[1].get("data_offsets", [-1])[0]
    ):
        if (
            not isinstance(name, str)
            or not isinstance(descriptor, dict)
            or set(descriptor) != {"dtype", "shape", "data_offsets"}
        ):
            raise MaterializationError("safetensors tensor descriptor changed")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if (
            dtype not in _DTYPE_BYTES
            or not isinstance(shape, list)
            or not all(type(item) is int and item >= 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(item) is int and item >= 0 for item in offsets)
            or offsets[0] != cursor
            or offsets[1] < offsets[0]
            or offsets[1] - offsets[0] != math.prod(shape) * _DTYPE_BYTES[dtype]
        ):
            raise MaterializationError(f"safetensors tensor layout changed: {name}")
        tensors[name] = dict(descriptor)
        cursor = offsets[1]
    if not tensors or 8 + header_size + cursor != metadata.st_size:
        raise MaterializationError("safetensors payload length changed")
    return dict(raw_metadata), tensors, 8 + header_size


def _write_all(descriptor: int, payload: bytes | memoryview) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise MaterializationError("write was incomplete")
        view = view[written:]


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy(source: Path, destination: Path) -> None:
    _regular(source)
    output = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                _write_all(output, block)
        os.fsync(output)
    finally:
        os.close(output)


def _official_url(filename: str) -> str:
    name = _safe_name(filename, "official source filename")
    return (
        f"https://huggingface.co/{OFFICIAL_REPO}/resolve/{OFFICIAL_REVISION}/"
        f"{urllib.parse.quote(name, safe='')}?download=true"
    )


def _approved_download_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    hostname = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        hostname == "huggingface.co"
        or hostname.endswith(".huggingface.co")
        or hostname.endswith(".hf.co")
        or hostname.endswith(".xethub.hf.co")
    )


def _download_exact(
    destination: Path,
    receipt: Mapping[str, Any],
    *,
    label: str,
) -> int:
    """Resumably fetch one immutable public Hub object and return new bytes."""
    expected_digest, expected_size = _receipt(receipt, label)
    if destination.exists() or destination.is_symlink():
        _verify(destination, receipt, label)
        return 0
    partial = destination.with_name(f".{destination.name}.partial")
    offset = 0
    hasher = hashlib.sha256()
    partial_preexisting = partial.exists() or partial.is_symlink()
    if partial_preexisting:
        metadata = _regular(partial)
        if metadata.st_nlink != 1 or metadata.st_size > expected_size:
            raise MaterializationError(f"partial {label} is unsafe or exceeds its pinned size")
        with partial.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        offset = metadata.st_size
        if offset == expected_size:
            if hasher.hexdigest() != expected_digest:
                raise MaterializationError(f"complete partial {label} has the wrong identity")
            os.replace(partial, destination)
            _fsync_directory(destination.parent)
            return 0
    request = urllib.request.Request(
        _official_url(destination.name),
        headers={
            "User-Agent": "Aeon-Qwen3.8-Flash-Next-Materializer/1",
            **({"Range": f"bytes={offset}-"} if offset else {}),
        },
    )
    flags = os.O_WRONLY | os.O_CLOEXEC | (
        os.O_APPEND if partial_preexisting else os.O_CREAT | os.O_EXCL
    )
    descriptor = os.open(partial, flags, 0o600)
    downloaded = 0
    try:
        try:
            response_context = urllib.request.urlopen(request, timeout=120)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise MaterializationError(f"cannot download {label}: {exc}") from exc
        with response_context as response:
            final_url = response.geturl()
            if not isinstance(final_url, str) or not _approved_download_url(final_url):
                raise MaterializationError(f"{label} redirected outside approved HTTPS hosts")
            status = getattr(response, "status", None)
            content_range = response.headers.get("Content-Range", "")
            if offset:
                expected_range_prefix = f"bytes {offset}-"
                if (
                    status != 206
                    or not content_range.startswith(expected_range_prefix)
                    or not content_range.endswith(f"/{expected_size}")
                ):
                    raise MaterializationError(f"server did not honor the exact {label} resume range")
            elif status == 206:
                if not content_range.startswith("bytes 0-") or not content_range.endswith(
                    f"/{expected_size}"
                ):
                    raise MaterializationError(f"unexpected initial range for {label}")
            elif status not in {None, 200}:
                raise MaterializationError(f"unexpected HTTP status while downloading {label}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    length = int(content_length)
                except ValueError as exc:
                    raise MaterializationError(f"malformed content length for {label}") from exc
                if length != expected_size - offset:
                    raise MaterializationError(f"content length changed for {label}")
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                offset += len(block)
                downloaded += len(block)
                if offset > expected_size:
                    raise MaterializationError(f"download exceeded the pinned size for {label}")
                hasher.update(block)
                _write_all(descriptor, block)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if offset != expected_size or hasher.hexdigest() != expected_digest:
        raise MaterializationError(f"downloaded {label} has the wrong identity")
    os.replace(partial, destination)
    _fsync_directory(destination.parent)
    return downloaded


def download_official_sources(thin_root: Path, destination: Path) -> dict[str, Any]:
    """Resolve the exact public FP8 PLE dependencies into a reusable local cache."""
    thin = thin_root.resolve(strict=True)
    _directory(thin)
    script_path = Path(__file__).resolve(strict=True)
    manifest = _read_json(thin / MANIFEST_NAME, maximum=MAX_MANIFEST_BYTES)
    _validate_manifest(manifest, script_path)

    raw_destination = destination.absolute()
    parent = raw_destination.parent.resolve(strict=True)
    _directory(parent)
    source_root = parent / raw_destination.name
    if raw_destination != source_root or source_root == thin or source_root.is_relative_to(thin):
        raise MaterializationError("official source cache must be outside the thin model")
    if source_root.is_symlink():
        raise MaterializationError("official source cache cannot be a symlink")
    if source_root.exists():
        _directory(source_root)
        source_root = source_root.resolve(strict=True)
    else:
        source_root.mkdir(mode=0o700, exist_ok=False)
        _fsync_directory(parent)

    official = manifest["official_fp8"]
    index_name = _safe_name(official["index_filename"], "official index")
    index_receipt = {
        "sha256": official["index_sha256"],
        "size": OFFICIAL_INDEX_SIZE,
    }
    source_receipts: dict[str, Mapping[str, Any]] = {index_name: index_receipt}
    for shard in manifest["ple_shards"]:
        name = _safe_name(shard["source_filename"], "official PLE filename")
        receipt = shard["source"]
        existing = source_receipts.get(name)
        if existing is not None and existing != receipt:
            raise MaterializationError("two PLE records disagree about one source file")
        source_receipts[name] = receipt

    downloaded = 0
    for name, receipt in sorted(source_receipts.items()):
        downloaded += _download_exact(
            source_root / name,
            receipt,
            label=f"official FP8 {name}",
        )
    return {
        "official_fp8_root": str(source_root),
        "repo": official["repo"],
        "revision": official["revision"],
        "files": len(source_receipts),
        "downloaded_bytes": downloaded,
    }


def _filter(source: Path, destination: Path, wanted: Iterable[str]) -> None:
    metadata, tensors, data_start = _read_header(source)
    names = tuple(sorted(set(wanted), key=lambda name: tensors[name]["data_offsets"][0]))
    if not names or any(name not in tensors for name in names):
        raise MaterializationError("filtered safetensors selection changed")
    header: dict[str, Any] = {"__metadata__": metadata}
    cursor = 0
    for name in names:
        source_descriptor = tensors[name]
        size = source_descriptor["data_offsets"][1] - source_descriptor["data_offsets"][0]
        header[name] = {
            "dtype": source_descriptor["dtype"],
            "shape": source_descriptor["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    output = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    source_descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
    try:
        _write_all(output, struct.pack("<Q", len(raw)))
        _write_all(output, raw)
        for name in names:
            start, end = tensors[name]["data_offsets"]
            offset = data_start + start
            remaining = end - start
            while remaining:
                block = os.pread(source_descriptor, min(8 * 1024 * 1024, remaining), offset)
                if not block:
                    raise MaterializationError("safetensors source was truncated")
                _write_all(output, block)
                offset += len(block)
                remaining -= len(block)
        os.fsync(output)
    finally:
        os.close(source_descriptor)
        os.close(output)


def _validate_manifest(value: Mapping[str, Any], script_path: Path) -> None:
    expected = {
        "schema_version",
        "complete",
        "official_fp8",
        "materializer_sha256",
        "checkpoint_tree_sha256",
        "canonical_files",
        "thin_file_map",
        "ple_shards",
    }
    if set(value) != expected or value.get("schema_version") != SCHEMA_VERSION or value.get("complete") is not True:
        raise MaterializationError("materialization manifest envelope changed")
    if value.get("materializer_sha256") != _sha256(script_path):
        raise MaterializationError("materializer script identity changed")
    official = value.get("official_fp8")
    if not isinstance(official, Mapping) or set(official) != {
        "repo",
        "revision",
        "files_manifest_sha256",
        "index_sha256",
        "index_filename",
    }:
        raise MaterializationError("official FP8 source receipt changed")
    if official != {
        "repo": OFFICIAL_REPO,
        "revision": OFFICIAL_REVISION,
        "files_manifest_sha256": OFFICIAL_FILES_MANIFEST_SHA256,
        "index_sha256": OFFICIAL_INDEX_SHA256,
        "index_filename": "model.safetensors.index.json",
    }:
        raise MaterializationError("official FP8 source identity changed")
    for field in ("files_manifest_sha256", "index_sha256", "checkpoint_tree_sha256"):
        digest = value.get(field) if field == "checkpoint_tree_sha256" else official.get(field)
        _receipt({"sha256": digest, "size": 1}, field)
    canonical = value.get("canonical_files")
    thin_file_map = value.get("thin_file_map")
    shards = value.get("ple_shards")
    if (
        not isinstance(canonical, dict)
        or not canonical
        or not isinstance(thin_file_map, dict)
        or not isinstance(shards, list)
        or len(shards) != 33
    ):
        raise MaterializationError("materialization file inventory changed")
    for name, receipt in canonical.items():
        _safe_name(name, "canonical filename")
        _receipt(receipt, f"canonical {name}")
    seen_targets: set[str] = set()
    for shard in shards:
        if not isinstance(shard, Mapping) or set(shard) != {
            "target_filename",
            "target",
            "source_filename",
            "source",
            "tensor_names",
            "filtered",
        }:
            raise MaterializationError("PLE shard receipt changed")
        target = _safe_name(shard.get("target_filename"), "PLE target filename")
        source = _safe_name(shard.get("source_filename"), "PLE source filename")
        tensors = shard.get("tensor_names")
        if target in seen_targets or target not in canonical or not source.endswith(".safetensors"):
            raise MaterializationError("PLE shard filename closure changed")
        seen_targets.add(target)
        _receipt(shard.get("target"), f"PLE target {target}")
        _receipt(shard.get("source"), f"PLE source {source}")
        if shard["target"] != canonical[target] or type(shard.get("filtered")) is not bool:
            raise MaterializationError("PLE target receipt changed")
        if not isinstance(tensors, list) or tensors != sorted(set(tensors)) or not tensors:
            raise MaterializationError("PLE tensor inventory changed")
    if set(thin_file_map) != set(canonical) - seen_targets:
        raise MaterializationError("thin canonical file map is incomplete")
    mapped_names = [
        _safe_name(value, "thin canonical filename")
        for value in thin_file_map.values()
    ]
    if len(mapped_names) != len(set(mapped_names)):
        raise MaterializationError("thin canonical file map aliases two files")


def materialize(
    thin_root: Path,
    official_fp8_root: Path,
    output: Path,
    *,
    receipt: Path | None = None,
) -> dict[str, Any]:
    thin = thin_root.resolve(strict=True)
    source_root = official_fp8_root.resolve(strict=True)
    _directory(thin)
    _directory(source_root)
    script_path = Path(__file__).resolve(strict=True)
    manifest_path = thin / MANIFEST_NAME
    manifest = _read_json(manifest_path, maximum=MAX_MANIFEST_BYTES)
    _validate_manifest(manifest, script_path)
    official = manifest["official_fp8"]
    index_path = source_root / _safe_name(official["index_filename"], "official index")
    if _sha256(index_path) != official["index_sha256"]:
        raise MaterializationError("official FP8 index identity changed")
    index = _read_json(index_path, maximum=MAX_INDEX_BYTES)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise MaterializationError("official FP8 index has no weight map")
    raw_output = output.absolute()
    parent = raw_output.parent.resolve(strict=True)
    _directory(parent)
    final = parent / raw_output.name
    if raw_output != final or final.exists() or final.is_symlink():
        raise MaterializationError("output must be a new canonical path")
    receipt_path: Path | None = None
    if receipt is not None:
        raw_receipt = receipt.absolute()
        receipt_parent = raw_receipt.parent.resolve(strict=True)
        _directory(receipt_parent)
        receipt_path = receipt_parent / raw_receipt.name
        if (
            raw_receipt != receipt_path
            or receipt_path.parent != parent
            or receipt_path.exists()
            or receipt_path.is_symlink()
            or receipt_path == final
        ):
            raise MaterializationError(
                "completion receipt must be a new sibling of the output"
            )
    temporary = parent / f".{final.name}.materialize-{os.getpid()}-{secrets.token_hex(8)}"
    temporary.mkdir(mode=0o700, exist_ok=False)
    canonical = manifest["canonical_files"]
    ple_targets = {item["target_filename"] for item in manifest["ple_shards"]}
    for name, receipt in sorted(canonical.items()):
        if name in ple_targets:
            continue
        source = thin / manifest["thin_file_map"][name]
        _verify(source, receipt, f"thin canonical {name}")
        _copy(source, temporary / name)
    for shard in manifest["ple_shards"]:
        source_name = shard["source_filename"]
        target_name = shard["target_filename"]
        source = source_root / source_name
        _verify(source, shard["source"], f"official FP8 {source_name}")
        tensors = shard["tensor_names"]
        if any(weight_map.get(name) != source_name for name in tensors):
            raise MaterializationError("official FP8 tensor-to-shard mapping changed")
        if shard["filtered"]:
            _filter(source, temporary / target_name, tensors)
        else:
            _copy(source, temporary / target_name)
        _verify(temporary / target_name, shard["target"], f"materialized {target_name}")
    lines: list[str] = []
    for name, receipt in sorted(canonical.items()):
        _verify(temporary / name, receipt, f"canonical {name}")
        lines.append(f"{receipt['sha256']}  {name}\n")
    sums_payload = "".join(lines).encode("ascii")
    sums = temporary / "SHA256SUMS"
    descriptor = os.open(sums, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        _write_all(descriptor, sums_payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    checkpoint_tree = hashlib.sha256(sums_payload).hexdigest()
    if checkpoint_tree != manifest["checkpoint_tree_sha256"]:
        raise MaterializationError("materialized checkpoint tree identity changed")
    _fsync_directory(temporary)
    os.rename(temporary, final)
    _fsync_directory(parent)
    result = {
        "output": str(final),
        "checkpoint_tree_sha256": checkpoint_tree,
        "canonical_file_count": len(canonical),
        "ple_shard_count": len(ple_targets),
    }
    if receipt_path is not None:
        completion = {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "complete": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "materialized_model_dir_sha256": hashlib.sha256(
                str(final).encode("utf-8")
            ).hexdigest(),
            "materialized_checkpoint_tree_sha256": checkpoint_tree,
            "ple_materialization_manifest_sha256": _sha256(manifest_path),
            "ple_materializer_sha256": _sha256(script_path),
            "official_fp8": {
                "repo": official["repo"],
                "revision": official["revision"],
                "files_manifest_sha256": official["files_manifest_sha256"],
                "index_sha256": official["index_sha256"],
            },
            "canonical_file_count": len(canonical),
            "ple_shard_count": len(ple_targets),
        }
        payload = (
            json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _write_exclusive(receipt_path, payload)
        _fsync_directory(parent)
        result["completion_receipt"] = str(receipt_path)
        result["completion_receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thin-model", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--official-fp8-root",
        type=Path,
        help="existing directory containing the exact pinned official FP8 sources",
    )
    source.add_argument(
        "--download-official-to",
        type=Path,
        help="reusable cache to populate resumably from the exact pinned public revision",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    source_result: dict[str, Any]
    if args.download_official_to is not None:
        source_result = download_official_sources(
            args.thin_model,
            args.download_official_to,
        )
        official_fp8_root = Path(source_result["official_fp8_root"])
        source_result["mode"] = "auto-downloaded-pinned-public-source"
    else:
        official_fp8_root = args.official_fp8_root
        source_result = {
            "official_fp8_root": str(official_fp8_root),
            "mode": "preexisting-pinned-source",
        }
    result = materialize(
        args.thin_model,
        official_fp8_root,
        args.output,
        receipt=args.receipt,
    )
    result["official_source"] = source_result
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
