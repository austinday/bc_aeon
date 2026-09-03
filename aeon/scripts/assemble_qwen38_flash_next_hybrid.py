#!/usr/bin/env python3
"""Assemble the immutable BF16/FP8-PLE Qwen3.8-Flash-Next hybrid.

This program is intentionally a worker-side, task-scratch operation.  It downloads
only the pinned official BF16 shards which contain non-PLE tensors, consumes only
the pinned official FP8 PLE shards staged by Fleet, filters the four mixed shards,
and atomically publishes a tensor-level hybrid plus a complete file manifest.  It
never selects or opens a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import urllib.parse
import urllib.request
from typing import Any, Iterable


SCHEMA_VERSION = "aeon-qwen38-flash-next-hybrid-v1"
FILES_SCHEMA = "aeon-pinned-hf-files-v1"
BF16_REPO = "Qwen/Qwen3.8-Flash-Next"
BF16_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
FP8_REPO = "Qwen/Qwen3.8-Flash-Next-FP8"
FP8_REVISION = "bcd9f01ddc9cff2316eb84281bebcd5b058bddce"
BF16_CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
BF16_INDEX_SHA256 = "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de"
BF16_FILES_SHA256 = "1f2c885695bb74cd6b908fdb5899b553afe64d650a20b81f0c14e557f4f256ad"
FP8_FILES_SHA256 = "9252137500962bd9d639f66316d8f22e1005f45e65065e5fc15efe9924d45e3a"
FP8_CONFIG_SHA256 = "99c11efba4012d0f760f4e4831a8d6cafd845044e21d0aa9e6d9e70a15a90a8d"
FP8_INDEX_SHA256 = "0419e2c2dfbb925257d7409405433a793cf7ff7d96f3eba882a815ec6d9fe7a6"
EXPECTED_TENSORS = 1_659
PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
PLE_SCALE = PLE_PREFIX + "weight_scale"
METADATA_FILES = (
    "LICENSE",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
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


class HybridAssemblyError(RuntimeError):
    """An immutable input or assembled artifact failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _regular(path: Path, *, maximum: int | None = None) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HybridAssemblyError(f"required file is absent: {path}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or metadata.st_size <= 0
        or (maximum is not None and metadata.st_size > maximum)
    ):
        raise HybridAssemblyError(f"file is not a private immutable owner file: {path}")
    return metadata


def _private_directory(path: Path, *, create: bool = False) -> Path:
    if create:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HybridAssemblyError(f"directory is absent: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise HybridAssemblyError(f"directory is not private and owned: {path}")
    return path


def _read_json(path: Path, *, maximum: int = 32 * 1024 * 1024) -> dict[str, Any]:
    _regular(path, maximum=maximum)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HybridAssemblyError(f"JSON is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise HybridAssemblyError(f"JSON root is not an object: {path}")
    return value


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise HybridAssemblyError(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_write_once(path: Path, payload: bytes) -> None:
    """Publish a small task-owned receipt atomically and reuse an exact result."""
    if path.exists() or path.is_symlink():
        _regular(path, maximum=32 * 1024 * 1024)
        if path.read_bytes() != payload:
            raise HybridAssemblyError(f"existing receipt identity changed: {path}")
        return
    partial = path.with_name(f".{path.name}.partial")
    if partial.exists() or partial.is_symlink():
        metadata = partial.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
        ):
            raise HybridAssemblyError(f"partial receipt identity changed: {partial}")
        partial.unlink()
        _fsync_parent(path)
    _exclusive_write(partial, payload)
    try:
        os.link(partial, path, follow_symlinks=False)
    except FileExistsError:
        _regular(path, maximum=32 * 1024 * 1024)
        if path.read_bytes() != payload:
            raise HybridAssemblyError(f"receipt publication raced: {path}")
    finally:
        partial.unlink()
    _fsync_parent(path)


def _validate_files_manifest(
    path: Path, *, digest: str, repo: str, revision: str
) -> dict[str, tuple[int, str | None]]:
    if _sha256(path) != digest:
        raise HybridAssemblyError(f"pinned files manifest changed: {path}")
    value = _read_json(path)
    files = value.get("files")
    if (
        value.get("schema_version") != FILES_SCHEMA
        or value.get("repo") != repo
        or value.get("revision") != revision
        or not isinstance(files, dict)
    ):
        raise HybridAssemblyError("pinned files manifest identity changed")
    result: dict[str, tuple[int, str | None]] = {}
    for name, receipt in files.items():
        pure = PurePosixPath(str(name))
        if (
            not isinstance(name, str)
            or pure.is_absolute()
            or len(pure.parts) != 1
            or name in {"", ".", ".."}
            or not isinstance(receipt, dict)
        ):
            raise HybridAssemblyError("pinned files manifest has an unsafe entry")
        size = receipt.get("size")
        digest_value = receipt.get("sha256")
        if (
            type(size) is not int
            or size <= 0
            or (
                digest_value is not None
                and (
                    not isinstance(digest_value, str)
                    or len(digest_value) != 64
                    or any(ch not in "0123456789abcdef" for ch in digest_value)
                )
            )
        ):
            raise HybridAssemblyError(f"pinned file receipt is malformed: {name}")
        result[name] = (size, digest_value)
    return result


def _read_header(path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]], int]:
    metadata = _regular(path)
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise HybridAssemblyError(f"safetensors prefix is truncated: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= 256 * 1024 * 1024 or header_size % 8:
            raise HybridAssemblyError(f"safetensors header length is invalid: {path}")
        raw = handle.read(header_size)
    if len(raw) != header_size:
        raise HybridAssemblyError(f"safetensors header is truncated: {path}")
    try:
        header = json.loads(raw.rstrip(b" "))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HybridAssemblyError(f"safetensors header is malformed: {path}") from exc
    if not isinstance(header, dict):
        raise HybridAssemblyError("safetensors header is not an object")
    raw_metadata = header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_metadata.items()
    ):
        raise HybridAssemblyError("safetensors metadata is malformed")
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
            raise HybridAssemblyError("safetensors tensor descriptor is malformed")
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
            raise HybridAssemblyError(f"safetensors tensor layout changed: {name}")
        tensors[name] = dict(descriptor)
        cursor = offsets[1]
    if not tensors or 8 + header_size + cursor != metadata.st_size:
        raise HybridAssemblyError(f"safetensors payload length changed: {path}")
    return dict(raw_metadata), tensors, 8 + header_size


def _filtered_safetensors(
    source: Path, destination: Path, wanted: Iterable[str]
) -> None:
    metadata, tensors, data_start = _read_header(source)
    names = tuple(
        sorted(set(wanted), key=lambda name: tensors[name]["data_offsets"][0])
    )
    if not names or any(name not in tensors for name in names):
        raise HybridAssemblyError("filtered safetensors selection is invalid")
    header: dict[str, Any] = {"__metadata__": metadata}
    cursor = 0
    for name in names:
        descriptor = tensors[name]
        size = descriptor["data_offsets"][1] - descriptor["data_offsets"][0]
        header[name] = {
            "dtype": descriptor["dtype"],
            "shape": descriptor["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    descriptor_out = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    descriptor_in = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.write(descriptor_out, struct.pack("<Q", len(raw)))
        os.write(descriptor_out, raw)
        for name in names:
            start, end = tensors[name]["data_offsets"]
            offset = data_start + start
            remaining = end - start
            while remaining:
                block = os.pread(descriptor_in, min(8 * 1024 * 1024, remaining), offset)
                if not block:
                    raise HybridAssemblyError(
                        "safetensors source truncated while filtering"
                    )
                view = memoryview(block)
                while view:
                    written = os.write(descriptor_out, view)
                    if written <= 0:
                        raise HybridAssemblyError(
                            "safetensors filtered write was incomplete"
                        )
                    view = view[written:]
                offset += len(block)
                remaining -= len(block)
        os.fsync(descriptor_out)
    finally:
        os.close(descriptor_in)
        os.close(descriptor_out)
    _read_header(destination)


def _verify_file(path: Path, receipt: tuple[int, str | None]) -> None:
    size, digest = receipt
    metadata = _regular(path)
    if metadata.st_size != size or digest is None or _sha256(path) != digest:
        raise HybridAssemblyError(f"pinned payload identity changed: {path.name}")


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _download(url: str, destination: Path, receipt: tuple[int, str | None]) -> None:
    size, digest = receipt
    if digest is None:
        raise HybridAssemblyError("download has no SHA-256 identity")
    if destination.exists() or destination.is_symlink():
        raise HybridAssemblyError("pinned download destination already exists")
    partial = destination.with_name(f".{destination.name}.partial")
    hasher = hashlib.sha256()
    written_total = 0
    flags = os.O_WRONLY | os.O_CLOEXEC
    if partial.exists() or partial.is_symlink():
        metadata = partial.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o077
            or not 0 <= metadata.st_size <= size
        ):
            raise HybridAssemblyError("partial download identity is unsafe")
        with partial.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        written_total = metadata.st_size
        if written_total == size:
            if hasher.hexdigest() == digest:
                os.replace(partial, destination)
                _fsync_parent(destination)
                return
            descriptor = os.open(partial, os.O_WRONLY | os.O_TRUNC | os.O_CLOEXEC)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            hasher = hashlib.sha256()
            written_total = 0
        else:
            flags |= os.O_APPEND
    else:
        flags |= os.O_CREAT | os.O_EXCL
    headers = {"User-Agent": "Aeon-Fleet/1"}
    if written_total:
        headers["Range"] = f"bytes={written_total}-"
    request = urllib.request.Request(url, headers=headers)
    descriptor = os.open(partial, flags, 0o600)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            final = urllib.parse.urlparse(response.geturl())
            if final.scheme != "https" or not (
                final.hostname == "huggingface.co"
                or (final.hostname or "").endswith(".huggingface.co")
                or (final.hostname or "").endswith(".hf.co")
                or (final.hostname or "").endswith(".xethub.hf.co")
            ):
                raise HybridAssemblyError(
                    "pinned download redirected outside approved HTTPS hosts"
                )
            if written_total:
                content_range = response.headers.get("Content-Range", "")
                if (
                    getattr(response, "status", None) != 206
                    or not content_range.startswith(f"bytes {written_total}-")
                    or not content_range.endswith(f"/{size}")
                ):
                    raise HybridAssemblyError(
                        "pinned download server did not honor the exact resume range"
                    )
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                written_total += len(block)
                if written_total > size:
                    raise HybridAssemblyError(
                        "pinned download exceeded its expected size"
                    )
                hasher.update(block)
                view = memoryview(block)
                while view:
                    count = os.write(descriptor, view)
                    if count <= 0:
                        raise HybridAssemblyError(
                            "pinned download write was incomplete"
                        )
                    view = view[count:]
    finally:
        os.fsync(descriptor)
        os.close(descriptor)
    if written_total != size or hasher.hexdigest() != digest:
        raise HybridAssemblyError("pinned download content identity changed")
    os.replace(partial, destination)
    _fsync_parent(destination)


def _copy_regular(path: Path, target: Path, *, maximum: int | None = None) -> None:
    """Copy an immutable source without links or source mutation."""
    _regular(path, maximum=maximum)
    descriptor = os.open(
        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                view = memoryview(chunk)
                while view:
                    count = os.write(descriptor, view)
                    if count <= 0:
                        raise HybridAssemblyError("immutable file copy was incomplete")
                    view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_metadata(source: Path, destination: Path, name: str) -> None:
    _copy_regular(source / name, destination / name, maximum=32 * 1024 * 1024)


def stage_sources(
    *,
    bf16_root: Path,
    bf16_files_manifest: Path,
    fp8_root: Path,
    fp8_files_manifest: Path,
    mtp_subset: Path,
    source_manifest: Path,
) -> dict[str, Any]:
    """Materialize and hash the exact trainer source view in task scratch."""
    _private_directory(bf16_root)
    _private_directory(fp8_root)
    bf16_files = _validate_files_manifest(
        bf16_files_manifest,
        digest=BF16_FILES_SHA256,
        repo=BF16_REPO,
        revision=BF16_REVISION,
    )
    fp8_files = _validate_files_manifest(
        fp8_files_manifest,
        digest=FP8_FILES_SHA256,
        repo=FP8_REPO,
        revision=FP8_REVISION,
    )
    bf16_index_path = bf16_root / "model.safetensors.index.json"
    fp8_index_path = fp8_root / "model.safetensors.index.json"
    if (
        _sha256(bf16_root / "config.json") != BF16_CONFIG_SHA256
        or _sha256(bf16_index_path) != BF16_INDEX_SHA256
        or _sha256(fp8_root / "config.json") != FP8_CONFIG_SHA256
        or _sha256(fp8_index_path) != FP8_INDEX_SHA256
    ):
        raise HybridAssemblyError("official staged metadata identity changed")
    bf16_map = _read_json(bf16_index_path).get("weight_map")
    fp8_map = _read_json(fp8_index_path, maximum=32 * 1024 * 1024).get("weight_map")
    if not isinstance(bf16_map, dict) or not isinstance(fp8_map, dict):
        raise HybridAssemblyError("official model index is malformed")
    bf16_shards = sorted(
        {
            shard
            for name, shard in bf16_map.items()
            if isinstance(name, str)
            and not name.startswith(PLE_PREFIX)
            and isinstance(shard, str)
        }
    )
    fp8_shards = sorted(
        {
            shard
            for name, shard in fp8_map.items()
            if isinstance(name, str)
            and name.startswith(PLE_PREFIX)
            and isinstance(shard, str)
        }
    )
    if len(bf16_shards) != 100 or len(fp8_shards) != 33:
        raise HybridAssemblyError("official source shard topology changed")
    checked: dict[str, str] = {}
    for shard in bf16_shards:
        if PurePosixPath(shard).name != shard or shard not in bf16_files:
            raise HybridAssemblyError("BF16 shard inventory is unsafe")
        path = bf16_root / shard
        if path.exists() or path.is_symlink():
            _verify_file(path, bf16_files[shard])
        else:
            url = (
                f"https://huggingface.co/{BF16_REPO}/resolve/{BF16_REVISION}/"
                f"{urllib.parse.quote(shard)}?download=true"
            )
            _download(url, path, bf16_files[shard])
        checked[str(path)] = _sha256(path)
    for shard in fp8_shards:
        if PurePosixPath(shard).name != shard or shard not in fp8_files:
            raise HybridAssemblyError("FP8 PLE shard inventory is unsafe")
        path = fp8_root / shard
        _verify_file(path, fp8_files[shard])
        checked[str(path)] = _sha256(path)
    for root, names in (
        (
            bf16_root,
            (
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ),
        ),
        (fp8_root, ("config.json", "model.safetensors.index.json")),
    ):
        for name in names:
            checked[str(root / name)] = _sha256(root / name)
    _regular(mtp_subset, maximum=8 * 1024**3)
    checked[str(mtp_subset)] = _sha256(mtp_subset)
    receipt = {
        "schema_version": "aeon-qwen38-flash-next-trainer-source-v1",
        "role": "externally-owned-pinned-hybrid-source",
        "sources": {
            "bf16": {"repo": BF16_REPO, "revision": BF16_REVISION},
            "fp8_ple": {"repo": FP8_REPO, "revision": FP8_REVISION},
        },
        "files": dict(sorted(checked.items())),
        "complete": True,
    }
    _durable_write_once(source_manifest, _canonical_json(receipt))
    return {
        "schema_version": receipt["schema_version"],
        "source_manifest_sha256": _sha256(source_manifest),
        "bf16_index_sha256": BF16_INDEX_SHA256,
        "fp8_index_sha256": FP8_INDEX_SHA256,
        "mtp_subset_sha256": checked[str(mtp_subset)],
        "bf16_shards": len(bf16_shards),
        "fp8_ple_shards": len(fp8_shards),
    }


def assemble(
    *,
    bf16_metadata_root: Path,
    bf16_files_manifest: Path,
    fp8_root: Path,
    fp8_files_manifest: Path,
    output: Path,
    preserve_bf16_sources: bool = False,
    preserve_fp8_sources: bool = False,
) -> dict[str, Any]:
    """Build and atomically publish the reviewed hybrid checkpoint."""
    _private_directory(bf16_metadata_root)
    _private_directory(fp8_root)
    _private_directory(output.parent)
    if output.exists() or output.is_symlink():
        raise HybridAssemblyError("hybrid output already exists")
    bf16_files = _validate_files_manifest(
        bf16_files_manifest,
        digest=BF16_FILES_SHA256,
        repo=BF16_REPO,
        revision=BF16_REVISION,
    )
    fp8_files = _validate_files_manifest(
        fp8_files_manifest,
        digest=FP8_FILES_SHA256,
        repo=FP8_REPO,
        revision=FP8_REVISION,
    )
    bf16_config_path = bf16_metadata_root / "config.json"
    bf16_index_path = bf16_metadata_root / "model.safetensors.index.json"
    if _sha256(bf16_config_path) != BF16_CONFIG_SHA256:
        raise HybridAssemblyError("official BF16 config changed")
    if _sha256(bf16_index_path) != BF16_INDEX_SHA256:
        raise HybridAssemblyError("official BF16 index changed")
    bf16_index = _read_json(bf16_index_path)
    fp8_index = _read_json(
        fp8_root / "model.safetensors.index.json", maximum=32 * 1024 * 1024
    )
    bf16_map = bf16_index.get("weight_map")
    fp8_map = fp8_index.get("weight_map")
    if not isinstance(bf16_map, dict) or not isinstance(fp8_map, dict):
        raise HybridAssemblyError("official model index is malformed")
    bf16_wanted = {name for name in bf16_map if not name.startswith(PLE_PREFIX)}
    fp8_wanted = {name for name in fp8_map if name.startswith(PLE_PREFIX)}
    if (
        len(bf16_wanted) != 1_530
        or len(fp8_wanted) != 129
        or PLE_SCALE not in fp8_wanted
        or bf16_wanted & fp8_wanted
    ):
        raise HybridAssemblyError("official tensor topology changed")

    temporary = output.with_name(f".{output.name}.partial")
    _private_directory(temporary, create=True)
    final_map: dict[str, str] = {}

    by_bf16_shard: dict[str, list[str]] = {}
    for name in bf16_wanted:
        shard = bf16_map.get(name)
        if not isinstance(shard, str) or PurePosixPath(shard).name != shard:
            raise HybridAssemblyError("official BF16 index contains an unsafe shard")
        by_bf16_shard.setdefault(shard, []).append(name)
    for shard, names in sorted(by_bf16_shard.items()):
        receipt = bf16_files.get(shard)
        if receipt is None:
            raise HybridAssemblyError(f"BF16 files receipt omits {shard}")
        downloaded = bf16_metadata_root / shard
        _verify_file(downloaded, receipt)
        _metadata, all_tensors, _start = _read_header(downloaded)
        if set(names) == set(all_tensors):
            target_name = f"bf16-{shard}"
            if preserve_bf16_sources:
                _copy_regular(downloaded, temporary / target_name)
            else:
                os.replace(downloaded, temporary / target_name)
        else:
            if set(names) - set(all_tensors):
                raise HybridAssemblyError("BF16 index/shard inventory changed")
            target_name = f"bf16-filtered-{shard}"
            _filtered_safetensors(downloaded, temporary / target_name, names)
            if not preserve_bf16_sources:
                downloaded.unlink()
        for name in names:
            final_map[name] = target_name

    by_fp8_shard: dict[str, list[str]] = {}
    for name in fp8_wanted:
        shard = fp8_map.get(name)
        if not isinstance(shard, str) or PurePosixPath(shard).name != shard:
            raise HybridAssemblyError("official FP8 index contains an unsafe shard")
        by_fp8_shard.setdefault(shard, []).append(name)
    for shard, names in sorted(by_fp8_shard.items()):
        source = fp8_root / shard
        receipt = fp8_files.get(shard)
        if receipt is None:
            raise HybridAssemblyError(f"FP8 files receipt omits {shard}")
        _verify_file(source, receipt)
        _metadata, all_tensors, _start = _read_header(source)
        if set(names) == set(all_tensors):
            target_name = f"fp8-ple-{shard}"
            if preserve_fp8_sources:
                _copy_regular(source, temporary / target_name)
            else:
                os.replace(source, temporary / target_name)
        else:
            if set(names) - set(all_tensors):
                raise HybridAssemblyError("FP8 index/shard inventory changed")
            target_name = f"fp8-ple-filtered-{shard}"
            _filtered_safetensors(source, temporary / target_name, names)
            if not preserve_fp8_sources:
                source.unlink()
        for name in names:
            final_map[name] = target_name

    if len(final_map) != EXPECTED_TENSORS:
        raise HybridAssemblyError("assembled hybrid tensor count changed")
    for name in METADATA_FILES:
        _copy_metadata(bf16_metadata_root, temporary, name)
    config = _read_json(bf16_config_path)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict) or text_config.get("split_ngram_parts") != 128:
        raise HybridAssemblyError("official BF16 config lost split PLE topology")
    # The hybrid config remains byte-exact official BF16.  The final builder is
    # the sole owner of adding the FP8 PLE runtime dtype to the release config.
    _copy_metadata(bf16_metadata_root, temporary, "config.json")
    total_tensor_bytes = 0
    for file_name in sorted(set(final_map.values())):
        _metadata, records, _start = _read_header(temporary / file_name)
        total_tensor_bytes += sum(
            item["data_offsets"][1] - item["data_offsets"][0]
            for item in records.values()
        )
    model_index = {
        "metadata": {"total_size": total_tensor_bytes},
        "weight_map": dict(sorted(final_map.items())),
    }
    _exclusive_write(
        temporary / "model.safetensors.index.json", _canonical_json(model_index)
    )
    receipts: dict[str, dict[str, int | str]] = {}
    for item in sorted(temporary.iterdir(), key=lambda path: path.name):
        metadata = _regular(item)
        receipts[item.name] = {"sha256": _sha256(item), "size": metadata.st_size}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "artifact": "qwen38-flash-next-tensor-hybrid",
        "sources": {
            "bf16": {"repo": BF16_REPO, "revision": BF16_REVISION},
            "fp8_ple": {"repo": FP8_REPO, "revision": FP8_REVISION},
        },
        "upstream_metadata": {
            "bf16_config_sha256": BF16_CONFIG_SHA256,
            "bf16_index_sha256": BF16_INDEX_SHA256,
        },
        "topology": {
            "tensor_count": EXPECTED_TENSORS,
            "bf16_source_expert_tensor_count": 96,
            "bf16_mtp_tensor_count": 31,
            "bf16_vision_tensor_count": 333,
            "fp8_ple_table_tensor_count": 128,
            "bf16_ple_scale_tensor_count": 1,
            "non_expert_non_mtp_tensor_count": 1_532,
        },
        "files": receipts,
    }
    _exclusive_write(temporary / "HYBRID_MANIFEST.json", _canonical_json(manifest))
    os.rename(temporary, output)
    return {
        "schema_version": SCHEMA_VERSION,
        "hybrid_manifest_sha256": _sha256(output / "HYBRID_MANIFEST.json"),
        "tensor_count": len(final_map),
        "tensor_bytes": total_tensor_bytes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-metadata-root", type=Path, required=True)
    parser.add_argument("--bf16-files-manifest", type=Path, required=True)
    parser.add_argument("--fp8-root", type=Path, required=True)
    parser.add_argument("--fp8-files-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--mtp-subset", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.stage_only:
        if arguments.mtp_subset is None or arguments.source_manifest is None:
            raise HybridAssemblyError(
                "stage-only requires MTP and source-manifest paths"
            )
        receipt = stage_sources(
            bf16_root=arguments.bf16_metadata_root,
            bf16_files_manifest=arguments.bf16_files_manifest,
            fp8_root=arguments.fp8_root,
            fp8_files_manifest=arguments.fp8_files_manifest,
            mtp_subset=arguments.mtp_subset,
            source_manifest=arguments.source_manifest,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if arguments.output is None:
        raise HybridAssemblyError("assembly requires --output")
    receipt = assemble(
        bf16_metadata_root=arguments.bf16_metadata_root,
        bf16_files_manifest=arguments.bf16_files_manifest,
        fp8_root=arguments.fp8_root,
        fp8_files_manifest=arguments.fp8_files_manifest,
        output=arguments.output,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
