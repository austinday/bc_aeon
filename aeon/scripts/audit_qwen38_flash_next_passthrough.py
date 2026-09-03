#!/usr/bin/env python3
"""Externally prove byte-exact Qwen3.8-Flash-Next pass-through tensors.

The tuned NVFP4 builder intentionally changes the routed experts and
``lm_head.weight``.  It also grafts the separately audited BF16 MTP shard.
Everything else in the assembled hybrid must survive byte-for-byte.  This
auditor is deliberately independent of the builder: it parses safetensors
headers itself, hashes raw tensor payload spans without torch/GPU allocation,
and writes a private, exclusive release-gate receipt outside both artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "aeon-qwen38-flash-next-passthrough-audit-v1"
CONTRACT_NAME = "Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c"
INDEX_FILENAME = "model.safetensors.index.json"
MAX_INDEX_BYTES = 128 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024
HASH_CHUNK_BYTES = 8 * 1024 * 1024

NUM_LAYERS = 48
NUM_EXPERTS = 512
HIDDEN_SIZE = 2560
EXPERT_INTERMEDIATE_SIZE = 640
VOCAB_SIZE = 248_320
SOURCE_TENSOR_COUNT = 1_659
OUTPUT_TENSOR_COUNT = 296_475
SOURCE_TENSOR_BYTES = 308_799_717_370
OUTPUT_TENSOR_BYTES = 135_156_121_594
PASSTHROUGH_TENSOR_COUNT = 1_531
PASSTHROUGH_TENSOR_BYTES = 60_722_106_874
PASSTHROUGH_NAME_SET_SHA256 = (
    "01285d115a282b7e928843e6f64d89d54cb03ffb12cf7a67313389eed5fe965a"
)
PASSTHROUGH_CATEGORY_COUNTS = {"other": 1_069, "ple": 129, "vision": 333}
PASSTHROUGH_DTYPE_COUNTS = {"BF16": 1_400, "F8_E4M3": 128, "I64": 3}
PASSTHROUGH_DTYPE_BYTES = {
    "BF16": 9_521_860_834,
    "F8_E4M3": 51_200_245_760,
    "I64": 280,
}

LM_HEAD = "lm_head.weight"
VISION_PREFIX = "model.visual."
PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."

MTP_NAMES = frozenset(
    {
        "mtp.fc_embedding.weight",
        "mtp.fc_hidden.weight",
        "mtp.hyper_connection_mixer.hc_norm.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
        "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
        "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
        "mtp.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight",
        "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
        "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
        "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
        "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
        "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
        "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
)

_SAFE_FILE_RE = re.compile(
    r"^(?!\.\.?$)(?!.*\.\.)(?!.*[/\\\x00-\x1f])[A-Za-z0-9_.-]{1,240}$"
)
_SAFE_TENSOR_RE = re.compile(r"^[A-Za-z0-9_.-]{1,512}$")
_SOURCE_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)$"
)
_OUTPUT_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\."
    r"(weight|weight_scale|weight_scale_2|input_scale)$"
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


class PassthroughAuditError(RuntimeError):
    """The external source/output identity proof failed closed."""


@dataclass(frozen=True)
class AuditContract:
    name: str
    num_layers: int
    num_experts: int
    hidden_size: int
    expert_intermediate_size: int
    vocab_size: int
    mtp_names: frozenset[str]
    source_tensor_count: int
    output_tensor_count: int
    source_tensor_bytes: int
    output_tensor_bytes: int
    passthrough_tensor_count: int
    passthrough_tensor_bytes: int
    passthrough_name_set_sha256: str
    passthrough_category_counts: Mapping[str, int]
    passthrough_dtype_counts: Mapping[str, int]
    passthrough_dtype_bytes: Mapping[str, int]
    vision_prefix: str = VISION_PREFIX
    ple_prefix: str = PLE_PREFIX
    lm_head: str = LM_HEAD


PRODUCTION_CONTRACT = AuditContract(
    name=CONTRACT_NAME,
    num_layers=NUM_LAYERS,
    num_experts=NUM_EXPERTS,
    hidden_size=HIDDEN_SIZE,
    expert_intermediate_size=EXPERT_INTERMEDIATE_SIZE,
    vocab_size=VOCAB_SIZE,
    mtp_names=MTP_NAMES,
    source_tensor_count=SOURCE_TENSOR_COUNT,
    output_tensor_count=OUTPUT_TENSOR_COUNT,
    source_tensor_bytes=SOURCE_TENSOR_BYTES,
    output_tensor_bytes=OUTPUT_TENSOR_BYTES,
    passthrough_tensor_count=PASSTHROUGH_TENSOR_COUNT,
    passthrough_tensor_bytes=PASSTHROUGH_TENSOR_BYTES,
    passthrough_name_set_sha256=PASSTHROUGH_NAME_SET_SHA256,
    passthrough_category_counts=PASSTHROUGH_CATEGORY_COUNTS,
    passthrough_dtype_counts=PASSTHROUGH_DTYPE_COUNTS,
    passthrough_dtype_bytes=PASSTHROUGH_DTYPE_BYTES,
)


@dataclass(frozen=True)
class TensorRecord:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class Shard:
    path: Path
    identity: FileIdentity
    data_start: int
    records: Mapping[str, TensorRecord]


@dataclass(frozen=True)
class CheckpointInventory:
    root: Path
    index_sha256: str
    weight_map: Mapping[str, str]
    shards: Mapping[str, Shard]
    tensor_count: int
    tensor_bytes: int
    file_bytes: int
    topology_sha256: str


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
    except (TypeError, ValueError) as exc:
        raise PassthroughAuditError("value is not canonical JSON") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_constant(value: str) -> None:
    raise PassthroughAuditError(f"non-finite JSON number {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PassthroughAuditError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _parse_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PassthroughAuditError(f"{label} is malformed JSON") from exc


def _identity(metadata: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _validate_regular_metadata(
    metadata: os.stat_result, path: Path, *, require_nonempty: bool = True
) -> None:
    forbidden = stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
        or metadata.st_mode & forbidden
        or (require_nonempty and metadata.st_size <= 0)
    ):
        raise PassthroughAuditError(f"unsafe regular file: {path}")


def _private_directory(path: Path) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PassthroughAuditError(f"directory is unavailable: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise PassthroughAuditError(f"directory is not private and owner-controlled: {path}")
    return path.resolve(strict=True)


def _safe_file(path: Path, *, maximum_bytes: int | None = None) -> FileIdentity:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PassthroughAuditError(f"required file is unavailable: {path}") from exc
    _validate_regular_metadata(metadata, path)
    if maximum_bytes is not None and metadata.st_size > maximum_bytes:
        raise PassthroughAuditError(f"file exceeds the reviewed size bound: {path}")
    return _identity(metadata)


def _read_exact_at(descriptor: int, size: int, offset: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    position = offset
    while remaining:
        try:
            chunk = os.pread(descriptor, remaining, position)
        except OSError as exc:
            raise PassthroughAuditError(f"failed reading {label}") from exc
        if not chunk:
            raise PassthroughAuditError(f"short read from {label}")
        chunks.append(chunk)
        remaining -= len(chunk)
        position += len(chunk)
    return b"".join(chunks)


def _open_verified(path: Path, expected: FileIdentity) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PassthroughAuditError(f"failed opening immutable input: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        _validate_regular_metadata(metadata, path)
        if _identity(metadata) != expected:
            raise PassthroughAuditError(f"input identity changed: {path}")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _load_index(root: Path) -> tuple[dict[str, Any], str, FileIdentity]:
    path = root / INDEX_FILENAME
    identity = _safe_file(path, maximum_bytes=MAX_INDEX_BYTES)
    descriptor = _open_verified(path, identity)
    try:
        payload = _read_exact_at(descriptor, identity.size, 0, str(path))
        if _identity(os.fstat(descriptor)) != identity:
            raise PassthroughAuditError("checkpoint index changed while being read")
    finally:
        os.close(descriptor)
    value = _parse_json(payload, str(path))
    if not isinstance(value, dict) or set(value) != {"metadata", "weight_map"}:
        raise PassthroughAuditError("checkpoint index fields changed")
    return value, _sha256_bytes(payload), identity


def _tensor_nbytes(dtype: str, shape: tuple[int, ...]) -> int:
    width = _DTYPE_BYTES.get(dtype)
    if width is None:
        raise PassthroughAuditError(f"unsupported safetensors dtype {dtype!r}")
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * width


def _read_shard(path: Path) -> Shard:
    identity = _safe_file(path)
    descriptor = _open_verified(path, identity)
    try:
        if identity.size < 10:
            raise PassthroughAuditError(f"safetensors file is too short: {path}")
        header_size = int.from_bytes(
            _read_exact_at(descriptor, 8, 0, str(path)), "little", signed=False
        )
        if (
            header_size < 2
            or header_size > MAX_SAFETENSORS_HEADER_BYTES
            or header_size > identity.size - 8
        ):
            raise PassthroughAuditError(f"invalid safetensors header length: {path}")
        raw_header = _read_exact_at(descriptor, header_size, 8, str(path))
        header = _parse_json(raw_header, str(path))
        if not isinstance(header, dict):
            raise PassthroughAuditError(f"safetensors header is not an object: {path}")
        metadata = header.pop("__metadata__", None)
        if metadata is not None and (
            not isinstance(metadata, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())
        ):
            raise PassthroughAuditError(f"invalid safetensors metadata: {path}")
        records: dict[str, TensorRecord] = {}
        for name, raw_record in header.items():
            if _SAFE_TENSOR_RE.fullmatch(name) is None:
                raise PassthroughAuditError(f"unsafe tensor name in {path}: {name!r}")
            if not isinstance(raw_record, dict) or set(raw_record) != {
                "data_offsets",
                "dtype",
                "shape",
            }:
                raise PassthroughAuditError(f"invalid tensor record in {path}: {name}")
            dtype = raw_record["dtype"]
            shape_raw = raw_record["shape"]
            offsets = raw_record["data_offsets"]
            if not isinstance(dtype, str) or not isinstance(shape_raw, list):
                raise PassthroughAuditError(f"invalid tensor dtype/shape in {path}: {name}")
            if not all(
                isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in shape_raw
            ):
                raise PassthroughAuditError(f"invalid tensor dimensions in {path}: {name}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(
                    isinstance(item, int) and not isinstance(item, bool) and item >= 0
                    for item in offsets
                )
            ):
                raise PassthroughAuditError(f"invalid tensor offsets in {path}: {name}")
            shape = tuple(shape_raw)
            start, end = offsets
            if end < start or end - start != _tensor_nbytes(dtype, shape):
                raise PassthroughAuditError(f"tensor byte span changed in {path}: {name}")
            records[name] = TensorRecord(dtype, shape, start, end)
        if not records:
            raise PassthroughAuditError(f"safetensors shard contains no tensors: {path}")
        cursor = 0
        for name, record in sorted(
            records.items(), key=lambda item: (item[1].start, item[1].end, item[0])
        ):
            if record.start != cursor:
                raise PassthroughAuditError(f"non-contiguous tensor payload in {path}: {name}")
            cursor = record.end
        data_start = 8 + header_size
        if data_start + cursor != identity.size:
            raise PassthroughAuditError(f"safetensors payload/file size mismatch: {path}")
        if _identity(os.fstat(descriptor)) != identity:
            raise PassthroughAuditError(f"safetensors changed while parsed: {path}")
    finally:
        os.close(descriptor)
    return Shard(path, identity, data_start, records)


def _inventory(root: Path, *, final: bool) -> CheckpointInventory:
    root = _private_directory(root)
    index, index_sha256, _index_identity = _load_index(root)
    metadata = index["metadata"]
    weight_map = index["weight_map"]
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"total_size"}
        or isinstance(metadata.get("total_size"), bool)
        or not isinstance(metadata.get("total_size"), int)
        or metadata["total_size"] <= 0
        or not isinstance(weight_map, dict)
        or not weight_map
    ):
        raise PassthroughAuditError("checkpoint index metadata/weight map is malformed")
    normalized_map: dict[str, str] = {}
    for name, filename in weight_map.items():
        if (
            not isinstance(name, str)
            or _SAFE_TENSOR_RE.fullmatch(name) is None
            or not isinstance(filename, str)
            or _SAFE_FILE_RE.fullmatch(filename) is None
            or not filename.endswith(".safetensors")
        ):
            raise PassthroughAuditError("checkpoint index contains an unsafe mapping")
        normalized_map[name] = filename

    indexed_files = set(normalized_map.values())
    disk_files: set[str] = set()
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise PassthroughAuditError("failed enumerating checkpoint root") from exc
    for path in entries:
        if path.name.endswith(".safetensors"):
            if _SAFE_FILE_RE.fullmatch(path.name) is None:
                raise PassthroughAuditError(f"unsafe safetensors filename: {path.name!r}")
            disk_files.add(path.name)
    if disk_files != indexed_files:
        missing = sorted(indexed_files - disk_files)[:3]
        extra = sorted(disk_files - indexed_files)[:3]
        raise PassthroughAuditError(
            f"indexed/on-disk safetensors set changed: missing={missing}, extra={extra}"
        )

    expected_by_file: dict[str, set[str]] = {name: set() for name in indexed_files}
    for name, filename in normalized_map.items():
        expected_by_file[filename].add(name)
    shards = {name: _read_shard(root / name) for name in sorted(indexed_files)}
    seen: set[str] = set()
    topology: dict[str, dict[str, Any]] = {}
    tensor_bytes = 0
    for filename, shard in shards.items():
        expected = expected_by_file[filename]
        if set(shard.records) != expected:
            raise PassthroughAuditError(f"index/shard tensor inventory changed: {filename}")
        overlap = seen & set(shard.records)
        if overlap:
            raise PassthroughAuditError(f"duplicate tensor across shards: {sorted(overlap)[0]}")
        seen.update(shard.records)
        for name, record in shard.records.items():
            tensor_bytes += record.nbytes
            topology[name] = {
                "bytes": record.nbytes,
                "dtype": record.dtype,
                "shape": list(record.shape),
                "shard": filename,
            }
    if seen != set(normalized_map):
        raise PassthroughAuditError("checkpoint tensor inventory does not close")
    file_bytes = sum(shard.identity.size for shard in shards.values())
    expected_total = file_bytes if final else tensor_bytes
    if metadata["total_size"] != expected_total:
        role = "final file" if final else "source tensor"
        raise PassthroughAuditError(f"checkpoint index {role} total_size changed")
    return CheckpointInventory(
        root=root,
        index_sha256=index_sha256,
        weight_map=normalized_map,
        shards=shards,
        tensor_count=len(normalized_map),
        tensor_bytes=tensor_bytes,
        file_bytes=file_bytes,
        topology_sha256=_sha256_bytes(_canonical_json(topology)),
    )


def _record(inventory: CheckpointInventory, name: str) -> TensorRecord:
    filename = inventory.weight_map.get(name)
    if filename is None:
        raise PassthroughAuditError(f"checkpoint omits tensor: {name}")
    try:
        return inventory.shards[filename].records[name]
    except KeyError as exc:
        raise PassthroughAuditError(f"checkpoint shard omits tensor: {name}") from exc


def _hash_tensor(inventory: CheckpointInventory, name: str) -> str:
    shard = inventory.shards[inventory.weight_map[name]]
    record = shard.records[name]
    descriptor = _open_verified(shard.path, shard.identity)
    digest = hashlib.sha256()
    try:
        remaining = record.nbytes
        position = shard.data_start + record.start
        while remaining:
            amount = min(remaining, HASH_CHUNK_BYTES)
            try:
                chunk = os.pread(descriptor, amount, position)
            except OSError as exc:
                raise PassthroughAuditError(f"failed hashing tensor payload: {name}") from exc
            if len(chunk) != amount:
                raise PassthroughAuditError(f"short tensor payload while hashing: {name}")
            digest.update(chunk)
            remaining -= amount
            position += amount
        if _identity(os.fstat(descriptor)) != shard.identity:
            raise PassthroughAuditError(f"tensor shard changed while hashing: {name}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _source_expert_name(layer: int, projection: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{projection}"


def _quantized_name(layer: int, expert: int, projection: str, component: str) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.{component}"
    )


def _category(name: str, contract: AuditContract) -> str:
    if name.startswith(contract.vision_prefix):
        return "vision"
    if name.startswith(contract.ple_prefix):
        return "ple"
    return "other"


def _contract_payload(contract: AuditContract) -> dict[str, Any]:
    return {
        "expert_intermediate_size": contract.expert_intermediate_size,
        "hidden_size": contract.hidden_size,
        "lm_head": contract.lm_head,
        "mtp_names_sha256": _sha256_bytes(_canonical_json(sorted(contract.mtp_names))),
        "name": contract.name,
        "num_experts": contract.num_experts,
        "num_layers": contract.num_layers,
        "output_tensor_bytes": contract.output_tensor_bytes,
        "output_tensor_count": contract.output_tensor_count,
        "passthrough_category_counts": dict(contract.passthrough_category_counts),
        "passthrough_dtype_bytes": dict(contract.passthrough_dtype_bytes),
        "passthrough_dtype_counts": dict(contract.passthrough_dtype_counts),
        "passthrough_name_set_sha256": contract.passthrough_name_set_sha256,
        "passthrough_tensor_bytes": contract.passthrough_tensor_bytes,
        "passthrough_tensor_count": contract.passthrough_tensor_count,
        "ple_prefix": contract.ple_prefix,
        "source_tensor_bytes": contract.source_tensor_bytes,
        "source_tensor_count": contract.source_tensor_count,
        "vision_prefix": contract.vision_prefix,
        "vocab_size": contract.vocab_size,
    }


def _validate_contract(contract: AuditContract) -> dict[str, Any]:
    expected_quantized = contract.num_layers * contract.num_experts * 3 * 4
    expected_source = (
        contract.passthrough_tensor_count
        + contract.num_layers * 2
        + len(contract.mtp_names)
        + 1
    )
    expected_output = (
        contract.passthrough_tensor_count
        + len(contract.mtp_names)
        + 1
        + expected_quantized
    )
    count_values = (
        list(contract.passthrough_category_counts.values())
        + list(contract.passthrough_dtype_counts.values())
        + list(contract.passthrough_dtype_bytes.values())
    )
    if (
        min(
            contract.num_layers,
            contract.num_experts,
            contract.hidden_size,
            contract.expert_intermediate_size,
            contract.vocab_size,
            contract.source_tensor_bytes,
            contract.output_tensor_bytes,
            contract.passthrough_tensor_bytes,
        )
        <= 0
        or contract.source_tensor_count != expected_source
        or contract.output_tensor_count != expected_output
        or sum(contract.passthrough_category_counts.values())
        != contract.passthrough_tensor_count
        or sum(contract.passthrough_dtype_counts.values())
        != contract.passthrough_tensor_count
        or sum(contract.passthrough_dtype_bytes.values())
        != contract.passthrough_tensor_bytes
        or set(contract.passthrough_category_counts) != {"other", "ple", "vision"}
        or set(contract.passthrough_dtype_counts) != set(contract.passthrough_dtype_bytes)
        or not re.fullmatch(r"[0-9a-f]{64}", contract.passthrough_name_set_sha256)
        or not contract.mtp_names
        or contract.lm_head in contract.mtp_names
        or contract.hidden_size % 16
        or contract.expert_intermediate_size % 16
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in count_values
        )
        or any(
            not isinstance(name, str)
            or not name.startswith("mtp.")
            or _SAFE_TENSOR_RE.fullmatch(name) is None
            for name in contract.mtp_names
        )
    ):
        raise PassthroughAuditError("audit contract is internally inconsistent")
    payload = _contract_payload(contract)
    return {
        "details": payload,
        "sha256": _sha256_bytes(_canonical_json(payload)),
    }


def _validate_source(
    source: CheckpointInventory, contract: AuditContract
) -> tuple[list[str], dict[str, TensorRecord]]:
    if (
        source.tensor_count != contract.source_tensor_count
        or source.tensor_bytes != contract.source_tensor_bytes
    ):
        raise PassthroughAuditError("source hybrid tensor count/bytes changed")
    source_names = set(source.weight_map)
    expected_source_experts = {
        _source_expert_name(layer, projection)
        for layer in range(contract.num_layers)
        for projection in ("gate_up_proj", "down_proj")
    }
    actual_source_experts = {
        name for name in source_names if _SOURCE_EXPERT_RE.fullmatch(name)
    }
    if actual_source_experts != expected_source_experts:
        raise PassthroughAuditError("source routed-expert tensor set changed")
    if {name for name in source_names if name.startswith("mtp.")} != set(
        contract.mtp_names
    ):
        raise PassthroughAuditError("source MTP tensor set changed")
    for layer in range(contract.num_layers):
        gate = _record(source, _source_expert_name(layer, "gate_up_proj"))
        down = _record(source, _source_expert_name(layer, "down_proj"))
        if (gate.dtype, gate.shape) != (
            "BF16",
            (
                contract.num_experts,
                2 * contract.expert_intermediate_size,
                contract.hidden_size,
            ),
        ) or (down.dtype, down.shape) != (
            "BF16",
            (
                contract.num_experts,
                contract.hidden_size,
                contract.expert_intermediate_size,
            ),
        ):
            raise PassthroughAuditError("source routed-expert dtype/shape changed")
    head = _record(source, contract.lm_head)
    if (head.dtype, head.shape) != (
        "BF16",
        (contract.vocab_size, contract.hidden_size),
    ):
        raise PassthroughAuditError("source lm_head dtype/shape changed")
    passthrough = sorted(
        source_names - expected_source_experts - set(contract.mtp_names) - {contract.lm_head}
    )
    if (
        len(passthrough) != contract.passthrough_tensor_count
        or _sha256_bytes(_canonical_json(passthrough))
        != contract.passthrough_name_set_sha256
    ):
        raise PassthroughAuditError("canonical pass-through tensor name set changed")
    records = {name: _record(source, name) for name in passthrough}
    categories: dict[str, int] = {"other": 0, "ple": 0, "vision": 0}
    dtype_counts: dict[str, int] = {}
    dtype_bytes: dict[str, int] = {}
    for name, record in records.items():
        category = _category(name, contract)
        categories[category] += 1
        dtype_counts[record.dtype] = dtype_counts.get(record.dtype, 0) + 1
        dtype_bytes[record.dtype] = dtype_bytes.get(record.dtype, 0) + record.nbytes
    if (
        categories != dict(contract.passthrough_category_counts)
        or dtype_counts != dict(contract.passthrough_dtype_counts)
        or dtype_bytes != dict(contract.passthrough_dtype_bytes)
        or sum(record.nbytes for record in records.values())
        != contract.passthrough_tensor_bytes
    ):
        raise PassthroughAuditError("pass-through category/dtype/byte contract changed")
    return passthrough, records


def _validate_output(
    output: CheckpointInventory,
    source: CheckpointInventory,
    passthrough: list[str],
    source_records: Mapping[str, TensorRecord],
    contract: AuditContract,
) -> None:
    if (
        output.tensor_count != contract.output_tensor_count
        or output.tensor_bytes != contract.output_tensor_bytes
    ):
        raise PassthroughAuditError("final checkpoint tensor count/bytes changed")
    output_names = set(output.weight_map)
    passthrough_set = set(passthrough)
    fixed = passthrough_set | set(contract.mtp_names) | {contract.lm_head}
    if not fixed <= output_names:
        raise PassthroughAuditError("final checkpoint omits preserved/tuned tensors")
    quantized = output_names - fixed
    expected_quantized_count = contract.num_layers * contract.num_experts * 3 * 4
    if len(quantized) != expected_quantized_count:
        raise PassthroughAuditError("final quantized tensor count changed")
    for name in quantized:
        match = _OUTPUT_EXPERT_RE.fullmatch(name)
        if match is None:
            raise PassthroughAuditError(f"unexpected final tensor outside closed topology: {name}")
        layer, expert = int(match.group(1)), int(match.group(2))
        projection, component = match.group(3), match.group(4)
        if not (0 <= layer < contract.num_layers and 0 <= expert < contract.num_experts):
            raise PassthroughAuditError(f"quantized expert coordinate is out of range: {name}")
        rows, columns = (
            (contract.expert_intermediate_size, contract.hidden_size)
            if projection in {"gate_proj", "up_proj"}
            else (contract.hidden_size, contract.expert_intermediate_size)
        )
        expected = {
            "weight": ("U8", (rows, columns // 2)),
            "weight_scale": ("F8_E4M3", (rows, columns // 16)),
            "weight_scale_2": ("F32", ()),
            "input_scale": ("F32", ()),
        }[component]
        record = _record(output, name)
        if (record.dtype, record.shape) != expected:
            raise PassthroughAuditError(f"quantized expert dtype/shape changed: {name}")
        expected_shard = (
            f"model-routed-experts-{layer + 1:05d}-of-{contract.num_layers:05d}."
            "safetensors"
        )
        if output.weight_map[name] != expected_shard:
            raise PassthroughAuditError(f"quantized expert shard mapping changed: {name}")
    for name in passthrough:
        if output.weight_map[name] != source.weight_map[name]:
            raise PassthroughAuditError(f"pass-through shard mapping changed: {name}")
        record = _record(output, name)
        source_record = source_records[name]
        if (record.dtype, record.shape, record.nbytes) != (
            source_record.dtype,
            source_record.shape,
            source_record.nbytes,
        ):
            raise PassthroughAuditError(f"pass-through dtype/shape changed: {name}")
    for name in contract.mtp_names:
        source_record = _record(source, name)
        output_record = _record(output, name)
        if (
            source_record.dtype != "BF16"
            or output_record.dtype != "BF16"
            or (source_record.shape, source_record.nbytes)
            != (output_record.shape, output_record.nbytes)
            or output.weight_map[name] != "model-mtp-bf16.safetensors"
        ):
            raise PassthroughAuditError(f"final MTP metadata/shard changed: {name}")
    head = _record(output, contract.lm_head)
    if (
        (head.dtype, head.shape)
        != ("BF16", (contract.vocab_size, contract.hidden_size))
        or output.weight_map[contract.lm_head] != "model-lm-head-bf16.safetensors"
    ):
        raise PassthroughAuditError("final tuned lm_head metadata/shard changed")


def _revalidate(inventory: CheckpointInventory) -> None:
    index, digest, _identity_value = _load_index(inventory.root)
    if digest != inventory.index_sha256 or index["weight_map"] != inventory.weight_map:
        raise PassthroughAuditError("checkpoint index changed during audit")
    disk_files = {
        path.name
        for path in inventory.root.iterdir()
        if path.name.endswith(".safetensors")
    }
    if disk_files != set(inventory.shards):
        raise PassthroughAuditError("on-disk safetensors set changed during audit")
    for shard in inventory.shards.values():
        current = _safe_file(shard.path)
        if current != shard.identity:
            raise PassthroughAuditError(f"checkpoint shard changed during audit: {shard.path}")


def _write_exclusive_private(path: Path, payload: bytes) -> None:
    parent = _private_directory(path.parent)
    destination = parent / path.name
    if _SAFE_FILE_RE.fullmatch(destination.name) is None:
        raise PassthroughAuditError("receipt filename is unsafe")
    if destination.exists() or destination.is_symlink():
        raise PassthroughAuditError("receipt already exists; refusing to overwrite")
    temporary = parent / f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    linked = False
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PassthroughAuditError("receipt write was incomplete")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as exc:
        raise PassthroughAuditError("receipt already exists; refusing to overwrite") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if linked:
            directory_descriptor = os.open(
                parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)


def audit_passthrough(
    hybrid: Path,
    checkpoint: Path,
    receipt: Path,
    *,
    contract: AuditContract = PRODUCTION_CONTRACT,
) -> dict[str, Any]:
    """Audit and persist an external, write-once pass-through receipt."""

    contract_receipt = _validate_contract(contract)
    script_path = Path(__file__).resolve(strict=True)
    script_sha256 = _sha256_bytes(script_path.read_bytes())
    source_root = _private_directory(hybrid)
    output_root = _private_directory(checkpoint)
    if source_root == output_root:
        raise PassthroughAuditError("source hybrid and final checkpoint must be distinct")
    receipt_resolved = receipt.resolve(strict=False)
    if receipt_resolved == source_root or source_root in receipt_resolved.parents:
        raise PassthroughAuditError("receipt must be outside the source hybrid")
    if receipt_resolved == output_root or output_root in receipt_resolved.parents:
        raise PassthroughAuditError("receipt must be outside the final checkpoint")

    source = _inventory(source_root, final=False)
    output = _inventory(output_root, final=True)
    passthrough, source_records = _validate_source(source, contract)
    _validate_output(output, source, passthrough, source_records, contract)

    payload_inventory: dict[str, dict[str, Any]] = {}
    category_payloads: dict[str, dict[str, str]] = {
        "other": {},
        "ple": {},
        "vision": {},
    }
    for index, name in enumerate(passthrough, 1):
        source_sha256 = _hash_tensor(source, name)
        output_sha256 = _hash_tensor(output, name)
        if source_sha256 != output_sha256:
            raise PassthroughAuditError(f"pass-through tensor payload changed: {name}")
        record = source_records[name]
        payload_inventory[name] = {
            "bytes": record.nbytes,
            "dtype": record.dtype,
            "sha256": source_sha256,
            "shape": list(record.shape),
        }
        category_payloads[_category(name, contract)][name] = source_sha256
        if index % 64 == 0 or index == len(passthrough):
            print(
                json.dumps(
                    {
                        "event": "passthrough_audit_progress",
                        "tensor_count": index,
                        "tensor_total": len(passthrough),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    _revalidate(source)
    _revalidate(output)
    if _sha256_bytes(script_path.read_bytes()) != script_sha256:
        raise PassthroughAuditError("auditor source changed during audit")
    inventory_sha256 = _sha256_bytes(_canonical_json(payload_inventory))
    category_sha256 = {
        name: _sha256_bytes(_canonical_json(values))
        for name, values in sorted(category_payloads.items())
    }
    result = {
        "auditor": {
            "file": script_path.name,
            "sha256": script_sha256,
        },
        "checkpoint": {
            "file_bytes": output.file_bytes,
            "index_sha256": output.index_sha256,
            "safetensors_file_count": len(output.shards),
            "tensor_bytes": output.tensor_bytes,
            "tensor_count": output.tensor_count,
            "topology_sha256": output.topology_sha256,
        },
        "complete": True,
        "contract": contract_receipt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "passthrough": {
            "canonical_name_set_sha256": contract.passthrough_name_set_sha256,
            "category_counts": dict(contract.passthrough_category_counts),
            "category_payload_sha256": category_sha256,
            "dtype_bytes": dict(contract.passthrough_dtype_bytes),
            "dtype_counts": dict(contract.passthrough_dtype_counts),
            "exact_raw_payload_identity": True,
            "payload_inventory_sha256": inventory_sha256,
            "shard_mapping_preserved": True,
            "tensor_bytes": contract.passthrough_tensor_bytes,
            "tensor_count": contract.passthrough_tensor_count,
        },
        "schema_version": SCHEMA_VERSION,
        "source_hybrid": {
            "file_bytes": source.file_bytes,
            "index_sha256": source.index_sha256,
            "safetensors_file_count": len(source.shards),
            "tensor_bytes": source.tensor_bytes,
            "tensor_count": source.tensor_count,
            "topology_sha256": source.topology_sha256,
        },
    }
    _write_exclusive_private(receipt, _canonical_json(result))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prove exact Qwen3.8-Flash-Next pass-through tensor identity"
    )
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = audit_passthrough(
            arguments.hybrid,
            arguments.checkpoint,
            arguments.receipt,
        )
    except PassthroughAuditError as exc:
        print(f"passthrough audit failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "checkpoint_index_sha256": result["checkpoint"]["index_sha256"],
                "passed": True,
                "payload_inventory_sha256": result["passthrough"][
                    "payload_inventory_sha256"
                ],
                "receipt": str(arguments.receipt),
                "schema_version": SCHEMA_VERSION,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
