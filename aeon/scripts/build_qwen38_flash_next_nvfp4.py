#!/usr/bin/env python3
"""Build the final Qwen3.8-Flash-Next routed-expert NVFP4 checkpoint.

The input is a hash-audited tensor-level hybrid: the immutable official BF16
checkpoint except for the 128 official FP8 PLE n-gram table shards.  This
builder merges a deliberately small ``lm_head`` LoRA in BF16, converts only
the 48 main-model routed-expert containers with NVIDIA ModelOpt 0.46, and
grafts the exact 31 official BF16 ``mtp.*`` tensors.  Vision, video, PLE, MTP,
and every non-routed language tensor are outside the quantization target.

Large inputs are never loaded as one model.  Safetensors shards and expert
layers are processed independently, the output is validated while it is still
private, and the final path is published with one atomic rename.  CUDA and
ModelOpt imports are lazy so the preflight and unit tests run on CPU-only hosts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import sys
import time
from typing import Any, Callable, Iterable, Mapping


SCHEMA_VERSION = "aeon-qwen38-flash-next-modelopt-nvfp4-v1"
HYBRID_SCHEMA = "aeon-qwen38-flash-next-hybrid-v1"
ADAPTER_SCHEMA = "aeon-qwen38-flash-next-lm-head-lora-v1"
OFFICIAL_BASELINE_SCHEMA = (
    "aeon-qwen38-flash-next-official-untuned-behavior-baseline-spec-v1"
)
BEHAVIOR_JUDGMENT_SCHEMA = "aeon-qwen38-flash-next-behavior-judgment-v2"
OFFICIAL_BASELINE_FILENAME = "official_untuned_behavior_baseline_spec.json"
SETTLED_BASELINE_FILENAME = "OFFICIAL_UNTUNED_BASELINE_SPEC.json"
TUNED_LM_HEAD_FILENAME = "model-lm-head-bf16.safetensors"
UNTUNED_LM_HEAD_FILENAME = "official-untuned-lm-head-bf16.safetensors"
SIBLING_MANIFEST_FILENAME = "BUILD_SIBLING_MANIFEST.json"
SIBLING_SCHEMA = "aeon-qwen38-flash-next-official-untuned-sibling-v1"
SUBSET_SCHEMA = "aeon-hf-safetensors-subset-v1"
HYBRID_MANIFEST_FILENAME = "HYBRID_MANIFEST.json"

BF16_REPO = "Qwen/Qwen3.8-Flash-Next"
BF16_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
BF16_CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
BF16_INDEX_SHA256 = "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de"
FP8_REPO = "Qwen/Qwen3.8-Flash-Next-FP8"
FP8_REVISION = "bcd9f01ddc9cff2316eb84281bebcd5b058bddce"
SCALE_REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
SCALE_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"

MODELOPT_VERSION = "0.46.0"
MODELOPT_WHEEL_SHA256 = (
    "1864b4e9921e287b065be3861ab48345144e673273ebb2b94bd9a6119a9eba8e"
)
MODELOPT_COMMIT = "43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a"

NUM_LAYERS = 48
NUM_EXPERTS = 512
HIDDEN_SIZE = 2560
EXPERT_INTERMEDIATE_SIZE = 640
VOCAB_SIZE = 248_320
MTP_TENSOR_COUNT = 31
VISION_TENSOR_COUNT = 333
PLE_TABLE_COUNT = 128
SOURCE_EXPERT_TENSOR_COUNT = NUM_LAYERS * 2
QUANTIZED_MODULE_COUNT = NUM_LAYERS * NUM_EXPERTS * 3
QUANTIZED_COMPONENT_COUNT = QUANTIZED_MODULE_COUNT * 4
SCALE_TENSOR_COUNT = QUANTIZED_MODULE_COUNT * 2
EXPECTED_HYBRID_TENSOR_COUNT = 1_659
EXPECTED_NON_EXPERT_NON_MTP_COUNT = 1_532
EXPECTED_OUTPUT_TENSOR_COUNT = (
    EXPECTED_NON_EXPERT_NON_MTP_COUNT + MTP_TENSOR_COUNT + QUANTIZED_COMPONENT_COUNT
)
MODELOPT_IGNORE = (
    "model.embed_tokens",
    "mtp.*",
    "model.mtp.*",
    "*.self_attn.*",
    "*.linear_attn.*",
    "*.mlp.gate*",
    "*.mlp.shared_expert.*",
    "*.mlp.shared_expert_gate*",
    "*hyper_connection*",
    "*.ple.*",
    "model.visual.*",
    "model.language_model.embed_tokens",
    "lm_head",
)

# Exact contract consumed by ``_validate_hybrid``.  The Fleet preparation job
# must emit these keys (and no others); each ``files`` value is an object with
# exactly ``{"sha256": <64 lowercase hex>, "size": <positive integer>}``.
# ``files`` must cover config.json, model.safetensors.index.json, every and only
# indexed safetensors shard, plus every tokenizer/preprocessor file intended to
# survive into the final artifact.
HYBRID_MANIFEST_CONTRACT: dict[str, Any] = {
    "schema_version": HYBRID_SCHEMA,
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
        "tensor_count": EXPECTED_HYBRID_TENSOR_COUNT,
        "bf16_source_expert_tensor_count": SOURCE_EXPERT_TENSOR_COUNT,
        "bf16_mtp_tensor_count": MTP_TENSOR_COUNT,
        "bf16_vision_tensor_count": VISION_TENSOR_COUNT,
        "fp8_ple_table_tensor_count": PLE_TABLE_COUNT,
        "bf16_ple_scale_tensor_count": 1,
        "non_expert_non_mtp_tensor_count": EXPECTED_NON_EXPERT_NON_MTP_COUNT,
    },
    "files": "<closed file-receipt map described above>",
}

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

LM_HEAD = "lm_head.weight"
LORA_A = "base_model.model.lm_head.lora_A.weight"
LORA_B = "base_model.model.lm_head.lora_B.weight"
VISION_PREFIX = "model.visual."
PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
PLE_SCALE = PLE_PREFIX + "weight_scale"

_SOURCE_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)$"
)
_OUTPUT_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_2|input_scale)$"
)
_PLE_TABLE_RE = re.compile(re.escape(PLE_PREFIX) + r"shard_(\d+)\.weight$")
_CUDA_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$"
)
_CLAIM_RE = re.compile(r"^gc-[A-Za-z0-9._:-]{8,200}$")
_RUNTIME_RE = re.compile(r"^fr-[a-f0-9]{32}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_SAFE_NAME_RE = re.compile(
    r"^(?!\.\.?$)(?!.*\.\.)(?!.*[/\\\x00-\x1f])[A-Za-z0-9_.-]{1,240}$"
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


class BuildError(RuntimeError):
    """An input, runtime binding, quantized tensor, or output failed closed."""


@dataclass(frozen=True)
class TensorRecord:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


@dataclass(frozen=True)
class TensorLocation:
    path: Path
    record: TensorRecord


@dataclass
class LayerScales:
    gate_up_input: Any
    gate_up_weight_scale_2: Any
    down_input: Any
    down_weight_scale_2: Any


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BuildError("value is not canonical JSON") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, maximum: int | None = None) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BuildError(f"required file is absent: {path}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o022
        or metadata.st_size <= 0
        or (maximum is not None and metadata.st_size > maximum)
    ):
        raise BuildError(f"file is not a safe, immutable owner file: {path}")
    return metadata


def _private_directory(path: Path, *, create: bool = False) -> Path:
    if create:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BuildError(f"directory is absent: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise BuildError(f"directory is not private and owned: {path}")
    return path


def _read_json(path: Path, *, maximum: int = 64 * 1024 * 1024) -> dict[str, Any]:
    _regular_file(path, maximum=maximum)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"JSON is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise BuildError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    )
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise BuildError(f"write was incomplete: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _expected_digest(value: str, label: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise BuildError(f"{label} must be one lowercase SHA-256")
    return value


def _load_manifest(path: Path, expected_sha256: str, schema: str) -> dict[str, Any]:
    expected = _expected_digest(expected_sha256, f"{path.name} manifest digest")
    if _sha256(path) != expected:
        raise BuildError(f"manifest SHA-256 changed: {path}")
    value = _read_json(path)
    if value.get("schema_version") != schema or value.get("complete") is not True:
        raise BuildError(f"manifest is not a complete {schema} receipt: {path}")
    return value


def _normalize_file_receipts(value: Any) -> dict[str, tuple[str, int]]:
    if not isinstance(value, dict) or not value:
        raise BuildError("artifact manifest has no file receipts")
    result: dict[str, tuple[str, int]] = {}
    for name, receipt in value.items():
        if not isinstance(name, str) or _SAFE_NAME_RE.fullmatch(name) is None:
            raise BuildError("artifact manifest contains an unsafe filename")
        if not isinstance(receipt, dict) or set(receipt) != {"sha256", "size"}:
            raise BuildError(f"file receipt is malformed: {name}")
        digest = receipt.get("sha256")
        size = receipt.get("size")
        if (
            not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or type(size) is not int
            or size <= 0
        ):
            raise BuildError(f"file receipt is malformed: {name}")
        result[name] = (digest, size)
    return result


def _verify_file_receipts(root: Path, receipts: Mapping[str, tuple[str, int]]) -> None:
    for name, (digest, size) in sorted(receipts.items()):
        path = root / name
        metadata = _regular_file(path)
        if metadata.st_size != size or _sha256(path) != digest:
            raise BuildError(f"artifact file identity changed: {name}")


def _tensor_descriptor(name: str, value: Any, data_start: int) -> TensorRecord:
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
        raise BuildError(f"safetensors descriptor is malformed: {name}")
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
        raise BuildError(f"safetensors descriptor is malformed: {name}")
    expected = math.prod(shape) * _DTYPE_BYTES[dtype]
    if offsets[1] - offsets[0] != expected:
        raise BuildError(f"safetensors tensor byte count changed: {name}")
    return TensorRecord(
        dtype=dtype,
        shape=tuple(shape),
        start=data_start + offsets[0],
        end=data_start + offsets[1],
    )


def _read_safetensors_header(
    path: Path,
) -> tuple[dict[str, str], dict[str, TensorRecord]]:
    metadata = _regular_file(path)
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise BuildError(f"safetensors prefix is truncated: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= 256 * 1024 * 1024 or header_size % 8:
            raise BuildError(f"safetensors header size is invalid: {path}")
        raw_header = handle.read(header_size)
    if len(raw_header) != header_size:
        raise BuildError(f"safetensors header is truncated: {path}")
    try:
        header = json.loads(raw_header.rstrip(b" "))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"safetensors header is malformed: {path}") from exc
    if not isinstance(header, dict):
        raise BuildError(f"safetensors header root is invalid: {path}")
    raw_metadata = header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in raw_metadata.items()
    ):
        raise BuildError(f"safetensors metadata is malformed: {path}")
    data_start = 8 + header_size
    records: dict[str, TensorRecord] = {}
    for name, value in header.items():
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 2048
            or any(character in name for character in "\x00\r\n")
        ):
            raise BuildError(f"unsafe safetensors tensor name: {path}")
        records[name] = _tensor_descriptor(name, value, data_start)
    ordered = sorted(records.items(), key=lambda item: (item[1].start, item[0]))
    cursor = data_start
    for name, record in ordered:
        if record.start != cursor:
            raise BuildError(f"safetensors data is not closed and contiguous: {name}")
        cursor = record.end
    if not records or cursor != metadata.st_size:
        raise BuildError(f"safetensors data length is inconsistent: {path}")
    return dict(raw_metadata), records


def _tensor_sha256(location: TensorLocation) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(location.path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        offset = location.record.start
        remaining = location.record.end - location.record.start
        while remaining:
            chunk = os.pread(descriptor, min(8 * 1024 * 1024, remaining), offset)
            if not chunk:
                raise BuildError(f"tensor data is truncated: {location.path}")
            digest.update(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _source_expert_name(layer: int, projection: str) -> str:
    if projection not in {"gate_up_proj", "down_proj"}:
        raise BuildError("unknown source expert projection")
    return f"model.language_model.layers.{layer}.mlp.experts.{projection}"


def _expert_module(layer: int, expert: int, projection: str) -> str:
    if projection not in {"gate_proj", "up_proj", "down_proj"}:
        raise BuildError("unknown output expert projection")
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"


def _expert_source_layer(name: str) -> int | None:
    match = _SOURCE_EXPERT_RE.fullmatch(name)
    if match is None:
        return None
    layer = int(match.group(1))
    return layer if 0 <= layer < NUM_LAYERS else None


def _ple_table_names() -> frozenset[str]:
    return frozenset(
        PLE_PREFIX + f"shard_{index}.weight" for index in range(PLE_TABLE_COUNT)
    )


def _expected_scale_names() -> frozenset[str]:
    return frozenset(
        f"{_expert_module(layer, expert, projection)}.{suffix}"
        for layer in range(NUM_LAYERS)
        for expert in range(NUM_EXPERTS)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("input_scale", "weight_scale_2")
    )


def _expected_quantized_names() -> frozenset[str]:
    return frozenset(
        f"{_expert_module(layer, expert, projection)}.{suffix}"
        for layer in range(NUM_LAYERS)
        for expert in range(NUM_EXPERTS)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")
    )


def _is_quantization_artifact(name: str) -> bool:
    return name.endswith(
        (
            ".weight_scale_inv",
            ".weight_scale_2",
            ".input_scale",
            ".weight_packed",
            ".input_global_scale",
            ".weight_global_scale",
        )
    )


def _validate_config(config: dict[str, Any]) -> None:
    text = config.get("text_config")
    vision = config.get("vision_config")
    if (
        config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]
        or config.get("model_type") != "qwen4_exp"
        or config.get("language_model_only") is not False
        or not isinstance(text, dict)
        or not isinstance(vision, dict)
        or text.get("model_type") != "qwen4_exp_text"
        or text.get("dtype") != "bfloat16"
        or text.get("num_hidden_layers") != NUM_LAYERS
        or text.get("num_experts") != NUM_EXPERTS
        or text.get("hidden_size") != HIDDEN_SIZE
        or text.get("moe_intermediate_size") != EXPERT_INTERMEDIATE_SIZE
        or text.get("vocab_size") != VOCAB_SIZE
        or text.get("num_experts_per_tok") != 10
        or text.get("mtp_num_hidden_layers") != 1
        or text.get("split_ngram_parts") != PLE_TABLE_COUNT
        or text.get("ngram_size") != 3
        or text.get("ple_layer_ids") != [2]
        or vision.get("depth") != 27
        or vision.get("out_hidden_size") != HIDDEN_SIZE
        or config.get("image_token_id") != 248056
        or config.get("video_token_id") != 248057
    ):
        raise BuildError("hybrid config is not the pinned Qwen3.8-Flash-Next topology")
    mtp = text.get("mtp")
    if not isinstance(mtp, dict) or mtp.get("num_hidden_layers") != 1:
        raise BuildError("hybrid config lost the native one-layer MTP topology")
    # The hybrid is tensor-level, not a whole-model FP8/quantized checkpoint.
    # Its final quantization config is generated below from a closed module set.
    if "quantization_config" in config:
        raise BuildError("hybrid config unexpectedly contains whole-model quantization")


def _validate_index(index: dict[str, Any]) -> dict[str, str]:
    weight_map = index.get("weight_map")
    if (
        not isinstance(weight_map, dict)
        or len(weight_map) != EXPECTED_HYBRID_TENSOR_COUNT
        or not all(
            isinstance(name, str)
            and name
            and isinstance(shard, str)
            and _SAFE_NAME_RE.fullmatch(shard) is not None
            and shard.endswith(".safetensors")
            for name, shard in weight_map.items()
        )
    ):
        raise BuildError("hybrid model index is malformed or has an unexpected size")
    return dict(weight_map)


def _index_safetensors(
    root: Path, weight_map: Mapping[str, str]
) -> tuple[
    dict[str, TensorLocation],
    dict[str, dict[str, str]],
    dict[str, tuple[str, ...]],
]:
    locations: dict[str, TensorLocation] = {}
    metadata_by_shard: dict[str, dict[str, str]] = {}
    names_by_shard: dict[str, tuple[str, ...]] = {}
    for shard in sorted(set(weight_map.values())):
        metadata, records = _read_safetensors_header(root / shard)
        indexed = sorted(
            name for name, filename in weight_map.items() if filename == shard
        )
        if set(records) != set(indexed):
            missing = sorted(set(indexed) - set(records))[:3]
            extra = sorted(set(records) - set(indexed))[:3]
            raise BuildError(
                f"hybrid index/shard mismatch for {shard}: missing={missing}, extra={extra}"
            )
        metadata_by_shard[shard] = metadata
        names_by_shard[shard] = tuple(indexed)
        for name, record in records.items():
            if name in locations:
                raise BuildError(f"duplicate tensor in hybrid shards: {name}")
            locations[name] = TensorLocation(root / shard, record)
    if set(locations) != set(weight_map):
        raise BuildError("hybrid tensor inventory does not close against its index")
    return locations, metadata_by_shard, names_by_shard


def _validate_hybrid(
    root: Path, manifest: dict[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, TensorLocation],
    dict[str, dict[str, str]],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, int]],
]:
    if set(manifest) != {
        "schema_version",
        "complete",
        "artifact",
        "sources",
        "upstream_metadata",
        "topology",
        "files",
    }:
        raise BuildError("hybrid manifest has missing or unreviewed top-level keys")
    if manifest.get("artifact") != "qwen38-flash-next-tensor-hybrid":
        raise BuildError("hybrid artifact name changed")
    sources = manifest.get("sources")
    if sources != {
        "bf16": {"repo": BF16_REPO, "revision": BF16_REVISION},
        "fp8_ple": {"repo": FP8_REPO, "revision": FP8_REVISION},
    }:
        raise BuildError(
            "hybrid source revisions are not the pinned official revisions"
        )
    upstream = manifest.get("upstream_metadata")
    if upstream != {
        "bf16_config_sha256": BF16_CONFIG_SHA256,
        "bf16_index_sha256": BF16_INDEX_SHA256,
    }:
        raise BuildError("hybrid did not bind the pinned official BF16 metadata")
    if manifest.get("topology") != HYBRID_MANIFEST_CONTRACT["topology"]:
        raise BuildError("hybrid manifest topology receipt changed")
    receipts = _normalize_file_receipts(manifest.get("files"))
    if "config.json" not in receipts or "model.safetensors.index.json" not in receipts:
        raise BuildError("hybrid manifest omits its config or index")
    if receipts["config.json"][0] != BF16_CONFIG_SHA256:
        raise BuildError("hybrid config is not byte-exact official BF16 metadata")
    _verify_file_receipts(root, receipts)
    config = _read_json(root / "config.json")
    _validate_config(config)
    index = _read_json(root / "model.safetensors.index.json")
    weight_map = _validate_index(index)
    receipted_shards = {name for name in receipts if name.endswith(".safetensors")}
    if set(weight_map.values()) != receipted_shards:
        raise BuildError(
            "hybrid safetensors receipt set does not close against the index"
        )
    locations, metadata_by_shard, names_by_shard = _index_safetensors(root, weight_map)

    source_experts = {
        name for name in locations if _expert_source_layer(name) is not None
    }
    expected_experts = {
        _source_expert_name(layer, projection)
        for layer in range(NUM_LAYERS)
        for projection in ("gate_up_proj", "down_proj")
    }
    if source_experts != expected_experts:
        raise BuildError(
            "hybrid does not contain exactly the 96 BF16 expert containers"
        )
    for layer in range(NUM_LAYERS):
        gate_up = locations[_source_expert_name(layer, "gate_up_proj")].record
        down = locations[_source_expert_name(layer, "down_proj")].record
        if gate_up.dtype != "BF16" or gate_up.shape != (
            NUM_EXPERTS,
            2 * EXPERT_INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        ):
            raise BuildError(f"BF16 gate/up expert container changed at layer {layer}")
        if down.dtype != "BF16" or down.shape != (
            NUM_EXPERTS,
            HIDDEN_SIZE,
            EXPERT_INTERMEDIATE_SIZE,
        ):
            raise BuildError(f"BF16 down expert container changed at layer {layer}")

    mtp = {name for name in locations if name.startswith("mtp.")}
    if mtp != MTP_NAMES:
        raise BuildError("hybrid MTP tensor set is not the exact 31-tensor BF16 set")
    if any(locations[name].record.dtype != "BF16" for name in MTP_NAMES):
        raise BuildError("hybrid MTP contains a non-BF16 tensor")
    vision = {name for name in locations if name.startswith(VISION_PREFIX)}
    if len(vision) != VISION_TENSOR_COUNT:
        raise BuildError("hybrid vision tensor count changed")
    if any(locations[name].record.dtype != "BF16" for name in vision):
        raise BuildError("hybrid vision stack is not BF16")
    ple_tables = {name for name in locations if _PLE_TABLE_RE.fullmatch(name)}
    if ple_tables != _ple_table_names():
        raise BuildError("hybrid PLE table set changed")
    for name in ple_tables:
        match = _PLE_TABLE_RE.fullmatch(name)
        assert match is not None
        record = locations[name].record
        if record.dtype != "F8_E4M3" or record.shape != (2_500_012, 160):
            raise BuildError(f"official FP8 PLE tensor changed: {name}")
    ple_scale = locations.get(PLE_SCALE)
    if (
        ple_scale is None
        or ple_scale.record.dtype != "BF16"
        or ple_scale.record.shape != (1,)
    ):
        raise BuildError("official FP8 PLE weight scale changed")
    head = locations.get(LM_HEAD)
    if (
        head is None
        or head.record.dtype != "BF16"
        or head.record.shape
        != (
            VOCAB_SIZE,
            HIDDEN_SIZE,
        )
    ):
        raise BuildError("hybrid lm_head is not the expected BF16 matrix")
    stale = {
        name
        for name in locations
        if _OUTPUT_EXPERT_RE.fullmatch(name) or _is_quantization_artifact(name)
    }
    # PLE's reviewed BF16 weight_scale is semantic model data, not a stale
    # ModelOpt component, and is intentionally not caught by the suffixes above.
    if stale:
        raise BuildError(
            f"hybrid contains stale quantization tensors: {sorted(stale)[:3]}"
        )
    if (
        len(locations) - len(source_experts) - len(mtp)
        != EXPECTED_NON_EXPERT_NON_MTP_COUNT
    ):
        raise BuildError("hybrid non-expert tensor count changed")
    return (
        config,
        index,
        weight_map,
        locations,
        metadata_by_shard,
        names_by_shard,
        receipts,
    )


def _validate_subset_manifest(
    *,
    path: Path,
    manifest: dict[str, Any],
    repo: str,
    revision: str,
    expected_names: frozenset[str],
) -> tuple[dict[str, TensorRecord], dict[str, str]]:
    if manifest.get("source") != {"repo": repo, "revision": revision}:
        raise BuildError("subset source revision changed")
    if (
        manifest.get("tensor_count") != len(expected_names)
        or manifest.get("output_file") != path.name
        or manifest.get("output_bytes") != _regular_file(path).st_size
        or manifest.get("output_sha256") != _sha256(path)
    ):
        raise BuildError("subset payload identity or tensor count changed")
    hashes = manifest.get("tensor_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(expected_names)
        or not all(
            isinstance(value, str) and _SHA256_RE.fullmatch(value)
            for value in hashes.values()
        )
    ):
        raise BuildError("subset tensor hash inventory changed")
    _metadata, records = _read_safetensors_header(path)
    if set(records) != set(expected_names):
        raise BuildError("subset safetensors inventory changed")
    for name in sorted(expected_names):
        if _tensor_sha256(TensorLocation(path, records[name])) != hashes[name]:
            raise BuildError(f"subset tensor SHA-256 changed: {name}")
    return records, dict(hashes)


def _validate_mtp_subset(
    path: Path, manifest: dict[str, Any], hybrid: Mapping[str, TensorLocation]
) -> tuple[dict[str, TensorRecord], dict[str, str]]:
    records, hashes = _validate_subset_manifest(
        path=path,
        manifest=manifest,
        repo=BF16_REPO,
        revision=BF16_REVISION,
        expected_names=MTP_NAMES,
    )
    if any(record.dtype != "BF16" for record in records.values()):
        raise BuildError("official MTP subset contains a non-BF16 tensor")
    for name in sorted(MTP_NAMES):
        if _tensor_sha256(hybrid[name]) != hashes[name]:
            raise BuildError(f"hybrid and official BF16 MTP differ: {name}")
    return records, hashes


def _validate_scale_subset(
    path: Path, manifest: dict[str, Any]
) -> tuple[dict[str, TensorRecord], dict[str, str]]:
    expected = _expected_scale_names()
    records, hashes = _validate_subset_manifest(
        path=path,
        manifest=manifest,
        repo=SCALE_REPO,
        revision=SCALE_REVISION,
        expected_names=expected,
    )
    if any(
        record.dtype != "F32" or record.shape not in {(), (1,)}
        for record in records.values()
    ):
        raise BuildError(
            "ModelOpt expert scale subset contains a non-scalar F32 tensor"
        )
    return records, hashes


def _validate_adapter(
    path: Path, manifest: dict[str, Any]
) -> tuple[int, float, float, dict[str, TensorRecord]]:
    if (
        manifest.get("artifact") != "qwen38-flash-next-lm-head-lora"
        or manifest.get("base") != {"repo": BF16_REPO, "revision": BF16_REVISION}
        or manifest.get("target_modules") != ["lm_head"]
        or manifest.get("training_precision") != "bfloat16"
        or manifest.get("merge_order") != "before_nvfp4"
        or manifest.get("lora_dropout") != 0.0
        or manifest.get("gate_status")
        != "training-integrity-passed-semantic-qualification-pending"
        or manifest.get("output_file") != path.name
        or manifest.get("output_sha256") != _sha256(path)
    ):
        raise BuildError("behavior adapter provenance, scope, or gate changed")
    rank = manifest.get("rank")
    alpha = manifest.get("alpha")
    bound = manifest.get("max_relative_frobenius_norm")
    if (
        type(rank) is not int
        or not 1 <= rank <= 64
        or not isinstance(alpha, (int, float))
        or isinstance(alpha, bool)
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0
        or not isinstance(bound, (int, float))
        or isinstance(bound, bool)
        or not math.isfinite(float(bound))
        or not 0 < float(bound) <= 0.05
    ):
        raise BuildError("behavior adapter rank, alpha, or movement bound changed")
    hashes = manifest.get("tensor_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != {LORA_A, LORA_B}
        or not all(
            isinstance(value, str) and _SHA256_RE.fullmatch(value)
            for value in hashes.values()
        )
    ):
        raise BuildError("behavior adapter tensor receipt changed")
    _metadata, records = _read_safetensors_header(path)
    if set(records) != {LORA_A, LORA_B}:
        raise BuildError("behavior adapter contains tensors outside lm_head LoRA")
    if records[LORA_A].dtype != "BF16" or records[LORA_A].shape != (rank, HIDDEN_SIZE):
        raise BuildError("behavior adapter LoRA A topology changed")
    if records[LORA_B].dtype != "BF16" or records[LORA_B].shape != (VOCAB_SIZE, rank):
        raise BuildError("behavior adapter LoRA B topology changed")
    for name in (LORA_A, LORA_B):
        if _tensor_sha256(TensorLocation(path, records[name])) != hashes[name]:
            raise BuildError(f"behavior adapter tensor SHA-256 changed: {name}")
    baseline_receipt = manifest.get("official_untuned_baseline")
    expected_baseline_keys = {
        "file",
        "sha256",
        "schema_version",
        "judgment_schema_version",
        "eval_sha256",
        "producer_script_sha256",
        "source_manifest_sha256",
    }
    if (
        not isinstance(baseline_receipt, dict)
        or set(baseline_receipt) != expected_baseline_keys
        or baseline_receipt.get("file") != OFFICIAL_BASELINE_FILENAME
        or baseline_receipt.get("schema_version") != OFFICIAL_BASELINE_SCHEMA
        or baseline_receipt.get("judgment_schema_version") != BEHAVIOR_JUDGMENT_SCHEMA
        or any(
            not isinstance(baseline_receipt.get(field), str)
            or _SHA256_RE.fullmatch(str(baseline_receipt.get(field))) is None
            for field in (
                "sha256",
                "eval_sha256",
                "producer_script_sha256",
                "source_manifest_sha256",
            )
        )
    ):
        raise BuildError("official untuned behavior baseline receipt changed")
    baseline_path = path.parent / OFFICIAL_BASELINE_FILENAME
    if baseline_path.parent != path.parent:
        raise BuildError("official untuned behavior baseline path escaped adapter root")
    baseline = _load_manifest(
        baseline_path,
        str(baseline_receipt["sha256"]),
        OFFICIAL_BASELINE_SCHEMA,
    )
    try:
        from aeon.scripts import train_qwen38_flash_next_behavior as trainer
    except ImportError as exc:
        raise BuildError("behavior baseline validator is unavailable") from exc
    try:
        trainer.validate_official_baseline_spec(
            baseline, expected_eval_sha256=str(baseline_receipt["eval_sha256"])
        )
    except trainer.BehaviorTrainingError as exc:
        raise BuildError(
            "official untuned behavior baseline validation failed"
        ) from exc
    if (
        trainer.OFFICIAL_BASELINE_SCHEMA != OFFICIAL_BASELINE_SCHEMA
        or trainer.BEHAVIOR_JUDGMENT_SCHEMA != BEHAVIOR_JUDGMENT_SCHEMA
        or baseline["producer"]["script_sha256"]
        != baseline_receipt["producer_script_sha256"]
        or baseline["source"]["external_source_manifest_sha256"]
        != baseline_receipt["source_manifest_sha256"]
    ):
        raise BuildError("official untuned behavior baseline binding changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or files.get(OFFICIAL_BASELINE_FILENAME) != {
        "sha256": baseline_receipt["sha256"],
        "size": baseline_path.stat().st_size,
    }:
        raise BuildError("adapter manifest does not bind the baseline file")
    return rank, float(alpha), float(bound), records


def _merge_lm_head_chunkwise(
    base: Any,
    lora_a: Any,
    lora_b: Any,
    *,
    alpha: float,
    maximum_relative_norm: float,
    chunk_rows: int,
) -> tuple[Any, float]:
    """Merge ``B @ A`` without materializing the full FP32 delta matrix."""
    import torch

    if (
        base.dtype != torch.bfloat16
        or lora_a.dtype != torch.bfloat16
        or lora_b.dtype != torch.bfloat16
        or base.shape != (VOCAB_SIZE, HIDDEN_SIZE)
        or lora_a.ndim != 2
        or lora_b.ndim != 2
        or lora_a.shape[1] != HIDDEN_SIZE
        or lora_b.shape != (VOCAB_SIZE, lora_a.shape[0])
        or not 1 <= chunk_rows <= 16_384
        or not 0 < maximum_relative_norm <= 0.05
    ):
        raise BuildError("lm_head LoRA merge inputs are outside the reviewed bounds")
    rank = lora_a.shape[0]
    scale = float(alpha) / rank
    a_float = lora_a.float()
    output = torch.empty_like(base)
    delta_norm_sq = 0.0
    base_norm_sq = 0.0
    for start in range(0, VOCAB_SIZE, chunk_rows):
        end = min(start + chunk_rows, VOCAB_SIZE)
        base_chunk = base[start:end].float()
        delta = torch.matmul(lora_b[start:end].float(), a_float).mul_(scale)
        if not torch.isfinite(delta).all():
            raise BuildError("lm_head LoRA produced a non-finite delta")
        delta_norm_sq += float(delta.square().sum(dtype=torch.float64))
        base_norm_sq += float(base_chunk.square().sum(dtype=torch.float64))
        output[start:end].copy_((base_chunk + delta).to(torch.bfloat16))
        del base_chunk, delta
    if not base_norm_sq > 0:
        raise BuildError("lm_head base matrix has zero or invalid norm")
    relative = math.sqrt(delta_norm_sq / base_norm_sq)
    if not math.isfinite(relative) or relative > maximum_relative_norm + 1e-12:
        raise BuildError(
            f"lm_head LoRA movement {relative:.8f} exceeds {maximum_relative_norm:.8f}"
        )
    return output, relative


def _fleet_binding(torch_module: Any) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    claim = os.environ.get("GPU_AGENT_CLAIM_ID", "")
    runtime = os.environ.get("AEON_QUANT_RUNTIME_ID", "")
    try:
        limit_gb = float(os.environ["GPU_MEM_LIMIT_GB"])
        reserve_gb = float(os.environ.get("GPU_RESERVE_GB", "6"))
    except (KeyError, ValueError) as exc:
        raise BuildError("Fleet GPU memory cap or reserve is absent/malformed") from exc
    if (
        _CUDA_UUID_RE.fullmatch(visible) is None
        or _CLAIM_RE.fullmatch(claim) is None
        or _RUNTIME_RE.fullmatch(runtime) is None
        or not 64.0 <= limit_gb <= 90.0
        or reserve_gb < 6.0
        or not torch_module.cuda.is_available()
        or torch_module.cuda.device_count() != 1
    ):
        raise BuildError("reviewed one-GPU Fleet binding is absent")
    properties = torch_module.cuda.get_device_properties(0)
    total_gb = properties.total_memory / 1024**3
    capability = torch_module.cuda.get_device_capability(0)
    if (
        not 88.0 <= total_gb <= 100.0
        or limit_gb + reserve_gb > total_gb + 0.05
        or "RTX PRO 6000" not in properties.name.upper()
        or tuple(capability) < (12, 0)
    ):
        raise BuildError(
            "leased GPU is not one 96GB RTX PRO 6000 Blackwell with reserve"
        )
    torch_module.cuda.set_per_process_memory_fraction(limit_gb / total_gb, 0)
    return {
        "claim_id_sha256": hashlib.sha256(claim.encode("utf-8")).hexdigest(),
        "gpu_uuid_sha256": hashlib.sha256(visible.encode("utf-8")).hexdigest(),
        "runtime_id": runtime,
        "gpu_name": properties.name,
        "gpu_total_gb": total_gb,
        "gpu_mem_limit_gb": limit_gb,
        "gpu_reserve_gb": reserve_gb,
        "compute_capability": list(capability),
    }


def _load_modelopt_backend(
    wheel: Path,
) -> tuple[Callable[[Any, Any], tuple[Any, Any, Any]], str]:
    if _sha256(wheel) != MODELOPT_WHEEL_SHA256:
        raise BuildError("ModelOpt 0.46 wheel identity changed")
    try:
        version = importlib.metadata.version("nvidia-modelopt")
    except importlib.metadata.PackageNotFoundError as exc:
        raise BuildError("nvidia-modelopt is not installed") from exc
    if version != MODELOPT_VERSION:
        raise BuildError(
            f"expected nvidia-modelopt {MODELOPT_VERSION}, found {version}"
        )
    try:
        import torch
        from modelopt.torch.quantization.qtensor import NVFP4QTensor
    except ImportError as exc:
        raise BuildError("ModelOpt NVFP4QTensor could not be imported") from exc

    def backend(source: Any, scale_2: Any) -> tuple[Any, Any, Any]:
        cuda_source = source.cuda(non_blocking=False)
        cuda_scale = scale_2.cuda(non_blocking=False)
        quantized, block_scale, returned_scale = NVFP4QTensor.quantize(
            cuda_source,
            block_size=16,
            weights_scaling_factor_2=cuda_scale,
        )
        packed = quantized._quantized_data.cpu().contiguous()
        blocks = block_scale.cpu().contiguous()
        global_scale = returned_scale.cpu().contiguous()
        del cuda_source, cuda_scale, quantized, block_scale, returned_scale
        torch.cuda.empty_cache()
        return packed, blocks, global_scale

    return backend, version


def _positive_scalar(tensor: Any, name: str) -> float:
    import torch

    if tensor.dtype != torch.float32 or tensor.numel() != 1:
        raise BuildError(f"ModelOpt reference scale is not one F32 scalar: {name}")
    value = float(tensor.reshape(()))
    if not math.isfinite(value) or value <= 0:
        raise BuildError(f"ModelOpt reference scale is not finite and positive: {name}")
    return value


def _load_layer_scales(handle: Any, layer: int) -> LayerScales:
    import torch

    gate_input = torch.empty(NUM_EXPERTS, dtype=torch.float32)
    gate_scale = torch.empty(NUM_EXPERTS, dtype=torch.float32)
    down_input = torch.empty(NUM_EXPERTS, dtype=torch.float32)
    down_scale = torch.empty(NUM_EXPERTS, dtype=torch.float32)
    for expert in range(NUM_EXPERTS):
        gate = _expert_module(layer, expert, "gate_proj")
        up = _expert_module(layer, expert, "up_proj")
        down = _expert_module(layer, expert, "down_proj")
        gate_in_tensor = handle.get_tensor(gate + ".input_scale")
        up_in_tensor = handle.get_tensor(up + ".input_scale")
        gate_scale_tensor = handle.get_tensor(gate + ".weight_scale_2")
        up_scale_tensor = handle.get_tensor(up + ".weight_scale_2")
        gate_in = _positive_scalar(gate_in_tensor, gate + ".input_scale")
        up_in = _positive_scalar(up_in_tensor, up + ".input_scale")
        gate_s = _positive_scalar(gate_scale_tensor, gate + ".weight_scale_2")
        up_s = _positive_scalar(up_scale_tensor, up + ".weight_scale_2")
        if gate_in != up_in or gate_s != up_s:
            raise BuildError(
                f"reference gate/up fused scales diverged at layer {layer}, expert {expert}"
            )
        gate_input[expert] = gate_in
        gate_scale[expert] = gate_s
        down_input[expert] = _positive_scalar(
            handle.get_tensor(down + ".input_scale"), down + ".input_scale"
        )
        down_scale[expert] = _positive_scalar(
            handle.get_tensor(down + ".weight_scale_2"), down + ".weight_scale_2"
        )
    return LayerScales(gate_input, gate_scale, down_input, down_scale)


def _validate_quantized_result(
    source: Any, packed: Any, blocks: Any, scale: Any, label: str
) -> None:
    import torch

    expected_packed = (*source.shape[:-1], source.shape[-1] // 2)
    expected_blocks = (*source.shape[:-1], source.shape[-1] // 16)
    if (
        source.shape[-1] % 16
        or packed.dtype != torch.uint8
        or tuple(packed.shape) != expected_packed
        or str(blocks.dtype) != "torch.float8_e4m3fn"
        or tuple(blocks.shape) != expected_blocks
        or scale.dtype != torch.float32
        or scale.numel() not in {NUM_EXPERTS, 1}
        or not torch.isfinite(scale.float()).all()
        or not (scale.float() > 0).all()
    ):
        raise BuildError(f"ModelOpt returned an invalid NVFP4 tensor set: {label}")


def _quantize_expert_layer(
    gate_up: Any,
    down: Any,
    scales: LayerScales,
    layer: int,
    backend: Callable[[Any, Any], tuple[Any, Any, Any]],
) -> dict[str, Any]:
    import torch

    if (
        not 0 <= layer < NUM_LAYERS
        or gate_up.dtype != torch.bfloat16
        or tuple(gate_up.shape)
        != (NUM_EXPERTS, 2 * EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE)
        or down.dtype != torch.bfloat16
        or tuple(down.shape) != (NUM_EXPERTS, HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE)
        or tuple(scales.gate_up_input.shape) != (NUM_EXPERTS,)
        or tuple(scales.gate_up_weight_scale_2.shape) != (NUM_EXPERTS,)
        or tuple(scales.down_input.shape) != (NUM_EXPERTS,)
        or tuple(scales.down_weight_scale_2.shape) != (NUM_EXPERTS,)
        or any(
            tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
            or not (tensor > 0).all()
            for tensor in (
                scales.gate_up_input,
                scales.gate_up_weight_scale_2,
                scales.down_input,
                scales.down_weight_scale_2,
            )
        )
    ):
        raise BuildError("expert layer or reference scale topology changed")
    gate_packed, gate_blocks, gate_scale = backend(
        gate_up, scales.gate_up_weight_scale_2.reshape(NUM_EXPERTS, 1, 1)
    )
    down_packed, down_blocks, returned_down_scale = backend(
        down, scales.down_weight_scale_2.reshape(NUM_EXPERTS, 1, 1)
    )
    _validate_quantized_result(gate_up, gate_packed, gate_blocks, gate_scale, "gate_up")
    _validate_quantized_result(
        down, down_packed, down_blocks, returned_down_scale, "down"
    )
    gate_scale = gate_scale.reshape(-1)
    returned_down_scale = returned_down_scale.reshape(-1)
    if gate_scale.numel() == 1:
        gate_scale = gate_scale.expand(NUM_EXPERTS)
    if returned_down_scale.numel() == 1:
        returned_down_scale = returned_down_scale.expand(NUM_EXPERTS)
    if not torch.equal(gate_scale, scales.gate_up_weight_scale_2) or not torch.equal(
        returned_down_scale, scales.down_weight_scale_2
    ):
        raise BuildError("ModelOpt did not preserve the pinned expert global scales")
    result: dict[str, Any] = {}
    split = EXPERT_INTERMEDIATE_SIZE
    for expert in range(NUM_EXPERTS):
        for projection, packed, blocks in (
            ("gate_proj", gate_packed[expert, :split], gate_blocks[expert, :split]),
            ("up_proj", gate_packed[expert, split:], gate_blocks[expert, split:]),
        ):
            module = _expert_module(layer, expert, projection)
            result[module + ".weight"] = packed.contiguous()
            result[module + ".weight_scale"] = blocks.contiguous()
            result[module + ".weight_scale_2"] = gate_scale[expert].clone().reshape(())
            result[module + ".input_scale"] = (
                scales.gate_up_input[expert].clone().reshape(())
            )
        module = _expert_module(layer, expert, "down_proj")
        result[module + ".weight"] = down_packed[expert].contiguous()
        result[module + ".weight_scale"] = down_blocks[expert].contiguous()
        result[module + ".weight_scale_2"] = (
            returned_down_scale[expert].clone().reshape(())
        )
        result[module + ".input_scale"] = scales.down_input[expert].clone().reshape(())
    if set(result) != {
        name
        for name in _expected_quantized_names()
        if name.startswith(f"model.language_model.layers.{layer}.")
    }:
        raise BuildError(
            f"quantized routed-expert output did not close at layer {layer}"
        )
    return result


def _modelopt_quant_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ModelOpt 0.46's unified NVFP4 and converted HF forms.

    This is the compact unified-export configuration used by the pinned
    Qwen4Exp ModelOpt reference: target ``Linear`` and close every non-routed
    linear family with the exact wildcard ignore list.  The tensor inventory
    independently proves that only the routed experts have packed components.
    """
    ignore = list(MODELOPT_IGNORE)
    unified = {
        "producer": {"name": "modelopt", "version": MODELOPT_VERSION},
        "quantization": {
            "exclude_modules": ignore,
            "group_size": 16,
            "quant_algo": "NVFP4",
        },
    }
    group = {
        "input_activations": {
            "dynamic": False,
            "group_size": 16,
            "num_bits": 4,
            "type": "float",
        },
        "weights": {
            "dynamic": False,
            "group_size": 16,
            "num_bits": 4,
            "type": "float",
        },
        "targets": ["Linear"],
    }
    hf = {
        "config_groups": {"group_0": group},
        "ignore": ignore,
        "quant_algo": "NVFP4",
        "quant_method": "modelopt",
        "producer": {"name": "modelopt", "version": MODELOPT_VERSION},
    }
    return unified, hf


def _final_config(source: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # JSON round-trip provides a deep copy while refusing non-JSON values.
    config = json.loads(_canonical_json(source))
    text = config["text_config"]
    text["ple_embedding_dtype"] = "float8_e4m3fn"
    unified, hf = _modelopt_quant_configs()
    config["quantization_config"] = hf
    config["transformers_version"] = "5.16.0"
    return config, unified


def _validate_final_config(config: Mapping[str, Any]) -> None:
    expected_unified, expected_hf = _modelopt_quant_configs()
    del expected_unified
    text = config.get("text_config")
    if (
        not isinstance(text, dict)
        or text.get("ple_embedding_dtype") != "float8_e4m3fn"
        or text.get("split_ngram_parts") != PLE_TABLE_COUNT
        or text.get("mtp_num_hidden_layers") != 1
        or config.get("quantization_config") != expected_hf
    ):
        raise BuildError(
            "final config is not the closed ModelOpt/PLE/MTP configuration"
        )


def _load_tensor(location: TensorLocation, name: str) -> Any:
    from safetensors import safe_open

    with safe_open(location.path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _copy_metadata(
    root: Path,
    partial: Path,
    receipts: Mapping[str, tuple[str, int]],
) -> list[str]:
    reserved = {
        "config.json",
        "model.safetensors.index.json",
        "BUILD_MANIFEST.json",
        "VALIDATION_REPORT.json",
        "hf_quant_config.json",
        HYBRID_MANIFEST_FILENAME,
        "SHA256SUMS",
    }
    copied: list[str] = []
    for name in sorted(receipts):
        if name.endswith(".safetensors") or name in reserved:
            continue
        destination = partial / name
        shutil.copy2(root / name, destination)
        destination.chmod(0o600)
        copied.append(name)
    return copied


def _copy_hybrid_manifest(
    source: Path,
    partial: Path,
    expected_sha256: str,
) -> str:
    """Copy the externally receipted hybrid manifest into the checkpoint.

    The manifest cannot receipt itself in its closed ``files`` map, so it is
    deliberately copied outside ``_copy_metadata`` and then included in the
    final SHA256SUMS closure.  Its fixed destination name is part of the release
    contract; the input path's basename is not trusted as checkpoint metadata.
    """

    expected = _expected_digest(expected_sha256, "hybrid manifest digest")
    source_metadata = _regular_file(source, maximum=16 * 1024 * 1024)
    if _sha256(source) != expected:
        raise BuildError("hybrid manifest identity changed before copy")
    destination = partial / HYBRID_MANIFEST_FILENAME
    if destination.exists() or destination.is_symlink():
        raise BuildError("hybrid manifest destination already exists")
    shutil.copy2(source, destination)
    destination.chmod(0o600)
    destination_metadata = _regular_file(destination, maximum=16 * 1024 * 1024)
    if (
        destination_metadata.st_size != source_metadata.st_size
        or _sha256(destination) != expected
    ):
        raise BuildError("copied hybrid manifest identity changed")
    return HYBRID_MANIFEST_FILENAME


def _write_passthrough_shards(
    *,
    root: Path,
    partial: Path,
    untuned_partial: Path,
    names_by_shard: Mapping[str, tuple[str, ...]],
    metadata_by_shard: Mapping[str, Mapping[str, str]],
    locations: Mapping[str, TensorLocation],
    adapter_path: Path,
    adapter_rank: int,
    adapter_alpha: float,
    adapter_bound: float,
    chunk_rows: int,
) -> tuple[dict[str, str], float, list[str], list[str], dict[str, Any]]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        lora_a = handle.get_tensor(LORA_A)
        lora_b = handle.get_tensor(LORA_B)
    if lora_a.shape[0] != adapter_rank:
        raise BuildError("adapter rank changed between preflight and merge")
    head_location = locations.get(LM_HEAD)
    if head_location is None:
        raise BuildError("hybrid source lost lm_head before isolated merge")
    with safe_open(head_location.path, framework="pt", device="cpu") as handle:
        official_head = handle.get_tensor(LM_HEAD)
    tuned_head, relative_norm = _merge_lm_head_chunkwise(
        official_head,
        lora_a,
        lora_b,
        alpha=adapter_alpha,
        maximum_relative_norm=adapter_bound,
        chunk_rows=chunk_rows,
    )
    tuned_head_path = partial / TUNED_LM_HEAD_FILENAME
    untuned_head_path = untuned_partial / UNTUNED_LM_HEAD_FILENAME
    save_file({LM_HEAD: tuned_head}, tuned_head_path, metadata={"format": "pt"})
    save_file({LM_HEAD: official_head}, untuned_head_path, metadata={"format": "pt"})
    tuned_head_path.chmod(0o600)
    untuned_head_path.chmod(0o600)
    del tuned_head, official_head
    gc.collect()
    tuned_header = _read_safetensors_header(tuned_head_path)[1]
    untuned_header = _read_safetensors_header(untuned_head_path)[1]
    if set(tuned_header) != {LM_HEAD} or set(untuned_header) != {LM_HEAD}:
        raise BuildError("isolated lm_head shard inventory changed")
    head_receipt = {
        "tensor_name": LM_HEAD,
        "dtype": "BF16",
        "shape": [VOCAB_SIZE, HIDDEN_SIZE],
        "tuned_file": TUNED_LM_HEAD_FILENAME,
        "tuned_file_sha256": _sha256(tuned_head_path),
        "tuned_tensor_sha256": _tensor_sha256(
            TensorLocation(tuned_head_path, tuned_header[LM_HEAD])
        ),
        "official_untuned_file": UNTUNED_LM_HEAD_FILENAME,
        "official_untuned_file_sha256": _sha256(untuned_head_path),
        "official_untuned_tensor_sha256": _tensor_sha256(
            TensorLocation(untuned_head_path, untuned_header[LM_HEAD])
        ),
    }
    output_map: dict[str, str] = {LM_HEAD: TUNED_LM_HEAD_FILENAME}
    copied: list[str] = []
    rewritten: list[str] = []
    for shard, names in sorted(names_by_shard.items()):
        retained = [
            name
            for name in names
            if _expert_source_layer(name) is None
            and name not in MTP_NAMES
            and name != LM_HEAD
        ]
        if not retained:
            continue
        changed = len(retained) != len(names)
        destination = partial / shard
        if not changed:
            shutil.copy2(root / shard, destination)
            destination.chmod(0o600)
            copied.append(shard)
        else:
            tensors: dict[str, Any] = {}
            with safe_open(root / shard, framework="pt", device="cpu") as handle:
                for name in retained:
                    if locations[name].path != root / shard:
                        raise BuildError(f"hybrid tensor moved across shards: {name}")
                    tensor = handle.get_tensor(name)
                    tensors[name] = tensor
            save_file(
                tensors,
                destination,
                metadata=dict(metadata_by_shard[shard]) or {"format": "pt"},
            )
            destination.chmod(0o600)
            rewritten.append(shard)
            del tensors
            gc.collect()
        for name in retained:
            output_map[name] = shard
    return output_map, relative_norm, copied, rewritten, head_receipt


def _write_quantized_experts(
    *,
    partial: Path,
    locations: Mapping[str, TensorLocation],
    scale_path: Path,
    backend: Callable[[Any, Any], tuple[Any, Any, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    output_map: dict[str, str] = {}
    receipts: list[dict[str, Any]] = []
    with safe_open(scale_path, framework="pt", device="cpu") as scale_handle:
        for layer in range(NUM_LAYERS):
            gate_name = _source_expert_name(layer, "gate_up_proj")
            down_name = _source_expert_name(layer, "down_proj")
            gate_up = _load_tensor(locations[gate_name], gate_name)
            down = _load_tensor(locations[down_name], down_name)
            scales = _load_layer_scales(scale_handle, layer)
            tensors = _quantize_expert_layer(gate_up, down, scales, layer, backend)
            filename = (
                f"model-routed-experts-{layer + 1:05d}-of-{NUM_LAYERS:05d}.safetensors"
            )
            destination = partial / filename
            if destination.exists() or destination.is_symlink():
                raise BuildError(f"quantized expert filename collision: {filename}")
            save_file(
                tensors,
                destination,
                metadata={
                    "format": "pt",
                    "producer": "modelopt",
                    "modelopt_version": MODELOPT_VERSION,
                    "quant_algo": "NVFP4",
                    "block_size": "16",
                },
            )
            destination.chmod(0o600)
            for name in tensors:
                if name in output_map:
                    raise BuildError(f"duplicate quantized tensor: {name}")
                output_map[name] = filename
            receipts.append(
                {
                    "layer": layer,
                    "file": filename,
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                    "tensor_count": len(tensors),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "routed_expert_layer_quantized",
                        "layer": layer,
                        "file": filename,
                        "bytes": destination.stat().st_size,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del gate_up, down, scales, tensors
            gc.collect()
            torch.cuda.empty_cache()
    if set(output_map) != set(_expected_quantized_names()):
        raise BuildError("quantized expert file set does not close")
    return output_map, receipts


def _graft_mtp(source: Path, partial: Path) -> tuple[str, dict[str, str]]:
    filename = "model-mtp-bf16.safetensors"
    destination = partial / filename
    if destination.exists() or destination.is_symlink():
        raise BuildError("dedicated MTP filename collides with a retained hybrid shard")
    shutil.copy2(source, destination)
    destination.chmod(0o600)
    if _sha256(destination) != _sha256(source):
        raise BuildError("official BF16 MTP copy changed")
    return filename, {name: filename for name in MTP_NAMES}


def _actual_output_locations(
    root: Path,
) -> tuple[dict[str, TensorLocation], dict[str, dict[str, TensorRecord]]]:
    locations: dict[str, TensorLocation] = {}
    by_file: dict[str, dict[str, TensorRecord]] = {}
    for path in sorted(root.glob("*.safetensors")):
        _metadata, records = _read_safetensors_header(path)
        by_file[path.name] = records
        for name, record in records.items():
            if name in locations:
                raise BuildError(f"duplicate output tensor: {name}")
            locations[name] = TensorLocation(path, record)
    return locations, by_file


def _validate_output_scales(
    root: Path, by_file: Mapping[str, Mapping[str, TensorRecord]]
) -> None:
    import torch
    from safetensors import safe_open

    for filename in sorted(
        name for name in by_file if name.startswith("model-routed-experts-")
    ):
        records = by_file[filename]
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            for name, record in records.items():
                if name.endswith((".input_scale", ".weight_scale_2")):
                    tensor = handle.get_tensor(name)
                    if (
                        record.dtype != "F32"
                        or record.shape != ()
                        or tensor.dtype != torch.float32
                        or not torch.isfinite(tensor)
                        or float(tensor) <= 0
                    ):
                        raise BuildError(
                            f"output scale is not finite positive F32: {name}"
                        )


def _validate_output(
    *,
    partial: Path,
    output_map: Mapping[str, str],
    hybrid_locations: Mapping[str, TensorLocation],
    mtp_source: Path,
) -> dict[str, Any]:
    locations, by_file = _actual_output_locations(partial)
    expected_names = (
        {
            name
            for name in hybrid_locations
            if _expert_source_layer(name) is None and name not in MTP_NAMES
        }
        | set(MTP_NAMES)
        | set(_expected_quantized_names())
    )
    if (
        set(locations) != expected_names
        or set(output_map) != expected_names
        or len(locations) != EXPECTED_OUTPUT_TENSOR_COUNT
    ):
        raise BuildError(
            "actual output tensor set does not close against the expected topology"
        )
    for name, location in locations.items():
        if output_map.get(name) != location.path.name:
            raise BuildError(f"output index points at the wrong shard: {name}")
    for name in _expected_quantized_names():
        match = _OUTPUT_EXPERT_RE.fullmatch(name)
        if match is None:
            raise BuildError(f"invalid closed quantized tensor name: {name}")
        projection = match.group(3)
        component = match.group(4)
        record = locations[name].record
        rows, columns = (
            (EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE)
            if projection in {"gate_proj", "up_proj"}
            else (HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE)
        )
        expected = {
            "weight": ("U8", (rows, columns // 2)),
            "weight_scale": ("F8_E4M3", (rows, columns // 16)),
            "weight_scale_2": ("F32", ()),
            "input_scale": ("F32", ()),
        }[component]
        if (record.dtype, record.shape) != expected:
            raise BuildError(f"output NVFP4 component topology changed: {name}")
    _validate_output_scales(partial, by_file)

    preserved_names = sorted(
        {name for name in hybrid_locations if name.startswith(VISION_PREFIX)}
        | set(_ple_table_names())
        | {PLE_SCALE}
    )
    preserved_hashes: dict[str, str] = {}
    for name in preserved_names:
        source_hash = _tensor_sha256(hybrid_locations[name])
        output_hash = _tensor_sha256(locations[name])
        if source_hash != output_hash:
            raise BuildError(f"vision/PLE tensor was not preserved exactly: {name}")
        preserved_hashes[name] = source_hash
    mtp_path = partial / "model-mtp-bf16.safetensors"
    if _sha256(mtp_path) != _sha256(mtp_source):
        raise BuildError("final MTP shard is not the exact official BF16 subset")
    if any(locations[name].record.dtype != "BF16" for name in MTP_NAMES):
        raise BuildError("final MTP contains a non-BF16 tensor")
    return {
        "source_hybrid_tensor_count": len(hybrid_locations),
        "output_tensor_count": len(locations),
        "quantized_module_count": QUANTIZED_MODULE_COUNT,
        "quantized_component_count": QUANTIZED_COMPONENT_COUNT,
        "source_expert_tensor_count_removed": SOURCE_EXPERT_TENSOR_COUNT,
        "mtp_tensor_count": len(MTP_NAMES),
        "vision_tensor_count": VISION_TENSOR_COUNT,
        "ple_table_tensor_count": PLE_TABLE_COUNT,
        "vision_ple_exact": True,
        "mtp_exact": True,
        "preserved_tensor_sha256_digest": hashlib.sha256(
            _canonical_json(preserved_hashes)
        ).hexdigest(),
    }


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_sha256sums(root: Path) -> dict[str, str]:
    files = sorted(
        path for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    sums = {path.name: _sha256(path) for path in files}
    payload = "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    path = root / "SHA256SUMS"
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        view = memoryview(payload.encode("utf-8"))
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise BuildError("SHA256SUMS write was incomplete")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sums


def _tree_sha256(root: Path) -> str:
    return _sha256(root / "SHA256SUMS")


def _build_untuned_sibling(
    *,
    tuned_partial: Path,
    untuned_partial: Path,
    tuned_index: Mapping[str, Any],
    output_map: Mapping[str, str],
    tuned_sums: Mapping[str, str],
    head_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    rewritten_allowlist = [
        "BUILD_MANIFEST.json",
        "SHA256SUMS",
        "VALIDATION_REPORT.json",
        "model.safetensors.index.json",
        TUNED_LM_HEAD_FILENAME,
        UNTUNED_LM_HEAD_FILENAME,
    ]
    shared_paths: list[str] = []
    for source in sorted(path for path in tuned_partial.iterdir() if path.is_file()):
        if source.name in {
            "BUILD_MANIFEST.json",
            "SHA256SUMS",
            "VALIDATION_REPORT.json",
            "model.safetensors.index.json",
            TUNED_LM_HEAD_FILENAME,
        }:
            continue
        destination = untuned_partial / source.name
        if destination.exists() or destination.is_symlink():
            raise BuildError("untuned sibling common-file collision")
        os.link(source, destination, follow_symlinks=False)
        source_stat = source.lstat()
        destination_stat = destination.lstat()
        if (
            source_stat.st_dev != destination_stat.st_dev
            or source_stat.st_ino != destination_stat.st_ino
            or not stat.S_ISREG(destination_stat.st_mode)
        ):
            raise BuildError("untuned sibling did not preserve a common-file hardlink")
        shared_paths.append(source.name)
    tuned_validation = _read_json(tuned_partial / "VALIDATION_REPORT.json")
    untuned_validation = json.loads(_canonical_json(tuned_validation))
    untuned_validation.update(
        {
            "lm_head_lora_merged_before_quantization": False,
            "lm_head_lora_relative_frobenius_norm": 0.0,
            "runtime_validation_status": (
                "official-untuned-baseline-unvalidated-canary"
            ),
        }
    )
    _write_json(untuned_partial / "VALIDATION_REPORT.json", untuned_validation)
    tuned_build = _read_json(tuned_partial / "BUILD_MANIFEST.json")
    untuned_build = json.loads(_canonical_json(tuned_build))
    untuned_build["status"] = "official-untuned-baseline-unvalidated-canary"
    behavior_source = untuned_build["sources"]["behavior_adapter"]
    behavior_source["target_modules"] = []
    behavior_source["gate_status"] = "not-applied-official-untuned-baseline"
    preserved = untuned_build["quantization"]["preserved"]
    preserved[:] = [
        "official BF16 lm_head without adapter"
        if item == "BF16 lm_head after bounded LoRA merge"
        else item
        for item in preserved
    ]
    untuned_build["build"]["checkpoint_role"] = "official_untuned"
    _write_json(untuned_partial / "BUILD_MANIFEST.json", untuned_build)
    untuned_map = dict(output_map)
    untuned_map[LM_HEAD] = UNTUNED_LM_HEAD_FILENAME
    untuned_index = json.loads(_canonical_json(tuned_index))
    untuned_index["weight_map"] = untuned_map
    _write_json(untuned_partial / "model.safetensors.index.json", untuned_index)
    untuned_sums = _write_sha256sums(untuned_partial)
    expected_untuned_files = {
        path.name
        for path in untuned_partial.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(untuned_sums) != expected_untuned_files:
        raise BuildError("official untuned sibling SHA256SUMS closure failed")
    non_head_inventory = {
        name: {
            "shard": shard,
            "shard_sha256": tuned_sums[shard],
        }
        for name, shard in sorted(output_map.items())
        if name != LM_HEAD
    }
    non_head_digest = hashlib.sha256(_canonical_json(non_head_inventory)).hexdigest()
    unique_shared: dict[tuple[int, int], int] = {}
    for name in shared_paths:
        metadata = (tuned_partial / name).lstat()
        unique_shared.setdefault((metadata.st_dev, metadata.st_ino), metadata.st_size)
    hardlink_identity = {
        "shared_regular_file_count": len(shared_paths),
        "shared_unique_bytes": sum(unique_shared.values()),
        "shared_paths_sha256": hashlib.sha256(
            _canonical_json(sorted(shared_paths))
        ).hexdigest(),
        "same_device_and_inode": True,
        "rewritten_allowlist": rewritten_allowlist,
    }
    manifest = {
        "schema_version": SIBLING_SCHEMA,
        "complete": True,
        "tuned_checkpoint_tree_sha256": _tree_sha256(tuned_partial),
        "official_untuned_checkpoint_tree_sha256": _tree_sha256(untuned_partial),
        "tuned_lm_head_tensor_sha256": head_receipt["tuned_tensor_sha256"],
        "official_untuned_lm_head_tensor_sha256": head_receipt[
            "official_untuned_tensor_sha256"
        ],
        "non_lm_head_tensor_inventory_sha256": non_head_digest,
        "non_lm_head_tensors_byte_identical": True,
        "hardlink_identity": hardlink_identity,
    }
    return manifest, untuned_sums


def _write_terminal_result(value: Mapping[str, Any]) -> None:
    raw = os.environ.get("AEON_QUANT_RESULT_PATH")
    if not raw:
        return
    path = Path(raw)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        payload = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
            + b"\n"
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise BuildError("terminal result write was incomplete")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _combine_output_maps(*maps: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in maps:
        overlap = set(result) & set(value)
        if overlap:
            raise BuildError(f"output tensor maps overlap: {sorted(overlap)[:3]}")
        result.update(value)
    return result


def _preflight_distinct_paths(paths: Iterable[Path]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise BuildError(
            "all input, manifest, wheel, and output paths must be distinct"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Hybrid manifest contract (exact top-level keys):\n"
            + json.dumps(HYBRID_MANIFEST_CONTRACT, indent=2, sort_keys=True)
        ),
    )
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--hybrid-manifest", type=Path, required=True)
    parser.add_argument("--hybrid-manifest-sha256", required=True)
    parser.add_argument("--mtp-subset", type=Path, required=True)
    parser.add_argument("--mtp-manifest", type=Path, required=True)
    parser.add_argument("--mtp-manifest-sha256", required=True)
    parser.add_argument("--expert-scales", type=Path, required=True)
    parser.add_argument("--expert-scales-manifest", type=Path, required=True)
    parser.add_argument("--expert-scales-manifest-sha256", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--adapter-manifest-sha256", required=True)
    parser.add_argument("--modelopt-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora-chunk-rows", type=int, default=4096)
    args = parser.parse_args(argv)

    hybrid = args.hybrid.resolve()
    hybrid_manifest_path = args.hybrid_manifest.resolve()
    mtp_subset = args.mtp_subset.resolve()
    mtp_manifest_path = args.mtp_manifest.resolve()
    expert_scales = args.expert_scales.resolve()
    scales_manifest_path = args.expert_scales_manifest.resolve()
    adapter = args.adapter.resolve()
    adapter_manifest_path = args.adapter_manifest.resolve()
    wheel = args.modelopt_wheel.resolve()
    output = args.output.resolve()
    untuned_output = output.parent / "official-untuned-model"
    sibling_manifest_path = output.parent / SIBLING_MANIFEST_FILENAME
    _preflight_distinct_paths(
        (
            hybrid,
            hybrid_manifest_path,
            mtp_subset,
            mtp_manifest_path,
            expert_scales,
            scales_manifest_path,
            adapter,
            adapter_manifest_path,
            wheel,
            output,
        )
    )
    if any(
        path.exists() or path.is_symlink()
        for path in (output, untuned_output, sibling_manifest_path)
    ):
        raise BuildError("final tuned/untuned sibling output already exists")
    if not 1 <= args.lora_chunk_rows <= 16_384:
        raise BuildError("lora-chunk-rows must be between 1 and 16384")
    _private_directory(hybrid)
    _private_directory(output.parent)
    _regular_file(wheel)

    # CPU-only, hash-complete preflight happens before the Fleet lease is used.
    hybrid_manifest = _load_manifest(
        hybrid_manifest_path, args.hybrid_manifest_sha256, HYBRID_SCHEMA
    )
    (
        source_config,
        source_index,
        _source_map,
        hybrid_locations,
        metadata_by_shard,
        names_by_shard,
        file_receipts,
    ) = _validate_hybrid(hybrid, hybrid_manifest)
    mtp_manifest = _load_manifest(
        mtp_manifest_path, args.mtp_manifest_sha256, SUBSET_SCHEMA
    )
    _mtp_records, mtp_tensor_hashes = _validate_mtp_subset(
        mtp_subset, mtp_manifest, hybrid_locations
    )
    scales_manifest = _load_manifest(
        scales_manifest_path, args.expert_scales_manifest_sha256, SUBSET_SCHEMA
    )
    _scale_records, scale_tensor_hashes = _validate_scale_subset(
        expert_scales, scales_manifest
    )
    adapter_manifest = _load_manifest(
        adapter_manifest_path, args.adapter_manifest_sha256, ADAPTER_SCHEMA
    )
    adapter_rank, adapter_alpha, adapter_bound, _adapter_records = _validate_adapter(
        adapter, adapter_manifest
    )
    baseline_receipt = dict(adapter_manifest["official_untuned_baseline"])
    baseline_source = adapter.parent / str(baseline_receipt["file"])
    if _sha256(wheel) != MODELOPT_WHEEL_SHA256:
        raise BuildError("pinned ModelOpt wheel identity changed")

    import torch

    fleet = _fleet_binding(torch)
    backend, modelopt_version = _load_modelopt_backend(wheel)
    started = time.time()
    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    untuned_partial = untuned_output.with_name(
        f".{untuned_output.name}.partial-{os.getpid()}"
    )
    if any(path.exists() or path.is_symlink() for path in (partial, untuned_partial)):
        raise BuildError("task-specific partial output path already exists")
    _private_directory(partial, create=True)
    _private_directory(untuned_partial, create=True)

    try:
        copied_metadata = _copy_metadata(hybrid, partial, file_receipts)
        copied_metadata.append(
            _copy_hybrid_manifest(
                hybrid_manifest_path,
                partial,
                args.hybrid_manifest_sha256,
            )
        )
        settled_baseline = partial / SETTLED_BASELINE_FILENAME
        shutil.copy2(baseline_source, settled_baseline)
        settled_baseline.chmod(0o600)
        if _sha256(settled_baseline) != baseline_receipt["sha256"]:
            raise BuildError("settled official behavior baseline changed during copy")
        copied_metadata.append(SETTLED_BASELINE_FILENAME)
        (
            passthrough_map,
            relative_norm,
            copied_shards,
            rewritten_shards,
            head_receipt,
        ) = _write_passthrough_shards(
            root=hybrid,
            partial=partial,
            untuned_partial=untuned_partial,
            names_by_shard=names_by_shard,
            metadata_by_shard=metadata_by_shard,
            locations=hybrid_locations,
            adapter_path=adapter,
            adapter_rank=adapter_rank,
            adapter_alpha=adapter_alpha,
            adapter_bound=adapter_bound,
            chunk_rows=args.lora_chunk_rows,
        )
        quantized_map, expert_receipts = _write_quantized_experts(
            partial=partial,
            locations=hybrid_locations,
            scale_path=expert_scales,
            backend=backend,
        )
        mtp_filename, mtp_map = _graft_mtp(mtp_subset, partial)
        output_map = _combine_output_maps(passthrough_map, quantized_map, mtp_map)
        if len(output_map) != EXPECTED_OUTPUT_TENSOR_COUNT:
            raise BuildError("final index tensor count changed")

        config, unified_quant_config = _final_config(source_config)
        _validate_final_config(config)
        _write_json(partial / "config.json", config)
        _write_json(partial / "hf_quant_config.json", unified_quant_config)
        index = json.loads(_canonical_json(source_index))
        index["weight_map"] = output_map
        index.setdefault("metadata", {})["total_size"] = sum(
            path.stat().st_size for path in partial.glob("*.safetensors")
        )
        _write_json(partial / "model.safetensors.index.json", index)

        # Close the owner-write TOCTOU window before validation/promotion.  The
        # second hybrid pass is intentionally expensive: a 300+ GB artifact is
        # never published from inputs that changed during a long shardwise run.
        _verify_file_receipts(hybrid, file_receipts)
        for path, expected in (
            (hybrid_manifest_path, args.hybrid_manifest_sha256),
            (mtp_manifest_path, args.mtp_manifest_sha256),
            (scales_manifest_path, args.expert_scales_manifest_sha256),
            (adapter_manifest_path, args.adapter_manifest_sha256),
        ):
            if _sha256(path) != expected:
                raise BuildError(f"input manifest changed during build: {path.name}")
        if (
            _sha256(mtp_subset) != mtp_manifest["output_sha256"]
            or _sha256(expert_scales) != scales_manifest["output_sha256"]
            or _sha256(adapter) != adapter_manifest["output_sha256"]
            or _sha256(baseline_source) != baseline_receipt["sha256"]
            or _sha256(wheel) != MODELOPT_WHEEL_SHA256
        ):
            raise BuildError("payload or ModelOpt wheel changed during build")

        validation = _validate_output(
            partial=partial,
            output_map=output_map,
            hybrid_locations=hybrid_locations,
            mtp_source=mtp_subset,
        )
        validation.update(
            {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "lm_head_lora_merged_before_quantization": True,
                "lm_head_lora_relative_frobenius_norm": relative_norm,
                "lm_head_lora_relative_frobenius_norm_limit": adapter_bound,
                "routed_expert_target_regex": _OUTPUT_EXPERT_RE.pattern,
                "source_expert_target_regex": _SOURCE_EXPERT_RE.pattern,
                "quantized_weight_dtype": "U8 packed E2M1",
                "block_scale_dtype": "F8_E4M3",
                "block_size": 16,
                "non_expert_transformer_weight_cpu_offload": False,
                "runtime_validation_status": "unvalidated-canary",
            }
        )
        _write_json(partial / "VALIDATION_REPORT.json", validation)

        build_manifest = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "status": "unvalidated-canary",
            "sources": {
                "hybrid": {
                    "manifest": HYBRID_MANIFEST_FILENAME,
                    "manifest_sha256": args.hybrid_manifest_sha256,
                    "sources": hybrid_manifest["sources"],
                },
                "official_bf16_mtp": {
                    "repo": BF16_REPO,
                    "revision": BF16_REVISION,
                    "manifest": mtp_manifest_path.name,
                    "manifest_sha256": args.mtp_manifest_sha256,
                    "payload_sha256": _sha256(mtp_subset),
                    "tensor_hash_inventory_sha256": hashlib.sha256(
                        _canonical_json(mtp_tensor_hashes)
                    ).hexdigest(),
                },
                "modelopt_reference_scales": {
                    "repo": SCALE_REPO,
                    "revision": SCALE_REVISION,
                    "manifest": scales_manifest_path.name,
                    "manifest_sha256": args.expert_scales_manifest_sha256,
                    "payload_sha256": _sha256(expert_scales),
                    "tensor_hash_inventory_sha256": hashlib.sha256(
                        _canonical_json(scale_tensor_hashes)
                    ).hexdigest(),
                },
                "behavior_adapter": {
                    "manifest": adapter_manifest_path.name,
                    "manifest_sha256": args.adapter_manifest_sha256,
                    "payload_sha256": _sha256(adapter),
                    "target_modules": ["lm_head"],
                    "gate_status": (
                        "training-integrity-passed-semantic-qualification-pending"
                    ),
                    "official_untuned_baseline": {
                        **baseline_receipt,
                        "file": SETTLED_BASELINE_FILENAME,
                    },
                },
            },
            "quantization": {
                "tool": "NVIDIA ModelOpt",
                "version": modelopt_version,
                "commit": MODELOPT_COMMIT,
                "wheel_sha256": MODELOPT_WHEEL_SHA256,
                "algorithm": "NVFP4 W4A4",
                "block_size": 16,
                "source_target_regex": _SOURCE_EXPERT_RE.pattern,
                "output_target_regex": _OUTPUT_EXPERT_RE.pattern,
                "quantized_modules": QUANTIZED_MODULE_COUNT,
                "preserved": [
                    "full vision/image/video stack",
                    "official FP8 PLE n-gram tables and BF16 scale",
                    "official BF16 MTP",
                    "all non-routed language-transformer tensors",
                    "BF16 lm_head after bounded LoRA merge",
                ],
            },
            "runtime_placement": {
                "transformer_weights": "GPU; no transformer-weight CPU offload",
                "ple_ngram_embedding": "eligible for SGLang host/RAM offload",
                "ple_tensor_dtype": "float8_e4m3fn",
                "ple_shards": PLE_TABLE_COUNT,
                "mtp": "GPU with the main model",
            },
            "build": {
                "builder": Path(__file__).name,
                "builder_sha256": _sha256(Path(__file__).resolve()),
                "checkpoint_role": "tuned",
                "python": sys.version,
                "torch": torch.__version__,
                "fleet": fleet,
                "elapsed_seconds": time.time() - started,
                "metadata_files_copied": copied_metadata,
                "passthrough_shards_copied": copied_shards,
                "passthrough_shards_rewritten": rewritten_shards,
                "expert_shards": expert_receipts,
                "mtp_shard": mtp_filename,
            },
            "validation": validation,
            "required_release_gates": [
                "clean SGLang load on one RTX PRO 6000 96GB",
                "text semantic/capability and retained-safeguard suite",
                "image inference suite",
                "video inference suite",
                "MTP on/off throughput and acceptance benchmark",
                "VRAM/RAM measurement with PLE host offload",
            ],
        }
        _write_json(partial / "BUILD_MANIFEST.json", build_manifest)
        sums = _write_sha256sums(partial)
        expected_sum_files = {
            path.name
            for path in partial.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        }
        if set(sums) != expected_sum_files:
            raise BuildError("SHA256SUMS file closure failed")
        sibling_manifest, _untuned_sums = _build_untuned_sibling(
            tuned_partial=partial,
            untuned_partial=untuned_partial,
            tuned_index=index,
            output_map=output_map,
            tuned_sums=sums,
            head_receipt=head_receipt,
        )
        _fsync_tree(partial)
        _fsync_tree(untuned_partial)
        partial.rename(output)
        untuned_partial.rename(untuned_output)
        _write_json(sibling_manifest_path, sibling_manifest)
        parent_descriptor = os.open(
            output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        # Fleet may settle this exact task-owned partial for diagnosis.  It is
        # never promoted and this builder never deletes it automatically.
        raise

    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(output),
                "official_untuned_output": str(untuned_output),
                "sibling_manifest": str(sibling_manifest_path),
                "tensor_count": EXPECTED_OUTPUT_TENSOR_COUNT,
                "elapsed_seconds": time.time() - started,
                "status": "unvalidated-canary",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        return_code = main()
    except BaseException as exc:
        _write_terminal_result(
            {
                "schema_version": SCHEMA_VERSION,
                "success": False,
                "failure_type": type(exc).__name__,
                "failure": str(exc)[:1000],
                "completed_at": time.time(),
            }
        )
        raise
    _write_terminal_result(
        {
            "schema_version": SCHEMA_VERSION,
            "success": True,
            "completed_at": time.time(),
        }
    )
    raise SystemExit(return_code)
