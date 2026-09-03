#!/usr/bin/env python3
"""Create an immutable Qwen3.8 Flash-Next sibling with NVFP4 MTP experts.

The source checkpoint is never modified.  Unchanged files are hard-linked only
from a closed, read-only source tree on the same filesystem; any shard containing
a replaced tensor is rewritten byte-for-byte without those tensors.  The two BF16
fused MTP routed-expert tensors and three BF16 MTP shared-expert projections are
replaced in the output index by ModelOpt NVFP4 group-16 tensors.  All other
indexed tensor bytes remain on the exact source inodes and are covered by a
tensor-level preservation digest.

This tool deliberately has no download or Fleet lifecycle logic.  It refuses
to run until an external, exact download has a complete ``SHA256SUMS`` closure.
The actual conversion is GPU-backed and must be launched by a reviewed Fleet
profile; importing and validating this module is CPU-only and side-effect free.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
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
import tempfile
from typing import Any

from aeon.scripts import build_qwen38_flash_next_nvfp4 as base


SCHEMA_VERSION = "aeon-qwen38-flash-next-mtp-nvfp4-sibling-v4"
SOURCE_REPOSITORY = "mazinb/Qwen3.8-Flash-Next-Uncensored-NVFP4"
MODELOPT_VERSION = "0.46.0"
NUM_EXPERTS = 512
HIDDEN_SIZE = 2560
INTERMEDIATE_SIZE = 640
BLOCK_SIZE = 16
MTP_GATE_UP = "mtp.layers.0.mlp.experts.gate_up_proj"
MTP_DOWN = "mtp.layers.0.mlp.experts.down_proj"
MTP_SHARED_GATE = "mtp.layers.0.mlp.shared_expert.gate_proj.weight"
MTP_SHARED_UP = "mtp.layers.0.mlp.shared_expert.up_proj.weight"
MTP_SHARED_DOWN = "mtp.layers.0.mlp.shared_expert.down_proj.weight"
OUTPUT_SHARD = "model-mtp-all-experts-nvfp4.safetensors"
BUILD_MANIFEST = "MTP_NVFP4_MANIFEST.json"
SOURCE_MANIFEST_COPY = "SOURCE_SHA256SUMS"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_CUDA_UUID = re.compile(r"^GPU-[0-9A-Fa-f-]{32,40}$")
_CLAIM = re.compile(r"^gc-[A-Za-z0-9._:-]{8,200}$")
_RUNTIME = re.compile(r"^fr-[a-f0-9]{32}$")
_SAFE_FILE = re.compile(r"^(?!\.\.?$)(?!.*\.\.)(?!.*[/\\\x00-\x1f])[A-Za-z0-9_.-]{1,240}$")
_MTP_OUTPUT = re.compile(
    r"^mtp\.layers\.0\.mlp\.(?:experts\.(\d+)|shared_expert)\."
    r"(gate_proj|up_proj|down_proj)\."
    r"(weight|weight_scale|weight_scale_2|input_scale)$"
)


class MTPQuantizationError(RuntimeError):
    """The source, conversion, or output violated the closed artifact contract."""


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, read_only: bool = False) -> os.stat_result:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
        or (read_only and metadata.st_mode & 0o222)
    ):
        raise MTPQuantizationError(f"unsafe artifact inode: {path.name}")
    return metadata


def _load_json(path: Path, *, maximum: int = 16 * 1024 * 1024) -> dict[str, Any]:
    metadata = _regular(path)
    if not 0 < metadata.st_size <= maximum:
        raise MTPQuantizationError(f"JSON artifact size is invalid: {path.name}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise MTPQuantizationError(f"JSON artifact is malformed: {path.name}") from exc
    if not isinstance(value, dict):
        raise MTPQuantizationError(f"JSON artifact root is invalid: {path.name}")
    return value


def _source_closure(root: Path, expected_manifest_sha256: str) -> dict[str, tuple[str, int]]:
    if not root.is_absolute() or not _SHA.fullmatch(expected_manifest_sha256):
        raise MTPQuantizationError("source path or manifest identity is invalid")
    root_metadata = root.lstat()
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_ISLNK(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or root_metadata.st_mode & 0o022
    ):
        raise MTPQuantizationError("source closure root is unsafe")
    manifest = root / "SHA256SUMS"
    _regular(manifest, read_only=True)
    if _sha256(manifest) != expected_manifest_sha256:
        raise MTPQuantizationError("source SHA256SUMS identity changed")
    receipts: dict[str, tuple[str, int]] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None or _SAFE_FILE.fullmatch(match.group(2)) is None:
            raise MTPQuantizationError("source SHA256SUMS is malformed")
        name = match.group(2)
        if name in receipts or name == "SHA256SUMS":
            raise MTPQuantizationError("source SHA256SUMS is not unique")
        path = root / name
        metadata = _regular(path, read_only=True)
        if _sha256(path) != match.group(1):
            raise MTPQuantizationError(f"source file identity changed: {name}")
        receipts[name] = (match.group(1), metadata.st_size)
    actual = {
        item.name
        for item in root.iterdir()
        if item.name != "SHA256SUMS" and _regular(item, read_only=True)
    }
    if actual != set(receipts):
        raise MTPQuantizationError("source tree differs from SHA256SUMS closure")
    required = {"config.json", "hf_quant_config.json", "model.safetensors.index.json"}
    if not required <= set(receipts):
        raise MTPQuantizationError("source metadata closure is incomplete")
    return receipts


def _validate_source_config(config: Mapping[str, Any]) -> None:
    text = config.get("text_config")
    quant = config.get("quantization_config")
    if (
        config.get("model_type") != "qwen4_exp"
        or not isinstance(text, Mapping)
        or text.get("num_hidden_layers") != 48
        or text.get("num_experts") != NUM_EXPERTS
        or text.get("hidden_size") != HIDDEN_SIZE
        or text.get("moe_intermediate_size") != INTERMEDIATE_SIZE
        or text.get("mtp_num_hidden_layers") != 1
        or text.get("max_position_embeddings", 0) < 128_000
        or not isinstance(quant, Mapping)
        or str(quant.get("quant_algo", "")).upper() != "NVFP4"
        or quant.get("quant_method") != "modelopt"
    ):
        raise MTPQuantizationError("source is not the reviewed Qwen3.8 Flash-Next NVFP4 topology")


def _read_safetensors_header(path: Path) -> dict[str, base.TensorRecord]:
    metadata = _regular(path, read_only=True)
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise MTPQuantizationError("safetensors prefix is truncated")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= 256 * 1024 * 1024 or header_size % 8:
            raise MTPQuantizationError("safetensors header size is invalid")
        raw = handle.read(header_size)
    try:
        header = json.loads(raw.rstrip(b" "))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MTPQuantizationError("safetensors header is malformed") from exc
    if not isinstance(header, dict):
        raise MTPQuantizationError("safetensors header root is invalid")
    raw_metadata = header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict):
        raise MTPQuantizationError("safetensors metadata is malformed")
    data_start = 8 + header_size
    records: dict[str, base.TensorRecord] = {}
    for name, descriptor in header.items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise MTPQuantizationError("safetensors descriptor is malformed")
        try:
            dtype = descriptor["dtype"]
            shape = descriptor["shape"]
            offsets = descriptor["data_offsets"]
        except KeyError as exc:
            raise MTPQuantizationError("safetensors descriptor is incomplete") from exc
        if (
            dtype not in base._DTYPE_BYTES
            or not isinstance(shape, list)
            or not all(type(item) is int and item >= 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(item) is int and item >= 0 for item in offsets)
            or offsets[1] - offsets[0] != math.prod(shape) * base._DTYPE_BYTES[dtype]
        ):
            raise MTPQuantizationError("safetensors descriptor topology is invalid")
        records[name] = base.TensorRecord(
            dtype=dtype,
            shape=tuple(shape),
            start=data_start + offsets[0],
            end=data_start + offsets[1],
        )
    cursor = data_start
    for name, record in sorted(records.items(), key=lambda item: (item[1].start, item[0])):
        if record.start != cursor:
            raise MTPQuantizationError(f"safetensors data is not contiguous: {name}")
        cursor = record.end
    if not records or cursor != metadata.st_size:
        raise MTPQuantizationError("safetensors data length is inconsistent")
    return records


def _locations(root: Path, index: Mapping[str, Any]) -> dict[str, base.TensorLocation]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise MTPQuantizationError("source safetensors index is malformed")
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str) or _SAFE_FILE.fullmatch(shard) is None:
            raise MTPQuantizationError("source safetensors index contains an unsafe entry")
        by_shard.setdefault(shard, []).append(name)
    locations: dict[str, base.TensorLocation] = {}
    for shard, names in by_shard.items():
        path = root / shard
        records = _read_safetensors_header(path)
        if set(records) != set(names):
            raise MTPQuantizationError(f"safetensors index does not close shard: {shard}")
        locations.update({name: base.TensorLocation(path, records[name]) for name in names})
    if set(locations) != set(weight_map):
        raise MTPQuantizationError("source tensor location closure failed")
    return locations


def _load_tensor(location: base.TensorLocation, name: str) -> Any:
    from safetensors import safe_open

    with safe_open(location.path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _modelopt_backend(
    wheel: Path,
) -> tuple[Callable[[Any], tuple[Any, Any, Any]], str, Mapping[str, Any]]:
    _regular(wheel, read_only=True)
    if not wheel.is_absolute() or _sha256(wheel) != base.MODELOPT_WHEEL_SHA256:
        raise MTPQuantizationError("pinned ModelOpt wheel identity changed")
    try:
        version = importlib.metadata.version("nvidia-modelopt")
    except importlib.metadata.PackageNotFoundError as exc:
        raise MTPQuantizationError("nvidia-modelopt is not installed") from exc
    if version != MODELOPT_VERSION:
        raise MTPQuantizationError(f"expected nvidia-modelopt {MODELOPT_VERSION}, found {version}")
    try:
        import torch
        from modelopt.torch.quantization.qtensor import NVFP4QTensor
    except ImportError as exc:
        raise MTPQuantizationError("ModelOpt NVFP4 backend is unavailable") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise MTPQuantizationError("conversion requires one Fleet-bound CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    claim = os.environ.get("GPU_AGENT_CLAIM_ID", "")
    runtime = os.environ.get("AEON_QUANT_RUNTIME_ID", "")
    try:
        limit_gib = float(os.environ["GPU_MEM_LIMIT_GB"])
        reserve_gib = float(os.environ.get("GPU_RESERVE_GB", "6"))
    except (KeyError, ValueError) as exc:
        raise MTPQuantizationError("Fleet GPU cap binding is absent") from exc
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024**3
    if (
        _CUDA_UUID.fullmatch(visible) is None
        or _CLAIM.fullmatch(claim) is None
        or _RUNTIME.fullmatch(runtime) is None
        or not 12 <= limit_gib <= 88.5
        or reserve_gib < 6
        or limit_gib + reserve_gib > total_gib + 0.05
        or not 94 <= total_gib <= 100
        or "RTX PRO 6000" not in properties.name.upper()
        or tuple(torch.cuda.get_device_capability(0)) < (12, 0)
    ):
        raise MTPQuantizationError("reviewed Fleet GPU binding is absent")
    torch.cuda.set_per_process_memory_fraction(limit_gib / total_gib, 0)
    fleet_binding = {
        "runtime_id": runtime,
        "claim_id_sha256": hashlib.sha256(claim.encode()).hexdigest(),
        "gpu_uuid_sha256": hashlib.sha256(visible.encode()).hexdigest(),
        "gpu_mem_limit_gib": limit_gib,
        "gpu_reserve_gib": reserve_gib,
        "gpu_total_gib": total_gib,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
    }

    def quantize(source: Any) -> tuple[Any, Any, Any]:
        cuda_source = source.cuda(non_blocking=False)
        quantized, block_scale, global_scale = NVFP4QTensor.quantize(
            cuda_source, block_size=BLOCK_SIZE
        )
        result = (
            quantized._quantized_data.cpu().contiguous(),
            block_scale.cpu().contiguous(),
            global_scale.cpu().contiguous(),
        )
        del cuda_source, quantized, block_scale, global_scale
        torch.cuda.empty_cache()
        return result

    return quantize, version, fleet_binding


def _validate_quantized(source: Any, packed: Any, scales: Any, scale_2: Any, label: str) -> None:
    import torch

    if (
        source.dtype != torch.bfloat16
        or source.shape[-1] % BLOCK_SIZE
        or packed.dtype != torch.uint8
        or tuple(packed.shape) != (*source.shape[:-1], source.shape[-1] // 2)
        or str(scales.dtype) != "torch.float8_e4m3fn"
        or tuple(scales.shape) != (*source.shape[:-1], source.shape[-1] // BLOCK_SIZE)
        or scale_2.dtype != torch.float32
        or scale_2.numel() not in {1, NUM_EXPERTS}
        or not torch.isfinite(scale_2.float()).all()
        or not (scale_2.float() > 0).all()
    ):
        raise MTPQuantizationError(f"ModelOpt returned invalid NVFP4 tensors: {label}")


def quantize_mtp_experts(
    gate_up: Any,
    down: Any,
    backend: Callable[[Any], tuple[Any, Any, Any]],
) -> dict[str, Any]:
    """Quantize only the fused MTP routed experts into ModelOpt HF tensors."""
    import torch

    if (
        gate_up.dtype != torch.bfloat16
        or tuple(gate_up.shape) != (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
        or down.dtype != torch.bfloat16
        or tuple(down.shape) != (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    ):
        raise MTPQuantizationError("MTP routed-expert BF16 topology changed")
    gate_data, gate_scales, gate_scale_2 = backend(gate_up)
    down_data, down_scales, down_scale_2 = backend(down)
    _validate_quantized(gate_up, gate_data, gate_scales, gate_scale_2, "gate_up")
    _validate_quantized(down, down_data, down_scales, down_scale_2, "down")
    gate_scale_2 = gate_scale_2.reshape(-1)
    down_scale_2 = down_scale_2.reshape(-1)
    if gate_scale_2.numel() == 1:
        gate_scale_2 = gate_scale_2.expand(NUM_EXPERTS)
    if down_scale_2.numel() == 1:
        down_scale_2 = down_scale_2.expand(NUM_EXPERTS)
    output: dict[str, Any] = {}
    one = torch.tensor(1.0, dtype=torch.float32)
    for expert in range(NUM_EXPERTS):
        for projection, packed, blocks in (
            ("gate_proj", gate_data[expert, :INTERMEDIATE_SIZE], gate_scales[expert, :INTERMEDIATE_SIZE]),
            ("up_proj", gate_data[expert, INTERMEDIATE_SIZE:], gate_scales[expert, INTERMEDIATE_SIZE:]),
            ("down_proj", down_data[expert], down_scales[expert]),
        ):
            prefix = f"mtp.layers.0.mlp.experts.{expert}.{projection}"
            output[prefix + ".weight"] = packed.contiguous()
            output[prefix + ".weight_scale"] = blocks.contiguous()
            output[prefix + ".weight_scale_2"] = (
                gate_scale_2[expert] if projection != "down_proj" else down_scale_2[expert]
            ).clone().reshape(())
            output[prefix + ".input_scale"] = one.clone()
    if len(output) != NUM_EXPERTS * 3 * 4 or any(_MTP_OUTPUT.fullmatch(name) is None for name in output):
        raise MTPQuantizationError("quantized MTP tensor name closure failed")
    return output


def quantize_mtp_shared_expert(
    gate: Any,
    up: Any,
    down: Any,
    backend: Callable[[Any], tuple[Any, Any, Any]],
) -> dict[str, Any]:
    """Quantize the MTP shared expert into the direct ModelOpt linear layout."""
    import torch

    if (
        gate.dtype != torch.bfloat16
        or tuple(gate.shape) != (INTERMEDIATE_SIZE, HIDDEN_SIZE)
        or up.dtype != torch.bfloat16
        or tuple(up.shape) != (INTERMEDIATE_SIZE, HIDDEN_SIZE)
        or down.dtype != torch.bfloat16
        or tuple(down.shape) != (HIDDEN_SIZE, INTERMEDIATE_SIZE)
    ):
        raise MTPQuantizationError("MTP shared-expert BF16 topology changed")
    output: dict[str, Any] = {}
    one = torch.tensor(1.0, dtype=torch.float32)
    for projection, source in (
        ("gate_proj", gate),
        ("up_proj", up),
        ("down_proj", down),
    ):
        packed, scales, scale_2 = backend(source)
        _validate_quantized(source, packed, scales, scale_2, f"shared_{projection}")
        prefix = f"mtp.layers.0.mlp.shared_expert.{projection}"
        output[prefix + ".weight"] = packed.contiguous()
        output[prefix + ".weight_scale"] = scales.contiguous()
        output[prefix + ".weight_scale_2"] = scale_2.reshape(-1)[0].clone().reshape(())
        output[prefix + ".input_scale"] = one.clone()
    if len(output) != 12 or any(_MTP_OUTPUT.fullmatch(name) is None for name in output):
        raise MTPQuantizationError("quantized MTP shared-expert tensor closure failed")
    return output


def _mtp_ignore() -> list[str]:
    """Exclude every BF16 MTP family except its routed and shared experts."""
    return [
        "mtp.fc_*",
        "mtp.hyper_connection_mixer.*",
        "mtp.layers.*.attn_hyper_connection.*",
        "mtp.layers.*.mlp.gate*",
        "mtp.layers.*.mlp_hyper_connection.*",
        "mtp.layers.*.self_attn.*",
        "mtp.pre_fc_*",
        "model.mtp.fc_*",
        "model.mtp.hyper_connection_mixer.*",
        "model.mtp.layers.*.attn_hyper_connection.*",
        "model.mtp.layers.*.mlp.gate*",
        "model.mtp.layers.*.mlp_hyper_connection.*",
        "model.mtp.layers.*.self_attn.*",
        "model.mtp.pre_fc_*",
    ]


def _updated_quant_config(source: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(source))
    if isinstance(result.get("ignore"), list):
        owner = result
        key = "ignore"
    elif isinstance(result.get("quantization"), dict):
        owner = result["quantization"]
        key = "exclude_modules"
    else:
        raise MTPQuantizationError("source ModelOpt ignore container is malformed")
    ignore = owner.get(key)
    if not isinstance(ignore, list) or not all(isinstance(item, str) for item in ignore):
        raise MTPQuantizationError("source ModelOpt ignore list is malformed")
    removed = {"mtp.*", "model.mtp.*"}
    if not removed <= set(ignore):
        raise MTPQuantizationError("source does not explicitly exclude the BF16 MTP")
    # The upstream blanket shared-expert exclusions also match the MTP draft.
    # Narrow them to the 48 target-model layers while opening only the single
    # MTP shared expert that this sibling actually converts.
    blanket_shared = {
        "*.mlp.shared_expert.*",
        "*.mlp.shared_expert*",
    }
    main_shared = [
        f"{prefix}.layers.{layer}.mlp.shared_expert.*"
        for prefix in ("model.language_model", "language_model.model")
        for layer in range(48)
    ]
    owner[key] = [
        item for item in ignore
        if item not in removed and item not in blanket_shared
    ] + main_shared + _mtp_ignore()
    return result


def _preserved_digest(
    locations: Mapping[str, base.TensorLocation], selected: set[str]
) -> tuple[int, str]:
    aggregate = hashlib.sha256()
    count = 0
    for name in sorted(set(locations) - selected):
        digest = base._tensor_sha256(locations[name])
        aggregate.update(name.encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
        count += 1
    return count, aggregate.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))
    path.chmod(0o444)


def _write_safetensors_subset(
    source: Path,
    destination: Path,
    records: Mapping[str, base.TensorRecord],
) -> None:
    """Copy a strict tensor subset while preserving every retained tensor byte."""
    if not records:
        raise MTPQuantizationError("refusing to write an empty safetensors subset")
    ordered = sorted(records.items(), key=lambda item: (item[1].start, item[0]))
    cursor = 0
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for name, record in ordered:
        size = record.end - record.start
        header[name] = {
            "dtype": record.dtype,
            "shape": list(record.shape),
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    raw_header = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    raw_header += b" " * ((-len(raw_header)) % 8)
    if not 2 <= len(raw_header) <= 256 * 1024 * 1024:
        raise MTPQuantizationError("rewritten safetensors header size is invalid")
    source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
    destination_fd = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o400
    )
    try:
        os.write(destination_fd, struct.pack("<Q", len(raw_header)))
        os.write(destination_fd, raw_header)
        for _name, record in ordered:
            offset = record.start
            remaining = record.end - record.start
            while remaining:
                chunk = os.pread(source_fd, min(8 * 1024 * 1024, remaining), offset)
                if not chunk:
                    raise MTPQuantizationError("source tensor truncated during rewrite")
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    if written <= 0:
                        raise MTPQuantizationError("short safetensors subset write")
                    view = view[written:]
                offset += len(chunk)
                remaining -= len(chunk)
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)
        os.close(source_fd)
    _read_safetensors_header(destination)


def _write_sums(root: Path) -> str:
    files = sorted(item for item in root.iterdir() if item.is_file() and item.name != "SHA256SUMS")
    lines = [f"{_sha256(item)}  {item.name}\n" for item in files]
    path = root / "SHA256SUMS"
    path.write_text("".join(lines), encoding="ascii")
    path.chmod(0o444)
    return _sha256(path)


def convert(
    source: Path,
    destination: Path,
    *,
    source_manifest_sha256: str,
    source_revision: str,
    modelopt_wheel: Path | None = None,
    backend: Callable[[Any], tuple[Any, Any, Any]] | None = None,
    modelopt_version: str | None = None,
    fleet_binding: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if (
        not source.is_absolute()
        or not destination.is_absolute()
        or source == destination
        or destination.exists()
        or re.fullmatch(r"[0-9a-f]{40,64}", source_revision) is None
    ):
        raise MTPQuantizationError("source/destination/revision contract is invalid")
    receipts = _source_closure(source, source_manifest_sha256)
    config = _load_json(source / "config.json")
    hf_quant = _load_json(source / "hf_quant_config.json")
    index = _load_json(source / "model.safetensors.index.json", maximum=256 * 1024 * 1024)
    _validate_source_config(config)
    locations = _locations(source, index)
    selected = {
        MTP_GATE_UP, MTP_DOWN, MTP_SHARED_GATE, MTP_SHARED_UP, MTP_SHARED_DOWN,
    }
    if not selected <= set(locations):
        raise MTPQuantizationError("source BF16 MTP expert closure is absent")
    if backend is None:
        if modelopt_wheel is None:
            raise MTPQuantizationError("pinned ModelOpt wheel receipt is absent")
        backend, detected, fleet_binding = _modelopt_backend(modelopt_wheel)
        modelopt_version = detected
    if modelopt_version != MODELOPT_VERSION or not isinstance(fleet_binding, Mapping):
        raise MTPQuantizationError("ModelOpt conversion identity changed")

    gate_up = _load_tensor(locations[MTP_GATE_UP], MTP_GATE_UP)
    down = _load_tensor(locations[MTP_DOWN], MTP_DOWN)
    quantized = quantize_mtp_experts(gate_up, down, backend)
    del gate_up, down
    shared_gate = _load_tensor(locations[MTP_SHARED_GATE], MTP_SHARED_GATE)
    shared_up = _load_tensor(locations[MTP_SHARED_UP], MTP_SHARED_UP)
    shared_down = _load_tensor(locations[MTP_SHARED_DOWN], MTP_SHARED_DOWN)
    shared_quantized = quantize_mtp_shared_expert(
        shared_gate, shared_up, shared_down, backend
    )
    if set(quantized) & set(shared_quantized):
        raise MTPQuantizationError("MTP quantized tensor names collide")
    quantized.update(shared_quantized)
    del shared_gate, shared_up, shared_down, shared_quantized
    preserved_count, preserved_sha256 = _preserved_digest(locations, selected)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise MTPQuantizationError("source and destination are not on one filesystem")
    partial = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    try:
        reserved = {
            "SHA256SUMS", "config.json", "hf_quant_config.json",
            "model.safetensors.index.json", OUTPUT_SHARD, BUILD_MANIFEST,
            SOURCE_MANIFEST_COPY,
        }
        if reserved & (set(receipts) - {"SHA256SUMS", "config.json", "hf_quant_config.json", "model.safetensors.index.json"}):
            raise MTPQuantizationError("source collides with sibling-owned metadata")
        rewritten_shards = {locations[name].path.name for name in selected}
        for name in sorted(set(receipts) - {"config.json", "hf_quant_config.json", "model.safetensors.index.json"}):
            if name in rewritten_shards:
                retained = {
                    tensor_name: location.record
                    for tensor_name, location in locations.items()
                    if location.path.name == name and tensor_name not in selected
                }
                if retained:
                    _write_safetensors_subset(source / name, partial / name, retained)
                continue
            os.link(source / name, partial / name, follow_symlinks=False)
            if (source / name).stat().st_ino != (partial / name).stat().st_ino:
                raise MTPQuantizationError("unchanged file hardlink identity failed")

        from safetensors.torch import save_file

        save_file(quantized, partial / OUTPUT_SHARD, metadata={"format": "pt"})
        (partial / OUTPUT_SHARD).chmod(0o444)
        output_index = json.loads(json.dumps(index))
        weight_map = output_index["weight_map"]
        del weight_map[MTP_GATE_UP]
        del weight_map[MTP_DOWN]
        weight_map.update({name: OUTPUT_SHARD for name in quantized})
        output_index["metadata"] = dict(output_index.get("metadata") or {})
        old_total = output_index["metadata"].get("total_size")
        if type(old_total) is not int or old_total <= 0:
            raise MTPQuantizationError("source index total_size is invalid")
        removed_bytes = sum(
            locations[name].record.end - locations[name].record.start for name in selected
        )
        added_bytes = sum(value.numel() * value.element_size() for value in quantized.values())
        output_index["metadata"]["total_size"] = old_total - removed_bytes + added_bytes

        output_hf_quant = _updated_quant_config(hf_quant)
        output_config = json.loads(json.dumps(config))
        output_config["quantization_config"] = _updated_quant_config(config["quantization_config"])
        _write_json(partial / "config.json", output_config)
        _write_json(partial / "hf_quant_config.json", output_hf_quant)
        _write_json(partial / "model.safetensors.index.json", output_index)
        (partial / SOURCE_MANIFEST_COPY).write_bytes((source / "SHA256SUMS").read_bytes())
        (partial / SOURCE_MANIFEST_COPY).chmod(0o444)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "source": {
                "repository": SOURCE_REPOSITORY,
                "revision": source_revision,
                "sha256sums_sha256": source_manifest_sha256,
            },
            "conversion": {
                "algorithm": "NVFP4",
                "group_size": BLOCK_SIZE,
                "calibration": "none_rtn",
                "input_scale": 1.0,
                "modelopt_version": modelopt_version,
                "modelopt_wheel_sha256": base.MODELOPT_WHEEL_SHA256,
                "fleet_binding": dict(fleet_binding),
                "selected_source_tensors": sorted(selected),
                "output_tensor_count": len(quantized),
                "output_shard": OUTPUT_SHARD,
            },
            "preservation": {
                "tensor_count": preserved_count,
                "name_and_tensor_sha256_digest": preserved_sha256,
                "mechanism": "same-inode-hardlinks-plus-byte-preserving-rewritten-source-shards",
                "rewritten_shards": sorted(rewritten_shards),
            },
        }
        _write_json(partial / BUILD_MANIFEST, manifest)
        closed_locations = _locations(partial, output_index)
        if set(closed_locations) != set(weight_map):
            raise MTPQuantizationError("output physical tensor closure changed")
        sums_sha256 = _write_sums(partial)
        os.rename(partial, destination)
        return dict(manifest, sha256sums_sha256=sums_sha256)
    except Exception:
        if partial.exists() and partial.parent == destination.parent and partial.name.startswith(f".{destination.name}.partial-"):
            shutil.rmtree(partial)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--modelopt-wheel", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = convert(
        arguments.source.resolve(), arguments.destination.resolve(),
        source_manifest_sha256=arguments.source_manifest_sha256,
        source_revision=arguments.source_revision,
        modelopt_wheel=arguments.modelopt_wheel.resolve(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
