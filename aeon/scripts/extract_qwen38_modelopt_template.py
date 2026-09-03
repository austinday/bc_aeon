#!/usr/bin/env python3
"""Extract only calibrated scalar scales from a pinned ModelOpt checkpoint.

The remote checkpoint is nearly 20 GB, but all activation/global weight scales
needed to re-quantize an exact BF16 derivative occupy only a few kilobytes near
the start of its safetensors data sections.  This tool uses immutable-revision
HTTP range requests and emits a small, hashable offline template artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request
from typing import Any

from safetensors.torch import save_file
import torch


REPO = "Mantrah/Qwen3.8-27B-NVFP4-GDN"
REVISION = "53097a45c1c56d0689da5f0aa6bce28c3d670338"
CONFIG_SHA256 = "61a72634c98777cdb42c8f38485bbed79a903008405ea80f561f6f3ecf827fce"
INDEX_SHA256 = "55c40c4f33a186e01555aec2dd1ccbbf81d9d6b03739d1e552a0bb2b07d302d7"
ALLOWED_SHARDS = {
    "model-00001-of-00002.safetensors": 9_972_777_720,
    "model-00002-of-00002.safetensors": 8_499_571_984,
    "model_mtp.safetensors": 849_400_424,
}
SCALE_SUFFIXES = (".input_scale", ".weight_scale_2")
SCHEMA_VERSION = "aeon-qwen38-modelopt-template-v1"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class TemplateError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _range(url: str, start: int, end: int) -> tuple[bytes, str | None]:
    if start < 0 or end < start or end - start > 2 * 1024 * 1024:
        raise TemplateError("remote range exceeds the reviewed bound")
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "Aeon-Qwen38-template/1",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read(end - start + 2)
        content_range = response.headers.get("Content-Range")
        status = response.status
    if status != 206 or len(payload) != end - start + 1:
        raise TemplateError("server did not honor the exact byte range")
    return payload, content_range


def _header(url: str, expected_size: int) -> tuple[dict[str, Any], int]:
    prefix, content_range = _range(url, 0, 7)
    match = re.fullmatch(r"bytes 0-7/([0-9]+)", content_range or "")
    if match is None or int(match.group(1)) != expected_size:
        raise TemplateError("remote shard size changed")
    header_size = struct.unpack("<Q", prefix)[0]
    if not 1 <= header_size <= 1024 * 1024:
        raise TemplateError("safetensors header size is outside its bound")
    raw, _ = _range(url, 8, 7 + header_size)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TemplateError("remote safetensors header is malformed") from exc
    if not isinstance(value, dict):
        raise TemplateError("remote safetensors header is not an object")
    return value, 8 + header_size


def _decode_f32(raw: bytes, shape: list[int]) -> torch.Tensor:
    count = 1
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise TemplateError("scale tensor shape is malformed")
        count *= dimension
    if len(raw) != count * 4:
        raise TemplateError("scale tensor byte count is malformed")
    values = struct.unpack(f"<{count}f", raw) if count else ()
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def _load_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if _sha256_bytes(raw) != expected_sha256:
        raise TemplateError(f"pinned metadata identity changed: {path.name}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TemplateError(f"pinned metadata is malformed: {path.name}") from exc
    if not isinstance(value, dict):
        raise TemplateError(f"pinned metadata root is invalid: {path.name}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = args.metadata_dir.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise TemplateError("template output already exists")

    config = _load_json(metadata / "config.json", CONFIG_SHA256)
    index = _load_json(metadata / "model.safetensors.index.json", INDEX_SHA256)
    weight_map = index.get("weight_map")
    quantization = config.get("quantization_config")
    layers = quantization.get("quantized_layers") if isinstance(quantization, dict) else None
    if (
        not isinstance(weight_map, dict)
        or not isinstance(layers, dict)
        or len(layers) != 401
        or set(weight_map.values()) != set(ALLOWED_SHARDS)
    ):
        raise TemplateError("pinned template model structure changed")

    tensors: dict[str, torch.Tensor] = {}
    shard_receipts: dict[str, Any] = {}
    base = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
    for shard, expected_size in sorted(ALLOWED_SHARDS.items()):
        if _SAFE_NAME.fullmatch(shard) is None:
            raise TemplateError("unsafe template shard name")
        header, data_start = _header(base + shard, expected_size)
        selected: dict[str, dict[str, Any]] = {}
        for name, descriptor in header.items():
            if name == "__metadata__" or not name.endswith(SCALE_SUFFIXES):
                continue
            if not isinstance(descriptor, dict) or descriptor.get("dtype") != "F32":
                raise TemplateError(f"template scale dtype changed: {name}")
            offsets = descriptor.get("data_offsets")
            shape = descriptor.get("shape")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(item, int) for item in offsets)
                or offsets[0] < 0
                or offsets[1] <= offsets[0]
                or not isinstance(shape, list)
            ):
                raise TemplateError(f"template scale descriptor changed: {name}")
            selected[name] = descriptor
        if not selected:
            continue
        range_end = max(item["data_offsets"][1] for item in selected.values())
        if range_end > 64 * 1024:
            raise TemplateError("template scale prefix exceeded its reviewed bound")
        raw, _ = _range(base + shard, data_start, data_start + range_end - 1)
        for name, descriptor in selected.items():
            start, end = descriptor["data_offsets"]
            tensors[name] = _decode_f32(raw[start:end], descriptor["shape"])
        shard_receipts[shard] = {
            "bytes": expected_size,
            "header_sha256": _sha256_bytes(
                json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
            ),
            "scale_prefix_bytes": range_end,
            "scale_tensors": len(selected),
        }

    expected = {
        f"{name}.input_scale" for name in layers
    } | {
        f"{name}.weight_scale_2"
        for name, layer in layers.items()
        if isinstance(layer, dict) and str(layer.get("quant_algo", "")).upper() == "NVFP4"
    }
    if set(tensors) != expected or len(tensors) != 801:
        missing = sorted(expected - set(tensors))[:5]
        extra = sorted(set(tensors) - expected)[:5]
        raise TemplateError(f"template scale set changed: missing={missing} extra={extra}")
    for name, tensor in tensors.items():
        if tensor.numel() != 1 or not torch.isfinite(tensor).all() or tensor.item() <= 0:
            raise TemplateError(f"template scale is not a positive scalar: {name}")

    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.parent.chmod(0o700)
    save_file(
        tensors,
        output,
        metadata={
            "format": "pt",
            "schema_version": SCHEMA_VERSION,
            "repo": REPO,
            "revision": REVISION,
            "config_sha256": CONFIG_SHA256,
            "index_sha256": INDEX_SHA256,
        },
    )
    output.chmod(0o600)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "repo": REPO,
        "revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "scale_count": len(tensors),
        "scales_sha256": _sha256(output),
        "shards": shard_receipts,
    }
    receipt_path = output.with_suffix(output.suffix + ".json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt_path.chmod(0o600)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
