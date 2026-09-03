#!/usr/bin/env python3
"""Build the exact Aeon Qwen3.8 low-latency checkpoint derivative.

The released checkpoint already stores the 64-layer language body in NVFP4,
but leaves its untied embedding/output matrices and native MTP module in BF16.
This CPU-only builder keeps every NVFP4/body/vision tensor byte-identical and
replaces only those BF16 matrices with symmetric INT8 weights (group-128 for
the heads/embeddings and group-32 for the outlier-sensitive MTP module).  It
also creates a 40,960-row MTP-only output head.  Target verification still
uses the complete target vocabulary; the shortlist only makes draft proposals
cheaper, so speculative decoding remains distribution preserving.

The source checkpoint is never modified.  The derivative is assembled under a
private sibling directory, fully validated, and renamed into place atomically.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SCHEMA_VERSION = "aeon-qwen38-speed-variant-v3"
SOURCE = Path("/home/aday/.aeon/models/Qwen3.8-27B-ARA-abliterated-NVFP4-MTP")
OUTPUT = Path(
    "/home/aday/.aeon/models/.speed-variants/"
    "Qwen3.8-27B-ARA-abliterated-NVFP4-MTP-speed-int8-vocab40960-v3"
)
EXPECTED_SOURCE_BUILD_MANIFEST = (
    "1a3ba1eb88d0507bdef3798a6db59830dc076199b7db7d111201f6997588220e"
)
EXPECTED_SOURCE_SHA256S = (
    "e7eca7ebee03c4f27482d4fe421ca1fac9f1d9986663a51fd7614361010c1237"
)
EXPECTED_DRAFT_VOCAB_SHA256 = (
    "b64b6dfcf5441eb995ddf77d3d37b018e91b88c56ad1b4c5774ad8fbfac1c388"
)
EXPECTED_DRAFT_VOCAB_COUNT = 40_960
EXPECTED_VOCAB_SIZE = 248_320
HEAD_GROUP_SIZE = 128
MTP_GROUP_SIZE = 32
QUANT_BITS = 8
QMAX = 127
_SUM_LINE = re.compile(r"^([a-f0-9]{64})  ([^/].*)$")

EMBEDDING_WEIGHT = "model.language_model.embed_tokens.weight"
LM_HEAD_WEIGHT = "lm_head.weight"
MTP_LINEAR_MODULES = (
    "mtp.fc",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj",
)
QUANTIZED_WEIGHTS = (
    EMBEDDING_WEIGHT,
    LM_HEAD_WEIGHT,
    *(f"{module}.weight" for module in MTP_LINEAR_MODULES),
)


class SpeedVariantError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_directory(path: Path, *, create: bool = False) -> Path:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise SpeedVariantError(f"directory is not private and owned: {path}")
    return path


def _regular_owned(path: Path, *, allow_readable: bool = False) -> os.stat_result:
    metadata = path.lstat()
    forbidden = 0o022 if allow_readable else 0o077
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink < 1
        or metadata.st_mode & forbidden
    ):
        raise SpeedVariantError(f"file identity or permissions are unsafe: {path}")
    return metadata


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpeedVariantError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SpeedVariantError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Any, *, sort_keys: bool = True) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=sort_keys, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _parse_source_sums(source: Path) -> dict[str, str]:
    sums = source / "SHA256SUMS"
    _regular_owned(sums)
    if _sha256(sums) != EXPECTED_SOURCE_SHA256S:
        raise SpeedVariantError("source SHA256SUMS identity changed")
    result: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        match = _SUM_LINE.fullmatch(line)
        if match is None or ".." in Path(match.group(2)).parts:
            raise SpeedVariantError("source checksum manifest is malformed")
        result[match.group(2)] = match.group(1)
    if not result:
        raise SpeedVariantError("source checksum manifest is empty")
    for relative, expected in result.items():
        path = source / relative
        _regular_owned(path)
        if path.parent != source or _sha256(path) != expected:
            raise SpeedVariantError(f"source checksum mismatch: {relative}")
    return result


def _load_source_index(source: Path) -> tuple[dict[str, Any], dict[str, str]]:
    index = _read_json(source / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise SpeedVariantError("source weight map is malformed")
    missing = set(QUANTIZED_WEIGHTS) - set(weight_map)
    if missing:
        raise SpeedVariantError(f"source is missing speed tensors: {sorted(missing)}")
    return index, dict(weight_map)


def _load_draft_ids(path: Path) -> torch.Tensor:
    _regular_owned(path, allow_readable=True)
    if _sha256(path) != EXPECTED_DRAFT_VOCAB_SHA256:
        raise SpeedVariantError("draft vocabulary identity changed")
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpeedVariantError("draft vocabulary is malformed") from exc
    if (
        not isinstance(values, list)
        or len(values) != EXPECTED_DRAFT_VOCAB_COUNT
        or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        or values != sorted(set(values))
        or values[0] < 0
        or values[-1] >= EXPECTED_VOCAB_SIZE
    ):
        raise SpeedVariantError("draft vocabulary contract changed")
    return torch.tensor(values, dtype=torch.int64)


def _pack_int8(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT8 values into compressed-tensors little-endian INT32."""
    if values.dtype != torch.int8 or values.ndim != 2 or values.shape[1] % 4:
        raise SpeedVariantError("INT8 pack input has an invalid shape or dtype")
    unsigned = values.reshape(values.shape[0], -1, 4).to(torch.int32) + 128
    packed = unsigned[..., 0]
    packed = packed | (unsigned[..., 1] << 8)
    packed = packed | (unsigned[..., 2] << 16)
    packed = packed | (unsigned[..., 3] << 24)
    return packed.contiguous()


def _quantize_int8(
    weight: torch.Tensor, *, group_size: int = HEAD_GROUP_SIZE, row_chunk: int = 512
) -> tuple[dict[str, torch.Tensor], float]:
    if (
        weight.device.type != "cpu"
        or weight.ndim != 2
        or group_size not in {32, 64, 128}
        or weight.shape[1] % group_size
        or weight.shape[1] % 4
        or not weight.dtype.is_floating_point
    ):
        raise SpeedVariantError("weight is not a supported CPU matrix")
    output_size, input_size = weight.shape
    packed = torch.empty((output_size, input_size // 4), dtype=torch.int32)
    scales = torch.empty(
        (output_size, input_size // group_size), dtype=torch.bfloat16
    )
    error_sq = 0.0
    source_sq = 0.0
    for start in range(0, output_size, row_chunk):
        end = min(start + row_chunk, output_size)
        source = weight[start:end].to(torch.float32)
        grouped = source.reshape(end - start, input_size // group_size, group_size)
        scale = grouped.abs().amax(dim=-1, keepdim=True).div(QMAX).clamp_min(1e-12)
        quantized = grouped.div(scale).round().clamp(-QMAX, QMAX).to(torch.int8)
        stored_scale = scale.squeeze(-1).to(torch.bfloat16)
        restored = quantized.to(torch.float32).mul(stored_scale.float().unsqueeze(-1))
        error_sq += float((restored - grouped).square().sum(dtype=torch.float64))
        source_sq += float(grouped.square().sum(dtype=torch.float64))
        packed[start:end] = _pack_int8(quantized.reshape(end - start, input_size))
        scales[start:end] = stored_scale
    relative_error = math.sqrt(error_sq / max(source_sq, 1e-30))
    if not math.isfinite(relative_error) or relative_error >= 0.01:
        raise SpeedVariantError(
            f"INT8 group-{group_size} round-trip error is too high: {relative_error}"
        )
    return {
        "weight_packed": packed,
        "weight_scale": scales,
        "weight_shape": torch.tensor([output_size, input_size], dtype=torch.int64),
    }, relative_error


def _int8_group(
    targets: Iterable[str], *, group_size: int = HEAD_GROUP_SIZE
) -> dict[str, Any]:
    return {
        "format": "pack-quantized",
        "input_activations": None,
        "output_activations": None,
        "targets": list(targets),
        "weights": {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": group_size,
            "num_bits": QUANT_BITS,
            "observer": "memoryless_minmax",
            "observer_kwargs": {},
            "scale_dtype": None,
            "strategy": "group",
            "symmetric": True,
            "type": "int",
            "zp_dtype": None,
        },
    }


def _runtime_suffix_target(module: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_.]+", module):
        raise SpeedVariantError("runtime target is malformed")
    return rf"re:(?:^|.*\.){re.escape(module)}$"


def _ordered_quant_groups(groups: dict[str, Any]) -> dict[str, Any]:
    """Put path-specific schemes ahead of the generic Linear NVFP4 scheme."""
    base = {
        name: value
        for name, value in groups.items()
        if not name.startswith("group_speed_")
    }
    speed = {
        "group_speed_output_heads": _int8_group(
            [
                _runtime_suffix_target("lm_head"),
                _runtime_suffix_target("draft_lm_head"),
            ]
        ),
        "group_speed_embeddings": _int8_group(
            [_runtime_suffix_target("embed_tokens")]
        ),
        "group_speed_mtp": _int8_group(
            [_runtime_suffix_target(module) for module in MTP_LINEAR_MODULES],
            group_size=MTP_GROUP_SIZE,
        ),
    }
    return {**speed, **base}


def _copy_base_tree(source: Path, partial: Path, changed_shards: set[str]) -> None:
    skip = {
        "BUILD_MANIFEST.json",
        "README.md",
        "SHA256SUMS",
        "VALIDATION_REPORT.json",
        "config.json",
        "model.safetensors.index.json",
    }
    for path in sorted(source.iterdir(), key=lambda item: item.name):
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise SpeedVariantError(f"source contains a non-regular entry: {path.name}")
        if path.name in skip or path.name in changed_shards:
            continue
        destination = partial / path.name
        if path.suffix == ".safetensors":
            os.link(path, destination)
        else:
            shutil.copy2(path, destination)
        destination.chmod(0o600)


def _rewrite_shard(
    source: Path,
    partial: Path,
    shard: str,
    weights: list[str],
    weight_map: dict[str, str],
) -> dict[str, float]:
    source_path = source / shard
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(source_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for name in handle.keys():
            tensors[name] = handle.get_tensor(name)
    errors: dict[str, float] = {}
    for name in weights:
        tensor = tensors.pop(name, None)
        if tensor is None or tensor.dtype != torch.bfloat16:
            raise SpeedVariantError(f"expected BF16 source tensor is absent: {name}")
        module = name.removesuffix(".weight")
        group_size = MTP_GROUP_SIZE if module in MTP_LINEAR_MODULES else HEAD_GROUP_SIZE
        components, error = _quantize_int8(tensor, group_size=group_size)
        del weight_map[name]
        for suffix, value in components.items():
            component_name = f"{module}.{suffix}"
            if component_name in tensors or component_name in weight_map:
                raise SpeedVariantError(f"quantized component already exists: {component_name}")
            tensors[component_name] = value
            weight_map[component_name] = shard
        errors[module] = error
    destination = partial / shard
    save_file(tensors, destination, metadata=metadata or {"format": "pt"})
    destination.chmod(0o600)
    return errors


def _add_draft_head(
    partial: Path,
    ids: torch.Tensor,
    weight_map: dict[str, str],
) -> tuple[str, int]:
    head_shard = weight_map["lm_head.weight_packed"]
    with safe_open(partial / head_shard, framework="pt", device="cpu") as handle:
        packed = handle.get_tensor("lm_head.weight_packed").index_select(0, ids)
        scales = handle.get_tensor("lm_head.weight_scale").index_select(0, ids)
        shape = handle.get_tensor("lm_head.weight_shape")
    if int(shape[0]) != EXPECTED_VOCAB_SIZE:
        raise SpeedVariantError("target output vocabulary changed")
    extra_name = "model-speed-draft-head.safetensors"
    tensors = {
        "mtp.draft_lm_head.weight_packed": packed.contiguous(),
        "mtp.draft_lm_head.weight_scale": scales.contiguous(),
        "mtp.draft_lm_head.weight_shape": torch.tensor(
            [ids.numel(), int(shape[1])], dtype=torch.int64
        ),
    }
    save_file(tensors, partial / extra_name, metadata={"format": "pt"})
    (partial / extra_name).chmod(0o600)
    for name in tensors:
        weight_map[name] = extra_name
    ids_name = "mtp_draft_vocab_ids.safetensors"
    save_file({"ids": ids.contiguous()}, partial / ids_name, metadata={"format": "pt"})
    (partial / ids_name).chmod(0o600)
    return extra_name, sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def _rewrite_config(source: Path, partial: Path) -> dict[str, Any]:
    config = _read_json(source / "config.json")
    quant = config.get("quantization_config")
    if not isinstance(quant, dict) or quant.get("format") != "nvfp4-pack-quantized":
        raise SpeedVariantError("source is not the expected compressed NVFP4 model")
    groups = quant.get("config_groups")
    ignores = quant.get("ignore")
    if not isinstance(groups, dict) or not isinstance(ignores, list):
        raise SpeedVariantError("source quantization config is malformed")
    quantized_modules = {LM_HEAD_WEIGHT.removesuffix(".weight"), *MTP_LINEAR_MODULES}
    quant["ignore"] = [item for item in ignores if item not in quantized_modules]
    # vLLM wraps the target language model under ``language_model`` while its
    # speculative model is rooted directly at ``mtp``.  Suffix regexes match
    # both runtime hierarchies.  They must precede the generic ``Linear``
    # target because compressed-tensors selects the first matching scheme.
    quant["config_groups"] = _ordered_quant_groups(groups)
    config["aeon_speed_variant"] = {
        "schema_version": SCHEMA_VERSION,
        "body": "source NVFP4 tensors unchanged",
        "int8_group_size": {
            "heads_and_embeddings": HEAD_GROUP_SIZE,
            "mtp": MTP_GROUP_SIZE,
        },
        "draft_vocabulary_size": EXPECTED_DRAFT_VOCAB_COUNT,
    }
    # Config-group order is executable first-match precedence in
    # compressed-tensors, so it must not be alphabetically reordered.
    _write_json(partial / "config.json", config, sort_keys=False)
    return config


def _validate_output(
    partial: Path,
    weight_map: dict[str, str],
    changed_shards: set[str],
    source: Path,
) -> dict[str, Any]:
    actual: dict[str, str] = {}
    tensor_count = 0
    for path in sorted(partial.iterdir(), key=lambda item: item.name):
        _regular_owned(path)
        if path.suffix != ".safetensors" or path.name == "mtp_draft_vocab_ids.safetensors":
            continue
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in actual:
                    raise SpeedVariantError(f"duplicate output tensor: {name}")
                actual[name] = path.name
                tensor_count += 1
    if actual != weight_map:
        missing = sorted(set(weight_map) - set(actual))[:10]
        extra = sorted(set(actual) - set(weight_map))[:10]
        raise SpeedVariantError(f"output index mismatch: missing={missing}, extra={extra}")
    for shard in changed_shards:
        if os.path.samefile(source / shard, partial / shard):
            raise SpeedVariantError(f"changed shard was accidentally hardlinked: {shard}")
    required = {
        "lm_head.weight_packed",
        "model.language_model.embed_tokens.weight_packed",
        "mtp.fc.weight_packed",
        "mtp.draft_lm_head.weight_packed",
    }
    if not required.issubset(actual):
        raise SpeedVariantError("output is missing a required speed tensor")
    return {
        "tensor_count": tensor_count,
        "safetensors_bytes": sum(path.stat().st_size for path in partial.glob("*.safetensors")),
        "changed_shards": sorted(changed_shards),
    }


def _write_receipts(
    partial: Path,
    source: Path,
    draft_vocab: Path,
    errors: dict[str, float],
    validation: dict[str, Any],
) -> None:
    readme = "# Aeon Qwen3.8 27B NVFP4 low-latency canary\n\n"
    readme += "The exact released uncensored Aeon NVFP4 body and vision tensors are unchanged. "
    readme += "Only the BF16 token embeddings, target LM head, and native MTP linear weights "
    readme += "were converted to symmetric INT8 (group-128 heads/embeddings, group-32 MTP). "
    readme += "A 40,960-token INT8 output-head "
    readme += "slice is used only for MTP proposals; target verification still uses the full "
    readme += "248,320-token head. This artifact is canary-only until speed and semantic gates pass.\n"
    (partial / "README.md").write_text(readme, encoding="utf-8")
    (partial / "README.md").chmod(0o600)
    shutil.copy2(draft_vocab, partial / "draft_vocab_ids.json")
    (partial / "draft_vocab_ids.json").chmod(0o600)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "canary_unvalidated",
        "source": {
            "path": str(source),
            "build_manifest_sha256": EXPECTED_SOURCE_BUILD_MANIFEST,
            "sha256s_sha256": EXPECTED_SOURCE_SHA256S,
        },
        "draft_vocabulary": {
            "source": "syv-ai/qwen38-27b-rtx3090@2738b38fdd40d455b4cdbc35d7763f0d47203af0",
            "sha256": EXPECTED_DRAFT_VOCAB_SHA256,
            "tokens": EXPECTED_DRAFT_VOCAB_COUNT,
        },
        "quantization": {
            "body": "unchanged NVFP4 W4A4 group-16",
            "speed_tensors": {
                "heads_and_embeddings": "symmetric INT8 weight-only group-128",
                "mtp": "symmetric INT8 weight-only group-32",
            },
            "round_trip_relative_error": errors,
        },
        "validation": validation,
        "required_runtime_patch": "aeon-qwen38-speed-heads-a047e",
        "builder_sha256": _sha256(Path(__file__).resolve()),
    }
    _write_json(partial / "BUILD_MANIFEST.json", manifest)
    lines = []
    for path in sorted(partial.iterdir(), key=lambda item: item.name):
        if path.name == "SHA256SUMS":
            continue
        _regular_owned(path)
        lines.append(f"{_sha256(path)}  {path.name}\n")
    (partial / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    (partial / "SHA256SUMS").chmod(0o600)


def build(source: Path, output: Path, draft_vocab: Path) -> dict[str, Any]:
    if source != SOURCE or output != OUTPUT:
        raise SpeedVariantError("only the reviewed exact source/output paths are allowed")
    _private_directory(source)
    if output.parent.exists():
        _private_directory(output.parent)
    else:
        _private_directory(output.parent, create=True)
    if output.exists() or output.is_symlink():
        raise SpeedVariantError("output already exists; refusing to overwrite it")
    if _sha256(source / "BUILD_MANIFEST.json") != EXPECTED_SOURCE_BUILD_MANIFEST:
        raise SpeedVariantError("source build manifest identity changed")
    source_sums = _parse_source_sums(source)
    index, weight_map = _load_source_index(source)
    ids = _load_draft_ids(draft_vocab)
    shards_to_weights: dict[str, list[str]] = {}
    for name in QUANTIZED_WEIGHTS:
        shards_to_weights.setdefault(weight_map[name], []).append(name)
    changed_shards = set(shards_to_weights)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    temporary.chmod(0o700)
    try:
        _copy_base_tree(source, temporary, changed_shards)
        errors: dict[str, float] = {}
        for shard in sorted(changed_shards):
            errors.update(
                _rewrite_shard(
                    source,
                    temporary,
                    shard,
                    shards_to_weights[shard],
                    weight_map,
                )
            )
        extra_name, draft_bytes = _add_draft_head(temporary, ids, weight_map)
        _rewrite_config(source, temporary)
        index["weight_map"] = dict(sorted(weight_map.items()))
        index.setdefault("metadata", {})["total_size"] = sum(
            path.stat().st_size for path in temporary.glob("*.safetensors")
        )
        _write_json(temporary / "model.safetensors.index.json", index)
        validation = _validate_output(temporary, weight_map, changed_shards, source)
        validation["draft_head_file"] = extra_name
        validation["draft_head_tensor_bytes"] = draft_bytes
        validation["verified_source_files"] = len(source_sums)
        _write_receipts(temporary, source, draft_vocab, errors, validation)
        for path in temporary.iterdir():
            _regular_owned(path)
        os.replace(temporary, output)
        result = {
            "output": str(output),
            "build_manifest_sha256": _sha256(output / "BUILD_MANIFEST.json"),
            "sha256s_sha256": _sha256(output / "SHA256SUMS"),
            "bytes": sum(path.stat().st_size for path in output.iterdir()),
            "quantized_modules": len(errors),
            "max_relative_error": max(errors.values()),
        }
        return result
    except BaseException:
        if (
            temporary.exists()
            and temporary.parent == output.parent
            and temporary.name.startswith(f".{output.name}.partial-")
        ):
            shutil.rmtree(temporary)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--draft-vocab", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build(args.source.resolve(), args.output.resolve(), args.draft_vocab.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
