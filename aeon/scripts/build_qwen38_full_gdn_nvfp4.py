#!/usr/bin/env python3
"""Build an exact-ARA, full-GDN ModelOpt NVFP4 Qwen3.8 checkpoint.

This converter is intentionally shardwise.  It regenerates every language
projection from the reviewed uncensored BF16 source, reuses only calibrated
activation scales from a pinned base-model template, and preserves all other
tensors bit-for-bit.  No system prompt or tokenizer content is changed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time
from typing import Any

import torch
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from safetensors import safe_open
from safetensors.torch import load_file, save_file


SCHEMA_VERSION = "aeon-qwen38-ara-modelopt-full-gdn-nvfp4-v1"
SOURCE_REPO = "trohrbaugh/Qwen3.8-27B-heretic-ara"
SOURCE_REVISION = "a67ae100d933c0d17af3232bda35825979fc63ce"
OFFICIAL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TEMPLATE_REPO = "Mantrah/Qwen3.8-27B-NVFP4-GDN"
TEMPLATE_REVISION = "53097a45c1c56d0689da5f0aa6bce28c3d670338"
TEMPLATE_CONFIG_SHA256 = (
    "61a72634c98777cdb42c8f38485bbed79a903008405ea80f561f6f3ecf827fce"
)
TEMPLATE_INDEX_SHA256 = (
    "55c40c4f33a186e01555aec2dd1ccbbf81d9d6b03739d1e552a0bb2b07d302d7"
)
TEMPLATE_SCALES_SHA256 = (
    "859fd870b5bf0579feb136a48dd868d422e37b1f4364eaeedef91b4fd62c92b3"
)
SOURCE_WEIGHT_SHA256 = {
    "model-00001-of-00006.safetensors": "55a4ad961830c6dfae435ba5d718c40dcd6169feb26c368007f6ba8c5f0329db",
    "model-00002-of-00006.safetensors": "74e61e9be6b6f6b02e8e0ae2f7f360f9df2c271860b4f52974ef64bdefcee274",
    "model-00003-of-00006.safetensors": "3c71faa739ce1f74875363b4ed0136a21da028208f11815471917ac2b17b50a7",
    "model-00004-of-00006.safetensors": "ebf94d3caa061031ce2f183adfca73128e7d7708aa78bebd55e1278358fc2f1b",
    "model-00005-of-00006.safetensors": "c843c9f461d6533eef4141d000b9e5a03fe328825346cbe7a062104b546a6e0c",
    "model-00006-of-00006.safetensors": "f7c99ba96930a0a4a8e7850660912ac71210622aa6e709d7df9af0e641d44451",
    "model-auxiliary.safetensors": "1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe",
}
SOURCE_METADATA_SHA256 = {
    "config.json": "5a1911420c23cca59e18efe1685e66f73fa6daee946ee43f4afc9a92f4bfc43d",
    "model.safetensors.index.json": "b0eb836dd3b5d2261cbf9e49913c02fd4e2ae886b0f2129e363c0a6156673d37",
    "tokenizer.json": "6f32ce20dc35f57a7f9ad1eac03525bd7d30f9df8cea6507e958279cc3657706",
    "tokenizer_config.json": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
}
COPY_FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
_CUDA_UUID = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$"
)
_CLAIM = re.compile(r"^gc-[A-Za-z0-9._:-]{8,200}$")


class QuantizationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuantizationError(f"malformed JSON: {path}") from exc
    if not isinstance(value, dict):
        raise QuantizationError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _write_terminal_result(value: dict[str, Any]) -> None:
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
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _private_directory(path: Path, *, create: bool = False) -> Path:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise QuantizationError(f"directory is not private and owned: {path}")
    return path


def _fleet_binding() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    claim = os.environ.get("GPU_AGENT_CLAIM_ID", "")
    runtime = os.environ.get("AEON_QUANT_RUNTIME_ID", "")
    try:
        limit_gb = float(os.environ["GPU_MEM_LIMIT_GB"])
    except (KeyError, ValueError) as exc:
        raise QuantizationError("GPU_MEM_LIMIT_GB is absent or malformed") from exc
    if (
        _CUDA_UUID.fullmatch(visible) is None
        or _CLAIM.fullmatch(claim) is None
        or re.fullmatch(r"fr-[a-f0-9]{32}", runtime) is None
        or not 40.0 <= limit_gb <= 42.0
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise QuantizationError("reviewed single-GPU Fleet binding is absent")
    properties = torch.cuda.get_device_properties(0)
    total_gb = properties.total_memory / 1024**3
    if total_gb < 47.0 or limit_gb + 6.0 > total_gb + 0.1:
        raise QuantizationError("leased GPU does not preserve six GiB of headroom")
    torch.cuda.set_per_process_memory_fraction(limit_gb / total_gb, 0)
    return {
        "claim_id": claim,
        "gpu_uuid": visible,
        "gpu_name": properties.name,
        "gpu_total_gb": total_gb,
        "gpu_mem_limit_gb": limit_gb,
        "runtime_id": runtime,
    }


def _validate_source(source: Path) -> tuple[dict[str, Any], dict[str, str]]:
    for name, digest in {**SOURCE_WEIGHT_SHA256, **SOURCE_METADATA_SHA256}.items():
        path = source / name
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
            or _sha256(path) != digest
        ):
            raise QuantizationError(f"exact BF16 source identity changed: {name}")
    reproduction = _json(source / "reproduce/reproduce.json")
    if (
        reproduction.get("model") != "Qwen/Qwen3.8-27B"
        or reproduction.get("model_commit") != OFFICIAL_REVISION
        or reproduction.get("weights_sha256") != SOURCE_WEIGHT_SHA256
    ):
        raise QuantizationError("ARA reproduction receipt changed")
    index = _json(source / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if (
        not isinstance(weight_map, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items())
        or set(weight_map.values()) != set(SOURCE_WEIGHT_SHA256)
    ):
        raise QuantizationError("source weight map changed")
    return index, dict(weight_map)


def _validate_template(
    config_path: Path, scales_path: Path
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    if _sha256(config_path) != TEMPLATE_CONFIG_SHA256:
        raise QuantizationError("template config identity changed")
    if _sha256(scales_path) != TEMPLATE_SCALES_SHA256:
        raise QuantizationError("template scale identity changed")
    config = _json(config_path)
    quantization = config.get("quantization_config")
    layers = quantization.get("quantized_layers") if isinstance(quantization, dict) else None
    if not isinstance(layers, dict) or len(layers) != 401:
        raise QuantizationError("template quantized-layer map changed")
    normalized: dict[str, dict[str, Any]] = {}
    algorithms: dict[str, int] = {}
    for name, value in layers.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise QuantizationError("template quantized-layer entry is malformed")
        algorithm = str(value.get("quant_algo", "")).upper()
        if algorithm not in {"NVFP4", "FP8"}:
            raise QuantizationError("template quantization algorithm changed")
        algorithms[algorithm] = algorithms.get(algorithm, 0) + 1
        normalized[name] = dict(value)
    if algorithms != {"NVFP4": 400, "FP8": 1} or normalized.get("lm_head", {}).get(
        "quant_algo"
    ) != "FP8":
        raise QuantizationError("template precision allocation changed")
    scales = load_file(scales_path, device="cpu")
    expected = {f"{name}.input_scale" for name in normalized} | {
        f"{name}.weight_scale_2"
        for name, layer in normalized.items()
        if str(layer["quant_algo"]).upper() == "NVFP4"
    }
    if set(scales) != expected:
        raise QuantizationError("template scale tensor set changed")
    for name, tensor in scales.items():
        if tensor.dtype != torch.float32 or tensor.numel() != 1 or tensor.item() <= 0:
            raise QuantizationError(f"template scale tensor changed: {name}")
    return config, scales, normalized


def _fused_group(module: str) -> str:
    parent, leaf = module.rsplit(".", 1)
    if ".self_attn." in module and leaf in {"q_proj", "k_proj", "v_proj"}:
        return f"{parent}.qkv_proj"
    if ".linear_attn." in module and leaf in {"in_proj_qkv", "in_proj_z"}:
        return f"{parent}.in_proj_qkvz"
    if ".mlp." in module and leaf in {"gate_proj", "up_proj"}:
        return f"{parent}.gate_up_proj"
    return module


def _groups(
    layers: dict[str, dict[str, Any]], weight_map: dict[str, str]
) -> list[tuple[str, str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for module, layer in sorted(layers.items()):
        weight = f"{module}.weight"
        try:
            weight_map[weight]
        except KeyError as exc:
            raise QuantizationError(f"ARA source lacks template module: {module}") from exc
        algorithm = str(layer["quant_algo"]).upper()
        group = _fused_group(module) if algorithm == "NVFP4" else module
        grouped.setdefault(group, []).append(module)
    result = []
    for group, modules in sorted(grouped.items()):
        shards = {weight_map[f"{module}.weight"] for module in modules}
        if len(shards) != 1:
            raise QuantizationError(f"fused group crosses source shards: {group}")
        shard = shards.pop()
        result.append((shard, group, modules))
    return result


_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _sample_nvfp4_error(
    source: torch.Tensor,
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> float:
    rows = torch.linspace(
        0, source.shape[0] - 1, min(16, source.shape[0]), dtype=torch.int64
    )
    packed_rows = packed[rows.to(packed.device)]
    low = packed_rows & 0x0F
    high = packed_rows >> 4
    codes = torch.empty(
        (*packed_rows.shape[:-1], packed_rows.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    codes[..., 0::2] = low
    codes[..., 1::2] = high
    sign = torch.where(codes & 8, -1.0, 1.0)
    values = _E2M1.to(packed.device)[(codes & 7).long()] * sign
    scales = block_scale[rows.to(block_scale.device)].float() * global_scale.float()
    restored = values.view(values.shape[0], -1, 16) * scales.unsqueeze(-1)
    original = source[rows].to(restored.device, torch.float32).view_as(restored)
    numerator = (restored - original).square().sum(dtype=torch.float64)
    denominator = original.square().sum(dtype=torch.float64).clamp_min(1e-30)
    return math.sqrt(float(numerator / denominator))


def _quantize_nvfp4_group(
    modules: list[str],
    tensors: dict[str, torch.Tensor],
    scales: dict[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    source_weights = [tensors[f"{module}.weight"] for module in modules]
    if any(weight.dtype != torch.bfloat16 or weight.ndim != 2 for weight in source_weights):
        raise QuantizationError("NVFP4 source weight is not a BF16 matrix")
    input_values = [scales[f"{module}.input_scale"] for module in modules]
    if any(not torch.equal(input_values[0], value) for value in input_values[1:]):
        raise QuantizationError("fused group does not share one activation scale")
    shared_amax = max(float(weight.abs().max()) for weight in source_weights)
    if not math.isfinite(shared_amax) or shared_amax <= 0:
        raise QuantizationError("NVFP4 fused weight amax is invalid")
    global_scale = torch.tensor(
        shared_amax / (6.0 * 448.0), dtype=torch.float32, device="cuda"
    )
    errors: dict[str, float] = {}
    additions: dict[str, torch.Tensor] = {}
    for module, source in zip(modules, source_weights, strict=True):
        cuda_source = source.cuda(non_blocking=False)
        quantized, block_scale, scale_2 = NVFP4QTensor.quantize(
            cuda_source, block_size=16, weights_scaling_factor_2=global_scale
        )
        error = _sample_nvfp4_error(
            source, quantized._quantized_data, block_scale, scale_2
        )
        if not math.isfinite(error) or error >= 0.12:
            raise QuantizationError(f"NVFP4 sample error is too high for {module}: {error}")
        tensors[f"{module}.weight"] = quantized._quantized_data.cpu().contiguous()
        additions[f"{module}.weight_scale"] = block_scale.cpu().contiguous()
        additions[f"{module}.weight_scale_2"] = scale_2.cpu().contiguous()
        additions[f"{module}.input_scale"] = scales[
            f"{module}.input_scale"
        ].clone().contiguous()
        errors[module] = error
        del cuda_source, quantized, block_scale, scale_2
        torch.cuda.empty_cache()
    return errors, additions


def _quantize_fp8(
    module: str, tensors: dict[str, torch.Tensor], scales: dict[str, torch.Tensor]
) -> tuple[float, dict[str, torch.Tensor]]:
    source = tensors[f"{module}.weight"]
    if source.dtype != torch.bfloat16 or source.ndim != 2:
        raise QuantizationError("FP8 source weight is not a BF16 matrix")
    cuda_source = source.cuda(non_blocking=False)
    weight_scale = cuda_source.abs().amax().float() / 448.0
    if not torch.isfinite(weight_scale) or weight_scale.item() <= 0:
        raise QuantizationError("FP8 weight scale is invalid")
    quantized = (cuda_source.float() / weight_scale).clamp(-448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    rows = torch.linspace(
        0, source.shape[0] - 1, min(32, source.shape[0]), dtype=torch.int64
    )
    original = source[rows].cuda().float()
    restored = quantized[rows.cuda()].float() * weight_scale
    error = math.sqrt(
        float(
            (restored - original).square().sum(dtype=torch.float64)
            / original.square().sum(dtype=torch.float64).clamp_min(1e-30)
        )
    )
    if not math.isfinite(error) or error >= 0.04:
        raise QuantizationError(f"FP8 sample error is too high for {module}: {error}")
    tensors[f"{module}.weight"] = quantized.cpu().contiguous()
    additions = {
        f"{module}.weight_scale": weight_scale.cpu().contiguous(),
        f"{module}.input_scale": scales[f"{module}.input_scale"].clone().contiguous(),
    }
    del cuda_source, quantized, restored, original
    torch.cuda.empty_cache()
    return error, additions


def _copy_metadata(source: Path, output: Path) -> None:
    for name in COPY_FILES:
        path = source / name
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise QuantizationError(f"source metadata inode changed: {name}")
        shutil.copy2(path, output / name)
        (output / name).chmod(0o600)


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--template-scales", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise QuantizationError("final output already exists")
    _private_directory(source)
    _private_directory(output.parent)
    source_index, source_map = _validate_source(source)
    template_config, scales, layers = _validate_template(
        args.template_config.resolve(), args.template_scales.resolve()
    )
    groups = _groups(layers, source_map)
    if len(groups) != 257:
        raise QuantizationError(f"unexpected fused group count: {len(groups)}")
    fleet = _fleet_binding()

    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    _private_directory(partial, create=True)
    started = time.time()
    errors: dict[str, float] = {}
    output_map = dict(source_map)
    total_size = 0
    groups_by_shard: dict[str, list[tuple[str, list[str]]]] = {}
    for shard, group, modules in groups:
        groups_by_shard.setdefault(shard, []).append((group, modules))

    try:
        for shard in sorted(SOURCE_WEIGHT_SHA256):
            source_path = source / shard
            with safe_open(source_path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {"format": "pt"}
                tensors = {name: handle.get_tensor(name) for name in handle.keys()}
            for _group, modules in groups_by_shard.get(shard, []):
                algorithm = str(layers[modules[0]]["quant_algo"]).upper()
                if algorithm == "NVFP4":
                    group_errors, additions = _quantize_nvfp4_group(
                        modules, tensors, scales
                    )
                    errors.update(group_errors)
                elif algorithm == "FP8" and len(modules) == 1:
                    error, additions = _quantize_fp8(modules[0], tensors, scales)
                    errors[modules[0]] = error
                else:
                    raise QuantizationError("unsupported precision in fused group")
                for name, tensor in additions.items():
                    if name in tensors or name in output_map:
                        raise QuantizationError(f"duplicate quantized component: {name}")
                    tensors[name] = tensor
                    output_map[name] = shard
            destination = partial / shard
            save_file(tensors, destination, metadata=metadata)
            destination.chmod(0o600)
            total_size += destination.stat().st_size
            print(
                json.dumps(
                    {
                        "event": "shard_written",
                        "shard": shard,
                        "bytes": destination.stat().st_size,
                        "modules": sum(
                            len(modules)
                            for _group, modules in groups_by_shard.get(shard, [])
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del tensors
            gc.collect()
            torch.cuda.empty_cache()

        _copy_metadata(source, partial)
        config = _json(source / "config.json")
        quantization = template_config["quantization_config"]
        quantization["producer"] = {"name": "modelopt", "version": "0.46.0"}
        config["quantization_config"] = quantization
        _write_json(partial / "config.json", config)
        index = dict(source_index)
        index["weight_map"] = output_map
        index.setdefault("metadata", {})["total_size"] = total_size
        _write_json(partial / "model.safetensors.index.json", index)

        expected_keys = set(source_map) | {
            f"{module}.{suffix}"
            for module, layer in layers.items()
            for suffix in (
                ("input_scale", "weight_scale", "weight_scale_2")
                if str(layer["quant_algo"]).upper() == "NVFP4"
                else ("input_scale", "weight_scale")
            )
        }
        if set(output_map) != expected_keys or len(errors) != 401:
            raise QuantizationError("output tensor map does not match the reviewed layout")
        validation = {
            "schema_version": SCHEMA_VERSION,
            "source_tensor_count": len(source_map),
            "output_tensor_count": len(output_map),
            "nvfp4_module_count": 400,
            "fp8_module_count": 1,
            "fused_group_count": len(groups),
            "non_quantized_tensor_count": len(source_map) - 401,
            "sample_error_max": max(errors.values()),
            "sample_error_mean": sum(errors.values()) / len(errors),
            "errors": errors,
        }
        _write_json(partial / "VALIDATION_REPORT.json", validation)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "source": {
                "repo": SOURCE_REPO,
                "revision": SOURCE_REVISION,
                "official_revision": OFFICIAL_REVISION,
                "weight_sha256": SOURCE_WEIGHT_SHA256,
                "metadata_sha256": SOURCE_METADATA_SHA256,
            },
            "quantization": {
                "layout": "400 NVFP4 W4A4 language projections + FP8 lm_head",
                "block_size": 16,
                "weight_scale": "exact ARA BF16 max calibration",
                "activation_scale_template": {
                    "repo": TEMPLATE_REPO,
                    "revision": TEMPLATE_REVISION,
                    "config_sha256": TEMPLATE_CONFIG_SHA256,
                    "index_sha256": TEMPLATE_INDEX_SHA256,
                    "scales_sha256": TEMPLATE_SCALES_SHA256,
                },
                "preserved": [
                    "vision tower",
                    "embedding",
                    "GDN conv1d/in_proj_a/in_proj_b/state tensors",
                    "native MTP module",
                    "norms and biases",
                ],
            },
            "fleet": fleet,
            "versions": {
                "python": sys.version,
                "torch": torch.__version__,
                "modelopt": __import__("modelopt").__version__,
            },
            "validation": {
                "report": "VALIDATION_REPORT.json",
                "sample_error_max": validation["sample_error_max"],
                "sample_error_mean": validation["sample_error_mean"],
            },
            "elapsed_seconds": time.time() - started,
            "status": "unvalidated-canary",
        }
        _write_json(partial / "BUILD_MANIFEST.json", manifest)
        sums = []
        for path in sorted(item for item in partial.iterdir() if item.is_file()):
            sums.append(f"{_sha256(path)}  {path.name}")
        (partial / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
        (partial / "SHA256SUMS").chmod(0o600)
        _fsync_tree(partial)
        partial.rename(output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        # Preserve a failed partial for the reviewed worker's bounded settlement
        # and diagnostics; it is removed only by that worker's exact cleanup.
        raise
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(output),
                "bytes": total_size,
                "elapsed_seconds": time.time() - started,
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
