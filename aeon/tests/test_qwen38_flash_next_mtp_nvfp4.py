from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from aeon.scripts import quantize_qwen38_flash_next_mtp_nvfp4 as converter


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_updated_quant_config_opens_only_mtp_routed_experts() -> None:
    source = {"ignore": ["model.embed_tokens", "mtp.*", "model.mtp.*"]}
    result = converter._updated_quant_config(source)
    assert "mtp.*" not in result["ignore"]
    assert "model.mtp.*" not in result["ignore"]
    assert "mtp.layers.*.self_attn.*" in result["ignore"]
    assert not any(item in {"mtp.layers.*.mlp.experts.*", "model.mtp.layers.*.mlp.experts.*"} for item in result["ignore"])
    assert "*.mlp.shared_expert.*" not in result["ignore"]
    assert "model.language_model.layers.0.mlp.shared_expert.*" in result["ignore"]
    assert "model.language_model.layers.47.mlp.shared_expert.*" in result["ignore"]
    assert "model.language_model.layers.48.mlp.shared_expert.*" not in result["ignore"]


def test_updated_hf_quant_config_preserves_nested_schema() -> None:
    source = {
        "producer": {"name": "modelopt"},
        "quantization": {
            "quant_algo": "NVFP4",
            "exclude_modules": ["mtp.*", "model.mtp.*", "model.embed_tokens"],
        },
    }
    result = converter._updated_quant_config(source)
    excludes = result["quantization"]["exclude_modules"]
    assert "mtp.*" not in excludes
    assert "mtp.layers.*.self_attn.*" in excludes
    assert result["producer"] == source["producer"]


def test_quantized_result_is_exact_modelopt_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(converter, "NUM_EXPERTS", 2)
    monkeypatch.setattr(converter, "HIDDEN_SIZE", 16)
    monkeypatch.setattr(converter, "INTERMEDIATE_SIZE", 16)
    gate = torch.zeros((2, 32, 16), dtype=torch.bfloat16)
    down = torch.zeros((2, 16, 16), dtype=torch.bfloat16)

    def backend(value: torch.Tensor):
        packed = torch.zeros((*value.shape[:-1], value.shape[-1] // 2), dtype=torch.uint8)
        scales = torch.zeros((*value.shape[:-1], value.shape[-1] // 16), dtype=torch.float8_e4m3fn)
        scale_count = value.shape[0] if value.ndim == 3 else 1
        return packed, scales, torch.ones(scale_count, dtype=torch.float32)

    result = converter.quantize_mtp_experts(gate, down, backend)
    assert len(result) == 2 * 3 * 4
    assert result["mtp.layers.0.mlp.experts.1.down_proj.input_scale"].item() == 1.0
    assert result["mtp.layers.0.mlp.experts.0.gate_proj.weight"].shape == (16, 8)


def test_quantizer_refuses_changed_bf16_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(converter, "NUM_EXPERTS", 2)
    monkeypatch.setattr(converter, "HIDDEN_SIZE", 16)
    monkeypatch.setattr(converter, "INTERMEDIATE_SIZE", 16)
    with pytest.raises(converter.MTPQuantizationError, match="topology changed"):
        converter.quantize_mtp_experts(
            torch.zeros((2, 31, 16), dtype=torch.bfloat16),
            torch.zeros((2, 16, 16), dtype=torch.bfloat16),
            lambda value: (value, value, value),
        )


def test_shared_expert_quantized_result_is_direct_modelopt_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(converter, "HIDDEN_SIZE", 16)
    monkeypatch.setattr(converter, "INTERMEDIATE_SIZE", 16)

    def backend(value: torch.Tensor):
        packed = torch.zeros((*value.shape[:-1], value.shape[-1] // 2), dtype=torch.uint8)
        scales = torch.zeros((*value.shape[:-1], value.shape[-1] // 16), dtype=torch.float8_e4m3fn)
        return packed, scales, torch.ones(1, dtype=torch.float32)

    result = converter.quantize_mtp_shared_expert(
        torch.zeros((16, 16), dtype=torch.bfloat16),
        torch.zeros((16, 16), dtype=torch.bfloat16),
        torch.zeros((16, 16), dtype=torch.bfloat16),
        backend,
    )
    assert len(result) == 12
    assert result["mtp.layers.0.mlp.shared_expert.down_proj.weight"].shape == (16, 8)
    assert result["mtp.layers.0.mlp.shared_expert.gate_proj.input_scale"].item() == 1.0


def test_source_closure_requires_read_only_exact_files(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "hf_quant_config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    for item in tmp_path.iterdir():
        item.chmod(0o444)
    sums = "".join(f"{_sha(item)}  {item.name}\n" for item in sorted(tmp_path.iterdir()))
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(sums)
    manifest.chmod(0o444)
    closure = converter._source_closure(tmp_path.resolve(), _sha(manifest))
    assert set(closure) == {"config.json", "hf_quant_config.json", "model.safetensors.index.json"}
    (tmp_path / "config.json").chmod(0o644)
    with pytest.raises(converter.MTPQuantizationError, match="unsafe artifact inode"):
        converter._source_closure(tmp_path.resolve(), _sha(manifest))


def test_source_closure_refuses_unmanifested_file(tmp_path: Path) -> None:
    for name in ("config.json", "hf_quant_config.json", "model.safetensors.index.json"):
        path = tmp_path / name
        path.write_text("{}")
        path.chmod(0o444)
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text("".join(f"{_sha(item)}  {item.name}\n" for item in sorted(tmp_path.iterdir())))
    manifest.chmod(0o444)
    extra = tmp_path / "extra.json"
    extra.write_text("{}")
    extra.chmod(0o444)
    with pytest.raises(converter.MTPQuantizationError, match="differs"):
        converter._source_closure(tmp_path.resolve(), _sha(manifest))


def test_locations_refuses_index_shard_mismatch(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    save_file({"present": torch.ones(1)}, shard)
    shard.chmod(0o444)
    with pytest.raises(converter.MTPQuantizationError, match="does not close"):
        converter._locations(tmp_path, {"weight_map": {"missing": shard.name}})


def test_convert_refuses_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    with pytest.raises(converter.MTPQuantizationError, match="contract is invalid"):
        converter.convert(
            source.resolve(), destination.resolve(),
            source_manifest_sha256="0" * 64, source_revision="a" * 40,
            backend=lambda value: value, modelopt_version=converter.MODELOPT_VERSION,
            fleet_binding={"runtime_id": "test"},
        )


def test_convert_creates_closed_sibling_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(converter, "NUM_EXPERTS", 2)
    monkeypatch.setattr(converter, "HIDDEN_SIZE", 16)
    monkeypatch.setattr(converter, "INTERMEDIATE_SIZE", 16)
    source = tmp_path / "source"
    destination = tmp_path / "sibling"
    source.mkdir()
    source.chmod(0o700)
    tensors = {
        converter.MTP_GATE_UP: torch.zeros((2, 32, 16), dtype=torch.bfloat16),
        converter.MTP_DOWN: torch.zeros((2, 16, 16), dtype=torch.bfloat16),
        converter.MTP_SHARED_GATE: torch.zeros((16, 16), dtype=torch.bfloat16),
        converter.MTP_SHARED_UP: torch.zeros((16, 16), dtype=torch.bfloat16),
        converter.MTP_SHARED_DOWN: torch.zeros((16, 16), dtype=torch.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": torch.arange(16, dtype=torch.bfloat16),
    }
    shard = source / "model.safetensors"
    save_file(tensors, shard)
    total_size = sum(value.numel() * value.element_size() for value in tensors.values())
    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "num_hidden_layers": 48,
            "num_experts": 2,
            "hidden_size": 16,
            "moe_intermediate_size": 16,
            "mtp_num_hidden_layers": 1,
            "max_position_embeddings": 262144,
        },
        "quantization_config": {
            "quant_algo": "NVFP4",
            "quant_method": "modelopt",
            "ignore": ["model.embed_tokens", "mtp.*", "model.mtp.*"],
        },
    }
    hf_quant = {
        "quant_algo": "NVFP4",
        "quant_method": "modelopt",
        "ignore": ["model.embed_tokens", "mtp.*", "model.mtp.*"],
    }
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {name: shard.name for name in tensors},
    }
    for name, value in (
        ("config.json", config),
        ("hf_quant_config.json", hf_quant),
        ("model.safetensors.index.json", index),
    ):
        (source / name).write_text(json.dumps(value))
    for item in source.iterdir():
        item.chmod(0o444)
    sums = source / "SHA256SUMS"
    sums.write_text("".join(f"{_sha(item)}  {item.name}\n" for item in sorted(source.iterdir())))
    sums.chmod(0o444)
    source_hashes = {item.name: _sha(item) for item in source.iterdir()}

    def backend(value: torch.Tensor):
        packed = torch.zeros((*value.shape[:-1], value.shape[-1] // 2), dtype=torch.uint8)
        scales = torch.zeros((*value.shape[:-1], value.shape[-1] // 16), dtype=torch.float8_e4m3fn)
        scale_count = value.shape[0] if value.ndim == 3 else 1
        return packed, scales, torch.ones(scale_count, dtype=torch.float32)

    receipt = converter.convert(
        source.resolve(), destination.resolve(),
        source_manifest_sha256=_sha(sums), source_revision="a" * 40,
        backend=backend, modelopt_version=converter.MODELOPT_VERSION,
        fleet_binding={"runtime_id": "fr-test"},
    )
    assert receipt["complete"] is True
    assert {item.name: _sha(item) for item in source.iterdir()} == source_hashes
    assert (source / shard.name).stat().st_ino != (destination / shard.name).stat().st_ino
    rewritten = converter._read_safetensors_header(destination / shard.name)
    assert set(rewritten) == {"mtp.pre_fc_norm_hidden.weight"}
    output_index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert converter.MTP_GATE_UP not in output_index["weight_map"]
    assert converter.MTP_DOWN not in output_index["weight_map"]
    assert output_index["weight_map"]["mtp.pre_fc_norm_hidden.weight"] == shard.name
    assert output_index["weight_map"]["mtp.layers.0.mlp.experts.1.down_proj.weight"] == converter.OUTPUT_SHARD
    assert output_index["weight_map"]["mtp.layers.0.mlp.shared_expert.down_proj.weight"] == converter.OUTPUT_SHARD
    physical = converter._locations(destination, output_index)
    assert set(physical) == set(output_index["weight_map"])
    output_config = json.loads((destination / "config.json").read_text())
    assert "mtp.*" not in output_config["quantization_config"]["ignore"]
    assert _sha(destination / "SHA256SUMS") == receipt["sha256sums_sha256"]
