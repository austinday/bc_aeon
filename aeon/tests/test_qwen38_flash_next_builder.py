from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from aeon.scripts import build_qwen38_flash_next_nvfp4 as builder


def _tensor_hash(path: Path, name: str) -> str:
    _metadata, records = builder._read_safetensors_header(path)
    return builder._tensor_sha256(builder.TensorLocation(path, records[name]))


def test_hybrid_manifest_contract_is_precise_and_pinned() -> None:
    contract = builder.HYBRID_MANIFEST_CONTRACT
    assert contract == {
        "schema_version": "aeon-qwen38-flash-next-hybrid-v1",
        "complete": True,
        "artifact": "qwen38-flash-next-tensor-hybrid",
        "sources": {
            "bf16": {
                "repo": "Qwen/Qwen3.8-Flash-Next",
                "revision": "f5d08274bafd880402bd16f5e3e6c514136ec06c",
            },
            "fp8_ple": {
                "repo": "Qwen/Qwen3.8-Flash-Next-FP8",
                "revision": "bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
            },
        },
        "upstream_metadata": {
            "bf16_config_sha256": builder.BF16_CONFIG_SHA256,
            "bf16_index_sha256": builder.BF16_INDEX_SHA256,
        },
        "topology": {
            "tensor_count": 1659,
            "bf16_source_expert_tensor_count": 96,
            "bf16_mtp_tensor_count": 31,
            "bf16_vision_tensor_count": 333,
            "fp8_ple_table_tensor_count": 128,
            "bf16_ple_scale_tensor_count": 1,
            "non_expert_non_mtp_tensor_count": 1532,
        },
        "files": "<closed file-receipt map described above>",
    }
    assert builder.EXPECTED_OUTPUT_TENSOR_COUNT == 296_475
    assert len(builder.MTP_NAMES) == 31


def test_expert_target_classification_is_closed() -> None:
    assert (
        builder._expert_source_layer(
            "model.language_model.layers.47.mlp.experts.gate_up_proj"
        )
        == 47
    )
    assert (
        builder._expert_source_layer(
            "model.language_model.layers.0.mlp.experts.down_proj"
        )
        == 0
    )
    assert builder._expert_source_layer("mtp.layers.0.mlp.experts.down_proj") is None
    assert (
        builder._expert_source_layer("model.visual.blocks.0.mlp.experts.down_proj")
        is None
    )
    assert builder._expert_source_layer(builder.PLE_PREFIX + "shard_0.weight") is None
    assert (
        builder._expert_source_layer(
            "model.language_model.layers.0.mlp.experts.0.down_proj.weight"
        )
        is None
    )


def test_manifest_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": builder.HYBRID_SCHEMA,
                "complete": True,
            }
        )
    )
    manifest.chmod(0o600)
    with pytest.raises(builder.BuildError, match="manifest SHA-256 changed"):
        builder._load_manifest(manifest, "0" * 64, builder.HYBRID_SCHEMA)


def test_safetensors_header_and_raw_tensor_hash_are_exact(tmp_path: Path) -> None:
    path = tmp_path / "tiny.safetensors"
    save_file(
        {
            "one": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
            "two": torch.tensor([3.0], dtype=torch.float32),
        },
        path,
        metadata={"format": "pt"},
    )
    path.chmod(0o600)
    metadata, records = builder._read_safetensors_header(path)
    assert metadata == {"format": "pt"}
    assert records["one"].dtype == "BF16"
    assert records["one"].shape == (2, 4)
    assert records["two"].dtype == "F32"
    assert len(_tensor_hash(path, "one")) == 64


def test_subset_validation_rejects_tensor_hash_change(tmp_path: Path) -> None:
    path = tmp_path / "subset.safetensors"
    save_file({"scale": torch.tensor(1.0)}, path)
    path.chmod(0o600)
    manifest = {
        "source": {"repo": "owner/model", "revision": "a" * 40},
        "tensor_count": 1,
        "output_file": path.name,
        "output_bytes": path.stat().st_size,
        "output_sha256": builder._sha256(path),
        "tensor_sha256": {"scale": "0" * 64},
    }
    with pytest.raises(builder.BuildError, match="subset tensor SHA-256 changed"):
        builder._validate_subset_manifest(
            path=path,
            manifest=manifest,
            repo="owner/model",
            revision="a" * 40,
            expected_names=frozenset({"scale"}),
        )


def test_lm_head_lora_merge_is_chunkwise_and_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "VOCAB_SIZE", 7)
    monkeypatch.setattr(builder, "HIDDEN_SIZE", 4)
    base = torch.arange(28, dtype=torch.float32).reshape(7, 4).to(torch.bfloat16)
    a = torch.tensor([[0.25, 0.0, -0.25, 0.5], [0.0, 0.5, 0.25, 0.0]]).to(
        torch.bfloat16
    )
    b = (torch.arange(14, dtype=torch.float32).reshape(7, 2) / 100).to(torch.bfloat16)
    merged, relative = builder._merge_lm_head_chunkwise(
        base,
        a,
        b,
        alpha=0.5,
        maximum_relative_norm=0.05,
        chunk_rows=2,
    )
    expected = (base.float() + (b.float() @ a.float()) * 0.25).to(torch.bfloat16)
    assert torch.equal(merged, expected)
    assert 0 < relative < 0.05


def test_lm_head_lora_merge_enforces_movement_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "VOCAB_SIZE", 4)
    monkeypatch.setattr(builder, "HIDDEN_SIZE", 4)
    base = torch.ones((4, 4), dtype=torch.bfloat16)
    a = torch.ones((1, 4), dtype=torch.bfloat16)
    b = torch.ones((4, 1), dtype=torch.bfloat16)
    with pytest.raises(builder.BuildError, match="exceeds"):
        builder._merge_lm_head_chunkwise(
            base,
            a,
            b,
            alpha=1.0,
            maximum_relative_norm=0.01,
            chunk_rows=2,
        )


def test_synthetic_expert_quantization_emits_only_four_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "NUM_LAYERS", 1)
    monkeypatch.setattr(builder, "NUM_EXPERTS", 2)
    monkeypatch.setattr(builder, "HIDDEN_SIZE", 16)
    monkeypatch.setattr(builder, "EXPERT_INTERMEDIATE_SIZE", 16)
    gate_up = torch.ones((2, 32, 16), dtype=torch.bfloat16)
    down = torch.ones((2, 16, 16), dtype=torch.bfloat16)
    scales = builder.LayerScales(
        gate_up_input=torch.tensor([1.0, 2.0]),
        gate_up_weight_scale_2=torch.tensor([0.1, 0.2]),
        down_input=torch.tensor([3.0, 4.0]),
        down_weight_scale_2=torch.tensor([0.3, 0.4]),
    )

    def fake_backend(source: torch.Tensor, scale: torch.Tensor):
        packed = torch.zeros(
            (*source.shape[:-1], source.shape[-1] // 2), dtype=torch.uint8
        )
        blocks = torch.ones(
            (*source.shape[:-1], source.shape[-1] // 16),
            dtype=torch.float8_e4m3fn,
        )
        return packed, blocks, scale.clone().to(torch.float32)

    result = builder._quantize_expert_layer(gate_up, down, scales, 0, fake_backend)
    assert len(result) == 2 * 3 * 4
    assert all(
        name.startswith("model.language_model.layers.0.mlp.experts.") for name in result
    )
    for expert in range(2):
        gate = builder._expert_module(0, expert, "gate_proj")
        up = builder._expert_module(0, expert, "up_proj")
        assert torch.equal(result[gate + ".input_scale"], result[up + ".input_scale"])
        assert torch.equal(
            result[gate + ".weight_scale_2"], result[up + ".weight_scale_2"]
        )


class _FakeProperties:
    total_memory = 96 * 1024**3
    name = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"


class _FakeCuda:
    def __init__(self, count: int = 1) -> None:
        self.count = count
        self.fraction: float | None = None

    @staticmethod
    def is_available() -> bool:
        return True

    def device_count(self) -> int:
        return self.count

    @staticmethod
    def get_device_properties(_index: int) -> _FakeProperties:
        return _FakeProperties()

    @staticmethod
    def get_device_capability(_index: int) -> tuple[int, int]:
        return (12, 0)

    def set_per_process_memory_fraction(self, value: float, _index: int) -> None:
        self.fraction = value


class _FakeTorch:
    def __init__(self, count: int = 1) -> None:
        self.cuda = _FakeCuda(count)


def _fleet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES", "GPU-12345678-1234-1234-1234-123456789abc"
    )
    monkeypatch.setenv("GPU_AGENT_CLAIM_ID", "gc-abcdefgh")
    monkeypatch.setenv("AEON_QUANT_RUNTIME_ID", "fr-" + "a" * 32)
    monkeypatch.setenv("GPU_MEM_LIMIT_GB", "88")
    monkeypatch.setenv("GPU_RESERVE_GB", "6")


def test_fleet_binding_accepts_one_96gb_blackwell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fleet_env(monkeypatch)
    fake = _FakeTorch()
    receipt = builder._fleet_binding(fake)
    assert receipt["gpu_mem_limit_gb"] == 88
    assert receipt["gpu_reserve_gb"] == 6
    assert "claim_id" not in receipt and "gpu_uuid" not in receipt
    assert (
        receipt["claim_id_sha256"] == builder.hashlib.sha256(b"gc-abcdefgh").hexdigest()
    )
    assert (
        receipt["gpu_uuid_sha256"]
        == builder.hashlib.sha256(
            b"GPU-12345678-1234-1234-1234-123456789abc"
        ).hexdigest()
    )
    assert fake.cuda.fraction == pytest.approx(88 / 96)


def test_fleet_binding_rejects_more_than_one_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fleet_env(monkeypatch)
    with pytest.raises(builder.BuildError, match="one-GPU Fleet binding"):
        builder._fleet_binding(_FakeTorch(count=2))


def test_modelopt_config_has_one_closed_nvfp4_group() -> None:
    unified, hf = builder._modelopt_quant_configs()
    assert unified["quantization"] == {
        "exclude_modules": list(builder.MODELOPT_IGNORE),
        "group_size": 16,
        "quant_algo": "NVFP4",
    }
    assert hf["config_groups"]["group_0"]["targets"] == ["Linear"]
    assert hf["ignore"] == list(builder.MODELOPT_IGNORE)
    assert "mtp.*" in hf["ignore"]
    assert "model.visual.*" in hf["ignore"]
    assert "*.ple.*" in hf["ignore"]
    assert hf["quant_algo"] == "NVFP4"
    assert hf["quant_method"] == "modelopt"
    assert hf["producer"] == {"name": "modelopt", "version": "0.46.0"}


def test_sha256sums_excludes_itself_and_covers_every_final_file(tmp_path: Path) -> None:
    for name, content in (("config.json", b"{}\n"), ("tiny.bin", b"payload")):
        path = tmp_path / name
        path.write_bytes(content)
        path.chmod(0o600)
    sums = builder._write_sha256sums(tmp_path)
    assert set(sums) == {"config.json", "tiny.bin"}
    lines = (tmp_path / "SHA256SUMS").read_text().splitlines()
    assert lines == [f"{sums[name]}  {name}" for name in sorted(sums)]
    assert (
        hashlib.sha256((tmp_path / "tiny.bin").read_bytes()).hexdigest()
        == sums["tiny.bin"]
    )


def test_hybrid_manifest_copy_is_exact_fixed_name_and_receiptable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-manifest-with-arbitrary-name.json"
    source.write_bytes(b'{"complete":true}\n')
    source.chmod(0o600)
    partial = tmp_path / "model.partial"
    partial.mkdir(mode=0o700)

    copied = builder._copy_hybrid_manifest(
        source,
        partial,
        builder._sha256(source),
    )

    destination = partial / builder.HYBRID_MANIFEST_FILENAME
    assert copied == "HYBRID_MANIFEST.json"
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o600
    assert builder._write_sha256sums(partial) == {
        "HYBRID_MANIFEST.json": builder._sha256(source)
    }


def test_hybrid_manifest_copy_refuses_collision_or_digest_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b'{"complete":true}\n')
    source.chmod(0o600)
    partial = tmp_path / "model.partial"
    partial.mkdir(mode=0o700)

    with pytest.raises(builder.BuildError, match="identity changed"):
        builder._copy_hybrid_manifest(source, partial, "0" * 64)
    assert not (partial / builder.HYBRID_MANIFEST_FILENAME).exists()

    destination = partial / builder.HYBRID_MANIFEST_FILENAME
    destination.write_bytes(b"pre-existing")
    with pytest.raises(builder.BuildError, match="already exists"):
        builder._copy_hybrid_manifest(
            source,
            partial,
            builder._sha256(source),
        )


def test_untuned_sibling_is_truthful_and_counts_shared_storage_once(
    tmp_path: Path,
) -> None:
    tuned = tmp_path / "model.partial"
    untuned = tmp_path / "official-untuned-model.partial"
    tuned.mkdir()
    untuned.mkdir()
    common = tuned / "model-common.safetensors"
    common.write_bytes(b"shared-non-head-weights")
    hybrid_manifest = tuned / builder.HYBRID_MANIFEST_FILENAME
    hybrid_manifest.write_bytes(b'{"complete":true}\n')
    (tuned / builder.TUNED_LM_HEAD_FILENAME).write_bytes(b"tuned-head")
    (untuned / builder.UNTUNED_LM_HEAD_FILENAME).write_bytes(b"official-head")
    tuned_index = {
        "metadata": {"total_size": 123},
        "weight_map": {
            builder.LM_HEAD: builder.TUNED_LM_HEAD_FILENAME,
            "model.language_model.embed_tokens.weight": common.name,
        },
    }
    builder._write_json(tuned / "model.safetensors.index.json", tuned_index)
    builder._write_json(
        tuned / "BUILD_MANIFEST.json",
        {
            "status": "success",
            "sources": {
                "behavior_adapter": {
                    "target_modules": ["lm_head"],
                    "gate_status": "passed",
                }
            },
            "quantization": {"preserved": ["BF16 lm_head after bounded LoRA merge"]},
            "build": {"checkpoint_role": "tuned"},
        },
    )
    builder._write_json(
        tuned / "VALIDATION_REPORT.json",
        {
            "lm_head_lora_merged_before_quantization": True,
            "lm_head_lora_relative_frobenius_norm": 0.001,
            "runtime_validation_status": "pending-live-qualification",
        },
    )
    tuned_sums = builder._write_sha256sums(tuned)

    manifest, untuned_sums = builder._build_untuned_sibling(
        tuned_partial=tuned,
        untuned_partial=untuned,
        tuned_index=tuned_index,
        output_map=tuned_index["weight_map"],
        tuned_sums=tuned_sums,
        head_receipt={
            "tuned_tensor_sha256": "1" * 64,
            "official_untuned_tensor_sha256": "2" * 64,
        },
    )

    shared = untuned / common.name
    shared_hybrid_manifest = untuned / builder.HYBRID_MANIFEST_FILENAME
    assert (shared.stat().st_dev, shared.stat().st_ino) == (
        common.stat().st_dev,
        common.stat().st_ino,
    )
    assert (
        shared_hybrid_manifest.stat().st_dev,
        shared_hybrid_manifest.stat().st_ino,
    ) == (hybrid_manifest.stat().st_dev, hybrid_manifest.stat().st_ino)
    assert manifest["hardlink_identity"] == {
        "shared_regular_file_count": 2,
        "shared_unique_bytes": common.stat().st_size + hybrid_manifest.stat().st_size,
        "shared_paths_sha256": hashlib.sha256(
            builder._canonical_json(
                sorted([common.name, builder.HYBRID_MANIFEST_FILENAME])
            )
        ).hexdigest(),
        "same_device_and_inode": True,
        "rewritten_allowlist": [
            "BUILD_MANIFEST.json",
            "SHA256SUMS",
            "VALIDATION_REPORT.json",
            "model.safetensors.index.json",
            builder.TUNED_LM_HEAD_FILENAME,
            builder.UNTUNED_LM_HEAD_FILENAME,
        ],
    }
    assert manifest["hardlink_identity"]["shared_unique_bytes"] != (
        common.stat().st_size + shared.stat().st_size
    )
    untuned_build = json.loads((untuned / "BUILD_MANIFEST.json").read_text())
    assert untuned_build["build"]["checkpoint_role"] == "official_untuned"
    assert untuned_build["sources"]["behavior_adapter"] == {
        "target_modules": [],
        "gate_status": "not-applied-official-untuned-baseline",
    }
    untuned_validation = json.loads((untuned / "VALIDATION_REPORT.json").read_text())
    assert untuned_validation["lm_head_lora_merged_before_quantization"] is False
    assert untuned_validation["lm_head_lora_relative_frobenius_norm"] == 0.0
    untuned_index = json.loads((untuned / "model.safetensors.index.json").read_text())
    assert untuned_index["metadata"]["total_size"] == 123
    assert set(untuned_sums) == {
        "BUILD_MANIFEST.json",
        "VALIDATION_REPORT.json",
        "model.safetensors.index.json",
        builder.UNTUNED_LM_HEAD_FILENAME,
        builder.HYBRID_MANIFEST_FILENAME,
        common.name,
    }
