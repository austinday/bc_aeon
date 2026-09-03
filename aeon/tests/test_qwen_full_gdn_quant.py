from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from aeon.core import qwen_full_gdn_quant_adapter as adapter
from aeon.scripts import build_qwen38_full_gdn_nvfp4 as builder
from aeon.scripts import extract_qwen38_modelopt_template as extractor
from aeon.scripts import qwen_full_gdn_quant_worker as worker


def test_quantization_payload_is_closed() -> None:
    assert adapter.AeonQwenFullGDNQuantAdapter._payload({}) == {
        "recipe": "full-gdn-max-v1"
    }
    assert adapter.AeonQwenFullGDNQuantAdapter._payload(
        {"recipe": "full-gdn-max-v1"}
    ) == {"recipe": "full-gdn-max-v1"}
    with pytest.raises(adapter.QwenFullGDNQuantError):
        adapter.AeonQwenFullGDNQuantAdapter._payload({"recipe": "other"})
    with pytest.raises(adapter.QwenFullGDNQuantError):
        adapter.AeonQwenFullGDNQuantAdapter._payload({"extra": True})


def test_profile_artifact_identity_binds_every_external_input() -> None:
    sources = adapter._source_manifest()
    identity = adapter._expected_artifact_identity(sources)
    assert identity == {
        "modelopt_wheel": worker.MODELOPT_WHEEL_SHA256,
        "source_manifest": adapter._canonical_sha256(sources),
        "source_tree": adapter._canonical_sha256(
            {
                "weights": worker.SOURCE_WEIGHT_SHA256,
                "metadata": worker.SOURCE_METADATA_SHA256,
            }
        ),
        "template_config": worker.TEMPLATE_CONFIG_SHA256,
        "template_scales": worker.TEMPLATE_SCALES_SHA256,
    }


@pytest.mark.parametrize(
    ("module", "suffix"),
    [
        ("model.layers.0.linear_attn.in_proj_qkv", ".in_proj_qkvz"),
        ("model.layers.0.linear_attn.in_proj_z", ".in_proj_qkvz"),
        ("model.layers.3.self_attn.q_proj", ".qkv_proj"),
        ("model.layers.3.self_attn.k_proj", ".qkv_proj"),
        ("model.layers.0.mlp.gate_proj", ".gate_up_proj"),
        ("model.layers.0.mlp.up_proj", ".gate_up_proj"),
        ("model.layers.0.mlp.down_proj", ".down_proj"),
    ],
)
def test_fused_group_layout(module: str, suffix: str) -> None:
    assert builder._fused_group(module).endswith(suffix)


def test_group_builder_refuses_cross_shard_fusion() -> None:
    layers = {
        "model.layers.0.mlp.gate_proj": {"quant_algo": "NVFP4"},
        "model.layers.0.mlp.up_proj": {"quant_algo": "NVFP4"},
    }
    weight_map = {
        "model.layers.0.mlp.gate_proj.weight": "one.safetensors",
        "model.layers.0.mlp.up_proj.weight": "two.safetensors",
    }
    with pytest.raises(builder.QuantizationError, match="crosses source shards"):
        builder._groups(layers, weight_map)


def test_template_float_decoder_handles_scalar() -> None:
    value = extractor._decode_f32(b"\x00\x00\x80?", [])
    assert value.dtype == torch.float32
    assert value.shape == torch.Size([])
    assert value.item() == 1.0


def test_probe_retries_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = adapter.AeonQwenFullGDNQuantAdapter()
    monkeypatch.setattr(
        instance,
        "_runtime_identity",
        lambda _runtime: ("fr-" + "a" * 32, "b" * 64, 12345),
    )

    def unavailable(*_args, **_kwargs):
        raise adapter.QwenFullGDNQuantTransportError("link unavailable")

    monkeypatch.setattr(instance, "_runtime_action", unavailable)
    with pytest.raises(adapter.QwenFullGDNQuantTransportError, match="link unavailable"):
        instance.probe({})


def test_process_exit_between_proc_reads_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raced_read(_path: Path) -> bytes:
        raise OSError("procfs entry disappeared")

    monkeypatch.setattr(Path, "read_bytes", raced_read)
    monkeypatch.setattr(Path, "exists", lambda _path: False)
    assert worker._process_alive({}, 12345) is False


def test_source_verifier_accepts_exact_empty_package_marker(tmp_path: Path) -> None:
    marker = tmp_path / "__init__.py"
    marker.touch(mode=0o600)
    empty_sha256 = hashlib.sha256(b"").hexdigest()

    assert worker._verify_regular(
        marker,
        empty_sha256,
        maximum=2 * 1024 * 1024,
        allow_empty=True,
    ) == 0
    with pytest.raises(worker.QuantWorkerError, match="staged file identity changed"):
        worker._verify_regular(marker, empty_sha256, maximum=2 * 1024 * 1024)
