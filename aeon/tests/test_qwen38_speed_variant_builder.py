from __future__ import annotations

import json
import math
import re

import torch

from aeon.scripts import build_qwen38_speed_variant as builder


def _unpack_int8(packed: torch.Tensor) -> torch.Tensor:
    values = []
    for shift in (0, 8, 16, 24):
        item = ((packed.to(torch.int64) >> shift) & 0xFF).to(torch.int16)
        values.append((item - 128).to(torch.int8))
    return torch.stack(values, dim=-1).reshape(packed.shape[0], -1)


def test_int8_pack_round_trips_signed_values():
    values = torch.tensor(
        [[-127, -64, -1, 0, 1, 63, 126, 127]], dtype=torch.int8
    )
    assert torch.equal(_unpack_int8(builder._pack_int8(values)), values)


def test_group_quantization_is_bounded_and_reconstructable():
    torch.manual_seed(7)
    source = torch.randn(9, 256, dtype=torch.bfloat16)
    components, error = builder._quantize_int8(source, row_chunk=4)
    unpacked = _unpack_int8(components["weight_packed"]).to(torch.float32)
    scales = components["weight_scale"].to(torch.float32).repeat_interleave(
        builder.HEAD_GROUP_SIZE, dim=1
    )
    restored = unpacked * scales
    measured = float((restored - source.float()).norm() / source.float().norm())
    assert components["weight_shape"].tolist() == [9, 256]
    assert math.isclose(error, measured, rel_tol=0.02)
    assert measured < 0.01


def test_mixed_precision_group_is_explicit_pack_quantized_int8():
    group = builder._int8_group(["lm_head"])
    assert group["format"] == "pack-quantized"
    assert group["targets"] == ["lm_head"]
    assert group["input_activations"] is None
    assert group["weights"]["type"] == "int"
    assert group["weights"]["num_bits"] == 8
    assert group["weights"]["group_size"] == 128


def test_mtp_group_uses_finer_int8_scales():
    group = builder._int8_group(
        builder.MTP_LINEAR_MODULES, group_size=builder.MTP_GROUP_SIZE
    )
    assert group["weights"]["group_size"] == 32


def test_runtime_targets_match_wrapped_and_speculative_module_paths():
    head = builder._runtime_suffix_target("lm_head")[3:]
    draft_head = builder._runtime_suffix_target("draft_lm_head")[3:]
    embedding = builder._runtime_suffix_target("embed_tokens")[3:]
    mtp = builder._runtime_suffix_target("mtp.layers.0.mlp.down_proj")[3:]
    assert re.match(head, "language_model.lm_head")
    assert re.match(draft_head, "mtp.draft_lm_head")
    assert re.match(embedding, "language_model.model.embed_tokens")
    assert re.match(embedding, "mtp.embed_tokens")
    assert re.match(mtp, "mtp.layers.0.mlp.down_proj")


def test_specific_speed_groups_precede_generic_linear_scheme():
    groups = builder._ordered_quant_groups(
        {"group_0": {"targets": ["Linear"], "weights": {}}}
    )
    assert list(groups) == [
        "group_speed_output_heads",
        "group_speed_embeddings",
        "group_speed_mtp",
        "group_0",
    ]


def test_config_writer_preserves_executable_group_precedence(tmp_path):
    path = tmp_path / "config.json"
    groups = builder._ordered_quant_groups(
        {"group_0": {"targets": ["Linear"], "weights": {}}}
    )
    builder._write_json(
        path,
        {"quantization_config": {"config_groups": groups}},
        sort_keys=False,
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert list(loaded["quantization_config"]["config_groups"]) == list(groups)
