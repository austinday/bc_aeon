"""Synthetic regressions for the external Flash-Next pass-through proof."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from aeon.scripts import audit_qwen38_flash_next_passthrough as auditor


@dataclass(frozen=True)
class AuditFixture:
    hybrid: Path
    checkpoint: Path
    receipts: Path
    contract: auditor.AuditContract
    passthrough: dict[str, torch.Tensor]


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700)
    path.chmod(0o700)


def _save(root: Path, filename: str, tensors: dict[str, torch.Tensor]) -> Path:
    path = root / filename
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        path,
        metadata={"format": "pt"},
    )
    path.chmod(0o600)
    return path


def _tensor_bytes(tensors: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def _write_index(
    root: Path,
    weight_map: dict[str, str],
    tensors: dict[str, torch.Tensor],
    *,
    final: bool,
) -> None:
    total_size = (
        sum((root / filename).stat().st_size for filename in set(weight_map.values()))
        if final
        else _tensor_bytes(tensors)
    )
    payload = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    path = root / auditor.INDEX_FILENAME
    path.write_bytes(auditor._canonical_json(payload))
    path.chmod(0o600)


def _quantized_tensors(
    *, num_layers: int, num_experts: int, hidden: int, intermediate: int
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(num_layers):
        for expert in range(num_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                rows, columns = (
                    (intermediate, hidden)
                    if projection in {"gate_proj", "up_proj"}
                    else (hidden, intermediate)
                )
                prefix = (
                    f"model.language_model.layers.{layer}.mlp.experts.{expert}."
                    f"{projection}"
                )
                tensors[prefix + ".weight"] = torch.zeros(
                    (rows, columns // 2), dtype=torch.uint8
                )
                tensors[prefix + ".weight_scale"] = torch.ones(
                    (rows, columns // 16), dtype=torch.float8_e4m3fn
                )
                tensors[prefix + ".weight_scale_2"] = torch.tensor(
                    0.25, dtype=torch.float32
                )
                tensors[prefix + ".input_scale"] = torch.tensor(
                    0.5, dtype=torch.float32
                )
    return tensors


def _fixture(tmp_path: Path) -> AuditFixture:
    hybrid = tmp_path / "hybrid"
    checkpoint = tmp_path / "checkpoint"
    receipts = tmp_path / "receipts"
    for path in (hybrid, checkpoint, receipts):
        _private_directory(path)

    hidden = 16
    intermediate = 16
    vocab = 8
    mtp_names = frozenset({"mtp.test.weight"})
    passthrough = {
        "model.language_model.norm.weight": torch.tensor(
            [1.0, 2.0], dtype=torch.bfloat16
        ),
        "model.language_model.synthetic_buffer": torch.tensor([7], dtype=torch.int64),
        (
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
            "shard_0.weight"
        ): torch.tensor([1, 2, 3], dtype=torch.uint8),
        (
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
            "weight_scale"
        ): torch.tensor([0.5], dtype=torch.bfloat16),
        "model.visual.synthetic.weight": torch.tensor(
            [3.0, 4.0], dtype=torch.bfloat16
        ),
    }
    source_experts = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.zeros(
            (1, 2 * intermediate, hidden), dtype=torch.bfloat16
        ),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.zeros(
            (1, hidden, intermediate), dtype=torch.bfloat16
        ),
    }
    mtp = {"mtp.test.weight": torch.tensor([5.0, 6.0], dtype=torch.bfloat16)}
    source_head = {
        auditor.LM_HEAD: torch.zeros((vocab, hidden), dtype=torch.bfloat16)
    }
    source_tensors = {**passthrough, **source_experts, **mtp, **source_head}
    source_filename = "model-source.safetensors"
    _save(hybrid, source_filename, source_tensors)
    source_map = {name: source_filename for name in source_tensors}
    _write_index(hybrid, source_map, source_tensors, final=False)

    output_pass = {name: tensor.clone() for name, tensor in passthrough.items()}
    output_mtp = {name: tensor.clone() for name, tensor in mtp.items()}
    output_head = {
        auditor.LM_HEAD: torch.ones((vocab, hidden), dtype=torch.bfloat16)
    }
    quantized = _quantized_tensors(
        num_layers=1,
        num_experts=1,
        hidden=hidden,
        intermediate=intermediate,
    )
    _save(checkpoint, source_filename, output_pass)
    _save(checkpoint, "model-mtp-bf16.safetensors", output_mtp)
    _save(checkpoint, "model-lm-head-bf16.safetensors", output_head)
    quant_filename = "model-routed-experts-00001-of-00001.safetensors"
    _save(checkpoint, quant_filename, quantized)
    output_map = {name: source_filename for name in output_pass}
    output_map.update({name: "model-mtp-bf16.safetensors" for name in output_mtp})
    output_map[auditor.LM_HEAD] = "model-lm-head-bf16.safetensors"
    output_map.update({name: quant_filename for name in quantized})
    output_tensors = {**output_pass, **output_mtp, **output_head, **quantized}
    _write_index(checkpoint, output_map, output_tensors, final=True)

    passthrough_names = sorted(passthrough)
    dtype_counts: dict[str, int] = {}
    dtype_bytes: dict[str, int] = {}
    torch_to_safe = {
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.uint8: "U8",
    }
    for tensor in passthrough.values():
        dtype = torch_to_safe[tensor.dtype]
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + tensor.numel() * tensor.element_size()
    contract = auditor.AuditContract(
        name="synthetic-qwen-flash-next",
        num_layers=1,
        num_experts=1,
        hidden_size=hidden,
        expert_intermediate_size=intermediate,
        vocab_size=vocab,
        mtp_names=mtp_names,
        source_tensor_count=len(source_tensors),
        output_tensor_count=len(output_tensors),
        source_tensor_bytes=_tensor_bytes(source_tensors),
        output_tensor_bytes=_tensor_bytes(output_tensors),
        passthrough_tensor_count=len(passthrough),
        passthrough_tensor_bytes=_tensor_bytes(passthrough),
        passthrough_name_set_sha256=auditor._sha256_bytes(
            auditor._canonical_json(passthrough_names)
        ),
        passthrough_category_counts={"other": 2, "ple": 2, "vision": 1},
        passthrough_dtype_counts=dtype_counts,
        passthrough_dtype_bytes=dtype_bytes,
    )
    return AuditFixture(hybrid, checkpoint, receipts, contract, passthrough)


def _replace_tensor(path: Path, name: str, replacement: torch.Tensor) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {tensor_name: handle.get_tensor(tensor_name) for tensor_name in handle.keys()}
    tensors[name] = replacement
    path.unlink()
    save_file(tensors, path, metadata={"format": "pt"})
    path.chmod(0o600)


def test_external_audit_writes_closed_private_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt_path = fixture.receipts / "PASSTHROUGH_AUDIT.json"

    result = auditor.audit_passthrough(
        fixture.hybrid,
        fixture.checkpoint,
        receipt_path,
        contract=fixture.contract,
    )

    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted == result
    assert result["schema_version"] == auditor.SCHEMA_VERSION
    assert result["complete"] is True
    assert result["passed"] is True
    assert result["passthrough"]["exact_raw_payload_identity"] is True
    assert result["passthrough"]["tensor_count"] == 5
    assert result["contract"]["sha256"] == auditor._sha256_bytes(
        auditor._canonical_json(result["contract"]["details"])
    )
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600

    with pytest.raises(auditor.PassthroughAuditError, match="already exists"):
        auditor.audit_passthrough(
            fixture.hybrid,
            fixture.checkpoint,
            receipt_path,
            contract=fixture.contract,
        )


def test_external_audit_rejects_same_shape_payload_mutation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    name = "model.visual.synthetic.weight"
    mutated = fixture.passthrough[name].clone()
    mutated[0] = 99
    _replace_tensor(fixture.checkpoint / "model-source.safetensors", name, mutated)
    # File size and every index/topology field remain valid; only raw tensor bytes differ.

    with pytest.raises(auditor.PassthroughAuditError, match="payload changed"):
        auditor.audit_passthrough(
            fixture.hybrid,
            fixture.checkpoint,
            fixture.receipts / "mutated.json",
            contract=fixture.contract,
        )


def test_external_audit_rejects_unindexed_safetensors(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _save(
        fixture.checkpoint,
        "unexpected.safetensors",
        {"unexpected.weight": torch.ones(1, dtype=torch.bfloat16)},
    )

    with pytest.raises(auditor.PassthroughAuditError, match="on-disk safetensors set"):
        auditor.audit_passthrough(
            fixture.hybrid,
            fixture.checkpoint,
            fixture.receipts / "extra.json",
            contract=fixture.contract,
        )


def test_external_audit_enforces_canonical_name_set_digest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    wrong_contract = replace(
        fixture.contract,
        passthrough_name_set_sha256="0" * 64,
    )

    with pytest.raises(auditor.PassthroughAuditError, match="name set changed"):
        auditor.audit_passthrough(
            fixture.hybrid,
            fixture.checkpoint,
            fixture.receipts / "wrong-names.json",
            contract=wrong_contract,
        )


def test_external_audit_never_writes_inside_inputs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(auditor.PassthroughAuditError, match="outside the final checkpoint"):
        auditor.audit_passthrough(
            fixture.hybrid,
            fixture.checkpoint,
            fixture.checkpoint / "audit.json",
            contract=fixture.contract,
        )
