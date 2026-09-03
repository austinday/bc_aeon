from __future__ import annotations

import copy

import pytest
import torch

from aeon.core import dflash_adaptation as adaptation
from aeon.core.dflash_dpace import dpace_weighted_loss, dpace_weights


def test_reviewed_parameter_selection() -> None:
    selected = {
        "fc.weight",
        "candidate_selector.hidden_projection.weight",
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
        "layers.0.attention_conv.kernel_projection.weight",
        "layers.0.mlp_conv.base_kernel",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "norm.weight",
    }
    rejected = {
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.mlp.up_proj.weight",
    }
    assert all(adaptation.trainable_parameter(name) for name in selected)
    assert not any(adaptation.trainable_parameter(name) for name in rejected)
    assert adaptation.FULL_ADAPTATION_MODE in adaptation.ADAPTATION_MODES


def test_exact_draft_config_validation() -> None:
    config = copy.deepcopy(adaptation.EXPECTED_DRAFT_CONFIG)
    adaptation.validate_draft_config(config, label="test")
    config["dflash_config"]["target_layer_ids"] = [6, 20, 34, 48, 62]
    with pytest.raises(RuntimeError, match="reviewed architecture"):
        adaptation.validate_draft_config(config, label="test")


def test_critical_config_ignores_serving_architecture_metadata() -> None:
    config = copy.deepcopy(adaptation.EXPECTED_DRAFT_CONFIG)
    config["architectures"] = ["DFlash2DraftModel"]
    assert adaptation.critical_draft_config(config) == adaptation.EXPECTED_DRAFT_CONFIG


def test_dpace_weights_match_reference_and_reset_per_block() -> None:
    nll = torch.tensor([[[0.2, 0.7, 1.1], [0.4, 0.9, 0.3]]])
    mask = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]])
    alpha = 0.5
    smooth = torch.where(
        mask.bool(),
        (1.0 - alpha) * torch.exp(-nll) + alpha,
        torch.ones_like(nll),
    )
    prefix = torch.cumprod(smooth, dim=-1)
    expected = torch.flip(
        torch.cumsum(torch.flip(prefix * mask, dims=[-1]), dim=-1),
        dims=[-1],
    )
    assert torch.allclose(dpace_weights(nll, mask, alpha=alpha), expected)


def test_dpace_selector_mask_keeps_missing_candidate_in_confidence_path() -> None:
    nll = torch.tensor([[[0.2, 100.0, 0.4]]], requires_grad=True)
    valid = torch.ones_like(nll)
    supervised = torch.tensor([[[1.0, 0.0, 1.0]]])
    loss, denominator = dpace_weighted_loss(
        nll,
        valid,
        alpha=0.5,
        supervised_mask=supervised,
    )
    loss.backward()
    assert denominator.item() == 1.0
    assert nll.grad is not None
    assert nll.grad[0, 0, 1].item() == 0.0
    assert 0.0 < nll.grad[0, 0, 2].item() < nll.grad[0, 0, 0].item()
