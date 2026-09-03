"""Numerically stable D-PACE weights for exact-target DFlash2 training."""

from __future__ import annotations

import math

import torch


def dpace_weights(
    token_nll: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Return detached D-PACE weights for ``[batch, blocks, depth]`` NLLs.

    The cumulative confidence and continuation-value sum reset structurally at
    every block. Invalid padding positions are neutral in the confidence
    product and absent from the continuation sum.
    """
    if token_nll.ndim != 3 or token_nll.shape != valid_mask.shape:
        raise ValueError("D-PACE tensors must share [batch, blocks, depth] shape")
    if not token_nll.is_floating_point():
        raise ValueError("D-PACE token NLL must be floating point")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("D-PACE alpha must be finite and in [0, 1]")

    with torch.no_grad():
        nll = token_nll.detach().float()
        mask_bool = valid_mask.bool()
        mask = mask_bool.to(nll.dtype)
        smooth = (1.0 - alpha) * torch.exp(-nll) + alpha
        smooth = torch.where(mask_bool, smooth, torch.ones_like(smooth))
        cumulative_confidence = torch.cumprod(smooth, dim=-1)
        continuation_value = torch.flip(
            torch.cumsum(
                torch.flip(cumulative_confidence * mask, dims=[-1]),
                dim=-1,
            ),
            dims=[-1],
        )
    return continuation_value.to(token_nll.dtype)


def dpace_weighted_loss(
    token_nll: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    alpha: float,
    supervised_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the D-PACE weighted sum with a data-independent denominator.

    ``valid_mask`` controls the accepted-prefix confidence path. An optional
    ``supervised_mask`` can remove positions from the differentiable loss while
    retaining their confidence effect, which DFlash2 needs when a ground-truth
    token is absent from the selector's top-k candidate set.
    """
    weights = dpace_weights(token_nll, valid_mask, alpha=alpha)
    effective_mask = valid_mask if supervised_mask is None else supervised_mask
    if effective_mask.shape != token_nll.shape:
        raise ValueError("D-PACE supervised mask shape changed")
    denominator = token_nll.new_tensor(
        max(float(token_nll.shape[0] * token_nll.shape[1]), 1.0)
    )
    loss = (
        token_nll * weights * effective_mask.to(token_nll.dtype)
    ).sum() / denominator
    return loss, denominator.detach()
