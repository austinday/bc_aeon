"""Enforce Aeon's coordinator lease as a PyTorch allocator hard cap."""
from __future__ import annotations

import os


def _install_limit() -> None:
    claim = os.environ.get("GPU_AGENT_CLAIM_ID", "")
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not claim.startswith("gc-") or not selector.startswith("GPU-"):
        raise RuntimeError("ComfyUI requires a coordinator claim and UUID CUDA selector")

    try:
        limit_gb = float(os.environ["GPU_MEM_LIMIT_GB"])
        reserve_gb = float(os.environ.get("GPU_RESERVE_GB", "6"))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("GPU_MEM_LIMIT_GB and GPU_RESERVE_GB must be numeric") from exc
    if limit_gb <= 0 or reserve_gb < 6:
        raise RuntimeError("GPU_MEM_LIMIT_GB must be positive and GPU_RESERVE_GB at least 6")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ComfyUI must see exactly its one coordinator-leased GPU")
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if limit_gb + reserve_gb > total_gb + 0.01:
        raise RuntimeError(
            f"Lease cap {limit_gb:g}GB plus {reserve_gb:g}GB reserve exceeds "
            f"the {total_gb:.1f}GB GPU"
        )
    torch.cuda.set_per_process_memory_fraction(limit_gb / total_gb, 0)
    print(
        f"Aeon GPU lease enforced: claim={claim} selector={selector} "
        f"hard_cap={limit_gb:g}GB reserve={reserve_gb:g}GB",
        flush=True,
    )


_install_limit()
