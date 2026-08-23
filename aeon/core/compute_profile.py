"""Typed admission requirements for Aeon's coordinator-managed GPU runtimes.

These profiles describe the host headroom required before a runtime may start.
They are deliberately independent of CUDA numbering. ComfyUI remains local;
Qwen placement may use any separately enabled host capability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeProfile:
    """Coordinator admission floors for one known Aeon runtime."""

    key: str
    min_host_memory_gb: float
    min_host_commit_gb: float
    min_disk_free_gb: float
    min_shm_free_gb: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.key, str)
            or not self.key
            or any(
                ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
                for ch in self.key
            )
        ):
            raise ValueError("compute profile key must be lowercase and filename-safe")
        for field_name in (
            "min_host_memory_gb",
            "min_host_commit_gb",
            "min_disk_free_gb",
            "min_shm_free_gb",
        ):
            raw = getattr(self, field_name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{field_name} must be a finite number")
            value = float(raw)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be positive")

    def coordinator_args(self) -> list[str]:
        """Return the exact resource filters understood by gpu_coord.py."""

        return [
            "--min-host-memory-gb", f"{self.min_host_memory_gb:g}",
            "--min-host-commit-gb", f"{self.min_host_commit_gb:g}",
            "--min-disk-free-gb", f"{self.min_disk_free_gb:g}",
            "--min-shm-free-gb", f"{self.min_shm_free_gb:g}",
        ]


# Qwen's NVFP4 load includes CPU-side checkpoint mapping and a capped four-way
# FlashInfer/CUDA JIT build. 96 GiB of currently available RAM and commit
# headroom covers the observed ~63 GiB load transient plus two independent 8 GiB
# tmpfs bounds (runtime/compiler cache and private shm), API/engine processes,
# and OS/renter headroom. The floors are intentionally limited to the roomy
# 96-GB-class host profile and the reviewed 48-GB compact candidate.
QWEN38_VLLM_PROFILE = ComputeProfile(
    key="qwen38-vllm",
    min_host_memory_gb=96,
    min_host_commit_gb=96,
    min_disk_free_gb=32,
    min_shm_free_gb=16,
)


# ComfyUI image/edit/video workflows can transiently hold decoded frames and
# model staging buffers in host memory. The container has an 8 GiB shm mount;
# the extra two GiB floor prevents admitting at the exact exhaustion boundary.
COMFYUI_PROFILE = ComputeProfile(
    key="comfyui",
    min_host_memory_gb=48,
    min_host_commit_gb=56,
    min_disk_free_gb=24,
    min_shm_free_gb=10,
)
