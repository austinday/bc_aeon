"""GPU detection for portable, VRAM-adaptive model deployment.

Single detection path (nvidia-smi) shared by the Python runtime, the deploy
planner, and setup_environment.sh (via the JSON CLI: `python -m aeon.core.gpu`).

The harness only targets symmetric dual-Blackwell machines (2x RTX 5000 = 48 GB
each, or 2x RTX 6000 PRO = 96 GB each), so callers generally use min() VRAM
across the detected GPUs defensively rather than assuming all are identical.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from typing import List

MIB_PER_GIB = 1024.0


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_gib: float
    free_gib: float


def detect_gpus(timeout: int = 15) -> List[GpuInfo]:
    """Return per-GPU info via nvidia-smi. Empty list if no GPUs / no driver."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=timeout,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []

    gpus: List[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            total = float(parts[2]) / MIB_PER_GIB
            free = float(parts[3]) / MIB_PER_GIB
        except ValueError:
            continue
        gpus.append(GpuInfo(index=idx, name=name, total_gib=round(total, 2), free_gib=round(free, 2)))
    gpus.sort(key=lambda g: g.index)
    return gpus


def min_total_vram_gib(gpus: List[GpuInfo]) -> float:
    """Smallest per-GPU total VRAM, the safe planning denominator."""
    return min((g.total_gib for g in gpus), default=0.0)


def num_gpus(gpus: List[GpuInfo]) -> int:
    return len(gpus)


def _main() -> None:
    gpus = detect_gpus()
    print(json.dumps({
        "count": len(gpus),
        "min_total_gib": min_total_vram_gib(gpus),
        "gpus": [asdict(g) for g in gpus],
    }))


if __name__ == "__main__":
    _main()
