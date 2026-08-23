"""Coordinator-backed GPU discovery for renter-safe deployment planning."""
from __future__ import annotations

import json
import socket
import subprocess
from dataclasses import asdict, dataclass
from typing import List

MIB_PER_GIB = 1024.0
COORD = "/home/aday/website_hosting/gpu_coord.py"
COORD_DIR = "/home/aday/website_hosting/ads"
LOCAL_HOST = "192.168.0.177"
LOCAL_HOSTNAME = "DAY2RTX6000PRO"


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_gib: float
    free_gib: float


def detect_gpus(timeout: int = 15) -> List[GpuInfo]:
    """Return only coordinator-approved, ACL-open GPUs on the local host."""
    if socket.gethostname() != LOCAL_HOSTNAME:
        return []
    try:
        result = subprocess.run(
            ["python3", COORD, "status", "--json"], cwd=COORD_DIR,
            capture_output=True, text=True, timeout=timeout, check=True,
        )
        inventory = json.loads(result.stdout)
    except (subprocess.SubprocessError, OSError, ValueError):
        return []
    allowed = {
        "AVAILABLE", "SHARED_AVAILABLE", "RESERVED", "RESERVED_RUNNING",
        "RESERVED_STALE",
    }
    gpus = [
        GpuInfo(
            index=int(item["physical_gpu"]),
            name=str(item.get("model") or "coordinator GPU"),
            total_gib=round(float(item["memory_total_mib"]) / MIB_PER_GIB, 2),
            free_gib=round(float(item.get("vram_share_capacity_mib") or 0) / MIB_PER_GIB, 2),
        )
        for item in inventory
        if item.get("host") == LOCAL_HOST
        and item.get("acl") == "OPEN"
        and item.get("state") in allowed
        and item.get("memory_total_mib") is not None
    ]
    return sorted(gpus, key=lambda gpu: gpu.index)


def min_total_vram_gib(gpus: List[GpuInfo]) -> float:
    return min((gpu.total_gib for gpu in gpus), default=0.0)


def num_gpus(gpus: List[GpuInfo]) -> int:
    return len(gpus)


def _main() -> None:
    gpus = detect_gpus()
    print(json.dumps({
        "count": len(gpus),
        "min_total_gib": min_total_vram_gib(gpus),
        "gpus": [asdict(gpu) for gpu in gpus],
    }))


if __name__ == "__main__":
    _main()
