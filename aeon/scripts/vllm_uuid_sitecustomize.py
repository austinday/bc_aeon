"""Teach vLLM 0.23 to honor UUID-valued CUDA_VISIBLE_DEVICES.

vLLM 0.23 assumes every CUDA selector is a numeric physical-device index.  The
fleet coordinator deliberately returns stable GPU UUIDs instead.  Install this
module as ``sitecustomize.py`` on ``PYTHONPATH`` for the vLLM process; it keeps
the UUID selector intact and resolves only vLLM's NVML bookkeeping index from
that exact UUID.
"""
from __future__ import annotations

import math
import os


def _install() -> None:
    from vllm.platforms.interface import Platform

    original = Platform.device_id_to_physical_device_id.__func__

    @classmethod
    def device_id_to_physical_device_id(cls: type[Platform], device_id: int) -> int:
        visible = os.environ.get(cls.device_control_env_var, "")
        if not visible:
            return device_id
        selectors = visible.split(",")
        selector = selectors[device_id]
        if not selector.startswith("GPU-"):
            return original(cls, device_id)

        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByUUID(selector)
            return int(pynvml.nvmlDeviceGetIndex(handle))
        finally:
            pynvml.nvmlShutdown()

    Platform.device_id_to_physical_device_id = device_id_to_physical_device_id

    claim = os.environ.get("GPU_AGENT_CLAIM_ID", "")
    if not claim.startswith("gc-"):
        raise RuntimeError("vLLM requires an active coordinator claim")
    try:
        planned_gb = float(os.environ["GPU_PLANNED_VRAM_GB"])
        reserve_gb = float(os.environ.get("GPU_RESERVE_GB", "6"))
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            "GPU_PLANNED_VRAM_GB and GPU_RESERVE_GB must be numeric"
        ) from exc
    if (
        not math.isfinite(planned_gb)
        or not math.isfinite(reserve_gb)
        or planned_gb <= 0
        or reserve_gb < 6
        or os.environ.get("GPU_LEASE_EXCLUSIVE") != "1"
    ):
        raise RuntimeError("Invalid exclusive vLLM lease plan or renter reserve")
    # Do not initialize CUDA or install a per-process PyTorch allocator fraction
    # here. vLLM V1 is multiprocess and that fraction is neither aggregate nor a
    # direct-CUDA allocation cap. This runtime uses the fleet policy's explicit
    # exclusive-lease + >=6 GiB reserve exception; gpu-memory-utilization remains
    # a measured allocation plan, not a cgroup-style hard VRAM cap.


_install()
