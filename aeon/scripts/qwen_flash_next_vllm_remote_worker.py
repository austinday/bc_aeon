#!/usr/bin/env python3
"""Remote .179 GPU1 entry point for the exact vLLM qualification worker.

The qualification implementation remains single-source.  This wrapper changes
only the Fleet-reviewed host, GPU, port, run-root, and image-archive-root
bindings before dispatch.  It is copied into the attempt-owned source closure
and is safe to invoke again when the worker spawns its supervisor process.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import sys


PROFILE_ID = "aeon-qwen38-flash-next-vllm-canary-179-gpu1"
HOST = "192.168.0.179"
HOSTNAME = "DAY2XRTX6000-2"
PHYSICAL_GPU = 1
HOST_PORT = 18059
RUN_ROOT = PurePosixPath("/home/aday/.local/state/fleet-compute/runs")


def _source_root() -> Path:
    path = Path(__file__).resolve(strict=True)
    # <run>/source/aeon/scripts/this_file.py
    root = path.parents[2]
    if root.name != "source" or root.parent.parent != Path(RUN_ROOT):
        raise RuntimeError("remote canary wrapper escaped its Fleet run source")
    return root


def main() -> int:
    source = _source_root()
    sys.path.insert(0, str(source))

    from aeon.core import qwen_flash_next_vllm_contract as contract
    from aeon.scripts import qwen_flash_next_vllm_canary_worker as worker

    contract.PROFILE_ID = PROFILE_ID
    contract.HOST = HOST
    contract.PHYSICAL_GPU = PHYSICAL_GPU
    worker.HOST = HOST
    worker.HOSTNAME = HOSTNAME
    worker.PHYSICAL_GPU = PHYSICAL_GPU
    worker.HOST_PORT = HOST_PORT
    worker.CANONICAL_OUTPUT_ROOT = RUN_ROOT
    worker.IMAGE_ARCHIVE_ROOT = RUN_ROOT / Path(__file__).resolve().parents[3].name / "runtime-images"
    # The shared worker re-execs its own module for supervision.  Bind that
    # exact operation back to this wrapper so the remote constants are applied
    # in the child as well.
    worker.__file__ = str(Path(__file__).resolve(strict=True))
    return worker.main()


if __name__ == "__main__":
    raise SystemExit(main())
