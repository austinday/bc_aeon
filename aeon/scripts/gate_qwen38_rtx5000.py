#!/usr/bin/env python3
"""Launch or retire the exact disabled RTX 5000 Qwen release candidate.

This is an operator-driven release gate, not a placement path. Normal Aeon
sessions cannot select the candidate until its immutable capability is enabled
with a release receipt after testing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeon.core.compute_profile import QWEN38_VLLM_PROFILE
from aeon.core.gpu_queue import (
    PeriodicLeaseHeartbeat,
    QWEN_LEASE_FILE,
    current_lease,
    heartbeat_vram,
    reserve_named_lease,
)
from aeon.core.model_catalog import QWEN38_SERVED_NAME
from aeon.core.qwen_capabilities import (
    RTX5000_RELEASE_CANDIDATE_KEY,
    qwen_release_candidate_capability,
)
from aeon.core.qwen_fleet_runtime import (
    capability_deploy_environment,
    remote_preflight,
    remote_state,
    start_managed_remote_runtime,
    stop_managed_remote_runtime,
)
from aeon.core.qwen_runtime import RUNTIME_ROOT, QwenRuntimeError


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "aeon_qwen38_rtx5000_release_gate"
REMOTE_PORT = 8033


def _base_environment(image_id: str) -> dict[str, str]:
    plan = {
        "tier": "solo",
        "image": image_id,
        "entry_name": QWEN38_SERVED_NAME,
        "container_name": CONTAINER_NAME,
        "health_port": REMOTE_PORT,
        "context_limit": 114688,
        "nodes": [
            {
                "role": "node",
                "devices": "0",
                "port": REMOTE_PORT,
                "ctx": 114688,
                "cpu_offload_gib": 0.0,
                "container": CONTAINER_NAME,
            }
        ],
    }
    return {
        "AEON_DEPLOY_PLAN": json.dumps(
            plan, sort_keys=True, separators=(",", ":")
        ),
        "AEON_RUNTIME_IMAGE_REF": image_id,
        "AEON_SERVED_NAME": QWEN38_SERVED_NAME,
        "AEON_MTP_METHOD": "mtp",
        "AEON_MTP_NMAX": "3",
        "AEON_MTP_SELECTION_MANIFEST": "data/qwen38_mtp_selection.json",
        "AEON_KV_QUANT": "fp8_per_token_head",
        "AEON_VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
    }


def start() -> int:
    capability, manifest_sha256 = qwen_release_candidate_capability(
        RTX5000_RELEASE_CANDIDATE_KEY
    )
    if remote_state() is not None or current_lease(QWEN_LEASE_FILE) is not None:
        raise QwenRuntimeError("an Aeon Qwen lease/runtime already exists")
    source, preflight = remote_preflight(
        capability, manifest_sha256, PACKAGE_ROOT
    )
    if preflight.get("image_id") != capability.image_id:
        raise QwenRuntimeError("release-candidate image identity changed")

    lease = None
    unavailable: list[int] = []
    for physical_gpu in capability.allowed_physical_gpus:
        try:
            lease = reserve_named_lease(
                required_gb=float(capability.vram_budget_gb),
                purpose=(
                    "Aeon Qwen3.8 RTX 5000 128k release gate: memory, quality, "
                    "serial, batch, MTP, and multimodal validation"
                ),
                state_file=QWEN_LEASE_FILE,
                profile=QWEN38_VLLM_PROFILE,
                timeout=0,
                gpu_id=physical_gpu,
                host=capability.host,
                min_vram_gb=capability.min_physical_vram_gb,
                run_dir_root=RUNTIME_ROOT,
                exclusive=True,
                release_gate_capability_key=capability.key,
            )
            break
        except TimeoutError:
            unavailable.append(physical_gpu)
    if lease is None:
        raise QwenRuntimeError(
            f"no coordinator-safe candidate GPU is available: {unavailable}"
        )

    environment = capability_deploy_environment(
        capability, _base_environment(str(capability.image_id)), lease
    )
    heartbeat = PeriodicLeaseHeartbeat(
        state_file=QWEN_LEASE_FILE,
        note="Aeon RTX 5000 128k release candidate is starting",
        interval_seconds=240,
        require_pid=False,
        promote_when_pid_available=True,
    )
    heartbeat.start(immediate=True)
    try:
        state = start_managed_remote_runtime(
            capability,
            manifest_sha256,
            source,
            lease,
            environment,
            container_name=CONTAINER_NAME,
            port=REMOTE_PORT,
            heartbeat_pid=lambda pid: heartbeat.promote_to_exact_pid(
                lambda: int(pid)
            ),
            progress_check=heartbeat.raise_if_failed,
        )
        heartbeat.raise_if_failed()
        heartbeat_vram(
            int(state["container_pid"]),
            "Aeon RTX 5000 128k release candidate is ready for gates",
            QWEN_LEASE_FILE,
        )
    finally:
        heartbeat.stop()
    print(
        json.dumps(
            {
                "state": "ready",
                "base_url": "http://127.0.0.1:8033/v1",
                "capability": capability.key,
                "context_tokens": capability.context_tokens,
                "claim_id": lease["claim_id"],
                "host": lease["host"],
                "physical_gpu": lease["physical_gpu"],
                "gpu_uuid": lease["gpu_uuid"],
                "container_pid": state["container_pid"],
            },
            sort_keys=True,
        )
    )
    return 0


def stop(reason: str) -> int:
    state = remote_state()
    if state is None:
        raise QwenRuntimeError("no managed remote Qwen runtime exists")
    capability, _current_manifest = qwen_release_candidate_capability(
        str(state["runtime_capability_key"])
    )
    stopped = stop_managed_remote_runtime(
        capability,
        str(state["runtime_capability_manifest_sha256"]),
        str(state["source_manifest_sha256"]),
        release_reason=reason,
    )
    if not stopped:
        raise QwenRuntimeError("exact release-candidate teardown is ambiguous")
    print(json.dumps({"state": "stopped", "claim_id": state["claim_id"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("start")
    stopper = subcommands.add_parser("stop")
    stopper.add_argument(
        "--reason", default="Aeon RTX 5000 release gate completed"
    )
    args = parser.parse_args()
    return start() if args.command == "start" else stop(args.reason)


if __name__ == "__main__":
    raise SystemExit(main())
