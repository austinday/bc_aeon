#!/usr/bin/env python3
"""Refuse legacy direct launch; retain exact historical teardown recovery.

Release qualification now submits durable demand through Fleet Compute. This
module cannot acquire a coordinator claim or start a runtime. The stop command
remains only so an operator following the recovery runbook can retire an exact
receipt created by an older version of this script.
"""

from __future__ import annotations

import argparse
import json

from aeon.core.qwen_capabilities import qwen_release_candidate_capability
from aeon.core.qwen_fleet_runtime import (
    remote_state,
    stop_managed_remote_runtime,
)
from aeon.core.qwen_runtime import QwenRuntimeError


def start() -> int:
    raise QwenRuntimeError(
        "legacy direct coordinator release-gate launch is retired; submit the "
        "aeon-qwen38-compact-178-release-gate service through Fleet Compute"
    )


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
    subcommands.add_parser("start", help="fail closed; direct launch is retired")
    stopper = subcommands.add_parser(
        "stop", help="recover one exact receipt created by the retired launcher"
    )
    stopper.add_argument(
        "--reason", default="Aeon RTX 5000 release gate completed"
    )
    args = parser.parse_args()
    return start() if args.command == "start" else stop(args.reason)


if __name__ == "__main__":
    raise SystemExit(main())
