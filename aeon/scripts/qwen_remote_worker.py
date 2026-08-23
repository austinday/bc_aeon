#!/usr/bin/env python3
"""Exact, non-daemon worker-side adapter for Aeon's remote Qwen container.

The orchestrating Aeon process remains on .177 and owns coordinator admission,
heartbeats, retry, and release.  This program is copied as part of the immutable
Aeon source closure and is invoked only through fixed BatchMode SSH.  It manages
one exact Docker receipt on the already-selected worker and never contacts or
reimplements the fleet coordinator.
"""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

from aeon.core.qwen_capabilities import (
    QwenCapabilityError,
    qwen_release_candidate_capability,
    qwen_runtime_capability,
)
from aeon.core.qwen_runtime import (
    ArtifactIdentity,
    QwenRuntimeError,
    RUNTIME_ROOT,
    _private_json_read,
    _private_json_write,
    _source_identity,
    clear_runtime_state,
    current_runtime_state,
    load_artifact_identity,
    local_container_pid,
    local_image_id,
    local_image_size,
    qwen_runtime_liveness,
    reuse_qwen_runtime,
    start_local_runtime,
    stop_qwen_runtime,
)


MAX_REQUEST_BYTES = 1024 * 1024
MODEL_CACHE_ROOT = RUNTIME_ROOT / "models"
RELEASE_ROOT = RUNTIME_ROOT / "releases"


def _request() -> dict[str, Any]:
    payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if not payload or len(payload) > MAX_REQUEST_BYTES:
        raise QwenRuntimeError("remote runtime request size is invalid")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("remote runtime request is malformed") from exc
    if not isinstance(value, dict):
        raise QwenRuntimeError("remote runtime request is not an object")
    return value


def _capability(request: dict[str, Any]):
    try:
        if request.get("release_gate") is True:
            capability, manifest_sha256 = qwen_release_candidate_capability(
                request.get("capability_key")
            )
        elif request.get("release_gate") is False:
            capability, manifest_sha256 = qwen_runtime_capability(
                request.get("capability_key"), require_enabled=True
            )
        else:
            raise QwenCapabilityError("remote release-gate marker is malformed")
    except QwenCapabilityError as exc:
        raise QwenRuntimeError("remote runtime capability is unavailable") from exc
    if (
        capability.runtime_adapter != "remote-docker"
        or socket.gethostname() != capability.hostname
        or request.get("capability_manifest_sha256") != manifest_sha256
    ):
        raise QwenRuntimeError("remote runtime host/capability identity changed")
    return capability, manifest_sha256


def _release_paths(request: dict[str, Any], capability) -> tuple[Path, Path, Path]:
    source_sha256 = request.get("source_manifest_sha256")
    model_sha256 = request.get("model_sha256s_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in source_sha256)
        or not isinstance(model_sha256, str)
        or len(model_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in model_sha256)
    ):
        raise QwenRuntimeError("remote release identity is malformed")
    package_root = RELEASE_ROOT / source_sha256
    model_dir = MODEL_CACHE_ROOT / model_sha256
    preflight = RUNTIME_ROOT / f"preflight-{capability.key}.json"
    if request.get("package_root") != str(package_root) or request.get(
        "model_dir"
    ) != str(model_dir):
        raise QwenRuntimeError("remote release path is not content-addressed")
    return package_root, model_dir, preflight


def _artifact_payload(identity: ArtifactIdentity) -> dict[str, Any]:
    return {
        "model_dir": str(identity.model_dir),
        "manifest_sha256": identity.manifest_sha256,
        "sha256s_sha256": identity.sha256s_sha256,
        "files": list(identity.files),
        "total_bytes": identity.total_bytes,
        "root_device": identity.root_device,
        "root_inode": identity.root_inode,
        "file_stats": [list(item) for item in identity.file_stats],
    }


def _artifact_from_payload(value: Any) -> ArtifactIdentity:
    if not isinstance(value, dict):
        raise QwenRuntimeError("remote model preflight receipt is malformed")
    try:
        return ArtifactIdentity(
            model_dir=Path(value["model_dir"]),
            manifest_sha256=str(value["manifest_sha256"]),
            sha256s_sha256=str(value["sha256s_sha256"]),
            files=tuple(str(item) for item in value["files"]),
            total_bytes=int(value["total_bytes"]),
            root_device=int(value["root_device"]),
            root_inode=int(value["root_inode"]),
            file_stats=tuple(tuple(item) for item in value["file_stats"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QwenRuntimeError("remote model preflight receipt changed") from exc


def _preflight(request: dict[str, Any]) -> dict[str, Any]:
    capability, manifest_sha256 = _capability(request)
    package_root, model_dir, preflight_path = _release_paths(request, capability)
    source = _source_identity(package_root, RUNTIME_ROOT / "preflight")
    if source.manifest_sha256 != request["source_manifest_sha256"]:
        raise QwenRuntimeError("remote source release changed")
    artifact = load_artifact_identity(model_dir, verify_payload=True)
    if (
        artifact.manifest_sha256 != capability.model_manifest_sha256
        or artifact.manifest_sha256 != request.get("model_manifest_sha256")
        or artifact.sha256s_sha256 != request["model_sha256s_sha256"]
    ):
        raise QwenRuntimeError("remote model release changed")
    image_id = local_image_id(str(capability.image_id))
    image_size = local_image_size(image_id)
    if image_id != capability.image_id:
        raise QwenRuntimeError("remote image release changed")
    receipt = {
        "schema_version": 1,
        "capability_key": capability.key,
        "capability_manifest_sha256": manifest_sha256,
        "source_manifest_sha256": source.manifest_sha256,
        "model_sha256s_sha256": artifact.sha256s_sha256,
        "model_manifest_sha256": artifact.manifest_sha256,
        "artifact": _artifact_payload(artifact),
        "image_id": image_id,
        "image_size_bytes": image_size,
        "verified_at": time.time(),
    }
    _private_json_write(preflight_path, receipt)
    return {
        "state": "preflight_ready",
        "capability_key": capability.key,
        "source_manifest_sha256": source.manifest_sha256,
        "model_manifest_sha256": artifact.manifest_sha256,
        "model_sha256s_sha256": artifact.sha256s_sha256,
        "image_id": image_id,
        "image_size_bytes": image_size,
    }


def _load_preflight(request: dict[str, Any], capability) -> tuple[Path, Path, dict[str, Any]]:
    package_root, model_dir, preflight_path = _release_paths(request, capability)
    receipt = _private_json_read(preflight_path)
    if (
        receipt is None
        or receipt.get("schema_version") != 1
        or receipt.get("capability_key") != capability.key
        or receipt.get("capability_manifest_sha256")
        != request.get("capability_manifest_sha256")
        or receipt.get("source_manifest_sha256")
        != request.get("source_manifest_sha256")
        or receipt.get("model_sha256s_sha256")
        != request.get("model_sha256s_sha256")
        or receipt.get("model_manifest_sha256")
        != request.get("model_manifest_sha256")
        or receipt.get("image_id") != capability.image_id
    ):
        raise QwenRuntimeError("remote runtime preflight is absent or stale")
    return package_root, model_dir, receipt


def _state_for_request(request: dict[str, Any], capability) -> dict[str, Any] | None:
    package_root, model_dir, _preflight_path = _release_paths(request, capability)
    state = current_runtime_state()
    if state is None:
        return None
    if any(
        state.get(key) != expected
        for key, expected in (
            ("runtime_capability_key", capability.key),
            (
                "runtime_capability_manifest_sha256",
                request.get("capability_manifest_sha256"),
            ),
            ("source_manifest_sha256", request.get("source_manifest_sha256")),
            ("model_sha256s_sha256", request.get("model_sha256s_sha256")),
            ("model_manifest_sha256", request.get("model_manifest_sha256")),
            ("source_dir", str(Path(state["run_dir"]) / f"local-source-{request.get('source_manifest_sha256')}")),
            ("model_dir", str(model_dir)),
        )
    ):
        raise QwenRuntimeError("remote saved runtime differs from its request")
    if package_root.name != request.get("source_manifest_sha256"):
        raise QwenRuntimeError("remote source path identity changed")
    return state


def _start(request: dict[str, Any]) -> dict[str, Any]:
    capability, _manifest_sha256 = _capability(request)
    package_root, model_dir, preflight = _load_preflight(request, capability)
    lease = request.get("lease")
    deploy_environment = request.get("deploy_environment")
    if not isinstance(lease, dict) or not isinstance(deploy_environment, dict):
        raise QwenRuntimeError("remote start request lacks its exact lease/plan")
    artifact = _artifact_from_payload(preflight.get("artifact"))
    if artifact.model_dir != model_dir:
        raise QwenRuntimeError("remote model preflight path changed")
    state = start_local_runtime(
        lease,
        deploy_environment,
        package_root=package_root,
        model_dir=model_dir,
        container_name=str(request.get("container_name") or ""),
        image=str(capability.image_id),
        port=int(request.get("port") or 0),
        artifact_identity=artifact,
        image_identity=str(preflight["image_id"]),
        image_size_bytes=int(preflight["image_size_bytes"]),
        coordinator_verify_func=False,
        final_heartbeat_func=lambda *_args, **_kwargs: None,
        heartbeat_promoter=lambda: int(local_container_pid() or 0),
    )
    return {
        "state": "ready",
        "container_id": state["container_id"],
        "container_pid": state["container_pid"],
        "claim_id": state["claim_id"],
    }


def _status(request: dict[str, Any]) -> dict[str, Any]:
    capability, _manifest_sha256 = _capability(request)
    state = _state_for_request(request, capability)
    if state is None:
        return {"state": "gone", "container_pid": None, "container_id": None}
    if state.get("runtime_capability_key") != capability.key:
        raise QwenRuntimeError("remote saved runtime belongs to another capability")
    liveness = qwen_runtime_liveness()
    return {
        "state": liveness,
        "phase": state.get("phase"),
        "container_pid": state.get("container_pid"),
        "container_id": state.get("container_id"),
        "claim_id": state.get("claim_id"),
        "scratch_cleaned": state.get("scratch_cleaned"),
    }


def _reuse(request: dict[str, Any]) -> dict[str, Any]:
    capability, _manifest_sha256 = _capability(request)
    package_root, _model_dir, _preflight = _load_preflight(request, capability)
    lease = request.get("lease")
    config = request.get("config")
    if not isinstance(lease, dict) or not isinstance(config, dict):
        raise QwenRuntimeError("remote reuse request is malformed")
    pid = reuse_qwen_runtime(
        config=config,
        package_root=package_root,
        lease_override=lease,
        coordinator_verify_func=False,
    )
    if pid is None:
        return {"state": "gone", "container_pid": None}
    return {"state": "active", "container_pid": pid}


def _stop(request: dict[str, Any]) -> dict[str, Any]:
    capability, _manifest_sha256 = _capability(request)
    _state_for_request(request, capability)
    stopped = stop_qwen_runtime(allow_lost_lease=True)
    state = current_runtime_state()
    return {
        "state": "stopped" if stopped else "ambiguous",
        "scratch_cleaned": None if state is None else state.get("scratch_cleaned"),
    }


def _clear(request: dict[str, Any]) -> dict[str, Any]:
    capability, _manifest_sha256 = _capability(request)
    state = _state_for_request(request, capability)
    if state is None:
        return {"state": "cleared"}
    if (
        state.get("phase") != "releasing"
        or state.get("scratch_cleaned") is not True
        or qwen_runtime_liveness() != "gone"
    ):
        raise QwenRuntimeError("remote runtime is not safe to clear")
    clear_runtime_state()
    return {"state": "cleared"}


_ACTIONS = {
    "preflight": _preflight,
    "start": _start,
    "status": _status,
    "reuse": _reuse,
    "stop": _stop,
    "clear": _clear,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in _ACTIONS:
        print(json.dumps({"ok": False, "error": "invalid_action"}))
        return 64
    try:
        result = _ACTIONS[sys.argv[1]](_request())
    except (QwenRuntimeError, QwenCapabilityError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": type(exc).__name__, "detail": str(exc)},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
