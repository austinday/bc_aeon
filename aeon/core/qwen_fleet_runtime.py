"""Coordinator-owned placement and transport for release-compatible Qwen workers.

Local Docker remains implemented by :mod:`aeon.core.qwen_runtime`.  This module
adds the .177-side half of the fixed SSH remote-Docker adapter: immutable source
staging, worker preflight/start/reuse/stop calls, exact-PID heartbeats, and a
receipted loopback tunnel.  It is invoked only by a foreground Aeon session.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from .gpu_queue import QWEN_LEASE_FILE
from .qwen_capabilities import (
    RTX5000_RELEASE_CANDIDATE_KEY,
    QwenCapabilityError,
    QwenRuntimeCapability,
    qwen_runtime_capability,
)
from .qwen_runtime import (
    RUNTIME_ROOT,
    SOURCE_FILES,
    QwenRuntimeError,
    SourceIdentity,
    _private_json_read,
    _private_json_write,
    _source_identity,
)


REMOTE_STATE_FILE = RUNTIME_ROOT / "remote-runtime.json"
REMOTE_RELEASE_ROOT = Path("/home/aday/.aeon/runtime/qwen38/releases")
REMOTE_MODEL_ROOT = Path("/home/aday/.aeon/runtime/qwen38/models")
REMOTE_PYTHON = Path(
    "/home/aday/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12"
)
REMOTE_WRAPPER = Path("/home/aday/bin/fleet-low-priority")
LOCAL_PORT = 8033
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_CONTAINER_ID_RE = re.compile(r"^[a-f0-9]{64}$")
_CLAIM_RE = re.compile(r"^gc-[A-Za-z0-9._:-]{1,196}$")
_OWNER_RE = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]{8,120}$")
_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REMOTE_PHASES = frozenset({"starting", "ready", "releasing"})


def _source_sha256(source: SourceIdentity | str) -> str:
    value = source.manifest_sha256 if isinstance(source, SourceIdentity) else source
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise QwenRuntimeError("remote source manifest is malformed")
    return value


def _ssh_base(capability: QwenRuntimeCapability) -> list[str]:
    if capability.runtime_adapter != "remote-docker":
        raise QwenRuntimeError("capability is not a remote Docker release")
    return [
        "/usr/bin/ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ControlPersist=no",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=6",
        f"aday@{capability.host}",
    ]


def _release_path(source: SourceIdentity | str) -> Path:
    return REMOTE_RELEASE_ROOT / _source_sha256(source)


def _request_base(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    source: SourceIdentity | str,
) -> dict[str, Any]:
    if _SHA256_RE.fullmatch(manifest_sha256) is None:
        raise QwenRuntimeError("capability manifest identity is malformed")
    return {
        "capability_key": capability.key,
        "capability_manifest_sha256": manifest_sha256,
        "source_manifest_sha256": _source_sha256(source),
        "model_manifest_sha256": capability.model_manifest_sha256,
        "model_sha256s_sha256": capability.model_sha256s_sha256,
        "package_root": str(_release_path(source)),
        "model_dir": str(REMOTE_MODEL_ROOT / capability.model_sha256s_sha256),
        "release_gate": capability.key == RTX5000_RELEASE_CANDIDATE_KEY,
    }


def stage_remote_source(
    capability: QwenRuntimeCapability,
    package_root: Path,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> SourceIdentity:
    """Copy only the immutable source closure to its content-addressed worker root."""

    source = _source_identity(package_root, RUNTIME_ROOT / "remote-preflight")
    destination = _release_path(source)
    make_command = [
        *_ssh_base(capability),
        "/usr/bin/env",
        "-i",
        "PATH=/usr/bin:/bin",
        "HOME=/home/aday",
        "LANG=C",
        "LC_ALL=C",
        "/usr/bin/mkdir",
        "-p",
        "-m",
        "0700",
        str(destination),
    ]
    make = None
    for attempt in range(3):
        make = command_runner(
            make_command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if make.returncode == 0:
            break
        if attempt < 2:
            time.sleep(2)
    assert make is not None
    if make.returncode != 0:
        raise QwenRuntimeError("remote release root could not be prepared")
    ssh_transport = " ".join(_ssh_base(capability)[:-1])
    transfer_command = [
        "/usr/bin/bash",
        str(REMOTE_WRAPPER),
        "/usr/bin/rsync",
        "-aR",
        "--checksum",
        "--protect-args",
        "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
        "-e",
        ssh_transport,
        "--",
        *SOURCE_FILES,
        f"aday@{capability.host}:{destination}/",
    ]
    transfer = None
    for attempt in range(3):
        transfer = command_runner(
            transfer_command,
            cwd=str(package_root),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if transfer.returncode == 0:
            break
        if attempt < 2:
            time.sleep(2)
    assert transfer is not None
    if transfer.returncode != 0:
        raise QwenRuntimeError("remote immutable source staging failed")
    return source


def _remote_command(
    capability: QwenRuntimeCapability, source: SourceIdentity | str, action: str
) -> list[str]:
    if action not in {"preflight", "start", "status", "reuse", "stop", "clear"}:
        raise QwenRuntimeError("invalid remote runtime action")
    worker = _release_path(source) / "aeon/scripts/qwen_remote_worker.py"
    return [
        *_ssh_base(capability),
        "/usr/bin/env",
        "-i",
        "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME=/home/aday",
        "LANG=C",
        "LC_ALL=C",
        "USE_TF=0",
        "USE_FLAX=0",
        f"PYTHONPATH={_release_path(source)}",
        "PYTHONDONTWRITEBYTECODE=1",
        "/usr/bin/bash",
        str(REMOTE_WRAPPER),
        str(REMOTE_PYTHON),
        str(worker),
        action,
    ]


def _parse_response(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if len(result.stdout or "") > 262144:
        raise QwenRuntimeError("remote runtime response is unbounded")
    try:
        value = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("remote runtime response is malformed") from exc
    if not isinstance(value, dict) or value.get("ok") is not True:
        detail = value.get("detail") if isinstance(value, dict) else None
        if not isinstance(detail, str) or len(detail) > 500:
            detail = "remote runtime refused the request"
        raise QwenRuntimeError(detail)
    if result.returncode != 0:
        raise QwenRuntimeError("remote runtime returned contradictory success")
    return value


def remote_call(
    capability: QwenRuntimeCapability,
    source: SourceIdentity | str,
    action: str,
    request: Mapping[str, Any],
    *,
    timeout: float,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    payload = json.dumps(
        dict(request), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    attempts = 3 if action in {"preflight", "status", "reuse"} else 1
    result = None
    for attempt in range(attempts):
        result = command_runner(
            _remote_command(capability, source, action),
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.stdout or result.returncode not in {0, 255}:
            break
        if attempt + 1 < attempts:
            time.sleep(2)
    assert result is not None
    return _parse_response(result)


def remote_preflight(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    package_root: Path,
) -> tuple[SourceIdentity, dict[str, Any]]:
    source = stage_remote_source(capability, package_root)
    request = _request_base(capability, manifest_sha256, source)
    result = remote_call(
        capability, source, "preflight", request, timeout=1800
    )
    return source, result


def capability_deploy_environment(
    capability: QwenRuntimeCapability,
    base_environment: Mapping[str, Any],
    lease: Mapping[str, Any],
) -> dict[str, str]:
    if (
        capability.vram_budget_gb is None
        or capability.gpu_memory_utilization is None
        or capability.max_num_seqs is None
        or capability.max_batched_tokens is None
    ):
        raise QwenRuntimeError("remote capability lacks its release plan")
    environment = {str(key): str(value) for key, value in base_environment.items()}
    try:
        plan = json.loads(environment["AEON_DEPLOY_PLAN"])
        nodes = plan["nodes"]
        if plan.get("tier") != "solo" or not isinstance(nodes, list) or len(nodes) != 1:
            raise ValueError
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenRuntimeError("base Qwen deployment plan is malformed") from exc
    nodes[0]["ctx"] = capability.context_tokens
    nodes[0]["devices"] = str(lease["gpu_uuid"])
    plan["context_limit"] = capability.context_tokens
    plan["image"] = str(capability.image_id)
    environment.update(
        {
            "AEON_DEPLOY_PLAN": json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ),
            "AEON_GPU_MEM_UTIL": f"{capability.gpu_memory_utilization:g}",
            "AEON_LLM_VRAM_BUDGET_GB": f"{capability.vram_budget_gb:g}",
            "AEON_MAX_NUM_SEQS": str(capability.max_num_seqs),
            "AEON_MAX_NUM_BATCHED": str(capability.max_batched_tokens),
            "GPU_AGENT_CLAIM_ID": str(lease["claim_id"]),
            "GPU_LEASE_OWNER": str(lease["owner"]),
            "GPU_LEASE_RUN_DIR": str(lease["run_dir"]),
            "CUDA_VISIBLE_DEVICES": str(lease["gpu_uuid"]),
            "GPU_PLANNED_VRAM_GB": f"{capability.vram_budget_gb:g}",
            "GPU_RESERVE_GB": "6",
        }
    )
    return environment


def _capability_for_state(
    state: Mapping[str, Any], *, require_enabled: bool
) -> tuple[QwenRuntimeCapability, str]:
    key = state.get("runtime_capability_key")
    try:
        capability, current_manifest_sha256 = qwen_runtime_capability(
            key, require_enabled=require_enabled
        )
    except QwenCapabilityError as exc:
        raise QwenRuntimeError("remote runtime capability is unavailable") from exc
    if any(
        state.get(field) != expected
        for field, expected in (
            ("runtime_adapter", capability.runtime_adapter),
            ("host", capability.host),
            ("expected_hostname", capability.hostname),
            ("model_manifest_sha256", capability.model_manifest_sha256),
            ("model_sha256s_sha256", capability.model_sha256s_sha256),
        )
    ):
        raise QwenRuntimeError("remote runtime capability receipt changed")
    if state.get("physical_gpu") not in capability.allowed_physical_gpus:
        raise QwenRuntimeError("remote runtime GPU is outside its capability")
    if require_enabled and state.get(
        "runtime_capability_manifest_sha256"
    ) != current_manifest_sha256:
        raise QwenRuntimeError("remote runtime capability manifest changed")
    return capability, current_manifest_sha256


def _validate_remote_state(
    value: Any, *, require_enabled: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "phase",
        "runtime_capability_key",
        "runtime_capability_manifest_sha256",
        "runtime_adapter",
        "host",
        "expected_hostname",
        "physical_gpu",
        "gpu_uuid",
        "claim_id",
        "owner",
        "run_dir",
        "source_manifest_sha256",
        "model_manifest_sha256",
        "model_sha256s_sha256",
        "container_name",
        "container_id",
        "container_pid",
        "remote_port",
        "local_port",
        "deploy_environment",
        "tunnel_nonce",
        "tunnel_pid",
        "tunnel_create_time",
        "updated_at",
    }:
        raise QwenRuntimeError("remote runtime receipt fields changed")
    capability, _manifest_sha256 = _capability_for_state(
        value, require_enabled=require_enabled
    )
    manifest = value.get("runtime_capability_manifest_sha256")
    physical_gpu = value.get("physical_gpu")
    remote_port = value.get("remote_port")
    local_port = value.get("local_port")
    updated_at = value.get("updated_at")
    container_id = value.get("container_id")
    container_pid = value.get("container_pid")
    tunnel_nonce = value.get("tunnel_nonce")
    tunnel_pid = value.get("tunnel_pid")
    tunnel_create_time = value.get("tunnel_create_time")
    environment = value.get("deploy_environment")
    run_dir = value.get("run_dir")
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or value.get("phase") not in _REMOTE_PHASES
        or not isinstance(manifest, str)
        or _SHA256_RE.fullmatch(manifest) is None
        or type(physical_gpu) is not int
        or physical_gpu not in capability.allowed_physical_gpus
        or not isinstance(value.get("gpu_uuid"), str)
        or _UUID_RE.fullmatch(value["gpu_uuid"]) is None
        or not isinstance(value.get("claim_id"), str)
        or _CLAIM_RE.fullmatch(value["claim_id"]) is None
        or not isinstance(value.get("owner"), str)
        or _OWNER_RE.fullmatch(value["owner"]) is None
        or not isinstance(run_dir, str)
        or not run_dir.startswith("/home/aday/.aeon/runtime/qwen38/")
        or not isinstance(value.get("source_manifest_sha256"), str)
        or _SHA256_RE.fullmatch(value["source_manifest_sha256"]) is None
        or value.get("model_manifest_sha256") != capability.model_manifest_sha256
        or not isinstance(value.get("container_name"), str)
        or _CONTAINER_RE.fullmatch(value["container_name"]) is None
        or (container_id is not None and (
            not isinstance(container_id, str)
            or _CONTAINER_ID_RE.fullmatch(container_id) is None
        ))
        or (container_pid is not None and (type(container_pid) is not int or container_pid <= 1))
        or type(remote_port) is not int
        or not 1024 <= remote_port <= 65535
        or local_port != LOCAL_PORT
        or not isinstance(environment, dict)
        or len(environment) > 128
        or any(
            not isinstance(key, str)
            or not isinstance(item, str)
            or len(key) > 128
            or len(item) > 262144
            for key, item in environment.items()
        )
        or (tunnel_nonce is not None and (
            not isinstance(tunnel_nonce, str)
            or _SHA256_RE.fullmatch(tunnel_nonce) is None
        ))
        or (tunnel_pid is not None and (type(tunnel_pid) is not int or tunnel_pid <= 1))
        or (
            tunnel_create_time is not None
            and (type(tunnel_create_time) is not int or tunnel_create_time <= 0)
        )
        or not (
            (tunnel_nonce is None and tunnel_pid is None and tunnel_create_time is None)
            or (tunnel_nonce is not None and tunnel_pid is None and tunnel_create_time is None)
            or (tunnel_nonce is not None and tunnel_pid is not None and tunnel_create_time is not None)
        )
        or isinstance(updated_at, bool)
        or not isinstance(updated_at, (int, float))
        or not 0 < float(updated_at) < time.time() + 300
    ):
        raise QwenRuntimeError("remote runtime receipt is malformed")
    return dict(value)


def remote_state(*, require_enabled: bool = False) -> dict[str, Any] | None:
    value = _private_json_read(REMOTE_STATE_FILE)
    if value is None:
        return None
    return _validate_remote_state(value, require_enabled=require_enabled)


def _remote_state_matches_lease(
    state: Mapping[str, Any], lease: Mapping[str, Any]
) -> bool:
    keys = (
        "runtime_capability_key",
        "runtime_capability_manifest_sha256",
        "runtime_adapter",
        "host",
        "physical_gpu",
        "gpu_uuid",
        "claim_id",
        "owner",
        "run_dir",
    )
    return all(state.get(key) == lease.get(key) for key in keys)


def start_remote_runtime(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    source: SourceIdentity,
    lease: Mapping[str, Any],
    deploy_environment: Mapping[str, Any],
    *,
    container_name: str,
    port: int,
    heartbeat_pid: Callable[[int], None],
    progress_check: Callable[[], None] | None = None,
    timeout: float = 2100,
) -> dict[str, Any]:
    request = {
        **_request_base(capability, manifest_sha256, source),
        "lease": dict(lease),
        "deploy_environment": dict(deploy_environment),
        "container_name": container_name,
        "port": int(port),
    }
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
    process = subprocess.Popen(
        _remote_command(capability, source, "start"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    process.stdin.write(payload)
    process.stdin.close()
    process.stdin = None
    deadline = time.monotonic() + float(timeout)
    bound_pid: int | None = None
    while process.poll() is None:
        if progress_check is not None:
            progress_check()
        try:
            status = remote_call(
                capability,
                source,
                "status",
                _request_base(capability, manifest_sha256, source),
                timeout=20,
            )
            pid = status.get("container_pid")
            if not isinstance(pid, bool) and isinstance(pid, int) and pid > 1:
                if bound_pid is not None and pid != bound_pid:
                    raise QwenRuntimeError("remote container PID changed during startup")
                if bound_pid is None:
                    heartbeat_pid(pid)
                    bound_pid = pid
        except QwenRuntimeError:
            if bound_pid is not None:
                raise
        if time.monotonic() >= deadline:
            raise QwenRuntimeError("remote Qwen startup exceeded its bounded timeout")
        time.sleep(2)
    stdout, _stderr = process.communicate(timeout=5)
    result = _parse_response(
        subprocess.CompletedProcess(process.args, process.returncode, stdout, "")
    )
    pid = result.get("container_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise QwenRuntimeError("remote Qwen start has no exact PID")
    if bound_pid is None:
        heartbeat_pid(pid)
    elif pid != bound_pid:
        raise QwenRuntimeError("remote Qwen ready PID changed")
    return result


def _process_create_time(pid: int) -> int:
    payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    end = payload.rfind(")")
    if end < 0:
        raise QwenRuntimeError("tunnel process stat is malformed")
    fields = payload[end + 2 :].split()
    return int(fields[19])


def _tunnel_argv(
    capability: QwenRuntimeCapability, remote_port: int, nonce: str
) -> list[str]:
    if _SHA256_RE.fullmatch(nonce) is None:
        raise QwenRuntimeError("remote tunnel nonce is malformed")
    return [
        *_ssh_base(capability)[:-1],
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        f"ControlPath=/home/aday/.aeon/runtime/qwen38/tunnel-{nonce}.sock",
        "-L",
        f"127.0.0.1:{LOCAL_PORT}:127.0.0.1:{int(remote_port)}",
        _ssh_base(capability)[-1],
    ]


def _process_argv(pid: int) -> list[str]:
    try:
        metadata = Path(f"/proc/{pid}").stat()
        if metadata.st_uid != os.geteuid():
            raise QwenRuntimeError("remote tunnel process owner changed")
        payload = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if payload and payload[-1] == b"":
            payload.pop()
        return [item.decode("utf-8") for item in payload]
    except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
        raise QwenRuntimeError("remote tunnel process identity is unavailable") from exc


def _tunnel_candidates(expected_argv: list[str]) -> list[int]:
    candidates: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            if _process_argv(pid) == expected_argv:
                candidates.append(pid)
        except QwenRuntimeError:
            continue
    return candidates


def start_tunnel(
    capability: QwenRuntimeCapability,
    state: Mapping[str, Any],
    *,
    health_timeout: float = 30,
) -> dict[str, Any]:
    checked = _validate_remote_state(
        dict(state),
        require_enabled=capability.key != RTX5000_RELEASE_CANDIDATE_KEY,
    )
    if checked.get("tunnel_pid") is not None:
        if tunnel_is_exact(checked):
            return checked
        raise QwenRuntimeError("remote tunnel receipt is ambiguous")
    nonce = checked.get("tunnel_nonce") or os.urandom(32).hex()
    intent = {
        **checked,
        "tunnel_nonce": nonce,
        "tunnel_pid": None,
        "tunnel_create_time": None,
        "updated_at": time.time(),
    }
    _private_json_write(REMOTE_STATE_FILE, intent)
    argv = _tunnel_argv(capability, int(checked["remote_port"]), nonce)
    candidates = _tunnel_candidates(argv)
    if len(candidates) > 1:
        raise QwenRuntimeError("remote tunnel intent has duplicate processes")
    process = None
    if candidates:
        pid = candidates[0]
    else:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid = process.pid
    receipt = {
        **intent,
        "tunnel_pid": pid,
        "tunnel_create_time": _process_create_time(pid),
        "updated_at": time.time(),
    }
    _private_json_write(REMOTE_STATE_FILE, receipt)
    deadline = time.monotonic() + health_timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise QwenRuntimeError("remote Qwen tunnel exited before health")
        if not tunnel_is_exact(receipt):
            raise QwenRuntimeError("remote Qwen tunnel identity changed before health")
        try:
            response = requests.get("http://127.0.0.1:8033/health", timeout=2)
            if response.status_code == 200:
                return receipt
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise QwenRuntimeError("remote Qwen tunnel did not become healthy")


def tunnel_is_exact(state: Mapping[str, Any]) -> bool:
    pid = state.get("tunnel_pid")
    create_time = state.get("tunnel_create_time")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 1
        or isinstance(create_time, bool)
        or not isinstance(create_time, int)
    ):
        return False
    try:
        if _process_create_time(pid) != create_time:
            return False
        capability, _manifest = _capability_for_state(state, require_enabled=False)
        return _process_argv(pid) == _tunnel_argv(
            capability, int(state["remote_port"]), str(state["tunnel_nonce"])
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, QwenRuntimeError):
        return False


def tunnel_liveness(state: Mapping[str, Any]) -> str:
    pid = state.get("tunnel_pid")
    if pid is None:
        return "gone"
    if type(pid) is not int or pid <= 1:
        return "ambiguous"
    try:
        Path(f"/proc/{pid}").stat()
    except FileNotFoundError:
        return "gone"
    except OSError:
        return "ambiguous"
    return "active" if tunnel_is_exact(state) else "ambiguous"


def stop_tunnel(state: Mapping[str, Any]) -> bool:
    if state.get("tunnel_pid") is None:
        return True
    liveness = tunnel_liveness(state)
    if liveness == "gone":
        return True
    if liveness != "active":
        return False
    pid = int(state["tunnel_pid"])
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if not tunnel_is_exact(state):
            return True
        time.sleep(0.1)
    return False


def remote_runtime_liveness() -> str:
    """Return active/exited/gone/ambiguous for the exact saved worker runtime."""

    try:
        state = remote_state()
        if state is None:
            return "gone"
        capability, _current_manifest = _capability_for_state(
            state, require_enabled=False
        )
        result = remote_call(
            capability,
            str(state["source_manifest_sha256"]),
            "status",
            _request_base(
                capability,
                str(state["runtime_capability_manifest_sha256"]),
                str(state["source_manifest_sha256"]),
            ),
            timeout=30,
        )
        status = result.get("state")
        if status not in {"active", "exited", "gone"}:
            return "ambiguous"
        if status == "gone":
            return "gone"
        if result.get("claim_id") != state["claim_id"]:
            return "ambiguous"
        container_id = result.get("container_id")
        if (
            not isinstance(container_id, str)
            or _CONTAINER_ID_RE.fullmatch(container_id) is None
            or (
                state.get("container_id") is not None
                and container_id != state["container_id"]
            )
        ):
            return "ambiguous"
        pid = result.get("container_pid")
        if status == "active" and (
            type(pid) is not int
            or pid <= 1
            or (
                state.get("container_pid") is not None
                and pid != state["container_pid"]
            )
        ):
            return "ambiguous"
        return str(status)
    except Exception:
        return "ambiguous"


def remote_container_pid() -> int | None:
    state = remote_state(require_enabled=True)
    if state is None:
        return None
    if remote_runtime_liveness() != "active":
        if state.get("phase") == "starting" and state.get("container_pid") is None:
            return None
        raise QwenRuntimeError("remote Qwen container PID is not exactly active")
    pid = state.get("container_pid")
    if type(pid) is not int or pid <= 1:
        raise QwenRuntimeError("remote Qwen receipt has no exact active PID")
    return pid


def source_receipt_hash(source: SourceIdentity) -> str:
    return hashlib.sha256(source.manifest_bytes).hexdigest()


def start_managed_remote_runtime(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    source: SourceIdentity,
    lease: Mapping[str, Any],
    deploy_environment: Mapping[str, Any],
    *,
    container_name: str,
    port: int,
    heartbeat_pid: Callable[[int], None],
    progress_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from .qwen_runtime import verify_coordinator_lease

    checked = verify_coordinator_lease(lease)
    intent = {
        "schema_version": 1,
        "phase": "starting",
        "runtime_capability_key": capability.key,
        "runtime_capability_manifest_sha256": manifest_sha256,
        "runtime_adapter": capability.runtime_adapter,
        "host": capability.host,
        "expected_hostname": capability.hostname,
        "physical_gpu": checked["physical_gpu"],
        "gpu_uuid": checked["gpu_uuid"],
        "claim_id": checked["claim_id"],
        "owner": checked["owner"],
        "run_dir": checked["run_dir"],
        "source_manifest_sha256": source.manifest_sha256,
        "model_manifest_sha256": capability.model_manifest_sha256,
        "model_sha256s_sha256": capability.model_sha256s_sha256,
        "container_name": container_name,
        "container_id": None,
        "container_pid": None,
        "remote_port": int(port),
        "local_port": LOCAL_PORT,
        "deploy_environment": dict(deploy_environment),
        "tunnel_nonce": None,
        "tunnel_pid": None,
        "tunnel_create_time": None,
        "updated_at": time.time(),
    }
    if remote_state() is not None:
        raise QwenRuntimeError("a remote Qwen lifecycle receipt already exists")
    _private_json_write(REMOTE_STATE_FILE, intent)
    def bind_remote_pid(pid: int) -> None:
        current = remote_state()
        if current is None or current.get("claim_id") != intent["claim_id"]:
            raise QwenRuntimeError("remote startup receipt changed before PID binding")
        saved_pid = current.get("container_pid")
        if saved_pid is not None and saved_pid != pid:
            raise QwenRuntimeError("remote startup PID identity changed")
        _private_json_write(
            REMOTE_STATE_FILE,
            {**current, "container_pid": pid, "updated_at": time.time()},
        )
        heartbeat_pid(pid)

    result = start_remote_runtime(
        capability,
        manifest_sha256,
        source,
        checked,
        deploy_environment,
        container_name=container_name,
        port=port,
        heartbeat_pid=bind_remote_pid,
        progress_check=progress_check,
    )
    container_id = result.get("container_id")
    container_pid = result.get("container_pid")
    if (
        not isinstance(container_id, str)
        or _CONTAINER_ID_RE.fullmatch(container_id) is None
        or type(container_pid) is not int
        or container_pid <= 1
    ):
        raise QwenRuntimeError("remote Qwen ready identity is malformed")
    current = remote_state(
        require_enabled=capability.key != RTX5000_RELEASE_CANDIDATE_KEY
    )
    if current is None or not _remote_state_matches_lease(current, checked):
        raise QwenRuntimeError("remote Qwen receipt changed before readiness")
    ready = {
        **current,
        "phase": "ready",
        "container_id": container_id,
        "container_pid": container_pid,
        "updated_at": time.time(),
    }
    _private_json_write(REMOTE_STATE_FILE, ready)
    return start_tunnel(capability, ready)


def reuse_managed_remote_runtime(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    source: SourceIdentity,
    lease: Mapping[str, Any],
    *,
    container_name: str,
    port: int,
) -> int | None:
    from .qwen_runtime import verify_coordinator_lease

    state = remote_state(require_enabled=True)
    if state is None:
        return None
    checked = verify_coordinator_lease(lease)
    if any(
        state.get(key) != expected
        for key, expected in (
            ("runtime_capability_key", capability.key),
            ("runtime_capability_manifest_sha256", manifest_sha256),
            ("host", capability.host),
            ("physical_gpu", checked["physical_gpu"]),
            ("gpu_uuid", checked["gpu_uuid"]),
            ("claim_id", checked["claim_id"]),
            ("owner", checked["owner"]),
            ("run_dir", checked["run_dir"]),
            ("source_manifest_sha256", source.manifest_sha256),
            ("container_name", container_name),
            ("remote_port", int(port)),
        )
    ):
        raise QwenRuntimeError("remote runtime receipt differs from its exact lease")
    request = {
        **_request_base(capability, manifest_sha256, source),
        "lease": dict(checked),
        "config": {
            "container_name": container_name,
            "health_port": int(port),
            "_deploy_env": dict(state["deploy_environment"]),
        },
    }
    result = remote_call(capability, source, "reuse", request, timeout=60)
    pid = result.get("container_pid")
    if result.get("state") == "gone":
        return None
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise QwenRuntimeError("remote reuse has no exact container PID")
    if state.get("container_pid") is not None and state["container_pid"] != pid:
        raise QwenRuntimeError("remote reuse container PID changed")
    state = {
        **state,
        "phase": "ready",
        "container_pid": pid,
        "updated_at": time.time(),
    }
    _private_json_write(REMOTE_STATE_FILE, state)
    tunnel_status = tunnel_liveness(state)
    if tunnel_status == "gone":
        without_tunnel = {
            **state,
            "tunnel_nonce": None,
            "tunnel_pid": None,
            "tunnel_create_time": None,
        }
        _private_json_write(REMOTE_STATE_FILE, without_tunnel)
        start_tunnel(capability, without_tunnel)
    elif tunnel_status != "active":
        raise QwenRuntimeError("remote Qwen tunnel identity is ambiguous")
    return pid


def stop_managed_remote_runtime(
    capability: QwenRuntimeCapability,
    manifest_sha256: str,
    source: SourceIdentity,
    *,
    release_reason: str,
) -> bool:
    from .gpu_queue import release_vram

    state = remote_state()
    if state is None:
        return True
    saved_capability, _current_manifest = _capability_for_state(
        state, require_enabled=False
    )
    if (
        capability.key != saved_capability.key
        or manifest_sha256 != state["runtime_capability_manifest_sha256"]
        or _source_sha256(source) != state["source_manifest_sha256"]
    ):
        raise QwenRuntimeError("remote stop identity differs from its saved receipt")
    request = _request_base(
        saved_capability,
        str(state["runtime_capability_manifest_sha256"]),
        str(state["source_manifest_sha256"]),
    )
    if state.get("phase") != "releasing":
        result = remote_call(
            saved_capability,
            str(state["source_manifest_sha256"]),
            "stop",
            request,
            timeout=90,
        )
        unlaunched = (
            state.get("phase") == "starting"
            and state.get("container_id") is None
            and state.get("container_pid") is None
        )
        if (
            result.get("state") != "stopped"
            or (result.get("scratch_cleaned") is not True and not unlaunched)
        ):
            return False
        if not stop_tunnel(state):
            return False
        state = {
            **state,
            "phase": "releasing",
            "tunnel_nonce": None,
            "tunnel_pid": None,
            "tunnel_create_time": None,
            "updated_at": time.time(),
        }
        _private_json_write(REMOTE_STATE_FILE, state)
    release_vram(
        release_reason,
        QWEN_LEASE_FILE,
        expected_claim_id=str(state["claim_id"]),
    )
    cleared = remote_call(
        saved_capability,
        str(state["source_manifest_sha256"]),
        "clear",
        request,
        timeout=30,
    )
    if cleared.get("state") != "cleared":
        return False
    current = remote_state()
    if current is None or current.get("claim_id") != state["claim_id"]:
        raise QwenRuntimeError("remote release receipt changed before clear")
    metadata = REMOTE_STATE_FILE.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        raise QwenRuntimeError("remote release receipt is unsafe")
    REMOTE_STATE_FILE.unlink()
    return True
