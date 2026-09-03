"""Fleet adapter for Aeon's dedicated local/worker video-rendering service."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import psutil
import requests
from fleet_compute.adapters import AdapterLaunchError, RuntimeContext
from fleet_compute.models import (
    ArtifactCacheBinding,
    LaunchResult,
    ProbeResult,
    ProbeState,
    StopResult,
    StorageFinalizationResult,
    StoragePreparationResult,
)

from .comfy_fleet_adapter import AeonComfyFleetAdapter
from .fleet_hosts import network_address
from .utils.io import read_bounded_fd
from .video_artifact_cache import create_video_artifact_cache_backend
from .video_comfy_release import (
    VIDEO_ADAPTER_ID,
    VIDEO_ARTIFACTS_BY_ID,
    VIDEO_IMAGE_ID,
    VIDEO_LOCAL_PROFILE_ID,
    VIDEO_LOCAL_PROFILE_IDENTITIES,
    VIDEO_PROFILE_IDENTITIES,
    VIDEO_SERVICE_ID,
    VIDEO_WORKER_CACHE_ROOT,
    VIDEO_WORKER_HOSTNAMES,
    VIDEO_WORKER_PROFILE_ID,
    VIDEO_WORKER_SCRATCH_ROOT,
)

REMOTE_PYTHON = (
    "/home/aday/.local/share/uv/python/"
    "cpython-3.12-linux-x86_64-gnu/bin/python3.12"
)
REMOTE_WRAPPER = "/home/aday/bin/fleet-low-priority"
RECEIPT_NAME = "video-comfy-runtime.json"
_RUNTIME_ID = re.compile(r"^fr-[0-9a-f]{32}$")
_CLAIM_ID = re.compile(r"^gc-[A-Za-z0-9._:-]{1,196}$")


class VideoComfyFleetError(RuntimeError):
    pass


def _private_json_write(path: Path, value: Mapping[str, Any]) -> None:
    directory = path.parent
    metadata = directory.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise VideoComfyFleetError("video runtime directory is not private")
    payload = (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = directory / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise VideoComfyFleetError("video runtime receipt write was short")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def _private_json_read(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= 64 * 1024
        ):
            raise VideoComfyFleetError("video runtime receipt is unsafe")
        payload = read_bounded_fd(descriptor, 64 * 1024)
        if len(payload) != metadata.st_size:
            raise VideoComfyFleetError("video runtime receipt changed while reading")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VideoComfyFleetError("video runtime receipt is malformed") from exc
    if not isinstance(value, dict):
        raise VideoComfyFleetError("video runtime receipt is malformed")
    return value


class AeonVideoComfyFleetAdapter:
    """Use `.177` when free and exact worker `.179` as the safe fallback."""

    def __init__(self) -> None:
        self._local = AeonComfyFleetAdapter()
        self.artifact_cache_backend = create_video_artifact_cache_backend()
        self._prepared: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _ssh(host: str) -> list[str]:
        if host not in VIDEO_WORKER_HOSTNAMES:
            raise VideoComfyFleetError("video worker host is not release-qualified")
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
            f"aday@{network_address(host)}",
        ]

    @classmethod
    def _remote_run(
        cls,
        host: str,
        arguments: list[str],
        *,
        timeout: float = 60,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                *cls._ssh(host),
                shlex.join(
                    [
                        "/usr/bin/env",
                        "-i",
                        "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
                        "HOME=/home/aday",
                        "LANG=C",
                        "LC_ALL=C",
                        "/usr/bin/bash",
                        REMOTE_WRAPPER,
                        *arguments,
                    ]
                ),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode:
            raise VideoComfyFleetError("video worker command failed")
        if len(result.stdout or "") > 1024 * 1024 or len(result.stderr or "") > 8192:
            raise VideoComfyFleetError("video worker command output exceeded its bound")
        return result

    @staticmethod
    def _is_local(runtime_or_context: Any) -> bool:
        lease = getattr(runtime_or_context, "lease", None)
        host = lease.host if lease is not None else runtime_or_context.get("host")
        return host == "192.168.0.177"

    @staticmethod
    def _validate_worker_context(context: RuntimeContext) -> None:
        lease = context.lease
        if (
            context.profile.profile_id != VIDEO_WORKER_PROFILE_ID
            or context.profile.adapter != VIDEO_ADAPTER_ID
            or context.profile.service_id != VIDEO_SERVICE_ID
            or dict(context.profile.artifact_identity) != VIDEO_PROFILE_IDENTITIES
            or lease.host not in VIDEO_WORKER_HOSTNAMES
            or lease.vram_budget_gb != 40
            or not lease.exclusive
            or lease.memory_total_mib is None
            or lease.memory_total_mib < 42 * 1024
            or _RUNTIME_ID.fullmatch(context.runtime_id) is None
        ):
            raise VideoComfyFleetError("video worker lease/profile identity changed")
        expected_scratch = VIDEO_WORKER_SCRATCH_ROOT / context.runtime_id
        if (
            context.scratch_path != str(expected_scratch)
            or context.lease.run_dir != str(expected_scratch)
        ):
            raise VideoComfyFleetError("video worker scratch identity changed")

    @staticmethod
    def _cached_artifacts(
        context: RuntimeContext,
    ) -> dict[str, ArtifactCacheBinding]:
        bindings = dict(context.cached_artifacts)
        if set(bindings) != set(VIDEO_ARTIFACTS_BY_ID):
            raise VideoComfyFleetError("video worker cache bundle is incomplete")
        for artifact_id, binding in bindings.items():
            expected = VIDEO_ARTIFACTS_BY_ID[artifact_id]
            path = PurePosixPath(binding.worker_path)
            try:
                path.relative_to(PurePosixPath(str(VIDEO_WORKER_CACHE_ROOT)))
            except ValueError as exc:
                raise VideoComfyFleetError(
                    "video worker cache binding escaped its root"
                ) from exc
            if (
                binding.artifact_id != artifact_id
                or binding.kind is not expected.kind
                or binding.digest_sha256 != expected.digest_sha256
                or binding.size_bytes <= 0
            ):
                raise VideoComfyFleetError("video worker cache binding changed")
        return bindings

    @classmethod
    def _remote_storage_metrics(
        cls, host: str, scratch_path: str, runtime_id: str, *, create: bool
    ) -> tuple[str, int, int]:
        script = r'''
import json, os, pathlib, stat, sys
path_raw, runtime_id, expected, create_raw = sys.argv[1:5]
assert os.uname().nodename == expected
root = pathlib.PurePosixPath("/home/aday/.local/state/fleet-compute/runs")
path = pathlib.PurePosixPath(path_raw)
assert path == root/runtime_id and runtime_id.startswith("fr-") and ".." not in path.parts
if create_raw == "1":
    os.makedirs(path, mode=0o700, exist_ok=True); os.chmod(path, 0o700)
    output = path/"output"; os.makedirs(output, mode=0o700, exist_ok=True); os.chmod(output, 0o700)
meta = os.lstat(path)
assert stat.S_ISDIR(meta.st_mode) and not stat.S_ISLNK(meta.st_mode)
assert meta.st_uid == os.geteuid() and not meta.st_mode & 0o077
assert not os.path.ismount(path)
values = os.statvfs(path)
print(json.dumps({"device":str(meta.st_dev),"free":values.f_bavail*values.f_frsize,
 "inodes":values.f_favail},sort_keys=True))
'''
        result = cls._remote_run(
            host,
            [
                REMOTE_PYTHON,
                "-I",
                "-S",
                "-B",
                "-c",
                script,
                scratch_path,
                runtime_id,
                VIDEO_WORKER_HOSTNAMES[host],
                "1" if create else "0",
            ],
            timeout=30,
        )
        try:
            value = json.loads(result.stdout)
            return str(value["device"]), int(value["free"]), int(value["inodes"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VideoComfyFleetError(
                "video worker storage metrics are malformed"
            ) from exc

    @staticmethod
    def _container_name(runtime_id: str) -> str:
        if _RUNTIME_ID.fullmatch(runtime_id) is None:
            raise VideoComfyFleetError("video runtime identity is malformed")
        return f"aeon_video_comfyui_fr_{runtime_id[3:]}"

    @classmethod
    def _remote_container(
        cls, host: str, name: str
    ) -> dict[str, Any] | None:
        result = cls._remote_run(
            host,
            [
                "/usr/bin/docker",
                "container",
                "inspect",
                "--format",
                "{{json .Id}} {{json .Config.Labels}} {{json .State.Pid}} "
                "{{json .State.Running}} {{json .Image}}",
                name,
            ],
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            # `docker inspect` uses the same non-zero status for an absent
            # container and for daemon/transport failures. Prove absence with a
            # second, exact-name census so callers never release a lease merely
            # because Docker was temporarily unreachable.
            census = cls._remote_run(
                host,
                [
                    "/usr/bin/docker",
                    "container",
                    "ls",
                    "--all",
                    "--no-trunc",
                    "--filter",
                    f"name={name}",
                    "--format",
                    "{{json .Names}}",
                ],
                timeout=30,
                check=False,
            )
            if census.returncode:
                raise VideoComfyFleetError(
                    "video worker container absence could not be proved"
                )
            try:
                observed = [
                    json.loads(line)
                    for line in census.stdout.splitlines()
                    if line.strip()
                ]
            except (TypeError, json.JSONDecodeError) as exc:
                raise VideoComfyFleetError(
                    "video worker container absence proof is malformed"
                ) from exc
            if name in observed:
                raise VideoComfyFleetError(
                    "video worker exact container could not be inspected"
                )
            return None
        try:
            decoder = json.JSONDecoder()
            values: list[Any] = []
            source = result.stdout.strip()
            while source:
                value, end = decoder.raw_decode(source)
                values.append(value)
                source = source[end:].lstrip()
            container_id, labels, pid, running, image_id = values
            return {
                "container_id": str(container_id),
                "labels": labels if isinstance(labels, dict) else {},
                "pid": int(pid),
                "running": running is True,
                "image_id": str(image_id),
            }
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VideoComfyFleetError(
                "video worker container receipt is malformed"
            ) from exc

    @classmethod
    def _prelaunch_absent(cls, runtime: Mapping[str, Any]) -> bool:
        runtime_id = str(runtime.get("runtime_id") or "")
        host = str(runtime.get("host") or "")
        claim_id = str(runtime.get("claim_id") or "")
        physical_gpu = runtime.get("physical_gpu")
        if (
            runtime.get("profile_id") != VIDEO_WORKER_PROFILE_ID
            or runtime.get("adapter") != VIDEO_ADAPTER_ID
            or runtime.get("mode") != "service"
            or runtime.get("state") not in {"starting", "quarantined"}
            or host not in VIDEO_WORKER_HOSTNAMES
            or _RUNTIME_ID.fullmatch(runtime_id) is None
            or _CLAIM_ID.fullmatch(claim_id) is None
            or isinstance(physical_gpu, bool)
            or not isinstance(physical_gpu, int)
            or physical_gpu not in {0, 1}
            or runtime.get("run_dir")
            != str(VIDEO_WORKER_SCRATCH_ROOT / runtime_id)
            or runtime.get("pid") is not None
            or runtime.get("process_identity") is not None
            or runtime.get("endpoint") is not None
        ):
            return False
        return cls._remote_container(
            host, cls._container_name(runtime_id)
        ) is None

    @staticmethod
    def _container_matches(
        receipt: Mapping[str, Any], runtime: Mapping[str, Any]
    ) -> bool:
        labels = receipt.get("labels") or {}
        return bool(
            isinstance(labels, dict)
            and labels.get("com.bc_aeon.component") == "video-comfyui"
            and labels.get("com.bc_aeon.claim") == runtime.get("claim_id")
            and labels.get("com.bc_aeon.runtime") == runtime.get("runtime_id")
            and receipt.get("image_id") == VIDEO_IMAGE_ID
            and receipt.get("container_id") == runtime.get("container_id")
        )

    @staticmethod
    def _tunnel_argv(state: Mapping[str, Any]) -> list[str]:
        return [
            "/usr/bin/ssh",
            "-N",
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
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=6",
            "-L",
            f"127.0.0.1:{int(state['local_port'])}:127.0.0.1:{int(state['remote_port'])}",
            f"aday@{network_address(str(state['host']))}",
        ]

    @classmethod
    def _tunnel_liveness(cls, state: Mapping[str, Any]) -> str:
        pid = state.get("tunnel_pid")
        create_time = state.get("tunnel_create_time")
        if not isinstance(pid, int) or pid <= 1:
            return "gone"
        try:
            process = psutil.Process(pid)
            if process.uids().real != os.geteuid():
                return "ambiguous"
            if abs(process.create_time() - float(create_time)) > 1e-6:
                return "gone"
            argv = process.cmdline()
        except psutil.NoSuchProcess:
            return "gone"
        except (OSError, TypeError, ValueError, psutil.Error):
            return "ambiguous"
        return "active" if argv == cls._tunnel_argv(state) else "ambiguous"

    @staticmethod
    def _port_available(port: int) -> bool:
        candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            candidate.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            candidate.close()

    @classmethod
    def _choose_local_port(cls, runtime_id: str) -> int:
        start = 22000 + (int(runtime_id[-8:], 16) % 18000)
        for offset in range(128):
            port = 22000 + ((start - 22000 + offset) % 18000)
            if cls._port_available(port):
                return port
        raise VideoComfyFleetError("no private loopback tunnel port is available")

    @classmethod
    def _start_tunnel(cls, state: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
        if not cls._port_available(int(state["local_port"])):
            raise VideoComfyFleetError("video tunnel port is already occupied")
        process = subprocess.Popen(
            cls._tunnel_argv(state),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            create_time = psutil.Process(process.pid).create_time()
        except (OSError, psutil.Error) as exc:
            process.terminate()
            raise VideoComfyFleetError("video tunnel identity is unavailable") from exc
        updated = {
            **state,
            "tunnel_pid": process.pid,
            "tunnel_create_time": create_time,
        }
        _private_json_write(receipt_path, updated)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise VideoComfyFleetError("video tunnel exited during startup")
            if cls._tunnel_liveness(updated) == "active":
                return updated
            time.sleep(0.05)
        raise VideoComfyFleetError("video tunnel identity did not become stable")

    @classmethod
    def _stop_tunnel(cls, state: Mapping[str, Any]) -> bool:
        liveness = cls._tunnel_liveness(state)
        if liveness == "gone":
            return True
        if liveness != "active":
            return False
        pid = int(state["tunnel_pid"])
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if cls._tunnel_liveness(state) == "gone":
                return True
            time.sleep(0.05)
        return False

    @staticmethod
    def _healthy(endpoint: str) -> bool:
        try:
            response = requests.get(
                f"{endpoint}/system_stats",
                timeout=(2, 10),
                allow_redirects=False,
                proxies={"http": "", "https": ""},
            )
            return response.status_code == 200 and len(response.content) <= 2 * 1024 * 1024
        except (OSError, requests.RequestException):
            return False

    @classmethod
    def _runtime_from_receipt(
        cls, runtime: Mapping[str, Any], state: Mapping[str, Any]
    ) -> dict[str, Any]:
        physical_gpu = runtime.get("physical_gpu")
        expected_remote_port = (
            28188 + physical_gpu
            if isinstance(physical_gpu, int) and not isinstance(physical_gpu, bool)
            else None
        )
        required = {
            "schema_version",
            "runtime_id",
            "profile_id",
            "host",
            "claim_id",
            "container_name",
            "container_id",
            "container_pid",
            "remote_port",
            "local_port",
            "tunnel_pid",
            "tunnel_create_time",
            "image_id",
            "process_identity",
        }
        if (
            not required <= set(state)
            or state.get("schema_version") != 1
            or state.get("runtime_id") != runtime.get("runtime_id")
            or state.get("profile_id") != VIDEO_WORKER_PROFILE_ID
            or state.get("profile_id") != runtime.get("profile_id")
            or state.get("host") not in VIDEO_WORKER_HOSTNAMES
            or state.get("host") != runtime.get("host")
            or state.get("claim_id") != runtime.get("claim_id")
            or state.get("container_name")
            != cls._container_name(str(runtime.get("runtime_id") or ""))
            or re.fullmatch(r"[0-9a-f]{64}", str(state.get("container_id") or ""))
            is None
            or isinstance(state.get("container_pid"), bool)
            or not isinstance(state.get("container_pid"), int)
            or int(state["container_pid"]) <= 1
            or expected_remote_port is None
            or state.get("remote_port") != expected_remote_port
            or isinstance(state.get("local_port"), bool)
            or not isinstance(state.get("local_port"), int)
            or not 22000 <= int(state["local_port"]) < 40000
            or isinstance(state.get("tunnel_pid"), bool)
            or not isinstance(state.get("tunnel_pid"), int)
            or int(state["tunnel_pid"]) <= 1
            or isinstance(state.get("tunnel_create_time"), bool)
            or not isinstance(state.get("tunnel_create_time"), (int, float))
            or float(state["tunnel_create_time"]) <= 0
            or state.get("image_id") != VIDEO_IMAGE_ID
            or re.fullmatch(
                r"video-comfy:[0-9a-f]{64}",
                str(state.get("process_identity") or ""),
            )
            is None
            or state.get("process_identity") != runtime.get("process_identity")
            or state.get("container_pid") != runtime.get("pid")
        ):
            raise VideoComfyFleetError("video runtime receipt identity changed")
        return dict(state)

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        if self._is_local(context):
            if (
                context.profile.profile_id != VIDEO_LOCAL_PROFILE_ID
                or context.profile.adapter != VIDEO_ADAPTER_ID
                or context.profile.service_id != VIDEO_SERVICE_ID
                or dict(context.profile.artifact_identity)
                != VIDEO_LOCAL_PROFILE_IDENTITIES
            ):
                raise VideoComfyFleetError("local video profile identity changed")
            return self._local.prepare_storage(context)
        context.startup_check()
        self._validate_worker_context(context)
        bindings = self._cached_artifacts(context)
        filesystem, free_bytes, free_inodes = self._remote_storage_metrics(
            context.lease.host,
            str(context.scratch_path),
            context.runtime_id,
            create=True,
        )
        prepared = {"bindings": bindings, "filesystem_id": filesystem}
        with self._lock:
            self._prepared[context.runtime_id] = prepared
        context.startup_check()
        return StoragePreparationResult(
            scratch_path=context.scratch_path,
            filesystem_id=filesystem,
            free_bytes_after_stage=free_bytes,
            free_inodes_after_stage=free_inodes,
            staged_bytes=0,
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        if self._is_local(context):
            return self._local.launch(context)
        self._validate_worker_context(context)
        with self._lock:
            prepared = self._prepared.get(context.runtime_id)
        if prepared is None:
            raise AdapterLaunchError(
                "video worker preflight was not retained", process_absent=True
            )
        bindings: dict[str, ArtifactCacheBinding] = prepared["bindings"]
        name = self._container_name(context.runtime_id)
        if self._remote_container(context.lease.host, name) is not None:
            raise VideoComfyFleetError(
                "an existing video container is not owned by this Fleet attempt"
            )
        remote_port = 28188 + int(context.lease.physical_gpu)
        local_port = self._choose_local_port(context.runtime_id)
        output_dir = str(PurePosixPath(str(context.scratch_path)) / "output")
        environment = {
            **context.lease.required_environment,
            "GPU_RESERVE_GB": "6",
            "AEON_VIDEO_RUNTIME_ID": context.runtime_id,
            "AEON_VIDEO_CONTAINER_NAME": name,
            "AEON_VIDEO_REMOTE_PORT": str(remote_port),
            "AEON_VIDEO_OUTPUT_DIR": output_dir,
            "AEON_VIDEO_LAUNCHER": bindings["video-worker-launcher"].worker_path,
            "AEON_VIDEO_ALLOCATOR_CAP": bindings["video-allocator-cap"].worker_path,
            "AEON_VIDEO_H3_MODEL": bindings["video-h3-model"].worker_path,
            "AEON_VIDEO_H3_ENCODER": bindings["video-h3-encoder"].worker_path,
            "AEON_VIDEO_H3_VIDEO_VAE": bindings["video-h3-video-vae"].worker_path,
            "AEON_VIDEO_H3_AUDIO_VAE": bindings["video-h3-audio-vae"].worker_path,
            "AEON_VIDEO_LTX_MODEL": bindings["video-ltx-model"].worker_path,
            "AEON_VIDEO_LTX_ENCODER": bindings["video-ltx-encoder"].worker_path,
            "AEON_VIDEO_LTX_CONNECTORS": bindings["video-ltx-connectors"].worker_path,
            "AEON_VIDEO_LTX_VAE": bindings["video-ltx-vae"].worker_path,
        }
        context.heartbeat(None, "video worker artifacts and storage verified")
        launcher = bindings["video-worker-launcher"].worker_path
        result = self._remote_run(
            context.lease.host,
            [
                "/usr/bin/env",
                *[f"{key}={value}" for key, value in sorted(environment.items())],
                "/usr/bin/bash",
                launcher,
            ],
            timeout=180,
            check=False,
        )
        receipt = self._remote_container(context.lease.host, name)
        if result.returncode != 0:
            if receipt is None:
                raise AdapterLaunchError(
                    "video worker launch failed before container creation",
                    process_absent=True,
                )
            raise VideoComfyFleetError("video worker launch failed after container creation")
        if receipt is None:
            raise AdapterLaunchError(
                "video worker returned no container", process_absent=True
            )
        identity_view = {
            "runtime_id": context.runtime_id,
            "claim_id": context.lease.claim_id,
            "container_id": receipt["container_id"],
        }
        process_identity = "video-comfy:" + hashlib.sha256(
            json.dumps(identity_view, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        state = {
            "schema_version": 1,
            "runtime_id": context.runtime_id,
            "profile_id": context.profile.profile_id,
            "host": context.lease.host,
            "claim_id": context.lease.claim_id,
            "container_name": name,
            "container_id": receipt["container_id"],
            "container_pid": receipt["pid"],
            "remote_port": remote_port,
            "local_port": local_port,
            "tunnel_pid": None,
            "tunnel_create_time": None,
            "image_id": VIDEO_IMAGE_ID,
            "process_identity": process_identity,
        }
        if not self._container_matches(receipt, state) or not receipt["running"]:
            raise VideoComfyFleetError("video worker container identity changed")
        receipt_path = context.run_dir / RECEIPT_NAME
        _private_json_write(receipt_path, state)
        state = self._start_tunnel(state, receipt_path)
        endpoint = f"http://127.0.0.1:{local_port}"
        deadline = time.monotonic() + context.profile.startup_timeout_seconds
        while time.monotonic() < deadline:
            context.startup_check()
            current = self._remote_container(context.lease.host, name)
            if current is None:
                raise AdapterLaunchError(
                    "video worker container disappeared during startup",
                    process_absent=True,
                )
            if not self._container_matches(current, state):
                raise VideoComfyFleetError("video worker identity changed during startup")
            context.heartbeat(
                int(current["pid"]), "video worker exact container is starting"
            )
            if current["running"] and self._healthy(endpoint):
                return LaunchResult(
                    pid=int(current["pid"]),
                    process_identity=process_identity,
                    endpoint=endpoint,
                )
            time.sleep(2)
        raise VideoComfyFleetError("video worker did not become healthy in time")

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        if self._is_local(runtime):
            return self._local.probe(runtime)
        try:
            receipt_path = Path(str(runtime["run_dir"])) / RECEIPT_NAME
            state = self._runtime_from_receipt(runtime, _private_json_read(receipt_path))
            container = self._remote_container(
                str(runtime["host"]), state["container_name"]
            )
        except FileNotFoundError:
            # A broker restart can interrupt artifact staging before launch has
            # created a receipt. Prove that this exact runtime never crossed the
            # container boundary so recovery may release its lease instead of
            # permanently quarantining an empty pre-launch attempt.
            try:
                prelaunch_absent = self._prelaunch_absent(runtime)
            except VideoComfyFleetError:
                prelaunch_absent = False
            if prelaunch_absent:
                return ProbeResult(
                    ProbeState.ABSENT,
                    False,
                    True,
                    "video worker is provably absent before launch",
                )
            return ProbeResult(
                ProbeState.UNKNOWN,
                False,
                False,
                "video worker pre-launch identity is ambiguous",
            )
        except (KeyError, OSError, VideoComfyFleetError):
            return ProbeResult(
                ProbeState.UNKNOWN, False, False, "video worker identity is unavailable"
            )
        if container is None:
            return ProbeResult(
                ProbeState.ABSENT, False, True, "video worker container is absent"
            )
        if not self._container_matches(container, state):
            return ProbeResult(
                ProbeState.UNKNOWN, False, False, "video worker identity changed"
            )
        if not container["running"]:
            return ProbeResult(
                ProbeState.ABSENT, False, True, "video worker container exited"
            )
        tunnel = self._tunnel_liveness(state)
        if tunnel == "gone":
            try:
                state = self._start_tunnel(state, receipt_path)
            except (OSError, VideoComfyFleetError):
                return ProbeResult(
                    ProbeState.STARTING,
                    True,
                    False,
                    "video worker tunnel recovery is pending",
                )
        elif tunnel != "active":
            return ProbeResult(
                ProbeState.UNKNOWN, True, False, "video worker tunnel identity changed"
            )
        endpoint = f"http://127.0.0.1:{int(state['local_port'])}"
        if not self._healthy(endpoint):
            return ProbeResult(
                ProbeState.STARTING, True, False, "video worker health is pending"
            )
        return ProbeResult(
            ProbeState.READY, True, False, "video worker is healthy"
        )

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        if self._is_local(runtime):
            return self._local.stop(runtime, reason=reason)
        try:
            state = self._runtime_from_receipt(
                runtime, _private_json_read(Path(str(runtime["run_dir"])) / RECEIPT_NAME)
            )
        except FileNotFoundError:
            try:
                if self._prelaunch_absent(runtime):
                    return StopResult(
                        True,
                        True,
                        "video worker is provably absent before launch",
                    )
            except VideoComfyFleetError:
                pass
            return StopResult(
                False, False, "video worker pre-launch identity is ambiguous"
            )
        except (KeyError, OSError, VideoComfyFleetError):
            return StopResult(False, False, "video worker receipt is unavailable")
        if not self._stop_tunnel(state):
            return StopResult(False, False, "video worker tunnel identity changed")
        container = self._remote_container(str(runtime["host"]), state["container_name"])
        if container is None:
            return StopResult(True, True, "video worker is already absent")
        if not self._container_matches(container, state):
            return StopResult(False, False, "video worker container identity changed")
        result = self._remote_run(
            str(runtime["host"]),
            [
                "/usr/bin/docker",
                "container",
                "stop",
                "--time",
                "30",
                state["container_name"],
            ],
            timeout=45,
            check=False,
        )
        if result.returncode:
            raise VideoComfyFleetError("video worker did not stop cleanly")
        stopped = self._remote_container(str(runtime["host"]), state["container_name"])
        if stopped is not None:
            labels = stopped.get("labels") or {}
            if (
                stopped.get("container_id") != state["container_id"]
                or labels.get("com.bc_aeon.claim") != runtime.get("claim_id")
                or labels.get("com.bc_aeon.runtime") != runtime.get("runtime_id")
            ):
                return StopResult(False, False, "stopped video worker identity changed")
            removed = self._remote_run(
                str(runtime["host"]),
                ["/usr/bin/docker", "container", "rm", state["container_name"]],
                timeout=30,
                check=False,
            )
            if removed.returncode:
                raise VideoComfyFleetError("stopped video worker was not removed")
        absent = self._remote_container(
            str(runtime["host"]), state["container_name"]
        ) is None
        return StopResult(absent, absent, reason)

    @classmethod
    def _cleanup_worker_scratch(
        cls,
        host: str,
        scratch_path: str,
        runtime_id: str,
        filesystem_id: str,
    ) -> tuple[bool, int]:
        script = r'''
import json, os, pathlib, stat, sys
raw, runtime_id, expected, fsid = sys.argv[1:5]
assert os.uname().nodename == expected
root=pathlib.Path("/home/aday/.local/state/fleet-compute/runs"); path=pathlib.Path(raw)
assert path == root/runtime_id and runtime_id.startswith("fr-")
try: meta=path.lstat()
except FileNotFoundError:
    print(json.dumps({"removed":True,"bytes":0})); raise SystemExit(0)
assert stat.S_ISDIR(meta.st_mode) and not stat.S_ISLNK(meta.st_mode)
assert meta.st_uid==os.geteuid() and str(meta.st_dev)==fsid and not os.path.ismount(path)
total=0
for item in sorted(path.rglob("*"), key=lambda value:len(value.parts), reverse=True):
    item_meta=item.lstat()
    assert str(item_meta.st_dev)==fsid and not stat.S_ISLNK(item_meta.st_mode)
    assert not os.path.ismount(item)
    if stat.S_ISREG(item_meta.st_mode):
        # The reviewed ComfyUI image writes generated output as container root.
        # The exact receipt-bound attempt directory, same-filesystem check,
        # regular-file type, and single-link proof establish ownership; the
        # owner of every directory must still be the Fleet service account so a
        # root-owned or pre-existing tree can never be traversed or removed.
        assert item_meta.st_uid in {0, os.geteuid()} and item_meta.st_nlink==1
        total+=item_meta.st_size; item.unlink()
    elif stat.S_ISDIR(item_meta.st_mode):
        assert item_meta.st_uid==os.geteuid(); item.rmdir()
    else: raise AssertionError
path.rmdir()
print(json.dumps({"removed":not path.exists(),"bytes":total},sort_keys=True))
'''
        result = cls._remote_run(
            host,
            [
                REMOTE_PYTHON,
                "-I",
                "-S",
                "-B",
                "-c",
                script,
                scratch_path,
                runtime_id,
                VIDEO_WORKER_HOSTNAMES[host],
                filesystem_id,
            ],
            timeout=300,
        )
        try:
            value = json.loads(result.stdout)
            return value.get("removed") is True, int(value.get("bytes", 0))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VideoComfyFleetError("video worker cleanup receipt is malformed") from exc

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        if self._is_local(runtime):
            return self._local.finalize_storage(runtime, storage)
        receipt_path = Path(str(runtime.get("run_dir") or "")) / RECEIPT_NAME
        try:
            state = self._runtime_from_receipt(runtime, _private_json_read(receipt_path))
            if self._remote_container(str(runtime["host"]), state["container_name"]) is not None:
                raise VideoComfyFleetError("video worker still has a container")
            if self._tunnel_liveness(state) != "gone":
                raise VideoComfyFleetError("video worker still has a tunnel")
            filesystem_id = str(storage["filesystem_id"])
        except FileNotFoundError:
            # Artifact-cache admission can fail before adapter preparation. In
            # that exact state there is intentionally no runtime receipt, PID,
            # process identity, container, or tunnel; settle only the computed
            # attempt scratch and never infer ownership of another process.
            runtime_id = str(runtime.get("runtime_id") or "")
            host = str(runtime.get("host") or "")
            if (
                runtime.get("profile_id") != VIDEO_WORKER_PROFILE_ID
                or host not in VIDEO_WORKER_HOSTNAMES
                or _RUNTIME_ID.fullmatch(runtime_id) is None
                or runtime.get("pid") is not None
                or runtime.get("process_identity") is not None
                or self._remote_container(host, self._container_name(runtime_id))
                is not None
            ):
                return StorageFinalizationResult(
                    True, False, 0, "video pre-launch cleanup identity is ambiguous"
                )
            try:
                filesystem_id, _free, _inodes = self._remote_storage_metrics(
                    host,
                    str(storage["scratch_path"]),
                    runtime_id,
                    create=False,
                )
            except (KeyError, VideoComfyFleetError):
                # The adapter never created remote scratch. The cleanup proof
                # treats exact absence as success before consulting this token.
                filesystem_id = "prelaunch-absent"
            try:
                removed, reclaimed = self._cleanup_worker_scratch(
                    host,
                    str(storage["scratch_path"]),
                    runtime_id,
                    filesystem_id,
                )
            except (KeyError, VideoComfyFleetError) as exc:
                return StorageFinalizationResult(
                    True, False, 0, f"video worker cleanup pending: {exc}"
                )
            return StorageFinalizationResult(
                True,
                removed,
                reclaimed,
                "video pre-launch scratch is absent or removed",
            )
        try:
            removed, reclaimed = self._cleanup_worker_scratch(
                str(runtime["host"]),
                str(storage["scratch_path"]),
                str(runtime["runtime_id"]),
                filesystem_id,
            )
        except (KeyError, OSError, VideoComfyFleetError) as exc:
            return StorageFinalizationResult(
                True, False, 0, f"video worker cleanup pending: {exc}"
            )
        return StorageFinalizationResult(
            True,
            removed,
            reclaimed,
            "video output was transferred through the private API; worker scratch removed",
        )


def create_fleet_adapter() -> AeonVideoComfyFleetAdapter:
    return AeonVideoComfyFleetAdapter()
