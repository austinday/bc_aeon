"""Reviewed Fleet Compute adapter for Aeon's exact Qwen runtime.

The broker owns the cooperative coordinator claim.  This adapter owns only the
Aeon runtime it creates from the immutable capability receipt; it never scans,
stops, or removes an unrelated container.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import threading
import time
from typing import Any, Mapping

import requests

from fleet_compute.adapters import AdapterLaunchError, RuntimeContext
from fleet_compute.models import (
    LaunchResult,
    ProbeResult,
    ProbeState,
    StopResult,
    StorageFinalizationResult,
    StoragePreparationResult,
)

from .deploy_planner import plan
from .gpu import GpuInfo
from .model_catalog import QWEN38_MODEL_NAME, by_name
from .qwen_artifact_cache import create_artifact_cache_backend
from .qwen_capabilities import qwen_runtime_capability, require_enabled_qwen_target
from .qwen_capabilities import (
    QwenCapabilityError,
    RTX5000_RELEASE_CANDIDATE_KEY,
    require_qwen_release_candidate_target,
)
from .qwen_fleet_runtime import (
    capability_deploy_environment,
    fleet_remote_runtime_resources,
    qwen_remote_artifact_cache,
    remote_preflight,
    recover_remote_uncommitted_intent,
    remote_runtime_liveness,
    remote_state,
    restore_managed_remote_tunnel,
    start_managed_remote_runtime,
    stop_managed_remote_runtime,
    tunnel_liveness,
)
from .qwen_runtime import (
    QWEN_RELEASE_ATTENTION_BACKEND,
    QwenRuntimeError,
    SourceIdentity,
    _source_identity,
    clear_runtime_state,
    current_runtime_state,
    load_artifact_identity,
    local_container_pid,
    local_image_id,
    local_image_size,
    qwen_runtime_liveness,
    start_local_runtime,
    stop_qwen_runtime,
)
from .fleet_hosts import network_address


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
AEON_HOME = Path(os.environ.get("AEON_HOME", "/home/aday/.aeon")).resolve()
LOCAL_HOST = "192.168.0.177"
LOCAL_ENDPOINT = "http://127.0.0.1:8033/v1"
EXPECTED_HOSTNAMES = {
    "192.168.0.177": "DAY2RTX6000PRO",
    "192.168.0.178": "DAY2XRTX5000",
    "192.168.0.180": "DAY2XRTX5000PRO-2",
}
_RUNTIME_ID = re.compile(r"^fr-[0-9a-f]{32}$")
_RUNTIME_SOURCE_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_FLEET_RUN_ROOT = Path("/home/aday/.local/state/fleet-compute/runs")
_REMOTE_PYTHON = (
    "/home/aday/.local/share/uv/python/"
    "cpython-3.12-linux-x86_64-gnu/bin/python3.12"
)
_ATTENTION_BACKEND_DIGESTS = {
    "9d18098b598382de6483b25f2dedcc843b4c7c9998fef5df1f371c4a20f29a0f": (
        QWEN_RELEASE_ATTENTION_BACKEND
    ),
}
_TRITON_PROFILE_IDS = frozenset(
    {
        "aeon-qwen38-standard",
        "aeon-qwen38-compact-workers",
        "aeon-qwen38-compact-178-release-gate",
        "aeon-qwen38-compact-178",
        "aeon-qwen38-compact-180",
    }
)
_RELEASE_GATE_PROFILE_IDS = frozenset(
    {"aeon-qwen38-compact-178-release-gate"}
)


class _StartupHeartbeat:
    def __init__(self, context: RuntimeContext) -> None:
        self.context = context
        self.pid: int | None = None
        self.error: BaseException | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name=f"fleet-aeon-heartbeat-{context.runtime_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.context.heartbeat(None, "Aeon Qwen reviewed runtime startup")
        self.thread.start()

    def promote(self, pid: int) -> int:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
            raise QwenRuntimeError("Aeon broker startup PID is invalid")
        self.pid = pid
        self.context.heartbeat(pid, "Aeon Qwen startup bound to exact runtime PID")
        return pid

    def check(self) -> None:
        if self.error is not None:
            raise QwenRuntimeError("Aeon broker startup heartbeat failed") from self.error

    def close(self) -> None:
        self.stop_event.set()
        # A heartbeat may already be inside the broker/coordinator's bounded
        # 60-second call. Prove it has stopped before launch returns so this
        # adapter can never emit a late heartbeat after PID promotion/teardown.
        self.thread.join(timeout=65)
        if self.thread.is_alive():
            raise QwenRuntimeError("Aeon broker startup heartbeat did not stop")
        self.check()

    def _run(self) -> None:
        while not self.stop_event.wait(240):
            try:
                self.context.heartbeat(
                    self.pid, "Aeon Qwen reviewed runtime is still starting"
                )
            except BaseException as exc:
                self.error = exc
                return


class AeonQwenFleetAdapter:
    """One exact, release-bound Qwen service across reviewed host variants."""

    def __init__(self) -> None:
        self._prepared: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.artifact_cache_backend = create_artifact_cache_backend()

    @staticmethod
    def _attention_backend(context: RuntimeContext) -> str:
        digest = context.profile.artifact_identity.get("attention_backend")
        backend = _ATTENTION_BACKEND_DIGESTS.get(digest)
        if backend is None:
            raise QwenRuntimeError("Fleet profile lacks an exact attention backend")
        if context.profile.profile_id not in _TRITON_PROFILE_IDS:
            raise QwenRuntimeError("Fleet attention backend/profile identity changed")
        return backend

    @staticmethod
    def _require_runtime_source(
        context: RuntimeContext, source: SourceIdentity
    ) -> None:
        expected = context.profile.artifact_identity.get("runtime_source")
        observed = (
            source.manifest_sha256
            if isinstance(source, SourceIdentity)
            else None
        )
        if (
            not isinstance(expected, str)
            or _RUNTIME_SOURCE_SHA256.fullmatch(expected) is None
            or observed != expected
        ):
            raise QwenRuntimeError(
                "Fleet profile runtime source differs from the staged Aeon release"
            )

    @staticmethod
    def _capability(context: RuntimeContext):
        if context.profile.profile_id in _RELEASE_GATE_PROFILE_IDS:
            capability, manifest_sha256 = require_qwen_release_candidate_target(
                RTX5000_RELEASE_CANDIDATE_KEY,
                context.lease.host,
                context.lease.physical_gpu,
            )
        else:
            capability, manifest_sha256 = require_enabled_qwen_target(
                context.lease.host, context.lease.physical_gpu
            )
        if (
            capability.vram_budget_gb is None
            or abs(capability.vram_budget_gb - context.lease.vram_budget_gb) > 1e-9
            or capability.min_physical_vram_gb
            != context.profile.min_physical_vram_gb
            or capability.exclusive is not context.profile.exclusive
        ):
            raise QwenRuntimeError("Fleet profile differs from Aeon capability")
        expected = context.profile.artifact_identity
        if (
            expected.get("image") != str(capability.image_id).removeprefix("sha256:")
            or expected.get("model_manifest") != capability.model_manifest_sha256
            or expected.get("model_sha256s") != capability.model_sha256s_sha256
            or expected.get("runtime_capabilities") != manifest_sha256
        ):
            raise QwenRuntimeError("Fleet profile artifact identity changed")
        return capability, manifest_sha256

    @staticmethod
    def _lease(context: RuntimeContext, capability, manifest_sha256: str) -> dict[str, Any]:
        lease = context.lease
        if lease.memory_total_mib is None or lease.model is None:
            raise QwenRuntimeError("coordinator lease omitted physical GPU identity")
        profile_is_release_gate = (
            context.profile.profile_id in _RELEASE_GATE_PROFILE_IDS
        )
        if profile_is_release_gate:
            try:
                expected, expected_manifest = require_qwen_release_candidate_target(
                    RTX5000_RELEASE_CANDIDATE_KEY,
                    lease.host,
                    lease.physical_gpu,
                )
            except QwenCapabilityError as exc:
                raise QwenRuntimeError(
                    "Fleet release-gate profile/capability binding changed"
                ) from exc
            if capability != expected or manifest_sha256 != expected_manifest:
                raise QwenRuntimeError(
                    "Fleet release-gate profile/capability binding changed"
                )
        elif capability.key == RTX5000_RELEASE_CANDIDATE_KEY:
            raise QwenRuntimeError(
                "Fleet release-gate profile/capability binding changed"
            )
        return {
            "schema_version": 3,
            "record_type": "lease",
            "claim_id": lease.claim_id,
            "owner": lease.owner,
            "project": "bc-aeon",
            "purpose": context.profile.purpose,
            "host": lease.host,
            "physical_gpu": lease.physical_gpu,
            "gpu_uuid": lease.gpu_uuid,
            "model": lease.model,
            "memory_total_mib": lease.memory_total_mib,
            "vram_budget_mib": round(lease.vram_budget_gb * 1024),
            "vram_budget_gb": lease.vram_budget_gb,
            "exclusive": lease.exclusive,
            "run_dir": lease.run_dir,
            "compute_profile": capability.compute_profile,
            "min_host_memory_gb": context.profile.min_host_memory_gb,
            "min_host_commit_gb": context.profile.min_host_commit_gb,
            "min_disk_free_gb": context.profile.min_disk_free_gb,
            "min_shm_free_gb": context.profile.min_shm_free_gb,
            "runtime_capability_key": capability.key,
            "runtime_capability_manifest_sha256": manifest_sha256,
            "runtime_adapter": capability.runtime_adapter,
            "release_gate": profile_is_release_gate,
            "reserved_at": time.time(),
        }

    @staticmethod
    def _base_plan(attention_backend: str) -> tuple[Any, dict[str, str]]:
        entry = by_name(QWEN38_MODEL_NAME)
        if entry is None:
            raise QwenRuntimeError("Aeon Qwen catalog entry is unavailable")
        planned = plan(
            entry,
            [GpuInfo(index=0, name="RTX 6000 Pro", total_gib=96, free_gib=96)],
            mode="solo",
        )
        environment = {
            **planned.env,
            "AEON_HOME": str(AEON_HOME),
            "AEON_VLLM_ATTENTION_BACKEND": attention_backend,
        }
        return planned, environment

    @staticmethod
    def _fs_metrics(path: Path) -> tuple[str, int, int]:
        metadata = path.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise QwenRuntimeError("Fleet runtime directory is not private and owned")
        values = os.statvfs(path)
        return (
            str(metadata.st_dev),
            values.f_bavail * values.f_frsize,
            values.f_favail,
        )

    @staticmethod
    def _remote_metrics(host: str, path: str, *, create: bool) -> tuple[str, int, int]:
        hostname = EXPECTED_HOSTNAMES.get(host)
        if hostname is None or host == LOCAL_HOST:
            raise QwenRuntimeError("unsupported remote Aeon host")
        script = """
import json
import os
import stat
import sys

path, expected, create_raw = sys.argv[1:4]
assert os.uname().nodename == expected
if create_raw == "1":
    os.makedirs(path, mode=0o700, exist_ok=True)
try:
    metadata = os.lstat(path)
except FileNotFoundError:
    print(json.dumps({"state": "absent"}, sort_keys=True))
    raise SystemExit(0)
assert stat.S_ISDIR(metadata.st_mode)
assert metadata.st_uid == os.geteuid()
assert not metadata.st_mode & 0o077
values = os.statvfs(path)
print(json.dumps({
    "state": "present",
    "device": str(metadata.st_dev),
    "free": values.f_bavail * values.f_frsize,
    "inodes": values.f_favail,
}, sort_keys=True))
"""
        result = subprocess.run(
            [
                "/usr/bin/ssh", "-T", "-o", "BatchMode=yes", "-o",
                "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes", "-o",
                "IdentitiesOnly=yes", f"aday@{network_address(host)}", shlex.join([
                    _REMOTE_PYTHON, "-c", script, path, hostname,
                    "1" if create else "0",
                ]),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 or len(result.stdout) > 4096:
            raise QwenRuntimeError("worker storage metrics are unavailable")
        try:
            value = json.loads(result.stdout)
            if value.get("state") == "absent":
                raise FileNotFoundError(path)
            if value.get("state") != "present":
                raise ValueError
            return str(value["device"]), int(value["free"]), int(value["inodes"])
        except FileNotFoundError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise QwenRuntimeError("worker storage metrics are malformed") from exc

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        context.startup_check()
        if _RUNTIME_ID.fullmatch(context.runtime_id) is None:
            raise QwenRuntimeError("Fleet runtime identity is malformed")
        capability, manifest_sha256 = self._capability(context)
        attention_backend = self._attention_backend(context)
        lease = self._lease(context, capability, manifest_sha256)
        planned, base_environment = self._base_plan(attention_backend)
        environment = capability_deploy_environment(
            capability, base_environment, lease
        )
        prepared: dict[str, Any] = {
            "capability": capability,
            "manifest_sha256": manifest_sha256,
            "lease": lease,
            "plan": planned,
            "environment": environment,
        }
        if capability.runtime_adapter == "local-docker":
            source = _source_identity(
                PACKAGE_ROOT, context.run_dir / "local-source-preflight"
            )
            self._require_runtime_source(context, source)
            model_dir = AEON_HOME / "models" / environment["AEON_LOCAL_MODEL_DIR"]
            artifact = load_artifact_identity(
                model_dir,
                verify_payload=True,
                progress_check=context.startup_check,
            )
            image = str(json.loads(environment["AEON_DEPLOY_PLAN"])["image"])
            image_id = local_image_id(image)
            image_size = local_image_size(image_id)
            if image_id != capability.image_id:
                raise QwenRuntimeError("local Qwen image differs from capability")
            prepared.update(
                artifact=artifact,
                image=image,
                image_id=image_id,
                image_size=image_size,
                model_dir=model_dir,
                source=source,
            )
            filesystem_id, free_bytes, free_inodes = self._fs_metrics(context.run_dir)
            staged_bytes = 0
        elif capability.runtime_adapter == "remote-docker":
            if context.scratch_path != context.lease.run_dir:
                raise QwenRuntimeError("worker scratch must equal the exact lease run directory")
            resources = fleet_remote_runtime_resources(
                context.lease.run_dir,
                context.lease.physical_gpu,
                host=context.lease.host,
            )
            if (
                resources["runtime_id"] != context.runtime_id
                or context.run_dir != Path(resources["run_dir"])
            ):
                raise QwenRuntimeError("Fleet runtime ID and lease run directory differ")
            _device, before, _before_inodes = self._remote_metrics(
                capability.host, context.lease.run_dir, create=True
            )
            source_identity = _source_identity(
                PACKAGE_ROOT,
                context.run_dir / "remote-source-preflight",
            )
            artifact_cache = qwen_remote_artifact_cache(
                capability,
                source_identity,
                context.cached_artifacts,
            )
            source, receipt = remote_preflight(
                capability,
                manifest_sha256,
                PACKAGE_ROOT,
                artifact_cache=artifact_cache,
                startup_check=context.startup_check,
            )
            context.startup_check()
            self._require_runtime_source(context, source)
            if (
                receipt.get("image_id") != capability.image_id
                or receipt.get("model_manifest_sha256")
                != capability.model_manifest_sha256
                or receipt.get("model_sha256s_sha256")
                != capability.model_sha256s_sha256
            ):
                raise QwenRuntimeError("worker Qwen preflight differs from capability")
            filesystem_id, free_bytes, free_inodes = self._remote_metrics(
                capability.host, context.lease.run_dir, create=False
            )
            staged_bytes = max(0, before - free_bytes)
            prepared.update(
                source=source,
                remote_receipt=receipt,
                artifact_cache=artifact_cache,
            )
            prepared["remote_resources"] = resources
        else:
            raise QwenRuntimeError("Aeon capability has an unsupported runtime adapter")
        context.startup_check()
        with self._lock:
            self._prepared[context.runtime_id] = prepared
        return StoragePreparationResult(
            scratch_path=context.scratch_path,
            filesystem_id=filesystem_id,
            free_bytes_after_stage=free_bytes,
            free_inodes_after_stage=free_inodes,
            staged_bytes=staged_bytes,
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        with self._lock:
            prepared = self._prepared.get(context.runtime_id)
        if prepared is None:
            raise AdapterLaunchError(
                "Aeon storage/artifact preflight was not retained", process_absent=True
            )
        capability = prepared["capability"]
        if capability.runtime_adapter == "local-docker":
            if current_runtime_state() is not None or remote_state() is not None:
                raise RuntimeError("an exact Aeon Qwen lifecycle receipt already exists")
        elif remote_state(context.lease.run_dir) is not None:
            raise RuntimeError("this exact Aeon Qwen lifecycle receipt already exists")
        heartbeat = _StartupHeartbeat(context)
        heartbeat.start()
        try:
            def startup_check() -> None:
                context.startup_check()
                heartbeat.check()

            if capability.runtime_adapter == "local-docker":
                state = start_local_runtime(
                    prepared["lease"],
                    prepared["environment"],
                    package_root=PACKAGE_ROOT,
                    model_dir=prepared["model_dir"],
                    container_name=prepared["plan"].container_name,
                    image=prepared["image"],
                    port=prepared["plan"].health_port,
                    artifact_identity=prepared["artifact"],
                    image_identity=prepared["image_id"],
                    image_size_bytes=prepared["image_size"],
                    progress_check=startup_check,
                    heartbeat_promoter=lambda: heartbeat.promote(
                        int(local_container_pid() or 0)
                    ),
                    coordinator_verify_func=False,
                    final_heartbeat_func=lambda pid, _note, *_args: heartbeat.promote(pid),
                )
            else:
                resources = prepared["remote_resources"]
                state = start_managed_remote_runtime(
                    capability,
                    prepared["manifest_sha256"],
                    prepared["source"],
                    prepared["lease"],
                    prepared["environment"],
                    container_name=resources["container_name"],
                    port=resources["remote_port"],
                    heartbeat_pid=heartbeat.promote,
                    artifact_cache=prepared["artifact_cache"],
                    progress_check=startup_check,
                )
            startup_check()
            pid = int(state["container_pid"])
            identity = f"{capability.key}:{state['container_id']}"
            endpoint = (
                LOCAL_ENDPOINT
                if capability.runtime_adapter == "local-docker"
                else f"http://127.0.0.1:{state['local_port']}/v1"
            )
            return LaunchResult(pid=pid, process_identity=identity, endpoint=endpoint)
        except BaseException as exc:
            run_dir = context.lease.run_dir
            liveness = (
                qwen_runtime_liveness()
                if capability.runtime_adapter == "local-docker"
                else remote_runtime_liveness(run_dir)
            )
            receipt_absent = (
                current_runtime_state() is None
                if capability.runtime_adapter == "local-docker"
                else remote_state(run_dir) is None
            )
            if liveness == "gone" and receipt_absent:
                raise AdapterLaunchError(
                    f"Aeon exact runtime launch failed before process creation: {exc}",
                    process_absent=True,
                ) from exc
            raise
        finally:
            heartbeat.close()

    @staticmethod
    def _state_matches(runtime: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
        identity = runtime.get("process_identity")
        expected_identity = (
            f"{state.get('runtime_capability_key')}:{state.get('container_id')}"
        )
        return identity == expected_identity and all(
            state.get(key) == runtime.get(key)
            for key in ("claim_id", "owner", "host", "physical_gpu", "gpu_uuid", "run_dir")
        )

    @staticmethod
    def _precontainer_intent_matches(
        runtime: Mapping[str, Any], state: Mapping[str, Any]
    ) -> bool:
        """Match the broker record to one PID-less per-runtime launch intent."""

        runtime_id = runtime.get("runtime_id")
        run_dir = state.get("run_dir")
        return (
            isinstance(runtime_id, str)
            and _RUNTIME_ID.fullmatch(runtime_id) is not None
            and isinstance(run_dir, str)
            and Path(run_dir).name == runtime_id
            and runtime.get("state") == "quarantined"
            and runtime.get("process_absent") == 0
            and state.get("phase") in {"starting", "releasing"}
            and runtime.get("pid") is None
            and runtime.get("process_identity") is None
            and runtime.get("endpoint") is None
            and all(
                state.get(field) is None
                for field in (
                    "container_id",
                    "container_pid",
                    "tunnel_nonce",
                    "tunnel_pid",
                    "tunnel_create_time",
                )
            )
            and all(
                state.get(key) == runtime.get(key)
                for key in (
                    "claim_id",
                    "owner",
                    "host",
                    "physical_gpu",
                    "gpu_uuid",
                    "run_dir",
                )
            )
        )

    @staticmethod
    def _uncommitted_container_intent_matches(
        runtime: Mapping[str, Any], state: Mapping[str, Any]
    ) -> bool:
        """Match one quarantined PID-bound launch awaiting atomic recovery."""

        runtime_id = runtime.get("runtime_id")
        run_dir = state.get("run_dir")
        container_id = state.get("container_id")
        container_pid = state.get("container_pid")
        return (
            isinstance(runtime_id, str)
            and _RUNTIME_ID.fullmatch(runtime_id) is not None
            and isinstance(run_dir, str)
            and Path(run_dir).name == runtime_id
            and runtime.get("state") == "quarantined"
            and runtime.get("process_absent") == 0
            and state.get("phase") in {"starting", "releasing"}
            and runtime.get("pid") is None
            and runtime.get("process_identity") is None
            and runtime.get("endpoint") is None
            and (
                container_id is None
                or (
                    isinstance(container_id, str)
                    and re.fullmatch(r"[a-f0-9]{64}", container_id) is not None
                )
            )
            and (state.get("phase") != "releasing" or container_id is not None)
            and type(container_pid) is int
            and container_pid > 1
            and all(
                state.get(field) is None
                for field in (
                    "tunnel_nonce",
                    "tunnel_pid",
                    "tunnel_create_time",
                )
            )
            and all(
                state.get(key) == runtime.get(key)
                for key in (
                    "claim_id",
                    "owner",
                    "host",
                    "physical_gpu",
                    "gpu_uuid",
                    "run_dir",
                )
            )
        )

    @staticmethod
    def _local_uncommitted_intent_matches(
        runtime: Mapping[str, Any], state: Mapping[str, Any]
    ) -> bool:
        """Match one exact local launch that Fleet never durably published."""

        container_id = state.get("container_id")
        container_pid = state.get("container_pid")
        return (
            runtime.get("host") == LOCAL_HOST
            and runtime.get("state") == "quarantined"
            and runtime.get("pid") is None
            and runtime.get("process_identity") is None
            and runtime.get("endpoint") is None
            and state.get("phase")
            in {"preparing", "preflight", "launching", "ready", "releasing"}
            and (
                container_id is None
                or (
                    isinstance(container_id, str)
                    and re.fullmatch(r"[a-f0-9]{64}", container_id) is not None
                )
            )
            and (
                container_pid is None
                or (
                    type(container_pid) is int
                    and container_pid > 1
                )
            )
            and all(
                state.get(key) == runtime.get(key)
                for key in (
                    "claim_id",
                    "owner",
                    "host",
                    "physical_gpu",
                    "gpu_uuid",
                    "run_dir",
                )
            )
        )

    @classmethod
    def _recover_local_uncommitted_intent(
        cls, runtime: Mapping[str, Any], state: Mapping[str, Any]
    ) -> bool:
        """Stop and clear only the exact receipt-bound local launch."""

        if not cls._local_uncommitted_intent_matches(runtime, state):
            return False
        if not stop_qwen_runtime(allow_lost_lease=True):
            return False
        releasing = current_runtime_state()
        if (
            releasing is None
            or releasing.get("phase") != "releasing"
            or releasing.get("container_pid") is not None
            or releasing.get("scratch_cleaned") is not True
            or not all(
                releasing.get(key) == runtime.get(key)
                for key in (
                    "claim_id",
                    "owner",
                    "host",
                    "physical_gpu",
                    "gpu_uuid",
                    "run_dir",
                )
            )
            or qwen_runtime_liveness() != "gone"
        ):
            return False
        clear_runtime_state()
        return current_runtime_state() is None

    @staticmethod
    def _endpoint_ready(state: Mapping[str, Any]) -> bool:
        port = state.get("local_port")
        if type(port) is not int or not 1024 <= port <= 65535:
            return False
        try:
            response = requests.get(
                f"http://127.0.0.1:{port}/v1/models",
                timeout=(2, 10),
                allow_redirects=False,
                proxies={"http": "", "https": ""},
            )
            if response.status_code != 200 or len(response.content) > 256 * 1024:
                return False
            payload = response.json()
            served = state.get("served_name")
            if served is None:
                served = (state.get("deploy_environment") or {}).get("AEON_SERVED_NAME")
            return any(
                isinstance(item, dict) and item.get("id") == served
                for item in payload.get("data", [])
            )
        except (OSError, requests.RequestException, TypeError, ValueError):
            return False

    def _probe_after_tunnel_restore_failure(
        self,
        runtime: Mapping[str, Any],
        run_dir: str,
    ) -> ProbeResult:
        """Classify a failed repair only after re-proving the exact runtime."""

        try:
            state = remote_state(run_dir)
            if state is None or not self._state_matches(runtime, state):
                return ProbeResult(
                    ProbeState.UNKNOWN,
                    False,
                    False,
                    "Aeon tunnel recovery changed the runtime receipt",
                )
            liveness = remote_runtime_liveness(run_dir)
            if liveness in {"gone", "exited"}:
                return ProbeResult(
                    ProbeState.ABSENT,
                    False,
                    True,
                    f"Aeon runtime became {liveness} during tunnel recovery",
                )
            if liveness != "active":
                return ProbeResult(
                    ProbeState.UNKNOWN,
                    False,
                    False,
                    "Aeon runtime liveness changed during tunnel recovery",
                )
            pid = state.get("container_pid")
            if type(pid) is not int or pid != runtime.get("pid"):
                return ProbeResult(
                    ProbeState.UNKNOWN,
                    False,
                    False,
                    "Aeon runtime PID changed during tunnel recovery",
                )
            tunnel_status = tunnel_liveness(state)
            if tunnel_status == "gone":
                return ProbeResult(
                    ProbeState.STARTING,
                    True,
                    False,
                    "Aeon exact tunnel recovery will be retried",
                )
            if tunnel_status != "active":
                return ProbeResult(
                    ProbeState.UNKNOWN,
                    False,
                    False,
                    "Aeon tunnel identity changed during recovery",
                )
            if state.get("phase") != "ready" or not self._endpoint_ready(state):
                return ProbeResult(
                    ProbeState.STARTING,
                    True,
                    False,
                    "Aeon exact restored tunnel is still becoming ready",
                )
            return ProbeResult(
                ProbeState.READY,
                True,
                False,
                "Aeon exact tunnel recovered despite an earlier health error",
            )
        except Exception:
            return ProbeResult(
                ProbeState.UNKNOWN,
                False,
                False,
                "Aeon tunnel recovery state is ambiguous",
            )

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        is_local = runtime.get("host") == LOCAL_HOST
        run_dir = str(runtime.get("run_dir") or "")
        state = (
            current_runtime_state()
            if is_local
            else remote_state(run_dir)
        )
        if state is None:
            # Receipt disappearance is not process-absence evidence.  Without
            # the immutable container/PID binding there is no exact target for
            # either a local Docker inspection or the worker's lock-held
            # absence transaction, so keep the coordinator claim quarantined.
            return ProbeResult(
                ProbeState.UNKNOWN,
                False,
                False,
                "Aeon runtime receipt is absent; exact process absence is unproven",
            )
        if is_local and self._local_uncommitted_intent_matches(runtime, state):
            try:
                recovered = self._recover_local_uncommitted_intent(runtime, state)
            except Exception:
                recovered = False
            if recovered:
                return ProbeResult(
                    ProbeState.ABSENT,
                    False,
                    True,
                    "Aeon exact local uncommitted startup was safely recovered",
                    prelaunch_cleanup_verified=True,
                )
            return ProbeResult(
                ProbeState.UNKNOWN,
                False,
                False,
                "Aeon local uncommitted startup recovery is incomplete",
            )
        if not is_local and self._precontainer_intent_matches(runtime, state):
            try:
                capability, _current_manifest = qwen_runtime_capability(
                    str(state["runtime_capability_key"]), require_enabled=False
                )
                recovered = stop_managed_remote_runtime(
                    capability,
                    str(state["runtime_capability_manifest_sha256"]),
                    str(state["source_manifest_sha256"]),
                    release_reason="recover exact Fleet pre-container runtime",
                    release_claim=False,
                    run_dir=run_dir,
                    require_unlaunched=True,
                )
            except Exception:
                recovered = False
            if recovered:
                return ProbeResult(
                    ProbeState.ABSENT,
                    False,
                    True,
                    "Aeon exact pre-container worker was atomically recovered",
                    prelaunch_cleanup_verified=True,
                )
            return ProbeResult(
                ProbeState.UNKNOWN,
                False,
                False,
                "Aeon pre-container worker recovery is unsupported or ambiguous",
            )
        if not is_local and self._uncommitted_container_intent_matches(
            runtime, state
        ):
            try:
                recovered = recover_remote_uncommitted_intent(run_dir)
            except Exception:
                recovered = False
            if recovered:
                return ProbeResult(
                    ProbeState.ABSENT,
                    False,
                    True,
                    "Aeon exact uncommitted startup was safely recovered",
                    prelaunch_cleanup_verified=True,
                )
            return ProbeResult(
                ProbeState.UNKNOWN,
                False,
                False,
                "Aeon uncommitted startup recovery is incomplete",
            )
        if not self._state_matches(runtime, state):
            return ProbeResult(ProbeState.UNKNOWN, False, False, "Aeon runtime receipt identity changed")
        liveness = (
            qwen_runtime_liveness()
            if is_local
            else remote_runtime_liveness(run_dir)
        )
        if liveness in {"gone", "exited"}:
            return ProbeResult(ProbeState.ABSENT, False, True, f"Aeon runtime is {liveness}")
        if liveness != "active":
            return ProbeResult(ProbeState.UNKNOWN, False, False, "Aeon runtime liveness is ambiguous")
        pid = state.get("container_pid")
        if type(pid) is not int or pid != runtime.get("pid"):
            return ProbeResult(ProbeState.UNKNOWN, False, False, "Aeon runtime PID identity changed")
        if not is_local:
            tunnel_status = tunnel_liveness(state)
            if tunnel_status == "gone":
                try:
                    state = restore_managed_remote_tunnel(run_dir)
                except Exception:
                    return self._probe_after_tunnel_restore_failure(
                        runtime,
                        run_dir,
                    )
                if not self._state_matches(runtime, state):
                    return ProbeResult(
                        ProbeState.UNKNOWN,
                        False,
                        False,
                        "Aeon restored tunnel receipt identity changed",
                    )
            elif tunnel_status != "active":
                return ProbeResult(
                    ProbeState.UNKNOWN,
                    False,
                    False,
                    "Aeon loopback tunnel identity changed",
                )
        if state.get("phase") != "ready" or not self._endpoint_ready(state):
            return ProbeResult(ProbeState.STARTING, True, False, "Aeon runtime is still becoming ready")
        return ProbeResult(ProbeState.READY, True, False, "Aeon exact runtime is ready")

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        is_local = runtime.get("host") == LOCAL_HOST
        state = (
            current_runtime_state()
            if is_local
            else remote_state(str(runtime.get("run_dir") or ""))
        )
        if state is None:
            return StopResult(
                False,
                False,
                "Aeon runtime receipt is absent; exact process absence is unproven",
            )
        precontainer_intent = (
            not is_local and self._precontainer_intent_matches(runtime, state)
        )
        if not precontainer_intent and not self._state_matches(runtime, state):
            return StopResult(False, False, "Aeon runtime receipt identity changed")
        try:
            if is_local:
                stopped = stop_qwen_runtime(allow_lost_lease=True)
                if stopped:
                    remaining = current_runtime_state()
                    if (
                        remaining is None
                        or remaining.get("phase") != "releasing"
                        or remaining.get("scratch_cleaned") is not True
                        or not self._state_matches(runtime, remaining)
                    ):
                        return StopResult(False, False, "Aeon local stop journal changed")
                    clear_runtime_state()
            else:
                capability, _current_manifest = qwen_runtime_capability(
                    str(state["runtime_capability_key"]), require_enabled=False
                )
                stopped = stop_managed_remote_runtime(
                    capability,
                    str(state["runtime_capability_manifest_sha256"]),
                    str(state["source_manifest_sha256"]),
                    release_reason=reason,
                    release_claim=False,
                    run_dir=str(runtime["run_dir"]),
                    require_unlaunched=precontainer_intent,
                )
            return StopResult(
                process_absent=bool(stopped),
                identity_matched=True,
                note=reason if stopped else "Aeon exact runtime is still stopping",
            )
        except Exception as exc:
            raise QwenRuntimeError("Aeon exact runtime stop failed") from exc

    @staticmethod
    def _remote_cleanup(host: str, path: str) -> int:
        hostname = EXPECTED_HOSTNAMES.get(host)
        script = """
import json
import os
import shutil
import stat
import sys

path, expected = sys.argv[1:3]
assert os.uname().nodename == expected
root = os.lstat(path)
assert stat.S_ISDIR(root.st_mode)
assert root.st_uid == os.geteuid()
assert not root.st_mode & 0o077
total = 0
for base, directories, files in os.walk(path, topdown=True, followlinks=False):
    for name in directories + files:
        target = os.path.join(base, name)
        metadata = os.lstat(target)
        assert metadata.st_uid == os.geteuid()
        assert not stat.S_ISLNK(metadata.st_mode)
        assert stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
        total += metadata.st_size
shutil.rmtree(path)
print(json.dumps({"reclaimed": total}, sort_keys=True))
"""
        result = subprocess.run(
            [
                "/usr/bin/ssh", "-T", "-o", "BatchMode=yes", "-o",
                "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes", "-o",
                "IdentitiesOnly=yes", f"aday@{network_address(host)}", shlex.join([
                    _REMOTE_PYTHON, "-c", script, path, str(hostname),
                ]),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 or len(result.stdout) > 4096:
            raise QwenRuntimeError("worker scratch cleanup was not verified")
        try:
            return int(json.loads(result.stdout)["reclaimed"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise QwenRuntimeError("worker scratch cleanup response is malformed") from exc

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        if runtime.get("host") == LOCAL_HOST:
            return StorageFinalizationResult(True, True, 0, "canonical host retained")
        scratch = storage.get("scratch_path")
        runtime_id = runtime.get("runtime_id")
        run_dir = runtime.get("run_dir")
        physical_gpu = runtime.get("physical_gpu")
        if (
            not isinstance(runtime_id, str)
            or _RUNTIME_ID.fullmatch(runtime_id) is None
            or not isinstance(run_dir, str)
            or Path(run_dir).parent != _FLEET_RUN_ROOT
            or Path(run_dir).name != runtime_id
            or runtime.get("host") not in EXPECTED_HOSTNAMES
            or runtime.get("host") == LOCAL_HOST
            or type(physical_gpu) is not int
            or runtime.get("process_absent") != 1
            or type(runtime.get("process_absent")) is not int
            or runtime.get("state") not in {"stopped", "lost"}
            or scratch != run_dir
            or not isinstance(scratch, str)
        ):
            raise QwenRuntimeError("worker scratch manifest identity changed")
        resources = fleet_remote_runtime_resources(
            run_dir, physical_gpu, host=str(runtime["host"])
        )
        if resources["runtime_id"] != runtime_id or remote_state(run_dir) is not None:
            raise QwenRuntimeError("worker runtime is not safe for exact finalization")
        try:
            self._remote_metrics(str(runtime["host"]), scratch, create=False)
        except FileNotFoundError:
            return StorageFinalizationResult(True, True, 0, "worker scratch already absent")
        reclaimed = self._remote_cleanup(str(runtime["host"]), scratch)
        return StorageFinalizationResult(
            True, True, reclaimed, "exact Aeon worker scratch removed"
        )


def create_fleet_adapter() -> AeonQwenFleetAdapter:
    return AeonQwenFleetAdapter()
