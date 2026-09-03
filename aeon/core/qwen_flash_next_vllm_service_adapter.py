"""Exact local Fleet service lifecycle for a qualified Flash-Next vLLM binding.

The checked-in profile is disabled.  This adapter neither creates a binding nor
downloads artifacts; after explicit promotion it revalidates the same immutable
runtime/checkpoint/image/qualification contract used by the canary.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading
import time
from typing import Any, Mapping

import requests

from fleet_compute.adapters import AdapterLaunchError, RuntimeContext
from fleet_compute.models import (
    ComputeProfile,
    LaunchResult,
    Lease,
    ProbeResult,
    ProbeState,
    StopResult,
    StorageFinalizationResult,
    StoragePreparationResult,
)

from aeon.core import qwen_flash_next_vllm_contract as contract
from aeon.core import qwen_flash_next_vllm_service_binding as binding_module
from aeon.scripts import qwen_flash_next_vllm_canary_worker as canary
from aeon.scripts import qualify_qwen38_flash_next_vllm as qualify


PROFILE_ID = binding_module.PROFILE_ID
ADAPTER_NAME = "aeon-qwen38-flash-next-vllm-service-v1"
HOST_PORT = 18048
CONTAINER_PORT = canary.CONTAINER_PORT
TASK_MEMORY_GIB = canary.TASK_MEMORY_GIB
SERVICE_MEMORY_HEADROOM_BYTES = 8 * 1024**3
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_NAME = "vllm-service-container.json"
PROCESS_PREFIX = "aeon-vllm-service"
# Production shares one least-busy Fleet router with the 27B fallback. Keep the
# artifact-specific qualification name first while also accepting the common
# rolling-compatible token used across the pool.
SERVICE_SERVED_MODEL = "Qwen3.8-27B-ARA-NVFP4-MTP"
SERVICE_SERVED_MODELS = (contract.SERVED_MODEL, SERVICE_SERVED_MODEL)
# The immediately preceding production adapter launched the same qualified
# artifact with its artifact-specific alias. Retain that one exact source
# identity solely so restart recovery can reconstruct and verify its command
# before recovery or teardown; new launches always use SERVICE_SERVED_MODEL.
PRE_COMPATIBILITY_ADAPTER_SOURCES = frozenset(
    {"b3cf89676a9ac6304ae6261621f335210fe94ff255db8efaf83043a654cd0b9e"}
)
# The immediately preceding adapter predates the coordinator's reviewed Docker
# child-process attribution label. Keep its exact label shape only for controlled
# recovery/teardown; all new containers carry ``owner=aday`` so GPU worker PIDs
# whose exec-time environment was sanitized remain attributable to their lease.
PRE_OWNER_LABEL_ADAPTER_SOURCES = frozenset(
    {"85fde65ea341e1d1ea5f1d047bc58809b0e64d060c47fb9528bc078cb0407787"}
)
_RUNTIME = re.compile(r"^fr-[0-9a-f]{32}$")
_CONTAINER = re.compile(r"^[0-9a-f]{64}$")
_PROCESS = re.compile(
    rf"^{PROCESS_PREFIX}:(fr-[0-9a-f]{{32}}):([0-9a-f]{{64}}):"
    r"([0-9a-f]{64}):([0-9]+):([0-9]+)$"
)


class VllmServiceAdapterError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_artifact_identity(
    binding: binding_module.VllmServiceBinding,
) -> dict[str, str]:
    return {
        **binding.artifact_identity,
        "service_adapter_source": _sha256(Path(__file__)),
        "service_binding_source": _sha256(Path(binding_module.__file__)),
    }


def _name(runtime_id: str) -> str:
    return f"aeon-vllm-service-{runtime_id}"


def _labels(context: RuntimeContext, binding: binding_module.VllmServiceBinding) -> dict[str, str]:
    labels = {
        "aeon.fleet.profile": PROFILE_ID,
        "aeon.fleet.runtime": context.runtime_id,
        "aeon.fleet.claim-sha256": hashlib.sha256(
            context.lease.claim_id.encode()
        ).hexdigest(),
        "aeon.fleet.binding": binding.sha256,
        "aeon.fleet.image": str(binding.raw["derived_image_digest"]),
        "aeon.fleet.checkpoint": str(binding.raw["checkpoint_manifest_sha256"]),
    }
    profile = getattr(context, "profile", None)
    identity = getattr(profile, "artifact_identity", {})
    source = identity.get("service_adapter_source") if isinstance(identity, Mapping) else None
    if source not in PRE_OWNER_LABEL_ADAPTER_SOURCES:
        labels["owner"] = "aday"
    return labels


def _inspect(reference: str) -> Mapping[str, Any] | None:
    return canary._inspect(reference)


def _start_ticks(pid: int) -> int:
    payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    end = payload.rfind(")")
    if end < 0:
        raise VllmServiceAdapterError("container process stat is malformed")
    return int(payload[end + 2 :].split()[19])


def _validate_service_cgroup(
    path: Path,
    pid: int,
    *,
    require_settled_headroom: bool,
) -> None:
    """Validate the hard cap without rejecting harmless load-time reclaim.

    The canary starts from a fresh cgroup and still requires zero ``max``
    events. A long-lived production load can transiently touch the same hard
    limit while the kernel reclaims checkpoint page cache. After load, require
    zero OOM signals and a real settled margin instead of treating those
    non-OOM reclaim counters as permanent service failure.
    """

    try:
        memory_max = int((path / "memory.max").read_text(encoding="ascii"))
        memory_current = int(
            (path / "memory.current").read_text(encoding="ascii")
        )
        processes = {
            int(value)
            for value in (path / "cgroup.procs")
            .read_text(encoding="ascii")
            .split()
        }
        events: dict[str, int] = {}
        for line in (path / "memory.events").read_text(encoding="ascii").splitlines():
            fields = line.split()
            if len(fields) != 2 or not fields[1].isdigit() or fields[0] in events:
                raise VllmServiceAdapterError(
                    "service cgroup memory events are malformed"
                )
            events[fields[0]] = int(fields[1])
    except (OSError, ValueError) as exc:
        raise VllmServiceAdapterError(
            "service cgroup resource attestation failed"
        ) from exc
    expected_max = TASK_MEMORY_GIB * 1024**3
    if (
        memory_max != expected_max
        or pid not in processes
        or any(
            events.get(name) != 0
            for name in ("oom", "oom_kill", "oom_group_kill")
        )
        or (
            require_settled_headroom
            and memory_current > expected_max - SERVICE_MEMORY_HEADROOM_BYTES
        )
    ):
        raise VllmServiceAdapterError(
            "service cgroup cap/OOM/headroom attestation failed"
        )


def _service_task_cgroup(
    pid: int,
    container_id: str,
    *,
    require_settled_headroom: bool = True,
) -> Path:
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise VllmServiceAdapterError(
            "service container cgroup identity is unreadable"
        ) from exc
    unified = [line.split(":", 2)[2] for line in lines if line.startswith("0::")]
    if len(unified) != 1:
        raise VllmServiceAdapterError(
            "service container has no exact cgroup-v2 path"
        )
    relative = PurePosixPath(unified[0])
    if (
        not relative.is_absolute()
        or ".." in relative.parts
        or not any(
            container_id in part or container_id[:12] in part
            for part in relative.parts
        )
    ):
        raise VllmServiceAdapterError(
            "service container task cgroup identity changed"
        )
    try:
        path = Path("/sys/fs/cgroup").joinpath(*relative.parts[1:]).resolve(
            strict=True
        )
    except OSError as exc:
        raise VllmServiceAdapterError(
            "service container cgroup path is unavailable"
        ) from exc
    _validate_service_cgroup(
        path,
        pid,
        require_settled_headroom=require_settled_headroom,
    )
    return path


def _atomic_receipt(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _served_models_for_context(context: RuntimeContext) -> tuple[str, ...]:
    profile = getattr(context, "profile", None)
    identity = getattr(profile, "artifact_identity", {})
    source = identity.get("service_adapter_source") if isinstance(identity, Mapping) else None
    if source in PRE_COMPATIBILITY_ADAPTER_SOURCES:
        return (contract.SERVED_MODEL,)
    return SERVICE_SERVED_MODELS


def _served_model_for_context(context: RuntimeContext) -> str:
    return _served_models_for_context(context)[0]


def _compatibility_alias_ready(base_url: str, aliases: tuple[str, ...]) -> None:
    """Prove the pool alias works without requiring vLLM to echo that alias.

    vLLM accepts every value passed to ``--served-model-name`` but reports the
    first configured value as the canonical response model.  The response must
    still identify one of the exact configured aliases; requiring it to echo the
    requested compatibility alias leaves an otherwise healthy dual-alias server
    stuck in startup forever.
    """

    result = qualify._bounded_json(
        requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": SERVICE_SERVED_MODEL,
                "messages": [
                    {"role": "user", "content": "Reply with exactly READY-AEON."}
                ],
                "max_tokens": 16,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": False,
            },
            timeout=qualify.REQUEST_TIMEOUT_SECONDS,
        )
    )
    if result.get("model") not in aliases:
        raise qualify.VllmQualificationError(
            "compatibility response model identity changed"
        )
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise qualify.VllmQualificationError(
            "compatibility-alias response is malformed"
        )
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or "READY-AEON" not in content:
        raise qualify.VllmQualificationError(
            "port is live but compatibility-alias readiness failed"
        )


def _semantic_ready(context: RuntimeContext) -> Mapping[str, Any]:
    base_url = f"http://127.0.0.1:{HOST_PORT}"
    aliases = _served_models_for_context(context)
    if len(aliases) == 1:
        return qualify.semantic_ready(base_url, aliases[0])
    health = requests.get(f"{base_url}/health", timeout=10)
    if health.status_code != 200:
        raise qualify.VllmQualificationError("vLLM health check failed")
    models = qualify._bounded_json(
        requests.get(f"{base_url}/v1/models", timeout=10)
    )
    observed = [
        item.get("id")
        for item in models.get("data", [])
        if isinstance(item, Mapping)
    ]
    if observed != list(aliases):
        raise qualify.VllmQualificationError(
            "vLLM production served-model aliases changed"
        )
    _compatibility_alias_ready(base_url, aliases)
    return {
        "health": True,
        "served_models": list(aliases),
        "semantic_probe": True,
    }


def _request(context: RuntimeContext, binding: binding_module.VllmServiceBinding) -> dict[str, Any]:
    return {
        "runtime_id": context.runtime_id,
        "gpu_uuid": context.lease.gpu_uuid,
        "claim_id": context.lease.claim_id,
        "checkpoint_path": str(binding.checkpoint_path),
        "checkpoint_manifest_path": str(binding.checkpoint_manifest_path),
        "checkpoint_manifest_sha256": str(
            binding.raw["checkpoint_manifest_sha256"]
        ),
        "derived_image_digest": str(binding.raw["derived_image_digest"]),
        "derived_image_config_digest": str(
            binding.raw["derived_image_config_digest"]
        ),
        "derived_image_archive_path": str(binding.derived_image_archive_path),
        "derived_image_archive_sha256": str(
            binding.raw["derived_image_archive_sha256"]
        ),
        "served_model": _served_model_for_context(context),
        "runtime": contract.expected_runtime(),
    }


def _create_command(
    context: RuntimeContext,
    binding: binding_module.VllmServiceBinding,
    evidence: Path,
    *,
    image_reference: str | None = None,
) -> list[str]:
    request = _request(context, binding)
    claim_hash = hashlib.sha256(context.lease.claim_id.encode()).hexdigest()
    gpu_hash = hashlib.sha256(context.lease.gpu_uuid.encode()).hexdigest()
    command = [
        "container",
        "create",
        "--name",
        _name(context.runtime_id),
        "--pull=never",
        "--user",
        f"{os.geteuid()}:{os.getegid()}",
        "--entrypoint",
        "python3",
        "--init=false",
        "--restart=no",
        "--memory",
        f"{TASK_MEMORY_GIB}g",
        "--memory-swap",
        f"{TASK_MEMORY_GIB}g",
        "--shm-size",
        "64g",
        "--pids-limit",
        "4096",
        "--ipc",
        "private",
        "--ulimit",
        "memlock=-1:-1",
        "--network",
        "bridge",
        "--publish",
        f"127.0.0.1:{HOST_PORT}:{CONTAINER_PORT}",
        "--gpus",
        f"device={context.lease.gpu_uuid}",
        "--env",
        f"CUDA_VISIBLE_DEVICES={context.lease.gpu_uuid}",
        "--env",
        f"GPU_AGENT_CLAIM_ID={context.lease.claim_id}",
        "--env",
        f"GPU_MEM_LIMIT_GB={contract.VRAM_CAP_GIB}",
        "--env",
        f"AEON_RUNTIME_ID={context.runtime_id}",
        "--env",
        "AEON_ENGINE_ATTESTATION=1",
        "--env",
        "AEON_CANARY_ARM=mtp_on",
        "--env",
        "AEON_MTP_ENABLED=1",
        "--env",
        f"AEON_CHECKPOINT_MANIFEST_SHA256={binding.raw['checkpoint_manifest_sha256']}",
        "--env",
        f"AEON_DERIVED_IMAGE_CONFIG_DIGEST={binding.raw['derived_image_config_digest']}",
        "--env",
        f"AEON_LEASE_CLAIM_SHA256={claim_hash}",
        "--env",
        f"AEON_LEASE_GPU_UUID_SHA256={gpu_hash}",
        "--env",
        f"AEON_CANARY_HOST={contract.HOST}",
        "--env",
        f"AEON_CANARY_PHYSICAL_GPU={contract.PHYSICAL_GPU}",
        "--env",
        "AEON_CANARY_EXCLUSIVE=1",
        "--env",
        f"AEON_CHECKPOINT_REPOSITORY={contract.CHECKPOINT_REPOSITORY}",
        "--env",
        f"AEON_CHECKPOINT_REVISION={contract.CHECKPOINT_REVISION}",
        "--env",
        f"AEON_BASE_IMAGE_AMD64_DIGEST={contract.BASE_IMAGE_AMD64_DIGEST}",
        "--env",
        f"AEON_SERVED_MODEL={_served_model_for_context(context)}",
        "--env",
        "VLLM_PLE_CPU_OFFLOAD=1",
        "--env",
        "VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800",
        "--env",
        "VLLM_USE_V2_MODEL_RUNNER=0",
        "--env",
        "OMP_NUM_THREADS=1",
        "--env",
        "MKL_NUM_THREADS=1",
        "--env",
        "OPENBLAS_NUM_THREADS=1",
        "--env",
        "OMP_WAIT_POLICY=PASSIVE",
        "--env",
        "TORCH_CUDA_ARCH_LIST=12.0f",
        "--mount",
        f"type=bind,src={binding.checkpoint_path},dst=/model,readonly",
        "--mount",
        f"type=bind,src={PACKAGE_ROOT},dst=/aeon-source,readonly",
        "--mount",
        f"type=bind,src={evidence},dst=/evidence",
    ]
    for key, value in _labels(context, binding).items():
        command.extend(("--label", f"{key}={value}"))
    server_command = canary.server_command(request, mtp_enabled=True)
    served_name_index = server_command.index("--served-model-name")
    server_command[served_name_index + 1 : served_name_index + 2] = list(
        _served_models_for_context(context)
    )
    command.extend(
        (
            image_reference
            or canary._oci_load_digest(binding.derived_image_archive_path),
            "/aeon-source/aeon/scripts/qwen_flash_next_container_supervisor.py",
            "--output",
            "/evidence/cuda-memory.json",
            "--freeze",
            "/evidence/freeze",
            "--context",
            "/evidence/runtime-context.json",
            "--runtime-id",
            context.runtime_id,
            "--arm",
            "tuned_mtp_on_winner",
            "--claim-sha256",
            claim_hash,
            "--gpu-uuid",
            context.lease.gpu_uuid,
            "--checkpoint-tree-sha256",
            str(binding.raw["checkpoint_manifest_sha256"]),
            "--",
            *server_command,
        )
    )
    return command


def _verify(
    item: Mapping[str, Any],
    context: RuntimeContext,
    binding: binding_module.VllmServiceBinding,
    *,
    running: bool,
) -> tuple[str, int]:
    container_id = str(item.get("Id") or "")
    config = item.get("Config")
    host_config = item.get("HostConfig")
    state = item.get("State")
    mounts = item.get("Mounts")
    network = item.get("NetworkSettings")
    create = _create_command(context, binding, context.run_dir / "evidence")
    runnable_image = canary._oci_load_digest(binding.derived_image_archive_path)
    image_index = create.index(runnable_image)
    expected_command = create[image_index + 1 :]
    if (
        _CONTAINER.fullmatch(container_id) is None
        or item.get("Name") != "/" + _name(context.runtime_id)
        or not isinstance(config, Mapping)
        or not isinstance(host_config, Mapping)
        or not isinstance(state, Mapping)
        or not isinstance(mounts, list)
        or not isinstance(network, Mapping)
        or config.get("Image") != runnable_image
        or config.get("User") != f"{os.geteuid()}:{os.getegid()}"
        or config.get("Entrypoint") != ["python3"]
        or config.get("Cmd") != expected_command
        or host_config.get("Memory") != TASK_MEMORY_GIB * 1024**3
        or host_config.get("MemorySwap") != TASK_MEMORY_GIB * 1024**3
        or host_config.get("ShmSize") != 64 * 1024**3
        or host_config.get("PidsLimit") != 4096
        or host_config.get("IpcMode") != "private"
        or host_config.get("CapAdd") not in (None, [])
        or host_config.get("CapDrop") not in (None, [])
        or host_config.get("SecurityOpt") not in (None, [])
        or state.get("Running") is not running
    ):
        raise VllmServiceAdapterError("exact production container changed")
    labels = config.get("Labels")
    environment = set(config.get("Env") or ())
    expected_environment = {
        f"CUDA_VISIBLE_DEVICES={context.lease.gpu_uuid}",
        f"GPU_AGENT_CLAIM_ID={context.lease.claim_id}",
        f"GPU_MEM_LIMIT_GB={contract.VRAM_CAP_GIB}",
        f"AEON_RUNTIME_ID={context.runtime_id}",
        "AEON_ENGINE_ATTESTATION=1",
        "AEON_CANARY_ARM=mtp_on",
        "AEON_MTP_ENABLED=1",
        f"AEON_CHECKPOINT_MANIFEST_SHA256={binding.raw['checkpoint_manifest_sha256']}",
        f"AEON_DERIVED_IMAGE_CONFIG_DIGEST={binding.raw['derived_image_config_digest']}",
        "AEON_LEASE_CLAIM_SHA256="
        + hashlib.sha256(context.lease.claim_id.encode()).hexdigest(),
        "AEON_LEASE_GPU_UUID_SHA256="
        + hashlib.sha256(context.lease.gpu_uuid.encode()).hexdigest(),
        f"AEON_CANARY_HOST={contract.HOST}",
        f"AEON_CANARY_PHYSICAL_GPU={contract.PHYSICAL_GPU}",
        "AEON_CANARY_EXCLUSIVE=1",
        f"AEON_CHECKPOINT_REPOSITORY={contract.CHECKPOINT_REPOSITORY}",
        f"AEON_CHECKPOINT_REVISION={contract.CHECKPOINT_REVISION}",
        f"AEON_BASE_IMAGE_AMD64_DIGEST={contract.BASE_IMAGE_AMD64_DIGEST}",
        f"AEON_SERVED_MODEL={_served_model_for_context(context)}",
        "VLLM_USE_V2_MODEL_RUNNER=0",
    }
    if (
        not isinstance(labels, Mapping)
        or any(labels.get(key) != value for key, value in _labels(context, binding).items())
        or not expected_environment.issubset(environment)
    ):
        raise VllmServiceAdapterError("production container lease binding changed")
    model_mounts = [mount for mount in mounts if mount.get("Destination") == "/model"]
    device_requests = host_config.get("DeviceRequests") or []
    port_bindings = host_config.get("PortBindings")
    expected_bindings = [{"HostIp": "127.0.0.1", "HostPort": str(HOST_PORT)}]
    ports = network.get("Ports")
    bindings = ports.get(f"{CONTAINER_PORT}/tcp") if isinstance(ports, Mapping) else None
    if (
        len(model_mounts) != 1
        or model_mounts[0].get("Source") != str(binding.checkpoint_path)
        or model_mounts[0].get("RW") is not False
        or len(device_requests) != 1
        or not isinstance(device_requests[0], Mapping)
        or device_requests[0].get("DeviceIDs") != [context.lease.gpu_uuid]
        or device_requests[0].get("Capabilities") != [["gpu"]]
        # Docker records the requested loopback binding in HostConfig at create
        # time, but does not populate NetworkSettings.Ports until the container
        # has started.  Validate both representations when each is authoritative.
        or not isinstance(port_bindings, Mapping)
        or port_bindings.get(f"{CONTAINER_PORT}/tcp") != expected_bindings
        or (running and bindings != expected_bindings)
    ):
        raise VllmServiceAdapterError("GPU/model/loopback container binding changed")
    pid = state.get("Pid")
    if running and (type(pid) is not int or pid <= 1):
        raise VllmServiceAdapterError("production container PID is malformed")
    if not running and (type(pid) is not int or pid != 0):
        raise VllmServiceAdapterError("stopped production container PID is malformed")
    return container_id, int(pid or 0)


def _validate_saved_receipt(
    context: RuntimeContext,
    binding: binding_module.VllmServiceBinding,
    *,
    container_id: str,
    pid: int,
    ticks: int,
    cgroup: Path | None = None,
) -> None:
    """Bind a saved runtime identity to its immutable launch receipt.

    A stopped container has no live cgroup after a host restart, but its exact
    task-owned container ID/config plus this immutable receipt still proves
    which Fleet runtime created it.  Live recovery additionally supplies the
    currently verified cgroup path.
    """

    receipt = canary._private_file(
        context.run_dir / RECEIPT_NAME, maximum=64 * 1024
    )
    if (
        receipt.get("runtime_id") != context.runtime_id
        or receipt.get("container_id") != container_id
        or receipt.get("binding_sha256") != binding.sha256
        or receipt.get("pid") != pid
        or receipt.get("start_ticks") != ticks
        or (cgroup is not None and receipt.get("cgroup") != str(cgroup))
        or receipt.get("engine_runtime_sha256")
        != _sha256(context.run_dir / "evidence/engine-runtime.json")
    ):
        raise VllmServiceAdapterError("saved production receipt identity changed")


class AeonQwenFlashNextVllmServiceAdapter:
    def __init__(self) -> None:
        self._prepared: dict[str, binding_module.VllmServiceBinding] = {}
        self._contexts: dict[str, RuntimeContext] = {}
        self._lock = threading.RLock()
        self._verified_binding: binding_module.VllmServiceBinding | None = None

    def _load_verified_binding(self) -> binding_module.VllmServiceBinding:
        """Perform the expensive immutable closure proof once per broker process."""

        with self._lock:
            if self._verified_binding is None:
                self._verified_binding = binding_module.load_binding()
            return self._verified_binding

    @staticmethod
    def _validate_context(
        context: RuntimeContext,
        binding: binding_module.VllmServiceBinding,
        *,
        recovery: bool = False,
    ) -> None:
        lease = context.lease
        profile = context.profile
        observed_identity = dict(profile.artifact_identity)
        expected_identity = expected_artifact_identity(binding)
        observed_adapter_source = observed_identity.get("service_adapter_source")
        if recovery:
            # Recovery may be performed by a patched owning adapter after the
            # original source hash was durably snapshotted. It cannot relaunch;
            # every immutable model/runtime/binding identity must still match.
            observed_identity.pop("service_adapter_source", None)
            expected_identity.pop("service_adapter_source", None)
        if (
            _RUNTIME.fullmatch(context.runtime_id) is None
            or context.payload
            or context.job_id is not None
            or profile.profile_id != PROFILE_ID
            or profile.enabled is not True
            or profile.service_id != binding_module.SERVICE_ID
            or profile.serving_pool_id != "aeon-qwen38-ara-114688-v1"
            or profile.lane_max_replicas != 1
            or profile.max_replicas != 2
            or observed_identity != expected_identity
            or (
                recovery
                and (
                    not isinstance(observed_adapter_source, str)
                    or re.fullmatch(r"[0-9a-f]{64}", observed_adapter_source) is None
                )
            )
            or lease.host != contract.HOST
            or lease.physical_gpu != contract.PHYSICAL_GPU
            or lease.exclusive is not True
            or lease.vram_budget_gb != contract.VRAM_CAP_GIB
            or lease.memory_total_mib is None
            or lease.memory_total_mib < 94 * 1024
            or lease.memory_total_mib / 1024 - lease.vram_budget_gb < 6
            or context.scratch_path is not None
            or str(context.run_dir) != lease.run_dir
        ):
            raise VllmServiceAdapterError("production profile/lease binding changed")

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        binding = self._load_verified_binding()
        self._validate_context(context, binding)
        if _inspect(_name(context.runtime_id)) is not None:
            raise VllmServiceAdapterError("task-owned container name already exists")
        try:
            canary._ensure_image_loaded(_request(context, binding))
        except canary.CanaryWorkerError as exc:
            raise VllmServiceAdapterError(str(exc)) from exc
        metadata = context.run_dir.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise VllmServiceAdapterError("Fleet run directory is unsafe")
        evidence = context.run_dir / "evidence"
        evidence.mkdir(mode=0o700)
        values = os.statvfs(context.run_dir)
        with self._lock:
            self._prepared[context.runtime_id] = binding
            self._contexts[context.runtime_id] = context
        return StoragePreparationResult(
            context.scratch_path,
            str(metadata.st_dev),
            values.f_bavail * values.f_frsize,
            values.f_favail,
            0,
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        with self._lock:
            binding = self._prepared.get(context.runtime_id)
        if binding is None:
            raise AdapterLaunchError("production preflight is absent", process_absent=True)
        evidence = context.run_dir / "evidence"
        created_id: str | None = None
        try:
            created = canary._docker(_create_command(context, binding, evidence), timeout=180)
            created_id = created.stdout.strip()
            if created.returncode != 0 or _CONTAINER.fullmatch(created_id) is None:
                if _inspect(_name(context.runtime_id)) is None:
                    raise AdapterLaunchError("container create failed", process_absent=True)
                raise VllmServiceAdapterError("container create result is ambiguous")
            item = _inspect(created_id)
            if item is None:
                raise AdapterLaunchError("created container disappeared", process_absent=True)
            _verify(item, context, binding, running=False)
            started = canary._docker(["container", "start", created_id], timeout=120)
            if started.returncode != 0 or started.stdout.strip() != created_id:
                raise VllmServiceAdapterError("exact container did not start")
            item = _inspect(created_id)
            if item is None:
                raise AdapterLaunchError("started container disappeared", process_absent=True)
            _container_id, pid = _verify(item, context, binding, running=True)
            ticks = _start_ticks(pid)
            cgroup = canary._task_cgroup(pid, created_id)
            _atomic_receipt(
                evidence / "runtime-context.json",
                {
                    "container_id": created_id,
                    "container_pid": pid,
                    "cgroup_path": str(cgroup),
                    "container_pid_in_cgroup": True,
                },
            )
            deadline = time.monotonic() + context.profile.startup_timeout_seconds
            while time.monotonic() < deadline:
                context.startup_check()
                context.heartbeat(pid, "qualified vLLM Flash-Next is loading")
                item = _inspect(created_id)
                if item is None:
                    raise AdapterLaunchError("service container disappeared", process_absent=True)
                _verify(item, context, binding, running=True)
                try:
                    _semantic_ready(context)
                    cgroup = _service_task_cgroup(pid, created_id)
                    engine = canary._runtime_receipt(
                        _request(context, binding),
                        evidence,
                        container_id=created_id,
                        pid=pid,
                        mtp_enabled=True,
                    )
                    if engine["placement"] != {
                        "transformer_weights": "cuda",
                        "mtp_weights": "cuda",
                        "lm_head": "cuda",
                        "vision_weights": "cuda",
                        "ple_table": "cpu_worker_pinned_h2d",
                        "other_cpu_model_components": [],
                    }:
                        raise VllmServiceAdapterError("service placement differs from qualification")
                    receipt = {
                        "runtime_id": context.runtime_id,
                        "container_id": created_id,
                        "binding_sha256": binding.sha256,
                        "pid": pid,
                        "start_ticks": ticks,
                        "cgroup": str(cgroup),
                        "engine_runtime_sha256": _sha256(evidence / "engine-runtime.json"),
                    }
                    _atomic_receipt(context.run_dir / RECEIPT_NAME, receipt)
                    identity = f"{PROCESS_PREFIX}:{context.runtime_id}:{created_id}:{binding.sha256}:{pid}:{ticks}"
                    return LaunchResult(pid, identity, f"http://127.0.0.1:{HOST_PORT}/v1")
                except (
                    canary.CanaryWorkerError,
                    qualify.VllmQualificationError,
                    requests.RequestException,
                    OSError,
                ):
                    time.sleep(5)
            raise VllmServiceAdapterError("semantic service readiness timed out")
        except AdapterLaunchError:
            raise
        except BaseException as exc:
            absent = created_id is None or _inspect(created_id) is None
            raise AdapterLaunchError(
                f"vLLM service launch failed: {exc}", process_absent=absent
            ) from exc

    def _saved(
        self, runtime: Mapping[str, Any]
    ) -> tuple[RuntimeContext, binding_module.VllmServiceBinding, str, int, int]:
        match = _PROCESS.fullmatch(str(runtime.get("process_identity") or ""))
        if (
            match is None
            or match.group(1) != runtime.get("runtime_id")
            or runtime.get("profile_id") != PROFILE_ID
            or runtime.get("host") != contract.HOST
            or int(match.group(4)) != runtime.get("pid")
        ):
            raise VllmServiceAdapterError("saved production identity changed")
        with self._lock:
            context = self._contexts.get(match.group(1))
        recovered = context is None
        if recovered:
            context = self._context_from_runtime(runtime)
        binding = self._load_verified_binding()
        self._validate_context(context, binding, recovery=recovered)
        if binding.sha256 != match.group(3):
            raise VllmServiceAdapterError("production binding changed")
        return context, binding, match.group(2), int(match.group(4)), int(match.group(5))

    @staticmethod
    def _context_from_runtime(runtime: Mapping[str, Any]) -> RuntimeContext:
        """Reconstruct the exact durable lease contract after a Fleet restart."""

        try:
            raw_profile = json.loads(str(runtime["profile_json"]))
            raw_payload = json.loads(str(runtime["payload_json"]))
            if not isinstance(raw_profile, dict) or not isinstance(raw_payload, dict):
                raise ValueError("runtime profile/payload snapshot is malformed")
            profile = ComputeProfile.from_dict(raw_profile)
            lease = Lease(
                claim_id=str(runtime["claim_id"]),
                owner=str(runtime["owner"]),
                host=str(runtime["host"]),
                physical_gpu=int(runtime["physical_gpu"]),
                gpu_uuid=str(runtime["gpu_uuid"]),
                vram_budget_gb=float(runtime["vram_budget_gb"]),
                exclusive=bool(runtime["exclusive"]),
                run_dir=str(runtime["run_dir"]),
                model=(
                    str(runtime["gpu_model"])
                    if runtime.get("gpu_model") is not None
                    else None
                ),
                memory_total_mib=(
                    int(runtime["memory_total_mib"])
                    if runtime.get("memory_total_mib") is not None
                    else None
                ),
            )
            canonical = Path(str(runtime["canonical_output_path"]))
            scratch = runtime.get("scratch_path")
            if scratch is not None:
                scratch = str(scratch)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise VllmServiceAdapterError(
                f"durable runtime context is incomplete: {exc}"
            ) from exc
        return RuntimeContext(
            runtime_id=str(runtime["runtime_id"]),
            profile=profile,
            lease=lease,
            run_dir=Path(str(runtime["run_dir"])),
            payload=raw_payload,
            job_id=(str(runtime["job_id"]) if runtime.get("job_id") else None),
            scratch_path=scratch,
            canonical_output_path=canonical,
            heartbeat=lambda _pid, _note: None,
            startup_check=lambda: None,
        )

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        try:
            if (
                runtime.get("state") == "quarantined"
                and runtime.get("pid") is None
                and runtime.get("process_identity") is None
                and runtime.get("endpoint") is None
            ):
                # A Fleet restart has joined the bounded startup thread, so no
                # adapter launch can still be racing this audit. Reconstruct and
                # revalidate the durable claim-bound context before treating the
                # deterministic task-owned container namespace as authoritative.
                context = self._context_from_runtime(runtime)
                binding = self._load_verified_binding()
                self._validate_context(context, binding, recovery=True)
                receipt = context.run_dir / RECEIPT_NAME
                if receipt.exists():
                    raise VllmServiceAdapterError(
                        "pidless quarantine unexpectedly has a launch receipt"
                    )
                item = _inspect(_name(context.runtime_id))
                if item is not None:
                    state = item.get("State")
                    running = (
                        isinstance(state, Mapping)
                        and state.get("Running") is True
                    )
                    container_id, pid = _verify(
                        item, context, binding, running=running
                    )
                    if running:
                        # The previous broker was terminated only after joining
                        # its bounded startup transaction.  Before stopping the
                        # surviving task-owned container, bind its live PID and
                        # cgroup back to the attestation receipt written before
                        # the readiness loop.  A name or label match alone is not
                        # enough to authorize process interaction.
                        cgroup = _service_task_cgroup(
                            pid,
                            container_id,
                            require_settled_headroom=False,
                        )
                        attestation = canary._private_file(
                            context.run_dir / "evidence/runtime-context.json",
                            maximum=64 * 1024,
                        )
                        if (
                            attestation.get("container_id") != container_id
                            or attestation.get("container_pid") != pid
                            or attestation.get("cgroup_path") != str(cgroup)
                            or attestation.get("container_pid_in_cgroup") is not True
                        ):
                            raise VllmServiceAdapterError(
                                "pidless startup attestation identity changed"
                            )
                        stopped = canary._docker(
                            ["container", "stop", "--time", "30", container_id],
                            timeout=60,
                        )
                        if stopped.returncode != 0:
                            raise VllmServiceAdapterError(
                                "exact pidless startup container is still stopping"
                            )
                        item = _inspect(container_id)
                        if item is not None:
                            _verify(item, context, binding, running=False)
                    removed = canary._docker(
                        ["container", "rm", container_id], timeout=30
                    )
                    if removed.returncode != 0 or _inspect(container_id) is not None:
                        raise VllmServiceAdapterError(
                            "exact stopped quarantine container cleanup failed"
                        )
                if _inspect(_name(context.runtime_id)) is None:
                    return ProbeResult(
                        ProbeState.ABSENT,
                        False,
                        True,
                        "exact pidless production container namespace is absent",
                        prelaunch_cleanup_verified=True,
                    )
                raise VllmServiceAdapterError(
                    "pidless quarantine retains an exact task-owned container"
                )
            context, binding, container_id, pid, ticks = self._saved(runtime)
            item = _inspect(container_id)
            if item is None:
                return ProbeResult(ProbeState.ABSENT, False, True, "exact container is absent")
            state = item.get("State")
            running = isinstance(state, Mapping) and state.get("Running") is True
            _identity, observed_pid = _verify(
                item, context, binding, running=running
            )
            if not running:
                _validate_saved_receipt(
                    context,
                    binding,
                    container_id=container_id,
                    pid=pid,
                    ticks=ticks,
                )
                return ProbeResult(
                    ProbeState.ABSENT,
                    False,
                    True,
                    "exact production container is stopped after host recovery",
                )
            _semantic_ready(context)
            cgroup = _service_task_cgroup(pid, container_id)
            if (
                observed_pid != pid
                or _start_ticks(pid) != ticks
            ):
                raise VllmServiceAdapterError("container PID identity changed")
            _validate_saved_receipt(
                context,
                binding,
                container_id=container_id,
                pid=pid,
                ticks=ticks,
                cgroup=cgroup,
            )
            return ProbeResult(ProbeState.READY, True, False, "qualified vLLM Flash-Next is ready")
        except (
            VllmServiceAdapterError,
            canary.CanaryWorkerError,
            requests.RequestException,
            OSError,
        ) as exc:
            return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        try:
            context, binding, container_id, pid, ticks = self._saved(runtime)
            item = _inspect(container_id)
            if item is None:
                return StopResult(True, True, reason)
            state = item.get("State")
            running = isinstance(state, Mapping) and state.get("Running") is True
            _identity, observed_pid = _verify(
                item, context, binding, running=running
            )
            cgroup = None
            if running:
                if observed_pid != pid or _start_ticks(pid) != ticks:
                    raise VllmServiceAdapterError(
                        "production container PID identity changed"
                    )
                cgroup = _service_task_cgroup(pid, container_id)
            _validate_saved_receipt(
                context,
                binding,
                container_id=container_id,
                pid=pid,
                ticks=ticks,
                cgroup=cgroup,
            )
            if running:
                stopped = canary._docker(
                    ["container", "stop", "--time", "30", container_id],
                    timeout=60,
                )
                if stopped.returncode != 0:
                    return StopResult(False, True, "exact container is still stopping")
                item = _inspect(container_id)
            if item is not None:
                _verify(item, context, binding, running=False)
                removed = canary._docker(["container", "rm", container_id], timeout=30)
                if removed.returncode != 0:
                    return StopResult(False, True, "stopped exact container remains")
            return StopResult(_inspect(container_id) is None, True, reason)
        except (VllmServiceAdapterError, canary.CanaryWorkerError, OSError) as exc:
            return StopResult(False, False, str(exc))

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        if (
            runtime.get("host") != contract.HOST
            or storage.get("scratch_path") is not None
            or storage.get("canonical_output_path")
            != runtime.get("canonical_output_path")
        ):
            raise VllmServiceAdapterError("service storage identity changed")
        return StorageFinalizationResult(
            True,
            bool(runtime.get("process_absent")),
            0,
            "canonical .177 binding and runtime receipts retained; no automatic cleanup",
        )


def create_fleet_adapter() -> AeonQwenFlashNextVllmServiceAdapter:
    return AeonQwenFlashNextVllmServiceAdapter()
