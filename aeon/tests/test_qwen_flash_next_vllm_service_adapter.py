from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import call, patch

import pytest

from aeon.core import qwen_flash_next_vllm_contract as contract
from aeon.core import qwen_flash_next_vllm_service_adapter as service
from aeon.core import qwen_flash_next_vllm_service_binding as binding_module


@pytest.fixture(autouse=True)
def _exact_test_load_identity(monkeypatch):
    monkeypatch.setattr(
        service.canary, "_oci_load_digest", lambda _path: "sha256:" + "a" * 64
    )


def _fixture(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"image")
    manifest = checkpoint / "manifest.json"
    manifest.write_bytes(b"manifest")
    raw = {
        "qualification_receipt_sha256": "f" * 64,
        "checkpoint_manifest_sha256": "b" * 64,
        "derived_image_digest": "sha256:" + "a" * 64,
        "derived_image_config_digest": "a" * 64,
        "derived_image_archive_sha256": "d" * 64,
        "canary_artifact_identity": {
            name: "9" * 64
            for name in binding_module.CANARY_IDENTITY_FIELDS
        },
    }
    raw["canary_artifact_identity"].update(
        checkpoint_manifest=raw["checkpoint_manifest_sha256"],
        derived_image=raw["derived_image_digest"].removeprefix("sha256:"),
        derived_image_config=raw["derived_image_config_digest"],
        derived_image_archive=raw["derived_image_archive_sha256"],
    )
    bound = binding_module.VllmServiceBinding(
        path=tmp_path / "binding.json",
        sha256="c" * 64,
        qualification_receipt=tmp_path / "qualification.json",
        checkpoint_path=checkpoint,
        checkpoint_manifest_path=manifest,
        derived_image_archive_path=archive,
        raw=raw,
    )
    lease = SimpleNamespace(gpu_uuid="GPU-" + "1" * 32, claim_id="claim-exact")
    context = SimpleNamespace(
        runtime_id="fr-" + "2" * 32,
        lease=lease,
        run_dir=tmp_path / "run",
    )
    context.run_dir.mkdir()
    return context, bound


def test_production_command_reuses_exact_vllm_mtp3_contract(tmp_path: Path) -> None:
    context, bound = _fixture(tmp_path)
    command = service._create_command(context, bound, context.run_dir / "evidence")
    assert command.count("--gpus") == 1
    assert command[command.index("--gpus") + 1] == f"device={context.lease.gpu_uuid}"
    assert f"GPU_AGENT_CLAIM_ID={context.lease.claim_id}" in command
    assert f"GPU_MEM_LIMIT_GB={contract.VRAM_CAP_GIB}" in command
    assert "VLLM_PLE_CPU_OFFLOAD=1" in command
    assert "VLLM_PLE_FP8_CHECKPOINT=1" not in command
    assert "VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800" in command
    assert "VLLM_USE_V2_MODEL_RUNNER=0" in command
    assert "OMP_NUM_THREADS=1" in command
    assert "TORCH_CUDA_ARCH_LIST=12.0f" in command
    assert not any("PYTORCH_CUDA_ALLOC_CONF=" in item for item in command)
    assert "--cap-add" not in command
    assert command.count("--publish") == 1
    assert f"127.0.0.1:{service.HOST_PORT}:{service.CONTAINER_PORT}" in command
    served_index = command.index("--served-model-name")
    assert command[served_index + 1 : served_index + 3] == list(
        service.SERVICE_SERVED_MODELS
    )
    assert f"AEON_SERVED_MODEL={contract.SERVED_MODEL}" in command
    assert "--speculative-config" in command
    speculative = command[command.index("--speculative-config") + 1]
    assert json.loads(speculative) == contract.expected_runtime()[
        "speculative_config"
    ]
    assert "--tensor-parallel-size" in command
    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert "owner=aday" in command


def test_recovery_reconstructs_the_exact_pre_compatibility_alias(tmp_path: Path) -> None:
    context, bound = _fixture(tmp_path)
    context.profile = SimpleNamespace(
        artifact_identity={
            "service_adapter_source": next(
                iter(service.PRE_COMPATIBILITY_ADAPTER_SOURCES)
            )
        }
    )

    command = service._create_command(context, bound, context.run_dir / "evidence")

    assert command[command.index("--served-model-name") + 1] == contract.SERVED_MODEL
    assert f"AEON_SERVED_MODEL={contract.SERVED_MODEL}" in command


def test_recovery_reconstructs_the_exact_pre_owner_label_shape(tmp_path: Path) -> None:
    context, bound = _fixture(tmp_path)
    context.profile = SimpleNamespace(
        artifact_identity={
            "service_adapter_source": next(
                iter(service.PRE_OWNER_LABEL_ADAPTER_SOURCES)
            )
        }
    )

    command = service._create_command(context, bound, context.run_dir / "evidence")

    assert "owner=aday" not in command


def test_binding_closure_verification_is_single_flight_per_broker_process() -> None:
    adapter = service.AeonQwenFlashNextVllmServiceAdapter()
    binding = object()

    with patch.object(binding_module, "load_binding", return_value=binding) as load:
        assert adapter._load_verified_binding() is binding
        assert adapter._load_verified_binding() is binding

    load.assert_called_once_with()


def test_service_cgroup_allows_non_oom_load_reclaim_with_settled_headroom(
    tmp_path: Path,
) -> None:
    cap = service.TASK_MEMORY_GIB * 1024**3
    (tmp_path / "memory.max").write_text(str(cap))
    (tmp_path / "memory.current").write_text(
        str(cap - service.SERVICE_MEMORY_HEADROOM_BYTES)
    )
    (tmp_path / "cgroup.procs").write_text("1234\n")
    (tmp_path / "memory.events").write_text(
        "low 0\nhigh 0\nmax 134\noom 0\noom_kill 0\noom_group_kill 0\n"
    )

    service._validate_service_cgroup(
        tmp_path, 1234, require_settled_headroom=True
    )


def test_service_cgroup_rejects_oom_or_insufficient_settled_headroom(
    tmp_path: Path,
) -> None:
    cap = service.TASK_MEMORY_GIB * 1024**3
    (tmp_path / "memory.max").write_text(str(cap))
    (tmp_path / "memory.current").write_text(
        str(cap - service.SERVICE_MEMORY_HEADROOM_BYTES + 1)
    )
    (tmp_path / "cgroup.procs").write_text("1234\n")
    (tmp_path / "memory.events").write_text(
        "low 0\nhigh 0\nmax 1\noom 0\noom_kill 0\noom_group_kill 0\n"
    )
    with pytest.raises(service.VllmServiceAdapterError, match="headroom"):
        service._validate_service_cgroup(
            tmp_path, 1234, require_settled_headroom=True
        )

    (tmp_path / "memory.current").write_text(
        str(cap - service.SERVICE_MEMORY_HEADROOM_BYTES)
    )
    (tmp_path / "memory.events").write_text(
        "low 0\nhigh 0\nmax 1\noom 1\noom_kill 0\noom_group_kill 0\n"
    )
    with pytest.raises(service.VllmServiceAdapterError, match="OOM"):
        service._validate_service_cgroup(
            tmp_path, 1234, require_settled_headroom=True
        )


def test_dual_alias_readiness_accepts_vllm_canonical_response_model(
    tmp_path: Path,
) -> None:
    context, _bound = _fixture(tmp_path)
    context.profile = SimpleNamespace(artifact_identity={})
    health = SimpleNamespace(status_code=200)
    models = SimpleNamespace(
        status_code=200,
        content=json.dumps(
            {"data": [{"id": item} for item in service.SERVICE_SERVED_MODELS]}
        ).encode(),
    )
    response = SimpleNamespace(
        status_code=200,
        content=json.dumps(
            {
                "model": contract.SERVED_MODEL,
                "choices": [
                    {"message": {"role": "assistant", "content": "READY-AEON"}}
                ],
            }
        ).encode(),
    )

    with (
        patch.object(service.requests, "get", side_effect=[health, models]),
        patch.object(service.requests, "post", return_value=response) as post,
    ):
        result = service._semantic_ready(context)

    assert result["served_models"] == list(service.SERVICE_SERVED_MODELS)
    assert post.call_args.kwargs["json"]["model"] == service.SERVICE_SERVED_MODEL


def test_dual_alias_readiness_rejects_unknown_response_model(tmp_path: Path) -> None:
    context, _bound = _fixture(tmp_path)
    context.profile = SimpleNamespace(artifact_identity={})
    health = SimpleNamespace(status_code=200)
    models = SimpleNamespace(
        status_code=200,
        content=json.dumps(
            {"data": [{"id": item} for item in service.SERVICE_SERVED_MODELS]}
        ).encode(),
    )
    response = SimpleNamespace(
        status_code=200,
        content=json.dumps(
            {
                "model": "unexpected-model",
                "choices": [
                    {"message": {"role": "assistant", "content": "READY-AEON"}}
                ],
            }
        ).encode(),
    )

    with (
        patch.object(service.requests, "get", side_effect=[health, models]),
        patch.object(service.requests, "post", return_value=response),
        pytest.raises(
            service.qualify.VllmQualificationError,
            match="response model identity changed",
        ),
    ):
        service._semantic_ready(context)


def test_production_container_verification_rejects_gpu_uuid_drift(tmp_path: Path) -> None:
    context, bound = _fixture(tmp_path)
    create = service._create_command(context, bound, context.run_dir / "evidence")
    runnable_image = "sha256:" + bound.raw["derived_image_config_digest"]
    image_index = create.index(runnable_image)
    labels = service._labels(context, bound)
    environment = [
        create[index + 1]
        for index, item in enumerate(create[:image_index])
        if item == "--env"
    ]
    item = {
        "Id": "3" * 64,
        "Name": "/" + service._name(context.runtime_id),
        "Config": {
            "Image": runnable_image,
            "User": f"{service.os.geteuid()}:{service.os.getegid()}",
            "Entrypoint": ["python3"],
            "Cmd": create[image_index + 1 :],
            "Labels": labels,
            "Env": environment,
        },
        "HostConfig": {
            "Memory": service.TASK_MEMORY_GIB * 1024**3,
            "MemorySwap": service.TASK_MEMORY_GIB * 1024**3,
            "ShmSize": 64 * 1024**3,
            "PidsLimit": 4096,
            "IpcMode": "private",
            "CapAdd": None,
            "CapDrop": None,
            "SecurityOpt": None,
            "PortBindings": {
                f"{service.CONTAINER_PORT}/tcp": [
                    {"HostIp": "127.0.0.1", "HostPort": str(service.HOST_PORT)}
                ]
            },
            "DeviceRequests": [
                {"DeviceIDs": ["GPU-" + "9" * 32], "Capabilities": [["gpu"]]}
            ],
        },
        "State": {"Running": True, "Pid": 1234},
        "Mounts": [
            {"Destination": "/model", "Source": str(bound.checkpoint_path), "RW": False}
        ],
        "NetworkSettings": {
            "Ports": {
                f"{service.CONTAINER_PORT}/tcp": [
                    {"HostIp": "127.0.0.1", "HostPort": str(service.HOST_PORT)}
                ]
            }
        },
    }
    try:
        service._verify(item, context, bound, running=True)
    except service.VllmServiceAdapterError as exc:
        assert "GPU/model/loopback" in str(exc)
    else:
        raise AssertionError("changed GPU UUID was accepted")


def test_created_container_uses_host_config_for_loopback_binding(tmp_path: Path) -> None:
    context, bound = _fixture(tmp_path)
    create = service._create_command(context, bound, context.run_dir / "evidence")
    runnable_image = "sha256:" + bound.raw["derived_image_config_digest"]
    image_index = create.index(runnable_image)
    item = {
        "Id": "3" * 64,
        "Name": "/" + service._name(context.runtime_id),
        "Config": {
            "Image": runnable_image,
            "User": f"{service.os.geteuid()}:{service.os.getegid()}",
            "Entrypoint": ["python3"],
            "Cmd": create[image_index + 1 :],
            "Labels": service._labels(context, bound),
            "Env": [
                create[index + 1]
                for index, value in enumerate(create[:image_index])
                if value == "--env"
            ],
        },
        "HostConfig": {
            "Memory": service.TASK_MEMORY_GIB * 1024**3,
            "MemorySwap": service.TASK_MEMORY_GIB * 1024**3,
            "ShmSize": 64 * 1024**3,
            "PidsLimit": 4096,
            "IpcMode": "private",
            "CapAdd": None,
            "CapDrop": None,
            "SecurityOpt": None,
            "PortBindings": {
                f"{service.CONTAINER_PORT}/tcp": [
                    {"HostIp": "127.0.0.1", "HostPort": str(service.HOST_PORT)}
                ]
            },
            "DeviceRequests": [
                {"DeviceIDs": [context.lease.gpu_uuid], "Capabilities": [["gpu"]]}
            ],
        },
        "State": {"Running": False, "Pid": 0},
        "Mounts": [
            {"Destination": "/model", "Source": str(bound.checkpoint_path), "RW": False}
        ],
        "NetworkSettings": {"Ports": {}},
    }
    service._verify(item, context, bound, running=False)


def test_production_context_requires_the_one_rtx_promotion_shape(
    tmp_path: Path,
) -> None:
    context, bound = _fixture(tmp_path)
    context.payload = {}
    context.job_id = None
    context.scratch_path = None
    context.lease.host = contract.HOST
    context.lease.physical_gpu = contract.PHYSICAL_GPU
    context.lease.exclusive = True
    context.lease.vram_budget_gb = contract.VRAM_CAP_GIB
    context.lease.memory_total_mib = 96 * 1024
    context.lease.run_dir = str(context.run_dir)
    context.profile = SimpleNamespace(
        profile_id=service.PROFILE_ID,
        enabled=True,
        service_id=binding_module.SERVICE_ID,
        serving_pool_id="aeon-qwen38-ara-114688-v1",
        lane_max_replicas=1,
        max_replicas=2,
        artifact_identity=service.expected_artifact_identity(bound),
    )
    service.AeonQwenFlashNextVllmServiceAdapter._validate_context(context, bound)
    context.profile.max_replicas = 1
    with pytest.raises(service.VllmServiceAdapterError, match="binding changed"):
        service.AeonQwenFlashNextVllmServiceAdapter._validate_context(context, bound)


def test_setup_registers_disabled_production_adapter() -> None:
    setup = (Path(__file__).resolve().parents[2] / "setup.py").read_text()
    assert (
        "aeon-qwen38-flash-next-vllm-service-v1 = "
        "aeon.core.qwen_flash_next_vllm_service_adapter:create_fleet_adapter"
    ) in setup


def test_stop_never_signals_container_after_identity_mismatch(tmp_path: Path) -> None:
    context, bound = _fixture(tmp_path)
    adapter = service.AeonQwenFlashNextVllmServiceAdapter()
    runtime = {"runtime_id": context.runtime_id}
    with (
        patch.object(
            adapter,
            "_saved",
            return_value=(context, bound, "3" * 64, 1234, 5678),
        ),
        patch.object(service, "_inspect", return_value={"Id": "changed"}),
        patch.object(service.canary, "_docker") as docker,
    ):
        stopped = adapter.stop(runtime, reason="preempted")
    assert stopped.process_absent is False
    assert stopped.identity_matched is False
    docker.assert_not_called()


def test_probe_treats_exact_receipted_stopped_container_as_absent(
    tmp_path: Path,
) -> None:
    context, bound = _fixture(tmp_path)
    adapter = service.AeonQwenFlashNextVllmServiceAdapter()
    runtime = {"runtime_id": context.runtime_id}
    container_id = "3" * 64
    with (
        patch.object(
            adapter,
            "_saved",
            return_value=(context, bound, container_id, 1234, 5678),
        ),
        patch.object(
            service,
            "_inspect",
            return_value={"State": {"Running": False, "Pid": 0}},
        ),
        patch.object(service, "_verify", return_value=(container_id, 0)),
        patch.object(service, "_validate_saved_receipt") as receipt,
        patch.object(service, "_semantic_ready") as semantic_ready,
    ):
        result = adapter.probe(runtime)

    assert result.state is service.ProbeState.ABSENT
    assert result.process_absent is True
    assert result.process_identity_verified is False
    receipt.assert_called_once_with(
        context,
        bound,
        container_id=container_id,
        pid=1234,
        ticks=5678,
    )
    semantic_ready.assert_not_called()


def test_stop_removes_exact_receipted_container_already_stopped_by_reboot(
    tmp_path: Path,
) -> None:
    context, bound = _fixture(tmp_path)
    adapter = service.AeonQwenFlashNextVllmServiceAdapter()
    runtime = {"runtime_id": context.runtime_id}
    container_id = "3" * 64
    stopped_item = {"State": {"Running": False, "Pid": 0}}
    with (
        patch.object(
            adapter,
            "_saved",
            return_value=(context, bound, container_id, 1234, 5678),
        ),
        patch.object(service, "_inspect", side_effect=[stopped_item, None, None]),
        patch.object(service, "_verify", return_value=(container_id, 0)),
        patch.object(service, "_validate_saved_receipt") as receipt,
        patch.object(
            service.canary,
            "_docker",
            return_value=SimpleNamespace(returncode=0),
        ) as docker,
    ):
        result = adapter.stop(runtime, reason="host recovered")

    assert result.process_absent is True
    assert result.identity_matched is True
    receipt.assert_called_once_with(
        context,
        bound,
        container_id=container_id,
        pid=1234,
        ticks=5678,
        cgroup=None,
    )
    assert docker.call_args_list == [
        call(["container", "rm", container_id], timeout=30)
    ]


def test_probe_proves_exact_pidless_quarantine_absence(tmp_path: Path) -> None:
    context, bound = _fixture(tmp_path)
    context.payload = {}
    context.job_id = None
    context.scratch_path = None
    context.lease.host = contract.HOST
    context.lease.physical_gpu = contract.PHYSICAL_GPU
    context.lease.exclusive = True
    context.lease.vram_budget_gb = contract.VRAM_CAP_GIB
    context.lease.memory_total_mib = 96 * 1024
    context.lease.run_dir = str(context.run_dir)
    profile = SimpleNamespace(
        profile_id=service.PROFILE_ID,
        enabled=True,
        service_id=binding_module.SERVICE_ID,
        serving_pool_id="aeon-qwen38-ara-114688-v1",
        lane_max_replicas=1,
        max_replicas=2,
        artifact_identity=service.expected_artifact_identity(bound),
    )
    reconstructed = SimpleNamespace(
        runtime_id=context.runtime_id,
        profile=profile,
        lease=context.lease,
        run_dir=context.run_dir,
        payload={},
        job_id=None,
        scratch_path=None,
    )
    runtime = {
        "runtime_id": context.runtime_id,
        "state": "quarantined",
        "pid": None,
        "process_identity": None,
        "endpoint": None,
    }
    adapter = service.AeonQwenFlashNextVllmServiceAdapter()
    with (
        patch.object(adapter, "_context_from_runtime", return_value=reconstructed),
        patch.object(binding_module, "load_binding", return_value=bound),
        patch.object(service, "_inspect", return_value=None),
    ):
        result = adapter.probe(runtime)
    assert result.state is service.ProbeState.ABSENT
    assert result.process_absent is True
    assert result.prelaunch_cleanup_verified is True


def test_probe_cleans_exact_attested_running_pidless_startup(
    tmp_path: Path,
) -> None:
    context, bound = _fixture(tmp_path)
    context.payload = {}
    context.job_id = None
    context.scratch_path = None
    context.lease.host = contract.HOST
    context.lease.physical_gpu = contract.PHYSICAL_GPU
    context.lease.exclusive = True
    context.lease.vram_budget_gb = contract.VRAM_CAP_GIB
    context.lease.memory_total_mib = 96 * 1024
    context.lease.run_dir = str(context.run_dir)
    context.profile = SimpleNamespace(
        profile_id=service.PROFILE_ID,
        enabled=True,
        service_id=binding_module.SERVICE_ID,
        serving_pool_id="aeon-qwen38-ara-114688-v1",
        lane_max_replicas=1,
        max_replicas=2,
        artifact_identity=service.expected_artifact_identity(bound),
    )
    container_id = "3" * 64
    pid = 1234
    cgroup = Path("/sys/fs/cgroup/fleet/exact")
    runtime = {
        "runtime_id": context.runtime_id,
        "state": "quarantined",
        "pid": None,
        "process_identity": None,
        "endpoint": None,
    }
    adapter = service.AeonQwenFlashNextVllmServiceAdapter()
    running = {"State": {"Running": True}}
    stopped = {"State": {"Running": False}}
    docker_result = SimpleNamespace(returncode=0)
    with (
        patch.object(adapter, "_context_from_runtime", return_value=context),
        patch.object(binding_module, "load_binding", return_value=bound),
        patch.object(
            service,
            "_inspect",
            side_effect=[running, stopped, None, None],
        ),
        patch.object(service, "_verify", side_effect=[(container_id, pid), (container_id, 0)]),
        patch.object(service, "_service_task_cgroup", return_value=cgroup),
        patch.object(
            service.canary,
            "_private_file",
            return_value={
                "container_id": container_id,
                "container_pid": pid,
                "cgroup_path": str(cgroup),
                "container_pid_in_cgroup": True,
            },
        ),
        patch.object(service.canary, "_docker", return_value=docker_result) as docker,
    ):
        result = adapter.probe(runtime)

    assert result.state is service.ProbeState.ABSENT
    assert result.prelaunch_cleanup_verified is True
    assert docker.call_args_list[0].args[0] == [
        "container", "stop", "--time", "30", container_id
    ]
    assert docker.call_args_list[1].args[0] == ["container", "rm", container_id]


def test_launch_publishes_attestation_context_before_readiness(tmp_path: Path) -> None:
    source = Path(service.__file__).read_text()
    context_write = source.index('evidence / "runtime-context.json"')
    readiness_loop = source.index("deadline = time.monotonic()")
    assert context_write < readiness_loop
