#!/usr/bin/env python3
"""Hermetic safety regressions for the local-only Qwen lifecycle."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aeon.core import gpu_queue
from aeon.core import qwen_capabilities
from aeon.core import qwen_runtime as runtime
from aeon.core.compute_profile import (
    COMFYUI_PROFILE,
    QWEN38_VLLM_PROFILE,
    ComputeProfile,
)


def completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def changed(value):
    if value is None:
        return "unexpected"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + "-changed"
    if isinstance(value, list):
        return [*copy.deepcopy(value), "unexpected"]
    if isinstance(value, dict):
        result = copy.deepcopy(value)
        result["unexpected"] = True
        return result
    raise AssertionError(f"no mutation for {type(value)!r}")


class HermeticDockerRootMixin:
    """Keep Docker CLI config receipts out of the user's durable runtime root."""

    def setUp(self):
        super().setUp()
        self._docker_runtime_temp = tempfile.TemporaryDirectory()
        self._docker_runtime_patch = patch.object(
            runtime, "RUNTIME_ROOT", Path(self._docker_runtime_temp.name)
        )
        self._docker_runtime_patch.start()

    def tearDown(self):
        self._docker_runtime_patch.stop()
        self._docker_runtime_temp.cleanup()
        super().tearDown()


def lease(root: Path | None = None, **changes):
    owner = "bc-aeon-hermetic-owner"
    root = root or runtime.RUNTIME_ROOT
    capability, capability_manifest_sha256 = (
        qwen_capabilities.active_qwen_runtime_capability()
    )
    value = {
        "claim_id": "gc-20260821T000000Z-deadbeef",
        "owner": owner,
        "project": gpu_queue.PROJECT,
        "purpose": "hermetic Qwen",
        "host": runtime.LOCAL_COORD_HOST,
        "physical_gpu": 0,
        "gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "memory_total_mib": 97887,
        "vram_budget_gb": runtime.QWEN_PLANNED_VRAM_GB,
        "vram_budget_mib": round(runtime.QWEN_PLANNED_VRAM_GB * 1024),
        "exclusive": True,
        "run_dir": str(root / f"aeon-qwen38-vllm-{owner}"),
        "compute_profile": QWEN38_VLLM_PROFILE.key,
        "min_host_memory_gb": 96.0,
        "min_host_commit_gb": 96.0,
        "min_disk_free_gb": 32.0,
        "min_shm_free_gb": 16.0,
        "runtime_capability_key": capability.key,
        "runtime_capability_manifest_sha256": capability_manifest_sha256,
        "runtime_adapter": capability.runtime_adapter,
    }
    value.update(changes)
    return value


def inventory_for(value=None, **target_changes):
    value = value or lease()
    claim = {
        "claim_id": value["claim_id"],
        "owner": value["owner"],
        "gpu_uuid": value["gpu_uuid"],
        "run_dir": value["run_dir"],
        "vram_budget_mib": value["vram_budget_mib"],
        # gpu_coord.py persists this in SQLite and exposes status claims as
        # canonical INTEGER 0/1, unlike the reserve response's JSON boolean.
        "exclusive": 1,
    }
    target = {
        "host": value["host"],
        "physical_gpu": value["physical_gpu"],
        "uuid": value["gpu_uuid"],
        "acl": "OPEN",
        "state": "RESERVED_RUNNING",
        "vast_watchdog_active": True,
        "memory_total_mib": 97887,
        "host_memory_available_mib": 120 * 1024,
        "host_commit_headroom_mib": 120 * 1024,
        "host_disk_available_mib": 100 * 1024,
        "host_shm_available_mib": 32 * 1024,
        "claims": [claim],
    }
    target.update(target_changes)
    return [target]


def make_model(base: Path) -> runtime.ArtifactIdentity:
    root = base / "model"
    root.mkdir(mode=0o700)
    payloads = {
        "BUILD_MANIFEST.json": json.dumps({"complete": True, "status": "validated"}),
        "config.json": "{}",
        "model.safetensors.index.json": "{}",
        "model-00001.safetensors": "immutable-test-shard",
    }
    for name, payload in payloads.items():
        path = root / name
        path.write_text(payload, encoding="utf-8")
        path.chmod(0o600)
    sums = "".join(
        f"{hashlib.sha256(payload.encode()).hexdigest()}  {name}\n"
        for name, payload in payloads.items()
    )
    (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
    (root / "SHA256SUMS").chmod(0o600)
    return runtime.load_artifact_identity(
        root,
        command_runner=lambda *_args, **_kwargs: completed(),
    )


def add_model_verification_sidecar(model_dir: Path) -> Path:
    sums_sha256 = hashlib.sha256((model_dir / "SHA256SUMS").read_bytes()).hexdigest()
    marker = model_dir / ".podcast-sha256-verified"
    marker.write_bytes(f"{sums_sha256}\n".encode("ascii"))
    marker.chmod(0o600)
    return marker


def make_source_tree(base: Path) -> Path:
    """Create the exact immutable host-launch closure used by source tests."""

    root = base / "source"
    root.mkdir(mode=0o700)
    for relative in runtime.SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        current = path.parent
        while current != root:
            current.chmod(0o700)
            current = current.parent
        path.write_text(f"hermetic release input: {relative}\n", encoding="utf-8")
        path.chmod(0o600)
    write_source_manifest(root)
    return root


def write_source_manifest(root: Path) -> None:
    manifest = root / runtime.SOURCE_MANIFEST_FILE
    manifest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest.write_bytes(
        b"".join(
            hashlib.sha256((root / relative).read_bytes()).hexdigest().encode("ascii")
            + b"  "
            + relative.encode("utf-8")
            + b"\n"
            for relative in runtime.SOURCE_FILES
        )
    )
    manifest.chmod(0o600)


def deploy_environment(image: str = "aeon_vllm:latest"):
    return {
        "AEON_DEPLOY_PLAN": json.dumps(
            {
                "entry_name": "Qwen3.8-27B-ARA-NVFP4-MTP",
                "tier": "solo",
                "image": image,
                "container_name": "aeon_qwen_test",
                "health_port": 8033,
                "nodes": [
                    {
                        "container": "aeon_qwen_test",
                        "devices": "0",
                        "port": 8033,
                        "ctx": runtime.QWEN_CONTEXT_TOKENS,
                        "cpu_offload_gib": 0.0,
                    }
                ],
            }
        ),
        "AEON_SERVED_NAME": "Qwen3.8-27B-ARA-NVFP4-MTP",
        "AEON_GPU_MEM_UTIL": "0.415",
        "AEON_LLM_VRAM_BUDGET_GB": "48.7",
        "AEON_MAX_NUM_SEQS": "1",
        "AEON_MAX_NUM_BATCHED": "32768",
        "AEON_VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
        "AEON_KV_QUANT": "fp8_per_token_head",
        "AEON_MTP_METHOD": "mtp",
        "AEON_MTP_NMAX": "3",
        "AEON_MTP_SELECTION_MANIFEST": "data/qwen38_mtp_selection.json",
    }


def inspect_state(tmp: Path):
    sources = {}
    for destination, name in (
        ("/usr/local/bin/fleet-low-priority", "wrapper"),
        ("/workspace/aeon_runtime/sitecustomize.py", "sitecustomize.py"),
        ("/models", "models"),
    ):
        path = tmp / name
        if name == "models":
            path.mkdir(mode=0o700)
        else:
            path.write_text(name, encoding="utf-8")
            path.chmod(0o600)
        metadata = path.lstat()
        sources[destination] = {
            "source": str(path),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
        }
    value = lease()
    return {
        **value,
        "expected_hostname": runtime.LOCAL_COORD_HOSTNAME,
        "container_id": "b" * 64,
        "container_name": "aeon_qwen_test",
        "image_id": qwen_capabilities.STANDARD_IMAGE_ID,
        "wrapper_sha256": "e" * 64,
        "docker_sha256": "f" * 64,
        "local_port": 8033,
        "remote_port": 8033,
        "gpu_uuid": value["gpu_uuid"],
        "container_command": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
        "container_environment": {
            "PATH": "/usr/bin",
            "TMPDIR": runtime.QWEN_CONTAINER_TMPDIR,
            "GPU_AGENT_CLAIM_ID": value["claim_id"],
            "AEON_RUNTIME_CAPABILITY_KEY": value["runtime_capability_key"],
            "AEON_RUNTIME_CAPABILITY_MANIFEST_SHA256": value[
                "runtime_capability_manifest_sha256"
            ],
        },
        "launch_nonce": "c" * 64,
        "container_labels": {
            "com.bc_aeon.component": "qwen38-vllm",
            "com.bc_aeon.claim": value["claim_id"],
            "com.bc_aeon.launch-nonce": "c" * 64,
            "com.bc_aeon.launch-spec": "d" * 64,
            "com.bc_aeon.runtime-capability": value["runtime_capability_key"],
            "com.bc_aeon.runtime-capability-manifest": value[
                "runtime_capability_manifest_sha256"
            ],
        },
        "container_mounts": sources,
        "image_base_exposed_ports": {"8000/tcp": {}},
        "container_tmpfs_options": runtime._container_tmpfs_options(
            executable=True
        ),
        "teardown_only": False,
    }


def inspect_payload(state, *, running=True, name=None):
    cache = (
        "rw,exec,nosuid,nodev,size=8589934592,uid="
        f"{os.geteuid()},gid={os.getegid()},mode=0700"
    )
    mounts = [
        {
            "Type": "bind",
            "Source": receipt["source"],
            "Destination": destination,
            "Mode": "",
            "RW": False,
            "Propagation": "rprivate",
        }
        for destination, receipt in state["container_mounts"].items()
    ]
    mounts.append(
        {
            "Type": "tmpfs",
            "Source": "",
            "Destination": "/workspace/cache",
            "Mode": "",
            "RW": True,
            "Propagation": "",
        }
    )
    host_mounts = [
        {
            "Type": "bind",
            "Source": receipt["source"],
            "Target": destination,
            "ReadOnly": True,
            "Consistency": "",
        }
        for destination, receipt in state["container_mounts"].items()
    ]
    runtime_port = f"{state['remote_port']}/tcp"
    port_bindings = {
        runtime_port: [
            {"HostIp": "127.0.0.1", "HostPort": str(state["local_port"])}
        ]
    }
    exposed_ports = {**state["image_base_exposed_ports"], runtime_port: {}}
    network_ports = {
        **{port: None for port in state["image_base_exposed_ports"]},
        **port_bindings,
    }
    return {
        "Id": state["container_id"],
        "Image": state["image_id"],
        "Name": name or f"/{state['container_name']}",
        "Path": "/usr/local/bin/fleet-low-priority",
        "Args": state["container_command"],
        "Platform": "linux",
        "Config": {
            "Hostname": state["container_name"],
            "Domainname": "",
            "User": f"{os.geteuid()}:{os.getegid()}",
            "AttachStdin": False,
            "AttachStdout": False,
            "AttachStderr": False,
            "ExposedPorts": exposed_ports,
            "Tty": False,
            "OpenStdin": False,
            "StdinOnce": False,
            "Entrypoint": ["/usr/local/bin/fleet-low-priority"],
            "Cmd": state["container_command"],
            "Env": [f"{key}={value}" for key, value in state["container_environment"].items()],
            "Labels": dict(state["container_labels"]),
            "WorkingDir": "",
            "Healthcheck": None,
            "ArgsEscaped": False,
            "Image": state["image_id"],
            "Volumes": None,
            "OnBuild": None,
            "StopSignal": "",
            "StopTimeout": None,
            "Shell": None,
        },
        "State": {"Running": running, "Pid": 4321 if running else 0},
        "HostConfig": {
            "Binds": None,
            "ContainerIDFile": str(Path(state["run_dir"]) / "container.cid"),
            "PortBindings": port_bindings,
            "DeviceRequests": [
                {
                    "Driver": "",
                    "Count": 0,
                    "DeviceIDs": [state["gpu_uuid"]],
                    "Capabilities": [["gpu"]],
                    "Options": {},
                }
            ],
            "OomScoreAdj": 1000,
            "CpuShares": 2,
            "BlkioWeight": 10,
            "PidsLimit": 1024,
            "Memory": 0,
            "NanoCpus": 0,
            "CgroupParent": "",
            "BlkioWeightDevice": [],
            "BlkioDeviceReadBps": [],
            "BlkioDeviceWriteBps": [],
            "BlkioDeviceReadIOps": [],
            "BlkioDeviceWriteIOps": [],
            "CpuPeriod": 0,
            "CpuQuota": 0,
            "CpuRealtimePeriod": 0,
            "CpuRealtimeRuntime": 0,
            "CpusetCpus": "",
            "CpusetMems": "",
            "Devices": [],
            "DeviceCgroupRules": None,
            "MemoryReservation": 0,
            "MemorySwap": 0,
            "MemorySwappiness": None,
            "OomKillDisable": None,
            "Ulimits": None,
            "CpuCount": 0,
            "CpuPercent": 0,
            "IOMaximumIOps": 0,
            "IOMaximumBandwidth": 0,
            "IpcMode": "private",
            "CgroupnsMode": "private",
            "Cgroup": "",
            "PidMode": "",
            "UTSMode": "",
            "UsernsMode": "",
            "ShmSize": 8 * 1024**3,
            "ReadonlyRootfs": True,
            "Privileged": False,
            "AutoRemove": False,
            "PublishAllPorts": False,
            "CapAdd": None,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "NetworkMode": "bridge",
            "Runtime": "runc",
            "Isolation": "",
            "VolumeDriver": "",
            "VolumesFrom": None,
            "ConsoleSize": [0, 0],
            "Dns": [],
            "DnsOptions": [],
            "DnsSearch": [],
            "ExtraHosts": None,
            "GroupAdd": None,
            "Links": None,
            "Tmpfs": {"/workspace/cache": cache},
            "Mounts": host_mounts,
            "LogConfig": {"Type": "local", "Config": {"max-file": "3", "max-size": "10m"}},
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "MaskedPaths": [
                "/proc/asound",
                "/proc/acpi",
                "/proc/kcore",
                "/proc/keys",
                "/proc/latency_stats",
                "/proc/timer_list",
                "/proc/timer_stats",
                "/proc/sched_debug",
                "/proc/scsi",
                "/sys/firmware",
                "/sys/devices/virtual/powercap",
            ],
            "ReadonlyPaths": [
                "/proc/bus",
                "/proc/fs",
                "/proc/irq",
                "/proc/sys",
                "/proc/sysrq-trigger",
            ],
            "Init": False,
            "UseApiSocket": False,
            "Annotations": None,
        },
        "Mounts": mounts,
        "NetworkSettings": {
            "SandboxID": "3" * 64,
            "SandboxKey": "/var/run/docker/netns/0123456789ab",
            "Ports": network_ports,
            "Networks": {
                "bridge": {
                    "IPAMConfig": None,
                    "Links": None,
                    "Aliases": None,
                    "DriverOpts": None,
                    "NetworkID": "1" * 64,
                    "EndpointID": "2" * 64,
                    "Gateway": "172.17.0.1",
                    "IPAddress": "172.17.0.2",
                    "IPPrefixLen": 16,
                    "DNSNames": [
                        state["container_name"],
                        state["container_id"][:12],
                    ],
                }
            },
        },
    }


def durable_runtime_state(tmp: Path):
    """Build a complete, schema-valid runtime receipt without live resources."""

    state = inspect_state(tmp)
    source_hash = "3" * 64
    state.update(
        {
            "schema_version": runtime.SCHEMA_VERSION,
            "phase": "ready",
            "container_pid": 4321,
            "image": "aeon_vllm:latest",
            "image_size_bytes": 1,
            "model_dir": str(tmp / "models"),
            "model_manifest_sha256": "1" * 64,
            "model_sha256s_sha256": "2" * 64,
            "model_files": ["config.json"],
            "model_bytes": 1,
            "model_root_device": 1,
            "model_root_inode": 1,
            "model_file_stats": [["config.json", 1, 1, 0o600, 1, 1, 1]],
            "source_manifest_sha256": source_hash,
            "source_dir": str(
                Path(state["run_dir"]) / f"local-source-{source_hash}"
            ),
            "source_files": [*runtime._LEGACY_SOURCE_FILES, "SOURCE_SHA256SUMS"],
            "launch_spec_sha256": "d" * 64,
            "scratch_cleaned": False,
            "cidfile_recovery_authorized": False,
            "image_base_environment": {},
            "image_base_labels": {},
            "served_name": "Qwen3.8-27B-ARA-NVFP4-MTP",
        }
    )
    return state


def make_legacy_runtime_state(root: Path, *, executable: bool):
    """Create one exact synthetic schema-6 source closure and runtime receipt."""

    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    state = durable_runtime_state(root)
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(mode=0o700)
    source_seed = root / "legacy-source-seed"
    source_seed.mkdir(mode=0o700)
    files = []
    for relative in runtime._LEGACY_SOURCE_FILES:
        path = source_seed / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        current = path.parent
        while current != source_seed:
            current.chmod(0o700)
            current = current.parent
        payload = f"legacy source {relative}; executable={executable}\n".encode()
        path.write_bytes(payload)
        path.chmod(0o600)
        files.append((relative, hashlib.sha256(payload).hexdigest()))
    manifest = "".join(
        f"{digest}  {relative}\n" for relative, digest in files
    ).encode()
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    stage = run_dir / f"local-source-{manifest_sha256}"
    source_seed.rename(stage)
    manifest_path = stage / "SOURCE_SHA256SUMS"
    manifest_path.write_bytes(manifest)
    manifest_path.chmod(0o600)
    runtime_sha256 = dict(files)["aeon/core/qwen_runtime.py"]
    sitecustomize = stage / "aeon/scripts/vllm_uuid_sitecustomize.py"
    metadata = sitecustomize.lstat()
    state.update(
        {
            "schema_version": runtime.LEGACY_SCHEMA_VERSION,
            "source_manifest_sha256": manifest_sha256,
            "source_dir": str(stage),
            "source_files": [*runtime._LEGACY_SOURCE_FILES, "SOURCE_SHA256SUMS"],
            "container_tmpfs_options": None,
            "teardown_only": None,
        }
    )
    state["container_mounts"]["/workspace/aeon_runtime/sitecustomize.py"] = {
        "source": str(sitecustomize),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
    }
    state.pop("container_tmpfs_options")
    state.pop("teardown_only")
    return state, (manifest_sha256, runtime_sha256, executable)


class RuntimeCapabilityTests(unittest.TestCase):
    def test_registry_has_one_immutable_local_docker_capability(self):
        registry = qwen_capabilities.load_qwen_runtime_capabilities()
        capability = registry.active
        self.assertEqual(
            (
                capability.key,
                capability.host,
                capability.hostname,
                capability.runtime_adapter,
                capability.allowed_physical_gpus,
                capability.coordinator_gpu,
            ),
            (
                "qwen38-standard-177-local-docker",
                "192.168.0.177",
                "DAY2RTX6000PRO",
                "local-docker",
                (0,),
                0,
            ),
        )
        self.assertRegex(registry.manifest_sha256, r"^[0-9a-f]{64}$")
        with self.assertRaises(AttributeError):
            capability.host = "192.168.0.179"

    def test_disabled_and_unknown_targets_never_become_capacity(self):
        with self.assertRaises(qwen_capabilities.QwenCapabilityError):
            qwen_capabilities.require_enabled_qwen_target("192.168.0.179", 0)
        for host, expected_key in (
            (
                "192.168.0.178",
                qwen_capabilities.RTX5000_178_RELEASE_CAPABILITY_KEY,
            ),
            (
                "192.168.0.180",
                qwen_capabilities.RTX5000_180_RELEASE_CAPABILITY_KEY,
            ),
        ):
            for physical_gpu in (0, 1):
                capability, _manifest = (
                    qwen_capabilities.require_enabled_qwen_target(
                        host, physical_gpu
                    )
                )
                self.assertEqual(capability.key, expected_key)
        with self.assertRaises(qwen_capabilities.QwenCapabilityError):
            qwen_capabilities.require_enabled_qwen_target("192.168.0.250", 0)
        with self.assertRaises(qwen_capabilities.QwenCapabilityError):
            qwen_capabilities.require_enabled_qwen_target("192.168.0.177", 1)
        for malformed_gpu in (False, 0.0, "0", None):
            with self.subTest(physical_gpu=malformed_gpu), self.assertRaises(
                qwen_capabilities.QwenCapabilityError
            ):
                qwen_capabilities.require_enabled_qwen_target(
                    "192.168.0.177", malformed_gpu
                )

    def test_manifest_target_mutations_require_a_code_and_release_update(self):
        raw = json.loads(
            qwen_capabilities.CAPABILITY_MANIFEST_FILE.read_text(encoding="utf-8")
        )
        mutations = (
            (0, "host", "192.168.0.179"),
            (0, "runtime_adapter", "unavailable"),
            (0, "allowed_physical_gpus", [1]),
            (0, "context_tokens", 65536),
            (0, "vram_budget_gb", 41.7),
            (0, "image_id", "sha256:" + "a" * 64),
            (1, "enabled", True),
            (2, "enabled", False),
        )
        for index, field, value in mutations:
            changed_manifest = copy.deepcopy(raw)
            changed_manifest["capabilities"][index][field] = value
            with self.subTest(index=index, field=field), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "capabilities.json"
                path.write_text(json.dumps(changed_manifest), encoding="utf-8")
                with self.assertRaises(qwen_capabilities.QwenCapabilityError):
                    qwen_capabilities.load_qwen_runtime_capabilities(path)

    def test_manifest_receipt_hash_binds_exact_bytes(self):
        registry = qwen_capabilities.load_qwen_runtime_capabilities()
        raw = json.loads(
            qwen_capabilities.CAPABILITY_MANIFEST_FILE.read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "capabilities.json"
            path.write_text(json.dumps(raw, separators=(",", ":")), encoding="utf-8")
            reformatted = qwen_capabilities.load_qwen_runtime_capabilities(path)
        self.assertEqual(reformatted.capabilities, registry.capabilities)
        self.assertNotEqual(reformatted.manifest_sha256, registry.manifest_sha256)

    def test_capability_receipt_rejects_every_identity_mutation(self):
        capability, manifest_sha256 = (
            qwen_capabilities.active_qwen_runtime_capability()
        )
        receipt = {
            "key": capability.key,
            "manifest_sha256": manifest_sha256,
            "runtime_adapter": capability.runtime_adapter,
            "host": capability.host,
            "physical_gpu": capability.coordinator_gpu,
        }
        self.assertEqual(
            qwen_capabilities.validate_qwen_capability_receipt(**receipt),
            capability,
        )
        mutations = {
            "key": "qwen38-unknown",
            "manifest_sha256": "0" * 64,
            "runtime_adapter": "remote-unreleased",
            "host": "192.168.0.179",
            "physical_gpu": 1,
        }
        for field, value in mutations.items():
            with self.subTest(field=field), self.assertRaises(
                qwen_capabilities.QwenCapabilityError
            ):
                qwen_capabilities.validate_qwen_capability_receipt(
                    **{**receipt, field: value}
                )

    def test_rtx5000_release_is_enabled_and_receipt_bound(self):
        capability, manifest_sha256 = qwen_capabilities.qwen_runtime_capability(
            qwen_capabilities.RTX5000_RELEASE_CAPABILITY_KEY
        )
        self.assertTrue(capability.enabled)
        self.assertEqual(capability.host, "192.168.0.180")
        self.assertEqual(capability.context_tokens, 131072)
        self.assertEqual(capability.vram_budget_gb, 41.25)
        self.assertEqual(
            capability.release_receipt_sha256,
            "5eda720a4e168733fd0881cedb144a69f892c707439a8e64304e24ec6a04a91a",
        )
        enabled, _manifest = qwen_capabilities.enabled_qwen_runtime_capabilities()
        self.assertIn(capability, enabled)
        self.assertEqual(
            qwen_capabilities.validate_qwen_capability_receipt(
                key=capability.key,
                manifest_sha256=manifest_sha256,
                runtime_adapter=capability.runtime_adapter,
                host=capability.host,
                physical_gpu=0,
            ),
            capability,
        )
        promoted, promoted_manifest = qwen_capabilities.qwen_runtime_capability(
            qwen_capabilities.RTX5000_178_RELEASE_CAPABILITY_KEY
        )
        self.assertTrue(promoted.enabled)
        self.assertEqual(promoted.host, "192.168.0.178")
        self.assertEqual(promoted.context_tokens, 131072)
        self.assertEqual(promoted.vram_budget_gb, 41.25)
        self.assertEqual(promoted.max_num_seqs, 8)
        self.assertEqual(promoted_manifest, manifest_sha256)

    def test_compact_host_receipt_is_semantically_bound_to_its_capability(self):
        capability, _manifest = qwen_capabilities.qwen_runtime_capability(
            qwen_capabilities.RTX5000_180_RELEASE_CAPABILITY_KEY
        )
        payload = qwen_capabilities.RTX5000_180_RELEASE_RECEIPT_FILE.read_bytes()
        qwen_capabilities._validate_packaged_remote_release_receipt(
            capability, payload
        )
        receipt = json.loads(payload)
        mutations = []
        for path, value in (
            (("host",), "192.168.0.178"),
            (("status",), "provisional"),
            (("runtime", "model_manifest_sha256"), "0" * 64),
            (("gates", "long_context_exact_recall"), False),
            (("raw_reports", "long_batch_sha256"), "not-a-hash"),
        ):
            changed_receipt = copy.deepcopy(receipt)
            selected = changed_receipt
            for component in path[:-1]:
                selected = selected[component]
            selected[path[-1]] = value
            mutations.append((path, changed_receipt))
        for path, changed_receipt in mutations:
            with self.subTest(path=path), self.assertRaises(
                qwen_capabilities.QwenCapabilityError
            ):
                qwen_capabilities._validate_packaged_remote_release_receipt(
                    capability,
                    json.dumps(changed_receipt).encode("utf-8"),
                )

    def test_178_release_is_enabled_and_exact_receipt_bound(self):
        promoted_key = qwen_capabilities.RTX5000_178_RELEASE_CAPABILITY_KEY
        existing_key = qwen_capabilities.RTX5000_180_RELEASE_CAPABILITY_KEY
        self.assertEqual(promoted_key, "qwen38-compact-178-128k")
        self.assertEqual(
            qwen_capabilities.COMPACT_REMOTE_DOCKER_CAPABILITY_KEYS,
            frozenset({promoted_key, existing_key}),
        )
        self.assertNotEqual(
            qwen_capabilities._PACKAGED_REMOTE_RELEASE_RECEIPTS[promoted_key],
            qwen_capabilities._PACKAGED_REMOTE_RELEASE_RECEIPTS[existing_key],
        )
        capability, manifest = qwen_capabilities.qwen_runtime_capability(
            promoted_key
        )
        self.assertTrue(capability.enabled)
        self.assertEqual(capability.host, "192.168.0.178")
        self.assertEqual(
            capability.release_receipt_sha256,
            "fef559cd0b88506b7b0b29f12cd6c1fdee8b525fa2962358c16048529804f13d",
        )

        payload = qwen_capabilities.RTX5000_178_RELEASE_RECEIPT_FILE.read_bytes()
        self.assertEqual(
            stat.S_IMODE(
                qwen_capabilities.RTX5000_178_RELEASE_RECEIPT_FILE.stat().st_mode
            ),
            0o644,
        )
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            capability.release_receipt_sha256,
        )
        qwen_capabilities._validate_packaged_remote_release_receipt(
            capability, payload
        )
        receipt = json.loads(payload)
        self.assertEqual(receipt["status"], "passed")
        self.assertIs(receipt["gates"]["exact_teardown_and_release"], True)
        self.assertEqual(
            receipt["raw_reports"]["exact_teardown_sha256"],
            "af5d320b6db7629f4ca2b505f08ee16a6cde2e903cfb62b53ff9fa1426f53417",
        )
        for physical_gpu in (0, 1):
            self.assertEqual(
                qwen_capabilities.validate_qwen_capability_receipt(
                    key=promoted_key,
                    manifest_sha256=manifest,
                    runtime_adapter="remote-docker",
                    host="192.168.0.178",
                    physical_gpu=physical_gpu,
                ),
                capability,
            )
        with self.assertRaises(qwen_capabilities.QwenCapabilityError):
            qwen_capabilities.qwen_release_candidate_capability(
                qwen_capabilities.RTX5000_RELEASE_CANDIDATE_KEY
            )

    def test_178_release_receipt_rejects_every_evidence_mutation(self):
        capability, _manifest = qwen_capabilities.qwen_runtime_capability(
            qwen_capabilities.RTX5000_178_RELEASE_CAPABILITY_KEY
        )
        receipt = json.loads(
            qwen_capabilities.RTX5000_178_RELEASE_RECEIPT_FILE.read_text(
                encoding="utf-8"
            )
        )
        mutations = (
            (("status",), "provisional"),
            (("capability_candidate_manifest_sha256",), "0" * 64),
            (("host",), "192.168.0.180"),
            (("runtime", "largest_sampled_memory_used_mib"), 42241),
            (("gates", "semantic_mtp_requests_passed"), 14),
            (("gates", "long_context_exact_recall"), False),
            (("gates", "normal_aeon_workspace_pwd"), "/tmp"),
            (("gates", "exact_teardown_and_release"), False),
            (("raw_reports", "mtp_k3_sha256"), "0" * 64),
            (("raw_reports", "normal_aeon_sha256"), "0" * 64),
            (("raw_reports", "exact_teardown_sha256"), "0" * 64),
        )
        for path, value in mutations:
            changed_receipt = copy.deepcopy(receipt)
            selected = changed_receipt
            for component in path[:-1]:
                selected = selected[component]
            selected[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(
                qwen_capabilities.QwenCapabilityError
            ):
                qwen_capabilities._validate_packaged_remote_release_receipt(
                    capability, json.dumps(changed_receipt).encode("utf-8")
                )

    def test_178_release_receipt_contains_no_transient_runtime_identifiers(self):
        receipt = json.loads(
            qwen_capabilities.RTX5000_178_RELEASE_RECEIPT_FILE.read_text(
                encoding="utf-8"
            )
        )
        forbidden_keys = {
            "base_url",
            "claim_id",
            "container_id",
            "container_name",
            "endpoint",
            "gpu_uuid",
            "local_port",
            "physical_gpu",
            "pid",
            "remote_port",
            "run_dir",
            "source_dir",
            "ticket_id",
        }

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertNotIn(key, forbidden_keys)
                    yield from walk(item)
            elif isinstance(value, list):
                for item in value:
                    yield from walk(item)
            elif isinstance(value, str):
                yield value

        strings = tuple(walk(receipt))
        self.assertEqual(
            [value for value in strings if value.startswith("/home/")],
            ["/home/aday/NexusAgentDashboard/bc_aeon"],
        )
        for marker in ("://", "fd-", "fr-", "gc-"):
            with self.subTest(marker=marker):
                self.assertFalse(any(marker in value for value in strings))

    def test_retired_manifest_is_recovery_only_and_key_scoped(self):
        capability, current = qwen_capabilities.qwen_runtime_capability(
            qwen_capabilities.LOCAL_DOCKER_CAPABILITY_KEY
        )
        retired = (
            "52e2d54b70c14eefac3d5cae796b1f1ce40ececb95961a42d1c8ec6457254b6a"
        )
        receipt = {
            "key": capability.key,
            "manifest_sha256": retired,
            "runtime_adapter": capability.runtime_adapter,
            "host": capability.host,
            "physical_gpu": 0,
        }
        self.assertNotEqual(retired, current)
        with self.assertRaises(qwen_capabilities.QwenCapabilityError):
            qwen_capabilities.validate_qwen_capability_receipt(**receipt)
        self.assertEqual(
            qwen_capabilities.validate_qwen_capability_receipt(
                **receipt, allow_retired_manifest=True
            ),
            capability,
        )
        with self.assertRaises(qwen_capabilities.QwenCapabilityError):
            qwen_capabilities.validate_qwen_capability_receipt(
                **{**receipt, "key": qwen_capabilities.RTX5000_RELEASE_CANDIDATE_KEY},
                allow_retired_manifest=True,
            )

    def test_current_manifest_snapshot_is_prepared_only_for_future_recovery(self):
        prepared = (
            "d36efd8a0b7b6c22bc10803b11bbc48ee61e9cc4893fef04aa230cb0ce223f96"
        )
        self.assertEqual(
            qwen_capabilities._RETIRED_ENABLED_MANIFEST_KEYS[prepared],
            frozenset(
                {
                    qwen_capabilities.LOCAL_DOCKER_CAPABILITY_KEY,
                    qwen_capabilities.RTX5000_180_RELEASE_CAPABILITY_KEY,
                }
            ),
        )
        self.assertNotIn(
            qwen_capabilities.RTX5000_RELEASE_CANDIDATE_KEY,
            qwen_capabilities._RETIRED_ENABLED_MANIFEST_KEYS[prepared],
        )
        self.assertNotIn(
            qwen_capabilities.RTX5000_178_RELEASE_CAPABILITY_KEY,
            qwen_capabilities._RETIRED_ENABLED_MANIFEST_KEYS[prepared],
        )


class RetiredDirectReleaseGateTests(unittest.TestCase):
    def test_direct_start_refuses_before_any_coordinator_or_runtime_action(self):
        from aeon.scripts import gate_qwen38_rtx5000 as retired_gate

        source = Path(retired_gate.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from aeon.core.gpu_queue", source)
        self.assertNotIn("reserve_named_lease(", source)
        self.assertNotIn("start_managed_remote_runtime(", source)
        with self.assertRaisesRegex(
            runtime.QwenRuntimeError, "direct coordinator release-gate launch is retired"
        ):
            retired_gate.start()


class ReservationTests(unittest.TestCase):
    def test_qwen_reserve_is_local_gpu0_exclusive_with_exact_profile(self):
        payload = {
            **lease(),
            "owner": "owner-local-qwen",
            "claim_id": "gc-local-qwen",
            "purpose": "local qwen",
        }
        calls = []

        def coord(*args, **_kwargs):
            calls.append(args)
            if args[0] == "new-owner":
                return completed("owner-local-qwen\n")
            return completed(json.dumps(payload))

        with tempfile.TemporaryDirectory() as temp, patch.object(
            gpu_queue, "_coord", side_effect=coord
        ), patch.object(gpu_queue, "_update_compute_presence"):
            result = gpu_queue.reserve_named_lease(
                required_gb=48.7,
                purpose="local qwen",
                state_file=Path(temp) / "lease.json",
                profile=QWEN38_VLLM_PROFILE,
                host=gpu_queue.LOCAL_COORD_HOST,
                gpu_id=0,
                min_vram_gb=90,
                run_dir_root=Path(temp),
                timeout=0,
                exclusive=True,
            )
        reserve = calls[1]
        self.assertEqual(result["vram_budget_mib"], round(48.7 * 1024))
        capability, manifest_sha256 = (
            qwen_capabilities.active_qwen_runtime_capability()
        )
        self.assertEqual(result["runtime_capability_key"], capability.key)
        self.assertEqual(
            result["runtime_capability_manifest_sha256"], manifest_sha256
        )
        self.assertEqual(result["runtime_adapter"], capability.runtime_adapter)
        self.assertIn("--exclusive", reserve)
        self.assertEqual(reserve[reserve.index("--host") + 1], gpu_queue.LOCAL_COORD_HOST)
        self.assertEqual(reserve[reserve.index("--gpu") + 1], "0")
        for value in ("96", "96", "32", "16"):
            self.assertIn(value, reserve)

    def test_remote_unpinned_nonexclusive_and_gpu1_qwen_are_rejected(self):
        common = dict(
            required_gb=48.7,
            purpose="bad qwen",
            state_file=Path("/tmp/not-written"),
            profile=QWEN38_VLLM_PROFILE,
            min_vram_gb=90,
            exclusive=True,
        )
        with self.assertRaises(ValueError):
            gpu_queue.reserve_named_lease(**common, host=None)
        with self.assertRaises(ValueError):
            gpu_queue.reserve_named_lease(**common, host=gpu_queue.LOCAL_COORD_HOST, gpu_id=1)
        with self.assertRaises(ValueError):
            gpu_queue.reserve_named_lease(
                **{**common, "exclusive": False}, host=gpu_queue.LOCAL_COORD_HOST
            )

    def test_durable_wait_backoff_and_cancel_are_truthful(self):
        payload = {
            **lease(),
            "owner": "owner-qwen-wait",
            "claim_id": "gc-qwen-wait",
            "purpose": "wait qwen",
        }
        sleeps = []
        with tempfile.TemporaryDirectory() as temp, patch.object(
            gpu_queue,
            "_coord",
            side_effect=[
                completed("owner-qwen-wait\n"),
                completed(returncode=2),
                completed(returncode=2),
                completed(json.dumps(payload)),
            ],
        ), patch.object(gpu_queue, "_update_compute_presence") as presence:
            gpu_queue.reserve_named_lease(
                required_gb=48.7,
                purpose="wait qwen",
                state_file=Path(temp) / "lease.json",
                profile=QWEN38_VLLM_PROFILE,
                host=gpu_queue.LOCAL_COORD_HOST,
                gpu_id=0,
                min_vram_gb=90,
                run_dir_root=Path(temp),
                durable_wait=True,
                sleep_func=sleeps.append,
                exclusive=True,
            )
        self.assertEqual(sleeps, [15.0, 30.0])
        summaries = [call.args[2] for call in presence.call_args_list]
        self.assertTrue(
            any("Waiting for Qwen-compatible compute" in item for item in summaries)
        )

    def test_system_exit_during_wait_clears_definitive_claim_free_intent(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
            gpu_queue,
            "_coord",
            side_effect=[completed("owner-qwen-cancel\n"), completed(returncode=2)],
        ), patch.object(gpu_queue, "_update_compute_presence"):
            state_path = Path(temp) / "lease.json"
            with self.assertRaises(SystemExit):
                gpu_queue.reserve_named_lease(
                    required_gb=48.7,
                    purpose="cancel qwen",
                    state_file=state_path,
                    profile=QWEN38_VLLM_PROFILE,
                    host=gpu_queue.LOCAL_COORD_HOST,
                    gpu_id=0,
                    min_vram_gb=90,
                    run_dir_root=Path(temp),
                    durable_wait=True,
                    sleep_func=lambda _delay: (_ for _ in ()).throw(SystemExit(0)),
                    exclusive=True,
                )
            self.assertFalse(state_path.exists())

    def test_ambiguous_reserve_intent_blocks_second_reserve(self):
        calls = []

        def coord(*args, **_kwargs):
            calls.append(args[0])
            if args[0] == "new-owner":
                return completed("owner-qwen-ambiguous\n")
            if args[0] == "reserve":
                raise subprocess.TimeoutExpired("reserve", 20)
            return completed(returncode=1)

        with tempfile.TemporaryDirectory() as temp, patch.object(
            gpu_queue, "_coord", side_effect=coord
        ), patch.object(gpu_queue, "_update_compute_presence"):
            path = Path(temp) / "lease.json"
            kwargs = dict(
                required_gb=48.7,
                purpose="ambiguous",
                state_file=path,
                profile=QWEN38_VLLM_PROFILE,
                host=gpu_queue.LOCAL_COORD_HOST,
                gpu_id=0,
                min_vram_gb=90,
                run_dir_root=Path(temp),
                exclusive=True,
            )
            with self.assertRaises(gpu_queue.ReservationQuarantinedError):
                gpu_queue.reserve_named_lease(**kwargs)
            calls.clear()
            with self.assertRaises(gpu_queue.ReservationQuarantinedError):
                gpu_queue.reserve_named_lease(**kwargs)
            self.assertNotIn("reserve", calls)

    def test_reconciled_local_clear_preserves_foreign_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lease.json"
            value = {
                **lease(Path(temp)),
                "record_type": gpu_queue.LEASE_RECORD,
            }
            gpu_queue._save_state(value, path)
            with self.assertRaises(gpu_queue.ReservationQuarantinedError):
                gpu_queue.clear_reconciled_lease_state(
                    path,
                    expected_claim_id=value["claim_id"],
                    expected_owner="foreign-owner",
                    expected_run_dir=value["run_dir"],
                )
            self.assertEqual(gpu_queue.current_lease(path)["owner"], value["owner"])

    def test_known_claim_id_never_reconciles_under_foreign_global_identity(self):
        base = lease()
        intent = gpu_queue._reservation_intent(
            owner=base["owner"],
            run_dir=base["run_dir"],
            purpose=base["purpose"],
            profile=QWEN38_VLLM_PROFILE,
            required_gb=base["vram_budget_gb"],
            physical_floor_gb=90,
            exclusive=True,
            host=base["host"],
            gpu_id=0,
        )
        intent["claim_id"] = base["claim_id"]
        intent["recovered_lease"] = base
        exact_target = inventory_for(base)[0]
        cases = []
        foreign_owner = json.loads(json.dumps(exact_target))
        foreign_owner["claims"][0]["owner"] = "foreign-owner"
        cases.append(foreign_owner)
        foreign_run = json.loads(json.dumps(exact_target))
        foreign_run["claims"][0]["run_dir"] = "/foreign/run"
        cases.append(foreign_run)
        foreign_host = json.loads(json.dumps(exact_target))
        foreign_host["host"] = "192.168.0.179"
        cases.append(foreign_host)
        foreign_gpu = json.loads(json.dumps(exact_target))
        foreign_gpu["physical_gpu"] = 1
        cases.append(foreign_gpu)
        target_uuid_changed = json.loads(json.dumps(exact_target))
        target_uuid_changed["uuid"] = "GPU-bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
        cases.append(target_uuid_changed)
        claim_uuid_changed = json.loads(json.dumps(exact_target))
        claim_uuid_changed["claims"][0]["gpu_uuid"] = (
            "GPU-bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
        )
        cases.append(claim_uuid_changed)
        budget_changed = json.loads(json.dumps(exact_target))
        budget_changed["claims"][0]["vram_budget_mib"] -= 1
        cases.append(budget_changed)
        sharing_changed = json.loads(json.dumps(exact_target))
        sharing_changed["claims"][0]["exclusive"] = False
        cases.append(sharing_changed)
        duplicate = [exact_target, json.loads(json.dumps(exact_target))]
        for inventory in [*[ [item] for item in cases ], duplicate]:
            with self.subTest(inventory=inventory), self.assertRaises(
                gpu_queue.ReservationQuarantinedError
            ):
                matches = gpu_queue._reservation_matches(inventory, intent)
                if matches:
                    gpu_queue._recovered_lease(intent, *matches[0])

        absent = json.loads(json.dumps(exact_target))
        absent["claims"] = []
        self.assertEqual(gpu_queue._reservation_matches([absent], intent), [])

        owner_collision = json.loads(json.dumps(exact_target))
        owner_collision["claims"][0]["claim_id"] = "gc-foreign-collision"
        with self.assertRaises(gpu_queue.ReservationQuarantinedError):
            gpu_queue._reservation_matches([owner_collision], intent)

    def test_known_claim_mismatch_stays_durably_quarantined_without_release(self):
        base = lease()
        intent = gpu_queue._reservation_intent(
            owner=base["owner"],
            run_dir=base["run_dir"],
            purpose=base["purpose"],
            profile=QWEN38_VLLM_PROFILE,
            required_gb=base["vram_budget_gb"],
            physical_floor_gb=90,
            exclusive=True,
            host=base["host"],
            gpu_id=0,
        )
        intent["claim_id"] = base["claim_id"]
        intent["recovered_lease"] = base
        moved = inventory_for(base)[0]
        moved["claims"][0]["owner"] = "foreign-owner"
        calls = []

        def coord(*args, **_kwargs):
            calls.append(args)
            return completed(json.dumps([moved]))

        with tempfile.TemporaryDirectory() as temp, patch.object(
            gpu_queue, "_coord", side_effect=coord
        ):
            state_file = Path(temp) / "lease.json"
            gpu_queue._save_state(intent, state_file)
            with self.assertRaises(gpu_queue.ReservationQuarantinedError):
                gpu_queue.reconcile_reservation_intent(state_file)
            self.assertEqual(gpu_queue._current_record(state_file), intent)
        self.assertEqual([call[0] for call in calls], ["status"])

    def test_malformed_claim_inventory_never_proves_known_claim_absence(self):
        base = lease()
        intent = gpu_queue._reservation_intent(
            owner=base["owner"],
            run_dir=base["run_dir"],
            purpose=base["purpose"],
            profile=QWEN38_VLLM_PROFILE,
            required_gb=base["vram_budget_gb"],
            physical_floor_gb=90,
            exclusive=True,
            host=base["host"],
            gpu_id=0,
        )
        intent["claim_id"] = base["claim_id"]
        intent["recovered_lease"] = base
        exact_target = inventory_for(base)[0]
        malformed_targets = []
        missing = copy.deepcopy(exact_target)
        missing.pop("claims")
        malformed_targets.append(missing)
        for malformed_claims in (None, False, {}):
            target = copy.deepcopy(exact_target)
            target["claims"] = malformed_claims
            malformed_targets.append(target)
        only_gpu1 = copy.deepcopy(exact_target)
        only_gpu1["physical_gpu"] = 1
        only_gpu1["uuid"] = "GPU-bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
        only_gpu1["claims"] = []
        malformed_targets.append(only_gpu1)
        changed_target_uuid = copy.deepcopy(exact_target)
        changed_target_uuid["uuid"] = "GPU-bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
        changed_target_uuid["claims"] = []
        malformed_targets.append(changed_target_uuid)
        for target in malformed_targets:
            with self.subTest(claims=target.get("claims", "missing")), tempfile.TemporaryDirectory() as temp, patch.object(
                gpu_queue,
                "_coord",
                return_value=completed(json.dumps([target])),
            ) as coord:
                state_file = Path(temp) / "lease.json"
                gpu_queue._save_state(intent, state_file)
                with self.assertRaises(gpu_queue.ReservationQuarantinedError):
                    gpu_queue.reconcile_reservation_intent(state_file)
                self.assertEqual(gpu_queue._current_record(state_file), intent)
                self.assertEqual(coord.call_args.args[:2], ("status", "--json"))
        unrelated_unavailable = {
            "host": "192.168.0.179",
            "physical_gpu": None,
            "state": "UNAVAILABLE",
        }
        exact_absent = copy.deepcopy(exact_target)
        exact_absent["claims"] = []
        self.assertEqual(
            gpu_queue._reservation_matches(
                [exact_absent, unrelated_unavailable], intent
            ),
            [],
        )
        self.assertEqual(
            gpu_queue._reservation_matches(
                [exact_target, unrelated_unavailable], intent
            ),
            [(exact_target, exact_target["claims"][0])],
        )

    def test_exclusive_qwen_gpu_is_never_a_tool_candidate(self):
        inventory = [
            {"host": gpu_queue.LOCAL_COORD_HOST, "physical_gpu": 0, "acl": "OPEN", "state": "RESERVED_RUNNING", "vram_share_capacity_mib": 90000},
            {"host": gpu_queue.LOCAL_COORD_HOST, "physical_gpu": 1, "acl": "OPEN", "state": "AVAILABLE", "vram_share_capacity_mib": 40000},
        ]
        qwen = lease()
        self.assertEqual(
            gpu_queue.select_tool_gpu(inventory, 20, qwen),
            1,
        )
        self.assertIsNone(
            gpu_queue.select_tool_gpu(inventory[:1], 20, qwen)
        )
        for malformed_exclusive in (False, 1, 0, "true", None):
            with self.subTest(exclusive=malformed_exclusive):
                self.assertIsNone(
                    gpu_queue.select_tool_gpu(
                        inventory,
                        20,
                        {**qwen, "exclusive": malformed_exclusive},
                    )
                )
        for malformed_gpu in (False, 0.0, "0", None):
            with self.subTest(qwen_gpu=malformed_gpu):
                self.assertIsNone(
                    gpu_queue.select_tool_gpu(
                        inventory,
                        20,
                        {**qwen, "physical_gpu": malformed_gpu},
                    )
                )
        malformed_receipts = (
            {"host": "192.168.0.179"},
            {"runtime_capability_manifest_sha256": "0" * 64},
            {"compute_profile": "wrong-profile"},
            {"min_host_memory_gb": 1},
            {"memory_total_mib": float(qwen["memory_total_mib"])},
            {"vram_budget_mib": float(qwen["vram_budget_mib"])},
        )
        for change in malformed_receipts:
            with self.subTest(qwen_receipt=change):
                self.assertIsNone(
                    gpu_queue.select_tool_gpu(inventory, 20, {**qwen, **change})
                )

    def test_tool_selector_rejects_malformed_numeric_inventory(self):
        base = {
            "host": gpu_queue.LOCAL_COORD_HOST,
            "physical_gpu": 1,
            "acl": "OPEN",
            "state": "AVAILABLE",
            "vram_share_capacity_mib": 40000,
        }
        for malformed in (float("nan"), float("inf"), float("-inf"), True, "40000", None):
            with self.subTest(capacity=malformed):
                self.assertIsNone(
                    gpu_queue.select_tool_gpu(
                        [{**base, "vram_share_capacity_mib": malformed}], 20
                    )
                )
        for malformed in (float("nan"), float("inf"), float("-inf"), True, "20", None):
            with self.subTest(required_gb=malformed):
                self.assertIsNone(gpu_queue.select_tool_gpu([base], malformed))
        for malformed_gpu in (False, 1.0, "1", None):
            with self.subTest(inventory_gpu=malformed_gpu):
                self.assertIsNone(
                    gpu_queue.select_tool_gpu(
                        [{**base, "physical_gpu": malformed_gpu}], 20
                    )
                )


class AdmissionIdentityTests(unittest.TestCase):
    def test_stricter_remote_resource_floors_are_accepted_but_weaker_are_not(self):
        stricter = runtime._validate_lease(
            lease(min_disk_free_gb=80.0, min_shm_free_gb=24.0)
        )
        self.assertEqual(stricter["min_disk_free_gb"], 80.0)
        self.assertEqual(stricter["min_shm_free_gb"], 24.0)

        for field in (
            "min_host_memory_gb",
            "min_host_commit_gb",
            "min_disk_free_gb",
            "min_shm_free_gb",
        ):
            weakened = getattr(QWEN38_VLLM_PROFILE, field) - 0.5
            with self.subTest(field=field), self.assertRaisesRegex(
                runtime.QwenRuntimeError, "lease profile changed"
            ):
                runtime._validate_lease(lease(**{field: weakened}))

    def test_lease_receipt_uses_round_exclusive_local_gpu0_and_floors(self):
        value = lease()
        with patch.object(runtime, "_coord", return_value=completed(json.dumps(inventory_for(value)))):
            self.assertEqual(runtime.verify_coordinator_lease(value)["physical_gpu"], 0)
        for change in (
            {"physical_gpu": 1},
            {"host": "192.168.0.179"},
            {"runtime_capability_key": "qwen38-unknown"},
            {"runtime_capability_manifest_sha256": "0" * 64},
            {"runtime_adapter": "remote-unreleased"},
            {"exclusive": False},
            {"vram_budget_mib": round(48.7 * 1024) - 1},
            {"vram_budget_gb": 49.0, "vram_budget_mib": round(49 * 1024)},
        ):
            with self.subTest(change=change), self.assertRaises(runtime.QwenRuntimeError):
                runtime._validate_lease({**value, **change})
        with patch.object(
            runtime, "_coord", return_value=completed(json.dumps(inventory_for(value, host_memory_available_mib=95 * 1024)))
        ), self.assertRaises(runtime.QwenLeaseLostError):
            runtime.verify_coordinator_lease(value)

    def test_nonfinite_and_boolean_coordinator_evidence_fails_closed(self):
        value = lease()
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(receipt_total=nonfinite), self.assertRaises(
                runtime.QwenRuntimeError
            ):
                runtime._validate_lease({**value, "memory_total_mib": nonfinite})
            with self.subTest(receipt_budget=nonfinite), self.assertRaises(
                runtime.QwenRuntimeError
            ):
                runtime._validate_lease(
                    {
                        **value,
                        "vram_budget_gb": nonfinite,
                        "vram_budget_mib": value["vram_budget_mib"],
                    }
                )
            for floor in (
                "host_memory_available_mib",
                "host_commit_headroom_mib",
                "host_disk_available_mib",
                "host_shm_available_mib",
            ):
                with self.subTest(floor=floor, value=nonfinite), patch.object(
                    runtime,
                    "_coord",
                    return_value=completed(
                        json.dumps(inventory_for(value, **{floor: nonfinite}))
                    ),
                ), self.assertRaises(runtime.QwenLeaseLostError):
                    runtime.verify_coordinator_lease(value)

        with self.assertRaises(runtime.QwenRuntimeError):
            runtime._validate_lease(
                {**value, "memory_total_mib": float(value["memory_total_mib"])}
            )

        with patch.object(
            runtime,
            "_coord",
            return_value=completed(
                json.dumps(inventory_for(value, physical_gpu=False))
            ),
        ), self.assertRaises(runtime.QwenLeaseLostError):
            runtime.verify_coordinator_lease(value)

        for malformed_total in (
            False,
            float(value["memory_total_mib"]),
            str(value["memory_total_mib"]),
            None,
            float("nan"),
            float("inf"),
            float("-inf"),
            1,
            value["memory_total_mib"] + 1,
        ):
            with self.subTest(
                live_total=malformed_total
            ), patch.object(
                runtime,
                "_coord",
                return_value=completed(
                    json.dumps(
                        inventory_for(value, memory_total_mib=malformed_total)
                    )
                ),
            ), self.assertRaises(runtime.QwenLeaseLostError):
                runtime.verify_coordinator_lease(value)

        float_budget_inventory = inventory_for(value)
        float_budget_inventory[0]["claims"][0]["vram_budget_mib"] = float(
            value["vram_budget_mib"]
        )
        with patch.object(
            runtime,
            "_coord",
            return_value=completed(json.dumps(float_budget_inventory)),
        ), self.assertRaises(runtime.QwenLeaseLostError):
            runtime.verify_coordinator_lease(value)

        for malformed_exclusive in (True, False, 0, 1.0, "1", None):
            status_inventory = inventory_for(value)
            status_inventory[0]["claims"][0]["exclusive"] = malformed_exclusive
            with self.subTest(
                live_claim_exclusive=malformed_exclusive
            ), patch.object(
                runtime,
                "_coord",
                return_value=completed(json.dumps(status_inventory)),
            ), self.assertRaises(runtime.QwenLeaseLostError):
                runtime.verify_coordinator_lease(value)

    def test_nonfinite_profiles_planner_and_runtime_state_fail_closed(self):
        profile_fields = (
            "min_host_memory_gb",
            "min_host_commit_gb",
            "min_disk_free_gb",
            "min_shm_free_gb",
        )
        for field in profile_fields:
            for nonfinite in (
                float("nan"),
                float("inf"),
                float("-inf"),
                True,
                "1",
            ):
                values = {
                    "min_host_memory_gb": 1.0,
                    "min_host_commit_gb": 1.0,
                    "min_disk_free_gb": 1.0,
                    "min_shm_free_gb": 1.0,
                    field: nonfinite,
                }
                with self.subTest(field=field, value=nonfinite), self.assertRaises(
                    ValueError
                ):
                    ComputeProfile(key="malformed", **values)

        environment = deploy_environment()
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            changed_environment = {
                **environment,
                "AEON_GPU_MEM_UTIL": str(nonfinite),
            }
            with self.subTest(utility=nonfinite), self.assertRaises(
                runtime.QwenRuntimeError
            ):
                runtime._planner_contract(
                    changed_environment,
                    lease(),
                    Mock(),
                    qwen_capabilities.STANDARD_IMAGE_ID,
                    Path("/tmp/hermetic-package"),
                    container_name="aeon_qwen_test",
                    port=8033,
                )

        value = lease()
        state = {
            **value,
            "schema_version": runtime.SCHEMA_VERSION,
            "phase": "preparing",
            "expected_hostname": runtime.LOCAL_COORD_HOSTNAME,
            "container_name": "aeon_qwen_test",
            "container_id": None,
            "image": "aeon_vllm:latest",
            "image_id": qwen_capabilities.STANDARD_IMAGE_ID,
            "model_dir": "/tmp/hermetic-model",
            "model_manifest_sha256": "1" * 64,
            "model_sha256s_sha256": "2" * 64,
            "model_files": ["config.json"],
            "model_bytes": 1,
            "model_file_stats": [["config.json", 1, 1, 0o600, 1, 1, 1]],
            "source_manifest_sha256": "3" * 64,
            "source_dir": f'{value["run_dir"]}/local-source-{"3" * 64}',
            "source_files": ["aeon/main.py"],
            "launch_nonce": "4" * 64,
            "launch_spec_sha256": "5" * 64,
            "wrapper_sha256": "6" * 64,
            "docker_sha256": "7" * 64,
            "image_size_bytes": 1,
            "local_port": 8033,
            "remote_port": 8033,
            "scratch_cleaned": False,
            "container_command": [],
            "container_environment": {},
            "container_labels": {},
            "container_mounts": {},
            "image_base_environment": {},
            "image_base_labels": {},
            "image_base_exposed_ports": {},
            "container_tmpfs_options": runtime._container_tmpfs_options(
                executable=True
            ),
            "teardown_only": False,
            "served_name": "Qwen3.8-27B-ARA-NVFP4-MTP",
        }
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(runtime_budget=nonfinite), patch.object(
                runtime,
                "_private_json_read",
                return_value={**state, "vram_budget_gb": nonfinite},
            ), self.assertRaises(runtime.QwenRuntimeError):
                runtime.current_runtime_state(Path("/not/read"))

        self.assertTrue(runtime.runtime_state_matches_lease(state, value))
        with patch.object(runtime, "_private_json_read", return_value=state):
            self.assertEqual(runtime.current_runtime_state(Path("/not/read")), state)
        for malformed_recovery in (0, 1, 0.0, "true", None, [], {}):
            changed_state = {
                **state,
                "cidfile_recovery_authorized": malformed_recovery,
            }
            with self.subTest(
                cidfile_recovery_authorized=malformed_recovery
            ), patch.object(
                runtime, "_private_json_read", return_value=changed_state
            ), self.assertRaises(runtime.QwenRuntimeError):
                runtime.current_runtime_state(Path("/not/read"))
        recovered_state = {**state, "cidfile_recovery_authorized": True}
        with patch.object(runtime, "_private_json_read", return_value=recovered_state):
            self.assertEqual(
                runtime.current_runtime_state(Path("/not/read")), recovered_state
            )
        old_schema = {**state, "schema_version": runtime.SCHEMA_VERSION - 1}
        with patch.object(
            runtime, "_private_json_read", return_value=old_schema
        ), self.assertRaises(runtime.QwenRuntimeError):
            runtime.current_runtime_state(Path("/not/read"))
        self.assertFalse(runtime.runtime_state_matches_lease(old_schema, value))

        numeric_fields = (
            "memory_total_mib",
            "vram_budget_gb",
            "min_host_memory_gb",
            "min_host_commit_gb",
            "min_disk_free_gb",
            "min_shm_free_gb",
        )
        malformed_numbers = (
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            "96",
            None,
        )
        for field in numeric_fields:
            for malformed in malformed_numbers:
                changed_state = {**state, field: malformed}
                with self.subTest(
                    runtime_field=field, value=malformed
                ), patch.object(
                    runtime, "_private_json_read", return_value=changed_state
                ), self.assertRaises(runtime.QwenRuntimeError):
                    runtime.current_runtime_state(Path("/not/read"))
                self.assertFalse(
                    runtime.runtime_state_matches_lease(changed_state, value)
                )

        for malformed in (
            float(value["vram_budget_mib"]),
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            str(value["vram_budget_mib"]),
            None,
        ):
            changed_state = {**state, "vram_budget_mib": malformed}
            with self.subTest(
                runtime_field="vram_budget_mib", value=malformed
            ), patch.object(
                runtime, "_private_json_read", return_value=changed_state
            ), self.assertRaises(runtime.QwenRuntimeError):
                runtime.current_runtime_state(Path("/not/read"))
            self.assertFalse(runtime.runtime_state_matches_lease(changed_state, value))

        for field, malformed_values in (
            ("physical_gpu", (False, 0.0, "0", None)),
            ("exclusive", (False, 1, 0, "true", None)),
            ("compute_profile", ("wrong-profile", None)),
        ):
            for malformed in malformed_values:
                changed_state = {**state, field: malformed}
                with self.subTest(
                    runtime_field=field, value=malformed
                ), patch.object(
                    runtime, "_private_json_read", return_value=changed_state
                ), self.assertRaises(runtime.QwenRuntimeError):
                    runtime.current_runtime_state(Path("/not/read"))
                self.assertFalse(
                    runtime.runtime_state_matches_lease(changed_state, value)
                )

        float_total = {
            **state,
            "memory_total_mib": float(state["memory_total_mib"]),
        }
        with patch.object(
            runtime, "_private_json_read", return_value=float_total
        ), self.assertRaises(runtime.QwenRuntimeError):
            runtime.current_runtime_state(Path("/not/read"))
        self.assertFalse(runtime.runtime_state_matches_lease(float_total, value))

        changed_total = {**state, "memory_total_mib": state["memory_total_mib"] + 1}
        with patch.object(runtime, "_private_json_read", return_value=changed_total):
            self.assertEqual(
                runtime.current_runtime_state(Path("/not/read")), changed_total
            )
        self.assertFalse(runtime.runtime_state_matches_lease(changed_total, value))

    def test_reservation_receipts_require_finite_totals_and_literal_boolean(self):
        base = lease()
        intent = gpu_queue._reservation_intent(
            owner=base["owner"],
            run_dir=base["run_dir"],
            purpose=base["purpose"],
            profile=QWEN38_VLLM_PROFILE,
            required_gb=base["vram_budget_gb"],
            physical_floor_gb=90,
            exclusive=True,
            host=base["host"],
            gpu_id=0,
        )
        target = inventory_for(base)[0]
        claim = target["claims"][0]
        for malformed in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(total=malformed), self.assertRaises(
                gpu_queue.ReservationQuarantinedError
            ):
                gpu_queue._validate_reservation_receipt(
                    {**base, "memory_total_mib": malformed}, intent=intent
                )
        with self.assertRaises(gpu_queue.ReservationQuarantinedError):
            gpu_queue._validate_reservation_receipt(
                {**base, "memory_total_mib": float(base["memory_total_mib"])},
                intent=intent,
            )
        with self.assertRaises(gpu_queue.ReservationQuarantinedError):
            gpu_queue._recovered_lease(
                intent,
                {**target, "memory_total_mib": float(target["memory_total_mib"])},
                claim,
            )
        for malformed in (1, "true"):
            with self.subTest(receipt_exclusive=malformed), self.assertRaises(
                gpu_queue.ReservationQuarantinedError
            ):
                gpu_queue._validate_reservation_receipt(
                    {**base, "exclusive": malformed}, intent=intent
                )

        recovered = gpu_queue._recovered_lease(intent, target, claim)
        self.assertIs(recovered["exclusive"], True)
        for malformed in (True, False, 0, 1.0, "1", None):
            with self.subTest(status_exclusive=malformed), self.assertRaises(
                gpu_queue.ReservationQuarantinedError
            ):
                gpu_queue._recovered_lease(
                    intent,
                    target,
                    {**claim, "exclusive": malformed},
                )

        release_state = inspect_state(Path(tempfile.mkdtemp()))
        false_gpu_inventory = inventory_for(release_state, physical_gpu=False)
        with patch.object(
            runtime,
            "_coord",
            return_value=completed(json.dumps(false_gpu_inventory)),
        ), self.assertRaises(runtime.QwenRuntimeError):
            runtime._coordinator_claim_matches(release_state)

    def test_sitecustomize_rejects_nonfinite_lease_environment(self):
        script = Path(runtime.__file__).resolve().parents[1] / "scripts" / (
            "vllm_uuid_sitecustomize.py"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            interface = root / "vllm" / "platforms" / "interface.py"
            interface.parent.mkdir(parents=True)
            (root / "vllm" / "__init__.py").write_text("", encoding="utf-8")
            (root / "vllm" / "platforms" / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            interface.write_text(
                "class Platform:\n"
                "    device_control_env_var = 'CUDA_VISIBLE_DEVICES'\n"
                "    @classmethod\n"
                "    def device_id_to_physical_device_id(cls, value):\n"
                "        return value\n",
                encoding="utf-8",
            )
            for key in ("GPU_PLANNED_VRAM_GB", "GPU_RESERVE_GB"):
                for malformed in ("nan", "inf", "-inf"):
                    environment = {
                        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                        "PYTHONPATH": str(root),
                        "GPU_AGENT_CLAIM_ID": "gc-hermetic",
                        "GPU_PLANNED_VRAM_GB": "48.7",
                        "GPU_RESERVE_GB": "6",
                        "GPU_LEASE_EXCLUSIVE": "1",
                        key: malformed,
                    }
                    result = subprocess.run(
                        [sys.executable, str(script)],
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    with self.subTest(key=key, value=malformed):
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("Invalid exclusive vLLM lease plan", result.stderr)

    def test_model_full_hash_exact_set_and_immutable_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            identity = make_model(Path(temp))
            runtime.revalidate_artifact_identity(identity)
            extra = identity.model_dir / "generation_config.json"
            extra.write_text("{}", encoding="utf-8")
            extra.chmod(0o600)
            with self.assertRaises(runtime.QwenRuntimeError):
                runtime.load_artifact_identity(identity.model_dir, verify_payload=False)
            extra.unlink()
            target = identity.model_dir / "config.json"
            target.chmod(0o620)
            with self.assertRaises(runtime.QwenRuntimeError):
                runtime.load_artifact_identity(identity.model_dir, verify_payload=False)

    def test_model_allows_exact_private_sha256_verification_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = make_model(Path(temp))
            marker = add_model_verification_sidecar(baseline.model_dir)
            identity = runtime.load_artifact_identity(
                baseline.model_dir, verify_payload=False
            )

            self.assertEqual(identity.manifest_sha256, baseline.manifest_sha256)
            self.assertEqual(identity.sha256s_sha256, baseline.sha256s_sha256)
            self.assertEqual(identity.files, baseline.files)
            self.assertEqual(identity.total_bytes, baseline.total_bytes + 65)
            self.assertEqual(marker.stat().st_size, 65)
            self.assertIn(marker.name, {item[0] for item in identity.file_stats})
            runtime.revalidate_artifact_identity(identity)

    def test_model_rejects_malformed_sha256_verification_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            identity = make_model(Path(temp))
            marker = add_model_verification_sidecar(identity.model_dir)
            exact = marker.read_bytes()
            malformed_payloads = (
                b"0" * 64 + b"\n",
                exact[:-1],
                exact + b"\n",
                exact.upper(),
            )
            for payload in malformed_payloads:
                with self.subTest(payload=payload):
                    marker.write_bytes(payload)
                    marker.chmod(0o600)
                    with self.assertRaises(runtime.QwenRuntimeError):
                        runtime.load_artifact_identity(
                            identity.model_dir, verify_payload=False
                        )

    def test_model_rejects_unsafe_or_nonexclusive_verification_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            identity = make_model(Path(temp))
            marker = add_model_verification_sidecar(identity.model_dir)
            for mode in (0o400, 0o640, 0o700):
                with self.subTest(mode=oct(mode)):
                    marker.chmod(mode)
                    with self.assertRaises(runtime.QwenRuntimeError):
                        runtime.load_artifact_identity(
                            identity.model_dir, verify_payload=False
                        )
            marker.chmod(0o600)
            link = marker.with_name("marker-hardlink")
            os.link(marker, link)
            with self.assertRaises(runtime.QwenRuntimeError):
                runtime.load_artifact_identity(identity.model_dir, verify_payload=False)

    def test_model_sidecar_does_not_allow_any_other_extra_file(self):
        with tempfile.TemporaryDirectory() as temp:
            identity = make_model(Path(temp))
            add_model_verification_sidecar(identity.model_dir)
            extra = identity.model_dir / "generation_config.json"
            extra.write_text("{}", encoding="utf-8")
            extra.chmod(0o600)
            with self.assertRaises(runtime.QwenRuntimeError):
                runtime.load_artifact_identity(identity.model_dir, verify_payload=False)

    def test_model_sidecar_is_bound_into_post_reserve_stat_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = make_model(Path(temp))
            marker = add_model_verification_sidecar(baseline.model_dir)
            identity = runtime.load_artifact_identity(
                baseline.model_dir, verify_payload=False
            )
            marker.write_bytes(b"0" * 64 + b"\n")
            marker.chmod(0o600)
            with self.assertRaises(runtime.QwenRuntimeError):
                runtime.revalidate_artifact_identity(identity)

    def test_model_hash_launcher_bypasses_env_shebang_and_uses_fixed_path(self):
        with tempfile.TemporaryDirectory() as temp:
            identity = make_model(Path(temp))
            calls = []

            def runner(argv, **kwargs):
                calls.append((list(argv), kwargs))
                return completed()

            runtime.load_artifact_identity(
                identity.model_dir, verify_payload=True, command_runner=runner
            )
        self.assertEqual(len(calls), 1)
        argv, kwargs = calls[0]
        self.assertEqual(
            argv[:3],
            [
                str(runtime.HOST_BASH),
                str(runtime.FLEET_LOW_PRIORITY),
                str(runtime.HOST_SHA256SUM),
            ],
        )
        self.assertEqual(kwargs["env"], runtime.HOST_LAUNCH_ENV)

    def test_post_reserve_stat_receipt_detects_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            identity = make_model(Path(temp))
            target = identity.model_dir / "config.json"
            old = target.with_suffix(".old")
            target.rename(old)
            target.write_text("{}", encoding="utf-8")
            target.chmod(0o600)
            with self.assertRaises(runtime.QwenRuntimeError):
                runtime.revalidate_artifact_identity(identity)

    def test_source_receipt_rejects_mutable_file_or_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in runtime.SOURCE_FILES:
                path = root / relative
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                current = path.parent
                while current != root:
                    current.chmod(0o700)
                    current = current.parent
                path.write_text(relative, encoding="utf-8")
                path.chmod(0o600)
            write_source_manifest(root)
            runtime._source_identity(root, root / "run")
            target = root / runtime.SOURCE_FILES[0]
            target.chmod(0o620)
            with self.assertRaises(runtime.QwenRuntimeError):
                runtime._source_identity(root, root / "run")
            target.chmod(0o600)
            (root / "aeon/core").chmod(0o720)
            with self.assertRaises(runtime.QwenRuntimeError):
                runtime._source_identity(root, root / "run")

    def test_source_receipt_covers_only_qwen_serving_dependencies(self):
        required = {
            "aeon/__init__.py",
            "aeon/core/__init__.py",
            "aeon/core/action_schema.py",
            "aeon/core/compute_profile.py",
            "aeon/core/deploy_planner.py",
            "aeon/core/fleet_hosts.py",
            "aeon/core/gpu.py",
            "aeon/core/gpu_queue.py",
            "aeon/core/model_catalog.py",
            "aeon/core/mtp_tuning.py",
            "aeon/core/qwen_artifact_cache.py",
            "aeon/core/qwen_capabilities.py",
            "aeon/core/qwen_fleet_runtime.py",
            "aeon/core/qwen_runtime.py",
            "aeon/core/sampling.py",
            "aeon/core/utils/io.py",
            "aeon/core/data/qwen38_mtp_selection.json",
            "aeon/core/data/qwen38_rtx5000_178_128k_release_receipt.json",
            "aeon/core/data/qwen38_rtx5000_128k_release_receipt.json",
            "aeon/core/data/qwen_runtime_capabilities.json",
            "aeon/scripts/vllm_uuid_sitecustomize.py",
            "aeon/scripts/warmup_qwen38_vllm.py",
            "aeon/scripts/qwen_remote_worker.py",
        }
        self.assertTrue(required.issubset(set(runtime.SOURCE_FILES)))
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = make_source_tree(base)
            before = runtime._source_identity(root, base / "unused-run")
            for relative in (
                "aeon/core/action_schema.py",
                "aeon/core/qwen_capabilities.py",
                "aeon/core/sampling.py",
                "aeon/core/data/qwen38_rtx5000_178_128k_release_receipt.json",
                "aeon/core/data/qwen38_rtx5000_128k_release_receipt.json",
                "aeon/core/data/qwen_runtime_capabilities.json",
            ):
                target = root / relative
                original = target.read_bytes()
                target.write_bytes(original + b"# changed\n")
                target.chmod(0o600)
                write_source_manifest(root)
                after = runtime._source_identity(root, base / "unused-run")
                self.assertNotEqual(after.manifest_sha256, before.manifest_sha256)
                target.write_bytes(original)
                target.chmod(0o600)
                write_source_manifest(root)

            main = root / "aeon/main.py"
            main.write_bytes(b"# interactive harness change\n")
            main.chmod(0o600)
            self.assertEqual(
                runtime._source_identity(root, base / "unused-run").manifest_sha256,
                before.manifest_sha256,
            )

    def test_packaged_source_manifest_exactly_matches_current_closure(self):
        package_root = Path(runtime.__file__).resolve().parents[2]
        generated = b"".join(
            hashlib.sha256((package_root / relative).read_bytes())
            .hexdigest()
            .encode("ascii")
            + b"  "
            + relative.encode("utf-8")
            + b"\n"
            for relative in runtime.SOURCE_FILES
        )
        self.assertEqual(
            (package_root / runtime.SOURCE_MANIFEST_FILE).read_bytes(), generated
        )

    def test_staged_warmup_dependency_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source_root = make_source_tree(base)
            runtime_root = base / "runtime"
            run_dir = runtime_root / "aeon-qwen38-vllm-bc-aeon-hermetic-owner"
            with patch.object(runtime, "RUNTIME_ROOT", runtime_root):
                expected = runtime._source_identity(source_root, run_dir)
                staged = runtime._prepare_source_stage(
                    source_root, run_dir, expected_identity=expected
                )
                runtime._validate_source_stage(staged)
                dependency = staged.stage_dir / "aeon/core/sampling.py"
                dependency.write_bytes(dependency.read_bytes() + b"# changed\n")
                dependency.chmod(0o600)
                with self.assertRaises(runtime.QwenRuntimeError):
                    runtime._validate_source_stage(staged)

    def test_warmup_runs_only_from_explicit_staged_pythonpath(self):
        source = Path(runtime.__file__).read_text(encoding="utf-8")
        warmup = Path(__file__).resolve().parents[1].joinpath(
            "scripts/warmup_qwen38_vllm.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"PYTHONPATH": str(source.stage_dir)', source)
        self.assertIn("cwd=str(source.stage_dir)", source)
        self.assertIn("AEON_STAGED_SOURCE_ROOT", source)
        self.assertIn("str(HOST_BASH),\n                str(FLEET_LOW_PRIORITY),\n                str(HOST_PYTHON)", source)
        self.assertIn("_assert_staged_imports", warmup)
        self.assertIn("action_schema", warmup)
        self.assertIn("sampling", warmup)

        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp) / "stage"
            package_root = Path(__file__).resolve().parents[2]
            for relative in (
                "aeon/__init__.py",
                "aeon/core/__init__.py",
                "aeon/core/action_schema.py",
                "aeon/core/sampling.py",
                "aeon/scripts/warmup_qwen38_vllm.py",
            ):
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((package_root / relative).read_bytes())
            probe = """
import json
from pathlib import Path
from aeon.scripts import warmup_qwen38_vllm as warmup
warmup._assert_staged_imports()
print(json.dumps([
    str(Path(warmup._action_schema.__file__).resolve()),
    str(Path(warmup._sampling.__file__).resolve()),
]))
"""
            result = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=stage,
                env={
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "PYTHONPATH": str(stage),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "AEON_STAGED_SOURCE_ROOT": str(stage),
                },
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for imported in json.loads(result.stdout):
                Path(imported).relative_to(stage)

    def test_exact_release_tuple_and_every_resource_knob_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = make_model(Path(temp))
            value = lease(Path(temp))
            with patch("aeon.core.mtp_tuning.load_selection", return_value=(3, {})):
                command, served = runtime._planner_contract(
                    deploy_environment(),
                    value,
                    artifact,
                    "sha256:" + "a" * 64,
                    Path(temp),
                    container_name="aeon_qwen_test",
                    port=8033,
                )
            self.assertEqual(served, "Qwen3.8-27B-ARA-NVFP4-MTP")
            self.assertIn("114688", command)
            self.assertIn("0.415", command)
            self.assertIn("32768", command)
            self.assertEqual(command.count("--no-enable-log-requests"), 1)
            self.assertNotIn("--disable-log-requests", command)
            self.assertEqual(command.count("--enable-auto-tool-choice"), 1)
            self.assertEqual(command.count("--tool-call-parser"), 1)
            tool_parser_index = command.index("--tool-call-parser")
            self.assertEqual(command[tool_parser_index + 1], "qwen3_coder")
            mutations = {
                "AEON_GPU_MEM_UTIL": "0.9",
                "AEON_MAX_NUM_SEQS": "2",
                "AEON_MAX_NUM_BATCHED": "65536",
                "AEON_LLM_VRAM_BUDGET_GB": "49",
                "AEON_MTP_METHOD": "eagle",
            }
            for key, changed in mutations.items():
                environment = {**deploy_environment(), key: changed}
                with self.subTest(key=key), patch(
                    "aeon.core.mtp_tuning.load_selection", return_value=(3, {})
                ), self.assertRaises(runtime.QwenRuntimeError):
                    runtime._planner_contract(
                        environment, value, artifact, "sha256:" + "a" * 64,
                        Path(temp), container_name="aeon_qwen_test", port=8033,
                    )
            environment = deploy_environment()
            plan = json.loads(environment["AEON_DEPLOY_PLAN"])
            plan["nodes"][0]["ctx"] = 262144
            environment["AEON_DEPLOY_PLAN"] = json.dumps(plan)
            with patch("aeon.core.mtp_tuning.load_selection", return_value=(3, {})), self.assertRaises(runtime.QwenRuntimeError):
                runtime._planner_contract(
                    environment, value, artifact, "sha256:" + "a" * 64,
                    Path(temp), container_name="aeon_qwen_test", port=8033,
                )

    def test_flashinfer_attention_backend_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = make_model(Path(temp))
            value = lease(Path(temp))
            environment = deploy_environment()
            environment["AEON_VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
            with patch(
                "aeon.core.mtp_tuning.load_selection", return_value=(3, {})
            ), self.assertRaises(runtime.QwenRuntimeError):
                runtime._planner_contract(
                    environment, value, artifact, "sha256:" + "a" * 64,
                    Path(temp), container_name="aeon_qwen_test", port=8033,
                )

    def test_image_port_receipt_volumes_and_environment_are_exact(self):
        expected_ports = {"8000/tcp": {}}
        self.assertEqual(
            runtime._normalise_image_config(
                {"Env": [], "Volumes": None, "ExposedPorts": expected_ports}
            ),
            ({}, {}, expected_ports),
        )
        with self.assertRaises(runtime.QwenRuntimeError):
            runtime._normalise_image_config(
                {
                    "Env": [],
                    "Volumes": {"/data": {}},
                    "ExposedPorts": expected_ports,
                }
            )
        malformed_ports = (
            None,
            {},
            {"9000/tcp": {}},
            {"8000/tcp": None},
            {"8000/tcp": {}, "9000/tcp": {}},
        )
        for exposed_ports in malformed_ports:
            with self.subTest(exposed_ports=exposed_ports), self.assertRaises(
                runtime.QwenRuntimeError
            ):
                runtime._normalise_image_config(
                    {"Env": [], "Volumes": None, "ExposedPorts": exposed_ports}
                )
        with self.assertRaises(runtime.QwenRuntimeError):
            runtime._normalise_image_config(
                {
                    "Env": ["A=1", "A=2"],
                    "Volumes": None,
                    "ExposedPorts": expected_ports,
                }
            )


class LegacyRuntimeMigrationTests(HermeticDockerRootMixin, unittest.TestCase):
    def test_only_two_reviewed_schema6_predecessors_are_registered(self):
        self.assertEqual(
            runtime._LEGACY_TMPFS_PREDECESSORS,
            (
                (
                    "f5e7a0722dceeb4c45558ad1cf5390278db4324a7d36b003077551dc7fe6c67a",
                    "b319ebcd59aebd8bc74fd5a82e9d2d7b2575ab85ec3430e408167c3b0a9b4857",
                    True,
                ),
                (
                    "cac5152b23a87e9a406e3b12f60aa8d304e545d23d5fd9e0ff02468c8c8288e6",
                    "283692460d93de68ba933ad82cd1d214265a06435a936f83c1ce328ddf454786",
                    False,
                ),
            ),
        )

    def test_exact_schema6_closures_migrate_to_bound_teardown_only_policy(self):
        for executable in (True, False):
            with self.subTest(executable=executable), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with patch.object(runtime, "RUNTIME_ROOT", root):
                    legacy, predecessor = make_legacy_runtime_state(
                        root, executable=executable
                    )
                    with patch.object(
                        runtime, "_LEGACY_TMPFS_PREDECESSORS", (predecessor,)
                    ), patch.object(runtime, "_private_json_read", return_value=legacy):
                        migrated = runtime.current_runtime_state(root / "runtime.json")
                    self.assertEqual(migrated["schema_version"], runtime.SCHEMA_VERSION)
                    self.assertIs(migrated["teardown_only"], True)
                    self.assertEqual(
                        migrated["migrated_from_schema"],
                        runtime.LEGACY_SCHEMA_VERSION,
                    )
                    self.assertEqual(
                        migrated["legacy_qwen_runtime_sha256"], predecessor[1]
                    )
                    self.assertEqual(
                        migrated["container_tmpfs_options"],
                        runtime._container_tmpfs_options(executable=executable),
                    )
                    item = inspect_payload(migrated)
                    item["HostConfig"]["Tmpfs"]["/workspace/cache"] = migrated[
                        "container_tmpfs_options"
                    ]
                    with patch.object(
                        runtime, "_LEGACY_TMPFS_PREDECESSORS", (predecessor,)
                    ), patch.object(
                        runtime, "_mounts_match_live_pid", return_value=True
                    ):
                        self.assertEqual(
                            runtime._inspect_identity(item, migrated),
                            ("active", 4321),
                        )

    def test_schema6_migration_rejects_unmapped_mutated_or_prefilled_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(runtime, "RUNTIME_ROOT", root):
                legacy, predecessor = make_legacy_runtime_state(root, executable=True)
                with patch.object(runtime, "_private_json_read", return_value=legacy), self.assertRaises(
                    runtime.QwenRuntimeError
                ):
                    runtime.current_runtime_state(root / "runtime.json")

                prefilled = {**legacy, "teardown_only": True}
                with patch.object(
                    runtime, "_LEGACY_TMPFS_PREDECESSORS", (predecessor,)
                ), patch.object(
                    runtime, "_private_json_read", return_value=prefilled
                ), self.assertRaises(runtime.QwenRuntimeError):
                    runtime.current_runtime_state(root / "runtime.json")

                staged_runtime = (
                    Path(legacy["source_dir"]) / "aeon/core/qwen_runtime.py"
                )
                staged_runtime.write_bytes(staged_runtime.read_bytes() + b"changed\n")
                staged_runtime.chmod(0o600)
                with patch.object(
                    runtime, "_LEGACY_TMPFS_PREDECESSORS", (predecessor,)
                ), patch.object(
                    runtime, "_private_json_read", return_value=legacy
                ), self.assertRaises(runtime.QwenRuntimeError):
                    runtime.current_runtime_state(root / "runtime.json")

    def test_schema7_policy_and_migration_markers_are_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(runtime, "RUNTIME_ROOT", root):
                current = durable_runtime_state(root)
                malformed = (
                    {**current, "container_tmpfs_options": runtime._container_tmpfs_options(executable=False)},
                    {**current, "container_tmpfs_options": None},
                    {**current, "teardown_only": 1},
                    {**current, "migrated_from_schema": runtime.LEGACY_SCHEMA_VERSION},
                    {**current, "schema_version": float(runtime.SCHEMA_VERSION)},
                    {**current, "schema_version": True},
                )
                for state in malformed:
                    with self.subTest(state=state), patch.object(
                        runtime, "_private_json_read", return_value=state
                    ), self.assertRaises(runtime.QwenRuntimeError):
                        runtime.current_runtime_state(root / "runtime.json")

                for malformed_schema in (
                    float(runtime.LEGACY_SCHEMA_VERSION),
                    False,
                    str(runtime.LEGACY_SCHEMA_VERSION),
                ):
                    with self.subTest(legacy_schema=malformed_schema), self.assertRaises(
                        runtime.QwenRuntimeError
                    ):
                        runtime._migrate_legacy_runtime_state(
                            {"schema_version": malformed_schema}
                        )

                second = root / "second"
                with patch.object(runtime, "RUNTIME_ROOT", second):
                    legacy, predecessor = make_legacy_runtime_state(
                        second, executable=True
                    )
                with patch.object(runtime, "RUNTIME_ROOT", second), patch.object(
                    runtime, "_LEGACY_TMPFS_PREDECESSORS", (predecessor,)
                ):
                    migrated = runtime._migrate_legacy_runtime_state(legacy)
                    for key, value in (
                        ("container_tmpfs_options", "rw,noexec"),
                        ("legacy_qwen_runtime_sha256", "0" * 64),
                        ("migrated_from_schema", False),
                    ):
                        changed_state = {**migrated, key: value}
                        with self.subTest(key=key), patch.object(
                            runtime, "_private_json_read", return_value=changed_state
                        ), self.assertRaises(runtime.QwenRuntimeError):
                            runtime.current_runtime_state(root / "second/runtime.json")

    def test_teardown_only_state_is_never_reused(self):
        state = {**durable_runtime_state(runtime.RUNTIME_ROOT), "teardown_only": True}
        state.update(
            {
                "source_manifest_sha256": runtime._LEGACY_TMPFS_PREDECESSORS[0][0],
                "legacy_qwen_runtime_sha256": runtime._LEGACY_TMPFS_PREDECESSORS[0][1],
                "migrated_from_schema": runtime.LEGACY_SCHEMA_VERSION,
            }
        )
        runner = Mock()
        with patch.object(runtime, "current_runtime_state", return_value=state), self.assertRaises(
            runtime.QwenRuntimeError
        ):
            runtime.reuse_qwen_runtime(
                config={}, package_root=Path("/not/used"), command_runner=runner
            )
        runner.assert_not_called()

    def test_stop_persists_complete_migration_before_any_container_action(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state_path = root / "runtime.json"
            with patch.object(runtime, "RUNTIME_ROOT", root):
                legacy, predecessor = make_legacy_runtime_state(root, executable=False)
                runtime._private_json_write(state_path, legacy)
                events = []
                original_write = runtime._private_json_write

                def journal(path, value):
                    events.append(
                        ("write", value["schema_version"], value["phase"])
                    )
                    original_write(path, value)

                def resolve(value, **_kwargs):
                    events.append(("resolve", value["schema_version"], value["phase"]))
                    return "active", 4321, value

                def runner(*_args, **_kwargs):
                    events.append(("runner",))
                    return completed()

                with patch.object(
                    runtime, "_LEGACY_TMPFS_PREDECESSORS", (predecessor,)
                ), patch.object(
                    runtime, "docker_client_sha256", return_value=legacy["docker_sha256"]
                ), patch.object(
                    runtime,
                    "low_priority_wrapper_sha256",
                    return_value=legacy["wrapper_sha256"],
                ), patch.object(
                    runtime, "current_lease", return_value=lease(root)
                ), patch.object(
                    runtime, "_resolve_container", side_effect=resolve
                ), patch.object(
                    runtime, "_docker_inspect", side_effect=[{}, {}, None]
                ), patch.object(
                    runtime, "_inspect_identity", return_value=("exited", None)
                ), patch.object(
                    runtime, "_label_candidates", return_value=[]
                ), patch.object(
                    runtime, "_cleanup_run_directory", return_value=True
                ), patch.object(
                    runtime, "_private_json_write", side_effect=journal
                ):
                    self.assertTrue(
                        runtime.stop_qwen_runtime(
                            state_path=state_path,
                            command_runner=runner,
                            allow_lost_lease=True,
                        )
                    )
                self.assertEqual(events[0], ("write", runtime.SCHEMA_VERSION, "ready"))
                self.assertLess(
                    next(index for index, event in enumerate(events) if event[0] == "write"),
                    next(index for index, event in enumerate(events) if event[0] == "resolve"),
                )
                self.assertEqual(sum(event[0] == "runner" for event in events), 2)
                with patch.object(
                    runtime, "_LEGACY_TMPFS_PREDECESSORS", (predecessor,)
                ):
                    final = runtime.current_runtime_state(state_path)
                self.assertIs(final["teardown_only"], True)
                self.assertIs(final["scratch_cleaned"], True)

    def test_tmpfs_policy_is_bound_into_launch_digest(self):
        artifact = SimpleNamespace(
            manifest_sha256="1" * 64,
            sha256s_sha256="2" * 64,
            model_dir=Path("/models"),
        )
        source = SimpleNamespace(stage_dir=Path("/source"))
        arguments = dict(
            lease=lease(),
            deploy_environment={},
            artifact=artifact,
            source=source,
            image_id=qwen_capabilities.STANDARD_IMAGE_ID,
            image_config={},
            package_root=Path("/source"),
            container_name="aeon_qwen_test",
            port=8033,
            launch_nonce="c" * 64,
        )
        with patch.object(
            runtime,
            "_planner_contract",
            return_value=(["python3", "-m", "vllm"], "served"),
        ), patch.object(
            runtime, "_normalise_image_config", return_value=({}, {}, {"8000/tcp": {}})
        ), patch.object(runtime, "_mount_receipt", return_value={"source": "/exact"}):
            current = runtime._container_contract(**arguments)
            with patch.object(
                runtime,
                "_container_tmpfs_options",
                return_value="rw,nosuid,nodev,size=8589934592,uid=1000,gid=1000,mode=0700",
            ):
                predecessor = runtime._container_contract(**arguments)
        self.assertNotEqual(
            current["launch_spec_sha256"], predecessor["launch_spec_sha256"]
        )
        for contract in (current, predecessor):
            self.assertEqual(
                contract["labels"]["com.bc_aeon.launch-spec"],
                contract["launch_spec_sha256"],
            )


class DockerReceiptTests(HermeticDockerRootMixin, unittest.TestCase):
    def assert_inspect_rejected(self, item, state):
        with patch.object(
            runtime, "_mounts_match_live_pid", return_value=True
        ), self.assertRaises(runtime.QwenRuntimeError):
            runtime._inspect_identity(item, state)

    def test_full_receipt_accepts_exact_private_bounded_container(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            item = inspect_payload(state)
            with patch.object(runtime, "_mounts_match_live_pid", return_value=True):
                self.assertEqual(runtime._inspect_identity(item, state), ("active", 4321))

    def test_log_request_flag_is_bound_to_the_pinned_vllm_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            state["container_command"] += ["--no-enable-log-requests"]
            item = inspect_payload(state)
            with patch.object(runtime, "_mounts_match_live_pid", return_value=True):
                self.assertEqual(runtime._inspect_identity(item, state), ("active", 4321))

            changed_command = [
                "--disable-log-requests"
                if value == "--no-enable-log-requests"
                else value
                for value in state["container_command"]
            ]
            stale = copy.deepcopy(item)
            stale["Args"] = changed_command
            stale["Config"]["Cmd"] = changed_command
            self.assert_inspect_rejected(stale, state)

    def test_active_container_accepts_only_exact_missing_tmpfs_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            production = inspect_payload(state)
            production["Mounts"] = [
                mount
                for mount in production["Mounts"]
                if mount["Destination"] != "/workspace/cache"
            ]
            production["NetworkSettings"]["Networks"]["bridge"][
                "DNSNames"
            ] = None
            with patch.object(
                runtime, "_mounts_match_live_pid", return_value=True
            ) as live_mounts:
                self.assertEqual(
                    runtime._inspect_identity(production, state), ("active", 4321)
                )
            live_mounts.assert_called_once_with(state, 4321)

            missing_tmpfs_intent = copy.deepcopy(production)
            missing_tmpfs_intent["HostConfig"]["Tmpfs"] = {}
            self.assert_inspect_rejected(missing_tmpfs_intent, state)

            changed_tmpfs_intent = copy.deepcopy(production)
            changed_tmpfs_intent["HostConfig"]["Tmpfs"][
                "/workspace/cache"
            ] = changed_tmpfs_intent["HostConfig"]["Tmpfs"][
                "/workspace/cache"
            ].replace("mode=0700", "mode=0777")
            self.assert_inspect_rejected(changed_tmpfs_intent, state)

            exact_tmpfs = production["HostConfig"]["Tmpfs"]["/workspace/cache"]
            for changed_options in (
                exact_tmpfs.replace("rw,exec,", "rw,"),
                exact_tmpfs.replace("exec", "noexec"),
                exact_tmpfs + ",suid",
            ):
                changed_tmpfs = copy.deepcopy(production)
                changed_tmpfs["HostConfig"]["Tmpfs"][
                    "/workspace/cache"
                ] = changed_options
                with self.subTest(tmpfs_options=changed_options):
                    self.assert_inspect_rejected(changed_tmpfs, state)

            missing_bind = copy.deepcopy(production)
            missing_bind["Mounts"].pop()
            self.assert_inspect_rejected(missing_bind, state)

            for destination, read_write in (
                ("/workspace/cache", False),
                ("/workspace/cache/tmp", True),
                ("/foreign", True),
            ):
                extra = copy.deepcopy(production)
                extra["Mounts"].append(
                    {
                        "Type": "tmpfs",
                        "Source": "",
                        "Destination": destination,
                        "Mode": "",
                        "RW": read_write,
                        "Propagation": "",
                    }
                )
                with self.subTest(destination=destination, rw=read_write):
                    self.assert_inspect_rejected(extra, state)

            with patch.object(
                runtime, "_mounts_match_live_pid", return_value=False
            ), self.assertRaises(runtime.QwenRuntimeError):
                runtime._inspect_identity(production, state)

            missing_dns_names = copy.deepcopy(production)
            missing_dns_names["NetworkSettings"]["Networks"]["bridge"].pop(
                "DNSNames"
            )
            self.assert_inspect_rejected(missing_dns_names, state)

            container_name = state["container_name"]
            short_id = state["container_id"][:12]
            for dns_names in (
                [],
                [container_name],
                [short_id],
                [container_name, short_id, "foreign"],
                [container_name, short_id, container_name],
                False,
                "",
                {},
            ):
                changed_dns = copy.deepcopy(production)
                changed_dns["NetworkSettings"]["Networks"]["bridge"][
                    "DNSNames"
                ] = dns_names
                with self.subTest(dns_names=dns_names):
                    self.assert_inspect_rejected(changed_dns, state)

    def test_documented_omitted_config_defaults_are_canonical_only(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            baseline = inspect_payload(state)
            omitted_defaults = (
                "Healthcheck",
                "ArgsEscaped",
                "OnBuild",
                "StopSignal",
                "StopTimeout",
                "Shell",
            )
            item = copy.deepcopy(baseline)
            for field in omitted_defaults:
                item["Config"].pop(field)
            with patch.object(runtime, "_mounts_match_live_pid", return_value=True):
                self.assertEqual(
                    runtime._inspect_identity(item, state), ("active", 4321)
                )
            malformed = {
                "ArgsEscaped": (None, 0, 0.0, "false"),
                "StopSignal": (None, False, 0, "SIGTERM"),
                "Healthcheck": (False, {}, []),
                "StopTimeout": (False, 0, "0"),
            }
            for field, values in malformed.items():
                for value in values:
                    item = copy.deepcopy(baseline)
                    item["Config"][field] = value
                    with self.subTest(field=field, value=value):
                        self.assert_inspect_rejected(item, state)

    def test_explicit_gpu_device_request_uses_literal_zero_count(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            baseline = inspect_payload(state)
            request = baseline["HostConfig"]["DeviceRequests"][0]
            self.assertEqual(request["Count"], 0)
            for malformed in (False, -1, 1, 0.0, "0", None):
                item = copy.deepcopy(baseline)
                item["HostConfig"]["DeviceRequests"][0]["Count"] = malformed
                with self.subTest(count=malformed):
                    self.assert_inspect_rejected(item, state)

    def test_host_default_encodings_and_extra_restrictive_masks_are_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            item = inspect_payload(state)
            host = item["HostConfig"]
            host["Dns"] = None
            host["Ulimits"] = []
            host["MaskedPaths"] = [
                "/proc/interrupts",
                "/sys/devices/system/cpu/cpu127/thermal_throttle",
                *reversed(host["MaskedPaths"]),
            ]
            with patch.object(runtime, "_mounts_match_live_pid", return_value=True):
                self.assertEqual(
                    runtime._inspect_identity(item, state), ("active", 4321)
                )
            malformed_masks = (
                list(runtime._BASELINE_MASKED_PATHS[:-1]),
                [*runtime._BASELINE_MASKED_PATHS, "/etc/shadow"],
                [*runtime._BASELINE_MASKED_PATHS, runtime._BASELINE_MASKED_PATHS[0]],
                [*runtime._BASELINE_MASKED_PATHS, False],
            )
            for masked in malformed_masks:
                item = inspect_payload(state)
                item["HostConfig"]["MaskedPaths"] = masked
                with self.subTest(masked=masked):
                    self.assert_inspect_rejected(item, state)

    def test_exited_container_accepts_only_dockers_detached_runtime_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            exited = inspect_payload(state, running=False)
            exited["Mounts"] = [
                mount
                for mount in exited["Mounts"]
                if mount["Destination"] != "/workspace/cache"
            ]
            exited["NetworkSettings"]["Ports"] = {}
            bridge = exited["NetworkSettings"]["Networks"]["bridge"]
            bridge.update(
                {
                    "DNSNames": None,
                    "EndpointID": "",
                    "Gateway": "",
                    "IPAddress": "",
                    "IPPrefixLen": 0,
                    "IPv6Gateway": "",
                    "GlobalIPv6Address": "",
                    "GlobalIPv6PrefixLen": 0,
                    "MacAddress": "",
                }
            )
            self.assertEqual(runtime._inspect_identity(exited, state), ("exited", None))

            running = copy.deepcopy(exited)
            running["State"] = {"Running": True, "Pid": 4321}
            self.assert_inspect_rejected(running, state)

            bad_endpoint = copy.deepcopy(exited)
            bad_endpoint["NetworkSettings"]["Networks"]["bridge"]["IPAddress"] = (
                "172.17.0.2"
            )
            self.assert_inspect_rejected(bad_endpoint, state)

            for ambiguous_pid in (False, 1, 4321, 0.0, "0", None):
                bad_pid = copy.deepcopy(exited)
                bad_pid["State"]["Pid"] = ambiguous_pid
                with self.subTest(exited_pid=ambiguous_pid):
                    self.assert_inspect_rejected(bad_pid, state)

            missing_bind = copy.deepcopy(exited)
            missing_bind["Mounts"].pop()
            self.assert_inspect_rejected(missing_bind, state)

            for destination, malformed_rw in (("/models", 0), ("/workspace/cache", 1)):
                malformed_mount = inspect_payload(state)
                next(
                    mount
                    for mount in malformed_mount["Mounts"]
                    if mount["Destination"] == destination
                )["RW"] = malformed_rw
                with self.subTest(destination=destination, rw=malformed_rw):
                    self.assert_inspect_rejected(malformed_mount, state)

    def test_inherited_image_port_is_receipted_but_never_published(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            baseline = inspect_payload(state)
            self.assertEqual(
                baseline["Config"]["ExposedPorts"],
                {"8000/tcp": {}, "8033/tcp": {}},
            )
            self.assertEqual(
                baseline["HostConfig"]["PortBindings"],
                {
                    "8033/tcp": [
                        {"HostIp": "127.0.0.1", "HostPort": "8033"}
                    ]
                },
            )
            self.assertFalse(baseline["HostConfig"]["PublishAllPorts"])
            self.assertIsNone(baseline["NetworkSettings"]["Ports"]["8000/tcp"])

            mutations = []
            missing_config = copy.deepcopy(baseline)
            missing_config["Config"]["ExposedPorts"].pop("8000/tcp")
            mutations.append(missing_config)
            extra_config = copy.deepcopy(baseline)
            extra_config["Config"]["ExposedPorts"]["9000/tcp"] = {}
            mutations.append(extra_config)
            host_published = copy.deepcopy(baseline)
            host_published["HostConfig"]["PortBindings"]["8000/tcp"] = [
                {"HostIp": "127.0.0.1", "HostPort": "8000"}
            ]
            mutations.append(host_published)
            missing_network = copy.deepcopy(baseline)
            missing_network["NetworkSettings"]["Ports"].pop("8000/tcp")
            mutations.append(missing_network)
            network_published = copy.deepcopy(baseline)
            network_published["NetworkSettings"]["Ports"]["8000/tcp"] = [
                {"HostIp": "127.0.0.1", "HostPort": "8000"}
            ]
            mutations.append(network_published)
            for index, item in enumerate(mutations):
                with self.subTest(mutation=index):
                    self.assert_inspect_rejected(item, state)

    def test_every_config_and_top_level_identity_field_is_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            baseline = inspect_payload(state)
            for field, value in baseline["Config"].items():
                item = copy.deepcopy(baseline)
                item["Config"][field] = changed(value)
                with self.subTest(config=field):
                    self.assert_inspect_rejected(item, state)
            for field in ("Id", "Image", "Name", "Path", "Args", "Platform"):
                item = copy.deepcopy(baseline)
                item[field] = changed(item[field])
                with self.subTest(top_level=field):
                    self.assert_inspect_rejected(item, state)
            item = copy.deepcopy(baseline)
            item["Config"]["UnreceiptedFutureCapability"] = False
            self.assert_inspect_rejected(item, state)

    def test_every_hostconfig_capability_resource_namespace_and_runtime_is_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            baseline = inspect_payload(state)
            for field, value in baseline["HostConfig"].items():
                item = copy.deepcopy(baseline)
                item["HostConfig"][field] = changed(value)
                with self.subTest(host_config=field):
                    self.assert_inspect_rejected(item, state)
            for field, value in baseline["HostConfig"]["DeviceRequests"][0].items():
                item = copy.deepcopy(baseline)
                item["HostConfig"]["DeviceRequests"][0][field] = changed(value)
                with self.subTest(device_request=field):
                    self.assert_inspect_rejected(item, state)
            for index, mount in enumerate(baseline["HostConfig"]["Mounts"]):
                for field, value in mount.items():
                    item = copy.deepcopy(baseline)
                    item["HostConfig"]["Mounts"][index][field] = changed(value)
                    with self.subTest(host_mount=index, field=field):
                        self.assert_inspect_rejected(item, state)
            item = copy.deepcopy(baseline)
            item["HostConfig"]["UnreceiptedFutureCapability"] = False
            self.assert_inspect_rejected(item, state)

    def test_every_top_level_mount_field_and_network_boundary_is_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            baseline = inspect_payload(state)
            for index, mount in enumerate(baseline["Mounts"]):
                for field, value in mount.items():
                    item = copy.deepcopy(baseline)
                    item["Mounts"][index][field] = changed(value)
                    with self.subTest(mount=index, field=field):
                        self.assert_inspect_rejected(item, state)
            network = baseline["NetworkSettings"]
            for field in ("Ports",):
                item = copy.deepcopy(baseline)
                item["NetworkSettings"][field] = changed(network[field])
                with self.subTest(network=field):
                    self.assert_inspect_rejected(item, state)
            item = copy.deepcopy(baseline)
            item["NetworkSettings"]["Networks"]["foreign"] = {}
            self.assert_inspect_rejected(item, state)
            item = copy.deepcopy(baseline)
            item["NetworkSettings"]["UnreceiptedFutureField"] = ""
            self.assert_inspect_rejected(item, state)
            item = copy.deepcopy(baseline)
            item["NetworkSettings"]["Bridge"] = ""
            self.assert_inspect_rejected(item, state)
            for field in ("IPAMConfig", "Links", "Aliases", "DriverOpts"):
                item = copy.deepcopy(baseline)
                item["NetworkSettings"]["Networks"]["bridge"][field] = {
                    "unexpected": True
                }
                with self.subTest(bridge=field):
                    self.assert_inspect_rejected(item, state)
            item = copy.deepcopy(baseline)
            item["NetworkSettings"]["Networks"]["bridge"]["Unreceipted"] = None
            self.assert_inspect_rejected(item, state)
            expected_dns = [state["container_name"], state["container_id"][:12]]
            dns_mutations = (
                None,
                [state["container_name"]],
                [*expected_dns, "foreign-alias"],
                [*expected_dns, state["container_name"]],
                ["wrong-name", state["container_id"][:12]],
            )
            for dns_names in dns_mutations:
                item = copy.deepcopy(baseline)
                bridge = item["NetworkSettings"]["Networks"]["bridge"]
                if dns_names is None:
                    bridge.pop("DNSNames")
                else:
                    bridge["DNSNames"] = dns_names
                with self.subTest(dns_names=dns_names):
                    self.assert_inspect_rejected(item, state)

    def test_exact_exited_is_distinct_from_identity_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            self.assertEqual(runtime._inspect_identity(inspect_payload(state, running=False), state), ("exited", None))
            bad = inspect_payload(state, running=False)
            bad["Config"]["Labels"]["com.bc_aeon.claim"] = "gc-foreign"
            with self.assertRaises(runtime.QwenRuntimeError):
                runtime._inspect_identity(bad, state)

    def test_docker_command_is_uuid_only_private_bounded_and_offline(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime, "RUNTIME_ROOT", Path(temp)):
            state = inspect_state(Path(temp))
            state["run_dir"] = str(Path(temp) / "aeon-qwen38-vllm-bc-aeon-hermetic-owner")
            command = runtime._docker_run_command(state)
        joined = " ".join(command)
        self.assertIn(f"device={state['gpu_uuid']}", joined)
        self.assertNotIn("device=0", joined)
        self.assertIn("--shm-size 8589934592 --ipc private", joined)
        self.assertIn("--pids-limit 1024", joined)
        self.assertIn("--read-only", command)
        self.assertIn("--cap-drop", command)
        self.assertIn("--runtime runc", joined)
        self.assertIn("--cgroupns private", joined)
        self.assertIn("--hostname aeon_qwen_test", joined)
        self.assertEqual(
            [command[index + 1] for index, value in enumerate(command) if value == "--publish"],
            ["127.0.0.1:8033:8033"],
        )
        self.assertNotIn("--expose", command)
        self.assertNotIn("--attach", command)
        self.assertIn("--log-driver local", joined)
        self.assertIn(
            (
                "--tmpfs /workspace/cache:rw,exec,nosuid,nodev,size=8589934592,"
                f"uid={os.geteuid()},gid={os.getegid()},mode=0700"
            ),
            joined,
        )
        self.assertNotIn("noexec", joined)
        self.assertIn(
            f"--env TMPDIR={runtime.QWEN_CONTAINER_TMPDIR}", joined
        )
        self.assertNotIn("/workspace/cache/tmp", command)
        self.assertNotIn("mkdir", command)
        self.assertNotIn("-c", command)
        self.assertNotIn(".cache/huggingface:/", joined)
        self.assertEqual(
            command[:4],
            [
                str(runtime.HOST_BASH),
                str(runtime.FLEET_LOW_PRIORITY),
                str(runtime.HOST_BASH),
                str(runtime.DOCKER),
            ],
        )

    def test_tmpdir_is_the_existing_private_bounded_cache_mount(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            self.assertEqual(
                state["container_environment"]["TMPDIR"],
                "/workspace/cache",
            )
            exact = inspect_payload(state)
            with patch.object(runtime, "_mounts_match_live_pid", return_value=True):
                self.assertEqual(
                    runtime._inspect_identity(exact, state), ("active", 4321)
                )

            for stale in ("/workspace/cache/tmp", "/tmp"):
                changed_env = copy.deepcopy(exact)
                changed_env["Config"]["Env"] = [
                    f"{key}={stale if key == 'TMPDIR' else value}"
                    for key, value in state["container_environment"].items()
                ]
                with self.subTest(tmpdir=stale), self.assertRaises(
                    runtime.QwenRuntimeError
                ):
                    runtime._inspect_identity(changed_env, state)

    def test_docker_environment_is_fixed_empty_local_daemon_config(self):
        inherited = {
            "DOCKER_HOST": "tcp://untrusted.invalid:2375",
            "DOCKER_CONTEXT": "foreign",
            "DOCKER_TLS_VERIFY": "1",
            "DOCKER_CERT_PATH": "/tmp/foreign",
        }
        with patch.dict(os.environ, inherited, clear=False):
            first = runtime._docker_cli_environment()
            second = runtime._docker_cli_environment()
        self.assertEqual(first, second)
        self.assertEqual(first["DOCKER_HOST"], "unix:///var/run/docker.sock")
        self.assertEqual(set(first), {*runtime.HOST_LAUNCH_ENV, "DOCKER_HOST", "DOCKER_CONFIG"})
        self.assertNotIn("DOCKER_CONTEXT", first)
        self.assertNotIn("DOCKER_TLS_VERIFY", first)
        config_dir = Path(first["DOCKER_CONFIG"])
        self.assertEqual(config_dir.parent, runtime.RUNTIME_ROOT)
        self.assertEqual(list(config_dir.iterdir()), [])
        self.assertEqual(stat.S_IMODE(config_dir.lstat().st_mode), 0o700)

        (config_dir / "config.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(runtime.QwenRuntimeError):
            runtime._docker_cli_environment()

    def test_transitive_host_executable_receipts_change_with_real_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {
                "docker": root / "bin" / "docker",
                "real_docker": root / "system" / "docker",
                "bash": root / "system" / "bash",
                "fleet": root / "bin" / "fleet-low-priority",
                "nice": root / "system" / "nice",
                "ionice": root / "system" / "ionice",
                "sha256sum": root / "system" / "sha256sum",
                "python": root / "system" / "python3.12",
            }
            for directory in {path.parent for path in paths.values()}:
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                directory.chmod(0o700)
            for name, path in paths.items():
                path.write_text(f"{name}\n", encoding="utf-8")
                path.chmod(0o700)
            patches = (
                patch.object(runtime, "DOCKER", paths["docker"]),
                patch.object(runtime, "REAL_DOCKER", paths["real_docker"]),
                patch.object(runtime, "HOST_BASH", paths["bash"]),
                patch.object(runtime, "FLEET_LOW_PRIORITY", paths["fleet"]),
                patch.object(runtime, "HOST_NICE", paths["nice"]),
                patch.object(runtime, "HOST_IONICE", paths["ionice"]),
                patch.object(runtime, "HOST_SHA256SUM", paths["sha256sum"]),
                patch.object(runtime, "HOST_PYTHON", paths["python"]),
                patch.object(runtime, "SYSTEM_EXECUTABLE_UIDS", frozenset({os.geteuid()})),
            )
            for active in patches:
                active.start()
            try:
                docker_before = runtime.docker_client_sha256()
                low_before = runtime.low_priority_wrapper_sha256()
                for key in ("real_docker", "bash"):
                    original = paths[key].read_bytes()
                    paths[key].write_bytes(original + b"changed\n")
                    paths[key].chmod(0o700)
                    self.assertNotEqual(runtime.docker_client_sha256(), docker_before)
                    paths[key].write_bytes(original)
                    paths[key].chmod(0o700)
                for key in ("bash", "nice", "ionice", "sha256sum", "python"):
                    original = paths[key].read_bytes()
                    paths[key].write_bytes(original + b"changed\n")
                    paths[key].chmod(0o700)
                    self.assertNotEqual(runtime.low_priority_wrapper_sha256(), low_before)
                    paths[key].write_bytes(original)
                    paths[key].chmod(0o700)
            finally:
                for active in reversed(patches):
                    active.stop()

    def test_system_executable_receipt_binds_multicall_symlink_and_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "coreutils"
            alias = root / "nice"
            binary.write_text("multicall\n", encoding="utf-8")
            binary.chmod(0o700)
            alias.symlink_to(binary.name)
            root.chmod(0o700)
            with patch.object(
                runtime, "SYSTEM_EXECUTABLE_UIDS", frozenset({os.geteuid()})
            ):
                before = runtime._system_executable_receipt(
                    alias, label="test nice"
                )
                self.assertEqual(before["kind"], "symlink")
                self.assertEqual(before["resolved_path"], str(binary))
                binary.write_text("changed\n", encoding="utf-8")
                binary.chmod(0o700)
                after = runtime._system_executable_receipt(
                    alias, label="test nice"
                )
                self.assertNotEqual(before["sha256"], after["sha256"])

    def test_every_docker_command_site_uses_the_same_sanitized_environment(self):
        safe_env = {
            **runtime.HOST_LAUNCH_ENV,
            "DOCKER_HOST": runtime.DOCKER_HOST,
            "DOCKER_CONFIG": str(runtime.RUNTIME_ROOT / runtime.DOCKER_CONFIG_DIRNAME),
        }
        calls = []
        image_id = "sha256:" + "a" * 64
        container_id = "b" * 64
        with tempfile.TemporaryDirectory() as temp:
            docker_root = Path(temp) / "docker-root"
            docker_root.mkdir(mode=0o700)
            state = inspect_state(Path(temp))

            def runner(argv, **kwargs):
                calls.append((list(argv), kwargs.get("env")))
                arguments = list(argv)[2:]
                if arguments[:4] == ["image", "inspect", "--format", "{{.Id}}"]:
                    return completed(image_id + "\n")
                if arguments[:4] == ["image", "inspect", "--format", "{{.Size}}"]:
                    return completed("1024\n")
                if arguments[:2] == ["image", "inspect"]:
                    return completed(json.dumps([{"Id": image_id, "Config": {}}]))
                if arguments == ["info", "--format", "{{json .DockerRootDir}}"]:
                    return completed(json.dumps(str(docker_root)))
                if arguments == ["inspect", container_id]:
                    return completed(json.dumps([{"Id": container_id}]))
                if arguments[:3] == ["ps", "-aq", "--no-trunc"]:
                    return completed()
                raise AssertionError(arguments)

            with patch.object(
                runtime, "_docker_cli_environment", return_value=dict(safe_env)
            ):
                self.assertEqual(runtime.local_image_id("aeon_vllm:latest", command_runner=runner), image_id)
                self.assertEqual(runtime.local_image_size(image_id, command_runner=runner), 1024)
                self.assertEqual(runtime._image_config(image_id, command_runner=runner), {})
                self.assertEqual(runtime._docker_root(runner), docker_root)
                self.assertEqual(runtime._docker_inspect(container_id, runner), {"Id": container_id})
                self.assertEqual(runtime._label_candidates(state, runner), [])
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(environment == safe_env for _argv, environment in calls))

        source = Path(runtime.__file__).read_text(encoding="utf-8")
        # Docker preflight, inspect/absence, nonce recovery, create, stop and rm
        # are all explicit call sites. Each must route through the same helper.
        self.assertEqual(source.count("_docker_command("), 10)
        self.assertGreaterEqual(source.count("env=_docker_cli_environment()"), 9)
        start = source.index("result = command_runner(\n            docker_command")
        self.assertIn(
            "env=_docker_cli_environment()",
            source[start : start + 240],
        )

    def test_docker_inspect_accepts_only_exact_id_anchored_not_found(self):
        container_id = "a" * 64
        messages = (
            f"Error: No such object: {container_id}",
            f"Error response from daemon: No such container: {container_id}\n",
            f"Error: No such object: {container_id}\r\n",
        )
        for message in messages:
            for missing_stdout in ("", "[]\n"):
                responses = [
                    completed(missing_stdout, returncode=1, stderr=message),
                    completed("28.0.0\n"),
                ]

                def runner(*_args, **_kwargs):
                    return responses.pop(0)

                with self.subTest(message=message, stdout=missing_stdout):
                    self.assertIsNone(runtime._docker_inspect(container_id, runner))
                    self.assertEqual(responses, [])

        pinned_wrapper = [
            completed(
                "[]\n",
                returncode=1,
                stderr=f"error: no such object: {container_id}\n",
            ),
            completed("29.2.0\n"),
        ]
        self.assertIsNone(
            runtime._docker_inspect(
                container_id,
                lambda *_args, **_kwargs: pinned_wrapper.pop(0),
            )
        )
        self.assertEqual(pinned_wrapper, [])

    def test_lowercase_wrapper_not_found_requires_every_observed_byte(self):
        container_id = "a" * 64
        other_id = "b" * 64
        malformed = (
            ("", f"error: no such object: {container_id}\n"),
            ("[]\n", f"error: no such object: {container_id}"),
            ("[]\n", f"error: no such object: {container_id}\r\n"),
            ("[]\n", f"error: no such object: {container_id}\n\n"),
            ("[]\n", f"error: no such object: {other_id}\n"),
            ("[]\n", f"prefix error: no such object: {container_id}\n"),
            ("[]\n", f"error: no such object: {container_id} suffix\n"),
        )
        for stdout, stderr in malformed:
            with self.subTest(stdout=stdout, stderr=stderr), self.assertRaises(
                runtime.QwenRuntimeError
            ):
                runtime._docker_inspect(
                    container_id,
                    lambda *_args, **_kwargs: completed(
                        stdout, returncode=1, stderr=stderr
                    ),
                )

        for returncode in (2, 3, 64, 127, -1):
            runner = Mock(
                return_value=completed(
                    "[]\n",
                    returncode=returncode,
                    stderr=f"error: no such object: {container_id}\n",
                )
            )
            with self.subTest(returncode=returncode), self.assertRaises(
                runtime.QwenRuntimeError
            ):
                runtime._docker_inspect(container_id, runner)
            runner.assert_called_once()

    def test_canonical_not_found_also_requires_docker_returncode_one(self):
        container_id = "a" * 64
        diagnostics = (
            f"Error: No such object: {container_id}\n",
            f"Error response from daemon: No such container: {container_id}\n",
        )
        for returncode in (2, 3, 64, 127, -1):
            for stderr in diagnostics:
                runner = Mock(
                    return_value=completed(
                        "[]\n", returncode=returncode, stderr=stderr
                    )
                )
                with self.subTest(
                    returncode=returncode, stderr=stderr
                ), self.assertRaises(runtime.QwenRuntimeError):
                    runtime._docker_inspect(container_id, runner)
                runner.assert_called_once()

    def test_docker_inspect_rejects_ambiguous_errors_and_unhealthy_daemon(self):
        container_id = "a" * 64
        other_id = "b" * 64
        ambiguous = (
            "",
            "permission denied",
            "unexpected EOF",
            f"Error: No such object: {other_id}",
            f"prefix Error: No such object: {container_id}",
            f"Error: No such object: {container_id} suffix",
            f"Error: No such object: {container_id}\n\n",
        )
        for diagnostic in ambiguous:
            with self.subTest(diagnostic=diagnostic), self.assertRaises(
                runtime.QwenRuntimeError
            ):
                runtime._docker_inspect(
                    container_id,
                    lambda *_args, **_kwargs: completed(
                        returncode=1, stderr=diagnostic
                    ),
                )
        for unexpected_stdout in ("unexpected stdout", "[]", "[]\r\n", "[ ]\n"):
            with self.subTest(stdout=unexpected_stdout), self.assertRaises(
                runtime.QwenRuntimeError
            ):
                runtime._docker_inspect(
                    container_id,
                    lambda *_args, **_kwargs: completed(
                        unexpected_stdout,
                        returncode=1,
                        stderr=f"Error: No such object: {container_id}",
                    ),
                )
        unhealthy_daemons = (
            completed(returncode=1, stderr="daemon unavailable"),
            completed(),
            completed("28.0.0\n", stderr="warning"),
            completed("not a version with spaces\n"),
        )
        for daemon in unhealthy_daemons:
            responses = [
                completed(
                    returncode=1,
                    stderr=f"Error: No such object: {container_id}",
                ),
                daemon,
            ]
            with self.subTest(daemon=daemon), self.assertRaises(
                runtime.QwenRuntimeError
            ):
                runtime._docker_inspect(
                    container_id, lambda *_args, **_kwargs: responses.pop(0)
                )

        def unavailable(*_args, **_kwargs):
            raise subprocess.TimeoutExpired("docker", 20)

        with self.assertRaises(runtime.QwenRuntimeError):
            runtime._docker_inspect(container_id, unavailable)

    def test_docker_inspect_success_is_one_exact_same_id_object(self):
        container_id = "a" * 64
        exact = {"Id": container_id, "Config": {}}
        self.assertEqual(
            runtime._docker_inspect(
                container_id,
                lambda *_args, **_kwargs: completed(json.dumps([exact])),
            ),
            exact,
        )
        malformed = (
            "",
            "{}",
            "[]",
            json.dumps([exact, exact]),
            json.dumps([{"Id": "b" * 64}]),
            json.dumps(["not-an-object"]),
        )
        for payload in malformed:
            with self.subTest(payload=payload), self.assertRaises(
                runtime.QwenRuntimeError
            ):
                runtime._docker_inspect(
                    container_id,
                    lambda *_args, **_kwargs: completed(payload),
                )
        with self.assertRaises(runtime.QwenRuntimeError):
            runtime._docker_inspect(
                container_id,
                lambda *_args, **_kwargs: completed(
                    json.dumps([exact]), stderr="warning"
                ),
            )

    def test_nonce_scan_is_exact_and_absence_requires_empty_scan(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            calls = []

            def scan_runner(argv, **_kwargs):
                calls.append(argv)
                return completed()

            self.assertEqual(runtime._label_candidates(state, scan_runner), [])
            self.assertIn(
                f"label=com.bc_aeon.claim={state['claim_id']}", calls[0]
            )
            self.assertIn(
                f"label=com.bc_aeon.launch-nonce={state['launch_nonce']}",
                calls[0],
            )
            with patch.object(runtime, "_docker_inspect", return_value=None), patch.object(
                runtime, "_label_candidates", return_value=[]
            ):
                self.assertEqual(runtime._resolve_container(state)[:2], ("gone", None))
            responses = [
                completed(
                    "[]\n",
                    returncode=1,
                    stderr=f"Error: No such object: {state['container_id']}\n",
                ),
                completed("28.0.0\n"),
                completed(),
            ]
            self.assertEqual(
                runtime._resolve_container(
                    state,
                    command_runner=lambda *_args, **_kwargs: responses.pop(0),
                )[:2],
                ("gone", None),
            )
            self.assertEqual(responses, [])
            with patch.object(runtime, "_docker_inspect", return_value=None), patch.object(
                runtime, "_label_candidates", return_value=["e" * 64]
            ), self.assertRaises(runtime.QwenRuntimeError):
                runtime._resolve_container(state)

    def test_final_gate_rechecks_claim_after_resources_and_before_create(self):
        fake_char = SimpleNamespace(st_mode=stat.S_IFCHR | 0o600)
        calls = []
        with patch.object(runtime, "_validate_lease", return_value=lease()), patch.object(
            runtime.socket, "gethostname", return_value=runtime.LOCAL_COORD_HOSTNAME
        ), patch.object(Path, "lstat", return_value=fake_char), patch.object(
            runtime, "_read_meminfo", return_value=(200 * 1024**3, 200 * 1024**3)
        ), patch.object(runtime, "_disk_free", return_value=200 * 1024**3), patch.object(
            runtime, "_docker_root", return_value=Path("/var/lib/docker")
        ), patch.object(runtime, "docker_client_sha256", return_value="a" * 64), patch.object(
            runtime, "low_priority_wrapper_sha256", return_value="b" * 64
        ), patch.object(runtime, "verify_coordinator_lease", side_effect=lambda *_: calls.append("coord")), patch.object(
            runtime.subprocess, "run", return_value=completed("user:aday:rwx\n")
        ):
            runtime.final_launch_admission_gate(
                lease(),
                expected_wrapper_sha256="b" * 64,
                expected_docker_sha256="a" * 64,
                command_runner=lambda *_args, **_kwargs: completed("user:aday:rwx\n"),
            )
        self.assertEqual(calls, ["coord"])
        source = Path(runtime.__file__).read_text(encoding="utf-8")
        start = source.index(
            "final_launch_admission_gate(", source.index("def start_local_runtime")
        )
        create = source.index("result = command_runner(", start)
        between = source[start:create]
        self.assertNotIn("load_artifact_identity", between)
        self.assertNotIn("_container_contract", between)
        create_call = source[create : source.index("if result.returncode", create)]
        self.assertIn("umask=0o077", create_call)

    def test_final_gate_rejects_wrapper_or_docker_change_before_acl_or_create(self):
        for docker_hash, wrapper_hash in (("c" * 64, "b" * 64), ("a" * 64, "c" * 64)):
            command_runner = Mock()
            with self.subTest(docker_hash=docker_hash[:1], wrapper_hash=wrapper_hash[:1]), patch.object(
                runtime, "_validate_lease", return_value=lease()
            ), patch.object(runtime, "docker_client_sha256", return_value=docker_hash), patch.object(
                runtime, "low_priority_wrapper_sha256", return_value=wrapper_hash
            ), self.assertRaises(runtime.QwenRuntimeError):
                runtime.final_launch_admission_gate(
                    lease(),
                    expected_wrapper_sha256="b" * 64,
                    expected_docker_sha256="a" * 64,
                    command_runner=command_runner,
                )
            command_runner.assert_not_called()


class CrashConsistencyTests(HermeticDockerRootMixin, unittest.TestCase):
    def test_lost_run_response_recovers_one_nonce_id_not_name(self):
        with tempfile.TemporaryDirectory() as temp:
            state = inspect_state(Path(temp))
            state["run_dir"] = str(
                Path(temp) / "aeon-qwen38-vllm-bc-aeon-hermetic-owner"
            )
            state["container_id"] = None
            candidate = "e" * 64
            item = inspect_payload({**state, "container_id": candidate})
            runner = Mock()
            with patch.object(runtime, "RUNTIME_ROOT", Path(temp)), patch.object(runtime, "_read_cidfile", return_value=None), patch.object(
                runtime, "_label_candidates", return_value=[candidate]
            ), patch.object(runtime, "_docker_inspect", return_value=item), patch.object(
                runtime, "_inspect_identity", return_value=("active", 4321)
            ), patch.object(runtime, "_private_json_write") as write:
                status, pid, adopted = runtime._resolve_container(
                    state, command_runner=runner, adopt=True, state_path=Path(temp) / "runtime.json"
                )
            self.assertEqual((status, pid, adopted["container_id"]), ("active", 4321, candidate))
            write.assert_called_once()
            with patch.object(runtime, "RUNTIME_ROOT", Path(temp)), patch.object(runtime, "_read_cidfile", return_value=None), patch.object(
                runtime, "_label_candidates", return_value=[candidate, "f" * 64]
            ), self.assertRaises(runtime.QwenRuntimeError):
                runtime._resolve_container(state, command_runner=runner)

    def test_unsafe_cidfile_is_ignored_for_exact_nonce_adoption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "aeon-qwen38-vllm-bc-aeon-hermetic-owner"
            run_dir.mkdir(mode=0o700)
            cidfile = run_dir / "container.cid"
            cidfile.write_text("a" * 64 + "\n", encoding="ascii")
            cidfile.chmod(0o664)
            state = {
                **inspect_state(root),
                "phase": "preflight",
                "run_dir": str(run_dir),
                "container_id": None,
            }
            candidate = "e" * 64
            item = {"Id": candidate}
            with patch.object(runtime, "RUNTIME_ROOT", root), patch.object(
                runtime, "_label_candidates", return_value=[candidate]
            ) as labels, patch.object(
                runtime, "_docker_inspect", return_value=item
            ) as inspect, patch.object(
                runtime, "_inspect_identity", return_value=("exited", None)
            ), patch.object(runtime, "_private_json_write") as write:
                status, pid, adopted = runtime._resolve_container(
                    state,
                    adopt=True,
                    state_path=root / "runtime.json",
                )
            self.assertEqual((status, pid), ("exited", None))
            self.assertEqual(adopted["container_id"], candidate)
            self.assertIs(adopted["cidfile_recovery_authorized"], True)
            labels.assert_called_once()
            inspect.assert_called_once_with(candidate, runtime.subprocess.run)
            write.assert_called_once()

    def test_unsafe_cidfile_absence_and_multiplicity_are_label_proven(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "aeon-qwen38-vllm-bc-aeon-hermetic-owner"
            run_dir.mkdir(mode=0o700)
            cidfile = run_dir / "container.cid"
            cidfile.write_text("not-an-id\n", encoding="ascii")
            cidfile.chmod(0o664)
            state = {
                **inspect_state(root),
                "phase": "preflight",
                "run_dir": str(run_dir),
                "container_id": None,
            }
            with patch.object(runtime, "RUNTIME_ROOT", root), patch.object(
                runtime, "_label_candidates", return_value=[]
            ), patch.object(runtime, "_private_json_write") as write:
                status, pid, recovered = runtime._resolve_container(
                    state, adopt=True, state_path=root / "runtime.json"
                )
            self.assertEqual((status, pid), ("gone", None))
            self.assertIs(recovered["cidfile_recovery_authorized"], True)
            write.assert_called_once()

            with patch.object(runtime, "RUNTIME_ROOT", root), patch.object(
                runtime, "_label_candidates", return_value=["e" * 64, "f" * 64]
            ), self.assertRaises(runtime.QwenRuntimeError):
                runtime._resolve_container(state)

    def test_saved_immutable_id_still_journals_legacy_cidfile_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "aeon-qwen38-vllm-bc-aeon-hermetic-owner"
            run_dir.mkdir(mode=0o700)
            cidfile = run_dir / "container.cid"
            cidfile.write_text("malformed\n", encoding="ascii")
            cidfile.chmod(0o664)
            candidate = "b" * 64
            state = {
                **inspect_state(root),
                "phase": "ready",
                "run_dir": str(run_dir),
                "container_id": candidate,
                "container_pid": 4321,
            }
            with patch.object(runtime, "RUNTIME_ROOT", root), patch.object(
                runtime, "_docker_inspect", return_value={"Id": candidate}
            ), patch.object(
                runtime, "_inspect_identity", return_value=("active", 4321)
            ), patch.object(
                runtime, "_label_candidates", return_value=[candidate]
            ) as labels, patch.object(runtime, "_private_json_write") as write:
                status, pid, adopted = runtime._resolve_container(
                    state, adopt=True, state_path=root / "runtime.json"
                )
            self.assertEqual((status, pid), ("active", 4321))
            self.assertIs(adopted["cidfile_recovery_authorized"], True)
            labels.assert_called_once()
            write.assert_called_once()

            with patch.object(runtime, "RUNTIME_ROOT", root), patch.object(
                runtime, "_docker_inspect", return_value={"Id": candidate}
            ), patch.object(
                runtime, "_inspect_identity", return_value=("active", 4321)
            ), patch.object(
                runtime, "_label_candidates", return_value=["e" * 64]
            ), self.assertRaises(runtime.QwenRuntimeError):
                runtime._resolve_container(state, adopt=True)

    def test_recovered_cidfile_cleanup_relaxes_only_contents_and_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "aeon-qwen38-vllm-bc-aeon-hermetic-owner"
            run_dir.mkdir(mode=0o700)
            state = {**inspect_state(root), "run_dir": str(run_dir)}
            cidfile = run_dir / "container.cid"
            cidfile.write_text("malformed\n", encoding="ascii")
            cidfile.chmod(0o664)
            with patch.object(runtime, "RUNTIME_ROOT", root):
                self.assertFalse(runtime._remove_cidfile(state))
                self.assertTrue(cidfile.exists())
                self.assertTrue(
                    runtime._remove_cidfile(state, recovery_authorized=True)
                )
            self.assertFalse(cidfile.exists())

            target = run_dir / "target"
            target.write_text("foreign\n", encoding="ascii")
            cidfile.symlink_to(target)
            with patch.object(runtime, "RUNTIME_ROOT", root):
                self.assertFalse(
                    runtime._remove_cidfile(state, recovery_authorized=True)
                )
            self.assertTrue(cidfile.is_symlink())
            cidfile.unlink()

            cidfile.write_text("a" * 64, encoding="ascii")
            hardlink = run_dir / "hardlink"
            os.link(cidfile, hardlink)
            with patch.object(runtime, "RUNTIME_ROOT", root):
                self.assertFalse(
                    runtime._remove_cidfile(state, recovery_authorized=True)
                )
            self.assertTrue(cidfile.exists())

    def test_exact_exited_nonce_candidate_recovers_stop_with_legacy_cidfile_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "aeon-qwen38-vllm-bc-aeon-hermetic-owner"
            run_dir.mkdir(mode=0o700)
            cidfile = run_dir / "container.cid"
            cidfile.write_text("untrusted-content\n", encoding="ascii")
            cidfile.chmod(0o664)
            state = {
                **inspect_state(root),
                "phase": "preflight",
                "run_dir": str(run_dir),
                "container_id": None,
                "scratch_cleaned": False,
            }
            value = lease(root)
            candidate = "e" * 64
            item = {"Id": candidate}
            writes = []
            runner = Mock(return_value=completed())
            with patch.object(runtime, "RUNTIME_ROOT", root), patch.object(
                runtime, "current_runtime_state", return_value=state
            ), patch.object(runtime, "current_lease", return_value=value), patch.object(
                runtime, "runtime_state_matches_lease", return_value=True
            ), patch.object(
                runtime, "docker_client_sha256", return_value=state["docker_sha256"]
            ), patch.object(
                runtime,
                "low_priority_wrapper_sha256",
                return_value=state["wrapper_sha256"],
            ), patch.object(
                runtime, "_label_candidates", side_effect=[[candidate], []]
            ), patch.object(
                runtime, "_docker_inspect", side_effect=[item, item, None]
            ), patch.object(
                runtime, "_inspect_identity", return_value=("exited", None)
            ), patch.object(
                runtime, "cleanup_local_source_stage", return_value=True
            ), patch.object(
                runtime,
                "_private_json_write",
                side_effect=lambda _path, value: writes.append(dict(value)),
            ):
                self.assertTrue(runtime.stop_qwen_runtime(command_runner=runner))
            self.assertFalse(run_dir.exists())
            self.assertTrue(
                any(value.get("cidfile_recovery_authorized") is True for value in writes)
            )
            self.assertTrue(writes[-1]["scratch_cleaned"])
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.args[0][-3:-1], ["rm", "-v"])

    def test_exact_id_stop_uses_id_and_journals_before_release(self):
        state = {**inspect_state(Path(tempfile.mkdtemp())), "phase": "ready", "scratch_cleaned": False}
        value = lease(Path(state["run_dir"]).parent)
        writes = []
        commands = []
        command_environments = []

        def runner(argv, **kwargs):
            commands.append(argv)
            command_environments.append(kwargs.get("env"))
            return completed()

        with patch.object(runtime, "current_runtime_state", return_value=state), patch.object(
            runtime, "current_lease", return_value=value
        ), patch.object(
            runtime, "docker_client_sha256", return_value=state["docker_sha256"]
        ), patch.object(
            runtime, "low_priority_wrapper_sha256", return_value=state["wrapper_sha256"]
        ), patch.object(runtime, "runtime_state_matches_lease", return_value=True), patch.object(
            runtime, "_resolve_container", side_effect=[("active", 4321, state)]
        ), patch.object(runtime, "_docker_inspect", side_effect=[{}, {}, None]), patch.object(
            runtime, "_inspect_identity", return_value=("exited", None)
        ), patch.object(runtime, "_label_candidates", return_value=[]), patch.object(
            runtime, "_cleanup_run_directory", return_value=True
        ), patch.object(runtime, "_private_json_write", side_effect=lambda _p, value: writes.append(dict(value))):
            self.assertTrue(runtime.stop_qwen_runtime(command_runner=runner))
        self.assertEqual(commands[0][-1], state["container_id"])
        self.assertEqual(commands[1][-1], state["container_id"])
        self.assertNotIn(state["container_name"], commands[1])
        self.assertEqual(
            command_environments,
            [runtime._docker_cli_environment(), runtime._docker_cli_environment()],
        )
        self.assertEqual(writes[0]["phase"], "releasing")
        self.assertTrue(writes[-1]["scratch_cleaned"])

    def test_release_to_state_clear_crash_replays_when_claim_already_absent(self):
        state = {**inspect_state(Path(tempfile.mkdtemp())), "phase": "releasing", "scratch_cleaned": True}
        with patch.object(runtime, "current_runtime_state", return_value=state), patch.object(
            runtime, "current_lease", return_value=None
        ), patch.object(runtime, "_coordinator_claim_matches", side_effect=[(0, None), (0, None)]), patch.object(
            runtime, "clear_runtime_state"
        ) as clear:
            self.assertTrue(runtime.finalize_releasing_qwen_runtime())
        clear.assert_called_once()

    def test_coordinator_absent_clears_only_exact_remaining_local_lease(self):
        state = {**inspect_state(Path(tempfile.mkdtemp())), "phase": "releasing", "scratch_cleaned": True}
        value = lease(Path(state["run_dir"]).parent)
        with patch.object(runtime, "current_runtime_state", return_value=state), patch.object(
            runtime, "current_lease", return_value=value
        ), patch.object(runtime, "runtime_state_matches_lease", return_value=True), patch.object(
            runtime, "_coordinator_claim_matches", side_effect=[(0, None), (0, None)]
        ), patch.object(
            runtime, "clear_reconciled_lease_state"
        ) as local_clear, patch.object(runtime, "release_vram") as coordinator_release, patch.object(
            runtime, "clear_runtime_state"
        ) as runtime_clear:
            self.assertTrue(runtime.finalize_releasing_qwen_runtime())
        local_clear.assert_called_once_with(
            runtime.QWEN_LEASE_FILE,
            expected_claim_id=state["claim_id"],
            expected_owner=state["owner"],
            expected_run_dir=state["run_dir"],
        )
        coordinator_release.assert_not_called()
        runtime_clear.assert_called_once()

    def test_release_reconciliation_quarantines_global_claim_id_collisions(self):
        state = inspect_state(Path(tempfile.mkdtemp()))
        exact = inventory_for(state)[0]
        cases = []
        moved = json.loads(json.dumps(exact))
        moved["claims"][0]["owner"] = "foreign-owner"
        cases.append([moved])
        occupied = json.loads(json.dumps(exact))
        occupied["claims"][0]["claim_id"] = "gc-foreign-occupied"
        cases.append([occupied])
        for target_key, changed in (
            ("host", "192.168.0.179"),
            ("physical_gpu", 1),
            ("uuid", "GPU-bbbbbbbb-cccc-dddd-eeee-ffffffffffff"),
        ):
            mutated = json.loads(json.dumps(exact))
            mutated[target_key] = changed
            cases.append([mutated])
        claim_uuid = json.loads(json.dumps(exact))
        claim_uuid["claims"][0]["gpu_uuid"] = (
            "GPU-bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
        )
        cases.append([claim_uuid])
        budget = json.loads(json.dumps(exact))
        budget["claims"][0]["vram_budget_mib"] -= 1
        cases.append([budget])
        float_budget = json.loads(json.dumps(exact))
        float_budget["claims"][0]["vram_budget_mib"] = float(
            float_budget["claims"][0]["vram_budget_mib"]
        )
        cases.append([float_budget])
        float_total = json.loads(json.dumps(exact))
        float_total["memory_total_mib"] = float(float_total["memory_total_mib"])
        cases.append([float_total])
        for malformed_exclusive in (True, False, 0, 1.0, "1", None):
            malformed_status = json.loads(json.dumps(exact))
            malformed_status["claims"][0]["exclusive"] = malformed_exclusive
            cases.append([malformed_status])
        cases.append([exact, json.loads(json.dumps(exact))])
        for inventory in cases:
            with self.subTest(inventory=inventory), patch.object(
                runtime, "_coord", return_value=completed(json.dumps(inventory))
            ), self.assertRaises(runtime.QwenRuntimeError):
                runtime._coordinator_claim_matches(state)

    def test_malformed_release_inventory_cannot_clear_lease_or_runtime(self):
        state = {
            **inspect_state(Path(tempfile.mkdtemp())),
            "phase": "releasing",
            "scratch_cleaned": True,
        }
        value = lease(Path(state["run_dir"]).parent)
        exact_target = inventory_for(value)[0]
        malformed_targets = []
        missing = copy.deepcopy(exact_target)
        missing.pop("claims")
        malformed_targets.append(missing)
        for malformed_claims in (None, False, {}):
            target = copy.deepcopy(exact_target)
            target["claims"] = malformed_claims
            malformed_targets.append(target)
        only_gpu1 = copy.deepcopy(exact_target)
        only_gpu1["physical_gpu"] = 1
        only_gpu1["uuid"] = "GPU-bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
        only_gpu1["claims"] = []
        malformed_targets.append(only_gpu1)
        changed_target_uuid = copy.deepcopy(exact_target)
        changed_target_uuid["uuid"] = "GPU-bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
        changed_target_uuid["claims"] = []
        malformed_targets.append(changed_target_uuid)
        for target in malformed_targets:
            with self.subTest(claims=target.get("claims", "missing")), patch.object(
                runtime, "current_runtime_state", return_value=state
            ), patch.object(
                runtime, "current_lease", return_value=value
            ), patch.object(
                runtime, "_coord", return_value=completed(json.dumps([target]))
            ), patch.object(
                runtime, "clear_reconciled_lease_state"
            ) as clear_lease, patch.object(
                runtime, "clear_runtime_state"
            ) as clear_runtime, patch.object(
                runtime, "release_vram"
            ) as release, self.assertRaises(runtime.QwenRuntimeError):
                runtime.finalize_releasing_qwen_runtime()
            clear_lease.assert_not_called()
            clear_runtime.assert_not_called()
            release.assert_not_called()

    def test_unrelated_unavailable_worker_does_not_wedge_exact_local_claim_view(self):
        state = inspect_state(Path(tempfile.mkdtemp()))
        exact = inventory_for(state)[0]
        unrelated_unavailable = {
            "host": "192.168.0.179",
            "physical_gpu": None,
            "state": "UNAVAILABLE",
        }
        with patch.object(
            runtime,
            "_coord",
            return_value=completed(json.dumps([exact, unrelated_unavailable])),
        ):
            count, claim = runtime._coordinator_claim_matches(state)
        self.assertEqual(count, 1)
        self.assertEqual(claim["claim_id"], state["claim_id"])
        exact_absent = copy.deepcopy(exact)
        exact_absent["claims"] = []
        with patch.object(
            runtime,
            "_coord",
            return_value=completed(
                json.dumps([exact_absent, unrelated_unavailable])
            ),
        ):
            self.assertEqual(runtime._coordinator_claim_matches(state), (0, None))

    def test_active_lost_claim_requires_exact_stop_before_readmission(self):
        source = Path(__file__).resolve().parents[1] / "main.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("stop_qwen_runtime(allow_lost_lease=True)", text)
        self.assertIn("finalize_releasing_qwen_runtime", text)
        self.assertIn("stop_managed_remote_runtime(", text)
        self.assertIn("after claim loss", text)

    def test_heartbeat_claim_loss_has_exact_stop_then_foreground_readmission(self):
        source = Path(__file__).resolve().parents[1] / "main.py"
        text = source.read_text(encoding="utf-8")
        ensure = text.index("def ensure_qwen_compute(self)")
        exit_method = text.index("def enter(self, model_config", ensure)
        block = text[ensure:exit_method]
        self.assertIn("start_llamacpp_server_serialized(config)", block)
        self.assertIn("self._start_qwen_heartbeat(config)", block)


class FinalOwnerTeardownTests(unittest.TestCase):
    def test_failed_last_owner_stop_is_durable_and_retried(self):
        from aeon import main

        model = "hermetic-qwen-model"
        reference = main._process_reference()
        for first_failure in (False, RuntimeError("transient stop")):
            with self.subTest(first_failure=type(first_failure).__name__), tempfile.TemporaryDirectory() as temp, patch.object(
                main, "MODEL_REGISTRY_PATH", str(Path(temp) / "registry.json")
            ), patch.object(
                main, "MODEL_REGISTRY_LOCK_PATH", str(Path(temp) / "registry.lock")
            ), patch.object(
                main, "QWEN_STARTUP_LOCK_PATH", str(Path(temp) / "lifecycle.lock")
            ), patch.object(
                main, "get_llamacpp_config", return_value={"model": model}
            ), patch.object(
                main, "stop_llamacpp_server", side_effect=[first_failure, True]
            ) as stop:
                main._write_model_registry({model: [reference]}, set())
                with self.assertRaises(RuntimeError):
                    main.unregister_models_for_agent([model])
                _registry, pending = main._read_model_registry()
                self.assertEqual(pending, {model})
                self.assertEqual(_registry, {model: [reference]})
                main.unregister_models_for_agent([model])
                registry, pending = main._read_model_registry()
                self.assertEqual((registry, pending), ({}, set()))
                self.assertEqual(stop.call_count, 2)

    def test_startup_ghost_retry_preserves_dead_owner_until_exact_stop(self):
        from aeon import main

        model = "hermetic-qwen-model"
        dead_reference = {"pid": 2147483647, "process_create_time": 1.0}
        with tempfile.TemporaryDirectory() as temp, patch.object(
            main, "MODEL_REGISTRY_PATH", str(Path(temp) / "registry.json")
        ), patch.object(
            main, "MODEL_REGISTRY_LOCK_PATH", str(Path(temp) / "registry.lock")
        ), patch.object(
            main, "QWEN_STARTUP_LOCK_PATH", str(Path(temp) / "lifecycle.lock")
        ), patch.object(
            main, "get_llamacpp_config", return_value={"model": model}
        ), patch.object(
            main, "LLAMACPP_MODELS", [{"model": model}]
        ), patch.object(
            main, "stop_llamacpp_server", side_effect=[False, True]
        ) as stop, patch.object(
            runtime, "current_runtime_state", return_value=None
        ):
            main._write_model_registry({model: [dead_reference]}, set())
            main.cleanup_ghost_llamacpp_containers()
            registry, pending = main._read_model_registry()
            self.assertEqual(registry, {model: [dead_reference]})
            self.assertEqual(pending, {model})
            main.cleanup_ghost_llamacpp_containers()
            self.assertEqual(main._read_model_registry(), ({}, set()))
            self.assertEqual(stop.call_count, 2)

    def test_ambiguous_stop_keeps_heartbeat_until_exact_absence(self):
        from aeon import main

        heartbeat = Mock()
        heartbeat.beat_once.side_effect = [RuntimeError("beat transport"), None]
        config = {"model": "hermetic-qwen", "_startup_heartbeat": heartbeat}
        with patch.object(main, "current_lease", create=True), patch(
            "aeon.core.qwen_runtime.current_runtime_state", return_value={"phase": "ready"}
        ), patch(
            "aeon.core.qwen_runtime.stop_qwen_runtime", side_effect=[False, False, True]
        ), patch(
            "aeon.core.qwen_runtime.qwen_runtime_liveness", side_effect=["ambiguous", "active"]
        ), patch(
            "aeon.core.qwen_runtime.finalize_releasing_qwen_runtime", return_value=True
        ), patch.object(main.time, "sleep", return_value=None):
            self.assertTrue(main._stop_llamacpp_server_locked(config))
        self.assertEqual(heartbeat.beat_once.call_count, 2)
        heartbeat.stop.assert_called_once()

    def test_session_cleanup_stays_retryable_while_teardown_pending(self):
        from aeon import main

        manager = main.SessionManager()
        manager._models_used = ["hermetic-qwen"]
        heartbeat = Mock()
        manager._lease_heartbeats = [heartbeat]
        with patch.object(main, "terminate_all_sub_agents"), patch.object(
            main, "cleanup_transient_tools"
        ), patch.object(
            main, "unregister_models_for_agent", side_effect=RuntimeError("pending")
        ), self.assertRaises(RuntimeError):
            manager.exit()
        self.assertFalse(manager._cleanup_done)
        self.assertFalse(manager._cleanup_in_progress)
        heartbeat.stop.assert_not_called()


class LoopbackHttpTests(unittest.TestCase):
    def test_stdlib_loopback_get_streams_bounded_response_without_redirects(self):
        response = Mock()
        response.status = 200
        response.headers = {"content-length": "2"}
        response.read.side_effect = [b"{}", b""]
        connection = Mock()
        connection.getresponse.return_value = response

        with patch.object(
            runtime.http.client, "HTTPConnection", return_value=connection
        ) as factory:
            result = runtime._loopback_get(
                "http://127.0.0.1:8033/v1/models", timeout=3
            )
            self.assertEqual(runtime._bounded_loopback_body(result, 32), b"{}")

        factory.assert_called_once_with("127.0.0.1", 8033, timeout=3)
        connection.request.assert_called_once_with(
            "GET",
            "/v1/models",
            headers={"Accept": "application/json", "Connection": "close"},
        )
        response.close.assert_called_once_with()
        connection.close.assert_called_once_with()

    def test_loopback_get_rejects_every_nonliteral_destination(self):
        for url in (
            "http://localhost:8033/health",
            "http://192.168.0.178:8033/health",
            "https://127.0.0.1:8033/health",
            "http://user@127.0.0.1:8033/health",
            "http://127.0.0.1:80/health",
            "http://127.0.0.1:8033/health#fragment",
        ):
            with self.subTest(url=url), patch.object(
                runtime.http.client, "HTTPConnection"
            ) as connection, self.assertRaises(runtime.QwenRuntimeError):
                runtime._loopback_get(url, timeout=1)
            connection.assert_not_called()


class LoadingHeartbeatTests(unittest.TestCase):
    def test_schema6_teardown_finishes_before_fresh_admission(self):
        from aeon import main

        config = {
            "model": "hermetic-qwen",
            "container_name": "aeon_qwen_test",
            "health_port": 8033,
            "_deploy_env": {"AEON_LOCAL_MODEL_DIR": "hermetic-model"},
        }
        for stopped, finalized, expected in (
            (True, True, main._RetryQwenAdmission),
            (False, True, main._RetryExactQwenClaim),
            (True, False, main._RetryExactQwenClaim),
        ):
            with self.subTest(stopped=stopped, finalized=finalized), patch.object(
                runtime, "current_runtime_state", return_value={"teardown_only": True}
            ), patch.object(
                runtime, "stop_qwen_runtime", return_value=stopped
            ) as stop, patch.object(
                runtime, "finalize_releasing_qwen_runtime", return_value=finalized
            ) as finalize, patch.object(
                runtime, "reuse_qwen_runtime"
            ) as reuse, self.assertRaises(expected):
                main.start_llamacpp_server(config)
            stop.assert_called_once_with(allow_lost_lease=True)
            if stopped:
                finalize.assert_called_once_with(
                    "Aeon retired an exact schema-6 Qwen runtime before readmission"
                )
            else:
                finalize.assert_not_called()
            reuse.assert_not_called()

    def test_saved_runtime_retry_surfaces_fixed_root_identity_failure(self):
        from aeon import main

        config = {
            "model": "hermetic-qwen",
            "container_name": "aeon_qwen_test",
            "health_port": 8033,
            "_deploy_env": {"AEON_LOCAL_MODEL_DIR": "hermetic-model"},
        }
        with patch.object(
            runtime, "current_runtime_state", return_value={"phase": "preflight"}
        ), patch.object(
            runtime,
            "reuse_qwen_runtime",
            side_effect=runtime.QwenRuntimeError("container Config receipt changed"),
        ), patch.object(
            runtime, "reconcile_gone_qwen_runtime", return_value="ambiguous"
        ):
            with self.assertRaises(main._RetryExactQwenClaim) as caught:
                main.start_llamacpp_server(config)
        self.assertIn("QwenRuntimeError: container Config receipt changed", str(caught.exception))

    def test_serialized_retry_prints_and_publishes_root_failure(self):
        from contextlib import nullcontext
        from aeon import main

        retry = main._RetryExactQwenClaim(
            "Root failure was QwenRuntimeError: container Config receipt changed"
        )
        with patch.object(
            main, "_qwen_lifecycle_lock", return_value=nullcontext()
        ), patch.object(
            main, "start_llamacpp_server", side_effect=[retry, True]
        ), patch.object(main.time, "sleep"), patch.object(
            gpu_queue, "_update_compute_presence"
        ) as presence, patch("builtins.print") as output:
            self.assertTrue(main.start_llamacpp_server_serialized({}))
        self.assertTrue(
            any(str(retry) in str(call) for call in output.call_args_list)
        )
        self.assertEqual(presence.call_args.args[0], "unavailable")
        self.assertIn(str(retry), presence.call_args.args[2])

    def test_loading_reuse_creates_one_immediate_exact_heartbeat_and_transfers_it(self):
        from aeon import main

        config = {
            "model": "hermetic-qwen",
            "container_name": "aeon_qwen_test",
            "health_port": 8033,
            "_deploy_env": {"AEON_LOCAL_MODEL_DIR": "hermetic-model"},
        }
        heartbeat = Mock()
        heartbeat.start.return_value = heartbeat
        heartbeat.promote_to_exact_pid.return_value = 4321
        heartbeat_type = Mock(return_value=heartbeat)
        with patch.object(runtime, "current_runtime_state", return_value={"phase": "launching"}), patch.object(
            runtime, "reuse_qwen_runtime", side_effect=[
                runtime.QwenRuntimeLoadingError("loading"),
                runtime.QwenRuntimeLoadingError("loading"),
                4321,
            ]
        ), patch.object(runtime, "local_container_pid", return_value=4321), patch.object(
            gpu_queue, "PeriodicLeaseHeartbeat", heartbeat_type
        ), patch.object(gpu_queue, "heartbeat_vram") as exact_beat:
            for _ in range(2):
                with self.assertRaises(main._RetryExactQwenClaim):
                    main.start_llamacpp_server(config)
            self.assertTrue(main.start_llamacpp_server(config))
        heartbeat_type.assert_called_once()
        kwargs = heartbeat_type.call_args.kwargs
        self.assertTrue(kwargs["require_pid"])
        self.assertLessEqual(kwargs["interval_seconds"], 240)
        heartbeat.start.assert_called_once_with(immediate=True)
        self.assertIs(config["_startup_heartbeat"], heartbeat)
        self.assertEqual(heartbeat.promote_to_exact_pid.call_count, 1)
        exact_beat.assert_called_once()

    def test_latched_loading_heartbeat_is_replaced_after_exact_revalidation(self):
        from aeon import main

        config = {
            "model": "hermetic-qwen",
            "container_name": "aeon_qwen_test",
            "health_port": 8033,
            "_deploy_env": {"AEON_LOCAL_MODEL_DIR": "hermetic-model"},
        }
        beats = []

        def old_beat(*args):
            beats.append(args)
            if len(beats) == 2:
                raise RuntimeError("transient heartbeat transport")

        with patch.object(gpu_queue, "_update_compute_presence"):
            old = gpu_queue.PeriodicLeaseHeartbeat(
                state_file=Path("/tmp/hermetic-qwen-old-heartbeat.json"),
                note="old exact loading heartbeat",
                pid_provider=lambda: 4321,
                interval_seconds=240,
                require_pid=True,
                heartbeat_func=old_beat,
            )
            old.beat_once()
            with self.assertRaises(RuntimeError):
                old.beat_once()
            # A later successful exact beat deliberately does not erase the
            # failure latch or restart the failed periodic worker.
            self.assertEqual(old.promote_to_exact_pid(), 4321)
            with self.assertRaises(RuntimeError):
                old.raise_if_failed()
        config["_startup_heartbeat"] = old
        fresh = Mock()
        fresh.start.return_value = fresh
        heartbeat_type = Mock(return_value=fresh)
        with patch.object(
            runtime, "current_runtime_state", return_value={"phase": "launching"}
        ), patch.object(
            runtime,
            "reuse_qwen_runtime",
            side_effect=[runtime.QwenRuntimeLoadingError("loading"), 4321],
        ), patch.object(
            runtime, "local_container_pid", return_value=4321
        ), patch.object(
            gpu_queue, "PeriodicLeaseHeartbeat", heartbeat_type
        ), patch.object(gpu_queue, "heartbeat_vram"):
            with self.assertRaises(main._RetryExactQwenClaim):
                main.start_llamacpp_server(config)
            self.assertIs(config["_startup_heartbeat"], fresh)
            fresh.start.assert_called_once_with(immediate=True)
            self.assertTrue(main.start_llamacpp_server(config))
        heartbeat_type.assert_called_once()

    def test_replacement_immediate_failure_remains_retryable_same_claim(self):
        from aeon import main

        config = {
            "model": "hermetic-qwen",
            "container_name": "aeon_qwen_test",
            "health_port": 8033,
            "_deploy_env": {"AEON_LOCAL_MODEL_DIR": "hermetic-model"},
        }
        heartbeat_class = gpu_queue.PeriodicLeaseHeartbeat
        created = []

        def heartbeat_factory(**kwargs):
            heartbeat = heartbeat_class(
                **kwargs,
                heartbeat_func=lambda *_args: (_ for _ in ()).throw(
                    RuntimeError("continued heartbeat transport failure")
                ),
            )
            created.append(heartbeat)
            return heartbeat

        with patch.object(
            runtime, "current_runtime_state", return_value={"phase": "launching"}
        ), patch.object(
            runtime,
            "reuse_qwen_runtime",
            side_effect=runtime.QwenRuntimeLoadingError("loading"),
        ) as reuse, patch.object(
            runtime, "local_container_pid", return_value=4321
        ), patch.object(
            gpu_queue, "PeriodicLeaseHeartbeat", side_effect=heartbeat_factory
        ), patch.object(
            gpu_queue, "_update_compute_presence"
        ), patch.object(gpu_queue, "current_lease", return_value=None):
            for _ in range(2):
                with self.assertRaises(main._RetryExactQwenClaim):
                    main.start_llamacpp_server(config)
        self.assertEqual(reuse.call_count, 2)
        self.assertEqual(len(created), 2)
        self.assertIs(config["_startup_heartbeat"], created[-1])
        for heartbeat in created:
            with self.assertRaises(RuntimeError):
                heartbeat.raise_if_failed()

    def test_fresh_create_promotes_exact_pid_before_endpoint_probe(self):
        source = Path(runtime.__file__).read_text(encoding="utf-8")
        launch = source.index("state = {", source.index("def start_local_runtime"))
        promote = source.index("heartbeat_promoter()", launch)
        endpoint = source.index("_wait_for_endpoint(", promote)
        self.assertLess(promote, endpoint)


class StaticIntegrationTests(unittest.TestCase):
    def test_released_worker_and_alternate_launchers_are_fail_closed(self):
        runtime_source = Path(runtime.__file__).read_text(encoding="utf-8")
        main_source = Path(__file__).resolve().parents[1].joinpath("main.py").read_text(encoding="utf-8")
        self.assertEqual(runtime.APPROVED_HOSTS, {runtime.LOCAL_COORD_HOST: runtime.LOCAL_COORD_HOSTNAME})
        registry = qwen_capabilities.load_qwen_runtime_capabilities()
        self.assertEqual(
            [item.host for item in registry.enabled],
            ["192.168.0.177", "192.168.0.178", "192.168.0.180"],
        )
        self.assertTrue(
            all(
                not item.enabled
                for item in registry.capabilities
                if item.host
                not in {"192.168.0.177", "192.168.0.178", "192.168.0.180"}
            )
        )
        self.assertIn("enabled_qwen_runtime_capabilities", main_source)
        self.assertNotIn("stage_remote_image", runtime_source)
        launcher = Path(__file__).resolve().parents[1] / "scripts/launch_vllam_adaptive.sh"
        launcher = launcher.with_name("launch_vllm_adaptive.sh").read_text(encoding="utf-8")
        self.assertIn("disabled", launcher)
        self.assertNotIn("docker run", launcher)
        sweep = Path(__file__).resolve().parents[1].joinpath(
            "scripts/run_qwen38_mtp_sweep.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("direct Qwen benchmark launching is disabled", sweep)
        self.assertNotIn("docker run", sweep)

    def test_main_reserve_uses_enabled_capability_and_full_hash_precedes_reserve(self):
        source = Path(__file__).resolve().parents[1].joinpath("main.py").read_text(encoding="utf-8")
        full_hash = source.index("load_artifact_identity(model_dir, verify_payload=True)")
        reserve = source.index("lease = reserve_named_lease(", full_hash)
        launch = source.index("start_local_runtime(", reserve)
        self.assertLess(full_hash, reserve)
        block = source[reserve:launch]
        self.assertIn("host=capability.host", block)
        self.assertIn("gpu_id=gpu_id", block)
        self.assertIn(
            "min_vram_gb=capability.min_physical_vram_gb", block
        )
        self.assertIn("exclusive=capability.exclusive", block)
        self.assertNotIn("host=None", block)

    def test_signal_and_registry_locks_cover_startup_and_are_durable(self):
        source = Path(__file__).resolve().parents[1].joinpath("main.py").read_text(encoding="utf-8")
        enter = source.index("def enter(self, model_config")
        signal_install = source.index("signal.signal(signal.SIGTERM", enter)
        register = source.index("register_models_for_agent", enter)
        self.assertLess(signal_install, register)
        self.assertIn('QWEN_RUNTIME_ROOT / "lifecycle.lock"', source)
        self.assertIn('QWEN_RUNTIME_ROOT / "model_registry.json"', source)
        exit_start = source.index("def exit(self)")
        unregister = source.index("unregister_models_for_agent", exit_start)
        heartbeat_stop = source.index("heartbeat.stop()", exit_start)
        self.assertLess(unregister, heartbeat_stop)
        self.assertIn("if cleanup_succeeded", source[exit_start:])
        self.assertIn("cancel_pending_reservation(QWEN_LEASE_PATH)", source)
        handler = source[source.index("def _signal_handler", enter):source.index("def _atexit_handler", enter)]
        self.assertIn("raise SystemExit", handler)
        self.assertNotIn("self.exit()", handler)

    def test_false_hard_cap_and_worker_capacity_claims_are_absent(self):
        paths = [
            Path(runtime.__file__),
            Path(__file__).resolve().parents[2] / "README.md",
            Path(__file__).resolve().parents[1] / "main.py",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn("GPU_MEM_LIMIT_GB", combined)
        self.assertNotIn("PyTorch hard cap", combined)
        self.assertIn("measured aggregate peak plan", combined)

    def test_profile_private_ipc_cache_logs_pids_and_receipt_are_documented(self):
        self.assertEqual(
            (
                QWEN38_VLLM_PROFILE.min_host_memory_gb,
                QWEN38_VLLM_PROFILE.min_host_commit_gb,
                QWEN38_VLLM_PROFILE.min_disk_free_gb,
                QWEN38_VLLM_PROFILE.min_shm_free_gb,
            ),
            (96, 96, 32, 16),
        )
        self.assertEqual(runtime.QWEN_CONTAINER_SHM_BYTES, 8 * 1024**3)
        self.assertEqual(runtime.QWEN_RUNTIME_CACHE_BYTES, 8 * 1024**3)
        self.assertNotEqual(QWEN38_VLLM_PROFILE, COMFYUI_PROFILE)


if __name__ == "__main__":
    unittest.main()
