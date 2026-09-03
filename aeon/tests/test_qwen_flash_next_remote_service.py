from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fleet_compute.models import ArtifactCacheBinding, ArtifactKind, ComputeProfile, Lease

from aeon.core import qwen_flash_next_remote_service_adapter as remote
from aeon.core import qwen_flash_next_service_adapter as service
from aeon.scripts import qwen_flash_next_remote_service_worker as worker


WORKSPACE = Path(__file__).resolve().parents[3]
PROFILE_PATH = (
    WORKSPACE / "fleet_compute/profiles.d/aeon-qwen38-flash-next-179.json"
)


def _raw_profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _fake_oci_archive(path: Path) -> tuple[str, str]:
    layer_payload = b"fake-gzip-layer"
    layer_digest = hashlib.sha256(layer_payload).hexdigest()
    diff_digest = hashlib.sha256(b"fake-uncompressed-layer").hexdigest()
    config_payload = json.dumps(
        {
            "architecture": "amd64",
            "os": "linux",
            "config": {"Labels": dict(service.runtime_contract.EXPECTED_IMAGE_LABELS)},
            "rootfs": {"type": "layers", "diff_ids": [f"sha256:{diff_digest}"]},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    config_digest = hashlib.sha256(config_payload).hexdigest()
    manifest_payload = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": f"sha256:{config_digest}",
                "size": len(config_payload),
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "digest": f"sha256:{layer_digest}",
                    "size": len(layer_payload),
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest_digest = hashlib.sha256(manifest_payload).hexdigest()
    index_payload = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": f"sha256:{manifest_digest}",
                    "size": len(manifest_payload),
                    "annotations": {
                        "io.containerd.image.name": remote.IMAGE_OCI_NAME,
                        "org.opencontainers.image.ref.name": remote.IMAGE_OCI_REF_NAME,
                    },
                    "platform": {"architecture": "amd64", "os": "linux"},
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payloads = {
        "oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
        "index.json": index_payload,
        f"blobs/sha256/{manifest_digest}": manifest_payload,
        f"blobs/sha256/{config_digest}": config_payload,
        f"blobs/sha256/{layer_digest}": layer_payload,
    }
    with tarfile.open(path, "w") as archive:
        for directory in ("blobs", "blobs/sha256"):
            member = tarfile.TarInfo(directory)
            member.type = tarfile.DIRTYPE
            member.mode = 0o755
            archive.addfile(member)
        for name, payload in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(payload))
    path.chmod(0o600)
    return manifest_digest, config_digest


def _winner_command(
    *,
    cpu_offload: str = "0",
    moe_backend: str = worker.REQUIRED_MOE_RUNNER_BACKEND,
) -> list[str]:
    return [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        "/model",
        "--served-model-name",
        service.SERVED_ALIAS,
        "--host",
        "0.0.0.0",
        "--port",
        str(worker.CONTAINER_PORT),
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--quantization",
        service.runtime_contract.QUANTIZATION,
        "--reasoning-parser",
        service.runtime_contract.REASONING_PARSER,
        "--prefill-attention-backend",
        service.runtime_contract.PREFILL_ATTENTION_BACKEND,
        "--decode-attention-backend",
        service.runtime_contract.DECODE_ATTENTION_BACKEND,
        "--speculative-draft-model-quantization",
        service.runtime_contract.MTP_DRAFT_QUANTIZATION,
        "--ple-offload-embedding",
        "--cpu-offload-gb",
        cpu_offload,
        "--mamba-ssm-dtype",
        "bfloat16",
        "--max-running-requests",
        "4",
        "--cuda-graph-config",
        '{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}',
        "--linear-attn-backend",
        "triton",
        "--linear-attn-decode-backend",
        "cutedsl",
        "--linear-attn-prefill-backend",
        "cutedsl",
        "--linear-attn-verify-backend",
        "triton",
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        moe_backend,
        "--fp4-gemm-backend",
        "flashinfer_cutlass",
        "--speculative-moe-a2a-backend",
        "none",
        "--speculative-moe-runner-backend",
        moe_backend,
        "--chunked-prefill-size",
        "8192",
        "--mem-fraction-static",
        "0.84",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "1",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "2",
    ]


def _lease() -> Lease:
    return Lease(
        claim_id="gc-" + "1" * 32,
        owner="fleet-compute",
        host=remote.HOST,
        physical_gpu=1,
        gpu_uuid="GPU-11111111-2222-3333-4444-555555555555",
        vram_budget_gb=88.0,
        exclusive=True,
        run_dir=str(worker.RUN_ROOT / ("fr-" + "a" * 32)),
        model="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        memory_total_mib=96 * 1024,
    )


def test_checked_in_remote_profile_is_inert_exact_and_host_only() -> None:
    raw = _raw_profile()
    canonical = dict(raw)
    manifest = canonical.pop("manifest_sha256")
    cache = raw["artifact_cache"]

    assert raw["enabled"] is False
    assert manifest == hashlib.sha256(
        json.dumps(
            canonical, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert raw["adapter"] == remote.ADAPTER_NAME
    assert raw["service_id"] == service.SERVICE_ID
    assert raw["variant_priority"] == 10
    assert raw["max_replicas"] == 2
    assert raw["serving_pool_id"] == service.SERVING_POOL_ID
    assert raw["lane_max_replicas"] == service.FLASH_LANE_MAX_REPLICAS
    assert raw["personal_priority"] == 30
    assert raw["vram_budget_gb"] == 88.0
    assert raw["min_physical_vram_gb"] - raw["vram_budget_gb"] >= 6
    assert raw["exclusive"] is True
    assert raw["placements"] == [{"host": remote.HOST, "enabled": True}]
    assert "physical_gpu" not in raw["placements"][0]
    assert raw["stage_bytes_max"] == remote.STAGE_BYTES_MAX
    assert raw["min_disk_free_gb"] == remote.MIN_DISK_FREE_GB
    assert cache == remote.promoted_artifact_cache_for_release(service.ZERO_SHA256)
    assert remote.validate_promoted_artifact_cache(
        cache, raw["artifact_identity"]
    ) == raw["stage_bytes_max"]
    assert [item["artifact_id"] for item in cache["artifacts"]] == [
        remote.RELEASE_ARTIFACT_ID,
        remote.MODEL_ARTIFACT_ID,
        remote.IMAGE_ARTIFACT_ID,
    ]
    identity = raw["artifact_identity"]
    assert identity["adapter_source"] == remote.adapter_source_sha256()
    assert identity["remote_staging_contract"] == (
        remote.remote_staging_contract_sha256()
    )
    assert identity["image"] == service.SGLANG_IMAGE_DIGEST.removeprefix("sha256:")
    assert identity["image_config"] == (
        service.SGLANG_IMAGE_CONFIG_DIGEST.removeprefix("sha256:")
    )
    assert identity["image_local_id"] == (
        service.SGLANG_IMAGE_ID.removeprefix("sha256:")
    )
    assert identity["image_archive"] == service.SGLANG_IMAGE_ARCHIVE_SHA256
    for key in (
        "binding",
        "checkpoint_tree",
        "materialized_checkpoint_tree",
        "materialization_receipt",
        "ple_materialization_manifest",
        "ple_materializer",
        "release_tree",
    ):
        assert identity[key] == service.ZERO_SHA256


def test_raw_oci_config_is_never_substituted_for_containerd_launch_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = "b" * 64
    config = "c" * 64
    monkeypatch.setattr(service, "SGLANG_IMAGE_DIGEST", f"sha256:{manifest}")
    monkeypatch.setattr(service, "SGLANG_IMAGE_CONFIG_DIGEST", f"sha256:{config}")
    monkeypatch.setattr(service, "SGLANG_IMAGE_ID", f"sha256:{manifest}")
    identity = dict(_raw_profile()["artifact_identity"])
    identity.update(image=manifest, image_config=config, image_local_id=manifest)

    cache = remote.promoted_artifact_cache_for_release(service.ZERO_SHA256)
    image = cache["artifacts"][-1]
    assert image["identity_key"] == "image_local_id"
    assert image["digest_sha256"] == manifest
    assert remote.validate_promoted_artifact_cache(cache, identity) > 0

    substituted = json.loads(json.dumps(cache))
    substituted_image = substituted["artifacts"][-1]
    substituted_image["digest_sha256"] = config
    with pytest.raises(
        remote.RemoteFlashNextServiceError,
        match="promoted remote cache artifact identity changed",
    ):
        remote.validate_promoted_artifact_cache(substituted, identity)


def test_flash_oci_validator_keeps_manifest_config_and_archive_separate(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "flash.oci.tar"
    manifest, config = _fake_oci_archive(archive)
    descriptor = archive.open("rb")
    try:
        remote._validate_flash_oci_layout_fd(
            descriptor.fileno(),
            manifest_digest=manifest,
            config_digest=config,
            archive_size=archive.stat().st_size,
            expected_labels=service.runtime_contract.EXPECTED_IMAGE_LABELS,
            allowed_link_counts=frozenset({1}),
        )
        with pytest.raises(
            remote.ArtifactCacheSafetyError,
            match="manifest descriptor changed",
        ):
            remote._validate_flash_oci_layout_fd(
                descriptor.fileno(),
                manifest_digest=config,
                config_digest=manifest,
                archive_size=archive.stat().st_size,
                expected_labels=service.runtime_contract.EXPECTED_IMAGE_LABELS,
                allowed_link_counts=frozenset({1}),
            )
    finally:
        descriptor.close()


def test_flash_remote_image_requires_manifest_repo_digest_and_labels() -> None:
    backend = remote.FlashNextArtifactBackend()
    exact = {
        "Id": service.SGLANG_IMAGE_ID,
        "Size": 1234,
        "Descriptor": {"digest": service.SGLANG_IMAGE_DIGEST},
        "RepoDigests": [worker.SGLANG_IMAGE_REPO_DIGEST],
        "Config": {"Labels": dict(service.runtime_contract.EXPECTED_IMAGE_LABELS)},
    }
    with patch.object(
        remote.cache_backend.AeonQwenArtifactBackend,
        "_remote_image_inspection",
        return_value=exact,
    ):
        assert (
            backend._remote_image_inspection(remote.HOST, service.SGLANG_IMAGE_ID)
            == exact
        )

    missing_repo = json.loads(json.dumps(exact))
    missing_repo["RepoDigests"] = []
    with (
        patch.object(
            remote.cache_backend.AeonQwenArtifactBackend,
            "_remote_image_inspection",
            return_value=missing_repo,
        ),
        pytest.raises(
            remote.ArtifactCacheSafetyError,
            match="manifest, repository, or label identity changed",
        ),
    ):
        backend._remote_image_inspection(remote.HOST, service.SGLANG_IMAGE_ID)

    with pytest.raises(
        remote.ArtifactCacheSafetyError,
        match="local image ID was replaced",
    ):
        backend._remote_image_inspection(
            remote.HOST, service.SGLANG_IMAGE_CONFIG_DIGEST
        )


def test_flash_oci_stager_uses_pinned_archive_and_manifest_local_id() -> None:
    backend = remote.FlashNextArtifactBackend()
    descriptor = remote.ArtifactDescriptor.from_dict(
        _raw_profile()["artifact_cache"]["artifacts"][-1]
    )
    exact_image = {"Size": 1234}
    progress: list[tuple[int, int]] = []
    committed: list[dict[str, object]] = []

    def commit(
        _host: str,
        _temporary: Path,
        _filesystem_id: str,
        _descriptor: remote.ArtifactDescriptor,
        receipt: dict[str, object],
    ) -> None:
        committed.append(receipt)

    with (
        patch.object(
            backend,
            "_canonical_flash_archive",
            return_value=(
                remote.CANONICAL_IMAGE_ARCHIVE,
                service.SGLANG_IMAGE_ARCHIVE_SHA256,
            ),
        ),
        patch.object(backend, "_prepare_remote_temporary"),
        patch.object(
            backend,
            "_remote_image_inspection",
            side_effect=[None, exact_image],
        ) as inspect_image,
        patch.object(backend, "_run_with_progress", return_value=("", "")) as run,
        patch.object(
            backend,
            "_remote_sha256",
            return_value=service.SGLANG_IMAGE_ARCHIVE_SHA256,
        ),
        patch.object(backend, "_commit_remote_oci_receipt", side_effect=commit),
    ):
        backend._stage_oci(
            remote.HOST,
            descriptor,
            Path("/home/aday/.local/state/fleet-compute/cache/.staging/f.partial"),
            "42",
            max_bytes_per_second=100_000_000,
            progress=lambda completed, total: progress.append((completed, total)),
        )

    assert inspect_image.call_args_list[0].args[1] == service.SGLANG_IMAGE_ID
    assert inspect_image.call_args_list[1].args[1] == service.SGLANG_IMAGE_ID
    assert len(run.call_args_list) == 2
    load_command = run.call_args_list[1].args[0]
    assert "image load" in load_command[-1]
    assert committed == [
        {
            "schema_version": remote.cache_backend.OCI_RECEIPT_SCHEMA,
            "image_id": service.SGLANG_IMAGE_ID,
            "image_size_bytes": 1234,
            "archive_payload_sha256": service.SGLANG_IMAGE_ARCHIVE_SHA256,
        }
    ]
    assert service.SGLANG_IMAGE_CONFIG_DIGEST not in json.dumps(committed)
    assert progress[-1] == (descriptor.transfer_bytes_max, descriptor.transfer_bytes_max)


def test_standalone_worker_runtime_constants_match_primary_contract() -> None:
    assert worker.SERVED_ALIAS == service.SERVED_ALIAS
    assert worker.DISPLAY_NAME == service.DISPLAY_NAME
    assert worker.ARTIFACT_NAME == service.ARTIFACT_NAME
    assert worker.SGLANG_IMAGE_REFERENCE == service.SGLANG_IMAGE_REFERENCE
    assert worker.SGLANG_IMAGE_DIGEST == service.SGLANG_IMAGE_DIGEST
    assert worker.SGLANG_IMAGE_CONFIG_DIGEST == service.SGLANG_IMAGE_CONFIG_DIGEST
    assert worker.SGLANG_IMAGE_ID == service.SGLANG_IMAGE_ID
    assert worker.SGLANG_IMAGE_ARCHIVE_SHA256 == service.SGLANG_IMAGE_ARCHIVE_SHA256
    assert worker.SGLANG_SOURCE_STACK_SHA256 == (
        service.SGLANG_SOURCE_STACK_SHA256
    )
    assert worker.EXPECTED_IMAGE_LABELS == dict(
        service.runtime_contract.EXPECTED_IMAGE_LABELS
    )
    assert remote.CANONICAL_IMAGE_ARCHIVE == Path(
        "/home/aday/.local/state/aeon-flash-next/runtime-images/"
        "qwen38-flash-next-sm120-headroom-a6c61-424e.oci.tar"
    )
    assert remote.IMAGE_ARCHIVE_SIZE_BYTES == 13_951_062_528
    assert remote.IMAGE_OCI_NAME == (
        "docker.io/aeon/sglang:qwen38-flash-next-sm120-headroom-a6c61-424e"
    )
    assert remote.IMAGE_OCI_REF_NAME == (
        "qwen38-flash-next-sm120-headroom-a6c61-424e"
    )
    assert worker.SGLANG_IMAGE_DIGEST == (
        "sha256:067473b3134f933ebc04a3c4774b16bd400a15afcaf9eec8230c57205f7e7719"
    )
    assert worker.SGLANG_IMAGE_CONFIG_DIGEST == (
        "sha256:ac23f9a937f1e82cc1bade15079a568a73e68b1cecbe4d4f326ba330418e0a36"
    )
    assert worker.SGLANG_IMAGE_ARCHIVE_SHA256 == (
        "f25ab76b3f48b55e1632e020e9fc4709766bae447c42564d2058f16a4bc13374"
    )
    assert worker.SGLANG_SOURCE_STACK_SHA256 == (
        "f9087c7d56219f49fb575c8b1008e923ddeea1ea878e46b20f8e5585317136ed"
    )
    assert worker.REQUIRED_MOE_RUNNER_BACKEND == "flashinfer_cutlass"
    assert worker.QUALIFIED_MOE_RUNNER_BACKENDS == frozenset(
        {"flashinfer_cutlass"}
    )
    assert service.runtime_contract.DECODE_ATTENTION_BACKEND == "trtllm_mha"


def test_remote_worker_requires_unlimited_memlock_for_pinned_ple() -> None:
    runtime_id = "fr-" + "a" * 32
    model = Path("/home/aday/.local/state/fleet-compute/cache/model")
    request = {
        "runtime_id": runtime_id,
        "binding_sha256": "1" * 64,
        "release_tree_sha256": "2" * 64,
        "materialized_checkpoint_tree_sha256": "3" * 64,
        "materialization_receipt_sha256": "4" * 64,
        "contract_sha256": "5" * 64,
        "lease": {
            "claim_id": "gc-" + "6" * 32,
            "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555",
        },
        "materialized_model": {"worker_path": str(model)},
        "container": {
            "name": f"aeon-qwen38-flash-next-179-{runtime_id}",
            "task_memory_bytes": 160 * 1024**3,
            "shm_bytes": 128 * 1024**3,
            "host_port": remote.REMOTE_PORT_BASE + 1,
            "image_reference": service.SGLANG_IMAGE_REFERENCE,
            "command": _winner_command(),
            "command_sha256": "7" * 64,
            "environment": dict(service.CONSTANT_RUNTIME_ENV),
        },
    }

    argv = worker._create_argv(request, model)

    assert argv.count("--ulimit") == 1
    assert argv[argv.index("--ulimit") + 1] == "memlock=-1:-1"

    item = {
        "Id": "8" * 64,
        "Name": "/" + request["container"]["name"],
        "Config": {
            "Image": service.SGLANG_IMAGE_REFERENCE,
            "User": f"{worker.os.geteuid()}:{worker.os.getegid()}",
            "Cmd": request["container"]["command"],
            "Labels": worker._labels(request),
            "Env": [
                f"{key}={value}"
                for key, value in request["container"]["environment"].items()
            ],
        },
        "HostConfig": {
            "Memory": request["container"]["task_memory_bytes"],
            "MemorySwap": request["container"]["task_memory_bytes"],
            "ShmSize": request["container"]["shm_bytes"],
            "PidsLimit": 4096,
            "Ulimits": [{"Name": "memlock", "Hard": -1, "Soft": -1}],
            "SecurityOpt": ["no-new-privileges=true"],
            "PortBindings": {
                f"{worker.CONTAINER_PORT}/tcp": [
                    {
                        "HostIp": "127.0.0.1",
                        "HostPort": str(request["container"]["host_port"]),
                    }
                ]
            },
            "DeviceRequests": [
                {
                    "DeviceIDs": [request["lease"]["gpu_uuid"]],
                    "Capabilities": [["gpu"]],
                }
            ],
        },
        "State": {"Running": False, "Pid": 0},
        "Mounts": [
            {
                "Destination": "/model",
                "Source": str(model),
                "RW": False,
                "Type": "bind",
            }
        ],
    }
    worker._container_identity(item, request, running=False)
    item["HostConfig"]["Ulimits"] = []
    with pytest.raises(worker.RemoteWorkerError, match="configuration"):
        worker._container_identity(item, request, running=False)


def test_remote_contract_is_private_static_offline_and_materialized() -> None:
    contract = remote.remote_staging_contract()
    encoded = json.dumps(contract, sort_keys=True)
    source = Path(remote.__file__).read_text(encoding="utf-8")
    worker_source = Path(worker.__file__).read_text(encoding="utf-8")

    assert contract["runtime_environment"] == service.CONSTANT_RUNTIME_ENV
    assert contract["materialized_model"] == {
        "artifact_id": remote.MODEL_ARTIFACT_ID,
        "canonical_path": str(service.MATERIALIZED_MODEL_DIR),
        "manifest_path": str(service.MATERIALIZED_MODEL_DIR / "SHA256SUMS"),
        "completion_receipt_path": str(service.MATERIALIZATION_RECEIPT),
        "size_bytes_max": remote.MODEL_SIZE_BYTES_MAX,
        "inode_count_max": remote.MODEL_INODE_COUNT_MAX,
        "transfer_bytes_max": remote.MODEL_TRANSFER_BYTES_MAX,
        "cold_peak_bytes_max": remote.MODEL_COLD_PEAK_BYTES_MAX,
        "mount_path": "/model",
        "mount_read_only": True,
        "source_role": "offline-materialized-canonical-checkpoint",
        "rsync_archive_hardlinks": True,
    }
    assert contract["lease"]["hard_physical_gpu"] is False
    assert "GPU-" not in encoded and "gc-" not in encoded
    assert "-aH" in source
    for forbidden in ("nvidia-smi", "pynvml", "gpu_coord.py", "--delete"):
        assert forbidden not in source
        assert forbidden not in worker_source


def test_remote_context_rejects_wrong_shared_pool_before_launch() -> None:
    raw = _raw_profile()
    raw.pop("manifest_sha256", None)
    raw["enabled"] = True
    raw["serving_pool_id"] = "wrong-pool"
    profile = ComputeProfile.from_dict(raw)
    lease = _lease()
    context = SimpleNamespace(profile=profile, lease=lease)
    binding = SimpleNamespace(task_memory_bytes=160 * 1024**3)

    with (
        patch.object(remote, "promoted_artifact_cache", return_value=raw["artifact_cache"]),
        patch.object(
            remote,
            "validate_promoted_artifact_cache",
            return_value=remote.STAGE_BYTES_MAX,
        ),
        patch.object(
            remote,
            "remote_artifact_identity",
            return_value=raw["artifact_identity"],
        ),
        pytest.raises(remote.RemoteFlashNextServiceError, match="fully promoted"),
    ):
        remote.AeonQwenFlashNextRemoteServiceAdapter._validate_context(
            context, binding
        )


def test_private_request_binds_exact_lease_model_and_three_cache_entries() -> None:
    lease = _lease()
    release_sha = "1" * 64
    checkpoint_sha = "2" * 64
    image_sha = service.SGLANG_IMAGE_ID.removeprefix("sha256:")
    context = SimpleNamespace(
        runtime_id="fr-" + "a" * 32,
        scratch_path=Path(lease.run_dir),
        lease=lease,
        cached_artifacts={
            remote.RELEASE_ARTIFACT_ID: ArtifactCacheBinding(
                artifact_id=remote.RELEASE_ARTIFACT_ID,
                kind=ArtifactKind.MANIFESTED_TREE,
                worker_path=str(remote.CACHE_ROOT / "sha256/11" / release_sha),
                digest_sha256=release_sha,
                size_bytes=100,
                inode_count=3,
                filesystem_id="42",
            ),
            remote.MODEL_ARTIFACT_ID: ArtifactCacheBinding(
                artifact_id=remote.MODEL_ARTIFACT_ID,
                kind=ArtifactKind.MANIFESTED_TREE,
                worker_path=str(remote.CACHE_ROOT / "sha256/22" / checkpoint_sha),
                digest_sha256=checkpoint_sha,
                size_bytes=200,
                inode_count=4,
                filesystem_id="42",
            ),
            remote.IMAGE_ARTIFACT_ID: ArtifactCacheBinding(
                artifact_id=remote.IMAGE_ARTIFACT_ID,
                kind=ArtifactKind.OCI_ARCHIVE,
                worker_path=str(remote.CACHE_ROOT / "sha256/12" / image_sha),
                digest_sha256=image_sha,
                size_bytes=100,
                inode_count=1,
                filesystem_id="42",
                payload_sha256="3" * 64,
            ),
        },
    )
    command = _winner_command()
    binding = SimpleNamespace(
        binding_sha256="4" * 64,
        release_manifest_sha256="5" * 64,
        release_tree_sha256=release_sha,
        checkpoint_tree_sha256=checkpoint_sha,
        materialized_checkpoint_tree_sha256=checkpoint_sha,
        ple_materialization_manifest_sha256="6" * 64,
        ple_materializer_sha256="7" * 64,
        materialization_receipt_sha256="8" * 64,
        runtime_config_sha256="9" * 64,
        qualification_sha256="a" * 64,
        qualification_mtp_off_sha256="b" * 64,
        qualification_mtp_on_sha256="c" * 64,
        task_memory_bytes=160 * 1024**3,
        container_command=tuple(command),
        command_sha256=remote._canonical_sha(
            {"command": command, "environment": service.CONSTANT_RUNTIME_ENV}
        ),
    )
    with patch.object(
        remote,
        "_binding_runtime_environment",
        return_value=dict(service.CONSTANT_RUNTIME_ENV),
    ):
        request = remote._request_payload(context, binding)

    assert set(request) >= {"release", "materialized_model", "image", "lease"}
    assert request["materialized_model"]["digest_sha256"] == checkpoint_sha
    assert "payload_sha256" not in request["materialized_model"]
    assert request["image"]["payload_sha256"] == "3" * 64
    assert request["container"]["host_port"] == remote.REMOTE_PORT_BASE + 1
    assert request["container"]["environment"] == {
        **service.CONSTANT_RUNTIME_ENV,
        **lease.required_environment,
    }
    worker._validate_container_request(request)
    assert request["container"]["command"][
        request["container"]["command"].index("--decode-attention-backend") + 1
    ] == "trtllm_mha"

    unsupported = json.loads(json.dumps(request))
    unsupported_command = _winner_command(moe_backend="flashinfer_trtllm")
    unsupported["container"]["command"] = unsupported_command
    unsupported["container"]["command_sha256"] = worker._canonical_sha(
        {
            "command": unsupported_command,
            "environment": service.CONSTANT_RUNTIME_ENV,
        }
    )
    with pytest.raises(worker.RemoteWorkerError, match="main/speculative"):
        worker._validate_container_request(unsupported)

    mismatched = json.loads(json.dumps(request))
    speculative_index = mismatched["container"]["command"].index(
        "--speculative-moe-runner-backend"
    )
    mismatched["container"]["command"][speculative_index + 1] = "flashinfer_trtllm"
    mismatched["container"]["command_sha256"] = worker._canonical_sha(
        {
            "command": mismatched["container"]["command"],
            "environment": service.CONSTANT_RUNTIME_ENV,
        }
    )
    with pytest.raises(worker.RemoteWorkerError, match="main/speculative"):
        worker._validate_container_request(mismatched)

    wrong_image = json.loads(json.dumps(request))
    wrong_image["container"]["image_reference"] = (
        "example.invalid/other@" + service.SGLANG_IMAGE_DIGEST
    )
    with pytest.raises(worker.RemoteWorkerError, match="container identity"):
        worker._validate_container_request(wrong_image)

    changed = json.loads(json.dumps(request))
    changed["container"]["command"] = _winner_command(cpu_offload="1")
    changed["container"]["command_sha256"] = worker._canonical_sha(
        {
            "command": changed["container"]["command"],
            "environment": service.CONSTANT_RUNTIME_ENV,
        }
    )
    with pytest.raises(worker.RemoteWorkerError, match="CPU offload"):
        worker._validate_container_request(changed)


def test_remote_artifact_identity_preserves_all_qualified_shared_digests() -> None:
    local = {
        key: hashlib.sha256(key.encode("utf-8")).hexdigest()
        for key in _raw_profile()["artifact_identity"]
        if key != "remote_staging_contract"
    }
    binding = SimpleNamespace(artifact_identity=local)
    identity = remote.remote_artifact_identity(binding)

    assert identity["adapter_source"] == remote.adapter_source_sha256()
    assert identity["remote_staging_contract"] == (
        remote.remote_staging_contract_sha256()
    )
    for key, digest in local.items():
        if key != "adapter_source":
            assert identity[key] == digest


def test_pidless_cleanup_recovery_requires_exact_absence_and_is_durable(
    tmp_path: Path,
) -> None:
    runtime_id = "fr-" + "d" * 32
    run_dir = tmp_path / runtime_id
    run_dir.mkdir(mode=0o700)
    runtime = {
        "runtime_id": runtime_id,
        "run_dir": str(run_dir),
        "physical_gpu": 0,
    }
    request = {
        "container": {"name": f"aeon-qwen38-flash-next-179-{runtime_id}"}
    }
    binding = SimpleNamespace(binding_sha256="e" * 64)
    with (
        patch.object(
            remote,
            "_pidless_request",
            return_value=(request, "f" * 64, binding),
        ),
        patch.object(remote, "_prelaunch_tunnel_absent", return_value=True),
        patch.object(remote, "_remote_run_absent", return_value=True),
        patch.object(remote, "_remote_named_container_absent", return_value=True),
    ):
        assert remote._recover_pidless_runtime(runtime)
        assert remote._recover_pidless_runtime(runtime)

    marker = remote._private_json(run_dir / remote.PRELAUNCH_CLEANUP_RECEIPT)
    assert marker == remote._cleanup_marker(
        runtime_id, "f" * 64, "e" * 64, state="complete"
    )
