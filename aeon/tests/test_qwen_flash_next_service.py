from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fleet_compute.models import ComputeProfile, Lease

from aeon.core import qwen_flash_next_service_adapter as service
from aeon.scripts import finalize_qwen38_flash_next_promotion as promotion


WORKSPACE = Path(__file__).resolve().parents[3]
PROFILES = WORKSPACE / "fleet_compute/profiles.d"


def _raw_profile(name: str) -> dict:
    return json.loads((PROFILES / name).read_text(encoding="utf-8"))


def _qualified_command(
    *,
    cpu_offload: str = "0",
    fraction: str = "0.88",
    steps: str = "3",
    drafts: str = "4",
    moe_backend: str = service.runtime_contract.PREFERRED_MOE_RUNNER_BACKEND,
    speculative_moe_backend: str | None = None,
) -> list[str]:
    speculative_moe_backend = speculative_moe_backend or moe_backend
    return [
        "/usr/bin/docker",
        "run",
        *[
            item
            for key, value in sorted(service.CONSTANT_RUNTIME_ENV.items())
            for item in ("--env", f"{key}={value}")
        ],
        "--mount",
        service.MATERIALIZED_MODEL_MOUNT,
        service.SGLANG_IMAGE_REFERENCE,
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
        str(service.CONTAINER_PORT),
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
        "--context-length",
        str(service.runtime_contract.SM120_VALIDATED_CONTEXT_LENGTH),
        "--max-total-tokens",
        str(service.runtime_contract.SM120_VALIDATED_CONTEXT_LENGTH),
        "--page-size",
        "64",
        "--speculative-draft-model-quantization",
        service.runtime_contract.MTP_DRAFT_QUANTIZATION,
        "--ple-offload-embedding",
        "--cpu-offload-gb",
        cpu_offload,
        "--mamba-ssm-dtype",
        "float32",
        "--max-running-requests",
        "4",
        "--cuda-graph-config",
        '{"decode":{"backend":"full","max_bs":4,"bs":[1,2,4]},"prefill":{"backend":"disabled"}}',
        "--linear-attn-backend",
        "triton",
        "--linear-attn-decode-backend",
        "triton",
        "--linear-attn-prefill-backend",
        "triton",
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
        speculative_moe_backend,
        "--chunked-prefill-size",
        "4096",
        "--mem-fraction-static",
        fraction,
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        steps,
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        drafts,
    ]


def _qualified_command_sha256(command: list[str]) -> str:
    image_index = command.index(service.SGLANG_IMAGE_REFERENCE)
    return service._canonical_sha(command[image_index + 1 :])


def _qualified_runtime_config(
    *,
    moe_backend: str = service.runtime_contract.PREFERRED_MOE_RUNNER_BACKEND,
    speculative_moe_backend: str | None = None,
) -> dict[str, str]:
    return {
        "moe_runner_backend": moe_backend,
        "speculative_moe_runner_backend": (
            speculative_moe_backend or moe_backend
        ),
    }


def _identity() -> dict[str, str]:
    raw = _raw_profile("aeon-qwen38-flash-next-177.json")
    result = {key: "a" * 64 for key in raw["artifact_identity"]}
    result.update(
        image=service.SGLANG_IMAGE_DIGEST.removeprefix("sha256:"),
        image_archive=service.SGLANG_IMAGE_ARCHIVE_SHA256,
        image_config=service.SGLANG_IMAGE_CONFIG_DIGEST.removeprefix("sha256:"),
        image_local_id=service.SGLANG_IMAGE_ID.removeprefix("sha256:"),
        sglang_source_commit=service.SGLANG_SOURCE_COMMIT_SHA256,
        qualification_assets_manifest=service.QUALIFICATION_ASSET_MANIFEST_SHA256,
    )
    return result


def _publication_contract(tmp_path: Path):
    root = tmp_path / "release"
    root.mkdir(mode=0o700, parents=True)
    sums = root / "SHA256SUMS"
    sums.write_text("a" * 64 + "  README.md\n", encoding="ascii")
    sums.chmod(0o600)
    files = {"README.md": ("a" * 64, 123)}
    release = {
        "root": root,
        "files": files,
        "release_tree_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    publication = {
        "schema_version": service.release_tool.PUBLICATION_RECEIPT_SCHEMA,
        "complete": True,
        "created_at": "2026-08-26T22:00:00+00:00",
        "repo_id": "aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
        "repo_type": "model",
        "visibility": "private",
        "authenticated_username": "aday777",
        "huggingface_hub_version": service.release_tool.HF_HUB_VERSION,
        "hf_xet_version": service.release_tool.HF_XET_VERSION,
        "huggingface_hub_wheel_sha256": (
            service.release_tool.HF_HUB_WHEEL_SHA256
        ),
        "hf_xet_wheel_sha256": service.release_tool.HF_XET_WHEEL_SHA256,
        "release_validator_wheels": {
            "requests": {
                "version": service.release_tool.REQUESTS_VERSION,
                "sha256": service.release_tool.REQUESTS_WHEEL_SHA256,
            },
            "charset_normalizer": {
                "version": service.release_tool.CHARSET_NORMALIZER_VERSION,
                "sha256": service.release_tool.CHARSET_NORMALIZER_WHEEL_SHA256,
            },
            "urllib3": {
                "version": service.release_tool.URLLIB3_VERSION,
                "sha256": service.release_tool.URLLIB3_WHEEL_SHA256,
            },
        },
        "upload_wheel_files_rehashed": True,
        "hf_xet_high_performance": True,
        "upload_bytes": 123 + sums.stat().st_size,
        "verified_private_quota_bytes": (
            service.release_tool.FREE_PRIVATE_STORAGE_BYTES
        ),
        "commit": "d" * 40,
        "remote_files": 2,
        "release_tree_sha256": release["release_tree_sha256"],
        "release_manifest_sha256": release["manifest_sha256"],
        "verification": dict(service.release_tool.PUBLICATION_VERIFICATION),
    }
    assert set(publication) == service.release_tool.PUBLICATION_RECEIPT_FIELDS
    return publication, release


def test_private_publication_receipt_matches_exact_uploader_contract(tmp_path) -> None:
    publication, release = _publication_contract(tmp_path)
    service._validate_publication_receipt(
        publication,
        repo_id="aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
        release=release,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repo_type", "dataset"),
        ("visibility", "public"),
        ("authenticated_username", "someone-else"),
        ("commit", "short"),
    ],
)
def test_private_publication_receipt_rejects_identity_drift(
    tmp_path: Path, field: str, value: str
) -> None:
    publication, release = _publication_contract(tmp_path)
    publication[field] = value
    with pytest.raises(service.FlashNextServiceError, match="not verified"):
        service._validate_publication_receipt(
            publication,
            repo_id="aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
            release=release,
        )


def test_private_publication_receipt_rejects_schema_or_verification_drift(
    tmp_path: Path,
) -> None:
    publication, release = _publication_contract(tmp_path)
    publication["unexpected"] = True
    with pytest.raises(service.FlashNextServiceError, match="fields changed"):
        service._validate_publication_receipt(
            publication,
            repo_id="aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
            release=release,
        )

    publication, release = _publication_contract(tmp_path / "second")
    verification = dict(publication["verification"])
    verification.pop("remote_release_tree_digest_exact")
    publication["verification"] = verification
    with pytest.raises(service.FlashNextServiceError, match="not verified"):
        service._validate_publication_receipt(
            publication,
            repo_id="aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
            release=release,
        )


def test_checked_in_candidate_is_inert_gpu0_only_and_fail_closed() -> None:
    raw = _raw_profile("aeon-qwen38-flash-next-177.json")
    canonical = dict(raw)
    manifest = canonical.pop("manifest_sha256")

    assert raw["enabled"] is False
    assert manifest == hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert raw["adapter"] == "aeon-qwen38-flash-next-service-v1"
    assert raw["service_id"] == service.SERVICE_ID
    assert raw["variant_priority"] == 10
    assert raw["vram_budget_gb"] == 88.0
    assert raw["min_physical_vram_gb"] == 94.0
    assert raw["max_replicas"] == 2
    assert raw["placements"] == [
        {"host": service.HOST, "physical_gpu": 0, "enabled": True}
    ]
    assert raw["serving_pool_id"] == service.SERVING_POOL_ID
    assert raw["lane_max_replicas"] == service.FLASH_LANE_MAX_REPLICAS
    assert raw["artifact_identity"]["image"] == (
        service.SGLANG_IMAGE_DIGEST.removeprefix("sha256:")
    )
    assert raw["artifact_identity"]["checkpoint_tree"] == service.ZERO_SHA256
    assert raw["artifact_identity"]["materialized_checkpoint_tree"] == (
        service.ZERO_SHA256
    )
    assert raw["artifact_identity"]["materialization_receipt"] == (
        service.ZERO_SHA256
    )
    assert raw["artifact_identity"]["ple_materialization_manifest"] == (
        service.ZERO_SHA256
    )
    assert raw["artifact_identity"]["ple_materializer"] == service.ZERO_SHA256
    assert raw["artifact_identity"]["qualification"] == service.ZERO_SHA256
    assert raw["artifact_identity"]["release_tree"] == service.ZERO_SHA256
    with pytest.raises(service.FlashNextServiceError, match="absent"):
        service.load_production_binding(PROFILES / "does-not-exist.json")


@pytest.mark.parametrize(
    "moe_backend",
    service.runtime_contract.QUALIFICATION_MOE_RUNNER_BACKENDS,
)
def test_production_command_is_qualification_derived_mtp_ple_no_cpu_offload(
    moe_backend: str,
) -> None:
    qualified = _qualified_command(moe_backend=moe_backend)
    command = service._production_container_command(
        qualified,
        repo_id="aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
        runtime_config=_qualified_runtime_config(moe_backend=moe_backend),
        expected_command_sha256=_qualified_command_sha256(qualified),
    )

    assert command[command.index("--model-path") + 1] == "/model"
    assert command[command.index("--served-model-name") + 1] == service.SERVED_ALIAS
    assert command[command.index("--speculative-algorithm") + 1] == "NEXTN"
    assert command[command.index("--speculative-num-steps") + 1] == "3"
    assert command[command.index("--speculative-num-draft-tokens") + 1] == "4"
    assert command[command.index("--reasoning-parser") + 1] == "qwen3"
    assert command[command.index("--prefill-attention-backend") + 1] == "triton"
    assert command[command.index("--decode-attention-backend") + 1] == "trtllm_mha"
    assert command[
        command.index("--speculative-draft-model-quantization") + 1
    ] == "unquant"
    assert "--ple-offload-embedding" in command
    assert command[command.index("--cpu-offload-gb") + 1] == "0"
    assert command[command.index("--moe-runner-backend") + 1] == moe_backend
    assert command[
        command.index("--speculative-moe-runner-backend") + 1
    ] == moe_backend

    substituted = _qualified_command(moe_backend=moe_backend)
    substituted[substituted.index(service.SGLANG_IMAGE_REFERENCE)] = (
        service.SGLANG_IMAGE_CONFIG_DIGEST
    )
    with pytest.raises(service.FlashNextServiceError, match="raw OCI config"):
        service._production_container_command(
            substituted,
            repo_id="aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
            runtime_config=_qualified_runtime_config(
                moe_backend=moe_backend
            ),
            expected_command_sha256=_qualified_command_sha256(qualified),
        )


def test_production_command_rejects_moe_backend_drift_from_qualified_winner() -> None:
    repo_id = "aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP"
    unsupported = _qualified_command(moe_backend="flashinfer_trtllm")
    with pytest.raises(service.FlashNextServiceError, match="must set --moe"):
        service._production_container_command(
            unsupported,
            repo_id=repo_id,
            runtime_config=_qualified_runtime_config(),
            expected_command_sha256=_qualified_command_sha256(unsupported),
        )
    with pytest.raises(
        service.FlashNextServiceError,
        match="main/speculative MoE backend pair",
    ):
        command = _qualified_command()
        service._production_container_command(
            command,
            repo_id=repo_id,
            runtime_config=_qualified_runtime_config(
                speculative_moe_backend="flashinfer_trtllm"
            ),
            expected_command_sha256=_qualified_command_sha256(command),
        )
    with pytest.raises(
        service.FlashNextServiceError,
        match="main/speculative MoE backend pair",
    ):
        command = _qualified_command(moe_backend="flashinfer_trtllm")
        service._production_container_command(
            command,
            repo_id=repo_id,
            runtime_config=_qualified_runtime_config(
                moe_backend="flashinfer_trtllm"
            ),
            expected_command_sha256=_qualified_command_sha256(command),
        )
    with pytest.raises(
        service.FlashNextServiceError,
        match="main/speculative MoE backend pair",
    ):
        command = _qualified_command(moe_backend="unreviewed")
        service._production_container_command(
            command,
            repo_id=repo_id,
            runtime_config=_qualified_runtime_config(moe_backend="unreviewed"),
            expected_command_sha256=_qualified_command_sha256(command),
        )


def test_production_command_rejects_allowed_but_unmeasured_argv_drift() -> None:
    measured = _qualified_command(fraction="0.88")
    changed = _qualified_command(fraction="0.86")
    with pytest.raises(service.FlashNextServiceError, match="measured command"):
        service._production_container_command(
            changed,
            repo_id="aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
            runtime_config=_qualified_runtime_config(),
            expected_command_sha256=_qualified_command_sha256(measured),
        )


def test_image_preflight_never_substitutes_raw_config_for_containerd_launch_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_digest = "sha256:" + ("b" * 64)
    config_digest = "sha256:" + ("c" * 64)
    manifest_reference = f"aeon/sglang:test@{manifest_digest}"
    monkeypatch.setattr(service, "SGLANG_IMAGE_ID", manifest_digest)
    monkeypatch.setattr(service, "SGLANG_IMAGE_DIGEST", manifest_digest)
    monkeypatch.setattr(service, "SGLANG_IMAGE_CONFIG_DIGEST", config_digest)
    monkeypatch.setattr(service, "SGLANG_IMAGE_REFERENCE", manifest_reference)

    inspected = {
        "Id": manifest_digest,
        "Descriptor": {"digest": manifest_digest},
        "RepoDigests": [],
        "Config": {"Labels": dict(service.runtime_contract.EXPECTED_IMAGE_LABELS)},
    }
    monkeypatch.setattr(
        service,
        "_docker",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps([inspected]),
            stderr="",
        ),
    )
    service._image_preflight()

    inspected["Id"] = config_digest
    with pytest.raises(service.FlashNextServiceError, match="config/manifest"):
        service._image_preflight()


@pytest.mark.parametrize(
    ("command", "message"),
    [
        (_qualified_command(cpu_offload="1"), "transformer CPU offload"),
        (_qualified_command(fraction="0.91"), "outside the reviewed selector"),
        (
            [item for item in _qualified_command() if item != "--ple-offload-embedding"],
            "PLE host offload",
        ),
    ],
)
def test_production_command_rejects_unqualified_offload_or_memory(command, message) -> None:
    with pytest.raises(service.FlashNextServiceError, match=message):
        service._production_container_command(
            command,
            repo_id="aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
            runtime_config=_qualified_runtime_config(),
            expected_command_sha256=_qualified_command_sha256(command),
        )


def test_docker_create_is_exact_uuid_cgroup_release_and_lease_environment(tmp_path) -> None:
    raw = _raw_profile("aeon-qwen38-flash-next-177.json")
    raw.pop("manifest_sha256", None)
    profile = ComputeProfile.from_dict(raw)
    lease = Lease(
        claim_id="gc-" + "a" * 32,
        owner="fleet-compute",
        host=service.HOST,
        physical_gpu=0,
        gpu_uuid="GPU-11111111-2222-3333-4444-555555555555",
        vram_budget_gb=88.0,
        exclusive=True,
        run_dir=str(tmp_path),
        model="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        memory_total_mib=96 * 1024,
    )
    context = SimpleNamespace(
        runtime_id="fr-" + "a" * 32,
        lease=lease,
        profile=profile,
    )
    publication_receipt = tmp_path / "publication.json"
    publication_receipt.write_text("{}\n", encoding="utf-8")
    binding = service.ProductionBinding(
        binding_path=tmp_path / "binding.json",
        binding_sha256="b" * 64,
        release_dir=service.RELEASE_DIR,
        repo_id="aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
        publication_receipt=publication_receipt,
        release_tree_sha256="c" * 64,
        release_manifest_sha256="d" * 64,
        checkpoint_tree_sha256="e" * 64,
        materialized_model_dir=service.MATERIALIZED_MODEL_DIR,
        materialized_checkpoint_tree_sha256="e" * 64,
        ple_materialization_manifest_sha256="4" * 64,
        ple_materializer_sha256="5" * 64,
        materialization_receipt=service.MATERIALIZATION_RECEIPT,
        materialization_receipt_sha256="6" * 64,
        materialized_model_size_bytes=125_000_000_000,
        materialized_model_inode_count=100,
        release_size_bytes=60_000_000_000,
        release_inode_count=80,
        runtime_config_sha256="f" * 64,
        qualification_sha256="1" * 64,
        qualification_mtp_off_sha256="2" * 64,
        qualification_mtp_on_sha256="3" * 64,
        task_memory_bytes=200 * 1024**3,
        container_command=tuple(
            service._production_container_command(
                (qualified := _qualified_command()),
                repo_id="aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
                runtime_config=_qualified_runtime_config(),
                expected_command_sha256=_qualified_command_sha256(qualified),
            )
        ),
    )

    argv = service._docker_create_argv(context, binding)
    joined = " ".join(argv)

    assert argv[:3] == ("container", "create", "--pull=never")
    assert f"device={lease.gpu_uuid}" in argv
    assert f"GPU_AGENT_CLAIM_ID={lease.claim_id}" in argv
    assert f"CUDA_VISIBLE_DEVICES={lease.gpu_uuid}" in argv
    assert "GPU_MEM_LIMIT_GB=88" in argv
    assert f"{binding.task_memory_bytes}b" in argv
    assert "memlock=-1:-1" in argv
    assert any(
        item
        == (
            f"type=bind,src={service.MATERIALIZED_MODEL_DIR},"
            "dst=/model,readonly"
        )
        for item in argv
    )
    assert not any(
        item == f"type=bind,src={service.RELEASE_DIR},dst=/model,readonly"
        for item in argv
    )
    assert service.SGLANG_IMAGE_REFERENCE in argv
    assert service.SGLANG_IMAGE_CONFIG_DIGEST not in argv
    assert "--cpu-offload-gb 0" in joined
    assert "--ple-offload-embedding" in argv
    assert binding.artifact_identity["materialized_checkpoint_tree"] == "e" * 64
    assert binding.artifact_identity["ple_materialization_manifest"] == "4" * 64
    assert binding.artifact_identity["ple_materializer"] == "5" * 64
    assert binding.artifact_identity["materialization_receipt"] == "6" * 64


@pytest.mark.parametrize(
    "entrypoint",
    (
        lambda: promotion.build_promoted_profiles({}, artifact_identity={}),
        lambda: promotion._validate_registry({}),
        lambda: promotion._replace_transaction({}, binding_payload={}),
        lambda: promotion.prepare_promotion(
            repo_id="retired/release",
            publication_receipt=Path("/tmp/must-not-be-read.json"),
        ),
    ),
)
def test_legacy_sglang_promotion_entrypoints_are_tombstoned(entrypoint) -> None:
    with pytest.raises(promotion.PromotionError, match="one-RTX vLLM"):
        entrypoint()


@pytest.mark.parametrize("execute", (False, True))
def test_legacy_sglang_promotion_command_never_authorizes_profiles(
    execute: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "--repo-id",
        "retired/release",
        "--publication-receipt",
        "/tmp/must-not-be-read.json",
    ]
    if execute:
        arguments.extend(("--execute", "--acknowledge", "old-acknowledgement"))
    assert promotion.main(arguments) == 1
    assert "retired SGLang four-profile promotion" in capsys.readouterr().err


def test_context_rejects_non_blackwell_or_less_than_six_gib_reserve(tmp_path) -> None:
    raw = _raw_profile("aeon-qwen38-flash-next-177.json")
    raw.pop("manifest_sha256", None)
    profile = ComputeProfile.from_dict(raw)
    identity = _identity()
    profile = replace(profile, enabled=True, artifact_identity=identity)
    binding = SimpleNamespace(artifact_identity=identity)
    lease = Lease(
        claim_id="gc-" + "a" * 32,
        owner="fleet-compute",
        host=service.HOST,
        physical_gpu=0,
        gpu_uuid="GPU-11111111-2222-3333-4444-555555555555",
        vram_budget_gb=88.0,
        exclusive=True,
        run_dir=str(tmp_path),
        model="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        memory_total_mib=int(93.5 * 1024),
    )
    context = SimpleNamespace(
        profile=profile,
        lease=lease,
        job_id=None,
        scratch_path=str(tmp_path),
        run_dir=tmp_path,
    )

    wrong_pool_context = SimpleNamespace(
        **{
            **vars(context),
            "profile": replace(profile, serving_pool_id="wrong-pool"),
        }
    )
    with pytest.raises(service.FlashNextServiceError, match="finalized release profile"):
        service.AeonQwenFlashNextServiceAdapter._validate_context(
            wrong_pool_context, binding
        )
    with pytest.raises(service.FlashNextServiceError, match="RTX PRO 6000 GPU0"):
        service.AeonQwenFlashNextServiceAdapter._validate_context(context, binding)
