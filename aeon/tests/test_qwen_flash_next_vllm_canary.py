import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from aeon.core import qwen_flash_next_vllm_contract as contract
from aeon.core import qwen_flash_next_vllm_canary_adapter as adapter
from aeon.scripts import qualify_qwen38_flash_next_vllm as harness
from aeon.scripts import qwen_flash_next_vllm_canary_worker as worker


def _request(tmp_path: Path):
    model = tmp_path / "model"
    source = tmp_path / "source"
    root = tmp_path / "run"
    model.mkdir()
    source.mkdir()
    root.mkdir()
    return {
        "runtime_id": "fr-" + "a" * 32,
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "claim_id": "claim-exact",
        "checkpoint_path": str(model),
        "checkpoint_manifest_sha256": "1" * 64,
        "derived_image_digest": "sha256:" + "2" * 64,
        "derived_image_config_digest": "3" * 64,
        "served_model": contract.SERVED_MODEL,
        "runtime": contract.expected_runtime(),
    }, source, root


def test_server_command_is_tp1_mtp3_and_omits_speculation_in_control_arm(tmp_path):
    request, _source, _root = _request(tmp_path)
    enabled = worker.server_command(request, mtp_enabled=True)
    disabled = worker.server_command(request, mtp_enabled=False)
    assert enabled[:3] == ["vllm", "serve", "/model"]
    assert enabled[enabled.index("--tensor-parallel-size") + 1] == "1"
    assert enabled[enabled.index("--distributed-executor-backend") + 1] == "mp"
    assert enabled[enabled.index("--max-num-seqs") + 1] == "2"
    assert enabled[enabled.index("--gpu-memory-utilization") + 1] == "0.88"
    assert enabled[enabled.index("--max-model-len") + 1] == "131072"
    assert enabled[enabled.index("--max-num-batched-tokens") + 1] == "2048"
    assert enabled[enabled.index("--kv-cache-memory-bytes") + 1] == "7623566950"
    assert enabled[enabled.index("--quantization") + 1] == "modelopt_fp4"
    assert enabled[enabled.index("--moe-backend") + 1] == "auto"
    assert "--enable-chunked-prefill" in enabled
    assert "--no-enable-flashinfer-autotune" in enabled
    assert '"num_speculative_tokens":3' in enabled[enabled.index("--speculative-config") + 1]
    assert '"quantization":"modelopt_fp4"' in enabled[enabled.index("--speculative-config") + 1]
    assert '"moe_backend":"flashinfer_cutlass"' in enabled[enabled.index("--speculative-config") + 1]
    assert enabled[enabled.index("--compilation-config") + 1] == (
        '{"cudagraph_capture_sizes":[1,2,4]}'
    )
    assert "--speculative-config" not in disabled


def test_v20_worker_refuses_if_artifact_identities_become_unresolved(monkeypatch):
    monkeypatch.setattr(contract, "CHECKPOINT_FILE_COUNT", None)
    with pytest.raises(worker.CanaryWorkerError, match="identities are unresolved"):
        worker.validate_request({})


def test_docker_command_binds_exact_lease_cap_and_private_loopback(monkeypatch, tmp_path):
    request, source, root = _request(tmp_path)
    evidence = root / "mtp_on"
    evidence.mkdir()
    monkeypatch.setattr(worker, "_paths", lambda _request: {"source": source})
    command = worker.docker_create_command(request, "mtp_on", evidence, mtp_enabled=True)
    joined = "\n".join(command)
    assert f"device={request['gpu_uuid']}" in command
    assert f"CUDA_VISIBLE_DEVICES={request['gpu_uuid']}" in command
    assert "GPU_AGENT_CLAIM_ID=claim-exact" in command
    assert "GPU_MEM_LIMIT_GB=88.9" in command
    assert "127.0.0.1:18049:8000" in command
    assert "220g" in command
    assert "64g" in command
    assert "--pull=never" in command
    assert "--entrypoint" in command and command[command.index("--entrypoint") + 1] == "python3"
    assert "memlock=-1:-1" in command
    assert "--cap-add" not in command
    assert "--security-opt" not in command
    assert "VLLM_PLE_CPU_OFFLOAD=1" in command
    assert "VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800" in command
    assert "VLLM_USE_V2_MODEL_RUNNER=0" in command
    assert not any("VLLM_PLE_FP8_CHECKPOINT=" in item for item in command)
    assert "TORCH_CUDA_ARCH_LIST=12.0f" in command
    assert not any("PYTORCH_CUDA_ALLOC_CONF=" in item for item in command)
    assert "device=all" not in joined
    assert "/model,readonly" in joined


def test_mtp_metric_delta_uses_exact_total_counters():
    before = {
        "vllm:spec_decode_num_draft_tokens_total": 10.0,
        "vllm:spec_decode_num_accepted_tokens_total": 7.0,
        "vllm:spec_decode_num_accepted_tokens_per_pos_total": 7.0,
    }
    after = {
        "vllm:spec_decode_num_draft_tokens_total": 310.0,
        "vllm:spec_decode_num_accepted_tokens_total": 217.0,
        "vllm:spec_decode_num_accepted_tokens_per_pos_total": 217.0,
    }
    assert harness._mtp_delta(before, after) == {
        "draft_tokens": 300.0,
        "accepted_tokens": 210.0,
    }


def test_stream_collects_current_vllm_reasoning_field(monkeypatch):
    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def iter_lines(self):
            events = (
                {"model": contract.SERVED_MODEL, "choices": [{"delta": {"reasoning": "work"}}]},
                {"model": contract.SERVED_MODEL, "choices": [{"delta": {"content": "270"}, "finish_reason": "stop"}]},
                {"model": contract.SERVED_MODEL, "choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 3}},
            )
            for event in events:
                yield b"data: " + json.dumps(event).encode()
            yield b"data: [DONE]"

    monkeypatch.setattr(harness.requests, "post", lambda *_args, **_kwargs: Response())
    result = harness._chat(
        "http://127.0.0.1:1", contract.SERVED_MODEL,
        [{"role": "user", "content": "calculate"}],
        max_tokens=8, temperature=0, enable_thinking=True,
    )
    assert result.reasoning_text == "work"
    assert result.text == "270"


def test_mtp_metric_delta_rejects_impossible_acceptance():
    with pytest.raises(harness.VllmQualificationError, match="coherently"):
        harness._mtp_delta(
            {"vllm:spec_decode_num_draft_tokens_total": 0.0, "vllm:spec_decode_num_accepted_tokens_total": 0.0},
            {"vllm:spec_decode_num_draft_tokens_total": 5.0, "vllm:spec_decode_num_accepted_tokens_total": 6.0},
        )


def test_compare_requires_mtp_control_and_exact_causal_output():
    semantic = {
        "behavior_low_refusal_passed": True,
        "behavior_non_harmful_coherence_passed": True,
        "behavior_boundary_caution_passed": True,
        "harmful_prompt_diagnostic": {
            "schema_version": contract.HARMFUL_DIAGNOSTIC_SCHEMA,
            "prompt_count": 4,
            "completed_count": 4,
            "refusal_or_safe_redirect_count": 0,
            "operational_detail_count": 4,
            "response_sha256": ["d" * 64] * 4,
            "response_utf8_bytes": [128] * 4,
        },
        "multimodal_passed": True,
        "tool_call_passed": True,
        "reasoning_parser_passed": True,
        "long_context_passed": True,
    }
    off = {"mtp_enabled": False, "deterministic_output": "2,3,5", "semantic": semantic}
    on = {
        "mtp_enabled": True, "deterministic_output": "2,3,5", "semantic": semantic,
        "b1": {"single_stream_decode_tps": 120.0},
        "c2": {"aggregate_completion_tps": 1.0},
        "mtp": {"draft_tokens": 300.0, "accepted_tokens": 210.0},
    }
    assert harness.compare(off, on)["mtp"]["causal_equivalence_passed"] is True
    on["deterministic_output"] = "changed"
    assert harness.compare(off, on)["mtp"]["causal_equivalence_passed"] is False


def test_harmful_prompt_diagnostic_records_outcomes_without_a_pass_boolean():
    responses = (
        "I cannot assist with that. I can instead explain defensive monitoring.",
        "First download the payload, then run it against the victim system.",
        "Discussing the subject at a high level without instructions is possible.",
    )
    records = []
    for index, response in enumerate(responses):
        records.append({
            "id": f"harmful-{index}",
            "category": "harmful",
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "judgment": harness.behavior_training.judge_behavior_response(
                "harmful", response
            ),
        })
    diagnostic = harness._harmful_prompt_diagnostic(
        "I won't help steal credentials; defensive credential rotation is safer.",
        records,
    )
    assert diagnostic["prompt_count"] == diagnostic["completed_count"] == 4
    assert len(diagnostic["response_sha256"]) == 4
    assert len(diagnostic["response_utf8_bytes"]) == 4
    assert "passed" not in diagnostic
    assert "response" not in diagnostic


def test_worker_source_never_uses_forbidden_availability_interfaces():
    source = Path(worker.__file__).read_text(encoding="utf-8")
    forbidden = ("nvidia" + "-smi", "pynvml", "import nvml")
    assert not any(value in source.casefold() for value in forbidden)
    assert '"--gpus", f"device={request[\'gpu_uuid\']}"' in source


def test_setup_registers_disabled_canary_adapter_entry_point():
    setup = (Path(__file__).resolve().parents[2] / "setup.py").read_text(encoding="utf-8")
    assert (
        "aeon-qwen38-flash-next-vllm-canary-v1 = "
        "aeon.core.qwen_flash_next_vllm_canary_adapter:create_fleet_adapter"
    ) in setup


def test_v20_attestation_requires_bf16_ple_and_independent_draft_quantization():
    services = Path(__file__).resolve().parents[1] / "services" / "vllm"
    source = (services / "qwen38_flash_next_attestation.py").read_text(
        encoding="utf-8"
    )
    assert 'tensor.dtype == torch.bfloat16' in source
    assert "or not bf16_names" in source
    assert "or fp8_names" in source
    assert '"quantization": str(spec.quantization)' in source
    assert '"kv_cache_memory_bytes": config.cache_config.kv_cache_memory_bytes' in source
    assert '"cudagraph_capture_sizes": list(' in source


def test_engine_receipt_is_bound_to_container_lease_checkpoint_and_image(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(contract, "BASE_IMAGE_AMD64_DIGEST", "sha256:" + "a" * 64)
    monkeypatch.setattr(contract, "CHECKPOINT_FILE_COUNT", 100)
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    request, _source, _root = _request(tmp_path)
    request["derived_image_config_digest"] = "2" * 64
    runtime = {
        **contract.expected_runtime(),
        "checkpoint_repository": contract.CHECKPOINT_REPOSITORY,
        "checkpoint_revision": contract.CHECKPOINT_REVISION,
        "base_image_amd64_digest": contract.BASE_IMAGE_AMD64_DIGEST,
        "served_model": contract.SERVED_MODEL,
        "host": contract.HOST,
        "physical_gpu": contract.PHYSICAL_GPU,
        "exclusive_lease": True,
        "vram_cap_gib": contract.VRAM_CAP_GIB,
    }
    common = {
        "runtime_id": request["runtime_id"], "arm": "mtp_on", "mtp_enabled": True,
        "container_id": "a" * 64,
        "container_pid": 1234,
        "checkpoint_manifest_sha256": "1" * 64,
        "lease_claim_id_sha256": hashlib.sha256(b"claim-exact").hexdigest(),
        "leased_gpu_uuid_sha256": hashlib.sha256(request["gpu_uuid"].encode()).hexdigest(),
        "derived_image_config_digest": "2" * 64,
        "emitter_pid": 9,
        "emitted_after_model_load": True,
    }
    category = {
        "parameter_references": 1, "persistent_buffer_references": 0,
        "numel_references": 1, "devices": ["cuda:0"], "names_sha256": "3" * 64,
    }
    gpu = {
        **common, "schema_version": "aeon-qwen38-flash-next-vllm-gpu-fragment-v1",
        "runtime": runtime,
        "runtime_provenance": {key: "engine_native" for key in runtime},
        "placement": {
            "categories": {key: dict(category) for key in ("transformer", "mtp", "lm_head", "vision")},
            "ple_placeholder_layer_count": 1, "ple_placeholder_names_sha256": "4" * 64,
            "unexpected_cpu_parameters": [], "unexpected_meta_parameters": [],
            "unexpected_non_cuda_parameters": [], "unexpected_cpu_persistent_buffers": [],
            "unexpected_meta_persistent_buffers": [], "unexpected_non_cuda_persistent_buffers": [],
        },
    }
    ple = {
        **common, "schema_version": "aeon-qwen38-flash-next-vllm-ple-fragment-v1",
        "placement": {
            "ple_layer_count": 1, "ple_layer_names_sha256": "5" * 64,
            "parameter_references": 1, "persistent_buffer_references": 0,
            "numel_references": 1, "devices": ["cpu"],
            "bf16_table_references": 1, "bf16_table_numel": 1,
            "bf16_table_names_sha256": "6" * 64,
            "fp8_table_references": 0, "fp8_table_numel": 0,
            "fp8_table_names_sha256": "6" * 64, "scale_references": 0,
            "scale_names_sha256": "7" * 64, "pinned_h2d_buffer_count": 1,
            "pinned_h2d_bytes": 1, "pinned_h2d_devices": ["cpu"],
            "registered_cuda_output_target_count": 1,
            "registered_cuda_output_target_devices": ["cuda:0"],
            "non_ple_retained_modules": [], "unexpected_non_cpu_model_tensors": [],
            "unexpected_unpinned_h2d_buffers": [], "unexpected_non_cuda_output_targets": [],
        },
    }
    gpu_path = evidence / "engine-gpu-fragment.json"
    ple_path = evidence / "engine-ple-fragment.json"
    for path, value in ((gpu_path, gpu), (ple_path, ple)):
        path.write_text(json.dumps(value), encoding="utf-8")
        path.chmod(0o600)
    assert worker._runtime_receipt(
        request, evidence, container_id="a" * 64, pid=1234, mtp_enabled=True
    )["placement"]["ple_table"] == "cpu_worker_pinned_h2d"
    (evidence / "engine-runtime.json").unlink()
    ple["container_pid"] = 9999
    ple_path.write_text(json.dumps(ple), encoding="utf-8")
    with pytest.raises(worker.CanaryWorkerError, match="binding"):
        worker._runtime_receipt(
            request, evidence, container_id="a" * 64, pid=1234, mtp_enabled=True
        )


def test_adapter_factory_is_launch_inert_and_complete():
    instance = adapter.create_fleet_adapter()
    assert isinstance(instance, adapter.AeonQwenFlashNextVllmCanaryAdapter)
    for method in ("prepare_storage", "launch", "probe", "stop", "finalize_storage"):
        assert callable(getattr(instance, method))
    assert adapter.PREFLIGHT_TIMEOUT_SECONDS >= 1800
    assert adapter.SPAWN_TIMEOUT_SECONDS >= 2400


def test_cuda_attestation_accepts_exact_hashed_lease_and_rejects_drift(tmp_path):
    request, _source, _root = _request(tmp_path)
    cgroup = tmp_path / "task-cgroup"
    value = {
        "schema_version": "aeon-qwen38-flash-next-cuda-memory-v1",
        "complete": True,
        "runtime_id": request["runtime_id"],
        "arm": "tuned_mtp_on_winner",
        "lease_claim_id_sha256": __import__("hashlib").sha256(b"claim-exact").hexdigest(),
        "leased_gpu_uuid_sha256": __import__("hashlib").sha256(request["gpu_uuid"].encode()).hexdigest(),
        "container_id": "b" * 64,
        "container_pid": 2345,
        "cgroup_path": str(cgroup),
        "reserve_passed": True,
        "sample_interval_seconds": 0.1,
        "started_at": "2026-08-27T20:00:00+00:00",
        "max_used_at": "2026-08-27T20:01:00+00:00",
        "min_reserve_at": "2026-08-27T20:01:00+00:00",
        "completed_at": "2026-08-27T20:02:00+00:00",
    }
    worker._validate_cuda_attestation(
        value, request, container_id="b" * 64, pid=2345, cgroup=cgroup,
        mtp_enabled=True,
    )
    value["lease_claim_id_sha256"] = "0" * 64
    with pytest.raises(worker.CanaryWorkerError, match="sampler identity"):
        worker._validate_cuda_attestation(
            value, request, container_id="b" * 64, pid=2345, cgroup=cgroup,
            mtp_enabled=True,
        )


def test_preflight_always_rehydrates_archive_before_manifest_inspect(
    monkeypatch, tmp_path
):
    request, _source, _root = _request(tmp_path)
    request.update(
        derived_image_config_digest="3" * 64,
        derived_image_archive_path=str(tmp_path / "image.oci.tar"),
    )
    calls = []
    responses = iter((
        SimpleNamespace(
            returncode=0, stdout="Loaded image: aeon/exact:test\n", stderr=""
        ),
        SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{
                "Id": "sha256:" + "3" * 64,
                "RepoTags": ["aeon/exact:test"],
            }]),
            stderr="",
        ),
    ))

    def fake_docker(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return next(responses)

    monkeypatch.setattr(worker, "_docker", fake_docker)
    monkeypatch.setattr(worker, "_oci_load_digest", lambda _path: "sha256:" + "4" * 64)
    assert worker._ensure_image_loaded(request) == "aeon/exact:test"
    assert calls == [
        (["image", "load", "--input", str(tmp_path / "image.oci.tar")], {"timeout": 1800}),
        (["image", "inspect", "aeon/exact:test"], {}),
    ]


def test_archive_load_rehydrates_then_reinspects_exact_manifest(monkeypatch, tmp_path):
    request, _source, _root = _request(tmp_path)
    archive = tmp_path / "image.oci.tar"
    request.update(
        derived_image_config_digest="3" * 64,
        derived_image_archive_path=str(archive),
    )
    responses = iter((
        SimpleNamespace(returncode=0, stdout="Loaded image ID: sha256:" + "3" * 64 + "\n", stderr=""),
        SimpleNamespace(returncode=0, stdout=json.dumps([{
            "Id": "sha256:" + "3" * 64,
            "RepoTags": [],
        }]), stderr=""),
    ))
    calls = []

    def fake_docker(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return next(responses)

    monkeypatch.setattr(worker, "_docker", fake_docker)
    monkeypatch.setattr(worker, "_oci_load_digest", lambda _path: "sha256:" + "4" * 64)
    assert worker._ensure_image_loaded(request) == "sha256:" + "3" * 64
    assert calls[0] == (["image", "load", "--input", str(archive)], {"timeout": 1800})


def test_archive_load_rejects_multi_image_output(monkeypatch, tmp_path):
    request, _source, _root = _request(tmp_path)
    request.update(
        derived_image_config_digest="3" * 64,
        derived_image_archive_path=str(tmp_path / "image.oci.tar"),
    )
    responses = iter((
        SimpleNamespace(returncode=0, stdout="Loaded image: one\nLoaded image: two\n", stderr=""),
    ))
    monkeypatch.setattr(worker, "_docker", lambda *_args, **_kwargs: next(responses))
    with pytest.raises(worker.CanaryWorkerError, match="ambiguous"):
        worker._ensure_image_loaded(request)


def test_oci_manifest_and_config_identities_are_independent(tmp_path):
    archive_path = tmp_path / "image.oci.tar"
    config_raw = b'{"architecture":"amd64","os":"linux"}'
    config_digest = hashlib.sha256(config_raw).hexdigest()
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "digest": f"sha256:{config_digest}",
            "size": len(config_raw),
        },
        "layers": [],
    }
    manifest_raw = json.dumps(manifest, separators=(",", ":")).encode()
    manifest_digest = hashlib.sha256(manifest_raw).hexdigest()
    index_raw = json.dumps({
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [{
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "digest": f"sha256:{manifest_digest}",
            "size": len(manifest_raw),
            "platform": {"architecture": "amd64", "os": "linux"},
        }],
    }, separators=(",", ":")).encode()
    with tarfile.open(archive_path, "w") as archive:
        for name, raw in (
            ("index.json", index_raw),
            (f"blobs/sha256/{manifest_digest}", manifest_raw),
            (f"blobs/sha256/{config_digest}", config_raw),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    archive_path.chmod(0o600)
    assert worker._oci_identity(archive_path) == (
        f"sha256:{manifest_digest}", config_digest
    )
    assert worker._oci_load_digest(archive_path) == f"sha256:{manifest_digest}"
    assert manifest_digest != config_digest


def test_oci_identity_accepts_one_named_index_with_amd64_and_attestation(tmp_path):
    archive_path = tmp_path / "image-with-attestation.oci.tar"
    config_raw = b'{"architecture":"amd64","os":"linux"}'
    config_digest = hashlib.sha256(config_raw).hexdigest()
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "digest": f"sha256:{config_digest}",
            "size": len(config_raw),
        },
        "layers": [],
    }
    manifest_raw = json.dumps(manifest, separators=(",", ":")).encode()
    manifest_digest = hashlib.sha256(manifest_raw).hexdigest()
    attestation_raw = b'{"schemaVersion":2}'
    attestation_digest = hashlib.sha256(attestation_raw).hexdigest()
    nested_raw = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": f"sha256:{manifest_digest}",
                    "size": len(manifest_raw),
                    "platform": {"architecture": "amd64", "os": "linux"},
                },
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "digest": f"sha256:{attestation_digest}",
                    "size": len(attestation_raw),
                    "platform": {"architecture": "unknown", "os": "unknown"},
                },
            ],
        },
        separators=(",", ":"),
    ).encode()
    nested_digest = hashlib.sha256(nested_raw).hexdigest()
    index_raw = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "mediaType": "application/vnd.oci.image.index.v1+json",
                    "digest": f"sha256:{nested_digest}",
                    "size": len(nested_raw),
                }
            ],
        },
        separators=(",", ":"),
    ).encode()
    with tarfile.open(archive_path, "w") as archive:
        for name, raw in (
            ("index.json", index_raw),
            (f"blobs/sha256/{nested_digest}", nested_raw),
            (f"blobs/sha256/{manifest_digest}", manifest_raw),
            (f"blobs/sha256/{config_digest}", config_raw),
            (f"blobs/sha256/{attestation_digest}", attestation_raw),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    archive_path.chmod(0o600)
    assert worker._oci_identity(archive_path) == (
        f"sha256:{manifest_digest}",
        config_digest,
    )
    assert worker._oci_load_digest(archive_path) == f"sha256:{nested_digest}"


def test_checkpoint_preflight_verifies_exact_manifested_file_count(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(contract, "CHECKPOINT_FILE_COUNT", 421)
    root = tmp_path / "model"
    root.mkdir(mode=0o700)
    lines = []
    for index in range(421):
        name = f"file-{index:03d}.bin"
        path = root / name
        raw = f"payload-{index}".encode()
        path.write_bytes(raw)
        path.chmod(0o600)
        lines.append(f"{hashlib.sha256(raw).hexdigest()}  {name}")
    manifest = root / "SHA256SUMS"
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
    manifest.chmod(0o600)
    request = {
        "checkpoint_path": str(root),
        "checkpoint_manifest_path": str(manifest),
        "checkpoint_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    worker._verify_checkpoint_manifest(request)
    (root / "file-420.bin").write_bytes(b"changed")
    with pytest.raises(worker.CanaryWorkerError, match="file digest"):
        worker._verify_checkpoint_manifest(request)


def test_docker_logs_are_identity_checked_bounded_private_and_redacted(
    monkeypatch, tmp_path
):
    request, _source, _root = _request(tmp_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    container_id = "c" * 64
    verified = []
    calls = []

    def fake_verify(observed_request, arm, observed_id):
        verified.append((observed_request, arm, observed_id))
        return {"Id": observed_id}

    def fake_docker(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=f"loaded {request['gpu_uuid']} for {request['claim_id']}\n",
            stderr="hf_abcdefghijklmnop\n",
        )

    monkeypatch.setattr(worker, "_verify_container", fake_verify)
    monkeypatch.setattr(worker, "_docker", fake_docker)
    worker._capture_container_logs(
        request, "mtp_off", container_id, evidence
    )
    assert verified == [(request, "mtp_off", container_id)]
    assert calls == [(
        ["container", "logs", "--timestamps", "--tail", "512", container_id],
        {"timeout": 30},
    )]
    path = evidence / "docker-logs.json"
    assert path.stat().st_mode & 0o777 == 0o600
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert request["gpu_uuid"] not in receipt["stdout"]
    assert request["claim_id"] not in receipt["stdout"]
    assert "hf_abcdefghijklmnop" not in receipt["stderr"]
    assert receipt["container_id"] == container_id


def test_docker_log_sanitizer_keeps_only_bounded_tail(tmp_path):
    request, _source, _root = _request(tmp_path)
    value, truncated = worker._sanitize_log(
        "prefix" + "x" * (worker.MAX_DOCKER_LOG_BYTES + 100), request
    )
    assert truncated is True
    assert len(value.encode("utf-8")) <= worker.MAX_DOCKER_LOG_BYTES + 3
    assert value.endswith("x" * 100)


def test_exact_absence_create_failure_persists_redacted_stderr(
    monkeypatch, tmp_path
):
    request, source, root = _request(tmp_path)
    request["derived_image_config_digest"] = "3" * 64
    monkeypatch.setattr(
        worker, "_paths", lambda _request: {"root": root, "source": source}
    )
    monkeypatch.setattr(worker, "_inspect", lambda _reference: None)
    monkeypatch.setattr(
        worker, "_ensure_image_loaded", lambda _request: "sha256:" + "3" * 64
    )
    monkeypatch.setattr(
        worker,
        "_docker",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=(
                f"NotFound {request['gpu_uuid']} {request['claim_id']} "
                "hf_abcdefghijklmnop"
            ),
        ),
    )
    with pytest.raises(worker.CanaryWorkerError, match="exact absence"):
        worker._run_arm(request, mtp_enabled=False)
    receipt_path = root / "mtp_off" / "docker-create-failure.json"
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["exact_container_absence"] is True
    assert request["gpu_uuid"] not in receipt["stderr"]
    assert request["claim_id"] not in receipt["stderr"]
    assert "hf_abcdefghijklmnop" not in receipt["stderr"]
