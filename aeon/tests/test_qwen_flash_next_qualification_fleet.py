from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from aeon.core import qwen_flash_next_qualification_adapter as adapter
from aeon.scripts import qwen_flash_next_qualification_worker as worker


WORKSPACE = Path(__file__).resolve().parents[3]
PROFILE_PATH = (
    WORKSPACE / "fleet_compute/profiles.d/aeon-qwen38-flash-next-qualification.json"
)


def _private(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True)
    path.chmod(0o700)
    return path


def test_checked_in_qualification_profile_is_enabled_host_only_and_fail_closed() -> (
    None
):
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    canonical = dict(profile)
    manifest = canonical.pop("manifest_sha256")

    assert profile["enabled"] is True
    assert profile["version"] == 12
    assert (
        manifest
        == hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert profile["mode"] == "batch"
    assert profile["adapter"] == "aeon-qwen38-flash-next-qualification-v1"
    assert profile["personal_priority"] == 0
    assert profile["vram_budget_gb"] == 88.0
    assert profile["min_physical_vram_gb"] == 94.0
    assert profile["max_replicas"] == 1
    assert profile["stage_bytes_max"] == 1_000_000_000
    assert profile["runtime_growth_bytes_max"] == 10_000_000_000
    assert profile["worker_free_reserve_bytes"] == 20_000_000_000
    assert profile["min_disk_free_gb"] == 31
    assert profile["startup_timeout_seconds"] == 2400
    assert "qualification run is retained" in profile["hard_cap_mechanism"]
    assert "physical CUDA sampler with >=90% cadence density" in profile[
        "hard_cap_mechanism"
    ]
    assert "<=88 GiB usage" in profile["hard_cap_mechanism"]
    assert "unlimited container memlock" in profile["hard_cap_mechanism"]
    assert profile["placements"] == [
        {"host": worker.HOST, "physical_gpu": 0, "enabled": True}
    ]
    assert profile["artifact_identity"] == adapter.expected_artifact_identity()


def test_source_receipts_allow_only_reviewed_empty_package_markers(
    tmp_path: Path,
) -> None:
    receipts = adapter._source_receipts()
    empty_sha256 = hashlib.sha256(b"").hexdigest()

    assert worker.EMPTY_SOURCE_FILES <= set(adapter.SOURCE_FILES)
    assert {name: receipts[name] for name in worker.EMPTY_SOURCE_FILES} == {
        name: {"sha256": empty_sha256, "size": 0} for name in worker.EMPTY_SOURCE_FILES
    }

    unreviewed = tmp_path / "empty.py"
    unreviewed.write_bytes(b"")
    unreviewed.chmod(0o600)
    with pytest.raises(
        adapter.FlashNextQualificationError,
        match="qualification source is unsafe",
    ):
        adapter._receipt(unreviewed)


def test_source_files_are_a_hermetic_worker_import_closure(tmp_path: Path) -> None:
    staged = tmp_path / "source"
    for name in adapter.SOURCE_FILES:
        target = staged / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes((adapter.PACKAGE_ROOT / name).read_bytes())
        target.chmod(0o600)

    environment = {
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(staged),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(staged / "aeon/scripts/qwen_flash_next_qualification_worker.py"),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_prompt_fitter_accepts_bounded_search_overshoot_and_reproves_exact_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_length = 12
    target_tokens = 10
    observed_token_counts: list[int] = []

    monkeypatch.setattr(
        worker.runtime_contract,
        "SM120_VALIDATED_CONTEXT_LENGTH",
        context_length,
    )
    monkeypatch.setattr(
        worker,
        "_prompt_text",
        lambda _workload_id, _request_index, *, record_count, padding_count, padding_unit: (
            f"{record_count}:{padding_count}:{padding_unit}"
        ),
    )

    def endpoint(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == "/v1/detokenize":
            return {"text": "deterministic rendered prompt"}
        assert path == "/v1/tokenize"
        text = payload["messages"][0]["content"]
        record_count, padding_count, _unit = text.split(":", maxsplit=2)
        count = 2 + int(record_count) * 6 + int(padding_count)
        observed_token_counts.append(count)
        return {
            "tokens": list(range(count)),
            "count": count,
            "max_model_len": context_length,
        }

    monkeypatch.setattr(worker, "_endpoint_json", endpoint)

    material = worker._fit_prompt("prefill_65152_256", 0, target_tokens)

    # Exponential search legitimately reaches 14 before binary/exact fitting.
    assert context_length + 2 in observed_token_counts
    assert observed_token_counts[-1] == target_tokens
    assert len(material.tokens) == target_tokens
    assert material.max_model_len == context_length


@pytest.mark.parametrize(
    "response",
    (
        {"tokens": list(range(33)), "count": 33, "max_model_len": 8},
        {"tokens": [1], "count": 1.0, "max_model_len": 8},
        {"tokens": [True], "count": 1, "max_model_len": 8},
    ),
)
def test_live_tokenizer_rejects_unbounded_or_malformed_search_response(
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, Any],
) -> None:
    monkeypatch.setattr(worker.runtime_contract, "SM120_VALIDATED_CONTEXT_LENGTH", 8)
    monkeypatch.setattr(worker, "_endpoint_json", lambda *_args, **_kwargs: response)

    with pytest.raises(
        worker.QualificationWorkerError,
        match="live chat tokenizer response is invalid",
    ):
        worker._tokenize_messages(
            [{"role": "user", "content": "bounded"}],
            allow_oversized_search=True,
        )


def test_live_tokenizer_search_overshoot_remains_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {"tokens": list(range(9)), "count": 9, "max_model_len": 8}
    monkeypatch.setattr(worker.runtime_contract, "SM120_VALIDATED_CONTEXT_LENGTH", 8)
    monkeypatch.setattr(worker, "_endpoint_json", lambda *_args, **_kwargs: response)
    messages = [{"role": "user", "content": "bounded"}]

    assert worker._tokenize_messages(
        messages, allow_oversized_search=True
    ) == (list(range(9)), 8)
    with pytest.raises(
        worker.QualificationWorkerError,
        match="live chat tokenizer response is invalid",
    ):
        worker._tokenize_messages(messages)


@pytest.mark.parametrize("target_tokens", (0, -1, 13, True, 1.0))
def test_prompt_fitter_rejects_target_outside_validated_context_before_tokenizing(
    monkeypatch: pytest.MonkeyPatch,
    target_tokens: Any,
) -> None:
    monkeypatch.setattr(worker.runtime_contract, "SM120_VALIDATED_CONTEXT_LENGTH", 12)
    monkeypatch.setattr(
        worker,
        "_tokenize_messages",
        lambda *_args, **_kwargs: pytest.fail("invalid target reached tokenizer"),
    )

    with pytest.raises(
        worker.QualificationWorkerError,
        match="prompt target is outside the validated context length",
    ):
        worker._fit_prompt("prefill_65152_256", 0, target_tokens)


def test_worker_accepts_only_exact_canonical_runtime_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_id = "fr-" + "a" * 32
    canonical_root = _private(tmp_path / "canonical")
    run = _private(canonical_root / runtime_id)
    source = _private(run / "source")
    assets = _private(run / "assets")
    source_file = source / "worker.py"
    package = _private(source / "aeon/core")
    package_marker = package / "__init__.py"
    asset_file = assets / "image.jpg"
    source_file.write_bytes(b"worker")
    source_file.chmod(0o600)
    package_marker.write_bytes(b"")
    package_marker.chmod(0o600)
    asset_file.write_bytes(b"image")
    asset_file.chmod(0o600)
    build_root = _private(tmp_path / "build")
    build = _private(build_root / ("fr-" + "b" * 32))

    def receipt(path: Path) -> dict[str, int | str]:
        return {"sha256": worker._sha256(path), "size": path.stat().st_size}

    request = {
        "schema_version": worker.SCHEMA,
        "runtime_id": runtime_id,
        "job_id": "fj-" + "c" * 32,
        "host": worker.HOST,
        "hostname": worker.HOSTNAME,
        "physical_gpu": 0,
        "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
        "claim_id": "gc-test-claim",
        "owner": "test-owner",
        "vram_budget_gb": worker.VRAM_BUDGET_GB,
        "exclusive": True,
        "scratch_path": str(run),
        "checkpoint_path": str(build / worker.TUNED_CHECKPOINT_NAME),
        "official_untuned_checkpoint_path": str(build / worker.UNTUNED_CHECKPOINT_NAME),
        "build_sibling_manifest_path": str(build / worker.SIBLING_MANIFEST_NAME),
        "checkpoint_tree_sha256": "1" * 64,
        "official_untuned_checkpoint_tree_sha256": "2" * 64,
        "build_sibling_manifest_sha256": "3" * 64,
        "builder_sha256": "4" * 64,
        "repo_id": "aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
        "source_files": {
            "worker.py": receipt(source_file),
            "aeon/core/__init__.py": receipt(package_marker),
        },
        "asset_files": {"image.jpg": receipt(asset_file)},
        "sglang_commit": worker.SGLANG_COMMIT,
        "sglang_image_digest": worker.IMAGE_DIGEST,
        "sglang_image_config_digest": worker.IMAGE_CONFIG_DIGEST,
        "sglang_image_id": worker.IMAGE_ID,
        "sglang_image_archive_sha256": worker.IMAGE_ARCHIVE_SHA256,
        "task_memory_gb": worker.TASK_MEMORY_GB,
        "max_accounted_vram_gb": worker.VRAM_BUDGET_GB,
        "preferred_moe_runner_backend": (
            worker.runtime_contract.PREFERRED_MOE_RUNNER_BACKEND
        ),
        "qualification_moe_runner_backends": list(
            worker.runtime_contract.QUALIFICATION_MOE_RUNNER_BACKENDS
        ),
        "cutlass_nvfp4_scale_duplication_bytes": (
            worker.runtime_contract.CUTLASS_NVFP4_SCALE_DUPLICATION_BYTES
        ),
        "cutlass_min_cuda_reserve_bytes": (
            worker.runtime_contract.CUTLASS_MIN_CUDA_RESERVE_BYTES
        ),
        "cutlass_min_geometric_mean_speedup": (
            worker.runtime_contract.CUTLASS_MIN_GEOMETRIC_MEAN_SPEEDUP
        ),
    }
    request_path = run / "qualification-request.json"
    monkeypatch.setattr(worker, "CANONICAL_OUTPUT_ROOT", PurePosixPath(canonical_root))
    monkeypatch.setattr(worker, "CHECKPOINT_ROOT", PurePosixPath(build_root))
    monkeypatch.setattr(
        worker.os, "uname", lambda: SimpleNamespace(nodename=worker.HOSTNAME)
    )

    raw = (json.dumps(request, indent=2, sort_keys=True) + "\n").encode()
    request_path.write_bytes(raw)
    request_path.chmod(0o600)
    assert (
        worker._validate_request(request_path, hashlib.sha256(raw).hexdigest())
        == request
    )

    unreviewed = source / "empty.py"
    unreviewed.write_bytes(b"")
    unreviewed.chmod(0o600)
    request["source_files"]["empty.py"] = receipt(unreviewed)
    changed = (json.dumps(request, indent=2, sort_keys=True) + "\n").encode()
    request_path.write_bytes(changed)
    request_path.chmod(0o600)
    with pytest.raises(
        worker.QualificationWorkerError,
        match="unsafe private file",
    ):
        worker._validate_request(request_path, hashlib.sha256(changed).hexdigest())
    del request["source_files"]["empty.py"]

    request["scratch_path"] = str(canonical_root / ("fr-" + "d" * 32))
    changed = (json.dumps(request, indent=2, sort_keys=True) + "\n").encode()
    request_path.write_bytes(changed)
    request_path.chmod(0o600)
    with pytest.raises(
        worker.QualificationWorkerError,
        match="qualification request contract changed",
    ):
        worker._validate_request(request_path, hashlib.sha256(changed).hexdigest())


def test_adapter_stages_directly_in_canonical_output_and_returns_no_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_id = "fr-" + "e" * 32
    run_root = PurePosixPath(tmp_path / "lease-runs")
    canonical_root = _private(tmp_path / "canonical")
    canonical = canonical_root / runtime_id
    control_run = Path(run_root / runtime_id)
    tree = "1" * 64
    untuned_tree = "2" * 64
    sibling_sha = "3" * 64
    builder_sha = "4" * 64
    payload = {
        "checkpoint": tmp_path / "build/model",
        "untuned_checkpoint": tmp_path / "build/official-untuned-model",
        "sibling_manifest": tmp_path / "build/BUILD_SIBLING_MANIFEST.json",
        "parent": tmp_path / "build",
        "checkpoint_tree_sha256": tree,
        "official_untuned_checkpoint_tree_sha256": untuned_tree,
        "build_sibling_manifest_sha256": sibling_sha,
        "builder_sha256": builder_sha,
        "repo_id": "aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
    }
    profile = SimpleNamespace(
        profile_id=adapter.PROFILE_ID,
        project=adapter.PROJECT,
        enabled=True,
        artifact_identity=adapter.expected_artifact_identity(),
    )
    lease = SimpleNamespace(
        host=adapter.HOST,
        physical_gpu=0,
        memory_total_mib=96 * 1024,
        model="NVIDIA RTX PRO 6000 Blackwell",
        vram_budget_gb=worker.VRAM_BUDGET_GB,
        exclusive=True,
        run_dir=str(control_run),
        gpu_uuid="GPU-01234567-89ab-cdef-0123-456789abcdef",
        claim_id="gc-test-claim",
        owner="test-owner",
    )
    heartbeats: list[tuple[int | None, str]] = []
    context = SimpleNamespace(
        runtime_id=runtime_id,
        job_id="fj-" + "f" * 32,
        payload={},
        profile=profile,
        lease=lease,
        run_dir=control_run,
        scratch_path=None,
        canonical_output_path=canonical,
        heartbeat=lambda pid, note: heartbeats.append((pid, note)),
    )
    instance = adapter.AeonQwenFlashNextQualificationAdapter()
    monkeypatch.setattr(adapter, "RUN_ROOT", run_root)
    monkeypatch.setattr(adapter, "CANONICAL_OUTPUT_ROOT", canonical_root)
    monkeypatch.setattr(instance, "_payload", lambda _raw: payload)
    monkeypatch.setattr(
        adapter.release_tool,
        "validate_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(checkpoint_tree_sha256=tree),
    )
    monkeypatch.setattr(
        worker, "validate_sibling_artifact", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        adapter,
        "_source_receipts",
        lambda: {
            name: {"sha256": "5" * 64, "size": 1} for name in adapter.SOURCE_FILES
        },
    )
    profile.artifact_identity = adapter.expected_artifact_identity()
    monkeypatch.setattr(
        adapter,
        "_asset_receipts",
        lambda: {
            **{name: {"sha256": "6" * 64, "size": 1} for name in adapter.ASSET_FILES},
            "manifest.json": {
                "sha256": "dd8a1138007e0f17ba2ad50f045fd327a0b7bb1714c45d1e1d648434d835547f",
                "size": 1,
            },
        },
    )
    metric_calls: list[tuple[str, bool]] = []

    def metrics(path: str, *, create: bool) -> tuple[str, int, int, int]:
        metric_calls.append((path, create))
        return ("device", 40_000_000_000, 1_000_000, 0 if create else 100)

    staged: list[str] = []
    written: list[tuple[Path, bytes]] = []
    worker_calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(adapter, "_metrics", metrics)
    monkeypatch.setattr(adapter, "_local_dirs", lambda path: staged.append(path))
    monkeypatch.setattr(
        adapter,
        "_stage_local",
        lambda _source, destination, **_kwargs: staged.append(destination),
    )
    monkeypatch.setattr(
        adapter, "_write_private", lambda path, raw: written.append((path, raw))
    )

    def worker_action(
        source: str,
        action: str,
        request: str,
        _digest: str,
        _extra: str | None = None,
        *,
        timeout: float = 120,
    ) -> dict[str, Any]:
        assert timeout == 14_400
        worker_calls.append((source, action, request))
        return {
            "checkpoint_tree_sha256": tree,
            "official_untuned_checkpoint_tree_sha256": untuned_tree,
            "build_sibling_manifest_sha256": sibling_sha,
            "sglang_commit": worker.SGLANG_COMMIT,
            "sglang_image_digest": worker.IMAGE_DIGEST,
            "sglang_image_config_digest": worker.IMAGE_CONFIG_DIGEST,
            "sglang_image_id": worker.IMAGE_ID,
            "sglang_image_archive_sha256": worker.IMAGE_ARCHIVE_SHA256,
            "max_accounted_vram_gb": worker.VRAM_BUDGET_GB,
            "max_cgroup_memory_gb": worker.TASK_MEMORY_GB,
            "preferred_moe_runner_backend": (
                worker.runtime_contract.PREFERRED_MOE_RUNNER_BACKEND
            ),
            "qualification_moe_runner_backends": list(
                worker.runtime_contract.QUALIFICATION_MOE_RUNNER_BACKENDS
            ),
            "cutlass_nvfp4_scale_duplication_bytes": (
                worker.runtime_contract.CUTLASS_NVFP4_SCALE_DUPLICATION_BYTES
            ),
            "cutlass_min_cuda_reserve_bytes": (
                worker.runtime_contract.CUTLASS_MIN_CUDA_RESERVE_BYTES
            ),
            "cutlass_min_geometric_mean_speedup": (
                worker.runtime_contract.CUTLASS_MIN_GEOMETRIC_MEAN_SPEEDUP
            ),
        }

    monkeypatch.setattr(adapter, "_worker_action", worker_action)

    prepared = instance.prepare_storage(context)

    assert prepared.scratch_path is None
    assert metric_calls == [(str(canonical), True), (str(canonical), False)]
    assert written[0][0] == canonical / "qualification-request.json"
    request = json.loads(written[0][1])
    assert request["scratch_path"] == str(canonical)
    assert str(control_run) not in written[0][1].decode()
    assert worker_calls == [
        (
            f"{canonical}/source",
            "preflight",
            f"{canonical}/qualification-request.json",
        )
    ]
    assert all(str(control_run) not in item for item in staged)
    assert heartbeats

    bad_context = SimpleNamespace(**{**vars(context), "scratch_path": str(control_run)})
    with pytest.raises(
        adapter.FlashNextQualificationError,
        match="lease is not exact canonical",
    ):
        instance.prepare_storage(bad_context)


def test_sm120_commands_are_closed_and_mtp_geometry_is_native() -> None:
    baseline = worker.RuntimeTuning.safe_baseline()
    off = worker._server_command(
        worker.ARM_TUNED_MTP_OFF, model_path="/model", tuning=baseline
    )
    on_tuning = worker.replace(baseline, nextn=(2, 3))
    on = worker._server_command(
        worker.ARM_TUNED_MTP_ON, model_path="/model", tuning=on_tuning
    )

    assert off[off.index("--mamba-ssm-dtype") + 1] == "bfloat16"
    assert off[off.index("--linear-attn-backend") + 1] == "triton"
    assert off[off.index("--linear-attn-verify-backend") + 1] == "triton"
    assert off[off.index("--moe-runner-backend") + 1] == "flashinfer_cutlass"
    assert off[off.index("--speculative-moe-runner-backend") + 1] == (
        "flashinfer_cutlass"
    )
    assert off[off.index("--fp4-gemm-backend") + 1] == "flashinfer_cutlass"
    assert off[off.index("--reasoning-parser") + 1] == "qwen3"
    assert off[off.index("--prefill-attention-backend") + 1] == "triton"
    assert off[off.index("--decode-attention-backend") + 1] == "trtllm_mha"
    assert off[off.index("--speculative-draft-model-quantization") + 1] == "unquant"
    assert off[off.index("--max-running-requests") + 1] == "4"
    assert off[off.index("--max-total-tokens") + 1] == "65536"
    assert off[off.index("--page-size") + 1] == "64"
    assert off[off.index("--max-mamba-cache-size") + 1] == "20"
    assert off[off.index("--mem-fraction-static") + 1] == "0.92"
    assert "--speculative-algorithm" not in off
    assert on[on.index("--speculative-algorithm") + 1] == "NEXTN"
    assert on[on.index("--speculative-num-steps") + 1] == "2"
    assert on[on.index("--speculative-num-draft-tokens") + 1] == "3"
    assert on[on.index("--speculative-eagle-topk") + 1] == "1"
    assert (
        worker._runtime_config(worker.ARM_TUNED_MTP_ON, on_tuning)[
            "runtime_environment"
        ]
        == worker.CONSTANT_RUNTIME_ENV
    )

    sm120_fast = worker.replace(
        baseline,
        linear_decode_backend="flashinfer",
        linear_prefill_backend="triton",
        mamba_ssm_dtype="bfloat16",
    )
    fast_command = worker._server_command(
        worker.ARM_TUNED_MTP_OFF,
        model_path="/model",
        tuning=sm120_fast,
    )
    assert (
        fast_command[fast_command.index("--linear-attn-decode-backend") + 1]
        == "flashinfer"
    )
    assert (
        fast_command[fast_command.index("--linear-attn-verify-backend") + 1]
        == "flashinfer"
    )

    with pytest.raises(worker.QualificationWorkerError):
        worker.RuntimeTuning(
            **{
                **baseline.__dict__,
                "nextn": (3, 3),
            }
        )
    with pytest.raises(worker.QualificationWorkerError):
        worker.replace(
            baseline,
            linear_decode_backend="flashinfer",
            mamba_ssm_dtype="float32",
        )
    with pytest.raises(worker.QualificationWorkerError):
        worker._server_command(
            worker.ARM_TUNED_MTP_ON, model_path="/model", tuning=baseline
        )


def test_unsettled_sm120_image_fails_before_docker_inspection(monkeypatch) -> None:
    monkeypatch.setattr(
        worker, "IMAGE_CONFIG_DIGEST", worker.runtime_contract.UNSET_DIGEST
    )
    monkeypatch.setattr(
        worker,
        "_docker",
        lambda *_args, **_kwargs: pytest.fail("Docker must not be inspected"),
    )
    with pytest.raises(
        worker.QualificationWorkerError,
        match="SM120 SGLang image manifest/config identities are not settled",
    ):
        worker._image_preflight()


def test_local_image_preflight_loads_only_exact_absent_pinned_oci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "runtime.oci.tar"
    archive.write_bytes(b"pinned-oci")
    archive.chmod(0o600)
    monkeypatch.setattr(worker, "IMAGE_ARCHIVE_PATH", archive)
    monkeypatch.setattr(worker, "IMAGE_ARCHIVE_SIZE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(
        worker,
        "_sha256",
        lambda path: (
            worker.IMAGE_ARCHIVE_SHA256
            if path == archive
            else hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    )
    exact_image = SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            [
                {
                    "Id": worker.IMAGE_ID,
                    "Descriptor": {"digest": worker.IMAGE_DIGEST},
                    "RepoDigests": [
                        worker.runtime_contract.QUALIFIED_IMAGE_REPO_DIGEST
                    ],
                    "Config": {
                        "Labels": dict(worker.runtime_contract.EXPECTED_IMAGE_LABELS)
                    },
                }
            ]
        ),
        stderr="",
    )
    calls: list[list[str]] = []
    responses = iter(
        [
            SimpleNamespace(
                returncode=1,
                stdout="",
                stderr=f"Error response from daemon: No such image: {worker.IMAGE_REFERENCE}\n",
            ),
            SimpleNamespace(returncode=0, stdout="Loaded image\n", stderr=""),
            exact_image,
        ]
    )

    def fake_docker(arguments: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(arguments)
        return next(responses)

    monkeypatch.setattr(worker, "_docker", fake_docker)
    worker._image_preflight()

    assert calls == [
        ["image", "inspect", worker.IMAGE_REFERENCE],
        ["image", "load", "--input", str(archive)],
        ["image", "inspect", worker.IMAGE_REFERENCE],
    ]


def test_local_image_preflight_repairs_exact_manifest_only_docker_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "runtime.oci.tar"
    archive.write_bytes(b"pinned-oci")
    archive.chmod(0o600)
    monkeypatch.setattr(worker, "IMAGE_ARCHIVE_PATH", archive)
    monkeypatch.setattr(worker, "IMAGE_ARCHIVE_SIZE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(worker, "_sha256", lambda _path: worker.IMAGE_ARCHIVE_SHA256)
    manifest_identity = {
        "Id": worker.IMAGE_ID,
        "Descriptor": {"digest": worker.IMAGE_DIGEST},
        "RepoDigests": [worker.runtime_contract.QUALIFIED_IMAGE_REPO_DIGEST],
    }
    calls: list[list[str]] = []
    responses = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            **manifest_identity,
                            "Architecture": "",
                            "Os": "",
                            "RootFS": {},
                            "Config": {},
                        }
                    ]
                ),
                stderr="",
            ),
            SimpleNamespace(returncode=0, stdout="Loaded image\n", stderr=""),
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            **manifest_identity,
                            "Architecture": "amd64",
                            "Os": "linux",
                            "RootFS": {"Type": "layers"},
                            "Config": {
                                "Labels": dict(
                                    worker.runtime_contract.EXPECTED_IMAGE_LABELS
                                )
                            },
                        }
                    ]
                ),
                stderr="",
            ),
        ]
    )

    def fake_docker(arguments: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(arguments)
        return next(responses)

    monkeypatch.setattr(worker, "_docker", fake_docker)
    worker._image_preflight()

    assert calls == [
        ["image", "inspect", worker.IMAGE_REFERENCE],
        ["image", "load", "--input", str(archive)],
        ["image", "inspect", worker.IMAGE_REFERENCE],
    ]


def test_local_image_preflight_never_loads_on_ambiguous_daemon_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "runtime.oci.tar"
    archive.write_bytes(b"pinned-oci")
    archive.chmod(0o600)
    monkeypatch.setattr(worker, "IMAGE_ARCHIVE_PATH", archive)
    monkeypatch.setattr(worker, "IMAGE_ARCHIVE_SIZE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(worker, "_sha256", lambda _path: worker.IMAGE_ARCHIVE_SHA256)
    calls: list[list[str]] = []

    def denied(arguments: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(arguments)
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="permission denied while trying to connect to the Docker API",
        )

    monkeypatch.setattr(worker, "_docker", denied)
    with pytest.raises(
        worker.QualificationWorkerError,
        match="image availability is ambiguous",
    ):
        worker._image_preflight()
    assert calls == [["image", "inspect", worker.IMAGE_REFERENCE]]


def test_runtime_binding_partitions_live_fields_without_raw_hardware_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _private(tmp_path / "source")
    supervisor = source / "aeon/scripts/qwen_flash_next_container_supervisor.py"
    supervisor.parent.mkdir(mode=0o700, parents=True)
    supervisor.write_text("# pinned\n", encoding="utf-8")
    supervisor.chmod(0o600)
    checkpoint = _private(tmp_path / "model")
    evidence = _private(tmp_path / "evidence")
    raw_uuid = "GPU-11111111-2222-3333-4444-555555555555"
    raw_claim = "gc-owner-private-claim"
    request = {
        "gpu_uuid": raw_uuid,
        "claim_id": raw_claim,
        "runtime_id": "fr-" + "a" * 32,
    }
    tuning = worker.RuntimeTuning.safe_baseline()
    command = worker._server_command(
        worker.ARM_TUNED_MTP_OFF, model_path="/model", tuning=tuning
    )
    labels = {"aeon.fleet.runtime": request["runtime_id"]}
    item: dict[str, Any] = {
        "Config": {
            "Image": worker.IMAGE_ID,
            "User": f"{os.geteuid()}:{os.getegid()}",
            "Cmd": ["python3", "/qualification/supervisor.py"],
        },
        "HostConfig": {
            "DeviceRequests": [{"DeviceIDs": [raw_uuid], "Capabilities": [["gpu"]]}],
            "Memory": worker.TASK_MEMORY_BYTES,
            "MemorySwap": worker.TASK_MEMORY_BYTES,
            "ShmSize": 32 * 1024**3,
            "PidsLimit": 4096,
            "Ulimits": [{"Name": "memlock", "Hard": -1, "Soft": -1}],
            "SecurityOpt": ["no-new-privileges=true"],
        },
        "NetworkSettings": {
            "Ports": {
                f"{worker.CONTAINER_PORT}/tcp": [
                    {"HostIp": "127.0.0.1", "HostPort": str(worker.HOST_PORT)}
                ]
            }
        },
        "Mounts": [],
    }

    class Client:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def get_json(self, path: str) -> tuple[dict[str, Any], str]:
            assert path == "/server_info"
            return {"raw": True}, "f" * 64

    monkeypatch.setattr(worker, "_paths", lambda _request: {"source": source})
    monkeypatch.setattr(
        worker,
        "_checkpoint_for_arm",
        lambda _request, _arm: (checkpoint, "c" * 64),
    )
    monkeypatch.setattr(worker, "_labels", lambda *_args, **_kwargs: labels)
    monkeypatch.setattr(worker.qualify, "EndpointClient", Client)
    monkeypatch.setattr(
        worker.qualify,
        "_sanitize_server_info",
        lambda *_args, **_kwargs: {
            "tp_size": 1,
            "mamba_ssm_dtype": "float32",
            "mem_fraction_static": 0.88,
        },
    )

    binding = worker._runtime_config_binding(
        request,
        worker.ARM_TUNED_MTP_OFF,
        command,
        {"root": evidence},
        tuning,
        item,
    )

    assert binding["command_sha256"] == worker._canonical_sha(command)
    assert binding["live_server_info_fields"] == [
        "mamba_ssm_dtype",
        "mem_fraction_static",
        "tp_size",
    ]
    assert (
        set(binding["live_server_info_fields"])
        | set(binding["unexposed_server_info_fields"])
        == worker.qualify.RUNTIME_CONFIG_FIELDS
    )
    encoded = json.dumps(binding, sort_keys=True)
    assert raw_uuid not in encoded
    assert raw_claim not in encoded
    assert str(checkpoint) not in encoded
    assert hashlib.sha256(raw_uuid.encode()).hexdigest() not in encoded


def test_release_runtime_uses_portable_read_only_materialized_model_mount() -> None:
    off_tuning = worker.RuntimeTuning.safe_baseline()
    on_tuning = worker.replace(off_tuning, nextn=(3, 4))
    off_config = worker._runtime_config(worker.ARM_TUNED_MTP_OFF, off_tuning)
    on_config = worker._runtime_config(worker.ARM_TUNED_MTP_ON, on_tuning)
    request = {
        "repo_id": "aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP",
        "checkpoint_tree_sha256": "c" * 64,
    }
    runtime = worker._runtime_release_config(
        request,
        {
            "runtime_identity": {
                "config_sha256": worker._canonical_sha(off_config),
                "runtime_config": off_config,
            }
        },
        {
            "runtime_identity": {
                "config_sha256": worker._canonical_sha(on_config),
                "runtime_config": on_config,
            }
        },
        off_tuning=off_tuning,
        on_tuning=on_tuning,
    )

    assert runtime["model_path_contract"] == {
        "checkpoint_tree_sha256": "c" * 64,
        "host_path_placeholder": worker.MATERIALIZED_MODEL_PLACEHOLDER,
        "container_path": "/model",
        "mount_read_only": True,
        "source_role": "offline-materialized-canonical-checkpoint",
    }
    assert runtime["launch_contract"] == worker.release_tool.LAUNCH_CONTRACT
    assert set(runtime["arms"]) == {
        "tuned_mtp_off",
        "tuned_mtp_on_winner",
    }
    for arm in runtime["arms"].values():
        command = arm["command"]
        assert command[command.index("--mount") + 1] == (
            "type=bind,src=@AEON_MATERIALIZED_MODEL_PATH@,dst=/model,readonly"
        )
        assert command[command.index("--model-path") + 1] == "/model"
        assert command[command.index("--context-length") + 1] == "65536"
        assert command[command.index("--max-total-tokens") + 1] == "65536"
        assert command[command.index("--page-size") + 1] == "64"
        assert worker.IMAGE_REFERENCE in command
        assert worker.IMAGE_CONFIG_DIGEST not in command
        assert arm["environment"] == worker.CONSTANT_RUNTIME_ENV


def test_final_mtp_pair_changes_only_native_speculative_fields() -> None:
    winner = worker.RuntimeTuning(
        moe_runner_backend=worker.runtime_contract.PREFERRED_MOE_RUNNER_BACKEND,
        cuda_graph="full",
        linear_decode_backend="triton",
        linear_prefill_backend="cutedsl",
        replay_ssm=True,
        mamba_ssm_dtype="float32",
        nextn=(3, 4),
        chunked_prefill_size=8192,
        mem_fraction_static="0.86",
    )

    off = worker._final_mtp_off_tuning(winner)
    off_config = worker._runtime_config(worker.ARM_TUNED_MTP_OFF, off)
    on_config = worker._runtime_config(worker.ARM_TUNED_MTP_ON, winner)
    delta = {key for key in off_config if off_config[key] != on_config[key]}

    assert off.replay_ssm is True
    assert off.linear_prefill_backend == "cutedsl"
    assert off.chunked_prefill_size == 8192
    assert off.mem_fraction_static == "0.86"
    assert delta == {
        "requested_speculative_algorithm",
        "speculative_algorithm",
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
    }


def test_selector_workloads_fit_the_pinned_sglang_page_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert worker.WORKLOAD_SPECS == worker.qualify._WORKLOAD_SPECS
    worker._validate_workload_scheduler_budgets()

    monkeypatch.setitem(
        worker.WORKLOAD_SPECS,
        "prefill_65152_256",
        (1, 65_216, 256),
    )
    with pytest.raises(
        worker.QualificationWorkerError,
        match="exceeds the SGLang scheduler budget",
    ):
        worker._validate_workload_scheduler_budgets()


def test_replay_reference_preserves_a_bf16_flashinfer_state_winner() -> None:
    winner = worker.RuntimeTuning(
        moe_runner_backend=worker.runtime_contract.PREFERRED_MOE_RUNNER_BACKEND,
        cuda_graph="full",
        linear_decode_backend="flashinfer",
        linear_prefill_backend="triton",
        replay_ssm=False,
        mamba_ssm_dtype="bfloat16",
        nextn=(3, 4),
        chunked_prefill_size=4096,
        mem_fraction_static="0.88",
    )

    assert worker._replay_off_reference_tuning(winner) is winner


def test_selector_pipeline_is_staged_with_closed_lineage_and_first_passing_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[worker.CandidateOutcome] = []

    def fake_run_candidate(
        _request: dict[str, Any],
        *,
        ordinal: int,
        candidate_id: str,
        phase: str,
        tuning: worker.RuntimeTuning,
        parent: worker.CandidateOutcome | None,
    ) -> worker.CandidateOutcome:
        identity = worker.CandidateIdentity(
            candidate_id,
            phase,
            ordinal,
            parent.identity.candidate_id if parent else None,
            parent.config_sha256 if parent else None,
        )
        workloads = [
            {
                "workload_id": workload_id,
                "metrics": {
                    "completion_tps": 100.0,
                    "effective_prefill_tps": 100.0,
                },
            }
            for workload_id in sorted(worker.PHASE_WORKLOADS[phase])
        ]
        config = worker._runtime_config(worker.ARM_SELECTION, tuning)
        report = {
            "passed": True,
            "runtime_identity": {
                "config_sha256": worker._canonical_sha(config),
            },
            "workload_evidence": {"workloads": workloads},
            "workload_validation": {"passed": True},
            "resources": {
                "memory_limit_and_oom_events_zero_before_and_after": True,
                "vram_budget_passed": True,
                "ram_budget_passed": True,
                "physical_cuda_reserve_passed": True,
                "physical_cuda_memory": {
                    "min_reserve_bytes": 9 * 1024**3,
                },
            },
        }
        outcome = worker.CandidateOutcome(
            identity, tuning, report, tmp_path / f"{identity.key}.arm.json"
        )
        captured.append(outcome)
        return outcome

    monkeypatch.setattr(worker, "_progress", lambda *_args: None)
    monkeypatch.setattr(worker, "_run_candidate", fake_run_candidate)
    monkeypatch.setattr(
        worker.qualify,
        "_validate_one_state_dtype_peer_equivalence",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        worker,
        "_rank_mtp_finalists",
        lambda _reference, finalists: finalists[0],
    )

    outcomes, winner = worker._selection_pipeline({})
    by_id = {outcome.identity.candidate_id: outcome for outcome in outcomes}

    assert outcomes == captured
    assert len(outcomes) == 27
    assert set(by_id) >= {
        "moe_cutlass",
        "graph_eager",
        "graph_full",
        "gdn_tt_fp32",
        "gdn_ct_fp32",
        "gdn_tc_fp32",
        "gdn_cc_fp32",
        "state_ft_fp32_ref",
        "state_ft_bf16",
        "mtp_s1_d2",
        "mtp_s2_d3",
        "mtp_s3_d4",
        "mtp_none_finalist_ref",
        "replay_none_ref",
        "replay_tt_fp32",
        "replay_tc_fp32",
        "chunk_4096",
        "chunk_8192",
        "mem_084",
    }
    assert "mem_086" not in by_id and "mem_088" not in by_id
    assert winner.identity.candidate_id == "mem_084"
    assert by_id["moe_cutlass"].identity.parent_candidate_id is None
    assert by_id["moe_cutlass"].tuning.mamba_ssm_dtype == "bfloat16"
    assert by_id["moe_cutlass"].tuning.mem_fraction_static == "0.92"
    assert by_id["moe_cutlass"].tuning.cuda_graph == "disabled"
    assert by_id["graph_eager"].identity.parent_candidate_id == "moe_cutlass"
    assert by_id["graph_full"].identity.parent_candidate_id == "graph_eager"
    assert by_id["mtp_none_finalist_ref"].identity.parent_candidate_id == ("mtp_s1_d2")
    for code in ("cc", "ct"):
        assert by_id[f"state_{code}_bf16"].identity.parent_candidate_id == (
            f"state_{code}_fp32_ref"
        )
    assert by_id["state_ft_bf16"].identity.parent_candidate_id == ("state_ft_fp32_ref")
    assert by_id["replay_tt_fp32"].identity.parent_candidate_id == ("replay_none_ref")
    assert by_id["replay_tc_fp32"].identity.parent_candidate_id == ("replay_none_ref")


def test_behavior_fail_fast_skips_doomed_final_boots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    categories = sorted(worker.qualify.behavior_training.NON_HARMFUL_CATEGORIES)

    def report(total: int, by_category: dict[str, int]) -> dict[str, Any]:
        counts = {
            category: {
                "unnecessary_refusals": by_category.get(category, 0),
            }
            for category in categories
        }
        return {
            "behavioral_gate": {
                "summary": {
                    "counts": counts,
                    "non_harmful_unnecessary_refusals": total,
                }
            }
        }

    monkeypatch.setattr(
        worker.qualify,
        "_validated_behavior_report",
        lambda value, *, label: {"summary": value["summary"]},
    )
    with pytest.raises(worker.QualificationWorkerError, match="zero unnecessary"):
        worker._require_behavioral_improvement_possible(report(0, {}))

    official = report(2, {categories[0]: 1, categories[1]: 1})
    tuned = report(1, {categories[0]: 0, categories[1]: 1})
    worker._require_behavioral_improvement_possible(official)
    worker._require_behavioral_improvement_possible(official, tuned)

    regressed_category = report(1, {categories[0]: 0, categories[2]: 1})
    with pytest.raises(worker.QualificationWorkerError, match="strictly reduce"):
        worker._require_behavioral_improvement_possible(
            official, regressed_category
        )


def test_cutlass_moe_requires_complete_gates_and_comfortable_reserve(
    tmp_path: Path,
) -> None:
    baseline = worker.RuntimeTuning.safe_baseline()

    def outcome(
        *,
        throughput: float,
        reserve_bytes: int,
    ) -> worker.CandidateOutcome:
        identity = worker.CandidateIdentity(
            "moe_cutlass",
            "moe_backend",
            0,
            None,
            None,
        )
        config = worker._runtime_config(worker.ARM_SELECTION, baseline)
        return worker.CandidateOutcome(
            identity,
            baseline,
            {
                "passed": True,
                "runtime_identity": {
                    "config_sha256": worker._canonical_sha(config),
                },
                "workload_evidence": {
                    "workloads": [
                        {
                            "workload_id": workload_id,
                            "metrics": {
                                "completion_tps": throughput,
                                "effective_prefill_tps": throughput,
                            },
                        }
                        for workload_id in sorted(worker.PHASE_WORKLOADS["moe_backend"])
                    ]
                },
                "workload_validation": {"passed": True},
                "resources": {
                    "memory_limit_and_oom_events_zero_before_and_after": True,
                    "vram_budget_passed": True,
                    "ram_budget_passed": True,
                    "physical_cuda_reserve_passed": True,
                    "physical_cuda_memory": {
                        "min_reserve_bytes": reserve_bytes,
                    },
                },
            },
            tmp_path / "moe_cutlass.arm.json",
        )

    cutlass = outcome(
        throughput=104.0,
        reserve_bytes=8 * 1024**3,
    )

    assert worker._require_moe_cutlass(cutlass) is cutlass
    low_reserve = worker.CandidateOutcome(
        cutlass.identity,
        cutlass.tuning,
        {
            **cutlass.report,
            "resources": {
                **cutlass.report["resources"],
                "physical_cuda_memory": {
                    "min_reserve_bytes": 8 * 1024**3 - 1,
                },
            },
        },
        cutlass.report_path,
    )
    with pytest.raises(worker.QualificationWorkerError, match="comfortable"):
        worker._require_moe_cutlass(low_reserve)


def test_mtp_finalist_selector_prefers_paired_ci_lower_before_point_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def outcome(candidate_id: str, marker: str) -> worker.CandidateOutcome:
        identity = worker.CandidateIdentity(
            candidate_id,
            "mtp_finalist",
            0,
            "mtp_s1_d2",
            "a" * 64,
        )
        return worker.CandidateOutcome(
            identity,
            worker.replace(worker.RuntimeTuning.safe_baseline(), nextn=(1, 2)),
            {"marker": marker},
            tmp_path / f"{candidate_id}.json",
        )

    reference = outcome("mtp_none_finalist_ref", "off")
    finalists = (
        outcome("mtp_s1_d2_forward", "high-ci"),
        outcome("mtp_s1_d2_reverse", "high-ci"),
        outcome("mtp_s2_d3_forward", "high-point"),
        outcome("mtp_s2_d3_reverse", "high-point"),
    )
    monkeypatch.setattr(worker, "_rank_phase", lambda rows: list(rows))

    def rows(report: dict[str, str], _workload: str) -> list[dict[str, Any]]:
        marker = report["marker"]
        elapsed = {"off": 1.0, "high-ci": 0.8, "high-point": 0.7}[marker]
        return [
            {"completion_tokens": 100, "elapsed_seconds": elapsed, "marker": marker}
        ]

    monkeypatch.setattr(worker.qualify, "_completion_speed_rows", rows)
    monkeypatch.setattr(
        worker.qualify,
        "_paired_bootstrap_ci",
        lambda _off, on, **_kwargs: (
            (1.06, 1.20) if on[0]["marker"] == "high-ci" else (1.05, 1.30)
        ),
    )

    selected = worker._rank_mtp_finalists(reference, finalists)

    assert selected.identity.candidate_id == "mtp_s1_d2_forward"


def test_state_dtype_peer_regression_eliminates_only_bf16_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def outcome(candidate_id: str, marker: str) -> worker.CandidateOutcome:
        identity = worker.CandidateIdentity(
            candidate_id,
            "state_dtype",
            0,
            "gdn_tt_fp32"
            if candidate_id.endswith("_fp32_ref")
            else "state_tt_fp32_ref",
            "a" * 64,
        )
        return worker.CandidateOutcome(
            identity,
            worker.RuntimeTuning.safe_baseline(),
            {"marker": marker},
            tmp_path / f"{candidate_id}.json",
        )

    reference = outcome("state_tt_fp32_ref", "reference")
    bf16 = outcome("state_tt_bf16", "regressive")

    def validate(_peer: dict[str, str], candidate: dict[str, str]) -> None:
        if candidate["marker"] == "regressive":
            raise worker.qualify.StateDtypePeerRegression("bounded regression")

    monkeypatch.setattr(
        worker.qualify, "_validate_one_state_dtype_peer_equivalence", validate
    )

    admitted = worker._state_dtype_admissible_candidates([reference], [bf16])

    assert admitted == [reference]


def test_selector_boot_failure_receipt_is_private_sanitized_and_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _private(tmp_path / "output")
    sibling = tmp_path / worker.SIBLING_MANIFEST_NAME
    sibling.write_text(
        json.dumps(
            {
                "tuned_lm_head_tensor_sha256": "5" * 64,
                "non_lm_head_tensor_inventory_sha256": "6" * 64,
            }
        ),
        encoding="utf-8",
    )
    sibling.chmod(0o600)
    raw_claim = "gc-owner-private-selector-attempt"
    raw_uuid = "GPU-11111111-2222-3333-4444-555555555555"
    request = {
        "runtime_id": "fr-" + "1" * 32,
        "claim_id": raw_claim,
        "gpu_uuid": raw_uuid,
        "checkpoint_tree_sha256": "2" * 64,
        "build_sibling_manifest_sha256": "3" * 64,
    }
    baseline = worker.RuntimeTuning.safe_baseline()
    parent_config = worker._runtime_config(worker.ARM_SELECTION, baseline)
    parent = worker.CandidateOutcome(
        worker.CandidateIdentity("graph_eager", "graph", 1, "moe_cutlass", "a" * 64),
        baseline,
        {"runtime_identity": {"config_sha256": worker._canonical_sha(parent_config)}},
        output / "parent.arm.json",
    )
    failure = worker.SelectionBootFailure(
        stage="server_readiness",
        code="server_readiness_timeout",
        detail_sha256="7" * 64,
        container_config_sha256="8" * 64,
    )
    monkeypatch.setattr(
        worker,
        "_paths",
        lambda _request: {"output": output, "sibling_manifest": sibling},
    )
    monkeypatch.setattr(
        worker,
        "_run_arm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    def docker_summary(
        path: Path,
        *,
        failure_stage: str,
        failure_code: str,
        failure_detail_sha256: str,
        command_sha256: str,
        container_config_sha256: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        empty = {
            "sha256": hashlib.sha256(b"").hexdigest(),
            "utf8_bytes": 0,
            "truncated": False,
        }
        state = {
            "status": "exited",
            "running": False,
            "paused": False,
            "restarting": False,
            "oom_killed": False,
            "dead": False,
            "pid": 0,
            "exit_code": 1,
            "error": empty,
            "started_at": "2026-08-27T09:48:14Z",
            "finished_at": "2026-08-27T09:49:44Z",
        }
        return {
            "schema_version": (
                worker.qualify.SELECTION_DOCKER_FAILURE_SUMMARY_SCHEMA_VERSION
            ),
            "sidecar_name": path.name,
            "sidecar_sha256": "9" * 64,
            "sidecar_size_bytes": 100,
            "failure_stage": failure_stage,
            "failure_code": failure_code,
            "failure_detail_sha256": failure_detail_sha256,
            "container_id_sha256": "a" * 64,
            "command_sha256": command_sha256,
            "container_config_sha256": container_config_sha256,
            "captured_at": worker._now(),
            "docker_logs_exit_code": 0,
            "docker_state": state,
            "docker_state_sha256": worker._canonical_sha(state),
            "stdout": empty,
            "stderr": empty,
        }

    monkeypatch.setattr(
        worker, "_validate_selection_docker_failure_sidecar", docker_summary
    )

    attempt = worker._run_candidate(
        request,
        ordinal=3,
        candidate_id="graph_full",
        phase="graph",
        tuning=worker.replace(baseline, cuda_graph="full"),
        parent=parent,
    )

    assert isinstance(attempt, worker.CandidateAttempt)
    receipt = json.loads(attempt.report_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == (
        worker.qualify.SELECTION_ATTEMPT_SCHEMA_VERSION
    )
    assert receipt["ordered_index"] == 3
    assert receipt["failure_stage"] == "server_readiness"
    assert receipt["container_config_sha256"] == "8" * 64
    assert receipt["docker_failure_diagnostic"]["docker_state"]["exit_code"] == 1
    assert receipt["diagnostic_sidecars"] == {
        "03-graph-graph_full.docker-failure.json": "9" * 64
    }
    assert attempt.report_path.stat().st_mode & 0o777 == 0o600
    encoded = attempt.report_path.read_text(encoding="utf-8")
    assert raw_claim not in encoded
    assert raw_uuid not in encoded
    assert "claim_id" not in receipt
    worker.qualify._selection_candidate_record(
        attempt.report_path, expected_ordered_index=3
    )


def test_selector_docker_failure_sidecar_is_bounded_sanitized_and_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _private(tmp_path / "output")
    raw_claim = "gc-owner-private-docker-log"
    raw_uuid = "GPU-11111111-2222-3333-4444-555555555555"
    request = {
        "runtime_id": "fr-" + "1" * 32,
        "claim_id": raw_claim,
        "gpu_uuid": raw_uuid,
        "scratch_path": "/home/aday/private/run",
        "checkpoint_path": "/home/aday/private/model",
        "official_untuned_checkpoint_path": "/home/aday/private/untuned",
        "build_sibling_manifest_path": "/home/aday/private/sibling.json",
    }
    candidate = worker.CandidateIdentity("moe_cutlass", "moe_backend", 0, None, None)
    tuning = worker.RuntimeTuning.safe_baseline()
    container_id = "b" * 64
    projection = {"binding": "exact"}
    container_config_sha256 = worker._canonical_sha(projection)
    failure = worker.SelectionBootFailure(
        stage="server_readiness",
        code="container_exited_during_load",
        detail_sha256="7" * 64,
        container_config_sha256=container_config_sha256,
    )
    state = {
        "Status": "exited",
        "Running": False,
        "Paused": False,
        "Restarting": False,
        "OOMKilled": False,
        "Dead": False,
        "Pid": 0,
        "ExitCode": 1,
        "Error": f"daemon error for {raw_claim}",
        "StartedAt": "2026-08-27T09:48:14Z",
        "FinishedAt": "2026-08-27T09:49:44Z",
    }
    item = {"Id": container_id, "State": state}
    stdout = (
        "x" * (worker.qualify.MAX_SELECTION_DOCKER_LOG_TAIL_BYTES + 1024)
        + f"\nclaim={raw_claim}\nTRACEBACK-END\n"
    )
    stderr = (
        f"uuid={raw_uuid} token=hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        "Authorization: Bearer private-token /home/aday/private/file\n"
    )

    monkeypatch.setattr(worker, "_paths", lambda _request: {"output": output})
    monkeypatch.setattr(worker, "_verify_container", lambda *_args, **_kwargs: item)
    monkeypatch.setattr(
        worker, "_container_config_projection", lambda *_args, **_kwargs: projection
    )

    def docker(arguments: list[str], *, timeout: float = 120):
        assert arguments == [
            "container",
            "logs",
            "--tail",
            str(worker.SELECTION_DOCKER_LOG_TAIL_LINES),
            container_id,
        ]
        assert timeout == 30
        return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(worker, "_docker", docker)

    summary = worker._persist_selection_docker_failure(
        request,
        container_id=container_id,
        command=["python3", "-m", "sglang.launch_server"],
        evidence={"root": tmp_path / "evidence"},
        tuning=tuning,
        candidate=candidate,
        failure=failure,
    )

    sidecar_path = output / "00-moe_backend-moe_cutlass.docker-failure.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    encoded = sidecar_path.read_text(encoding="utf-8")
    assert sidecar_path.stat().st_mode & 0o777 == 0o600
    assert sidecar_path.stat().st_size <= (
        worker.qualify.MAX_SELECTION_DOCKER_FAILURE_SIDECAR_BYTES
    )
    assert sidecar["docker_state"]["exit_code"] == 1
    assert sidecar["stdout"]["truncated"] is True
    assert sidecar["stdout"]["utf8_bytes"] <= (
        worker.qualify.MAX_SELECTION_DOCKER_LOG_TAIL_BYTES
    )
    assert "TRACEBACK-END" in sidecar["stdout"]["tail"]
    assert raw_claim not in encoded
    assert raw_uuid not in encoded
    assert "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in encoded
    assert "private-token" not in encoded
    assert "/home/aday/private" not in encoded
    assert container_id not in encoded
    assert summary["sidecar_sha256"] == worker._sha256(sidecar_path)
    assert summary["sidecar_size_bytes"] == sidecar_path.stat().st_size
    assert summary["docker_state"]["exit_code"] == 1

    attempt_path = output / "00-moe_backend-moe_cutlass.attempt.json"
    worker._atomic_json(
        attempt_path,
        {
            "failure_stage": failure.stage,
            "failure_code": failure.code,
            "failure_detail_sha256": failure.detail_sha256,
            "command_sha256": worker._canonical_sha(
                ["python3", "-m", "sglang.launch_server"]
            ),
            "container_config_sha256": container_config_sha256,
            "diagnostic_sidecars": {sidecar_path.name: summary["sidecar_sha256"]},
            "docker_failure_diagnostic": summary,
        },
    )
    attempt = worker.CandidateAttempt(candidate, tuning, attempt_path)
    worker._revalidate_candidate_attempt_diagnostic(request, attempt)

    sidecar["stdout"]["tail"] += "tampered"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    sidecar_path.chmod(0o600)
    with pytest.raises(worker.QualificationWorkerError, match="digest changed"):
        worker._revalidate_candidate_attempt_diagnostic(request, attempt)


def test_selector_pipeline_continues_optional_preidentity_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempted = {
        "graph_full",
        "gdn_cc_fp32",
        "state_tc_bf16",
        "mtp_s3_d4",
        "mtp_s2_d3_reverse",
        "replay_tc_fp32",
        "chunk_8192",
        "mem_084",
    }

    def fake_run_candidate(
        _request: dict[str, Any],
        *,
        ordinal: int,
        candidate_id: str,
        phase: str,
        tuning: worker.RuntimeTuning,
        parent: worker.CandidateOutcome | None,
    ) -> worker.CandidateEvidence:
        identity = worker.CandidateIdentity(
            candidate_id,
            phase,
            ordinal,
            parent.identity.candidate_id if parent else None,
            parent.config_sha256 if parent else None,
        )
        if candidate_id in attempted:
            return worker.CandidateAttempt(
                identity, tuning, tmp_path / f"{identity.key}.attempt.json"
            )
        config = worker._runtime_config(worker.ARM_SELECTION, tuning)
        report = {
            "passed": True,
            "runtime_identity": {
                "config_sha256": worker._canonical_sha(config),
            },
            "workload_evidence": {
                "workloads": [
                    {
                        "workload_id": workload_id,
                        "metrics": {
                            "completion_tps": 100.0,
                            "effective_prefill_tps": 100.0,
                        },
                    }
                    for workload_id in sorted(worker.PHASE_WORKLOADS[phase])
                ]
            },
            "workload_validation": {"passed": True},
            "resources": {
                "memory_limit_and_oom_events_zero_before_and_after": True,
                "vram_budget_passed": True,
                "ram_budget_passed": True,
                "physical_cuda_reserve_passed": True,
                "physical_cuda_memory": {
                    "min_reserve_bytes": 9 * 1024**3,
                },
            },
        }
        return worker.CandidateOutcome(
            identity, tuning, report, tmp_path / f"{identity.key}.arm.json"
        )

    monkeypatch.setattr(worker, "_progress", lambda *_args: None)
    monkeypatch.setattr(worker, "_run_candidate", fake_run_candidate)
    monkeypatch.setattr(
        worker.qualify,
        "_validate_one_state_dtype_peer_equivalence",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        worker,
        "_rank_mtp_finalists",
        lambda _reference, finalists: finalists[0],
    )

    evidence, winner = worker._selection_pipeline({})
    evidence_by_id = {item.identity.candidate_id: item for item in evidence}

    assert attempted <= {
        candidate_id
        for candidate_id, item in evidence_by_id.items()
        if isinstance(item, worker.CandidateAttempt)
    }
    assert len(evidence) == 28
    assert winner.identity.candidate_id == "mem_086"
    assert [item.identity.ordinal for item in evidence] == list(range(len(evidence)))


def test_scratch_accounting_counts_hardlinks_once_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    root = _private(tmp_path / "scratch")
    first = root / "first.bin"
    second = root / "second.bin"
    first.write_bytes(b"x" * 8192)
    first.chmod(0o600)
    os.link(first, second)

    expected = root.lstat().st_blocks * 512 + first.lstat().st_blocks * 512
    assert worker._safe_tree_bytes(root) == expected
    assert first.lstat().st_ino == second.lstat().st_ino

    os.symlink(first, root / "unsafe-link")
    with pytest.raises(worker.QualificationWorkerError, match="unsafe inode"):
        worker._safe_tree_bytes(root)


def test_adapter_creates_only_missing_private_canonical_root_and_runtime_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _private(tmp_path / "artifacts")
    canonical_root = parent / "qualification"
    runtime_id = "fr-" + "6" * 32
    run = canonical_root / runtime_id
    monkeypatch.setattr(adapter, "CANONICAL_OUTPUT_ROOT", canonical_root)

    _filesystem, _free, _inodes, _allocated = adapter._metrics(str(run), create=True)

    assert canonical_root.is_dir()
    assert canonical_root.stat().st_mode & 0o777 == 0o700
    assert run.is_dir()
    assert run.stat().st_mode & 0o777 == 0o700


def test_adapter_refuses_unsafe_preexisting_canonical_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _private(tmp_path / "artifacts")
    canonical_root = parent / "qualification"
    canonical_root.mkdir(mode=0o755)
    canonical_root.chmod(0o755)
    runtime_id = "fr-" + "5" * 32
    monkeypatch.setattr(adapter, "CANONICAL_OUTPUT_ROOT", canonical_root)

    with pytest.raises(
        adapter.FlashNextQualificationError,
        match="canonical qualification root is unsafe",
    ):
        adapter._metrics(str(canonical_root / runtime_id), create=True)


def test_local_qualification_cleanup_actions_retain_canonical_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = _private(tmp_path / "scratch")
    retained = scratch / "retained.bin"
    retained.write_bytes(b"retain canonical .177 qualification run")
    retained.chmod(0o600)
    manifest = scratch / "MANIFEST.sha256"
    manifest.write_text("evidence\n", encoding="utf-8")
    manifest.chmod(0o600)
    digest = worker._sha256(manifest)
    settled = scratch / "settled.json"
    settled.write_text(
        json.dumps(
            {
                "runtime_id": "fr-" + "1" * 32,
                "manifest_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    settled.chmod(0o600)
    paths = {
        "scratch": scratch,
        "manifest": manifest,
        "settled": settled,
        "spawn": scratch / "absent-spawn.json",
        "status": scratch / "absent-status.json",
        "output": scratch / "absent-output",
    }
    request = {"runtime_id": "fr-" + "1" * 32}
    monkeypatch.setattr(worker, "_paths", lambda _request: paths)
    monkeypatch.setattr(worker, "_status", lambda _request: {"state": "completed"})

    settled_result = worker._cleanup(request, digest)
    prelaunch_result = worker._cleanup_prelaunch(request)

    assert settled_result == {"state": "retained", "reclaimed_bytes": 0}
    assert prelaunch_result == {"state": "retained", "reclaimed_bytes": 0}
    assert retained.read_bytes() == b"retain canonical .177 qualification run"
    assert scratch.is_dir()


def test_adapter_finalization_never_requests_local_canonical_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = adapter.AeonQwenFlashNextQualificationAdapter()
    runtime_id = "fr-" + "7" * 32
    run_root = PurePosixPath(tmp_path / "lease-runs")
    canonical_root = _private(tmp_path / "canonical")
    canonical = _private(canonical_root / runtime_id)
    output = _private(canonical / "output")
    partial = output / "partial-evidence.json"
    partial.write_text('{"partial":true}\n', encoding="utf-8")
    partial.chmod(0o600)
    source = _private(canonical / "source")
    assets = _private(canonical / "assets")
    request = canonical / "qualification-request.json"
    request.write_bytes(b"request\n")
    request.chmod(0o600)
    retained = source / "worker.py"
    retained.write_bytes(b"worker\n")
    retained.chmod(0o600)
    media = assets / "image.jpg"
    media.write_bytes(b"image\n")
    media.chmod(0o600)
    digest = "a" * 64
    runtime = {
        "runtime_id": runtime_id,
        "profile_id": adapter.PROFILE_ID,
        "host": adapter.HOST,
        "physical_gpu": 0,
        "run_dir": str(run_root / runtime_id),
        "pid": 1234,
        "process_absent": 1,
        "process_identity": (
            f"aeon-flash-next-qualification:{runtime_id}:{digest}:1234"
        ),
    }
    monkeypatch.setattr(adapter, "RUN_ROOT", run_root)
    monkeypatch.setattr(adapter, "CANONICAL_OUTPUT_ROOT", canonical_root)
    events: list[str] = []

    def metrics(_path: str, *, create: bool) -> tuple[str, int, int, int]:
        assert create is False
        events.append("metrics")
        return "device", 1, 1, 1

    monkeypatch.setattr(
        adapter,
        "_metrics",
        metrics,
    )

    prelaunch = instance.finalize_storage(
        {**runtime, "process_identity": None, "pid": None},
        {"scratch_path": None, "canonical_output_path": str(canonical)},
    )
    assert prelaunch.output_settled is True
    assert prelaunch.cleanup_complete is True
    assert prelaunch.reclaimed_bytes == 0
    assert "retained" in prelaunch.note
    assert events == ["metrics"]
    events.clear()

    with pytest.raises(
        adapter.FlashNextQualificationError,
        match="storage gained worker scratch",
    ):
        instance.finalize_storage(
            {**runtime, "process_identity": None, "pid": None},
            {
                "scratch_path": str(run_root / runtime_id),
                "canonical_output_path": str(canonical),
            },
        )
    assert events == []

    actions: list[str] = []

    def incomplete_action(_runtime, name: str, extra=None, *, timeout=120):
        actions.append(name)
        if name == "status":
            return {"state": "failed", "pid": None}
        pytest.fail(f"unexpected qualification lifecycle action: {name}")

    monkeypatch.setattr(instance, "_action", incomplete_action)
    before_paths = tuple(
        sorted(
            (item.relative_to(canonical).as_posix(), item.is_dir())
            for item in canonical.rglob("*")
        )
    )
    before_files = {
        item.relative_to(canonical).as_posix(): item.read_bytes()
        for item in canonical.rglob("*")
        if item.is_file()
    }
    incomplete = instance.finalize_storage(
        runtime,
        {
            "scratch_path": None,
            "canonical_output_path": str(canonical),
            "terminal_success": 0,
        },
    )
    assert actions == ["status"]
    assert incomplete.output_settled is True
    assert incomplete.cleanup_complete is True
    assert incomplete.reclaimed_bytes == 0
    assert "incomplete qualification retained" in incomplete.note
    assert (
        tuple(
            sorted(
                (item.relative_to(canonical).as_posix(), item.is_dir())
                for item in canonical.rglob("*")
            )
        )
        == before_paths
    )
    assert {
        item.relative_to(canonical).as_posix(): item.read_bytes()
        for item in canonical.rglob("*")
        if item.is_file()
    } == before_files
    assert not (canonical / "qualification-settled.json").exists()
    assert events == ["metrics"]
    actions.clear()
    events.clear()

    validated: list[Path] = []

    def local_valid(path: Path) -> tuple[bool, str]:
        events.append("local_valid")
        validated.append(path)
        return True, digest

    monkeypatch.setattr(instance, "_local_valid", local_valid)

    def action(_runtime, name: str, extra=None, *, timeout=120):
        actions.append(name)
        if name == "status":
            return {"state": "completed", "pid": None}
        if name == "settle-status":
            return {"manifest_sha256": digest}
        if name == "mark-settled" and extra == digest:
            return {"state": "settled"}
        pytest.fail(f"unexpected qualification lifecycle action: {name}")

    monkeypatch.setattr(instance, "_action", action)
    completed = instance.finalize_storage(
        runtime,
        {
            "scratch_path": None,
            "canonical_output_path": str(canonical),
            "terminal_success": 0,
        },
    )

    assert actions == ["status", "settle-status", "mark-settled"]
    assert events == ["metrics", "local_valid"]
    assert validated == [output]
    assert completed.output_settled is True
    assert completed.cleanup_complete is True
    assert completed.reclaimed_bytes == 0
    assert "retained" in completed.note
    assert retained.read_bytes() == b"worker\n"
    assert media.read_bytes() == b"image\n"
    assert request.read_bytes() == b"request\n"
    assert partial.read_text(encoding="utf-8") == '{"partial":true}\n'
    assert canonical.is_dir()


@pytest.mark.parametrize(
    ("storage_extra", "process_absent", "lifecycle"),
    (
        pytest.param({}, 1, {"state": "failed", "pid": None}, id="missing-outcome"),
        pytest.param(
            {"terminal_success": None},
            1,
            {"state": "failed", "pid": None},
            id="null-outcome",
        ),
        pytest.param(
            {"terminal_success": 1},
            1,
            {"state": "failed", "pid": None},
            id="successful-outcome",
        ),
        pytest.param(
            {"terminal_success": 0},
            0,
            {"state": "failed", "pid": None},
            id="process-not-durably-absent",
        ),
        pytest.param(
            {"terminal_success": 0},
            1,
            {"state": "failed", "pid": 1234},
            id="failed-with-pid",
        ),
        pytest.param(
            {"terminal_success": 0},
            1,
            {"state": "running", "pid": 1234},
            id="running",
        ),
        pytest.param(
            {"terminal_success": 0},
            1,
            {"state": "unknown", "pid": 1234},
            id="unknown",
        ),
        pytest.param(
            {"terminal_success": 0},
            1,
            {"state": "absent", "pid": None},
            id="absent-without-failed-status",
        ),
    ),
)
def test_adapter_incomplete_finalization_fails_closed_on_ambiguous_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage_extra: dict[str, Any],
    process_absent: int,
    lifecycle: dict[str, Any],
) -> None:
    instance = adapter.AeonQwenFlashNextQualificationAdapter()
    runtime_id = "fr-" + "2" * 32
    run_root = PurePosixPath(tmp_path / "lease-runs")
    canonical_root = _private(tmp_path / "canonical")
    canonical = _private(canonical_root / runtime_id)
    _private(canonical / "output")
    digest = "c" * 64
    runtime = {
        "runtime_id": runtime_id,
        "profile_id": adapter.PROFILE_ID,
        "host": adapter.HOST,
        "physical_gpu": 0,
        "run_dir": str(run_root / runtime_id),
        "pid": 1234,
        "process_absent": process_absent,
        "process_identity": (
            f"aeon-flash-next-qualification:{runtime_id}:{digest}:1234"
        ),
    }
    storage = {
        "scratch_path": None,
        "canonical_output_path": str(canonical),
        **storage_extra,
    }
    monkeypatch.setattr(adapter, "RUN_ROOT", run_root)
    monkeypatch.setattr(adapter, "CANONICAL_OUTPUT_ROOT", canonical_root)
    monkeypatch.setattr(
        adapter,
        "_metrics",
        lambda _path, *, create: ("device", 1, 1, 1),
    )
    actions: list[str] = []

    def action(_runtime, name: str, extra=None, *, timeout=120):
        actions.append(name)
        if name == "status":
            return lifecycle
        pytest.fail(f"unexpected qualification lifecycle action: {name}")

    monkeypatch.setattr(instance, "_action", action)

    with pytest.raises(adapter.FlashNextQualificationError, match="ambiguous"):
        instance.finalize_storage(runtime, storage)

    assert actions == ["status"]
    assert not (canonical / "qualification-settled.json").exists()


@pytest.mark.parametrize(
    "contradiction",
    ("output/MANIFEST.sha256", "qualification-settled.json"),
)
def test_adapter_incomplete_finalization_rejects_closed_result_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contradiction: str,
) -> None:
    instance = adapter.AeonQwenFlashNextQualificationAdapter()
    runtime_id = "fr-" + "3" * 32
    run_root = PurePosixPath(tmp_path / "lease-runs")
    canonical_root = _private(tmp_path / "canonical")
    canonical = _private(canonical_root / runtime_id)
    _private(canonical / "output")
    marker = canonical / contradiction
    marker.write_bytes(b"" if marker.name == "MANIFEST.sha256" else b"{}\n")
    marker.chmod(0o600)
    digest = "d" * 64
    runtime = {
        "runtime_id": runtime_id,
        "profile_id": adapter.PROFILE_ID,
        "host": adapter.HOST,
        "physical_gpu": 0,
        "run_dir": str(run_root / runtime_id),
        "pid": 1234,
        "process_absent": 1,
        "process_identity": (
            f"aeon-flash-next-qualification:{runtime_id}:{digest}:1234"
        ),
    }
    monkeypatch.setattr(adapter, "RUN_ROOT", run_root)
    monkeypatch.setattr(adapter, "CANONICAL_OUTPUT_ROOT", canonical_root)
    monkeypatch.setattr(
        adapter,
        "_metrics",
        lambda _path, *, create: ("device", 1, 1, 1),
    )
    actions: list[str] = []

    def action(_runtime, name: str, extra=None, *, timeout=120):
        actions.append(name)
        if name == "status":
            return {"state": "failed", "pid": None}
        pytest.fail(f"unexpected qualification lifecycle action: {name}")

    monkeypatch.setattr(instance, "_action", action)
    before = marker.read_bytes()

    with pytest.raises(adapter.FlashNextQualificationError, match="ambiguous"):
        instance.finalize_storage(
            runtime,
            {
                "scratch_path": None,
                "canonical_output_path": str(canonical),
                "terminal_success": 0,
            },
        )

    assert actions == ["status"]
    assert marker.read_bytes() == before


def test_adapter_incomplete_finalization_rejects_unsafe_canonical_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = adapter.AeonQwenFlashNextQualificationAdapter()
    runtime_id = "fr-" + "4" * 32
    run_root = PurePosixPath(tmp_path / "lease-runs")
    canonical_root = _private(tmp_path / "canonical")
    canonical = _private(canonical_root / runtime_id)
    output = _private(canonical / "output")
    partial = output / "partial.bin"
    partial.write_bytes(b"partial")
    partial.chmod(0o600)
    (output / "unsafe-link").symlink_to(partial)
    digest = "e" * 64
    runtime = {
        "runtime_id": runtime_id,
        "profile_id": adapter.PROFILE_ID,
        "host": adapter.HOST,
        "physical_gpu": 0,
        "run_dir": str(run_root / runtime_id),
        "pid": 1234,
        "process_absent": 1,
        "process_identity": (
            f"aeon-flash-next-qualification:{runtime_id}:{digest}:1234"
        ),
    }
    monkeypatch.setattr(adapter, "RUN_ROOT", run_root)
    monkeypatch.setattr(adapter, "CANONICAL_OUTPUT_ROOT", canonical_root)
    actions: list[str] = []
    monkeypatch.setattr(
        instance,
        "_action",
        lambda *_args, **_kwargs: actions.append("unexpected"),
    )

    with pytest.raises(adapter.FlashNextQualificationError, match="unsafe inode"):
        instance.finalize_storage(
            runtime,
            {
                "scratch_path": None,
                "canonical_output_path": str(canonical),
                "terminal_success": 0,
            },
        )

    assert actions == []
    assert partial.read_bytes() == b"partial"


def test_fresh_adapter_probe_derives_canonical_path_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_id = "fr-" + "8" * 32
    digest = "b" * 64
    pid = 4321
    run_root = PurePosixPath(tmp_path / "lease-runs")
    canonical_root = _private(tmp_path / "canonical")
    runtime = {
        "runtime_id": runtime_id,
        "profile_id": adapter.PROFILE_ID,
        "host": adapter.HOST,
        "physical_gpu": 0,
        "run_dir": str(run_root / runtime_id),
        "pid": pid,
        "process_identity": (
            f"aeon-flash-next-qualification:{runtime_id}:{digest}:{pid}"
        ),
    }
    calls: list[tuple[str, str, str, str]] = []
    monkeypatch.setattr(adapter, "RUN_ROOT", run_root)
    monkeypatch.setattr(adapter, "CANONICAL_OUTPUT_ROOT", canonical_root)

    def worker_action(
        source: str,
        action: str,
        request: str,
        request_digest: str,
        _extra: str | None = None,
        *,
        timeout: float = 120,
    ) -> dict[str, Any]:
        assert timeout == 90
        calls.append((source, action, request, request_digest))
        return {"state": "completed"}

    monkeypatch.setattr(adapter, "_worker_action", worker_action)

    result = adapter.AeonQwenFlashNextQualificationAdapter().probe(runtime)

    canonical = canonical_root / runtime_id
    assert result.state is adapter.ProbeState.COMPLETED
    assert calls == [
        (
            f"{canonical}/source",
            "status",
            f"{canonical}/qualification-request.json",
            digest,
        )
    ]


def test_local_staging_has_no_ssh_or_checkpoint_tree_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "worker.py"
    source.write_text("# exact\n", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(arguments: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((arguments, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)

    adapter._stage_local(source, "/private/scratch/source/worker.py", timeout=123)

    assert len(calls) == 1
    argv, options = calls[0]
    assert argv[:2] == [adapter.LOW_PRIORITY, "/usr/bin/rsync"]
    assert "-a" in argv and "-H" in argv
    assert not any("delete" in item for item in argv)
    assert "/usr/bin/ssh" not in argv
    assert not any(item.startswith("aday@") for item in argv)
    assert str(source) in argv
    assert worker.TUNED_CHECKPOINT_NAME not in argv
    assert worker.UNTUNED_CHECKPOINT_NAME not in argv
    assert options["cwd"] is None
    assert options["timeout"] == 123

    adapter_source = Path(adapter.__file__).read_text(encoding="utf-8")
    assert "/usr/bin/ssh" not in adapter_source
    assert "aday@" not in adapter_source
    assert "_rsync_build" not in adapter_source


def test_adapter_source_closure_includes_worker_harness_and_sampler() -> None:
    assert {
        "aeon/scripts/qwen_flash_next_qualification_worker.py",
        "aeon/scripts/qwen_flash_next_container_supervisor.py",
        "aeon/scripts/audit_qwen38_flash_next_passthrough.py",
        "aeon/scripts/qualify_qwen38_flash_next_endpoint.py",
        "aeon/scripts/release_qwen38_flash_next.py",
        "aeon/scripts/materialize_qwen38_flash_next_ple.py",
        "aeon/scripts/train_qwen38_flash_next_behavior.py",
        "aeon/core/qwen_flash_next_runtime_contract.py",
    } <= set(adapter.SOURCE_FILES)
    identity = adapter.expected_artifact_identity()
    assert set(identity) == {
        "adapter_source",
        "builder_source",
        "container_supervisor_source",
        "harness_source",
        "image",
        "image_archive",
        "image_config",
        "image_local_id",
        "materializer_source",
        "qualification_assets_manifest",
        "release_validator_source",
        "runtime_contract_source",
        "sglang_source_commit",
        "source_manifest",
        "worker_source",
    }
