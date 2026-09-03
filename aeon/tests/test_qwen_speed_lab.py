from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import types

import pytest
import torch

from aeon.core import qwen_speed_lab_adapter as adapter
from aeon.scripts import qwen_speed_lab_worker as worker


def _request(
    runtime_id: str = "fr-" + "a" * 32,
    host: str = adapter.HOST,
    variant: str | None = None,
) -> dict:
    scratch = f"/home/aday/.local/state/fleet-compute/runs/{runtime_id}"
    host_config = adapter.HOST_CONFIGS[host]
    variant = variant or adapter.VARIANT
    variant_config = adapter.VARIANT_CONFIGS[variant]
    draft = worker.EXPECTED_DRAFT_ARTIFACTS[variant_config["draft_id"]]
    model = worker.EXPECTED_MODEL_ARTIFACTS[variant_config["model_id"]]
    model_dir = (
        f"{scratch}/production-model"
        if model.get("stage_per_attempt") is True
        else model["model_dir"]
    )
    feature_capture = variant_config["feature_capture"]
    return {
        "schema_version": worker.SCHEMA_VERSION,
        "runtime_id": runtime_id,
        "job_id": "fj-speed-test",
        "host": host,
        "hostname": host_config["hostname"],
        "claim_id": "gc-speed-test-12345678",
        "owner": "aeon-speed-test-owner",
        "physical_gpu": 0,
        "gpu_uuid": "GPU-12345678-abcd-1234-abcd-123456789abc",
        "vram_budget_gb": variant_config["vram_budget_gb"],
        "exclusive": True,
        "feature_dataset_bytes": (
            adapter.FEATURE_DATASET.stat().st_size if feature_capture else None
        ),
        "feature_dataset_path": (
            f"{scratch}/feature-train.jsonl" if feature_capture else None
        ),
        "feature_dataset_rows": (
            adapter.FEATURE_DATASET_ROWS if feature_capture else None
        ),
        "feature_dataset_sha256": (
            adapter.FEATURE_DATASET_SHA256 if feature_capture else None
        ),
        "scratch_path": scratch,
        "source_root": f"{scratch}/source",
        "source_files": {"aeon/test.py": "b" * 64},
        "model_id": variant_config["model_id"],
        "model_dir": model_dir,
        "model_manifest_sha256": model["manifest_sha256"],
        "model_sha256s_sha256": model["sha256s_sha256"],
        "image_id": worker.EXPECTED_IMAGE_ID,
        "image_size_bytes": worker.EXPECTED_IMAGE_SIZE_BYTES,
        "engine_archive_sha256": worker.EXPECTED_ENGINE_ARCHIVE_SHA256,
        "engine_closure": (
            worker.EXPECTED_ENGINE_CLOSURE
            if host == worker.BARE_HOST
            else None
        ),
        "draft_id": variant_config["draft_id"],
        "draft_model_dir": draft["model_dir"],
        "draft_revision": draft["revision"],
        "draft_config_sha256": draft["config_sha256"],
        "draft_model_sha256": draft["model_sha256"],
        "container_name": f"aeon-speed-{runtime_id}",
        "port": 18033,
        "variant": variant,
        "runtime": {
            "attention_backend": variant_config["attention_backend"],
            "async_scheduling": False,
            "compilation_profile": variant_config["compilation_profile"],
            "context_tokens": variant_config["context_tokens"],
            "cuda_launch_blocking": False,
            "dspark_draft_topk": variant_config["dspark_draft_topk"],
            "enable_adaptive_verification": variant_config[
                "enable_adaptive_verification"
            ],
            "enable_flashinfer_autotune": adapter._host_runtime_value(
                variant_config, host_config, "enable_flashinfer_autotune"
            ),
            "enable_prefix_caching": variant_config["enable_prefix_caching"],
            "enable_per_request_metrics": variant_config[
                "enable_per_request_metrics"
            ],
            "feature_capture": feature_capture,
            "gdn_decode_kernel": "cuda",
            "gpu_memory_utilization": adapter._host_runtime_value(
                variant_config, host_config, "gpu_memory_utilization"
            ),
            "kv_cache_dtype": variant_config["kv_cache_dtype"],
            "local_argmax_reduction": (variant_config["speculative_method"] == "mtp"),
            "mamba_cache_dtype": variant_config["mamba_cache_dtype"],
            "mamba_cache_mode": "align",
            "mamba_ssm_cache_dtype": variant_config["mamba_ssm_cache_dtype"],
            "max_batched_tokens": adapter._host_runtime_value(
                variant_config, host_config, "max_batched_tokens"
            ),
            "max_num_seqs": 1,
            "model_runner": "v2",
            "nvfp4_a16": model["nvfp4_a16"],
            "relaxed_greedy_logit_margin": variant_config[
                "relaxed_greedy_logit_margin"
            ],
            "speculative_method": variant_config["speculative_method"],
            "speculative_tokens": variant_config["speculative_tokens"],
            "use_flashinfer_sampler": adapter._host_runtime_value(
                variant_config, host_config, "use_flashinfer_sampler"
            ),
        },
        "benchmark": {
            "max_tokens": 512,
            "quality_repeats": 2,
            "repeats": 5,
            "sampling_profile": "aeon-greedy-medium",
        },
    }


def test_enabled_profile_will_bind_current_sources_and_unchanged_prompt_bundle():
    profile_path = (
        Path(__file__).resolve().parents[3]
        / "fleet_compute/profiles.d/aeon-qwen38-speed-lab.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    sources = adapter._source_manifest()
    prompt, prompt_sources = adapter._prompt_bundle()
    assert "aeon/core/qwen_speed_lab_adapter.py" in sources
    assert len(prompt_sources) >= 5
    assert len(prompt) >= 4096
    assert profile["artifact_identity"] == {
        "bare_engine_archive": adapter.ENGINE_ARCHIVE_SHA256,
        "bare_engine_closure": adapter.ENGINE_CLOSURE_IDENTITY[
            "manifest_sha256"
        ],
        "bare_python_executable": adapter.ENGINE_CLOSURE_IDENTITY[
            "python_executable_sha256"
        ],
        "dflash2_bf16_config": adapter.DRAFT_ARTIFACTS["bf16"]["config_sha256"],
        "dflash2_bf16_model": adapter.DRAFT_ARTIFACTS["bf16"]["model_sha256"],
        "dflash2_bf16_revision": adapter.DRAFT_ARTIFACTS["bf16"]["revision_sha256"],
        "dflash2_aeonv1_config": adapter.DRAFT_ARTIFACTS["aeonv1"]["config_sha256"],
        "dflash2_aeonv1_model": adapter.DRAFT_ARTIFACTS["aeonv1"]["model_sha256"],
        "dflash2_aeonv1_revision": adapter.DRAFT_ARTIFACTS["aeonv1"]["revision_sha256"],
        "dflash2_aeonfullv1_config": adapter.DRAFT_ARTIFACTS["aeonfullv1"][
            "config_sha256"
        ],
        "dflash2_aeonfullv1_model": adapter.DRAFT_ARTIFACTS["aeonfullv1"][
            "model_sha256"
        ],
        "dflash2_aeonfullv1_revision": adapter.DRAFT_ARTIFACTS["aeonfullv1"][
            "revision_sha256"
        ],
        "dflash2_w4a16_config": adapter.DRAFT_ARTIFACTS["w4a16"]["config_sha256"],
        "dflash2_w4a16_model": adapter.DRAFT_ARTIFACTS["w4a16"]["model_sha256"],
        "dflash2_w4a16_revision": adapter.DRAFT_ARTIFACTS["w4a16"]["revision_sha256"],
        "dflash2_w8a16_config": adapter.DRAFT_ARTIFACTS["w8a16"]["config_sha256"],
        "dflash2_w8a16_model": adapter.DRAFT_ARTIFACTS["w8a16"]["model_sha256"],
        "dflash2_w8a16_revision": adapter.DRAFT_ARTIFACTS["w8a16"]["revision_sha256"],
        "dspark_nvfp4_config": adapter.DRAFT_ARTIFACTS["dspark-nvfp4"]["config_sha256"],
        "dspark_nvfp4_hf_quant_config": adapter.DRAFT_ARTIFACTS["dspark-nvfp4"][
            "hf_quant_config_sha256"
        ],
        "dspark_nvfp4_model": adapter.DRAFT_ARTIFACTS["dspark-nvfp4"]["model_sha256"],
        "dspark_nvfp4_revision": adapter.DRAFT_ARTIFACTS["dspark-nvfp4"][
            "revision_sha256"
        ],
        "feature_dataset": adapter.FEATURE_DATASET_SHA256,
        "image": adapter.IMAGE_ID.removeprefix("sha256:"),
        "model_w4a4_manifest": adapter.MODEL_ARTIFACTS["w4a4"]["manifest_sha256"],
        "model_w4a4_sha256s": adapter.MODEL_ARTIFACTS["w4a4"]["sha256s_sha256"],
        "model_w4a16_manifest": adapter.MODEL_ARTIFACTS["w4a16"]["manifest_sha256"],
        "model_w4a16_sha256s": adapter.MODEL_ARTIFACTS["w4a16"]["sha256s_sha256"],
        "model_fullgdn_manifest": adapter.MODEL_ARTIFACTS["fullgdn"][
            "manifest_sha256"
        ],
        "model_fullgdn_sha256s": adapter.MODEL_ARTIFACTS["fullgdn"][
            "sha256s_sha256"
        ],
        "prompt_bundle": hashlib.sha256(prompt).hexdigest(),
        "source_manifest": adapter._canonical_sha256(sources),
    }


def test_disabled_production_k3_canary_is_exact_and_179_only():
    profile_path = (
        Path(__file__).resolve().parents[3]
        / "fleet_compute/profiles.d/"
        "aeon-qwen38-production-k3-v026-canary-179.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    sources = adapter._source_manifest()
    prompt, _prompt_sources = adapter._prompt_bundle()
    variant = adapter.VARIANT_CONFIGS[adapter.PRODUCTION_K3_CANARY_VARIANT]
    assert profile["enabled"] is False
    assert profile["mode"] == "batch"
    assert profile["adapter"] == "aeon-qwen38-speed-lab-v1"
    assert profile["vram_budget_gb"] == 48.7
    assert profile["min_physical_vram_gb"] == 90.0
    assert profile["stage_bytes_max"] == 21_000_000_000
    assert profile["min_disk_free_gb"] == 72
    assert profile["placements"] == [
        {"host": "192.168.0.179", "enabled": True}
    ]
    assert profile["artifact_identity"] == {
        "bare_engine_archive": adapter.ENGINE_ARCHIVE_SHA256,
        "bare_engine_closure": adapter.ENGINE_CLOSURE_IDENTITY[
            "manifest_sha256"
        ],
        "bare_python_executable": adapter.ENGINE_CLOSURE_IDENTITY[
            "python_executable_sha256"
        ],
        "dflash2_bf16_config": adapter.DRAFT_ARTIFACTS["bf16"]["config_sha256"],
        "dflash2_bf16_model": adapter.DRAFT_ARTIFACTS["bf16"]["model_sha256"],
        "dflash2_bf16_revision": adapter.DRAFT_ARTIFACTS["bf16"][
            "revision_sha256"
        ],
        "image": adapter.IMAGE_ID.removeprefix("sha256:"),
        "model_production_manifest": adapter.MODEL_ARTIFACTS["production"][
            "manifest_sha256"
        ],
        "model_production_sha256s": adapter.MODEL_ARTIFACTS["production"][
            "sha256s_sha256"
        ],
        "prompt_bundle": hashlib.sha256(prompt).hexdigest(),
        "runtime_variant": adapter._canonical_sha256(variant),
        "source_manifest": adapter._canonical_sha256(sources),
    }
    raw = dict(profile)
    expected_manifest = raw.pop("manifest_sha256")
    assert expected_manifest == hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_production_k3_canary_profile_and_lease_are_variant_bound():
    profile_path = (
        Path(__file__).resolve().parents[3]
        / "fleet_compute/profiles.d/"
        "aeon-qwen38-production-k3-v026-canary-179.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    sources = adapter._source_manifest()
    prompt, _prompt_sources = adapter._prompt_bundle()
    variant = adapter.VARIANT_CONFIGS[adapter.PRODUCTION_K3_CANARY_VARIANT]
    run_dir = Path("/home/aday/.local/state/fleet-compute/runs/fr-" + "9" * 32)
    context = types.SimpleNamespace(
        profile=types.SimpleNamespace(
            profile_id=profile["profile_id"],
            artifact_identity=profile["artifact_identity"],
        ),
        lease=types.SimpleNamespace(
            host="192.168.0.179",
            memory_total_mib=97870,
            vram_budget_gb=48.7,
            exclusive=True,
            run_dir=run_dir,
        ),
        scratch_path=run_dir,
    )
    adapter.AeonQwenSpeedLabAdapter._profile_identity(
        context, sources, prompt, variant
    )
    context.lease.host = "192.168.0.180"
    with pytest.raises(adapter.QwenSpeedLabError, match="lease differs"):
        adapter.AeonQwenSpeedLabAdapter._profile_identity(
            context, sources, prompt, variant
        )


def test_payload_is_allowlisted_and_bounded():
    assert adapter.AeonQwenSpeedLabAdapter._payload({}) == {
        "variant": adapter.VARIANT,
        "repeats": 5,
        "quality_repeats": 2,
        "max_tokens": 512,
        "sampling_profile": "aeon-greedy-medium",
    }
    with pytest.raises(adapter.QwenSpeedLabError):
        adapter.AeonQwenSpeedLabAdapter._payload({"command": "python arbitrary.py"})
    with pytest.raises(adapter.QwenSpeedLabError):
        adapter.AeonQwenSpeedLabAdapter._payload({"repeats": 99})
    with pytest.raises(adapter.QwenSpeedLabError):
        adapter.AeonQwenSpeedLabAdapter._payload({"sampling_profile": "anything"})


def test_container_cleanup_reconciles_transient_exact_container_removal(monkeypatch):
    request = _request()
    container_id = "a" * 64
    stopped = {"State": {"Running": False}}
    inspections = iter((stopped, stopped, None))
    calls = []

    monkeypatch.setattr(
        worker, "_inspect_container", lambda _request, _container_id: next(inspections)
    )

    def fake_docker(_request, *args, **_kwargs):
        calls.append(args)
        if len(calls) == 1:
            return worker.subprocess.CompletedProcess(args, 1, "", "transient")
        return worker.subprocess.CompletedProcess(args, 0, container_id + "\n", "")

    monkeypatch.setattr(worker, "_docker", fake_docker)
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: None)
    assert worker._cleanup_container(request, container_id) is True
    assert calls == [("rm", container_id), ("rm", container_id)]


def test_worker_request_and_container_command_keep_every_gpu_binding(
    monkeypatch, tmp_path
):
    request = _request()
    local_request = tmp_path / "speed-lab-request.json"
    payload = (json.dumps(request, sort_keys=True) + "\n").encode()
    local_request.write_bytes(payload)
    local_request.chmod(0o600)
    monkeypatch.setattr(worker.socket, "gethostname", lambda: worker.EXPECTED_HOSTNAME)
    # Validate the exact same bytes while substituting only the local test inode.
    original_path = Path
    monkeypatch.setattr(
        worker,
        "Path",
        lambda value=".": (
            tmp_path if str(value) == request["scratch_path"] else original_path(value)
        ),
    )
    validated = worker._validate_request(
        local_request, hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(worker, "Path", original_path)
    command = worker._container_command(validated)
    gpu_index = command.index("--gpus")
    assert command[gpu_index + 1] == f"device={request['gpu_uuid']}"
    assert f"CUDA_VISIBLE_DEVICES={request['gpu_uuid']}" in command
    assert f"GPU_AGENT_CLAIM_ID={request['claim_id']}" in command
    assert "VLLM_GDN_DECODE_KERNEL=cuda" in command
    assert "CUDA_LAUNCH_BLOCKING=0" in command
    assert "VLLM_USE_V2_MODEL_RUNNER=1" in command
    assert "MTP_DRAFT_VOCAB=1" in command
    assert "AEON_NVFP4_A16=0" in command
    assert "AEON_RELAXED_GREEDY_LOGIT_MARGIN=0" in command
    assert f"AEON_SPEED_LAB_RUNTIME_ID={request['runtime_id']}" in command
    assert f"AEON_SPEED_LAB_MODEL_ID={request['model_id']}" in command
    assert f"AEON_SPEED_LAB_MODEL_SHA256S={request['model_sha256s_sha256']}" in command
    assert f"AEON_SPEED_LAB_DRAFT_SHA256={request['draft_model_sha256']}" in command
    assert f"AEON_SPEED_LAB_PORT={request['port']}" in command
    assert "VLLM_USE_FLASHINFER_SAMPLER=1" in command
    assert "PYTHONFAULTHANDLER=1" in command
    assert "PYTHONUNBUFFERED=1" in command
    assert "--no-async-scheduling" in command
    assert "--no-enable-flashinfer-autotune" in command
    assert "--enable-per-request-metrics" not in command
    assert command[command.index("--mamba-cache-mode") + 1] == "align"
    assert "--tensor-parallel-size" in command
    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert command[command.index("--kv-cache-dtype") + 1] == "auto"
    speculative = json.loads(command[command.index("--speculative-config") + 1])
    assert speculative == {
        "method": "dflash",
        "model": "/draft",
        "num_speculative_tokens": 7,
    }
    assert any(
        f"type=bind,src={request['draft_model_dir']},dst=/draft,readonly" == value
        for value in command
    )
    scripts_root = Path(request["source_root"]) / "aeon/scripts"
    assert (
        f"type=bind,src={scripts_root / 'speed_lab_sitecustomize/sitecustomize.py'},"
        "dst=/workspace/aeon_runtime/sitecustomize.py,readonly"
    ) in command
    assert (
        f"type=bind,src={scripts_root / 'vllm_uuid_sitecustomize.py'},"
        "dst=/workspace/aeon_runtime/vllm_uuid_sitecustomize.py,readonly"
    ) in command


def test_worker_rejects_runtime_variant_mutation(monkeypatch, tmp_path):
    request = _request()
    request["runtime"]["speculative_tokens"] = 3
    payload = (json.dumps(request, sort_keys=True) + "\n").encode()
    path = tmp_path / "speed-lab-request.json"
    path.write_bytes(payload)
    path.chmod(0o600)
    monkeypatch.setattr(worker.socket, "gethostname", lambda: worker.EXPECTED_HOSTNAME)
    original_path = Path
    monkeypatch.setattr(
        worker,
        "Path",
        lambda value=".": (
            tmp_path if str(value) == request["scratch_path"] else original_path(value)
        ),
    )
    with pytest.raises(worker.SpeedLabError, match="variant configuration"):
        worker._validate_request(path, hashlib.sha256(payload).hexdigest())


def test_worker_accepts_allowlisted_single_gpu_bringup_host(monkeypatch, tmp_path):
    host = "192.168.0.179"
    request = _request(host=host)
    payload = (json.dumps(request, sort_keys=True) + "\n").encode()
    path = tmp_path / "speed-lab-request.json"
    path.write_bytes(payload)
    path.chmod(0o600)
    monkeypatch.setattr(
        worker.socket,
        "gethostname",
        lambda: adapter.HOST_CONFIGS[host]["hostname"],
    )
    original_path = Path
    monkeypatch.setattr(
        worker,
        "Path",
        lambda value=".": (
            tmp_path if str(value) == request["scratch_path"] else original_path(value)
        ),
    )
    validated = worker._validate_request(path, hashlib.sha256(payload).hexdigest())
    assert validated["runtime"]["gpu_memory_utilization"] == 0.42


def test_worker_accepts_exact_production_k3_v026_canary(monkeypatch, tmp_path):
    host = "192.168.0.179"
    request = _request(
        host=host, variant=adapter.PRODUCTION_K3_CANARY_VARIANT
    )
    payload = (json.dumps(request, sort_keys=True) + "\n").encode()
    path = tmp_path / "speed-lab-request.json"
    path.write_bytes(payload)
    path.chmod(0o600)
    monkeypatch.setattr(
        worker.socket, "gethostname", lambda: adapter.HOST_CONFIGS[host]["hostname"]
    )
    original_path = Path
    monkeypatch.setattr(
        worker,
        "Path",
        lambda value=".": (
            tmp_path if str(value) == request["scratch_path"] else original_path(value)
        ),
    )
    validated = worker._validate_request(path, hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(worker, "Path", original_path)
    assert validated["vram_budget_gb"] == 48.7
    assert validated["model_id"] == "production"
    assert validated["model_manifest_sha256"] == (
        "1a3ba1eb88d0507bdef3798a6db59830dc076199b7db7d111201f6997588220e"
    )
    assert validated["model_sha256s_sha256"] == (
        "e7eca7ebee03c4f27482d4fe421ca1fac9f1d9986663a51fd7614361010c1237"
    )
    command = worker._bare_command(validated)
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.415"
    assert command[command.index("--max-model-len") + 1] == "114688"
    assert command[command.index("--max-num-batched-tokens") + 1] == "32768"
    assert command[command.index("--kv-cache-dtype") + 1] == (
        "fp8_per_token_head"
    )
    assert "--no-enable-flashinfer-autotune" in command
    assert "--enable-per-request-metrics" in command
    assert "--compilation-config" not in command
    speculative = json.loads(command[command.index("--speculative-config") + 1])
    assert speculative == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "use_local_argmax_reduction": True,
    }
    # The generic preflight hashes the BF16 draft closure, but native MTP does
    # not pass any external draft model to vLLM.
    assert "model" not in speculative


def test_worker_rejects_production_k3_canary_on_non_rtx6000_host(
    monkeypatch, tmp_path
):
    host = "192.168.0.180"
    request = _request(
        host=host, variant=adapter.PRODUCTION_K3_CANARY_VARIANT
    )
    payload = (json.dumps(request, sort_keys=True) + "\n").encode()
    path = tmp_path / "speed-lab-request.json"
    path.write_bytes(payload)
    path.chmod(0o600)
    monkeypatch.setattr(
        worker.socket, "gethostname", lambda: adapter.HOST_CONFIGS[host]["hostname"]
    )
    original_path = Path
    monkeypatch.setattr(
        worker,
        "Path",
        lambda value=".": (
            tmp_path if str(value) == request["scratch_path"] else original_path(value)
        ),
    )
    with pytest.raises(worker.SpeedLabError, match="lease contract"):
        worker._validate_request(path, hashlib.sha256(payload).hexdigest())


def test_production_model_staging_is_exact_bounded_and_never_deleting(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append((command, _kwargs))
        return adapter.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    heartbeats = []
    context = types.SimpleNamespace(
        heartbeat=lambda pid, detail: heartbeats.append((pid, detail))
    )
    scratch = "/home/aday/.local/state/fleet-compute/runs/fr-" + "8" * 32
    target = adapter.AeonQwenSpeedLabAdapter._stage_production_model(
        context,
        "192.168.0.179",
        "DAY2XRTX6000-2",
        scratch,
        adapter.MODEL_ARTIFACTS["production"],
    )
    assert target == f"{scratch}/production-model"
    assert len(calls) == 2
    transfer = calls[1][0]
    assert "/usr/bin/rsync" in transfer
    assert "--checksum" in transfer
    assert "--partial" in transfer
    assert "--delete" not in transfer
    assert transfer[-1] == f"aday@192.168.8.112:{target}/"
    assert heartbeats == [
        (None, "Qwen production model staging is active"),
        (None, "Qwen speed-lab artifact preflight"),
    ]


def test_bare_bringup_keeps_uuid_claim_cap_and_exact_engine_path(tmp_path):
    request = _request(host="192.168.0.179")
    request["scratch_path"] = str(tmp_path / "scratch")
    request["source_root"] = str(tmp_path / "scratch/source")
    environment = worker._bare_environment(request)
    command = worker._bare_command(request)
    assert environment["CUDA_VISIBLE_DEVICES"] == request["gpu_uuid"]
    assert environment["GPU_AGENT_CLAIM_ID"] == request["claim_id"]
    assert environment["GPU_PLANNED_VRAM_GB"] == "41.25"
    assert environment["CUDA_LAUNCH_BLOCKING"] == "0"
    assert environment["PYTHONFAULTHANDLER"] == "1"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["VLLM_RPC_BASE_PATH"].startswith("/dev/shm/aeon-vrpc-")
    assert len(environment["VLLM_RPC_BASE_PATH"]) < 64
    assert environment["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert environment["AEON_NVFP4_A16"] == "0"
    assert environment["AEON_RELAXED_GREEDY_LOGIT_MARGIN"] == "0"
    assert environment["AEON_SPEED_LAB_MODEL_ID"] == request["model_id"]
    assert (
        environment["AEON_SPEED_LAB_MODEL_SHA256S"] == request["model_sha256s_sha256"]
    )
    assert environment["AEON_SPEED_LAB_DRAFT_SHA256"] == request["draft_model_sha256"]
    assert environment["AEON_SPEED_LAB_PORT"] == str(request["port"])
    assert str(worker.ENGINE_SITE) in environment["PYTHONPATH"]
    assert any(request["draft_model_dir"] in value for value in command)
    assert str(worker.REMOTE_PYTHON) in command
    assert str(worker.DOCKER) not in command
    assert "--swap-space" not in command
    assert request["model_dir"] in command
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.42"
    assert "--enable-flashinfer-autotune" in command
    assert "--enable-per-request-metrics" not in command


def test_bare_startup_reports_exit_status_when_proc_identity_races(monkeypatch):
    request = _request(host="192.168.0.179")

    def unreadable(_request, _pid):
        raise worker.SpeedLabError("bare server identity is unreadable")

    monkeypatch.setattr(worker, "_bare_process_alive", unreadable)
    monkeypatch.setattr(worker, "_bare_exit_description", lambda _pid: "exit status 1")
    with pytest.raises(
        worker.SpeedLabError,
        match="bare speed server exited during startup: exit status 1",
    ):
        worker._wait_ready(request, "bare-12345", [False])


def test_bare_identity_survives_reviewed_process_title_rewrite(monkeypatch, tmp_path):
    request = _request(host="192.168.0.179")
    proc = tmp_path / "proc"
    proc.mkdir()
    environment = [
        f"GPU_AGENT_CLAIM_ID={request['claim_id']}",
        f"CUDA_VISIBLE_DEVICES={request['gpu_uuid']}",
        f"AEON_SPEED_LAB_RUNTIME_ID={request['runtime_id']}",
        f"AEON_SPEED_LAB_MODEL_ID={request['model_id']}",
        f"AEON_SPEED_LAB_MODEL_SHA256S={request['model_sha256s_sha256']}",
        f"AEON_SPEED_LAB_DRAFT_SHA256={request['draft_model_sha256']}",
        f"AEON_SPEED_LAB_PORT={request['port']}",
        "AEON_NVFP4_A16=0",
        "AEON_RELAXED_GREEDY_LOGIT_MARGIN=0",
        f"PYTHONPATH={worker.ENGINE_SITE}",
    ]
    (proc / "environ").write_bytes("\0".join(environment).encode() + b"\0")
    initial_command = [
        str(worker.REMOTE_PYTHON),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        request["model_dir"],
        "--port",
        str(request["port"]),
    ]
    (proc / "cmdline").write_bytes("\0".join(initial_command).encode() + b"\0")
    original_path = Path
    monkeypatch.setattr(
        worker,
        "Path",
        lambda value=".": proc if str(value) == "/proc/12345" else original_path(value),
    )
    assert worker._bare_spawn_identity(request, 12345) is True
    assert worker._bare_process_alive(request, 12345) is True

    (proc / "cmdline").write_bytes(b"VLLM::EngineCore\0")
    assert worker._bare_spawn_identity(request, 12345) is False
    assert worker._bare_process_alive(request, 12345) is True

    stat_fields = ["S", "999", "12345", "12345", *(["0"] * 15), "424242"]
    (proc / "stat").write_text(
        f"12345 (python3.12) {' '.join(stat_fields)}\n", encoding="ascii"
    )
    identity = worker._bare_process_stat(12345)
    assert identity is not None
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    request["scratch_path"] = str(scratch)
    state = {
        "schema_version": worker.SCHEMA_VERSION,
        "runtime_id": request["runtime_id"],
        "container_pid": 12345,
        **identity,
    }
    state_path = scratch / "worker-state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    state_path.chmod(0o600)
    (proc / "environ").write_bytes(b"process-title-overwrote-environ\0")
    assert worker._bare_process_alive(request, 12345) is True

    stat_fields[19] = "424243"
    (proc / "stat").write_text(
        f"12345 (python3.12) {' '.join(stat_fields)}\n", encoding="ascii"
    )
    assert worker._bare_process_alive(request, 12345) is False


def test_allowlisted_mtp_and_non_speculative_server_arguments():
    mtp_variant = "nightly-v2-fused-gdn-int8-heads-mtp-k4-bf16kv"
    mtp_command = worker._container_command(_request(variant=mtp_variant))
    speculative = json.loads(mtp_command[mtp_command.index("--speculative-config") + 1])
    assert speculative == {
        "method": "mtp",
        "num_speculative_tokens": 4,
        "use_local_argmax_reduction": True,
    }
    ar_request = _request(variant="nightly-v2-fused-gdn-int8-heads-ar-bf16kv")
    assert "--speculative-config" not in worker._container_command(ar_request)

    fp8_variant = "nightly-v2-fused-gdn-int8-heads-dflash2-w4a16-k6-triton-fp8kv"
    fp8_command = worker._container_command(_request(variant=fp8_variant))
    assert fp8_command[fp8_command.index("--kv-cache-dtype") + 1] == "fp8"

    blackwell_variant = "nightly-v2-fused-gdn-int8-heads-dflash2-k7-flashinfer-fp8kv"
    blackwell_request = _request(variant=blackwell_variant)
    blackwell_command = worker._container_command(blackwell_request)
    assert (
        blackwell_command[blackwell_command.index("--attention-backend") + 1]
        == "FLASHINFER"
    )
    assert blackwell_command[blackwell_command.index("--kv-cache-dtype") + 1] == "fp8"
    assert f"VLLM_DISABLED_KERNELS={worker.DISABLED_KERNELS}" in blackwell_command
    speculative = json.loads(
        blackwell_command[blackwell_command.index("--speculative-config") + 1]
    )
    assert speculative == {
        "method": "dflash",
        "model": "/draft",
        "num_speculative_tokens": 7,
    }
    assert any(
        f"type=bind,src={blackwell_request['draft_model_dir']},dst=/draft,readonly"
        == value
        for value in blackwell_command
    )

    dspark_variant = (
        "nightly-v2-fused-gdn-int8-heads-dspark-nvfp4-k7-topk64-flashinfer-fp8kv"
    )
    dspark_request = _request(variant=dspark_variant)
    dspark_command = worker._container_command(dspark_request)
    dspark_speculative = json.loads(
        dspark_command[dspark_command.index("--speculative-config") + 1]
    )
    assert dspark_request["draft_id"] == "dspark-nvfp4"
    assert dspark_speculative == {
        "dspark_draft_topk": 64,
        "enable_adaptive_verification": False,
        "method": "dspark",
        "model": "/draft",
        "num_speculative_tokens": 7,
    }

    fusion_variant = (
        "nightly-v2-fused-gdn-int8-heads-dflash2-k7-flashinfer-fp8kv-"
        "attnquant-partition"
    )
    fusion_request = _request(variant=fusion_variant)
    fusion_command = worker._container_command(fusion_request)
    compilation = json.loads(
        fusion_command[fusion_command.index("--compilation-config") + 1]
    )
    assert compilation == {
        "pass_config": {
            "enable_qk_norm_rope_fusion": False,
            "fuse_attn_quant": True,
        },
        "use_inductor_graph_partition": True,
    }

    fullgraph_variant = (
        "nightly-v2-fused-gdn-int8-heads-dflash2-k7-flashinfer-fp8kv-"
        "attnquant-fullgraph"
    )
    fullgraph_command = worker._container_command(_request(variant=fullgraph_variant))
    fullgraph_compilation = json.loads(
        fullgraph_command[fullgraph_command.index("--compilation-config") + 1]
    )
    assert fullgraph_compilation["splitting_ops"] == []
    assert fullgraph_compilation["pass_config"]["fuse_attn_quant"] is True

    piecewise_variant = (
        "nightly-v2-full-gdn-nvfp4-dflash2-k7-flashinfer-fp8kv-piecewise"
    )
    piecewise_request = _request(variant=piecewise_variant)
    piecewise_command = worker._container_command(piecewise_request)
    piecewise_compilation = json.loads(
        piecewise_command[piecewise_command.index("--compilation-config") + 1]
    )
    assert piecewise_request["runtime"]["compilation_profile"] == "piecewise"
    assert piecewise_compilation == {"cudagraph_mode": "PIECEWISE"}

    native_variant = (
        "nightly-v2-full-gdn-nvfp4-dflash2-k7-flashinfer-fp8kv-native-full"
    )
    native_request = _request(variant=native_variant)
    native_command = worker._container_command(native_request)
    native_attention = json.loads(
        native_command[native_command.index("--attention-config") + 1]
    )
    assert native_request["runtime"]["compilation_profile"] == (
        "flashinfer-native-full"
    )
    assert native_attention == {"use_trtllm_attention": False}
    assert "--compilation-config" not in native_command

    w8_variant = "nightly-v2-fused-gdn-int8-heads-dflash2-w8-k6-flashinfer-fp8kv"
    w8_request = _request(variant=w8_variant)
    w8_command = worker._container_command(w8_request)
    assert w8_request["draft_id"] == "w8a16"
    assert any(
        f"type=bind,src={w8_request['draft_model_dir']},dst=/draft,readonly" == value
        for value in w8_command
    )
    assert "AEON_NVFP4_A16=0" in w8_command
    assert "AEON_DSPARK_BF16_HEADS=0" in w8_command

    a16_variant = "nightly-v2-nvfp4a16-int8-heads-dflash2-w8-k6-flashinfer-fp8kv"
    a16_request = _request(variant=a16_variant)
    a16_command = worker._container_command(a16_request)
    assert a16_request["model_id"] == "w4a16"
    assert a16_request["runtime"]["nvfp4_a16"] is True
    assert any(
        f"type=bind,src={a16_request['model_dir']},dst=/models,readonly" == value
        for value in a16_command
    )
    assert "AEON_NVFP4_A16=1" in a16_command

    dspark_request = _request(
        variant=(
            "nightly-v2-fused-gdn-int8-heads-dspark-nvfp4-k7-flashinfer-fp8kv"
        )
    )
    dspark_command = worker._container_command(dspark_request)
    assert "AEON_DSPARK_BF16_HEADS=1" in dspark_command

    a16_triton_variant = "nightly-v2-nvfp4a16-int8-heads-dflash2-k6-triton-fp8kv"
    a16_triton_request = _request(variant=a16_triton_variant)
    a16_triton_command = worker._container_command(a16_triton_request)
    assert a16_triton_request["model_id"] == "w4a16"
    assert (
        a16_triton_command[a16_triton_command.index("--attention-backend") + 1]
        == "TRITON_ATTN"
    )
    assert "AEON_NVFP4_A16=1" in a16_triton_command


def test_adapter_and_worker_variant_contracts_match():
    assert adapter.VARIANT_CONFIGS == worker.VARIANT_CONFIGS
    assert adapter.WORKER_SCHEMA_VERSION == worker.SCHEMA_VERSION


def test_probe_retries_transport_failure_without_declaring_identity_unknown(monkeypatch):
    instance = adapter.AeonQwenSpeedLabAdapter()
    monkeypatch.setattr(
        instance,
        "_runtime_identity",
        lambda _runtime: ("fr-" + "a" * 32, "b" * 64, 12345),
    )

    def unavailable(*_args, **_kwargs):
        raise adapter.QwenSpeedLabTransportError("link unavailable")

    monkeypatch.setattr(instance, "_runtime_action", unavailable)
    with pytest.raises(adapter.QwenSpeedLabTransportError, match="link unavailable"):
        instance.probe({})


def test_packed_dflash_context_projection_preserves_layer_kv_order(monkeypatch):
    class FakeDFlashModel:
        def _build_context_kv_buffers(self, _layers_attn, _has_bias):
            raise AssertionError("dense path must not run for packed weights")

        def _project_context_kv(
            self,
            context_states,
            num_ctx,
            num_layers,
            num_kv_heads,
            head_dim,
        ):
            flat = torch.nn.functional.linear(
                context_states, self._fused_kv_weight, self._fused_kv_bias
            )
            all_kv = (
                flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )
            return all_kv[0], all_kv[1]

    class FakeProjection:
        input_size = 8
        bias = None

        def __init__(self, packed, scale):
            self.weight_packed = packed
            self.weight_scale = scale

    class FakeAttention:
        q_size = 4

        def __init__(self, packed, scale):
            self.qkv_proj = FakeProjection(packed, scale)
            self.k_norm = types.SimpleNamespace(weight=torch.ones(2))

    unpack_calls = []

    def fake_unpack(packed, num_bits, shape, packed_dim):
        unpack_calls.append((packed.clone(), num_bits, shape, packed_dim))
        assert num_bits == 4
        assert shape == torch.Size((4, 8))
        assert packed_dim == 1
        return packed.repeat(1, 8)

    fake_dflash = types.ModuleType("vllm.model_executor.models.qwen3_dflash")
    fake_dflash.DFlashQwen3Model = FakeDFlashModel
    fake_dflash.logger = types.SimpleNamespace(info=lambda *_args: None)
    fake_dspark = types.ModuleType("vllm.model_executor.models.qwen3_dspark")

    class FakeDSparkMarkovHead:
        calls = []

        def __init__(
            self,
            vocab_size,
            draft_vocab_size,
            markov_rank,
            prefix,
            quant_config=None,
        ):
            self.calls.append(
                (
                    vocab_size,
                    draft_vocab_size,
                    markov_rank,
                    prefix,
                    quant_config,
                )
            )

    fake_dspark.DSparkMarkovHead = FakeDSparkMarkovHead
    fake_models = types.ModuleType("vllm.model_executor.models")
    fake_models.qwen3_dflash = fake_dflash
    fake_models.qwen3_dspark = fake_dspark
    fake_executor = types.ModuleType("vllm.model_executor")
    fake_layers = types.ModuleType("vllm.model_executor.layers")
    fake_quantization = types.ModuleType("vllm.model_executor.layers.quantization")
    fake_base_config = types.ModuleType(
        "vllm.model_executor.layers.quantization.base_config"
    )
    fake_linear = types.ModuleType("vllm.model_executor.layers.linear")

    class FakeQuantizationConfig:
        _ignore_unexpected_suffixes = (".q_scale",)

    class FakeMergedColumnParallelLinear:
        def load_weights(self, weights):
            yield from (name for name, _tensor in weights)

    class FakeQKVParallelLinear:
        def load_weights(self, weights):
            yield from (name for name, _tensor in weights)

    fake_base_config.QuantizationConfig = FakeQuantizationConfig
    fake_linear.MergedColumnParallelLinear = FakeMergedColumnParallelLinear
    fake_linear.QKVParallelLinear = FakeQKVParallelLinear
    fake_vllm = types.ModuleType("vllm")
    fake_triton_utils = types.ModuleType("vllm.triton_utils")

    class FakeTriton:
        @staticmethod
        def jit(**_kwargs):
            return lambda function: function

    fake_triton_utils.triton = FakeTriton()
    fake_triton_utils.tl = types.SimpleNamespace()
    compressed = types.ModuleType("compressed_tensors")
    compressors = types.ModuleType("compressed_tensors.compressors")
    pack_quantized = types.ModuleType("compressed_tensors.compressors.pack_quantized")
    pack_base = types.ModuleType("compressed_tensors.compressors.pack_quantized.base")
    pack_base.unpack_from_int32 = fake_unpack
    monkeypatch.setitem(
        sys.modules, "vllm_uuid_sitecustomize", types.ModuleType("guard")
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", fake_triton_utils)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", fake_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", fake_layers)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.quantization", fake_quantization
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.base_config",
        fake_base_config,
    )
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", fake_linear)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", fake_models)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.qwen3_dflash", fake_dflash
    )
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.qwen3_dspark", fake_dspark
    )
    monkeypatch.setitem(sys.modules, "compressed_tensors", compressed)
    monkeypatch.setitem(sys.modules, "compressed_tensors.compressors", compressors)
    monkeypatch.setitem(
        sys.modules, "compressed_tensors.compressors.pack_quantized", pack_quantized
    )
    monkeypatch.setitem(
        sys.modules,
        "compressed_tensors.compressors.pack_quantized.base",
        pack_base,
    )
    monkeypatch.setenv("AEON_NVFP4_A16", "1")
    monkeypatch.setenv("AEON_DSPARK_BF16_HEADS", "1")

    sitecustomize = (
        Path(__file__).resolve().parents[1]
        / "scripts/speed_lab_sitecustomize/sitecustomize.py"
    )
    spec = importlib.util.spec_from_file_location(
        "aeon_speed_sitecustomize_test", sitecustomize
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert FakeQuantizationConfig._ignore_unexpected_suffixes == (
        ".q_scale",
        ".input_global_scale",
    )
    FakeDSparkMarkovHead(248320, 248320, 256, "model.markov_head", object())
    assert FakeDSparkMarkovHead.calls == [
        (248320, 248320, 256, "model.markov_head", None)
    ]
    for linear_cls in (FakeMergedColumnParallelLinear, FakeQKVParallelLinear):
        assert list(
            linear_cls().load_weights(
                [
                    ("weight_packed", object()),
                    ("input_global_scale", object()),
                    ("nested.input_global_scale", object()),
                ]
            )
        ) == ["weight_packed"]

    packed_weights = (
        torch.arange(1, 9, dtype=torch.int32).reshape(8, 1),
        torch.arange(11, 19, dtype=torch.int32).reshape(8, 1),
    )
    scales = (
        torch.full((8, 1), 0.5),
        torch.full((8, 1), 0.25),
    )
    model = FakeDFlashModel()
    model.hidden_norm = types.SimpleNamespace(weight=torch.ones(8))
    attentions = [
        FakeAttention(packed, scale)
        for packed, scale in zip(packed_weights, scales, strict=True)
    ]
    model._build_context_kv_buffers(attentions, False)
    values = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
    all_k, all_v = model._project_context_kv(values, 2, 2, 1, 2)

    dense_kv = [
        packed[4:].to(torch.float32).repeat(1, 8) * scale[4:]
        for packed, scale in zip(packed_weights, scales, strict=True)
    ]
    projected = [values @ weight.T for weight in dense_kv]
    assert torch.equal(all_k[:, :, 0, :], torch.stack([p[:, :2] for p in projected]))
    assert torch.equal(all_v[:, :, 0, :], torch.stack([p[:, 2:] for p in projected]))
    assert len(unpack_calls) == 2
    assert torch.equal(unpack_calls[0][0], packed_weights[0][4:])
    assert torch.equal(unpack_calls[1][0], packed_weights[1][4:])


def test_status_recognizes_supervisor_during_state_receipt_handoff(
    monkeypatch, tmp_path
):
    request = _request(host="192.168.0.179")
    request["scratch_path"] = str(tmp_path)
    request_path = tmp_path / "speed-lab-request.json"
    request_path.write_text("exact request bytes", encoding="utf-8")
    request_path.chmod(0o600)
    digest = hashlib.sha256(request_path.read_bytes()).hexdigest()
    spawn = tmp_path / "supervisor-spawn.json"
    spawn.write_text(
        json.dumps(
            {
                "schema_version": worker.SCHEMA_VERSION,
                "runtime_id": request["runtime_id"],
                "request_sha256": digest,
                "worker_pid": 12345,
                "created_at": 1.0,
            }
        ),
        encoding="utf-8",
    )
    spawn.chmod(0o600)
    monkeypatch.setattr(worker, "_process_alive", lambda _request, _pid: True)
    assert worker._status(request) == {
        "state": "running",
        "pid": 12345,
        "phase": "supervisor_spawning",
        "container_pid": None,
    }


def test_source_staging_forces_private_worker_permissions(monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return adapter.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(adapter.subprocess, "run", fake_run)
    adapter.AeonQwenSpeedLabAdapter._stage_sources(
        "192.168.0.179",
        "DAY2XRTX6000-2",
        "/home/aday/.local/state/fleet-compute/runs/fr-" + "b" * 32,
        {"aeon/__init__.py": "c" * 64},
    )
    assert len(calls) == 2
    assert "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=" in calls[1]


def test_prelaunch_failure_cleanup_is_exact_and_process_absent(monkeypatch, tmp_path):
    runtime_id = "fr-" + "d" * 32
    scratch = tmp_path / runtime_id
    scratch.mkdir(mode=0o700)
    request = {
        "runtime_id": runtime_id,
        "host": "192.168.0.179",
        "scratch_path": str(scratch),
        "source_root": f"{scratch}/source",
    }
    request_path = scratch / "speed-lab-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    request_path.chmod(0o600)
    runtime = {
        "runtime_id": runtime_id,
        "host": "192.168.0.179",
        "run_dir": str(scratch),
        "process_identity": None,
        "pid": None,
        "process_absent": 1,
    }
    storage = {
        "scratch_path": str(scratch),
        "canonical_output_path": str(tmp_path / "canonical"),
        "terminal_success": 0,
        "terminal_note": "storage preparation failed before launch: test",
    }
    monkeypatch.setattr(adapter, "REMOTE_RUN_ROOT", PurePosixPath(str(tmp_path)))
    cleaned = []
    monkeypatch.setattr(
        adapter.AeonQwenSpeedLabAdapter,
        "_cleanup_prelaunch_scratch",
        staticmethod(
            lambda host, path, digest, rid: (
                cleaned.append((host, path, digest, rid)) or 123
            )
        ),
    )
    result = adapter.AeonQwenSpeedLabAdapter().finalize_storage(runtime, storage)
    assert result.output_settled is True
    assert result.cleanup_complete is True
    assert result.reclaimed_bytes == 123
    assert cleaned[0][0:2] == ("192.168.0.179", str(scratch))
    assert cleaned[0][3] == runtime_id


def test_prelaunch_payload_rejection_can_settle_only_an_absent_scratch(
    monkeypatch, tmp_path
):
    runtime_id = "fr-" + "e" * 32
    scratch = tmp_path / runtime_id
    runtime = {
        "runtime_id": runtime_id,
        "host": "192.168.0.180",
        "run_dir": str(scratch),
        "process_identity": None,
        "pid": None,
        "process_absent": 1,
    }
    storage = {
        "scratch_path": str(scratch),
        "canonical_output_path": str(tmp_path / "canonical"),
        "terminal_success": 0,
        "terminal_note": "storage preparation failed before launch: rejected payload",
    }
    monkeypatch.setattr(adapter, "REMOTE_RUN_ROOT", PurePosixPath(str(tmp_path)))
    cleaned = []
    monkeypatch.setattr(
        adapter.AeonQwenSpeedLabAdapter,
        "_cleanup_prelaunch_scratch",
        staticmethod(
            lambda host, path, digest, rid: (
                cleaned.append((host, path, digest, rid)) or 0
            )
        ),
    )
    result = adapter.AeonQwenSpeedLabAdapter().finalize_storage(runtime, storage)
    assert result.output_settled is True
    assert result.cleanup_complete is True
    assert result.reclaimed_bytes == 0
    assert cleaned == [("192.168.0.180", str(scratch), None, runtime_id)]
