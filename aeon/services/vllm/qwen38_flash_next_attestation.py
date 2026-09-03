"""Post-load placement evidence for Aeon's Qwen3.8 Flash-Next canary.

This module runs inside the exact vLLM engine processes.  It emits independent
GPU-model and PLE CPU-worker fragments; the Fleet worker validates and merges
them only after the API is semantically ready.  Requested CLI flags alone are
never treated as tensor-placement evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.layers.ple_offload_layer import PleOffloadLayer


GPU_SCHEMA = "aeon-qwen38-flash-next-vllm-gpu-fragment-v1"
PLE_SCHEMA = "aeon-qwen38-flash-next-vllm-ple-fragment-v1"
GPU_OUTPUT = Path("/evidence/engine-gpu-fragment.json")
PLE_OUTPUT = Path("/evidence/engine-ple-fragment.json")
CONTEXT = Path("/evidence/runtime-context.json")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME = re.compile(r"^fr-[0-9a-f]{32}$")
_CONTAINER = re.compile(r"^[0-9a-f]{64}$")
_GPU = re.compile(r"^GPU-[0-9A-Fa-f-]{32,64}$")


class AttestationError(RuntimeError):
    pass


def _sha_text(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def _require_environment() -> dict[str, Any] | None:
    if os.environ.get("AEON_ENGINE_ATTESTATION") != "1":
        return None
    runtime_id = os.environ.get("AEON_RUNTIME_ID", "")
    arm = os.environ.get("AEON_CANARY_ARM", "")
    checkpoint = os.environ.get("AEON_CHECKPOINT_MANIFEST_SHA256", "")
    image_config = os.environ.get("AEON_DERIVED_IMAGE_CONFIG_DIGEST", "")
    claim = os.environ.get("GPU_AGENT_CLAIM_ID", "")
    gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    claim_hash = os.environ.get("AEON_LEASE_CLAIM_SHA256", "")
    gpu_hash = os.environ.get("AEON_LEASE_GPU_UUID_SHA256", "")
    mtp_raw = os.environ.get("AEON_MTP_ENABLED", "")
    if (
        _RUNTIME.fullmatch(runtime_id) is None
        or arm not in {"mtp_off", "mtp_on"}
        or _SHA.fullmatch(checkpoint) is None
        or _SHA.fullmatch(image_config) is None
        or not claim
        or _GPU.fullmatch(gpu_uuid) is None
        or _SHA.fullmatch(claim_hash) is None
        or claim_hash != hashlib.sha256(claim.encode()).hexdigest()
        or _SHA.fullmatch(gpu_hash) is None
        or gpu_hash != hashlib.sha256(gpu_uuid.encode()).hexdigest()
        or mtp_raw not in {"0", "1"}
    ):
        raise AttestationError("Aeon engine attestation environment is malformed")
    return {
        "runtime_id": runtime_id,
        "arm": arm,
        "mtp_enabled": mtp_raw == "1",
        "checkpoint_manifest_sha256": checkpoint,
        "lease_claim_id_sha256": claim_hash,
        "leased_gpu_uuid_sha256": gpu_hash,
        "derived_image_config_digest": image_config,
    }


def _private_context() -> dict[str, Any]:
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        try:
            metadata = CONTEXT.lstat()
            raw = CONTEXT.read_text(encoding="utf-8")
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        except (OSError, UnicodeDecodeError) as exc:
            raise AttestationError("runtime context is unreadable") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or not 0 < metadata.st_size <= 65536
        ):
            raise AttestationError("runtime context is not private and bounded")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AttestationError("runtime context is malformed") from exc
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "container_id",
                "container_pid",
                "cgroup_path",
                "container_pid_in_cgroup",
            }
            or _CONTAINER.fullmatch(str(value.get("container_id") or "")) is None
            or type(value.get("container_pid")) is not int
            or value["container_pid"] <= 1
            or value.get("container_pid_in_cgroup") is not True
            or not str(value.get("cgroup_path") or "").startswith("/sys/fs/cgroup/")
        ):
            raise AttestationError("runtime context closure changed")
        return value
    raise AttestationError("runtime context was not published before attestation")


def _atomic_private(path: Path, value: dict[str, Any]) -> None:
    parent = path.parent.resolve(strict=True)
    metadata = parent.lstat()
    if (
        parent != Path("/evidence")
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or path.exists()
        or path.is_symlink()
    ):
        raise AttestationError("attestation output boundary is unsafe")
    temporary = parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AttestationError("attestation write was incomplete")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _common(schema: str) -> dict[str, Any] | None:
    binding = _require_environment()
    if binding is None:
        return None
    context = _private_context()
    return {
        "schema_version": schema,
        **binding,
        "container_id": context["container_id"],
        "container_pid": context["container_pid"],
        "emitter_pid": os.getpid(),
        "emitted_after_model_load": True,
    }


def _category_for(name: str) -> str:
    fields = set(name.casefold().split("."))
    if "lm_head" in fields:
        return "lm_head"
    if fields.intersection({"visual", "vision", "vision_model", "vision_tower"}):
        return "vision"
    return "transformer"


def _empty_category() -> dict[str, Any]:
    return {
        "parameters": [],
        "persistent_buffers": [],
        "numel_references": 0,
        "devices": set(),
    }


def _record_tensor(category: dict[str, Any], kind: str, name: str, tensor: torch.Tensor) -> None:
    category[kind].append(name)
    category["numel_references"] += int(tensor.numel())
    category["devices"].add(str(tensor.device))


def _model_tensor_evidence(model: nn.Module, *, prefix: str, mtp: bool) -> tuple[dict[str, Any], list[str], list[str]]:
    categories = {name: _empty_category() for name in ("transformer", "mtp", "lm_head", "vision")}
    ple_names = [name for name, module in model.named_modules() if isinstance(module, PleOffloadLayer)]
    ple_prefixes = tuple(f"{name}." for name in ple_names if name)
    unexpected_parameters: list[str] = []
    unexpected_buffers: list[str] = []
    for module_name, module in model.named_modules():
        if module_name in ple_names or (ple_prefixes and module_name.startswith(ple_prefixes)):
            continue
        for local_name, tensor in module._parameters.items():
            if tensor is None:
                continue
            name = ".".join(value for value in (prefix, module_name, local_name) if value)
            category_name = "mtp" if mtp else _category_for(name)
            _record_tensor(categories[category_name], "parameters", name, tensor)
            if tensor.device.type != "cuda":
                unexpected_parameters.append(f"{name}:{tensor.device}")
        for local_name, tensor in module._buffers.items():
            if tensor is None or local_name in module._non_persistent_buffers_set:
                continue
            name = ".".join(value for value in (prefix, module_name, local_name) if value)
            category_name = "mtp" if mtp else _category_for(name)
            _record_tensor(categories[category_name], "persistent_buffers", name, tensor)
            if tensor.device.type != "cuda":
                unexpected_buffers.append(f"{name}:{tensor.device}")
    return categories, ple_names, sorted(unexpected_parameters + unexpected_buffers)


def _merge_categories(target: dict[str, Any], source: dict[str, Any]) -> None:
    for name in target:
        target[name]["parameters"].extend(source[name]["parameters"])
        target[name]["persistent_buffers"].extend(source[name]["persistent_buffers"])
        target[name]["numel_references"] += source[name]["numel_references"]
        target[name]["devices"].update(source[name]["devices"])


def _finalize_categories(categories: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in categories.items():
        names = sorted(value["parameters"] + value["persistent_buffers"])
        result[name] = {
            "parameter_references": len(value["parameters"]),
            "persistent_buffer_references": len(value["persistent_buffers"]),
            "numel_references": value["numel_references"],
            "devices": sorted(value["devices"]),
            "names_sha256": _sha_text(names),
        }
    return result


def _runtime(worker: Any) -> tuple[dict[str, Any], dict[str, str]]:
    config = worker.vllm_config
    spec = config.speculative_config
    reasoning = getattr(config.structured_outputs_config, "reasoning_parser", "")
    mtp = (
        None
        if spec is None
        else {
            "method": str(spec.method),
            "num_speculative_tokens": int(spec.num_speculative_tokens),
            "quantization": str(spec.quantization),
            "moe_backend": str(spec.moe_backend),
        }
    )
    runtime = {
        "provider": "vllm",
        "tensor_parallel_size": int(config.parallel_config.tensor_parallel_size),
        "distributed_executor_backend": str(config.parallel_config.distributed_executor_backend),
        "gpu_memory_utilization": float(config.cache_config.gpu_memory_utilization),
        "kv_cache_memory_bytes": config.cache_config.kv_cache_memory_bytes,
        "max_model_len": int(config.model_config.max_model_len),
        "max_num_seqs": int(config.scheduler_config.max_num_seqs),
        "max_num_batched_tokens": int(config.scheduler_config.max_num_batched_tokens),
        "kv_cache_dtype": str(config.cache_config.cache_dtype),
        "quantization": str(config.model_config.quantization),
        "moe_backend": str(config.kernel_config.moe_backend),
        "enable_prefix_caching": bool(config.cache_config.enable_prefix_caching),
        "enable_chunked_prefill": bool(config.scheduler_config.enable_chunked_prefill),
        "enable_flashinfer_autotune": bool(
            config.kernel_config.enable_flashinfer_autotune
        ),
        "cudagraph_capture_sizes": list(
            config.compilation_config.cudagraph_capture_sizes or []
        ),
        "speculative_config": mtp,
        "enable_auto_tool_choice": True,
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": reasoning,
        "ple_cpu_offload": bool(worker._ple_offload_enabled),
        "ple_fp8_checkpoint": os.environ.get("VLLM_PLE_FP8_CHECKPOINT") == "1",
        "ple_offload_ready_timeout_seconds": int(
            os.environ.get("VLLM_PLE_OFFLOAD_READY_TIMEOUT", "0")
        ),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "pytorch_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "checkpoint_repository": os.environ.get("AEON_CHECKPOINT_REPOSITORY"),
        "checkpoint_revision": os.environ.get("AEON_CHECKPOINT_REVISION"),
        "base_image_amd64_digest": os.environ.get("AEON_BASE_IMAGE_AMD64_DIGEST"),
        "served_model": os.environ.get("AEON_SERVED_MODEL"),
        "host": os.environ.get("AEON_CANARY_HOST"),
        "physical_gpu": int(os.environ.get("AEON_CANARY_PHYSICAL_GPU", "-1")),
        "exclusive_lease": os.environ.get("AEON_CANARY_EXCLUSIVE") == "1",
        "vram_cap_gib": float(os.environ.get("GPU_MEM_LIMIT_GB", "nan")),
    }
    provenance = {
        key: "engine_vllm_config"
        for key in (
            "tensor_parallel_size",
            "distributed_executor_backend",
            "gpu_memory_utilization",
            "kv_cache_memory_bytes",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "kv_cache_dtype",
            "quantization",
            "moe_backend",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "enable_flashinfer_autotune",
            "cudagraph_capture_sizes",
            "speculative_config",
            "reasoning_parser",
            "ple_cpu_offload",
        )
    }
    for key in set(runtime).difference(provenance):
        provenance[key] = "verified_docker"
    for key in (
        "ple_fp8_checkpoint", "ple_offload_ready_timeout_seconds",
        "torch_cuda_arch_list", "pytorch_alloc_conf",
    ):
        provenance[key] = "verified_env"
    for key in tuple(provenance):
        if provenance[key] == "engine_vllm_config":
            provenance[key] = "engine_native"
    return runtime, dict(sorted(provenance.items()))


def emit_gpu_fragment(worker: Any) -> None:
    common = _common(GPU_SCHEMA)
    if common is None:
        return
    if int(worker.vllm_config.parallel_config.tensor_parallel_size) != 1:
        raise AttestationError("Aeon placement attestation supports only TP=1")
    target = worker.model_runner.get_model()
    if not isinstance(target, nn.Module):
        raise AttestationError("loaded target model is not an nn.Module")
    categories, ple_names, unexpected = _model_tensor_evidence(
        target, prefix="target", mtp=False
    )
    mtp_enabled = bool(common["mtp_enabled"])
    draft_model = worker.model_runner.get_draft_model()
    if mtp_enabled:
        if not isinstance(draft_model, nn.Module):
            raise AttestationError("MTP is enabled but no loaded draft model exists")
        draft_categories, draft_ple, draft_unexpected = _model_tensor_evidence(
            draft_model, prefix="mtp", mtp=True
        )
        if draft_ple:
            raise AttestationError("MTP draft unexpectedly contains PLE offload layers")
        _merge_categories(categories, draft_categories)
        unexpected.extend(draft_unexpected)
    elif isinstance(draft_model, nn.Module):
        raise AttestationError("MTP-off control unexpectedly loaded a draft model")
    finalized = _finalize_categories(categories)
    required = ("transformer", "lm_head", "vision") + (("mtp",) if mtp_enabled else ())
    if (
        unexpected
        or not ple_names
        or any(finalized[name]["parameter_references"] <= 0 for name in required)
        or any(finalized[name]["devices"] != ["cuda:0"] for name in required)
        or (not mtp_enabled and finalized["mtp"]["parameter_references"] != 0)
    ):
        raise AttestationError("loaded GPU model placement did not meet the exact contract")
    runtime, provenance = _runtime(worker)
    if (runtime["speculative_config"] is not None) != mtp_enabled:
        raise AttestationError("live speculative configuration differs from the arm")
    placement = {
        "categories": finalized,
        "ple_placeholder_layer_count": len(ple_names),
        "ple_placeholder_names_sha256": _sha_text(ple_names),
        "unexpected_cpu_parameters": [value for value in unexpected if value.endswith(":cpu")],
        "unexpected_meta_parameters": [value for value in unexpected if value.endswith(":meta")],
        "unexpected_non_cuda_parameters": unexpected,
        "unexpected_cpu_persistent_buffers": [],
        "unexpected_meta_persistent_buffers": [],
        "unexpected_non_cuda_persistent_buffers": [],
    }
    _atomic_private(
        GPU_OUTPUT,
        {**common, "runtime": runtime, "runtime_provenance": provenance, "placement": placement},
    )


def emit_ple_fragment(runner: Any) -> None:
    common = _common(PLE_SCHEMA)
    if common is None:
        return
    layers = runner._layers
    if not layers or any(not isinstance(layer, PleOffloadLayer) for layer in layers.values()):
        raise AttestationError("PLE worker retained an unexpected module")
    parameter_names: list[str] = []
    buffer_names: list[str] = []
    fp8_names: list[str] = []
    bf16_names: list[str] = []
    scale_names: list[str] = []
    unexpected_model: list[str] = []
    numel = 0
    fp8_numel = 0
    bf16_numel = 0
    devices: set[str] = set()
    for layer_name, layer in layers.items():
        for name, tensor in layer.named_parameters():
            full_name = f"{layer_name}.{name}"
            parameter_names.append(full_name)
            numel += int(tensor.numel())
            devices.add(str(tensor.device))
            if tensor.device.type != "cpu" or tensor.is_meta:
                unexpected_model.append(f"{full_name}:{tensor.device}")
            if tensor.dtype == torch.float8_e4m3fn and name.endswith("ngram_embedding.weight"):
                fp8_names.append(full_name)
                fp8_numel += int(tensor.numel())
            if tensor.dtype == torch.bfloat16 and name.endswith("ngram_embedding.weight"):
                bf16_names.append(full_name)
                bf16_numel += int(tensor.numel())
        for module_name, module in layer.named_modules():
            for name, tensor in module._buffers.items():
                if tensor is None or name in module._non_persistent_buffers_set:
                    continue
                relative = ".".join(value for value in (module_name, name) if value)
                full_name = f"{layer_name}.{relative}"
                buffer_names.append(full_name)
                numel += int(tensor.numel())
                devices.add(str(tensor.device))
                if tensor.device.type != "cpu" or tensor.is_meta:
                    unexpected_model.append(f"{full_name}:{tensor.device}")
                if name.endswith("weight_scale"):
                    scale_names.append(full_name)
    pinned = [tensor for by_layer in runner._pinned_bufs.values() for tensor in by_layer.values()]
    unpinned = [f"{index}:{tensor.device}" for index, tensor in enumerate(pinned) if tensor.device.type != "cpu" or not tensor.is_pinned()]
    targets = [target for by_layer in runner._worker_targets.values() for values in by_layer.values() for target in values]
    bad_targets = [f"{index}:{target.gpu_output_buffer.device}" for index, target in enumerate(targets) if target.gpu_output_buffer.device.type != "cuda"]
    if (
        unexpected_model
        or unpinned
        or bad_targets
        or devices != {"cpu"}
        or not bf16_names
        or fp8_names
        or scale_names
        or not pinned
        or not targets
    ):
        raise AttestationError("PLE CPU worker placement did not meet the exact contract")
    placement = {
        "ple_layer_count": len(layers),
        "ple_layer_names_sha256": _sha_text(list(layers)),
        "parameter_references": len(parameter_names),
        "persistent_buffer_references": len(buffer_names),
        "numel_references": numel,
        "devices": sorted(devices),
        "bf16_table_references": len(bf16_names),
        "bf16_table_numel": bf16_numel,
        "bf16_table_names_sha256": _sha_text(bf16_names),
        "fp8_table_references": len(fp8_names),
        "fp8_table_numel": fp8_numel,
        "fp8_table_names_sha256": _sha_text(fp8_names),
        "scale_references": len(scale_names),
        "scale_names_sha256": _sha_text(scale_names),
        "pinned_h2d_buffer_count": len(pinned),
        "pinned_h2d_bytes": sum(tensor.numel() * tensor.element_size() for tensor in pinned),
        "pinned_h2d_devices": sorted({str(tensor.device) for tensor in pinned}),
        "registered_cuda_output_target_count": len(targets),
        "registered_cuda_output_target_devices": sorted({str(target.gpu_output_buffer.device) for target in targets}),
        "non_ple_retained_modules": [],
        "unexpected_non_cpu_model_tensors": unexpected_model,
        "unexpected_unpinned_h2d_buffers": unpinned,
        "unexpected_non_cuda_output_targets": bad_targets,
    }
    _atomic_private(PLE_OUTPUT, {**common, "placement": placement})
