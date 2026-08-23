#!/usr/bin/env python3
"""Build an atomic Qwen3.8-27B abliterated NVFP4 checkpoint.

The selected BF16 source is already abliterated with a reproducible Heretic ARA
recipe.  This script applies W4A4 NVFP4 PTQ to language-model Linear modules,
keeps the vision tower and recurrent convolutions in BF16, and grafts the
source checkpoint's native MTP tensors back after Transformers serialization.

It is deliberately fleet-aware: one coordinator UUID, claim ID, and truthful
hard VRAM cap are required before importing or allocating CUDA model weights.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_REPO = "trohrbaugh/Qwen3.8-27B-heretic-ara"
SOURCE_REVISION = "a67ae100d933c0d17af3232bda35825979fc63ce"
OFFICIAL_REPO = "Qwen/Qwen3.8-27B"
OFFICIAL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CALIBRATION_REPO = "neuralmagic/calibration"
CALIBRATION_REVISION = "fb6bc2f8c66543876fb31613f5872b9030220e15"
CUDA_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
CLAIM_RE = re.compile(r"^gc-[0-9A-Za-z-]+$")
MTP_PREFIX = "mtp."
VISION_PREFIX = "model.visual."


def fail(message: str) -> "NoReturn":
    raise SystemExit(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def require_fleet_binding(torch: Any) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    claim = os.environ.get("GPU_AGENT_CLAIM_ID", "")
    if not CUDA_UUID_RE.fullmatch(visible):
        fail("CUDA_VISIBLE_DEVICES must be one coordinator-returned GPU UUID")
    if not CLAIM_RE.fullmatch(claim):
        fail("GPU_AGENT_CLAIM_ID must be an active coordinator claim")
    try:
        limit_gb = float(os.environ["GPU_MEM_LIMIT_GB"])
        reserve_gb = float(os.environ.get("GPU_RESERVE_GB", "6"))
    except (KeyError, ValueError):
        fail("GPU_MEM_LIMIT_GB and GPU_RESERVE_GB must be numeric")
    if limit_gb <= 0 or reserve_gb < 6:
        fail("GPU_MEM_LIMIT_GB must be positive and GPU_RESERVE_GB at least 6")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        fail("expected exactly one coordinator-pinned CUDA device")
    properties = torch.cuda.get_device_properties(0)
    total_gb = properties.total_memory / 1024**3
    if limit_gb + reserve_gb > total_gb + 0.05:
        fail(
            f"hard cap {limit_gb:g}GB plus reserve {reserve_gb:g}GB exceeds "
            f"the {total_gb:.2f}GB device"
        )
    torch.cuda.set_per_process_memory_fraction(limit_gb / total_gb, 0)
    return {
        "claim_id": claim,
        "gpu_uuid": visible,
        "gpu_name": properties.name,
        "gpu_total_gb": total_gb,
        "gpu_mem_limit_gb": limit_gb,
        "gpu_reserve_gb": reserve_gb,
    }


@contextlib.contextmanager
def transformers_pread_loader() -> Any:
    """Avoid Transformers 5 bulk mmap commit accounting on strict-overcommit hosts.

    Transformers 5 opens every safetensors shard before dispatching weights. Its
    default private mmap backend therefore reserves the checkpoint's full virtual
    size even when most parameters are destined for disk offload. The safetensors
    pread backend keeps slices lazy without reserving all six source mappings.
    """
    import accelerate.utils.offload as accelerate_offload
    import transformers.modeling_utils as modeling_utils

    original_safe_open = modeling_utils.safe_open
    original_accelerate_safe_open = accelerate_offload.safe_open

    def safe_open_pread(*args: Any, **kwargs: Any) -> Any:
        device = kwargs.get("device", args[2] if len(args) > 2 else "cpu")
        if str(device).startswith("cpu") and kwargs.get("backend") in (None, "mmap"):
            kwargs["backend"] = "pread"
        return original_safe_open(*args, **kwargs)

    modeling_utils.safe_open = safe_open_pread
    accelerate_offload.safe_open = safe_open_pread
    try:
        yield
    finally:
        modeling_utils.safe_open = original_safe_open
        accelerate_offload.safe_open = original_accelerate_safe_open


def validate_source(source: Path) -> tuple[dict[str, str], dict[str, str]]:
    required = (
        source / "config.json",
        source / "model.safetensors.index.json",
        source / "reproduce" / "reproduce.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        fail(f"source checkpoint is incomplete: {missing}")
    reproduction = json.loads(required[2].read_text())
    if reproduction.get("model") != OFFICIAL_REPO:
        fail("source reproduction manifest names the wrong official model")
    if reproduction.get("model_commit") != OFFICIAL_REVISION:
        fail("source reproduction manifest names the wrong official revision")
    expected = reproduction.get("weights_sha256")
    if not isinstance(expected, dict) or not expected:
        fail("source reproduction manifest has no weight hashes")
    actual_files = {
        path.name: path
        for path in source.glob("*.safetensors")
        if path.is_file()
    }
    if set(actual_files) != set(expected):
        fail(
            "source weight set mismatch: "
            f"expected={sorted(expected)} actual={sorted(actual_files)}"
        )
    actual: dict[str, str] = {}
    for name, path in sorted(actual_files.items()):
        digest = sha256_file(path)
        if digest != expected[name]:
            fail(f"source weight SHA-256 mismatch: {name}")
        actual[name] = digest

    index = json.loads(required[1].read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        fail("source model index has no weight map")
    missing_shards = sorted(set(weight_map.values()) - set(actual_files))
    if missing_shards:
        fail(f"source index references missing shards: {missing_shards}")
    mtp = sorted(key for key in weight_map if key.startswith(MTP_PREFIX))
    vision = sorted(key for key in weight_map if key.startswith(VISION_PREFIX))
    if len(mtp) != 15:
        fail(f"expected 15 native MTP tensors, found {len(mtp)}")
    if len(vision) != 333:
        fail(f"expected 333 vision tensors, found {len(vision)}")
    return actual, weight_map


def fetch_calibration_rows(count: int, destination: Path) -> list[dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "dataset": CALIBRATION_REPO,
            "config": "LLM",
            "split": "train",
            "offset": 0,
            "length": count,
        }
    )
    url = "https://datasets-server.huggingface.co/rows?" + params
    request = urllib.request.Request(url, headers={"User-Agent": "bc-aeon-ptq/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    rows = []
    for record in payload.get("rows", []):
        row = record.get("row", {})
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            rows.append({"text": text, "row_idx": str(record.get("row_idx"))})
    if len(rows) != count:
        fail(f"expected {count} calibration rows, received {len(rows)}")
    snapshot = {
        "schema_version": "qwen38-nvfp4-calibration-v1",
        "dataset": CALIBRATION_REPO,
        "dataset_revision": CALIBRATION_REVISION,
        "config": "LLM",
        "split": "train",
        "rows": rows,
    }
    destination.write_text(json.dumps(snapshot, indent=2) + "\n")
    return [{"text": row["text"]} for row in rows]


def tensor_file_for(index: dict[str, str], source: Path, name: str) -> Path:
    try:
        return source / index[name]
    except KeyError:
        fail(f"tensor is absent from source index: {name}")


def graft_mtp(
    source: Path,
    source_index: dict[str, str],
    output: Path,
    save_file: Any,
    safe_open: Any,
) -> list[str]:
    mtp_names = sorted(name for name in source_index if name.startswith(MTP_PREFIX))
    tensors = {}
    handles: dict[Path, Any] = {}
    try:
        for name in mtp_names:
            shard = tensor_file_for(source_index, source, name)
            if shard not in handles:
                handles[shard] = safe_open(shard, framework="pt", device="cpu")
            tensor = handles[shard].get_tensor(name)
            if str(tensor.dtype) != "torch.bfloat16":
                fail(f"native MTP tensor is not BF16: {name} ({tensor.dtype})")
            tensors[name] = tensor
        mtp_file = output / "model-mtp-bf16.safetensors"
        save_file(tensors, mtp_file, metadata={"format": "pt"})
    finally:
        handles.clear()

    index_path = output / "model.safetensors.index.json"
    if index_path.is_file():
        out_index = json.loads(index_path.read_text())
        weight_map = out_index.get("weight_map", {})
    else:
        candidates = sorted(
            path
            for path in output.glob("*.safetensors")
            if path.name != "model-mtp-bf16.safetensors"
        )
        if len(candidates) != 1:
            fail("cannot construct output model index from ambiguous safetensors")
        weight_map = {}
        with safe_open(candidates[0], framework="pt", device="cpu") as handle:
            for name in handle.keys():
                weight_map[name] = candidates[0].name
        out_index = {"metadata": {}, "weight_map": weight_map}
    for name in mtp_names:
        if name in weight_map:
            fail(f"output unexpectedly already contains MTP tensor: {name}")
        weight_map[name] = "model-mtp-bf16.safetensors"
    out_index.setdefault("metadata", {})["total_size"] = sum(
        path.stat().st_size for path in output.glob("*.safetensors")
    )
    index_path.write_text(json.dumps(out_index, indent=2, sort_keys=True) + "\n")

    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    if text_config.get("mtp_num_hidden_layers") != 1:
        fail("output config lost mtp_num_hidden_layers=1")
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        fail("output config has no quantization_config")
    ignores = list(quantization.get("ignore") or [])
    for name in mtp_names:
        module = name.removesuffix(".weight")
        if module not in ignores:
            ignores.append(module)
    quantization["ignore"] = sorted(set(ignores))
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return mtp_names


def load_tensor(
    root: Path, index: dict[str, str], name: str, safe_open: Any
) -> Any:
    shard = tensor_file_for(index, root, name)
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def validate_output(
    source: Path,
    source_index: dict[str, str],
    output: Path,
    safe_open: Any,
    torch: Any,
) -> dict[str, Any]:
    index_path = output / "model.safetensors.index.json"
    if not index_path.is_file():
        fail("output model index is missing")
    out_index = json.loads(index_path.read_text()).get("weight_map", {})
    if not isinstance(out_index, dict):
        fail("output model index is invalid")
    missing = sorted(set(source_index) - set(out_index))
    packed_sources = {
        f"{name.removesuffix('.weight_packed')}.weight"
        for name in out_index
        if name.endswith(".weight_packed")
    }
    if set(missing) != packed_sources:
        unexpected = sorted(set(missing) - packed_sources)
        orphaned = sorted(packed_sources - set(missing))
        fail(
            "output tensor replacement mismatch: "
            f"unexpected_missing={unexpected[:5]}, orphaned_packed={orphaned[:5]}"
        )
    compressed_suffixes = (
        ".input_global_scale",
        ".weight_global_scale",
        ".weight_packed",
        ".weight_scale",
    )
    incomplete = []
    for name in missing:
        stem = name.removesuffix(".weight")
        absent = [suffix for suffix in compressed_suffixes if f"{stem}{suffix}" not in out_index]
        if absent:
            incomplete.append((name, absent))
    if incomplete:
        fail(f"incomplete NVFP4 tensor replacements: {incomplete[:5]}")
    missing_shards = sorted(set(out_index.values()) - {p.name for p in output.glob("*.safetensors")})
    if missing_shards:
        fail(f"output index references missing shards: {missing_shards}")

    mtp_names = sorted(name for name in source_index if name.startswith(MTP_PREFIX))
    vision_names = sorted(name for name in source_index if name.startswith(VISION_PREFIX))
    for name in mtp_names:
        left = load_tensor(source, source_index, name, safe_open)
        right = load_tensor(output, out_index, name, safe_open)
        if left.dtype != torch.bfloat16 or not torch.equal(left, right):
            fail(f"MTP tensor was not preserved exactly: {name}")

    # Check every vision tensor. The tower is explicitly outside the NVFP4 recipe;
    # exact equality catches an accidentally broad regex or lossy re-save.
    for name in vision_names:
        left = load_tensor(source, source_index, name, safe_open)
        right = load_tensor(output, out_index, name, safe_open)
        if left.dtype != torch.bfloat16 or not torch.equal(left, right):
            fail(f"vision tensor was not preserved exactly: {name}")
    return {
        "source_tensor_count": len(source_index),
        "output_tensor_count": len(out_index),
        "nvfp4_replaced_weight_count": len(missing),
        "nvfp4_components_per_weight": list(compressed_suffixes),
        "mtp_tensor_count": len(mtp_names),
        "vision_tensor_count": len(vision_names),
        "mtp_exact": True,
        "vision_exact": True,
    }


def write_readme(output: Path) -> None:
    output.joinpath("README.md").write_text(
        """---
license: apache-2.0
base_model: trohrbaugh/Qwen3.8-27B-heretic-ara
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - qwen3_5
  - qwen3.8
  - abliterated
  - nvfp4
  - mtp
  - multimodal
  - compressed-tensors
---

# Qwen3.8-27B ARA Abliterated NVFP4 + MTP

Atomic W4A4 NVFP4 derivative of the reproducible Heretic ARA BF16 checkpoint.
The vision tower, recurrent convolutions, language head, and all 15 native MTP
tensors remain BF16. The MTP tensors were grafted from the hash-verified source
after Transformers serialization and verified bit-exact; all 333 vision tensors
were also verified bit-exact against the BF16 source.

This artifact is unvalidated until its text-capability, vision, refusal-surface,
MTP-acceptance, and clean-load gates pass. See `BUILD_MANIFEST.json` and
`SHA256SUMS` for exact provenance.
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offload-folder", type=Path, required=True)
    parser.add_argument("--max-cpu-memory-gb", type=int, default=32)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    args = parser.parse_args()
    if args.calibration_samples < 1:
        fail("calibration-samples must be positive")
    if args.max_cpu_memory_gb < 8:
        fail("max-cpu-memory-gb must be at least 8")
    source = args.source.resolve()
    output = args.output.resolve()
    offload_folder = args.offload_folder.resolve()
    if offload_folder == source or offload_folder == output:
        fail("offload-folder must be distinct from source and output")
    if output.exists():
        fail(f"final output already exists: {output}")
    partial = output.with_name(
        f"{output.name}.partial-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    )
    if partial.exists():
        fail(f"partial output already exists: {partial}")
    partial.mkdir(parents=True)
    offload_folder.mkdir(parents=True, exist_ok=True)
    if any(offload_folder.iterdir()):
        fail(f"offload-folder must start empty: {offload_folder}")

    print("Verifying the complete BF16 source before GPU loading...", flush=True)
    source_hashes, source_index = validate_source(source)
    calibration_path = partial / "calibration_snapshot.json"
    calibration_rows = fetch_calibration_rows(
        args.calibration_samples, calibration_path
    )

    # Heavy imports happen only after filesystem/source validation.
    import datasets
    import llmcompressor
    import safetensors
    import torch
    import transformers
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.utils.dev import load_context
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, AutoProcessor

    fleet = require_fleet_binding(torch)
    print(
        "Fleet binding: "
        f"{fleet['gpu_name']}, cap={fleet['gpu_mem_limit_gb']:g}GiB, "
        f"reserve={fleet['gpu_reserve_gb']:g}GiB, claim={fleet['claim_id']}",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(source, trust_remote_code=False)
    print(
        "Loading through compressed-tensors auto_offload with "
        f"a {args.max_cpu_memory_gb}GiB CPU ceiling and disk spillover...",
        flush=True,
    )
    with transformers_pread_loader(), load_context(AutoModelForImageTextToText):
        model = AutoModelForImageTextToText.from_pretrained(
            source,
            dtype=torch.bfloat16,
            device_map="auto_offload",
            max_memory={"cpu": f"{args.max_cpu_memory_gb}GiB"},
            offload_folder=offload_folder,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
    tokenized_rows = [
        processor.tokenizer(
            row["text"],
            padding=False,
            max_length=args.max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )
        for row in calibration_rows
    ]
    calibration = datasets.Dataset.from_list(tokenized_rows)
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=["lm_head", "re:.*visual.*", "re:.*conv1d.*", "re:.*mtp.*"],
    )
    print(
        f"Calibrating W4A4 NVFP4 with {len(calibration)} text rows at "
        f"{args.max_seq_length} tokens...",
        flush=True,
    )
    oneshot(
        model=model,
        dataset=calibration,
        recipe=recipe,
        pipeline="sequential",
        sequential_offload_device="cpu",
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(calibration),
    )
    with transformers_pread_loader():
        model.save_pretrained(
            partial,
            safe_serialization=True,
            max_shard_size="5GB",
            save_compressed=True,
            save_original_format=False,
        )
    processor.save_pretrained(partial)
    for name in ("chat_template.jinja", "LICENSE"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, partial / name)

    # Release CUDA allocations before CPU-side tensor graft/integrity checks.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    mtp_names = graft_mtp(source, source_index, partial, save_file, safe_open)
    integrity = validate_output(source, source_index, partial, safe_open, torch)
    write_readme(partial)

    versions = {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "llmcompressor": llmcompressor.__version__,
        "datasets": datasets.__version__,
        "safetensors": safetensors.__version__,
    }
    manifest = {
        "schema_version": "qwen38-ara-abliterated-nvfp4-build-v1",
        "complete": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "official_repo": OFFICIAL_REPO,
            "official_revision": OFFICIAL_REVISION,
            "weight_sha256": source_hashes,
            "reproduction_manifest_sha256": sha256_file(
                source / "reproduce" / "reproduce.json"
            ),
        },
        "abliteration": {
            "method": "Heretic 1.2.0+custom arbitrary-rank ablation",
            "source_reported_refusals": "0/100",
            "source_reported_kl_divergence": 0.05345083400607109,
            "source_manifest": "reproduce/reproduce.json",
        },
        "quantization": {
            "scheme": "NVFP4 W4A4 group-16 compressed-tensors",
            "targets": ["Linear"],
            "ignore": [
                "lm_head",
                "re:.*visual.*",
                "re:.*conv1d.*",
                "re:.*mtp.*",
            ],
            "calibration_repo": CALIBRATION_REPO,
            "calibration_revision": CALIBRATION_REVISION,
            "calibration_samples": args.calibration_samples,
            "max_seq_length": args.max_seq_length,
            "calibration_snapshot_sha256": sha256_file(calibration_path),
            "pipeline": "sequential",
            "offload": {
                "loader": "compressed-tensors auto_offload",
                "source_backend": "safetensors pread",
                "max_cpu_memory_gb": args.max_cpu_memory_gb,
                "folder": str(offload_folder),
            },
            "save_original_format": False,
        },
        "fleet": fleet,
        "integrity": integrity,
        "mtp_tensors": mtp_names,
        "versions": versions,
        "build_script_sha256": sha256_file(Path(__file__).resolve()),
        "status": "unvalidated",
    }
    manifest_path = partial / "BUILD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    hash_lines = []
    for path in sorted(
        item
        for item in partial.iterdir()
        if item.is_file() and item.name != "SHA256SUMS"
    ):
        hash_lines.append(f"{sha256_file(path)}  {path.name}")
    partial.joinpath("SHA256SUMS").write_text("\n".join(hash_lines) + "\n")
    fsync_tree(partial)
    partial.rename(output)
    parent_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    print(f"Published unvalidated atomic candidate: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
