#!/usr/bin/env python3
"""Fail-closed .179 lifecycle for the Qwen3.8-Flash-Next build batch."""

from __future__ import annotations

import base64
import csv
import email
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Any
import zipfile

from aeon.scripts import assemble_qwen38_flash_next_hybrid as assembler


SCHEMA_VERSION = "aeon-qwen38-flash-next-build-worker-v1"
RESULT_SCHEMA = "aeon-qwen38-flash-next-build-result-v1"
HOST = "192.168.0.179"
HOSTNAME = "DAY2XRTX6000-2"
SCRATCH_ROOT = PurePosixPath("/home/aday/.local/state/fleet-compute/runs")
LOCAL_HOST = "192.168.0.177"
LOCAL_HOSTNAME = "DAY2RTX6000PRO"
LOCAL_CANONICAL_ROOT = PurePosixPath(
    "/home/aday/.local/state/fleet-compute/artifacts/aeon-qwen38-flash-next-build"
)
LOCAL_SOURCE_ROOT = Path("/home/aday/NexusAgentDashboard/bc_aeon")
LOCAL_BF16_ROOT = Path(
    "/home/aday/.local/state/aeon-flash-next/sources/qwen-bf16-metadata"
)
LOCAL_FP8_ROOT = Path(
    "/home/aday/.aeon/models/.sources/"
    "Qwen3.8-Flash-Next-FP8-bcd9f01ddc9cff2316eb84281bebcd5b058bddce"
)
LOCAL_STATE = Path("/home/aday/.local/state/aeon-flash-next")
LOCAL_MODELOPT = Path(
    "/home/aday/.local/state/aeon-qwen38-quant/sources/"
    "nvidia_modelopt-0.46.0-py3-none-any.whl"
)
LOCAL_MODELOPT_RUNTIME = Path(
    "/home/aday/.local/state/aeon-qwen38-quant/sources/"
    "modelopt-0.46.0-runtime-cp312-linux-x86_64"
)
LOCAL_MODELOPT_RUNTIME_MANIFEST = LOCAL_MODELOPT_RUNTIME / "MANIFEST.json"
LOCAL_RESUME_MANIFEST = LOCAL_MODELOPT_RUNTIME / "FR_F373_RESUME.json"
LOCAL_TRANSFORMERS = LOCAL_STATE / "sources/transformers-5.16.1-py3-none-any.whl"
LOCAL_TOKENIZERS = LOCAL_STATE / (
    "sources/tokenizers-0.23.1-cp310-abi3-manylinux_2_17_x86_64."
    "manylinux2014_x86_64.whl"
)
LOCAL_MTP = LOCAL_STATE / "official-mtp/mtp-bf16.safetensors"
LOCAL_MTP_MANIFEST = LOCAL_STATE / "official-mtp/mtp-bf16.manifest.json"
LOCAL_SCALES = LOCAL_STATE / "calibration/radixark-modelopt-expert-scales.safetensors"
LOCAL_SCALES_MANIFEST = (
    LOCAL_STATE / "calibration/radixark-modelopt-expert-scales.manifest.json"
)
LOCAL_BF16_FILES = LOCAL_STATE / "sources/qwen-bf16-files.json"
LOCAL_FP8_FILES = LOCAL_STATE / "sources/qwen-fp8-files.json"
LOCAL_TRAIN = LOCAL_SOURCE_ROOT / "aeon/behavioral_sft/data/train.jsonl"
LOCAL_EVAL = LOCAL_SOURCE_ROOT / "aeon/behavioral_sft/data/eval.jsonl"
LOW_PRIORITY = Path("/home/aday/bin/fleet-low-priority")
BASH = Path("/usr/bin/bash")
# This is an immutable, historically named environment directory.  Its label is
# not treated as a version receipt; _verify_environment binds every reviewed
# distribution before the staged pipeline can become launchable.
ENV_PYTHON = Path(
    "/home/aday/.aeon/runtime/qwen38/training-envs/"
    "nemo-9fb92970-torch291-cu128/bin/python"
)
ENV_SITE = Path(
    "/home/aday/.aeon/runtime/qwen38/training-envs/"
    "nemo-9fb92970-torch291-cu128/lib/python3.12/site-packages"
)
MODELOPT_SHA256 = "1864b4e9921e287b065be3861ab48345144e673273ebb2b94bd9a6119a9eba8e"
MODELOPT_RUNTIME_MANIFEST_SHA256 = (
    "7fb09995ca0fad2789a88d60079d0a2aa44a4a5c95e7330b9dbbd49ddf2a1f79"
)
RESUME_MANIFEST_SHA256 = (
    "6141858c2abd92bd7176ed2a44f5457de73113885419ca42b7b0ad1e1b079252"
)
FULL_RECIPE = "behavior-r4-expert-nvfp4-v1"
RESUME_RECIPE = "quant-only-resume-fr-f373-v1"
RESUME_RUNTIME_ID = "fr-f373582e1dc84651b212ab76f6b05436"
RESUME_ROOT = Path(LOCAL_CANONICAL_ROOT) / RESUME_RUNTIME_ID
RESUME_POST_STAGE_DISK_FLOOR_BYTES = 190_000_000_000
FULL_LOCAL_POST_STAGE_DISK_FLOOR_BYTES = 240_000_000_000
REMOTE_POST_STAGE_DISK_FLOOR_BYTES = 190_000_000_000
TRANSFORMERS_SHA256 = "2f2d5b98a5ad3718713653734298fa620754ed683702a635ebb587df3ed29c7e"
TOKENIZERS_SHA256 = "5075b405006415ea148a992d093699c66eb01952bf59f4d5727089a98bda45a4"
SGLANG_COMMIT = "dac5523d1e5d2f4297fec40ef02fc76fb0f662d1"
SGLANG_IMAGE_DIGEST = "a9d6f66c2f7309cd435abeca0baccde7c512b15cb7180b3d89334b22a5e01eb7"
SGLANG_IMAGE_REFERENCE = (
    "aeon/sglang:qwen38-flash-next-sm120-dac5523@sha256:" + SGLANG_IMAGE_DIGEST
)
_SHA = re.compile(r"^[a-f0-9]{64}$")
_RUNTIME = re.compile(r"^fr-[a-f0-9]{32}$")
_CLAIM = re.compile(r"^gc-[A-Za-z0-9._:-]{8,200}$")
_OWNER = re.compile(r"^[A-Za-z0-9._:-]{3,240}$")
_UUID = re.compile(r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$")
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
SPAWN_LOG_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_APPEND
REVIEWED_DISTRIBUTIONS = {
    "PyYAML": "6.0.3",
    "accelerate": "1.12.0",
    "annotated-types": "0.7.0",
    "antlr4-python3-runtime": "4.9.3",
    "huggingface-hub": "1.28.0",
    "ninja": "1.13.0",
    "numpy": "2.5.2",
    "nvidia-ml-py": "13.610.43",
    "nvidia-modelopt": "0.46.0",
    "omegaconf": "2.3.1",
    "packaging": "26.3",
    "peft": "0.19.1",
    "psutil": "7.2.2",
    "pulp": "3.3.2",
    "pydantic": "2.13.4",
    "pydantic-core": "2.46.4",
    "regex": "2026.7.19",
    "rich": "15.0.0",
    "safetensors": "0.8.0",
    "scipy": "1.18.1",
    "setuptools": "84.0.0",
    "tokenizers": "0.23.1",
    "torch": "2.10.0+cu130",
    "tqdm": "4.70.0",
    "transformers": "5.16.1",
    "typing-extensions": "4.16.0",
    "typing-inspection": "0.4.2",
}
NVFP4_QTENSOR_IMPORT = "modelopt.torch.quantization.qtensor.nvfp4_tensor.NVFP4QTensor"


class FlashBuildWorkerError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_dir(path: Path, *, create: bool = False) -> Path:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise FlashBuildWorkerError(f"private directory identity changed: {path}")
    return path


def _read_json(path: Path, maximum: int = 64 * 1024 * 1024) -> dict[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= maximum
    ):
        raise FlashBuildWorkerError(f"private JSON identity changed: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FlashBuildWorkerError(f"private JSON is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise FlashBuildWorkerError("private JSON is not an object")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    _private_dir(path.parent, create=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        payload = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FlashBuildWorkerError("private JSON write was incomplete")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _paths(request: dict[str, Any]) -> dict[str, Path]:
    scratch = Path(request["scratch_path"])
    fixtures = scratch / "fixtures"
    output = scratch / "output"
    local = request.get("host") == LOCAL_HOST
    resume = request.get("recipe") == RESUME_RECIPE
    return {
        "scratch": scratch,
        "source": Path(request["source_root"]),
        "output": output,
        "bf16": LOCAL_BF16_ROOT if local else scratch / "inputs/bf16",
        "fp8": LOCAL_FP8_ROOT if local else scratch / "inputs/fp8-ple",
        "fixtures": fixtures,
        "mtp": LOCAL_MTP if local else fixtures / "mtp-bf16.safetensors",
        "mtp_manifest": (
            LOCAL_MTP_MANIFEST if local else fixtures / "mtp-bf16.manifest.json"
        ),
        "scales": LOCAL_SCALES if local else fixtures / "expert-scales.safetensors",
        "scales_manifest": (
            LOCAL_SCALES_MANIFEST if local else fixtures / "expert-scales.manifest.json"
        ),
        "modelopt": (
            LOCAL_MODELOPT
            if local
            else fixtures / "nvidia_modelopt-0.46.0-py3-none-any.whl"
        ),
        "transformers": (
            LOCAL_TRANSFORMERS
            if local
            else fixtures / "transformers-5.16.1-py3-none-any.whl"
        ),
        "tokenizers": (
            LOCAL_TOKENIZERS
            if local
            else fixtures / "tokenizers-0.23.1-cp310-abi3-manylinux_2_17_x86_64."
            "manylinux2014_x86_64.whl"
        ),
        "modelopt_runtime_manifest": (
            LOCAL_MODELOPT_RUNTIME_MANIFEST
            if local
            else fixtures / "modelopt-runtime-manifest.json"
        ),
        "resume_manifest": (
            LOCAL_RESUME_MANIFEST if local else fixtures / "fr-f373-resume.json"
        ),
        "bf16_files": LOCAL_BF16_FILES if local else fixtures / "qwen-bf16-files.json",
        "fp8_files": LOCAL_FP8_FILES if local else fixtures / "qwen-fp8-files.json",
        "train": LOCAL_TRAIN if local else fixtures / "behavior-train.jsonl",
        "eval": LOCAL_EVAL if local else fixtures / "behavior-eval.jsonl",
        "overlay": scratch / "python-overlay",
        "source_manifest": scratch / "trainer-source-manifest.json",
        "hybrid": RESUME_ROOT / "hybrid" if resume else scratch / "hybrid",
        "behavior": (
            RESUME_ROOT / "behavior-adapter" if resume else scratch / "behavior-adapter"
        ),
        "behavior_receipt": (
            RESUME_ROOT / "behavior-receipt.json"
            if resume
            else scratch / "behavior-receipt.json"
        ),
        "request": scratch / "qwen-flash-next-build-request.json",
        "preflight": output / "preflight.json",
        "model": output / "model",
        "official_untuned_model": output / "official-untuned-model",
        "sibling_manifest": output / "BUILD_SIBLING_MANIFEST.json",
        "result": output / "result.json",
        "build_result": output / "builder-result.json",
        "qualification": output / "qualification-required.json",
        "log": output / "build.log",
        "spawn": scratch / "spawn.json",
        "manifest": output / "MANIFEST.sha256",
        "settled": scratch / "settled.json",
    }


def _runtime_wheel_paths(request: dict[str, Any]) -> dict[str, Path]:
    root = (
        LOCAL_MODELOPT_RUNTIME
        if request.get("host") == LOCAL_HOST
        else _paths(request)["fixtures"]
    )
    manifest = _read_json(_paths(request)["modelopt_runtime_manifest"])
    wheels = manifest.get("runtime_wheels")
    if not isinstance(wheels, dict):
        raise FlashBuildWorkerError("ModelOpt runtime wheel manifest is malformed")
    return {str(name): root / str(name) for name in wheels}


def _validate_request(path: Path, expected: str) -> dict[str, Any]:
    if _SHA.fullmatch(expected) is None or _sha256(path) != expected:
        raise FlashBuildWorkerError("request digest changed")
    request = _read_json(path)
    required = {
        "schema_version",
        "runtime_id",
        "job_id",
        "host",
        "hostname",
        "claim_id",
        "owner",
        "physical_gpu",
        "gpu_uuid",
        "vram_budget_gb",
        "exclusive",
        "min_host_memory_gb",
        "min_host_commit_gb",
        "post_stage_disk_floor_bytes",
        "min_shm_free_gb",
        "scratch_path",
        "source_root",
        "source_files",
        "input_files",
        "fixture_files",
        "recipe",
        "resume_source_manifest_sha256",
        "sglang_commit",
        "sglang_image_digest",
    }
    runtime = str(request.get("runtime_id") or "")
    scratch = PurePosixPath(str(request.get("scratch_path") or ""))
    local = request.get("host") == LOCAL_HOST
    expected_scratch_root = LOCAL_CANONICAL_ROOT if local else SCRATCH_ROOT
    expected_source_root = (
        PurePosixPath(LOCAL_SOURCE_ROOT) if local else scratch / "source"
    )
    expected_hostname = LOCAL_HOSTNAME if local else HOSTNAME
    recipe = request.get("recipe")
    expected_post_stage_disk_floor = (
        RESUME_POST_STAGE_DISK_FLOOR_BYTES
        if recipe == RESUME_RECIPE
        else (
            FULL_LOCAL_POST_STAGE_DISK_FLOOR_BYTES
            if local
            else REMOTE_POST_STAGE_DISK_FLOOR_BYTES
        )
    )
    if (
        set(request) != required
        or request.get("schema_version") != SCHEMA_VERSION
        or _RUNTIME.fullmatch(runtime) is None
        or request.get("host") not in {HOST, LOCAL_HOST}
        or scratch.parent != expected_scratch_root
        or scratch.name != runtime
        or PurePosixPath(str(request.get("source_root") or "")) != expected_source_root
        or path != Path(scratch) / "qwen-flash-next-build-request.json"
        or request.get("hostname") != expected_hostname
        or socket.gethostname() != expected_hostname
        or not isinstance(request.get("job_id"), str)
        or not request["job_id"]
        or _CLAIM.fullmatch(str(request.get("claim_id") or "")) is None
        or _OWNER.fullmatch(str(request.get("owner") or "")) is None
        or request.get("physical_gpu") not in ({0} if local else {0, 1})
        or isinstance(request.get("physical_gpu"), bool)
        or _UUID.fullmatch(str(request.get("gpu_uuid") or "")) is None
        or request.get("exclusive") is not True
        or float(request.get("vram_budget_gb") or 0) != 88.0
        or float(request.get("min_host_memory_gb") or 0) != 170.0
        or float(request.get("min_host_commit_gb") or 0) != 162.0
        or request.get("post_stage_disk_floor_bytes") != expected_post_stage_disk_floor
        or float(request.get("min_shm_free_gb") or 0) != 16.0
        or request.get("sglang_commit") != SGLANG_COMMIT
        or request.get("sglang_image_digest") != SGLANG_IMAGE_DIGEST
        or recipe not in {FULL_RECIPE, RESUME_RECIPE}
        or (
            request.get("recipe") == FULL_RECIPE
            and request.get("resume_source_manifest_sha256") is not None
        )
        or (
            request.get("recipe") == RESUME_RECIPE
            and (
                not local
                or request.get("resume_source_manifest_sha256")
                != RESUME_MANIFEST_SHA256
                or scratch == RESUME_ROOT
            )
        )
        or not isinstance(request.get("source_files"), dict)
        or not isinstance(request.get("input_files"), dict)
        or not isinstance(request.get("fixture_files"), dict)
    ):
        raise FlashBuildWorkerError("build request differs from its reviewed schema")
    return request


def _verify_local_manifests(
    request: dict[str, Any], paths: dict[str, Path]
) -> tuple[int, int, int]:
    source_bytes = 0
    for relative, receipt in sorted(request["source_files"].items()):
        if _SAFE_RELATIVE.fullmatch(relative) is None or not isinstance(receipt, dict):
            raise FlashBuildWorkerError("canonical source manifest is unsafe")
        source_bytes += _verify_regular(
            paths["source"] / relative, receipt, 4 * 1024**2, private=False
        )

    input_bytes = 0
    for relative, receipt in sorted(request["input_files"].items()):
        if not isinstance(receipt, dict):
            raise FlashBuildWorkerError("canonical input manifest is unsafe")
        prefix, separator, name = relative.partition("/")
        if (
            separator != "/"
            or PurePosixPath(name).name != name
            or prefix not in {"bf16", "fp8-ple"}
        ):
            raise FlashBuildWorkerError("canonical input path is unsafe")
        root = paths["bf16"] if prefix == "bf16" else paths["fp8"]
        input_bytes += _verify_regular(root / name, receipt, 32 * 1024**3)

    fixture_paths = {
        "mtp-bf16.safetensors": paths["mtp"],
        "mtp-bf16.manifest.json": paths["mtp_manifest"],
        "expert-scales.safetensors": paths["scales"],
        "expert-scales.manifest.json": paths["scales_manifest"],
        "nvidia_modelopt-0.46.0-py3-none-any.whl": paths["modelopt"],
        "transformers-5.16.1-py3-none-any.whl": paths["transformers"],
        "tokenizers-0.23.1-cp310-abi3-manylinux_2_17_x86_64."
        "manylinux2014_x86_64.whl": paths["tokenizers"],
        "qwen-bf16-files.json": paths["bf16_files"],
        "qwen-fp8-files.json": paths["fp8_files"],
        "behavior-train.jsonl": paths["train"],
        "behavior-eval.jsonl": paths["eval"],
        "modelopt-runtime-manifest.json": paths["modelopt_runtime_manifest"],
        "fr-f373-resume.json": paths["resume_manifest"],
    }
    fixture_paths.update(_runtime_wheel_paths(request))
    if set(request["fixture_files"]) != set(fixture_paths):
        raise FlashBuildWorkerError("canonical fixture manifest changed")
    fixture_bytes = sum(
        _verify_regular(
            fixture_paths[name],
            request["fixture_files"][name],
            32 * 1024**3,
            private=name not in {"behavior-train.jsonl", "behavior-eval.jsonl"},
        )
        for name in sorted(fixture_paths)
    )
    return source_bytes, input_bytes, fixture_bytes


def _verify_regular(
    path: Path, receipt: dict[str, Any], maximum: int, *, private: bool = True
) -> int:
    metadata = path.lstat()
    digest, size = receipt.get("sha256"), receipt.get("size")
    if (
        _SHA.fullmatch(str(digest or "")) is None
        or type(size) is not int
        or size <= 0
        or size > maximum
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & (0o077 if private else 0o022)
        or metadata.st_size != size
        or _sha256(path) != digest
    ):
        raise FlashBuildWorkerError(f"staged file identity changed: {path}")
    return size


def _verify_manifested(root: Path, values: dict[str, Any], maximum: int) -> int:
    _private_dir(root)
    total = 0
    for relative, receipt in sorted(values.items()):
        if _SAFE_RELATIVE.fullmatch(relative) is None or not isinstance(receipt, dict):
            raise FlashBuildWorkerError("staged manifest contains an unsafe path")
        total += _verify_regular(root / relative, receipt, maximum)
    return total


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _wheel_files(
    path: Path, expected: dict[str, Any] | None
) -> dict[str, tuple[str, int]]:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise FlashBuildWorkerError("reviewed overlay wheel is malformed") from exc
    with archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise FlashBuildWorkerError("wheel contains duplicate members")
        files: dict[str, tuple[str, int]] = {}
        metadata_names: list[str] = []
        record_names: list[str] = []
        for member in infos:
            name = member.filename
            target = PurePosixPath(name)
            mode = member.external_attr >> 16
            if (
                not name
                or target.is_absolute()
                or ".." in target.parts
                or "\\" in name
                or member.flag_bits & 0x1
                or stat.S_ISLNK(mode)
                or (
                    mode != 0
                    and not (
                        (member.is_dir() and stat.S_IFMT(mode) in {0, stat.S_IFDIR})
                        or (
                            not member.is_dir()
                            and stat.S_IFMT(mode) in {0, stat.S_IFREG}
                        )
                    )
                )
            ):
                raise FlashBuildWorkerError("wheel contains an unsafe member")
            if member.is_dir():
                continue
            if name.count("/") == 1 and name.endswith(".dist-info/METADATA"):
                metadata_names.append(name)
            if name.count("/") == 1 and name.endswith(".dist-info/RECORD"):
                record_names.append(name)
            raw = archive.read(member)
            files[name] = (
                hashlib.sha256(raw).hexdigest(),
                0o700 if mode & 0o111 else 0o600,
            )
        if len(metadata_names) != 1 or len(record_names) != 1:
            raise FlashBuildWorkerError("wheel metadata topology changed")
        raw_metadata = archive.read(metadata_names[0])
        raw_record = archive.read(record_names[0])
        message = email.message_from_bytes(raw_metadata)
        requires = message.get_all("Requires-Dist", [])
        try:
            rows = list(csv.reader(io.StringIO(raw_record.decode("utf-8"))))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise FlashBuildWorkerError("wheel RECORD is malformed") from exc
        if {row[0] for row in rows if len(row) == 3} != set(files) or any(
            len(row) != 3 for row in rows
        ):
            raise FlashBuildWorkerError("wheel RECORD does not close")
        for name, digest, size in rows:
            raw = archive.read(name)
            if name == record_names[0]:
                if digest or size:
                    raise FlashBuildWorkerError("wheel RECORD self-entry changed")
                continue
            try:
                algorithm, encoded = digest.split("=", 1)
                actual = (
                    base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
                    .rstrip(b"=")
                    .decode()
                )
            except (ValueError, UnicodeDecodeError) as exc:
                raise FlashBuildWorkerError("wheel RECORD digest is malformed") from exc
            if algorithm != "sha256" or encoded != actual or int(size) != len(raw):
                raise FlashBuildWorkerError("wheel RECORD payload changed")
        if expected is not None:
            required = {
                "distribution",
                "version",
                "requires_python",
                "size",
                "sha256",
                "metadata_sha256",
                "requires_dist_sha256",
                "record_sha256",
                "member_count",
                "member_names_sha256",
            }
            if (
                set(expected) != required
                or message.get("Name") != expected["distribution"]
                or message.get("Version") != expected["version"]
                or message.get("Requires-Python") != expected["requires_python"]
                or hashlib.sha256(raw_metadata).hexdigest()
                != expected["metadata_sha256"]
                or _canonical_sha(requires) != expected["requires_dist_sha256"]
                or hashlib.sha256(raw_record).hexdigest() != expected["record_sha256"]
                or len(files) != expected["member_count"]
                or _canonical_sha(sorted(files)) != expected["member_names_sha256"]
            ):
                raise FlashBuildWorkerError("wheel package metadata changed")
    return files


def _overlay_tree_digest(root: Path, expected: dict[str, tuple[str, int]]) -> str:
    _private_dir(root)
    actual_files: set[str] = set()
    receipt_name = ".aeon-overlay-receipt.json"
    values: dict[str, dict[str, Any]] = {}
    for item in root.rglob("*"):
        metadata = item.lstat()
        relative = item.relative_to(root).as_posix()
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise FlashBuildWorkerError("overlay directory identity changed")
            continue
        if relative == receipt_name:
            continue
        required = expected.get(relative)
        if (
            required is None
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != required[1]
            or _sha256(item) != required[0]
        ):
            raise FlashBuildWorkerError("overlay payload identity changed")
        actual_files.add(relative)
        values[relative] = {"sha256": required[0], "mode": required[1]}
    if actual_files != set(expected):
        raise FlashBuildWorkerError("overlay file closure changed")
    return _canonical_sha(values)


def _extract_overlay(request: dict[str, Any]) -> None:
    paths = _paths(request)
    manifest_path = paths["modelopt_runtime_manifest"]
    if _sha256(manifest_path) != MODELOPT_RUNTIME_MANIFEST_SHA256:
        raise FlashBuildWorkerError("ModelOpt runtime manifest identity changed")
    manifest = _read_json(manifest_path)
    runtime_wheels = manifest.get("runtime_wheels")
    if (
        manifest.get("schema_version")
        != "aeon-qwen38-modelopt-runtime-wheel-closure-v1"
        or manifest.get("complete") is not True
        or manifest.get("target")
        != {
            "implementation": "CPython",
            "python": "3.12",
            "platform": "Linux",
            "machine": "x86_64",
        }
        or not isinstance(runtime_wheels, dict)
        or len(runtime_wheels) != 12
    ):
        raise FlashBuildWorkerError("ModelOpt runtime manifest contract changed")
    reviewed = (
        (paths["modelopt"], MODELOPT_SHA256, None),
        (paths["transformers"], TRANSFORMERS_SHA256, None),
        (paths["tokenizers"], TOKENIZERS_SHA256, None),
    )
    wheel_paths = _runtime_wheel_paths(request)
    expected_files: dict[str, tuple[str, int]] = {}
    for path, digest, expected in (
        *reviewed,
        *(
            (wheel_paths[name], str(spec["sha256"]), spec)
            for name, spec in sorted(runtime_wheels.items())
        ),
    ):
        size = path.lstat().st_size
        receipt = {"sha256": digest, "size": size}
        if expected is not None:
            receipt["size"] = expected["size"]
        _verify_regular(path, receipt, 64 * 1024**2)
        for name, value in _wheel_files(path, expected).items():
            if name in expected_files and expected_files[name] != value:
                raise FlashBuildWorkerError("overlay wheel member collision")
            if name in expected_files:
                raise FlashBuildWorkerError("overlay wheel member is duplicated")
            expected_files[name] = value
    expected_digest = _canonical_sha(
        {
            name: {"sha256": value[0], "mode": value[1]}
            for name, value in sorted(expected_files.items())
        }
    )
    if paths["overlay"].exists():
        receipt = _read_json(paths["overlay"] / ".aeon-overlay-receipt.json")
        if (
            receipt
            != {
                "schema_version": "aeon-qwen38-modelopt-python-overlay-v1",
                "manifest_sha256": MODELOPT_RUNTIME_MANIFEST_SHA256,
                "tree_sha256": expected_digest,
                "file_count": len(expected_files),
            }
            or _overlay_tree_digest(paths["overlay"], expected_files) != expected_digest
        ):
            raise FlashBuildWorkerError("existing Python overlay receipt changed")
        return
    temporary = paths["scratch"] / ".python-overlay.partial"
    if temporary.exists() or temporary.is_symlink():
        raise FlashBuildWorkerError("stale Python overlay partial exists")
    _private_dir(temporary, create=True)
    for path, _digest, expected in (
        *reviewed,
        *(
            (wheel_paths[name], str(spec["sha256"]), spec)
            for name, spec in sorted(runtime_wheels.items())
        ),
    ):
        files = _wheel_files(path, expected)
        with zipfile.ZipFile(path) as archive:
            for name, (_digest, mode) in files.items():
                destination = temporary / name
                parent = temporary
                for component in PurePosixPath(name).parent.parts:
                    parent = parent / component
                    if parent.exists() or parent.is_symlink():
                        _private_dir(parent)
                    else:
                        parent.mkdir(mode=0o700)
                        parent.chmod(0o700)
                        _private_dir(parent)
                descriptor = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    mode,
                )
                try:
                    view = memoryview(archive.read(name))
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise FlashBuildWorkerError(
                                "wheel extraction was incomplete"
                            )
                        view = view[written:]
                finally:
                    os.close(descriptor)
                destination.chmod(mode)
    if _overlay_tree_digest(temporary, expected_files) != expected_digest:
        raise FlashBuildWorkerError("extracted Python overlay changed")
    _atomic_json(
        temporary / ".aeon-overlay-receipt.json",
        {
            "schema_version": "aeon-qwen38-modelopt-python-overlay-v1",
            "manifest_sha256": MODELOPT_RUNTIME_MANIFEST_SHA256,
            "tree_sha256": expected_digest,
            "file_count": len(expected_files),
        },
    )
    temporary.rename(paths["overlay"])


def _pythonpath(request: dict[str, Any]) -> str:
    paths = _paths(request)
    return ":".join((str(paths["source"]), str(paths["overlay"]), str(ENV_SITE)))


def _verify_environment(request: dict[str, Any]) -> dict[str, Any]:
    _extract_overlay(request)
    script = (
        "import importlib.metadata as m,json;"
        "from modelopt.torch.quantization.qtensor import NVFP4QTensor;"
        "names=['PyYAML','accelerate','annotated-types','antlr4-python3-runtime',"
        "'huggingface-hub','ninja','numpy','nvidia-ml-py','nvidia-modelopt',"
        "'omegaconf','packaging','peft','psutil','pulp','pydantic','pydantic-core',"
        "'regex','rich','safetensors','scipy','setuptools','tokenizers','torch',"
        "'tqdm','transformers','typing-extensions','typing-inspection'];"
        "print(json.dumps({'versions':{k:m.version(k) for k in names},"
        "'nvfp4_qtensor':NVFP4QTensor.__module__+'.'+NVFP4QTensor.__name__},"
        "sort_keys=True))"
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/aday",
        "LANG": "C",
        "LC_ALL": "C",
        "CUDA_VISIBLE_DEVICES": "void",
        "PYTHONPATH": _pythonpath(request),
        "PYTHONNOUSERSITE": "1",
        "USE_TF": "0",
        "USE_FLAX": "0",
    }
    result = subprocess.run(
        [str(ENV_PYTHON), "-c", script],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        receipt = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FlashBuildWorkerError("Python environment receipt is malformed") from exc
    if (
        result.returncode != 0
        or receipt.get("versions") != REVIEWED_DISTRIBUTIONS
        or receipt.get("nvfp4_qtensor") != NVFP4_QTENSOR_IMPORT
    ):
        raise FlashBuildWorkerError("reviewed Python environment is unavailable")
    return {
        "versions": {
            str(key): str(value) for key, value in REVIEWED_DISTRIBUTIONS.items()
        },
        "nvfp4_qtensor": receipt["nvfp4_qtensor"],
    }


def _validate_resume_source(request: dict[str, Any]) -> dict[str, Any]:
    if (
        request.get("recipe") != RESUME_RECIPE
        or request.get("host") != LOCAL_HOST
        or request.get("resume_source_manifest_sha256") != RESUME_MANIFEST_SHA256
    ):
        raise FlashBuildWorkerError("quant-only resume was not explicitly bound")
    paths = _paths(request)
    if _sha256(paths["resume_manifest"]) != RESUME_MANIFEST_SHA256:
        raise FlashBuildWorkerError("quant-only resume receipt changed")
    provenance = _read_json(paths["resume_manifest"])
    source_runtime = provenance.get("source_runtime")
    storage = provenance.get("storage")
    release = provenance.get("coordinator_release")
    receipts = provenance.get("receipts")
    failure = provenance.get("failure_boundary")
    if (
        set(provenance)
        != {
            "schema_version",
            "complete",
            "canonical_root",
            "source_runtime",
            "storage",
            "coordinator_release",
            "receipts",
            "failure_boundary",
        }
        or provenance.get("schema_version")
        != "aeon-qwen38-flash-next-quant-resume-source-v1"
        or provenance.get("complete") is not True
        or provenance.get("canonical_root") != str(RESUME_ROOT)
        or not isinstance(source_runtime, dict)
        or source_runtime.get("runtime_id") != RESUME_RUNTIME_ID
        or source_runtime.get("state") != "stopped"
        or source_runtime.get("process_absent") is not True
        or source_runtime.get("host") != LOCAL_HOST
        or source_runtime.get("physical_gpu") != 0
        or source_runtime.get("request_sha256")
        != "ec34659a33773cfbdda3cfc1986c1b7372fe53903c0d410e6645d1fc6c49df7c"
        or not isinstance(storage, dict)
        or storage.get("state") != "complete"
        or storage.get("output_settled") is not True
        or storage.get("cleanup_complete") is not True
        or storage.get("canonical_output_path") != str(RESUME_ROOT)
        or not isinstance(release, dict)
        or release.get("event_id") != 14263
        or release.get("event_type") != "released"
        or release.get("claim_id") != source_runtime.get("claim_id")
        or release.get("owner") != source_runtime.get("owner")
        or not isinstance(receipts, dict)
        or not isinstance(failure, dict)
        or failure
        != {
            "result_failure": "NVFP4 builder failed",
            "builder_failure": "ModelOpt NVFP4QTensor could not be imported",
            "final_model_absent": True,
            "official_untuned_model_absent": True,
            "sibling_manifest_absent": True,
            "quantized_layer_files": 0,
        }
    ):
        raise FlashBuildWorkerError("quant-only resume provenance is malformed")
    if RESUME_ROOT == paths["scratch"] or RESUME_ROOT.parent != paths["scratch"].parent:
        raise FlashBuildWorkerError("quant-only resume output is not a fresh sibling")
    for pid_name in ("pid", "modelopt_pid"):
        pid = source_runtime.get(pid_name)
        if type(pid) is not int or pid <= 1 or Path(f"/proc/{pid}").exists():
            raise FlashBuildWorkerError("prior build process absence is not proven")
    try:
        os.killpg(source_runtime["pid"], 0)
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise FlashBuildWorkerError(
                "prior build process-group absence is not proven"
            ) from exc
    else:
        raise FlashBuildWorkerError("prior build process group is still live")
    for directory in (
        RESUME_ROOT,
        paths["hybrid"],
        paths["behavior"],
        RESUME_ROOT / "output",
    ):
        _private_dir(directory)
    expected_receipts = {
        "qwen-flash-next-build-request.json",
        "spawn.json",
        "output/preflight.json",
        "output/result.json",
        "output/builder-result.json",
        "output/MANIFEST.sha256",
        "hybrid/HYBRID_MANIFEST.json",
        "behavior-receipt.json",
        "behavior-adapter/aeon_behavior_manifest.json",
    }
    if set(receipts) != expected_receipts:
        raise FlashBuildWorkerError("quant-only resume receipt set changed")
    for relative, file_receipt in receipts.items():
        if _SAFE_RELATIVE.fullmatch(relative) is None or not isinstance(
            file_receipt, dict
        ):
            raise FlashBuildWorkerError("quant-only resume receipt path is unsafe")
        _verify_regular(RESUME_ROOT / relative, file_receipt, 64 * 1024**2)
    old_request = _read_json(RESUME_ROOT / "qwen-flash-next-build-request.json")
    spawn = _read_json(RESUME_ROOT / "spawn.json")
    result = _read_json(RESUME_ROOT / "output/result.json")
    builder_result = _read_json(RESUME_ROOT / "output/builder-result.json")
    if (
        old_request.get("runtime_id") != RESUME_RUNTIME_ID
        or old_request.get("claim_id") != source_runtime.get("claim_id")
        or old_request.get("owner") != source_runtime.get("owner")
        or old_request.get("scratch_path") != str(RESUME_ROOT)
        or spawn.get("runtime_id") != RESUME_RUNTIME_ID
        or spawn.get("request_sha256") != source_runtime.get("request_sha256")
        or spawn.get("pid") != source_runtime.get("pid")
        or result.get("success") is not False
        or result.get("failure") != failure["result_failure"]
        or builder_result.get("success") is not False
        or builder_result.get("failure") != failure["builder_failure"]
    ):
        raise FlashBuildWorkerError("prior build lifecycle receipt changed")
    output = RESUME_ROOT / "output"
    if {item.name for item in output.iterdir()} != {
        "MANIFEST.sha256",
        "build.log",
        "builder-result.json",
        "preflight.json",
        "result.json",
    }:
        raise FlashBuildWorkerError("prior failed output closure changed")
    manifest_entries: dict[str, str] = {}
    for line in (output / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  ([A-Za-z0-9_.-]+)", line)
        if match is None or match.group(2) in manifest_entries:
            raise FlashBuildWorkerError("prior failed output manifest is malformed")
        manifest_entries[match.group(2)] = match.group(1)
    if set(manifest_entries) != {
        "build.log",
        "builder-result.json",
        "preflight.json",
        "result.json",
    } or any(
        _sha256(output / name) != digest for name, digest in manifest_entries.items()
    ):
        raise FlashBuildWorkerError("prior failed output manifest changed")
    from aeon.scripts import build_qwen38_flash_next_nvfp4 as builder

    hybrid_manifest = builder._load_manifest(
        paths["hybrid"] / "HYBRID_MANIFEST.json",
        receipts["hybrid/HYBRID_MANIFEST.json"]["sha256"],
        builder.HYBRID_SCHEMA,
    )
    *_, hybrid_locations, _metadata, _names, hybrid_files = builder._validate_hybrid(
        paths["hybrid"], hybrid_manifest
    )
    if {item.name for item in paths["hybrid"].iterdir()} != {
        "HYBRID_MANIFEST.json",
        *hybrid_files,
    }:
        raise FlashBuildWorkerError("resume hybrid contains unreceipted files")
    behavior_receipt = _read_json(paths["behavior_receipt"])
    behavior_manifest_path = paths["behavior"] / "aeon_behavior_manifest.json"
    if (
        behavior_receipt.get("status") != "completed"
        or behavior_receipt.get("manifest_sha256")
        != receipts["behavior-adapter/aeon_behavior_manifest.json"]["sha256"]
        or behavior_receipt.get("source_manifest_sha256")
        != "c447659c92938c913aaca8f37c847a1bad44c070cd76e211056cde1dba9d5670"
    ):
        raise FlashBuildWorkerError("prior behavior receipt changed")
    behavior_manifest = builder._load_manifest(
        behavior_manifest_path,
        receipts["behavior-adapter/aeon_behavior_manifest.json"]["sha256"],
        builder.ADAPTER_SCHEMA,
    )
    builder._validate_adapter(
        paths["behavior"] / "adapter_model.safetensors", behavior_manifest
    )
    behavior_files = behavior_manifest.get("files")
    if not isinstance(behavior_files, dict) or {
        item.name for item in paths["behavior"].iterdir()
    } != {
        "aeon_behavior_manifest.json",
        *behavior_files,
    }:
        raise FlashBuildWorkerError("resume behavior adapter file closure changed")
    for name, file_receipt in behavior_files.items():
        _verify_regular(paths["behavior"] / name, file_receipt, 16 * 1024**2)
    closure = {
        "resume_manifest_sha256": RESUME_MANIFEST_SHA256,
        "hybrid_manifest_sha256": receipts["hybrid/HYBRID_MANIFEST.json"]["sha256"],
        "behavior_receipt_sha256": receipts["behavior-receipt.json"]["sha256"],
        "behavior_manifest_sha256": receipts[
            "behavior-adapter/aeon_behavior_manifest.json"
        ]["sha256"],
        "hybrid_tensor_count": len(hybrid_locations),
        "hybrid_file_count": len(hybrid_files),
    }
    return {**closure, "closure_sha256": _canonical_sha(closure)}


def _verify_acl(request: dict[str, Any]) -> None:
    result = subprocess.run(
        ["/usr/bin/getfacl", "-cp", "--", f"/dev/nvidia{request['physical_gpu']}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0 or "user:aday:---" in result.stdout.splitlines():
        raise FlashBuildWorkerError("leased GPU is renter-blocked or ambiguous")


def _resources(request: dict[str, Any]) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].rstrip(":") in {
            "MemAvailable",
            "CommitLimit",
            "Committed_AS",
        }:
            values[fields[0].rstrip(":")] = int(fields[1]) * 1024
    stats = os.statvfs(_paths(request)["scratch"])
    shm = os.statvfs("/dev/shm")
    receipt = {
        "scratch_device": int(_paths(request)["scratch"].lstat().st_dev),
        "memory_available_bytes": values.get("MemAvailable", 0),
        "commit_available_bytes": values.get("CommitLimit", 0)
        - values.get("Committed_AS", 0),
        "disk_free_bytes": stats.f_bavail * stats.f_frsize,
        "disk_free_inodes": stats.f_favail,
        "shm_free_bytes": shm.f_bavail * shm.f_frsize,
    }
    gib = 1024**3
    if (
        receipt["memory_available_bytes"] < float(request["min_host_memory_gb"]) * gib
        or receipt["commit_available_bytes"]
        < float(request["min_host_commit_gb"]) * gib
        or receipt["disk_free_bytes"] < int(request["post_stage_disk_floor_bytes"])
        or receipt["shm_free_bytes"] < float(request["min_shm_free_gb"]) * gib
    ):
        raise FlashBuildWorkerError("host resources changed after reservation")
    return receipt


def _preflight(request: dict[str, Any], digest: str) -> dict[str, Any]:
    paths = _paths(request)
    _private_dir(paths["scratch"])
    _private_dir(paths["output"], create=True)
    if set(item.name for item in paths["output"].iterdir()) - {"preflight.json"}:
        raise FlashBuildWorkerError("fresh build output contains lifecycle artifacts")
    _atomic_json(
        paths["preflight"],
        {
            "schema_version": SCHEMA_VERSION,
            "request_sha256": digest,
            "state": "staging",
            "host_resources": {"scratch_device": int(paths["scratch"].lstat().st_dev)},
        },
    )
    versions = _verify_environment(request)
    if request.get("host") == LOCAL_HOST:
        source_bytes, input_bytes, fixture_bytes = _verify_local_manifests(
            request, paths
        )
    else:
        source_bytes = _verify_manifested(
            paths["source"], request["source_files"], 4 * 1024**2
        )
        input_bytes = _verify_manifested(
            paths["scratch"] / "inputs", request["input_files"], 32 * 1024**3
        )
        fixture_bytes = _verify_manifested(
            paths["fixtures"], request["fixture_files"], 32 * 1024**3
        )
    _verify_acl(request)
    resume_source = None
    if request["recipe"] == RESUME_RECIPE:
        resume_source = _validate_resume_source(request)
        stage = {
            "schema_version": "aeon-qwen38-flash-next-quant-resume-stage-v1",
            "hybrid_manifest_sha256": resume_source["hybrid_manifest_sha256"],
            "resume_closure_sha256": resume_source["closure_sha256"],
        }
    else:
        stage = assembler.stage_sources(
            bf16_root=paths["bf16"],
            bf16_files_manifest=paths["bf16_files"],
            fp8_root=paths["fp8"],
            fp8_files_manifest=paths["fp8_files"],
            mtp_subset=paths["mtp"],
            source_manifest=paths["source_manifest"],
        )
    resources = _resources(request)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "request_sha256": digest,
        "source_bytes": source_bytes,
        "input_bytes": input_bytes,
        "fixture_bytes": fixture_bytes,
        "source_stage": stage,
        "environment": versions,
        "recipe": request["recipe"],
        "resume_source": resume_source,
        "host_resources": resources,
        "verified_at": time.time(),
    }
    _atomic_json(paths["preflight"], receipt)
    return {"state": "preflight_ready", **receipt}


def _remove_exact_work(path: Path, *, missing_ok: bool = False) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        if missing_ok:
            return
        raise FlashBuildWorkerError("trainer work directory is absent") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or os.path.ismount(path)
        or not shutil.rmtree.avoids_symlink_attacks
    ):
        raise FlashBuildWorkerError("trainer work directory identity changed")
    for item in path.rglob("*"):
        item_metadata = item.lstat()
        if (
            item_metadata.st_uid != os.geteuid()
            or item_metadata.st_dev != metadata.st_dev
            or stat.S_ISLNK(item_metadata.st_mode)
            or os.path.ismount(item)
            or not (
                stat.S_ISDIR(item_metadata.st_mode)
                or stat.S_ISREG(item_metadata.st_mode)
            )
            or (stat.S_ISREG(item_metadata.st_mode) and item_metadata.st_nlink != 1)
        ):
            raise FlashBuildWorkerError(
                "trainer work directory contains an unsafe inode"
            )
    shutil.rmtree(path)


def _run_behavior_trainer(
    command: list[str],
    *,
    environment: dict[str, str],
    source: Path,
    log: Any,
    work: Path,
) -> None:
    try:
        result = subprocess.run(
            command,
            env=environment,
            cwd=source,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    except BaseException:
        _remove_exact_work(work, missing_ok=True)
        raise
    if result.returncode != 0:
        _remove_exact_work(work, missing_ok=True)
        raise FlashBuildWorkerError("behavior trainer failed")
    _remove_exact_work(work)


def _pipeline(request: dict[str, Any]) -> None:
    paths = _paths(request)
    environment = {
        "PATH": "/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/home/aday",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": _pythonpath(request),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "USE_TF": "0",
        "USE_FLAX": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "8",
        "GPU_AGENT_CLAIM_ID": request["claim_id"],
        "CUDA_VISIBLE_DEVICES": request["gpu_uuid"],
        "GPU_MEM_LIMIT_GB": "88.0",
        "GPU_PLANNED_VRAM_GB": "88.0",
        "GPU_RESERVE_GB": "6.0",
        "GPU_LEASE_RUN_DIR": str(paths["scratch"]),
        "GPU_LEASE_OWNER": request["owner"],
        "GPU_LEASE_EXCLUSIVE": "1",
        "AEON_BEHAVIOR_RUNTIME_ID": request["runtime_id"],
        "AEON_QUANT_RUNTIME_ID": request["runtime_id"],
        "AEON_QUANT_RESULT_PATH": str(paths["build_result"]),
    }
    preflight = _read_json(paths["preflight"])
    stage = preflight["source_stage"]
    if (
        preflight.get("request_sha256") != _sha256(paths["request"])
        or preflight.get("recipe") != request["recipe"]
        or preflight.get("environment", {}).get("nvfp4_qtensor") != NVFP4_QTENSOR_IMPORT
    ):
        raise FlashBuildWorkerError("ModelOpt import preflight is stale")
    resume_before = None
    with paths["log"].open("ab", buffering=0) as log:
        if request["recipe"] == RESUME_RECIPE:
            resume_before = _validate_resume_source(request)
            if (
                preflight.get("resume_source", {}).get("closure_sha256")
                != resume_before["closure_sha256"]
                or stage.get("resume_closure_sha256") != resume_before["closure_sha256"]
            ):
                raise FlashBuildWorkerError("resume source changed after preflight")
            hybrid_receipt = {
                "hybrid_manifest_sha256": resume_before["hybrid_manifest_sha256"]
            }
        else:
            trainer = [
                str(BASH),
                str(LOW_PRIORITY),
                str(ENV_PYTHON),
                "-m",
                "aeon.scripts.train_qwen38_flash_next_behavior",
                "--bf16-root",
                str(paths["bf16"]),
                "--bf16-index",
                str(paths["bf16"] / "model.safetensors.index.json"),
                "--bf16-index-sha256",
                stage["bf16_index_sha256"],
                "--fp8-ple-root",
                str(paths["fp8"]),
                "--fp8-ple-index",
                str(paths["fp8"] / "model.safetensors.index.json"),
                "--fp8-ple-index-sha256",
                stage["fp8_index_sha256"],
                "--mtp-subset",
                str(paths["mtp"]),
                "--mtp-subset-sha256",
                stage["mtp_subset_sha256"],
                "--source-manifest",
                str(paths["source_manifest"]),
                "--source-manifest-sha256",
                stage["source_manifest_sha256"],
                "--train-jsonl",
                str(paths["train"]),
                "--eval-jsonl",
                str(paths["eval"]),
                "--output-dir",
                str(paths["behavior"]),
                "--receipt",
                str(paths["behavior_receipt"]),
                "--learning-rate",
                "5e-6",
                "--max-sequence-length",
                "512",
                "--feature-batch-size",
                "4",
                "--cpu-memory-gib",
                "152",
                "--disk-memory-gib",
                "150",
            ]
            trainer_work = paths["scratch"] / f".behavior-work-{request['runtime_id']}"
            _run_behavior_trainer(
                trainer,
                environment=environment,
                source=paths["source"],
                log=log,
                work=trainer_work,
            )
            hybrid_receipt = assembler.assemble(
                bf16_metadata_root=paths["bf16"],
                bf16_files_manifest=paths["bf16_files"],
                fp8_root=paths["fp8"],
                fp8_files_manifest=paths["fp8_files"],
                output=paths["hybrid"],
                # On .177 the task atomically transitions newly staged BF16
                # shards into its canonical hybrid. Preserve shared FP8 input.
                preserve_fp8_sources=request["host"] == LOCAL_HOST,
            )
        behavior_receipt = _read_json(paths["behavior_receipt"])
        behavior_manifest = paths["behavior"] / "aeon_behavior_manifest.json"
        if behavior_receipt.get("status") != "completed" or behavior_receipt.get(
            "manifest_sha256"
        ) != _sha256(behavior_manifest):
            raise FlashBuildWorkerError("behavior adapter receipt is invalid")
        builder = [
            str(BASH),
            str(LOW_PRIORITY),
            str(ENV_PYTHON),
            "-m",
            "aeon.scripts.build_qwen38_flash_next_nvfp4",
            "--hybrid",
            str(paths["hybrid"]),
            "--hybrid-manifest",
            str(paths["hybrid"] / "HYBRID_MANIFEST.json"),
            "--hybrid-manifest-sha256",
            hybrid_receipt["hybrid_manifest_sha256"],
            "--mtp-subset",
            str(paths["mtp"]),
            "--mtp-manifest",
            str(paths["mtp_manifest"]),
            "--mtp-manifest-sha256",
            request["fixture_files"]["mtp-bf16.manifest.json"]["sha256"],
            "--expert-scales",
            str(paths["scales"]),
            "--expert-scales-manifest",
            str(paths["scales_manifest"]),
            "--expert-scales-manifest-sha256",
            request["fixture_files"]["expert-scales.manifest.json"]["sha256"],
            "--adapter",
            str(paths["behavior"] / "adapter_model.safetensors"),
            "--adapter-manifest",
            str(behavior_manifest),
            "--adapter-manifest-sha256",
            _sha256(behavior_manifest),
            "--modelopt-wheel",
            str(paths["modelopt"]),
            "--output",
            str(paths["model"]),
        ]
        result = subprocess.run(
            builder,
            env=environment,
            cwd=paths["source"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            raise FlashBuildWorkerError("NVFP4 builder failed")
    if resume_before is not None:
        resume_after = _validate_resume_source(request)
        if resume_after != resume_before:
            raise FlashBuildWorkerError("resume source changed during quantization")
    build = _read_json(paths["model"] / "BUILD_MANIFEST.json")
    if build.get("complete") is not True:
        raise FlashBuildWorkerError("final build manifest is incomplete")
    sibling = _read_json(paths["sibling_manifest"])
    if (
        sibling.get("schema_version")
        != "aeon-qwen38-flash-next-official-untuned-sibling-v1"
        or sibling.get("complete") is not True
        or sibling.get("tuned_checkpoint_tree_sha256")
        != _sha256(paths["model"] / "SHA256SUMS")
        or sibling.get("official_untuned_checkpoint_tree_sha256")
        != _sha256(paths["official_untuned_model"] / "SHA256SUMS")
    ):
        raise FlashBuildWorkerError("official untuned sibling receipt is invalid")
    _atomic_json(
        paths["qualification"],
        {
            "schema_version": "aeon-qwen38-flash-next-qualification-required-v1",
            "status": "pending-separate-fleet-qualified-.177-runtime",
            "sglang_commit": SGLANG_COMMIT,
            "sglang_image": SGLANG_IMAGE_REFERENCE,
            "required": [
                "text",
                "image",
                "video",
                "MTP-off/on-three-trial",
                "VRAM-and-cgroup-RAM",
            ],
            "mtp_gates": {
                "median_ratio_min": 1.10,
                "ci_lower_bound_min": 1.03,
                "accept_length_min": 1.0,
                "failures": 0,
            },
        },
    )
    _atomic_json(
        paths["result"],
        {
            "schema_version": RESULT_SCHEMA,
            "success": True,
            "build_manifest_sha256": _sha256(paths["model"] / "BUILD_MANIFEST.json"),
            "sibling_manifest_sha256": _sha256(paths["sibling_manifest"]),
            "official_untuned_checkpoint_tree_sha256": sibling[
                "official_untuned_checkpoint_tree_sha256"
            ],
            "qualification_status": "pending",
            "recipe": request["recipe"],
            "resume_source_closure_sha256": (
                resume_before["closure_sha256"] if resume_before else None
            ),
            "completed_at": time.time(),
        },
    )


def _process_alive(request: dict[str, Any], pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    proc = Path(f"/proc/{pid}")
    try:
        environment = (proc / "environ").read_bytes().split(b"\0")
        command = (proc / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError as exc:
        raise FlashBuildWorkerError("build process identity is unreadable") from exc
    return (
        f"GPU_AGENT_CLAIM_ID={request['claim_id']}".encode() in environment
        and f"CUDA_VISIBLE_DEVICES={request['gpu_uuid']}".encode() in environment
        and b"GPU_MEM_LIMIT_GB=88.0" in environment
        and str(
            _paths(request)["source"] / "aeon/scripts/qwen_flash_next_build_worker.py"
        ).encode()
        in command
        and b"pipeline" in command
    )


def _process_group_alive(pgid: int) -> bool:
    if isinstance(pgid, bool) or not isinstance(pgid, int) or pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise FlashBuildWorkerError(
            "build process-group identity is ambiguous"
        ) from exc
    return True


def _spawn(request: dict[str, Any], digest: str) -> dict[str, Any]:
    paths = _paths(request)
    if (
        _read_json(paths["preflight"]).get("request_sha256") != digest
        or paths["spawn"].exists()
    ):
        raise FlashBuildWorkerError("preflight is stale or lifecycle exists")
    descriptor = os.open(
        paths["log"],
        SPAWN_LOG_FLAGS,
        0o600,
    )
    environment = {
        "PATH": "/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/home/aday",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": _pythonpath(request),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GPU_AGENT_CLAIM_ID": request["claim_id"],
        "CUDA_VISIBLE_DEVICES": request["gpu_uuid"],
        "GPU_MEM_LIMIT_GB": "88.0",
    }
    try:
        process = subprocess.Popen(
            [
                str(BASH),
                str(LOW_PRIORITY),
                str(ENV_PYTHON),
                str(paths["source"] / "aeon/scripts/qwen_flash_next_build_worker.py"),
                "pipeline",
                str(paths["request"]),
                digest,
            ],
            cwd=paths["source"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(descriptor)
    _atomic_json(
        paths["spawn"],
        {
            "schema_version": SCHEMA_VERSION,
            "runtime_id": request["runtime_id"],
            "request_sha256": digest,
            "pid": process.pid,
            "created_at": time.time(),
        },
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if _process_alive(request, process.pid):
            return {"state": "running", "pid": process.pid}
        if process.poll() is not None:
            raise FlashBuildWorkerError("build pipeline exited during spawn")
        time.sleep(0.1)
    raise FlashBuildWorkerError("build process identity did not become visible")


def _spawn_receipt(request: dict[str, Any]) -> dict[str, Any] | None:
    path = _paths(request)["spawn"]
    if not path.is_file():
        return None
    value = _read_json(path)
    if value.get("runtime_id") != request["runtime_id"] or value.get(
        "request_sha256"
    ) != _sha256(_paths(request)["request"]):
        raise FlashBuildWorkerError("spawn receipt identity changed")
    return value


def _terminal(request: dict[str, Any], reason: str) -> dict[str, Any]:
    _remove_exact_work(
        _paths(request)["scratch"] / f".behavior-work-{request['runtime_id']}",
        missing_ok=True,
    )
    path = _paths(request)["result"]
    if path.is_file():
        result = _read_json(path)
    else:
        result = {
            "schema_version": RESULT_SCHEMA,
            "success": False,
            "failure": reason[:1000],
            "completed_at": time.time(),
        }
        _atomic_json(path, result)
    if result.get("schema_version") != RESULT_SCHEMA:
        raise FlashBuildWorkerError("terminal schema changed")
    return result


def _status(request: dict[str, Any]) -> dict[str, Any]:
    receipt = _spawn_receipt(request)
    pid = receipt.get("pid") if receipt else None
    if isinstance(pid, int) and _process_alive(request, pid):
        phase = (
            "validated_resume_then_quantize"
            if request.get("recipe") == RESUME_RECIPE
            else "tune_then_quantize"
        )
        return {"state": "running", "pid": pid, "phase": phase}
    if receipt is None:
        return {"state": "absent", "pid": None}
    if isinstance(pid, int) and _process_group_alive(pid):
        return {"state": "ambiguous", "pid": pid, "phase": "task-process-group-remains"}
    result = _terminal(request, "build process exited without a terminal receipt")
    return {
        "state": "completed" if result.get("success") is True else "failed",
        "pid": pid,
        "result": result,
    }


def _stop(request: dict[str, Any]) -> dict[str, Any]:
    receipt = _spawn_receipt(request)
    pid = receipt.get("pid") if receipt else None
    if isinstance(pid, int) and (
        _process_alive(request, pid) or _process_group_alive(pid)
    ):
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and _process_group_alive(pid):
            time.sleep(0.5)
        if _process_group_alive(pid):
            return {"state": "ambiguous", "process_absent": False}
    _terminal(request, "build was stopped before successful completion")
    return {"state": "stopped", "process_absent": True}


def _write_manifest(request: dict[str, Any]) -> str:
    paths = _paths(request)
    output = _private_dir(paths["output"])
    manifest = paths["manifest"]
    if manifest.exists():
        return _sha256(manifest)
    _terminal(request, "build ended before settlement")
    entries: list[tuple[str, str]] = []
    total = 0
    seen_inodes: set[tuple[int, int]] = set()
    for item in sorted(output.rglob("*")):
        metadata = item.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise FlashBuildWorkerError("output directory is unsafe")
            continue
        relative = item.relative_to(output).as_posix()
        if (
            item == manifest
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_ISLNK(metadata.st_mode)
            or _SAFE_RELATIVE.fullmatch(relative) is None
        ):
            raise FlashBuildWorkerError("output contains an unsafe inode")
        inode = (metadata.st_dev, metadata.st_ino)
        if inode not in seen_inodes:
            seen_inodes.add(inode)
            total += metadata.st_size
        if total > 150 * 1024**3:
            raise FlashBuildWorkerError("output exceeded its bound")
        entries.append((_sha256(item), relative))
    temporary = manifest.with_name(".MANIFEST.sha256.tmp")
    temporary.write_text(
        "".join(f"{digest}  {name}\n" for digest, name in entries), encoding="utf-8"
    )
    temporary.chmod(0o600)
    os.replace(temporary, manifest)
    return _sha256(manifest)


def _settle_status(request: dict[str, Any]) -> dict[str, Any]:
    receipt = _spawn_receipt(request)
    pid = receipt.get("pid") if receipt else None
    if isinstance(pid, int) and (
        _process_alive(request, pid) or _process_group_alive(pid)
    ):
        raise FlashBuildWorkerError("cannot settle while task process group is alive")
    result = _terminal(request, "build ended before settlement")
    digest = _write_manifest(request)
    return {"state": "settle_ready", "manifest_sha256": digest, "result": result}


def _mark_settled(request: dict[str, Any], digest: str) -> dict[str, Any]:
    if _settle_status(request)["manifest_sha256"] != digest:
        raise FlashBuildWorkerError("settled manifest changed")
    _atomic_json(
        _paths(request)["settled"],
        {
            "schema_version": SCHEMA_VERSION,
            "runtime_id": request["runtime_id"],
            "manifest_sha256": digest,
        },
    )
    return {"state": "settled", "manifest_sha256": digest}


def _safe_tree_bytes(root: Path) -> int:
    root_metadata = _private_dir(root).lstat()
    total = 0
    seen_inodes: set[tuple[int, int]] = set()
    for item in root.rglob("*"):
        metadata = item.lstat()
        if (
            metadata.st_uid != os.geteuid()
            or metadata.st_dev != root_metadata.st_dev
            or stat.S_ISLNK(metadata.st_mode)
            or os.path.ismount(item)
            or not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode))
        ):
            raise FlashBuildWorkerError("scratch contains an unsafe inode")
        inode = (metadata.st_dev, metadata.st_ino)
        if inode not in seen_inodes:
            seen_inodes.add(inode)
            total += metadata.st_blocks * 512
    return total


def _preflight_missing_cleanup_token(
    request: dict[str, Any], digest: str
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Prove an exact request-owned staging tree never entered preflight.

    The request digest is Fleet's descriptor-bound object token.  A missing
    preflight receipt is cleanup-eligible only while the tree remains the exact
    private, same-filesystem subset declared by that request and before any
    process or lifecycle output could have been created.
    """

    paths = _paths(request)
    scratch = _private_dir(paths["scratch"])
    scratch_metadata = scratch.lstat()
    parent_metadata = scratch.parent.lstat()
    if (
        PurePosixPath(str(scratch)).parent != SCRATCH_ROOT
        or scratch.name != request.get("runtime_id")
        or paths["source"] != scratch / "source"
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or stat.S_ISLNK(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or parent_metadata.st_gid != os.getegid()
        or parent_metadata.st_mode & 0o002
        or scratch_metadata.st_dev != parent_metadata.st_dev
        or os.path.ismount(scratch)
        or os.path.lexists(paths["preflight"])
        or os.path.lexists(paths["spawn"])
    ):
        raise FlashBuildWorkerError("incomplete staging lifecycle is ambiguous")

    request_path = paths["request"]
    request_metadata = request_path.lstat()
    if (
        not stat.S_ISREG(request_metadata.st_mode)
        or stat.S_ISLNK(request_metadata.st_mode)
        or request_metadata.st_uid != os.geteuid()
        or request_metadata.st_nlink != 1
        or request_metadata.st_mode & 0o077
        or request_metadata.st_dev != scratch_metadata.st_dev
        or _sha256(request_path) != digest
    ):
        raise FlashBuildWorkerError("incomplete staging request token changed")

    declared: dict[Path, dict[str, Any]] = {
        request_path: {"sha256": digest, "size": request_metadata.st_size},
    }
    groups = (
        (paths["source"], request.get("source_files")),
        (paths["scratch"] / "inputs", request.get("input_files")),
        (paths["fixtures"], request.get("fixture_files")),
    )
    for root, values in groups:
        if not isinstance(values, dict):
            raise FlashBuildWorkerError("incomplete staging descriptor is malformed")
        for relative, receipt in values.items():
            if (
                not isinstance(relative, str)
                or _SAFE_RELATIVE.fullmatch(relative) is None
                or not isinstance(receipt, dict)
            ):
                raise FlashBuildWorkerError(
                    "incomplete staging descriptor contains an unsafe path"
                )
            declared[root / relative] = receipt

    allowed_directories = {scratch, paths["output"]}
    for item in declared:
        parent = item.parent
        while parent != scratch:
            try:
                parent.relative_to(scratch)
            except ValueError as exc:
                raise FlashBuildWorkerError(
                    "incomplete staging descriptor escaped its task root"
                ) from exc
            allowed_directories.add(parent)
            parent = parent.parent

    seen_files: set[Path] = set()
    for walk_root, directory_names, file_names in os.walk(
        scratch, topdown=True, followlinks=False
    ):
        walk_path = Path(walk_root)
        for name in directory_names:
            item = walk_path / name
            metadata = item.lstat()
            if (
                item not in allowed_directories
                or not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o077
                or metadata.st_dev != scratch_metadata.st_dev
                or os.path.ismount(item)
            ):
                raise FlashBuildWorkerError(
                    "incomplete staging contains an unsafe directory"
                )
        for name in file_names:
            item = walk_path / name
            receipt = declared.get(item)
            if receipt is None:
                raise FlashBuildWorkerError(
                    "incomplete staging contains an undeclared file"
                )
            size = receipt.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise FlashBuildWorkerError(
                    "incomplete staging descriptor contains an invalid size"
                )
            _verify_regular(item, receipt, size)
            if item.lstat().st_dev != scratch_metadata.st_dev:
                raise FlashBuildWorkerError("incomplete staging filesystem changed")
            seen_files.add(item)

    if request_path not in seen_files or any(paths["output"].iterdir()):
        raise FlashBuildWorkerError("incomplete staging lifecycle output is ambiguous")
    return (
        (scratch_metadata.st_dev, scratch_metadata.st_ino),
        (
            request_metadata.st_dev,
            request_metadata.st_ino,
            request_metadata.st_size,
            request_metadata.st_mtime_ns,
        ),
    )


def _revalidate_preflight_missing_cleanup_token(
    request: dict[str, Any],
    digest: str,
    token: tuple[tuple[int, int], tuple[int, int, int, int]],
) -> None:
    paths = _paths(request)
    scratch_metadata = paths["scratch"].lstat()
    request_metadata = paths["request"].lstat()
    if (
        (scratch_metadata.st_dev, scratch_metadata.st_ino) != token[0]
        or (
            request_metadata.st_dev,
            request_metadata.st_ino,
            request_metadata.st_size,
            request_metadata.st_mtime_ns,
        )
        != token[1]
        or os.path.ismount(paths["scratch"])
        or os.path.lexists(paths["preflight"])
        or os.path.lexists(paths["spawn"])
        or _sha256(paths["request"]) != digest
        or not shutil.rmtree.avoids_symlink_attacks
    ):
        raise FlashBuildWorkerError("incomplete staging cleanup token changed")


def _cleanup(
    request: dict[str, Any], digest: str, *, prelaunch: bool = False
) -> dict[str, Any]:
    if request.get("host") == LOCAL_HOST:
        raise FlashBuildWorkerError("canonical .177 build data is never auto-cleaned")
    paths = _paths(request)
    missing_preflight_token = None
    if os.path.lexists(paths["preflight"]):
        preflight = _read_json(paths["preflight"])
        if (
            preflight.get("host_resources", {}).get("scratch_device")
            != paths["scratch"].lstat().st_dev
        ):
            raise FlashBuildWorkerError("scratch filesystem identity changed")
    elif prelaunch:
        missing_preflight_token = _preflight_missing_cleanup_token(request, digest)
    else:
        raise FlashBuildWorkerError("preflight receipt is absent")
    receipt = _spawn_receipt(request)
    pid = receipt.get("pid") if receipt else None
    if isinstance(pid, int) and (
        _process_alive(request, pid) or _process_group_alive(pid)
    ):
        raise FlashBuildWorkerError("runtime process group is not absent")
    if prelaunch:
        if receipt is not None or paths["result"].exists() or paths["model"].exists():
            raise FlashBuildWorkerError("prelaunch cleanup found lifecycle output")
    else:
        marker = _read_json(paths["settled"])
        if (
            marker.get("runtime_id") != request["runtime_id"]
            or marker.get("manifest_sha256") != digest
            or _sha256(paths["manifest"]) != digest
        ):
            raise FlashBuildWorkerError("scratch is not durably settled")
    scratch = _private_dir(paths["scratch"])
    reclaimed = _safe_tree_bytes(scratch)
    if missing_preflight_token is not None:
        _revalidate_preflight_missing_cleanup_token(
            request, digest, missing_preflight_token
        )
    shutil.rmtree(scratch)
    return {"state": "cleaned", "reclaimed_bytes": reclaimed}


def main() -> int:
    if len(sys.argv) not in {4, 5}:
        print(json.dumps({"ok": False, "error": "invalid_arguments"}))
        return 64
    action, raw, digest = sys.argv[1:4]
    extra = sys.argv[4] if len(sys.argv) == 5 else None
    try:
        request = _validate_request(Path(raw), digest)
        if action == "pipeline":
            try:
                _pipeline(request)
            except BaseException as exc:
                _atomic_json(
                    _paths(request)["result"],
                    {
                        "schema_version": RESULT_SCHEMA,
                        "success": False,
                        "failure_type": type(exc).__name__,
                        "failure": str(exc)[:1000],
                        "completed_at": time.time(),
                    },
                )
                raise
            return 0
        if action == "preflight":
            result = _preflight(request, digest)
        elif action == "spawn":
            result = _spawn(request, digest)
        elif action == "status":
            result = _status(request)
        elif action == "stop":
            result = _stop(request)
        elif action == "settle-status":
            result = _settle_status(request)
        elif action == "mark-settled" and extra is not None:
            result = _mark_settled(request, extra)
        elif action == "cleanup" and extra is not None:
            result = _cleanup(request, extra)
        elif action == "cleanup-prelaunch":
            result = _cleanup(request, digest, prelaunch=True)
        else:
            raise FlashBuildWorkerError("invalid action or missing manifest digest")
    except BaseException as exc:
        print(
            json.dumps(
                {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:500]},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"ok": True, "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
