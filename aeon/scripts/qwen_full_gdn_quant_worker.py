#!/usr/bin/env python3
"""Worker-side lifecycle for the exact-ARA full-GDN NVFP4 conversion."""

from __future__ import annotations

import hashlib
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


SCHEMA_VERSION = "aeon-qwen38-full-gdn-quant-worker-v1"
RESULT_SCHEMA_VERSION = "aeon-qwen38-ara-modelopt-full-gdn-nvfp4-v1"
HOST = "192.168.0.180"
HOSTNAME = "DAY2XRTX5000PRO-2"
SCRATCH_ROOT = PurePosixPath("/home/aday/.local/state/fleet-compute/runs")
REMOTE_PYTHON = Path(
    "/home/aday/.local/share/uv/python/"
    "cpython-3.12-linux-x86_64-gnu/bin/python3.12"
)
ENGINE_SITE = Path(
    "/home/aday/.aeon/runtime/qwen38/engines/"
    "604c2525974bf41416e76c1f34ed014a1393d55617b4c7d7fc05d6c93754d9eb/"
    "venv/lib/python3.12/site-packages"
)
USER_SITE = Path("/home/aday/.local/lib/python3.12/site-packages")
SYSTEM_SITE = Path("/usr/lib/python3/dist-packages")
LOW_PRIORITY = Path("/home/aday/bin/fleet-low-priority")
BASH = Path("/usr/bin/bash")
MODELOPT_WHEEL_SHA256 = (
    "1864b4e9921e287b065be3861ab48345144e673273ebb2b94bd9a6119a9eba8e"
)
TEMPLATE_CONFIG_SHA256 = (
    "61a72634c98777cdb42c8f38485bbed79a903008405ea80f561f6f3ecf827fce"
)
TEMPLATE_SCALES_SHA256 = (
    "859fd870b5bf0579feb136a48dd868d422e37b1f4364eaeedef91b4fd62c92b3"
)
SOURCE_WEIGHT_SHA256 = {
    "model-00001-of-00006.safetensors": "55a4ad961830c6dfae435ba5d718c40dcd6169feb26c368007f6ba8c5f0329db",
    "model-00002-of-00006.safetensors": "74e61e9be6b6f6b02e8e0ae2f7f360f9df2c271860b4f52974ef64bdefcee274",
    "model-00003-of-00006.safetensors": "3c71faa739ce1f74875363b4ed0136a21da028208f11815471917ac2b17b50a7",
    "model-00004-of-00006.safetensors": "ebf94d3caa061031ce2f183adfca73128e7d7708aa78bebd55e1278358fc2f1b",
    "model-00005-of-00006.safetensors": "c843c9f461d6533eef4141d000b9e5a03fe328825346cbe7a062104b546a6e0c",
    "model-00006-of-00006.safetensors": "f7c99ba96930a0a4a8e7850660912ac71210622aa6e709d7df9af0e641d44451",
    "model-auxiliary.safetensors": "1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe",
}
SOURCE_METADATA_SHA256 = {
    "config.json": "5a1911420c23cca59e18efe1685e66f73fa6daee946ee43f4afc9a92f4bfc43d",
    "model.safetensors.index.json": "b0eb836dd3b5d2261cbf9e49913c02fd4e2ae886b0f2129e363c0a6156673d37",
    "tokenizer.json": "6f32ce20dc35f57a7f9ad1eac03525bd7d30f9df8cea6507e958279cc3657706",
    "tokenizer_config.json": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
}
_SHA = re.compile(r"^[a-f0-9]{64}$")
_RUNTIME = re.compile(r"^fr-[a-f0-9]{32}$")
_CLAIM = re.compile(r"^gc-[A-Za-z0-9._:-]{8,200}$")
_OWNER = re.compile(r"^[A-Za-z0-9._:-]{3,240}$")
_UUID = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$"
)
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


class QuantWorkerError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _private_directory(path: Path, *, create: bool = False) -> Path:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise QuantWorkerError(f"private directory identity changed: {path}")
    return path


def _atomic_json(path: Path, value: Any) -> None:
    _private_directory(path.parent, create=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _read_json(path: Path, maximum: int = 8 * 1024 * 1024) -> dict[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= maximum
    ):
        raise QuantWorkerError(f"private JSON identity changed: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuantWorkerError(f"private JSON is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise QuantWorkerError("private JSON is not an object")
    return value


def _paths(request: dict[str, Any]) -> dict[str, Path]:
    scratch = Path(request["scratch_path"])
    output = scratch / "output"
    fixtures = scratch / "fixtures"
    return {
        "scratch": scratch,
        "source_root": Path(request["source_root"]),
        "input": scratch / "input/source",
        "fixtures": fixtures,
        "template_config": fixtures / "template-config.json",
        "template_scales": fixtures / "template-scales.safetensors",
        "modelopt_wheel": fixtures / "nvidia_modelopt-0.46.0-py3-none-any.whl",
        "overlay": scratch / "modelopt-overlay",
        "request": scratch / "qwen-full-gdn-quant-request.json",
        "output": output,
        "model": output / "model",
        "preflight": output / "preflight.json",
        "result": output / "result.json",
        "log": output / "quantization.log",
        "spawn": scratch / "spawn.json",
        "manifest": output / "MANIFEST.sha256",
        "settled": scratch / "settled.json",
    }


def _validate_request(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _SHA.fullmatch(expected_sha256) is None:
        raise QuantWorkerError("request digest is malformed")
    request = _read_json(path)
    if _sha256(path) != expected_sha256:
        raise QuantWorkerError("request bytes changed")
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
        "min_disk_free_gb",
        "min_shm_free_gb",
        "scratch_path",
        "source_root",
        "source_files",
        "source_weight_sha256",
        "source_metadata_sha256",
        "source_tree_sha256",
        "template_config_sha256",
        "template_scales_sha256",
        "modelopt_wheel_sha256",
    }
    runtime = request.get("runtime_id")
    scratch = PurePosixPath(str(request.get("scratch_path") or ""))
    source_root = PurePosixPath(str(request.get("source_root") or ""))
    if (
        set(request) != required
        or request.get("schema_version") != SCHEMA_VERSION
        or not isinstance(runtime, str)
        or _RUNTIME.fullmatch(runtime) is None
        or scratch.parent != SCRATCH_ROOT
        or scratch.name != runtime
        or source_root != scratch / "source"
        or path != Path(scratch) / "qwen-full-gdn-quant-request.json"
        or request.get("host") != HOST
        or request.get("hostname") != HOSTNAME
        or socket.gethostname() != HOSTNAME
        or not isinstance(request.get("job_id"), str)
        or not request["job_id"]
        or _CLAIM.fullmatch(str(request.get("claim_id") or "")) is None
        or _OWNER.fullmatch(str(request.get("owner") or "")) is None
        or request.get("physical_gpu") not in {0, 1}
        or isinstance(request.get("physical_gpu"), bool)
        or _UUID.fullmatch(str(request.get("gpu_uuid") or "")) is None
        or request.get("exclusive") is not True
        or float(request.get("vram_budget_gb") or 0) != 41.25
        or float(request.get("min_host_memory_gb") or 0) != 80.0
        or float(request.get("min_host_commit_gb") or 0) != 80.0
        or float(request.get("min_disk_free_gb") or 0) != 96.0
        or float(request.get("min_shm_free_gb") or 0) != 16.0
        or request.get("source_weight_sha256") != SOURCE_WEIGHT_SHA256
        or request.get("source_metadata_sha256") != SOURCE_METADATA_SHA256
        or request.get("source_tree_sha256")
        != _canonical_sha256(
            {"weights": SOURCE_WEIGHT_SHA256, "metadata": SOURCE_METADATA_SHA256}
        )
        or request.get("template_config_sha256") != TEMPLATE_CONFIG_SHA256
        or request.get("template_scales_sha256") != TEMPLATE_SCALES_SHA256
        or request.get("modelopt_wheel_sha256") != MODELOPT_WHEEL_SHA256
        or not isinstance(request.get("source_files"), dict)
    ):
        raise QuantWorkerError("quantization request differs from its reviewed schema")
    return request


def _verify_regular(
    path: Path,
    digest: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> int:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or metadata.st_size > maximum
        or (metadata.st_size == 0 and not allow_empty)
        or _sha256(path) != digest
    ):
        raise QuantWorkerError(f"staged file identity changed: {path}")
    return metadata.st_size


def _verify_sources(request: dict[str, Any]) -> int:
    root = _private_directory(_paths(request)["source_root"])
    total = 0
    for relative, digest in sorted(request["source_files"].items()):
        if _SAFE_RELATIVE.fullmatch(relative) is None or _SHA.fullmatch(digest) is None:
            raise QuantWorkerError("source manifest is malformed")
        # Python package markers may intentionally be empty. Their exact empty
        # digest is still bound by the manifest, so permitting zero bytes here
        # does not weaken source identity verification.
        total += _verify_regular(
            root / relative,
            digest,
            maximum=2 * 1024 * 1024,
            allow_empty=True,
        )
    return total


def _verify_input(request: dict[str, Any]) -> int:
    root = _private_directory(_paths(request)["input"])
    total = 0
    for name, digest in {**SOURCE_WEIGHT_SHA256, **SOURCE_METADATA_SHA256}.items():
        total += _verify_regular(root / name, digest, maximum=11 * 1024**3)
    reproduction = root / "reproduce/reproduce.json"
    metadata = reproduction.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or metadata.st_size > 128 * 1024
    ):
        raise QuantWorkerError("source reproduction receipt is unsafe")
    receipt = json.loads(reproduction.read_text(encoding="utf-8"))
    if receipt.get("weights_sha256") != SOURCE_WEIGHT_SHA256:
        raise QuantWorkerError("source reproduction weight receipt changed")
    return total


def _extract_overlay(request: dict[str, Any]) -> None:
    paths = _paths(request)
    _verify_regular(paths["modelopt_wheel"], MODELOPT_WHEEL_SHA256, maximum=4 * 1024**2)
    if paths["overlay"].exists():
        _private_directory(paths["overlay"])
        return
    temporary = paths["overlay"].with_name(".modelopt-overlay.partial")
    if temporary.exists():
        raise QuantWorkerError("stale ModelOpt overlay partial exists")
    _private_directory(temporary, create=True)
    with zipfile.ZipFile(paths["modelopt_wheel"]) as archive:
        for member in archive.infolist():
            target = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (
                target.is_absolute()
                or ".." in target.parts
                or stat.S_ISLNK(mode)
                or not (member.is_dir() or stat.S_ISREG(mode) or mode == 0)
            ):
                raise QuantWorkerError("ModelOpt wheel contains an unsafe member")
        archive.extractall(temporary)
    for item in temporary.rglob("*"):
        if item.is_dir():
            item.chmod(0o700)
        elif item.is_file():
            item.chmod(0o600)
        else:
            raise QuantWorkerError("ModelOpt overlay contains an unsafe inode")
    temporary.rename(paths["overlay"])


def _verify_environment(request: dict[str, Any]) -> dict[str, str]:
    _extract_overlay(request)
    paths = _paths(request)
    pythonpath = ":".join(
        str(path)
        for path in (paths["overlay"], ENGINE_SITE, USER_SITE, SYSTEM_SITE)
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/aday",
        "LANG": "C",
        "LC_ALL": "C",
        "CUDA_VISIBLE_DEVICES": "void",
        "PYTHONPATH": pythonpath,
        "PYTHONNOUSERSITE": "1",
        "USE_TF": "0",
        "USE_FLAX": "0",
    }
    script = (
        "import json,modelopt,safetensors,torch;"
        "print(json.dumps({'modelopt':modelopt.__version__,'safetensors':safetensors.__version__,'torch':torch.__version__},sort_keys=True))"
    )
    result = subprocess.run(
        [str(REMOTE_PYTHON), "-c", script],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
    )
    try:
        versions = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise QuantWorkerError("quantization environment receipt is malformed") from exc
    expected = {
        "modelopt": "0.46.0",
        "safetensors": "0.7.0",
        "torch": "2.9.1+cu128",
    }
    if result.returncode != 0 or versions != expected:
        raise QuantWorkerError("quantization environment versions changed")
    return versions


def _verify_acl(request: dict[str, Any]) -> None:
    device = Path(f"/dev/nvidia{request['physical_gpu']}")
    result = subprocess.run(
        ["/usr/bin/getfacl", "-cp", "--", str(device)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0 or "user:aday:---" in result.stdout.splitlines():
        raise QuantWorkerError("leased GPU is renter-blocked or ambiguous")


def _host_resources(request: dict[str, Any]) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].rstrip(":") in {
            "MemAvailable",
            "CommitLimit",
            "Committed_AS",
        }:
            values[fields[0].rstrip(":")] = int(fields[1]) * 1024
    if set(values) != {"MemAvailable", "CommitLimit", "Committed_AS"}:
        raise QuantWorkerError("host memory accounting is unavailable")
    scratch_stats = os.statvfs(_paths(request)["scratch"])
    shm_stats = os.statvfs("/dev/shm")
    receipt = {
        "memory_available_bytes": values["MemAvailable"],
        "commit_available_bytes": values["CommitLimit"] - values["Committed_AS"],
        "disk_free_bytes": scratch_stats.f_bavail * scratch_stats.f_frsize,
        "disk_free_inodes": scratch_stats.f_favail,
        "shm_free_bytes": shm_stats.f_bavail * shm_stats.f_frsize,
    }
    gib = 1024**3
    floors = {
        "memory_available_bytes": int(request["min_host_memory_gb"] * gib),
        "commit_available_bytes": int(request["min_host_commit_gb"] * gib),
        "disk_free_bytes": int(request["min_disk_free_gb"] * gib),
        "shm_free_bytes": int(request["min_shm_free_gb"] * gib),
    }
    if any(receipt[name] < floor for name, floor in floors.items()):
        raise QuantWorkerError("host resources changed after reservation")
    return receipt


def _preflight(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    paths = _paths(request)
    _private_directory(paths["scratch"])
    _private_directory(paths["output"], create=True)
    source_bytes = _verify_sources(request)
    input_bytes = _verify_input(request)
    _private_directory(paths["fixtures"])
    _verify_regular(
        paths["template_config"], TEMPLATE_CONFIG_SHA256, maximum=128 * 1024
    )
    _verify_regular(
        paths["template_scales"], TEMPLATE_SCALES_SHA256, maximum=256 * 1024
    )
    versions = _verify_environment(request)
    _verify_acl(request)
    resources = _host_resources(request)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "source_tree_sha256": request["source_tree_sha256"],
        "source_bytes": source_bytes,
        "input_bytes": input_bytes,
        "template_config_sha256": TEMPLATE_CONFIG_SHA256,
        "template_scales_sha256": TEMPLATE_SCALES_SHA256,
        "modelopt_wheel_sha256": MODELOPT_WHEEL_SHA256,
        "environment": versions,
        "host_resources": resources,
        "verified_at": time.time(),
    }
    _atomic_json(paths["preflight"], receipt)
    return {"state": "preflight_ready", **receipt}


def _process_alive(request: dict[str, Any], pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    proc = Path(f"/proc/{pid}")
    for attempt in range(3):
        try:
            environment = (proc / "environ").read_bytes().split(b"\0")
            command = (proc / "cmdline").read_bytes().split(b"\0")
            break
        except (FileNotFoundError, ProcessLookupError):
            return False
        except OSError as exc:
            if not proc.exists():
                return False
            if attempt < 2:
                time.sleep(0.02)
                continue
            raise QuantWorkerError("quantization process identity is unreadable") from exc
    paths = _paths(request)
    return (
        f"GPU_AGENT_CLAIM_ID={request['claim_id']}".encode() in environment
        and f"CUDA_VISIBLE_DEVICES={request['gpu_uuid']}".encode() in environment
        and b"GPU_MEM_LIMIT_GB=41.25" in environment
        and f"AEON_QUANT_RUNTIME_ID={request['runtime_id']}".encode() in environment
        and b"aeon.scripts.build_qwen38_full_gdn_nvfp4" in command
        and str(paths["input"]).encode() in command
        and str(paths["model"]).encode() in command
    )


def _spawn(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    paths = _paths(request)
    preflight = _read_json(paths["preflight"])
    if preflight.get("request_sha256") != request_sha256:
        raise QuantWorkerError("quantization preflight receipt is stale")
    if paths["spawn"].exists() or paths["result"].exists() or paths["model"].exists():
        raise QuantWorkerError("quantization lifecycle already exists")
    pythonpath = ":".join(
        str(path)
        for path in (
            paths["source_root"],
            paths["overlay"],
            ENGINE_SITE,
            USER_SITE,
            SYSTEM_SITE,
        )
    )
    environment = {
        "PATH": "/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/home/aday",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": pythonpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "USE_TF": "0",
        "USE_FLAX": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "8",
        "GPU_AGENT_CLAIM_ID": request["claim_id"],
        "CUDA_VISIBLE_DEVICES": request["gpu_uuid"],
        "GPU_MEM_LIMIT_GB": "41.25",
        "AEON_QUANT_RUNTIME_ID": request["runtime_id"],
        "AEON_QUANT_RESULT_PATH": str(paths["result"]),
    }
    descriptor = os.open(
        paths["log"], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    old_umask = os.umask(0o077)
    try:
        process = subprocess.Popen(
            [
                str(BASH),
                str(LOW_PRIORITY),
                str(REMOTE_PYTHON),
                "-m",
                "aeon.scripts.build_qwen38_full_gdn_nvfp4",
                "--source",
                str(paths["input"]),
                "--template-config",
                str(paths["template_config"]),
                "--template-scales",
                str(paths["template_scales"]),
                "--output",
                str(paths["model"]),
            ],
            cwd=paths["source_root"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.umask(old_umask)
        os.close(descriptor)
    _atomic_json(
        paths["spawn"],
        {
            "schema_version": SCHEMA_VERSION,
            "runtime_id": request["runtime_id"],
            "request_sha256": request_sha256,
            "pid": process.pid,
            "created_at": time.time(),
        },
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if _process_alive(request, process.pid):
            return {"state": "running", "pid": process.pid}
        if process.poll() is not None:
            raise QuantWorkerError("quantization process exited during spawn")
        time.sleep(0.1)
    raise QuantWorkerError("quantization process identity did not become visible")


def _spawn_receipt(request: dict[str, Any]) -> dict[str, Any] | None:
    path = _paths(request)["spawn"]
    if not path.is_file():
        return None
    receipt = _read_json(path)
    if (
        receipt.get("runtime_id") != request["runtime_id"]
        or receipt.get("request_sha256") != _sha256(_paths(request)["request"])
    ):
        raise QuantWorkerError("quantization spawn receipt identity changed")
    return receipt


def _terminal(request: dict[str, Any], reason: str) -> dict[str, Any]:
    path = _paths(request)["result"]
    if path.is_file():
        result = _read_json(path)
    else:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "success": False,
            "failure_type": "ProcessExited",
            "failure": reason[:1000],
            "completed_at": time.time(),
        }
        _atomic_json(path, result)
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise QuantWorkerError("quantization terminal schema changed")
    return result


def _status(request: dict[str, Any]) -> dict[str, Any]:
    receipt = _spawn_receipt(request)
    pid = receipt.get("pid") if receipt else None
    if isinstance(pid, int) and _process_alive(request, pid):
        return {"state": "running", "pid": pid, "phase": "quantizing_full_gdn"}
    if receipt is None:
        return {"state": "absent", "pid": None, "phase": "not_spawned"}
    result = _terminal(request, "quantization process exited without a terminal receipt")
    return {
        "state": "completed" if result.get("success") is True else "failed",
        "pid": pid,
        "phase": "terminal",
        "result": result,
    }


def _stop(request: dict[str, Any]) -> dict[str, Any]:
    receipt = _spawn_receipt(request)
    pid = receipt.get("pid") if receipt else None
    if isinstance(pid, int) and _process_alive(request, pid):
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and _process_alive(request, pid):
            time.sleep(0.5)
        if _process_alive(request, pid):
            return {"state": "ambiguous", "process_absent": False}
    _terminal(request, "quantization was stopped before successful completion")
    return {"state": "stopped", "process_absent": True}


def _write_manifest(request: dict[str, Any]) -> str:
    paths = _paths(request)
    output = _private_directory(paths["output"])
    manifest = paths["manifest"]
    if manifest.exists():
        return _sha256(manifest)
    result = _terminal(request, "quantization ended before output settlement")
    if result.get("success") is True:
        model = _private_directory(paths["model"])
        build = _read_json(model / "BUILD_MANIFEST.json")
        if build.get("schema_version") != RESULT_SCHEMA_VERSION or build.get("complete") is not True:
            raise QuantWorkerError("quantized model build receipt is invalid")
    entries: list[tuple[str, str]] = []
    total_bytes = 0
    for item in sorted(output.rglob("*")):
        metadata = item.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise QuantWorkerError("quantization output directory is mutable")
            continue
        relative = item.relative_to(output).as_posix()
        if (
            item == manifest
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_ISLNK(metadata.st_mode)
            or _SAFE_RELATIVE.fullmatch(relative) is None
        ):
            raise QuantWorkerError("quantization output contains an unsafe inode")
        total_bytes += metadata.st_size
        if total_bytes > 22 * 1024**3:
            raise QuantWorkerError("quantization output exceeded its growth bound")
        entries.append((_sha256(item), relative))
    if not any(relative == "result.json" for _digest, relative in entries):
        raise QuantWorkerError("quantization output lacks its terminal receipt")
    temporary = manifest.with_name(".MANIFEST.sha256.tmp")
    temporary.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in entries),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, manifest)
    return _sha256(manifest)


def _settle_status(request: dict[str, Any]) -> dict[str, Any]:
    receipt = _spawn_receipt(request)
    pid = receipt.get("pid") if receipt else None
    if isinstance(pid, int) and _process_alive(request, pid):
        raise QuantWorkerError("quantization output cannot settle while process is alive")
    result = _terminal(request, "quantization ended before output settlement")
    manifest_sha256 = _write_manifest(request)
    files = []
    for line in _paths(request)["manifest"].read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        if match is None or _SAFE_RELATIVE.fullmatch(match.group(2)) is None:
            raise QuantWorkerError("quantization output manifest is malformed")
        path = _paths(request)["output"] / match.group(2)
        if _sha256(path) != match.group(1):
            raise QuantWorkerError("quantization output changed after manifest")
        files.append(
            {"name": match.group(2), "sha256": match.group(1), "bytes": path.stat().st_size}
        )
    return {
        "state": "settle_ready",
        "manifest_sha256": manifest_sha256,
        "files": files,
        "result": result,
    }


def _mark_settled(request: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    status = _settle_status(request)
    if status["manifest_sha256"] != manifest_sha256:
        raise QuantWorkerError("settled output manifest identity changed")
    _atomic_json(
        _paths(request)["settled"],
        {
            "schema_version": SCHEMA_VERSION,
            "runtime_id": request["runtime_id"],
            "manifest_sha256": manifest_sha256,
            "settled_at": time.time(),
        },
    )
    return {"state": "settled", "manifest_sha256": manifest_sha256}


def _cleanup(request: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    paths = _paths(request)
    marker = _read_json(paths["settled"])
    receipt = _spawn_receipt(request)
    pid = receipt.get("pid") if receipt else None
    if (
        marker.get("runtime_id") != request["runtime_id"]
        or marker.get("manifest_sha256") != manifest_sha256
        or isinstance(pid, int)
        and _process_alive(request, pid)
        or _sha256(paths["manifest"]) != manifest_sha256
    ):
        raise QuantWorkerError("quantization scratch is not safe to clean")
    scratch = _private_directory(paths["scratch"])
    reclaimed = 0
    for item in scratch.rglob("*"):
        metadata = item.lstat()
        if (
            metadata.st_uid != os.geteuid()
            or stat.S_ISLNK(metadata.st_mode)
            or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode))
        ):
            raise QuantWorkerError("quantization scratch contains an unsafe inode")
        reclaimed += metadata.st_size
    shutil.rmtree(scratch)
    return {"state": "cleaned", "reclaimed_bytes": reclaimed}


def _cleanup_prelaunch(request: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(request)
    for path in (
        paths["spawn"],
        paths["settled"],
        paths["result"],
        paths["manifest"],
        paths["model"],
    ):
        if path.exists() or path.is_symlink():
            raise QuantWorkerError("quantization prelaunch cleanup found lifecycle output")
    scratch = _private_directory(paths["scratch"])
    reclaimed = 0
    for item in scratch.rglob("*"):
        metadata = item.lstat()
        if (
            metadata.st_uid != os.geteuid()
            or stat.S_ISLNK(metadata.st_mode)
            or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode))
        ):
            raise QuantWorkerError("quantization prelaunch scratch contains an unsafe inode")
        reclaimed += metadata.st_size
    shutil.rmtree(scratch)
    return {"state": "cleaned", "reclaimed_bytes": reclaimed}


def main() -> int:
    if len(sys.argv) not in {4, 5}:
        print(json.dumps({"ok": False, "error": "invalid_arguments"}))
        return 64
    action, raw_path, request_sha256 = sys.argv[1:4]
    extra = sys.argv[4] if len(sys.argv) == 5 else None
    allowed = {
        "preflight",
        "spawn",
        "status",
        "stop",
        "settle-status",
        "mark-settled",
        "cleanup",
        "cleanup-prelaunch",
    }
    if action not in allowed:
        print(json.dumps({"ok": False, "error": "invalid_action"}))
        return 64
    try:
        request = _validate_request(Path(raw_path), request_sha256)
        if action == "preflight":
            result = _preflight(request, request_sha256)
        elif action == "spawn":
            result = _spawn(request, request_sha256)
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
            result = _cleanup_prelaunch(request)
        else:
            raise QuantWorkerError("action requires an exact manifest digest")
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
