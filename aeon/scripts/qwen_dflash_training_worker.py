#!/usr/bin/env python3
"""Worker-side lifecycle supervisor for exact-target DFlash2 adaptation."""

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


SCHEMA_VERSION = "aeon-qwen38-dflash-training-worker-v2"
HOST = "192.168.0.179"
HOSTNAME = "DAY2XRTX6000-2"
SCRATCH_ROOT = PurePosixPath("/home/aday/.local/state/fleet-compute/runs")
LOW_PRIORITY = Path("/home/aday/bin/fleet-low-priority")
BASH = Path("/usr/bin/bash")
ENV_PYTHON = Path(
    "/home/aday/.aeon/runtime/qwen38/training-envs/"
    "nemo-9fb92970-torch291-cu128/bin/python"
)
ENV_SITE = Path(
    "/home/aday/.aeon/runtime/qwen38/training-envs/"
    "nemo-9fb92970-torch291-cu128/lib/python3.12/site-packages"
)
ENV_WHEEL = Path(
    "/home/aday/.aeon/runtime/qwen38/training-sources/wheels/"
    "causal_conv1d-1.7.0+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
ENV_WHEEL_SHA256 = (
    "b7e2f034c5e230432cebb0f98eda4e01557536c4c37ea01a2c3a1a8ae5166954"
)
TARGET_DIR = Path(
    "/home/aday/.aeon/models/.speed-sources/"
    "Qwen3.8-27B-heretic-ara-a67ae100d933c0d17af3232bda35825979fc63ce"
)
DRAFT_DIR = Path(
    "/home/aday/.aeon/runtime/qwen38/drafts/"
    "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"
)
TARGET_FILES = {
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "config.json": "5a1911420c23cca59e18efe1685e66f73fa6daee946ee43f4afc9a92f4bfc43d",
    "model-00001-of-00006.safetensors": "55a4ad961830c6dfae435ba5d718c40dcd6169feb26c368007f6ba8c5f0329db",
    "model-00002-of-00006.safetensors": "74e61e9be6b6f6b02e8e0ae2f7f360f9df2c271860b4f52974ef64bdefcee274",
    "model-00003-of-00006.safetensors": "3c71faa739ce1f74875363b4ed0136a21da028208f11815471917ac2b17b50a7",
    "model-00004-of-00006.safetensors": "ebf94d3caa061031ce2f183adfca73128e7d7708aa78bebd55e1278358fc2f1b",
    "model-00005-of-00006.safetensors": "c843c9f461d6533eef4141d000b9e5a03fe328825346cbe7a062104b546a6e0c",
    "model-00006-of-00006.safetensors": "f7c99ba96930a0a4a8e7850660912ac71210622aa6e709d7df9af0e641d44451",
    "model.safetensors.index.json": "b0eb836dd3b5d2261cbf9e49913c02fd4e2ae886b0f2129e363c0a6156673d37",
    "tokenizer.json": "6f32ce20dc35f57a7f9ad1eac03525bd7d30f9df8cea6507e958279cc3657706",
    "tokenizer_config.json": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
}
DRAFT_FILES = {
    "config.json": "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980",
    "model.safetensors": "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c",
}
ENV_SENTINELS = {
    "causal_conv1d/__init__.py": "26b92128a1387720c5da882b25a9668f1a340a5707deb0a124106e4a308bc250",
    "causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so": "90f05d99a990dcfe71dbe746f3c38cb2d22bda6f022b8226d9f548b372e94c6f",
    "datasets/__init__.py": "71decfe26d1df7b8eed4c7b0be6dab3ea58ac4c370b4b77042ad03ae7d2e5554",
    "datasets/packaged_modules/json/json.py": "a906fb371f281161a35a8b930e5f843ea8dbf93df4f163c4b1230832af8651e0",
    "torch/version.py": "fbd6ecefcebde9cfb435c195a1a58a26a904fdbb62c3bbc9ed84d606215b3d75",
    "torch/lib/libtorch_cuda.so": "2c28bf7f912baa8c1849f1c3eda6f90f2fc6c9dfd22326d183c6f6daabab4fd2",
    "torch/lib/libc10_cuda.so": "d61548fae97498e6f0e99418a1d1218ce209ba0be66aaba7da1ee6fac8f506ad",
    "transformers/models/qwen3_5/modeling_qwen3_5.py": "395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6",
    "nemo_automodel/recipes/llm/train_dflash.py": "3897ebd38f08a8a31ac0299e8c38947b12e8c9127bcaf0120ae1209a9c9413b3",
    "nemo_automodel/recipes/llm/train_dflash2.py": "f9e0e9eb5cc6deee9233c514615e8e48806d587ca84451225b1800affae330f4",
    "nemo_automodel/components/speculative/dflash/draft_qwen3_dflash2.py": "594e62242c07c40c3d19a8444de384d44240cc6c48db432a7d044d4d595109a4",
    "nemo_automodel/components/speculative/dflash/target.py": "96f02cd68496f30eb07ae96b7ffcd4875ac64bb7f48f87b16db6b08f072e699f",
    "fla/ops/gated_delta_rule/__init__.py": "5edd21847af70840b26439812196a476e7e4ddbbef21afd5fe2066a64630a67c",
    "pandas/__init__.py": "742f85c135ee654bff1203efcee7664bf82fc7305ac3912ff304a063f5f7ea3c",
    "pyarrow/__init__.py": "04ef85e330eb3cda057b5f4f8e9c06891536bf06d7b26341d392153ecaa861e2",
    "pyarrow/lib.cpython-312-x86_64-linux-gnu.so": "62bf46ed6658e9ecd315be20d291af37313d07e72a895fe10cd561cbf9680260",
    "triton/__init__.py": "96e68192cab5dd11fa30d5dc95d56c29ceecda53477fc3af8f37bee026b7a021",
}
ENV_VERSIONS = {
    "causal-conv1d": "1.7.0",
    "datasets": "4.8.5",
    "dill": "0.4.1",
    "flash-linear-attention": "0.5.2",
    "multiprocess": "0.70.19",
    "nemo-automodel": "0.5.0",
    "pandas": "3.0.0",
    "pyarrow": "24.0.0",
    "python-dateutil": "2.9.0.post0",
    "safetensors": "0.8.0",
    "six": "1.17.0",
    "torch": "2.10.0+cu130",
    "torchdata": "0.11.0",
    "transformers": "5.12.1",
    "xxhash": "3.7.0",
}
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_RUNTIME_RE = re.compile(r"^fr-[a-f0-9]{32}$")
_CLAIM_RE = re.compile(r"^gc-[A-Za-z0-9._:-]{8,200}$")
_OWNER_RE = re.compile(r"^[A-Za-z0-9._:-]{3,240}$")
_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$"
)
_SAFE_RELATIVE_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


class TrainingWorkerError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise TrainingWorkerError(f"private directory identity changed: {path}")
    return path


def _atomic_json(path: Path, value: Any) -> None:
    _private_directory(path.parent, create=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        os.write(descriptor, payload)
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
        raise TrainingWorkerError(f"private JSON identity changed: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingWorkerError(f"private JSON is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise TrainingWorkerError("private JSON is not an object")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _paths(request: dict[str, Any]) -> dict[str, Path]:
    scratch = Path(request["scratch_path"])
    output = scratch / "output"
    return {
        "scratch": scratch,
        "source": Path(request["source_root"]),
        "dataset": scratch / "train.jsonl",
        "config": scratch / "training-config.json",
        "request": scratch / "dflash-training-request.json",
        "output": output,
        "preflight": output / "preflight.json",
        "log": output / "training.log",
        "result": output / "result.json",
        "manifest": output / "MANIFEST.sha256",
        "spawn": scratch / "spawn.json",
        "settled": scratch / "settled.json",
    }


def _validate_request(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _SHA_RE.fullmatch(expected_sha256) is None:
        raise TrainingWorkerError("request digest is malformed")
    request = _read_json(path)
    if _sha256(path) != expected_sha256:
        raise TrainingWorkerError("request bytes changed")
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
        "target_dir",
        "target_files",
        "target_tree_sha256",
        "draft_dir",
        "draft_files",
        "environment_python",
        "environment_sentinels",
        "environment_versions",
        "environment_sha256",
        "environment_wheel_sha256",
        "dataset_sha256",
        "dataset_bytes",
        "dataset_rows",
        "canonical_dataset_sha256",
        "training_config_sha256",
        "run_mode",
    }
    runtime_id = request.get("runtime_id")
    scratch = PurePosixPath(str(request.get("scratch_path") or ""))
    source = PurePosixPath(str(request.get("source_root") or ""))
    if (
        set(request) != required
        or request.get("schema_version") != SCHEMA_VERSION
        or not isinstance(runtime_id, str)
        or _RUNTIME_RE.fullmatch(runtime_id) is None
        or scratch.parent != SCRATCH_ROOT
        or scratch.name != runtime_id
        or source != scratch / "source"
        or path != Path(scratch) / "dflash-training-request.json"
        or request.get("host") != HOST
        or request.get("hostname") != HOSTNAME
        or socket.gethostname() != HOSTNAME
        or not isinstance(request.get("job_id"), str)
        or not request["job_id"]
        or _CLAIM_RE.fullmatch(str(request.get("claim_id") or "")) is None
        or _OWNER_RE.fullmatch(str(request.get("owner") or "")) is None
        or request.get("physical_gpu") not in {0, 1}
        or isinstance(request.get("physical_gpu"), bool)
        or _UUID_RE.fullmatch(str(request.get("gpu_uuid") or "")) is None
        or request.get("exclusive") is not True
        or float(request.get("vram_budget_gb") or 0) != 88.0
        or float(request.get("min_host_memory_gb") or 0) != 100.0
        or float(request.get("min_host_commit_gb") or 0) != 100.0
        or float(request.get("min_disk_free_gb") or 0) != 47.0
        or float(request.get("min_shm_free_gb") or 0) != 16.0
        or request.get("target_dir") != str(TARGET_DIR)
        or request.get("target_files") != TARGET_FILES
        or request.get("target_tree_sha256") != _canonical_sha256(TARGET_FILES)
        or request.get("draft_dir") != str(DRAFT_DIR)
        or request.get("draft_files") != DRAFT_FILES
        or request.get("environment_python") != str(ENV_PYTHON)
        or request.get("environment_sentinels") != ENV_SENTINELS
        or request.get("environment_versions") != ENV_VERSIONS
        or request.get("environment_sha256")
        != _canonical_sha256({"sentinels": ENV_SENTINELS, "versions": ENV_VERSIONS})
        or request.get("environment_wheel_sha256") != ENV_WHEEL_SHA256
        or _SHA_RE.fullmatch(str(request.get("dataset_sha256") or "")) is None
        or _SHA_RE.fullmatch(str(request.get("canonical_dataset_sha256") or "")) is None
        or _SHA_RE.fullmatch(str(request.get("training_config_sha256") or "")) is None
        or request.get("canonical_dataset_sha256")
        != "61b8e150651ecc14c47e1068ce36fc130bb56e18117b3b68e098390defea92f5"
        or request.get("run_mode")
        not in {
            "smoke",
            "calibrate",
            "adapt-v1",
            "calibrate-full",
            "adapt-full-v1",
            "adapt-full-dpace-v2",
        }
        or not isinstance(request.get("dataset_rows"), int)
        or request["dataset_rows"]
        != {
            "smoke": 1,
            "calibrate": 1,
            "adapt-v1": 64,
            "calibrate-full": 1,
            "adapt-full-v1": 64,
            "adapt-full-dpace-v2": 256,
        }[request["run_mode"]]
        or not isinstance(request.get("dataset_bytes"), int)
        or not 1 <= request["dataset_bytes"] <= 16 * 1024 * 1024
    ):
        raise TrainingWorkerError("training request identity changed")
    sources = request.get("source_files")
    if not isinstance(sources, dict) or not sources:
        raise TrainingWorkerError("training source manifest is empty")
    for relative, digest in sources.items():
        if (
            not isinstance(relative, str)
            or _SAFE_RELATIVE_RE.fullmatch(relative) is None
            or _SHA_RE.fullmatch(str(digest)) is None
        ):
            raise TrainingWorkerError("training source manifest is malformed")
    return request


def _verify_regular(
    path: Path,
    digest: str,
    *,
    maximum: int | None = None,
    uv_environment: bool = False,
) -> int:
    metadata = path.lstat()
    write_mask = 0o002 if uv_environment else 0o022
    links_safe = 1 <= metadata.st_nlink <= 2 if uv_environment else metadata.st_nlink == 1
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & write_mask
        or not links_safe
        or (maximum is not None and metadata.st_size > maximum)
    ):
        raise TrainingWorkerError(f"artifact identity is unsafe: {path}")
    if _sha256(path) != digest:
        raise TrainingWorkerError(f"artifact digest changed: {path.name}")
    return metadata.st_size


def _verify_tree(
    root: Path, expected: dict[str, str], *, uv_environment: bool = False
) -> int:
    if not root.is_dir() or root.is_symlink():
        raise TrainingWorkerError(f"artifact root is unsafe: {root}")
    total = 0
    for relative, digest in expected.items():
        total += _verify_regular(
            root / relative, digest, uv_environment=uv_environment
        )
    return total


def _verify_sources(request: dict[str, Any]) -> None:
    root = _private_directory(Path(request["source_root"]))
    actual: set[str] = set()
    for item in root.rglob("*"):
        metadata = item.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
                raise TrainingWorkerError("source directory is mutable")
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            raise TrainingWorkerError("source tree contains an unsafe inode")
        actual.add(item.relative_to(root).as_posix())
    if actual != set(request["source_files"]):
        raise TrainingWorkerError("source tree file set changed")
    for relative, digest in request["source_files"].items():
        if _sha256(root / relative) != digest:
            raise TrainingWorkerError(f"source digest changed: {relative}")


def _verify_environment(request: dict[str, Any]) -> None:
    if not ENV_PYTHON.is_file() or not os.access(ENV_PYTHON, os.X_OK):
        raise TrainingWorkerError("training interpreter is absent")
    # uv installs immutable wheels as two-link files into a group-writable
    # owner tree. Bind those exact bytes and package versions while still
    # rejecting world-writable files, extra hardlinks, or a changed owner.
    _verify_tree(
        ENV_SITE, request["environment_sentinels"], uv_environment=True
    )
    _verify_regular(ENV_WHEEL, ENV_WHEEL_SHA256, maximum=300 * 1024 * 1024)
    script = (
        "import importlib.metadata,json,torch;"
        "names=" + repr(sorted(ENV_VERSIONS)) + ";"
        "print(json.dumps({n:importlib.metadata.version(n) for n in names},sort_keys=True));"
        "print(torch.__version__)"
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/aday",
        "LANG": "C",
        "LC_ALL": "C",
        "CUDA_VISIBLE_DEVICES": "void",
        "USE_TF": "0",
        "USE_FLAX": "0",
        "PYTHONNOUSERSITE": "1",
    }
    result = subprocess.run(
        [str(ENV_PYTHON), "-c", script],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = result.stdout.splitlines()
    try:
        versions = json.loads(lines[0])
    except (IndexError, json.JSONDecodeError) as exc:
        raise TrainingWorkerError("training environment receipt is malformed") from exc
    if result.returncode != 0 or versions != ENV_VERSIONS or lines[1] != "2.10.0+cu130":
        raise TrainingWorkerError("training environment versions changed")


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
        raise TrainingWorkerError("leased GPU is renter-blocked or ambiguous")


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
        raise TrainingWorkerError("host memory accounting is unavailable")
    commit_available = values["CommitLimit"] - values["Committed_AS"]
    scratch_stats = os.statvfs(_paths(request)["scratch"])
    shm_stats = os.statvfs("/dev/shm")
    receipt = {
        "memory_available_bytes": values["MemAvailable"],
        "commit_available_bytes": commit_available,
        "disk_free_bytes": scratch_stats.f_bavail * scratch_stats.f_frsize,
        "disk_free_inodes": scratch_stats.f_favail,
        "shm_free_bytes": shm_stats.f_bavail * shm_stats.f_frsize,
    }
    gib = 1024**3
    required = {
        "memory_available_bytes": int(request["min_host_memory_gb"] * gib),
        "commit_available_bytes": int(request["min_host_commit_gb"] * gib),
        "disk_free_bytes": int(request["min_disk_free_gb"] * gib),
        "shm_free_bytes": int(request["min_shm_free_gb"] * gib),
    }
    if any(receipt[name] < minimum for name, minimum in required.items()):
        raise TrainingWorkerError("host resources changed after reservation")
    return receipt


def _preflight(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    paths = _paths(request)
    _private_directory(paths["scratch"])
    _private_directory(paths["output"], create=True)
    _verify_sources(request)
    target_bytes = _verify_tree(TARGET_DIR, TARGET_FILES)
    draft_bytes = _verify_tree(DRAFT_DIR, DRAFT_FILES)
    _verify_environment(request)
    dataset_bytes = _verify_regular(
        paths["dataset"], request["dataset_sha256"], maximum=16 * 1024 * 1024
    )
    _verify_regular(paths["config"], request["training_config_sha256"], maximum=128 * 1024)
    _verify_acl(request)
    resources = _host_resources(request)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "target_tree_sha256": _canonical_sha256(TARGET_FILES),
        "target_verified_bytes": target_bytes,
        "draft_model_sha256": DRAFT_FILES["model.safetensors"],
        "draft_verified_bytes": draft_bytes,
        "environment_sha256": request["environment_sha256"],
        "dataset_sha256": request["dataset_sha256"],
        "dataset_bytes": dataset_bytes,
        "dataset_rows": request["dataset_rows"],
        "run_mode": request["run_mode"],
        "host_resources": resources,
        "verified_at": time.time(),
    }
    _atomic_json(paths["preflight"], receipt)
    return {"state": "preflight_ready", **receipt}


def _process_alive(request: dict[str, Any], pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    proc = Path(f"/proc/{pid}")
    # A short-lived trainer can disappear between opening ``environ`` and
    # ``cmdline``.  procfs reports that race as several OSError variants (not
    # consistently FileNotFoundError), and treating the first occurrence as an
    # ambiguous live process quarantines an otherwise complete batch.  Retry
    # only while the PID still exists; an inaccessible/reused live PID remains
    # fail-closed after the bounded retries.
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
            raise TrainingWorkerError("training process identity is unreadable") from exc
    return (
        f"GPU_AGENT_CLAIM_ID={request['claim_id']}".encode() in environment
        and f"CUDA_VISIBLE_DEVICES={request['gpu_uuid']}".encode() in environment
        and b"GPU_MEM_LIMIT_GB=88" in environment
        and f"AEON_DFLASH_RUNTIME_ID={request['runtime_id']}".encode() in environment
        and b"aeon.scripts.train_qwen38_dflash2_exact" in command
        and str(_paths(request)["config"]).encode() in command
    )


def _spawn(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    paths = _paths(request)
    preflight = _read_json(paths["preflight"])
    if preflight.get("request_sha256") != request_sha256:
        raise TrainingWorkerError("training preflight receipt is stale")
    if paths["spawn"].exists() or paths["result"].exists():
        raise TrainingWorkerError("training lifecycle already exists")
    environment = {
        "PATH": "/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/home/aday",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": request["source_root"],
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
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "OMP_NUM_THREADS": "8",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29679",
        "GPU_AGENT_CLAIM_ID": request["claim_id"],
        "CUDA_VISIBLE_DEVICES": request["gpu_uuid"],
        "GPU_MEM_LIMIT_GB": "88",
        "AEON_DFLASH_RUNTIME_ID": request["runtime_id"],
        "AEON_DFLASH_RESULT_PATH": str(paths["result"]),
    }
    descriptor = os.open(
        paths["log"],
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    old_umask = os.umask(0o077)
    try:
        process = subprocess.Popen(
            [
                str(BASH),
                str(LOW_PRIORITY),
                str(ENV_PYTHON),
                "-m",
                "aeon.scripts.train_qwen38_dflash2_exact",
                "-c",
                str(paths["config"]),
            ],
            cwd=request["source_root"],
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
            raise TrainingWorkerError("training process exited during spawn")
        time.sleep(0.1)
    raise TrainingWorkerError("training process identity did not become visible")


def _spawn_receipt(request: dict[str, Any]) -> dict[str, Any] | None:
    path = _paths(request)["spawn"]
    if not path.is_file():
        return None
    receipt = _read_json(path)
    if (
        receipt.get("runtime_id") != request["runtime_id"]
        or receipt.get("request_sha256") != _sha256(_paths(request)["request"])
    ):
        raise TrainingWorkerError("training spawn receipt identity changed")
    return receipt


def _ensure_terminal(request: dict[str, Any], reason: str) -> dict[str, Any]:
    path = _paths(request)["result"]
    if path.is_file():
        result = _read_json(path)
    else:
        result = {
            "schema_version": "aeon-qwen38-dflash-adaptation-result-v1",
            "success": False,
            "failure_type": "ProcessExited",
            "failure": reason[:1000],
            "gpu_cap": None,
        }
        _atomic_json(path, result)
    if result.get("schema_version") != "aeon-qwen38-dflash-adaptation-result-v1":
        raise TrainingWorkerError("training terminal schema changed")
    return result


def _status(request: dict[str, Any]) -> dict[str, Any]:
    receipt = _spawn_receipt(request)
    pid = receipt.get("pid") if receipt else None
    if isinstance(pid, int) and _process_alive(request, pid):
        return {"state": "running", "pid": pid, "phase": "adapting_dflash2"}
    if receipt is None:
        return {"state": "absent", "pid": None, "phase": "not_spawned"}
    result = _ensure_terminal(request, "training process exited without a terminal receipt")
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
    _ensure_terminal(request, "training was stopped before successful completion")
    return {"state": "stopped", "process_absent": True}


def _write_manifest(request: dict[str, Any]) -> str:
    paths = _paths(request)
    output = _private_directory(paths["output"])
    manifest = paths["manifest"]
    if manifest.exists():
        return _sha256(manifest)
    latest = output / "training/checkpoints/LATEST"
    if latest.is_symlink():
        result = _read_json(paths["result"])
        target = os.readlink(latest)
        target_path = latest.parent / target
        target_metadata = target_path.lstat()
        if (
            re.fullmatch(r"epoch_[0-9]+_step_[0-9]+", target) is None
            or not stat.S_ISDIR(target_metadata.st_mode)
            or target_metadata.st_uid != os.geteuid()
        ):
            raise TrainingWorkerError("checkpoint pointer is unsafe")
        checkpoint_match = re.fullmatch(
            r"epoch_([0-9]+)_step_([0-9]+)", target
        )
        if checkpoint_match is None:
            raise TrainingWorkerError("checkpoint identity is malformed")
        latest.unlink()
        pointer_receipt = latest.with_name("LATEST.txt")
        descriptor = os.open(
            pointer_receipt,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        try:
            os.write(descriptor, f"{target}\n".encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if result.get("success") is not True:
            # Vast preemption is expected. Preserve the newest fully-published
            # checkpoint in the canonical failure artifact so a later reviewed
            # run can warm-start from its exact model rather than losing every
            # completed optimizer step.
            result["partial_checkpoint"] = {
                "name": target,
                "epoch": int(checkpoint_match.group(1)),
                "global_step": int(checkpoint_match.group(2)),
            }
            _atomic_json(paths["result"], result)
    entries: list[tuple[str, str]] = []
    total_bytes = 0
    for item in sorted(output.rglob("*")):
        metadata = item.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
                raise TrainingWorkerError("output directory is mutable")
            continue
        relative = item.relative_to(output).as_posix()
        if (
            item == manifest
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_ISLNK(metadata.st_mode)
            or _SAFE_RELATIVE_RE.fullmatch(relative) is None
        ):
            raise TrainingWorkerError("output contains an unsafe inode or path")
        total_bytes += metadata.st_size
        if total_bytes > 20 * 1024**3:
            raise TrainingWorkerError("training output exceeded its growth bound")
        entries.append((_sha256(item), relative))
    if not any(relative == "result.json" for _digest, relative in entries):
        raise TrainingWorkerError("training output has no terminal receipt")
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
        raise TrainingWorkerError("training output cannot settle while process is alive")
    result = _ensure_terminal(request, "training ended before output settlement")
    manifest_sha256 = _write_manifest(request)
    files = []
    manifest = _paths(request)["manifest"]
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        if match is None or _SAFE_RELATIVE_RE.fullmatch(match.group(2)) is None:
            raise TrainingWorkerError("training output manifest is malformed")
        path = _paths(request)["output"] / match.group(2)
        if _sha256(path) != match.group(1):
            raise TrainingWorkerError("training output changed after manifest")
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
        raise TrainingWorkerError("settled output manifest identity changed")
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
        raise TrainingWorkerError("training scratch is not safe to clean")
    scratch = _private_directory(paths["scratch"])
    reclaimed = 0
    for item in scratch.rglob("*"):
        metadata = item.lstat()
        if (
            metadata.st_uid != os.geteuid()
            or stat.S_ISLNK(metadata.st_mode)
            or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode))
        ):
            raise TrainingWorkerError("training scratch contains an unsafe inode")
        reclaimed += metadata.st_size
    shutil.rmtree(scratch)
    return {"state": "cleaned", "reclaimed_bytes": reclaimed}


def main() -> int:
    if len(sys.argv) not in {4, 5}:
        print(json.dumps({"ok": False, "error": "invalid_arguments"}))
        return 64
    action, raw_path, request_sha256 = sys.argv[1:4]
    extra = sys.argv[4] if len(sys.argv) == 5 else None
    if action not in {
        "preflight",
        "spawn",
        "status",
        "stop",
        "settle-status",
        "mark-settled",
        "cleanup",
    }:
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
        else:
            raise TrainingWorkerError("action requires an exact manifest digest")
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
