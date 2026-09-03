"""Disabled-by-default Fleet adapter for the .179 Flash-Next build batch."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sqlite3
import socket
import stat
import subprocess
import threading
from typing import Any, Mapping

from fleet_compute.adapters import AdapterLaunchError, RuntimeContext
from fleet_compute.models import (
    LaunchResult,
    ProbeResult,
    ProbeState,
    StopResult,
    StorageFinalizationResult,
    StoragePreparationResult,
)

from aeon.core import qwen_flash_next_legacy_recovery as legacy_recovery
from aeon.core.fleet_hosts import network_address
from aeon.scripts import assemble_qwen38_flash_next_hybrid as assembler
from aeon.scripts import qwen_flash_next_build_worker as worker


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "aeon-qwen38-flash-next-build-v2"
PROJECT = "aeon-qwen38-flash-next-build"
HOST = worker.HOST
HOSTNAME = worker.HOSTNAME
NETWORK_HOST = network_address(HOST)
LOCAL_HOST = worker.LOCAL_HOST
LOCAL_HOSTNAME = worker.LOCAL_HOSTNAME
LOCAL_CANONICAL_ROOT = Path(worker.LOCAL_CANONICAL_ROOT)
LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
REMOTE_PYTHON = str(worker.ENV_PYTHON)
REMOTE_RUN_ROOT = worker.SCRATCH_ROOT
BF16_METADATA = Path(
    "/home/aday/.local/state/aeon-flash-next/sources/qwen-bf16-metadata"
)
FP8_METADATA = Path("/home/aday/.local/state/aeon-flash-next/sources/qwen-fp8-metadata")
FP8_SOURCE = Path(
    "/home/aday/.aeon/models/.sources/Qwen3.8-Flash-Next-FP8-bcd9f01ddc9cff2316eb84281bebcd5b058bddce"
)
STATE = Path("/home/aday/.local/state/aeon-flash-next")
SETTLEMENT_OUTPUT_MAX_BYTES = 150 * 1024**3
SETTLEMENT_FREE_RESERVE_BYTES = 20_000_000_000
LONG_TRANSFER_TIMEOUT_SECONDS = 4 * 60 * 60
MODELOPT = Path(
    "/home/aday/.local/state/aeon-qwen38-quant/sources/nvidia_modelopt-0.46.0-py3-none-any.whl"
)
TRANSFORMERS = STATE / "sources/transformers-5.16.1-py3-none-any.whl"
TOKENIZERS = STATE / (
    "sources/tokenizers-0.23.1-cp310-abi3-manylinux_2_17_x86_64."
    "manylinux2014_x86_64.whl"
)
MODELOPT_RUNTIME = worker.LOCAL_MODELOPT_RUNTIME
MODELOPT_RUNTIME_MANIFEST = worker.LOCAL_MODELOPT_RUNTIME_MANIFEST
RESUME_MANIFEST = worker.LOCAL_RESUME_MANIFEST
SOURCE_MODELOPT_RUNTIME_MANIFEST = (
    PACKAGE_ROOT / "aeon/core/data/qwen38_modelopt_runtime_wheels.json"
)
SOURCE_RESUME_MANIFEST = (
    PACKAGE_ROOT / "aeon/core/data/qwen38_flash_next_fr_f373_resume.json"
)
FLEET_DATABASE = Path("/home/aday/.local/state/fleet-compute/broker.sqlite3")
COORDINATOR = Path("/home/aday/website_hosting/gpu_coord.py")
BF16_FILES = STATE / "sources/qwen-bf16-files.json"
FP8_FILES = STATE / "sources/qwen-fp8-files.json"
MTP = STATE / "official-mtp/mtp-bf16.safetensors"
MTP_MANIFEST = STATE / "official-mtp/mtp-bf16.manifest.json"
SCALES = STATE / "calibration/radixark-modelopt-expert-scales.safetensors"
SCALES_MANIFEST = STATE / "calibration/radixark-modelopt-expert-scales.manifest.json"
TRAIN = PACKAGE_ROOT / "aeon/behavioral_sft/data/train.jsonl"
EVAL = PACKAGE_ROOT / "aeon/behavioral_sft/data/eval.jsonl"
SOURCE_FILES = (
    "aeon/__init__.py",
    "aeon/behavioral_sft/__init__.py",
    "aeon/behavioral_sft/validator.py",
    "aeon/core/fleet_hosts.py",
    "aeon/core/qwen_flash_next_build_adapter.py",
    "aeon/core/qwen_flash_next_legacy_recovery.py",
    "aeon/core/data/qwen38_modelopt_runtime_wheels.json",
    "aeon/core/data/qwen38_flash_next_fr_f373_resume.json",
    "aeon/scripts/assemble_qwen38_flash_next_hybrid.py",
    "aeon/scripts/train_qwen38_flash_next_behavior.py",
    "aeon/scripts/build_qwen38_flash_next_nvfp4.py",
    "aeon/scripts/qwen_flash_next_build_worker.py",
)
BF16_METADATA_NAMES = (
    *assembler.METADATA_FILES,
    "config.json",
    "model.safetensors.index.json",
)
_RUNTIME = re.compile(r"^fr-[a-f0-9]{32}$")
_SHA = re.compile(r"^[a-f0-9]{64}$")
_PROCESS = re.compile(
    r"^aeon-flash-next-build:(fr-[a-f0-9]{32}):([a-f0-9]{64}):([0-9]+)$"
)
_SAFE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


class FlashNextBuildError(RuntimeError):
    pass


class FlashNextBuildTransportError(FlashNextBuildError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _receipt(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o022
        or metadata.st_size <= 0
    ):
        raise FlashNextBuildError(f"canonical input is unsafe: {path}")
    return {"sha256": _sha256(path), "size": metadata.st_size}


def _source_manifest() -> dict[str, dict[str, Any]]:
    return {name: _receipt(PACKAGE_ROOT / name) for name in SOURCE_FILES}


def _modelopt_runtime_manifest() -> dict[str, Any]:
    if (
        _sha256(SOURCE_MODELOPT_RUNTIME_MANIFEST)
        != worker.MODELOPT_RUNTIME_MANIFEST_SHA256
        or _sha256(MODELOPT_RUNTIME_MANIFEST) != worker.MODELOPT_RUNTIME_MANIFEST_SHA256
        or SOURCE_MODELOPT_RUNTIME_MANIFEST.read_bytes()
        != MODELOPT_RUNTIME_MANIFEST.read_bytes()
    ):
        raise FlashNextBuildError("ModelOpt runtime manifest copies diverged")
    value = json.loads(MODELOPT_RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != "aeon-qwen38-modelopt-runtime-wheel-closure-v1"
        or value.get("complete") is not True
        or not isinstance(value.get("runtime_wheels"), dict)
        or len(value["runtime_wheels"]) != 12
    ):
        raise FlashNextBuildError("ModelOpt runtime manifest changed")
    return value


def _modelopt_runtime_tree() -> dict[str, dict[str, Any]]:
    manifest = _modelopt_runtime_manifest()
    expected = {
        "MANIFEST.json",
        "FR_F373_RESUME.json",
        "antlr4-python3-runtime-4.9.3.tar.gz",
        "wheel-0.48.0-py3-none-any.whl",
        *manifest["runtime_wheels"],
    }
    actual = {item.name for item in MODELOPT_RUNTIME.iterdir()}
    if actual != expected:
        raise FlashNextBuildError("ModelOpt runtime source tree changed")
    return {name: _receipt(MODELOPT_RUNTIME / name) for name in sorted(expected)}


def _ple_shards() -> tuple[str, ...]:
    value = json.loads(
        (FP8_METADATA / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict):
        raise FlashNextBuildError("FP8 index is malformed")
    shards = tuple(
        sorted(
            {
                shard
                for name, shard in weight_map.items()
                if isinstance(name, str)
                and name.startswith(assembler.PLE_PREFIX)
                and isinstance(shard, str)
            }
        )
    )
    if len(shards) != 33 or any(PurePosixPath(name).name != name for name in shards):
        raise FlashNextBuildError("FP8 PLE shard topology changed")
    return shards


def expected_artifact_identity(
    sources: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    sources = sources or _source_manifest()
    ple = {name: _receipt(FP8_SOURCE / name)["sha256"] for name in _ple_shards()}
    return {
        "bf16_files": _sha256(BF16_FILES),
        "fp8_files": _sha256(FP8_FILES),
        "fp8_ple_tree": _canonical_sha(ple),
        "mtp": _sha256(MTP),
        "mtp_manifest": _sha256(MTP_MANIFEST),
        "expert_scales": _sha256(SCALES),
        "expert_scales_manifest": _sha256(SCALES_MANIFEST),
        "modelopt_wheel": _sha256(MODELOPT),
        "modelopt_runtime_manifest": _sha256(MODELOPT_RUNTIME_MANIFEST),
        "modelopt_runtime_tree": _canonical_sha(_modelopt_runtime_tree()),
        "quant_resume_source": _sha256(RESUME_MANIFEST),
        "transformers_wheel": _sha256(TRANSFORMERS),
        "tokenizers_wheel": _sha256(TOKENIZERS),
        "behavior_corpus": _canonical_sha(
            {"train": _sha256(TRAIN), "eval": _sha256(EVAL)}
        ),
        "source_manifest": _canonical_sha(sources),
        "sglang_commit": hashlib.sha256(worker.SGLANG_COMMIT.encode()).hexdigest(),
        "sglang_image": worker.SGLANG_IMAGE_DIGEST,
    }


def _ssh() -> list[str]:
    return [
        "/usr/bin/ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ControlPersist=no",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=6",
        f"aday@{NETWORK_HOST}",
    ]


def _remote(
    source: str,
    action: str,
    request: str,
    digest: str,
    extra: str | None = None,
    timeout: float = 120,
) -> dict[str, Any]:
    if action not in {
        "preflight",
        "spawn",
        "status",
        "stop",
        "settle-status",
        "mark-settled",
        "cleanup",
        "cleanup-prelaunch",
    }:
        raise FlashNextBuildError("invalid worker action")
    command = [
        "/usr/bin/env",
        "-i",
        "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME=/home/aday",
        "LANG=C",
        "LC_ALL=C",
        f"PYTHONPATH={source}",
        "PYTHONDONTWRITEBYTECODE=1",
        "/usr/bin/bash",
        LOW_PRIORITY,
        REMOTE_PYTHON,
        f"{source}/aeon/scripts/qwen_flash_next_build_worker.py",
        action,
        request,
        digest,
    ]
    if extra is not None:
        command.append(extra)
    result = subprocess.run(
        [*_ssh(), shlex.join(command)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode == 255:
        raise FlashNextBuildTransportError("worker transport is unavailable")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FlashNextBuildTransportError("worker returned no valid response") from exc
    if (
        result.returncode != 0
        or not isinstance(value, dict)
        or value.get("ok") is not True
    ):
        raise FlashNextBuildError(
            f"worker {action} failed: {value.get('detail') if isinstance(value, dict) else 'unknown'}"
        )
    response = value.get("result")
    if not isinstance(response, dict):
        raise FlashNextBuildError("worker result is malformed")
    return response


def _local(
    source: str,
    action: str,
    request: str,
    digest: str,
    extra: str | None = None,
    timeout: float = 120,
) -> dict[str, Any]:
    if action not in {
        "preflight",
        "spawn",
        "status",
        "stop",
        "settle-status",
        "mark-settled",
    }:
        raise FlashNextBuildError("invalid canonical-host worker action")
    command = [
        "/usr/bin/env",
        "-i",
        "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME=/home/aday",
        "LANG=C",
        "LC_ALL=C",
        f"PYTHONPATH={source}",
        "PYTHONDONTWRITEBYTECODE=1",
        "/usr/bin/bash",
        LOW_PRIORITY,
        REMOTE_PYTHON,
        f"{source}/aeon/scripts/qwen_flash_next_build_worker.py",
        action,
        request,
        digest,
    ]
    if extra is not None:
        command.append(extra)
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FlashNextBuildTransportError(
            "canonical-host worker returned no valid response"
        ) from exc
    if (
        result.returncode != 0
        or not isinstance(value, dict)
        or value.get("ok") is not True
    ):
        raise FlashNextBuildError(
            f"worker {action} failed: "
            f"{value.get('detail') if isinstance(value, dict) else 'unknown'}"
        )
    response = value.get("result")
    if not isinstance(response, dict):
        raise FlashNextBuildError("worker result is malformed")
    return response


def _worker_call(
    host: str,
    source: str,
    action: str,
    request: str,
    digest: str,
    extra: str | None = None,
    timeout: float = 120,
) -> dict[str, Any]:
    if host == LOCAL_HOST:
        return _local(source, action, request, digest, extra, timeout)
    if host == HOST:
        return _remote(source, action, request, digest, extra, timeout)
    raise FlashNextBuildError("build host is outside the reviewed placement set")


def _metrics(path: str, create: bool) -> tuple[str, int, int, int]:
    script = """import json,os,stat,sys
p,e,c=sys.argv[1:]; assert os.uname().nodename==e
if c=='1': os.makedirs(p,mode=0o700,exist_ok=True); os.chmod(p,0o700)
try:m=os.lstat(p)
except FileNotFoundError: print(json.dumps({'state':'absent'}));raise SystemExit
assert stat.S_ISDIR(m.st_mode) and m.st_uid==os.geteuid() and not m.st_mode&0o077
v=os.statvfs(p);a=m.st_blocks*512
for r,ds,fs in os.walk(p,topdown=True,followlinks=False):
 for n in [*ds,*fs]:
  q=os.path.join(r,n);x=os.lstat(q);assert x.st_uid==os.geteuid() and x.st_dev==m.st_dev and not os.path.ismount(q) and not stat.S_ISLNK(x.st_mode) and (stat.S_ISDIR(x.st_mode) or stat.S_ISREG(x.st_mode));a+=x.st_blocks*512
print(json.dumps({'state':'present','device':str(m.st_dev),'free':v.f_bavail*v.f_frsize,'inodes':v.f_favail,'allocated':a}))"""
    result = subprocess.run(
        [
            *_ssh(),
            shlex.join(
                [
                    "/usr/bin/bash",
                    LOW_PRIORITY,
                    REMOTE_PYTHON,
                    "-c",
                    script,
                    path,
                    HOSTNAME,
                    "1" if create else "0",
                ]
            ),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise FlashNextBuildError("worker storage metrics unavailable")
    value = json.loads(result.stdout)
    if value.get("state") == "absent":
        raise FileNotFoundError(path)
    return (
        str(value["device"]),
        int(value["free"]),
        int(value["inodes"]),
        int(value["allocated"]),
    )


def _local_metrics(path: str, create: bool) -> tuple[str, int, int, int]:
    root = Path(path)
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        root.chmod(0o700)
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or root.is_mount()
    ):
        raise FlashNextBuildError("canonical build directory is unsafe")
    allocated = metadata.st_blocks * 512
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        for name in (*directories, *files):
            item = Path(current) / name
            item_metadata = item.lstat()
            if (
                item_metadata.st_uid != os.geteuid()
                or item_metadata.st_dev != metadata.st_dev
                or item.is_mount()
                or stat.S_ISLNK(item_metadata.st_mode)
                or not (
                    stat.S_ISDIR(item_metadata.st_mode)
                    or stat.S_ISREG(item_metadata.st_mode)
                )
            ):
                raise FlashNextBuildError("canonical build tree is unsafe")
            allocated += item_metadata.st_blocks * 512
    filesystem = os.statvfs(root)
    return (
        str(metadata.st_dev),
        filesystem.f_bavail * filesystem.f_frsize,
        filesystem.f_favail,
        allocated,
    )


class _Heartbeat:
    def __init__(self, context: RuntimeContext):
        self.context = context
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.context.heartbeat(None, "Staging pinned Flash-Next hybrid sources")
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.error is not None:
            raise FlashNextBuildError("preparation heartbeat failed") from self.error

    def _run(self):
        while not self.stop.wait(120):
            try:
                self.context.heartbeat(
                    None, "Pinned Flash-Next source staging remains active"
                )
            except BaseException as exc:
                self.error = exc
                return


def _verify_resume_operational_state() -> dict[str, Any]:
    if (
        _sha256(SOURCE_RESUME_MANIFEST) != worker.RESUME_MANIFEST_SHA256
        or _sha256(RESUME_MANIFEST) != worker.RESUME_MANIFEST_SHA256
        or SOURCE_RESUME_MANIFEST.read_bytes() != RESUME_MANIFEST.read_bytes()
    ):
        raise FlashNextBuildError("quant-only resume receipt copies diverged")
    receipt = json.loads(RESUME_MANIFEST.read_text(encoding="utf-8"))
    source = receipt.get("source_runtime")
    storage_receipt = receipt.get("storage")
    release = receipt.get("coordinator_release")
    if (
        not isinstance(source, dict)
        or not isinstance(storage_receipt, dict)
        or not isinstance(release, dict)
    ):
        raise FlashNextBuildError("quant-only resume receipt is malformed")
    for field in ("pid", "modelopt_pid"):
        pid = source.get(field)
        if type(pid) is not int or pid <= 1 or Path(f"/proc/{pid}").exists():
            raise FlashNextBuildError("prior build process absence is not proven")
    try:
        os.killpg(source["pid"], 0)
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise FlashNextBuildError(
                "prior build process-group absence is not proven"
            ) from exc
    else:
        raise FlashNextBuildError("prior build process group is still live")
    try:
        connection = sqlite3.connect(
            f"file:{FLEET_DATABASE}?mode=ro", uri=True, timeout=5
        )
        connection.row_factory = sqlite3.Row
        runtime = connection.execute(
            "SELECT * FROM runtimes WHERE runtime_id = ?", (worker.RESUME_RUNTIME_ID,)
        ).fetchone()
        storage = connection.execute(
            "SELECT * FROM storage_attempts WHERE runtime_id = ?",
            (worker.RESUME_RUNTIME_ID,),
        ).fetchone()
        job = connection.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (source.get("job_id"),)
        ).fetchone()
    except sqlite3.Error as exc:
        raise FlashNextBuildError("Fleet resume state is unreadable") from exc
    finally:
        if "connection" in locals():
            connection.close()
    if (
        runtime is None
        or runtime["state"] != "stopped"
        or runtime["process_absent"] != 1
        or runtime["claim_id"] != source.get("claim_id")
        or runtime["owner"] != source.get("owner")
        or runtime["pid"] != source.get("pid")
        or runtime["process_identity"] != source.get("process_identity")
        or runtime["deployment_revision"] != source.get("deployment_revision")
        or storage is None
        or storage["state"] != "complete"
        or storage["output_settled"] != 1
        or storage["cleanup_complete"] != 1
        or storage["canonical_output_path"]
        != storage_receipt.get("canonical_output_path")
        or job is None
        or job["state"] != "cancelled"
        or job["runtime_id"] != worker.RESUME_RUNTIME_ID
    ):
        raise FlashNextBuildError("Fleet resume state is not terminal and settled")
    if socket.gethostname() != LOCAL_HOSTNAME:
        raise FlashNextBuildError("resume claim readback is only allowed on .177")
    history = subprocess.run(
        [
            str(COORDINATOR),
            "history",
            "--claim",
            str(source["claim_id"]),
            "--limit",
            "20",
            "--json",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    status = subprocess.run(
        [str(COORDINATOR), "status", "--json"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        events = json.loads(history.stdout)
        active = json.loads(status.stdout)
    except json.JSONDecodeError as exc:
        raise FlashNextBuildError("coordinator resume readback is malformed") from exc
    if (
        history.returncode != 0
        or status.returncode != 0
        or not isinstance(events, list)
        or not events
        or {key: events[0].get(key) for key in release} != release
        or not isinstance(active, list)
        or any(
            item.get("claim_id") == source["claim_id"]
            for item in active
            if isinstance(item, dict)
        )
    ):
        raise FlashNextBuildError("prior coordinator claim release is not proven")
    return {
        "resume_manifest_sha256": worker.RESUME_MANIFEST_SHA256,
        "source_runtime_id": worker.RESUME_RUNTIME_ID,
        "released_event_id": release["event_id"],
        "fleet_storage_complete": True,
        "prior_process_absent": True,
    }


class AeonQwenFlashNextBuildAdapter:
    def __init__(self):
        self._prepared: dict[str, dict[str, str]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _payload(payload: Mapping[str, Any]) -> str:
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"recipe"}
            or payload.get("recipe") != worker.RESUME_RECIPE
        ):
            raise FlashNextBuildError("build payload is not reviewed")
        return worker.RESUME_RECIPE

    @staticmethod
    def _write(path: Path, payload: bytes) -> None:
        if path.exists() or path.is_symlink():
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o077
                or path.read_bytes() != payload
            ):
                raise FlashNextBuildError("existing local request identity changed")
            return
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _dirs(scratch: str) -> None:
        script = """import os,pathlib,stat,sys
r=pathlib.Path(sys.argv[1]);assert os.uname().nodename==sys.argv[2]
for p in (r,r/'source',r/'inputs',r/'inputs/bf16',r/'inputs/fp8-ple',r/'fixtures',r/'output'):
 p.mkdir(mode=0o700,parents=True,exist_ok=True);p.chmod(0o700);m=p.lstat();assert stat.S_ISDIR(m.st_mode) and m.st_uid==os.geteuid() and not m.st_mode&0o077"""
        result = subprocess.run(
            [
                *_ssh(),
                shlex.join(
                    [
                        "/usr/bin/bash",
                        LOW_PRIORITY,
                        REMOTE_PYTHON,
                        "-c",
                        script,
                        scratch,
                        HOSTNAME,
                    ]
                ),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise FlashNextBuildError("remote directory preparation failed")

    @staticmethod
    def _rsync(
        items: list[str], destination: str, cwd: Path | None = None, timeout: int = 1800
    ) -> None:
        transport = " ".join(shlex.quote(item) for item in _ssh()[:-1])
        result = subprocess.run(
            [
                "/usr/bin/bash",
                LOW_PRIORITY,
                "/usr/bin/rsync",
                "-a",
                "--checksum",
                "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
                "--protect-args",
                "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
                "-e",
                transport,
                *items,
                f"aday@{NETWORK_HOST}:{destination}",
            ],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise FlashNextBuildError("staging transfer failed")

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        if _RUNTIME.fullmatch(context.runtime_id) is None or context.job_id is None:
            raise FlashNextBuildError("runtime/job identity malformed")
        recipe = self._payload(context.payload)
        if recipe == worker.RESUME_RECIPE:
            _verify_resume_operational_state()
        sources = _source_manifest()
        if (
            context.profile.profile_id != PROFILE_ID
            or context.profile.project != PROJECT
            or context.profile.artifact_identity != expected_artifact_identity(sources)
        ):
            raise FlashNextBuildError("profile artifact identity changed")
        lease = context.lease
        local = lease.host == LOCAL_HOST
        if (
            lease.host not in {HOST, LOCAL_HOST}
            or lease.memory_total_mib is None
            or lease.memory_total_mib < 94 * 1024
            or abs(lease.vram_budget_gb - 88.0) > 1e-9
            or lease.exclusive is not True
            or (local and lease.physical_gpu != 0)
            or (local and context.scratch_path is not None)
            or (
                local
                and context.canonical_output_path
                != LOCAL_CANONICAL_ROOT / context.runtime_id
            )
            or (not local and context.scratch_path != lease.run_dir)
            or (
                context.profile.runtime_growth_bytes_max
                + context.profile.worker_free_reserve_bytes
                != worker.RESUME_POST_STAGE_DISK_FLOOR_BYTES
            )
        ):
            raise FlashNextBuildError(
                "lease differs from reviewed Flash build contract"
            )
        scratch = str(context.canonical_output_path if local else context.scratch_path)
        source_root = str(PACKAGE_ROOT) if local else f"{scratch}/source"
        request_path = f"{scratch}/qwen-flash-next-build-request.json"
        input_files: dict[str, dict[str, Any]] = {}
        if recipe == worker.FULL_RECIPE:
            for name in BF16_METADATA_NAMES:
                input_files[f"bf16/{name}"] = _receipt(BF16_METADATA / name)
            for name in ("config.json", "model.safetensors.index.json"):
                input_files[f"fp8-ple/{name}"] = _receipt(FP8_METADATA / name)
            for name in _ple_shards():
                input_files[f"fp8-ple/{name}"] = _receipt(FP8_SOURCE / name)
        runtime_manifest = _modelopt_runtime_manifest()
        fixtures = {
            "mtp-bf16.safetensors": _receipt(MTP),
            "mtp-bf16.manifest.json": _receipt(MTP_MANIFEST),
            "expert-scales.safetensors": _receipt(SCALES),
            "expert-scales.manifest.json": _receipt(SCALES_MANIFEST),
            "nvidia_modelopt-0.46.0-py3-none-any.whl": _receipt(MODELOPT),
            "transformers-5.16.1-py3-none-any.whl": _receipt(TRANSFORMERS),
            "tokenizers-0.23.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": _receipt(
                TOKENIZERS
            ),
            "qwen-bf16-files.json": _receipt(BF16_FILES),
            "qwen-fp8-files.json": _receipt(FP8_FILES),
            "behavior-train.jsonl": _receipt(TRAIN),
            "behavior-eval.jsonl": _receipt(EVAL),
            "modelopt-runtime-manifest.json": _receipt(MODELOPT_RUNTIME_MANIFEST),
            "fr-f373-resume.json": _receipt(RESUME_MANIFEST),
        }
        fixtures.update(
            {
                name: _receipt(MODELOPT_RUNTIME / name)
                for name in runtime_manifest["runtime_wheels"]
            }
        )
        request = {
            "schema_version": worker.SCHEMA_VERSION,
            "runtime_id": context.runtime_id,
            "job_id": context.job_id,
            "host": lease.host,
            "hostname": LOCAL_HOSTNAME if local else HOSTNAME,
            "claim_id": lease.claim_id,
            "owner": lease.owner,
            "physical_gpu": lease.physical_gpu,
            "gpu_uuid": lease.gpu_uuid,
            "vram_budget_gb": lease.vram_budget_gb,
            "exclusive": lease.exclusive,
            "min_host_memory_gb": context.profile.min_host_memory_gb,
            "min_host_commit_gb": context.profile.min_host_commit_gb,
            "post_stage_disk_floor_bytes": worker.RESUME_POST_STAGE_DISK_FLOOR_BYTES,
            "min_shm_free_gb": context.profile.min_shm_free_gb,
            "scratch_path": scratch,
            "source_root": source_root,
            "source_files": sources,
            "input_files": input_files,
            "fixture_files": fixtures,
            "recipe": recipe,
            "resume_source_manifest_sha256": (
                worker.RESUME_MANIFEST_SHA256
                if recipe == worker.RESUME_RECIPE
                else None
            ),
            "sglang_commit": worker.SGLANG_COMMIT,
            "sglang_image_digest": worker.SGLANG_IMAGE_DIGEST,
        }
        raw = (
            json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        digest = hashlib.sha256(raw).hexdigest()
        local = context.run_dir / "qwen-flash-next-build-request.json"
        self._write(local, raw)
        if lease.host == LOCAL_HOST:
            device, _free, _inodes, before = _local_metrics(scratch, True)
            self._write(Path(request_path), raw)
        else:
            device, _free, _inodes, before = _metrics(scratch, True)
        with _Heartbeat(context):
            if lease.host != LOCAL_HOST:
                self._dirs(scratch)
                self._rsync(
                    ["-R", "--", *sources],
                    f"{source_root}/",
                    cwd=PACKAGE_ROOT,
                    timeout=600,
                )
                self._rsync(["--", str(local)], request_path)
                if recipe == worker.FULL_RECIPE:
                    for name in BF16_METADATA_NAMES:
                        self._rsync(
                            ["--", str(BF16_METADATA / name)],
                            f"{scratch}/inputs/bf16/{name}",
                        )
                    for name in ("config.json", "model.safetensors.index.json"):
                        self._rsync(
                            ["--", str(FP8_METADATA / name)],
                            f"{scratch}/inputs/fp8-ple/{name}",
                        )
                    for name in _ple_shards():
                        self._rsync(
                            ["--", str(FP8_SOURCE / name)],
                            f"{scratch}/inputs/fp8-ple/{name}",
                            timeout=1800,
                        )
                local_fixtures = {
                    "mtp-bf16.safetensors": MTP,
                    "mtp-bf16.manifest.json": MTP_MANIFEST,
                    "expert-scales.safetensors": SCALES,
                    "expert-scales.manifest.json": SCALES_MANIFEST,
                    "nvidia_modelopt-0.46.0-py3-none-any.whl": MODELOPT,
                    "transformers-5.16.1-py3-none-any.whl": TRANSFORMERS,
                    "tokenizers-0.23.1-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl": TOKENIZERS,
                    "qwen-bf16-files.json": BF16_FILES,
                    "qwen-fp8-files.json": FP8_FILES,
                    "behavior-train.jsonl": TRAIN,
                    "behavior-eval.jsonl": EVAL,
                    "modelopt-runtime-manifest.json": MODELOPT_RUNTIME_MANIFEST,
                    "fr-f373-resume.json": RESUME_MANIFEST,
                }
                local_fixtures.update(
                    {
                        name: MODELOPT_RUNTIME / name
                        for name in runtime_manifest["runtime_wheels"]
                    }
                )
                for name, path in local_fixtures.items():
                    self._rsync(
                        ["--", str(path)],
                        f"{scratch}/fixtures/{name}",
                        timeout=1800,
                    )
            preflight = None
            for stage_attempt in range(1, 4):
                try:
                    preflight = _worker_call(
                        lease.host,
                        source_root,
                        "preflight",
                        request_path,
                        digest,
                        timeout=LONG_TRANSFER_TIMEOUT_SECONDS,
                    )
                    break
                except FlashNextBuildTransportError:
                    # The remote process may still exist after an SSH failure.
                    # Never create a concurrent staging writer in that ambiguity.
                    raise
                except FlashNextBuildError:
                    if stage_attempt == 3:
                        raise
                    context.heartbeat(
                        None,
                        "Retrying exited source stager with verified shards and partials",
                    )
            if preflight is None:
                raise FlashNextBuildError("source preflight produced no receipt")
        if preflight.get("request_sha256") != digest:
            raise FlashNextBuildError("preflight receipt changed")
        if lease.host == LOCAL_HOST:
            filesystem, free, inodes, allocated = _local_metrics(scratch, False)
        else:
            filesystem, free, inodes, allocated = _metrics(scratch, False)
        if filesystem != device:
            raise FlashNextBuildError("worker filesystem changed")
        with self._lock:
            self._prepared[context.runtime_id] = {
                "digest": digest,
                "request": request_path,
                "source": source_root,
                "host": lease.host,
            }
        return StoragePreparationResult(
            context.scratch_path, filesystem, free, inodes, max(0, allocated - before)
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        with self._lock:
            prepared = self._prepared.get(context.runtime_id)
        if prepared is None:
            raise AdapterLaunchError("preflight receipt absent", process_absent=True)
        try:
            result = _worker_call(
                prepared["host"],
                prepared["source"],
                "spawn",
                prepared["request"],
                prepared["digest"],
                timeout=90,
            )
            pid = result.get("pid")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
                raise FlashNextBuildError("PID malformed")
            context.heartbeat(pid, "Flash-Next tune-before-quant pipeline bound to PID")
            return LaunchResult(
                pid,
                f"aeon-flash-next-build:{context.runtime_id}:{prepared['digest']}:{pid}",
            )
        except BaseException as exc:
            status = _worker_call(
                prepared["host"],
                prepared["source"],
                "status",
                prepared["request"],
                prepared["digest"],
                timeout=60,
            )
            if status.get("state") == "absent":
                raise AdapterLaunchError(
                    f"pipeline failed before process creation: {exc}",
                    process_absent=True,
                ) from exc
            raise

    @staticmethod
    def _identity(runtime: Mapping[str, Any]) -> tuple[str, str, int]:
        match = _PROCESS.fullmatch(str(runtime.get("process_identity") or ""))
        if (
            match is None
            or match.group(1) != runtime.get("runtime_id")
            or int(match.group(3)) != runtime.get("pid")
            or runtime.get("host") not in {HOST, LOCAL_HOST}
            or PurePosixPath(str(runtime.get("run_dir") or "")).parent
            != REMOTE_RUN_ROOT
            or (runtime.get("host") == LOCAL_HOST and runtime.get("physical_gpu") != 0)
        ):
            raise FlashNextBuildError("runtime identity changed")
        return match.group(1), match.group(2), int(match.group(3))

    @classmethod
    def _action(
        cls,
        runtime: Mapping[str, Any],
        action: str,
        extra: str | None = None,
        timeout: float = 120,
    ) -> dict[str, Any]:
        runtime_id, digest, _pid = cls._identity(runtime)
        host = str(runtime["host"])
        if host == LOCAL_HOST:
            scratch = str(LOCAL_CANONICAL_ROOT / runtime_id)
            source = str(PACKAGE_ROOT)
        else:
            scratch = str(runtime["run_dir"])
            source = f"{scratch}/source"
        return _worker_call(
            host,
            source,
            action,
            f"{scratch}/qwen-flash-next-build-request.json",
            digest,
            extra,
            timeout,
        )

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        legacy = legacy_recovery.probe_legacy_pidless_build(runtime)
        if legacy is not None:
            return legacy
        try:
            _rid, _digest, pid = self._identity(runtime)
            status = self._action(runtime, "status", timeout=90)
        except FlashNextBuildTransportError:
            raise
        except FlashNextBuildError as exc:
            return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))
        state = status.get("state")
        if state == "running" and status.get("pid") == pid:
            return ProbeResult(
                ProbeState.RUNNING, True, False, "Flash-Next tune/quant build running"
            )
        if state == "completed":
            return ProbeResult(
                ProbeState.COMPLETED, False, True, "Flash-Next checkpoint built"
            )
        if state == "failed":
            return ProbeResult(
                ProbeState.FAILED,
                False,
                True,
                str((status.get("result") or {}).get("failure") or "build failed")[
                    :500
                ],
            )
        if state == "absent":
            return ProbeResult(ProbeState.ABSENT, False, True, "build process absent")
        return ProbeResult(ProbeState.UNKNOWN, False, False, "build state ambiguous")

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        try:
            result = self._action(runtime, "stop", timeout=150)
        except FlashNextBuildError as exc:
            return StopResult(False, False, str(exc))
        absent = result.get("process_absent") is True
        return StopResult(absent, True, reason if absent else "build is still stopping")

    @staticmethod
    def _local_valid(path: Path) -> tuple[bool, str | None]:
        manifest = path / "MANIFEST.sha256"
        if not manifest.is_file():
            return False, None
        root_metadata = path.lstat()
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_uid != os.geteuid()
            or root_metadata.st_mode & 0o077
        ):
            raise FlashNextBuildError("canonical output directory is unsafe")
        total = 0
        seen_inodes: set[tuple[int, int]] = set()
        expected = {"MANIFEST.sha256"}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
            if match is None or _SAFE.fullmatch(match.group(2)) is None:
                raise FlashNextBuildError("canonical manifest malformed")
            item = path / match.group(2)
            metadata = item.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_ISLNK(metadata.st_mode)
                or _sha256(item) != match.group(1)
            ):
                raise FlashNextBuildError("canonical output changed")
            inode = (metadata.st_dev, metadata.st_ino)
            if inode not in seen_inodes:
                seen_inodes.add(inode)
                total += metadata.st_size
            expected.add(match.group(2))
            if total > 150 * 1024**3:
                raise FlashNextBuildError("canonical output exceeds bound")
        actual = {"MANIFEST.sha256"}
        for item in path.rglob("*"):
            metadata = item.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                    raise FlashNextBuildError("canonical output subdirectory is unsafe")
                continue
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise FlashNextBuildError("canonical output contains an unsafe inode")
            actual.add(item.relative_to(path).as_posix())
        if actual != expected:
            raise FlashNextBuildError("canonical output closure changed")
        return True, _sha256(manifest)

    @staticmethod
    def _copy(remote: str, local: Path) -> None:
        existing_parent = local
        while not existing_parent.exists():
            parent = existing_parent.parent
            if parent == existing_parent:
                raise FlashNextBuildError("canonical output parent is unavailable")
            existing_parent = parent
        parent_metadata = existing_parent.lstat()
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or stat.S_ISLNK(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.geteuid()
            or parent_metadata.st_mode & 0o077
        ):
            raise FlashNextBuildError("canonical output parent is unsafe")
        filesystem = os.statvfs(existing_parent)
        free = filesystem.f_bavail * filesystem.f_frsize
        required = SETTLEMENT_OUTPUT_MAX_BYTES + SETTLEMENT_FREE_RESERVE_BYTES
        if free < required:
            raise FlashNextBuildError(
                "canonical output filesystem lacks the reviewed settlement reserve"
            )
        local.mkdir(mode=0o700, parents=True, exist_ok=True)
        local.chmod(0o700)
        transport = " ".join(shlex.quote(item) for item in _ssh()[:-1])
        result = subprocess.run(
            [
                "/usr/bin/bash",
                LOW_PRIORITY,
                "/usr/bin/rsync",
                "-aH",
                "--checksum",
                "--protect-args",
                "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
                "-e",
                transport,
                "--",
                f"aday@{NETWORK_HOST}:{remote}/",
                f"{local}/",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=LONG_TRANSFER_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise FlashNextBuildError("output settlement transfer failed")

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        local_host = runtime.get("host") == LOCAL_HOST
        if runtime.get("process_identity") is None:
            if local_host:
                return StorageFinalizationResult(
                    True,
                    True,
                    0,
                    "canonical .177 prelaunch data retained; no automatic cleanup",
                )
            scratch = str(runtime.get("run_dir") or "")
            local = Path(scratch) / "qwen-flash-next-build-request.json"
            if not local.is_file():
                raise FlashNextBuildError("prelaunch request absent")
            try:
                _metrics(scratch, False)
            except FileNotFoundError:
                return StorageFinalizationResult(
                    True, True, 0, "prelaunch worker scratch is already absent"
                )
            digest = _sha256(local)
            result = _remote(
                f"{scratch}/source",
                "cleanup-prelaunch",
                f"{scratch}/qwen-flash-next-build-request.json",
                digest,
                timeout=1800,
            )
            return StorageFinalizationResult(
                True, True, int(result["reclaimed_bytes"]), "prelaunch scratch cleaned"
            )
        self._identity(runtime)
        canonical = Path(str(storage["canonical_output_path"]))
        if local_host:
            status = self._action(runtime, "settle-status", timeout=3600)
            manifest = str(status.get("manifest_sha256") or "")
            if _SHA.fullmatch(manifest) is None:
                raise FlashNextBuildError("canonical .177 output manifest malformed")
            output = canonical / "output"
            valid, local_digest = (
                self._local_valid(output) if output.exists() else (False, None)
            )
            if not valid:
                raise FlashNextBuildError("canonical .177 output is incomplete")
            if local_digest != manifest:
                raise FlashNextBuildError("canonical .177 output manifest changed")
            return StorageFinalizationResult(
                True,
                True,
                0,
                "Flash-Next output completed directly on canonical .177 storage; "
                "build inputs retained",
            )
        valid, local_digest = (
            self._local_valid(canonical) if canonical.exists() else (False, None)
        )
        scratch = str(runtime["run_dir"])
        try:
            _metrics(scratch, False)
        except FileNotFoundError:
            if valid:
                return StorageFinalizationResult(
                    True,
                    True,
                    0,
                    "Flash-Next output is settled and worker scratch is absent",
                )
            raise FlashNextBuildError(
                "worker scratch vanished before durable output settlement"
            )
        status = self._action(runtime, "settle-status", timeout=3600)
        manifest = str(status.get("manifest_sha256") or "")
        if _SHA.fullmatch(manifest) is None:
            raise FlashNextBuildError("worker manifest malformed")
        if not valid:
            self._copy(f"{scratch}/output", canonical)
            valid, local_digest = self._local_valid(canonical)
        if not valid or local_digest != manifest:
            raise FlashNextBuildError("canonical output differs from worker")
        self._action(runtime, "mark-settled", manifest, 3600)
        cleaned = self._action(runtime, "cleanup", manifest, 3600)
        reclaimed = cleaned.get("reclaimed_bytes")
        if (
            isinstance(reclaimed, bool)
            or not isinstance(reclaimed, int)
            or reclaimed < 0
        ):
            raise FlashNextBuildError("cleanup receipt malformed")
        return StorageFinalizationResult(
            True, True, reclaimed, "Flash-Next checkpoint settled durably on .177"
        )


def create_fleet_adapter() -> AeonQwenFlashNextBuildAdapter:
    return AeonQwenFlashNextBuildAdapter()
