"""Fleet adapter for exact-target Qwen3.8 DFlash2 adaptation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
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

from aeon.scripts import qwen_dflash_training_worker as worker
from aeon.core.dflash_adaptation import ADAPTATION_MODE, FULL_ADAPTATION_MODE
from aeon.core.fleet_hosts import network_address


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "aeon-qwen38-dflash-adapt"
PROJECT = "aeon-dflash-adapt"
HOST = worker.HOST
HOSTNAME = worker.HOSTNAME
NETWORK_HOST = network_address(HOST)
REMOTE_PYTHON = (
    "/home/aday/.local/share/uv/python/"
    "cpython-3.12-linux-x86_64-gnu/bin/python3.12"
)
LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
REMOTE_RUN_ROOT = worker.SCRATCH_ROOT
CANONICAL_DATASET = Path(
    "/home/aday/.local/state/aeon-qwen38-dflash-data/"
    "ara-prefix530e-greedy-v2-256/train.jsonl"
)
CANONICAL_DATASET_SHA256 = (
    "61b8e150651ecc14c47e1068ce36fc130bb56e18117b3b68e098390defea92f5"
)
CANONICAL_DATASET_ROWS = 256
SOURCE_FILES = (
    "aeon/__init__.py",
    "aeon/core/__init__.py",
    "aeon/core/dflash_adaptation.py",
    "aeon/core/dflash_dpace.py",
    "aeon/core/fleet_hosts.py",
    "aeon/core/qwen_dflash_training_adapter.py",
    "aeon/scripts/qwen_dflash_training_worker.py",
    "aeon/scripts/train_qwen38_dflash2_exact.py",
)
TRAINING_PLANS = {
    "smoke": {
        "adaptation_mode": ADAPTATION_MODE,
        "dataset_rows": 1,
        "num_anchors": 8,
        "num_epochs": 1,
    },
    "calibrate": {
        "adaptation_mode": ADAPTATION_MODE,
        "dataset_rows": 1,
        "num_anchors": 64,
        "num_epochs": 1,
    },
    "adapt-v1": {
        "adaptation_mode": ADAPTATION_MODE,
        "dataset_rows": 64,
        "num_anchors": 64,
        "num_epochs": 2,
    },
    "calibrate-full": {
        "adaptation_mode": FULL_ADAPTATION_MODE,
        "dataset_rows": 1,
        "num_anchors": 64,
        "num_epochs": 1,
    },
    "adapt-full-v1": {
        "adaptation_mode": FULL_ADAPTATION_MODE,
        "dataset_rows": 64,
        "num_anchors": 64,
        "num_epochs": 2,
    },
    "adapt-full-dpace-v2": {
        "adaptation_mode": FULL_ADAPTATION_MODE,
        "dataset_rows": 256,
        "num_anchors": 64,
        "num_epochs": 2,
        "grad_accumulation_steps": 8,
        "training_objective": "dpace-v1",
        "dpace_alpha": 0.5,
        "learning_rate": 0.0006,
        "warmup_ratio": 0.04,
    },
}
_RUNTIME_RE = re.compile(r"^fr-[a-f0-9]{32}$")
_PROCESS_IDENTITY_RE = re.compile(
    r"^aeon-dflash-adapt:(fr-[a-f0-9]{32}):([a-f0-9]{64}):([0-9]+)$"
)
_SAFE_RELATIVE_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


class QwenDFlashTrainingError(RuntimeError):
    pass


class QwenDFlashTrainingTransportError(QwenDFlashTrainingError):
    """A retryable failure before the reviewed remote protocol answered."""


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


def _source_manifest() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = PACKAGE_ROOT / relative
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
            or metadata.st_nlink != 1
        ):
            raise QwenDFlashTrainingError(f"training source is unsafe: {relative}")
        result[relative] = _sha256(path)
    return result


def _expected_artifact_identity(sources: dict[str, str]) -> dict[str, str]:
    return {
        "canonical_dataset": CANONICAL_DATASET_SHA256,
        "draft_config": worker.DRAFT_FILES["config.json"],
        "draft_model": worker.DRAFT_FILES["model.safetensors"],
        "draft_tree": _canonical_sha256(worker.DRAFT_FILES),
        "environment": _canonical_sha256(
            {
                "sentinels": worker.ENV_SENTINELS,
                "versions": worker.ENV_VERSIONS,
            }
        ),
        "environment_wheel": worker.ENV_WHEEL_SHA256,
        "source_manifest": _canonical_sha256(sources),
        "target_tree": _canonical_sha256(worker.TARGET_FILES),
        "training_plans": _canonical_sha256(TRAINING_PLANS),
    }


def _ssh_base() -> list[str]:
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


def _remote_command(
    source_root: str,
    action: str,
    request: str,
    digest: str,
    extra: str | None = None,
) -> list[str]:
    if action not in {
        "preflight",
        "spawn",
        "status",
        "stop",
        "settle-status",
        "mark-settled",
        "cleanup",
    }:
        raise QwenDFlashTrainingError("invalid training worker action")
    command = [
        "/usr/bin/env",
        "-i",
        "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME=/home/aday",
        "LANG=C",
        "LC_ALL=C",
        f"PYTHONPATH={source_root}",
        "PYTHONDONTWRITEBYTECODE=1",
        "/usr/bin/bash",
        LOW_PRIORITY,
        REMOTE_PYTHON,
        f"{source_root}/aeon/scripts/qwen_dflash_training_worker.py",
        action,
        request,
        digest,
    ]
    if extra is not None:
        command.append(extra)
    return [*_ssh_base(), shlex.join(command)]


def _remote_action(
    source_root: str,
    action: str,
    request: str,
    digest: str,
    *,
    extra: str | None = None,
    timeout: float = 120,
) -> dict[str, Any]:
    result = subprocess.run(
        _remote_command(source_root, action, request, digest, extra),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if len(result.stdout) > 2 * 1024 * 1024 or len(result.stderr) > 256 * 1024:
        raise QwenDFlashTrainingError("training worker response exceeded its bound")
    if result.returncode == 255:
        raise QwenDFlashTrainingTransportError("training worker transport is unavailable")
    try:
        value = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenDFlashTrainingTransportError(
            "training worker transport returned no valid response"
        ) from exc
    if result.returncode != 0 or not isinstance(value, dict) or value.get("ok") is not True:
        detail = value.get("detail") if isinstance(value, dict) else None
        raise QwenDFlashTrainingError(
            f"training worker {action} failed: {detail or 'unknown error'}"
        )
    response = value.get("result")
    if not isinstance(response, dict):
        raise QwenDFlashTrainingError("training worker result is malformed")
    return response


def _remote_metrics(path: str, *, create: bool) -> tuple[str, int, int, int]:
    script = r"""
import json, os, stat, sys
path, expected, create = sys.argv[1:]
assert os.uname().nodename == expected
if create == "1":
 os.makedirs(path, mode=0o700, exist_ok=True); os.chmod(path, 0o700)
try: metadata = os.lstat(path)
except FileNotFoundError:
 print(json.dumps({"state": "absent"})); raise SystemExit(0)
assert stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.geteuid() and not metadata.st_mode & 0o077
values = os.statvfs(path); allocated = metadata.st_blocks * 512
for root, directories, files in os.walk(path, topdown=True, followlinks=False):
 for name in [*directories, *files]:
  item = os.path.join(root, name); item_metadata = os.lstat(item)
  assert item_metadata.st_uid == os.geteuid() and not stat.S_ISLNK(item_metadata.st_mode)
  assert stat.S_ISDIR(item_metadata.st_mode) or stat.S_ISREG(item_metadata.st_mode)
  allocated += item_metadata.st_blocks * 512
print(json.dumps({"state":"present", "device":str(metadata.st_dev), "free":values.f_bavail*values.f_frsize, "inodes":values.f_favail, "allocated":allocated}, sort_keys=True))
"""
    result = subprocess.run(
        [
            *_ssh_base(),
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
        timeout=60,
    )
    if result.returncode != 0 or len(result.stdout) > 4096:
        raise QwenDFlashTrainingError("training worker storage metrics are unavailable")
    try:
        value = json.loads(result.stdout)
        if value.get("state") == "absent":
            raise FileNotFoundError(path)
        return (
            str(value["device"]),
            int(value["free"]),
            int(value["inodes"]),
            int(value["allocated"]),
        )
    except FileNotFoundError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenDFlashTrainingError("training worker storage metrics are malformed") from exc


class _PreparationHeartbeat:
    def __init__(self, context: RuntimeContext) -> None:
        self.context = context
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_PreparationHeartbeat":
        self.context.heartbeat(None, "Exact Qwen3.8 DFlash2 artifact preflight")
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        if self.error is not None:
            raise QwenDFlashTrainingError("training preflight heartbeat failed") from self.error

    def _run(self) -> None:
        while not self.stop.wait(240):
            try:
                self.context.heartbeat(None, "Exact DFlash2 artifact preflight is active")
            except BaseException as exc:
                self.error = exc
                return


class AeonQwenDFlashTrainingAdapter:
    """One reviewed single-GPU adaptation lane on the 96 GB worker."""

    def __init__(self) -> None:
        self._prepared: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _payload(payload: Mapping[str, Any]) -> dict[str, str]:
        if not isinstance(payload, Mapping) or set(payload) - {"run_mode"}:
            raise QwenDFlashTrainingError("training payload has unsupported fields")
        run_mode = payload.get("run_mode", "smoke")
        if not isinstance(run_mode, str) or run_mode not in TRAINING_PLANS:
            raise QwenDFlashTrainingError("training mode is not reviewed")
        return {"run_mode": run_mode}

    @staticmethod
    def _profile_identity(
        context: RuntimeContext, sources: dict[str, str]
    ) -> None:
        if (
            context.profile.profile_id != PROFILE_ID
            or context.profile.project != PROJECT
            or context.profile.artifact_identity
            != _expected_artifact_identity(sources)
        ):
            raise QwenDFlashTrainingError("training profile identity changed")
        lease = context.lease
        if (
            lease.host != HOST
            or lease.memory_total_mib is None
            or lease.memory_total_mib < 94 * 1024
            or abs(lease.vram_budget_gb - 88.0) > 1e-9
            or lease.exclusive is not True
            or context.scratch_path != lease.run_dir
        ):
            raise QwenDFlashTrainingError("training lease differs from its reviewed profile")

    @staticmethod
    def _write_private(path: Path, payload: bytes) -> None:
        if path.exists() or path.is_symlink():
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise QwenDFlashTrainingError("local staging path is unsafe")
            path.unlink()
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _dataset(run_mode: str) -> tuple[bytes, int]:
        metadata = CANONICAL_DATASET.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or metadata.st_nlink != 1
            or _sha256(CANONICAL_DATASET) != CANONICAL_DATASET_SHA256
        ):
            raise QwenDFlashTrainingError("canonical adaptation dataset identity changed")
        payload = CANONICAL_DATASET.read_bytes()
        lines = payload.splitlines(keepends=True)
        if len(lines) != CANONICAL_DATASET_ROWS or not all(line.endswith(b"\n") for line in lines):
            raise QwenDFlashTrainingError("canonical adaptation dataset framing changed")
        selected = lines[: int(TRAINING_PLANS[run_mode]["dataset_rows"])]
        for line in selected:
            if not isinstance(json.loads(line), dict):
                raise QwenDFlashTrainingError("adaptation dataset row is malformed")
        return b"".join(selected), len(selected)

    @staticmethod
    def _training_config(scratch: str, run_mode: str) -> dict[str, Any]:
        plan = TRAINING_PLANS[run_mode]
        output = f"{scratch}/output/training"
        return {
            "recipe": "ExactTargetDFlash2Recipe",
            "dist_env": {"backend": "nccl", "timeout_minutes": 30},
            "recipe_args": {
                "target_model_name_or_path": str(worker.TARGET_DIR),
                "target_attn_implementation": "sdpa",
                "target_force_hf": True,
                "train_data_path": f"{scratch}/train.jsonl",
                "val_data_path": None,
                "output_dir": output,
                "seq_length": 10240,
                "micro_batch_size": 1,
                "grad_accumulation_steps": plan.get(
                    "grad_accumulation_steps", 1
                ),
                "num_workers": 0,
                "num_epochs": plan["num_epochs"],
                "draft_num_hidden_layers": 5,
                "block_size": 8,
                "num_anchors": plan["num_anchors"],
                "loss_decay_gamma": 4.0,
                "mask_token_id": 248070,
                "target_layer_ids": [5, 19, 33, 47, 61],
                "attention_backend": "flex_attention",
                "draft_sliding_window": 2048,
                "draft_num_attention_heads": 32,
                "draft_num_key_value_heads": 8,
                "draft_head_dim": 128,
                "conv_kernel_size": 2,
                "conv_group_size": 16,
                "selector_rank": 256,
                "selector_top_k": 16,
                "selector_loss_weight": 1.0,
                "trust_remote_code": False,
                "shuffle_seed": 42,
                "log_every_steps": 1,
                "max_grad_norm": 1.0,
                "ckpt_every_steps": (
                    4 if plan.get("training_objective") == "dpace-v1" else None
                ),
                "save_checkpoint_every_epoch": False,
                "adaptation_mode": plan["adaptation_mode"],
                "training_objective": plan.get(
                    "training_objective", "dflash-decay-v1"
                ),
                "dpace_alpha": plan.get("dpace_alpha", 0.5),
                "warm_start_draft_path": str(worker.DRAFT_DIR),
                "warm_start_model_sha256": worker.DRAFT_FILES["model.safetensors"],
                "warm_start_config_sha256": worker.DRAFT_FILES["config.json"],
            },
            "optimizer": {
                "lr": plan.get("learning_rate", 0.0001),
                "betas": [0.9, 0.95],
                "weight_decay": 0.0,
                "warmup_ratio": plan.get("warmup_ratio", 0.05),
                "min_lr_ratio": 0.1,
            },
            "checkpoint": {
                "enabled": True,
                "checkpoint_dir": f"{output}/checkpoints",
                "model_save_format": "safetensors",
                "save_consolidated": True,
                "is_async": False,
                "max_recent_checkpoints": (
                    1 if plan.get("training_objective") == "dpace-v1" else None
                ),
            },
        }

    @staticmethod
    def _stage_sources(scratch: str, sources: dict[str, str]) -> None:
        source_root = f"{scratch}/source"
        make_script = r"""
import os, stat, sys
scratch, source, expected = sys.argv[1:]
assert os.uname().nodename == expected
os.makedirs(scratch, mode=0o700, exist_ok=True); os.chmod(scratch, 0o700)
os.makedirs(source, mode=0o700, exist_ok=True); os.chmod(source, 0o700)
for path in (scratch, source):
 metadata = os.lstat(path)
 assert stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.geteuid() and not metadata.st_mode & 0o077
"""
        made = subprocess.run(
            [
                *_ssh_base(),
                shlex.join(
                    [
                        "/usr/bin/bash",
                        LOW_PRIORITY,
                        REMOTE_PYTHON,
                        "-c",
                        make_script,
                        scratch,
                        source_root,
                        HOSTNAME,
                    ]
                ),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if made.returncode != 0:
            raise QwenDFlashTrainingError("training source root could not be prepared")
        transport = " ".join(shlex.quote(item) for item in _ssh_base()[:-1])
        transfer = subprocess.run(
            [
                "/usr/bin/bash",
                LOW_PRIORITY,
                "/usr/bin/rsync",
                "-aR",
                "--checksum",
                "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
                "--protect-args",
                "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
                "-e",
                transport,
                "--",
                *sources,
                f"aday@{NETWORK_HOST}:{source_root}/",
            ],
            cwd=PACKAGE_ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if transfer.returncode != 0:
            raise QwenDFlashTrainingError("training source staging failed")

    @staticmethod
    def _stage_file(local: Path, remote: str) -> None:
        transport = " ".join(shlex.quote(item) for item in _ssh_base()[:-1])
        transfer = subprocess.run(
            [
                "/usr/bin/bash",
                LOW_PRIORITY,
                "/usr/bin/rsync",
                "-a",
                "--checksum",
                "--chmod=Fu=rw,Fgo=",
                "--protect-args",
                "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
                "-e",
                transport,
                "--",
                str(local),
                f"aday@{NETWORK_HOST}:{remote}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if transfer.returncode != 0:
            raise QwenDFlashTrainingError("training fixture staging failed")

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        if _RUNTIME_RE.fullmatch(context.runtime_id) is None or context.job_id is None:
            raise QwenDFlashTrainingError("training runtime/job identity is malformed")
        payload = self._payload(context.payload)
        run_mode = payload["run_mode"]
        sources = _source_manifest()
        self._profile_identity(context, sources)
        scratch = str(context.scratch_path)
        source_root = f"{scratch}/source"
        request_path = f"{scratch}/dflash-training-request.json"

        dataset, dataset_rows = self._dataset(run_mode)
        config = self._training_config(scratch, run_mode)
        config_bytes = (
            json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        local_dataset = context.run_dir / "train.jsonl"
        local_config = context.run_dir / "training-config.json"
        self._write_private(local_dataset, dataset)
        self._write_private(local_config, config_bytes)
        request = {
            "schema_version": worker.SCHEMA_VERSION,
            "runtime_id": context.runtime_id,
            "job_id": context.job_id,
            "host": HOST,
            "hostname": HOSTNAME,
            "claim_id": context.lease.claim_id,
            "owner": context.lease.owner,
            "physical_gpu": context.lease.physical_gpu,
            "gpu_uuid": context.lease.gpu_uuid,
            "vram_budget_gb": context.lease.vram_budget_gb,
            "exclusive": context.lease.exclusive,
            "min_host_memory_gb": context.profile.min_host_memory_gb,
            "min_host_commit_gb": context.profile.min_host_commit_gb,
            "min_disk_free_gb": context.profile.min_disk_free_gb,
            "min_shm_free_gb": context.profile.min_shm_free_gb,
            "scratch_path": scratch,
            "source_root": source_root,
            "source_files": sources,
            "target_dir": str(worker.TARGET_DIR),
            "target_files": worker.TARGET_FILES,
            "target_tree_sha256": _canonical_sha256(worker.TARGET_FILES),
            "draft_dir": str(worker.DRAFT_DIR),
            "draft_files": worker.DRAFT_FILES,
            "environment_python": str(worker.ENV_PYTHON),
            "environment_sentinels": worker.ENV_SENTINELS,
            "environment_versions": worker.ENV_VERSIONS,
            "environment_sha256": _canonical_sha256(
                {"sentinels": worker.ENV_SENTINELS, "versions": worker.ENV_VERSIONS}
            ),
            "environment_wheel_sha256": worker.ENV_WHEEL_SHA256,
            "dataset_sha256": hashlib.sha256(dataset).hexdigest(),
            "dataset_bytes": len(dataset),
            "dataset_rows": dataset_rows,
            "canonical_dataset_sha256": CANONICAL_DATASET_SHA256,
            "training_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "run_mode": run_mode,
        }
        request_bytes = (
            json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        local_request = context.run_dir / "dflash-training-request.json"
        self._write_private(local_request, request_bytes)

        before_device, _free, _inodes, before_allocated = _remote_metrics(
            scratch, create=True
        )
        self._stage_sources(scratch, sources)
        self._stage_file(local_dataset, f"{scratch}/train.jsonl")
        self._stage_file(local_config, f"{scratch}/training-config.json")
        self._stage_file(local_request, request_path)
        with _PreparationHeartbeat(context):
            preflight = _remote_action(
                source_root,
                "preflight",
                request_path,
                request_sha256,
                timeout=1800,
            )
        if (
            preflight.get("target_tree_sha256") != _canonical_sha256(worker.TARGET_FILES)
            or preflight.get("draft_model_sha256")
            != worker.DRAFT_FILES["model.safetensors"]
            or preflight.get("environment_sha256")
            != request["environment_sha256"]
            or preflight.get("dataset_sha256") != request["dataset_sha256"]
            or preflight.get("dataset_rows") != dataset_rows
            or preflight.get("run_mode") != run_mode
        ):
            raise QwenDFlashTrainingError("training worker preflight identity changed")
        filesystem, free_bytes, free_inodes, allocated = _remote_metrics(
            scratch, create=False
        )
        if filesystem != before_device:
            raise QwenDFlashTrainingError("training worker filesystem changed")
        with self._lock:
            self._prepared[context.runtime_id] = {
                "request_sha256": request_sha256,
                "request_path": request_path,
                "source_root": source_root,
            }
        return StoragePreparationResult(
            scratch_path=context.scratch_path,
            filesystem_id=filesystem,
            free_bytes_after_stage=free_bytes,
            free_inodes_after_stage=free_inodes,
            staged_bytes=max(0, allocated - before_allocated),
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        with self._lock:
            prepared = self._prepared.get(context.runtime_id)
        if prepared is None:
            raise AdapterLaunchError(
                "DFlash2 training preflight receipt is absent", process_absent=True
            )
        try:
            result = _remote_action(
                prepared["source_root"],
                "spawn",
                prepared["request_path"],
                prepared["request_sha256"],
                timeout=60,
            )
            pid = result.get("pid")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
                raise QwenDFlashTrainingError("training PID is malformed")
            context.heartbeat(pid, "Exact Qwen3.8 DFlash2 trainer bound to its PID")
            identity = (
                f"aeon-dflash-adapt:{context.runtime_id}:"
                f"{prepared['request_sha256']}:{pid}"
            )
            return LaunchResult(pid=pid, process_identity=identity)
        except BaseException as exc:
            status = _remote_action(
                prepared["source_root"],
                "status",
                prepared["request_path"],
                prepared["request_sha256"],
                timeout=60,
            )
            if status.get("state") == "absent":
                raise AdapterLaunchError(
                    f"DFlash2 training failed before process creation: {exc}",
                    process_absent=True,
                ) from exc
            raise

    @staticmethod
    def _runtime_identity(runtime: Mapping[str, Any]) -> tuple[str, str, int]:
        match = _PROCESS_IDENTITY_RE.fullmatch(str(runtime.get("process_identity") or ""))
        if (
            match is None
            or match.group(1) != runtime.get("runtime_id")
            or int(match.group(3)) != runtime.get("pid")
            or runtime.get("host") != HOST
            or PurePosixPath(str(runtime.get("run_dir") or "")).parent
            != REMOTE_RUN_ROOT
        ):
            raise QwenDFlashTrainingError("training runtime identity changed")
        return match.group(1), match.group(2), int(match.group(3))

    @classmethod
    def _runtime_action(
        cls,
        runtime: Mapping[str, Any],
        action: str,
        *,
        extra: str | None = None,
        timeout: float = 120,
    ) -> dict[str, Any]:
        _runtime_id, digest, _pid = cls._runtime_identity(runtime)
        scratch = str(runtime["run_dir"])
        return _remote_action(
            f"{scratch}/source",
            action,
            f"{scratch}/dflash-training-request.json",
            digest,
            extra=extra,
            timeout=timeout,
        )

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        try:
            _runtime_id, _digest, pid = self._runtime_identity(runtime)
            status = self._runtime_action(runtime, "status", timeout=60)
        except QwenDFlashTrainingTransportError:
            raise
        except QwenDFlashTrainingError as exc:
            return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))
        state = status.get("state")
        if state == "running":
            if status.get("pid") != pid:
                return ProbeResult(
                    ProbeState.UNKNOWN, False, False, "training PID identity changed"
                )
            return ProbeResult(
                ProbeState.RUNNING,
                True,
                False,
                f"Exact DFlash2 adaptation is {status.get('phase') or 'running'}",
            )
        if state == "completed":
            result = status.get("result") or {}
            return ProbeResult(
                ProbeState.COMPLETED,
                False,
                True,
                f"Exact DFlash2 adaptation completed at step {result.get('global_step')}",
            )
        if state == "failed":
            detail = str((status.get("result") or {}).get("failure") or "training failed")
            return ProbeResult(ProbeState.FAILED, False, True, detail[:500])
        if state == "absent":
            return ProbeResult(ProbeState.ABSENT, False, True, "trainer is absent")
        return ProbeResult(ProbeState.UNKNOWN, False, False, "training state is ambiguous")

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        try:
            result = self._runtime_action(runtime, "stop", timeout=150)
        except QwenDFlashTrainingError as exc:
            return StopResult(False, False, str(exc))
        absent = result.get("process_absent") is True
        return StopResult(absent, True, reason if absent else "trainer is still stopping")

    @staticmethod
    def _local_output_valid(path: Path) -> tuple[bool, str | None]:
        manifest = path / "MANIFEST.sha256"
        if not manifest.is_file():
            return False, None
        metadata = path.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise QwenDFlashTrainingError("canonical training output is unsafe")
        expected = {"MANIFEST.sha256"}
        total = 0
        for line in manifest.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
            if match is None or _SAFE_RELATIVE_RE.fullmatch(match.group(2)) is None:
                raise QwenDFlashTrainingError("canonical output manifest is malformed")
            candidate = path / match.group(2)
            candidate_meta = candidate.lstat()
            if (
                not stat.S_ISREG(candidate_meta.st_mode)
                or candidate_meta.st_uid != os.geteuid()
                or stat.S_ISLNK(candidate_meta.st_mode)
                or _sha256(candidate) != match.group(1)
            ):
                raise QwenDFlashTrainingError("canonical output digest changed")
            total += candidate_meta.st_size
            if total > 20 * 1024**3:
                raise QwenDFlashTrainingError("canonical output exceeded its bound")
            expected.add(match.group(2))
        actual: set[str] = {"MANIFEST.sha256"}
        for item in path.rglob("*"):
            item_meta = item.lstat()
            if stat.S_ISDIR(item_meta.st_mode):
                if item_meta.st_uid != os.geteuid() or item_meta.st_mode & 0o022:
                    raise QwenDFlashTrainingError("canonical output directory is unsafe")
                continue
            if not stat.S_ISREG(item_meta.st_mode) or stat.S_ISLNK(item_meta.st_mode):
                raise QwenDFlashTrainingError("canonical output contains an unsafe inode")
            actual.add(item.relative_to(path).as_posix())
        if actual != expected:
            raise QwenDFlashTrainingError("canonical output file set changed")
        return True, _sha256(manifest)

    @staticmethod
    def _copy_output(remote: str, local: Path) -> None:
        local.mkdir(mode=0o700, parents=True, exist_ok=True)
        local.chmod(0o700)
        transport = " ".join(shlex.quote(item) for item in _ssh_base()[:-1])
        result = subprocess.run(
            [
                "/usr/bin/bash",
                LOW_PRIORITY,
                "/usr/bin/rsync",
                "-a",
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
            timeout=1800,
        )
        if result.returncode != 0:
            raise QwenDFlashTrainingError("training output settlement transfer failed")

    @staticmethod
    def _prelaunch_identity(
        runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> tuple[str, str | None]:
        runtime_id = str(runtime.get("runtime_id") or "")
        scratch = str(runtime.get("run_dir") or "")
        canonical = Path(str(storage.get("canonical_output_path") or ""))
        if (
            _RUNTIME_RE.fullmatch(runtime_id) is None
            or runtime.get("host") != HOST
            or PurePosixPath(scratch).parent != REMOTE_RUN_ROOT
            or PurePosixPath(scratch).name != runtime_id
            or str(storage.get("scratch_path") or "") != scratch
            or runtime.get("process_identity") is not None
            or runtime.get("pid") is not None
            or runtime.get("process_absent") not in {1, True}
            or storage.get("terminal_success") not in {0, False}
            or canonical.exists()
        ):
            raise QwenDFlashTrainingError("prelaunch failure identity changed")
        request_path = Path(scratch) / "dflash-training-request.json"
        if not request_path.exists():
            return scratch, None
        metadata = request_path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or not 1 <= metadata.st_size <= 8 * 1024 * 1024
        ):
            raise QwenDFlashTrainingError("prelaunch request is unsafe")
        payload = request_path.read_bytes()
        request = json.loads(payload)
        if (
            request.get("runtime_id") != runtime_id
            or request.get("host") != HOST
            or request.get("scratch_path") != scratch
            or request.get("source_root") != f"{scratch}/source"
        ):
            raise QwenDFlashTrainingError("prelaunch request binding changed")
        return scratch, hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _cleanup_prelaunch(scratch: str, digest: str | None) -> int:
        script = r"""
import hashlib, json, os, pathlib, shutil, stat, sys
scratch, expected_name, expected_digest, runtime_id = sys.argv[1:]
assert os.uname().nodename == expected_name
root = pathlib.Path(scratch)
assert root.parent == pathlib.Path("/home/aday/.local/state/fleet-compute/runs") and root.name == runtime_id
try: root_meta = root.lstat()
except FileNotFoundError:
 print(json.dumps({"state":"absent", "reclaimed_bytes":0})); raise SystemExit(0)
assert expected_digest != "absent"
assert stat.S_ISDIR(root_meta.st_mode) and root_meta.st_uid == os.geteuid() and not root_meta.st_mode & 0o077
request_path = root / "dflash-training-request.json"
payload = request_path.read_bytes(); request_meta = request_path.lstat()
assert stat.S_ISREG(request_meta.st_mode) and request_meta.st_uid == os.geteuid() and not request_meta.st_mode & 0o077
assert hashlib.sha256(payload).hexdigest() == expected_digest
request = json.loads(payload)
assert request.get("runtime_id") == runtime_id and request.get("scratch_path") == scratch
assert request.get("host") == "192.168.0.179" and request.get("source_root") == f"{scratch}/source"
for relative in ("spawn.json", "settled.json", "output/result.json", "output/MANIFEST.sha256"):
 assert not (root / relative).exists() and not (root / relative).is_symlink()
reclaimed = root_meta.st_size
for item in root.rglob("*"):
 metadata = item.lstat(); assert metadata.st_uid == os.geteuid() and not stat.S_ISLNK(metadata.st_mode)
 assert stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode); reclaimed += metadata.st_size
shutil.rmtree(root)
print(json.dumps({"state":"cleaned", "reclaimed_bytes":reclaimed}, sort_keys=True))
"""
        result = subprocess.run(
            [
                *_ssh_base(),
                shlex.join(
                    [
                        "/usr/bin/bash",
                        LOW_PRIORITY,
                        REMOTE_PYTHON,
                        "-c",
                        script,
                        scratch,
                        HOSTNAME,
                        digest or "absent",
                        PurePosixPath(scratch).name,
                    ]
                ),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=300,
        )
        try:
            value = json.loads(result.stdout)
            reclaimed = value["reclaimed_bytes"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise QwenDFlashTrainingError("prelaunch cleanup receipt is malformed") from exc
        if (
            result.returncode != 0
            or value.get("state") not in {"absent", "cleaned"}
            or isinstance(reclaimed, bool)
            or not isinstance(reclaimed, int)
            or reclaimed < 0
        ):
            raise QwenDFlashTrainingError("prelaunch cleanup was not verified")
        return reclaimed

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        if runtime.get("process_identity") is None:
            scratch, digest = self._prelaunch_identity(runtime, storage)
            reclaimed = self._cleanup_prelaunch(scratch, digest)
            return StorageFinalizationResult(
                True,
                True,
                reclaimed,
                "DFlash2 prelaunch scratch cleaned; no process was created",
            )
        self._runtime_identity(runtime)
        canonical = Path(str(storage["canonical_output_path"]))
        valid, local_manifest = (
            self._local_output_valid(canonical) if canonical.exists() else (False, None)
        )
        scratch = str(runtime["run_dir"])
        try:
            _remote_metrics(scratch, create=False)
        except FileNotFoundError:
            if valid:
                return StorageFinalizationResult(
                    True, True, 0, "training output already settled and scratch absent"
                )
            raise QwenDFlashTrainingError("training scratch vanished before settlement")
        status = self._runtime_action(runtime, "settle-status", timeout=300)
        manifest_sha = str(status.get("manifest_sha256") or "")
        if not re.fullmatch(r"[a-f0-9]{64}", manifest_sha):
            raise QwenDFlashTrainingError("training output manifest digest is malformed")
        if not valid:
            self._copy_output(f"{scratch}/output", canonical)
            valid, local_manifest = self._local_output_valid(canonical)
        if not valid or local_manifest != manifest_sha:
            raise QwenDFlashTrainingError("canonical output differs from worker manifest")
        self._runtime_action(runtime, "mark-settled", extra=manifest_sha, timeout=300)
        cleaned = self._runtime_action(
            runtime, "cleanup", extra=manifest_sha, timeout=600
        )
        reclaimed = cleaned.get("reclaimed_bytes")
        if isinstance(reclaimed, bool) or not isinstance(reclaimed, int) or reclaimed < 0:
            raise QwenDFlashTrainingError("training cleanup receipt is malformed")
        return StorageFinalizationResult(
            True, True, reclaimed, "DFlash2 output settled durably on .177"
        )


def create_fleet_adapter() -> AeonQwenDFlashTrainingAdapter:
    return AeonQwenDFlashTrainingAdapter()
