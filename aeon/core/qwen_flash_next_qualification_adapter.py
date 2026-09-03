"""Fleet batch adapter for pre-release Flash-Next qualification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
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

from aeon.scripts import build_qwen38_flash_next_nvfp4 as builder
from aeon.scripts import materialize_qwen38_flash_next_ple as ple_materializer
from aeon.scripts import qwen_flash_next_qualification_worker as worker
from aeon.scripts import qwen_flash_next_container_supervisor as container_supervisor
from aeon.scripts import qualify_qwen38_flash_next_endpoint as harness
from aeon.scripts import release_qwen38_flash_next as release_tool


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "aeon-qwen38-flash-next-qualification"
PROJECT = "aeon-qwen38-flash-next-qualification"
HOST = worker.HOST
HOSTNAME = worker.HOSTNAME
LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
WORKER_PYTHON = (
    "/home/aday/.aeon/runtime/qwen38/training-envs/"
    "nemo-9fb92970-torch291-cu128/bin/python"
)
RUN_ROOT = worker.SCRATCH_ROOT
CANONICAL_OUTPUT_ROOT = Path(worker.CANONICAL_OUTPUT_ROOT)
CHECKPOINT_ROOT = Path(
    "/home/aday/.local/state/fleet-compute/artifacts/aeon-qwen38-flash-next-build"
)
ASSET_ROOT = Path("/home/aday/.local/state/aeon-flash-next/qualification-assets")
SOURCE_FILES = (
    "aeon/__init__.py",
    "aeon/core/__init__.py",
    "aeon/core/qwen_flash_next_runtime_contract.py",
    "aeon/behavioral_sft/__init__.py",
    "aeon/behavioral_sft/validator.py",
    "aeon/behavioral_sft/data/eval.jsonl",
    "aeon/scripts/__init__.py",
    "aeon/scripts/audit_qwen38_flash_next_passthrough.py",
    "aeon/scripts/build_qwen38_flash_next_nvfp4.py",
    "aeon/scripts/materialize_qwen38_flash_next_ple.py",
    "aeon/scripts/qualify_qwen38_flash_next_endpoint.py",
    "aeon/scripts/release_qwen38_flash_next.py",
    "aeon/scripts/train_qwen38_flash_next_behavior.py",
    "aeon/scripts/qwen_flash_next_qualification_worker.py",
    "aeon/scripts/qwen_flash_next_container_supervisor.py",
)
ASSET_FILES = ("manifest.json", "candy.JPG", "jobs_presenting_ipod.mp4")
_RUNTIME = re.compile(r"^fr-[0-9a-f]{32}$")
_PROCESS = re.compile(
    r"^aeon-flash-next-qualification:(fr-[0-9a-f]{32}):"
    r"([0-9a-f]{64}):([0-9]+)$"
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


class FlashNextQualificationError(RuntimeError):
    pass


class FlashNextQualificationTransportError(FlashNextQualificationError):
    pass


def _canonical_run_path(runtime_id: str, value: str | Path | None = None) -> Path:
    if _RUNTIME.fullmatch(runtime_id) is None:
        raise FlashNextQualificationError("qualification runtime ID is malformed")
    expected = CANONICAL_OUTPUT_ROOT / runtime_id
    if value is not None and Path(str(value)) != expected:
        raise FlashNextQualificationError("canonical qualification path changed")
    return expected


def _canonical_root(*, create: bool) -> None:
    parent = CANONICAL_OUTPUT_ROOT.parent
    try:
        parent_metadata = parent.lstat()
    except FileNotFoundError as exc:
        raise FlashNextQualificationError(
            "canonical qualification parent is absent"
        ) from exc
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or stat.S_ISLNK(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or parent_metadata.st_mode & 0o077
        or os.path.ismount(parent)
        or parent.resolve(strict=True) != parent
    ):
        raise FlashNextQualificationError("canonical qualification parent is unsafe")
    if create:
        try:
            CANONICAL_OUTPUT_ROOT.mkdir(mode=0o700, parents=False, exist_ok=False)
        except FileExistsError:
            pass
    metadata = CANONICAL_OUTPUT_ROOT.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or metadata.st_dev != parent_metadata.st_dev
        or os.path.ismount(CANONICAL_OUTPUT_ROOT)
        or CANONICAL_OUTPUT_ROOT.resolve(strict=True) != CANONICAL_OUTPUT_ROOT
    ):
        raise FlashNextQualificationError("canonical qualification root is unsafe")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _receipt(path: Path, *, allow_empty: bool = False) -> dict[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
        or (metadata.st_size == 0 and not allow_empty)
    ):
        raise FlashNextQualificationError(f"qualification source is unsafe: {path}")
    return {"sha256": _sha256(path), "size": metadata.st_size}


def _source_receipts() -> dict[str, dict[str, Any]]:
    return {
        name: _receipt(
            PACKAGE_ROOT / name,
            allow_empty=name in worker.EMPTY_SOURCE_FILES,
        )
        for name in SOURCE_FILES
    }


def _asset_receipts() -> dict[str, dict[str, Any]]:
    return {name: _receipt(ASSET_ROOT / name) for name in ASSET_FILES}


def expected_artifact_identity() -> dict[str, str]:
    return {
        "adapter_source": _sha256(Path(__file__)),
        "builder_source": _sha256(Path(builder.__file__)),
        "harness_source": _sha256(Path(harness.__file__)),
        "image": worker.IMAGE_DIGEST.removeprefix("sha256:"),
        "image_archive": worker.IMAGE_ARCHIVE_SHA256,
        "image_config": worker.IMAGE_CONFIG_DIGEST.removeprefix("sha256:"),
        "image_local_id": worker.IMAGE_ID.removeprefix("sha256:"),
        "materializer_source": _sha256(Path(ple_materializer.__file__)),
        "qualification_assets_manifest": _sha256(ASSET_ROOT / "manifest.json"),
        "release_validator_source": _sha256(Path(release_tool.__file__)),
        "runtime_contract_source": _sha256(
            PACKAGE_ROOT / "aeon/core/qwen_flash_next_runtime_contract.py"
        ),
        "source_manifest": _canonical_sha(_source_receipts()),
        "sglang_source_commit": hashlib.sha256(
            worker.SGLANG_COMMIT.encode("ascii")
        ).hexdigest(),
        "worker_source": _sha256(Path(worker.__file__)),
        "container_supervisor_source": _sha256(Path(container_supervisor.__file__)),
    }


def _worker_action(
    source: str,
    action: str,
    request: str,
    digest: str,
    extra: str | None = None,
    *,
    timeout: float = 120,
) -> dict[str, Any]:
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
        raise FlashNextQualificationError("qualification worker action is invalid")
    command = [
        "/usr/bin/env",
        "-i",
        "HOME=/home/aday",
        "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "LANG=C",
        "LC_ALL=C",
        f"PYTHONPATH={source}",
        "PYTHONDONTWRITEBYTECODE=1",
        LOW_PRIORITY,
        WORKER_PYTHON,
        f"{source}/aeon/scripts/qwen_flash_next_qualification_worker.py",
        action,
        request,
        digest,
    ]
    if extra is not None:
        command.append(extra)
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FlashNextQualificationTransportError(
            "local qualification worker is unavailable"
        ) from exc
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FlashNextQualificationTransportError(
            "qualification worker returned no valid response"
        ) from exc
    if (
        result.returncode != 0
        or not isinstance(value, dict)
        or value.get("ok") is not True
    ):
        detail = value.get("detail") if isinstance(value, dict) else "unknown"
        raise FlashNextQualificationError(f"worker {action} failed: {detail}")
    response = value.get("result")
    if not isinstance(response, dict):
        raise FlashNextQualificationError("qualification worker response is malformed")
    return response


def _metrics(path: str, *, create: bool) -> tuple[str, int, int, int]:
    target = Path(path)
    pure = PurePosixPath(path)
    if (
        pure.parent != PurePosixPath(CANONICAL_OUTPUT_ROOT)
        or _RUNTIME.fullmatch(pure.name) is None
    ):
        raise FlashNextQualificationError("canonical qualification path changed")
    _canonical_root(create=create)
    if create:
        target.mkdir(mode=0o700, parents=False, exist_ok=False)
        target.chmod(0o700)
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        raise
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or os.path.ismount(target)
    ):
        raise FlashNextQualificationError("canonical qualification path is unsafe")
    allocated = metadata.st_blocks * 512
    seen = {(metadata.st_dev, metadata.st_ino)}
    for item in target.rglob("*"):
        observed = item.lstat()
        if (
            observed.st_uid != os.geteuid()
            or observed.st_dev != metadata.st_dev
            or os.path.ismount(item)
            or stat.S_ISLNK(observed.st_mode)
            or not (stat.S_ISDIR(observed.st_mode) or stat.S_ISREG(observed.st_mode))
        ):
            raise FlashNextQualificationError(
                "canonical qualification tree contains an unsafe inode"
            )
        key = (observed.st_dev, observed.st_ino)
        if key not in seen:
            allocated += observed.st_blocks * 512
            seen.add(key)
    values = os.statvfs(target)
    return (
        str(metadata.st_dev),
        values.f_bavail * values.f_frsize,
        values.f_favail,
        allocated,
    )


def _stage_local(
    source: Path,
    destination: str,
    *,
    relative_root: Path | None = None,
    contents: bool = False,
    timeout: float = 7200,
) -> None:
    item = f"{source}/" if contents else str(source)
    arguments = [
        LOW_PRIORITY,
        "/usr/bin/rsync",
        "-a",
        "-H",
        "--checksum",
        "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
        "--protect-args",
    ]
    if relative_root is not None:
        arguments.extend(("-R", "--", str(source.relative_to(relative_root))))
    else:
        arguments.extend(("--", item))
    arguments.append(destination)
    result = subprocess.run(
        arguments,
        cwd=relative_root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise FlashNextQualificationError("local qualification staging failed")


def _local_dirs(scratch: str) -> None:
    root = Path(scratch)
    if (
        PurePosixPath(scratch).parent != PurePosixPath(CANONICAL_OUTPUT_ROOT)
        or _RUNTIME.fullmatch(root.name) is None
    ):
        raise FlashNextQualificationError("canonical qualification path changed")
    for path in (root, root / "source", root / "assets"):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)
        metadata = path.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or path.is_symlink()
        ):
            raise FlashNextQualificationError(
                "canonical qualification directory preparation failed"
            )


def _write_private(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FlashNextQualificationError("qualification request path already exists")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _Heartbeat:
    def __init__(self, context: RuntimeContext) -> None:
        self.context = context
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_Heartbeat":
        self.context.heartbeat(None, "Staging exact Flash-Next qualification inputs")
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        if self.error is not None:
            raise FlashNextQualificationError(
                "qualification staging heartbeat failed"
            ) from self.error

    def _run(self) -> None:
        while not self.stop.wait(120):
            try:
                self.context.heartbeat(
                    None, "Qualification staging/preflight remains active"
                )
            except BaseException as exc:
                self.error = exc
                return


class AeonQwenFlashNextQualificationAdapter:
    def __init__(self) -> None:
        self._prepared: dict[str, dict[str, str]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping) or set(payload) != {
            "checkpoint_path",
            "checkpoint_tree_sha256",
            "official_untuned_checkpoint_path",
            "official_untuned_checkpoint_tree_sha256",
            "build_sibling_manifest_path",
            "build_sibling_manifest_sha256",
            "builder_sha256",
            "repo_id",
        }:
            raise FlashNextQualificationError("qualification payload fields changed")
        try:
            checkpoint = Path(str(payload["checkpoint_path"])).resolve(strict=True)
            untuned = Path(str(payload["official_untuned_checkpoint_path"])).resolve(
                strict=True
            )
            sibling_manifest = Path(
                str(payload["build_sibling_manifest_path"])
            ).resolve(strict=True)
            checkpoint.relative_to(CHECKPOINT_ROOT.resolve(strict=True))
            untuned.relative_to(CHECKPOINT_ROOT.resolve(strict=True))
            sibling_manifest.relative_to(CHECKPOINT_ROOT.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise FlashNextQualificationError(
                "checkpoint sibling is outside Fleet build artifacts"
            ) from exc
        parent = checkpoint.parent
        if (
            checkpoint.name != worker.TUNED_CHECKPOINT_NAME
            or untuned.name != worker.UNTUNED_CHECKPOINT_NAME
            or sibling_manifest.name != worker.SIBLING_MANIFEST_NAME
            or untuned.parent != parent
            or sibling_manifest.parent != parent
            or parent == CHECKPOINT_ROOT.resolve(strict=True)
        ):
            raise FlashNextQualificationError(
                "tuned/untuned checkpoints do not share one canonical build-job parent"
            )
        tree = str(payload["checkpoint_tree_sha256"])
        untuned_tree = str(payload["official_untuned_checkpoint_tree_sha256"])
        sibling_manifest_sha = str(payload["build_sibling_manifest_sha256"])
        builder_sha = str(payload["builder_sha256"])
        if any(
            _SHA.fullmatch(value) is None
            for value in (tree, untuned_tree, sibling_manifest_sha, builder_sha)
        ):
            raise FlashNextQualificationError(
                "checkpoint/sibling/builder digest is malformed"
            )
        repo_id = str(payload["repo_id"])
        release_tool._validate_repo_id(repo_id)
        return {
            "checkpoint": checkpoint,
            "untuned_checkpoint": untuned,
            "sibling_manifest": sibling_manifest,
            "parent": parent,
            "checkpoint_tree_sha256": tree,
            "official_untuned_checkpoint_tree_sha256": untuned_tree,
            "build_sibling_manifest_sha256": sibling_manifest_sha,
            "builder_sha256": builder_sha,
            "repo_id": repo_id,
        }

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        if _RUNTIME.fullmatch(context.runtime_id) is None or context.job_id is None:
            raise FlashNextQualificationError(
                "qualification runtime/job ID is malformed"
            )
        payload = self._payload(context.payload)
        checkpoint = payload["checkpoint"]
        tree = payload["checkpoint_tree_sha256"]
        builder_sha = payload["builder_sha256"]
        repo_id = payload["repo_id"]
        if (
            context.profile.profile_id != PROFILE_ID
            or context.profile.project != PROJECT
            or context.profile.enabled is not True
            or context.profile.artifact_identity != expected_artifact_identity()
            or any(
                value == "0" * 64
                for value in context.profile.artifact_identity.values()
            )
        ):
            raise FlashNextQualificationError(
                "qualification profile is not source exact"
            )
        lease = context.lease
        if (
            lease.host != HOST
            or lease.physical_gpu != 0
            or lease.memory_total_mib is None
            or lease.memory_total_mib < 94 * 1024
            or "rtx" not in str(lease.model or "").casefold()
            or "6000" not in str(lease.model or "").casefold()
            or abs(lease.vram_budget_gb - worker.VRAM_BUDGET_GB) > 1e-9
            or lease.exclusive is not True
            or context.scratch_path is not None
            or context.run_dir != Path(lease.run_dir)
            or PurePosixPath(lease.run_dir) != RUN_ROOT / context.runtime_id
            or context.canonical_output_path != _canonical_run_path(context.runtime_id)
        ):
            raise FlashNextQualificationError(
                "lease is not exact canonical .177 RTX PRO 6000 GPU0"
            )
        if lease.memory_total_mib / 1024 - worker.VRAM_BUDGET_GB < 6:
            raise FlashNextQualificationError(
                "qualification lease lacks six GiB reserve"
            )
        canonical = release_tool.validate_checkpoint(
            checkpoint,
            expected_builder_sha256=builder_sha,
            verify_hashes=True,
        )
        if canonical.checkpoint_tree_sha256 != tree:
            raise FlashNextQualificationError(
                "canonical checkpoint tree differs from payload"
            )
        worker.validate_sibling_artifact(
            payload["parent"],
            expected_tuned_tree_sha256=tree,
            expected_untuned_tree_sha256=payload[
                "official_untuned_checkpoint_tree_sha256"
            ],
            expected_manifest_sha256=payload["build_sibling_manifest_sha256"],
            verify_hashes=True,
            require_hardlinks=True,
        )
        sources = _source_receipts()
        assets = _asset_receipts()
        if assets["manifest.json"]["sha256"] != (
            "dd8a1138007e0f17ba2ad50f045fd327a0b7bb1714c45d1e1d648434d835547f"
        ):
            raise FlashNextQualificationError("qualification asset manifest changed")
        scratch = str(
            _canonical_run_path(context.runtime_id, context.canonical_output_path)
        )
        source_root = f"{scratch}/source"
        request_path = f"{scratch}/qualification-request.json"
        request = {
            "schema_version": worker.SCHEMA,
            "runtime_id": context.runtime_id,
            "job_id": context.job_id,
            "host": HOST,
            "hostname": HOSTNAME,
            "physical_gpu": lease.physical_gpu,
            "gpu_uuid": lease.gpu_uuid,
            "claim_id": lease.claim_id,
            "owner": lease.owner,
            "vram_budget_gb": lease.vram_budget_gb,
            "exclusive": lease.exclusive,
            "scratch_path": scratch,
            "checkpoint_path": str(payload["checkpoint"]),
            "official_untuned_checkpoint_path": str(payload["untuned_checkpoint"]),
            "build_sibling_manifest_path": str(payload["sibling_manifest"]),
            "checkpoint_tree_sha256": tree,
            "official_untuned_checkpoint_tree_sha256": payload[
                "official_untuned_checkpoint_tree_sha256"
            ],
            "build_sibling_manifest_sha256": payload["build_sibling_manifest_sha256"],
            "builder_sha256": builder_sha,
            "repo_id": repo_id,
            "source_files": sources,
            "asset_files": assets,
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
        raw = (
            json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        digest = hashlib.sha256(raw).hexdigest()
        filesystem, _free, _inodes, before = _metrics(scratch, create=True)
        with _Heartbeat(context):
            _local_dirs(scratch)
            for name in SOURCE_FILES:
                _stage_local(
                    PACKAGE_ROOT / name,
                    f"{source_root}/",
                    relative_root=PACKAGE_ROOT,
                    timeout=600,
                )
            for name in ASSET_FILES:
                _stage_local(
                    ASSET_ROOT / name,
                    f"{scratch}/assets/{name}",
                    timeout=600,
                )
            _write_private(Path(request_path), raw)
            preflight = _worker_action(
                source_root, "preflight", request_path, digest, timeout=14_400
            )
        if (
            preflight.get("checkpoint_tree_sha256") != tree
            or preflight.get("official_untuned_checkpoint_tree_sha256")
            != payload["official_untuned_checkpoint_tree_sha256"]
            or preflight.get("build_sibling_manifest_sha256")
            != payload["build_sibling_manifest_sha256"]
            or preflight.get("sglang_commit") != worker.SGLANG_COMMIT
            or preflight.get("sglang_image_digest") != worker.IMAGE_DIGEST
            or preflight.get("sglang_image_config_digest") != worker.IMAGE_CONFIG_DIGEST
            or preflight.get("sglang_image_id") != worker.IMAGE_ID
            or preflight.get("sglang_image_archive_sha256")
            != worker.IMAGE_ARCHIVE_SHA256
            or preflight.get("max_accounted_vram_gb") != 88.0
            or preflight.get("max_cgroup_memory_gb") != worker.TASK_MEMORY_GB
            or preflight.get("preferred_moe_runner_backend")
            != worker.runtime_contract.PREFERRED_MOE_RUNNER_BACKEND
            or preflight.get("qualification_moe_runner_backends")
            != list(worker.runtime_contract.QUALIFICATION_MOE_RUNNER_BACKENDS)
            or preflight.get("cutlass_nvfp4_scale_duplication_bytes")
            != worker.runtime_contract.CUTLASS_NVFP4_SCALE_DUPLICATION_BYTES
            or preflight.get("cutlass_min_cuda_reserve_bytes")
            != worker.runtime_contract.CUTLASS_MIN_CUDA_RESERVE_BYTES
            or preflight.get("cutlass_min_geometric_mean_speedup")
            != worker.runtime_contract.CUTLASS_MIN_GEOMETRIC_MEAN_SPEEDUP
        ):
            raise FlashNextQualificationError(
                "qualification preflight identity changed"
            )
        after_filesystem, free, inodes, allocated = _metrics(scratch, create=False)
        if after_filesystem != filesystem:
            raise FlashNextQualificationError(
                "canonical qualification filesystem changed"
            )
        with self._lock:
            self._prepared[context.runtime_id] = {
                "digest": digest,
                "request": request_path,
                "source": source_root,
            }
        return StoragePreparationResult(
            None,
            filesystem,
            free,
            inodes,
            max(0, allocated - before),
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        with self._lock:
            prepared = self._prepared.get(context.runtime_id)
        if prepared is None:
            raise AdapterLaunchError(
                "qualification preflight is absent", process_absent=True
            )
        try:
            result = _worker_action(
                prepared["source"],
                "spawn",
                prepared["request"],
                prepared["digest"],
                timeout=90,
            )
            pid = result.get("pid")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
                raise FlashNextQualificationError(
                    "qualification supervisor PID is malformed"
                )
            context.heartbeat(
                pid,
                "Staged selector and fresh final qualification bound to supervisor",
            )
            return LaunchResult(
                pid,
                f"aeon-flash-next-qualification:{context.runtime_id}:"
                f"{prepared['digest']}:{pid}",
            )
        except BaseException as exc:
            status = _worker_action(
                prepared["source"],
                "status",
                prepared["request"],
                prepared["digest"],
                timeout=60,
            )
            if status.get("state") in {"absent", "completed", "failed"}:
                raise AdapterLaunchError(
                    f"qualification failed before a live supervisor remained: {exc}",
                    process_absent=True,
                ) from exc
            raise

    @staticmethod
    def _runtime_id(runtime: Mapping[str, Any]) -> str:
        runtime_id = str(runtime.get("runtime_id") or "")
        if (
            _RUNTIME.fullmatch(runtime_id) is None
            or runtime.get("profile_id") != PROFILE_ID
            or runtime.get("host") != HOST
            or runtime.get("physical_gpu") != 0
            or PurePosixPath(str(runtime.get("run_dir") or "")) != RUN_ROOT / runtime_id
        ):
            raise FlashNextQualificationError(
                "saved qualification runtime identity changed"
            )
        return runtime_id

    @classmethod
    def _identity(cls, runtime: Mapping[str, Any]) -> tuple[str, str, int]:
        runtime_id = cls._runtime_id(runtime)
        match = _PROCESS.fullmatch(str(runtime.get("process_identity") or ""))
        if (
            match is None
            or match.group(1) != runtime_id
            or int(match.group(3)) != runtime.get("pid")
        ):
            raise FlashNextQualificationError(
                "saved qualification runtime identity changed"
            )
        return match.group(1), match.group(2), int(match.group(3))

    @classmethod
    def _action(
        cls,
        runtime: Mapping[str, Any],
        action: str,
        extra: str | None = None,
        *,
        timeout: float = 120,
    ) -> dict[str, Any]:
        runtime_id, digest, _pid = cls._identity(runtime)
        scratch = str(_canonical_run_path(runtime_id))
        return _worker_action(
            f"{scratch}/source",
            action,
            f"{scratch}/qualification-request.json",
            digest,
            extra,
            timeout=timeout,
        )

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        try:
            _runtime, _digest, pid = self._identity(runtime)
            status = self._action(runtime, "status", timeout=90)
        except FlashNextQualificationTransportError:
            raise
        except FlashNextQualificationError as exc:
            return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))
        state = status.get("state")
        if state == "running" and status.get("pid") == pid:
            return ProbeResult(
                ProbeState.RUNNING,
                True,
                False,
                "staged selector/final qualification is running",
            )
        if state == "completed":
            return ProbeResult(
                ProbeState.COMPLETED,
                False,
                True,
                "selector/text/image/video/behavior/MTP/resource qualification passed",
            )
        if state == "failed":
            failure = (status.get("status") or {}).get("failure")
            return ProbeResult(
                ProbeState.FAILED,
                False,
                True,
                str(failure or "qualification failed")[:500],
            )
        if state == "absent":
            return ProbeResult(
                ProbeState.ABSENT, False, True, "qualification supervisor absent"
            )
        return ProbeResult(
            ProbeState.UNKNOWN, False, False, "qualification lifecycle is ambiguous"
        )

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        try:
            result = self._action(runtime, "stop", timeout=120)
        except FlashNextQualificationError as exc:
            return StopResult(False, False, str(exc))
        absent = result.get("process_absent") is True
        return StopResult(
            absent, True, reason if absent else "qualification is still stopping"
        )

    @staticmethod
    def _local_valid(root: Path) -> tuple[bool, str | None]:
        manifest = root / "MANIFEST.sha256"
        if not manifest.is_file():
            return False, None
        metadata = root.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise FlashNextQualificationError(
                "canonical qualification output is unsafe"
            )
        expected = {"MANIFEST.sha256"}
        total = 0
        for line in manifest.read_text(encoding="ascii").splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if match is None or _SAFE_RELATIVE.fullmatch(match.group(2)) is None:
                raise FlashNextQualificationError(
                    "qualification output manifest is malformed"
                )
            item = root / match.group(2)
            item_metadata = item.lstat()
            if (
                not stat.S_ISREG(item_metadata.st_mode)
                or item_metadata.st_uid != os.geteuid()
                or stat.S_ISLNK(item_metadata.st_mode)
                or _sha256(item) != match.group(1)
            ):
                raise FlashNextQualificationError(
                    "canonical qualification evidence changed"
                )
            total += item_metadata.st_size
            expected.add(match.group(2))
        if total > worker.MAX_OUTPUT_BYTES:
            raise FlashNextQualificationError(
                "canonical qualification evidence exceeds bound"
            )
        actual = {
            item.relative_to(root).as_posix()
            for item in root.rglob("*")
            if item.is_file()
        }
        if actual != expected:
            raise FlashNextQualificationError("canonical qualification closure changed")
        return True, _sha256(manifest)

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        runtime_id = self._runtime_id(runtime)
        if "scratch_path" not in storage or storage["scratch_path"] is not None:
            raise FlashNextQualificationError(
                "canonical .177 qualification storage gained worker scratch"
            )
        canonical = _canonical_run_path(
            runtime_id, str(storage.get("canonical_output_path") or "")
        )
        if runtime.get("process_identity") is None:
            try:
                _metrics(str(canonical), create=False)
            except FileNotFoundError:
                return StorageFinalizationResult(
                    True, True, 0, "prelaunch canonical run is absent"
                )
            return StorageFinalizationResult(
                True,
                True,
                0,
                "canonical .177 prelaunch qualification run retained; "
                "no automatic cleanup",
            )
        self._identity(runtime)
        try:
            _metrics(str(canonical), create=False)
        except FileNotFoundError:
            raise FlashNextQualificationError(
                "canonical qualification run vanished before settlement"
            )
        output = canonical / "output"
        valid, local_digest = (
            self._local_valid(output) if output.exists() else (False, None)
        )
        lifecycle = self._action(runtime, "status", timeout=120)
        if lifecycle.get("state") == "failed":
            if (
                storage.get("terminal_success") != 0
                or runtime.get("process_absent") != 1
                or lifecycle.get("pid") is not None
                or (output / "MANIFEST.sha256").exists()
                or (output / "MANIFEST.sha256").is_symlink()
                or (canonical / "qualification-settled.json").exists()
                or (canonical / "qualification-settled.json").is_symlink()
            ):
                raise FlashNextQualificationError(
                    "failed qualification storage outcome remains ambiguous"
                )
            return StorageFinalizationResult(
                True,
                True,
                0,
                "terminal incomplete qualification retained on canonical .177 "
                "storage with no automatic cleanup",
            )
        if lifecycle.get("state") != "completed":
            raise FlashNextQualificationError(
                "qualification storage lifecycle remains ambiguous"
            )
        status = self._action(runtime, "settle-status", timeout=600)
        manifest = str(status.get("manifest_sha256") or "")
        if _SHA.fullmatch(manifest) is None:
            raise FlashNextQualificationError(
                "qualification evidence manifest is malformed"
            )
        if not valid or local_digest != manifest:
            raise FlashNextQualificationError(
                "canonical qualification evidence changed"
            )
        self._action(runtime, "mark-settled", manifest, timeout=600)
        return StorageFinalizationResult(
            True,
            True,
            0,
            "qualification evidence completed directly on canonical .177 storage; "
            "exact local run retained "
            "with no automatic cleanup",
        )


def create_fleet_adapter() -> AeonQwenFlashNextQualificationAdapter:
    return AeonQwenFlashNextQualificationAdapter()
