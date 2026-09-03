"""Fleet batch adapter for the local Flash-Next vLLM release canary."""

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
    LaunchResult, ProbeResult, ProbeState, StopResult,
    StorageFinalizationResult, StoragePreparationResult,
)

from aeon.core import qwen_flash_next_vllm_contract as contract
from aeon.scripts import qwen_flash_next_container_supervisor as cuda_supervisor
from aeon.scripts import qwen_flash_next_vllm_canary_worker as worker
from aeon.scripts import qualify_qwen38_flash_next_vllm as harness


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = contract.PROFILE_ID
PROJECT = contract.PROFILE_ID
LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
PYTHON = "/usr/bin/python3"
PREFLIGHT_TIMEOUT_SECONDS = 1800
SPAWN_TIMEOUT_SECONDS = 2400
CANONICAL_OUTPUT_ROOT = Path(worker.CANONICAL_OUTPUT_ROOT)
RUN_ROOT = worker.RUN_ROOT
ASSET_ROOT = Path("/home/aday/.local/state/aeon-flash-next/qualification-assets")
CHECKPOINT_ROOT = Path("/home/aday/.local/state/aeon-flash-next/models")
IMAGE_ARCHIVE_ROOT = Path(
    "/home/aday/.local/state/aeon-flash-next/runtime-images"
)
SOURCE_FILES = (
    "aeon/__init__.py", "aeon/core/__init__.py",
    "aeon/core/qwen_flash_next_vllm_contract.py",
    "aeon/behavioral_sft/__init__.py",
    "aeon/behavioral_sft/data/eval.jsonl",
    "aeon/scripts/__init__.py",
    "aeon/scripts/qwen_flash_next_container_supervisor.py",
    "aeon/scripts/qualify_qwen38_flash_next_vllm.py",
    "aeon/scripts/qwen_flash_next_vllm_canary_worker.py",
    "aeon/scripts/train_qwen38_flash_next_behavior.py",
)
ASSET_FILES = ("candy.JPG",)
_RUNTIME = re.compile(r"^fr-[0-9a-f]{32}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_PROCESS = re.compile(r"^aeon-vllm-canary:(fr-[0-9a-f]{32}):([0-9a-f]{64}):([0-9]+)$")


class VllmCanaryAdapterError(RuntimeError):
    pass


class VllmCanaryTransportError(VllmCanaryAdapterError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt(path: Path) -> Mapping[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise VllmCanaryAdapterError(f"canary source is unsafe: {path}")
    return {"sha256": _sha256(path), "size": metadata.st_size}


def _source_receipts() -> Mapping[str, Mapping[str, Any]]:
    return {name: _receipt(PACKAGE_ROOT / name) for name in SOURCE_FILES}


def expected_artifact_identity(payload: Mapping[str, str]) -> dict[str, str]:
    return {
        "adapter_source": _sha256(Path(__file__)),
        "worker_source": _sha256(Path(worker.__file__)),
        "harness_source": _sha256(Path(harness.__file__)),
        "cuda_sampler_source": _sha256(Path(cuda_supervisor.__file__)),
        "runtime_contract_source": _sha256(Path(contract.__file__)),
        "source_manifest": hashlib.sha256(json.dumps(_source_receipts(), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "checkpoint_manifest": payload["checkpoint_manifest_sha256"],
        "derived_image": payload["derived_image_digest"].removeprefix("sha256:"),
        "derived_image_config": payload["derived_image_config_digest"],
        "derived_image_archive": payload["derived_image_archive_sha256"],
    }


def _canonical_root(runtime_id: str, *, create: bool) -> Path:
    if _RUNTIME.fullmatch(runtime_id) is None:
        raise VllmCanaryAdapterError("canary runtime ID is malformed")
    parent = CANONICAL_OUTPUT_ROOT.parent
    metadata = parent.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise VllmCanaryAdapterError("canary artifact parent is unsafe")
    CANONICAL_OUTPUT_ROOT.mkdir(mode=0o700, exist_ok=True)
    root = CANONICAL_OUTPUT_ROOT / runtime_id
    if create:
        root.mkdir(mode=0o700)
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode) or root_metadata.st_uid != os.geteuid() or root_metadata.st_mode & 0o077:
        raise VllmCanaryAdapterError("canonical canary run is unsafe")
    return root


def _metrics(root: Path) -> tuple[str, int, int, int]:
    metadata = root.lstat()
    allocated = metadata.st_blocks * 512
    seen = {(metadata.st_dev, metadata.st_ino)}
    for item in root.rglob("*"):
        observed = item.lstat()
        if observed.st_dev != metadata.st_dev or observed.st_uid != os.geteuid() or stat.S_ISLNK(observed.st_mode) or not (stat.S_ISREG(observed.st_mode) or stat.S_ISDIR(observed.st_mode)):
            raise VllmCanaryAdapterError("canonical canary tree contains an unsafe inode")
        identity = (observed.st_dev, observed.st_ino)
        if identity not in seen:
            allocated += observed.st_blocks * 512
            seen.add(identity)
    values = os.statvfs(root)
    return str(metadata.st_dev), values.f_bavail * values.f_frsize, values.f_favail, allocated


def _write_private(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise VllmCanaryAdapterError("canary request write was incomplete")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage(source: Path, destination: Path, *, relative_root: Path | None = None) -> None:
    command = [LOW_PRIORITY, "/usr/bin/rsync", "-a", "-H", "--checksum", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=", "--protect-args"]
    if relative_root is not None:
        command.extend(("-R", "--", str(source.relative_to(relative_root))))
    else:
        command.extend(("--", str(source)))
    command.append(str(destination))
    result = subprocess.run(command, cwd=relative_root, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise VllmCanaryAdapterError("canary staging failed")


def _worker_action(source: Path, action: str, request: Path, digest: str, *, timeout: float = 120) -> Mapping[str, Any]:
    if action not in {"preflight", "spawn", "status", "stop"}:
        raise VllmCanaryAdapterError("canary worker action is invalid")
    command = [
        "/usr/bin/env", "-i", "HOME=/home/aday", "PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C",
        f"PYTHONPATH={source}", "PYTHONDONTWRITEBYTECODE=1", LOW_PRIORITY, PYTHON,
        str(source / "aeon/scripts/qwen_flash_next_vllm_canary_worker.py"), action, str(request), digest,
    ]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VllmCanaryTransportError("local canary worker is unavailable") from exc
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VllmCanaryTransportError("canary worker returned no valid response") from exc
    if result.returncode != 0 or not isinstance(value, Mapping) or value.get("ok") is not True or not isinstance(value.get("result"), Mapping):
        detail = value.get("detail") if isinstance(value, Mapping) else "unknown"
        raise VllmCanaryAdapterError(f"canary worker {action} failed: {detail}")
    return value["result"]


class _Heartbeat:
    def __init__(self, context: RuntimeContext, detail: str) -> None:
        self.context = context
        self.detail = detail
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_Heartbeat":
        self.context.heartbeat(None, self.detail)
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.error is not None:
            raise VllmCanaryAdapterError("canary heartbeat failed") from self.error

    def _run(self) -> None:
        while not self.stop_event.wait(60):
            try:
                self.context.heartbeat(None, self.detail)
            except BaseException as exc:
                self.error = exc
                return


class AeonQwenFlashNextVllmCanaryAdapter:
    def __init__(self) -> None:
        self._prepared: dict[str, tuple[Path, Path, str]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _payload(value: Mapping[str, Any]) -> dict[str, str]:
        if set(value) != {
            "checkpoint_path", "checkpoint_manifest_path",
            "checkpoint_manifest_sha256", "derived_image_digest",
            "derived_image_config_digest", "derived_image_archive_path",
            "derived_image_archive_sha256",
        }:
            raise VllmCanaryAdapterError("canary payload fields changed")
        result = {key: str(item) for key, item in value.items()}
        if any(_SHA.fullmatch(result[key]) is None for key in (
            "checkpoint_manifest_sha256", "derived_image_config_digest",
            "derived_image_archive_sha256",
        )):
            raise VllmCanaryAdapterError("canary artifact digest is malformed")
        if not result["derived_image_digest"].startswith("sha256:") or _SHA.fullmatch(result["derived_image_digest"][7:]) is None:
            raise VllmCanaryAdapterError("derived image digest is malformed")
        checkpoint = Path(result["checkpoint_path"]).resolve(strict=True)
        manifest = Path(result["checkpoint_manifest_path"]).resolve(strict=True)
        archive = Path(result["derived_image_archive_path"]).resolve(strict=True)
        checkpoint.relative_to(CHECKPOINT_ROOT.resolve(strict=True))
        manifest.relative_to(checkpoint)
        archive.relative_to(IMAGE_ARCHIVE_ROOT.resolve(strict=True))
        if _sha256(manifest) != result["checkpoint_manifest_sha256"]:
            raise VllmCanaryAdapterError("checkpoint manifest digest changed")
        if _sha256(archive) != result["derived_image_archive_sha256"]:
            raise VllmCanaryAdapterError("derived image archive digest changed")
        archive_manifest, archive_config = worker._oci_identity(archive)
        if (
            archive_manifest != result["derived_image_digest"]
            or archive_config != result["derived_image_config_digest"]
        ):
            raise VllmCanaryAdapterError(
                "derived OCI manifest/config identity changed"
            )
        result["checkpoint_path"] = str(checkpoint)
        result["checkpoint_manifest_path"] = str(manifest)
        result["derived_image_archive_path"] = str(archive)
        return result

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        if context.job_id is None:
            raise VllmCanaryAdapterError("canary is batch-only")
        payload = self._payload(context.payload)
        lease = context.lease
        if (
            context.profile.profile_id != PROFILE_ID or context.profile.project != PROJECT
            or context.profile.enabled is not True
            or context.profile.artifact_identity != expected_artifact_identity(payload)
            or not context.profile.artifact_identity
            or lease.host != contract.HOST or lease.physical_gpu != contract.PHYSICAL_GPU
            or lease.exclusive is not True or lease.vram_budget_gb != contract.VRAM_CAP_GIB
            or lease.memory_total_mib is None or lease.memory_total_mib < 94 * 1024
            or lease.memory_total_mib / 1024 - lease.vram_budget_gb < 6
            or context.scratch_path is not None
            or context.run_dir != Path(lease.run_dir)
            or PurePosixPath(lease.run_dir) != RUN_ROOT / context.runtime_id
            or context.canonical_output_path != CANONICAL_OUTPUT_ROOT / context.runtime_id
        ):
            raise VllmCanaryAdapterError("canary profile/lease is not exact canonical .177 GPU0")
        root = _canonical_root(context.runtime_id, create=True)
        before_fs, _free, _inodes, before = _metrics(root)
        source = root / "source"
        assets = root / "assets"
        source.mkdir(mode=0o700)
        assets.mkdir(mode=0o700)
        for name in SOURCE_FILES:
            _stage(PACKAGE_ROOT / name, source, relative_root=PACKAGE_ROOT)
        for name in ASSET_FILES:
            _stage(ASSET_ROOT / name, assets / name)
        sources = _source_receipts()
        asset_receipts = {name: _receipt(ASSET_ROOT / name) for name in ASSET_FILES}
        request = {
            "schema_version": worker.SCHEMA, "runtime_id": context.runtime_id,
            "job_id": context.job_id, "host": contract.HOST, "hostname": worker.HOSTNAME,
            "physical_gpu": lease.physical_gpu, "gpu_uuid": lease.gpu_uuid,
            "claim_id": lease.claim_id, "owner": lease.owner, "exclusive": lease.exclusive,
            "vram_cap_gib": lease.vram_budget_gb, "canonical_output_path": str(root),
            "checkpoint_path": payload["checkpoint_path"],
            "checkpoint_manifest_path": payload["checkpoint_manifest_path"],
            "checkpoint_manifest_sha256": payload["checkpoint_manifest_sha256"],
            "derived_image_digest": payload["derived_image_digest"],
            "derived_image_config_digest": payload["derived_image_config_digest"],
            "derived_image_archive_path": payload["derived_image_archive_path"],
            "derived_image_archive_sha256": payload["derived_image_archive_sha256"],
            "served_model": contract.SERVED_MODEL, "runtime": contract.expected_runtime(),
            "source_files": sources, "asset_files": asset_receipts,
        }
        raw = json.dumps(request, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        digest = hashlib.sha256(raw).hexdigest()
        request_path = root / "canary-request.json"
        _write_private(request_path, raw)
        with _Heartbeat(context, "Hash-verifying exact vLLM canary artifacts"):
            preflight = _worker_action(
                source, "preflight", request_path, digest,
                timeout=PREFLIGHT_TIMEOUT_SECONDS,
            )
        if preflight.get("request_sha256") != digest or preflight.get("checkpoint_manifest_sha256") != payload["checkpoint_manifest_sha256"] or preflight.get("derived_image_digest") != payload["derived_image_digest"] or preflight.get("vram_cap_gib") != contract.VRAM_CAP_GIB:
            raise VllmCanaryAdapterError("canary preflight identity changed")
        if preflight.get("derived_image_archive_sha256") != payload["derived_image_archive_sha256"]:
            raise VllmCanaryAdapterError("canary image archive preflight changed")
        filesystem, free, inodes, allocated = _metrics(root)
        if filesystem != before_fs:
            raise VllmCanaryAdapterError("canonical canary filesystem changed")
        with self._lock:
            self._prepared[context.runtime_id] = (source, request_path, digest)
        return StoragePreparationResult(None, filesystem, free, inodes, max(0, allocated - before))

    def launch(self, context: RuntimeContext) -> LaunchResult:
        with self._lock:
            prepared = self._prepared.get(context.runtime_id)
        if prepared is None:
            raise AdapterLaunchError("canary preflight is absent", process_absent=True)
        source, request, digest = prepared
        try:
            with _Heartbeat(
                context, "Revalidating artifacts before vLLM canary supervisor"
            ):
                result = _worker_action(
                    source, "spawn", request, digest,
                    timeout=SPAWN_TIMEOUT_SECONDS,
                )
            pid = result.get("pid")
            if type(pid) is not int or pid <= 1:
                raise VllmCanaryAdapterError("canary supervisor PID is malformed")
            context.heartbeat(pid, "vLLM MTP-off/on release canary is running")
            return LaunchResult(pid, f"aeon-vllm-canary:{context.runtime_id}:{digest}:{pid}")
        except BaseException as exc:
            status = _worker_action(source, "status", request, digest, timeout=30)
            if status.get("state") in {"absent", "completed", "failed"}:
                raise AdapterLaunchError(f"canary launch failed: {exc}", process_absent=True) from exc
            raise

    @staticmethod
    def _identity(runtime: Mapping[str, Any]) -> tuple[str, str, int, Path, Path]:
        runtime_id = str(runtime.get("runtime_id") or "")
        match = _PROCESS.fullmatch(str(runtime.get("process_identity") or ""))
        if (
            match is None or match.group(1) != runtime_id or int(match.group(3)) != runtime.get("pid")
            or runtime.get("profile_id") != PROFILE_ID or runtime.get("host") != contract.HOST
            or runtime.get("physical_gpu") != contract.PHYSICAL_GPU
            or PurePosixPath(str(runtime.get("run_dir") or "")) != RUN_ROOT / runtime_id
        ):
            raise VllmCanaryAdapterError("saved canary runtime identity changed")
        root = _canonical_root(runtime_id, create=False)
        return runtime_id, match.group(2), int(match.group(3)), root / "source", root / "canary-request.json"

    @classmethod
    def _action(cls, runtime: Mapping[str, Any], action: str, *, timeout: float = 120) -> Mapping[str, Any]:
        _runtime_id, digest, _pid, source, request = cls._identity(runtime)
        return _worker_action(source, action, request, digest, timeout=timeout)

    @staticmethod
    def _null_identity_quarantine_status(
        runtime: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Settle only an exactly absent launch whose PID return raced a probe."""

        runtime_id = str(runtime.get("runtime_id") or "")
        if (
            runtime.get("state") != "quarantined"
            or runtime.get("pid") is not None
            or runtime.get("process_identity") is not None
            or _RUNTIME.fullmatch(runtime_id) is None
            or runtime.get("profile_id") != PROFILE_ID
            or runtime.get("host") != contract.HOST
            or runtime.get("physical_gpu") != contract.PHYSICAL_GPU
            or PurePosixPath(str(runtime.get("run_dir") or ""))
            != RUN_ROOT / runtime_id
        ):
            raise VllmCanaryAdapterError(
                "null-identity quarantine is not an exact canary launch race"
            )
        root = _canonical_root(runtime_id, create=False)
        source = root / "source"
        request = root / "canary-request.json"
        digest = _sha256(request)
        status = _worker_action(source, "status", request, digest, timeout=30)
        if status.get("state") not in {"absent", "completed", "failed"}:
            raise VllmCanaryAdapterError(
                "null-identity quarantine still has a live or ambiguous process"
            )
        return status

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        null_identity_terminal = False
        try:
            _runtime_id, _digest, pid, _source, _request = self._identity(runtime)
            status = self._action(runtime, "status", timeout=30)
        except VllmCanaryTransportError:
            raise
        except VllmCanaryAdapterError as exc:
            try:
                status = self._null_identity_quarantine_status(runtime)
            except VllmCanaryAdapterError:
                return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))
            pid = None
            null_identity_terminal = True
        state = status.get("state")
        if null_identity_terminal:
            return ProbeResult(
                ProbeState.ABSENT,
                False,
                True,
                str(status.get("failure") or "PID-less canary launch is absent")[:500],
                True,
            )
        if state == "running" and status.get("pid") == pid:
            return ProbeResult(ProbeState.RUNNING, True, False, "vLLM canary arms are running")
        if state == "completed":
            return ProbeResult(ProbeState.COMPLETED, False, True, "vLLM speed/MTP/semantic/resource gates passed")
        if state == "failed":
            return ProbeResult(ProbeState.FAILED, False, True, str(status.get("failure") or "vLLM canary failed")[:500])
        if state == "absent":
            return ProbeResult(ProbeState.ABSENT, False, True, "vLLM canary supervisor absent")
        return ProbeResult(ProbeState.UNKNOWN, False, False, "vLLM canary lifecycle is ambiguous")

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        try:
            result = self._action(runtime, "stop", timeout=120)
        except VllmCanaryAdapterError as exc:
            return StopResult(False, False, str(exc))
        absent = result.get("process_absent") is True
        return StopResult(absent, True, reason if absent else "vLLM canary is still stopping")

    def finalize_storage(self, runtime: Mapping[str, Any], storage: Mapping[str, Any]) -> StorageFinalizationResult:
        runtime_id, _digest, _pid, _source, _request = self._identity(runtime)
        if storage.get("scratch_path") is not None:
            raise VllmCanaryAdapterError("canonical .177 canary gained worker scratch")
        root = _canonical_root(runtime_id, create=False)
        status = self._action(runtime, "status", timeout=30)
        if status.get("state") == "failed":
            return StorageFinalizationResult(True, True, 0, "failed canary evidence retained on canonical .177 storage; no automatic cleanup")
        if status.get("state") != "completed" or runtime.get("process_absent") != 1:
            raise VllmCanaryAdapterError("canary settlement remains ambiguous")
        manifest = root / "output" / "MANIFEST.sha256"
        expected = str(status.get("manifest_sha256") or "")
        if _SHA.fullmatch(expected) is None or _sha256(manifest) != expected:
            raise VllmCanaryAdapterError("canary output manifest changed")
        return StorageFinalizationResult(True, True, 0, "qualified vLLM evidence settled directly on canonical .177 storage; exact run retained")


def create_fleet_adapter() -> AeonQwenFlashNextVllmCanaryAdapter:
    return AeonQwenFlashNextVllmCanaryAdapter()
