"""Disabled-by-default Fleet lane for MTP-only NVFP4 conversion on .177 GPU 0."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import subprocess
import threading
from typing import Any, Mapping

from fleet_compute.adapters import AdapterLaunchError, RuntimeContext
from fleet_compute.models import (
    LaunchResult, ProbeResult, ProbeState, StopResult,
    StorageFinalizationResult, StoragePreparationResult,
)

from aeon.scripts import quantize_qwen38_flash_next_mtp_nvfp4 as converter
from aeon.scripts import qwen_flash_next_mtp_quant_worker as worker


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "aeon-qwen38-flash-next-mtp-nvfp4-build"
PROJECT = PROFILE_ID
ADAPTER_ID = "aeon-qwen38-flash-next-mtp-nvfp4-build-v1"
HOST = worker.HOST
HOSTNAME = worker.HOSTNAME
LOW_PRIORITY = Path("/home/aday/bin/fleet-low-priority")
MODELOPT_WHEEL = Path(
    "/home/aday/.local/state/aeon-qwen38-quant/sources/"
    "nvidia_modelopt-0.46.0-py3-none-any.whl"
)
CLOSURE_ROOT = Path(
    "/home/aday/.local/state/aeon-flash-next/models/mazinb-uncensored-closures"
)
DERIVATIVE_ROOT = Path(
    "/home/aday/.local/state/aeon-flash-next/models/mazinb-uncensored-mtp-nvfp4"
)
ARTIFACT_ROOT = Path(
    "/home/aday/.local/state/fleet-compute/artifacts/"
    "aeon-qwen38-flash-next-mtp-nvfp4-build"
)
RUN_ROOT = PurePosixPath("/home/aday/.local/state/fleet-compute/runs")
SOURCE_FILES = (
    "aeon/__init__.py",
    "aeon/core/__init__.py",
    "aeon/core/qwen_flash_next_mtp_quant_adapter.py",
    "aeon/scripts/__init__.py",
    "aeon/scripts/build_qwen38_flash_next_nvfp4.py",
    "aeon/scripts/quantize_qwen38_flash_next_mtp_nvfp4.py",
    "aeon/scripts/qwen_flash_next_mtp_quant_worker.py",
)
_RUNTIME = re.compile(r"^fr-[a-f0-9]{32}$")
_REVISION = re.compile(r"^[a-f0-9]{40,64}$")
_SHA = re.compile(r"^[a-f0-9]{64}$")
_PROCESS = re.compile(
    r"^aeon-mtp-nvfp4:(fr-[a-f0-9]{32}):([a-f0-9]{64}):([0-9]+)$"
)


class MTPNVFP4FleetError(RuntimeError):
    pass


class _PreparationHeartbeat:
    def __init__(self, context: RuntimeContext) -> None:
        self.context = context
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_PreparationHeartbeat":
        self.context.heartbeat(None, "MTP NVFP4 source closure verification started")
        self.thread.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        if self.error is not None and _type is None:
            raise MTPNVFP4FleetError("preparation heartbeat failed") from self.error

    def _run(self) -> None:
        while not self.stop.wait(45):
            try:
                self.context.heartbeat(None, "MTP NVFP4 source verification/staging active")
            except BaseException as exc:
                self.error = exc
                return


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


def _source_manifest() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = PACKAGE_ROOT / relative
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
        ):
            raise MTPNVFP4FleetError(f"adapter source is unsafe: {relative}")
        result[relative] = _sha256(path)
    return result


def expected_artifact_identity(
    sources: Mapping[str, str] | None = None,
) -> dict[str, str]:
    return {
        "modelopt_wheel": converter.base.MODELOPT_WHEEL_SHA256,
        "source_manifest": _canonical_sha(dict(sources or _source_manifest())),
        "recipe": hashlib.sha256(converter.SCHEMA_VERSION.encode()).hexdigest(),
    }


def _write_private(path: Path, raw: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    if path.exists() or path.is_symlink():
        raise MTPNVFP4FleetError("refusing to replace existing run evidence")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise MTPNVFP4FleetError("short evidence write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _action(
    source_root: Path, action: str, request: Path, digest: str, timeout: float,
) -> dict[str, Any]:
    if action not in {"preflight", "spawn", "status", "stop"}:
        raise MTPNVFP4FleetError("worker action is not reviewed")
    python = source_root.parent / "modelopt-env/bin/python"
    if not python.is_file():
        raise MTPNVFP4FleetError("pinned ModelOpt environment is absent")
    result = subprocess.run(
        [
            "/usr/bin/bash", str(LOW_PRIORITY), "/usr/bin/env", "-i",
            "HOME=/home/aday",
            "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
            "LANG=C", "LC_ALL=C", f"PYTHONPATH={source_root}",
            "PYTHONDONTWRITEBYTECODE=1", str(python),
            str(source_root / "aeon/scripts/qwen_flash_next_mtp_quant_worker.py"),
            action, str(request), digest,
        ],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout,
    )
    if len(result.stdout) > 4 * 1024 * 1024 or len(result.stderr) > 256 * 1024:
        raise MTPNVFP4FleetError("worker response exceeded its bound")
    try:
        reply = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MTPNVFP4FleetError("worker returned no valid response") from exc
    if result.returncode or not isinstance(reply, dict) or reply.get("ok") is not True:
        detail = reply.get("detail") if isinstance(reply, dict) else None
        raise MTPNVFP4FleetError(str(detail or "worker action failed")[:500])
    response = reply.get("result")
    if not isinstance(response, dict):
        raise MTPNVFP4FleetError("worker response is malformed")
    return response


class AeonQwenFlashNextMTPNVFP4Adapter:
    def __init__(self) -> None:
        self._prepared: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _payload(payload: Mapping[str, Any]) -> tuple[Path, str, str, Path]:
        if not isinstance(payload, Mapping) or set(payload) != {
            "source_path", "source_manifest_sha256", "source_revision"
        }:
            raise MTPNVFP4FleetError("payload must bind one closed source")
        source = Path(str(payload["source_path"]))
        manifest = str(payload["source_manifest_sha256"])
        revision = str(payload["source_revision"])
        if (
            not source.is_absolute()
            or source.parent != CLOSURE_ROOT
            or source.name in {"", ".", ".."}
            or _SHA.fullmatch(manifest) is None
            or _REVISION.fullmatch(revision) is None
        ):
            raise MTPNVFP4FleetError("source closure binding is malformed")
        destination = DERIVATIVE_ROOT / (
            f"Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP-All-NVFP4-v4-{revision[:12]}"
        )
        return source, manifest, revision, destination

    @staticmethod
    def _contract(context: RuntimeContext, sources: Mapping[str, str]) -> None:
        lease = context.lease
        if (
            _RUNTIME.fullmatch(context.runtime_id) is None
            or context.job_id is None
            or socket.gethostname() != HOSTNAME
            or context.profile.profile_id != PROFILE_ID
            or context.profile.project != PROJECT
            or context.profile.adapter != ADAPTER_ID
            or context.profile.enabled is not True
            or context.profile.artifact_identity != expected_artifact_identity(sources)
            or lease.host != HOST
            or lease.physical_gpu != 0
            or lease.exclusive is not True
            or lease.memory_total_mib is None
            or lease.memory_total_mib < 94 * 1024
            or abs(lease.vram_budget_gb - worker.VRAM_CAP_GIB) > 1e-9
            or context.scratch_path is not None
            or context.run_dir != Path(lease.run_dir)
            or PurePosixPath(lease.run_dir).parent != RUN_ROOT
            or context.canonical_output_path.parent != ARTIFACT_ROOT
        ):
            raise MTPNVFP4FleetError("Fleet/profile/lease contract changed")

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        source, manifest, revision, destination = self._payload(context.payload)
        sources = _source_manifest()
        self._contract(context, sources)
        with _PreparationHeartbeat(context):
            source_metadata = source.lstat()
            if (
                not stat.S_ISDIR(source_metadata.st_mode)
                or stat.S_ISLNK(source_metadata.st_mode)
                or source_metadata.st_uid != os.geteuid()
                or source_metadata.st_mode & 0o077
            ):
                raise MTPNVFP4FleetError("clean source closure root is unsafe")
            if _sha256(MODELOPT_WHEEL) != converter.base.MODELOPT_WHEEL_SHA256:
                raise MTPNVFP4FleetError("pinned ModelOpt wheel identity changed")
            if destination.exists() or destination.is_symlink():
                raise MTPNVFP4FleetError("unique destination already exists")
            DERIVATIVE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
            derivative_metadata = DERIVATIVE_ROOT.lstat()
            if (
                not stat.S_ISDIR(derivative_metadata.st_mode)
                or derivative_metadata.st_uid != os.geteuid()
                or derivative_metadata.st_mode & 0o077
            ):
                raise MTPNVFP4FleetError("derivative root is unsafe")
            if source.stat().st_dev != DERIVATIVE_ROOT.stat().st_dev:
                raise MTPNVFP4FleetError("source and derivative roots are not one filesystem")
            evidence = context.canonical_output_path
            source_root = evidence / "source"
            source_root.mkdir(mode=0o700, parents=True, exist_ok=False)
            for relative, digest in sources.items():
                target = source_root / relative
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                shutil.copyfile(PACKAGE_ROOT / relative, target)
                target.chmod(0o600)
                if _sha256(target) != digest:
                    raise MTPNVFP4FleetError("staged adapter source changed")
            staged_wheel = evidence / MODELOPT_WHEEL.name
            shutil.copyfile(MODELOPT_WHEEL, staged_wheel)
            staged_wheel.chmod(0o400)
            if _sha256(staged_wheel) != converter.base.MODELOPT_WHEEL_SHA256:
                raise MTPNVFP4FleetError("staged ModelOpt wheel identity changed")
            environment = evidence / "modelopt-env"
            created = subprocess.run(
                [
                    "/usr/bin/bash", str(LOW_PRIORITY), "/usr/bin/python3", "-m",
                    "venv", "--system-site-packages", str(environment),
                ],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300,
            )
            if created.returncode:
                raise MTPNVFP4FleetError("pinned ModelOpt environment creation failed")
            installed = subprocess.run(
                [
                    "/usr/bin/bash", str(LOW_PRIORITY),
                    str(environment / "bin/python"), "-m", "pip", "install",
                    "--no-deps", "--disable-pip-version-check", str(staged_wheel),
                ],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600,
            )
            if installed.returncode:
                raise MTPNVFP4FleetError("pinned ModelOpt wheel installation failed")
        request = {
            "schema_version": worker.SCHEMA, "runtime_id": context.runtime_id,
            "job_id": context.job_id, "host": HOST, "hostname": HOSTNAME,
            "claim_id": context.lease.claim_id, "gpu_uuid": context.lease.gpu_uuid,
            "physical_gpu": 0, "vram_cap_gib": worker.VRAM_CAP_GIB,
            "exclusive": True, "source_path": str(source),
            "source_manifest_sha256": manifest, "source_revision": revision,
            "destination_path": str(destination), "modelopt_wheel": str(staged_wheel),
            "modelopt_wheel_sha256": converter.base.MODELOPT_WHEEL_SHA256,
            "source_files": sources,
        }
        raw = (json.dumps(request, sort_keys=True, allow_nan=False) + "\n").encode()
        request_path = evidence / "mtp-quant-request.json"
        _write_private(request_path, raw)
        digest = hashlib.sha256(raw).hexdigest()
        with _PreparationHeartbeat(context):
            preflight = _action(source_root, "preflight", request_path, digest, 7200)
        if (
            preflight.get("request_sha256") != digest
            or preflight.get("source_manifest_sha256") != manifest
            or preflight.get("source_revision") != revision
            or preflight.get("modelopt_wheel_sha256")
            != converter.base.MODELOPT_WHEEL_SHA256
        ):
            raise MTPNVFP4FleetError("worker preflight receipt changed")
        values = os.statvfs(evidence)
        staged = (
            sum((source_root / name).stat().st_size for name in SOURCE_FILES)
            + staged_wheel.stat().st_size + len(raw)
        )
        with self._lock:
            self._prepared[context.runtime_id] = {
                "source_root": source_root, "request": request_path, "digest": digest,
            }
        return StoragePreparationResult(
            scratch_path=context.scratch_path, filesystem_id=str(evidence.stat().st_dev),
            free_bytes_after_stage=values.f_bavail * values.f_frsize,
            free_inodes_after_stage=values.f_favail, staged_bytes=staged,
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        with self._lock:
            prepared = self._prepared.get(context.runtime_id)
        if prepared is None:
            raise AdapterLaunchError("MTP conversion preflight is absent", process_absent=True)
        result = _action(
            prepared["source_root"], "spawn", prepared["request"], prepared["digest"], 120
        )
        pid = result.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
            raise MTPNVFP4FleetError("worker PID is malformed")
        context.heartbeat(pid, "MTP-only NVFP4 converter bound to exact Fleet lease")
        return LaunchResult(
            pid=pid,
            process_identity=f"aeon-mtp-nvfp4:{context.runtime_id}:{prepared['digest']}:{pid}",
        )

    @staticmethod
    def _runtime(runtime: Mapping[str, Any]) -> tuple[Path, Path, str, int]:
        match = _PROCESS.fullmatch(str(runtime.get("process_identity") or ""))
        runtime_id = str(runtime.get("runtime_id") or "")
        canonical = ARTIFACT_ROOT / runtime_id
        if (
            match is None or match.group(1) != runtime_id
            or int(match.group(3)) != runtime.get("pid") or runtime.get("host") != HOST
            or PurePosixPath(str(runtime.get("run_dir") or "")).parent != RUN_ROOT
        ):
            raise MTPNVFP4FleetError("runtime identity changed")
        return canonical / "source", canonical / "mtp-quant-request.json", match.group(2), int(match.group(3))

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        try:
            source, request, digest, pid = self._runtime(runtime)
            status = _action(source, "status", request, digest, 120)
        except MTPNVFP4FleetError as exc:
            return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))
        state = status.get("state")
        if state == "running" and status.get("pid") == pid:
            return ProbeResult(ProbeState.RUNNING, True, False, "MTP conversion running")
        if state == "completed":
            return ProbeResult(ProbeState.COMPLETED, False, True, "MTP conversion completed")
        if state == "failed":
            return ProbeResult(ProbeState.FAILED, False, True, str(status.get("failure") or "conversion failed")[:500])
        if state == "absent":
            return ProbeResult(ProbeState.ABSENT, False, True, "converter absent")
        return ProbeResult(ProbeState.UNKNOWN, False, False, "converter state ambiguous")

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        try:
            source, request, digest, _pid = self._runtime(runtime)
            result = _action(source, "stop", request, digest, 180)
        except MTPNVFP4FleetError as exc:
            return StopResult(False, False, str(exc))
        absent = result.get("process_absent") is True
        return StopResult(absent, True, reason if absent else "converter is still stopping")

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        source_root, request_path, digest, _pid = self._runtime(runtime)
        status = _action(source_root, "status", request_path, digest, 120)
        if status.get("state") != "completed":
            raise MTPNVFP4FleetError("conversion is not complete")
        request = worker.load_request(request_path, digest)
        destination = Path(str(request["destination_path"]))
        manifest = destination / "SHA256SUMS"
        if (
            destination.parent != DERIVATIVE_ROOT
            or not manifest.is_file()
            or _sha256(manifest) != status.get("sha256sums_sha256")
        ):
            raise MTPNVFP4FleetError("settled derivative identity changed")
        converter._source_closure(destination, str(status["sha256sums_sha256"]))
        receipt = {
            "schema_version": "aeon-qwen38-flash-next-mtp-nvfp4-settlement-v1",
            "runtime_id": runtime["runtime_id"], "request_sha256": digest,
            "destination_path": str(destination),
            "sha256sums_sha256": status["sha256sums_sha256"],
            "source_manifest_sha256": request["source_manifest_sha256"],
            "source_revision": request["source_revision"],
        }
        receipt_path = Path(str(storage["canonical_output_path"])) / "settlement-receipt.json"
        _write_private(
            receipt_path,
            (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        )
        return StorageFinalizationResult(
            True, True, 0,
            "derivative and exact manifests retained on canonical .177; no automatic deletion",
        )


def create_fleet_adapter() -> AeonQwenFlashNextMTPNVFP4Adapter:
    return AeonQwenFlashNextMTPNVFP4Adapter()
