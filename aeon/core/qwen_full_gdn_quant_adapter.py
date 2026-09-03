"""Reviewed Fleet adapter for exact-ARA full-GDN ModelOpt NVFP4 conversion."""

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

from aeon.scripts import qwen_full_gdn_quant_worker as worker
from aeon.core.fleet_hosts import network_address


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "aeon-qwen38-full-gdn-quant"
PROJECT = "aeon-qwen38-full-gdn-quant"
HOST = worker.HOST
HOSTNAME = worker.HOSTNAME
NETWORK_HOST = network_address(HOST)
REMOTE_PYTHON = str(worker.REMOTE_PYTHON)
LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
REMOTE_RUN_ROOT = worker.SCRATCH_ROOT
CANONICAL_SOURCE = Path(
    "/home/aday/.aeon/models/.speed-sources/"
    "Qwen3.8-27B-heretic-ara-a67ae100d933c0d17af3232bda35825979fc63ce"
)
TEMPLATE_CONFIG = Path(
    "/home/aday/.local/state/aeon-qwen38-quant/"
    "template-mantrah-53097a45/config.json"
)
TEMPLATE_SCALES = Path(
    "/home/aday/.local/state/aeon-qwen38-quant/"
    "template-mantrah-53097a45/calibration-scales.safetensors"
)
MODELOPT_WHEEL = Path(
    "/home/aday/.local/state/aeon-qwen38-quant/sources/"
    "nvidia_modelopt-0.46.0-py3-none-any.whl"
)
SOURCE_FILES = (
    "aeon/__init__.py",
    "aeon/core/__init__.py",
    "aeon/core/fleet_hosts.py",
    "aeon/core/qwen_full_gdn_quant_adapter.py",
    "aeon/scripts/build_qwen38_full_gdn_nvfp4.py",
    "aeon/scripts/qwen_full_gdn_quant_worker.py",
)
_RUNTIME_RE = re.compile(r"^fr-[a-f0-9]{32}$")
_PROCESS_IDENTITY_RE = re.compile(
    r"^aeon-full-gdn-quant:(fr-[a-f0-9]{32}):([a-f0-9]{64}):([0-9]+)$"
)
_SAFE_RELATIVE_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


class QwenFullGDNQuantError(RuntimeError):
    pass


class QwenFullGDNQuantTransportError(QwenFullGDNQuantError):
    """Retryable failure before the reviewed remote protocol answered."""


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
            raise QwenFullGDNQuantError(f"quantization source is unsafe: {relative}")
        result[relative] = _sha256(path)
    return result


def _expected_artifact_identity(sources: dict[str, str]) -> dict[str, str]:
    return {
        "modelopt_wheel": worker.MODELOPT_WHEEL_SHA256,
        "source_manifest": _canonical_sha256(sources),
        "source_tree": _canonical_sha256(
            {
                "weights": worker.SOURCE_WEIGHT_SHA256,
                "metadata": worker.SOURCE_METADATA_SHA256,
            }
        ),
        "template_config": worker.TEMPLATE_CONFIG_SHA256,
        "template_scales": worker.TEMPLATE_SCALES_SHA256,
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
        "cleanup-prelaunch",
    }:
        raise QwenFullGDNQuantError("invalid quantization worker action")
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
        f"{source_root}/aeon/scripts/qwen_full_gdn_quant_worker.py",
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
    if len(result.stdout) > 4 * 1024 * 1024 or len(result.stderr) > 256 * 1024:
        raise QwenFullGDNQuantError("quantization worker response exceeded its bound")
    if result.returncode == 255:
        raise QwenFullGDNQuantTransportError(
            "quantization worker transport is unavailable"
        )
    try:
        value = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenFullGDNQuantTransportError(
            "quantization worker transport returned no valid response"
        ) from exc
    if result.returncode != 0 or not isinstance(value, dict) or value.get("ok") is not True:
        detail = value.get("detail") if isinstance(value, dict) else None
        raise QwenFullGDNQuantError(
            f"quantization worker {action} failed: {detail or 'unknown error'}"
        )
    response = value.get("result")
    if not isinstance(response, dict):
        raise QwenFullGDNQuantError("quantization worker result is malformed")
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
 print(json.dumps({"state":"absent"})); raise SystemExit(0)
assert stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.geteuid() and not metadata.st_mode & 0o077
values = os.statvfs(path); allocated = metadata.st_blocks * 512
for root, directories, files in os.walk(path, topdown=True, followlinks=False):
 for name in [*directories, *files]:
  item = os.path.join(root, name); item_metadata = os.lstat(item)
  assert item_metadata.st_uid == os.geteuid() and not stat.S_ISLNK(item_metadata.st_mode)
  assert stat.S_ISDIR(item_metadata.st_mode) or stat.S_ISREG(item_metadata.st_mode)
  allocated += item_metadata.st_blocks * 512
print(json.dumps({"state":"present","device":str(metadata.st_dev),"free":values.f_bavail*values.f_frsize,"inodes":values.f_favail,"allocated":allocated},sort_keys=True))
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
        timeout=120,
    )
    if result.returncode != 0 or len(result.stdout) > 4096:
        raise QwenFullGDNQuantError("quantization storage metrics are unavailable")
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
        raise QwenFullGDNQuantError("quantization storage metrics are malformed") from exc


class _PreparationHeartbeat:
    def __init__(self, context: RuntimeContext) -> None:
        self.context = context
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_PreparationHeartbeat":
        self.context.heartbeat(None, "Staging exact Qwen3.8 BF16 quantization input")
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        if self.error is not None:
            raise QwenFullGDNQuantError("quantization preparation heartbeat failed") from self.error

    def _run(self) -> None:
        while not self.stop.wait(240):
            try:
                self.context.heartbeat(None, "Exact Qwen3.8 quantization staging is active")
            except BaseException as exc:
                self.error = exc
                return


class AeonQwenFullGDNQuantAdapter:
    """One reviewed shardwise conversion lane on the 48 GB canary worker."""

    def __init__(self) -> None:
        self._prepared: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _payload(payload: Mapping[str, Any]) -> dict[str, str]:
        if not isinstance(payload, Mapping) or set(payload) - {"recipe"}:
            raise QwenFullGDNQuantError("quantization payload has unsupported fields")
        recipe = payload.get("recipe", "full-gdn-max-v1")
        if recipe != "full-gdn-max-v1":
            raise QwenFullGDNQuantError("quantization recipe is not reviewed")
        return {"recipe": recipe}

    @staticmethod
    def _profile_identity(context: RuntimeContext, sources: dict[str, str]) -> None:
        if (
            context.profile.profile_id != PROFILE_ID
            or context.profile.project != PROJECT
            or context.profile.artifact_identity != _expected_artifact_identity(sources)
        ):
            raise QwenFullGDNQuantError("quantization profile identity changed")
        lease = context.lease
        if (
            lease.host != HOST
            or lease.memory_total_mib is None
            or lease.memory_total_mib < 47 * 1024
            or abs(lease.vram_budget_gb - 41.25) > 1e-9
            or lease.exclusive is not True
            or context.scratch_path != lease.run_dir
        ):
            raise QwenFullGDNQuantError("quantization lease differs from its profile")

    @staticmethod
    def _write_private(path: Path, payload: bytes) -> None:
        if path.exists() or path.is_symlink():
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise QwenFullGDNQuantError("local staging path is unsafe")
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
    def _prepare_remote_dirs(scratch: str) -> None:
        script = r"""
import os, pathlib, stat, sys
scratch, expected = sys.argv[1:]
assert os.uname().nodename == expected
root = pathlib.Path(scratch)
for path in (root, root/'source', root/'input', root/'input/source', root/'fixtures'):
 path.mkdir(mode=0o700, parents=True, exist_ok=True); path.chmod(0o700)
 metadata=path.lstat(); assert stat.S_ISDIR(metadata.st_mode) and metadata.st_uid==os.geteuid() and not metadata.st_mode & 0o077
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
                    ]
                ),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise QwenFullGDNQuantError("quantization remote directories failed")

    @staticmethod
    def _rsync(arguments: list[str], destination: str, *, cwd: Path | None, timeout: int) -> None:
        transport = " ".join(shlex.quote(item) for item in _ssh_base()[:-1])
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
                *arguments,
                f"aday@{NETWORK_HOST}:{destination}",
            ],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise QwenFullGDNQuantError("quantization staging transfer failed")

    @classmethod
    def _stage_sources(cls, scratch: str, sources: dict[str, str]) -> None:
        cls._rsync(
            ["-R", "--", *sources],
            f"{scratch}/source/",
            cwd=PACKAGE_ROOT,
            timeout=300,
        )

    @classmethod
    def _stage_input(cls, scratch: str) -> None:
        cls._rsync(
            ["--", f"{CANONICAL_SOURCE}/"],
            f"{scratch}/input/source/",
            cwd=None,
            timeout=1800,
        )

    @classmethod
    def _stage_file(cls, local: Path, remote: str) -> None:
        metadata = local.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise QwenFullGDNQuantError(f"canonical fixture is unsafe: {local}")
        cls._rsync(["--", str(local)], remote, cwd=None, timeout=300)

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        if _RUNTIME_RE.fullmatch(context.runtime_id) is None or context.job_id is None:
            raise QwenFullGDNQuantError("quantization runtime/job identity is malformed")
        self._payload(context.payload)
        sources = _source_manifest()
        self._profile_identity(context, sources)
        scratch = str(context.scratch_path)
        source_root = f"{scratch}/source"
        request_path = f"{scratch}/qwen-full-gdn-quant-request.json"
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
            "source_weight_sha256": worker.SOURCE_WEIGHT_SHA256,
            "source_metadata_sha256": worker.SOURCE_METADATA_SHA256,
            "source_tree_sha256": _canonical_sha256(
                {
                    "weights": worker.SOURCE_WEIGHT_SHA256,
                    "metadata": worker.SOURCE_METADATA_SHA256,
                }
            ),
            "template_config_sha256": worker.TEMPLATE_CONFIG_SHA256,
            "template_scales_sha256": worker.TEMPLATE_SCALES_SHA256,
            "modelopt_wheel_sha256": worker.MODELOPT_WHEEL_SHA256,
        }
        request_bytes = (
            json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        local_request = context.run_dir / "qwen-full-gdn-quant-request.json"
        self._write_private(local_request, request_bytes)

        before_device, _free, _inodes, before_allocated = _remote_metrics(
            scratch, create=True
        )
        with _PreparationHeartbeat(context):
            self._prepare_remote_dirs(scratch)
            self._stage_sources(scratch, sources)
            self._stage_file(local_request, request_path)
            self._stage_file(
                TEMPLATE_CONFIG, f"{scratch}/fixtures/template-config.json"
            )
            self._stage_file(
                TEMPLATE_SCALES, f"{scratch}/fixtures/template-scales.safetensors"
            )
            self._stage_file(
                MODELOPT_WHEEL,
                f"{scratch}/fixtures/nvidia_modelopt-0.46.0-py3-none-any.whl",
            )
            self._stage_input(scratch)
            preflight = _remote_action(
                source_root,
                "preflight",
                request_path,
                request_sha256,
                timeout=1800,
            )
        if (
            preflight.get("source_tree_sha256") != request["source_tree_sha256"]
            or preflight.get("template_config_sha256") != worker.TEMPLATE_CONFIG_SHA256
            or preflight.get("template_scales_sha256") != worker.TEMPLATE_SCALES_SHA256
            or preflight.get("modelopt_wheel_sha256") != worker.MODELOPT_WHEEL_SHA256
            or (preflight.get("environment") or {}).get("modelopt") != "0.46.0"
        ):
            raise QwenFullGDNQuantError("quantization worker preflight identity changed")
        filesystem, free_bytes, free_inodes, allocated = _remote_metrics(
            scratch, create=False
        )
        if filesystem != before_device:
            raise QwenFullGDNQuantError("quantization worker filesystem changed")
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
                "full-GDN quantization preflight receipt is absent", process_absent=True
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
                raise QwenFullGDNQuantError("quantization PID is malformed")
            context.heartbeat(pid, "Exact Qwen3.8 full-GDN quantizer bound to PID")
            identity = (
                f"aeon-full-gdn-quant:{context.runtime_id}:"
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
                    f"full-GDN quantization failed before process creation: {exc}",
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
            raise QwenFullGDNQuantError("quantization runtime identity changed")
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
            f"{scratch}/qwen-full-gdn-quant-request.json",
            digest,
            extra=extra,
            timeout=timeout,
        )

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        try:
            _runtime_id, _digest, pid = self._runtime_identity(runtime)
            status = self._runtime_action(runtime, "status", timeout=90)
        except QwenFullGDNQuantTransportError:
            raise
        except QwenFullGDNQuantError as exc:
            return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))
        state = status.get("state")
        if state == "running":
            if status.get("pid") != pid:
                return ProbeResult(
                    ProbeState.UNKNOWN, False, False, "quantization PID changed"
                )
            return ProbeResult(
                ProbeState.RUNNING,
                True,
                False,
                "Exact Qwen3.8 full-GDN conversion is running",
            )
        if state == "completed":
            return ProbeResult(
                ProbeState.COMPLETED,
                False,
                True,
                "Exact Qwen3.8 full-GDN conversion completed",
            )
        if state == "failed":
            detail = str(
                (status.get("result") or {}).get("failure") or "quantization failed"
            )
            return ProbeResult(ProbeState.FAILED, False, True, detail[:500])
        if state == "absent":
            return ProbeResult(ProbeState.ABSENT, False, True, "quantizer is absent")
        return ProbeResult(
            ProbeState.UNKNOWN, False, False, "quantization state is ambiguous"
        )

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        try:
            result = self._runtime_action(runtime, "stop", timeout=150)
        except QwenFullGDNQuantError as exc:
            return StopResult(False, False, str(exc))
        absent = result.get("process_absent") is True
        return StopResult(absent, True, reason if absent else "quantizer is still stopping")

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
            raise QwenFullGDNQuantError("canonical quantization output is unsafe")
        expected = {"MANIFEST.sha256"}
        total = 0
        for line in manifest.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
            if match is None or _SAFE_RELATIVE_RE.fullmatch(match.group(2)) is None:
                raise QwenFullGDNQuantError("canonical output manifest is malformed")
            candidate = path / match.group(2)
            candidate_meta = candidate.lstat()
            if (
                not stat.S_ISREG(candidate_meta.st_mode)
                or candidate_meta.st_uid != os.geteuid()
                or stat.S_ISLNK(candidate_meta.st_mode)
                or _sha256(candidate) != match.group(1)
            ):
                raise QwenFullGDNQuantError("canonical output digest changed")
            total += candidate_meta.st_size
            if total > 22 * 1024**3:
                raise QwenFullGDNQuantError("canonical output exceeded its bound")
            expected.add(match.group(2))
        actual = {"MANIFEST.sha256"}
        for item in path.rglob("*"):
            item_meta = item.lstat()
            if stat.S_ISDIR(item_meta.st_mode):
                if item_meta.st_uid != os.geteuid() or item_meta.st_mode & 0o022:
                    raise QwenFullGDNQuantError("canonical output directory is unsafe")
                continue
            if not stat.S_ISREG(item_meta.st_mode) or stat.S_ISLNK(item_meta.st_mode):
                raise QwenFullGDNQuantError("canonical output has an unsafe inode")
            actual.add(item.relative_to(path).as_posix())
        if actual != expected:
            raise QwenFullGDNQuantError("canonical output file set changed")
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
            raise QwenFullGDNQuantError("quantization output settlement transfer failed")

    @staticmethod
    def _prelaunch_request(runtime: Mapping[str, Any]) -> tuple[str, str]:
        runtime_id = str(runtime.get("runtime_id") or "")
        scratch = str(runtime.get("run_dir") or "")
        local_request = Path(scratch) / "qwen-full-gdn-quant-request.json"
        if (
            _RUNTIME_RE.fullmatch(runtime_id) is None
            or runtime.get("host") != HOST
            or PurePosixPath(scratch).parent != REMOTE_RUN_ROOT
            or not local_request.is_file()
        ):
            raise QwenFullGDNQuantError("prelaunch quantization identity changed")
        payload = local_request.read_bytes()
        request = json.loads(payload)
        if (
            request.get("runtime_id") != runtime_id
            or request.get("scratch_path") != scratch
            or request.get("host") != HOST
        ):
            raise QwenFullGDNQuantError("prelaunch request binding changed")
        return scratch, hashlib.sha256(payload).hexdigest()

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        if runtime.get("process_identity") is None:
            scratch, digest = self._prelaunch_request(runtime)
            result = _remote_action(
                f"{scratch}/source",
                "cleanup-prelaunch",
                f"{scratch}/qwen-full-gdn-quant-request.json",
                digest,
                timeout=900,
            )
            reclaimed = result.get("reclaimed_bytes")
            if isinstance(reclaimed, bool) or not isinstance(reclaimed, int):
                raise QwenFullGDNQuantError("prelaunch cleanup receipt is malformed")
            return StorageFinalizationResult(
                True,
                True,
                reclaimed,
                "full-GDN prelaunch scratch cleaned; no process was created",
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
                    True, True, 0, "quantization output settled and scratch absent"
                )
            raise QwenFullGDNQuantError("quantization scratch vanished before settlement")
        status = self._runtime_action(runtime, "settle-status", timeout=1800)
        manifest_sha = str(status.get("manifest_sha256") or "")
        if re.fullmatch(r"[a-f0-9]{64}", manifest_sha) is None:
            raise QwenFullGDNQuantError("quantization manifest digest is malformed")
        if not valid:
            self._copy_output(f"{scratch}/output", canonical)
            valid, local_manifest = self._local_output_valid(canonical)
        if not valid or local_manifest != manifest_sha:
            raise QwenFullGDNQuantError("canonical output differs from worker manifest")
        self._runtime_action(runtime, "mark-settled", extra=manifest_sha, timeout=1800)
        cleaned = self._runtime_action(
            runtime, "cleanup", extra=manifest_sha, timeout=1800
        )
        reclaimed = cleaned.get("reclaimed_bytes")
        if isinstance(reclaimed, bool) or not isinstance(reclaimed, int) or reclaimed < 0:
            raise QwenFullGDNQuantError("quantization cleanup receipt is malformed")
        return StorageFinalizationResult(
            True, True, reclaimed, "full-GDN NVFP4 output settled durably on .177"
        )


def create_fleet_adapter() -> AeonQwenFullGDNQuantAdapter:
    return AeonQwenFullGDNQuantAdapter()
