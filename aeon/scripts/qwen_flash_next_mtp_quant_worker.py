#!/usr/bin/env python3
"""Exact local lifecycle for the MTP-only NVFP4 sibling conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from aeon.scripts import quantize_qwen38_flash_next_mtp_nvfp4 as converter


SCHEMA = "aeon-qwen38-flash-next-mtp-nvfp4-worker-v1"
HOST = "192.168.0.177"
HOSTNAME = "DAY2RTX6000PRO"
VRAM_CAP_GIB = 24.0
RESERVE_GIB = 6.0
_SHA = re.compile(r"^[a-f0-9]{64}$")
_RUNTIME = re.compile(r"^fr-[a-f0-9]{32}$")
_PID: int | None = None


class WorkerError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private(path: Path, maximum: int = 1024 * 1024) -> Mapping[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= maximum
    ):
        raise WorkerError("worker artifact is unsafe")
    value = json.loads(path.read_bytes())
    if not isinstance(value, Mapping):
        raise WorkerError("worker artifact is malformed")
    return value


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise WorkerError("worker artifact write failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def load_request(path: Path, digest: str) -> Mapping[str, Any]:
    if _SHA.fullmatch(digest) is None or _sha256(path) != digest:
        raise WorkerError("request identity changed")
    request = _private(path)
    required = {
        "schema_version", "runtime_id", "job_id", "host", "hostname",
        "claim_id", "gpu_uuid", "physical_gpu", "vram_cap_gib", "exclusive",
        "source_path", "source_manifest_sha256", "source_revision",
        "destination_path", "modelopt_wheel", "modelopt_wheel_sha256",
        "source_files",
    }
    if (
        set(request) != required
        or request.get("schema_version") != SCHEMA
        or _RUNTIME.fullmatch(str(request.get("runtime_id") or "")) is None
        or request.get("host") != HOST
        or request.get("hostname") != HOSTNAME
        or request.get("physical_gpu") != 0
        or request.get("vram_cap_gib") != VRAM_CAP_GIB
        or request.get("exclusive") is not True
        or request.get("modelopt_wheel_sha256") != converter.base.MODELOPT_WHEEL_SHA256
    ):
        raise WorkerError("request contract changed")
    return request


def _verify_staged_sources(request: Mapping[str, Any], request_path: Path) -> None:
    source_root = request_path.parent / "source"
    expected = request.get("source_files")
    if not isinstance(expected, Mapping) or not expected:
        raise WorkerError("staged source closure is malformed")
    actual: set[str] = set()
    for item in source_root.rglob("*"):
        relative = item.relative_to(source_root).as_posix()
        metadata = item.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
                raise WorkerError("staged source directory is unsafe")
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            raise WorkerError("staged source file is unsafe")
        actual.add(relative)
    if actual != set(expected):
        raise WorkerError("staged source closure changed")
    for relative, digest in expected.items():
        if not isinstance(relative, str) or _SHA.fullmatch(str(digest)) is None:
            raise WorkerError("staged source receipt is malformed")
        if _sha256(source_root / relative) != digest:
            raise WorkerError("staged source identity changed")


def paths(request: Mapping[str, Any], request_path: Path) -> Mapping[str, Path]:
    root = request_path.parent
    return {
        "root": root,
        "status": root / "mtp-quant-status.json",
        "stdout": root / "mtp-quant.stdout",
        "stderr": root / "mtp-quant.stderr",
        "pid": root / "mtp-quant.pid",
    }


def _matches(pid: int, request: Path, digest: str) -> bool:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    expected = [
        os.fsencode(str(Path(__file__).resolve())), b"supervise",
        os.fsencode(str(request)), digest.encode(),
    ]
    return all(item in raw for item in expected)


def action_preflight(request_path: Path, digest: str) -> Mapping[str, Any]:
    request = load_request(request_path, digest)
    _verify_staged_sources(request, request_path)
    source = Path(str(request["source_path"]))
    converter._source_closure(source, str(request["source_manifest_sha256"]))
    wheel = Path(str(request["modelopt_wheel"]))
    if converter._sha256(wheel) != request["modelopt_wheel_sha256"]:
        raise WorkerError("ModelOpt wheel identity changed")
    return {
        "request_sha256": digest,
        "source_manifest_sha256": request["source_manifest_sha256"],
        "source_revision": request["source_revision"],
        "modelopt_wheel_sha256": request["modelopt_wheel_sha256"],
    }


def _supervise(request_path: Path, digest: str) -> int:
    global _PID
    request = load_request(request_path, digest)
    evidence = paths(request, request_path)
    _PID = os.getpid()
    def stop(_signum: int, _frame: Any) -> None:
        raise WorkerError("conversion interrupted by Fleet")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    _atomic(evidence["status"], {
        "schema_version": SCHEMA, "state": "running", "pid": _PID,
        "runtime_id": request["runtime_id"], "request_sha256": digest,
    })
    try:
        receipt = converter.convert(
            Path(str(request["source_path"])), Path(str(request["destination_path"])),
            source_manifest_sha256=str(request["source_manifest_sha256"]),
            source_revision=str(request["source_revision"]),
            modelopt_wheel=Path(str(request["modelopt_wheel"])),
        )
        _atomic(evidence["status"], {
            "schema_version": SCHEMA, "state": "completed", "pid": _PID,
            "runtime_id": request["runtime_id"], "request_sha256": digest,
            "destination_path": request["destination_path"],
            "sha256sums_sha256": receipt["sha256sums_sha256"],
        })
        return 0
    except BaseException as exc:
        _atomic(evidence["status"], {
            "schema_version": SCHEMA, "state": "failed", "pid": _PID,
            "runtime_id": request["runtime_id"], "request_sha256": digest,
            "failure": f"{type(exc).__name__}: {str(exc)[:400]}",
        })
        return 1


def action_spawn(request_path: Path, digest: str) -> Mapping[str, Any]:
    request = load_request(request_path, digest)
    action_preflight(request_path, digest)
    evidence = paths(request, request_path)
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "supervise", str(request_path), digest],
        stdin=subprocess.DEVNULL,
        stdout=evidence["stdout"].open("xb"),
        stderr=evidence["stderr"].open("xb"),
        start_new_session=True,
        close_fds=True,
        env={
            "HOME": "/home/aday", "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C", "LC_ALL": "C", "PYTHONPATH": str(request_path.parent / "source"),
            "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": str(request["gpu_uuid"]),
            "GPU_AGENT_CLAIM_ID": str(request["claim_id"]),
            "GPU_MEM_LIMIT_GB": str(VRAM_CAP_GIB), "GPU_RESERVE_GB": str(RESERVE_GIB),
            "AEON_QUANT_RUNTIME_ID": str(request["runtime_id"]),
        },
    )
    evidence["pid"].write_text(f"{process.pid}\n", encoding="ascii")
    evidence["pid"].chmod(0o600)
    return {"pid": process.pid}


def action_status(request_path: Path, digest: str) -> Mapping[str, Any]:
    request = load_request(request_path, digest)
    status_path = paths(request, request_path)["status"]
    if not status_path.exists():
        return {"state": "absent", "pid": None}
    status = dict(_private(status_path))
    pid = status.get("pid")
    if status.get("state") == "running" and (
        type(pid) is not int or not _matches(pid, request_path, digest)
    ):
        return {"state": "failed", "pid": None, "failure": "supervisor identity vanished"}
    return status


def action_stop(request_path: Path, digest: str) -> Mapping[str, Any]:
    status = action_status(request_path, digest)
    pid = status.get("pid")
    if status.get("state") == "running" and type(pid) is int:
        if not _matches(pid, request_path, digest):
            raise WorkerError("refusing to signal changed process identity")
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        if Path(f"/proc/{pid}").exists():
            return {"process_absent": False}
    return {"process_absent": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "spawn", "status", "stop", "supervise"))
    parser.add_argument("request", type=Path)
    parser.add_argument("digest")
    arguments = parser.parse_args(argv)
    if arguments.action == "supervise":
        return _supervise(arguments.request, arguments.digest)
    try:
        result = {
            "preflight": action_preflight,
            "spawn": action_spawn,
            "status": action_status,
            "stop": action_stop,
        }[arguments.action](arguments.request, arguments.digest)
        print(json.dumps({"ok": True, "result": result}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "detail": f"{type(exc).__name__}: {str(exc)[:400]}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
