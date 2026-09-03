"""Reviewed Fleet adapter for Aeon's owner-labeled ComfyUI service."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any, Mapping

import requests

from fleet_compute.adapters import AdapterLaunchError, RuntimeContext
from fleet_compute.models import (
    LaunchResult,
    ProbeResult,
    ProbeState,
    StopResult,
    StorageFinalizationResult,
    StoragePreparationResult,
)


LOCAL_HOST = "192.168.0.177"
EXPECTED_HOSTNAME = "DAY2RTX6000PRO"
CONTAINER = "aeon_comfyui"
COMPONENT = "comfyui"
IMAGE = "aeon_comfyui:latest"
EXPECTED_IMAGE_ID = "sha256:e87d7bcd4da3b5826e03740585ee22a5c78bf5f4468e881495375798f677ba8d"
EXPECTED_START_SHA256 = "b55df925c5e98c8e1118c68966f91542c2c82cd670011ba239a570fba2410d85"
EXPECTED_CAP_SHA256 = "d42d48c48b5bf8329b3d3b63f5a815c3258bf4d9ee9e6e3002c38340543fabf8"
ENDPOINT = "http://127.0.0.1:8188"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
START_SCRIPT = PACKAGE_ROOT / "aeon" / "scripts" / "start_comfyui.sh"
SITE_CUSTOMIZE = PACKAGE_ROOT / "aeon" / "scripts" / "comfyui_sitecustomize.py"
AEON_HOME = Path(os.environ.get("AEON_HOME", "/home/aday/.aeon")).resolve()
_RUNTIME_ID = re.compile(r"^fr-[0-9a-f]{32}$")


class ComfyFleetError(RuntimeError):
    pass


def _run(command: list[str], *, timeout: float = 60, check: bool = True):
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode:
        raise ComfyFleetError((result.stderr or result.stdout or "command failed").strip())
    return result


def _inspect(format_value: str):
    return _run(
        ["docker", "container", "inspect", "--format", format_value, CONTAINER],
        timeout=20,
        check=False,
    )


def _container_receipt() -> dict[str, Any] | None:
    result = _inspect(
        '{{json .Id}} {{json .Config.Labels}} {{json .State.Pid}} '
        '{{json .State.Running}} {{json .Image}}'
    )
    if result.returncode != 0:
        return None
    try:
        decoder = json.JSONDecoder()
        values: list[Any] = []
        source = result.stdout.strip()
        while source:
            value, end = decoder.raw_decode(source)
            values.append(value)
            source = source[end:].lstrip()
        container_id, labels, pid, running, image_id = values
        return {
            "container_id": str(container_id),
            "labels": labels if isinstance(labels, dict) else {},
            "pid": int(pid),
            "running": running is True,
            "image_id": str(image_id),
        }
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ComfyFleetError("ComfyUI container receipt is malformed") from exc


def _healthy() -> bool:
    try:
        response = requests.get(
            f"{ENDPOINT}/system_stats",
            timeout=(2, 10),
            allow_redirects=False,
            proxies={"http": "", "https": ""},
        )
        return response.status_code == 200 and len(response.content) <= 2 * 1024 * 1024
    except (OSError, requests.RequestException):
        return False


def _owned(receipt: Mapping[str, Any], claim_id: str) -> bool:
    labels = receipt.get("labels") or {}
    return (
        isinstance(labels, dict)
        and labels.get("com.bc_aeon.component") == COMPONENT
        and labels.get("com.bc_aeon.claim") == claim_id
        and receipt.get("image_id") == EXPECTED_IMAGE_ID
    )


class AeonComfyFleetAdapter:
    """Own only the exact container started for the broker's current claim."""

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        if context.lease.host != LOCAL_HOST or context.lease.physical_gpu != 0:
            raise ComfyFleetError("Aeon ComfyUI is authorized only on .177 physical GPU 0")
        if _RUNTIME_ID.fullmatch(context.runtime_id) is None:
            raise ComfyFleetError("Fleet runtime identity is malformed")
        if context.scratch_path is not None:
            raise ComfyFleetError("canonical .177 ComfyUI must not use worker scratch")
        if os.uname().nodename != EXPECTED_HOSTNAME:
            raise ComfyFleetError("ComfyUI adapter is not on the orchestrator")
        if context.profile.artifact_identity.get("image") != EXPECTED_IMAGE_ID.removeprefix("sha256:"):
            raise ComfyFleetError("ComfyUI profile image identity changed")
        if (
            context.lease.vram_budget_gb != context.profile.vram_budget_gb
            or not context.lease.exclusive
            or context.lease.memory_total_mib is None
            or context.lease.memory_total_mib < 90 * 1024
        ):
            raise ComfyFleetError("ComfyUI lease capability differs from its profile")
        for path in (START_SCRIPT, SITE_CUSTOMIZE, AEON_HOME / "models" / "comfyui"):
            if not path.exists():
                raise ComfyFleetError(f"required ComfyUI artifact is unavailable: {path.name}")
        for key, path, expected in (
            ("launcher", START_SCRIPT, EXPECTED_START_SHA256),
            ("allocator_cap", SITE_CUSTOMIZE, EXPECTED_CAP_SHA256),
        ):
            with path.open("rb") as handle:
                observed = hashlib.file_digest(handle, "sha256").hexdigest()
            if observed != expected or context.profile.artifact_identity.get(key) != expected:
                raise ComfyFleetError(f"ComfyUI {key} identity changed")
        image_id = _run(["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE], timeout=30).stdout.strip()
        if image_id != EXPECTED_IMAGE_ID:
            raise ComfyFleetError("installed ComfyUI image identity changed")
        metadata = context.run_dir.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise ComfyFleetError("Fleet run directory is not private and owned")
        values = os.statvfs(AEON_HOME)
        return StoragePreparationResult(
            scratch_path=None,
            filesystem_id=str(os.lstat(AEON_HOME).st_dev),
            free_bytes_after_stage=values.f_bavail * values.f_frsize,
            free_inodes_after_stage=values.f_favail,
            staged_bytes=0,
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        if _container_receipt() is not None:
            raise AdapterLaunchError(
                "an existing ComfyUI container is not owned by this Fleet attempt",
                process_absent=True,
            )
        environment = os.environ.copy()
        environment.update(context.lease.required_environment)
        environment.update(AEON_HOME=str(AEON_HOME), GPU_RESERVE_GB="6")
        context.heartbeat(None, "Aeon ComfyUI exact image and storage verified")
        result = subprocess.run(
            ["/home/aday/bin/fleet-low-priority", "bash", str(START_SCRIPT)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=environment,
        )
        if result.returncode:
            receipt = _container_receipt()
            if receipt is None:
                raise AdapterLaunchError(
                    "ComfyUI launch failed before container creation",
                    process_absent=True,
                )
            raise ComfyFleetError("ComfyUI launch failed after container creation")
        deadline = time.monotonic() + context.profile.startup_timeout_seconds
        receipt = _container_receipt()
        while time.monotonic() < deadline:
            receipt = _container_receipt()
            if receipt is None:
                raise AdapterLaunchError("ComfyUI container disappeared during startup", process_absent=True)
            if not _owned(receipt, context.lease.claim_id):
                raise ComfyFleetError("ComfyUI container identity changed during startup")
            if receipt["running"] and receipt["pid"] > 1:
                context.heartbeat(receipt["pid"], "Aeon ComfyUI startup bound to exact container PID")
                if _healthy():
                    return LaunchResult(
                        pid=receipt["pid"],
                        process_identity=f"{receipt['container_id']}:{context.lease.claim_id}",
                        endpoint=ENDPOINT,
                    )
            time.sleep(2)
        raise ComfyFleetError("ComfyUI did not become healthy before its startup deadline")

    @staticmethod
    def _matches(runtime: Mapping[str, Any], receipt: Mapping[str, Any]) -> bool:
        return (
            _owned(receipt, str(runtime.get("claim_id") or ""))
            and runtime.get("process_identity")
            == f"{receipt.get('container_id')}:{runtime.get('claim_id')}"
            and runtime.get("pid") == receipt.get("pid")
        )

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        receipt = _container_receipt()
        if receipt is None:
            return ProbeResult(ProbeState.ABSENT, False, True, "Aeon ComfyUI is absent")
        if not self._matches(runtime, receipt):
            return ProbeResult(ProbeState.UNKNOWN, False, False, "Aeon ComfyUI identity changed")
        if not receipt["running"]:
            return ProbeResult(ProbeState.ABSENT, False, True, "Aeon ComfyUI exited")
        if not _healthy():
            return ProbeResult(ProbeState.STARTING, True, False, "Aeon ComfyUI health is pending")
        return ProbeResult(ProbeState.READY, True, False, "Aeon ComfyUI is healthy")

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        receipt = _container_receipt()
        if receipt is None:
            return StopResult(True, True, "Aeon ComfyUI is already absent")
        if not self._matches(runtime, receipt):
            return StopResult(False, False, "Aeon ComfyUI identity changed")
        result = _run(["docker", "container", "stop", "--time", "30", CONTAINER], timeout=45, check=False)
        if result.returncode:
            raise ComfyFleetError("exact ComfyUI container did not stop cleanly")
        # Remove only the now-stopped, still identity-matched owner container.
        stopped = _container_receipt()
        if stopped is not None:
            if not self._matches(runtime, stopped):
                return StopResult(False, False, "stopped ComfyUI identity changed")
            result = _run(["docker", "container", "rm", CONTAINER], timeout=30, check=False)
            if result.returncode:
                raise ComfyFleetError("exact stopped ComfyUI container was not removed")
        return StopResult(_container_receipt() is None, True, reason)

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        if runtime.get("host") != LOCAL_HOST or storage.get("scratch_path") is not None:
            raise ComfyFleetError("ComfyUI storage manifest changed")
        return StorageFinalizationResult(True, True, 0, "canonical ComfyUI output retained on .177")


def create_fleet_adapter() -> AeonComfyFleetAdapter:
    return AeonComfyFleetAdapter()
