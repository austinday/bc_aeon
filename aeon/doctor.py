"""Read-only installation and fleet-readiness checks for ``aeon doctor``."""

from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from aeon.core.fleet_backend import FleetBackendError, select_compute_backend
from aeon.core.qwen_capabilities import (
    QwenCapabilityError,
    STANDARD_IMAGE_ID,
    enabled_qwen_runtime_capabilities,
)
from aeon.core.workspace_instructions import (
    WorkspaceInstructionError,
    discover_workspace_instructions,
)


COORDINATOR = Path("/home/aday/website_hosting/gpu_coord.py")
LOW_PRIORITY_WRAPPER = Path("/home/aday/bin/fleet-low-priority")
MODEL_DIRECTORY_NAME = "Qwen3.8-27B-ARA-abliterated-NVFP4-MTP"


def _check(name: str, ok: bool, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "required": required, "detail": detail}


def _regular_owned(path: Path, *, executable: bool = False) -> bool:
    try:
        metadata = path.stat()
    except OSError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and (not executable or bool(metadata.st_mode & stat.S_IXUSR))
    )


def collect_diagnostics() -> dict[str, Any]:
    workspace = Path.cwd().resolve()
    aeon_home = Path(os.environ.get("AEON_HOME", str(Path.home() / ".aeon"))).expanduser()
    model_dir = aeon_home / "models" / MODEL_DIRECTORY_NAME
    checks: list[dict[str, Any]] = []

    checks.append(
        _check(
            "python",
            sys.version_info >= (3, 10),
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )
    checks.append(_check("workspace", workspace.is_dir(), str(workspace)))
    checks.append(
        _check(
            "orchestrator",
            socket.gethostname() == "DAY2RTX6000PRO",
            socket.gethostname(),
        )
    )
    checks.append(
        _check("gpu coordinator", _regular_owned(COORDINATOR), str(COORDINATOR))
    )
    checks.append(
        _check(
            "low-priority wrapper",
            _regular_owned(LOW_PRIORITY_WRAPPER, executable=True),
            str(LOW_PRIORITY_WRAPPER),
        )
    )

    required_model_files = (
        "config.json",
        "model.safetensors.index.json",
        "BUILD_MANIFEST.json",
        "SHA256SUMS",
    )
    missing = [name for name in required_model_files if not (model_dir / name).is_file()]
    manifest_ok = False
    if not missing:
        try:
            manifest = json.loads((model_dir / "BUILD_MANIFEST.json").read_text(encoding="utf-8"))
            manifest_ok = manifest.get("complete") is True and manifest.get("status") == "validated"
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    checks.append(
        _check(
            "Qwen3.8 artifact",
            not missing and manifest_ok,
            str(model_dir) if not missing else f"missing: {', '.join(missing)}",
        )
    )

    image_id = ""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "aeon_vllm:latest", "--format", "{{.Id}}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            image_id = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    checks.append(
        _check(
            "Qwen runtime image",
            image_id == STANDARD_IMAGE_ID,
            image_id or "aeon_vllm:latest is unavailable",
        )
    )

    try:
        capabilities, manifest_sha256 = enabled_qwen_runtime_capabilities()
        detail = ", ".join(
            f"{item.host}:{item.context_tokens // 1024}k/{item.runtime_adapter}"
            for item in capabilities
        )
        checks.append(
            _check(
                "Qwen fleet releases",
                bool(capabilities),
                f"{detail} (manifest {manifest_sha256[:12]})",
            )
        )
    except QwenCapabilityError as exc:
        checks.append(_check("Qwen fleet releases", False, str(exc)))

    try:
        backend, reason = select_compute_backend()
        checks.append(_check("compute backend", True, f"{backend}: {reason}"))
    except FleetBackendError as exc:
        backend = "unavailable"
        checks.append(_check("compute backend", False, str(exc)))

    try:
        instructions = discover_workspace_instructions(workspace)
        detail = ", ".join(str(item.path) for item in instructions) or "none"
        checks.append(_check("workspace instructions", True, detail, required=False))
    except WorkspaceInstructionError as exc:
        checks.append(_check("workspace instructions", False, str(exc)))

    return {
        "ok": all(item["ok"] for item in checks if item["required"]),
        "workspace": str(workspace),
        "backend": backend,
        "primary_model": "Qwen3.8-27B-ARA-NVFP4-MTP",
        "checks": checks,
    }


def run_doctor(*, as_json: bool = False) -> int:
    report = collect_diagnostics()
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Aeon doctor — workspace {report['workspace']}")
        for item in report["checks"]:
            marker = "OK" if item["ok"] else ("WARN" if not item["required"] else "FAIL")
            print(f"[{marker:4}] {item['name']}: {item['detail']}")
        print("Ready." if report["ok"] else "Not ready; resolve the failed required checks.")
    return 0 if report["ok"] else 1
