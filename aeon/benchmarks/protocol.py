"""Content-bound identities for every benchmark execution layer.

Version labels alone are not provenance.  These digests bind the bytes that
queue and score cases, launch either harness, and execute the reviewed Aeon
tools.  They intentionally cover source rather than importing it, keeping this
module side-effect free and making a queued run fail closed after source drift.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Iterable, Sequence


EXECUTOR_PROTOCOL_VERSION = "8"
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_EXECUTOR_SOURCES = (
    _PACKAGE_ROOT / "benchmarks" / "executor.py",
    _PACKAGE_ROOT / "benchmarks" / "catalog.py",
    _PACKAGE_ROOT / "core" / "benchmark_model_telemetry.py",
    _PACKAGE_ROOT / "core" / "benchmark_receipt.py",
    _PACKAGE_ROOT / "core" / "benchmark_simulator.py",
)
_RUNNER_SOURCES = (
    _PACKAGE_ROOT / "benchmarks" / "runner.py",
    _PACKAGE_ROOT / "benchmarks" / "service.py",
    _PACKAGE_ROOT / "benchmarks" / "worker.py",
)
_HARNESS_SOURCES = (
    _PACKAGE_ROOT / "cli.py",
    _PACKAGE_ROOT / "main.py",
    _PACKAGE_ROOT / "core" / "benchmark_model_telemetry.py",
    _PACKAGE_ROOT / "core" / "llm.py",
    _PACKAGE_ROOT / "core" / "prompt_enhancer.py",
    *sorted((_PACKAGE_ROOT / "harnesses").glob("*.py")),
)
_TOOL_SOURCES = (
    _PACKAGE_ROOT / "core" / "action_schema.py",
    _PACKAGE_ROOT / "core" / "agent_protocol.py",
    _PACKAGE_ROOT / "core" / "benchmark_receipt.py",
    _PACKAGE_ROOT / "core" / "benchmark_simulator.py",
    _PACKAGE_ROOT / "core" / "fleet_backend.py",
    _PACKAGE_ROOT / "core" / "tool_resources.py",
    _PACKAGE_ROOT / "core" / "worker.py",
    _PACKAGE_ROOT / "services" / "browser" / "browser_util.py",
    _PACKAGE_ROOT / "services" / "browser" / "human_motion.py",
    _PACKAGE_ROOT / "services" / "browser" / "media_safety.py",
    _PACKAGE_ROOT / "services" / "browser" / "server.py",
    *sorted((_PACKAGE_ROOT / "tools").rglob("*.py")),
)


def _source_sha256(paths: Iterable[Path], *, scope: str) -> str:
    """Hash an ordered set of exact regular source files and their labels."""

    digest = hashlib.sha256()
    digest.update(scope.encode("ascii") + b"\0")
    seen: set[Path] = set()
    for index, path in enumerate(paths):
        candidate = Path(path)
        if candidate in seen:
            raise RuntimeError("benchmark protocol source is duplicated")
        seen.add(candidate)
        metadata = candidate.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("benchmark protocol source is not a regular file")
        body = candidate.read_bytes()
        label = (
            os.fspath(candidate.relative_to(_PACKAGE_ROOT))
            if _PACKAGE_ROOT in candidate.parents
            else f"fixture-{index}:{candidate.name}"
        ).encode("utf-8")
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


def executor_source_sha256(paths: Sequence[Path] | None = None) -> str:
    """Hash exact prompt, scoring, receipt, and fixture executor bytes."""

    return _source_sha256(
        _EXECUTOR_SOURCES if paths is None else paths,
        scope="executor",
    )


def runner_source_sha256(paths: Sequence[Path] | None = None) -> str:
    """Hash exact queue, deadline, evidence, and worker runner bytes."""

    return _source_sha256(
        _RUNNER_SOURCES if paths is None else paths,
        scope="runner",
    )


def harness_source_sha256(paths: Sequence[Path] | None = None) -> str:
    """Hash both reviewed harness launch/execution implementations."""

    return _source_sha256(
        _HARNESS_SOURCES if paths is None else paths,
        scope="harness",
    )


def tool_source_sha256(paths: Sequence[Path] | None = None) -> str:
    """Hash the complete reviewed MCP/tool profile and browser implementation."""

    return _source_sha256(
        _TOOL_SOURCES if paths is None else paths,
        scope="tools",
    )


EXECUTOR_PROTOCOL_SHA256 = executor_source_sha256()
RUNNER_SOURCE_SHA256 = runner_source_sha256()
HARNESS_SOURCE_SHA256 = harness_source_sha256()
TOOL_SOURCE_SHA256 = tool_source_sha256()


__all__ = (
    "EXECUTOR_PROTOCOL_SHA256",
    "EXECUTOR_PROTOCOL_VERSION",
    "HARNESS_SOURCE_SHA256",
    "RUNNER_SOURCE_SHA256",
    "TOOL_SOURCE_SHA256",
    "executor_source_sha256",
    "harness_source_sha256",
    "runner_source_sha256",
    "tool_source_sha256",
)
