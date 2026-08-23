"""Discover bounded ``AGENTS.md`` instructions for standalone Aeon runs.

Aeon is normally launched from the project it should work on.  The instruction
chain therefore follows the current workspace, not Aeon's install directory:

* the operator-wide ``~/AGENTS.md`` (when present), then
* every ``AGENTS.md`` from the workspace boundary down to the current directory.

Later, more-specific files are rendered last and therefore take precedence.
Symlinks, non-regular files, invalid UTF-8, and unbounded instruction sets fail
closed instead of being silently ignored.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path


INSTRUCTION_FILENAME = "AGENTS.md"
MAX_INSTRUCTION_FILE_BYTES = 128 * 1024
MAX_INSTRUCTION_CHAIN_BYTES = 256 * 1024


class WorkspaceInstructionError(RuntimeError):
    """An applicable workspace instruction file could not be trusted."""


@dataclass(frozen=True)
class WorkspaceInstructionDocument:
    path: Path
    content: str


def _read_regular_utf8(path: Path) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise WorkspaceInstructionError(
            f"Applicable instruction file is unavailable: {path}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise WorkspaceInstructionError(
                f"Applicable instruction source is not a regular file: {path}"
            )
        if metadata.st_size > MAX_INSTRUCTION_FILE_BYTES:
            raise WorkspaceInstructionError(
                f"Applicable instruction file exceeds {MAX_INSTRUCTION_FILE_BYTES} bytes: {path}"
            )
        chunks: list[bytes] = []
        remaining = MAX_INSTRUCTION_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_INSTRUCTION_FILE_BYTES:
            raise WorkspaceInstructionError(
                f"Applicable instruction file exceeds {MAX_INSTRUCTION_FILE_BYTES} bytes: {path}"
            )
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceInstructionError(
                f"Applicable instruction file is not valid UTF-8: {path}"
            ) from exc
    finally:
        os.close(descriptor)


def _workspace_boundary(workspace: Path, home: Path) -> Path:
    """Choose a finite discovery boundary without treating Aeon's source as cwd."""

    if workspace.is_relative_to(home):
        return home
    for candidate in (workspace, *workspace.parents):
        if (candidate / ".git").exists():
            return candidate
    return workspace


def discover_workspace_instructions(
    workspace: str | Path | None = None,
    *,
    home: str | Path | None = None,
) -> tuple[WorkspaceInstructionDocument, ...]:
    """Return the applicable global-to-local instruction chain."""

    try:
        current = Path(workspace or os.getcwd()).expanduser().resolve(strict=True)
        home_path = Path(home or Path.home()).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WorkspaceInstructionError("Workspace instruction boundary is unavailable") from exc
    if not current.is_dir() or not home_path.is_dir():
        raise WorkspaceInstructionError("Workspace instruction boundary is not a directory")

    boundary = _workspace_boundary(current, home_path)
    candidates: list[Path] = []
    global_path = home_path / INSTRUCTION_FILENAME
    if global_path.exists() or global_path.is_symlink():
        candidates.append(global_path)

    relative = current.relative_to(boundary)
    cursor = boundary
    if cursor != home_path:
        candidate = cursor / INSTRUCTION_FILENAME
        if candidate.exists() or candidate.is_symlink():
            candidates.append(candidate)
    for part in relative.parts:
        cursor = cursor / part
        candidate = cursor / INSTRUCTION_FILENAME
        if candidate.exists() or candidate.is_symlink():
            candidates.append(candidate)

    documents: list[WorkspaceInstructionDocument] = []
    seen: set[Path] = set()
    total = 0
    for candidate in candidates:
        absolute = Path(os.path.abspath(candidate))
        if absolute in seen:
            continue
        seen.add(absolute)
        content = _read_regular_utf8(absolute)
        total += len(content.encode("utf-8"))
        if total > MAX_INSTRUCTION_CHAIN_BYTES:
            raise WorkspaceInstructionError(
                f"Applicable instruction chain exceeds {MAX_INSTRUCTION_CHAIN_BYTES} bytes"
            )
        documents.append(WorkspaceInstructionDocument(path=absolute, content=content))
    return tuple(documents)


def format_workspace_instructions(
    documents: tuple[WorkspaceInstructionDocument, ...],
) -> str:
    """Render a clearly sourced, global-to-local instruction layer."""

    if not documents:
        return ""
    sections = [
        "**APPLICABLE WORKSPACE INSTRUCTIONS**\n"
        "The files below are ordered from broadest to most specific. Later files "
        "override earlier files when they conflict."
    ]
    for document in documents:
        sections.append(
            f"--- BEGIN {document.path} ---\n"
            f"{document.content.rstrip()}\n"
            f"--- END {document.path} ---"
        )
    return "\n\n" + "\n\n".join(sections)


def load_workspace_instruction_section(
    workspace: str | Path | None = None,
) -> str:
    return format_workspace_instructions(discover_workspace_instructions(workspace))
