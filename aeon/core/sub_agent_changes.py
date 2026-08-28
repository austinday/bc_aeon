"""Bounded, reviewable change receipts for isolated mutable sub-agents.

Mutable children work in detached Git worktrees.  They never write into the
principal's tree directly.  On a terminal turn the wrapper snapshots the exact
repository delta into an owner-private patch, and the principal may later apply
that immutable receipt through the dedicated integration tool.

This module deliberately contains no Worker/tool imports so the wrapper can use
it before loading the tool catalog.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from aeon.core import runtime_signals as rt


MUTABLE_WORKSPACE_RECEIPT = "mutable-workspace.json"
MUTABLE_CHANGE_RECEIPT = "mutable-changes.json"
MUTABLE_PATCH_FILE = "mutable-changes.patch"
MUTABLE_INTEGRATION_RECEIPT = "mutable-integration.json"
SUB_AGENT_REPORT_COLLECTION_RECEIPT = "report-collected.json"
SUB_AGENT_REPORT_PROGRESS_RECEIPT = "report-read-progress.json"

MAX_RECEIPT_BYTES = 128 * 1024
MAX_PATCH_BYTES = 32 * 1024 * 1024
MAX_CHANGED_PATHS = 4096
_COMMIT_RE = re.compile(r"\A[0-9a-f]{40,64}\Z")


class SubAgentChangeError(RuntimeError):
    """A mutable-child change receipt cannot be produced or trusted."""


def read_owned_json(path: Path, *, max_bytes: int = MAX_RECEIPT_BYTES) -> dict:
    """Read one small owner-private, non-linked JSON object."""

    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise SubAgentChangeError(f"required receipt is unavailable: {candidate.name}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o022
        or metadata.st_size < 2
        or metadata.st_size > max_bytes
    ):
        raise SubAgentChangeError(f"receipt failed owner/private file validation: {candidate.name}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SubAgentChangeError(f"receipt is not valid JSON: {candidate.name}") from exc
    if not isinstance(value, dict):
        raise SubAgentChangeError(f"receipt is not a JSON object: {candidate.name}")
    return value


def validate_relative_path(value: object) -> str:
    """Return one safe repository-relative POSIX path, or raise."""

    text = str(value or "")
    path = PurePosixPath(text)
    if (
        not text
        or len(text) > 2000
        or text != path.as_posix()
        or path.is_absolute()
        or ".." in path.parts
        or any(character in text for character in ("\x00", "\r", "\n"))
    ):
        raise SubAgentChangeError("change receipt contains an unsafe repository path")
    return text


def _git(
    workspace: Path,
    *arguments: str,
    timeout: int = 30,
    stdout=None,
    index_file: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run fixed, hook-free Git inside the already bounded child scope."""

    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("GIT_"):
            environment.pop(key, None)
    if index_file is not None:
        environment["GIT_INDEX_FILE"] = str(index_file)
    try:
        return subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(workspace),
                *arguments,
            ],
            stdout=stdout if stdout is not None else subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise SubAgentChangeError("Git timed out while snapshotting mutable work") from exc


def _git_text(
    workspace: Path,
    *arguments: str,
    timeout: int = 30,
    index_file: Path | None = None,
) -> str:
    result = _git(
        workspace,
        *arguments,
        timeout=timeout,
        index_file=index_file,
    )
    if result.returncode != 0:
        detail = bytes(result.stderr or b"").decode("utf-8", errors="replace")
        raise SubAgentChangeError(
            f"Git could not snapshot mutable work: {detail.strip()[-500:] or 'unknown error'}"
        )
    try:
        return bytes(result.stdout or b"").decode("utf-8")
    except UnicodeError as exc:
        raise SubAgentChangeError("Git returned a non-UTF-8 path receipt") from exc


def snapshot_mutable_changes(
    workspace: Path,
    output_dir: Path,
    agent_id: str,
) -> dict:
    """Freeze one detached worktree delta before publishing terminal status."""

    workspace = Path(workspace).resolve(strict=True)
    output_dir = Path(output_dir).resolve(strict=True)
    binding = read_owned_json(output_dir / MUTABLE_WORKSPACE_RECEIPT)
    if (
        binding.get("schema") != 1
        or binding.get("agent_id") != str(agent_id)
        or binding.get("read_only") is not False
    ):
        raise SubAgentChangeError("mutable workspace receipt identity is invalid")

    base_commit = str(binding.get("base_commit") or "").strip().lower()
    if not _COMMIT_RE.fullmatch(base_commit):
        raise SubAgentChangeError("mutable workspace receipt has an invalid base commit")
    expected_root = Path(str(binding.get("worktree_repository") or "")).resolve()
    actual_root = Path(
        _git_text(workspace, "rev-parse", "--show-toplevel", timeout=10).strip()
    ).resolve(strict=True)
    if actual_root != expected_root:
        raise SubAgentChangeError("mutable worktree repository identity changed")

    relative_workspace = validate_relative_path(
        binding.get("relative_workspace") or "."
    )
    expected_workspace = (
        actual_root if relative_workspace == "." else actual_root / relative_workspace
    ).resolve(strict=True)
    if expected_workspace != workspace:
        raise SubAgentChangeError("mutable child workspace no longer matches its receipt")

    _git_text(actual_root, "cat-file", "-e", f"{base_commit}^{{commit}}", timeout=10)
    child_head = _git_text(
        actual_root,
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
        timeout=10,
    ).strip().lower()
    if child_head != base_commit:
        raise SubAgentChangeError(
            "mutable child changed its detached HEAD; only uncommitted workspace edits are transferable"
        )

    # Use a private temporary index so intent-to-add can expose untracked source
    # files without mutating either the child's index or the repository's shared
    # Git common directory. Ignored build/cache output remains excluded.
    index_descriptor, index_name = tempfile.mkstemp(
        dir=output_dir,
        prefix=".mutable-index-",
    )
    os.close(index_descriptor)
    os.unlink(index_name)
    index_path = Path(index_name)
    try:
        _git_text(
            actual_root,
            "read-tree",
            base_commit,
            timeout=30,
            index_file=index_path,
        )
        _git_text(
            actual_root,
            "add",
            "-N",
            "--",
            relative_workspace,
            timeout=30,
            index_file=index_path,
        )

        names = _git_text(
            actual_root,
            "diff",
            "--name-only",
            "--no-renames",
            "--no-ext-diff",
            "--no-textconv",
            "-z",
            base_commit,
            "--",
            relative_workspace,
            timeout=30,
            index_file=index_path,
        ).split("\x00")
        changed_paths = [validate_relative_path(item) for item in names if item]
        if len(changed_paths) > MAX_CHANGED_PATHS:
            raise SubAgentChangeError(
                f"mutable child changed too many paths ({len(changed_paths)} > {MAX_CHANGED_PATHS})"
            )

        raw_parts = _git_text(
            actual_root,
            "diff",
            "--raw",
            "--no-renames",
            "--no-ext-diff",
            "--no-textconv",
            "-z",
            base_commit,
            "--",
            relative_workspace,
            timeout=30,
            index_file=index_path,
        ).split("\x00")
        raw_parts = [item for item in raw_parts if item]
        if len(raw_parts) % 2:
            raise SubAgentChangeError("Git returned a malformed raw change manifest")
        path_changes = []
        raw_paths = []
        header_re = re.compile(
            r"\A:(?P<old>[0-7]{6}) (?P<new>[0-7]{6}) "
            r"[0-9a-f]+ [0-9a-f]+ (?P<status>[AMD])\Z"
        )
        for offset in range(0, len(raw_parts), 2):
            match = header_re.fullmatch(raw_parts[offset])
            if match is None:
                raise SubAgentChangeError(
                    "mutable patch contains a rename, copy, conflict, or unsupported file type"
                )
            path = validate_relative_path(raw_parts[offset + 1])
            old_mode = match.group("old")
            new_mode = match.group("new")
            if old_mode not in {"000000", "100644", "100755"} or new_mode not in {
                "000000",
                "100644",
                "100755",
            }:
                raise SubAgentChangeError(
                    "mutable patch contains a symlink, submodule, or non-regular file"
                )
            raw_paths.append(path)
            path_changes.append(
                {
                    "path": path,
                    "status": match.group("status"),
                    "old_mode": old_mode,
                    "new_mode": new_mode,
                }
            )
        if raw_paths != changed_paths:
            raise SubAgentChangeError("Git change manifests disagree on exact paths")

        for item in path_changes:
            target = actual_root / item["path"]
            if item["status"] == "D":
                item["final_size"] = 0
                item["final_sha256"] = ""
                continue
            try:
                metadata = target.lstat()
            except OSError as exc:
                raise SubAgentChangeError(
                    f"changed path disappeared before snapshot: {item['path']}"
                ) from exc
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise SubAgentChangeError(
                    "mutable patch contains a symlink, linked file, or non-regular file"
                )
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            final = target.lstat()
            if (
                final.st_dev,
                final.st_ino,
                final.st_size,
                final.st_mtime_ns,
            ) != (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            ):
                raise SubAgentChangeError(
                    f"changed path moved during snapshot: {item['path']}"
                )
            item["final_size"] = int(metadata.st_size)
            item["final_sha256"] = digest.hexdigest()

        patch_path = output_dir / MUTABLE_PATCH_FILE
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_dir,
            prefix=".mutable-changes-",
        )
        try:
            with os.fdopen(descriptor, "wb") as patch_stream:
                diff = _git(
                    actual_root,
                    "diff",
                    "--binary",
                    "--full-index",
                    "--no-renames",
                    "--no-ext-diff",
                    "--no-textconv",
                    base_commit,
                    "--",
                    relative_workspace,
                    timeout=60,
                    stdout=patch_stream,
                    index_file=index_path,
                )
            if diff.returncode != 0:
                detail = bytes(diff.stderr or b"").decode("utf-8", errors="replace")
                raise SubAgentChangeError(
                    f"Git could not create mutable patch: {detail.strip()[-500:] or 'unknown error'}"
                )
            patch_size = os.path.getsize(temporary_name)
            if patch_size > MAX_PATCH_BYTES:
                raise SubAgentChangeError(
                    f"mutable patch exceeds bounded size ({patch_size} > {MAX_PATCH_BYTES} bytes)"
                )
            digest = hashlib.sha256()
            with open(temporary_name, "rb") as patch_stream:
                for chunk in iter(lambda: patch_stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, patch_path)
            temporary_name = ""
        finally:
            if temporary_name:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass
    finally:
        try:
            if index_path.exists():
                index_path.unlink()
        except OSError:
            pass

    receipt = {
        "schema": 1,
        "agent_id": str(agent_id),
        "base_commit": base_commit,
        "child_head": child_head,
        "relative_workspace": relative_workspace,
        "patch_file": MUTABLE_PATCH_FILE,
        "patch_sha256": digest.hexdigest(),
        "patch_bytes": patch_size,
        "changed_paths": changed_paths,
        "path_changes": path_changes,
        "empty": patch_size == 0,
    }
    rt.atomic_write_json(output_dir / MUTABLE_CHANGE_RECEIPT, receipt)
    os.chmod(output_dir / MUTABLE_CHANGE_RECEIPT, 0o600)
    return receipt


def validate_patch_file(path: Path, *, expected_size: int, expected_sha256: str) -> None:
    """Verify the immutable patch artifact immediately before integration."""

    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise SubAgentChangeError("mutable patch artifact is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o022
        or metadata.st_size != expected_size
        or metadata.st_size > MAX_PATCH_BYTES
    ):
        raise SubAgentChangeError("mutable patch failed owner/private file validation")
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SubAgentChangeError("mutable patch could not be read") from exc
    if digest.hexdigest() != expected_sha256:
        raise SubAgentChangeError("mutable patch digest does not match its receipt")
