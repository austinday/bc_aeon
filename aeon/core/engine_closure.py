"""Canonical, drift-resistant identity for an extracted bare Python engine tree."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping


ENGINE_CLOSURE_SCHEMA = "aeon-bare-engine-closure-v1"
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class EngineClosureError(RuntimeError):
    """The extracted engine is unsafe, unstable, or not the reviewed closure."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        value not in {"", "."}
        and not path.is_absolute()
        and path.as_posix() == value
        and ".." not in path.parts
        and len(value.encode("utf-8")) <= 4096
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _stable_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_private_regular(path: Path, *, maximum_bytes: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise EngineClosureError("engine closure receipt is unreadable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or before.st_mode & 0o022
        or not 0 < before.st_size <= maximum_bytes
    ):
        raise EngineClosureError("engine closure receipt is unsafe")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EngineClosureError("engine closure receipt is unreadable") from exc
    try:
        opened = os.fstat(descriptor)
        if _stable_metadata(opened) != _stable_metadata(before):
            raise EngineClosureError("engine closure receipt changed before read")
        payload = bytearray()
        while len(payload) <= maximum_bytes:
            block = os.read(descriptor, min(64 * 1024, maximum_bytes + 1 - len(payload)))
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
        if (
            len(payload) > maximum_bytes
            or _stable_metadata(after) != _stable_metadata(before)
        ):
            raise EngineClosureError("engine closure receipt changed during read")
    finally:
        os.close(descriptor)
    try:
        final = path.lstat()
    except OSError as exc:
        raise EngineClosureError("engine closure receipt disappeared") from exc
    if _stable_metadata(final) != _stable_metadata(before):
        raise EngineClosureError("engine closure receipt path changed")
    return bytes(payload)


def load_engine_closure_receipt(path: Path) -> dict[str, Any]:
    """Load and structurally validate one reviewed compact closure receipt."""

    try:
        raw = json.loads(_read_private_regular(path, maximum_bytes=64 * 1024))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EngineClosureError("engine closure receipt is unreadable") from exc
    required = {
        "schema",
        "archive_sha256",
        "root",
        "manifest_sha256",
        "entries",
        "files",
        "directories",
        "symlinks",
        "regular_bytes",
        "allowed_symlinks",
        "python_executable_sha256",
        "python_executable_bytes",
        "python_version",
        "python_cache_tag",
        "python_soabi",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise EngineClosureError("engine closure receipt schema changed")
    links = raw.get("allowed_symlinks")
    if (
        raw.get("schema") != ENGINE_CLOSURE_SCHEMA
        or raw.get("root") != "venv"
        or _SHA256_RE.fullmatch(str(raw.get("archive_sha256") or "")) is None
        or _SHA256_RE.fullmatch(str(raw.get("manifest_sha256") or "")) is None
        or _SHA256_RE.fullmatch(
            str(raw.get("python_executable_sha256") or "")
        )
        is None
        or not isinstance(raw.get("python_version"), str)
        or not raw["python_version"]
        or not isinstance(raw.get("python_cache_tag"), str)
        or not raw["python_cache_tag"]
        or not isinstance(raw.get("python_soabi"), str)
        or not raw["python_soabi"]
        or not isinstance(links, dict)
        or any(
            not isinstance(relative, str)
            or not _safe_relative(relative)
            or not isinstance(target, str)
            or not target
            or "\x00" in target
            for relative, target in links.items()
        )
    ):
        raise EngineClosureError("engine closure receipt identity changed")
    for name in (
        "entries",
        "files",
        "directories",
        "regular_bytes",
        "python_executable_bytes",
    ):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise EngineClosureError("engine closure receipt bounds changed")
    symlinks = raw.get("symlinks")
    if (
        isinstance(symlinks, bool)
        or not isinstance(symlinks, int)
        or symlinks < 0
        or symlinks != len(links)
        or raw["entries"] != raw["files"] + raw["directories"] + symlinks
    ):
        raise EngineClosureError("engine closure receipt counts changed")
    return raw


def closure_request_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact non-origin fields carried in one worker request."""

    return {
        "schema": receipt["schema"],
        "manifest_sha256": receipt["manifest_sha256"],
        "entries": receipt["entries"],
        "files": receipt["files"],
        "directories": receipt["directories"],
        "symlinks": receipt["symlinks"],
        "regular_bytes": receipt["regular_bytes"],
        "allowed_symlinks": dict(receipt["allowed_symlinks"]),
        "python_executable_sha256": receipt["python_executable_sha256"],
        "python_executable_bytes": receipt["python_executable_bytes"],
        "python_version": receipt["python_version"],
        "python_cache_tag": receipt["python_cache_tag"],
        "python_soabi": receipt["python_soabi"],
    }


def _owned_and_private(metadata: os.stat_result, relative: str) -> None:
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise EngineClosureError(f"engine closure entry is mutable or unowned: {relative}")


def _hash_regular_file(path: Path, before: os.stat_result, relative: str) -> str:
    if before.st_nlink != 1:
        raise EngineClosureError(f"engine closure file has aliases: {relative}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EngineClosureError(f"engine closure file is unreadable: {relative}") from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stable_metadata(opened) != _stable_metadata(before)
        ):
            raise EngineClosureError(f"engine closure file changed before read: {relative}")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if _stable_metadata(after) != _stable_metadata(before):
            raise EngineClosureError(f"engine closure file changed during read: {relative}")
    finally:
        os.close(descriptor)
    try:
        final = path.lstat()
    except OSError as exc:
        raise EngineClosureError(f"engine closure file disappeared: {relative}") from exc
    if _stable_metadata(final) != _stable_metadata(before):
        raise EngineClosureError(f"engine closure file path changed: {relative}")
    return digest.hexdigest()


def verify_regular_file_identity(
    path: Path, *, expected_sha256: str, expected_bytes: int
) -> None:
    """Verify one private regular file through a no-follow stable descriptor."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EngineClosureError("engine executable is unreadable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o022
        or metadata.st_size != expected_bytes
        or _SHA256_RE.fullmatch(expected_sha256) is None
        or _hash_regular_file(path, metadata, path.name) != expected_sha256
    ):
        raise EngineClosureError("engine executable identity changed")


def measure_engine_closure(
    root: Path,
    allowed_symlinks: Mapping[str, str],
    *,
    max_entries: int | None = None,
    max_regular_bytes: int | None = None,
) -> dict[str, Any]:
    """Hash every path in an extracted engine without following a symlink."""

    root = Path(root)
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise EngineClosureError("engine closure root is unreadable") from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise EngineClosureError("engine closure root is not a directory")
    root_device = root_metadata.st_dev

    reviewed_links = dict(allowed_symlinks)
    records: list[dict[str, Any]] = []
    directory_snapshots: list[
        tuple[Path, str, tuple[int, ...], tuple[str, ...]]
    ] = []
    observed_links: dict[str, str] = {}
    files = directories = symlinks = regular_bytes = 0
    stack: list[tuple[Path, str]] = [(root, ".")]

    while stack:
        path, relative = stack.pop()
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise EngineClosureError(
                f"engine closure entry is unreadable: {relative}"
            ) from exc
        mode = stat.S_IMODE(metadata.st_mode)
        if metadata.st_dev != root_device:
            raise EngineClosureError(
                f"engine closure entry crosses a filesystem boundary: {relative}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            _owned_and_private(metadata, relative)
            try:
                children = sorted(path.iterdir(), key=lambda child: child.name)
                final = path.lstat()
            except OSError as exc:
                raise EngineClosureError(
                    f"engine closure directory is unreadable: {relative}"
                ) from exc
            if _stable_metadata(final) != _stable_metadata(metadata):
                raise EngineClosureError(
                    f"engine closure directory changed during scan: {relative}"
                )
            child_names = tuple(child.name for child in children)
            directory_snapshots.append(
                (path, relative, _stable_metadata(metadata), child_names)
            )
            directories += 1
            records.append({"mode": mode, "path": relative, "type": "directory"})
            for child in reversed(children):
                child_relative = (
                    child.name if relative == "." else f"{relative}/{child.name}"
                )
                if not _safe_relative(child_relative):
                    raise EngineClosureError("engine closure path is unsafe")
                stack.append((child, child_relative))
        elif stat.S_ISREG(metadata.st_mode):
            _owned_and_private(metadata, relative)
            if max_regular_bytes is not None and (
                regular_bytes + metadata.st_size > max_regular_bytes
            ):
                raise EngineClosureError("engine closure regular-byte bound exceeded")
            files += 1
            regular_bytes += metadata.st_size
            records.append(
                {
                    "mode": mode,
                    "path": relative,
                    "sha256": _hash_regular_file(path, metadata, relative),
                    "size": metadata.st_size,
                    "type": "file",
                }
            )
        elif stat.S_ISLNK(metadata.st_mode):
            if metadata.st_uid != os.geteuid() or metadata.st_nlink != 1:
                raise EngineClosureError(
                    f"engine closure symlink is unowned or aliased: {relative}"
                )
            try:
                target = os.readlink(path)
                final = path.lstat()
            except OSError as exc:
                raise EngineClosureError(
                    f"engine closure symlink is unreadable: {relative}"
                ) from exc
            if (
                reviewed_links.get(relative) != target
                or _stable_metadata(final) != _stable_metadata(metadata)
            ):
                raise EngineClosureError(
                    f"engine closure symlink is unreviewed or changed: {relative}"
                )
            observed_links[relative] = target
            symlinks += 1
            records.append({"path": relative, "target": target, "type": "symlink"})
        else:
            raise EngineClosureError(f"engine closure inode type is unsafe: {relative}")

        if max_entries is not None and len(records) > max_entries:
            raise EngineClosureError("engine closure entry bound exceeded")

    if observed_links != reviewed_links:
        raise EngineClosureError("engine closure symlink path set changed")
    for path, relative, expected_metadata, expected_names in directory_snapshots:
        try:
            final = path.lstat()
            final_names = tuple(sorted(child.name for child in path.iterdir()))
        except OSError as exc:
            raise EngineClosureError(
                f"engine closure directory disappeared: {relative}"
            ) from exc
        if _stable_metadata(final) != expected_metadata or final_names != expected_names:
            raise EngineClosureError(
                f"engine closure directory path set changed: {relative}"
            )

    records.sort(key=lambda record: record["path"])
    manifest_sha256 = _canonical_sha256(
        {"entries": records, "schema": ENGINE_CLOSURE_SCHEMA}
    )
    return {
        "schema": ENGINE_CLOSURE_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "entries": len(records),
        "files": files,
        "directories": directories,
        "symlinks": symlinks,
        "regular_bytes": regular_bytes,
        "allowed_symlinks": reviewed_links,
    }


def verify_engine_closure(root: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute and compare the full extracted closure against its receipt."""

    expected = closure_request_identity(receipt)
    expected_tree = {
        name: expected[name]
        for name in (
            "schema",
            "manifest_sha256",
            "entries",
            "files",
            "directories",
            "symlinks",
            "regular_bytes",
            "allowed_symlinks",
        )
    }
    measured = measure_engine_closure(
        root,
        expected_tree["allowed_symlinks"],
        max_entries=expected_tree["entries"],
        max_regular_bytes=expected_tree["regular_bytes"],
    )
    if measured != expected_tree:
        raise EngineClosureError("engine closure identity changed")
    return measured
