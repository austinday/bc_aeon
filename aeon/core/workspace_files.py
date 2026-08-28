"""Descriptor-bound confinement for model-facing workspace file operations.

The language model may name paths, but it does not get to widen the directory
Aeon was launched in.  This module resolves paths lexically beneath one
launch-bound root and walks every component with ``openat``/``O_NOFOLLOW``.
Reads therefore cannot escape through a symlink race, and writes replace a file
through the already-verified parent descriptor rather than re-resolving a path.
"""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import os
from pathlib import Path
import secrets
import stat
from typing import Iterator


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)

# These are tool-owned stores or common credential containers, not project
# source.  Provider/Fleet tools expose narrow opaque capabilities for them.
_SENSITIVE_COMPONENTS = frozenset(
    {
        ".aeon",
        ".aeon-command-scratch",
        ".aws",
        ".claude",
        ".codex",
        ".docker",
        ".git",
        ".gnupg",
        ".kube",
        ".ssh",
        "aeon_output",
    }
)
_SENSITIVE_BASENAMES = frozenset(
    {
        ".env",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
_SENSITIVE_PREFIXES = (
    (".config", "anthropic"),
    (".config", "gcloud"),
    (".config", "gemini"),
    (".config", "gh"),
    (".config", "google-gemini"),
    (".config", "grok"),
    (".local", "share", "keyrings"),
)


class WorkspacePathError(RuntimeError):
    """A model-selected file path could not be admitted safely."""


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


@dataclass(frozen=True)
class WorkspaceFilePath:
    absolute: Path
    parts: tuple[str, ...]


@dataclass
class OpenWorkspaceFile:
    descriptor: int
    identity: tuple[int, int, int, int]

    @property
    def proc_path(self) -> str:
        # Handlers may reopen the file several times.  /proc/self/fd keeps those
        # reads bound to this exact already-admitted inode.
        return f"/proc/self/fd/{self.descriptor}"

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "OpenWorkspaceFile":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class WorkspaceFileBoundary:
    """One immutable launch-workspace capability."""

    def __init__(
        self,
        root: str | Path,
        expected_identity: tuple[int, int] | None = None,
    ) -> None:
        try:
            canonical = Path(root).expanduser().resolve(strict=True)
            metadata = canonical.stat(follow_symlinks=False)
        except (OSError, TypeError, ValueError) as exc:
            raise WorkspacePathError(
                "COMMAND BLOCKED: the agent's launch workspace is unavailable."
            ) from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise WorkspacePathError(
                "COMMAND BLOCKED: the agent's launch workspace is not a directory."
            )
        actual = (int(metadata.st_dev), int(metadata.st_ino))
        if expected_identity is not None and tuple(expected_identity) != actual:
            raise WorkspacePathError(
                "COMMAND BLOCKED: the agent's launch workspace identity changed."
            )
        self.root = canonical
        self.root_identity = actual

    @classmethod
    def from_worker(cls, worker) -> "WorkspaceFileBoundary":
        root = getattr(worker, "workspace_root", None)
        expected = getattr(worker, "workspace_root_identity", None)
        if root is None:
            # Lightweight test/embedding workers do not necessarily use the
            # full Worker class.  Production Worker always supplies both fields.
            root = Path.cwd()
        return cls(root, expected_identity=expected)

    def bind(self, raw_path: str) -> WorkspaceFilePath:
        if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
            raise WorkspacePathError("Error: file_path parameter is invalid.")
        raw = Path(raw_path.strip()).expanduser()
        candidate = raw if raw.is_absolute() else self.root / raw
        # abspath/normpath is lexical.  Do not use resolve(), which would follow
        # model-selected symlinks before the descriptor walk can reject them.
        candidate = Path(os.path.abspath(os.path.normpath(os.fspath(candidate))))
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspacePathError(
                "COMMAND BLOCKED: the requested file is outside this agent's "
                "launch workspace."
            ) from exc
        parts = tuple(relative.parts)
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise WorkspacePathError("Error: a specific workspace file is required.")
        lowered = tuple(part.lower() for part in parts)
        if any(part in _SENSITIVE_COMPONENTS for part in lowered):
            raise WorkspacePathError(
                "COMMAND BLOCKED: direct file access to agent state, repository "
                "metadata, or credential storage is not permitted; use its reviewed tool."
            )
        if any(lowered[: len(prefix)] == prefix for prefix in _SENSITIVE_PREFIXES):
            raise WorkspacePathError(
                "COMMAND BLOCKED: direct file access to provider credential storage "
                "is not permitted; use its reviewed tool."
            )
        basename = lowered[-1]
        if basename in _SENSITIVE_BASENAMES or (
            basename.startswith(".env.")
            and not basename.endswith((".example", ".sample", ".template"))
        ):
            raise WorkspacePathError(
                "COMMAND BLOCKED: direct access to a credential-like file is not "
                "permitted; use an opaque Nexus credential capability."
            )
        return WorkspaceFilePath(candidate, parts)

    def _open_root(self) -> int:
        try:
            descriptor = os.open(
                self.root,
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
            )
        except OSError as exc:
            raise WorkspacePathError(
                "COMMAND BLOCKED: the agent's launch workspace cannot be reopened safely."
            ) from exc
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (int(metadata.st_dev), int(metadata.st_ino)) != self.root_identity
        ):
            os.close(descriptor)
            raise WorkspacePathError(
                "COMMAND BLOCKED: the agent's launch workspace identity changed."
            )
        return descriptor

    def _open_parent(self, path: WorkspaceFilePath, *, create: bool) -> tuple[int, str]:
        current = self._open_root()
        try:
            for component in path.parts[:-1]:
                try:
                    child = os.open(
                        component,
                        os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                        dir_fd=current,
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, mode=0o755, dir_fd=current)
                    child = os.open(
                        component,
                        os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC,
                        dir_fd=current,
                    )
                metadata = os.fstat(child)
                if not stat.S_ISDIR(metadata.st_mode):
                    os.close(child)
                    raise WorkspacePathError(
                        "COMMAND BLOCKED: a requested path component is not a safe directory."
                    )
                os.close(current)
                current = child
            return current, path.parts[-1]
        except FileNotFoundError as exc:
            os.close(current)
            raise WorkspacePathError(f"Error: File not found: {path.absolute}") from exc
        except OSError as exc:
            os.close(current)
            if exc.errno in {getattr(os, "ELOOP", 40), getattr(os, "ENOTDIR", 20)}:
                raise WorkspacePathError(
                    "COMMAND BLOCKED: refusing a path containing a symlink or "
                    "non-directory ancestor."
                ) from exc
            raise WorkspacePathError(
                f"Error: could not traverse workspace path: {type(exc).__name__}: {exc}"
            ) from exc
        except Exception:
            os.close(current)
            raise

    def open_regular(self, path: WorkspaceFilePath) -> OpenWorkspaceFile:
        parent, leaf = self._open_parent(path, create=False)
        try:
            descriptor = os.open(
                leaf,
                os.O_RDONLY | _NOFOLLOW | _CLOEXEC,
                dir_fd=parent,
            )
        except FileNotFoundError as exc:
            raise WorkspacePathError(f"Error: File not found: {path.absolute}") from exc
        except OSError as exc:
            if exc.errno == getattr(os, "ELOOP", 40):
                raise WorkspacePathError(
                    "COMMAND BLOCKED: refusing direct access through a symlink."
                ) from exc
            raise WorkspacePathError(
                f"Error: could not open workspace file: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            os.close(parent)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            if stat.S_ISDIR(metadata.st_mode):
                raise WorkspacePathError(f"Error: {path.absolute} is a directory, not a file.")
            raise WorkspacePathError("COMMAND BLOCKED: requested path is not a regular file.")
        if int(metadata.st_nlink) != 1:
            os.close(descriptor)
            raise WorkspacePathError(
                "COMMAND BLOCKED: refusing a multiply-linked file because its content "
                "may be owned outside this workspace."
            )
        return OpenWorkspaceFile(descriptor, _identity(metadata))

    def identity_is_current(
        self,
        path: WorkspaceFilePath,
        expected: tuple[int, int, int, int],
    ) -> bool:
        try:
            with self.open_regular(path) as current:
                return current.identity == tuple(expected)
        except WorkspacePathError:
            return False

    def atomic_write(
        self,
        path: WorkspaceFilePath,
        content: str | bytes,
        *,
        binary: bool,
        expected_identity: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int]:
        """Atomically replace ``path`` through its verified parent descriptor."""

        parent, leaf = self._open_parent(path, create=True)
        temporary = f".aeon_tmp_{secrets.token_hex(16)}"
        descriptor = -1
        try:
            try:
                current = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                current = None
            if current is not None:
                if not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode):
                    raise WorkspacePathError(
                        "COMMAND BLOCKED: refusing to replace a symlink or non-regular file."
                    )
                if int(current.st_nlink) != 1:
                    raise WorkspacePathError(
                        "COMMAND BLOCKED: refusing to replace a multiply-linked file."
                    )
                if expected_identity is None or _identity(current) != tuple(expected_identity):
                    raise WorkspacePathError(
                        "COMMAND BLOCKED: the destination changed concurrently before write."
                    )
                mode = stat.S_IMODE(current.st_mode)
            else:
                if expected_identity is not None:
                    raise WorkspacePathError(
                        "COMMAND BLOCKED: the destination disappeared before write."
                    )
                mode = 0o644

            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC,
                0o600,
                dir_fd=parent,
            )
            if binary:
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    descriptor = -1
                    handle.write(content)  # type: ignore[arg-type]
                    handle.flush()
                    os.fsync(handle.fileno())
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
                    descriptor = -1
                    handle.write(content)  # type: ignore[arg-type]
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(
                temporary,
                leaf,
                src_dir_fd=parent,
                dst_dir_fd=parent,
            )
            os.chmod(leaf, mode, dir_fd=parent, follow_symlinks=False)
            os.fsync(parent)
            written = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISREG(written.st_mode) or int(written.st_nlink) != 1:
                raise WorkspacePathError(
                    "COMMAND BLOCKED: the written destination is not a unique regular file."
                )
            return _identity(written)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            with contextlib.suppress(FileNotFoundError, OSError):
                os.unlink(temporary, dir_fd=parent)
            os.close(parent)

    @contextlib.contextmanager
    def open_for_read(self, path: WorkspaceFilePath) -> Iterator[OpenWorkspaceFile]:
        opened = self.open_regular(path)
        try:
            yield opened
        finally:
            opened.close()
