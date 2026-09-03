"""Configuration for the Aeon remote console.

Secrets are environment/runtime state, never repository configuration. Production
starts fail closed unless an HTTPS origin is explicitly named.
"""

from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from aeon.core.model_identity import AEON_DEFAULT_MODEL_NAME


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def lexical_absolute_path(value: str | Path) -> Path:
    """Normalize dots and ``~`` without dereferencing any path component."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def open_directory_no_symlinks(path: str | Path) -> int:
    """Open an absolute directory after rejecting every symbolic-link component."""

    if getattr(os, "O_NOFOLLOW", None) is None or getattr(os, "O_DIRECTORY", None) is None:
        raise RuntimeError(
            "This platform cannot safely open remote state without following links"
        )
    absolute = lexical_absolute_path(path)
    parts = absolute.parts
    if not absolute.is_absolute() or not parts:
        raise RuntimeError("State directory path must be absolute")
    descriptor = os.open(absolute.anchor, _DIRECTORY_OPEN_FLAGS)
    try:
        for component in parts[1:]:
            child = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_private_child_directory(parent_fd: int, name: str) -> int:
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        raise RuntimeError("Private state directory name is invalid")
    created = False
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    child_fd = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    try:
        metadata = os.fstat(child_fd)
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or (metadata.st_dev, metadata.st_ino) != (entry.st_dev, entry.st_ino)
        ):
            raise RuntimeError(
                "Remote state directories must be real directories owned by the service user"
            )
        os.fchmod(child_fd, 0o700)
        confirmed = os.fstat(child_fd)
        if stat.S_IMODE(confirmed.st_mode) != 0o700:
            raise RuntimeError("Remote state directories must use mode 0700")
        if created:
            os.fsync(parent_fd)
        return child_fd
    except Exception:
        os.close(child_fd)
        raise


def ensure_private_directory(path: str | Path) -> int:
    """Create/open one owner-private directory without following path links."""

    absolute = lexical_absolute_path(path)
    if absolute == Path(absolute.anchor):
        raise RuntimeError("The filesystem root cannot be used as remote state")
    parent = absolute.parent
    if parent == absolute:
        raise RuntimeError("Remote state directory has no safe parent")
    try:
        parent_fd = open_directory_no_symlinks(parent)
    except FileNotFoundError:
        # Create missing ancestors one component at a time through already-open
        # directory descriptors. Existing components are never followed links.
        parts = parent.parts
        descriptor = os.open(parent.anchor, _DIRECTORY_OPEN_FLAGS)
        try:
            for component in parts[1:]:
                try:
                    child = os.open(
                        component,
                        _DIRECTORY_OPEN_FLAGS,
                        dir_fd=descriptor,
                    )
                except FileNotFoundError:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    child = os.open(
                        component,
                        _DIRECTORY_OPEN_FLAGS,
                        dir_fd=descriptor,
                    )
                    os.fchmod(child, 0o700)
                os.close(descriptor)
                descriptor = child
            parent_fd = descriptor
        except Exception:
            os.close(descriptor)
            raise
    try:
        return _open_private_child_directory(parent_fd, absolute.name)
    finally:
        os.close(parent_fd)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _paths(value: str | None, default: Iterable[Path]) -> tuple[Path, ...]:
    raw = [p for p in (value or "").split(os.pathsep) if p.strip()]
    paths = [Path(p).expanduser().resolve() for p in raw] if raw else list(default)
    return tuple(dict.fromkeys(paths))


@dataclass(frozen=True)
class RemoteConfig:
    project_root: Path
    state_dir: Path
    allowed_roots: tuple[Path, ...]
    allowed_origins: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    python_executable: str = sys.executable
    tmux_binary: str = "/usr/bin/tmux"
    session_hours: int = 12
    remembered_session_days: int = 30
    # Aeon Remote keeps its defense-in-depth default. Nexus explicitly opts into
    # its single-screen password/OIDC flow in the task-owned service unit.
    require_totp: bool = True
    allow_insecure_http: bool = False
    default_model: str = AEON_DEFAULT_MODEL_NAME
    coordinator_path: Path = Path("/home/aday/website_hosting/gpu_coord.py")
    coordinator_cwd: Path = Path("/home/aday/website_hosting/ads")
    expected_coordinator_host: str = "DAY2RTX6000PRO"

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_dir", lexical_absolute_path(self.state_dir))

    @property
    def database_path(self) -> Path:
        return self.state_dir / "remote.sqlite3"

    @property
    def instance_state_dir(self) -> Path:
        return self.state_dir / "instances"

    @property
    def cookie_name(self) -> str:
        return "aeon_remote_dev" if self.allow_insecure_http else "__Host-aeon_remote"

    @property
    def cookie_samesite(self) -> str:
        # The session is created before the browser is redirected to Cloudflare's
        # OIDC callback. Lax sends it on the subsequent top-level HTTPS return to
        # Nexus, while HttpOnly + CSRF + exact-Origin checks protect mutations.
        # Standalone Aeon Remote never leaves its origin during TOTP login and
        # therefore keeps the stronger Strict behavior.
        return "strict" if self.require_totp else "lax"

    @classmethod
    def from_env(cls, *, validate_server: bool = True) -> "RemoteConfig":
        project_root = Path(
            os.environ.get("AEON_REMOTE_PROJECT_ROOT", Path(__file__).resolve().parents[2])
        ).expanduser().resolve()
        state_dir = Path(
            os.environ.get("AEON_REMOTE_STATE_DIR", "~/.aeon/remote")
        )
        default_root = Path(os.environ.get("AEON_REMOTE_DEFAULT_ROOT", "~/aeon_workspaces"))
        roots = _paths(
            os.environ.get("AEON_REMOTE_ALLOWED_ROOTS"),
            [default_root.expanduser().resolve()],
        )
        origins = tuple(
            origin.strip().rstrip("/")
            for origin in os.environ.get("AEON_REMOTE_ORIGINS", "").split(",")
            if origin.strip()
        )
        hosts = tuple(
            host.strip()
            for host in os.environ.get("AEON_REMOTE_HOSTS", "").split(",")
            if host.strip()
        )
        if not hosts:
            hosts = tuple(
                parsed.hostname for parsed in map(urlparse, origins) if parsed.hostname
            ) or ("127.0.0.1", "localhost")

        config = cls(
            project_root=project_root,
            state_dir=state_dir,
            allowed_roots=roots,
            allowed_origins=origins,
            allowed_hosts=hosts,
            python_executable=os.environ.get("AEON_REMOTE_PYTHON", sys.executable),
            tmux_binary=os.environ.get("AEON_REMOTE_TMUX", "/usr/bin/tmux"),
            session_hours=max(1, int(os.environ.get("AEON_REMOTE_SESSION_HOURS", "12"))),
            remembered_session_days=max(
                1, int(os.environ.get("AEON_REMOTE_REMEMBER_DAYS", "30"))
            ),
            require_totp=not _truthy(os.environ.get("AEON_REMOTE_DISABLE_TOTP")),
            allow_insecure_http=_truthy(os.environ.get("AEON_REMOTE_INSECURE_HTTP")),
            default_model=os.environ.get(
                "AEON_REMOTE_MODEL", AEON_DEFAULT_MODEL_NAME
            ).strip()
            or AEON_DEFAULT_MODEL_NAME,
            coordinator_path=Path(
                os.environ.get(
                    "AEON_REMOTE_GPU_COORD", "/home/aday/website_hosting/gpu_coord.py"
                )
            ).expanduser().resolve(),
            coordinator_cwd=Path(
                os.environ.get(
                    "AEON_REMOTE_GPU_COORD_CWD", "/home/aday/website_hosting/ads"
                )
            ).expanduser().resolve(),
        )
        if validate_server:
            config.validate_server()
        return config

    def validate_server(self) -> None:
        if not self.project_root.is_dir():
            raise RuntimeError(f"Aeon project root does not exist: {self.project_root}")
        if not self.allow_insecure_http:
            if not self.allowed_origins:
                raise RuntimeError(
                    "AEON_REMOTE_ORIGINS must name the exact public HTTPS origin"
                )
            bad = [origin for origin in self.allowed_origins if not origin.startswith("https://")]
            if bad:
                raise RuntimeError(f"Production origins must use HTTPS: {bad}")
        if not self.allowed_roots:
            raise RuntimeError("At least one AEON_REMOTE_ALLOWED_ROOTS entry is required")

    def prepare_state(self) -> None:
        state_fd = ensure_private_directory(self.state_dir)
        try:
            instances_fd = _open_private_child_directory(state_fd, "instances")
            os.close(instances_fd)
        finally:
            os.close(state_fd)
