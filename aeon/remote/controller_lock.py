"""Single-controller lease for the shared Aeon Remote state registry."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import socket
import stat
from pathlib import Path

from .config import lexical_absolute_path, open_directory_no_symlinks


CONTROLLER_LOCK_FILENAME = "controller.lock"


class ControllerLockError(RuntimeError):
    """Raised when exclusive control of one remote registry cannot be proven."""


class ControllerReadLease:
    """Filesystem-read-only exclusion against a live controller lifetime."""

    def __init__(
        self,
        singleton: socket.socket,
        name: bytes,
        state_path: Path,
        *,
        directory_fd: int = -1,
        directory_identity: tuple[int, int] | None = None,
        lock_fd: int = -1,
        lock_identity: tuple[int, int] | None = None,
    ):
        self._singleton = singleton
        self._name = name
        self.state_path = state_path
        self._directory_fd = directory_fd
        self._directory_identity = directory_identity
        self._lock_fd = lock_fd
        self._lock_identity = lock_identity

    @property
    def active(self) -> bool:
        return self._singleton is not None and self._singleton.fileno() >= 0

    def assert_current(self) -> None:
        if not self.active:
            raise ControllerLockError("Controller read lease is no longer active")
        try:
            if self._singleton.getsockname() != self._name:
                raise ControllerLockError("Controller read lease identity changed")
            if self._directory_fd >= 0:
                retained_directory = os.fstat(self._directory_fd)
                current_directory_fd = open_directory_no_symlinks(self.state_path)
                try:
                    current_directory = os.fstat(current_directory_fd)
                finally:
                    os.close(current_directory_fd)
                if (
                    retained_directory.st_uid != os.geteuid()
                    or stat.S_IMODE(retained_directory.st_mode) != 0o700
                    or (retained_directory.st_dev, retained_directory.st_ino)
                    != self._directory_identity
                    or (current_directory.st_dev, current_directory.st_ino)
                    != self._directory_identity
                ):
                    raise ControllerLockError(
                        "Controller read lease state directory identity changed"
                    )
                try:
                    current_lock = os.stat(
                        CONTROLLER_LOCK_FILENAME,
                        dir_fd=self._directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    if self._lock_fd >= 0:
                        raise ControllerLockError(
                            "Controller read lease lock identity changed"
                        ) from None
                else:
                    if self._lock_fd < 0:
                        raise ControllerLockError(
                            "Controller read lease lock identity changed"
                        )
                    retained_lock = os.fstat(self._lock_fd)
                    if (
                        not stat.S_ISREG(retained_lock.st_mode)
                        or retained_lock.st_uid != os.geteuid()
                        or stat.S_IMODE(retained_lock.st_mode) != 0o600
                        or retained_lock.st_nlink != 1
                        or (retained_lock.st_dev, retained_lock.st_ino)
                        != self._lock_identity
                        or (current_lock.st_dev, current_lock.st_ino)
                        != self._lock_identity
                    ):
                        raise ControllerLockError(
                            "Controller read lease lock identity changed"
                        )
        except OSError as exc:
            raise ControllerLockError(
                "Controller read lease identity can no longer be proven"
            ) from exc

    def close(self) -> None:
        singleton, self._singleton = self._singleton, None
        directory_fd, self._directory_fd = self._directory_fd, -1
        lock_fd, self._lock_fd = self._lock_fd, -1
        try:
            if lock_fd >= 0:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            if lock_fd >= 0:
                os.close(lock_fd)
            if directory_fd >= 0:
                os.close(directory_fd)
            if singleton is not None:
                singleton.close()

    def __enter__(self) -> ControllerReadLease:
        self.assert_current()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class ControllerLock:
    """A lifetime abstract singleton plus an auditable no-follow file lock."""

    def __init__(
        self,
        fd: int,
        path: Path,
        *,
        directory_fd: int,
        directory_identity: tuple[int, int],
        lock_identity: tuple[int, int],
        singleton: socket.socket,
        singleton_name: bytes,
    ):
        self._fd = fd
        self.path = path
        self._directory_fd = directory_fd
        self._directory_identity = directory_identity
        self._lock_identity = lock_identity
        self._singleton = singleton
        self._singleton_name = singleton_name

    @staticmethod
    def _abstract_singleton_name(state_path: Path) -> bytes:
        identity = str(os.geteuid()).encode("ascii") + b":" + os.fsencode(state_path)
        digest = hashlib.sha256(identity).hexdigest().encode("ascii")
        return b"\0aeon-nexus-controller-" + digest

    @classmethod
    def _bind_singleton(cls, state_path: Path) -> tuple[socket.socket, bytes]:
        singleton_name = cls._abstract_singleton_name(state_path)
        singleton = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        singleton.set_inheritable(False)
        try:
            singleton.bind(singleton_name)
        except OSError as exc:
            singleton.close()
            if exc.errno == errno.EADDRINUSE:
                raise ControllerLockError(
                    f"Another Aeon/Nexus controller is already active for {state_path}"
                ) from exc
            raise ControllerLockError(
                "Could not establish the Aeon/Nexus controller singleton"
            ) from exc
        return singleton, singleton_name

    @classmethod
    def acquire_read_lease(cls, state_dir: str | Path) -> ControllerReadLease:
        """Exclude a controller without creating or changing filesystem state."""

        state_path = lexical_absolute_path(state_dir)
        singleton, name = cls._bind_singleton(state_path)
        directory_fd = -1
        lock_fd = -1
        try:
            try:
                directory_fd = open_directory_no_symlinks(state_path)
            except FileNotFoundError:
                lease = ControllerReadLease(singleton, name, state_path)
                singleton = None
                return lease
            directory_stat = os.fstat(directory_fd)
            if (
                directory_stat.st_uid != os.geteuid()
                or stat.S_IMODE(directory_stat.st_mode) != 0o700
            ):
                raise ControllerLockError(
                    "Aeon Remote state directory must be an owner-only mode-0700 directory"
                )
            try:
                lock_fd = os.open(
                    CONTROLLER_LOCK_FILENAME,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                lock_identity = None
            else:
                lock_stat = os.fstat(lock_fd)
                entry = os.stat(
                    CONTROLLER_LOCK_FILENAME,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(lock_stat.st_mode)
                    or lock_stat.st_uid != os.geteuid()
                    or stat.S_IMODE(lock_stat.st_mode) != 0o600
                    or lock_stat.st_nlink != 1
                    or (lock_stat.st_dev, lock_stat.st_ino)
                    != (entry.st_dev, entry.st_ino)
                ):
                    raise ControllerLockError(
                        "Controller lock must be an owner-only mode-0600 regular file"
                    )
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ControllerLockError(
                        f"Another Aeon/Nexus controller is already active for {state_path}"
                    ) from exc
                lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
            lease = ControllerReadLease(
                singleton,
                name,
                state_path,
                directory_fd=directory_fd,
                directory_identity=(directory_stat.st_dev, directory_stat.st_ino),
                lock_fd=lock_fd,
                lock_identity=lock_identity,
            )
            singleton = None
            directory_fd = -1
            lock_fd = -1
            return lease
        except ControllerLockError:
            raise
        except OSError as exc:
            raise ControllerLockError(
                "Could not safely acquire the controller read lease"
            ) from exc
        finally:
            if lock_fd >= 0:
                os.close(lock_fd)
            if directory_fd >= 0:
                os.close(directory_fd)
            if singleton is not None:
                singleton.close()

    @classmethod
    def acquire(cls, state_dir: str | Path) -> ControllerLock:
        state_path = lexical_absolute_path(state_dir)
        nofollow = getattr(os, "O_NOFOLLOW", None)
        directory = getattr(os, "O_DIRECTORY", None)
        if nofollow is None or directory is None:
            raise ControllerLockError(
                "This platform cannot safely open the controller lock without following links"
            )

        directory_fd = -1
        lock_fd = -1
        singleton: socket.socket | None = None
        lock_path = state_path / CONTROLLER_LOCK_FILENAME
        try:
            singleton, singleton_name = cls._bind_singleton(state_path)

            directory_fd = open_directory_no_symlinks(state_path)
            directory_stat = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(directory_stat.st_mode)
                or directory_stat.st_uid != os.geteuid()
                or stat.S_IMODE(directory_stat.st_mode) != 0o700
            ):
                raise ControllerLockError(
                    "Aeon Remote state directory must be an owner-only mode-0700 directory"
                )

            lock_fd = os.open(
                CONTROLLER_LOCK_FILENAME,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | nofollow,
                0o600,
                dir_fd=directory_fd,
            )
            lock_stat = os.fstat(lock_fd)
            path_stat = os.stat(
                CONTROLLER_LOCK_FILENAME,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(lock_stat.st_mode)
                or lock_stat.st_uid != os.geteuid()
                or stat.S_IMODE(lock_stat.st_mode) != 0o600
                or lock_stat.st_nlink != 1
                or (lock_stat.st_dev, lock_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
            ):
                raise ControllerLockError(
                    "Controller lock must be an owner-only mode-0600 regular file"
                )
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ControllerLockError(
                    f"Another Aeon/Nexus controller is already active for {state_path}"
                ) from exc

            # Revalidate the directory entry after acquisition so a replaced path
            # cannot make this process believe it locked the registry's live inode.
            path_stat = os.stat(
                CONTROLLER_LOCK_FILENAME,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if (lock_stat.st_dev, lock_stat.st_ino) != (
                path_stat.st_dev,
                path_stat.st_ino,
            ):
                raise ControllerLockError(
                    "Controller lock changed while exclusive control was acquired"
                )
            # Reopen the lexical directory path after the file lease is held. An
            # unlink/rename race must not strand a controller on a detached tree.
            confirmation_fd = open_directory_no_symlinks(state_path)
            try:
                confirmation_stat = os.fstat(confirmation_fd)
                if (directory_stat.st_dev, directory_stat.st_ino) != (
                    confirmation_stat.st_dev,
                    confirmation_stat.st_ino,
                ):
                    raise ControllerLockError(
                        "Controller state directory changed while control was acquired"
                    )
            finally:
                os.close(confirmation_fd)
            lease = cls(
                lock_fd,
                lock_path,
                directory_fd=directory_fd,
                directory_identity=(directory_stat.st_dev, directory_stat.st_ino),
                lock_identity=(lock_stat.st_dev, lock_stat.st_ino),
                singleton=singleton,
                singleton_name=singleton_name,
            )
            lock_fd = -1
            directory_fd = -1
            singleton = None
            return lease
        except ControllerLockError:
            raise
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                detail = "Controller state or lock path must not be a symbolic link"
            else:
                detail = "Could not safely acquire the Aeon/Nexus controller lock"
            raise ControllerLockError(detail) from exc
        finally:
            if lock_fd >= 0:
                os.close(lock_fd)
            if directory_fd >= 0:
                os.close(directory_fd)
            if singleton is not None:
                singleton.close()

    @property
    def active(self) -> bool:
        return (
            self._fd >= 0
            and self._directory_fd >= 0
            and self._singleton is not None
            and self._singleton.fileno() >= 0
        )

    def assert_current(self) -> None:
        """Fail closed if any retained or lexical controller identity changed."""

        if not self.active:
            raise ControllerLockError("Aeon/Nexus controller lock is no longer active")
        try:
            if self._singleton.getsockname() != self._singleton_name:
                raise ControllerLockError("Aeon/Nexus controller singleton changed")

            retained_directory = os.fstat(self._directory_fd)
            if (
                not stat.S_ISDIR(retained_directory.st_mode)
                or retained_directory.st_uid != os.geteuid()
                or stat.S_IMODE(retained_directory.st_mode) != 0o700
                or (retained_directory.st_dev, retained_directory.st_ino)
                != self._directory_identity
            ):
                raise ControllerLockError("Controller state directory identity changed")

            current_directory_fd = open_directory_no_symlinks(self.path.parent)
            try:
                current_directory = os.fstat(current_directory_fd)
                if (
                    current_directory.st_uid != os.geteuid()
                    or stat.S_IMODE(current_directory.st_mode) != 0o700
                    or (current_directory.st_dev, current_directory.st_ino)
                    != self._directory_identity
                ):
                    raise ControllerLockError(
                        "Controller state directory identity changed"
                    )
            finally:
                os.close(current_directory_fd)

            retained_lock = os.fstat(self._fd)
            current_lock = os.stat(
                CONTROLLER_LOCK_FILENAME,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(retained_lock.st_mode)
                or retained_lock.st_uid != os.geteuid()
                or stat.S_IMODE(retained_lock.st_mode) != 0o600
                or retained_lock.st_nlink != 1
                or (retained_lock.st_dev, retained_lock.st_ino)
                != self._lock_identity
                or self._lock_identity != (current_lock.st_dev, current_lock.st_ino)
            ):
                raise ControllerLockError("Controller lock file identity changed")
        except ControllerLockError:
            raise
        except OSError as exc:
            raise ControllerLockError("Controller identity can no longer be proven") from exc

    def close(self) -> None:
        if (
            self._fd < 0
            and self._directory_fd < 0
            and (self._singleton is None or self._singleton.fileno() < 0)
        ):
            return
        fd, self._fd = self._fd, -1
        directory_fd, self._directory_fd = self._directory_fd, -1
        singleton, self._singleton = self._singleton, None
        try:
            if fd >= 0:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            if fd >= 0:
                os.close(fd)
            if directory_fd >= 0:
                os.close(directory_fd)
            if singleton is not None:
                singleton.close()

    def __enter__(self) -> ControllerLock:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
