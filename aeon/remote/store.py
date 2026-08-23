"""SQLite-backed durable registry for remote users, sessions, and Aeon tabs."""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import sqlite3
import stat
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from .agent_settings import catalog_for, normalize_settings
from .config import (
    ensure_private_directory,
    lexical_absolute_path,
    open_directory_no_symlinks,
)


class _ClosingConnection(sqlite3.Connection):
    """sqlite3's context manager commits but does not close; this one does both."""

    def _guard_operation(self) -> None:
        guard = getattr(self, "_operation_guard", None)
        if guard is not None:
            guard()

    def execute(self, *args, **kwargs):
        self._guard_operation()
        return super().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        self._guard_operation()
        return super().executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        self._guard_operation()
        return super().executescript(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    totp_secret TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS web_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    csrf_token TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    last_seen REAL NOT NULL,
    user_agent_hash TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS web_sessions_expiry ON web_sessions(expires_at);
CREATE TABLE IF NOT EXISTS login_attempts (
    rate_key TEXT NOT NULL,
    attempted_at REAL NOT NULL,
    succeeded INTEGER NOT NULL,
    attempt_id TEXT
);
CREATE INDEX IF NOT EXISTS login_attempts_key_time ON login_attempts(rate_key, attempted_at);
CREATE TABLE IF NOT EXISTS instances (
    id TEXT PRIMARY KEY,
    host_id TEXT NOT NULL DEFAULT '192.168.0.177',
    kind TEXT NOT NULL DEFAULT 'aeon',
    shell_backed INTEGER NOT NULL DEFAULT 0,
    last_agent_kind TEXT,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    tmux_name TEXT NOT NULL UNIQUE,
    workspace TEXT NOT NULL,
    objective TEXT NOT NULL DEFAULT '',
    max_iterations INTEGER,
    model TEXT,
    status TEXT NOT NULL,
    desired_state TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_started_at REAL,
    transport_pid INTEGER,
    transport_process_create_time REAL,
    last_error TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL,
    launch_origin TEXT NOT NULL DEFAULT 'web'
);
CREATE TABLE IF NOT EXISTS instruction_profiles (
    id TEXT PRIMARY KEY,
    agent_kind TEXT NOT NULL,
    name TEXT NOT NULL COLLATE NOCASE,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    created_by TEXT NOT NULL,
    UNIQUE(agent_kind, name)
);
CREATE TABLE IF NOT EXISTS instruction_profile_versions (
    id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES instruction_profiles(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL CHECK(version_number > 0),
    label TEXT NOT NULL,
    content TEXT NOT NULL CHECK(length(CAST(content AS BLOB)) <= 65536),
    content_sha256 TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_ref TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    created_by TEXT NOT NULL,
    UNIQUE(profile_id, version_number)
);
CREATE INDEX IF NOT EXISTS instruction_profile_versions_profile
    ON instruction_profile_versions(profile_id, version_number DESC);
CREATE TABLE IF NOT EXISTS instance_instruction_bindings (
    instance_id TEXT PRIMARY KEY REFERENCES instances(id) ON DELETE CASCADE,
    desired_profile_version_id TEXT
        REFERENCES instruction_profile_versions(id) ON DELETE SET NULL,
    applied_profile_version_id TEXT
        REFERENCES instruction_profile_versions(id) ON DELETE SET NULL,
    desired_local_revision INTEGER NOT NULL DEFAULT 0
        CHECK(desired_local_revision >= 0),
    applied_local_revision INTEGER NOT NULL DEFAULT 0
        CHECK(applied_local_revision >= 0),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS instance_local_instruction_versions (
    instance_id TEXT NOT NULL REFERENCES instances(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL CHECK(revision > 0),
    content TEXT NOT NULL CHECK(length(CAST(content AS BLOB)) <= 65536),
    content_sha256 TEXT NOT NULL,
    created_at REAL NOT NULL,
    created_by TEXT NOT NULL,
    PRIMARY KEY(instance_id, revision)
);
CREATE TABLE IF NOT EXISTS instance_agent_settings (
    instance_id TEXT NOT NULL REFERENCES instances(id) ON DELETE CASCADE,
    agent_kind TEXT NOT NULL
        CHECK(agent_kind IN ('aeon','codex','claude','grok')),
    desired_model TEXT NOT NULL,
    desired_effort TEXT NOT NULL,
    applied_model TEXT,
    applied_effort TEXT,
    updated_at REAL NOT NULL,
    applied_at REAL,
    PRIMARY KEY(instance_id, agent_kind),
    CHECK(
        (applied_model IS NULL AND applied_effort IS NULL)
        OR (applied_model IS NOT NULL AND applied_effort IS NOT NULL)
    )
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at REAL NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    instance_id TEXT,
    client_ip TEXT NOT NULL DEFAULT '',
    details_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS audit_time ON audit_log(occurred_at);
"""


class RemoteStore:
    _DATABASE_SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")

    def __init__(
        self,
        path: str | Path,
        *,
        read_only: bool = False,
        controller_guard: Callable[[], None] | None = None,
    ):
        if getattr(os, "O_NOFOLLOW", None) is None:
            raise RuntimeError(
                "This platform cannot safely open the remote registry without following links"
            )
        self.path = lexical_absolute_path(path)
        if self.path == Path(self.path.anchor):
            raise RuntimeError("Remote registry path must name a database file")
        self._read_only = bool(read_only)
        self._lock = threading.RLock()
        if controller_guard is not None and not callable(controller_guard):
            raise TypeError("controller guard must be callable")
        if self._read_only and controller_guard is None:
            raise RuntimeError("Read-only registry access requires an exclusive read lease")
        self._controller_guard = controller_guard
        self._assert_controller()
        if self._read_only:
            parent_fd = open_directory_no_symlinks(self.path.parent)
        else:
            parent_fd = ensure_private_directory(self.path.parent)
        try:
            self._parent_identity = self._validate_parent_fd(parent_fd)
            database_fd = self._open_database_fd(parent_fd, create=not self._read_only)
            try:
                metadata = self._validate_database_fd(parent_fd, database_fd)
                self._database_identity = (metadata.st_dev, metadata.st_ino)
                if not self._read_only:
                    os.fsync(database_fd)
            finally:
                os.close(database_fd)
            self._validate_sidecars(parent_fd)
        finally:
            os.close(parent_fd)
        if not self._read_only:
            self._initialize()

    def set_controller_guard(self, guard: Callable[[], None] | None) -> None:
        """Require a retained controller identity before future DB access."""

        if guard is not None and not callable(guard):
            raise TypeError("controller guard must be callable")
        with self._lock:
            self._controller_guard = guard

    def _assert_controller(self) -> None:
        guard = self._controller_guard
        if guard is not None:
            guard()

    def _validate_parent_fd(self, parent_fd: int) -> tuple[int, int]:
        metadata = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise RuntimeError(
                "Remote registry directory must be an owner-only mode-0700 directory"
            )
        return metadata.st_dev, metadata.st_ino

    def _open_parent_fd(self) -> int:
        try:
            parent_fd = open_directory_no_symlinks(self.path.parent)
        except OSError as exc:
            raise RuntimeError("Remote registry directory is unavailable or unsafe") from exc
        try:
            identity = self._validate_parent_fd(parent_fd)
            if identity != self._parent_identity:
                raise RuntimeError("Remote registry directory identity changed")
            return parent_fd
        except Exception:
            os.close(parent_fd)
            raise

    def _open_database_fd(self, parent_fd: int, *, create: bool) -> int:
        access = os.O_RDONLY if self._read_only else os.O_RDWR
        flags = access | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            return os.open(self.path.name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            if not create:
                raise RuntimeError("Remote registry database does not exist") from None
        except OSError as exc:
            raise RuntimeError("Remote registry database is unavailable or unsafe") from exc

        try:
            return os.open(
                self.path.name,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            # Another serialized initializer may have won the create race. Open
            # and validate that exact entry rather than weakening O_EXCL.
            try:
                return os.open(self.path.name, flags, dir_fd=parent_fd)
            except OSError as exc:
                raise RuntimeError(
                    "Remote registry database is unavailable or unsafe"
                ) from exc
        except OSError as exc:
            raise RuntimeError("Could not safely create remote registry database") from exc

    def _validate_database_fd(self, parent_fd: int, database_fd: int):
        metadata = os.fstat(database_fd)
        try:
            entry = os.stat(
                self.path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RuntimeError("Remote registry database identity is unavailable") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != (entry.st_dev, entry.st_ino)
        ):
            raise RuntimeError(
                "Remote registry database must be a singly-linked regular file "
                "owned by the service user"
            )
        if self._read_only:
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise RuntimeError("Remote registry database must use mode 0600")
        else:
            os.fchmod(database_fd, 0o600)
            metadata = os.fstat(database_fd)
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise RuntimeError("Remote registry database must use mode 0600")
        return metadata

    def _validate_sidecars(self, parent_fd: int) -> None:
        access = os.O_RDONLY if self._read_only else os.O_RDWR
        flags = access | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        for suffix in self._DATABASE_SIDECAR_SUFFIXES:
            name = f"{self.path.name}{suffix}"
            try:
                descriptor = os.open(name, flags, dir_fd=parent_fd)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RuntimeError("Remote registry sidecar is unavailable or unsafe") from exc
            try:
                metadata = os.fstat(descriptor)
                entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                    or (metadata.st_dev, metadata.st_ino)
                    != (entry.st_dev, entry.st_ino)
                ):
                    raise RuntimeError(
                        "Remote registry sidecars must be singly-linked regular files "
                        "owned by the service user"
                    )
                if self._read_only:
                    raise RuntimeError(
                        "Read-only registry access requires a checkpointed database "
                        "without sidecars"
                    )
                else:
                    os.fchmod(descriptor, 0o600)
                    if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
                        raise RuntimeError("Remote registry sidecars must use mode 0600")
            finally:
                os.close(descriptor)

    def _validated_database_fd(self) -> tuple[int, int]:
        self._assert_controller()
        parent_fd = self._open_parent_fd()
        try:
            self._validate_sidecars(parent_fd)
            database_fd = self._open_database_fd(parent_fd, create=False)
            try:
                metadata = self._validate_database_fd(parent_fd, database_fd)
                if (metadata.st_dev, metadata.st_ino) != self._database_identity:
                    raise RuntimeError("Remote registry database identity changed")
                self._assert_controller()
                return parent_fd, database_fd
            except Exception:
                os.close(database_fd)
                raise
        except Exception:
            os.close(parent_fd)
            raise

    def _assert_operation_current(self) -> None:
        parent_fd, database_fd = self._validated_database_fd()
        os.close(database_fd)
        os.close(parent_fd)

    def _connect(self) -> sqlite3.Connection:
        parent_fd, database_fd = self._validated_database_fd()
        conn: sqlite3.Connection | None = None
        try:
            mode = "ro&immutable=1" if self._read_only else "rw"
            conn = sqlite3.connect(
                f"file:/proc/self/fd/{database_fd}?mode={mode}",
                timeout=10,
                isolation_level=None,
                factory=_ClosingConnection,
                uri=True,
            )
            conn._operation_guard = self._assert_operation_current
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=10000")
            if self._read_only:
                conn.execute("PRAGMA query_only=ON")
            # SQLite now owns a separate descriptor for the exact database inode.
            # Revalidate every lexical identity before exposing the connection.
            self._validate_sidecars(parent_fd)
            metadata = self._validate_database_fd(parent_fd, database_fd)
            if (metadata.st_dev, metadata.st_ino) != self._database_identity:
                raise RuntimeError("Remote registry database identity changed")
            if self._validate_parent_fd(parent_fd) != self._parent_identity:
                raise RuntimeError("Remote registry directory identity changed")
            self._assert_controller()
            return conn
        except Exception:
            if conn is not None:
                conn.close()
            raise
        finally:
            os.close(database_fd)
            os.close(parent_fd)

    @contextmanager
    def _schema_file_lock(self):
        """Serialize schema initialization across Nexus/standalone processes."""

        name = f".{self.path.name}.initialize.lock"
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        parent_fd = self._open_parent_fd()
        fd = -1
        try:
            fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
            metadata = os.fstat(fd)
            entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
                or (metadata.st_dev, metadata.st_ino)
                != (entry.st_dev, entry.st_ino)
            ):
                raise RuntimeError("Remote registry initialization lock is unsafe")
            fcntl.flock(fd, fcntl.LOCK_EX)
            entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (metadata.st_dev, metadata.st_ino) != (entry.st_dev, entry.st_ino):
                raise RuntimeError("Remote registry initialization lock changed")
            yield
        finally:
            try:
                if fd >= 0:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                if fd >= 0:
                    os.close(fd)
                os.close(parent_fd)

    def _initialize(self) -> None:
        with self._schema_file_lock(), self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.executescript(SCHEMA)
            # The remote console predates ordinary-CLI adoption. Keep existing
            # private registries usable without treating their web-created rows
            # as locally authorized workspace exceptions.
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Inspect only after the write lock is held. Standalone Aeon
                # Remote and Nexus may initialize the same registry together;
                # neither may act on a stale pre-lock column snapshot.
                instance_columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(instances)")
                }
                login_attempt_columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(login_attempts)")
                }
                if "attempt_id" not in login_attempt_columns:
                    conn.execute("ALTER TABLE login_attempts ADD COLUMN attempt_id TEXT")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS login_attempts_attempt_id "
                    "ON login_attempts(attempt_id)"
                )
                if "launch_origin" not in instance_columns:
                    conn.execute(
                        "ALTER TABLE instances ADD COLUMN launch_origin TEXT "
                        "NOT NULL DEFAULT 'web'"
                    )
                if "host_id" not in instance_columns:
                    # Every pre-fleet terminal and agent was launched locally on
                    # the orchestrator.  Preserve that identity explicitly; a
                    # missing value must never be interpreted as an arbitrary
                    # worker selected by the browser.
                    conn.execute(
                        "ALTER TABLE instances ADD COLUMN host_id TEXT "
                        "NOT NULL DEFAULT '192.168.0.177'"
                    )
                if "kind" not in instance_columns:
                    # Every registry row created before terminal tabs existed is an
                    # Aeon session, including locally adopted and browser-created
                    # rows. SQLite applies this non-null default to existing rows.
                    conn.execute(
                        "ALTER TABLE instances ADD COLUMN kind TEXT "
                        "NOT NULL DEFAULT 'aeon'"
                    )
                if "shell_backed" not in instance_columns:
                    conn.execute(
                        "ALTER TABLE instances ADD COLUMN shell_backed INTEGER "
                        "NOT NULL DEFAULT 0"
                    )
                if "last_agent_kind" not in instance_columns:
                    conn.execute(
                        "ALTER TABLE instances ADD COLUMN last_agent_kind TEXT"
                    )
                if "transport_pid" not in instance_columns:
                    conn.execute(
                        "ALTER TABLE instances ADD COLUMN transport_pid INTEGER"
                    )
                if "transport_process_create_time" not in instance_columns:
                    conn.execute(
                        "ALTER TABLE instances ADD COLUMN "
                        "transport_process_create_time REAL"
                    )
                # These repairs are intentionally idempotent and run on every
                # initialization. A process crash after ALTER but before the old
                # conditional UPDATE must not strand terminal rows as direct
                # processes forever.
                conn.execute(
                    "UPDATE instances SET shell_backed=1 WHERE kind='terminal'"
                )
                conn.execute(
                    "UPDATE instances SET last_agent_kind=kind "
                    "WHERE shell_backed=1 AND last_agent_kind IS NULL "
                    "AND kind IN ('aeon','codex','claude','grok')"
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _dict(row):
        return dict(row) if row is not None else None

    def admin_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users WHERE enabled=1").fetchone()[0])

    def put_user(
        self, username: str, password_hash: str, totp_secret: str, *, replace: bool = False
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE username=? COLLATE NOCASE", (username,)
            ).fetchone()
            if existing and not replace:
                raise ValueError(f"User already exists: {username}")
            conn.execute("BEGIN IMMEDIATE")
            if existing:
                conn.execute(
                    "UPDATE users SET username=?,password_hash=?,totp_secret=?,"
                    "enabled=1,updated_at=? WHERE id=?",
                    (username, password_hash, totp_secret, now, existing["id"]),
                )
                conn.execute("DELETE FROM web_sessions WHERE user_id=?", (existing["id"],))
            else:
                conn.execute(
                    "INSERT INTO users(username,password_hash,totp_secret,created_at,updated_at) "
                    "VALUES(?,?,?,?,?)",
                    (username, password_hash, totp_secret, now, now),
                )
            conn.execute("COMMIT")

    def get_user(self, username: str):
        with self._connect() as conn:
            return self._dict(
                conn.execute(
                    "SELECT * FROM users WHERE username=? COLLATE NOCASE", (username,)
                ).fetchone()
            )

    def create_web_session(
        self,
        user_id: int,
        token_hash: str,
        csrf_token: str,
        expires_at: float,
        user_agent_hash: str,
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO web_sessions(token_hash,user_id,csrf_token,created_at,expires_at,"
                "last_seen,user_agent_hash) VALUES(?,?,?,?,?,?,?)",
                (token_hash, user_id, csrf_token, now, expires_at, now, user_agent_hash),
            )

    def get_web_session(self, token_hash: str):
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT s.*,u.username FROM web_sessions s JOIN users u ON u.id=s.user_id "
                "WHERE s.token_hash=? AND s.revoked=0 AND s.expires_at>? AND u.enabled=1",
                (token_hash, now),
            ).fetchone()
            if not row:
                return None
            if now - row["last_seen"] > 60:
                conn.execute(
                    "UPDATE web_sessions SET last_seen=? WHERE token_hash=?", (now, token_hash)
                )
            return dict(row)

    def revoke_web_session(self, token_hash: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE web_sessions SET revoked=1 WHERE token_hash=?", (token_hash,))

    def login_blocked(self, rate_key: str, *, window_seconds=900, max_failures=5) -> bool:
        cutoff = time.time() - window_seconds
        with self._connect() as conn:
            failures = conn.execute(
                "SELECT COUNT(*) FROM login_attempts WHERE rate_key=? AND succeeded=0 "
                "AND attempted_at>?",
                (rate_key, cutoff),
            ).fetchone()[0]
            return failures >= max_failures

    def reserve_login_attempt(
        self,
        rate_limits: dict[str, int],
        *,
        window_seconds: int = 900,
        retention_seconds: int = 86400,
        max_rows: int = 4096,
    ) -> str | None:
        """Atomically reserve one expensive authentication attempt.

        Each key is already a domain-separated digest, so the registry never
        receives a username or network address.  A provisional failed row is
        inserted before Argon2 runs; a crash therefore consumes capacity rather
        than bypassing the limiter.  ``None`` means at least one bucket is full.
        """

        limits = {
            str(key): int(limit)
            for key, limit in rate_limits.items()
            if str(key) and int(limit) > 0
        }
        if not limits:
            raise ValueError("At least one positive login rate limit is required")
        if max_rows < len(limits):
            raise ValueError("Login-attempt row cap is too small")

        now = time.time()
        cutoff = now - window_seconds
        retention_cutoff = now - retention_seconds
        attempt_id = secrets.token_hex(16)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "DELETE FROM login_attempts WHERE attempted_at<?",
                    (retention_cutoff,),
                )
                # Rows outside the active decision window are not limiter
                # evidence. Reclaim them before applying the hard bound; never
                # evict a still-active reservation to admit another attacker.
                conn.execute(
                    "DELETE FROM login_attempts WHERE attempted_at<=?",
                    (cutoff,),
                )
                for rate_key, limit in limits.items():
                    failures = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM login_attempts "
                            "WHERE rate_key=? AND succeeded=0 AND attempted_at>?",
                            (rate_key, cutoff),
                        ).fetchone()[0]
                    )
                    if failures >= limit:
                        conn.execute("ROLLBACK")
                        return None
                row_count = int(
                    conn.execute("SELECT COUNT(*) FROM login_attempts").fetchone()[0]
                )
                if row_count + len(limits) > max_rows:
                    conn.execute("ROLLBACK")
                    return None
                conn.executemany(
                    "INSERT INTO login_attempts"
                    "(rate_key,attempted_at,succeeded,attempt_id) VALUES(?,?,0,?)",
                    ((key, now, attempt_id) for key in limits),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return attempt_id

    def complete_login_attempt(
        self,
        attempt_id: str,
        *,
        succeeded: bool,
        clear_rate_keys: tuple[str, ...] = (),
    ) -> None:
        """Complete a reserved attempt without adding a second failure row."""

        if not attempt_id:
            raise ValueError("Login attempt id is required")
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if succeeded:
                    keys = tuple(dict.fromkeys(clear_rate_keys))
                    if keys:
                        placeholders = ",".join("?" for _ in keys)
                        conn.execute(
                            f"DELETE FROM login_attempts WHERE rate_key IN ({placeholders})",
                            keys,
                        )
                    else:
                        conn.execute(
                            "DELETE FROM login_attempts WHERE attempt_id=?",
                            (attempt_id,),
                        )
                # Failed attempts are already recorded provisionally.  Leaving
                # them untouched is both the result and the crash-safe behavior.
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def record_login_attempt(self, rate_key: str, succeeded: bool) -> None:
        cutoff = time.time() - 86400
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO login_attempts(rate_key,attempted_at,succeeded) VALUES(?,?,?)",
                (rate_key, time.time(), int(succeeded)),
            )
            conn.execute("DELETE FROM login_attempts WHERE attempted_at<?", (cutoff,))

    def clear_login_attempts(self, rate_key: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM login_attempts WHERE rate_key=?", (rate_key,))

    def audit(
        self,
        action: str,
        *,
        actor: str,
        instance_id: str | None = None,
        client_ip: str = "",
        details: dict | None = None,
    ) -> None:
        # Never pass terminal input/output, credentials, or cookies in details.
        payload = json.dumps(details or {}, separators=(",", ":"), default=str)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO audit_log(occurred_at,actor,action,instance_id,client_ip,details_json) "
                    "VALUES(?,?,?,?,?,?)",
                    (time.time(), actor, action, instance_id, client_ip, payload),
                )
                if action == "login_failed":
                    # This event is unauthenticated and attacker-triggerable.
                    # Retain a useful bounded trail without permitting durable
                    # database growth or eviction of authenticated lifecycle
                    # events.
                    conn.execute(
                        "DELETE FROM audit_log WHERE action='login_failed' AND id NOT IN "
                        "(SELECT id FROM audit_log WHERE action='login_failed' "
                        "ORDER BY id DESC LIMIT 2048)"
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def create_instance(self, record: dict) -> None:
        fields = (
            "id", "host_id", "kind", "shell_backed", "last_agent_kind", "name",
            "tmux_name", "workspace", "objective",
            "max_iterations", "model", "status", "desired_state", "created_at",
            "updated_at", "last_started_at", "transport_pid",
            "transport_process_create_time", "last_error", "created_by", "launch_origin",
        )
        defaults = {
            "host_id": "192.168.0.177",
            "kind": "aeon",
            "shell_backed": 0,
        }
        values = tuple(record.get(field, defaults.get(field)) for field in fields)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT INTO instances({','.join(fields)}) "
                f"VALUES({','.join('?' for _ in fields)})",
                values,
            )

    def list_instances(self) -> list[dict]:
        with self._connect() as conn:
            return [
                dict(row)
                for row in conn.execute("SELECT * FROM instances ORDER BY created_at")
            ]

    def get_instance(self, instance_id: str):
        with self._connect() as conn:
            return self._dict(
                conn.execute("SELECT * FROM instances WHERE id=?", (instance_id,)).fetchone()
            )

    def update_instance(self, instance_id: str, **updates) -> None:
        allowed = {
            "name", "tmux_name", "status", "desired_state", "updated_at", "last_started_at",
            "last_error", "objective", "max_iterations", "model", "kind",
            "workspace", "shell_backed", "last_agent_kind", "transport_pid",
            "transport_process_create_time",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        if not values:
            return
        values.setdefault("updated_at", time.time())
        assignment = ",".join(f"{key}=?" for key in values)
        with self._lock, self._connect() as conn:
            conn.execute(
                f"UPDATE instances SET {assignment} WHERE id=?",
                (*values.values(), instance_id),
            )

    @staticmethod
    def _agent_setting_dict(row, *, instance_id: str, agent_kind: str) -> dict:
        """Return one validated desired/applied launch setting record.

        Missing rows are represented by the reviewed provider defaults without
        creating database state during a read.  A malformed row fails closed so
        it can never become launcher input merely because it came from SQLite.
        """

        catalog = catalog_for(agent_kind)
        if row is None:
            desired_model, desired_effort = normalize_settings(
                agent_kind,
                model=catalog.default_model,
                effort=catalog.default_effort,
            )
            return {
                "instance_id": instance_id,
                "agent_kind": agent_kind,
                "desired_model": desired_model,
                "desired_effort": desired_effort,
                "applied_model": None,
                "applied_effort": None,
                "updated_at": None,
                "applied_at": None,
            }

        value = dict(row)
        desired_model, desired_effort = normalize_settings(
            agent_kind,
            model=value.get("desired_model"),
            effort=value.get("desired_effort"),
        )
        applied_model = value.get("applied_model")
        applied_effort = value.get("applied_effort")
        if (applied_model is None) != (applied_effort is None):
            raise ValueError("Agent launch setting has incomplete applied state")
        if applied_model is not None:
            applied_model, applied_effort = normalize_settings(
                agent_kind,
                model=applied_model,
                effort=applied_effort,
            )
        value.update(
            {
                "desired_model": desired_model,
                "desired_effort": desired_effort,
                "applied_model": applied_model,
                "applied_effort": applied_effort,
            }
        )
        return value

    def get_agent_setting(self, instance_id: str, agent_kind: str) -> dict:
        catalog_for(agent_kind)
        with self._connect() as conn:
            if conn.execute(
                "SELECT 1 FROM instances WHERE id=?", (instance_id,)
            ).fetchone() is None:
                raise ValueError("Unknown session")
            row = conn.execute(
                "SELECT * FROM instance_agent_settings "
                "WHERE instance_id=? AND agent_kind=?",
                (instance_id, agent_kind),
            ).fetchone()
        return self._agent_setting_dict(
            row,
            instance_id=instance_id,
            agent_kind=agent_kind,
        )

    def list_agent_settings(self, instance_id: str) -> dict[str, dict]:
        with self._connect() as conn:
            if conn.execute(
                "SELECT 1 FROM instances WHERE id=?", (instance_id,)
            ).fetchone() is None:
                raise ValueError("Unknown session")
            rows = {
                row["agent_kind"]: row
                for row in conn.execute(
                    "SELECT * FROM instance_agent_settings WHERE instance_id=?",
                    (instance_id,),
                )
            }
        return {
            kind: self._agent_setting_dict(
                rows.get(kind),
                instance_id=instance_id,
                agent_kind=kind,
            )
            for kind in ("aeon", "codex", "claude", "grok")
        }

    def put_agent_setting(
        self,
        instance_id: str,
        agent_kind: str,
        *,
        model: str | None,
        effort: str | None,
    ) -> dict:
        desired_model, desired_effort = normalize_settings(
            agent_kind,
            model=model,
            effort=effort,
        )
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if conn.execute(
                    "SELECT 1 FROM instances WHERE id=?", (instance_id,)
                ).fetchone() is None:
                    raise ValueError("Unknown session")
                conn.execute(
                    "INSERT INTO instance_agent_settings("
                    "instance_id,agent_kind,desired_model,desired_effort,"
                    "applied_model,applied_effort,updated_at,applied_at) "
                    "VALUES(?,?,?,?,NULL,NULL,?,NULL) "
                    "ON CONFLICT(instance_id,agent_kind) DO UPDATE SET "
                    "desired_model=excluded.desired_model,"
                    "desired_effort=excluded.desired_effort,"
                    "updated_at=excluded.updated_at",
                    (
                        instance_id,
                        agent_kind,
                        desired_model,
                        desired_effort,
                        now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM instance_agent_settings "
                    "WHERE instance_id=? AND agent_kind=?",
                    (instance_id, agent_kind),
                ).fetchone()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return self._agent_setting_dict(
            row,
            instance_id=instance_id,
            agent_kind=agent_kind,
        )

    def mark_agent_setting_applied(
        self,
        instance_id: str,
        agent_kind: str,
        *,
        model: str | None,
        effort: str | None,
    ) -> dict:
        """Record only a launcher-verified immutable settings snapshot.

        The conflict branch deliberately leaves the current desired values
        untouched.  If a newer preference is saved after launch preparation,
        the just-started process remains truthfully represented as pending.
        """

        applied_model, applied_effort = normalize_settings(
            agent_kind,
            model=model,
            effort=effort,
        )
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if conn.execute(
                    "SELECT 1 FROM instances WHERE id=?", (instance_id,)
                ).fetchone() is None:
                    raise ValueError("Unknown session")
                conn.execute(
                    "INSERT INTO instance_agent_settings("
                    "instance_id,agent_kind,desired_model,desired_effort,"
                    "applied_model,applied_effort,updated_at,applied_at) "
                    "VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(instance_id,agent_kind) DO UPDATE SET "
                    "applied_model=excluded.applied_model,"
                    "applied_effort=excluded.applied_effort,"
                    "applied_at=excluded.applied_at",
                    (
                        instance_id,
                        agent_kind,
                        applied_model,
                        applied_effort,
                        applied_model,
                        applied_effort,
                        now,
                        now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM instance_agent_settings "
                    "WHERE instance_id=? AND agent_kind=?",
                    (instance_id, agent_kind),
                ).fetchone()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return self._agent_setting_dict(
            row,
            instance_id=instance_id,
            agent_kind=agent_kind,
        )

    def transition_shell_mode(
        self,
        instance_id: str,
        *,
        expected_kind: str,
        kind: str,
        workspace: str | None = None,
        last_agent_kind: str | None = None,
        clear_profile: bool = False,
        status: str = "running",
        last_error: str = "",
    ) -> dict:
        """Atomically change one shell-backed tab's foreground mode.

        The caller resolves every executable and workspace.  This transaction
        only prevents two browser actions from racing the same tab and, when
        switching between different agent families, drops an incompatible base
        profile while retaining the tab's persistent local-role revisions.
        """

        last_error = str(last_error or "")[:500]
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM instances WHERE id=?", (instance_id,)
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise ValueError("Unknown session")
            if row["kind"] != expected_kind or int(row["shell_backed"] or 0) != 1:
                conn.execute("ROLLBACK")
                raise ValueError("Session mode changed concurrently")
            updates = {
                "kind": kind,
                "status": status,
                "desired_state": "running",
                "updated_at": now,
                "last_error": last_error,
            }
            if workspace is not None:
                updates["workspace"] = workspace
            if last_agent_kind is not None:
                updates["last_agent_kind"] = last_agent_kind
            assignment = ",".join(f"{key}=?" for key in updates)
            conn.execute(
                f"UPDATE instances SET {assignment} WHERE id=?",
                (*updates.values(), instance_id),
            )
            if clear_profile:
                conn.execute(
                    "UPDATE instance_instruction_bindings SET "
                    "desired_profile_version_id=NULL,"
                    "applied_profile_version_id=NULL,updated_at=? "
                    "WHERE instance_id=?",
                    (now, instance_id),
                )
            result = conn.execute(
                "SELECT * FROM instances WHERE id=?", (instance_id,)
            ).fetchone()
            conn.execute("COMMIT")
            return dict(result)

    def delete_instance(self, instance_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM instances WHERE id=?", (instance_id,))

    def recent_workspaces(self) -> list[str]:
        with self._connect() as conn:
            return [
                row[0]
                for row in conn.execute(
                    "SELECT workspace FROM instances GROUP BY workspace "
                    "ORDER BY MAX(updated_at) DESC LIMIT 50"
                )
            ]

    def recent_audit(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            )
            return [dict(row) for row in rows]
