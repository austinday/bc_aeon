"""Small, local presence records for running Aeon processes.

The files written here are deliberately status-only.  They are safe for a local
control plane to discover, but they never contain conversation history, prompts,
tool parameters, tool output, environment variables, or command lines.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import socket
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import psutil


SCHEMA_VERSION = 1
REMOTE_INSTANCE_ENV = "AEON_REMOTE_INSTANCE_ID"

_UUID_RE = re.compile(
    r"^(?:[0-9a-fA-F]{32}|"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_URL_RE = re.compile(r"(?i)\b(?:https?|wss?)://[^\s<>'\"]+")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:api[\s_-]*key|access[\s_-]*key|auth(?:orization)?|cookie|"
    r"credential|otp|passphrase|password|passwd|private[\s_-]*key|secret|"
    r"session[\s_-]*id|token)\b"
    r"\s*(?:=|:)\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_KNOWN_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[oprsu]_[A-Za-z0-9]{20,}|"
    r"AKIA[0-9A-Z]{16})\b"
)
_LONG_OPAQUE_RE = re.compile(r"\b[A-Za-z0-9_+/=-]{40,}\b")
_PHASE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_COMPUTE_STATES = frozenset({"idle", "waiting_for_compute", "allocated", "unavailable"})
_UNSET = object()

_identity_lock = threading.Lock()
_identity_pid: Optional[int] = None
_fallback_instance_id: Optional[str] = None
_cached_identity: Optional[dict[str, Optional[str]]] = None
_active_presence: Optional["Presence"] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def validate_remote_instance_id(value: Any) -> Optional[str]:
    """Return a filename-safe canonical UUID hex string, or ``None``.

    UUID's permissive parser accepts values such as braces and URNs.  The remote
    identifier crosses a process boundary and is later used in a filename, so we
    intentionally accept only the ordinary 32-hex and hyphenated UUID forms.
    """

    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not _UUID_RE.fullmatch(candidate):
        return None
    try:
        parsed = uuid.UUID(candidate)
    except (ValueError, AttributeError):
        return None
    if parsed.int == 0:
        return None
    return parsed.hex


def _fallback_process_id() -> str:
    global _identity_pid, _fallback_instance_id, _cached_identity
    pid = os.getpid()
    with _identity_lock:
        # Regenerate after fork: a copied module global must not give two
        # processes the same fallback identity.
        if _identity_pid != pid or _fallback_instance_id is None:
            _identity_pid = pid
            _fallback_instance_id = uuid.uuid4().hex
            _cached_identity = None
        return _fallback_instance_id


def _resolve_identity(environ: Mapping[str, str]) -> dict[str, Optional[str]]:
    remote_id = validate_remote_instance_id(environ.get(REMOTE_INSTANCE_ENV))
    if remote_id:
        return {
            "instance_id": remote_id,
            "remote_instance_id": remote_id,
            "launch_origin": "remote",
        }
    return {
        "instance_id": _fallback_process_id(),
        "remote_instance_id": None,
        "launch_origin": "local",
    }


def process_identity() -> dict[str, Optional[str]]:
    """Return the launch identity, cached for the lifetime of this process."""

    global _identity_pid, _cached_identity
    pid = os.getpid()
    with _identity_lock:
        if _identity_pid == pid and _cached_identity is not None:
            return dict(_cached_identity)

    # _resolve_identity may take the same lock while creating a fallback UUID.
    resolved = _resolve_identity(os.environ)
    with _identity_lock:
        _identity_pid = pid
        _cached_identity = dict(resolved)
        return dict(resolved)


def process_instance_id() -> str:
    """Stable instance identifier used by Worker and its presence record."""

    return str(process_identity()["instance_id"])


def get_active_presence() -> Optional["Presence"]:
    """Return this process's latest registered presence writer, if any."""

    presence = _active_presence
    if (
        presence is not None
        and presence.pid == os.getpid()
        and not presence._exited
    ):
        return presence
    return None


def sanitize_summary(value: Any, *, max_chars: int = 240) -> str:
    """Create a bounded, single-line status summary with common secrets removed."""

    if value is None:
        return ""
    text = _CONTROL_RE.sub(" ", str(value))
    if "PRIVATE KEY-----" in text.upper():
        return "[redacted sensitive content]"
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    text = _SECRET_ASSIGNMENT_RE.sub("[redacted credential]", text)
    text = _KNOWN_TOKEN_RE.sub("[redacted credential]", text)
    text = _LONG_OPAQUE_RE.sub("[redacted opaque value]", text)
    text = _URL_RE.sub("[url]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 1)].rstrip() + "…"
    return text


def manifest_process_is_live(manifest: Mapping[str, Any]) -> bool:
    """Verify that a manifest still names the exact live OS process.

    A PID alone is unsafe because the kernel can reuse it.  Comparing psutil's
    process creation time makes old manifests unambiguously stale without deleting
    their useful historical status.
    """

    if manifest.get("phase") == "exited":
        return False
    try:
        pid = int(manifest["pid"])
        expected = float(manifest["process_create_time"])
        if pid <= 0 or expected <= 0:
            return False
        actual = float(psutil.Process(pid).create_time())
    except (KeyError, TypeError, ValueError, psutil.Error, OSError):
        return False
    return abs(actual - expected) <= 0.02


class Presence:
    """Own one unique, atomically updated presence manifest for an Aeon run."""

    def __init__(
        self,
        *,
        presence_dir: Optional[Path] = None,
        cwd: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
        register_atexit: bool = True,
    ) -> None:
        global _active_presence
        identity = process_identity() if environ is None else _resolve_identity(environ)
        self.instance_id = str(identity["instance_id"])
        self.remote_instance_id = identity["remote_instance_id"]
        self.launch_origin = str(identity["launch_origin"])
        self.run_id = uuid.uuid4().hex
        self.pid = os.getpid()
        self.process_create_time = float(psutil.Process(self.pid).create_time())
        self._lock = threading.RLock()
        self._exited = False

        if presence_dir is None:
            presence_dir = Path.home() / ".aeon" / "remote" / "presence"
        self.presence_dir = Path(presence_dir).expanduser()
        self._prepare_directory()
        self.path = self.presence_dir / (
            f"presence-{self.instance_id}-{self.run_id}.json"
        )

        try:
            working_directory = str(Path(cwd if cwd is not None else os.getcwd()).resolve())
        except (OSError, RuntimeError):
            working_directory = "[unavailable]"
        now = _utc_now()
        self._record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "instance_id": self.instance_id,
            "remote_instance_id": self.remote_instance_id,
            "launch_origin": self.launch_origin,
            "hostname": socket.gethostname(),
            "cwd": working_directory,
            "pid": self.pid,
            "process_create_time": self.process_create_time,
            "phase": "startup",
            "iteration": 0,
            "objective_summary": "",
            "intent_summary": "",
            "current_plan_summary": "",
            "model": "",
            "started_at": now,
            "updated_at": now,
            "objective_started_at": None,
            "completed_at": None,
            "error_at": None,
            "error_type": None,
            "exited_at": None,
            "compute_state": "idle",
            "compute_profile": "",
            "compute_status_summary": "",
            "compute_wait_started_at": None,
            "compute_updated_at": now,
        }
        self._write_atomic()
        _active_presence = self
        if register_atexit:
            atexit.register(self.mark_exit)

    def _prepare_directory(self) -> None:
        self.presence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.presence_dir.is_symlink() or not self.presence_dir.is_dir():
            raise OSError(f"Presence path is not a private directory: {self.presence_dir}")
        os.chmod(self.presence_dir, 0o700)

    def _write_atomic(self) -> None:
        temp_path: Optional[str] = None
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=".presence-", suffix=".tmp", dir=str(self.presence_dir)
            )
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._record, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
            temp_path = None
            os.chmod(self.path, 0o600)
            try:
                dir_fd = os.open(self.presence_dir, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                # The rename is still atomic on filesystems that cannot fsync a
                # directory; durability across sudden power loss is best-effort.
                pass
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    @property
    def manifest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._record)

    def update(
        self,
        *,
        phase: Any = _UNSET,
        iteration: Any = _UNSET,
        objective: Any = _UNSET,
        intent: Any = _UNSET,
        current_plan: Any = _UNSET,
        model: Any = _UNSET,
    ) -> None:
        with self._lock:
            if phase is not _UNSET:
                normalized_phase = str(phase).strip().lower().replace(" ", "_")
                normalized_phase = (
                    normalized_phase if _PHASE_RE.fullmatch(normalized_phase) else "unknown"
                )
                self._record["phase"] = normalized_phase
                # An exec attempt can fail after mark_exit().  Any subsequent
                # real lifecycle update revives this same run truthfully.
                if normalized_phase != "exited" and self._exited:
                    self._exited = False
                    self._record["exited_at"] = None
            if iteration is not _UNSET:
                try:
                    self._record["iteration"] = max(0, int(iteration))
                except (TypeError, ValueError):
                    pass
            if objective is not _UNSET:
                self._record["objective_summary"] = sanitize_summary(
                    objective, max_chars=280
                )
            if intent is not _UNSET:
                self._record["intent_summary"] = sanitize_summary(intent, max_chars=240)
            if current_plan is not _UNSET:
                self._record["current_plan_summary"] = sanitize_summary(
                    current_plan, max_chars=360
                )
            if model is not _UNSET:
                self._record["model"] = sanitize_summary(model, max_chars=160)
            self._record["updated_at"] = _utc_now()
            self._write_atomic()

    def start_objective(self, objective: Any, *, model: Any = _UNSET) -> None:
        with self._lock:
            now = _utc_now()
            self._record["objective_started_at"] = now
            self._record["completed_at"] = None
            self._record["error_at"] = None
            self._record["error_type"] = None
            self._record["intent_summary"] = ""
            self.update(
                phase="objective",
                iteration=0,
                objective=objective,
                current_plan="",
                model=model,
            )

    def mark_completed(self, *, current_plan: Any = _UNSET) -> None:
        with self._lock:
            self._record["completed_at"] = _utc_now()
            self.update(phase="completed", current_plan=current_plan)

    def mark_error(self, error: BaseException) -> None:
        with self._lock:
            # Exception messages can embed prompts, tool output, URLs, or tokens.
            # The class is enough to make the failure diagnosable without copying
            # any of that content into the control plane.
            self._record["error_at"] = _utc_now()
            self._record["error_type"] = sanitize_summary(
                type(error).__name__, max_chars=80
            )
            self.update(phase="error")

    def update_compute(self, *, state: Any, profile: Any, summary: Any) -> None:
        """Record a bounded compute admission state without lease identifiers."""

        normalized_state = str(state).strip().lower()
        if normalized_state not in _COMPUTE_STATES:
            normalized_state = "unavailable"
        with self._lock:
            now = _utc_now()
            previous = self._record.get("compute_state")
            self._record["compute_state"] = normalized_state
            self._record["compute_profile"] = sanitize_summary(profile, max_chars=80)
            self._record["compute_status_summary"] = sanitize_summary(
                summary, max_chars=200
            )
            if normalized_state == "waiting_for_compute":
                if previous != "waiting_for_compute":
                    self._record["compute_wait_started_at"] = now
            else:
                self._record["compute_wait_started_at"] = None
            self._record["compute_updated_at"] = now
            self._record["updated_at"] = now
            self._write_atomic()

    def mark_exit(self) -> None:
        with self._lock:
            if self._exited:
                return
            self._exited = True
            self._record["exited_at"] = _utc_now()
            try:
                self.update(phase="exited")
            except Exception:
                # atexit callbacks must never obscure the process's real result.
                pass
