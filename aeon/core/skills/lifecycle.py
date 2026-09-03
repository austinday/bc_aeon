"""Revision-bound lifecycle state for one agent's learned skills.

The protocol text remains the portable skill artifact.  This sibling store records
why an agent was allowed to create it and what happened when it was reused.  The
metadata is deliberately advisory: it cannot grant tool, request, or filesystem
authority, and a content digest mismatch makes the record stale rather than
silently blessing edited instructions.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from aeon.core.skills.knowledge import contains_persisted_secret
from aeon.core.utils.io import read_bounded_fd

MAX_PRIVATE_SKILLS = 16
MAX_LIFECYCLE_BYTES = 64 * 1024
SKILL_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
REVISION_RE = re.compile(r"^[0-9a-f]{64}$")
VALID_STATUSES = frozenset({"ready", "needs_review", "quarantined"})
VALID_OUTCOMES = frozenset({"success", "adapted", "failed", "not_applicable"})


class LearnedSkillError(ValueError):
    """Learned-skill metadata failed its private integrity contract."""


def skill_revision(content: str) -> str:
    return hashlib.sha256(str(content).encode("utf-8")).hexdigest()


def _component(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not SKILL_COMPONENT_RE.fullmatch(text):
        raise LearnedSkillError(f"{label} is invalid")
    return text


def _bounded_note(value: object, *, maximum: int = 2000) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\x00" in text or len(text.encode("utf-8")) > maximum:
        raise LearnedSkillError(f"skill outcome note exceeds the {maximum}-byte limit")
    if text and contains_persisted_secret(text):
        raise LearnedSkillError(
            "secret-like credentials cannot be stored in skill lifecycle notes"
        )
    return text


class LearnedSkillStore:
    """Crash-safe metadata stored below one exact private skill overlay."""

    VERSION = 1

    def __init__(self, instance_dir: str | os.PathLike[str] | None):
        # Keep lexical identity so a symlink cannot disappear through resolve().
        self.instance_dir = (
            Path(instance_dir).expanduser().absolute() if instance_dir else None
        )
        self.root = self.instance_dir / ".skill-state" if self.instance_dir else None

    def _ensure_directory(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, mode=0o700, exist_ok=True)
            metadata = path.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or path.resolve(strict=True) != path.absolute()
            ):
                raise LearnedSkillError("learned-skill storage is not private")
            os.chmod(path, 0o700, follow_symlinks=False)
        except LearnedSkillError:
            raise
        except OSError as exc:
            raise LearnedSkillError("learned-skill storage is unavailable") from exc

    def _record_path(
        self, category: object, skill_name: object, *, create: bool
    ) -> Path:
        if self.root is None:
            raise LearnedSkillError(
                "learned skills require an agent-specific private skill directory"
            )
        category_value = _component(category, label="skill category")
        name_value = _component(skill_name, label="skill name")
        if create:
            self._ensure_directory(self.root)
            self._ensure_directory(self.root / category_value)
        return self.root / category_value / f"{name_value}.json"

    @contextmanager
    def _exclusive_lock(self):
        """Serialize lifecycle read-modify-write operations across processes."""

        if self.root is None:
            raise LearnedSkillError(
                "learned skills require an agent-specific private skill directory"
            )
        self._ensure_directory(self.root)
        root_fd = None
        lock_fd = None
        try:
            root_metadata = self.root.lstat()
            root_fd = os.open(
                self.root,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened_root = os.fstat(root_fd)
            if (
                opened_root.st_dev != root_metadata.st_dev
                or opened_root.st_ino != root_metadata.st_ino
            ):
                raise LearnedSkillError(
                    "learned-skill storage changed while opening"
                )
            lock_fd = os.open(
                ".lifecycle.lock",
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=root_fd,
            )
            os.fchmod(lock_fd, 0o600)
            lock_metadata = os.fstat(lock_fd)
            if (
                not stat.S_ISREG(lock_metadata.st_mode)
                or lock_metadata.st_uid != os.geteuid()
                or lock_metadata.st_nlink != 1
            ):
                raise LearnedSkillError("learned-skill lock is not private")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        except LearnedSkillError:
            raise
        except OSError as exc:
            raise LearnedSkillError("learned-skill lock is unavailable") from exc
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)
            if root_fd is not None:
                os.close(root_fd)

    @staticmethod
    def _validate_document(document: object, *, expected_path: str) -> dict[str, Any]:
        if not isinstance(document, dict) or document.get("version") != 1:
            raise LearnedSkillError("learned-skill metadata version is invalid")
        required = {
            "version",
            "skill_path",
            "content_revision",
            "status",
            "evidence",
            "created_at",
            "updated_at",
            "revision_number",
            "usage",
            "last_outcome",
        }
        if set(document) != required or document.get("skill_path") != expected_path:
            raise LearnedSkillError("learned-skill metadata identity is invalid")
        if not REVISION_RE.fullmatch(str(document.get("content_revision") or "")):
            raise LearnedSkillError("learned-skill content revision is invalid")
        if document.get("status") not in VALID_STATUSES:
            raise LearnedSkillError("learned-skill status is invalid")
        evidence = document.get("evidence")
        if not isinstance(evidence, list) or not evidence or len(evidence) > 16:
            raise LearnedSkillError("learned-skill evidence references are invalid")
        for item in evidence:
            if (
                not isinstance(item, dict)
                or set(item) != {"note_id", "revision"}
                or not re.fullmatch(r"note-[0-9a-f]{32}", str(item.get("note_id") or ""))
                or not REVISION_RE.fullmatch(str(item.get("revision") or ""))
            ):
                raise LearnedSkillError("learned-skill evidence reference is invalid")
        if (
            not isinstance(document.get("created_at"), (int, float))
            or not isinstance(document.get("updated_at"), (int, float))
            or not isinstance(document.get("revision_number"), int)
            or int(document.get("revision_number") or 0) < 1
        ):
            raise LearnedSkillError("learned-skill timestamps are invalid")
        usage = document.get("usage")
        expected_usage = {
            "activations",
            "successes",
            "adaptations",
            "failures",
            "not_applicable",
        }
        if (
            not isinstance(usage, dict)
            or set(usage) != expected_usage
            or any(not isinstance(usage[key], int) or usage[key] < 0 for key in expected_usage)
        ):
            raise LearnedSkillError("learned-skill usage counters are invalid")
        outcome = document.get("last_outcome")
        if outcome is not None:
            if (
                not isinstance(outcome, dict)
                or set(outcome) != {"outcome", "note", "at", "content_revision"}
                or outcome.get("outcome") not in VALID_OUTCOMES
                or not isinstance(outcome.get("note"), str)
                or len(outcome["note"].encode("utf-8")) > 2000
                or contains_persisted_secret(outcome["note"])
                or not isinstance(outcome.get("at"), (int, float))
                or not REVISION_RE.fullmatch(str(outcome.get("content_revision") or ""))
            ):
                raise LearnedSkillError("learned-skill outcome is invalid")
        return dict(document)

    def read(
        self,
        category: object,
        skill_name: object,
        *,
        current_content_revision: str = "",
    ) -> dict[str, Any] | None:
        path = self._record_path(category, skill_name, create=False)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise LearnedSkillError("learned-skill metadata is unavailable") from exc
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size < 2
            or metadata.st_size > MAX_LIFECYCLE_BYTES
        ):
            raise LearnedSkillError("learned-skill metadata is not private")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if opened.st_dev != metadata.st_dev or opened.st_ino != metadata.st_ino:
                raise LearnedSkillError("learned-skill metadata changed while opening")
            payload = read_bounded_fd(descriptor, MAX_LIFECYCLE_BYTES)
        finally:
            os.close(descriptor)
        if len(payload) > MAX_LIFECYCLE_BYTES:
            raise LearnedSkillError("learned-skill metadata is too large")
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise LearnedSkillError("learned-skill metadata is invalid") from exc
        skill_path = f"{_component(category, label='skill category')}/{_component(skill_name, label='skill name')}"
        public = self._validate_document(document, expected_path=skill_path)
        public["metadata_revision"] = hashlib.sha256(payload).hexdigest()
        stale = bool(
            current_content_revision
            and public["content_revision"] != current_content_revision
        )
        public["metadata_stale"] = stale
        if stale:
            public["status"] = "needs_review"
        return public

    def _write(self, category: str, skill_name: str, document: Mapping[str, Any]) -> dict[str, Any]:
        path = self._record_path(category, skill_name, create=True)
        payload = (
            json.dumps(dict(document), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        if len(payload) > MAX_LIFECYCLE_BYTES:
            raise LearnedSkillError("learned-skill metadata is too large")
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise LearnedSkillError("could not write learned-skill metadata")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, path)
            os.chmod(path, 0o600, follow_symlinks=False)
            directory_fd = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except LearnedSkillError:
            raise
        except OSError as exc:
            raise LearnedSkillError("could not save learned-skill metadata") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except (FileNotFoundError, OSError):
                pass
        saved = self.read(
            category,
            skill_name,
            current_content_revision=str(document["content_revision"]),
        )
        if saved is None:
            raise LearnedSkillError("saved learned-skill metadata is unavailable")
        return saved

    def save_protocol(
        self,
        *,
        category: str,
        skill_name: str,
        content_revision: str,
        evidence: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        with self._exclusive_lock():
            return self._save_protocol_unlocked(
                category=category,
                skill_name=skill_name,
                content_revision=content_revision,
                evidence=evidence,
            )

    def _save_protocol_unlocked(
        self,
        *,
        category: str,
        skill_name: str,
        content_revision: str,
        evidence: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        if not REVISION_RE.fullmatch(str(content_revision or "")):
            raise LearnedSkillError("learned-skill content revision is invalid")
        refs = [
            {"note_id": str(item.get("note_id") or ""), "revision": str(item.get("revision") or "")}
            for item in evidence
        ]
        # Validate the references before any protocol metadata is published.
        for item in refs:
            if (
                not re.fullmatch(r"note-[0-9a-f]{32}", item["note_id"])
                or not REVISION_RE.fullmatch(item["revision"])
            ):
                raise LearnedSkillError("learned-skill evidence reference is invalid")
        if not refs or len(refs) > 16:
            raise LearnedSkillError("a learned skill requires 1-16 evidence notes")
        existing = self.read(category, skill_name)
        now = time.time()
        usage = (
            dict(existing["usage"])
            if existing is not None
            else {
                "activations": 0,
                "successes": 0,
                "adaptations": 0,
                "failures": 0,
                "not_applicable": 0,
            }
        )
        document = {
            "version": self.VERSION,
            "skill_path": f"{_component(category, label='skill category')}/{_component(skill_name, label='skill name')}",
            "content_revision": content_revision,
            "status": "ready",
            "evidence": refs,
            "created_at": float(existing["created_at"]) if existing else now,
            "updated_at": now,
            "revision_number": int(existing["revision_number"]) + 1 if existing else 1,
            "usage": usage,
            "last_outcome": existing.get("last_outcome") if existing else None,
        }
        return self._write(category, skill_name, document)

    def record_activation(
        self, *, category: str, skill_name: str, content_revision: str
    ) -> dict[str, Any]:
        with self._exclusive_lock():
            return self._record_activation_unlocked(
                category=category,
                skill_name=skill_name,
                content_revision=content_revision,
            )

    def _record_activation_unlocked(
        self, *, category: str, skill_name: str, content_revision: str
    ) -> dict[str, Any]:
        record = self.read(
            category, skill_name, current_content_revision=content_revision
        )
        if record is None:
            raise LearnedSkillError(
                "this private skill predates the evidence lifecycle and must be revalidated"
            )
        if record["status"] != "ready" or record.get("metadata_stale"):
            raise LearnedSkillError(
                f"this private skill is {record['status']} and must be revised from fresh evidence"
            )
        document = {key: value for key, value in record.items() if key not in {"metadata_revision", "metadata_stale"}}
        document["usage"] = dict(document["usage"])
        document["usage"]["activations"] += 1
        document["updated_at"] = time.time()
        return self._write(category, skill_name, document)

    def record_outcome(
        self,
        *,
        category: str,
        skill_name: str,
        content_revision: str,
        outcome: str,
        note: str,
    ) -> dict[str, Any]:
        with self._exclusive_lock():
            return self._record_outcome_unlocked(
                category=category,
                skill_name=skill_name,
                content_revision=content_revision,
                outcome=outcome,
                note=note,
            )

    def _record_outcome_unlocked(
        self,
        *,
        category: str,
        skill_name: str,
        content_revision: str,
        outcome: str,
        note: str,
    ) -> dict[str, Any]:
        value = str(outcome or "").strip()
        if value not in VALID_OUTCOMES:
            raise LearnedSkillError("skill outcome is invalid")
        clean_note = _bounded_note(note)
        if value in {"adapted", "failed"} and not clean_note:
            raise LearnedSkillError("adapted or failed outcomes require a concise note")
        record = self.read(
            category, skill_name, current_content_revision=content_revision
        )
        if record is None:
            raise LearnedSkillError("learned-skill metadata is missing")
        if record.get("metadata_stale"):
            raise LearnedSkillError("the skill changed since activation")
        document = {key: value for key, value in record.items() if key not in {"metadata_revision", "metadata_stale"}}
        document["usage"] = dict(document["usage"])
        counter = {
            "success": "successes",
            "adapted": "adaptations",
            "failed": "failures",
            "not_applicable": "not_applicable",
        }[value]
        document["usage"][counter] += 1
        if value == "failed":
            document["status"] = "quarantined"
        elif value == "adapted":
            document["status"] = "needs_review"
        document["last_outcome"] = {
            "outcome": value,
            "note": clean_note,
            "at": time.time(),
            "content_revision": content_revision,
        }
        document["updated_at"] = time.time()
        return self._write(category, skill_name, document)

    def remove(self, category: str, skill_name: str) -> None:
        with self._exclusive_lock():
            self._remove_unlocked(category, skill_name)

    def _remove_unlocked(self, category: str, skill_name: str) -> None:
        path = self._record_path(category, skill_name, create=False)
        try:
            metadata = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise LearnedSkillError("learned-skill metadata is not a regular file")
            path.unlink()
        except FileNotFoundError:
            return
        except LearnedSkillError:
            raise
        except OSError as exc:
            raise LearnedSkillError("could not remove learned-skill metadata") from exc
        for directory in (path.parent, self.root):
            if directory is None:
                continue
            try:
                directory.rmdir()
            except OSError:
                pass


__all__ = (
    "LearnedSkillError",
    "LearnedSkillStore",
    "MAX_PRIVATE_SKILLS",
    "VALID_OUTCOMES",
    "skill_revision",
)
