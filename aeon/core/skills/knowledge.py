"""Owner-private, per-agent knowledge used to develop reusable skills.

Knowledge notes are agent state, not project files and not authority.  They live
beside one managed instance's private skill overlay, survive process restarts,
and can be copied only by the authenticated Nexus transfer workflow.
"""

from __future__ import annotations

import hashlib
import hmac
import fcntl
import json
import os
import re
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping


MAX_SKILL_KNOWLEDGE_BYTES = 64 * 1024
MAX_SKILL_KNOWLEDGE_NOTES = 128
SKILL_NOTE_ID_RE = re.compile(r"^note-[0-9a-f]{32}$")
SKILL_PATH_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}/[A-Za-z0-9][A-Za-z0-9_-]{0,79}$"
)
_PERSISTED_SECRET_RE = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"(?:password|passwd|token|secret|api[_ -]?key|authorization)\s*[:=]\s*\S+|"
    r"bearer\s+[A-Za-z0-9._~+/-]{16,})",
    re.IGNORECASE,
)


class SkillKnowledgeError(ValueError):
    """A private skill-knowledge operation failed its integrity contract."""


def contains_persisted_secret(value: object) -> bool:
    """Recognize credential material that must stay in Nexus credential storage."""

    return bool(_PERSISTED_SECRET_RE.search(str(value or "")))


def _normalized_learning(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SkillKnowledgeError("learning must be an object")
    expected = {
        "candidate_skill_path",
        "procedure",
        "verification",
        "procedure_stable",
        "uncertainty",
    }
    if set(value) != expected:
        raise SkillKnowledgeError("learning fields are invalid")
    candidate = str(value.get("candidate_skill_path") or "").strip()
    if not SKILL_PATH_RE.fullmatch(candidate):
        raise SkillKnowledgeError("learning candidate_skill_path is invalid")
    uncertainty = str(value.get("uncertainty") or "").strip().lower()
    if uncertainty not in {"low", "medium", "high"}:
        raise SkillKnowledgeError("learning uncertainty must be low, medium, or high")
    if not isinstance(value.get("procedure_stable"), bool):
        raise SkillKnowledgeError("learning procedure_stable must be boolean")
    return {
        "candidate_skill_path": candidate,
        "procedure": _bounded_text(
            value.get("procedure"), label="learning procedure", maximum=16 * 1024
        ),
        "verification": _bounded_text(
            value.get("verification"), label="learning verification", maximum=8 * 1024
        ),
        "procedure_stable": bool(value["procedure_stable"]),
        "uncertainty": uncertainty,
    }


def _normalized_experience(value: object) -> dict[str, Any] | None:
    """Validate harness-captured receipts supporting a learning claim."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SkillKnowledgeError("experience must be an object")
    expected = {
        "request_id",
        "attempt_count",
        "failure_count",
        "success_count",
        "recovered_after_failure",
        "receipts",
    }
    if set(value) != expected:
        raise SkillKnowledgeError("experience fields are invalid")
    request_id = str(value.get("request_id") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id):
        raise SkillKnowledgeError("experience request identity is invalid")
    receipts = value.get("receipts")
    if not isinstance(receipts, list) or not receipts or len(receipts) > 64:
        raise SkillKnowledgeError("experience receipts are invalid")
    clean_receipts = []
    for receipt in receipts:
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "tool",
            "status",
            "error_code",
            "summary_sha256",
        }:
            raise SkillKnowledgeError("experience receipt is invalid")
        tool = str(receipt.get("tool") or "")[:200]
        status_value = str(receipt.get("status") or "")
        error_code = str(receipt.get("error_code") or "")[:100]
        summary_sha256 = str(receipt.get("summary_sha256") or "")
        if (
            not tool
            or status_value
            not in {"ok", "failed", "blocked", "pending", "no_change", "skipped"}
            or not re.fullmatch(r"[0-9a-f]{64}", summary_sha256)
        ):
            raise SkillKnowledgeError("experience receipt fields are invalid")
        clean_receipts.append(
            {
                "tool": tool,
                "status": status_value,
                "error_code": error_code,
                "summary_sha256": summary_sha256,
            }
        )
    try:
        attempts = int(value.get("attempt_count"))
        failures = int(value.get("failure_count"))
        successes = int(value.get("success_count"))
    except (TypeError, ValueError) as exc:
        raise SkillKnowledgeError("experience counters are invalid") from exc
    if (
        attempts != len(clean_receipts)
        or failures < 0
        or successes < 0
        or failures + successes > attempts
        or not isinstance(value.get("recovered_after_failure"), bool)
    ):
        raise SkillKnowledgeError("experience counters are inconsistent")
    return {
        "request_id": request_id,
        "attempt_count": attempts,
        "failure_count": failures,
        "success_count": successes,
        "recovered_after_failure": bool(value["recovered_after_failure"]),
        "receipts": clean_receipts,
    }


def _experience_proves_recovery(value: object) -> bool:
    """Derive earned recovery from receipts instead of trusting claim counters."""

    if not isinstance(value, Mapping):
        return False
    receipts = value.get("receipts")
    if not isinstance(receipts, list):
        return False
    failure_indexes = [
        index
        for index, receipt in enumerate(receipts)
        if isinstance(receipt, Mapping) and receipt.get("status") == "failed"
    ]
    success_indexes = [
        index
        for index, receipt in enumerate(receipts)
        if isinstance(receipt, Mapping) and receipt.get("status") == "ok"
    ]
    return bool(
        value.get("failure_count") == len(failure_indexes)
        and value.get("success_count") == len(success_indexes)
        and value.get("attempt_count") == len(receipts)
        and value.get("recovered_after_failure") is True
        and any(
            failure < success
            for failure in failure_indexes
            for success in success_indexes
        )
    )


def _bounded_text(value: object, *, label: str, maximum: int) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text or "\x00" in text:
        raise SkillKnowledgeError(f"{label} must be non-empty text")
    if len(text.encode("utf-8")) > maximum:
        raise SkillKnowledgeError(f"{label} exceeds the {maximum}-byte limit")
    return text


def _normalized_related_skills(values: object) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise SkillKnowledgeError("related_skill_paths must be a list")
    normalized: list[str] = []
    for raw in values[:32]:
        value = str(raw or "").strip()
        if not SKILL_PATH_RE.fullmatch(value):
            raise SkillKnowledgeError(
                "related_skill_paths must contain '<category>/<skill_name>' values"
            )
        if value not in normalized:
            normalized.append(value)
    return normalized


class SkillKnowledgeStore:
    """Crash-safe bounded storage beneath one exact managed instance."""

    VERSION = 1

    def __init__(self, root: str | os.PathLike[str] | None):
        # Preserve the supplied path identity. Resolving here would turn a
        # symlinked storage root into an apparently ordinary target directory
        # before the no-symlink integrity checks get a chance to reject it.
        self.root = Path(root).expanduser().absolute() if root else None

    def _ensure_root(self) -> Path:
        if self.root is None:
            raise SkillKnowledgeError(
                "skill knowledge is available only to a managed agent instance"
            )
        try:
            self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
            metadata = self.root.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or self.root.resolve(strict=True) != self.root.absolute()
            ):
                raise SkillKnowledgeError("skill knowledge storage is not private")
            os.chmod(self.root, 0o700, follow_symlinks=False)
        except SkillKnowledgeError:
            raise
        except OSError as exc:
            raise SkillKnowledgeError("skill knowledge storage is unavailable") from exc
        return self.root

    def _root_fd(self, *, create: bool) -> int:
        root = self._ensure_root() if create else self.root
        if root is None:
            raise SkillKnowledgeError(
                "skill knowledge is available only to a managed agent instance"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            metadata = root.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or root.resolve(strict=True) != root.absolute()
            ):
                raise SkillKnowledgeError("skill knowledge storage is not private")
            fd = os.open(root, flags)
            opened = os.fstat(fd)
            if opened.st_dev != metadata.st_dev or opened.st_ino != metadata.st_ino:
                os.close(fd)
                raise SkillKnowledgeError(
                    "skill knowledge storage changed while opening"
                )
            return fd
        except FileNotFoundError:
            raise
        except SkillKnowledgeError:
            raise
        except OSError as exc:
            raise SkillKnowledgeError("skill knowledge storage is unavailable") from exc

    @contextmanager
    def _exclusive_lock(self):
        """Serialize revision checks and writes across agent/dashboard processes."""

        root_fd = self._root_fd(create=True)
        lock_fd = None
        try:
            lock_fd = os.open(
                ".wiki.lock",
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=root_fd,
            )
            os.fchmod(lock_fd, 0o600)
            metadata = os.fstat(lock_fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
            ):
                raise SkillKnowledgeError("skill knowledge lock is not private")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        except SkillKnowledgeError:
            raise
        except OSError as exc:
            raise SkillKnowledgeError("skill knowledge lock is unavailable") from exc
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)
            os.close(root_fd)

    @staticmethod
    def _filename(note_id: str) -> str:
        value = str(note_id or "").strip()
        if not SKILL_NOTE_ID_RE.fullmatch(value):
            raise SkillKnowledgeError("note_id is invalid")
        return f"{value}.json"

    def _read_payload(self, note_id: str) -> tuple[dict[str, Any], bytes] | None:
        filename = self._filename(note_id)
        try:
            root_fd = self._root_fd(create=False)
        except FileNotFoundError:
            return None
        fd = None
        try:
            try:
                fd = os.open(
                    filename,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=root_fd,
                )
            except FileNotFoundError:
                return None
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
                or metadata.st_size < 2
                or metadata.st_size > MAX_SKILL_KNOWLEDGE_BYTES + 4096
            ):
                raise SkillKnowledgeError("skill knowledge note is not private")
            payload = os.read(fd, MAX_SKILL_KNOWLEDGE_BYTES + 4097)
            if len(payload) > MAX_SKILL_KNOWLEDGE_BYTES + 4096:
                raise SkillKnowledgeError("skill knowledge note is too large")
            try:
                document = json.loads(payload.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise SkillKnowledgeError("skill knowledge note is invalid") from exc
            if not isinstance(document, dict) or document.get("version") != self.VERSION:
                raise SkillKnowledgeError("skill knowledge note version is invalid")
            if document.get("id") != note_id:
                raise SkillKnowledgeError("skill knowledge note identity is invalid")
            title = document.get("title")
            content = document.get("content")
            related = document.get("related_skill_paths")
            origin = document.get("origin")
            learning = document.get("learning")
            experience = document.get("experience")
            if (
                not isinstance(title, str)
                or not title.strip()
                or len(title.encode("utf-8")) > 240
                or not isinstance(content, str)
                or not content.strip()
                or len(content.encode("utf-8")) > MAX_SKILL_KNOWLEDGE_BYTES
                or not isinstance(related, list)
                or len(related) > 32
                or any(not isinstance(item, str) or not SKILL_PATH_RE.fullmatch(item) for item in related)
                or not isinstance(origin, dict)
                or not isinstance(document.get("created_at"), (int, float))
                or not isinstance(document.get("updated_at"), (int, float))
            ):
                raise SkillKnowledgeError("skill knowledge note fields are invalid")
            _normalized_learning(learning)
            _normalized_experience(experience)
            return document, payload
        finally:
            if fd is not None:
                os.close(fd)
            os.close(root_fd)

    @staticmethod
    def _public(document: Mapping[str, Any], payload: bytes) -> dict[str, Any]:
        return {
            "id": str(document.get("id") or ""),
            "title": str(document.get("title") or ""),
            "content": str(document.get("content") or ""),
            "related_skill_paths": list(document.get("related_skill_paths") or []),
            "created_at": float(document.get("created_at") or 0.0),
            "updated_at": float(document.get("updated_at") or 0.0),
            "origin": dict(document.get("origin") or {}),
            "learning": (
                dict(document["learning"])
                if isinstance(document.get("learning"), dict)
                else None
            ),
            "experience": (
                dict(document["experience"])
                if isinstance(document.get("experience"), dict)
                else None
            ),
            "skill_evidence_eligible": bool(
                isinstance(document.get("learning"), dict)
                and document["learning"].get("procedure_stable") is True
                and document["learning"].get("uncertainty") == "low"
                and _experience_proves_recovery(document.get("experience"))
                and dict(document.get("origin") or {}).get("kind") == "agent-authored"
            ),
            "revision": hashlib.sha256(payload).hexdigest(),
        }

    def list_notes(self) -> list[dict[str, Any]]:
        if self.root is None:
            return []
        try:
            root_fd = self._root_fd(create=False)
        except FileNotFoundError:
            return []
        try:
            names = sorted(os.listdir(root_fd))
        except OSError as exc:
            raise SkillKnowledgeError("skill knowledge storage is unavailable") from exc
        finally:
            os.close(root_fd)
        notes: list[dict[str, Any]] = []
        for name in names:
            if not name.endswith(".json"):
                continue
            note_id = name[:-5]
            if not SKILL_NOTE_ID_RE.fullmatch(note_id):
                continue
            loaded = self._read_payload(note_id)
            if loaded is None:
                continue
            notes.append(self._public(*loaded))
            if len(notes) > MAX_SKILL_KNOWLEDGE_NOTES:
                raise SkillKnowledgeError("this agent has too many skill knowledge notes")
        return sorted(notes, key=lambda item: (item["title"].casefold(), item["id"]))

    def read_note(self, note_id: str) -> dict[str, Any] | None:
        loaded = self._read_payload(note_id)
        return self._public(*loaded) if loaded else None

    def search_notes(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        """Search the bounded wiki with a deterministic in-memory inverted index."""

        text = _bounded_text(query, label="query", maximum=500)
        terms = list(dict.fromkeys(re.findall(r"[A-Za-z0-9_-]{2,}", text.casefold())))
        if not terms:
            raise SkillKnowledgeError("query must contain a searchable term")
        try:
            bounded_limit = max(1, min(20, int(limit)))
        except (TypeError, ValueError) as exc:
            raise SkillKnowledgeError("search limit is invalid") from exc
        phrase = text.casefold()
        matches: list[tuple[int, float, dict[str, Any]]] = []
        for note in self.list_notes():
            title = str(note.get("title") or "").casefold()
            content = str(note.get("content") or "").casefold()
            related = " ".join(note.get("related_skill_paths") or []).casefold()
            learning = json.dumps(note.get("learning") or {}, ensure_ascii=False).casefold()
            score = sum(
                (8 if term in title else 0)
                + (6 if term in related else 0)
                + (3 if term in learning else 0)
                + (1 if term in content else 0)
                for term in terms
            )
            haystack = f"{title}\n{related}\n{learning}\n{content}"
            if phrase in haystack:
                score += 10
            if score:
                matches.append((score, float(note.get("updated_at") or 0.0), note))
        matches.sort(key=lambda item: (-item[0], -item[1], item[2]["id"]))
        return [
            {
                **note,
                "search_score": score,
                "preview": str(note.get("content") or "")[:700],
            }
            for score, _updated, note in matches[:bounded_limit]
        ]

    def save_note(
        self,
        *,
        title: str,
        content: str,
        related_skill_paths: object = None,
        note_id: str = "",
        expected_revision: str = "",
        origin: Mapping[str, Any] | None = None,
        learning: Mapping[str, Any] | None = None,
        experience: Mapping[str, Any] | None = None,
        clear_learning: bool = False,
    ) -> dict[str, Any]:
        with self._exclusive_lock():
            return self._save_note_unlocked(
                title=title,
                content=content,
                related_skill_paths=related_skill_paths,
                note_id=note_id,
                expected_revision=expected_revision,
                origin=origin,
                learning=learning,
                experience=experience,
                clear_learning=clear_learning,
            )

    def _save_note_unlocked(
        self,
        *,
        title: str,
        content: str,
        related_skill_paths: object = None,
        note_id: str = "",
        expected_revision: str = "",
        origin: Mapping[str, Any] | None = None,
        learning: Mapping[str, Any] | None = None,
        experience: Mapping[str, Any] | None = None,
        clear_learning: bool = False,
    ) -> dict[str, Any]:
        clean_title = _bounded_text(title, label="title", maximum=240)
        clean_content = _bounded_text(
            content, label="content", maximum=MAX_SKILL_KNOWLEDGE_BYTES
        )
        related = _normalized_related_skills(related_skill_paths)
        clean_learning = _normalized_learning(learning)
        clean_experience = _normalized_experience(experience)
        if clear_learning and (learning is not None or experience is not None):
            raise SkillKnowledgeError(
                "clear_learning cannot be combined with replacement learning"
            )
        if (clean_learning is None) != (clean_experience is None):
            raise SkillKnowledgeError(
                "learning and its harness-captured experience must be saved together"
            )
        existing = None
        created_at = time.time()
        if note_id:
            existing = self._read_payload(note_id)
            if existing is None:
                raise SkillKnowledgeError("skill knowledge note no longer exists")
            current_revision = hashlib.sha256(existing[1]).hexdigest()
            if not expected_revision or not hmac.compare_digest(
                current_revision, expected_revision
            ):
                raise SkillKnowledgeError(
                    "skill knowledge note changed since it was loaded"
                )
            created_at = float(existing[0].get("created_at") or created_at)
            if learning is None and experience is None and not clear_learning:
                clean_learning = _normalized_learning(existing[0].get("learning"))
                clean_experience = _normalized_experience(existing[0].get("experience"))
            elif clear_learning:
                clean_learning = None
                clean_experience = None
        else:
            if clear_learning:
                raise SkillKnowledgeError(
                    "clear_learning is valid only when updating an existing note"
                )
            if len(self.list_notes()) >= MAX_SKILL_KNOWLEDGE_NOTES:
                raise SkillKnowledgeError("this agent has too many skill knowledge notes")
            note_id = f"note-{uuid.uuid4().hex}"
        if contains_persisted_secret(
            f"{clean_title}\n{clean_content}\n"
            f"{json.dumps(clean_learning or {}, ensure_ascii=False)}"
        ):
            raise SkillKnowledgeError(
                "secret-like credentials cannot be stored in the skill wiki; use an "
                "opaque Nexus credential handle"
            )
        now = time.time()
        inherited_origin = existing[0].get("origin") if existing and origin is None else None
        safe_origin = {
            str(key)[:64]: str(value)[:240]
            for key, value in dict(origin or inherited_origin or {"kind": "agent-authored"}).items()
            if str(key) and value is not None
        }
        document = {
            "version": self.VERSION,
            "id": note_id,
            "title": clean_title,
            "content": clean_content,
            "related_skill_paths": related,
            "created_at": created_at,
            "updated_at": now,
            "origin": safe_origin,
            "learning": clean_learning,
            "experience": clean_experience,
        }
        payload = (
            json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        if len(payload) > MAX_SKILL_KNOWLEDGE_BYTES + 4096:
            raise SkillKnowledgeError("skill knowledge note is too large")
        root_fd = self._root_fd(create=True)
        temporary = f".{note_id}.{uuid.uuid4().hex}.tmp"
        fd = None
        try:
            fd = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=root_fd,
            )
            os.fchmod(fd, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise SkillKnowledgeError("could not write skill knowledge note")
                view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = None
            os.replace(temporary, f"{note_id}.json", src_dir_fd=root_fd, dst_dir_fd=root_fd)
            os.fsync(root_fd)
        except SkillKnowledgeError:
            raise
        except OSError as exc:
            raise SkillKnowledgeError("could not save skill knowledge note") from exc
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.unlink(temporary, dir_fd=root_fd)
            except (FileNotFoundError, OSError):
                pass
            os.close(root_fd)
        saved = self.read_note(note_id)
        if saved is None:
            raise SkillKnowledgeError("saved skill knowledge note is unavailable")
        return saved

    def delete_note(self, note_id: str, *, expected_revision: str) -> None:
        with self._exclusive_lock():
            self._delete_note_unlocked(
                note_id, expected_revision=expected_revision
            )

    def _delete_note_unlocked(self, note_id: str, *, expected_revision: str) -> None:
        loaded = self._read_payload(note_id)
        if loaded is None:
            raise SkillKnowledgeError("skill knowledge note no longer exists")
        current_revision = hashlib.sha256(loaded[1]).hexdigest()
        if not expected_revision or not hmac.compare_digest(
            current_revision, expected_revision
        ):
            raise SkillKnowledgeError("skill knowledge note changed since it was loaded")
        root_fd = self._root_fd(create=False)
        try:
            os.unlink(self._filename(note_id), dir_fd=root_fd)
            os.fsync(root_fd)
        except OSError as exc:
            raise SkillKnowledgeError("could not delete skill knowledge note") from exc
        finally:
            os.close(root_fd)


__all__ = (
    "MAX_SKILL_KNOWLEDGE_BYTES",
    "MAX_SKILL_KNOWLEDGE_NOTES",
    "SKILL_NOTE_ID_RE",
    "SKILL_PATH_RE",
    "contains_persisted_secret",
    "SkillKnowledgeError",
    "SkillKnowledgeStore",
)
