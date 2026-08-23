"""Private, versioned instruction profiles for managed agent sessions.

The records in this module describe only instructions that Nexus can observe on
the local machine.  They are not, and must never be presented as, hidden vendor
system prompts.  Prompt bodies stay in the mode-600 remote SQLite registry and
are intentionally excluded from audit records, process arguments, and process
environments.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
import time
from pathlib import Path

from .store import RemoteStore


MAX_INSTRUCTION_BYTES = 64 * 1024
MAX_NAME_CHARS = 80
MAX_LABEL_CHARS = 80
MAX_SOURCE_REF_CHARS = 1024

AGENT_KINDS = frozenset({"aeon", "codex", "claude", "grok"})
SOURCE_KINDS = frozenset({"manual", "workspace", "aeon_fixed"})
WORKSPACE_FILENAMES = {
    "codex": "AGENTS.md",
    "claude": "CLAUDE.md",
    "grok": "AGENTS.md",
}
AEON_DIRECTIVE_FILES = (
    "core_directives.txt",
    "docker_directives.txt",
    "important_reminders.txt",
    "primary_agent_instructions.txt",
)
_KIND_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class InstructionProfileError(ValueError):
    """Base class for safe, user-displayable profile errors."""


class InstructionNotFound(InstructionProfileError):
    """A requested profile, version, instance, or source does not exist."""


class InstructionConflict(InstructionProfileError):
    """An optimistic revision or unique-name check failed."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_text(value: str | None, *, field: str, maximum: int) -> str:
    rendered = (value or "").strip()
    if not rendered:
        raise InstructionProfileError(f"{field} is required")
    if len(rendered) > maximum:
        raise InstructionProfileError(f"{field} must be at most {maximum} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in rendered):
        raise InstructionProfileError(f"{field} contains a control character")
    return rendered


def _instruction_text(value: str | None) -> str:
    rendered = value or ""
    if "\x00" in rendered:
        raise InstructionProfileError("Instruction text contains an invalid NUL character")
    if len(rendered.encode("utf-8")) > MAX_INSTRUCTION_BYTES:
        raise InstructionProfileError(
            f"Instruction text must be at most {MAX_INSTRUCTION_BYTES} UTF-8 bytes"
        )
    return rendered


def _agent_kind(value: str | None) -> str:
    rendered = (value or "").strip().lower()
    if not _KIND_RE.fullmatch(rendered) or rendered not in AGENT_KINDS:
        raise InstructionProfileError(
            f"Agent kind must be one of: {', '.join(sorted(AGENT_KINDS))}"
        )
    return rendered


class InstructionProfileService:
    """Bounded persistence and allowlisted discovery for local instructions."""

    disclosure = (
        "Locally known instructions only; vendor-managed hidden system prompts "
        "are not observable here."
    )

    def __init__(
        self,
        store: RemoteStore,
        *,
        project_root: str | Path,
        allowed_roots: tuple[str | Path, ...] | list[str | Path],
    ):
        self.store = store
        self.project_root = Path(project_root).expanduser().resolve(strict=True)
        if not self.project_root.is_dir():
            raise InstructionProfileError("Aeon project root is not a directory")
        roots = []
        for root in allowed_roots:
            resolved = Path(root).expanduser().resolve(strict=True)
            if not resolved.is_dir():
                raise InstructionProfileError("An allowed workspace root is not a directory")
            roots.append(resolved)
        if not roots:
            raise InstructionProfileError("At least one allowed workspace root is required")
        self.allowed_roots = tuple(dict.fromkeys(roots))

    @staticmethod
    def _read_regular_utf8(path: Path) -> str:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except (FileNotFoundError, OSError) as exc:
            raise InstructionNotFound(f"Known instruction source is unavailable: {path.name}") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise InstructionProfileError(
                    f"Known instruction source is not a regular file: {path.name}"
                )
            if metadata.st_size > MAX_INSTRUCTION_BYTES:
                raise InstructionProfileError(
                    f"Known instruction source exceeds {MAX_INSTRUCTION_BYTES} bytes: {path.name}"
                )
            chunks: list[bytes] = []
            remaining = MAX_INSTRUCTION_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 65536))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > MAX_INSTRUCTION_BYTES:
                raise InstructionProfileError(
                    f"Known instruction source exceeds {MAX_INSTRUCTION_BYTES} bytes: {path.name}"
                )
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise InstructionProfileError(
                    f"Known instruction source is not valid UTF-8: {path.name}"
                ) from exc
        finally:
            os.close(descriptor)

    def _validated_workspace(self, workspace: str | Path) -> tuple[Path, Path]:
        try:
            resolved = Path(workspace).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise InstructionNotFound("Workspace does not exist") from exc
        if not resolved.is_dir():
            raise InstructionProfileError("Workspace must be a directory")
        matching = [root for root in self.allowed_roots if resolved.is_relative_to(root)]
        if not matching:
            raise InstructionProfileError("Workspace is outside the configured allowed roots")
        # The most-specific configured root defines the discovery boundary.
        boundary = max(matching, key=lambda item: len(item.parts))
        return resolved, boundary

    @staticmethod
    def _compose_documents(documents: list[dict]) -> str:
        if not documents:
            return ""
        if len(documents) == 1:
            return documents[0]["content"]
        sections = []
        for document in documents:
            sections.append(
                f"<!-- Nexus locally known source: {document['source_ref']} -->\n"
                f"{document['content']}"
            )
        return _instruction_text("\n\n".join(sections))

    def discover_known_instructions(
        self, agent_kind: str, workspace: str | Path
    ) -> dict:
        """Read only fixed Aeon directives or applicable workspace instruction files.

        Codex and Claude discovery walks from the most-specific allowed root to
        the workspace and reads only exact ``AGENTS.md`` or ``CLAUDE.md`` names.
        Symlinks and non-regular files fail closed.  Grok uses the same AGENTS.md
        project-rule convention documented by its official CLI.
        """

        kind = _agent_kind(agent_kind)
        resolved_workspace, boundary = self._validated_workspace(workspace)
        documents: list[dict] = []
        source_kind = "workspace"

        if kind == "aeon":
            source_kind = "aeon_fixed"
            prompt_root = (self.project_root / "aeon" / "core" / "prompts").resolve(
                strict=True
            )
            if not prompt_root.is_relative_to(self.project_root):
                raise InstructionProfileError("Aeon directive directory escaped project root")
            for filename in AEON_DIRECTIVE_FILES:
                candidate = prompt_root / filename
                content = self._read_regular_utf8(candidate)
                documents.append(
                    {
                        "source_ref": f"aeon/core/prompts/{filename}",
                        "content": content,
                        "content_sha256": _sha256_text(content),
                    }
                )
        elif kind in WORKSPACE_FILENAMES:
            filename = WORKSPACE_FILENAMES[kind]
            relative = resolved_workspace.relative_to(boundary)
            directories = [boundary]
            cursor = boundary
            for part in relative.parts:
                cursor = cursor / part
                directories.append(cursor)
            for directory in directories:
                candidate = directory / filename
                try:
                    candidate.lstat()
                except FileNotFoundError:
                    continue
                content = self._read_regular_utf8(candidate)
                ref = candidate.relative_to(boundary).as_posix()
                documents.append(
                    {
                        "source_ref": ref,
                        "content": content,
                        "content_sha256": _sha256_text(content),
                    }
                )
        else:
            raise InstructionNotFound(
                "No allowlisted local instruction-file convention is known for this agent"
            )

        if not documents:
            raise InstructionNotFound(
                f"No locally known {WORKSPACE_FILENAMES.get(kind, 'instruction')} source was found"
            )
        content = self._compose_documents(documents)
        source_refs = [document["source_ref"] for document in documents]
        source_ref = json.dumps(source_refs, separators=(",", ":"))
        if len(source_ref) > MAX_SOURCE_REF_CHARS:
            raise InstructionProfileError("Known instruction source reference is too long")
        return {
            "agent_kind": kind,
            "workspace": str(resolved_workspace),
            "source_kind": source_kind,
            "source_ref": source_ref,
            "documents": documents,
            "content": content,
            "content_sha256": _sha256_text(content),
            "disclosure": self.disclosure,
        }

    @staticmethod
    def _profile_row(conn: sqlite3.Connection, profile_id: str):
        return conn.execute(
            "SELECT p.*,v.id AS latest_version_id,v.version_number AS latest_version_number,"
            "v.label AS latest_version_label,v.content_sha256 AS latest_content_sha256 "
            "FROM instruction_profiles p LEFT JOIN instruction_profile_versions v "
            "ON v.profile_id=p.id AND v.version_number=(SELECT MAX(v2.version_number) "
            "FROM instruction_profile_versions v2 WHERE v2.profile_id=p.id) "
            "WHERE p.id=?",
            (profile_id,),
        ).fetchone()

    def create_profile(self, *, agent_kind: str, name: str, actor: str) -> dict:
        kind = _agent_kind(agent_kind)
        clean_name = _bounded_text(name, field="Profile name", maximum=MAX_NAME_CHARS)
        clean_actor = _bounded_text(actor, field="Actor", maximum=MAX_NAME_CHARS)
        profile_id = f"ip-{secrets.token_hex(12)}"
        now = time.time()
        try:
            with self.store._lock, self.store._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO instruction_profiles(id,agent_kind,name,created_at,updated_at,created_by) "
                    "VALUES(?,?,?,?,?,?)",
                    (profile_id, kind, clean_name, now, now, clean_actor),
                )
                row = self._profile_row(conn, profile_id)
                conn.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            raise InstructionConflict("A profile with that name already exists") from exc
        return dict(row)

    def list_profiles(self, *, agent_kind: str | None = None) -> list[dict]:
        parameters: tuple = ()
        where = ""
        if agent_kind is not None:
            where = "WHERE p.agent_kind=?"
            parameters = (_agent_kind(agent_kind),)
        with self.store._connect() as conn:
            rows = conn.execute(
                "SELECT p.*,v.id AS latest_version_id,v.version_number AS latest_version_number,"
                "v.label AS latest_version_label,v.content_sha256 AS latest_content_sha256 "
                "FROM instruction_profiles p LEFT JOIN instruction_profile_versions v "
                "ON v.profile_id=p.id AND v.version_number=(SELECT MAX(v2.version_number) "
                "FROM instruction_profile_versions v2 WHERE v2.profile_id=p.id) "
                f"{where} ORDER BY p.agent_kind,p.name COLLATE NOCASE",
                parameters,
            )
            return [dict(row) for row in rows]

    def get_profile(self, profile_id: str) -> dict:
        with self.store._connect() as conn:
            row = self._profile_row(conn, profile_id)
        if row is None:
            raise InstructionNotFound("Unknown instruction profile")
        return dict(row)

    @staticmethod
    def _insert_version(
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        label: str,
        content: str,
        source_kind: str,
        source_ref: str,
        actor: str,
    ) -> dict:
        profile = conn.execute(
            "SELECT id FROM instruction_profiles WHERE id=?", (profile_id,)
        ).fetchone()
        if profile is None:
            raise InstructionNotFound("Unknown instruction profile")
        next_number = int(
            conn.execute(
                "SELECT COALESCE(MAX(version_number),0)+1 FROM instruction_profile_versions "
                "WHERE profile_id=?",
                (profile_id,),
            ).fetchone()[0]
        )
        version_id = f"ipv-{secrets.token_hex(12)}"
        now = time.time()
        digest = _sha256_text(content)
        conn.execute(
            "INSERT INTO instruction_profile_versions("
            "id,profile_id,version_number,label,content,content_sha256,source_kind,source_ref,"
            "created_at,created_by) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                version_id,
                profile_id,
                next_number,
                label,
                content,
                digest,
                source_kind,
                source_ref,
                now,
                actor,
            ),
        )
        conn.execute(
            "UPDATE instruction_profiles SET updated_at=? WHERE id=?", (now, profile_id)
        )
        return dict(
            conn.execute(
                "SELECT * FROM instruction_profile_versions WHERE id=?", (version_id,)
            ).fetchone()
        )

    def save_version(
        self,
        profile_id: str,
        *,
        label: str,
        content: str,
        actor: str,
        source_kind: str = "manual",
        source_ref: str = "",
    ) -> dict:
        clean_label = _bounded_text(label, field="Version label", maximum=MAX_LABEL_CHARS)
        clean_content = _instruction_text(content)
        clean_actor = _bounded_text(actor, field="Actor", maximum=MAX_NAME_CHARS)
        clean_source_kind = (source_kind or "").strip().lower()
        if clean_source_kind not in SOURCE_KINDS:
            raise InstructionProfileError("Unknown instruction source kind")
        clean_source_ref = (source_ref or "").strip()
        if len(clean_source_ref) > MAX_SOURCE_REF_CHARS or any(
            ord(char) < 32 or ord(char) == 127 for char in clean_source_ref
        ):
            raise InstructionProfileError("Instruction source reference is invalid")
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            version = self._insert_version(
                conn,
                profile_id=profile_id,
                label=clean_label,
                content=clean_content,
                source_kind=clean_source_kind,
                source_ref=clean_source_ref,
                actor=clean_actor,
            )
            conn.execute("COMMIT")
        return version

    def create_profile_from_known_source(
        self,
        *,
        agent_kind: str,
        name: str,
        workspace: str | Path,
        label: str,
        actor: str,
    ) -> dict:
        discovered = self.discover_known_instructions(agent_kind, workspace)
        kind = discovered["agent_kind"]
        clean_name = _bounded_text(name, field="Profile name", maximum=MAX_NAME_CHARS)
        clean_label = _bounded_text(label, field="Version label", maximum=MAX_LABEL_CHARS)
        clean_actor = _bounded_text(actor, field="Actor", maximum=MAX_NAME_CHARS)
        profile_id = f"ip-{secrets.token_hex(12)}"
        now = time.time()
        try:
            with self.store._lock, self.store._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO instruction_profiles(id,agent_kind,name,created_at,updated_at,created_by) "
                    "VALUES(?,?,?,?,?,?)",
                    (profile_id, kind, clean_name, now, now, clean_actor),
                )
                version = self._insert_version(
                    conn,
                    profile_id=profile_id,
                    label=clean_label,
                    content=discovered["content"],
                    source_kind=discovered["source_kind"],
                    source_ref=discovered["source_ref"],
                    actor=clean_actor,
                )
                profile = dict(self._profile_row(conn, profile_id))
                conn.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            raise InstructionConflict("A profile with that name already exists") from exc
        return {"profile": profile, "version": version, "disclosure": self.disclosure}

    def list_versions(self, profile_id: str) -> list[dict]:
        self.get_profile(profile_id)
        with self.store._connect() as conn:
            rows = conn.execute(
                "SELECT id,profile_id,version_number,label,content_sha256,source_kind,source_ref,"
                "created_at,created_by FROM instruction_profile_versions WHERE profile_id=? "
                "ORDER BY version_number DESC",
                (profile_id,),
            )
            return [dict(row) for row in rows]

    def get_version(self, version_id: str) -> dict:
        with self.store._connect() as conn:
            row = conn.execute(
                "SELECT v.*,p.agent_kind,p.name AS profile_name "
                "FROM instruction_profile_versions v JOIN instruction_profiles p "
                "ON p.id=v.profile_id WHERE v.id=?",
                (version_id,),
            ).fetchone()
        if row is None:
            raise InstructionNotFound("Unknown instruction profile version")
        result = dict(row)
        result["disclosure"] = self.disclosure
        return result

    @staticmethod
    def _instance(conn: sqlite3.Connection, instance_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT id,kind,shell_backed FROM instances WHERE id=?", (instance_id,)
        ).fetchone()
        if row is None:
            raise InstructionNotFound("Unknown agent instance")
        kind = row["kind"] or "aeon"
        local_identity_terminal = (
            kind == "terminal" and int(row["shell_backed"] or 0) == 1
        )
        if kind not in AGENT_KINDS and not local_identity_terminal:
            raise InstructionProfileError("Instruction profiles apply only to agent instances")
        return row

    @staticmethod
    def _ensure_binding(conn: sqlite3.Connection, instance_id: str) -> None:
        now = time.time()
        conn.execute(
            "INSERT OR IGNORE INTO instance_instruction_bindings("
            "instance_id,created_at,updated_at) VALUES(?,?,?)",
            (instance_id, now, now),
        )

    @staticmethod
    def _version_for_instance(
        conn: sqlite3.Connection, instance_kind: str, version_id: str | None
    ) -> sqlite3.Row | None:
        if version_id is None:
            return None
        row = conn.execute(
            "SELECT v.*,p.agent_kind,p.name AS profile_name "
            "FROM instruction_profile_versions v JOIN instruction_profiles p "
            "ON p.id=v.profile_id WHERE v.id=?",
            (version_id,),
        ).fetchone()
        if row is None:
            raise InstructionNotFound("Unknown instruction profile version")
        if row["agent_kind"] != instance_kind:
            raise InstructionProfileError(
                "Instruction profile kind does not match the agent instance"
            )
        return row

    @staticmethod
    def _local_content(
        conn: sqlite3.Connection, instance_id: str, revision: int
    ) -> tuple[str, str]:
        if revision == 0:
            return "", _sha256_text("")
        row = conn.execute(
            "SELECT content,content_sha256 FROM instance_local_instruction_versions "
            "WHERE instance_id=? AND revision=?",
            (instance_id, revision),
        ).fetchone()
        if row is None:
            raise InstructionNotFound("Unknown local instruction revision")
        return row["content"], row["content_sha256"]

    def _binding_from_conn(self, conn: sqlite3.Connection, instance_id: str) -> dict:
        instance = self._instance(conn, instance_id)
        row = conn.execute(
            "SELECT * FROM instance_instruction_bindings WHERE instance_id=?",
            (instance_id,),
        ).fetchone()
        if row is None:
            desired_version_id = None
            applied_version_id = None
            desired_local_revision = 0
            applied_local_revision = 0
            created_at = None
            updated_at = None
        else:
            desired_version_id = row["desired_profile_version_id"]
            applied_version_id = row["applied_profile_version_id"]
            desired_local_revision = row["desired_local_revision"]
            applied_local_revision = row["applied_local_revision"]
            created_at = row["created_at"]
            updated_at = row["updated_at"]

        terminal_identity_only = instance["kind"] == "terminal"
        desired_version = None if terminal_identity_only else self._version_for_instance(
            conn, instance["kind"], desired_version_id
        )
        applied_version = None if terminal_identity_only else self._version_for_instance(
            conn, instance["kind"], applied_version_id
        )
        desired_local_content, desired_local_sha = self._local_content(
            conn, instance_id, desired_local_revision
        )
        applied_local_content, applied_local_sha = self._local_content(
            conn, instance_id, applied_local_revision
        )
        return {
            "instance_id": instance_id,
            "agent_kind": instance["kind"],
            "desired_profile_version": dict(desired_version) if desired_version else None,
            "applied_profile_version": dict(applied_version) if applied_version else None,
            "desired_local_revision": desired_local_revision,
            "applied_local_revision": applied_local_revision,
            "desired_local_content": desired_local_content,
            "desired_local_content_sha256": desired_local_sha,
            "applied_local_content": applied_local_content,
            "applied_local_content_sha256": applied_local_sha,
            "base_pending": False if terminal_identity_only else desired_version_id != applied_version_id,
            "local_pending": desired_local_revision != applied_local_revision,
            "pending": (
                (not terminal_identity_only and desired_version_id != applied_version_id)
                or desired_local_revision != applied_local_revision
            ),
            "created_at": created_at,
            "updated_at": updated_at,
            "disclosure": self.disclosure,
        }

    def get_instance_binding(self, instance_id: str) -> dict:
        with self.store._connect() as conn:
            return self._binding_from_conn(conn, instance_id)

    def select_profile_version(
        self, instance_id: str, version_id: str | None
    ) -> dict:
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            instance = self._instance(conn, instance_id)
            if instance["kind"] == "terminal":
                raise InstructionProfileError(
                    "Shared agent-family profiles are unavailable until an agent is active"
                )
            self._version_for_instance(conn, instance["kind"], version_id)
            self._ensure_binding(conn, instance_id)
            conn.execute(
                "UPDATE instance_instruction_bindings SET desired_profile_version_id=?,"
                "updated_at=? WHERE instance_id=?",
                (version_id, time.time(), instance_id),
            )
            result = self._binding_from_conn(conn, instance_id)
            conn.execute("COMMIT")
        return result

    def save_local_role(
        self,
        instance_id: str,
        *,
        content: str,
        expected_revision: int,
        actor: str,
    ) -> dict:
        clean_content = _instruction_text(content)
        clean_actor = _bounded_text(actor, field="Actor", maximum=MAX_NAME_CHARS)
        if not isinstance(expected_revision, int) or expected_revision < 0:
            raise InstructionProfileError("Expected local revision is invalid")
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._instance(conn, instance_id)
            self._ensure_binding(conn, instance_id)
            binding = conn.execute(
                "SELECT desired_local_revision FROM instance_instruction_bindings "
                "WHERE instance_id=?",
                (instance_id,),
            ).fetchone()
            current_revision = int(binding["desired_local_revision"])
            if current_revision != expected_revision:
                raise InstructionConflict(
                    f"Local instructions changed; current revision is {current_revision}"
                )
            current_content, _digest = self._local_content(
                conn, instance_id, current_revision
            )
            if current_content != clean_content:
                next_revision = current_revision + 1
                now = time.time()
                conn.execute(
                    "INSERT INTO instance_local_instruction_versions("
                    "instance_id,revision,content,content_sha256,created_at,created_by) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        instance_id,
                        next_revision,
                        clean_content,
                        _sha256_text(clean_content),
                        now,
                        clean_actor,
                    ),
                )
                conn.execute(
                    "UPDATE instance_instruction_bindings SET desired_local_revision=?,"
                    "updated_at=? WHERE instance_id=?",
                    (next_revision, now, instance_id),
                )
            result = self._binding_from_conn(conn, instance_id)
            conn.execute("COMMIT")
        return result

    def list_local_role_versions(self, instance_id: str) -> list[dict]:
        with self.store._connect() as conn:
            self._instance(conn, instance_id)
            rows = conn.execute(
                "SELECT instance_id,revision,content_sha256,created_at,created_by "
                "FROM instance_local_instruction_versions WHERE instance_id=? "
                "ORDER BY revision DESC",
                (instance_id,),
            )
            return [dict(row) for row in rows]

    def get_local_role_version(self, instance_id: str, revision: int) -> dict:
        if not isinstance(revision, int) or revision < 0:
            raise InstructionProfileError("Local revision is invalid")
        with self.store._connect() as conn:
            self._instance(conn, instance_id)
            content, digest = self._local_content(conn, instance_id, revision)
            if revision == 0:
                return {
                    "instance_id": instance_id,
                    "revision": 0,
                    "content": content,
                    "content_sha256": digest,
                    "created_at": None,
                    "created_by": None,
                }
            row = conn.execute(
                "SELECT * FROM instance_local_instruction_versions "
                "WHERE instance_id=? AND revision=?",
                (instance_id, revision),
            ).fetchone()
            return dict(row)

    def select_local_role_version(self, instance_id: str, revision: int) -> dict:
        if not isinstance(revision, int) or revision < 0:
            raise InstructionProfileError("Local revision is invalid")
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._instance(conn, instance_id)
            self._local_content(conn, instance_id, revision)
            self._ensure_binding(conn, instance_id)
            conn.execute(
                "UPDATE instance_instruction_bindings SET desired_local_revision=?,updated_at=? "
                "WHERE instance_id=?",
                (revision, time.time(), instance_id),
            )
            result = self._binding_from_conn(conn, instance_id)
            conn.execute("COMMIT")
        return result

    def launch_snapshot(self, instance_id: str) -> dict:
        """Return the exact desired revisions for a launcher to apply privately."""

        binding = self.get_instance_binding(instance_id)
        desired = binding["desired_profile_version"]
        return {
            "instance_id": instance_id,
            "agent_kind": binding["agent_kind"],
            "profile_version_id": desired["id"] if desired else None,
            "profile_content": desired["content"] if desired else "",
            "profile_content_sha256": desired["content_sha256"] if desired else _sha256_text(""),
            "local_revision": binding["desired_local_revision"],
            "local_content": binding["desired_local_content"],
            "local_content_sha256": binding["desired_local_content_sha256"],
            "disclosure": self.disclosure,
        }

    def launch_snapshot_for_agent_kind(
        self,
        instance_id: str,
        *,
        agent_kind: str,
        preserve_profile: bool,
    ) -> dict:
        """Prepare a terminal's target-agent layers without changing its mode.

        Same-tab activation must complete provider and instruction preflight
        before it commits ``terminal -> agent``. The persistent local-role layer
        is valid across agent families; a base profile is retained only for the
        same family and otherwise rendered empty until the eventual atomic mode
        transition clears the incompatible binding.
        """

        kind = _agent_kind(agent_kind)
        with self.store._connect() as conn:
            instance = conn.execute(
                "SELECT id FROM instances WHERE id=?", (instance_id,)
            ).fetchone()
            if instance is None:
                raise InstructionNotFound("Unknown agent instance")
            row = conn.execute(
                "SELECT * FROM instance_instruction_bindings WHERE instance_id=?",
                (instance_id,),
            ).fetchone()
            desired_version_id = (
                row["desired_profile_version_id"]
                if row is not None and preserve_profile
                else None
            )
            local_revision = int(row["desired_local_revision"]) if row else 0
            desired = self._version_for_instance(conn, kind, desired_version_id)
            local_content, local_sha = self._local_content(
                conn, instance_id, local_revision
            )
        return {
            "instance_id": instance_id,
            "agent_kind": kind,
            "profile_version_id": desired["id"] if desired else None,
            "profile_content": desired["content"] if desired else "",
            "profile_content_sha256": (
                desired["content_sha256"] if desired else _sha256_text("")
            ),
            "local_revision": local_revision,
            "local_content": local_content,
            "local_content_sha256": local_sha,
            "disclosure": self.disclosure,
        }

    def mark_applied(
        self,
        instance_id: str,
        *,
        profile_version_id: str | None,
        local_revision: int,
    ) -> dict:
        """Record an exact launcher-confirmed snapshot, even if edits are now pending."""

        if not isinstance(local_revision, int) or local_revision < 0:
            raise InstructionProfileError("Applied local revision is invalid")
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            instance = self._instance(conn, instance_id)
            self._version_for_instance(conn, instance["kind"], profile_version_id)
            self._local_content(conn, instance_id, local_revision)
            self._ensure_binding(conn, instance_id)
            conn.execute(
                "UPDATE instance_instruction_bindings SET applied_profile_version_id=?,"
                "applied_local_revision=?,updated_at=? WHERE instance_id=?",
                (profile_version_id, local_revision, time.time(), instance_id),
            )
            result = self._binding_from_conn(conn, instance_id)
            conn.execute("COMMIT")
        return result
