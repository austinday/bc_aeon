import hashlib
import hmac
import fcntl
import os
import re
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional
from aeon.core.paths import PROJECT_ROOT
from aeon.core.skills.knowledge import SkillKnowledgeError, SkillKnowledgeStore
from aeon.core.skills.lifecycle import (
    LearnedSkillError,
    LearnedSkillStore,
    MAX_PRIVATE_SKILLS,
)

INSTANCE_SKILLS_DIR_ENV = "AEON_INSTANCE_SKILLS_DIR"
MAX_SKILL_CONTENT_BYTES = 64 * 1024


class SkillContentError(ValueError):
    """A skill protocol cannot be loaded as trusted bounded context."""


class SkillContentTooLarge(SkillContentError):
    """A skill exceeds the bounded advisory-context byte ceiling."""


def _safe_component(value: str) -> bool:
    return bool(
        value
        and not value.startswith(".")
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value)
    )


def _has_skill_content(directory: Path) -> bool:
    """Return True if `directory` exists and contains at least one .txt skill file."""
    try:
        if not directory.is_dir():
            return False
        return any(directory.rglob("*.txt"))
    except OSError:
        return False


class SkillsManager:
    """
    Manages retrieval of skill protocols from the filesystem.

    The skills directory is resolved relative to the INSTALLED package location
    (the directory this file lives in), so it works regardless of the current
    working directory and regardless of whether aeon is run from a source
    checkout or a pip install. Python resolves __file__ to wherever the package
    physically is, so this is the same mechanism the prompts loader already uses.

    Resolution order (first candidate that actually contains .txt files wins):
      1. AEON_SKILLS_DIR env override (explicit pointer; also useful for live
         editing of skills without reinstalling).
      2. Package-relative: the directory containing this file. Correct for both
         source checkouts and pip installs.
      3. cwd-relative: covers running from a source checkout while aeon was
         imported from a stale/incomplete site-packages copy.
      4. PROJECT_ROOT-relative: legacy last resort.
    """

    def __init__(self, *, instance_dir: str | os.PathLike[str] | None = None):
        package_skills = Path(__file__).resolve().parent

        candidates = []
        env_dir = os.environ.get("AEON_SKILLS_DIR")
        if env_dir:
            candidates.append(Path(env_dir).expanduser())
        candidates.append(package_skills)
        candidates.append(Path.cwd() / "aeon" / "core" / "skills")
        candidates.append(PROJECT_ROOT / "aeon" / "core" / "skills")

        self.base_dir = None
        for candidate in candidates:
            if _has_skill_content(candidate):
                self.base_dir = candidate.resolve()
                break

        if self.base_dir is None:
            # No skill .txt files found anywhere. Still point at the package
            # directory (the correct location) so the agent keeps running; the
            # per-call methods below degrade gracefully to empty results. This
            # almost always means package data was not installed.
            self.base_dir = package_skills
            print(
                f"[WARNING] SkillsManager found no skill .txt files. Defaulting to "
                f"package directory: {self.base_dir}. If skills are missing, reinstall "
                f"aeon (pip install .) so packaged skills ship to site-packages, or set "
                f"the AEON_SKILLS_DIR environment variable to your skills directory.",
                file=sys.stderr,
            )

        # Managed Nexus agents receive a server-derived private overlay. Skills
        # authored at runtime are written only there, while packaged skills stay
        # readable as shared, immutable defaults. A process without this explicit
        # capability remains read-only; runtime tools never fall back to the
        # packaged catalog.
        explicit_instance_dir = str(instance_dir or "").strip()
        instance_dir_value = explicit_instance_dir or os.environ.get(
            INSTANCE_SKILLS_DIR_ENV, ""
        ).strip()
        if not instance_dir_value:
            # A managed agent that exec-restarts itself predating the explicit
            # overlay variable still retains both server-derived identities.
            # Recover only when the transcript parent exactly matches the
            # durable instance ID; arbitrary standalone environment values do
            # not gain a writable path through this compatibility bridge.
            remote_id = os.environ.get("AEON_REMOTE_INSTANCE_ID", "").strip()
            transcript = os.environ.get("AEON_CHAT_TRANSCRIPT_PATH", "").strip()
            if re.fullmatch(r"[0-9a-f]{32}", remote_id) and transcript:
                transcript_path = Path(transcript).expanduser()
                if transcript_path.parent.name == remote_id:
                    instance_dir_value = str(transcript_path.parent / "skills")
        # Preserve lexical identity.  Resolving here would turn a symlinked
        # writable overlay into an apparently ordinary target before the private
        # storage checks can reject it.
        self.instance_dir = (
            Path(instance_dir_value).expanduser().absolute()
            if instance_dir_value
            else None
        )
        self.mutable_dir = self.instance_dir
        self.knowledge_dir = (
            self.instance_dir.parent / "skill-wiki" if self.instance_dir else None
        )

    def list_categories(self) -> List[str]:
        """Return the union of shared and current-instance skill categories."""
        categories = set()
        for root in (self.base_dir, self.instance_dir):
            if root is None:
                continue
            try:
                categories.update(
                    entry.name
                    for entry in root.iterdir()
                    if entry.is_dir()
                    and not entry.is_symlink()
                    and _safe_component(entry.name)
                    and any(entry.glob("*.txt"))
                )
            except OSError:
                continue
        return sorted(categories)

    def get_skills_in_category(self, category_path: str) -> List[str]:
        """
        Returns a list of skill names (filenames without .txt) in the given category.
        """
        if not _safe_component(category_path):
            return []
        try:
            skills = set()
            for root in (self.base_dir, self.instance_dir):
                if root is None:
                    continue
                cat_dir = root / category_path
                if cat_dir.is_dir() and not cat_dir.is_symlink():
                    skills.update(
                        f.stem for f in cat_dir.glob("*.txt") if _safe_component(f.stem)
                    )
            return sorted(skills)
        except Exception as e:
            print(f"[ERROR] SkillsManager.get_skills_in_category failed: {e}", file=sys.stderr)
            return []

    def get_skill_content(self, category_path: str, skill_name: str) -> Optional[str]:
        """
        Reads the content of a specific skill protocol file.
        """
        if not _safe_component(category_path) or not _safe_component(skill_name):
            raise SkillContentError("skill category and name must be safe path components")
        try:
            # The current agent's version overrides a packaged protocol of the
            # same name without altering what any other instance sees.
            for root in (self.instance_dir, self.base_dir):
                if root is None:
                    continue
                if root == self.instance_dir:
                    try:
                        root_metadata = root.lstat()
                    except FileNotFoundError:
                        continue
                    if (
                        not stat.S_ISDIR(root_metadata.st_mode)
                        or root_metadata.st_uid != os.geteuid()
                        or stat.S_IMODE(root_metadata.st_mode) != 0o700
                        or root.resolve(strict=True) != root.absolute()
                    ):
                        raise SkillContentError("private skill storage is not owner-safe")
                canonical_root = root.resolve(strict=True)
                category_dir = canonical_root / category_path
                skill_file = category_dir / f"{skill_name}.txt"
                try:
                    if category_dir.is_symlink() or skill_file.is_symlink():
                        raise SkillContentError("skill paths may not contain symlinks")
                    category_metadata = category_dir.lstat()
                    metadata = skill_file.lstat()
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise SkillContentError("skill protocol is not a regular file")
                if root == self.instance_dir and (
                    not stat.S_ISDIR(category_metadata.st_mode)
                    or category_metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(category_metadata.st_mode) != 0o700
                    or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_nlink != 1
                ):
                    raise SkillContentError("private skill protocol is not owner-private")
                canonical_file = skill_file.resolve(strict=True)
                try:
                    canonical_file.relative_to(canonical_root)
                except ValueError as exc:
                    raise SkillContentError("skill protocol escapes its skill root") from exc
                if metadata.st_size > MAX_SKILL_CONTENT_BYTES:
                    raise SkillContentTooLarge(
                        f"skill protocol is {metadata.st_size} bytes; maximum is "
                        f"{MAX_SKILL_CONTENT_BYTES} bytes"
                    )
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(skill_file, flags)
                try:
                    opened = os.fstat(descriptor)
                    if (
                        opened.st_dev != metadata.st_dev
                        or opened.st_ino != metadata.st_ino
                        or not stat.S_ISREG(opened.st_mode)
                    ):
                        raise SkillContentError("skill protocol changed while opening")
                    with os.fdopen(descriptor, "rb", closefd=False) as stream:
                        payload = stream.read(MAX_SKILL_CONTENT_BYTES + 1)
                    if len(payload) > MAX_SKILL_CONTENT_BYTES:
                        raise SkillContentTooLarge(
                            f"skill protocol exceeds the {MAX_SKILL_CONTENT_BYTES}-byte maximum"
                        )
                finally:
                    os.close(descriptor)
                return payload.decode("utf-8").strip()
        except SkillContentError:
            raise
        except (OSError, UnicodeError) as e:
            print(f"[ERROR] SkillsManager.get_skill_content failed: {e}", file=sys.stderr)
        return None

    def get_mutable_skill_file(self, category_path: str, skill_name: str) -> Path:
        """Return the only path runtime CRUD tools may mutate."""
        if not _safe_component(category_path) or not _safe_component(skill_name):
            raise SkillContentError(
                "skill category and name must be safe path components"
            )
        if self.mutable_dir is None:
            raise SkillContentError(
                "runtime skill changes require an agent-specific private overlay"
            )
        return self.mutable_dir / category_path / f"{skill_name}.txt"

    def ensure_private_overlay(self) -> Path:
        """Create and verify the exact owner-private runtime skill root."""

        root = self.instance_dir
        if root is None:
            raise SkillContentError(
                "runtime skill changes require an agent-specific private overlay"
            )
        try:
            root.mkdir(parents=True, mode=0o700, exist_ok=True)
            metadata = root.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or root.resolve(strict=True) != root.absolute()
            ):
                raise SkillContentError("private skill storage is not owner-safe")
            os.chmod(root, 0o700, follow_symlinks=False)
        except SkillContentError:
            raise
        except OSError as exc:
            raise SkillContentError("private skill storage is unavailable") from exc
        return root

    @contextmanager
    def state_lock(self):
        """Lock this agent's protocol, lifecycle, and wiki state transaction."""

        root = self.ensure_private_overlay()
        root_fd = None
        lock_fd = None
        try:
            root_metadata = root.lstat()
            root_fd = os.open(
                root,
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
                raise SkillContentError(
                    "private skill storage changed while opening"
                )
            lock_fd = os.open(
                ".agent-state.lock",
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
                raise SkillContentError("private skill state lock is not owner-safe")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        except SkillContentError:
            raise
        except OSError as exc:
            raise SkillContentError("private skill state lock is unavailable") from exc
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(lock_fd)
            if root_fd is not None:
                os.close(root_fd)

    def knowledge_store(self) -> SkillKnowledgeStore:
        """Return this managed instance's durable skill-development wiki."""

        return SkillKnowledgeStore(self.knowledge_dir)

    def learned_store(self) -> LearnedSkillStore:
        """Return lifecycle metadata for this exact agent's learned protocols."""

        return LearnedSkillStore(self.instance_dir)

    def private_skill_count(self) -> int:
        """Count bounded private protocols without following symlinks."""

        if self.instance_dir is None:
            return 0
        try:
            self.ensure_private_overlay()
            count = 0
            for category in self.instance_dir.iterdir():
                if not _safe_component(category.name) or not category.is_dir() or category.is_symlink():
                    continue
                count += sum(
                    1
                    for path in category.glob("*.txt")
                    if _safe_component(path.stem) and path.is_file() and not path.is_symlink()
                )
                if count > MAX_PRIVATE_SKILLS:
                    break
            return count
        except SkillContentError:
            raise
        except OSError as exc:
            raise SkillContentError(
                "private skill catalog could not be counted safely"
            ) from exc

    def get_skill_record(self, category_path: str, skill_name: str) -> Optional[dict[str, object]]:
        """Read one effective protocol with explicit origin and lifecycle state."""

        content = self.get_skill_content(category_path, skill_name)
        if not content:
            return None
        revision = hashlib.sha256(content.encode("utf-8")).hexdigest()
        private_path = (
            self.instance_dir / category_path / f"{skill_name}.txt"
            if self.instance_dir is not None
            else None
        )
        scope = "private" if private_path is not None and private_path.is_file() else "shared"
        lifecycle = None
        if scope == "private":
            try:
                lifecycle = self.learned_store().read(
                    category_path,
                    skill_name,
                    current_content_revision=revision,
                )
                if lifecycle is None:
                    lifecycle = {
                        "status": "needs_review",
                        "metadata_stale": False,
                        "evidence_stale": True,
                        "integrity_ambiguous": False,
                        "error": (
                            "this private skill predates the earned-skill lifecycle and "
                            "must be revised from verified evidence"
                        ),
                    }
                else:
                    evidence_stale = False
                    for reference in lifecycle.get("evidence", []):
                        note = self.knowledge_store().read_note(reference["note_id"])
                        learning = note.get("learning") if note else None
                        if (
                            note is None
                            or not hmac.compare_digest(
                                str(note.get("revision") or ""),
                                str(reference.get("revision") or ""),
                            )
                            or note.get("skill_evidence_eligible") is not True
                            or not isinstance(learning, dict)
                            or learning.get("candidate_skill_path")
                            != f"{category_path}/{skill_name}"
                        ):
                            evidence_stale = True
                            break
                    lifecycle["evidence_stale"] = evidence_stale
                    if evidence_stale and lifecycle.get("status") == "ready":
                        lifecycle["status"] = "needs_review"
            except (LearnedSkillError, SkillKnowledgeError) as exc:
                lifecycle = {
                    "status": "needs_review",
                    "metadata_stale": True,
                    "evidence_stale": True,
                    "integrity_ambiguous": True,
                    "error": str(exc),
                }
        return {
            "category": category_path,
            "name": skill_name,
            "skill_path": f"{category_path}/{skill_name}",
            "content": content,
            "revision": revision,
            "scope": scope,
            "editable": scope == "private",
            "transferable": scope == "private",
            "overrides_shared": bool(
                scope == "private"
                and (self.base_dir / category_path / f"{skill_name}.txt").is_file()
            ),
            "lifecycle": lifecycle,
        }

    def list_effective_skills(self) -> list[dict[str, object]]:
        """Return shared plus private effective protocols with explicit origin.

        A private protocol with the same path intentionally overrides the shared
        package protocol for this one agent, matching ``get_skill_content``.
        """

        records: dict[str, dict[str, object]] = {}
        for scope, root in (("shared", self.base_dir), ("private", self.instance_dir)):
            if root is None:
                continue
            try:
                categories = sorted(root.iterdir(), key=lambda path: path.name)
            except OSError:
                continue
            for category_dir in categories:
                if (
                    not _safe_component(category_dir.name)
                    or not category_dir.is_dir()
                    or category_dir.is_symlink()
                ):
                    continue
                try:
                    files = sorted(category_dir.glob("*.txt"), key=lambda path: path.name)
                except OSError:
                    continue
                for path in files:
                    if not _safe_component(path.stem):
                        continue
                    skill_path = f"{category_dir.name}/{path.stem}"
                    try:
                        content = self.get_skill_content(category_dir.name, path.stem)
                    except SkillContentError:
                        continue
                    # During the shared pass a private override would make
                    # get_skill_content return private text. Delay that path to
                    # the private pass so its origin remains truthful.
                    private_override = bool(
                        scope == "shared"
                        and self.instance_dir is not None
                        and (self.instance_dir / category_dir.name / path.name).is_file()
                    )
                    if private_override or not content:
                        continue
                    record = self.get_skill_record(category_dir.name, path.stem)
                    if record is not None:
                        records[skill_path] = record
        return [records[key] for key in sorted(records)]
