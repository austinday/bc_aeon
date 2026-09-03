"""
Runtime skill activation tools.

A "skill" is a revision-bound advisory playbook distilled from verified experience.
These tools keep learned procedures useful without turning them into authority:

  - activate_skill   -> loads a ready playbook for the current request.
  - deactivate_skill -> records its outcome and unpins it.
  - create_skill     -> requires low-uncertainty failure/recovery evidence.
  - wiki tools       -> maintain searchable, revision-checked working knowledge.

Both are BaseTool subclasses, so the dynamic loader picks them up automatically
(the `worker` dependency is supplied by main.py's loader deps). No manual
registration is required.
"""

import difflib
import hashlib
import hmac
import json
import os
import re
import uuid

from aeon.tools.base import BaseTool
from aeon.core.skills.manager import (
    MAX_SKILL_CONTENT_BYTES,
    SkillContentError,
    SkillContentTooLarge,
    SkillsManager,
)
from aeon.core.skills.lifecycle import (
    LearnedSkillError,
    MAX_PRIVATE_SKILLS,
    VALID_OUTCOMES,
    skill_revision,
)
from aeon.core.skills.knowledge import (
    MAX_SKILL_KNOWLEDGE_BYTES,
    SKILL_NOTE_ID_RE,
    SKILL_PATH_RE,
    SkillKnowledgeError,
    contains_persisted_secret,
)


def _safe_component(name: str) -> bool:
    """True if `name` is a safe single path component for a skill category/name:
    1-80 characters, alphanumeric first, then alphanumeric, dash, or underscore."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return False
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        return False
    return True


REQUIRED_LEARNED_SKILL_SECTIONS = (
    "when to use",
    "preconditions",
    "procedure",
    "verification",
    "stop or adapt",
)
_SKILL_LIFECYCLE_TOOLS = frozenset(
    {
        "activate_skill",
        "deactivate_skill",
        "create_skill",
        "delete_skill",
        "read_skill",
        "remember_skill_knowledge",
        "list_skill_knowledge",
        "read_skill_knowledge",
        "search_skill_knowledge",
        "delete_skill_knowledge",
    }
)


def _protocol_sections(content: str) -> dict[str, str]:
    """Return the five canonical learned-protocol sections when unambiguous."""

    text = str(content or "").replace("\r\n", "\n").replace("\r", "\n")
    selected = []
    for match in re.finditer(r"(?m)^#{1,6}\s+([^#\n]+?)\s*$", text):
        name = match.group(1).strip().casefold()
        if name in REQUIRED_LEARNED_SKILL_SECTIONS:
            selected.append((name, match.start(), match.end()))
    if [item[0] for item in selected] != list(REQUIRED_LEARNED_SKILL_SECTIONS):
        return {}
    sections = {}
    for index, (name, _start, end) in enumerate(selected):
        next_start = selected[index + 1][1] if index + 1 < len(selected) else len(text)
        sections[name] = text[end:next_start].strip()
    return sections


def _protocol_error(content: str) -> str:
    """Return why a proposed learned playbook cannot remain adaptable."""

    sections = _protocol_sections(content)
    if not sections:
        return (
            "content must contain each required heading exactly once and in this order: "
            + ", ".join(REQUIRED_LEARNED_SKILL_SECTIONS)
        )
    empty = [name for name, body in sections.items() if not body]
    if empty:
        return "content has empty required section(s): " + ", ".join(empty)
    return ""


def _protocol_evidence_error(
    sm: SkillsManager, content: str, references: list[dict]
) -> str:
    """Bind the shortcut itself to the newest cited recovered procedure."""

    notes = []
    for reference in references:
        note = sm.knowledge_store().read_note(reference["note_id"])
        if note is not None:
            notes.append(note)
    if not notes:
        return "the cited recovery evidence is unavailable"
    newest = max(
        notes,
        key=lambda item: (float(item.get("updated_at") or 0.0), str(item.get("id") or "")),
    )
    learning = newest.get("learning") or {}
    sections = _protocol_sections(content)
    expected_procedure = str(learning.get("procedure") or "").strip()
    expected_verification = str(learning.get("verification") or "").strip()
    if sections.get("procedure") != expected_procedure:
        return (
            "the Procedure section must exactly match the newest cited learning note's "
            "verified procedure"
        )
    if sections.get("verification") != expected_verification:
        return (
            "the Verification section must exactly match the newest cited learning note's "
            "verification method"
        )
    return ""


def _experience_from_worker(worker) -> dict | None:
    """Capture recent factual receipts already observed in this exact request.

    Only a genuine tool failure followed by a later successful result counts as
    earned recovery.  Policy refusals and idempotent no-ops are useful context,
    but they are not trial-and-error evidence for minting a skill.
    """

    contract = getattr(worker, "request_contract", None)
    request_id = str(getattr(contract, "request_id", "") or "")
    results = list(getattr(contract, "results", ()) or ())
    receipts = []
    for result in results:
        tool_name = str(getattr(result, "tool_name", "") or "")
        if not tool_name or tool_name in _SKILL_LIFECYCLE_TOOLS:
            continue
        raw_status = getattr(result, "status", "")
        status_value = str(getattr(raw_status, "value", raw_status) or "")
        if status_value in {"pending", "skipped"}:
            continue
        summary = str(getattr(result, "summary", "") or "")
        receipts.append(
            {
                "tool": tool_name[:200],
                "status": status_value,
                "error_code": str(getattr(result, "error_code", "") or "")[:100],
                "summary_sha256": hashlib.sha256(summary.encode("utf-8")).hexdigest(),
            }
        )
    # Keep the evidence record bounded and bias it toward the observations
    # nearest the learning decision.  Recompute every counter over exactly the
    # stored window so long requests cannot produce internally inconsistent
    # notes.
    receipts = receipts[-64:]
    if not request_id or not receipts:
        return None
    failure_indexes = [
        index for index, receipt in enumerate(receipts)
        if receipt["status"] == "failed"
    ]
    success_indexes = [
        index for index, receipt in enumerate(receipts)
        if receipt["status"] == "ok"
    ]
    recovered = any(failure < success for failure in failure_indexes for success in success_indexes)
    return {
        "request_id": request_id,
        "attempt_count": len(receipts),
        "failure_count": len(failure_indexes),
        "success_count": len(success_indexes),
        "recovered_after_failure": recovered,
        "receipts": receipts,
    }


def _validated_skill_evidence(sm: SkillsManager, skill_path: str, references: object) -> list[dict]:
    if not isinstance(references, list) or not 1 <= len(references) <= 16:
        raise SkillKnowledgeError("evidence must contain 1-16 note references")
    validated = []
    seen = set()
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {"note_id", "revision"}:
            raise SkillKnowledgeError("each evidence reference needs note_id and revision")
        note_id = str(reference.get("note_id") or "")
        revision = str(reference.get("revision") or "")
        if note_id in seen:
            raise SkillKnowledgeError("an evidence note was supplied more than once")
        seen.add(note_id)
        note = sm.knowledge_store().read_note(note_id)
        if note is None:
            raise SkillKnowledgeError(f"evidence note '{note_id}' no longer exists")
        if revision != str(note.get("revision") or ""):
            raise SkillKnowledgeError(f"evidence note '{note_id}' changed since it was read")
        learning = note.get("learning") or {}
        if learning.get("candidate_skill_path") != skill_path:
            raise SkillKnowledgeError(
                f"evidence note '{note_id}' belongs to a different candidate skill"
            )
        if note.get("skill_evidence_eligible") is not True:
            raise SkillKnowledgeError(
                f"evidence note '{note_id}' does not prove low-uncertainty recovery after failure"
            )
        validated.append({"note_id": note_id, "revision": revision})
    return validated


def _atomic_skill_write(path, payload: bytes) -> None:
    """Publish one owner-private skill without following a link."""

    category_dir = path.parent
    category_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    if category_dir.is_symlink():
        raise SkillContentError("private skill category may not be a symlink")
    os.chmod(category_dir, 0o700, follow_symlinks=False)
    temporary = category_dir / f".{path.name}.{uuid.uuid4().hex}.tmp"
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
                raise SkillContentError("could not write private skill")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        os.chmod(path, 0o600, follow_symlinks=False)
        directory_fd = os.open(
            category_dir,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except (FileNotFoundError, OSError):
            pass


def _all_skill_paths(sm):
    """Best-effort list of every '<category>/<skill>' path (empty on failure)."""
    paths = []
    try:
        for cat in sm.list_categories():
            for skill in sm.get_skills_in_category(cat):
                paths.append(f"{cat}/{skill}")
    except Exception:
        pass
    return paths


def _not_found_msg(sm, skill_path, category):
    """Shared 'no such skill' message: list siblings in the category if it exists,
    else suggest the closest real paths across all categories."""
    available = sm.get_skills_in_category(category)
    if available:
        return (f"Error: no skill '{skill_path}'. Available in '{category}': "
                f"{', '.join(sorted(available))}")
    close = difflib.get_close_matches(skill_path, _all_skill_paths(sm), n=3, cutoff=0.4)
    hint = (f" Did you mean: {', '.join(close)}?" if close
            else " Check the SKILLS section for valid '<category>/<skill_name>' paths.")
    return f"Error: no skill found at '{skill_path}'.{hint}"


class ActivateSkillTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(
            name="activate_skill",
            description=(
                "Load an optional evidence-informed playbook for the CURRENT task. Skills are never applied "
                "automatically and never override current evidence, workspace policy, or the user's request. "
                "Before activating, inspect applicability and preconditions. While active, adapt or deactivate "
                "as soon as live results differ; never repeat a disproven step just because it appears in a skill.\n"
                "Schema:\n"
                "  skill_path (str, required): '<category>/<skill_name>', e.g. 'research/deep_research'.\n"
                "Example: {\"tool_name\": \"activate_skill\", \"parameters\": {\"skill_path\": \"research/deep_research\"}}"
            )
        )
        self.worker = worker

    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "skill_path": {
                    "type": "string",
                    "pattern": SKILL_PATH_RE.pattern,
                }
            },
            "required": ["skill_path"],
            "additionalProperties": False,
        }

    def execute(self, skill_path: str = None, **kwargs) -> str:
        if not self.worker:
            return "Error: Worker context missing."
        if not skill_path or "/" not in skill_path:
            return "Error: skill_path must be '<category>/<skill_name>' (e.g. 'research/deep_research')."
        request_id = str(
            getattr(getattr(self.worker, "request_contract", None), "request_id", "")
            or getattr(self.worker, "request_id", "")
            or ""
        )
        if not request_id:
            return "Error: a skill can be activated only inside a current request contract."

        category, _, skill_name = skill_path.partition("/")
        if not _safe_component(category) or not _safe_component(skill_name):
            return f"Error: invalid skill_path '{skill_path}'."
        active = getattr(self.worker, "active_skill", None)
        if active:
            active_path = str(active.get("path") or "unknown")
            if active.get("paused"):
                return (
                    f"Error: skill '{active_path}' is paused after contrary evidence. First call "
                    "deactivate_skill with an honest outcome; revise it from fresh evidence before reuse if needed."
                )
            return (
                f"Error: skill '{active_path}' is already active. Deactivate it with an honest "
                "outcome before activating another playbook so usage evidence is not discarded."
            )
        sm = SkillsManager()
        try:
            record = sm.get_skill_record(category, skill_name)
        except SkillContentTooLarge as exc:
            return (
                f"Error: Skill '{skill_path}' cannot be activated because {exc}. "
                "Shorten the protocol before trying again."
            )
        except SkillContentError as exc:
            return f"Error: Skill '{skill_path}' failed integrity checks: {exc}."
        if not record:
            available = sm.get_skills_in_category(category)
            if available:
                return (
                    f"Error: Skill '{skill_name}' not found in category '{category}'. "
                    f"Available in '{category}': {', '.join(sorted(available))}"
                )
            # Category itself is likely mistyped — suggest the closest real paths.
            close = difflib.get_close_matches(skill_path, _all_skill_paths(sm), n=3, cutoff=0.4)
            hint = f" Did you mean: {', '.join(close)}?" if close else \
                   " Check the SKILLS section for valid '<category>/<skill_name>' paths."
            return f"Error: No skill found at '{skill_path}'.{hint}"

        content = str(record["content"])
        digest = str(record["revision"])
        scope = str(record["scope"])
        lifecycle_status = "shared"
        if scope == "private":
            try:
                with sm.state_lock():
                    locked_record = sm.get_skill_record(category, skill_name)
                    if (
                        locked_record is None
                        or locked_record.get("scope") != "private"
                    ):
                        return (
                            f"Error: learned skill '{skill_path}' changed while it was being activated."
                        )
                    record = locked_record
                    content = str(record["content"])
                    digest = str(record["revision"])
                    lifecycle = record.get("lifecycle") or {}
                    lifecycle_status = str(
                        lifecycle.get("status") or "needs_review"
                    )
                    if (
                        lifecycle_status != "ready"
                        or lifecycle.get("metadata_stale")
                    ):
                        return (
                            f"Error: learned skill '{skill_path}' is {lifecycle_status}, not ready. "
                            "Read it and its wiki evidence, then revise it with create_skill using fresh "
                            "failed-then-successful evidence; do not force activation."
                        )
                    _validated_skill_evidence(
                        sm, skill_path, list(lifecycle.get("evidence") or [])
                    )
                    sm.learned_store().record_activation(
                        category=category,
                        skill_name=skill_name,
                        content_revision=digest,
                    )
            except (SkillContentError, LearnedSkillError, SkillKnowledgeError) as exc:
                return f"Error: learned skill '{skill_path}' could not be activated: {exc}"
        self.worker.active_skill = {
            "path": skill_path,
            "content": content,
            "sha256": digest,
            "scope": scope,
            "status": lifecycle_status,
            "paused": False,
            "request_id": str(
                request_id
            ),
        }
        self.worker.expanded_categories.add(f"skill:{category}")

        print(f"{self.C_GREEN}\U0001F3AF SKILL ACTIVATED: {skill_path} \u2014 advisory playbook loaded.{self.C_RESET}")
        return (
            f"Skill '{skill_path}' is now active as ADVISORY prior experience. Check its preconditions "
            "against live state on every step. Deactivate with an outcome when done, inapplicable, or contradicted.\n\n"
            f"--- ACTIVE PLAYBOOK: {skill_path} ---\n{content}"
        )


class CreateSkillTool(BaseTool):
    """Distill an earned shortcut into this agent's private skill overlay."""

    def __init__(self, worker=None):
        super().__init__(
            name="create_skill",
            description=(
                "Create or revise a learned playbook only after this agent had to recover from a failed "
                "approach and then verified a stable, low-uncertainty procedure that is likely to recur "
                "and materially shortens future work. Do not create one for every small or one-off task. "
                "A complicated first-try "
                "success belongs in the searchable skill wiki, not in a skill. First save an eligible "
                "remember_skill_knowledge note; pass its exact ID and revision as evidence. Skills remain "
                "advisory, per-agent, revision-bound, and limited in number. One recovery episode may earn "
                "at most one coherent skill; keep extra findings in the wiki. The markdown must contain exact "
                "headings: When to use, Preconditions, Procedure, Verification, and Stop or adapt. Never "
                "store secrets. Procedure and Verification must exactly match those structured fields in "
                "the newest cited learning note. Revisions require the current skill revision and fresh evidence."
            )
        )
        self.worker = worker

    def parameter_schema(self) -> dict:
        """Expose required authoring inputs to constrained decoding.

        ``execute`` retains defensive defaults for direct/legacy callers, but
        those defaults are not valid model calls.  In particular, an omitted
        protocol body must be rejected before this mutation tool is entered.
        """

        component = {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
            "pattern": r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$",
        }
        evidence_ref = {
            "type": "object",
            "properties": {
                "note_id": {"type": "string", "pattern": SKILL_NOTE_ID_RE.pattern},
                "revision": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
            },
            "required": ["note_id", "revision"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "category": dict(component),
                "skill_name": dict(component),
                "content": {
                    "type": "string",
                    "minLength": 1,
                    # A character ceiling is safe at decode time; the runtime
                    # byte check below remains authoritative for UTF-8 text.
                    "maxLength": MAX_SKILL_CONTENT_BYTES,
                },
                "evidence": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 16,
                    "items": evidence_ref,
                },
                "overwrite": {"type": "boolean"},
                "expected_revision": {
                    "type": "string",
                    "pattern": r"^[0-9a-f]{64}$",
                },
            },
            "required": ["category", "skill_name", "content", "evidence"],
            "additionalProperties": False,
        }

    def validate_parameters(self, parameters) -> str:
        error = super().validate_parameters(parameters)
        if error:
            return error
        category = parameters["category"]
        skill_name = parameters["skill_name"]
        content = parameters["content"]
        if not _safe_component(category):
            return "category must be a safe, non-empty path component"
        if not _safe_component(skill_name):
            return "skill_name must be a safe, non-empty path component"
        if not content.strip():
            return "content must contain non-whitespace protocol text"
        payload_size = len(content.encode("utf-8"))
        if payload_size > MAX_SKILL_CONTENT_BYTES:
            return (
                f"content exceeds the {MAX_SKILL_CONTENT_BYTES}-byte UTF-8 maximum"
            )
        protocol_error = _protocol_error(content)
        if protocol_error:
            return protocol_error
        overwrite = bool(parameters.get("overwrite", False))
        expected_revision = str(parameters.get("expected_revision") or "")
        if overwrite != bool(expected_revision):
            return "overwrite=true and expected_revision must be supplied together"
        return ""

    def execute(
        self,
        category: str = None,
        skill_name: str = None,
        content: str = None,
        evidence: list[dict] | None = None,
        overwrite: bool = False,
        expected_revision: str = "",
    ) -> str:
        if not category or not skill_name:
            return "Error: both 'category' and 'skill_name' are required."
        if content is None or not str(content).strip():
            return "Error: 'content' is required and must be the full protocol text (non-empty)."
        payload_size = len(str(content).encode("utf-8"))
        if payload_size > MAX_SKILL_CONTENT_BYTES:
            return (
                f"Error: skill content is {payload_size} bytes; the maximum is "
                f"{MAX_SKILL_CONTENT_BYTES} bytes. Shorten the protocol."
            )
        if not _safe_component(category):
            return (f"Error: invalid category '{category}'. Use a simple folder name "
                    f"(letters, digits, '-', '_'), no slashes or leading dots.")
        if not _safe_component(skill_name):
            return (f"Error: invalid skill_name '{skill_name}'. Use a simple name "
                    f"(letters, digits, '-', '_'), no slashes or spaces.")
        clean_content = str(content).strip()
        protocol_error = _protocol_error(clean_content)
        if protocol_error:
            return f"Error: {protocol_error}."
        if contains_persisted_secret(clean_content):
            return (
                "COMMAND BLOCKED: secret-like credentials cannot be stored in a learned skill; "
                "use an opaque Nexus credential handle."
            )
        if bool(overwrite) != bool(str(expected_revision or "")):
            return "Error: overwrite=true and expected_revision are required together."

        sm = SkillsManager()
        skill_path = f"{category}/{skill_name}"
        old_payload = None
        skill_file = None
        state_guard = sm.state_lock()
        try:
            state_guard.__enter__()
        except SkillContentError as exc:
            return f"Error writing skill: {type(exc).__name__}: {exc}"
        try:
            validated_evidence = _validated_skill_evidence(sm, skill_path, evidence)
            evidence_protocol_error = _protocol_evidence_error(
                sm, clean_content, validated_evidence
            )
            if evidence_protocol_error:
                return f"Error: {evidence_protocol_error}."
            evidence_request_ids = {
                str(
                    (
                        sm.knowledge_store().read_note(reference["note_id"]) or {}
                    ).get("experience", {}).get("request_id")
                    or ""
                )
                for reference in validated_evidence
            }
            for other in sm.list_effective_skills():
                if (
                    other.get("scope") != "private"
                    or other.get("skill_path") == skill_path
                ):
                    continue
                other_lifecycle = other.get("lifecycle") or {}
                for reference in other_lifecycle.get("evidence", []):
                    prior_note = sm.knowledge_store().read_note(reference["note_id"])
                    prior_request = str(
                        (prior_note.get("experience") or {}).get("request_id") or ""
                    ) if prior_note else ""
                    if prior_request and prior_request in evidence_request_ids:
                        return (
                            "Error: one recovery episode may earn at most one learned skill. "
                            f"Merge this shortcut into '{other['skill_path']}' or keep the "
                            "additional finding in the wiki."
                        )
            skill_file = sm.get_mutable_skill_file(category, skill_name)
            if skill_file.is_symlink():
                raise SkillContentError("private skill path may not be a symlink")
            existed = skill_file.is_file()
            shared_collision = (sm.base_dir / category / f"{skill_name}.txt").is_file()
            if shared_collision and not existed:
                return (
                    f"Error: '{skill_path}' is a shared base skill. Choose a distinct name; "
                    "runtime learning cannot shadow the baked-in catalog."
                )
            if existed and not overwrite:
                return (
                    f"Error: skill '{skill_path}' already exists. Read it for its revision, "
                    "then pass overwrite=true, expected_revision, and fresh evidence."
                )
            if not existed and overwrite:
                return f"Error: private skill '{skill_path}' no longer exists; do not overwrite stale state."
            if not existed and sm.private_skill_count() >= MAX_PRIVATE_SKILLS:
                return (
                    f"Error: this agent already has the {MAX_PRIVATE_SKILLS}-skill limit. "
                    "Retire a stale learned skill before creating another."
                )
            existing_lifecycle = None
            if existed:
                old_payload = skill_file.read_bytes()
                old_content = old_payload.decode("utf-8").strip()
                current_revision = skill_revision(old_content)
                if not hmac.compare_digest(current_revision, str(expected_revision)):
                    return f"Error: skill '{skill_path}' changed since it was read."
                existing_lifecycle = sm.learned_store().read(
                    category, skill_name, current_content_revision=current_revision
                )
                if existing_lifecycle is not None:
                    old_note_ids = {
                        item["note_id"] for item in existing_lifecycle.get("evidence", [])
                    }
                    new_note_ids = {item["note_id"] for item in validated_evidence}
                    if new_note_ids.issubset(old_note_ids):
                        return (
                            f"Error: revising '{skill_path}' requires at least one new or newly "
                            "revised failure-to-success evidence note."
                        )
                    old_request_ids = set()
                    for reference in existing_lifecycle.get("evidence", []):
                        old_note = sm.knowledge_store().read_note(reference["note_id"])
                        if old_note is None or not hmac.compare_digest(
                            str(old_note.get("revision") or ""),
                            str(reference.get("revision") or ""),
                        ):
                            raise SkillKnowledgeError(
                                "existing skill evidence changed or disappeared; retire the "
                                "unverifiable revision instead of blessing it"
                            )
                        old_request_ids.add(
                            str((old_note.get("experience") or {}).get("request_id") or "")
                        )
                    new_request_ids = {
                        str(
                            (
                                sm.knowledge_store().read_note(reference["note_id"]) or {}
                            ).get("experience", {}).get("request_id")
                            or ""
                        )
                        for reference in validated_evidence
                    }
                    if new_request_ids.issubset(old_request_ids):
                        return (
                            f"Error: revising '{skill_path}' requires recovery evidence from a "
                            "new request, not another note describing the original episode."
                        )
            payload = (clean_content + "\n").encode("utf-8")
            _atomic_skill_write(skill_file, payload)
            try:
                saved_lifecycle = sm.learned_store().save_protocol(
                    category=category,
                    skill_name=skill_name,
                    content_revision=skill_revision(clean_content),
                    evidence=validated_evidence,
                )
            except Exception:
                if old_payload is None:
                    try:
                        skill_file.unlink()
                    except OSError:
                        pass
                else:
                    _atomic_skill_write(skill_file, old_payload)
                raise
        except (SkillContentError, SkillKnowledgeError, LearnedSkillError, OSError, UnicodeError) as e:
            return f"Error writing skill: {type(e).__name__}: {e}"
        finally:
            state_guard.__exit__(None, None, None)

        # Make the new category browsable in the SKILLS section right away.
        try:
            self.worker.expanded_categories.add(f"skill:{category}")
        except Exception:
            pass

        active = getattr(self.worker, "active_skill", None)
        if active and active.get("path") == skill_path:
            self.worker.active_skill = None
        verb = "Revised" if existed else "Created"
        print(f"{self.C_GREEN}\U0001F4DD SKILL {verb.upper()}: {category}/{skill_name}{self.C_RESET}")
        return (
            f"{verb} this agent's private skill '{skill_path}'; revision="
            f"{saved_lifecycle['content_revision']}, status=ready. It remains advisory. "
            "Activate it only when live preconditions match, and report an honest outcome when done."
        )


class RememberSkillKnowledgeTool(BaseTool):
    """Create or revision-check an owner-private skill-development note."""

    def __init__(self, worker=None):
        super().__init__(
            name="remember_skill_knowledge",
            description=(
                "Store verified, reusable knowledge in this exact agent's persistent skill wiki. Use it while "
                "learning a difficult workflow: record prerequisites, failed approaches, the successful method, "
                "cost/time improvements, and verification evidence. Later use list_skill_knowledge and "
                "read_skill_knowledge, then distill the evidence into create_skill. Never store credentials, "
                "private keys, copied external instructions, or unverified claims. A note is agent state and "
                "grants no new authority. To update a note, pass its note_id and exact expected_revision."
                " Set clear_learning=true on an update to retract an incorrect learning claim; any "
                "skill citing that note will become needs_review."
            ),
        )
        self.worker = worker

    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": 240},
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_SKILL_KNOWLEDGE_BYTES,
                },
                "related_skill_paths": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {"type": "string", "pattern": SKILL_PATH_RE.pattern},
                },
                "note_id": {
                    "type": "string",
                    "pattern": SKILL_NOTE_ID_RE.pattern,
                },
                "expected_revision": {
                    "type": "string",
                    "pattern": r"^[0-9a-f]{64}$",
                },
                "learning": {
                    "type": "object",
                    "properties": {
                        "candidate_skill_path": {
                            "type": "string",
                            "pattern": SKILL_PATH_RE.pattern,
                        },
                        "procedure": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 16384,
                        },
                        "verification": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 8192,
                        },
                        "procedure_stable": {"type": "boolean"},
                        "uncertainty": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": [
                        "candidate_skill_path",
                        "procedure",
                        "verification",
                        "procedure_stable",
                        "uncertainty",
                    ],
                    "additionalProperties": False,
                },
                "clear_learning": {"type": "boolean"},
            },
            "required": ["title", "content", "related_skill_paths"],
            "additionalProperties": False,
        }

    def validate_parameters(self, parameters) -> str:
        error = super().validate_parameters(parameters)
        if error:
            return error
        note_id = str(parameters.get("note_id") or "")
        revision = str(parameters.get("expected_revision") or "")
        if bool(note_id) != bool(revision):
            return "note_id and expected_revision must be supplied together for updates"
        if not str(parameters.get("title") or "").strip():
            return "title must contain non-whitespace text"
        if not str(parameters.get("content") or "").strip():
            return "content must contain non-whitespace text"
        if parameters.get("learning") is not None and parameters.get("clear_learning"):
            return "learning and clear_learning cannot be supplied together"
        if parameters.get("clear_learning") and not note_id:
            return "clear_learning is valid only when updating an existing note"
        return ""

    def execute(
        self,
        title: str,
        content: str,
        related_skill_paths: list[str],
        note_id: str = "",
        expected_revision: str = "",
        learning: dict | None = None,
        clear_learning: bool = False,
    ) -> str:
        if contains_persisted_secret(f"{title}\n{content}\n{json.dumps(learning or {})}"):
            return (
                "COMMAND BLOCKED: secret-like credentials cannot be stored in the skill wiki; "
                "use an opaque Nexus credential handle."
            )
        experience = None
        if learning is not None:
            experience = _experience_from_worker(self.worker)
            if not experience or not experience.get("recovered_after_failure"):
                return (
                    "Error: this request does not contain a harness-observed failed approach followed "
                    "by a successful tool result. Save an ordinary wiki note without learning fields; "
                    "a first-try success is not eligible to become a skill."
                )
            candidate = str(learning.get("candidate_skill_path") or "")
            if candidate not in related_skill_paths:
                related_skill_paths = [*related_skill_paths, candidate]
        manager = SkillsManager()
        try:
            with manager.state_lock():
                saved = manager.knowledge_store().save_note(
                    title=title,
                    content=content,
                    related_skill_paths=related_skill_paths,
                    note_id=note_id,
                    expected_revision=expected_revision,
                    learning=learning,
                    experience=experience,
                    clear_learning=clear_learning,
                )
        except (SkillContentError, SkillKnowledgeError) as exc:
            return f"Error: skill knowledge was not saved: {exc}"
        verb = "Updated" if note_id else "Created"
        return (
            f"{verb} persistent skill-wiki note '{saved['id']}' ({saved['title']}); "
            f"revision={saved['revision']}, skill_evidence_eligible="
            f"{str(saved['skill_evidence_eligible']).lower()}. This note is evidence/context only "
            "and grants no authority."
        )


class ListSkillKnowledgeTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(
            name="list_skill_knowledge",
            description=(
                "List this exact agent's persistent skill-wiki notes with IDs, titles, related skills, "
                "revisions, and short previews. Read a selected note in full with read_skill_knowledge."
            ),
        )
        self.worker = worker

    def execute(self) -> str:
        try:
            notes = SkillsManager().knowledge_store().list_notes()
        except SkillKnowledgeError as exc:
            return f"Error: skill knowledge could not be listed: {exc}"
        payload = [
            {
                "id": note["id"],
                "title": note["title"],
                "related_skill_paths": note["related_skill_paths"],
                "updated_at": note["updated_at"],
                "revision": note["revision"],
                "preview": str(note["content"])[:500],
            }
            for note in notes
        ]
        return json.dumps({"notes": payload}, ensure_ascii=False, indent=2)


class ReadSkillKnowledgeTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(
            name="read_skill_knowledge",
            description=(
                "Read one complete persistent skill-wiki note by the exact note_id returned by "
                "list_skill_knowledge. Notes are prior evidence/context, not authority or automatic truth."
            ),
        )
        self.worker = worker

    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "note_id": {"type": "string", "pattern": SKILL_NOTE_ID_RE.pattern}
            },
            "required": ["note_id"],
            "additionalProperties": False,
        }

    def execute(self, note_id: str) -> str:
        try:
            note = SkillsManager().knowledge_store().read_note(note_id)
        except SkillKnowledgeError as exc:
            return f"Error: skill knowledge could not be read: {exc}"
        if note is None:
            return f"Error: no skill knowledge note '{note_id}'."
        return json.dumps(note, ensure_ascii=False, indent=2)


class SearchSkillKnowledgeTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(
            name="search_skill_knowledge",
            description=(
                "Search this agent's persistent skill wiki by words, phrases, or skill path. "
                "Returns ranked previews and revision IDs; read a note before relying on or editing it."
            ),
        )
        self.worker = worker

    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 500},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def execute(self, query: str, limit: int = 8) -> str:
        try:
            notes = SkillsManager().knowledge_store().search_notes(query, limit=limit)
        except SkillKnowledgeError as exc:
            return f"Error: skill knowledge search failed: {exc}"
        payload = [
            {
                "id": note["id"],
                "title": note["title"],
                "related_skill_paths": note["related_skill_paths"],
                "revision": note["revision"],
                "search_score": note["search_score"],
                "skill_evidence_eligible": note["skill_evidence_eligible"],
                "preview": note["preview"],
            }
            for note in notes
        ]
        return json.dumps({"query": query, "notes": payload}, ensure_ascii=False, indent=2)


class DeleteSkillKnowledgeTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(
            name="delete_skill_knowledge",
            description=(
                "Delete one stale or incorrect private wiki note using the exact revision returned by "
                "read/search/list. Refuses deletion while a learned skill cites the note as evidence."
            ),
        )
        self.worker = worker

    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "note_id": {"type": "string", "pattern": SKILL_NOTE_ID_RE.pattern},
                "expected_revision": {
                    "type": "string",
                    "pattern": r"^[0-9a-f]{64}$",
                },
            },
            "required": ["note_id", "expected_revision"],
            "additionalProperties": False,
        }

    def execute(self, note_id: str, expected_revision: str) -> str:
        sm = SkillsManager()
        try:
            with sm.state_lock():
                for record in sm.list_effective_skills():
                    if record.get("scope") != "private":
                        continue
                    lifecycle = record.get("lifecycle") or {}
                    if lifecycle.get("integrity_ambiguous"):
                        return (
                            "Error: learned-skill metadata failed integrity checks, so the harness cannot "
                            "prove this note is unreferenced. Repair or retire the affected skill first."
                        )
                    if any(
                        ref.get("note_id") == note_id
                        for ref in lifecycle.get("evidence", [])
                    ):
                        return (
                            f"Error: note '{note_id}' is evidence for learned skill "
                            f"'{record['skill_path']}'. Retire that skill before deleting its evidence."
                        )
                sm.knowledge_store().delete_note(
                    note_id, expected_revision=expected_revision
                )
        except (SkillContentError, SkillKnowledgeError) as exc:
            return f"Error: skill knowledge was not deleted: {exc}"
        return f"Deleted persistent skill-wiki note '{note_id}'."


class ReadSkillTool(BaseTool):
    """Read a skill's FULL protocol text without activating it — the read half of
    self-modifying skills, so the agent can inspect a protocol before editing or
    deleting it (unlike activate_skill, which pins it, or the SKILLS section, which
    only shows a truncated preview)."""

    def __init__(self, worker=None):
        super().__init__(
            name="read_skill",
            description=(
                "Return the COMPLETE text of an existing skill protocol WITHOUT activating it. Use this to "
                "inspect current applicability, origin, revision, lifecycle, and evidence before using, "
                "revision-safe editing, or deletion. Read-only: no side effects, nothing is pinned.\n"
                "Schema:\n"
                "  skill_path (str, required): '<category>/<skill_name>', e.g. 'research/deep_research'.\n"
                "Example: {\"tool_name\": \"read_skill\", \"parameters\": {\"skill_path\": \"research/deep_research\"}}"
            )
        )
        self.worker = worker

    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "skill_path": {
                    "type": "string",
                    "pattern": SKILL_PATH_RE.pattern,
                }
            },
            "required": ["skill_path"],
            "additionalProperties": False,
        }

    def execute(self, skill_path: str = None, **kwargs) -> str:
        if not skill_path or "/" not in skill_path:
            return "Error: skill_path must be '<category>/<skill_name>' (e.g. 'research/deep_research')."
        category, _, skill_name = skill_path.partition("/")
        if not _safe_component(category) or not _safe_component(skill_name):
            return f"Error: invalid skill_path '{skill_path}'."
        sm = SkillsManager()
        try:
            record = sm.get_skill_record(category, skill_name)
        except SkillContentTooLarge as exc:
            return f"Error: Skill '{skill_path}' cannot be read because {exc}."
        except SkillContentError as exc:
            return f"Error: Skill '{skill_path}' failed integrity checks: {exc}."
        if not record:
            return _not_found_msg(sm, skill_path, category)
        metadata = {
            key: record.get(key)
            for key in (
                "skill_path",
                "revision",
                "scope",
                "editable",
                "overrides_shared",
                "lifecycle",
            )
        }
        return (
            "--- SKILL RECORD ---\n"
            + json.dumps(metadata, ensure_ascii=False, indent=2)
            + f"\n--- SKILL CONTENT: {skill_path} ---\n{record['content']}"
        )


class DeleteSkillTool(BaseTool):
    """Retire one revision of an agent-authored playbook."""

    def __init__(self, worker=None):
        super().__init__(
            name="delete_skill",
            description=(
                "Retire a stale, wrong, or no-longer-useful skill created by this agent. Shared base "
                "skills cannot be removed. Requires the exact revision from read_skill and a reason. "
                "The old protocol and reason are archived in the private wiki before removal, so the "
                "decision remains recoverable and searchable."
            )
        )
        self.worker = worker

    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "skill_path": {"type": "string", "pattern": SKILL_PATH_RE.pattern},
                "expected_revision": {
                    "type": "string",
                    "pattern": r"^[0-9a-f]{64}$",
                },
                "reason": {"type": "string", "minLength": 1, "maxLength": 2000},
            },
            "required": ["skill_path", "expected_revision", "reason"],
            "additionalProperties": False,
        }

    def execute(
        self,
        skill_path: str = None,
        expected_revision: str = "",
        reason: str = "",
        **kwargs,
    ) -> str:
        if not skill_path or "/" not in skill_path:
            return "Error: skill_path must be '<category>/<skill_name>' (e.g. 'research/old_protocol')."
        category, _, skill_name = skill_path.partition("/")
        if not _safe_component(category) or not _safe_component(skill_name):
            return f"Error: invalid skill_path '{skill_path}'."

        if not str(reason or "").strip():
            return "Error: reason is required so retirement remains explainable."
        sm = SkillsManager()
        try:
            with sm.state_lock():
                return self._retire_locked(
                    sm=sm,
                    category=category,
                    skill_name=skill_name,
                    skill_path=skill_path,
                    expected_revision=expected_revision,
                    reason=str(reason).strip(),
                )
        except SkillContentError as exc:
            return f"Error: Skill '{skill_path}' failed integrity checks: {exc}."

    def _retire_locked(
        self,
        *,
        sm: SkillsManager,
        category: str,
        skill_name: str,
        skill_path: str,
        expected_revision: str,
        reason: str,
    ) -> str:
        """Retire only the exact revision observed under the agent-state lock."""

        try:
            record = sm.get_skill_record(category, skill_name)
        except SkillContentError as exc:
            return f"Error: Skill '{skill_path}' failed integrity checks: {exc}."
        if not record:
            return _not_found_msg(sm, skill_path, category)
        if record.get("scope") != "private":
            return (
                f"Error: skill '{skill_path}' is a shared packaged protocol. "
                "Only skills created by this agent can be deleted."
            )
        if not hmac.compare_digest(
            str(record.get("revision") or ""), str(expected_revision or "")
        ):
            return f"Error: skill '{skill_path}' changed since it was read."

        try:
            skill_file = sm.get_mutable_skill_file(category, skill_name)
        except SkillContentError as exc:
            return f"Error: {exc}"
        cat_dir = skill_file.parent
        if not skill_file.is_file():
            return f"Error: private skill '{skill_path}' no longer exists."
        archived = None
        archive_warning = ""
        archive_content = (
            f"Retirement reason: {reason}\n\n"
            f"Retired revision: {record['revision']}\n\n"
            f"--- Retired protocol ---\n{record['content']}"
        )
        try:
            archived = sm.knowledge_store().save_note(
                title=f"Retired skill: {skill_path}",
                content=archive_content,
                related_skill_paths=[skill_path],
            )
        except SkillKnowledgeError:
            # A protocol containing credential-like text must remain deletable.
            # Preserve only non-sensitive provenance when the full snapshot is
            # unsafe, and never let a full/corrupt wiki trap a bad playbook.
            safe_reason = (
                "[reason omitted because it contained secret-like text]"
                if contains_persisted_secret(reason)
                else reason
            )
            try:
                archived = sm.knowledge_store().save_note(
                    title=f"Retired skill record: {skill_path}",
                    content=(
                        f"Retirement reason: {safe_reason}\n\n"
                        f"Retired revision: {record['revision']}\n"
                        "Protocol text was omitted because it could not be archived safely."
                    ),
                    related_skill_paths=[skill_path],
                )
                archive_warning = " The archived record omits the protocol text."
            except SkillKnowledgeError:
                archive_warning = (
                    " No wiki archive could be written safely; the exact retired revision "
                    f"was {record['revision']}."
                )
        try:
            skill_file.unlink()
        except OSError as exc:
            return f"Error retiring skill: {type(exc).__name__}: {exc}"
        lifecycle_warning = ""
        try:
            sm.learned_store().remove(category, skill_name)
        except LearnedSkillError as exc:
            # The protocol is already gone and therefore cannot be activated.
            # Report cleanup trouble without falsely claiming deletion failed.
            lifecycle_warning = f" Lifecycle metadata cleanup needs review: {exc}."

        # If we just deleted the active protocol, unpin it so the agent isn't
        # following a skill that no longer exists.
        unpinned = ""
        try:
            active = getattr(self.worker, "active_skill", None)
            if active and active.get("path") == skill_path:
                self.worker.active_skill = None
                unpinned = " It was the active protocol, so it has been unpinned."
        except Exception:
            pass

        # Remove only the now-empty exact category and its stale expansion marker.
        try:
            cat_dir.rmdir()
        except OSError:
            pass
        try:
            self.worker.expanded_categories.discard(f"skill:{category}")
        except Exception:
            pass

        print(f"{self.C_YELLOW}\U0001F5D1 SKILL RETIRED: {skill_path}{self.C_RESET}")
        archive_result = (
            f" Archived the retirement in wiki note '{archived['id']}'."
            if archived is not None
            else ""
        )
        return (
            f"Retired skill '{skill_path}' at revision {record['revision']}."
            f"{archive_result}{archive_warning}{lifecycle_warning}{unpinned}"
        )


class DeactivateSkillTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(
            name="deactivate_skill",
            description=(
                "Stop using the active advisory playbook and report what live evidence showed. "
                "Outcomes: success; adapted (worked only after changing it); failed; or not_applicable. "
                "Adapted/failed require a concise note and immediately mark learned skills for review or "
                "quarantine, preventing blind retries."
            )
        )
        self.worker = worker

    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": sorted(VALID_OUTCOMES)},
                "note": {"type": "string", "maxLength": 2000},
            },
            "required": ["outcome"],
            "additionalProperties": False,
        }

    def execute(self, outcome: str = "", note: str = "", **kwargs) -> str:
        if not self.worker:
            return "Error: Worker context missing."
        active = getattr(self.worker, "active_skill", None)
        if not active:
            return "No skill is currently active."
        outcome = str(outcome or "").strip()
        note = str(note or "").strip()
        if outcome not in VALID_OUTCOMES:
            return "Error: outcome must be success, adapted, failed, or not_applicable."
        if outcome in {"adapted", "failed"} and not note:
            return "Error: adapted and failed outcomes require a concise note."
        if note and contains_persisted_secret(note):
            return (
                "COMMAND BLOCKED: secret-like credentials cannot be stored in a skill "
                "outcome note. Redact the value and report the outcome again."
            )
        if active.get("paused") and outcome == "success":
            return (
                "Error: the playbook was paused by contrary evidence; report adapted, failed, "
                "or not_applicable rather than success."
            )
        path = active.get("path", "unknown")
        lifecycle_status = str(active.get("status") or "shared")
        lifecycle_error = ""
        if active.get("scope") == "private" and "/" in path:
            category, _, skill_name = path.partition("/")
            manager = SkillsManager()
            try:
                with manager.state_lock():
                    saved = manager.learned_store().record_outcome(
                        category=category,
                        skill_name=skill_name,
                        content_revision=str(active.get("sha256") or ""),
                        outcome=outcome,
                        note=note,
                    )
                    lifecycle_status = str(saved["status"])
                    if outcome in {"adapted", "failed"} and note:
                        try:
                            manager.knowledge_store().save_note(
                                title=f"Skill {outcome}: {path}",
                                content=(
                                    f"Observed outcome: {outcome}\n"
                                    f"Skill revision: {active.get('sha256', '')}\n\n{note}"
                                ),
                                related_skill_paths=[path],
                            )
                        except SkillKnowledgeError:
                            pass
            except (SkillContentError, LearnedSkillError) as exc:
                lifecycle_error = str(exc)
        self.worker.active_skill = None
        print(f"{self.C_CYAN}\u2713 SKILL DEACTIVATED: {path}{self.C_RESET}")
        suffix = f" Lifecycle update failed safely: {lifecycle_error}" if lifecycle_error else ""
        return (
            f"Skill '{path}' deactivated with outcome={outcome}, status={lifecycle_status}, "
            f"and unpinned from context.{suffix}"
        )
