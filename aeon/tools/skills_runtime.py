"""
Runtime skill activation tools.

A "skill" is just protocol text. Expanding a skill category only *shows* that text
once; it then scrolls out of the action log and the agent tends to drift off it.
These tools make skill usage explicit and sticky:

  - activate_skill   -> loads the full protocol, pins it to the worker, and the
                        worker re-injects it into EVERY subsequent prompt as the
                        ACTIVE SKILL PROTOCOL block (the closest thing to
                        "enforcement" an injection-based system has).
  - deactivate_skill -> unpins it.

Both are BaseTool subclasses, so the dynamic loader picks them up automatically
(the `worker` dependency is supplied by main.py's loader deps). No manual
registration is required.
"""

import difflib
import re

from aeon.tools.base import BaseTool
from aeon.core.skills.manager import SkillsManager


def _safe_component(name: str) -> bool:
    """True if `name` is a safe single path component for a skill category/name:
    no separators, no traversal, only word chars/dash/dot, and not all-dots."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return False
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        return False
    return name.strip(".") != ""


def _all_skill_paths(sm):
    """Best-effort list of every '<category>/<skill>' path (empty on failure)."""
    paths = []
    try:
        categories = [d.name for d in sm.base_dir.iterdir() if d.is_dir() and not d.name.startswith('__')]
        for cat in categories:
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
                "Adopts a skill protocol for the CURRENT task. The full protocol text is loaded into your "
                "context AND pinned to every following prompt as the ACTIVE SKILL PROTOCOL, so you follow "
                "it step-by-step until you deactivate it. THIS IS HOW A SKILL IS USED \u2014 skills are never "
                "applied automatically. Activate the moment you notice the objective matches a skill listed "
                "in the SKILLS section.\n"
                "Schema:\n"
                "  skill_path (str, required): '<category>/<skill_name>', e.g. 'research/web_research'.\n"
                "Example: {\"tool_name\": \"activate_skill\", \"parameters\": {\"skill_path\": \"research/web_research\"}}"
            )
        )
        self.worker = worker

    def execute(self, skill_path: str = None, **kwargs) -> str:
        if not self.worker:
            return "Error: Worker context missing."
        if not skill_path or "/" not in skill_path:
            return "Error: skill_path must be '<category>/<skill_name>' (e.g. 'research/web_research')."

        category, _, skill_name = skill_path.partition("/")
        sm = SkillsManager()
        content = sm.get_skill_content(category, skill_name)
        if not content:
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

        self.worker.active_skill = {"path": skill_path, "content": content}
        self.worker.expanded_categories.add(f"skill:{category}")

        print(f"{self.C_GREEN}\U0001F3AF SKILL ACTIVATED: {skill_path} \u2014 protocol pinned to context until deactivated.{self.C_RESET}")
        return (
            f"Skill '{skill_path}' is now ACTIVE and pinned to your context every turn. "
            f"Follow its steps in order. Call deactivate_skill once the protocol is complete.\n\n"
            f"--- ACTIVE PROTOCOL: {skill_path} ---\n{content}"
        )


class CreateSkillTool(BaseTool):
    """Author a NEW reusable skill protocol by asking, with zero path guesswork.

    Skills are plain protocol text read live from disk each turn, so a skill
    created here is immediately visible in the SKILLS section and activatable with
    activate_skill in the SAME session — no restart needed. It is written into the
    aeon source tree, so it also persists across restarts and reinstalls.
    """

    def __init__(self, worker=None):
        super().__init__(
            name="create_skill",
            description=(
                "Create a new reusable skill protocol (a step-by-step procedure you can activate for "
                "matching tasks). The skill is written to the correct location automatically and becomes "
                "usable IMMEDIATELY this session via activate_skill('<category>/<skill_name>') — no "
                "restart required — and persists across restarts. Use this instead of write_file when "
                "adding a skill; you do NOT need to know where skills live.\n"
                "Write the protocol as clear, numbered, imperative steps (Objective, Process, Verification). "
                "Start the file with one or two '# ' comment lines describing WHEN the skill applies — the "
                "skill router reads those to match it to future tasks.\n"
                "Schema:\n"
                "  category (str, required): Skill category folder, e.g. 'research', 'coding' (created if new).\n"
                "  skill_name (str, required): Skill name (no spaces/slashes), e.g. 'web_research'.\n"
                "  content (str, required): The full protocol text (use a __BLOCK_N__ block).\n"
                "  overwrite (bool, optional, default=false): Allow replacing an existing skill of the same name.\n"
                "Example: {\"tool_name\": \"create_skill\", \"parameters\": {\"category\": \"coding\", "
                "\"skill_name\": \"api_migration\", \"content\": \"__BLOCK_1__\"}}"
            )
        )
        self.worker = worker

    def execute(self, category: str = None, skill_name: str = None,
                content: str = None, overwrite: bool = False) -> str:
        if not category or not skill_name:
            return "Error: both 'category' and 'skill_name' are required."
        if content is None or not str(content).strip():
            return "Error: 'content' is required and must be the full protocol text (non-empty)."
        if not _safe_component(category):
            return (f"Error: invalid category '{category}'. Use a simple folder name "
                    f"(letters, digits, '-', '_'), no slashes or leading dots.")
        if not _safe_component(skill_name):
            return (f"Error: invalid skill_name '{skill_name}'. Use a simple name "
                    f"(letters, digits, '-', '_'), no slashes or spaces.")

        sm = SkillsManager()
        try:
            cat_dir = sm.base_dir / category
            skill_file = cat_dir / f"{skill_name}.txt"
            existed = skill_file.exists()
            if existed and not overwrite:
                return (f"Error: skill '{category}/{skill_name}' already exists. Pass overwrite=true to "
                        f"replace it, or choose a different skill_name.")
            cat_dir.mkdir(parents=True, exist_ok=True)
            text = str(content)
            if not text.endswith("\n"):
                text += "\n"
            skill_file.write_text(text, encoding="utf-8")
        except Exception as e:
            return f"Error writing skill: {type(e).__name__}: {e}"

        # Make the new category browsable in the SKILLS section right away.
        try:
            self.worker.expanded_categories.add(f"skill:{category}")
        except Exception:
            pass

        verb = "Updated" if existed else "Created"
        print(f"{self.C_GREEN}\U0001F4DD SKILL {verb.upper()}: {category}/{skill_name}{self.C_RESET}")
        return (
            f"{verb} skill '{category}/{skill_name}' at {skill_file}. It is live NOW — activate it with "
            f"activate_skill('{category}/{skill_name}') to pin and follow it, and it will persist across "
            f"restarts. If this skill needs a NEW tool, add the tool under aeon/tools/ and call "
            f"restart_aeon so the tool loads."
        )


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
                "inspect a skill before modifying it (rewrite it with create_skill overwrite=true) or "
                "deleting it (delete_skill). Read-only: no side effects, nothing is pinned.\n"
                "Schema:\n"
                "  skill_path (str, required): '<category>/<skill_name>', e.g. 'research/web_research'.\n"
                "Example: {\"tool_name\": \"read_skill\", \"parameters\": {\"skill_path\": \"research/web_research\"}}"
            )
        )
        self.worker = worker

    def execute(self, skill_path: str = None, **kwargs) -> str:
        if not skill_path or "/" not in skill_path:
            return "Error: skill_path must be '<category>/<skill_name>' (e.g. 'research/web_research')."
        category, _, skill_name = skill_path.partition("/")
        sm = SkillsManager()
        content = sm.get_skill_content(category, skill_name)
        if not content:
            return _not_found_msg(sm, skill_path, category)
        return f"--- SKILL: {skill_path} ---\n{content}"


class DeleteSkillTool(BaseTool):
    """Remove an existing skill protocol — the delete half of self-modifying skills.
    Takes effect immediately (skills are read live) and persists across restarts."""

    def __init__(self, worker=None):
        super().__init__(
            name="delete_skill",
            description=(
                "Permanently remove an existing skill protocol. Effective IMMEDIATELY this session (it "
                "disappears from the SKILLS section) and persists across restarts. If the skill is "
                "currently active it is also unpinned. Read it first with read_skill if you are unsure. "
                "To CHANGE a skill instead of removing it, use create_skill with overwrite=true.\n"
                "Schema:\n"
                "  skill_path (str, required): '<category>/<skill_name>', e.g. 'research/old_protocol'.\n"
                "Example: {\"tool_name\": \"delete_skill\", \"parameters\": {\"skill_path\": \"research/old_protocol\"}}"
            )
        )
        self.worker = worker

    def execute(self, skill_path: str = None, **kwargs) -> str:
        if not skill_path or "/" not in skill_path:
            return "Error: skill_path must be '<category>/<skill_name>' (e.g. 'research/old_protocol')."
        category, _, skill_name = skill_path.partition("/")
        if not _safe_component(category) or not _safe_component(skill_name):
            return f"Error: invalid skill_path '{skill_path}'."

        sm = SkillsManager()
        if not sm.get_skill_content(category, skill_name):
            return _not_found_msg(sm, skill_path, category)

        cat_dir = sm.base_dir / category
        skill_file = cat_dir / f"{skill_name}.txt"
        try:
            skill_file.unlink()
        except Exception as e:
            return f"Error deleting skill: {type(e).__name__}: {e}"

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

        # Remove the category folder if it is now empty (no leftover empty categories
        # cluttering the SKILLS section), and drop any stale expansion marker.
        try:
            if cat_dir.is_dir() and not any(cat_dir.glob("*.txt")):
                # Only remove if truly empty of skills; ignore __pycache__ etc.
                for leftover in cat_dir.iterdir():
                    if leftover.name != "__pycache__":
                        break
                else:
                    import shutil
                    shutil.rmtree(cat_dir, ignore_errors=True)
            self.worker.expanded_categories.discard(f"skill:{category}")
        except Exception:
            pass

        print(f"{self.C_YELLOW}\U0001F5D1 SKILL DELETED: {skill_path}{self.C_RESET}")
        return f"Deleted skill '{skill_path}'.{unpinned}"


class DeactivateSkillTool(BaseTool):
    def __init__(self, worker=None):
        super().__init__(
            name="deactivate_skill",
            description=(
                "Stops following the currently active skill protocol and unpins it from your context. "
                "Call this once the protocol's steps are done or it no longer applies.\n"
                "Schema: (no parameters)\n"
                "Example: {\"tool_name\": \"deactivate_skill\", \"parameters\": {}}"
            )
        )
        self.worker = worker

    def execute(self, **kwargs) -> str:
        if not self.worker:
            return "Error: Worker context missing."
        active = getattr(self.worker, "active_skill", None)
        if not active:
            return "No skill is currently active."
        path = active.get("path", "unknown")
        self.worker.active_skill = None
        print(f"{self.C_CYAN}\u2713 SKILL DEACTIVATED: {path}{self.C_RESET}")
        return f"Skill '{path}' deactivated and unpinned from context."
