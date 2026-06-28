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

from aeon.tools.base import BaseTool
from aeon.core.skills.manager import SkillsManager


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
