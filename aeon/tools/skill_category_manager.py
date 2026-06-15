from typing import Any
from aeon.core.worker import Worker

class ExpandSkillCategory:
    """Expands a skill category to reveal the skill protocols within it.
    Use this when your task might benefit from specialized skills (e.g., research, coding standards) 
    that are not in your default tool list. Expanding a category makes its skills visible in your context.
    """
    def __init__(self, worker: Worker):
        self.worker = worker
        self.name = "expand_skill_category"

    def execute(self, category_path: str) -> str:
        # Use the 'skill:' prefix to distinguish from tool categories in the expanded_categories set
        skill_key = f"skill:{category_path}"
        self.worker.expanded_categories.add(skill_key)
        return f"Skill category '{category_path}' expanded. Its protocols are now visible in your context."

class CollapseSkillCategory:
    """Collapses a skill category to free context space. Use this when you no longer need the 
    specialized skills in a category. Collapsed skills are hidden from your context but remain callable if needed.
    """
    def __init__(self, worker: Worker):
        self.worker = worker
        self.name = "collapse_skill_category"

    def execute(self, category_path: str) -> str:
        skill_key = f"skill:{category_path}"
        if skill_key in self.worker.expanded_categories:
            self.worker.expanded_categories.remove(skill_key)
            return f"Skill category '{category_path}' collapsed."
        return f"Skill category '{category_path}' was not expanded."