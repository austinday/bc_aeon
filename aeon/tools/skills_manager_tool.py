from typing import Any
from aeon.tools.base import BaseTool

class ExpandSkillsCategory(BaseTool):
    def __init__(self, worker=None, llm_client=None, **kwargs):
        super().__init__(
            name="expand_skills_category", 
            description="Expands a skill category to reveal the skills within it.", 
            **kwargs
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, category_path: str) -> str:
        if not self.worker:
            return "Error: Worker not initialized."
        
        self.worker.expanded_categories.add(category_path)
        return f"Skill category '{category_path}' has been expanded."

class CollapseSkillsCategory(BaseTool):
    def __init__(self, worker=None, llm_client=None, **kwargs):
        super().__init__(
            name="collapse_skills_category", 
            description="Collapses a skill category to hide its skills and save context space.", 
            **kwargs
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, category_path: str) -> str:
        if not self.worker:
            return "Error: Worker not initialized."
        
        if category_path in self.worker.expanded_categories:
            self.worker.expanded_categories.remove(category_path)
            return f"Skill category '{category_path}' has been collapsed."
        return f"Skill category '{category_path}' was not expanded."