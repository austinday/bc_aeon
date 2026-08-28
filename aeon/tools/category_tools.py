import difflib
from .base import BaseTool
from .categories import (
    get_category_at_path, get_all_category_paths,
    count_tools_in_category,
)
from ..core.prompts import TOOL_DESC_EXPAND_CATEGORY, TOOL_DESC_COLLAPSE_CATEGORY


def _skill_category_names():
    """Best-effort list of skill category names (empty on any failure)."""
    try:
        from aeon.core.skills.manager import SkillsManager
        sm = SkillsManager()
        return sm.list_categories()
    except Exception:
        return []


def _suggest_category(category_path, candidates):
    """Return a ' Did you mean: ...' hint for the closest known categories."""
    close = difflib.get_close_matches(category_path, candidates, n=3, cutoff=0.4)
    if close:
        return f" Did you mean: {', '.join(close)}?"
    return f" Available: {', '.join(candidates)}" if candidates else ""


class ExpandToolCategoryTool(BaseTool):
    """Expands a tool category to reveal the tools or subcategories within it."""
    def __init__(self, worker):
        super().__init__(
            name='expand_tool_category',
            description=TOOL_DESC_EXPAND_CATEGORY
        )
        self.worker = worker

    def execute(self, category_path: str) -> str:
        if not category_path:
            return 'Error: category_path parameter is required.'

        # 1. Try as a Tool Category
        cat = get_category_at_path(category_path)
        if cat is not None:
            self.worker.expanded_categories.add(category_path)
            parts = []
            if 'subcategories' in cat:
                sub_names = list(cat['subcategories'].keys())
                sub_summaries = []
                for sn in sub_names:
                    sc = cat['subcategories'][sn]
                    tc = count_tools_in_category(f'{category_path}/{sn}')
                    sub_summaries.append(f'{sn} ({tc} tool{"s" if tc != 1 else ""})')
                parts.append(f"Subcategories revealed: {', '.join(sub_summaries)}")
            if 'tools' in cat:
                parts.append(f"Tools revealed: {', '.join(cat['tools'])}")
            summary = '; '.join(parts) if parts else 'Category expanded (empty).'
            return f"Expanded tool category '{category_path}'. {summary}"

        # 2. Try as a Skill Category
        from aeon.core.skills.manager import SkillsManager
        sm = SkillsManager()
        skills = sm.get_skills_in_category(category_path)
        if skills:
            # Use 'skill:' prefix in expanded_categories to distinguish from tools
            self.worker.expanded_categories.add(f"skill:{category_path}")
            return f"Expanded skill category '{category_path}'. Revealed:\n" + "\n".join(skills)
        
        # 3. Not found in either — suggest the closest tool OR skill category.
        candidates = get_all_category_paths() + _skill_category_names()
        return (
            f"Error: Category '{category_path}' not found in tools or skills."
            f"{_suggest_category(category_path, candidates)}"
        )


class CollapseToolCategoryTool(BaseTool):
    """Collapses a tool category (and all its subcategories) to save context space."""
    def __init__(self, worker):
        super().__init__(
            name='collapse_tool_category',
            description=TOOL_DESC_COLLAPSE_CATEGORY
        )
        self.worker = worker

    def execute(self, category_path: str) -> str:
        if not category_path:
            return 'Error: category_path parameter is required.'

        cat = get_category_at_path(category_path)
        if cat is None:
            candidates = get_all_category_paths() + _skill_category_names()
            return (
                f"Error: Category '{category_path}' not found."
                f"{_suggest_category(category_path, candidates)}"
            )

        # Remove this path and all sub-paths (for both tools and skills)
        prefix = category_path + '/'
        skill_prefix = 'skill:' + category_path
        skill_sub_prefix = skill_prefix + '/'
        
        to_remove = {p for p in self.worker.expanded_categories
                     if p == category_path 
                     or p.startswith(prefix) 
                     or p == skill_prefix 
                     or p.startswith(skill_sub_prefix)}

        if not to_remove:
            return f"Category '{category_path}' was not expanded."

        self.worker.expanded_categories -= to_remove
        return f"Collapsed '{category_path}' (removed {len(to_remove)} expansion(s)). Tools hidden from context to save space."
