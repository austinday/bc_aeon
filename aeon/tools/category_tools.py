from .base import BaseTool
from .categories import (
    get_category_at_path, get_all_category_paths,
    count_tools_in_category,
)
from ..core.prompts import TOOL_DESC_EXPAND_CATEGORY, TOOL_DESC_COLLAPSE_CATEGORY


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

        cat = get_category_at_path(category_path)
        if cat is None:
            all_paths = get_all_category_paths()
            return (
                f"Error: Category '{category_path}' not found.\n"
                f"Available categories: {', '.join(all_paths)}"
            )

        self.worker.expanded_categories.add(category_path)

        # Build a summary of what was revealed
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
        return f"Expanded '{category_path}'. {summary}"


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
            all_paths = get_all_category_paths()
            return (
                f"Error: Category '{category_path}' not found.\n"
                f"Available categories: {', '.join(all_paths)}"
            )

        # Remove this path and all sub-paths
        prefix = category_path + '/'
        to_remove = {p for p in self.worker.expanded_categories
                     if p == category_path or p.startswith(prefix)}

        if not to_remove:
            return f"Category '{category_path}' was not expanded."

        self.worker.expanded_categories -= to_remove
        return f"Collapsed '{category_path}' (removed {len(to_remove)} expansion(s)). Tools hidden from context to save space."
