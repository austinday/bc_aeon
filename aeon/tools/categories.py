"""
Tool Category Registry for the Aeon Agent.

Defines hierarchical tool categories that allow the agent to expand/collapse
groups of tools to manage context space. Top-level tools are always visible.
Categorized tools are only shown when their category is expanded.

Structure:
  Each category dict can have:
    - 'description': Brief description shown when collapsed
    - 'tools': List of tool names directly in this category
    - 'subcategories': Dict of nested categories (same structure, arbitrary depth)
"""

# Tools that are ALWAYS visible in the agent's context (core workflow tools).
# Any registered tool NOT listed here and NOT in any category is also shown as top-level.
TOP_LEVEL_TOOLS = {
    'think', 'say_to_user', 'get_user_input',
    'open_file', 'close_file', 'write_file', 'str_replace',
    'search_web', 'run_command', 'task_complete',
    'run_command_async', 'job_output', 'kill_job',
    'memorize', 'forget',
    'verify_self_modification', 'restart_aeon',
    'expand_tool_category', 'collapse_tool_category',
}

# Hierarchical tool categories.
# When collapsed, only the category description + tool count is shown.
# When expanded, subcategories and/or tool descriptions become visible.
TOOL_CATEGORIES = {
    'image_tools': {
        'description': 'AI image generation, editing, analysis, and print preprocessing (FLUX/ComfyUI, Qwen3.5 vision)',
        'tools': [
            'generate_image',
            'edit_image',
            'analyze_image',
        ],
    },
    'video_tools': {
        'description': 'AI video creation & editing (LTX-2.3 via ComfyUI): text/image-to-video, '
                       'storyboard keyframes, extend, and restyle. See the video_director skill for '
                       'orchestrating multi-shot, character-consistent sequences from any assets.',
        'tools': [
            'generate_video',
        ],
    },
    'web_browser': {
        'description': 'Undetected web browsing, DOM extraction, and Set-of-Mark interaction',
        'tools': [
            'browser_navigate',
            'browser_interact',
            'browser_close_tab',
        ],
    },
}


def get_category_at_path(path: str):
    """Retrieve a category node by its slash-separated path (e.g. 'image_tools').
    Returns the category dict or None if not found."""
    parts = path.strip('/').split('/')
    node = {'subcategories': TOOL_CATEGORIES}
    for part in parts:
        subs = node.get('subcategories', {})
        if part not in subs:
            return None
        node = subs[part]
    return node


def get_all_category_paths():
    """Return a flat list of all valid category paths (e.g. ['image_tools', ...])."""
    paths = []
    def _walk(prefix, categories):
        for name, cat in categories.items():
            path = f'{prefix}/{name}' if prefix else name
            paths.append(path)
            if 'subcategories' in cat:
                _walk(path, cat['subcategories'])
    _walk('', TOOL_CATEGORIES)
    return paths


def get_tools_in_category(path: str):
    """Return all tool names under a category path (recursively including subcategories)."""
    cat = get_category_at_path(path)
    if cat is None:
        return set()
    tools = set(cat.get('tools', []))
    for sub_name in cat.get('subcategories', {}):
        sub_path = f'{path}/{sub_name}'
        tools |= get_tools_in_category(sub_path)
    return tools


def get_all_categorized_tools():
    """Return the set of all tool names that belong to any category."""
    all_tools = set()
    for path in get_all_category_paths():
        cat = get_category_at_path(path)
        if cat:
            all_tools.update(cat.get('tools', []))
    return all_tools


def count_tools_in_category(path: str):
    """Count total tools under a category path (recursively)."""
    return len(get_tools_in_category(path))
