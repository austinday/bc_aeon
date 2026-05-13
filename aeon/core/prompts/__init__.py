"""
Central loader for all prompts, directives, and tool descriptions.
All LLM-facing text should be loaded from this module.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent

def _load(filename: str) -> str:
    """Load a prompt file and return its contents stripped of trailing whitespace."""
    filepath = _PROMPTS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    return filepath.read_text(encoding='utf-8').rstrip()

# =============================================================================
# CORE DIRECTIVES
# =============================================================================
CORE_DIRECTIVES = _load('core_directives.txt')
DOCKER_DIRECTIVES = _load('docker_directives.txt')
IMPORTANT_REMINDERS = _load('important_reminders.txt')

# =============================================================================
# AGENT INSTRUCTIONS
# =============================================================================
PRIMARY_AGENT_INSTRUCTIONS = _load('primary_agent_instructions.txt')

# =============================================================================
# LLM PROMPT TEMPLATES (for llm.py)
# =============================================================================
COMPRESS_ACTION_LOG_PROMPT = _load('compress_action_log_prompt.txt')
ANALYZE_INTERRUPTION_PROMPT = _load('analyze_interruption_prompt.txt')
SUMMARIZE_TEXT_PROMPT = _load('summarize_text_prompt.txt')
COMPRESS_MEMORIES_PROMPT = _load('compress_memories_prompt.txt')

# =============================================================================
# TOOL PROMPT TEMPLATES
# =============================================================================
THINK_TOOL_PROMPT = _load('think_tool_prompt.txt')

# =============================================================================
# TOOL DESCRIPTIONS
# =============================================================================
TOOL_DESC_THINK = _load('tool_desc_think.txt')
TOOL_DESC_SAY_TO_USER = _load('tool_desc_say_to_user.txt')
TOOL_DESC_OPEN_FILE = _load('tool_desc_open_file.txt')
TOOL_DESC_CLOSE_FILE = _load('tool_desc_close_file.txt')
TOOL_DESC_WRITE_FILE = _load('tool_desc_write_file.txt')
TOOL_DESC_STR_REPLACE = _load('tool_desc_str_replace.txt')
TOOL_DESC_SEARCH_WEB = _load('tool_desc_search_web.txt')
TOOL_DESC_RUN_COMMAND = _load('tool_desc_run_command.txt')
TOOL_DESC_TASK_COMPLETE = _load('tool_desc_task_complete.txt')
TOOL_DESC_GET_USER_INPUT = _load('tool_desc_get_user_input.txt')
TOOL_DESC_MEMORIZE = _load('tool_desc_memorize.txt')
TOOL_DESC_FORGET = _load('tool_desc_forget.txt')
# === SUB-AGENT TOOLS ===
TOOL_DESC_SPAWN_SUB_AGENT = _load('tool_desc_spawn_sub_agent.txt')
TOOL_DESC_GET_SUB_AGENT_REPORT = _load('tool_desc_get_sub_agent_report.txt')
TOOL_DESC_KILL_SUB_AGENT = _load('tool_desc_kill_sub_agent.txt')
# === GENERATIVE TOOLS (ComfyUI-backed) ===
TOOL_DESC_GENERATE_IMAGE = _load('tool_desc_generate_image.txt')
TOOL_DESC_EDIT_IMAGE = _load('tool_desc_edit_image.txt')
TOOL_DESC_GENERATE_VIDEO = _load('tool_desc_generate_video.txt')
# === VISION TOOLS (vLLM-backed) ===
TOOL_DESC_ANALYZE_IMAGE = _load('tool_desc_analyze_image.txt')
# === WEB BROWSER TOOLS ===
TOOL_DESC_BROWSER_NAVIGATE = _load('tool_desc_browser_navigate.txt')
TOOL_DESC_BROWSER_INTERACT = _load('tool_desc_browser_interact.txt')
TOOL_DESC_BROWSER_CLOSE_TAB = _load('tool_desc_browser_close_tab.txt')
TOOL_DESC_BROWSER_SWITCH_TAB = _load('tool_desc_browser_switch_tab.txt')
# === SELF-MODIFICATION TOOLS ===
TOOL_DESC_RESTART_AEON = _load('tool_desc_restart_aeon.txt')
TOOL_DESC_EXPAND_CATEGORY = _load('tool_desc_expand_category.txt')
TOOL_DESC_COLLAPSE_CATEGORY = _load('tool_desc_collapse_category.txt')

# =============================================================================
# CONTEXT SECTIONS
# =============================================================================
OBJECTIVE_SECTION = _load('objective_section.txt')
TOOLS_SECTION = _load('tools_section.txt')
