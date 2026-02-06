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
# AGENT INSTRUCTIONS (Planner, Executor, Milestone Analyzer)
# =============================================================================
PLANNER_INSTRUCTIONS = _load('planner_instructions.txt')
EXECUTOR_INSTRUCTIONS = _load('executor_instructions.txt')
MILESTONE_ANALYZER_INSTRUCTIONS = _load('milestone_analyzer_instructions.txt')

# =============================================================================
# LLM PROMPT TEMPLATES (for llm.py)
# =============================================================================
SUMMARIZE_EXECUTION_PROMPT = _load('summarize_execution_prompt.txt')
ANALYZE_INTERRUPTION_PROMPT = _load('analyze_interruption_prompt.txt')
SUMMARIZE_TEXT_PROMPT = _load('summarize_text_prompt.txt')

# =============================================================================
# TOOL PROMPT TEMPLATES
# =============================================================================
THINK_TOOL_PROMPT = _load('think_tool_prompt.txt')
RESEARCH_OBJECTIVE_TEMPLATE = _load('research_objective_template.txt')

# =============================================================================
# TOOL DESCRIPTIONS
# =============================================================================
TOOL_DESC_THINK = _load('tool_desc_think.txt')
TOOL_DESC_SAY_TO_USER = _load('tool_desc_say_to_user.txt')
TOOL_DESC_OPEN_FILE = _load('tool_desc_open_file.txt')
TOOL_DESC_CLOSE_FILE = _load('tool_desc_close_file.txt')
TOOL_DESC_WRITE_FILE = _load('tool_desc_write_file.txt')
TOOL_DESC_SEARCH_WEB = _load('tool_desc_search_web.txt')
TOOL_DESC_RUN_COMMAND = _load('tool_desc_run_command.txt')
TOOL_DESC_TASK_COMPLETE = _load('tool_desc_task_complete.txt')
TOOL_DESC_GET_USER_INPUT = _load('tool_desc_get_user_input.txt')
TOOL_DESC_CONDUCT_RESEARCH = _load('tool_desc_conduct_research.txt')
TOOL_DESC_DOCKER_EXEC = _load('tool_desc_docker_exec.txt')
TOOL_DESC_DOCKER_WRITE_FILE = _load('tool_desc_docker_write_file.txt')
TOOL_DESC_DOCKER_READ_FILE = _load('tool_desc_docker_read_file.txt')
TOOL_DESC_SUBMIT_FINDINGS = _load('tool_desc_submit_findings.txt')

# =============================================================================
# VESTIGIAL / LEGACY (kept for compatibility)
# =============================================================================
CONVERSATION_HISTORY = _load('conversation_history.txt')
OBJECTIVE_SECTION = _load('objective_section.txt')
RESPONSE_FORMAT = _load('response_format.txt')
TOOLS_SECTION = _load('tools_section.txt')
