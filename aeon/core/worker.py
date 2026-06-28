import json
import re
import time
import sys
import os
import uuid
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import List, Any, Dict, Callable, Optional

from .llm import LLMClient
from .system_info import get_runtime_info
from .logger import get_logger
from .utils import estimate_tokens
from .prompts import (
    CORE_DIRECTIVES,
    DOCKER_DIRECTIVES,
    IMPORTANT_REMINDERS,
    PRIMARY_AGENT_INSTRUCTIONS,
    TOOLS_SECTION,
    OBJECTIVE_SECTION
)
from aeon.core.skills.manager import SkillsManager

# Colors for terminal output
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[95m'
C_RESET = '\033[0m'
C_BLUE = '\033[96m'

# Tools through which the principal actively engages its sub-agents. Touching any
# of them resets the "you're ignoring your students" idle nudge.
SUB_AGENT_TOOLS = {
    "spawn_sub_agent", "gather_sub_agents", "get_sub_agent_report",
    "kill_sub_agent", "steer_sub_agent", "get_sub_agent_status",
}


class Worker:
    def __init__(self, llm_client: LLMClient, tools: List[Any] = None, print_func: Callable = print, debug_mode: bool = False, debug_log_path: Optional[str] = None):
        self.llm_client = llm_client
        self.debug_log_path = debug_log_path
        self.tools = {tool.name: tool for tool in tools} if tools else {}        
        # Ensure prompt files exist for all tools and categories
        from aeon.core.prompts.manager import ensure_prompt_files
        from aeon.tools.categories import get_all_category_paths
        ensure_prompt_files(list(self.tools.keys()), get_all_category_paths())
        
        self.logger = get_logger()
        self.print_func = print_func
        self.debug_mode = debug_mode

        # Initialize debug logging ONCE per worker instance
        self._debug_initialized = False
        if self.debug_mode:
            self._init_debug_logging()

        # --- STATE MODEL ---
        self.current_plan = "No plan formulated yet."
        self.open_files = {}
        self.memories = {}  # Key-value persistent memory
        self.last_observation = "None."
        self.action_log = []  # Persistent factual record of attempts (intents + results)
        self.open_files_mtime = {}  # Tracks last modified time of open files to avoid redundant reads
        self.pending_iteration_state = None # Holds intent/actions while awaiting result
        self._recent_commands = []  # Rolling window for loop detection
        self._recent_outputs = []   # Corresponding outputs for loop detection
        self.expanded_categories = set()  # Tracks which tool categories are currently expanded
        self.notified_sub_agents = set()  # Tracks which sub-agent terminal results the principal has actively collected (read/gathered)
        self.stuck_reason = None  # Set by loop-detection; a sub-agent publishes this so its principal sees it's looping
        self._blackboard_seen = 0  # Line count of the shared blackboard at last digest, to report new findings
        self._last_sub_agent_action_iter = 0  # Iteration the principal last engaged a sub-agent tool (for the idle nudge)
        self.open_files_access_order = []  # Tracks order of file access for LRU suggestions
        self.recent_intents = deque(maxlen=3)  # Tracks recent intents for loop detection
        self.prev_prompt_tokens = 0  # Tracks context size of previous iteration for growth metrics
        self.action_log_summary = ""  # Non-destructive summary of older action log entries
        self.instance_id = str(uuid.uuid4())[:8]  # Unique ID for this Aeon run instance
        self.MAX_REPEAT_WINDOW = 5  # How many recent commands to track
        self.REPEAT_THRESHOLD = 2   # How many identical commands before warning
        self.effective_iterations = 0
        self.prompt_cache = {}  # Cache for tool and category directives to avoid disk I/O

        # Load directives from central prompts module
        self.base_directives = CORE_DIRECTIVES
        self.docker_directives = DOCKER_DIRECTIVES
        self.important_reminders = IMPORTANT_REMINDERS
        self.max_history_tokens = 30000
        self.current_objective = None
        self.model_name = None  # Set by main.py for restart persistence
        self.active_skill = None  # {'path': ..., 'content': ...} when a skill protocol is active

    def _init_debug_logging(self):
        """Initialize debug logging once per worker instance."""
        if self._debug_initialized:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_path = Path.home() / f"aeon_debug_{ts}.log"
        self.print_func(f"{C_YELLOW}Debug logging enabled: {self.debug_path}{C_RESET}")
        self._debug_initialized = True

    def _sync_open_files(self, max_content_len: int = 250000):
        """Synchronize open_files cache with disk state using mtime to avoid redundant reads."""
        from aeon.tools.analyzers import FileAnalyzer
        paths = list(self.open_files.keys())
        for path in paths:
            if not os.path.exists(path):
                del self.open_files[path]
                if path in self.open_files_mtime:
                    del self.open_files_mtime[path]
                self.logger.info(f"Removed deleted file from context: {path}")
                continue
            try:
                current_mtime = os.path.getmtime(path)
                if self.open_files_mtime.get(path) == current_mtime:
                    # We only skip if the content is already within the current limit.
                    # If the limit decreased, we might need to re-sync to truncate.
                    if len(self.open_files.get(path, "")) <= max_content_len:
                        continue
                
                analyzer = FileAnalyzer(path)
                result = analyzer.analyze()
                summary_type = result.get('summary_type', '')
                
                if summary_type == 'opaque_binary':
                    content = f"File '{path}' is a binary file that cannot be displayed. Use a script to analyze it."
                elif summary_type == 'error':
                    content = f"Error reading file: {result.get('error_message', 'Unknown error')}"
                elif summary_type in ('empty_file', 'empty'):
                    content = '(empty file)'
                elif summary_type == 'full_content':
                    raw = result.get('content', '')
                    if isinstance(raw, (dict, list)):
                        content = json.dumps(raw, indent=2)
                    else:
                        content = str(raw)
                else:
                    parts = [f'[File Summary: {summary_type}]']
                    for key, value in result.items():
                        if key in ('file_name', 'file_size_bytes', 'summary_type'):
                            continue
                        if isinstance(value, (dict, list)):
                            parts.append(f'{key}: {json.dumps(value, indent=2, default=str)}')
                        else:
                            parts.append(f'{key}: {value}')
                    content = '\n'.join(parts)

                if len(content) > max_content_len:
                    content = f"File '{path}' content is too large ({len(content):,} chars) to open directly. Limit is {max_content_len:,} chars. Use a script to analyze this file."

                if self.open_files[path] != content:
                    self.open_files[path] = content
                
                # Update mtime cache after successful sync
                self.open_files_mtime[path] = current_mtime
            except Exception as e:
                self.logger.error(f"Error syncing file {path}: {e}")

    def register_tools(self, tools_list: List[Any]):
        for tool in tools_list:
            tool.worker = self
            self.tools[tool.name] = tool

    def update_open_file(self, path: str, content: str):
        abs_path = os.path.abspath(path)
        self.open_files[abs_path] = content
        
        # LRU Update: Move to end of list (most recent)
        if abs_path in self.open_files_access_order:
            self.open_files_access_order.remove(abs_path)
        self.open_files_access_order.append(abs_path)
        
        try:
            self.open_files_mtime[abs_path] = os.path.getmtime(abs_path)
        except OSError:
            pass

    def close_file(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        target = None
        if abs_path in self.open_files:
            target = abs_path
        elif path in self.open_files:
            target = path
        
        if target:
            del self.open_files[target]
            if target in self.open_files_access_order:
                self.open_files_access_order.remove(target)
            return True
        return False

    def is_file_open(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        return abs_path in self.open_files or path in self.open_files

    def _get_active_tool_directives(self) -> str:
        """Collect directives from currently expanded categories and all active tools
        (top-level tools + tools in expanded categories)."""
        from aeon.tools.categories import (
            TOP_LEVEL_TOOLS, 
            get_all_categorized_tools, get_tools_in_category
        )
        from aeon.core.prompts.manager import load_cat_prompt, load_tool_prompt
        
        active_directives = []
        categorized = get_all_categorized_tools()
        
        # Determine which tools are currently "active" (visible)
        active_tool_names = set(TOP_LEVEL_TOOLS)
        # Add tools that are not categorized at all
        for name in self.tools:
            if name not in categorized:
                active_tool_names.add(name)
        
        # Add tools in expanded categories
        for cat_path in self.expanded_categories:
            active_tool_names.update(get_tools_in_category(cat_path))
            
        # Process tools in alphabetical order for consistency
        for name in sorted(active_tool_names):
            if name not in self.prompt_cache:
                self.prompt_cache[name] = load_tool_prompt(name)
            tool_directives = self.prompt_cache[name]
            for d in tool_directives:
                active_directives.append(f"- {name}: {d}")
        
        # Process expanded categories in alphabetical order
        for cat_path in sorted(self.expanded_categories):
            if cat_path not in self.prompt_cache:
                self.prompt_cache[cat_path] = load_cat_prompt(cat_path)
            cat_directives = self.prompt_cache[cat_path]
            for d in cat_directives:
                active_directives.append(f"- {cat_path}: {d}")            
        if not active_directives:
            return ""            
        return "\n".join(active_directives)
    def _get_skills_description(self) -> str:
        """Build skills description with category-aware rendering."""
        from aeon.core.skills.manager import SkillsManager
        sm = SkillsManager()
        
        # We need to find all categories in the skills directory
        # Since SkillsManager doesn't have a list_categories, we derive it from the filesystem
        try:
            skills_root = sm.base_dir
            categories = [d.name for d in skills_root.iterdir() if d.is_dir()]
        except Exception as e:
            return f"Error loading skills categories: {e}"

        if not categories:
            return "No skills available."

        active_path = self.active_skill.get('path') if self.active_skill else None

        lines = ["**SKILLS** (reusable step-by-step protocols; they are NOT applied automatically)"]
        if active_path:
            lines.append(f"ACTIVE PROTOCOL: {active_path} (pinned in full below; call deactivate_skill once its steps are complete).")
        else:
            lines.append(
                "No skill is active. If the current objective matches one of the protocols below, call "
                "activate_skill('<category>/<skill_name>') BEFORE starting work so the protocol is pinned "
                "and followed. To read a protocol first without committing, use expand_tool_category('<category>')."
            )

        for cat in sorted(categories):
            # A skill category is 'expanded' (browsable) when its skill: key is set.
            is_expanded = f"skill:{cat}" in self.expanded_categories
            skills = sm.get_skills_in_category(cat)

            if is_expanded:
                lines.append(f"[-] {cat}:")
                for skill in sorted(skills):
                    content = sm.get_skill_content(cat, skill)
                    summary = (content[:200].replace('\n', ' ') + "...") if content else "(empty)"
                    marker = " (ACTIVE)" if active_path == f"{cat}/{skill}" else ""
                    lines.append(f"  - {cat}/{skill}{marker}: {summary}")
            else:
                count = len(skills)
                lines.append(f"[+] {cat}: ({count} skill{'s' if count != 1 else ''})")
        
        return "\n".join(lines)

    def _get_tools_description(self) -> str:
        """Build tool descriptions with category-aware rendering.

        Top-level tools are always shown with full descriptions.
        Categorized tools are only shown when their category is expanded.
        Uncategorized tools (not in TOP_LEVEL_TOOLS or any category) are shown as top-level.
        """
        from aeon.tools.categories import (
            TOOL_CATEGORIES, TOP_LEVEL_TOOLS,
            get_all_categorized_tools,
        )
        categorized = get_all_categorized_tools()

        # Part 1: Top-level tools (always visible with full descriptions)
        top_level_descs = []
        for name, tool in self.tools.items():
            if name in TOP_LEVEL_TOOLS or name not in categorized:
                top_level_descs.append(f"- {name}: {tool.description}")

        result = "\n\n".join(top_level_descs)

        # Part 2: Tool categories (collapsible tree)
        category_lines = self._render_categories(TOOL_CATEGORIES, '', 0)
        if category_lines:
            result += '\n\n**TOOL CATEGORIES** (use expand_tool_category / collapse_tool_category to manage)\n'
            result += '\n'.join(category_lines)

        return result

    def _render_categories(self, categories: dict, parent_path: str, depth: int) -> list:
        """Recursively render tool categories as a tree with [+]/[-] indicators."""
        from aeon.tools.categories import count_tools_in_category

        lines = []
        indent = '  ' * depth

        for name, cat in categories.items():
            path = f'{parent_path}/{name}' if parent_path else name
            # Check both raw path and skill-prefixed path
            is_expanded = (path in self.expanded_categories) or (f"skill:{path}" in self.expanded_categories)
            desc = cat.get('description', '')
            tool_count = count_tools_in_category(path)

            if is_expanded:
                lines.append(f'{indent}[-] {name}: {desc}')

                # Show direct tools in this category with full descriptions
                for tool_name in cat.get('tools', []):
                    if tool_name in self.tools:
                        lines.append(f'{indent}  - {tool_name}: {self.tools[tool_name].description}')
                    else:
                        lines.append(f'{indent}  - {tool_name}: (not loaded)')

                # Recurse into subcategories
                if 'subcategories' in cat:
                    lines.extend(self._render_categories(
                        cat['subcategories'], path, depth + 1
                    ))
            else:
                suffix = f' ({tool_count} tool{"s" if tool_count != 1 else ""})'
                lines.append(f'{indent}[+] {name}: {desc}{suffix}')

        return lines

    def _format_open_files(self, max_content_len: int = 250000) -> str:
        self._sync_open_files(max_content_len=max_content_len)
        if not self.open_files:
            return "No files currently open."
        try:
            from aeon.core.paths import PROJECT_ROOT
            root = PROJECT_ROOT
        except Exception:
            root = None

        def _disp(p):
            try:
                return os.path.relpath(p, root) if root else p
            except Exception:
                return p

        manifest = ", ".join(_disp(p) for p in self.open_files)
        out = [
            f"{len(self.open_files)} file(s) are ALREADY loaded in full below "
            f"({manifest}). Their complete current contents are in your context. "
            f"Do NOT call open_file on any of these — read them where they are and let "
            f"your next action advance the task."
        ]
        for path, content in self.open_files.items():
            out.append(f"--- FILE: {_disp(path)}  (abs: {path}) ---\n{content}\n--- END FILE: {_disp(path)} ---")
        return "\n\n".join(out)

    def _format_sub_agent_digest(self, current_iteration: int) -> str:
        """Build the always-on SUB-AGENTS awareness block, injected EVERY turn.

        This is the mechanism that lets the principal behave like an advisor
        watching its graduate students instead of blocking to poll them: each
        turn it passively sees every running agent's live step, activity age,
        and stall/loop/freeze flags, plus any finished-but-unread reports and
        new shared-blackboard findings -- with no blocking call. Returns '' when
        there is nothing to report so the section disappears entirely.
        """
        from aeon.core.sub_agent_state import resolve, norm_status, read_progress
        base = Path(os.getcwd()) / "aeon_output" / self.instance_id / "sub_agents"
        if not base.exists():
            return ""
        dirs = [d for d in base.iterdir() if d.is_dir() and (d / "pid.txt").exists()]
        if not dirs:
            return ""

        running = 0
        flagged = False
        lines = []
        for d in sorted(dirs, key=lambda p: p.name):
            sid = d.name.split("-")[0]
            is_term, status, _ = resolve(d)
            if is_term:
                base_status = norm_status(status)
                if f"{d.name}_{base_status}" in self.notified_sub_agents:
                    continue  # already collected -> don't clutter the digest
                if base_status == "COMPLETED":
                    lines.append(f"- [{sid}] ✓ FINISHED, report UNREAD — "
                                 f"read it now with get_sub_agent_report(agent_id='{sid}').")
                elif base_status == "KILLED":
                    lines.append(f"- [{sid}] KILLED (uncollected).")
                else:
                    lines.append(f"- [{sid}] ✗ {status} (unread) — "
                                 f"get_sub_agent_report(agent_id='{sid}').")
                continue
            running += 1
            pr = read_progress(d)
            age = pr["age"]
            age_str = f"{age:.0f}s ago" if age is not None else "unknown"
            sfx = (f" on '{pr['step']}'" if pr["step"] else "") + \
                  (f" (iter {pr['iteration']})" if pr["iteration"] else "")
            if pr["frozen"]:
                flagged = True
                lines.append(f"- [{sid}] ⚠ FROZEN — stopped heartbeating; it cannot recover. "
                             f"kill_sub_agent(agent_id='{sid}').")
            elif pr["stuck_reason"]:
                flagged = True
                lines.append(f"- [{sid}] ⚠ LOOPING — {pr['stuck_reason']} "
                             f"steer_sub_agent(agent_id='{sid}', guidance=...) with a new approach, or kill_sub_agent.")
            elif age is not None and age > 180:
                flagged = True
                lines.append(f"- [{sid}] ⚠ STALLED — no progress for {age:.0f}s{sfx}. "
                             f"Confirm with get_sub_agent_report, then steer_sub_agent or kill_sub_agent.")
            else:
                lines.append(f"- [{sid}] RUNNING (healthy) — last progress {age_str}{sfx}.")

        if not lines:
            return ""

        out = [
            "**SUB-AGENTS** (your dispatched graduate students; you are their advisor). "
            "Review this EVERY turn: steer the ones drifting, read finished reports, relay useful "
            "findings between them, and meanwhile keep advancing your OWN orthogonal work. There is "
            "NO blocking wait — never sit idle just because students are running."
        ]
        out.extend(lines)

        # New shared-blackboard findings since the last turn.
        try:
            bb = Path(os.getcwd()) / "aeon_output" / "blackboard.jsonl"
            if bb.exists():
                with bb.open("r", encoding="utf-8") as f:
                    count = sum(1 for _ in f)
                new = count - self._blackboard_seen
                self._blackboard_seen = count
                if new > 0:
                    out.append(f"→ {new} new finding(s) on the shared blackboard since last turn "
                               f"— call blackboard_read, then relay anything relevant to the right student via steer_sub_agent.")
        except Exception:
            pass

        # Engagement nudge: students running but unsupervised for several turns.
        idle_turns = current_iteration - self._last_sub_agent_action_iter
        if running and idle_turns >= 3:
            out.append(f"→ {running} student(s) have been running for {idle_turns} turns without you "
                       f"engaging them. Check their progress and steer/redirect as needed, or push your own "
                       f"orthogonal work forward — do not leave them unsupervised.")
        elif flagged:
            out.append("→ Flagged students above need attention: steer them with a corrected approach, "
                       "or kill_sub_agent the ones whose work you no longer need.")

        return "\n".join(out)

    def _format_memories(self) -> str:
        if not self.memories:
            return "No memories recorded yet."
        
        formatted = []
        for k, v in self.memories.items():
            if isinstance(v, dict):
                val = v.get('value', '')
                cat = v.get('category', 'general')
                ts = v.get('timestamp', 'unknown')
                formatted.append(f"[{cat}] {k}: {val} (Saved: {ts})")
            else:
                formatted.append(f"{k}: {v}")
        return "\n".join(formatted)

    def _truncate_output(self, text: str, max_chars: int = 50000) -> str:
        """Deterministic head+tail truncation. Prioritizes tail (where errors appear)."""
        if len(text) <= max_chars:
            return text
        head_budget = max_chars // 4       # 25% head
        tail_budget = max_chars - head_budget  # 75% tail
        omitted = len(text) - max_chars
        return (
            text[:head_budget]
            + f"\n\n... [{omitted:,} CHARS TRUNCATED] ...\n\n"
            + text[-tail_budget:]
        )

    def _format_attempt_log(self) -> str:
        """Format the full, uncompressed attempt log."""
        if not self.action_log and not self.pending_iteration_state:
            return "(No actions taken yet.)"
        
        lines = list(self.action_log)
        if self.pending_iteration_state:
            p = self.pending_iteration_state
            actions_str = ", ".join(p['actions'])
            lines.append(f"[Iter {p['iter']}]\n- Intent: {p['intent']}\n- Actions: {actions_str}\n- Result: (Pending...)")
            
        return "\n\n".join(lines)

    def _get_compressed_attempt_log(self, pressure: str = "Low") -> str:
        """Return a version of the action log suitable for the prompt (summary + recent).
        Adjusts the number of retained recent entries based on context pressure.
        """
        if not self.action_log and not self.pending_iteration_state:
            return "(No actions taken yet.)"
        
        # If log is small, just return full format
        full_log = self._format_attempt_log()
        if estimate_tokens(full_log) < 12000:
            return full_log
        
        # Dynamic window for recent entries based on pressure
        # Low: 12, Moderate: 8, High: 5, Critical: 3
        recent_map = {"Low": 12, "Moderate": 8, "High": 5, "CRITICAL": 3}
        recent_count = recent_map.get(pressure, 10)
        recent_entries = self.action_log[-recent_count:]
        
        lines = []
        if self.action_log_summary:
            lines.append(f"[HISTORICAL SUMMARY]\n{self.action_log_summary}")
        
        lines.extend(recent_entries)
        
        if self.pending_iteration_state:
            p = self.pending_iteration_state
            actions_str = ", ".join(p['actions'])
            lines.append(f"[Iter {p['iter']}]\n- Intent: {p['intent']}\n- Actions: {actions_str}\n- Result: (Pending...)")
            
        return "\n\n".join(lines)

    def _reset_state(self, initial_observation="Project started."):
        self.current_plan = "Initial state. Need to formulate a plan."
        self.open_files = {}
        self.memories = {}
        self.last_observation = initial_observation
        self.action_log.clear()
        self.pending_iteration_state = None
        self._recent_commands.clear()
        self._recent_outputs.clear()
        self.expanded_categories.clear()
        self.notified_sub_agents.clear()
        self.active_skill = None
        self.effective_iterations = 0
        self.stuck_reason = None
        self._blackboard_seen = 0
        self._last_sub_agent_action_iter = 0

    def serialize_state(self) -> dict:
        """Serialize worker state for persistence across restarts."""
        return {
            'memories': dict(self.memories),
            'current_plan': self.current_plan,
            'action_log': list(self.action_log),
            'action_log_summary': self.action_log_summary,
            'objective': self.current_objective or '',
            'expanded_categories': list(self.expanded_categories),
            'notified_sub_agents': list(self.notified_sub_agents),
            'active_skill': self.active_skill,
            'instance_id': self.instance_id,
            'open_files_list': list(self.open_files.keys()),
            'open_files_access_order': list(self.open_files_access_order),
        }

    def restore_state(self, state: dict):
        """Restore worker state from a previous serialization (used after restart)."""
        self.memories = state.get('memories', {})
        self.action_log = state.get('action_log', [])
        self.action_log_summary = state.get('action_log_summary', "")
        self.expanded_categories = set(state.get('expanded_categories', []))
        self.notified_sub_agents = set(state.get('notified_sub_agents', []))
        self.active_skill = state.get('active_skill', None)
        self.open_files_access_order = state.get('open_files_access_order', [])
        
        # Restore the list of open files (placeholders will be synced to actual content by _sync_open_files)
        open_files_list = state.get('open_files_list', [])
        for path in open_files_list:
            self.open_files[path] = "Restoring from state..."
        reason = state.get('reason', 'code changes')

        # Append a clear record that the restart happened
        self.action_log.append(
            f'[RESTART COMPLETED]\n'
            f'- Reason: {reason}\n'
            f'- pip install: SUCCESS\n'
            f'- Process relaunch: SUCCESS\n'
            f'- State restore: SUCCESS (memories, action log preserved)\n'
            f'- Result: Agent is NOW running the updated code. The restart is DONE.'
        )

        # Override the plan - the old plan is stale and will cause loops
        self.current_plan = (
            f'Restart completed successfully. The agent is now running with updated code ({reason}). '
            f'Next steps: verify the changes work as expected, then proceed with or complete the objective. '
            f'DO NOT call restart_aeon again unless you make additional NEW code changes.'
        )

        # Set last_observation with very explicit language to prevent re-restart loops
        self.last_observation = (
            f'=== RESTART COMPLETE ===\n'
            f'The agent process has been SUCCESSFULLY restarted. Details:\n'
            f'- Code changes applied: {reason}\n'
            f'- The updated code is NOW ACTIVE in this running process.\n'
            f'- All persistent memories and action history have been restored.\n'
            f'\n'
            f'CRITICAL: The restart is FINISHED. Do NOT call restart_aeon again.\n'
            f'Your code changes are ALREADY LIVE. Proceed with verifying them or completing the task.'
        )

    def _save_objective(self, objective: str):
        self.current_objective = objective
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{timestamp}] OBJECTIVE UPDATE:\n{objective}\n{'-'*40}\n"
            with open(".previous_objective.txt", "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            self.logger.error(f"Failed to save objective to file: {e}")

    def _build_primary_agent_context(self, tool_list_str: str, system_specs: str,
                                     memories_str: str, objective: str, open_files_str: str,
                                     active_tool_directives: str, attempt_log_str: str,
                                     context_diagnostics: str = "", sub_agent_digest: str = "") -> str:
        """Build the full context prompt for the Primary Agent."""
        reminders_section = f"**IMPORTANT REMINDERS**\n{self.important_reminders}\n\n" if self.important_reminders.strip() else ""

        tools_text = TOOLS_SECTION.format(tools=tool_list_str)
        objective_text = OBJECTIVE_SECTION.format(objective=objective)

        diag_section = f"\n**CONTEXT DIAGNOSTICS**\n{context_diagnostics}\n" if context_diagnostics else ""

        skills_text = self._get_skills_description()
        active_skill_section = self._format_active_skill()

        sub_agent_section = f"\n{sub_agent_digest}\n" if sub_agent_digest else ""

        return f"""{self.base_directives}

{self.docker_directives}

**OPEN TOOL DIRECTIVES**
{active_tool_directives if active_tool_directives else 'None'}

{tools_text}

{skills_text}

{reminders_section}**PERSISTENT MEMORIES**
{memories_str}

**ATTEMPT LOG** (Historical record of intents and results)
{attempt_log_str}

{system_specs}
{diag_section}
**CURRENT PLAN**
{self.current_plan}

**OPEN FILES**
===[ IN WORKING MEMORY ]===
{open_files_str}
===[ END OPEN FILES ]===

**LAST STEP RESULT**
{self.last_observation}
{sub_agent_section}{active_skill_section}
{PRIMARY_AGENT_INSTRUCTIONS}

{objective_text}"""

    def _format_active_skill(self) -> str:
        """Render the pinned active skill protocol block, or '' if none is active.

        This block is re-injected into EVERY prompt while a skill is active, which is
        what keeps the agent following an injection-based protocol instead of drifting.
        """
        if not self.active_skill:
            return ""
        path = self.active_skill.get('path', 'unknown')
        content = self.active_skill.get('content', '')
        return (
            f"\n**ACTIVE SKILL PROTOCOL: {path}**\n"
            f"You have committed to this protocol. Work through its steps in order and do NOT abandon it "
            f"until you call deactivate_skill. Where a step calls for a dedicated tool (memorize, "
            f"spawn_sub_agent, etc.), use that tool rather than improvising.\n"
            f"--- BEGIN PROTOCOL ---\n{content}\n--- END PROTOCOL ---\n"
        )

    def _resolve_tool_name(self, tool_name: str) -> Optional[str]:
        """Auto-correct a tool name only when there is exactly ONE unambiguous
        normalized match (case/dash/space differences). Returns the canonical
        name, or None if there is no safe single match."""
        if not tool_name:
            return None

        def norm(s):
            return s.lower().replace('-', '_').replace(' ', '_')

        target = norm(tool_name)
        matches = [name for name in self.tools if norm(name) == target]
        return matches[0] if len(matches) == 1 else None

    def _suggest_tools(self, tool_name: str, n: int = 3) -> str:
        """Return a ' Did you mean: ...' hint listing the closest real tool
        names, so the model can self-correct a hallucinated tool in one turn."""
        import difflib
        close = difflib.get_close_matches(tool_name, list(self.tools.keys()), n=n, cutoff=0.5)
        if not close:
            return " Use expand_tool_category to discover available tools."
        return f" Did you mean: {', '.join(close)}?"

    def _summarize_action(self, tool_name: str, params) -> str:
        """One-line, readable summary of a tool call for terminal display.

        Each parameter value is truncated so a huge payload (e.g. a full file in
        write_file) never floods the terminal."""
        if not isinstance(params, dict) or not params:
            return f"{tool_name}()"
        parts = []
        for k, v in params.items():
            v_str = str(v).replace('\n', ' ').strip()
            if len(v_str) > 50:
                v_str = v_str[:50] + '\u2026'
            parts.append(f"{k}={v_str}")
        inner = ", ".join(parts)
        if len(inner) > 220:
            inner = inner[:219] + '\u2026'
        return f"{tool_name}({inner})"

    def _clean_action_json(self, raw_str: str) -> str:
        clean_json = raw_str.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:].lstrip()
        elif clean_json.startswith("```"):
            clean_json = clean_json[3:].lstrip()
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3].rstrip()
        return clean_json.strip()

    # --- MAIN LOOP ---

    def _log_reasoning_trace(self, iteration, trace_data):
        if getattr(self, "debug_log_path", None):
            import json
            try:
                with open(self.debug_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trace_data) + "\n")
            except Exception:
                pass

    def run(self, objective: str, max_iterations: Optional[int] = None, step_callback: Optional[Callable[[int, int, str], None]] = None, terminal_tools: List[str] = None):
        if terminal_tools is None:
            terminal_tools = ['task_complete', 'restart_aeon']

        self.logger.info("Starting Execution for: %s", objective)
        self._save_objective(objective)

        iteration = 0
        self.last_observation = f"User input received: {objective}"
        self.print_func(f"{C_GREEN}Objective: {objective}{C_RESET}\n")

        graceful_exit_triggered = False

        while True:
            try:
                iter_start_time = time.time()
                iteration += 1
                self.llm_client.set_iteration(iteration)

                display_max = max_iterations if max_iterations is not None else 999
                if step_callback:
                    # Use the current intent as the step description for telemetry
                    step_desc = intent if 'intent' in locals() else "Thinking"
                    step_callback(iteration, display_max, step_desc)

                if max_iterations is not None and iteration > max_iterations:
                    if not graceful_exit_triggered:
                        graceful_exit_triggered = True
                        msg = f"SYSTEM ALERT: Max iterations ({max_iterations}) reached. You have ONE final step. You MUST use a terminal tool ({', '.join(terminal_tools)}) NOW to report your findings."
                        self.last_observation = msg
                        self.print_func(f"{C_RED}Max iterations reached. Forcing final report.{C_RESET}")
                    else:
                        self.print_func(f"{C_RED}Agent failed to exit. Terminating.{C_RESET}")
                        break

                self.print_func(f"\n{C_BLUE}{'='*60}\n ITERATION {iteration}\n{'='*60}{C_RESET}")

                if self.active_skill:
                    self.print_func(f"{C_GREEN}\U0001F3AF Active skill: {self.active_skill.get('path')}{C_RESET}")

                # --- BACKGROUND AGENT TERMINAL UI ---
                active_agents = []
                sub_agent_dir = Path(os.getcwd()) / "aeon_output" / self.instance_id / "sub_agents"
                if sub_agent_dir.exists():
                    for agent_dir in sub_agent_dir.iterdir():
                        if agent_dir.is_dir() and (agent_dir / "status.txt").exists():
                            if (agent_dir / "status.txt").read_text().strip() == "RUNNING":
                                active_agents.append(agent_dir.name[:8])
                if active_agents:
                    self.print_func(f"\033[90m[Background] Active sub-agents ({len(active_agents)}): {', '.join(active_agents)}\033[0m")

                # Non-destructive action log compression to preserve context focus
                full_log_str = self._format_attempt_log()
                if estimate_tokens(full_log_str) > 12000:
                    # Only re-compress if we have enough new entries to justify it (e.g., 5 new entries)
                    # or if no summary exists yet.
                    if not self.action_log_summary or len(self.action_log) % 5 == 0:
                        self.print_func(f"{C_CYAN}Updating action log summary to preserve context focus...{C_RESET}")
                        recent_count = 10
                        older_history = self.action_log[:-recent_count] if len(self.action_log) > recent_count else self.action_log
                        log_text = "\n\n".join(older_history)
                        self.action_log_summary = self.llm_client.compress_action_log(log_text)

                # --- SUB-AGENT AWARENESS DIGEST ---
                # Passive, always-on. Built every turn and injected into the prompt
                # so the principal continuously SEES what its students are doing
                # (and which need steering/reading) without any blocking poll. This
                # replaces the old fire-once "[SYSTEM ALERT]" notification, which
                # only fired on terminal transitions and prematurely marked agents
                # "collected". notified_sub_agents is now set ONLY when the principal
                # actively reads a report (gather/get_sub_agent_report).
                sub_agent_digest = self._format_sub_agent_digest(iteration)

                # --- PRE-PROMPT CONTEXT ANALYSIS ---
                # We estimate the prompt size first to determine pressure and dynamic limits
                # This allows us to pass diagnostics and adjust log size BEFORE the LLM call.
                
                # Initial rough estimate of the prompt without the action log and open files
                base_ctx_est = estimate_tokens(
                    self.base_directives + self.docker_directives + PRIMARY_AGENT_INSTRUCTIONS + 
                    self.current_plan + self.last_observation + objective
                )
                # Add estimates for other components
                tool_list_str = self._get_tools_description()
                active_tool_directives = self._get_active_tool_directives()
                memories_str = self._format_memories()
                # We use a default limit for the initial estimation
                open_files_str = self._format_open_files(max_content_len=250000)
                
                est_total = (
                    base_ctx_est + 
                    estimate_tokens(tool_list_str) + 
                    estimate_tokens(active_tool_directives) + 
                    estimate_tokens(memories_str) + 
                    estimate_tokens(open_files_str) + 
                    12000 # Buffer for action log
                )
                
                ctx_limit = self.llm_client.context_limit
                pressure_pct = (est_total / ctx_limit) * 100
                if pressure_pct < 50: pressure = "Low"
                elif pressure_pct < 80: pressure = "Moderate"
                elif pressure_pct < 95: pressure = "High"
                else: pressure = "CRITICAL"

                # Determine dynamic truncation limit based on context pressure
                if pressure == "Low": dyn_limit = 100000
                elif pressure == "Moderate": dyn_limit = 50000
                elif pressure == "High": dyn_limit = 20000
                else: dyn_limit = 10000

                # Re-format open files using the actual dynamic limit determined by pressure
                open_files_str = self._format_open_files(max_content_len=dyn_limit)

                # Gather final context components
                system_specs = get_runtime_info()
                attempt_log_str = self._get_compressed_attempt_log(pressure=pressure)

                # Automatic Memory Compression: Trigger if pressure is high and memories are significant
                if pressure in ["High", "CRITICAL"] and estimate_tokens(memories_str) > 2000:
                    self.print_func(f"{C_CYAN}Context pressure is {pressure}. Compressing memories to save space...{C_RESET}")
                    compressed_mems = self.llm_client.compress_memories(memories_str)
                    if compressed_mems:
                        self.memories = compressed_mems
                        memories_str = self._format_memories() # Update string for the current prompt
                        self.print_func(f"{C_GREEN}Memories compressed successfully.{C_RESET}")

                # Build the prompt, including diagnostics if pressure is elevated
                # This empowers the agent to proactively close files or summarize.
                diagnostic_str = ""
                if pressure != "Low":
                    breakdown = [
                        f"Context Pressure: {pressure} ({pressure_pct:.1f}%)",
                        f"Estimated Context: ~{est_total} / {ctx_limit} tokens",
                        f"Directives: ~{estimate_tokens(self.base_directives + self.docker_directives + PRIMARY_AGENT_INSTRUCTIONS)} tokens",
                        f"Active Tool Directives: ~{estimate_tokens(active_tool_directives)} tokens",
                        f"Tools: ~{estimate_tokens(tool_list_str)} tokens",
                        f"Memories: ~{estimate_tokens(memories_str)} tokens",
                        f"Attempt Log: ~{estimate_tokens(attempt_log_str)} tokens",
                        f"Open Files Total: ~{estimate_tokens(open_files_str)} tokens",
                    ]
                    if self.open_files:
                        for path, content in self.open_files.items():
                            breakdown.append(f"  - {os.path.basename(path)}: ~{estimate_tokens(content)} tokens")
                    diagnostic_str = "\n".join(breakdown)

                prompt = self._build_primary_agent_context(
                    tool_list_str, system_specs, memories_str, objective, open_files_str, 
                    active_tool_directives, attempt_log_str, context_diagnostics=diagnostic_str,
                    sub_agent_digest=sub_agent_digest
                )

                if max_iterations is not None:
                    rem_iters = max_iterations - self.effective_iterations
                    prompt += f"\n\nSYSTEM REMINDER: You have {rem_iters} effective iterations remaining to complete this task. Plan accordingly."
                    if rem_iters <= 0:
                        self.print_func(f"{C_RED}Iteration budget exhausted. Forcing final report.{C_RESET}")
                        self.last_observation = "SYSTEM ALERT: Iteration budget exhausted. You MUST use 'task_complete' to report your final status."

                # Final token count and growth tracking
                prompt_tokens = estimate_tokens(prompt)
                growth = prompt_tokens - self.prev_prompt_tokens
                growth_str = f"{growth:+} tokens" if self.prev_prompt_tokens > 0 else "N/A (first iter)"
                self.prev_prompt_tokens = prompt_tokens

                if prompt_tokens > ctx_limit * 0.85:
                    pct = prompt_tokens / ctx_limit * 100
                    self.print_func(f"{C_RED}WARNING: Prompt is ~{prompt_tokens} tokens ({pct:.0f}% of {ctx_limit} context limit). Close files or context will be truncated!{C_RESET}")
                    
                    # Size-aware LRU closing suggestions: prioritize old AND large files
                    if self.open_files_access_order:
                        scored_files = []
                        for idx, path in enumerate(self.open_files_access_order):
                            size = len(self.open_files.get(path, ""))
                            score = size / (idx + 1)
                            scored_files.append((score, path))
                        
                        scored_files.sort(key=lambda x: x[0], reverse=True)
                        top_candidates = [p for s, p in scored_files[:2]]
                        suggestions = ", ".join([os.path.basename(p) for p in top_candidates])
                        
                        # Inject suggestion into diagnostics so the agent sees it
                        if diagnostic_str:
                            diagnostic_str += f"\nSUGGESTION: Consider closing old/large files: {suggestions}"
                        self.print_func(f"{C_YELLOW}Suggestion: Consider closing old/large files: {suggestions}{C_RESET}")

                    if prompt_tokens > ctx_limit * 0.95:
                        raise RuntimeError(f"Context limit exceeded ({prompt_tokens} > {ctx_limit * 0.95} limit). Throwing error as requested.")

                self.print_func("Thinking (Primary Agent)...")

                # === PRIMARY AGENT CALL ===
                response_str = self.llm_client.get_primary_agent_response(prompt=prompt, diagnostic_str=diagnostic_str)
                if self.debug_mode:
                    pass

                try:
                    response_data = json.loads(response_str)
                except json.JSONDecodeError as e:
                    self.print_func(f"{C_RED}Primary Agent JSON Parse Error: {e}{C_RESET}")
                    self.last_observation = f"JSON Parse Error: {e}"
                    continue

                # Extract fields
                thought = response_data.get("thought", "(No thought provided)")
                previous_result_summary = response_data.get("previous_result_summary", "N/A (first turn)")
                intent = response_data.get("intent", "(No intent provided)")
                updated_plan = response_data.get("updated_plan")
                actions = response_data.get("actions", [])

                if self.debug_mode and hasattr(self, 'debug_path'):
                    try:
                        with open(self.debug_path, "a", encoding="utf-8") as f:
                            safe_plan = str(updated_plan).replace('\n', ' ')[:150]
                            f.write(f"[Iter {iteration}] Summary: {previous_result_summary[:100]}... | Thought: {thought[:100]}... | Intent: {intent} | Plan: {safe_plan}\n")
                    except Exception as e:
                        self.logger.warning(f"Failed to write to debug file: {e}")

                if updated_plan:
                    if isinstance(updated_plan, list):
                        self.current_plan = "\n".join(updated_plan)
                    else:
                        self.current_plan = str(updated_plan)

                self.print_func(f"\n{C_CYAN}--- PREVIOUS RESULT SUMMARY ---{C_RESET}")
                self.print_func(f"{previous_result_summary}")

                self.print_func(f"\n{C_CYAN}--- THOUGHT ---{C_RESET}")
                self.print_func(f"{thought}")
                
                self.print_func(f"\n{C_CYAN}--- INTENT ---{C_RESET}")
                self.print_func(f"{intent}")

                if updated_plan:
                    self.print_func(f"\n{C_CYAN}--- UPDATED PLAN ---{C_RESET}")
                    self.print_func(f"{self.current_plan}")

                if not actions:
                    self.print_func(f"{C_RED}No actions returned by agent.{C_RESET}")
                    self.last_observation = "Error: You returned an empty action list. You must take at least one action."
                    continue
                
                self.effective_iterations += 1

                # Resolve pending iteration state now that we have the summary
                if self.pending_iteration_state:
                    p = self.pending_iteration_state
                    acts_str = ", ".join(p['actions'])
                    finished_entry = f"[Iter {p['iter']}]\n- Intent: {p['intent']}\n- Actions: {acts_str}\n- Result: {previous_result_summary}"
                    self.action_log.append(finished_entry)
                    self.pending_iteration_state = None

                # === EXECUTION PHASE ===
                if step_callback:
                    step_callback(iteration, display_max, "Executing")

                self.print_func(f"\n{C_YELLOW}--- EXECUTION ---{C_RESET}")

                combined_summary_parts = []
                actions_taken_str = []
                full_actions_taken_str = []
                user_input_handled = False
                completion_blocked = False

                if len(actions) > 15:
                    actions = actions[:15]
                    self.logger.warning("Truncated actions to 15")

                # Show the agent's summarized tool-call choices for this turn up front,
                # before the (already-visible) tool output streams in below.
                queued = []
                for a in actions:
                    tn = (a.get("tool_name") or "?").strip()
                    queued.append(self._summarize_action(tn, a.get("parameters", {})))
                self.print_func(f"{C_CYAN}Tool calls ({len(actions)}): {' | '.join(queued)}{C_RESET}")

                for idx, action_data in enumerate(actions):
                    tool_name = action_data.get("tool_name")
                    params = action_data.get("parameters", {})

                    if not tool_name:
                        combined_summary_parts.append(f"Action {idx+1}: Missing tool_name.")
                        continue
                    
                    # Normalize tool name to prevent whitespace/case issues
                    tool_name = tool_name.strip()

                    if tool_name not in self.tools:
                        resolved = self._resolve_tool_name(tool_name)
                        if resolved:
                            # Unambiguous case/format variant (e.g. "Run_Command",
                            # "run-command") -> auto-correct so a trivial typo doesn't
                            # waste a whole iteration.
                            self.print_func(f"{C_YELLOW}Auto-corrected tool name '{tool_name}' -> '{resolved}'.{C_RESET}")
                            tool_name = resolved
                        else:
                            hint = self._suggest_tools(tool_name)
                            combined_summary_parts.append(
                                f"Action {idx+1}: Tool '{tool_name}' not found.{hint}")
                            continue

                    # Track active engagement with sub-agents so the idle nudge can
                    # tell when students are being left to drift unsupervised.
                    if tool_name in SUB_AGENT_TOOLS:
                        self._last_sub_agent_action_iter = iteration

                    full_action_desc = f"{tool_name}({params})" if params else f"{tool_name}()"
                    display_action_desc = f"{tool_name}({str(params)[:40]}...)" if params else f"{tool_name}()"
                    actions_taken_str.append(display_action_desc)
                    full_actions_taken_str.append(full_action_desc)

                    # Per-action marker so each tool's streamed output sits under its own header.
                    self.print_func(f"{C_BLUE}\u25B6 [{idx+1}/{len(actions)}] {self._summarize_action(tool_name, params)}{C_RESET}")

                    if tool_name in terminal_tools:
                        # HARD GUARD: don't let the principal finish while it still has
                        # dispatched students running or finished-but-unread. It must
                        # either collect their reports or, if it no longer needs the
                        # work, explicitly release them with kill_sub_agent.
                        if tool_name == "task_complete":
                            from aeon.tools.sub_agent import uncollected_sub_agents
                            sa_base = Path(os.getcwd()) / "aeon_output" / self.instance_id / "sub_agents"
                            pending = uncollected_sub_agents(sa_base, self.notified_sub_agents)
                            if pending:
                                still_running = [sid for sid, st in pending if st == "RUNNING"]
                                unread = [(sid, st) for sid, st in pending if st != "RUNNING"]
                                parts = ["COMMAND BLOCKED: you cannot task_complete while dispatched sub-agents are unresolved."]
                                if still_running:
                                    parts.append(
                                        "Still RUNNING: " + ", ".join(still_running) +
                                        ". Either let them finish and read their reports (get_sub_agent_report), "
                                        "or — if you ALREADY have what you need and no longer want their work — "
                                        "release each with kill_sub_agent(agent_id=...).")
                                if unread:
                                    parts.append(
                                        "Finished but UNREAD: " + ", ".join(f"{sid}({st})" for sid, st in unread) +
                                        ". Read each with get_sub_agent_report before finishing.")
                                parts.append("Resolve every student (collect or kill), THEN call task_complete.")
                                block_msg = " ".join(parts)
                                self.print_func(f"{C_RED}{block_msg}{C_RESET}")
                                self.last_observation = block_msg
                                self.action_log.append(
                                    f"[Iter {iteration}]\n- Intent: {intent}\n- Actions: task_complete\n"
                                    f"- Result: BLOCKED — unresolved sub-agents ({[sid for sid, _ in pending]}). "
                                    f"Collect or kill them first.")
                                completion_blocked = True
                                break
                        try:
                            tool = self.tools[tool_name]
                            result_str = str(tool.execute(**params))
                        except Exception as e:
                            result_str = f"Error executing terminal tool {tool_name}: {e}"
                        self.print_func(f"\n{C_GREEN}{result_str}{C_RESET}")
                        
                        self.pending_iteration_state = {
                            'iter': iteration,
                            'intent': intent,
                            'actions': actions_taken_str
                        }
                        p = self.pending_iteration_state
                        acts_str = ", ".join(p['actions'])
                        finished_entry = f"[Iter {p['iter']}]\n- Intent: {p['intent']}\n- Actions: {acts_str}\n- Result: Task marked complete. {result_str}"
                        self.action_log.append(finished_entry)
                        self.pending_iteration_state = None
                        
                        if step_callback: step_callback(iteration, display_max, "Complete")
                        return

                    elif tool_name == "get_user_input":
                        try:
                            self.print_func(f"{C_YELLOW}Agent Request: {params.get('prompt', 'Please provide input:')}\n> {C_RESET}")
                            user_in = input()
                        except EOFError:
                            return

                        self.print_func(f"{C_CYAN}Analyzing user input...{C_RESET}")
                        analysis = self.llm_client.analyze_interruption(objective, user_in)
                        classification = analysis.get('classification', 'ADVICE')
                        updated_text = analysis.get('updated_text', user_in)
                        reasoning = analysis.get('reasoning', '')

                        self.print_func(f"{C_CYAN}Classification: {classification} | Reason: {reasoning}{C_RESET}")

                        if classification == 'NEW_TASK':
                            self._reset_state()
                            objective = updated_text
                            self._save_objective(objective)
                            self.last_observation = f"New task received: {objective}"
                            self.print_func(f"{C_GREEN}New objective: {objective}{C_RESET}")
                            iteration = 0
                        elif classification == 'MODIFY_OBJECTIVE':
                            objective = updated_text
                            self._save_objective(objective)
                            self.last_observation = f"Objective modified: {objective}"
                            self.print_func(f"{C_GREEN}Modified objective: {objective}{C_RESET}")
                        else:  # ADVICE
                            prior_outputs = ""
                            if combined_summary_parts:
                                raw_output = "\n\n".join(combined_summary_parts)
                                raw_output = self._truncate_output(raw_output)
                                prior_outputs = f"Prior Action Outputs:\n{raw_output}\n\n"
                            
                            self.last_observation = f"{prior_outputs}User responded to prompt '{params.get('prompt', '')}': {updated_text}"
                            self.print_func(f"{C_GREEN}Input noted, continuing.{C_RESET}")

                        self.pending_iteration_state = {
                            'iter': iteration,
                            'intent': intent,
                            'actions': actions_taken_str
                        }
                        p = self.pending_iteration_state
                        acts_str = ", ".join(p['actions'])
                        finished_entry = f"[Iter {p['iter']}]\n- Intent: {p['intent']}\n- Actions: {acts_str}\n- Result: User input handled."
                        self.action_log.append(finished_entry)
                        self.pending_iteration_state = None
                        
                        user_input_handled = True
                        break  # Stop execution chain to process input

                    else:
                        try:
                            tool = self.tools[tool_name]
                            raw_result = tool.execute(**params)
                        except TypeError as e:
                            raw_result = f"Tool Parameter Error: {e}"
                        except Exception as e:
                            raw_result = f"Tool Execution Error: {type(e).__name__}: {e}"

                        result_str = str(raw_result)
                        # Truncate individual action output using a fraction of the dynamic limit
                        truncated_result = self._truncate_output(result_str, max_chars=dyn_limit // 5)
                        combined_summary_parts.append(f"Action {idx+1} ({tool_name}):\n{truncated_result}")

                        # Stop chain on command failure so the agent can react immediately
                        is_fail = "COMMAND FAILED" in result_str or result_str.strip().startswith("Error:")
                        if is_fail:
                            break

                # Skip summarization if user input was already handled directly
                if user_input_handled:
                    continue

                # task_complete was blocked (unresolved sub-agents). last_observation
                # already holds the block message; loop so the principal acts on it.
                if completion_blocked:
                    self.pending_iteration_state = None
                    continue

                # Deterministic truncation — raw output preserved, no LLM interpretation
                if not combined_summary_parts:
                    raw_output = "No actions produced output."
                else:
                    raw_output = "\n\n".join(combined_summary_parts)
                    # Use the dynamic limit based on context pressure
                    raw_output = self._truncate_output(raw_output, max_chars=dyn_limit)

                actions_list = ", ".join(actions_taken_str) if actions_taken_str else "none"
                self.last_observation = f"Actions: [{actions_list}]\nOutput:\n{raw_output}"

                # --- LOOP DETECTION ---
                # Build a fingerprint of the commands executed this iteration
                cmd_fingerprint = "|".join(full_actions_taken_str)
                output_fingerprint = raw_output.strip()[:2000]  # First 2k chars for comparison
                self._recent_commands.append(cmd_fingerprint)
                self._recent_outputs.append(output_fingerprint)
                # Keep only the rolling window
                if len(self._recent_commands) > self.MAX_REPEAT_WINDOW:
                    self._recent_commands.pop(0)
                    self._recent_outputs.pop(0)
                # Check for repeated identical command+output pairs
                loop_detected = False
                if len(self._recent_commands) >= self.REPEAT_THRESHOLD:
                    recent_pairs = list(zip(self._recent_commands[-self.REPEAT_THRESHOLD:], self._recent_outputs[-self.REPEAT_THRESHOLD:]))
                    if len(set(recent_pairs)) == 1:
                        loop_detected = True
                        repeat_count = self.REPEAT_THRESHOLD
                        # Count actual streak length
                        for i in range(len(self._recent_commands) - 1, -1, -1):
                            if (self._recent_commands[i], self._recent_outputs[i]) == recent_pairs[0]:
                                repeat_count = len(self._recent_commands) - i
                            else:
                                break
                        # Publish the loop so that, IF this worker is a sub-agent, its
                        # principal sees a LOOPING flag in the digest (a fast loop keeps
                        # the heartbeat fresh, so it would otherwise look healthy).
                        self.stuck_reason = (f"self-reported loop: ran the same command(s) {repeat_count}x "
                                             f"with identical output.")
                        loop_warning = (
                            f"\n\n** LOOP DETECTED: You have run the SAME command(s) {repeat_count} times in a row "
                            f"and received IDENTICAL output each time. The situation is NOT changing. **\n"
                            f"You MUST do something DIFFERENT now. You are STUCK.\n"
                            f"- You MUST use the `think` tool on your next turn to explicitly analyze WHY the previous command failed.\n"
                            f"- Write out the exact error message, list three possible root causes, and select the most likely one before taking any further action.\n"
                            f"- Zoom out and target a different part of the problem, or try a completely different approach.\n"
                            f"- DO NOT run the same command again."
                        )
                        self.last_observation += loop_warning
                        self.print_func(f"{C_RED}{loop_warning}{C_RESET}")
                if not loop_detected:
                    # Progress was made (commands/outputs changed) -> clear any prior
                    # loop flag so a recovered sub-agent stops showing as LOOPING.
                    self.stuck_reason = None

                # Cache the pending iteration state to be finalized next iteration
                self.pending_iteration_state = {
                    'iter': iteration,
                    'intent': intent,
                    'actions': actions_taken_str
                }

                iter_duration = time.time() - iter_start_time
                self.print_func(f"{C_CYAN}Iter {iteration} | {iter_duration:.2f}s | Prompt:{prompt_tokens} ({growth_str}) | Pressure:{pressure}{C_RESET}")

            except Exception as e:
                self.print_func(f"\n{C_RED}CRITICAL ERROR IN ITERATION: {e}{C_RESET}")
                self.logger.error(f"Iteration failed: {e}", exc_info=True)
                if "Context limit exceeded" in str(e):
                    raise
                time.sleep(2)

            except KeyboardInterrupt:
                self.print_func(f"\n{C_RED}PAUSED (User Interrupt).{C_RESET}")
                try:
                    self.print_func(f"{C_YELLOW}Enter guidance, press Enter to resume, or type 'exit' to quit.{C_RESET}")
                    user_guidance = input(f"{C_BLUE}User Guidance > {C_RESET}")

                    if not user_guidance.strip():
                        self.print_func("Resuming...")
                        continue

                    if user_guidance.lower() in ['exit', 'quit']:
                        self.print_func("Aborting task.")
                        break

                    # Analyze the interruption using the LLM
                    self.print_func(f"{C_CYAN}Analyzing guidance...{C_RESET}")
                    analysis = self.llm_client.analyze_interruption(objective, user_guidance)
                    classification = analysis.get('classification', 'ADVICE')
                    updated_text = analysis.get('updated_text', user_guidance)
                    reasoning = analysis.get('reasoning', '')

                    self.print_func(f"{C_CYAN}Classification: {classification} | Reason: {reasoning}{C_RESET}")

                    if classification == 'NEW_TASK':
                        self._reset_state()
                        objective = updated_text
                        self._save_objective(objective)
                        self.last_observation = f"New task received: {objective}"
                        self.print_func(f"{C_GREEN}New objective: {objective}{C_RESET}")
                        iteration = 0

                    elif classification == 'MODIFY_OBJECTIVE':
                        objective = updated_text
                        self._save_objective(objective)
                        self.last_observation = f"Objective modified: {objective}"
                        self.print_func(f"{C_GREEN}Modified objective: {objective}{C_RESET}")

                    else:  # ADVICE
                        self.last_observation = f"USER GUIDANCE: {updated_text}"
                        self.print_func(f"{C_GREEN}Guidance noted, continuing.{C_RESET}")

                except (KeyboardInterrupt, EOFError):
                    self.print_func(f"\n{C_RED}Forced Exit.{C_RESET}")
                    raise KeyboardInterrupt
