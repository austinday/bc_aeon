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

# Read-only observation of background/passive state (sub-agents, jobs, blackboard).
# A turn made up ENTIRELY of these is the principal *watching* (polling) rather
# than *doing*. Two consequences: (1) such a turn must NOT trip loop detection —
# the background state is genuinely advancing even when the poll output looks
# byte-identical, so repeated check-ins are legitimate, not a stuck loop; (2) a
# run of such turns is the idle-babysitting anti-pattern we steer against.
OBSERVATION_TOOLS = {
    "gather_sub_agents", "get_sub_agent_report", "get_sub_agent_status",
    "job_output", "blackboard_read",
}
# Observation plus reflection/communication: none of these advance the actual
# task. A turn whose every action is passive means the principal did no real work.
PASSIVE_TOOLS = OBSERVATION_TOOLS | {"think", "say_to_user"}

# Tools that observe or reflect but do NOT change task/world state. The loop guard
# fingerprints only the CONSEQUENTIAL (non-passive) actions of a turn, so a model
# cannot launder a repeated dead action by padding the turn with a think() or a
# read() — which is exactly what the STUCK directive tells it to do, and exactly
# how a real run slipped the guard (thought it was clicking a button forever).
# browser_read is included: re-reading the page is inspection, not progress.
NON_CONSEQUENTIAL_TOOLS = PASSIVE_TOOLS | {"browser_read"}

# Substrings a tool emits when a state-changing action failed or changed nothing.
# Shared by _derive_ground_truth_outcome (builds the log tag) and
# _turn_made_no_progress (the boolean the semantic-stall detector keys on) so the
# two never drift.
NO_PROGRESS_ERROR_MARKERS = ("COMMAND FAILED", "Tool Execution Error", "Tool Parameter Error",
                             "Browser Error during", "Browser action failed", "Error during ",
                             "Error executing")

# Parameters that only change how a result is PRESENTED or ASSERTED, never what
# the action does to the page/world. The loop guard must ignore them when it
# fingerprints an action: a weak model re-clicking the same element re-decorates
# its own call every turn (adds/drops tab_id=default, toggles compare/visual,
# restates expected_text). If those incidental differences changed the
# fingerprint, the repeat streak would keep resetting and never reach the hard
# block — the exact "clicked Next forever, only ever got the soft notice" failure.
INCIDENTAL_PARAM_KEYS = frozenset({
    "include_vision", "visual", "compare", "expected_text",
})


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
        if self.tools:
            # Tools handed to the constructor (register_tools also refreshes).
            self._refresh_action_schema()

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
        # Type-ahead interruption is handled by the shared console reader
        # (aeon.core.console); the worker just enables it around a run and pulls
        # the stashed message. See _start_input_listener / _take_pending_message.
        self._recent_commands = []  # Rolling window for loop detection
        self._recent_outputs = []   # Corresponding outputs for loop detection
        self.expanded_categories = set()  # Tracks which tool categories are currently expanded
        self.notified_sub_agents = set()  # Tracks which sub-agent terminal results the principal has actively collected (read/gathered)
        self.notified_jobs = set()  # Tracks which background-job terminal results have been read (so the digest flags each once)
        self.stuck_reason = None  # Set by loop-detection; a sub-agent publishes this so its principal sees it's looping
        self._blackboard_seen = 0  # Line count of the shared blackboard at last digest, to report new findings
        self._last_sub_agent_action_iter = 0  # Iteration the principal last engaged a sub-agent tool (for the idle nudge)
        self._consecutive_passive_turns = 0  # Run of turns doing only observation/think/say (idle-babysitting detector)
        self.open_files_access_order = []  # Tracks order of file access for LRU suggestions
        self.recent_intents = deque(maxlen=3)  # Tracks recent intents for loop detection
        self._recent_turn_fps = deque(maxlen=3)  # Per-turn consequential fingerprint (parallels recent_intents) — lets the intent-stall tell varied work from spinning
        self._loop_blocked_fingerprint = None  # Consequential command under a hard loop block (refused until it changes)
        self._loop_block_hits = 0  # How many turns in a row the block has refused the same action (escalation)
        self._no_progress_streak = 0  # Consecutive state-changing turns that made no progress under the same approach
        self._last_struct_fp = ""  # Structural fingerprint (tool+verb, text dropped) of the last consequential turn
        self._stuck_banner = ""  # Top-of-prompt STUCK banner, set by loop/oscillation detection
        self.prev_prompt_tokens = 0  # Tracks context size of previous iteration for growth metrics
        self.action_log_summary = ""  # Non-destructive summary of older action log entries
        self._summarized_upto = 0  # Index into action_log below which entries are already folded into the summary
        self.instance_id = str(uuid.uuid4())[:8]  # Unique ID for this Aeon run instance
        # Cross-run session persistence (aeon_output/session_state.json). The
        # sub-agent wrapper turns this OFF: sub-agents share the principal's cwd
        # (workspace symlink), so with it on they clobber the principal's session
        # file every iteration AND inherit its memories at boot.
        self.persist_session = True
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
        self.last_say_to_user = None  # Most recent say_to_user text; a sub-agent's final report
        # Set by the resume_previous_session tool to a restored objective; the run
        # loop adopts it (with a fresh iteration budget) at the top of the next turn.
        self._resume_objective = None
        self.model_name = None  # Set by main.py for restart persistence
        self.active_skill = None  # {'path': ..., 'content': ...} when a skill protocol is active
        # Screenshot(s) to attach to the NEXT prompt so the multimodal model SEES
        # the current page as a human would. Set by the browser tool, consumed once
        # per turn, and never accumulated (only the latest view is ever attached).
        self.visual_context = []
        # Browser isolation unit: the principal uses the shared, persistent
        # 'default' profile (logins survive); sub_agent_wrapper overrides this so
        # each sub-agent browses in its own isolated context (own cookies/session).
        self.browser_profile = "default"

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
        self._refresh_action_schema()

    def _refresh_action_schema(self):
        """(Re)build the turn schema from the registered tools and hand it to the
        LLM client, which asks the server to grammar-constrain generation to it
        (vLLM structured outputs). This is what makes malformed JSON and
        hallucinated tool names impossible at the source instead of errors to
        recover from. Best-effort: on any failure the client keeps its previous
        schema (or None -> legacy parse path), never breaking the loop."""
        try:
            from aeon.core.action_schema import build_turn_schema
            if self.tools:
                self.llm_client.set_action_schema(build_turn_schema(list(self.tools.keys())))
        except Exception as e:
            self.logger.warning(f"Could not install structured-output schema: {e}")

    def set_visual_context(self, image_paths, replace: bool = True):
        """Register screenshot file path(s) for the multimodal model to look at on
        the NEXT turn. The browser tool calls this so the deciding model sees the
        rendered page directly. `replace=True` (default) keeps only the newest view
        so frames never accumulate across turns (bounded context, one image/turn)."""
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        image_paths = [p for p in (image_paths or []) if p]
        if replace:
            self.visual_context = list(image_paths)
        else:
            self.visual_context.extend(image_paths)

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

        # Lone-student anti-pattern: a SINGLE running sub-agent gives zero
        # parallelism — whatever it is doing, you (the principal) could be doing
        # that one thread yourself. The only reason to run one is if you are
        # ALSO working a different thread in parallel right now. Steer toward
        # either fanning out (more students for other independent threads) or
        # doing your own orthogonal work alongside it — never just watching it.
        if running == 1:
            out.append("→ You have only ONE student running. A lone sub-agent is no faster than doing "
                       "the work yourself — it only pays off if YOU are working a different thread in "
                       "parallel. So this turn: spawn additional sub-agents for other independent sub-tasks, "
                       "OR drive your own orthogonal work forward. Do NOT spend turns merely supervising a "
                       "single student.")

        # NOTE: the idle-poll anti-pattern (several turns of only watching/thinking)
        # is steered from the run loop's IDLE WARNING, which also covers background
        # jobs and the no-background-work case — not duplicated here.

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

    def _format_background_jobs_digest(self) -> str:
        """Build the always-on BACKGROUND JOBS block (the run_command_async
        counterpart to the SUB-AGENTS digest). Each turn the agent passively sees
        every running job's command + elapsed time, and any finished/failed job
        ONCE (until it reads it with job_output, which marks it notified). No
        blocking call. Returns '' when there is nothing to report."""
        from aeon.tools.jobs import resolve_job, read_command, status_keyword
        base = Path(os.getcwd()) / "aeon_output" / self.instance_id / "jobs"
        if not base.exists():
            return ""
        dirs = [d for d in base.iterdir() if d.is_dir() and (d / "pid.txt").exists()]
        if not dirs:
            return ""

        running = 0
        lines = []
        for d in sorted(dirs, key=lambda p: p.name):
            jid = d.name.split("-")[0]
            cmd = read_command(d)
            cmd_short = (cmd[:70] + "…") if len(cmd) > 70 else cmd
            is_term, status, _ = resolve_job(d)
            if is_term:
                kw = status_keyword(status)
                if f"{d.name}_{kw}" in self.notified_jobs:
                    continue  # already read -> don't clutter the digest
                if kw == "COMPLETED":
                    lines.append(f"- [{jid}] ✓ DONE (exit 0) — `{cmd_short}` — "
                                 f"read with job_output(job_id='{jid}').")
                elif kw == "KILLED":
                    lines.append(f"- [{jid}] KILLED — `{cmd_short}`.")
                elif kw == "TIMEOUT":
                    lines.append(f"- [{jid}] ⚠ {status} — `{cmd_short}` — "
                                 f"job_output(job_id='{jid}'); re-run with a larger timeout if needed.")
                else:
                    lines.append(f"- [{jid}] ✗ {status} — `{cmd_short}` — "
                                 f"job_output(job_id='{jid}').")
                continue
            running += 1
            try:
                el = time.time() - (d / "pid.txt").stat().st_mtime
                el_str = f"{el:.0f}s"
            except Exception:
                el_str = "?"
            lines.append(f"- [{jid}] RUNNING ({el_str}) — `{cmd_short}`.")

        if not lines:
            return ""
        out = [
            "**BACKGROUND JOBS** (detached commands you launched with run_command_async; non-blocking). "
            "A finished or failed job is flagged here ONCE — read it with job_output before relying on its "
            "result. Running jobs keep going while you work; kill_job to stop one. Don't idle-poll."
        ]
        out.extend(lines)
        return "\n".join(out)

    # Memories whose key/category names one of these are exact-value data
    # (credentials, tokens, IDs...). They are NEVER passed through the LLM memory
    # compressor — a paraphrased password or dropped API key is silent data loss —
    # but are merged back verbatim after compression.
    _SENSITIVE_MEMORY_MARKERS = ("credential", "password", "secret", "token",
                                 "key", "login", "auth", "account", "cookie")

    @classmethod
    def _is_sensitive_memory(cls, key: str, value) -> bool:
        hay = str(key).lower()
        if isinstance(value, dict):
            hay += " " + str(value.get("category", "")).lower()
        return any(m in hay for m in cls._SENSITIVE_MEMORY_MARKERS)

    def _format_memories(self, mems: Optional[dict] = None) -> str:
        if mems is None:
            mems = self.memories
        if not mems:
            return "No memories recorded yet."

        formatted = []
        for k, v in mems.items():
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

    @staticmethod
    def _normalize_cmd(text: str) -> str:
        """Normalize a command fingerprint so trivially reformatted-but-identical
        commands compare equal (whitespace only — commands are case-sensitive)."""
        return re.sub(r"\s+", " ", (text or "")).strip()

    @staticmethod
    def _canonical_params(params) -> str:
        """Canonicalize a tool's parameters for loop-fingerprinting: drop
        None-valued and presentation-only keys, treat a defaulted tab_id as absent,
        and sort what's left. This makes 'the same action' compare equal even when
        a weak model re-decorates its own call each turn (adds/drops tab_id=default,
        toggles compare/visual, restates expected_text) — the churn that used to
        keep the repeat streak from ever reaching the hard block, so a dead action
        (e.g. clicking the same Next button) only ever drew the soft notice and was
        allowed to spin forever."""
        if not isinstance(params, dict):
            return str(params)
        norm = {}
        for k, v in params.items():
            if v is None or k in INCIDENTAL_PARAM_KEYS:
                continue
            # Absent tab_id == "default": canonicalize both to "not present" so a
            # call gains/loses tab_id=default without changing its fingerprint.
            if k == "tab_id" and v in ("", "default"):
                continue
            norm[k] = v
        return "{" + ", ".join(f"{k}={norm[k]!r}" for k in sorted(norm)) + "}"

    def _consequential_fp(self, actions) -> str:
        """Fingerprint only the state-changing actions of a turn, dropping passive
        tools (think / read / observe). This is what the loop guard keys on, so
        `think + click(X)` and a bare `click(X)` share one fingerprint — padding a
        repeated dead action with a think() no longer disarms the block or resets
        the repeat streak. Parameters are canonicalized (see _canonical_params) so
        incidental call decoration doesn't mint a fresh fingerprint each turn.
        Returns "" for a turn that did nothing consequential (pure think/read),
        which the guard treats as transparent (neither a repeat nor a reset)."""
        parts = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            t = (a.get("tool_name") or a.get("tool") or "").strip()
            if not t or t in NON_CONSEQUENTIAL_TOOLS:
                continue
            p = a.get("parameters") or a.get("args") or {}
            parts.append(f"{t}({self._canonical_params(p)})")
        return self._normalize_cmd("|".join(parts))

    def _structural_fp(self, actions) -> str:
        """Coarser than _consequential_fp: for a tool that carries an action VERB
        (browser_interact's click/type, run_command's command word) it keeps the
        tool+verb but drops the free-text target — so two turns that make the same
        move while varying one incidental value (a fresh username on each signup
        attempt) share a fingerprint even though their _consequential_fp differs.
        This is what lets the semantic-stall detector see through that.

        But for a VERB-LESS tool the text argument IS the whole substance of the
        action — a search_web query, an image prompt, a write_file path. Collapsing
        those to the bare tool name made the stall detector treat genuinely
        DIFFERENT calls (two different web searches) as the SAME repeated move,
        firing 'semantic stall' on legitimate, varied work. So when there is no
        verb, fold the canonical params in, keeping distinct substantive calls
        distinct here too (identical repeats are still caught by the exact-repeat
        block, which keys on _consequential_fp)."""
        parts = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            t = (a.get("tool_name") or a.get("tool") or "").strip()
            if not t or t in NON_CONSEQUENTIAL_TOOLS:
                continue
            p = a.get("parameters") or a.get("args") or {}
            verb = ""
            if isinstance(p, dict):
                raw = str(p.get("action") or p.get("command") or "").strip()
                verb = raw.split()[0][:24] if raw.split() else ""
            if verb:
                parts.append(f"{t}:{verb}")
            else:
                parts.append(f"{t}({self._canonical_params(p)})")
        return "|".join(parts)

    @staticmethod
    def _turn_made_no_progress(raw_output: str, consequential: bool) -> bool:
        """Boolean form of the no-progress markers _derive_ground_truth_outcome scans:
        True when a state-CHANGING turn failed, was blocked, or changed nothing (URL +
        elements identical, or a form is still invalid). Passive turns are never
        'no progress' (inspection isn't an attempt). Used by the semantic-stall
        detector to count attempts that keep failing even as their params vary."""
        if not consequential:
            return False
        text = raw_output or ""
        low = text.lower()
        if "command blocked" in low:
            return True
        if any(m in text for m in NO_PROGRESS_ERROR_MARKERS):
            return True
        if "NO CHANGE:" in text or "FORM VALIDATION" in text:
            return True
        return False

    @staticmethod
    def _note_contradicts_outcome(note: str, outcome: str) -> bool:
        """True when the model's self-narration claims success/progress but the
        DERIVED ground truth says the turn failed or changed nothing. This is the
        exact confabulation that let a stuck agent write 'successfully advanced' for
        a no-op click; flagging it in the log stops that fiction from compounding."""
        if not note or not outcome:
            return False
        if not outcome.upper().startswith(
                ("NO EFFECT", "FORM STILL INVALID", "ERROR", "BLOCKED", "NO PROGRESS")):
            return False
        low = note.lower()
        success_words = ("success", "advanced", "proceeded", "completed", "filled in",
                         "now on", "loaded successfully", "moved to", "submitted", "accepted",
                         "worked", "went through", "was created", "logged in", "signed in",
                         "next step", "proceeding to")
        return any(w in low for w in success_words)

    @staticmethod
    def _normalize_output(text: str) -> str:
        """Normalize command output for loop comparison by stripping volatile
        tokens, so 'the same result' to a human compares equal even when a
        timestamp / counter / pid / address differs.

        Keying loop detection on raw byte-identity was too brittle: any single
        varying token made real loops slip through undetected. Trade-off: output
        whose only change is a genuinely-climbing counter also reads as
        'unchanged'. We accept that — an agent re-running the identical command
        3x is exactly the stuck pattern to break, and the hard block only forbids
        that one command, never a different next step."""
        if not text:
            return ""
        s = text[:2000]
        s = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", s)   # ANSI escape sequences
        s = re.sub(r"0x[0-9a-fA-F]+", "0xHEX", s)       # hex addresses / handles
        s = re.sub(r"\b[0-9a-fA-F]{8,}\b", "HEX", s)    # long hashes / uuid chunks
        s = re.sub(r"\d+", "N", s)                       # timestamps, pids, counters, elapsed
        s = re.sub(r"\s+", " ", s)                       # collapse whitespace
        return s.strip().lower()

    _INTENT_STOPWORDS = frozenset(
        "a an the is are was were be to of for and or why how i it this that on in "
        "at do does with my your we you re-".split())

    @classmethod
    def _intent_similarity(cls, a: str, b: str) -> float:
        """Jaccard overlap of the *content* words of two intent strings, in [0, 1].
        Used for the stall detector: the model rewords the same goal every turn, so
        exact string equality almost never fires. Stopwords are dropped first so
        two rewordings of one goal score high while unrelated intents that merely
        share filler words ('check the', 'do the') score low."""
        sa = {w for w in a.split() if w not in cls._INTENT_STOPWORDS}
        sb = {w for w in b.split() if w not in cls._INTENT_STOPWORDS}
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @staticmethod
    def _first_error_snippet(text: str, limit: int = 160) -> str:
        """First output line that names an error/failure, trimmed — enough to tell
        WHICH failure without dumping the whole result into the log."""
        for line in (text or "").splitlines():
            ll = line.lower()
            if "error" in ll or "failed" in ll:
                s = line.strip()
                return ": " + (s[:limit] + "…" if len(s) > limit else s)
        return "."

    @staticmethod
    def _derive_ground_truth_outcome(raw_output: str, consequential: bool,
                                     loop_detected: bool = False, repeat_count: int = 0) -> str:
        """Derive a factual, model-INDEPENDENT outcome tag for a turn from the ACTUAL
        tool output. This is the fix for the log recording the model's own
        `previous_result_summary` — the rosy self-narration that let a stuck agent
        write 'clicked Next' for a click that did nothing, so its own history never
        showed the no-op. Returns '' when the output shows nothing notable (caller
        then keeps the model's note as the record). Markers are emitted verbatim by
        the tools; scanned strongest-first (a block/error dominates a no-op, which
        dominates a still-invalid form). No-op and validation only count for a turn
        that actually tried to change something (a deliberate re-read is not a
        no-op)."""
        text = raw_output or ""
        low = text.lower()
        tag = ""
        if "command blocked" in low:
            tag = "BLOCKED by loop guard — the action was refused, not executed (it was a repeat)."
        elif any(m in text for m in NO_PROGRESS_ERROR_MARKERS):
            tag = "ERROR — the action failed" + Worker._first_error_snippet(text)
        elif consequential and "NO CHANGE:" in text:
            tag = ("NO EFFECT — the action did NOT change the page (URL + interactive elements "
                   "identical to before). Retrying it cannot help; the cause is a precondition "
                   "elsewhere (unfilled/invalid field, disabled or wrong target, needs scroll).")
        elif consequential and "FORM VALIDATION" in text:
            tag = "FORM STILL INVALID — a required field is unmet, so the submit/next control stays blocked."
        if loop_detected and repeat_count >= 2:
            streak = f" [LOOP: this same action has now repeated {repeat_count}x with no change]"
            tag = (tag + streak) if tag else ("NO PROGRESS —" + streak.lstrip())
        return tag

    def _collapse_repeated_entries(self, lines: list) -> list:
        """Collapse runs of attempt-log entries with the same actions AND result
        into a single entry annotated with the repeat count, so the model can
        literally SEE it has been repeating instead of inferring it from a long
        log. Compares on Actions + (ground-truth) Result, ignoring iter number,
        intent wording, and the subordinate 'Agent's note' — which the model
        rewords every turn and which used to defeat this collapse entirely."""
        def key(entry: str):
            m_a = re.search(r"- Actions: (.*?)(?:\n- Result:|\Z)", entry, re.S)
            m_r = re.search(r"- Result: (.*?)(?:\n- Agent's note:|\Z)", entry, re.S)
            a = re.sub(r"\s+", " ", (m_a.group(1) if m_a else "")).strip()
            r = re.sub(r"\s+", " ", (m_r.group(1) if m_r else "")).strip()
            return (a, r)

        out, i = [], 0
        while i < len(lines):
            k = key(lines[i])
            j = i + 1
            while j < len(lines) and k != ("", "") and key(lines[j]) == k:
                j += 1
            count = j - i
            if count > 1:
                out.append(lines[i].rstrip() +
                           f"\n- NOTE: this same action+result repeated {count}x in a row (no change).")
            else:
                out.append(lines[i])
            i = j
        return out

    def _format_attempt_log(self) -> str:
        """Format the full, uncompressed attempt log."""
        if not self.action_log and not self.pending_iteration_state:
            return "(No actions taken yet.)"

        lines = self._collapse_repeated_entries(list(self.action_log))
        if self.pending_iteration_state:
            p = self.pending_iteration_state
            actions_str = ", ".join(p['actions'])
            # The ground-truth outcome of the just-finished action is already known
            # (derived at stash time), so show it now instead of a bare "Pending" —
            # this is the model's most immediate, un-spun feedback on its last move.
            res = (p.get('outcome') or "").strip() or "(Pending...)"
            lines.append(f"[Iter {p['iter']}]\n- Intent: {p['intent']}\n- Actions: {actions_str}\n- Result: {res}")

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

        lines.extend(self._collapse_repeated_entries(recent_entries))

        if self.pending_iteration_state:
            p = self.pending_iteration_state
            actions_str = ", ".join(p['actions'])
            res = (p.get('outcome') or "").strip() or "(Pending...)"
            lines.append(f"[Iter {p['iter']}]\n- Intent: {p['intent']}\n- Actions: {actions_str}\n- Result: {res}")

        return "\n\n".join(lines)

    def _reset_state(self, initial_observation="Project started."):
        self.current_plan = "Initial state. Need to formulate a plan."
        self.open_files = {}
        self.memories = {}
        self.last_observation = initial_observation
        self.action_log.clear()
        self.action_log_summary = ""  # a stale summary must not describe the previous objective
        self._summarized_upto = 0
        self.pending_iteration_state = None
        self._recent_commands.clear()
        self._recent_outputs.clear()
        self._loop_blocked_fingerprint = None
        self._loop_block_hits = 0
        self._no_progress_streak = 0
        self._last_struct_fp = ""
        self._stuck_banner = ""
        self.recent_intents.clear()
        self._recent_turn_fps.clear()
        self.expanded_categories.clear()
        self.notified_sub_agents.clear()
        self.notified_jobs.clear()
        self.active_skill = None
        self.effective_iterations = 0
        self.stuck_reason = None
        self._blackboard_seen = 0
        self._last_sub_agent_action_iter = 0
        self._consecutive_passive_turns = 0
        self.visual_context = []
        self.last_say_to_user = None
        self._resume_objective = None

    def serialize_state(self) -> dict:
        """Serialize worker state for persistence across restarts."""
        return {
            'memories': dict(self.memories),
            'current_plan': self.current_plan,
            'action_log': list(self.action_log),
            'action_log_summary': self.action_log_summary,
            'summarized_upto': self._summarized_upto,
            'objective': self.current_objective or '',
            'expanded_categories': list(self.expanded_categories),
            'notified_sub_agents': list(self.notified_sub_agents),
            'notified_jobs': list(self.notified_jobs),
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
        self._summarized_upto = min(int(state.get('summarized_upto', 0) or 0), len(self.action_log))
        self.expanded_categories = set(state.get('expanded_categories', []))
        self.notified_sub_agents = set(state.get('notified_sub_agents', []))
        self.notified_jobs = set(state.get('notified_jobs', []))
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

    # --- CROSS-RUN PERSISTENCE ---
    # serialize_state/restore_state above cover the in-process restart_aeon hop
    # (via a /tmp pid file). These persist the same state to a STABLE, project-local
    # path so memories — and, for a resumed objective, the plan and attempt log —
    # survive the process exiting entirely. Without this a fresh `aeon.main` starts
    # with total amnesia despite "persistent memory" being a headline feature.

    def _session_state_path(self) -> Path:
        # aeon_output/ is gitignored and already the per-workspace output root, so
        # the file is naturally scoped to the project the agent is working in.
        return Path(os.getcwd()) / "aeon_output" / "session_state.json"

    def _stop_dump_path(self) -> Path:
        """Where an interrupted session's resumable state is written on stop.

        Distinct from session_state.json (the per-iteration auto-checkpoint): the
        NEXT process — e.g. one running the objective 'continue from where you left
        off' — overwrites session_state.json every turn, so a dedicated stop-dump
        is what the resume_previous_session tool reads to pick up the interrupted
        work faithfully."""
        return Path(os.getcwd()) / "aeon_output" / "interrupted_session.json"

    def _write_stop_dump(self, reason: str = "interrupted"):
        """Snapshot the current state to the stop-dump file so a later run can
        resume this objective when the user says 'continue from where you left
        off'. Best-effort: never raises into the shutdown/interrupt path."""
        if not self.persist_session:
            return
        try:
            path = self._stop_dump_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            data = self.serialize_state()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data['saved_at'] = ts
            data['stopped_at'] = ts
            data['stop_reason'] = reason
            data['pid'] = os.getpid()
            tmp = str(path) + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, default=str)
            os.replace(tmp, path)
            self.print_func(
                f"{C_YELLOW}\U0001F4BE State saved for resume ({path}). Next run, tell me "
                f"'continue from where you left off' to pick this up.{C_RESET}")
        except Exception as e:
            self.logger.warning(f"Failed to write stop dump: {e}")

    def resume_from_dump(self) -> str:
        """Load the previous session's stop dump (or the latest auto-checkpoint)
        and set up to CONTINUE its objective from where it left off. Restores
        memories, plan, attempt log, active skill, and open files, and signals the
        run loop to adopt the restored objective next turn. Returns a summary for
        the model. Backs the resume_previous_session tool."""
        # Prefer whichever of the stop-dump / auto-checkpoint is NEWER, so the most
        # recent activity wins (a clean Ctrl+C dump, or the last iteration before a
        # hard kill that never ran the stop path).
        candidates = [p for p in (self._stop_dump_path(), self._session_state_path()) if p.exists()]
        if not candidates:
            return ("No previous session state was found in this workspace "
                    f"({self._stop_dump_path()}). There is nothing to resume — ask the user what "
                    "objective to work on, or start fresh.")
        src = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            with open(src, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return f"Found a state dump at {src} but could not read it: {e}. Ask the user to restate the task."

        prev_obj = (data.get('objective') or '').strip()
        if not prev_obj:
            return (f"The state dump at {src} records no objective, so there is nothing concrete to "
                    "resume. Ask the user to restate the task.")

        mems = data.get('memories')
        if isinstance(mems, dict):
            self.memories = mems
        self.action_log = list(data.get('action_log') or [])
        self.action_log_summary = data.get('action_log_summary', "")
        self._summarized_upto = min(int(data.get('summarized_upto', 0) or 0), len(self.action_log))
        if data.get('current_plan'):
            self.current_plan = data['current_plan']
        self.active_skill = data.get('active_skill') or None
        self.expanded_categories = set(data.get('expanded_categories') or [])
        self.open_files_access_order = list(data.get('open_files_access_order') or [])
        # Placeholders; _sync_open_files repopulates real content from disk next turn.
        self.open_files = {p: "Restoring from state..." for p in (data.get('open_files_list') or [])}

        # Signal the run loop to switch the live objective to the restored one.
        self._resume_objective = prev_obj

        stopped_at = data.get('stopped_at') or data.get('saved_at') or 'a previous session'
        recent = self._collapse_repeated_entries(self.action_log[-4:]) if self.action_log else []
        recent_str = "\n\n".join(recent) if recent else "(no prior actions recorded)"
        return (
            f"RESUMED the previous session (stopped {stopped_at}).\n"
            f"- Objective restored: {prev_obj}\n"
            f"- Plan restored: {self.current_plan}\n"
            f"- Restored {len(self.memories)} memory item(s), {len(self.action_log)} attempt-log "
            f"entr(ies), {len(self.open_files)} open file(s).\n"
            f"- Most recent prior actions:\n{recent_str}\n\n"
            f"You are now continuing THAT objective from where it left off. Review the restored plan and "
            f"attempt log, then take the NEXT concrete step — do not restart work that is already done."
        )

    def _persist_session_state(self):
        """Atomically write the current state to the stable session file. Best-effort:
        any failure is logged and swallowed so persistence never breaks the loop."""
        if not self.persist_session:
            return
        try:
            path = self._session_state_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            data = self.serialize_state()
            data['saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fd, tmp = None, str(path) + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as e:
            self.logger.warning(f"Failed to persist session state: {e}")

    def _maybe_load_persisted_state(self, objective: str):
        """Once per process, hydrate from the stable session file if present.

        Memories (durable facts that transcend a single objective) are always
        restored. The plan and attempt log are objective-specific, so they are
        only restored when the persisted objective matches the one we are
        resuming — otherwise a brand-new task would inherit a stale plan and loop.
        Skipped entirely if state is already populated (e.g. a restart_aeon resume
        already ran restore_state), so we never clobber live state.
        """
        if not self.persist_session:
            return  # sub-agents neither inherit nor write the shared session file
        if getattr(self, '_persisted_loaded', False):
            return
        self._persisted_loaded = True
        # A restart resume (or any prior population) already set up state — don't touch it.
        if self.memories or self.action_log:
            return
        path = self._session_state_path()
        if not path.exists():
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load persisted session state: {e}")
            return

        restored = []
        mems = data.get('memories')
        if isinstance(mems, dict) and mems:
            self.memories = mems
            restored.append(f"{len(mems)} memorie(s)")

        prev_obj = (data.get('objective') or '').strip()
        if prev_obj and prev_obj == (objective or '').strip():
            if data.get('action_log'):
                self.action_log = list(data['action_log'])
                self.action_log_summary = data.get('action_log_summary', "")
                self._summarized_upto = min(int(data.get('summarized_upto', 0) or 0), len(self.action_log))
                restored.append(f"{len(self.action_log)} attempt-log entr(ies)")
            if data.get('current_plan'):
                self.current_plan = data['current_plan']
                restored.append("plan")

        if restored:
            saved_at = data.get('saved_at', 'a previous session')
            note = (f"SYSTEM: Restored persistent state from {saved_at} "
                    f"({', '.join(restored)}). Review your PERSISTENT MEMORIES before acting; "
                    f"some facts (paths, IDs, decisions) may be from earlier work.")
            self.last_observation = f"{self.last_observation}\n\n{note}" if self.last_observation else note
            self.print_func(f"{C_GREEN}\U0001F4BE {note}{C_RESET}")

    def _save_objective(self, objective: str):
        self.current_objective = objective
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{timestamp}] OBJECTIVE UPDATE:\n{objective}\n{'-'*40}\n"
            with open(".previous_objective.txt", "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            self.logger.error(f"Failed to save objective to file: {e}")

    def _recent_progress_digest(self, n: int = 6, max_chars: int = 3000) -> str:
        """A short digest of what the agent has actually done, for the interruption
        integrator to reason against so it never treats finished work as unstarted."""
        if not self.action_log:
            return "(nothing done yet)"
        recent = self._collapse_repeated_entries(self.action_log[-n:])
        return self._truncate_output("\n\n".join(recent), max_chars=max_chars)

    # ------------------------------------------------------------------
    # Type-ahead interruption: while a run is in flight the shared console
    # reader (aeon.core.console) accepts a typed line and interrupts this loop,
    # so a new instruction stops the current step and gets folded in — same
    # path as Ctrl+C. All actual reading (with full readline editing) happens in
    # that one reader; here we just enable/disable type-ahead around a run and
    # pull the message it stashed.
    # ------------------------------------------------------------------
    def _start_input_listener(self):
        from aeon.core.console import console
        console().enable_typeahead()

    def _stop_input_listener(self):
        from aeon.core.console import console
        console().disable_typeahead()

    def _take_pending_message(self):
        """Fetch and clear any unsolicited type-ahead line the reader stashed."""
        from aeon.core.console import console
        return console().take_pending()

    def _blocking_read_line(self, prompt: Optional[str] = None) -> str:
        """Read one line of SOLICITED input (get_user_input / guidance prompt)
        through the shared readline-backed console reader."""
        from aeon.core.console import console
        try:
            return console().readline(prompt or "")
        except (EOFError, KeyboardInterrupt):
            return ''

    def _integrate_user_input(self, objective: str, user_text: str, iteration: int):
        """Fold a mid-run user interruption into the ongoing work intelligently
        instead of the old erase-or-ignore binary. The integrator sees the
        objective, plan and progress, then picks:
          - REVISE : reconcile objective+plan with the input, keep all context;
          - CONSULT: goal unchanged, make the agent think about the input and
                     decide for itself whether to change course;
          - REPLACE: rare clean break -> wipe and restart.
        The user's message is also recorded durably in the action log so it is
        not lost once last_observation rolls over next turn.
        Returns (objective, reset_iteration)."""
        analysis = self.llm_client.integrate_interruption(
            objective, self.current_plan, self._recent_progress_digest(), user_text)
        mode = (analysis.get('mode') or 'REVISE').strip().upper()
        new_obj = (analysis.get('objective') or '').strip() or objective
        new_plan = (analysis.get('plan') or '').strip()
        directive = (analysis.get('directive') or '').strip()
        reasoning = (analysis.get('reasoning') or '').strip()
        self.print_func(f"{C_CYAN}Interruption -> {mode} | {reasoning}{C_RESET}")

        reset_iteration = False
        if mode == 'REPLACE':
            self._reset_state()
            objective = new_obj
            self._save_objective(objective)
            if new_plan:
                self.current_plan = new_plan
            self.last_observation = directive or f"New task: {objective}"
            reset_iteration = True
            self.print_func(f"{C_GREEN}New objective: {objective}{C_RESET}")
        elif mode == 'CONSULT':
            note = directive or (
                f"The user said: \"{user_text}\". Consider it, answer if it is a question, "
                f"then decide for yourself whether your current approach should change.")
            self.last_observation = (
                "** USER INTERJECTION (goal unchanged) **\n"
                f"{note}\n"
                "Use your `think` tool first to work through this before acting.")
            self.print_func(f"{C_GREEN}Consulting on input; objective preserved.{C_RESET}")
        else:  # REVISE (default)
            objective = new_obj
            self._save_objective(objective)
            if new_plan:
                self.current_plan = new_plan
            note = directive or f"The user's input has been folded into the objective."
            self.last_observation = (
                "** OBJECTIVE REVISED from user input **\n"
                f"{note}\n"
                f"Updated objective: {objective}\n"
                "Update your plan this turn (updated_plan) so it reflects BOTH what you have "
                "already completed and this change — do not restart finished work.")
            self.print_func(f"{C_GREEN}Objective revised: {objective}{C_RESET}")

        # Durable record: survives last_observation being overwritten next turn.
        self.action_log.append(
            f"[Iter {iteration}] USER INTERRUPTION\n- User said: {user_text}\n"
            f"- Handling: {mode} — {reasoning}")
        self.pending_iteration_state = None
        return objective, reset_iteration

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

        # A STUCK banner (set by loop/oscillation detection on the previous turn)
        # goes at the very TOP so a weak model can't miss it under the tool list.
        banner = f"{self._stuck_banner}\n\n" if self._stuck_banner else ""

        return f"""{banner}{self.base_directives}

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

    def _normalize_actions(self, actions) -> list:
        """Coerce the model's `actions` field into a clean list of action dicts.

        Tolerates two common model mistakes: emitting a single action object
        instead of a one-element array, and wrapping the call as
        {"tool_name": ..., "parameters": ...} vs {"tool": ..., "args": ...}.
        Drops entries that aren't dicts so the executor never iterates over
        stray strings.
        """
        if isinstance(actions, dict):
            # A single action object, or a dict-of-actions keyed by index/name.
            if "tool_name" in actions or "tool" in actions:
                actions = [actions]
            else:
                actions = list(actions.values())
        if not isinstance(actions, list):
            return []

        normalized = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            # Accept a few common key aliases for robustness.
            if "tool_name" not in a:
                if "tool" in a:
                    a["tool_name"] = a.get("tool")
                elif "name" in a:
                    a["tool_name"] = a.get("name")
            if "parameters" not in a:
                if "args" in a:
                    a["parameters"] = a.get("args")
                elif "params" in a:
                    a["parameters"] = a.get("params")
            normalized.append(a)
        return normalized

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

    def _tool_signature_hint(self, tool_name: str) -> str:
        """Return ' Expected parameters: ...' describing the tool's execute()
        signature, so a mis-shaped call can be corrected in the next turn."""
        tool = self.tools.get(tool_name)
        if tool is None:
            return ""
        try:
            import inspect
            sig = inspect.signature(tool.execute)
            required, optional = [], []
            for pname, p in sig.parameters.items():
                if pname == 'self' or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                if p.default is inspect.Parameter.empty:
                    required.append(pname)
                else:
                    optional.append(pname)
            parts = []
            if required:
                parts.append(f"required: {', '.join(required)}")
            if optional:
                parts.append(f"optional: {', '.join(optional)}")
            spec = '; '.join(parts) if parts else 'no parameters'
            return f" Expected parameters for {tool_name} ({spec})."
        except (ValueError, TypeError):
            return ""

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
        """Public entrypoint: run one objective to completion. Wraps the loop so a
        background stdin listener is live for its whole duration (letting the user
        type a message mid-run to interrupt it) and is always torn down afterward,
        handing stdin back to the REPL."""
        self._start_input_listener()
        try:
            return self._run_objective(objective, max_iterations=max_iterations,
                                       step_callback=step_callback, terminal_tools=terminal_tools)
        finally:
            self._stop_input_listener()

    def _run_objective(self, objective: str, max_iterations: Optional[int] = None, step_callback: Optional[Callable[[int, int, str], None]] = None, terminal_tools: List[str] = None):
        if terminal_tools is None:
            terminal_tools = ['task_complete', 'restart_aeon']

        self.logger.info("Starting Execution for: %s", objective)
        self._save_objective(objective)

        iteration = 0
        self.last_observation = f"User input received: {objective}"
        self._maybe_load_persisted_state(objective)

        # Pre-flight skill routing: one utility-model call that names the best-
        # matching skill protocol (or nothing) so the agent activates it on turn 1
        # instead of relying on the per-turn skill_check reflection to notice.
        # Fully best-effort — route_skills returns '' on any failure or no match.
        try:
            routing = self.llm_client.route_skills(objective)
        except Exception:
            routing = ""
        if routing:
            self.last_observation = f"{self.last_observation}\n\n{routing}"
            self.print_func(f"{C_CYAN}{routing}{C_RESET}")

        self.print_func(f"{C_GREEN}Objective: {objective}{C_RESET}\n")

        graceful_exit_triggered = False

        while True:
            try:
                # Type-ahead backstop: if the user submitted a message while the
                # previous step ran, fold it in before doing anything else. The
                # interrupt path (except KeyboardInterrupt) handles the common
                # case immediately; this catches a message whose interrupt landed
                # between iterations.
                pending = self._take_pending_message()
                if pending is not None:
                    objective, reset_iter = self._integrate_user_input(objective, pending, iteration)
                    if reset_iter:
                        iteration = 0

                # The resume_previous_session tool restored a prior session and
                # asked us to continue ITS objective — adopt it now with a fresh
                # iteration budget, so the rest of the loop plans against the real
                # task instead of the "continue from where you left off" instruction.
                if self._resume_objective:
                    objective = self._resume_objective
                    self._resume_objective = None
                    self._save_objective(objective)
                    self.effective_iterations = 0
                    iteration = 0
                    self.print_func(f"{C_GREEN}▶ Resuming previous objective: {objective}{C_RESET}")

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
                        # INCREMENTAL: fold only the entries not yet summarized into
                        # the existing summary, instead of re-summarizing the entire
                        # (ever-growing) history from scratch every 5 turns.
                        # Keep >= the largest display window (12 at Low pressure in
                        # _get_compressed_attempt_log) so no entry appears both
                        # summarized and verbatim.
                        recent_count = 12
                        cutoff = max(0, len(self.action_log) - recent_count)
                        new_entries = self.action_log[self._summarized_upto:cutoff]
                        if new_entries:
                            self.print_func(f"{C_CYAN}Updating action log summary to preserve context focus...{C_RESET}")
                            if self.action_log_summary:
                                log_text = (f"[EXISTING SUMMARY OF OLDER HISTORY]\n{self.action_log_summary}\n\n"
                                            f"[NEW ENTRIES TO FOLD INTO THE SUMMARY]\n" + "\n\n".join(new_entries))
                            else:
                                log_text = "\n\n".join(new_entries)
                            self.action_log_summary = self.llm_client.compress_action_log(log_text)
                            self._summarized_upto = cutoff

                # --- SUB-AGENT AWARENESS DIGEST ---
                # Passive, always-on. Built every turn and injected into the prompt
                # so the principal continuously SEES what its students are doing
                # (and which need steering/reading) without any blocking poll. This
                # replaces the old fire-once "[SYSTEM ALERT]" notification, which
                # only fired on terminal transitions and prematurely marked agents
                # "collected". notified_sub_agents is now set ONLY when the principal
                # actively reads a report (gather/get_sub_agent_report).
                sub_agent_digest = self._format_sub_agent_digest(iteration)
                # Background jobs ride the same always-on awareness channel as
                # sub-agents (same injection point, same auto-recovery rebuilds).
                jobs_digest = self._format_background_jobs_digest()
                if jobs_digest:
                    sub_agent_digest = f"{sub_agent_digest}\n\n{jobs_digest}" if sub_agent_digest else jobs_digest

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

                # Automatic Memory Compression: Trigger if pressure is high and memories are significant.
                # Sensitive entries (credentials/tokens/IDs) are exempted from the LLM
                # rewrite — a paraphrased password is silent data loss — and merged
                # back verbatim afterwards.
                if pressure in ["High", "CRITICAL"] and estimate_tokens(memories_str) > 2000:
                    self.print_func(f"{C_CYAN}Context pressure is {pressure}. Compressing memories to save space...{C_RESET}")
                    protected = {k: v for k, v in self.memories.items()
                                 if self._is_sensitive_memory(k, v)}
                    compressible = {k: v for k, v in self.memories.items() if k not in protected}
                    if compressible:
                        compressed_mems = self.llm_client.compress_memories(self._format_memories(compressible))
                        if compressed_mems:
                            self.memories = {**compressed_mems, **protected}  # protected survive verbatim
                            memories_str = self._format_memories() # Update string for the current prompt
                            kept = f" ({len(protected)} sensitive entr(ies) kept verbatim)" if protected else ""
                            self.print_func(f"{C_GREEN}Memories compressed successfully{kept}.{C_RESET}")

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

                # Screenshot(s) the browser produced last turn, to attach to THIS
                # call so the model sees the page. (The guidance note is appended
                # after context-shedding below, so a rebuilt prompt can't drop it.)
                turn_images = list(self.visual_context)

                if max_iterations is not None:
                    rem_iters = max_iterations - self.effective_iterations
                    prompt += f"\n\nSYSTEM REMINDER: You have {rem_iters} effective iterations remaining to complete this task. Plan accordingly."
                    if rem_iters <= 0:
                        self.print_func(f"{C_RED}Iteration budget exhausted. Forcing final report.{C_RESET}")
                        # Append to the PROMPT (already built): writing this to
                        # last_observation here would be overwritten at end of turn
                        # and never reach the model.
                        prompt += ("\nSYSTEM ALERT: Iteration budget exhausted. You MUST use "
                                   "'task_complete' THIS turn to report your final status.")

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
                        # GRACEFUL CONTEXT RECOVERY: rather than crash the whole
                        # run, shed the largest/oldest open files until we are back
                        # under 90%, then continue. The agent is told what was
                        # closed so it can re-open or memorize as needed. We only
                        # raise if there is nothing left to shed.
                        target = ctx_limit * 0.90
                        shed = []
                        while prompt_tokens > target and self.open_files_access_order:
                            scored = sorted(
                                ((len(self.open_files.get(p, "")) / (i + 1), p)
                                 for i, p in enumerate(self.open_files_access_order)),
                                reverse=True)
                            victim = scored[0][1]
                            self.close_file(victim)
                            shed.append(os.path.basename(victim))
                            open_files_str = self._format_open_files(max_content_len=dyn_limit)
                            prompt = self._build_primary_agent_context(
                                tool_list_str, system_specs, memories_str, objective, open_files_str,
                                active_tool_directives, attempt_log_str, context_diagnostics=diagnostic_str,
                                sub_agent_digest=sub_agent_digest)
                            prompt_tokens = estimate_tokens(prompt)

                        if shed:
                            note = (f"SYSTEM: Context exceeded 95% of the {ctx_limit}-token limit. "
                                    f"Auto-closed {len(shed)} large/old file(s) to recover: {', '.join(shed)}. "
                                    f"Re-open any you still need with open_file, and memorize key facts so "
                                    f"they survive future context pressure.")
                            prompt += f"\n\n{note}"
                            prompt_tokens = estimate_tokens(prompt)
                            self.print_func(f"{C_YELLOW}{note}{C_RESET}")

                        if prompt_tokens > ctx_limit * 0.95:
                            raise RuntimeError(
                                f"Context limit exceeded ({prompt_tokens} > {ctx_limit * 0.95:.0f}) "
                                f"with no open files left to shed.")

                # Append the screenshot guidance LAST — after any context-shedding
                # rebuild above — so it is always present when an image is attached.
                if turn_images:
                    prompt += (
                        "\n\n**ATTACHED SCREENSHOT (CURRENT BROWSER PAGE)**\n"
                        "The image attached to this message is the page exactly as it renders in the "
                        "browser right now — look at it directly, as a human would, to judge layout, read "
                        "text, spot what changed, and catch anything visual (CAPTCHAs, consent walls, "
                        "modals, errors). Then act using the [id]s in the INTERACTIVE ELEMENTS list, which "
                        "index the exact same page. The screenshot and the element list describe ONE page; "
                        "use both together."
                    )

                self.print_func("Thinking (Primary Agent)...")

                # === PRIMARY AGENT CALL ===
                # Attach the current page screenshot (if any) so the multimodal
                # model sees the page itself. Consume it now: a view is shown for
                # exactly the one decision that follows the browser action, never
                # re-sent (the model re-looks by calling browser_read).
                if turn_images:
                    self.print_func(f"{C_CYAN}\U0001F441 Attaching {len(turn_images)} page screenshot(s) for the model to see.{C_RESET}")
                self.visual_context = []
                response_str = self.llm_client.get_primary_agent_response(
                    prompt=prompt, diagnostic_str=diagnostic_str,
                    images=turn_images or None)
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
                # updated_plan is now an optional plain string. If a model still block-
                # encodes it and the block goes missing, an unsubstituted "__BLOCK_N__"
                # placeholder can arrive here — ignore it (keep the prior plan) rather
                # than storing the literal placeholder as the plan.
                if isinstance(updated_plan, str) and "__BLOCK" in updated_plan and len(updated_plan.strip()) < 20:
                    updated_plan = None
                actions = self._normalize_actions(response_data.get("actions", []))

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

                # A terminal tool (task_complete/restart_aeon) returns immediately
                # and drops any actions queued AFTER it. Models often place
                # task_complete first with the deliverable (say_to_user) after it,
                # which would lose the deliverable. Stably move terminal tools to
                # the END so every other action this turn still runs first.
                def _is_terminal(a):
                    return (a.get("tool_name") or "").strip() in terminal_tools

                if any(_is_terminal(a) for a in actions) and not _is_terminal(actions[-1]):
                    non_terminal = [a for a in actions if not _is_terminal(a)]
                    terminal = [a for a in actions if _is_terminal(a)]
                    if non_terminal:
                        self.print_func(f"{C_YELLOW}Reordered: running {len(non_terminal)} action(s) before the terminal tool so nothing queued after it is lost.{C_RESET}")
                    actions = non_terminal + terminal

                self.effective_iterations += 1

                # Resolve pending iteration state now that we have the summary
                if self.pending_iteration_state:
                    p = self.pending_iteration_state
                    acts_str = ", ".join(p['actions'])
                    # Ground truth (derived from the actual tool output last turn) is
                    # the authoritative Result; the model's own summary is kept only as
                    # a clearly-subordinate note. When ground truth flagged nothing
                    # notable (a normal, effective action), fall back to the model's
                    # summary as before — no need to second-guess a turn that worked.
                    outcome = (p.get('outcome') or "").strip()
                    note = self._truncate_output((previous_result_summary or "").strip(), max_chars=400)
                    if outcome:
                        finished_entry = (f"[Iter {p['iter']}]\n- Intent: {p['intent']}\n"
                                          f"- Actions: {acts_str}\n- Result: {outcome}\n"
                                          f"- Agent's note: {note or '(none)'}")
                        # Confabulation guard: if the model's note claims success but the
                        # measured Result says the turn failed/changed nothing, mark the
                        # mismatch inline so the fiction can't quietly drive the next turn.
                        if self._note_contradicts_outcome(note, outcome):
                            finished_entry += (
                                "\n- ⚠ NOTE-VS-REALITY MISMATCH: the note above claims progress, but the "
                                "measured Result says the action did NOT succeed. Trust the Result — do "
                                "not act as though it worked.")
                    else:
                        finished_entry = (f"[Iter {p['iter']}]\n- Intent: {p['intent']}\n"
                                          f"- Actions: {acts_str}\n- Result: {note}")
                    self.action_log.append(finished_entry)
                    self.pending_iteration_state = None

                # === EXECUTION PHASE ===
                if step_callback:
                    step_callback(iteration, display_max, "Executing")

                self.print_func(f"\n{C_YELLOW}--- EXECUTION ---{C_RESET}")

                combined_summary_parts = []
                actions_taken_str = []
                turn_tool_names = []  # Resolved tool names actually run this turn (for loop/idle classification)
                user_input_handled = False
                completion_blocked = False

                MAX_ACTIONS = 15
                dropped_actions = 0
                if len(actions) > MAX_ACTIONS:
                    dropped_actions = len(actions) - MAX_ACTIONS
                    actions = actions[:MAX_ACTIONS]
                    warn = (f"SYSTEM: You queued {dropped_actions + MAX_ACTIONS} actions; only the first "
                            f"{MAX_ACTIONS} run this turn. The remaining {dropped_actions} were dropped — "
                            f"re-issue them next turn after seeing these results.")
                    self.logger.warning(warn)
                    self.print_func(f"{C_YELLOW}{warn}{C_RESET}")
                    combined_summary_parts.append(warn)

                # Show the agent's summarized tool-call choices for this turn up front,
                # before the (already-visible) tool output streams in below.
                queued = []
                for a in actions:
                    tn = (a.get("tool_name") or "?").strip()
                    queued.append(self._summarize_action(tn, a.get("parameters", {})))
                self.print_func(f"{C_CYAN}Tool calls ({len(actions)}): {' | '.join(queued)}{C_RESET}")

                # --- HARD LOOP BLOCK (enforcement) ---
                # The previous turn tripped the 3x repeat guard and armed a block on
                # this exact consequential action. A prose warning is not enough for a
                # weak model, so we make the no-op impossible: if the proposed turn's
                # CONSEQUENTIAL fingerprint matches the looping one, refuse to run it
                # and force a different move. Crucially we key on the consequential
                # fingerprint (passive think/read stripped), so padding the turn with
                # a think() can't launder the same dead action past the block.
                if self._loop_blocked_fingerprint:
                    proposed_fp = self._consequential_fp(actions)
                    if proposed_fp and proposed_fp == self._loop_blocked_fingerprint:
                        self._loop_block_hits += 1
                        if self._loop_block_hits >= 3:
                            # Repeatedly hammering the same dead action: escalate from
                            # "try something different" to "this path is confirmed dead;
                            # switch categorically or surface the blocker."
                            block_msg = (
                                f"** COMMAND BLOCKED (loop guard, {self._loop_block_hits}x) — CONFIRMED DEAD END. **\n"
                                "This exact action has now been refused repeatedly; it does NOT work and will "
                                "never work. STOP trying variations of it (re-clicking the same element, "
                                "re-running the same command).\n"
                                "This turn you MUST do ONE of:\n"
                                "- Act on a DIFFERENT element / URL / target, or\n"
                                "- Execute your stated FALLBACK plan (a different method entirely), or\n"
                                "- If no path forward exists, call task_complete and report the blocker plainly.\n"
                                "Do NOT re-issue the blocked action; it will keep being refused."
                            )
                        else:
                            block_msg = (
                                "** COMMAND BLOCKED (loop guard): this is the SAME action that just produced no "
                                "change 3+ times, so it was NOT executed — running it again cannot help. **\n"
                                "You MUST do something DIFFERENT this turn:\n"
                                "- Use `think` (in a separate turn) to write the exact failure, three candidate "
                                "root causes, and pick one.\n"
                                "- Then act on a DIFFERENT element/target, change the approach, or switch sub-task.\n"
                                "Padding this action with a think() will NOT get it past the block — the action "
                                "itself must change."
                            )
                        self.print_func(f"{C_RED}{block_msg}{C_RESET}")
                        self.last_observation = block_msg
                        self.action_log.append(
                            f"[Iter {iteration}]\n- Intent: {intent}\n- Actions: {', '.join(queued)}\n"
                            f"- Result: BLOCKED by loop guard ({self._loop_block_hits}x; same consequential "
                            f"action, not executed).")
                        self._persist_session_state()
                        continue
                    if proposed_fp:
                        # A genuinely DIFFERENT consequential action -> loop broken; disarm.
                        self._loop_blocked_fingerprint = None
                        self._loop_block_hits = 0
                        self._stuck_banner = ""
                    # else: a pure think/read turn -> leave the block armed (thinking is
                    # allowed and encouraged) but do not disarm on padding alone.

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

                    display_action_desc = f"{tool_name}({str(params)[:40]}...)" if params else f"{tool_name}()"
                    actions_taken_str.append(display_action_desc)
                    turn_tool_names.append(tool_name)

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

                        # Fold this FINAL turn's outputs into last_observation so
                        # anything reading it after run() returns (e.g. the
                        # sub-agent wrapper's fallback report) sees this turn,
                        # not the previous one.
                        self.last_observation = self._truncate_output(
                            "\n\n".join(combined_summary_parts + [result_str]), max_chars=8000)

                        self._persist_session_state()
                        if step_callback: step_callback(iteration, display_max, "Complete")
                        return

                    elif tool_name == "get_user_input":
                        try:
                            user_in = self._blocking_read_line(
                                f"{C_YELLOW}Agent Request: {params.get('prompt', 'Please provide input:')}\n> {C_RESET}")
                        except EOFError:
                            return

                        if not user_in.strip():
                            # EOF / non-interactive stdin / bare Enter: there is
                            # no guidance to integrate. Feeding "" to the LLM
                            # integrator wasted a call and could rewrite the
                            # objective off nothing.
                            self.last_observation = (
                                "No user input was received (empty line or non-interactive session). "
                                "Do not ask again — proceed autonomously using your best judgment, and "
                                "surface any blocker in your final report (say_to_user / task_complete).")
                            user_input_handled = True
                            break

                        self.print_func(f"{C_CYAN}Integrating user input...{C_RESET}")
                        # Preserve any output produced earlier this same turn so it
                        # isn't lost when the interruption rewrites last_observation.
                        prior_outputs = ""
                        if combined_summary_parts:
                            raw_output = self._truncate_output("\n\n".join(combined_summary_parts))
                            prior_outputs = f"Prior action outputs this turn:\n{raw_output}\n\n"

                        objective, reset_iter = self._integrate_user_input(objective, user_in, iteration)
                        if prior_outputs:
                            self.last_observation = prior_outputs + self.last_observation
                        if reset_iter:
                            iteration = 0

                        user_input_handled = True
                        break  # Stop execution chain to process input

                    else:
                        try:
                            tool = self.tools[tool_name]
                            if not isinstance(params, dict):
                                raise TypeError(
                                    f"'parameters' must be a JSON object, got {type(params).__name__}")
                            raw_result = tool.execute(**params)
                        except TypeError as e:
                            # Surface the tool's real signature so the model can fix
                            # the call this turn instead of guessing again.
                            raw_result = f"Tool Parameter Error: {e}.{self._tool_signature_hint(tool_name)}"
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

                # --- TURN CLASSIFICATION (loop/idle detection) ---
                # An "observation-only" turn polls background state (sub-agents,
                # jobs, blackboard) and nothing else. Its output is naturally
                # near-identical turn-to-turn while work proceeds in the
                # background, so it must NOT feed loop detection (that produced
                # the false "LOOP DETECTED" on a principal legitimately checking
                # in on its students). A "passive" turn additionally counts pure
                # think/say turns as doing-no-real-work, for the idle-babysitting
                # detector below.
                ran = [t for t in turn_tool_names if t]
                observation_only = bool(ran) and all(t in OBSERVATION_TOOLS for t in ran)
                passive_turn = bool(ran) and all(t in PASSIVE_TOOLS for t in ran)
                if passive_turn:
                    self._consecutive_passive_turns += 1
                else:
                    self._consecutive_passive_turns = 0

                loop_detected = False
                repeat_count = 0  # consecutive-repeat streak, for the ground-truth outcome tag
                no_progress = False  # this (state-changing) turn failed / changed nothing — for the semantic-stall detector
                # Polling background state is legitimate even when its output is
                # byte-identical turn after turn, so skip the entire loop /
                # oscillation / stall machinery on observation-only turns. This is
                # what stops a principal that is correctly checking in on its
                # students (or background jobs) from being falsely flagged STUCK.
                if not observation_only:
                    # --- LOOP / REPEAT DETECTION ---
                    # Fingerprint only the CONSEQUENTIAL (state-changing) actions of the
                    # turn plus a noise-normalized view of output. Two reasons this is
                    # narrow: (1) keying on raw byte-identity was too brittle — a
                    # timestamp/pid/counter/ANSI code made "the same result" compare
                    # unequal, so real loops never tripped; (2) keying on the WHOLE turn
                    # let a model launder a repeated dead action by padding it with a
                    # think()/read() (which the STUCK directive literally tells it to do),
                    # resetting the streak and disarming the block. A pure think/read turn
                    # has an empty consequential fingerprint and is transparent here: it
                    # neither counts as a repeat nor resets an in-progress streak.
                    norm_cmd = self._consequential_fp(actions)
                    no_progress = self._turn_made_no_progress(raw_output, bool(norm_cmd))
                    if norm_cmd:
                        norm_out = self._normalize_output(raw_output)
                        self._recent_commands.append(norm_cmd)
                        self._recent_outputs.append(norm_out)
                        if len(self._recent_commands) > self.MAX_REPEAT_WINDOW:
                            self._recent_commands.pop(0)
                            self._recent_outputs.pop(0)

                        # Consecutive run of the identical action (output may vary), and
                        # of the identical action+output pair, counted back from this turn.
                        cmd_streak = 0
                        for c in reversed(self._recent_commands):
                            if c == norm_cmd:
                                cmd_streak += 1
                            else:
                                break
                        pair_streak = 0
                        for c, o in zip(reversed(self._recent_commands), reversed(self._recent_outputs)):
                            if c == norm_cmd and o == norm_out:
                                pair_streak += 1
                            else:
                                break

                        if cmd_streak >= self.REPEAT_THRESHOLD:
                            loop_detected = True
                            repeat_count = cmd_streak
                            out_phrase = ("identical output each time" if pair_streak >= cmd_streak
                                          else "no meaningful change in output")
                            # Graduated: the first repeat (2x) gets a measured nudge — an
                            # action can legitimately repeat once (e.g. a re-run after an
                            # unrelated edit). Escalate to the hard STUCK protocol AND arm
                            # an execution block only at 3x+, where it really is spinning.
                            if repeat_count <= 2:
                                loop_warning = (
                                    f"\n\n** REPEAT NOTICE: you ran the same action(s) twice with {out_phrase} — "
                                    f"no real change. Before repeating again, confirm a repeat will actually do "
                                    f"something different; if not, change the input, the approach, or the sub-task. **"
                                )
                                warn_color = C_YELLOW
                            else:
                                # Publish the loop so that, IF this worker is a sub-agent, its
                                # principal sees a LOOPING flag in the digest (a fast loop keeps
                                # the heartbeat fresh, so it would otherwise look healthy).
                                self.stuck_reason = (f"self-reported loop: ran the same action(s) {repeat_count}x "
                                                     f"with {out_phrase}.")
                                # Arm a HARD block on the consequential fingerprint: next turn
                                # this action is refused outright (a weak model ignores prose
                                # nudges) even if padded with a think(); raise a top-of-prompt
                                # banner it cannot miss. Reset the escalation counter.
                                self._loop_blocked_fingerprint = norm_cmd
                                self._loop_block_hits = 0
                                self._stuck_banner = (
                                    "===[ ! STUCK — READ THIS FIRST ]===\n"
                                    f"You have run the SAME action {repeat_count}x in a row with {out_phrase}. "
                                    "Repeating it is now BLOCKED and will not execute — padding it with a think() "
                                    "will NOT get it past the block. Do NOT try it again.\n"
                                    "This turn: use `think` to diagnose the failure, then act on a DIFFERENT "
                                    "target, change the approach, or switch to a different sub-task.\n"
                                    "===[ END STUCK ]==="
                                )
                                loop_warning = (
                                    f"\n\n** LOOP DETECTED: You have run the SAME action(s) {repeat_count} times in a row "
                                    f"with {out_phrase}. The situation is NOT changing. **\n"
                                    f"You MUST do something DIFFERENT now. You are STUCK.\n"
                                    f"- Repeating this action is now BLOCKED — it will be refused, not executed, and "
                                    f"padding it with a think() will not help.\n"
                                    f"- Use the `think` tool to state the exact error, list three possible root "
                                    f"causes, and select the most likely one.\n"
                                    f"- Then act on a DIFFERENT target, change the approach, or switch sub-tasks."
                                )
                                warn_color = C_RED
                            self.last_observation += loop_warning
                            self.print_func(f"{warn_color}{loop_warning}{C_RESET}")

                        # --- OSCILLATION (2-CYCLE) DETECTION ---
                        # The check above only catches CONSECUTIVE identical turns. An agent
                        # ping-ponging between two states (A,B,A,B) — e.g. toggling a setting
                        # back and forth — never has two identical turns in a row, so it
                        # slips through. Detect a repeated 2-cycle over the last 4 turns.
                        if not loop_detected and len(self._recent_commands) >= 4:
                            pairs = list(zip(self._recent_commands[-4:], self._recent_outputs[-4:]))
                            a, b, c, d = pairs
                            if a == c and b == d and a != b:
                                loop_detected = True
                                self.stuck_reason = ("self-reported oscillation: alternating between two "
                                                     "states (A,B,A,B) with no net progress.")
                                self._stuck_banner = (
                                    "===[ ! STUCK — READ THIS FIRST ]===\n"
                                    "You are alternating between TWO actions (A,B,A,B) and making no net progress. "
                                    "Break the cycle: use `think`, then choose a THIRD, different approach.\n"
                                    "===[ END STUCK ]==="
                                )
                                osc_warning = (
                                    "\n\n** OSCILLATION DETECTED: You are alternating between TWO actions/states "
                                    "(A, B, A, B) and making no net progress — each undoes or ignores the other. **\n"
                                    "Stop the back-and-forth. Use the `think` tool to identify why these two steps "
                                    "conflict, then choose a THIRD, different approach that breaks the cycle."
                                )
                                self.last_observation += osc_warning
                                self.print_func(f"{C_RED}{osc_warning}{C_RESET}")

                    # --- INTENT-LEVEL STALL DETECTION ---
                    # Catches spinning on the same GOAL across turns even when the exact
                    # commands/outputs vary. Uses fuzzy token overlap, not exact string
                    # equality — the model rewords the same intent every turn, so exact
                    # matching almost never fired.
                    norm_intent = re.sub(r"\s+", " ", (intent or "").strip().lower())[:160]
                    self.recent_intents.append(norm_intent)
                    self._recent_turn_fps.append(norm_cmd)
                    # Doing genuinely VARIED work is not a stall even if the stated
                    # goal (and its wording) holds steady: a research phase that
                    # fires three DIFFERENT searches shares an intent but is making
                    # progress, so the wording-similarity heuristic alone would
                    # false-fire. Suppress the warning when every turn in the window
                    # ran a DISTINCT consequential action.
                    window_fps = [f for f in self._recent_turn_fps if f]
                    varied_work = (len(window_fps) == self.recent_intents.maxlen
                                   and len(set(window_fps)) == len(window_fps))
                    if (not loop_detected and norm_intent and not varied_work
                            and len(self.recent_intents) == self.recent_intents.maxlen):
                        ints = [s for s in self.recent_intents if s]
                        if len(ints) == self.recent_intents.maxlen and all(
                                self._intent_similarity(ints[i], ints[i + 1]) >= 0.5
                                for i in range(len(ints) - 1)):
                            stall_note = (
                                f"\n\n** STALL WARNING: your stated intent has been essentially the same "
                                f"for {self.recent_intents.maxlen} turns ('{intent[:120]}') without resolving it. "
                                f"You may be making no real progress on this goal. Re-read your ATTEMPT LOG, "
                                f"question the assumption behind this intent, and either change approach or switch "
                                f"to a different sub-task. **"
                            )
                            self.last_observation += stall_note
                            self.print_func(f"{C_YELLOW}{stall_note}{C_RESET}")

                    # --- SEMANTIC STALL (varying-detail loop) ---
                    # The exact-fingerprint block above is disarmed when the model
                    # changes one incidental value each turn (a fresh username per
                    # signup try) while the move and its no-progress result stay the
                    # same — the real failure that spun ~18 turns. Count consecutive
                    # state-changing turns that made NO progress under the SAME action
                    # STRUCTURE (tool+verb, text dropped) and escalate to the hard
                    # top-of-prompt banner, since the inline stall note above was shown
                    # in the wild to be ignored for many turns.
                    if norm_cmd and no_progress and not loop_detected:
                        struct_fp = self._structural_fp(actions)
                        if struct_fp and struct_fp == self._last_struct_fp:
                            self._no_progress_streak += 1
                        else:
                            self._no_progress_streak = 1
                        self._last_struct_fp = struct_fp
                        if self._no_progress_streak >= 3:
                            stop = self._no_progress_streak >= 5
                            self.stuck_reason = (
                                f"semantic stall: {self._no_progress_streak} consecutive attempts at the "
                                f"same move, all no-progress (varying a detail is not a new approach).")
                            if stop:
                                self._stuck_banner = (
                                    "===[ ! STUCK — READ THIS FIRST ]===\n"
                                    f"{self._no_progress_streak} attempts at this SAME move have all failed the same "
                                    "way. Tweaking one value (another username, another suffix) is NOT a new approach "
                                    "and will keep failing. STOP this sub-task NOW: switch to a genuinely different "
                                    "method/provider/target, or if none exists, report the blocker to the user "
                                    "(say_to_user / task_complete). Do NOT attempt this move again.\n"
                                    "===[ END STUCK ]===")
                            else:
                                self._stuck_banner = (
                                    "===[ ! STUCK — READ THIS FIRST ]===\n"
                                    f"You have made the SAME move {self._no_progress_streak}x with the same no-progress "
                                    "result, changing only an incidental detail each time — that is NOT progress. This "
                                    "turn: use `think` to name the real blocker, then change the METHOD (different "
                                    "target / provider / approach), not just another value.\n"
                                    "===[ END STUCK ]===")
                            self.last_observation += "\n\n" + self._stuck_banner
                            self.print_func(f"{C_RED}\n\n{self._stuck_banner}{C_RESET}")

                    if norm_cmd and not loop_detected and not no_progress:
                        # A CONSEQUENTIAL action ran, it was not a loop, and it actually
                        # changed something -> real progress; clear any prior loop flag,
                        # stuck banner and semantic-stall streak so a recovered agent
                        # stops showing as LOOPING / STUCK. A pure think/read turn (empty
                        # norm_cmd) does NOT clear it — thinking about being stuck is not
                        # getting unstuck; nor does a no-progress turn (handled above).
                        self.stuck_reason = None
                        self._stuck_banner = ""
                        self._no_progress_streak = 0
                        self._last_struct_fp = ""

                # --- IDLE-BABYSITTING DETECTION ---
                # The distinct failure the loop detector should NOT own: the
                # principal spends turn after turn only watching background work
                # (polling/think/say) and doing none of its own. Surfaced as a
                # hard steer here AND in the SUB-AGENTS digest; especially wrong
                # when there is a single lone sub-agent it is merely supervising.
                if self._consecutive_passive_turns >= 2:
                    idle_note = (
                        f"\n\n** IDLE WARNING: your last {self._consecutive_passive_turns} turns did NO real work "
                        f"— only watching/polling background agents or thinking. Supervising is not a substitute "
                        f"for working. THIS turn either advance your OWN orthogonal sub-task (edit/run/write), "
                        f"spawn additional sub-agents for other independent threads, or collect a finished "
                        f"report — do not poll again. **"
                    )
                    self.last_observation += idle_note
                    self.print_func(f"{C_YELLOW}{idle_note}{C_RESET}")

                # Cache the pending iteration state to be finalized next iteration.
                # Derive the ground-truth outcome NOW, from the real tool output —
                # not next turn from the model's `previous_result_summary`, which is
                # the unreliable self-narration this record must not depend on.
                outcome = self._derive_ground_truth_outcome(
                    raw_output, consequential=bool(norm_cmd),
                    loop_detected=loop_detected, repeat_count=repeat_count)
                self.pending_iteration_state = {
                    'iter': iteration,
                    'intent': intent,
                    'actions': actions_taken_str,
                    'outcome': outcome,
                }

                # Persist after each completed iteration so memories/plan/log survive
                # a crash or a clean exit, not just an in-process restart_aeon.
                self._persist_session_state()

                iter_duration = time.time() - iter_start_time
                self.print_func(f"{C_CYAN}Iter {iteration} | {iter_duration:.2f}s | Prompt:{prompt_tokens} ({growth_str}) | Pressure:{pressure}{C_RESET}")

            except Exception as e:
                self.print_func(f"\n{C_RED}CRITICAL ERROR IN ITERATION: {e}{C_RESET}")
                self.logger.error(f"Iteration failed: {e}", exc_info=True)
                if "Context limit exceeded" in str(e):
                    raise
                # Surface the failure to the model so the NEXT turn can adapt
                # instead of re-sending the identical prompt and looping (which
                # silently burns API calls). Formatting/JSON failures get targeted
                # guidance; other errors get a generic recovery note.
                err_str = str(e)
                if "Primary Agent failed" in err_str or "JSON" in err_str:
                    self.last_observation = (
                        "SYSTEM: Your previous response could not be parsed into a valid action plan after "
                        "several attempts. SIMPLIFY your next response: emit ONE small, strictly-valid JSON "
                        "object and nothing else. Multi-line code/text goes INSIDE JSON string values with "
                        "standard escaping (newlines as \\n, quotes as \\\"). Start with a single simple "
                        "action.\n"
                        f"(Underlying error: {err_str[:300]})"
                    )
                else:
                    self.last_observation = (
                        f"SYSTEM: The previous iteration failed with an error and was skipped. "
                        f"Reassess and try a different, simpler next step.\n(Error: {err_str[:300]})"
                    )
                time.sleep(2)

            except KeyboardInterrupt:
                # Two ways in: (1) the user TYPED a message mid-run — the listener
                # stashed it and interrupted us; we already have the text, so fold
                # it straight in. (2) a bare Ctrl+C with nothing typed — pause and
                # prompt for guidance, exactly as before.
                typed = self._take_pending_message()
                try:
                    if typed is not None:
                        self.print_func(f"\n{C_RED}Interrupted — reading your message.{C_RESET}")
                        self.print_func(f"{C_CYAN}Integrating: {typed}{C_RESET}")
                        objective, reset_iter = self._integrate_user_input(objective, typed, iteration)
                        if reset_iter:
                            iteration = 0
                        continue

                    self.print_func(f"\n{C_RED}PAUSED (User Interrupt).{C_RESET}")
                    # Snapshot resumable state NOW: if the user exits (or the
                    # process is then killed), a later run can pick this objective
                    # back up via the resume_previous_session tool.
                    self._write_stop_dump("ctrl-c")
                    user_guidance = self._blocking_read_line(
                        f"{C_YELLOW}Enter guidance, press Enter to resume, or type 'exit' to quit.{C_RESET}\n"
                        f"{C_BLUE}User Guidance > {C_RESET}")

                    if not user_guidance.strip():
                        self.print_func("Resuming...")
                        continue

                    if user_guidance.lower() in ['exit', 'quit']:
                        self.print_func("Aborting task.")
                        break

                    # Integrate the guidance in full context (objective/plan/progress).
                    self.print_func(f"{C_CYAN}Integrating guidance...{C_RESET}")
                    objective, reset_iter = self._integrate_user_input(objective, user_guidance, iteration)
                    if reset_iter:
                        iteration = 0

                except (KeyboardInterrupt, EOFError):
                    self.print_func(f"\n{C_RED}Forced Exit.{C_RESET}")
                    raise KeyboardInterrupt
