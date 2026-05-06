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

# Colors for terminal output
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[95m'
C_RESET = '\033[0m'
C_BLUE = '\033[96m'


class Worker:
    def __init__(self, llm_client: LLMClient, tools: List[Any] = None, print_func: Callable = print, debug_mode: bool = False):
        self.llm_client = llm_client
        self.tools = {tool.name: tool for tool in tools} if tools else {}
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
        self.pending_iteration_state = None # Holds intent/actions while awaiting result
        self._recent_commands = []  # Rolling window for loop detection
        self._recent_outputs = []   # Corresponding outputs for loop detection
        self.expanded_categories = set()  # Tracks which tool categories are currently expanded
        self.notified_sub_agents = set()  # Tracks which sub-agent terminal states have been alerted to the main agent
        self.instance_id = str(uuid.uuid4())[:8]  # Unique ID for this Aeon run instance
        self.MAX_REPEAT_WINDOW = 5  # How many recent commands to track
        self.REPEAT_THRESHOLD = 2   # How many identical commands before warning
        self.effective_iterations = 0

        # Load directives from central prompts module
        self.base_directives = CORE_DIRECTIVES
        self.docker_directives = DOCKER_DIRECTIVES
        self.important_reminders = IMPORTANT_REMINDERS
        self.max_history_tokens = 30000
        self.current_objective = None
        self.model_name = None  # Set by main.py for restart persistence

    def _init_debug_logging(self):
        """Initialize debug logging once per worker instance."""
        if self._debug_initialized:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_path = Path.home() / f"aeon_debug_{ts}.log"
        self.print_func(f"{C_YELLOW}Debug logging enabled: {self.debug_path}{C_RESET}")
        self._debug_initialized = True

    def _sync_open_files(self):
        """Synchronize open_files cache with disk state."""
        from aeon.tools.analyzers import FileAnalyzer
        paths = list(self.open_files.keys())
        for path in paths:
            if not os.path.exists(path):
                del self.open_files[path]
                self.logger.info(f"Removed deleted file from context: {path}")
                continue
            try:
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

                if len(content) > 250000:
                    content = f"File '{path}' content is too large ({len(content):,} chars) to open directly. Limit is 250,000 chars. Use a script to analyze this file."

                if self.open_files[path] != content:
                    self.open_files[path] = content
            except Exception as e:
                self.logger.error(f"Error syncing file {path}: {e}")

    def register_tools(self, tools_list: List[Any]):
        for tool in tools_list:
            tool.worker = self
            self.tools[tool.name] = tool

    def update_open_file(self, path: str, content: str):
        abs_path = os.path.abspath(path)
        self.open_files[abs_path] = content

    def close_file(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        if abs_path in self.open_files:
            del self.open_files[abs_path]
            return True
        if path in self.open_files:
            del self.open_files[path]
            return True
        return False

    def is_file_open(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        return abs_path in self.open_files or path in self.open_files

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
            is_expanded = path in self.expanded_categories
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

    def _format_open_files(self) -> str:
        self._sync_open_files()
        if not self.open_files:
            return "No files currently open."
        out = []
        for path, content in self.open_files.items():
            out.append(f"--- FILE: {path} ---\n{content}\n--- END FILE: {path} ---")
        return "\n\n".join(out)

    def _format_memories(self) -> str:
        if not self.memories:
            return "No memories recorded yet."
        return "\n".join([f"{k}: {v}" for k, v in self.memories.items()])

    def _truncate_output(self, text: str, max_chars: int = 10000) -> str:
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
        """Format the persistent attempt log (intents + results)."""
        if not self.action_log and not self.pending_iteration_state:
            return "(No actions taken yet.)"
        
        lines = []
        for entry in self.action_log:
            lines.append(entry)
            
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
        self.effective_iterations = 0

    def serialize_state(self) -> dict:
        """Serialize worker state for persistence across restarts."""
        return {
            'memories': dict(self.memories),
            'current_plan': self.current_plan,
            'action_log': list(self.action_log),
            'objective': self.current_objective or '',
            'expanded_categories': list(self.expanded_categories),
            'notified_sub_agents': list(self.notified_sub_agents),
            'instance_id': self.instance_id,
        }

    def restore_state(self, state: dict):
        """Restore worker state from a previous serialization (used after restart)."""
        self.memories = state.get('memories', {})
        self.action_log = state.get('action_log', [])
        self.expanded_categories = set(state.get('expanded_categories', []))
        self.notified_sub_agents = set(state.get('notified_sub_agents', []))
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
                                     memories_str: str, objective: str, open_files_str: str) -> str:
        """Build the full context prompt for the Primary Agent."""
        reminders_section = f"**IMPORTANT REMINDERS**\n{self.important_reminders}\n\n" if self.important_reminders.strip() else ""

        tools_text = TOOLS_SECTION.format(tools=tool_list_str)
        objective_text = OBJECTIVE_SECTION.format(objective=objective)
        attempt_log_str = self._format_attempt_log()

        return f"""{self.base_directives}

{self.docker_directives}

{tools_text}
{reminders_section}
**PERSISTENT MEMORIES**
{memories_str}

**ATTEMPT LOG** (Historical record of intents and results)
{attempt_log_str}

{system_specs}

**CURRENT PLAN**
{self.current_plan}

**OPEN FILES**
===[ IN WORKING MEMORY ]===
{open_files_str}
===[ END OPEN FILES ]===

**LAST STEP RESULT**
{self.last_observation}

{PRIMARY_AGENT_INSTRUCTIONS}

{objective_text}"""

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
                iteration += 1
                self.llm_client.set_iteration(iteration)

                display_max = max_iterations if max_iterations is not None else 999
                if step_callback:
                    step_callback(iteration, display_max, "Thinking")

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

                # Enforce strict 10-iteration limit on the action log
                if len(self.action_log) > 10:
                    self.print_func(f"{C_CYAN}Truncating action log to last 10 iterations to preserve context focus...{C_RESET}")
                    self.action_log = self.action_log[-10:]

                # --- SUB-AGENT NOTIFICATION CHECK ---
                sub_agent_dir = Path(os.getcwd()) / "aeon_output" / self.instance_id / "sub_agents"
                if sub_agent_dir.exists():
                    for agent_dir in sub_agent_dir.iterdir():
                        if agent_dir.is_dir():
                            status_path = agent_dir / "status.txt"
                            if status_path.exists():
                                status = status_path.read_text().strip()
                                if status in ["COMPLETED", "FAILED", "KILLED"]:
                                    state_key = f"{agent_dir.name}_{status}"
                                    if state_key not in self.notified_sub_agents:
                                        notification = f"\n[SYSTEM ALERT] Sub-agent {agent_dir.name} has transitioned to status: {status}. Use get_sub_agent_report to review its findings."
                                        self.last_observation += notification
                                        self.notified_sub_agents.add(state_key)
                                        # Cleanup status file now that agent has been notified
                                        try:
                                            status_path.unlink()
                                        except Exception as e:
                                            self.logger.error(f"Failed to cleanup status file {status_path}: {e}")

                # Sync open files before building context
                self._sync_open_files()

                # Gather context components
                system_specs = get_runtime_info()
                tool_list_str = self._get_tools_description()
                memories_str = self._format_memories()
                open_files_str = self._format_open_files()

                # Build Primary Agent prompt
                prompt = self._build_primary_agent_context(
                    tool_list_str, system_specs, memories_str, objective, open_files_str
                )

                if max_iterations is not None:
                    rem_iters = max_iterations - self.effective_iterations
                    prompt += f"\n\nSYSTEM REMINDER: You have {rem_iters} effective iterations remaining to complete this task. Plan accordingly."
                    if rem_iters <= 0:
                        self.print_func(f"{C_RED}Iteration budget exhausted. Forcing termination.{C_RESET}")
                        self.last_observation = "SYSTEM ALERT: Iteration budget exhausted. You MUST use 'task_complete' to report your final status."

                # Context overflow warning
                prompt_tokens = estimate_tokens(prompt)
                
                # --- Context Diagnostic Breakdown ---
                breakdown = [
                    f"Directives: ~{estimate_tokens(self.base_directives + self.docker_directives + PRIMARY_AGENT_INSTRUCTIONS)} tokens",
                    f"Tools: ~{estimate_tokens(tool_list_str)} tokens",
                    f"Memories: ~{estimate_tokens(memories_str)} tokens",
                    f"Attempt Log: ~{estimate_tokens(self._format_attempt_log())} tokens",
                    f"State & Plan: ~{estimate_tokens(system_specs + self.current_plan + self.last_observation)} tokens",
                    f"Open Files Total: ~{estimate_tokens(open_files_str)} tokens",
                ]
                if self.open_files:
                    for path, content in self.open_files.items():
                        breakdown.append(f"  - {os.path.basename(path)}: ~{estimate_tokens(content)} tokens")
                diagnostic_str = "\n".join(breakdown)
                # ------------------------------------

                ctx_limit = self.llm_client.context_limit
                if prompt_tokens > ctx_limit * 0.85:
                    pct = prompt_tokens / ctx_limit * 100
                    self.print_func(f"{C_RED}WARNING: Prompt is ~{prompt_tokens} tokens ({pct:.0f}% of {ctx_limit} context limit). Close files or context will be truncated!{C_RESET}")
                    if prompt_tokens > ctx_limit * 0.95:
                        raise RuntimeError(f"Context limit exceeded ({prompt_tokens} > {ctx_limit * 0.95} limit). Throwing error as requested.")

                self.print_func("Thinking (Primary Agent)...")

                # === PRIMARY AGENT CALL ===
                response_str = self.llm_client.get_primary_agent_response(prompt=prompt, diagnostic_str=diagnostic_str)
                if self.debug_mode:
                    self.print_func(f"{C_YELLOW}[DEBUG] Primary Agent Raw Output:\n{response_str}{C_RESET}")

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

                if len(actions) > 15:
                    actions = actions[:15]
                    self.logger.warning("Truncated actions to 15")

                for idx, action_data in enumerate(actions):
                    tool_name = action_data.get("tool_name")
                    params = action_data.get("parameters", {})

                    if not tool_name:
                        combined_summary_parts.append(f"Action {idx+1}: Missing tool_name.")
                        continue

                    if tool_name not in self.tools:
                        combined_summary_parts.append(f"Action {idx+1}: Tool '{tool_name}' not found.")
                        continue

                    self.print_func(f"{C_YELLOW}Executing (Step {idx+1}):{C_RESET} {tool_name} {params}")
                    
                    full_action_desc = f"{tool_name}({params})" if params else f"{tool_name}()"
                    display_action_desc = f"{tool_name}({str(params)[:40]}...)" if params else f"{tool_name}()"
                    actions_taken_str.append(display_action_desc)
                    full_actions_taken_str.append(full_action_desc)

                    if tool_name in terminal_tools:
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
                        combined_summary_parts.append(f"Action {idx+1} ({tool_name}):\n{result_str}")

                        # Stop chain on command failure so the agent can react immediately
                        is_fail = "COMMAND FAILED" in result_str or result_str.strip().startswith("Error:")
                        if is_fail:
                            break

                # Skip summarization if user input was already handled directly
                if user_input_handled:
                    continue

                # Deterministic truncation — raw output preserved, no LLM interpretation
                if not combined_summary_parts:
                    raw_output = "No actions produced output."
                else:
                    raw_output = "\n\n".join(combined_summary_parts)
                    raw_output = self._truncate_output(raw_output)

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
                if len(self._recent_commands) >= self.REPEAT_THRESHOLD:
                    recent_pairs = list(zip(self._recent_commands[-self.REPEAT_THRESHOLD:], self._recent_outputs[-self.REPEAT_THRESHOLD:]))
                    if len(set(recent_pairs)) == 1:
                        repeat_count = self.REPEAT_THRESHOLD
                        # Count actual streak length
                        for i in range(len(self._recent_commands) - 1, -1, -1):
                            if (self._recent_commands[i], self._recent_outputs[i]) == recent_pairs[0]:
                                repeat_count = len(self._recent_commands) - i
                            else:
                                break
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
                
                # Cache the pending iteration state to be finalized next iteration
                self.pending_iteration_state = {
                    'iter': iteration,
                    'intent': intent,
                    'actions': actions_taken_str
                }

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
