import json
import re
import time
import sys
import os
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import List, Any, Dict, Callable, Optional

from .llm import LLMClient
from .system_info import get_runtime_info
from .logger import get_logger
from .prompts import (
    CORE_DIRECTIVES,
    DOCKER_DIRECTIVES,
    IMPORTANT_REMINDERS,
    PLANNER_INSTRUCTIONS,
    EXECUTOR_INSTRUCTIONS,
    PREFLIGHT_INSTRUCTIONS,
    MILESTONE_ANALYZER_INSTRUCTIONS,
)

# Colors for terminal output
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_RESET = '\033[0m'
C_BLUE = '\033[94m'


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
        self.recent_history = deque(maxlen=50) 
        self.completed_milestones = []  # Foundational progress markers, append-only
        self.last_observation = "None."
        
        # Load directives from central prompts module
        self.base_directives = CORE_DIRECTIVES
        self.docker_directives = DOCKER_DIRECTIVES
        self.important_reminders = IMPORTANT_REMINDERS
        self.max_history_tokens = 25000

    def _init_debug_logging(self):
        """Initialize debug logging once per worker instance."""
        if self._debug_initialized:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = Path.home() / f"aeon_debug_{ts}.log"
        self.llm_client.set_debug_path(debug_path)
        self.print_func(f"{C_YELLOW}Debug logging enabled: {debug_path}{C_RESET}")
        self._debug_initialized = True

    def register_tools(self, tools_list: List[Any]):
        for tool in tools_list:
            self.tools[tool.name] = tool

    def update_open_file(self, path: str, content: str):
        # Normalize to absolute path for consistency
        abs_path = os.path.abspath(path)
        self.open_files[abs_path] = content

    def close_file(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        if abs_path in self.open_files:
            del self.open_files[abs_path]
            return True
        # Also check original path for backwards compatibility
        if path in self.open_files:
            del self.open_files[path]
            return True
        return False

    def is_file_open(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        return abs_path in self.open_files or path in self.open_files

    def _get_tools_description(self) -> str:
        descs = []
        for name, tool in self.tools.items():
            descs.append(f"- {name}: {tool.description}")
        return "\n".join(descs)

    def _get_preflight_tools_description(self) -> str:
        """Get tool descriptions for only file management tools (pre-flight phase)."""
        preflight_tools = ['open_file', 'close_file']
        descs = []
        for name in preflight_tools:
            if name in self.tools:
                tool = self.tools[name]
                descs.append(f"- {name}: {tool.description}")
        return "\n".join(descs)

    def _format_open_files(self) -> str:
        if not self.open_files:
            return "No files currently open."
        out = []
        for path, content in self.open_files.items():
            out.append(f"--- FILE: {path} ---\n{content}\n--- END FILE ---")
        return "\n\n".join(out)

    def _format_open_files_list(self) -> str:
        """Return just a list of currently open file paths (no content).
        Used for pre-flight executor context."""
        if not self.open_files:
            return "No files currently open."
        paths = list(self.open_files.keys())
        return "\n".join([f"  - {path}" for path in paths])

    def _format_open_files_compact(self) -> str:
        """Compact file manifest for the planner: names, sizes, and a brief peek.
        The planner needs to know WHAT is open, not read every line."""
        if not self.open_files:
            return "No files currently open."
        out = []
        for path, content in self.open_files.items():
            lines = content.splitlines()
            line_count = len(lines)
            head = lines[:8]
            tail = lines[-4:] if line_count > 12 else []
            peek = '\n'.join(head)
            if tail:
                peek += f'\n  ... ({line_count - 12} lines omitted) ...\n' + '\n'.join(tail)
            out.append(f"--- FILE: {path} ({line_count} lines) ---\n{peek}\n--- END ---")
        return "\n\n".join(out)

    def _format_open_files_for_executor(self, suggested_actions: str) -> str:
        """Show full content only for files referenced in the suggested actions.
        Other open files get a one-line stub so the executor knows they exist
        but isn't distracted by their content."""
        if not self.open_files:
            return "No files currently open."
        # Build a lowercase search corpus from the suggested actions
        actions_lower = suggested_actions.lower()
        out_relevant = []
        out_background = []
        for path, content in self.open_files.items():
            basename = os.path.basename(path).lower()
            # Match on basename or full path appearing in the actions text
            if basename in actions_lower or path.lower() in actions_lower:
                out_relevant.append(f"--- FILE: {path} ---\n{content}\n--- END FILE ---")
            else:
                line_count = content.count('\n') + 1
                out_background.append(f"  {path} ({line_count} lines) — not referenced in current actions, use open_file to view")
        parts = []
        if out_relevant:
            parts.append("\n\n".join(out_relevant))
        if out_background:
            parts.append("Other open files (content hidden):\n" + "\n".join(out_background))
        # Fallback: if nothing matched, show everything (safe default)
        if not out_relevant and self.open_files:
            return self._format_open_files()
        return "\n\n".join(parts)

    def _format_history(self) -> str:
        """Format history with tiered detail levels and token budgeting.

        Tiers (counting from most recent):
          - FULL   (last 3):  Complete action + full summary
          - BRIEF  (next 7):  Action + first 2 sentences of summary
          - MINIMAL (older):  One-line action label with OK/FAIL tag

        Fills newest-first until max_history_tokens budget is exhausted.
        Zero extra LLM calls - purely algorithmic compression.
        """
        if not self.recent_history:
            return "No recent history."

        items = list(self.recent_history)  # oldest-first from deque
        total = len(items)
        budget_chars = self.max_history_tokens * 4  # rough chars-per-token
        used_chars = 0
        formatted = []  # collects entries newest-first

        for idx_from_end, step in enumerate(reversed(items)):
            iteration = step['iteration']
            action = step['action']
            summary = step.get('summary', '')

            if idx_from_end < 3:
                # FULL tier - complete context for most recent work
                entry = f"STEP {iteration} [FULL]:\nAction: {action}\nResult Summary: {summary}\n"
            elif idx_from_end < 10:
                # BRIEF tier - action + first 2 sentences
                brief = self._first_n_sentences(summary, 2)
                entry = f"STEP {iteration} [BRIEF]:\nAction: {action}\nResult: {brief}\n"
            else:
                # MINIMAL tier - one-line label with pass/fail
                status = 'FAIL' if any(kw in summary.upper() for kw in ('FAILED', 'ERROR', 'STUCK')) else 'OK'
                entry = f"STEP {iteration}: {action} [{status}]\n"

            entry_chars = len(entry)
            if used_chars + entry_chars > budget_chars:
                remaining = total - len(formatted)
                formatted.append(f"... [{remaining} older steps omitted due to context budget] ...")
                break
            formatted.append(entry)
            used_chars += entry_chars

        # Reverse back to chronological order for readability
        formatted.reverse()
        return "\n".join(formatted)

    @staticmethod
    def _first_n_sentences(text: str, n: int) -> str:
        """Extract roughly the first n sentences from text."""
        if not text:
            return ''
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        result = ' '.join(sentences[:n])
        if len(sentences) > n:
            result += ' [...]'
        return result
        
    def _format_milestones(self) -> str:
        if not self.completed_milestones:
            return "No milestones completed yet."
        return "\n".join([f"[x] {m}" for m in self.completed_milestones])

    def _reset_state(self, initial_observation="Project started."):
        self.current_plan = "Initial state. Need to formulate a plan."
        self.open_files = {}
        self.recent_history.clear()
        self.completed_milestones = []
        self.last_observation = initial_observation

    def _save_objective(self, objective: str):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{timestamp}] OBJECTIVE UPDATE:\n{objective}\n{'-'*40}\n"
            with open(".previous_objective.txt", "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            self.logger.error(f"Failed to save objective to file: {e}")

    def _build_planner_context(self, tool_list_str: str, system_specs: str, 
                               milestones_str: str, objective: str, history_str: str, open_files_str: str) -> str:
        """Build the complete planner prompt with instructions at the end."""
        reminders_section = f"**Important Reminders**\n{self.important_reminders}\n" if self.important_reminders.strip() else ""
        
        return f"""{self.base_directives}

{self.docker_directives}

**Available Tools**
{tool_list_str}

{reminders_section}

{system_specs}

**Completed Milestones (Foundational Progress)**
{milestones_str}

**Open Files (Working Memory)**
{open_files_str}

**Objective**
{objective}

**Current Saved Plan**
{self.current_plan}

**Recent History (Last 10 steps)**
{history_str}

**Last Observation (From previous step)**
{self.last_observation}

{PLANNER_INSTRUCTIONS}"""

    def _build_preflight_executor_context(self, tool_list_str: str, system_specs: str,
                                          suggested_actions: str, open_files_list: str) -> str:
        """Build the pre-flight executor prompt for context gathering phase."""
        reminders_section = f"**Important Reminders**\n{self.important_reminders}\n" if self.important_reminders.strip() else ""

        return f"""{self.base_directives}

{self.docker_directives}

**Available Tools (Pre-flight Phase - File Management Only)**
{tool_list_str}

{reminders_section}

{system_specs}

**Currently Open Files (Paths Only)**
{open_files_list}

**Planner's Suggested Actions**
{suggested_actions}

{PREFLIGHT_INSTRUCTIONS}"""

    def _build_executor_context(self, tool_list_str: str, milestones_str: str,
                                objective: str,
                                suggested_actions: str, open_files_str: str) -> str:
        """Build the complete executor prompt with instructions at the end.
        Note: The full plan is intentionally omitted. The executor receives the
        distilled suggested_actions from the planner which contains everything
        it needs. Sending the full plan wastes context and confuses weaker models."""
        reminders_section = f"**Important Reminders**\n{self.important_reminders}\n" if self.important_reminders.strip() else ""

        return f"""{self.base_directives}

{self.docker_directives}

**Available Tools**
{tool_list_str}

{reminders_section}

**Completed Milestones (Foundational Progress)**
{milestones_str}

**Objective**
{objective}

**Your Task (from Planner)**
{suggested_actions}

**Open Files (Working Memory)**
{open_files_str}

{EXECUTOR_INSTRUCTIONS}"""

    def _build_base_context(self, tool_list_str: str) -> str:
        """Build base context without role-specific instructions (used for milestone analyzer)."""
        return f"""{self.base_directives}

{self.docker_directives}

**Available Tools**
{tool_list_str}
"""

    def _analyze_milestones(self, objective: str, iteration: int, 
                            actions_taken: List[str], iteration_result: str) -> None:
        """Analyze the iteration results to identify any completed milestones.
        Milestones are foundational progress markers that won't need to be redone.
        Uses a minimal prompt - the analyzer only needs the objective, existing
        milestones, and this iteration's results. No tools/directives/history needed.
        """
        try:
            milestones_str = self._format_milestones()
            
            analysis_context = f"""{MILESTONE_ANALYZER_INSTRUCTIONS}

**Objective**
{objective}

**Already Completed Milestones**
{milestones_str}

**This Iteration (#{iteration})**
Actions Taken: {', '.join(actions_taken)}
Result:
{iteration_result}
"""
            
            response = self.llm_client.analyze_milestones(analysis_context)
            
            if response and isinstance(response, dict):
                new_milestones = response.get("milestones_achieved", [])
                if new_milestones and isinstance(new_milestones, list):
                    for milestone in new_milestones:
                        if milestone and isinstance(milestone, str) and milestone.strip() and milestone not in self.completed_milestones:
                            self.completed_milestones.append(milestone.strip())
                            self.print_func(f"{C_GREEN}>> MILESTONE ACHIEVED: {milestone}{C_RESET}")
                            
        except Exception as e:
            self.logger.warning(f"Milestone analysis failed: {e}")
            # Non-fatal - continue without milestone update

    def _clean_action_json(self, raw_str: str) -> str:
        """Clean and extract JSON from potentially markdown-wrapped LLM response."""
        clean_json = raw_str.strip()
        
        # Handle ```json with optional whitespace/newline
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:].lstrip()
        elif clean_json.startswith("```"):
            clean_json = clean_json[3:].lstrip()
        
        # Handle closing fence with optional whitespace
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3].rstrip()
        
        return clean_json.strip()

    # --- MAIN LOOP ---
    def run(self, objective: str, max_iterations: Optional[int] = None, step_callback: Optional[Callable[[int, int, str], None]] = None, terminal_tools: List[str] = None):
        if terminal_tools is None:
            terminal_tools = ['task_complete']
            
        self.logger.info("Starting Execution for: %s", objective)
        self._save_objective(objective)
        
        # Debug logging is now initialized in __init__, not here
        # This prevents multiple debug files per worker instance

        iteration = 0
        # Prime the observation with the objective to ensure the Planner registers it immediately
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
                        msg = f"SYSTEM ALERT: Max iterations ({max_iterations}) reached. You have ONE final step. You MUST use a terminal tool ({', '.join(terminal_tools)}) NOW to report your findings (even if incomplete)."
                        self.last_observation = msg
                        self.print_func(f"{C_RED}Max iterations reached. Forcing final report.{C_RESET}")
                    else:
                        self.print_func(f"{C_RED}Agent failed to exit. Terminating.{C_RESET}")
                        break

                self.print_func(f"{C_BLUE}--- Iteration {iteration} ---{C_RESET}")
                
                # Gather context components
                system_specs = get_runtime_info()
                tool_list_str = self._get_tools_description()
                milestones_str = self._format_milestones()
                history_str = self._format_history()
                open_files_str = self._format_open_files()
                
                # Build planner prompt (instructions at end for emphasis)
                # Planner gets compact file manifest; executor gets full content
                compact_files_str = self._format_open_files_compact()
                planner_prompt = self._build_planner_context(
                    tool_list_str, system_specs, milestones_str, objective, history_str, compact_files_str
                )

                # --- PLANNER ---
                self.print_func("Thinking (Planning)...")
                
                try:
                    plan_response_str = self.llm_client.get_plan(prompt=planner_prompt)
                    if self.debug_mode:
                        self.print_func(f"{C_YELLOW}[DEBUG] Planner Raw Output:\n{plan_response_str}{C_RESET}")
                    
                    suggested_actions_str = "No specific actions suggested."
                    
                    # PARSE PLANNER RESPONSE
                    plan_data = json.loads(plan_response_str)
                    self.current_plan = plan_data.get("updated_plan") or self.current_plan
                except Exception as e:
                    self.print_func(f"{C_RED}PLANNER CRASHED: {e}{C_RESET}")
                    plan_data = {}
                
                # Format suggested actions for executor (now free-form text from planner)
                next_actions = plan_data.get("next_actions") or ""
                if next_actions:
                    if isinstance(next_actions, str):
                        suggested_actions_str = next_actions
                    elif isinstance(next_actions, list):
                        # Backwards compatibility if planner still outputs list
                        action_lines = []
                        for act in next_actions:
                            if isinstance(act, dict):
                                action_lines.append(f"- {act.get('tool', 'unknown')}: {act.get('purpose', 'N/A')}")
                            else:
                                action_lines.append(f"- {act}")
                        suggested_actions_str = "\n".join(action_lines)
                    else:
                        suggested_actions_str = str(next_actions)
                
                analysis = plan_data.get("analysis", "")
                iteration_strategy = plan_data.get("iteration_strategy", "single_step")
                risk_notes = plan_data.get("risk_notes", "")

                self.print_func(f"{C_CYAN}Analysis:{C_RESET} {analysis}")
                self.print_func(f"{C_CYAN}Strategy:{C_RESET} {iteration_strategy}")
                if risk_notes:
                    self.print_func(f"{C_YELLOW}Risks:{C_RESET} {risk_notes}")
                
                plan_lines = self.current_plan.split('\n')
                if len(plan_lines) > 8:
                    preview = '\n'.join(plan_lines[:4]) + f'\n{C_YELLOW}... [plan truncated] ...{C_RESET}\n' + '\n'.join(plan_lines[-4:])
                else:
                    preview = self.current_plan
                self.print_func(f"{C_CYAN}Plan Update:{C_RESET}\n{preview}")
                    

                # --- PRE-FLIGHT EXECUTOR (Context Gathering Phase) ---
                if step_callback:
                    step_callback(iteration, display_max, "Pre-flight")

                self.print_func("Pre-flight: Gathering file context...")
                
                preflight_tool_list = self._get_preflight_tools_description()
                open_files_list = self._format_open_files_list()
                preflight_prompt = self._build_preflight_executor_context(
                    preflight_tool_list, system_specs,
                    suggested_actions_str, open_files_list
                )

                try:
                    preflight_json = self.llm_client.get_action(prompt=preflight_prompt)
                    preflight_data = json.loads(self._clean_action_json(preflight_json))
                    preflight_actions = preflight_data.get("actions", [])
                    
                    if not preflight_actions:
                        self.print_func("Pre-flight: File context already optimal.")
                    else:
                        # Execute pre-flight actions (simple - just open/close files)
                        for action in preflight_actions:
                            tool_name = action.get("tool_name")
                            params = action.get("parameters", {})
                            
                            if tool_name not in ["open_file", "close_file"]:
                                self.logger.warning(f"Pre-flight: Ignoring invalid tool '{tool_name}'")
                                continue
                                
                            if tool_name not in self.tools:
                                self.logger.warning(f"Pre-flight: Tool '{tool_name}' not found")
                                continue
                            
                            try:
                                result = self.tools[tool_name].execute(**params)
                                self.print_func(f"Pre-flight: {tool_name} {params.get('file_path', '?')}")
                            except Exception as e:
                                self.logger.warning(f"Pre-flight {tool_name} failed: {e}")
                                # Continue - main executor can retry if needed
                except Exception as e:
                    self.logger.warning(f"Pre-flight phase failed: {e}. Proceeding to main execution.")
                    # Continue anyway - main executor has open_file if needed

                # --- MAIN EXECUTOR (with retry loop) ---
                if step_callback:
                    step_callback(iteration, display_max, "Executing")

                # Now format files with full content for main executor
                executor_files_str = self._format_open_files()
                executor_prompt = self._build_executor_context(
                    tool_list_str, milestones_str,
                    objective,
                    suggested_actions_str, executor_files_str
                )

                max_exec_retries = 3
                last_fail_step = -1
                stuck_count = 0
                exec_error_feedback = ""
                final_summary = "No actions executed."
                final_actions_taken = []

                for exec_attempt in range(max_exec_retries + 1):
                    # On retries, augment prompt with error feedback and refreshed file state
                    if exec_attempt > 0:
                        self.print_func(f"{C_YELLOW}Executor self-correcting (retry {exec_attempt}/{max_exec_retries})...{C_RESET}")
                        refreshed_files = self._format_open_files()
                        retry_addendum = (
                            f"\n\n**EXECUTION ERROR FEEDBACK (Retry {exec_attempt}/{max_exec_retries})**\n"
                            f"Your previous set of actions failed during execution. Here is what happened:\n"
                            f"{exec_error_feedback}\n\n"
                            f"**Updated Open Files (may reflect changes from successful steps)**\n"
                            f"{refreshed_files}\n\n"
                            f"Revise your actions to fix the error. Do NOT repeat the exact same failing action unchanged."
                        )
                        current_prompt = executor_prompt + retry_addendum
                    else:
                        current_prompt = executor_prompt

                    action_json_str = self.llm_client.get_action(prompt=current_prompt)
                    if self.debug_mode:
                        self.print_func(f"{C_YELLOW}[DEBUG] Executor Raw Output:\n{action_json_str}{C_RESET}")

                    # --- Parse actions ---
                    actions = []
                    parse_failed = False
                    try:
                        clean_json = self._clean_action_json(action_json_str)
                        parsed = json.loads(clean_json)
                        if isinstance(parsed, list):
                            actions = parsed
                        elif isinstance(parsed, dict):
                            if "actions" in parsed and isinstance(parsed["actions"], list):
                                actions = parsed["actions"]
                            elif "tool_name" in parsed:
                                actions = [parsed]
                    except json.JSONDecodeError as e:
                        exec_error_feedback = f"Your response was not valid JSON. Parse error: {e}. Raw start: {action_json_str[:300]}..."
                        self.print_func(f"{C_RED}JSON Parse Failed: {e}{C_RESET}")
                        parse_failed = True
                    except Exception as e:
                        exec_error_feedback = f"Unexpected parse error: {e}"
                        self.print_func(f"{C_RED}Action Parse Failed: {e}{C_RESET}")
                        parse_failed = True

                    if parse_failed:
                        if exec_attempt == max_exec_retries:
                            final_summary = f"Failed to parse executor response after {max_exec_retries + 1} attempts. Last error: {exec_error_feedback}"
                        continue

                    if not actions:
                        exec_error_feedback = "No actions were parsed from your response. You MUST return at least one action."
                        if exec_attempt == max_exec_retries:
                            final_summary = "No actions parsed from executor after all retries."
                        continue

                    # --- Execute actions ---
                    combined_summary_parts = []
                    actions_taken_str = []
                    error_at_step = -1
                    hit_early_exit = False

                    if len(actions) > 15:
                        actions = actions[:15]
                        self.logger.warning("Truncated actions to 15")

                    for idx, action_data in enumerate(actions):
                        tool_name = action_data.get("tool_name")
                        params = action_data.get("parameters", {})
                        allow_failure = action_data.get("allow_failure", False)

                        if not tool_name:
                            combined_summary_parts.append(f"Action {idx+1}: Missing tool_name in action data.")
                            error_at_step = idx
                            break

                        if tool_name not in self.tools:
                            combined_summary_parts.append(f"Action {idx+1}: Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")
                            error_at_step = idx
                            break

                        self.print_func(f"{C_YELLOW}Executing (Step {idx+1}):{C_RESET} {tool_name} {params}")
                        actions_taken_str.append(tool_name)

                        if tool_name in terminal_tools:
                            try:
                                tool = self.tools[tool_name]
                                result_str = str(tool.execute(**params))
                            except Exception as e:
                                result_str = f"Error executing terminal tool {tool_name}: {e}"
                            self.print_func(f"\n{C_GREEN}{result_str}{C_RESET}")
                            self.recent_history.append({"iteration": iteration, "action": tool_name, "summary": result_str})
                            if step_callback: step_callback(iteration, display_max, "Complete")
                            return

                        elif tool_name == "get_user_input":
                            try:
                                self.print_func(f"{C_YELLOW}Agent Request: {params.get('prompt')}\n> {C_RESET}")
                                user_in = input()
                                combined_summary_parts.append(f"User Input: {user_in}")
                            except EOFError:
                                return
                            hit_early_exit = True
                            break

                        else:
                            try:
                                tool = self.tools[tool_name]
                                raw_result = tool.execute(**params)
                            except TypeError as e:
                                raw_result = f"Tool Parameter Error: {e}. Check required parameters for '{tool_name}'."
                            except Exception as e:
                                raw_result = f"Tool Execution Error: {type(e).__name__}: {e}"

                            result_str = str(raw_result)
                            combined_summary_parts.append(f"Action {idx+1} ({tool_name}):\n{result_str}")

                            if "COMMAND FAILED" in result_str or result_str.strip().startswith("Error:"):
                                if not allow_failure:
                                    error_at_step = idx
                                    break

                    # --- Summarize this attempt ---
                    if not combined_summary_parts:
                        attempt_summary = "No actions executed."
                    else:
                        full_raw_output = "\n\n".join(combined_summary_parts)
                        if len(full_raw_output) < 200 and len(actions) == 1 and actions[0].get("tool_name") != "run_command":
                            attempt_summary = full_raw_output
                        else:
                            command_context = f"Chain: {', '.join(actions_taken_str)}"
                            attempt_summary = self.llm_client.summarize_execution(command_context, full_raw_output)

                    # --- Decide: success, retry, or escalate ---
                    if error_at_step == -1 or hit_early_exit:
                        # All actions succeeded (or user input was requested)
                        final_summary = attempt_summary
                        final_actions_taken = actions_taken_str
                        break

                    # Error occurred — check if we're making forward progress
                    if error_at_step <= last_fail_step:
                        stuck_count += 1
                    else:
                        # Failing later in the chain means earlier steps got fixed
                        stuck_count = 0
                    last_fail_step = error_at_step

                    # Stuck on same (or earlier) step too many times -> escalate to planner
                    if stuck_count >= 2:
                        self.print_func(f"{C_RED}Executor stuck at step {error_at_step + 1} after {exec_attempt + 1} attempts. Escalating to planner.{C_RESET}")
                        final_summary = (
                            f"EXECUTOR STUCK at step {error_at_step + 1} across {exec_attempt + 1} retries. "
                            f"Needs planner intervention.\n{attempt_summary}"
                        )
                        final_actions_taken = actions_taken_str
                        break

                    # Still have retries left — build detailed feedback for the executor
                    if exec_attempt < max_exec_retries:
                        success_lines = [f"  Step {i+1} ({actions_taken_str[i]}): OK" for i in range(error_at_step)]
                        fail_tool = actions_taken_str[error_at_step] if error_at_step < len(actions_taken_str) else "unknown"
                        fail_line = f"  Step {error_at_step + 1} ({fail_tool}): FAILED"
                        exec_error_feedback = (
                            f"Executed {len(actions_taken_str)} action(s). Failed at step {error_at_step + 1}.\n"
                            + ("\n".join(success_lines) + "\n" if success_lines else "")
                            + fail_line + "\n"
                            + f"Error details:\n{attempt_summary}"
                        )
                        self.print_func(
                            f"{C_YELLOW}Error at step {error_at_step + 1}. "
                            f"Retrying executor ({exec_attempt + 1}/{max_exec_retries})...{C_RESET}"
                        )
                    else:
                        # Out of retries entirely
                        final_summary = (
                            f"EXECUTOR FAILED after {max_exec_retries + 1} attempts. "
                            f"Last error at step {error_at_step + 1}.\n{attempt_summary}"
                        )
                        final_actions_taken = actions_taken_str

                self.last_observation = final_summary
                self.recent_history.append({"iteration": iteration, "action": f"Chain: {len(final_actions_taken)} tools", "summary": final_summary})

                # --- MILESTONE ANALYSIS (after iteration completes) ---
                # Analyze if any foundational milestones were achieved this iteration
                self._analyze_milestones(
                    objective=objective,
                    iteration=iteration,
                    actions_taken=final_actions_taken,
                    iteration_result=final_summary,
                )

            except KeyboardInterrupt:
                self.print_func(f"\n{C_RED}PAUSED (User Interrupt).{C_RESET}")
                try:
                    self.print_func(f"{C_YELLOW}Interruption Detected. Enter guidance to steer the agent, or press Enter to resume.{C_RESET}")
                    self.print_func(f"{C_YELLOW}(Type 'exit' or press Ctrl+C again to abort task){C_RESET}")
                    user_guidance = input(f"{C_BLUE}User Guidance > {C_RESET}")
                    
                    if not user_guidance.strip():
                        self.print_func("Resuming...")
                        continue
                        
                    if user_guidance.lower() in ['exit', 'quit']:
                        self.print_func("Aborting task.")
                        break

                    self.print_func("Analyzing interruption...")
                    analysis = self.llm_client.analyze_interruption(objective, user_guidance)
                    
                    intent = analysis.get("classification", "ADVICE")
                    updated_text = analysis.get("updated_text", user_guidance)
                    reasoning = analysis.get("reasoning", "")
                    
                    self.print_func(f"Interpretation: {intent} ({reasoning})")
                    
                    if intent == "NEW_TASK":
                        self.print_func(f"{C_GREEN}Switching to NEW TASK: {updated_text}{C_RESET}")
                        objective = updated_text
                        self._reset_state(initial_observation=f"Task switched by user: {user_guidance}")
                        self._save_objective(objective)
                    elif intent == "MODIFY_OBJECTIVE":
                        self.print_func(f"{C_GREEN}Updating OBJECTIVE: {updated_text}{C_RESET}")
                        objective = updated_text
                        self.last_observation = f"USER INTERRUPTION: {user_guidance} (Objective Updated)"
                        self._save_objective(objective)
                    else:
                        self.print_func(f"{C_GREEN}Advice received.{C_RESET}")
                        self.last_observation = f"USER ADVICE: {user_guidance}"
                        
                except (KeyboardInterrupt, EOFError):
                    self.print_func(f"\n{C_RED}Forced Exit.{C_RESET}")
                    break

    def estimate_tokens(self, text):
        return len(text) // 4
