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
from .utils import estimate_tokens
from .prompts import (
    CORE_DIRECTIVES,
    DOCKER_DIRECTIVES,
    IMPORTANT_REMINDERS,
    PRIMARY_AGENT_INSTRUCTIONS,
    TOOLS_SECTION,
    OBJECTIVE_SECTION,
    CONVERSATION_HISTORY
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
        self.memories = {}  # Key-value persistent memory
        self.last_observation = "None."
        self.action_log = deque(maxlen=50)  # Persistent factual record of actions taken

        # Load directives from central prompts module
        self.base_directives = CORE_DIRECTIVES
        self.docker_directives = DOCKER_DIRECTIVES
        self.important_reminders = IMPORTANT_REMINDERS
        self.max_history_tokens = 30000

    def _init_debug_logging(self):
        """Initialize debug logging once per worker instance."""
        if self._debug_initialized:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = Path.home() / f"aeon_debug_{ts}.log"
        self.llm_client.set_debug_path(debug_path)
        self.print_func(f"{C_YELLOW}Debug logging enabled: {debug_path}{C_RESET}")
        self._debug_initialized = True

    def _sync_open_files(self):
        """Synchronize open_files cache with disk state."""
        paths = list(self.open_files.keys())
        for path in paths:
            if not os.path.exists(path):
                del self.open_files[path]
                self.logger.info(f"Removed deleted file from context: {path}")
                continue
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                if self.open_files[path] != content:
                    self.open_files[path] = content
            except Exception:
                pass

    def register_tools(self, tools_list: List[Any]):
        for tool in tools_list:
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
        descs = []
        for name, tool in self.tools.items():
            descs.append(f"- {name}: {tool.description}")
        return "\n\n".join(descs)

    def _format_open_files(self) -> str:
        self._sync_open_files()
        if not self.open_files:
            return "No files currently open."
        out = []
        for path, content in self.open_files.items():
            out.append(f">>> FILE: {path} >>>\n{content}\n<<< END FILE: {path} <<<")
        return "\n\n".join(out)

    def _format_history(self) -> str:
        """Format history with tiered detail levels and token budgeting.
        
        Tiers (by recency from most recent):
        - FULL:    0-6   (last 7 steps)  - complete context
        - BRIEF:   7-11  (next 5 steps)  - action + first 2 sentences
        - MINIMAL: 12-20 (next 9 steps)  - one-line label with pass/fail
        """
        if not self.recent_history:
            return "No recent history."

        items = list(self.recent_history)  # oldest-first from deque
        budget_tokens = self.max_history_tokens
        used_tokens = 0
        formatted = []  # collects entries newest-first

        for idx_from_end, step in enumerate(reversed(items)):
            if idx_from_end > 20:
                break

            iteration = step['iteration']
            action = step['action']
            summary = step.get('summary', '')

            if idx_from_end < 7:
                # FULL tier
                entry = f"STEP {iteration} [FULL]:\nAction: {action}\nResult Summary: {summary}\n"
            elif idx_from_end < 12:
                # BRIEF tier — deterministic char truncation (no LLM interpretation)
                brief = summary[:300]
                if len(summary) > 300:
                    brief += ' [...]'
                entry = f"STEP {iteration} [BRIEF]:\nAction: {action}\nOutput: {brief}\n"
            else:
                # MINIMAL tier
                status = 'FAIL' if any(kw in summary.upper() for kw in ('FAILED', 'ERROR', 'STUCK')) else 'OK'
                entry = f"STEP {iteration}: {action} [{status}]\n"

            entry_tokens = estimate_tokens(entry)
            if used_tokens + entry_tokens > budget_tokens:
                remaining = len(items) - len(formatted)
                formatted.append(f"... [{remaining} older steps omitted due to context budget] ...")
                break
            formatted.append(entry)
            used_tokens += entry_tokens

        # Reverse back to chronological order
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

    def _format_memories(self) -> str:
        if not self.memories:
            return "No memories recorded yet."
        return "\n".join([f"{k}: {v}" for k, v in self.memories.items()])

    def _truncate_output(self, text: str, max_chars: int = 6000) -> str:
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

    def _format_action_log(self) -> str:
        """Format the persistent action log. Purely factual, no interpretation."""
        if not self.action_log:
            return "(No actions taken yet.)"
        lines = []
        for entry in self.action_log:
            status = 'OK' if entry['ok'] else 'FAIL'
            lines.append(f"[iter {entry['iter']}] {entry['tool']}({entry['target']}) [{status}]")
        return "\n".join(lines)

    def _reset_state(self, initial_observation="Project started."):
        self.current_plan = "Initial state. Need to formulate a plan."
        self.open_files = {}
        self.recent_history.clear()
        self.memories = {}
        self.last_observation = initial_observation
        self.action_log.clear()

    def _save_objective(self, objective: str):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{timestamp}] OBJECTIVE UPDATE:\n{objective}\n{'-'*40}\n"
            with open(".previous_objective.txt", "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            self.logger.error(f"Failed to save objective to file: {e}")

    def _build_primary_agent_context(self, tool_list_str: str, system_specs: str,
                               memories_str: str, objective: str, history_str: str, open_files_str: str) -> str:
        """Build the full context prompt for the Primary Agent."""
        reminders_section = f"**IMPORTANT REMINDERS**\n{self.important_reminders}\n\n" if self.important_reminders.strip() else ""

        tools_text = TOOLS_SECTION.format(tools=tool_list_str)
        objective_text = OBJECTIVE_SECTION.format(objective=objective)
        action_log_str = self._format_action_log()

        return f"""{self.base_directives}

{self.docker_directives}

{tools_text}
{reminders_section}**PERSISTENT MEMORIES**
{memories_str}

**ACTION LOG** (factual record of all actions this session — do NOT redo completed work)
{action_log_str}

{system_specs}

**CURRENT PLAN**
{self.current_plan}

{CONVERSATION_HISTORY}
{history_str}

**OPEN FILES**
{'='*60}
(These are loaded in your working memory. Do NOT re-open them.)
{'='*60}
{open_files_str}
{'='*60}
END OPEN FILES
{'='*60}

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
            terminal_tools = ['task_complete']

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

                # Sync open files before building context
                self._sync_open_files()

                # Gather context components
                system_specs = get_runtime_info()
                tool_list_str = self._get_tools_description()
                memories_str = self._format_memories()
                history_str = self._format_history()
                open_files_str = self._format_open_files()

                # Build Primary Agent prompt
                prompt = self._build_primary_agent_context(
                    tool_list_str, system_specs, memories_str, objective, history_str, open_files_str
                )

                self.print_func("Thinking (Primary Agent)...")

                # === PRIMARY AGENT CALL ===
                response_str = self.llm_client.get_primary_agent_response(prompt=prompt)
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
                updated_plan = response_data.get("updated_plan")
                actions = response_data.get("actions", [])

                if updated_plan:
                    if isinstance(updated_plan, list):
                        self.current_plan = "\n".join(updated_plan)
                    else:
                        self.current_plan = str(updated_plan)

                self.print_func(f"\n{C_CYAN}--- THOUGHT ---{C_RESET}")
                self.print_func(f"{thought}")

                if updated_plan:
                    self.print_func(f"\n{C_CYAN}--- UPDATED PLAN ---{C_RESET}")
                    self.print_func(f"{self.current_plan}")

                if not actions:
                    self.print_func(f"{C_RED}No actions returned by agent.{C_RESET}")
                    self.last_observation = "Error: You returned an empty action list. You must take at least one action."
                    continue

                # === EXECUTION PHASE ===
                if step_callback:
                    step_callback(iteration, display_max, "Executing")

                self.print_func(f"\n{C_YELLOW}--- EXECUTION ---{C_RESET}")

                combined_summary_parts = []
                actions_taken_str = []
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
                            self.last_observation = f"User responded to prompt '{params.get('prompt', '')}': {updated_text}"
                            self.print_func(f"{C_GREEN}Input noted, continuing.{C_RESET}")

                        self.recent_history.append({"iteration": iteration, "action": "get_user_input", "summary": self.last_observation})
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

                        # Record to persistent action log (deterministic facts only)
                        is_fail = "COMMAND FAILED" in result_str or result_str.strip().startswith("Error:")
                        target = str(params.get('file_path', params.get('command', params.get('query', ''))))[:80]
                        self.action_log.append({
                            'iter': iteration,
                            'tool': tool_name,
                            'target': target,
                            'ok': not is_fail,
                        })

                        # Stop chain on command failure so the agent can react immediately
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
                self.recent_history.append({"iteration": iteration, "action": f"Chain: {', '.join(actions_taken_str)}", "summary": raw_output})

            except Exception as e:
                self.print_func(f"\n{C_RED}CRITICAL ERROR IN ITERATION: {e}{C_RESET}")
                self.logger.error(f"Iteration failed: {e}", exc_info=True)
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
                    break
