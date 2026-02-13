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
    PRIMARY_AGENT_INSTRUCTIONS
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

    def _sync_open_files(self):
        """Synchronize open_files cache with disk state."""
        paths = list(self.open_files.keys())
        for path in paths:
            if not os.path.exists(path):
                del self.open_files[path]
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
        # Normalize to absolute path for consistency
        abs_path = os.path.abspath(path)
        self.open_files[abs_path] = content

    def close_file(self, path: str) -> bool:
        abs_path = os.path.abspath(path)
        if abs_path in self.open_files:
            del self.open_files[abs_path]
            return True
        # Also check original path for backward compatibility
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

    def _format_open_files(self) -> str:
        self._sync_open_files()
        if not self.open_files:
            return "No files currently open."
        out = []
        for path, content in self.open_files.items():
            out.append(f"--- FILE: {path} ---\n{content}\n--- END FILE ---")
        return "\n\n".join(out)

    def _format_history(self) -> str:
        """Format history with tiered detail levels and token budgeting."""
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
        
    def _format_memories(self) -> str:
        if not self.memories:
            return "No memories recorded yet."
        return "\n".join([f"{k}: {v}" for k, v in self.memories.items()])

    def _reset_state(self, initial_observation="Project started."):
        self.current_plan = "Initial state. Need to formulate a plan."
        self.open_files = {}
        self.recent_history.clear()
        self.memories = {}
        self.last_observation = initial_observation

    def _save_objective(self, objective: str):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{timestamp}] OBJECTIVE UPDATE:\n{objective}\n{'-'*40}\n"
            with open(".previous_objective.txt", "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            self.logger.error(f"Failed to save objective to file: {e}")

    def _format_open_file_list(self) -> str:
        self._sync_open_files()
        if not self.open_files:
            return "(none)"
        return "\n".join(f"  - {path}" for path in self.open_files.keys())

    def _build_primary_agent_context(self, tool_list_str: str, system_specs: str, 
                               memories_str: str, objective: str, history_str: str, open_files_str: str) -> str:
        """Build the context for the unified Primary Agent."""
        reminders_section = f"**Important Reminders**\n{self.important_reminders}\n" if self.important_reminders.strip() else ""
        open_file_list = self._format_open_file_list()
        
        return f"""{self.base_directives}

{self.docker_directives}

**Available Tools**
{tool_list_str}

{reminders_section}

**Persistent Memories (Key Details)**
{memories_str}

{system_specs}

**Current Saved Plan**
{self.current_plan}

**Recent History (Last 10 steps)**
{history_str}

**Currently Open Files (already loaded in context)**
{open_file_list}

**Open Files (Full Content)**
{open_files_str}

**Last Observation (Result of previous step)**
{self.last_observation}

**Objective**
{objective}

{PRIMARY_AGENT_INSTRUCTIONS}"""

    def _build_base_context(self, tool_list_str: str) -> str:
        return f"""{self.base_directives}

{self.docker_directives}

**Available Tools**
{tool_list_str}
"""

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
                            self.print_func(f"{C_YELLOW}Agent Request: {params.get('prompt')}\n> {C_RESET}")
                            user_in = input()
                            combined_summary_parts.append(f"User Input: {user_in}")
                        except EOFError:
                            return
                        break # Stop execution chain to process input

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
                        if "COMMAND FAILED" in result_str or result_str.strip().startswith("Error:"):
                             break

                # Summarize results
                if not combined_summary_parts:
                    attempt_summary = "No actions executed."
                else:
                    full_raw_output = "\n\n".join(combined_summary_parts)
                    if len(full_raw_output) < 200 and len(actions) == 1:
                        attempt_summary = full_raw_output
                    else:
                        command_context = f"Chain: {', '.join(actions_taken_str)}"
                        attempt_summary = self.llm_client.summarize_execution(command_context, full_raw_output)

                self.last_observation = attempt_summary
                self.recent_history.append({"iteration": iteration, "action": f"Chain: {len(actions_taken_str)} tools", "summary": attempt_summary})

            except Exception as e:
                self.print_func(f"\n{C_RED}CRITICAL ERROR IN ITERATION: {e}{C_RESET}")
                self.logger.error(f"Iteration failed: {e}", exc_info=True)
                time.sleep(2)

            except KeyboardInterrupt:
                self.print_func(f"\n{C_RED}PAUSED (User Interrupt).{C_RESET}")
                try:
                    self.print_func(f"{C_YELLOW}Interruption Detected. Enter guidance or press Enter to resume.{C_RESET}")
                    user_guidance = input(f"{C_BLUE}User Guidance > {C_RESET}")
                    
                    if not user_guidance.strip():
                        self.print_func("Resuming...")
                        continue
                    
                    if user_guidance.lower() in ['exit', 'quit']:
                        self.print_func("Aborting task.")
                        break

                    self.print_func(f"{C_GREEN}Guidance received.{C_RESET}")
                    self.last_observation = f"USER INTERJECTION: {user_guidance}"
                        
                except (KeyboardInterrupt, EOFError):
                    self.print_func(f"\n{C_RED}Forced Exit.{C_RESET}")
                    break

    def estimate_tokens(self, text):
        return len(text) // 4
