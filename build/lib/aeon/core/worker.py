import json
import re
import time
import sys
from datetime import datetime
from collections import deque
from importlib import resources
from typing import List, Any, Dict, Callable, Optional

from .llm import LLMClient
from .system_info import get_runtime_info
from .logger import get_logger

# Colors for terminal output
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_RESET = '\033[0m'
C_BLUE = '\033[94m'

class Worker:
    def __init__(self, llm_client: LLMClient, tools: List[Any] = None, print_func: Callable = print):
        self.llm_client = llm_client
        self.tools = {tool.name: tool for tool in tools} if tools else {}
        self.logger = get_logger()
        self.print_func = print_func
        
        # --- STATE MODEL ---
        self.current_plan = "No plan formulated yet."
        self.open_files = {} 
        self.recent_history = deque(maxlen=10) 
        self.last_observation = "None."
        
        self.base_directives = ""
        self.max_history_tokens = 25000
        self._load_directives()

    def register_tools(self, tools_list: List[Any]):
        for tool in tools_list:
            self.tools[tool.name] = tool

    def update_open_file(self, path: str, content: str):
        self.open_files[path] = content

    def close_file(self, path: str) -> bool:
        if path in self.open_files:
            del self.open_files[path]
            return True
        return False

    def is_file_open(self, path: str) -> bool:
        return path in self.open_files

    def _load_directives(self):
        try:
            txt = resources.files('aeon.core.prompts') / 'core_directives.txt'
            self.base_directives = txt.read_text(encoding='utf-8')
        except Exception:
            self.base_directives = "Be helpful, efficient, and precise."

    def _get_tools_description(self) -> str:
        descs = []
        for name, tool in self.tools.items():
            descs.append(f"- {name}: {tool.description}")
        return "\n".join(descs)

    def _format_open_files(self) -> str:
        if not self.open_files:
            return "No files currently open."
        out = []
        for path, content in self.open_files.items():
            out.append(f"--- FILE: {path} ---\n{content}\n--- END FILE ---")
        return "\n\n".join(out)

    def _format_history(self) -> str:
        if not self.recent_history:
            return "No recent history."
        out = []
        for step in self.recent_history:
            out.append(f"STEP {step['iteration']}:\nAction: {step['action']}\nResult Summary: {step['summary']}\n")
        return "\n".join(out)

    def _reset_state(self, initial_observation="Project started."):
        self.current_plan = "Initial state. Need to formulate a plan."
        self.open_files = {}
        self.recent_history.clear()
        self.last_observation = initial_observation

    def _save_objective(self, objective: str):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{timestamp}] OBJECTIVE UPDATE:\n{objective}\n{'-'*40}\n"
            with open(".previous_objective.txt", "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            self.logger.error(f"Failed to save objective to file: {e}")

    # --- MAIN LOOP ---
    def run(self, objective: str, max_iterations: Optional[int] = None, step_callback: Optional[Callable[[int, int, str], None]] = None, terminal_tools: List[str] = None):
        """Runs the worker loop. 
        Args:
            objective: The goal.
            max_iterations: Limit on steps. None = infinite.
            step_callback: Optional function(iteration, max_iterations, status_msg).
            terminal_tools: List of tool names that cause the loop to exit successfully (default: ['task_complete'])
        """
        if terminal_tools is None:
            terminal_tools = ['task_complete']
            
        self.logger.info("Starting Execution for: %s", objective)
        self._save_objective(objective)
        
        iteration = 0
        
        # Initialize last_observation if this is a fresh start
        if self.last_observation == "None.":
            self.last_observation = "Project started."

        self.print_func(f"{C_GREEN}Objective: {objective}{C_RESET}\n")

        graceful_exit_triggered = False

        while True:
            try:
                iteration += 1
                
                # Display max as '∞' if None
                display_max = max_iterations if max_iterations is not None else 999
                
                if step_callback:
                    step_callback(iteration, display_max, "Thinking")

                # --- GRACEFUL EXIT LOGIC ---
                if max_iterations is not None and iteration > max_iterations:
                    if not graceful_exit_triggered:
                        # ONE LAST CHANCE
                        graceful_exit_triggered = True
                        msg = f"SYSTEM ALERT: Max iterations ({max_iterations}) reached. You have ONE final step. You MUST use a terminal tool ({', '.join(terminal_tools)}) NOW to report your findings (even if incomplete)."
                        self.last_observation = msg
                        self.print_func(f"{C_RED}Max iterations reached. Forcing final report.{C_RESET}")
                    else:
                        self.print_func(f"{C_RED}Agent failed to exit. Terminating.{C_RESET}")
                        break

                self.print_func(f"{C_BLUE}--- Iteration {iteration} ---{C_RESET}")
                
                # --- PREPARE CONTEXT ---
                system_specs = get_runtime_info()
                tool_list_str = self._get_tools_description()
                system_context = f"""**System Info**
{system_specs}

**Core Directives**
{self.base_directives}

**Available Tools**
{tool_list_str}
"""

                # --- PLANNER ---
                self.print_func("Thinking (Planning)...")
                history_str = self._format_history()
                
                plan_response_str = self.llm_client.get_plan(
                    system_context=system_context,
                    user_objective=objective,
                    history_str=history_str,
                    current_plan=self.current_plan,
                    last_observation=self.last_observation
                )
                
                next_step_suggestion = ""
                try:
                    plan_data = json.loads(plan_response_str)
                    self.current_plan = plan_data.get("updated_plan", self.current_plan)
                    next_step_suggestion = plan_data.get("next_step_suggestion", "")
                    thought_process = plan_data.get("thought_process", "")
                    self.print_func(f"{C_CYAN}Thought:{C_RESET} {thought_process}")
                    
                    # FIX: Show smarter Plan Preview (Start and End of plan)
                    plan_lines = self.current_plan.split('\n')
                    if len(plan_lines) > 8:
                        preview = '\n'.join(plan_lines[:4]) + f'\n{C_YELLOW}... [plan truncated] ...{C_RESET}\n' + '\n'.join(plan_lines[-4:])
                    else:
                        preview = self.current_plan
                    self.print_func(f"{C_CYAN}Plan Update:{C_RESET}\n{preview}")
                    
                except json.JSONDecodeError:
                    next_step_suggestion = "Analyze previous error."

                # --- EXECUTOR ---
                if step_callback:
                    step_callback(iteration, display_max, "Executing")

                open_files_str = self._format_open_files()
                action_json_str = self.llm_client.get_action(
                    system_context=system_context,
                    plan=self.current_plan,
                    suggestion=next_step_suggestion,
                    open_files_context=open_files_str
                )

                actions = []
                try:
                    clean_json = action_json_str.strip()
                    if clean_json.startswith("```json"): clean_json = clean_json[7:]
                    if clean_json.startswith("```"): clean_json = clean_json[3:]
                    if clean_json.endswith("```"): clean_json = clean_json[:-3]
                    clean_json = clean_json.strip()
                    parsed = json.loads(clean_json)
                    
                    if isinstance(parsed, list): actions = parsed
                    elif isinstance(parsed, dict):
                        if "actions" in parsed and isinstance(parsed["actions"], list):
                            actions = parsed["actions"]
                        elif "tool_name" in parsed:
                            actions = [parsed]
                except Exception as e:
                    self.last_observation = f"Error parsing action: {e}"
                    self.recent_history.append({"iteration": iteration, "action": "Parse Error", "summary": self.last_observation})
                    continue

                if not actions:
                    continue

                # --- EXECUTION CHAIN ---
                combined_summary_parts = []
                actions_taken_str = []
                restart_main_loop = False 
                
                if len(actions) > 15: actions = actions[:15]

                for idx, action_data in enumerate(actions):
                    if restart_main_loop: break

                    tool_name = action_data.get("tool_name")
                    params = action_data.get("parameters", {})
                    allow_failure = action_data.get("allow_failure", False)
                    
                    if not tool_name or tool_name not in self.tools:
                        combined_summary_parts.append(f"Action {idx+1}: Tool '{tool_name}' not found.")
                        break

                    self.print_func(f"{C_YELLOW}Executing (Step {idx+1}):{C_RESET} {tool_name} {params}")
                    actions_taken_str.append(f"{tool_name}")

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
                        break
                    
                    else:
                        try:
                            tool = self.tools[tool_name]
                            raw_result = tool.execute(**params)
                        except Exception as e:
                            raw_result = f"Tool Execution Error: {e}"

                        result_str = str(raw_result)
                        combined_summary_parts.append(f"Action {idx+1} ({tool_name}):\n{result_str}")
                        
                        if "COMMAND FAILED" in result_str or result_str.strip().startswith("Error:"):
                            if not allow_failure:
                                break

                # --- SUMMARIZER ---
                if not combined_summary_parts:
                     summary = "No actions executed."
                else:
                    full_raw_output = "\n\n".join(combined_summary_parts)
                    if len(full_raw_output) < 200 and len(actions) == 1 and actions[0].get('tool_name') != "run_command":
                        summary = full_raw_output
                    else:
                        # FIX: Remove redundant "Summarizing..." print
                        command_context = f"Chain: {', '.join(actions_taken_str)}"
                        summary = self.llm_client.summarize_execution(command_context, full_raw_output)

                self.last_observation = summary
                self.recent_history.append({"iteration": iteration, "action": f"Chain: {len(actions)} tools", "summary": summary})

            except KeyboardInterrupt:
                self.print_func(f"\n{C_RED}PAUSED.{C_RESET}")
                break

    def estimate_tokens(self, text):
        return len(text) // 4
