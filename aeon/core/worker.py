import json
import re
import time
from datetime import datetime
from collections import deque
from importlib import resources
from typing import List, Any, Dict

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
    def __init__(self, llm_client: LLMClient, tools: List[Any] = None):
        self.llm_client = llm_client
        self.tools = {tool.name: tool for tool in tools} if tools else {}
        self.logger = get_logger()
        
        # --- STATE MODEL ---
        self.current_plan = "No plan formulated yet."
        self.open_files = {} # Dict[path, content]
        # Increased memory window to prevent "amnesia" regarding previous successes
        self.recent_history = deque(maxlen=10) # Stores last 10 {step, action, summary}
        self.last_observation = "None."
        
        self.base_directives = ""
        self.max_history_tokens = 25000 # Unused in state-based, but kept for interface compatibility
        self._load_directives()

    def register_tools(self, tools_list: List[Any]):
        """Dynamically add tools."""
        for tool in tools_list:
            self.tools[tool.name] = tool

    # --- TAB MANAGEMENT ---
    def update_open_file(self, path: str, content: str):
        """Called by OpenFileTool."""
        self.open_files[path] = content

    def close_file(self, path: str) -> bool:
        """Called by CloseFileTool."""
        # Simple check for direct match or relative path match
        if path in self.open_files:
            del self.open_files[path]
            return True
        return False

    def is_file_open(self, path: str) -> bool:
        return path in self.open_files

    # --- HELPERS ---
    def _load_directives(self):
        try:
            # We load core directives.
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
        """Resets the internal state for a fresh start."""
        self.current_plan = "Initial state. Need to formulate a plan."
        self.open_files = {}
        self.recent_history.clear()
        self.last_observation = initial_observation

    def _save_objective(self, objective: str):
        """Saves the objective to a hidden history file."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"[{timestamp}] OBJECTIVE UPDATE:\n{objective}\n{'-'*40}\n"
            
            with open(".previous_objective.txt", "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            self.logger.error(f"Failed to save objective to file: {e}")

    # --- MAIN LOOP ---
    def run(self, objective: str):
        self.logger.info("Starting State-Based Execution for: %s", objective)
        
        # Save initial objective
        self._save_objective(objective)
        
        iteration = 0
        self._reset_state()

        print(f"{C_GREEN}Objective: {objective}{C_RESET}\n")

        # We wrap the entire loop in a try/except for the outer KeyboardInterrupt (hard kill),
        # but we also handle internal pauses.
        while True:
            try:
                iteration += 1
                print(f"{C_BLUE}--- Iteration {iteration} ---{C_RESET}")
                
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

                # -----------------------------------------------------------
                # STEP 1: PLANNER LLM
                # -----------------------------------------------------------
                print("Thinking (Planning)...")
                history_str = self._format_history()
                
                plan_response_str = self.llm_client.get_plan(
                    system_context=system_context,
                    user_objective=objective,
                    history_str=history_str,
                    current_plan=self.current_plan,
                    last_observation=self.last_observation
                )
                
                # Parse Plan
                next_step_suggestion = ""
                try:
                    plan_data = json.loads(plan_response_str)
                    self.current_plan = plan_data.get("updated_plan", self.current_plan)
                    next_step_suggestion = plan_data.get("next_step_suggestion", "")
                    thought_process = plan_data.get("thought_process", "")
                    
                    print(f"{C_CYAN}Thought:{C_RESET} {thought_process}")
                    print(f"{C_CYAN}Plan Update:{C_RESET} {self.current_plan[:150]}...")
                    print(f"{C_CYAN}Suggestion:{C_RESET} {next_step_suggestion}")
                    
                except json.JSONDecodeError:
                    self.logger.error("Planner output invalid JSON.")
                    next_step_suggestion = "Analyze the previous error and retry."

                # -----------------------------------------------------------
                # STEP 2: ACTION LLM
                # -----------------------------------------------------------
                open_files_str = self._format_open_files()
                
                action_json_str = self.llm_client.get_action(
                    system_context=system_context,
                    plan=self.current_plan,
                    suggestion=next_step_suggestion,
                    open_files_context=open_files_str
                )

                # Parse Action
                tool_name = None
                params = {}
                try:
                    action_obj = json.loads(action_json_str)
                    tool_name = action_obj.get("tool_name")
                    params = action_obj.get("parameters", {})
                except json.JSONDecodeError:
                    self.logger.error("Action output invalid JSON.")
                    self.last_observation = f"Error: Agent produced invalid JSON for action: {action_json_str}"
                    self.recent_history.append({"iteration": iteration, "action": "Failed JSON parse", "summary": self.last_observation})
                    continue

                if not tool_name or tool_name not in self.tools:
                    msg = f"Error: Tool '{tool_name}' not found."
                    print(f"{C_RED}{msg}{C_RESET}")
                    self.last_observation = msg
                    self.recent_history.append({"iteration": iteration, "action": str(tool_name), "summary": msg})
                    continue

                # -----------------------------------------------------------
                # STEP 3: EXECUTION
                # -----------------------------------------------------------
                print(f"{C_YELLOW}Executing:{C_RESET} {tool_name} {params}")
                
                if tool_name == "task_complete":
                    print(f"\n{C_GREEN}Task Completed: {params.get('reason')}{C_RESET}")
                    print(f"The agent considers the task done. You can verify the results now.")
                    print(f"Type feedback to refine, a new directive to continue, or press ENTER to finish this session.")
                    
                    try:
                        user_response = input(f"{C_YELLOW}Feedback/Next Task > {C_RESET}").strip()
                    except EOFError:
                        break

                    if not user_response:
                        break
                    
                    if user_response.lower() in ['exit', 'quit']:
                        break

                    # Analyze intent
                    print("Analyzing input...")
                    intent_data = self.llm_client.analyze_interruption(objective, user_response)
                    intent = intent_data.get("classification")
                    updated_text = intent_data.get("updated_text", user_response)
                    reasoning = intent_data.get("reasoning", "")

                    print(f"{C_CYAN}Interpretation: {intent} ({reasoning}){C_RESET}")

                    if intent == "NEW_TASK":
                         print(f"{C_GREEN}Resetting for new task: {updated_text}{C_RESET}")
                         objective = updated_text
                         self._save_objective(objective)
                         self._reset_state(initial_observation=f"Task reset by user. New Objective: {objective}")
                         iteration = 0
                         continue
                    
                    elif intent == "MODIFY_OBJECTIVE":
                        print(f"{C_GREEN}Updating objective: {updated_text}{C_RESET}")
                        objective = updated_text
                        self._save_objective(objective)
                        self.last_observation = f"Task marked complete, but user modified objective: {updated_text}. Previous context retained."
                        self.recent_history.append({"iteration": iteration, "action": "User Modification", "summary": self.last_observation})
                        continue
                    
                    else: # ADVICE
                         print(f"{C_GREEN}Resuming with feedback...{C_RESET}")
                         self.last_observation = f"Task marked complete, but user provided feedback: {updated_text}. Resume work."
                         self.recent_history.append({"iteration": iteration, "action": "User Feedback", "summary": self.last_observation})
                         continue

                if tool_name == "get_user_input":
                    user_in = input(f"{C_YELLOW}Agent requests input: {params.get('prompt')}\n> {C_RESET}")
                    raw_result = f"User Input Provided: {user_in}"
                else:
                    try:
                        tool = self.tools[tool_name]
                        raw_result = tool.execute(**params)
                    except Exception as e:
                        raw_result = f"Tool Execution Error: {e}"

                # -----------------------------------------------------------
                # STEP 4: SUMMARIZER LLM
                # -----------------------------------------------------------
                if len(str(raw_result)) < 200 and tool_name not in ["run_command"]:
                        summary = str(raw_result)
                else:
                    print("Summarizing output...")
                    command_context = f"Tool: {tool_name}, Params: {params}"
                    summary = self.llm_client.summarize_execution(command_context, str(raw_result))

                self.last_observation = summary
                
                self.recent_history.append({
                    "iteration": iteration,
                    "action": f"{tool_name} {json.dumps(params)}",
                    "summary": summary
                })

            except KeyboardInterrupt:
                print(f"\n{C_RED}PAUSED BY USER (Ctrl+C).{C_RESET}")
                print(f"Type a message to inject advice, modify the objective, or 'exit' to quit.")
                print(f"Press ENTER without typing to simply resume.")
                try:
                    user_input = input(f"{C_YELLOW}User Input > {C_RESET}").strip()
                except KeyboardInterrupt:
                    print("\nTerminating...")
                    break
                
                if not user_input:
                    print("Resuming...")
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    break

                # Analyze User Intent
                print("Analyzing input...")
                intent_data = self.llm_client.analyze_interruption(objective, user_input)
                intent = intent_data.get("classification")
                updated_text = intent_data.get("updated_text", user_input)
                reasoning = intent_data.get("reasoning", "")

                print(f"{C_CYAN}Interpretation: {intent} ({reasoning}){C_RESET}")

                if intent == "NEW_TASK":
                    print(f"{C_GREEN}Resetting agent for new task: {updated_text}{C_RESET}")
                    objective = updated_text
                    self._save_objective(objective)
                    self._reset_state(initial_observation=f"Task reset by user. New Objective: {objective}")
                    iteration = 0
                
                elif intent == "MODIFY_OBJECTIVE":
                    print(f"{C_GREEN}Updating objective to: {updated_text}{C_RESET}")
                    objective = updated_text
                    self._save_objective(objective)
                    self.last_observation += f"\n\n[USER INTERRUPTION]: User modified the objective. New Objective: {objective}. Verify previous steps against new requirements."
                
                else: # ADVICE
                    print(f"{C_GREEN}Injecting advice into context...{C_RESET}")
                    self.last_observation += f"\n\n[USER INTERRUPTION]: User provided advice/hint: {updated_text}"

    # Legacy method compatibility (if any tools call it)
    def get_history(self):
        return [str(h) for h in self.recent_history]
    
    def set_history(self, history):
        pass 
    
    def estimate_tokens(self, text):
        return len(text) // 4
