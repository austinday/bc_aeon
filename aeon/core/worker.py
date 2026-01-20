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

                # Parse Action (Handling List, Dict, or Markdown wrapped)
                actions = []
                try:
                    # 1. Clean Markdown artifacts
                    clean_json = action_json_str.strip()
                    if clean_json.startswith("```json"):
                        clean_json = clean_json[7:]
                    if clean_json.startswith("```"):
                         clean_json = clean_json[3:]
                    if clean_json.endswith("```"):
                        clean_json = clean_json[:-3]
                    clean_json = clean_json.strip()

                    # 2. Permissive Parsing
                    parsed = json.loads(clean_json)
                    
                    if isinstance(parsed, list):
                        # Direct list of actions
                        actions = parsed
                    elif isinstance(parsed, dict):
                        if "actions" in parsed and isinstance(parsed["actions"], list):
                            # Standard wrapper
                            actions = parsed["actions"]
                        elif "tool_name" in parsed:
                            # Legacy single action
                            actions = [parsed]
                        else:
                             raise ValueError("JSON dict must contain 'actions' list or 'tool_name'.")
                    else:
                        raise ValueError(f"Invalid JSON structure: {type(parsed)}")

                except json.JSONDecodeError:
                    self.logger.error("Action output invalid JSON.")
                    self.last_observation = f"Error: Agent produced invalid JSON for action: {action_json_str}"
                    self.recent_history.append({"iteration": iteration, "action": "Failed JSON parse", "summary": self.last_observation})
                    continue
                except Exception as e:
                    self.logger.error(f"Action parsing error: {e}")
                    self.last_observation = f"Error parsing action: {e}"
                    self.recent_history.append({"iteration": iteration, "action": "Invalid Action Format", "summary": self.last_observation})
                    continue

                if not actions:
                    msg = "Error: No actions provided in the response."
                    print(f"{C_RED}{msg}{C_RESET}")
                    self.last_observation = msg
                    self.recent_history.append({"iteration": iteration, "action": "Empty Action List", "summary": msg})
                    continue

                # -----------------------------------------------------------
                # STEP 3: EXECUTION CHAIN
                # -----------------------------------------------------------
                combined_summary_parts = []
                actions_taken_str = []
                
                # Chain Control Flags
                restart_main_loop = False 
                
                # Max actions config
                MAX_ACTIONS = 15
                if len(actions) > MAX_ACTIONS:
                    print(f"{C_YELLOW}Warning: Truncating action list from {len(actions)} to {MAX_ACTIONS}.{C_RESET}")
                    actions = actions[:MAX_ACTIONS]

                for idx, action_data in enumerate(actions):
                    if restart_main_loop:
                        break

                    tool_name = action_data.get("tool_name")
                    params = action_data.get("parameters", {})
                    allow_failure = action_data.get("allow_failure", False)
                    
                    if not tool_name or tool_name not in self.tools:
                        msg = f"Error: Tool '{tool_name}' not found."
                        print(f"{C_RED}{msg}{C_RESET}")
                        combined_summary_parts.append(f"Action {idx+1}: {msg}")
                        actions_taken_str.append(f"{tool_name} (Failed)")
                        # Break chain on missing tool
                        break

                    print(f"{C_YELLOW}Executing (Step {idx+1}/{len(actions)}):{C_RESET} {tool_name} {params}")
                    actions_taken_str.append(f"{tool_name}")

                    # --- SPECIAL FLOW CONTROL TOOLS ---
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
                             restart_main_loop = True
                             break
                        
                        elif intent == "MODIFY_OBJECTIVE":
                            print(f"{C_GREEN}Updating objective: {updated_text}{C_RESET}")
                            objective = updated_text
                            self._save_objective(objective)
                            self.last_observation = f"Task marked complete, but user modified objective: {updated_text}. Previous context retained."
                            self.recent_history.append({"iteration": iteration, "action": "User Modification", "summary": self.last_observation})
                            restart_main_loop = True
                            break
                        
                        else: # ADVICE
                             print(f"{C_GREEN}Resuming with feedback...{C_RESET}")
                             self.last_observation = f"Task marked complete, but user provided feedback: {updated_text}. Resume work."
                             self.recent_history.append({"iteration": iteration, "action": "User Feedback", "summary": self.last_observation})
                             restart_main_loop = True
                             break

                    elif tool_name == "get_user_input":
                        user_in = input(f"{C_YELLOW}Agent requests input: {params.get('prompt')}\n> {C_RESET}")
                        raw_result = f"User Input Provided: {user_in}"
                        combined_summary_parts.append(f"Action {idx+1} ({tool_name}): {raw_result}"
                                                      f"\n[CHAIN STOPPED]: Stopped to wait for user input.")
                        # Force break to allow planner to digest input
                        break
                    
                    else:
                        # --- STANDARD EXECUTION ---
                        try:
                            tool = self.tools[tool_name]
                            raw_result = tool.execute(**params)
                        except Exception as e:
                            raw_result = f"Tool Execution Error: {e}"

                        # Accumulate result
                        result_str = str(raw_result)
                        combined_summary_parts.append(f"Action {idx+1} ({tool_name}):\n{result_str}")
                        
                        # PRINT RESULT TO USER (Truncated)
                        if tool_name not in ["run_command", "say_to_user"]:
                            preview = result_str[:500] + "..." if len(result_str) > 500 else result_str
                            print(f"{C_CYAN}Result:{C_RESET} {preview}")

                        # --- STOP-ON-FAIL HEURISTIC ---
                        # Check for common failure indicators in tool output
                        # "COMMAND FAILED" comes from RunCommandTool
                        # "Error:" or "ERROR:" are common prefixes in our tools
                        # "Tool Execution Error:" comes from the try/except block above
                        is_failure = (
                            "COMMAND FAILED" in result_str or 
                            result_str.strip().startswith("Error:") or 
                            result_str.strip().startswith("ERROR:") or
                            result_str.startswith("Tool Execution Error:")
                        )
                        
                        if is_failure:
                            if allow_failure:
                                print(f"{C_YELLOW}Step {idx+1} Failed, but allow_failure=True. Continuing chain.{C_RESET}")
                                combined_summary_parts.append(f"Action {idx+1} Failed (Ignored due to allow_failure).")
                            else:
                                print(f"{C_RED}Step {idx+1} Failed. Stopping chain.{C_RESET}")
                                combined_summary_parts.append(f"\n[CHAIN STOPPED]: Step {idx+1} failed. Subsequent actions aborted.")
                                break
                
                if restart_main_loop:
                    continue

                # -----------------------------------------------------------
                # STEP 4: SUMMARIZER LLM
                # -----------------------------------------------------------
                if not combined_summary_parts:
                     self.last_observation = "No actions executed."
                     summary = "No actions executed."
                else:
                    full_raw_output = "\n\n".join(combined_summary_parts)
                    
                    # Small optimization: If output is tiny and just 1 action, don't use LLM
                    if len(full_raw_output) < 200 and len(actions) == 1 and actions[0].get('tool_name') not in ["run_command"]:
                        summary = full_raw_output
                    else:
                        print("Summarizing output...")
                        # Describe the full chain in command context
                        command_context = f"Chain of {len(actions_taken_str)} actions: {', '.join(actions_taken_str)}"
                        summary = self.llm_client.summarize_execution(command_context, full_raw_output)

                self.last_observation = summary
                
                self.recent_history.append({
                    "iteration": iteration,
                    "action": f"Chain: {json.dumps([a.get('tool_name') for a in actions])}",
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
