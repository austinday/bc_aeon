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
from .directives import CORE_DIRECTIVES, DOCKER_DIRECTIVES

# Colors for terminal output
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_RESET = '\033[0m'
C_BLUE = '\033[94m'

# Planner prompt sections
PLANNER_INSTRUCTIONS = """CRITICAL: Your response must be ONLY a valid JSON object. No text before the opening {. No text after the closing }. Use double quotes (") only, never single quotes ('). No markdown formatting or code fences.

**Instructions**
You are a high level planning agent. You will analyze the full context given (objectives, history, observations, milestones, system state, etc...) and create a strategic plan for how to achieve and validate the successful implementation of the objective from the current project state. Asses the current state of the project, what has been completed and what is still needed. Plan multiple steps ahead when the path is clear or conversational, if conversational, respond and conclude. Plan one step ahead if the next step is unclear, high risk, or complex. Internally, plan ahead step by step and consider multiple paths and solutions to get to the objective. For very complex tasks, consider using the conduct_research tool. Consider the probabilities of all the possible paths, then suggest the most probable solution, but consider alternative paths as suggestions or risks to be aware of.

**Your job is HIGH-LEVEL THINKING, not crafting tool calls.** Write your next_actions as free-form natural language descriptions of what should be done next. The Executor will translate your ideas into actual tool calls. Do NOT write JSON or structured parameters - just describe the intent.

**Output Format:**
You must output a JSON object:
{
  "analysis": "Brief assessment of current state and progress...",
  "updated_plan": "## Remaining Steps\n- [ ] Step 1...\n- [ ] Step 2...",
  "next_actions": "Free-form description of what to do next. Example: First, search the web for information about X. Then, write a Dockerfile that includes Y. Finally, run the container to test.",
  "iteration_strategy": "single_step" | "multi_step",
  "risk_notes": "Any concerns or things to watch for"
}

WRONG (will cause errors):
- {'analysis': ...}  <- Single quotes are invalid JSON
- ```json {...} ```  <- Markdown fences break parsing
- Let me think... {...}  <- Text before JSON
- {...} I'll explain...  <- Text after JSON
- {...},  <- Trailing comma

CORRECT:
{"analysis": "...", "updated_plan": "...", "next_actions": [...], "iteration_strategy": "...", "risk_notes": "..."}

Output ONLY the JSON object now:"""

# Executor prompt sections  
EXECUTOR_INSTRUCTIONS = """CRITICAL: Your response must be ONLY a valid JSON object. No text before the opening {. No text after the closing }. Use double quotes (") only, never single quotes ('). No markdown formatting or code fences.

You are the Execution Agent. Your task is to translate the plan into concrete tool calls.

**Your Responsibilities:**
1. Read the current plan and suggested next actions
2. Formulate precise tool calls with exact parameters
3. Execute the suggested actions faithfully
4. For conversations: respond to user AND use terminal tools (task_complete/get_user_input) to properly end or continue

**Critical Rules:**
- NEVER return empty actions. You MUST always output at least one tool call.
- If the objective is conversational (greeting, question, simple request): use say_to_user THEN task_complete or get_user_input
- Never leave a conversation hanging - always conclude with a terminal action when appropriate
- Include ALL required parameters for each tool call
- If multiple actions are suggested and safe, execute them all in one iteration

**Output Format:**
You MUST output a JSON object with an "actions" list:
{"actions": [{"tool_name": "say_to_user", "parameters": {"message": "Hello!"}}, {"tool_name": "task_complete", "parameters": {"reason": "Greeted user as requested."}}]}

Another example with allow_failure:
{"actions": [{"tool_name": "run_command", "parameters": {"command": "ls -la"}, "allow_failure": true}]}

WRONG (will cause errors):
- {'actions': [...]}  <- Single quotes are invalid JSON
- ```json {...} ```  <- Markdown fences break parsing
- Let me analyze... {...}  <- Text before JSON
- {...} Now I'll explain...  <- Text after JSON
- {"actions": [...],}  <- Trailing comma

CORRECT:
{"actions": [{"tool_name": "run_command", "parameters": {"command": "ls"}}]}

Output ONLY the JSON object now:"""

MILESTONE_ANALYZER_INSTRUCTIONS = """**Instructions**
You are analyzing the results of the most recent agent iteration to determine if any significant MILESTONES were achieved.

A MILESTONE is:
- A concrete, verifiable step toward completing the objective
- Something foundational that won't need to be redone (e.g., "Created project structure", "Database connection established", "Core algorithm implemented and tested")
- NOT minor actions like "opened a file" or "ran a command"

Review the iteration results and determine if any milestones were completed.

**Output Format**
You must output a JSON object:
{
  "analysis": "Brief analysis of what happened this iteration...",
  "milestones_achieved": ["Milestone 1", "Milestone 2"] or [] if none
}

Be conservative - only mark true milestones, not routine steps."""

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
        self.recent_history = deque(maxlen=10) 
        self.completed_milestones = []  # Foundational progress markers, append-only
        self.last_observation = "None."
        
        self.base_directives = CORE_DIRECTIVES
        self.docker_directives = DOCKER_DIRECTIVES
        self.max_history_tokens = 25000
        
        # Load important reminders
        self._load_reminders()

    def _init_debug_logging(self):
        """Initialize debug logging once per worker instance."""
        if self._debug_initialized:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = Path.home() / f"aeon_debug_{ts}.log"
        self.llm_client.set_debug_path(debug_path)
        self.print_func(f"{C_YELLOW}Debug logging enabled: {debug_path}{C_RESET}")
        self._debug_initialized = True

    def _load_reminders(self):
        path = Path(__file__).parent / "prompts" / "important_reminders.txt"
        self.important_reminders = path.read_text().strip() if path.exists() else ""

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
        return f"""{self.base_directives}

{self.docker_directives}

**Available Tools**
{tool_list_str}

**Important Reminders**
{self.important_reminders}

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

    def _build_executor_context(self, tool_list_str: str, milestones_str: str,
                                 plan: str, suggested_actions: str, open_files_str: str) -> str:
        """Build the complete executor prompt with instructions at the end."""
        return f"""{self.base_directives}

{self.docker_directives}

**Available Tools**
{tool_list_str}

**Important Reminders**
{self.important_reminders}

**Completed Milestones (Foundational Progress)**
{milestones_str}

**Current Plan**
{plan}

**Suggested Next Actions (from Planner)**
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
                            actions_taken: List[str], iteration_result: str,
                            base_context: str) -> None:
        """Analyze the iteration results to identify any completed milestones.
        Milestones are foundational progress markers that won't need to be redone.
        """
        try:
            milestones_str = self._format_milestones()
            history_str = self._format_history()
            
            analysis_context = f"""{base_context}

{MILESTONE_ANALYZER_INSTRUCTIONS}

**Objective**
{objective}

**Already Completed Milestones**
{milestones_str}

**Recent History**
{history_str}

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
        if self.last_observation == "None.":
            self.last_observation = "Project started."

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
                planner_prompt = self._build_planner_context(
                    tool_list_str, system_specs, milestones_str, objective, history_str, open_files_str
                )

                # --- PLANNER ---
                self.print_func("Thinking (Planning)...")
                
                plan_response_str = self.llm_client.get_plan(prompt=planner_prompt)
                
                suggested_actions_str = "No specific actions suggested."
                try:
                    plan_data = json.loads(plan_response_str)
                    self.current_plan = plan_data.get("updated_plan", self.current_plan)
                    
                    # Format suggested actions for executor (now free-form text from planner)
                    next_actions = plan_data.get("next_actions", "")
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
                    
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse planner response as JSON: {e}")
                    suggested_actions_str = "Planner output was not valid JSON. Analyze previous state and proceed."

                # --- EXECUTOR ---
                if step_callback:
                    step_callback(iteration, display_max, "Executing")

                # open_files_str is already generated above for the planner
                
                # Build executor prompt (separate from planner, instructions at end)
                executor_prompt = self._build_executor_context(
                    tool_list_str, milestones_str, self.current_plan, 
                    suggested_actions_str, open_files_str
                )
                
                action_json_str = self.llm_client.get_action(prompt=executor_prompt)

                actions = []
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
                    self.last_observation = f"Error parsing action JSON: {e}. Raw response: {action_json_str[:200]}..."
                    self.logger.error(f"JSON parse error: {e}")
                    self.recent_history.append({"iteration": iteration, "action": "Parse Error", "summary": self.last_observation})
                    continue
                except Exception as e:
                    self.last_observation = f"Unexpected error parsing action: {e}"
                    self.logger.error(f"Action parse error: {e}")
                    self.recent_history.append({"iteration": iteration, "action": "Parse Error", "summary": self.last_observation})
                    continue

                if not actions:
                    self.last_observation = "No actions parsed from executor response."
                    self.recent_history.append({"iteration": iteration, "action": "No Actions", "summary": self.last_observation})
                    continue

                combined_summary_parts = []
                actions_taken_str = []
                restart_main_loop = False 
                
                if len(actions) > 15: 
                    actions = actions[:15]
                    self.logger.warning(f"Truncated actions from {len(actions)} to 15")

                for idx, action_data in enumerate(actions):
                    if restart_main_loop: break

                    tool_name = action_data.get("tool_name")
                    params = action_data.get("parameters", {})
                    allow_failure = action_data.get("allow_failure", False)
                    
                    if not tool_name:
                        combined_summary_parts.append(f"Action {idx+1}: Missing tool_name in action data.")
                        break
                    
                    if tool_name not in self.tools:
                        combined_summary_parts.append(f"Action {idx+1}: Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")
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
                        except TypeError as e:
                            raw_result = f"Tool Parameter Error: {e}. Check required parameters for '{tool_name}'."
                        except Exception as e:
                            raw_result = f"Tool Execution Error: {type(e).__name__}: {e}"

                        result_str = str(raw_result)
                        combined_summary_parts.append(f"Action {idx+1} ({tool_name}):\n{result_str}")
                        
                        if "COMMAND FAILED" in result_str or result_str.strip().startswith("Error:"):
                            if not allow_failure:
                                break

                if not combined_summary_parts:
                    summary = "No actions executed."
                else:
                    full_raw_output = "\n\n".join(combined_summary_parts)
                    if len(full_raw_output) < 200 and len(actions) == 1 and actions[0].get('tool_name') != "run_command":
                        summary = full_raw_output
                    else:
                        command_context = f"Chain: {', '.join(actions_taken_str)}"
                        summary = self.llm_client.summarize_execution(command_context, full_raw_output)

                self.last_observation = summary
                self.recent_history.append({"iteration": iteration, "action": f"Chain: {len(actions)} tools", "summary": summary})

                # --- MILESTONE ANALYSIS (after iteration completes) ---
                # Analyze if any foundational milestones were achieved this iteration
                base_context = self._build_base_context(tool_list_str)
                self._analyze_milestones(
                    objective=objective,
                    iteration=iteration,
                    actions_taken=actions_taken_str,
                    iteration_result=summary,
                    base_context=base_context
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
