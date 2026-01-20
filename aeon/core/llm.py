import os
import openai
import pathlib
import sys
import json
from typing import Dict
sys.setrecursionlimit(2000)
from .system_info import get_runtime_info
from .logger import get_logger

class LLMClient:
    """A client for interacting with a Large Language Model."""
    def __init__(self, provider: str = "grok"):
        self.provider = provider
        
        if provider == "gemini":
            self.strong_model = "gemini-3-pro-preview"
            self.weak_model = "gemini-flash-latest"
            api_key_filename = "gemini_api_key.txt"
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            # Default to Grok
            self.strong_model = "grok-4-1-fast-reasoning-latest"
            self.weak_model = "grok-4-1-fast-non-reasoning-latest"
            api_key_filename = "grok_api_key.txt"
            base_url = "https://api.x.ai/v1"
        
        # Load important reminders
        self.important_reminders_path = pathlib.Path(__file__).parent / "prompts" / "important_reminders.txt"
        self.important_reminders = ""
        if self.important_reminders_path.is_file():
            with open(self.important_reminders_path, 'r') as f:
                self.important_reminders = f.read().strip()
        
        api_key_path = pathlib.Path.home() / api_key_filename
        api_key = None
        if api_key_path.is_file():
            with open(api_key_path, 'r') as f:
                api_key = f.readline().strip()

        if not api_key:
            raise ValueError(f"{provider.capitalize()} API key not found in ~/{api_key_filename}. Please create this file and add your API key to it.")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.logger = get_logger()

    def _add_system_context(self, prompt: str) -> str:
        """Adds real-time system context to the prompt."""
        runtime_info = get_runtime_info()
        return f"{runtime_info}\n\nUser Prompt:\n{prompt}"

    # --- NEW STATE-BASED METHODS ---

    def get_plan(self, system_context: str, user_objective: str, history_str: str, current_plan: str, last_observation: str) -> str:
        """Step 1: The Planner. Returns a strategic plan and a next step suggestion."""
        prompt = f"""{system_context}

OBJECTIVE: {user_objective}

CURRENT SAVED PLAN:
{current_plan}

RECENT HISTORY (Last 10 steps):
{history_str}

LAST OBSERVATION (From previous step):
{last_observation}

INSTRUCTIONS:
1. Review the Objective, Current Plan, Recent History, and Last Observation.
2. Look at the meta, higher-level strategic view. 
3. Determine if the project is progressing, stagnating, or looping.
4. Review the Recent History for KEY ACCOMPLISHMENTS (e.g., "Docker image built", "Environment verified").
5. If stagnating, repeating the same fixes (7+ times), or looping, PIVOT immediately. Create a new strategy.
6. If progressing, refine the Current Plan.
7. Maintain a section in your `updated_plan` called "## Verified Milestones" to track assets (like built images) that persist even if a specific test fails.
8. Suggest the IMMEDIATE NEXT ACTION (single tool call suggestion) to move forward.

OUTPUT FORMAT:
You must output a JSON object:
{{
  "thought_process": "Analysis of the situation... I have identified these key milestones: ...",
  "updated_plan": "## Verified Milestones\n- Image built...\n\n## Next Steps\n1. Step one... 2. Step two...",
  "next_step_suggestion": "Run command 'ls -la' to check files..."
}}

IMPORTANT REMINDERS:
{self.important_reminders}
"""
        self.logger.info(f'Full prompt for get_plan: {prompt}')
        model = self.strong_model
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10000,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Planner Error: {e}")
            # Return a valid JSON structure so the worker doesn't crash
            return json.dumps({
                "thought_process": f"Error in planning: {e}",
                "updated_plan": current_plan,
                "next_step_suggestion": "Retry previous action or check logs."
            })

    def get_action(self, system_context: str, plan: str, suggestion: str, open_files_context: str) -> str:
        """Step 2: The Executor. Returns the specific tool call."""
        prompt = f"""{system_context}

CURRENT PLAN:
{plan}

SUGGESTED NEXT STEP:
{suggestion}

OPEN FILES (Short Term Memory):
{open_files_context}

INSTRUCTIONS:
1. Based on the Plan and Suggestion, formulate the exact JSON for the tool execution.
2. Verify you are using valid tools.
3. If you need to read a file, use open_file. If you need to write, use write_file.

OUTPUT FORMAT:
MUST be a single valid JSON object for the tool call.
Example: {{"tool_name": "run_command", "parameters": {{"command": "ls -la"}}}}

IMPORTANT REMINDERS:
{self.important_reminders}
"""
        self.logger.info(f'Full prompt for get_action: {prompt}')
        model = self.strong_model
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, 
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Action Error: {e}")
            return ""

    def summarize_execution(self, command_context: str, raw_output: str) -> str:
        """Step 4: The Observer. Summarizes the raw output."""
        prompt = f"""You are an Objective Observer.
        
COMMAND/ACTION RUN:
{command_context}

RAW OUTPUT/LOGS:
{raw_output}

INSTRUCTIONS:
1. Objectively summarize the output.
2. HIGHLIGHT ALL ERRORS, WARNINGS, and FAILURES. Do not gloss over them.
3. Note what was successfully accomplished. Explicitly flag KEY ACCOMPLISHMENTS (e.g., "Docker image successfully built", "Script syntax valid").
4. If it was a file read, summarize the content relevance.
5. Be concise but COMPLETE. This summary will be the only record of this action.
6. Helpful doesn't mean optimistic. It means catching problems early but acknowledging wins.

OUTPUT:
A text summary.

IMPORTANT REMINDERS:
{self.important_reminders}
"""
        self.logger.info(f'Full prompt for summarize_execution: {prompt}')
        model = self.strong_model 
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=10000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error summarizing output: {e}. Raw output start: {raw_output[:500]}"

    def analyze_interruption(self, current_objective: str, user_input: str) -> Dict[str, str]:
        """Analyzes user input during a pause to determine intent."""
        prompt = f"""The user has interrupted the agent's execution loop.
CURRENT OBJECTIVE: {current_objective}
USER INPUT: {user_input}

INSTRUCTIONS:
Determine the user's intent. Choose exactly one of the following classifications:
1. "NEW_TASK": The user wants to stop the current task completely and start something entirely new.
2. "MODIFY_OBJECTIVE": The user wants to keep the current progress/files but slightly alter the goal or constraints.
3. "ADVICE": The user is giving a hint, correcting a mistake, or providing data to help the agent continuously solve the CURRENT objective.

OUTPUT FORMAT:
Return a JSON object:
{{
  "classification": "NEW_TASK" | "MODIFY_OBJECTIVE" | "ADVICE",
  "reasoning": "Brief explanation...",
  "updated_text": "The text to use (e.g., the new objective or the formatted advice string)"
}}

IMPORTANT REMINDERS:
{self.important_reminders}
"""
        self.logger.info(f'Full prompt for analyze_interruption: {prompt}')
        model = self.strong_model
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "classification": "ADVICE",
                "reasoning": f"Error parsing intent: {e}",
                "updated_text": user_input
            }

    # --- LEGACY METHODS (Kept for compatibility) ---

    def reason(self, prompt: str) -> str:
        """Legacy reasoner."""
        final_prompt = self._add_system_context(prompt) + f"\n\nIMPORTANT REMINDERS:\n{self.important_reminders}"
        self.logger.info(f'Full prompt for reason: {final_prompt}')
        try:
            response = self.client.chat.completions.create(
                model=self.strong_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.2,
                max_tokens=30000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def summarize_text(self, text_to_summarize: str, query: str) -> str:
        """Summarizes text (used by search tool)."""
        system_prompt = f"You are a helpful assistant that summarizes web search results. Provide a concise summary of the following content, focusing on what is most relevant to the user's query: '{query}'\n\nIMPORTANT REMINDERS:\n{self.important_reminders}"
        full_prompt = f"System: {system_prompt}\nUser: {text_to_summarize}"
        self.logger.info(f'Full prompt for summarize_text: {full_prompt}')
        try:
            response = self.client.chat.completions.create(
                model=self.weak_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_to_summarize}
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Could not summarize the text due to an error: {e}"
