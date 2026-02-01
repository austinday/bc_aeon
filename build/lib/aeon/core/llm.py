import os
import openai
import pathlib
import sys
import json
import re
from typing import Dict
sys.setrecursionlimit(2000)
from .system_info import get_runtime_info
from .logger import get_logger

class LLMClient:
    """A client for interacting with Large Language Models (Cloud or Local)."""
    def __init__(self, provider: str = "local"):
        self.provider = provider
        self.logger = get_logger()
        
        # Defaults
        api_key = None
        base_url = None
        
        # --- 1. LOCAL PROVIDER (The Brain) ---
        if provider == "local":
            self.logger.info("Initializing Local Brain (Ollama Backend)...")
            # GPU 0: Planner (DeepSeek R1)
            self.planner_client = openai.OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="ollama" 
            )
            self.planner_model = "deepseek-r1:70b"
            
            # GPU 1: Executor (Qwen 2.5)
            self.executor_client = openai.OpenAI(
                base_url="http://localhost:8001/v1",
                api_key="ollama"
            )
            self.executor_model = "qwen2.5:72b"
            
            self.summarizer_client = self.executor_client
            self.summarizer_model = self.executor_model
            
            self._load_reminders()
            return

        # --- 2. CLOUD PROVIDERS ---
        elif provider == "gemini":
            self.strong_model = "gemini-3-pro-preview"
            self.weak_model = "gemini-flash-latest"
            api_key_filename = "gemini_api_key.txt"
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif provider == "gemini-flash":
            self.strong_model = "gemini-flash-latest"
            self.weak_model = "gemini-flash-latest"
            api_key_filename = "gemini_api_key.txt"
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            # Default to Grok
            self.strong_model = "grok-4-1-fast-reasoning-latest"
            self.weak_model = "grok-4-1-fast-non-reasoning-latest"
            api_key_filename = "grok_api_key.txt"
            base_url = "https://api.x.ai/v1"
        
        # Load API Key
        api_key_path = pathlib.Path.home() / api_key_filename
        if api_key_path.is_file():
            with open(api_key_path, 'r') as f:
                api_key = f.readline().strip()

        if not api_key:
            raise ValueError(f"{provider.capitalize()} API key not found in ~/{api_key_filename}.")
        
        # For Cloud, clients are unified
        self.planner_client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.executor_client = self.planner_client
        self.summarizer_client = self.planner_client
        
        self.planner_model = self.strong_model
        self.executor_model = self.strong_model
        self.summarizer_model = self.weak_model
        
        self._load_reminders()

    def _load_reminders(self):
        self.important_reminders_path = pathlib.Path(__file__).parent / "prompts" / "important_reminders.txt"
        self.important_reminders = ""
        if self.important_reminders_path.is_file():
            with open(self.important_reminders_path, 'r') as f:
                self.important_reminders = f.read().strip()

    def _add_system_context(self, prompt: str) -> str:
        runtime_info = get_runtime_info()
        return f"{runtime_info}\n\nUser Prompt:\n{prompt}"

    def _clean_json_response(self, content: str) -> str:
        """Extracts JSON from text, handling <think> blocks from R1. Only active for local."""
        if not content: return ""
        
        # 1. Strip <think> blocks (DeepSeek R1 specific)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # 2. Extract JSON block if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[0].strip()
            
        # 3. Fallback: find first { and last }
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != -1:
            return content[start:end]
            
        return content

    # --- ROUTING LOGIC ---

    def get_plan(self, system_context: str, user_objective: str, history_str: str, current_plan: str, last_observation: str) -> str:
        """ROUTING: Uses the PLANNER (GPU 0 / Strong Model)."""
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
4. Review the Recent History for KEY ACCOMPLISHMENTS.
5. If stagnating, repeating the same fixes (7+ times), or looping, PIVOT immediately.
6. If progressing, refine the Current Plan.
7. Suggest the IMMEDIATE NEXT ACTION (single tool call suggestion).

OUTPUT FORMAT:
You must output a JSON object:
{{
  "thought_process": "Analysis of the situation...",
  "updated_plan": "## Verified Milestones\n...\n## Next Steps\n...",
  "next_step_suggestion": "Run command 'ls -la'..."
}}

IMPORTANT REMINDERS:
{self.important_reminders}
"""
        self.logger.info(f'Planning with model: {self.planner_model}')
        try:
            # STRICT SEPARATION: Cloud vs Local
            if self.provider == "local":
                # Local: No JSON enforcement (avoids R1 crash), use Cleaner
                response = self.planner_client.chat.completions.create(
                    model=self.planner_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=8192
                )
                raw_content = response.choices[0].message.content
                return self._clean_json_response(raw_content)
            else:
                # Cloud: Use JSON enforcement, NO Cleaner (Preserve original behavior)
                response = self.planner_client.chat.completions.create(
                    model=self.planner_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=8192,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Planner Error: {e}")
            return json.dumps({
                "thought_process": f"Error in planning: {e}",
                "updated_plan": current_plan,
                "next_step_suggestion": "Retry previous action or check logs."
            })

    def get_action(self, system_context: str, plan: str, suggestion: str, open_files_context: str) -> str:
        """ROUTING: Uses the EXECUTOR (GPU 1 / Fast Model)."""
        prompt = f"""{system_context}

CURRENT PLAN:
{plan}

SUGGESTED NEXT STEP:
{suggestion}

OPEN FILES (Short Term Memory):
{open_files_context}

INSTRUCTIONS:
1. Based on the Plan and Suggestion, formulate the exact JSON for the tool execution.
2. ACTION CHAINING: You may output a list of actions.
3. Verify you are using valid tools.

OUTPUT FORMAT:
MUST be a single valid JSON object with an "actions" key.
Example: {{"actions": [{{"tool_name": "write_file", "parameters": {{"file_path": "test.py", "content": "..."}}}}]}}

IMPORTANT REMINDERS:
{self.important_reminders}
"""
        self.logger.info(f'Executing with model: {self.executor_model}')
        try:
            response = self.executor_client.chat.completions.create(
                model=self.executor_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4000
            )
            raw_content = response.choices[0].message.content
            
            if self.provider == "local":
                return self._clean_json_response(raw_content)
            else:
                return raw_content
                
        except Exception as e:
            self.logger.error(f"Action Error: {e}")
            return ""

    def summarize_execution(self, command_context: str, raw_output: str) -> str:
        """ROUTING: Uses the EXECUTOR (GPU 1 / Fast Model)."""
        prompt = f"""You are an Objective Observer.
        
COMMAND/ACTION RUN:
{command_context}

RAW OUTPUT/LOGS:
{raw_output}

INSTRUCTIONS:
1. Objectively summarize the output.
2. HIGHLIGHT ALL ERRORS, WARNINGS, and FAILURES.
3. Note what was successfully accomplished.
4. Be concise but COMPLETE.

OUTPUT:
A text summary.
"""
        try:
            response = self.executor_client.chat.completions.create(
                model=self.executor_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error summarizing output: {e}."

    def analyze_interruption(self, current_objective: str, user_input: str) -> Dict[str, str]:
        """ROUTING: Uses the EXECUTOR (GPU 1 / Fast Model)."""
        prompt = f"""USER INTERRUPTION ANALYSIS
CURRENT OBJECTIVE: {current_objective}
USER INPUT: {user_input}

Determine intent: "NEW_TASK", "MODIFY_OBJECTIVE", or "ADVICE".
Return JSON: {{ "classification": "...", "reasoning": "...", "updated_text": "..." }}
"""
        try:
            response = self.executor_client.chat.completions.create(
                model=self.executor_model,
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

    def reason(self, prompt: str) -> str:
        """ROUTING: Uses the PLANNER (GPU 0 / Strong Model). Used by 'think' tool."""
        final_prompt = self._add_system_context(prompt)
        try:
            # No forced json for reason tool
            response = self.planner_client.chat.completions.create(
                model=self.planner_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.2,
                max_tokens=8192,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def summarize_text(self, text_to_summarize: str, query: str) -> str:
        """ROUTING: Uses the SUMMARIZER (Weak Model for Cloud / Executor for Local). Used by 'search' tool."""
        system_prompt = f"Summarize web search results relevant to: '{query}'"
        try:
            response = self.summarizer_client.chat.completions.create(
                model=self.summarizer_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_to_summarize}
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Could not summarize text: {e}"
