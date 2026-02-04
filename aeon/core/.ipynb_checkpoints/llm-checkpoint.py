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
from .utils import estimate_tokens

class LLMClient:
    """A client for interacting with Large Language Models, organized by capability (Strong vs Weak)."""
    def __init__(self, strong_model: str = None, weak_model: str = None):
        self.logger = get_logger()
        
        # Defaults if not provided (Local Layout)
        self.strong_model = strong_model or "deepseek-r1:70b"
        self.weak_model = weak_model or "qwen2.5:72b"
        
        self.context_limit = 128000
        
        # --- CLIENT INITIALIZATION ---
        self.strong_client = self._init_client(self.strong_model, "strong")
        self.weak_client = self._init_client(self.weak_model, "weak")
        
        self._load_reminders()

    def _init_client(self, model_name: str, role: str):
        """Determines the correct client (Local/Cloud) based on the model name."""
        
        # 1. Google Gemini
        if "gemini" in model_name.lower():
            self.context_limit = 1000000
            api_key_path = pathlib.Path.home() / "gemini_api_key.txt"
            if not api_key_path.exists():
                raise ValueError(f"Gemini API key not found for {model_name}")
            with open(api_key_path, 'r') as f: api_key = f.read().strip()
            return openai.OpenAI(
                api_key=api_key, 
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            
        # 2. xAI Grok
        elif "grok" in model_name.lower():
            self.context_limit = 128000
            api_key_path = pathlib.Path.home() / "grok_api_key.txt"
            if not api_key_path.exists():
                raise ValueError(f"Grok API key not found for {model_name}")
            with open(api_key_path, 'r') as f: api_key = f.read().strip()
            return openai.OpenAI(
                api_key=api_key, 
                base_url="https://api.x.ai/v1"
            )
            
        # 3. Local (Ollama)
        else:
            # Heuristic: Strong/Reasoning models -> GPU 0 (Port 8000)
            #            Weak/Fast models -> GPU 1 (Port 8001)
            # This preserves the split-GPU architecture while allowing flag overrides.
            
            base_url = "http://localhost:8000/v1" if role == "strong" else "http://localhost:8001/v1"
            self.logger.info(f"Mapping {role} model '{model_name}' to {base_url}")
            
            return openai.OpenAI(
                base_url=base_url,
                api_key="ollama"
            )

    def _load_reminders(self):
        self.important_reminders_path = pathlib.Path(__file__).parent / "prompts" / "important_reminders.txt"
        self.important_reminders = ""
        if self.important_reminders_path.is_file():
            with open(self.important_reminders_path, 'r') as f:
                self.important_reminders = f.read().strip()

    def _add_system_context(self, prompt: str) -> str:
        runtime_info = get_runtime_info()
        return f"{runtime_info}\n\nUser Prompt:\n{prompt}"

    def _check_context_size(self, text: str, model_name: str):
        """Guardrail: Hard stop if context overflows (Local Only)."""
        count = estimate_tokens(text)
        if count > self.context_limit:
            if "gemini" not in model_name and "grok" not in model_name:
                # STRICT FAIL-FAST FOR LOCAL
                msg = (f"CONTEXT OVERFLOW: Input size {count} exceeds limit {self.context_limit}. "
                       "Aborting to prevent silent truncation. "
                       "Reduce file tree size or history.")
                C_RED = '\033[91m'
                C_RESET = '\033[0m'
                print(f"\n{C_RED}!!! CRITICAL: {msg} !!!{C_RESET}\n")
                raise ValueError(msg)
            else:
                # WARNING ONLY FOR CLOUD
                C_YELLOW = '\033[93m'
                C_RESET = '\033[0m'
                print(f"\n{C_YELLOW}!!! WARNING: Context Overflow ({count} tokens) !!!{C_RESET}")

    def _clean_json_response(self, content: str) -> str:
        """Extracts JSON from text, handling <think> blocks from R1."""
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
        """ROUTING: Uses the STRONG Model (Planner/Reasoning)."""
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
        self._check_context_size(prompt, self.strong_model)
        self.logger.info(f'Planning with STRONG model: {self.strong_model}')
        try:
            # Local/R1 models crash with response_format=json_object, so we conditionalize
            is_json_mode = "gemini" in self.strong_model or "grok" in self.strong_model
            
            kwargs = {
                "model": self.strong_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 8192
            }
            
            if is_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.strong_client.chat.completions.create(**kwargs)
            raw_content = response.choices[0].message.content
            
            if not is_json_mode:
                return self._clean_json_response(raw_content)
            else:
                return raw_content
            
        except Exception as e:
            self.logger.error(f"Planner Error: {e}")
            return json.dumps({
                "thought_process": f"Error in planning: {e}",
                "updated_plan": current_plan,
                "next_step_suggestion": "Retry previous action or check logs."
            })

    def get_action(self, system_context: str, plan: str, suggestion: str, open_files_context: str) -> str:
        """ROUTING: Uses the WEAK Model (Executor/Fast)."""
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
        self._check_context_size(prompt, self.weak_model)
        self.logger.info(f'Executing with WEAK model: {self.weak_model}')
        try:
            response = self.weak_client.chat.completions.create(
                model=self.weak_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4000
            )
            raw_content = response.choices[0].message.content
            
            # Clean local responses
            if "gemini" not in self.weak_model and "grok" not in self.weak_model:
                return self._clean_json_response(raw_content)
            else:
                return raw_content
                
        except Exception as e:
            self.logger.error(f"Action Error: {e}")
            return ""

    def summarize_execution(self, command_context: str, raw_output: str) -> str:
        """ROUTING: Uses the WEAK Model (Summarizer)."""
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
        self._check_context_size(prompt, self.weak_model)
        try:
            response = self.weak_client.chat.completions.create(
                model=self.weak_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error summarizing output: {e}."

    def analyze_interruption(self, current_objective: str, user_input: str) -> Dict[str, str]:
        """ROUTING: Uses the WEAK Model (Fast)."""
        prompt = f"""USER INTERRUPTION ANALYSIS
CURRENT OBJECTIVE: {current_objective}
USER INPUT: {user_input}

Determine intent: "NEW_TASK", "MODIFY_OBJECTIVE", or "ADVICE".
Return JSON: {{ "classification": "...", "reasoning": "...", "updated_text": "..." }}
"""
        try:
            # Use json mode for weak model if possible
            response_format = {"type": "json_object"} if "gemini" in self.weak_model or "grok" in self.weak_model else None
            
            response = self.weak_client.chat.completions.create(
                model=self.weak_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                response_format=response_format
            )
            content = response.choices[0].message.content
            if not response_format:
                content = self._clean_json_response(content)
            return json.loads(content)
        except Exception as e:
            return {
                "classification": "ADVICE",
                "reasoning": f"Error parsing intent: {e}",
                "updated_text": user_input
            }

    def reason(self, prompt: str) -> str:
        """ROUTING: Uses the STRONG Model (Reasoning). Used by 'think' tool."""
        final_prompt = self._add_system_context(prompt)
        self._check_context_size(final_prompt, self.strong_model)
        try:
            response = self.strong_client.chat.completions.create(
                model=self.strong_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.2,
                max_tokens=8192,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def summarize_text(self, text_to_summarize: str, query: str) -> str:
        """ROUTING: Uses the WEAK Model."""
        system_prompt = f"Summarize web search results relevant to: '{query}'"
        self._check_context_size(text_to_summarize + system_prompt, self.weak_model)
        try:
            response = self.weak_client.chat.completions.create(
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
            return f"Could not summarize text: {e}"
