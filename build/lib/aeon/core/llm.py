import os
import openai
import pathlib
import sys
import json
import re
from datetime import datetime
from typing import Dict, Optional
sys.setrecursionlimit(2000)
from .system_info import get_runtime_info
from .logger import get_logger
from .utils import estimate_tokens

class LLMClient:
    """A client for interacting with Large Language Models (Cloud or Local)."""
    def __init__(self, provider: str = "local", local_strong: str = None, local_weak: str = None):
        self.provider = provider
        self.logger = get_logger()
        self.debug_path: Optional[pathlib.Path] = None
        self.current_iteration = 0
        
        # --- 1. LOCAL PROVIDER ---
        if provider == "local":
            # Single Brain Node on Port 8000 handles both models
            self.planner_client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="ollama")
            self.planner_model = local_strong or "deepseek-r1:70b"
            self.executor_client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="ollama")
            self.executor_model = local_weak or "qwen2.5:72b"
            self.summarizer_client = self.executor_client
            self.summarizer_model = self.executor_model
            self.context_limit = 128000 
        # --- 2. CLOUD PROVIDERS ---
        elif provider == "gemini":
            self.setup_cloud("gemini-3-pro-preview", "gemini-flash-latest", "gemini_api_key.txt", "https://generativelanguage.googleapis.com/v1beta/openai/")
            self.context_limit = 1000000
        else:
            self.setup_cloud("grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning", "grok_api_key.txt", "https://api.x.ai/v1")
            self.context_limit = 128000

    def setup_cloud(self, strong, weak, key_file, url):
        api_key_path = pathlib.Path.home() / key_file
        if not api_key_path.exists():
            raise FileNotFoundError(f"API key file not found: {api_key_path}")
        with open(api_key_path, 'r') as f: 
            api_key = f.readline().strip()
        if not api_key:
            raise ValueError(f"API key file is empty: {api_key_path}")
        self.planner_client = openai.OpenAI(api_key=api_key, base_url=url)
        self.executor_client = self.planner_client
        self.summarizer_client = self.planner_client
        self.planner_model = strong
        self.executor_model = strong
        self.summarizer_model = weak

    def set_debug_path(self, path: pathlib.Path): 
        self.debug_path = path
        
    def set_iteration(self, iteration: int): 
        self.current_iteration = iteration

    def _log_to_debug(self, m_type, m_name, prompt, resp):
        if not self.debug_path: 
            return
        try:
            with open(self.debug_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\nITER: {self.current_iteration} | {m_type} | {m_name}\n{'='*80}\nPROMPT:\n{prompt}\n{'-'*40}\nRESPONSE:\n{resp}\n")
        except Exception as e:
            self.logger.warning(f"Failed to write to debug log: {e}")

    def _clean_json_response(self, content: str) -> str:
        """Clean LLM response to extract JSON, handling common LLM formatting quirks."""
        if not content:
            return "{}"
        
        # Remove <think> tags and their content (including orphaned closing tags)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'</think>', '', content)
        content = re.sub(r'<think>', '', content)
        
        # Remove markdown code fences
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        content = content.strip()
        
        # Use brace matching to find the first complete JSON object
        # This handles text before/after the JSON
        brace_count = 0
        json_start = -1
        json_end = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                if json_start == -1:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    json_end = i + 1
                    break
        
        if json_start != -1 and json_end != -1:
            return content[json_start:json_end]
        
        # Fallback: try simple regex
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return match.group(0)
        
        self.logger.warning(f"No JSON object found in response: {content[:200]}...")
        return "{}"

    def get_plan(self, prompt: str, max_retries: int = 3) -> str:
        """Get plan from planner LLM with retry logic for JSON parsing errors."""
        current_prompt = prompt
        last_error = None
        
        for attempt in range(max_retries):
            try:
                resp = self.planner_client.chat.completions.create(
                    model=self.planner_model, 
                    messages=[{"role": "user", "content": current_prompt}], 
                    temperature=0.3
                )
                raw = resp.choices[0].message.content
                self._log_to_debug("PLANNER", self.planner_model, current_prompt, raw)
                
                cleaned = self._clean_json_response(raw)
                
                # Validate JSON parsing
                try:
                    json.loads(cleaned)
                    return cleaned
                except json.JSONDecodeError as e:
                    last_error = f"JSON parse error: {str(e)}"
                    self.logger.warning(f"Planner attempt {attempt + 1}/{max_retries} failed: {last_error}")
                    
                    if attempt < max_retries - 1:
                        # Add error feedback to prompt for retry
                        current_prompt = prompt + f"\n\n** RETRY - YOUR PREVIOUS RESPONSE HAD INVALID JSON **\nError: {last_error}\nRaw output started with: {raw[:300]}...\n\nYou MUST output ONLY a valid JSON object. No text before {{ or after }}. Use double quotes only."
                    
            except Exception as e:
                self._log_to_debug("PLANNER_ERR", self.planner_model, current_prompt, str(e))
                self.logger.error(f"Planner LLM call failed: {e}")
                raise
        
        # All retries exhausted, return empty but valid JSON
        self.logger.error(f"Planner failed after {max_retries} attempts. Last error: {last_error}")
        return '{}'

    def get_action(self, prompt: str, max_retries: int = 3) -> str:
        """Get action from executor LLM with retry logic for JSON parsing errors."""
        current_prompt = prompt
        last_error = None
        
        for attempt in range(max_retries):
            try:
                resp = self.executor_client.chat.completions.create(
                    model=self.executor_model, 
                    messages=[{"role": "user", "content": current_prompt}], 
                    temperature=0.1
                )
                raw = resp.choices[0].message.content
                self._log_to_debug("EXECUTOR", self.executor_model, current_prompt, raw)
                
                cleaned = self._clean_json_response(raw)
                
                # Validate JSON parsing
                try:
                    json.loads(cleaned)
                    return cleaned
                except json.JSONDecodeError as e:
                    last_error = f"JSON parse error: {str(e)}"
                    self.logger.warning(f"Executor attempt {attempt + 1}/{max_retries} failed: {last_error}")
                    
                    if attempt < max_retries - 1:
                        # Add error feedback to prompt for retry
                        current_prompt = prompt + f"\n\n** RETRY - YOUR PREVIOUS RESPONSE HAD INVALID JSON **\nError: {last_error}\nRaw output started with: {raw[:300]}...\n\nYou MUST output ONLY a valid JSON object. No text before {{ or after }}. Use double quotes only."
                    
            except Exception as e:
                self._log_to_debug("EXEC_ERR", self.executor_model, current_prompt, str(e))
                self.logger.error(f"Executor LLM call failed: {e}")
                raise
        
        # All retries exhausted, return empty but valid JSON
        self.logger.error(f"Executor failed after {max_retries} attempts. Last error: {last_error}")
        return '{"actions": []}'

    def analyze_milestones(self, analysis_context: str) -> Dict:
        """Analyze iteration results to identify completed milestones.
        Uses the summarizer model (weaker/faster) since this is a lightweight analysis task.
        """
        try:
            resp = self.summarizer_client.chat.completions.create(
                model=self.summarizer_model,
                messages=[{"role": "user", "content": analysis_context}],
                temperature=0.1
            )
            raw = resp.choices[0].message.content
            self._log_to_debug("MILESTONE_ANALYZER", self.summarizer_model, analysis_context, raw)
            
            # Parse JSON response
            clean_json = self._clean_json_response(raw)
            return json.loads(clean_json)
        except json.JSONDecodeError as e:
            self._log_to_debug("MILESTONE_PARSE_ERR", self.summarizer_model, analysis_context, str(e))
            self.logger.warning(f"Failed to parse milestone analysis JSON: {e}")
            return {}
        except Exception as e:
            self._log_to_debug("MILESTONE_ERR", self.summarizer_model, analysis_context, str(e))
            self.logger.warning(f"Milestone analysis failed: {e}")
            return {}

    def _truncate_with_tail(self, text: str, head_len: int = 500, tail_len: int = 1000) -> str:
        """Truncate text keeping both head (context) and tail (errors)."""
        if len(text) <= (head_len + tail_len):
            return text
        return text[:head_len] + f"\n... [TRUNCATED {len(text) - (head_len + tail_len)} CHARS] ...\n" + text[-tail_len:]

    def summarize_execution(self, ctx, raw_out) -> str:
        """Summarize execution output for history."""
        # Smart truncate before sending to LLM to prevent context overflow
        # Use a generous 20k window for the LLM context, biasing towards tail (error logs)
        safe_out = self._truncate_with_tail(raw_out, head_len=4000, tail_len=16000)
        
        prompt = f"Summarize this execution result. IF THE OUTPUT CONTAINS ERRORS, FAILURES, OR EXIT CODE 1, YOU MUST STATE THIS CLEARLY AT THE START. Context: {ctx}\nOutput: {safe_out}"
        try:
            resp = self.summarizer_client.chat.completions.create(
                model=self.summarizer_model, 
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e:
            self.logger.warning(f"Summarize execution failed: {e}")
            # LOUD FAILURE: Explicitly report the crash to the agent
            tail_sample = raw_out[-1000:] if len(raw_out) > 1000 else raw_out
            return (f"!! SYSTEM ERROR: SUMMARIZATION FAILED !!\n"
                    f"Reason: {str(e)}\n"
                    f"Output Length: {len(raw_out)}\n"
                    f"--- RAW TAIL (Last 1000 chars) ---\n{tail_sample}")

    def analyze_interruption(self, obj, inp) -> Dict:
        """Analyze user interruption to classify intent."""
        prompt = f"""Analyze this user interruption and classify their intent.
Current objective: {obj}
User input: {inp}

Respond with JSON: {{"classification": "NEW_TASK" | "MODIFY_OBJECTIVE" | "ADVICE", "updated_text": "...", "reasoning": "..."}}"""
        try:
            resp = self.executor_client.chat.completions.create(
                model=self.executor_model, 
                messages=[{"role": "user", "content": prompt}], 
                response_format={"type": "json_object"}
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            self.logger.warning(f"Interruption analysis failed: {e}")
            return {"classification": "ADVICE", "updated_text": inp, "reasoning": "Failed to analyze"}

    def reason(self, prompt: str) -> str:
        """General reasoning/thinking call."""
        try:
            resp = self.planner_client.chat.completions.create(
                model=self.planner_model, 
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Reason call failed: {e}")
            return f"Error during reasoning: {e}"

    def summarize_text(self, text: str, query: str) -> str:
        """Summarize text in context of a query."""
        prompt = f"Query: {query}\nText: {text}\n\nProvide a concise summary relevant to the query."
        try:
            resp = self.summarizer_client.chat.completions.create(
                model=self.summarizer_model, 
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e:
            self.logger.warning(f"Summarize text failed: {e}")
            return f"Failed to summarize: {e}"
