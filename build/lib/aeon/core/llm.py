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
from .prompts import (
    SUMMARIZE_EXECUTION_PROMPT,
    ANALYZE_INTERRUPTION_PROMPT,
    SUMMARIZE_TEXT_PROMPT,
)

# ANSI Colors for debug printing
C_YELLOW = '\033[93m'
C_RESET = '\033[0m'

class LLMClient:
    """A client for interacting with Large Language Models (Cloud or Local).
    
    Uses two model tiers:
    - Primary (strong): Powers the main agent loop (reasoning + action selection).
    - Utility (weak): Powers summarization, interruption analysis, and other support tasks.
    """
    def __init__(self, strong_config: dict = None, weak_config: dict = None):
        self.logger = get_logger()
        self.debug_path: Optional[pathlib.Path] = None
        self.current_iteration = 0

        # Default configs for backward compatibility
        if strong_config is None:
            strong_config = {'model': 'qwen3:235b-iq4xs', 'provider': 'local', 'context_limit': 128000}
        if weak_config is None:
            weak_config = {'model': 'llama4:16x17b', 'provider': 'local', 'context_limit': 128000}

        # Primary model (strong) - used for the main agent loop
        self.primary_client = self._create_client(strong_config)
        self.primary_model = strong_config['model']

        # Utility model (weak) - used for summarization, analysis, etc.
        self.utility_client = self._create_client(weak_config)
        self.utility_model = weak_config['model']

        self.context_limit = min(
            strong_config.get('context_limit', 128000),
            weak_config.get('context_limit', 128000)
        )

    def _create_client(self, config: dict):
        """Create an OpenAI-compatible client from a model config dict."""
        if config['provider'] == 'local':
            return openai.OpenAI(base_url='http://localhost:8000/v1', api_key='ollama')
        else:
            api_key_path = pathlib.Path.home() / config['api_key_file']
            if not api_key_path.exists():
                raise FileNotFoundError(f'API key file not found: {api_key_path}')
            with open(api_key_path, 'r') as f:
                api_key = f.readline().strip()
            if not api_key:
                raise ValueError(f'API key file is empty: {api_key_path}')
            return openai.OpenAI(api_key=api_key, base_url=config['base_url'])

    def set_debug_path(self, path: pathlib.Path): 
        self.debug_path = path
        
    def set_iteration(self, iteration: int): 
        self.current_iteration = iteration

    def _log_to_debug(self, m_type, m_name, prompt, resp):
        """Log LLM interaction to debug file with high visibility."""
        if not self.debug_path: 
            return
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = (
                f"\n\n\n{'#'*100}\n"
                f"# TYPE:       {m_type}\n"
                f"# TIMESTAMP: {timestamp}\n"
                f"# ITERATION: {self.current_iteration}\n"
                f"# MODEL:      {m_name}\n"
                f"{'#'*100}\n"
            )
            
            # Format Prompt Section
            prompt_block = (
                f"\n{'>'*40} PROMPT {'>'*40}\n"
                f"{str(prompt)}\n"
                f"{'<'*40} END PROMPT {'<'*36}\n"
            )
            
            # Format Response Section
            response_block = (
                f"\n{'>'*40} RESPONSE {'>'*38}\n"
                f"{str(resp)}\n"
                f"{'<'*40} END RESPONSE {'<'*34}\n"
            )
            
            with open(self.debug_path, "a", encoding="utf-8") as f:
                f.write(header)
                f.write(prompt_block)
                f.write(response_block)
                f.write("\n" + "-"*100 + "\n") # Trailing separator
                
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
            if char == '\\' :
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

    def get_primary_agent_response(self, prompt: str, max_retries: int = 3) -> str:
        """Get combined reasoning and action from the Primary Agent (Strong Model)."""
        current_prompt = prompt
        last_error = None
        
        for attempt in range(max_retries):
            try:
                resp = self.primary_client.chat.completions.create(
                    model=self.primary_model, 
                    messages=[{"role": "user", "content": current_prompt}], 
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                raw = resp.choices[0].message.content
                
                if self.debug_path:
                    print(f"{C_YELLOW}[LLM RAW - PRIMARY AGENT]\n{raw}{C_RESET}")
                
                self._log_to_debug("PRIMARY_AGENT", self.primary_model, current_prompt, raw)
                
                cleaned = self._clean_json_response(raw)
                
                # Validate JSON parsing and Schema
                try:
                    parsed = json.loads(cleaned)
                    if not parsed:
                        raise ValueError("Empty JSON object returned.")
                    # Schema check: Must have 'actions'
                    if 'actions' not in parsed:
                         raise ValueError("JSON missing required 'actions' field.")
                    return cleaned
                except (json.JSONDecodeError, ValueError) as e:
                    last_error = f"JSON validation error: {str(e)}"
                    self.logger.warning(f"Primary Agent attempt {attempt + 1}/{max_retries} failed: {last_error}")
                    
                    if attempt < max_retries - 1:
                        current_prompt = prompt + f"\n\n** RETRY - YOUR PREVIOUS RESPONSE WAS INVALID **\nError: {last_error}\nRaw output started with: {raw[:300]}...\n\nYou MUST output ONLY a valid JSON object containing 'thought' and 'actions'."
                    
            except Exception as e:
                self._log_to_debug("PRIMARY_AGENT_ERR", self.primary_model, current_prompt, str(e))
                self.logger.error(f"Primary Agent LLM call failed: {e}")
                raise
        
        error_msg = f"Primary Agent failed after {max_retries} attempts. Last error: {last_error}"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _truncate_with_tail(self, text: str, head_len: int = 500, tail_len: int = 1000) -> str:
        """Truncate text keeping both head (context) and tail (errors)."""
        if len(text) <= (head_len + tail_len):
            return text
        return text[:head_len] + f"\n... [TRUNCATED {len(text) - (head_len + tail_len)} CHARS] ...\n" + text[-tail_len:]

    def summarize_execution(self, ctx, raw_out) -> str:
        """Summarize execution output for history."""
        safe_out = self._truncate_with_tail(raw_out, head_len=4000, tail_len=16000)
        prompt = SUMMARIZE_EXECUTION_PROMPT.format(ctx=ctx, safe_out=safe_out)
        try:
            resp = self.utility_client.chat.completions.create(
                model=self.utility_model, 
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp.choices[0].message.content
            self._log_to_debug("SUMMARIZE_EXECUTION", self.utility_model, prompt, content)
            return content
        except Exception as e:
            self.logger.warning(f"Summarize execution failed: {e}")
            tail_sample = raw_out[-1000:] if len(raw_out) > 1000 else raw_out
            return (f"!! SYSTEM ERROR: SUMMARIZATION FAILED !!\n"
                    f"Reason: {str(e)}\n"
                    f"Output Length: {len(raw_out)}\n"
                    f"--- RAW TAIL (Last 1000 chars) ---\n{tail_sample}")

    def analyze_interruption(self, obj, inp) -> Dict:
        """Analyze user interruption to classify intent."""
        prompt = ANALYZE_INTERRUPTION_PROMPT.format(obj=obj, inp=inp)
        try:
            resp = self.utility_client.chat.completions.create(
                model=self.utility_model, 
                messages=[{"role": "user", "content": prompt}], 
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            self._log_to_debug("ANALYZE_INTERRUPTION", self.utility_model, prompt, content)
            return json.loads(content)
        except Exception as e:
            self.logger.warning(f"Interruption analysis failed: {e}")
            return {"classification": "ADVICE", "updated_text": inp, "reasoning": "Failed to analyze"}

    def reason(self, prompt: str) -> str:
        """General reasoning/thinking call (uses primary/strong model)."""
        try:
            resp = self.primary_client.chat.completions.create(
                model=self.primary_model, 
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp.choices[0].message.content
            self._log_to_debug("REASONING (THINK TOOL)", self.primary_model, prompt, content)
            return content
        except Exception as e:
            self.logger.error(f"Reason call failed: {e}")
            return f"Error during reasoning: {e}"

    def summarize_text(self, text: str, query: str) -> str:
        """Summarize text in context of a query."""
        prompt = SUMMARIZE_TEXT_PROMPT.format(query=query, text=text)
        try:
            resp = self.utility_client.chat.completions.create(
                model=self.utility_model, 
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp.choices[0].message.content
            self._log_to_debug("SUMMARIZE_TEXT (WEB SEARCH)", self.utility_model, prompt, content)
            return content
        except Exception as e:
            self.logger.warning(f"Summarize text failed: {e}")
            return f"Failed to summarize: {e}"
