import os
import time
import openai
import pathlib
import sys
import json
import re
import subprocess
import requests
from datetime import datetime
from typing import Dict, Optional
sys.setrecursionlimit(2000)
from .system_info import get_runtime_info
from .logger import get_logger
from .utils import estimate_tokens
from .prompts import (
    COMPRESS_ACTION_LOG_PROMPT,
    ANALYZE_INTERRUPTION_PROMPT,
    SUMMARIZE_TEXT_PROMPT,
)

# ANSI Colors for debug printing
C_YELLOW = '\033[93m'
C_RESET = '\033[0m'

class VertexAIClient:
    def __init__(self, project_id, model_id):
        self.project_id = project_id
        self.model_id = model_id
        self.chat = self.Chat(self)
        self._cached_token = None
        self._token_expiry = 0

    def get_access_token(self):
        if self._cached_token and time.time() < self._token_expiry - 300:
            return self._cached_token
        try:
            token = subprocess.check_output(['gcloud', 'auth', 'print-access-token'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
            self._cached_token = token
            self._token_expiry = time.time() + 3600
            return token
        except subprocess.CalledProcessError:
            print(f'\n{C_YELLOW}Error: Failed to get access token via gcloud.{C_RESET}')
            print(f'{C_YELLOW}Please authenticate by running: gcloud auth login{C_RESET}')
            sys.exit(1)

    class Chat:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self.Completions(parent)

        class Completions:
            def __init__(self, parent):
                self.parent = parent

            def create(self, model, messages, temperature=0.7, response_format=None):
                contents = []
                system_prompt = None
                for msg in messages:
                    if msg['role'] == 'system':
                        system_prompt = msg['content']
                    else:
                        role = 'user' if msg['role'] == 'user' else 'model'
                        contents.append({'role': role, 'parts': [{'text': msg['content']}]})
                
                url = f'https://aiplatform.googleapis.com/v1/projects/{self.parent.project_id}/locations/global/publishers/google/models/{self.parent.model_id}:generateContent'
                headers = {
                    'Authorization': f'Bearer {self.parent.get_access_token()}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'contents': contents,
                    'generationConfig': {'temperature': temperature, 'maxOutputTokens': 8192},
                    'safetySettings': [
                        {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'OFF'},
                        {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'OFF'}
                    ]
                }
                if response_format and response_format.get('type') == 'json_object':
                    data['generationConfig']['responseMimeType'] = 'application/json'
                    
                if system_prompt:
                    data['systemInstruction'] = {
                        'parts': [{'text': system_prompt}]
                    }
                    
                response = requests.post(url, headers=headers, json=data)
                if response.status_code != 200:
                    raise Exception(f'Vertex AI API Error {response.status_code}: {response.text}')
                
                resp_json = response.json()
                try:
                    text = resp_json['candidates'][0]['content']['parts'][0]['text']
                except (KeyError, IndexError):
                    text = ''
                
                # Mock response object to match expected OpenAI schema
                class MockChoice:
                    def __init__(self, text):
                        self.message = type('MockMessage', (), {'content': text})()
                class MockResponse:
                    def __init__(self, text):
                        self.choices = [MockChoice(text)]
                        self.usage = type('MockUsage', (), {'completion_tokens': len(text)//4})()
                        
                return MockResponse(text)

class LLMClient:
    """A client for interacting with Large Language Models (Cloud or Local).

    Uses two model tiers:
    - Primary (strong): Powers the main agent loop (reasoning + action selection).
    - Utility (weak): Powers summarization, interruption analysis, and other support tasks.
    """
    def __init__(self, strong_config: dict, weak_config: dict):
        self.logger = get_logger()
        self.debug_path: Optional[pathlib.Path] = None
        self.current_iteration = 0

        if strong_config is None:
            raise ValueError("strong_config is required. Select a model at startup or provide --strong flag.")
        if weak_config is None:
            raise ValueError("weak_config is required. Select a model at startup or provide --weak flag.")

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
        elif config['provider'] == 'llamacpp':
            return openai.OpenAI(base_url=config['base_url'], api_key='no-key-needed')
        elif config['provider'] == 'vertex':
            return VertexAIClient(config['project_id'], config['model'])
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
            prompt_block = (
                f"\n{'>'*40} PROMPT {'>'*40}\n"
                f"{str(prompt)}\n"
                f"{'<'*40} END PROMPT {'<'*36}\n"
            )
            response_block = (
                f"\n{'>'*40} RESPONSE {'>'*38}\n"
                f"{str(resp)}\n"
                f"{'<'*40} END RESPONSE {'<'*34}\n"
            )
            with open(self.debug_path, "a", encoding="utf-8") as f:
                f.write(header)
                f.write(prompt_block)
                f.write(response_block)
                f.write("\n" + "-"*100 + "\n")
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

    def _find_json_end(self, raw: str) -> int:
        """Find the position right after the outermost JSON closing brace.

        Returns the index after '}', or -1 if no valid JSON object found.
        Properly handles strings and escape sequences so braces inside
        string values are not counted.
        """
        start = raw.find('{')
        if start == -1:
            return -1

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(raw)):
            ch = raw[i]

            if escape_next:
                escape_next = False
                continue

            if ch == '\\' and in_string:
                escape_next = True
                continue

            if ch == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i + 1

        return -1

    def _extract_content_blocks(self, raw: str, json_end: int) -> dict:
        """Extract content blocks from text AFTER the JSON object.

        Only searches raw[json_end:] so delimiters inside JSON string
        values are never matched.

        v2 format (preferred):
            --- BEGIN BLOCK_1 ---
            content here
            --- END BLOCK_1 ---

        v1 format (backward compatible):
            <<<BLOCK_1>>>
            content here
            <<<END_BLOCK_1>>>

        The v2 parser is flexible about dashes, spacing, and underscores.
        """
        blocks = {}
        remainder = raw[json_end:] if json_end > 0 else raw

        # v2: word-based delimiters (flexible about decoration)
        v2_pattern = (
            r'^[^\S\n]*-*\s*BEGIN[\s_]+BLOCK[\s_]*(\d+)\s*-*\s*$'
            r'\n?'
            r'(.*?)'
            r'^[^\S\n]*-*\s*END[\s_]+BLOCK[\s_]*\1\s*-*\s*$'
        )
        for match in re.finditer(v2_pattern, remainder, re.DOTALL | re.MULTILINE):
            block_id = match.group(1)
            content = match.group(2)
            if content.endswith('\n'):
                content = content[:-1]
            blocks[f'BLOCK_{block_id}'] = content

        # v1 fallback: angle brackets (flexible about 2-4 brackets)
        if not blocks:
            v1_pattern = r'<{2,4}(BLOCK_[A-Za-z0-9_]+)>{2,4}\n?(.*?)<{2,4}END_\1>{2,4}'
            for match in re.finditer(v1_pattern, remainder, re.DOTALL):
                block_id = match.group(1)
                content = match.group(2)
                if content.endswith('\n'):
                    content = content[:-1]
                blocks[block_id] = content

        return blocks

    def _extract_inline_content(self, value: str):
        """Fallback: extract content from a JSON string value with embedded delimiters.

        This handles the failure mode where the model puts delimiters AND content
        inside the JSON string instead of using the two-part system. For example:
            "content": "<<BLOCK_1>>\\n#!/usr/bin/env python3\\nimport os\\n<<<END_BLOCK_1>>>"

        Returns the extracted content, or None if no inline embedding detected.
        """
        # v2 inline: --- BEGIN BLOCK_N --- ... --- END BLOCK_N ---
        v2_inline = re.search(
            r'(?:^|\n)\s*-*\s*BEGIN[\s_]+BLOCK[\s_]*\d+\s*-*\s*\n'
            r'(.*?)'
            r'\n\s*-*\s*END[\s_]+BLOCK[\s_]*\d+\s*-*\s*(?:\n|$)',
            value, re.DOTALL
        )
        if v2_inline:
            return v2_inline.group(1)

        # v1 inline: <<BLOCK_N>> ... <<END_BLOCK_N>>
        v1_inline = re.search(
            r'<{2,4}BLOCK_\w+>{2,4}\n?(.*?)\n?<{2,4}END_BLOCK_\w+>{2,4}',
            value, re.DOTALL
        )
        if v1_inline:
            content = v1_inline.group(1)
            if content.endswith('\n'):
                content = content[:-1]
            return content

        # Placeholder-as-prefix: __BLOCK_1__\ncontent or <<BLOCK_1>>\ncontent
        tag_prefix = re.match(
            r'^[_<]{1,4}BLOCK[\s_]*\d+[_>]{1,4}\s*\n(.*)',
            value, re.DOTALL
        )
        if tag_prefix:
            content = tag_prefix.group(1)
            # Strip trailing v2 delimiter
            content = re.sub(
                r'\n\s*-*\s*END[\s_]+BLOCK[\s_]*\d+\s*-*\s*$', '', content)
            # Strip trailing v1 delimiter
            content = re.sub(
                r'\n\s*<{2,4}END_BLOCK_\w+>{2,4}\s*$', '', content)
            return content

        return None

    def _substitute_blocks(self, obj, blocks: dict):
        """Recursively substitute __BLOCK_N__ placeholders in parsed JSON.

        Three-tier resolution:
        1. Exact placeholder (__BLOCK_N__ or <<BLOCK_N>>)  ->  substitute from blocks dict
        2. Inline-embedded delimiters (Qwen failure mode)   ->  extract from string value
        3. Neither                                          ->  leave unchanged
        """
        if isinstance(obj, dict):
            return {k: self._substitute_blocks(v, blocks) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_blocks(item, blocks) for item in obj]
        elif isinstance(obj, str):
            stripped = obj.strip()

            # --- Tier 1: Exact placeholder match ---
            placeholder_match = re.match(
                r'^(?:__BLOCK[_\s]*(\d+)__|<{2,4}BLOCK[_\s]*(\d+)>{2,4})$',
                stripped
            )
            if placeholder_match:
                num = placeholder_match.group(1) or placeholder_match.group(2)
                key = f'BLOCK_{num}'
                if key in blocks:
                    return blocks[key]

            # --- Tier 2: Inline fallback ---
            # Only fires if the value has newlines and mentions BLOCK
            if '\n' in obj and 'BLOCK' in obj:
                extracted = self._extract_inline_content(obj)
                if extracted is not None:
                    return extracted

            return obj
        return obj

    def get_primary_agent_response(self, prompt: str, max_retries: int = 3) -> str:
        """Get combined reasoning and action from the Primary Agent (Strong Model)."""
        current_prompt = prompt
        last_error = None

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                resp = self.primary_client.chat.completions.create(
                    model=self.primary_model,
                    messages=[{"role": "user", "content": current_prompt}],
                    temperature=0.2,
                )
                elapsed = time.time() - start_time
                raw = resp.choices[0].message.content

                try:
                    comp_tokens = resp.usage.completion_tokens
                except AttributeError:
                    comp_tokens = estimate_tokens(raw)
                tps = comp_tokens / elapsed if elapsed > 0 else 0
                print(f"\033[96m[Performance] {self.primary_model} speed: {tps:.2f} t/s ({comp_tokens} tokens in {elapsed:.2f}s)\033[0m")

                if self.debug_path:
                    print(f"{C_YELLOW}[LLM RAW - PRIMARY AGENT]\n{raw}{C_RESET}")

                self._log_to_debug("PRIMARY_AGENT", self.primary_model, current_prompt, raw)

                # Step 1: Find where the JSON object ends
                json_end = self._find_json_end(raw)

                # Step 2: Extract content blocks from AFTER the JSON only
                blocks = self._extract_content_blocks(raw, json_end)

                # Step 3: Extract just the JSON portion
                json_str = raw[:json_end] if json_end > 0 else raw
                cleaned = self._clean_json_response(json_str)

                try:
                    parsed = json.loads(cleaned)
                    if not parsed:
                        raise ValueError("Empty JSON object returned.")
                    if 'actions' not in parsed:
                        raise ValueError("JSON missing required 'actions' field.")

                    # Step 4: Substitute content blocks into parsed JSON
                    parsed = self._substitute_blocks(parsed, blocks)
                    if blocks and self.debug_path:
                        print(f"{C_YELLOW}[LLM] Substituted {len(blocks)} content block(s){C_RESET}")

                    return json.dumps(parsed)
                except (json.JSONDecodeError, ValueError) as e:
                    last_error = f"JSON validation error: {str(e)}"
                    self.logger.warning(f"Primary Agent attempt {attempt + 1}/{max_retries} failed: {last_error}")

                    if attempt < max_retries - 1:
                        current_prompt = prompt + f"\n\n** RETRY - YOUR PREVIOUS RESPONSE WAS INVALID **\nError: {last_error}\nRaw output started with: {raw[:300]}...\n\nYou MUST output a valid JSON object containing 'thought' and 'actions'. \nCRITICAL: JSON values must be static strings. Do not put Python operations (like '+' or '*') or complex escape characters inside the JSON. For multi-line or complex strings, use content blocks (--- BEGIN BLOCK_N --- ... --- END BLOCK_N ---)."

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

    def compress_action_log(self, log_text: str) -> str:
        """Compress a long action log down to ~25% of its size using the utility model."""
        prompt = COMPRESS_ACTION_LOG_PROMPT.format(log=log_text)
        try:
            resp = self.utility_client.chat.completions.create(
                model=self.utility_model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp.choices[0].message.content
            self._log_to_debug("COMPRESS_ACTION_LOG", self.utility_model, prompt, content)
            return content
        except Exception as e:
            self.logger.warning(f"Action log compression failed: {e}")
            return log_text

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
            cleaned = self._clean_json_response(content)
            return json.loads(cleaned)
        except Exception as e:
            self.logger.warning(f"Interruption analysis failed: {e}")
            return {"classification": "ADVICE", "updated_text": inp, "reasoning": f"Failed to analyze: {e}"}

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
