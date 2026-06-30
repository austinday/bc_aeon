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
    COMPRESS_MEMORIES_PROMPT,
)

# ANSI Colors for debug printing
C_YELLOW = '\033[93m'
C_RESET = '\033[0m'

class LLMClient:
    """A client for interacting with a LOCAL Large Language Model.

    One model powers everything: the main agent loop (reasoning + action
    selection) and all support tasks (summarization, prompt enhancement, etc.).

    Aeon is local-only: the client only ever talks to an on-machine inference
    server (Ollama / llama.cpp / vLLM). There is no cloud/API path, and no
    fallback to a different model -- if the configured model fails, the call
    raises so the failure is visible rather than silently degraded.
    """
    def __init__(self, config: dict):
        self.logger = get_logger()
        self.debug_path: Optional[pathlib.Path] = None
        self.current_iteration = 0

        if config is None:
            raise ValueError("config is required. Select a model at startup or provide --model.")

        self.provider = config['provider']
        self.client = self._create_client(config)
        self.model = config['model']            # catalog/display name: logging, llama.cpp self-heal lookup
        self.api_model = config.get('api_model') or self.model  # id sent to the server (vLLM served name)
        self.context_limit = config.get('context_limit', 128000)

        # Support tasks (skill routing, JSON repair/recovery, summarization,
        # log/memory compression, interruption analysis, prompt enhancement) run
        # on the same single local model as the main loop. There is no separate
        # utility tier and no fallback model.
        self.utility_client, self.utility_model = self.client, self.api_model

    def _create_client(self, config: dict):
        """Create an OpenAI-compatible client for a LOCAL inference server.

        Aeon is local-only: the only permitted providers are on-machine servers
        (Ollama / llama.cpp / vLLM). Any other provider -- e.g. a cloud/API model
        -- is rejected so prompts and context can never leave this machine.
        """
        provider = config['provider']
        if provider == 'local':
            return openai.OpenAI(base_url='http://localhost:8013/v1', api_key='ollama')
        elif provider in ['llamacpp', 'vllm']:
            return openai.OpenAI(base_url=config['base_url'], api_key='no-key-needed')
        raise ValueError(
            f"Unsupported provider '{provider}'. Aeon is local-only; only "
            "'local', 'llamacpp', and 'vllm' models are allowed (no cloud/API)."
        )

    def set_debug_path(self, path: pathlib.Path):
        self.debug_path = path

    def set_iteration(self, iteration: int):
        self.current_iteration = iteration

    def _log_to_debug(self, m_type, m_name, prompt, resp):
        """Legacy debug logger - removed to prevent log flooding."""
        pass

    def route_skills(self, objective: str) -> str:
        """Pre-flight skill router. Scans available skill protocols and returns a
        short '[SKILL ROUTING]' directive naming the best-matching skill (or none)
        for the given objective. Runs on the utility model so it adds minimal cost,
        and is fully best-effort: any failure returns '' so the agent proceeds
        exactly as before. Tools are deliberately NOT routed here -- they are
        managed by the collapsible-category system and the model already sees the
        top-level set every turn; the real gap is that skills get ignored.
        """
        try:
            from aeon.core.skills.manager import SkillsManager
            sm = SkillsManager()
            try:
                categories = [d.name for d in sm.base_dir.iterdir() if d.is_dir()]
            except Exception:
                return ""

            catalog = []
            for cat in sorted(categories):
                for skill in sorted(sm.get_skills_in_category(cat)):
                    content = sm.get_skill_content(cat, skill) or ""
                    # The first 1-2 comment lines of each protocol describe when it applies.
                    desc_lines = [ln.lstrip("# ").strip()
                                  for ln in content.splitlines()[:4]
                                  if ln.strip().startswith("#")]
                    desc = " ".join(desc_lines)[:240] if desc_lines else "(no description)"
                    catalog.append(f"- {cat}/{skill}: {desc}")

            if not catalog:
                return ""

            catalog_str = "\n".join(catalog)
            prompt = (
                "You are a skill router for an autonomous agent. Given the agent's task and a catalog of "
                "available skill protocols (reusable step-by-step procedures), decide whether ONE clearly "
                "applies. Be selective: recommend a skill ONLY if the task genuinely matches it. Trivial, "
                "conversational, or one-off tasks should get NONE.\n\n"
                f"TASK:\n{objective}\n\n"
                f"SKILL CATALOG:\n{catalog_str}\n\n"
                "Respond with ONLY a valid JSON object, no prose, no markdown fences:\n"
                '{\"skill\": \"<category>/<skill_name>\" or null, \"reason\": \"<one sentence>\"}'
            )
            resp = self.utility_client.chat.completions.create(
                model=self.utility_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            cleaned = self._clean_json_response(content)
            data = json.loads(cleaned)
            skill = data.get("skill")
            reason = data.get("reason", "")
            if not skill or str(skill).lower() == "null":
                return ""

            # Validate the routed skill actually exists before suggesting it.
            if "/" in str(skill):
                cat, _, name = str(skill).partition("/")
                if name not in sm.get_skills_in_category(cat):
                    return ""
            else:
                return ""

            return (f"[SKILL ROUTING] This task strongly matches the '{skill}' skill protocol "
                    f"({reason}). You should activate it with activate_skill('{skill}') as your first "
                    f"action, then follow its steps, unless it is clearly wrong for the actual task.")
        except Exception as e:
            self.logger.warning(f"Skill routing failed (continuing without it): {e}")
            return ""

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
            "content": "<<BLOCK_1>>\n#!/usr/bin/env python3\nimport os\n<<<END_BLOCK_1>>>"

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

    def _substitute_blocks(self, obj, blocks: dict, missing_blocks: list = None):
        """Recursively substitute __BLOCK_N__ placeholders in parsed JSON.

        Three-tier resolution:
        1. Exact placeholder (__BLOCK_N__ or <<BLOCK_N>>)  ->  substitute from blocks dict
        2. Inline-embedded delimiters (Qwen failure mode)   ->  extract from string value
        3. Neither                                          ->  leave unchanged
        """
        if missing_blocks is None:
            missing_blocks =[]

        if isinstance(obj, dict):
            return {k: self._substitute_blocks(v, blocks, missing_blocks) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_blocks(item, blocks, missing_blocks) for item in obj]
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
                else:
                    if key not in missing_blocks:
                        missing_blocks.append(key)
                    return obj  # Return placeholder untouched for now

            # --- Tier 2: Inline fallback ---
            # Only fires if the value has newlines and mentions BLOCK
            if '\n' in obj and 'BLOCK' in obj:
                extracted = self._extract_inline_content(obj)
                if extracted is not None:
                    return extracted

            return obj
        return obj

    def _recover_missing_block(self, missing_key: str, parsed_json: dict, original_prompt: str) -> Optional[str]:
        """Deploy a surgical LLM call to recover a specific missing code block."""
        intent = parsed_json.get('intent', 'Unknown intent')
        
        recovery_prompt = (
            f"{original_prompt}\n\n"
            f"=================================================\n"
            f"SYSTEM RECOVERY ALERT:\n"
            f"You previously decided on the following intent: '{intent}'.\n"
            f"However, you forgot to provide the code for {missing_key}.\n\n"
            f"Your ONLY task is to write the exact, raw code/text that belongs in {missing_key}.\n"
            f"DO NOT wrap it in JSON. DO NOT write a thought process. DO NOT write markdown fences.\n"
            f"Output ONLY the content that should replace the {missing_key} placeholder."
        )
        
        try:
            resp = self.utility_client.chat.completions.create(
                model=self.utility_model,
                messages=[{"role": "user", "content": recovery_prompt}],
                temperature=0.1
            )
            content = resp.choices[0].message.content.strip()
            
            if content.startswith("```") and content.endswith("```"):
                lines = content.split("\n")
                if len(lines) >= 3:
                    content = "\n".join(lines[1:-1])
                else:
                    content = content.strip("`")
                    
            self._log_to_debug("BLOCK_RECOVERY", self.utility_model, recovery_prompt, content)
            return content
        except Exception as e:
            self.logger.warning(f"Block recovery failed for {missing_key}: {e}")
            return None

    def _local_json_repair(self, raw_string: str) -> Optional[str]:
        """Deterministically fix the most common, low-risk JSON malformations
        WITHOUT an LLM round-trip: trailing commas before } or ], and Python
        literals (True/False/None) leaking in as bare words. Returns a valid
        JSON string if the repair parses, else None. Strings are respected so
        commas/words inside values are never touched.

        This is tried before the utility-model repair, turning the common case
        into a fast, free, deterministic fix.
        """
        if not raw_string:
            return None
        out = []
        in_string = False
        escape = False
        n = len(raw_string)
        literals = {'True': 'true', 'False': 'false', 'None': 'null'}
        i = 0
        while i < n:
            ch = raw_string[i]
            if escape:
                out.append(ch)
                escape = False
                i += 1
                continue
            if ch == '\\' and in_string:
                out.append(ch)
                escape = True
                i += 1
                continue
            if ch == '"':
                in_string = not in_string
                out.append(ch)
                i += 1
                continue
            if in_string:
                out.append(ch)
                i += 1
                continue
            # --- outside any string value below ---
            if ch == ',':
                # Drop a comma immediately followed (ignoring whitespace) by } or ]
                j = i + 1
                while j < n and raw_string[j] in ' \t\r\n':
                    j += 1
                if j < n and raw_string[j] in '}]':
                    i += 1  # skip the trailing comma
                    continue
            # Replace a bare Python literal (word-bounded) with its JSON form.
            if ch in ('T', 'F', 'N'):
                prev = raw_string[i - 1] if i > 0 else ''
                matched = False
                for word, repl in literals.items():
                    if raw_string.startswith(word, i):
                        nxt = raw_string[i + len(word)] if i + len(word) < n else ''
                        if not (prev.isalnum() or prev == '_') and not (nxt.isalnum() or nxt == '_'):
                            out.append(repl)
                            i += len(word)
                            matched = True
                            break
                if matched:
                    continue
            out.append(ch)
            i += 1
        candidate = ''.join(out)

        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            return None

    def _repair_json(self, raw_string: str, error_msg: str) -> Optional[str]:
        """Attempt to use the isolated utility model to fix malformed JSON."""
        prompt = (
            "You are a strict JSON repair parsing system. Your only job is to take a malformed JSON string and output valid JSON.\n"
            "Instructions:\n"
            "1. The user will provide a string that was supposed to be a JSON object containing an AI's action plan.\n"
            "2. The AI improperly escaped quotes or newlines inside a string value (usually inside 'content', 'patch', or 'command').\n"
            "3. Extract the keys and values and format them into perfectly escaped, valid JSON.\n"
            "4. DO NOT change any of the underlying code, intent, or logic. Only fix the JSON syntax.\n"
            "5. Output ONLY the valid JSON object. No markdown, no explanations.\n\n"
            "Malformed Input:\n"
            f"{raw_string}"
        )
        
        try:
            resp = self.utility_client.chat.completions.create(
                model=self.utility_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = resp.choices[0].message.content
            self._log_to_debug("JSON_REPAIR", self.utility_model, prompt, content)
            return self._clean_json_response(content)
        except Exception as e:
            self.logger.warning(f"JSON repair failed: {e}")
            return None

    def _handle_connection_error(self, error):
        """Handle API connection errors with exponential backoff and GPU recovery check."""
        self.logger.warning(f"Connection error detected: {error}. Entering recovery mode...")
        
        start_time = time.time()
        
        # Check if we are using a local model that we can self-heal
        llamacpp_config = None
        try:
            from aeon.main import get_llamacpp_config
            llamacpp_config = get_llamacpp_config(self.model)
        except ImportError:
            pass
        
        if llamacpp_config:
            self.logger.info(f"Local model {self.model} detected. Pausing for 5 minutes before attempting self-healing...")
            time.sleep(300)
            delay = 60
            max_delay = 600
        else:
            delay = 1
            max_delay = 60
            max_total_wait = 600
        
        while True:
            self.logger.info("Checking for GPU/Server recovery...")
            
            try:
                if llamacpp_config:
                    from aeon.main import start_llamacpp_server
                    from aeon.core.gpu_queue import get_real_vram
                    
                    self.logger.info(f"Preparing to self-heal {self.model}...")
                    
                    # Kill potentially hung containers first to free VRAM
                    containers_to_kill = [llamacpp_config['container_name']] + llamacpp_config.get('additional_containers', [])
                    for c_name in containers_to_kill:
                        subprocess.run(['docker', 'rm', '-f', c_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    time.sleep(5)  # Allow VRAM to free up
                    
                    vram = get_real_vram()
                    max_free = max(vram.values()) if vram else 0
                    
                    # We need enough VRAM (e.g., > 20GB) to reasonably start a model,
                    # or if `get_real_vram` fails (returns empty), we just try it anyway.
                    if not vram or max_free > 20.0:
                        self.logger.info(f"Sufficient VRAM detected (Max free: {max_free:.1f}GB). Running start script...")
                        success = start_llamacpp_server(llamacpp_config)
                        if success:
                            # Verify API works
                            self.client.models.list()
                            self.logger.info("Self-healing successful! Resuming agent...")
                            return True
                        else:
                            self.logger.warning("Self-heal script failed. GPU might still be occupied.")
                    else:
                        self.logger.info(f"Not enough VRAM to self-heal (Max free: {max_free:.1f}GB). Waiting...")
                else:
                    # OpenAI-compatible local server check: list models
                    self.client.models.list()
                    self.logger.info("Server recovery detected! Resuming agent...")
                    return True
            except Exception as e:
                self.logger.warning(f"Recovery check failed: {e}")
            
            if not llamacpp_config and (time.time() - start_time) > max_total_wait:
                self.logger.error("Recovery timed out after 10 minutes.")
                return False
                
            self.logger.info(f"Waiting {delay}s before next recovery attempt...")
            time.sleep(delay)
            delay = min(delay * 2, max_delay)

    def get_primary_agent_response(self, prompt: str, max_retries: int = 3, diagnostic_str: Optional[str] = None) -> str:
        """Get combined reasoning and action from the Primary Agent (Strong Model)."""
        current_prompt = prompt
        last_error = None

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                # Stream the response to accurately measure TTFT vs pure generation time
                resp_stream = self.client.chat.completions.create(
                    model=self.api_model,
                    messages=[{"role": "user", "content": current_prompt}],
                    temperature=0.2,
                    stream=True,
                    # Ask the server for a final usage chunk so we can report the
                    # model's REAL generated-token count, not a tiktoken estimate
                    # (cl100k mis-counts Gemma tokens, making t/s look far too low).
                    stream_options={"include_usage": True},
                )

                first_token_time = None
                raw_chunks =[]
                server_completion_tokens = None

                for chunk in resp_stream:
                    # The final usage-only chunk has empty choices; don't let it set TTFT.
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        if first_token_time is None:
                            first_token_time = time.time()
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            raw_chunks.append(delta.content)
                    usage = getattr(chunk, 'usage', None)
                    if usage is not None and getattr(usage, 'completion_tokens', None):
                        server_completion_tokens = usage.completion_tokens

                end_time = time.time()
                raw = "".join(raw_chunks)
                ttft = (first_token_time - start_time) if first_token_time else 0
                gen_time = (end_time - first_token_time) if first_token_time else 0
                # Prefer the server's real token count; fall back to the estimate.
                comp_tokens = server_completion_tokens or estimate_tokens(raw)

                tps = comp_tokens / gen_time if gen_time > 0 else 0
                print(f"\033[96m[Performance] {self.model} speed: {tps:.2f} t/s (TTFT: {ttft:.2f}s | {comp_tokens} tokens in {gen_time:.2f}s)\033[0m")

                if self.debug_path:
                    print(f"{C_YELLOW}[LLM RAW - PRIMARY AGENT]\n{raw}{C_RESET}")

                self._log_to_debug("PRIMARY_AGENT", self.model, current_prompt, raw)

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
                    missing_blocks =[]
                    parsed = self._substitute_blocks(parsed, blocks, missing_blocks)
                    
                    # --- TARGETED BLOCK RECOVERY ---
                    if missing_blocks:
                        if self.debug_path:
                            print(f"{C_YELLOW}[LLM] Missing blocks detected: {missing_blocks}. Initiating recovery...{C_RESET}")
                        
                        for mb in missing_blocks:
                            recovered_text = self._recover_missing_block(mb, parsed, current_prompt)
                            if recovered_text:
                                blocks[mb] = recovered_text
                            else:
                                raise ValueError(f"Failed to surgically recover missing {mb}.")
                        
                        # Run substitution one more time now that we have the blocks
                        missing_blocks.clear()
                        parsed = self._substitute_blocks(parsed, blocks, missing_blocks)
                        if missing_blocks:
                            raise ValueError(f"Still missing blocks after recovery: {missing_blocks}")

                    if blocks and self.debug_path:
                        print(f"{C_YELLOW}[LLM] Substituted {len(blocks)} content block(s){C_RESET}")

                    return json.dumps(parsed)
                except (json.JSONDecodeError, ValueError) as e:
                    last_error = f"JSON validation error: {str(e)}"
                    self.logger.warning(f"Primary Agent attempt {attempt + 1}/{max_retries} failed: {last_error}")

                    # --- ISOLATED FIXER AGENT INJECTION ---
                    is_decode_error = isinstance(e, json.JSONDecodeError) or "Expecting" in str(e) or "Unterminated" in str(e)
                    is_empty_error = "Empty JSON" in str(e)
                    
                    if is_decode_error and not is_empty_error:
                        # FAST PATH: try a deterministic local repair (trailing
                        # commas, Python literals) before spending a utility-model
                        # call. Handles the most common malformations for free.
                        local_fix = self._local_json_repair(json_str)
                        if local_fix:
                            try:
                                parsed = json.loads(local_fix)
                                if parsed and 'actions' in parsed:
                                    parsed = self._substitute_blocks(parsed, blocks)
                                    if self.debug_path:
                                        print(f"{C_YELLOW}[LLM] Local JSON repair succeeded (no model call).{C_RESET}")
                                    return json.dumps(parsed)
                            except (json.JSONDecodeError, ValueError):
                                pass

                        if self.debug_path:
                            print(f"{C_YELLOW}[LLM] Malformed JSON detected. Routing to Fixer Agent ({self.model})...{C_RESET}")

                        repaired_json_str = self._repair_json(json_str, str(e))
                        if repaired_json_str:
                            try:
                                parsed = json.loads(repaired_json_str)
                                if parsed and 'actions' in parsed:
                                    parsed = self._substitute_blocks(parsed, blocks)
                                    if self.debug_path:
                                        print(f"{C_YELLOW}[LLM] Fixer Agent successfully repaired the JSON.{C_RESET}")
                                    return json.dumps(parsed)
                            except (json.JSONDecodeError, ValueError) as repair_err:
                                self.logger.warning(f"Fixer Agent failed to produce valid JSON: {repair_err}")
                                if self.debug_path:
                                    print(f"{C_YELLOW}[LLM] Fixer Agent repair failed. Falling back to primary retry loop...{C_RESET}")
                    # --------------------------------------

                    if diagnostic_str:
                        print(f"\n{C_YELLOW}--- CONTEXT ROT DIAGNOSTIC ---{C_RESET}")
                        print(f"{C_YELLOW}JSON formatting error detected (Attempt {attempt + 1}). Breakdown of current context window:{C_RESET}")
                        print(f"{C_YELLOW}{diagnostic_str}{C_RESET}")
                        print(f"{C_YELLOW}------------------------------{C_RESET}\n")

                    if attempt < max_retries - 1:
                        current_prompt = prompt + f"\n\n** RETRY - YOUR PREVIOUS RESPONSE WAS INVALID **\nError: {last_error}\nRaw output started with: {raw[:300]}...\n\nYou MUST output a valid JSON object containing 'thought' and 'actions'. \nCRITICAL: JSON values must be static strings. Do not put Python operations (like '+' or '*') or complex escape characters inside the JSON. For multi-line or complex strings, use content blocks (--- BEGIN BLOCK_N --- ... --- END BLOCK_N ---)."

            except (openai.APIConnectionError, openai.InternalServerError, requests.exceptions.ConnectionError) as e:
                if self._handle_connection_error(e):
                    continue # Recovery successful, retry the request
                raise
            except Exception as e:
                self._log_to_debug("PRIMARY_AGENT_ERR", self.model, current_prompt, str(e))
                self.logger.error(f"Primary Agent LLM call failed: {e}")
                last_error = f"API Error: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
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

    def compress_memories(self, memories_text: str) -> Dict:
        """Compresses the persistent memories using the utility model and returns a dictionary."""
        prompt = COMPRESS_MEMORIES_PROMPT.format(memories=memories_text)
        try:
            resp = self.utility_client.chat.completions.create(
                model=self.utility_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            self._log_to_debug("COMPRESS_MEMORIES", self.utility_model, prompt, content)
            
            cleaned = self._clean_json_response(content)
            return json.loads(cleaned)
        except Exception as e:
            self.logger.warning(f"Memory compression failed: {e}")
            return {}

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
            resp = self.client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp.choices[0].message.content
            self._log_to_debug("REASONING (THINK TOOL)", self.model, prompt, content)
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
