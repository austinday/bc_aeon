import os
import io
import base64
import time
import openai
import pathlib
import sys
import json
import re
import subprocess
import requests
from datetime import datetime
from typing import Dict, List, Optional
sys.setrecursionlimit(2000)
from .system_info import get_runtime_info
from .logger import get_logger
from .utils import estimate_tokens
from .prompts import (
    COMPRESS_ACTION_LOG_PROMPT,
    ANALYZE_INTERRUPTION_PROMPT,
    INTEGRATE_RESUME_PROMPT,
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

        # --- STRUCTURED OUTPUTS (grammar-constrained decoding) ---
        # The worker hands us the turn schema (aeon.core.action_schema) once its
        # tools are registered. When set, the primary-agent call asks the server
        # to CONSTRAIN generation to that schema (vLLM/xgrammar masks invalid
        # tokens at the sampler), so malformed JSON and hallucinated tool names
        # cannot be generated at all. _structured_mode tracks which request
        # style this server accepts and degrades gracefully:
        #   'response_format' (OpenAI-standard json_schema; vLLM >= 0.9, newer
        #                      llama.cpp/Ollama) -> 'guided_json' (vLLM-native
        #   extra_body, older servers) -> 'legacy' (unconstrained + the parse/
        #   repair cascade below, exactly the old behavior).
        self.action_schema: Optional[Dict] = None
        self._structured_mode: Optional[str] = None  # None = unprobed

    def _create_client(self, config: dict):
        """Create an OpenAI-compatible client for a LOCAL inference server.

        Aeon is local-only: the only permitted providers are on-machine servers
        (Ollama / llama.cpp / vLLM). Any other provider -- e.g. a cloud/API model
        -- is rejected so prompts and context can never leave this machine.
        """
        provider = config['provider']
        if provider == 'local':
            # The Ollama brain container maps host port 8000 -> 11434 (see
            # scripts/start_brain.sh) and serves the OpenAI-compatible API under
            # /v1. Port 8013 is the llama.cpp/vLLM load balancer — pointing an
            # Ollama model there sends its chats to the wrong server entirely.
            return openai.OpenAI(base_url='http://localhost:8000/v1', api_key='ollama')
        elif provider in ['llamacpp', 'vllm']:
            return openai.OpenAI(base_url=config['base_url'], api_key='no-key-needed')
        raise ValueError(
            f"Unsupported provider '{provider}'. Aeon is local-only; only "
            "'local', 'llamacpp', and 'vllm' models are allowed (no cloud/API)."
        )

    def set_debug_path(self, path: pathlib.Path):
        self.debug_path = path

    def set_action_schema(self, schema: Optional[Dict]):
        """Install (or clear) the turn schema used for grammar-constrained
        decoding of primary-agent responses. Called by Worker.register_tools so
        the 'tool_name' enum always matches the actually-registered tools."""
        self.action_schema = schema
        # Re-probe on a schema change only if we had given up: a previously
        # working mode keeps working with a new schema.
        if self._structured_mode == "legacy":
            self._structured_mode = None

    def _structured_request_kwargs(self) -> Optional[Dict]:
        """Extra kwargs for chat.completions.create that constrain generation
        to self.action_schema, per the currently-trusted request style.
        Returns None when structured decoding is unavailable (no schema, or the
        server rejected both styles) — callers then use the legacy parse path."""
        if not self.action_schema or self._structured_mode == "legacy":
            return None
        if self._structured_mode == "guided_json":
            return {"extra_body": {"guided_json": self.action_schema,
                                   "repetition_penalty": 1.05}}
        # Default / 'response_format': the OpenAI-standard structured-outputs
        # request. vLLM 0.9+ (xgrammar), newer llama.cpp and Ollama accept this.
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "aeon_turn", "strict": True,
                                "schema": self.action_schema},
            },
            "extra_body": {"repetition_penalty": 1.05},
        }

    def _downgrade_structured_mode(self, err: Exception) -> bool:
        """After a BadRequest that names the structured-output machinery,
        step down one tier and report True (caller should retry the call).
        Returns False when the error is unrelated to structured outputs."""
        msg = str(err).lower()
        if not any(k in msg for k in ("response_format", "json_schema", "guided",
                                      "structured", "grammar", "schema")):
            return False
        if self._structured_mode in (None, "response_format"):
            self._structured_mode = "guided_json"
            self.logger.warning(
                "Server rejected response_format json_schema; retrying with "
                "vLLM-native guided_json.")
            return True
        if self._structured_mode == "guided_json":
            self._structured_mode = "legacy"
            self.logger.warning(
                "Server rejected guided_json too; falling back to legacy "
                "unconstrained decoding + parse/repair for this session.")
            return True
        return False

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

        # FIRST: a quick reachability probe. A single dropped connection or a
        # momentarily-busy server must NOT trigger the heavy self-heal below,
        # which force-removes the model containers (and used to do so after a
        # blind 5-minute sleep) even when the server was actually fine.
        for probe_delay in (2, 5, 10):
            time.sleep(probe_delay)
            try:
                self.client.models.list()
                self.logger.info("Server is reachable again (transient error). Resuming agent...")
                return True
            except Exception:
                pass

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

    # Longest-side cap for a screenshot handed to the model. Set to the real
    # browser viewport (1920x1080) so the page is NOT downscaled — the model reads
    # exactly the pixels a human would. Gemma-4 pan-and-scans within this bound.
    VISION_MAX_DIM = 1920

    def _encode_image_data_url(self, image_path: str) -> Optional[str]:
        """Return a JPEG data: URL for an OpenAI-style multimodal message, or None
        (never raises) if the file is missing/undecodable or PIL is absent, so a
        screenshot problem degrades to a text-only turn instead of crashing.

        Fast path: a browser screenshot is ALREADY a right-sized JPEG, so we base64
        its original bytes — no resize, no second lossy re-encode, less latency.
        Only oversized or non-JPEG inputs are decoded, downscaled, and re-encoded."""
        try:
            if not image_path or not os.path.exists(image_path):
                return None
            from PIL import Image  # lazy: PIL ships with the vision/browser stack
            with Image.open(image_path) as img:
                fmt = (img.format or "").upper()
                w, h = img.size  # available from open() without a full decode
                if fmt in ("JPEG", "JPG") and max(w, h) <= self.VISION_MAX_DIM:
                    with open(image_path, "rb") as f:
                        raw = f.read()
                    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("utf-8")
                img.load()
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                if max(w, h) > self.VISION_MAX_DIM:
                    scale = self.VISION_MAX_DIM / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            self.logger.warning(f"Could not encode screenshot {image_path} for vision: {e}")
            return None

    def _content_with_images(self, text: str, image_urls: List[str]):
        """Assemble the chat 'content' from a text part and PRE-ENCODED image data
        URLs: a plain string when there are none, else a multimodal [text, image...]
        list so the model SEES the page directly alongside its full text context."""
        if not image_urls:
            return text
        parts = [{"type": "text", "text": text}]
        for url in image_urls:
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts if len(parts) > 1 else text

    @staticmethod
    def _msg_text(message: Dict) -> str:
        """Text of a chat message whose content may be a plain string or a
        multimodal [text, image...] parts list."""
        c = message.get("content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c
                            if isinstance(p, dict) and p.get("type") == "text")
        return str(c)

    def _build_user_content(self, text: str, images: Optional[List[str]]):
        """Encode image FILE PATHS and build the user content (encodes each path).
        Used by direct callers/tests; the main loop pre-encodes once and calls
        _content_with_images to avoid re-encoding across retries."""
        urls = [self._encode_image_data_url(p) for p in (images or [])]
        return self._content_with_images(text, [u for u in urls if u])

    def get_primary_agent_response(self, prompt: Optional[str] = None, max_retries: int = 3,
                                   diagnostic_str: Optional[str] = None,
                                   images: Optional[List[str]] = None,
                                   messages: Optional[List[Dict]] = None) -> str:
        """Get combined reasoning and action from the Primary Agent (Strong Model).

        When ``images`` (file paths) are supplied — e.g. the current browser
        screenshot — they are attached to the user turn as a multimodal message so
        the deciding model looks at the rendered page itself, not a text summary of
        it. Requires a multimodal model (Gemma-4); a text-only model simply ignores
        the image parts.

        When ``messages`` is given (message-history mode) it is used as the chat
        message list: the LAST message is the current-turn user message (images and
        any retry note attach to it), and the earlier messages (system + prior
        turns) form the stable, cache-friendly prefix. Otherwise a single user
        message is built from ``prompt`` (the default, unchanged behavior)."""
        # Stable prefix (system + history) vs the current-turn user text. A retry
        # note is applied to the user text only, so the prefix stays byte-identical
        # across attempts (and across turns — that is the whole point of caching).
        if messages:
            prefix_messages = [dict(m) for m in messages[:-1]]
            _last = messages[-1]
            base_user_text = _last.get("content", "") if isinstance(_last.get("content"), str) else (prompt or "")
        else:
            prefix_messages = []
            base_user_text = prompt or ""
        retry_suffix = ""
        full_prompt_text = base_user_text
        last_error = None
        # Encode attached screenshots ONCE, not once per retry attempt. If this
        # model has already told us it can't accept images (a text-only build),
        # don't even try — degrade to text-only instead of failing every turn.
        if getattr(self, "_vision_supported", True):
            image_urls = [self._encode_image_data_url(p) for p in (images or [])]
            image_urls = [u for u in image_urls if u]
        else:
            image_urls = []

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                # Assemble this attempt's messages: stable prefix + the current user
                # message (base text + optional retry note + images). full_prompt_text
                # is the concatenation used for token calibration and debug logging.
                user_text = base_user_text + retry_suffix
                req_messages = prefix_messages + [
                    {"role": "user", "content": self._content_with_images(user_text, image_urls)}]
                full_prompt_text = "\n".join(self._msg_text(m) for m in req_messages)

                # Grammar-constrained decoding: when the worker installed a turn
                # schema, the server's sampler is constrained to it (vLLM/xgrammar
                # masks invalid tokens), so the response is GUARANTEED to be a
                # single schema-valid JSON object — the parse/repair cascade below
                # becomes a dead path. Degrades per _downgrade_structured_mode if
                # this server can't do it.
                structured_kwargs = self._structured_request_kwargs()
                sampling_kwargs = dict(structured_kwargs or
                                       {"extra_body": {"repetition_penalty": 1.05}})
                # NOTE: no frequency_penalty here, deliberately. It accumulates on
                # repeated tokens — and JSON's structural tokens ('"', ',', '}')
                # are the most-repeated tokens in a long response. Production logs
                # showed "Expecting ',' delimiter" failures clustered deep in the
                # output (char 400-3200), exactly where the accumulated penalty
                # starts suppressing delimiters. The mild flat repetition_penalty
                # (1.05, vLLM extra_body) keeps the anti-runaway nudge without the
                # compounding structural damage; max_tokens is the hard backstop.

                # Stream the response to accurately measure TTFT vs pure generation time
                resp_stream = self.client.chat.completions.create(
                    model=self.api_model,
                    messages=req_messages,
                    temperature=0.2,
                    stream=True,
                    # Hard ceiling on one turn's output. Without this a low-temp model
                    # that hits a confusing input (e.g. an unparseable CAPTCHA frame)
                    # can enter a repetition loop and emit tens of thousands of tokens
                    # — a real incident here was 85k tokens / 11 min in a single turn.
                    # A normal turn (thought + actions, or a file write inside a JSON
                    # string) is well under this; the cap only bites a runaway.
                    max_tokens=16384,
                    # Ask the server for a final usage chunk so we can report the
                    # model's REAL generated-token count, not a tiktoken estimate
                    # (cl100k mis-counts Gemma tokens, making t/s look far too low).
                    stream_options={"include_usage": True},
                    **sampling_kwargs,
                )

                first_token_time = None
                raw_chunks =[]
                server_completion_tokens = None
                server_prompt_tokens = None
                server_cached_tokens = None
                finish_reason = None

                for chunk in resp_stream:
                    # The final usage-only chunk has empty choices; don't let it set TTFT.
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        if first_token_time is None:
                            first_token_time = time.time()
                        choice = chunk.choices[0]
                        delta = choice.delta
                        if hasattr(delta, 'content') and delta.content:
                            raw_chunks.append(delta.content)
                        if getattr(choice, 'finish_reason', None):
                            finish_reason = choice.finish_reason
                    usage = getattr(chunk, 'usage', None)
                    if usage is not None and getattr(usage, 'completion_tokens', None):
                        server_completion_tokens = usage.completion_tokens
                    if usage is not None and getattr(usage, 'prompt_tokens', None):
                        server_prompt_tokens = usage.prompt_tokens
                    # Prefix-cache hit count (vLLM reports it in prompt_tokens_details
                    # when --enable-prefix-caching is on). Surfaced below so the
                    # cache-friendly prompt ordering is visible per turn.
                    if usage is not None:
                        ptd = getattr(usage, 'prompt_tokens_details', None)
                        cached = getattr(ptd, 'cached_tokens', None) if ptd is not None else None
                        if cached is None and isinstance(ptd, dict):
                            cached = ptd.get('cached_tokens')
                        if cached is not None:
                            server_cached_tokens = cached

                # Calibrate estimate_tokens against the server's REAL prompt token
                # count (free — it's already in the usage chunk), so the worker's
                # context-pressure math tracks the served model's tokenizer, not
                # cl100k. Text-only turns only: image tokens would inflate the ratio.
                if server_prompt_tokens and not image_urls:
                    try:
                        from .utils.tokens import calibrate
                        calibrate(full_prompt_text, server_prompt_tokens)
                    except Exception:
                        pass

                end_time = time.time()
                raw = "".join(raw_chunks)
                ttft = (first_token_time - start_time) if first_token_time else 0
                gen_time = (end_time - first_token_time) if first_token_time else 0
                # Prefer the server's real token count; fall back to the estimate.
                comp_tokens = server_completion_tokens or estimate_tokens(raw)

                tps = comp_tokens / gen_time if gen_time > 0 else 0
                # Prompt / prefix-cache readout: high 'cached' across turns means the
                # static prompt prefix is being reused (low TTFT). A low value turn
                # after turn signals the ordering is being busted by volatile content.
                if server_prompt_tokens:
                    if server_cached_tokens is not None:
                        pct = 100.0 * server_cached_tokens / max(1, server_prompt_tokens)
                        prompt_str = f" | prompt {server_prompt_tokens} ({pct:.0f}% cached)"
                    else:
                        prompt_str = f" | prompt {server_prompt_tokens}"
                else:
                    prompt_str = ""
                print(f"\033[96m[Performance] {self.model} speed: {tps:.2f} t/s (TTFT: {ttft:.2f}s | {comp_tokens} tokens in {gen_time:.2f}s{prompt_str})\033[0m")

                if self.debug_path:
                    print(f"{C_YELLOW}[LLM RAW - PRIMARY AGENT]\n{raw}{C_RESET}")

                self._log_to_debug("PRIMARY_AGENT", self.model, full_prompt_text, raw)

                # A response cut off at max_tokens can't be a complete JSON object
                # (grammar-constrained or not). Retry with a terseness note rather
                # than feeding a guaranteed-broken string to the parser.
                if finish_reason == "length":
                    last_error = ("Response truncated at the max_tokens ceiling "
                                  "(finish_reason=length) — incomplete JSON.")
                    self.logger.warning(
                        f"Primary Agent attempt {attempt + 1}/{max_retries}: {last_error}")
                    if attempt < max_retries - 1:
                        retry_suffix = (
                            "\n\n** RETRY - YOUR PREVIOUS RESPONSE WAS CUT OFF (too long) **\n"
                            "Your response exceeded the output limit and was truncated. Be BRIEF: "
                            "shorten 'thought' to a few sentences, and if you were writing a large "
                            "file, write a smaller piece of it this turn (or split the work across "
                            "multiple str_replace/write_file turns).")
                        continue
                    break

                # --- STRUCTURED FAST PATH ---
                # Grammar-constrained output IS the JSON object — parse directly.
                # Any failure here is unexpected (a server-side gap, not a model
                # mistake), so log it loudly and fall through to the tolerant
                # legacy pipeline below rather than crashing the turn.
                if structured_kwargs is not None:
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict) and parsed.get('actions') is not None:
                            return json.dumps(parsed)
                        self.logger.warning(
                            "Structured output parsed but missing 'actions'; "
                            "falling through to legacy parsing this turn.")
                    except (json.JSONDecodeError, ValueError) as se:
                        self.logger.warning(
                            f"Structured output was not clean JSON ({se}); "
                            f"falling through to legacy parsing this turn.")

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

                    # updated_plan is optional and must NOT depend on the block system.
                    # If a model block-encoded it and the block is missing, DROP the plan
                    # (the worker keeps the prior one) instead of firing the recovery
                    # reprompt — that reprompt was observed to derail the model into
                    # meta-reasoning about blocks instead of doing the task.
                    up = parsed.get('updated_plan')
                    if isinstance(up, str):
                        m = re.match(r'^\s*(?:__BLOCK[_\s]*(\d+)__|<{2,4}BLOCK[_\s]*(\d+)>{2,4})\s*$', up)
                        if m:
                            num = m.group(1) or m.group(2)
                            parsed['updated_plan'] = ""
                            # Only stop recovering this block if nothing else references it.
                            key = f'BLOCK_{num}'
                            if key in missing_blocks and f'__BLOCK_{num}__' not in json.dumps(
                                    parsed.get('actions', [])):
                                missing_blocks.remove(key)

                    # --- TARGETED BLOCK RECOVERY ---
                    if missing_blocks:
                        if self.debug_path:
                            print(f"{C_YELLOW}[LLM] Missing blocks detected: {missing_blocks}. Initiating recovery...{C_RESET}")
                        
                        for mb in missing_blocks:
                            recovered_text = self._recover_missing_block(mb, parsed, full_prompt_text)
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
                        retry_suffix = f"\n\n** RETRY - YOUR PREVIOUS RESPONSE WAS INVALID **\nError: {last_error}\nRaw output started with: {raw[:300]}...\n\nYou MUST output exactly ONE valid JSON object containing 'thought' and 'actions', and nothing else.\nCRITICAL: JSON values must be static strings with standard JSON escaping — newlines as \\n, quotes as \\\", backslashes as \\\\. No Python operations (like '+' or '*'), no markdown fences, no text before or after the JSON."

            except (openai.APIConnectionError, openai.InternalServerError, requests.exceptions.ConnectionError) as e:
                if self._handle_connection_error(e):
                    continue # Recovery successful, retry the request
                raise
            except openai.BadRequestError as e:
                # The server rejected the request. First: if the rejection names
                # the structured-output machinery (response_format/guided/schema),
                # step down one tier (response_format -> guided_json -> legacy)
                # and retry THIS attempt — an older server simply doesn't speak
                # the newer request style; the turn itself is fine.
                if self._downgrade_structured_mode(e):
                    continue
                # If it was because THIS model can't accept images (a text-only
                # build served where a multimodal one was expected), degrade
                # gracefully: stop sending screenshots for the rest of the session
                # and retry text-only THIS turn, rather than crashing every
                # browser turn with a 400.
                msg = str(e).lower()
                if image_urls and ("multimodal" in msg or "image" in msg):
                    self.logger.warning("Model rejected image input; falling back to text-only for this session.")
                    print(f"{C_YELLOW}[LLM] The served model is NOT multimodal — it cannot see screenshots. "
                          f"Falling back to text-only browsing (element list) for the rest of this session. "
                          f"To use vision, serve a multimodal Gemma-4 (Gemma4ForConditionalGeneration).{C_RESET}")
                    self._vision_supported = False
                    image_urls = []
                    continue  # retry this attempt without the image
                self._log_to_debug("PRIMARY_AGENT_ERR", self.model, full_prompt_text, str(e))
                self.logger.error(f"Primary Agent bad request: {e}")
                last_error = f"API Error: {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise
            except Exception as e:
                self._log_to_debug("PRIMARY_AGENT_ERR", self.model, full_prompt_text, str(e))
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

    def integrate_interruption(self, obj, plan, progress, inp) -> Dict:
        """Reason about a mid-run user interruption in full context (objective,
        plan, progress so far, the message) and return how to fold it in:
        a mode (REVISE / CONSULT / REPLACE), a reconciled objective and plan, and
        a concrete directive for the agent's next turn. Uses the primary model —
        interruptions are rare and the decision is high-stakes."""
        prompt = ANALYZE_INTERRUPTION_PROMPT.format(obj=obj, plan=plan, progress=progress, inp=inp)
        try:
            resp = self.client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            self._log_to_debug("INTEGRATE_INTERRUPTION", self.api_model, prompt, content)
            cleaned = self._clean_json_response(content)
            return json.loads(cleaned)
        except Exception as e:
            self.logger.warning(f"Interruption integration failed: {e}")
            # Safe fallback: treat as a course-correction that preserves context,
            # surfacing the user's raw words rather than guessing a rewrite.
            return {"mode": "CONSULT", "objective": obj, "plan": "",
                    "directive": (f"The user interjected: \"{inp}\". Consider it, respond if it is a "
                                  f"question, and decide whether to adjust your approach."),
                    "reasoning": f"Integration failed ({e}); preserved context and surfaced input."}

    def integrate_resume(self, prev_objective, prev_plan, progress, new_instruction) -> Dict:
        """Merge the user's resume instruction (the new-session prompt) with the
        PREVIOUS session's objective. The user may just want to continue, or may
        redirect/modify the trajectory on restart; this reconciles the two into the
        objective the agent should now pursue. Returns {objective, directive,
        reasoning}. Best-effort: on any failure, falls back to the previous
        objective unchanged so resume never breaks."""
        prompt = INTEGRATE_RESUME_PROMPT.format(
            prev_objective=prev_objective, prev_plan=prev_plan,
            progress=progress, new_instruction=new_instruction)
        try:
            resp = self.client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            self._log_to_debug("INTEGRATE_RESUME", self.api_model, prompt, content)
            data = json.loads(self._clean_json_response(content))
            if not (data.get("objective") or "").strip():
                data["objective"] = prev_objective
            return data
        except Exception as e:
            self.logger.warning(f"Resume integration failed: {e}")
            return {"objective": prev_objective, "directive": "",
                    "reasoning": f"Integration failed ({e}); kept the previous objective."}

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
