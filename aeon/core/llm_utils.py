import re
import json
import logging
from typing import Dict, Optional, List

logger = logging.getLogger("aeon")

def clean_json_response(content: str) -> str:
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

    logger.warning(f"No JSON object found in response: {content[:200]}...")
    return "{}"

def find_json_end(raw: str) -> int:
    """Find the position right after the outermost JSON closing brace."""
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

def extract_content_blocks(raw: str, json_end: int) -> dict:
    """Extract content blocks from text AFTER the JSON object."""
    blocks = {}
    remainder = raw[json_end:] if json_end > 0 else raw

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

    if not blocks:
        v1_pattern = r'<{2,4}(BLOCK_[A-Za-z0-9_]+)>{2,4}\n?(.*?)<{2,4}END_\1>{2,4}'
        for match in re.finditer(v1_pattern, remainder, re.DOTALL):
            block_id = match.group(1)
            content = match.group(2)
            if content.endswith('\n'):
                content = content[:-1]
            blocks[block_id] = content

    return blocks

def extract_inline_content(value: str) -> Optional[str]:
    """Fallback: extract content from a JSON string value with embedded delimiters."""
    v2_inline = re.search(
        r'(?:^|\n)\s*-*\s*BEGIN[\s_]+BLOCK[\s_]*\d+\s*-*\s*\n'
        r'(.*?)'
        r'\n\s*-*\s*END[\s_]+BLOCK[\s_]*\d+\s*-*\s*(?:\n|$)',
        value, re.DOTALL
    )
    if v2_inline:
        return v2_inline.group(1)

    v1_inline = re.search(
        r'<{2,4}BLOCK_\w+>{2,4}\n?(.*?)\n?<{2,4}END_BLOCK_\w+>{2,4}',
        value, re.DOTALL
    )
    if v1_inline:
        content = v1_inline.group(1)
        if content.endswith('\n'):
            content = content[:-1]
        return content

    tag_prefix = re.match(
        r'^[_<]{1,4}BLOCK[\s_]*\d+[_>]{1,4}\s*\n(.*)',
        value, re.DOTALL
    )
    if tag_prefix:
        content = tag_prefix.group(1)
        content = re.sub(r'\n\s*-*\s*END[\s_]+BLOCK[\s_]*\d+\s*-*\s*$', '', content)
        content = re.sub(r'\n\s*<{2,4}END_BLOCK_\w+>{2,4}\s*$', '', content)
        return content

    return None

def substitute_blocks(obj, blocks: dict, missing_blocks: list = None):
    """Recursively substitute __BLOCK_N__ placeholders in parsed JSON."""
    if missing_blocks is None:
        missing_blocks = []

    if isinstance(obj, dict):
        return {k: substitute_blocks(v, blocks, missing_blocks) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_blocks(item, blocks, missing_blocks) for item in obj]
    elif isinstance(obj, str):
        stripped = obj.strip()
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
                return obj

        if '\n' in obj and 'BLOCK' in obj:
            extracted = extract_inline_content(obj)
            if extracted is not None:
                return extracted
        return obj
    return obj