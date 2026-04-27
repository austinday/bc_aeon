from .base import BaseTool
import os
import json
import base64
import difflib
import re
from ..core.prompts import (
    TOOL_DESC_OPEN_FILE,
    TOOL_DESC_CLOSE_FILE,
    TOOL_DESC_WRITE_FILE,
    TOOL_DESC_STR_REPLACE,
)
from .analyzers import FileAnalyzer

# Max characters before rejecting full-content files
MAX_FILE_READ_SIZE = 250000

# Fuzzy match confidence threshold (0.0 - 1.0)
FUZZY_MATCH_THRESHOLD = 0.6  # Must be reasonably confident before applying a fuzzy replacement


class OpenFileTool(BaseTool):
    def __init__(self, worker):
        super().__init__(
            name='open_file',
            description=TOOL_DESC_OPEN_FILE
        )
        self.worker = worker

    def execute(self, file_path: str) -> str:
        if not file_path:
            return 'Error: file_path parameter is required.'

        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f'Error: File not found: {file_path}'
        if os.path.isdir(abs_path):
            return f'Error: {file_path} is a directory. Refer to the Project Tree in your system context to see files, then open a specific file.'

        # Return accurate status if file is already loaded
        if self.worker.is_file_open(file_path) or self.worker.is_file_open(abs_path):
            return f"File '{file_path}' is already open in working memory. No need to re-open it."

        try:
            analyzer = FileAnalyzer(abs_path)
            result = analyzer.analyze()
        except Exception as e:
            return f'Error analyzing file: {type(e).__name__}: {e}'

        summary_type = result.get('summary_type', '')

        if summary_type == 'opaque_binary':
            return f"File '{file_path}' is a binary file that cannot be displayed. Use a script to analyze it."

        if summary_type == 'error':
            return f"Error reading file: {result.get('error_message', 'Unknown error')}"

        if summary_type in ('empty_file', 'empty'):
            content = '(empty file)'

        elif summary_type == 'full_content':
            raw = result.get('content', '')
            if isinstance(raw, (dict, list)):
                content = json.dumps(raw, indent=2)
            else:
                content = str(raw)
        else:
            # Structured summary (dataframe, sequence_summary, archive_contents, etc.)
            parts = [f'[File Summary: {summary_type}]']
            for key, value in result.items():
                if key in ('file_name', 'file_size_bytes', 'summary_type'):
                    continue
                if isinstance(value, (dict, list)):
                    parts.append(f'{key}: {json.dumps(value, indent=2, default=str)}')
                else:
                    parts.append(f'{key}: {value}')
            content = '\n'.join(parts)

        if len(content) > MAX_FILE_READ_SIZE:
            return (
                f"File '{file_path}' content is too large ({len(content):,} chars) to open directly. "
                f"Limit is {MAX_FILE_READ_SIZE:,} chars. Use a script to analyze this file."
            )

        self.worker.update_open_file(abs_path, content)
        slots_used = len(self.worker.open_files)
        return f"File '{file_path}' opened in working memory. ({slots_used} files open)"


class CloseFileTool(BaseTool):
    def __init__(self, worker):
        super().__init__(
            name='close_file',
            description=TOOL_DESC_CLOSE_FILE
        )
        self.worker = worker

    def execute(self, file_path: str) -> str:
        if not file_path:
            return 'Error: file_path parameter is required.'
        if self.worker.close_file(file_path):
            return f"File '{file_path}' closed."
        return f"File '{file_path}' was not open."


class StrReplaceTool(BaseTool):
    """A tool to make targeted string replacements in files using unified diff blocks or fuzzy matching fallback."""
    def __init__(self, worker):
        super().__init__(
            name='str_replace',
            description=TOOL_DESC_STR_REPLACE
        )
        self.worker = worker
        self._consecutive_failures = {}  # {abs_path: count} tracks fuzzy match failures per file

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize trailing whitespace on each line for comparison."""
        lines = text.splitlines()
        return '\n'.join(line.rstrip() for line in lines)

    def _find_fuzzy_match(self, content: str, search_str: str) -> tuple:
        """Find the best fuzzy match for search_str in content using a sliding window."""
        search_lines = search_str.splitlines(keepends=True)
        content_lines = content.splitlines(keepends=True)

        if not search_lines or not content_lines:
            return None, 0.0

        window_size = len(search_lines)
        best_score = 0.0
        best_start = -1
        best_end = -1

        for delta in [0, -1, 1, -2, 2]:
            adj_size = window_size + delta
            if adj_size < 1 or adj_size > len(content_lines):
                continue
            for i in range(len(content_lines) - adj_size + 1):
                window_text = ''.join(content_lines[i:i + adj_size])
                score = difflib.SequenceMatcher(
                    None, search_str, window_text, autojunk=False
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_start = i
                    best_end = i + adj_size

        if best_score >= FUZZY_MATCH_THRESHOLD and best_start >= 0:
            matched_text = ''.join(content_lines[best_start:best_end])
            return matched_text, best_score

        return None, best_score

    def _apply_single_replace(self, abs_path: str, file_path: str, content: str, old_str: str, new_str: str) -> tuple:
        match_method = 'exact'
        matched_text = old_str

        # --- PHASE 1: Exact match ---
        count = content.count(old_str)
        if count == 1:
            self._consecutive_failures.pop(abs_path, None)  # Reset on success
        elif count > 1:
            return content, None, (
                f'Error: The SEARCH block matched {count} times in {file_path}. '
                f'It must be unique. Add more surrounding context to narrow the match.'
            )
        else:
            # --- PHASE 2: Whitespace-normalized match ---
            norm_content = self._normalize_whitespace(content)
            norm_search = self._normalize_whitespace(old_str)
            norm_count = norm_content.count(norm_search)

            if norm_count == 1:
                norm_pos = norm_content.find(norm_search)
                norm_lines_before = norm_content[:norm_pos].count('\n')
                search_line_count = old_str.count('\n') + (0 if old_str.endswith('\n') else 1)
                original_lines = content.splitlines(keepends=True)
                start_line = norm_lines_before
                end_line = start_line + search_line_count
                if end_line <= len(original_lines):
                    matched_text = ''.join(original_lines[start_line:end_line])
                    match_method = 'whitespace-normalized'
                    self._consecutive_failures.pop(abs_path, None)  # Reset on success
                else:
                    matched_text = None
            elif norm_count > 1:
                return content, None, (
                    f'Error: SEARCH block matched {norm_count} times after whitespace normalization in {file_path}. '
                    f'Add more surrounding context to narrow the match.'
                )
            else:
                matched_text = None

            # --- PHASE 3: Fuzzy match ---
            if matched_text is None:
                fuzzy_match, score = self._find_fuzzy_match(content, old_str)
                if fuzzy_match is not None:
                    matched_text = fuzzy_match
                    match_method = f'fuzzy (confidence: {score:.1%})'
                    self._consecutive_failures.pop(abs_path, None)  # Reset on success
                else:
                    fail_count = self._consecutive_failures.get(abs_path, 0) + 1
                    self._consecutive_failures[abs_path] = fail_count

                    search_preview = old_str[:200] + ('...' if len(old_str) > 200 else '')
                    first_line_of_search = old_str.split('\n')[0].strip()

                    diagnostic_lines = []
                    content_lines = content.splitlines()
                    for i, line in enumerate(content_lines):
                        if first_line_of_search and first_line_of_search[:30] in line:
                            start = max(0, i - 1)
                            end = min(len(content_lines), i + 4)
                            snippet = '\n'.join(f'  L{start+j+1}: {content_lines[start+j]}' for j in range(end - start))
                            diagnostic_lines.append(snippet)
                            break

                    diagnostic = ''
                    if diagnostic_lines:
                        diagnostic = (
                            f'\nNearest partial match found around these lines:\n'
                            f'{diagnostic_lines[0]}\n'
                            f'Compare carefully with your SEARCH block - the mismatch may be due to '
                            f'escape characters, quotes, or whitespace that got mangled.'
                        )

                    if fail_count >= 3:
                        self._consecutive_failures[abs_path] = 0  # Reset counter
                        return content, None, (
                            f'Error: str_replace has failed {fail_count} times on {file_path}. '
                            f'Best fuzzy score was {score:.1%} (threshold: {FUZZY_MATCH_THRESHOLD:.0%}). '
                            f'The text you provided does not match what is actually in the file.\n'
                            f'\n*** MANDATORY: Stop using str_replace for this file. '
                            f'Use open_file to read the current full content, then use write_file to rewrite the ENTIRE file. ***'
                            f'{diagnostic}'
                        )
                    else:
                        return content, None, (
                            f'Error: Could not find a match for SEARCH block in {file_path} '
                            f'(attempt {fail_count}/3 before escalation to write_file). '
                            f'Best fuzzy score was {score:.1%} (threshold: {FUZZY_MATCH_THRESHOLD:.0%}). '
                            f'Searched for: {search_preview!r}'
                            f'{diagnostic}\n'
                            f'Re-open the file with open_file and copy the EXACT text you want to replace.'
                        )

        if match_method != 'exact':
            match_count = content.count(matched_text)
            if match_count != 1:
                return content, None, (
                    f'Error: The {match_method} match appears {match_count} times in {file_path}. '
                    f'Cannot safely replace. Add more context.'
                )

        new_content = content.replace(matched_text, new_str, 1)

        if new_content == content:
            return content, None, f'Warning: Replacement produced identical content in {file_path}. No changes written.'

        return new_content, match_method, None

    def execute(self, file_path: str, patch: str = None, old_str: str = None, new_str: str = '') -> str:
        if not file_path:
            return 'Error: file_path parameter is required.'
        if not patch and not old_str:
            return 'Error: Must provide either patch or old_str parameter.'
        if new_str is None:
            new_str = ''

        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f'Error: File not found: {file_path}'
        if os.path.isdir(abs_path):
            return f'Error: {file_path} is a directory, not a file.'

        try:
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            return f'Error reading file: {type(e).__name__}: {e}'

        current_content = content
        methods_used = []

        if patch:
            blocks = re.findall(r'<<<<\s*SEARCH\n?(.*?)\n?====\n?(.*?)\n?>>>>\s*REPLACE', patch, re.DOTALL)
            if not blocks:
                return "Error: Could not parse SEARCH/REPLACE blocks. Ensure you use <<<< SEARCH, ====, and >>>> REPLACE correctly."
            
            for s_str, r_str in blocks:
                if not s_str: continue
                new_c, method, err = self._apply_single_replace(abs_path, file_path, current_content, s_str, r_str)
                if err and err.startswith('Error'):
                    return err
                if method:
                    methods_used.append(method)
                current_content = new_c
        else:
            new_c, method, err = self._apply_single_replace(abs_path, file_path, current_content, old_str, new_str)
            if err and err.startswith('Error'):
                return err
            if method:
                methods_used.append(method)
            current_content = new_c

        if current_content == content:
            return f"Warning: No changes were made to {file_path}. Content is identical."

        try:
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(current_content)
        except Exception as e:
            return f'Error writing file: {type(e).__name__}: {e}'

        if self.worker.is_file_open(abs_path) or self.worker.is_file_open(file_path):
            self.worker.update_open_file(abs_path, current_content)

        method_str = ", ".join(set(methods_used)) if methods_used else "exact"
        block_count = len(methods_used) if patch else 1
        return f"Successfully applied {block_count} patch block(s) to {file_path} (matched via {method_str})."


class WriteFileTool(BaseTool):
    def __init__(self, worker):
        super().__init__(
            name='write_file',
            description=TOOL_DESC_WRITE_FILE
        )
        self.worker = worker

    def execute(self, file_path: str, content: str) -> str:
        if not file_path:
            return 'Error: file_path parameter is required.'
        if content is None:
            return 'Error: content parameter is required (can be empty string).'

        is_binary = False
        if content.startswith('base64:'):
            try:
                content_decoded = base64.b64decode(content[7:])
                try:
                    content_decoded = content_decoded.decode('utf-8')
                except UnicodeDecodeError:
                    is_binary = True
            except Exception as e:
                return f'Error decoding base64 content: {e}'
        else:
            content_decoded = content

        abs_path = os.path.abspath(file_path)
        try:
            parent_dir = os.path.dirname(abs_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            if is_binary:
                with open(abs_path, 'wb') as f:
                    f.write(content_decoded)
            else:
                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(content_decoded)

            # Always remove from working memory after writing.
            self.worker.close_file(file_path)
            self.worker.close_file(abs_path)

            return f'Successfully wrote to {file_path}.'
        except PermissionError:
            return f'Error: Permission denied writing to {file_path}'
        except Exception as e:
            return f'Error writing file: {type(e).__name__}: {e}'
