from .base import BaseTool
import os
import json
import base64
import difflib
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
    """A tool to make targeted string replacements in files with fuzzy matching fallback."""
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
        """Find the best fuzzy match for search_str in content using a sliding window.

        Returns (matched_text, score) or (None, best_score) if below threshold.
        Uses line-based windowing with SequenceMatcher (Ratcliff/Obershelp algorithm),
        similar to how git and diff tools find similar blocks.
        """
        search_lines = search_str.splitlines(keepends=True)
        content_lines = content.splitlines(keepends=True)

        if not search_lines or not content_lines:
            return None, 0.0

        window_size = len(search_lines)
        best_score = 0.0
        best_start = -1
        best_end = -1

        # Try windows of size: exact, +/-1, +/-2 lines to handle
        # cases where the model missed or added a line
        for delta in [0, -1, 1, -2, 2]:
            adj_size = window_size + delta
            if adj_size < 1 or adj_size > len(content_lines):
                continue
            for i in range(len(content_lines) - adj_size + 1):
                window_text = ''.join(content_lines[i:i + adj_size])
                # Use SequenceMatcher for similarity scoring
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

    def execute(self, file_path: str, old_str: str, new_str: str = '') -> str:
        if not file_path:
            return 'Error: file_path parameter is required.'
        if old_str is None or old_str == '':
            return 'Error: old_str parameter is required and must not be empty.'
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

        match_method = 'exact'
        matched_text = old_str

        # --- PHASE 1: Exact match ---
        count = content.count(old_str)
        if count == 1:
            # Perfect: exactly one exact match
            self._consecutive_failures.pop(abs_path, None)  # Reset on success
        elif count > 1:
            return (
                f'Error: old_str matched {count} times in {file_path}. '
                f'It must be unique. Add more surrounding context to narrow the match.'
            )
        else:
            # --- PHASE 2: Whitespace-normalized match ---
            norm_content = self._normalize_whitespace(content)
            norm_search = self._normalize_whitespace(old_str)
            norm_count = norm_content.count(norm_search)

            if norm_count == 1:
                # Find the actual text in the original content that corresponds
                # to the normalized match position
                norm_pos = norm_content.find(norm_search)
                # Map normalized position back to original content by
                # counting through lines
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
                return (
                    f'Error: old_str matched {norm_count} times after whitespace normalization in {file_path}. '
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
                    # Track consecutive failures for this file
                    fail_count = self._consecutive_failures.get(abs_path, 0) + 1
                    self._consecutive_failures[abs_path] = fail_count

                    # Build diagnostic context: show what the file actually contains near where the match might be
                    search_preview = old_str[:200] + ('...' if len(old_str) > 200 else '')
                    first_line_of_search = old_str.split('\n')[0].strip()

                    # Try to find partial line matches to help diagnose the mismatch
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
                            f'Compare carefully with your old_str - the mismatch may be due to '
                            f'escape characters, quotes, or whitespace that got mangled in JSON encoding.'
                        )

                    if fail_count >= 3:
                        # Third strike: escalate to write_file
                        self._consecutive_failures[abs_path] = 0  # Reset counter
                        return (
                            f'Error: str_replace has failed {fail_count} times on {file_path}. '
                            f'Best fuzzy score was {score:.1%} (threshold: {FUZZY_MATCH_THRESHOLD:.0%}). '
                            f'The old_str you are providing does not match what is actually in the file. '
                            f'This is likely caused by escape sequences (like \\033) being mangled during JSON encoding.\n'
                            f'\n*** MANDATORY: Stop using str_replace for this file. '
                            f'Use open_file to read the current full content, then use write_file to rewrite the ENTIRE file with your changes applied. ***'
                            f'{diagnostic}'
                        )
                    else:
                        return (
                            f'Error: Could not find a match for old_str in {file_path} '
                            f'(attempt {fail_count}/3 before escalation to write_file). '
                            f'Best fuzzy score was {score:.1%} (threshold: {FUZZY_MATCH_THRESHOLD:.0%}). '
                            f'Searched for: {search_preview!r}'
                            f'{diagnostic}\n'
                            f'Re-open the file with open_file and copy the EXACT text you want to replace.'
                        )

        # --- PERFORM REPLACEMENT ---
        # Verify uniqueness of matched_text in the content (for fuzzy/normalized matches)
        if match_method != 'exact':
            match_count = content.count(matched_text)
            if match_count != 1:
                return (
                    f'Error: The {match_method} match appears {match_count} times in {file_path}. '
                    f'Cannot safely replace. Add more context to old_str.'
                )

        new_content = content.replace(matched_text, new_str, 1)

        # Verify the replacement actually changed something
        if new_content == content:
            return (
                f'Warning: Replacement produced identical content in {file_path}. '
                f'The old_str and new_str may be equivalent after matching. No changes written.'
            )

        try:
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            return f'Error writing file: {type(e).__name__}: {e}'

        # Update working memory if the file is open
        if self.worker.is_file_open(abs_path) or self.worker.is_file_open(file_path):
            self.worker.update_open_file(abs_path, new_content)

        # Build result message
        old_line_count = old_str.count('\n') + 1
        new_line_count = new_str.count('\n') + 1 if new_str else 0
        if match_method == 'exact':
            return f"Successfully replaced {old_line_count} lines with {new_line_count} lines in {file_path}."
        else:
            return (
                f"Successfully replaced {old_line_count} lines with {new_line_count} lines in {file_path} "
                f"(matched via {match_method})."
            )


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

        if content.startswith('base64:'):
            try:
                content_decoded = base64.b64decode(content[7:]).decode('utf-8')
            except Exception as e:
                return f'Error decoding base64 content: {e}'
        else:
            content_decoded = content

        abs_path = os.path.abspath(file_path)
        try:
            parent_dir = os.path.dirname(abs_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

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
