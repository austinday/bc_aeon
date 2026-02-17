from .base import BaseTool
import os
import json
import base64
from ..core.prompts import (
    TOOL_DESC_OPEN_FILE,
    TOOL_DESC_CLOSE_FILE,
    TOOL_DESC_WRITE_FILE,
    TOOL_DESC_EDIT_FILE,
)
from .analyzers import FileAnalyzer

# Max characters before rejecting full-content files
MAX_FILE_READ_SIZE = 250000


class OpenFileTool(BaseTool):
    def __init__(self, worker):
        super().__init__(
            name="open_file",
            description=TOOL_DESC_OPEN_FILE
        )
        self.worker = worker

    def execute(self, file_path: str) -> str:
        if not file_path:
            return "Error: file_path parameter is required."

        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f"Error: File not found: {file_path}"
        if os.path.isdir(abs_path):
            return f"Error: {file_path} is a directory. Refer to the 'Project Tree' in your system context to see files, then open a specific file."

        # Return accurate status if file is already loaded
        if self.worker.is_file_open(file_path) or self.worker.is_file_open(abs_path):
            return f"File '{file_path}' is already open in working memory. No need to re-open it."

        try:
            analyzer = FileAnalyzer(abs_path)
            result = analyzer.analyze()
        except Exception as e:
            return f"Error analyzing file: {type(e).__name__}: {e}"

        summary_type = result.get('summary_type', '')

        if summary_type == 'opaque_binary':
            return f"File '{file_path}' is a binary file that cannot be displayed. Use a script to analyze it."

        if summary_type == 'error':
            return f"Error reading file: {result.get('error_message', 'Unknown error')}"

        if summary_type in ('empty_file', 'empty'):
            content = "(empty file)"

        elif summary_type == 'full_content':
            raw = result.get('content', '')
            if isinstance(raw, (dict, list)):
                content = json.dumps(raw, indent=2)
            else:
                content = str(raw)
            if len(content) > MAX_FILE_READ_SIZE:
                return (
                    f"File '{file_path}' content is too large ({len(content):,} chars) to open directly. "
                    f"Limit is {MAX_FILE_READ_SIZE:,} chars. Use a script to analyze this file."
                )
        else:
            # Structured summary (dataframe, sequence_summary, archive_contents, etc.)
            parts = [f"[File Summary: {summary_type}]"]
            for key, value in result.items():
                if key in ('file_name', 'file_size_bytes', 'summary_type'):
                    continue
                if isinstance(value, (dict, list)):
                    parts.append(f"{key}: {json.dumps(value, indent=2, default=str)}")
                else:
                    parts.append(f"{key}: {value}")
            content = "\n".join(parts)

        self.worker.update_open_file(abs_path, content)
        slots_used = len(self.worker.open_files)
        return f"File '{file_path}' opened in working memory. ({slots_used} files open)"


class EditFileTool(BaseTool):
    """A tool to make targeted edits to a file via unique string replacement."""
    def __init__(self, worker):
        super().__init__(
            name='edit_file',
            description=TOOL_DESC_EDIT_FILE
        )
        self.worker = worker

    @staticmethod
    def _normalize(s):
        """Normalize a string for fuzzy comparison: collapse whitespace, strip escape chars."""
        import re as _re
        # Collapse all whitespace runs (spaces, tabs) to single space, preserve newlines
        s = _re.sub(r'[^\S\n]+', ' ', s)
        # Remove backslash escapes before quotes (common bash/python nesting issue)
        s = s.replace('\\"', '"').replace("\\'", "'")
        return s.strip()

    @staticmethod
    def _find_best_match(content, old_str, context_lines=5):
        """Find the most similar region in content to old_str using difflib."""
        import difflib
        old_lines = old_str.splitlines()
        content_lines = content.splitlines()
        if not old_lines or not content_lines:
            return None
        matcher = difflib.SequenceMatcher(None, [], old_lines)
        best_ratio = 0.0
        best_start = 0
        # Slide a window of similar size over the file
        window = max(len(old_lines), 1)
        for i in range(max(1, len(content_lines) - window + 1)):
            chunk = content_lines[i:i + window + 2]  # slightly larger window
            matcher.set_seq1(chunk)
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
        if best_ratio < 0.3:
            return None
        # Return context around best match
        start = max(0, best_start - context_lines)
        end = min(len(content_lines), best_start + window + context_lines + 2)
        preview = '\n'.join(
            f'{i+1:4d} | {content_lines[i]}'
            for i in range(start, end)
        )
        return best_ratio, best_start + 1, preview

    def execute(self, file_path: str, old_str: str, new_str: str = '') -> str:
        if not file_path:
            return 'Error: file_path parameter is required.'
        if not old_str:
            return 'Error: old_str parameter is required and cannot be empty.'
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

        count = content.count(old_str)

        # --- FALLBACK: normalized matching if exact match fails ---
        used_normalized = False
        if count == 0:
            norm_old = self._normalize(old_str)
            # Try to find a unique region that matches after normalization
            # Scan content line-by-line to find matching region
            content_lines = content.splitlines(keepends=True)
            old_lines = old_str.splitlines()
            old_line_count = len(old_lines)

            candidates = []
            for i in range(len(content_lines) - old_line_count + 1):
                chunk = ''.join(content_lines[i:i + old_line_count])
                if self._normalize(chunk) == norm_old:
                    candidates.append((i, chunk))

            if len(candidates) == 1:
                # Unique normalized match found -- use the actual raw text
                actual_old = candidates[0][1]
                # Perform the replacement with the actual raw text
                if actual_old.endswith('\n') and not old_str.endswith('\n'):
                    actual_old = actual_old.rstrip('\n')
                count = content.count(actual_old)
                if count == 1:
                    old_str = actual_old
                    used_normalized = True
                else:
                    count = 0  # fall through to error
            elif len(candidates) > 1:
                return (
                    f'Error: old_str matches {len(candidates)} locations in {file_path} '
                    f'after normalizing whitespace/escapes. '
                    f'Include more surrounding context to disambiguate.'
                )

        if count == 0:
            # Still not found -- provide helpful diagnostics
            match_info = self._find_best_match(content, old_str)
            if match_info:
                ratio, line_num, preview = match_info
                return (
                    f'Error: old_str not found in {file_path} (even after normalizing whitespace/escapes). '
                    f'Closest match ({ratio:.0%} similar) near line {line_num}:\n'
                    f'--- File content near best match ---\n{preview}\n'
                    f'--- Your old_str (repr) ---\n{repr(old_str[:300])}\n'
                    f'HINT: Check for escaped quotes (\\" vs "), tabs vs spaces, '
                    f'or trailing whitespace differences.'
                )
            else:
                total_lines = len(content.splitlines())
                preview_lines = content.splitlines()[:20]
                preview = '\n'.join(preview_lines)
                return (
                    f'Error: old_str not found in {file_path} ({total_lines} lines). '
                    f'No similar region found. The text may not exist in this file.\n'
                    f'--- First 20 lines of file ---\n{preview}'
                )

        if count > 1:
            return (
                f'Error: old_str is not unique in {file_path} '
                f'(found {count} occurrences). '
                f'Include more surrounding context in old_str to make it unique.'
            )

        new_content = content.replace(old_str, new_str, 1)

        try:
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except PermissionError:
            return f'Error: Permission denied writing to {file_path}'
        except Exception as e:
            return f'Error writing file: {type(e).__name__}: {e}'

        # If the file is open in working memory, update the cached content
        if self.worker.is_file_open(file_path) or self.worker.is_file_open(abs_path):
            self.worker.update_open_file(abs_path, new_content)

        lines_removed = old_str.count('\n') + 1
        lines_added = new_str.count('\n') + 1
        norm_note = ' (matched via normalized whitespace/escapes)' if used_normalized else ''
        return (
            f'Successfully edited {file_path}. '
            f'Replaced {lines_removed} line(s) with {lines_added} line(s).{norm_note}'
        )


class CloseFileTool(BaseTool):
    def __init__(self, worker):
        super().__init__(
            name="close_file",
            description=TOOL_DESC_CLOSE_FILE
        )
        self.worker = worker

    def execute(self, file_path: str) -> str:
        if not file_path:
            return "Error: file_path parameter is required."
        if self.worker.close_file(file_path):
            return f"File '{file_path}' closed."
        return f"File '{file_path}' was not open."


class WriteFileTool(BaseTool):
    def __init__(self, worker):
        super().__init__(
            name="write_file",
            description=TOOL_DESC_WRITE_FILE
        )
        self.worker = worker

    def execute(self, file_path: str, content: str) -> str:
        if not file_path:
            return "Error: file_path parameter is required."
        if content is None:
            return "Error: content parameter is required (can be empty string)."

        if content.startswith("base64:"):
            try:
                content_decoded = base64.b64decode(content[7:]).decode("utf-8")
            except Exception as e:
                return f"Error decoding base64 content: {e}"
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
            # The agent wrote the content so it already knows what is in it.
            self.worker.close_file(file_path)
            self.worker.close_file(abs_path)

            return f"Successfully wrote to {file_path}."
        except PermissionError:
            return f"Error: Permission denied writing to {file_path}"
        except Exception as e:
            return f"Error writing file: {type(e).__name__}: {e}"
