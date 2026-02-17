from .base import BaseTool
import os
import json
import base64
from ..core.prompts import (
    TOOL_DESC_OPEN_FILE,
    TOOL_DESC_CLOSE_FILE,
    TOOL_DESC_WRITE_FILE,
    TOOL_DESC_EDIT_FILE_LINES,
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


class EditFileLinesTool(BaseTool):
    """Edit a file by replacing a range of lines identified by line number.

    Automatically adjusts line numbers when multiple edits target the same
    file within a single iteration. The agent always uses the ORIGINAL line
    numbers it saw in OPEN FILES at the start of the iteration.

    Offset logic:
    - Each edit records (original_start, original_end, delta) where
      delta = lines_added - lines_removed.
    - Subsequent edits look at all prior edits: if a prior edit's original
      range ended before our original start, we shift by its delta.
    - Overlapping edits (targeting lines within an already-edited range)
      are detected and rejected with a clear error.
    - The offset list is stored on worker.line_offsets[abs_path] and
      cleared at the start of each iteration by the worker.
    """
    def __init__(self, worker):
        super().__init__(
            name='edit_file_lines',
            description=TOOL_DESC_EDIT_FILE_LINES
        )
        self.worker = worker

    def _check_overlap(self, abs_path: str, orig_start: int, orig_end: int) -> str:
        """Check if this edit overlaps with any prior edit. Returns error string or None."""
        prior_edits = self.worker.line_offsets.get(abs_path, [])
        for p_start, p_end, _delta in prior_edits:
            # Two ranges [a,b] and [c,d] overlap iff a <= d and c <= b
            if orig_start <= p_end and p_start <= orig_end:
                return (
                    f'Error: Edit range (lines {orig_start}-{orig_end}) overlaps with a '
                    f'prior edit in this batch (lines {p_start}-{p_end}). '
                    f'Overlapping edits in the same batch are not supported. '
                    f'Combine them into a single edit, or use write_file to rewrite the whole file.'
                )
        return None

    def _compute_shift(self, abs_path: str, orig_start: int) -> int:
        """Compute cumulative line shift from all prior edits that are above orig_start."""
        prior_edits = self.worker.line_offsets.get(abs_path, [])
        shift = 0
        for p_start, p_end, p_delta in prior_edits:
            # Prior edit ended strictly before our original start = it's above us
            if p_end < orig_start:
                shift += p_delta
        return shift

    def _record_edit(self, abs_path: str, orig_start: int, orig_end: int, delta: int):
        """Record this edit for future offset calculations."""
        if abs_path not in self.worker.line_offsets:
            self.worker.line_offsets[abs_path] = []
        self.worker.line_offsets[abs_path].append((orig_start, orig_end, delta))

    def execute(self, file_path: str, start_line: int, end_line: int = None, new_content: str = '') -> str:
        if not file_path:
            return 'Error: file_path parameter is required.'
        if start_line is None:
            return 'Error: start_line parameter is required (1-indexed).'

        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f'Error: File not found: {file_path}'
        if os.path.isdir(abs_path):
            return f'Error: {file_path} is a directory, not a file.'

        try:
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception as e:
            return f'Error reading file: {type(e).__name__}: {e}'

        total_lines = len(lines)

        # Validate start_line
        try:
            start_line = int(start_line)
        except (TypeError, ValueError):
            return f'Error: start_line must be an integer, got {type(start_line).__name__}.'
        if start_line < 1:
            return f'Error: start_line must be >= 1 (got {start_line}). Lines are 1-indexed.'

        # Handle end_line
        insert_mode = False
        if end_line is not None:
            try:
                end_line = int(end_line)
            except (TypeError, ValueError):
                return f'Error: end_line must be an integer, got {type(end_line).__name__}.'
            if end_line == 0:
                insert_mode = True
        else:
            end_line = start_line

        if new_content is None:
            new_content = ''

        # Save the original line numbers the agent specified
        orig_start = start_line
        orig_end = end_line if not insert_mode else start_line

        # Check for overlapping edits in this batch
        if not insert_mode:
            overlap_err = self._check_overlap(abs_path, orig_start, orig_end)
            if overlap_err:
                return overlap_err

        # Compute shift from prior edits in this iteration
        shift = self._compute_shift(abs_path, orig_start)
        adj_start = max(1, orig_start + shift)
        adj_end = max(1, orig_end + shift) if not insert_mode else adj_start

        # Prepare replacement lines
        new_lines = []
        if new_content:
            new_lines = new_content.splitlines(True)
            if new_lines and not new_lines[-1].endswith('\n'):
                new_lines[-1] += '\n'

        if insert_mode:
            if adj_start > total_lines:
                lines.extend(new_lines)
                action = f'Appended {len(new_lines)} line(s) to end of file'
            else:
                idx = max(0, adj_start - 1)
                lines[idx:idx] = new_lines
                action = f'Inserted {len(new_lines)} line(s) before original line {orig_start}'
            delta = len(new_lines)
            self._record_edit(abs_path, orig_start, orig_start, delta)
        else:
            if adj_start > total_lines:
                return (
                    f'Error: start_line {orig_start} (adjusted to {adj_start}) '
                    f'is beyond end of file ({total_lines} lines). '
                    f'To append, use end_line=0 for insert mode.'
                )
            if adj_end < adj_start:
                return (
                    f'Error: end_line ({orig_end}) must be >= start_line ({orig_start}), '
                    f'or set end_line=0 for insert mode.'
                )

            s = adj_start - 1  # 0-indexed
            e = min(adj_end, total_lines)
            removed_count = e - s
            lines[s:e] = new_lines
            delta = len(new_lines) - removed_count
            self._record_edit(abs_path, orig_start, orig_end, delta)
            action = (
                f'Replaced original lines {orig_start}-{orig_end} '
                f'(adjusted to {adj_start}-{min(adj_end, total_lines)}, '
                f'{removed_count} removed, {len(new_lines)} added)'
            )

        # Write back
        try:
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except PermissionError:
            return f'Error: Permission denied writing to {file_path}'
        except Exception as e:
            return f'Error writing file: {type(e).__name__}: {e}'

        # Update working memory if file is open
        new_content_full = ''.join(lines)
        if self.worker.is_file_open(file_path) or self.worker.is_file_open(abs_path):
            self.worker.update_open_file(abs_path, new_content_full)

        return f'Successfully edited {file_path}. {action}. File now has {len(lines)} line(s).'


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
            self.worker.close_file(file_path)
            self.worker.close_file(abs_path)

            # Clear any line offsets since the file was fully rewritten
            self.worker.line_offsets.pop(abs_path, None)

            return f"Successfully wrote to {file_path}."
        except PermissionError:
            return f"Error: Permission denied writing to {file_path}"
        except Exception as e:
            return f"Error writing file: {type(e).__name__}: {e}"
