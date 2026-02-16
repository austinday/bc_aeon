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

        if count == 0:
            preview_lines = content.splitlines()[:20]
            preview = '\n'.join(preview_lines)
            return (
                f'Error: old_str not found in {file_path}. '
                f'The string to replace does not exist in the file. '
                f'Double-check exact whitespace, indentation, and spelling.\n'
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
        return (
            f'Successfully edited {file_path}. '
            f'Replaced {lines_removed} line(s) with {lines_added} line(s).'
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
