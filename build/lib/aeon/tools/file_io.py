from .base import BaseTool
import os
import base64
import json
from ..core.prompts import (
    TOOL_DESC_OPEN_FILE,
    TOOL_DESC_CLOSE_FILE,
    TOOL_DESC_WRITE_FILE,
    TOOL_DESC_EDIT_FILE,
)

# Max characters before forcing script usage
MAX_FILE_READ_SIZE = 250000 
# Max characters for a "cell" in a summary (CSV/JSON/FASTA)
MAX_CELL_SIZE = 100

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
            return f"Error: {file_path} is a directory. Use run_command with 'ls' to list contents."
        
        file_size = os.path.getsize(abs_path)
        if file_size > MAX_FILE_READ_SIZE:
             # Deterministic rejection of large files
             return (f"File '{file_path}' is too large ({file_size:,} bytes) to open directly. "
                     f"Limit is {MAX_FILE_READ_SIZE:,} bytes. Please use a script (python/bash) "
                     "to analyze this file and print relevant details.")

        ext = os.path.splitext(abs_path)[1].lower()
        content = ""

        try:
            # --- Smart Logic ---
            if ext in ['.csv', '.tsv']:
                content = self._summarize_tabular(abs_path, sep=',' if ext == '.csv' else '\t')
            elif ext == '.json':
                content = self._summarize_json(abs_path)
            elif ext in ['.fasta', '.fa', '.fna']:
                content = self._summarize_fasta(abs_path)
            elif ext in ['.pdb', '.cif', '.bam', '.gz', '.zip', '.tar', '.pdf', '.jpg', '.png']:
                # Binary or structure files that shouldn't be opened directly
                return f"File '{file_path}' is a binary/structure/large format file. Do not open directly. Use a script to analyze headers or content."
            else:
                # Default: Read full text for scripts/docs
                with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

            # Update Worker State with absolute path for consistency
            self.worker.update_open_file(abs_path, content)
            return f"File '{file_path}' opened in Short Term Memory."

        except UnicodeDecodeError as e:
            return f"Error: File appears to be binary or has encoding issues: {e}"
        except Exception as e:
            return f"Error opening file: {type(e).__name__}: {e}"

    def _truncate(self, text):
        s = str(text)
        if len(s) > MAX_CELL_SIZE:
            return s[:MAX_CELL_SIZE] + "..."
        return s

    def _summarize_tabular(self, path, sep):
        # Read header and first data row manually to avoid pandas dep if strictly needed, 
        # but pure python is safer for environment agnostic.
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        if not lines:
            return "[Empty CSV/TSV]"
        
        header = lines[0].strip().split(sep)
        header_trunc = [self._truncate(h) for h in header]
        
        summary = f"Headers: {header_trunc}\n"
        
        if len(lines) > 1:
            row1 = lines[1].strip().split(sep)
            row1_trunc = [self._truncate(r) for r in row1]
            summary += f"Row 1 Example: {row1_trunc}\n"
            summary += f"[Total Lines: {len(lines)}]"
        else:
            summary += "[No Data Rows]"
            
        return summary

    def _summarize_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            schema = "List of Objects"
            if data:
                # Sample first item
                sample = data[0]
                # Truncate values in sample
                if isinstance(sample, dict):
                    sample = {k: self._truncate(v) for k, v in sample.items()}
                else:
                    sample = self._truncate(sample)
                content = f"Schema: {schema}\nLength: {len(data)} items\nSample Item 0: {json.dumps(sample, indent=2)}"
            else:
                content = "Schema: Empty List"
        elif isinstance(data, dict):
            keys = list(data.keys())
            # Shallow truncated sample
            sample = {k: self._truncate(v) for k, v in data.items() if isinstance(v, (str, int, float, bool))}
            content = f"Top-level Keys ({len(keys)}): {keys[:20]}{'...' if len(keys) > 20 else ''}\nShallow Sample: {json.dumps(sample, indent=2)}"
        else:
            content = f"JSON Content: {self._truncate(data)}"
        
        return content

    def _summarize_fasta(self, path):
        # Read first few lines, truncate long sequences
        summary = []
        with open(path, 'r', encoding='utf-8') as f:
            for _ in range(10): # First 10 lines max
                line = f.readline()
                if not line: break
                line = line.strip()
                if line.startswith('>'):
                    summary.append(self._truncate(line))
                else:
                    summary.append(self._truncate(line))
        return "\n".join(summary) + "\n[...File Truncated for View...]"

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

        # Count occurrences to enforce uniqueness
        count = content.count(old_str)

        if count == 0:
            # Provide a helpful snippet of the file so the executor can orient
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

        # Exactly one occurrence - perform the replacement
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
            # Ensure parent directory exists (handle files in current dir correctly)
            parent_dir = os.path.dirname(abs_path)
            if parent_dir:  # Only makedirs if there's actually a parent directory
                os.makedirs(parent_dir, exist_ok=True)
            
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(content_decoded)
            
            # If file is currently "open" in tabs, update the tab content immediately
            # We close it to force a re-read/refresh to ensure consistency.
            if self.worker.is_file_open(file_path):
                self.worker.close_file(file_path)
                return f"Successfully wrote to {file_path}. (File closed from memory to ensure freshness)."
            
            return f"Successfully wrote to {file_path}."
        except PermissionError:
            return f"Error: Permission denied writing to {file_path}"
        except Exception as e:
            return f"Error writing file: {type(e).__name__}: {e}"
