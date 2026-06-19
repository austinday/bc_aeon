from .base import BaseTool
import os
import json
import base64
import difflib
import re
import tempfile
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
FUZZY_MATCH_THRESHOLD = 0.6

# Directories that hold stale duplicates / vendored code / junk. Pruned when
# searching for path suggestions so we never point the agent at a build copy,
# a trashed copy, or a dependency, and never trigger an unbounded scan.
_SUGGEST_PRUNE_DIRS = {
    'build', 'dist', 'node_modules', '.git', '__pycache__', '.ipynb_checkpoints',
    '.venv', 'venv', '.mypy_cache', '.pytest_cache', '.ruff_cache', '.cache',
    'site-packages', '.local', '.trash-0', 'aeon_output',
}
_SUGGEST_MAX_HITS = 8
_SUGGEST_MAX_FILES_SCANNED = 20000


def _project_root():
    """Resolve the canonical project root, falling back to cwd. Never raises."""
    try:
        from ..core.paths import PROJECT_ROOT
        root = str(PROJECT_ROOT)
        if root and os.path.isdir(root):
            return root
    except Exception:
        pass
    return os.getcwd()


def _suggest_paths(missing_path: str) -> str:
    """Return a 'did you mean' hint for a not-found path, or '' if nothing useful.

    Searches ONLY the project root (bounded, prune-listed) for files sharing the
    requested basename. Fully exception-guarded: any failure yields '' so this can
    never turn a normal not-found into a tool crash.
    """
    try:
        target = os.path.basename(missing_path.rstrip('/'))
        if not target:
            return ''
        root = _project_root()
        target_lower = target.lower()

        exact, fuzzy = [], []
        scanned = 0
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune heavy/duplicate/junk dirs in-place so os.walk skips them entirely.
            dirnames[:] = [
                d for d in dirnames
                if d not in _SUGGEST_PRUNE_DIRS and not d.endswith('.egg-info')
            ]
            for fn in filenames:
                scanned += 1
                if scanned > _SUGGEST_MAX_FILES_SCANNED:
                    break
                if fn == target:
                    exact.append(os.path.join(dirpath, fn))
                elif fn.lower() == target_lower:
                    exact.append(os.path.join(dirpath, fn))
                elif target_lower in fn.lower() or fn.lower() in target_lower:
                    if len(fuzzy) < _SUGGEST_MAX_HITS:
                        fuzzy.append(os.path.join(dirpath, fn))
            if scanned > _SUGGEST_MAX_FILES_SCANNED:
                break
            if len(exact) >= _SUGGEST_MAX_HITS:
                break

        hits = exact if exact else fuzzy
        if not hits:
            return (f"\nNo file named '{target}' exists anywhere in the project "
                    f"({root}). The path you used does not exist - re-check the "
                    f"Project Tree in your context rather than retrying variants of it.")

        hits = hits[:_SUGGEST_MAX_HITS]
        lines = "\n".join(f"  - {p}" for p in hits)
        kind = "exact filename match" if exact else "similarly named file"
        return (f"\nNo file exists at that path, but the following {kind}(es) were "
                f"found in the project - did you mean one of these?\n{lines}\n"
                f"Use the exact path above. Do NOT keep retrying the original path.")
    except Exception:
        return ''


def _atomic_write(abs_path: str, content, binary: bool = False):
    """Write content to abs_path atomically (temp file in the same dir + os.replace)."""
    parent = os.path.dirname(abs_path) or '.'
    os.makedirs(parent, exist_ok=True)

    orig_mode = None
    if os.path.exists(abs_path):
        try:
            orig_mode = os.stat(abs_path).st_mode
        except OSError:
            orig_mode = None

    fd, tmp = tempfile.mkstemp(dir=parent, prefix='.aeon_tmp_')
    try:
        if binary:
            with os.fdopen(fd, 'wb') as f:
                f.write(content)
        else:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
        os.replace(tmp, abs_path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise

    try:
        os.chmod(abs_path, orig_mode if orig_mode is not None else 0o644)
    except OSError:
        pass


def _edit_failures(worker):
    """Per-file failure counter, stored on the worker so str_replace and write_file
    can coordinate escalation/reset across separate tool instances."""
    if worker is None:
        return {}
    if not hasattr(worker, '_edit_failures'):
        worker._edit_failures = {}
    return worker._edit_failures


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
            return f'Error: File not found: {file_path}{_suggest_paths(file_path)}'
        if os.path.isdir(abs_path):
            return f'Error: {file_path} is a directory. Refer to the Project Tree in your system context to see files, then open a specific file.'

        if self.worker.is_file_open(file_path) or self.worker.is_file_open(abs_path):
            return (
                f"NO-OP: '{file_path}' is ALREADY in your OPEN FILES section with its full, "
                f"current content. This call changed nothing. Read it where it is and make your "
                f"next action advance the task — do NOT call open_file on it again."
            )

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

        lines = content.splitlines()
        numbered_lines = [f"{i}: {line}" for i, line in enumerate(lines, 1)]
        display_content = '\n'.join(numbered_lines)

        return f"File '{file_path}' opened in working memory. ({slots_used} files open)\n\n---\n{display_content}"


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
    """Targeted SEARCH/REPLACE edits with exact, whitespace-tolerant, and fuzzy
    matching. After repeated failures on a file it hard-escalates to write_file
    instead of letting the model loop forever on a mismatching SEARCH block."""

    MAX_FAILURES_BEFORE_ESCALATION = 2

    def __init__(self, worker):
        super().__init__(
            name='str_replace',
            description=TOOL_DESC_STR_REPLACE
        )
        self.worker = worker

    def _normalize_whitespace(self, text: str) -> str:
        lines = text.splitlines()
        return '\n'.join(line.rstrip() for line in lines)

    def _find_fuzzy_match(self, content: str, search_str: str) -> tuple:
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

    def _apply_single_replace(self, file_path, content, old_str, new_str):
        """Apply one replacement. Returns (new_content, method_used, error)."""
        stripped_old = old_str.strip()

        line_range_match = re.match(r'^L(\d+)(?:-L(\d+))?$', stripped_old)
        if line_range_match:
            try:
                start_line = int(line_range_match.group(1))
                end_line = int(line_range_match.group(2)) if line_range_match.group(2) else start_line
                content_lines = content.splitlines(keepends=True)
                num_lines = len(content_lines)
                if start_line < 1 or start_line > num_lines:
                    return content, None, f'Error: Start line {start_line} is out of bounds for {file_path} (1-{num_lines}).'
                if end_line < 1 or end_line > num_lines:
                    return content, None, f'Error: End line {end_line} is out of bounds for {file_path} (1-{num_lines}).'
                if start_line > end_line:
                    return content, None, f'Error: Start line {start_line} is greater than end line {end_line} in {file_path}.'
                prefix = "".join(content_lines[:start_line - 1])
                suffix = "".join(content_lines[end_line:])
                new_content = prefix + new_str + suffix
                if new_content == content:
                    return content, None, f'Warning: Replacement produced identical content in {file_path}. No changes written.'
                return new_content, f'line-range (L{start_line}-L{end_line})', None
            except Exception as e:
                return content, None, f'Error processing line range: {e}'

        processed_old_str = old_str
        lines = old_str.splitlines(keepends=True)
        modified_lines = []
        changed = False
        for line in lines:
            m = re.match(r'^(\d+):\s?', line)
            if m:
                modified_lines.append(line[m.end():])
                changed = True
            else:
                modified_lines.append(line)
        if changed:
            processed_old_str = ''.join(modified_lines)

        match_method = 'exact'
        matched_text = None

        count = content.count(processed_old_str)
        if count == 1:
            matched_text = processed_old_str
        elif count > 1:
            return content, None, (
                f'Error: The SEARCH block matched {count} times in {file_path}. '
                f'It must be unique. Add more surrounding context to narrow the match.'
            )
        else:
            count_orig = content.count(old_str)
            if count_orig == 1:
                matched_text = old_str
            elif count_orig > 1:
                return content, None, (
                    f'Error: The SEARCH block matched {count_orig} times in {file_path}. '
                    f'It must be unique. Add more surrounding context to narrow the match.'
                )

        if matched_text is None:
            norm_content = self._normalize_whitespace(content)
            norm_search = self._normalize_whitespace(processed_old_str)
            norm_count = norm_content.count(norm_search)
            if norm_count == 1:
                norm_pos = norm_content.find(norm_search)
                norm_lines_before = norm_content[:norm_pos].count('\n')
                search_line_count = processed_old_str.count('\n') + (0 if processed_old_str.endswith('\n') else 1)
                original_lines = content.splitlines(keepends=True)
                start_line_idx = norm_lines_before
                end_line_idx = start_line_idx + search_line_count
                if end_line_idx <= len(original_lines):
                    matched_text = ''.join(original_lines[start_line_idx:end_line_idx])
                    match_method = 'whitespace-normalized'
            elif norm_count > 1:
                return content, None, (
                    f'Error: SEARCH block matched {norm_count} times after whitespace normalization in {file_path}. '
                    f'Add more surrounding context to narrow the match.'
                )

        if matched_text is None:
            fuzzy_match, score = self._find_fuzzy_match(content, old_str)
            if fuzzy_match is not None:
                matched_text = fuzzy_match
                match_method = f'fuzzy ({score:.0%})'
            else:
                search_preview = old_str[:200] + ('...' if len(old_str) > 200 else '')
                first_line_of_search = old_str.split('\n')[0].strip()
                diagnostic = ''
                content_lines = content.splitlines()
                for i, line in enumerate(content_lines):
                    if first_line_of_search and first_line_of_search[:30] in line:
                        start = max(0, i - 1)
                        end = min(len(content_lines), i + 4)
                        snippet = '\n'.join(
                            f'  L{start + j + 1}: {content_lines[start + j]}'
                            for j in range(end - start)
                        )
                        diagnostic = (
                            '\nNearest region in the file:\n' + snippet +
                            '\nCompare it carefully against your SEARCH block (watch for '
                            'differing quotes, escapes, or whitespace).'
                        )
                        break
                return content, None, (
                    f'Error: Could not find a match for the SEARCH block in {file_path}. '
                    f'Best fuzzy score was {score:.0%} (threshold {FUZZY_MATCH_THRESHOLD:.0%}). '
                    f'Searched for: {search_preview!r}{diagnostic}'
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

    def _escalation_note(self, file_path, count):
        if count >= self.MAX_FAILURES_BEFORE_ESCALATION:
            return (
                f'\n\n*** str_replace has failed {count} time(s) on {file_path} and is now '
                f'LOCKED for this file. You MUST use write_file to rewrite the ENTIRE file '
                f'with the corrected content. Open the file first to get its current content. ***'
            )
        return (
            f'\n\n(Attempt {count}/{self.MAX_FAILURES_BEFORE_ESCALATION} before str_replace is '
            f'disabled for this file and you must switch to write_file.)'
        )

    def execute(self, file_path: str, patch: str = None, old_str: str = None, new_str: str = '') -> str:
        if not file_path:
            return 'Error: file_path parameter is required.'
        if not patch and not old_str:
            return 'Error: Must provide either a patch or an old_str parameter.'
        if new_str is None:
            new_str = ''

        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return f'Error: File not found: {file_path}{_suggest_paths(file_path)}'
        if os.path.isdir(abs_path):
            return f'Error: {file_path} is a directory, not a file.'

        failures = _edit_failures(self.worker)

        if failures.get(abs_path, 0) >= self.MAX_FAILURES_BEFORE_ESCALATION:
            return (
                f'str_replace is disabled for {file_path} after '
                f'{failures[abs_path]} consecutive failed attempts.\n'
                f'*** Use write_file to rewrite the ENTIRE file. *** '
                f'Open the file to read its current content, then write_file with the full corrected content.'
            )

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
                failures[abs_path] = failures.get(abs_path, 0) + 1
                msg = (
                    'Error: Could not parse any SEARCH/REPLACE blocks. Use this exact format '
                    '(inside a content block):\n'
                    '<<<< SEARCH\n<existing code>\n====\n<replacement code>\n>>>> REPLACE'
                )
                return msg + self._escalation_note(file_path, failures[abs_path])

            for s_str, r_str in blocks:
                if not s_str:
                    continue
                new_c, method, err = self._apply_single_replace(file_path, current_content, s_str, r_str)
                if err and err.startswith('Error'):
                    failures[abs_path] = failures.get(abs_path, 0) + 1
                    return err + self._escalation_note(file_path, failures[abs_path])
                if method:
                    methods_used.append(method)
                current_content = new_c
        else:
            new_c, method, err = self._apply_single_replace(file_path, current_content, old_str, new_str)
            if err and err.startswith('Error'):
                failures[abs_path] = failures.get(abs_path, 0) + 1
                return err + self._escalation_note(file_path, failures[abs_path])
            if method:
                methods_used.append(method)
            current_content = new_c

        if current_content == content:
            return f"Warning: No changes were made to {file_path}. Content is identical."

        try:
            _atomic_write(abs_path, current_content)
        except Exception as e:
            return f'Error writing file: {type(e).__name__}: {e}'

        failures.pop(abs_path, None)

        if self.worker.is_file_open(abs_path) or self.worker.is_file_open(file_path):
            self.worker.update_open_file(abs_path, current_content)

        method_str = ", ".join(dict.fromkeys(methods_used)) if methods_used else "exact"
        block_count = len(methods_used) if patch else 1
        return f"Successfully applied {block_count} edit block(s) to {file_path} (matched via {method_str})."


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
        if isinstance(content, str) and content.startswith('base64:'):
            try:
                decoded = base64.b64decode(content[7:])
                try:
                    content_decoded = decoded.decode('utf-8')
                except UnicodeDecodeError:
                    content_decoded = decoded
                    is_binary = True
            except Exception as e:
                return f'Error decoding base64 content: {e}'
        else:
            content_decoded = content

        abs_path = os.path.abspath(file_path)
        if os.path.isdir(abs_path):
            return f'Error: {file_path} is a directory, not a file.'

        try:
            _atomic_write(abs_path, content_decoded, binary=is_binary)
        except PermissionError:
            return f'Error: Permission denied writing to {file_path}'
        except Exception as e:
            return f'Error writing file: {type(e).__name__}: {e}'

        self.worker.close_file(file_path)
        self.worker.close_file(abs_path)

        _edit_failures(self.worker).pop(abs_path, None)

        if is_binary:
            return f'Successfully wrote {file_path} ({len(content_decoded):,} bytes, binary).'
        line_count = content_decoded.count('\n') + 1 if content_decoded else 0
        return f'Successfully wrote {file_path} ({line_count:,} lines).'
