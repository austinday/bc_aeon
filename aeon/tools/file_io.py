from .base import BaseTool
import os
import json
import base64
import difflib
import re
import hashlib
from ..core.agent_protocol import SideEffect, ToolResult, ToolStatus
from ..core.prompts import (
    TOOL_DESC_OPEN_FILE,
    TOOL_DESC_CLOSE_FILE,
    TOOL_DESC_WRITE_FILE,
    TOOL_DESC_STR_REPLACE,
)
from .analyzers import FileAnalyzer
from ..core.workspace_files import (
    WorkspaceFileBoundary,
    WorkspaceFilePath,
    WorkspacePathError,
)

# Max characters before rejecting full-content files
MAX_FILE_READ_SIZE = 250000
MAX_FILE_RECEIPT_BYTES = 8 * 1024 * 1024

# Fuzzy edits are opt-in and deliberately strict. Exact/unique matching remains
# the default because a plausible edit in the wrong function is worse than a
# clean refusal.
FUZZY_MATCH_THRESHOLD = 0.92

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


def _project_root(bound_root=None):
    """Resolve the workspace root for path suggestions. Never raises.

    File operations are workspace-relative, so 'did you mean' hints must search
    the user's current workspace, not the aeon install/source tree.
    """
    if bound_root is not None:
        return str(bound_root)
    try:
        from ..core.paths import get_workspace_root
        root = str(get_workspace_root())
        if root and os.path.isdir(root):
            return root
    except Exception:
        pass
    return os.getcwd()


def _suggest_paths(missing_path: str, *, root=None) -> str:
    """Return a 'did you mean' hint for a not-found path, or '' if nothing useful.

    Searches ONLY the project root (bounded, prune-listed) for files sharing the
    requested basename. Fully exception-guarded: any failure yields '' so this can
    never turn a normal not-found into a tool crash.
    """
    try:
        target = os.path.basename(missing_path.rstrip('/'))
        if not target:
            return ''
        root = _project_root(root)
        target_lower = target.lower()

        exact, fuzzy = [], []
        all_basenames = {}  # basename -> full path, for an edit-distance fallback
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
                all_basenames.setdefault(fn, os.path.join(dirpath, fn))
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
            # Edit-distance fallback catches transposition/typo cases that neither
            # an exact nor a substring match would find (e.g. 'wrokflow.py').
            import difflib
            close = difflib.get_close_matches(target, list(all_basenames), n=_SUGGEST_MAX_HITS, cutoff=0.7)
            hits = [all_basenames[b] for b in close]
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


def _secure_sha256(
    boundary: WorkspaceFileBoundary,
    path: WorkspaceFilePath,
) -> str | None:
    with boundary.open_for_read(path) as opened:
        return _file_sha256(
            opened.proc_path,
            expected_identity=opened.identity,
        )


def _file_sha256(
    path: str,
    *,
    max_bytes: int = MAX_FILE_RECEIPT_BYTES,
    expected_identity: tuple[int, int, int, int] | None = None,
) -> str | None:
    """Return an exact small-file receipt, or ``None`` above the CPU bound."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        initial = os.fstat(handle.fileno())
        initial_identity = (
            int(initial.st_dev),
            int(initial.st_ino),
            int(initial.st_size),
            int(initial.st_mtime_ns),
        )
        if expected_identity is not None and initial_identity != expected_identity:
            raise ValueError("file changed between analysis and receipt")
        if initial.st_size > max_bytes:
            return None
        total = 0
        for chunk in iter(lambda: handle.read(min(1024 * 1024, max_bytes - total + 1)), b""):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("file grew beyond the bounded SHA256 receipt limit")
            digest.update(chunk)
        final = os.fstat(handle.fileno())
        if (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
        ) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
        ):
            raise ValueError("file changed while computing its SHA256 receipt")
    return digest.hexdigest()


def _unified_diff(old: str, new: str, path: str, max_lines: int = 80) -> str:
    """Return a compact unified diff of an applied edit, or '' if nothing useful.

    Gives the model an objective view of WHAT changed and WHERE, so a fuzzy or
    whitespace-normalized match that landed in the wrong region is visible
    instead of being hidden behind a bare 'success' message. Bounded so a large
    rewrite never floods the context: past max_lines it degrades to a +adds/-dels
    summary line.
    """
    try:
        if old == new:
            return ''
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f'a/{path}', tofile=f'b/{path}', lineterm='', n=2,
        ))
        if not diff:
            return ''
        adds = sum(1 for d in diff if d.startswith('+') and not d.startswith('+++'))
        dels = sum(1 for d in diff if d.startswith('-') and not d.startswith('---'))
        if len(diff) > max_lines:
            body = '\n'.join(diff[:max_lines])
            return (f"\n--- DIFF (+{adds}/-{dels} lines, truncated to first {max_lines}) ---\n"
                    f"{body}\n... [diff truncated] ...")
        return f"\n--- DIFF (+{adds}/-{dels} lines) ---\n" + '\n'.join(diff)
    except Exception:
        return ''


def _syntax_error(abs_path: str, content) -> str:
    """Return a parse error before replacement, or ``''`` when acceptable."""
    if not isinstance(content, str):
        return ''
    ext = os.path.splitext(abs_path)[1].lower()
    try:
        if ext == '.py':
            try:
                compile(content, abs_path, 'exec')
            except SyntaxError as e:
                return f"invalid Python: {e.msg} (line {e.lineno})"
        elif ext == '.json':
            try:
                json.loads(content)
            except ValueError as e:
                return f"invalid JSON: {e}"
        elif ext in ('.yaml', '.yml'):
            try:
                import yaml  # PyYAML is optional; skip the check if unavailable
            except ImportError:
                return ''
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                first = str(e).splitlines()[0] if str(e) else 'parse error'
                return f"invalid YAML: {first}"
    except Exception:
        return ''
    return ''


def _protected_guard(abs_path: str):
    """Return a refusal message if abs_path is a protected self-improvement
    guardrail, else None. Fully guarded so a missing module never breaks editing."""
    try:
        from ..core.protected import guard
        return guard(abs_path)
    except Exception:
        return None


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
        self._files = WorkspaceFileBoundary.from_worker(worker)

    @staticmethod
    def _result(
        status: ToolStatus,
        summary: str,
        *,
        error_code: str = "",
        retryable: bool = False,
    ) -> ToolResult:
        """Build an explicit receipt without interpreting observed file text."""

        return ToolResult(
            tool_name="open_file",
            status=status,
            changed=False,
            summary=summary,
            error_code=error_code,
            retryable=retryable,
            side_effect=SideEffect.READ_ONLY,
        )

    @classmethod
    def _error_result(cls, message: str) -> ToolResult:
        if str(message).startswith("COMMAND BLOCKED:"):
            return cls._result(
                ToolStatus.BLOCKED,
                str(message),
                error_code="tool_blocked",
            )
        return cls._result(
            ToolStatus.FAILED,
            str(message),
            error_code="tool_failed",
            # Workspace path/shape/read failures are deterministic until some
            # other action changes the file. Retrying the same call in-place
            # only creates a loop; transport-backed readers use typed transient
            # codes and the Worker affords them one exact replay.
            retryable=False,
        )

    def execute(self, file_path: str) -> ToolResult:
        if not file_path:
            return self._error_result('Error: file_path parameter is required.')

        try:
            bound_path = self._files.bind(file_path)
        except WorkspacePathError as exc:
            return self._error_result(str(exc))
        abs_path = str(bound_path.absolute)

        if self.worker.is_file_open(file_path) or self.worker.is_file_open(abs_path):
            return self._result(
                ToolStatus.NO_CHANGE,
                (
                    f"NO-OP: '{file_path}' is ALREADY in your OPEN FILES section with its full, "
                    f"current content. This call changed nothing. Read it where it is and make your "
                    f"next action advance the task — do NOT call open_file on it again."
                ),
                error_code="no_change",
            )

        try:
            with self._files.open_for_read(bound_path) as opened:
                analyzer = FileAnalyzer(opened.proc_path, display_path=abs_path)
                result = analyzer.analyze()
                summary_type = result.get('summary_type', '')

                if summary_type == 'opaque_binary':
                    return self._result(
                        ToolStatus.OK,
                        f"File '{file_path}' is a binary file that cannot be displayed. Use a script to analyze it.",
                    )
                if summary_type == 'error':
                    return self._error_result(
                        f"Error reading file: {result.get('error_message', 'Unknown error')}"
                    )

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
                    lines = content.splitlines()
                    head = lines[:150]
                    tail = lines[-50:] if len(lines) > 200 else []
                    preview_parts = [f"{i}: {ln}" for i, ln in enumerate(head, 1)]
                    if tail:
                        tail_start = len(lines) - len(tail) + 1
                        preview_parts.append(f"... [{len(lines) - len(head) - len(tail):,} lines omitted] ...")
                        preview_parts.extend(f"{tail_start + j}: {ln}" for j, ln in enumerate(tail))
                    preview = "\n".join(preview_parts)
                    return self._result(
                        ToolStatus.OK,
                        (
                            f"File '{file_path}' is too large to load fully ({len(content):,} chars, "
                            f"{len(lines):,} lines; limit {MAX_FILE_READ_SIZE:,} chars), so it was NOT "
                            f"added to OPEN FILES. Below is a head+tail PREVIEW. For targeted access use "
                            f"run_command (e.g. grep -n PATTERN, sed -n 'START,ENDp', tail -n N).\n\n"
                            f"--- PREVIEW: {file_path} ---\n{preview}"
                        ),
                    )

                if not analyzer.identity_is_current() or analyzer.file_identity != opened.identity:
                    return self._error_result(
                        f"Error reading file: '{file_path}' changed while it was being analyzed; "
                        "retry once the writer is finished."
                    )
                receipt = _file_sha256(
                    opened.proc_path,
                    max_bytes=MAX_FILE_RECEIPT_BYTES,
                    expected_identity=opened.identity,
                )
                if not self._files.identity_is_current(bound_path, opened.identity):
                    return self._error_result(
                        f"Error reading file: '{file_path}' changed while it was being analyzed; "
                        "retry once the writer is finished."
                    )
        except WorkspacePathError as exc:
            message = str(exc)
            if message.startswith("Error: File not found:"):
                message += _suggest_paths(file_path, root=self._files.root)
            return self._error_result(message)
        except (OSError, ValueError) as exc:
            return self._error_result(
                f"Error reading file: could not establish a stable receipt: {exc}"
            )
        except Exception as exc:
            return self._error_result(
                f'Error analyzing file: {type(exc).__name__}: {exc}'
            )

        self.worker.update_open_file(abs_path, content)
        slots_used = len(self.worker.open_files)

        lines = content.splitlines()
        numbered_lines = [f"{i}: {line}" for i, line in enumerate(lines, 1)]
        display_content = '\n'.join(numbered_lines)

        receipt_line = (
            f"SHA256: {receipt}"
            if receipt is not None
            else (
                "SHA256: omitted (input exceeds the "
                f"{MAX_FILE_RECEIPT_BYTES:,}-byte receipt CPU limit)"
            )
        )
        return self._result(
            ToolStatus.OK,
            (
                f"File '{file_path}' opened in working memory. ({slots_used} files open)\n"
                f"{receipt_line}\n\n---\n{display_content}"
            ),
        )


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
    """Hash-bound targeted replacement with optional high-confidence fuzzing."""

    MAX_FAILURES_BEFORE_ESCALATION = 2

    def __init__(self, worker):
        super().__init__(
            name='str_replace',
            description=TOOL_DESC_STR_REPLACE
        )
        self.worker = worker
        self._files = WorkspaceFileBoundary.from_worker(worker)

    @staticmethod
    def _parameter_properties() -> dict:
        return {
            "file_path": {"type": "string", "minLength": 1},
            "expected_sha256": {
                "type": "string",
                "pattern": r"^[0-9a-fA-F]{64}$",
            },
            "allow_fuzzy": {"type": "boolean"},
        }

    def parameter_schema(self) -> dict:
        """Require one complete, unambiguous edit form during decoding."""

        patch_properties = self._parameter_properties()
        patch_properties["patch"] = {"type": "string", "minLength": 1}
        exact_properties = self._parameter_properties()
        exact_properties.update(
            {
                "old_str": {"type": "string", "minLength": 1},
                "new_str": {"type": "string"},
            }
        )
        # Full object branches are intentional.  The pinned xgrammar runtime
        # does not reliably intersect partial oneOf branches with sibling object
        # properties, so each branch independently carries every constraint.
        return {
            "oneOf": [
                {
                    "type": "object",
                    "properties": patch_properties,
                    "required": ["file_path", "expected_sha256", "patch"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": exact_properties,
                    "required": ["file_path", "expected_sha256", "old_str"],
                    "additionalProperties": False,
                },
            ]
        }

    @staticmethod
    def _parsed_patch_blocks(patch: str) -> list[tuple[str, str]]:
        return re.findall(
            r'<<<<\s*SEARCH\n?(.*?)\n?====\n?(.*?)\n?>>>>\s*REPLACE',
            patch,
            re.DOTALL,
        )

    def validate_parameters(self, parameters) -> str:
        """Mirror the cross-field schema before any file is opened."""

        if not isinstance(parameters, dict):
            return "parameters must be a JSON object"
        allowed = {
            "file_path",
            "patch",
            "old_str",
            "new_str",
            "expected_sha256",
            "allow_fuzzy",
        }
        unknown = sorted(set(parameters) - allowed)
        if unknown:
            return "unknown parameter(s): " + ", ".join(unknown)
        missing = [
            name
            for name in ("file_path", "expected_sha256")
            if name not in parameters
        ]
        if missing:
            return "missing required parameter(s): " + ", ".join(missing)

        string_fields = {
            "file_path",
            "patch",
            "old_str",
            "new_str",
            "expected_sha256",
        }
        wrong_types = sorted(
            name
            for name in string_fields.intersection(parameters)
            if not isinstance(parameters[name], str)
        )
        if "allow_fuzzy" in parameters and not isinstance(
            parameters["allow_fuzzy"], bool
        ):
            wrong_types.append("allow_fuzzy")
        if wrong_types:
            return "wrong JSON type for parameter(s): " + ", ".join(
                sorted(wrong_types)
            )
        if not parameters["file_path"].strip():
            return "file_path must be non-empty text"
        if not re.fullmatch(
            r"[0-9a-fA-F]{64}",
            parameters["expected_sha256"],
        ):
            return "expected_sha256 must be a 64-character hexadecimal digest"

        if "patch" in parameters and "old_str" in parameters:
            return "provide exactly one edit form: patch or old_str"
        has_patch = bool(parameters.get("patch", "").strip())
        has_old = bool(parameters.get("old_str", "").strip())
        if has_patch == has_old:
            return "provide exactly one non-empty edit form: patch or old_str"
        if has_patch:
            if "new_str" in parameters:
                return "new_str belongs to the old_str edit form, not patch"
            blocks = self._parsed_patch_blocks(parameters["patch"])
            if not blocks or any(not search for search, _replacement in blocks):
                return (
                    "patch must contain at least one non-empty "
                    "<<<< SEARCH ... ==== ... >>>> REPLACE block"
                )
        return ""

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

    def _match_locations(self, content, needle, max_show=6):
        """Return ' Matches at lines: ...' listing the 1-based start line of each
        occurrence of needle in content, so the model can see which copy is which
        and add the right disambiguating context."""
        if not needle:
            return ''
        lines = []
        start = 0
        while True:
            pos = content.find(needle, start)
            if pos == -1:
                break
            lines.append(content.count('\n', 0, pos) + 1)
            start = pos + 1
            if len(lines) > max_show:
                break
        if not lines:
            return ''
        shown = ', '.join(str(n) for n in lines[:max_show])
        more = '' if len(lines) <= max_show else f' (+{len(lines) - max_show} more)'
        return f' Matches start at line(s): {shown}{more}.'

    def _apply_single_replace(self, file_path, content, old_str, new_str, allow_fuzzy=False):
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
                f'{self._match_locations(content, processed_old_str)}'
            )
        else:
            count_orig = content.count(old_str)
            if count_orig == 1:
                matched_text = old_str
            elif count_orig > 1:
                return content, None, (
                    f'Error: The SEARCH block matched {count_orig} times in {file_path}. '
                    f'It must be unique. Add more surrounding context to narrow the match.'
                    f'{self._match_locations(content, old_str)}'
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

        if matched_text is None and allow_fuzzy:
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

        if matched_text is None:
            search_preview = old_str[:200] + ('...' if len(old_str) > 200 else '')
            return content, None, (
                f'Error: Could not find one exact or whitespace-normalized match in {file_path}. '
                f'Searched for: {search_preview!r}. Re-open the file and copy a unique current block; '
                f'or explicitly set allow_fuzzy=true for a >= {FUZZY_MATCH_THRESHOLD:.0%} unique match.'
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
                f'LOCKED for this file. Re-open the current file, diagnose why the expected '
                f'block differs, and choose a new exact approach. Do not blindly overwrite it. ***'
            )
        return (
            f'\n\n(Attempt {count}/{self.MAX_FAILURES_BEFORE_ESCALATION} before str_replace is '
            f'disabled for this file. Re-open it and use a more specific exact block.)'
        )

    def execute(
        self,
        file_path: str,
        patch: str = None,
        old_str: str = None,
        new_str: str = '',
        expected_sha256: str = None,
        allow_fuzzy: bool = False,
    ) -> str:
        if not file_path:
            return 'Error: file_path parameter is required.'
        if not patch and not old_str:
            return 'Error: Must provide either a patch or an old_str parameter.'
        if new_str is None:
            new_str = ''

        try:
            bound_path = self._files.bind(file_path)
        except WorkspacePathError as exc:
            return str(exc)
        abs_path = str(bound_path.absolute)

        blocked = _protected_guard(abs_path)
        if blocked:
            return blocked

        failures = _edit_failures(self.worker)

        if failures.get(abs_path, 0) >= self.MAX_FAILURES_BEFORE_ESCALATION:
            return (
                f'str_replace is disabled for {file_path} after '
                f'{failures[abs_path]} consecutive failed attempts.\n'
                f'Open the current file, diagnose the mismatch, and choose a new exact edit. '
                f'Do not bypass this failure with a blind full-file overwrite.'
            )

        try:
            with self._files.open_for_read(bound_path) as opened:
                with open(opened.proc_path, 'r', encoding='utf-8', errors='replace') as handle:
                    content = handle.read()
                current_sha256 = _file_sha256(
                    opened.proc_path,
                    expected_identity=opened.identity,
                )
                source_identity = opened.identity
                if not self._files.identity_is_current(bound_path, source_identity):
                    return (
                        f"COMMAND BLOCKED: '{file_path}' changed while it was read. "
                        "Re-open it and retry from current content."
                    )
        except WorkspacePathError as exc:
            message = str(exc)
            if message.startswith("Error: File not found:"):
                message += _suggest_paths(file_path, root=self._files.root)
            return message
        except Exception as exc:
            return f'Error reading file: {type(exc).__name__}: {exc}'
        if not expected_sha256:
            return (
                f"COMMAND BLOCKED: expected_sha256 is required for editing existing file "
                f"'{file_path}'. Re-open it and use the SHA256 receipt."
            )
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(expected_sha256)):
            return "Error: expected_sha256 must be a 64-character hexadecimal SHA-256 digest."
        if current_sha256.lower() != str(expected_sha256).lower():
            return (
                f"COMMAND BLOCKED: '{file_path}' changed since it was observed "
                f"(expected {str(expected_sha256)[:12]}…, current {current_sha256[:12]}…). "
                "Re-open it and build the edit from current content."
            )

        current_content = content
        methods_used = []

        if patch:
            blocks = self._parsed_patch_blocks(patch)
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
                new_c, method, err = self._apply_single_replace(
                    file_path, current_content, s_str, r_str, allow_fuzzy=allow_fuzzy
                )
                if err and err.startswith('Error'):
                    failures[abs_path] = failures.get(abs_path, 0) + 1
                    return err + self._escalation_note(file_path, failures[abs_path])
                if method:
                    methods_used.append(method)
                current_content = new_c
        else:
            new_c, method, err = self._apply_single_replace(
                file_path, current_content, old_str, new_str, allow_fuzzy=allow_fuzzy
            )
            if err and err.startswith('Error'):
                failures[abs_path] = failures.get(abs_path, 0) + 1
                return err + self._escalation_note(file_path, failures[abs_path])
            if method:
                methods_used.append(method)
            current_content = new_c

        if current_content == content:
            return f"Warning: No changes were made to {file_path}. Content is identical."

        syntax_error = _syntax_error(abs_path, current_content)
        if syntax_error:
            return (
                f"Error: Refusing to replace {file_path}; proposed content has {syntax_error}. "
                "The original file is unchanged."
            )
        if not self._files.identity_is_current(bound_path, source_identity):
            return (
                f"COMMAND BLOCKED: '{file_path}' changed concurrently before write. "
                "The original was not overwritten; re-open and retry from current content."
            )

        try:
            self._files.atomic_write(
                bound_path,
                current_content,
                binary=False,
                expected_identity=source_identity,
            )
            new_sha256 = _secure_sha256(self._files, bound_path)
        except WorkspacePathError as exc:
            return str(exc)
        except Exception as exc:
            return f'Error writing file: {type(exc).__name__}: {exc}'

        failures.pop(abs_path, None)

        if self.worker.is_file_open(abs_path) or self.worker.is_file_open(file_path):
            self.worker.update_open_file(abs_path, current_content)

        method_str = ", ".join(dict.fromkeys(methods_used)) if methods_used else "exact"
        block_count = len(methods_used) if patch else 1
        diff_str = _unified_diff(content, current_content, os.path.basename(file_path))
        return (f"Successfully applied {block_count} edit block(s) to {file_path} "
                f"(matched via {method_str}). New SHA256: {new_sha256}.{diff_str}")


class WriteFileTool(BaseTool):
    def __init__(self, worker):
        super().__init__(
            name='write_file',
            description=TOOL_DESC_WRITE_FILE
        )
        self.worker = worker
        self._files = WorkspaceFileBoundary.from_worker(worker)

    def execute(self, file_path: str, content: str, expected_sha256: str = None) -> str:
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

        try:
            bound_path = self._files.bind(file_path)
        except WorkspacePathError as exc:
            return str(exc)
        abs_path = str(bound_path.absolute)

        blocked = _protected_guard(abs_path)
        if blocked:
            return blocked

        # Capture prior text content (bounded) so we can show the model a diff of
        # what its overwrite actually changed. New file -> existed stays False.
        existed = False
        old_content = None
        old_sha256 = None
        source_identity = None
        try:
            with self._files.open_for_read(bound_path) as opened:
                existed = True
                source_identity = opened.identity
                old_sha256 = _file_sha256(
                    opened.proc_path,
                    expected_identity=source_identity,
                )
                if not is_binary and source_identity[2] <= MAX_FILE_READ_SIZE:
                    with open(opened.proc_path, 'r', encoding='utf-8', errors='replace') as handle:
                        old_content = handle.read()
                if not self._files.identity_is_current(bound_path, source_identity):
                    return (
                        f"COMMAND BLOCKED: '{file_path}' changed while it was read. "
                        "Re-open it and retry from current content."
                    )
        except WorkspacePathError as exc:
            if not str(exc).startswith("Error: File not found:"):
                return str(exc)
        if existed:
            if not expected_sha256:
                return (
                    f"COMMAND BLOCKED: write_file will not blindly overwrite existing file "
                    f"'{file_path}'. Re-open it and pass its expected_sha256, or use str_replace."
                )
            if not re.fullmatch(r"[0-9a-fA-F]{64}", str(expected_sha256)):
                return "Error: expected_sha256 must be a 64-character hexadecimal SHA-256 digest."
            if old_sha256.lower() != str(expected_sha256).lower():
                return (
                    f"COMMAND BLOCKED: '{file_path}' changed since observation "
                    f"(expected {str(expected_sha256)[:12]}…, current {old_sha256[:12]}…)."
                )
        elif expected_sha256:
            return (
                f"COMMAND BLOCKED: '{file_path}' no longer exists, so the supplied "
                "observation receipt cannot authorize creating a different file."
            )

        if existed and not is_binary and old_content is not None and old_content == content_decoded:
            return f"NO CHANGE: {file_path} already has identical content; nothing was written."

        if not is_binary:
            syntax_error = _syntax_error(abs_path, content_decoded)
            if syntax_error:
                return (
                    f"Error: Refusing to write {file_path}; proposed content has {syntax_error}. "
                    "No file was changed."
                )
        if existed and not self._files.identity_is_current(bound_path, source_identity):
            return (
                f"COMMAND BLOCKED: '{file_path}' changed concurrently before write. "
                "No overwrite was attempted."
            )

        try:
            self._files.atomic_write(
                bound_path,
                content_decoded,
                binary=is_binary,
                expected_identity=source_identity,
            )
            new_sha256 = _secure_sha256(self._files, bound_path)
        except PermissionError:
            return f'Error: Permission denied writing to {file_path}'
        except WorkspacePathError as exc:
            return str(exc)
        except Exception as exc:
            return f'Error writing file: {type(exc).__name__}: {exc}'

        self.worker.close_file(file_path)
        self.worker.close_file(abs_path)

        _edit_failures(self.worker).pop(abs_path, None)

        if is_binary:
            return (
                f'Successfully wrote {file_path} ({len(content_decoded):,} bytes, binary). '
                f'New SHA256: {new_sha256}.'
            )

        line_count = content_decoded.count('\n') + 1 if content_decoded else 0
        if not existed:
            verb = f'Created {file_path} ({line_count:,} lines)'
            diff_str = ''
        else:
            verb = f'Overwrote {file_path} ({line_count:,} lines)'
            diff_str = (_unified_diff(old_content, content_decoded, os.path.basename(file_path))
                        if old_content is not None else '')
        return f'Successfully {verb}. New SHA256: {new_sha256}.{diff_str}'
