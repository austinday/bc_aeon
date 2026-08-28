import statistics
from typing import Dict, Any, List
from collections import deque

from ..limits import (
    ResourceLimitError,
    bounded_binary_readline,
    limit_error,
    regular_file_stat,
    scan_text_lines,
)

# This is a forward declaration for type hinting to avoid circular import
if False:
    from ..file_analyzer import FileAnalyzer

def filter_sample_lines(lines: List[str]) -> List[str]:
    filtered = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if stripped_line.startswith(('#', '!', '//')):
            continue
        if len(stripped_line) > 2 and all(c in '-_=#*' for c in stripped_line):
            continue
        filtered.append(stripped_line)
    return filtered

def analyze_generic_text(analyzer: 'FileAnalyzer') -> Dict[str, Any]:
    try:
        scan = scan_text_lines(
            analyzer.file_path,
            max_rows=analyzer.TEXT_ANALYSIS_SAMPLE_LINES,
            max_bytes=analyzer.MAX_TEXT_PREFIX_BYTES,
            max_line_bytes=analyzer.MAX_TEXT_LINE_BYTES,
            sample_rows=analyzer.TEXT_ANALYSIS_SAMPLE_LINES,
        )
        lines_sample = [line.strip() for line in scan.sampled_lines]

        non_empty_lines = filter_sample_lines(lines_sample)
        if not non_empty_lines:
            return summarize_unrecognized_text(analyzer)

        column_counts = [len(line.split()) for line in non_empty_lines]
        if len(column_counts) > 1 and statistics.mean(column_counts) >= analyzer.STRUCTURED_TEXT_MIN_AVG_COLUMNS:
            if statistics.stdev(column_counts) < analyzer.STRUCTURED_TEXT_COLUMN_STD_DEV_THRESHOLD:
                # Return a special dict to be handled by the main analyzer
                return {"summary_type": "_structured_text_internal", "is_likely_structured": True}

        return summarize_unrecognized_text(analyzer)
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception:
        return summarize_unrecognized_text(analyzer)

def summarize_unrecognized_text(analyzer: 'FileAnalyzer') -> Dict[str, Any]:
    should_count_lines = analyzer.file_size <= analyzer.LARGE_FILE_THRESHOLD_BYTES
    scan = scan_text_lines(
        analyzer.file_path,
        max_rows=(analyzer.MAX_TABULAR_SCAN_ROWS if should_count_lines else 10),
        max_bytes=(
            analyzer.LARGE_FILE_THRESHOLD_BYTES + 1
            if should_count_lines
            else analyzer.MAX_TEXT_PREFIX_BYTES
        ),
        max_line_bytes=analyzer.MAX_TEXT_LINE_BYTES,
        sample_rows=10,
    )
    sample_lines = [line.strip() for line in scan.sampled_lines]

    summary = {
        "summary_type": "unrecognized_text_summary",
        "file_format": analyzer.file_extension.lstrip('.') or 'txt',
        "content_sample": "\n".join(sample_lines),
        "description": "File appears to be generic text. A small sample is provided."
    }
    if should_count_lines and not scan.truncated:
        summary["line_count"] = scan.row_count
    else:
        summary["description"] += " Line count omitted for large file."
    return summary

def summarize_log_file(analyzer: 'FileAnalyzer') -> Dict[str, Any]:
    """Summarizes log files efficiently without loading entire file into memory."""
    try:
        # Define keywords that indicate a problem, case-insensitively
        problem_keywords = {'error', 'warning', 'failed', 'exception', 'fatal', 'critical', 'traceback'}
        
        head_lines = []
        tail_buffer = deque(maxlen=100)  # Keep last 100 lines in memory
        unique_error_lines = set()
        unique_error_overflow = False
        line_count = 0
        bytes_scanned = 0
        file_size = regular_file_stat(analyzer.file_path).st_size

        with open(analyzer.file_path, 'rb') as f:
            while (
                line_count < analyzer.MAX_LOG_SCAN_ROWS
                and bytes_scanned < analyzer.MAX_LOG_SCAN_BYTES
            ):
                remaining = analyzer.MAX_LOG_SCAN_BYTES - bytes_scanned
                if remaining < analyzer.MAX_TEXT_LINE_BYTES:
                    raw = f.readline(remaining + 1)
                    if len(raw) > remaining:
                        break
                else:
                    raw = bounded_binary_readline(f, analyzer.MAX_TEXT_LINE_BYTES)
                if not raw:
                    break
                if not raw.endswith((b'\n', b'\r')) and f.tell() < file_size:
                    break
                bytes_scanned += len(raw)
                line_count += 1
                line = raw.decode('utf-8', errors='replace')
                stripped = line.strip()[:analyzer.MAX_LOG_SAMPLE_LINE_CHARS]
                
                # Collect head
                if line_count <= 20:
                    head_lines.append(stripped)
                
                # Always add to tail buffer (deque auto-evicts old entries)
                tail_buffer.append(stripped)
                
                # Check for error keywords
                if any(keyword in line.lower() for keyword in problem_keywords):
                    if len(unique_error_lines) < analyzer.MAX_LOG_UNIQUE_ERRORS:
                        unique_error_lines.add(stripped)
                    elif stripped not in unique_error_lines:
                        unique_error_overflow = True
            scan_truncated = f.tell() < file_size
        error_count = (
            f">={len(unique_error_lines):,}"
            if unique_error_overflow
            else len(unique_error_lines)
        )

        return {
            "summary_type": "log_file_summary", 
            "file_format": "log",
            "line_count": line_count,
            "scan_truncated": scan_truncated,
            "bytes_scanned": bytes_scanned,
            "head_sample": "\n".join(head_lines),
            "tail_sample": "\n".join(tail_buffer),
            "error_sample": "\n".join(list(unique_error_lines)[:50]),  # Limit error samples
            "unique_error_count": error_count,
            "description": "Log file summary showing the head, tail, and a de-duplicated sample of lines containing error-related keywords."
        }
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": str(e)}
