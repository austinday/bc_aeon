import statistics
from typing import Dict, Any, List
from collections import deque

# This is a forward declaration for type hinting to avoid circular import
if False:
    from ..file_analyzer import FileAnalyzer

def filter_sample_lines(lines: List[str]) -> List[str]:
    filtered = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line: continue
        if stripped_line.startswith(('#', '!', '//')): continue
        if len(stripped_line) > 2 and all(c in '-_=#*' for c in stripped_line): continue
        filtered.append(stripped_line)
    return filtered

def analyze_generic_text(analyzer: 'FileAnalyzer') -> Dict[str, Any]:
    try:
        with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines_sample = [next(f, '').strip() for _ in range(analyzer.TEXT_ANALYSIS_SAMPLE_LINES)]
        
        non_empty_lines = filter_sample_lines(lines_sample)
        if not non_empty_lines:
            return summarize_unrecognized_text(analyzer)

        column_counts = [len(line.split()) for line in non_empty_lines]
        if len(column_counts) > 1 and statistics.mean(column_counts) >= analyzer.STRUCTURED_TEXT_MIN_AVG_COLUMNS:
            if statistics.stdev(column_counts) < analyzer.STRUCTURED_TEXT_COLUMN_STD_DEV_THRESHOLD:
                # Return a special dict to be handled by the main analyzer
                return {"summary_type": "_structured_text_internal", "is_likely_structured": True}

        return summarize_unrecognized_text(analyzer)
    except Exception:
        return summarize_unrecognized_text(analyzer)

def summarize_unrecognized_text(analyzer: 'FileAnalyzer') -> Dict[str, Any]:
    should_count_lines = analyzer.file_size <= analyzer.LARGE_FILE_THRESHOLD_BYTES
    with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
        num_lines = -1
        if should_count_lines:
            num_lines = sum(1 for _ in f)
            f.seek(0)
        sample_lines = [next(f, '').strip() for _ in range(10)]
    
    summary = {
        "summary_type": "unrecognized_text_summary",
        "file_format": analyzer.file_extension.lstrip('.') or 'txt',
        "content_sample": "\n".join(sample_lines),
        "description": "File appears to be generic text. A small sample is provided."
    }
    if num_lines != -1:
        summary["line_count"] = num_lines
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
        line_count = 0
        
        with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_count += 1
                stripped = line.strip()
                
                # Collect head
                if line_count <= 20:
                    head_lines.append(stripped)
                
                # Always add to tail buffer (deque auto-evicts old entries)
                tail_buffer.append(stripped)
                
                # Check for error keywords
                if any(keyword in line.lower() for keyword in problem_keywords):
                    unique_error_lines.add(stripped)

        return {
            "summary_type": "log_file_summary", 
            "file_format": "log",
            "line_count": line_count,
            "head_sample": "\n".join(head_lines),
            "tail_sample": "\n".join(tail_buffer),
            "error_sample": "\n".join(list(unique_error_lines)[:50]),  # Limit error samples
            "unique_error_count": len(unique_error_lines),
            "description": "Log file summary showing the head, tail, and a de-duplicated sample of lines containing error-related keywords."
        }
    except Exception as e:
        return {"summary_type": "error", "error_message": str(e)}
