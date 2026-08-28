from typing import Dict, Any

from ..limits import ResourceLimitError, limit_error, read_bounded_bytes, regular_file_stat


def summarize_code(analyzer) -> Dict[str, Any]:
    try:
        current_size = regular_file_stat(analyzer.file_path).st_size
        if current_size > analyzer.MAX_FULL_CONTENT_BYTES:
            preview_bytes = analyzer.MAX_TEXT_PREFIX_BYTES
            with open(analyzer.file_path, "rb") as handle:
                head = handle.read(preview_bytes)
                handle.seek(-min(preview_bytes, current_size), 2)
                tail = handle.read(preview_bytes)
            return {
                "summary_type": "bounded_text_preview",
                "file_format": analyzer.file_extension.lstrip('.') or 'txt',
                "head_sample": head.decode("utf-8", errors="replace"),
                "tail_sample": tail.decode("utf-8", errors="replace"),
                "description": (
                    f"File exceeds the {analyzer.MAX_FULL_CONTENT_BYTES:,}-byte full-content "
                    "limit. Bounded byte head and tail samples are provided; line count omitted."
                ),
            }
        raw = read_bounded_bytes(
            analyzer.file_path,
            analyzer.MAX_FULL_CONTENT_BYTES,
            label="text/code file",
        )
        content = raw.decode("utf-8", errors="replace")
        line_count = content.count('\n') + 1
        return {
            "summary_type": "full_content", 
            "file_format": analyzer.file_extension.lstrip('.') or 'txt', 
            "content": content,
            "line_count": line_count
        }
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": str(e)}
