import json
from typing import Dict, Any

from .text import analyze_generic_text
from ..limits import (
    ResourceLimitError,
    limit_error,
    read_bounded_bytes,
    read_text_prefix,
    regular_file_stat,
)

def get_json_schema(data: Any, depth=0) -> Any:
    if depth > 5:
        return "..."
    if isinstance(data, dict):
        return {k: get_json_schema(v, depth+1) for k, v in list(data.items())[:5]}
    elif isinstance(data, list):
        return [get_json_schema(data[0], depth+1)] if data else "list[]"
    else:
        return str(type(data).__name__)

def summarize_json(analyzer) -> Dict[str, Any]:
    try:
        current_size = regular_file_stat(analyzer.file_path).st_size
        if current_size > analyzer.MAX_JSON_PARSE_BYTES:
            sample, omitted = read_text_prefix(
                analyzer.file_path, analyzer.MAX_TEXT_PREFIX_BYTES
            )
            return {
                "summary_type": "bounded_json_preview",
                "file_format": "json",
                "content_sample": sample,
                "schema": "not parsed (input byte limit)",
                "sample_truncated": omitted,
                "description": (
                    f"JSON exceeds the {analyzer.MAX_JSON_PARSE_BYTES:,}-byte in-process "
                    "parse limit. A bounded text prefix is provided without materializing it."
                ),
            }

        raw = read_bounded_bytes(
            analyzer.file_path,
            analyzer.MAX_JSON_PARSE_BYTES,
            label="JSON file",
        )
        data = json.loads(raw.decode("utf-8"))
        if len(raw) <= analyzer.MAX_JSON_PREVIEW_SIZE:
            return {"summary_type": "full_content", "file_format": "json", "content": data}

        schema = get_json_schema(data)

        if isinstance(data, dict):
            sample = {k: data[k] for k in list(data.keys())[:2]}
        elif isinstance(data, list):
            sample = data[:2]
        else:
            sample = data

        sample_str = json.dumps(sample, indent=2)
        if len(sample_str) > 2000:
            sample_str = sample_str[:2000] + "\n... [Sample truncated]"

        return {
            "summary_type": "schema_and_sample",
            "file_format": "json",
            "schema": schema,
            "sample_data": sample_str,
        }
    except (json.JSONDecodeError, UnicodeDecodeError):
        return analyze_generic_text(analyzer)
    except ResourceLimitError as e:
        return limit_error(e)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not parse JSON: {e}"}
