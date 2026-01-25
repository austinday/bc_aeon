import json
from typing import Dict, Any

from .text import analyze_generic_text

def get_json_schema(data: Any, depth=0) -> Any:
    if depth > 5:
        return "..."
    if isinstance(data, dict):
        return {k: get_json_schema(v, depth+1) for k, v in data.items()}
    elif isinstance(data, list):
        return [get_json_schema(data[0], depth+1)] if data else "list[]"
    else:
        return str(type(data).__name__)

def summarize_json(analyzer) -> Dict[str, Any]:
    try:
        with open(analyzer.file_path, 'r', encoding='utf-8') as f:
            if analyzer.file_size <= analyzer.MAX_JSON_PREVIEW_SIZE:
                return {"summary_type": "full_content", "file_format": "json", "content": json.load(f)}
            else:
                data = json.load(f)
                return {"summary_type": "schema", "file_format": "json", "schema": get_json_schema(data)}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return analyze_generic_text(analyzer)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not parse JSON: {e}"}
