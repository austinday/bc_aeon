import json
from typing import Dict, Any

from .text import analyze_generic_text

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
        with open(analyzer.file_path, 'r', encoding='utf-8') as f:
            if analyzer.file_size <= analyzer.MAX_JSON_PREVIEW_SIZE:
                return {"summary_type": "full_content", "file_format": "json", "content": json.load(f)}
            else:
                data = json.load(f)
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
                    "sample_data": sample_str
                }
    except (json.JSONDecodeError, UnicodeDecodeError):
        return analyze_generic_text(analyzer)
    except Exception as e:
        return {"summary_type": "error", "error_message": f"Could not parse JSON: {e}"}
