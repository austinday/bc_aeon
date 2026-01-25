from typing import Dict, Any

def summarize_code(analyzer) -> Dict[str, Any]:
    try:
        with open(analyzer.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        line_count = content.count('\n') + 1
        return {
            "summary_type": "full_content", 
            "file_format": analyzer.file_extension.lstrip('.') or 'txt', 
            "content": content,
            "line_count": line_count
        }
    except Exception as e:
        return {"summary_type": "error", "error_message": str(e)}
