from typing import Dict, Any

def summarize_empty(analyzer) -> Dict[str, Any]:
    return {"summary_type": "empty_file", "file_format": analyzer.file_extension.lstrip('.'), "description": "This is an empty file."}

def summarize_opaque(analyzer) -> Dict[str, Any]:
    return {
        "summary_type": "opaque_binary", "file_format": analyzer.file_extension.lstrip('.'),
        "warning": "Content is not displayed for this file type for security or technical reasons."
    }

def is_likely_binary(analyzer) -> bool:
    if analyzer.file_size == 0:
        return False
    try:
        with open(analyzer.file_path, 'rb') as f:
            chunk = f.read(analyzer.BINARY_CHECK_BYTES)
        if b'\x00' in chunk:
            return True
        text_chars = sum(1 for byte in chunk if 31 < byte < 127 or byte in (9, 10, 13))
        return (len(chunk) - text_chars) / len(chunk) > analyzer.NON_TEXT_CHAR_THRESHOLD
    except Exception:
        return True
