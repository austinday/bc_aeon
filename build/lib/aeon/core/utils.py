import tiktoken
from typing import Optional

def estimate_tokens(text: str) -> int:
    """Estimate token count for text using tiktoken (cl100k_base encoding) with fallback."""
    try:
        encoder = tiktoken.get_encoding('cl100k_base')
        return len(encoder.encode(text))
    except (ImportError, Exception):
        # Fallback to approximate estimation if tiktoken unavailable
        return len(text) // 4 + 1
