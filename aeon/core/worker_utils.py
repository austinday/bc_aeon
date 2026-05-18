import json
from typing import Any

# Colors for terminal output
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[95m'
C_RESET = '\033[0m'
C_BLUE = '\033[96m'

def truncate_output(text: str, max_chars: int = 50000) -> str:
    """Deterministic head+tail truncation. Prioritizes tail (where errors appear)."""
    if len(text) <= max_chars:
        return text
    
    omitted = len(text) - max_chars
    msg = f"\n\n... [{omitted:,} CHARS TRUNCATED] ...\n\n"
    msg_len = len(msg)
    
    # Adjust budget to fit the message within max_chars
    available_budget = max_chars - msg_len
    if available_budget < 0:
        return text[:max_chars] # Fallback for extremely small max_chars
        
    head_budget = available_budget // 4
    tail_budget = available_budget - head_budget
    
    return text[:head_budget] + msg + text[len(text)-tail_budget:]