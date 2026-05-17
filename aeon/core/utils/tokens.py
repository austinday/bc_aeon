def estimate_tokens(text: str) -> int:
    """
    Provides a rough estimate of the number of tokens in a string.
    Used for context window management when a precise tokenizer is unavailable.
    """
    if not text:
        return 0
    # Average English word is ~4 characters. 
    # A common heuristic is 1 token approx 4 characters or 0.75 words.
    return len(text) // 4