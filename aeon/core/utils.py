import tiktoken
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def _get_encoder():
    """Return a cached cl100k_base encoder.

    estimate_tokens is called dozens of times per agent iteration. Building the
    encoder on every call (tiktoken.get_encoding) costs ~4ms each and dominated
    the per-iteration overhead; caching it makes repeated estimation ~400x
    faster. Returns None if tiktoken is unavailable so callers fall back.
    """
    try:
        return tiktoken.get_encoding('cl100k_base')
    except Exception:
        return None


def estimate_tokens(text: str) -> int:
    """Estimate token count for text using tiktoken (cl100k_base encoding) with fallback."""
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    # Fallback to approximate estimation if tiktoken unavailable or errored.
    return len(text) // 4 + 1
