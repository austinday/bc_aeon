from functools import lru_cache

try:
    import tiktoken
except Exception:  # pragma: no cover - tiktoken is a hard dep but degrade gracefully
    tiktoken = None


@lru_cache(maxsize=1)
def _get_encoder():
    """Return a cached cl100k_base encoder.

    estimate_tokens is called dozens of times per agent iteration (every turn
    the harness re-estimates context pressure across the prompt, attempt log,
    memories, and open files). Building the encoder on every call
    (tiktoken.get_encoding) costs ~4ms each and dominated the per-iteration
    overhead; caching it makes repeated estimation ~400x faster. Returns None if
    tiktoken is unavailable so callers fall back to a char-based heuristic.
    """
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_tokens(text: str) -> int:
    """Estimate token count for ``text``.

    Uses tiktoken (cl100k_base) when available for an accurate count, and falls
    back to a ~4-chars-per-token heuristic otherwise. cl100k is not the exact
    tokenizer of the local model (Qwen3.6/Gemma-4), but it is far closer than
    len//4 and is what the context-pressure thresholds are tuned against.
    """
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    return len(text) // 4 + 1
