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


# --- SERVER CALIBRATION ---
# cl100k_base is not the served model's tokenizer (Qwen3.6/Gemma-4); on
# code-heavy context the counts can drift 10-30%, making the context-pressure
# thresholds mushy. Every primary-agent call already returns the server's REAL
# prompt_tokens in its usage chunk, so the LLM client feeds (text, actual)
# pairs to calibrate(): a clamped EMA of actual/raw-estimate that estimate_tokens
# then applies. Free (no extra calls), best-effort, and bounded so one bad
# sample can never skew estimates far.
_calibration = 1.0
_EMA_ALPHA = 0.2
_RATIO_MIN, _RATIO_MAX = 0.5, 3.0


def _raw_estimate(text: str) -> int:
    """Uncalibrated estimate: tiktoken cl100k when available, else ~4 chars/token."""
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    return len(text) // 4 + 1


def calibrate(text: str, actual_tokens: int) -> None:
    """Fold one observed (text, server-reported token count) pair into the
    calibration ratio. Ignores empty/absurd samples; short texts are skipped
    (their ratio is too noisy to be worth folding in)."""
    global _calibration
    if not text or not actual_tokens or actual_tokens <= 0:
        return
    raw = _raw_estimate(text)
    if raw < 500:  # too small a sample to be meaningful
        return
    ratio = actual_tokens / raw
    if not (_RATIO_MIN <= ratio <= _RATIO_MAX):
        return  # image tokens or a mismatched sample — don't poison the EMA
    _calibration = (1 - _EMA_ALPHA) * _calibration + _EMA_ALPHA * ratio


def _reset_calibration() -> None:
    """Test hook."""
    global _calibration
    _calibration = 1.0


def estimate_tokens(text: str) -> int:
    """Estimate token count for ``text``.

    Uses tiktoken (cl100k_base) when available, scaled by the live server
    calibration (see calibrate) so estimates track the actually-served model's
    tokenizer. Falls back to a ~4-chars-per-token heuristic without tiktoken.
    """
    if not text:
        return 0
    return max(1, round(_raw_estimate(text) * _calibration))
