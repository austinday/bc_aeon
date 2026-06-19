"""
Cross-process liveness + atomic file IO primitives for sub-agent supervision.

- touch()/activity_age()/reset(): an in-process heartbeat. Any ongoing work
  (an LLM stream chunk, a command output line, a planning iteration) calls
  touch(); the sub-agent watchdog reads activity_age() to distinguish a
  slow-but-alive agent from a genuinely hung one.
- atomic_write_text/json: write-to-temp + os.replace so a reader in ANOTHER
  process always sees either the complete old file or the complete new file,
  never a torn read. This eliminates the partial-read races that plagued the
  old non-atomic status/output/telemetry writes.

No fsync: we need atomicity (no torn reads by a live peer), not crash
durability, so os.replace alone is correct and cheap enough for a 5s cadence.
This module imports nothing from aeon, so it is safe to import anywhere
(no cycles), and touch() is a harmless no-op in the primary agent.
"""

import os
import time
import json
import tempfile
import threading

_lock = threading.Lock()
_last_activity = time.time()


def touch():
    """Signal that work is ongoing (resets the staleness clock)."""
    global _last_activity
    with _lock:
        _last_activity = time.time()


def activity_age() -> float:
    """Seconds since the last touch()."""
    with _lock:
        return time.time() - _last_activity


def reset():
    """Reset the heartbeat to 'now' (call at the start of a run)."""
    touch()


def atomic_write_text(path, text: str):
    path = str(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_aeon_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))
