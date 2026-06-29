"""Durable experiment ledger for self-improvement.

Recursive improvement over many cycles only works if the agent remembers what it
already tried: which hypotheses it tested, the diffs it applied, the scores they
earned, and whether they were kept or rejected. Without this the loop re-proposes
the same dead ends forever. Records are append-only JSONL under aeon_output so
they survive restarts (and ride along with the cross-run state persistence).
"""
import json
import time
from datetime import datetime
from pathlib import Path


def _ledger_path() -> Path:
    from pathlib import Path as _P
    import os
    return _P(os.getcwd()) / "aeon_output" / "selfimprove_ledger.jsonl"


def record(entry: dict) -> dict:
    """Append one experiment record. A timestamp is added if absent. Best-effort."""
    if not isinstance(entry, dict):
        entry = {"note": str(entry)}
    entry.setdefault("at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    entry.setdefault("epoch", time.time())
    try:
        p = _ledger_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass
    return entry


def read_all() -> list:
    """All records oldest-first ([] if none)."""
    p = _ledger_path()
    out = []
    try:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return out


def last_scorecard():
    """The most recent recorded scorecard (the current champion baseline), or None."""
    for entry in reversed(read_all()):
        sc = entry.get("scorecard")
        if isinstance(sc, dict) and "score" in sc:
            return sc
    return None


def summary(limit: int = 10) -> str:
    """Human-readable tail of the ledger for injecting into the agent's context."""
    records = read_all()
    if not records:
        return "No self-improvement experiments recorded yet."
    lines = [f"Self-improvement ledger ({len(records)} experiment(s); last {min(limit, len(records))}):"]
    for r in records[-limit:]:
        sc = r.get("scorecard") or {}
        score = sc.get("score")
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
        decision = r.get("decision", "?")
        hyp = (r.get("hypothesis") or r.get("label") or "").replace("\n", " ")[:80]
        lines.append(f"- [{r.get('at', '?')}] score={score_str} decision={decision} :: {hyp}")
    return "\n".join(lines)
