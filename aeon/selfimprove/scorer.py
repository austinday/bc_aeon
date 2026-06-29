"""Aggregate task results into a scorecard and compare against the champion.

The ratchet that makes self-improvement *recursive* lives here: a candidate is
only worth integrating if it does not regress the baseline. ``compare`` surfaces
exactly which tasks regressed so a candidate that trades one capability for
another is caught instead of being hidden behind a flat average.
"""


def build_scorecard(results: list, weights: dict = None) -> dict:
    """``results``: list of {task, passed, detail, metric}. Returns a scorecard
    with a weighted pass-rate ``score`` in [0, 1] plus per-task detail."""
    weights = weights or {}
    total_w = 0.0
    got_w = 0.0
    tasks = []
    for r in results:
        tid = r.get("task", "?")
        w = float(weights.get(tid, 1.0))
        passed = bool(r.get("passed"))
        total_w += w
        if passed:
            got_w += w
        tasks.append({
            "task": tid, "passed": passed,
            "detail": r.get("detail", ""), "metric": r.get("metric"),
        })
    score = (got_w / total_w) if total_w else 0.0
    return {
        "score": round(score, 4),
        "passed": sum(1 for t in tasks if t["passed"]),
        "total": len(tasks),
        "tasks": tasks,
    }


def compare(candidate: dict, baseline: dict) -> dict:
    """Decide accept/reject of ``candidate`` vs ``baseline`` (either may be None).

    Accept only if the score holds or improves AND no task that passed in the
    baseline now fails (no silent capability trade-off).
    """
    cand_score = candidate.get("score", 0.0)
    if not baseline:
        return {
            "decision": "accept" if cand_score > 0 else "reject",
            "reason": f"no baseline; candidate score {cand_score:.2f}",
            "regressions": [], "delta": cand_score,
        }
    base_score = baseline.get("score", 0.0)
    base_pass = {t["task"] for t in baseline.get("tasks", []) if t.get("passed")}
    cand_pass = {t["task"] for t in candidate.get("tasks", []) if t.get("passed")}
    regressions = sorted(base_pass - cand_pass)
    delta = round(cand_score - base_score, 4)
    if regressions:
        decision, reason = "reject", f"regressed task(s): {', '.join(regressions)}"
    elif cand_score < base_score:
        decision, reason = "reject", f"score dropped {base_score:.2f} -> {cand_score:.2f}"
    elif cand_score > base_score:
        decision, reason = "accept", f"score improved {base_score:.2f} -> {cand_score:.2f}"
    else:
        decision, reason = "accept", f"score held at {cand_score:.2f}, no regressions"
    return {"decision": decision, "reason": reason, "regressions": regressions, "delta": delta}


def format_scorecard(sc: dict, cmp: dict = None) -> str:
    lines = [f"SCORE: {sc.get('score', 0):.2f}  ({sc.get('passed', 0)}/{sc.get('total', 0)} tasks passed)"]
    for t in sc.get("tasks", []):
        mark = "✓" if t.get("passed") else "✗"
        detail = (t.get("detail") or "").split("\n")[0][:100]
        lines.append(f"  {mark} {t['task']}: {detail}")
    if cmp:
        lines.append(f"DECISION: {cmp['decision'].upper()} — {cmp['reason']} (Δ {cmp.get('delta', 0):+.2f})")
    return "\n".join(lines)
