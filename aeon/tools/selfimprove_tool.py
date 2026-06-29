"""Tools exposing the self-improvement fitness signal to the agent."""
import os

from .base import BaseTool


class RunSelfBenchmarkTool(BaseTool):
    def __init__(self, worker):
        super().__init__(
            name="run_self_benchmark",
            description=(
                "Measure the agent's CURRENT capability by running the self-improvement benchmark in "
                "an isolated sandbox copy of the source, and compare the score to the last recorded "
                "baseline (the champion). Use this to (a) establish a baseline BEFORE a self-modification, "
                "and (b) AFTER restarting, confirm the change held or improved the score and did not "
                "regress any task. A regression is the signal to revert_aeon. Results are appended to the "
                "self-improvement ledger.\n"
                "Schema:\n"
                "  record (bool, optional, default=true): Append this run to the ledger as the new baseline.\n"
                '  note (str, optional): A short hypothesis/label for this run (e.g. what change it measures).\n'
                'Example: {"tool_name": "run_self_benchmark", "parameters": {"note": "baseline before planner edit"}}'
            ),
        )
        self.worker = worker

    def execute(self, record: bool = True, note: str = "") -> str:
        try:
            from ..core.paths import PROJECT_ROOT
            from ..selfimprove import evaluate, scorer, ledger
        except Exception as e:
            return f"Error: self-improvement substrate unavailable: {e}"

        try:
            sc = evaluate.evaluate(root=str(PROJECT_ROOT))
        except Exception as e:
            return f"Error running benchmark: {type(e).__name__}: {e}"

        baseline = ledger.last_scorecard()
        cmp = scorer.compare(sc, baseline)

        if record:
            try:
                ledger.record({
                    "kind": "benchmark",
                    "hypothesis": note or "capability measurement",
                    "scorecard": sc,
                    "decision": cmp["decision"],
                    "reason": cmp["reason"],
                })
            except Exception:
                pass

        out = [scorer.format_scorecard(sc, cmp)]
        if not sc.get("isolated"):
            out.append("(NOTE: ran in place — sandbox isolation was unavailable.)")
        out.append("")
        out.append(ledger.summary(limit=5))
        return "\n".join(out)
