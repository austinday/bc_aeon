"""Single-task entrypoint, run as a subprocess against a candidate's code.

The evaluator launches ``python -m aeon.selfimprove.runtask <task_id>`` with the
candidate worktree on PYTHONPATH, so the task executes the CANDIDATE'S aeon code,
not the running process's. The result is emitted as a single JSON line on stdout
(prefixed) so the parent can parse it regardless of any other task chatter.
"""
import json
import sys

RESULT_PREFIX = "@@RUNTASK_RESULT@@ "


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(RESULT_PREFIX + json.dumps({"passed": False, "detail": "no task id", "metric": None}))
        return 2
    task_id = argv[0]
    try:
        from aeon.selfimprove import benchmark
        passed, detail, metric = benchmark.run_task(task_id)
    except Exception as e:
        passed, detail, metric = False, f"runtask crashed: {type(e).__name__}: {e}", None
    print(RESULT_PREFIX + json.dumps({
        "task": task_id, "passed": bool(passed), "detail": str(detail), "metric": metric,
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
