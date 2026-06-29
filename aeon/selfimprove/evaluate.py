"""Isolated, scored evaluation of a candidate's code.

Capability is measured in a throwaway SANDBOX copy of the working tree (tracked
AND untracked files, so brand-new uncommitted modules are included), launched as
subprocesses with the sandbox on PYTHONPATH. Evaluation therefore never imports
or mutates the running process's source, and several candidates could be scored
in parallel. A copy-based sandbox (rather than a git worktree) is used precisely
because a candidate's new files are usually still untracked at evaluation time.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import benchmark, scorer
from .runtask import RESULT_PREFIX

# Directories never worth copying into a sandbox (heavy, regenerated, or noise).
_IGNORE = shutil.ignore_patterns(
    ".git", "aeon_output", "build", "dist", "__pycache__", ".ipynb_checkpoints",
    "*.egg-info", ".venv", "venv", ".mypy_cache", ".pytest_cache", "node_modules",
    "aeon_models", "data",
)


def _make_sandbox(root: Path):
    """Return (sandbox_dir, cleanup_fn): an isolated copy of the candidate source.

    Falls back to running in place (no isolation) only if the copy fails, so an
    evaluation never silently does nothing.
    """
    try:
        tmp = tempfile.mkdtemp(prefix="aeon_eval_")
        dest = os.path.join(tmp, "src")
        shutil.copytree(root, dest, ignore=_IGNORE, symlinks=True)
        return Path(dest), (lambda: shutil.rmtree(tmp, ignore_errors=True))
    except Exception:
        return Path(root), (lambda: None)


def _run_one(base: Path, task_id: str, timeout: int = 120) -> dict:
    env = dict(os.environ)
    # Prepend the sandbox so its aeon package shadows any pip-installed one, and
    # repoint AEON_PROJECT_ROOT at the sandbox — otherwise an inherited value pins
    # PROJECT_ROOT back to the real source and the candidate isn't truly isolated.
    env["PYTHONPATH"] = str(base) + os.pathsep + env.get("PYTHONPATH", "")
    env["AEON_PROJECT_ROOT"] = str(base)
    try:
        p = subprocess.run(
            [sys.executable, "-B", "-m", "aeon.selfimprove.runtask", task_id],
            cwd=str(base), env=env, capture_output=True, text=True, timeout=timeout,
        )
        out = p.stdout or ""
        for line in reversed(out.splitlines()):
            if line.startswith(RESULT_PREFIX):
                return json.loads(line[len(RESULT_PREFIX):])
        return {"task": task_id, "passed": False,
                "detail": f"no result line (rc={p.returncode}): {(p.stderr or out)[-200:]}",
                "metric": None}
    except subprocess.TimeoutExpired:
        return {"task": task_id, "passed": False, "detail": f"timed out after {timeout}s", "metric": None}
    except Exception as e:
        return {"task": task_id, "passed": False, "detail": f"{type(e).__name__}: {e}", "metric": None}


def evaluate(root=None, task_ids=None, timeout: int = 120) -> dict:
    """Run the deterministic benchmark against the candidate at ``root`` in an
    isolated sandbox and return a scorecard (see :func:`scorer.build_scorecard`)."""
    if root is None:
        from ..core.paths import PROJECT_ROOT
        root = PROJECT_ROOT
    root = Path(root)
    task_ids = task_ids or benchmark.deterministic_ids()
    weights = {tid: benchmark.TASKS[tid][1] for tid in task_ids if tid in benchmark.TASKS}

    base, cleanup = _make_sandbox(root)
    isolated = str(base) != str(root)
    try:
        results = [_run_one(base, tid, timeout=timeout) for tid in task_ids]
    finally:
        try:
            cleanup()
        except Exception:
            pass

    sc = scorer.build_scorecard(results, weights=weights)
    sc["isolated"] = isolated
    return sc
