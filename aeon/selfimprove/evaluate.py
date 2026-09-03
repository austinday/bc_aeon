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


class CandidateEvaluationBoundaryUnavailable(RuntimeError):
    """Modified source cannot be imported under the required OS boundary."""


def _candidate_evaluation_boundary_available() -> bool:
    """Loaded-code latch for the not-yet-proven masked-home sandbox."""

    return False


_TREE_IGNORE = shutil.ignore_patterns(
    ".git", "aeon_output", "build", "dist", "__pycache__", ".ipynb_checkpoints",
    "*.egg-info", ".venv", "venv", ".mypy_cache", ".pytest_cache", "node_modules",
    "aeon_models",
)


def _candidate_copy_ignore(root: Path):
    """Ignore root-only data without dropping nested package data directories."""

    canonical_root = root.resolve()

    def ignore(directory, names):
        ignored = set(_TREE_IGNORE(directory, names))
        if Path(directory).resolve() == canonical_root and "data" in names:
            ignored.add("data")
        return ignored

    return ignore


def _make_sandbox(root: Path):
    """Return (sandbox_dir, cleanup_fn): an isolated copy of the candidate source.

    Copy failure is fatal. Running modified code in place would turn a failed
    isolation step into full principal authority.
    """
    root = root.resolve(strict=True)
    for directory, subdirs, files in os.walk(root, followlinks=False):
        for name in [*subdirs, *files]:
            candidate = Path(directory) / name
            if not candidate.is_symlink():
                continue
            try:
                candidate.resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as exc:
                raise CandidateEvaluationBoundaryUnavailable(
                    f"candidate tree contains an escaping/unresolved symlink: {candidate}"
                ) from exc
    tmp = tempfile.mkdtemp(prefix="aeon_eval_")
    try:
        dest = os.path.join(tmp, "src")
        shutil.copytree(
            root,
            dest,
            ignore=_candidate_copy_ignore(root),
            symlinks=True,
        )
        return Path(dest), (lambda: shutil.rmtree(tmp, ignore_errors=True))
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _run_one(base: Path, task_id: str, timeout: int = 120) -> dict:
    if not _candidate_evaluation_boundary_available():
        return {
            "task": task_id,
            "passed": False,
            "detail": (
                "candidate evaluation blocked: an actively-probed masked-home "
                "dependency sandbox is not installed"
            ),
            "metric": None,
        }
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
    if not _candidate_evaluation_boundary_available():
        raise CandidateEvaluationBoundaryUnavailable(
            "candidate benchmark is blocked until an actively-probed masked-home "
            "dependency sandbox is installed; no candidate code was executed"
        )
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
