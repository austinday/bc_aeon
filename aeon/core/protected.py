"""Protected-core enforcement for self-modification.

Recursive self-modification has a specific failure mode: the cheapest way for an
agent to raise its own score is often to weaken the thing doing the scoring —
delete a hard benchmark case, loosen a success check, or neuter the rollback
machinery so a bad change can no longer be undone. Left unchecked, an optimizer
converges on gaming its own evaluator instead of getting better (Goodhart).

This module marks a SMALL set of files as protected: the self-improvement safety
machinery itself (the benchmark/scorer, the checkpoint/rollback engine, the test
gates, and this guard). The file-editing tools call ``guard()`` before writing
and refuse to touch a protected file unless a human sets
``AEON_ALLOW_PROTECTED_EDIT=1``. The agent stays free to improve everything else
— tools, prompts, skills, core loop logic — it just cannot silently dismantle
the mechanisms that measure and revert its own changes.
"""
import os
from pathlib import Path

OVERRIDE_ENV = "AEON_ALLOW_PROTECTED_EDIT"

# Paths (relative to the aeon source root) that constitute the self-modification
# "constitution". Each is machinery that grades or guards a change; letting the
# agent edit them unsupervised would defeat the whole point of measuring/reverting.
_PROTECTED_RELATIVE = (
    "aeon/core/protected.py",     # this guard itself
    "aeon/core/checkpoint.py",    # the durable rollback/checkpoint engine
    "aeon/core/bootguard.py",     # the crashed-boot auto-recovery handshake
    "aeon/tools/revert.py",       # the agent-callable revert tool
    "aeon/selfimprove",           # benchmark + scorer + evaluator + ledger (whole package)
    "aeon/smoke_test.py",         # the import gate
    "aeon/tests/test_core.py",    # the regression gate
)


def _source_root() -> Path:
    try:
        from aeon.core.paths import PROJECT_ROOT
        return Path(PROJECT_ROOT)
    except Exception:
        # paths.py is at <root>/aeon/core/paths.py -> root is two parents up.
        return Path(__file__).resolve().parents[2]


def protected_paths() -> list:
    """Absolute, resolved protected paths, including any user-supplied extras
    listed one-per-line in ``<root>/.aeon_protected`` (lets a human widen the set
    without code changes; '#'-comments and blanks ignored)."""
    root = _source_root()
    rels = list(_PROTECTED_RELATIVE)
    extra = root / ".aeon_protected"
    try:
        if extra.exists():
            for line in extra.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    rels.append(line)
    except Exception:
        pass
    out = []
    for rel in rels:
        try:
            out.append((root / rel).resolve())
        except Exception:
            continue
    return out


def is_protected(abs_path: str) -> bool:
    """True if ``abs_path`` is a protected file or lives inside a protected dir."""
    try:
        target = Path(abs_path).resolve()
    except Exception:
        return False
    for p in protected_paths():
        if target == p:
            return True
        try:
            target.relative_to(p)  # target is inside protected directory p
            return True
        except ValueError:
            continue
    return False


def override_enabled() -> bool:
    return os.environ.get(OVERRIDE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def guard(abs_path: str):
    """Return a refusal message if editing ``abs_path`` is blocked, else None."""
    if not is_protected(abs_path) or override_enabled():
        return None
    return (
        f"BLOCKED: '{abs_path}' is a PROTECTED self-improvement guardrail (benchmark/scorer, "
        f"checkpoint & rollback engine, boot-recovery handshake, or a test gate). Editing it "
        f"would let a self-modification weaken the very mechanism that measures and reverts "
        f"changes, so it is refused. If a human genuinely intends this edit, set "
        f"{OVERRIDE_ENV}=1 in the environment and retry. Otherwise improve a NON-protected "
        f"component (tools, prompts, skills, core loop logic) instead."
    )
