"""Generational boot handshake for self-modification restarts.

The hazard this closes: ``_execute_restart`` smoke-tests the new code as a
subprocess, but then ``os.execv`` relaunches through the ``--resume`` path — a
DIFFERENT code path the smoke test never exercises. If the new code boots fine in
``smoke_test.py`` yet crashes specifically on resume, the process is already
replaced and (in the old design) the backup already deleted: the agent is bricked
with broken code installed and nothing left to roll back to.

The fix is a known-good handshake, exactly like an A/B OS update:

  1. Just before ``os.execv``, write a PENDING marker naming the checkpoint that
     the new code can be rolled back to.
  2. The relaunched process, once it has booted far enough to be healthy
     (state restored, about to run), calls :func:`mark_boot_ok` to clear it.
  3. Any fresh ``aeon.main`` startup that finds a STILL-PENDING marker knows the
     previous boot died before going healthy, and :func:`check_and_recover`
     restores the checkpoint + reinstalls so the agent comes back on good code.

The marker lives at a STABLE (non-PID) path so it survives the ``execv`` and a
hard crash. All functions are best-effort and never raise into the boot path.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from . import checkpoint


def _marker_path() -> Path:
    return Path(os.getcwd()) / "aeon_output" / ".aeon_boot_pending.json"


def mark_pending(aeon_code_dir: str, ckpt_ref: str, reason: str = "") -> None:
    """Record that we are about to boot new code revertable to ``ckpt_ref``."""
    try:
        p = _marker_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "aeon_code_dir": str(aeon_code_dir),
                "checkpoint": ckpt_ref or "",
                "reason": reason,
                "pid": os.getpid(),
            }, f)
        os.replace(tmp, p)
    except Exception:
        pass


def mark_boot_ok() -> None:
    """Clear the pending marker — the relaunched process booted healthily."""
    try:
        p = _marker_path()
        if p.exists():
            p.unlink()
    except Exception:
        pass


def check_and_recover(print_func=print) -> dict:
    """At a FRESH startup, if a boot is still pending the previous relaunch crashed
    before going healthy. Roll its checkpoint back and reinstall so the agent
    returns on known-good code. No-op (and harmless) when nothing is pending.

    Returns a small status dict for logging/telemetry.
    """
    p = _marker_path()
    if not p.exists():
        return {"recovered": False}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    # Consume the marker first so a failure here can't cause a recovery loop.
    try:
        p.unlink()
    except Exception:
        pass

    aeon_code_dir = data.get("aeon_code_dir") or os.getcwd()
    ckpt = data.get("checkpoint") or ""
    print_func(
        f"\033[91m[BOOTGUARD] Previous restart booted broken code and never went healthy. "
        f"Rolling back to checkpoint '{ckpt or '(none)'}' and reinstalling.\033[0m"
    )
    result = {"recovered": True, "checkpoint": ckpt, "restored": False, "reinstalled": False}
    if ckpt:
        r = checkpoint.restore_checkpoint(aeon_code_dir, ckpt)
        result["restored"] = bool(r.get("ok"))
        if not r.get("ok"):
            print_func(f"\033[91m[BOOTGUARD] Checkpoint restore failed: {r.get('reason')}.\033[0m")
    try:
        pip = subprocess.run(
            [sys.executable, "-m", "pip", "install", ".", "--quiet"],
            cwd=aeon_code_dir, capture_output=True, text=True, timeout=300,
        )
        result["reinstalled"] = pip.returncode == 0
    except Exception as e:
        print_func(f"\033[91m[BOOTGUARD] Reinstall after rollback failed: {e}.\033[0m")
    if result["restored"]:
        print_func("\033[92m[BOOTGUARD] Recovered: agent is running the rolled-back code.\033[0m")
    return result
