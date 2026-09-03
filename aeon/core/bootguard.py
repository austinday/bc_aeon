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
     restores the checkpoint. The caller then re-execs from that canonical source
     tree; no package build/install hook runs in recovery.

The marker lives at a STABLE (non-PID) path so it survives the ``execv`` and a
hard crash. Cleanup and recovery remain best-effort; marker creation returns an
explicit durability verdict because process replacement must not proceed when
its rollback precondition could not be established.
"""
import json
import os
import stat
from pathlib import Path

from . import checkpoint
from .paths import PROJECT_ROOT
from .utils.io import read_bounded_fd

_MAX_BOOT_MARKER_BYTES = 16 * 1024


def _marker_path() -> Path:
    # STABLE across working directories: a crashed relaunch in workspace A must be
    # recoverable by a fresh start in workspace B (the marker names aeon_code_dir,
    # so recovery acts on the right source tree regardless of where aeon runs).
    # The old cwd-relative location silently skipped recovery unless the next
    # start happened to be in the same directory.
    aeon_home = os.environ.get("AEON_HOME") or os.path.expanduser("~/.aeon")
    return Path(aeon_home) / "boot_pending.json"


def _legacy_marker_path() -> Path:
    """Pre-2026-07 location (cwd-relative); still consumed for compatibility."""
    return Path(os.getcwd()) / "aeon_output" / ".aeon_boot_pending.json"


def mark_pending(aeon_code_dir: str, ckpt_ref: str, reason: str = "") -> bool:
    """Durably record an impending boot; report whether publication succeeded."""

    temporary: Path | None = None
    try:
        p = _marker_path()
        p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent = p.parent.lstat()
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.geteuid()
        ):
            return False
        os.chmod(p.parent, 0o700, follow_symlinks=False)
        parent = p.parent.lstat()
        if (parent.st_mode & 0o777) != 0o700:
            return False
        payload = json.dumps(
            {
                "aeon_code_dir": str(aeon_code_dir),
                "checkpoint": ckpt_ref or "",
                "reason": reason,
                "pid": os.getpid(),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        temporary = p.parent / f".{p.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    return False
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, p)
        temporary = None
        os.chmod(p, 0o600, follow_symlinks=False)
        parent_descriptor = os.open(
            p.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return True
    except Exception:
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def mark_boot_ok() -> bool:
    """Clear pending markers and report whether no marker remains."""

    cleared = True
    for p in (_marker_path(), _legacy_marker_path()):
        try:
            p.lstat()
        except FileNotFoundError:
            continue
        except Exception:
            cleared = False
            continue
        try:
            p.unlink()
        except Exception:
            cleared = False
    return cleared


def check_and_recover(print_func=print) -> dict:
    """At a FRESH startup, if a boot is still pending the previous relaunch crashed
    before going healthy. Roll its checkpoint back so the caller can re-exec from
    known-good canonical source. No-op when nothing is pending.

    Returns a small status dict for logging/telemetry.
    """
    p = None
    for candidate in (_marker_path(), _legacy_marker_path()):
        try:
            candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            continue
        p = candidate
        break
    if p is None:
        return {"recovered": False}
    descriptor = None
    try:
        descriptor = os.open(
            p,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size < 2
            or metadata.st_size > _MAX_BOOT_MARKER_BYTES
        ):
            raise ValueError("boot marker is not owner-safe")
        data = json.loads(
            read_bounded_fd(descriptor, _MAX_BOOT_MARKER_BYTES).decode("utf-8")
        )
    except Exception:
        data = {}
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(data, dict):
        data = {}
    # Consume the marker first so a failure here can't cause a recovery loop.
    try:
        p.unlink()
    except Exception:
        pass

    aeon_code_dir = data.get("aeon_code_dir") or ""
    ckpt = data.get("checkpoint") or ""
    result = {
        "recovered": True,
        "checkpoint": ckpt,
        "aeon_code_dir": "",
        "restored": False,
        "restart_required": False,
    }
    if not str(aeon_code_dir).strip():
        print_func("\033[91m[BOOTGUARD] Recovery source is missing.\033[0m")
        return result
    try:
        canonical = Path(PROJECT_ROOT).resolve(strict=True)
        requested = Path(str(aeon_code_dir)).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        print_func(f"\033[91m[BOOTGUARD] Invalid recovery source: {exc}.\033[0m")
        return result
    if requested != canonical:
        print_func(
            "\033[91m[BOOTGUARD] Refusing a recovery marker for a non-canonical "
            f"source tree: {requested}.\033[0m"
        )
        return result
    result["aeon_code_dir"] = str(canonical)
    print_func(
        f"\033[91m[BOOTGUARD] Previous restart booted broken code and never went healthy. "
        f"Rolling back to checkpoint '{ckpt or '(none)'}'.\033[0m"
    )
    if ckpt:
        r = checkpoint.restore_checkpoint(canonical, ckpt)
        result["restored"] = bool(r.get("ok"))
        if not r.get("ok"):
            print_func(f"\033[91m[BOOTGUARD] Checkpoint restore failed: {r.get('reason')}.\033[0m")
    if result["restored"]:
        result["restart_required"] = True
        print_func(
            "\033[92m[BOOTGUARD] Source restored; re-exec is required before "
            "starting a model session.\033[0m"
        )
    return result
