#!/usr/bin/env python3
"""Run one bounded normal-Aeon qualification and emit sanitized JSON only.

The child command is the ordinary ``python -m aeon -n --start`` path.  Raw
terminal output is held only in an owner-private temporary directory, bounded,
parsed locally, and deleted.  The durable result contains no prompt, reasoning,
nonce, endpoint, process ID, ticket ID, or raw transcript.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "aeon-qwen38-normal-agent-gate-v1"
PROFILE = "aeon-qwen38-compact-178-release-gate"
CANONICAL_CWD = Path("/home/aday/NexusAgentDashboard/bc_aeon")
EXPECTED_PWD = str(CANONICAL_CWD)
FINAL_MESSAGE = f"AEON_NORMAL_AGENT_PASS: verified pwd is {EXPECTED_PWD}"
MAX_ITERATIONS = 4
DEFAULT_TIMEOUT_SECONDS = 1_800
MAX_LOG_BYTES = 2 * 1024 * 1024
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OBJECTIVE = (
    "This is a read-only normal-Aeon release qualification. Use run_command exactly "
    "once with command pwd and timeout 30. Observe its successful receipt. If and only "
    f"if it proves the exact directory {EXPECTED_PWD}, return one final response exactly: "
    f"{FINAL_MESSAGE} Do not call any other tool, mutate anything, expose private "
    "reasoning, or claim that process cleanup has already happened."
)


class NormalAeonGateError(RuntimeError):
    """The ordinary Aeon process did not produce every bounded release proof."""


def child_command() -> list[str]:
    """Return the exact ordinary Aeon CLI invocation used by this gate."""

    return [
        sys.executable,
        "-m",
        "aeon",
        "-n",
        "--start",
        _OBJECTIVE,
        "--max-iterations",
        str(MAX_ITERATIONS),
    ]


def child_environment(base: dict[str, str] | None = None, state_dir: Path | None = None) -> dict[str, str]:
    """Build an isolated environment while retaining the exact Fleet profile."""

    environment = dict(os.environ if base is None else base)
    if environment.get("AEON_FLEET_PROFILE") != PROFILE:
        raise NormalAeonGateError(
            f"AEON_FLEET_PROFILE must be set exactly to {PROFILE}"
        )
    if environment.get("AEON_SKIP_VISION_SELFTEST") == "1":
        raise NormalAeonGateError("vision self-test bypass is forbidden")
    environment.pop("AEON_SKIP_VISION_SELFTEST", None)
    environment.pop("AEON_CHAT_TRANSCRIPT_PATH", None)
    environment.pop("AEON_CHAT_WRITER_PID", None)
    environment.pop("AEON_REMOTE_INSTANCE_ID", None)
    environment["AEON_DISABLE_AUTO_TMUX"] = "1"
    if state_dir is not None:
        environment["AEON_STATE_DIR"] = str(state_dir)
    return environment


def _kill_exact_child(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def _run_child(
    command: list[str], environment: dict[str, str], *, timeout_seconds: int
) -> tuple[int, str, float]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="aeon-normal-gate-") as temporary:
        temporary_path = Path(temporary)
        os.chmod(temporary_path, 0o700)
        log_path = temporary_path / "child.log"
        with log_path.open("xb") as log:
            os.chmod(log_path, 0o600)
            process = subprocess.Popen(
                command,
                cwd=CANONICAL_CWD,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while process.poll() is None:
                if log_path.stat().st_size > MAX_LOG_BYTES:
                    _kill_exact_child(process)
                    raise NormalAeonGateError("normal Aeon output exceeded its bound")
                if time.monotonic() - started > timeout_seconds:
                    _kill_exact_child(process)
                    raise NormalAeonGateError("normal Aeon gate timed out")
                time.sleep(0.25)
            return_code = int(process.returncode)
        payload = log_path.read_bytes()
        if len(payload) > MAX_LOG_BYTES:
            raise NormalAeonGateError("normal Aeon output exceeded its bound")
        transcript = payload.decode("utf-8", errors="replace")
    return return_code, transcript, time.monotonic() - started


def validate_sanitized_transcript(return_code: int, transcript: str) -> dict[str, bool]:
    """Recompute gates from bounded terminal evidence without returning it."""

    if not isinstance(transcript, str) or len(transcript.encode()) > MAX_LOG_BYTES:
        raise NormalAeonGateError("normal Aeon transcript is malformed or oversized")
    clean = _ANSI_RE.sub("", transcript).replace("\r", "")
    forbidden = (
        "[VISION SELF-TEST] Skipped",
        "Vision is UNVERIFIED",
        "FATAL: VISION SELF-TEST FAILED",
        "COMMAND FAILED",
        "COMMAND TIMED OUT",
        "COMPLETION BLOCKED",
        "Fleet broker ticket release failed",
        "[ERROR] Fatal error",
    )
    if any(marker in clean for marker in forbidden):
        raise NormalAeonGateError("normal Aeon emitted a failure or bypass marker")
    if return_code != 0:
        raise NormalAeonGateError("normal Aeon process did not exit successfully")

    vision = len(re.findall(r"^\[VISION SELF-TEST\] PASS\b", clean, re.MULTILINE)) == 1
    actions = re.findall(r"^▶ \[[0-9]+/[0-9]+\] (.+)$", clean, re.MULTILINE)
    expected_action = "run_command(command=pwd, timeout=30)"
    pwd_action = actions == [expected_action]
    pwd_receipt = (
        clean.count(f"COMMAND SUCCESS\n\nOUTPUT:\n{EXPECTED_PWD}") == 1
        and re.search(rf"^{re.escape(EXPECTED_PWD)}$", clean, re.MULTILINE) is not None
    )
    truthful_final = len(
        re.findall(rf"^{re.escape(FINAL_MESSAGE)}$", clean, re.MULTILINE)
    ) == 1
    release = clean.count("[SESSION] Fleet broker ticket release verified.") == 1
    cleanup = clean.count("[SESSION] Cleanup complete.") == 1
    gates = {
        "process_exit_zero": True,
        "startup_vision_selftest": vision,
        "single_exact_pwd_action": pwd_action,
        "exact_pwd_receipt": pwd_receipt,
        "truthful_final": truthful_final,
        "ticket_release_verified": release,
        "session_cleanup_complete": cleanup,
    }
    if not all(gates.values()):
        failed = ",".join(name for name, passed in gates.items() if not passed)
        raise NormalAeonGateError(f"normal Aeon gate evidence is incomplete: {failed}")
    return gates


def _atomic_private_json(path: Path, value: dict[str, Any]) -> None:
    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise NormalAeonGateError("result parent is not a real directory")
    os.chmod(parent, 0o700)
    if path.exists() or path.is_symlink():
        raise NormalAeonGateError("refusing to overwrite a release-gate result")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def run_gate(output: Path, *, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    if Path.cwd().resolve() != CANONICAL_CWD:
        raise NormalAeonGateError(f"gate must run from {CANONICAL_CWD}")
    if not isinstance(timeout_seconds, int) or not 60 <= timeout_seconds <= 3_600:
        raise NormalAeonGateError("timeout must stay within 60..3600 seconds")
    with tempfile.TemporaryDirectory(prefix="aeon-normal-state-") as state:
        state_path = Path(state)
        os.chmod(state_path, 0o700)
        environment = child_environment(state_dir=state_path)
        return_code, transcript, elapsed = _run_child(
            child_command(), environment, timeout_seconds=timeout_seconds
        )
    gates = validate_sanitized_transcript(return_code, transcript)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "profile": PROFILE,
        "canonical_cwd": EXPECTED_PWD,
        "elapsed_seconds": elapsed,
        "gates": gates,
    }
    _atomic_private_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS
    )
    args = parser.parse_args(argv)
    result = run_gate(args.output, timeout_seconds=args.timeout_seconds)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
