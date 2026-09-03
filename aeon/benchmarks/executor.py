"""Reviewed, bounded bridges from benchmark scenarios to real Aeon harnesses.

The benchmark worker itself is launched by :mod:`aeon.benchmarks.service`
through the fixed ``fleet-low-priority`` wrapper.  This module then starts only
one of the two reviewed harness modules.  Both harnesses acquire model service
through Fleet Compute; this layer never selects a host, GPU, claim, or endpoint.
"""

from __future__ import annotations

import importlib.util
import os
import re
import secrets
import selectors
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence, TYPE_CHECKING

import requests

from aeon.core.fleet_backend import BENCHMARK_COMPUTE_STATUS_FD_ENV
from aeon.core.model_identity import AEON_DEFAULT_MODEL_NAME
from aeon.harnesses.launch import build_harness_argv
from aeon.harnesses.opencode_install import opencode_status

from aeon.core.benchmark_receipt import (
    CAPABILITY_RECEIPT_KEY_ENV,
    CAPABILITY_RECEIPT_PATH_ENV,
    MAX_CAPABILITY_RECEIPT_BYTES,
    FleetWaitCapabilityReceipt,
    decode_capability_receipts,
)
from .runner import (
    ExecutionCancelled,
    ExecutionRequest,
    ExecutorUnavailable,
    ExecutorUnresolved,
)
from .catalog import DEFAULT_MODEL_ID
from .service import FLEET_LOW_PRIORITY, RUN_ID_RE

if TYPE_CHECKING:
    from .service import BenchmarkService


MAX_CAPTURE_BYTES = 4 * 1024 * 1024
TERMINATION_GRACE_SECONDS = 5.0
EXECUTOR_CLOSE_GRACE_SECONDS = 12.0
BROWSER_FIXTURE_TIMEOUT_SECONDS = 8.0
BENCHMARK_BROWSER_PROFILE = "benchmark-000000000000"
_HARNESS_IDS = frozenset({"opencode", "legacy-aeon"})
_MODEL_LOGICAL_NAMES = {DEFAULT_MODEL_ID: AEON_DEFAULT_MODEL_NAME}
_SAFE_HARNESS_ENVIRONMENT = frozenset(
    {
        "AEON_COMPUTE_BACKEND",
        "AEON_OPENCODE_HOME",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "PATH",
        "PYTHONPATH",
        "TMPDIR",
        "TZ",
        "USER",
        "VIRTUAL_ENV",
    }
)


@dataclass(frozen=True)
class ProcessResult:
    state: str
    returncode: int | None
    output: bytes
    # ``wall_ms`` remains the total end-to-end duration for API compatibility.
    wall_ms: float
    compute_wait_ms: float = 0.0
    capability_receipts: tuple[FleetWaitCapabilityReceipt, ...] = ()

    @property
    def active_wall_ms(self) -> float:
        return max(0.0, float(self.wall_ms) - float(self.compute_wait_ms))


ProcessRunner = Callable[
    [
        Sequence[str],
        Path,
        Mapping[str, str],
        float,
        Callable[[], bool],
        Callable[[str], None],
    ],
    ProcessResult,
]
ReadinessChecker = Callable[[str | None], Mapping[str, object]]
BrowserFixtureClient = Callable[[str, str], bool]


def _safe_wrapper() -> bool:
    try:
        metadata = Path(FLEET_LOW_PRIORITY).lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and metadata.st_nlink == 1
        and metadata.st_mode & stat.S_IXUSR
        and not stat.S_IMODE(metadata.st_mode) & 0o022
    )


def runtime_execution_status(harness_id: str | None = None) -> dict[str, object]:
    """Return a sanitized, read-only readiness decision for Nexus."""

    if harness_id is not None and harness_id not in _HARNESS_IDS:
        return {
            "supported": False,
            "reason": "The selected benchmark harness is not reviewed.",
        }
    if not _safe_wrapper():
        return {
            "supported": False,
            "reason": "The owner batch wrapper is unavailable or unsafe.",
        }
    if importlib.util.find_spec("aeon.main") is None:
        return {
            "supported": False,
            "reason": "The Aeon runtime package is unavailable.",
        }
    if harness_id in {None, "opencode"}:
        try:
            status = opencode_status()
        except Exception:
            status = {}
        if status.get("ready") is not True:
            return {
                "supported": False,
                "reason": "The pinned OpenCode runtime is not ready.",
            }
    return {"supported": True, "reason": ""}


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    starttime: int


@dataclass(frozen=True)
class _CapabilityReceiptTarget:
    path: Path
    key: str
    device: int
    inode: int


def _process_identity(pid: int) -> _ProcessIdentity | None:
    """Bind a task-owned PID to its kernel start time to prevent PID reuse."""

    try:
        proc = Path("/proc") / str(pid)
        if proc.stat().st_uid != os.geteuid():
            return None
        raw = (proc / "stat").read_text(encoding="ascii")
        fields = raw[raw.rfind(")") + 2 :].split()
        starttime = int(fields[19])
    except (OSError, UnicodeError, ValueError, IndexError):
        return None
    return _ProcessIdentity(pid=pid, starttime=starttime)


def _identity_alive(identity: _ProcessIdentity) -> bool:
    if _process_identity(identity.pid) != identity:
        return False
    try:
        raw = (Path("/proc") / str(identity.pid) / "stat").read_text(encoding="ascii")
        state = raw[raw.rfind(")") + 2 :].split()[0]
    except (OSError, UnicodeError, IndexError):
        return False
    return state != "Z"


def _direct_children(identity: _ProcessIdentity) -> list[_ProcessIdentity]:
    """Read only Linux's child lists for a proven task-owned process."""

    if not _identity_alive(identity):
        return []
    result: dict[int, _ProcessIdentity] = {}
    task_root = Path("/proc") / str(identity.pid) / "task"
    try:
        threads = list(task_root.iterdir())
    except OSError:
        return []
    for thread in threads:
        try:
            values = (thread / "children").read_text(encoding="ascii").split()
        except (OSError, UnicodeError):
            continue
        for value in values:
            try:
                pid = int(value)
            except ValueError:
                continue
            child = _process_identity(pid)
            if child is not None:
                result[pid] = child
    return list(result.values())


def _descendants(root: _ProcessIdentity) -> list[_ProcessIdentity]:
    seen = {root.pid}
    pending = [root]
    result: list[_ProcessIdentity] = []
    while pending:
        parent = pending.pop()
        for child in _direct_children(parent):
            if child.pid in seen:
                continue
            seen.add(child.pid)
            result.append(child)
            pending.append(child)
    return result


def _signal_identity(identity: _ProcessIdentity, signum: int) -> None:
    if not _identity_alive(identity):
        return
    try:
        os.kill(identity.pid, signum)
    except (ProcessLookupError, PermissionError):
        return


def _terminate_process_tree(child: subprocess.Popen[bytes]) -> bool:
    """Stop and prove exit of the exact harness tree, including new sessions.

    OpenCode intentionally gives its own CLI/MCP subtree another process group.
    A group-only kill therefore is insufficient.  We snapshot only descendants
    exposed by the task-owned root's ``/proc/.../children`` links, bind every PID
    to its start time, and revalidate before signaling.
    """

    root = _process_identity(child.pid)
    if root is None:
        return child.poll() is not None
    known: dict[int, _ProcessIdentity] = {
        item.pid: item for item in _descendants(root)
    }
    # The root's group catches ordinary children; individually signal captured
    # descendants as well so OpenCode's separate session cannot escape.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        _signal_identity(root, signal.SIGTERM)
    for identity in reversed(tuple(known.values())):
        _signal_identity(identity, signal.SIGTERM)

    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        if _identity_alive(root):
            for identity in _descendants(root):
                if identity.pid not in known:
                    known[identity.pid] = identity
                    _signal_identity(identity, signal.SIGTERM)
        if child.poll() is not None and not any(
            _identity_alive(identity) for identity in known.values()
        ):
            return True
        time.sleep(0.05)

    for identity in reversed(tuple(known.values())):
        _signal_identity(identity, signal.SIGKILL)
    _signal_identity(root, signal.SIGKILL)
    try:
        child.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        return False
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        if not any(_identity_alive(identity) for identity in known.values()):
            return True
        time.sleep(0.05)
    return False


def run_bounded_process(
    argv: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    cancel_requested: Callable[[], bool],
    compute_state_changed: Callable[[str], None] | None = None,
) -> ProcessResult:
    """Run one harness with a deadline that excludes proven Fleet waiting.

    The child owns one normal :class:`BrokerServiceSession`. Its sanitized
    compute transitions arrive over an inherited anonymous pipe. Only an exact
    ``waiting_for_compute`` transition pauses the case budget; cancellation and
    process-tree supervision remain active throughout the wait.
    """

    started = time.monotonic()
    status_read, status_write = os.pipe()
    os.set_blocking(status_write, False)
    child_environment = {
        str(key): str(value) for key, value in environment.items()
    }
    child_environment[BENCHMARK_COMPUTE_STATUS_FD_ENV] = str(status_write)
    try:
        child = subprocess.Popen(
            [str(item) for item in argv],
            cwd=cwd,
            env=child_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            close_fds=True,
            pass_fds=(status_write,),
            start_new_session=True,
            shell=False,
        )
    except BaseException:
        os.close(status_read)
        os.close(status_write)
        raise
    os.close(status_write)
    selector: selectors.BaseSelector | None = None
    output = bytearray()
    status_buffer = bytearray()
    state = "stuck"
    compute_wait_seconds = 0.0
    waiting_for_compute = False
    last_tick = started

    def account_until(now: float) -> None:
        nonlocal compute_wait_seconds, last_tick, remaining_budget
        elapsed = max(0.0, now - last_tick)
        if waiting_for_compute:
            compute_wait_seconds += elapsed
        else:
            remaining_budget -= elapsed
        last_tick = now

    remaining_budget = max(1.0, float(timeout_seconds))
    try:
        if child.stdout is None:
            raise RuntimeError("benchmark harness output pipe is unavailable")
        selector = selectors.DefaultSelector()
        descriptor = child.stdout.fileno()
        os.set_blocking(descriptor, False)
        os.set_blocking(status_read, False)
        selector.register(descriptor, selectors.EVENT_READ, "output")
        selector.register(status_read, selectors.EVENT_READ, "compute")
        state = "exited"
        eof = False
        while not eof or child.poll() is None:
            account_until(time.monotonic())
            if cancel_requested():
                state = "cancelled" if _terminate_process_tree(child) else "stuck"
                break
            if remaining_budget <= 0:
                state = "timeout" if _terminate_process_tree(child) else "stuck"
                break
            for key, _mask in selector.select(
                timeout=min(0.1, remaining_budget)
                if not waiting_for_compute
                else 0.1
            ):
                try:
                    chunk = os.read(
                        key.fd, 4096 if key.data == "compute" else 64 * 1024
                    )
                except BlockingIOError:
                    continue
                if not chunk:
                    if key.data == "output":
                        eof = True
                    else:
                        try:
                            selector.unregister(status_read)
                        except (KeyError, ValueError):
                            pass
                    continue
                if key.data == "compute":
                    status_buffer.extend(chunk)
                    if len(status_buffer) > 4096:
                        status_buffer.clear()
                        try:
                            selector.unregister(status_read)
                        except (KeyError, ValueError):
                            pass
                        continue
                    while b"\n" in status_buffer:
                        raw_state, _, tail = status_buffer.partition(b"\n")
                        status_buffer = bytearray(tail)
                        try:
                            compute_state = raw_state.decode("ascii")
                        except UnicodeDecodeError:
                            continue
                        # Attribute all elapsed time before the transition to
                        # the prior state.  This preserves both total and
                        # broker-proven wait duration instead of merely
                        # dropping wait intervals from the deadline.
                        account_until(time.monotonic())
                        if compute_state == "waiting_for_compute":
                            waiting_for_compute = True
                        elif compute_state in {"allocated", "idle", "unavailable"}:
                            waiting_for_compute = False
                        else:
                            continue
                        if compute_state_changed is not None:
                            compute_state_changed(compute_state)
                    continue
                if len(output) + len(chunk) > MAX_CAPTURE_BYTES:
                    remaining = max(0, MAX_CAPTURE_BYTES - len(output))
                    output.extend(chunk[:remaining])
                    state = "stuck"
                    _terminate_process_tree(child)
                    eof = True
                    break
                output.extend(chunk)
        if state == "exited" and child.poll() is None:
            state = "stuck"
            _terminate_process_tree(child)
    finally:
        if selector is not None:
            selector.close()
        os.close(status_read)
        if child.stdout is not None:
            child.stdout.close()
        if child.poll() is None:
            _terminate_process_tree(child)
    finished = time.monotonic()
    account_until(finished)
    total_wall_ms = max(0.0, (finished - started) * 1000.0)
    compute_wait_ms = min(total_wall_ms, max(0.0, compute_wait_seconds * 1000.0))
    return ProcessResult(
        state=state,
        returncode=child.poll(),
        output=bytes(output),
        wall_ms=total_wall_ms,
        compute_wait_ms=compute_wait_ms,
    )


def _png_fixture() -> bytes:
    """Create a tiny deterministic RGB PNG using only the standard library."""

    width, height = 160, 80
    rows = []
    for _y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((220, 35, 35) if x < width // 2 else (30, 75, 220))
        rows.append(bytes(row))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows), level=9))
        + chunk(b"IEND", b"")
    )


class FleetHarnessExecutor:
    """Execute semantic cases through a real Fleet-backed harness process."""

    def __init__(
        self,
        service: "BenchmarkService",
        run_id: str,
        *,
        process_runner: ProcessRunner = run_bounded_process,
        readiness_checker: ReadinessChecker = runtime_execution_status,
        browser_fixture_client: BrowserFixtureClient | None = None,
    ) -> None:
        if RUN_ID_RE.fullmatch(run_id) is None:
            raise ValueError("invalid benchmark run ID")
        self.service = service
        self.run_id = run_id
        self._process_runner = process_runner
        self._readiness_checker = readiness_checker
        self._fixture_client = browser_fixture_client
        # Canonical-host storage policy forbids automatic recursive deletion.
        # Retain the exact owner-private run workspace for manual, audited
        # cleanup; public APIs and evidence never expose its prompts or paths.
        work_root = service.root / "work"
        work_root.mkdir(mode=0o700, exist_ok=True)
        work_metadata = work_root.lstat()
        if (
            not stat.S_ISDIR(work_metadata.st_mode)
            or stat.S_ISLNK(work_metadata.st_mode)
            or work_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(work_metadata.st_mode) != 0o700
        ):
            raise ExecutorUnavailable("benchmark work root is unsafe")
        self.workspace = work_root / run_id
        self.workspace.mkdir(mode=0o700)
        state_root = service.root / "harness-state"
        state_root.mkdir(mode=0o700, exist_ok=True)
        state_metadata = state_root.lstat()
        if (
            not stat.S_ISDIR(state_metadata.st_mode)
            or stat.S_ISLNK(state_metadata.st_mode)
            or state_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(state_metadata.st_mode) != 0o700
        ):
            raise ExecutorUnavailable("benchmark harness state root is unsafe")
        self.harness_state = state_root / run_id
        self.harness_state.mkdir(mode=0o700)
        harness_state_metadata = self.harness_state.lstat()
        if (
            not stat.S_ISDIR(harness_state_metadata.st_mode)
            or stat.S_ISLNK(harness_state_metadata.st_mode)
            or harness_state_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(harness_state_metadata.st_mode) != 0o700
            or self.harness_state.is_relative_to(self.workspace)
            or self.workspace.is_relative_to(self.harness_state)
        ):
            raise ExecutorUnavailable("benchmark harness state is unsafe")
        self._lifecycle_lock = threading.RLock()
        self._child_cancel = threading.Event()
        self._active_done = threading.Event()
        self._active_done.set()
        self._prepared = False
        self._closing = False
        self._closed = False
        self._compute_waiting = threading.Event()
        self._session_fixture_seeded = False
        self.browser_session_id = f"oc-{run_id[4:]}"
        self.browser_profile = BENCHMARK_BROWSER_PROFILE
        self._prepare_fixtures()

    def __enter__(self) -> "FleetHarnessExecutor":
        return self

    def __exit__(self, _kind, _value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closing = True
            self._child_cancel.set()
        if not self._active_done.wait(timeout=EXECUTOR_CLOSE_GRACE_SECONDS):
            raise ExecutorUnresolved("benchmark harness termination was not proved")
        with self._lifecycle_lock:
            if self._session_fixture_seeded:
                self._fixture_operation("session-v1", "cleanup")
                self._session_fixture_seeded = False
            if self._fixture_client is None:
                self._close_browser_fixture()
            self._closed = True

    def cancel(self, _request: ExecutionRequest | None = None) -> None:
        with self._lifecycle_lock:
            self._child_cancel.set()

    def prepare(self, _request: ExecutionRequest) -> None:
        """Create the cancellation token before the runner starts its thread."""

        with self._lifecycle_lock:
            if self._closed or self._closing:
                raise ExecutorUnavailable("benchmark executor is closed")
            if not self._active_done.is_set():
                raise ExecutorUnavailable("a prior benchmark harness is unresolved")
            self._child_cancel = threading.Event()
            self._compute_waiting.clear()
            self._prepared = True

    def _prepare_fixtures(self) -> None:
        fixture_root = self.workspace / "fixtures"
        result_root = self.workspace / "results"
        state_root = self.workspace / "state"
        for directory in (fixture_root, result_root, state_root):
            directory.mkdir(mode=0o700)
        (fixture_root / "read-token.txt").write_text(
            "BENCH_READ_LARK_7319\n", encoding="utf-8"
        )
        (fixture_root / "vision.png").write_bytes(_png_fixture())
        for path in fixture_root.iterdir():
            path.chmod(0o600)

    def _environment(
        self,
        request: ExecutionRequest,
        *,
        receipt_target: _CapabilityReceiptTarget | None = None,
    ) -> dict[str, str]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in _SAFE_HARNESS_ENVIRONMENT
            if isinstance(key, str)
            and isinstance(value, str)
            and "\x00" not in key
            and "\x00" not in value
        }
        environment.update(
            {
                "AEON_COMPUTE_BACKEND": "broker",
                # OpenCode requires its supervisor/config/session state to be
                # disjoint from the model-facing workspace. Legacy Aeon accepts
                # the same per-run private sibling root.
                "AEON_STATE_DIR": str(self.harness_state),
                "AEON_INSTANCE_SKILLS_DIR": str(self.workspace / "skills"),
                "AEON_BROWSER_PROFILE": self.browser_profile,
                "AEON_BROWSER_SESSION_ID": self.browser_session_id,
                "AEON_OPENCODE_TURN_TIMEOUT_SECONDS": str(request.timeout_seconds),
                "CUDA_VISIBLE_DEVICES": "void",
                "GPU_DEVICE_ORDINAL": "-1",
                "HIP_VISIBLE_DEVICES": "-1",
                "NVIDIA_VISIBLE_DEVICES": "void",
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                "PYTHONUNBUFFERED": "1",
                "ROCR_VISIBLE_DEVICES": "-1",
            }
        )
        if request.scenario.category != "vision":
            environment["AEON_SKIP_VISION_SELFTEST"] = "1"
        else:
            environment.pop("AEON_SKIP_VISION_SELFTEST", None)
        if receipt_target is not None:
            environment[CAPABILITY_RECEIPT_PATH_ENV] = str(receipt_target.path)
            environment[CAPABILITY_RECEIPT_KEY_ENV] = receipt_target.key
        return environment

    def _new_capability_receipt_target(
        self, request: ExecutionRequest
    ) -> _CapabilityReceiptTarget:
        root = self.workspace / "results"
        label = re.sub(r"[^a-z0-9]+", "-", request.scenario.case_id.casefold()).strip("-")
        path = root / (
            f".{label}-r{request.repetition}-{secrets.token_hex(12)}.receipt"
        )
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        return _CapabilityReceiptTarget(
            path=path,
            key=secrets.token_hex(32),
            device=metadata.st_dev,
            inode=metadata.st_ino,
        )

    @staticmethod
    def _read_capability_receipts(
        target: _CapabilityReceiptTarget,
    ) -> tuple[FleetWaitCapabilityReceipt, ...]:
        descriptor: int | None = None
        try:
            descriptor = os.open(
                target.path,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_dev != target.device
                or metadata.st_ino != target.inode
                or metadata.st_size > MAX_CAPABILITY_RECEIPT_BYTES
            ):
                return ()
            payload = os.read(descriptor, MAX_CAPABILITY_RECEIPT_BYTES + 1)
        except OSError:
            return ()
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if len(payload) != metadata.st_size:
            return ()
        return decode_capability_receipts(payload, key=target.key)

    def _command(
        self,
        request: ExecutionRequest,
        prompt: str,
        *,
        resume: bool = False,
    ) -> list[str]:
        logical_model = _MODEL_LOGICAL_NAMES.get(request.model_id)
        if logical_model is None:
            # A future catalog addition must provide an explicit Fleet-backed
            # launch mapping. Never benchmark a different model while recording
            # the caller-selected identity.
            raise ExecutorUnavailable("selected benchmark model has no reviewed runtime mapping")
        command = build_harness_argv(
            sys.executable,
            request.harness_id,
            logical_model,
            resume_unfinished=resume,
            start_objective=prompt,
        )
        command.extend(
            [
                "--max-iterations",
                "12",
                "--non-interactive",
                "--browser-profile",
                self.browser_profile,
            ]
        )
        return command

    def _cancel_requested(self) -> bool:
        return self._child_cancel.is_set() or self.service._cancel_requested(self.run_id)

    def deadline_paused(self, _request: ExecutionRequest | None = None) -> bool:
        """Return whether the one active harness proved it is waiting in Fleet."""

        return self._compute_waiting.is_set()

    def _compute_state_changed(self, state: str) -> None:
        if state == "waiting_for_compute":
            self._compute_waiting.set()
            self.service._set_active_state(self.run_id, "waiting_for_compute")
            return
        self._compute_waiting.clear()
        self.service._set_active_state(self.run_id, "running")

    def _run(
        self,
        request: ExecutionRequest,
        prompt: str,
        *,
        resume: bool = False,
    ) -> ProcessResult:
        if self._closed:
            raise ExecutorUnavailable("benchmark executor is closed")
        status = self._readiness_checker(request.harness_id)
        if status.get("supported") is not True:
            raise ExecutorUnavailable("selected harness runtime is unavailable")
        receipt_target = self._new_capability_receipt_target(request)
        result = self._process_runner(
            self._command(request, prompt, resume=resume),
            self.workspace,
            self._environment(request, receipt_target=receipt_target),
            float(request.timeout_seconds),
            self._cancel_requested,
            self._compute_state_changed,
        )
        observed_receipts = self._read_capability_receipts(receipt_target)
        if observed_receipts:
            result = replace(
                result,
                capability_receipts=(
                    *result.capability_receipts,
                    *observed_receipts,
                ),
            )
        if result.state == "cancelled":
            raise ExecutionCancelled()
        return result

    def _browser_fixture(self, fixture_id: str, operation: str) -> bool:
        """Seed/verify only the authenticated service's closed fixture catalog."""

        if fixture_id not in {"observe-v1", "form-v1", "session-v1", "vision-v1"}:
            return False
        if operation not in {"seed", "reopen", "verify", "cleanup"}:
            return False
        if operation in {"reopen", "cleanup"} and fixture_id != "session-v1":
            return False
        try:
            from aeon.tools.browser import (
                BROWSER_API_URL,
                _browser_service_identity,
                browser_auth_headers,
                ensure_browser_running,
            )

            ensure_browser_running()
            _browser_service_identity()
            response = requests.post(
                f"{BROWSER_API_URL}/benchmark_fixture",
                json={
                    "session_id": self.browser_session_id,
                    "tab_id": "benchmark",
                    "profile": self.browser_profile,
                    "fixture_id": fixture_id,
                    "operation": operation,
                },
                headers=browser_auth_headers(),
                # Stay strictly inside the runner's cooperative-cancel grace.
                # The preceding health request is separately bounded to two
                # seconds by the reviewed browser client.
                timeout=BROWSER_FIXTURE_TIMEOUT_SECONDS,
                allow_redirects=False,
                proxies={"http": "", "https": ""},
            )
            if response.status_code != 200:
                response.close()
                return False
            document = response.json()
            response.close()
        except Exception:
            return False
        if not isinstance(document, Mapping) or document.get("fixture_id") != fixture_id:
            return False
        if operation == "seed":
            return document.get("status") == "seeded"
        if operation == "reopen":
            return (
                document.get("status") == "reopened"
                and document.get("passed") is True
            )
        if operation == "cleanup":
            return (
                document.get("status") == "cleaned"
                and document.get("passed") is True
            )
        return document.get("status") == "verified" and document.get("passed") is True

    def _fixture_operation(self, fixture_id: str, operation: str) -> bool:
        client = self._fixture_client
        if client is not None:
            try:
                succeeded = client(fixture_id, operation) is True
            except Exception:
                return False
        else:
            succeeded = self._browser_fixture(fixture_id, operation)
        if succeeded and fixture_id == "session-v1":
            if operation == "seed":
                self._session_fixture_seeded = True
            elif operation == "cleanup":
                self._session_fixture_seeded = False
        return succeeded

    def _close_browser_fixture(self) -> None:
        try:
            from aeon.tools.browser import BROWSER_API_URL, browser_auth_headers

            requests.post(
                f"{BROWSER_API_URL}/close_session",
                json={
                    "session_id": self.browser_session_id,
                    "profile": self.browser_profile,
                },
                headers=browser_auth_headers(),
                timeout=2,
                allow_redirects=False,
                proxies={"http": "", "https": ""},
            ).close()
        except Exception:
            pass

    @staticmethod
    def _decoded(result: ProcessResult) -> str:
        return result.output.decode("utf-8", errors="replace")

    @staticmethod
    def _timing_record(result: ProcessResult) -> dict[str, float]:
        return {
            "wall_ms": max(0.0, float(result.wall_ms)),
            "active_wall_ms": result.active_wall_ms,
            "compute_wait_ms": max(0.0, float(result.compute_wait_ms)),
        }

    @staticmethod
    def _terminal_failure(result: ProcessResult) -> dict[str, object] | None:
        if result.state in {"timeout", "stuck"}:
            return {
                "status": result.state,
                "score": 0.0,
                **FleetHarnessExecutor._timing_record(result),
            }
        if result.returncode != 0:
            return {
                "status": "failed",
                "score": 0.0,
                **FleetHarnessExecutor._timing_record(result),
            }
        return None

    def _single(
        self,
        request: ExecutionRequest,
        prompt: str,
        predicate: Callable[[str], bool],
        *,
        tool: bool = False,
        vision: bool = False,
        resume: bool = False,
        expected_tools: tuple[str, ...] = (),
    ) -> dict[str, object]:
        result = self._run(request, prompt, resume=resume)
        failure = self._terminal_failure(result)
        if failure is not None:
            return failure
        output = self._decoded(result)
        tool_observed = not expected_tools or self._tool_observed(
            request, output, expected_tools
        )
        passed = bool(predicate(output)) and tool_observed
        record: dict[str, object] = {
            "status": "passed" if passed else "failed",
            "score": 1.0 if passed else 0.0,
            **self._timing_record(result),
        }
        if tool:
            record["tool_success"] = passed
        if vision:
            record["vision_score"] = 1.0 if passed else 0.0
        return record

    @staticmethod
    def _combined_timings(
        record: Mapping[str, object], result: ProcessResult
    ) -> dict[str, float]:
        return {
            field: float(record.get(field, 0.0)) + value
            for field, value in FleetHarnessExecutor._timing_record(result).items()
        }

    @staticmethod
    def _tool_observed(
        request: ExecutionRequest,
        output: str,
        names: tuple[str, ...],
    ) -> bool:
        if request.harness_id == "opencode":
            return any(
                f"OpenCode · {name} ·" in output
                or f"OpenCode · aeon_{name} ·" in output
                for name in names
            )
        return any(f"] {name}(" in output for name in names)

    def execute(self, request: ExecutionRequest) -> Mapping[str, object]:
        with self._lifecycle_lock:
            if self._closed or self._closing:
                raise ExecutorUnavailable("benchmark executor is closed")
            if not self._prepared:
                if not self._active_done.is_set():
                    raise ExecutorUnavailable("a prior benchmark harness is unresolved")
                self._child_cancel = threading.Event()
                self._compute_waiting.clear()
            self._prepared = False
            self._active_done.clear()
        try:
            return self._execute_prepared(request)
        finally:
            self._active_done.set()

    def _execute_prepared(self, request: ExecutionRequest) -> Mapping[str, object]:
        if request.model_id != "local/qwen" or request.tool_profile_id != "fleet-local":
            raise ExecutorUnavailable("benchmark model or tool profile is unavailable")
        case_id = request.scenario.case_id
        if case_id == "smoke.direct":
            marker = "BENCH_SMOKE_DIRECT_7Q2"
            return self._single(
                request,
                f"Return exactly {marker} and nothing else. Do not use tools.",
                lambda output: marker in output,
            )
        if case_id == "smoke.bounded":
            marker = "BENCH_SMOKE_323"
            return self._single(
                request,
                "Compute 17 multiplied by 19. Return only BENCH_SMOKE_ followed "
                "by the decimal answer.",
                lambda output: marker in output,
            )
        if case_id == "tools.local_read":
            marker = "BENCH_READ_LARK_7319"
            path = self.workspace / "fixtures" / "read-token.txt"
            return self._single(
                request,
                f"Use the Aeon open_file tool to read {path}. Return only the token "
                "stored in that file; do not infer it from this request.",
                lambda output: marker in output,
                tool=True,
                expected_tools=("open_file",),
            )
        if case_id == "tools.mutate_verify":
            # Every repetition receives a fresh target and marker.  Reusing the
            # first mutation could otherwise let a later repetition pass on
            # stale state despite a failed or skipped write.
            marker = f"BENCH_WRITE_FINCH_4821_R{request.repetition}"
            path = (
                self.workspace
                / "results"
                / f"mutation-repetition-{request.repetition}.txt"
            )

            def changed(_output: str) -> bool:
                try:
                    metadata = path.lstat()
                    return (
                        stat.S_ISREG(metadata.st_mode)
                        and not stat.S_ISLNK(metadata.st_mode)
                        and metadata.st_uid == os.geteuid()
                        and metadata.st_nlink == 1
                        and path.read_text(encoding="utf-8").strip() == marker
                    )
                except (OSError, UnicodeError):
                    return False

            return self._single(
                request,
                f"Use the Aeon write_file tool to create {path} containing exactly "
                f"{marker} followed by a newline. Then report completion.",
                changed,
                tool=True,
                expected_tools=("write_file",),
            )
        if case_id == "tools.fleet_wait":
            result = self._run(
                request,
                "Call fleet_batch_capabilities exactly once, inspect its structured "
                "result, then briefly report whether it exposes only reviewed recipes "
                "and durable waiting. Do not infer the answer from this request.",
            )
            failure = self._terminal_failure(result)
            if failure is not None:
                failure["tool_success"] = False
                return failure
            # This receipt is emitted by the typed tool implementation through
            # an executor-created HMAC-bound file.  Neither the prompt nor model
            # output participates in this success decision.
            passed = len(result.capability_receipts) == 1
            return {
                "status": "passed" if passed else "failed",
                "score": 1.0 if passed else 0.0,
                **self._timing_record(result),
                "tool_success": passed,
            }
        if case_id == "browser.observe":
            if not self._fixture_operation("observe-v1", "seed"):
                return {"status": "unsupported", "score": 0.0, "browser_success": False}
            result = self._single(
                request,
                "Use browser_read on tab_id benchmark with include_vision false. "
                "Return only the controlled value shown on the page.",
                lambda output: "ORBIT-5521" in output,
                tool=True,
                expected_tools=("browser_read",),
            )
            result["browser_success"] = result.get("status") == "passed"
            return result
        if case_id == "browser.form":
            if not self._fixture_operation("form-v1", "seed"):
                return {"status": "unsupported", "score": 0.0, "browser_success": False}
            result = self._single(
                request,
                "Use the browser tools on tab_id benchmark. Fill first name Ada, "
                "email ada@example.invalid, choose Research, accept the test terms, "
                "and submit the registration. Stop after the page confirms acceptance.",
                lambda _output: self._fixture_operation("form-v1", "verify"),
                tool=True,
                expected_tools=("browser_interact",),
            )
            result["browser_success"] = result.get("status") == "passed"
            return result
        if case_id == "browser.session":
            if not self._fixture_operation("session-v1", "seed"):
                return {"status": "unsupported", "score": 0.0, "browser_success": False}
            first = self._single(
                request,
                "Use browser tools on tab_id benchmark. Activate Sign in to fixture, "
                "but do not activate Continue session. Stop after the page says Signed in.",
                lambda _output: True,
                tool=True,
                expected_tools=("browser_interact",),
            )
            if first.get("status") != "passed":
                first["browser_success"] = False
                return first
            # The authenticated fixture endpoint proves the sign-in state,
            # closes the exact tab, and opens a fresh page in the same browser
            # profile/context.  A second harness turn must then use that state.
            if not self._fixture_operation("session-v1", "reopen"):
                first.update({"status": "failed", "score": 0.0, "browser_success": False})
                return first
            second = self._single(
                request,
                "The controlled benchmark tab was closed and reopened. Use browser "
                "tools on tab_id benchmark and activate Continue session. Stop after "
                "the page says Session preserved.",
                lambda _output: self._fixture_operation("session-v1", "verify"),
                tool=True,
                resume=True,
                expected_tools=("browser_interact",),
            )
            for field, value in self._combined_timings(second, ProcessResult(
                state="exited",
                returncode=0,
                output=b"",
                wall_ms=float(first.get("wall_ms", 0.0)),
                compute_wait_ms=float(first.get("compute_wait_ms", 0.0)),
            )).items():
                second[field] = value
            second["browser_success"] = second.get("status") == "passed"
            return second
        if case_id == "vision.image":
            path = self.workspace / "fixtures" / "vision.png"
            return self._single(
                request,
                f"Use analyze_image on {path}. Report the observed colors exactly as "
                "left=<color>;right=<color>.",
                lambda output: "left=red" in output.lower()
                and "right=blue" in output.lower(),
                tool=True,
                vision=True,
                expected_tools=("analyze_image",),
            )
        if case_id == "vision.browser":
            if not self._fixture_operation("vision-v1", "seed"):
                return {
                    "status": "unsupported",
                    "score": 0.0,
                    "browser_success": False,
                    "vision_score": 0.0,
                }
            result = self._single(
                request,
                "Use browser_read on tab_id benchmark with include_vision true and a "
                "clean visual. Report only the number of large circles and their color.",
                lambda output: "3" in output and "magenta" in output.lower(),
                tool=True,
                vision=True,
                expected_tools=("browser_read",),
            )
            result["browser_success"] = result.get("status") == "passed"
            return result
        if case_id == "context.recall":
            if request.harness_id != "opencode":
                return {"status": "unsupported", "score": 0.0}
            setup = self._run(
                request,
                "Remember the token BENCH_CONTEXT_OTTER_6043 for the next turn. "
                "Reply only ACK.",
            )
            failure = self._terminal_failure(setup)
            if failure is not None:
                return failure
            recalled = self._single(
                request,
                "Return only the token I asked you to remember in the prior turn.",
                lambda output: "BENCH_CONTEXT_OTTER_6043" in output,
                resume=True,
            )
            recalled.update(self._combined_timings(recalled, setup))
            return recalled
        if case_id == "context.loop":
            marker = "BENCH_LOOP_STOPPED"
            missing = self.workspace / "fixtures" / "intentionally-missing.txt"
            return self._single(
                request,
                f"Try the open_file tool on {missing} once. When it reports that the "
                f"file is absent, do not retry the unchanged action; return {marker}.",
                lambda output: marker in output,
                tool=True,
                expected_tools=("open_file",),
            )
        if case_id == "context.compaction":
            # There is no deterministic public compaction trigger in either harness.
            return {"status": "unsupported", "score": 0.0}
        return {"status": "unsupported", "score": 0.0}


__all__ = (
    "FleetHarnessExecutor",
    "BrowserFixtureClient",
    "BENCHMARK_BROWSER_PROFILE",
    "EXECUTOR_CLOSE_GRACE_SECONDS",
    "MAX_CAPTURE_BYTES",
    "ProcessResult",
    "ProcessRunner",
    "ReadinessChecker",
    "run_bounded_process",
    "runtime_execution_status",
)
