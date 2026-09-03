"""Interactive Aeon supervisor backed by the pinned OpenCode agent loop."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from aeon.core.chat_transcript import (
    CHAT_TRANSCRIPT_ENV,
    CHAT_WRITER_PID_ENV,
    append_assistant_message_from_environment,
    append_progress_message_from_environment,
    clear_chat_messages_from_environment,
)
from aeon.core.continuous_mode import (
    ContinuousModeError,
    NEXUS_CONTINUOUS_WAKE_COMMAND,
    load_continuous_mode_from_environment,
)
from aeon.core.fleet_backend import BrokerServiceSession, FleetBackendError
from aeon.core.model_identity import AEON_DEFAULT_MODEL_NAME
from aeon.core.presence import (
    Presence,
    process_instance_id,
    sanitize_summary,
    validate_remote_instance_id,
)
from aeon.core.vision_selftest import VisionSelfTestError, run_vision_self_test

from .model_proxy import FleetModelProxy
from .opencode_completion import (
    COMPLETION_AUTHORITY_SHA256_ENV,
    COMPLETION_KEY_FILE_ENV,
    COMPLETION_NONCE_ENV,
    COMPLETION_STATE_ENV,
    OpenCodeCompletionError,
    authority_sha256,
    validate_completion,
)
from .opencode_config import (
    DEFAULT_OPENCODE_STEPS,
    MAX_OPENCODE_STEPS,
    OpenCodeConfigError,
    _atomic_private_bytes,
    _private_directory,
    isolated_environment,
    materialize_authority,
    materialize_config,
    materialize_instructions,
)
from .opencode_install import OpenCodeInstallError, resolve_opencode_binary


SESSION_FILE = "session.json"
MAX_STDERR_BYTES = 64 * 1024
MAX_EVENT_LINE_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_TEXT_BYTES = 4 * 1024 * 1024
CONTINUOUS_IDENTICAL_FAILURE_LIMIT = 3
CONTINUOUS_FAILURE_PLATEAU_LIMIT = 5
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,127}$")
_BROWSER_SESSION_RE = re.compile(r"^oc-[0-9a-f]{32}$")


class OpenCodeRuntimeError(RuntimeError):
    """The pinned OpenCode run could not complete safely."""


class OpenCodeSessionUnavailable(OpenCodeRuntimeError):
    """A saved OpenCode session disappeared and may be retried once fresh."""


def _bounded_timeout() -> float:
    raw = os.environ.get("AEON_OPENCODE_TURN_TIMEOUT_SECONDS", "900")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 900.0
    return max(30.0, min(value, 1800.0))


def _state_root() -> Path:
    transcript = os.environ.get(CHAT_TRANSCRIPT_ENV, "")
    if transcript:
        path = Path(transcript)
        if path.is_absolute():
            return _private_directory(path.parent / "opencode")
    configured = os.environ.get("AEON_STATE_DIR", "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / ".aeon" / "state"
    workspace_id = __import__("hashlib").sha256(
        str(Path.cwd().resolve(strict=True)).encode("utf-8")
    ).hexdigest()[:20]
    # A direct CLI process has no transcript-owned instance directory.  Give it
    # the same process-stable identity used by Worker/presence so two Aeon
    # processes in one workspace never share authority, config, session, or
    # browser-tab files.  Managed Nexus instances already take the transcript
    # branch above and retain restart-stable state there.
    instance_id = validate_remote_instance_id(
        os.environ.get("AEON_REMOTE_INSTANCE_ID")
    ) or process_instance_id()
    return _private_directory(root / "opencode" / workspace_id / instance_id)


def _load_session_id(root: Path) -> str | None:
    path = root / SESSION_FILE
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OpenCodeRuntimeError("OpenCode session state is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > 4096
        ):
            raise OpenCodeRuntimeError("OpenCode session state is not owner-private")
        raw = os.read(descriptor, 4097)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OpenCodeRuntimeError("OpenCode session state is invalid") from exc
    session_id = value.get("session_id") if isinstance(value, dict) else None
    workspace = value.get("workspace") if isinstance(value, dict) else None
    if not isinstance(session_id, str) or not _SESSION_RE.fullmatch(session_id):
        raise OpenCodeRuntimeError("OpenCode session identity is invalid")
    if workspace != str(Path.cwd().resolve(strict=True)):
        raise OpenCodeRuntimeError("OpenCode session belongs to another workspace")
    return session_id


def _save_session_id(root: Path, session_id: str) -> None:
    if not _SESSION_RE.fullmatch(session_id):
        raise OpenCodeRuntimeError("OpenCode returned an invalid session identity")
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "workspace": str(Path.cwd().resolve(strict=True)),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    _atomic_private_bytes(root, SESSION_FILE, payload)


def _discard_session_id(root: Path) -> None:
    path = root / SESSION_FILE
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            raise OpenCodeRuntimeError("OpenCode session state is unsafe")
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise OpenCodeRuntimeError("OpenCode session state could not be reset") from exc


def _stderr_reader(stream: Any, sink: collections.deque[bytes]) -> None:
    total = 0
    while True:
        chunk = stream.read(4096)
        if not chunk:
            return
        if total < MAX_STDERR_BYTES:
            retained = chunk[: MAX_STDERR_BYTES - total]
            sink.append(retained)
            total += len(retained)


def _terminate_child(child: subprocess.Popen[bytes]) -> None:
    # OpenCode owns an MCP subprocess. Signal the exact session/process group
    # created below so a bounded turn cannot strand descendants.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        if child.poll() is None:
            child.terminate()
    # The OpenCode leader may exit before its MCP grandchild finishes stopping a
    # receipted transient service. Wait for the exact process group, not merely
    # the leader, so a retry cannot overlap that cleanup.
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        time.sleep(0.05)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        if child.poll() is None:
            child.kill()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            child.kill()
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def _install_termination_handlers() -> dict[int, Any]:
    """Turn supervisor termination into a cleanup-preserving Python exit."""

    if threading.current_thread() is not threading.main_thread():
        return {}

    def terminate(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    previous: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate)
    return previous


def _restore_termination_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


class OpenCodeTurnRunner:
    def __init__(
        self,
        *,
        binary: Path,
        root: Path,
        proxy: FleetModelProxy,
        logical_model: str,
        max_steps: int,
        resume: bool,
        browser_profile: str = "default",
    ) -> None:
        workspace = Path.cwd().resolve(strict=True)
        exact_root = root.resolve(strict=True)
        if (
            exact_root == workspace
            or exact_root.is_relative_to(workspace)
            or workspace.is_relative_to(exact_root)
        ):
            raise OpenCodeRuntimeError(
                "OpenCode supervisor state must be disjoint from the workspace"
            )
        self.binary = binary
        self.root = exact_root
        self.proxy = proxy
        self.logical_model = logical_model
        self.max_steps = max_steps
        raw_profile = str(browser_profile or "default")
        safe_profile = "".join(
            character if character.isalnum() or character in "-_." else "-"
            for character in raw_profile
        )
        self.browser_profile = safe_profile.strip("-.")[:64] or "default"
        self.session_id = _load_session_id(exact_root) if resume else None
        self.turns = 0
        self._completion_key = b""
        self._completion_nonce = ""
        self._completion_authority = ""
        self._completion_instance_id = ""

    def clear(self) -> None:
        self.session_id = None
        _discard_session_id(self.root)

    def _environment(self, prompt: str) -> dict[str, str]:
        remote_instance_id = validate_remote_instance_id(
            os.environ.get("AEON_REMOTE_INSTANCE_ID")
        )
        worker_instance_id = remote_instance_id or process_instance_id()
        authority = materialize_authority(self.root, prompt)
        instructions = materialize_instructions(
            self.root,
            instance_id=remote_instance_id,
        )
        config = materialize_config(
            self.root,
            base_url=self.proxy.base_url,
            bearer_token=self.proxy.token,
            instruction_path=instructions,
            max_steps=self.max_steps,
        )
        environment = isolated_environment(
            os.environ,
            directory=self.root,
            config_path=config,
            authority_path=authority,
            base_url=self.proxy.base_url,
            bearer_token=self.proxy.token,
            logical_model=self.logical_model,
            wire_model=self.proxy.wire_model,
        )
        environment["AEON_BROWSER_SESSION_ID"] = os.environ.setdefault(
            "AEON_BROWSER_SESSION_ID", f"oc-{uuid.uuid4().hex}"
        )
        environment["AEON_BROWSER_PROFILE"] = self.browser_profile
        # Each `opencode run` creates a fresh MCP process. Bind all of those
        # workers to the supervisor's stable identity so reviewed Aeon memory
        # and request state can be restored rather than split by child PID.
        environment["AEON_OPENCODE_INSTANCE_ID"] = worker_instance_id
        environment["AEON_OPENCODE_BROWSER_STATE"] = str(self.root / "browser-tab.txt")
        # This per-turn capability lets the MCP process authenticate its exact
        # legacy Worker evidence to this supervisor. It is never inherited from
        # the caller or exposed to command payloads.
        self._completion_key = secrets.token_bytes(32)
        self._completion_nonce = secrets.token_hex(32)
        self._completion_authority = str(prompt or "")
        self._completion_instance_id = worker_instance_id
        environment[COMPLETION_STATE_ENV] = str(self.root / "completion-state.json")
        key_path = _atomic_private_bytes(
            self.root,
            "completion-key.bin",
            self._completion_key,
        )
        environment[COMPLETION_KEY_FILE_ENV] = str(key_path)
        environment[COMPLETION_NONCE_ENV] = self._completion_nonce
        environment[COMPLETION_AUTHORITY_SHA256_ENV] = authority_sha256(prompt)
        return environment

    def run(self, prompt: str) -> tuple[str, dict[str, Any]]:
        from aeon.core.console import TurnStopRequested, console

        environment = self._environment(prompt)
        command = [
            str(self.binary),
            "--pure",
            "run",
            "--format",
            "json",
            "--auto",
            "--model",
            "nexus-fleet/qwen",
            "--agent",
            "aeon",
            "--title",
            "Aeon",
            "--dir",
            str(Path.cwd()),
        ]
        if self.session_id:
            command.extend(["--session", self.session_id])

        child = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path.cwd(),
            env=environment,
            start_new_session=True,
        )
        assert child.stdin is not None and child.stdout is not None and child.stderr is not None
        try:
            child.stdin.write(prompt.encode("utf-8"))
            child.stdin.close()
        except BaseException:
            _terminate_child(child)
            cancel_active = getattr(self.proxy, "cancel_active_turn", None)
            if callable(cancel_active):
                cancel_active()
            for stream in (child.stdin, child.stdout, child.stderr):
                try:
                    stream.close()
                except OSError:
                    pass
            raise
        selector: selectors.BaseSelector | None = None
        input_console = None
        error_thread: threading.Thread | None = None
        gateway_cleanup_error: BaseException | None = None
        try:
            errors: collections.deque[bytes] = collections.deque()
            error_thread = threading.Thread(
                target=_stderr_reader, args=(child.stderr, errors), daemon=True
            )
            error_thread.start()
            output_text: list[str] = []
            tool_calls = 0
            steps = 0
            output_bytes = 0
            session_missing = False
            started = time.monotonic()
            deadline = started + _bounded_timeout()
            selector = selectors.DefaultSelector()
            stdout_fd = child.stdout.fileno()
            os.set_blocking(stdout_fd, False)
            selector.register(stdout_fd, selectors.EVENT_READ)
            pending = bytearray()
            input_console = console()
            input_console.enable_typeahead()
            stopped = False
            timed_out = False
        except BaseException:
            _terminate_child(child)
            cancel_active = getattr(self.proxy, "cancel_active_turn", None)
            if callable(cancel_active):
                cancel_active()
            if selector is not None:
                selector.close()
            if input_console is not None:
                try:
                    input_console.disable_typeahead()
                except Exception:
                    pass
            if error_thread is not None and error_thread.is_alive():
                error_thread.join(timeout=2)
            for stream in (child.stdout, child.stderr):
                try:
                    stream.close()
                except OSError:
                    pass
            raise

        def handle_event(raw: bytes) -> None:
            nonlocal output_bytes, session_missing, steps, tool_calls
            if not raw.strip():
                return
            try:
                event = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                return
            if not isinstance(event, dict):
                return
            event_session = event.get("sessionID")
            if isinstance(event_session, str) and _SESSION_RE.fullmatch(event_session):
                if self.session_id is None:
                    self.session_id = event_session
                    _save_session_id(self.root, event_session)
                elif event_session != self.session_id:
                    raise OpenCodeRuntimeError("OpenCode event changed session identity")
            event_type = event.get("type")
            part = event.get("part") if isinstance(event.get("part"), dict) else {}
            if event_type == "text":
                text = str(part.get("text") or "").strip()
                if text:
                    output_bytes += len(text.encode("utf-8"))
                    if output_bytes > MAX_OUTPUT_TEXT_BYTES:
                        raise OpenCodeRuntimeError(
                            "OpenCode response exceeded its bounded output limit"
                        )
                    output_text.append(text)
            elif event_type == "tool_use":
                tool_calls += 1
                tool_name = str(part.get("tool") or "tool")[:100]
                state = part.get("state") if isinstance(part.get("state"), dict) else {}
                status = str(state.get("status") or "completed")[:40]
                progress = f"OpenCode · {tool_name} · {status}"
                print(progress, flush=True)
                append_progress_message_from_environment(progress)
            elif event_type == "step_finish":
                steps += 1
            elif event_type == "error":
                error_summary = json.dumps(event, ensure_ascii=True, default=str)[:4096]
                session_missing = "session not found" in error_summary.lower()
                print("OpenCode reported a model/session error.", file=sys.stderr)

        try:
            with input_console.interruptible():
                stdout_eof = False
                while True:
                    if time.monotonic() >= deadline:
                        timed_out = True
                        _terminate_child(child)
                        break
                    if input_console.has_stop_request():
                        stopped = True
                        _terminate_child(child)
                        break
                    ready = selector.select(timeout=0.25)
                    if ready:
                        try:
                            chunk = os.read(stdout_fd, 64 * 1024)
                        except BlockingIOError:
                            chunk = None
                        if chunk == b"":
                            stdout_eof = True
                        elif chunk:
                            pending.extend(chunk)
                            while True:
                                newline = pending.find(b"\n")
                                if newline < 0:
                                    break
                                raw = bytes(pending[:newline])
                                del pending[: newline + 1]
                                handle_event(raw)
                            if len(pending) > MAX_EVENT_LINE_BYTES:
                                raise OpenCodeRuntimeError(
                                    "OpenCode emitted an oversized event"
                                )
                    if stdout_eof:
                        if pending:
                            handle_event(bytes(pending))
                            pending.clear()
                        break
        except (TurnStopRequested, KeyboardInterrupt):
            stopped = True
            _terminate_child(child)
        finally:
            selector.close()
            input_console.disable_typeahead()
            if stopped:
                input_console.take_stop_request()
            if child.poll() is None:
                _terminate_child(child)
            try:
                self.proxy.cancel_active_turn()
            except BaseException as exc:
                gateway_cleanup_error = exc
            error_thread.join(timeout=2)
            child.stderr.close()
            child.stdout.close()

        elapsed = time.monotonic() - started
        stderr_text = b"".join(errors).decode("utf-8", errors="replace")
        if gateway_cleanup_error is not None:
            raise OpenCodeRuntimeError(
                "OpenCode model request cancellation could not be proven"
            ) from gateway_cleanup_error
        if stopped:
            raise OpenCodeRuntimeError("The current response was stopped by the owner")
        if timed_out:
            raise OpenCodeRuntimeError("OpenCode exceeded the bounded turn deadline")
        if child.returncode != 0 or session_missing:
            if self.session_id and (
                session_missing or "session not found" in stderr_text.lower()
            ):
                self.clear()
                raise OpenCodeSessionUnavailable(
                    "OpenCode session state was unavailable"
                )
            raise OpenCodeRuntimeError("OpenCode exited before completing the turn")
        final = "\n\n".join(dict.fromkeys(output_text)).strip()
        if not final:
            raise OpenCodeRuntimeError("OpenCode completed without a final response")
        try:
            validate_completion(
                path=self.root / "completion-state.json",
                key=self._completion_key,
                nonce=self._completion_nonce,
                authority=self._completion_authority,
                instance_id=self._completion_instance_id,
                workspace=str(Path.cwd().resolve(strict=True)),
                final_text=final,
                tool_calls=tool_calls,
                project_manager=os.environ.get("AEON_MAIN_ORCHESTRATOR") == "1",
            )
        except OpenCodeCompletionError as exc:
            raise OpenCodeRuntimeError(str(exc)) from exc
        # Text stays private until the supervisor has accepted the exact legacy
        # evidence state. Nexus transcript publication happens in main().
        print(final, flush=True)
        self.turns += 1
        return final, {
            "wall_seconds": round(elapsed, 3),
            "steps": steps,
            "tool_calls": tool_calls,
            "session_id": self.session_id or "",
        }


def _close_browser_session(browser_profile: str) -> None:
    session_id = os.environ.get("AEON_BROWSER_SESSION_ID", "")
    if not _BROWSER_SESSION_RE.fullmatch(session_id):
        return
    try:
        import requests
        from aeon.tools.browser import browser_auth_headers

        requests.post(
            "http://127.0.0.1:8030/close_session",
            json={"session_id": session_id, "profile": browser_profile},
            headers=browser_auth_headers(),
            timeout=2,
            allow_redirects=False,
            proxies={"http": "", "https": ""},
        ).close()
    except Exception:
        return


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aeon-opencode")
    parser.add_argument("--model", required=True)
    parser.add_argument("--start", default="")
    parser.add_argument("--resume-unfinished", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_OPENCODE_STEPS)
    parser.add_argument("--non-interactive", "-n", action="store_true")
    parser.add_argument(
        "--browser-profile",
        default=os.environ.get("AEON_BROWSER_PROFILE", "default"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.model != AEON_DEFAULT_MODEL_NAME:
        print("The OpenCode harness accepts only the reviewed Aeon logical model.", file=sys.stderr)
        return 2
    if not 1 <= args.max_iterations <= MAX_OPENCODE_STEPS:
        print(
            f"OpenCode steps must be between 1 and {MAX_OPENCODE_STEPS}.",
            file=sys.stderr,
        )
        return 2
    if args.non_interactive and not args.start:
        print("--non-interactive requires --start", file=sys.stderr)
        return 2

    os.environ[CHAT_WRITER_PID_ENV] = str(os.getpid())
    if not _BROWSER_SESSION_RE.fullmatch(
        os.environ.get("AEON_BROWSER_SESSION_ID", "")
    ):
        os.environ["AEON_BROWSER_SESSION_ID"] = f"oc-{uuid.uuid4().hex}"
    presence: Presence | None = None
    fleet: BrokerServiceSession | None = None
    proxy: FleetModelProxy | None = None
    release_failed = False
    exit_code = 0
    previous_handlers = _install_termination_handlers()
    try:
        binary = resolve_opencode_binary()
        root = _state_root()
        presence = Presence(cwd=os.getcwd())
        presence.update(phase="startup", model=args.model, intent="Starting OpenCode harness")
        fleet = BrokerServiceSession(
            consumer=f"aeon-opencode/{process_instance_id()}"
        )
        print("[SESSION] Requesting local Qwen through Fleet Compute...", flush=True)
        endpoint = fleet.start()
        wire_model = __import__(
            "aeon.core.model_identity", fromlist=["wire_model_for_runtime_profiles"]
        ).wire_model_for_runtime_profiles(fleet.runtime_profiles)
        if os.environ.get("AEON_SKIP_VISION_SELFTEST") != "1":
            print("[VISION SELF-TEST] Verifying the Fleet model...", flush=True)
            run_vision_self_test(endpoint, wire_model, compute_guard=fleet.ensure_ready)
        else:
            print("[VISION SELF-TEST] Skipped; vision is unverified.", file=sys.stderr)
        proxy = FleetModelProxy(fleet)
        proxy.start()
        runner = OpenCodeTurnRunner(
            binary=binary,
            root=root,
            proxy=proxy,
            logical_model=args.model,
            max_steps=args.max_iterations,
            resume=args.resume_unfinished,
            browser_profile=args.browser_profile,
        )
        presence.update(phase="completed", intent="Ready for a message", model=args.model)
        print(f"Aeon Ready (OpenCode {binary.name}; {args.model})", flush=True)

        next_prompt = str(args.start or "").strip()
        next_is_continuous = False
        repeated_failure_key = ""
        repeated_failure_streak = 0
        consecutive_failure_streak = 0
        while True:
            if not next_prompt:
                if args.non_interactive:
                    break
                from aeon.core.console import TurnStopRequested, console

                try:
                    next_prompt = console().readline("> ").strip()
                    next_is_continuous = False
                except (EOFError, TurnStopRequested, KeyboardInterrupt):
                    break
            if next_prompt in {"exit", "quit"}:
                break
            if next_prompt == NEXUS_CONTINUOUS_WAKE_COMMAND:
                next_prompt = ""
                next_is_continuous = False
                continue
            if next_prompt.lower() == "/clear":
                runner.clear()
                clear_chat_messages_from_environment()
                message = "OpenCode context cleared. The next message starts a fresh session."
                append_assistant_message_from_environment(message)
                print(message, flush=True)
                next_prompt = ""
                next_is_continuous = False
                continue

            current_is_continuous = next_is_continuous
            presence.start_objective(next_prompt, model=args.model)
            turn_error: OpenCodeRuntimeError | None = None
            try:
                final, _metrics = runner.run(next_prompt)
            except OpenCodeSessionUnavailable:
                # One fresh retry avoids making the owner resend a perfectly
                # valid request after OpenCode pruned local session metadata.
                try:
                    final, _metrics = runner.run(next_prompt)
                except OpenCodeRuntimeError as exc:
                    turn_error = exc
            except OpenCodeRuntimeError as exc:
                turn_error = exc
            if turn_error is not None:
                message = str(turn_error)
                append_assistant_message_from_environment(message)
                print(f"[OPENCODE] {message}", file=sys.stderr)
                presence.mark_error(turn_error)
                if current_is_continuous:
                    failure_key = sanitize_summary(
                        type(turn_error).__name__ + ": " + message
                    )
                    repeated_failure_streak = (
                        repeated_failure_streak + 1
                        if failure_key == repeated_failure_key
                        else 1
                    )
                    repeated_failure_key = failure_key
                    consecutive_failure_streak += 1
                else:
                    repeated_failure_key = ""
                    repeated_failure_streak = 0
                    consecutive_failure_streak = 0
            else:
                append_assistant_message_from_environment(final)
                presence.mark_completed()
                repeated_failure_key = ""
                repeated_failure_streak = 0
                consecutive_failure_streak = 0
            next_prompt = ""
            next_is_continuous = False
            if args.non_interactive:
                if turn_error is not None:
                    exit_code = 1
                break
            from aeon.core.console import console

            if console().has_pending():
                next_prompt = console().take_pending() or ""
                next_is_continuous = False
                continue
            try:
                continuous = load_continuous_mode_from_environment()
            except ContinuousModeError as exc:
                print(f"[CONTINUOUS MODE] Disabled: {exc}", file=sys.stderr)
                continuous = None
            if continuous is not None and continuous.enabled:
                circuit_open = current_is_continuous and (
                    repeated_failure_streak >= CONTINUOUS_IDENTICAL_FAILURE_LIMIT
                    or consecutive_failure_streak >= CONTINUOUS_FAILURE_PLATEAU_LIMIT
                )
                if circuit_open:
                    notice = (
                        "Continuous mode paused after repeated OpenCode failures. "
                        "Send a message to retry or change the task."
                    )
                    append_assistant_message_from_environment(notice)
                    print(f"[CONTINUOUS MODE] {notice}", file=sys.stderr)
                else:
                    next_prompt = continuous.prompt()
                    next_is_continuous = True
    except (
        FleetBackendError,
        OpenCodeConfigError,
        OpenCodeInstallError,
        OpenCodeRuntimeError,
        VisionSelfTestError,
        OSError,
        ValueError,
    ) as exc:
        if presence is not None:
            presence.mark_error(exc)
        print(f"Aeon OpenCode startup failed: {sanitize_summary(type(exc).__name__ + ': ' + str(exc))}", file=sys.stderr)
        exit_code = 1
    finally:
        _close_browser_session(args.browser_profile)
        if proxy is not None:
            try:
                proxy.close()
            except Exception:
                exit_code = 1
                print("OpenCode model gateway cleanup failed.", file=sys.stderr)
        if fleet is not None:
            try:
                fleet.close()
            except Exception:
                release_failed = True
                print("Fleet ticket release could not be verified.", file=sys.stderr)
        if presence is not None:
            try:
                presence.mark_exit()
            except Exception:
                exit_code = 1
                print("Aeon presence cleanup failed.", file=sys.stderr)
        if release_failed:
            exit_code = 1
        _restore_termination_handlers(previous_handlers)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "OpenCodeRuntimeError",
    "OpenCodeSessionUnavailable",
    "OpenCodeTurnRunner",
    "main",
)
