"""Hermetic regressions for OpenCode completion and cancellation boundaries."""

from __future__ import annotations

import io
import json
import os
import signal
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from aeon.core.process_resources import (
    cancel_process_resources,
    register_receipted_command,
    register_service_owner,
)
from aeon.core.fleet_backend import BrokerServiceSession
from aeon.core.worker import Worker
from aeon.harnesses.opencode_completion import (
    CompletionStateWriter,
    OpenCodeCompletionError,
    authority_sha256,
    validate_completion,
)
from aeon.harnesses.opencode_mcp import (
    _install_termination_handlers,
    _opencode_input_schema,
    _restore_termination_handlers,
)
from aeon.harnesses.model_proxy import FleetModelProxy
from aeon.harnesses.opencode_runtime import OpenCodeRuntimeError, OpenCodeTurnRunner
from aeon.tools.base import BaseTool
from aeon.tools.command_fleet_guard import (
    inaccessible_sandbox_paths,
    scrubbed_payload_environment,
)
from aeon.tools.file_io import StrReplaceTool
from aeon.tools.system import RunCommandTool


@pytest.fixture(autouse=True)
def _restore_test_runtime_write_modes(tmp_path: Path):
    """Let pytest remove test-only trees that intentionally model mode 0500."""

    yield
    for path in tuple(tmp_path.rglob("*")):
        try:
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                path.chmod(0o700)
        except OSError:
            pass


class _LLM:
    context_limit = 100_000
    last_reasoning_content = ""
    last_generation_performance = None

    def set_action_schema(self, _schema) -> None:
        return None

    def set_iteration(self, _iteration) -> None:
        return None


class _Write(BaseTool):
    def __init__(self) -> None:
        super().__init__("write_file", "test writer")

    def execute(self, file_path: str, content: str) -> str:
        return f"Successfully wrote {file_path}."


class _Read(BaseTool):
    def __init__(self) -> None:
        super().__init__("open_file", "test reader")

    def execute(self, file_path: str) -> str:
        return f"Current exact content of {file_path}: x=1"


def _record(worker: Worker, iteration: int, tool_name: str, parameters: dict) -> None:
    turn = {
        "intent": f"test {tool_name}",
        "actions": [{"tool_name": tool_name, "parameters": parameters}],
    }
    receipts, interrupted, restart = worker._execute_protocol_actions(turn, iteration)
    assert not interrupted
    assert not restart
    assert len(receipts) == 1
    assert receipts[0].successful
    worker._record_protocol_tool_turn(turn, receipts, iteration)
    worker.effective_iterations += 1


def _completion_fixture(tmp_path: Path) -> tuple[Worker, CompletionStateWriter, dict]:
    authority = "Replace x.py content with x=1 and verify it."
    worker = Worker(llm_client=_LLM(), print_func=lambda *_args, **_kwargs: None)
    worker.persist_session = False
    worker.instance_id = "a" * 32
    worker.register_tools([_Write(), _Read()])
    worker._begin_protocol_request(authority)
    values = {
        "path": tmp_path / "completion-state.json",
        "key": b"k" * 32,
        "nonce": "b" * 64,
        "authority": authority,
        "instance_id": worker.instance_id,
        "workspace": str(tmp_path.resolve(strict=True)),
        "final_text": "Updated x.py and verified the exact contents.",
    }
    writer = CompletionStateWriter(
        path=values["path"],
        key=values["key"],
        nonce=values["nonce"],
        authority_digest=authority_sha256(authority),
        instance_id=values["instance_id"],
        workspace=values["workspace"],
    )
    return worker, writer, values


def test_completion_gate_requires_exact_post_write_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.chmod(tmp_path, 0o700)
    monkeypatch.chdir(tmp_path)
    worker, writer, values = _completion_fixture(tmp_path)

    _record(worker, 1, "write_file", {"file_path": "x.py", "content": "x=1"})
    writer.publish(worker, tool_invocations=1)
    with pytest.raises(OpenCodeCompletionError, match="exact readback"):
        validate_completion(**values, tool_calls=1)

    _record(worker, 2, "open_file", {"file_path": "x.py"})
    writer.publish(worker, tool_invocations=2)
    validate_completion(**values, tool_calls=2)

    with pytest.raises(OpenCodeCompletionError, match="observed tool calls"):
        validate_completion(**values, tool_calls=3)


def test_completion_gate_rejects_missing_tampered_and_wrong_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.chmod(tmp_path, 0o700)
    monkeypatch.chdir(tmp_path)
    worker, writer, values = _completion_fixture(tmp_path)
    _record(worker, 1, "write_file", {"file_path": "x.py", "content": "x=1"})
    _record(worker, 2, "open_file", {"file_path": "x.py"})
    writer.publish(worker, tool_invocations=2)

    for override in (
        {"nonce": "c" * 64},
        {"authority": "Read a different file."},
        {"instance_id": "d" * 32},
        {"workspace": str(tmp_path / "other")},
    ):
        arguments = dict(values)
        arguments.update(override)
        with pytest.raises(OpenCodeCompletionError, match="another turn"):
            validate_completion(**arguments, tool_calls=2)

    missing = dict(values)
    missing["path"] = tmp_path / "missing.json"
    with pytest.raises(OpenCodeCompletionError, match="unavailable"):
        validate_completion(**missing, tool_calls=2)

    document = json.loads(values["path"].read_text(encoding="utf-8"))
    document["tool_invocations"] = 99
    values["path"].write_text(json.dumps(document), encoding="utf-8")
    os.chmod(values["path"], 0o600)
    with pytest.raises(OpenCodeCompletionError, match="integrity"):
        validate_completion(**values, tool_calls=2)


def test_completion_gate_accepts_plain_answer_but_not_unsupported_change(
    tmp_path: Path,
) -> None:
    validate_completion(
        path=tmp_path / "absent.json",
        key=b"k" * 32,
        nonce="a" * 64,
        authority="What is two plus two?",
        instance_id="b" * 32,
        workspace=str(tmp_path),
        final_text="Four.",
        tool_calls=0,
    )
    with pytest.raises(OpenCodeCompletionError, match="COMPLETION BLOCKED"):
        validate_completion(
            path=tmp_path / "absent.json",
            key=b"k" * 32,
            nonce="a" * 64,
            authority="Update x.py and verify it.",
            instance_id="b" * 32,
            workspace=str(tmp_path),
            final_text="Updated and verified x.py.",
            tool_calls=0,
        )


def test_opencode_schema_adapter_keeps_str_replace_boundary_unambiguous() -> None:
    tool = StrReplaceTool(worker=SimpleNamespace())
    schema = _opencode_input_schema("str_replace", tool)

    assert "oneOf" not in schema
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["expected_sha256", "file_path"]
    assert {"patch", "old_str", "new_str"} <= set(schema["properties"])
    digest = "a" * 64
    assert "exactly one" in tool.validate_parameters(
        {"file_path": "x.py", "expected_sha256": digest}
    )
    assert "exactly one" in tool.validate_parameters(
        {
            "file_path": "x.py",
            "expected_sha256": digest,
            "patch": "<<<< SEARCH\nx\n====\ny\n>>>> REPLACE",
            "old_str": "x",
        }
    )


def test_completion_capability_is_file_backed_and_command_inaccessible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    workspace.mkdir(mode=0o700)
    state.mkdir(mode=0o700)
    monkeypatch.chdir(workspace)
    runner = OpenCodeTurnRunner(
        binary=tmp_path / "opencode",
        root=state,
        proxy=SimpleNamespace(
            base_url="http://127.0.0.1:19001/v1",
            token="proxy-token",
            wire_model="Qwen3.8-27B-ARA-NVFP4-MTP",
        ),
        logical_model="Qwen3.8-27B-ARA-NVFP4-MTP",
        max_steps=4,
        resume=False,
    )

    environment = runner._environment("Inspect x.py")

    assert "AEON_OPENCODE_COMPLETION_KEY" not in environment
    key_path = Path(environment["AEON_OPENCODE_COMPLETION_KEY_FILE"])
    assert key_path.parent == state
    assert key_path.stat().st_size == 32
    assert key_path.stat().st_mode & 0o077 == 0
    scrubbed = scrubbed_payload_environment(environment)
    assert not any(name.startswith("AEON_OPENCODE_") for name in scrubbed)
    assert str(state.resolve()) in inaccessible_sandbox_paths(environment)


def test_turn_runner_rejects_supervisor_state_inside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    state = workspace / "model-writable-state"
    state.mkdir(parents=True, mode=0o700)
    monkeypatch.chdir(workspace)

    with pytest.raises(OpenCodeRuntimeError, match="disjoint"):
        OpenCodeTurnRunner(
            binary=tmp_path / "opencode",
            root=state,
            proxy=SimpleNamespace(),
            logical_model="Qwen3.8-27B-ARA-NVFP4-MTP",
            max_steps=4,
            resume=False,
        )


class _ServiceOwner:
    def __init__(self) -> None:
        self.stop_calls = 0
        self.close_calls = 0

    def request_stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_process_cleanup_stops_only_registered_ephemeral_resources() -> None:
    command_stops: list[str] = []
    owner = _ServiceOwner()
    durable_job = SimpleNamespace(cancel_calls=0)
    register_receipted_command(lambda: command_stops.append("exact-receipt"))
    register_service_owner(owner)

    assert cancel_process_resources() == []
    assert command_stops == ["exact-receipt"]
    assert owner.stop_calls == 1
    assert owner.close_calls == 1
    assert durable_job.cancel_calls == 0
    assert cancel_process_resources() == []
    assert command_stops == ["exact-receipt"]


class _BrokerClient:
    ticket_id = "fd-0123456789abcdef0123456789abcdef"

    def __init__(self) -> None:
        self.consumer = ""
        self.released: list[str] = []

    def _snapshot(self, state: str, compute_state: str) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "profile_id": "aeon-qwen38-standard",
            "service_id": "aeon-qwen38-standard",
            "consumer": self.consumer,
            "state": state,
            "compute_state": compute_state,
            "endpoint": (
                "http://127.0.0.1:8033/v1" if compute_state == "ready" else None
            ),
            "runtime_profiles": (
                ["aeon-qwen38-compact-workers"] if compute_state == "ready" else []
            ),
        }

    def acquire_service(self, **arguments) -> dict:
        self.consumer = arguments["consumer"]
        return self._snapshot("active", "ready")

    def renew_service(self, _ticket_id: str, *, ttl_seconds: float) -> dict:
        assert ttl_seconds > 0
        return self._snapshot("active", "ready")

    def release_service(self, ticket_id: str) -> dict:
        self.released.append(ticket_id)
        return self._snapshot("released", "inactive")


def test_process_cleanup_releases_automatically_registered_broker_session() -> None:
    client = _BrokerClient()
    session = BrokerServiceSession(client=client, consumer="aeon/test-cleanup")
    assert session.start() == "http://127.0.0.1:8033/v1"

    assert cancel_process_resources() == []

    assert client.released == [client.ticket_id]
    assert session.ticket_id is None
    assert session.endpoint is None


def test_broker_session_close_is_serialized_and_idempotent() -> None:
    class BlockingClient(_BrokerClient):
        def __init__(self) -> None:
            super().__init__()
            self.release_entered = threading.Event()
            self.release_continue = threading.Event()

        def release_service(self, ticket_id: str) -> dict:
            self.released.append(ticket_id)
            self.release_entered.set()
            assert self.release_continue.wait(timeout=2)
            return self._snapshot("released", "inactive")

    client = BlockingClient()
    session = BrokerServiceSession(client=client, consumer="aeon/test-close-race")
    session.start()
    results: list[dict[str, str] | None] = []
    first = threading.Thread(target=lambda: results.append(session.close()))
    second = threading.Thread(target=lambda: results.append(session.close()))
    first.start()
    assert client.release_entered.wait(timeout=2)
    second.start()
    time.sleep(0.05)
    assert client.released == [client.ticket_id]
    client.release_continue.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert client.released == [client.ticket_id]
    assert len(results) == 2
    assert {result is None for result in results} == {False, True}


def test_mcp_sigterm_runs_cooperative_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "aeon.harnesses.opencode_mcp.cancel_process_resources",
        lambda: calls.append("cleanup") or [],
    )
    previous = _install_termination_handlers()
    try:
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        with pytest.raises(SystemExit) as stopped:
            handler(signal.SIGTERM, None)
        assert stopped.value.code == 128 + signal.SIGTERM
        assert calls == ["cleanup"]
    finally:
        _restore_termination_handlers(previous)


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = io.StringIO("")
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise AssertionError(f"fake process was not stopped (timeout={timeout})")
        return self.returncode


def test_active_run_command_is_receipt_stopped_on_process_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    process = _FakeProcess()
    receipt = object()
    launched = threading.Event()
    return_handle = threading.Event()
    stopped: list[object] = []
    finalized: list[object] = []
    boundary = SimpleNamespace(cwd=str(tmp_path))

    monkeypatch.setattr(
        "aeon.tools.system.guard_fleet_shell_command", lambda value: value
    )
    monkeypatch.setattr(
        "aeon.tools.system.resolve_command_cwd", lambda *_args, **_kwargs: tmp_path
    )
    monkeypatch.setattr(
        "aeon.tools.system.prepare_fleet_shell_boundary",
        lambda **_kwargs: (boundary, {}),
    )

    def launch(*_args, **kwargs):
        callback = kwargs.get("on_receipt")
        assert callable(callback)
        callback(receipt)
        launched.set()
        assert return_handle.wait(timeout=5)
        return SimpleNamespace(
            process=process,
            receipt=receipt,
            initial_output="",
        )

    def stop(exact_receipt) -> None:
        stopped.append(exact_receipt)
        process.returncode = 143

    monkeypatch.setattr("aeon.tools.system.launch_sandbox_service", launch)
    monkeypatch.setattr("aeon.tools.system.stop_sandbox_service", stop)
    monkeypatch.setattr(
        "aeon.tools.system.finalize_sandbox_service",
        lambda handle: finalized.append(handle),
    )
    result: list[str] = []
    thread = threading.Thread(
        target=lambda: result.append(RunCommandTool().execute("echo test", timeout=30))
    )
    thread.start()
    assert launched.wait(timeout=2)

    try:
        assert cancel_process_resources() == []
    finally:
        return_handle.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert stopped == [receipt]
    assert finalized
    assert result and "COMMAND FAILED" in result[0]


class _Fleet:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.runtime_profiles = ("aeon-qwen38-standard",)
        self.guards = 0

    def set_endpoint_change_handler(self, handler) -> None:
        self.endpoint_change_handler = handler

    def ensure_ready(self) -> None:
        self.guards += 1


class _BlockingModel(BaseHTTPRequestHandler):
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    call_lock = threading.Lock()

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        length = int(self.headers["content-length"])
        self.rfile.read(length)
        with type(self).call_lock:
            type(self).calls += 1
            call_number = type(self).calls
        if call_number == 1:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.flush()
            type(self).entered.set()
            type(self).release.wait(timeout=15)
            return
        body = json.dumps(
            {
                "id": "after-cancel",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def test_model_proxy_cancels_hung_upstream_before_reopening() -> None:
    _BlockingModel.entered = threading.Event()
    _BlockingModel.release = threading.Event()
    _BlockingModel.calls = 0
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _BlockingModel)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    fleet = _Fleet(f"http://127.0.0.1:{upstream.server_port}/v1")
    proxy = FleetModelProxy(fleet)
    proxy.start()
    client_errors: list[str] = []

    def blocked_request() -> None:
        try:
            requests.post(
                proxy.base_url + "/chat/completions",
                headers={"Authorization": f"Bearer {proxy.token}"},
                json={"model": "ignored", "messages": []},
                timeout=15,
            )
        except requests.RequestException as exc:
            client_errors.append(type(exc).__name__)

    client = threading.Thread(target=blocked_request)
    client.start()
    try:
        assert _BlockingModel.entered.wait(timeout=5)
        started = time.monotonic()
        proxy.cancel_active_turn()
        assert time.monotonic() - started < 5
        client.join(timeout=5)
        assert not client.is_alive()

        retried = requests.post(
            proxy.base_url + "/chat/completions",
            headers={"Authorization": f"Bearer {proxy.token}"},
            json={"model": "ignored", "messages": []},
            timeout=5,
        )
        assert retried.status_code == 200
        assert retried.json()["id"] == "after-cancel"
        assert _BlockingModel.calls == 2
    finally:
        _BlockingModel.release.set()
        client.join(timeout=2)
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)
