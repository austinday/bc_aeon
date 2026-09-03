"""Protocol integration against the exact pinned OpenCode executable."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from aeon.core.model_identity import AEON_DEFAULT_MODEL_NAME
from aeon.core.runtime_instructions import RUNTIME_INSTRUCTIONS_ENV
from aeon.harnesses.model_proxy import FleetModelProxy
from aeon.harnesses.opencode_install import OpenCodeInstallError, resolve_opencode_binary
from aeon.harnesses.opencode_runtime import OpenCodeTurnRunner


class _Fleet:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.runtime_profiles = ("aeon-qwen38-standard",)
        self.guards = 0

    def set_endpoint_change_handler(self, handler) -> None:
        self.endpoint_change_handler = handler

    def ensure_ready(self) -> None:
        self.guards += 1


class _StreamingModel(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, _format: str, *_args: object) -> None:
        return

    @staticmethod
    def _chunk(delta: dict, finish_reason: str | None = None) -> dict:
        return {
            "id": "chatcmpl-nexus-test",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "Qwen3.8-27B-ARA-NVFP4-MTP",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        length = int(self.headers["content-length"])
        request = json.loads(self.rfile.read(length))
        type(self).requests.append(request)
        messages = request.get("messages") if isinstance(request, dict) else []
        has_tool_result = any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in (messages if isinstance(messages, list) else [])
        )
        if has_tool_result:
            events = [
                self._chunk({"role": "assistant", "content": "TOOL_ROUND_TRIP_OK"}),
                self._chunk({}, "stop"),
            ]
        else:
            events = [
                self._chunk(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_readme",
                                "type": "function",
                                "function": {
                                    "name": "aeon_open_file",
                                    "arguments": '{"file_path":"README.md"}',
                                },
                            }
                        ],
                    }
                ),
                self._chunk({}, "tool_calls"),
            ]
        body = b"".join(
            b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"
            for event in events
        ) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def test_pinned_opencode_runs_real_mcp_tool_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        binary = resolve_opencode_binary()
    except OpenCodeInstallError as exc:
        pytest.skip(f"pinned OpenCode binary is not installed: {exc}")

    project_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(project_root)
    monkeypatch.delenv(RUNTIME_INSTRUCTIONS_ENV, raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)
    monkeypatch.setenv("AEON_OPENCODE_TURN_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("AEON_BROWSER_SESSION_ID", "oc-" + "a" * 32)
    state_root = tmp_path / "state"
    state_root.mkdir(mode=0o700)

    _StreamingModel.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingModel)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    fleet = _Fleet(f"http://127.0.0.1:{upstream.server_port}/v1")
    proxy = FleetModelProxy(fleet)
    proxy.start()
    try:
        runner = OpenCodeTurnRunner(
            binary=binary,
            root=state_root,
            proxy=proxy,
            logical_model=AEON_DEFAULT_MODEL_NAME,
            max_steps=4,
            resume=False,
        )
        final, metrics = runner.run(
            "Read README.md using open_file and report its first heading."
        )
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    assert final == "TOOL_ROUND_TRIP_OK"
    assert metrics["tool_calls"] == 1
    assert metrics["steps"] == 2
    assert metrics["session_id"]
    assert fleet.guards >= 2
    assert len(_StreamingModel.requests) == 2
    assert all(
        request["model"] == "Qwen3.8-27B-ARA-NVFP4-MTP"
        for request in _StreamingModel.requests
    )
    assert any(
        message.get("role") == "tool" and message.get("content")
        for message in _StreamingModel.requests[-1]["messages"]
        if isinstance(message, dict)
    )
