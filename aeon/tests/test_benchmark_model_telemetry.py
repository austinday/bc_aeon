"""Hermetic coverage for exact benchmark model-call and token evidence."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import httpx
import openai
import pytest
import requests

from aeon.core.benchmark_model_telemetry import (
    ResponseUsageCapture,
    UsageAccumulator,
    authoritative_token_usage,
    summarize_model_calls,
)
from aeon.core.benchmark_receipt import (
    CAPABILITY_RECEIPT_KEY_ENV,
    CAPABILITY_RECEIPT_PATH_ENV,
    TRACE_CASE_ID_ENV,
    TRACE_NONCE_ENV,
    TRACE_REPETITION_ENV,
    TRACE_RUN_ID_ENV,
    BenchmarkTelemetryError,
    ModelCallReceipt,
    begin_model_call,
    decode_capability_receipts,
    finish_model_call,
)
from aeon.core.llm import LLMClient
from aeon.core.model_catalog import VISION_MODEL_NAME
from aeon.benchmarks.executor import FleetHarnessExecutor, ProcessResult
from aeon.benchmarks import protocol as benchmark_protocol
from aeon.harnesses import model_proxy as model_proxy_module
from aeon.harnesses.model_proxy import FleetModelProxy
from aeon.tools.sub_agent import GetSubAgentReport


def _trace_environment(monkeypatch, tmp_path: Path) -> tuple[Path, str, dict[str, object]]:
    receipt_path = tmp_path / "model.receipt"
    descriptor = os.open(receipt_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    key = "a" * 64
    expected: dict[str, object] = {
        "run_id": "run-" + "b" * 32,
        "case_id": "tools.local_read",
        "repetition": 1,
        "trace_nonce": "c" * 64,
    }
    monkeypatch.setenv(CAPABILITY_RECEIPT_PATH_ENV, str(receipt_path))
    monkeypatch.setenv(CAPABILITY_RECEIPT_KEY_ENV, key)
    monkeypatch.setenv(TRACE_RUN_ID_ENV, str(expected["run_id"]))
    monkeypatch.setenv(TRACE_CASE_ID_ENV, str(expected["case_id"]))
    monkeypatch.setenv(TRACE_REPETITION_ENV, str(expected["repetition"]))
    monkeypatch.setenv(TRACE_NONCE_ENV, str(expected["trace_nonce"]))
    return receipt_path, key, expected


def _decode(path: Path, key: str, expected: dict[str, object]):
    return decode_capability_receipts(path.read_bytes(), key=key, **expected)


def test_receipts_count_errors_without_undercounting_retry_tokens(
    monkeypatch, tmp_path: Path
) -> None:
    path, key, expected = _trace_environment(monkeypatch, tmp_path)
    failed = begin_model_call("aeon_task_model")
    assert finish_model_call(failed, outcome="transport_error")
    succeeded = begin_model_call("aeon_task_model")
    assert finish_model_call(
        succeeded,
        outcome="succeeded",
        prompt_tokens=13,
        completion_tokens=5,
        total_tokens=18,
    )
    records = _decode(path, key, expected)
    assert [item.event_sequence for item in records if isinstance(item, ModelCallReceipt)] == [
        1,
        2,
        3,
        4,
    ]
    assert summarize_model_calls(records, expected_sources=("aeon_task_model",)) == {
        "model_call_count": 2,
        "prompt_tokens": None,
        "peak_prompt_tokens": None,
        "context_tokens": None,
        "completion_tokens": None,
    }


def test_receipts_publish_exact_usage_when_every_call_is_complete(
    monkeypatch, tmp_path: Path
) -> None:
    path, key, expected = _trace_environment(monkeypatch, tmp_path)
    handle = begin_model_call("aeon_task_model")
    assert finish_model_call(
        handle,
        outcome="succeeded",
        prompt_tokens=13,
        completion_tokens=5,
        total_tokens=18,
    )
    second = begin_model_call("aeon_task_model")
    assert finish_model_call(
        second,
        outcome="succeeded",
        prompt_tokens=29,
        completion_tokens=3,
        total_tokens=32,
    )
    assert summarize_model_calls(
        _decode(path, key, expected), expected_sources=("aeon_task_model",)
    ) == {
        "model_call_count": 2,
        "prompt_tokens": 42,
        "peak_prompt_tokens": 29,
        "context_tokens": 50,
        "completion_tokens": 8,
    }

    process = ProcessResult(
        "exited",
        0,
        b"done",
        4.0,
        capability_receipts=_decode(path, key, expected),
        model_call_sources=("aeon_task_model",),
    )
    record = {
        **FleetHarnessExecutor._timing_record(process),
        **FleetHarnessExecutor._observability_record(process),
    }
    FleetHarnessExecutor._merge_process_observability(record, process)
    assert record["model_turn_count"] == 2
    assert record["model_call_count"] == 4
    assert record["prompt_tokens"] == 84
    assert record["peak_prompt_tokens"] == 29
    assert record["context_tokens"] == 100
    assert record["completion_tokens"] == 16


def test_missing_receipts_are_unknown_not_zero() -> None:
    assert summarize_model_calls(
        (), expected_sources=("aeon_task_model",)
    ) == {
        "model_call_count": None,
        "prompt_tokens": None,
        "peak_prompt_tokens": None,
        "context_tokens": None,
        "completion_tokens": None,
    }


def test_model_telemetry_sources_are_content_bound(monkeypatch) -> None:
    llm_source = Path(__file__).resolve().parents[1] / "core" / "llm.py"
    telemetry_source = (
        Path(__file__).resolve().parents[1] / "core" / "benchmark_model_telemetry.py"
    )
    assert llm_source in benchmark_protocol._HARNESS_SOURCES
    assert telemetry_source in benchmark_protocol._HARNESS_SOURCES
    assert telemetry_source in benchmark_protocol._EXECUTOR_SOURCES

    original_read = Path.read_bytes

    def altered_read(path: Path) -> bytes:
        body = original_read(path)
        return body + b"\n# simulated drift\n" if path == llm_source else body

    monkeypatch.setattr(Path, "read_bytes", altered_read)
    assert (
        benchmark_protocol.harness_source_sha256()
        != benchmark_protocol.HARNESS_SOURCE_SHA256
    )


def test_model_transport_cannot_start_when_receipt_append_fails(
    monkeypatch, tmp_path: Path
) -> None:
    _trace_environment(monkeypatch, tmp_path)
    calls = 0

    def create(**_kwargs):
        nonlocal calls
        calls += 1
        return _legacy_stream(usage=None)

    client = _legacy_client(create)
    with patch(
        "aeon.core.benchmark_receipt._stream_transaction", return_value=False
    ):
        with pytest.raises(BenchmarkTelemetryError):
            client.task_completion_create(model=VISION_MODEL_NAME, messages=[])
    assert calls == 0


def test_partial_usage_and_unfinished_attempt_keep_tokens_unknown(
    monkeypatch, tmp_path: Path
) -> None:
    path, key, expected = _trace_environment(monkeypatch, tmp_path)
    first = begin_model_call("aeon_task_model")
    assert finish_model_call(
        first,
        outcome="succeeded",
        prompt_tokens=4,
        completion_tokens=2,
        total_tokens=None,
    )
    assert begin_model_call("aeon_task_model") is not None
    records = _decode(path, key, expected)
    assert summarize_model_calls(records, expected_sources=("aeon_task_model",)) == {
        "model_call_count": 2,
        "prompt_tokens": None,
        "peak_prompt_tokens": None,
        "context_tokens": None,
        "completion_tokens": None,
    }


def test_context_mismatch_and_tampering_cannot_complete_or_decode_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    path, key, expected = _trace_environment(monkeypatch, tmp_path)
    handle = begin_model_call("aeon_task_model")
    monkeypatch.setenv(TRACE_CASE_ID_ENV, "instruction.unknown")
    assert not finish_model_call(
        handle,
        outcome="succeeded",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
    )
    monkeypatch.setenv(TRACE_CASE_ID_ENV, str(expected["case_id"]))
    assert len(_decode(path, key, expected)) == 1
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 1
    assert decode_capability_receipts(bytes(payload), key=key, **expected) == ()


def test_authoritative_usage_accepts_aliases_but_never_derives_total() -> None:
    assert authoritative_token_usage(
        {"input_tokens": 9, "output_tokens": 3, "total_tokens": 12}
    ) is not None
    assert authoritative_token_usage(
        {"prompt_tokens": 9, "completion_tokens": 3}
    ) is None
    assert authoritative_token_usage(
        {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 13}
    ) is None
    accumulator = UsageAccumulator()
    accumulator.observe({"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12})
    accumulator.observe({"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13})
    assert accumulator.result is None


def test_stream_usage_capture_is_fragment_safe_and_missing_usage_stays_unknown() -> None:
    payload = (
        b'data: {"choices":[{"delta":{"content":"ok"}}],"usage":null}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":21,'
        b'"completion_tokens":8,"total_tokens":29}}\n\n'
        b"data: [DONE]\n\n"
    )
    capture = ResponseUsageCapture("text/event-stream; charset=utf-8")
    for offset in range(0, len(payload), 7):
        capture.feed(payload[offset : offset + 7])
    assert capture.finish() == authoritative_token_usage(
        {"prompt_tokens": 21, "completion_tokens": 8, "total_tokens": 29}
    )
    missing = ResponseUsageCapture("application/json")
    missing.feed(b'{"choices":[]}')
    assert missing.finish() is None


class _ProxyUpstream(BaseHTTPRequestHandler):
    statuses = [500, 200]

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers["content-length"]))
        status = type(self).statuses.pop(0)
        document = (
            {"error": {"message": "retry"}}
            if status != 200
            else {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
        )
        body = json.dumps(document).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


class _FleetSession:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.runtime_profiles = ("aeon-qwen38-standard",)

    def set_endpoint_change_handler(self, _handler) -> None:
        return

    def ensure_ready(self) -> None:
        return


def test_opencode_proxy_counts_generation_errors_and_excludes_health_gets(
    monkeypatch, tmp_path: Path
) -> None:
    path, key, expected = _trace_environment(monkeypatch, tmp_path)
    _ProxyUpstream.statuses = [500, 200]
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _ProxyUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = FleetModelProxy(
        _FleetSession(f"http://127.0.0.1:{upstream.server_port}/v1")
    )
    proxy.start()
    session = requests.Session()
    session.trust_env = False
    headers = {"Authorization": f"Bearer {proxy.token}"}
    try:
        assert session.get(proxy.base_url + "/models", headers=headers, timeout=5).status_code == 200
        for expected_status in (500, 200):
            response = session.post(
                proxy.base_url + "/chat/completions",
                headers=headers,
                json={"model": "ignored", "messages": []},
                timeout=5,
            )
            assert response.status_code == expected_status
            response.close()
    finally:
        session.close()
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)
    records = _decode(path, key, expected)
    assert summarize_model_calls(records, expected_sources=("opencode_proxy",)) == {
        "model_call_count": 2,
        "prompt_tokens": None,
        "peak_prompt_tokens": None,
        "context_tokens": None,
        "completion_tokens": None,
    }


class _ProxySSEUpstream(BaseHTTPRequestHandler):
    requests: list[dict[str, object]] = []

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        document = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append(document)
        body = (
            b'data: {"choices":[{"delta":{"content":"ok"}}],"usage":null}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":17,'
            b'"completion_tokens":6,"total_tokens":23}}\n\n'
            b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def test_opencode_benchmark_injects_stream_usage_and_captures_exact_sse(
    monkeypatch, tmp_path: Path
) -> None:
    path, key, expected = _trace_environment(monkeypatch, tmp_path)
    _ProxySSEUpstream.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _ProxySSEUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = FleetModelProxy(
        _FleetSession(f"http://127.0.0.1:{upstream.server_port}/v1")
    )
    proxy.start()
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(
            proxy.base_url + "/chat/completions",
            headers={"Authorization": f"Bearer {proxy.token}"},
            json={"model": "ignored", "messages": [], "stream": True},
            timeout=5,
        )
        assert response.status_code == 200
        assert "data:" in response.text
        response.close()
    finally:
        session.close()
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)
    assert _ProxySSEUpstream.requests[0]["stream_options"] == {"include_usage": True}
    assert summarize_model_calls(
        _decode(path, key, expected), expected_sources=("opencode_proxy",)
    ) == {
        "model_call_count": 1,
        "prompt_tokens": 17,
        "peak_prompt_tokens": 17,
        "context_tokens": 23,
        "completion_tokens": 6,
    }


def test_opencode_nonbenchmark_stream_request_is_not_mutated(monkeypatch) -> None:
    for name in (
        CAPABILITY_RECEIPT_PATH_ENV,
        CAPABILITY_RECEIPT_KEY_ENV,
        TRACE_RUN_ID_ENV,
        TRACE_CASE_ID_ENV,
        TRACE_REPETITION_ENV,
        TRACE_NONCE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    _ProxySSEUpstream.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _ProxySSEUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = FleetModelProxy(
        _FleetSession(f"http://127.0.0.1:{upstream.server_port}/v1")
    )
    proxy.start()
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(
            proxy.base_url + "/chat/completions",
            headers={"Authorization": f"Bearer {proxy.token}"},
            json={"model": "ignored", "messages": [], "stream": True},
            timeout=5,
        )
        assert response.status_code == 200
        response.close()
    finally:
        session.close()
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)
    assert "stream_options" not in _ProxySSEUpstream.requests[0]


def test_opencode_proxy_rechecks_bound_after_benchmark_mutation(
    monkeypatch, tmp_path: Path
) -> None:
    path, key, expected = _trace_environment(monkeypatch, tmp_path)
    _ProxySSEUpstream.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _ProxySSEUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = FleetModelProxy(
        _FleetSession(f"http://127.0.0.1:{upstream.server_port}/v1")
    )
    proxy.start()
    session = requests.Session()
    session.trust_env = False
    raw = b'{"model":"x","messages":[],"stream":true}'
    monkeypatch.setattr(model_proxy_module, "MAX_REQUEST_BYTES", len(raw))
    try:
        response = session.post(
            proxy.base_url + "/chat/completions",
            headers={
                "Authorization": f"Bearer {proxy.token}",
                "Content-Type": "application/json",
            },
            data=raw,
            timeout=5,
        )
        assert response.status_code == 413
        response.close()
    finally:
        session.close()
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)
    assert _ProxySSEUpstream.requests == []
    assert summarize_model_calls(
        _decode(path, key, expected), expected_sources=("opencode_proxy",)
    )["model_call_count"] is None


def _legacy_client(create) -> LLMClient:
    config = {
        "provider": "vllm",
        "model": VISION_MODEL_NAME,
        "api_model": VISION_MODEL_NAME,
        "base_url": "http://127.0.0.1:8000/v1",
    }
    with patch.object(LLMClient, "_create_client", return_value=object()):
        client = LLMClient(config)
    transport = NS(chat=NS(completions=NS(create=create)))
    client.client = transport
    client.utility_client = transport
    client.action_schema = {"type": "object"}
    client._structured_mode = "response_format"
    client._vision_supported = True
    return client


def _legacy_stream(*, usage: object):
    answer = json.dumps({"intent": "done", "actions": []})
    return iter(
        [
            NS(choices=[NS(delta=NS(content=answer), finish_reason=None)], usage=None),
            NS(choices=[NS(delta=NS(content=None), finish_reason="stop")], usage=None),
            NS(choices=[], usage=usage),
        ]
    )


def test_legacy_primary_counts_transport_retry_and_complete_stream_usage(
    monkeypatch, tmp_path: Path
) -> None:
    path, key, expected = _trace_environment(monkeypatch, tmp_path)
    attempts = 0

    def create(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise openai.APIConnectionError(
                request=httpx.Request(
                    "POST", "http://127.0.0.1:8000/v1/chat/completions"
                )
            )
        return _legacy_stream(
            usage=NS(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
                prompt_tokens_details=None,
            )
        )

    client = _legacy_client(create)
    client._handle_connection_error = lambda *_args, **_kwargs: True
    assert json.loads(client.get_primary_agent_response("state"))["intent"] == "done"
    records = _decode(path, key, expected)
    assert summarize_model_calls(records, expected_sources=("aeon_task_model",)) == {
        "model_call_count": 2,
        "prompt_tokens": None,
        "peak_prompt_tokens": None,
        "context_tokens": None,
        "completion_tokens": None,
    }
    assert [
        item.outcome
        for item in records
        if isinstance(item, ModelCallReceipt) and item.phase == "finished"
    ] == ["transport_error", "succeeded"]


def test_legacy_partial_usage_is_unknown_and_support_calls_are_included(
    monkeypatch, tmp_path: Path
) -> None:
    path, key, expected = _trace_environment(monkeypatch, tmp_path)
    client = _legacy_client(
        lambda **_kwargs: _legacy_stream(
            usage=NS(
                prompt_tokens=6,
                completion_tokens=2,
                total_tokens=None,
                prompt_tokens_details=None,
            )
        )
    )
    assert json.loads(client.get_primary_agent_response("state"))["intent"] == "done"
    records = _decode(path, key, expected)
    assert summarize_model_calls(records, expected_sources=("aeon_task_model",)) == {
        "model_call_count": 1,
        "prompt_tokens": None,
        "peak_prompt_tokens": None,
        "context_tokens": None,
        "completion_tokens": None,
    }

    support_calls = 0

    def support(**_kwargs):
        nonlocal support_calls
        support_calls += 1
        return NS(choices=[NS(message=NS(content="summary"))], usage=None)

    client.utility_client = NS(chat=NS(completions=NS(create=support)))
    client.set_iteration(1)
    assert client.summarize_text("text", "query") == "summary"
    assert support_calls == 1
    updated = _decode(path, key, expected)
    assert summarize_model_calls(updated, expected_sources=("aeon_task_model",))[
        "model_call_count"
    ] == 2


def test_llm_client_defers_to_reviewed_opencode_proxy_without_double_counting(
    monkeypatch, tmp_path: Path
) -> None:
    path, key, expected = _trace_environment(monkeypatch, tmp_path)
    proxy_url = "http://127.0.0.1:19001/v1"
    proxy_token = "private-test-token"
    monkeypatch.setenv("AEON_OPENCODE_PROXY_URL", proxy_url)
    monkeypatch.setenv("AEON_OPENCODE_PROXY_TOKEN", proxy_token)
    config = {
        "provider": "vllm",
        "model": VISION_MODEL_NAME,
        "api_model": VISION_MODEL_NAME,
        "base_url": proxy_url,
        "api_key": proxy_token,
    }
    with patch.object(LLMClient, "_create_client", return_value=object()):
        client = LLMClient(config)
    client.client = NS(
        chat=NS(
            completions=NS(
                create=lambda **_kwargs: NS(
                    choices=[NS(message=NS(content="ok"))],
                    usage=NS(
                        prompt_tokens=4,
                        completion_tokens=1,
                        total_tokens=5,
                    ),
                )
            )
        )
    )
    client.task_completion_create(model=VISION_MODEL_NAME, messages=[])
    assert path.read_bytes() == b""
    assert decode_capability_receipts(path.read_bytes(), key=key, **expected) == ()


def test_principal_subagent_monitor_uses_counted_task_generation(tmp_path: Path) -> None:
    output_dir = tmp_path / "sub-agents"
    agent_dir = output_dir / "deadbeef-agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent.log").write_text("working on integration\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def task_completion_create(**kwargs):
        calls.append(kwargs)
        return NS(choices=[NS(message=NS(content="keep waiting"))])

    direct = NS(
        chat=NS(
            completions=NS(
                create=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("direct uncounted model transport was used")
                )
            )
        )
    )
    llm = NS(
        model=VISION_MODEL_NAME,
        task_completion_create=task_completion_create,
        client=direct,
    )
    worker = NS(sub_agent_output_dir=lambda: output_dir)
    tool = GetSubAgentReport(worker=worker, llm_client=llm)
    with patch("aeon.tools.sub_agent.resolve", return_value=(False, "RUNNING", None)):
        result = tool.execute("deadbeef")
    assert "keep waiting" in result
    assert len(calls) == 1
