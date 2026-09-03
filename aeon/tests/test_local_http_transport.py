"""Hermetic transport-boundary regressions for Aeon's local services."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aeon.core import vision_selftest
from aeon.core.model_catalog import VISION_MODEL_NAME
from aeon.tools.vision import AnalyzeImageTool


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_fleet_benchmark_sitecustomize_is_exact_loopback_only() -> None:
    guard = PACKAGE_ROOT / "aeon/scripts/local_http_sitecustomize"
    program = r'''import json
import requests
import sitecustomize

calls = []
def fake(self, method, url, *args, **kwargs):
    calls.append({
        "method": method,
        "url": url,
        "allow_redirects": kwargs.get("allow_redirects"),
        "proxies": kwargs.get("proxies"),
        "trust_env_during": self.trust_env,
    })
    return object()

sitecustomize._ORIGINAL_REQUEST = fake
session = requests.Session()
session.trust_env = True
session.get("http://127.0.0.1:18033/v1/models", allow_redirects=True)
external_blocked = False
try:
    session.get("http://127.0.0.1:18034/v1/models")
except RuntimeError:
    external_blocked = True
print(json.dumps({"calls": calls, "restored": session.trust_env,
                  "external_blocked": external_blocked}))
'''
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(guard),
        "PYTHONDONTWRITEBYTECODE": "1",
        "AEON_LOCAL_HTTP_PORT": "18033",
        # Deliberately hostile values: the guard must not pass them through.
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "http://127.0.0.1:9",
    }
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PACKAGE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "calls": [
            {
                "method": "GET",
                "url": "http://127.0.0.1:18033/v1/models",
                "allow_redirects": False,
                "proxies": {"http": "", "https": ""},
                "trust_env_during": False,
            }
        ],
        "restored": True,
        "external_blocked": True,
    }


def test_direct_local_requests_calls_explicitly_disable_proxy_and_redirects() -> None:
    """Keep direct local calls honest; MTP is guarded before module import."""

    files = (
        "aeon/core/comfy_fleet_adapter.py",
        "aeon/core/fleet_adapter.py",
        "aeon/core/qwen_fast_service_adapter.py",
        "aeon/core/qwen_fleet_runtime.py",
        "aeon/core/qwen_runtime.py",
        "aeon/core/video_comfy_fleet_adapter.py",
        "aeon/core/vision_selftest.py",
        "aeon/main.py",
        "aeon/scripts/benchmark_qwen38_long_batch.py",
        "aeon/scripts/benchmark_qwen38_speed.py",
        "aeon/scripts/generate_qwen38_dflash_data.py",
        "aeon/scripts/qwen_speed_lab_worker.py",
        "aeon/scripts/warmup_qwen38_vllm.py",
        "aeon/tools/browser.py",
        "aeon/tools/generate_image.py",
        "aeon/tools/generate_video.py",
        "aeon/tools/vision.py",
    )
    violations: list[str] = []
    for relative in files:
        source = (PACKAGE_ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or not isinstance(node.func.value, ast.Name)
                or node.func.value.id != "requests"
                or node.func.attr not in {"get", "post", "put", "patch", "delete", "request"}
            ):
                continue
            keywords = {item.arg for item in node.keywords if item.arg is not None}
            has_guard_bundle = any(item.arg is None for item in node.keywords)
            if not has_guard_bundle and not {"allow_redirects", "proxies"} <= keywords:
                violations.append(f"{relative}:{node.lineno}")
    assert violations == []


def test_analyze_image_rejects_oversized_input_before_decode() -> None:
    tool = AnalyzeImageTool()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "oversized.png"
        with path.open("wb") as handle:
            handle.truncate(tool.MAX_INPUT_BYTES + 1)
        with patch("aeon.tools.vision.Image.open") as image_open:
            error = tool._validate_image(str(path))
    assert "input limit" in error
    image_open.assert_not_called()


def test_analyze_image_rejects_excess_pixels_before_decode() -> None:
    tool = AnalyzeImageTool()
    fake_image = MagicMock()
    fake_image.__enter__.return_value = fake_image
    fake_image.size = (tool.MAX_INPUT_SIDE, tool.MAX_INPUT_SIDE)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "header.png"
        path.write_bytes(b"not-empty")
        with patch("aeon.tools.vision.Image.open", return_value=fake_image):
            error = tool._validate_image(str(path))
    assert "exceed" in error
    fake_image.verify.assert_not_called()


def test_analyze_image_revalidates_fleet_immediately_before_http() -> None:
    from PIL import Image

    tool = AnalyzeImageTool()
    guard = MagicMock(side_effect=RuntimeError("ticket expired"))
    tool.worker = SimpleNamespace(
        compute_guard=guard,
        model_config={
            "provider": "vllm",
            "base_url": "http://127.0.0.1:8033/v1",
            "api_model": VISION_MODEL_NAME,
        },
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "small.png"
        Image.new("RGB", (16, 16), "white").save(path)
        with patch("aeon.tools.vision.requests.post") as post:
            result = tool.execute(str(path), "Describe it")
    assert "Fleet compute changed" in result
    guard.assert_called_once_with()
    post.assert_not_called()


def test_analyze_image_uses_endpoint_promoted_by_adjacent_guard() -> None:
    from PIL import Image

    tool = AnalyzeImageTool()
    config = {
        "provider": "vllm",
        "base_url": "http://127.0.0.1:8033/v1",
        "api_model": VISION_MODEL_NAME,
    }

    def guard() -> None:
        config["base_url"] = "http://127.0.0.1:8044/v1"

    tool.worker = SimpleNamespace(compute_guard=guard, model_config=config)
    response_payload = {
        "choices": [{"message": {"content": "promoted"}}]
    }
    response = MagicMock(status_code=200, headers={})
    response.iter_content.return_value = [json.dumps(response_payload).encode()]
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "small.png"
        Image.new("RGB", (16, 16), "white").save(path)
        with patch("aeon.tools.vision.requests.post", return_value=response) as post:
            result = tool.execute(str(path), "Describe it")
    assert "promoted" in result
    assert post.call_args.args[0] == "http://127.0.0.1:8044/v1/chat/completions"


def test_startup_vision_probe_guards_each_http_attempt() -> None:
    events: list[str] = []

    def guard() -> None:
        events.append("guard")

    def ask(*_args, **_kwargs):
        events.append("http")
        return "ACE346", None

    with (
        patch.object(vision_selftest, "_make_nonce", return_value="ACE346"),
        patch.object(vision_selftest, "_render_probe", return_value=b"jpeg"),
        patch.object(vision_selftest, "_save_probe", return_value="<test>"),
        patch.object(vision_selftest, "_ask_vision", side_effect=ask),
    ):
        assert vision_selftest.run_vision_self_test(
            "http://127.0.0.1:8033/v1",
            VISION_MODEL_NAME,
            compute_guard=guard,
        ) == "ACE346"
    assert events == ["guard", "http", "guard", "http"]


def test_startup_vision_probe_rejects_noncanonical_endpoint_before_guard() -> None:
    guard = MagicMock()
    with pytest.raises(vision_selftest.VisionSelfTestError, match="exact Fleet"):
        vision_selftest.run_vision_self_test(
            "http://localhost:8033/v1",
            VISION_MODEL_NAME,
            compute_guard=guard,
        )
    guard.assert_not_called()
