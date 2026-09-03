from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aeon.core.agent_protocol import SideEffect, ToolStatus, infer_tool_policy
from aeon.core.tool_resources import ToolComputeRoute, tool_resource_policy
from aeon.remote.mcp_capability import MCP_URL_ENV
from aeon.remote.self_settings import (
    SELF_SETTINGS_TOKEN_FILE_ENV,
    SELF_SETTINGS_TOKEN_FILENAME,
)
from aeon.tools.huggingface_publish import (
    HuggingFaceAccountTool,
    HuggingFacePublishModelTool,
    HuggingFaceVerifyPublicationTool,
)


class _Response:
    def __init__(self, document: dict[str, object]):
        self.body = json.dumps(document).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, amount: int = -1) -> bytes:
        return self.body if amount < 0 else self.body[:amount]


class HuggingFacePublishToolTests(unittest.TestCase):
    def _environment(self, directory: Path) -> dict[str, str]:
        token_path = directory / SELF_SETTINGS_TOKEN_FILENAME
        token_path.write_text("x" * 64, encoding="ascii")
        token_path.chmod(0o600)
        return {
            MCP_URL_ENV: "http://127.0.0.1:8765/internal/agent/mcp",
            SELF_SETTINGS_TOKEN_FILE_ENV: str(token_path),
            "AEON_REMOTE_INSTANCE_ID": "a" * 32,
        }

    def test_tools_have_reviewed_effects_and_route(self):
        expected = {
            "huggingface_account": SideEffect.READ_ONLY,
            "huggingface_publish_model": SideEffect.EXTERNAL_MUTATION,
            "huggingface_verify_publication": SideEffect.READ_ONLY,
        }
        for name, effect in expected.items():
            with self.subTest(name=name):
                self.assertEqual(infer_tool_policy(name).side_effect, effect)
                self.assertEqual(
                    tool_resource_policy(name).route,
                    ToolComputeRoute.EXTERNAL_PROVIDER,
                )

    def test_publish_sends_only_artifact_identity_and_returns_verified_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = self._environment(Path(temporary))
            captured = {}

            def open_local(request, *, timeout):
                captured["url"] = request.full_url
                captured["payload"] = json.loads(request.data)
                captured["authorization"] = request.headers["Authorization"]
                return _Response(
                    {
                        "status": "ok",
                        "operation": "publish",
                        "repository": "Alday777/useful-model",
                        "commit": "a" * 40,
                        "manifest_sha256": "b" * 64,
                        "verified": True,
                        "visibility": "public",
                        "local_mutation": False,
                        "external_mutation": True,
                    }
                )

            with (
                patch.dict(os.environ, environment, clear=True),
                patch("aeon.tools.mcp._open_local", open_local),
            ):
                result = HuggingFacePublishModelTool().execute(
                    "Alday777/useful-model", "/workspace/release", "Initial release"
                )

            self.assertEqual(result.status, ToolStatus.OK)
            self.assertTrue(result.changed)
            self.assertTrue(captured["url"].endswith("/huggingface-publish"))
            self.assertEqual(captured["payload"]["folder"], "/workspace/release")
            self.assertNotIn("token", captured["payload"])
            self.assertNotIn("hf_", json.dumps(result.raw))
            self.assertTrue(str(captured["authorization"]).startswith("Bearer "))

    def test_read_tools_use_account_and_verify_endpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = self._environment(Path(temporary))
            urls = []

            def open_local(request, *, timeout):
                urls.append(request.full_url)
                return _Response(
                    {
                        "status": "ok",
                        "operation": "account",
                        "account": "Alday777",
                        "verified": True,
                        "local_mutation": False,
                        "external_mutation": False,
                    }
                )

            with (
                patch.dict(os.environ, environment, clear=True),
                patch("aeon.tools.mcp._open_local", open_local),
            ):
                HuggingFaceAccountTool().execute()
                HuggingFaceVerifyPublicationTool().execute(
                    "Alday777/useful-model", "/workspace/release"
                )

            self.assertTrue(urls[0].endswith("/huggingface-account"))
            self.assertTrue(urls[1].endswith("/huggingface-verify"))


if __name__ == "__main__":
    unittest.main()
