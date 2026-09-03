from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aeon.remote.mcp_capability import (
    MCP_DELEGATION_ID_ENV,
    MCP_DELEGATION_TOKEN_FILE_ENV,
    MCP_URL_ENV,
)
from aeon.remote.self_settings import (
    SELF_SETTINGS_TOKEN_FILE_ENV,
    SELF_SETTINGS_TOKEN_FILENAME,
)
from aeon.tools.mcp import (
    ListMcpCredentialsTool,
    ListPaymentAddressesTool,
    ListProviderCredentialsTool,
)


class _Response:
    def __init__(self, payload: dict[str, object]):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, amount: int = -1) -> bytes:
        return self._body if amount < 0 else self._body[:amount]


class McpAgentToolTests(unittest.TestCase):
    def test_bounded_delegation_uses_only_its_expiring_proxy_capability(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            directory.chmod(0o700)
            token_path = directory / "mcp-delegation.token"
            token_path.write_text("x" * 64, encoding="ascii")
            token_path.chmod(0o600)
            delegation_id = "12345678-1234-4234-8234-123456789abc"
            environment = {
                MCP_URL_ENV: "http://127.0.0.1:8765/internal/agent/mcp",
                MCP_DELEGATION_ID_ENV: delegation_id,
                MCP_DELEGATION_TOKEN_FILE_ENV: str(token_path),
            }
            captured = {}

            def open_local(request, *, timeout):
                captured["request"] = request
                captured["timeout"] = timeout
                return _Response(
                    {
                        "credentials": [
                            {
                                "id": "mcp_" + "a" * 32,
                                "label": "Work Gmail",
                                "account": "work@gmail.com",
                            }
                        ]
                    }
                )

            with (
                patch.dict(os.environ, environment, clear=True),
                patch("aeon.tools.mcp._open_local", open_local),
            ):
                tool = ListMcpCredentialsTool()
                result = tool.execute()

            self.assertFalse(tool.is_internal)
            self.assertIn("Work Gmail", result)
            request = captured["request"]
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(
                request.full_url,
                "http://127.0.0.1:8765/internal/agent/mcp/credentials",
            )
            self.assertEqual(
                request.get_header("X-nexus-mcp-delegation"), delegation_id
            )
            self.assertEqual(request.get_header("Authorization"), f"Bearer {'x' * 64}")

    def test_allowed_public_payment_addresses_are_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            directory.chmod(0o700)
            token_path = directory / SELF_SETTINGS_TOKEN_FILENAME
            token_path.write_text("y" * 64, encoding="ascii")
            token_path.chmod(0o600)
            environment = {
                MCP_URL_ENV: "http://127.0.0.1:8765/internal/agent/mcp",
                SELF_SETTINGS_TOKEN_FILE_ENV: str(token_path),
                "AEON_REMOTE_INSTANCE_ID": "a" * 32,
            }
            captured = {}

            def open_local(request, *, timeout):
                captured["request"] = request
                return _Response(
                    {
                        "payment_addresses": [
                            {
                                "id": "bitcoin_" + "b" * 32,
                                "label": "Project donations",
                                "provider": "bitcoin",
                                "public_value": "1BoatSLRHtKNngkdXEeobR76b53LETtpyT",
                            }
                        ]
                    }
                )

            with (
                patch.dict(os.environ, environment, clear=True),
                patch("aeon.tools.mcp._open_local", open_local),
            ):
                result = ListPaymentAddressesTool().execute()

            self.assertIn("1BoatSLRHtKNngkdXEeobR76b53LETtpyT", result)
            self.assertEqual(captured["request"].get_method(), "GET")
            self.assertEqual(
                captured["request"].full_url,
                "http://127.0.0.1:8765/internal/agent/mcp/payment-addresses",
            )

    def test_provider_credentials_use_the_non_mcp_inventory_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            directory.chmod(0o700)
            token_path = directory / SELF_SETTINGS_TOKEN_FILENAME
            token_path.write_text("z" * 64, encoding="ascii")
            token_path.chmod(0o600)
            environment = {
                MCP_URL_ENV: "http://127.0.0.1:8765/internal/agent/mcp",
                SELF_SETTINGS_TOKEN_FILE_ENV: str(token_path),
                "AEON_REMOTE_INSTANCE_ID": "c" * 32,
            }
            captured = {}

            def open_local(request, *, timeout):
                captured["request"] = request
                return _Response(
                    {
                        "credentials": [
                            {
                                "id": "huggingface_" + "d" * 32,
                                "label": "Model publisher",
                                "provider": "huggingface",
                                "account": "example-org",
                            }
                        ]
                    }
                )

            with (
                patch.dict(os.environ, environment, clear=True),
                patch("aeon.tools.mcp._open_local", open_local),
            ):
                result = ListProviderCredentialsTool().execute()

            self.assertIn("Model publisher", result)
            self.assertIn("huggingface", result)
            self.assertEqual(captured["request"].get_method(), "GET")
            self.assertEqual(
                captured["request"].full_url,
                "http://127.0.0.1:8765/internal/agent/mcp/provider-credentials",
            )


if __name__ == "__main__":
    unittest.main()
