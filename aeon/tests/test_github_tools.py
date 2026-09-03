from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from aeon.core.agent_protocol import SideEffect, ToolStatus, infer_tool_policy
from aeon.core.tool_resources import ToolComputeRoute, tool_resource_policy
from aeon.remote.mcp_capability import MCP_URL_ENV
from aeon.remote.self_settings import (
    SELF_SETTINGS_TOKEN_FILE_ENV,
    SELF_SETTINGS_TOKEN_FILENAME,
)
from aeon.tools.github import (
    GitHubCommitTool,
    GitHubPushTool,
    GitHubRepositoriesTool,
    GitHubStatusTool,
    GitHubVerifyRemoteTool,
)


class _Response:
    def __init__(self, payload: dict[str, object]):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, amount: int = -1) -> bytes:
        return self.body if amount < 0 else self.body[:amount]


class GitHubToolTests(unittest.TestCase):
    def _environment(self, directory: Path) -> dict[str, str]:
        token_path = directory / SELF_SETTINGS_TOKEN_FILENAME
        token_path.write_text("x" * 64, encoding="ascii")
        token_path.chmod(0o600)
        return {
            MCP_URL_ENV: "http://127.0.0.1:8765/internal/agent/mcp",
            SELF_SETTINGS_TOKEN_FILE_ENV: str(token_path),
            "AEON_REMOTE_INSTANCE_ID": "a" * 32,
        }

    def test_tools_have_distinct_static_effects_and_one_reviewed_route(self):
        expected = {
            "github_repositories": SideEffect.READ_ONLY,
            "github_status": SideEffect.READ_ONLY,
            "github_commit": SideEffect.LOCAL_MUTATION,
            "github_push": SideEffect.EXTERNAL_MUTATION,
            "github_verify_remote": SideEffect.READ_ONLY,
        }
        for name, effect in expected.items():
            with self.subTest(name=name):
                self.assertEqual(infer_tool_policy(name).side_effect, effect)
                self.assertEqual(
                    tool_resource_policy(name).route,
                    ToolComputeRoute.EXTERNAL_PROVIDER,
                )

    def test_commit_and_push_return_typed_mutation_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            directory.chmod(0o700)
            environment = self._environment(directory)
            requests = []

            def open_local(request, *, timeout):
                requests.append((request, timeout))
                operation = request.full_url.rsplit("/", 1)[-1]
                if operation == "github-commit":
                    return _Response(
                        {
                            "status": "ok",
                            "operation": "commit",
                            "local_mutation": True,
                            "external_mutation": False,
                            "repository": "/workspace/project",
                            "head": "1" * 40,
                            "committed_paths": ["README.md"],
                        }
                    )
                return _Response(
                    {
                        "status": "ok",
                        "operation": "push",
                        "local_mutation": False,
                        "external_mutation": True,
                        "repository": "/workspace/project",
                        "head": "1" * 40,
                        "remote_head": "1" * 40,
                        "verified": True,
                    }
                )

            with (
                patch.dict(os.environ, environment, clear=True),
                patch("aeon.tools.mcp._open_local", open_local),
            ):
                commit = GitHubCommitTool().execute(
                    "/workspace/project", "Update docs", ["README.md"]
                )
                push = GitHubPushTool().execute("/workspace/project")

            self.assertEqual(commit.status, ToolStatus.OK)
            self.assertTrue(commit.changed)
            self.assertEqual(commit.side_effect, SideEffect.LOCAL_MUTATION)
            self.assertEqual(push.status, ToolStatus.OK)
            self.assertTrue(push.changed)
            self.assertEqual(push.side_effect, SideEffect.EXTERNAL_MUTATION)
            self.assertEqual(requests[0][0].full_url.rsplit("/", 1)[-1], "github-commit")
            self.assertEqual(requests[1][0].full_url.rsplit("/", 1)[-1], "github-push")
            self.assertEqual(json.loads(requests[0][0].data)["paths"], ["README.md"])

    def test_structured_gateway_refusal_stays_blocked_and_nonretryable(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            directory.chmod(0o700)
            environment = self._environment(directory)
            detail = {
                "status": "error",
                "operation": "preflight",
                "error_code": "github_credential_not_allowed",
                "message": "Allow exactly one configured GitHub credential for this agent in Nexus.",
                "retryable": False,
                "local_mutation": False,
                "external_mutation": False,
            }

            def refused(request, *, timeout):
                body = json.dumps({"detail": detail}).encode("utf-8")
                raise urllib.error.HTTPError(
                    request.full_url,
                    403,
                    "Forbidden",
                    {},
                    io.BytesIO(body),
                )

            with (
                patch.dict(os.environ, environment, clear=True),
                patch("aeon.tools.mcp._open_local", refused),
            ):
                result = GitHubRepositoriesTool().execute()

            self.assertEqual(result.status, ToolStatus.BLOCKED)
            self.assertFalse(result.changed)
            self.assertFalse(result.retryable)
            self.assertEqual(result.error_code, "github_credential_not_allowed")

    def test_unverified_push_success_is_never_mapped_to_no_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            directory.chmod(0o700)
            environment = self._environment(directory)

            def unverified(_request, *, timeout):
                return _Response(
                    {
                        "status": "ok",
                        "operation": "push",
                        "local_mutation": False,
                        "external_mutation": False,
                        "repository": "/workspace/project",
                        "remote": {"name": "origin"},
                        "head": "1" * 40,
                        "remote_head": "2" * 40,
                        "verified": False,
                    }
                )

            with (
                patch.dict(os.environ, environment, clear=True),
                patch("aeon.tools.mcp._open_local", unverified),
            ):
                result = GitHubPushTool().execute("/workspace/project")

            self.assertEqual(result.status, ToolStatus.FAILED)
            self.assertNotEqual(result.status, ToolStatus.NO_CHANGE)
            self.assertFalse(result.changed)
            self.assertEqual(result.error_code, "remote_outcome_ambiguous")
            self.assertTrue(result.raw["verification_required"])

    def test_read_tools_use_only_the_expected_loopback_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            directory.chmod(0o700)
            environment = self._environment(directory)
            urls = []

            def open_local(request, *, timeout):
                urls.append(request.full_url)
                return _Response(
                    {
                        "status": "ok",
                        "operation": "status",
                        "local_mutation": False,
                        "external_mutation": False,
                        "repository": {"path": "/workspace/project"},
                    }
                )

            with (
                patch.dict(os.environ, environment, clear=True),
                patch("aeon.tools.mcp._open_local", open_local),
            ):
                GitHubStatusTool().execute("/workspace/project")
                GitHubVerifyRemoteTool().execute("/workspace/project")

            self.assertTrue(urls[0].endswith("/github-status"))
            self.assertTrue(urls[1].endswith("/github-verify"))


if __name__ == "__main__":
    unittest.main()
