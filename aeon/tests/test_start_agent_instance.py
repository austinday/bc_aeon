from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aeon.core.durable_agent_guard import VerifiedNexusAgentStart
from aeon.remote.project_manager import PROJECT_MANAGER_INSTANCE_ID
from aeon.tools.start_agent_instance import StartAgentInstanceTool
from aeon.tools import start_agent_instance as start_tool


class _Response:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int) -> bytes:
        return self.payload


class StartAgentInstanceToolTests(unittest.TestCase):
    @staticmethod
    def _registered_tool(request_id: str = "request-" + "a" * 32):
        tool = StartAgentInstanceTool()
        tool.worker = SimpleNamespace(request_id=request_id)
        return tool

    def test_endpoint_is_exact_port_and_canonical_origin(self):
        accepted = (
            "http://127.0.0.1:8765/internal/orchestrator/agents",
            "http://[::1]:8765/internal/orchestrator/agents",
            "http://172.19.0.1:8765/internal/orchestrator/agents",
        )
        for value in accepted:
            with self.subTest(value=value):
                self.assertEqual(StartAgentInstanceTool._endpoint(value), value)
        for value in (
            "http://127.0.0.1/internal/orchestrator/agents",
            "http://127.0.0.1:9999/internal/orchestrator/agents",
            "http://localhost:8765/internal/orchestrator/agents",
            "http://127.0.0.1:8765/internal/orchestrator/agents?next=http://remote",
            "http://127.0.0.1:8765/internal/orchestrator/agents\n",
        ):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                StartAgentInstanceTool._endpoint(value)

    def test_local_transport_disables_proxies_and_redirects(self):
        request = start_tool.urllib.request.Request(
            "http://127.0.0.1:8765/internal/orchestrator/agents"
        )
        opener = Mock()
        with patch.object(
            start_tool.urllib.request, "build_opener", return_value=opener
        ) as build:
            start_tool._local_urlopen(request, timeout=20)
        handlers = build.call_args.args
        self.assertEqual(handlers[0].proxies, {})
        self.assertIsInstance(handlers[1], start_tool._NoRedirect)
        opener.open.assert_called_once_with(request, timeout=20)

    def test_directives_require_explicit_authorization_and_observed_result(self):
        with patch.dict(os.environ, {"AEON_REMOTE_INSTANCE_ID": "ordinary"}, clear=False):
            tool = StartAgentInstanceTool()

        self.assertIn("explicitly directs creation now", tool.description)
        self.assertIn("requests a plan, not this tool", tool.description)
        self.assertIn("stays idle", tool.description)
        self.assertNotIn("max_iterations", tool.description)
        self.assertEqual(tool.directives, [])

    def test_only_project_manager_sees_the_tool(self):
        with patch.dict(os.environ, {"AEON_REMOTE_INSTANCE_ID": "ordinary"}, clear=False):
            tool = StartAgentInstanceTool()
        self.assertTrue(tool.is_internal)
        self.assertIn("only the primary", tool.execute("Worker", "/tmp"))

    def test_project_manager_uses_local_owner_only_capability(self):
        with tempfile.TemporaryDirectory() as temporary:
            token_path = Path(temporary) / "capability"
            token_path.write_text("x" * 64, encoding="ascii")
            token_path.chmod(0o600)
            environment = {
                "AEON_REMOTE_INSTANCE_ID": PROJECT_MANAGER_INSTANCE_ID,
                "NEXUS_INTERNAL_ORCHESTRATOR_URL": (
                    "http://127.0.0.1:8765/internal/orchestrator/agents"
                ),
                "NEXUS_ORCHESTRATOR_TOKEN_FILE": str(token_path),
            }
            captured = {}

            def urlopen(request, timeout):
                captured["request"] = request
                captured["timeout"] = timeout
                return _Response({
                    "instance": {
                        "id": "agent-123",
                        "name": "Research worker",
                        "workspace": "/home/aday/work",
                        "kind": "aeon",
                        "status": "idle",
                        "awaiting_objective": True,
                    }
                })

            with (
                patch.dict(os.environ, environment, clear=False),
                patch("aeon.tools.start_agent_instance._local_urlopen", urlopen),
            ):
                tool = self._registered_tool()
                result = tool.execute(
                    "Research worker",
                    "/home/aday/work",
                )

            self.assertFalse(tool.is_internal)
            self.assertIsInstance(result, VerifiedNexusAgentStart)
            self.assertEqual(result.instance["id"], "agent-123")
            self.assertIn("agent-123", str(result))
            self.assertEqual(captured["timeout"], 20)
            request = captured["request"]
            self.assertEqual(request.full_url, environment["NEXUS_INTERNAL_ORCHESTRATOR_URL"])
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(
                request.get_header("Authorization"),
                f"Bearer {'x' * 64}",
            )
            self.assertEqual(
                request.get_header("X-nexus-orchestrator-instance"),
                PROJECT_MANAGER_INSTANCE_ID,
            )
            payload = json.loads(request.data)
            creation_request_id = payload.pop("creation_request_id")
            self.assertRegex(creation_request_id, r"^agent-request-[0-9a-f]{64}$")
            self.assertEqual(len(creation_request_id), 78)
            self.assertEqual(
                payload,
                {
                    "name": "Research worker",
                    "workspace": "/home/aday/work",
                    "kind": "aeon",
                    "goal": "",
                    "personality": "",
                    "system_prompt": "",
                    "continuous_mode": False,
                },
            )
            self.assertIn("awaiting the user's first message", str(result))
            self.assertIn("No Aeon process or objective has started", str(result))

    def test_project_manager_can_start_a_configured_continuous_agent(self):
        with tempfile.TemporaryDirectory() as temporary:
            token_path = Path(temporary) / "capability"
            token_path.write_text("x" * 64, encoding="ascii")
            token_path.chmod(0o600)
            environment = {
                "AEON_REMOTE_INSTANCE_ID": PROJECT_MANAGER_INSTANCE_ID,
                "NEXUS_INTERNAL_ORCHESTRATOR_URL": (
                    "http://127.0.0.1:8765/internal/orchestrator/agents"
                ),
                "NEXUS_ORCHESTRATOR_TOKEN_FILE": str(token_path),
            }
            captured = {}

            def urlopen(request, timeout):
                captured["payload"] = json.loads(request.data)
                return _Response({
                    "instance": {
                        "id": "agent-continuous",
                        "name": "Portfolio steward",
                        "workspace": "/home/aday/work",
                        "kind": "aeon",
                        "status": "running",
                        "awaiting_objective": False,
                        "continuous_mode": {
                            "enabled": True,
                            "goal": "maximize risk adjusted returns",
                        },
                    }
                })

            with (
                patch.dict(os.environ, environment, clear=False),
                patch("aeon.tools.start_agent_instance._local_urlopen", urlopen),
            ):
                result = self._registered_tool().execute(
                    "Portfolio steward",
                    "/home/aday/work",
                    goal="maximize risk adjusted returns",
                    personality="Analytical and skeptical",
                    system_prompt="Never trade without existing user authority.",
                    continuous_mode=True,
                    allowed_credentials=["github.token", "github", "bitcoin_exact"],
                )

        self.assertIsInstance(result, VerifiedNexusAgentStart)
        self.assertIn("continuous mode enabled", str(result))
        creation_request_id = captured["payload"].pop("creation_request_id")
        self.assertRegex(creation_request_id, r"^agent-request-[0-9a-f]{64}$")
        self.assertEqual(
            captured["payload"],
            {
                "name": "Portfolio steward",
                "workspace": "/home/aday/work",
                "kind": "aeon",
                "goal": "maximize risk adjusted returns",
                "personality": "Analytical and skeptical",
                "system_prompt": "Never trade without existing user authority.",
                "continuous_mode": True,
                "allowed_credentials": ["github", "bitcoin_exact"],
            },
        )

    def test_creation_request_identity_is_stable_per_request_and_agent_name(self):
        first = self._registered_tool("request-one")
        retry = self._registered_tool("request-one")
        other_name = self._registered_tool("request-one")
        other_turn = self._registered_tool("request-two")

        key = first._creation_request_id("Research Agent")
        self.assertEqual(key, retry._creation_request_id("research agent"))
        self.assertNotEqual(key, other_name._creation_request_id("Build Agent"))
        self.assertNotEqual(key, other_turn._creation_request_id("Research Agent"))
        self.assertRegex(key, r"^agent-request-[0-9a-f]{64}$")
        self.assertEqual(len(key), 78)

    def test_project_binding_is_forwarded_and_bound_to_the_verified_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            token_path = Path(temporary) / "capability"
            token_path.write_text("x" * 64, encoding="ascii")
            token_path.chmod(0o600)
            project_id = "pr-" + "a" * 32
            environment = {
                "AEON_REMOTE_INSTANCE_ID": PROJECT_MANAGER_INSTANCE_ID,
                "NEXUS_INTERNAL_ORCHESTRATOR_URL": (
                    "http://127.0.0.1:8765/internal/orchestrator/agents"
                ),
                "NEXUS_ORCHESTRATOR_TOKEN_FILE": str(token_path),
            }
            captured = {}

            def urlopen(request, timeout):
                captured["payload"] = json.loads(request.data)
                return _Response({
                    "instance": {
                        "id": "agent-project",
                        "name": "Project worker",
                        "workspace": "/home/aday/work",
                        "kind": "aeon",
                        "status": "idle",
                        "awaiting_objective": True,
                        "project_id": project_id,
                    }
                })

            with (
                patch.dict(os.environ, environment, clear=False),
                patch("aeon.tools.start_agent_instance._local_urlopen", urlopen),
            ):
                result = self._registered_tool().execute(
                    "Project worker",
                    "/home/aday/work",
                    project_id=project_id,
                )

            self.assertIsInstance(result, VerifiedNexusAgentStart)
            self.assertEqual(result.instance["project_id"], project_id)
            self.assertEqual(captured["payload"]["project_id"], project_id)

            def mismatched_urlopen(request, timeout):
                return _Response({
                    "instance": {
                        "id": "agent-project",
                        "name": "Project worker",
                        "workspace": "/home/aday/work",
                        "kind": "aeon",
                        "status": "idle",
                        "awaiting_objective": True,
                        "project_id": "pr-" + "b" * 32,
                    }
                })

            with (
                patch.dict(os.environ, environment, clear=False),
                patch(
                    "aeon.tools.start_agent_instance._local_urlopen",
                    mismatched_urlopen,
                ),
            ):
                mismatch = self._registered_tool().execute(
                    "Project worker",
                    "/home/aday/work",
                    project_id=project_id,
                )
            self.assertNotIsInstance(mismatch, VerifiedNexusAgentStart)
            self.assertIn("mismatched project id", mismatch)

    def test_unsafe_capability_permissions_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            token_path = Path(temporary) / "capability"
            token_path.write_text("x" * 64, encoding="ascii")
            token_path.chmod(0o644)
            with patch.dict(
                os.environ,
                {
                    "AEON_REMOTE_INSTANCE_ID": PROJECT_MANAGER_INSTANCE_ID,
                    "NEXUS_INTERNAL_ORCHESTRATOR_URL": (
                        "http://127.0.0.1:8765/internal/orchestrator/agents"
                    ),
                    "NEXUS_ORCHESTRATOR_TOKEN_FILE": str(token_path),
                },
                clear=False,
            ):
                result = self._registered_tool().execute(
                    "Worker", "/home/aday/work"
                )
            self.assertIn("not owner-safe", result)

    def test_mismatched_nexus_record_is_not_returned_as_verified_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            token_path = Path(temporary) / "capability"
            token_path.write_text("x" * 64, encoding="ascii")
            token_path.chmod(0o600)
            environment = {
                "AEON_REMOTE_INSTANCE_ID": PROJECT_MANAGER_INSTANCE_ID,
                "NEXUS_INTERNAL_ORCHESTRATOR_URL": (
                    "http://127.0.0.1:8765/internal/orchestrator/agents"
                ),
                "NEXUS_ORCHESTRATOR_TOKEN_FILE": str(token_path),
            }
            response = _Response({
                "instance": {
                    "id": "agent-123",
                    "name": "Unexpected agent",
                    "workspace": "/home/aday/work",
                    "kind": "aeon",
                    "status": "running",
                }
            })
            with (
                patch.dict(os.environ, environment, clear=False),
                patch(
                    "aeon.tools.start_agent_instance._local_urlopen",
                    return_value=response,
                ),
            ):
                result = self._registered_tool().execute(
                    "Research worker", "/home/aday/work"
                )

            self.assertNotIsInstance(result, VerifiedNexusAgentStart)
            self.assertIn("mismatched agent name", result)


if __name__ == "__main__":
    unittest.main()
