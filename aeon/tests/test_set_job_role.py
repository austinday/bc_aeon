from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aeon.remote.config import RemoteConfig
from aeon.remote.instances import InstanceManager
from aeon.remote.self_settings import (
    SELF_SETTINGS_TOKEN_FILENAME,
    SELF_SETTINGS_TOKEN_FILE_ENV,
    SELF_SETTINGS_URL_ENV,
    read_self_settings_token,
)
from aeon.scripts.sub_agent_wrapper import SUB_AGENT_FORBIDDEN_TOOLS
from aeon.tools.categories import TOP_LEVEL_TOOLS
from aeon.tools.loader import load_tools_from_directory
from aeon.tools.set_job_role import SetJobRoleTool


class _Response:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int) -> bytes:
        return self.payload


class SetJobRoleToolTests(unittest.TestCase):
    def _environment(self, token_path: Path) -> dict[str, str]:
        return {
            "AEON_REMOTE_INSTANCE_ID": "a" * 32,
            SELF_SETTINGS_URL_ENV: (
                "http://127.0.0.1:8765/internal/agent/job-role"
            ),
            SELF_SETTINGS_TOKEN_FILE_ENV: str(token_path),
        }

    def test_tool_is_hidden_outside_a_managed_session(self):
        with patch.dict(os.environ, {}, clear=True):
            tool = SetJobRoleTool()
            names = {
                item.name
                for item in load_tools_from_directory(
                    "aeon.tools", dependencies={}
                )
            }
        self.assertTrue(tool.is_internal)
        self.assertNotIn("set_job_role", names)
        self.assertIn("managed Nexus session", tool.execute("Release lead"))

    def test_tool_is_top_level_for_principals_and_forbidden_to_sub_agents(self):
        with tempfile.TemporaryDirectory() as temporary:
            token_path = Path(temporary) / "agent-self-settings.token"
            token_path.write_text("x" * 64, encoding="ascii")
            token_path.chmod(0o600)
            with patch.dict(
                os.environ, self._environment(token_path), clear=True
            ):
                names = {
                    item.name
                    for item in load_tools_from_directory(
                        "aeon.tools", dependencies={}
                    )
                }
        self.assertIn("set_job_role", names)
        self.assertIn("set_job_role", TOP_LEVEL_TOOLS)
        self.assertIn("set_job_role", SUB_AGENT_FORBIDDEN_TOOLS)

    def test_manager_issues_distinct_owner_only_capabilities_per_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = RemoteConfig(
                project_root=Path(__file__).resolve().parents[2],
                state_dir=root / "state",
                allowed_roots=(root,),
                allowed_origins=("https://nexus.test",),
                allowed_hosts=("nexus.test",),
                require_totp=False,
            )
            config.prepare_state()
            manager = InstanceManager(object(), config)
            with patch.dict(
                os.environ,
                {
                    "NEXUS_INTERNAL_ORCHESTRATOR_URL": (
                        "http://127.0.0.1:8765/internal/orchestrator/agents"
                    )
                },
                clear=False,
            ):
                first = manager._self_settings_environment({"id": "1" * 32})
                second = manager._self_settings_environment({"id": "2" * 32})

            self.assertEqual(
                first[SELF_SETTINGS_URL_ENV],
                "http://127.0.0.1:8765/internal/agent/job-role",
            )
            first_path = (
                config.instance_state_dir / ("1" * 32) / SELF_SETTINGS_TOKEN_FILENAME
            )
            second_path = (
                config.instance_state_dir / ("2" * 32) / SELF_SETTINGS_TOKEN_FILENAME
            )
            self.assertEqual(first[SELF_SETTINGS_TOKEN_FILE_ENV], str(first_path))
            self.assertEqual(second[SELF_SETTINGS_TOKEN_FILE_ENV], str(second_path))
            self.assertEqual(first_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(second_path.stat().st_mode & 0o777, 0o600)
            first_token = read_self_settings_token(first_path)
            second_token = read_self_settings_token(second_path)
            self.assertNotEqual(first_token, second_token)
            self.assertNotIn(first_token, repr(first))
            self.assertNotIn(second_token, repr(second))

    def test_updates_only_implicit_calling_instance(self):
        with tempfile.TemporaryDirectory() as temporary:
            token_path = Path(temporary) / "agent-self-settings.token"
            token_path.write_text("x" * 64, encoding="ascii")
            token_path.chmod(0o600)
            captured = {}

            def urlopen(request, timeout):
                captured["request"] = request
                captured["timeout"] = timeout
                return _Response(
                    {
                        "ok": True,
                        "scope": "session",
                        "revision": 3,
                        "override_active": True,
                        "changed": True,
                        "apply_state": "live",
                    }
                )

            with (
                patch.dict(os.environ, self._environment(token_path), clear=True),
                patch("aeon.tools.set_job_role._open_local", urlopen),
            ):
                result = SetJobRoleTool().execute(
                    "Act as the release coordinator for this workspace."
                )

            self.assertIn("this session", result)
            self.assertIn("next model turn", result)
            self.assertEqual(captured["timeout"], 15)
            request = captured["request"]
            self.assertEqual(
                request.full_url,
                "http://127.0.0.1:8765/internal/agent/job-role",
            )
            self.assertEqual(request.get_method(), "PUT")
            self.assertEqual(
                request.get_header("Authorization"), f"Bearer {'x' * 64}"
            )
            self.assertEqual(
                request.get_header("X-nexus-agent-instance"), "a" * 32
            )
            self.assertEqual(
                json.loads(request.data),
                {
                    "job_role": "Act as the release coordinator for this workspace.",
                    "use_default": False,
                },
            )
            self.assertNotIn("instance", json.loads(request.data))

    def test_can_restore_default_without_supplying_role_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            token_path = Path(temporary) / "agent-self-settings.token"
            token_path.write_text("y" * 64, encoding="ascii")
            token_path.chmod(0o600)
            captured = {}

            def urlopen(request, timeout):
                captured["body"] = json.loads(request.data)
                return _Response(
                    {
                        "ok": True,
                        "scope": "session",
                        "revision": 4,
                        "override_active": False,
                        "changed": True,
                        "apply_state": "pending",
                    }
                )

            with (
                patch.dict(os.environ, self._environment(token_path), clear=True),
                patch("aeon.tools.set_job_role._open_local", urlopen),
            ):
                result = SetJobRoleTool().execute(use_default=True)

            self.assertEqual(
                captured["body"], {"job_role": "", "use_default": True}
            )
            self.assertIn("shared default", result)
            self.assertIn("saved", result)

    def test_blank_role_requires_explicit_default_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            token_path = Path(temporary) / "agent-self-settings.token"
            token_path.write_text("z" * 64, encoding="ascii")
            token_path.chmod(0o600)
            with patch.dict(
                os.environ, self._environment(token_path), clear=True
            ):
                result = SetJobRoleTool().execute("   ")
        self.assertIn("cannot be blank", result)

    def test_unsafe_capability_and_nonlocal_endpoint_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            token_path = Path(temporary) / "agent-self-settings.token"
            token_path.write_text("z" * 64, encoding="ascii")
            token_path.chmod(0o644)
            environment = self._environment(token_path)
            with patch.dict(os.environ, environment, clear=True):
                unsafe_file = SetJobRoleTool().execute("Release lead")
            self.assertIn("not owner-safe", unsafe_file)

            token_path.chmod(0o600)
            environment[SELF_SETTINGS_URL_ENV] = (
                "https://example.com/internal/agent/job-role"
            )
            with patch.dict(os.environ, environment, clear=True):
                unsafe_url = SetJobRoleTool().execute("Release lead")
            self.assertIn("not an approved local URL", unsafe_url)


if __name__ == "__main__":
    unittest.main()
