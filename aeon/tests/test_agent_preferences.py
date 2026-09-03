"""Hermetic persistence/API/launcher tests for per-tab agent preferences."""

from __future__ import annotations

import json
import os
import shlex
import sqlite3
import tempfile
import time
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from aeon.core.model_identity import AEON_DEFAULT_MODEL_NAME
from aeon.harnesses.catalog import (
    DEFAULT_HARNESS_ID,
    LEGACY_AEON_HARNESS_ID,
    OPENCODE_HARNESS_ID,
)
from aeon.remote.app import create_app
from aeon.remote.instances import InstanceError, InstanceLaunchError, InstanceManager
from aeon.remote.instruction_profiles import InstructionProfileService
from aeon.remote.security import AuthService, generate_totp_secret
from aeon.remote.store import RemoteStore
from aeon.tests.test_remote import FakeTmux, RemoteFixture


class AgentPreferenceFixture(RemoteFixture):
    def setUp(self):
        super().setUp()
        self.fake = FakeTmux()
        self.manager = InstanceManager(
            self.store,
            self.config,
            command_runner=self.fake,
            pane_prompt_checker=self.fake.pane_at_prompt,
            pane_foreground_checker=self.fake.pane_has_managed_foreground,
        )

    def terminal(self, name: str = "Settings terminal") -> dict:
        return self.manager.create_terminal(
            name=name,
            workspace=str(self.workspace),
            actor="admin",
        )


class TestAgentPreferenceStore(AgentPreferenceFixture):
    def test_harness_defaults_and_applied_snapshot_are_independent(self):
        terminal = self.terminal("Harness settings")
        initial = self.store.get_harness_setting(terminal["id"])
        self.assertEqual(initial["desired_harness"], DEFAULT_HARNESS_ID)
        self.assertIsNone(initial["applied_harness"])
        with self.store._connect() as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM instance_harness_settings "
                    "WHERE instance_id=?",
                    (terminal["id"],),
                ).fetchone()[0],
                0,
            )

        historical = self.store.mark_harness_setting_applied(
            terminal["id"], LEGACY_AEON_HARNESS_ID
        )
        self.assertEqual(historical["desired_harness"], DEFAULT_HARNESS_ID)
        self.assertEqual(
            historical["applied_harness"], LEGACY_AEON_HARNESS_ID
        )
        self.store.put_harness_setting(
            terminal["id"], LEGACY_AEON_HARNESS_ID
        )
        changed = self.store.put_harness_setting(
            terminal["id"], OPENCODE_HARNESS_ID
        )
        self.assertEqual(changed["desired_harness"], OPENCODE_HARNESS_ID)
        self.assertEqual(
            changed["applied_harness"], LEGACY_AEON_HARNESS_ID
        )

        with self.assertRaises(ValueError):
            self.store.put_harness_setting(terminal["id"], "arbitrary-harness")
        self.store.delete_instance(terminal["id"])
        with self.store._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM instance_harness_settings WHERE instance_id=?",
                (terminal["id"],),
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_desired_and_applied_are_independent_and_delete_cascades(self):
        terminal = self.terminal()
        initial = self.store.get_agent_setting(terminal["id"], "codex")
        self.assertEqual(initial["desired_model"], "")
        self.assertIsNone(initial["applied_model"])

        self.store.put_agent_setting(
            terminal["id"], "codex", model="gpt-5.6-terra", effort="high"
        )
        self.store.mark_agent_setting_applied(
            terminal["id"], "codex", model="gpt-5.6-terra", effort="high"
        )
        self.store.put_agent_setting(
            terminal["id"], "codex", model="gpt-5.6-sol", effort="xhigh"
        )
        # A launcher commits its immutable preparation snapshot and must not
        # overwrite a preference saved later.
        value = self.store.mark_agent_setting_applied(
            terminal["id"], "codex", model="gpt-5.6-terra", effort="high"
        )
        self.assertEqual(value["desired_model"], "gpt-5.6-sol")
        self.assertEqual(value["desired_effort"], "xhigh")
        self.assertEqual(value["applied_model"], "gpt-5.6-terra")
        self.assertEqual(value["applied_effort"], "high")

        payload = self.manager.get_agent_settings(terminal["id"])
        codex = payload["settings"]["codex"]
        self.assertTrue(codex["pending"])
        self.assertFalse(codex["current_process_verified"])
        self.assertEqual(codex["applied_scope"], "historical")
        self.assertEqual(codex["apply_mode"], "last_verified")
        self.store.delete_instance(terminal["id"])
        with self.store._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM instance_agent_settings WHERE instance_id=?",
                (terminal["id"],),
            ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_legacy_registry_migration_is_restart_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "legacy.sqlite3"
            now = time.time()
            with sqlite3.connect(database) as conn:
                conn.executescript(
                    """
                    CREATE TABLE instances (
                        id TEXT PRIMARY KEY,
                        host_id TEXT NOT NULL DEFAULT '192.168.0.177',
                        kind TEXT NOT NULL DEFAULT 'aeon',
                        shell_backed INTEGER NOT NULL DEFAULT 0,
                        last_agent_kind TEXT,
                        name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                        tmux_name TEXT NOT NULL UNIQUE,
                        workspace TEXT NOT NULL,
                        objective TEXT NOT NULL DEFAULT '',
                        max_iterations INTEGER,
                        model TEXT,
                        status TEXT NOT NULL,
                        desired_state TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        last_started_at REAL,
                        last_error TEXT NOT NULL DEFAULT '',
                        created_by TEXT NOT NULL,
                        launch_origin TEXT NOT NULL DEFAULT 'web'
                    );
                    """
                )
                conn.execute(
                    "INSERT INTO instances VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "legacy-terminal",
                        "192.168.0.177",
                        "terminal",
                        1,
                        None,
                        "Legacy terminal",
                        "terminal-legacy",
                        "/tmp",
                        "",
                        None,
                        None,
                        "running",
                        "running",
                        now,
                        now,
                        now,
                        "",
                        "fixture",
                        "web",
                    ),
                )

            migrated = RemoteStore(database)
            migrated.put_agent_setting(
                "legacy-terminal",
                "claude",
                model="sonnet",
                effort="high",
            )
            restarted = RemoteStore(database)
            value = restarted.get_agent_setting("legacy-terminal", "claude")
            self.assertEqual(value["desired_model"], "sonnet")
            self.assertEqual(value["desired_effort"], "high")
            self.assertIsNone(value["applied_model"])
            harness = restarted.get_harness_setting("legacy-terminal")
            self.assertEqual(harness["desired_harness"], OPENCODE_HARNESS_ID)
            self.assertIsNone(harness["applied_harness"])

    def test_all_missing_rows_have_reviewed_defaults_without_read_side_effects(self):
        terminal = self.terminal()
        payload = self.manager.get_agent_settings(terminal["id"])
        self.assertEqual(set(payload["settings"]), {"aeon", "codex", "claude", "grok"})
        self.assertEqual(
            payload["settings"]["aeon"]["desired"]["model"],
            AEON_DEFAULT_MODEL_NAME,
        )
        self.assertEqual(
            payload["settings"]["aeon"]["desired"]["harness"],
            OPENCODE_HARNESS_ID,
        )
        self.assertEqual(
            {item["id"] for item in payload["settings"]["aeon"]["catalog"]["harnesses"]},
            {OPENCODE_HARNESS_ID, LEGACY_AEON_HARNESS_ID},
        )
        self.assertFalse(payload["settings"]["aeon"]["model_editable"])
        self.assertTrue(payload["settings"]["aeon"]["pending"])
        self.assertFalse(payload["settings"]["aeon"]["current_process_verified"])
        self.assertEqual(payload["settings"]["aeon"]["applied_scope"], "none")
        self.assertEqual(payload["settings"]["aeon"]["apply_mode"], "never_applied")
        with self.store._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM instance_agent_settings WHERE instance_id=?",
                (terminal["id"],),
            ).fetchone()[0]
        self.assertEqual(count, 0)
        with self.store._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM instance_harness_settings WHERE instance_id=?",
                (terminal["id"],),
            ).fetchone()[0]
        self.assertEqual(count, 0)


class TestAgentPreferenceAPI(AgentPreferenceFixture):
    def setUp(self):
        super().setUp()
        hasher = PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1)
        self.auth = AuthService(self.store, self.config, password_hasher=hasher)
        self.password = "correct horse battery staple"
        self.store.put_user(
            "admin",
            self.auth.hash_password(self.password),
            generate_totp_secret(),
        )
        self.terminal_record = self.terminal("API settings")

    def test_authenticated_contract_and_strict_mutation(self):
        app = create_app(
            self.config,
            store=self.store,
            manager=self.manager,
            auth=self.auth,
        )
        route = f"/api/instances/{self.terminal_record['id']}/agent-settings"
        with TestClient(app) as client:
            self.assertEqual(client.get(route).status_code, 401)
            login = client.post(
                "/api/login",
                headers={"Origin": "http://testserver"},
                json={"username": "admin", "password": self.password},
            )
            self.assertEqual(login.status_code, 200)
            csrf = login.json()["csrf_token"]

            initial = client.get(route)
            self.assertEqual(initial.status_code, 200)
            body = initial.json()
            self.assertEqual(body["selected_kind"], "aeon")
            self.assertIn("catalog", body["settings"]["codex"])
            self.assertEqual(
                body["settings"]["aeon"]["desired"]["harness"],
                OPENCODE_HARNESS_ID,
            )
            self.assertIsNone(body["settings"]["aeon"]["applied"])
            self.assertIn("harnesses", body["settings"]["aeon"]["catalog"])
            self.assertNotIn("argv", json.dumps(body))

            aeon_update = client.put(
                route,
                headers={
                    "Origin": "http://testserver",
                    "X-CSRF-Token": csrf,
                },
                json={
                    "kind": "aeon",
                    "model": AEON_DEFAULT_MODEL_NAME,
                    "effort": "",
                    "harness": LEGACY_AEON_HARNESS_ID,
                },
            )
            self.assertEqual(aeon_update.status_code, 200, aeon_update.text)
            self.assertEqual(
                aeon_update.json()["settings"]["aeon"]["desired"]["harness"],
                LEGACY_AEON_HARNESS_ID,
            )

            omitted_harness = client.put(
                route,
                headers={
                    "Origin": "http://testserver",
                    "X-CSRF-Token": csrf,
                },
                json={
                    "kind": "aeon",
                    "model": AEON_DEFAULT_MODEL_NAME,
                    "effort": "",
                },
            )
            self.assertEqual(omitted_harness.status_code, 200)
            self.assertEqual(
                omitted_harness.json()["settings"]["aeon"]["desired"]["harness"],
                LEGACY_AEON_HARNESS_ID,
            )

            invalid_harness = client.put(
                route,
                headers={
                    "Origin": "http://testserver",
                    "X-CSRF-Token": csrf,
                },
                json={
                    "kind": "aeon",
                    "model": AEON_DEFAULT_MODEL_NAME,
                    "effort": "",
                    "harness": "unreviewed-harness",
                },
            )
            self.assertEqual(invalid_harness.status_code, 400)

            mutation = {
                "kind": "codex",
                "model": "gpt-5.6-terra",
                "effort": "xhigh",
            }
            missing_csrf = client.put(
                route,
                headers={"Origin": "http://testserver"},
                json=mutation,
            )
            self.assertEqual(missing_csrf.status_code, 403)
            updated = client.put(
                route,
                headers={
                    "Origin": "http://testserver",
                    "X-CSRF-Token": csrf,
                },
                json=mutation,
            )
            self.assertEqual(updated.status_code, 200)
            codex = updated.json()["settings"]["codex"]
            self.assertEqual(codex["desired"]["model"], "gpt-5.6-terra")
            self.assertEqual(codex["desired"]["effort"], "xhigh")
            self.assertIsNone(codex["applied"])
            self.assertTrue(codex["pending"])

            rejected = client.put(
                route,
                headers={
                    "Origin": "http://testserver",
                    "X-CSRF-Token": csrf,
                },
                json={
                    "kind": "codex",
                    "model": "$(arbitrary-command)",
                    "effort": "high",
                },
            )
            self.assertEqual(rejected.status_code, 400)

            wrong_kind_harness = client.put(
                route,
                headers={
                    "Origin": "http://testserver",
                    "X-CSRF-Token": csrf,
                },
                json={
                    "kind": "codex",
                    "model": "gpt-5.6-terra",
                    "effort": "high",
                    "harness": LEGACY_AEON_HARNESS_ID,
                },
            )
            self.assertEqual(wrong_kind_harness.status_code, 400)

        audit = self.store.recent_audit(1)[0]
        self.assertEqual(audit["action"], "agent_settings_updated")
        self.assertEqual(
            set(json.loads(audit["details_json"])),
            {"kind", "changed", "apply_mode"},
        )
        self.assertNotIn("gpt-5.6", audit["details_json"])

    def test_continuous_mode_api_is_authenticated_csrf_guarded_and_validated(self):
        app = create_app(
            self.config,
            store=self.store,
            manager=self.manager,
            auth=self.auth,
        )
        route = f"/api/instances/{self.terminal_record['id']}/continuous-mode"
        with TestClient(app) as client:
            self.assertEqual(client.get(route).status_code, 401)
            login = client.post(
                "/api/login",
                headers={"Origin": "http://testserver"},
                json={"username": "admin", "password": self.password},
            )
            csrf = login.json()["csrf_token"]
            initial = client.get(route)
            self.assertEqual(initial.status_code, 200)
            self.assertFalse(initial.json()["enabled"])

            missing_csrf = client.put(
                route,
                headers={"Origin": "http://testserver"},
                json={"enabled": True, "goal": "keep improving this project"},
            )
            self.assertEqual(missing_csrf.status_code, 403)
            rejected = client.put(
                route,
                headers={
                    "Origin": "http://testserver",
                    "X-CSRF-Token": csrf,
                },
                json={"enabled": True, "goal": "two words"},
            )
            self.assertEqual(rejected.status_code, 400)
            updated = client.put(
                route,
                headers={
                    "Origin": "http://testserver",
                    "X-CSRF-Token": csrf,
                },
                json={"enabled": True, "goal": "keep improving this project"},
            )
            self.assertEqual(updated.status_code, 200, updated.text)
            self.assertTrue(updated.json()["enabled"])
            self.assertEqual(updated.json()["goal"], "keep improving this project")

        audit = self.store.recent_audit(1)[0]
        self.assertEqual(audit["action"], "continuous_mode_updated")
        details = json.loads(audit["details_json"])
        self.assertEqual(
            set(details),
            {
                "enabled",
                "goal_present",
                "live_wake",
                "turn_stop_acknowledged",
                "idle_restart",
            },
        )
        self.assertNotIn("keep improving", audit["details_json"])


class TestAgentPreferenceLaunch(AgentPreferenceFixture):
    def _provider_context(self, kind: str, argv: tuple[str, ...], environment: dict):
        return (
            patch(
                "aeon.remote.instances.provider_status",
                return_value={"installed": True, "connected": True},
            ),
            patch(
                "aeon.remote.instances.provider_agent_command",
                return_value=SimpleNamespace(argv=argv),
            ),
            patch(
                "aeon.remote.instances.subscription_environment",
                return_value=environment,
            ),
        )

    def test_codex_prompt_is_private_while_settings_use_cli_precedence(self):
        instruction_service = InstructionProfileService(
            self.store,
            project_root=self.config.project_root,
            allowed_roots=self.config.allowed_roots,
        )
        manager = InstanceManager(
            self.store,
            self.config,
            command_runner=self.fake,
            instruction_service=instruction_service,
            pane_prompt_checker=self.fake.pane_at_prompt,
            pane_foreground_checker=self.fake.pane_has_managed_foreground,
        )
        terminal = manager.create_terminal(
            name="Codex settings",
            workspace=str(self.workspace),
            actor="admin",
        )
        prompt_sentinel = "CONFIDENTIAL-PERSISTENT-ROLE-🧭-MUST-STAY-OFF-ARGV"
        instruction_service.save_local_role(
            terminal["id"],
            content=prompt_sentinel,
            expected_revision=0,
            actor="admin",
        )
        project_config = self.workspace / ".codex"
        project_config.mkdir()
        (project_config / "config.toml").write_text(
            'model = "project-model-must-not-win"\n'
            'model_reasoning_effort = "low"\n',
            encoding="utf-8",
        )
        codex_home = Path(self.temp.name) / "codex-home"
        codex_home.mkdir(mode=0o700)
        manager.update_agent_settings(
            terminal["id"],
            kind="codex",
            model="gpt-5.6-terra",
            effort="xhigh",
            actor="admin",
        )
        contexts = self._provider_context(
            "codex",
            ("/safe/codex", "--no-alt-screen"),
            {
                "CODEX_HOME": str(codex_home),
                "HOME": str(Path(self.temp.name)),
                "PATH": "/safe/bin",
            },
        )
        with contexts[0], contexts[1], contexts[2]:
            agent = manager.activate_agent(
                terminal["id"], kind="codex", actor="admin"
            )

        command = self.fake.loaded_payloads[-1]
        self.assertIn("--model gpt-5.6-terra", command)
        self.assertIn("model_reasoning_effort=", command)
        self.assertIn("xhigh", command)
        self.assertNotIn(prompt_sentinel, command)
        self.assertNotIn("project-model-must-not-win", command)
        profile_name = f"nexus-{terminal['id']}"
        self.assertIn(f"--profile {profile_name}", command)
        profile = codex_home / f"{profile_name}.config.toml"
        self.assertEqual(profile.stat().st_mode & 0o777, 0o600)
        body = profile.read_text(encoding="utf-8")
        self.assertIn(prompt_sentinel, body)
        self.assertIn(prompt_sentinel, tomllib.loads(body)["developer_instructions"])
        self.assertNotIn('model = "gpt-5.6-terra"', body)
        self.assertNotIn("model_reasoning_effort", body)
        self.assertEqual(agent["applied_agent_model"], "gpt-5.6-terra")
        self.assertEqual(agent["applied_reasoning_effort"], "xhigh")
        self.assertFalse(agent["agent_settings_pending"])

    def test_codex_conflicting_or_ambiguous_project_instructions_fail_closed(self):
        cases = ("defined", "malformed", "symbolic", "oversized", "fifo")
        for case in cases:
            with self.subTest(case=case):
                instruction_service = InstructionProfileService(
                    self.store,
                    project_root=self.config.project_root,
                    allowed_roots=self.config.allowed_roots,
                )
                manager = InstanceManager(
                    self.store,
                    self.config,
                    command_runner=self.fake,
                    instruction_service=instruction_service,
                    pane_prompt_checker=self.fake.pane_at_prompt,
                    pane_foreground_checker=self.fake.pane_has_managed_foreground,
                )
                terminal = manager.create_terminal(
                    name=f"Codex conflict {case}",
                    workspace=str(self.workspace),
                    actor="admin",
                )
                prompt_sentinel = f"PRIVATE-{case}-PROMPT-SENTINEL"
                instruction_service.save_local_role(
                    terminal["id"],
                    content=prompt_sentinel,
                    expected_revision=0,
                    actor="admin",
                )
                ancestor_config = self.root / ".codex"
                ancestor_config.mkdir(exist_ok=True)
                config = ancestor_config / "config.toml"
                target = Path(self.temp.name) / f"outside-{case}.toml"
                if case == "defined":
                    config.write_text(
                        'developer_instructions = "project override"\n',
                        encoding="utf-8",
                    )
                elif case == "malformed":
                    config.write_text('developer_instructions = "unterminated\n')
                elif case == "symbolic":
                    target.write_text('developer_instructions = "linked"\n')
                    config.symlink_to(target)
                elif case == "oversized":
                    config.write_bytes(b"#" * (256 * 1024 + 1))
                else:
                    config.unlink(missing_ok=True)
                    os.mkfifo(config)
                codex_home = Path(self.temp.name) / f"codex-home-{case}"
                codex_home.mkdir(mode=0o700)
                contexts = self._provider_context(
                    "codex",
                    ("/safe/codex", "--no-alt-screen"),
                    {
                        "CODEX_HOME": str(codex_home),
                        "HOME": str(Path(self.temp.name)),
                        "PATH": "/safe/bin",
                    },
                )
                payload_count = len(self.fake.loaded_payloads)
                with contexts[0], contexts[1], contexts[2]:
                    with self.assertRaisesRegex(
                        InstanceError,
                        "developer_instructions|parsed safely|ambiguous|too large|regular file",
                    ):
                        manager.activate_agent(
                            terminal["id"], kind="codex", actor="admin"
                        )
                self.assertEqual(len(self.fake.loaded_payloads), payload_count)
                self.assertEqual(
                    self.store.get_instance(terminal["id"])["kind"], "terminal"
                )
                audit = "\n".join(
                    row["details_json"] for row in self.store.recent_audit(100)
                )
                self.assertNotIn(prompt_sentinel, audit)
                if config.is_symlink():
                    config.unlink()
                elif config.exists():
                    config.unlink()

    def test_legacy_direct_codex_resume_uses_overrides_but_stays_historical(self):
        instruction_service = InstructionProfileService(
            self.store,
            project_root=self.config.project_root,
            allowed_roots=self.config.allowed_roots,
        )
        manager = InstanceManager(
            self.store,
            self.config,
            command_runner=self.fake,
            instruction_service=instruction_service,
            pane_prompt_checker=self.fake.pane_at_prompt,
            pane_foreground_checker=self.fake.pane_has_managed_foreground,
        )
        codex_home = Path(self.temp.name) / "legacy-codex-home"
        codex_home.mkdir(mode=0o700)
        environment = {
            "CODEX_HOME": str(codex_home),
            "HOME": str(Path(self.temp.name)),
            "PATH": "/safe/bin",
        }
        contexts = self._provider_context(
            "codex", ("/safe/codex", "--no-alt-screen"), environment
        )
        with contexts[0], contexts[1], contexts[2]:
            direct = manager.create_instance(
                kind="codex",
                name="Legacy direct settings",
                workspace=str(self.workspace),
                objective="",
                max_iterations=None,
                actor="admin",
            )
        manager.force_stop(
            direct["id"], confirmation=direct["name"], actor="admin"
        )
        prompt_sentinel = "PRIVATE-LEGACY-RESUME-PROMPT"
        instruction_service.save_local_role(
            direct["id"],
            content=prompt_sentinel,
            expected_revision=0,
            actor="admin",
        )
        manager.update_agent_settings(
            direct["id"],
            kind="codex",
            model="gpt-5.6-sol",
            effort="high",
            actor="admin",
        )
        project_config = self.workspace / ".codex"
        project_config.mkdir()
        (project_config / "config.toml").write_text(
            'model = "project-model-must-not-win"\n'
            'model_reasoning_effort = "minimal"\n',
            encoding="utf-8",
        )
        contexts = self._provider_context(
            "codex", ("/safe/codex", "--no-alt-screen"), environment
        )
        with (
            contexts[0],
            contexts[1],
            contexts[2],
            patch(
                "aeon.remote.instances.provider_status",
                return_value={"installed": True, "connected": True},
            ),
        ):
            manager.resume_instance(direct["id"], actor="admin")

        launch = [call for call in self.fake.calls if call[1] == "new-session"][-1]
        rendered = "\x00".join(launch)
        self.assertIn("--model\x00gpt-5.6-sol", rendered)
        self.assertIn('model_reasoning_effort="high"', rendered)
        self.assertNotIn(prompt_sentinel, rendered)
        self.assertNotIn("project-model-must-not-win", rendered)
        setting = manager.get_agent_settings(direct["id"])["settings"]["codex"]
        self.assertFalse(setting["current_process_verified"])
        self.assertEqual(setting["applied_scope"], "historical")
        self.assertEqual(setting["apply_mode"], "last_verified")
        self.assertTrue(setting["pending"])

    def test_claude_and_grok_receive_only_allowlisted_fixed_options(self):
        cases = (
            ("claude", "sonnet", "max", ("--model", "sonnet", "--effort", "max")),
            ("grok", "grok-4.5", "", ("--model", "grok-4.5")),
        )
        for kind, model, effort, expected in cases:
            with self.subTest(kind=kind):
                terminal = self.terminal(f"{kind} settings")
                self.manager.update_agent_settings(
                    terminal["id"],
                    kind=kind,
                    model=model,
                    effort=effort,
                    actor="admin",
                )
                contexts = self._provider_context(
                    kind,
                    (f"/safe/{kind}",),
                    {"HOME": str(Path(self.temp.name)), "PATH": "/safe/bin"},
                )
                with contexts[0], contexts[1], contexts[2]:
                    agent = self.manager.activate_agent(
                        terminal["id"], kind=kind, actor="admin"
                    )
                command = self.fake.loaded_payloads[-1]
                rendered = " ".join(expected)
                self.assertIn(rendered, command)
                self.assertEqual(agent["applied_agent_model"], model)
                self.assertEqual(agent["applied_reasoning_effort"], effort)

    def test_aeon_release_is_fixed_and_applied_after_verified_start(self):
        terminal = self.terminal("Aeon settings")
        with self.assertRaisesRegex(InstanceError, "not available|fixed"):
            self.manager.update_agent_settings(
                terminal["id"],
                kind="aeon",
                model="unreviewed-local-model",
                effort="",
                actor="admin",
            )
        agent = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )
        self.assertIn(
            f"--model {shlex.quote(AEON_DEFAULT_MODEL_NAME)}",
            self.fake.loaded_payloads[-1],
        )
        self.assertIn(
            "aeon.harnesses.opencode_runtime", self.fake.loaded_payloads[-1]
        )
        self.assertEqual(agent["applied_agent_model"], AEON_DEFAULT_MODEL_NAME)
        self.assertEqual(agent["agent_harness"], OPENCODE_HARNESS_ID)
        self.assertEqual(agent["applied_agent_harness"], OPENCODE_HARNESS_ID)
        self.assertFalse(agent["agent_settings_pending"])

    def test_legacy_harness_preference_controls_launch_and_applied_truth(self):
        terminal = self.terminal("Legacy Aeon harness")
        self.manager.update_agent_settings(
            terminal["id"],
            kind="aeon",
            model=AEON_DEFAULT_MODEL_NAME,
            effort="",
            harness=LEGACY_AEON_HARNESS_ID,
            actor="admin",
        )
        agent = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )
        command = self.fake.loaded_payloads[-1]
        self.assertIn("aeon.main", command)
        self.assertNotIn("aeon.harnesses.opencode_runtime", command)
        setting = self.manager.get_agent_settings(terminal["id"])["settings"][
            "aeon"
        ]
        self.assertEqual(
            setting["desired"]["harness"], LEGACY_AEON_HARNESS_ID
        )
        self.assertEqual(
            setting["applied"]["harness"], LEGACY_AEON_HARNESS_ID
        )
        self.assertTrue(setting["current_process_verified"])
        self.assertFalse(setting["pending"])
        self.assertEqual(agent["applied_agent_harness"], LEGACY_AEON_HARNESS_ID)

    def test_failed_start_preserves_previous_applied_and_instruction_binding(self):
        instruction_service = InstructionProfileService(
            self.store,
            project_root=self.config.project_root,
            allowed_roots=self.config.allowed_roots,
        )
        manager = InstanceManager(
            self.store,
            self.config,
            command_runner=self.fake,
            instruction_service=instruction_service,
            pane_prompt_checker=self.fake.pane_at_prompt,
            pane_foreground_checker=self.fake.pane_has_managed_foreground,
        )
        terminal = manager.create_terminal(
            name="Codex failure",
            workspace=str(self.workspace),
            actor="admin",
        )
        codex_home = Path(self.temp.name) / "failure-codex-home"
        codex_home.mkdir(mode=0o700)
        environment = {
            "CODEX_HOME": str(codex_home),
            "HOME": str(Path(self.temp.name)),
            "PATH": "/safe/bin",
        }
        manager.update_agent_settings(
            terminal["id"],
            kind="codex",
            model="gpt-5.6-terra",
            effort="high",
            actor="admin",
        )
        contexts = self._provider_context("codex", ("/safe/codex",), environment)
        with contexts[0], contexts[1], contexts[2]:
            manager.activate_agent(terminal["id"], kind="codex", actor="admin")
        profile = instruction_service.create_profile(
            agent_kind="codex", name="Persistent role", actor="admin"
        )
        version = instruction_service.save_version(
            profile["id"],
            label="v1",
            content="Private role sentinel that must not enter audit or argv.",
            actor="admin",
        )
        instruction_service.select_profile_version(terminal["id"], version["id"])
        manager.end_agent(terminal["id"], actor="admin")
        manager.update_agent_settings(
            terminal["id"],
            kind="codex",
            model="gpt-5.6-sol",
            effort="low",
            actor="admin",
        )
        before_binding = instruction_service.get_instance_binding(terminal["id"])

        failed_contexts = self._provider_context(
            "codex", ("/safe/provider-that-exits",), environment
        )
        with failed_contexts[0], failed_contexts[1], failed_contexts[2]:
            with self.assertRaises(InstanceLaunchError) as raised:
                manager.activate_agent(terminal["id"], kind="codex", actor="admin")
        self.assertFalse(raised.exception.launched)
        setting = self.store.get_agent_setting(terminal["id"], "codex")
        self.assertEqual(setting["desired_model"], "gpt-5.6-sol")
        self.assertEqual(setting["applied_model"], "gpt-5.6-terra")
        self.assertEqual(setting["applied_effort"], "high")
        self.assertEqual(
            instruction_service.get_instance_binding(terminal["id"]),
            before_binding,
        )
        audit = "\n".join(
            row["details_json"] for row in self.store.recent_audit(100)
        )
        self.assertNotIn("Private role sentinel", audit)

    def test_crash_after_mode_commit_recovers_exact_applied_snapshot(self):
        terminal = self.terminal("Crash recovery")
        codex_home = Path(self.temp.name) / "recovery-codex-home"
        codex_home.mkdir(mode=0o700)
        self.manager.update_agent_settings(
            terminal["id"],
            kind="codex",
            model="gpt-5.6-luna",
            effort="medium",
            actor="admin",
        )
        contexts = self._provider_context(
            "codex",
            ("/safe/codex",),
            {
                "CODEX_HOME": str(codex_home),
                "HOME": str(Path(self.temp.name)),
                "PATH": "/safe/bin",
            },
        )
        with (
            contexts[0],
            contexts[1],
            contexts[2],
            patch.object(
                self.store,
                "mark_agent_setting_applied",
                side_effect=RuntimeError("simulated registry interruption"),
            ),
        ):
            with self.assertRaises(InstanceLaunchError) as raised:
                self.manager.activate_agent(
                    terminal["id"], kind="codex", actor="admin"
                )
        self.assertTrue(raised.exception.launched)
        pending = (
            self.config.instance_state_dir
            / terminal["id"]
            / "managed-agent.pending.json"
        )
        self.assertTrue(pending.is_file())
        recovered = self.manager.get_instance(terminal["id"])
        self.assertEqual(recovered["applied_agent_model"], "gpt-5.6-luna")
        self.assertEqual(recovered["applied_reasoning_effort"], "medium")
        self.assertFalse(recovered["agent_settings_pending"])
        self.assertEqual(recovered["last_error"], "")
        self.assertFalse(pending.exists())
        setting = self.manager.get_agent_settings(terminal["id"])["settings"]["codex"]
        self.assertTrue(setting["current_process_verified"])
        self.assertEqual(setting["applied_scope"], "current_process")
        self.assertEqual(setting["apply_mode"], "current_start")

    def test_aeon_harness_receipt_recovers_after_registry_interruption(self):
        terminal = self.terminal("Harness receipt recovery")
        with patch.object(
            self.store,
            "mark_harness_setting_applied",
            side_effect=RuntimeError("simulated harness registry interruption"),
        ):
            with self.assertRaises(InstanceLaunchError) as raised:
                self.manager.activate_agent(
                    terminal["id"], kind="aeon", actor="admin"
                )
        self.assertTrue(raised.exception.launched)
        pending = (
            self.config.instance_state_dir
            / terminal["id"]
            / "managed-agent.pending.json"
        )
        self.assertTrue(pending.is_file())

        recovered = self.manager.get_instance(terminal["id"])
        self.assertEqual(recovered["applied_agent_harness"], OPENCODE_HARNESS_ID)
        self.assertFalse(recovered["agent_settings_pending"])
        self.assertEqual(recovered["last_error"], "")
        self.assertFalse(pending.exists())
        setting = self.manager.get_agent_settings(terminal["id"])["settings"]["aeon"]
        self.assertEqual(setting["applied"]["harness"], OPENCODE_HARNESS_ID)
        self.assertTrue(setting["current_process_verified"])

    def test_stored_applied_snapshot_becomes_historical_outside_verified_process(self):
        terminal = self.terminal("Historical settings")
        active = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )
        current = self.manager.get_agent_settings(active["id"])["settings"]["aeon"]
        self.assertFalse(current["pending"])
        self.assertTrue(current["current_process_verified"])
        self.assertEqual(current["applied_scope"], "current_process")
        self.assertEqual(current["apply_mode"], "current_start")

        self.manager.end_agent(active["id"], actor="admin")
        historical = self.manager.get_agent_settings(active["id"])["settings"]["aeon"]
        self.assertTrue(historical["pending"])
        self.assertFalse(historical["current_process_verified"])
        self.assertEqual(historical["applied_scope"], "historical")
        self.assertEqual(historical["apply_mode"], "last_verified")
        self.assertIsNotNone(historical["applied"])

    def test_running_preference_without_harness_receipt_stays_historical(self):
        terminal = self.terminal("Pre-harness running Aeon")
        active = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )
        with self.store._connect() as conn:
            conn.execute(
                "DELETE FROM instance_harness_settings WHERE instance_id=?",
                (active["id"],),
            )

        setting = self.manager.get_agent_settings(active["id"])["settings"]["aeon"]
        self.assertEqual(setting["desired"]["harness"], OPENCODE_HARNESS_ID)
        self.assertEqual(setting["applied"]["harness"], None)
        self.assertFalse(setting["current_process_verified"])
        self.assertEqual(setting["applied_scope"], "historical")
        self.assertEqual(setting["apply_mode"], "last_verified")
        self.assertTrue(setting["pending"])

    def test_live_agent_reports_next_start_after_desired_setting_changes(self):
        terminal = self.terminal("Pending next start")
        self.manager.update_agent_settings(
            terminal["id"],
            kind="codex",
            model="gpt-5.6-terra",
            effort="high",
            actor="admin",
        )
        contexts = self._provider_context(
            "codex",
            ("/safe/codex",),
            {"HOME": str(Path(self.temp.name)), "PATH": "/safe/bin"},
        )
        with contexts[0], contexts[1], contexts[2]:
            self.manager.activate_agent(
                terminal["id"], kind="codex", actor="admin"
            )

        payload = self.manager.update_agent_settings(
            terminal["id"],
            kind="codex",
            model="gpt-5.6-sol",
            effort="xhigh",
            actor="admin",
        )
        setting = payload["settings"]["codex"]
        self.assertTrue(setting["current_process_verified"])
        self.assertEqual(setting["applied_scope"], "current_process")
        self.assertEqual(setting["apply_mode"], "next_start")
        self.assertTrue(setting["pending"])
        self.assertEqual(setting["applied"]["model"], "gpt-5.6-terra")
        self.assertEqual(setting["desired"]["model"], "gpt-5.6-sol")

    def test_missing_or_corrupt_settings_fail_closed_for_a_live_agent(self):
        terminal = self.terminal("Uncertain current settings")
        active = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )
        with self.store._connect() as conn:
            conn.execute(
                "DELETE FROM instance_agent_settings "
                "WHERE instance_id=? AND agent_kind='aeon'",
                (active["id"],),
            )
        unknown = self.manager.get_agent_settings(active["id"])["settings"]["aeon"]
        self.assertTrue(unknown["current_process_verified"])
        self.assertEqual(unknown["applied_scope"], "none")
        self.assertEqual(unknown["apply_mode"], "unknown_current")
        self.assertTrue(unknown["pending"])

        self.store.mark_agent_setting_applied(
            active["id"],
            "aeon",
            model=AEON_DEFAULT_MODEL_NAME,
            effort="",
        )
        with self.store._connect() as conn:
            conn.execute(
                "UPDATE instance_agent_settings SET desired_model='unreviewed-model' "
                "WHERE instance_id=? AND agent_kind='aeon'",
                (active["id"],),
            )
        public = self.manager.get_instance(active["id"])
        self.assertIsNone(public["agent_runtime_settings"])
        self.assertTrue(public["agent_settings_pending"])
        with self.assertRaisesRegex(InstanceError, "unavailable"):
            self.manager.get_agent_settings(active["id"])

    def test_unrecordable_new_launch_never_relabels_old_snapshot_current(self):
        terminal = self.terminal("Unrecordable settings")
        active = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )
        self.manager.end_agent(active["id"], actor="admin")
        self.manager._pane_foreground_checker = lambda _record, _pane: False

        with patch("aeon.remote.instances.AGENT_START_TIMEOUT_SECONDS", 0.03):
            with self.assertRaises(InstanceLaunchError) as raised:
                self.manager.activate_agent(
                    terminal["id"], kind="aeon", actor="admin"
                )
        self.assertTrue(raised.exception.launched)
        public = self.manager.get_instance(terminal["id"])
        self.assertEqual(public["status"], "error")
        self.assertTrue(public["force_stop_required"])
        self.assertTrue(public["agent_settings_pending"])
        setting = self.manager.get_agent_settings(terminal["id"])["settings"]["aeon"]
        self.assertFalse(setting["current_process_verified"])
        self.assertEqual(setting["applied_scope"], "historical")
        self.assertEqual(setting["apply_mode"], "last_verified")
        self.assertTrue(setting["pending"])

    def test_persistently_unrecordable_applied_snapshot_stays_historical(self):
        terminal = self.terminal("Unrecordable applied snapshot")
        active = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )
        self.manager.end_agent(active["id"], actor="admin")

        with patch.object(
            self.store,
            "mark_agent_setting_applied",
            side_effect=RuntimeError("persistent registry interruption"),
        ):
            with self.assertRaises(InstanceLaunchError) as raised:
                self.manager.activate_agent(
                    terminal["id"], kind="aeon", actor="admin"
                )
            self.assertTrue(raised.exception.launched)
            public = self.manager.get_instance(terminal["id"])
            setting = self.manager.get_agent_settings(terminal["id"])["settings"][
                "aeon"
            ]

        self.assertTrue(public["last_error"])
        self.assertTrue(public["agent_settings_pending"])
        self.assertFalse(setting["current_process_verified"])
        self.assertEqual(setting["applied_scope"], "historical")
        self.assertEqual(setting["apply_mode"], "last_verified")
        self.assertTrue(setting["pending"])

    def test_exited_agent_never_reports_a_current_start(self):
        terminal = self.terminal("Exited settings")
        active = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )
        record = self.store.get_instance(active["id"])
        self.fake.sessions[record["tmux_name"]].update(dead=True, exit_code=9)

        public = self.manager.get_instance(active["id"])
        self.assertEqual(public["status"], "exited")
        self.assertTrue(public["agent_settings_pending"])
        setting = self.manager.get_agent_settings(active["id"])["settings"]["aeon"]
        self.assertFalse(setting["current_process_verified"])
        self.assertEqual(setting["applied_scope"], "historical")
        self.assertEqual(setting["apply_mode"], "last_verified")
        self.assertTrue(setting["pending"])


if __name__ == "__main__":
    unittest.main()
