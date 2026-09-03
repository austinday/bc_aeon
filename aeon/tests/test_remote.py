"""Hermetic tests for Aeon Remote.

No test starts Aeon, touches a real tmux session, or invokes the GPU coordinator.
"""

from __future__ import annotations

import asyncio
import fcntl
import io
import json
import os
import pty
import signal
import sqlite3
import stat
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from argon2 import PasswordHasher
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from aeon.remote.app import _websocket_session, create_app
from aeon.remote.cli import init_admin, status as remote_status
from aeon.remote.config import RemoteConfig
from aeon.remote.controller_lock import ControllerLock, ControllerLockError
from aeon.remote.instances import InstanceError, InstanceLaunchError, InstanceManager
from aeon.remote.instruction_profiles import InstructionProfileService
from aeon.remote.security import (
    AuthService,
    AuthenticationError,
    LoginRateLimited,
    generate_totp_secret,
    token_digest,
    totp_code,
    verify_totp,
)
from aeon.remote.store import RemoteStore
from aeon.remote.terminal import (
    _enqueue_terminal_output,
    _forward_browser_input,
    _forward_terminal_input,
    _forward_terminal_scroll,
    _is_terminal_response,
    _normalize_terminal_snapshot,
    _resize_attached_client,
    bridge_terminal,
)
import aeon.core.chat_transcript as chat_transcript
from aeon.core.chat_transcript import (
    ChatTranscriptError,
    append_chat_message,
    build_chat_delivery_envelope,
)
from aeon.tools.command_fleet_guard import FleetCommandGuardError
from aeon.core.skills.knowledge import SkillKnowledgeStore
from aeon.core.skills.manager import SkillsManager


_FAKE_TMUX_PANE_FIELD_SEPARATOR = "__AEON_REMOTE_PANE_FIELD_6B1E__"


class FakeTmux:
    def __init__(self):
        self.sessions = {}
        self.calls = []
        self.buffers = {}
        self.loaded_payloads = []
        self.list_panes_error = False
        self.malformed_pane_output = False
        self.list_sessions_error = False
        self.kill_session_error = False
        self.query_error_after_kill = False
        self.load_buffer_error = False
        self.paste_buffer_error = False
        self.paste_deliver_then_error = False
        self.delete_buffer_error = False
        self.list_buffers_error = False

    @staticmethod
    def _execute_pending(item):
        pending = item.get("pending", "")
        separators = [index for index in (pending.find("\r"), pending.find("\n")) if index >= 0]
        if not separators:
            return
        end = min(separators)
        command = pending[:end]
        item["pending"] = pending[end + 1 :]
        if (
            "aeon.main" in command
            or "aeon.harnesses.opencode_runtime" in command
        ):
            item["command"] = "python3"
            item["agent_mode"] = True
            item["managed_agent"] = True
            item["at_prompt"] = False
        elif any(name in command for name in ("/codex", "/claude", "/grok")):
            item["command"] = "node"
            item["agent_mode"] = True
            item["managed_agent"] = True
            item["at_prompt"] = False

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        command = args[1]
        if command == "new-session":
            name = args[args.index("-s") + 1]
            current_command = "bash" if "/bin/bash" in args else "python3"
            self.sessions[name] = {
                "dead": False,
                "pid": 987654,
                "exit": 0,
                "cwd": args[args.index("-c") + 1],
                "command": current_command,
                "pending": "",
                "agent_mode": False,
                "managed_agent": False,
                "at_prompt": current_command == "bash",
                "interrupt_returns_prompt": True,
                "delayed_browser_input": "",
            }
            return subprocess.CompletedProcess(args, 0, "", "")
        if command == "list-panes":
            if self.list_panes_error:
                return subprocess.CompletedProcess(args, 2, "", "query failed")
            if self.malformed_pane_output:
                return subprocess.CompletedProcess(args, 0, "malformed\n", "")
            target = args[args.index("-t") + 1].removeprefix("=").removesuffix(":")
            item = self.sessions.get(target)
            if not item:
                return subprocess.CompletedProcess(args, 1, "", "missing")
            output = (
                f"{int(item['dead'])}{_FAKE_TMUX_PANE_FIELD_SEPARATOR}{item['pid']}"
                f"{_FAKE_TMUX_PANE_FIELD_SEPARATOR}{item['exit']}"
                f"{_FAKE_TMUX_PANE_FIELD_SEPARATOR}{item['command']}\n"
            )
            return subprocess.CompletedProcess(args, 0, output, "")
        if command == "list-sessions":
            if self.list_sessions_error:
                return subprocess.CompletedProcess(args, 2, "", "query failed")
            output = "".join(f"{name}\n" for name in self.sessions)
            return subprocess.CompletedProcess(args, 0, output, "")
        if command == "display-message":
            target = args[args.index("-t") + 1].removeprefix("=").removesuffix(":")
            item = self.sessions.get(target)
            if not item or item["dead"]:
                return subprocess.CompletedProcess(args, 1, "", "missing")
            return subprocess.CompletedProcess(args, 0, f"{item['cwd']}\n", "")
        if command == "kill-session":
            target = args[args.index("-t") + 1].removeprefix("=")
            if self.query_error_after_kill:
                self.list_panes_error = True
            if self.kill_session_error:
                return subprocess.CompletedProcess(args, 1, "", "kill failed")
            self.sessions.pop(target, None)
            return subprocess.CompletedProcess(args, 0, "", "")
        if command == "detach-client":
            target = args[args.index("-s") + 1].removeprefix("=")
            item = self.sessions.get(target)
            if item:
                # Model unread bytes in an attach-client PTY: detaching that
                # client discards them before server-built input is sent.
                item["delayed_browser_input"] = ""
            return subprocess.CompletedProcess(args, 0, "", "")
        if command == "capture-pane":
            return subprocess.CompletedProcess(args, 0, "saved output\n", "")
        if command == "load-buffer":
            if self.load_buffer_error:
                return subprocess.CompletedProcess(args, 1, "", "load failed")
            separator = args.index(";") if ";" in args else None
            load_args = args if separator is None else args[:separator]
            name = load_args[load_args.index("-b") + 1]
            payload = kwargs.get("input", "")
            self.buffers[name] = payload
            self.loaded_payloads.append(payload)
            if separator is not None:
                queued = [args[0], *args[separator + 1 :]]
                if len(queued) < 2 or queued[1] != "paste-buffer":
                    return subprocess.CompletedProcess(
                        args, 1, "", "malformed command sequence"
                    )
                pasted = self._paste_buffer(queued)
                return subprocess.CompletedProcess(
                    args, pasted.returncode, pasted.stdout, pasted.stderr
                )
            return subprocess.CompletedProcess(args, 0, "", "")
        if command == "list-buffers":
            if self.list_buffers_error:
                return subprocess.CompletedProcess(args, 2, "", "query failed")
            output = "".join(f"{name}\n" for name in self.buffers)
            return subprocess.CompletedProcess(args, 0, output, "")
        if command == "delete-buffer":
            if self.delete_buffer_error:
                return subprocess.CompletedProcess(args, 1, "", "delete failed")
            name = args[args.index("-b") + 1]
            self.buffers.pop(name, None)
            return subprocess.CompletedProcess(args, 0, "", "")
        if command == "paste-buffer":
            return self._paste_buffer(args)
        if command == "send-keys":
            target = args[args.index("-t") + 1].removeprefix("=").removesuffix(":")
            item = self.sessions.get(target)
            if not item:
                return subprocess.CompletedProcess(args, 1, "", "missing")
            if "-l" in args:
                item["pending"] = (
                    item.get("delayed_browser_input", "")
                    + args[args.index("-l") + 1]
                )
            elif args[-1] == "Enter" and item.get("pending"):
                item["pending"] += "\r"
                self._execute_pending(item)
            elif (
                args[-1] == "C-c"
                and item.get("agent_mode")
                and item.get("interrupt_returns_prompt", True)
            ):
                item["command"] = "bash"
                item["agent_mode"] = False
                item["managed_agent"] = False
                item["at_prompt"] = True
            elif (
                args[-1] == "C-c"
                and not item.get("agent_mode")
                and item.get("command") == "bash"
            ):
                item["pending"] = ""
                item["command"] = "bash"
                item["at_prompt"] = True
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    def _paste_buffer(self, args):
        name = args[args.index("-b") + 1]
        target = args[args.index("-t") + 1].removeprefix("=").removesuffix(":")
        item = self.sessions.get(target)
        if not item:
            return subprocess.CompletedProcess(args, 1, "", "missing")
        deliver = not self.paste_buffer_error or self.paste_deliver_then_error
        if deliver:
            item["pending"] = (
                item.get("delayed_browser_input", "")
                + item.get("pending", "")
                + self.buffers.get(name, "")
            )
            item["delayed_browser_input"] = ""
            self._execute_pending(item)
        if self.paste_buffer_error:
            return subprocess.CompletedProcess(args, 1, "", "paste failed")
        if "-d" in args:
            self.buffers.pop(name, None)
        return subprocess.CompletedProcess(args, 0, "", "")

    def pane_at_prompt(self, record, _pane):
        item = self.sessions.get(record["tmux_name"])
        return bool(item and item.get("at_prompt") and item.get("command") == "bash")

    def pane_has_managed_foreground(self, record, _pane):
        item = self.sessions.get(record["tmux_name"])
        return bool(item and item.get("managed_agent"))


class RemoteFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.low_priority_wrapper = "/verified/fleet-low-priority"
        self.low_priority_patcher = patch(
            "aeon.remote.instances.require_fleet_low_priority_wrapper",
            return_value=self.low_priority_wrapper,
        )
        self.low_priority_patcher.start()
        self.addCleanup(self.low_priority_patcher.stop)
        self.chat_delivery_ack_patcher = patch(
            "aeon.remote.instances.wait_for_chat_delivery_consumed",
            return_value=True,
        )
        self.chat_delivery_ack_patcher.start()
        self.addCleanup(self.chat_delivery_ack_patcher.stop)
        base = Path(self.temp.name)
        self.root = base / "workspaces"
        self.root.mkdir()
        self.workspace = self.root / "project"
        self.workspace.mkdir()
        self.config = RemoteConfig(
            project_root=Path(__file__).resolve().parents[2],
            state_dir=base / "state",
            allowed_roots=(self.root,),
            allowed_origins=("http://testserver",),
            allowed_hosts=("testserver",),
            python_executable="/usr/bin/python3",
            tmux_binary="/usr/bin/tmux",
            require_totp=False,
            allow_insecure_http=True,
            expected_coordinator_host="definitely-not-this-host",
        )
        self.config.prepare_state()
        self.store = RemoteStore(self.config.database_path)

    def tearDown(self):
        self.temp.cleanup()


class TestRemoteSecurity(RemoteFixture):
    def setUp(self):
        super().setUp()
        self.hasher = PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1)
        self.auth = AuthService(self.store, self.config, password_hasher=self.hasher)
        self.secret = generate_totp_secret()
        self.password = "correct horse battery staple"
        self.store.put_user(
            "admin", self.auth.hash_password(self.password), self.secret
        )
        self.fake = FakeTmux()
        self.manager = InstanceManager(
            self.store,
            self.config,
            command_runner=self.fake,
            pane_prompt_checker=self.fake.pane_at_prompt,
            pane_foreground_checker=self.fake.pane_has_managed_foreground,
        )
        self.client = TestClient(
            create_app(
                self.config,
                store=self.store,
                manager=self.manager,
                auth=self.auth,
            )
        )

    def login(self):
        return self.client.post(
            "/api/login",
            headers={"Origin": "http://testserver"},
            json={
                "username": "admin",
                "password": self.password,
                "remember": False,
            },
        )

    def test_shared_registry_allows_only_one_web_controller_lifespan(self):
        startup_calls: list[str] = []

        class StartupManager:
            def __init__(self, label):
                self.label = label

            def bootstrap(self):
                startup_calls.append(f"{self.label}:bootstrap")

        first = create_app(
            self.config,
            store=self.store,
            manager=StartupManager("first"),
            auth=self.auth,
            startup_initializer=lambda: startup_calls.append("first:home"),
        )
        second = create_app(
            self.config,
            store=self.store,
            manager=StartupManager("second"),
            auth=self.auth,
            startup_initializer=lambda: startup_calls.append("second:home"),
        )

        with TestClient(first) as first_client:
            self.assertEqual(first_client.get("/healthz").status_code, 200)
            self.assertEqual(startup_calls, ["first:bootstrap", "first:home"])
            with self.assertRaisesRegex(
                ControllerLockError, "controller is already active"
            ):
                with TestClient(second):
                    pass
            self.assertEqual(startup_calls, ["first:bootstrap", "first:home"])

        # Shutdown/TestClient exit releases the lifetime lease for a clean restart.
        with TestClient(second) as second_client:
            self.assertEqual(second_client.get("/healthz").status_code, 200)
        self.assertEqual(
            startup_calls,
            ["first:bootstrap", "first:home", "second:bootstrap", "second:home"],
        )

    def test_lifespan_store_guard_fails_closed_after_lock_replacement(self):
        app = create_app(
            self.config,
            store=self.store,
            manager=self.manager,
            auth=self.auth,
        )
        with TestClient(app) as client:
            self.assertEqual(client.get("/healthz").status_code, 200)
            lock_path = self.config.state_dir / "controller.lock"
            lock_path.rename(self.config.state_dir / "controller.lock.detached")
            lock_path.write_text("replacement", encoding="utf-8")
            lock_path.chmod(0o600)
            with self.assertRaises(ControllerLockError):
                client.get("/healthz")

    def test_totp_window_and_rejection(self):
        now = 1_700_000_000
        code = totp_code(self.secret, now)
        self.assertTrue(verify_totp(self.secret, code, now))
        self.assertTrue(verify_totp(self.secret, code, now + 30))
        self.assertFalse(verify_totp(self.secret, "12345x", now))

    def test_normal_provisioning_still_rejects_short_passwords(self):
        with self.assertRaisesRegex(ValueError, "at least 14"):
            self.auth.hash_password("test-only")

    def test_explicit_short_password_override_never_weakens_below_eight(self):
        value = self.auth.hash_password("eight888", minimum_length=8)
        self.assertTrue(value.startswith("$argon2"))
        with self.assertRaisesRegex(ValueError, "cannot be below 8"):
            self.auth.hash_password("anything", minimum_length=7)

    def test_password_only_adapter_fails_closed_when_totp_is_explicitly_enabled(self):
        protected = AuthService(
            self.store,
            replace(self.config, require_totp=True),
            password_hasher=self.hasher,
        )
        with self.assertRaises(AuthenticationError):
            protected.authenticate_password(
                "admin",
                self.password,
                client_ip="127.0.0.1",
                user_agent="test",
                remember=False,
            )

    def test_success_returns_and_audits_canonical_stored_username(self):
        self.store.put_user(
            "Austin",
            self.auth.hash_password("another correct horse battery staple"),
            "",
        )
        result = self.auth.authenticate_password(
            "aUsTiN",
            "another correct horse battery staple",
            client_ip="127.0.0.2",
            user_agent="test",
            remember=False,
        )

        self.assertEqual(result.username, "Austin")
        audit = self.store.recent_audit(limit=1)[0]
        self.assertEqual(audit["action"], "login_succeeded")
        self.assertEqual(audit["actor"], "Austin")

    def test_login_uses_httponly_lax_cookie_and_server_side_token(self):
        response = self.login()
        self.assertEqual(response.status_code, 200, response.text)
        cookie = response.headers["set-cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=lax", cookie)
        raw = self.client.cookies.get(self.config.cookie_name)
        self.assertTrue(raw)
        with sqlite3.connect(self.config.database_path) as conn:
            stored = conn.execute("SELECT token_hash FROM web_sessions").fetchone()[0]
        self.assertEqual(stored, token_digest(raw))
        self.assertNotEqual(stored, raw)

    def test_remembered_session_slides_but_logout_cannot_be_revived(self):
        result = self.auth.authenticate_password(
            "admin",
            self.password,
            client_ip="127.0.0.2",
            user_agent="phone",
            remember=True,
        )
        digest = token_digest(result.token)
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute(
                "UPDATE web_sessions SET expires_at=? WHERE token_hash=?",
                (time.time() + 3600, digest),
            )
        self.assertTrue(
            self.auth.refresh_session(
                result.token,
                lifetime_seconds=30 * 86400,
            )
        )
        refreshed = self.auth.session(result.token)
        self.assertGreater(refreshed["expires_at"], time.time() + 29 * 86400)

        self.auth.logout(result.token)
        self.assertFalse(
            self.auth.refresh_session(
                result.token,
                lifetime_seconds=30 * 86400,
            )
        )

    def test_standalone_totp_mode_keeps_strict_cookie(self):
        self.assertEqual(replace(self.config, require_totp=True).cookie_samesite, "strict")
        self.assertEqual(self.config.cookie_samesite, "lax")

    def test_csrf_and_origin_are_required_for_state_changes(self):
        response = self.login()
        csrf = response.json()["csrf_token"]
        missing = self.client.post(
            "/api/workspaces",
            headers={"Origin": "http://testserver"},
            json={"root": str(self.root), "name": "missing-csrf"},
        )
        self.assertEqual(missing.status_code, 403)
        evil = self.client.post(
            "/api/workspaces",
            headers={"Origin": "https://evil.example", "X-CSRF-Token": csrf},
            json={"root": str(self.root), "name": "evil-origin"},
        )
        self.assertEqual(evil.status_code, 403)
        accepted = self.client.post(
            "/api/workspaces",
            headers={"Origin": "http://testserver", "X-CSRF-Token": csrf},
            json={"root": str(self.root), "name": "safe-project"},
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertTrue((self.root / "safe-project").is_dir())

    def test_terminal_routes_are_csrf_protected_and_reject_command_fields(self):
        csrf = self.login().json()["csrf_token"]
        body = {"name": "Phone shell", "workspace": str(self.workspace)}
        missing = self.client.post(
            "/api/terminals",
            headers={"Origin": "http://testserver"},
            json=body,
        )
        self.assertEqual(missing.status_code, 403)

        injected = self.client.post(
            "/api/terminals",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": csrf,
            },
            json={**body, "command": "rm -rf /", "env": {"PATH": "/tmp"}},
        )
        self.assertEqual(injected.status_code, 422)

        created = self.client.post(
            "/api/terminals",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": csrf,
            },
            json=body,
        )
        self.assertEqual(created.status_code, 200, created.text)
        terminal = created.json()["instance"]
        self.assertEqual(terminal["kind"], "terminal")
        self.assertEqual(terminal["current_directory"], str(self.workspace.resolve()))
        launch = next(call for call in self.fake.calls if call[1] == "new-session")
        bash_at = launch.index("/bin/bash")
        self.assertEqual(launch[bash_at : bash_at + 3], ["/bin/bash", "--noprofile", "--rcfile"])
        self.assertTrue(launch[bash_at + 3].endswith("/managed-shell.rc"))
        self.assertEqual(launch[bash_at + 4], "-i")

        derived_only = self.client.post(
            f"/api/instances/{terminal['id']}/start-aeon-here",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": csrf,
            },
            json={"workspace": "/tmp", "command": "anything"},
        )
        self.assertEqual(derived_only.status_code, 422)

        rejected_activation = self.client.post(
            f"/api/instances/{terminal['id']}/activate-agent",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": csrf,
            },
            json={"kind": "aeon", "workspace": "/tmp"},
        )
        self.assertEqual(rejected_activation.status_code, 422)

        started = self.client.post(
            f"/api/instances/{terminal['id']}/activate-agent",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": csrf,
            },
            json={"kind": "aeon"},
        )
        self.assertEqual(started.status_code, 200, started.text)
        self.assertEqual(started.json()["instance"]["kind"], "aeon")
        self.assertEqual(started.json()["instance"]["id"], terminal["id"])

        ended = self.client.post(
            f"/api/instances/{terminal['id']}/end-agent",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": csrf,
            },
            json={},
        )
        self.assertEqual(ended.status_code, 200, ended.text)
        self.assertEqual(ended.json()["instance"]["kind"], "terminal")

    def test_browser_direct_agent_creation_is_disabled_and_terminal_name_is_optional(self):
        csrf = self.login().json()["csrf_token"]
        headers = {
            "Origin": "http://testserver",
            "X-CSRF-Token": csrf,
        }
        calls_before = len(self.fake.calls)
        rejected = self.client.post(
            "/api/instances",
            headers=headers,
            json={
                "kind": "aeon",
                "name": "Bypass",
                "workspace": str(self.workspace),
                "objective": "browser argv objective",
            },
        )
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("Create a terminal", rejected.json()["detail"])
        self.assertFalse(
            any(call[1] == "new-session" for call in self.fake.calls[calls_before:])
        )

        created = self.client.post(
            "/api/terminals",
            headers=headers,
            json={"workspace": str(self.workspace)},
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(created.json()["instance"]["name"], "Terminal 1")

    def test_terminal_name_is_durable_csrf_protected_and_collision_safe(self):
        csrf = self.login().json()["csrf_token"]
        headers = {
            "Origin": "http://testserver",
            "X-CSRF-Token": csrf,
        }
        first = self.client.post(
            "/api/terminals",
            headers=headers,
            json={"workspace": str(self.workspace)},
        ).json()["instance"]
        second = self.client.post(
            "/api/terminals",
            headers=headers,
            json={"workspace": str(self.workspace)},
        ).json()["instance"]
        original_tmux_name = self.store.get_instance(first["id"])["tmux_name"]

        missing_csrf = self.client.put(
            f"/api/instances/{first['id']}/name",
            headers={"Origin": "http://testserver"},
            json={"name": "Research lead"},
        )
        self.assertEqual(missing_csrf.status_code, 403)

        renamed = self.client.put(
            f"/api/instances/{first['id']}/name",
            headers=headers,
            json={"name": "Research lead"},
        )
        self.assertEqual(renamed.status_code, 200, renamed.text)
        self.assertEqual(renamed.json()["instance"]["name"], "Research lead")
        self.assertEqual(
            self.store.get_instance(first["id"])["tmux_name"], original_tmux_name
        )

        collision = self.client.put(
            f"/api/instances/{second['id']}/name",
            headers=headers,
            json={"name": "research LEAD"},
        )
        self.assertEqual(collision.status_code, 400)
        self.assertIn("already in use", collision.json()["detail"])

        injected = self.client.put(
            f"/api/instances/{first['id']}/name",
            headers=headers,
            json={"name": "Safe", "tmux_name": "other"},
        )
        self.assertEqual(injected.status_code, 422)

        audit = next(
            row
            for row in self.store.recent_audit(limit=20)
            if row["action"] == "instance_renamed"
        )
        self.assertEqual(audit["instance_id"], first["id"])
        self.assertEqual(json.loads(audit["details_json"]), {"name": "Research lead"})

    def test_api_exposes_and_recovers_force_stop_required_terminal(self):
        csrf = self.login().json()["csrf_token"]
        headers = {
            "Origin": "http://testserver",
            "X-CSRF-Token": csrf,
        }
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        record = self.store.get_instance(terminal["id"])
        self.fake.sessions[record["tmux_name"]].update(
            command="python3",
            at_prompt=False,
            managed_agent=False,
            agent_mode=False,
        )
        self.store.update_instance(terminal["id"], status="starting")

        listed = self.client.get("/api/instances")
        self.assertEqual(listed.status_code, 200, listed.text)
        item = next(
            value
            for value in listed.json()["instances"]
            if value["id"] == terminal["id"]
        )
        self.assertEqual(item["status"], "error")
        self.assertTrue(item["force_stop_required"])

        stopped = self.client.post(
            f"/api/instances/{terminal['id']}/force-stop",
            headers=headers,
            json={"confirmation": terminal["name"]},
        )
        self.assertEqual(stopped.status_code, 200, stopped.text)
        self.assertEqual(stopped.json()["instance"]["status"], "stopped")
        self.assertFalse(
            stopped.json()["instance"]["force_stop_required"]
        )

    def test_kill_route_is_csrf_protected_and_removes_only_after_exact_confirmation(self):
        csrf = self.login().json()["csrf_token"]
        headers = {
            "Origin": "http://testserver",
            "X-CSRF-Token": csrf,
        }
        terminal = self.manager.create_terminal(
            name="Disposable child",
            workspace=str(self.workspace),
            actor="admin",
        )
        record = self.store.get_instance(terminal["id"])
        calls_before = len(self.fake.calls)

        missing_csrf = self.client.post(
            f"/api/instances/{terminal['id']}/kill",
            headers={"Origin": "http://testserver"},
            json={"confirmation": terminal["name"]},
        )
        self.assertEqual(missing_csrf.status_code, 403, missing_csrf.text)
        self.assertIsNotNone(self.store.get_instance(terminal["id"]))
        self.assertEqual(len(self.fake.calls), calls_before)

        wrong_name = self.client.post(
            f"/api/instances/{terminal['id']}/kill",
            headers=headers,
            json={"confirmation": "not the visible name"},
        )
        self.assertEqual(wrong_name.status_code, 400, wrong_name.text)
        self.assertIsNotNone(self.store.get_instance(terminal["id"]))
        self.assertIn(record["tmux_name"], self.fake.sessions)

        killed = self.client.post(
            f"/api/instances/{terminal['id']}/kill",
            headers=headers,
            json={"confirmation": terminal["name"]},
        )
        self.assertEqual(killed.status_code, 200, killed.text)
        self.assertEqual(killed.json(), {"deleted": True})
        self.assertIsNone(self.store.get_instance(terminal["id"]))
        self.assertNotIn(record["tmux_name"], self.fake.sessions)
        listed_ids = {
            item["id"] for item in self.client.get("/api/instances").json()["instances"]
        }
        self.assertNotIn(terminal["id"], listed_ids)
        lifecycle_audits = [
            row["action"]
            for row in reversed(self.store.recent_audit(limit=20))
            if row["instance_id"] == terminal["id"]
        ]
        self.assertEqual(
            lifecycle_audits[-2:],
            ["instance_force_stopped", "instance_deleted"],
        )

    def test_security_headers_and_no_public_api_without_login(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.client.get("/api/instances").status_code, 401)

    def test_open_websocket_closes_when_captured_session_is_revoked(self):
        login = self.login()
        csrf = login.json()["csrf_token"]
        raw_session = self.client.cookies.get(self.config.cookie_name)
        terminal = self.manager.create_terminal(
            name="Revocation fixture",
            workspace=str(self.workspace),
            actor="admin",
        )

        async def parked_bridge(websocket, manager, instance_id, **_kwargs):
            await asyncio.Event().wait()

        with (
            patch("aeon.remote.app.bridge_terminal", new=parked_bridge),
            patch("aeon.remote.app.WEBSOCKET_SESSION_RECHECK_SECONDS", 0.01),
            self.client.websocket_connect(
                f"/ws/instances/{terminal['id']}",
                headers={"Origin": "http://testserver"},
                subprotocols=["aeon-v1", f"csrf.{csrf}"],
            ) as websocket,
        ):
            self.store.revoke_web_session(token_digest(raw_session))
            with self.assertRaises(WebSocketDisconnect) as closed:
                websocket.receive_text()
            self.assertEqual(closed.exception.code, 4401)

    def test_login_rate_limit(self):
        for _ in range(5):
            response = self.client.post(
                "/api/login",
                headers={"Origin": "http://testserver"},
                json={
                    "username": "admin",
                    "password": "wrong password forever",
                },
            )
            self.assertEqual(response.status_code, 401)
        blocked = self.client.post(
            "/api/login",
            headers={"Origin": "http://testserver"},
            json={
                "username": "admin",
                "password": self.password,
            },
        )
        self.assertEqual(blocked.status_code, 429)

    def test_login_rate_limit_cannot_be_bypassed_with_varying_usernames(self):
        for index in range(12):
            with self.assertRaises(AuthenticationError):
                self.auth.authenticate_password(
                    f"unknown-{index}",
                    "wrong password forever",
                    client_ip="203.0.113.41",
                    user_agent="test",
                    remember=False,
                )
        with self.assertRaises(LoginRateLimited):
            self.auth.authenticate_password(
                "one-more-unknown-user",
                "wrong password forever",
                client_ip="203.0.113.41",
                user_agent="test",
                remember=False,
            )

    def test_success_clears_peer_and_account_rate_buckets(self):
        kwargs = {
            "client_ip": "203.0.113.42",
            "user_agent": "test",
            "remember": False,
        }
        for _ in range(4):
            with self.assertRaises(AuthenticationError):
                self.auth.authenticate_password("admin", "wrong", **kwargs)
        result = self.auth.authenticate_password("admin", self.password, **kwargs)
        self.assertEqual(result.username, "admin")
        for _ in range(5):
            with self.assertRaises(AuthenticationError):
                self.auth.authenticate_password("admin", "wrong", **kwargs)
        with self.assertRaises(LoginRateLimited):
            self.auth.authenticate_password("admin", "wrong", **kwargs)

    def test_login_rate_registry_contains_no_plaintext_identity_or_peer(self):
        with self.assertRaises(AuthenticationError):
            self.auth.authenticate_password(
                "sensitive-unknown-name",
                "wrong",
                client_ip="203.0.113.43",
                user_agent="test",
                remember=False,
            )
        with self.store._connect() as conn:
            rows = conn.execute(
                "SELECT rate_key,attempt_id FROM login_attempts"
            ).fetchall()
            failed_audit = conn.execute(
                "SELECT actor,client_ip FROM audit_log WHERE action='login_failed'"
            ).fetchall()
        serialized = repr([tuple(row) for row in rows] + [tuple(row) for row in failed_audit])
        self.assertNotIn("sensitive-unknown-name", serialized)
        self.assertNotIn("203.0.113.43", serialized)
        self.assertTrue(rows)
        self.assertTrue(all(len(row["rate_key"]) == 64 for row in rows))

    def test_login_attempt_registry_is_durably_bounded_and_pruned(self):
        with patch("aeon.remote.store.time.time", return_value=1_000_000):
            for index in range(4):
                attempt_id = self.store.reserve_login_attempt(
                    {f"key-{index}": 1}, max_rows=4
                )
                self.assertIsNotNone(attempt_id)
            self.assertIsNone(
                self.store.reserve_login_attempt({"blocked-by-global-cap": 1}, max_rows=4)
            )
        with self.store._connect() as conn:
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM login_attempts").fetchone()[0],
                4,
            )
        with patch("aeon.remote.store.time.time", return_value=1_100_000):
            self.assertIsNotNone(
                self.store.reserve_login_attempt({"fresh-key": 1}, max_rows=4)
            )
        with self.store._connect() as conn:
            keys = {
                row[0] for row in conn.execute("SELECT rate_key FROM login_attempts")
            }
        self.assertEqual(keys, {"fresh-key"})

    def test_unauthenticated_failure_audit_is_bounded(self):
        with self.store._connect() as conn:
            conn.executemany(
                "INSERT INTO audit_log"
                "(occurred_at,actor,action,instance_id,client_ip,details_json) "
                "VALUES(?,?,?,NULL,?,?)",
                (
                    (float(index), "sha256:test", "login_failed", "sha256:peer", "{}")
                    for index in range(2055)
                ),
            )
        self.store.audit(
            "login_failed",
            actor="sha256:new",
            client_ip="sha256:new-peer",
        )
        with self.store._connect() as conn:
            failed_count = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE action='login_failed'"
            ).fetchone()[0]
        self.assertEqual(failed_count, 2048)

    def test_only_two_argon_verifications_run_concurrently(self):
        entered = 0
        entered_lock = threading.Lock()
        both_entered = threading.Event()
        release = threading.Event()

        class BlockingHasher:
            @staticmethod
            def hash(_value):
                return "dummy-hash"

            @staticmethod
            def verify(_stored, _supplied):
                nonlocal entered
                with entered_lock:
                    entered += 1
                    if entered == 2:
                        both_entered.set()
                release.wait(timeout=5)
                return False

        auth = AuthService(self.store, self.config, password_hasher=BlockingHasher())
        errors = []

        def authenticate(index):
            try:
                auth.authenticate_password(
                    f"parallel-{index}",
                    "wrong",
                    client_ip=f"203.0.113.{50 + index}",
                    user_agent="test",
                    remember=False,
                )
            except Exception as exc:  # noqa: BLE001 - asserted below
                errors.append(exc)

        first = threading.Thread(target=authenticate, args=(1,))
        second = threading.Thread(target=authenticate, args=(2,))
        first.start()
        second.start()
        self.assertTrue(both_entered.wait(timeout=5))
        authenticate(3)
        release.set()
        first.join(timeout=5)
        second.join(timeout=5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(entered, 2)
        self.assertEqual(sum(isinstance(exc, LoginRateLimited) for exc in errors), 1)
        self.assertEqual(sum(isinstance(exc, AuthenticationError) for exc in errors), 3)


class TestControllerLock(RemoteFixture):
    def test_lock_is_nonblocking_cross_process_and_released_exactly(self):
        script = """
import sys
from pathlib import Path
from aeon.remote.controller_lock import ControllerLock

lease = ControllerLock.acquire(Path(sys.argv[1]))
print("ready", flush=True)
sys.stdin.readline()
lease.close()
"""
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(self.config.state_dir)],
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(child.stdout.readline().strip(), "ready")
            with self.assertRaisesRegex(
                ControllerLockError, "controller is already active"
            ):
                ControllerLock.acquire(self.config.state_dir)
            child.stdin.write("release\n")
            child.stdin.flush()
            stdout, stderr = child.communicate(timeout=5)
            self.assertEqual(child.returncode, 0, msg=f"{stdout}\n{stderr}")
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)

        with ControllerLock.acquire(self.config.state_dir) as restarted:
            self.assertTrue(restarted.active)
        self.assertFalse(restarted.active)

    def test_lock_rejects_wrong_mode_and_symbolic_link(self):
        lock_path = self.config.state_dir / "controller.lock"
        lock_path.write_text("", encoding="utf-8")
        lock_path.chmod(0o644)
        with self.assertRaisesRegex(ControllerLockError, "mode-0600"):
            ControllerLock.acquire(self.config.state_dir)

        lock_path.unlink()
        target = self.config.state_dir.parent / "not-the-controller-lock"
        target.write_text("", encoding="utf-8")
        target.chmod(0o600)
        lock_path.symlink_to(target)
        with self.assertRaisesRegex(ControllerLockError, "symbolic link"):
            ControllerLock.acquire(self.config.state_dir)

    def test_read_lease_detects_legacy_file_only_controller_without_mutation(self):
        lock_path = self.config.state_dir / "controller.lock"
        lock_path.write_text("legacy", encoding="utf-8")
        lock_path.chmod(0o600)
        descriptor = os.open(lock_path, os.O_RDWR)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        before = (
            lock_path.stat().st_ino,
            lock_path.stat().st_mtime_ns,
            lock_path.stat().st_size,
            lock_path.read_bytes(),
        )
        try:
            with self.assertRaisesRegex(
                ControllerLockError, "controller is already active"
            ):
                ControllerLock.acquire_read_lease(self.config.state_dir)
            after = (
                lock_path.stat().st_ino,
                lock_path.stat().st_mtime_ns,
                lock_path.stat().st_size,
                lock_path.read_bytes(),
            )
            self.assertEqual(after, before)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def test_direct_lock_replacement_is_detected_and_cannot_split_brain(self):
        lease = ControllerLock.acquire(self.config.state_dir)
        lock_path = self.config.state_dir / "controller.lock"
        detached = self.config.state_dir / "controller.lock.detached"
        lock_path.rename(detached)
        lock_path.write_text("replacement", encoding="utf-8")
        lock_path.chmod(0o600)
        try:
            with self.assertRaisesRegex(ControllerLockError, "identity changed"):
                lease.assert_current()
            with self.assertRaisesRegex(
                ControllerLockError, "controller is already active"
            ):
                ControllerLock.acquire(self.config.state_dir)
        finally:
            lease.close()

        with ControllerLock.acquire(self.config.state_dir) as restarted:
            restarted.assert_current()

    def test_state_directory_replacement_is_detected_and_cannot_split_brain(self):
        lease = ControllerLock.acquire(self.config.state_dir)
        detached = self.config.state_dir.with_name("state.detached")
        self.config.state_dir.rename(detached)
        self.config.state_dir.mkdir(mode=0o700)
        try:
            with self.assertRaisesRegex(ControllerLockError, "identity changed"):
                lease.assert_current()
            with self.assertRaisesRegex(
                ControllerLockError, "controller is already active"
            ):
                ControllerLock.acquire(self.config.state_dir)
        finally:
            lease.close()

        with ControllerLock.acquire(self.config.state_dir) as restarted:
            restarted.assert_current()

    def test_store_guard_runs_before_database_path_access(self):
        lease = ControllerLock.acquire(self.config.state_dir)
        self.store.set_controller_guard(lease.assert_current)
        lock_path = self.config.state_dir / "controller.lock"
        lock_path.rename(self.config.state_dir / "controller.lock.detached")
        lock_path.write_text("replacement", encoding="utf-8")
        lock_path.chmod(0o600)
        try:
            with (
                patch.object(
                    self.store,
                    "_open_parent_fd",
                    side_effect=AssertionError("database path was accessed"),
                ),
                self.assertRaises(ControllerLockError),
            ):
                self.store.admin_count()
        finally:
            self.store.set_controller_guard(None)
            lease.close()

    def test_open_connection_rechecks_controller_before_each_statement(self):
        lease = ControllerLock.acquire(self.config.state_dir)
        self.store.set_controller_guard(lease.assert_current)
        connection = self.store._connect()
        lock_path = self.config.state_dir / "controller.lock"
        lock_path.rename(self.config.state_dir / "controller.lock.detached")
        lock_path.write_text("replacement", encoding="utf-8")
        lock_path.chmod(0o600)
        try:
            with self.assertRaises(ControllerLockError):
                connection.execute("CREATE TABLE forbidden_after_lock_loss (id INTEGER)")
        finally:
            connection.close()
            self.store.set_controller_guard(None)
            lease.close()
        with sqlite3.connect(self.config.database_path) as verification:
            self.assertIsNone(
                verification.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type='table' AND name='forbidden_after_lock_loss'"
                ).fetchone()
            )


class TestRemoteStateAndStoreSafety(unittest.TestCase):
    @staticmethod
    def _config(state: Path) -> RemoteConfig:
        return RemoteConfig(
            project_root=Path(__file__).resolve().parents[2],
            state_dir=state,
            allowed_roots=(state.parent,),
            allowed_origins=("http://testserver",),
            allowed_hosts=("testserver",),
            require_totp=False,
            allow_insecure_http=True,
        )

    def test_env_state_symlink_is_preserved_lexically_and_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            victim = base / "victim"
            victim.mkdir(mode=0o755)
            state = base / "configured-state"
            state.symlink_to(victim, target_is_directory=True)
            environment = {
                "AEON_REMOTE_STATE_DIR": str(state),
                "AEON_REMOTE_PROJECT_ROOT": str(Path(__file__).resolve().parents[2]),
            }
            with patch.dict(os.environ, environment, clear=False):
                config = RemoteConfig.from_env(validate_server=False)
            self.assertEqual(config.state_dir, state.absolute())
            with self.assertRaises((OSError, RuntimeError)):
                config.prepare_state()
            self.assertEqual(stat.S_IMODE(victim.stat().st_mode), 0o755)
            self.assertFalse((victim / "instances").exists())

    def test_missing_no_follow_support_fails_before_state_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            with (
                patch.object(os, "O_NOFOLLOW", None),
                self.assertRaisesRegex(RuntimeError, "without following links"),
            ):
                self._config(state).prepare_state()
            self.assertFalse(state.exists())

    def test_symlinked_instances_directory_is_rejected_without_touching_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            state = base / "state"
            state.mkdir(mode=0o700)
            victim = base / "instances-victim"
            victim.mkdir(mode=0o755)
            (state / "instances").symlink_to(victim, target_is_directory=True)

            with self.assertRaises((OSError, RuntimeError)):
                self._config(state).prepare_state()
            self.assertEqual(stat.S_IMODE(victim.stat().st_mode), 0o755)
            self.assertEqual(list(victim.iterdir()), [])

    def test_database_symlink_is_rejected_without_mutating_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            state = base / "state"
            state.mkdir(mode=0o700)
            victim = base / "victim.sqlite3"
            victim.write_bytes(b"not a nexus registry")
            victim.chmod(0o644)
            before = victim.read_bytes()
            (state / "remote.sqlite3").symlink_to(victim)

            with self.assertRaisesRegex(RuntimeError, "unavailable or unsafe"):
                RemoteStore(state / "remote.sqlite3")
            self.assertEqual(victim.read_bytes(), before)
            self.assertEqual(stat.S_IMODE(victim.stat().st_mode), 0o644)

    def test_database_hardlink_is_rejected_without_mutating_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            state = base / "state"
            state.mkdir(mode=0o700)
            victim = base / "victim.sqlite3"
            victim.write_bytes(b"not a nexus registry")
            victim.chmod(0o644)
            database = state / "remote.sqlite3"
            os.link(victim, database)

            with self.assertRaisesRegex(RuntimeError, "singly-linked"):
                RemoteStore(database)
            self.assertEqual(stat.S_IMODE(victim.stat().st_mode), 0o644)
            self.assertEqual(victim.stat().st_nlink, 2)

    def test_read_only_store_never_creates_or_migrates_registry_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            missing = base / "missing" / "remote.sqlite3"
            with (
                ControllerLock.acquire_read_lease(missing.parent) as read_lease,
                self.assertRaises((OSError, RuntimeError)),
            ):
                RemoteStore(
                    missing,
                    read_only=True,
                    controller_guard=read_lease.assert_current,
                )
            self.assertFalse(missing.parent.exists())

            state = base / "state"
            state.mkdir(mode=0o700)
            database = state / "remote.sqlite3"
            with sqlite3.connect(database) as conn:
                conn.execute(
                    "CREATE TABLE instances (id TEXT PRIMARY KEY, created_at REAL NOT NULL)"
                )
            database.chmod(0o600)
            before_bytes = database.read_bytes()
            before_mtime = database.stat().st_mtime_ns

            with ControllerLock.acquire_read_lease(state) as read_lease:
                store = RemoteStore(
                    database,
                    read_only=True,
                    controller_guard=read_lease.assert_current,
                )
                self.assertEqual(store.list_instances(), [])
            self.assertEqual(database.read_bytes(), before_bytes)
            self.assertEqual(database.stat().st_mtime_ns, before_mtime)
            self.assertFalse((state / ".remote.sqlite3.initialize.lock").exists())
            with sqlite3.connect(database) as conn:
                columns = [
                    row[1] for row in conn.execute("PRAGMA table_info(instances)")
                ]
            self.assertEqual(columns, ["id", "created_at"])

    def test_read_only_store_rejects_live_sidecars_without_changing_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            database = state / "remote.sqlite3"
            RemoteStore(database)
            writer = sqlite3.connect(database)
            try:
                writer.execute("PRAGMA wal_autocheckpoint=0")
                writer.execute(
                    "INSERT INTO audit_log(occurred_at,actor,action) VALUES(1,'test','test')"
                )
                writer.commit()
                sidecars = [
                    path
                    for suffix in ("-wal", "-shm", "-journal")
                    if (path := Path(f"{database}{suffix}")).exists()
                ]
                self.assertTrue(sidecars)
                before = {
                    path.name: (
                        path.stat().st_ino,
                        path.stat().st_mtime_ns,
                        path.stat().st_size,
                        path.read_bytes(),
                    )
                    for path in sidecars
                }
                with (
                    ControllerLock.acquire_read_lease(state) as read_lease,
                    self.assertRaisesRegex(RuntimeError, "without sidecars"),
                ):
                    RemoteStore(
                        database,
                        read_only=True,
                        controller_guard=read_lease.assert_current,
                    )
                after = {
                    path.name: (
                        path.stat().st_ino,
                        path.stat().st_mtime_ns,
                        path.stat().st_size,
                        path.read_bytes(),
                    )
                    for path in sidecars
                }
                self.assertEqual(after, before)
            finally:
                writer.close()

    def test_writable_store_tolerates_sqlite_unlinking_sidecar_during_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            database = state / "remote.sqlite3"
            store = RemoteStore(database)
            sidecar = Path(f"{database}-wal")
            sidecar.write_bytes(b"transient sqlite sidecar")
            sidecar.chmod(0o600)
            real_stat = os.stat

            def unlink_before_identity(path, *args, **kwargs):
                if path == sidecar.name and kwargs.get("dir_fd") is not None:
                    sidecar.unlink(missing_ok=True)
                    raise FileNotFoundError(2, "No such file or directory", path)
                return real_stat(path, *args, **kwargs)

            with patch("aeon.remote.store.os.stat", side_effect=unlink_before_identity):
                self.assertEqual(store.admin_count(), 0)
            self.assertFalse(sidecar.exists())

    def test_store_rejects_database_and_parent_replacement_after_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            state = base / "state"
            store = RemoteStore(state / "remote.sqlite3")
            database = state / "remote.sqlite3"
            database.rename(state / "remote.sqlite3.detached")
            database.write_bytes(b"")
            database.chmod(0o600)
            with self.assertRaisesRegex(RuntimeError, "database identity changed"):
                store.admin_count()

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            state = base / "state"
            store = RemoteStore(state / "remote.sqlite3")
            state.rename(base / "state.detached")
            state.mkdir(mode=0o700)
            with self.assertRaisesRegex(RuntimeError, "directory identity changed"):
                store.admin_count()

    def test_open_connection_rechecks_database_identity_before_statement(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            store = RemoteStore(state / "remote.sqlite3")
            connection = store._connect()
            database = state / "remote.sqlite3"
            detached = state / "remote.sqlite3.detached"
            database.rename(detached)
            database.write_bytes(b"")
            database.chmod(0o600)
            try:
                with self.assertRaisesRegex(RuntimeError, "database identity changed"):
                    connection.execute("CREATE TABLE forbidden_after_db_swap (id INTEGER)")
            finally:
                connection.close()
            with sqlite3.connect(detached) as verification:
                self.assertIsNone(
                    verification.execute(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type='table' AND name='forbidden_after_db_swap'"
                    ).fetchone()
                )


class TestAdminProvisioning(unittest.TestCase):
    def test_explicit_short_password_is_read_privately_and_only_hash_is_stored(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            chosen = "fixture8!"
            args = SimpleNamespace(
                username="local-user",
                replace=True,
                allow_short_password=True,
                password_only=True,
            )
            output = io.StringIO()
            environment = {
                "AEON_REMOTE_STATE_DIR": str(state),
                "AEON_REMOTE_PROJECT_ROOT": str(Path(__file__).resolve().parents[2]),
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch("aeon.remote.cli.getpass.getpass", side_effect=[chosen, chosen]),
                redirect_stdout(output),
            ):
                self.assertEqual(init_admin(args), 0)

            rendered = output.getvalue()
            self.assertNotIn(chosen, rendered)
            self.assertNotIn("TOTP", rendered)
            self.assertNotIn("otpauth", rendered)
            user = RemoteStore(state / "remote.sqlite3").get_user("local-user")
            self.assertTrue(user["password_hash"].startswith("$argon2"))
            self.assertNotIn(chosen, user["password_hash"])
            self.assertEqual(user["totp_secret"], "")

    def test_replacement_canonicalizes_username_without_duplicate_admin(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RemoteStore(Path(temporary) / "remote.sqlite3")
            store.put_user("Austin", "$argon2id$old", "legacy")
            user = store.get_user("Austin")
            store.create_web_session(
                user["id"], "old-token-digest", "old-csrf", time.time() + 3600, "ua"
            )
            store.put_user("austin", "$argon2id$new", "", replace=True)

            self.assertEqual(store.admin_count(), 1)
            self.assertEqual(store.get_user("AUSTIN")["username"], "austin")
            self.assertIsNone(store.get_web_session("old-token-digest"))


class TestReadOnlyStatus(unittest.TestCase):
    @staticmethod
    def _snapshot(directory: Path) -> dict[str, tuple[int, int, bytes]]:
        result = {}
        for path in sorted(directory.iterdir()):
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode):
                result[path.name] = (
                    stat.S_IMODE(metadata.st_mode),
                    metadata.st_mtime_ns,
                    path.read_bytes(),
                )
        return result

    def test_absent_state_returns_error_without_creating_any_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "missing" / "state"
            environment = {
                "AEON_REMOTE_STATE_DIR": str(state),
                "AEON_REMOTE_PROJECT_ROOT": str(Path(__file__).resolve().parents[2]),
            }
            errors = io.StringIO()
            with patch.dict(os.environ, environment, clear=False), redirect_stderr(errors):
                self.assertEqual(remote_status(SimpleNamespace()), 2)
            self.assertFalse(state.exists())
            self.assertIn("unavailable", errors.getvalue())

    def test_status_is_read_only_when_no_controller_is_active(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            database = state / "remote.sqlite3"
            store = RemoteStore(database)
            store.put_user("admin", "$argon2id$fixture", "")
            environment = {
                "AEON_REMOTE_STATE_DIR": str(state),
                "AEON_REMOTE_PROJECT_ROOT": str(Path(__file__).resolve().parents[2]),
            }
            output = io.StringIO()
            errors = io.StringIO()
            before = self._snapshot(state)
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(
                    RemoteConfig,
                    "prepare_state",
                    side_effect=AssertionError("status prepared state"),
                ),
                patch.object(
                    RemoteStore,
                    "_initialize",
                    side_effect=AssertionError("status initialized registry"),
                ),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                self.assertEqual(remote_status(SimpleNamespace()), 0)
            self.assertEqual(self._snapshot(state), before)
            self.assertEqual(errors.getvalue(), "")
            self.assertIn("Administrators: 1", output.getvalue())

    def test_status_refuses_active_controller_without_touching_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            RemoteStore(state / "remote.sqlite3")
            environment = {
                "AEON_REMOTE_STATE_DIR": str(state),
                "AEON_REMOTE_PROJECT_ROOT": str(Path(__file__).resolve().parents[2]),
            }
            errors = io.StringIO()
            with ControllerLock.acquire(state) as lease:
                before = self._snapshot(state)
                with (
                    patch.dict(os.environ, environment, clear=False),
                    redirect_stderr(errors),
                ):
                    self.assertEqual(remote_status(SimpleNamespace()), 2)
                lease.assert_current()
                self.assertEqual(self._snapshot(state), before)
            self.assertIn("active", errors.getvalue())


class TestRemoteStoreMigration(unittest.TestCase):
    def test_worker_transport_receipt_migration_is_concurrent_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "remote.sqlite3"
            with sqlite3.connect(database) as conn:
                conn.executescript(
                    """
                    CREATE TABLE instances (
                        id TEXT PRIMARY KEY,
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

            barrier = threading.Barrier(8)
            failures = []

            def initialize() -> None:
                try:
                    barrier.wait(timeout=5)
                    RemoteStore(database)
                except Exception as exc:  # pragma: no cover - asserted below
                    failures.append(exc)

            threads = [threading.Thread(target=initialize) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(failures, [])
            RemoteStore(database)
            with sqlite3.connect(database) as conn:
                columns = [
                    row[1] for row in conn.execute("PRAGMA table_info(instances)")
                ]
            self.assertEqual(columns.count("transport_pid"), 1)
            self.assertEqual(
                columns.count("transport_process_create_time"), 1
            )
            self.assertEqual(columns.count("awaiting_objective"), 1)
            self.assertEqual(columns.count("deferred_message_id"), 1)

    def test_legacy_login_attempt_table_adds_attempt_id_before_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "remote.sqlite3"
            with sqlite3.connect(database) as conn:
                conn.execute(
                    "CREATE TABLE login_attempts ("
                    "rate_key TEXT NOT NULL, attempted_at REAL NOT NULL, "
                    "succeeded INTEGER NOT NULL)"
                )
                conn.execute(
                    "INSERT INTO login_attempts(rate_key,attempted_at,succeeded) "
                    "VALUES('legacy-digest',0,0)"
                )
            store = RemoteStore(database)
            with sqlite3.connect(database) as conn:
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(login_attempts)")
                }
                indexes = {
                    row[1] for row in conn.execute("PRAGMA index_list(login_attempts)")
                }
                legacy = conn.execute(
                    "SELECT attempt_id FROM login_attempts WHERE rate_key='legacy-digest'"
                ).fetchone()
            self.assertIn("attempt_id", columns)
            self.assertIn("login_attempts_attempt_id", indexes)
            self.assertEqual(legacy, (None,))
            self.assertIsNotNone(store.reserve_login_attempt({"new-digest": 1}))

    def test_existing_web_and_local_rows_migrate_to_aeon_kind(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "remote.sqlite3"
            now = time.time()
            with sqlite3.connect(database) as conn:
                conn.executescript(
                    """
                    CREATE TABLE instances (
                        id TEXT PRIMARY KEY,
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
                values = (
                    "created", "running", now, now, None, "", "fixture"
                )
                conn.execute(
                    "INSERT INTO instances VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "web-id", "Legacy web", "aeon-web", "/tmp", "", None,
                        None, *values, "web",
                    ),
                )
                conn.execute(
                    "INSERT INTO instances VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "local-id", "Legacy local", "aeon-local", "/tmp", "", None,
                        None, *values, "local",
                    ),
                )

            store = RemoteStore(database)
            self.assertEqual(store.get_instance("web-id")["kind"], "aeon")
            self.assertEqual(store.get_instance("local-id")["kind"], "aeon")
            self.assertEqual(store.get_instance("web-id")["shell_backed"], 0)
            self.assertIsNone(store.get_instance("web-id")["last_agent_kind"])
            self.assertIsNone(store.get_instance("web-id")["transport_pid"])
            self.assertIsNone(
                store.get_instance("web-id")["transport_process_create_time"]
            )
            self.assertEqual(store.get_instance("web-id")["awaiting_objective"], 0)
            self.assertIsNone(store.get_instance("web-id")["deferred_message_id"])
            with sqlite3.connect(database) as conn:
                columns = {
                    row[1]: row for row in conn.execute("PRAGMA table_info(instances)")
                }
            self.assertEqual(columns["kind"][2].upper(), "TEXT")
            self.assertEqual(columns["kind"][3], 1)
            self.assertEqual(columns["kind"][4], "'aeon'")
            self.assertEqual(columns["shell_backed"][2].upper(), "INTEGER")
            self.assertEqual(columns["shell_backed"][3], 1)
            self.assertIn("last_agent_kind", columns)
            self.assertIn("transport_pid", columns)
            self.assertIn("transport_process_create_time", columns)
            self.assertEqual(columns["awaiting_objective"][3], 1)
            self.assertEqual(columns["awaiting_objective"][4], "0")
            self.assertIn("deferred_message_id", columns)

    def test_existing_terminal_rows_migrate_to_shell_backed_tabs(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "remote.sqlite3"
            now = time.time()
            with sqlite3.connect(database) as conn:
                conn.executescript(
                    """
                    CREATE TABLE instances (
                        id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL DEFAULT 'aeon',
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
                    "INSERT INTO instances VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "terminal-id", "terminal", "Legacy terminal", "terminal-old",
                        "/tmp", "", None, None, "running", "running", now, now,
                        now, "", "fixture", "web",
                    ),
                )

            store = RemoteStore(database)
            migrated = store.get_instance("terminal-id")
            self.assertEqual(migrated["kind"], "terminal")
            self.assertEqual(migrated["shell_backed"], 1)
            self.assertIsNone(migrated["last_agent_kind"])

    def test_partial_shell_migration_is_repaired_idempotently(self):
        """A crash after ALTER but before backfill cannot strand terminal rows."""

        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "remote.sqlite3"
            now = time.time()
            with sqlite3.connect(database) as conn:
                conn.executescript(
                    """
                    CREATE TABLE instances (
                        id TEXT PRIMARY KEY,
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
                    "INSERT INTO instances VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "terminal-id", "terminal", 0, None, "Legacy terminal",
                        "terminal-old", "/tmp", "", None, None, "running",
                        "running", now, now, now, "", "fixture", "web",
                    ),
                )
                conn.execute(
                    "INSERT INTO instances VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "agent-id", "codex", 1, None, "Legacy same-tab agent",
                        "codex-old", "/tmp", "", None, None, "running",
                        "running", now, now, now, "", "fixture", "web",
                    ),
                )

            store = RemoteStore(database)
            self.assertEqual(store.get_instance("terminal-id")["shell_backed"], 1)
            self.assertEqual(store.get_instance("agent-id")["last_agent_kind"], "codex")
            # A second initialization is the same repair, not a one-shot branch.
            again = RemoteStore(database)
            self.assertEqual(again.get_instance("terminal-id")["shell_backed"], 1)
            self.assertEqual(again.get_instance("agent-id")["last_agent_kind"], "codex")

    def test_concurrent_initializers_serialize_column_migration(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "remote.sqlite3"
            now = time.time()
            with sqlite3.connect(database) as conn:
                conn.executescript(
                    """
                    CREATE TABLE instances (
                        id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL DEFAULT 'aeon',
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
                    "INSERT INTO instances VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "terminal-id", "terminal", "Terminal", "terminal-old",
                        "/tmp", "", None, None, "running", "running", now, now,
                        now, "", "fixture", "web",
                    ),
                )
            barrier = threading.Barrier(3)
            stores: list[RemoteStore] = []
            errors: list[Exception] = []

            def initialize():
                barrier.wait()
                try:
                    stores.append(RemoteStore(database))
                except Exception as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            threads = [threading.Thread(target=initialize) for _ in range(2)]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=3)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(stores), 2)
            self.assertEqual(stores[0].get_instance("terminal-id")["shell_backed"], 1)


class TestInstanceManager(RemoteFixture):
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

    def _cross_kind_profile_fixture(self):
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
            workspace=str(self.workspace), actor="admin"
        )
        self.store.transition_shell_mode(
            terminal["id"],
            expected_kind="terminal",
            kind="aeon",
            last_agent_kind="aeon",
        )
        profile = instruction_service.create_profile(
            agent_kind="aeon", name="Prior Aeon role", actor="admin"
        )
        version = instruction_service.save_version(
            profile["id"],
            label="v1",
            content="Keep the prior Aeon role exact.",
            actor="admin",
        )
        desired = instruction_service.select_profile_version(
            terminal["id"], version["id"]
        )
        instruction_service.mark_applied(
            terminal["id"],
            profile_version_id=version["id"],
            local_revision=desired["desired_local_revision"],
        )
        self.store.transition_shell_mode(
            terminal["id"],
            expected_kind="aeon",
            kind="terminal",
        )
        with self.store._connect() as conn:
            binding = dict(
                conn.execute(
                    "SELECT * FROM instance_instruction_bindings WHERE instance_id=?",
                    (terminal["id"],),
                ).fetchone()
            )
        return manager, terminal, binding

    def test_workspace_symlink_escape_is_rejected(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (self.root / "escape").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(InstanceError):
            self.manager.validate_workspace(self.root / "escape")

    def test_agent_project_binding_is_exact_and_inferred_from_active_root(self):
        now = time.time()
        project_id = "pr-" + "a" * 32
        self.store.create_project(
            {
                "id": project_id,
                "name": "Exact project",
                "root": str(self.workspace.resolve()),
                "description": "fixture",
                "default_agent_kind": "aeon",
                "status": "active",
                "created_at": now,
                "updated_at": now,
                "created_by": "admin",
            }
        )

        inferred = self.manager.create_instance(
            name="Inferred project agent",
            workspace=str(self.workspace),
            objective="",
            max_iterations=None,
            actor="admin",
            defer_until_message=True,
        )
        self.assertEqual(inferred["project_id"], project_id)
        self.assertEqual(
            self.store.get_instance(inferred["id"])["project_id"], project_id
        )

        explicit = self.manager.create_instance(
            name="Explicit project agent",
            workspace=str(self.workspace),
            objective="",
            max_iterations=None,
            actor="admin",
            defer_until_message=True,
            project_id=project_id,
        )
        self.assertEqual(explicit["project_id"], project_id)

        other = self.root / "other"
        other.mkdir()
        with self.assertRaisesRegex(InstanceError, "must match its project root"):
            self.manager.create_instance(
                name="Mismatched project agent",
                workspace=str(other),
                objective="",
                max_iterations=None,
                actor="admin",
                defer_until_message=True,
                project_id=project_id,
            )
        with self.assertRaisesRegex(InstanceError, "Project identity is invalid"):
            self.manager.create_instance(
                name="Malformed project agent",
                workspace=str(self.workspace),
                objective="",
                max_iterations=None,
                actor="admin",
                defer_until_message=True,
                project_id="../../not-a-project",
            )

    def test_agent_created_skills_are_private_listed_and_revision_checked(self):
        instance = self.manager.create_instance(
            name="Skill agent", workspace=str(self.workspace), objective="Build useful things",
            max_iterations=None, actor="admin",
        )
        skill_dir = self.config.instance_state_dir / instance["id"] / "skills" / "research"
        skill_dir.mkdir(parents=True, mode=0o700)
        os.chmod(skill_dir.parent, 0o700)
        os.chmod(skill_dir, 0o700)
        skill_file = skill_dir / "useful_versions.txt"
        skill_file.write_text("1. Find gaps.\n", encoding="utf-8")
        os.chmod(skill_file, 0o600)

        listed = self.manager.get_private_skills(instance["id"])
        self.assertTrue(listed["supported"])
        self.assertEqual([item["skill_path"] for item in listed["skills"]], ["research/useful_versions"])
        revision = listed["skills"][0]["revision"]
        updated = self.manager.update_private_skill(
            instance["id"], category="research", skill_name="useful_versions",
            content="1. Find verified gaps.\n", expected_revision=revision, actor="admin",
        )
        self.assertEqual(updated["skills"][0]["content"], "1. Find verified gaps.\n")
        self.assertEqual(stat.S_IMODE(skill_file.stat().st_mode), 0o600)
        with self.assertRaisesRegex(InstanceError, "changed since it was loaded"):
            self.manager.update_private_skill(
                instance["id"], category="research", skill_name="useful_versions",
                content="stale overwrite", expected_revision=revision, actor="admin",
            )

    def test_fresh_context_restarts_legacy_direct_agent_and_preserves_skills(self):
        instance = self.manager.create_instance(
            name="Legacy direct agent", workspace=str(self.workspace),
            objective="Research useful contributions", max_iterations=None,
            actor="admin",
        )
        self.assertFalse(instance["shell_backed"])
        instance_state = self.config.instance_state_dir / instance["id"]
        skill_dir = instance_state / "skills" / "research"
        skill_dir.mkdir(parents=True, mode=0o700)
        os.chmod(skill_dir.parent, 0o700)
        os.chmod(skill_dir, 0o700)
        skill = skill_dir / "validated_search.txt"
        skill.write_text("1. Verify exact metadata.\n", encoding="utf-8")
        os.chmod(skill, 0o600)

        worker_state_root = Path(self.temp.name) / "worker-state"
        with patch.dict(
            os.environ, {"AEON_STATE_DIR": str(worker_state_root)}, clear=False
        ):
            worker_session = self.manager._worker_session_directory(
                self.store.get_instance(instance["id"])
            )
            worker_session.mkdir(mode=0o700, parents=True)
            state_file = worker_session / "session_state.json"
            state_file.write_text(
                json.dumps({"memories": {"old": "conversation"}}),
                encoding="utf-8",
            )
            state_file.chmod(0o600)
            restarted = self.manager.fresh_restart_agent(
                instance["id"], actor="admin"
            )

        self.assertEqual(restarted["mode"], "agent")
        self.assertEqual(restarted["status"], "running")
        self.assertFalse(restarted["shell_backed"])
        self.assertEqual(restarted["objective"], "")
        self.assertEqual(json.loads(state_file.read_text(encoding="utf-8")), {})
        self.assertEqual(skill.read_text(encoding="utf-8"), "1. Verify exact metadata.\n")
        lifecycle_calls = [call[1] for call in self.fake.calls]
        self.assertIn("send-keys", lifecycle_calls)
        self.assertIn("kill-session", lifecycle_calls)
        self.assertGreaterEqual(lifecycle_calls.count("new-session"), 2)
        launches = [call for call in self.fake.calls if call[1] == "new-session"]
        self.assertNotIn("--start", launches[-1])
        self.assertNotIn("--resume-unfinished", launches[-1])
        self.assertNotIn("Research useful contributions", launches[-1])

    def test_private_skills_and_related_wiki_transfer_without_overwrite(self):
        source = self.manager.create_instance(
            name="Source specialist", workspace=str(self.workspace),
            objective="Learn a repeatable workflow", max_iterations=None, actor="admin",
        )
        target = self.manager.create_instance(
            name="Target specialist", workspace=str(self.workspace),
            objective="Reuse a proven workflow", max_iterations=None, actor="admin",
        )
        source_state = self.config.instance_state_dir / source["id"]
        category = source_state / "skills" / "conversion"
        category.mkdir(parents=True, mode=0o700)
        os.chmod(category.parent, 0o700)
        os.chmod(category, 0o700)
        source_file = category / "verified_export.txt"
        source_file.write_text("1. Validate identity.\n2. Export.", encoding="utf-8")
        os.chmod(source_file, 0o600)
        source_note = SkillKnowledgeStore(source_state / "skill-wiki").save_note(
            title="Export evidence",
            content="The second attempt passed an exact round-trip comparison.",
            related_skill_paths=["conversion/verified_export"],
            learning={
                "candidate_skill_path": "conversion/verified_export",
                "procedure": "Validate the source identity, then export it.",
                "verification": "The second attempt matched byte-for-byte.",
                "procedure_stable": True,
                "uncertainty": "low",
            },
            experience={
                "request_id": "transfer-fixture",
                "attempt_count": 2,
                "failure_count": 1,
                "success_count": 1,
                "recovered_after_failure": True,
                "receipts": [
                    {
                        "tool": "run_command",
                        "status": "failed",
                        "error_code": "mismatch",
                        "summary_sha256": "a" * 64,
                    },
                    {
                        "tool": "run_command",
                        "status": "ok",
                        "error_code": "",
                        "summary_sha256": "b" * 64,
                    },
                ],
            },
        )
        self.assertTrue(source_note["skill_evidence_eligible"])

        source_payload = self.manager.get_private_skills(source["id"])
        revision = next(
            item["revision"] for item in source_payload["effective_skills"]
            if item["skill_path"] == "conversion/verified_export"
        )
        self.assertEqual(source_payload["maximum_skills"], 16)
        target_root = self.config.instance_state_dir / target["id"] / "skills"
        target_root.mkdir(parents=True, mode=0o700)
        os.chmod(target_root, 0o700)
        SkillsManager(instance_dir=target_root).learned_store().save_protocol(
            category="conversion",
            skill_name="verified_export",
            content_revision=revision,
            evidence=[{"note_id": "note-" + "c" * 32, "revision": "d" * 64}],
        )
        target_payload = self.manager.get_private_skills(target["id"])
        self.assertIn(
            source["id"], {item["id"] for item in target_payload["transfer_sources"]}
        )

        transferred = self.manager.transfer_private_skills(
            target["id"], source_instance_id=source["id"],
            selections=[{
                "skill_path": "conversion/verified_export",
                "revision": revision,
            }],
            include_knowledge=False, actor="admin",
        )
        self.assertEqual(transferred["transfer"]["copied"], ["conversion/verified_export"])
        self.assertEqual(transferred["transfer"]["knowledge_notes_copied"], 0)
        copied = next(
            item for item in transferred["skills"]
            if item["skill_path"] == "conversion/verified_export"
        )
        self.assertEqual(copied["content"], "1. Validate identity.\n2. Export.")
        self.assertEqual(copied["revision"], revision)
        copied_effective = next(
            item for item in transferred["effective_skills"]
            if item["skill_path"] == "conversion/verified_export"
        )
        self.assertEqual(copied_effective["lifecycle"]["status"], "needs_review")
        self.assertTrue(copied_effective["lifecycle"]["evidence_stale"])
        self.assertEqual(transferred["knowledge_notes"], [])

        repeated = self.manager.transfer_private_skills(
            target["id"], source_instance_id=source["id"],
            selections=[{
                "skill_path": "conversion/verified_export",
                "revision": revision,
            }],
            include_knowledge=True, actor="admin",
        )
        self.assertEqual(repeated["transfer"]["copied"], [])
        self.assertEqual(
            repeated["transfer"]["already_known"], ["conversion/verified_export"]
        )
        self.assertEqual(repeated["transfer"]["knowledge_notes_copied"], 1)
        self.assertEqual(len(repeated["knowledge_notes"]), 1)
        transferred_note = repeated["knowledge_notes"][0]
        self.assertEqual(
            transferred_note["origin"]["source_instance_id"], source["id"]
        )
        self.assertEqual(transferred_note["origin"]["locally_earned"], "false")
        self.assertEqual(
            transferred_note["origin"]["source_origin_kind"], "agent-authored"
        )
        self.assertEqual(transferred_note["learning"], source_note["learning"])
        self.assertEqual(transferred_note["experience"], source_note["experience"])
        self.assertFalse(transferred_note["skill_evidence_eligible"])

        idempotent = self.manager.transfer_private_skills(
            target["id"], source_instance_id=source["id"],
            selections=[{
                "skill_path": "conversion/verified_export",
                "revision": revision,
            }],
            include_knowledge=True, actor="admin",
        )
        self.assertEqual(idempotent["transfer"]["knowledge_notes_copied"], 0)
        self.assertEqual(len(idempotent["knowledge_notes"]), 1)

        with self.assertRaisesRegex(InstanceError, "changed; refresh"):
            self.manager.transfer_private_skills(
                target["id"], source_instance_id=source["id"],
                selections=[{
                    "skill_path": "conversion/verified_export",
                    "revision": "0" * 64,
                }],
                include_knowledge=False, actor="admin",
            )

        deleted = self.manager.delete_private_skill(
            target["id"], category="conversion", skill_name="verified_export",
            expected_revision=copied["revision"],
            confirmation="delete conversion/verified_export", actor="admin",
        )
        self.assertNotIn(
            "conversion/verified_export",
            {item["skill_path"] for item in deleted["skills"]},
        )

    def test_private_skill_delete_requires_exact_confirmation(self):
        instance = self.manager.create_instance(
            name="Delete guard", workspace=str(self.workspace), objective="Maintain skills",
            max_iterations=None, actor="admin",
        )
        category = self.config.instance_state_dir / instance["id"] / "skills" / "review"
        category.mkdir(parents=True, mode=0o700)
        os.chmod(category.parent, 0o700)
        os.chmod(category, 0o700)
        skill = category / "obsolete.txt"
        skill.write_text("Old protocol.\n", encoding="utf-8")
        os.chmod(skill, 0o600)
        revision = self.manager.get_private_skills(instance["id"])["skills"][0]["revision"]

        with self.assertRaisesRegex(InstanceError, "Type 'delete review/obsolete'"):
            self.manager.delete_private_skill(
                instance["id"], category="review", skill_name="obsolete",
                expected_revision=revision, confirmation="yes", actor="admin",
            )
        self.assertTrue(skill.exists())

    def test_launch_is_direct_argv_not_shell_and_uses_exact_tmux_targets(self):
        objective = "Inspect the project; do not delete anything"
        instance = self.manager.create_instance(
            name="Safe agent",
            workspace=str(self.workspace),
            objective=objective,
            max_iterations=25,
            actor="admin",
        )
        launch = next(call for call in self.fake.calls if call[1] == "new-session")
        self.assertNotIn("sh", launch)
        self.assertNotIn("-c", launch[launch.index("/usr/bin/python3") :])
        self.assertIn("/usr/bin/python3", launch)
        self.assertIn("aeon.harnesses.opencode_runtime", launch)
        python_at = launch.index("/usr/bin/python3")
        self.assertEqual(launch[python_at - 1], self.low_priority_wrapper)
        self.assertIn("--model", launch)
        self.assertIn(self.config.default_model, launch)
        self.assertIn(objective, launch)
        self.assertNotIn("nvidia-smi", " ".join(" ".join(call) for call in self.fake.calls))
        target_calls = [call for call in self.fake.calls if "-t" in call]
        self.assertTrue(target_calls)
        self.assertTrue(
            all(call[call.index("-t") + 1].startswith("=") for call in target_calls)
        )
        self.assertEqual(instance["status"], "running")
        self.assertEqual(instance["kind"], "aeon")

    def test_iteration_limit_is_validated_for_the_selected_harness(self):
        before = len(self.store.list_instances())
        with self.assertRaisesRegex(InstanceError, "between 1 and 32 for OpenCode"):
            self.manager.create_instance(
                name="Too many OpenCode steps",
                workspace=str(self.workspace),
                objective="",
                max_iterations=33,
                actor="admin",
                defer_until_message=True,
            )
        self.assertEqual(len(self.store.list_instances()), before)

        legacy = self.manager.create_instance(
            name="Long legacy run",
            workspace=str(self.workspace),
            objective="",
            max_iterations=10_000,
            actor="admin",
            defer_until_message=True,
            harness="legacy-aeon",
        )
        setting = self.store.get_agent_setting(legacy["id"], "aeon")
        self.assertEqual(setting["desired_harness"], "legacy-aeon")
        with self.assertRaisesRegex(InstanceError, "between 1 and 32 for OpenCode"):
            self.manager.update_agent_settings(
                legacy["id"],
                kind="aeon",
                model=self.config.default_model,
                effort="",
                harness="opencode",
                actor="admin",
            )
        unchanged = self.store.get_agent_setting(legacy["id"], "aeon")
        self.assertEqual(unchanged["desired_harness"], "legacy-aeon")

        # Revalidate at launch to protect migrated or externally persisted rows.
        self.store.put_harness_setting(legacy["id"], "opencode")
        calls_before = len(self.fake.calls)
        with self.assertRaisesRegex(InstanceError, "between 1 and 32 for OpenCode"):
            self.manager._launch_record(self.store.get_instance(legacy["id"]))
        self.assertEqual(len(self.fake.calls), calls_before)

    def test_managed_agent_launch_fails_closed_when_priority_wrapper_drifts(self):
        for error in (
            FleetCommandGuardError("unsafe wrapper identity"),
            RuntimeError("wrapper verifier unavailable"),
        ):
            calls_before = len(self.fake.calls)
            with self.subTest(error=type(error).__name__), patch(
                "aeon.remote.instances.require_fleet_low_priority_wrapper",
                side_effect=error,
            ), self.assertRaisesRegex(
                InstanceError, "low-priority agent launcher"
            ):
                self.manager.create_instance(
                    name=f"Blocked agent {type(error).__name__}",
                    workspace=str(self.workspace),
                    objective="",
                    max_iterations=None,
                    actor="admin",
                )
            self.assertFalse(
                any(
                    call[1] == "new-session"
                    for call in self.fake.calls[calls_before:]
                )
            )

    def test_deferred_aeon_waits_for_exact_first_chat_message(self):
        deferred = self.manager.create_instance(
            name="Idle project agent",
            workspace=str(self.workspace),
            objective="",
            max_iterations=None,
            actor="project-manager",
            defer_until_message=True,
        )

        self.assertEqual(deferred["status"], "idle")
        self.assertEqual(deferred["desired_state"], "stopped")
        self.assertTrue(deferred["awaiting_objective"])
        self.assertEqual(deferred["objective"], "")
        self.assertIsNone(deferred["last_started_at"])
        self.assertEqual(self.manager.read_agent_chat(deferred["id"]), [])
        self.assertFalse(any(call[1] == "new-session" for call in self.fake.calls))

        # Resume is not an objective and cannot turn an idle registration into
        # an empty Aeon/model launch.
        resumed = self.manager.resume_instance(deferred["id"], actor="admin")
        self.assertEqual(resumed["status"], "idle")
        self.assertTrue(resumed["awaiting_objective"])
        self.assertFalse(any(call[1] == "new-session" for call in self.fake.calls))

        message_id = "msg-" + "d" * 32
        objective = "First inspect the repository and report what you find."
        first = self.manager.send_agent_chat_message(
            deferred["id"],
            objective,
            actor="admin",
            message_id=message_id,
        )
        launches = [call for call in self.fake.calls if call[1] == "new-session"]
        self.assertEqual(len(launches), 1)
        self.assertIn("--start", launches[0])
        self.assertEqual(launches[0][launches[0].index("--start") + 1], objective)
        self.assertEqual(launches[0].count(objective), 1)
        self.assertNotIn(objective, self.fake.loaded_payloads)
        self.assertEqual(first["id"], message_id)
        self.assertEqual(self.manager.read_agent_chat(deferred["id"]), [first])
        running = self.manager.get_instance(deferred["id"])
        self.assertFalse(running["awaiting_objective"])
        self.assertEqual(running["status"], "running")

        deliveries_before_retry = len(self.fake.loaded_payloads)
        retry = self.manager.send_agent_chat_message(
            deferred["id"],
            objective,
            actor="admin",
            message_id=message_id,
        )
        self.assertEqual(retry, first)
        self.assertEqual(
            len([call for call in self.fake.calls if call[1] == "new-session"]),
            1,
        )
        self.assertEqual(len(self.fake.loaded_payloads), deliveries_before_retry)

        # Model a lost success response after tmux launch but before the durable
        # awaiting flag was cleared. Retrying recovers state without a new launch
        # or a second PTY delivery.
        self.store.update_instance(
            deferred["id"],
            awaiting_objective=1,
        )
        recovered = self.manager.send_agent_chat_message(
            deferred["id"],
            objective,
            actor="admin",
            message_id=message_id,
        )
        self.assertEqual(recovered, first)
        self.assertFalse(
            self.store.get_instance(deferred["id"])["awaiting_objective"]
        )
        self.assertEqual(
            len([call for call in self.fake.calls if call[1] == "new-session"]),
            1,
        )
        self.assertEqual(len(self.fake.loaded_payloads), deliveries_before_retry)

        second = self.manager.send_agent_chat_message(
            deferred["id"],
            "Now summarize the test suite.",
            actor="admin",
        )
        self.assertEqual(second["content"], "Now summarize the test suite.")
        self.assertEqual(
            self.fake.loaded_payloads[-1],
            "\x1b[200~"
            + build_chat_delivery_envelope(second["id"], second["content"])
            + "\x1b[201~\r",
        )

    def test_deferred_creation_requires_an_empty_aeon_objective(self):
        with self.assertRaisesRegex(InstanceError, "without an objective"):
            self.manager.create_instance(
                name="Premature Aeon",
                workspace=str(self.workspace),
                objective="Start doing work",
                max_iterations=None,
                actor="admin",
                defer_until_message=True,
            )
        with self.assertRaisesRegex(InstanceError, "Only Aeon"):
            self.manager.create_instance(
                kind="codex",
                name="Deferred Codex",
                workspace=str(self.workspace),
                objective="",
                max_iterations=None,
                actor="admin",
                defer_until_message=True,
            )

    def test_chat_fork_clones_exact_prefix_and_restores_independent_state(self):
        source = self.manager.create_instance(
            name="Main investigation",
            workspace=str(self.workspace),
            objective="",
            max_iterations=32,
            actor="admin",
            defer_until_message=True,
        )
        user = self.manager.send_agent_chat_message(
            source["id"],
            "Inspect the implementation.",
            actor="admin",
            message_id="msg-" + "a" * 32,
        )
        transcript = self.manager._agent_chat_path(source["id"])
        plan = append_chat_message(
            transcript,
            role="plan",
            content="- [x] Inspect\n- [ ] Explain",
            message_id="msg-" + "b" * 32,
        )
        assistant = append_chat_message(
            transcript,
            role="assistant",
            content="The implementation has two relevant layers.",
            message_id="msg-" + "c" * 32,
            performance={
                "tokens_per_second": 101.2,
                "decode_tokens_per_second": 101.2,
                "end_to_end_tokens_per_second": 35.4,
                "completion_tokens": 24,
                "time_to_first_token_seconds": 0.42,
                "served_model": "Qwen3-Coder-Next-FP8",
            },
        )
        worker_state_root = Path(self.temp.name) / "worker-state"
        environment = {"AEON_STATE_DIR": str(worker_state_root)}
        with patch.dict(os.environ, environment, clear=False):
            checkpoint_directory = (
                self.manager._worker_session_directory(source) / "fork-checkpoints"
            )
            checkpoint_directory.mkdir(parents=True)
            checkpoint_directory.chmod(0o700)
            checkpoint = checkpoint_directory / f"{assistant['id']}.json.gz"
            import gzip

            with gzip.open(checkpoint, "wt", encoding="utf-8") as stream:
                json.dump(
                    {
                        "fork_checkpoint_schema": 1,
                        "fork_checkpoint_message_id": assistant["id"],
                        "memories": {"shared-fact": "retained"},
                        "action_log": ["inspected exact source"],
                        "history_messages": [
                            {"role": "user", "content": user["content"]},
                            {"role": "assistant", "content": assistant["content"]},
                        ],
                        "current_plan": plan["content"],
                    },
                    stream,
                )
            checkpoint.chmod(0o600)

            fork = self.manager.fork_agent_chat(
                source["id"], assistant["id"], actor="admin"
            )

            self.assertTrue(fork["temporary_fork"])
            self.assertEqual(fork["fork_parent_id"], source["id"])
            self.assertEqual(fork["fork_root_id"], source["id"])
            self.assertEqual(fork["fork_point_message_id"], assistant["id"])
            self.assertEqual(fork["fork_state_quality"], "checkpoint")
            self.assertTrue(fork["awaiting_objective"])
            self.assertEqual(
                [item["id"] for item in self.manager.read_agent_chat(fork["id"])],
                [user["id"], plan["id"], assistant["id"]],
            )
            self.assertEqual(
                self.manager.read_agent_chat(fork["id"])[-1]["performance"],
                assistant["performance"],
            )
            fork_state_path = (
                self.manager._worker_session_directory(fork) / "session_state.json"
            )
            fork_state = json.loads(fork_state_path.read_text(encoding="utf-8"))
            self.assertEqual(fork_state["memories"], {"shared-fact": "retained"})
            self.assertEqual(
                fork_state["fork_restore"]["source_instance_id"], source["id"]
            )

            branch_message = self.manager.send_agent_chat_message(
                fork["id"], "Explain only the second layer.", actor="admin"
            )
            self.assertEqual(branch_message["content"], "Explain only the second layer.")
            self.assertEqual(
                self.manager.read_agent_chat(source["id"]), [user, plan, assistant]
            )
            self.manager.close_agent_chat_fork(fork["id"], actor="admin")
            self.assertIsNone(self.store.get_instance(fork["id"]))
            self.assertIsNotNone(self.store.get_instance(source["id"]))
    def test_deferred_first_message_replaces_only_a_dead_same_tab_session(self):
        deferred = self.manager.create_instance(
            name="Dead-session retry",
            workspace=str(self.workspace),
            objective="",
            max_iterations=None,
            actor="project-manager",
            defer_until_message=True,
        )
        tmux_name = self.store.get_instance(deferred["id"])["tmux_name"]
        self.fake.sessions[tmux_name] = {
            "dead": True,
            "pid": 987000,
            "exit": 1,
            "cwd": str(self.workspace),
            "command": "python3",
            "pending": "",
            "agent_mode": False,
            "managed_agent": False,
            "at_prompt": False,
            "interrupt_returns_prompt": True,
            "delayed_browser_input": "",
        }

        saved = self.manager.send_agent_chat_message(
            deferred["id"],
            "Inspect the failure and report only.",
            actor="admin",
            message_id="msg-" + "e" * 32,
        )

        self.assertEqual(saved["content"], "Inspect the failure and report only.")
        self.assertFalse(self.fake.sessions[tmux_name]["dead"])
        self.assertFalse(self.manager.get_instance(deferred["id"])["awaiting_objective"])
        self.assertEqual(
            len([call for call in self.fake.calls if call[1] == "new-session"]),
            1,
        )

    def test_legacy_direct_aeon_uses_clean_environment_without_service_secrets(self):
        sentinels = {
            "NEXUS_OIDC_CLIENT_SECRET": "oidc-secret-sentinel",
            "CLOUDFLARE_API_TOKEN": "cloudflare-secret-sentinel",
            "OPENAI_API_KEY": "openai-secret-sentinel",
            "AEON_FLEET_SOCKET": "/tmp/forged-fleet-socket",
            "FLEET_TICKET": "forged-fleet-ticket",
            "GPU_AGENT_CLAIM_ID": "forged-gpu-claim",
            "GPU_MEM_LIMIT_GB": "forged-gpu-memory-limit",
            "CUDA_VISIBLE_DEVICES": "forged-cuda-selector",
            "DOCKER_HOST": "unix:///tmp/forged-docker.sock",
        }
        with patch.dict(os.environ, sentinels, clear=False):
            self.manager.create_instance(
                name="Clean legacy Aeon",
                workspace=str(self.workspace),
                objective="",
                max_iterations=None,
                actor="fixture",
            )
        launch = [call for call in self.fake.calls if call[1] == "new-session"][-1]
        env_at = launch.index("/usr/bin/env")
        python_at = launch.index("/usr/bin/python3")
        self.assertEqual(launch[env_at : env_at + 2], ["/usr/bin/env", "-i"])
        self.assertLess(env_at, python_at)
        self.assertEqual(launch[python_at - 1], self.low_priority_wrapper)
        self.assertIn("CUDA_VISIBLE_DEVICES=void", launch)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", launch)
        rendered = "\x00".join(launch)
        for secret in sentinels.values():
            self.assertNotIn(secret, rendered)

    def test_terminal_launch_is_fixed_unlogged_shell_and_resume_is_type_aware(self):
        terminal = self.manager.create_terminal(
            name="Managed shell",
            workspace=str(self.workspace),
            actor="admin",
        )
        self.assertEqual(terminal["kind"], "terminal")
        self.assertEqual(terminal["objective"], "")
        self.assertIsNone(terminal["model"])
        self.assertEqual(terminal["current_directory"], str(self.workspace.resolve()))
        launches = [call for call in self.fake.calls if call[1] == "new-session"]
        bash_at = launches[-1].index("/bin/bash")
        self.assertEqual(
            launches[-1][bash_at : bash_at + 3],
            ["/bin/bash", "--noprofile", "--rcfile"],
        )
        self.assertNotIn(self.low_priority_wrapper, launches[-1])
        self.assertTrue(launches[-1][bash_at + 3].endswith("/managed-shell.rc"))
        self.assertEqual(launches[-1][bash_at + 4], "-i")
        environment_args = launches[-1][launches[-1].index("-i") + 1 : bash_at]
        self.assertIn("TERM=tmux-256color", environment_args)
        self.assertIn("COLORTERM=truecolor", environment_args)
        rc_text = Path(launches[-1][bash_at + 3]).read_text(encoding="utf-8")
        self.assertIn("/usr/bin/dircolors --sh", rc_text)
        self.assertIn("alias ls='/usr/bin/ls --color=auto'", rc_text)
        self.assertIn("alias ll='/usr/bin/ls -alF --color=auto'", rc_text)
        self.assertIn("alias la='/usr/bin/ls -A --color=auto'", rc_text)
        self.assertNotIn(".bashrc", rc_text)
        self.assertFalse(any(call[1] == "pipe-pane" for call in self.fake.calls))
        self.assertFalse(
            any(
                item.startswith(("PYTHONPATH=", "AEON_REMOTE_INSTANCE_ID="))
                for item in launches[-1]
            )
        )
        audit = self.store.recent_audit(1)[0]
        self.assertEqual(audit["action"], "terminal_created")
        self.assertEqual(
            set(json.loads(audit["details_json"])), {"host_id", "name", "workspace"}
        )

        self.fake.sessions.clear()
        calls_before = len(self.fake.calls)
        resumed = self.manager.resume_instance(terminal["id"], actor="admin")
        self.assertEqual(resumed["kind"], "terminal")
        resume_calls = self.fake.calls[calls_before:]
        resumed_launch = next(call for call in resume_calls if call[1] == "new-session")
        bash_at = resumed_launch.index("/bin/bash")
        self.assertEqual(
            resumed_launch[bash_at : bash_at + 3],
            ["/bin/bash", "--noprofile", "--rcfile"],
        )
        self.assertTrue(resumed_launch[bash_at + 3].endswith("/managed-shell.rc"))
        self.assertEqual(resumed_launch[bash_at + 4], "-i")
        self.assertNotIn("aeon.main", resumed_launch)
        self.assertFalse(any(call[1] == "pipe-pane" for call in resume_calls))

    def test_terminal_names_fill_delete_gaps_and_concurrent_creates_do_not_collide(self):
        first = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        second = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        self.assertEqual((first["name"], second["name"]), ("Terminal 1", "Terminal 2"))
        self.manager.force_stop(
            first["id"], confirmation=first["name"], actor="admin"
        )
        self.manager.delete_instance(
            first["id"], confirmation=first["name"], actor="admin"
        )
        replacement = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        self.assertEqual(replacement["name"], "Terminal 1")

        names: list[str] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(3)

        def create():
            barrier.wait()
            try:
                value = self.manager.create_terminal(
                    workspace=str(self.workspace), actor="admin"
                )
                names.append(value["name"])
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        threads = [threading.Thread(target=create) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(set(names)), 2)
        self.assertEqual(set(names), {"Terminal 3", "Terminal 4"})

    def test_start_aeon_here_uses_exact_live_terminal_cwd_and_keeps_shell(self):
        terminal = self.manager.create_terminal(
            name="Workspace shell",
            workspace=str(self.workspace),
            actor="admin",
        )
        nested = self.workspace / "nested-project"
        nested.mkdir()
        terminal_record = self.store.get_instance(terminal["id"])
        tmux_name = terminal_record["tmux_name"]
        self.fake.sessions[tmux_name]["cwd"] = str(nested)

        calls_before = len(self.fake.calls)
        aeon = self.manager.start_aeon_here(terminal["id"], actor="admin")
        self.assertEqual(aeon["id"], terminal["id"])
        self.assertEqual(aeon["kind"], "aeon")
        self.assertEqual(aeon["mode"], "agent")
        self.assertTrue(aeon["shell_backed"])
        self.assertEqual(aeon["workspace"], str(nested.resolve()))
        self.assertEqual(aeon["objective"], "")
        self.assertEqual(aeon["name"], "Workspace shell")
        self.assertIn(tmux_name, self.fake.sessions)
        self.assertFalse(self.fake.sessions[tmux_name]["dead"])
        display_calls = [call for call in self.fake.calls if call[1] == "display-message"]
        self.assertTrue(display_calls)
        self.assertTrue(
            all(
                call[call.index("-t") + 1] == f"={tmux_name}:"
                for call in display_calls
                if tmux_name in call[call.index("-t") + 1]
            )
        )
        self.assertEqual(display_calls[-1][-1], "#{pane_current_path}")
        activation_calls = self.fake.calls[calls_before:]
        self.assertFalse(any(call[1] == "new-session" for call in activation_calls))
        command = self.fake.loaded_payloads[-1]
        self.assertIn("aeon.harnesses.opencode_runtime", command)
        self.assertIn(
            f"{self.low_priority_wrapper} /usr/bin/python3 "
            "-m aeon.harnesses.opencode_runtime",
            command,
        )
        self.assertIn(
            "AEON_INSTANCE_SKILLS_DIR="
            f"{self.config.instance_state_dir / terminal['id'] / 'skills'}",
            command,
        )
        self.assertNotIn("--start", command)
        self.assertTrue(command.endswith("\r"))
        self.assertFalse(
            any(call[1] == "send-keys" and "-l" in call for call in activation_calls)
        )
        audit = self.store.recent_audit(1)[0]
        self.assertEqual(audit["action"], "terminal_agent_started")
        self.assertEqual(
            set(json.loads(audit["details_json"])),
            {"kind", "workspace"},
        )
        returned = self.manager.end_agent(terminal["id"], actor="admin")
        self.assertEqual(returned["id"], terminal["id"])
        self.assertEqual(returned["kind"], "terminal")
        self.assertEqual(returned["mode"], "terminal")

    def test_same_tab_provider_activation_is_fixed_clean_and_reversible(self):
        terminal = self.manager.create_terminal(
            name="Provider shell",
            workspace=str(self.workspace),
            actor="admin",
        )
        record = self.store.get_instance(terminal["id"])
        nested = self.workspace / "provider-project"
        nested.mkdir()
        self.fake.sessions[record["tmux_name"]]["cwd"] = str(nested)
        secret = "must-not-enter-provider-environment"

        with (
            patch(
                "aeon.remote.instances.provider_status",
                return_value={"installed": True, "connected": True},
            ),
            patch(
                "aeon.remote.instances.provider_agent_command",
                return_value=SimpleNamespace(argv=("/safe/codex", "--no-alt-screen")),
            ),
            patch(
                "aeon.remote.instances.subscription_environment",
                return_value={"HOME": "/home/aday", "PATH": "/safe/bin"},
            ),
        ):
            agent = self.manager.activate_agent(
                terminal["id"], kind="codex", actor="admin"
            )

        self.assertEqual(agent["id"], terminal["id"])
        self.assertEqual(agent["kind"], "codex")
        self.assertEqual(agent["mode"], "agent")
        self.assertEqual(agent["workspace"], str(nested.resolve()))
        command = self.fake.loaded_payloads[-1]
        self.assertIn("/usr/bin/env -i", command)
        self.assertIn(
            f"{self.low_priority_wrapper} /safe/codex --no-alt-screen", command
        )
        self.assertIn("HOME=/home/aday", command)
        self.assertNotIn(secret, command)
        self.assertFalse(any(call[1] == "new-session" for call in self.fake.calls[-6:]))

        returned = self.manager.end_agent(terminal["id"], actor="admin")
        self.assertEqual(returned["id"], terminal["id"])
        self.assertEqual(returned["kind"], "terminal")
        self.assertEqual(returned["last_agent_kind"], "codex")

    def test_failed_cross_kind_activation_preserves_profile_and_last_agent_kind(self):
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
        codex_home = Path(self.temp.name) / "cross-kind-codex-home"
        codex_home.mkdir(mode=0o700)
        environment = {
            "CODEX_HOME": str(codex_home),
            "HOME": str(Path(self.temp.name)),
            "PATH": "/safe/bin",
        }
        terminal = manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        with (
            patch(
                "aeon.remote.instances.provider_status",
                return_value={"installed": True, "connected": True},
            ),
            patch(
                "aeon.remote.instances.provider_agent_command",
                return_value=SimpleNamespace(argv=("/safe/codex",)),
            ),
            patch(
                "aeon.remote.instances.subscription_environment",
                return_value=environment,
            ),
        ):
            manager.activate_agent(
                terminal["id"], kind="codex", actor="admin"
            )

        profile = instruction_service.create_profile(
            agent_kind="codex", name="Persistent Codex role", actor="admin"
        )
        version = instruction_service.save_version(
            profile["id"],
            label="v1",
            content="Review the current project carefully.",
            actor="admin",
        )
        instruction_service.select_profile_version(terminal["id"], version["id"])
        manager.end_agent(terminal["id"], actor="admin")
        before_record = self.store.get_instance(terminal["id"])
        with self.store._connect() as conn:
            before_binding = dict(
                conn.execute(
                    "SELECT * FROM instance_instruction_bindings WHERE instance_id=?",
                    (terminal["id"],),
                ).fetchone()
            )

        with patch(
            "aeon.remote.instances.provider_status",
            return_value={"installed": True, "connected": False},
        ):
            with self.assertRaisesRegex(InstanceError, "Connect claude"):
                manager.activate_agent(
                    terminal["id"], kind="claude", actor="admin"
                )

        after_record = self.store.get_instance(terminal["id"])
        with self.store._connect() as conn:
            after_binding = dict(
                conn.execute(
                    "SELECT * FROM instance_instruction_bindings WHERE instance_id=?",
                    (terminal["id"],),
                ).fetchone()
            )
        self.assertEqual(after_record["kind"], "terminal")
        self.assertEqual(after_record["last_agent_kind"], "codex")
        self.assertEqual(before_record["last_agent_kind"], "codex")
        self.assertEqual(after_binding, before_binding)
        self.assertEqual(
            after_binding["desired_profile_version_id"], version["id"]
        )

    def test_cross_kind_activation_fallback_retains_journal_until_profile_repair(self):
        manager, terminal, old_binding = self._cross_kind_profile_fixture()
        codex_home = Path(self.temp.name) / "fallback-codex-home"
        codex_home.mkdir(mode=0o700)
        manager._pane_foreground_checker = lambda _record, _pane: False

        with (
            patch(
                "aeon.remote.instances.provider_status",
                return_value={"installed": True, "connected": True},
            ),
            patch(
                "aeon.remote.instances.provider_agent_command",
                return_value=SimpleNamespace(argv=("/safe/codex",)),
            ),
            patch(
                "aeon.remote.instances.subscription_environment",
                return_value={
                    "CODEX_HOME": str(codex_home),
                    "HOME": str(Path(self.temp.name)),
                    "PATH": "/safe/bin",
                },
            ),
            patch.object(
                self.store,
                "transition_shell_mode",
                side_effect=RuntimeError("simulated transition failure"),
            ),
            patch("aeon.remote.instances.AGENT_START_TIMEOUT_SECONDS", 0.03),
        ):
            with self.assertRaises(InstanceLaunchError):
                manager.activate_agent(
                    terminal["id"], kind="codex", actor="admin"
                )

        pending = (
            self.config.instance_state_dir
            / terminal["id"]
            / "managed-agent.pending.json"
        )
        self.assertTrue(pending.is_file())
        self.assertEqual(self.store.get_instance(terminal["id"])["kind"], "codex")
        with self.store._connect() as conn:
            stranded = dict(
                conn.execute(
                    "SELECT * FROM instance_instruction_bindings WHERE instance_id=?",
                    (terminal["id"],),
                ).fetchone()
            )
        self.assertEqual(
            stranded["desired_profile_version_id"],
            old_binding["desired_profile_version_id"],
        )
        self.assertEqual(
            stranded["applied_profile_version_id"],
            old_binding["applied_profile_version_id"],
        )

        manager._pane_foreground_checker = self.fake.pane_has_managed_foreground
        repaired = manager.get_instance(terminal["id"])
        with self.store._connect() as conn:
            binding = dict(
                conn.execute(
                    "SELECT * FROM instance_instruction_bindings WHERE instance_id=?",
                    (terminal["id"],),
                ).fetchone()
            )
        self.assertEqual(repaired["kind"], "codex")
        self.assertTrue(repaired["force_stop_required"])
        self.assertIsNone(binding["desired_profile_version_id"])
        self.assertIsNone(binding["applied_profile_version_id"])
        self.assertEqual(
            binding["desired_local_revision"], old_binding["desired_local_revision"]
        )
        self.assertEqual(
            binding["applied_local_revision"], old_binding["applied_local_revision"]
        )
        self.assertFalse(pending.exists())

    def test_recovery_fallback_keeps_journal_for_next_atomic_profile_repair(self):
        manager, terminal, old_binding = self._cross_kind_profile_fixture()
        record = self.store.get_instance(terminal["id"])
        manager._write_pending_activation(
            record,
            target_kind="codex",
            workspace=str(self.workspace),
            previous_agent_kind="aeon",
            phase="command_sent",
        )
        self.fake.sessions[record["tmux_name"]].update(
            command="node",
            at_prompt=False,
            agent_mode=True,
            managed_agent=True,
        )
        pending = (
            self.config.instance_state_dir
            / terminal["id"]
            / "managed-agent.pending.json"
        )

        with patch.object(
            self.store,
            "transition_shell_mode",
            side_effect=RuntimeError("simulated recovery transaction failure"),
        ):
            first = manager.get_instance(terminal["id"])

        self.assertEqual(first["kind"], "codex")
        self.assertTrue(first["force_stop_required"])
        self.assertTrue(pending.is_file())
        with self.store._connect() as conn:
            stranded = dict(
                conn.execute(
                    "SELECT * FROM instance_instruction_bindings WHERE instance_id=?",
                    (terminal["id"],),
                ).fetchone()
            )
        self.assertEqual(
            stranded["desired_profile_version_id"],
            old_binding["desired_profile_version_id"],
        )

        repaired = manager.get_instance(terminal["id"])
        with self.store._connect() as conn:
            binding = dict(
                conn.execute(
                    "SELECT * FROM instance_instruction_bindings WHERE instance_id=?",
                    (terminal["id"],),
                ).fetchone()
            )
        self.assertEqual(repaired["kind"], "codex")
        self.assertTrue(repaired["force_stop_required"])
        self.assertIsNone(binding["desired_profile_version_id"])
        self.assertIsNone(binding["applied_profile_version_id"])
        self.assertEqual(
            binding["desired_local_revision"], old_binding["desired_local_revision"]
        )
        self.assertFalse(pending.exists())

    def test_disconnected_subscription_cannot_change_terminal_mode(self):
        terminal = self.manager.create_terminal(
            name="Disconnected provider shell",
            workspace=str(self.workspace),
            actor="admin",
        )
        with patch(
            "aeon.remote.instances.provider_status",
            return_value={"installed": True, "connected": False},
        ):
            with self.assertRaisesRegex(InstanceError, "Connect codex"):
                self.manager.activate_agent(
                    terminal["id"], kind="codex", actor="admin"
                )

        unchanged = self.manager.get_instance(terminal["id"])
        self.assertEqual(unchanged["kind"], "terminal")
        self.assertEqual(unchanged["mode"], "terminal")
        self.assertEqual(unchanged["status"], "running")

    def test_disconnected_direct_provider_is_rejected_before_registry_mutation(self):
        before = self.store.list_instances()
        calls_before = len(self.fake.calls)
        with patch(
            "aeon.remote.instances.provider_status",
            return_value={"installed": True, "connected": False},
        ):
            with self.assertRaisesRegex(InstanceError, "Connect codex"):
                self.manager.create_instance(
                    kind="codex",
                    name="Unconnected direct Codex",
                    workspace=str(self.workspace),
                    objective="",
                    max_iterations=None,
                    actor="project-manager",
                )

        self.assertEqual(self.store.list_instances(), before)
        self.assertEqual(len(self.fake.calls), calls_before)

    def test_legacy_direct_provider_resume_requires_fresh_connection(self):
        with (
            patch(
                "aeon.remote.instances.provider_status",
                return_value={"installed": True, "connected": True},
            ),
            patch(
                "aeon.remote.instances.provider_agent_command",
                return_value=SimpleNamespace(argv=("/safe/codex",)),
            ),
            patch(
                "aeon.remote.instances.subscription_environment",
                return_value={"HOME": "/home/aday", "PATH": "/safe/bin"},
            ),
        ):
            legacy = self.manager.create_instance(
                kind="codex",
                name="Legacy direct Codex",
                workspace=str(self.workspace),
                objective="",
                max_iterations=None,
                actor="fixture",
            )
        self.manager.force_stop(
            legacy["id"], confirmation=legacy["name"], actor="admin"
        )
        calls_before = len(self.fake.calls)

        with patch(
            "aeon.remote.instances.provider_status",
            return_value={"installed": True, "connected": False},
        ):
            with self.assertRaisesRegex(InstanceError, "Connect codex"):
                self.manager.resume_instance(legacy["id"], actor="admin")

        self.assertFalse(
            any(call[1] == "new-session" for call in self.fake.calls[calls_before:])
        )

    def test_slow_provider_probe_cannot_race_reconcile_back_to_terminal(self):
        terminal = self.manager.create_terminal(
            name="Serialized provider shell",
            workspace=str(self.workspace),
            actor="admin",
        )
        probe_entered = threading.Event()
        release_probe = threading.Event()
        reconcile_finished = threading.Event()
        results: dict[str, object] = {}

        def slow_status(_kind):
            probe_entered.set()
            if not release_probe.wait(timeout=2):
                raise AssertionError("test did not release provider probe")
            return {"installed": True, "connected": True}

        def activate():
            try:
                results["agent"] = self.manager.activate_agent(
                    terminal["id"], kind="codex", actor="admin"
                )
            except Exception as exc:  # pragma: no cover - asserted below
                results["activation_error"] = exc

        def reconcile():
            try:
                record = self.store.get_instance(terminal["id"])
                results["reconciled"] = self.manager.reconcile(record)
            except Exception as exc:  # pragma: no cover - asserted below
                results["reconcile_error"] = exc
            finally:
                reconcile_finished.set()

        with (
            patch("aeon.remote.instances.provider_status", side_effect=slow_status),
            patch(
                "aeon.remote.instances.provider_agent_command",
                return_value=SimpleNamespace(argv=("/safe/codex", "--no-alt-screen")),
            ),
            patch(
                "aeon.remote.instances.subscription_environment",
                return_value={"HOME": "/home/aday", "PATH": "/safe/bin"},
            ),
        ):
            activation_thread = threading.Thread(target=activate)
            activation_thread.start()
            self.assertTrue(probe_entered.wait(timeout=1))
            reconcile_thread = threading.Thread(target=reconcile)
            reconcile_thread.start()
            self.assertFalse(reconcile_finished.wait(timeout=0.05))
            release_probe.set()
            activation_thread.join(timeout=2)
            reconcile_thread.join(timeout=2)

        self.assertFalse(activation_thread.is_alive())
        self.assertFalse(reconcile_thread.is_alive())
        self.assertNotIn("activation_error", results)
        self.assertNotIn("reconcile_error", results)
        self.assertEqual(results["agent"]["kind"], "codex")
        self.assertEqual(results["reconciled"]["kind"], "codex")
        self.assertEqual(self.store.get_instance(terminal["id"])["kind"], "codex")

    def test_nested_bash_is_not_the_managed_base_prompt(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        record = self.store.get_instance(terminal["id"])
        pane = self.fake.sessions[record["tmux_name"]]
        pane.update(
            command="bash",
            at_prompt=False,
            agent_mode=False,
            managed_agent=False,
        )
        calls_before = len(self.fake.calls)

        with self.assertRaisesRegex(InstanceError, "shell prompt"):
            self.manager.activate_agent(
                terminal["id"], kind="aeon", actor="admin"
            )

        self.assertEqual(self.store.get_instance(terminal["id"])["kind"], "terminal")
        transition_calls = self.fake.calls[calls_before:]
        self.assertFalse(
            any(call[1] == "send-keys" for call in transition_calls),
            transition_calls,
        )

    def test_foreground_probe_validates_base_pid_even_when_tmux_reports_child(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        record = self.store.get_instance(terminal["id"])
        self.manager._pane_foreground_checker = None
        shell = {
            "pid": 100,
            "pgrp": 100,
            "session": 90,
            "tty": 1234,
            "tpgid": 200,
            "start_ticks": 10,
            "nonce": "a" * 64,
        }
        leader = {
            "pid": 200,
            "pgrp": 200,
            "session": 90,
            "tty": 1234,
            "tpgid": 200,
            "start_ticks": 20,
        }
        with (
            patch.object(self.manager, "_base_shell_process", return_value=shell),
            patch("aeon.remote.instances._proc_process_info", return_value=leader),
        ):
            foreground = self.manager._pane_foreground_job(
                record,
                {"dead": False, "pid": 100, "command": "python3"},
            )
        self.assertEqual(foreground["pid"], 200)
        self.assertEqual(foreground["shell_pid"], 100)

    def test_private_bash_hook_proves_prompt_and_rejects_nested_bash(self):
        """Exercise the marker and Linux job-control probe without real tmux."""

        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        record = self.store.get_instance(terminal["id"])
        launch = [call for call in self.fake.calls if call[1] == "new-session"][-1]
        bash_at = launch.index("/bin/bash")
        argv = launch[bash_at : bash_at + 5]
        child_pid, fd = pty.fork()
        if child_pid == 0:  # pragma: no cover - child replaces the test process
            os.execve(
                "/bin/bash",
                argv,
                {
                    "HOME": str(Path(self.temp.name)),
                    "PATH": "/usr/bin:/bin",
                    "SHELL": "/bin/bash",
                    "TERM": "xterm-256color",
                },
            )

        self.manager._pane_prompt_checker = None
        self.manager._pane_foreground_checker = None
        pane = {"dead": False, "pid": child_pid, "command": "bash"}
        try:
            deadline = time.monotonic() + 2
            while (
                time.monotonic() < deadline
                and not self.manager._pane_at_base_prompt(record, pane)
            ):
                time.sleep(0.02)
            self.assertTrue(self.manager._pane_at_base_prompt(record, pane))

            os.write(fd, b"bash -c 'sleep 2'\n")
            deadline = time.monotonic() + 2
            foreground = None
            while time.monotonic() < deadline:
                foreground = self.manager._pane_foreground_job(record, pane)
                if foreground is not None:
                    break
                time.sleep(0.02)
            self.assertIsNotNone(foreground)
            self.assertFalse(self.manager._pane_at_base_prompt(record, pane))
            self.assertTrue(
                self.manager._record_managed_agent_foreground(record, pane)
            )
            self.assertTrue(self.manager._managed_agent_is_foreground(record, pane))

            os.write(fd, b"\x03")
            deadline = time.monotonic() + 2
            while (
                time.monotonic() < deadline
                and not self.manager._pane_at_base_prompt(record, pane)
            ):
                time.sleep(0.02)
            self.assertTrue(self.manager._pane_at_base_prompt(record, pane))
            self.assertFalse(self.manager._managed_agent_is_foreground(record, pane))
        finally:
            try:
                os.write(fd, b"exit\n")
            except OSError:
                pass
            deadline = time.monotonic() + 0.5
            waited = 0
            try:
                while time.monotonic() < deadline:
                    waited, _status = os.waitpid(child_pid, os.WNOHANG)
                    if waited:
                        break
                    time.sleep(0.02)
            except (ChildProcessError, ProcessLookupError):
                waited = child_pid
            try:
                os.close(fd)
            except OSError:
                pass
            if not waited:
                try:
                    os.kill(child_pid, signal.SIGHUP)
                    os.waitpid(child_pid, 0)
                except (ChildProcessError, ProcessLookupError):
                    pass

    def test_agent_exit_followed_by_unrelated_job_is_never_signaled_or_relabeled(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        agent = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )
        record = self.store.get_instance(agent["id"])
        pane = self.fake.sessions[record["tmux_name"]]
        # The managed Aeon foreground exited, then a browser/user immediately
        # started an unrelated job before reconciliation observed the prompt.
        pane.update(
            command="python3",
            at_prompt=False,
            agent_mode=False,
            managed_agent=False,
        )
        reconciled = self.manager.get_instance(agent["id"])
        self.assertEqual(reconciled["kind"], "aeon")
        calls_before = len(self.fake.calls)

        with self.assertRaisesRegex(InstanceError, "not the managed agent"):
            self.manager.end_agent(agent["id"], actor="admin")

        end_calls = self.fake.calls[calls_before:]
        self.assertFalse(any(call[1] == "send-keys" for call in end_calls))
        self.assertEqual(self.store.get_instance(agent["id"])["kind"], "aeon")
        actionable = self.manager.get_instance(agent["id"])
        self.assertEqual(actionable["status"], "error")
        self.assertTrue(actionable["force_stop_required"])

    def test_unresponsive_agent_end_sends_only_control_signals_and_preserves_mode(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        agent = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )
        record = self.store.get_instance(agent["id"])
        self.fake.sessions[record["tmux_name"]]["interrupt_returns_prompt"] = False
        calls_before = len(self.fake.calls)

        with (
            patch("aeon.remote.instances.AGENT_END_TIMEOUT_SECONDS", 0.03),
            patch("aeon.remote.instances.AGENT_SECOND_INTERRUPT_DELAY_SECONDS", 1.0),
        ):
            result = self.manager.end_agent(agent["id"], actor="admin")

        self.assertEqual(result["kind"], "aeon")
        self.assertEqual(result["status"], "running")
        sent = [
            call for call in self.fake.calls[calls_before:] if call[1] == "send-keys"
        ]
        self.assertTrue(sent)
        self.assertTrue(all("-l" not in call for call in sent))
        self.assertFalse(any("exit" in call for call in sent))

    def test_agent_end_allows_verified_shell_handoff_to_finish_prompt(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        agent = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )

        with (
            patch.object(
                self.manager,
                "_pane_at_base_prompt",
                side_effect=[False, False, True],
            ),
            patch.object(
                self.manager,
                "_managed_agent_is_foreground",
                side_effect=[True, False],
            ),
            patch.object(
                self.manager, "_base_shell_has_foreground_control", return_value=True
            ),
        ):
            returned = self.manager.end_agent(agent["id"], actor="admin")

        self.assertEqual(returned["kind"], "terminal")
        self.assertEqual(returned["status"], "running")
        self.assertFalse(returned["force_stop_required"])

    def test_exact_prompt_repairs_stale_force_stop_error(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        agent = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )
        record = self.store.get_instance(agent["id"])
        self.fake.sessions[record["tmux_name"]].update(
            command="bash",
            at_prompt=True,
            agent_mode=False,
            managed_agent=False,
        )
        self.store.update_instance(
            agent["id"],
            status="error",
            last_error=(
                "Safety ambiguity; exact-name force stop required: stale handoff"
            ),
        )

        recovered = self.manager.get_instance(agent["id"])

        self.assertEqual(recovered["kind"], "terminal")
        self.assertEqual(recovered["status"], "running")
        self.assertFalse(recovered["force_stop_required"])

    def test_live_unrecordable_launch_never_falls_back_to_terminal_mode(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        self.manager._pane_foreground_checker = lambda _record, _pane: False

        with patch("aeon.remote.instances.AGENT_START_TIMEOUT_SECONDS", 0.03):
            with self.assertRaises(InstanceLaunchError) as caught:
                self.manager.activate_agent(
                    terminal["id"], kind="aeon", actor="admin"
                )

        self.assertTrue(caught.exception.launched)
        record = self.store.get_instance(terminal["id"])
        self.assertEqual(record["kind"], "aeon")
        self.assertEqual(record["status"], "error")
        self.assertIn("force stop", record["last_error"])
        public = self.manager.get_instance(terminal["id"])
        self.assertEqual(public["status"], "error")
        self.assertTrue(public["force_stop_required"])

    def test_malformed_activation_journal_remains_actionable_across_reconcile(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        pending = (
            self.config.instance_state_dir
            / terminal["id"]
            / "managed-agent.pending.json"
        )
        pending.write_text('{"target_kind":"aeon"', encoding="utf-8")
        pending.chmod(0o600)

        first = self.manager.get_instance(terminal["id"])
        second = self.manager.get_instance(terminal["id"])

        self.assertEqual(first["status"], "error")
        self.assertTrue(first["force_stop_required"])
        self.assertEqual(second["status"], "error")
        self.assertTrue(second["force_stop_required"])

    def test_missing_activation_journal_with_live_foreground_requires_force(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        record = self.store.get_instance(terminal["id"])
        self.fake.sessions[record["tmux_name"]].update(
            command="python3",
            at_prompt=False,
            managed_agent=False,
            agent_mode=False,
        )
        self.store.update_instance(terminal["id"], status="starting")

        first = self.manager.get_instance(terminal["id"])
        second = self.manager.get_instance(terminal["id"])

        self.assertEqual(first["kind"], "terminal")
        self.assertEqual(first["status"], "error")
        self.assertTrue(first["force_stop_required"])
        self.assertEqual(second["status"], "error")
        self.assertTrue(second["force_stop_required"])
        stopped = self.manager.force_stop(
            terminal["id"], confirmation=terminal["name"], actor="admin"
        )
        self.assertEqual(stopped["status"], "stopped")
        self.assertFalse(stopped["force_stop_required"])

    def test_missing_activation_journal_at_exact_prompt_recovers_terminal(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        self.store.update_instance(terminal["id"], status="starting")

        recovered = self.manager.get_instance(terminal["id"])

        self.assertEqual(recovered["kind"], "terminal")
        self.assertEqual(recovered["status"], "running")
        self.assertFalse(recovered["force_stop_required"])

    def test_restart_reconciles_crash_after_agent_pgid_before_mode_commit(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        with patch.object(
            self.store,
            "transition_shell_mode",
            side_effect=SystemExit("simulated service crash"),
        ):
            with self.assertRaises(SystemExit):
                self.manager.activate_agent(
                    terminal["id"], kind="aeon", actor="admin"
                )

        crashed = self.store.get_instance(terminal["id"])
        self.assertEqual(crashed["kind"], "terminal")
        self.assertEqual(crashed["status"], "starting")
        pending = (
            self.config.instance_state_dir
            / terminal["id"]
            / "managed-agent.pending.json"
        )
        self.assertTrue(pending.is_file())

        restarted = InstanceManager(
            self.store,
            self.config,
            command_runner=self.fake,
            pane_prompt_checker=self.fake.pane_at_prompt,
            pane_foreground_checker=self.fake.pane_has_managed_foreground,
        )
        with patch.object(
            restarted, "_managed_agent_is_foreground", return_value=True
        ):
            recovered = restarted.get_instance(terminal["id"])
        self.assertEqual(recovered["kind"], "aeon")
        self.assertEqual(recovered["status"], "running")
        self.assertFalse(pending.exists())

    def test_activation_detaches_stale_attach_input_before_fixed_argv(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        record = self.store.get_instance(terminal["id"])
        self.fake.sessions[record["tmux_name"]]["delayed_browser_input"] = "--evil "

        agent = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )

        self.assertEqual(agent["kind"], "aeon")
        calls = self.fake.calls
        detach_at = next(
            index for index, call in enumerate(calls) if call[1] == "detach-client"
        )
        paste_at = next(
            index
            for index, call in enumerate(calls)
            if "paste-buffer" in call[1:]
        )
        self.assertLess(detach_at, paste_at)
        self.assertNotIn("--evil", self.fake.loaded_payloads[-1])
        self.assertFalse(
            any(call[1] == "send-keys" and "Enter" in call for call in calls)
        )

    def test_atomic_agent_paste_failure_discards_edit_buffer_at_fresh_prompt(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        record = self.store.get_instance(terminal["id"])
        self.fake.paste_buffer_error = True
        calls_before = len(self.fake.calls)

        with self.assertRaises(InstanceLaunchError) as caught:
            self.manager.activate_agent(
                terminal["id"], kind="aeon", actor="admin"
            )

        self.assertFalse(caught.exception.launched)
        pane = self.fake.sessions[record["tmux_name"]]
        self.assertEqual(pane["pending"], "")
        self.assertTrue(pane["at_prompt"])
        self.assertFalse(pane["agent_mode"])
        durable = self.store.get_instance(terminal["id"])
        self.assertEqual(durable["kind"], "terminal")
        self.assertEqual(durable["status"], "running")
        pending = (
            self.config.instance_state_dir
            / terminal["id"]
            / "managed-agent.pending.json"
        )
        self.assertFalse(pending.exists())
        delivery_calls = self.fake.calls[calls_before:]
        self.assertEqual(
            sum("paste-buffer" in call[1:] for call in delivery_calls), 1
        )
        command_queue = next(
            call for call in delivery_calls if "paste-buffer" in call[1:]
        )
        self.assertEqual(command_queue[1], "load-buffer")
        self.assertIn(";", command_queue)
        self.assertLess(
            command_queue.index("load-buffer"), command_queue.index("paste-buffer")
        )
        self.assertFalse(
            any(call[1] == "send-keys" and "Enter" in call for call in delivery_calls)
        )
        self.assertEqual(self.fake.buffers, {})

    def test_private_tmux_paste_is_one_stdin_only_queue_and_scrubs_stale_buffer(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        prefix = f"nexus-input-{terminal['id'][:12]}-"
        stale_name = f"{prefix}{'a' * 32}"
        self.fake.buffers[stale_name] = "old secret"
        generation = self.manager.prepare_terminal_attachment(terminal["id"])[3]
        payload = "new password-like input"
        calls_before = len(self.fake.calls)

        accepted = self.manager.send_terminal_input(
            terminal["id"], payload, expected_generation=generation
        )

        self.assertTrue(accepted)
        calls = self.fake.calls[calls_before:]
        queues = [call for call in calls if call[1] == "load-buffer"]
        self.assertEqual(len(queues), 1)
        queue = queues[0]
        self.assertEqual(queue.count(";"), 1)
        separator = queue.index(";")
        self.assertEqual(queue[separator + 1], "paste-buffer")
        self.assertIn("-d", queue[separator + 1 :])
        self.assertNotIn(payload, queue)
        self.assertFalse(any(payload in argument for call in calls for argument in call))
        self.assertEqual(self.fake.loaded_payloads[-1], payload)
        self.assertEqual(self.fake.buffers, {})
        self.assertTrue(
            any(
                call[1] == "delete-buffer" and stale_name in call
                for call in calls
            )
        )

    def test_browser_attachment_hides_status_then_uses_read_only_tmux_client(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )

        args, environment = self.manager.tmux_attach_args(terminal["id"])

        self.assertEqual(args[1], "set-option")
        self.assertEqual(args[2], "-t")
        self.assertTrue(args[3].startswith("="))
        self.assertTrue(args[3].endswith(":"))
        self.assertEqual(args[4:7], ["status", "off", ";"])
        self.assertEqual(args[7:10], ["attach-session", "-f", "read-only"])
        self.assertNotIn("-C", args)
        self.assertNotIn("-r", args)
        self.assertEqual(args[-2], "-t")
        self.assertTrue(args[-1].startswith("="))
        self.assertEqual(environment["TERM"], "xterm-256color")
        self.assertEqual(environment["COLORTERM"], "truecolor")

    def test_browser_wheel_uses_tmux_copy_mode_without_pane_input(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        record = self.store.get_instance(terminal["id"])
        generation = self.manager.prepare_terminal_attachment(terminal["id"])[3]
        calls_before = len(self.fake.calls)
        loaded_before = list(self.fake.loaded_payloads)

        accepted = self.manager.scroll_terminal(
            terminal["id"], -3, expected_generation=generation
        )

        self.assertTrue(accepted)
        calls = self.fake.calls[calls_before:]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], "list-panes")
        queue = calls[1]
        target = f"={record['tmux_name']}:"
        self.assertEqual(
            queue[1:6], ["copy-mode", "-e", "-t", target, ";"]
        )
        self.assertEqual(queue[6], "send-keys")
        self.assertIn("-X", queue)
        self.assertEqual(queue[queue.index("-N") + 1], "3")
        self.assertEqual(queue[-1], "scroll-up")
        self.assertEqual(self.fake.loaded_payloads, loaded_before)
        self.assertEqual(self.fake.sessions[record["tmux_name"]]["pending"], "")

    def test_first_keystroke_after_browser_scroll_returns_to_live_bottom(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        generation = self.manager.prepare_terminal_attachment(terminal["id"])[3]
        self.assertTrue(
            self.manager.scroll_terminal(
                terminal["id"], -1, expected_generation=generation
            )
        )
        calls_before = len(self.fake.calls)

        accepted = self.manager.send_terminal_input(
            terminal["id"], "x", expected_generation=generation
        )

        self.assertTrue(accepted)
        calls = self.fake.calls[calls_before:]
        cancel_at = next(
            index
            for index, call in enumerate(calls)
            if call[1] == "copy-mode" and call[-1] == "cancel"
        )
        paste_at = next(
            index for index, call in enumerate(calls) if call[1] == "load-buffer"
        )
        self.assertLess(cancel_at, paste_at)
        self.assertNotIn(terminal["id"], self.manager._browser_scrolled_ids)

    def test_stale_or_unbounded_browser_scroll_is_rejected(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        generation = self.manager.prepare_terminal_attachment(terminal["id"])[3]
        with self.manager._lifecycle_lock(terminal["id"]):
            pass
        calls_before = len(self.fake.calls)

        self.assertFalse(
            self.manager.scroll_terminal(
                terminal["id"], -2, expected_generation=generation
            )
        )
        self.assertFalse(
            self.manager.scroll_terminal(
                terminal["id"], 101, expected_generation=generation + 1
            )
        )
        self.assertFalse(
            any(call[1] == "copy-mode" for call in self.fake.calls[calls_before:])
        )

    def test_private_tmux_buffer_cleanup_ambiguity_fails_closed(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        generation = self.manager.prepare_terminal_attachment(terminal["id"])[3]
        self.fake.paste_buffer_error = True
        self.fake.delete_buffer_error = True

        accepted = self.manager.send_terminal_input(
            terminal["id"], "must not be reported delivered", expected_generation=generation
        )

        self.assertFalse(accepted)
        self.assertTrue(self.fake.buffers)

    def test_private_tmux_buffer_query_failure_prevents_loading_payload(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        generation = self.manager.prepare_terminal_attachment(terminal["id"])[3]
        loaded_before = list(self.fake.loaded_payloads)
        self.fake.list_buffers_error = True

        accepted = self.manager.send_terminal_input(
            terminal["id"], "must remain browser-side", expected_generation=generation
        )

        self.assertFalse(accepted)
        self.assertEqual(self.fake.loaded_payloads, loaded_before)

    def test_atomic_agent_paste_error_after_delivery_is_force_recoverable(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        self.fake.paste_buffer_error = True
        self.fake.paste_deliver_then_error = True

        with self.assertRaises(InstanceLaunchError) as caught:
            self.manager.activate_agent(
                terminal["id"], kind="aeon", actor="admin"
            )

        self.assertTrue(caught.exception.launched)
        durable = self.store.get_instance(terminal["id"])
        self.assertEqual(durable["kind"], "aeon")
        self.assertEqual(durable["status"], "error")
        self.assertIn("force stop required", durable["last_error"])
        pending = (
            self.config.instance_state_dir
            / terminal["id"]
            / "managed-agent.pending.json"
        )
        self.assertTrue(pending.is_file())

    def test_attachment_generation_rejects_queued_pretransition_input(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        prepared = self.manager.prepare_terminal_attachment(terminal["id"])
        old_generation = prepared[3]
        with self.manager._lifecycle_lock(terminal["id"]):
            pass
        calls_before = len(self.fake.calls)

        accepted = self.manager.send_terminal_input(
            terminal["id"],
            "queued before transition",
            expected_generation=old_generation,
        )

        self.assertFalse(accepted)
        self.assertFalse(
            any(call[1] == "load-buffer" for call in self.fake.calls[calls_before:])
        )

    def test_resume_and_activation_stop_are_serialized(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        self.fake.sessions.clear()
        launches_before = sum(call[1] == "new-session" for call in self.fake.calls)
        barrier = threading.Barrier(3)
        results: list[object] = []

        def resume():
            barrier.wait()
            try:
                results.append(self.manager.resume_instance(terminal["id"], actor="admin"))
            except Exception as exc:
                results.append(exc)

        threads = [threading.Thread(target=resume) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(
            sum(call[1] == "new-session" for call in self.fake.calls),
            launches_before + 1,
        )
        self.assertEqual(sum(isinstance(value, dict) for value in results), 1)
        self.assertEqual(sum(isinstance(value, InstanceError) for value in results), 1)

        # Read-only reconciliation uses the serialization lock but must delay,
        # not lose, normal terminal input.
        attachment_generation = self.manager.prepare_terminal_attachment(
            terminal["id"]
        )[3]
        read_lock = self.manager._mode_lock(terminal["id"])
        read_lock.acquire()
        input_result: dict[str, bool] = {}
        input_thread = threading.Thread(
            target=lambda: input_result.setdefault(
                "accepted",
                self.manager.send_terminal_input(
                    terminal["id"],
                    "safe input",
                    expected_generation=attachment_generation,
                ),
            )
        )
        input_thread.start()
        time.sleep(0.03)
        self.assertTrue(input_thread.is_alive())
        read_lock.release()
        input_thread.join(timeout=2)
        self.assertTrue(input_result.get("accepted"))

        probe_entered = threading.Event()
        release_probe = threading.Event()
        activation_result: dict[str, object] = {}
        stop_result: dict[str, object] = {}

        def slow_status(_kind):
            probe_entered.set()
            if not release_probe.wait(timeout=2):
                raise AssertionError("test did not release provider probe")
            return {"installed": True, "connected": True}

        def activate():
            try:
                activation_result["value"] = self.manager.activate_agent(
                    terminal["id"], kind="codex", actor="admin"
                )
            except Exception as exc:
                activation_result["error"] = exc

        def stop():
            try:
                stop_result["value"] = self.manager.graceful_stop(
                    terminal["id"], actor="admin"
                )
            except Exception as exc:
                stop_result["error"] = exc

        with (
            patch("aeon.remote.instances.provider_status", side_effect=slow_status),
            patch(
                "aeon.remote.instances.provider_agent_command",
                return_value=SimpleNamespace(argv=("/safe/codex",)),
            ),
            patch(
                "aeon.remote.instances.subscription_environment",
                return_value={"HOME": "/home/aday", "PATH": "/safe/bin"},
            ),
        ):
            activation_thread = threading.Thread(target=activate)
            activation_thread.start()
            self.assertTrue(probe_entered.wait(timeout=1))
            with self.manager.terminal_input_guard(terminal["id"]) as allowed:
                self.assertFalse(allowed)
            self.assertFalse(
                self.manager.send_terminal_input(
                    terminal["id"],
                    "must be dropped",
                    expected_generation=attachment_generation,
                )
            )
            stop_thread = threading.Thread(target=stop)
            stop_thread.start()
            time.sleep(0.03)
            self.assertTrue(stop_thread.is_alive())
            release_probe.set()
            activation_thread.join(timeout=2)
            stop_thread.join(timeout=2)

        self.assertNotIn("error", activation_result)
        self.assertIsInstance(stop_result.get("error"), InstanceError)
        self.assertIn("End agent", str(stop_result["error"]))
        self.assertEqual(self.store.get_instance(terminal["id"])["kind"], "codex")

    def test_start_aeon_here_rejects_nonterminal_dead_and_outside_cwd(self):
        aeon = self.manager.create_instance(
            name="Existing Aeon",
            workspace=str(self.workspace),
            objective="",
            max_iterations=None,
            actor="admin",
        )
        with self.assertRaisesRegex(InstanceError, "managed terminal"):
            self.manager.start_aeon_here(aeon["id"], actor="admin")

        terminal = self.manager.create_terminal(
            name="Boundary shell",
            workspace=str(self.workspace),
            actor="admin",
        )
        record = self.store.get_instance(terminal["id"])
        outside = Path(self.temp.name) / "outside-terminal"
        outside.mkdir()
        self.fake.sessions[record["tmux_name"]]["cwd"] = str(outside)
        with self.assertRaisesRegex(InstanceError, "outside AEON_REMOTE_ALLOWED_ROOTS"):
            self.manager.start_aeon_here(terminal["id"], actor="admin")
        # A bad cwd is not exposed and must not make the overall listing fail.
        listed = self.manager.get_instance(terminal["id"])
        self.assertIsNone(listed["current_directory"])

        self.fake.sessions[record["tmux_name"]]["dead"] = True
        with self.assertRaisesRegex(InstanceError, "not running"):
            self.manager.start_aeon_here(terminal["id"], actor="admin")

    def test_terminal_graceful_stop_and_delete_use_exact_targets(self):
        terminal = self.manager.create_terminal(
            name="Temporary shell",
            workspace=str(self.workspace),
            actor="admin",
        )
        record = self.store.get_instance(terminal["id"])
        stopped = self.manager.graceful_stop(terminal["id"], actor="admin")
        self.assertEqual(stopped["status"], "stopping")
        stop_calls = [call for call in self.fake.calls if call[1] == "send-keys"]
        self.assertTrue(stop_calls)
        self.assertTrue(
            all(
                call[call.index("-t") + 1] == f"={record['tmux_name']}:"
                for call in stop_calls
            )
        )
        self.fake.sessions[record["tmux_name"]]["dead"] = True
        self.manager.delete_instance(
            terminal["id"], confirmation="Temporary shell", actor="admin"
        )
        self.assertIsNone(self.store.get_instance(terminal["id"]))

    def test_kill_instance_serializes_verified_stop_and_delete_against_resume(self):
        terminal = self.manager.create_terminal(
            name="Atomic child",
            workspace=str(self.workspace),
            actor="admin",
        )
        original_kill = self.manager._kill_session_and_verify_absent
        stop_entered = threading.Event()
        allow_stop = threading.Event()
        resume_started = threading.Event()
        resume_finished = threading.Event()
        kill_result: dict[str, object] = {}
        resume_result: dict[str, object] = {}

        def delayed_kill(*args, **kwargs):
            stop_entered.set()
            if not allow_stop.wait(timeout=2):
                raise AssertionError("test did not release verified force stop")
            return original_kill(*args, **kwargs)

        def kill():
            try:
                self.manager.kill_instance(
                    terminal["id"],
                    confirmation=terminal["name"],
                    actor="admin",
                )
                kill_result["deleted"] = True
            except Exception as exc:  # pragma: no cover - asserted below
                kill_result["error"] = exc

        def resume():
            resume_started.set()
            try:
                resume_result["value"] = self.manager.resume_instance(
                    terminal["id"], actor="admin"
                )
            except Exception as exc:
                resume_result["error"] = exc
            finally:
                resume_finished.set()

        with patch.object(
            self.manager,
            "_kill_session_and_verify_absent",
            side_effect=delayed_kill,
        ):
            kill_thread = threading.Thread(target=kill)
            kill_thread.start()
            self.assertTrue(stop_entered.wait(timeout=1))
            resume_thread = threading.Thread(target=resume)
            resume_thread.start()
            self.assertTrue(resume_started.wait(timeout=1))
            self.assertFalse(resume_finished.wait(timeout=0.05))
            allow_stop.set()
            kill_thread.join(timeout=2)
            resume_thread.join(timeout=2)

        self.assertFalse(kill_thread.is_alive())
        self.assertFalse(resume_thread.is_alive())
        self.assertEqual(kill_result, {"deleted": True})
        self.assertIsInstance(resume_result.get("error"), InstanceError)
        self.assertIn("Unknown session", str(resume_result["error"]))
        self.assertIsNone(self.store.get_instance(terminal["id"]))

    def test_tmux_query_errors_never_masquerade_as_absent_sessions(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        self.fake.list_panes_error = True
        with self.assertRaisesRegex(InstanceError, "exact tmux session pane"):
            self.manager.get_instance(terminal["id"])
        self.assertEqual(self.store.get_instance(terminal["id"])["status"], "running")

        self.fake.list_panes_error = False
        self.fake.sessions.clear()
        self.fake.list_sessions_error = True
        with self.assertRaisesRegex(InstanceError, "verify tmux session"):
            self.manager.get_instance(terminal["id"])

        self.fake.list_sessions_error = False
        self.fake.malformed_pane_output = True
        with self.assertRaisesRegex(InstanceError, "malformed pane state"):
            self.manager.get_instance(terminal["id"])

    def test_tmux_pane_format_uses_a_printable_sentinel_and_parses_real_shape(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )

        current = self.manager.get_instance(terminal["id"])

        pane_call = next(
            call for call in reversed(self.fake.calls) if call[1] == "list-panes"
        )
        pane_format = pane_call[pane_call.index("-F") + 1]
        expected_format = _FAKE_TMUX_PANE_FIELD_SEPARATOR.join(
            (
                "#{pane_dead}",
                "#{pane_pid}",
                "#{pane_dead_status}",
                "#{pane_current_command}",
            )
        )
        self.assertEqual(pane_format, expected_format)
        self.assertTrue(pane_format.isprintable())
        self.assertNotIn("\x1f", pane_format)
        self.assertEqual(current["pid"], 987654)
        self.assertEqual(current["process"], "bash")

    def test_tmux_pane_separator_collision_fails_closed(self):
        terminal = self.manager.create_terminal(
            workspace=str(self.workspace), actor="admin"
        )
        record = self.store.get_instance(terminal["id"])
        self.fake.sessions[record["tmux_name"]]["command"] = (
            f"bash{_FAKE_TMUX_PANE_FIELD_SEPARATOR}collision"
        )

        with self.assertRaisesRegex(InstanceError, "malformed pane state"):
            self.manager.get_instance(terminal["id"])

    def test_force_stop_kill_and_post_query_ambiguity_preserves_live_state(self):
        terminal = self.manager.create_terminal(
            name="Ambiguous force shell",
            workspace=str(self.workspace),
            actor="admin",
        )
        self.fake.kill_session_error = True
        self.fake.query_error_after_kill = True

        with self.assertRaisesRegex(InstanceError, "exact tmux session pane"):
            self.manager.force_stop(
                terminal["id"],
                confirmation=terminal["name"],
                actor="admin",
            )

        durable = self.store.get_instance(terminal["id"])
        self.assertEqual(durable["desired_state"], "running")
        self.assertEqual(durable["status"], "running")

    def test_kill_instance_never_deletes_when_force_stop_is_ambiguous(self):
        terminal = self.manager.create_terminal(
            name="Ambiguous disposable child",
            workspace=str(self.workspace),
            actor="admin",
        )
        self.fake.kill_session_error = True
        self.fake.query_error_after_kill = True

        with self.assertRaisesRegex(InstanceError, "exact tmux session pane"):
            self.manager.kill_instance(
                terminal["id"],
                confirmation=terminal["name"],
                actor="admin",
            )

        durable = self.store.get_instance(terminal["id"])
        self.assertIsNotNone(durable)
        self.assertEqual(durable["desired_state"], "running")
        self.assertEqual(durable["status"], "running")
        self.assertFalse(
            any(
                row["action"] == "instance_deleted"
                and row["instance_id"] == terminal["id"]
                for row in self.store.recent_audit(limit=20)
            )
        )

    def test_delete_query_error_preserves_durable_tab(self):
        terminal = self.manager.create_terminal(
            name="Ambiguous delete shell",
            workspace=str(self.workspace),
            actor="admin",
        )
        record = self.store.get_instance(terminal["id"])
        self.fake.sessions[record["tmux_name"]]["dead"] = True
        self.fake.list_panes_error = True

        with self.assertRaisesRegex(InstanceError, "exact tmux session pane"):
            self.manager.delete_instance(
                terminal["id"],
                confirmation=terminal["name"],
                actor="admin",
            )

        self.assertIsNotNone(self.store.get_instance(terminal["id"]))

    def test_direct_provider_kill_ambiguity_never_claims_stopped(self):
        with (
            patch(
                "aeon.remote.instances.provider_status",
                return_value={"installed": True, "connected": True},
            ),
            patch(
                "aeon.remote.instances.provider_agent_command",
                return_value=SimpleNamespace(argv=("/safe/codex",)),
            ),
            patch(
                "aeon.remote.instances.subscription_environment",
                return_value={"HOME": "/home/aday", "PATH": "/safe/bin"},
            ),
        ):
            provider = self.manager.create_instance(
                kind="codex",
                name="Ambiguous direct provider",
                workspace=str(self.workspace),
                objective="",
                max_iterations=None,
                actor="fixture",
            )
        self.fake.kill_session_error = True
        self.fake.query_error_after_kill = True

        with self.assertRaisesRegex(InstanceError, "exact tmux session pane"):
            self.manager.graceful_stop(provider["id"], actor="admin")

        durable = self.store.get_instance(provider["id"])
        self.assertNotEqual(durable["status"], "stopped")
        self.assertEqual(durable["desired_state"], "stopped")

    def test_terminal_stop_never_types_exit_into_a_foreground_command(self):
        terminal = self.manager.create_terminal(
            name="Busy shell",
            workspace=str(self.workspace),
            actor="admin",
        )
        record = self.store.get_instance(terminal["id"])
        self.fake.sessions[record["tmux_name"]]["command"] = "python3"
        calls_before = len(self.fake.calls)

        stopped = self.manager.graceful_stop(terminal["id"], actor="admin")
        self.assertEqual(stopped["status"], "stopping")
        stop_calls = self.fake.calls[calls_before:]
        sent = [call for call in stop_calls if call[1] == "send-keys"]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][-1], "C-c")
        self.assertFalse(any("exit" in call for call in sent))
        interrupt_at = stop_calls.index(sent[0])
        post_interrupt_probes = [
            call for call in stop_calls[interrupt_at + 1:] if call[1] == "list-panes"
        ]
        self.assertTrue(post_interrupt_probes)
        self.assertEqual(
            post_interrupt_probes[0][post_interrupt_probes[0].index("-t") + 1],
            f"={record['tmux_name']}:",
        )

        with self.assertRaisesRegex(InstanceError, "Confirmation"):
            self.manager.force_stop(
                terminal["id"], confirmation="wrong", actor="admin"
            )
        forced = self.manager.force_stop(
            terminal["id"], confirmation="Busy shell", actor="admin"
        )
        self.assertEqual(forced["status"], "stopped")

    def test_local_cli_adoption_is_unique_managed_tmux_without_shell(self):
        objective = "Continue this interactive task"
        # Local adoption is intentionally broader than the browser workspace
        # allowlist: an ordinary CLI may be started in any directory the local
        # account can already access, and that provenance remains explicit.
        local_workspace = Path(self.temp.name) / "outside-web-allowlist"
        local_workspace.mkdir()
        instance = self.manager.adopt_local_cli(
            workspace=local_workspace,
            cli_args=["--start", objective, "--debug"],
            objective=objective,
            max_iterations=None,
            model=None,
            actor="local-user",
        )
        launch = next(call for call in self.fake.calls if call[1] == "new-session")
        python_at = launch.index("/usr/bin/python3")
        self.assertEqual(
            launch[python_at:python_at + 4],
            ["/usr/bin/python3", "-m", "aeon.main", "--start"],
        )
        self.assertIn(objective, launch)
        self.assertIn("--debug", launch)
        self.assertNotIn("sh", launch)
        self.assertIn(f"AEON_REMOTE_INSTANCE_ID={instance['id']}", launch)
        self.assertIn(
            f"AEON_INSTANCE_SKILLS_DIR={self.config.instance_state_dir / instance['id'] / 'skills'}",
            launch,
        )
        stored = self.store.get_instance(instance["id"])
        self.assertEqual(stored["created_by"], "local-user")
        self.assertEqual(stored["workspace"], str(local_workspace.resolve()))
        self.assertEqual(stored["model"], self.config.default_model)
        self.assertEqual(stored["launch_origin"], "local")

        # The same exact locally authorized directory stays resumable from the
        # dashboard even though it is not a browser-allowed creation root.
        self.fake.sessions.clear()
        calls_before_resume = len(self.fake.calls)
        resumed = self.manager.resume_instance(instance["id"], actor="admin")
        self.assertEqual(resumed["status"], "running")
        resumed_launch = next(
            call
            for call in self.fake.calls[calls_before_resume:]
            if call[1] == "new-session"
        )
        self.assertIn("--resume-unfinished", resumed_launch)
        self.assertNotIn("--start", resumed_launch)

    def test_local_cli_adoption_uses_fixed_selected_opencode_harness(self):
        local_workspace = Path(self.temp.name) / "outside-web-opencode"
        local_workspace.mkdir()
        objective = "Inspect this workspace"

        instance = self.manager.adopt_local_cli(
            workspace=local_workspace,
            # OpenCode adoption ignores replayable compatibility argv and rebuilds
            # the exact reviewed command from typed fields below.
            cli_args=["--debug", "--harness", "legacy-aeon"],
            objective=objective,
            max_iterations=7,
            model=None,
            harness="opencode",
            browser_profile="local-profile",
            actor="local-user",
        )

        launch = next(call for call in self.fake.calls if call[1] == "new-session")
        python_at = launch.index("/usr/bin/python3")
        self.assertEqual(
            launch[python_at:python_at + 3],
            ["/usr/bin/python3", "-m", "aeon.harnesses.opencode_runtime"],
        )
        self.assertIn("--start", launch)
        self.assertIn(objective, launch)
        self.assertIn("--max-iterations", launch)
        self.assertIn("7", launch)
        self.assertIn("--browser-profile", launch)
        self.assertIn("local-profile", launch)
        self.assertNotIn("--debug", launch)
        self.assertNotIn("legacy-aeon", launch)
        setting = self.store.get_agent_setting(instance["id"], "aeon")
        self.assertEqual(setting["desired_harness"], "opencode")
        self.assertEqual(setting["applied_harness"], "opencode")

    def test_explicit_end_cancels_stale_running_worker_checkpoint(self):
        instance = self.manager.create_instance(
            name="Checkpointed agent",
            workspace=str(self.workspace),
            objective="finish the requested work",
            max_iterations=None,
            actor="admin",
        )
        record = self.store.get_instance(instance["id"])
        worker_state_root = Path(self.temp.name) / "explicit-end-worker-state"
        with patch.dict(
            os.environ, {"AEON_STATE_DIR": str(worker_state_root)}, clear=False
        ):
            directory = self.manager._worker_session_directory(record)
            directory.mkdir(mode=0o700, parents=True)
            state_path = directory / "session_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "instance_id": record["id"],
                        "execution_state": "running",
                        "objective": "finish the requested work",
                        "request_contract": {
                            "raw_request": "finish the requested work",
                            "state": "running",
                        },
                    }
                ),
                encoding="utf-8",
            )
            state_path.chmod(0o600)

            self.manager._cancel_worker_checkpoint_for_explicit_end(record)
            saved = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["execution_state"], "cancelled")
        self.assertEqual(saved["request_contract"]["state"], "cancelled")
        self.assertEqual(saved["stop_reason"], "nexus-explicit-end")
        self.assertEqual(saved["pid"], 0)

    def test_web_instance_never_inherits_local_workspace_bypass(self):
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        now = time.time()
        record = {
            "id": "1" * 32,
            "name": "Untrusted web row",
            "tmux_name": "aeon-111111111111",
            "workspace": str(outside.resolve()),
            "objective": "",
            "max_iterations": None,
            "model": self.config.default_model,
            "status": "created",
            "desired_state": "running",
            "created_at": now,
            "updated_at": now,
            "last_started_at": None,
            "last_error": "",
            "created_by": "admin",
            "launch_origin": "web",
        }
        self.store.create_instance(record)
        with self.assertRaisesRegex(InstanceError, "outside AEON_REMOTE_ALLOWED_ROOTS"):
            self.manager.resume_instance(record["id"], actor="admin")

    def test_force_stop_and_delete_require_exact_name_confirmation(self):
        instance = self.manager.create_instance(
            name="Protected agent",
            workspace=str(self.workspace),
            objective="",
            max_iterations=None,
            actor="admin",
        )
        with self.assertRaises(InstanceError):
            self.manager.force_stop(
                instance["id"], confirmation="wrong", actor="admin"
            )
        stopped = self.manager.force_stop(
            instance["id"], confirmation="Protected agent", actor="admin"
        )
        self.assertEqual(stopped["status"], "stopped")
        with self.assertRaises(InstanceError):
            self.manager.delete_instance(
                instance["id"], confirmation="wrong", actor="admin"
            )
        self.manager.delete_instance(
            instance["id"], confirmation="Protected agent", actor="admin"
        )
        self.assertIsNone(self.store.get_instance(instance["id"]))

    def test_missing_tmux_after_reboot_is_resumable_not_auto_relaunched(self):
        instance = self.manager.create_instance(
            name="Reboot-safe",
            workspace=str(self.workspace),
            objective="Long task",
            max_iterations=None,
            actor="admin",
        )
        record = self.store.get_instance(instance["id"])
        self.fake.sessions.clear()
        calls_before = len(self.fake.calls)
        reconciled = self.manager.reconcile(record)
        self.assertEqual(reconciled["status"], "interrupted")
        # Exact pane miss plus an independent list-sessions absence proof.
        self.assertEqual(len(self.fake.calls), calls_before + 2)
        self.assertFalse(
            any(call[1] == "new-session" for call in self.fake.calls[calls_before:])
        )

    def test_provider_agent_and_login_use_fixed_clean_environment_without_transcript(self):
        safe_home = Path(self.temp.name) / "provider-home"
        safe_home.mkdir(mode=0o700)
        clean_environment = {
            "HOME": str(safe_home),
            "PATH": "/usr/bin:/bin",
            "AEON_FLEET_SOCKET": "/tmp/forged-fleet-socket",
            "FLEET_TICKET": "forged-fleet-ticket",
            "GPU_AGENT_CLAIM_ID": "forged-gpu-claim",
            "CUDA_VISIBLE_DEVICES": "forged-cuda-selector",
            "DOCKER_HOST": "unix:///tmp/forged-docker.sock",
        }
        agent_command = SimpleNamespace(argv=("/safe/bin/claude",))
        login_command = SimpleNamespace(
            argv=("/safe/bin/claude", "auth", "login", "--claudeai")
        )
        with (
            patch(
                "aeon.remote.instances.provider_status",
                return_value={"installed": True, "connected": True},
            ),
            patch(
                "aeon.remote.instances.provider_agent_command",
                return_value=agent_command,
            ),
            patch(
                "aeon.remote.instances.provider_connect_command",
                return_value=login_command,
            ),
            patch(
                "aeon.remote.instances.subscription_environment",
                return_value=clean_environment,
            ),
        ):
            agent = self.manager.create_instance(
                kind="claude",
                name="Native Claude",
                workspace=str(self.workspace),
                objective="",
                max_iterations=None,
                actor="admin",
            )
            login = self.manager.create_provider_auth("claude", actor="admin")

        launches = [call for call in self.fake.calls if call[1] == "new-session"]
        self.assertEqual(len(launches), 2)
        for launch in launches:
            env_at = launch.index("/usr/bin/env")
            self.assertEqual(launch[env_at : env_at + 2], ["/usr/bin/env", "-i"])
            self.assertIn(f"HOME={safe_home}", launch)
            self.assertIn("PATH=/usr/bin:/bin", launch)
            wrapper_at = launch.index(self.low_priority_wrapper)
            self.assertGreater(wrapper_at, env_at)
            self.assertTrue(
                launch[wrapper_at + 1].startswith("/safe/bin/claude")
            )
            self.assertIn("CUDA_VISIBLE_DEVICES=void", launch)
            self.assertIn("NVIDIA_VISIBLE_DEVICES=void", launch)
            self.assertFalse(any(value.startswith("NEXUS_") for value in launch))
            rendered = "\x00".join(launch)
            for inherited in (
                "/tmp/forged-fleet-socket",
                "forged-fleet-ticket",
                "forged-gpu-claim",
                "forged-cuda-selector",
                "unix:///tmp/forged-docker.sock",
            ):
                self.assertNotIn(inherited, rendered)
            # Provider agents keep ordinary public networking; this boundary adds
            # priority/env containment, not the generic command tool's network ban.
            self.assertNotIn("systemd-run", launch)
            self.assertFalse(
                any("RestrictAddressFamilies" in value for value in launch)
            )
        self.assertEqual(agent["kind"], "claude")
        self.assertEqual(agent["provider"], "claude")
        self.assertFalse(agent["auth_session"])
        self.assertEqual(login["kind"], "claude_auth")
        self.assertEqual(login["provider"], "claude")
        self.assertTrue(login["auth_session"])
        self.assertFalse(any(call[1] == "pipe-pane" for call in self.fake.calls))
        with self.assertRaisesRegex(InstanceError, "do not accept"):
            self.manager.create_instance(
                kind="claude",
                name="Injected Claude",
                workspace=str(self.workspace),
                objective="browser supplied prompt",
                max_iterations=None,
                actor="admin",
            )

    def test_codex_instruction_overlay_is_private_file_backed_and_not_in_argv(self):
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
        codex_home = Path(self.temp.name) / "codex-home"
        codex_home.mkdir(mode=0o700)
        clean_environment = {
            "CODEX_HOME": str(codex_home),
            "HOME": str(Path(self.temp.name)),
            "PATH": "/usr/bin:/bin",
        }
        command = SimpleNamespace(argv=("/safe/bin/codex", "--no-alt-screen"))
        with (
            patch(
                "aeon.remote.instances.provider_status",
                return_value={"installed": True, "connected": True},
            ),
            patch(
                "aeon.remote.instances.provider_agent_command",
                return_value=command,
            ),
            patch(
                "aeon.remote.instances.subscription_environment",
                return_value=clean_environment,
            ),
        ):
            instance = manager.create_instance(
                kind="codex",
                name="Role Codex",
                workspace=str(self.workspace),
                objective="",
                max_iterations=None,
                actor="admin",
            )
            manager.force_stop(
                instance["id"], confirmation="Role Codex", actor="admin"
            )
            profile = instruction_service.create_profile(
                agent_kind="codex", name="Reviewer", actor="admin"
            )
            base_text = "Nexus base role sentinel: review architecture."
            local_text = "Nexus local role sentinel: focus on security."
            version = instruction_service.save_version(
                profile["id"], label="v1", content=base_text, actor="admin"
            )
            instruction_service.select_profile_version(instance["id"], version["id"])
            instruction_service.save_local_role(
                instance["id"],
                content=local_text,
                expected_revision=0,
                actor="admin",
            )
            manager.resume_instance(instance["id"], actor="admin")

        launch = [call for call in self.fake.calls if call[1] == "new-session"][-1]
        rendered_launch = "\x00".join(launch)
        self.assertNotIn(base_text, rendered_launch)
        self.assertNotIn(local_text, rendered_launch)
        self.assertIn("--profile", launch)
        profile_name = launch[launch.index("--profile") + 1]
        self.assertEqual(profile_name, f"nexus-{instance['id']}")
        profile_path = codex_home / f"{profile_name}.config.toml"
        self.assertEqual(profile_path.stat().st_mode & 0o777, 0o600)
        profile_body = profile_path.read_text(encoding="utf-8")
        self.assertIn(base_text, profile_body)
        self.assertIn(local_text, profile_body)
        binding = instruction_service.get_instance_binding(instance["id"])
        self.assertFalse(binding["pending"])
        audit_text = "\n".join(row["details_json"] for row in self.store.recent_audit(100))
        self.assertNotIn(base_text, audit_text)
        self.assertNotIn(local_text, audit_text)

    def test_project_manager_is_pinned_nexus_terminal_then_same_tab_aeon(self):
        config = replace(self.config, allowed_roots=(Path("/home/aday"),))
        fake = FakeTmux()
        manager = InstanceManager(
            self.store,
            config,
            command_runner=fake,
            pane_prompt_checker=fake.pane_at_prompt,
            pane_foreground_checker=fake.pane_has_managed_foreground,
        )
        self.assertFalse(any(call[1] == "new-session" for call in fake.calls))

        manager.ensure_default_home_terminal()
        project_manager = next(item for item in manager.list_instances() if item["pinned"])
        self.assertEqual(project_manager["workspace"], "/home/aday")
        self.assertEqual(project_manager["status"], "running")
        self.assertEqual(project_manager["kind"], "terminal")
        self.assertTrue(project_manager["shell_backed"])
        self.assertTrue(project_manager["always_present"])
        launch = next(call for call in fake.calls if call[1] == "new-session")
        bash_at = launch.index("/bin/bash")
        self.assertEqual(
            launch[bash_at : bash_at + 3],
            ["/bin/bash", "--noprofile", "--rcfile"],
        )
        self.assertTrue(launch[bash_at + 3].endswith("/managed-shell.rc"))
        self.assertEqual(launch[bash_at + 4], "-i")
        fake.sessions.clear()
        launches_before_recovery = sum(
            call[1] == "new-session" for call in fake.calls
        )
        manager.ensure_default_home_terminal()
        self.assertEqual(
            sum(call[1] == "new-session" for call in fake.calls),
            launches_before_recovery + 1,
        )
        capability_path = config.state_dir / "orchestrator-control.token"
        capability_path.write_text("test-capability", encoding="ascii")
        capability_path.chmod(0o600)
        with patch.dict(
            os.environ,
            {
                "NEXUS_INTERNAL_ORCHESTRATOR_URL": (
                    "http://127.0.0.1:8765/internal/orchestrator/agents"
                )
            },
            clear=False,
        ):
            agent = manager.activate_agent(
                project_manager["id"], kind="aeon", actor="admin"
            )
        self.assertEqual(agent["id"], project_manager["id"])
        self.assertEqual(agent["mode"], "agent")
        command = fake.loaded_payloads[-1]
        self.assertNotIn("--start", command)
        self.assertIn("AEON_MAIN_ORCHESTRATOR=1", command)
        self.assertIn("AEON_CHAT_TRANSCRIPT_PATH=", command)
        self.assertIn("NEXUS_INTERNAL_ORCHESTRATOR_URL=", command)
        self.assertIn(f"NEXUS_ORCHESTRATOR_TOKEN_FILE={capability_path}", command)
        launches_before_idempotent_start = len(fake.loaded_payloads)
        ensured = manager.ensure_main_orchestrator(actor="admin")
        self.assertEqual(ensured["id"], project_manager["id"])
        self.assertEqual(len(fake.loaded_payloads), launches_before_idempotent_start)

        # A managed Aeon may exit to the verified outer shell between Nexus
        # supervision passes.  The supervisor must reconcile and reactivate it
        # itself; recovery cannot depend on a browser list/refresh request.
        stored_project_manager = self.store.get_instance(project_manager["id"])
        pane = fake.sessions[stored_project_manager["tmux_name"]]
        pane.update(
            command="bash",
            agent_mode=False,
            managed_agent=False,
            at_prompt=True,
        )
        launches_before_supervised_recovery = len(fake.loaded_payloads)
        recovered = manager.ensure_main_orchestrator(actor="nexus-controller")
        self.assertEqual(recovered["mode"], "agent")
        self.assertEqual(recovered["kind"], "aeon")
        self.assertEqual(
            len(fake.loaded_payloads), launches_before_supervised_recovery + 1
        )

        sent = manager.send_main_orchestrator_message(
            "Please review the workspace.", actor="admin"
        )
        self.assertEqual(sent["role"], "user")
        self.assertEqual(sent["content"], "Please review the workspace.")
        self.assertEqual(manager.read_main_orchestrator_chat(), [sent])
        deliveries_after_sent = len(fake.loaded_payloads)
        self.assertEqual(
            manager.send_main_orchestrator_message(
                "Please review the workspace.",
                actor="admin",
                message_id=sent["id"],
            ),
            sent,
        )
        self.assertEqual(len(fake.loaded_payloads), deliveries_after_sent)
        self.assertEqual(
            fake.loaded_payloads[-1],
            "\x1b[200~"
            + build_chat_delivery_envelope(sent["id"], sent["content"])
            + "\x1b[201~\r",
        )
        stopped = manager.stop_main_orchestrator_turn(actor="admin")
        self.assertTrue(stopped)
        self.assertIn("/__nexus_stop_current_turn_", fake.loaded_payloads[-1])
        stored_project_manager = self.store.get_instance(project_manager["id"])
        self.assertTrue(fake.sessions[stored_project_manager["tmux_name"]]["agent_mode"])
        image_bytes = io.BytesIO()
        Image.new("RGB", (8, 6), "purple").save(image_bytes, format="PNG")
        media = manager.send_main_orchestrator_message(
            "What is shown?",
            actor="admin",
            uploads=[
                SimpleNamespace(
                    filename="../screen shot.png",
                    content_type="image/png",
                    file=io.BytesIO(image_bytes.getvalue()),
                )
            ],
        )
        attachment = media["attachments"][0]
        self.assertEqual(attachment["name"], "screen shot.png")
        self.assertNotIn(str(config.instance_state_dir), repr(media))
        attachment_path, resolved = manager.resolve_main_orchestrator_attachment(
            attachment["id"]
        )
        self.assertEqual(resolved, attachment)
        self.assertEqual(attachment_path.stat().st_mode & 0o777, 0o600)
        agent_path, agent_resolved = manager.resolve_agent_chat_attachment(
            project_manager["id"], attachment["id"]
        )
        self.assertEqual(agent_path, attachment_path)
        self.assertEqual(agent_resolved, attachment)
        self.assertIn(str(attachment_path), fake.loaded_payloads[-1])
        self.assertIn("use analyze_image", fake.loaded_payloads[-1])
        visible_transcript = manager._main_orchestrator_chat_path().read_text(
            encoding="utf-8"
        )
        self.assertNotIn(str(attachment_path), visible_transcript)
        self.assertNotIn("use analyze_image", visible_transcript)
        deliveries_after_media = len(fake.loaded_payloads)
        media_retry = manager.send_main_orchestrator_message(
            "What is shown?",
            actor="admin",
            message_id=media["id"],
            uploads=[
                SimpleNamespace(
                    filename="../screen shot.png",
                    content_type="image/png",
                    file=io.BytesIO(image_bytes.getvalue()),
                )
            ],
        )
        self.assertEqual(media_retry, media)
        self.assertEqual(len(fake.loaded_payloads), deliveries_after_media)
        changed_image_bytes = io.BytesIO()
        Image.new("RGB", (8, 6), "blue").save(
            changed_image_bytes, format="PNG"
        )
        self.assertEqual(
            len(changed_image_bytes.getvalue()), len(image_bytes.getvalue())
        )
        with self.assertRaisesRegex(InstanceError, "identity conflicts"):
            manager.send_main_orchestrator_message(
                "What is shown?",
                actor="admin",
                message_id=media["id"],
                uploads=[
                    SimpleNamespace(
                        filename="../screen shot.png",
                        content_type="image/png",
                        file=io.BytesIO(changed_image_bytes.getvalue()),
                    )
                ],
            )
        self.assertEqual(len(fake.loaded_payloads), deliveries_after_media)

        self.assertEqual(manager.read_main_orchestrator_chat(), [sent, media])
        deliveries_before_invalid = len(fake.loaded_payloads)
        with self.assertRaisesRegex(InstanceError, "control characters"):
            manager.send_main_orchestrator_message(
                "do not deliver\x1bthis", actor="admin"
            )
        self.assertEqual(len(fake.loaded_payloads), deliveries_before_invalid)
        self.assertEqual(manager.read_main_orchestrator_chat(), [sent, media])

        ended = manager.end_agent(project_manager["id"], actor="admin")
        self.assertEqual(ended["mode"], "terminal")
        self.assertEqual(ended["status"], "running")
        with patch.dict(
            os.environ,
            {
                "NEXUS_INTERNAL_ORCHESTRATOR_URL": (
                    "http://127.0.0.1:8765/internal/orchestrator/agents"
                )
            },
            clear=False,
        ):
            restarted = manager.ensure_main_orchestrator(actor="admin")
        self.assertEqual(restarted["id"], project_manager["id"])
        self.assertEqual(restarted["mode"], "agent")
        self.assertEqual(restarted["kind"], "aeon")
        self.assertEqual(manager.read_main_orchestrator_chat(), [sent, media])

        crash_message_id = "msg-" + "7" * 32
        crash_message = "Recover after the append-to-state crash."
        original_state_write = chat_transcript._write_delivery_entries_locked
        injected = False

        def fail_committed_state_once(directory_fd, entries):
            nonlocal injected
            crash_entry = entries.get(crash_message_id)
            if (
                not injected
                and crash_entry is not None
                and crash_entry.get("state") == "committed"
            ):
                injected = True
                raise ChatTranscriptError("injected state receipt failure")
            return original_state_write(directory_fd, entries)

        deliveries_before_state_crash = len(fake.loaded_payloads)
        with patch.object(
            chat_transcript,
            "_write_delivery_entries_locked",
            side_effect=fail_committed_state_once,
        ):
            with self.assertRaisesRegex(InstanceError, "history could not be saved"):
                manager.send_main_orchestrator_message(
                    crash_message,
                    actor="admin",
                    message_id=crash_message_id,
                )
        with self.assertRaisesRegex(InstanceError, "delivery is ambiguous"):
            manager.send_main_orchestrator_message(
                crash_message,
                actor="admin",
                message_id=crash_message_id,
            )
        self.assertEqual(
            len(fake.loaded_payloads), deliveries_before_state_crash + 1
        )
        self.assertEqual(
            sum(
                item.get("id") == crash_message_id
                for item in manager.read_main_orchestrator_chat()
            ),
            1,
        )

        ambiguous_message_id = "msg-" + "8" * 32
        deliveries_before_commit_failure = len(fake.loaded_payloads)
        with patch(
            "aeon.remote.instances.commit_chat_delivery",
            side_effect=ChatTranscriptError("injected pre-append commit failure"),
        ):
            with self.assertRaisesRegex(InstanceError, "history could not be saved"):
                manager.send_main_orchestrator_message(
                    "Never paste this ambiguous turn twice.",
                    actor="admin",
                    message_id=ambiguous_message_id,
                )
        with self.assertRaisesRegex(InstanceError, "delivery is ambiguous"):
            manager.send_main_orchestrator_message(
                "Never paste this ambiguous turn twice.",
                actor="admin",
                message_id=ambiguous_message_id,
            )
        self.assertEqual(
            len(fake.loaded_payloads), deliveries_before_commit_failure + 1
        )

        unacknowledged_message_id = "msg-" + "9" * 32
        deliveries_before_unacknowledged = len(fake.loaded_payloads)
        with patch(
            "aeon.remote.instances.wait_for_chat_delivery_consumed",
            return_value=False,
        ):
            with self.assertRaisesRegex(InstanceError, "delivery is ambiguous"):
                manager.send_main_orchestrator_message(
                    "Require an exact receiver acknowledgement.",
                    actor="admin",
                    message_id=unacknowledged_message_id,
                )
        with self.assertRaisesRegex(InstanceError, "delivery is ambiguous"):
            manager.send_main_orchestrator_message(
                "Require an exact receiver acknowledgement.",
                actor="admin",
                message_id=unacknowledged_message_id,
            )
        self.assertEqual(
            len(fake.loaded_payloads), deliveries_before_unacknowledged + 1
        )
        self.assertEqual(
            sum(
                item.get("id") == unacknowledged_message_id
                for item in manager.read_main_orchestrator_chat()
            ),
            1,
        )

        worker_state_root = Path(self.temp.name) / "worker-state"
        with patch.dict(
            os.environ, {"AEON_STATE_DIR": str(worker_state_root)}, clear=False
        ):
            worker_session = manager._worker_session_directory(
                self.store.get_instance(project_manager["id"])
            )
            worker_session.mkdir(mode=0o700, parents=True)
            (worker_session / "session_state.json").write_text(
                json.dumps({"memories": {"old": "context"}}), encoding="utf-8"
            )
            (worker_session / "session_state.json").chmod(0o600)
            (worker_session / "interrupted_session.json").write_text(
                json.dumps({"objective": "old task"}), encoding="utf-8"
            )
            (worker_session / "interrupted_session.json").chmod(0o600)
            fresh = manager.ensure_main_orchestrator(
                actor="admin", fresh_context=True
            )
        self.assertEqual(fresh["mode"], "agent")
        self.assertEqual(
            json.loads((worker_session / "session_state.json").read_text()), {}
        )
        self.assertFalse((worker_session / "interrupted_session.json").exists())
        self.assertEqual(manager.read_main_orchestrator_chat(), [])
        reset_audits = [
            item
            for item in self.store.recent_audit(100)
            if item["action"] == "agent_context_reset"
        ]
        self.assertEqual(len(reset_audits), 1)

        invalid_target = Path(self.temp.name) / "invalid-interrupted-state"
        invalid_target.write_text("not owned context", encoding="utf-8")
        interrupted = worker_session / "interrupted_session.json"
        interrupted.symlink_to(invalid_target)
        with patch.dict(
            os.environ, {"AEON_STATE_DIR": str(worker_state_root)}, clear=False
        ):
            with self.assertRaisesRegex(
                InstanceError, "Interrupted agent context identity is invalid"
            ):
                manager.ensure_main_orchestrator(
                    actor="admin", fresh_context=True
                )
            with self.assertRaisesRegex(
                InstanceError, "waiting for a verified fresh-context reset"
            ):
                manager.ensure_main_orchestrator(actor="nexus-controller")
            interrupted.unlink()
            recovered_fresh = manager.ensure_main_orchestrator(
                actor="admin", fresh_context=True
            )
        self.assertEqual(recovered_fresh["mode"], "agent")
        calls_before_protected_kill = len(fake.calls)
        with self.assertRaisesRegex(InstanceError, "permanent"):
            manager.kill_instance(
                project_manager["id"],
                confirmation=project_manager["name"],
                actor="admin",
            )
        self.assertEqual(len(fake.calls), calls_before_protected_kill)
        self.assertIsNotNone(self.store.get_instance(project_manager["id"]))

        ordinary = manager.create_terminal(
            name="Ordinary terminal",
            workspace="/home/aday/NexusAgentDashboard",
            actor="admin",
        )
        with patch.dict(
            os.environ,
            {
                "NEXUS_INTERNAL_ORCHESTRATOR_URL": (
                    "http://127.0.0.1:8765/internal/orchestrator/agents"
                )
            },
            clear=False,
        ):
            manager.activate_agent(ordinary["id"], kind="aeon", actor="admin")
        ordinary_command = fake.loaded_payloads[-1]
        self.assertNotIn("NEXUS_INTERNAL_ORCHESTRATOR_URL=", ordinary_command)
        self.assertNotIn("NEXUS_ORCHESTRATOR_TOKEN_FILE=", ordinary_command)
        self.assertIn("AEON_CHAT_TRANSCRIPT_PATH=", ordinary_command)
        initial_chat_revision = manager.agent_chat_revision(ordinary["id"])
        ordinary_message = manager.send_agent_chat_message(
            ordinary["id"], "Talk through this task.", actor="admin"
        )
        message_chat_revision = manager.agent_chat_revision(ordinary["id"])
        self.assertNotEqual(message_chat_revision, initial_chat_revision)
        self.assertEqual(ordinary_message["content"], "Talk through this task.")
        self.assertEqual(manager.read_agent_chat(ordinary["id"]), [ordinary_message])
        self.assertEqual(
            fake.loaded_payloads[-1],
            "\x1b[200~"
            + build_chat_delivery_envelope(
                ordinary_message["id"], ordinary_message["content"]
            )
            + "\x1b[201~\r",
        )
        voice_id = "msg-" + "b" * 32
        deliveries_before_retry = len(fake.loaded_payloads)
        durable = manager.send_agent_chat_message(
            ordinary["id"],
            "Do not lose this voice turn.",
            actor="admin",
            message_id=voice_id,
        )
        retried = manager.send_agent_chat_message(
            ordinary["id"],
            "Do not lose this voice turn.",
            actor="admin",
            message_id=voice_id,
        )
        self.assertNotEqual(
            manager.agent_chat_revision(ordinary["id"]),
            message_chat_revision,
        )
        self.assertEqual(durable, retried)
        self.assertEqual(durable["id"], voice_id)
        self.assertEqual(len(fake.loaded_payloads), deliveries_before_retry + 1)
        self.assertTrue(manager.stop_agent_chat_turn(ordinary["id"], actor="admin"))
        self.assertIn("/__nexus_stop_current_turn_", fake.loaded_payloads[-1])
        with self.assertRaisesRegex(InstanceError, "permanent"):
            manager.delete_instance(
                project_manager["id"],
                confirmation=project_manager["name"],
                actor="admin",
            )

    def test_controller_starts_and_maintains_primary_without_a_browser(self):
        config = replace(self.config, allowed_roots=(Path("/home/aday"),))
        fake = FakeTmux()
        manager = InstanceManager(
            self.store,
            config,
            command_runner=fake,
            pane_prompt_checker=fake.pane_at_prompt,
            pane_foreground_checker=fake.pane_has_managed_foreground,
        )
        capability_path = config.state_dir / "orchestrator-control.token"
        capability_path.write_text("test-capability", encoding="ascii")
        capability_path.chmod(0o600)

        with patch.dict(
            os.environ,
            {
                "NEXUS_INTERNAL_ORCHESTRATOR_URL": (
                    "http://127.0.0.1:8765/internal/orchestrator/agents"
                )
            },
            clear=False,
        ):
            manager.ensure_persistent_main_orchestrator()
            launches = len(fake.loaded_payloads)
            manager.ensure_persistent_main_orchestrator()

        primary = next(item for item in manager.list_instances() if item["pinned"])
        self.assertEqual(primary["mode"], "agent")
        self.assertEqual(primary["kind"], "aeon")
        self.assertEqual(primary["status"], "running")
        self.assertEqual(len(fake.loaded_payloads), launches)

        # A reboot removes the tmux server without first returning the managed
        # Aeon to its outer shell.  The controller must recreate that exact
        # pinned shell and reactivate the agent without browser involvement.
        fake.sessions.clear()
        shell_launches = sum(call[1] == "new-session" for call in fake.calls)
        agent_launches = len(fake.loaded_payloads)
        with patch.dict(
            os.environ,
            {
                "NEXUS_INTERNAL_ORCHESTRATOR_URL": (
                    "http://127.0.0.1:8765/internal/orchestrator/agents"
                )
            },
            clear=False,
        ):
            manager.ensure_persistent_main_orchestrator()

        recovered = next(
            item for item in manager.list_instances() if item["pinned"]
        )
        self.assertEqual(recovered["mode"], "agent")
        self.assertEqual(recovered["kind"], "aeon")
        self.assertEqual(recovered["status"], "running")
        self.assertEqual(
            sum(call[1] == "new-session" for call in fake.calls),
            shell_launches + 1,
        )
        self.assertEqual(len(fake.loaded_payloads), agent_launches + 1)


class TestRemoteStaticSafety(unittest.TestCase):
    def test_initial_tmux_snapshot_uses_terminal_line_endings(self):
        snapshot = (
            b"first\n"
            b"\x1b[01;34msecond\x1b[0m\r\n"
            b"third\runchanged"
        )

        normalized = _normalize_terminal_snapshot(snapshot)

        self.assertEqual(
            normalized,
            b"first\r\n"
            b"\x1b[01;34msecond\x1b[0m\r\n"
            b"third\runchanged",
        )

    def test_only_exact_bounded_xterm_startup_replies_use_attach_pty(self):
        accepted = (
            "\x1b[?1;2c",
            "\x1b[>0;276;0c",
            "\x1b[>12345;54321;99999c",
            "\x1b]10;rgb:e7e7/e7e7/e7e7\x1b\\",
            "\x1b]11;rgb:0000/1111/aBcD\x1b\\",
        )
        rejected = (
            "\x1b[A",
            "\x1b[?1;2cuser-input",
            "\x1b[?1;2c\x1b[>0;276;0c",
            "\x1b[>0;123456;0c",
            "\x1b]12;rgb:e7e7/e7e7/e7e7\x1b\\",
            "\x1b]10;rgb:12345/0000/0000\x1b\\",
            "\x1b]10;rgb:e7e7/e7e7/e7e7\x07",
            "ordinary text",
        )

        self.assertTrue(all(_is_terminal_response(value) for value in accepted))
        self.assertTrue(all(not _is_terminal_response(value) for value in rejected))

    def test_browser_terminal_reply_never_reaches_pane_input_manager(self):
        class Manager:
            def __init__(self):
                self.calls = []

            def send_terminal_input(
                self, instance_id, data, *, expected_generation
            ):
                self.calls.append((instance_id, data, expected_generation))
                return True

        class Socket:
            def __init__(self):
                self.closed = []

            async def close(self, *, code):
                self.closed.append(code)

        async def scenario():
            manager = Manager()
            socket = Socket()
            read_fd, write_fd = os.pipe()
            try:
                report = "\x1b]10;rgb:e7e7/e7e7/e7e7\x1b\\"
                accepted = await _forward_browser_input(
                    socket,
                    manager,
                    "instance",
                    report,
                    4,
                    write_fd,
                )
                self.assertTrue(accepted)
                self.assertEqual(os.read(read_fd, 4096), report.encode("ascii"))
                self.assertEqual(manager.calls, [])

                accepted = await _forward_browser_input(
                    socket,
                    manager,
                    "instance",
                    "\x1b[A",
                    4,
                    write_fd,
                )
                self.assertTrue(accepted)
                self.assertEqual(manager.calls, [("instance", "\x1b[A", 4)])
                self.assertEqual(socket.closed, [])
            finally:
                os.close(read_fd)
                os.close(write_fd)

        asyncio.run(scenario())

    def test_resize_notifies_exact_tmux_attach_client(self):
        with (
            patch("aeon.remote.terminal.fcntl.ioctl") as ioctl,
            patch("aeon.remote.terminal.os.kill") as kill,
        ):
            _resize_attached_client(7, 1234, -100, 9000)

        packed = struct.pack("HHHH", 10, 500, 0, 0)
        ioctl.assert_called_once_with(7, termios.TIOCSWINSZ, packed)
        kill.assert_called_once_with(1234, signal.SIGWINCH)

    def test_full_terminal_output_queue_always_retains_eof_sentinel(self):
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=256)
        for _ in range(256):
            queue.put_nowait(b"display")

        _enqueue_terminal_output(queue, None)

        self.assertEqual(queue.qsize(), 256)
        values = [queue.get_nowait() for _ in range(256)]
        self.assertIsNone(values[-1])

    def test_stale_bridge_generation_closes_before_forwarding_queued_input(self):
        class GenerationManager:
            def __init__(self):
                self.current_generation = 2
                self.calls = []

            def send_terminal_input(
                self, instance_id, data, *, expected_generation
            ):
                self.calls.append((instance_id, data, expected_generation))
                return expected_generation == self.current_generation

        class Socket:
            def __init__(self):
                self.closed = []

            async def close(self, *, code):
                self.closed.append(code)

        async def scenario():
            manager = GenerationManager()
            socket = Socket()
            accepted = await _forward_terminal_input(
                socket,
                manager,
                "instance",
                "queued-before-transition",
                1,
            )
            self.assertFalse(accepted)
            self.assertEqual(socket.closed, [1012])
            self.assertEqual(
                manager.calls,
                [("instance", "queued-before-transition", 1)],
            )

        asyncio.run(scenario())

    def test_stale_bridge_generation_closes_before_forwarding_wheel_scroll(self):
        class GenerationManager:
            def __init__(self):
                self.current_generation = 2
                self.calls = []

            def scroll_terminal(
                self, instance_id, lines, *, expected_generation
            ):
                self.calls.append((instance_id, lines, expected_generation))
                return expected_generation == self.current_generation

        class Socket:
            def __init__(self):
                self.closed = []

            async def close(self, *, code):
                self.closed.append(code)

        async def scenario():
            manager = GenerationManager()
            socket = Socket()
            accepted = await _forward_terminal_scroll(
                socket,
                manager,
                "instance",
                -4,
                1,
            )
            self.assertFalse(accepted)
            self.assertEqual(socket.closed, [1012])
            self.assertEqual(manager.calls, [("instance", -4, 1)])

        asyncio.run(scenario())

    def test_websocket_session_lookup_never_blocks_asyncio_loop(self):
        class SlowAuth:
            @staticmethod
            def session(_raw_session_token):
                time.sleep(0.15)
                return {"username": "admin"}

        async def scenario():
            started = time.monotonic()
            task = asyncio.create_task(_websocket_session(SlowAuth(), "token"))
            await asyncio.sleep(0.02)
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.1)
            self.assertEqual((await task)["username"], "admin")

        asyncio.run(scenario())

    def test_terminal_attachment_preflight_never_blocks_asyncio_loop(self):
        class SlowManager:
            @staticmethod
            def prepare_terminal_attachment(_instance_id):
                time.sleep(0.15)
                return ["/definitely/missing-tmux"], {}, b"", 0

        async def scenario():
            started = time.monotonic()
            task = asyncio.create_task(
                bridge_terminal(SimpleNamespace(), SlowManager(), "instance")
            )
            await asyncio.sleep(0.02)
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.1)
            await asyncio.gather(task, return_exceptions=True)

        asyncio.run(scenario())

    def test_frontend_does_not_store_credentials_or_inject_untrusted_html(self):
        static = Path(__file__).resolve().parents[1] / "remote" / "static"
        js = (static / "app.js").read_text(encoding="utf-8")
        html = (static / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("localStorage", js)
        self.assertNotIn("sessionStorage", js)
        self.assertNotIn(".innerHTML", js)
        self.assertIn("integrity=\"sha384-", html)
        self.assertNotIn("<script>", html)

    def test_standalone_frontend_uses_terminal_first_agent_lifecycle(self):
        static = Path(__file__).resolve().parents[1] / "remote" / "static"
        js = (static / "app.js").read_text(encoding="utf-8")
        html = (static / "index.html").read_text(encoding="utf-8")
        self.assertIn('api("/api/terminals"', js)
        self.assertIn("/activate-agent", js)
        self.assertIn("/end-agent", js)
        self.assertIn("instance.force_stop_required === true", js)
        self.assertIn("live || forceRequired", js)
        self.assertIn('["aeon", "codex", "claude", "grok"].includes(instance.kind)', js)
        self.assertIn("MAX_RECONNECT_ATTEMPTS", js)
        self.assertIn("state.socket = null", js)
        self.assertIn("scheduleTerminalReconnect(instanceId, generation)", js)
        self.assertIn('window.addEventListener("online"', js)
        self.assertIn('document.addEventListener("visibilitychange"', js)
        self.assertNotIn('api("/api/instances", {\n        method: "POST"', js)
        self.assertNotIn('id="instance-objective"', html)
        self.assertNotIn('id="instance-iterations"', html)
        # Standalone Aeon Remote preserves its opt-out TOTP default. Nexus serves
        # a different static directory and explicitly selects password-only OIDC.
        self.assertIn("Authenticator code", html)
        self.assertIn('id="login-otp"', html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
