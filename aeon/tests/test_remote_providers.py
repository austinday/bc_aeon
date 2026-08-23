"""Hermetic provider-command tests; no login, network, or tmux is started."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from aeon.remote.providers import (
    PROVIDER_IDS,
    ProviderError,
    ProviderUnavailableError,
    list_provider_statuses,
    provider_agent_command,
    provider_connect_command,
    provider_spec,
    provider_status,
    subscription_environment,
)


class ProviderFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.bin_dir = root / "bin"
        self.bin_dir.mkdir(mode=0o700)
        self.executables = {}
        for name in ("codex", "claude", "grok"):
            path = self.bin_dir / name
            path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            path.chmod(0o700)
            self.executables[name] = str(path)

    def tearDown(self):
        self.temp.cleanup()

    def which(self, name):
        return self.executables.get(name)


class TestProviderAllowlist(ProviderFixture):
    def test_only_reviewed_provider_ids_are_available(self):
        self.assertEqual(PROVIDER_IDS, frozenset({"codex", "claude", "grok"}))
        self.assertEqual(provider_spec("codex").instruction_filename, "AGENTS.md")
        self.assertEqual(provider_spec("claude").instruction_filename, "CLAUDE.md")
        self.assertEqual(provider_spec("grok").instruction_filename, "AGENTS.md")
        for value in ("Grok", "Codex", " codex", "codex ", "", None, 7):
            with self.subTest(value=value), self.assertRaises(ProviderError):
                provider_spec(value)  # type: ignore[arg-type]

    def test_unavailable_and_world_writable_executables_fail_closed(self):
        with self.assertRaises(ProviderUnavailableError):
            provider_connect_command("codex", which=lambda _name: None)

        codex = Path(self.executables["codex"])
        codex.chmod(0o707)
        result = provider_status("codex", which=self.which)
        self.assertFalse(result["installed"])
        self.assertFalse(result["connected"])


class TestProviderStatus(ProviderFixture):
    def test_status_uses_fixed_argv_and_discards_every_stream(self):
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return SimpleNamespace(returncode=0)

        result = provider_status(
            "codex",
            runner=runner,
            which=self.which,
            environ={
                "PATH": "/safe",
                "OPENAI_API_KEY": "never-forward",
                "NEXUS_OIDC_PRIVATE_SECRET": "never-forward-either",
            },
        )
        self.assertEqual(
            result,
            {
                "id": "codex",
                "label": "OpenAI Codex",
                "installed": True,
                "connected": True,
            },
        )
        argv, kwargs = calls[0]
        self.assertEqual(argv, [self.executables["codex"], "login", "status"])
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(kwargs["stdout"], subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], subprocess.DEVNULL)
        self.assertFalse(kwargs["check"])
        self.assertEqual(kwargs["timeout"], 8)
        self.assertNotIn("shell", kwargs)
        self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
        self.assertNotIn("NEXUS_OIDC_PRIVATE_SECRET", kwargs["env"])

    def test_nonzero_timeout_and_runner_error_are_disconnected(self):
        nonzero = provider_status(
            "claude",
            runner=lambda _argv, **_kwargs: SimpleNamespace(returncode=1),
            which=self.which,
        )
        self.assertTrue(nonzero["installed"])
        self.assertFalse(nonzero["connected"])

        def timeout(_argv, **_kwargs):
            raise subprocess.TimeoutExpired("claude", 8)

        timed_out = provider_status("claude", runner=timeout, which=self.which)
        self.assertTrue(timed_out["installed"])
        self.assertFalse(timed_out["connected"])

    def test_list_order_is_stable_and_never_probes_grok(self):
        resolved = []
        commands = []

        def which(name):
            resolved.append(name)
            return self.executables[name]

        def runner(argv, **_kwargs):
            commands.append(argv)
            return SimpleNamespace(returncode=0)

        values = list_provider_statuses(
            runner=runner,
            which=which,
        )
        self.assertEqual(
            [item["id"] for item in values], ["codex", "claude", "grok"]
        )
        self.assertEqual(resolved, ["codex", "claude", "grok"])
        self.assertEqual(len(commands), 2)
        self.assertIsNone(values[2]["connected"])

    def test_grok_connected_state_is_unknown_without_running_a_probe(self):
        def runner(_argv, **_kwargs):
            raise AssertionError("Grok has no reviewed status command")

        result = provider_status("grok", runner=runner, which=self.which)
        self.assertEqual(
            result,
            {
                "id": "grok",
                "label": "xAI Grok Build",
                "installed": True,
                "connected": None,
            },
        )


class TestProviderCommands(ProviderFixture):
    def test_connect_commands_are_fixed_direct_argv(self):
        codex = provider_connect_command("codex", which=self.which)
        claude = provider_connect_command("claude", which=self.which)
        grok = provider_connect_command("grok", which=self.which)
        self.assertEqual(
            codex.argv,
            (self.executables["codex"], "login", "--device-auth"),
        )
        self.assertEqual(
            claude.argv,
            (self.executables["claude"], "auth", "login", "--claudeai"),
        )
        self.assertEqual(
            grok.argv,
            (self.executables["grok"], "login", "--device-auth"),
        )
        self.assertNotIn(
            "OPENAI_API_KEY",
            subscription_environment(
                "codex", {"OPENAI_API_KEY": "secret", "SAFE": "yes"}
            ),
        )
        self.assertEqual(
            subscription_environment(
                "claude",
                {
                    "ANTHROPIC_API_KEY": "secret",
                    "CLAUDE_CODE_USE_BEDROCK": "1",
                    "CLAUDE_CONFIG_DIR": "/private/cache",
                    "SAFE": "yes",
                },
            ),
            {"CLAUDE_CONFIG_DIR": "/private/cache"},
        )

    def test_agent_commands_are_fixed_and_interactive(self):
        codex = provider_agent_command("codex", which=self.which)
        claude = provider_agent_command("claude", which=self.which)
        grok = provider_agent_command("grok", which=self.which)
        self.assertEqual(
            codex.argv,
            (self.executables["codex"], "--no-alt-screen"),
        )
        self.assertEqual(claude.argv, (self.executables["claude"],))
        self.assertEqual(grok.argv, (self.executables["grok"],))
        self.assertEqual(codex.purpose, "agent")
        self.assertEqual(claude.purpose, "agent")
        self.assertEqual(grok.purpose, "agent")

    def test_environment_copy_never_mutates_caller_mapping(self):
        source = {
            "ANTHROPIC_AUTH_TOKEN": "secret",
            "UNRELATED_SERVICE_SECRET": "also-secret",
            "PATH": "/safe",
        }
        result = subscription_environment("claude", source)
        self.assertEqual(source["ANTHROPIC_AUTH_TOKEN"], "secret")
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", result)
        self.assertNotIn("UNRELATED_SERVICE_SECRET", result)
        self.assertEqual(result["PATH"], "/safe")


if __name__ == "__main__":
    unittest.main(verbosity=2)
