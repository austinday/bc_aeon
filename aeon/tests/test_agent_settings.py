import unittest

from aeon.core.model_catalog import QWEN38_MODEL_NAME
from aeon.remote.agent_settings import (
    AgentSettingsError,
    normalize_settings,
    public_catalog,
)


class AgentSettingsTests(unittest.TestCase):
    def test_aeon_is_bound_to_the_validated_release(self):
        self.assertEqual(
            normalize_settings("aeon", model=QWEN38_MODEL_NAME, effort=None),
            (QWEN38_MODEL_NAME, ""),
        )
        with self.assertRaises(AgentSettingsError):
            normalize_settings("aeon", model="another-model", effort="")

    def test_codex_and_claude_are_strict_allowlists(self):
        self.assertEqual(
            normalize_settings("codex", model="gpt-5.6-terra", effort="xhigh"),
            ("gpt-5.6-terra", "xhigh"),
        )
        self.assertEqual(
            normalize_settings("claude", model="sonnet", effort="max"),
            ("sonnet", "max"),
        )
        for kind, model, effort in (
            ("codex", "$(touch /tmp/no)", "high"),
            ("claude", "sonnet", "unlimited"),
            ("grok", "grok-unknown", ""),
        ):
            with self.subTest(kind=kind, model=model, effort=effort):
                with self.assertRaises(AgentSettingsError):
                    normalize_settings(kind, model=model, effort=effort)

    def test_grok_effort_stays_provider_default(self):
        self.assertEqual(
            normalize_settings("grok", model="grok-4.5", effort=None),
            ("grok-4.5", ""),
        )
        with self.assertRaises(AgentSettingsError):
            normalize_settings("grok", model="grok-4.5", effort="high")

    def test_public_catalog_contains_no_command_fragments(self):
        payload = public_catalog("codex")
        self.assertTrue(payload["model_editable"])
        self.assertIn(
            {"id": "gpt-5.6-luna", "label": "gpt-5.6-luna"},
            payload["models"],
        )
        self.assertNotIn("argv", payload)
        self.assertNotIn("environment", payload)


if __name__ == "__main__":
    unittest.main()
