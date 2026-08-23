"""Focused tests for the private per-instance instruction runtime handoff."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aeon.core.runtime_instructions import (
    GROK_AGENT_PROFILE_FILENAME,
    PROVIDER_INSTRUCTIONS_FILENAME,
    RUNTIME_INSTRUCTIONS_ENV,
    RUNTIME_INSTRUCTIONS_FILENAME,
    RuntimeInstructionError,
    format_runtime_instruction_layers,
    load_runtime_instructions,
    materialize_provider_instruction_text,
    materialize_grok_agent_profile,
    materialize_runtime_instructions,
    runtime_instruction_layers_from_snapshot,
)
from aeon.core.worker import Worker


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _snapshot(
    *,
    profile: str = "Base profile instructions.",
    local: str = "This instance owns release coordination.",
    profile_id: str | None = "ipv-0123456789abcdef",
    revision: int = 1,
    agent_kind: str = "aeon",
) -> dict:
    return {
        "instance_id": "782bde8e-6593-4d1e-ad81-dc915dd5af31",
        "agent_kind": agent_kind,
        "profile_version_id": profile_id,
        "profile_content": profile,
        "profile_content_sha256": _digest(profile),
        "local_revision": revision,
        "local_content": local,
        "local_content_sha256": _digest(local),
        # Launch snapshots include UI disclosure metadata; it is not materialized.
        "disclosure": "Locally known instructions only.",
    }


class RuntimeInstructionFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.private_root = self.base / "instances"
        self.private_root.mkdir(mode=0o700)
        os.chmod(self.private_root, 0o700)
        self.instance_dir = self.private_root / "instance-one"

    def tearDown(self):
        self.temporary.cleanup()


class TestRuntimeInstructionMaterialization(RuntimeInstructionFixture):
    def test_round_trip_is_atomic_owner_private_and_repr_hides_bodies(self):
        snapshot = _snapshot(profile="profile π", local="local role λ")

        path = materialize_runtime_instructions(snapshot, self.instance_dir)
        layers = load_runtime_instructions(path)

        self.assertEqual(path, self.instance_dir / RUNTIME_INSTRUCTIONS_FILENAME)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.instance_dir.stat().st_mode), 0o700)
        self.assertEqual(path.stat().st_uid, os.geteuid())
        self.assertEqual(layers.profile_content, "profile π")
        self.assertEqual(layers.local_content, "local role λ")
        self.assertEqual(
            layers.applied_identity,
            {"profile_version_id": snapshot["profile_version_id"], "local_revision": 1},
        )
        self.assertNotIn("profile π", repr(layers))
        self.assertNotIn("local role λ", repr(layers))
        self.assertEqual(list(self.instance_dir.glob("*.tmp")), [])

    def test_replacing_a_snapshot_is_coherent_and_immediately_visible(self):
        path = materialize_runtime_instructions(
            _snapshot(profile="profile version one"), self.instance_dir
        )
        first_inode = path.stat().st_ino

        materialize_runtime_instructions(
            _snapshot(profile="profile version two", profile_id="ipv-fedcba9876543210"),
            self.instance_dir,
        )
        layers = load_runtime_instructions(path)

        self.assertEqual(layers.profile_content, "profile version two")
        self.assertEqual(layers.profile_version_id, "ipv-fedcba9876543210")
        self.assertNotEqual(path.stat().st_ino, first_inode)

    def test_existing_symlink_target_and_symlink_directory_fail_closed(self):
        self.instance_dir.mkdir(mode=0o700)
        secret = self.base / "unrelated.txt"
        secret.write_text("leave unchanged", encoding="utf-8")
        target = self.instance_dir / RUNTIME_INSTRUCTIONS_FILENAME
        target.symlink_to(secret)

        with self.assertRaisesRegex(RuntimeInstructionError, "regular file"):
            materialize_runtime_instructions(_snapshot(), self.instance_dir)
        self.assertEqual(secret.read_text(encoding="utf-8"), "leave unchanged")

        target.unlink()
        real_directory = self.private_root / "real-instance"
        real_directory.mkdir(mode=0o700)
        alias = self.private_root / "linked-instance"
        alias.symlink_to(real_directory, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeInstructionError, "symbolic link"):
            materialize_runtime_instructions(_snapshot(), alias)

    def test_wrong_modes_invalid_json_and_digest_mismatch_fail_closed(self):
        path = materialize_runtime_instructions(_snapshot(), self.instance_dir)
        os.chmod(path, 0o640)
        with self.assertRaisesRegex(RuntimeInstructionError, "mode 0600"):
            load_runtime_instructions(path)

        os.chmod(path, 0o600)
        path.write_text("not json", encoding="utf-8")
        os.chmod(path, 0o600)
        with self.assertRaisesRegex(RuntimeInstructionError, "valid JSON"):
            load_runtime_instructions(path)

        bad = _snapshot()
        bad["profile_content_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeInstructionError, "digest does not match"):
            materialize_runtime_instructions(bad, self.private_root / "bad-digest")

        path.write_text("{}", encoding="utf-8")
        os.chmod(path, 0o600)
        os.chmod(self.instance_dir, 0o750)
        with self.assertRaisesRegex(RuntimeInstructionError, "mode 0700"):
            load_runtime_instructions(path)

    def test_absent_environment_is_an_empty_optional_layer(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(RUNTIME_INSTRUCTIONS_ENV, None)
            layers = load_runtime_instructions()
        self.assertTrue(layers.is_empty)
        self.assertEqual(layers.agent_kind, "aeon")
        self.assertEqual(format_runtime_instruction_layers(layers), "")

    def test_expected_instance_identity_prevents_cross_instance_assignment(self):
        path = materialize_runtime_instructions(_snapshot(), self.instance_dir)
        with self.assertRaisesRegex(RuntimeInstructionError, "another instance"):
            load_runtime_instructions(path, expected_instance_id="different-instance")
        with self.assertRaisesRegex(RuntimeInstructionError, "another agent kind"):
            load_runtime_instructions(path, expected_agent_kind="codex")

    def test_provider_overlay_is_generic_atomic_private_text(self):
        snapshot = _snapshot(
            profile="Codex workspace instructions, exactly.",
            local="Own the deployment review.",
            agent_kind="codex",
        )
        layers = runtime_instruction_layers_from_snapshot(snapshot)
        rendered = format_runtime_instruction_layers(layers)

        path = materialize_provider_instruction_text(snapshot, self.instance_dir)

        self.assertEqual(path, self.instance_dir / PROVIDER_INSTRUCTIONS_FILENAME)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(path.read_text(encoding="utf-8"), rendered)
        self.assertIn("locally selected instruction layer", rendered)
        self.assertIn("Codex workspace instructions, exactly.", rendered)
        self.assertIn("Own the deployment review.", rendered)
        self.assertLess(
            rendered.index("Codex workspace instructions, exactly."),
            rendered.index("Own the deployment review."),
        )
        self.assertNotIn("Aeon", rendered)
        self.assertNotIn('"profile_content"', rendered)

    def test_provider_overlay_rejects_aeon_and_existing_symlink(self):
        with self.assertRaisesRegex(RuntimeInstructionError, "provider agent"):
            materialize_provider_instruction_text(_snapshot(), self.instance_dir)

        self.instance_dir.mkdir(mode=0o700)
        unrelated = self.base / "unrelated-provider.txt"
        unrelated.write_text("unchanged", encoding="utf-8")
        (self.instance_dir / PROVIDER_INSTRUCTIONS_FILENAME).symlink_to(unrelated)
        with self.assertRaisesRegex(RuntimeInstructionError, "regular file"):
            materialize_provider_instruction_text(
                _snapshot(agent_kind="claude"), self.instance_dir
            )
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "unchanged")

    def test_grok_profile_is_private_file_backed_extend_definition(self):
        snapshot = _snapshot(
            profile="Grok shared overlay sentinel.",
            local="Grok private identity sentinel.",
            agent_kind="grok",
        )

        path = materialize_grok_agent_profile(snapshot, self.instance_dir)

        self.assertEqual(path, self.instance_dir / GROK_AGENT_PROFILE_FILENAME)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        content = path.read_text(encoding="utf-8")
        self.assertRegex(
            content,
            r"\A---\nname: nexus-[0-9a-f]{20}\n"
            r"description: Nexus-managed persistent instruction overlay\n---\n\n",
        )
        self.assertIn("Grok shared overlay sentinel.", content)
        self.assertIn("Grok private identity sentinel.", content)
        self.assertNotIn("promptMode:", content)

    def test_grok_profile_rejects_other_provider_and_existing_symlink(self):
        with self.assertRaisesRegex(RuntimeInstructionError, "Grok snapshot"):
            materialize_grok_agent_profile(
                _snapshot(agent_kind="codex"), self.instance_dir
            )

        self.instance_dir.mkdir(mode=0o700)
        unrelated = self.base / "unrelated-grok-profile.md"
        unrelated.write_text("unchanged", encoding="utf-8")
        (self.instance_dir / GROK_AGENT_PROFILE_FILENAME).symlink_to(unrelated)
        with self.assertRaisesRegex(RuntimeInstructionError, "regular file"):
            materialize_grok_agent_profile(
                _snapshot(agent_kind="grok"), self.instance_dir
            )
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "unchanged")


class TestAeonWorkerRuntimeInstructions(RuntimeInstructionFixture):
    @staticmethod
    def _worker_shell() -> Worker:
        worker = Worker.__new__(Worker)
        worker.base_directives = "BUILT-IN BASE"
        worker.docker_directives = "BUILT-IN DOCKER"
        worker.important_reminders = ""
        worker._get_skills_description = lambda: "BUILT-IN SKILLS"
        worker.current_plan = "Current plan"
        worker._stuck_banner = ""
        worker.active_skill = None
        worker.last_observation = "No observation"
        return worker

    def test_both_prompt_paths_append_profile_then_instance_role(self):
        profile = "EXACT PROFILE BODY\nsecond profile line"
        local = "EXACT LOCAL BODY\nsecond local line"
        path = materialize_runtime_instructions(
            _snapshot(profile=profile, local=local), self.instance_dir
        )
        worker = self._worker_shell()

        with mock.patch.dict(os.environ, {RUNTIME_INSTRUCTIONS_ENV: str(path)}):
            system_prompt = worker._build_system_message("Objective", "tools", "")
            compatibility_prompt = worker._build_primary_agent_context(
                "tools",
                "project tree",
                "stats",
                "memories",
                "Objective",
                "open files",
                "",
                "attempt log",
            )

        for prompt in (system_prompt, compatibility_prompt):
            self.assertIn(profile, prompt)
            self.assertIn(local, prompt)
            self.assertLess(prompt.index(profile), prompt.index(local))
            self.assertLess(prompt.index(local), prompt.index("\n**OBJECTIVE**\nObjective"))

    def test_prompt_build_reloads_updates_and_configured_corruption_is_fatal(self):
        path = materialize_runtime_instructions(
            _snapshot(profile="first live profile"), self.instance_dir
        )
        worker = self._worker_shell()

        with mock.patch.dict(os.environ, {RUNTIME_INSTRUCTIONS_ENV: str(path)}):
            first = worker._build_system_message("Objective", "tools", "")
            materialize_runtime_instructions(
                _snapshot(profile="updated live profile", profile_id="ipv-update123"),
                self.instance_dir,
            )
            second = worker._build_system_message("Objective", "tools", "")
            self.assertIn("first live profile", first)
            self.assertNotIn("first live profile", second)
            self.assertIn("updated live profile", second)

            path.write_text("corrupt", encoding="utf-8")
            os.chmod(path, 0o600)
            with self.assertRaises(RuntimeInstructionError):
                worker._build_system_message("Objective", "tools", "")


if __name__ == "__main__":
    unittest.main()
