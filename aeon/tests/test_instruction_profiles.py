"""Hermetic tests for locally known, versioned instruction profiles."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from aeon.remote.instruction_profiles import (
    AEON_DIRECTIVE_FILES,
    MAX_INSTRUCTION_BYTES,
    InstructionConflict,
    InstructionNotFound,
    InstructionProfileError,
    InstructionProfileService,
)
from aeon.remote.store import RemoteStore


class InstructionProfileFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.project_root = self.base / "bc_aeon"
        self.prompt_root = self.project_root / "aeon" / "core" / "prompts"
        self.prompt_root.mkdir(parents=True)
        for filename in AEON_DIRECTIVE_FILES:
            (self.prompt_root / filename).write_text(
                f"known Aeon text from {filename}\n", encoding="utf-8"
            )
        self.allowed_root = self.base / "workspaces"
        self.workspace = self.allowed_root / "project" / "nested"
        self.workspace.mkdir(parents=True)
        self.store = RemoteStore(self.base / "state" / "remote.sqlite3")
        self.service = InstructionProfileService(
            self.store,
            project_root=self.project_root,
            allowed_roots=(self.allowed_root,),
        )

    def tearDown(self):
        self.temporary.cleanup()

    def create_instance(
        self, instance_id: str, kind: str = "aeon", *, shell_backed: bool = False
    ) -> None:
        now = time.time()
        self.store.create_instance(
            {
                "id": instance_id,
                "kind": kind,
                "shell_backed": int(shell_backed),
                "name": f"Agent {instance_id}",
                "tmux_name": f"agent-{instance_id}",
                "workspace": str(self.workspace),
                "objective": "",
                "max_iterations": None,
                "model": None,
                "status": "created",
                "desired_state": "running",
                "created_at": now,
                "updated_at": now,
                "last_started_at": None,
                "last_error": "",
                "created_by": "admin",
                "launch_origin": "web",
            }
        )


class TestKnownInstructionDiscovery(InstructionProfileFixture):
    def test_aeon_discovery_reads_only_fixed_directive_files(self):
        result = self.service.discover_known_instructions("aeon", self.workspace)

        self.assertEqual(result["source_kind"], "aeon_fixed")
        self.assertEqual(len(result["documents"]), len(AEON_DIRECTIVE_FILES))
        self.assertIn("known Aeon text from core_directives.txt", result["content"])
        self.assertIn("Locally known instructions only", result["disclosure"])

    def test_codex_discovery_is_ancestor_only_and_ordered_general_to_specific(self):
        (self.allowed_root / "AGENTS.md").write_text("root rules", encoding="utf-8")
        (self.allowed_root / "project" / "AGENTS.md").write_text(
            "project rules", encoding="utf-8"
        )
        # A sibling is under the allowed root but is not applicable to this workspace.
        sibling = self.allowed_root / "sibling"
        sibling.mkdir()
        (sibling / "AGENTS.md").write_text("sibling secret", encoding="utf-8")

        result = self.service.discover_known_instructions("codex", self.workspace)

        self.assertEqual(
            [item["source_ref"] for item in result["documents"]],
            ["AGENTS.md", "project/AGENTS.md"],
        )
        self.assertLess(result["content"].index("root rules"), result["content"].index("project rules"))
        self.assertNotIn("sibling secret", result["content"])

    def test_discovery_rejects_outside_workspace_symlink_and_oversized_source(self):
        outside = self.base / "outside"
        outside.mkdir()
        with self.assertRaises(InstructionProfileError):
            self.service.discover_known_instructions("codex", outside)

        secret = self.base / "not-an-instruction-file"
        secret.write_text("must not be read", encoding="utf-8")
        (self.workspace / "AGENTS.md").symlink_to(secret)
        with self.assertRaises(InstructionNotFound):
            self.service.discover_known_instructions("codex", self.workspace)
        (self.workspace / "AGENTS.md").unlink()

        (self.workspace / "AGENTS.md").write_bytes(b"x" * (MAX_INSTRUCTION_BYTES + 1))
        with self.assertRaisesRegex(InstructionProfileError, "exceeds"):
            self.service.discover_known_instructions("codex", self.workspace)

    def test_grok_uses_only_its_documented_agents_file_convention(self):
        (self.workspace / "AGENTS.md").write_text("grok project rules", encoding="utf-8")

        result = self.service.discover_known_instructions("grok", self.workspace)

        self.assertEqual(result["source_kind"], "workspace")
        self.assertEqual(result["documents"][0]["source_ref"], "project/nested/AGENTS.md")
        self.assertEqual(result["content"], "grok project rules")


class TestInstructionProfilePersistence(InstructionProfileFixture):
    def test_profile_versions_are_immutable_numbered_and_content_is_not_listed(self):
        profile = self.service.create_profile(
            agent_kind="aeon", name="Default directives", actor="admin"
        )
        first = self.service.save_version(
            profile["id"], label="Imported", content="first body", actor="admin"
        )
        second = self.service.save_version(
            profile["id"], label="Refined", content="second body", actor="admin"
        )

        self.assertEqual((first["version_number"], second["version_number"]), (1, 2))
        self.assertEqual(self.service.get_version(first["id"])["content"], "first body")
        summaries = self.service.list_versions(profile["id"])
        self.assertEqual([item["version_number"] for item in summaries], [2, 1])
        self.assertTrue(all("content" not in item for item in summaries))
        self.assertEqual(self.service.get_profile(profile["id"])["latest_version_id"], second["id"])

        with self.assertRaises(InstructionConflict):
            self.service.create_profile(
                agent_kind="aeon", name="default DIRECTIVES", actor="admin"
            )

    def test_source_import_creates_profile_and_version_in_one_transaction(self):
        imported = self.service.create_profile_from_known_source(
            agent_kind="aeon",
            name="Shipped Aeon directives",
            workspace=self.workspace,
            label="Imported from local source",
            actor="admin",
        )

        self.assertEqual(imported["version"]["version_number"], 1)
        self.assertEqual(imported["version"]["source_kind"], "aeon_fixed")
        self.assertNotIn(imported["version"]["content"], str(self.store.recent_audit()))

    def test_instruction_text_limit_is_utf8_bytes_and_nul_is_rejected(self):
        profile = self.service.create_profile(
            agent_kind="aeon", name="Bounded", actor="admin"
        )
        with self.assertRaisesRegex(InstructionProfileError, "UTF-8 bytes"):
            self.service.save_version(
                profile["id"],
                label="Too large",
                content="é" * (MAX_INSTRUCTION_BYTES // 2 + 1),
                actor="admin",
            )
        with self.assertRaisesRegex(InstructionProfileError, "NUL"):
            self.service.save_version(
                profile["id"], label="Bad", content="before\x00after", actor="admin"
            )

    def test_registry_and_wal_live_beneath_private_directory(self):
        mode = os.stat(self.store.path).st_mode & 0o777
        parent_mode = os.stat(self.store.path.parent).st_mode & 0o777
        self.assertEqual(mode, 0o600)
        self.assertEqual(parent_mode, 0o700)


class TestInstanceInstructionBindings(InstructionProfileFixture):
    def setUp(self):
        super().setUp()
        self.create_instance("aeon-one")
        profile = self.service.create_profile(
            agent_kind="aeon", name="Aeon role base", actor="admin"
        )
        self.v1 = self.service.save_version(
            profile["id"], label="v1", content="base one", actor="admin"
        )
        self.v2 = self.service.save_version(
            profile["id"], label="v2", content="base two", actor="admin"
        )

    def test_blank_local_role_and_exact_applied_vs_pending_revisions(self):
        blank = self.service.get_instance_binding("aeon-one")
        self.assertEqual(blank["desired_local_content"], "")
        self.assertFalse(blank["pending"])

        selected = self.service.select_profile_version("aeon-one", self.v1["id"])
        self.assertTrue(selected["base_pending"])
        captured = self.service.launch_snapshot("aeon-one")
        self.assertEqual(captured["profile_content"], "base one")
        self.assertEqual(captured["local_revision"], 0)

        # Edits made while the launcher applies its captured snapshot remain pending.
        self.service.select_profile_version("aeon-one", self.v2["id"])
        edited = self.service.save_local_role(
            "aeon-one", content="You manage releases.", expected_revision=0, actor="admin"
        )
        self.assertEqual(edited["desired_local_revision"], 1)
        applied_old = self.service.mark_applied(
            "aeon-one", profile_version_id=captured["profile_version_id"], local_revision=0
        )
        self.assertEqual(applied_old["applied_profile_version"]["id"], self.v1["id"])
        self.assertTrue(applied_old["base_pending"])
        self.assertTrue(applied_old["local_pending"])

        applied_current = self.service.mark_applied(
            "aeon-one", profile_version_id=self.v2["id"], local_revision=1
        )
        self.assertFalse(applied_current["pending"])
        self.assertEqual(applied_current["applied_local_content"], "You manage releases.")

    def test_local_role_is_optimistic_immutable_and_can_load_an_old_revision(self):
        first = self.service.save_local_role(
            "aeon-one", content="First role", expected_revision=0, actor="admin"
        )
        self.assertEqual(first["desired_local_revision"], 1)
        second = self.service.save_local_role(
            "aeon-one", content="Second role", expected_revision=1, actor="admin"
        )
        self.assertEqual(second["desired_local_revision"], 2)

        with self.assertRaisesRegex(InstructionConflict, "current revision is 2"):
            self.service.save_local_role(
                "aeon-one", content="stale browser edit", expected_revision=1, actor="admin"
            )
        self.assertEqual(self.service.get_local_role_version("aeon-one", 1)["content"], "First role")
        restored = self.service.select_local_role_version("aeon-one", 1)
        self.assertEqual(restored["desired_local_content"], "First role")
        self.assertEqual(
            [item["revision"] for item in self.service.list_local_role_versions("aeon-one")],
            [2, 1],
        )

    def test_profile_kind_must_match_and_only_shell_terminals_receive_local_identity(self):
        self.create_instance("codex-one", kind="codex")
        self.create_instance("terminal-one", kind="terminal")
        self.create_instance("shell-terminal", kind="terminal", shell_backed=True)
        with self.assertRaisesRegex(InstructionProfileError, "does not match"):
            self.service.select_profile_version("codex-one", self.v1["id"])
        with self.assertRaisesRegex(InstructionProfileError, "only to agent"):
            self.service.save_local_role(
                "terminal-one", content="role", expected_revision=0, actor="admin"
            )
        local = self.service.save_local_role(
            "shell-terminal",
            content="role prepared before activation",
            expected_revision=0,
            actor="admin",
        )
        self.assertEqual(local["agent_kind"], "terminal")
        self.assertEqual(local["desired_local_content"], "role prepared before activation")
        self.assertIsNone(local["desired_profile_version"])
        with self.assertRaisesRegex(
            InstructionProfileError, "unavailable until an agent is active"
        ):
            self.service.select_profile_version("shell-terminal", None)

    def test_terminal_identity_hides_but_preserves_profile_for_same_kind_activation(self):
        self.service.select_profile_version("aeon-one", self.v1["id"])
        self.service.save_local_role(
            "aeon-one",
            content="Durable local identity",
            expected_revision=0,
            actor="admin",
        )
        self.store.update_instance(
            "aeon-one", kind="terminal", shell_backed=1, last_agent_kind="aeon"
        )

        terminal_binding = self.service.get_instance_binding("aeon-one")
        self.assertEqual(terminal_binding["agent_kind"], "terminal")
        self.assertIsNone(terminal_binding["desired_profile_version"])
        self.assertFalse(terminal_binding["base_pending"])
        self.assertEqual(terminal_binding["desired_local_content"], "Durable local identity")

        launch = self.service.launch_snapshot_for_agent_kind(
            "aeon-one", agent_kind="aeon", preserve_profile=True
        )
        self.assertEqual(launch["profile_version_id"], self.v1["id"])
        self.assertEqual(launch["local_content"], "Durable local identity")

    def test_instance_delete_cascades_private_binding_and_local_versions(self):
        self.service.select_profile_version("aeon-one", self.v1["id"])
        self.service.save_local_role(
            "aeon-one", content="ephemeral role", expected_revision=0, actor="admin"
        )
        self.store.delete_instance("aeon-one")
        with self.store._connect() as conn:
            bindings = conn.execute(
                "SELECT COUNT(*) FROM instance_instruction_bindings"
            ).fetchone()[0]
            locals_count = conn.execute(
                "SELECT COUNT(*) FROM instance_local_instruction_versions"
            ).fetchone()[0]
        self.assertEqual(bindings, 0)
        self.assertEqual(locals_count, 0)


if __name__ == "__main__":
    unittest.main()
