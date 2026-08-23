from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aeon.core.workspace_instructions import (
    WorkspaceInstructionError,
    discover_workspace_instructions,
    format_workspace_instructions,
)


class WorkspaceInstructionTests(unittest.TestCase):
    def test_global_to_local_order_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            workspace = home / "project" / "nested"
            workspace.mkdir(parents=True)
            (home / "AGENTS.md").write_text("global rule\n", encoding="utf-8")
            (home / "project" / "AGENTS.md").write_text(
                "project rule\n", encoding="utf-8"
            )
            (workspace / "AGENTS.md").write_text("local rule\n", encoding="utf-8")

            documents = discover_workspace_instructions(workspace, home=home)
            self.assertEqual(
                [item.content.strip() for item in documents],
                ["global rule", "project rule", "local rule"],
            )
            rendered = format_workspace_instructions(documents)
            self.assertLess(rendered.index("global rule"), rendered.index("local rule"))
            self.assertIn(str(workspace / "AGENTS.md"), rendered)

    def test_global_file_is_not_duplicated_at_home_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            (home / "AGENTS.md").write_text("once", encoding="utf-8")
            documents = discover_workspace_instructions(home, home=home)
            self.assertEqual(len(documents), 1)

    def test_symlinked_instruction_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            target = home / "real.md"
            target.write_text("do not follow", encoding="utf-8")
            (home / "AGENTS.md").symlink_to(target)
            with self.assertRaises(WorkspaceInstructionError):
                discover_workspace_instructions(home, home=home)

    def test_worker_prompt_appends_workspace_layer(self):
        from aeon.core import worker as worker_module

        with patch.object(
            worker_module, "load_workspace_instruction_section", return_value="\nworkspace-marker"
        ), patch.object(
            worker_module, "format_aeon_runtime_instructions", return_value="\nprivate-marker"
        ), patch.object(worker_module, "load_runtime_instructions", return_value=object()):
            rendered = worker_module.Worker._runtime_instruction_section()
        self.assertEqual(rendered, "\nprivate-marker\nworkspace-marker")


if __name__ == "__main__":
    unittest.main()
