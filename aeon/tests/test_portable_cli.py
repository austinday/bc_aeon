from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PortableCliTests(unittest.TestCase):
    def test_help_and_version_are_workspace_clean(self):
        source_root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(source_root)
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            for arguments, expected in ((["--help"], "Run it from"), (["--version"], "Aeon 0.2.0")):
                with self.subTest(arguments=arguments):
                    result = subprocess.run(
                        [sys.executable, "-m", "aeon", *arguments],
                        cwd=workspace,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        timeout=15,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(expected, result.stdout)
                    self.assertNotIn("Scanning for ghost", result.stdout + result.stderr)
                    self.assertEqual(list(workspace.iterdir()), [])

    def test_invalid_model_is_rejected_before_runtime_import(self):
        source_root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(source_root)
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            result = subprocess.run(
                [sys.executable, "-m", "aeon", "--model", "some-other-model"],
                cwd=workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("only the primary model", result.stderr)
            self.assertNotIn("Scanning for ghost", result.stdout + result.stderr)
            self.assertEqual(list(workspace.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
