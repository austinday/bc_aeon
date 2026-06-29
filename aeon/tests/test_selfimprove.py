"""Unit tests for the self-modification / self-improvement substrate.

Hermetic: temp dirs and temp git repos only; no models, network, or pip installs.
Imported into test_core so they run as part of the restart-time regression gate.
"""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from aeon.core import protected, checkpoint, bootguard
from aeon.selfimprove import scorer, ledger, benchmark


class _Chdir:
    """Context manager: run inside a temp cwd so cwd-relative state (ledger,
    boot marker) never touches the real aeon_output."""
    def __enter__(self):
        self._old = os.getcwd()
        self._tmp = tempfile.mkdtemp()
        os.chdir(self._tmp)
        return self._tmp
    def __exit__(self, *a):
        os.chdir(self._old)


def _init_repo(d):
    def git(*a):
        return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True)
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    return git


class TestProtectedGuard(unittest.TestCase):
    def test_protects_its_own_machinery(self):
        self.assertTrue(protected.is_protected(protected.__file__))
        self.assertTrue(protected.is_protected(checkpoint.__file__))

    def test_allows_ordinary_file(self):
        self.assertFalse(protected.is_protected("/tmp/some_user_script.py"))
        self.assertIsNone(protected.guard("/tmp/some_user_script.py"))

    def test_guard_blocks_then_override_allows(self):
        self.assertIsNotNone(protected.guard(protected.__file__))
        os.environ[protected.OVERRIDE_ENV] = "1"
        try:
            self.assertIsNone(protected.guard(protected.__file__))
        finally:
            os.environ.pop(protected.OVERRIDE_ENV, None)


class TestCheckpoint(unittest.TestCase):
    def test_create_restore_roundtrip(self):
        d = tempfile.mkdtemp()
        git = _init_repo(d)
        pkg = Path(d) / "aeon"
        pkg.mkdir()
        f = pkg / "f.txt"
        f.write_text("v1\n")
        git("add", "-A")
        git("commit", "-qm", "init")

        ck = checkpoint.create_checkpoint(d, "test")
        self.assertTrue(ck["ok"], ck)
        # modify + add a new file, then restore
        f.write_text("v2\n")
        (pkg / "added.txt").write_text("new\n")
        res = checkpoint.restore_checkpoint(d, ck["tag"])
        self.assertTrue(res["ok"], res)
        self.assertEqual(f.read_text().strip(), "v1")
        self.assertFalse((pkg / "added.txt").exists(),
                         "file added after the checkpoint should be removed on restore")

    def test_list_and_nongit(self):
        d = tempfile.mkdtemp()
        self.assertFalse(checkpoint.is_git_repo(d))
        self.assertEqual(checkpoint.create_checkpoint(d, "x")["ok"], False)
        git = _init_repo(d)
        (Path(d) / "aeon").mkdir()
        (Path(d) / "aeon" / "f.txt").write_text("a\n")
        git("add", "-A")
        git("commit", "-qm", "i")
        ck = checkpoint.create_checkpoint(d, "one")
        tags = [r["tag"] for r in checkpoint.list_checkpoints(d)]
        self.assertIn(ck["tag"], tags)


class TestScorer(unittest.TestCase):
    def test_scorecard_math(self):
        results = [
            {"task": "a", "passed": True}, {"task": "b", "passed": True},
            {"task": "c", "passed": False}, {"task": "d", "passed": True},
        ]
        sc = scorer.build_scorecard(results)
        self.assertEqual(sc["passed"], 3)
        self.assertEqual(sc["total"], 4)
        self.assertEqual(sc["score"], 0.75)

    def test_compare_accept_improve(self):
        base = scorer.build_scorecard([{"task": "a", "passed": False}])
        cand = scorer.build_scorecard([{"task": "a", "passed": True}])
        cmp = scorer.compare(cand, base)
        self.assertEqual(cmp["decision"], "accept")

    def test_compare_rejects_regression(self):
        base = scorer.build_scorecard([{"task": "a", "passed": True}, {"task": "b", "passed": True}])
        cand = scorer.build_scorecard([{"task": "a", "passed": False}, {"task": "b", "passed": True}])
        cmp = scorer.compare(cand, base)
        self.assertEqual(cmp["decision"], "reject")
        self.assertIn("a", cmp["regressions"])

    def test_compare_no_baseline(self):
        cand = scorer.build_scorecard([{"task": "a", "passed": True}])
        self.assertEqual(scorer.compare(cand, None)["decision"], "accept")


class TestLedger(unittest.TestCase):
    def test_record_and_baseline(self):
        with _Chdir():
            ledger.record({"kind": "benchmark", "scorecard": {"score": 0.5, "tasks": []}})
            ledger.record({"kind": "benchmark", "scorecard": {"score": 0.9, "tasks": []}})
            self.assertEqual(len(ledger.read_all()), 2)
            self.assertEqual(ledger.last_scorecard()["score"], 0.9)
            self.assertIn("experiment", ledger.summary())


class TestBootguard(unittest.TestCase):
    def test_marker_lifecycle(self):
        with _Chdir():
            self.assertEqual(bootguard.check_and_recover(print_func=lambda *a: None)["recovered"], False)
            bootguard.mark_pending("/some/dir", "aeon-ckpt/x", reason="r")
            self.assertTrue(bootguard._marker_path().exists())
            data = json.loads(bootguard._marker_path().read_text())
            self.assertEqual(data["checkpoint"], "aeon-ckpt/x")
            bootguard.mark_boot_ok()
            self.assertFalse(bootguard._marker_path().exists())


class TestBenchmarkTasks(unittest.TestCase):
    def test_all_deterministic_tasks_pass_in_process(self):
        for tid in benchmark.deterministic_ids():
            passed, detail, _metric = benchmark.run_task(tid)
            self.assertTrue(passed, f"benchmark task {tid} failed: {detail}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
