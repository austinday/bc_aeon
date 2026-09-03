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
from unittest import mock

from aeon.core import protected, checkpoint, bootguard
from aeon.selfimprove import scorer, ledger, benchmark, evaluate


class _Chdir:
    """Context manager: run inside a temp cwd so cwd-relative state (ledger,
    boot marker) never touches the real aeon_output."""
    def __enter__(self):
        self._old = os.getcwd()
        self._old_aeon_home = os.environ.get("AEON_HOME")
        self._tmp = tempfile.mkdtemp()
        os.chdir(self._tmp)
        os.environ["AEON_HOME"] = str(Path(self._tmp) / ".aeon")
        return self._tmp
    def __exit__(self, *a):
        os.chdir(self._old)
        if self._old_aeon_home is None:
            os.environ.pop("AEON_HOME", None)
        else:
            os.environ["AEON_HOME"] = self._old_aeon_home


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

    def test_fleet_and_tool_execution_boundaries_are_protected(self):
        root = Path(__file__).resolve().parents[2]
        for relative in (
            "aeon/core/fleet_backend.py",
            "aeon/benchmarks/executor.py",
            "aeon/benchmarks/runner.py",
            "aeon/harnesses/model_proxy.py",
            "aeon/harnesses/opencode_config.py",
            "aeon/harnesses/opencode_mcp.py",
            "aeon/harnesses/opencode_runtime.py",
            "aeon/core/fleet_adapter.py",
            "aeon/core/qwen_runtime.py",
            "aeon/core/qwen_fast_service_adapter.py",
            "aeon/core/qwen_speed_lab_adapter.py",
            "aeon/core/qwen_dflash_training_adapter.py",
            "aeon/core/qwen_full_gdn_quant_adapter.py",
            "aeon/core/tool_resources.py",
            "aeon/core/worker.py",
            "aeon/main.py",
            "aeon/tools/command_fleet_guard.py",
            "aeon/tools/system.py",
            "aeon/tools/jobs.py",
            "aeon/tools/sub_agent.py",
            "aeon/tools/generate_image.py",
            "aeon/tools/generate_video.py",
            "aeon/tools/vision.py",
            "aeon/tools/composite_image.py",
            "aeon/tools/file_io.py",
            "aeon/tools/analyzers/file_analyzer.py",
            "aeon/tools/search.py",
            "aeon/tools/browser.py",
            "aeon/tools/external_expert.py",
            "aeon/tools/start_agent_instance.py",
            "aeon/remote/instances.py",
            "aeon/scripts/qwen_remote_worker.py",
            "aeon/scripts/qwen_speed_lab_worker.py",
            "aeon/scripts/searxng_service.py",
            "aeon/scripts/browser_service.py",
            "aeon/scripts/start_browser.sh",
            "aeon/services/browser/server.py",
        ):
            with self.subTest(relative=relative):
                self.assertTrue(protected.is_protected(str(root / relative)))

        self.assertTrue(
            protected.is_protected(
                str(root.parent / "fleet_compute" / "profiles.d" / "example.json")
            )
        )
        self.assertTrue(
            protected.is_protected("/home/aday/website_hosting/gpu_coord.py")
        )
        self.assertTrue(
            protected.is_protected(
                "/tmp/example-workspace/aeon_output/session/jobs/job/service_receipt.json"
            )
        )
        self.assertTrue(
            protected.is_protected(
                "/tmp/example-workspace/.aeon-command-scratch/unit/receipt.json"
            )
        )

    def test_entire_regression_suite_is_protected(self):
        self.assertTrue(protected.is_protected(__file__))

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
    def test_marker_publication_failure_is_reported_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {"AEON_HOME": temporary}
        ), mock.patch.object(
            bootguard.os, "replace", side_effect=OSError("simulated fsync boundary")
        ):
            self.assertFalse(
                bootguard.mark_pending(
                    "/some/dir", "aeon-ckpt/x", reason="test"
                )
            )
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_malformed_marker_root_is_consumed_without_crashing_startup(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {"AEON_HOME": temporary}
        ):
            marker = bootguard._marker_path()
            marker.write_text("[]", encoding="utf-8")
            marker.chmod(0o600)

            result = bootguard.check_and_recover(print_func=lambda *_args: None)

            self.assertTrue(result["recovered"])
            self.assertFalse(result["restored"])
            self.assertFalse(marker.exists())

    def test_healthy_boot_marker_clear_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {"AEON_HOME": temporary}
        ):
            self.assertTrue(
                bootguard.mark_pending(
                    "/some/dir", "aeon-ckpt/x", reason="test"
                )
            )
            marker = bootguard._marker_path()
            original_unlink = Path.unlink

            def refuse_marker(path, *args, **kwargs):
                if path == marker:
                    raise OSError("simulated read-only state")
                return original_unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", new=refuse_marker):
                self.assertFalse(bootguard.mark_boot_ok())
            self.assertTrue(marker.exists())

    def test_marker_lifecycle(self):
        with _Chdir():
            self.assertEqual(bootguard.check_and_recover(print_func=lambda *a: None)["recovered"], False)
            self.assertTrue(
                bootguard.mark_pending("/some/dir", "aeon-ckpt/x", reason="r")
            )
            self.assertTrue(bootguard._marker_path().exists())
            data = json.loads(bootguard._marker_path().read_text())
            self.assertEqual(data["checkpoint"], "aeon-ckpt/x")
            self.assertTrue(bootguard.mark_boot_ok())
            self.assertFalse(bootguard._marker_path().exists())

    def test_recovery_restores_canonical_source_and_requires_reexec(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            root = Path(td) / "source"
            home.mkdir()
            root.mkdir()
            with mock.patch.dict(os.environ, {"AEON_HOME": str(home)}), mock.patch.object(
                bootguard, "PROJECT_ROOT", root
            ):
                self.assertTrue(
                    bootguard.mark_pending(
                        str(root), "aeon-ckpt/known-good", reason="test"
                    )
                )
                with mock.patch.object(
                    checkpoint,
                    "restore_checkpoint",
                    return_value={"ok": True},
                ) as restore:
                    result = bootguard.check_and_recover(print_func=lambda *a: None)

            restore.assert_called_once_with(root.resolve(), "aeon-ckpt/known-good")
            self.assertTrue(result["recovered"])
            self.assertTrue(result["restored"])
            self.assertTrue(result["restart_required"])
            self.assertEqual(result["aeon_code_dir"], str(root.resolve()))
            self.assertNotIn("reinstalled", result)

    def test_recovery_refuses_noncanonical_source_without_restoring(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            canonical = Path(td) / "canonical"
            other = Path(td) / "other"
            home.mkdir()
            canonical.mkdir()
            other.mkdir()
            with mock.patch.dict(os.environ, {"AEON_HOME": str(home)}), mock.patch.object(
                bootguard, "PROJECT_ROOT", canonical
            ):
                self.assertTrue(
                    bootguard.mark_pending(str(other), "aeon-ckpt/wrong-tree")
                )
                with mock.patch.object(checkpoint, "restore_checkpoint") as restore:
                    result = bootguard.check_and_recover(print_func=lambda *a: None)

            restore.assert_not_called()
            self.assertTrue(result["recovered"])
            self.assertFalse(result["restored"])
            self.assertFalse(result["restart_required"])


class TestBenchmarkTasks(unittest.TestCase):
    def test_candidate_evaluation_is_fail_closed_before_copy_or_process(self):
        with mock.patch.object(evaluate, "_make_sandbox") as sandbox, mock.patch.object(
            evaluate.subprocess, "run"
        ) as run:
            with self.assertRaisesRegex(
                evaluate.CandidateEvaluationBoundaryUnavailable,
                "masked-home",
            ):
                evaluate.evaluate(root=Path(__file__).resolve().parents[2])
        sandbox.assert_not_called()
        run.assert_not_called()

    def test_direct_candidate_task_runner_is_fail_closed_before_process(self):
        with mock.patch.object(evaluate.subprocess, "run") as run:
            result = evaluate._run_one(Path.cwd(), "tools_import_clean")
        self.assertFalse(result["passed"])
        self.assertIn("masked-home", result["detail"])
        run.assert_not_called()

    def test_candidate_copy_keeps_nested_package_data_but_ignores_root_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "candidate"
            (root / "data").mkdir(parents=True)
            (root / "data" / "large.bin").write_bytes(b"root-data")
            (root / "aeon" / "core" / "data").mkdir(parents=True)
            (root / "aeon" / "core" / "data" / "required.json").write_text(
                "{}", encoding="utf-8"
            )
            copied, cleanup = evaluate._make_sandbox(root)
            try:
                self.assertFalse((copied / "data").exists())
                self.assertTrue(
                    (copied / "aeon" / "core" / "data" / "required.json").is_file()
                )
            finally:
                cleanup()

    def test_candidate_copy_never_falls_back_to_in_place(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            evaluate.shutil, "copytree", side_effect=OSError("synthetic copy failure")
        ):
            with self.assertRaisesRegex(OSError, "synthetic copy failure"):
                evaluate._make_sandbox(Path(temporary))

    def test_candidate_copy_rejects_escaping_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "candidate"
            root.mkdir()
            (root / "escape").symlink_to(Path(temporary).parent)
            with self.assertRaisesRegex(
                evaluate.CandidateEvaluationBoundaryUnavailable,
                "escaping/unresolved symlink",
            ):
                evaluate._make_sandbox(root)

    def test_all_deterministic_tasks_pass_in_process(self):
        for tid in benchmark.deterministic_ids():
            passed, detail, _metric = benchmark.run_task(tid)
            self.assertTrue(passed, f"benchmark task {tid} failed: {detail}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
