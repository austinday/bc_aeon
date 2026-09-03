"""Hermetic restart lifecycle tests: no model, broker, service, or exec calls."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aeon import main
from aeon.core import bootguard, checkpoint, paths
from aeon.tools.restart import RestartAeonTool
from aeon.tools.revert import RevertAeonTool


class _Session:
    def __init__(self, events, *, result=True, broker_service=None, error=None):
        self.events = events
        self.result = result
        self._broker_service = broker_service
        self.error = error

    def exit(self):
        self.events.append("session-exit")
        if self.error is not None:
            raise self.error
        return self.result


class _Presence:
    def __init__(self, events):
        self.events = events

    def mark_exit(self):
        self.events.append("presence-exit")


class RestartLifecycleTests(unittest.TestCase):
    def test_restart_boot_recovery_requires_checkpoint_and_durable_marker(self):
        with mock.patch.object(bootguard, "mark_pending") as marker:
            with self.assertRaisesRegex(RuntimeError, "checkpoint is unavailable"):
                main._establish_restart_boot_recovery("/canonical", "")
            marker.assert_not_called()

            marker.return_value = False
            with self.assertRaisesRegex(RuntimeError, "marker could not be published"):
                main._establish_restart_boot_recovery(
                    "/canonical", "aeon-ckpt/known-good", reason="test"
                )

            marker.return_value = True
            main._establish_restart_boot_recovery(
                "/canonical", "aeon-ckpt/known-good", reason="test"
            )

    def test_restart_never_executes_candidate_without_masked_home_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "restart.json"
            state_path.write_text(
                __import__("json").dumps({
                    "objective": "continue safely",
                    "aeon_code_dir": str(paths.PROJECT_ROOT),
                }),
                encoding="utf-8",
            )
            worker = mock.Mock()
            worker.action_log = []
            with mock.patch.object(main, "RESTART_STATE_PATH", str(state_path)), \
                    mock.patch.object(main, "terminate_all_sub_agents") as terminate, \
                    mock.patch.object(main.subprocess, "run") as run, \
                    mock.patch.object(main.os, "execvpe") as execute:
                result = main._execute_restart(mock.Mock(), worker)
                state_removed = not state_path.exists()

        self.assertEqual(result, "continue safely")
        self.assertIn("RESTART BLOCKED", worker.last_observation)
        self.assertTrue(state_removed)
        terminate.assert_not_called()
        run.assert_not_called()
        execute.assert_not_called()

    def test_session_exit_retains_malformed_release_proof(self):
        manager = main.SessionManager()
        broker = mock.Mock()
        broker.ticket_id = "fd-0123456789abcdef0123456789abcdef"
        broker.close.return_value = {
            "state": "released",
            "compute_state": "ready",
        }
        manager._broker_service = broker

        with mock.patch.object(main, "terminate_all_sub_agents"), mock.patch.object(
            main, "cleanup_transient_tools"
        ):
            self.assertFalse(manager.exit())

        self.assertIs(manager._broker_service, broker)
        self.assertFalse(manager._cleanup_done)

    def test_session_exit_accepts_proven_absence_without_release_call(self):
        manager = main.SessionManager()
        broker = mock.Mock()
        broker.ticket_id = None
        broker.close.return_value = None
        manager._broker_service = broker

        with mock.patch.object(main, "terminate_all_sub_agents"), mock.patch.object(
            main, "cleanup_transient_tools"
        ):
            self.assertTrue(manager.exit())

        self.assertIsNone(manager._broker_service)
        self.assertTrue(manager._cleanup_done)

    def test_browser_cleanup_only_closes_authenticated_exact_session(self):
        headers = {"Authorization": "Bearer hermetic"}
        browser_http = mock.Mock()
        session = mock.MagicMock()
        session.return_value.__enter__.return_value = browser_http
        with mock.patch(
            "aeon.tools.browser.browser_auth_headers", return_value=headers
        ), mock.patch.object(main.requests, "Session", session):
            main.cleanup_transient_tools("travel")

        self.assertFalse(browser_http.trust_env)
        browser_http.post.assert_called_once_with(
            "http://127.0.0.1:8030/close_session",
            json={"session_id": str(os.getpid()), "profile": "travel"},
            headers=headers,
            timeout=2,
            allow_redirects=False,
        )

    def test_browser_cleanup_has_no_container_or_pid_registry_lifecycle(self):
        source = Path(main.__file__).read_text(encoding="utf-8")
        cleanup_body = source.split("def cleanup_transient_tools", 1)[1].split(
            "# =============================================================================", 1
        )[0]
        self.assertIn("browser_auth_headers", cleanup_body)
        self.assertIn("/close_session", cleanup_body)
        self.assertIn("trust_env = False", cleanup_body)
        self.assertIn("allow_redirects=False", cleanup_body)
        for forbidden in (
            "docker",
            "browser_registry",
            "/proc/",
            "os.kill",
            "subprocess",
        ):
            self.assertNotIn(forbidden, cleanup_body)

    def test_restart_source_must_be_exact_canonical_tree(self):
        with tempfile.TemporaryDirectory() as td:
            canonical = Path(td) / "canonical"
            alternate = Path(td) / "alternate"
            (canonical / "aeon").mkdir(parents=True)
            (alternate / "aeon").mkdir(parents=True)
            with mock.patch.object(paths, "PROJECT_ROOT", canonical):
                self.assertEqual(
                    main._canonical_restart_source(str(canonical)),
                    str(canonical.resolve()),
                )
                with self.assertRaisesRegex(ValueError, "canonical Aeon tree"):
                    main._canonical_restart_source(str(alternate))

    def test_restart_tool_refuses_alternate_package_tree(self):
        with tempfile.TemporaryDirectory() as td:
            canonical = Path(td) / "canonical"
            alternate = Path(td) / "alternate"
            canonical.mkdir()
            alternate.mkdir()
            tool = RestartAeonTool(worker=mock.Mock())
            with mock.patch.object(
                tool, "_default_code_dir", return_value=str(canonical)
            ):
                result = tool.execute(aeon_code_dir=str(alternate))
        self.assertIn("only reloads its canonical source tree", result)

    def test_restart_tool_writes_no_state_while_candidate_gate_is_unavailable(self):
        worker = mock.Mock()
        tool = RestartAeonTool(worker=worker)
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "restart.json"
            with mock.patch(
                "aeon.tools.restart.RESTART_STATE_PATH", str(state_path)
            ):
                result = tool.execute()
            state_exists = state_path.exists()

        self.assertIn("restart is blocked", result)
        self.assertFalse(state_exists)
        worker.serialize_state.assert_not_called()

    def test_revert_mutates_nothing_while_restart_boundary_is_unavailable(self):
        worker = mock.Mock()
        tool = RevertAeonTool(worker=worker)
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "restart.json"
            with mock.patch(
                "aeon.tools.restart.restart_validation_boundary_available",
                return_value=False,
            ) as boundary, mock.patch.object(
                checkpoint, "is_git_repo"
            ) as is_git_repo, mock.patch.object(
                checkpoint, "list_checkpoints"
            ) as list_checkpoints, mock.patch.object(
                checkpoint, "restore_checkpoint"
            ) as restore_checkpoint, mock.patch(
                "aeon.tools.revert.RESTART_STATE_PATH", str(state_path)
            ):
                result = tool.execute(checkpoint="aeon-checkpoint-synthetic")
            state_exists = state_path.exists()

        self.assertIn("revert is blocked", result)
        self.assertIn("No checkpoint, git/source, or restart-state mutation", result)
        self.assertFalse(state_exists)
        boundary.assert_called_once_with()
        is_git_repo.assert_not_called()
        list_checkpoints.assert_not_called()
        restore_checkpoint.assert_not_called()
        worker.serialize_state.assert_not_called()

    def test_exec_environment_puts_canonical_source_first(self):
        with tempfile.TemporaryDirectory() as td:
            canonical = Path(td) / "source"
            canonical.mkdir()
            environment = main._restart_exec_environment(
                str(canonical),
                {
                    "PYTHONPATH": os.pathsep.join(
                        ["/other/one", str(canonical), "/other/two"]
                    ),
                    "KEPT": "yes",
                },
            )
        self.assertEqual(
            environment["PYTHONPATH"].split(os.pathsep),
            [str(canonical.resolve()), "/other/one", "/other/two"],
        )
        self.assertEqual(environment["AEON_PROJECT_ROOT"], str(canonical.resolve()))
        self.assertEqual(environment["KEPT"], "yes")

    def test_unresolved_broker_release_blocks_exec(self):
        for cleanup_result, broker_service in ((False, object()), (True, object())):
            events = []
            session = _Session(
                events,
                result=cleanup_result,
                broker_service=broker_service,
            )
            with self.subTest(cleanup_result=cleanup_result), mock.patch.object(
                main.os, "execvpe"
            ) as execute:
                with self.assertRaises(main._RestartExecBlocked):
                    main._exec_after_exact_session_close(
                        session,
                        worker=None,
                        argv=[sys.executable, "-m", "aeon.main"],
                        environment={},
                    )
                execute.assert_not_called()
                self.assertEqual(events, ["session-exit"])

    def test_cleanup_exception_blocks_exec(self):
        events = []
        session = _Session(events, error=RuntimeError("synthetic release error"))
        with mock.patch.object(main.os, "execvpe") as execute:
            with self.assertRaisesRegex(
                main._RestartExecBlocked, "session cleanup raised"
            ):
                main._exec_after_exact_session_close(
                    session,
                    worker=None,
                    argv=[sys.executable, "-m", "aeon.main"],
                    environment={},
                )
        execute.assert_not_called()
        self.assertEqual(events, ["session-exit"])

    def test_exact_close_precedes_presence_exit_and_exec(self):
        events = []
        session = _Session(events, result=True, broker_service=None)
        worker = mock.Mock()
        worker.presence = _Presence(events)

        def failed_exec(*_args):
            events.append("exec")
            raise OSError("synthetic exec failure")

        with mock.patch.object(main.os, "execvpe", side_effect=failed_exec):
            with self.assertRaisesRegex(
                main._RestartExecBlocked, "after exact session cleanup"
            ):
                main._exec_after_exact_session_close(
                    session,
                    worker,
                    [sys.executable, "-m", "aeon.main"],
                    {"PYTHONPATH": "/canonical"},
                )
        self.assertEqual(events, ["session-exit", "presence-exit", "exec"])

    def test_restart_and_bootguard_do_not_execute_pip(self):
        restart_source = Path(main.__file__).read_text(encoding="utf-8")
        bootguard_source = Path(bootguard.__file__).read_text(encoding="utf-8")
        self.assertNotIn("'-m', 'pip'", restart_source)
        self.assertNotIn('"-m", "pip"', bootguard_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
