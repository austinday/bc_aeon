from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aeon.core.agent_protocol import ExecutionState, RequestContract, RunOutcome
from aeon.core.continuous_mode import (
    CONTINUOUS_MODE_ENV,
    ContinuousModeError,
    ContinuousModeState,
    NEXUS_CONTINUOUS_WAKE_COMMAND,
    load_continuous_mode,
    normalize_continuous_goal,
    serialize_continuous_mode,
)
from aeon.core.worker import Worker
from aeon.core.console import ConsoleInput, NEXUS_STOP_TURN_COMMAND
from aeon.main import (
    _continuous_objective,
    _run_objective_chain,
    _wait_for_continuous_compute,
    _wait_for_continuous_recovery,
)
from aeon.remote.instances import InstanceError, InstanceLaunchError, InstanceManager
from aeon.remote.instruction_profiles import InstructionProfileService
from aeon.tests.test_remote import FakeTmux, RemoteFixture


class ContinuousModeCoreTests(unittest.TestCase):
    def test_enabled_goal_requires_more_than_two_words(self):
        for value in ("", "one", "one two", "  one\n two  "):
            with self.subTest(value=value), self.assertRaises(ContinuousModeError):
                normalize_continuous_goal(value, enabled=True)
        self.assertEqual(
            normalize_continuous_goal("  improve returns safely  ", enabled=True),
            "improve returns safely",
        )
        self.assertEqual(normalize_continuous_goal("two words", enabled=False), "two words")

    def test_private_control_file_round_trip_and_symlink_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "continuous.json"
            path.write_bytes(
                serialize_continuous_mode(
                    ContinuousModeState(enabled=True, goal="keep improving this project")
                )
            )
            path.chmod(0o600)
            loaded = load_continuous_mode(path)
            self.assertTrue(loaded.enabled)
            self.assertEqual(loaded.goal, "keep improving this project")

            link = root / "link.json"
            link.symlink_to(path)
            with self.assertRaises(ContinuousModeError):
                load_continuous_mode(link)

    def test_private_control_file_survives_legal_short_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "continuous.json"
            path.write_bytes(
                serialize_continuous_mode(
                    ContinuousModeState(
                        enabled=True, goal="keep improving this project"
                    )
                )
            )
            path.chmod(0o600)
            real_read = os.read

            def short_read(descriptor, size):
                return real_read(descriptor, min(size, 5))

            with patch("aeon.core.utils.io.os.read", side_effect=short_read):
                loaded = load_continuous_mode(path)

            self.assertTrue(loaded.enabled)
            self.assertEqual(loaded.goal, "keep improving this project")

    def test_continuation_is_not_treated_as_answer_to_pending_question(self):
        worker = Worker.__new__(Worker)
        contract = RequestContract.from_request("Do the task")
        contract.state = ExecutionState.WAITING_USER
        contract.pending_question = "May I place the trade?"
        worker.request_contract = contract
        worker.execution_state = ExecutionState.WAITING_USER
        worker.pending_question = contract.pending_question
        worker.request_id = contract.request_id

        Worker.prepare_continuous_turn(
            worker, goal="keep improving the project safely"
        )

        self.assertIsNone(worker.request_contract)
        self.assertEqual(worker.execution_state, ExecutionState.DONE)
        self.assertEqual(worker.pending_question, "")
        self.assertEqual(worker.request_id, "")
        self.assertTrue(worker._next_request_is_continuous)

    def test_environment_state_builds_bounded_safety_preserving_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "continuous.json"
            path.write_bytes(
                serialize_continuous_mode(
                    ContinuousModeState(enabled=True, goal="improve verified returns safely")
                )
            )
            path.chmod(0o600)
            worker = unittest.mock.Mock()
            with patch.dict(os.environ, {CONTINUOUS_MODE_ENV: str(path)}):
                objective = _continuous_objective(worker)
        worker.prepare_continuous_turn.assert_called_once_with(
            goal="improve verified returns safely"
        )
        self.assertIn("improve verified returns safely", objective)
        self.assertIn("grants no new authority", objective)
        self.assertIn("financial", objective)

    def test_recovery_delta_is_forwarded_as_harness_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "continuous.json"
            path.write_bytes(
                serialize_continuous_mode(
                    ContinuousModeState(
                        enabled=True, goal="improve verified returns safely"
                    )
                )
            )
            path.chmod(0o600)
            worker = unittest.mock.Mock()
            recovery = "prior failed cycle; choose a different strategy"
            with patch.dict(os.environ, {CONTINUOUS_MODE_ENV: str(path)}):
                objective = _continuous_objective(
                    worker, recovery_context=recovery
                )

        worker.prepare_continuous_turn.assert_called_once_with(
            goal="improve verified returns safely",
            recovery_context=recovery
        )
        self.assertNotIn(recovery, objective)
        self.assertIn("same durable goal", objective)
        self.assertIn("Failure to find a competing artifact is not evidence", objective)
        self.assertIn("identity/version", objective)
        self.assertIn("license and redistribution terms", objective)
        self.assertIn("different source, method, hypothesis, modality", objective)
        self.assertIn("Do not retry an unchanged failure", objective)

    def test_private_wake_is_not_a_pending_user_message(self):
        input_console = ConsoleInput()
        input_console._tty = True
        input_console._started = True
        input_console._typeahead = True
        input_console._dispatch_line(NEXUS_CONTINUOUS_WAKE_COMMAND)
        self.assertFalse(input_console.has_pending())
        self.assertEqual(input_console.readline(), NEXUS_CONTINUOUS_WAKE_COMMAND)

    def test_private_wake_stays_control_input_in_between_turn_window(self):
        input_console = ConsoleInput()
        input_console._tty = True
        input_console._started = True
        input_console.enable_typeahead()
        input_console.disable_typeahead()
        input_console._dispatch_line(NEXUS_CONTINUOUS_WAKE_COMMAND)

        self.assertFalse(input_console.has_pending())
        self.assertEqual(input_console.readline(), NEXUS_CONTINUOUS_WAKE_COMMAND)

    def test_pending_user_or_handoff_preempts_without_disabling_continuous(self):
        input_console = ConsoleInput()
        input_console._tty = True
        input_console._started = True
        input_console.enable_typeahead()
        input_console._dispatch_line("NEXUS COLLABORATOR HANDOFF\nnew feedback")
        self.assertTrue(input_console.has_pending())
        worker = unittest.mock.Mock()

        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "continuous.json"
            original = serialize_continuous_mode(
                ContinuousModeState(
                    enabled=True, goal="keep improving the project safely"
                )
            )
            control.write_bytes(original)
            control.chmod(0o600)
            with (
                patch.dict(os.environ, {CONTINUOUS_MODE_ENV: str(control)}),
                patch("aeon.core.console.console", return_value=input_console),
                patch("aeon.main._execute_restart", return_value=None),
                patch("aeon.main._continuous_objective") as continuation,
            ):
                _run_objective_chain(object(), worker, "current continuous cycle")

            continuation.assert_not_called()
            self.assertEqual(control.read_bytes(), original)
            self.assertTrue(load_continuous_mode(control).enabled)
            self.assertEqual(
                input_console.take_pending(),
                "NEXUS COLLABORATOR HANDOFF\nnew feedback",
            )

    def test_between_turn_message_preempts_next_continuous_objective(self):
        input_console = ConsoleInput()
        input_console._tty = True
        input_console._started = True
        input_console.enable_typeahead()
        worker = unittest.mock.Mock()

        def finish_turn(*_args, **_kwargs):
            input_console.disable_typeahead()
            input_console._dispatch_line("feedback accepted at the turn boundary")

        worker.run.side_effect = finish_turn
        with (
            patch("aeon.core.console.console", return_value=input_console),
            patch("aeon.main._execute_restart", return_value=None),
            patch("aeon.main._continuous_objective") as continuation,
        ):
            _run_objective_chain(object(), worker, "current continuous cycle")

        continuation.assert_not_called()
        self.assertEqual(
            input_console.take_pending(),
            "feedback accepted at the turn boundary",
        )

    def test_identical_failed_cycles_back_off_and_receive_recovery_delta(self):
        input_console = unittest.mock.Mock()
        input_console.has_pending.return_value = False
        worker = unittest.mock.Mock()
        failed = RunOutcome(
            ExecutionState.FAILED,
            "Generation budget exhausted after 8192 tokens.",
        )
        worker.run.side_effect = [
            failed,
            RunOutcome(
                ExecutionState.FAILED,
                "Generation budget exhausted after 4096 tokens.",
            ),
            RunOutcome(ExecutionState.DONE, "A different strategy made progress."),
        ]

        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "continuous.json"
            original = serialize_continuous_mode(
                ContinuousModeState(
                    enabled=True, goal="keep improving the project safely"
                )
            )
            control.write_bytes(original)
            control.chmod(0o600)
            with (
                patch.dict(os.environ, {CONTINUOUS_MODE_ENV: str(control)}),
                patch("aeon.core.console.console", return_value=input_console),
                patch("aeon.main._execute_restart", return_value=None),
                patch(
                    "aeon.main._wait_for_continuous_recovery", return_value=True
                ) as wait,
                patch(
                    "aeon.main._continuous_objective",
                    side_effect=["recovery cycle one", "recovery cycle two", None],
                ) as continuation,
            ):
                _run_objective_chain(object(), worker, "initial cycle")

            self.assertEqual(control.read_bytes(), original)

        self.assertEqual(
            [call.args[0] for call in wait.call_args_list],
            [2.0, 5.0],
        )
        first_context = continuation.call_args_list[0].kwargs["recovery_context"]
        second_context = continuation.call_args_list[1].kwargs["recovery_context"]
        self.assertIn("state=failed", first_context)
        self.assertIn("identical_failure_streak=1", first_context)
        self.assertIn("identical_failure_streak=2", second_context)
        self.assertEqual(continuation.call_args_list[2].kwargs, {})

    def test_three_identical_failed_cycles_open_circuit_without_disabling_goal(self):
        input_console = unittest.mock.Mock()
        input_console.has_pending.return_value = False
        worker = unittest.mock.Mock()
        failed = RunOutcome(
            ExecutionState.FAILED,
            "Generation budget exhausted after 8192 tokens.",
        )
        worker.run.side_effect = [failed, failed, failed]

        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "continuous.json"
            original = serialize_continuous_mode(
                ContinuousModeState(
                    enabled=True, goal="keep improving the project safely"
                )
            )
            control.write_bytes(original)
            control.chmod(0o600)
            with (
                patch.dict(os.environ, {CONTINUOUS_MODE_ENV: str(control)}),
                patch("aeon.core.console.console", return_value=input_console),
                patch("aeon.main._execute_restart", return_value=None),
                patch(
                    "aeon.main._wait_for_continuous_recovery", return_value=True
                ) as wait,
                patch(
                    "aeon.main._continuous_objective",
                    side_effect=["recovery one", "recovery two"],
                ) as continuation,
                patch(
                    "aeon.core.chat_transcript.append_assistant_message_from_environment"
                ) as publish,
            ):
                _run_objective_chain(object(), worker, "initial cycle")

            self.assertEqual(control.read_bytes(), original)
            self.assertTrue(load_continuous_mode(control).enabled)

        self.assertEqual([call.args[0] for call in wait.call_args_list], [2.0, 5.0])
        self.assertEqual(continuation.call_count, 2)
        self.assertEqual(worker.run.call_count, 3)
        self.assertIn("paused after three", publish.call_args.args[0])
        worker._presence_update.assert_called_once()

    def test_alternating_failures_open_six_cycle_plateau_circuit(self):
        input_console = unittest.mock.Mock()
        input_console.has_pending.return_value = False
        worker = unittest.mock.Mock()
        worker.run.side_effect = [
            RunOutcome(ExecutionState.FAILED, message)
            for message in (
                "network route unavailable",
                "repository metadata rejected",
                "local validator failed",
                "publication capability missing",
                "artifact inventory disagrees",
                "documentation verification blocked",
            )
        ]

        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "continuous.json"
            original = serialize_continuous_mode(
                ContinuousModeState(enabled=True, goal="keep improving safely")
            )
            control.write_bytes(original)
            control.chmod(0o600)
            with (
                patch.dict(os.environ, {CONTINUOUS_MODE_ENV: str(control)}),
                patch("aeon.core.console.console", return_value=input_console),
                patch("aeon.main._execute_restart", return_value=None),
                patch(
                    "aeon.main._wait_for_continuous_recovery", return_value=True
                ) as wait,
                patch(
                    "aeon.main._continuous_objective",
                    side_effect=[f"recovery {index}" for index in range(5)],
                ) as continuation,
                patch(
                    "aeon.core.chat_transcript.append_assistant_message_from_environment"
                ) as publish,
            ):
                _run_objective_chain(object(), worker, "initial cycle")

            self.assertEqual(control.read_bytes(), original)

        self.assertEqual(wait.call_count, 5)
        self.assertEqual(continuation.call_count, 5)
        self.assertEqual(worker.run.call_count, 6)
        self.assertIn("six consecutive", publish.call_args.args[0])

    def test_recovery_backoff_is_preempted_by_new_user_input(self):
        class FakeConsole:
            def __init__(self):
                self.pending = False
                self.enabled = False

            def enable_typeahead(self):
                self.enabled = True

            def disable_typeahead(self):
                self.enabled = False

            def has_pending(self):
                return self.pending

            def has_stop_request(self):
                return False

        input_console = FakeConsole()
        now = [0.0]

        def sleep_and_deliver(seconds):
            now[0] += seconds
            input_console.pending = True

        with patch("aeon.main._continuous_mode_enabled", return_value=True):
            completed = _wait_for_continuous_recovery(
                60,
                input_console,
                clock=lambda: now[0],
                sleeper=sleep_and_deliver,
            )

        self.assertFalse(completed)
        self.assertFalse(input_console.enabled)
        self.assertLessEqual(now[0], 0.5)

    def test_verified_compute_wait_is_bounded_before_next_cycle(self):
        input_console = unittest.mock.Mock()
        input_console.has_pending.return_value = False
        worker = unittest.mock.Mock()
        worker.run.side_effect = [
            RunOutcome(ExecutionState.WAITING_COMPUTE, "durable Fleet job pending"),
            RunOutcome(ExecutionState.DONE, "job completed"),
        ]

        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "continuous.json"
            control.write_bytes(
                serialize_continuous_mode(
                    ContinuousModeState(
                        enabled=True, goal="keep building useful models safely"
                    )
                )
            )
            control.chmod(0o600)
            with (
                patch.dict(os.environ, {CONTINUOUS_MODE_ENV: str(control)}),
                patch("aeon.core.console.console", return_value=input_console),
                patch("aeon.main._execute_restart", return_value=None),
                patch(
                    "aeon.main._wait_for_continuous_compute", return_value=True
                ) as wait,
                patch(
                    "aeon.main._continuous_objective",
                    side_effect=["status cycle", None],
                ) as continuation,
            ):
                _run_objective_chain(object(), worker, "initial cycle")

        wait.assert_called_once_with(input_console)
        self.assertEqual(continuation.call_args_list[0].kwargs, {})

    def test_compute_wait_helper_remains_cancellable(self):
        with patch(
            "aeon.main._wait_for_continuous_recovery", return_value=False
        ) as wait:
            self.assertFalse(_wait_for_continuous_compute("console"))
        wait.assert_called_once_with(30.0, "console")


class ContinuousModeManagerTests(RemoteFixture):
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

    def _managed_continuous_agent(self, name: str = "Managed continuous") -> dict:
        terminal = self.manager.create_terminal(
            name=name,
            workspace=str(self.workspace),
            actor="admin",
        )
        agent = self.manager.activate_agent(
            terminal["id"], kind="aeon", actor="admin"
        )
        self.manager.update_continuous_mode(
            agent["id"],
            enabled=True,
            goal="keep improving the repository safely",
            actor="admin",
        )
        return self.store.get_instance(agent["id"])

    def _direct_continuous_agent(self, name: str = "Direct continuous") -> dict:
        agent = self.manager.create_instance(
            name=name,
            workspace=str(self.workspace),
            objective="Initial user request",
            max_iterations=None,
            actor="admin",
            continuous_enabled=True,
            continuous_goal="keep improving the repository safely",
        )
        return self.store.get_instance(agent["id"])

    def _return_agent_to_prompt(self, record: dict) -> None:
        pane = self.fake.sessions[record["tmux_name"]]
        pane.update(
            command="bash",
            agent_mode=False,
            managed_agent=False,
            at_prompt=True,
        )

    def test_live_toggle_persists_and_wakes_without_model_text(self):
        agent = self.manager.create_instance(
            name="Continuous worker",
            workspace=str(self.workspace),
            objective="Initial user request",
            max_iterations=None,
            actor="admin",
        )
        payload = self.manager.update_continuous_mode(
            agent["id"],
            enabled=True,
            goal="keep improving the repository safely",
            actor="admin",
        )
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["goal"], "keep improving the repository safely")
        self.assertEqual(self.fake.loaded_payloads[-1], NEXUS_CONTINUOUS_WAKE_COMMAND + "\r")
        control = self.config.instance_state_dir / agent["id"] / "continuous-mode.json"
        self.assertEqual(control.stat().st_mode & 0o777, 0o600)
        self.assertTrue(json.loads(control.read_text(encoding="utf-8"))["enabled"])

        payload = self.manager.update_continuous_mode(
            agent["id"],
            enabled=False,
            goal="keep improving the repository safely",
            actor="admin",
        )
        self.assertFalse(payload["enabled"])
        self.assertEqual(
            self.fake.loaded_payloads[-1],
            f"\x1b[200~{NEXUS_STOP_TURN_COMMAND}\x1b[201~\r",
        )

        stopped = self.manager.graceful_stop(agent["id"], actor="admin")
        self.assertEqual(stopped["continuous_mode"]["enabled"], False)
        self.assertEqual(stopped["continuous_mode"]["goal"], "keep improving the repository safely")

    def test_unacknowledged_direct_turn_stop_restarts_aeon_idle(self):
        agent = self._direct_continuous_agent("Unacknowledged stop")
        worker_state_root = Path(self.temp.name) / "worker-state"
        with patch.dict(
            os.environ, {"AEON_STATE_DIR": str(worker_state_root)}, clear=False
        ):
            worker_session = self.manager._worker_session_directory(agent)
            worker_session.mkdir(mode=0o700, parents=True)
            state_file = worker_session / "session_state.json"
            state_file.write_text(
                json.dumps({"execution_state": "running", "memories": {"keep": True}}),
                encoding="utf-8",
            )
            state_file.chmod(0o600)
            with patch.object(
                self.manager, "_wait_for_worker_turn_stop", return_value=False
            ):
                payload = self.manager.update_continuous_mode(
                    agent["id"],
                    enabled=False,
                    goal="keep improving the repository safely",
                    actor="admin",
                )

        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["goal"], "keep improving the repository safely")
        self.assertEqual(
            json.loads(state_file.read_text(encoding="utf-8"))["execution_state"],
            "cancelled",
        )
        launches = [call for call in self.fake.calls if call[1] == "new-session"]
        self.assertEqual(len(launches), 2)
        self.assertNotIn("--start", launches[-1])
        self.assertNotIn("--resume-unfinished", launches[-1])
        lifecycle_calls = [call[1] for call in self.fake.calls]
        self.assertIn("kill-session", lifecycle_calls)

    def test_acknowledged_turn_stop_does_not_restart_aeon(self):
        agent = self._direct_continuous_agent("Acknowledged stop")
        worker_state_root = Path(self.temp.name) / "worker-state"
        with patch.dict(
            os.environ, {"AEON_STATE_DIR": str(worker_state_root)}, clear=False
        ):
            worker_session = self.manager._worker_session_directory(agent)
            worker_session.mkdir(mode=0o700, parents=True)
            state_file = worker_session / "session_state.json"
            state_file.write_text(
                json.dumps({"execution_state": "running"}), encoding="utf-8"
            )
            state_file.chmod(0o600)
            with patch.object(
                self.manager, "_wait_for_worker_turn_stop", return_value=True
            ):
                payload = self.manager.update_continuous_mode(
                    agent["id"],
                    enabled=False,
                    goal="keep improving the repository safely",
                    actor="admin",
                )

        self.assertFalse(payload["enabled"])
        launches = [call for call in self.fake.calls if call[1] == "new-session"]
        self.assertEqual(len(launches), 1)

    def test_turn_stop_pane_read_failure_requests_managed_recovery(self):
        agent = self._direct_continuous_agent("Pane read failure")
        with patch.object(
            self.manager,
            "_pane_info",
            side_effect=InstanceError("tmux observation temporarily unavailable"),
        ):
            acknowledged = self.manager._wait_for_worker_turn_stop(
                agent,
                timeout=0.1,
            )

        self.assertFalse(acknowledged)

    def test_fresh_restart_never_clears_context_while_old_agent_is_live(self):
        agent = self._managed_continuous_agent("Restart boundary")
        still_live = self.manager.get_instance(agent["id"])

        with (
            patch.object(
                self.manager,
                "_end_agent_locked",
                return_value=still_live,
            ),
            patch.object(self.manager, "_reset_agent_context_locked") as reset,
            self.assertRaisesRegex(InstanceError, "verified managed terminal"),
        ):
            self.manager.fresh_restart_agent(agent["id"], actor="admin")

        reset.assert_not_called()

    def test_disable_rolls_back_when_live_foreground_is_ambiguous(self):
        agent = self._managed_continuous_agent("Ambiguous stop boundary")
        self.fake.sessions[agent["tmux_name"]].update(
            command="python3",
            agent_mode=False,
            managed_agent=False,
            at_prompt=False,
        )

        with self.assertRaisesRegex(InstanceError, "identity could not be verified"):
            self.manager.update_continuous_mode(
                agent["id"],
                enabled=False,
                goal="keep improving the repository safely",
                actor="admin",
            )

        self.assertTrue(self.store.get_continuous_mode(agent["id"]).enabled)

    def test_short_goal_and_provider_mode_fail_closed(self):
        agent = self.manager.create_instance(
            name="Short goal worker",
            workspace=str(self.workspace),
            objective="Initial request",
            max_iterations=None,
            actor="admin",
        )
        with self.assertRaisesRegex(InstanceError, "more than two words"):
            self.manager.update_continuous_mode(
                agent["id"], enabled=True, goal="two words", actor="admin"
            )

        terminal = self.manager.create_terminal(
            name="Provider tab", workspace=str(self.workspace), actor="admin"
        )
        self.store.update_instance(terminal["id"], last_agent_kind="codex")
        with self.assertRaisesRegex(InstanceError, "only for local Aeon"):
            self.manager.update_continuous_mode(
                terminal["id"],
                enabled=True,
                goal="keep working on this",
                actor="admin",
            )

    def test_disabled_goal_save_needs_no_live_pane_and_enable_ambiguity_rolls_back(self):
        agent = self.manager.create_instance(
            name="Ambiguous continuous worker",
            workspace=str(self.workspace),
            objective="Initial request",
            max_iterations=None,
            actor="admin",
        )
        self.fake.list_panes_error = True
        saved = self.manager.update_continuous_mode(
            agent["id"],
            enabled=False,
            goal="keep improving this project",
            actor="admin",
        )
        self.assertFalse(saved["enabled"])
        with self.assertRaises(InstanceError):
            self.manager.update_continuous_mode(
                agent["id"],
                enabled=True,
                goal="keep improving this project",
                actor="admin",
            )
        restored = self.store.get_continuous_mode(agent["id"])
        self.assertFalse(restored.enabled)
        self.assertEqual(restored.goal, "keep improving this project")

    def test_configured_creation_persists_identity_before_continuous_launch(self):
        instructions = InstructionProfileService(
            self.store,
            project_root=self.config.project_root,
            allowed_roots=self.config.allowed_roots,
        )
        manager = InstanceManager(
            self.store,
            self.config,
            command_runner=self.fake,
            instruction_service=instructions,
            pane_prompt_checker=self.fake.pane_at_prompt,
            pane_foreground_checker=self.fake.pane_has_managed_foreground,
        )
        identity = "# Private identity\nKeep the exact configured personality."
        agent = manager.create_instance(
            name="Configured continuous worker",
            workspace=str(self.workspace),
            objective="",
            max_iterations=None,
            actor="project-manager",
            continuous_enabled=True,
            continuous_goal="continuously improve verified project outcomes",
            local_instructions=identity,
        )

        self.assertEqual(agent["status"], "running")
        self.assertFalse(agent["awaiting_objective"])
        self.assertTrue(agent["continuous_mode"]["enabled"])
        binding = instructions.get_instance_binding(agent["id"])
        self.assertEqual(binding["desired_local_content"], identity)
        launch = next(call for call in self.fake.calls if call[1] == "new-session")
        rendered = " ".join(launch)
        self.assertIn(CONTINUOUS_MODE_ENV, rendered)
        self.assertNotIn("continuously improve verified project outcomes", rendered)

    def test_supervisor_recovers_exact_prompt_with_checkpoint_resume(self):
        agent = self._managed_continuous_agent()
        self._return_agent_to_prompt(agent)

        self.manager.ensure_persistent_continuous_instances()

        recovered = self.store.get_instance(agent["id"])
        self.assertEqual(recovered["kind"], "aeon")
        self.assertEqual(recovered["desired_state"], "running")
        self.assertIn("aeon.harnesses.opencode_runtime", self.fake.loaded_payloads[-1])
        self.assertIn("--resume-unfinished", self.fake.loaded_payloads[-1])

    def test_supervisor_leaves_verified_managed_foreground_untouched(self):
        agent = self._managed_continuous_agent("Healthy continuous")
        payloads_before = list(self.fake.loaded_payloads)

        self.manager.ensure_persistent_continuous_instances()

        self.assertEqual(self.fake.loaded_payloads, payloads_before)
        self.assertEqual(self.store.get_instance(agent["id"])["kind"], "aeon")

    def test_supervisor_recreates_missing_managed_shell_before_resume(self):
        agent = self._managed_continuous_agent("Missing shell continuous")
        self.fake.sessions.pop(agent["tmux_name"])
        launches_before = sum(call[1] == "new-session" for call in self.fake.calls)

        self.manager.ensure_persistent_continuous_instances()

        launches_after = sum(call[1] == "new-session" for call in self.fake.calls)
        self.assertEqual(launches_after, launches_before + 1)
        self.assertEqual(self.store.get_instance(agent["id"])["kind"], "aeon")
        self.assertIn("--resume-unfinished", self.fake.loaded_payloads[-1])

    def test_supervisor_rereads_disable_and_honors_end_and_stop(self):
        disabled = self._managed_continuous_agent("Disabled continuous")
        self._return_agent_to_prompt(disabled)
        stale_enabled_row = dict(self.store.get_instance(disabled["id"]))
        self.manager.update_continuous_mode(
            disabled["id"],
            enabled=False,
            goal="keep improving the repository safely",
            actor="admin",
        )
        launches_before = len(self.fake.loaded_payloads)
        with patch.object(
            self.store, "list_instances", return_value=[stale_enabled_row]
        ):
            self.manager.ensure_persistent_continuous_instances()
        self.assertEqual(len(self.fake.loaded_payloads), launches_before)

        ended = self._managed_continuous_agent("Ended continuous")
        returned = self.manager.end_agent(ended["id"], actor="admin")
        self.assertFalse(returned["continuous_mode"]["enabled"])

        stopped_agent = self._managed_continuous_agent("Stopped continuous")
        self._return_agent_to_prompt(stopped_agent)
        self.manager.get_instance(stopped_agent["id"])
        stopped = self.manager.graceful_stop(stopped_agent["id"], actor="admin")
        self.assertEqual(stopped["desired_state"], "stopped")
        self.assertFalse(stopped["continuous_mode"]["enabled"])
        launches_before = len(self.fake.loaded_payloads)
        self.manager.ensure_persistent_continuous_instances()
        self.assertEqual(len(self.fake.loaded_payloads), launches_before)

    def test_supervisor_refuses_ambiguous_and_unsafe_rows(self):
        ambiguous = self._managed_continuous_agent("Ambiguous continuous")
        pane = self.fake.sessions[ambiguous["tmux_name"]]
        pane.update(
            command="python3",
            agent_mode=False,
            managed_agent=False,
            at_prompt=False,
        )
        launches_before = len(self.fake.loaded_payloads)
        self.manager.ensure_persistent_continuous_instances()
        self.assertEqual(len(self.fake.loaded_payloads), launches_before)

        awaiting = self._managed_continuous_agent("Awaiting continuous")
        self._return_agent_to_prompt(awaiting)
        self.store.update_instance(awaiting["id"], awaiting_objective=1)
        self.manager.ensure_persistent_continuous_instances()
        self.assertEqual(len(self.fake.loaded_payloads), launches_before + 2)

        collaborator = self._managed_continuous_agent("Collaborator continuous")
        self._return_agent_to_prompt(collaborator)
        collaborator_launches = len(self.fake.loaded_payloads)
        with patch.object(
            self.store,
            "get_collaboration_portal_for_instance",
            return_value={"id": "collab-test", "status": "active"},
        ):
            self.manager.ensure_persistent_continuous_instances()
        self.assertEqual(len(self.fake.loaded_payloads), collaborator_launches)

        malformed = self._managed_continuous_agent("Malformed continuous")
        self._return_agent_to_prompt(malformed)
        malformed_launches = len(self.fake.loaded_payloads)
        with patch.object(
            self.store,
            "get_continuous_mode",
            side_effect=ValueError("invalid enabled state"),
        ):
            self.manager.ensure_persistent_continuous_instances()
        self.assertEqual(len(self.fake.loaded_payloads), malformed_launches)

    def test_supervisor_leaves_live_direct_aeon_untouched(self):
        agent = self._direct_continuous_agent("Live direct continuous")
        launches_before = sum(call[1] == "new-session" for call in self.fake.calls)
        kills_before = sum(call[1] == "kill-session" for call in self.fake.calls)
        pane_before = dict(self.fake.sessions[agent["tmux_name"]])

        self.manager.ensure_persistent_continuous_instances()

        launches_after = sum(call[1] == "new-session" for call in self.fake.calls)
        self.assertEqual(launches_after, launches_before)
        self.assertEqual(
            sum(call[1] == "kill-session" for call in self.fake.calls),
            kills_before,
        )
        self.assertEqual(self.fake.sessions[agent["tmux_name"]], pane_before)

    def test_supervisor_resumes_dead_direct_aeon_from_checkpoint(self):
        agent = self._direct_continuous_agent("Dead direct continuous")
        self.fake.sessions[agent["tmux_name"]]["dead"] = True
        launches_before = sum(call[1] == "new-session" for call in self.fake.calls)
        kills_before = sum(call[1] == "kill-session" for call in self.fake.calls)

        self.manager.ensure_persistent_continuous_instances()

        launches = [call for call in self.fake.calls if call[1] == "new-session"]
        self.assertEqual(len(launches), launches_before + 1)
        self.assertEqual(
            sum(call[1] == "kill-session" for call in self.fake.calls),
            kills_before + 1,
        )
        self.assertIn("--resume-unfinished", launches[-1])
        self.assertNotIn("--start", launches[-1])
        self.assertEqual(self.store.get_instance(agent["id"])["status"], "running")

    def test_supervisor_resumes_missing_direct_aeon_without_kill(self):
        agent = self._direct_continuous_agent("Missing direct continuous")
        self.fake.sessions.pop(agent["tmux_name"])
        launches_before = sum(call[1] == "new-session" for call in self.fake.calls)
        kills_before = sum(call[1] == "kill-session" for call in self.fake.calls)

        self.manager.ensure_persistent_continuous_instances()

        launches = [call for call in self.fake.calls if call[1] == "new-session"]
        self.assertEqual(len(launches), launches_before + 1)
        self.assertEqual(
            sum(call[1] == "kill-session" for call in self.fake.calls),
            kills_before,
        )
        self.assertIn("--resume-unfinished", launches[-1])

    def test_supervisor_honors_direct_disable_and_stopped_state(self):
        disabled = self._direct_continuous_agent("Disabled direct continuous")
        self.fake.sessions.pop(disabled["tmux_name"])
        self.manager.update_continuous_mode(
            disabled["id"],
            enabled=False,
            goal="keep improving the repository safely",
            actor="admin",
        )

        stopped = self._direct_continuous_agent("Stopped direct continuous")
        self.manager.graceful_stop(stopped["id"], actor="admin")
        self.manager.update_continuous_mode(
            stopped["id"],
            enabled=True,
            goal="keep improving the repository safely",
            actor="admin",
        )
        self.fake.sessions.pop(stopped["tmux_name"], None)
        launches_before = sum(call[1] == "new-session" for call in self.fake.calls)

        self.manager.ensure_persistent_continuous_instances()

        self.assertEqual(
            sum(call[1] == "new-session" for call in self.fake.calls),
            launches_before,
        )
        self.assertFalse(self.store.get_continuous_mode(disabled["id"]).enabled)
        self.assertTrue(self.store.get_continuous_mode(stopped["id"]).enabled)
        self.assertEqual(
            self.store.get_instance(stopped["id"])["desired_state"], "stopped"
        )

    def test_supervisor_never_replaces_ambiguous_live_direct_pane(self):
        agent = self._direct_continuous_agent("Ambiguous direct continuous")
        pane = self.fake.sessions[agent["tmux_name"]]
        pane.update(command="bash", agent_mode=False, managed_agent=False)
        pane_before = dict(pane)
        launches_before = sum(call[1] == "new-session" for call in self.fake.calls)
        kills_before = sum(call[1] == "kill-session" for call in self.fake.calls)
        signals_before = sum(call[1] == "send-keys" for call in self.fake.calls)

        self.manager.ensure_persistent_continuous_instances()

        self.assertEqual(self.fake.sessions[agent["tmux_name"]], pane_before)
        self.assertEqual(
            sum(call[1] == "new-session" for call in self.fake.calls),
            launches_before,
        )
        self.assertEqual(
            sum(call[1] == "kill-session" for call in self.fake.calls),
            kills_before,
        )
        self.assertEqual(
            sum(call[1] == "send-keys" for call in self.fake.calls),
            signals_before,
        )

    def test_supervisor_direct_failure_retains_intent_and_backs_off(self):
        agent = self._direct_continuous_agent("Backoff direct continuous")
        self.fake.sessions.pop(agent["tmux_name"])

        def fail_not_launched(instance_id, **_kwargs):
            self.store.update_instance(instance_id, desired_state="stopped")
            raise InstanceLaunchError("transient launch failure", launched=False)

        with (
            patch.object(
                self.manager,
                "_resume_instance_locked",
                side_effect=fail_not_launched,
            ) as resume,
            patch("aeon.remote.instances.time.monotonic", return_value=100.0),
        ):
            self.manager.ensure_persistent_continuous_instances()
            self.manager.ensure_persistent_continuous_instances()

        self.assertEqual(resume.call_count, 1)
        self.assertEqual(
            self.store.get_instance(agent["id"])["desired_state"], "running"
        )

    def test_supervisor_failed_launch_retries_with_capped_backoff(self):
        agent = self._managed_continuous_agent("Backoff continuous")
        self._return_agent_to_prompt(agent)

        def fail_not_launched(instance_id, **_kwargs):
            self.store.update_instance(instance_id, desired_state="stopped")
            raise InstanceLaunchError("transient launch failure", launched=False)

        with (
            patch.object(
                self.manager,
                "_activate_agent_locked",
                side_effect=fail_not_launched,
            ) as activate,
            patch("aeon.remote.instances.time.monotonic", return_value=100.0),
        ):
            self.manager.ensure_persistent_continuous_instances()
            self.manager.ensure_persistent_continuous_instances()
        self.assertEqual(activate.call_count, 1)
        self.assertTrue(activate.call_args.kwargs["resume_unfinished"])
        self.assertEqual(
            self.store.get_instance(agent["id"])["desired_state"], "running"
        )

        with (
            patch.object(
                self.manager,
                "_activate_agent_locked",
                side_effect=fail_not_launched,
            ) as activate_again,
            patch("aeon.remote.instances.time.monotonic", return_value=105.0),
        ):
            self.manager.ensure_persistent_continuous_instances()
        self.assertEqual(activate_again.call_count, 1)

        with patch("aeon.remote.instances.time.monotonic", return_value=200.0):
            for _ in range(12):
                self.manager._defer_continuous_recovery(agent["id"])
        _attempts, retry_at = self.manager._continuous_recovery_backoff[agent["id"]]
        self.assertEqual(retry_at, 500.0)


if __name__ == "__main__":
    unittest.main()
