"""Hermetic tests for exact, resource-safe bounded-agent termination."""

from __future__ import annotations

import inspect
import json
import signal
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import call, patch

from aeon.core import sub_agent_state as state
from aeon.core.sub_agent_environment import (
    SUB_AGENT_FLEET_CLOSE_WORST_CASE_SECONDS,
)


AGENT_ID = "12345678-1234-1234-1234-123456789abc"


class SubAgentTerminationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

        self.legacy_dir = self.root / "agent-1234"
        self.legacy_dir.mkdir()
        self.legacy_reference = {
            "schema": 1,
            "agent_id": self.legacy_dir.name,
            "pid": 4242,
            "pgid": 4242,
            "start_ticks": 99,
        }
        (self.legacy_dir / state.PROCESS_REF).write_text(
            json.dumps(self.legacy_reference), encoding="utf-8"
        )

        self.agent_dir = self.root / AGENT_ID
        self.agent_dir.mkdir()
        self.scope_unit, self.slice_unit = state.sub_agent_systemd_units(AGENT_ID)
        self.slice_group = (
            "/user.slice/user-1000.slice/user@1000.service/" + self.slice_unit
        )
        self.scope_group = f"{self.slice_group}/{self.scope_unit}"
        self.reference = {
            "schema": 2,
            "agent_id": AGENT_ID,
            "pid": 5252,
            "launcher_pid": 5252,
            "launcher_start_ticks": 101,
            "scope_unit": self.scope_unit,
            "slice_unit": self.slice_unit,
            "scope_invocation_id": "a" * 32,
            "slice_invocation_id": "b" * 32,
            "scope_control_group": self.scope_group,
            "slice_control_group": self.slice_group,
            "scope_control_group_id": 7001,
            "slice_control_group_id": 7000,
        }
        (self.agent_dir / state.PROCESS_REF).write_text(
            json.dumps(self.reference), encoding="utf-8"
        )

        self.cgroup_root = self.root / "cgroup"
        self.slice_path = self._cgroup_path(self.slice_group)
        self.scope_path = self._cgroup_path(self.scope_group)
        self.scope_path.mkdir(parents=True)

    def _cgroup_path(self, control_group):
        return self.cgroup_root.joinpath(*PurePosixPath(control_group).parts[1:])

    def _write_populated(self, value):
        (self.slice_path / "cgroup.events").write_text(
            f"populated {int(value)}\nfrozen 0\n", encoding="ascii"
        )

    def _scope_properties(self, *, present=True, **overrides):
        values = {
            "Id": self.scope_unit,
            "LoadState": "loaded" if present else "not-found",
            "ActiveState": "active" if present else "inactive",
            "SubState": "running" if present else "dead",
            "Transient": "yes" if present else "no",
            "InvocationID": "a" * 32 if present else "",
            "ControlGroup": self.scope_group if present else "",
            "ControlGroupId": "7001" if present else "0",
            "Slice": self.slice_unit if present else "",
            "DevicePolicy": "closed" if present else "auto",
            "KillMode": "control-group",
            "TimeoutStopUSec": "30s" if present else "1min 30s",
            "SendSIGKILL": "yes",
        }
        values.update(overrides)
        return values

    def _slice_properties(self, *, present=True, **overrides):
        values = {
            "Id": self.slice_unit,
            "LoadState": "loaded",
            "ActiveState": "active" if present else "inactive",
            "SubState": "active" if present else "dead",
            "Transient": "no",
            "InvocationID": "b" * 32 if present else "",
            "ControlGroup": self.slice_group if present else "",
            "ControlGroupId": "7000" if present else "0",
        }
        values.update(overrides)
        return values

    def _show(self, scope=None, slice_=None):
        scope = self._scope_properties() if scope is None else scope
        slice_ = self._slice_properties() if slice_ is None else slice_

        def fake(unit, _properties):
            if unit == self.scope_unit:
                return dict(scope)
            if unit == self.slice_unit:
                return dict(slice_)
            self.fail(f"unexpected unit readback: {unit}")

        return fake

    def test_sigterm_grace_exceeds_complete_fleet_close_worst_case(self):
        self.assertGreater(
            state.TERMINATION_GRACE_SECONDS,
            SUB_AGENT_FLEET_CLOSE_WORST_CASE_SECONDS,
        )

    def test_units_are_flat_uuid_derived_and_command_policy_is_fixed(self):
        command = state.sub_agent_systemd_command(AGENT_ID, ["/usr/bin/true"])

        self.assertRegex(self.scope_unit, r"^aeon-subagent-[0-9a-f]{32}\.scope$")
        self.assertRegex(self.slice_unit, r"^aeon_subagent_[0-9a-f]{32}\.slice$")
        self.assertEqual(command[0], "/usr/bin/systemd-run")
        separator = command.index("--")
        self.assertEqual(
            command[separator + 1:],
            ["/home/aday/bin/fleet-low-priority", "/usr/bin/true"],
        )
        for required in (
            "--user",
            "--scope",
            "--collect",
            "--expand-environment=no",
            f"--unit={self.scope_unit}",
            f"--slice={self.slice_unit}",
            "--property=DevicePolicy=closed",
            "--property=KillMode=control-group",
            "--property=TimeoutStopSec=30s",
            "--property=SendSIGKILL=yes",
        ):
            self.assertEqual(command.count(required), 1)

    def test_model_command_cannot_override_low_priority_prefix(self):
        attacker_argv = [
            "/tmp/model-selected-wrapper",
            "/home/aday/bin/fleet-low-priority",
            "/usr/bin/true",
        ]
        command = state.sub_agent_systemd_command(AGENT_ID, attacker_argv)
        payload = command[command.index("--") + 1:]

        self.assertEqual(payload[0], "/home/aday/bin/fleet-low-priority")
        self.assertEqual(payload[1:], attacker_argv)

    def test_low_priority_wrapper_identity_drift_fails_before_command_build(self):
        with patch.object(state, "_require_fleet_low_priority_wrapper", side_effect=(
            state.ProcessIdentityError("synthetic drift")
        )):
            with self.assertRaisesRegex(state.ProcessIdentityError, "synthetic drift"):
                state.sub_agent_systemd_command(AGENT_ID, ["/usr/bin/true"])

    def test_launch_preflight_refuses_active_unit_reuse(self):
        with patch.object(state, "_systemctl_show", side_effect=self._show()):
            with self.assertRaisesRegex(state.ProcessIdentityError, "refusing to reuse"):
                state.assert_sub_agent_systemd_units_available(AGENT_ID)

        inactive = self._show(
            scope=self._scope_properties(present=False),
            slice_=self._slice_properties(present=False),
        )
        with patch.object(state, "_systemctl_show", side_effect=inactive):
            state.assert_sub_agent_systemd_units_available(AGENT_ID)

    def test_capture_writes_exact_schema2_receipt(self):
        wrapper = [
            "/usr/bin/python3",
            "-m",
            "aeon.scripts.sub_agent_wrapper",
            "--agent_id",
            AGENT_ID,
            "--output_dir",
            str(self.agent_dir),
        ]
        launcher = state.sub_agent_systemd_command(AGENT_ID, wrapper)
        with (
            patch.object(state, "_proc_start_ticks", return_value=101),
            patch.object(state, "_proc_args", return_value=launcher),
            patch.object(state, "_systemctl_show", side_effect=self._show()),
        ):
            receipt = state.capture_sub_agent_process(
                self.agent_dir,
                5252,
                scope_unit=self.scope_unit,
                slice_unit=self.slice_unit,
                timeout_seconds=0,
            )

        self.assertEqual(receipt, self.reference)

    def test_recursive_leaf_events_keep_agent_live_after_scope_collection(self):
        self._write_populated(True)
        self.scope_path.rmdir()
        with (
            patch.object(state, "CGROUP_ROOT", self.cgroup_root),
            patch.object(
                state,
                "_systemctl_show",
                side_effect=self._show(scope=self._scope_properties(present=False)),
            ),
        ):
            self.assertTrue(state._schema2_liveness(self.agent_dir, self.reference))

    def test_recursive_leaf_events_prove_exact_descendant_absence(self):
        self._write_populated(False)
        self.scope_path.rmdir()
        with (
            patch.object(state, "CGROUP_ROOT", self.cgroup_root),
            patch.object(
                state,
                "_systemctl_show",
                side_effect=self._show(scope=self._scope_properties(present=False)),
            ),
        ):
            self.assertFalse(state._schema2_liveness(self.agent_dir, self.reference))

    def test_missing_scope_identity_with_existing_scope_cgroup_is_ambiguous(self):
        self._write_populated(False)
        with (
            patch.object(state, "CGROUP_ROOT", self.cgroup_root),
            patch.object(
                state,
                "_systemctl_show",
                side_effect=self._show(scope=self._scope_properties(present=False)),
            ),
        ):
            with self.assertRaisesRegex(state.ProcessIdentityError, "missing systemd"):
                state._schema2_liveness(self.agent_dir, self.reference)

    def test_slice_invocation_drift_refuses_liveness_authority(self):
        self._write_populated(True)
        with (
            patch.object(state, "CGROUP_ROOT", self.cgroup_root),
            patch.object(
                state,
                "_systemctl_show",
                side_effect=self._show(
                    slice_=self._slice_properties(InvocationID="c" * 32)
                ),
            ),
        ):
            with self.assertRaisesRegex(state.ProcessIdentityError, "slice identity drifted"):
                state._schema2_liveness(self.agent_dir, self.reference)

    def test_slice_cgroup_id_drift_refuses_liveness_authority(self):
        self._write_populated(True)
        with (
            patch.object(state, "CGROUP_ROOT", self.cgroup_root),
            patch.object(
                state,
                "_systemctl_show",
                side_effect=self._show(
                    slice_=self._slice_properties(ControlGroupId="8000")
                ),
            ),
        ):
            with self.assertRaisesRegex(state.ProcessIdentityError, "slice identity drifted"):
                state._schema2_liveness(self.agent_dir, self.reference)

    def test_empty_leaf_slice_is_stopped_then_revalidated_absent(self):
        self._write_populated(False)
        self.scope_path.rmdir()
        stopped = {"value": False}

        def show(unit, _properties):
            if unit == self.scope_unit:
                return self._scope_properties(present=False)
            if unit == self.slice_unit:
                return self._slice_properties(present=not stopped["value"])
            self.fail(f"unexpected unit readback: {unit}")

        def stop(unit):
            self.assertEqual(unit, self.slice_unit)
            (self.slice_path / "cgroup.events").unlink()
            self.slice_path.rmdir()
            stopped["value"] = True

        with (
            patch.object(state, "CGROUP_ROOT", self.cgroup_root),
            patch.object(state, "_systemctl_show", side_effect=show),
            patch.object(state, "_systemctl_stop", side_effect=stop) as exact_stop,
        ):
            state._retire_empty_schema2_slice(self.agent_dir, self.reference)

        exact_stop.assert_called_once_with(self.slice_unit)

    def test_schema2_sigterm_cleanup_avoids_sigkill(self):
        readback = (self.reference, {}, {"invocation_id": "b" * 32})
        with (
            patch.object(state, "_schema2_liveness", side_effect=[True, False]),
            patch.object(state, "_schema2_readback", return_value=readback),
            patch.object(state, "_retire_empty_schema2_slice") as retire,
            patch.object(state, "_systemctl_signal") as send,
        ):
            stopped = state.terminate_sub_agent(
                self.agent_dir, grace_seconds=0.1, poll_seconds=0.01
            )

        self.assertTrue(stopped)
        self.assertEqual(send.call_args_list, [call(self.slice_unit, signal.SIGTERM)])
        retire.assert_called_once_with(self.agent_dir, self.reference)

    def test_schema2_escalation_revalidates_and_uses_exact_slice(self):
        readback = (self.reference, {}, {"invocation_id": "b" * 32})
        with (
            patch.object(
                state, "_schema2_liveness", side_effect=[True, True, False]
            ),
            patch.object(state, "_schema2_readback", return_value=readback) as readback_call,
            patch.object(state, "_read_cgroup_populated", return_value=True),
            patch.object(state, "_retire_empty_schema2_slice"),
            patch.object(state, "_systemctl_signal") as send,
        ):
            stopped = state.terminate_sub_agent(
                self.agent_dir, grace_seconds=0, poll_seconds=0.01
            )

        self.assertTrue(stopped)
        self.assertEqual(
            send.call_args_list,
            [
                call(self.slice_unit, signal.SIGTERM),
                call(self.slice_unit, signal.SIGKILL),
            ],
        )
        self.assertGreaterEqual(readback_call.call_count, 2)

    def test_schema2_identity_drift_during_grace_refuses_escalation(self):
        readback = (self.reference, {}, {"invocation_id": "b" * 32})
        with (
            patch.object(
                state,
                "_schema2_liveness",
                side_effect=[True, state.ProcessIdentityError("slice identity drifted")],
            ),
            patch.object(state, "_schema2_readback", return_value=readback),
            patch.object(state, "_systemctl_signal") as send,
        ):
            with self.assertRaisesRegex(state.ProcessIdentityError, "identity drifted"):
                state.terminate_sub_agent(
                    self.agent_dir, grace_seconds=0.1, poll_seconds=0.01
                )

        self.assertEqual(send.call_args_list, [call(self.slice_unit, signal.SIGTERM)])

    def test_nonzero_systemctl_result_is_fail_closed(self):
        result = type(
            "Result",
            (),
            {"returncode": 1, "stdout": "", "stderr": "unit changed"},
        )()
        with patch.object(state.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(state.ProcessIdentityError, "returned 1"):
                state._systemctl_signal(self.slice_unit, signal.SIGTERM)

    def test_legacy_sigterm_grace_allows_proven_group_absence(self):
        with (
            patch.object(
                state, "_validate_legacy_sub_agent", return_value=(4242, 4242)
            ),
            patch.object(state.os, "pidfd_open", return_value=73),
            patch.object(state, "_pidfd_exited", return_value=True),
            patch.object(state, "_legacy_group_absent", return_value=True),
            patch.object(state.os, "getsid", return_value=4242),
            patch.object(state.os, "killpg") as killpg,
            patch.object(state.os, "close") as close,
        ):
            stopped = state.terminate_sub_agent(
                self.legacy_dir, grace_seconds=0.1, poll_seconds=0.01
            )

        self.assertTrue(stopped)
        self.assertEqual(killpg.call_args_list, [call(4242, signal.SIGTERM)])
        close.assert_called_once_with(73)

    def test_legacy_dead_leader_with_remaining_group_refuses_escalation(self):
        with (
            patch.object(
                state, "_validate_legacy_sub_agent", return_value=(4242, 4242)
            ),
            patch.object(state.os, "pidfd_open", return_value=74),
            patch.object(state, "_pidfd_exited", return_value=True),
            patch.object(state, "_legacy_group_absent", return_value=False),
            patch.object(state.os, "getsid", return_value=4242),
            patch.object(state.os, "killpg") as killpg,
            patch.object(state.os, "close"),
        ):
            with self.assertRaisesRegex(
                state.ProcessIdentityError, "refusing unprovable descendant"
            ):
                state.terminate_sub_agent(
                    self.legacy_dir, grace_seconds=0, poll_seconds=0.01
                )

        self.assertEqual(killpg.call_args_list, [call(4242, signal.SIGTERM)])

    def test_legacy_leader_already_gone_with_remaining_group_is_ambiguous(self):
        with (
            patch.object(
                state, "_validate_legacy_sub_agent", side_effect=ProcessLookupError
            ),
            patch.object(state, "_legacy_group_absent", return_value=False),
            patch.object(state.os, "pidfd_open") as pidfd_open,
            patch.object(state.os, "killpg") as killpg,
        ):
            with self.assertRaisesRegex(
                state.ProcessIdentityError, "already exited.*group remains"
            ):
                state.terminate_sub_agent(self.legacy_dir)

        pidfd_open.assert_not_called()
        killpg.assert_not_called()

    def test_legacy_live_leader_escalates_only_its_revalidated_group(self):
        with (
            patch.object(
                state, "_validate_legacy_sub_agent", return_value=(4242, 4242)
            ) as validate,
            patch.object(state.os, "pidfd_open", return_value=75),
            patch.object(state, "_pidfd_exited", return_value=False),
            patch.object(state.os, "getsid", return_value=4242),
            patch.object(state.os, "killpg") as killpg,
            patch.object(state.os, "close"),
        ):
            stopped = state.terminate_sub_agent(
                self.legacy_dir, grace_seconds=0, poll_seconds=0.01
            )

        self.assertTrue(stopped)
        self.assertEqual(
            killpg.call_args_list,
            [call(4242, signal.SIGTERM), call(4242, signal.SIGKILL)],
        )
        self.assertGreaterEqual(validate.call_count, 3)

    def test_module_never_enumerates_global_procfs(self):
        source = inspect.getsource(state)
        self.assertNotIn('Path("/proc").iterdir()', source)
        self.assertNotIn("pidfd_send_signal", source)


if __name__ == "__main__":
    unittest.main()
