"""Hermetic tests for the lazy, protected Project Manager registry row."""

from __future__ import annotations

import unittest
import uuid

from aeon.remote.project_manager import (
    PROJECT_MANAGER_INSTANCE_ID,
    PROJECT_MANAGER_NAME,
    PROJECT_MANAGER_TMUX_NAME,
    PROJECT_MANAGER_WORKSPACE,
    ProjectManagerError,
    ProjectManagerInvariantError,
    ProjectManagerProtectedError,
    build_project_manager_record,
    dormant_project_manager_status,
    ensure_project_manager,
    is_first_project_manager_activation,
    is_project_manager_id,
    project_manager_public_flags,
    reject_project_manager_deletion,
)


class FakeStore:
    """Minimal store double; it has no process, tmux, filesystem, or GPU API."""

    def __init__(self):
        self.rows = {}
        self.create_calls = []
        self.race_record = None
        self.fail_creation = False

    def get_instance(self, instance_id):
        row = self.rows.get(instance_id)
        return dict(row) if row is not None else None

    def create_instance(self, record):
        self.create_calls.append(dict(record))
        if self.race_record is not None:
            self.rows[self.race_record["id"]] = dict(self.race_record)
            raise RuntimeError("simulated uniqueness race")
        if self.fail_creation:
            raise RuntimeError("simulated database failure")
        self.rows[record["id"]] = dict(record)


class TestProjectManagerLifecycle(unittest.TestCase):
    def test_canonical_record_is_uuid_shaped_exact_home_and_dormant(self):
        record = build_project_manager_record(default_model="fixture-model", now=123.5)

        self.assertEqual(uuid.UUID(hex=PROJECT_MANAGER_INSTANCE_ID).hex, record["id"])
        self.assertEqual(len(record["id"]), 32)
        self.assertEqual(record["name"], PROJECT_MANAGER_NAME)
        self.assertEqual(record["tmux_name"], PROJECT_MANAGER_TMUX_NAME)
        self.assertEqual(record["workspace"], PROJECT_MANAGER_WORKSPACE)
        self.assertEqual(record["workspace"], "/home/aday")
        self.assertEqual(record["kind"], "terminal")
        self.assertEqual(record["shell_backed"], 1)
        self.assertEqual(record["last_agent_kind"], "aeon")
        self.assertEqual(record["status"], "idle")
        self.assertEqual(record["desired_state"], "stopped")
        self.assertIsNone(record["last_started_at"])
        self.assertEqual(record["model"], "fixture-model")
        self.assertEqual(record["created_at"], 123.5)
        self.assertEqual(record["updated_at"], 123.5)

    def test_ensure_creates_only_once_and_never_rewrites_runtime_state(self):
        store = FakeStore()
        created, was_created = ensure_project_manager(
            store, default_model="fixture-model", now=100
        )
        self.assertTrue(was_created)
        self.assertEqual(len(store.create_calls), 1)
        self.assertEqual(created["status"], "idle")

        store.rows[PROJECT_MANAGER_INSTANCE_ID].update(
            status="running", desired_state="running", model="new-selection"
        )
        existing, was_created = ensure_project_manager(
            store, default_model="ignored-model", now=200
        )
        self.assertFalse(was_created)
        self.assertEqual(len(store.create_calls), 1)
        self.assertEqual(existing["status"], "running")
        self.assertEqual(existing["desired_state"], "running")
        self.assertEqual(existing["model"], "new-selection")

    def test_ensure_accepts_only_a_valid_winner_of_a_creation_race(self):
        store = FakeStore()
        winner = build_project_manager_record(default_model="winner", now=90)
        store.race_record = winner

        existing, was_created = ensure_project_manager(
            store, default_model="loser", now=100
        )
        self.assertFalse(was_created)
        self.assertEqual(existing["model"], "winner")
        self.assertEqual(len(store.create_calls), 1)

        invalid_store = FakeStore()
        invalid_store.race_record = {**winner, "tmux_name": "wrong-target"}
        with self.assertRaises(ProjectManagerInvariantError):
            ensure_project_manager(invalid_store, default_model="fixture")

    def test_current_workspace_is_mutable_runtime_state(self):
        store = FakeStore()
        record = build_project_manager_record(default_model="fixture", now=100)
        record["workspace"] = "/home/aday/dashboard"
        store.rows[record["id"]] = record

        existing, was_created = ensure_project_manager(
            store, default_model="ignored", now=200
        )

        self.assertFalse(was_created)
        self.assertEqual(existing["workspace"], "/home/aday/dashboard")
        self.assertEqual(store.create_calls, [])

    def test_user_facing_name_is_mutable_without_changing_protected_identity(self):
        store = FakeStore()
        record = build_project_manager_record(default_model="fixture", now=100)
        record["name"] = "Operations lead"
        store.rows[record["id"]] = record

        existing, was_created = ensure_project_manager(
            store, default_model="ignored", now=200
        )

        self.assertFalse(was_created)
        self.assertEqual(existing["name"], "Operations lead")
        self.assertEqual(store.create_calls, [])

    def test_creation_failure_without_a_durable_row_fails_closed(self):
        store = FakeStore()
        store.fail_creation = True
        with self.assertRaisesRegex(ProjectManagerError, "Could not materialize"):
            ensure_project_manager(store, default_model="fixture")
        self.assertEqual(store.rows, {})

    def test_protected_identity_is_not_silently_repaired(self):
        store = FakeStore()
        record = build_project_manager_record(default_model="fixture", now=100)
        record["launch_origin"] = "web"
        store.rows[record["id"]] = record

        with self.assertRaisesRegex(
            ProjectManagerInvariantError, "launch_origin"
        ):
            ensure_project_manager(store, default_model="fixture")
        self.assertEqual(len(store.create_calls), 0)

    def test_public_flags_and_idle_reconciliation_are_exact_id_only(self):
        record = build_project_manager_record(default_model="fixture", now=100)
        flags = project_manager_public_flags(record)
        self.assertEqual(
            flags,
            {
                "pinned": True,
                "always_present": True,
                "lazy_start": True,
                "role": "project_manager",
            },
        )
        self.assertEqual(
            dormant_project_manager_status(record, pane_exists=False), "idle"
        )
        self.assertEqual(
            dormant_project_manager_status(
                record, pane_exists=True, pane_dead=True
            ),
            "idle",
        )
        self.assertIsNone(
            dormant_project_manager_status(record, pane_exists=True, pane_dead=False)
        )

        running = {**record, "desired_state": "running"}
        self.assertIsNone(
            dormant_project_manager_status(running, pane_exists=False)
        )
        ordinary = {**record, "id": "0" * 32, "name": PROJECT_MANAGER_NAME}
        self.assertFalse(project_manager_public_flags(ordinary)["pinned"])
        self.assertIsNone(
            dormant_project_manager_status(ordinary, pane_exists=False)
        )

    def test_delete_guard_blocks_only_the_stable_project_manager_id(self):
        self.assertTrue(is_project_manager_id(PROJECT_MANAGER_INSTANCE_ID))
        with self.assertRaisesRegex(ProjectManagerProtectedError, "permanent"):
            reject_project_manager_deletion(PROJECT_MANAGER_INSTANCE_ID)

        # A lookalike name is not enough to make an unrelated row undeletable.
        reject_project_manager_deletion("0" * 32)

    def test_only_the_virgin_project_manager_uses_its_initial_objective(self):
        record = build_project_manager_record(default_model="fixture", now=100)
        self.assertTrue(is_first_project_manager_activation(record))
        self.assertFalse(
            is_first_project_manager_activation({**record, "last_started_at": 101})
        )
        self.assertFalse(
            is_first_project_manager_activation({**record, "id": "0" * 32})
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
