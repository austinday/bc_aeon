from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from aeon.remote.store import RemoteStore


class McpPermissionStoreTests(unittest.TestCase):
    def test_permissions_replace_atomically_and_deleted_credentials_revoke_globally(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "remote.sqlite3"
            store = RemoteStore(database)
            now = time.time()
            for instance_id in ("1" * 32, "2" * 32):
                store.create_instance(
                    {
                        "id": instance_id,
                        "name": f"Agent {instance_id[0]}",
                        "tmux_name": f"aeon-{instance_id[:12]}",
                        "workspace": temporary,
                        "objective": "",
                        "max_iterations": None,
                        "model": "fixture",
                        "status": "created",
                        "desired_state": "stopped",
                        "created_at": now,
                        "updated_at": now,
                        "last_started_at": None,
                        "last_error": "",
                        "created_by": "test",
                        "launch_origin": "web",
                    }
                )

            first = "mcp_" + "a" * 32
            second = "mcp_" + "b" * 32
            self.assertEqual(
                store.set_instance_credentials(
                    "1" * 32, [second, first, first], actor="owner"
                ),
                [first, second],
            )
            store.set_instance_credentials("2" * 32, [first], actor="owner")
            self.assertEqual(store.list_instance_credentials("1" * 32), [first, second])

            reopened = RemoteStore(database)
            self.assertEqual(reopened.list_instance_credentials("1" * 32), [first, second])
            self.assertEqual(reopened.list_instance_credentials("2" * 32), [first])

            self.assertEqual(reopened.revoke_credential(first), 2)
            self.assertEqual(reopened.list_instance_credentials("1" * 32), [second])
            self.assertEqual(reopened.list_instance_credentials("2" * 32), [])

    def test_live_notification_marker_tracks_the_exact_permission_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RemoteStore(Path(temporary) / "remote.sqlite3")
            instance_id = "3" * 32
            now = time.time()
            store.create_instance(
                {
                    "id": instance_id,
                    "name": "Live agent",
                    "tmux_name": "aeon-live-agent",
                    "workspace": temporary,
                    "objective": "",
                    "max_iterations": None,
                    "model": "fixture",
                    "status": "running",
                    "desired_state": "running",
                    "created_at": now,
                    "updated_at": now,
                    "last_started_at": now,
                    "last_error": "",
                    "created_by": "test",
                    "launch_origin": "web",
                }
            )
            store.set_instance_credentials(instance_id, ["github"], actor="owner")
            self.assertFalse(
                store.instance_credentials_notification_current(instance_id, ["github"])
            )
            store.mark_instance_credentials_notified(instance_id, ["github"])
            self.assertTrue(
                store.instance_credentials_notification_current(instance_id, ["github"])
            )

            store.set_instance_credentials(
                instance_id, ["github", "mcp_" + "a" * 32], actor="owner"
            )
            self.assertFalse(
                store.instance_credentials_notification_current(
                    instance_id, ["github", "mcp_" + "a" * 32]
                )
            )
            with self.assertRaisesRegex(ValueError, "changed before notification"):
                store.mark_instance_credentials_notified(instance_id, ["github"])


if __name__ == "__main__":
    unittest.main()
