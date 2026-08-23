"""Model-free tests for Aeon's automatic local presence records."""

from __future__ import annotations

import json
import stat
import tempfile
import unittest
import uuid
from pathlib import Path

from aeon.core.presence import (
    Presence,
    manifest_process_is_live,
    process_instance_id,
    validate_remote_instance_id,
)
from aeon.core.worker import Worker


class TestPresenceManifest(unittest.TestCase):
    def test_remote_id_validation_and_fallback_identity_are_stable(self):
        remote = uuid.uuid4()
        self.assertEqual(validate_remote_instance_id(str(remote)), remote.hex)
        self.assertEqual(validate_remote_instance_id(remote.hex.upper()), remote.hex)
        self.assertIsNone(validate_remote_instance_id("../not-an-instance"))
        self.assertIsNone(validate_remote_instance_id("00000000000000000000000000000000"))

        first = process_instance_id()
        second = process_instance_id()
        self.assertEqual(first, second)
        self.assertEqual(uuid.UUID(first).hex, first)

    def test_manifest_is_private_atomic_sanitized_and_uniquely_named(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            presence_dir = base / "presence"
            remote = uuid.uuid4()
            presence = Presence(
                presence_dir=presence_dir,
                cwd=str(base),
                environ={"AEON_REMOTE_INSTANCE_ID": str(remote)},
                register_atexit=False,
            )
            presence.start_objective(
                "Deploy https://user:pass@example.test/path?q=hidden "
                "token=super-secret-value "
                "sk-abcdefghijklmnopqrstuvwxyz1234567890",
                model="local-model",
            )
            presence.update(
                phase="acting",
                iteration=7,
                intent="Call API with Authorization: Bearer abcdefghijklmnop",
                current_plan="First step; password: hunter2; opaque=" + "a" * 60,
            )

            raw = presence.path.read_text(encoding="utf-8")
            record = json.loads(raw)
            self.assertEqual(record["remote_instance_id"], remote.hex)
            self.assertEqual(record["instance_id"], remote.hex)
            self.assertEqual(record["launch_origin"], "remote")
            self.assertEqual(record["cwd"], str(base.resolve()))
            self.assertEqual(record["pid"], presence.pid)
            self.assertGreater(record["process_create_time"], 0)
            self.assertEqual(record["phase"], "acting")
            self.assertEqual(record["iteration"], 7)
            self.assertEqual(record["model"], "local-model")
            self.assertNotIn("prompt", record)
            self.assertNotIn("tool_output", record)
            for secret in (
                "super-secret-value",
                "abcdefghijklmnopqrstuvwxyz1234567890",
                "abcdefghijklmnop",
                "hunter2",
                "user:pass",
                "q=hidden",
                "a" * 60,
            ):
                self.assertNotIn(secret, raw)

            self.assertEqual(stat.S_IMODE(presence_dir.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(presence.path.stat().st_mode), 0o600)
            self.assertFalse(list(presence_dir.glob("*.tmp")))
            self.assertTrue(manifest_process_is_live(record))

            presence.update_compute(
                state="waiting_for_compute",
                profile="qwen38-vllm",
                summary="Waiting for capacity; token=must-not-survive",
            )
            compute = json.loads(presence.path.read_text(encoding="utf-8"))
            self.assertEqual(compute["compute_state"], "waiting_for_compute")
            self.assertEqual(compute["compute_profile"], "qwen38-vllm")
            self.assertIsNotNone(compute["compute_wait_started_at"])
            self.assertNotIn("must-not-survive", json.dumps(compute))
            presence.update_compute(
                state="allocated",
                profile="qwen38-vllm",
                summary="Coordinator-approved local compute is allocated",
            )
            compute = json.loads(presence.path.read_text(encoding="utf-8"))
            self.assertEqual(compute["compute_state"], "allocated")
            self.assertIsNone(compute["compute_wait_started_at"])

            wrong_identity = dict(record)
            wrong_identity["process_create_time"] += 10
            self.assertFalse(manifest_process_is_live(wrong_identity))

            second = Presence(
                presence_dir=presence_dir,
                cwd=str(base),
                environ={"AEON_REMOTE_INSTANCE_ID": remote.hex},
                register_atexit=False,
            )
            self.assertNotEqual(second.path, presence.path)
            self.assertTrue(presence.path.exists())
            self.assertTrue(second.path.exists())

            presence.mark_error(RuntimeError("password=do-not-store-this"))
            errored = json.loads(presence.path.read_text(encoding="utf-8"))
            self.assertEqual(errored["phase"], "error")
            self.assertEqual(errored["error_type"], "RuntimeError")
            self.assertNotIn("do-not-store-this", json.dumps(errored))
            presence.mark_exit()
            exited = json.loads(presence.path.read_text(encoding="utf-8"))
            self.assertEqual(exited["phase"], "exited")
            self.assertFalse(manifest_process_is_live(exited))

    def test_invalid_remote_id_is_not_consumed(self):
        with tempfile.TemporaryDirectory() as temp:
            presence = Presence(
                presence_dir=Path(temp) / "presence",
                environ={"AEON_REMOTE_INSTANCE_ID": "../../escape"},
                register_atexit=False,
            )
            self.assertIsNone(presence.remote_instance_id)
            self.assertEqual(presence.launch_origin, "local")
            self.assertEqual(uuid.UUID(presence.instance_id).hex, presence.instance_id)


class RecordingPresence:
    def __init__(self):
        self.instance_id = uuid.uuid4().hex
        self.events = []

    def start_objective(self, objective, **fields):
        self.events.append(("objective", objective, fields))

    def update(self, **fields):
        self.events.append(("update", fields))

    def mark_completed(self, **fields):
        self.events.append(("completed", fields))

    def mark_error(self, error):
        self.events.append(("error", type(error).__name__))


class DummyLLM:
    pass


class TestWorkerPresenceLifecycle(unittest.TestCase):
    def _worker(self):
        presence = RecordingPresence()
        worker = Worker(llm_client=DummyLLM(), presence=presence)
        worker.model_name = "test-model"
        worker._start_input_listener = lambda: None
        worker._stop_input_listener = lambda: None
        return worker, presence

    def test_worker_records_objective_and_completion(self):
        worker, presence = self._worker()
        worker._run_objective = lambda *args, **kwargs: "done"
        self.assertEqual(worker.run("bounded objective"), "done")
        self.assertEqual(presence.events[0][0], "objective")
        self.assertEqual(presence.events[0][1], "bounded objective")
        self.assertEqual(presence.events[-1][0], "completed")
        self.assertEqual(worker.instance_id, presence.instance_id)

    def test_worker_records_fatal_error_without_swallowing_it(self):
        worker, presence = self._worker()

        def fail(*args, **kwargs):
            raise RuntimeError("sensitive tool output")

        worker._run_objective = fail
        with self.assertRaisesRegex(RuntimeError, "sensitive tool output"):
            worker.run("objective")
        self.assertEqual(presence.events[-1], ("error", "RuntimeError"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
