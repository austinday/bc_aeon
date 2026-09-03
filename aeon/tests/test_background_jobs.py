from __future__ import annotations

import os
import re
import signal
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from aeon.core import process_identity
from aeon.tools.jobs import JobOutput, KillJob, RunCommandAsync, jobs_base, resolve_job


def _job_id(message: str) -> str:
    match = re.search(r"Job ID: ([0-9a-f-]{36})", message)
    if match is None:
        raise AssertionError(message)
    return match.group(1)


class BackgroundJobIdentityTests(unittest.TestCase):
    def test_pid_reuse_is_rejected_before_a_group_signal(self):
        reference = {
            "schema": 1,
            "identity": "job/test",
            "pid": 1234,
            "pgid": 1234,
            "start_ticks": 100,
        }
        with (
            patch.object(
                process_identity,
                "_proc_fields",
                return_value=("S", 1, 1234, 101),
            ),
            patch.object(process_identity.os, "getpgid", return_value=1234),
            patch.object(process_identity.os, "killpg") as killpg,
        ):
            with self.assertRaises(process_identity.ProcessIdentityError):
                process_identity.signal_process_group(
                    reference, "job/test", signal.SIGKILL
                )
        killpg.assert_not_called()

    def test_short_job_completes_from_a_path_containing_a_quote(self):
        with tempfile.TemporaryDirectory(prefix="aeon-job-'quote-") as temporary:
            previous = os.getcwd()
            os.chdir(temporary)
            try:
                worker = types.SimpleNamespace(
                    instance_id="job-test", notified_jobs=set()
                )
                job_id = _job_id(
                    RunCommandAsync(worker=worker).execute(
                        "printf 'receipt-safe\\n'", timeout=10
                    )
                )
                job_dir = jobs_base(worker) / job_id
                self.assertEqual(job_dir.stat().st_mode & 0o777, 0o700)
                deadline = time.monotonic() + 5
                terminal = False
                status = "RUNNING"
                while time.monotonic() < deadline:
                    terminal, status, _code = resolve_job(job_dir)
                    if terminal:
                        break
                    time.sleep(0.01)
                self.assertTrue(terminal)
                self.assertEqual(status, "COMPLETED")
                self.assertIn("receipt-safe", JobOutput(worker=worker).execute(job_id))
            finally:
                os.chdir(previous)

    def test_kill_uses_exact_unit_and_invocation_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.getcwd()
            os.chdir(temporary)
            worker = types.SimpleNamespace(instance_id="job-test", notified_jobs=set())
            job_id = None
            try:
                job_id = _job_id(
                    RunCommandAsync(worker=worker).execute("sleep 30", timeout=60)
                )
                result = KillJob(worker=worker).execute(job_id)
                self.assertIn("exact unit/InvocationID receipt", result)
                self.assertEqual(resolve_job(jobs_base(worker) / job_id)[1], "KILLED")
            finally:
                if job_id is not None:
                    KillJob(worker=worker).execute(job_id)
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
