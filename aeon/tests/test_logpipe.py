"""Regression tests for the tmux terminal transcript sink."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aeon.remote.logpipe import stream_to_log


class LogPipeTests(unittest.TestCase):
    def test_partial_chunk_is_written_before_input_pipe_closes(self):
        """A live pipe must not wait for a full 64 KiB buffered read."""

        read_fd, write_fd = os.pipe()
        raw_input = os.fdopen(read_fd, "rb", buffering=0)
        buffered_input = io.BufferedReader(raw_input)
        payload = b"small live terminal update\n"
        observed_while_open = False

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terminal.log"
            with patch.object(
                sys, "stdin", SimpleNamespace(buffer=buffered_input)
            ):
                thread = threading.Thread(
                    target=stream_to_log,
                    args=(path,),
                    daemon=True,
                )
                thread.start()
                try:
                    os.write(write_fd, payload)
                    deadline = time.monotonic() + 1.0
                    while time.monotonic() < deadline:
                        if path.exists() and path.read_bytes() == payload:
                            observed_while_open = True
                            break
                        time.sleep(0.01)
                finally:
                    os.close(write_fd)
                    thread.join(timeout=1.0)
                    buffered_input.close()

            self.assertTrue(
                observed_while_open,
                "sub-64 KiB output was buffered until the live input pipe closed",
            )
            self.assertFalse(thread.is_alive())
            self.assertEqual(path.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
