"""Bounded raw terminal transcript sink used by tmux pipe-pane."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def stream_to_log(path: Path, max_bytes: int = 50 * 1024 * 1024) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    input_fd = sys.stdin.buffer.fileno()
    output = None
    try:
        while True:
            # BufferedReader.read(size) may wait for ``size`` bytes while tmux's
            # pipe remains open.  Read the pipe descriptor directly so even a
            # small burst becomes visible in the live terminal log immediately.
            chunk = os.read(input_fd, 65536)
            if not chunk:
                break
            if output is None:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                output = os.fdopen(fd, "ab", buffering=0)
            if output.tell() + len(chunk) > max_bytes:
                output.flush()
                os.fsync(output.fileno())
                output.close()
                output = None
                rotated = path.with_suffix(path.suffix + ".1")
                os.replace(path, rotated)
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                output = os.fdopen(fd, "ab", buffering=0)
            output.write(chunk)
    finally:
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
            output.close()
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Aeon remote terminal transcript sink")
    parser.add_argument("--path", required=True)
    parser.add_argument("--max-mib", type=int, default=50)
    args = parser.parse_args()
    stream_to_log(Path(args.path), max(1, args.max_mib) * 1024 * 1024)


if __name__ == "__main__":
    main()
