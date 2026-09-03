"""External benchmark worker launched only through fleet-low-priority."""

from __future__ import annotations

import argparse
import signal
import threading
from pathlib import Path
from typing import Sequence

from .executor import FleetHarnessExecutor
from .runner import run_benchmark
from .service import BenchmarkService, RUN_ID_RE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one durable Aeon benchmark")
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _install_termination_handlers() -> dict[int, object]:
    """Turn service termination into normal context-manager unwinding."""

    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: dict[int, object] = {}

    def terminate(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate)
    return previous


def _restore_termination_handlers(previous: dict[int, object]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not RUN_ID_RE.fullmatch(args.run_id):
        return 2
    previous_handlers = _install_termination_handlers()
    service: BenchmarkService | None = None
    try:
        service = BenchmarkService(Path(args.root))
        with FleetHarnessExecutor(service, args.run_id) as executor:
            result = run_benchmark(service, args.run_id, executor=executor)
    except Exception:
        if service is not None:
            try:
                service._mark_failed(args.run_id, error_code="runner_failed")
            except Exception:
                pass
        return 1
    finally:
        _restore_termination_handlers(previous_handlers)
    return 0 if result.get("status") in {"succeeded", "cancelled"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
