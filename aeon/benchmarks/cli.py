"""Sanitized command-line interface for the durable benchmark service."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Callable, Sequence

from .service import BenchmarkService


DEFAULT_BENCHMARK_ROOT = Path.home() / ".local" / "share" / "aeon" / "benchmarks"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m aeon.benchmarks")
    parser.add_argument(
        "--root",
        default=os.environ.get("AEON_BENCHMARK_HOME", str(DEFAULT_BENCHMARK_ROOT)),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog")
    listing = commands.add_parser("list")
    listing.add_argument("--limit", type=int, default=100)
    show = commands.add_parser("show")
    show.add_argument("run_id")
    cancel = commands.add_parser("cancel")
    cancel.add_argument("run_id")
    submit = commands.add_parser("submit")
    submit.add_argument("--request-id", default=None)
    submit.add_argument("--suite", required=True)
    submit.add_argument("--harness", required=True)
    submit.add_argument("--model", required=True)
    submit.add_argument("--tool-profile", default=None)
    submit.add_argument("--repetitions", type=int, default=1)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    service_factory: Callable[[Path], BenchmarkService] = BenchmarkService,
) -> int:
    args = _parser().parse_args(argv)
    try:
        service = service_factory(Path(args.root))
        if args.command == "catalog":
            result = service.catalog()
        elif args.command == "list":
            result = service.list_runs(limit=args.limit)
        elif args.command == "show":
            result = service.get_run(args.run_id)
        elif args.command == "cancel":
            result = service.cancel(args.run_id)
        else:
            request = {
                "request_id": args.request_id or f"br-{uuid.uuid4().hex}",
                "suite_id": args.suite,
                "harness_id": args.harness,
                "model_id": args.model,
                "repetitions": args.repetitions,
            }
            if args.tool_profile:
                request["tool_profile_id"] = args.tool_profile
            result = service.submit(request)
    except (KeyError, TypeError, ValueError, RuntimeError):
        print(json.dumps({"error": "benchmark_request_failed"}, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = ("DEFAULT_BENCHMARK_ROOT", "main")
