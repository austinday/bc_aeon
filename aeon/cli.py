"""Lightweight console front door for Aeon.

Help, version, and diagnostics intentionally avoid importing ``aeon.main`` so
they cannot run model/container lifecycle code.
"""

from __future__ import annotations

import argparse
import sys

from aeon import PRIMARY_MODEL_NAME, __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aeon",
        description=(
            "Aeon — a fleet-local autonomous agent harness powered by the "
            "Qwen3.8-27B primary model. Run it from the directory you want it to work in."
        ),
        epilog=(
            'Examples:\n'
            '  aeon\n'
            '  aeon --start "Summarize this repository"\n'
            '  aeon -n --start "Run the tests and fix failures"\n'
            '  aeon doctor'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"Aeon {__version__}")
    parser.add_argument("--debug", action="store_true", help="Enable detailed local LLM logging")
    parser.add_argument("--debug-log", metavar="PATH", help="Reasoning trace JSONL path")
    parser.add_argument("--model", metavar="NAME", help="Compatibility selector; only Qwen3.8 is accepted")
    parser.add_argument("--menu", "-i", action="store_true", help="Open optional account/configuration menu")
    parser.add_argument("--dual", action="store_true", help="Reserved; no dual-copy release is currently enabled")
    parser.add_argument("--start", metavar="OBJECTIVE", help="Start work immediately")
    parser.add_argument("--non-interactive", "-n", action="store_true", help="Exit after the objective completes")
    parser.add_argument("--max-iterations", metavar="N", type=int, help="Maximum turns for one objective")
    parser.add_argument("--no-warmup", action="store_true", help="Skip model warmup")
    parser.add_argument("--resume", metavar="PATH", help=argparse.SUPPRESS)
    parser.add_argument("--browser-profile", metavar="NAME", help="Persistent browser profile name")
    return parser


def main(argv: list[str] | None = None) -> int | None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "doctor":
        doctor_parser = argparse.ArgumentParser(prog="aeon doctor", description="Read-only Aeon readiness checks")
        doctor_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
        options = doctor_parser.parse_args(arguments[1:])
        from aeon.doctor import run_doctor

        return run_doctor(as_json=options.json)

    # Parse here only to make help/version side-effect free and reject malformed
    # invocations before the heavyweight runtime is imported. aeon.main performs
    # the authoritative parse again after import.
    parser = _parser()
    options = parser.parse_args(arguments)
    if options.model is not None and options.model != PRIMARY_MODEL_NAME:
        parser.error(f"only the primary model {PRIMARY_MODEL_NAME!r} is supported")
    if options.dual:
        parser.error("--dual is reserved until a coordinator-safe dual-copy profile is released")
    if options.max_iterations is not None and options.max_iterations < 1:
        parser.error("--max-iterations must be a positive integer")
    if options.non_interactive and not (options.start or options.resume):
        parser.error('--non-interactive requires --start "<objective>"')
    from aeon.main import cli as runtime_cli

    return runtime_cli(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
