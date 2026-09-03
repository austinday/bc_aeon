"""Lightweight console front door for Aeon.

Help, version, and diagnostics intentionally avoid importing ``aeon.main`` so
they cannot run model/container lifecycle code.
"""

from __future__ import annotations

import argparse
import sys

from aeon import __version__
from aeon.core.model_identity import AEON_DEFAULT_MODEL_NAME
from aeon.harnesses.catalog import (
    DEFAULT_HARNESS_ID,
    HARNESS_CATALOG,
    LEGACY_AEON_HARNESS_ID,
    OPENCODE_HARNESS_ID,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aeon",
        description=(
            "Aeon — a fleet-local autonomous agent harness powered by the "
            "Qwen3.8-Flash-Next primary model with an automatic Qwen3.8-27B "
            "RTX 5000 fallback. Run it from the directory you want it to work in."
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
    parser.add_argument(
        "--harness",
        choices=tuple(HARNESS_CATALOG),
        default=DEFAULT_HARNESS_ID,
        help="Agent harness (default: opencode; legacy-aeon remains available)",
    )
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
    if options.model is not None and options.model != AEON_DEFAULT_MODEL_NAME:
        parser.error(
            f"only the primary model {AEON_DEFAULT_MODEL_NAME!r} is supported"
        )
    if options.dual:
        parser.error("--dual is reserved until a coordinator-safe dual-copy profile is released")
    if options.max_iterations is not None and options.max_iterations < 1:
        parser.error("--max-iterations must be a positive integer")
    if options.non_interactive and not (options.start or options.resume):
        parser.error('--non-interactive requires --start "<objective>"')
    if options.harness == LEGACY_AEON_HARNESS_ID:
        # aeon.main remains an interchangeable compatibility harness but knows
        # nothing about the front-door selector itself.
        legacy_arguments: list[str] = []
        skip_next = False
        for argument in arguments:
            if skip_next:
                skip_next = False
                continue
            if argument == "--harness":
                skip_next = True
                continue
            if argument.startswith("--harness="):
                continue
            legacy_arguments.append(argument)
        from aeon.main import cli as runtime_cli

        return runtime_cli(legacy_arguments)

    if options.harness != OPENCODE_HARNESS_ID:  # pragma: no cover - argparse guards this
        parser.error("unsupported harness")

    incompatible = [
        name
        for enabled, name in (
            (options.debug, "--debug"),
            (options.debug_log is not None, "--debug-log"),
            (options.menu, "--menu"),
            (options.no_warmup, "--no-warmup"),
            (options.resume is not None, "--resume"),
        )
        if enabled
    ]
    if incompatible:
        parser.error(
            f"{', '.join(incompatible)} require --harness {LEGACY_AEON_HARNESS_ID}"
        )

    from aeon.harnesses.opencode_config import MAX_OPENCODE_STEPS

    if options.max_iterations is not None and options.max_iterations > MAX_OPENCODE_STEPS:
        parser.error(
            f"--max-iterations cannot exceed {MAX_OPENCODE_STEPS} with the OpenCode harness"
        )
    runtime_arguments = ["--model", options.model or AEON_DEFAULT_MODEL_NAME]
    if options.start:
        runtime_arguments.extend(["--start", options.start])
    if options.non_interactive:
        runtime_arguments.append("--non-interactive")
    if options.max_iterations is not None:
        runtime_arguments.extend(["--max-iterations", str(options.max_iterations)])
    if options.browser_profile:
        runtime_arguments.extend(["--browser-profile", options.browser_profile])

    # Interactive local starts retain Aeon's managed-tmux adoption. Pass only
    # parsed, rebuilt OpenCode arguments so neither browser state nor stale
    # front-door selectors can become an inner command.
    from aeon.main import _auto_adopt_tmux

    if _auto_adopt_tmux(
        options,
        cli_args=runtime_arguments,
        harness=OPENCODE_HARNESS_ID,
    ):
        return None

    from aeon.harnesses.opencode_runtime import main as runtime_main

    return runtime_main(runtime_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
