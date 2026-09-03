#!/usr/bin/env python3
"""Permanent fail-closed tombstone for the retired SGLang promotion command.

The former implementation rewrote four Fleet profiles into a two-Flash-lane
SGLang serving pool. That topology predates the final one-RTX vLLM release and
is incompatible with its atomic replacement of the compact lane. Keeping a
non-authorizing command at the historical import/CLI path prevents old operator
notes or automation from silently recreating a mixed serving pool.

The current promotion is deliberately not implemented here. It is gated by the
disabled ``aeon-qwen38-flash-next-vllm-177`` profile, its exact v20 canary
receipt and promotion binding, and the reviewed Fleet reload procedure.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any, NoReturn


RETIRED_REASON = (
    "retired SGLang four-profile promotion cannot authorize the one-RTX vLLM "
    "release; use the exact v20 vLLM canary, binding, and atomic Fleet rollout"
)


class PromotionError(RuntimeError):
    """The retired promotion surface is permanently non-authorizing."""


def _retired() -> NoReturn:
    raise PromotionError(RETIRED_REASON)


def build_promoted_profiles(
    current: Mapping[str, Mapping[str, Any]],
    *,
    artifact_identity: Mapping[str, str],
    remote_artifact_cache: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Refuse direct callers of the former pure profile transformer."""

    del current, artifact_identity, remote_artifact_cache
    _retired()


def _validate_registry(
    replacements: Mapping[str, Mapping[str, Any]],
    *,
    profiles_dir: Path | None = None,
) -> None:
    """Refuse validation through the obsolete mixed-pool contract."""

    del replacements, profiles_dir
    _retired()


def _replace_transaction(
    replacements: Mapping[str, Mapping[str, Any]],
    *,
    binding_payload: Mapping[str, Any],
) -> None:
    """Refuse every legacy filesystem mutation entry point."""

    del replacements, binding_payload
    _retired()


def prepare_promotion(
    *,
    repo_id: str,
    publication_receipt: Path,
    verify_release_hashes: bool = True,
) -> NoReturn:
    """Refuse preview and execute callers before reading release evidence."""

    del repo_id, publication_receipt, verify_release_hashes
    _retired()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Preserve the historical argument shape so old automation reaches the
    # explicit tombstone instead of failing ambiguously in argument parsing.
    parser.add_argument("--repo-id")
    parser.add_argument("--publication-receipt", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    print(f"promotion failed: {RETIRED_REASON}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
