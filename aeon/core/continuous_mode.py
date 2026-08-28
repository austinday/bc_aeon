"""Owner-controlled continuous-mode settings for one managed Aeon process.

Nexus atomically replaces a small mode-600 JSON file.  Aeon re-reads that file
only between conversational turns, so disabling the mode never races or mutates
an in-flight tool action.  The file is a control signal, not an authorization
expander: the generated continuation explicitly preserves all existing safety
and approval boundaries.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path


CONTINUOUS_MODE_ENV = "AEON_CONTINUOUS_MODE_PATH"
CONTINUOUS_MODE_FILENAME = "continuous-mode.json"
CONTINUOUS_MODE_SCHEMA_VERSION = 1
MAX_CONTINUOUS_GOAL_BYTES = 20_000

# This line is delivered only through Nexus's private, server-built tmux buffer.
# It is never treated as a user message or placed in model history.
NEXUS_CONTINUOUS_WAKE_COMMAND = "/__nexus_continuous_mode_changed_64c92f1a__"


class ContinuousModeError(ValueError):
    """Continuous-mode state is invalid or cannot be trusted."""


def _word_count(value: str) -> int:
    return len(re.findall(r"\S+", value, flags=re.UNICODE))


def normalize_continuous_goal(goal: object, *, enabled: bool = False) -> str:
    if not isinstance(goal, str):
        raise ContinuousModeError("Continuous goal must be text")
    value = goal.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\x00" in value:
        raise ContinuousModeError("Continuous goal contains an invalid NUL character")
    if len(value.encode("utf-8")) > MAX_CONTINUOUS_GOAL_BYTES:
        raise ContinuousModeError(
            f"Continuous goal must be at most {MAX_CONTINUOUS_GOAL_BYTES} UTF-8 bytes"
        )
    if enabled and _word_count(value) <= 2:
        raise ContinuousModeError(
            "Continuous mode requires a goal containing more than two words"
        )
    return value


@dataclass(frozen=True)
class ContinuousModeState:
    enabled: bool = False
    goal: str = ""
    updated_at: float | None = None

    def prompt(self) -> str:
        goal = normalize_continuous_goal(self.goal, enabled=True)
        return (
            "CONTINUOUS MODE: Begin another autonomous work cycle toward the "
            "durable user-configured goal below. This is not an answer to any "
            "question you previously asked, and it grants no new authority.\n\n"
            f"GOAL:\n{goal}\n\n"
            "This is the same durable goal as the preceding autonomous cycles, not an "
            "unrelated task. Reuse the durable plan, memories, evidence records, and "
            "project artifacts, while rechecking anything freshness-sensitive; report "
            "only the material delta from earlier cycles. Choose the "
            "highest-value safe next action, make concrete progress, and verify it. "
            "A final response ends only this cycle: never describe the overall search as "
            "exhausted, say there is no distinct safe work left, or stop merely because a "
            "conversational turn ended or optional guidance would help. When one lane is "
            "blocked or repeatedly unproductive, deliberately branch to a materially "
            "different source, method, hypothesis, modality, scale, or contribution type. "
            "Do not retry an unchanged failure until a named precondition changes.\n\n"
            "EVIDENCE DISCIPLINE: a successful write, readback, hash, or test validates "
            "only that artifact or behavior; it does not validate factual claims stored "
            "inside it. Failure to find a competing artifact is not evidence that none "
            "exists. Reserve words such as validated, confirmed, covered, closed, winner, "
            "and decision-ready for conclusions supported by current primary-source "
            "evidence and all material prerequisites. Before committing to a candidate, "
            "check identity/version, release date, size and architecture, license and "
            "redistribution terms, real resource/toolchain needs, existing alternatives, "
            "differentiation and user value, and a reproducible validation plan. Record "
            "unknowns and contradictions explicitly, and demote a candidate before "
            "building downstream templates or promotional assets. Prefer authoritative "
            "metadata/APIs and repository evidence over search-result inference.\n\n"
            "Preserve every existing safety, permission, financial, credential, "
            "renter-priority, and external-side-effect boundary. If one useful action truly "
            "requires missing authority or an external change, retain that exact blocker "
            "and pursue a different safe line of work rather than fabricating progress, "
            "bypassing a refusal, or repeating the same status report."
        )


def serialize_continuous_mode(state: ContinuousModeState) -> bytes:
    goal = normalize_continuous_goal(state.goal, enabled=state.enabled)
    return (
        json.dumps(
            {
                "version": CONTINUOUS_MODE_SCHEMA_VERSION,
                "enabled": bool(state.enabled),
                "goal": goal,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def load_continuous_mode(path_value: str | os.PathLike[str] | None) -> ContinuousModeState:
    if not path_value:
        return ContinuousModeState()
    path = Path(path_value)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size < 1
            or metadata.st_size > MAX_CONTINUOUS_GOAL_BYTES + 1024
        ):
            raise ContinuousModeError("Continuous-mode control file is not owner-safe")
        payload = os.read(descriptor, MAX_CONTINUOUS_GOAL_BYTES + 1025)
        if len(payload) > MAX_CONTINUOUS_GOAL_BYTES + 1024:
            raise ContinuousModeError("Continuous-mode control file is too large")
        document = json.loads(payload.decode("utf-8"))
    except FileNotFoundError:
        return ContinuousModeState()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuousModeError("Continuous-mode control file is invalid") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(document, dict) or document.get("version") != CONTINUOUS_MODE_SCHEMA_VERSION:
        raise ContinuousModeError("Continuous-mode control version is invalid")
    enabled = document.get("enabled")
    if not isinstance(enabled, bool):
        raise ContinuousModeError("Continuous-mode enabled state is invalid")
    goal = normalize_continuous_goal(document.get("goal"), enabled=enabled)
    return ContinuousModeState(enabled=enabled, goal=goal)


def load_continuous_mode_from_environment() -> ContinuousModeState:
    return load_continuous_mode(os.environ.get(CONTINUOUS_MODE_ENV))


__all__ = (
    "CONTINUOUS_MODE_ENV",
    "CONTINUOUS_MODE_FILENAME",
    "ContinuousModeError",
    "ContinuousModeState",
    "MAX_CONTINUOUS_GOAL_BYTES",
    "NEXUS_CONTINUOUS_WAKE_COMMAND",
    "load_continuous_mode",
    "load_continuous_mode_from_environment",
    "normalize_continuous_goal",
    "serialize_continuous_mode",
)
