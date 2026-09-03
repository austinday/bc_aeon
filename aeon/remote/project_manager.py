"""Durable, lazy lifecycle metadata for Nexus's main orchestrator.

The main orchestrator is *always present* as a registry row. Its base mode is a
plain shell rooted at ``/home/aday``; the authenticated Nexus chat starts Aeon in
that same logical instance as needed. Materializing the row itself never starts
tmux, a model, or a coordinator lease.

This module intentionally depends on a tiny store protocol (``get_instance``
and ``create_instance``) so provisioning can be unit-tested without touching a
live RemoteStore database or process state.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from typing import Any


# Existing remote instance IDs are UUID4 hex strings.  A namespaced UUID keeps
# the protected row compatible with code that expects the same 32-hex shape,
# while remaining stable across service restarts and database reconstruction.
PROJECT_MANAGER_INSTANCE_ID = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://nexus.bananacoconut.com/instances/project-manager",
).hex
PROJECT_MANAGER_NAME = "Main orchestrator"
PROJECT_MANAGER_TMUX_NAME = f"aeon-project-manager-{PROJECT_MANAGER_INSTANCE_ID[:12]}"
PROJECT_MANAGER_WORKSPACE = "/home/aday"
PROJECT_MANAGER_LAUNCH_ORIGIN = "system"
PROJECT_MANAGER_CREATED_BY = "nexus"
PROJECT_MANAGER_OBJECTIVE = (
    "Act as the persistent project manager for Nexus, Aeon, and Fleet Compute. "
    "Coordinate work carefully across the user's projects and preserve renter-first "
    "compute policy."
)


class ProjectManagerError(RuntimeError):
    """Base error for the protected Project Manager registry row."""


class ProjectManagerInvariantError(ProjectManagerError):
    """The stable ID exists, but its protected identity is not trustworthy."""


class ProjectManagerProtectedError(ProjectManagerError):
    """A destructive operation targeted the always-present Project Manager tab."""


def is_project_manager_id(instance_id: object) -> bool:
    """Return true only for the one stable protected instance ID."""

    return instance_id == PROJECT_MANAGER_INSTANCE_ID


def is_project_manager_record(record: Mapping[str, Any] | None) -> bool:
    """Identify the protected row by stable ID, never by a user-editable name."""

    return bool(record) and is_project_manager_id(record.get("id"))


def build_project_manager_record(
    *,
    default_model: str,
    now: float | None = None,
) -> dict[str, Any]:
    """Build the dormant canonical row without performing any I/O.

    ``status=idle`` and ``desired_state=stopped`` are deliberate: a tab can be
    durable and visible without keeping local inference, tmux, or a GPU lease
        warm. The model name is metadata for a later explicit chat activation only.
    """

    created_at = time.time() if now is None else float(now)
    return {
        "id": PROJECT_MANAGER_INSTANCE_ID,
        "kind": "terminal",
        "shell_backed": 1,
        "last_agent_kind": "aeon",
        "name": PROJECT_MANAGER_NAME,
        "tmux_name": PROJECT_MANAGER_TMUX_NAME,
        "workspace": PROJECT_MANAGER_WORKSPACE,
        "objective": PROJECT_MANAGER_OBJECTIVE,
        "max_iterations": None,
        "model": default_model,
        "status": "idle",
        "desired_state": "stopped",
        "created_at": created_at,
        "updated_at": created_at,
        "last_started_at": None,
        "last_error": "",
        "created_by": PROJECT_MANAGER_CREATED_BY,
        "launch_origin": PROJECT_MANAGER_LAUNCH_ORIGIN,
    }


def validate_project_manager_record(record: Mapping[str, Any]) -> None:
    """Fail closed if protected identity fields were replaced or corrupted.

    Runtime fields such as kind/mode, status, desired state, model, objective,
    and prompt selections are intentionally not locked: they change during
    normal same-tab terminal/agent use.
    """

    # The stable ID and process target are protected.  ``name`` is deliberately
    # absent: it is a user-facing durable tab label and may be renamed without
    # changing the Project Manager's lifecycle identity.
    expected = {
        "id": PROJECT_MANAGER_INSTANCE_ID,
        "tmux_name": PROJECT_MANAGER_TMUX_NAME,
        "created_by": PROJECT_MANAGER_CREATED_BY,
        "launch_origin": PROJECT_MANAGER_LAUNCH_ORIGIN,
    }
    mismatches = [
        field for field, value in expected.items() if record.get(field) != value
    ]
    if mismatches:
        fields = ", ".join(sorted(mismatches))
        raise ProjectManagerInvariantError(
            f"Project Manager protected identity mismatch: {fields}"
        )


def ensure_project_manager(
    store: Any,
    *,
    default_model: str,
    now: float | None = None,
) -> tuple[dict[str, Any], bool]:
    """Return the durable Project Manager row, creating only its idle record.

    The boolean reports whether this caller created the row.  A competing
    request may win the uniqueness race; in that case the winner is re-read and
    accepted only when all protected identity fields match exactly.
    """

    existing = store.get_instance(PROJECT_MANAGER_INSTANCE_ID)
    if existing is not None:
        validate_project_manager_record(existing)
        return dict(existing), False

    record = build_project_manager_record(default_model=default_model, now=now)
    try:
        store.create_instance(record)
    except Exception as exc:
        raced = store.get_instance(PROJECT_MANAGER_INSTANCE_ID)
        if raced is None:
            raise ProjectManagerError(
                "Could not materialize the Project Manager tab"
            ) from exc
        validate_project_manager_record(raced)
        return dict(raced), False

    created = store.get_instance(PROJECT_MANAGER_INSTANCE_ID)
    if created is None:
        raise ProjectManagerError(
            "Project Manager creation completed without a durable registry row"
        )
    validate_project_manager_record(created)
    return dict(created), True


def project_manager_public_flags(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return UI-safe lifecycle flags for one instance record."""

    pinned = is_project_manager_record(record)
    return {
        "pinned": pinned,
        "always_present": pinned,
        "lazy_start": pinned,
        "role": "project_manager" if pinned else None,
    }


def dormant_project_manager_status(
    record: Mapping[str, Any],
    *,
    pane_exists: bool,
    pane_dead: bool = False,
) -> str | None:
    """Return ``idle`` for a dormant placeholder, otherwise defer to normal logic.

    A Project Manager whose desired state is running but whose pane disappeared
    must still be reported as interrupted by the normal reconciler.  This helper
    only prevents a deliberately dormant, pinned row from being mislabeled as a
    deleted/stopped ordinary session.
    """

    if not is_project_manager_record(record):
        return None
    if record.get("desired_state") != "stopped":
        return None
    if not pane_exists or pane_dead:
        return "idle"
    return None


def is_first_project_manager_activation(record: Mapping[str, Any]) -> bool:
    """Identify the virgin placeholder so its initial objective is not skipped.

    The generic resume endpoint normally substitutes a continuation objective.
    This one row is created dormant, so its first explicit ``resume`` is really
    its initial launch and should receive the canonical Project Manager role.
    """

    return is_project_manager_record(record) and record.get("last_started_at") is None


def reject_project_manager_deletion(instance_id: object) -> None:
    """Raise before any pane kill or database deletion of the pinned row."""

    if is_project_manager_id(instance_id):
        raise ProjectManagerProtectedError(
            "The Project Manager tab is permanent; stop it to release compute instead"
        )
