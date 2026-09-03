"""Process-local cooperative cleanup registry for interruptible tool work.

Only ephemeral resources owned by this Python process belong here. Durable
Fleet batch jobs are intentionally represented by IDs, not close callbacks, so
process termination never cancels submitted work.
"""

from __future__ import annotations

import threading
import uuid
import weakref
from collections.abc import Callable
from typing import Any


_lock = threading.RLock()
_commands: dict[str, Callable[[], Any]] = {}
_service_owners: weakref.WeakSet[Any] = weakref.WeakSet()


def register_receipted_command(stop_and_prove: Callable[[], Any]) -> str:
    if not callable(stop_and_prove):
        raise TypeError("receipted command cleanup must be callable")
    token = uuid.uuid4().hex
    with _lock:
        _commands[token] = stop_and_prove
    return token


def unregister_receipted_command(token: str | None) -> None:
    if not token:
        return
    with _lock:
        _commands.pop(str(token), None)


def register_service_owner(owner: Any) -> None:
    """Track an ephemeral owner exposing request_stop()/close()."""

    if not callable(getattr(owner, "close", None)):
        raise TypeError("process-local service owner must be closeable")
    with _lock:
        _service_owners.add(owner)


def unregister_service_owner(owner: Any) -> None:
    with _lock:
        _service_owners.discard(owner)


def cancel_process_resources() -> list[str]:
    """Stop exact transient commands and close ephemeral service tickets.

    Returns bounded error labels so a signal boundary can fail closed without
    leaking receipts, unit names, endpoints, or ticket identifiers.
    """

    with _lock:
        commands = list(_commands.items())
        owners = list(_service_owners)
    errors: list[str] = []
    for owner in owners:
        request_stop = getattr(owner, "request_stop", None)
        try:
            if callable(request_stop):
                request_stop()
        except Exception as exc:
            errors.append(f"service_cancel:{type(exc).__name__}")
    for token, cleanup in commands:
        try:
            cleanup()
        except Exception as exc:
            errors.append(f"command_stop:{type(exc).__name__}")
        else:
            unregister_receipted_command(token)
    for owner in owners:
        try:
            owner.close()
        except Exception as exc:
            errors.append(f"service_close:{type(exc).__name__}")
        else:
            unregister_service_owner(owner)
    return errors[:20]


__all__ = (
    "cancel_process_resources",
    "register_receipted_command",
    "register_service_owner",
    "unregister_receipted_command",
    "unregister_service_owner",
)
