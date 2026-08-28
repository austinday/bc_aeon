"""Narrow local capability for one managed Aeon to edit its own Job Role.

The bearer value lives in an owner-only per-instance file.  Agent processes get
only that file's path and an allowlisted local endpoint; neither the model nor a
tool call chooses an instance identifier.
"""

from __future__ import annotations

import os
import re
import secrets
import stat
from pathlib import Path
from urllib.parse import urlparse, urlunparse


SELF_JOB_ROLE_PATH = "/internal/agent/job-role"
COLLABORATION_PORTAL_PATH = "/internal/agent/collaboration-portals"
COLLABORATION_HANDOFF_PATH = "/internal/agent/collaboration-handoff"
SELF_SETTINGS_URL_ENV = "NEXUS_INTERNAL_SELF_SETTINGS_URL"
SELF_SETTINGS_TOKEN_FILE_ENV = "NEXUS_SELF_SETTINGS_TOKEN_FILE"
SELF_SETTINGS_TOKEN_FILENAME = "agent-self-settings.token"
MAX_SELF_JOB_ROLE_BYTES = 16 * 1024
MAX_CAPABILITY_BYTES = 256

_ORCHESTRATOR_PATH = "/internal/orchestrator/agents"
_ALLOWED_ENDPOINT_ORIGINS = frozenset(
    {
        "http://127.0.0.1:8765",
        "http://[::1]:8765",
        "http://172.19.0.1:8765",
    }
)
_INSTANCE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CAPABILITY_RE = re.compile(r"^[A-Za-z0-9_-]{32,256}$")


class SelfSettingsCapabilityError(RuntimeError):
    """A self-settings endpoint, identity, or capability is unsafe."""


def validate_managed_instance_id(value: object) -> str:
    if not isinstance(value, str) or not _INSTANCE_ID_RE.fullmatch(value):
        raise SelfSettingsCapabilityError("Managed agent identity is invalid")
    return value


def normalize_job_role(value: object) -> str:
    if not isinstance(value, str):
        raise SelfSettingsCapabilityError("Job Role must be text")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise SelfSettingsCapabilityError(
            "Job Role cannot be blank; use use_default=true to restore the default"
        )
    if "\x00" in normalized:
        raise SelfSettingsCapabilityError("Job Role contains an invalid NUL character")
    if any(
        ord(character) < 32 and character not in {"\n", "\t"}
        for character in normalized
    ):
        raise SelfSettingsCapabilityError("Job Role contains an invalid control character")
    if len(normalized.encode("utf-8")) > MAX_SELF_JOB_ROLE_BYTES:
        raise SelfSettingsCapabilityError(
            f"Job Role must be at most {MAX_SELF_JOB_ROLE_BYTES} UTF-8 bytes"
        )
    return normalized


def _validated_endpoint(value: str, *, expected_path: str) -> str:
    if not isinstance(value, str):
        raise SelfSettingsCapabilityError(
            "Nexus self-settings endpoint is not an approved local URL"
        )
    parsed = urlparse(value)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if (
        not value.isascii()
        or any(ord(character) < 33 for character in value)
        or origin not in _ALLOWED_ENDPOINT_ORIGINS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != expected_path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise SelfSettingsCapabilityError(
            "Nexus self-settings endpoint is not an approved local URL"
        )
    return value


def self_settings_endpoint_from_orchestrator(value: str) -> str:
    """Derive the narrow endpoint from Nexus's reviewed local control origin."""

    approved = _validated_endpoint(value, expected_path=_ORCHESTRATOR_PATH)
    parsed = urlparse(approved)
    return urlunparse(parsed._replace(path=SELF_JOB_ROLE_PATH))


def validate_self_settings_endpoint(value: str) -> str:
    return _validated_endpoint(value, expected_path=SELF_JOB_ROLE_PATH)


def collaboration_portal_endpoint_from_self_settings(value: str) -> str:
    """Derive the self-bound, owner-approval portal-request endpoint."""

    approved = validate_self_settings_endpoint(value)
    parsed = urlparse(approved)
    return urlunparse(parsed._replace(path=COLLABORATION_PORTAL_PATH))


def validate_collaboration_portal_endpoint(value: str) -> str:
    return _validated_endpoint(value, expected_path=COLLABORATION_PORTAL_PATH)


def collaboration_handoff_endpoint_from_self_settings(value: str) -> str:
    """Derive the self-bound handoff endpoint from the issued local URL."""

    approved = validate_self_settings_endpoint(value)
    parsed = urlparse(approved)
    return urlunparse(parsed._replace(path=COLLABORATION_HANDOFF_PATH))


def validate_collaboration_handoff_endpoint(value: str) -> str:
    return _validated_endpoint(value, expected_path=COLLABORATION_HANDOFF_PATH)


def new_self_settings_token() -> str:
    return secrets.token_urlsafe(48)


def read_self_settings_token(path_value: str | Path) -> str:
    """Read one owner-only regular capability without following links."""

    try:
        rendered = os.fspath(path_value)
    except TypeError as exc:
        raise SelfSettingsCapabilityError("Nexus capability path is invalid") from exc
    if not isinstance(rendered, str) or not rendered or "\x00" in rendered:
        raise SelfSettingsCapabilityError("Nexus capability path is invalid")
    absolute = Path(os.path.abspath(os.path.expanduser(rendered)))
    try:
        if absolute.resolve(strict=True) != absolute:
            raise SelfSettingsCapabilityError("Nexus capability path is not direct")
    except OSError as exc:
        raise SelfSettingsCapabilityError("Nexus capability is unavailable") from exc

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd: int | None = None
    descriptor: int | None = None
    try:
        directory_fd = os.open(absolute.parent, directory_flags)
        directory_metadata = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(directory_metadata.st_mode) != 0o700
        ):
            raise SelfSettingsCapabilityError(
                "Nexus capability directory is not owner-safe"
            )
        descriptor = os.open(absolute.name, file_flags, dir_fd=directory_fd)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size < 32
            or metadata.st_size > MAX_CAPABILITY_BYTES
        ):
            raise SelfSettingsCapabilityError("Nexus capability file is not owner-safe")
        raw = os.read(descriptor, MAX_CAPABILITY_BYTES + 1)
        if len(raw) > MAX_CAPABILITY_BYTES:
            raise SelfSettingsCapabilityError("Nexus capability is malformed")
        token = raw.decode("ascii", errors="strict").strip()
    except SelfSettingsCapabilityError:
        raise
    except (OSError, UnicodeError) as exc:
        raise SelfSettingsCapabilityError("Nexus capability is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd is not None:
            os.close(directory_fd)
    if not _CAPABILITY_RE.fullmatch(token):
        raise SelfSettingsCapabilityError("Nexus capability is malformed")
    return token
