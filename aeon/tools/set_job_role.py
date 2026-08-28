"""Managed-agent tool for changing only the caller's Nexus Job Role."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from aeon.remote.self_settings import (
    SELF_SETTINGS_TOKEN_FILE_ENV,
    SELF_SETTINGS_URL_ENV,
    SelfSettingsCapabilityError,
    normalize_job_role,
    read_self_settings_token,
    validate_managed_instance_id,
    validate_self_settings_endpoint,
)
from aeon.tools.base import BaseTool


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _open_local(request: urllib.request.Request, *, timeout: float):
    """Reach Nexus directly; never proxy or redirect a bearer capability."""

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirectHandler(),
    )
    return opener.open(request, timeout=timeout)


class SetJobRoleTool(BaseTool):
    """Update the private Job Role layer for this managed Nexus instance."""

    def __init__(self):
        super().__init__(
            name="set_job_role",
            description=(
                "Change your own Job Role for this Nexus session only. Use this "
                "only when the user explicitly asks you to change how you should "
                "operate. Pass job_role with the new role, or pass use_default=true "
                "with no job_role to restore the shared default. The tool cannot "
                "change another agent or the shared default; an update is loaded on "
                "your next model turn."
            ),
        )
        self.is_internal = not all(
            (
                os.environ.get("AEON_REMOTE_INSTANCE_ID"),
                os.environ.get(SELF_SETTINGS_URL_ENV),
                os.environ.get(SELF_SETTINGS_TOKEN_FILE_ENV),
            )
        )

    def execute(self, job_role: str = "", use_default: bool = False) -> str:
        if self.is_internal:
            return "Error: Job Role self-service is available only in a managed Nexus session."
        if not isinstance(use_default, bool):
            return "Error: use_default must be true or false."
        if use_default and str(job_role or "").strip():
            return "Error: omit job_role when use_default=true."
        try:
            normalized_role = "" if use_default else normalize_job_role(job_role)
            instance_id = validate_managed_instance_id(
                os.environ.get("AEON_REMOTE_INSTANCE_ID")
            )
            endpoint = validate_self_settings_endpoint(
                os.environ.get(SELF_SETTINGS_URL_ENV, "")
            )
            token = read_self_settings_token(
                os.environ.get(SELF_SETTINGS_TOKEN_FILE_ENV, "")
            )
        except SelfSettingsCapabilityError as exc:
            return f"Error: Job Role self-service is unavailable: {exc}"

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(
                {"job_role": normalized_role, "use_default": use_default}
            ).encode("utf-8"),
            method="PUT",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-Nexus-Agent-Instance": instance_id,
            },
        )
        try:
            with _open_local(request, timeout=15) as response:
                raw = response.read(16385)
                if len(raw) > 16384:
                    raise RuntimeError("Nexus returned an oversized response")
                document = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read(8192)
            try:
                detail = json.loads(raw.decode("utf-8")).get("detail")
            except (UnicodeError, json.JSONDecodeError, AttributeError):
                detail = None
            return f"Error: Nexus refused the Job Role update: {detail or f'HTTP {exc.code}'}"
        except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError) as exc:
            return f"Error: Nexus could not update the Job Role: {exc}"

        if not isinstance(document, dict) or document.get("scope") != "session":
            return "Error: Nexus returned a malformed Job Role response."
        revision = document.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            return "Error: Nexus returned a malformed Job Role revision."
        if not isinstance(document.get("changed"), bool) or document.get(
            "apply_state"
        ) not in {"live", "pending"}:
            return "Error: Nexus returned a malformed Job Role state."
        role_state = "the shared default" if use_default else "a session override"
        change_state = "updated" if document.get("changed") else "already set"
        if document.get("apply_state") == "live":
            timing = "It will be active on your next model turn."
        else:
            timing = "It is saved and will become active after Nexus refreshes or restarts this session."
        return (
            f"Job Role {change_state} for this session as {role_state} "
            f"(revision {revision}). {timing}"
        )
