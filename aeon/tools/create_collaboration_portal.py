"""Managed-agent request for an owner-approved collaboration portal.

This tool never provisions a sibling, credential, or public endpoint. It can
only ask Nexus to pin a bounded proposal for the owner to review in the UI.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from aeon.core.collaborator_mode import (
    CollaboratorModeError,
    load_collaborator_mode_from_environment,
    normalize_collaborator_name,
    normalize_project_brief,
)
from aeon.remote.self_settings import (
    SELF_SETTINGS_TOKEN_FILE_ENV,
    SELF_SETTINGS_URL_ENV,
    SelfSettingsCapabilityError,
    collaboration_portal_endpoint_from_self_settings,
    read_self_settings_token,
    validate_managed_instance_id,
)
from aeon.tools.base import BaseTool


_REQUEST_ID_RE = re.compile(r"^collab-request-[0-9a-f]{32}$")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _open_local(request: urllib.request.Request, *, timeout: float):
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirectHandler(),
    )
    return opener.open(request, timeout=timeout)


class CreateCollaborationPortalTool(BaseTool):
    """Pin a proposal which only the Nexus owner can approve and provision."""

    def __init__(self):
        super().__init__(
            name="create_collaboration_portal",
            description=(
                "Request an owner-reviewed collaboration portal for this managed "
                "agent. Pass a short name and the exact project brief safe to share "
                "with an outside collaborator. This only pins a proposal for the "
                "owner; it does not create a sibling, credential, public URL, or "
                "access. Use only when the owner explicitly asked for this setup."
            ),
        )
        try:
            state = load_collaborator_mode_from_environment()
            validate_managed_instance_id(os.environ.get("AEON_REMOTE_INSTANCE_ID"))
        except (CollaboratorModeError, SelfSettingsCapabilityError):
            state = None
        self.is_internal = not bool(
            state is not None
            and not state.enabled
            and os.environ.get(SELF_SETTINGS_URL_ENV)
            and os.environ.get(SELF_SETTINGS_TOKEN_FILE_ENV)
        )

    def execute(self, name: str, project_brief: str) -> str:
        if self.is_internal:
            return (
                "Error: collaboration portal requests are available only to a "
                "managed owner agent outside collaborator mode."
            )
        try:
            portal_name = normalize_collaborator_name(name)
            brief = normalize_project_brief(project_brief)
            state = load_collaborator_mode_from_environment()
            if state.enabled:
                raise SelfSettingsCapabilityError(
                    "Collaborator siblings cannot request another portal"
                )
            instance_id = validate_managed_instance_id(
                os.environ.get("AEON_REMOTE_INSTANCE_ID")
            )
            endpoint = collaboration_portal_endpoint_from_self_settings(
                os.environ.get(SELF_SETTINGS_URL_ENV, "")
            )
            token = read_self_settings_token(
                os.environ.get(SELF_SETTINGS_TOKEN_FILE_ENV, "")
            )
        except (CollaboratorModeError, SelfSettingsCapabilityError) as exc:
            return f"Error: collaboration portal request is unavailable: {exc}"

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(
                {"name": portal_name, "project_brief": brief},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-Nexus-Agent-Instance": instance_id,
            },
        )
        try:
            with _open_local(request, timeout=15) as response:
                raw = response.read(16_385)
                if len(raw) > 16_384:
                    raise RuntimeError("Nexus returned an oversized response")
                document = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read(8_192)
            try:
                detail = json.loads(raw.decode("utf-8")).get("detail")
            except (UnicodeError, json.JSONDecodeError, AttributeError):
                detail = None
            return (
                "Error: Nexus refused the collaboration portal request: "
                f"{detail or f'HTTP {exc.code}'}"
            )
        except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError) as exc:
            return f"Error: Nexus could not pin the collaboration proposal: {exc}"

        if not isinstance(document, dict) or set(document) != {
            "request_id",
            "status",
            "owner_notice_id",
        }:
            return "Error: Nexus returned an unsafe collaboration proposal response."
        request_id = document.get("request_id")
        if (
            not _REQUEST_ID_RE.fullmatch(str(request_id or ""))
            or document.get("status") != "awaiting_owner_approval"
            or document.get("owner_notice_id") != request_id
        ):
            return "Error: Nexus returned a malformed collaboration proposal response."
        return (
            "Collaboration portal proposal pinned for owner approval "
            f"({request_id}). No sibling, credential, or public access exists yet."
        )


__all__ = ("CreateCollaborationPortalTool",)
