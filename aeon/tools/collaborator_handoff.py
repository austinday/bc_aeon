"""Only model-callable effect available to a collaborator-mode sibling."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request

from aeon.core.chat_transcript import (
    CHAT_TRANSCRIPT_ENV,
    ChatTranscriptError,
    read_chat_messages,
)
from aeon.core.collaborator_mode import (
    load_collaborator_mode_from_environment,
    normalize_handoff_message,
)
from aeon.remote.self_settings import (
    SELF_SETTINGS_TOKEN_FILE_ENV,
    SELF_SETTINGS_URL_ENV,
    SelfSettingsCapabilityError,
    collaboration_handoff_endpoint_from_self_settings,
    read_self_settings_token,
    validate_managed_instance_id,
)
from aeon.tools.base import BaseTool


_HANDOFF_ID_RE = re.compile(r"^handoff-[0-9a-f]{32}$")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _open_local(request: urllib.request.Request, *, timeout: float):
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirectHandler(),
    )
    return opener.open(request, timeout=timeout)


class SendCollaboratorHandoffTool(BaseTool):
    """Queue relevant collaborator input for the bound working agent."""

    def __init__(self):
        super().__init__(
            name="send_collaborator_handoff",
            description=(
                "Send a concise, faithful summary of material collaborator input "
                "to the working agent. Include concrete requirements, feedback, "
                "advice, promised work, questions, or proposed change orders and "
                "preserve uncertainty. Do not use this for greetings or duplicate "
                "updates. Routing is fixed by Nexus; this tool accepts no target."
            ),
        )
        try:
            state = load_collaborator_mode_from_environment()
            instance_id = validate_managed_instance_id(
                os.environ.get("AEON_REMOTE_INSTANCE_ID")
            )
        except (SelfSettingsCapabilityError, ValueError):
            state = None
            instance_id = ""
        self.is_internal = not bool(
            state is not None
            and state.enabled
            and state.collaborator_instance_id == instance_id
            and os.environ.get(SELF_SETTINGS_URL_ENV)
            and os.environ.get(SELF_SETTINGS_TOKEN_FILE_ENV)
            and os.environ.get(CHAT_TRANSCRIPT_ENV)
        )

    @staticmethod
    def _handoff_id(
        instance_id: str, source_message_id: str, message: str
    ) -> str:
        digest = hashlib.sha256(
            f"{instance_id}\0{source_message_id}\0{message}".encode("utf-8")
        ).hexdigest()[:32]
        return f"handoff-{digest}"

    def execute(self, message: str) -> str:
        if self.is_internal:
            return "Error: collaborator handoff is unavailable outside collaborator mode."
        try:
            normalized = normalize_handoff_message(message)
            state = load_collaborator_mode_from_environment()
            instance_id = validate_managed_instance_id(
                os.environ.get("AEON_REMOTE_INSTANCE_ID")
            )
            if (
                not state.enabled
                or state.collaborator_instance_id != instance_id
            ):
                raise SelfSettingsCapabilityError(
                    "Collaborator launch identity is mismatched"
                )
            endpoint = collaboration_handoff_endpoint_from_self_settings(
                os.environ.get(SELF_SETTINGS_URL_ENV, "")
            )
            token = read_self_settings_token(
                os.environ.get(SELF_SETTINGS_TOKEN_FILE_ENV, "")
            )
            turns = read_chat_messages(os.environ.get(CHAT_TRANSCRIPT_ENV, ""))
            source = next(
                (item for item in reversed(turns) if item.get("role") == "user"),
                None,
            )
            if source is None:
                raise ChatTranscriptError(
                    "No captured external user turn is available"
                )
        except (ChatTranscriptError, SelfSettingsCapabilityError, ValueError) as exc:
            return f"Error: collaborator handoff is unavailable: {exc}"

        handoff_id = self._handoff_id(instance_id, source["id"], normalized)
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(
                {
                    "message": normalized,
                    "handoff_id": handoff_id,
                    # Harness-selected from the owner-private transcript; this
                    # identity is deliberately not a model parameter.
                    "source_message_id": source["id"],
                },
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
            return f"Error: Nexus refused the collaborator handoff: {detail or f'HTTP {exc.code}'}"
        except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError) as exc:
            return f"Error: Nexus could not queue the collaborator handoff: {exc}"

        if not isinstance(document, dict) or set(document) != {
            "id",
            "status",
            "delivered_at",
        }:
            return "Error: Nexus returned a malformed collaborator handoff response."
        response_id = document.get("id")
        status = document.get("status")
        delivered_at = document.get("delivered_at")
        if (
            response_id != handoff_id
            or not _HANDOFF_ID_RE.fullmatch(str(response_id or ""))
            or status not in {"queued", "delivered", "failed"}
            or (status in {"queued", "failed"} and delivered_at is not None)
            or (
                status == "delivered"
                and (
                    isinstance(delivered_at, bool)
                    or not isinstance(delivered_at, (int, float))
                    or delivered_at <= 0
                )
            )
        ):
            return "Error: Nexus returned a malformed collaborator handoff response."
        if status == "delivered":
            return f"Collaborator handoff delivered ({response_id})."
        if status == "failed":
            return (
                "Error: Nexus could not safely confirm the collaborator handoff; "
                f"it will not retry automatically ({response_id})."
            )
        return (
            "Collaborator handoff durably queued for automatic delivery when the "
            f"working agent is ready ({response_id})."
        )


__all__ = ("SendCollaboratorHandoffTool",)
