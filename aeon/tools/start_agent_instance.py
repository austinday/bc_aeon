"""Project-Manager-only bridge to Nexus's durable standalone agent lifecycle."""

from __future__ import annotations

import json
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from aeon.core.durable_agent_guard import (
    VerifiedNexusAgentStart,
    verified_start_receipt,
)
from aeon.remote.project_manager import PROJECT_MANAGER_INSTANCE_ID
from aeon.tools.base import BaseTool


_ENDPOINT_PATH = "/internal/orchestrator/agents"
_ALLOWED_ENDPOINT_ORIGINS = frozenset(
    {
        "http://127.0.0.1:8765",
        "http://[::1]:8765",
        "http://172.19.0.1:8765",
    }
)
_CREDENTIAL_ID_ALIASES = {"github.token": "github"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "Nexus redirect refused", headers, fp
        )


def _local_urlopen(request: urllib.request.Request, *, timeout: int):
    """Open an exact local request with neither proxying nor redirects."""

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
    )
    return opener.open(request, timeout=timeout)


class StartAgentInstanceTool(BaseTool):
    """Create a durable Nexus-managed agent without becoming a nested sub-agent."""

    def __init__(self):
        # The released fast-service evidence binds the historical tool-prompt file.
        # Do not inject that now-stale, eager-objective text at runtime. The current
        # contract lives in this schema/description, the main-orchestrator role, and
        # DurableAgentTurnGuard's fail-closed action and receipt checks.
        super().__init__(
            name="start_agent_instance",
            description=(
                "After the user explicitly directs creation now, register a standalone "
                "Nexus-managed Aeon, Codex, Claude, or Grok agent in an existing "
                "target directory. A can/could/would/how question requests a plan, "
                "not this tool. This capability is available only to the primary "
                "Project Manager. Parameters: name, directory, kind (default aeon), "
                "goal, personality, system_prompt, and continuous_mode. The new agent "
                "may receive allowed_credentials as exact IDs from "
                "list_mcp_credentials; omitted means no credential access. The new agent "
                "is registered as its own manageable Nexus tab. A non-continuous Aeon "
                "tab stays idle until the user sends that tab its first message. Set "
                "continuous_mode=true only when the user requested autonomous ongoing "
                "work and provide their goal verbatim enough to preserve its scope; the "
                "goal must contain at least three words. Personality and system_prompt "
                "become that tab's private persistent instruction layer."
            ),
            directives=[],
        )
        self.is_internal = (
            os.environ.get("AEON_REMOTE_INSTANCE_ID")
            != PROJECT_MANAGER_INSTANCE_ID
        )

    @staticmethod
    def _token(path_value: str) -> str:
        path = Path(path_value)
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
                or metadata.st_size > 256
            ):
                raise RuntimeError("Nexus capability file is not owner-safe")
            token = os.read(descriptor, 257).decode("ascii", errors="strict").strip()
        finally:
            os.close(descriptor)
        if len(token) < 32 or len(token) > 256:
            raise RuntimeError("Nexus capability is malformed")
        return token

    @staticmethod
    def _endpoint(value: str) -> str:
        parsed = urlparse(value)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if (
            not value.isascii()
            or any(ord(character) < 33 for character in value)
            or origin not in _ALLOWED_ENDPOINT_ORIGINS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != _ENDPOINT_PATH
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError("Nexus orchestrator endpoint is not an approved local URL")
        return value

    def execute(
        self,
        name: str,
        directory: str,
        kind: str = "aeon",
        goal: str = "",
        personality: str = "",
        system_prompt: str = "",
        continuous_mode: bool = False,
        allowed_credentials: list[str] | None = None,
    ) -> str | VerifiedNexusAgentStart:
        if self.is_internal:
            return "Error: only the primary Nexus Project Manager may start standalone agents."
        try:
            endpoint = self._endpoint(
                os.environ.get("NEXUS_INTERNAL_ORCHESTRATOR_URL", "")
            )
            token = self._token(
                os.environ.get("NEXUS_ORCHESTRATOR_TOKEN_FILE", "")
            )
        except (OSError, RuntimeError, UnicodeError) as exc:
            return f"Error: standalone agent control is unavailable: {exc}"

        payload = {
            "name": name,
            "workspace": directory,
            "kind": kind,
            "goal": goal,
            "personality": personality,
            "system_prompt": system_prompt,
            "continuous_mode": continuous_mode,
        }
        if allowed_credentials:
            # Older prompts and persisted plans may refer to GitHub by the
            # owner-private backing filename. Nexus's public capability ID is
            # `github`; normalize only this exact historical alias and preserve
            # every other ID for Nexus to validate fail closed.
            payload["allowed_credentials"] = list(dict.fromkeys(
                _CREDENTIAL_ID_ALIASES.get(credential_id, credential_id)
                for credential_id in allowed_credentials
            ))
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "X-Nexus-Orchestrator-Instance": PROJECT_MANAGER_INSTANCE_ID,
            },
        )
        try:
            with _local_urlopen(request, timeout=20) as response:
                raw = response.read(65537)
                if len(raw) > 65536:
                    raise RuntimeError("Nexus returned an oversized response")
                document = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read(8192)
            try:
                detail = json.loads(raw.decode("utf-8")).get("detail")
            except (UnicodeError, json.JSONDecodeError, AttributeError):
                detail = None
            return f"Error: Nexus refused the agent start: {detail or f'HTTP {exc.code}'}"
        except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError) as exc:
            return f"Error: Nexus could not start the agent: {exc}"

        instance = document.get("instance") if isinstance(document, dict) else None
        try:
            # Return typed evidence, not success-looking prose. The worker accepts
            # only this receipt as proof that the authenticated Nexus bridge wrote
            # and returned a durable instance record in the current user turn.
            return verified_start_receipt(
                instance,
                expected_name=name,
                expected_workspace=directory,
                expected_kind=kind,
                expected_continuous=continuous_mode,
                expected_goal=goal,
            )
        except (OSError, ValueError) as exc:
            return f"Error: {exc}."
