"""Credential-scoped MCP tools backed by Nexus's owner-side gateway."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from aeon.remote.mcp_capability import (
    MCP_DELEGATION_ID_ENV,
    MCP_DELEGATION_TOKEN_FILE_ENV,
    MCP_URL_ENV,
    mcp_action_endpoint,
    mcp_endpoint_from_self_settings,
)
from aeon.remote.self_settings import (
    SELF_SETTINGS_TOKEN_FILE_ENV,
    SELF_SETTINGS_URL_ENV,
    SelfSettingsCapabilityError,
    read_self_settings_token,
    validate_managed_instance_id,
)
from aeon.tools.base import BaseTool


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _open_local(request: urllib.request.Request, *, timeout: float):
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
    )
    return opener.open(request, timeout=timeout)


class NexusGatewayError(RuntimeError):
    """A typed, sanitized failure returned by a Nexus owner-side capability."""

    def __init__(self, detail: dict[str, object], *, status_code: int) -> None:
        self.detail = dict(detail)
        self.status_code = status_code
        super().__init__(str(detail.get("message") or f"Nexus returned HTTP {status_code}"))


class _McpTool(BaseTool):
    def _capability(self, action: str) -> tuple[str, str, dict[str, str]]:
        try:
            base = os.environ.get(MCP_URL_ENV, "")
            if not base and os.environ.get(SELF_SETTINGS_URL_ENV):
                base = mcp_endpoint_from_self_settings(
                    os.environ.get(SELF_SETTINGS_URL_ENV, "")
                )
            endpoint = mcp_action_endpoint(base, action)
            delegation_id = os.environ.get(MCP_DELEGATION_ID_ENV, "").strip()
            if delegation_id:
                token = read_self_settings_token(
                    os.environ.get(MCP_DELEGATION_TOKEN_FILE_ENV, "")
                )
                headers = {"X-Nexus-MCP-Delegation": delegation_id}
            else:
                instance_id = validate_managed_instance_id(
                    os.environ.get("AEON_REMOTE_INSTANCE_ID")
                )
                token = read_self_settings_token(
                    os.environ.get(SELF_SETTINGS_TOKEN_FILE_ENV, "")
                )
                headers = {"X-Nexus-Agent-Instance": instance_id}
        except SelfSettingsCapabilityError as exc:
            raise RuntimeError(str(exc)) from exc
        return endpoint, token, headers

    def _request(
        self,
        action: str,
        *,
        method: str = "POST",
        payload: dict[str, object] | None = None,
        timeout: float = 30,
    ) -> dict[str, object]:
        endpoint, token, identity_headers = self._capability(action)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            **identity_headers,
        }
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            endpoint,
            method=method,
            headers=headers,
            data=data,
        )
        try:
            with _open_local(request, timeout=timeout) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024:
                    raise RuntimeError("Nexus returned an oversized MCP response")
                document = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read(8192)
            try:
                detail = json.loads(raw.decode("utf-8")).get("detail")
            except (UnicodeError, json.JSONDecodeError, AttributeError):
                detail = None
            if isinstance(detail, dict):
                raise NexusGatewayError(detail, status_code=exc.code) from None
            raise RuntimeError(detail or f"Nexus returned HTTP {exc.code}") from None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Nexus MCP gateway is unavailable: {exc}") from exc
        if not isinstance(document, dict):
            raise RuntimeError("Nexus returned an invalid MCP response")
        return document


class ConnectMcpAccountTool(_McpTool):
    def __init__(self):
        super().__init__(
            name="connect_mcp_account",
            description=(
                "Connect a reusable account through a reviewed remote MCP server after "
                "the user explicitly asks to connect it. Pass service='gmail' or "
                "service='robinhood-trading' and an "
                "optional account_label that will distinguish this account in Nexus. "
                "The tool returns the provider's official login/consent link; give that link to the "
                "user. After they finish, Nexus stores the OAuth grant privately, adds "
                "the named connection to Setup & settings, and allows it for the main "
                "orchestrator. Robinhood connects only its dedicated Agentic Trading account; "
                "never request a Robinhood password, 2FA code, or token from the user. "
                "Only the main Project Manager can use this tool."
            ),
        )
        self.is_internal = not (
            os.environ.get(MCP_URL_ENV) or os.environ.get(SELF_SETTINGS_URL_ENV)
        )

    def execute(self, service: str, account_label: str = "") -> str:
        if os.environ.get("AEON_MAIN_ORCHESTRATOR") != "1":
            return "Error: only the main Project Manager may create shared MCP connections."
        try:
            document = self._request(
                "connect",
                payload={"service": service, "label": account_label},
            )
        except RuntimeError as exc:
            return f"Error: MCP connection could not start: {exc}"
        url = document.get("authorization_url")
        server = document.get("server")
        if (
            not isinstance(url, str)
            or not url.startswith("https://")
            or not isinstance(server, dict)
        ):
            return "Error: Nexus returned an invalid MCP authorization response."
        return (
            f"Open this one-time {server.get('label', 'MCP')} authorization link and "
            f"finish login: {url}\nAfter the browser returns to Nexus, the named "
            "credential will be available in agent settings."
        )


class ListMcpCredentialsTool(_McpTool):
    def __init__(self):
        super().__init__(
            name="list_mcp_credentials",
            description=(
                "List only the reusable MCP account connections currently allowed for "
                "this agent or bounded sub-agent; results contain IDs, labels, accounts, "
                "and server metadata but never OAuth tokens."
            ),
        )
        self.is_internal = not (
            os.environ.get(MCP_URL_ENV) or os.environ.get(SELF_SETTINGS_URL_ENV)
        )

    def execute(self) -> str:
        try:
            return json.dumps(
                self._request("credentials", method="GET"),
                ensure_ascii=False,
                indent=2,
            )
        except RuntimeError as exc:
            return f"Error: MCP credentials could not be listed: {exc}"


class ListProviderCredentialsTool(_McpTool):
    def __init__(self):
        super().__init__(
            name="list_provider_credentials",
            description=(
                "List first-class site/provider credentials explicitly allowed for "
                "this agent, including Hugging Face metadata, without returning any "
                "secret. These are separate from remote MCP accounts: a Hugging Face "
                "credential will never appear in list_mcp_credentials. A listing proves "
                "only that Nexus stores and grants the credential; it does not imply "
                "that an upload/publication tool or external-action authority exists."
            ),
        )
        self.is_internal = not (
            os.environ.get(MCP_URL_ENV) or os.environ.get(SELF_SETTINGS_URL_ENV)
        )

    def execute(self) -> str:
        try:
            return json.dumps(
                self._request("provider-credentials", method="GET"),
                ensure_ascii=False,
                indent=2,
            )
        except RuntimeError as exc:
            return f"Error: provider credentials could not be listed: {exc}"


class ListPaymentAddressesTool(_McpTool):
    def __init__(self):
        super().__init__(
            name="list_payment_addresses",
            description=(
                "List the public payment addresses explicitly allowed for this agent. "
                "Use one when the user asks for a donation or payment line, including "
                "on a model or Hugging Face page. Results never include private keys, "
                "seed phrases, or signing authority."
            ),
        )
        self.is_internal = not (
            os.environ.get(MCP_URL_ENV) or os.environ.get(SELF_SETTINGS_URL_ENV)
        )

    def execute(self) -> str:
        try:
            return json.dumps(
                self._request("payment-addresses", method="GET"),
                ensure_ascii=False,
                indent=2,
            )
        except RuntimeError as exc:
            return f"Error: payment addresses could not be listed: {exc}"


class ListMcpToolsTool(_McpTool):
    def __init__(self):
        super().__init__(
            name="list_mcp_tools",
            description=(
                "Ask one allowed MCP account connection for its current tool schemas; "
                "pass the exact credential_id returned by list_mcp_credentials."
            ),
        )
        self.is_internal = not (
            os.environ.get(MCP_URL_ENV) or os.environ.get(SELF_SETTINGS_URL_ENV)
        )

    def execute(self, credential_id: str) -> str:
        try:
            return json.dumps(
                self._request("tools", payload={"credential_id": credential_id}),
                ensure_ascii=False,
                indent=2,
            )
        except RuntimeError as exc:
            return f"Error: MCP tools could not be listed: {exc}"


class CallMcpTool(_McpTool):
    def __init__(self):
        super().__init__(
            name="call_mcp_tool",
            description=(
                "Call one tool on an allowed MCP account connection through Nexus. "
                "First use list_mcp_tools, then pass credential_id, the exact tool_name, "
                "and its arguments object. This can read or change an external account, "
                "so Aeon's external-action approval policy applies and the OAuth token "
                "never enters the agent process."
            ),
        )
        self.is_internal = not (
            os.environ.get(MCP_URL_ENV) or os.environ.get(SELF_SETTINGS_URL_ENV)
        )

    def execute(
        self,
        credential_id: str,
        tool_name: str,
        arguments: dict[str, object] | None = None,
        source_files: list[str] | None = None,
    ) -> str:
        # source_files is harness-only provenance. The request contract verifies
        # that every owner-named artifact was read and declared; file paths are
        # never forwarded to an external provider as invented tool arguments.
        del source_files
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return "Error: MCP tool arguments must be a JSON object."
        try:
            return json.dumps(
                self._request(
                    "call",
                    payload={
                        "credential_id": credential_id,
                        "tool_name": tool_name,
                        "arguments": arguments,
                    },
                    timeout=60,
                ),
                ensure_ascii=False,
                indent=2,
            )
        except RuntimeError as exc:
            return f"Error: MCP tool call failed: {exc}"
