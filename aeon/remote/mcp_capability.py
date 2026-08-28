"""Validation helpers for Nexus's local credential-scoped MCP gateway."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

from aeon.remote.self_settings import (
    SELF_JOB_ROLE_PATH,
    SelfSettingsCapabilityError,
    validate_self_settings_endpoint,
)


MCP_URL_ENV = "NEXUS_INTERNAL_MCP_URL"
MCP_DELEGATION_ID_ENV = "NEXUS_MCP_DELEGATION_ID"
MCP_DELEGATION_TOKEN_FILE_ENV = "NEXUS_MCP_DELEGATION_TOKEN_FILE"
MCP_BASE_PATH = "/internal/agent/mcp"
MCP_ACTIONS = frozenset(
    {
        "connect",
        "credentials",
        "provider-credentials",
        "payment-addresses",
        "tools",
        "call",
        "delegations",
        "github-repositories",
        "github-status",
        "github-commit",
        "github-push",
        "github-verify",
    }
)


def mcp_endpoint_from_self_settings(value: str) -> str:
    approved = validate_self_settings_endpoint(value)
    parsed = urlparse(approved)
    if parsed.path != SELF_JOB_ROLE_PATH:
        raise SelfSettingsCapabilityError("Nexus MCP endpoint is not approved")
    return urlunparse(parsed._replace(path=MCP_BASE_PATH))


def validate_mcp_base_endpoint(value: str) -> str:
    if not isinstance(value, str) or not value.isascii():
        raise SelfSettingsCapabilityError("Nexus MCP endpoint is not approved")
    parsed = urlparse(value)
    self_url = urlunparse(parsed._replace(path=SELF_JOB_ROLE_PATH))
    approved_self = validate_self_settings_endpoint(self_url)
    approved = urlparse(approved_self)
    if (
        parsed.scheme != approved.scheme
        or parsed.netloc != approved.netloc
        or parsed.path != MCP_BASE_PATH
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise SelfSettingsCapabilityError("Nexus MCP endpoint is not approved")
    return value


def mcp_action_endpoint(base: str, action: str) -> str:
    approved = validate_mcp_base_endpoint(base)
    value = str(action or "").strip()
    if value not in MCP_ACTIONS:
        raise SelfSettingsCapabilityError("Nexus MCP action is not approved")
    return f"{approved}/{value}"


__all__ = (
    "MCP_ACTIONS",
    "MCP_BASE_PATH",
    "MCP_DELEGATION_ID_ENV",
    "MCP_DELEGATION_TOKEN_FILE_ENV",
    "MCP_URL_ENV",
    "mcp_action_endpoint",
    "mcp_endpoint_from_self_settings",
    "validate_mcp_base_endpoint",
)
