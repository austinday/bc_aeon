"""Scoped GitHub tools backed exclusively by Nexus's owner-side gateway."""

from __future__ import annotations

import json
import os
from typing import Any

from aeon.core.agent_protocol import SideEffect, ToolResult, ToolStatus
from aeon.remote.mcp_capability import MCP_URL_ENV
from aeon.remote.self_settings import SELF_SETTINGS_URL_ENV
from aeon.tools.mcp import NexusGatewayError, _McpTool


_BLOCKED_CODES = frozenset(
    {
        "github_credential_ambiguous",
        "github_credential_not_allowed",
        "github_delegation_not_supported",
        "github_repository_owner_not_allowed",
        "github_repository_policy_unknown",
        "github_repository_visibility_not_allowed",
        "repository_metadata_not_allowed",
        "repository_not_allowed",
        "repository_config_not_allowed",
        "remote_not_allowed",
        "sensitive_path_refused",
        "possible_secret_refused",
        "workspace_unavailable",
    }
)


class _GitHubTool(_McpTool):
    def __init__(self, *, name: str, description: str) -> None:
        super().__init__(name=name, description=description)
        self.is_internal = not (
            os.environ.get(MCP_URL_ENV) or os.environ.get(SELF_SETTINGS_URL_ENV)
        )

    @staticmethod
    def _summary(document: dict[str, object]) -> str:
        encoded = json.dumps(document, ensure_ascii=False, indent=2)
        if len(encoded) <= 12_000:
            return encoded
        return encoded[:11_900] + "\n...[sanitized receipt truncated]"

    def _failure(self, error: Exception) -> ToolResult:
        raw: dict[str, object] = {}
        if isinstance(error, NexusGatewayError):
            detail = error.detail
            raw = dict(detail)
            code = str(detail.get("error_code") or "github_capability_failed")
            message = str(detail.get("message") or "Nexus refused the GitHub capability.")
            retryable = detail.get("retryable") is True
            status = (
                ToolStatus.BLOCKED
                if error.status_code == 403 or code in _BLOCKED_CODES
                else ToolStatus.FAILED
            )
        else:
            code = "github_gateway_unavailable"
            message = "The Nexus GitHub gateway is unavailable."
            retryable = True
            status = ToolStatus.FAILED
        return ToolResult(
            tool_name=self.name,
            status=status,
            changed=False,
            summary=message[:1600],
            error_code=code,
            retryable=retryable,
            side_effect=self.policy.side_effect,
            raw=raw,
        )

    def _receipt(
        self,
        action: str,
        *,
        method: str = "POST",
        payload: dict[str, object] | None = None,
        timeout: float = 30,
    ) -> ToolResult:
        try:
            document = self._request(
                action, method=method, payload=payload, timeout=timeout
            )
        except (NexusGatewayError, RuntimeError) as exc:
            return self._failure(exc)
        if document.get("status") != "ok":
            return self._failure(
                NexusGatewayError(
                    {
                        "error_code": str(
                            document.get("error_code") or "github_receipt_invalid"
                        ),
                        "message": str(
                            document.get("message")
                            or "Nexus returned an invalid GitHub receipt."
                        ),
                        "retryable": document.get("retryable") is True,
                    },
                    status_code=409,
                )
            )
        if self.name == "github_push" and (
            document.get("verified") is not True
            or document.get("head") != document.get("remote_head")
        ):
            # Defense in depth: even if an older or compromised Nexus endpoint
            # returns HTTP 200 after an unverified push, never reinterpret that
            # external outcome as a successful NO_CHANGE receipt.
            return self._failure(
                NexusGatewayError(
                    {
                        "status": "error",
                        "operation": "push",
                        "error_code": "remote_outcome_ambiguous",
                        "message": (
                            "The push did not include exact remote-HEAD verification; "
                            "run github_verify_remote before concluding the operation."
                        ),
                        "retryable": False,
                        "outcome_ambiguous": True,
                        "verification_required": True,
                        "repository": document.get("repository"),
                        "remote": document.get("remote"),
                        "head": document.get("head"),
                        "remote_head": document.get("remote_head"),
                    },
                    status_code=503,
                )
            )
        changed = bool(
            document.get("local_mutation") is True
            or document.get("external_mutation") is True
        )
        mutating = self.policy.side_effect in {
            SideEffect.LOCAL_MUTATION,
            SideEffect.EXTERNAL_MUTATION,
        }
        status = ToolStatus.NO_CHANGE if mutating and not changed else ToolStatus.OK
        evidence: list[str] = []
        for key in ("head", "remote_head"):
            value = document.get(key)
            if isinstance(value, str) and value:
                evidence.append(f"{key}={value[:64]}")
        repository = document.get("repository")
        artifacts: list[str] = []
        if isinstance(repository, str):
            artifacts.append(repository)
        elif isinstance(repository, dict) and isinstance(repository.get("path"), str):
            artifacts.append(str(repository["path"]))
        return ToolResult(
            tool_name=self.name,
            status=status,
            changed=changed,
            summary=self._summary(document),
            evidence=evidence,
            artifacts=artifacts,
            error_code="" if status == ToolStatus.OK else "no_change",
            retryable=False,
            side_effect=self.policy.side_effect,
            raw=document,
        )


class GitHubRepositoriesTool(_GitHubTool):
    def __init__(self) -> None:
        super().__init__(
            name="github_repositories",
            description=(
                "List bounded Git repositories inside this managed agent's exact "
                "Nexus workspace. Returns absolute repository roots, branch/head, "
                "dirty state, and sanitized GitHub remote identity. Use a returned "
                "root verbatim with the other GitHub tools. Requires exactly one "
                "GitHub credential allowed for this agent; never returns its token."
            ),
        )

    def execute(self) -> ToolResult:
        return self._receipt("github-repositories", method="GET")


class GitHubStatusTool(_GitHubTool):
    def __init__(self) -> None:
        super().__init__(
            name="github_status",
            description=(
                "Inspect one exact repository root returned by github_repositories. "
                "Returns branch/head, porcelain change paths, and staged/unstaged "
                "diff summaries without returning file contents or credentials. "
                "This operation is read-only."
            ),
        )

    def execute(self, repository: str) -> ToolResult:
        return self._receipt(
            "github-status", payload={"repository": repository}
        )


class GitHubCommitTool(_GitHubTool):
    def __init__(self) -> None:
        super().__init__(
            name="github_commit",
            description=(
                "Create one local Git commit from 1-128 exact repository-relative "
                "paths in a repository returned by github_repositories. The Nexus "
                "gateway uses an isolated index, preserves unrelated staged work, "
                "and refuses credential-like paths/content. This changes only the "
                "local repository; it does NOT push or change GitHub. Inspect "
                "github_status first."
            ),
        )

    def execute(
        self, repository: str, message: str, paths: list[str]
    ) -> ToolResult:
        return self._receipt(
            "github-commit",
            payload={
                "repository": repository,
                "message": message,
                "paths": paths,
            },
            timeout=60,
        )


class GitHubPushTool(_GitHubTool):
    def __init__(self) -> None:
        super().__init__(
            name="github_push",
            description=(
                "Push the current named local branch to the same branch on one "
                "reviewed github.com remote, then compare the exact remote commit. "
                "This is a separate EXTERNAL action and requires explicit user "
                "authority. Nexus never force-pushes, never rewrites the remote, "
                "never runs repository hooks with the credential, and never gives "
                "the token to Aeon. remote_name defaults to origin."
            ),
        )

    def execute(self, repository: str, remote_name: str = "origin") -> ToolResult:
        return self._receipt(
            "github-push",
            payload={"repository": repository, "remote_name": remote_name},
            timeout=180,
        )


class GitHubVerifyRemoteTool(_GitHubTool):
    def __init__(self) -> None:
        super().__init__(
            name="github_verify_remote",
            description=(
                "Read the exact branch commit from a reviewed github.com remote and "
                "compare it with local HEAD. This is read-only and is the required "
                "follow-up after an ambiguous push receipt. remote_name defaults to origin."
            ),
        )

    def execute(self, repository: str, remote_name: str = "origin") -> ToolResult:
        return self._receipt(
            "github-verify",
            payload={"repository": repository, "remote_name": remote_name},
            timeout=90,
        )


__all__ = (
    "GitHubCommitTool",
    "GitHubPushTool",
    "GitHubRepositoriesTool",
    "GitHubStatusTool",
    "GitHubVerifyRemoteTool",
)
