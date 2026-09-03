"""Credential-backed Hugging Face publication tools served by Nexus."""

from __future__ import annotations

import json
import os

from aeon.core.agent_protocol import SideEffect, ToolResult, ToolStatus
from aeon.remote.mcp_capability import MCP_URL_ENV
from aeon.remote.self_settings import SELF_SETTINGS_URL_ENV
from aeon.tools.mcp import NexusGatewayError, _McpTool


_BLOCKED_CODES = frozenset(
    {
        "huggingface_authorization_failed",
        "huggingface_credential_ambiguous",
        "huggingface_credential_not_allowed",
        "huggingface_credential_unavailable",
        "huggingface_delegation_not_supported",
        "huggingface_repository_invalid",
        "huggingface_repository_owner_not_allowed",
        "model_card_required",
        "possible_secret_refused",
        "publication_entry_not_allowed",
        "publication_folder_not_allowed",
        "sensitive_path_refused",
        "workspace_unavailable",
    }
)


class _HuggingFaceGatewayTool(_McpTool):
    def __init__(self, *, name: str, description: str) -> None:
        super().__init__(name=name, description=description)
        self.is_internal = not (
            os.environ.get(MCP_URL_ENV) or os.environ.get(SELF_SETTINGS_URL_ENV)
        )

    def _failure(self, error: Exception) -> ToolResult:
        raw: dict[str, object] = {}
        if isinstance(error, NexusGatewayError):
            raw = dict(error.detail)
            code = str(raw.get("error_code") or "huggingface_capability_failed")
            message = str(raw.get("message") or "Nexus refused the Hugging Face capability.")
            retryable = raw.get("retryable") is True
            status = (
                ToolStatus.BLOCKED
                if error.status_code == 403 or code in _BLOCKED_CODES
                else ToolStatus.FAILED
            )
        else:
            code = "huggingface_gateway_unavailable"
            message = "The Nexus Hugging Face gateway is unavailable."
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
                            document.get("error_code") or "huggingface_receipt_invalid"
                        ),
                        "message": str(
                            document.get("message")
                            or "Nexus returned an invalid Hugging Face receipt."
                        ),
                        "retryable": document.get("retryable") is True,
                    },
                    status_code=409,
                )
            )
        if self.name == "huggingface_publish_model" and document.get("verified") is not True:
            return self._failure(
                NexusGatewayError(
                    {
                        "error_code": "huggingface_publication_unverified",
                        "message": "The publication receipt did not prove the remote file set.",
                        "retryable": False,
                        "outcome_ambiguous": True,
                        "repository": document.get("repository"),
                    },
                    status_code=503,
                )
            )
        changed = document.get("external_mutation") is True
        status = (
            ToolStatus.NO_CHANGE
            if self.policy.side_effect == SideEffect.EXTERNAL_MUTATION and not changed
            else ToolStatus.OK
        )
        repository = str(document.get("repository") or "")
        evidence = []
        for key in ("commit", "manifest_sha256"):
            value = str(document.get(key) or "")
            if value:
                evidence.append(f"{key}={value[:128]}")
        return ToolResult(
            tool_name=self.name,
            status=status,
            changed=changed,
            summary=json.dumps(document, ensure_ascii=False, indent=2)[:12_000],
            evidence=evidence,
            artifacts=[repository] if repository else [],
            error_code="" if status == ToolStatus.OK else "no_change",
            retryable=False,
            side_effect=self.policy.side_effect,
            raw=document,
        )


class HuggingFaceAccountTool(_HuggingFaceGatewayTool):
    def __init__(self) -> None:
        super().__init__(
            name="huggingface_account",
            description=(
                "Verify the one Hugging Face credential assigned to this agent and "
                "list its authenticated writable namespaces. This is read-only and "
                "never returns the token. Use it before choosing owner/repository."
            ),
        )

    def execute(self) -> ToolResult:
        return self._receipt("huggingface-account", method="GET")


class HuggingFacePublishModelTool(_HuggingFaceGatewayTool):
    def __init__(self) -> None:
        super().__init__(
            name="huggingface_publish_model",
            description=(
                "Upload one documented model folder from this agent's exact workspace "
                "to an owner/name Hugging Face model repository through Nexus. This is "
                "an EXTERNAL mutation and requires explicit user authority to upload or "
                "publish on Hugging Face. The folder must include a non-empty top-level "
                "README.md and no links, credentials, or private runtime paths. New repos "
                "are staged privately, verified, and only then made public when public=true. "
                "The token never enters Aeon. Leave authority_operation='upload' "
                "unchanged so the harness can bind the exact "
                "owner-authorized external scope. A successful receipt proves the remote "
                "file set and final visibility."
            ),
        )

    def execute(
        self,
        repository: str,
        folder: str,
        commit_message: str,
        public: bool = True,
        authority_operation: str = "upload",
    ) -> ToolResult:
        if authority_operation.casefold() != "upload":
            return self._failure(
                NexusGatewayError(
                    {
                        "error_code": "huggingface_scope_invalid",
                        "message": "Hugging Face publication scope must remain authority_operation=upload.",
                        "retryable": False,
                    },
                    status_code=403,
                )
            )
        return self._receipt(
            "huggingface-publish",
            payload={
                "repository": repository,
                "folder": folder,
                "commit_message": commit_message,
                "public": public,
            },
            timeout=3600,
        )


class HuggingFaceVerifyPublicationTool(_HuggingFaceGatewayTool):
    def __init__(self) -> None:
        super().__init__(
            name="huggingface_verify_publication",
            description=(
                "Read back one Hugging Face model repository and compare its remote "
                "file inventory with an exact local publication folder. This is "
                "read-only, never returns the token, and is required after an ambiguous "
                "upload outcome before retrying or claiming success."
            ),
        )

    def execute(self, repository: str, folder: str) -> ToolResult:
        return self._receipt(
            "huggingface-verify",
            payload={"repository": repository, "folder": folder},
            timeout=120,
        )
