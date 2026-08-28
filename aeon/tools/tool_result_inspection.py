"""Bounded model access to harness-archived oversized tool evidence."""

from __future__ import annotations

import json

from aeon.core.agent_protocol import SideEffect, ToolResult, ToolStatus
from aeon.core.tool_result_archive import ToolResultArchiveError

from .base import BaseTool


class InspectToolResult(BaseTool):
    """Search or page through one opaque request-scoped tool-result reference."""

    def __init__(self, worker):
        super().__init__(
            name="inspect_tool_result",
            description=(
                "Inspect a full oversized tool result that Aeon archived outside model "
                "context. Pass the opaque reference from a tool receipt. Prefer a "
                "literal query to retrieve focused matches; with no query, read one "
                "bounded character page using offset and limit. This tool accepts no "
                "filesystem paths and cannot access another request's results."
            ),
        )
        self.worker = worker

    def execute(
        self,
        reference: str,
        query: str = "",
        offset: int = 0,
        limit: int = 2_000,
    ) -> ToolResult:
        try:
            result = self.worker.inspect_tool_result(
                reference=reference,
                query=query,
                offset=offset,
                limit=limit,
            )
        except ToolResultArchiveError as exc:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILED,
                changed=False,
                summary=f"Archived tool-result inspection failed: {exc}",
                error_code="tool_result_archive_unavailable",
                retryable=False,
                side_effect=SideEffect.READ_ONLY,
            )
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.OK,
            changed=False,
            summary=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            evidence=[
                f"sha256={result.get('sha256', '')}; mode={result.get('mode', '')}"
            ],
            side_effect=SideEffect.READ_ONLY,
        )
