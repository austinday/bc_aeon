"""Capability-gated interface to the closed benchmark behavioral simulator."""

from __future__ import annotations

from aeon.core.agent_protocol import SideEffect, ToolPolicy
from aeon.core.benchmark_simulator import (
    FLEET_OPERATIONS,
    PARALLEL_OPERATIONS,
    SCENARIO_TOOL_NAME,
    ScenarioInfrastructureError,
    ScenarioSession,
    load_scenario_capability,
)
from aeon.tools.base import BaseTool


class BenchmarkWorkflowTool(BaseTool):
    """Synthetic only: it has no Fleet, subprocess, network, or agent backend."""

    def __init__(self) -> None:
        capability = load_scenario_capability()
        policy = ToolPolicy(
            name=SCENARIO_TOOL_NAME,
            side_effect=SideEffect.CONTROL,
            observation_boundary=True,
            idempotent=False,
            approval_required=False,
            self_verifying=True,
            retry_limit=0,
        )
        super().__init__(
            name=SCENARIO_TOOL_NAME,
            description=(
                "Operate one closed deterministic workflow described by the current "
                "benchmark task. Use exact opaque IDs returned by prior observations. "
                "Choose operations from the current state; rejected, duplicate, idle, "
                "or unnecessary operations remain observable. This tool never contacts "
                "real Fleet Compute and never launches a real process or sub-agent."
            ),
            directives=[],
            policy=policy,
        )
        self._session: ScenarioSession | None = None
        self._operations: tuple[str, ...] = ()
        self.is_internal = True
        if capability is None:
            return
        try:
            self._session = ScenarioSession(capability)
        except ScenarioInfrastructureError:
            return
        self._operations = (
            FLEET_OPERATIONS
            if capability.scenario == "fleet"
            else PARALLEL_OPERATIONS
        )
        self.is_internal = False

    def parameter_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": list(self._operations)},
                "reference_ids": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string", "maxLength": 128},
                },
                "branch": {
                    "type": "string",
                    "enum": ["", "a", "b", "c", "extra"],
                },
            },
            "required": ["operation"],
            "additionalProperties": False,
        }

    def execute(
        self,
        operation: str,
        reference_ids: list[str] | None = None,
        branch: str = "",
    ) -> str:
        if self._session is None or operation not in self._operations:
            raise RuntimeError("benchmark workflow is unavailable")
        return self._session.execute(operation, reference_ids, branch)


__all__ = ("BenchmarkWorkflowTool",)
