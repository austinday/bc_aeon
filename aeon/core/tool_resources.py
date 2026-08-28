"""Explicit compute-route contract for every Aeon tool.

Tool side-effect policy answers *whether* a request authorizes an action. This
module answers the separate question of *where any compute comes from*. Keeping
the inventory explicit makes a newly added tool fail closed until its execution
route has been reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ToolResourceError(RuntimeError):
    """A tool has no reviewed compute route."""


class ToolComputeRoute(str, Enum):
    LOCAL_CPU = "local_cpu"
    HOST_SERVICE = "host_service"
    ACTIVE_MODEL = "active_model"
    FLEET_SERVICE = "fleet_service"
    FLEET_CHILD = "fleet_child"
    DYNAMIC_COMMAND = "dynamic_command"
    NEXUS_LIFECYCLE = "nexus_lifecycle"
    EXTERNAL_PROVIDER = "external_provider"


@dataclass(frozen=True)
class ToolResourcePolicy:
    route: ToolComputeRoute
    requires_primary_compute_guard: bool = False
    fleet_service: str | None = None
    host_service: str | None = None


_LOCAL_CPU_TOOLS = frozenset(
    {
        "activate_skill",
        "blackboard_post",
        "blackboard_read",
        "close_file",
        "collapse_skills_category",
        "collapse_tool_category",
        "composite_image",
        "create_skill",
        "deactivate_skill",
        "delete_skill",
        "delete_skill_knowledge",
        "expand_skills_category",
        "expand_tool_category",
        "forget",
        "gather_sub_agents",
        "get_sub_agent_status",
        "get_user_input",
        "job_output",
        "inspect_tool_result",
        "integrate_sub_agent_changes",
        "kill_job",
        "kill_sub_agent",
        "list_memories",
        "memorize",
        "open_file",
        "read_skill",
        "list_skill_knowledge",
        "read_skill_knowledge",
        "remember_skill_knowledge",
        "restart_aeon",
        "resume_previous_session",
        "revert_aeon",
        "run_self_benchmark",
        "say_to_user",
        "search_skill_knowledge",
        "steer_sub_agent",
        "str_replace",
        "task_complete",
        "write_file",
    }
)

_ACTIVE_MODEL_TOOLS = frozenset(
    {
        "analyze_image",
        "get_sub_agent_report",
        "think",
    }
)

_HOST_SERVICE_TOOLS = {
    "browser_capture_media": ("aeon-browser", False),
    "browser_close_tab": ("aeon-browser", False),
    "browser_extract": ("aeon-browser", False),
    "browser_find": ("aeon-browser", False),
    "browser_interact": ("aeon-browser", False),
    "browser_navigate": ("aeon-browser", False),
    "browser_read": ("aeon-browser", False),
    "browser_switch_tab": ("aeon-browser", False),
    # Search summarization calls the primary model after querying this local
    # CPU-only dependency, so it needs both classifications.
    "search_web": ("aeon-searxng", True),
}

_FLEET_SERVICE_TOOLS = {
    "edit_image": "aeon-comfyui",
    "generate_image": "aeon-comfyui",
    "generate_video": "aeon-video-comfyui",
}

_FLEET_CHILD_TOOLS = frozenset({"spawn_sub_agent", "verify_self_modification"})
_DYNAMIC_COMMAND_TOOLS = frozenset({"run_command", "run_command_async"})
_NEXUS_LIFECYCLE_TOOLS = frozenset(
    {
        "set_job_role",
        "start_agent_instance",
        "create_collaboration_portal",
        "send_collaborator_handoff",
    }
)
_EXTERNAL_PROVIDER_TOOLS = frozenset(
    {
        "call_mcp_tool",
        "connect_mcp_account",
        "consult_external_expert",
        "github_commit",
        "github_push",
        "github_repositories",
        "github_status",
        "github_verify_remote",
        "huggingface_model_info",
        "huggingface_model_search",
        "huggingface_repo_file",
        "list_mcp_credentials",
        "list_payment_addresses",
        "list_mcp_tools",
        "list_provider_credentials",
    }
)


def tool_resource_policy(name: str) -> ToolResourcePolicy:
    """Return one reviewed route, refusing unclassified tool names."""

    value = str(name or "").strip()
    if value in _LOCAL_CPU_TOOLS:
        return ToolResourcePolicy(ToolComputeRoute.LOCAL_CPU)
    if value in _ACTIVE_MODEL_TOOLS:
        return ToolResourcePolicy(
            ToolComputeRoute.ACTIVE_MODEL,
            requires_primary_compute_guard=True,
        )
    if value in _HOST_SERVICE_TOOLS:
        service, needs_model = _HOST_SERVICE_TOOLS[value]
        return ToolResourcePolicy(
            ToolComputeRoute.HOST_SERVICE,
            requires_primary_compute_guard=needs_model,
            host_service=service,
        )
    if value in _FLEET_SERVICE_TOOLS:
        return ToolResourcePolicy(
            ToolComputeRoute.FLEET_SERVICE,
            # Media tools may use the active model for prompt enhancement before
            # submitting their separate ComfyUI demand.
            requires_primary_compute_guard=True,
            fleet_service=_FLEET_SERVICE_TOOLS[value],
        )
    if value in _FLEET_CHILD_TOOLS:
        return ToolResourcePolicy(ToolComputeRoute.FLEET_CHILD)
    if value in _DYNAMIC_COMMAND_TOOLS:
        return ToolResourcePolicy(ToolComputeRoute.DYNAMIC_COMMAND)
    if value in _NEXUS_LIFECYCLE_TOOLS:
        return ToolResourcePolicy(ToolComputeRoute.NEXUS_LIFECYCLE)
    if value in _EXTERNAL_PROVIDER_TOOLS:
        # Expert disclosure review uses the active model; MCP traffic itself does not.
        return ToolResourcePolicy(
            ToolComputeRoute.EXTERNAL_PROVIDER,
            requires_primary_compute_guard=value == "consult_external_expert",
        )
    raise ToolResourceError(
        f"tool {value!r} has no reviewed compute-route declaration"
    )


def declared_tool_names() -> frozenset[str]:
    """Expose the complete manifest for hermetic inventory tests."""

    return frozenset(
        set(_LOCAL_CPU_TOOLS)
        | set(_ACTIVE_MODEL_TOOLS)
        | set(_HOST_SERVICE_TOOLS)
        | set(_FLEET_SERVICE_TOOLS)
        | set(_FLEET_CHILD_TOOLS)
        | set(_DYNAMIC_COMMAND_TOOLS)
        | set(_NEXUS_LIFECYCLE_TOOLS)
        | set(_EXTERNAL_PROVIDER_TOOLS)
    )
