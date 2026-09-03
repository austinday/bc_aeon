"""Focused registration and category-marker checks for learned-skill state."""

from types import SimpleNamespace

from aeon.core.agent_protocol import SideEffect, infer_tool_policy
from aeon.core.skills.knowledge import SKILL_PATH_RE
from aeon.core.tool_resources import ToolComputeRoute, tool_resource_policy
from aeon.tools.categories import TOP_LEVEL_TOOLS
from aeon.tools.skills_manager_tool import (
    CollapseSkillsCategory,
    ExpandSkillsCategory,
)
from aeon.tools.skills_runtime import (
    ActivateSkillTool,
    CreateSkillTool,
    ReadSkillTool,
    RememberSkillKnowledgeTool,
)


def test_skill_knowledge_lifecycle_tools_are_visible_local_cpu_capabilities() -> None:
    for name in ("search_skill_knowledge", "delete_skill_knowledge"):
        assert name in TOP_LEVEL_TOOLS
        assert tool_resource_policy(name).route == ToolComputeRoute.LOCAL_CPU


def test_skill_lifecycle_writes_are_agent_state_not_external_authority() -> None:
    for name in (
        "activate_skill",
        "deactivate_skill",
        "create_skill",
        "delete_skill",
        "delete_skill_knowledge",
        "remember_skill_knowledge",
    ):
        assert infer_tool_policy(name).side_effect == SideEffect.AGENT_STATE


def test_plural_skill_category_tools_use_worker_skill_marker() -> None:
    worker = SimpleNamespace(expanded_categories=set())

    expanded = ExpandSkillsCategory(worker=worker).execute("research")
    assert "expanded" in expanded
    assert worker.expanded_categories == {"skill:research"}

    collapsed = CollapseSkillsCategory(worker=worker).execute("research")
    assert "collapsed" in collapsed
    assert worker.expanded_categories == set()


def test_learning_evidence_fields_belong_to_wiki_note_schema() -> None:
    create_properties = CreateSkillTool().parameter_schema()["properties"]
    remember_properties = RememberSkillKnowledgeTool().parameter_schema()["properties"]

    assert "learning" not in create_properties
    assert "clear_learning" not in create_properties
    assert remember_properties["learning"]["required"] == [
        "candidate_skill_path",
        "procedure",
        "verification",
        "procedure_stable",
        "uncertainty",
    ]
    assert remember_properties["clear_learning"] == {"type": "boolean"}


def test_skill_read_and_activation_require_exact_path_schema() -> None:
    for tool in (ActivateSkillTool(), ReadSkillTool()):
        schema = tool.parameter_schema()
        assert schema["required"] == ["skill_path"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["skill_path"]["pattern"] == SKILL_PATH_RE.pattern
