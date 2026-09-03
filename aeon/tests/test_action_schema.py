"""Focused regressions for the decode-time Aeon turn envelope."""

from aeon.core.action_schema import TURN_FIELDS_REQUIRED, build_turn_schema
from aeon.core.prompts import (
    PRIMARY_AGENT_INSTRUCTIONS,
    TOOL_DESC_BROWSER_CAPTURE_MEDIA,
    TOOL_DESC_BROWSER_INTERACT,
    TOOL_DESC_EDIT_IMAGE,
    TOOL_DESC_GENERATE_VIDEO,
    TOOL_DESC_THINK,
)
from aeon.tools.base import BaseTool


EXPECTED_ACTION_REQUIRED = ["tool_name", "parameters"]
EXPECTED_GOAL_REFS_SCHEMA = {
    "type": "array",
    "maxItems": 13,
    "items": {"type": "string", "pattern": "^G(?:0|[1-9][0-9]?)$"},
}


def test_prompt_keeps_execution_policy_compact_and_action_oriented():
    assert "installed schema" in PRIMARY_AGENT_INSTRUCTIONS
    assert "goal_refs` are optional precision hints" in PRIMARY_AGENT_INSTRUCTIONS
    assert "Do not narrate internal" in PRIMARY_AGENT_INSTRUCTIONS
    assert "RECOVERY REQUIRED" in PRIMARY_AGENT_INSTRUCTIONS
    assert "Do not retry a barred action" in PRIMARY_AGENT_INSTRUCTIONS
    assert len(PRIMARY_AGENT_INSTRUCTIONS) < 2_500
    assert "Primary strong-model reasoning call at xhigh effort" in TOOL_DESC_THINK


def test_each_turn_branch_projects_the_complete_required_envelope():
    schema = build_turn_schema(["run_command"])
    branches = schema["oneOf"]

    for branch in branches:
        assert branch["type"] == "object"
        assert branch["required"] == TURN_FIELDS_REQUIRED
        assert branch["additionalProperties"] is False
        assert branch["properties"].keys() == schema["properties"].keys()
        assert branch["properties"]["intent"] == schema["properties"]["intent"]
        assert branch["properties"]["updated_plan"] == {
            "type": "string",
        }

    tool_branch = next(
        branch
        for branch in branches
        if branch["properties"]["kind"]["enum"] == ["tool_calls"]
    )
    assert tool_branch["properties"]["message"] == {
        "type": "string",
        "enum": [""],
    }
    assert tool_branch["properties"]["actions"]["minItems"] == 1
    assert tool_branch["properties"]["actions"]["maxItems"] == 15
    assert tool_branch["properties"]["actions"]["items"] == schema["properties"][
        "actions"
    ]["items"]
    for branch in branches:
        if branch is tool_branch:
            continue
        assert branch["properties"]["message"] == {
            "type": "string",
            "minLength": 1,
        }
        assert branch["properties"]["actions"]["maxItems"] == 0
        assert branch["properties"]["actions"]["items"] == schema["properties"][
            "actions"
        ]["items"]

    assert "updated_plan" not in TURN_FIELDS_REQUIRED
    assert "updated_plan" in schema["properties"]
    assert schema["additionalProperties"] is False


def test_each_union_branch_accepts_optional_goal_refs_and_retains_exact_tool_arguments():
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    }
    schema = build_turn_schema([{"name": "run_command", "parameters": parameters}])

    for branch in schema["oneOf"]:
        action_union = branch["properties"]["actions"]["items"]
        action = action_union["oneOf"][0]
        assert action["properties"]["tool_name"]["enum"] == ["run_command"]
        assert action["properties"]["parameters"] == parameters
        assert action["properties"]["goal_refs"] == EXPECTED_GOAL_REFS_SCHEMA
        assert action["required"] == EXPECTED_ACTION_REQUIRED
        assert action["additionalProperties"] is False


def test_zero_tool_schema_omits_impossible_tool_call_turns():
    schema = build_turn_schema([])

    assert "tool_calls" not in schema["properties"]["kind"]["enum"]
    assert all(
        branch["properties"]["kind"]["enum"] != ["tool_calls"]
        for branch in schema["oneOf"]
    )
    assert all(
        branch["properties"]["actions"]["maxItems"] == 0
        for branch in schema["oneOf"]
    )


def test_runtime_tool_examples_do_not_inject_optional_harness_bookkeeping():
    class ExampleTool(BaseTool):
        def __init__(self, description):
            super().__init__(
                "open_file",
                description,
            )

        def execute(self, file_path: str):
            return file_path

    descriptions = (
        'Fixture.\nExample: {"tool_name":"open_file","parameters":{"file_path":"x.py"}}',
        TOOL_DESC_GENERATE_VIDEO,
        TOOL_DESC_EDIT_IMAGE,
        TOOL_DESC_BROWSER_CAPTURE_MEDIA,
        TOOL_DESC_BROWSER_INTERACT,
    )
    for source in descriptions:
        rendered = ExampleTool(source).description
        example_lines = [
            line.strip()
            for line in rendered.splitlines()
            if '"tool_name"' in line and "{" in line
        ]
        assert example_lines
        assert all('"goal_refs"' not in line for line in example_lines)
