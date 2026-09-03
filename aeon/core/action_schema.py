"""Decode-time JSON schema for Aeon's compact turn protocol.

The model chooses exactly one turn kind: return a final answer, ask the user,
wait on an external state, or request tools. Tool branches are tied to each
tool's Python argument schema, so both hallucinated names and misspelled
parameters are rejected during decoding rather than discovered after execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, Iterable


TURN_FIELDS_REQUIRED = ["kind", "intent", "message", "actions"]
TURN_KINDS = ["tool_calls", "final", "ask_user", "wait"]


def _tool_entries(tools: Iterable[Any]) -> list[tuple[str, dict]]:
    entries: list[tuple[str, dict]] = []
    for tool in tools or []:
        if isinstance(tool, str):
            entries.append(
                (tool, {"type": "object", "additionalProperties": True})
            )
            continue
        if isinstance(tool, Mapping):
            name = str(tool.get("name") or "").strip()
            schema = tool.get("parameters")
        else:
            name = str(getattr(tool, "name", "") or "").strip()
            schema_builder = getattr(tool, "parameter_schema", None)
            schema = schema_builder() if callable(schema_builder) else None
        if not name:
            continue
        if not isinstance(schema, dict):
            schema = {"type": "object", "additionalProperties": True}
        entries.append((name, schema))
    # First definition wins, matching the tool loader's duplicate handling.
    return list(dict((name, schema) for name, schema in entries).items())


def _action_schema(entries: list[tuple[str, dict]]) -> Dict:
    goal_refs = {
        "type": "array",
        "maxItems": 13,
        "items": {"type": "string", "pattern": "^G(?:0|[1-9][0-9]?)$"},
    }
    if not entries:
        return {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "parameters": {"type": "object", "additionalProperties": True},
                "goal_refs": goal_refs,
            },
            "required": ["tool_name", "parameters"],
            "additionalProperties": False,
        }
    branches = []
    for name, parameters in entries:
        branches.append(
            {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string", "enum": [name]},
                    "parameters": parameters,
                    "goal_refs": deepcopy(goal_refs),
                },
                # Goal IDs are an optional precision hint.  Evidence ownership is
                # inferred from the typed tool receipt when they are omitted, so
                # the model does not have to operate harness bookkeeping.
                "required": ["tool_name", "parameters"],
                "additionalProperties": False,
            }
        )
    return {"oneOf": branches}


def build_turn_schema(tools: Iterable[Any]) -> Dict:
    """Build the schema for one model decision.

    ``tools`` may be tool objects (the production path), ``{"name",
    "parameters"}`` mappings, or plain names for compatibility tests.
    """

    entries = _tool_entries(tools)
    turn_kinds = list(TURN_KINDS if entries else [
        kind for kind in TURN_KINDS if kind != "tool_calls"
    ])
    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": turn_kinds},
            "intent": {"type": "string"},
            "message": {"type": "string"},
            "updated_plan": {"type": "string"},
            "actions": {
                "type": "array",
                "maxItems": 15,
                "items": _action_schema(entries),
            },
        },
        "required": list(TURN_FIELDS_REQUIRED),
        "additionalProperties": False,
    }
    # Constrain the control-flow choice during decoding, not merely after the
    # model has spent a turn producing an impossible combination. Each union
    # branch deliberately repeats the *complete* object schema. The xgrammar
    # build in the pinned vLLM runtime selects a ``oneOf`` branch directly and
    # does not reliably intersect a partial branch with sibling root
    # ``properties``. Partial branches therefore left action items effectively
    # unconstrained even though a standards-compliant JSON Schema evaluator
    # would apply both. Keep the root projection for consumers that inspect
    # ``properties`` and make every decode branch independently strict.
    branches = []
    for kind in turn_kinds:
        branch_properties = deepcopy(schema["properties"])
        branch_properties["kind"] = {"type": "string", "enum": [kind]}
        if kind == "tool_calls":
            branch_properties["message"] = {"type": "string", "enum": [""]}
            branch_properties["actions"]["minItems"] = 1
        else:
            branch_properties["message"] = {"type": "string", "minLength": 1}
            branch_properties["actions"]["maxItems"] = 0
        branches.append(
            {
                "type": "object",
                "properties": branch_properties,
                "required": list(TURN_FIELDS_REQUIRED),
                "additionalProperties": False,
            }
        )
    schema["oneOf"] = branches
    return schema
