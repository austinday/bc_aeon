"""JSON Schema for one agent turn — the contract enforced at DECODE time.

This is the core of the structured-output architecture: instead of asking the
model to free-form emit a JSON envelope and then parsing/repairing whatever
comes back (the old path — see the repair cascade in llm.py — which burned
multiple LLM calls per malformed turn), the schema built here is handed to the
inference server (vLLM structured outputs, xgrammar backend), whose sampler
masks every token that would violate it. Malformed JSON and hallucinated tool
names become impossible BY CONSTRUCTION, not errors to recover from.

Design constraints (deliberate):
- Only universally-supported JSON Schema keywords (type/properties/required/
  enum/items/additionalProperties). Fancier constraints (maxLength, maxItems)
  vary by grammar backend; an unsupported keyword would 400 the request and
  needlessly downgrade the whole session to the legacy parse path.
- Property order matters: xgrammar emits object properties in schema order, so
  'thought' comes FIRST — the model reasons before it acts, inside the grammar
  (there is no room for <think> preambles: the grammar owns token 0 onward).
- 'parameters' is a free-form object (additionalProperties: true). We constrain
  WHAT tool is called, not its arguments: over-constraining args from
  introspected signatures risks the grammar forbidding a legitimate call, and
  the worker already surfaces precise signature hints on a bad-arg TypeError.
- 'updated_plan' stays optional — omitting it keeps the previous plan.
"""
from __future__ import annotations

from typing import Dict, List

# Mirrors the worker's MAX_ACTIONS clamp (kept as a soft limit there; not
# grammar-enforced — see the design constraints above).
TURN_FIELDS_REQUIRED = [
    "thought", "previous_result_summary", "skill_check",
    "memory_check", "parallel_check", "intent", "actions",
]


def build_turn_schema(tool_names: List[str]) -> Dict:
    """Build the strict JSON schema for one agent turn.

    tool_names: the registered tool names; becomes the enum for 'tool_name' so
    the grammar cannot emit a tool that does not exist.
    """
    action_schema: Dict = {
        "type": "object",
        "properties": {
            "tool_name": {"type": "string"},
            "parameters": {"type": "object", "additionalProperties": True},
        },
        "required": ["tool_name", "parameters"],
        "additionalProperties": False,
    }
    if tool_names:
        action_schema["properties"]["tool_name"]["enum"] = sorted(set(tool_names))

    return {
        "type": "object",
        "properties": {
            # Order = generation order (xgrammar preserves it): reason first.
            "thought": {"type": "string"},
            "previous_result_summary": {"type": "string"},
            "skill_check": {"type": "string"},
            "memory_check": {"type": "string"},
            "parallel_check": {"type": "string"},
            "intent": {"type": "string"},
            "updated_plan": {"type": "string"},
            "actions": {"type": "array", "items": action_schema},
        },
        "required": list(TURN_FIELDS_REQUIRED),
        "additionalProperties": False,
    }
