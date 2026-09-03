"""Release-gate regressions for the Qwen3.8 MTP semantic probe."""

from copy import deepcopy
import json

import pytest

from aeon.scripts import benchmark_qwen38_mtp as benchmark


def _canonical_turn(case):
    return {
        "kind": "tool_calls",
        "intent": case["marker"],
        "message": "",
        "actions": [
            {
                "tool_name": case["expected_tool"],
                "parameters": deepcopy(case["expected_parameters"]),
            }
        ],
    }


@pytest.mark.parametrize("case", benchmark.CASES, ids=lambda case: case["name"])
def test_release_cases_accept_only_the_canonical_turn(case):
    turn = _canonical_turn(case)

    assert benchmark._validated_turn(json.dumps(turn), case) == turn


@pytest.mark.parametrize(
    "mutation",
    (
        lambda turn: turn.update(kind="final"),
        lambda turn: turn.update(message="premature success"),
        lambda turn: turn["actions"][0].update(action=turn["actions"][0].pop("tool_name")),
        lambda turn: turn["actions"][0].update(extra="not allowed"),
    ),
)
def test_release_gate_rejects_envelopes_that_only_look_semantically_plausible(mutation):
    case = benchmark.CASES[0]
    turn = _canonical_turn(case)
    mutation(turn)

    with pytest.raises(ValueError, match="turn-schema gate failed"):
        benchmark._validated_turn(json.dumps(turn), case)


def test_release_gate_rejects_the_wrong_exact_action():
    case = benchmark.CASES[2]
    turn = _canonical_turn(case)
    turn["actions"][0]["parameters"]["timeout"] = 31

    with pytest.raises(ValueError, match="agent-action gate failed"):
        benchmark._validated_turn(json.dumps(turn), case)
