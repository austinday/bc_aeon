"""Hermetic tests for the sanitized normal-Aeon qualification wrapper."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aeon.scripts import gate_qwen38_normal_aeon as gate


def _passing_transcript() -> str:
    return "\n".join(
        (
            "[VISION SELF-TEST] PASS — model read the probe code 'ACF347'. Vision trusted for browsing.",
            "▶ [1/1] run_command(command=pwd, timeout=30)",
            gate.EXPECTED_PWD,
            f"OK: COMMAND SUCCESS\n\nOUTPUT:\n{gate.EXPECTED_PWD}",
            gate.FINAL_MESSAGE,
            "[CONFIG] Non-interactive mode: objective complete, exiting.",
            "[SESSION] Fleet broker ticket release verified.",
            "[SESSION] Cleanup complete.",
        )
    )


def test_child_command_is_the_ordinary_bounded_aeon_cli() -> None:
    command = gate.child_command()

    assert command[:5] == [os.sys.executable, "-m", "aeon", "-n", "--start"]
    assert command[-2:] == ["--max-iterations", "4"]


def test_child_environment_requires_exact_profile_and_forces_real_vision(tmp_path: Path) -> None:
    environment = gate.child_environment(
        {
            "AEON_FLEET_PROFILE": gate.PROFILE,
            "AEON_CHAT_TRANSCRIPT_PATH": "/private/transcript",
            "AEON_REMOTE_INSTANCE_ID": "a" * 32,
        },
        tmp_path,
    )

    assert environment["AEON_FLEET_PROFILE"] == gate.PROFILE
    assert environment["AEON_DISABLE_AUTO_TMUX"] == "1"
    assert environment["AEON_STATE_DIR"] == str(tmp_path)
    assert "AEON_CHAT_TRANSCRIPT_PATH" not in environment
    assert "AEON_REMOTE_INSTANCE_ID" not in environment
    assert "AEON_SKIP_VISION_SELFTEST" not in environment

    with pytest.raises(gate.NormalAeonGateError, match="must be set exactly"):
        gate.child_environment({})
    with pytest.raises(gate.NormalAeonGateError, match="bypass is forbidden"):
        gate.child_environment(
            {
                "AEON_FLEET_PROFILE": gate.PROFILE,
                "AEON_SKIP_VISION_SELFTEST": "1",
            }
        )


def test_transcript_requires_every_normal_agent_gate() -> None:
    gates = gate.validate_sanitized_transcript(0, _passing_transcript())

    assert all(gates.values())
    assert gates["single_exact_pwd_action"] is True
    assert gates["ticket_release_verified"] is True


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda text: text.replace("[VISION SELF-TEST] PASS", "vision absent"), "startup_vision"),
        (
            lambda text: text.replace(
                "▶ [1/1] run_command(command=pwd, timeout=30)",
                "▶ [1/1] open_file(file_path=/etc/hostname)",
            ),
            "single_exact_pwd_action",
        ),
        (lambda text: text.replace(gate.FINAL_MESSAGE, "unverified success"), "truthful_final"),
        (
            lambda text: text.replace(
                "[SESSION] Fleet broker ticket release verified.", "release missing"
            ),
            "ticket_release_verified",
        ),
        (
            lambda text: text + "\n[WARN] Fleet broker ticket release failed: no proof",
            "failure or bypass",
        ),
    ),
)
def test_transcript_fails_closed_on_missing_or_contradictory_evidence(mutation, match) -> None:
    with pytest.raises(gate.NormalAeonGateError, match=match):
        gate.validate_sanitized_transcript(0, mutation(_passing_transcript()))


def test_run_gate_writes_only_sanitized_private_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "normal-agent.json"
    monkeypatch.setenv("AEON_FLEET_PROFILE", gate.PROFILE)
    monkeypatch.setattr(
        gate,
        "_run_child",
        lambda command, environment, *, timeout_seconds: (
            0,
            _passing_transcript(),
            12.5,
        ),
    )

    result = gate.run_gate(output, timeout_seconds=60)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted == result
    assert result["status"] == "passed"
    assert result["profile"] == gate.PROFILE
    assert result["gates"]["exact_pwd_receipt"] is True
    assert output.stat().st_mode & 0o777 == 0o600
    serialized = json.dumps(result)
    for forbidden in ("ACF347", "ticket_id", "endpoint", "reasoning", "process_id"):
        assert forbidden not in serialized

    with pytest.raises(gate.NormalAeonGateError, match="refusing to overwrite"):
        gate.run_gate(output, timeout_seconds=60)
