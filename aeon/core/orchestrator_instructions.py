"""Fixed role for the one Nexus main-orchestrator Aeon process."""

from __future__ import annotations

import os

from .prompts import MAIN_ORCHESTRATOR_INSTRUCTIONS, load_prompt


MAIN_ORCHESTRATOR_ENV = "AEON_MAIN_ORCHESTRATOR"

def main_orchestrator_instruction_section() -> str:
    """Return the fixed role only for a server-marked main orchestrator launch."""

    return (
        "\n\n" + load_prompt("main_orchestrator_instructions.txt")
        if os.environ.get(MAIN_ORCHESTRATOR_ENV) == "1"
        else ""
    )
