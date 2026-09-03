"""Fixed argv construction for the interchangeable Aeon harnesses.

The public Nexus agent kind remains ``aeon``.  A harness is an implementation
detail selected from the reviewed catalog; it is never a provider name or a
model-controlled executable.
"""

from __future__ import annotations

from .catalog import normalize_harness_id


def build_harness_argv(
    python_executable: str,
    harness_id: str,
    model: str,
    *,
    resume_unfinished: bool = False,
    start_objective: str = "",
) -> list[str]:
    """Return a fixed, shell-free command for one reviewed harness."""

    selected = normalize_harness_id(harness_id)
    executable = str(python_executable or "").strip()
    selected_model = str(model or "").strip()
    if not executable:
        raise ValueError("Python executable is required")
    if not selected_model:
        raise ValueError("Aeon model is required")

    module = (
        "aeon.harnesses.opencode_runtime"
        if selected == "opencode"
        else "aeon.main"
    )
    argv = [executable, "-m", module, "--model", selected_model]
    if resume_unfinished:
        argv.append("--resume-unfinished")
    if start_objective:
        argv.extend(["--start", str(start_objective)])
    return argv


__all__ = ("build_harness_argv",)
