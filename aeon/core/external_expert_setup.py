"""Interactive setup for subscription-backed external expert CLIs.

OAuth credentials are created and stored only by each provider's official CLI.
Aeon persists the provider choice, model, and reasoning effort, never a token.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from aeon.tools.external_expert import (
    ExternalExpertConfig,
    save_external_expert_settings,
)


PROVIDERS = {
    "codex": {
        "label": "Codex / ChatGPT subscription",
        "status": ["login", "status"],
        "login": ["login", "--device-auth"],
        "install": "Install/update the official Codex CLI, then restart Aeon.",
    },
    "claude": {
        "label": "Claude Pro / Max / Team subscription",
        "status": ["auth", "status", "--text"],
        "login": ["auth", "login", "--claudeai"],
        "install": "Install the official Claude Code CLI, then restart Aeon.",
    },
    "gemini": {
        "label": "Gemini Google account (AI Pro / Ultra supported)",
        "status": None,
        "login": [],
        "install": "Install @google/gemini-cli from the official Google package, then restart Aeon.",
    },
}


# Used only when an older Codex CLI cannot return its own catalog. The normal path
# asks the installed CLI after login, so its current catalog remains authoritative.
_CODEX_CATALOG_FALLBACK = (
    {
        "slug": "gpt-5.6-sol",
        "display_name": "GPT-5.6 Sol",
        "description": "Latest frontier agentic coding model.",
        "default_reasoning_level": "low",
        "supported_reasoning_levels": (
            "low", "medium", "high", "xhigh", "max", "ultra"
        ),
    },
    {
        "slug": "gpt-5.6-terra",
        "display_name": "GPT-5.6 Terra",
        "description": "Balanced agentic coding model for everyday work.",
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": (
            "low", "medium", "high", "xhigh", "max", "ultra"
        ),
    },
    {
        "slug": "gpt-5.6-luna",
        "display_name": "GPT-5.6 Luna",
        "description": "Fast and affordable agentic coding model.",
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": ("low", "medium", "high", "xhigh", "max"),
    },
)


def _codex_model_catalog(executable: str, *, runner=subprocess.run) -> list[dict]:
    """Return the models and effort levels advertised by the installed CLI."""
    for extra in ((), ("--bundled",)):
        try:
            result = runner(
                [executable, "debug", "models", *extra],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            payload = json.loads(result.stdout) if result.returncode == 0 else {}
        except (OSError, subprocess.SubprocessError, ValueError, TypeError):
            continue
        catalog = []
        for item in payload.get("models", []):
            if not isinstance(item, dict) or item.get("visibility", "list") != "list":
                continue
            slug = str(item.get("slug", "")).strip()
            if not slug:
                continue
            efforts = []
            for level in item.get("supported_reasoning_levels", []):
                effort = level.get("effort") if isinstance(level, dict) else level
                effort = str(effort or "").strip().lower()
                if effort and effort not in efforts:
                    efforts.append(effort)
            catalog.append(
                {
                    "slug": slug,
                    "display_name": str(item.get("display_name") or slug).strip(),
                    "description": str(item.get("description") or "").strip(),
                    "default_reasoning_level": str(
                        item.get("default_reasoning_level") or ""
                    ).strip().lower(),
                    "supported_reasoning_levels": tuple(efforts),
                }
            )
        if catalog:
            return catalog
    return [dict(item) for item in _CODEX_CATALOG_FALLBACK]


def _choose_codex_model(
    executable: str, *, input_fn=input, print_fn=print, runner=subprocess.run
) -> tuple[str, str] | None:
    catalog = _codex_model_catalog(executable, runner=runner)
    print_fn("\nChoose the Codex model used for difficult external consultations:")
    for index, model in enumerate(catalog, 1):
        description = f" — {model['description']}" if model["description"] else ""
        print_fn(f" {index}. {model['display_name']} ({model['slug']}){description}")
    default_model = catalog[0]
    while True:
        try:
            selected = input_fn(
                f"Choose Codex model [1 = {default_model['slug']}]: "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not selected:
            model = default_model
            break
        if selected.isdigit() and 1 <= int(selected) <= len(catalog):
            model = catalog[int(selected) - 1]
            break
        print_fn(f"Invalid choice. Enter 1-{len(catalog)}.")

    efforts = list(model["supported_reasoning_levels"])
    if not efforts:
        efforts = ["low", "medium", "high", "xhigh"]
    default_effort = model["default_reasoning_level"]
    if default_effort not in efforts:
        default_effort = "medium" if "medium" in efforts else efforts[0]
    print_fn(f"\nChoose reasoning effort for {model['display_name']}:")
    for index, effort in enumerate(efforts, 1):
        marker = " (model default)" if effort == default_effort else ""
        print_fn(f" {index}. {effort}{marker}")
    default_index = efforts.index(default_effort) + 1
    while True:
        try:
            selected = input_fn(
                f"Choose reasoning effort [{default_index} = {default_effort}]: "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not selected:
            effort = default_effort
            break
        if selected.isdigit() and 1 <= int(selected) <= len(efforts):
            effort = efforts[int(selected) - 1]
            break
        print_fn(f"Invalid choice. Enter 1-{len(efforts)}.")
    return model["slug"], effort


def state_dir() -> Path:
    return Path(
        os.environ.get("AEON_EXTERNAL_EXPERT_STATE_DIR", "~/.aeon/external_expert")
    ).expanduser().resolve()


def provider_status(provider: str, *, runner=subprocess.run, which=shutil.which):
    """Return True/False for known status, None when the CLI has no status command."""
    spec = PROVIDERS[provider]
    executable = which(provider)
    if not executable:
        return False
    if not spec["status"]:
        return None
    try:
        result = runner(
            [executable, *spec["status"]],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def external_expert_menu_label() -> str:
    config = ExternalExpertConfig.from_env()
    if not config.enabled:
        return "External expert account: configure/login (optional)"
    labels = {
        "codex": "Codex / ChatGPT",
        "claude": "Claude",
        "gemini": "Gemini",
        "api": "API",
    }
    provider = labels.get(config.backend, config.backend)
    if config.backend == "codex" and not (
        config.model and config.reasoning_effort
    ):
        return (
            "External expert account: Codex / ChatGPT needs model/effort "
            "selection (manage)"
        )
    model = f", model={config.model}" if config.model else ""
    effort = (
        f", effort={config.reasoning_effort}"
        if config.backend == "codex" and config.reasoning_effort
        else ""
    )
    return f"External expert account: {provider} enabled{model}{effort} (manage)"


def configure_external_expert_interactive(
    *, input_fn=input, print_fn=print, runner=subprocess.run, which=shutil.which
) -> bool:
    """Run one official CLI login flow and persist only the provider selection."""
    while True:
        print_fn("\n[EXTERNAL EXPERT] Optional subscription login")
        print_fn("Qwen3.8 remains Aeon's primary model. The selected account is used only")
        print_fn("for budgeted advice after Aeon's stall detector fires.\n")
        choices = ["codex", "claude", "gemini"]
        for index, provider in enumerate(choices, 1):
            spec = PROVIDERS[provider]
            installed = bool(which(provider))
            status = provider_status(provider, runner=runner, which=which) if installed else False
            if not installed:
                marker = "not installed"
            elif status is True:
                marker = "signed in"
            elif status is False:
                marker = "sign-in required"
            else:
                marker = "installed; sign in through CLI"
            print_fn(f" {index}. {spec['label']} [{marker}]")
        print_fn(" 4. Disable external expert")
        print_fn(" 5. Back to model selection")
        try:
            selected = input_fn("Choose external account (1-5): ").strip()
        except (KeyboardInterrupt, EOFError):
            return False
        if selected == "5":
            return False
        if selected == "4":
            save_external_expert_settings(state_dir(), {"enabled": False})
            print_fn("External expert disabled. Qwen3.8 will remain fully local.")
            return True
        if selected not in {"1", "2", "3"}:
            print_fn("Invalid choice.")
            continue

        provider = choices[int(selected) - 1]
        spec = PROVIDERS[provider]
        executable = which(provider)
        if not executable:
            print_fn(spec["install"])
            continue

        status = provider_status(provider, runner=runner, which=which)
        if status is True:
            try:
                reuse = input_fn("A valid login exists. Use this account? [Y/n]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                return False
            if reuse not in {"", "y", "yes"}:
                status = False
        if status is not True:
            print_fn("Launching the provider's official login flow. Aeon never sees the token.")
            if provider == "gemini":
                print_fn("Choose 'Sign in with Google', complete login, then use /quit to return.")
            try:
                result = runner([executable, *spec["login"]], check=False)
            except OSError as exc:
                print_fn(f"Could not launch {provider}: {exc}")
                continue
            if result.returncode != 0:
                print_fn(f"{provider} login did not complete successfully.")
                continue
            if provider != "gemini" and provider_status(
                provider, runner=runner, which=which
            ) is not True:
                print_fn(f"{provider} still reports that it is not signed in.")
                continue

        reasoning_effort = ""
        if provider == "codex":
            selection = _choose_codex_model(
                executable, input_fn=input_fn, print_fn=print_fn, runner=runner
            )
            if selection is None:
                return False
            model, reasoning_effort = selection
        else:
            try:
                model = input_fn(
                    "Optional exact expert model ID [Enter = provider default]: "
                ).strip()
            except (KeyboardInterrupt, EOFError):
                return False
        save_external_expert_settings(
            state_dir(),
            {
                "enabled": True,
                "backend": provider,
                "model": model,
                "reasoning_effort": reasoning_effort,
            },
        )
        if "AEON_EXTERNAL_EXPERT_ENABLED" in os.environ:
            print_fn(
                "Note: AEON_EXTERNAL_EXPERT_ENABLED in the environment overrides this saved choice."
            )
        detail = f" using {model}" if model else ""
        if reasoning_effort:
            detail += f" at {reasoning_effort} reasoning effort"
        print_fn(f"External expert configured through {spec['label']}{detail}.")
        return True
