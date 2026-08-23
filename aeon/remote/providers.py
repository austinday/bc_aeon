"""Fixed, subscription-backed provider commands for Aeon Remote/Nexus.

This module deliberately does not implement OAuth, accept tokens, or expose CLI
output.  It only resolves an explicitly allowlisted official CLI, runs a documented
read-only status command with every stream discarded when one exists, and returns
immutable direct-argv launch specifications for a separately managed interactive
terminal.

Provider credentials remain entirely in the official CLI's credential store.
Callers must never place an authentication URL, one-time code, account identity,
or provider output in an API response, audit record, or application log.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


STATUS_TIMEOUT_SECONDS = 8


class ProviderError(ValueError):
    """Base error for invalid or unavailable provider requests."""


class ProviderUnavailableError(ProviderError):
    """Raised when an allowlisted provider's executable is unavailable."""


@dataclass(frozen=True)
class ProviderSpec:
    """Server-owned command specification; no field is browser-controlled."""

    provider_id: str
    label: str
    binary: str
    status_args: tuple[str, ...] | None
    connect_args: tuple[str, ...]
    agent_args: tuple[str, ...]
    instruction_filename: str
    documentation_url: str
    stripped_environment: tuple[str, ...]


@dataclass(frozen=True)
class ProviderCommand:
    """A direct argv plus environment names that must not reach the CLI.

    The environment metadata matters because an API key or cloud-provider flag
    can take precedence over a cached consumer subscription.  A session manager
    can either pass :func:`subscription_environment` to a direct subprocess or
    use an equivalently fixed, non-shell environment-unsetting launcher.
    """

    provider_id: str
    purpose: str
    argv: tuple[str, ...]
    stripped_environment: tuple[str, ...]


_OPENAI_ENVIRONMENT = (
    "CODEX_ACCESS_TOKEN",
    "OPENAI_API_KEY",
)

_ANTHROPIC_ENVIRONMENT = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
    "CLAUDE_CODE_OAUTH_SCOPES",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_VERTEX",
)

_XAI_ENVIRONMENT = ("XAI_API_KEY",)

# Status probes need the user's official CLI credential cache and ordinary
# locale/runtime paths, not the web service's complete environment.  An
# allowlist prevents an unrelated Nexus/Cloudflare/database secret from being
# inherited merely because its variable name was not anticipated here.
_SAFE_ENVIRONMENT = frozenset(
    {
        "CODEX_CA_CERTIFICATE",
        "CODEX_HOME",
        "COLORTERM",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "NODE_EXTRA_CA_CERTS",
        "PATH",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TMPDIR",
        "USER",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "CLAUDE_CONFIG_DIR",
    }
)


# Keep this an explicit provider allowlist.  Adding another CLI requires an
# independent review of its installed help, authentication contract, status
# semantics, credential precedence, and interactive terminal behavior.
_PROVIDERS: Mapping[str, ProviderSpec] = {
    "codex": ProviderSpec(
        provider_id="codex",
        label="OpenAI Codex",
        binary="codex",
        status_args=("login", "status"),
        connect_args=("login", "--device-auth"),
        # Inline mode works predictably through the existing tmux/WebSocket bridge
        # and preserves terminal scrollback on reconnect.
        agent_args=("--no-alt-screen",),
        instruction_filename="AGENTS.md",
        documentation_url="https://developers.openai.com/codex/auth",
        stripped_environment=_OPENAI_ENVIRONMENT,
    ),
    "claude": ProviderSpec(
        provider_id="claude",
        label="Anthropic Claude Code",
        binary="claude",
        status_args=("auth", "status", "--text"),
        connect_args=("auth", "login", "--claudeai"),
        agent_args=(),
        instruction_filename="CLAUDE.md",
        documentation_url="https://code.claude.com/docs/en/authentication",
        stripped_environment=_ANTHROPIC_ENVIRONMENT,
    ),
    "grok": ProviderSpec(
        provider_id="grok",
        label="xAI Grok Build",
        binary="grok",
        # The current CLI and official product page do not document a
        # non-sensitive authentication-status command.  Never infer one from
        # config files, models output, account endpoints, or login behavior.
        status_args=None,
        connect_args=("login", "--device-auth"),
        agent_args=(),
        # xAI's official launch announcement says Grok Build loads AGENTS.md.
        instruction_filename="AGENTS.md",
        documentation_url="https://x.ai/news/grok-build-cli",
        stripped_environment=_XAI_ENVIRONMENT,
    ),
}

PROVIDER_IDS = frozenset(_PROVIDERS)


def provider_spec(provider_id: str) -> ProviderSpec:
    """Return an allowlisted provider or reject the identifier exactly."""
    if not isinstance(provider_id, str) or provider_id not in _PROVIDERS:
        raise ProviderError("Unsupported provider")
    return _PROVIDERS[provider_id]


def _resolve_executable(
    spec: ProviderSpec,
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> str | None:
    """Resolve one fixed binary to a non-world-writable executable file."""
    try:
        discovered = which(spec.binary)
    except (OSError, TypeError, ValueError):
        return None
    if not discovered or "\x00" in discovered:
        return None
    try:
        resolved = Path(discovered).expanduser().resolve(strict=True)
        info = resolved.stat()
        parent_info = resolved.parent.stat()
    except (OSError, RuntimeError, ValueError):
        return None
    if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
        return None
    # A world-writable executable or containing directory lets another local
    # principal swap the program between the check and the managed launch.
    if info.st_mode & stat.S_IWOTH or parent_info.st_mode & stat.S_IWOTH:
        return None
    return str(resolved)


def subscription_environment(
    provider_id: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment in which cached subscription auth has precedence.

    The original mapping is never changed.  Provider-native cache location
    variables (for example ``CODEX_HOME`` and ``CLAUDE_CONFIG_DIR``) are retained.
    Everything outside a small runtime allowlist is removed, including credential
    selectors that override subscriptions and unrelated web-service secrets.
    """
    spec = provider_spec(provider_id)
    source = os.environ if environ is None else environ
    result = {}
    for key, value in source.items():
        if key not in _SAFE_ENVIRONMENT or key in spec.stripped_environment:
            continue
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if "\x00" in key or "\x00" in value:
            continue
        result[key] = value
    return result


def provider_status(
    provider_id: str,
    *,
    runner: Callable[..., object] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    environ: Mapping[str, str] | None = None,
) -> dict[str, bool | str | None]:
    """Return only non-sensitive installed/connected state.

    Account identity and raw CLI output are intentionally unrecoverable: stdin,
    stdout, and stderr are all attached to ``DEVNULL``.  A timeout, signal,
    malformed runner result, or any process error fails closed as disconnected
    for reviewed status commands.  Providers without one report ``None`` rather
    than guessing from credential files or other CLI behavior.
    """
    spec = provider_spec(provider_id)
    executable = _resolve_executable(spec, which=which)
    installed = executable is not None
    connected: bool | None = None if spec.status_args is None else False
    if executable is not None and spec.status_args is not None:
        try:
            result = runner(
                [executable, *spec.status_args],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=STATUS_TIMEOUT_SECONDS,
                env=subscription_environment(provider_id, environ),
            )
            connected = getattr(result, "returncode", None) == 0
        except (OSError, subprocess.SubprocessError, TypeError, ValueError):
            connected = False
    return {
        "id": spec.provider_id,
        "label": spec.label,
        "installed": installed,
        "connected": connected,
    }


def list_provider_statuses(
    *,
    runner: Callable[..., object] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, bool | str | None]]:
    """Return the stable allowlist order without accepting caller identifiers."""
    return [
        provider_status(
            provider_id,
            runner=runner,
            which=which,
            environ=environ,
        )
        for provider_id in ("codex", "claude", "grok")
    ]


def _provider_command(
    provider_id: str,
    *,
    purpose: str,
    which: Callable[[str], str | None],
) -> ProviderCommand:
    spec = provider_spec(provider_id)
    executable = _resolve_executable(spec, which=which)
    if executable is None:
        raise ProviderUnavailableError(f"{spec.label} CLI is not safely installed")
    if purpose == "connect":
        args = spec.connect_args
    elif purpose == "agent":
        args = spec.agent_args
    else:  # Internal invariant, never caller-controlled.
        raise ProviderError("Unsupported provider command purpose")
    return ProviderCommand(
        provider_id=provider_id,
        purpose=purpose,
        argv=(executable, *args),
        stripped_environment=spec.stripped_environment,
    )


def provider_connect_command(
    provider_id: str,
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> ProviderCommand:
    """Return the fixed native subscription-login argv for a managed terminal."""
    return _provider_command(provider_id, purpose="connect", which=which)


def provider_agent_command(
    provider_id: str,
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> ProviderCommand:
    """Return the fixed native interactive-agent argv for a managed terminal."""
    return _provider_command(provider_id, purpose="agent", which=which)


__all__ = (
    "PROVIDER_IDS",
    "ProviderCommand",
    "ProviderError",
    "ProviderSpec",
    "ProviderUnavailableError",
    "list_provider_statuses",
    "provider_agent_command",
    "provider_connect_command",
    "provider_spec",
    "provider_status",
    "subscription_environment",
)
