"""Owner-private, version-pinned OpenCode configuration for one Aeon process."""

from __future__ import annotations

import json
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Mapping

from aeon.core.qwen_runtime import QWEN_CONTEXT_TOKENS
from aeon.core.runtime_instructions import (
    RUNTIME_INSTRUCTIONS_ENV,
    RuntimeInstructionError,
    format_runtime_instruction_layers,
    load_runtime_instructions,
)


MAX_AUTHORITY_BYTES = 40_000
MAX_OPENCODE_STEPS = 32
DEFAULT_OPENCODE_STEPS = 12

# OpenCode and its MCP subprocess must not inherit an interactive shell or
# long-lived service environment wholesale.  In particular, the reviewed Aeon
# command tool can inspect its own environment, so an API key that reaches this
# boundary becomes model-readable even though OpenCode's built-in shell is
# disabled.  Keep this a positive allowlist of non-secret runtime values and
# file-backed Nexus capabilities.  The one ephemeral model-gateway bearer is
# added below from a supervisor-owned argument, never from ``base``.
_OPENCODE_CHILD_ENVIRONMENT = frozenset(
    {
        # Ordinary process/runtime paths.  PYTHONPATH is deliberately rebuilt
        # from this installed package below instead of accepted from the caller.
        "COLORTERM",
        "DBUS_SESSION_BUS_ADDRESS",
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
        "TZ",
        "USER",
        "VIRTUAL_ENV",
        "XDG_RUNTIME_DIR",
        # Broker selection contains no allocation/lease authority.  Every tool
        # still submits its own durable request to the one Fleet broker.
        "AEON_COMPUTE_BACKEND",
        "AEON_FLEET_PROFILE",
        "AEON_FLEET_SOCKET",
        # Aeon state, behavior, browser, and bounded-runtime configuration.
        "AEON_BLACKBOARD_PATH",
        "AEON_BROWSER_PROFILE",
        "AEON_BROWSER_SESSION_ID",
        "AEON_BROWSER_TOKEN_FILE",
        "AEON_BENCHMARK_GPU_CAPABILITY_RECEIPT_KEY",
        "AEON_BENCHMARK_GPU_CAPABILITY_RECEIPT_PATH",
        "AEON_CHAT_TRANSCRIPT_PATH",
        "AEON_CHAT_WRITER_PID",
        "AEON_COLLABORATOR_MODE_PATH",
        "AEON_CONTINUOUS_MODE_PATH",
        "AEON_FORCED_REQUEST_MODE",
        "AEON_HOME",
        "AEON_INSTANCE_INSTRUCTIONS_FILE",
        "AEON_INSTANCE_SKILLS_DIR",
        "AEON_LOCAL_SEARCH",
        "AEON_MAIN_ORCHESTRATOR",
        "AEON_MAX_DECISION_MODEL_CALLS",
        "AEON_MAX_SUPPORT_MODEL_CALLS",
        "AEON_MESSAGE_HISTORY",
        "AEON_NO_FILE_LOG",
        "AEON_OPEN_FILES_CONTEXT_CHARS",
        "AEON_OPENCODE_HOME",
        "AEON_OPENCODE_TURN_TIMEOUT_SECONDS",
        "AEON_PRESERVE_REASONING_HISTORY",
        "AEON_READ_ONLY",
        "AEON_READ_ONLY_PARALLELISM",
        "AEON_REASONING_EFFORT",
        "AEON_SEARXNG_PORT",
        "AEON_SKIP_VISION_SELFTEST",
        "AEON_SKILLS_DIR",
        "AEON_STATE_DIR",
        # Principal capabilities are paths/loopback endpoints materialized and
        # validated by Nexus.  Credential contents remain only in private files.
        "AEON_REMOTE_INSTANCE_ID",
        "NEXUS_INTERNAL_MCP_URL",
        "NEXUS_INTERNAL_ORCHESTRATOR_URL",
        "NEXUS_INTERNAL_SELF_SETTINGS_URL",
        "NEXUS_MCP_DELEGATION_ID",
        "NEXUS_MCP_DELEGATION_TOKEN_FILE",
        "NEXUS_ORCHESTRATOR_TOKEN_FILE",
        "NEXUS_SELF_SETTINGS_TOKEN_FILE",
        # Framework import guards used by the existing Aeon tool stack.
        "USE_FLAX",
        "USE_TF",
    }
)
_NO_ACCELERATOR_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "void",
    "GPU_DEVICE_ORDINAL": "-1",
    "HIP_VISIBLE_DEVICES": "-1",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "ROCR_VISIBLE_DEVICES": "-1",
}

# OpenCode v1.18.27 ships these generic tools in its own process.  Aeon must use
# the reviewed MCP bridge instead so every filesystem, web, mutation, and agent
# action receives the existing RequestContract/Fleet checks and typed receipt.
# Keep both the legacy ``tools`` switch and the permission rules: the former is
# retained for config compatibility while the latter fails closed if OpenCode's
# tool-filtering implementation changes.
_OPENCODE_BUILTIN_TOOLS = (
    "bash",
    "read",
    "glob",
    "grep",
    "edit",
    "write",
    "patch",
    "apply_patch",
    "webfetch",
    "websearch",
    "task",
    "todowrite",
    "skill",
    "question",
    "lsp",
    "plan_exit",
    "execute",
)

_HARNESS_INSTRUCTIONS = """
You are Aeon, the local-model agent inside Nexus. OpenCode supplies the agent
loop, context compaction, and duplicate-call protection. Work directly and
finish simple requests quickly.

Use the `aeon_*` MCP tools for commands, file mutation, web search, browser
interaction, vision, Fleet jobs, credentials, GitHub, and external services.
Those tools retain Nexus authorization checks and Fleet Compute routing.
OpenCode's generic shell, mutation, web-fetch, and subagent tools are disabled.
The stateful Aeon browser retains its persistent human-like profile and visual
observations; use it for websites instead of improvising HTTP automation. Never
claim a CAPTCHA or access-control challenge was bypassed: report a challenge
that requires the owner.

Prefer the smallest sufficient action sequence. Verify material changes, do not
repeat an unchanged failed tool call, and give a concise factual final response.
Treat tool receipts as authoritative; model prose cannot grant permission or
prove a mutation succeeded.
""".strip()


class OpenCodeConfigError(RuntimeError):
    """An isolated OpenCode runtime directory or file was unsafe."""


def _private_directory(path: Path, *, create: bool = True) -> Path:
    path = path.expanduser().absolute()
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise OpenCodeConfigError("OpenCode state directory is unavailable") from exc
    if (
        resolved != path
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise OpenCodeConfigError(
            "OpenCode state directory must be owner-only and contain no symbolic links"
        )
    return path


def _atomic_private_bytes(directory: Path, name: str, payload: bytes) -> Path:
    directory = _private_directory(directory)
    if not name or "/" in name or name in {".", ".."}:
        raise OpenCodeConfigError("OpenCode state filename is invalid")
    temp = f".{name}.tmp-{secrets.token_hex(12)}"
    directory_fd = os.open(
        directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temp, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as exc:
        raise OpenCodeConfigError("OpenCode state could not be published") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temp, dir_fd=directory_fd)
        except OSError:
            pass
        os.close(directory_fd)
    target = directory / name
    metadata = target.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise OpenCodeConfigError("OpenCode state file is not owner-private")
    return target


def materialize_authority(directory: Path, request: str) -> Path:
    content = str(request or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    encoded = content.encode("utf-8")
    if not content or len(encoded) > MAX_AUTHORITY_BYTES or "\x00" in content:
        raise OpenCodeConfigError("OpenCode request authority is invalid or too large")
    return _atomic_private_bytes(directory, "authority.txt", encoded + b"\n")


def materialize_instructions(directory: Path, *, instance_id: str | None) -> Path:
    try:
        layers = load_runtime_instructions(
            os.environ.get(RUNTIME_INSTRUCTIONS_ENV) or None,
            expected_instance_id=instance_id,
            expected_agent_kind="aeon" if instance_id else None,
        )
    except RuntimeInstructionError as exc:
        raise OpenCodeConfigError("OpenCode runtime instructions are invalid") from exc
    inherited = format_runtime_instruction_layers(layers)
    text = _HARNESS_INSTRUCTIONS
    if inherited:
        text += "\n\n" + inherited.strip()
    return _atomic_private_bytes(
        directory, "instructions.md", (text.rstrip() + "\n").encode("utf-8")
    )


def materialize_config(
    directory: Path,
    *,
    base_url: str,
    bearer_token: str,
    instruction_path: Path,
    max_steps: int = DEFAULT_OPENCODE_STEPS,
) -> Path:
    steps = int(max_steps)
    if not 1 <= steps <= MAX_OPENCODE_STEPS:
        raise OpenCodeConfigError("OpenCode step limit is outside the reviewed range")
    config = {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "model": "nexus-fleet/qwen",
        "small_model": "nexus-fleet/qwen",
        "instructions": [str(instruction_path)],
        "provider": {
            "nexus-fleet": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Nexus Fleet Local",
                "options": {
                    "baseURL": base_url,
                    "apiKey": bearer_token,
                    "timeout": 600000,
                },
                "models": {
                    "qwen": {
                        "name": "Aeon Qwen",
                        "attachment": True,
                        "tool_call": True,
                        "limit": {
                            "context": int(QWEN_CONTEXT_TOKENS),
                            "output": 32768,
                        },
                        "modalities": {
                            "input": ["text", "image"],
                            "output": ["text"],
                        },
                    }
                },
            }
        },
        "agent": {
            "aeon": {
                "description": "Nexus Aeon with guarded local tools",
                "mode": "primary",
                "steps": steps,
            }
        },
        "mcp": {
            "aeon": {
                "type": "local",
                "command": [sys.executable, "-m", "aeon.harnesses.opencode_mcp"],
                "enabled": True,
                "timeout": 120000,
            }
        },
        "tools": {name: False for name in _OPENCODE_BUILTIN_TOOLS},
        "permission": {
            # Unknown future OpenCode capabilities stay hidden.  Only tools from
            # the reviewed local MCP server are admitted; that server performs
            # its own per-request capability and collaborator-mode filtering.
            "*": "deny",
            "aeon_*": "allow",
            **{name: "deny" for name in _OPENCODE_BUILTIN_TOOLS},
            "external_directory": "deny",
        },
    }
    payload = (
        json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return _atomic_private_bytes(directory, "opencode.json", payload)


def isolated_environment(
    base: Mapping[str, str],
    *,
    directory: Path,
    config_path: Path,
    authority_path: Path,
    base_url: str,
    bearer_token: str,
    logical_model: str,
    wire_model: str,
) -> dict[str, str]:
    root = _private_directory(directory)
    config_home = _private_directory(root / "config")
    data_home = _private_directory(root / "data")
    cache_home = _private_directory(root / "cache")
    environment = {
        key: value
        for key, value in base.items()
        if key in _OPENCODE_CHILD_ENVIRONMENT
        and isinstance(key, str)
        and isinstance(value, str)
        and "\x00" not in key
        and "\x00" not in value
    }
    # Never accept Python/Bun/Node loader configuration or any OPENCODE_* value
    # from a caller.  The MCP import root and the complete OpenCode configuration
    # are rebuilt from the installed Aeon package and private files here.
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    environment.setdefault("HOME", str(Path.home()))
    environment.setdefault("PATH", os.defpath)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update(_NO_ACCELERATOR_ENVIRONMENT)
    environment.update(
        {
            "OPENCODE_CONFIG": str(config_path),
            "OPENCODE_CONFIG_DIR": str(config_home),
            "XDG_CONFIG_HOME": str(config_home),
            "XDG_DATA_HOME": str(data_home),
            "XDG_CACHE_HOME": str(cache_home),
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
            "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "AEON_OPENCODE_AUTHORITY_FILE": str(authority_path),
            "AEON_OPENCODE_PROXY_URL": base_url,
            "AEON_OPENCODE_PROXY_TOKEN": bearer_token,
            "AEON_OPENCODE_LOGICAL_MODEL": logical_model,
            "AEON_OPENCODE_WIRE_MODEL": wire_model,
        }
    )
    return environment


__all__ = (
    "DEFAULT_OPENCODE_STEPS",
    "MAX_OPENCODE_STEPS",
    "OpenCodeConfigError",
    "isolated_environment",
    "materialize_authority",
    "materialize_config",
    "materialize_instructions",
)
