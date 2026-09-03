"""Owner-private, version-pinned OpenCode configuration for one Aeon process."""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import stat
import sys
from dataclasses import dataclass
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
SHARED_RUNTIME_LAYOUT_VERSION = 1
_READ_ONLY_DIRECTORY_MODE = 0o500
_SHARED_RUNTIME_MARKER = "layout.json"
_SHARED_RUNTIME_LOCK = ".layout.lock"

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
        "AEON_BENCHMARK_SCENARIO_CAPABILITY",
        "AEON_BENCHMARK_TRACE_CASE_ID",
        "AEON_BENCHMARK_TRACE_NONCE",
        "AEON_BENCHMARK_TRACE_REPETITION",
        "AEON_BENCHMARK_TRACE_RUN_ID",
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


@dataclass(frozen=True)
class SharedOpenCodeRuntime:
    """Version-bound immutable paths reused by isolated OpenCode sessions."""

    root: Path
    config_home: Path
    cache_home: Path


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


def _owned_directory(path: Path, *, modes: frozenset[int]) -> os.stat_result:
    """Validate one exact owner directory without accepting a symbolic link."""

    exact = path.expanduser().absolute()
    try:
        metadata = exact.lstat()
        resolved = exact.resolve(strict=True)
    except OSError as exc:
        raise OpenCodeConfigError("OpenCode shared runtime is unavailable") from exc
    if (
        resolved != exact
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) not in modes
    ):
        raise OpenCodeConfigError("OpenCode shared runtime directory is unsafe")
    return metadata


def _shared_runtime_lock(root: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(root / _SHARED_RUNTIME_LOCK, flags, 0o600)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise OpenCodeConfigError("OpenCode shared runtime lock is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _build_shared_runtime(root: Path) -> None:
    """Finish only the known empty layout after an interrupted first build."""

    expected_children = {
        root: {_SHARED_RUNTIME_LOCK, _SHARED_RUNTIME_MARKER, "config", "cache"},
        root / "config": {"opencode"},
        root / "config" / "opencode": set(),
        root / "cache": {"opencode"},
        root / "cache" / "opencode": {"bin"},
        root / "cache" / "opencode" / "bin": set(),
    }
    directories = tuple(expected_children)[1:]
    # Reject contamination before changing any mode.  A missing marker means a
    # prior initializer may have stopped partway through, but it never grants
    # authority to adopt arbitrary files from that partial tree.
    for directory, allowed in expected_children.items():
        if not directory.exists():
            continue
        _owned_directory(
            directory,
            modes=(
                frozenset({0o700})
                if directory == root
                else frozenset({0o700, _READ_ONLY_DIRECTORY_MODE})
            ),
        )
        observed = {entry.name for entry in directory.iterdir()}
        if not observed <= allowed:
            raise OpenCodeConfigError(
                "OpenCode shared runtime contains unexpected content"
            )

    try:
        for directory in directories:
            parent = directory.parent
            if stat.S_IMODE(parent.stat().st_mode) == _READ_ONLY_DIRECTORY_MODE:
                parent.chmod(0o700)
            if directory.exists():
                _owned_directory(
                    directory,
                    modes=frozenset({0o700, _READ_ONLY_DIRECTORY_MODE}),
                )
                if stat.S_IMODE(directory.stat().st_mode) == _READ_ONLY_DIRECTORY_MODE:
                    directory.chmod(0o700)
            else:
                directory.mkdir(mode=0o700)

        for directory in reversed(directories):
            directory.chmod(_READ_ONLY_DIRECTORY_MODE)
        marker = {
            "policy": "pure-read-only",
            "schema_version": SHARED_RUNTIME_LAYOUT_VERSION,
        }
        _atomic_private_bytes(
            root,
            _SHARED_RUNTIME_MARKER,
            (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "ascii"
            ),
        )
    except BaseException:
        # A killed first initializer can be completed on the next call, but it
        # must not leave a known partial directory writable in the meantime.
        for directory in reversed(directories):
            try:
                metadata = directory.lstat()
                if (
                    stat.S_ISDIR(metadata.st_mode)
                    and not stat.S_ISLNK(metadata.st_mode)
                    and metadata.st_uid == os.geteuid()
                ):
                    directory.chmod(_READ_ONLY_DIRECTORY_MODE)
            except OSError:
                pass
        raise


def _validate_shared_runtime(root: Path) -> SharedOpenCodeRuntime:
    config_home = root / "config"
    global_config = config_home / "opencode"
    cache_home = root / "cache"
    cache_root = cache_home / "opencode"
    cache_bin = cache_root / "bin"
    expected = {
        root: {_SHARED_RUNTIME_LOCK, _SHARED_RUNTIME_MARKER, "config", "cache"},
        config_home: {"opencode"},
        global_config: set(),
        cache_home: {"opencode"},
        cache_root: {"bin"},
        cache_bin: set(),
    }
    _owned_directory(root, modes=frozenset({0o700}))
    for directory in tuple(expected)[1:]:
        _owned_directory(directory, modes=frozenset({_READ_ONLY_DIRECTORY_MODE}))
    for directory, names in expected.items():
        if {entry.name for entry in directory.iterdir()} != names:
            raise OpenCodeConfigError(
                "OpenCode shared runtime contains unexpected content"
            )
    marker_path = root / _SHARED_RUNTIME_MARKER
    descriptor = -1
    try:
        marker_metadata = marker_path.lstat()
        if (
            not stat.S_ISREG(marker_metadata.st_mode)
            or stat.S_ISLNK(marker_metadata.st_mode)
            or marker_metadata.st_uid != os.geteuid()
            or marker_metadata.st_nlink != 1
            or stat.S_IMODE(marker_metadata.st_mode) != 0o600
            or marker_metadata.st_size > 256
        ):
            raise OpenCodeConfigError("OpenCode shared runtime marker is invalid")
        descriptor = os.open(
            marker_path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_metadata = os.fstat(descriptor)
        if (
            opened_metadata.st_dev != marker_metadata.st_dev
            or opened_metadata.st_ino != marker_metadata.st_ino
        ):
            raise OpenCodeConfigError("OpenCode shared runtime marker changed")
        raw_marker = os.read(descriptor, 257)
        if len(raw_marker) != marker_metadata.st_size:
            raise OpenCodeConfigError("OpenCode shared runtime marker changed")
        marker = json.loads(raw_marker.decode("ascii"))
        if marker != {
            "policy": "pure-read-only",
            "schema_version": SHARED_RUNTIME_LAYOUT_VERSION,
        }:
            raise OpenCodeConfigError("OpenCode shared runtime marker is invalid")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpenCodeConfigError("OpenCode shared runtime marker is invalid") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return SharedOpenCodeRuntime(
        root=root,
        config_home=config_home,
        cache_home=cache_home,
    )


def materialize_shared_runtime(directory: Path) -> SharedOpenCodeRuntime:
    """Create and validate the empty read-only runtime cache used by one pin.

    OpenCode installs its JavaScript plugin SDK into every writable config
    directory, even in pure mode.  Aeon exposes no JavaScript plugins, so one
    version-bound, non-writable config/cache layout prevents that unnecessary
    per-session dependency tree while retaining isolated data and state.
    """

    root = _private_directory(directory)
    lock = _shared_runtime_lock(root)
    try:
        marker = root / _SHARED_RUNTIME_MARKER
        try:
            marker.lstat()
            marker_exists = True
        except FileNotFoundError:
            marker_exists = False
        except OSError as exc:
            raise OpenCodeConfigError(
                "OpenCode shared runtime marker is unavailable"
            ) from exc
        if not marker_exists:
            _build_shared_runtime(root)
        return _validate_shared_runtime(root)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)


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
    # OpenCode's MCP child inherits the parent environment unless its config
    # overrides it.  Give Aeon's tools writable, instance-private XDG roots so
    # a package manager or bounded child never receives the read-only shared
    # OpenCode cache as its ordinary application cache.
    tool_runtime = _private_directory(directory / "tool-runtime")
    tool_environment = {
        name: str(_private_directory(tool_runtime / leaf))
        for name, leaf in (
            ("XDG_CONFIG_HOME", "config"),
            ("XDG_DATA_HOME", "data"),
            ("XDG_CACHE_HOME", "cache"),
            ("XDG_STATE_HOME", "state"),
        )
    }
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
                "environment": tool_environment,
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
    shared_runtime_directory: Path,
    config_path: Path,
    authority_path: Path,
    base_url: str,
    bearer_token: str,
    logical_model: str,
    wire_model: str,
) -> dict[str, str]:
    root = _private_directory(directory)
    shared_runtime = materialize_shared_runtime(shared_runtime_directory)
    data_home = _private_directory(root / "data")
    state_home = _private_directory(root / "state")
    opencode_home = _private_directory(root / "home")
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
            # Both values resolve to this same directory in pinned OpenCode.
            # Keeping it read-only also makes the SDK install path a no-op.
            "OPENCODE_CONFIG_DIR": str(shared_runtime.config_home / "opencode"),
            "XDG_CONFIG_HOME": str(shared_runtime.config_home),
            "XDG_DATA_HOME": str(data_home),
            "XDG_CACHE_HOME": str(shared_runtime.cache_home),
            "XDG_STATE_HOME": str(state_home),
            # Pinned OpenCode separately scans its idea of $HOME for a
            # top-level .opencode directory even with project config disabled.
            # Preserve the real HOME for Aeon tools while giving OpenCode an
            # empty, case-private discovery root.
            "OPENCODE_TEST_HOME": str(opencode_home),
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
    "SHARED_RUNTIME_LAYOUT_VERSION",
    "SharedOpenCodeRuntime",
    "isolated_environment",
    "materialize_authority",
    "materialize_config",
    "materialize_instructions",
    "materialize_shared_runtime",
)
