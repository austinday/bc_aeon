"""Persistent tmux-backed Aeon instance management."""

from __future__ import annotations

import json
import os
import secrets
import re
import shlex
import socket
import stat
import subprocess
import threading
import time
import tomllib
import uuid
from contextlib import contextmanager
from pathlib import Path

import psutil

from aeon.core.runtime_instructions import (
    RUNTIME_INSTRUCTIONS_ENV,
    RuntimeInstructionError,
    format_runtime_instruction_layers,
    load_runtime_instructions,
    materialize_grok_agent_profile,
    materialize_provider_instruction_text,
    materialize_runtime_instructions,
    runtime_instruction_layers_from_snapshot,
)

from .agent_settings import AgentSettingsError, normalize_settings, public_catalog
from .project_manager import (
    PROJECT_MANAGER_OBJECTIVE,
    ProjectManagerError,
    ProjectManagerProtectedError,
    dormant_project_manager_status,
    ensure_project_manager,
    is_first_project_manager_activation,
    is_project_manager_record,
    project_manager_public_flags,
    reject_project_manager_deletion,
)
from .instruction_profiles import InstructionProfileError
from .providers import (
    PROVIDER_IDS,
    ProviderError,
    provider_agent_command,
    provider_connect_command,
    provider_status,
    subscription_environment,
)


NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,63}$")
WORKSPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
PROVIDER_AUTH_KINDS = {
    f"{provider_id}_auth": provider_id for provider_id in PROVIDER_IDS
}
INSTANCE_KINDS = frozenset({"aeon", "terminal", *PROVIDER_IDS, *PROVIDER_AUTH_KINDS})
AGENT_INSTANCE_KINDS = frozenset({"aeon", *PROVIDER_IDS})
LOCAL_TERMINAL_HOST_ID = "192.168.0.177"
LOCAL_TERMINAL_HOSTNAME = "DAY2RTX6000PRO"
MANAGED_SHELL_BINARY = "/bin/bash"
MANAGED_SHELL_RC_FILENAME = "managed-shell.rc"
MANAGED_SHELL_IDENTITY_FILENAME = "managed-shell.identity"
MANAGED_SHELL_READY_FILENAME = "managed-shell.ready"
MANAGED_AGENT_IDENTITY_FILENAME = "managed-agent.identity.json"
MANAGED_AGENT_PENDING_FILENAME = "managed-agent.pending.json"
SHELL_PROMPT_REFRESH_TIMEOUT_SECONDS = 0.75
AGENT_START_TIMEOUT_SECONDS = 3.0
AGENT_END_TIMEOUT_SECONDS = 5.0
AGENT_SECOND_INTERRUPT_DELAY_SECONDS = 0.75
TERMINAL_RETURN_TIMEOUT_SECONDS = 0.75
ENV_BINARY = "/usr/bin/env"
TMUX_PANE_TERM = "tmux-256color"
BROWSER_TERMINAL_TERM = "xterm-256color"
TRUECOLOR_TERM = "truecolor"
CODEX_PROJECT_CONFIG_MAX_BYTES = 256 * 1024
FORCE_STOP_REQUIRED_PREFIX = "Safety ambiguity; exact-name force stop required: "
TMUX_NO_SERVER_RE = re.compile(
    r"^(?:no server running on [^\r\n]+|"
    r"error connecting to [^\r\n]+ \((?:No such file or directory|Connection refused)\))$"
)
# tmux renders control bytes in ``-F`` output with vis-style octal escapes (for
# example, an ASCII unit separator becomes the four printable bytes ``\037``).
# Use an explicit printable sentinel so the requested format and returned bytes
# agree.  A collision in a field creates the wrong field count and fails closed.
_TMUX_PANE_FIELD_SEPARATOR = "__AEON_REMOTE_PANE_FIELD_6B1E__"
_TMUX_PANE_FORMAT = _TMUX_PANE_FIELD_SEPARATOR.join(
    (
        "#{pane_dead}",
        "#{pane_pid}",
        "#{pane_dead_status}",
        "#{pane_current_command}",
    )
)


def _force_stop_required_error(detail: str) -> str:
    return f"{FORCE_STOP_REQUIRED_PREFIX}{detail}"[:500]


def _has_force_stop_required_error(record: dict) -> bool:
    return str(record.get("last_error") or "").startswith(
        FORCE_STOP_REQUIRED_PREFIX
    )


def _tmux_proves_no_server(result: subprocess.CompletedProcess) -> bool:
    return bool(
        result.returncode == 1
        and not result.stdout
        and TMUX_NO_SERVER_RE.fullmatch(str(result.stderr or "").strip())
    )


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _private_instance_directory(config, instance_id: str) -> Path:
    """Return one owner-only, non-symlinked instance state directory."""

    if not re.fullmatch(r"[0-9a-f]{32}", instance_id):
        raise InstanceError("The managed shell identity is invalid")
    parent = Path(config.instance_state_dir)
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise InstanceError("The managed shell state directory is unavailable") from exc
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
        or parent.resolve(strict=True) != parent.absolute()
    ):
        raise InstanceError("The managed shell state directory is not private")

    directory = parent / instance_id
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise InstanceError("Could not create managed shell state") from exc
    try:
        metadata = directory.lstat()
    except OSError as exc:
        raise InstanceError("Managed shell state is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or directory.resolve(strict=True) != directory.absolute()
    ):
        raise InstanceError("Managed shell state is not safely owned")
    return directory


def _publish_private_file(directory: Path, filename: str, payload: bytes) -> Path:
    """Atomically publish one mode-600 regular file in a validated directory."""

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(directory, directory_flags)
    except OSError as exc:
        raise InstanceError("Managed shell state is unavailable") from exc
    temporary_name = f".{filename}.{uuid.uuid4().hex}.tmp"
    temporary_fd = None
    try:
        try:
            existing = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode)
            or existing.st_uid != os.geteuid()
            or stat.S_IMODE(existing.st_mode) != 0o600
            or existing.st_nlink != 1
        ):
            raise InstanceError("Existing managed shell state is not private")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        temporary_fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
        os.fchmod(temporary_fd, 0o600)
        _write_all(temporary_fd, payload)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except InstanceError:
        raise
    except OSError as exc:
        raise InstanceError("Could not publish managed shell state") from exc
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except (FileNotFoundError, OSError):
            pass
        os.close(directory_fd)
    return directory / filename


def _read_private_file(
    directory: Path, filename: str, *, maximum_bytes: int
) -> bytes | None:
    """Read a small owner-only regular file without following a symlink."""

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(directory, directory_flags)
    except OSError:
        return None
    fd = None
    try:
        try:
            fd = os.open(filename, file_flags, dir_fd=directory_fd)
        except OSError:
            return None
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size < 1
            or metadata.st_size > maximum_bytes
        ):
            return None
        chunks = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > maximum_bytes:
            return None
        return payload
    except OSError:
        return None
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory_fd)


def _remove_private_file(
    directory: Path, filename: str, *, missing_ok: bool = True
) -> bool:
    """Remove one exact private regular file, failing closed on unsafe state."""

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(directory, directory_flags)
    except OSError:
        return False
    try:
        try:
            metadata = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return missing_ok
        except OSError:
            return False
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            return False
        try:
            os.unlink(filename, dir_fd=directory_fd)
        except OSError:
            return False
        return True
    finally:
        os.close(directory_fd)


def _proc_process_info(pid: int) -> dict[str, int] | None:
    """Return Linux job-control identity without trusting a process name."""

    if not isinstance(pid, int) or pid <= 1:
        return None
    proc = Path("/proc") / str(pid)
    try:
        metadata = proc.stat()
        raw = (proc / "stat").read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return None
    if metadata.st_uid != os.geteuid():
        return None
    # /proc/PID/stat's comm field is parenthesized and may itself contain spaces
    # or ')'. Split after the final closing parenthesis; the tail starts at field
    # 3 (state), making pgrp/session/tty/tpgid indices 2/3/4/5 and starttime 19.
    closing = raw.rfind(")")
    if closing < 0 or closing + 2 >= len(raw):
        return None
    fields = raw[closing + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        pgrp = int(fields[2])
        session = int(fields[3])
        tty = int(fields[4])
        tpgid = int(fields[5])
        start_ticks = int(fields[19])
        actual_pgrp = os.getpgid(pid)
        actual_session = os.getsid(pid)
    except (OSError, TypeError, ValueError):
        return None
    if actual_pgrp != pgrp or actual_session != session:
        return None
    return {
        "pid": pid,
        "pgrp": pgrp,
        "session": session,
        "tty": tty,
        "tpgid": tpgid,
        "start_ticks": start_ticks,
    }


def _managed_shell_environment() -> dict[str, str]:
    """Build a small non-secret environment for the user's managed shell."""

    environment = dict(os.environ)
    # The three reviewed provider filters share the runtime allowlist and each
    # removes its own credential override variables. Applying all three strips
    # every known API credential as well as unrelated Nexus/service secrets.
    for provider_id in sorted(PROVIDER_IDS):
        environment = subscription_environment(provider_id, environment)
    environment.setdefault("HOME", str(Path.home()))
    environment.setdefault("PATH", os.defpath)
    environment.setdefault("SHELL", MANAGED_SHELL_BINARY)
    # Programs run *inside* tmux and must use tmux's actual capability entry.
    # The external browser attach client uses xterm-256color separately below.
    environment["TERM"] = TMUX_PANE_TERM
    environment["COLORTERM"] = TRUECOLOR_TERM
    return environment


def _materialize_managed_shell(record: dict, config) -> list[str]:
    """Create a fresh prompt identity and return the fixed interactive Bash argv."""

    directory = _private_instance_directory(config, record["id"])
    marker = directory / MANAGED_SHELL_READY_FILENAME
    nonce = secrets.token_hex(32)
    identity = _publish_private_file(
        directory,
        MANAGED_SHELL_IDENTITY_FILENAME,
        f"{nonce}\n".encode("ascii"),
    )
    # Never accept a readiness file from a prior shell/PID, even if a PID was
    # later reused. The new per-launch nonce is persisted privately for service
    # restarts and is also baked into this one fixed rcfile.
    try:
        marker.unlink(missing_ok=True)
    except OSError as exc:
        raise InstanceError("Could not clear stale managed shell readiness") from exc
    if not _remove_private_file(directory, MANAGED_AGENT_IDENTITY_FILENAME):
        raise InstanceError("Existing managed agent identity is not private")
    if not _remove_private_file(directory, MANAGED_AGENT_PENDING_FILENAME):
        raise InstanceError("Existing managed agent activation state is not private")

    quoted_marker = shlex.quote(str(marker))
    quoted_nonce = shlex.quote(nonce)
    rc_payload = f"""# Managed by Nexus. This file contains no credentials.
set +o functrace
# Match an ordinary interactive Bash without executing user-controlled startup
# files.  The fixed system dircolors database gives GNU ls its standard file
# type palette; the aliases keep columns and color decisions owned by ls.
if [[ -x /usr/bin/dircolors ]]; then
    eval "$(/usr/bin/dircolors --sh)"
fi
alias ls='/usr/bin/ls --color=auto'
alias ll='/usr/bin/ls -alF --color=auto'
alias la='/usr/bin/ls -A --color=auto'
readonly __nexus_shell_ready_marker={quoted_marker}
readonly __nexus_shell_nonce={quoted_nonce}
__nexus_shell_clear_ready() {{
    command rm -f -- \"$__nexus_shell_ready_marker\"
}}
__nexus_shell_mark_ready() {{
    local __nexus_shell_tmp=\"${{__nexus_shell_ready_marker}}.tmp.${{BASHPID}}\"
    local __nexus_shell_pid=\"$BASHPID\"
    command rm -f -- \"$__nexus_shell_tmp\"
    ( umask 077; command printf '%s %s\\n' \"$__nexus_shell_nonce\" \"$__nexus_shell_pid\" >\"$__nexus_shell_tmp\" ) &&
        command chmod 600 \"$__nexus_shell_tmp\" &&
        command mv -f -- \"$__nexus_shell_tmp\" \"$__nexus_shell_ready_marker\"
}}
readonly -f __nexus_shell_clear_ready __nexus_shell_mark_ready
trap '__nexus_shell_clear_ready' DEBUG
PROMPT_COMMAND=__nexus_shell_mark_ready
PS1='\\u@\\h:\\w\\$ '
""".encode("utf-8")
    rcfile = _publish_private_file(directory, MANAGED_SHELL_RC_FILENAME, rc_payload)
    # Keep this local binding explicit: both files are required by the verifier,
    # and assigning the identity path prevents accidental removal as dead code.
    if identity.parent != rcfile.parent:  # pragma: no cover - construction invariant
        raise InstanceError("Managed shell identity publication failed")

    clean_environment = _managed_shell_environment()
    command = [ENV_BINARY, "-i"]
    command.extend(
        f"{key}={value}" for key, value in sorted(clean_environment.items())
    )
    command.extend(
        [
            MANAGED_SHELL_BINARY,
            "--noprofile",
            "--rcfile",
            str(rcfile),
            "-i",
        ]
    )
    return command


def _read_codex_project_config(directory: Path) -> dict | None:
    """Read one project config without following links or accepting races.

    Codex gives project config a higher precedence than a selected profile.  A
    confidential Nexus instruction body therefore cannot be made authoritative
    when a project layer defines the same scalar.  The preflight reads only
    bounded regular files and fails closed when an extant layer is ambiguous.
    """

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise InstanceError("Codex project configuration could not be inspected") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise InstanceError("Codex project configuration directory is symbolic")
    if not stat.S_ISDIR(metadata.st_mode):
        return None
    try:
        directory_fd = os.open(directory, flags)
    except OSError as exc:
        raise InstanceError("Codex project configuration is ambiguous") from exc
    file_fd = None
    try:
        opened_directory = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(opened_directory.st_mode)
            or opened_directory.st_dev != metadata.st_dev
            or opened_directory.st_ino != metadata.st_ino
        ):
            raise InstanceError("Codex project configuration changed during inspection")
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            expected_file = os.stat(
                "config.toml", dir_fd=directory_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise InstanceError("Codex project config.toml is ambiguous") from exc
        if not stat.S_ISREG(expected_file.st_mode):
            raise InstanceError("Codex project config.toml is not a regular file")
        try:
            file_fd = os.open("config.toml", file_flags, dir_fd=directory_fd)
        except FileNotFoundError:
            # The no-file case was already resolved by the preceding stat.
            # Disappearance now is a race, not a safely absent layer.
            raise InstanceError(
                "Codex project config.toml changed during inspection"
            )
        except OSError as exc:
            raise InstanceError("Codex project config.toml is ambiguous") from exc
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_dev != expected_file.st_dev
            or before.st_ino != expected_file.st_ino
        ):
            raise InstanceError("Codex project config.toml is not a regular file")
        if before.st_size > CODEX_PROJECT_CONFIG_MAX_BYTES:
            raise InstanceError("Codex project config.toml is too large to inspect safely")
        chunks = []
        remaining = CODEX_PROJECT_CONFIG_MAX_BYTES + 1
        while remaining:
            chunk = os.read(file_fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > CODEX_PROJECT_CONFIG_MAX_BYTES:
            raise InstanceError("Codex project config.toml is too large to inspect safely")
        after = os.fstat(file_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or len(payload) != after.st_size:
            raise InstanceError("Codex project config.toml changed during inspection")
        try:
            value = tomllib.loads(payload.decode("utf-8"))
        except (UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise InstanceError("Codex project config.toml could not be parsed safely") from exc
        if not isinstance(value, dict):  # pragma: no cover - tomllib contract
            raise InstanceError("Codex project config.toml is invalid")
        return value
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def _reject_codex_developer_instruction_override(
    *, workspace: Path, codex_home: Path
) -> None:
    """Reject higher-precedence project developer instructions.

    Scan every canonical ancestor because Codex project-root markers are
    user-configurable.  The actual CODEX_HOME directory is the user layer, not a
    project layer, and the selected Nexus profile intentionally overrides it.
    """

    try:
        resolved_workspace = workspace.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InstanceError("Codex workspace is unavailable") from exc
    if not resolved_workspace.is_dir():
        raise InstanceError("Codex workspace is not a directory")
    for ancestor in (resolved_workspace, *resolved_workspace.parents):
        directory = ancestor / ".codex"
        if directory == codex_home:
            continue
        value = _read_codex_project_config(directory)
        if value is not None and "developer_instructions" in value:
            raise InstanceError(
                "Codex project config.toml defines developer_instructions; "
                "remove that conflicting project override before starting this "
                "persistent agent"
            )


def _materialize_codex_profile(
    *,
    instance_id: str,
    instructions: str | None,
    workspace: Path,
    environment: dict[str, str],
) -> str:
    """Publish one private Codex config layer and return its opaque profile name.

    Codex officially accepts ``developer_instructions`` only as a config value,
    while its ``--profile`` flag loads a named file from CODEX_HOME.  The prompt
    therefore stays in a mode-600 file and process argv contains only a random-
    looking instance-derived profile name.
    """

    if not re.fullmatch(r"[0-9a-f]{32}", instance_id):
        raise InstanceError("The Codex instruction profile identity is invalid")
    if instructions is not None and (
        not isinstance(instructions, str) or "\x00" in instructions
    ):
        raise InstanceError("The Codex instruction profile is invalid")
    if instructions is not None and len(instructions.encode("utf-8")) > 128 * 1024:
        raise InstanceError("The Codex instruction profile is too large")
    home = environment.get("HOME")
    configured = environment.get("CODEX_HOME")
    if not configured and not home:
        raise InstanceError("Codex has no private configuration home")
    raw_directory = Path(configured or str(Path(home) / ".codex")).expanduser()
    absolute = Path(os.path.abspath(raw_directory))
    try:
        directory = absolute.resolve(strict=True)
        metadata = directory.stat()
    except (OSError, RuntimeError) as exc:
        raise InstanceError("Codex configuration home is unavailable") from exc
    if (
        directory != absolute
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise InstanceError("Codex configuration home is not safely owned")

    _reject_codex_developer_instruction_override(
        workspace=workspace,
        codex_home=directory,
    )

    profile_name = f"nexus-{instance_id}"
    filename = f"{profile_name}.config.toml"
    lines = ["# Managed by Nexus for one local agent instance.\n"]
    if instructions is not None:
        # JSON basic strings and TOML basic strings share these escapes, except
        # JSON may emit DEL literally when non-ASCII characters are preserved.
        # TOML forbids raw DEL but accepts its Unicode escape.  Keeping other
        # Unicode literal avoids invalid JSON surrogate-pair escapes in TOML.
        encoded_instructions = json.dumps(
            instructions, ensure_ascii=False
        ).replace("\x7f", "\\u007f")
        lines.append(
            "developer_instructions = "
            f"{encoded_instructions}\n"
        )
    payload = "".join(lines).encode("utf-8")
    try:
        parsed_profile = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:  # pragma: no cover - defensive
        raise InstanceError("The Codex instruction profile could not be encoded") from exc
    if instructions is not None and parsed_profile.get("developer_instructions") != instructions:
        raise InstanceError("The Codex instruction profile could not be encoded")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(directory, directory_flags)
    except OSError as exc:
        raise InstanceError("Codex configuration home is unavailable") from exc
    temporary_name = f".{filename}.{uuid.uuid4().hex}.tmp"
    temporary_fd = None
    try:
        opened_directory = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(opened_directory.st_mode)
            or opened_directory.st_dev != metadata.st_dev
            or opened_directory.st_ino != metadata.st_ino
            or opened_directory.st_uid != os.geteuid()
            or opened_directory.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise InstanceError("Codex configuration home changed during publication")
        try:
            existing = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode)
            or existing.st_uid != os.geteuid()
            or stat.S_IMODE(existing.st_mode) != 0o600
            or existing.st_nlink != 1
        ):
            raise InstanceError("Existing Codex instance profile is not private")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        temporary_fd = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
        os.fchmod(temporary_fd, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(temporary_fd, view)
            if written <= 0:
                raise InstanceError("Could not write the Codex instance profile")
            view = view[written:]
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except InstanceError:
        raise
    except OSError as exc:
        raise InstanceError("Could not publish the Codex instance profile") from exc
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        os.close(directory_fd)
    return profile_name


class InstanceError(Exception):
    pass


class InstanceLaunchError(InstanceError):
    def __init__(self, message: str, *, launched: bool):
        super().__init__(message)
        self.launched = launched


class InstanceManager:
    def __init__(
        self,
        store,
        config,
        *,
        command_runner=None,
        instruction_service=None,
        pane_prompt_checker=None,
        pane_foreground_checker=None,
    ):
        self.store = store
        self.config = config
        self._run = command_runner or subprocess.run
        self.instruction_service = instruction_service
        self._gpu_lock = threading.Lock()
        self._gpu_cache = (0.0, [])
        self._bootstrap_lock = threading.Lock()
        self._bootstrapped = False
        self._project_manager_lock = threading.Lock()
        self._terminal_creation_lock = threading.Lock()
        self._mode_locks_guard = threading.Lock()
        self._mode_locks: dict[str, threading.RLock] = {}
        self._transitioning_ids: set[str] = set()
        self._transition_generations: dict[str, int] = {}
        # tmux copy mode is pane state, not attach-client state. Remember panes
        # scrolled by the browser so the next real keystroke can return to the
        # live bottom before it is delivered to the shell or agent.
        self._browser_scrolled_ids: set[str] = set()
        self._pane_prompt_checker = pane_prompt_checker
        self._pane_foreground_checker = pane_foreground_checker
        # Test/integration probes can model tmux without real /proc PIDs. The
        # production path never uses this set; it persists a strict PGID record.
        self._injected_agent_foregrounds: set[str] = set()
        self._project_manager_enabled = any(
            Path("/home/aday").resolve().is_relative_to(root.resolve())
            for root in getattr(self.config, "allowed_roots", ())
        )

    def bootstrap(self) -> None:
        """Perform controller-owned registry initialization exactly once."""

        with self._bootstrap_lock:
            if self._bootstrapped:
                return
            if self._project_manager_enabled:
                try:
                    record, created = ensure_project_manager(
                        self.store, default_model=self.config.default_model
                    )
                except ProjectManagerError as exc:
                    raise InstanceError(str(exc)) from exc
                if created:
                    self.store.audit(
                        "project_manager_materialized",
                        actor="nexus",
                        instance_id=record["id"],
                        details={"workspace": record["workspace"]},
                    )
            self._bootstrapped = True

    def _normalize_dormant_project_manager(self, record: dict) -> dict:
        """Migrate the old direct-Aeon placeholder to its terminal base mode.

        Never rewrite a live pane: a Project Manager started by the prior
        release remains controllable until it exits. Once no process exists,
        the stable tab becomes a shell-backed Home terminal as requested.
        """

        if not is_project_manager_record(record) or int(record.get("shell_backed") or 0):
            return record
        pane = self._pane_info(record["tmux_name"])
        if pane is not None and not pane["dead"]:
            return record
        previous_kind = record.get("kind") or "aeon"
        self.store.update_instance(
            record["id"],
            kind="terminal",
            shell_backed=1,
            last_agent_kind=(
                previous_kind if previous_kind in AGENT_INSTANCE_KINDS else "aeon"
            ),
            status="idle",
            desired_state="stopped",
            last_error="",
        )
        return self.store.get_instance(record["id"])

    @staticmethod
    def _session_target(tmux_name: str) -> str:
        # '=' forces an exact tmux target match rather than a prefix match.
        return f"={tmux_name}"

    @staticmethod
    def _pane_target(tmux_name: str) -> str:
        # Pane commands need the explicit current-window suffix. A bare '=name'
        # is a valid session target but may resolve to an empty client target for
        # display-message instead of the session's pane.
        return f"={tmux_name}:"

    def _mode_lock(self, instance_id: str) -> threading.RLock:
        """Return the stable lifecycle lock for one durable browser tab."""

        with self._mode_locks_guard:
            return self._mode_locks.setdefault(instance_id, threading.RLock())

    @contextmanager
    def _lifecycle_lock(self, instance_id: str):
        """Serialize a mutation and publish a generation visible to input writers."""

        lock = self._mode_lock(instance_id)
        lock.acquire()
        with self._mode_locks_guard:
            self._transition_generations[instance_id] = (
                self._transition_generations.get(instance_id, 0) + 1
            )
            self._transitioning_ids.add(instance_id)
        try:
            yield
        finally:
            with self._mode_locks_guard:
                self._transitioning_ids.discard(instance_id)
            lock.release()

    @contextmanager
    def terminal_input_guard(self, instance_id: str):
        """Yield whether one WebSocket input write may proceed immediately.

        Lifecycle transitions hold the same lock while verifying the prompt and
        delivering fixed control/argv input. Browser input is dropped rather
        than blocking the asyncio loop or landing between those operations.
        """

        lock = self._mode_lock(instance_id)
        with self._mode_locks_guard:
            transitioning = instance_id in self._transitioning_ids
        acquired = False if transitioning else lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                lock.release()

    def _shell_directory(self, record: dict) -> Path | None:
        try:
            return _private_instance_directory(self.config, record["id"])
        except InstanceError:
            return None

    def _shell_nonce(self, record: dict) -> str | None:
        directory = self._shell_directory(record)
        if directory is None:
            return None
        payload = _read_private_file(
            directory, MANAGED_SHELL_IDENTITY_FILENAME, maximum_bytes=80
        )
        if payload is None:
            return None
        try:
            value = payload.decode("ascii")
        except UnicodeError:
            return None
        if not re.fullmatch(r"[0-9a-f]{64}\n", value):
            return None
        return value[:-1]

    def _base_shell_process(self, record: dict, pane: dict) -> dict | None:
        """Validate the exact server-launched shell process and job-control TTY."""

        if (
            int(record.get("shell_backed") or 0) != 1
            or pane is None
            or pane.get("dead")
            or not isinstance(pane.get("pid"), int)
        ):
            return None
        directory = self._shell_directory(record)
        nonce = self._shell_nonce(record)
        if directory is None or nonce is None:
            return None
        pid = pane["pid"]
        info = _proc_process_info(pid)
        if (
            info is None
            or info["pgrp"] != pid
            or info["session"] <= 1
            or info["tty"] == 0
        ):
            return None
        try:
            executable = (Path("/proc") / str(pid) / "exe").resolve(strict=True)
            argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        except OSError:
            return None
        if argv and argv[-1] == b"":
            argv.pop()
        expected = [
            MANAGED_SHELL_BINARY,
            "--noprofile",
            "--rcfile",
            str(directory / MANAGED_SHELL_RC_FILENAME),
            "-i",
        ]
        try:
            rendered = [value.decode("utf-8") for value in argv]
        except UnicodeError:
            return None
        if executable != Path(MANAGED_SHELL_BINARY).resolve() or rendered != expected:
            return None
        info["nonce"] = nonce
        return info

    def _pane_at_base_prompt(self, record: dict, pane: dict | None) -> bool:
        """Prove that the pane is at this launch's outer managed Bash prompt."""

        if pane is None or pane.get("dead"):
            return False
        if pane.get("command") != "bash":
            return False
        if self._pane_prompt_checker is not None:
            try:
                return bool(self._pane_prompt_checker(record, pane))
            except Exception:
                return False
        shell = self._base_shell_process(record, pane)
        if shell is None or shell["tpgid"] != shell["pgrp"]:
            return False
        directory = self._shell_directory(record)
        if directory is None:
            return False
        payload = _read_private_file(
            directory, MANAGED_SHELL_READY_FILENAME, maximum_bytes=128
        )
        if payload is None:
            return False
        try:
            marker = payload.decode("ascii")
        except UnicodeError:
            return False
        return marker == f"{shell['nonce']} {pane['pid']}\n"

    def _pane_foreground_job(self, record: dict, pane: dict | None) -> dict | None:
        """Return the exact foreground job identity beneath the managed shell."""

        if pane is None or pane.get("dead"):
            return None
        if self._pane_foreground_checker is not None:
            try:
                if self._pane_foreground_checker(record, pane):
                    return {"injected": 1}
            except Exception:
                pass
            return None
        shell = self._base_shell_process(record, pane)
        if (
            shell is None
            or shell["tpgid"] <= 1
            or shell["tpgid"] == shell["pgrp"]
        ):
            return None
        leader = _proc_process_info(shell["tpgid"])
        if (
            leader is None
            or leader["pid"] != leader["pgrp"]
            or leader["session"] != shell["session"]
            or leader["tty"] != shell["tty"]
            or leader["pgrp"] != shell["tpgid"]
        ):
            return None
        leader["shell_pid"] = shell["pid"]
        leader["shell_nonce"] = shell["nonce"]
        return leader

    def _record_managed_agent_foreground(self, record: dict, pane: dict) -> bool:
        foreground = self._pane_foreground_job(record, pane)
        if foreground is None:
            return False
        if foreground.get("injected"):
            self._injected_agent_foregrounds.add(record["id"])
            return True
        directory = self._shell_directory(record)
        if directory is None:
            return False
        payload = json.dumps(
            {
                key: foreground[key]
                for key in (
                    "shell_pid",
                    "shell_nonce",
                    "pid",
                    "pgrp",
                    "session",
                    "tty",
                    "start_ticks",
                )
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        try:
            _publish_private_file(
                directory, MANAGED_AGENT_IDENTITY_FILENAME, payload
            )
        except InstanceError:
            return False
        return True

    def _managed_agent_is_foreground(self, record: dict, pane: dict | None) -> bool:
        if self._pane_foreground_checker is not None:
            return (
                record["id"] in self._injected_agent_foregrounds
                and self._pane_foreground_job(record, pane) is not None
            )
        foreground = self._pane_foreground_job(record, pane)
        directory = self._shell_directory(record)
        if foreground is None or directory is None:
            return False
        payload = _read_private_file(
            directory, MANAGED_AGENT_IDENTITY_FILENAME, maximum_bytes=1024
        )
        if payload is None:
            return False
        try:
            identity = json.loads(payload)
        except (UnicodeError, ValueError, TypeError):
            return False
        if not isinstance(identity, dict) or set(identity) != {
            "shell_pid",
            "shell_nonce",
            "pid",
            "pgrp",
            "session",
            "tty",
            "start_ticks",
        }:
            return False
        return all(identity.get(key) == foreground.get(key) for key in identity)

    def _clear_shell_prompt_marker(self, record: dict) -> bool:
        if self._pane_prompt_checker is not None:
            return True
        directory = self._shell_directory(record)
        return bool(
            directory
            and _remove_private_file(
                directory,
                MANAGED_SHELL_READY_FILENAME,
                missing_ok=False,
            )
        )

    def _clear_managed_agent_identity(self, record: dict) -> bool:
        self._injected_agent_foregrounds.discard(record["id"])
        directory = self._shell_directory(record)
        return bool(
            directory
            and _remove_private_file(directory, MANAGED_AGENT_IDENTITY_FILENAME)
        )

    def _write_pending_activation(
        self,
        record: dict,
        *,
        target_kind: str,
        workspace: str,
        previous_agent_kind: str | None,
        agent_model: str | None = None,
        agent_effort: str | None = None,
        phase: str,
    ) -> None:
        if target_kind not in AGENT_INSTANCE_KINDS or phase not in {
            "prepared",
            "command_sent",
        }:
            raise InstanceError("Managed agent activation state is invalid")
        directory = self._shell_directory(record)
        if directory is None:
            raise InstanceError("Managed agent activation state is unavailable")
        payload = json.dumps(
            {
                "target_kind": target_kind,
                "workspace": workspace,
                "previous_agent_kind": previous_agent_kind,
                "clear_profile": bool(
                    previous_agent_kind and previous_agent_kind != target_kind
                ),
                "agent_model": agent_model,
                "agent_effort": agent_effort,
                "phase": phase,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _publish_private_file(directory, MANAGED_AGENT_PENDING_FILENAME, payload)

    def _read_pending_activation(self, record: dict) -> dict | None:
        directory = self._shell_directory(record)
        if directory is None:
            return None
        payload = _read_private_file(
            directory, MANAGED_AGENT_PENDING_FILENAME, maximum_bytes=8192
        )
        if payload is None:
            return None
        try:
            value = json.loads(payload)
        except (UnicodeError, ValueError, TypeError):
            return None
        if not isinstance(value, dict):
            return None
        legacy_keys = {
            "target_kind",
            "workspace",
            "previous_agent_kind",
            "clear_profile",
            "phase",
        }
        current_keys = {*legacy_keys, "agent_model", "agent_effort"}
        if set(value) == legacy_keys:
            # A process launched before model/effort persistence has no
            # trustworthy settings snapshot. Preserve lifecycle recovery but
            # do not invent an applied value for it.
            value["agent_model"] = None
            value["agent_effort"] = None
        elif set(value) != current_keys:
            return None
        if (
            value.get("target_kind") not in AGENT_INSTANCE_KINDS
            or value.get("phase") not in {"prepared", "command_sent"}
            or not isinstance(value.get("workspace"), str)
            or not isinstance(value.get("clear_profile"), bool)
            or (
                value.get("previous_agent_kind") is not None
                and value.get("previous_agent_kind") not in AGENT_INSTANCE_KINDS
            )
        ):
            return None
        try:
            workspace = self.validate_workspace(value["workspace"])
        except InstanceError:
            return None
        value["workspace"] = str(workspace)
        if (value["agent_model"] is None) != (value["agent_effort"] is None):
            return None
        if value["agent_model"] is not None:
            try:
                model, effort = normalize_settings(
                    value["target_kind"],
                    model=value["agent_model"],
                    effort=value["agent_effort"],
                )
            except (AgentSettingsError, ValueError):
                return None
            value["agent_model"] = model
            value["agent_effort"] = effort
        return value

    def _clear_pending_activation(self, record: dict) -> bool:
        directory = self._shell_directory(record)
        return bool(
            directory
            and _remove_private_file(directory, MANAGED_AGENT_PENDING_FILENAME)
        )

    def _mark_pending_agent_settings_applied(self, record: dict, pending: dict) -> None:
        """Idempotently commit the exact allowlisted snapshot used at launch."""

        if pending.get("agent_model") is None:
            # Legacy journals predate settings snapshots. Never infer which
            # provider default an already-running process received.
            return
        self.store.mark_agent_setting_applied(
            record["id"],
            pending["target_kind"],
            model=pending["agent_model"],
            effort=pending["agent_effort"],
        )

    def _base_shell_has_foreground_control(
        self, record: dict, pane: dict | None
    ) -> bool:
        """Prove only that the exact base shell owns its foreground pgrp."""

        if pane is None or pane.get("dead"):
            return False
        if self._pane_prompt_checker is not None:
            try:
                return bool(self._pane_prompt_checker(record, pane))
            except Exception:
                return False
        shell = self._base_shell_process(record, pane)
        return bool(shell and shell["tpgid"] == shell["pgrp"])

    def _recover_pending_activation(self, record: dict, pane: dict | None) -> dict:
        """Recover an activation interrupted between command send and DB mode commit."""

        directory = self._shell_directory(record)
        if directory is None:
            return record
        pending_path = directory / MANAGED_AGENT_PENDING_FILENAME
        try:
            pending_path.lstat()
            pending_exists = True
        except OSError:
            pending_exists = False
        if not pending_exists:
            # ``starting`` is written only after the durable activation journal.
            # If that journal vanished, never infer that an arbitrary live
            # foreground is an ordinary terminal. An exact prompt can be
            # recovered safely; every other live shape needs explicit force.
            if (
                (record.get("kind") or "aeon") == "terminal"
                and int(record.get("shell_backed") or 0) == 1
                and record.get("status") == "starting"
            ):
                if pane is None or pane.get("dead"):
                    self._clear_managed_agent_identity(record)
                    self.store.update_instance(
                        record["id"],
                        status="interrupted",
                        desired_state="stopped",
                        last_error="Managed shell exited during agent activation"[:500],
                    )
                elif self._pane_at_base_prompt(record, pane):
                    self._clear_managed_agent_identity(record)
                    self.store.update_instance(
                        record["id"],
                        status="running",
                        desired_state="running",
                        last_error="",
                    )
                else:
                    self.store.update_instance(
                        record["id"],
                        status="error",
                        desired_state="running",
                        last_error=_force_stop_required_error(
                            "the activation journal is missing while an unverified "
                            "foreground process is live"
                        ),
                    )
                return self.store.get_instance(record["id"])
            return record
        pending = self._read_pending_activation(record)
        if pending is None:
            self.store.update_instance(
                record["id"],
                status="error",
                last_error=_force_stop_required_error(
                    "the managed agent activation journal is invalid; stop and "
                    "reopen the tab"
                ),
            )
            return self.store.get_instance(record["id"])

        target = pending["target_kind"]
        kind = record.get("kind") or "aeon"
        if kind == target:
            # The ordinary transaction may have committed, or the fail-closed
            # fallback may have persisted only ``kind=target`` before a crash.
            # Idempotently complete the journaled workspace/profile transition
            # in one DB transaction before deleting the only crash evidence.
            # Preserve an existing force-required error for an unverified live
            # foreground; the profile repair must not make that process trusted.
            settings_proven = bool(
                pane
                and not pane.get("dead")
                and pending["phase"] == "command_sent"
                and self._managed_agent_is_foreground(record, pane)
            )
            try:
                repaired = self.store.transition_shell_mode(
                    record["id"],
                    expected_kind=target,
                    kind=target,
                    workspace=pending["workspace"],
                    last_agent_kind=target,
                    clear_profile=pending["clear_profile"],
                    status=record.get("status") or "running",
                    last_error=str(record.get("last_error") or "")[:500],
                )
            except Exception:
                self.store.update_instance(
                    record["id"],
                    status="error",
                    last_error=_force_stop_required_error(
                        "the interrupted cross-kind activation could not be "
                        "repaired atomically"
                    ),
                )
                return self.store.get_instance(record["id"])
            if settings_proven:
                try:
                    self._mark_pending_agent_settings_applied(repaired, pending)
                except Exception:
                    self.store.update_instance(
                        record["id"],
                        status="error",
                        last_error=(
                            "Agent is running, but its applied model/effort state "
                            "could not be repaired"
                        )[:500],
                    )
                    # Keep the immutable journal so a later reconciliation can
                    # idempotently finish this exact settings snapshot.
                    return self.store.get_instance(record["id"])
                if str(repaired.get("last_error") or "") in {
                    "Agent started, but its applied model/effort state could not be recorded",
                    "Agent is running, but its applied model/effort state could not be repaired",
                }:
                    self.store.update_instance(
                        record["id"], status="running", last_error=""
                    )
                    repaired = self.store.get_instance(record["id"])
            self._clear_pending_activation(repaired)
            return repaired
        if kind != "terminal" or int(record.get("shell_backed") or 0) != 1:
            self.store.update_instance(
                record["id"],
                status="error",
                last_error=_force_stop_required_error(
                    "the managed agent activation mode conflicts with its journal"
                ),
            )
            return self.store.get_instance(record["id"])

        if pane is None or pane.get("dead"):
            self._clear_pending_activation(record)
            self._clear_managed_agent_identity(record)
            self.store.update_instance(
                record["id"],
                status="interrupted",
                desired_state="stopped",
                last_error="Managed shell exited during agent activation"[:500],
            )
            return self.store.get_instance(record["id"])

        if self._pane_at_base_prompt(record, pane) or (
            pending["phase"] == "prepared"
            and self._base_shell_has_foreground_control(record, pane)
        ):
            self._clear_pending_activation(record)
            self._clear_managed_agent_identity(record)
            self.store.update_instance(
                record["id"], status="running", desired_state="running", last_error=""
            )
            return self.store.get_instance(record["id"])

        managed = self._managed_agent_is_foreground(record, pane)
        status = "running" if managed else "error"
        message = (
            ""
            if managed
            else _force_stop_required_error(
                "agent activation was interrupted before foreground identity "
                "could not be verified"
            )
        )
        try:
            recovered = self.store.transition_shell_mode(
                record["id"],
                expected_kind="terminal",
                kind=target,
                workspace=pending["workspace"],
                last_agent_kind=target,
                clear_profile=pending["clear_profile"],
                status=status,
                last_error=message,
            )
        except Exception:
            self.store.update_instance(
                record["id"],
                kind=target,
                last_agent_kind=target,
                status="error",
                desired_state="running",
                last_error=_force_stop_required_error(
                    "the activation recovery transaction failed before the "
                    "journaled profile transition completed"
                ),
            )
            # The fallback changes only the mode label. Keep the journal until
            # a later exact-kind transaction proves profile cleanup completed.
            return self.store.get_instance(record["id"])
        if managed:
            try:
                self._mark_pending_agent_settings_applied(recovered, pending)
            except Exception:
                self.store.update_instance(
                    record["id"],
                    status="error",
                    last_error=(
                        "Agent is running, but its applied model/effort state "
                        "could not be recovered"
                    )[:500],
                )
                return self.store.get_instance(record["id"])
        self._clear_pending_activation(record)
        return recovered

    @staticmethod
    def _within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def validate_workspace(self, workspace: str | Path, *, must_exist=True) -> Path:
        if "\x00" in str(workspace):
            raise InstanceError("Workspace contains an invalid NUL character")
        raw = Path(workspace).expanduser()
        try:
            resolved = raw.resolve(strict=must_exist)
        except FileNotFoundError as exc:
            raise InstanceError(f"Workspace does not exist: {raw}") from exc
        roots = [root.resolve() for root in self.config.allowed_roots]
        if not any(self._within(resolved, root) for root in roots):
            raise InstanceError("Workspace is outside AEON_REMOTE_ALLOWED_ROOTS")
        if must_exist and not resolved.is_dir():
            raise InstanceError(f"Workspace is not a directory: {resolved}")
        return resolved

    def create_workspace(self, root: str, name: str) -> str:
        if not WORKSPACE_RE.fullmatch((name or "").strip()):
            raise InstanceError(
                "Workspace name must be 1-80 letters, numbers, dots, dashes, or underscores"
            )
        requested_root = Path(root).expanduser().resolve()
        allowed = [item.resolve() for item in self.config.allowed_roots]
        if requested_root not in allowed:
            raise InstanceError("The selected workspace root is not allowed")
        requested_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination = requested_root / name.strip()
        if destination.exists():
            raise InstanceError(f"Workspace already exists: {destination}")
        destination.mkdir(mode=0o700)
        resolved = self.validate_workspace(destination)
        return str(resolved)

    def list_workspaces(self) -> dict:
        roots = []
        discovered = set(self.store.recent_workspaces())
        for root in self.config.allowed_roots:
            root = root.resolve()
            roots.append(str(root))
            if not root.is_dir():
                continue
            discovered.add(str(root))
            try:
                for entry in list(os.scandir(root))[:200]:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    candidate = Path(entry.path).resolve()
                    if self._within(candidate, root):
                        discovered.add(str(candidate))
            except OSError:
                continue
        valid = []
        for item in sorted(discovered):
            try:
                valid.append(str(self.validate_workspace(item)))
            except InstanceError:
                pass
        return {"roots": roots, "workspaces": valid}

    def list_terminal_hosts(self) -> dict:
        """Return the one host supported by standalone Aeon Remote.

        Nexus extends this method with its fixed worker allowlist.  Keeping the
        standalone default local-only prevents a copied deployment from turning
        a browser-provided address into an SSH target.
        """

        return {
            "hosts": [
                {
                    "host_id": LOCAL_TERMINAL_HOST_ID,
                    "address": LOCAL_TERMINAL_HOST_ID,
                    "hostname": LOCAL_TERMINAL_HOSTNAME,
                    "role": "orchestrator",
                    "local": True,
                    "connected": True,
                    "connection_state": "connected",
                    "terminal_capable": True,
                    "supported_agent_kinds": sorted(AGENT_INSTANCE_KINDS),
                    "agent_capability_reason": "",
                }
            ]
        }

    def validate_terminal_workspace(
        self, host_id: str, workspace: str | Path
    ) -> Path:
        """Validate a terminal's host and initial directory.

        Standalone Aeon Remote deliberately owns only the local host. Nexus's
        fleet-aware manager overrides this narrow hook for its fixed workers.
        """

        if host_id != LOCAL_TERMINAL_HOST_ID:
            raise InstanceError("The requested terminal host is not supported")
        return self.validate_workspace(workspace)

    def _tmux(self, *args, check=False, timeout=10, input_text=None):
        try:
            options = {
                "check": check,
                "capture_output": True,
                "text": True,
                "timeout": timeout,
            }
            if input_text is not None:
                options["input"] = input_text
            return self._run([self.config.tmux_binary, *args], **options)
        except FileNotFoundError as exc:
            raise InstanceError(f"tmux is not installed at {self.config.tmux_binary}") from exc
        except subprocess.TimeoutExpired as exc:
            raise InstanceError("tmux command timed out") from exc

    def _detach_session_clients(self, record: dict) -> None:
        """Detach exact attach clients so unread browser bytes cannot arrive later."""

        result = self._tmux(
            "detach-client", "-s", self._session_target(record["tmux_name"])
        )
        clients = self._tmux(
            "list-clients",
            "-t",
            self._session_target(record["tmux_name"]),
            "-F",
            "#{client_pid}",
        )
        # detach-client may report no matching clients as nonzero on some tmux
        # releases. Accept that only when an independent exact-session listing
        # succeeds and proves there are none; every other result fails closed.
        if clients.returncode != 0 or clients.stdout.strip():
            message = (result.stderr or clients.stderr or "could not detach terminal clients")
            raise InstanceError(message.strip()[:500])
        # A browser wheel may have put the shared pane into tmux copy mode.
        # Lifecycle input must always target the live display, so leave that
        # mode after the old attach clients have been proven absent.
        target = self._pane_target(record["tmux_name"])
        leave_scroll = self._tmux(
            "copy-mode",
            "-e",
            "-t",
            target,
            ";",
            "send-keys",
            "-X",
            "-t",
            target,
            "cancel",
        )
        if leave_scroll.returncode != 0:
            raise InstanceError("could not restore the terminal's live display")
        self._browser_scrolled_ids.discard(record["id"])

    def send_terminal_input(
        self, instance_id: str, data: str, *, expected_generation: int
    ) -> bool:
        """Paste browser input through a private tmux buffer under lifecycle lock.

        Browser bytes never enter an attach-client PTY and never appear in argv.
        This closes both already-queued and not-yet-registered attach races while
        keeping passwords/device codes out of process listings.
        """

        if (
            not isinstance(data, str)
            or len(data) > 65536
            or not isinstance(expected_generation, int)
            or expected_generation < 0
        ):
            return False
        lock = self._mode_lock(instance_id)
        with self._mode_locks_guard:
            if (
                instance_id in self._transitioning_ids
                or self._transition_generations.get(instance_id, 0)
                != expected_generation
            ):
                return False
        if not lock.acquire(blocking=False):
            # A read-only reconciliation/instruction probe may briefly hold the
            # lock. Waiting is safe in this worker thread, but only if no
            # lifecycle generation begins before we acquire it.
            lock.acquire()
        with self._mode_locks_guard:
            if (
                instance_id in self._transitioning_ids
                or self._transition_generations.get(instance_id, 0)
                != expected_generation
            ):
                lock.release()
                return False
        try:
            record = self.store.get_instance(instance_id)
            if not record:
                return False
            pane = self._pane_info(record["tmux_name"])
            if pane is None or pane["dead"]:
                return False
            if instance_id in self._browser_scrolled_ids:
                target = self._pane_target(record["tmux_name"])
                live_display = self._tmux(
                    "copy-mode",
                    "-e",
                    "-t",
                    target,
                    ";",
                    "send-keys",
                    "-X",
                    "-t",
                    target,
                    "cancel",
                )
                if live_display.returncode != 0:
                    return False
                self._browser_scrolled_ids.discard(instance_id)
            return self._paste_private_tmux_buffer(
                record, data, label="input"
            )
        finally:
            lock.release()

    def scroll_terminal(
        self, instance_id: str, lines: int, *, expected_generation: int
    ) -> bool:
        """Scroll tmux history without ever sending wheel bytes to the pane."""

        if (
            isinstance(lines, bool)
            or not isinstance(lines, int)
            or lines == 0
            or abs(lines) > 100
            or not isinstance(expected_generation, int)
            or expected_generation < 0
        ):
            return False
        lock = self._mode_lock(instance_id)
        with self._mode_locks_guard:
            if (
                instance_id in self._transitioning_ids
                or self._transition_generations.get(instance_id, 0)
                != expected_generation
            ):
                return False
        if not lock.acquire(blocking=False):
            lock.acquire()
        with self._mode_locks_guard:
            if (
                instance_id in self._transitioning_ids
                or self._transition_generations.get(instance_id, 0)
                != expected_generation
            ):
                lock.release()
                return False
        try:
            record = self.store.get_instance(instance_id)
            if not record:
                return False
            pane = self._pane_info(record["tmux_name"])
            if pane is None or pane["dead"]:
                return False
            target = self._pane_target(record["tmux_name"])
            direction = "scroll-down" if lines > 0 else "scroll-up"
            result = self._tmux(
                "copy-mode",
                "-e",
                "-t",
                target,
                ";",
                "send-keys",
                "-X",
                "-N",
                str(abs(lines)),
                "-t",
                target,
                direction,
            )
            if result.returncode != 0:
                return False
            # Keep this conservative even after a downward scroll. Reaching the
            # bottom exits copy mode automatically; an extra cancel before the
            # next keystroke is harmless and avoids a second state query.
            self._browser_scrolled_ids.add(instance_id)
            return True
        finally:
            lock.release()

    def _paste_private_tmux_buffer(
        self, record: dict, payload: str, *, label: str
    ) -> bool:
        """Paste stdin without putting sensitive text in argv or a durable buffer."""

        buffer_prefix = f"nexus-{label}-{record['id'][:12]}-"
        buffer_name = f"{buffer_prefix}{uuid.uuid4().hex}"

        # A prior service process could have died after tmux rejected a paste.  Its
        # private buffer is recognizable by this instance/operation namespace, so
        # scrub it before accepting another payload.  Only names are listed; buffer
        # contents are never read back into a log or exception.
        if not self._clear_private_tmux_buffers(buffer_prefix):
            return False

        delivered = False
        try:
            # Submit load + paste/delete as one tmux command sequence.  tmux queues
            # both commands server-side before executing the asynchronous stdin
            # load, so a Nexus process crash cannot occur between two client calls
            # and strand the payload.  The literal payload remains stdin-only.
            result = self._tmux(
                "load-buffer",
                "-b",
                buffer_name,
                "-",
                ";",
                "paste-buffer",
                "-d",
                "-b",
                buffer_name,
                "-t",
                self._pane_target(record["tmux_name"]),
                input_text=payload,
            )
            delivered = result.returncode == 0
        except InstanceError:
            # A timeout or client failure makes delivery ambiguous.  Cleanup and
            # exact absence proof below are still mandatory.
            delivered = False

        # `paste-buffer -d` removes the exact buffer on success.  On every result,
        # independently prove absence; if paste/target resolution failed, issue an
        # exact best-effort delete and verify again.  An unverifiable delete is a
        # failed delivery, never success.
        absent = self._private_tmux_buffer_absent(buffer_name)
        if absent is not True:
            try:
                self._tmux("delete-buffer", "-b", buffer_name)
            except InstanceError:
                pass
            absent = self._private_tmux_buffer_absent(buffer_name)
        return delivered and absent is True

    def _private_tmux_buffer_names(self) -> list[str] | None:
        """Return tmux buffer names only, or None when absence is unprovable."""

        try:
            result = self._tmux("list-buffers", "-F", "#{buffer_name}")
        except InstanceError:
            return None
        if _tmux_proves_no_server(result):
            return []
        if result.returncode != 0:
            return None
        raw = result.stdout
        if "\x00" in raw or "\r" in raw:
            return None
        if raw and not raw.endswith("\n"):
            return None
        names = raw.splitlines()
        if any(not name for name in names):
            return None
        return names

    def _private_tmux_buffer_absent(self, buffer_name: str) -> bool | None:
        names = self._private_tmux_buffer_names()
        if names is None:
            return None
        return buffer_name not in names

    def _clear_private_tmux_buffers(self, prefix: str) -> bool:
        """Delete all exact Nexus-owned stale buffers and prove they are absent."""

        names = self._private_tmux_buffer_names()
        if names is None:
            return False
        stale = [name for name in names if name.startswith(prefix)]
        for name in stale:
            try:
                self._tmux("delete-buffer", "-b", name)
            except InstanceError:
                pass
        remaining = self._private_tmux_buffer_names()
        return remaining is not None and not any(
            name.startswith(prefix) for name in remaining
        )

    def _discard_failed_agent_delivery(self, record: dict) -> bool:
        """Discard uncertain editable input and require a newly marked prompt."""

        try:
            pane = self._pane_info(record["tmux_name"])
            if not self._base_shell_has_foreground_control(record, pane):
                return False
            interrupted = self._tmux(
                "send-keys", "-t", self._pane_target(record["tmux_name"]), "C-c"
            )
            if interrupted.returncode != 0:
                return False
            deadline = time.monotonic() + SHELL_PROMPT_REFRESH_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                pane = self._pane_info(record["tmux_name"])
                if self._pane_at_base_prompt(record, pane):
                    return True
                if pane is None or pane["dead"]:
                    return False
                time.sleep(0.05)
        except InstanceError:
            return False
        return False

    def _handle_failed_agent_delivery(
        self,
        record: dict,
        *,
        target_kind: str,
        workspace: Path,
        previous_agent_kind: str | None,
    ) -> None:
        """Recover a failed atomic paste or persist an actionable ambiguity."""

        if self._discard_failed_agent_delivery(record):
            self._clear_managed_agent_identity(record)
            self._clear_pending_activation(record)
            self.store.update_instance(
                record["id"],
                status="running",
                desired_state="running",
                last_error="Agent command delivery failed before launch"[:500],
            )
            raise InstanceLaunchError(
                "The agent command was not delivered to the managed shell",
                launched=False,
            )

        message = _force_stop_required_error(
            "agent command delivery could not be verified or safely discarded"
        )
        try:
            self.store.transition_shell_mode(
                record["id"],
                expected_kind="terminal",
                kind=target_kind,
                workspace=str(workspace),
                last_agent_kind=target_kind,
                clear_profile=bool(
                    previous_agent_kind and previous_agent_kind != target_kind
                ),
                status="error",
                last_error=message,
            )
        except Exception:
            # Keep the private activation journal so restart reconciliation can
            # idempotently finish any cross-kind profile transition.
            self.store.update_instance(
                record["id"],
                kind=target_kind,
                last_agent_kind=target_kind,
                status="error",
                desired_state="running",
                last_error=message,
            )
        raise InstanceLaunchError(message, launched=True)

    def _refresh_managed_shell_prompt(self, record: dict) -> bool:
        """Discard queued client input and prove a new server-triggered prompt."""

        self._detach_session_clients(record)
        pane = self._pane_info(record["tmux_name"])
        if not self._pane_at_base_prompt(record, pane):
            return False
        if not self._clear_shell_prompt_marker(record):
            return False
        self._tmux(
            "send-keys", "-t", self._pane_target(record["tmux_name"]), "C-c"
        )
        deadline = time.monotonic() + SHELL_PROMPT_REFRESH_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            pane = self._pane_info(record["tmux_name"])
            if self._pane_at_base_prompt(record, pane):
                return True
            if pane is None or pane.get("dead"):
                return False
            time.sleep(0.05)
        return False

    def _pane_info(self, tmux_name: str):
        result = self._tmux(
            "list-panes",
            "-t",
            self._pane_target(tmux_name),
            "-F",
            _TMUX_PANE_FORMAT,
        )
        if result.returncode != 0:
            sessions = self._tmux(
                "list-sessions", "-F", "#{session_name}"
            )
            if _tmux_proves_no_server(sessions):
                return None
            if sessions.returncode != 0:
                raise InstanceError("Could not verify tmux session availability")
            raw_sessions = sessions.stdout
            if "\x00" in raw_sessions or "\r" in raw_sessions:
                raise InstanceError("tmux returned malformed session state")
            names = raw_sessions.splitlines()
            if any(not name for name in names):
                raise InstanceError("tmux returned malformed session state")
            if tmux_name not in names:
                return None
            raise InstanceError("Could not query the exact tmux session pane")
        if (
            not result.stdout.endswith("\n")
            or result.stdout.count("\n") != 1
            or "\r" in result.stdout
        ):
            raise InstanceError("tmux returned malformed pane state")
        parts = result.stdout[:-1].split(_TMUX_PANE_FIELD_SEPARATOR)
        if (
            len(parts) != 4
            or parts[0] not in {"0", "1"}
            or not parts[1].isdigit()
            or int(parts[1]) <= 1
            or (parts[2] and not parts[2].lstrip("-").isdigit())
            or not parts[3]
            or any(value in parts[3] for value in ("\x00", "\r", "\n"))
        ):
            raise InstanceError("tmux returned malformed pane state")
        return {
            "dead": parts[0] == "1",
            "pid": int(parts[1]),
            "exit_code": int(parts[2]) if parts[2].lstrip("-").isdigit() else None,
            "command": parts[3],
        }

    def _kill_session_and_verify_absent(
        self, record: dict, *, error_message: str
    ) -> None:
        """Remove one exact session and independently prove it is absent."""

        self._tmux(
            "kill-session", "-t", self._session_target(record["tmux_name"])
        )
        # A nonzero kill can race a natural exit and a zero kill can still be
        # followed by a query failure. Only the independent tri-state probe is
        # authoritative; errors propagate and an extant dead pane is not absent.
        if self._pane_info(record["tmux_name"]) is not None:
            raise InstanceError(error_message)

    def _pane_current_directory(self, tmux_name: str) -> Path:
        """Return one live pane's allowlisted cwd using an exact tmux target."""
        result = self._tmux(
            "display-message",
            "-p",
            "-t",
            self._pane_target(tmux_name),
            "#{pane_current_path}",
        )
        if result.returncode != 0:
            raise InstanceError("Could not determine the terminal's current directory")
        value = result.stdout
        if value.endswith("\n"):
            value = value[:-1]
        if not value or "\n" in value or "\r" in value or "\x00" in value:
            raise InstanceError("The terminal reported an invalid current directory")
        try:
            return self.validate_workspace(value, must_exist=True)
        except InstanceError:
            raise
        except (OSError, RuntimeError) as exc:
            raise InstanceError(
                "Could not safely resolve the terminal's current directory"
            ) from exc

    @staticmethod
    def _selected_agent_setting_kind(record: dict) -> str:
        kind = record.get("kind") or "aeon"
        if kind in AGENT_INSTANCE_KINDS:
            return kind
        previous = record.get("last_agent_kind")
        return previous if previous in AGENT_INSTANCE_KINDS else "aeon"

    @staticmethod
    def _agent_settings_capability(record: dict) -> tuple[bool, str]:
        if (record.get("host_id") or LOCAL_TERMINAL_HOST_ID) != LOCAL_TERMINAL_HOST_ID:
            return False, "Agent launch settings are unavailable on worker terminals"
        if (record.get("kind") or "aeon") in PROVIDER_AUTH_KINDS:
            return False, "Provider login terminals do not launch agents"
        return True, ""

    def _verified_current_agent_kind(
        self, record: dict, pane: dict | None
    ) -> str | None:
        """Return the active kind only for an exact live managed foreground.

        The durable settings row is launch history.  It becomes current-process
        truth only while the shell-backed foreground identity, lifecycle state,
        and active kind all agree.  Legacy direct sessions have no equivalent
        immutable PGID receipt and therefore remain historical/unknown.
        """

        kind = record.get("kind") or "aeon"
        if (
            kind not in AGENT_INSTANCE_KINDS
            or int(record.get("shell_backed") or 0) != 1
            or record.get("status") != "running"
            or record.get("desired_state") != "running"
            # A durable lifecycle/settings error is unresolved launch truth,
            # even if routine pane reconciliation later observes a live PGID.
            or bool(str(record.get("last_error") or ""))
            or pane is None
            or pane.get("dead")
            or not self._managed_agent_is_foreground(record, pane)
        ):
            return None
        return kind

    def _public_agent_setting(
        self,
        record: dict,
        setting: dict,
        *,
        verified_current_kind: str | None,
        activation_supported: bool,
        capability_reason: str,
    ) -> dict:
        kind = setting["agent_kind"]
        catalog = public_catalog(kind)
        applied = None
        if setting["applied_model"] is not None:
            applied = {
                "model": setting["applied_model"],
                "effort": setting["applied_effort"],
                "at": setting["applied_at"],
            }
        current_process_verified = verified_current_kind == kind
        applied_to_current_process = bool(
            applied is not None and current_process_verified
        )
        desired_matches_applied = bool(
            applied is not None
            and setting["desired_model"] == setting["applied_model"]
            and setting["desired_effort"] == setting["applied_effort"]
        )
        pending = not (current_process_verified and desired_matches_applied)
        if applied_to_current_process:
            applied_scope = "current_process"
            apply_mode = "current_start" if not pending else "next_start"
        elif applied is not None:
            applied_scope = "historical"
            apply_mode = "last_verified"
        elif current_process_verified:
            applied_scope = "none"
            apply_mode = "unknown_current"
        else:
            applied_scope = "none"
            apply_mode = "never_applied"
        return {
            "kind": kind,
            "desired": {
                "model": setting["desired_model"],
                "effort": setting["desired_effort"],
                "updated_at": setting["updated_at"],
            },
            "applied": applied,
            "pending": pending,
            "apply_mode": apply_mode,
            "applied_scope": applied_scope,
            "current_process_verified": current_process_verified,
            "activation_supported": activation_supported,
            "capability_reason": capability_reason,
            "model_editable": bool(
                activation_supported and catalog["model_editable"]
            ),
            "effort_editable": bool(
                activation_supported and catalog["effort_editable"]
            ),
            "catalog": catalog,
        }

    def _agent_settings_payload(
        self, record: dict, *, pane: dict | None = None
    ) -> dict:
        activation_supported, reason = self._agent_settings_capability(record)
        verified_current_kind = self._verified_current_agent_kind(record, pane)
        try:
            settings = self.store.list_agent_settings(record["id"])
        except (AgentSettingsError, ValueError) as exc:
            raise InstanceError("Agent launch settings are unavailable") from exc
        public = {
            kind: self._public_agent_setting(
                record,
                setting,
                verified_current_kind=verified_current_kind,
                activation_supported=activation_supported,
                capability_reason=reason,
            )
            for kind, setting in settings.items()
        }
        return {
            "instance_id": record["id"],
            "selected_kind": self._selected_agent_setting_kind(record),
            "activation_supported": activation_supported,
            "capability_reason": reason,
            "settings": public,
        }

    def get_agent_settings(self, instance_id: str) -> dict:
        with self._mode_lock(instance_id):
            record = self.store.get_instance(instance_id)
            if record is None:
                raise InstanceError("Unknown session")
            pane = None
            if (
                int(record.get("shell_backed") or 0) == 1
                and (record.get("kind") or "aeon") in AGENT_INSTANCE_KINDS
            ):
                try:
                    pane = self._pane_info(record["tmux_name"])
                except InstanceError:
                    # A query ambiguity can never promote historical settings to
                    # current-process truth.
                    pane = None
            return self._agent_settings_payload(record, pane=pane)

    def update_agent_settings(
        self,
        instance_id: str,
        *,
        kind: str,
        model: str,
        effort: str,
        actor: str,
        client_ip: str = "",
    ) -> dict:
        with self._mode_lock(instance_id):
            record = self.store.get_instance(instance_id)
            if record is None:
                raise InstanceError("Unknown session")
            supported, reason = self._agent_settings_capability(record)
            if not supported:
                raise InstanceError(reason)
            normalized_kind = (kind or "").strip().lower()
            if normalized_kind not in AGENT_INSTANCE_KINDS:
                raise InstanceError("Agent kind must be aeon, codex, claude, or grok")
            try:
                before = self.store.get_agent_setting(instance_id, normalized_kind)
                after = self.store.put_agent_setting(
                    instance_id,
                    normalized_kind,
                    model=model,
                    effort=effort,
                )
            except (AgentSettingsError, ValueError) as exc:
                raise InstanceError(str(exc)) from exc
            changed = bool(
                before["desired_model"] != after["desired_model"]
                or before["desired_effort"] != after["desired_effort"]
            )
            self.store.audit(
                "agent_settings_updated",
                actor=actor,
                instance_id=instance_id,
                client_ip=client_ip,
                # Keep model selections, prompts, and all credential-adjacent
                # material out of audit storage. The durable settings table is
                # the single source of truth.
                details={
                    "kind": normalized_kind,
                    "changed": changed,
                    "apply_mode": "next_start",
                },
            )
            try:
                pane = self._pane_info(record["tmux_name"])
            except InstanceError:
                pane = None
            return self._agent_settings_payload(record, pane=pane)

    def _public_record(self, record: dict, pane=None) -> dict:
        result = {
            key: record.get(key)
            for key in (
                "id", "host_id", "kind", "shell_backed", "last_agent_kind", "name",
                "workspace", "objective", "max_iterations", "model", "status",
                "desired_state", "created_at", "updated_at", "last_started_at",
                "last_error", "created_by", "launch_origin",
            )
        }
        result["host_id"] = result.get("host_id") or LOCAL_TERMINAL_HOST_ID
        result.update(
            {
                "host_address": result["host_id"],
                "host_hostname": (
                    LOCAL_TERMINAL_HOSTNAME
                    if result["host_id"] == LOCAL_TERMINAL_HOST_ID
                    else None
                ),
                "host_role": (
                    "orchestrator"
                    if result["host_id"] == LOCAL_TERMINAL_HOST_ID
                    else None
                ),
                "host_local": result["host_id"] == LOCAL_TERMINAL_HOST_ID,
                "host_connected": result["host_id"] == LOCAL_TERMINAL_HOST_ID,
                "host_connection_state": (
                    "connected"
                    if result["host_id"] == LOCAL_TERMINAL_HOST_ID
                    else "unsupported"
                ),
                "terminal_capable": result["host_id"] == LOCAL_TERMINAL_HOST_ID,
                "supported_agent_kinds": (
                    sorted(AGENT_INSTANCE_KINDS)
                    if result["host_id"] == LOCAL_TERMINAL_HOST_ID
                    else []
                ),
                "agent_capability_reason": (
                    ""
                    if result["host_id"] == LOCAL_TERMINAL_HOST_ID
                    else "This host is not supported by standalone Aeon Remote"
                ),
            }
        )
        result["kind"] = result.get("kind") or "aeon"
        result["shell_backed"] = bool(result.get("shell_backed"))
        result["force_stop_required"] = bool(
            result.get("status") == "error"
            and _has_force_stop_required_error(result)
        )
        result.update(project_manager_public_flags(record))
        if result["kind"] in PROVIDER_IDS:
            result["provider"] = result["kind"]
            result["auth_session"] = False
        elif result["kind"] in PROVIDER_AUTH_KINDS:
            result["provider"] = PROVIDER_AUTH_KINDS[result["kind"]]
            result["auth_session"] = True
        else:
            result["provider"] = None
            result["auth_session"] = False
        if result["kind"] in PROVIDER_AUTH_KINDS:
            result["mode"] = "auth"
            result["agent_kind"] = None
        elif result["shell_backed"]:
            result["mode"] = (
                "terminal" if result["kind"] == "terminal" else "agent"
            )
            result["agent_kind"] = (
                result["kind"] if result["kind"] in AGENT_INSTANCE_KINDS else None
            )
        else:
            result["mode"] = (
                "terminal" if result["kind"] == "terminal" else "agent"
            )
            result["agent_kind"] = (
                result["kind"] if result["kind"] in AGENT_INSTANCE_KINDS else None
            )
        if pane:
            result["pid"] = pane["pid"]
            result["exit_code"] = pane["exit_code"] if pane["dead"] else None
            result["process"] = pane["command"]
        else:
            result.update({"pid": None, "exit_code": None, "process": None})
        result["current_directory"] = None
        if (
            (result["kind"] == "terminal" or result["shell_backed"])
            and pane
            and not pane["dead"]
        ):
            try:
                result["current_directory"] = str(
                    self._pane_current_directory(record["tmux_name"])
                )
            except (InstanceError, OSError, RuntimeError):
                # A shell can move or disappear between the pane and cwd probes.
                # Listing remains available, but never exposes an unvalidated path.
                pass
        result["resources"] = self._process_resources(result["pid"]) if result["pid"] else None
        try:
            setting_payload = self._agent_settings_payload(record, pane=pane)
            selected_setting = setting_payload["settings"][
                setting_payload["selected_kind"]
            ]
        except InstanceError:
            selected_setting = None
        result["agent_runtime_settings"] = selected_setting
        result["agent_model"] = (
            selected_setting["desired"]["model"] if selected_setting else ""
        )
        result["reasoning_effort"] = (
            selected_setting["desired"]["effort"] if selected_setting else ""
        )
        result["applied_agent_model"] = (
            selected_setting["applied"]["model"]
            if selected_setting and selected_setting["applied"] is not None
            else None
        )
        result["applied_reasoning_effort"] = (
            selected_setting["applied"]["effort"]
            if selected_setting and selected_setting["applied"] is not None
            else None
        )
        # Missing/corrupt settings are uncertainty, never evidence that the
        # current process received the desired values.
        result["agent_settings_pending"] = (
            True if selected_setting is None else bool(selected_setting["pending"])
        )
        return result

    def reconcile(self, record: dict) -> dict:
        # Every lifecycle writer uses this same per-instance lock. Re-read only
        # after acquiring it so stale API/list snapshots cannot overwrite a
        # simultaneous activate, end, resume, stop, or delete operation.
        with self._mode_lock(record["id"]):
            refreshed = self.store.get_instance(record["id"])
            if refreshed is not None:
                record = refreshed
            record = self._normalize_dormant_project_manager(record)
            return self._reconcile_unlocked(record)

    def _reconcile_unlocked(self, record: dict) -> dict:
        pane = self._pane_info(record["tmux_name"])
        if int(record.get("shell_backed") or 0) == 1:
            record = self._recover_pending_activation(record, pane)
        force_stop_required = _has_force_stop_required_error(record)
        if (
            not force_stop_required
            and int(record.get("shell_backed") or 0) == 1
            and (record.get("kind") or "aeon") in AGENT_INSTANCE_KINDS
            and pane is not None
            and not pane["dead"]
            and self._pane_at_base_prompt(record, pane)
        ):
            try:
                record = self.store.transition_shell_mode(
                    record["id"],
                    expected_kind=record.get("kind") or "aeon",
                    kind="terminal",
                    status="running",
                )
                self._clear_managed_agent_identity(record)
                self._clear_pending_activation(record)
            except ValueError:
                refreshed = self.store.get_instance(record["id"])
                if refreshed is not None:
                    record = refreshed
        dormant = dormant_project_manager_status(
            record,
            pane_exists=pane is not None,
            pane_dead=bool(pane and pane["dead"]),
        )
        if force_stop_required and pane is not None and not pane["dead"]:
            # Routine telemetry reconciliation must preserve this durable,
            # actionable safety state instead of normalizing it to ``running``.
            status = "error"
        elif dormant is not None:
            status = dormant
        elif pane is None:
            status = "interrupted" if record["desired_state"] == "running" else "stopped"
        elif pane["dead"]:
            status = "exited" if record["desired_state"] == "running" else "stopped"
        elif record["desired_state"] == "stopped":
            status = "stopping"
        else:
            status = "running"
        if status != record["status"]:
            self.store.update_instance(record["id"], status=status)
            record = self.store.get_instance(record["id"])
        return self._public_record(record, pane)

    def list_instances(self) -> list[dict]:
        if self._project_manager_enabled:
            try:
                ensure_project_manager(
                    self.store, default_model=self.config.default_model
                )
            except ProjectManagerError as exc:
                raise InstanceError(str(exc)) from exc
        return [self.reconcile(record) for record in self.store.list_instances()]

    def ensure_default_home_terminal(self) -> None:
        """Open the pinned Home shell once for the Nexus application.

        This terminal-only action never starts Aeon, a provider CLI, a model
        runtime, or a compute reservation. A failed or later-interrupted shell
        is not retried in a loop; its durable tab remains visible for an
        explicit reopen action.
        """

        if not self._project_manager_enabled:
            return
        with self._project_manager_lock:
            try:
                record, _ = ensure_project_manager(
                    self.store, default_model=self.config.default_model
                )
            except ProjectManagerError as exc:
                raise InstanceError(str(exc)) from exc
            with self._lifecycle_lock(record["id"]):
                refreshed = self.store.get_instance(record["id"])
                if refreshed is not None:
                    record = refreshed
                record = self._normalize_dormant_project_manager(record)
                if (
                    record.get("kind") != "terminal"
                    or int(record.get("shell_backed") or 0) != 1
                    or record.get("status") == "error"
                ):
                    return
                try:
                    pane = self._pane_info(record["tmux_name"])
                except InstanceError as exc:
                    self.store.update_instance(
                        record["id"],
                        status="error",
                        desired_state="stopped",
                        last_error=str(exc)[:500],
                    )
                    return
                if pane is not None and not pane["dead"]:
                    return
                virgin = (
                    record.get("status") == "idle"
                    and record.get("desired_state") == "stopped"
                )
                interrupted_running_shell = record.get("desired_state") == "running"
                if not (virgin or interrupted_running_shell):
                    return
                try:
                    self._launch_record(record)
                except InstanceError:
                    # _launch_record persists a safe error state. Listing remains
                    # usable and no autonomous retry is scheduled.
                    return
                self.store.audit(
                    "project_manager_terminal_opened",
                    actor="nexus",
                    instance_id=record["id"],
                    details={"workspace": record["workspace"]},
                )

    def get_instance(self, instance_id: str) -> dict:
        record = self.store.get_instance(instance_id)
        if not record:
            raise InstanceError("Unknown session")
        return self.reconcile(record)

    def apply_pending_instructions(self, instance_id: str) -> bool:
        """Atomically refresh one running Aeon's private prompt layers.

        Returns false for stopped/provider/legacy sessions. Provider CLIs load
        their file-backed layer only at process start; their edits intentionally
        remain pending until restart.
        """

        if self.instruction_service is None:
            return False
        with self._mode_lock(instance_id):
            record = self.store.get_instance(instance_id)
            if not record or (record.get("kind") or "aeon") != "aeon":
                return False
            pane = self._pane_info(record["tmux_name"])
            if not pane or pane["dead"] or record.get("desired_state") != "running":
                return False
            if int(record.get("shell_backed") or 0):
                if not self._managed_agent_is_foreground(record, pane):
                    return False
            else:
                expected_path = (
                    self.config.instance_state_dir
                    / record["id"]
                    / "runtime-instructions.json"
                )
                environment = self._tmux(
                    "show-environment",
                    "-t",
                    self._session_target(record["tmux_name"]),
                    RUNTIME_INSTRUCTIONS_ENV,
                )
                expected_line = f"{RUNTIME_INSTRUCTIONS_ENV}={expected_path}"
                if (
                    environment.returncode != 0
                    or environment.stdout.strip() != expected_line
                ):
                    return False
            try:
                snapshot = self.instruction_service.launch_snapshot(instance_id)
                path = materialize_runtime_instructions(
                    snapshot, self.config.instance_state_dir / record["id"]
                )
                load_runtime_instructions(
                    path,
                    expected_instance_id=instance_id,
                    expected_agent_kind="aeon",
                )
                self.instruction_service.mark_applied(
                    instance_id,
                    profile_version_id=snapshot["profile_version_id"],
                    local_revision=snapshot["local_revision"],
                )
            except (InstructionProfileError, RuntimeInstructionError) as exc:
                raise InstanceError(str(exc)) from exc
            return True

    def _launch_record(
        self,
        record: dict,
        *,
        resume: bool = False,
        command_override: list[str] | None = None,
    ) -> None:
        kind = record.get("kind") or "aeon"
        if kind not in INSTANCE_KINDS:
            raise InstanceError("This session has an unsupported kind")
        if command_override is not None and kind != "aeon":
            raise InstanceError("A terminal command cannot be overridden")
        agent_setting = None
        if command_override is None and kind in AGENT_INSTANCE_KINDS:
            try:
                agent_setting = self.store.get_agent_setting(record["id"], kind)
            except (AgentSettingsError, ValueError) as exc:
                raise InstanceError("Agent launch settings are invalid") from exc
        if kind == "aeon" and record.get("launch_origin") == "local":
            # A local user may start Aeon in any directory they can already
            # access. Persist that provenance so dashboard resume can reuse only
            # the exact recorded path; browser-created rows still go through the
            # configured allowlist on every launch.
            try:
                workspace = Path(record["workspace"]).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise InstanceError(
                    f"Locally adopted workspace is unavailable: {record['workspace']}"
                ) from exc
            if not workspace.is_dir() or str(workspace) != record["workspace"]:
                raise InstanceError("Locally adopted workspace identity changed")
        else:
            workspace = self.validate_workspace(record["workspace"])
        tmux_name = record["tmux_name"]
        existing = self._pane_info(tmux_name)
        if existing and not existing["dead"]:
            raise InstanceError("This session is already running")
        if existing and existing["dead"]:
            self._tmux("kill-session", "-t", self._session_target(tmux_name))

        instruction_snapshot = None
        provider_instruction_path = None
        runtime_instruction_path = None
        if self.instruction_service is not None and kind in AGENT_INSTANCE_KINDS:
            try:
                instruction_snapshot = self.instruction_service.launch_snapshot(
                    record["id"]
                )
                instance_dir = self.config.instance_state_dir / record["id"]
                if kind == "aeon":
                    runtime_instruction_path = materialize_runtime_instructions(
                        instruction_snapshot, instance_dir
                    )
                elif kind == "claude":
                    provider_instruction_path = materialize_provider_instruction_text(
                        instruction_snapshot, instance_dir
                    )
                elif kind == "grok":
                    provider_instruction_path = materialize_grok_agent_profile(
                        instruction_snapshot, instance_dir
                    )
            except (InstructionProfileError, RuntimeInstructionError) as exc:
                raise InstanceError(str(exc)) from exc

        if command_override is not None:
            command = list(command_override)
        elif kind == "terminal":
            # Browser input can select only name and allowlisted workspace. The
            # executable, private prompt identity, arguments, and clean
            # environment remain fixed server-side.
            command = _materialize_managed_shell(record, self.config)
        elif kind == "aeon":
            objective = "Continue from where you left off." if resume else record["objective"]
            command = [self.config.python_executable, "-m", "aeon.main"]
            model = agent_setting["desired_model"]
            command.extend(["--model", model])
            if record.get("max_iterations"):
                command.extend(["--max-iterations", str(record["max_iterations"])])
            if objective:
                command.extend(["--start", objective])
        else:
            provider_id = PROVIDER_AUTH_KINDS.get(kind, kind)
            try:
                provider_command = (
                    provider_connect_command(provider_id)
                    if kind in PROVIDER_AUTH_KINDS
                    else provider_agent_command(provider_id)
                )
                clean_environment = subscription_environment(provider_id)
            except ProviderError as exc:
                raise InstanceError(str(exc)) from exc
            # A tmux server is long lived and can retain environment variables
            # from the service that created it.  Start provider CLIs through a
            # fixed env(1) clean-room argv so Nexus/OIDC/API secrets can never be
            # inherited.  Only the provider module's small non-secret allowlist
            # is restored; browser input controls none of this argv.
            provider_argv = list(provider_command.argv)
            desired_model = (
                agent_setting["desired_model"] if agent_setting is not None else ""
            )
            desired_effort = (
                agent_setting["desired_effort"] if agent_setting is not None else ""
            )
            if kind == "claude" and provider_instruction_path is not None:
                if desired_model:
                    provider_argv.extend(["--model", desired_model])
                if desired_effort:
                    provider_argv.extend(["--effort", desired_effort])
                provider_argv.extend(
                    ["--append-system-prompt-file", str(provider_instruction_path)]
                )
            elif kind == "claude":
                if desired_model:
                    provider_argv.extend(["--model", desired_model])
                if desired_effort:
                    provider_argv.extend(["--effort", desired_effort])
            elif kind == "codex":
                instructions = None
                if instruction_snapshot is not None:
                    layers = runtime_instruction_layers_from_snapshot(
                        instruction_snapshot
                    )
                    instructions = format_runtime_instruction_layers(layers)
                codex_options = []
                if desired_model:
                    codex_options.extend(["--model", desired_model])
                if desired_effort:
                    codex_options.extend(
                        [
                            "--config",
                            "model_reasoning_effort="
                            f"{json.dumps(desired_effort, ensure_ascii=True)}",
                        ]
                    )
                if instructions is not None:
                    profile_name = _materialize_codex_profile(
                        instance_id=record["id"],
                        instructions=instructions,
                        workspace=workspace,
                        environment=clean_environment,
                    )
                    codex_options.extend(["--profile", profile_name])
                provider_argv[1:1] = codex_options
            elif kind == "grok":
                if desired_model:
                    provider_argv.extend(["--model", desired_model])
                if provider_instruction_path is not None:
                    provider_argv.extend(["--agent", str(provider_instruction_path)])
            command = [ENV_BINARY, "-i"]
            command.extend(
                f"{key}={value}" for key, value in sorted(clean_environment.items())
            )
            command.extend(provider_argv)

        env_options = {}
        if kind == "aeon":
            env_options = {
                "PYTHONPATH": str(self.config.project_root),
                "AEON_REMOTE_INSTANCE_ID": record["id"],
                "USE_TF": "0",
                "USE_FLAX": "0",
            }
            if runtime_instruction_path is not None:
                env_options[RUNTIME_INSTRUCTIONS_ENV] = str(runtime_instruction_path)
            # Legacy direct Aeon rows and locally adopted CLI sessions remain
            # resumable, but must never inherit OIDC/Cloudflare/API credentials
            # retained by a long-lived tmux server. Runtime/cache paths come from
            # the same reviewed allowlist as managed shells; Aeon values are
            # explicit and server-derived.
            clean_environment = _managed_shell_environment()
            clean_environment.update(env_options)
            command = [ENV_BINARY, "-i", *(
                f"{key}={value}" for key, value in sorted(clean_environment.items())
            ), *command]
        tmux_command = [
            "new-session", "-d", "-s", tmux_name, "-c", str(workspace),
            "-x", "120", "-y", "36",
        ]
        for key, value in env_options.items():
            tmux_command.extend(["-e", f"{key}={value}"])
        tmux_command.extend(command)

        self.store.update_instance(
            record["id"],
            status="starting",
            desired_state="running",
            last_started_at=time.time(),
            last_error="",
        )
        result = self._tmux(*tmux_command)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "tmux launch failed").strip()[:500]
            self.store.update_instance(
                record["id"], status="error", desired_state="stopped", last_error=message
            )
            raise InstanceLaunchError(message, launched=False)

        try:
            self._tmux(
                "set-option", "-t", self._session_target(tmux_name), "remain-on-exit", "on"
            )
            self._tmux(
                "set-option", "-t", self._session_target(tmux_name), "history-limit", "100000"
            )
            if kind == "aeon":
                instance_dir = self.config.instance_state_dir / record["id"]
                instance_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                os.chmod(instance_dir, 0o700)
                transcript = instance_dir / "terminal.log"
                pipe_args = [
                    self.config.python_executable,
                    str(Path(__file__).with_name("logpipe.py")),
                    "--path",
                    str(transcript),
                ]
                pipe_command = "exec " + " ".join(shlex.quote(arg) for arg in pipe_args)
                self._tmux(
                    "pipe-pane", "-o", "-t", self._pane_target(tmux_name), pipe_command
                )
            self.store.update_instance(record["id"], status="running")
            if agent_setting is not None:
                pane = self._pane_info(tmux_name)
                if pane is None or pane["dead"]:
                    raise InstanceError("The agent exited before launch was verified")
                self.store.mark_agent_setting_applied(
                    record["id"],
                    kind,
                    model=agent_setting["desired_model"],
                    effort=agent_setting["desired_effort"],
                )
            if (
                instruction_snapshot is not None
                and kind in {"aeon", "codex", "claude", "grok"}
            ):
                self.instruction_service.mark_applied(
                    record["id"],
                    profile_version_id=instruction_snapshot["profile_version_id"],
                    local_revision=instruction_snapshot["local_revision"],
                )
        except Exception as exc:
            message = f"tmux session launched but management setup failed: {exc}"
            try:
                self.store.update_instance(record["id"], status="error", last_error=message[:500])
            except Exception:
                pass
            raise InstanceLaunchError(message, launched=True) from exc

    def adopt_local_cli(
        self,
        *,
        workspace: str | Path,
        cli_args: list[str],
        objective: str,
        max_iterations: int | None,
        model: str | None,
        actor: str,
    ) -> dict:
        """Put one already-authorized local CLI invocation into managed tmux.

        Unlike browser-created workspaces, the current local cwd is not checked
        against the web allowlist: the local user already has filesystem access.
        The launched executable is still fixed to ``python -m aeon.main`` and all
        arguments remain a direct argv, never a shell command.
        """
        try:
            workspace_path = Path(workspace).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise InstanceError(f"Local workspace does not exist: {workspace}") from exc
        if not workspace_path.is_dir():
            raise InstanceError(f"Local workspace is not a directory: {workspace_path}")
        if not isinstance(cli_args, list) or any(
            not isinstance(value, str) or "\x00" in value for value in cli_args
        ):
            raise InstanceError("Local Aeon argv is invalid")

        instance_id = uuid.uuid4().hex
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", workspace_path.name).strip("-._")
        stem = (stem or "aeon")[:48]
        name = f"{stem}-{instance_id[:8]}"
        now = time.time()
        record = {
            "id": instance_id,
            "kind": "aeon",
            "shell_backed": 0,
            "last_agent_kind": "aeon",
            "name": name,
            "tmux_name": f"aeon-{instance_id[:12]}",
            "workspace": str(workspace_path),
            "objective": (objective or "")[:20000],
            "max_iterations": max_iterations,
            "model": model or self.config.default_model,
            "status": "created",
            "desired_state": "running",
            "created_at": now,
            "updated_at": now,
            "last_started_at": None,
            "last_error": "",
            "created_by": actor,
            "launch_origin": "local",
        }
        try:
            self.store.create_instance(record)
        except Exception as exc:
            raise InstanceError(f"Could not register local Aeon instance: {exc}") from exc
        self.store.audit(
            "instance_adopted_locally",
            actor=actor,
            instance_id=instance_id,
            details={"name": name, "workspace": str(workspace_path)},
        )
        command = [self.config.python_executable, "-m", "aeon.main", *cli_args]
        self._launch_record(record, command_override=command)
        return self.get_instance(instance_id)

    def create_instance(
        self,
        *,
        kind: str = "aeon",
        name: str,
        workspace: str,
        objective: str,
        max_iterations: int | None,
        actor: str,
        client_ip: str = "",
    ) -> dict:
        kind = (kind or "aeon").strip().lower()
        if kind not in AGENT_INSTANCE_KINDS:
            raise InstanceError("Agent kind must be aeon, codex, claude, or grok")
        name = (name or "").strip()
        if not NAME_RE.fullmatch(name):
            raise InstanceError(
                "Name must be 1-64 letters, numbers, spaces, dots, dashes, or underscores"
            )
        workspace_path = self.validate_workspace(workspace)
        objective = (objective or "").strip()
        if "\x00" in objective:
            raise InstanceError("Objective contains an invalid NUL character")
        if len(objective) > 20000:
            raise InstanceError("Objective is too long")
        if max_iterations is not None and not (1 <= max_iterations <= 10000):
            raise InstanceError("max_iterations must be between 1 and 10000")
        if kind != "aeon" and (objective or max_iterations is not None):
            raise InstanceError(
                "Provider sessions do not accept an Aeon objective or iteration limit"
            )

        instance_id = uuid.uuid4().hex
        now = time.time()
        record = {
            "id": instance_id,
            "kind": kind,
            "shell_backed": 0,
            "last_agent_kind": kind,
            "name": name,
            "tmux_name": f"{kind}-{instance_id[:12]}",
            "workspace": str(workspace_path),
            "objective": objective,
            "max_iterations": max_iterations,
            "model": self.config.default_model if kind == "aeon" else None,
            "status": "created",
            "desired_state": "running",
            "created_at": now,
            "updated_at": now,
            "last_started_at": None,
            "last_error": "",
            "created_by": actor,
            "launch_origin": "web",
        }
        try:
            self.store.create_instance(record)
        except Exception as exc:
            raise InstanceError(f"Could not create instance: {exc}") from exc
        self.store.audit(
            "instance_created" if kind == "aeon" else "provider_agent_created",
            actor=actor,
            instance_id=instance_id,
            client_ip=client_ip,
            details={
                "name": name,
                "workspace": str(workspace_path),
                "kind": kind,
            },
        )
        self._launch_record(record)
        return self.get_instance(instance_id)

    def create_provider_auth(
        self,
        provider_id: str,
        *,
        actor: str,
        client_ip: str = "",
    ) -> dict:
        """Create one native provider login terminal without handling credentials."""

        provider_id = (provider_id or "").strip().lower()
        if provider_id not in PROVIDER_IDS:
            raise InstanceError("Unsupported provider")
        # Resolve and validate the fixed command before creating a durable row.
        try:
            provider_connect_command(provider_id)
        except ProviderError as exc:
            raise InstanceError(str(exc)) from exc
        try:
            workspace_path = self.validate_workspace("/home/aday")
        except InstanceError:
            # Standalone/test deployments may intentionally expose a narrower
            # root. Login itself is workspace-independent, so use only the
            # first configured allowlisted directory in that case.
            workspace_path = self.validate_workspace(self.config.allowed_roots[0])
        instance_id = uuid.uuid4().hex
        now = time.time()
        label = provider_id.capitalize()
        name = f"Connect {label} {instance_id[:8]}"
        kind = f"{provider_id}_auth"
        record = {
            "id": instance_id,
            "kind": kind,
            "shell_backed": 0,
            "last_agent_kind": None,
            "name": name,
            "tmux_name": f"{kind}-{instance_id[:12]}",
            "workspace": str(workspace_path),
            "objective": "",
            "max_iterations": None,
            "model": None,
            "status": "created",
            "desired_state": "running",
            "created_at": now,
            "updated_at": now,
            "last_started_at": None,
            "last_error": "",
            "created_by": actor,
            "launch_origin": "web",
        }
        try:
            self.store.create_instance(record)
        except Exception as exc:
            raise InstanceError("Could not create the provider login session") from exc
        self.store.audit(
            "provider_login_started",
            actor=actor,
            instance_id=instance_id,
            client_ip=client_ip,
            details={"provider": provider_id},
        )
        self._launch_record(record)
        return self.get_instance(instance_id)

    def create_terminal(
        self,
        *,
        name: str | None = None,
        workspace: str,
        host_id: str = LOCAL_TERMINAL_HOST_ID,
        actor: str,
        client_ip: str = "",
    ) -> dict:
        """Create a managed shell whose executable is never browser-controlled."""
        host_id = str(host_id or "").strip()
        requested_name = (name or "").strip()
        automatic_name = not requested_name or bool(
            re.fullmatch(r"Terminal [1-9][0-9]*", requested_name)
        )
        if requested_name and not NAME_RE.fullmatch(requested_name):
            raise InstanceError(
                "Name must be 1-64 letters, numbers, spaces, dots, dashes, or underscores"
            )
        workspace_path = self.validate_terminal_workspace(host_id, workspace)
        with self._terminal_creation_lock:
            if automatic_name:
                occupied = {
                    str(item.get("name") or "").casefold()
                    for item in self.store.list_instances()
                }
                number = 1
                while f"terminal {number}" in occupied:
                    number += 1
                requested_name = f"Terminal {number}"
            instance_id = uuid.uuid4().hex
            now = time.time()
            record = {
                "id": instance_id,
                "host_id": host_id,
                "kind": "terminal",
                "shell_backed": 1,
                "last_agent_kind": None,
                "name": requested_name,
                "tmux_name": f"terminal-{instance_id[:12]}",
                "workspace": str(workspace_path),
                "objective": "",
                "max_iterations": None,
                "model": None,
                "status": "created",
                "desired_state": "running",
                "created_at": now,
                "updated_at": now,
                "last_started_at": None,
                "last_error": "",
                "created_by": actor,
                "launch_origin": "web",
            }
            try:
                self.store.create_instance(record)
            except Exception as exc:
                raise InstanceError(f"Could not create terminal: {exc}") from exc
        self.store.audit(
            "terminal_created",
            actor=actor,
            instance_id=instance_id,
            client_ip=client_ip,
            details={
                "name": requested_name,
                "workspace": str(workspace_path),
                "host_id": host_id,
            },
        )
        self._launch_record(record)
        return self.get_instance(instance_id)

    def rename_instance(
        self,
        instance_id: str,
        *,
        name: str,
        actor: str,
        client_ip: str = "",
    ) -> dict:
        """Rename one durable browser tab without changing its tmux identity."""

        requested_name = (name or "").strip()
        if not NAME_RE.fullmatch(requested_name):
            raise InstanceError(
                "Name must be 1-64 letters, numbers, spaces, dots, dashes, or underscores"
            )
        with self._lifecycle_lock(instance_id):
            record = self.store.get_instance(instance_id)
            if not record:
                raise InstanceError("Unknown session")
            if record.get("name") == requested_name:
                return self.get_instance(instance_id)
            try:
                self.store.update_instance(instance_id, name=requested_name)
            except Exception as exc:
                raise InstanceError("That tab name is already in use") from exc
            self.store.audit(
                "instance_renamed",
                actor=actor,
                instance_id=instance_id,
                client_ip=client_ip,
                details={"name": requested_name},
            )
            return self.get_instance(instance_id)

    def _prepare_shell_agent_command(
        self,
        record: dict,
        kind: str,
        *,
        workspace: Path,
        preserve_profile: bool,
    ) -> tuple[list[str], dict | None, dict]:
        """Build one fixed agent argv for an existing managed shell tab."""

        if kind not in AGENT_INSTANCE_KINDS:
            raise InstanceError("Agent kind must be aeon, codex, claude, or grok")
        try:
            agent_setting = self.store.get_agent_setting(record["id"], kind)
        except (AgentSettingsError, ValueError) as exc:
            raise InstanceError("Agent launch settings are invalid") from exc
        instruction_snapshot = None
        provider_instruction_path = None
        runtime_instruction_path = None
        if self.instruction_service is not None:
            try:
                instruction_snapshot = (
                    self.instruction_service.launch_snapshot_for_agent_kind(
                        record["id"],
                        agent_kind=kind,
                        preserve_profile=preserve_profile,
                    )
                )
                instance_dir = self.config.instance_state_dir / record["id"]
                if kind == "aeon":
                    runtime_instruction_path = materialize_runtime_instructions(
                        instruction_snapshot, instance_dir
                    )
                elif kind == "claude":
                    provider_instruction_path = materialize_provider_instruction_text(
                        instruction_snapshot, instance_dir
                    )
                elif kind == "grok":
                    provider_instruction_path = materialize_grok_agent_profile(
                        instruction_snapshot, instance_dir
                    )
            except (InstructionProfileError, RuntimeInstructionError) as exc:
                raise InstanceError(str(exc)) from exc

        if kind == "aeon":
            clean_environment = subscription_environment("codex")
            clean_environment.update(
                {
                    "PYTHONPATH": str(self.config.project_root),
                    "AEON_REMOTE_INSTANCE_ID": record["id"],
                    "AEON_DISABLE_AUTO_TMUX": "1",
                    "USE_TF": "0",
                    "USE_FLAX": "0",
                }
            )
            if runtime_instruction_path is not None:
                clean_environment[RUNTIME_INSTRUCTIONS_ENV] = str(
                    runtime_instruction_path
                )
            argv = [
                self.config.python_executable,
                "-m",
                "aeon.main",
                "--model",
                agent_setting["desired_model"],
            ]
            # The protected Home tab receives its fixed project-manager role on
            # every fresh Aeon foreground. Ordinary terminals remain interactive.
            if is_project_manager_record(record):
                argv.extend(["--start", PROJECT_MANAGER_OBJECTIVE])
            command = [ENV_BINARY, "-i"]
            command.extend(
                f"{key}={value}" for key, value in sorted(clean_environment.items())
            )
            command.extend(argv)
            return command, instruction_snapshot, agent_setting

        try:
            status = provider_status(kind)
            if status.get("installed") is not True:
                raise InstanceError(f"The official {kind} CLI is not installed")
            if kind in {"codex", "claude"} and status.get("connected") is not True:
                raise InstanceError(
                    f"Connect {kind} in Settings before starting this agent"
                )
            provider_command = provider_agent_command(kind)
            clean_environment = subscription_environment(kind)
        except ProviderError as exc:
            raise InstanceError(str(exc)) from exc
        provider_argv = list(provider_command.argv)
        desired_model = agent_setting["desired_model"]
        desired_effort = agent_setting["desired_effort"]
        if kind == "claude" and provider_instruction_path is not None:
            if desired_model:
                provider_argv.extend(["--model", desired_model])
            if desired_effort:
                provider_argv.extend(["--effort", desired_effort])
            provider_argv.extend(
                ["--append-system-prompt-file", str(provider_instruction_path)]
            )
        elif kind == "claude":
            if desired_model:
                provider_argv.extend(["--model", desired_model])
            if desired_effort:
                provider_argv.extend(["--effort", desired_effort])
        elif kind == "codex":
            instructions = None
            if instruction_snapshot is not None:
                layers = runtime_instruction_layers_from_snapshot(instruction_snapshot)
                instructions = format_runtime_instruction_layers(layers)
            codex_options = []
            if desired_model:
                codex_options.extend(["--model", desired_model])
            if desired_effort:
                codex_options.extend(
                    [
                        "--config",
                        "model_reasoning_effort="
                        f"{json.dumps(desired_effort, ensure_ascii=True)}",
                    ]
                )
            if instructions is not None:
                profile_name = _materialize_codex_profile(
                    instance_id=record["id"],
                    instructions=instructions,
                    workspace=workspace,
                    environment=clean_environment,
                )
                codex_options.extend(["--profile", profile_name])
            provider_argv[1:1] = codex_options
        elif kind == "grok":
            if desired_model:
                provider_argv.extend(["--model", desired_model])
            if provider_instruction_path is not None:
                provider_argv.extend(["--agent", str(provider_instruction_path)])
        command = [ENV_BINARY, "-i"]
        command.extend(
            f"{key}={value}" for key, value in sorted(clean_environment.items())
        )
        command.extend(provider_argv)
        return command, instruction_snapshot, agent_setting

    def activate_agent(
        self,
        terminal_id: str,
        *,
        kind: str,
        actor: str,
        client_ip: str = "",
    ) -> dict:
        """Serialize one tab's terminal-to-agent transition."""

        normalized_kind = (kind or "").strip().lower()
        if normalized_kind not in AGENT_INSTANCE_KINDS:
            raise InstanceError("Agent kind must be aeon, codex, claude, or grok")
        with self._lifecycle_lock(terminal_id):
            return self._activate_agent_locked(
                terminal_id,
                kind=normalized_kind,
                actor=actor,
                client_ip=client_ip,
            )

    def _activate_agent_locked(
        self,
        terminal_id: str,
        *,
        kind: str,
        actor: str,
        client_ip: str = "",
    ) -> dict:
        """Run one fixed agent in the foreground of an existing terminal tab."""

        kind = (kind or "").strip().lower()
        if kind not in AGENT_INSTANCE_KINDS:
            raise InstanceError("Agent kind must be aeon, codex, claude, or grok")
        terminal = self.store.get_instance(terminal_id)
        if not terminal:
            raise InstanceError("Unknown session")
        if _has_force_stop_required_error(terminal):
            raise InstanceError(
                "This tab has an ambiguous foreground; use the exact-name force stop"
            )
        if (
            (terminal.get("kind") or "aeon") != "terminal"
            or int(terminal.get("shell_backed") or 0) != 1
        ):
            raise InstanceError("Starting an agent requires a managed terminal tab")
        if terminal.get("desired_state") != "running":
            raise InstanceError("The terminal tab is not running")
        pane = self._pane_info(terminal["tmux_name"])
        if not pane or pane["dead"]:
            raise InstanceError("The terminal tab is not running")
        if not self._refresh_managed_shell_prompt(terminal):
            raise InstanceError(
                "Return the terminal to its shell prompt before starting an agent"
            )
        workspace = self._pane_current_directory(terminal["tmux_name"])
        previous_agent_kind = terminal.get("last_agent_kind")
        try:
            command, instruction_snapshot, agent_setting = self._prepare_shell_agent_command(
                terminal,
                kind,
                workspace=workspace,
                preserve_profile=previous_agent_kind == kind,
            )
        except InstanceError:
            # Provider/auth/instruction preparation is deliberately complete
            # before the mode transaction, so a failed cross-kind attempt cannot
            # clear a profile or lie about last_agent_kind.
            raise

        # Explicit activation abandons any unsubmitted shell input, then types
        # only a server-built fixed command. Browser data controls neither argv
        # nor cwd, and prompt bodies live only in private files.
        self._detach_session_clients(terminal)
        pane = self._pane_info(terminal["tmux_name"])
        if not self._pane_at_base_prompt(terminal, pane):
            raise InstanceError(
                "The terminal left its managed prompt while the agent was prepared"
            )
        self._write_pending_activation(
            terminal,
            target_kind=kind,
            workspace=str(workspace),
            previous_agent_kind=previous_agent_kind,
            agent_model=agent_setting["desired_model"],
            agent_effort=agent_setting["desired_effort"],
            phase="prepared",
        )
        try:
            self.store.update_instance(
                terminal_id,
                status="starting",
                desired_state="running",
                last_error="",
            )
        except Exception:
            self._clear_pending_activation(terminal)
            raise
        if not self._clear_shell_prompt_marker(terminal):
            self._clear_pending_activation(terminal)
            self.store.update_instance(terminal_id, status="running")
            raise InstanceError(
                "The terminal prompt changed while the agent was being prepared"
            )
        try:
            self._write_pending_activation(
                terminal,
                target_kind=kind,
                workspace=str(workspace),
                previous_agent_kind=previous_agent_kind,
                agent_model=agent_setting["desired_model"],
                agent_effort=agent_setting["desired_effort"],
                phase="command_sent",
            )
        except Exception:
            # The exact base shell still owns the foreground; ask it for a fresh
            # prompt marker, then abandon the activation without sending argv.
            self._tmux(
                "send-keys",
                "-t",
                self._pane_target(terminal["tmux_name"]),
                "C-c",
            )
            self._clear_pending_activation(terminal)
            self.store.update_instance(terminal_id, status="running")
            raise
        command_text = shlex.join(command)
        if not self._paste_private_tmux_buffer(
            terminal, f"{command_text}\r", label="agent"
        ):
            self._handle_failed_agent_delivery(
                terminal,
                target_kind=kind,
                workspace=workspace,
                previous_agent_kind=previous_agent_kind,
            )
        started = None
        deadline = time.monotonic() + AGENT_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            current = self._pane_info(terminal["tmux_name"])
            if current is None or current["dead"]:
                break
            if self._record_managed_agent_foreground(terminal, current):
                started = current
                break
            time.sleep(0.05)
        if started is None:
            current = self._pane_info(terminal["tmux_name"])
            if self._pane_at_base_prompt(terminal, current):
                self._clear_managed_agent_identity(terminal)
                self._clear_pending_activation(terminal)
                self.store.update_instance(
                    terminal_id,
                    status="running",
                    desired_state="running",
                    last_error=f"The {kind} command exited before agent mode"[:500],
                )
                raise InstanceLaunchError(
                    f"The {kind} command exited before the terminal entered agent mode",
                    launched=False,
                )
            if current is not None and not current["dead"]:
                message = _force_stop_required_error(
                    f"the {kind} command started, but its foreground process "
                    "identity could not be recorded"
                )
                try:
                    self.store.transition_shell_mode(
                        terminal_id,
                        expected_kind="terminal",
                        kind=kind,
                        workspace=str(workspace),
                        last_agent_kind=kind,
                        clear_profile=bool(
                            previous_agent_kind and previous_agent_kind != kind
                        ),
                        status="error",
                        last_error=message,
                    )
                    self._clear_pending_activation(terminal)
                except Exception as exc:
                    # Retain an explicit agent/error record even if the richer
                    # transactional binding update failed. This is safer than a
                    # terminal-mode lie while an unknown foreground is alive.
                    self.store.update_instance(
                        terminal_id,
                        kind=kind,
                        last_agent_kind=kind,
                        status="error",
                        desired_state="running",
                        last_error=message[:500],
                    )
                    # Retain the journal: the non-transactional fallback did
                    # not prove that an incompatible cross-kind profile was
                    # cleared. Reconciliation will idempotently finish it.
                    raise InstanceLaunchError(message, launched=True) from exc
                raise InstanceLaunchError(message, launched=True)
            self._clear_managed_agent_identity(terminal)
            self._clear_pending_activation(terminal)
            self.store.update_instance(
                terminal_id,
                status="error",
                desired_state="stopped",
                last_error=f"The managed shell exited while {kind} was starting"[:500],
            )
            raise InstanceLaunchError(
                f"The managed shell exited while {kind} was starting",
                launched=False,
            )
        try:
            transitioned = self.store.transition_shell_mode(
                terminal_id,
                expected_kind="terminal",
                kind=kind,
                workspace=str(workspace),
                last_agent_kind=kind,
                clear_profile=bool(
                    previous_agent_kind and previous_agent_kind != kind
                ),
                status="running",
            )
        except Exception as exc:
            # A non-manager writer changed the row despite the lifecycle lock.
            # Interrupt only the exact PGID launched above; never type text.
            if self._managed_agent_is_foreground(terminal, started):
                self._tmux(
                    "send-keys",
                    "-t",
                    self._pane_target(terminal["tmux_name"]),
                    "C-c",
                )
            try:
                current = self._pane_info(terminal["tmux_name"])
                if self._pane_at_base_prompt(terminal, current):
                    self._clear_managed_agent_identity(terminal)
                    self.store.update_instance(
                        terminal_id,
                        status="running",
                        last_error="Agent launch was interrupted after a registry failure"[:500],
                    )
                else:
                    self.store.update_instance(
                        terminal_id,
                        kind=kind,
                        last_agent_kind=kind,
                        status="error",
                        desired_state="running",
                        last_error=_force_stop_required_error(
                            "the agent is still running after a registry failure"
                        ),
                    )
            except Exception:
                pass
            raise InstanceLaunchError(
                "The agent started but its durable mode transition failed",
                launched=True,
            ) from exc
        try:
            self.store.mark_agent_setting_applied(
                terminal_id,
                kind,
                model=agent_setting["desired_model"],
                effort=agent_setting["desired_effort"],
            )
        except Exception as exc:
            # The exact foreground and mode transition are already proven. Keep
            # the immutable activation journal; reconciliation can safely retry
            # this idempotent settings commit without stopping the agent.
            self.store.update_instance(
                terminal_id,
                status="error",
                last_error=(
                    "Agent started, but its applied model/effort state could not "
                    "be recorded"
                )[:500],
            )
            raise InstanceLaunchError(
                "The agent started but its applied settings were not recorded",
                launched=True,
            ) from exc
        self._clear_pending_activation(transitioned)
        self.store.update_instance(
            terminal_id,
            status="running",
            last_started_at=time.time(),
            last_error="",
        )
        if (
            instruction_snapshot is not None
            and kind in {"aeon", "codex", "claude", "grok"}
        ):
            try:
                self.instruction_service.mark_applied(
                    terminal_id,
                    profile_version_id=instruction_snapshot["profile_version_id"],
                    local_revision=instruction_snapshot["local_revision"],
                )
            except InstructionProfileError as exc:
                self.store.update_instance(
                    terminal_id,
                    last_error=f"Agent started; instruction state update failed: {exc}"[:500],
                )
        self.store.audit(
            "terminal_agent_started",
            actor=actor,
            instance_id=terminal_id,
            client_ip=client_ip,
            details={"kind": kind, "workspace": str(workspace)},
        )
        return self.get_instance(terminal_id)

    def start_aeon_here(
        self,
        terminal_id: str,
        *,
        actor: str,
        client_ip: str = "",
    ) -> dict:
        """Backward-compatible alias for same-tab Aeon activation."""
        return self.activate_agent(
            terminal_id,
            kind="aeon",
            actor=actor,
            client_ip=client_ip,
        )

    def end_agent(
        self, instance_id: str, *, actor: str, client_ip: str = ""
    ) -> dict:
        """Serialize one tab's agent-to-terminal transition."""

        with self._lifecycle_lock(instance_id):
            return self._end_agent_locked(
                instance_id, actor=actor, client_ip=client_ip
            )

    def _end_agent_locked(
        self, instance_id: str, *, actor: str, client_ip: str = ""
    ) -> dict:
        """Gracefully return one shell-backed agent tab to its Bash prompt."""

        record = self.store.get_instance(instance_id)
        if not record:
            raise InstanceError("Unknown session")
        if _has_force_stop_required_error(record):
            raise InstanceError(
                "The foreground identity is ambiguous; use the exact-name force stop"
            )
        kind = record.get("kind") or "aeon"
        if (
            int(record.get("shell_backed") or 0) != 1
            or kind not in AGENT_INSTANCE_KINDS
        ):
            raise InstanceError("This tab is not in agent mode")
        self._detach_session_clients(record)
        pane = self._pane_info(record["tmux_name"])
        if not pane or pane["dead"]:
            raise InstanceError("The agent tab is not running")
        if self._pane_at_base_prompt(record, pane):
            try:
                self.store.transition_shell_mode(
                    instance_id,
                    expected_kind=kind,
                    kind="terminal",
                    status="running",
                )
            except ValueError as exc:
                raise InstanceError(str(exc)) from exc
            self._clear_managed_agent_identity(record)
            return self.get_instance(instance_id)
        if not self._managed_agent_is_foreground(record, pane):
            self.store.update_instance(
                instance_id,
                status="error",
                desired_state="running",
                last_error=_force_stop_required_error(
                    "the foreground process is not the recorded managed agent"
                ),
            )
            raise InstanceError(
                "The foreground process is not the managed agent; return it to the "
                "shell or use the exact-name force stop"
            )
        self.store.update_instance(instance_id, status="stopping", last_error="")
        self._tmux(
            "send-keys", "-t", self._pane_target(record["tmux_name"]), "C-c"
        )
        started_wait = time.monotonic()
        deadline = started_wait + AGENT_END_TIMEOUT_SECONDS
        returned = False
        ambiguous = False
        second_interrupt_sent = False
        while time.monotonic() < deadline:
            current = self._pane_info(record["tmux_name"])
            if current is None or current["dead"]:
                break
            if self._pane_at_base_prompt(record, current):
                returned = True
                break
            if not self._managed_agent_is_foreground(record, current):
                ambiguous = True
                break
            if (
                kind == "aeon"
                and not second_interrupt_sent
                and time.monotonic() - started_wait
                >= AGENT_SECOND_INTERRUPT_DELAY_SECONDS
            ):
                # Aeon's first SIGINT requests a checkpointed pause. A second
                # control signal may finish that known managed PGID; no terminal
                # text is ever injected into an agent/tool subprocess.
                self._tmux(
                    "send-keys",
                    "-t",
                    self._pane_target(record["tmux_name"]),
                    "C-c",
                )
                second_interrupt_sent = True
            time.sleep(0.1)
        if returned:
            try:
                self.store.transition_shell_mode(
                    instance_id,
                    expected_kind=kind,
                    kind="terminal",
                    status="running",
                )
            except ValueError as exc:
                raise InstanceError(str(exc)) from exc
            self._clear_managed_agent_identity(record)
        else:
            message = (
                _force_stop_required_error(
                    "the managed agent yielded to a different foreground process; "
                    "no further signal was sent"
                )
                if ambiguous
                else "The managed agent is still running; End can be requested again"
            )
            self.store.update_instance(
                instance_id,
                status="error" if ambiguous else "running",
                desired_state="running",
                last_error=message[:500],
            )
        self.store.audit(
            "terminal_agent_end_requested",
            actor=actor,
            instance_id=instance_id,
            client_ip=client_ip,
            details={"kind": kind, "returned_to_terminal": returned},
        )
        return self.get_instance(instance_id)

    def resume_instance(self, instance_id: str, *, actor: str, client_ip="") -> dict:
        with self._lifecycle_lock(instance_id):
            return self._resume_instance_locked(
                instance_id, actor=actor, client_ip=client_ip
            )

    def _resume_instance_locked(
        self, instance_id: str, *, actor: str, client_ip=""
    ) -> dict:
        record = self.store.get_instance(instance_id)
        if not record:
            raise InstanceError("Unknown session")
        record = self._normalize_dormant_project_manager(record)
        pane = self._pane_info(record["tmux_name"])
        if pane and not pane["dead"]:
            raise InstanceError("Instance is already running")
        if (
            int(record.get("shell_backed") or 0) == 1
            and (record.get("kind") or "aeon") in AGENT_INSTANCE_KINDS
        ):
            try:
                record = self.store.transition_shell_mode(
                    instance_id,
                    expected_kind=record.get("kind") or "aeon",
                    kind="terminal",
                    status="created",
                )
            except ValueError as exc:
                raise InstanceError(str(exc)) from exc
            self._clear_managed_agent_identity(record)
        kind = record.get("kind") or "aeon"
        if not int(record.get("shell_backed") or 0) and kind in PROVIDER_IDS:
            try:
                status = provider_status(kind)
            except ProviderError as exc:
                raise InstanceError(str(exc)) from exc
            if status.get("installed") is not True:
                raise InstanceError(f"The official {kind} CLI is not installed")
            if kind in {"codex", "claude"} and status.get("connected") is not True:
                raise InstanceError(
                    f"Connect {kind} in Settings before resuming this legacy session"
                )
        self._launch_record(
            record,
            resume=not is_first_project_manager_activation(record),
        )
        self.store.audit(
            "instance_resumed", actor=actor, instance_id=instance_id, client_ip=client_ip
        )
        return self.get_instance(instance_id)

    def graceful_stop(self, instance_id: str, *, actor: str, client_ip="") -> dict:
        with self._lifecycle_lock(instance_id):
            return self._graceful_stop_locked(
                instance_id, actor=actor, client_ip=client_ip
            )

    def _graceful_stop_locked(
        self, instance_id: str, *, actor: str, client_ip=""
    ) -> dict:
        record = self.store.get_instance(instance_id)
        if not record:
            raise InstanceError("Unknown session")
        if _has_force_stop_required_error(record):
            raise InstanceError(
                "The foreground identity is ambiguous; use the exact-name force stop"
            )
        kind = record.get("kind") or "aeon"
        if (
            int(record.get("shell_backed") or 0) == 1
            and kind in AGENT_INSTANCE_KINDS
        ):
            raise InstanceError(
                "Use End agent to return this tab safely to its managed terminal"
            )
        pane = self._pane_info(record["tmux_name"])
        if pane is not None:
            self._detach_session_clients(record)
        if pane and not pane["dead"]:
            if (
                kind == "terminal"
                and int(record.get("shell_backed") or 0) == 1
                and self._pane_at_base_prompt(record, pane)
                and not self._clear_shell_prompt_marker(record)
            ):
                raise InstanceError(
                    "The terminal prompt changed before it could be stopped"
                )
        self.store.update_instance(
            instance_id, desired_state="stopped", status="stopping", last_error=""
        )
        if kind in PROVIDER_IDS or kind in PROVIDER_AUTH_KINDS:
            # Direct legacy provider/login panes have no managed base shell.
            # Offer SIGINT, then remove and independently prove absence of the
            # exact session before claiming a stopped state.
            if pane and not pane["dead"]:
                self._tmux(
                    "send-keys", "-t", self._pane_target(record["tmux_name"]), "C-c"
                )
                time.sleep(0.25)
            current = self._pane_info(record["tmux_name"])
            if current is not None:
                self._kill_session_and_verify_absent(
                    record,
                    error_message="The exact provider session could not be stopped",
                )
            self.store.update_instance(instance_id, status="stopped")
        elif pane and not pane["dead"]:
            self._tmux(
                "send-keys", "-t", self._pane_target(record["tmux_name"]), "C-c"
            )
            if kind == "terminal" and int(record.get("shell_backed") or 0) == 1:
                # Never type into an arbitrary foreground program. Wait for the
                # per-launch marker plus exact base PID/foreground pgrp proof.
                deadline = time.monotonic() + TERMINAL_RETURN_TIMEOUT_SECONDS
                current = None
                while time.monotonic() < deadline:
                    current = self._pane_info(record["tmux_name"])
                    if current is None or current["dead"]:
                        break
                    if self._pane_at_base_prompt(record, current):
                        break
                    time.sleep(0.05)
                if current and self._pane_at_base_prompt(record, current):
                    if not self._clear_shell_prompt_marker(record):
                        raise InstanceError(
                            "The terminal prompt changed before it could be stopped"
                        )
                    self._tmux(
                        "send-keys",
                        "-t",
                        self._pane_target(record["tmux_name"]),
                        "-l",
                        "exit",
                    )
                    self._tmux(
                        "send-keys",
                        "-t",
                        self._pane_target(record["tmux_name"]),
                        "Enter",
                    )
        else:
            self.store.update_instance(instance_id, status="stopped")
        self.store.audit(
            "instance_stop_requested",
            actor=actor,
            instance_id=instance_id,
            client_ip=client_ip,
        )
        return self.get_instance(instance_id)

    def force_stop(
        self, instance_id: str, *, confirmation: str, actor: str, client_ip=""
    ) -> dict:
        with self._lifecycle_lock(instance_id):
            return self._force_stop_locked(
                instance_id,
                confirmation=confirmation,
                actor=actor,
                client_ip=client_ip,
            )

    def _force_stop_locked(
        self, instance_id: str, *, confirmation: str, actor: str, client_ip=""
    ) -> dict:
        record = self.store.get_instance(instance_id)
        if not record:
            raise InstanceError("Unknown session")
        if confirmation != record["name"]:
            raise InstanceError("Confirmation must exactly match the instance name")
        pane = self._pane_info(record["tmux_name"])
        if pane:
            self._kill_session_and_verify_absent(
                record,
                error_message="The exact tmux session could not be force stopped",
            )
        self.store.update_instance(
            instance_id,
            desired_state="stopped",
            status="stopped",
            last_error="",
        )
        if int(record.get("shell_backed") or 0):
            self._clear_managed_agent_identity(record)
            self._clear_pending_activation(record)
        self.store.audit(
            "instance_force_stopped",
            actor=actor,
            instance_id=instance_id,
            client_ip=client_ip,
        )
        return self.get_instance(instance_id)

    def delete_instance(
        self, instance_id: str, *, confirmation: str, actor: str, client_ip=""
    ) -> None:
        try:
            reject_project_manager_deletion(instance_id)
        except ProjectManagerProtectedError as exc:
            raise InstanceError(str(exc)) from exc
        with self._lifecycle_lock(instance_id):
            self._delete_instance_locked(
                instance_id,
                confirmation=confirmation,
                actor=actor,
                client_ip=client_ip,
            )

    def _delete_instance_locked(
        self, instance_id: str, *, confirmation: str, actor: str, client_ip=""
    ) -> None:
        try:
            reject_project_manager_deletion(instance_id)
        except ProjectManagerProtectedError as exc:
            raise InstanceError(str(exc)) from exc
        record = self.store.get_instance(instance_id)
        if not record:
            raise InstanceError("Unknown session")
        if confirmation != record["name"]:
            raise InstanceError("Confirmation must exactly match the instance name")
        pane = self._pane_info(record["tmux_name"])
        if pane and not pane["dead"]:
            raise InstanceError("Stop the running instance before deleting its tab")
        if pane:
            self._kill_session_and_verify_absent(
                record,
                error_message="The stopped tmux session could not be removed",
            )
        self.store.delete_instance(instance_id)
        self.store.audit(
            "instance_deleted",
            actor=actor,
            instance_id=instance_id,
            client_ip=client_ip,
            details={"name": record["name"]},
        )

    def capture_pane(self, instance_id: str, lines=2000) -> bytes:
        record = self.store.get_instance(instance_id)
        if not record:
            raise InstanceError("Unknown session")
        result = self._tmux(
            "capture-pane",
            "-ep",
            "-t",
            self._pane_target(record["tmux_name"]),
            "-S",
            f"-{max(100, min(int(lines), 10000))}",
        )
        if result.returncode != 0:
            return b""
        return result.stdout.encode("utf-8", errors="replace")

    def prepare_terminal_attachment(
        self, instance_id: str
    ) -> tuple[list[str], dict, bytes, int]:
        """Atomically validate and prepare one output-only tmux attachment."""

        with self._mode_lock(instance_id):
            args, environment = self.tmux_attach_args(instance_id)
            initial = self.capture_pane(instance_id)
            with self._mode_locks_guard:
                generation = self._transition_generations.get(instance_id, 0)
            return args, environment, initial, generation

    def tmux_attach_args(self, instance_id: str) -> tuple[list[str], dict]:
        record = self.store.get_instance(instance_id)
        if not record:
            raise InstanceError("Unknown session")
        pane = self._pane_info(record["tmux_name"])
        if not pane:
            raise InstanceError("This terminal is not currently available")
        env = dict(os.environ)
        env.pop("TMUX", None)
        env["TERM"] = BROWSER_TERMINAL_TERM
        env["COLORTERM"] = TRUECOLOR_TERM
        return (
            [
                self.config.tmux_binary,
                "set-option",
                "-t",
                self._pane_target(record["tmux_name"]),
                "status",
                "off",
                ";",
                "attach-session",
                "-f",
                "read-only",
                "-t",
                self._session_target(record["tmux_name"]),
            ],
            env,
        )

    @staticmethod
    def _process_resources(pid: int | None):
        if not pid:
            return None
        try:
            root = psutil.Process(pid)
            processes = [root, *root.children(recursive=True)]
            return {
                "rss_bytes": sum(proc.memory_info().rss for proc in processes if proc.is_running()),
                "processes": len(processes),
                "cpu_seconds": round(
                    sum(
                        proc.cpu_times().user + proc.cpu_times().system
                        for proc in processes
                        if proc.is_running()
                    ),
                    2,
                ),
            }
        except (psutil.Error, OSError):
            return None

    def resource_snapshot(self) -> dict:
        memory = psutil.virtual_memory()
        disk_target = self.config.state_dir if self.config.state_dir.exists() else Path.home()
        disk = psutil.disk_usage(disk_target)
        return {
            "host": {
                "hostname": socket.gethostname(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "load": list(os.getloadavg()),
                "memory_total": memory.total,
                "memory_used": memory.used,
                "memory_percent": memory.percent,
                "disk_total": disk.total,
                "disk_used": disk.used,
                "disk_percent": disk.percent,
            },
            "gpus": self._gpu_snapshot(),
        }

    def _gpu_snapshot(self) -> list[dict]:
        now = time.monotonic()
        with self._gpu_lock:
            cached_at, cached = self._gpu_cache
            if now - cached_at < 10:
                return cached
            if socket.gethostname() != self.config.expected_coordinator_host:
                return []
            if not self.config.coordinator_path.is_file() or not self.config.coordinator_cwd.is_dir():
                return []
            try:
                result = subprocess.run(
                    [
                        self.config.python_executable,
                        str(self.config.coordinator_path),
                        "status",
                        "--json",
                    ],
                    cwd=self.config.coordinator_cwd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                raw = json.loads(result.stdout)
                safe = [
                    {
                        "host": item.get("host"),
                        "gpu": item.get("physical_gpu"),
                        "state": item.get("state"),
                        "model": item.get("model"),
                        "memory_total_mib": item.get("memory_total_mib"),
                        "memory_used_mib": item.get("memory_used_mib"),
                        "memory_free_mib": item.get("memory_free_mib"),
                        "safely_allocatable_mib": item.get("vram_share_capacity_mib"),
                        "utilization_pct": item.get("utilization_pct"),
                        "temperature_c": item.get("temperature_c"),
                        "lease_count": len(item.get("claims") or []),
                    }
                    for item in raw
                ]
            except (OSError, subprocess.SubprocessError, ValueError, TypeError):
                safe = []
            self._gpu_cache = (now, safe)
            return safe
