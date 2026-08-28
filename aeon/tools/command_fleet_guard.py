"""Fail-closed Fleet admission for Aeon's generic host shell tools.

``run_command`` and ``run_command_async`` are useful for ordinary host work, but
they are not GPU allocators.  This module is the single execution-side boundary
that rejects recognizable direct GPU inventory, selection, control, and launch
forms before either tool creates a process (or, for a background job, any job
state).  It intentionally does not accept a command or profile from the model as
proof of a Fleet lease: GPU work must enter through Fleet Compute's reviewed
service or batch-profile APIs instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
from typing import IO, Any, Iterable, Mapping
import uuid


FLEET_LOW_PRIORITY = Path("/home/aday/bin/fleet-low-priority")
SYSTEMD_RUN = Path("/usr/bin/systemd-run")
SYSTEMCTL = Path("/usr/bin/systemctl")
SYSTEM_PYTHON = Path("/usr/bin/python3")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_EXEC = PROJECT_ROOT / "aeon" / "scripts" / "command_service_exec.py"
SERVICE_CONTROLLER = PROJECT_ROOT / "aeon" / "scripts" / "command_service_controller.py"
GPU_COORDINATOR = Path("/home/aday/website_hosting/gpu_coord.py")
CPU_SANDBOX_SLICE_ENV = "AEON_CPU_SANDBOX_SLICE"
CPU_SANDBOX_SLICE_RE = re.compile(r"^aeon_subagent_[0-9a-f]{32}\.slice$")
SERVICE_NAME_RE = re.compile(r"^aeon-command-[0-9a-f]{32}\.service$")
INVOCATION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SERVICE_MARKER_PREFIX = "AEON_COMMAND_SANDBOX_GATED_V1"
SERVICE_RECEIPT_SCHEMA = 1
SERVICE_GATE_TIMEOUT = 10.0
SERVICE_STOP_TIMEOUT = 10.0


def _launch_source_digest(path: Path) -> str:
    """Hash a small trusted bootstrap source without executing or importing it."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


# The parent process and each transient service must execute one coherent
# command-sandbox protocol.  A long-running Aeon otherwise keeps the old Python
# controller in memory while systemd starts a newly edited bootstrap script from
# disk, producing an opaque failure loop until the agent is restarted.
_SERVICE_EXEC_DIGEST_AT_IMPORT = _launch_source_digest(SERVICE_EXEC)
_SERVICE_CONTROLLER_DIGEST_AT_IMPORT = _launch_source_digest(SERVICE_CONTROLLER)

# The host's user manager accepts IPAddressDeny= but does not enforce it.  The
# generic shell therefore has no networking at all: seccomp denies socket and
# socketpair, while this deliberately useless address-family allow-list is an
# independent AF_UNIX/AF_INET/AF_INET6 denial.  Browser/provider tools own the
# reviewed network-capable boundaries instead.
RESTRICTED_ADDRESS_FAMILIES = ("AF_NETLINK",)
DENIED_SOCKET_SYSCALLS = ("socket", "socketpair")

# These are executable safety boundaries, not model-selected files.  A generic
# command may edit normal project files, but it cannot rewrite the allocator,
# command sandbox, delegation boundary, or Comfy/Fleet adapters that constrain
# later tool calls. ProtectSystem=strict covers them outside the writable cwd;
# ReadOnlyPaths keeps the same guarantee when cwd contains this source tree.
_TRUSTED_GUARDRAIL_CANDIDATES = (
    Path("/home/aday/AGENTS.md"),
    Path("/home/aday/.codex/AGENTS.md"),
    PROJECT_ROOT.parent / "AGENTS.md",
    PROJECT_ROOT / "AGENTS.md",
    # The whole Aeon source tree (including .git and import-hook locations) is
    # read-only to an opaque shell.  Source edits use Aeon's guarded file tools.
    PROJECT_ROOT,
    PROJECT_ROOT / "aeon" / "tools" / "command_fleet_guard.py",
    PROJECT_ROOT / "aeon" / "tools" / "system.py",
    PROJECT_ROOT / "aeon" / "tools" / "jobs.py",
    PROJECT_ROOT / "aeon" / "core" / "tool_resources.py",
    PROJECT_ROOT / "aeon" / "core" / "fleet_backend.py",
    PROJECT_ROOT / "aeon" / "core" / "sub_agent_environment.py",
    PROJECT_ROOT / "aeon" / "core" / "sub_agent_state.py",
    PROJECT_ROOT / "aeon" / "scripts" / "sub_agent_wrapper.py",
    PROJECT_ROOT / "aeon" / "tools" / "sub_agent.py",
    PROJECT_ROOT / "aeon" / "core" / "comfy_fleet_adapter.py",
    PROJECT_ROOT / "aeon" / "core" / "video_comfy_fleet_adapter.py",
    PROJECT_ROOT / "aeon" / "core" / "video_comfy_release.py",
    PROJECT_ROOT / "aeon" / "tools" / "generate_image.py",
    PROJECT_ROOT / "aeon" / "tools" / "generate_video.py",
    SERVICE_EXEC,
    SERVICE_CONTROLLER,
)

_INACCESSIBLE_PATH_CANDIDATES = (
    Path("/run/docker.sock"),
    Path("/var/run/docker.sock"),
    Path("/run/containerd"),
    Path(f"/run/user/{os.getuid()}/docker.sock"),
    Path(f"/run/user/{os.getuid()}/podman/podman.sock"),
    Path(f"/run/user/{os.getuid()}/bus"),
    Path(f"/run/user/{os.getuid()}/systemd/private"),
    Path("/run/dbus/system_bus_socket"),
    GPU_COORDINATOR,
    # Same-UID shell payloads must not read Aeon service/broker state or local
    # subscription/API credential stores. Network-capable provider tools own
    # those exact paths and inject only their reviewed capabilities.
    Path("/home/aday/.aeon"),
    Path("/home/aday/.codex"),
    Path("/home/aday/.claude"),
    Path("/home/aday/.ssh"),
    Path("/home/aday/.aws"),
    Path("/home/aday/.kube"),
    Path("/home/aday/.docker"),
    Path("/home/aday/.config/gh"),
    Path("/home/aday/.config/gcloud"),
    Path("/home/aday/.config/gemini"),
    Path("/home/aday/.config/google-gemini"),
    Path("/home/aday/.config/grok"),
    Path("/home/aday/.config/anthropic"),
    Path("/home/aday/.local/share/keyrings"),
    Path("/home/aday/.netrc"),
    Path("/home/aday/.git-credentials"),
    Path("/home/aday/.npmrc"),
    Path("/home/aday/.pypirc"),
)

_SERVICE_SHOW_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "Type",
    "MainPID",
    "InvocationID",
    "ControlGroup",
    "Slice",
    "DevicePolicy",
    "RestrictAddressFamilies",
    "SystemCallFilter",
    "SystemCallErrorNumber",
    "PrivateTmp",
    "ProtectSystem",
    "ProtectHome",
    "ReadWritePaths",
    "ReadOnlyPaths",
    "InaccessiblePaths",
    "NoNewPrivileges",
    "RestrictNamespaces",
    "RestrictSUIDSGID",
    "LockPersonality",
    "KillMode",
    "SendSIGKILL",
    "CollectMode",
    "WorkingDirectory",
    "RuntimeMaxUSec",
    "TimeoutStopUSec",
)

_NO_ACCELERATOR_ENV = {
    "CUDA_VISIBLE_DEVICES": "void",
    "GPU_DEVICE_ORDINAL": "-1",
    "HIP_VISIBLE_DEVICES": "-1",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "ROCR_VISIBLE_DEVICES": "-1",
}
_REMOVED_ACCELERATOR_ENV = {
    "AEON_CPU_SANDBOX_SLICE",
    "BASH_ENV",
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
    "CUDA_MPS_LOG_DIRECTORY",
    "CUDA_MPS_PIPE_DIRECTORY",
    "GPU_AGENT_CLAIM_ID",
    "GPU_MEM_LIMIT_GB",
    "ENV",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_REQUIRE_CUDA",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "ZDOTDIR",
}

_REMOVED_PAYLOAD_ENV = _REMOVED_ACCELERATOR_ENV | {
    "CONTAINER_HOST",
    "DBUS_SESSION_BUS_ADDRESS",
    "DOCKER_CONTEXT",
    "DOCKER_HOST",
    "KUBECONFIG",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "NOTIFY_SOCKET",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "SYSTEMD_EXEC_PID",
    "WATCHDOG_PID",
    "WATCHDOG_USEC",
}

_PROTECTED_GPU_ENV = {
    "AEON_GPU_MEM_UTIL",
    "AEON_TOOL_GPU_POLICY",
    "CUDA_VISIBLE_DEVICES",
    "GPU_AGENT_CLAIM_ID",
    "GPU_DEVICE_ORDINAL",
    "GPU_LEASE_EXCLUSIVE",
    "GPU_LEASE_ID",
    "GPU_LEASE_OWNER",
    "GPU_LEASE_RUN_DIR",
    "GPU_MEM_LIMIT_GB",
    "GPU_MEM_UTIL",
    "GPU_PLANNED_VRAM_GB",
    "GPU_RESERVE_GB",
    "HIP_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
}
_DIRECT_GPU_COMMANDS = {
    "nvidia-smi",
    "nvidia-container-cli",
    "nvidia-ctk",
    "nvidia-debugdump",
    "nvidia-settings",
    "nvtop",
}
_GPU_LAUNCHERS = {
    "deepspeed",
    "llama-server",
    "ollama",
    "text-generation-launcher",
    "torchrun",
    "tritonserver",
    "vllm",
}
_TEXT_INSPECTION_COMMANDS = {
    "cmp",
    "diff",
    "echo",
    "grep",
    "head",
    "printf",
    "tail",
    "test",
}
_CONTAINER_CONTROL_COMMANDS = {
    "apptainer",
    "buildah",
    "buildctl",
    "buildkitd",
    "containerd",
    "conmon",
    "crictl",
    "crun",
    "ctr",
    "docker",
    "docker-compose",
    "dockerd",
    "helm",
    "incus",
    "kubectl",
    "lxc",
    "microk8s",
    "nerdctl",
    "oc",
    "podman",
    "podman-compose",
    "podman-remote",
    "runc",
    "singularity",
    "youki",
}
_PROCESS_SIGNAL_COMMANDS = {
    "kill",
    "killall",
    "pkill",
    "skill",
}
_PROTECTED_BOUNDARY_COMMANDS = {
    "background_job_exec.py",
    "command_service_controller.py",
    "command_service_exec.py",
    "fleet-low-priority",
    "sub_agent_wrapper.py",
}
_READ_ONLY_SYSTEMCTL_ACTIONS = {
    "cat",
    "get-default",
    "help",
    "is-active",
    "is-enabled",
    "is-failed",
    "list-dependencies",
    "list-jobs",
    "list-sockets",
    "list-timers",
    "list-unit-files",
    "list-units",
    "show",
    "show-environment",
    "status",
}
_SYSTEMCTL_OPTIONS_WITH_SEPARATE_VALUE = {
    "--host",
    "--image",
    "--image-policy",
    "--job-mode",
    "--kill-value",
    "--kill-whom",
    "--lines",
    "--machine",
    "--output",
    "--property",
    "--root",
    "--runtime-scope",
    "--signal",
    "--state",
    "--timestamp",
    "--transport",
    "--type",
    "-H",
    "-M",
    "-n",
    "-o",
    "-p",
}
_CONTROL_SOCKET_RE = re.compile(
    r"(?i)(?:/(?:var/)?run/docker\.sock\b|/(?:var/)?run/containerd(?:/|\b)|"
    r"/(?:var/)?run/(?:crio|podman)(?:/|\b)|/run/user/[0-9]+/(?:bus\b|systemd/private\b))"
)
_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
_NVML_RE = re.compile(
    r"(?i)(?:\bpynvml\b|\bnvidia[_-]?ml(?:[_-]?py)?\b|\bnvml[A-Z_a-z0-9]*\b|"
    r"libnvidia-ml\.so)"
)
_GPU_CODE_RE = re.compile(
    r"(?i)(?:\btorch\.cuda\b|\bcupy\.cuda\b|\bnumba\.cuda\b|"
    r"\bjax\.(?:devices|local_devices)\s*\(|"
    r"tensorflow.*(?:list_physical_devices|set_visible_devices).*gpu|"
    r"\b(?:cuda|gpu)\s*:\s*[0-9]+)"
)
_COORDINATOR_CODE_RE = re.compile(
    r"(?i)(?:\baeon\.core\.(?:gpu|gpu_queue)\b|\b(?:reserve_named_lease|"
    r"detect_gpus|gpu_coord(?:inator)?|coordinator_client)\b)"
)
_REDIRECTION_RE = re.compile(r"^(?:[0-9]+)?(?:<|>|<<|>>|<<<|<>|>&|<&|&>).*$")


class FleetCommandGuardError(ValueError):
    """A command cannot safely be admitted by a generic shell tool."""


@dataclass(frozen=True)
class FleetShellBoundary:
    """Immutable launch contract for one gated transient user service."""

    systemd_run: str
    systemctl: str
    low_priority: str
    service_exec: str
    cwd: str
    cwd_device: int
    cwd_inode: int
    unit_name: str
    nonce: str
    control_dir: str
    scratch_dir: str
    runtime_max_seconds: int | None
    slice_name: str | None
    guardrail_paths: tuple[str, ...]
    writable_paths: tuple[str, ...]
    inaccessible_paths: tuple[str, ...]
    landlock_hidden_paths: tuple[str, ...]

    def properties(self) -> tuple[str, ...]:
        runtime = (
            "infinity"
            if self.runtime_max_seconds is None
            else f"{self.runtime_max_seconds}s"
        )
        values = [
            "Type=exec",
            "CollectMode=inactive-or-failed",
            "KillMode=control-group",
            "SendSIGKILL=yes",
            "TimeoutStopSec=5s",
            f"RuntimeMaxSec={runtime}",
            "DevicePolicy=closed",
            f"RestrictAddressFamilies={' '.join(RESTRICTED_ADDRESS_FAMILIES)}",
            f"SystemCallFilter=~{' '.join(DENIED_SOCKET_SYSCALLS)}",
            "SystemCallErrorNumber=EPERM",
            "PrivateTmp=yes",
            "ProtectSystem=strict",
            "ProtectHome=read-only",
            f"ReadWritePaths={' '.join(shlex.quote(path) for path in self.writable_paths)}",
            f"ReadOnlyPaths={' '.join(shlex.quote(path) for path in self.guardrail_paths)}",
            "InaccessiblePaths="
            + " ".join(shlex.quote("-" + path) for path in self.inaccessible_paths),
            "NoNewPrivileges=yes",
            "RestrictNamespaces=yes",
            "RestrictSUIDSGID=yes",
            "LockPersonality=yes",
            "UMask=0077",
            "Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "Environment=CUDA_VISIBLE_DEVICES=void",
            "Environment=GPU_DEVICE_ORDINAL=-1",
            "Environment=HIP_VISIBLE_DEVICES=-1",
            "Environment=NVIDIA_VISIBLE_DEVICES=void",
            "Environment=ROCR_VISIBLE_DEVICES=-1",
            "UnsetEnvironment=BASH_ENV ENV ZDOTDIR LD_AUDIT LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONINSPECT PYTHONPATH PYTHONSTARTUP GPU_AGENT_CLAIM_ID GPU_MEM_LIMIT_GB AEON_CPU_SANDBOX_SLICE DBUS_SESSION_BUS_ADDRESS DOCKER_HOST CONTAINER_HOST KUBECONFIG",
        ]
        if self.slice_name:
            values.append(f"Slice={self.slice_name}")
        return tuple(values)

    def argv(self, spec_path: str) -> list[str]:
        argv = [
            self.systemd_run,
            "--user",
            "--wait",
            "--pipe",
            "--collect",
            "--quiet",
            "--no-ask-password",
            f"--unit={self.unit_name}",
            "--service-type=exec",
            "--expand-environment=no",
            f"--working-directory={self.cwd}",
        ]
        for value in self.properties():
            argv.extend(("--property", value))
        argv.extend(
            (
                "--",
                "/bin/bash",
                self.low_priority,
                str(SYSTEM_PYTHON),
                "-I",
                self.service_exec,
                spec_path,
                str(Path(self.control_dir) / "gate"),
                self.nonce,
            )
        )
        return argv


@dataclass(frozen=True)
class SandboxServiceReceipt:
    """Durable identity used for every later lifecycle operation."""

    unit_name: str
    invocation_id: str
    control_group: str
    main_pid: int
    command_digest: str
    cwd: str
    control_dir: str
    scratch_dir: str
    slice_name: str | None
    slice_control_group: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SERVICE_RECEIPT_SCHEMA,
            "unit_name": self.unit_name,
            "invocation_id": self.invocation_id,
            "control_group": self.control_group,
            "main_pid": self.main_pid,
            "command_digest": self.command_digest,
            "cwd": self.cwd,
            "control_dir": self.control_dir,
            "scratch_dir": self.scratch_dir,
            "slice_name": self.slice_name,
            "slice_control_group": self.slice_control_group,
        }


@dataclass
class SandboxServiceHandle:
    boundary: FleetShellBoundary
    process: subprocess.Popen
    receipt: SandboxServiceReceipt
    initial_output: str = ""
    output_file: IO[str] | None = None


def _refuse(reason: str) -> None:
    raise FleetCommandGuardError(
        "COMMAND REFUSED BY FLEET COMPUTE POLICY: "
        f"{reason} Generic shell tools cannot inspect, select, claim, or launch "
        "local GPU compute. Submit durable demand through Fleet Compute using an "
        "enabled service/profile or a reviewed batch adapter; if no compatible "
        "profile exists, add and review one first. Do not invoke gpu_coord.py "
        "directly. No process or background-job directory was created."
    )


def _syntax_error() -> None:
    raise FleetCommandGuardError(
        "COMMAND REFUSED: the shell command is malformed or cannot be parsed "
        "safely. Correct its quoting, grouping, operators, and redirections "
        "before retrying. No process or background-job directory was created."
    )


def _validate_shell_grouping(command: str) -> None:
    """Conservatively catch incomplete grouping that ``shlex`` does not reject."""

    stack: list[str] = []
    quote = ""
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote == "'":
            if char == "'":
                quote = ""
            index += 1
            continue
        if char == "'" and not quote:
            quote = "'"
            index += 1
            continue
        if char == '"':
            quote = "" if quote == '"' else '"'
            index += 1
            continue

        if (
            not quote
            and char == "#"
            and (index == 0 or command[index - 1] in " \t\r\n;&|(")
        ):
            newline = command.find("\n", index + 1)
            if newline < 0:
                break
            index = newline + 1
            continue

        # Backticks are command substitutions in both unquoted and double-quoted
        # text.  Their historical escaping grammar is subtle, so an unmatched
        # delimiter is rejected here while shlex handles ordinary quotes.
        if char == "`":
            expected = "`"
            if stack and stack[-1] == expected:
                stack.pop()
            else:
                stack.append(expected)
            index += 1
            continue

        if command.startswith("$(", index):
            stack.append(")")
            index += 2
            continue
        if command.startswith("${", index):
            stack.append("}")
            index += 2
            continue
        if not quote and char == "(":
            stack.append(")")
            index += 1
            continue
        if char in ")}" and stack and stack[-1] == char:
            stack.pop()
            index += 1
            continue
        if not quote and char == ")":
            _syntax_error()
        index += 1

    if escaped or quote or stack:
        _syntax_error()


def _find_backtick_end(command: str, start: int) -> int:
    escaped = False
    for index in range(start, len(command)):
        char = command[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "`":
            return index
    _syntax_error()


def _find_substitution_end(command: str, start: int) -> int:
    depth = 1
    quote = ""
    escaped = False
    index = start
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote == "'":
            if char == "'":
                quote = ""
            index += 1
            continue
        if char == "'" and not quote:
            quote = "'"
            index += 1
            continue
        if char == '"':
            quote = "" if quote == '"' else '"'
            index += 1
            continue
        if char == "`":
            index = _find_backtick_end(command, index + 1) + 1
            continue
        if command.startswith("$(", index):
            depth += 1
            index += 2
            continue
        if not quote and char == "(":
            depth += 1
        elif not quote and char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    _syntax_error()


def _command_substitution_payloads(command: str) -> Iterable[str]:
    """Yield executable substitution bodies, excluding single-quoted literals."""

    quote = ""
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote == "'":
            if char == "'":
                quote = ""
            index += 1
            continue
        if char == "'" and not quote:
            quote = "'"
            index += 1
            continue
        if char == '"':
            quote = "" if quote == '"' else '"'
            index += 1
            continue
        if (
            not quote
            and char == "#"
            and (index == 0 or command[index - 1] in " \t\r\n;&|(")
        ):
            newline = command.find("\n", index + 1)
            if newline < 0:
                return
            index = newline + 1
            continue
        if char == "`":
            end = _find_backtick_end(command, index + 1)
            yield command[index + 1:end]
            index = end + 1
            continue
        if command.startswith("$(", index) and not command.startswith("$((", index):
            end = _find_substitution_end(command, index + 2)
            yield command[index + 2:end]
            index = end + 1
            continue
        index += 1


def _lex_shell(command: str) -> list[str]:
    _validate_shell_grouping(command)
    try:
        lexer = shlex.shlex(
            command,
            posix=True,
            punctuation_chars=";&|()<>\n",
        )
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
    except (TypeError, ValueError):
        _syntax_error()
    if not tokens:
        _syntax_error()

    first = tokens[0]
    last = tokens[-1]
    if first in {"&&", "||", "|"} or last in {"&&", "||", "|", "("}:
        _syntax_error()
    if _REDIRECTION_RE.fullmatch(last):
        _syntax_error()
    for left, right in zip(tokens, tokens[1:]):
        if left in {"&&", "||", "|"} and right in {"&&", "||", "|", ";"}:
            _syntax_error()
    if "&" in tokens:
        _refuse(
            "an untracked shell background operator was requested. Use "
            "run_command_async for ordinary tracked CPU background work."
        )
    return tokens


def _is_separator(token: str) -> bool:
    if token == "\n":
        return True
    return bool(token) and all(char in ";&|()" for char in token)


def _segments(tokens: Iterable[str]) -> Iterable[list[str]]:
    current: list[str] = []
    for token in tokens:
        if _is_separator(token):
            if current:
                yield current
                current = []
            continue
        current.append(token)
    if current:
        yield current


def _basename(word: str) -> str:
    return word.rstrip("/").rsplit("/", 1)[-1].lower()


def _head_index(words: list[str]) -> int | None:
    index = 0
    while index < len(words):
        word = words[index]
        if _ASSIGNMENT_RE.match(word):
            index += 1
            continue
        if word.isdigit() and index + 1 < len(words) and _REDIRECTION_RE.match(words[index + 1]):
            index += 3
            continue
        if _REDIRECTION_RE.match(word):
            # A standalone redirect consumes its following path.  Combined
            # redirects (``>file``) consume only this token.
            index += 1 if len(word.lstrip("0123456789")) > 1 else 2
            continue
        return index
    return None


def _protected_assignment(token: str) -> str:
    match = _ASSIGNMENT_RE.match(token)
    if not match:
        return ""
    name = match.group(1).upper()
    if _protected_environment_name(name):
        return name
    return ""


def _protected_environment_name(name: str) -> bool:
    upper = name.upper()
    resource_markers = ("CLAIM", "COORD", "GPU", "LEASE", "VRAM")
    return (
        upper in {
            "AEON_CPU_SANDBOX_SLICE", "BASH_ENV", "CONTAINER_HOST",
            "DOCKER_CONTEXT", "DOCKER_HOST",
            "ENV", "KUBECONFIG", "ZDOTDIR",
        }
        or upper in _PROTECTED_GPU_ENV
        or upper.startswith("CUDA_MPS")
        or upper.startswith("NVIDIA_")
        or upper.startswith("FLEET_")
        or upper.startswith("AEON_FLEET_")
        or (upper.startswith("SLURM_") and "GPU" in upper)
        or (
            upper.startswith(("AEON_", "GPU_", "QWEN_"))
            and any(marker in upper for marker in resource_markers)
        )
    )


def _python_module(words: list[str]) -> str:
    try:
        index = words.index("-m")
    except ValueError:
        return ""
    return words[index + 1].lower() if index + 1 < len(words) else ""


def _python_inline_code(words: list[str]) -> str:
    for option in ("-c", "-ec", "-ce"):
        try:
            index = words.index(option)
        except ValueError:
            continue
        return words[index + 1] if index + 1 < len(words) else ""
    return ""


def _next_cli_word(
    words: list[str], start: int, options_with_separate_value: set[str]
) -> int | None:
    index = start
    while index < len(words):
        word = words[index]
        if word == "--":
            return index + 1 if index + 1 < len(words) else None
        if word in options_with_separate_value:
            index += 2
            continue
        if word.startswith("-"):
            index += 1
            continue
        return index
    return None


def _systemctl_is_read_only(words: list[str], command_index: int) -> bool:
    tail = words[command_index + 1:]
    lowered = [word.lower() for word in tail]
    for original, word in zip(tail, lowered):
        if (
            original in {"-H", "-M"}
            or word in {"--host", "--machine", "--transport"}
            or word.startswith(("--host=", "--machine=", "--transport="))
        ):
            return False
        # Short remote options accept their target either as the next word or in
        # the same argv element (``-Hhost`` / ``-Mmachine``).
        if original.startswith(("-H", "-M")) and original not in {"-H", "-M"}:
            return False
    action_index = _next_cli_word(
        words, command_index + 1, _SYSTEMCTL_OPTIONS_WITH_SEPARATE_VALUE
    )
    if action_index is None:
        # No verb is systemctl's read-only list-units default; --help/--version
        # also land here because option words are skipped.
        return True
    return words[action_index].lower() in _READ_ONLY_SYSTEMCTL_ACTIONS


def _service_is_read_only(words: list[str], command_index: int) -> bool:
    tail = [word.lower() for word in words[command_index + 1:]]
    if not tail or set(tail).issubset({"--help", "--version", "-h", "-v"}):
        return True
    if tail == ["--status-all"]:
        return True
    positional = [word for word in tail if not word.startswith("-")]
    return len(positional) == 2 and positional[1] == "status"


def _dynamic_command_head(word: str) -> bool:
    """True when bash, rather than admission, will choose the executable."""

    return (
        word in {"$", "`"}
        or "$" in word
        or "`" in word
        or any(marker in word for marker in ("*", "?", "["))
    )


def _container_control_command(name: str) -> bool:
    return name in _CONTAINER_CONTROL_COMMANDS or name.startswith(
        ("containerd-shim", "docker-", "lxc-")
    )


def _inspect_segment(words: list[str]) -> None:
    head_index = _head_index(words)
    raw_head = words[head_index] if head_index is not None else ""
    head = _basename(raw_head) if head_index is not None else ""
    inspection_only = bool(head) and (
        head in _TEXT_INSPECTION_COMMANDS
        or (
            head == "rg"
            and not any(
                word == "--pre" or word.startswith("--pre=")
                for word in words[head_index + 1:]
            )
        )
    )

    # Shell-leading and ``env``/``sudo`` assignment forms can forge or bypass
    # the UUID/cap/claim environment that only a Fleet adapter may supply.
    assignment_context = head in {
        "declare", "env", "export", "local", "readonly", "sudo",
    }
    for index, word in enumerate(words):
        variable = _protected_assignment(word)
        if variable and (
            head_index is None or index < head_index or assignment_context
        ):
            _refuse(f"direct assignment to lease-controlled {variable} was requested.")

    if head_index is None:
        return

    if _dynamic_command_head(raw_head):
        _refuse(
            "a variable-, substitution-, or glob-resolved executable would bypass "
            "the reviewed command identity. Use one explicit CPU executable path."
        )

    if any("/dev/nvidia" in word.lower() for word in words):
        _refuse("direct access to an NVIDIA device node was requested.")
    if any(_CONTROL_SOCKET_RE.search(word) for word in words):
        _refuse(
            "direct access to a container or user-service control socket was requested."
        )

    basenames = [_basename(word) for word in words]
    lowered = [word.lower() for word in words]
    if any(name in _PROTECTED_BOUNDARY_COMMANDS for name in basenames):
        _refuse(
            "the trusted low-priority execution boundary cannot be invoked or "
            "modified by the command it is responsible for containing."
        )
    if not inspection_only and any(
        name in {
            "bwrap", "chroot", "coproc", "daemonize", "disown", "doas",
            "firejail", "machinectl", "nohup", "nsenter", "pkexec", "runuser",
            "setsid", "start-stop-daemon", "su", "sudo", "sudoedit",
            "systemd-run", "unshare",
        }
        for name in basenames
    ):
        _refuse(
            "a privilege, scope, or namespace launcher could escape the "
            "device-closed execution boundary."
        )
    if inspection_only:
        return
    if head == "rg" and any(
        word == "--pre" or word.startswith("--pre=")
        for word in words[head_index + 1:]
    ):
        _refuse("ripgrep --pre would execute an unreviewed command outside admission.")
    for index, name in enumerate(basenames):
        if name == "ssh":
            _refuse(
                "generic SSH execution is outside the local device-closed scope. "
                "Remote owner work must use the owning Fleet adapter."
            )
        if name == "systemctl" and not _systemctl_is_read_only(words, index):
            _refuse(
                "only local read-only systemctl status/show/list operations are "
                "allowed; lifecycle, signal, configuration, and remote operations "
                "would act outside the verified scope."
            )
        if name == "service" and not _service_is_read_only(words, index):
            _refuse(
                "only read-only service status operations are allowed; lifecycle "
                "and signal operations would act outside the verified scope."
            )
        if name in _PROCESS_SIGNAL_COMMANDS:
            _refuse(
                "a generic process-signal command cannot prove task ownership. "
                "Use kill_job or the owning lifecycle tool with its exact receipt."
            )
        if name in {"at", "batch"}:
            _refuse("a deferred scheduler would launch work outside the verified scope.")
        if name == "crontab":
            tail = lowered[index + 1:]
            read_only = "-l" in tail and not set(tail).intersection({"-e", "-r"})
            positional = [word for word in tail if not word.startswith("-")]
            if "-u" in tail:
                # Exactly one positional username is the only argument accepted
                # by the read-only ``crontab -u NAME -l`` form.
                read_only = read_only and len(positional) == 1
            else:
                read_only = read_only and not positional
            if not read_only:
                _refuse("a cron mutation could schedule work outside the verified scope.")
        if name == "busctl" and set(lowered[index + 1:]).intersection(
            {"call", "emit", "set-property"}
        ):
            _refuse("a D-Bus method call could launch work outside the verified scope.")
        if name == "dbus-send":
            _refuse("a D-Bus method call could launch work outside the verified scope.")
        if name == "gdbus" and "call" in lowered[index + 1:]:
            _refuse("a D-Bus method call could launch work outside the verified scope.")
        if name == "qdbus" and not set(lowered[index + 1:]).issubset(
            {"--help", "--version", "-h", "-v"}
        ):
            _refuse("a D-Bus method call could launch work outside the verified scope.")
    if any(_container_control_command(name) for name in basenames):
        _refuse(
            "a container/runtime client was requested. Its daemon or remote control "
            "plane runs outside the device-closed scope and could inspect or alter "
            "renter/unknown containers. Use an owning reviewed Fleet adapter."
        )
    if not inspection_only and any(name in _DIRECT_GPU_COMMANDS for name in basenames):
        _refuse("a direct NVIDIA/NVML inventory or control command was requested.")
    if not inspection_only and "gpu_coord.py" in basenames:
        _refuse("direct use of the coordinator API was requested.")
    if not inspection_only and any(
        name.startswith("nvml") or name.startswith("pynvml") for name in basenames
    ):
        _refuse("direct NVML inventory/control code was requested.")

    # Shell-within-shell still crosses this exact boundary. Inspect literal -c
    # payloads recursively so quoting a prohibited command does not bypass the
    # outer admission pass.
    for index, name in enumerate(basenames):
        if name not in {"bash", "dash", "sh", "zsh"}:
            continue
        option_index = index + 1
        while option_index < len(words) and words[option_index].startswith("-"):
            if "c" in words[option_index].lstrip("-"):
                if option_index + 1 >= len(words):
                    _syntax_error()
                guard_fleet_shell_command(words[option_index + 1])
                break
            option_index += 1
    if head == "eval":
        if head_index + 1 >= len(words):
            _syntax_error()
        guard_fleet_shell_command(" ".join(words[head_index + 1:]))

    python_like = head.startswith("python") or head in {"pypy", "pypy3"}
    if python_like:
        module = _python_module(words)
        inline = _python_inline_code(words)
        script_words = [word for word in words[head_index + 1:] if not word.startswith("-")]
        if (
            module.endswith("gpu_coord")
            or module in {"aeon.core.gpu", "aeon.core.gpu_queue"}
            or any(_basename(word) == "gpu_coord.py" for word in script_words)
            or _COORDINATOR_CODE_RE.search(inline)
        ):
            _refuse("direct use of the coordinator API was requested.")
        if module in {"pynvml", "nvidia_ml_py"} or _NVML_RE.search(inline):
            _refuse("direct NVML inventory/control code was requested.")
        if module in {
            "torch.distributed.launch",
            "torch.distributed.run",
            "deepspeed.launcher.runner",
            "vllm.entrypoints.openai.api_server",
        }:
            _refuse("a direct distributed or GPU runtime launch was requested.")
        if _GPU_CODE_RE.search(inline):
            _refuse("inline code would access a local GPU outside Fleet Compute.")
        if any("comfyui" in word.lower() and _basename(word) == "main.py" for word in words):
            _refuse("a direct ComfyUI GPU runtime launch was requested.")

    if not inspection_only and any(name in _GPU_LAUNCHERS for name in basenames):
        _refuse("a direct GPU or distributed-compute launcher was requested.")

    for index, name in enumerate(basenames):
        if name == "accelerate" and "launch" in lowered[index + 1:]:
            _refuse("an Accelerate GPU/distributed launch was requested.")
        if name == "ray":
            ray_actions = set(lowered[index + 1:])
            if ray_actions.intersection({"start", "up", "exec", "submit", "job"}):
                _refuse("a direct Ray cluster or workload launch was requested.")
        if name in {"srun", "sbatch", "mpirun", "mpiexec"}:
            remainder = " ".join(lowered[index + 1:])
            if re.search(r"(?:--gres(?:=|\s+)gpu|--gpus?(?:=|\s+)|gpu-bind|cuda)", remainder):
                _refuse("a scheduler/MPI GPU launch was requested.")

    return


def guard_fleet_shell_command(command: str) -> str:
    """Return a validated command or raise before any shell-side effects occur."""

    if not isinstance(command, str) or not command.strip() or "\x00" in command:
        _syntax_error()
    tokens = _lex_shell(command)
    for payload in _command_substitution_payloads(command):
        guard_fleet_shell_command(payload)
    for segment in _segments(tokens):
        _inspect_segment(segment)
    return command


def _verify_executable(path: Path, *, expected_uid: int, label: str) -> str:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FleetCommandGuardError(
            f"COMMAND REFUSED: the required {label} is unavailable "
            f"({type(exc).__name__}). The requested command was not launched and "
            "no background-job directory was created."
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != expected_uid
        or metadata.st_mode & 0o022
        or not os.access(path, os.X_OK)
    ):
        raise FleetCommandGuardError(
            f"COMMAND REFUSED: the required {label} failed its regular-file, "
            "ownership, permissions, or executable check. The requested command "
            "was not launched and no background-job directory was created."
        )
    return str(path)


def _verify_regular_file(path: Path, *, expected_uid: int, label: str) -> str:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FleetCommandGuardError(
            f"COMMAND REFUSED: the required {label} is unavailable "
            f"({type(exc).__name__}). The requested command was not launched."
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != expected_uid
        or metadata.st_mode & 0o022
    ):
        raise FleetCommandGuardError(
            f"COMMAND REFUSED: the required {label} failed its regular-file, "
            "ownership, or permissions check. The requested command was not launched."
        )
    return str(path)


def require_fleet_low_priority_wrapper() -> str:
    """Return the verified owner-work wrapper path, failing closed if it drifted."""

    return _verify_executable(
        FLEET_LOW_PRIORITY,
        expected_uid=os.getuid(),
        label="/home/aday/bin/fleet-low-priority wrapper",
    )


def scrubbed_fleet_command_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a copy with no inherited Fleet authority or visible accelerator."""

    environment = dict(os.environ if source is None else source)
    for name in list(environment):
        upper = name.upper()
        if (
            upper in _REMOVED_ACCELERATOR_ENV
            or _protected_environment_name(upper)
        ):
            environment.pop(name, None)
    environment.update(_NO_ACCELERATOR_ENV)
    return environment


def scrubbed_payload_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the explicit environment the gated shell receives.

    A transient user service otherwise inherits the user manager's environment,
    which may differ from Aeon's and may contain stale Fleet authority.  The
    trusted bootstrap therefore execs the payload with this explicit mapping.
    """

    environment = scrubbed_fleet_command_environment(source)
    for name in list(environment):
        if name.upper() in _REMOVED_PAYLOAD_ENV:
            environment.pop(name, None)
    environment.pop(CPU_SANDBOX_SLICE_ENV, None)
    environment.update(_NO_ACCELERATOR_ENV)
    return environment


def scrubbed_service_controller_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the tiny environment needed by the trusted async controller.

    The controller needs the user-manager bus and, when present, the exact
    receipted parent CPU slice. It never receives principal credentials, Fleet
    lease authority, loader hooks, provider variables, or arbitrary user env.
    """

    original = os.environ if source is None else source
    allowed = {
        "DBUS_SESSION_BUS_ADDRESS",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "TZ",
        "USER",
        "XDG_RUNTIME_DIR",
    }
    environment: dict[str, str] = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    }
    for name in allowed:
        value = original.get(name)
        if isinstance(value, str) and "\x00" not in value:
            environment[name] = value
    slice_name = _validated_slice(original)
    if slice_name is not None:
        environment[CPU_SANDBOX_SLICE_ENV] = slice_name
    return environment


def _canonical_existing_path(path: Path, *, label: str) -> str:
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise FleetCommandGuardError(
            f"COMMAND REFUSED: required {label} is unavailable "
            f"({type(exc).__name__}). The requested command was not launched."
        ) from exc
    if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
        raise FleetCommandGuardError(
            f"COMMAND REFUSED: required {label} is not a regular file/directory. "
            "The requested command was not launched."
        )
    return str(resolved)


def trusted_guardrail_paths() -> tuple[str, ...]:
    """Return the fixed canonical files that a generic command cannot rewrite."""

    candidates = list(_TRUSTED_GUARDRAIL_CANDIDATES)
    # protected_paths() is the canonical self-modification constitution. Its
    # human override controls edit tools only and is intentionally ignored here:
    # generic shell payloads always receive the complete read-only path set.
    try:
        from aeon.core.protected import protected_paths

        candidates.extend(Path(path) for path in protected_paths())
    except Exception as exc:
        raise FleetCommandGuardError(
            "COMMAND REFUSED: canonical protected-path policy could not be loaded."
        ) from exc
    paths: list[str] = []
    for path in candidates:
        paths.append(_canonical_existing_path(path, label=f"guardrail {path}"))
    return tuple(dict.fromkeys(paths))


def inaccessible_sandbox_paths() -> tuple[str, ...]:
    """Return fixed control/coordinator paths hidden from every payload."""

    paths: list[str] = []
    for path in _INACCESSIBLE_PATH_CANDIDATES:
        # InaccessiblePaths accepts a '-' prefix for absent runtime sockets. The
        # coordinator source is different: it must exist and its canonical
        # target is hidden as well so a compatibility symlink cannot bypass it.
        if path == GPU_COORDINATOR:
            paths.append(_canonical_existing_path(path, label="GPU coordinator boundary"))
            paths.append(str(path.absolute()))
            continue
        paths.append(str(path.absolute()))
        try:
            paths.append(str(path.resolve(strict=True)))
        except OSError:
            pass
    return tuple(dict.fromkeys(paths))


def resolve_command_cwd(
    requested_cwd: str | Path | None,
    *,
    session_root: str | Path | None = None,
) -> Path:
    """Resolve an exact command workspace without widening the session root.

    Managed Project Manager sessions intentionally start at ``/home/aday`` so
    they can coordinate several projects.  Granting that broad ancestor to a
    Landlock payload is both inefficient and incompatible with the protected
    credential/Fleet descendants below it.  A caller may therefore select one
    exact descendant project as its command cwd.  The selected path is
    canonicalized, must already be a directory beneath the launch workspace,
    and may not overlap any inaccessible credential/control subtree.

    Omitting ``requested_cwd`` preserves the existing launch-cwd behavior.  In
    particular, it does not retroactively reject a legacy session whose root is
    itself an ancestor of protected paths; the sandbox's recursive read walker
    continues to exclude those descendants.
    """

    try:
        root = Path.cwd() if session_root is None else Path(session_root)
        root = root.resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the session workspace is unavailable."
        ) from exc
    if not root.is_dir():
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the session workspace is not a directory."
        )

    if requested_cwd is None or not str(requested_cwd).strip():
        return root
    raw = str(requested_cwd).strip()
    if "\x00" in raw:
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the requested working directory is invalid."
        )
    candidate_path = Path(raw).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = root / candidate_path
    try:
        candidate = candidate_path.resolve(strict=True)
    except OSError as exc:
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the requested working directory does not exist."
        ) from exc
    if not candidate.is_dir():
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the requested working directory is not a directory."
        )
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the requested working directory is outside this "
            "agent's launch workspace."
        ) from exc

    # The default/root path is handled above for backward compatibility.  An
    # explicitly narrowed path must never select either a protected subtree or
    # one of its ancestors, because Landlock grants are additive.
    if candidate != root:
        for protected_value in inaccessible_sandbox_paths():
            protected = Path(protected_value).resolve(strict=False)
            if _paths_overlap(candidate, protected):
                raise FleetCommandGuardError(
                    "COMMAND REFUSED: the requested working directory overlaps "
                    "a protected credential or control path."
                )
    return candidate


def _validated_slice(source: Mapping[str, str]) -> str | None:
    raw = source.get(CPU_SANDBOX_SLICE_ENV)
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or not CPU_SANDBOX_SLICE_RE.fullmatch(raw):
        raise FleetCommandGuardError(
            "COMMAND REFUSED: inherited AEON_CPU_SANDBOX_SLICE does not match "
            "the exact aeon_subagent_<32hex>.slice lifecycle receipt contract. "
            "The requested command was not launched."
        )
    return raw


def _control_root() -> Path:
    root = Path(f"/run/user/{os.getuid()}/aeon-command-control")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the owner-only transient-service control root "
            "failed its directory, ownership, or mode check."
        )
    return root


def _scratch_root(cwd: Path) -> Path:
    root = cwd / ".aeon-command-scratch"
    try:
        root.mkdir(mode=0o700, exist_ok=True)
        metadata = root.lstat()
    except OSError as exc:
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the cwd-local private scratch root could not be created."
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the cwd-local private scratch root failed its "
            "directory, ownership, or mode check."
        )
    return root


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _cleanup_scratch_dir(path: Path) -> None:
    """Remove only one cryptographically named task-created scratch tree."""

    try:
        if path.is_symlink():
            return
        resolved = path.resolve(strict=False)
        parent = resolved.parent.resolve(strict=True)
        metadata = parent.lstat()
        if (
            parent.name != ".aeon-command-scratch"
            or not re.fullmatch(r"aeon-command-[0-9a-f]{32}", resolved.name)
            or resolved.parent != parent
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
        ):
            return
        if resolved.exists() and not resolved.is_symlink():
            shutil.rmtree(resolved)
        try:
            parent.rmdir()
        except OSError:
            pass
    except OSError:
        pass


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp_aeon_service_", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_gate_in_place(path: Path, nonce: str) -> None:
    """Release a pre-opened gate inode after the receipt is durable."""

    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise FleetCommandGuardError("COMMAND REFUSED: service gate identity drifted.")
    payload = (json.dumps({"nonce": nonce}, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _manager_environment(source: Mapping[str, str] | None) -> dict[str, str]:
    # systemd-run/systemctl need the caller's user-bus routing. The payload never
    # receives this mapping; command_service_exec uses its explicit scrubbed env.
    environment = dict(os.environ if source is None else source)
    return environment


def prepare_fleet_shell_boundary(
    *,
    source_environment: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    session_root: str | Path | None = None,
    expected_cwd_identity: tuple[int, int] | None = None,
    runtime_max_seconds: int | None = None,
    internal_state_path: str | Path | None = None,
) -> tuple[FleetShellBoundary, dict[str, str]]:
    """Construct one immutable service contract without running user code."""

    source = dict(os.environ if source_environment is None else source_environment)
    systemd_run = _verify_executable(
        SYSTEMD_RUN,
        expected_uid=0,
        label="root-owned /usr/bin/systemd-run executable",
    )
    systemctl = _verify_executable(
        SYSTEMCTL,
        expected_uid=0,
        label="root-owned /usr/bin/systemctl executable",
    )
    low_priority = require_fleet_low_priority_wrapper()
    service_exec = _verify_regular_file(
        SERVICE_EXEC,
        expected_uid=os.getuid(),
        label="Aeon transient-service bootstrap",
    )
    service_controller = _verify_regular_file(
        SERVICE_CONTROLLER,
        expected_uid=os.getuid(),
        label="Aeon transient-service controller",
    )
    if (
        not _SERVICE_EXEC_DIGEST_AT_IMPORT
        or not _SERVICE_CONTROLLER_DIGEST_AT_IMPORT
        or _launch_source_digest(Path(service_exec)) != _SERVICE_EXEC_DIGEST_AT_IMPORT
        or _launch_source_digest(Path(service_controller))
        != _SERVICE_CONTROLLER_DIGEST_AT_IMPORT
    ):
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the command-sandbox source changed after this Aeon "
            "process started. Restart this exact agent to load one coherent sandbox "
            "protocol. The requested command was not launched."
        )
    # Re-resolve at the last pre-launch boundary against the original launch
    # root. A same-UID rename/symlink swap between tool admission and service
    # construction must narrow or refuse, never widen the Landlock grant.
    canonical_cwd = resolve_command_cwd(cwd, session_root=session_root)
    try:
        cwd_metadata = canonical_cwd.stat(follow_symlinks=False)
    except OSError as exc:
        raise FleetCommandGuardError(
            f"COMMAND REFUSED: the working directory is unavailable ({type(exc).__name__}). "
            "The requested command was not launched."
        ) from exc
    actual_identity = (int(cwd_metadata.st_dev), int(cwd_metadata.st_ino))
    if expected_cwd_identity is not None and actual_identity != tuple(expected_cwd_identity):
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the working directory changed after admission. "
            "The requested command was not launched."
        )
    source_guardrails = trusted_guardrail_paths()
    base_inaccessible = inaccessible_sandbox_paths()
    state_guardrail: tuple[str, ...] = ()
    if internal_state_path is not None:
        try:
            state_path = Path(internal_state_path).resolve(strict=True)
            relative = state_path.relative_to(canonical_cwd / "aeon_output")
            metadata = state_path.stat()
            uuid.UUID(relative.parts[-1])
        except (OSError, ValueError, IndexError) as exc:
            raise FleetCommandGuardError(
                "COMMAND REFUSED: internal job-state guardrail is not an exact "
                "Aeon-owned UUID directory."
            ) from exc
        if (
            not state_path.is_dir()
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
            or len(relative.parts) < 3
            or relative.parts[-2] != "jobs"
        ):
            raise FleetCommandGuardError(
                "COMMAND REFUSED: internal job-state guardrail failed ownership/path checks."
            )
        state_guardrail = (str(state_path),)
    if runtime_max_seconds is not None:
        if isinstance(runtime_max_seconds, bool):
            raise FleetCommandGuardError("COMMAND REFUSED: invalid service runtime limit.")
        try:
            runtime_max_seconds = int(runtime_max_seconds)
        except (TypeError, ValueError) as exc:
            raise FleetCommandGuardError("COMMAND REFUSED: invalid service runtime limit.") from exc
        if runtime_max_seconds <= 0:
            raise FleetCommandGuardError("COMMAND REFUSED: invalid service runtime limit.")

    slice_name = _validated_slice(source)
    unit_name = f"aeon-command-{secrets.token_hex(16)}.service"
    nonce = secrets.token_hex(32)
    unit_stem = unit_name.removesuffix(".service")
    control_dir = _control_root() / unit_stem
    scratch_dir = _scratch_root(canonical_cwd) / unit_stem
    try:
        control_dir.mkdir(mode=0o700, exist_ok=False)
        scratch_dir.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        _cleanup_control_dir(control_dir)
        _cleanup_scratch_dir(scratch_dir)
        raise FleetCommandGuardError(
            "COMMAND REFUSED: generated transient-service control identity collided."
        ) from exc
    except OSError as exc:
        _cleanup_control_dir(control_dir)
        _cleanup_scratch_dir(scratch_dir)
        raise FleetCommandGuardError(
            "COMMAND REFUSED: private transient-service state could not be created."
        ) from exc

    # Landlock grants positive path-beneath rights; a broad cwd grant cannot be
    # revoked for a protected child. Any overlap with canonical safety source,
    # or a protected asynchronous state directory, therefore makes the whole cwd
    # read-only. Only the exact task-private scratch directory stays writable.
    source_overlap = any(
        _paths_overlap(canonical_cwd, Path(path)) for path in source_guardrails
    )
    if source_overlap or state_guardrail:
        writable_paths = (str(scratch_dir),)
    else:
        writable_paths = (str(canonical_cwd),)
    guardrails = source_guardrails + state_guardrail
    boundary = FleetShellBoundary(
        systemd_run=systemd_run,
        systemctl=systemctl,
        low_priority=low_priority,
        service_exec=service_exec,
        cwd=str(canonical_cwd),
        cwd_device=actual_identity[0],
        cwd_inode=actual_identity[1],
        unit_name=unit_name,
        nonce=nonce,
        control_dir=str(control_dir),
        scratch_dir=str(scratch_dir),
        runtime_max_seconds=runtime_max_seconds,
        slice_name=slice_name,
        guardrail_paths=guardrails,
        writable_paths=writable_paths,
        inaccessible_paths=base_inaccessible,
        # The bootstrap reads its spec and pre-opens the gate before Landlock;
        # the model payload cannot subsequently read this control directory.
        # /dev is separately denied and only fixed pseudo-devices are allowed.
        landlock_hidden_paths=base_inaccessible + (str(control_dir), "/dev"),
    )
    return boundary, _manager_environment(source)


def _parse_show_output(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in output.splitlines():
        if "=" not in raw_line:
            continue
        name, value = raw_line.split("=", 1)
        if name in values:
            raise FleetCommandGuardError(
                f"COMMAND REFUSED: duplicate systemd readback property {name}."
            )
        values[name] = value
    return values


def _show_unit(
    systemctl: str,
    unit_name: str,
    environment: Mapping[str, str],
    properties: Iterable[str] = _SERVICE_SHOW_PROPERTIES,
) -> dict[str, str] | None:
    argv = [systemctl, "--user", "show", "--no-pager", unit_name]
    for name in properties:
        argv.extend(("--property", name))
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FleetCommandGuardError(
            f"COMMAND REFUSED: exact systemd unit readback failed ({type(exc).__name__})."
        ) from exc
    values = _parse_show_output(result.stdout)
    if result.returncode != 0 or values.get("LoadState") == "not-found":
        return None
    return values


def _split_systemd_words(value: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(value))
    except ValueError as exc:
        raise FleetCommandGuardError("COMMAND REFUSED: malformed systemd property readback.") from exc


def _normalize_paths(value: str) -> set[str]:
    normalized: set[str] = set()
    for word in _split_systemd_words(value):
        stripped = word.lstrip("-+")
        if stripped:
            normalized.add(str(Path(stripped)))
    return normalized


def _duration_seconds(value: str) -> float | None:
    if value.strip() == "infinity":
        return None
    units = {"us": 1e-6, "ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0}
    total = 0.0
    position = 0
    for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)(us|ms|s|min|h)", value):
        if value[position:match.start()].strip():
            raise FleetCommandGuardError("COMMAND REFUSED: malformed systemd duration readback.")
        total += float(match.group(1)) * units[match.group(2)]
        position = match.end()
    if position == 0 or value[position:].strip():
        raise FleetCommandGuardError("COMMAND REFUSED: malformed systemd duration readback.")
    return total


def _read_process_cgroup(pid: int) -> str:
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FleetCommandGuardError(
            "COMMAND REFUSED: gated service process disappeared during identity readback."
        ) from exc
    unified = [line.split(":", 2)[2] for line in lines if line.startswith("0::")]
    if len(unified) != 1 or not unified[0].startswith("/"):
        raise FleetCommandGuardError("COMMAND REFUSED: ambiguous gated service cgroup identity.")
    return unified[0]


def _read_slice_control_group(
    boundary: FleetShellBoundary,
    environment: Mapping[str, str],
) -> str | None:
    if boundary.slice_name is None:
        return None
    values = _show_unit(
        boundary.systemctl,
        boundary.slice_name,
        environment,
        ("Id", "LoadState", "ActiveState", "ControlGroup"),
    )
    if (
        values is None
        or values.get("Id") != boundary.slice_name
        or values.get("LoadState") != "loaded"
        or values.get("ActiveState") != "active"
        or not values.get("ControlGroup", "").startswith("/")
    ):
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the inherited Aeon CPU sandbox slice has no exact "
            "active cgroup receipt."
        )
    return values["ControlGroup"].rstrip("/")


def _verify_service_readback(
    boundary: FleetShellBoundary,
    environment: Mapping[str, str],
    marker_pid: int,
    command_digest: str,
) -> SandboxServiceReceipt:
    values = _show_unit(boundary.systemctl, boundary.unit_name, environment)
    if values is None:
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the gated transient service disappeared before verification."
        )
    exact = {
        "Id": boundary.unit_name,
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "Type": "exec",
        "DevicePolicy": "closed",
        "PrivateTmp": "yes",
        "ProtectSystem": "strict",
        "ProtectHome": "read-only",
        "NoNewPrivileges": "yes",
        "RestrictNamespaces": "yes",
        "RestrictSUIDSGID": "yes",
        "LockPersonality": "yes",
        "KillMode": "control-group",
        "SendSIGKILL": "yes",
        "CollectMode": "inactive-or-failed",
        "WorkingDirectory": boundary.cwd,
    }
    for name, expected in exact.items():
        if values.get(name) != expected:
            raise FleetCommandGuardError(
                f"COMMAND REFUSED: transient service readback drifted for {name}."
            )
    try:
        main_pid = int(values.get("MainPID", "0"))
    except ValueError as exc:
        raise FleetCommandGuardError("COMMAND REFUSED: invalid transient service MainPID.") from exc
    if main_pid != marker_pid or main_pid <= 1:
        raise FleetCommandGuardError(
            "COMMAND REFUSED: gated bootstrap PID does not match systemd MainPID."
        )
    invocation_id = values.get("InvocationID", "")
    if not INVOCATION_ID_RE.fullmatch(invocation_id):
        raise FleetCommandGuardError("COMMAND REFUSED: invalid systemd InvocationID receipt.")
    control_group = values.get("ControlGroup", "")
    if not control_group.endswith("/" + boundary.unit_name):
        raise FleetCommandGuardError("COMMAND REFUSED: transient service cgroup is not exact.")
    if _read_process_cgroup(main_pid) != control_group:
        raise FleetCommandGuardError(
            "COMMAND REFUSED: gated bootstrap is outside the receipted service cgroup."
        )
    if values.get("Slice") != (boundary.slice_name or values.get("Slice")):
        raise FleetCommandGuardError("COMMAND REFUSED: transient service Slice readback drifted.")
    slice_cgroup = _read_slice_control_group(boundary, environment)
    if slice_cgroup is not None and not control_group.startswith(slice_cgroup + "/"):
        raise FleetCommandGuardError(
            "COMMAND REFUSED: transient service is outside its receipted parent slice."
        )
    if _normalize_paths(values.get("ReadWritePaths", "")) != set(boundary.writable_paths):
        raise FleetCommandGuardError(
            "COMMAND REFUSED: writable path policy drifted from the exact cwd-local grant."
        )
    if _normalize_paths(values.get("ReadOnlyPaths", "")) != set(boundary.guardrail_paths):
        raise FleetCommandGuardError("COMMAND REFUSED: trusted guardrail files are not read-only.")
    if _normalize_paths(values.get("InaccessiblePaths", "")) != set(boundary.inaccessible_paths):
        raise FleetCommandGuardError("COMMAND REFUSED: control/coordinator paths are not inaccessible.")
    if tuple(_split_systemd_words(values.get("RestrictAddressFamilies", ""))) != (
        RESTRICTED_ADDRESS_FAMILIES
    ):
        raise FleetCommandGuardError("COMMAND REFUSED: address-family policy readback drifted.")
    if set(_split_systemd_words(values.get("SystemCallFilter", ""))) != {
        "~socket", "socketpair"
    }:
        raise FleetCommandGuardError("COMMAND REFUSED: socket seccomp policy readback drifted.")
    if values.get("SystemCallErrorNumber") != str(errno.EPERM):
        raise FleetCommandGuardError("COMMAND REFUSED: socket seccomp errno readback drifted.")
    runtime = _duration_seconds(values.get("RuntimeMaxUSec", ""))
    if boundary.runtime_max_seconds is None:
        if runtime is not None:
            raise FleetCommandGuardError("COMMAND REFUSED: unexpected transient runtime limit.")
    elif runtime is None or abs(runtime - boundary.runtime_max_seconds) > 0.01:
        raise FleetCommandGuardError("COMMAND REFUSED: transient runtime limit readback drifted.")
    stop_timeout = _duration_seconds(values.get("TimeoutStopUSec", ""))
    if stop_timeout is None or abs(stop_timeout - 5.0) > 0.01:
        raise FleetCommandGuardError("COMMAND REFUSED: service stop timeout readback drifted.")
    return SandboxServiceReceipt(
        unit_name=boundary.unit_name,
        invocation_id=invocation_id,
        control_group=control_group,
        main_pid=main_pid,
        command_digest=command_digest,
        cwd=boundary.cwd,
        control_dir=boundary.control_dir,
        scratch_dir=boundary.scratch_dir,
        slice_name=boundary.slice_name,
        slice_control_group=slice_cgroup,
    )


def _wait_for_marker_from_pipe(
    process: subprocess.Popen,
    boundary: FleetShellBoundary,
) -> tuple[int, str]:
    if process.stdout is None:
        raise FleetCommandGuardError("COMMAND REFUSED: service output pipe is unavailable.")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + SERVICE_GATE_TIMEOUT
    prior: list[str] = []
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            events = selector.select(min(0.2, max(0.0, deadline - time.monotonic())))
            if not events:
                continue
            line = process.stdout.readline()
            if not line:
                break
            marker = _parse_marker(line, boundary)
            if marker is not None:
                return marker, "".join(prior)
            prior.append(line)
    finally:
        selector.close()
    raise FleetCommandGuardError(
        "COMMAND REFUSED: transient service never reached its gated bootstrap. "
        "The requested command was not executed."
    )


def _parse_marker(line: str, boundary: FleetShellBoundary) -> int | None:
    parts = line.strip().split()
    if not parts or parts[0] != SERVICE_MARKER_PREFIX:
        return None
    if len(parts) != 3 or parts[1] != boundary.nonce or not parts[2].isdigit():
        raise FleetCommandGuardError("COMMAND REFUSED: invalid gated-service marker.")
    return int(parts[2])


def _wait_for_marker_in_file(path: Path, boundary: FleetShellBoundary) -> int:
    deadline = time.monotonic() + SERVICE_GATE_TIMEOUT
    offset = 0
    buffered = ""
    while time.monotonic() < deadline:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                chunk = stream.read()
                offset = stream.tell()
        except FileNotFoundError:
            chunk = ""
        buffered += chunk
        while "\n" in buffered:
            line, buffered = buffered.split("\n", 1)
            marker = _parse_marker(line, boundary)
            if marker is not None:
                return marker
        time.sleep(0.02)
    raise FleetCommandGuardError(
        "COMMAND REFUSED: transient service never reached its gated bootstrap. "
        "The requested command was not executed."
    )


def _write_service_spec(
    boundary: FleetShellBoundary,
    command: str,
    payload_environment: Mapping[str, str],
) -> tuple[Path, str]:
    digest = hashlib.sha256(command.encode("utf-8")).hexdigest()
    spec_path = Path(boundary.control_dir) / "spec.json"
    # The bootstrap pre-opens this exact inode before installing Landlock. The
    # controller later writes the nonce in place, after the receipt is durable;
    # no writable/readable control path is inherited by the payload.
    _atomic_write_json(Path(boundary.control_dir) / "gate", {"nonce": ""})
    for name in ("probe-write", "probe-rename", "probe-unlink"):
        _atomic_write_json(Path(boundary.control_dir) / name, {"fixture": name})
    _atomic_write_json(
        spec_path,
        {
            "schema": 1,
            "nonce": boundary.nonce,
            "command": command,
            "command_digest": digest,
            "environment": dict(payload_environment),
            "cwd": boundary.cwd,
            "cwd_device": boundary.cwd_device,
            "cwd_inode": boundary.cwd_inode,
            "scratch_dir": boundary.scratch_dir,
            "read_only_paths": list(boundary.guardrail_paths),
            "writable_paths": list(boundary.writable_paths),
            "inaccessible_paths": list(boundary.landlock_hidden_paths),
        },
    )
    return spec_path, digest


def launch_sandbox_service(
    command: str,
    boundary: FleetShellBoundary,
    manager_environment: Mapping[str, str],
    *,
    receipt_path: str | Path | None = None,
    output_path: str | Path | None = None,
    payload_environment: Mapping[str, str] | None = None,
) -> SandboxServiceHandle:
    """Launch, verify, receipt, and only then release one service payload."""

    payload = scrubbed_payload_environment(payload_environment)
    payload.update(
        {
            "TMPDIR": boundary.scratch_dir,
            "TMP": boundary.scratch_dir,
            "TEMP": boundary.scratch_dir,
            "AEON_COMMAND_SCRATCH_DIR": boundary.scratch_dir,
        }
    )
    try:
        launch_cwd_metadata = Path(boundary.cwd).stat(follow_symlinks=False)
    except OSError as exc:
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the working directory disappeared before launch. "
            "The requested command was not executed."
        ) from exc
    if (
        int(launch_cwd_metadata.st_dev) != boundary.cwd_device
        or int(launch_cwd_metadata.st_ino) != boundary.cwd_inode
    ):
        raise FleetCommandGuardError(
            "COMMAND REFUSED: the working directory changed before launch. "
            "The requested command was not executed."
        )
    spec_path, digest = _write_service_spec(boundary, command, payload)
    output_file: IO[str] | None = None
    stdout: Any = subprocess.PIPE
    if output_path is not None:
        output_file = Path(output_path).open("w", encoding="utf-8", buffering=1)
        stdout = output_file
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            boundary.argv(str(spec_path)),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=dict(manager_environment),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if output_path is None:
            marker_pid, initial_output = _wait_for_marker_from_pipe(process, boundary)
        else:
            marker_pid = _wait_for_marker_in_file(Path(output_path), boundary)
            initial_output = ""
        receipt = _verify_service_readback(boundary, manager_environment, marker_pid, digest)
        durable_path = (
            Path(receipt_path)
            if receipt_path is not None
            else Path(boundary.control_dir) / "service_receipt.json"
        )
        _atomic_write_json(durable_path, receipt.to_json())
        # The shim has already loaded the spec into private memory. Remove it
        # before opening the read-only gate so payload code cannot recover stale
        # manager/Fleet environment from the control directory.
        spec_path.unlink(missing_ok=True)
        _write_gate_in_place(Path(boundary.control_dir) / "gate", boundary.nonce)
        return SandboxServiceHandle(
            boundary=boundary,
            process=process,
            receipt=receipt,
            initial_output=initial_output,
            output_file=output_file,
        )
    except Exception:
        if process is not None:
            _stop_unreceipted_unit(boundary, manager_environment)
            try:
                process.wait(timeout=SERVICE_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass
        if output_file is not None:
            output_file.close()
        _cleanup_control_dir(Path(boundary.control_dir))
        _cleanup_scratch_dir(Path(boundary.scratch_dir))
        raise


def read_sandbox_receipt(path: str | Path) -> SandboxServiceReceipt:
    receipt_path = Path(path)
    try:
        metadata = receipt_path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
        ):
            raise ValueError("unsafe receipt file")
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FleetCommandGuardError("missing or unreadable service receipt") from exc
    try:
        receipt = SandboxServiceReceipt(
            unit_name=value["unit_name"],
            invocation_id=value["invocation_id"],
            control_group=value["control_group"],
            main_pid=int(value["main_pid"]),
            command_digest=value["command_digest"],
            cwd=value["cwd"],
            control_dir=value["control_dir"],
            scratch_dir=value["scratch_dir"],
            slice_name=value.get("slice_name"),
            slice_control_group=value.get("slice_control_group"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FleetCommandGuardError("invalid service receipt") from exc
    expected_stem = receipt.unit_name.removesuffix(".service")
    expected_control = Path(f"/run/user/{os.getuid()}/aeon-command-control") / expected_stem
    try:
        canonical_cwd = Path(receipt.cwd).resolve(strict=True)
    except OSError as exc:
        raise FleetCommandGuardError("invalid service receipt cwd") from exc
    expected_scratch = canonical_cwd / ".aeon-command-scratch" / expected_stem
    if (
        value.get("schema") != SERVICE_RECEIPT_SCHEMA
        or not SERVICE_NAME_RE.fullmatch(receipt.unit_name)
        or not INVOCATION_ID_RE.fullmatch(receipt.invocation_id)
        or not re.fullmatch(r"[0-9a-f]{64}", receipt.command_digest)
        or receipt.main_pid <= 1
        or not receipt.control_group.endswith("/" + receipt.unit_name)
        or not canonical_cwd.is_dir()
        or Path(receipt.cwd) != canonical_cwd
        or Path(receipt.control_dir) != expected_control
        or Path(receipt.scratch_dir) != expected_scratch
        or (receipt.slice_name is not None and not CPU_SANDBOX_SLICE_RE.fullmatch(receipt.slice_name))
        or (
            receipt.slice_name is not None
            and (
                not receipt.slice_control_group
                or not receipt.control_group.startswith(
                    str(receipt.slice_control_group).rstrip("/") + "/"
                )
            )
        )
    ):
        raise FleetCommandGuardError("invalid service receipt identity")
    return receipt


def reconcile_sandbox_service(
    receipt: SandboxServiceReceipt,
    *,
    source_environment: Mapping[str, str] | None = None,
) -> str:
    environment = _manager_environment(source_environment)
    values = _show_unit(str(SYSTEMCTL), receipt.unit_name, environment,
                        ("Id", "LoadState", "ActiveState", "InvocationID", "ControlGroup", "Slice"))
    if values is None:
        return "absent"
    if (
        values.get("Id") != receipt.unit_name
        or values.get("InvocationID") != receipt.invocation_id
        or (receipt.slice_name is not None and values.get("Slice") != receipt.slice_name)
    ):
        return "mismatch"
    if values.get("ActiveState") in {"active", "activating", "deactivating"}:
        if values.get("ControlGroup") != receipt.control_group:
            return "mismatch"
        return "running"
    # A completed unit may clear ControlGroup before --collect removes the unit,
    # while retaining its exact InvocationID. Never mistake a nonempty different
    # cgroup for that normal terminal transition.
    if values.get("ControlGroup") not in {"", receipt.control_group}:
        return "mismatch"
    return "terminal"


def stop_sandbox_service(
    receipt: SandboxServiceReceipt,
    *,
    source_environment: Mapping[str, str] | None = None,
) -> bool:
    """Stop exactly the receipted unit; never signal a numeric workload PID."""

    environment = _manager_environment(source_environment)
    state = reconcile_sandbox_service(receipt, source_environment=environment)
    if state == "absent":
        return False
    if state == "mismatch":
        raise FleetCommandGuardError(
            "REFUSED: the transient service name now has a different InvocationID/cgroup."
        )
    result = subprocess.run(
        [str(SYSTEMCTL), "--user", "stop", receipt.unit_name],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=environment,
        timeout=SERVICE_STOP_TIMEOUT,
        check=False,
    )
    if result.returncode != 0 and reconcile_sandbox_service(
        receipt, source_environment=environment
    ) != "absent":
        raise FleetCommandGuardError("REFUSED: exact transient service stop failed.")
    deadline = time.monotonic() + SERVICE_STOP_TIMEOUT
    while time.monotonic() < deadline:
        state = reconcile_sandbox_service(receipt, source_environment=environment)
        if state == "absent":
            return True
        if state == "mismatch":
            raise FleetCommandGuardError(
                "REFUSED: transient unit identity changed while awaiting cleanup."
            )
        time.sleep(0.05)
    raise FleetCommandGuardError("REFUSED: exact transient service did not unload after stop.")


def _stop_unreceipted_unit(
    boundary: FleetShellBoundary,
    environment: Mapping[str, str],
) -> None:
    # The cryptographic unit name was generated by this exact failed launch and
    # the payload gate is still closed. This is the only pre-receipt stop path.
    try:
        subprocess.run(
            [boundary.systemctl, "--user", "stop", boundary.unit_name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=dict(environment),
            timeout=SERVICE_STOP_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _cleanup_control_dir(control_dir: Path) -> None:
    try:
        root = _control_root().resolve(strict=True)
        resolved = control_dir.resolve(strict=False)
        if resolved.parent != root or not resolved.name.startswith("aeon-command-"):
            return
        for name in (
            "spec.json",
            "gate",
            "service_receipt.json",
            "probe-write",
            "probe-rename",
            "probe-renamed",
            "probe-unlink",
            "probe-shadow",
        ):
            path = resolved / name
            try:
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            except OSError:
                pass
        resolved.rmdir()
    except OSError:
        pass


def finalize_sandbox_service(handle: SandboxServiceHandle) -> None:
    if handle.output_file is not None and not handle.output_file.closed:
        handle.output_file.close()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        state = reconcile_sandbox_service(handle.receipt)
        if state == "absent":
            break
        if state == "mismatch":
            return
        time.sleep(0.05)
    _cleanup_control_dir(Path(handle.receipt.control_dir))
    _cleanup_scratch_dir(Path(handle.receipt.scratch_dir))


def discard_prepared_sandbox_boundary(boundary: FleetShellBoundary) -> None:
    """Clean exact task-private state from a preflight that was never launched."""

    _cleanup_control_dir(Path(boundary.control_dir))
    _cleanup_scratch_dir(Path(boundary.scratch_dir))


def sandbox_log_text(path: str | Path) -> str:
    """Read a service log while omitting the trusted pre-execution marker."""

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    return "".join(line for line in lines if not line.startswith(SERVICE_MARKER_PREFIX + " "))
