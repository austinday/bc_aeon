"""
Crash-safe terminal-state resolution and exact lifecycle control for bounded
sub-agents. Pure logic, stdlib-only (no aeon imports), so worker.py and the
sub-agent tools can all use it without import cycles.

resolve() is the single source of truth for "is this sub-agent done, and if so
what happened". It reads, in priority order:
  1. output.json  -- the durable terminal record (always written terminal-FIRST,
     atomically, and NEVER deleted). If present, the agent is terminal.
  2. status.txt   -- an explicit terminal token (COMPLETED/FAILED/KILLED).
  3. PID liveness -- if neither terminal record exists but the wrapper PROCESS is
     gone, the agent crashed/was-killed before writing a result. This is the case
     that previously made gather() block for its full timeout and leaked a
     concurrency slot forever; here it resolves immediately as FAILED.

New sub-agents run in one UUID-derived user-systemd scope inside one
UUID-derived leaf slice. Their durable schema-2 receipt pins both units by
InvocationID, ControlGroup, and ControlGroupId. The leaf slice's cgroup.events
``populated`` bit is recursive, so it proves descendant liveness without ever
enumerating unrelated processes or units. Schema-1 PID receipts remain readable
for existing sessions, but deliberately refuse descendant escalation once their
verified group leader has exited.
"""

import os
import json
import re
import time
import signal
import select
import stat
import subprocess
import threading
import uuid
from pathlib import Path, PurePosixPath

TERMINAL = {"COMPLETED", "FAILED", "BLOCKED", "CANCELLED", "KILLED"}
PROCESS_REF = "process.json"
# The child Fleet close path may spend up to 12 seconds cancelling an admission
# start, two seconds joining an in-flight renewer, and ten seconds proving the
# exact broker release.  Keep the principal's catchable-termination window above
# that complete 24-second boundary so SIGKILL is a last resort rather than a race
# with correct ticket cleanup.
TERMINATION_GRACE_SECONDS = 30.0
TERMINATION_POLL_SECONDS = 0.05
TERMINATION_KILL_SETTLE_SECONDS = 5.0

SYSTEMD_RUN = "/usr/bin/systemd-run"
SYSTEMCTL = "/usr/bin/systemctl"
FLEET_LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
CGROUP_ROOT = Path("/sys/fs/cgroup")
CPU_SANDBOX_SLICE_ENV = "AEON_CPU_SANDBOX_SLICE"
SYSTEMD_QUERY_TIMEOUT_SECONDS = 5.0
SYSTEMD_CAPTURE_TIMEOUT_SECONDS = 5.0
SYSTEMD_OUTPUT_LIMIT_BYTES = 32 * 1024

_INVOCATION_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")
_SCOPE_UNIT_RE = re.compile(r"\Aaeon-subagent-[0-9a-f]{32}\.scope\Z")
_SLICE_UNIT_RE = re.compile(r"\Aaeon_subagent_[0-9a-f]{32}\.slice\Z")
_SCOPE_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "Transient",
    "InvocationID",
    "ControlGroup",
    "ControlGroupId",
    "Slice",
    "DevicePolicy",
    "KillMode",
    "TimeoutStopUSec",
    "SendSIGKILL",
)
_SLICE_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "Transient",
    "InvocationID",
    "ControlGroup",
    "ControlGroupId",
)
_TERMINATION_LOCKS = {}
_TERMINATION_LOCKS_GUARD = threading.Lock()


class ProcessIdentityError(RuntimeError):
    """A lifecycle target cannot be proven to be the recorded sub-agent."""


def sub_agent_systemd_units(agent_id):
    """Return collision-resistant, flat systemd unit names for one UUID.

    Underscores in the slice name are intentional: unlike hyphens, they do not
    create an implicit systemd slice hierarchy. The resulting slice is one leaf
    boundary that nested command scopes can inherit.
    """

    text = str(agent_id)
    try:
        parsed = uuid.UUID(text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProcessIdentityError("schema-2 agent id is not a UUID") from exc
    if str(parsed) != text:
        raise ProcessIdentityError("schema-2 agent id is not canonical lowercase UUID text")
    return (
        f"aeon-subagent-{parsed.hex}.scope",
        f"aeon_subagent_{parsed.hex}.slice",
    )


def sub_agent_systemd_command(agent_id, command):
    """Wrap a bounded-agent argv in its exact CPU-only scope and leaf slice."""

    scope_unit, slice_unit = sub_agent_systemd_units(agent_id)
    argv = [str(item) for item in command]
    if not argv:
        raise ProcessIdentityError("sub-agent command is empty")
    low_priority = _require_fleet_low_priority_wrapper()
    return [
        SYSTEMD_RUN,
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--no-ask-password",
        "--expand-environment=no",
        f"--unit={scope_unit}",
        f"--slice={slice_unit}",
        "--property=DevicePolicy=closed",
        "--property=KillMode=control-group",
        "--property=TimeoutStopSec=30s",
        "--property=SendSIGKILL=yes",
        "--",
        low_priority,
        *argv,
    ]


def _require_fleet_low_priority_wrapper():
    """Verify the fixed owner-work wrapper before constructing a child argv."""

    path = Path(FLEET_LOW_PRIORITY)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProcessIdentityError(
            "required Fleet low-priority wrapper is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o022
        or not os.access(path, os.X_OK)
    ):
        raise ProcessIdentityError(
            "required Fleet low-priority wrapper failed identity validation"
        )
    return str(path)


def assert_sub_agent_systemd_units_available(agent_id):
    """Refuse reuse of either UUID-derived unit before launching a new agent."""

    scope_unit, slice_unit = sub_agent_systemd_units(agent_id)
    for unit, properties in (
        (scope_unit, _SCOPE_PROPERTIES),
        (slice_unit, _SLICE_PROPERTIES),
    ):
        readback = _systemctl_show(unit, properties)
        if _unit_identity(readback, unit) is not None:
            raise ProcessIdentityError(f"refusing to reuse active systemd unit {unit}")
        if readback.get("ActiveState") not in {"inactive", "failed"}:
            raise ProcessIdentityError(f"systemd unit {unit} is not safely inactive")


def _proc_start_ticks(pid):
    """Return Linux /proc start ticks (field 22), which change on PID reuse."""
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return int(raw[raw.rfind(")") + 2:].split()[19])


def _proc_args(pid):
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [part.decode("utf-8", "surrogateescape") for part in raw.split(b"\0") if part]


def _flag_value(args, flag):
    positions = [index for index, value in enumerate(args) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(args):
        raise ProcessIdentityError(f"missing or repeated {flag}")
    return args[positions[0] + 1]


def _legacy_reference_numbers(agent_dir, reference):
    agent_dir = Path(agent_dir).resolve()
    try:
        pid = int(reference["pid"])
        pgid = int(reference["pgid"])
        start_ticks = int(reference["start_ticks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProcessIdentityError("invalid process reference") from exc
    if reference.get("agent_id") != agent_dir.name or pid <= 1 or pgid != pid:
        raise ProcessIdentityError("process reference does not match the agent directory")
    return pid, pgid, start_ticks


def _validate_legacy_sub_agent(agent_dir, reference):
    agent_dir = Path(agent_dir).resolve()
    pid, pgid, start_ticks = _legacy_reference_numbers(agent_dir, reference)
    try:
        if _proc_start_ticks(pid) != start_ticks or os.getpgid(pid) != pgid:
            raise ProcessIdentityError("PID or process group was reused")
        args = _proc_args(pid)
    except FileNotFoundError as exc:
        raise ProcessLookupError(pid) from exc
    except (OSError, IndexError, TypeError, ValueError) as exc:
        raise ProcessIdentityError("legacy process identity is unreadable") from exc
    if not any(args[index:index + 2] == ["-m", "aeon.scripts.sub_agent_wrapper"]
               for index in range(max(0, len(args) - 1))):
        raise ProcessIdentityError("PID is not an Aeon sub-agent wrapper")
    if _flag_value(args, "--agent_id") != agent_dir.name:
        raise ProcessIdentityError("wrapper agent id differs from the durable record")
    if Path(_flag_value(args, "--output_dir")).resolve() != agent_dir:
        raise ProcessIdentityError("wrapper output directory differs from the durable record")
    return pid, pgid


def _validate_systemd_launcher(agent_dir, pid, start_ticks, scope_unit, slice_unit):
    """Validate only the known launcher PID; never enumerate procfs."""

    agent_dir = Path(agent_dir).resolve()
    try:
        if _proc_start_ticks(pid) != int(start_ticks):
            raise ProcessIdentityError("systemd-run launcher PID was reused")
        args = _proc_args(pid)
    except FileNotFoundError as exc:
        raise ProcessLookupError(pid) from exc
    except (OSError, IndexError, TypeError, ValueError) as exc:
        raise ProcessIdentityError("systemd-run launcher identity is unreadable") from exc
    if not args:
        raise ProcessIdentityError("sub-agent launch process has empty argv")
    if Path(args[0]).resolve() == Path(SYSTEMD_RUN):
        required = {
            "--user",
            "--scope",
            "--collect",
            "--expand-environment=no",
            f"--unit={scope_unit}",
            f"--slice={slice_unit}",
            "--property=DevicePolicy=closed",
            "--property=KillMode=control-group",
            "--property=TimeoutStopSec=30s",
            "--property=SendSIGKILL=yes",
        }
        if any(args.count(value) != 1 for value in required):
            raise ProcessIdentityError(
                "systemd-run launcher policy differs from the fixed contract"
            )
        separators = [index for index, value in enumerate(args) if value == "--"]
        if len(separators) != 1:
            raise ProcessIdentityError("systemd-run launcher has no exact payload boundary")
        payload = args[separators[0] + 1:]
        low_priority = _require_fleet_low_priority_wrapper()
        if not payload or payload[0] != low_priority:
            raise ProcessIdentityError(
                "launcher payload bypasses the fixed Fleet low-priority wrapper"
            )
        payload = payload[1:]
    else:
        # In --scope mode systemd-run may exec the payload in place after the
        # manager has moved this exact PID into the new scope. Start ticks remain
        # unchanged, while schema-2 unit readback proves the manager-side policy.
        payload = args
        try:
            priority = os.getpriority(os.PRIO_PROCESS, pid)
        except OSError as exc:
            raise ProcessIdentityError(
                "in-place sub-agent scheduling priority is unreadable"
            ) from exc
        if priority != 19:
            raise ProcessIdentityError(
                "in-place sub-agent bypassed the Fleet low-priority wrapper"
            )
    if len(payload) < 3 or payload[1:3] != ["-m", "aeon.scripts.sub_agent_wrapper"]:
        raise ProcessIdentityError("launcher payload is not the Aeon sub-agent wrapper")
    if _flag_value(payload, "--agent_id") != agent_dir.name:
        raise ProcessIdentityError("launcher agent id differs from its directory")
    if Path(_flag_value(payload, "--output_dir")).resolve() != agent_dir:
        raise ProcessIdentityError("launcher output directory differs from its receipt")


def _systemctl_environment():
    environment = os.environ.copy()
    environment.update({
        "LC_ALL": "C",
        "SYSTEMD_COLORS": "0",
        "SYSTEMD_PAGER": "",
    })
    return environment


def _run_systemctl(arguments):
    try:
        result = subprocess.run(
            [SYSTEMCTL, "--user", "--no-pager", *arguments],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=SYSTEMD_QUERY_TIMEOUT_SECONDS,
            check=False,
            env=_systemctl_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProcessIdentityError("exact user-systemd operation failed") from exc
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if len(stdout.encode("utf-8", "surrogateescape")) > SYSTEMD_OUTPUT_LIMIT_BYTES:
        raise ProcessIdentityError("user-systemd output exceeded the fixed bound")
    if len(stderr.encode("utf-8", "surrogateescape")) > SYSTEMD_OUTPUT_LIMIT_BYTES:
        raise ProcessIdentityError("user-systemd error output exceeded the fixed bound")
    if result.returncode != 0:
        detail = " ".join(stderr.strip().split())[:300]
        suffix = f": {detail}" if detail else ""
        raise ProcessIdentityError(
            f"exact user-systemd operation returned {result.returncode}{suffix}"
        )
    return stdout


def _systemctl_show(unit, properties):
    if not (_SCOPE_UNIT_RE.fullmatch(unit) or _SLICE_UNIT_RE.fullmatch(unit)):
        raise ProcessIdentityError("refusing an invalid sub-agent systemd unit name")
    arguments = ["show", *[f"--property={name}" for name in properties], unit]
    output = _run_systemctl(arguments)
    parsed = {}
    for line in output.splitlines():
        if not line or "=" not in line:
            raise ProcessIdentityError("malformed user-systemd property output")
        key, value = line.split("=", 1)
        if key not in properties or key in parsed:
            raise ProcessIdentityError("unexpected or duplicate user-systemd property")
        parsed[key] = value
    if set(parsed) != set(properties):
        raise ProcessIdentityError("incomplete user-systemd property output")
    return parsed


def _systemctl_signal(slice_unit, signum):
    if not _SLICE_UNIT_RE.fullmatch(slice_unit):
        raise ProcessIdentityError("refusing to signal an invalid slice name")
    try:
        signal_name = signal.Signals(signum).name
    except (TypeError, ValueError) as exc:
        raise ProcessIdentityError("invalid sub-agent signal") from exc
    _run_systemctl([
        "kill",
        f"--signal={signal_name}",
        "--kill-whom=all",
        slice_unit,
    ])


def _systemctl_stop(slice_unit):
    if not _SLICE_UNIT_RE.fullmatch(slice_unit):
        raise ProcessIdentityError("refusing to stop an invalid slice name")
    _run_systemctl(["stop", slice_unit])


def _parse_positive_int(value, label):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ProcessIdentityError(f"invalid {label}") from exc
    if isinstance(value, bool) or parsed <= 0:
        raise ProcessIdentityError(f"invalid {label}")
    return parsed


def _parse_systemd_duration_usec(value):
    """Parse the compact durations emitted by ``systemctl show``."""

    factors = {
        "us": 1,
        "ms": 1_000,
        "s": 1_000_000,
        "min": 60_000_000,
        "h": 3_600_000_000,
    }
    total = 0.0
    position = 0
    pattern = re.compile(r"\s*(\d+(?:\.\d+)?)(us|ms|s|min|h)")
    while position < len(str(value)):
        match = pattern.match(str(value), position)
        if match is None:
            raise ProcessIdentityError("invalid TimeoutStopUSec readback")
        total += float(match.group(1)) * factors[match.group(2)]
        position = match.end()
    return int(total)


def _safe_control_group(value, unit):
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or "\x00" in value
        or len(value) > 4096
    ):
        raise ProcessIdentityError(f"invalid {unit} control group")
    path = PurePosixPath(value)
    if str(path) != value or any(part in {".", ".."} for part in path.parts):
        raise ProcessIdentityError(f"unsafe {unit} control group")
    if path.name != unit:
        raise ProcessIdentityError(f"{unit} control group has the wrong leaf")
    return value


def _unit_identity(properties, unit, *, require_present=False):
    if properties.get("Id") != unit:
        raise ProcessIdentityError("systemd returned a different unit identity")
    invocation_id = properties.get("InvocationID", "")
    control_group = properties.get("ControlGroup", "")
    raw_control_group_id = properties.get("ControlGroupId", "")
    try:
        control_group_id = int(raw_control_group_id)
    except (TypeError, ValueError) as exc:
        raise ProcessIdentityError("invalid systemd ControlGroupId") from exc
    present = bool(invocation_id or control_group or control_group_id)
    if not present:
        if require_present:
            raise ProcessIdentityError(f"{unit} is not active")
        if invocation_id or control_group or control_group_id != 0:
            raise ProcessIdentityError(f"partial inactive identity for {unit}")
        return None
    if properties.get("LoadState") != "loaded":
        raise ProcessIdentityError(f"identity-bearing {unit} is not loaded")
    if not _INVOCATION_ID_RE.fullmatch(invocation_id):
        raise ProcessIdentityError(f"invalid {unit} InvocationID")
    return {
        "invocation_id": invocation_id,
        "control_group": _safe_control_group(control_group, unit),
        "control_group_id": _parse_positive_int(control_group_id, "ControlGroupId"),
    }


def _validate_scope_policy(properties, slice_unit):
    if properties.get("Transient") != "yes":
        raise ProcessIdentityError("sub-agent scope is not transient")
    if properties.get("Slice") != slice_unit:
        raise ProcessIdentityError("sub-agent scope escaped its recorded leaf slice")
    if properties.get("DevicePolicy") != "closed":
        raise ProcessIdentityError("sub-agent scope DevicePolicy drifted")
    if properties.get("KillMode") != "control-group":
        raise ProcessIdentityError("sub-agent scope KillMode drifted")
    if properties.get("SendSIGKILL") != "yes":
        raise ProcessIdentityError("sub-agent scope SendSIGKILL drifted")
    if _parse_systemd_duration_usec(properties.get("TimeoutStopUSec")) != 30_000_000:
        raise ProcessIdentityError("sub-agent scope TimeoutStopSec drifted")


def _capture_schema2_readback(scope_unit, slice_unit):
    scope_properties = _systemctl_show(scope_unit, _SCOPE_PROPERTIES)
    slice_properties = _systemctl_show(slice_unit, _SLICE_PROPERTIES)
    scope_identity = _unit_identity(scope_properties, scope_unit, require_present=True)
    slice_identity = _unit_identity(slice_properties, slice_unit, require_present=True)
    if scope_properties.get("ActiveState") not in {"active", "activating"}:
        raise ProcessIdentityError("sub-agent scope is not active")
    if slice_properties.get("ActiveState") not in {"active", "activating"}:
        raise ProcessIdentityError("sub-agent slice is not active")
    _validate_scope_policy(scope_properties, slice_unit)
    expected_scope_group = f"{slice_identity['control_group']}/{scope_unit}"
    if scope_identity["control_group"] != expected_scope_group:
        raise ProcessIdentityError("scope cgroup is not directly below its leaf slice")
    return scope_identity, slice_identity


def capture_sub_agent_process(
    agent_dir,
    pid,
    *,
    scope_unit=None,
    slice_unit=None,
    timeout_seconds=SYSTEMD_CAPTURE_TIMEOUT_SECONDS,
):
    """Capture a legacy process receipt or a schema-2 systemd/cgroup receipt."""

    agent_dir = Path(agent_dir).resolve()
    pid = int(pid)
    if scope_unit is None and slice_unit is None:
        reference = {
            "schema": 1,
            "agent_id": agent_dir.name,
            "pid": pid,
            "pgid": os.getpgid(pid),
            "start_ticks": _proc_start_ticks(pid),
        }
        _validate_legacy_sub_agent(agent_dir, reference)
        return reference
    expected_scope, expected_slice = sub_agent_systemd_units(agent_dir.name)
    if scope_unit != expected_scope or slice_unit != expected_slice:
        raise ProcessIdentityError("systemd unit names do not match the agent UUID")
    launcher_start_ticks = _proc_start_ticks(pid)
    _validate_systemd_launcher(
        agent_dir, pid, launcher_start_ticks, scope_unit, slice_unit
    )
    try:
        timeout = max(0.0, min(float(timeout_seconds), SYSTEMD_CAPTURE_TIMEOUT_SECONDS))
    except (TypeError, ValueError) as exc:
        raise ProcessIdentityError("invalid systemd receipt timeout") from exc
    deadline = time.monotonic() + timeout
    last_error = None
    while True:
        try:
            scope_identity, slice_identity = _capture_schema2_readback(
                scope_unit, slice_unit
            )
            break
        except ProcessIdentityError as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise ProcessIdentityError(
                    f"could not prove new sub-agent systemd identity: {last_error}"
                ) from exc
            try:
                _validate_systemd_launcher(
                    agent_dir, pid, launcher_start_ticks, scope_unit, slice_unit
                )
            except ProcessLookupError as gone:
                raise ProcessIdentityError(
                    "systemd-run exited before its exact unit receipt was captured"
                ) from gone
            time.sleep(min(TERMINATION_POLL_SECONDS, deadline - time.monotonic()))
    reference = {
        "schema": 2,
        "agent_id": agent_dir.name,
        "pid": pid,
        "launcher_pid": pid,
        "launcher_start_ticks": launcher_start_ticks,
        "scope_unit": scope_unit,
        "slice_unit": slice_unit,
        "scope_invocation_id": scope_identity["invocation_id"],
        "slice_invocation_id": slice_identity["invocation_id"],
        "scope_control_group": scope_identity["control_group"],
        "slice_control_group": slice_identity["control_group"],
        "scope_control_group_id": scope_identity["control_group_id"],
        "slice_control_group_id": slice_identity["control_group_id"],
    }
    return reference


def _load_process_reference(agent_dir):
    path = Path(agent_dir) / PROCESS_REF
    try:
        reference = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProcessIdentityError("missing or unreadable process identity") from exc
    schema = reference.get("schema") if isinstance(reference, dict) else None
    if type(schema) is not int or schema not in {1, 2}:
        raise ProcessIdentityError("unsupported process identity record")
    return reference


def _pidfd_exited(pidfd):
    """Return whether the exact pidfd identity has exited (including a zombie)."""

    readable, _, _ = select.select([pidfd], [], [], 0)
    return bool(readable)


def _legacy_group_absent(pgid):
    """Read-only absence proof for an old receipt after its leader exits."""

    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError as exc:
        raise ProcessIdentityError(
            "recorded process group exists but is not owner-signalable"
        ) from exc
    except OSError as exc:
        raise ProcessIdentityError("recorded process-group liveness is unreadable") from exc
    return False


def _schema2_shape(agent_dir, reference):
    agent_dir = Path(agent_dir).resolve()
    if reference.get("agent_id") != agent_dir.name:
        raise ProcessIdentityError("systemd receipt does not match the agent directory")
    expected_scope, expected_slice = sub_agent_systemd_units(agent_dir.name)
    if reference.get("scope_unit") != expected_scope:
        raise ProcessIdentityError("scope unit does not match the agent UUID")
    if reference.get("slice_unit") != expected_slice:
        raise ProcessIdentityError("slice unit does not match the agent UUID")
    for key in ("scope_invocation_id", "slice_invocation_id"):
        value = reference.get(key)
        if not isinstance(value, str) or not _INVOCATION_ID_RE.fullmatch(value):
            raise ProcessIdentityError(f"invalid receipt field {key}")
    launcher_pid = _parse_positive_int(
        reference.get("launcher_pid"), "launcher PID"
    )
    if launcher_pid <= 1 or reference.get("pid") != launcher_pid:
        raise ProcessIdentityError("schema-2 launcher PID fields disagree")
    _parse_positive_int(reference.get("launcher_start_ticks"), "launcher start ticks")
    scope_group = _safe_control_group(
        reference.get("scope_control_group"), expected_scope
    )
    slice_group = _safe_control_group(
        reference.get("slice_control_group"), expected_slice
    )
    if scope_group != f"{slice_group}/{expected_scope}":
        raise ProcessIdentityError("receipted scope cgroup escaped its leaf slice")
    return {
        "scope_unit": expected_scope,
        "slice_unit": expected_slice,
        "scope_invocation_id": reference["scope_invocation_id"],
        "slice_invocation_id": reference["slice_invocation_id"],
        "scope_control_group": scope_group,
        "slice_control_group": slice_group,
        "scope_control_group_id": _parse_positive_int(
            reference.get("scope_control_group_id"), "scope ControlGroupId"
        ),
        "slice_control_group_id": _parse_positive_int(
            reference.get("slice_control_group_id"), "slice ControlGroupId"
        ),
    }


def _cgroup_path(control_group):
    # ``_safe_control_group`` already rejected traversal and non-absolute paths.
    return CGROUP_ROOT.joinpath(*PurePosixPath(control_group).parts[1:])


def _read_cgroup_populated(control_group):
    events_path = _cgroup_path(control_group) / "cgroup.events"
    try:
        with events_path.open("r", encoding="ascii") as handle:
            raw = handle.read(4097)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ProcessIdentityError("exact leaf-slice cgroup.events is unreadable") from exc
    if len(raw) > 4096:
        raise ProcessIdentityError("cgroup.events exceeded the fixed read bound")
    values = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[1].isdigit() or fields[0] in values:
            raise ProcessIdentityError("malformed cgroup.events")
        values[fields[0]] = int(fields[1])
    if values.get("populated") not in {0, 1}:
        raise ProcessIdentityError("cgroup.events lacks a boolean populated field")
    return bool(values["populated"])


def _schema2_readback(agent_dir, reference):
    receipt = _schema2_shape(agent_dir, reference)
    scope_properties = _systemctl_show(receipt["scope_unit"], _SCOPE_PROPERTIES)
    slice_properties = _systemctl_show(receipt["slice_unit"], _SLICE_PROPERTIES)
    scope_identity = _unit_identity(scope_properties, receipt["scope_unit"])
    slice_identity = _unit_identity(slice_properties, receipt["slice_unit"])
    if scope_identity is not None:
        _validate_scope_policy(scope_properties, receipt["slice_unit"])
        if (
            scope_identity["invocation_id"] != receipt["scope_invocation_id"]
            or scope_identity["control_group"] != receipt["scope_control_group"]
            or scope_identity["control_group_id"] != receipt["scope_control_group_id"]
        ):
            raise ProcessIdentityError("sub-agent scope identity drifted")
    if slice_identity is not None and (
        slice_identity["invocation_id"] != receipt["slice_invocation_id"]
        or slice_identity["control_group"] != receipt["slice_control_group"]
        or slice_identity["control_group_id"] != receipt["slice_control_group_id"]
    ):
        raise ProcessIdentityError("sub-agent slice identity drifted")
    scope_path_exists = _cgroup_path(receipt["scope_control_group"]).exists()
    slice_path_exists = _cgroup_path(receipt["slice_control_group"]).exists()
    if scope_identity is None and scope_path_exists:
        raise ProcessIdentityError("scope cgroup exists under a missing systemd identity")
    if scope_identity is not None and not scope_path_exists:
        raise ProcessIdentityError("scope identity exists without its exact cgroup")
    if slice_identity is None and slice_path_exists:
        raise ProcessIdentityError("slice cgroup exists under a missing systemd identity")
    if slice_identity is not None and not slice_path_exists:
        raise ProcessIdentityError("slice identity exists without its exact cgroup")
    if scope_identity is not None and slice_identity is None:
        raise ProcessIdentityError("scope identity exists without its parent slice identity")
    return receipt, scope_identity, slice_identity


def _schema2_liveness(agent_dir, reference):
    receipt, _scope_identity, slice_identity = _schema2_readback(agent_dir, reference)
    if slice_identity is None:
        return False
    try:
        return _read_cgroup_populated(receipt["slice_control_group"])
    except FileNotFoundError:
        # A collection race is safe only when a second exact readback proves both
        # recorded units and cgroups are now absent.
        _receipt, scope_identity, slice_identity = _schema2_readback(
            agent_dir, reference
        )
        if scope_identity is None and slice_identity is None:
            return False
        raise ProcessIdentityError("leaf-slice cgroup disappeared under a live identity")


def _retire_empty_schema2_slice(agent_dir, reference):
    receipt, _scope_identity, slice_identity = _schema2_readback(agent_dir, reference)
    if slice_identity is None:
        return
    if _read_cgroup_populated(receipt["slice_control_group"]):
        raise ProcessIdentityError("refusing to stop a populated sub-agent slice")
    # Exact readback is deliberately adjacent to the unit-name operation. Unit
    # names are UUID-derived and never reused by Aeon.
    _systemctl_stop(receipt["slice_unit"])
    _receipt, scope_identity, slice_identity = _schema2_readback(
        agent_dir, reference
    )
    if scope_identity is not None or slice_identity is not None:
        raise ProcessIdentityError("sub-agent units remained active after exact slice stop")


def _terminate_schema2(agent_dir, reference, grace, interval):
    if not _schema2_liveness(agent_dir, reference):
        _retire_empty_schema2_slice(agent_dir, reference)
        return False
    # Revalidate immediately before each signal. The recursive slice cgroup, not
    # a process list, is the authority for every contained command scope.
    receipt, _scope_identity, slice_identity = _schema2_readback(agent_dir, reference)
    if slice_identity is None:
        return False
    _systemctl_signal(receipt["slice_unit"], signal.SIGTERM)
    deadline = time.monotonic() + grace
    while True:
        if not _schema2_liveness(agent_dir, reference):
            _retire_empty_schema2_slice(agent_dir, reference)
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))

    receipt, _scope_identity, slice_identity = _schema2_readback(agent_dir, reference)
    if slice_identity is None or not _read_cgroup_populated(receipt["slice_control_group"]):
        _retire_empty_schema2_slice(agent_dir, reference)
        return True
    _systemctl_signal(receipt["slice_unit"], signal.SIGKILL)
    kill_deadline = time.monotonic() + TERMINATION_KILL_SETTLE_SECONDS
    while True:
        if not _schema2_liveness(agent_dir, reference):
            _retire_empty_schema2_slice(agent_dir, reference)
            return True
        remaining = kill_deadline - time.monotonic()
        if remaining <= 0:
            raise ProcessIdentityError(
                "exact sub-agent slice remained populated after SIGKILL"
            )
        time.sleep(min(interval, remaining))


def _terminate_legacy(agent_dir, reference, grace, interval):
    pid, pgid, _start_ticks = _legacy_reference_numbers(agent_dir, reference)
    try:
        _validate_legacy_sub_agent(agent_dir, reference)
    except ProcessLookupError:
        if _legacy_group_absent(pgid):
            return False
        raise ProcessIdentityError(
            "legacy wrapper already exited while its numeric process group remains"
        )
    if not hasattr(os, "pidfd_open"):
        raise ProcessIdentityError("pidfd_open is unavailable; refusing an unsafe PID-only kill")
    try:
        pidfd = os.pidfd_open(pid, 0)
    except ProcessLookupError:
        if _legacy_group_absent(pgid):
            return False
        raise ProcessIdentityError(
            "legacy wrapper exited before pinning while its numeric process group remains"
        )
    try:
        _validate_legacy_sub_agent(agent_dir, reference)
        try:
            session_id = os.getsid(pid)
        except (OSError, ProcessLookupError) as exc:
            raise ProcessIdentityError("sub-agent session identity is unavailable") from exc
        if session_id != pgid:
            raise ProcessIdentityError(
                "sub-agent wrapper is not the leader of its recorded session"
            )
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        deadline = time.monotonic() + grace
        while True:
            if _pidfd_exited(pidfd):
                if _legacy_group_absent(pgid):
                    return True
            else:
                try:
                    _validate_legacy_sub_agent(agent_dir, reference)
                except ProcessLookupError as exc:
                    if not _pidfd_exited(pidfd):
                        raise ProcessIdentityError(
                            "legacy wrapper identity vanished before pidfd exit proof"
                        ) from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(interval, remaining))
        if _pidfd_exited(pidfd):
            if _legacy_group_absent(pgid):
                return True
            raise ProcessIdentityError(
                "legacy wrapper exited while its numeric process group remains; "
                "refusing unprovable descendant escalation"
            )
        try:
            _validate_legacy_sub_agent(agent_dir, reference)
        except ProcessLookupError as exc:
            if _pidfd_exited(pidfd):
                if _legacy_group_absent(pgid):
                    return True
                raise ProcessIdentityError(
                    "legacy wrapper exited before escalation while its numeric "
                    "process group remains"
                ) from exc
            raise ProcessIdentityError(
                "legacy wrapper identity vanished before escalation"
            ) from exc
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return True
    finally:
        os.close(pidfd)


def terminate_sub_agent(
    agent_dir,
    *,
    grace_seconds=TERMINATION_GRACE_SECONDS,
    poll_seconds=TERMINATION_POLL_SECONDS,
):
    """Stop the exact schema-2 slice (or conservative legacy wrapper group)."""

    lock_key = str(Path(agent_dir).resolve())
    with _TERMINATION_LOCKS_GUARD:
        lifecycle_lock = _TERMINATION_LOCKS.setdefault(lock_key, threading.Lock())
    with lifecycle_lock:
        try:
            grace = max(0.0, min(float(grace_seconds), TERMINATION_GRACE_SECONDS))
            interval = max(0.01, min(float(poll_seconds), 0.5))
        except (TypeError, ValueError) as exc:
            raise ProcessIdentityError("invalid sub-agent termination timing") from exc
        reference = _load_process_reference(agent_dir)
        if reference["schema"] == 2:
            return _terminate_schema2(agent_dir, reference, grace, interval)
        return _terminate_legacy(agent_dir, reference, grace, interval)


def read_progress(agent_dir, freeze_seconds=60):
    """Best-effort live progress for a (presumed running) sub-agent.

    Single source of truth for "what is this student doing right now", used by
    both the principal's always-on SUB-AGENTS digest and gather_sub_agents so
    they never disagree. Returns a dict:
      age          seconds since the sub-agent last signalled activity (or None)
      step         human-readable current step (or None)
      iteration    planning iteration (or None)
      frozen       True if progress.json itself stopped being written (whole-
                   process freeze the watchdog can't report on)
      stuck_reason short string when the sub-agent self-reported a loop (or None)
      wallclock    seconds the sub-agent has been running (or None)
    """
    agent_dir = Path(agent_dir)
    pj = agent_dir / "progress.json"
    try:
        if pj.exists():
            frozen = (time.time() - pj.stat().st_mtime) > freeze_seconds
            data = json.loads(pj.read_text(encoding="utf-8"))
            return {
                "age": data.get("activity_age"),
                "step": data.get("step"),
                "iteration": data.get("iteration"),
                "frozen": frozen,
                "stuck_reason": data.get("stuck_reason"),
                "wallclock": data.get("wallclock"),
            }
    except Exception:
        pass
    tj = agent_dir / "telemetry.json"
    try:
        if tj.exists():
            data = json.loads(tj.read_text(encoding="utf-8"))
            ts = data.get("timestamp")
            if ts:
                return {"age": max(0.0, time.time() - float(ts)), "step": data.get("current_step"),
                        "iteration": data.get("iteration"), "frozen": False,
                        "stuck_reason": None, "wallclock": None}
    except Exception:
        pass
    for fname in ("pid.txt", "status.txt"):
        p = agent_dir / fname
        if p.exists():
            return {"age": max(0.0, time.time() - p.stat().st_mtime), "step": None,
                    "iteration": None, "frozen": False, "stuck_reason": None, "wallclock": None}
    return {"age": None, "step": None, "iteration": None, "frozen": False,
            "stuck_reason": None, "wallclock": None}


def norm_status(status):
    """Collapse 'FAILED: <detail>' -> 'FAILED' for matching."""
    if not status:
        return ""
    return str(status).split(":", 1)[0].strip().upper()


def pid_alive(agent_dir):
    """True / False / None(unknown), using the immutable lifecycle receipt."""
    try:
        reference = _load_process_reference(agent_dir)
        if reference["schema"] == 2:
            return _schema2_liveness(agent_dir, reference)
        pid, _ = _validate_legacy_sub_agent(agent_dir, reference)
    except ProcessLookupError:
        return False
    except ProcessIdentityError:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def resolve(agent_dir):
    """Returns (is_terminal: bool, status: str, report: Optional[str])."""
    agent_dir = Path(agent_dir)
    output_path = agent_dir / "output.json"
    status_path = agent_dir / "status.txt"

    if output_path.exists():
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
            st = data.get("status", "COMPLETED")
            if "error" in data and norm_status(st) != "COMPLETED":
                return True, st, f"Error: {data['error']}"
            return True, st, str(data.get("result", "N/A"))
        except Exception:
            pass  # atomic writes make a torn read impossible; fall through if genuinely corrupt

    status_text = None
    if status_path.exists():
        try:
            status_text = status_path.read_text(encoding="utf-8").strip()
        except Exception:
            status_text = None

    if status_text and norm_status(status_text) in TERMINAL:
        return True, status_text, "(terminal status reported; no output.json found)"

    if pid_alive(agent_dir) is False:
        return True, "FAILED", ("Process exited without writing a result "
                                "(crashed during startup or was killed externally).")

    return False, (status_text or "RUNNING"), None
