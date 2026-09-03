#!/usr/bin/python3
"""Trusted, dependency-free gate used inside Aeon's CPU command service.

The transient unit starts this file through ``fleet-low-priority``.  It loads an
explicit scrubbed environment, announces that it is stopped at the execution
gate, and does not exec the requested shell until the out-of-service controller
has verified the exact systemd unit and durably stored its InvocationID receipt.
"""

from __future__ import annotations

import hashlib
import ctypes
import errno
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
import sys
import time


MARKER_PREFIX = "AEON_COMMAND_SANDBOX_GATED_V1"
GATE_TIMEOUT_SECONDS = 10.0
MAX_SPEC_BYTES = 2 * 1024 * 1024
NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

NO_DEVICE_ENV = {
    "CUDA_VISIBLE_DEVICES": "void",
    "GPU_DEVICE_ORDINAL": "-1",
    "HIP_VISIBLE_DEVICES": "-1",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "ROCR_VISIBLE_DEVICES": "-1",
}

SAFE_DEVICE_PATHS = (
    Path("/dev/null"),
    Path("/dev/zero"),
    Path("/dev/full"),
    Path("/dev/random"),
    Path("/dev/urandom"),
    Path("/dev/tty"),
    Path("/dev/ptmx"),
)
DEVICE_POLICY_PROBE = Path("/dev/net/tun")

# Linux Landlock ABI (x86_64 syscall numbers; the host contract is verified
# before creating a ruleset). We handle every write-like right plus device
# ioctl control through ABI 5.
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38
LL_EXECUTE = 1 << 0
LL_WRITE_FILE = 1 << 1
LL_READ_FILE = 1 << 2
LL_READ_DIR = 1 << 3
LL_REMOVE_DIR = 1 << 4
LL_REMOVE_FILE = 1 << 5
LL_MAKE_CHAR = 1 << 6
LL_MAKE_DIR = 1 << 7
LL_MAKE_REG = 1 << 8
LL_MAKE_SOCK = 1 << 9
LL_MAKE_FIFO = 1 << 10
LL_MAKE_BLOCK = 1 << 11
LL_MAKE_SYM = 1 << 12
LL_REFER = 1 << 13
LL_TRUNCATE = 1 << 14
LL_IOCTL_DEV = 1 << 15
LL_READ = LL_EXECUTE | LL_READ_FILE | LL_READ_DIR
LL_WRITE = (
    LL_WRITE_FILE
    | LL_REMOVE_DIR
    | LL_REMOVE_FILE
    | LL_MAKE_CHAR
    | LL_MAKE_DIR
    | LL_MAKE_REG
    | LL_MAKE_SOCK
    | LL_MAKE_FIFO
    | LL_MAKE_BLOCK
    | LL_MAKE_SYM
    | LL_REFER
    | LL_TRUNCATE
)
LL_SAFE_WRITE = LL_WRITE & ~(LL_MAKE_CHAR | LL_MAKE_BLOCK | LL_MAKE_SOCK)
LL_FILE_WRITE = LL_WRITE_FILE | LL_TRUNCATE


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


def _protected_environment_name(name: str) -> bool:
    upper = name.upper()
    markers = ("CLAIM", "COORD", "GPU", "LEASE", "VRAM")
    return (
        upper
        in {
            "AEON_CPU_SANDBOX_SLICE",
            "BASH_ENV",
            "CONTAINER_HOST",
            "DBUS_SESSION_BUS_ADDRESS",
            "DOCKER_CONTEXT",
            "DOCKER_HOST",
            "ENV",
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
            "ZDOTDIR",
        }
        or upper.startswith("CUDA_MPS")
        or upper.startswith("NVIDIA_")
        or upper.startswith("FLEET_")
        or upper.startswith("AEON_FLEET_")
        or (upper.startswith("SLURM_") and "GPU" in upper)
        or (
            upper.startswith(("AEON_", "GPU_", "QWEN_"))
            and any(marker in upper for marker in markers)
        )
    )


def _secure_regular_file(path: Path) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise ValueError("control file identity or mode is invalid")


def _load_spec(
    path: Path, expected_nonce: str
) -> tuple[
    str,
    dict[str, str],
    Path,
    Path,
    tuple[Path, ...],
    tuple[Path, ...],
    tuple[Path, ...],
]:
    _secure_regular_file(path)
    if path.stat().st_size > MAX_SPEC_BYTES:
        raise ValueError("control spec is too large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError("unsupported control spec")
    command = value.get("command")
    digest = value.get("command_digest")
    environment = value.get("environment")
    cwd = value.get("cwd")
    cwd_device = value.get("cwd_device")
    cwd_inode = value.get("cwd_inode")
    scratch_dir = value.get("scratch_dir")
    read_only_paths = value.get("read_only_paths")
    writable_paths = value.get("writable_paths")
    inaccessible_paths = value.get("inaccessible_paths")
    if (
        value.get("nonce") != expected_nonce
        or not isinstance(command, str)
        or not command.strip()
        or "\x00" in command
        or not isinstance(digest, str)
        or not DIGEST_RE.fullmatch(digest)
        or hashlib.sha256(command.encode("utf-8")).hexdigest() != digest
        or not isinstance(environment, dict)
        or not isinstance(cwd, str)
        or isinstance(cwd_device, bool)
        or not isinstance(cwd_device, int)
        or cwd_device < 0
        or isinstance(cwd_inode, bool)
        or not isinstance(cwd_inode, int)
        or cwd_inode <= 0
        or not isinstance(scratch_dir, str)
        or not isinstance(read_only_paths, list)
        or not isinstance(writable_paths, list)
        or not isinstance(inaccessible_paths, list)
    ):
        raise ValueError("control spec identity is invalid")
    clean: dict[str, str] = {}
    for name, item in environment.items():
        if not isinstance(name, str) or not isinstance(item, str):
            raise ValueError("payload environment contains protected state")
        safe_device_value = (
            name.upper() in NO_DEVICE_ENV and item == NO_DEVICE_ENV[name.upper()]
        )
        if (
            "=" in name
            or "\x00" in name
            or "\x00" in item
            or (_protected_environment_name(name) and not safe_device_value)
        ):
            raise ValueError("payload environment contains protected state")
        clean[name] = item
    clean.update(NO_DEVICE_ENV)
    canonical_cwd = Path(cwd).resolve(strict=True)
    active_cwd_metadata = os.stat(".", follow_symlinks=False)
    if (
        not canonical_cwd.is_dir()
        or canonical_cwd != Path.cwd().resolve(strict=True)
        or int(active_cwd_metadata.st_dev) != cwd_device
        or int(active_cwd_metadata.st_ino) != cwd_inode
    ):
        raise ValueError("payload cwd identity is invalid")

    def _absolute_paths(items: list, *, must_exist: bool) -> tuple[Path, ...]:
        paths: list[Path] = []
        for item in items:
            if not isinstance(item, str) or not Path(item).is_absolute():
                raise ValueError("sandbox path identity is invalid")
            candidate = Path(item)
            try:
                candidate = candidate.resolve(strict=must_exist)
            except OSError:
                if must_exist:
                    raise
                candidate = candidate.resolve(strict=False)
            paths.append(candidate)
        return tuple(dict.fromkeys(paths))

    protected = _absolute_paths(read_only_paths, must_exist=True)
    writable = _absolute_paths(writable_paths, must_exist=True)
    inaccessible = _absolute_paths(inaccessible_paths, must_exist=False)
    canonical_scratch = Path(scratch_dir).resolve(strict=True)
    expected_scratch = canonical_cwd / ".aeon-command-scratch" / path.parent.name
    scratch_metadata = canonical_scratch.lstat()
    if (
        canonical_scratch != expected_scratch
        or not stat.S_ISDIR(scratch_metadata.st_mode)
        or scratch_metadata.st_uid != os.getuid()
        or scratch_metadata.st_mode & 0o077
        or writable not in {(canonical_cwd,), (canonical_scratch,)}
        or any(not _is_at_or_below(item, canonical_cwd) for item in writable)
        or clean.get("TMPDIR") != str(canonical_scratch)
        or clean.get("TMP") != str(canonical_scratch)
        or clean.get("TEMP") != str(canonical_scratch)
        or clean.get("AEON_COMMAND_SCRATCH_DIR") != str(canonical_scratch)
    ):
        raise ValueError("payload writable-path identity is invalid")
    return (
        command,
        clean,
        canonical_cwd,
        canonical_scratch,
        protected,
        writable,
        inaccessible,
    )


def _is_at_or_below(path: Path, ancestor: Path) -> bool:
    try:
        path.relative_to(ancestor)
        return True
    except ValueError:
        return False


def _has_denied_descendant(path: Path, denied: tuple[Path, ...]) -> bool:
    return any(_is_at_or_below(item, path) for item in denied)


def _add_landlock_rule(libc, ruleset_fd: int, path: Path, rights: int) -> None:
    if rights == 0:
        return
    flags = os.O_PATH | os.O_CLOEXEC
    try:
        path_fd = os.open(path, flags)
    except FileNotFoundError:
        return
    try:
        attr = _LandlockPathBeneathAttr(rights, path_fd, 0)
        if libc.syscall(
            SYS_LANDLOCK_ADD_RULE,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attr),
            0,
        ) != 0:
            raise OSError(ctypes.get_errno(), "landlock_add_rule")
    finally:
        os.close(path_fd)


def _grant_read_tree(libc, ruleset_fd: int, path: Path, denied: tuple[Path, ...]) -> None:
    """Allow reads everywhere except exact inaccessible subtrees.

    Landlock rules are additive, so an ancestor of a denied path cannot receive
    a recursive rule. Descending only those few ancestor chains preserves normal
    reads across the rest of the host without granting the coordinator/control
    path through a broader ancestor.
    """

    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return
    if any(resolved == item or _is_at_or_below(resolved, item) for item in denied):
        return
    if not _has_denied_descendant(resolved, denied):
        if resolved.is_dir():
            _add_landlock_rule(libc, ruleset_fd, resolved, LL_READ)
        elif resolved.is_file():
            _add_landlock_rule(libc, ruleset_fd, resolved, LL_READ_FILE | LL_EXECUTE)
        return
    if not resolved.is_dir():
        _add_landlock_rule(libc, ruleset_fd, resolved, LL_READ_FILE | LL_EXECUTE)
        return
    try:
        children = list(resolved.iterdir())
    except OSError:
        return
    for child in children:
        if child.is_symlink():
            # The canonical target is reached and governed through its real
            # hierarchy; never attach a broad rule through a symlink alias.
            continue
        _grant_read_tree(libc, ruleset_fd, child, denied)


def _install_landlock(
    writable: tuple[Path, ...],
    denied: tuple[Path, ...],
    readable: tuple[Path, ...] = (),
) -> None:
    if platform.machine() not in {"x86_64", "amd64"}:
        raise OSError(errno.ENOSYS, "unsupported Landlock syscall architecture")
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        0,
        0,
        LANDLOCK_CREATE_RULESET_VERSION,
    )
    if abi < 5:
        raise OSError(ctypes.get_errno() or errno.ENOSYS, "Landlock ABI 5 required")
    attr = _LandlockRulesetAttr(LL_READ | LL_WRITE | LL_IOCTL_DEV)
    ruleset_fd = libc.syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(attr),
        ctypes.sizeof(attr),
        0,
    )
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        _grant_read_tree(libc, ruleset_fd, Path("/"), denied)
        # systemd's filesystem namespace exposes the merged-/usr compatibility
        # paths as separate mount traversals. Attach equivalent rules through
        # each fixed alias so the dynamic loader and normal executables work.
        for alias in (Path("/bin"), Path("/lib"), Path("/lib64"), Path("/sbin")):
            _add_landlock_rule(libc, ruleset_fd, alias, LL_READ)
        # ReadOnlyPaths introduces additional bind traversals. Attach positive
        # read rules through those exact service-visible paths as well; they do
        # not receive any write-like right.
        for path in readable:
            _add_landlock_rule(
                libc,
                ruleset_fd,
                path,
                LL_READ if path.is_dir() else (LL_READ_FILE | LL_EXECUTE),
            )
        # /dev is excluded from the broad read walk. Add back only fixed pseudo
        # devices needed by a non-interactive shell; accelerator, tun, VFIO,
        # storage, input, and other device nodes remain unreachable by path.
        for device in SAFE_DEVICE_PATHS:
            try:
                canonical_device = device.resolve(strict=True)
            except OSError:
                continue
            _add_landlock_rule(
                libc,
                ruleset_fd,
                canonical_device,
                LL_READ_FILE | LL_FILE_WRITE,
            )
        # ProtectSystem may introduce per-unit mount layers after path
        # canonicalization. Attach an explicit execute/read rule to the final
        # shell inode as well as the broad /usr hierarchy rule.
        _add_landlock_rule(
            libc,
            ruleset_fd,
            Path("/usr/bin/bash"),
            LL_READ_FILE | LL_EXECUTE,
        )
        _add_landlock_rule(
            libc,
            ruleset_fd,
            Path("/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"),
            LL_READ_FILE | LL_EXECUTE,
        )
        for path in writable:
            _add_landlock_rule(
                libc,
                ruleset_fd,
                path,
                (LL_READ | LL_SAFE_WRITE)
                if path.is_dir()
                else (LL_READ_FILE | LL_EXECUTE | LL_FILE_WRITE),
            )
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
        if libc.syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(ruleset_fd)


def _expect_permission_denied(label: str, operation) -> None:
    try:
        operation()
    except OSError as exc:
        if exc.errno in {errno.EPERM, errno.EACCES, errno.EAFNOSUPPORT}:
            return
        raise
    raise ValueError(f"{label} policy is not effective")


def _verify_device_probe_baseline() -> None:
    metadata = DEVICE_POLICY_PROBE.stat()
    if not stat.S_ISCHR(metadata.st_mode):
        raise ValueError("device isolation probe is not a character device")
    fd = os.open(DEVICE_POLICY_PROBE, os.O_RDONLY | os.O_CLOEXEC)
    os.close(fd)


def _active_policy_probes(
    protected: tuple[Path, ...], denied: tuple[Path, ...], probe_dir: Path
) -> None:
    protected_file = next((path for path in protected if path.is_file()), None)
    if protected_file is None:
        raise ValueError("no protected write probe target")
    def _open_protected() -> None:
        fd = os.open(protected_file, os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC)
        os.close(fd)

    _expect_permission_denied("protected-path write", _open_protected)
    existing_denied = next((path for path in denied if path.is_file()), None)
    if existing_denied is not None:
        def _open_denied() -> None:
            fd = os.open(existing_denied, os.O_RDONLY | os.O_CLOEXEC)
            os.close(fd)
        _expect_permission_denied("inaccessible-path read", _open_denied)

    def _open_device() -> None:
        fd = os.open(DEVICE_POLICY_PROBE, os.O_RDONLY | os.O_CLOEXEC)
        os.close(fd)

    _expect_permission_denied("non-standard device", _open_device)

    def _open_fixture(name: str, flags: int) -> None:
        fd = os.open(probe_dir / name, flags | os.O_CLOEXEC, 0o600)
        os.close(fd)

    _expect_permission_denied(
        "protected fixture write",
        lambda: _open_fixture("probe-write", os.O_WRONLY),
    )
    _expect_permission_denied(
        "protected create-shadow",
        lambda: _open_fixture("probe-shadow", os.O_WRONLY | os.O_CREAT | os.O_EXCL),
    )
    _expect_permission_denied(
        "protected rename",
        lambda: os.rename(probe_dir / "probe-rename", probe_dir / "probe-renamed"),
    )
    _expect_permission_denied(
        "protected unlink",
        lambda: os.unlink(probe_dir / "probe-unlink"),
    )
    for family, kind in (
        (socket.AF_UNIX, socket.SOCK_STREAM),
        (socket.AF_INET, socket.SOCK_STREAM),
        (socket.AF_INET6, socket.SOCK_STREAM),
        (socket.AF_NETLINK, socket.SOCK_RAW),
    ):
        def _socket(family=family, kind=kind) -> None:
            sock = socket.socket(family, kind)
            sock.close()

        _expect_permission_denied(f"socket family {family}", _socket)


def _gate_open(fd: int, nonce: str) -> bool:
    try:
        raw = os.pread(fd, 1024, 0)
        value = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and value.get("nonce") == nonce


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 3:
        return 2
    spec_path = Path(arguments[0])
    gate_path = Path(arguments[1])
    nonce = arguments[2]
    if (
        not spec_path.is_absolute()
        or not gate_path.is_absolute()
        or not NONCE_RE.fullmatch(nonce)
        or spec_path.parent != gate_path.parent
    ):
        return 2
    gate_fd = -1
    try:
        (
            command,
            environment,
            _cwd,
            _scratch,
            protected,
            writable,
            inaccessible,
        ) = _load_spec(spec_path, nonce)
        _secure_regular_file(gate_path)
        path_metadata = gate_path.lstat()
        gate_fd = os.open(gate_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        descriptor_metadata = os.fstat(gate_fd)
        if (
            path_metadata.st_dev != descriptor_metadata.st_dev
            or path_metadata.st_ino != descriptor_metadata.st_ino
        ):
            raise ValueError("service gate inode changed during open")
        # /dev/net/tun is a harmless, non-standard character device that is
        # readable on this host. Proving that before Landlock makes the later
        # denial an enforcement probe instead of a pre-existing DAC failure.
        _verify_device_probe_baseline()
        readable = tuple(
            path
            for path in protected
            if not any(
                _is_at_or_below(path, hidden) or _is_at_or_below(hidden, path)
                for hidden in inaccessible
            )
        )
        _install_landlock(writable, inaccessible, readable)
        _active_policy_probes(protected, inaccessible, spec_path.parent)
    except Exception as exc:
        if gate_fd >= 0:
            os.close(gate_fd)
        print(
            f"AEON_COMMAND_SANDBOX_BOOTSTRAP_ERROR {type(exc).__name__} "
            f"{getattr(exc, 'errno', '')} {exc}",
            flush=True,
        )
        return 125

    # This is the only pre-execution marker. stdout is inherited through
    # systemd-run --pipe; no AF_UNIX notify/control socket is required.
    print(f"{MARKER_PREFIX} {nonce} {os.getpid()}", flush=True)
    deadline = time.monotonic() + GATE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _gate_open(gate_fd, nonce):
            break
        time.sleep(0.02)
    else:
        os.close(gate_fd)
        return 125
    os.close(gate_fd)

    # No startup files, inherited manager environment, notify socket, or input
    # survive into the model-requested shell. stderr joins the stdout pipe/log.
    stage = "open-null"
    try:
        null_fd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
        stage = "dup-stdin"
        os.dup2(null_fd, 0)
        if null_fd != 0:
            os.close(null_fd)
        stage = "dup-stderr"
        os.dup2(1, 2)
        stage = "exec-bash"
        os.execve(
            "/usr/bin/bash",
            ["bash", "--noprofile", "--norc", "-c", command],
            environment,
        )
    except Exception as exc:
        print(
            f"AEON_COMMAND_SANDBOX_EXEC_ERROR {stage} {type(exc).__name__} "
            f"{getattr(exc, 'errno', '')}",
            flush=True,
        )
        return 126
    return 126


if __name__ == "__main__":
    raise SystemExit(main())
