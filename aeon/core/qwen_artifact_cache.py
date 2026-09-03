"""Fleet artifact-cache backend for Aeon's exact Qwen release.

Fleet owns durable references, admission, eviction decisions, and cache paths.
This backend owns only bounded, renter-yielding transport and Qwen-specific
semantic verification on the selected worker. It never inspects renter
containers and never removes an image or layer from Docker's global store.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import tarfile
import time
from typing import Any, Callable, Mapping

from fleet_compute.artifact_cache import (
    ArtifactCacheError,
    ArtifactCacheSafetyError,
    CacheEntryInspection,
    CacheRemovalReceipt,
    CacheRootInspection,
)
from fleet_compute.models import ArtifactDescriptor, ArtifactKind

from .qwen_fleet_runtime import (
    FLEET_WORKER_CACHE_ROOT,
    QWEN_IMAGE_CACHE_COLD_PEAK_BYTES,
    QWEN_IMAGE_CACHE_ARTIFACT_ID,
    QWEN_IMAGE_CACHE_MAX_INODES,
    QWEN_IMAGE_CACHE_RECEIPT_MAX_BYTES,
    QWEN_IMAGE_ARCHIVE_MAX_BYTES,
    QWEN_MODEL_CACHE_ARTIFACT_ID,
    QWEN_MODEL_CACHE_COLD_PEAK_BYTES,
    QWEN_MODEL_CACHE_MAX_BYTES,
    QWEN_MODEL_CACHE_MAX_INODES,
    QWEN_MODEL_CACHE_TRANSFER_MAX_BYTES,
    QWEN_SOURCE_CACHE_ARTIFACT_ID,
    QWEN_SOURCE_CACHE_COLD_PEAK_BYTES,
    QWEN_SOURCE_CACHE_MAX_BYTES,
    QWEN_SOURCE_CACHE_MAX_INODES,
    QWEN_SOURCE_CACHE_TRANSFER_MAX_BYTES,
    QWEN_STANDARD_IMAGE_CONFIG_SHA256,
)
from .qwen_runtime import (
    FLEET_LOW_PRIORITY,
    HOST_BASH,
    HOST_SHA256SUM,
    MAX_IMAGE_LOGICAL_BYTES,
    SOURCE_MANIFEST_FILE,
    _docker_cli_environment,
    _docker_command,
    _normalise_image_config,
    _source_identity,
    load_artifact_identity,
    local_image_id,
    local_image_size,
)
from .fleet_hosts import network_address


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HOSTNAMES = {
    "192.168.0.178": "DAY2XRTX5000",
    "192.168.0.180": "DAY2XRTX5000PRO-2",
}
REMOTE_PYTHON = (
    "/home/aday/.local/share/uv/python/"
    "cpython-3.12-linux-x86_64-gnu/bin/python3.12"
)
REMOTE_WRAPPER = "/home/aday/bin/fleet-low-priority"
REMOTE_DOCKER = "/home/aday/bin/docker"
HOST_PRLIMIT = "/usr/bin/prlimit"
CANONICAL_OCI_ROOT = Path(
    "/home/aday/.local/state/fleet-compute/artifacts/aeon-qwen38/oci"
)
CANONICAL_STATE_ROOT = Path("/home/aday/.local/state/fleet-compute")
CANONICAL_MODEL_ROOT = Path(
    "/home/aday/.aeon/models/Qwen3.8-27B-ARA-abliterated-NVFP4-MTP"
)
CANONICAL_MODEL_SHA256SUMS = (
    "e7eca7ebee03c4f27482d4fe421ca1fac9f1d9986663a51fd7614361010c1237"
)
OCI_RECEIPT_SCHEMA = 1
OWNERSHIP_XATTR = "user.fleet_compute_cache"
OWNERSHIP_SCHEMA = 1
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_OCI_TAR_MAX_MEMBERS = 100_000
_OCI_CONFIG_MAX_BYTES = 16 * 1024 * 1024
CANONICAL_FREE_RESERVE_BYTES = 20 * 1024**3
CANONICAL_FREE_RESERVE_INODES = 10_000
_REMOTE_CACHE_ROOT_PROOF = r'''
import os, pathlib, stat, sys
_proof_root = pathlib.PurePosixPath(sys.argv[2])
_proof_parent = pathlib.PurePosixPath("/home/aday/.local/state/fleet-compute/cache")
assert _proof_root.parent == _proof_parent and _proof_root != _proof_parent
assert _proof_root.is_absolute() and ".." not in _proof_root.parts
_proof_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_proof_fd = os.open("/", _proof_flags)
try:
    for _proof_part in _proof_root.parts[1:]:
        _proof_child = os.open(_proof_part, _proof_flags, dir_fd=_proof_fd)
        _proof_meta = os.fstat(_proof_child)
        assert stat.S_ISDIR(_proof_meta.st_mode)
        os.close(_proof_fd); _proof_fd = _proof_child
except BaseException:
    os.close(_proof_fd)
    raise
def _cache_mount_id(fd):
    with open(f"/proc/self/fdinfo/{fd}", "r", encoding="ascii") as _proof_info:
        _proof_values = [line.split(":",1)[1].strip() for line in _proof_info
                         if line.startswith("mnt_id:")]
    assert len(_proof_values) == 1 and _proof_values[0].isdecimal()
    return _proof_values[0]
_proof_mount_id = _cache_mount_id(_proof_fd)
'''


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_mode & 0o022
        or before.st_nlink != 1
        or not 0 < before.st_size <= 16 * 1024 * 1024
    ):
        os.close(descriptor)
        raise ArtifactCacheSafetyError("canonical checksum manifest is unsafe")
    digest = hashlib.sha256()
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ):
        raise ArtifactCacheSafetyError("canonical checksum manifest changed while read")
    return digest.hexdigest()


def _ownership_value(
    kind: ArtifactKind,
    digest: str,
    *,
    cache_root: Path = FLEET_WORKER_CACHE_ROOT,
) -> str:
    return _canonical_json(
        {
            "schema_version": OWNERSHIP_SCHEMA,
            "kind": kind.value,
            "digest_sha256": digest,
            "cache_root": str(cache_root),
        }
    ).decode("utf-8")


def _link_anonymous_noreplace(descriptor: int, parent: int, name: str) -> None:
    """Atomically publish an O_TMPFILE without ever replacing a path."""

    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(libc, "linkat", None)
    if linkat is None:
        raise ArtifactCacheSafetyError("anonymous no-clobber publish is unavailable")
    linkat.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    linkat.restype = ctypes.c_int
    proc_path = f"/proc/self/fd/{descriptor}"
    proc_meta = os.lstat(proc_path)
    if not stat.S_ISLNK(proc_meta.st_mode):
        raise ArtifactCacheSafetyError("procfs descriptor publication is unavailable")
    result = linkat(-100, os.fsencode(proc_path), parent, os.fsencode(name), 0x400)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ArtifactCacheSafetyError("canonical OCI archive appeared during publish")
    raise ArtifactCacheSafetyError("anonymous canonical OCI publish failed")


def _open_canonical_root(*, create: bool) -> int:
    """Open and retain an exact no-follow descriptor for the canonical OCI root."""

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    current = PurePosixPath("/")
    try:
        for part in PurePosixPath(str(CANONICAL_OCI_ROOT)).parts[1:]:
            current /= part
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if (
                    not create
                    or not current.is_relative_to(PurePosixPath(str(CANONICAL_STATE_ROOT)))
                ):
                    raise ArtifactCacheSafetyError(
                        "canonical OCI directory ancestor is missing"
                    ) from None
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=descriptor)
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise ArtifactCacheSafetyError("canonical OCI directory is unsafe")
            if current.is_relative_to(PurePosixPath("/home/aday")):
                unsafe = metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022
            else:
                unsafe = bool(metadata.st_mode & 0o002)
            if unsafe:
                os.close(child)
                raise ArtifactCacheSafetyError("canonical OCI directory is unsafe")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _validate_oci_archive_fd(
    descriptor: int,
    *,
    digest: str,
    maximum_bytes: int,
    allowed_link_counts: frozenset[int],
) -> int:
    """Validate exact Docker-save semantics on one already anchored file."""

    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
        or metadata.st_nlink not in allowed_link_counts
        or not 0 < metadata.st_size <= maximum_bytes
    ):
        raise ArtifactCacheSafetyError("canonical OCI archive is unsafe")
    try:
        stream_fd = os.dup(descriptor)
        os.lseek(stream_fd, 0, os.SEEK_SET)
        with os.fdopen(stream_fd, "rb", closefd=True) as stream, tarfile.open(
            fileobj=stream, mode="r:"
        ) as bundle:
            members: list[tarfile.TarInfo] = []
            manifest_payload: bytes | None = None
            config_payload: bytes | None = None
            config_name = f"{digest}.json"
            for member in bundle:
                members.append(member)
                if len(members) > _OCI_TAR_MAX_MEMBERS:
                    raise ArtifactCacheSafetyError(
                        "canonical OCI archive member bound changed"
                    )
                parts = PurePosixPath(member.name).parts
                if (
                    not parts
                    or member.name.startswith("/")
                    or ".." in parts
                    or not (member.isfile() or member.isdir())
                    or member.islnk()
                    or member.issym()
                    or member.size < 0
                    or member.size > maximum_bytes
                ):
                    raise ArtifactCacheSafetyError(
                        "canonical OCI archive contains an unsafe member"
                    )
                if member.name in {"manifest.json", config_name}:
                    if not member.isfile() or not 0 < member.size <= _OCI_CONFIG_MAX_BYTES:
                        raise ArtifactCacheSafetyError(
                            "canonical OCI metadata member is unsafe"
                        )
                    extracted = bundle.extractfile(member)
                    if extracted is None:
                        raise ArtifactCacheSafetyError(
                            "canonical OCI metadata member is missing"
                        )
                    payload = extracted.read(_OCI_CONFIG_MAX_BYTES + 1)
                    if len(payload) != member.size:
                        raise ArtifactCacheSafetyError(
                            "canonical OCI metadata member changed"
                        )
                    if member.name == "manifest.json":
                        manifest_payload = payload
                    else:
                        config_payload = payload
            if not members:
                raise ArtifactCacheSafetyError(
                    "canonical OCI archive member bound changed"
                )
            by_name = {member.name: member for member in members}
            if len(by_name) != len(members):
                raise ArtifactCacheSafetyError("canonical OCI archive repeats a member")
            manifest_member = by_name.get("manifest.json")
            if (
                manifest_member is None
                or not manifest_member.isfile()
                or manifest_member.size > _OCI_CONFIG_MAX_BYTES
                or manifest_payload is None
            ):
                raise ArtifactCacheSafetyError("canonical OCI manifest is unsafe")
            manifest = json.loads(manifest_payload.decode("utf-8"))
            if (
                not isinstance(manifest, list)
                or len(manifest) != 1
                or not isinstance(manifest[0], dict)
                or set(manifest[0]) != {"Config", "RepoTags", "Layers"}
                or manifest[0].get("Config") != config_name
                or manifest[0].get("RepoTags") not in (None, [])
                or not isinstance(manifest[0].get("Layers"), list)
                or not manifest[0]["Layers"]
            ):
                raise ArtifactCacheSafetyError("canonical OCI image identity is absent")
            layers = manifest[0]["Layers"]
            if len(set(layers)) != len(layers) or any(
                not isinstance(layer, str)
                or layer.startswith("/")
                or ".." in PurePosixPath(layer).parts
                or not layer.endswith("/layer.tar")
                for layer in layers
            ):
                raise ArtifactCacheSafetyError("canonical OCI layer closure is unsafe")
            config_member = by_name.get(config_name)
            if (
                config_member is None
                or not config_member.isfile()
                or not 0 < config_member.size <= _OCI_CONFIG_MAX_BYTES
                or config_payload is None
            ):
                raise ArtifactCacheSafetyError("canonical OCI config is unsafe")
            if hashlib.sha256(config_payload).hexdigest() != digest:
                raise ArtifactCacheSafetyError("canonical OCI config digest changed")
            config = json.loads(config_payload.decode("utf-8"))
            if not isinstance(config, dict) or not isinstance(config.get("config"), dict):
                raise ArtifactCacheSafetyError("canonical OCI config is malformed")
            required_regular = {"manifest.json", config_name, *layers}
            optional_regular: set[str] = set()
            for layer in layers:
                parent = PurePosixPath(layer).parent.as_posix()
                optional_regular.update({f"{parent}/VERSION", f"{parent}/json"})
            actual_regular = {member.name for member in members if member.isfile()}
            if (
                not required_regular <= actual_regular
                or not actual_regular <= required_regular | optional_regular
            ):
                raise ArtifactCacheSafetyError(
                    "canonical OCI archive has unreferenced payloads"
                )
            allowed_dirs: set[str] = set()
            for name in actual_regular:
                parts = PurePosixPath(name).parts[:-1]
                allowed_dirs.update(
                    PurePosixPath(*parts[:depth]).as_posix()
                    for depth in range(1, len(parts) + 1)
                )
            actual_dirs = {
                member.name.rstrip("/") for member in members if member.isdir()
            }
            if actual_dirs != allowed_dirs:
                raise ArtifactCacheSafetyError(
                    "canonical OCI archive directory closure changed"
                )
            payload_end = max(
                member.offset_data + ((member.size + 511) // 512) * 512
                for member in members
            )
            trailing_size = metadata.st_size - payload_end
            if not 1024 <= trailing_size <= 1024 * 1024:
                raise ArtifactCacheSafetyError(
                    "canonical OCI archive framing changed"
                )
            trailing = os.pread(descriptor, trailing_size, payload_end)
            if len(trailing) != trailing_size or any(trailing):
                raise ArtifactCacheSafetyError(
                    "canonical OCI archive has trailing payloads"
                )
    except (OSError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCacheSafetyError("canonical OCI archive is malformed") from exc
    after = os.fstat(descriptor)
    if (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    ):
        raise ArtifactCacheSafetyError("canonical OCI archive changed while read")
    return metadata.st_size


def _validate_oci_archive(
    archive: Path, *, digest: str, maximum_bytes: int
) -> int:
    """Validate one no-follow canonical Docker-save archive path."""

    descriptor = os.open(
        archive, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        return _validate_oci_archive_fd(
            descriptor,
            digest=digest,
            maximum_bytes=maximum_bytes,
            allowed_link_counts=frozenset({1}),
        )
    finally:
        os.close(descriptor)


class AeonQwenArtifactBackend:
    """Exact worker-cache transport for the three standard Qwen artifacts."""

    # Subclasses may own one distinct, profile-bound cache root.  ``None``
    # intentionally keeps the historical backend coupled to the module-level
    # constant, including hermetic tests that replace that constant.
    WORKER_CACHE_ROOT: Path | None = None

    def __init__(
        self,
        *,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._run = command_runner
        self._popen = popen_factory
        self._clock = clock

    @staticmethod
    def _host(host: str) -> str:
        hostname = EXPECTED_HOSTNAMES.get(host)
        if hostname is None:
            raise ArtifactCacheSafetyError(
                "Qwen artifact cache host is not release-qualified"
            )
        return hostname

    @staticmethod
    def _ssh(host: str) -> list[str]:
        AeonQwenArtifactBackend._host(host)
        return [
            "/usr/bin/ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPath=none",
            "-o",
            "ControlPersist=no",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=6",
            f"aday@{network_address(host)}",
        ]

    @classmethod
    def _cache_root(cls) -> Path:
        return cls.WORKER_CACHE_ROOT or FLEET_WORKER_CACHE_ROOT

    @classmethod
    def _worker_path(cls, path: str, *, allow_root: bool = False) -> Path:
        value = Path(path)
        root = cls._cache_root()
        try:
            relative = PurePosixPath(str(value)).relative_to(PurePosixPath(str(root)))
        except ValueError as exc:
            raise ArtifactCacheSafetyError(
                "Qwen artifact path escaped its Fleet cache root"
            ) from exc
        if (
            ".." in PurePosixPath(str(value)).parts
            or (not relative.parts and not allow_root)
        ):
            raise ArtifactCacheSafetyError("Qwen artifact cache path is unsafe")
        return value

    def _ownership_marker(self, kind: ArtifactKind, digest: str) -> str:
        return _ownership_value(kind, digest, cache_root=self._cache_root())

    def _remote_python(
        self,
        host: str,
        script: str,
        *arguments: str,
        timeout: float = 120,
        prove_root: bool = True,
    ) -> dict[str, Any]:
        hostname = self._host(host)
        command = [
            *self._ssh(host),
            shlex.join(
                [
                    "/usr/bin/env",
                    "-i",
                    "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
                    "HOME=/home/aday",
                    "LANG=C",
                    "LC_ALL=C",
                    "/usr/bin/bash",
                    REMOTE_WRAPPER,
                    REMOTE_PYTHON,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    (_REMOTE_CACHE_ROOT_PROOF + "\n" + script)
                    if prove_root
                    else script,
                    hostname,
                    str(self._cache_root()),
                    *arguments,
                ]
            ),
        ]
        result = self._run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if (
            result.returncode != 0
            or len(result.stdout or "") > 1024 * 1024
            or len(result.stderr or "") > 8192
        ):
            raise ArtifactCacheSafetyError("Qwen worker cache proof failed")
        try:
            value = json.loads(result.stdout)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactCacheSafetyError(
                "Qwen worker cache proof is malformed"
            ) from exc
        if not isinstance(value, dict):
            raise ArtifactCacheSafetyError("Qwen worker cache proof is malformed")
        return value

    def _run_with_progress(
        self,
        command: list[str],
        *,
        progress: Callable[[int, int], None],
        total: int,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        stdout: Any = subprocess.PIPE,
        timeout: float = 7200,
        progress_probe: Callable[[], int] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> tuple[str, str]:
        completed = 0
        progress(completed, total)
        popen_arguments: dict[str, Any] = {
            "cwd": cwd,
            "env": None if env is None else dict(env),
            "stdin": subprocess.DEVNULL,
            "stdout": stdout,
            "stderr": subprocess.PIPE,
        }
        if pass_fds:
            popen_arguments["pass_fds"] = pass_fds
        process = self._popen(command, **popen_arguments)
        deadline = self._clock() + timeout
        try:
            while True:
                try:
                    output, error = process.communicate(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    if self._clock() >= deadline:
                        raise ArtifactCacheError("Qwen artifact transfer timed out")
                    if progress_probe is not None:
                        observed = progress_probe()
                        if isinstance(observed, bool) or not isinstance(observed, int):
                            raise ArtifactCacheSafetyError(
                                "Qwen artifact progress proof is malformed"
                            )
                        if not 0 <= observed <= total:
                            raise ArtifactCacheSafetyError(
                                "Qwen artifact progress exceeded its bound"
                            )
                        completed = max(completed, observed)
                    progress(completed, total)
        except BaseException:
            if process.returncode is None:
                try:
                    process.terminate()
                except (AttributeError, OSError):
                    pass
                try:
                    process.communicate(timeout=5)
                except (AttributeError, OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                    except (AttributeError, OSError):
                        pass
                    try:
                        process.communicate(timeout=5)
                    except (AttributeError, OSError, subprocess.TimeoutExpired) as exc:
                        raise ArtifactCacheSafetyError(
                            "Qwen artifact child could not be reaped"
                        ) from exc
            raise
        if process.returncode != 0:
            raise ArtifactCacheError("Qwen artifact transfer command failed")
        if progress_probe is not None:
            observed = progress_probe()
            if isinstance(observed, bool) or not isinstance(observed, int):
                raise ArtifactCacheSafetyError(
                    "Qwen artifact progress proof is malformed"
                )
            if not 0 <= observed <= total:
                raise ArtifactCacheSafetyError(
                    "Qwen artifact progress exceeded its bound"
                )
            completed = max(completed, observed)
            progress(completed, total)
        return (
            "" if output is None else output.decode("utf-8", errors="replace"),
            "" if error is None else error.decode("utf-8", errors="replace"),
        )

    @staticmethod
    def _manifest_files(descriptor: ArtifactDescriptor) -> tuple[str, ...]:
        if (
            descriptor.kind is not ArtifactKind.MANIFESTED_TREE
            or descriptor.manifest_format != "sha256sum-v1"
            or descriptor.manifest_path is None
        ):
            raise ArtifactCacheSafetyError("Qwen manifested artifact contract changed")
        root = Path(descriptor.canonical_path)
        manifest = Path(descriptor.manifest_path)
        try:
            relative_manifest = manifest.relative_to(root).as_posix()
        except ValueError as exc:
            raise ArtifactCacheSafetyError(
                "Qwen artifact manifest escaped its canonical root"
            ) from exc
        metadata = manifest.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
            or metadata.st_nlink != 1
            or metadata.st_size > 16 * 1024 * 1024
        ):
            raise ArtifactCacheSafetyError("Qwen artifact manifest is unsafe")
        manifest_fd = os.open(
            manifest, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        try:
            anchored = os.fstat(manifest_fd)
            if (
                anchored.st_dev,
                anchored.st_ino,
                anchored.st_mode,
                anchored.st_uid,
                anchored.st_nlink,
                anchored.st_size,
            ) != (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_uid,
                metadata.st_nlink,
                metadata.st_size,
            ):
                raise ArtifactCacheSafetyError("Qwen artifact manifest changed")
            payload = b""
            while True:
                chunk = os.read(manifest_fd, 1024 * 1024)
                if not chunk:
                    break
                payload += chunk
                if len(payload) > 16 * 1024 * 1024:
                    raise ArtifactCacheSafetyError("Qwen artifact manifest is unsafe")
            after = os.fstat(manifest_fd)
        finally:
            os.close(manifest_fd)
        if (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            anchored.st_dev,
            anchored.st_ino,
            anchored.st_mode,
            anchored.st_uid,
            anchored.st_nlink,
            anchored.st_size,
            anchored.st_mtime_ns,
            anchored.st_ctime_ns,
        ):
            raise ArtifactCacheSafetyError("Qwen artifact manifest changed while read")
        if hashlib.sha256(payload).hexdigest() != descriptor.digest_sha256:
            raise ArtifactCacheSafetyError("Qwen artifact manifest digest changed")
        files: list[str] = []
        seen: set[str] = set()
        require_unaliased_files = (
            descriptor.artifact_id == QWEN_SOURCE_CACHE_ARTIFACT_ID
        )
        for raw in payload.decode("utf-8").splitlines():
            match = re.fullmatch(r"([a-f0-9]{64}) [ *](.+)", raw)
            if match is None:
                raise ArtifactCacheSafetyError("Qwen artifact manifest is malformed")
            relative = match.group(2)
            parts = PurePosixPath(relative).parts
            if (
                not parts
                or relative.startswith("/")
                or ".." in parts
                or relative in seen
            ):
                raise ArtifactCacheSafetyError("Qwen artifact manifest path is unsafe")
            path = root / relative
            item = path.lstat()
            if (
                not stat.S_ISREG(item.st_mode)
                or item.st_uid != os.geteuid()
                or item.st_mode & 0o022
                or (require_unaliased_files and item.st_nlink != 1)
            ):
                raise ArtifactCacheSafetyError("Qwen canonical artifact file is unsafe")
            seen.add(relative)
            files.append(relative)
        if relative_manifest in seen:
            raise ArtifactCacheSafetyError(
                "Qwen artifact manifest recursively lists itself"
            )
        return tuple([*files, relative_manifest])

    @staticmethod
    def _validate_descriptor(descriptor: ArtifactDescriptor) -> None:
        identities = {
            QWEN_SOURCE_CACHE_ARTIFACT_ID: ArtifactKind.MANIFESTED_TREE,
            QWEN_MODEL_CACHE_ARTIFACT_ID: ArtifactKind.MANIFESTED_TREE,
            QWEN_IMAGE_CACHE_ARTIFACT_ID: ArtifactKind.OCI_ARCHIVE,
        }
        if identities.get(descriptor.artifact_id) is not descriptor.kind:
            raise ArtifactCacheSafetyError("unsupported Qwen cache artifact")
        if descriptor.artifact_id == QWEN_SOURCE_CACHE_ARTIFACT_ID:
            if (
                descriptor.identity_key != "runtime_source"
                or Path(descriptor.canonical_path) != PACKAGE_ROOT
                or descriptor.digest_sha256
                != _sha256(PACKAGE_ROOT / SOURCE_MANIFEST_FILE)
                or Path(str(descriptor.manifest_path))
                != PACKAGE_ROOT / SOURCE_MANIFEST_FILE
                or descriptor.manifest_format != "sha256sum-v1"
                or descriptor.size_bytes_max != QWEN_SOURCE_CACHE_MAX_BYTES
                or descriptor.inode_count_max != QWEN_SOURCE_CACHE_MAX_INODES
                or descriptor.transfer_bytes_max
                != QWEN_SOURCE_CACHE_TRANSFER_MAX_BYTES
                or descriptor.cold_peak_bytes_max
                != QWEN_SOURCE_CACHE_COLD_PEAK_BYTES
            ):
                raise ArtifactCacheSafetyError("Qwen source cache origin changed")
        elif descriptor.artifact_id == QWEN_MODEL_CACHE_ARTIFACT_ID:
            if (
                descriptor.identity_key != "model_sha256s"
                or Path(descriptor.canonical_path) != CANONICAL_MODEL_ROOT
                or descriptor.digest_sha256 != CANONICAL_MODEL_SHA256SUMS
                or Path(str(descriptor.manifest_path))
                != CANONICAL_MODEL_ROOT / "SHA256SUMS"
                or descriptor.manifest_format != "sha256sum-v1"
                or descriptor.size_bytes_max != QWEN_MODEL_CACHE_MAX_BYTES
                or descriptor.inode_count_max != QWEN_MODEL_CACHE_MAX_INODES
                or descriptor.transfer_bytes_max
                != QWEN_MODEL_CACHE_TRANSFER_MAX_BYTES
                or descriptor.cold_peak_bytes_max
                != QWEN_MODEL_CACHE_COLD_PEAK_BYTES
            ):
                raise ArtifactCacheSafetyError("Qwen model cache identity changed")
        else:
            expected = Path(descriptor.canonical_path)
            try:
                expected.relative_to(CANONICAL_OCI_ROOT)
            except ValueError as exc:
                raise ArtifactCacheSafetyError("Qwen OCI origin changed") from exc
            if (
                descriptor.identity_key != "image"
                or descriptor.digest_sha256 != QWEN_STANDARD_IMAGE_CONFIG_SHA256
                or expected
                != CANONICAL_OCI_ROOT / f"{descriptor.digest_sha256}.tar"
                or descriptor.size_bytes_max
                != QWEN_IMAGE_CACHE_RECEIPT_MAX_BYTES
                or descriptor.inode_count_max != QWEN_IMAGE_CACHE_MAX_INODES
                or descriptor.transfer_bytes_max != QWEN_IMAGE_ARCHIVE_MAX_BYTES
                or descriptor.cold_peak_bytes_max
                != QWEN_IMAGE_CACHE_COLD_PEAK_BYTES
                or descriptor.manifest_path is not None
                or descriptor.manifest_format is not None
            ):
                raise ArtifactCacheSafetyError("Qwen OCI origin identity changed")

    @staticmethod
    def _verify_canonical_tree(
        descriptor: ArtifactDescriptor,
        progress_check: Callable[[], None],
    ) -> None:
        progress_check()
        if descriptor.artifact_id == QWEN_SOURCE_CACHE_ARTIFACT_ID:
            source = _source_identity(PACKAGE_ROOT, PACKAGE_ROOT / ".identity-only")
            if source.manifest_sha256 != descriptor.digest_sha256:
                raise ArtifactCacheSafetyError("Qwen source cache identity changed")
            return
        identity = load_artifact_identity(
            Path(descriptor.canonical_path),
            verify_payload=True,
            progress_check=progress_check,
        )
        if (
            identity.sha256s_sha256 != descriptor.digest_sha256
            or identity.total_bytes > descriptor.size_bytes_max
        ):
            raise ArtifactCacheSafetyError("Qwen model cache payload changed")

    def ensure_cache_root(
        self, *, host: str, root: str, owner_uid: int
    ) -> CacheRootInspection:
        if Path(root) != self._cache_root() or owner_uid != os.geteuid():
            raise ArtifactCacheSafetyError("Qwen worker cache root changed")
        script = r'''
import json, os, pathlib, stat, sys
expected, fixed_root, requested, uid_raw = sys.argv[1:5]
assert os.uname().nodename == expected
assert requested == fixed_root
assert int(uid_raw) == os.geteuid()
target = pathlib.PurePosixPath(requested)
assert target.is_absolute() and ".." not in target.parts
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
descriptor = os.open("/", flags)
current = pathlib.PurePosixPath("/")
try:
    for part in target.parts[1:]:
        current = current / part
        try:
            child = os.open(part, flags, dir_fd=descriptor)
        except FileNotFoundError:
            assert current.is_relative_to(pathlib.PurePosixPath("/home/aday/.local"))
            os.mkdir(part, mode=0o700, dir_fd=descriptor)
            child = os.open(part, flags, dir_fd=descriptor)
        child_meta = os.fstat(child)
        assert stat.S_ISDIR(child_meta.st_mode)
        if current.is_relative_to(pathlib.PurePosixPath("/home/aday")):
            assert child_meta.st_uid == os.geteuid() and not child_meta.st_mode & 0o022
        else:
            assert not child_meta.st_mode & 0o002
        os.close(descriptor); descriptor = child
    meta = os.fstat(descriptor)
    assert meta.st_uid == os.geteuid() and not meta.st_mode & 0o077
    values = os.fstatvfs(descriptor)
finally:
    os.close(descriptor)
print(json.dumps({"filesystem_id": str(meta.st_dev), "owner_uid": meta.st_uid,
 "is_directory": True, "is_symlink": False,
 "free_bytes": values.f_bavail * values.f_frsize,
 "free_inodes": values.f_favail}, sort_keys=True))
'''
        value = self._remote_python(
            host, script, root, str(owner_uid), prove_root=False
        )
        return CacheRootInspection(**value)

    def inspect_entry(
        self,
        *,
        host: str,
        path: str,
        descriptor: ArtifactDescriptor,
        expected_filesystem_id: str,
        verify_content: bool,
    ) -> CacheEntryInspection | None:
        self._validate_descriptor(descriptor)
        checked = self._worker_path(path)
        if not isinstance(verify_content, bool):
            raise ArtifactCacheSafetyError("Qwen cache verification mode changed")
        if descriptor.kind is ArtifactKind.OCI_ARCHIVE:
            return self._inspect_oci(
                host,
                checked,
                descriptor,
                expected_filesystem_id,
                verify_content=verify_content,
            )
        return self._inspect_tree(
            host,
            checked,
            descriptor,
            expected_filesystem_id,
            verify_content=verify_content,
        )

    def _inspect_tree(
        self,
        host: str,
        path: Path,
        descriptor: ArtifactDescriptor,
        filesystem_id: str,
        *,
        verify_content: bool,
    ) -> CacheEntryInspection | None:
        manifest_relative = Path(str(descriptor.manifest_path)).relative_to(
            Path(descriptor.canonical_path)
        ).as_posix()
        script = r'''
import hashlib, json, os, pathlib, re, stat, sys
expected, fixed_root, raw, fsid, manifest_rel, digest, verify_raw, attr_name, attr_value, max_bytes_raw, max_inodes_raw = sys.argv[1:12]
assert os.uname().nodename == expected
max_bytes, max_inodes = int(max_bytes_raw), int(max_inodes_raw)
root, path = pathlib.Path(fixed_root), pathlib.Path(raw)
def _cache_mount_id(fd):
    with open(f"/proc/self/fdinfo/{fd}","r",encoding="ascii") as info:
        values=[line.split(":",1)[1].strip() for line in info if line.startswith("mnt_id:")]
    assert len(values)==1 and values[0].isdecimal(); return values[0]
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
root_fd = os.open("/", flags)
for part in pathlib.PurePosixPath(fixed_root).parts[1:]:
    child = os.open(part, flags, dir_fd=root_fd)
    os.close(root_fd); root_fd = child
root_meta = os.fstat(root_fd); root_mount = _cache_mount_id(root_fd)
assert root_meta.st_uid == os.geteuid() and str(root_meta.st_dev) == fsid
try:
    relative = path.relative_to(root); meta = path.lstat()
except FileNotFoundError:
    print(json.dumps({"state": "absent"})); raise SystemExit(0)
assert relative.parts and ".." not in relative.parts
assert stat.S_ISDIR(meta.st_mode) and not stat.S_ISLNK(meta.st_mode)
assert meta.st_uid == os.geteuid() and not meta.st_mode & 0o022
assert str(meta.st_dev) == fsid
target_fd = os.dup(root_fd)
for part in relative.parts:
    child = os.open(part, flags, dir_fd=target_fd)
    assert _cache_mount_id(child) == root_mount
    os.close(target_fd); target_fd = child
anchored = os.fstat(target_fd)
assert (anchored.st_dev,anchored.st_ino,anchored.st_mode,anchored.st_uid,
        anchored.st_nlink,anchored.st_size,anchored.st_mtime_ns,
        anchored.st_ctime_ns) == (meta.st_dev,meta.st_ino,meta.st_mode,meta.st_uid,
        meta.st_nlink,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns)
def open_item(relative, directory):
    fd=os.dup(target_fd); parts=pathlib.PurePosixPath(relative).parts
    try:
        for index, part in enumerate(parts):
            wanted=(flags if directory or index+1 < len(parts) else
                    os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
            child=os.open(part,wanted,dir_fd=fd)
            assert _cache_mount_id(child) == root_mount
            os.close(fd); fd=child
        return fd
    except Exception:
        os.close(fd); raise
def prove_item(relative, item_meta):
    fd=open_item(relative,stat.S_ISDIR(item_meta.st_mode))
    try:
        anchored=os.fstat(fd)
        assert (anchored.st_dev,anchored.st_ino,anchored.st_mode,anchored.st_uid,
                anchored.st_nlink,anchored.st_size,anchored.st_mtime_ns,
                anchored.st_ctime_ns) == (item_meta.st_dev,item_meta.st_ino,
                item_meta.st_mode,item_meta.st_uid,item_meta.st_nlink,
                item_meta.st_size,item_meta.st_mtime_ns,item_meta.st_ctime_ns)
    finally:
        os.close(fd)
try:
    fleet_owned = os.getxattr(target_fd, attr_name).decode("utf-8") == attr_value
except OSError:
    fleet_owned = False
is_staging = path.parent == root / ".staging"
def token(records):
    if not fleet_owned:
        return None
    material = {"schema":1,"path":str(path),"marker":attr_value,
      "root":[meta.st_dev,meta.st_ino,meta.st_mode,meta.st_uid,meta.st_nlink,
              meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns],"members":records}
    return hashlib.sha256(json.dumps(material,sort_keys=True,separators=(",",":"),
                                      allow_nan=False).encode("utf-8")).hexdigest()
def snapshot():
    records, total = [], 0
    for item in path.rglob("*"):
        item_meta = item.lstat(); rel = item.relative_to(path).as_posix()
        assert item_meta.st_uid == os.geteuid() and str(item_meta.st_dev) == fsid
        assert not stat.S_ISLNK(item_meta.st_mode) and not item_meta.st_mode & 0o022
        if stat.S_ISDIR(item_meta.st_mode):
            assert not os.path.ismount(item)
        elif stat.S_ISREG(item_meta.st_mode):
            assert item_meta.st_nlink == 1 and not os.path.ismount(item)
            total += item_meta.st_size
        else:
            raise AssertionError
        prove_item(rel,item_meta)
        records.append([rel,item_meta.st_dev,item_meta.st_ino,item_meta.st_mode,
                        item_meta.st_uid,item_meta.st_nlink,item_meta.st_size,
                        item_meta.st_mtime_ns,item_meta.st_ctime_ns])
        assert total <= max_bytes and 1 + len(records) <= max_inodes
    return sorted(records), total
def invalid(size=0, inodes=1, can_resume=True):
    records, size = snapshot()
    inodes = 1 + len(records)
    if is_staging and verify_raw == "1" and can_resume:
        print(json.dumps({"state":"resumable"}, sort_keys=True))
        raise SystemExit(0)
    print(json.dumps({"state":"invalid", "filesystem_id":str(meta.st_dev),
     "owner_uid":meta.st_uid, "fleet_owned":fleet_owned,
     "identity_token":token(records), "is_staging":is_staging,
     "size_bytes":size, "inode_count":inodes}, sort_keys=True))
    raise SystemExit(0)
manifest = path / manifest_rel
try:
    manifest_meta = manifest.lstat()
    if not stat.S_ISREG(manifest_meta.st_mode) or stat.S_ISLNK(manifest_meta.st_mode):
        invalid()
    if (manifest_meta.st_uid != os.geteuid() or str(manifest_meta.st_dev) != fsid
            or manifest_meta.st_nlink != 1 or os.path.ismount(manifest)):
        invalid()
    if not 0 < manifest_meta.st_size <= 16*1024*1024:
        invalid(manifest_meta.st_size)
    manifest_fd=open_item(manifest_rel,False)
    try:
        anchored=os.fstat(manifest_fd)
        assert (anchored.st_dev,anchored.st_ino,anchored.st_mode,anchored.st_uid,
                anchored.st_nlink,anchored.st_size,anchored.st_mtime_ns,
                anchored.st_ctime_ns) == (manifest_meta.st_dev,manifest_meta.st_ino,
                manifest_meta.st_mode,manifest_meta.st_uid,manifest_meta.st_nlink,
                manifest_meta.st_size,manifest_meta.st_mtime_ns,manifest_meta.st_ctime_ns)
        payload=b""
        while True:
            chunk=os.read(manifest_fd,1024*1024)
            if not chunk: break
            payload += chunk; assert len(payload) <= 16*1024*1024
        after=os.fstat(manifest_fd)
        assert (after.st_dev,after.st_ino,after.st_mode,after.st_uid,after.st_nlink,
                after.st_size,after.st_mtime_ns,after.st_ctime_ns) == (
                anchored.st_dev,anchored.st_ino,anchored.st_mode,anchored.st_uid,
                anchored.st_nlink,anchored.st_size,anchored.st_mtime_ns,
                anchored.st_ctime_ns)
    finally:
        os.close(manifest_fd)
except OSError:
    invalid()
if hashlib.sha256(payload).hexdigest() != digest:
    invalid()
expected_files = {manifest_rel}; checks = []
for line in payload.decode("utf-8").splitlines():
    match = re.fullmatch(r"([a-f0-9]{64}) [ *](.+)", line)
    if (match is None or match.group(2).startswith("/")
            or ".." in pathlib.PurePosixPath(match.group(2)).parts
            or match.group(2) in expected_files):
        invalid()
    expected_files.add(match.group(2)); checks.append((match.group(1), match.group(2)))
actual_files, actual_dirs, total, records = set(), set(), 0, []
for item in path.rglob("*"):
    item_meta = item.lstat(); rel = item.relative_to(path).as_posix()
    if stat.S_ISDIR(item_meta.st_mode):
        if (item_meta.st_uid != os.geteuid() or item_meta.st_mode & 0o022
                or str(item_meta.st_dev) != fsid or stat.S_ISLNK(item_meta.st_mode)
                or os.path.ismount(item)):
            raise AssertionError
        actual_dirs.add(rel)
    elif stat.S_ISREG(item_meta.st_mode):
        if (item_meta.st_uid != os.geteuid() or item_meta.st_mode & 0o022
                or str(item_meta.st_dev) != fsid or stat.S_ISLNK(item_meta.st_mode)
                or item_meta.st_nlink != 1 or os.path.ismount(item)):
            raise AssertionError
        actual_files.add(rel); total += item_meta.st_size
    else:
        raise AssertionError
    prove_item(rel,item_meta)
    records.append([rel,item_meta.st_dev,item_meta.st_ino,item_meta.st_mode,
                    item_meta.st_uid,item_meta.st_nlink,item_meta.st_size,
                    item_meta.st_mtime_ns,item_meta.st_ctime_ns])
    if total > max_bytes or 1 + len(records) > max_inodes:
        invalid(total, 1 + len(records))
if actual_files != expected_files:
    invalid(can_resume=not bool(actual_files - expected_files))
expected_dirs = set()
for rel in expected_files:
    parts = pathlib.PurePosixPath(rel).parts[:-1]
    expected_dirs.update(pathlib.PurePosixPath(*parts[:i]).as_posix()
                         for i in range(1, len(parts)+1))
if actual_dirs != expected_dirs:
    invalid(can_resume=not bool(actual_dirs - expected_dirs))
if verify_raw == "1":
    record_by_name={record[0]:record[1:] for record in records}
    for wanted, rel in checks:
        hasher = hashlib.sha256(); fd=open_item(rel,False)
        try:
            while True:
                chunk=os.read(fd,1024*1024)
                if not chunk: break
                hasher.update(chunk)
            anchored=os.fstat(fd)
            assert [anchored.st_dev,anchored.st_ino,anchored.st_mode,
                    anchored.st_uid,anchored.st_nlink,anchored.st_size,
                    anchored.st_mtime_ns,anchored.st_ctime_ns] == record_by_name[rel]
        finally:
            os.close(fd)
        if hasher.hexdigest() != wanted:
            invalid()
after_root=os.fstat(target_fd)
assert (after_root.st_dev,after_root.st_ino,after_root.st_mode,after_root.st_uid,
        after_root.st_nlink,after_root.st_size,after_root.st_mtime_ns,
        after_root.st_ctime_ns) == (meta.st_dev,meta.st_ino,meta.st_mode,meta.st_uid,
        meta.st_nlink,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns)
print(json.dumps({"state":"present", "filesystem_id":str(meta.st_dev),
 "owner_uid":meta.st_uid, "size_bytes":total, "fleet_owned":fleet_owned,
 "inode_count":1+len(actual_files)+len(actual_dirs),
 "identity_token":token(sorted(records))}, sort_keys=True))
'''
        value = self._remote_python(
            host,
            script,
            str(path),
            filesystem_id,
            manifest_relative,
            descriptor.digest_sha256,
            "1" if verify_content else "0",
            OWNERSHIP_XATTR,
            self._ownership_marker(descriptor.kind, descriptor.digest_sha256),
            str(descriptor.size_bytes_max),
            str(descriptor.inode_count_max),
            timeout=3600 if verify_content else 60,
        )
        if value.get("state") in {"absent", "resumable"}:
            return None
        if value.get("state") == "invalid":
            return CacheEntryInspection(
                filesystem_id=str(value["filesystem_id"]),
                owner_uid=int(value["owner_uid"]),
                kind=ArtifactKind.MANIFESTED_TREE,
                semantic_digest_sha256=descriptor.digest_sha256,
                size_bytes=max(0, int(value.get("size_bytes", 0))),
                inode_count=max(1, int(value.get("inode_count", 1))),
                is_regular_file=False,
                is_directory=True,
                exact_members=False,
                payload_sha256=descriptor.digest_sha256,
                fleet_owned=bool(value.get("fleet_owned")),
                content_identity_verified=False,
                semantic_ready=not bool(value.get("is_staging")),
                identity_token=value.get("identity_token"),
            )
        if value.get("state") != "present":
            raise ArtifactCacheSafetyError("Qwen tree inspection is malformed")
        return CacheEntryInspection(
            filesystem_id=str(value["filesystem_id"]),
            owner_uid=int(value["owner_uid"]),
            kind=ArtifactKind.MANIFESTED_TREE,
            semantic_digest_sha256=descriptor.digest_sha256,
            size_bytes=int(value["size_bytes"]),
            inode_count=int(value["inode_count"]),
            is_regular_file=False,
            is_directory=True,
            exact_members=True,
            payload_sha256=descriptor.digest_sha256,
            fleet_owned=bool(value.get("fleet_owned")),
            content_identity_verified=verify_content,
            semantic_ready=True,
            identity_token=value.get("identity_token"),
        )

    def _remote_image_inspection(
        self, host: str, image_id: str
    ) -> dict[str, Any] | None:
        command = [
            *self._ssh(host),
            shlex.join(
                [
                    "/usr/bin/env",
                    "-i",
                    "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
                    "HOME=/home/aday",
                    "LANG=C",
                    "LC_ALL=C",
                    "/usr/bin/bash",
                    REMOTE_WRAPPER,
                    "/usr/bin/bash",
                    REMOTE_DOCKER,
                    "image",
                    "inspect",
                    image_id,
                ]
            ),
        ]
        result = self._run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            absent_messages = {
                f"Error response from daemon: No such image: {image_id}",
                f"Error: No such image: {image_id}",
            }
            if (
                result.returncode == 1
                and (result.stdout or "").strip() in {"", "[]"}
                and (result.stderr or "").strip() in absent_messages
            ):
                return None
            raise ArtifactCacheSafetyError(
                "remote Qwen image availability is ambiguous"
            )
        try:
            payload = json.loads(result.stdout)
            item = payload[0]
            if item.get("Id") != image_id:
                raise ValueError
            size = int(item["Size"])
            if not 0 < size <= MAX_IMAGE_LOGICAL_BYTES:
                raise ValueError
            _normalise_image_config(item["Config"])
            return item
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactCacheSafetyError(
                "remote Qwen image proof is malformed"
            ) from exc

    def _inspect_oci(
        self,
        host: str,
        path: Path,
        descriptor: ArtifactDescriptor,
        filesystem_id: str,
        *,
        verify_content: bool,
    ) -> CacheEntryInspection | None:
        script = r'''
import hashlib, json, os, pathlib, stat, sys
expected, fixed_root, raw, fsid, attr_name, attr_value, verify_raw = sys.argv[1:8]
assert os.uname().nodename == expected
root, path = pathlib.Path(fixed_root), pathlib.Path(raw)
def mount_id(fd):
    with open(f"/proc/self/fdinfo/{fd}","r",encoding="ascii") as info:
        values=[line.split(":",1)[1].strip() for line in info if line.startswith("mnt_id:")]
    assert len(values)==1 and values[0].isdecimal(); return values[0]
flags=os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC|os.O_NOFOLLOW
root_fd=os.open("/",flags)
for part in pathlib.PurePosixPath(fixed_root).parts[1:]:
    child=os.open(part,flags,dir_fd=root_fd)
    os.close(root_fd); root_fd=child
root_meta = os.fstat(root_fd); root_mount=mount_id(root_fd)
assert root_meta.st_uid == os.geteuid() and str(root_meta.st_dev) == fsid
try:
    relative = path.relative_to(root); meta = path.lstat()
except FileNotFoundError:
    print(json.dumps({"state":"absent"})); raise SystemExit(0)
assert relative.parts and ".." not in relative.parts
assert stat.S_ISREG(meta.st_mode) and not stat.S_ISLNK(meta.st_mode)
assert meta.st_uid == os.geteuid() and not meta.st_mode & 0o022
assert str(meta.st_dev) == fsid and meta.st_nlink == 1
target_fd=os.dup(root_fd)
for index, part in enumerate(relative.parts):
    wanted=(os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW
            if index+1 == len(relative.parts) else flags)
    child=os.open(part,wanted,dir_fd=target_fd)
    assert mount_id(child) == root_mount
    os.close(target_fd); target_fd=child
anchored=os.fstat(target_fd)
assert (anchored.st_dev,anchored.st_ino,anchored.st_mode,anchored.st_uid,
        anchored.st_nlink,anchored.st_size,anchored.st_mtime_ns,
        anchored.st_ctime_ns) == (meta.st_dev,meta.st_ino,meta.st_mode,meta.st_uid,
        meta.st_nlink,meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns)
try:
    fleet_owned = os.getxattr(target_fd, attr_name).decode("utf-8") == attr_value
except OSError:
    fleet_owned = False
material = {"schema":1,"path":str(path),"marker":attr_value,
 "root":[meta.st_dev,meta.st_ino,meta.st_mode,meta.st_uid,meta.st_nlink,
         meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns],"members":[]}
identity_token = (hashlib.sha256(json.dumps(material,sort_keys=True,
                  separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
                  if fleet_owned else None)
is_staging = path.parent == root / ".staging"
try:
    if meta.st_size > 64*1024:
        raise ValueError
    chunks=[]; remaining=64*1024+1
    while remaining > 0:
        chunk=os.read(target_fd,min(64*1024,remaining))
        if not chunk:
            break
        chunks.append(chunk); remaining-=len(chunk)
    payload=b"".join(chunks)
    if len(payload) > 64*1024:
        raise ValueError
    receipt = json.loads(payload.decode("utf-8"))
    after=os.fstat(target_fd)
    assert (after.st_dev,after.st_ino,after.st_mode,after.st_uid,after.st_nlink,
            after.st_size,after.st_mtime_ns,after.st_ctime_ns) == (
            anchored.st_dev,anchored.st_ino,anchored.st_mode,anchored.st_uid,
            anchored.st_nlink,anchored.st_size,anchored.st_mtime_ns,
            anchored.st_ctime_ns)
except Exception:
    if is_staging and verify_raw == "1":
        print(json.dumps({"state":"resumable"}, sort_keys=True))
        raise SystemExit(0)
    print(json.dumps({"state":"invalid", "filesystem_id":str(meta.st_dev),
     "owner_uid":meta.st_uid, "size_bytes":meta.st_size,
     "fleet_owned":fleet_owned, "identity_token":identity_token,
     "is_staging":is_staging}, sort_keys=True))
    raise SystemExit(0)
print(json.dumps({"state":"present", "filesystem_id":str(meta.st_dev),
 "owner_uid":meta.st_uid, "size_bytes":meta.st_size,
 "fleet_owned":fleet_owned, "receipt":receipt,
 "identity_token":identity_token, "is_staging":is_staging}, sort_keys=True))
'''
        value = self._remote_python(
            host,
            script,
            str(path),
            filesystem_id,
            OWNERSHIP_XATTR,
            self._ownership_marker(descriptor.kind, descriptor.digest_sha256),
            "1" if verify_content else "0",
        )
        if value.get("state") in {"absent", "resumable"}:
            return None
        if value.get("state") == "invalid":
            return CacheEntryInspection(
                filesystem_id=str(value["filesystem_id"]),
                owner_uid=int(value["owner_uid"]),
                kind=ArtifactKind.OCI_ARCHIVE,
                semantic_digest_sha256=descriptor.digest_sha256,
                size_bytes=max(1, int(value["size_bytes"])),
                inode_count=1,
                is_regular_file=True,
                is_directory=False,
                exact_members=False,
                payload_sha256=None,
                fleet_owned=bool(value.get("fleet_owned")),
                content_identity_verified=False,
                semantic_ready=not bool(value.get("is_staging")),
                identity_token=value.get("identity_token"),
            )
        def invalid_receipt() -> CacheEntryInspection:
            return CacheEntryInspection(
                filesystem_id=str(value["filesystem_id"]),
                owner_uid=int(value["owner_uid"]),
                kind=ArtifactKind.OCI_ARCHIVE,
                semantic_digest_sha256=descriptor.digest_sha256,
                size_bytes=max(1, int(value["size_bytes"])),
                inode_count=1,
                is_regular_file=True,
                is_directory=False,
                exact_members=False,
                payload_sha256=None,
                fleet_owned=bool(value.get("fleet_owned")),
                content_identity_verified=False,
                semantic_ready=not bool(value.get("is_staging")),
                identity_token=value.get("identity_token"),
            )

        receipt = value.get("receipt")
        image_id = f"sha256:{descriptor.digest_sha256}"
        if (
            not isinstance(receipt, dict)
            or set(receipt)
            != {
                "schema_version",
                "image_id",
                "image_size_bytes",
                "archive_payload_sha256",
            }
            or receipt.get("schema_version") != OCI_RECEIPT_SCHEMA
            or receipt.get("image_id") != image_id
            or isinstance(receipt.get("image_size_bytes"), bool)
            or not isinstance(receipt.get("image_size_bytes"), int)
            or not 0 < receipt["image_size_bytes"] <= MAX_IMAGE_LOGICAL_BYTES
            or not isinstance(receipt.get("archive_payload_sha256"), str)
            or _SHA256_RE.fullmatch(receipt["archive_payload_sha256"]) is None
        ):
            if bool(value.get("is_staging")) and verify_content:
                return None
            return invalid_receipt()
        image = self._remote_image_inspection(host, image_id)
        semantic_ready = (
            image is not None
            and int(image["Size"]) == receipt.get("image_size_bytes")
        )
        return CacheEntryInspection(
            filesystem_id=str(value["filesystem_id"]),
            owner_uid=int(value["owner_uid"]),
            kind=ArtifactKind.OCI_ARCHIVE,
            semantic_digest_sha256=descriptor.digest_sha256,
            size_bytes=int(value["size_bytes"]),
            inode_count=1,
            is_regular_file=True,
            is_directory=False,
            exact_members=True,
            payload_sha256=receipt["archive_payload_sha256"],
            fleet_owned=bool(value.get("fleet_owned")),
            content_identity_verified=True,
            semantic_ready=semantic_ready,
            identity_token=value.get("identity_token"),
        )

    def stage(
        self,
        *,
        host: str,
        descriptor: ArtifactDescriptor,
        temporary_path: str,
        expected_filesystem_id: str,
        max_bytes_per_second: int,
        progress: Callable[[int, int], None],
    ) -> None:
        self._validate_descriptor(descriptor)
        temporary = self._worker_path(temporary_path)
        if (
            isinstance(max_bytes_per_second, bool)
            or not isinstance(max_bytes_per_second, int)
            or not 0 < max_bytes_per_second <= 250_000_000
        ):
            raise ArtifactCacheSafetyError("Qwen cache transfer limit changed")
        if descriptor.kind is ArtifactKind.OCI_ARCHIVE:
            self._stage_oci(
                host,
                descriptor,
                temporary,
                expected_filesystem_id,
                max_bytes_per_second=max_bytes_per_second,
                progress=progress,
            )
        else:
            self._stage_tree(
                host,
                descriptor,
                temporary,
                expected_filesystem_id,
                max_bytes_per_second=max_bytes_per_second,
                progress=progress,
            )

    def _prepare_remote_temporary(
        self,
        host: str,
        temporary: Path,
        filesystem_id: str,
        descriptor: ArtifactDescriptor,
        *,
        directory: bool,
    ) -> bool:
        script = r'''
import ctypes, json, os, pathlib, re, stat, sys
expected, fixed_root, raw, fsid, directory, digest, attr_name, attr_value = sys.argv[1:9]
assert os.uname().nodename == expected
root, path = pathlib.PurePosixPath(fixed_root), pathlib.PurePosixPath(raw)
assert path.parent == root / ".staging" and ".." not in path.parts
match = re.fullmatch(r"([a-f0-9]{64})[.][a-f0-9]{1,128}[.]partial", path.name)
assert match is not None and match.group(1) == digest
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
root_fd = os.open("/", flags)
for root_part in root.parts[1:]:
    child_fd = os.open(root_part, flags, dir_fd=root_fd)
    os.close(root_fd); root_fd = child_fd
try:
    def mount_id(fd):
        with open(f"/proc/self/fdinfo/{fd}","r",encoding="ascii") as info:
            values=[line.split(":",1)[1].strip() for line in info if line.startswith("mnt_id:")]
        assert len(values)==1 and values[0].isdecimal(); return values[0]
    root_mount = mount_id(root_fd)
    root_meta = os.fstat(root_fd)
    assert root_meta.st_uid == os.geteuid() and str(root_meta.st_dev) == fsid
    assert not root_meta.st_mode & 0o077
    try:
        staging_fd = os.open(".staging", flags, dir_fd=root_fd)
    except FileNotFoundError:
        os.mkdir(".staging", mode=0o700, dir_fd=root_fd)
        staging_fd = os.open(".staging", flags, dir_fd=root_fd)
    try:
        staging_meta = os.fstat(staging_fd)
        assert staging_meta.st_uid == os.geteuid() and str(staging_meta.st_dev) == fsid
        assert not staging_meta.st_mode & 0o077 and mount_id(staging_fd) == root_mount
        created = False
        try:
            if directory == "1":
                target_fd = os.open(path.name, flags, dir_fd=staging_fd)
            else:
                target_fd = os.open(
                    path.name, os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW, dir_fd=staging_fd
                )
        except FileNotFoundError:
            created = True
            if directory == "1":
                os.mkdir(path.name, mode=0o700, dir_fd=staging_fd)
                target_fd = os.open(path.name, flags, dir_fd=staging_fd)
            else:
                # Publish the empty OCI staging inode only after its ownership
                # xattr exists. A crash before linkat leaves no visible orphan.
                target_fd = os.open(
                    ".",
                    os.O_RDWR|os.O_CLOEXEC|os.O_TMPFILE,
                    0o600,
                    dir_fd=staging_fd,
                )
                os.setxattr(target_fd, attr_name, attr_value.encode("utf-8"),
                            flags=os.XATTR_CREATE)
                libc = ctypes.CDLL(None, use_errno=True)
                linkat = libc.linkat
                linkat.argtypes = [ctypes.c_int,ctypes.c_char_p,ctypes.c_int,
                                   ctypes.c_char_p,ctypes.c_int]
                linkat.restype = ctypes.c_int
                proc_path=f"/proc/self/fd/{target_fd}"
                assert stat.S_ISLNK(os.lstat(proc_path).st_mode)
                assert linkat(-100,os.fsencode(proc_path),staging_fd,
                              os.fsencode(path.name),0x400) == 0
        try:
            target_meta = os.fstat(target_fd)
            assert target_meta.st_uid == os.geteuid() and str(target_meta.st_dev) == fsid
            assert not target_meta.st_mode & 0o077 and mount_id(target_fd) == root_mount
            if directory == "1":
                assert stat.S_ISDIR(target_meta.st_mode)
            else:
                assert stat.S_ISREG(target_meta.st_mode) and target_meta.st_nlink == 1
            if created and directory == "1":
                os.setxattr(target_fd, attr_name, attr_value.encode("utf-8"),
                            flags=os.XATTR_CREATE)
            assert os.getxattr(target_fd, attr_name).decode("utf-8") == attr_value
        finally:
            os.close(target_fd)
    finally:
        os.close(staging_fd)
finally:
    os.close(root_fd)
print(json.dumps({"ok":True,"created":created}, sort_keys=True))
'''
        value = self._remote_python(
            host,
            script,
            str(temporary),
            filesystem_id,
            "1" if directory else "0",
            descriptor.digest_sha256,
            OWNERSHIP_XATTR,
            self._ownership_marker(descriptor.kind, descriptor.digest_sha256),
        )
        if value.get("ok") is not True or type(value.get("created")) is not bool:
            raise ArtifactCacheSafetyError("Qwen cache staging path was not prepared")
        return bool(value["created"])

    def _remote_staging_usage(
        self,
        host: str,
        path: Path,
        filesystem_id: str,
        descriptor: ArtifactDescriptor,
    ) -> int:
        script = r'''
import json, os, pathlib, stat, sys
expected, fixed_root, raw, fsid, kind, attr_name, attr_value, maximum_raw = sys.argv[1:9]
assert os.uname().nodename == expected
root, path = pathlib.Path(fixed_root), pathlib.Path(raw)
root_meta = root.lstat()
assert stat.S_ISDIR(root_meta.st_mode) and not stat.S_ISLNK(root_meta.st_mode)
assert root_meta.st_uid == os.geteuid() and str(root_meta.st_dev) == fsid
relative = path.relative_to(root)
assert path.parent == root / ".staging" and relative.parts and ".." not in relative.parts
meta = path.lstat()
assert meta.st_uid == os.geteuid() and str(meta.st_dev) == fsid
assert not stat.S_ISLNK(meta.st_mode) and not os.path.ismount(path)
assert os.getxattr(path, attr_name, follow_symlinks=False).decode("utf-8") == attr_value
maximum = int(maximum_raw)
if kind == "oci_archive":
    assert stat.S_ISREG(meta.st_mode); total = meta.st_size
else:
    assert stat.S_ISDIR(meta.st_mode); total = 0
    for item in path.rglob("*"):
        try:
            item_meta = item.lstat()
        except FileNotFoundError:
            continue
        assert item_meta.st_uid == os.geteuid() and str(item_meta.st_dev) == fsid
        assert not stat.S_ISLNK(item_meta.st_mode) and not os.path.ismount(item)
        if stat.S_ISREG(item_meta.st_mode):
            assert item_meta.st_nlink == 1
            total += item_meta.st_size
        else:
            assert stat.S_ISDIR(item_meta.st_mode)
        assert total <= maximum
assert 0 <= total <= maximum
print(json.dumps({"bytes":total}, sort_keys=True))
'''
        value = self._remote_python(
            host,
            script,
            str(path),
            filesystem_id,
            descriptor.kind.value,
            OWNERSHIP_XATTR,
            self._ownership_marker(descriptor.kind, descriptor.digest_sha256),
            str(max(descriptor.transfer_bytes_max, descriptor.size_bytes_max)),
        )
        observed = value.get("bytes")
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise ArtifactCacheSafetyError("worker staging progress is malformed")
        return observed

    def _stage_tree(
        self,
        host: str,
        descriptor: ArtifactDescriptor,
        temporary: Path,
        filesystem_id: str,
        *,
        max_bytes_per_second: int,
        progress: Callable[[int, int], None],
    ) -> None:
        self._verify_canonical_tree(
            descriptor,
            lambda: progress(0, descriptor.transfer_bytes_max),
        )
        files = self._manifest_files(descriptor)
        self._prepare_remote_temporary(
            host,
            temporary,
            filesystem_id,
            descriptor,
            directory=True,
        )
        transport = " ".join(self._ssh(host)[:-1])
        bwlimit_kib = max(1, max_bytes_per_second // 1024)
        command = [
            str(HOST_BASH),
            str(FLEET_LOW_PRIORITY),
            "/usr/bin/rsync",
            "-aR",
            "--checksum",
            "--protect-args",
            f"--bwlimit={bwlimit_kib}",
            "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
            "-e",
            transport,
            "--",
            *files,
            f"aday@{network_address(host)}:{temporary}/",
        ]
        self._run_with_progress(
            command,
            cwd=descriptor.canonical_path,
            progress=progress,
            total=descriptor.transfer_bytes_max,
            progress_probe=lambda: self._remote_staging_usage(
                host, temporary, filesystem_id, descriptor
            ),
        )
        progress(descriptor.transfer_bytes_max, descriptor.transfer_bytes_max)

    def _fd_sha256(
        self,
        descriptor: int,
        *,
        progress: Callable[[int, int], None],
        total: int,
        expected_link_count: int,
    ) -> str:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_mode & 0o022
            or before.st_nlink != expected_link_count
            or not 0 < before.st_size <= total
        ):
            raise ArtifactCacheSafetyError("canonical OCI checksum input is unsafe")
        output, _error = self._run_with_progress(
            [
                str(HOST_BASH),
                str(FLEET_LOW_PRIORITY),
                str(HOST_SHA256SUM),
                f"/proc/self/fd/{descriptor}",
            ],
            progress=progress,
            total=total,
            timeout=3600,
            pass_fds=(descriptor,),
        )
        digest = output.split(maxsplit=1)[0] if output else ""
        if _SHA256_RE.fullmatch(digest) is None:
            raise ArtifactCacheSafetyError("canonical OCI checksum failed")
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise ArtifactCacheSafetyError("canonical OCI archive changed while hashed")
        return digest

    def _canonical_archive(
        self,
        descriptor: ArtifactDescriptor,
        *,
        progress: Callable[[int, int], None],
    ) -> tuple[Path, str, int]:
        archive = Path(descriptor.canonical_path)
        expected_name = f"{descriptor.digest_sha256}.tar"
        if archive != CANONICAL_OCI_ROOT / expected_name:
            raise ArtifactCacheSafetyError("canonical OCI path changed")
        image_id = f"sha256:{descriptor.digest_sha256}"
        observed_id = local_image_id(image_id)
        image_size = local_image_size(observed_id)
        if observed_id != image_id:
            raise ArtifactCacheSafetyError("canonical Qwen image identity changed")
        root_fd = _open_canonical_root(create=True)
        try:
            try:
                existing = os.open(
                    expected_name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=root_fd,
                )
            except FileNotFoundError:
                existing = -1
            if existing >= 0:
                try:
                    _validate_oci_archive_fd(
                        existing,
                        digest=descriptor.digest_sha256,
                        maximum_bytes=descriptor.transfer_bytes_max,
                        allowed_link_counts=frozenset({1}),
                    )
                    raw_sha = self._fd_sha256(
                        existing,
                        progress=progress,
                        total=descriptor.transfer_bytes_max,
                        expected_link_count=1,
                    )
                    current = os.stat(
                        expected_name, dir_fd=root_fd, follow_symlinks=False
                    )
                    anchored = os.fstat(existing)
                    if (current.st_dev, current.st_ino) != (
                        anchored.st_dev,
                        anchored.st_ino,
                    ):
                        raise ArtifactCacheSafetyError(
                            "canonical OCI archive path changed"
                        )
                    return archive, raw_sha, image_size
                finally:
                    os.close(existing)

            capacity = os.fstatvfs(root_fd)
            free_bytes = capacity.f_bavail * capacity.f_frsize
            if (
                free_bytes
                < descriptor.transfer_bytes_max + CANONICAL_FREE_RESERVE_BYTES
            ):
                raise ArtifactCacheError(
                    "canonical storage cannot safely materialize OCI"
                )
            if capacity.f_favail < CANONICAL_FREE_RESERVE_INODES + 1:
                raise ArtifactCacheError("canonical storage lacks OCI staging inodes")
            try:
                temporary = os.open(
                    ".",
                    os.O_RDWR | os.O_TMPFILE | os.O_CLOEXEC,
                    0o600,
                    dir_fd=root_fd,
                )
            except OSError as exc:
                raise ArtifactCacheSafetyError(
                    "canonical filesystem lacks anonymous staging"
                ) from exc
            try:
                with os.fdopen(os.dup(temporary), "wb", closefd=True) as handle:
                    self._run_with_progress(
                        [
                            str(HOST_BASH),
                            str(FLEET_LOW_PRIORITY),
                            HOST_PRLIMIT,
                            "--fsize="
                            f"{descriptor.transfer_bytes_max}:"
                            f"{descriptor.transfer_bytes_max}",
                            "--",
                            *_docker_command("image", "save", image_id),
                        ],
                        env=_docker_cli_environment(),
                        stdout=handle,
                        progress=progress,
                        total=descriptor.transfer_bytes_max,
                        progress_probe=lambda: os.fstat(temporary).st_size,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                _validate_oci_archive_fd(
                    temporary,
                    digest=descriptor.digest_sha256,
                    maximum_bytes=descriptor.transfer_bytes_max,
                    allowed_link_counts=frozenset({0}),
                )
                raw_sha = self._fd_sha256(
                    temporary,
                    progress=progress,
                    total=descriptor.transfer_bytes_max,
                    expected_link_count=0,
                )
                os.fsync(temporary)
                _link_anonymous_noreplace(temporary, root_fd, expected_name)
                os.fsync(root_fd)
                published = os.stat(
                    expected_name, dir_fd=root_fd, follow_symlinks=False
                )
                anchored = os.fstat(temporary)
                if (
                    (published.st_dev, published.st_ino)
                    != (anchored.st_dev, anchored.st_ino)
                    or anchored.st_nlink != 1
                ):
                    raise ArtifactCacheSafetyError(
                        "canonical OCI archive publication changed"
                    )
                return archive, raw_sha, image_size
            finally:
                os.close(temporary)
        finally:
            os.close(root_fd)

    def _remote_sha256(
        self,
        host: str,
        path: Path,
        *,
        progress: Callable[[int, int], None],
        total: int,
    ) -> str:
        command = [
            *self._ssh(host),
            shlex.join(
                [
                    "/usr/bin/env",
                    "-i",
                    "PATH=/usr/bin:/bin",
                    "HOME=/home/aday",
                    "LANG=C",
                    "LC_ALL=C",
                    "/usr/bin/bash",
                    REMOTE_WRAPPER,
                    str(HOST_SHA256SUM),
                    str(path),
                ]
            ),
        ]
        output, _error = self._run_with_progress(
            command,
            progress=progress,
            total=total,
            timeout=3600,
        )
        digest = output.split(maxsplit=1)[0]
        if _SHA256_RE.fullmatch(digest) is None:
            raise ArtifactCacheSafetyError("worker artifact checksum failed")
        return digest

    def _commit_remote_oci_receipt(
        self,
        host: str,
        temporary: Path,
        filesystem_id: str,
        descriptor: ArtifactDescriptor,
        receipt: Mapping[str, Any],
    ) -> None:
        script = r'''
import json, os, pathlib, re, stat, sys
expected, fixed_root, raw, fsid, digest, attr_name, attr_value, payload = sys.argv[1:9]
assert os.uname().nodename == expected
root, path = pathlib.PurePosixPath(fixed_root), pathlib.PurePosixPath(raw)
assert path.parent == root / ".staging" and ".." not in path.parts
match = re.fullmatch(r"([a-f0-9]{64})[.][a-f0-9]{1,128}[.]partial", path.name)
assert match is not None and match.group(1) == digest
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
root_fd = os.open("/", flags)
for root_part in root.parts[1:]:
    child_fd = os.open(root_part, flags, dir_fd=root_fd)
    os.close(root_fd); root_fd = child_fd
try:
    root_meta = os.fstat(root_fd)
    assert root_meta.st_uid == os.geteuid() and str(root_meta.st_dev) == fsid
    staging_fd = os.open(".staging", flags, dir_fd=root_fd)
    try:
        staging_meta = os.fstat(staging_fd)
        assert staging_meta.st_uid == os.geteuid() and str(staging_meta.st_dev) == fsid
        fd = os.open(path.name, os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW,
                     dir_fd=staging_fd)
        try:
            before = os.fstat(fd)
            assert stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid()
            assert str(before.st_dev) == fsid and before.st_nlink == 1
            assert not before.st_mode & 0o077
            assert os.getxattr(fd, attr_name).decode("utf-8") == attr_value
            encoded = payload.encode("utf-8")
            assert 0 < len(encoded) <= 64 * 1024
            os.lseek(fd, 0, os.SEEK_SET); os.ftruncate(fd, 0)
            written = 0
            while written < len(encoded):
                written += os.write(fd, encoded[written:])
            os.fsync(fd)
            after = os.fstat(fd)
            assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
            assert os.getxattr(fd, attr_name).decode("utf-8") == attr_value
        finally:
            os.close(fd)
    finally:
        os.close(staging_fd)
finally:
    os.close(root_fd)
print(json.dumps({"ok":True}))
'''
        value = self._remote_python(
            host,
            script,
            str(temporary),
            filesystem_id,
            descriptor.digest_sha256,
            OWNERSHIP_XATTR,
            self._ownership_marker(descriptor.kind, descriptor.digest_sha256),
            _canonical_json(receipt).decode("utf-8"),
        )
        if value != {"ok": True}:
            raise ArtifactCacheSafetyError("worker OCI receipt was not committed")

    def _stage_oci(
        self,
        host: str,
        descriptor: ArtifactDescriptor,
        temporary: Path,
        filesystem_id: str,
        *,
        max_bytes_per_second: int,
        progress: Callable[[int, int], None],
    ) -> None:
        total = descriptor.transfer_bytes_max
        image_id = f"sha256:{descriptor.digest_sha256}"
        observed_id = local_image_id(image_id)
        image_size = local_image_size(observed_id)
        if observed_id != image_id:
            raise ArtifactCacheSafetyError("canonical Qwen image identity changed")
        self._prepare_remote_temporary(
            host,
            temporary,
            filesystem_id,
            descriptor,
            directory=False,
        )
        warm_image = self._remote_image_inspection(host, image_id)
        if warm_image is not None:
            if int(warm_image["Size"]) != image_size:
                raise ArtifactCacheSafetyError("warm Qwen image identity changed")
            # No archive was transferred. The exact image config digest is the
            # deterministic no-transport payload identity for this tiny receipt.
            receipt = {
                "schema_version": OCI_RECEIPT_SCHEMA,
                "image_id": image_id,
                "image_size_bytes": image_size,
                "archive_payload_sha256": descriptor.digest_sha256,
            }
            self._commit_remote_oci_receipt(
                host, temporary, filesystem_id, descriptor, receipt
            )
            progress(total, total)
            return

        def canonical_keepalive(_completed: int, _total: int) -> None:
            progress(0, total)

        archive, payload_sha256, canonical_image_size = self._canonical_archive(
            descriptor, progress=canonical_keepalive
        )
        if canonical_image_size != image_size:
            raise ArtifactCacheSafetyError("canonical Qwen image size changed")
        receipt = {
            "schema_version": OCI_RECEIPT_SCHEMA,
            "image_id": image_id,
            "image_size_bytes": image_size,
            "archive_payload_sha256": payload_sha256,
        }
        transferred = 0

        def transfer_progress(completed: int, _reported_total: int) -> None:
            nonlocal transferred
            transferred = max(transferred, min(total, completed))
            progress(transferred, total)

        def post_transfer_keepalive(_completed: int, _reported_total: int) -> None:
            progress(transferred, total)

        transport = " ".join(self._ssh(host)[:-1])
        bwlimit_kib = max(1, max_bytes_per_second // 1024)
        self._run_with_progress(
            [
                str(HOST_BASH),
                str(FLEET_LOW_PRIORITY),
                "/usr/bin/rsync",
                "-a",
                "--inplace",
                "--checksum",
                "--protect-args",
                f"--bwlimit={bwlimit_kib}",
                "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
                "-e",
                transport,
                "--",
                str(archive),
                f"aday@{network_address(host)}:{temporary}",
            ],
            progress=transfer_progress,
            total=total,
            progress_probe=lambda: self._remote_staging_usage(
                host, temporary, filesystem_id, descriptor
            ),
        )
        remote_sha = self._remote_sha256(
            host,
            temporary,
            progress=post_transfer_keepalive,
            total=total,
        )
        if remote_sha != payload_sha256:
            raise ArtifactCacheSafetyError("worker OCI archive transfer changed")
        load_command = [
            *self._ssh(host),
            shlex.join(
                [
                    "/usr/bin/env",
                    "-i",
                    "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
                    "HOME=/home/aday",
                    "LANG=C",
                    "LC_ALL=C",
                    "/usr/bin/bash",
                    REMOTE_WRAPPER,
                    "/usr/bin/bash",
                    REMOTE_DOCKER,
                    "image",
                    "load",
                    "--input",
                    str(temporary),
                ]
            ),
        ]
        self._run_with_progress(
            load_command,
            progress=post_transfer_keepalive,
            total=total,
            timeout=3600,
        )
        image = self._remote_image_inspection(host, image_id)
        if image is None or int(image["Size"]) != image_size:
            raise ArtifactCacheSafetyError("loaded Qwen image identity changed")
        self._commit_remote_oci_receipt(
            host, temporary, filesystem_id, descriptor, receipt
        )
        progress(total, total)

    def promote(
        self,
        *,
        host: str,
        temporary_path: str,
        final_path: str,
        descriptor: ArtifactDescriptor,
        identity_token: str,
        expected_filesystem_id: str,
        owner_uid: int,
    ) -> None:
        self._validate_descriptor(descriptor)
        temporary = self._worker_path(temporary_path)
        final = self._worker_path(final_path)
        if (
            owner_uid != os.geteuid()
            or _SHA256_RE.fullmatch(identity_token) is None
        ):
            raise ArtifactCacheSafetyError("Qwen cache promotion owner changed")
        manifest_relative = ""
        if descriptor.kind is ArtifactKind.MANIFESTED_TREE:
            manifest_relative = Path(str(descriptor.manifest_path)).relative_to(
                Path(descriptor.canonical_path)
            ).as_posix()
        script = r'''
import ctypes, errno, hashlib, json, os, pathlib, re, stat, sys
expected, fixed_root, temp_raw, final_raw, fsid, uid_raw, kind, digest, attr_name, attr_value, wanted_token, manifest_rel, max_bytes_raw, max_inodes_raw = sys.argv[1:15]
assert os.uname().nodename == expected
root = pathlib.PurePosixPath(fixed_root)
temp, final, uid = pathlib.PurePosixPath(temp_raw), pathlib.PurePosixPath(final_raw), int(uid_raw)
match = re.fullmatch(r"([a-f0-9]{64})[.][a-f0-9]{1,128}[.]partial", temp.name)
assert temp.parent == root / ".staging" and match is not None and match.group(1) == digest
assert final == root / "sha256" / digest[:2] / digest
max_bytes, max_inodes = int(max_bytes_raw), int(max_inodes_raw)
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
root_fd = os.open("/", flags)
for root_part in root.parts[1:]:
    child_fd = os.open(root_part, flags, dir_fd=root_fd)
    os.close(root_fd); root_fd = child_fd
try:
    def mount_id(fd):
        with open(f"/proc/self/fdinfo/{fd}","r",encoding="ascii") as info:
            values=[line.split(":",1)[1].strip() for line in info if line.startswith("mnt_id:")]
        assert len(values)==1 and values[0].isdecimal(); return values[0]
    root_mount = mount_id(root_fd)
    root_meta = os.fstat(root_fd)
    assert root_meta.st_uid == uid and str(root_meta.st_dev) == fsid
    staging_fd = os.open(".staging", flags, dir_fd=root_fd)
    try:
        try:
            temp_fd = os.open(temp.name, os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW,
                              dir_fd=staging_fd)
        except IsADirectoryError:
            temp_fd = os.open(temp.name, flags, dir_fd=staging_fd)
        try:
            before = os.fstat(temp_fd); before_root = before
            assert before.st_uid == uid and str(before.st_dev) == fsid
            assert not before.st_mode & 0o077 and mount_id(temp_fd) == root_mount
            marker_raw = os.getxattr(temp_fd, attr_name).decode("utf-8")
            assert marker_raw == attr_value
            marker = json.loads(attr_value)
            assert marker == {"schema_version":1,"kind":kind,
              "digest_sha256":digest,"cache_root":str(root)}
            def snapshot():
                records, files, directories, measured = [], set(), set(), [0, 1]
                def scan(directory_fd, prefix=""):
                    for entry in os.scandir(directory_fd):
                        relative = f"{prefix}/{entry.name}" if prefix else entry.name
                        item = entry.stat(follow_symlinks=False)
                        assert item.st_uid == uid and str(item.st_dev) == fsid
                        assert not stat.S_ISLNK(item.st_mode) and not item.st_mode & 0o022
                        child_flags = (flags if stat.S_ISDIR(item.st_mode) else
                                       os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
                        child = os.open(entry.name, child_flags, dir_fd=directory_fd)
                        try:
                            anchored = os.fstat(child)
                            assert (anchored.st_dev,anchored.st_ino,anchored.st_mode,
                                    anchored.st_uid,anchored.st_nlink,anchored.st_size,
                                    anchored.st_mtime_ns,anchored.st_ctime_ns) == (
                                    item.st_dev,item.st_ino,item.st_mode,item.st_uid,
                                    item.st_nlink,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
                            assert mount_id(child) == root_mount
                            if stat.S_ISDIR(item.st_mode):
                                directories.add(relative); scan(child, relative)
                            elif stat.S_ISREG(item.st_mode):
                                assert item.st_nlink == 1
                                measured[0] += item.st_size; files.add(relative)
                            else:
                                raise AssertionError
                        finally:
                            os.close(child)
                        records.append([relative,item.st_dev,item.st_ino,item.st_mode,
                                        item.st_uid,item.st_nlink,item.st_size,
                                        item.st_mtime_ns,item.st_ctime_ns])
                        measured[1] += 1
                        assert measured[0] <= max_bytes and measured[1] <= max_inodes
                scan_fd=os.open(".",flags,dir_fd=temp_fd)
                try:
                    assert mount_id(scan_fd) == root_mount
                    scan(scan_fd)
                finally:
                    os.close(scan_fd)
                return sorted(records),files,directories,tuple(measured)
            records, files, directories, measured = [], set(), set(), (0, 1)
            if kind == "manifested_tree":
                assert stat.S_ISDIR(before.st_mode)
                records,files,directories,measured=snapshot()
                records_by_name={item[0]:item[1:] for item in records}
                def open_relative(relative):
                    fd=os.dup(temp_fd)
                    try:
                        parts=pathlib.PurePosixPath(relative).parts
                        for index, part in enumerate(parts):
                            wanted=(os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW
                                    if index+1 == len(parts) else flags)
                            child=os.open(part,wanted,dir_fd=fd)
                            os.close(fd); fd=child
                        return fd
                    except Exception:
                        os.close(fd); raise
                assert manifest_rel in files
                manifest_fd = open_relative(manifest_rel); payload=b""
                try:
                    while True:
                        chunk=os.read(manifest_fd,1024*1024)
                        if not chunk: break
                        payload += chunk; assert len(payload) <= 16*1024*1024
                    anchored=os.fstat(manifest_fd)
                    assert [anchored.st_dev,anchored.st_ino,anchored.st_mode,
                            anchored.st_uid,anchored.st_nlink,anchored.st_size,
                            anchored.st_mtime_ns,anchored.st_ctime_ns] == records_by_name[manifest_rel]
                finally:
                    os.close(manifest_fd)
                assert hashlib.sha256(payload).hexdigest() == digest
                expected_files={manifest_rel}; checks=[]
                for line in payload.decode("utf-8").splitlines():
                    item=re.fullmatch(r"([a-f0-9]{64}) [ *](.+)",line)
                    assert item is not None and not item.group(2).startswith("/")
                    assert ".." not in pathlib.PurePosixPath(item.group(2)).parts
                    assert item.group(2) not in expected_files
                    expected_files.add(item.group(2)); checks.append((item.group(1),item.group(2)))
                assert files == expected_files
                expected_dirs=set()
                for relative in expected_files:
                    parts=pathlib.PurePosixPath(relative).parts[:-1]
                    expected_dirs.update(pathlib.PurePosixPath(*parts[:depth]).as_posix()
                                         for depth in range(1,len(parts)+1))
                assert directories == expected_dirs
                for wanted, relative in checks:
                    fd=open_relative(relative); hasher=hashlib.sha256()
                    try:
                        while True:
                            chunk=os.read(fd,1024*1024)
                            if not chunk: break
                            hasher.update(chunk)
                        anchored=os.fstat(fd)
                        assert [anchored.st_dev,anchored.st_ino,anchored.st_mode,
                                anchored.st_uid,anchored.st_nlink,anchored.st_size,
                                anchored.st_mtime_ns,anchored.st_ctime_ns] == records_by_name[relative]
                    finally:
                        os.close(fd)
                    assert hasher.hexdigest() == wanted
            else:
                assert stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                assert before.st_size <= max_bytes
                os.lseek(temp_fd,0,os.SEEK_SET); payload=b""
                while True:
                    chunk=os.read(temp_fd,64*1024)
                    if not chunk: break
                    payload += chunk; assert len(payload) <= max_bytes
                receipt=json.loads(payload.decode("utf-8"))
                assert set(receipt) == {"schema_version","image_id",
                                        "image_size_bytes","archive_payload_sha256"}
                assert receipt["schema_version"] == 1
                assert receipt["image_id"] == "sha256:" + digest
                assert type(receipt["image_size_bytes"]) is int
                assert 0 < receipt["image_size_bytes"] <= 64*1024**3
                assert re.fullmatch(r"[a-f0-9]{64}",receipt["archive_payload_sha256"])
            # TEST_RACE_BARRIER_PROMOTE_POST_HASH
            if kind == "manifested_tree":
                verified_snapshot=snapshot()
                assert verified_snapshot == (records,files,directories,measured)
                records,files,directories,measured=verified_snapshot
            else:
                os.lseek(temp_fd,0,os.SEEK_SET); verified_payload=b""
                while True:
                    chunk=os.read(temp_fd,64*1024)
                    if not chunk: break
                    verified_payload += chunk; assert len(verified_payload) <= max_bytes
                assert verified_payload == payload
            material={"schema":1,"path":str(temp),"marker":attr_value,
              "root":[before.st_dev,before.st_ino,before.st_mode,before.st_uid,
                      before.st_nlink,before.st_size,before.st_mtime_ns,before.st_ctime_ns],
              "members":sorted(records)}
            token=hashlib.sha256(json.dumps(material,sort_keys=True,separators=(",",":"),
                                 allow_nan=False).encode("utf-8")).hexdigest()
            assert token == wanted_token
            after = os.fstat(temp_fd)
            assert (after.st_dev,after.st_ino,after.st_mode,after.st_uid,after.st_nlink,
                    after.st_size,after.st_mtime_ns,after.st_ctime_ns) == (
                    before_root.st_dev,before_root.st_ino,before_root.st_mode,
                    before_root.st_uid,before_root.st_nlink,before_root.st_size,
                    before_root.st_mtime_ns,before_root.st_ctime_ns)
            try:
                sha_fd = os.open("sha256", flags, dir_fd=root_fd)
            except FileNotFoundError:
                os.mkdir("sha256", mode=0o700, dir_fd=root_fd)
                sha_fd = os.open("sha256", flags, dir_fd=root_fd)
            try:
                sha_meta = os.fstat(sha_fd)
                assert sha_meta.st_uid == uid and str(sha_meta.st_dev) == fsid
                assert not sha_meta.st_mode & 0o077
                try:
                    prefix_fd = os.open(digest[:2], flags, dir_fd=sha_fd)
                except FileNotFoundError:
                    os.mkdir(digest[:2], mode=0o700, dir_fd=sha_fd)
                    prefix_fd = os.open(digest[:2], flags, dir_fd=sha_fd)
                try:
                    prefix_meta = os.fstat(prefix_fd)
                    assert prefix_meta.st_uid == uid and str(prefix_meta.st_dev) == fsid
                    assert not prefix_meta.st_mode & 0o077
                    libc = ctypes.CDLL(None, use_errno=True)
                    renameat2 = getattr(libc, "renameat2", None)
                    assert renameat2 is not None
                    renameat2.argtypes = [ctypes.c_int,ctypes.c_char_p,ctypes.c_int,
                                          ctypes.c_char_p,ctypes.c_uint]
                    renameat2.restype = ctypes.c_int
                    if kind == "manifested_tree":
                        assert snapshot() == (records,files,directories,measured)
                    else:
                        os.lseek(temp_fd,0,os.SEEK_SET); immediate_payload=b""
                        while True:
                            chunk=os.read(temp_fd,64*1024)
                            if not chunk: break
                            immediate_payload += chunk
                            assert len(immediate_payload) <= max_bytes
                        assert immediate_payload == payload
                    immediate=os.fstat(temp_fd)
                    assert (immediate.st_dev,immediate.st_ino,immediate.st_mode,
                            immediate.st_uid,immediate.st_nlink,immediate.st_size,
                            immediate.st_mtime_ns,immediate.st_ctime_ns) == (
                            before_root.st_dev,before_root.st_ino,before_root.st_mode,
                            before_root.st_uid,before_root.st_nlink,before_root.st_size,
                            before_root.st_mtime_ns,before_root.st_ctime_ns)
                    result = renameat2(staging_fd, os.fsencode(temp.name), prefix_fd,
                                       os.fsencode(digest), 1)
                    if result != 0:
                        error = ctypes.get_errno()
                        assert error not in {errno.EEXIST, errno.ENOTEMPTY}
                        raise OSError(error, os.strerror(error))
                    published = os.stat(digest, dir_fd=prefix_fd, follow_symlinks=False)
                    assert (published.st_dev,published.st_ino) == (before.st_dev,before.st_ino)
                finally:
                    os.close(prefix_fd)
            finally:
                os.close(sha_fd)
        finally:
            os.close(temp_fd)
    finally:
        os.close(staging_fd)
finally:
    os.close(root_fd)
print(json.dumps({"ok":True}))
'''
        value = self._remote_python(
            host,
            script,
            str(temporary),
            str(final),
            expected_filesystem_id,
            str(owner_uid),
            descriptor.kind.value,
            descriptor.digest_sha256,
            OWNERSHIP_XATTR,
            self._ownership_marker(descriptor.kind, descriptor.digest_sha256),
            identity_token,
            manifest_relative,
            str(descriptor.size_bytes_max),
            str(descriptor.inode_count_max),
        )
        if value != {"ok": True}:
            raise ArtifactCacheSafetyError("Qwen cache promotion failed")

    def remove(
        self,
        *,
        host: str,
        path: str,
        descriptor: ArtifactDescriptor,
        identity_token: str,
        expected_filesystem_id: str,
        owner_uid: int,
    ) -> CacheRemovalReceipt:
        self._validate_descriptor(descriptor)
        checked = self._worker_path(path)
        if (
            owner_uid != os.geteuid()
            or _SHA256_RE.fullmatch(identity_token) is None
        ):
            raise ArtifactCacheSafetyError("Qwen cache removal owner changed")
        manifest_relative = ""
        if descriptor.kind is ArtifactKind.MANIFESTED_TREE:
            manifest_relative = Path(str(descriptor.manifest_path)).relative_to(
                Path(descriptor.canonical_path)
            ).as_posix()
        script = r'''
import hashlib, json, os, pathlib, re, stat, sys
expected, fixed_root, raw, fsid, uid_raw, kind, digest, attr_name, attr_value, wanted_token, manifest_rel, max_final_bytes_raw, max_transfer_bytes_raw, max_inodes_raw = sys.argv[1:15]
assert os.uname().nodename == expected
root, path, uid = pathlib.PurePosixPath(fixed_root), pathlib.PurePosixPath(raw), int(uid_raw)
partial = path.parent == root / ".staging"
max_bytes = int(max_transfer_bytes_raw if partial else max_final_bytes_raw)
max_inodes = int(max_inodes_raw)
if partial:
    match = re.fullmatch(r"([a-f0-9]{64})[.][a-f0-9]{1,128}[.]partial", path.name)
    assert match is not None and match.group(1) == digest
else:
    assert path == root / "sha256" / digest[:2] / digest
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
root_fd = os.open("/", flags)
for root_part in root.parts[1:]:
    child_fd = os.open(root_part, flags, dir_fd=root_fd)
    os.close(root_fd); root_fd = child_fd
try:
    def mount_id(fd):
        with open(f"/proc/self/fdinfo/{fd}","r",encoding="ascii") as info:
            values=[line.split(":",1)[1].strip() for line in info if line.startswith("mnt_id:")]
        assert len(values)==1 and values[0].isdecimal(); return values[0]
    root_mount = mount_id(root_fd)
    root_meta = os.fstat(root_fd)
    assert root_meta.st_uid == uid and str(root_meta.st_dev) == fsid
    if partial:
        parent_fd = os.open(".staging", flags, dir_fd=root_fd)
    else:
        sha_fd = os.open("sha256", flags, dir_fd=root_fd)
        try:
            parent_fd = os.open(digest[:2], flags, dir_fd=sha_fd)
        finally:
            os.close(sha_fd)
    try:
        try:
            if kind == "manifested_tree":
                target_fd = os.open(path.name, flags, dir_fd=parent_fd)
            else:
                target_fd = os.open(path.name, os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW,
                                    dir_fd=parent_fd)
        except FileNotFoundError:
            print(json.dumps({"removed":False,"filesystem_id":fsid,"owner_uid":uid,
             "reclaimed_bytes":0,"reclaimed_inodes":0}, sort_keys=True)); raise SystemExit(0)
        try:
            meta = os.fstat(target_fd)
            assert meta.st_uid == uid and str(meta.st_dev) == fsid
            assert not meta.st_mode & 0o077 and mount_id(target_fd) == root_mount
            assert os.getxattr(target_fd, attr_name).decode("utf-8") == attr_value
            if kind == "manifested_tree":
                assert stat.S_ISDIR(meta.st_mode)
            else:
                assert stat.S_ISREG(meta.st_mode) and meta.st_nlink == 1
            def material(records):
                value = {"schema":1,"path":str(path),"marker":attr_value,
                  "root":[meta.st_dev,meta.st_ino,meta.st_mode,meta.st_uid,meta.st_nlink,
                          meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns],
                  "members":records}
                return hashlib.sha256(json.dumps(value,sort_keys=True,
                    separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
            def snapshot():
                records, files, directories, measured = [], [], [], [0, 1]
                def scan(directory_fd, prefix=""):
                    for entry in os.scandir(directory_fd):
                        item = entry.stat(follow_symlinks=False)
                        relative = f"{prefix}/{entry.name}" if prefix else entry.name
                        assert item.st_uid == uid and str(item.st_dev) == fsid
                        assert not stat.S_ISLNK(item.st_mode) and not item.st_mode & 0o022
                        child_flags = (flags if stat.S_ISDIR(item.st_mode) else
                                       os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
                        child = os.open(entry.name,child_flags,dir_fd=directory_fd)
                        anchored = os.fstat(child)
                        assert (anchored.st_dev,anchored.st_ino,anchored.st_mode,
                                anchored.st_uid,anchored.st_nlink,anchored.st_size,
                                anchored.st_mtime_ns,anchored.st_ctime_ns) == (
                                item.st_dev,item.st_ino,item.st_mode,item.st_uid,
                                item.st_nlink,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
                        assert mount_id(child) == root_mount
                        records.append([relative,item.st_dev,item.st_ino,item.st_mode,
                                        item.st_uid,item.st_nlink,item.st_size,
                                        item.st_mtime_ns,item.st_ctime_ns])
                        measured[1] += 1
                        if stat.S_ISDIR(item.st_mode):
                            try:
                                scan(child, relative)
                            finally:
                                os.close(child)
                            directories.append((relative,item.st_dev,item.st_ino,
                                                item.st_ctime_ns))
                        elif stat.S_ISREG(item.st_mode):
                            assert item.st_nlink == 1
                            measured[0] += item.st_size
                            files.append((relative,item.st_dev,item.st_ino,item.st_size,
                                          item.st_ctime_ns))
                            os.close(child)
                        else:
                            os.close(child); raise AssertionError
                        assert measured[0] <= max_bytes and measured[1] <= max_inodes
                scan_fd=os.open(".",flags,dir_fd=target_fd)
                try:
                    assert mount_id(scan_fd) == root_mount
                    scan(scan_fd)
                finally:
                    os.close(scan_fd)
                return sorted(records),sorted(files),sorted(directories),tuple(measured)
            records, files, directories, measured = [], [], [], (0, 1)
            if kind == "manifested_tree":
                records,files,directories,measured=snapshot()
                full_token = material(records)
                assert wanted_token == full_token
                if not partial:
                    manifest_parts = pathlib.PurePosixPath(manifest_rel).parts
                    manifest_fd = os.dup(target_fd)
                    try:
                        for index, part in enumerate(manifest_parts):
                            next_flags = (os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW
                                          if index + 1 == len(manifest_parts) else flags)
                            child = os.open(part, next_flags, dir_fd=manifest_fd)
                            os.close(manifest_fd); manifest_fd = child
                        payload = b""
                        while True:
                            chunk = os.read(manifest_fd, 1024*1024)
                            if not chunk: break
                            payload += chunk
                            assert len(payload) <= 16*1024*1024
                    finally:
                        os.close(manifest_fd)
                    assert hashlib.sha256(payload).hexdigest() == digest
                    expected_files = {manifest_rel}
                    checks = []
                    for line in payload.decode("utf-8").splitlines():
                        match = re.fullmatch(r"([a-f0-9]{64}) [ *](.+)", line)
                        assert match is not None and not match.group(2).startswith("/")
                        assert ".." not in pathlib.PurePosixPath(match.group(2)).parts
                        assert match.group(2) not in expected_files
                        expected_files.add(match.group(2)); checks.append((match.group(1),match.group(2)))
                    assert {item[0] for item in files} == expected_files
                    for wanted, relative in checks:
                        file_fd = os.dup(target_fd)
                        try:
                            parts = pathlib.PurePosixPath(relative).parts
                            for index, part in enumerate(parts):
                                next_flags = (os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW
                                              if index + 1 == len(parts) else flags)
                                child = os.open(part, next_flags, dir_fd=file_fd)
                                os.close(file_fd); file_fd = child
                            hasher = hashlib.sha256()
                            while True:
                                chunk = os.read(file_fd, 1024*1024)
                                if not chunk: break
                                hasher.update(chunk)
                            assert hasher.hexdigest() == wanted
                        finally:
                            os.close(file_fd)
            else:
                assert wanted_token == material([])
                if not partial:
                    os.lseek(target_fd, 0, os.SEEK_SET)
                    payload = b""
                    while True:
                        chunk = os.read(target_fd, 64*1024)
                        if not chunk: break
                        payload += chunk
                        assert len(payload) <= 64*1024
                    receipt = json.loads(payload.decode("utf-8"))
                    assert set(receipt) == {"schema_version","image_id",
                                            "image_size_bytes","archive_payload_sha256"}
                    assert receipt["schema_version"] == 1
                    assert receipt["image_id"] == "sha256:" + digest
                    assert re.fullmatch(r"[a-f0-9]{64}",receipt["archive_payload_sha256"])
            # TEST_RACE_BARRIER_REMOVE_POST_HASH
            if kind == "manifested_tree":
                verified_snapshot=snapshot()
                assert verified_snapshot == (records,files,directories,measured)
                assert wanted_token == material(verified_snapshot[0])
            elif not partial:
                os.lseek(target_fd,0,os.SEEK_SET); verified_payload=b""
                while True:
                    chunk=os.read(target_fd,64*1024)
                    if not chunk: break
                    verified_payload += chunk; assert len(verified_payload) <= max_bytes
                assert verified_payload == payload
            current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            assert (current.st_dev,current.st_ino,current.st_ctime_ns) == (
                    meta.st_dev,meta.st_ino,meta.st_ctime_ns)
            assert meta.st_size <= max_bytes if kind != "manifested_tree" else True
            reclaimed = [0, 1]
            if kind == "manifested_tree":
                record_by_path={item[0]:item[1:] for item in records}
                children={}
                for relative in record_by_path:
                    item=pathlib.PurePosixPath(relative)
                    parent=item.parent.as_posix()
                    if parent == ".": parent=""
                    children.setdefault(parent,set()).add(item.name)
                def names(directory_fd):
                    scan_fd=os.open(".",flags,dir_fd=directory_fd)
                    try:
                        assert mount_id(scan_fd) == root_mount
                        result={entry.name for entry in os.scandir(scan_fd)}
                        assert len(result) <= max_inodes
                        return result
                    finally:
                        os.close(scan_fd)
                def exact(item, expected_record):
                    return [item.st_dev,item.st_ino,item.st_mode,item.st_uid,
                            item.st_nlink,item.st_size,item.st_mtime_ns,
                            item.st_ctime_ns] == expected_record
                def erase(directory_fd, prefix=""):
                    remaining=set(children.get(prefix,set()))
                    assert names(directory_fd) == remaining
                    for name in sorted(tuple(remaining)):
                        relative=f"{prefix}/{name}" if prefix else name
                        expected_record=record_by_path[relative]
                        item=os.stat(name,dir_fd=directory_fd,follow_symlinks=False)
                        assert exact(item,expected_record)
                        if stat.S_ISDIR(item.st_mode):
                            child=os.open(name,flags,dir_fd=directory_fd)
                            try:
                                assert mount_id(child) == root_mount
                                assert exact(os.fstat(child),expected_record)
                                erase(child,relative)
                                assert names(child) == set()
                                emptied=os.fstat(child)
                                assert (emptied.st_dev,emptied.st_ino,emptied.st_mode,
                                        emptied.st_uid) == (item.st_dev,item.st_ino,
                                        item.st_mode,item.st_uid)
                                assert mount_id(child) == root_mount
                            finally:
                                os.close(child)
                            assert names(directory_fd) == remaining
                            current_item=os.stat(name,dir_fd=directory_fd,
                                                 follow_symlinks=False)
                            check=os.open(name,flags,dir_fd=directory_fd)
                            try:
                                assert mount_id(check) == root_mount
                                anchored_current=os.fstat(check)
                                assert (current_item.st_dev,current_item.st_ino,
                                        current_item.st_mode,current_item.st_uid,
                                        current_item.st_nlink,current_item.st_size,
                                        current_item.st_mtime_ns,current_item.st_ctime_ns) == (
                                        anchored_current.st_dev,anchored_current.st_ino,
                                        anchored_current.st_mode,anchored_current.st_uid,
                                        anchored_current.st_nlink,anchored_current.st_size,
                                        anchored_current.st_mtime_ns,
                                        anchored_current.st_ctime_ns) == (
                                        emptied.st_dev,emptied.st_ino,emptied.st_mode,
                                        emptied.st_uid,emptied.st_nlink,emptied.st_size,
                                        emptied.st_mtime_ns,emptied.st_ctime_ns)
                                os.rmdir(name,dir_fd=directory_fd)
                            finally:
                                os.close(check)
                        else:
                            assert stat.S_ISREG(item.st_mode) and item.st_nlink == 1
                            assert names(directory_fd) == remaining
                            child=os.open(name,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW,
                                          dir_fd=directory_fd)
                            try:
                                assert mount_id(child) == root_mount
                                assert exact(os.fstat(child),expected_record)
                                os.unlink(name,dir_fd=directory_fd)
                            finally:
                                os.close(child)
                        remaining.remove(name)
                        assert names(directory_fd) == remaining
                erase(target_fd)
                reclaimed=[measured[0],measured[1]]
                current = os.stat(path.name,dir_fd=parent_fd,follow_symlinks=False)
                anchored_current=os.fstat(target_fd)
                assert (current.st_dev,current.st_ino,current.st_mode,current.st_uid,
                        current.st_nlink,current.st_size,current.st_mtime_ns,
                        current.st_ctime_ns) == (anchored_current.st_dev,
                        anchored_current.st_ino,anchored_current.st_mode,
                        anchored_current.st_uid,anchored_current.st_nlink,
                        anchored_current.st_size,anchored_current.st_mtime_ns,
                        anchored_current.st_ctime_ns)
                assert (stat.S_ISDIR(current.st_mode) and current.st_uid == uid
                        and str(current.st_dev) == fsid
                        and mount_id(target_fd) == root_mount
                        and names(target_fd) == set())
                os.rmdir(path.name,dir_fd=parent_fd)
            else:
                current = os.stat(path.name,dir_fd=parent_fd,follow_symlinks=False)
                assert (current.st_dev,current.st_ino,current.st_ctime_ns) == (
                        meta.st_dev,meta.st_ino,meta.st_ctime_ns)
                assert (stat.S_ISREG(current.st_mode) and current.st_uid == uid
                        and str(current.st_dev) == fsid and current.st_nlink == 1
                        and mount_id(target_fd) == root_mount)
                reclaimed[0] = meta.st_size
                os.unlink(path.name,dir_fd=parent_fd)
        finally:
            os.close(target_fd)
    finally:
        os.close(parent_fd)
finally:
    os.close(root_fd)
print(json.dumps({"removed":True,"filesystem_id":fsid,"owner_uid":uid,
 "reclaimed_bytes":reclaimed[0],"reclaimed_inodes":reclaimed[1]}, sort_keys=True))
'''
        value = self._remote_python(
            host,
            script,
            str(checked),
            expected_filesystem_id,
            str(owner_uid),
            descriptor.kind.value,
            descriptor.digest_sha256,
            OWNERSHIP_XATTR,
            self._ownership_marker(descriptor.kind, descriptor.digest_sha256),
            identity_token,
            manifest_relative,
            str(descriptor.size_bytes_max),
            str(descriptor.transfer_bytes_max),
            str(descriptor.inode_count_max),
        )
        return CacheRemovalReceipt(**value)


def create_artifact_cache_backend() -> AeonQwenArtifactBackend:
    return AeonQwenArtifactBackend()
