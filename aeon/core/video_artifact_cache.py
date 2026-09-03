"""Exact bounded worker cache for Aeon's H3/LTX video renderer."""

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
from typing import Any, Callable

from fleet_compute.artifact_cache import (
    ArtifactCacheError,
    ArtifactCacheSafetyError,
    CacheEntryInspection,
    CacheRemovalReceipt,
    CacheRootInspection,
)
from fleet_compute.models import ArtifactDescriptor, ArtifactKind

from .video_comfy_release import (
    VIDEO_ARTIFACTS_BY_ID,
    VIDEO_IMAGE_ID,
    VIDEO_OCI_ARCHIVE_SHA256,
    VIDEO_WORKER_CACHE_ROOT,
    VIDEO_WORKER_HOSTNAMES,
)
from .fleet_hosts import network_address


REMOTE_PYTHON = (
    "/home/aday/.local/share/uv/python/"
    "cpython-3.12-linux-x86_64-gnu/bin/python3.12"
)
REMOTE_WRAPPER = "/home/aday/bin/fleet-low-priority"
OWNERSHIP_XATTR = "user.fleet_compute_cache"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_ROOT_PROOF = r'''
import os, pathlib, stat, sys
_root = pathlib.PurePosixPath(sys.argv[2])
assert _root == pathlib.PurePosixPath("/home/aday/.local/state/fleet-compute/cache/aeon-video-comfyui")
assert _root.is_absolute() and ".." not in _root.parts
_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_fd = os.open("/", _flags)
try:
    for _part in _root.parts[1:]:
        _child = os.open(_part, _flags, dir_fd=_fd)
        _meta = os.fstat(_child)
        assert stat.S_ISDIR(_meta.st_mode)
        os.close(_fd); _fd = _child
finally:
    os.close(_fd)
'''


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"


def _marker(kind: ArtifactKind, digest: str) -> str:
    return _canonical_json(
        {
            "schema_version": 1,
            "kind": kind.value,
            "digest_sha256": digest,
            "cache_root": str(VIDEO_WORKER_CACHE_ROOT),
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_video_oci_archive(
    archive: Path, *, digest: str, maximum_bytes: int
) -> int:
    """Validate Docker's current no-tag OCI-layout save for one platform image."""

    before = archive.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_mode & 0o022
        or before.st_nlink != 1
        or not 0 < before.st_size <= maximum_bytes
    ):
        raise ArtifactCacheSafetyError("canonical video OCI archive is unsafe")
    try:
        with tarfile.open(archive, mode="r:") as bundle:
            members = bundle.getmembers()
            if not 0 < len(members) <= 100_000:
                raise ArtifactCacheSafetyError(
                    "canonical video OCI member bound changed"
                )
            by_name: dict[str, tarfile.TarInfo] = {}
            for member in members:
                parts = PurePosixPath(member.name).parts
                if (
                    not parts
                    or member.name.startswith("/")
                    or ".." in parts
                    or not (member.isfile() or member.isdir())
                    or member.islnk()
                    or member.issym()
                    or member.name in by_name
                ):
                    raise ArtifactCacheSafetyError(
                        "canonical video OCI archive has an unsafe member"
                    )
                by_name[member.name] = member
            manifest_member = by_name.get("manifest.json")
            if (
                manifest_member is None
                or not manifest_member.isfile()
                or not 0 < manifest_member.size <= 16 * 1024 * 1024
            ):
                raise ArtifactCacheSafetyError(
                    "canonical video OCI manifest is unsafe"
                )
            manifest_handle = bundle.extractfile(manifest_member)
            if manifest_handle is None:
                raise ArtifactCacheSafetyError(
                    "canonical video OCI manifest is missing"
                )
            manifest = json.loads(manifest_handle.read().decode("utf-8"))
            if (
                not isinstance(manifest, list)
                or len(manifest) != 1
                or not isinstance(manifest[0], dict)
                or set(manifest[0]) != {"Config", "RepoTags", "Layers"}
                or manifest[0].get("RepoTags") not in (None, [])
                or not isinstance(manifest[0].get("Layers"), list)
                or not manifest[0]["Layers"]
            ):
                raise ArtifactCacheSafetyError(
                    "canonical video OCI image identity is absent"
                )
            config_name = manifest[0].get("Config")
            allowed_config_names = {
                f"{digest}.json",
                f"blobs/sha256/{digest}",
            }
            if config_name not in allowed_config_names:
                raise ArtifactCacheSafetyError(
                    "canonical video OCI config identity changed"
                )
            config_member = by_name.get(str(config_name))
            if (
                config_member is None
                or not config_member.isfile()
                or not 0 < config_member.size <= 16 * 1024 * 1024
            ):
                raise ArtifactCacheSafetyError(
                    "canonical video OCI config is unsafe"
                )
            config_handle = bundle.extractfile(config_member)
            if config_handle is None:
                raise ArtifactCacheSafetyError(
                    "canonical video OCI config is missing"
                )
            config_payload = config_handle.read()
            if hashlib.sha256(config_payload).hexdigest() != digest:
                raise ArtifactCacheSafetyError(
                    "canonical video OCI config digest changed"
                )
            config = json.loads(config_payload.decode("utf-8"))
            if not isinstance(config, dict) or not isinstance(config.get("config"), dict):
                raise ArtifactCacheSafetyError(
                    "canonical video OCI config is malformed"
                )
            layers = manifest[0]["Layers"]
            if any(
                not isinstance(layer, str)
                or layer.startswith("/")
                or ".." in PurePosixPath(layer).parts
                or layer not in by_name
                or not by_name[layer].isfile()
                for layer in layers
            ):
                raise ArtifactCacheSafetyError(
                    "canonical video OCI layer closure is unsafe"
                )
            regular = {name for name, member in by_name.items() if member.isfile()}
            required = {"manifest.json", str(config_name), *layers}
            visited_blobs: set[str] = set()
            index_member = by_name.get("index.json")
            if index_member is not None:
                if not index_member.isfile() or not 0 < index_member.size <= 16 * 1024 * 1024:
                    raise ArtifactCacheSafetyError(
                        "canonical video OCI index is unsafe"
                    )
                index_handle = bundle.extractfile(index_member)
                if index_handle is None:
                    raise ArtifactCacheSafetyError(
                        "canonical video OCI index is missing"
                    )
                index = json.loads(index_handle.read().decode("utf-8"))
                queue = list(index.get("manifests") or []) if isinstance(index, dict) else []
                while queue:
                    descriptor = queue.pop()
                    if not isinstance(descriptor, dict):
                        raise ArtifactCacheSafetyError(
                            "canonical video OCI descriptor is malformed"
                        )
                    raw_digest = descriptor.get("digest")
                    raw_size = descriptor.get("size")
                    if (
                        not isinstance(raw_digest, str)
                        or not raw_digest.startswith("sha256:")
                        or not re.fullmatch(r"[a-f0-9]{64}", raw_digest[7:])
                        or isinstance(raw_size, bool)
                        or not isinstance(raw_size, int)
                        or raw_size <= 0
                    ):
                        raise ArtifactCacheSafetyError(
                            "canonical video OCI descriptor identity changed"
                        )
                    blob_name = f"blobs/sha256/{raw_digest[7:]}"
                    blob = by_name.get(blob_name)
                    if blob is None or not blob.isfile() or blob.size != raw_size:
                        raise ArtifactCacheSafetyError(
                            "canonical video OCI descriptor target changed"
                        )
                    if blob_name in visited_blobs:
                        continue
                    blob_handle = bundle.extractfile(blob)
                    if blob_handle is None:
                        raise ArtifactCacheSafetyError(
                            "canonical video OCI descriptor target is missing"
                        )
                    hasher = hashlib.sha256()
                    chunks: list[bytes] = []
                    collect_json = blob.size <= 16 * 1024 * 1024
                    for chunk in iter(lambda: blob_handle.read(4 * 1024 * 1024), b""):
                        hasher.update(chunk)
                        if collect_json:
                            chunks.append(chunk)
                    if hasher.hexdigest() != raw_digest[7:]:
                        raise ArtifactCacheSafetyError(
                            "canonical video OCI blob digest changed"
                        )
                    visited_blobs.add(blob_name)
                    media_type = descriptor.get("mediaType")
                    if media_type in {
                        "application/vnd.oci.image.index.v1+json",
                        "application/vnd.docker.distribution.manifest.list.v2+json",
                    }:
                        child = json.loads(b"".join(chunks).decode("utf-8"))
                        if not isinstance(child, dict) or not isinstance(
                            child.get("manifests"), list
                        ):
                            raise ArtifactCacheSafetyError(
                                "canonical video OCI child index is malformed"
                            )
                        queue.extend(child["manifests"])
                    elif media_type in {
                        "application/vnd.oci.image.manifest.v1+json",
                        "application/vnd.docker.distribution.manifest.v2+json",
                    }:
                        child = json.loads(b"".join(chunks).decode("utf-8"))
                        if (
                            not isinstance(child, dict)
                            or not isinstance(child.get("config"), dict)
                            or not isinstance(child.get("layers"), list)
                        ):
                            raise ArtifactCacheSafetyError(
                                "canonical video OCI child manifest is malformed"
                            )
                        queue.append(child["config"])
                        queue.extend(child["layers"])
            allowed = {
                *required,
                *visited_blobs,
                "index.json",
                "oci-layout",
            }
            if not required <= regular or regular != allowed:
                raise ArtifactCacheSafetyError(
                    "canonical video OCI archive has unreferenced payloads"
                )
            allowed_dirs = set()
            for name in regular:
                parts = PurePosixPath(name).parts[:-1]
                allowed_dirs.update(
                    PurePosixPath(*parts[:depth]).as_posix()
                    for depth in range(1, len(parts) + 1)
                )
            actual_dirs = {
                name.rstrip("/")
                for name, member in by_name.items()
                if member.isdir()
            }
            if actual_dirs != allowed_dirs:
                raise ArtifactCacheSafetyError(
                    "canonical video OCI directory closure changed"
                )
    except (OSError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCacheSafetyError(
            "canonical video OCI archive is malformed"
        ) from exc
    after = archive.lstat()
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
        raise ArtifactCacheSafetyError(
            "canonical video OCI archive changed while inspected"
        )
    return before.st_size


class VideoArtifactCacheBackend:
    """Transfer only the profile's exact files to worker `.179`."""

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
        hostname = VIDEO_WORKER_HOSTNAMES.get(host)
        if hostname is None:
            raise ArtifactCacheSafetyError(
                "video artifact cache host is not release-qualified"
            )
        return hostname

    @classmethod
    def _ssh(cls, host: str) -> list[str]:
        cls._host(host)
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

    @staticmethod
    def _worker_path(value: str, *, allow_root: bool = False) -> Path:
        path = Path(value)
        try:
            relative = PurePosixPath(str(path)).relative_to(
                PurePosixPath(str(VIDEO_WORKER_CACHE_ROOT))
            )
        except ValueError as exc:
            raise ArtifactCacheSafetyError(
                "video artifact path escaped its Fleet cache root"
            ) from exc
        if ".." in PurePosixPath(str(path)).parts or (
            not relative.parts and not allow_root
        ):
            raise ArtifactCacheSafetyError("video artifact cache path is unsafe")
        return path

    @staticmethod
    def _validate_descriptor(descriptor: ArtifactDescriptor) -> None:
        expected = VIDEO_ARTIFACTS_BY_ID.get(descriptor.artifact_id)
        if (
            expected is None
            or descriptor.identity_key != expected.identity_key
            or descriptor.kind is not expected.kind
            or Path(descriptor.canonical_path) != expected.canonical_path
            or descriptor.digest_sha256 != expected.digest_sha256
            or descriptor.kind
            not in {ArtifactKind.FILE, ArtifactKind.OCI_ARCHIVE}
        ):
            raise ArtifactCacheSafetyError(
                "video artifact descriptor differs from the reviewed release"
            )

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
                    (_ROOT_PROOF + "\n" + script) if prove_root else script,
                    hostname,
                    str(VIDEO_WORKER_CACHE_ROOT),
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
            raise ArtifactCacheSafetyError("video worker cache proof failed")
        try:
            value = json.loads(result.stdout)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactCacheSafetyError(
                "video worker cache proof is malformed"
            ) from exc
        if not isinstance(value, dict):
            raise ArtifactCacheSafetyError("video worker cache proof is malformed")
        return value

    def _run_with_progress(
        self,
        command: list[str],
        *,
        progress: Callable[[int, int], None],
        total: int,
        progress_probe: Callable[[], int],
        timeout: float = 7200,
    ) -> None:
        completed = 0
        progress(0, total)
        process = self._popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = self._clock() + timeout
        try:
            while True:
                try:
                    output, error = process.communicate(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    if self._clock() >= deadline:
                        process.kill()
                        process.communicate()
                        raise ArtifactCacheError("video artifact transfer timed out")
                    observed = progress_probe()
                    if isinstance(observed, bool) or not isinstance(observed, int):
                        raise ArtifactCacheSafetyError(
                            "video artifact progress proof is malformed"
                        )
                    completed = max(completed, min(total, observed))
                    progress(completed, total)
        except BaseException:
            if process.returncode is None:
                try:
                    process.terminate()
                    process.communicate(timeout=5)
                except (AttributeError, subprocess.TimeoutExpired):
                    process.kill()
                    process.communicate(timeout=5)
            raise
        if process.returncode != 0:
            del output, error
            raise ArtifactCacheError("video artifact transfer command failed")
        observed = progress_probe()
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise ArtifactCacheSafetyError(
                "video artifact progress proof is malformed"
            )
        progress(max(completed, min(total, observed)), total)

    def ensure_cache_root(
        self, *, host: str, root: str, owner_uid: int
    ) -> CacheRootInspection:
        if Path(root) != VIDEO_WORKER_CACHE_ROOT or owner_uid != os.geteuid():
            raise ArtifactCacheSafetyError("video worker cache root changed")
        script = r'''
import json, os, pathlib, stat, sys
expected, fixed_root, requested, uid_raw = sys.argv[1:5]
assert os.uname().nodename == expected and requested == fixed_root
assert int(uid_raw) == os.geteuid()
target = pathlib.PurePosixPath(requested)
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
fd = os.open("/", flags); current = pathlib.PurePosixPath("/")
try:
    for part in target.parts[1:]:
        current = current / part
        try:
            child = os.open(part, flags, dir_fd=fd)
        except FileNotFoundError:
            assert current.is_relative_to(pathlib.PurePosixPath("/home/aday/.local"))
            os.mkdir(part, mode=0o700, dir_fd=fd)
            child = os.open(part, flags, dir_fd=fd)
        meta = os.fstat(child); assert stat.S_ISDIR(meta.st_mode)
        if current.is_relative_to(pathlib.PurePosixPath("/home/aday")):
            assert meta.st_uid == os.geteuid() and not meta.st_mode & 0o022
        else:
            assert not meta.st_mode & 0o002
        os.close(fd); fd = child
    os.fchmod(fd, 0o700); meta = os.fstat(fd); values = os.fstatvfs(fd)
finally:
    os.close(fd)
print(json.dumps({"filesystem_id":str(meta.st_dev),"owner_uid":meta.st_uid,
 "is_directory":True,"is_symlink":False,
 "free_bytes":values.f_bavail*values.f_frsize,"free_inodes":values.f_favail},
 sort_keys=True))
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
            raise ArtifactCacheSafetyError("video cache verification mode changed")
        script = r'''
import hashlib, json, os, pathlib, re, stat, sys
expected, fixed_root, raw, fsid, kind, digest, verify_raw, attr_name, attr_value, maximum_raw = sys.argv[1:11]
assert os.uname().nodename == expected
root, path = pathlib.Path(fixed_root), pathlib.Path(raw); maximum = int(maximum_raw)
root_meta = root.lstat(); assert stat.S_ISDIR(root_meta.st_mode)
assert root_meta.st_uid == os.geteuid() and str(root_meta.st_dev) == fsid
try:
    relative = path.relative_to(root); meta = path.lstat()
except FileNotFoundError:
    print(json.dumps({"state":"absent"})); raise SystemExit(0)
assert relative.parts and ".." not in relative.parts
assert stat.S_ISREG(meta.st_mode) and not stat.S_ISLNK(meta.st_mode)
assert meta.st_uid == os.geteuid() and meta.st_nlink == 1
assert str(meta.st_dev) == fsid and not os.path.ismount(path)
assert 0 <= meta.st_size <= maximum
cursor = root
for index, part in enumerate(relative.parts):
    cursor = cursor / part; item = cursor.lstat()
    assert not stat.S_ISLNK(item.st_mode) and item.st_uid == os.geteuid()
    assert str(item.st_dev) == fsid and not os.path.ismount(cursor)
    if index + 1 < len(relative.parts): assert stat.S_ISDIR(item.st_mode)
try:
    marker = os.getxattr(path, attr_name, follow_symlinks=False).decode("utf-8")
except OSError:
    marker = ""
fleet_owned = marker == attr_value
material = {"schema":1,"path":str(path),"marker":marker,
 "stat":[meta.st_dev,meta.st_ino,meta.st_mode,meta.st_uid,meta.st_nlink,
         meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns]}
token = (hashlib.sha256(json.dumps(material,sort_keys=True,separators=(",",":"),
 allow_nan=False).encode()).hexdigest() if fleet_owned else None)
payload = None
if verify_raw == "1":
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b""): hasher.update(chunk)
    payload = hasher.hexdigest()
print(json.dumps({"state":"present","filesystem_id":str(meta.st_dev),
 "owner_uid":meta.st_uid,"size_bytes":meta.st_size,"inode_count":1,
 "fleet_owned":fleet_owned,"identity_token":token,"payload_sha256":payload},
 sort_keys=True))
'''
        value = self._remote_python(
            host,
            script,
            str(checked),
            expected_filesystem_id,
            descriptor.kind.value,
            descriptor.digest_sha256,
            "1" if verify_content else "0",
            OWNERSHIP_XATTR,
            _marker(descriptor.kind, descriptor.digest_sha256),
            str(max(descriptor.size_bytes_max, descriptor.transfer_bytes_max)),
            timeout=3600 if verify_content else 60,
        )
        if value.get("state") == "absent":
            return None
        if value.get("state") != "present":
            raise ArtifactCacheSafetyError("video cache inspection is malformed")
        payload = value.get("payload_sha256")
        content_ok = bool(
            verify_content
            and isinstance(payload, str)
            and _SHA256.fullmatch(payload)
            and (
                payload == (
                    VIDEO_OCI_ARCHIVE_SHA256
                    if descriptor.kind is ArtifactKind.OCI_ARCHIVE
                    else descriptor.digest_sha256
                )
            )
        )
        semantic_ready = True
        if descriptor.kind is ArtifactKind.OCI_ARCHIVE:
            semantic_ready = self._remote_image_ready(host)
        return CacheEntryInspection(
            filesystem_id=str(value["filesystem_id"]),
            owner_uid=int(value["owner_uid"]),
            kind=descriptor.kind,
            semantic_digest_sha256=descriptor.digest_sha256,
            size_bytes=int(value["size_bytes"]),
            inode_count=1,
            is_regular_file=True,
            is_directory=False,
            fleet_owned=bool(value.get("fleet_owned")),
            content_identity_verified=content_ok,
            semantic_ready=semantic_ready,
            payload_sha256=payload if content_ok else None,
            identity_token=value.get("identity_token"),
        )

    def _remote_image_ready(self, host: str) -> bool:
        result = self._run(
            [
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
                        "/usr/bin/docker",
                        "image",
                        "inspect",
                        "--format",
                        "{{.Id}}",
                        VIDEO_IMAGE_ID,
                    ]
                ),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0 and result.stdout.strip() == VIDEO_IMAGE_ID

    def _prepare_temporary(
        self,
        host: str,
        path: Path,
        filesystem_id: str,
        descriptor: ArtifactDescriptor,
    ) -> None:
        script = r'''
import json, os, pathlib, re, stat, sys
expected, fixed_root, raw, fsid, digest, attr_name, attr_value = sys.argv[1:8]
assert os.uname().nodename == expected
root, path = pathlib.PurePosixPath(fixed_root), pathlib.PurePosixPath(raw)
match = re.fullmatch(r"([a-f0-9]{64})[.][a-f0-9]{1,128}[.]partial", path.name)
assert path.parent == root / ".staging" and match and match.group(1) == digest
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
root_fd = os.open("/", flags)
for part in root.parts[1:]:
    child = os.open(part, flags, dir_fd=root_fd); os.close(root_fd); root_fd = child
try:
    root_meta = os.fstat(root_fd); assert str(root_meta.st_dev) == fsid
    try: staging_fd = os.open(".staging", flags, dir_fd=root_fd)
    except FileNotFoundError:
        os.mkdir(".staging", mode=0o700, dir_fd=root_fd)
        staging_fd = os.open(".staging", flags, dir_fd=root_fd)
    try:
        try:
            target_fd = os.open(path.name, os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW,
                                dir_fd=staging_fd); created = False
        except FileNotFoundError:
            target_fd = os.open(path.name, os.O_RDWR|os.O_CREAT|os.O_EXCL|
                                os.O_CLOEXEC|os.O_NOFOLLOW, 0o600,
                                dir_fd=staging_fd); created = True
        try:
            meta = os.fstat(target_fd); assert stat.S_ISREG(meta.st_mode)
            assert meta.st_uid == os.geteuid() and meta.st_nlink == 1
            assert str(meta.st_dev) == fsid
            if created:
                os.setxattr(target_fd, attr_name, attr_value.encode(), flags=os.XATTR_CREATE)
            assert os.getxattr(target_fd, attr_name).decode() == attr_value
            os.fchmod(target_fd, 0o600)
        finally: os.close(target_fd)
    finally: os.close(staging_fd)
finally: os.close(root_fd)
print(json.dumps({"ok":True}))
'''
        value = self._remote_python(
            host,
            script,
            str(path),
            filesystem_id,
            descriptor.digest_sha256,
            OWNERSHIP_XATTR,
            _marker(descriptor.kind, descriptor.digest_sha256),
        )
        if value != {"ok": True}:
            raise ArtifactCacheSafetyError("video cache staging path was not prepared")

    def _remote_size(
        self, host: str, path: Path, filesystem_id: str, descriptor: ArtifactDescriptor
    ) -> int:
        script = r'''
import json, os, pathlib, stat, sys
expected, fixed_root, raw, fsid, attr_name, attr_value, maximum_raw = sys.argv[1:8]
assert os.uname().nodename == expected
path = pathlib.Path(raw); meta = path.lstat(); maximum = int(maximum_raw)
assert stat.S_ISREG(meta.st_mode) and not stat.S_ISLNK(meta.st_mode)
assert meta.st_uid == os.geteuid() and meta.st_nlink == 1
assert str(meta.st_dev) == fsid and not os.path.ismount(path)
assert os.getxattr(path, attr_name, follow_symlinks=False).decode() == attr_value
assert 0 <= meta.st_size <= maximum
print(json.dumps({"bytes":meta.st_size}))
'''
        value = self._remote_python(
            host,
            script,
            str(path),
            filesystem_id,
            OWNERSHIP_XATTR,
            _marker(descriptor.kind, descriptor.digest_sha256),
            str(max(descriptor.transfer_bytes_max, descriptor.size_bytes_max)),
        )
        return int(value["bytes"])

    def _load_remote_image(self, host: str, archive: Path) -> None:
        if self._remote_image_ready(host):
            return
        result = self._run(
            [
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
                        "/usr/bin/docker",
                        "image",
                        "load",
                        "--input",
                        str(archive),
                    ]
                ),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0 or not self._remote_image_ready(host):
            raise ArtifactCacheSafetyError("video worker image load failed")

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
            raise ArtifactCacheSafetyError("video cache transfer limit changed")
        source = Path(descriptor.canonical_path)
        metadata = source.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= descriptor.size_bytes_max
        ):
            raise ArtifactCacheSafetyError("canonical video artifact is unsafe")
        if descriptor.kind is ArtifactKind.FILE:
            if _sha256(source) != descriptor.digest_sha256:
                raise ArtifactCacheSafetyError(
                    "canonical video artifact identity changed"
                )
        else:
            _validate_video_oci_archive(
                source,
                digest=descriptor.digest_sha256,
                maximum_bytes=descriptor.size_bytes_max,
            )
        self._prepare_temporary(host, temporary, expected_filesystem_id, descriptor)
        transport = shlex.join(self._ssh(host)[:-1])
        bandwidth_kib = max(1, max_bytes_per_second // 1024)
        command = [
            "/home/aday/bin/fleet-low-priority",
            "/usr/bin/rsync",
            "--archive",
            "--checksum",
            "--inplace",
            "--chmod=Fu=rw,Fgo=",
            "--protect-args",
            f"--bwlimit={bandwidth_kib}",
            "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
            "-e",
            transport,
            "--",
            str(source),
            f"aday@{network_address(host)}:{temporary}",
        ]
        self._run_with_progress(
            command,
            progress=progress,
            total=descriptor.transfer_bytes_max,
            progress_probe=lambda: self._remote_size(
                host, temporary, expected_filesystem_id, descriptor
            ),
        )
        if descriptor.kind is ArtifactKind.OCI_ARCHIVE:
            self._load_remote_image(host, temporary)
        progress(descriptor.transfer_bytes_max, descriptor.transfer_bytes_max)

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
        if owner_uid != os.geteuid() or _SHA256.fullmatch(identity_token) is None:
            raise ArtifactCacheSafetyError("video cache promotion owner changed")
        script = r'''
import ctypes, errno, hashlib, json, os, pathlib, re, stat, sys
expected, fixed_root, temp_raw, final_raw, fsid, uid_raw, kind, digest, attr_name, attr_value, wanted, maximum_raw, payload_digest = sys.argv[1:14]
assert os.uname().nodename == expected
root = pathlib.PurePosixPath(fixed_root); temp = pathlib.PurePosixPath(temp_raw)
final = pathlib.PurePosixPath(final_raw); uid = int(uid_raw)
match = re.fullmatch(r"([a-f0-9]{64})[.][a-f0-9]{1,128}[.]partial", temp.name)
assert temp.parent == root/".staging" and match and match.group(1) == digest
assert final == root/"sha256"/digest[:2]/digest
assert kind in {"file","oci_archive"}; maximum = int(maximum_raw)
assert re.fullmatch(r"[a-f0-9]{64}", digest)
assert re.fullmatch(r"[a-f0-9]{64}", payload_digest)
flags = os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC|os.O_NOFOLLOW
root_fd = os.open("/", flags)
for part in root.parts[1:]:
    child = os.open(part, flags, dir_fd=root_fd); os.close(root_fd); root_fd = child
try:
    def mount_id(fd):
        with open(f"/proc/self/fdinfo/{fd}","r",encoding="ascii") as info:
            values=[line.split(":",1)[1].strip() for line in info if line.startswith("mnt_id:")]
        assert len(values)==1 and values[0].isdecimal(); return values[0]
    root_mount = mount_id(root_fd)
    root_meta = os.fstat(root_fd); assert root_meta.st_uid == uid
    assert str(root_meta.st_dev) == fsid and stat.S_ISDIR(root_meta.st_mode)
    assert not root_meta.st_mode & 0o077
    staging_fd = os.open(".staging", flags, dir_fd=root_fd)
    try:
        staging_meta = os.fstat(staging_fd)
        assert stat.S_ISDIR(staging_meta.st_mode) and staging_meta.st_uid == uid
        assert str(staging_meta.st_dev) == fsid and not staging_meta.st_mode & 0o077
        assert mount_id(staging_fd) == root_mount
        temp_fd = os.open(temp.name, os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW,
                          dir_fd=staging_fd)
        try:
            before = os.fstat(temp_fd); assert stat.S_ISREG(before.st_mode)
            assert before.st_uid == uid and before.st_nlink == 1
            assert str(before.st_dev) == fsid and 0 < before.st_size <= maximum
            assert not before.st_mode & 0o077 and mount_id(temp_fd) == root_mount
            marker_raw = os.getxattr(temp_fd, attr_name).decode(); assert marker_raw == attr_value
            marker = json.loads(marker_raw)
            assert marker == {"schema_version":1,"kind":kind,
                              "digest_sha256":digest,"cache_root":str(root)}
            material={"schema":1,"path":str(temp),"marker":marker_raw,
             "stat":[before.st_dev,before.st_ino,before.st_mode,before.st_uid,
                     before.st_nlink,before.st_size,before.st_mtime_ns,before.st_ctime_ns]}
            token=hashlib.sha256(json.dumps(material,sort_keys=True,separators=(",",":"),
              allow_nan=False).encode()).hexdigest(); assert token == wanted
            hasher=hashlib.sha256(); os.lseek(temp_fd,0,os.SEEK_SET); measured=0
            while True:
                chunk=os.read(temp_fd,4*1024*1024)
                if not chunk: break
                measured += len(chunk); assert measured <= maximum; hasher.update(chunk)
            assert measured == before.st_size and hasher.hexdigest() == payload_digest
            verified = os.fstat(temp_fd)
            assert (verified.st_dev,verified.st_ino,verified.st_mode,verified.st_uid,
                    verified.st_nlink,verified.st_size,verified.st_mtime_ns,
                    verified.st_ctime_ns) == (
                    before.st_dev,before.st_ino,before.st_mode,before.st_uid,
                    before.st_nlink,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
            try: sha_fd = os.open("sha256", flags, dir_fd=root_fd)
            except FileNotFoundError:
                os.mkdir("sha256", mode=0o700, dir_fd=root_fd)
                sha_fd = os.open("sha256", flags, dir_fd=root_fd)
            try:
                sha_meta = os.fstat(sha_fd)
                assert stat.S_ISDIR(sha_meta.st_mode) and sha_meta.st_uid == uid
                assert str(sha_meta.st_dev) == fsid and not sha_meta.st_mode & 0o077
                assert mount_id(sha_fd) == root_mount
                try: prefix_fd = os.open(digest[:2], flags, dir_fd=sha_fd)
                except FileNotFoundError:
                    os.mkdir(digest[:2], mode=0o700, dir_fd=sha_fd)
                    prefix_fd = os.open(digest[:2], flags, dir_fd=sha_fd)
                try:
                    prefix_meta = os.fstat(prefix_fd)
                    assert stat.S_ISDIR(prefix_meta.st_mode) and prefix_meta.st_uid == uid
                    assert str(prefix_meta.st_dev) == fsid and not prefix_meta.st_mode & 0o077
                    assert mount_id(prefix_fd) == root_mount
                    immediate = os.fstat(temp_fd)
                    assert (immediate.st_dev,immediate.st_ino,immediate.st_mode,
                            immediate.st_uid,immediate.st_nlink,immediate.st_size,
                            immediate.st_mtime_ns,immediate.st_ctime_ns) == (
                            before.st_dev,before.st_ino,before.st_mode,before.st_uid,
                            before.st_nlink,before.st_size,before.st_mtime_ns,
                            before.st_ctime_ns)
                    assert os.getxattr(temp_fd,attr_name).decode() == attr_value
                    assert mount_id(temp_fd) == root_mount
                    libc = ctypes.CDLL(None, use_errno=True); renameat2 = libc.renameat2
                    renameat2.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,
                                        ctypes.c_char_p,ctypes.c_uint]; renameat2.restype=ctypes.c_int
                    result=renameat2(staging_fd,os.fsencode(temp.name),prefix_fd,
                                     os.fsencode(digest),1)
                    if result:
                        error=ctypes.get_errno(); assert error not in {errno.EEXIST,errno.ENOTEMPTY}
                        raise OSError(error,os.strerror(error))
                    published=os.stat(digest,dir_fd=prefix_fd,follow_symlinks=False)
                    assert (published.st_dev,published.st_ino)==(before.st_dev,before.st_ino)
                finally: os.close(prefix_fd)
            finally: os.close(sha_fd)
        finally: os.close(temp_fd)
    finally: os.close(staging_fd)
finally: os.close(root_fd)
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
            _marker(descriptor.kind, descriptor.digest_sha256),
            identity_token,
            str(max(descriptor.size_bytes_max, descriptor.transfer_bytes_max)),
            (
                VIDEO_OCI_ARCHIVE_SHA256
                if descriptor.kind is ArtifactKind.OCI_ARCHIVE
                else descriptor.digest_sha256
            ),
        )
        if value != {"ok": True}:
            raise ArtifactCacheSafetyError("video cache promotion failed")

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
        if owner_uid != os.geteuid() or _SHA256.fullmatch(identity_token) is None:
            raise ArtifactCacheSafetyError("video cache removal identity changed")
        script = r'''
import hashlib, json, os, pathlib, re, stat, sys
expected, fixed_root, raw, fsid, uid_raw, kind, digest, attr_name, attr_value, wanted = sys.argv[1:11]
assert os.uname().nodename == expected
root=pathlib.Path(fixed_root); path=pathlib.Path(raw); uid=int(uid_raw)
partial = path.parent == root/".staging"
if partial:
    match=re.fullmatch(r"([a-f0-9]{64})[.][a-f0-9]{1,128}[.]partial",path.name)
    assert match and match.group(1)==digest
else: assert path == root/"sha256"/digest[:2]/digest
meta=path.lstat(); assert stat.S_ISREG(meta.st_mode) and not stat.S_ISLNK(meta.st_mode)
assert meta.st_uid==uid and meta.st_nlink==1 and str(meta.st_dev)==fsid
assert not os.path.ismount(path)
marker=os.getxattr(path,attr_name,follow_symlinks=False).decode(); assert marker==attr_value
material={"schema":1,"path":str(path),"marker":marker,
 "stat":[meta.st_dev,meta.st_ino,meta.st_mode,meta.st_uid,meta.st_nlink,
         meta.st_size,meta.st_mtime_ns,meta.st_ctime_ns]}
token=hashlib.sha256(json.dumps(material,sort_keys=True,separators=(",",":"),
 allow_nan=False).encode()).hexdigest(); assert token==wanted
parent=path.parent; name=path.name; parent_fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|
 os.O_CLOEXEC|os.O_NOFOLLOW)
try:
    current=os.stat(name,dir_fd=parent_fd,follow_symlinks=False)
    assert (current.st_dev,current.st_ino,current.st_ctime_ns)==(meta.st_dev,meta.st_ino,meta.st_ctime_ns)
    os.unlink(name,dir_fd=parent_fd)
finally: os.close(parent_fd)
print(json.dumps({"removed":True,"filesystem_id":fsid,"owner_uid":uid,
 "reclaimed_bytes":meta.st_size,"reclaimed_inodes":1},sort_keys=True))
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
            _marker(descriptor.kind, descriptor.digest_sha256),
            identity_token,
        )
        return CacheRemovalReceipt(**value)


def create_video_artifact_cache_backend() -> VideoArtifactCacheBackend:
    return VideoArtifactCacheBackend()
