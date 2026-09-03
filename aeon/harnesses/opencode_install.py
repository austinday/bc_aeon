"""Pinned, fail-closed OpenCode binary resolution and installation.

The status path never creates files or directories and never uses ``PATH``.
Installation accepts only the catalog's exact Linux x86-64 release archive,
extracts one regular file, validates its version, and publishes a complete
version directory with an atomic rename beneath ``AEON_OPENCODE_HOME``.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Mapping, Sequence
from urllib.parse import urlparse

from .catalog import OPENCODE_ARTIFACT, OPENCODE_HOME_ENV, HarnessArtifact


DEFAULT_OPENCODE_HOME = Path.home() / ".local" / "share" / "aeon" / "opencode"
INSTALL_RECEIPT_NAME = "install.json"
INSTALL_RECEIPT_SCHEMA = 1
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
VERSION_PROBE_TIMEOUT_SECONDS = 10.0
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_LINE_RE = re.compile(
    r"^(?:opencode(?:\s+version)?\s+)?v?(\d+\.\d+\.\d+)$", re.I
)
_ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)
_VERSION_PROBE_ENVIRONMENT = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TMPDIR",
        "TZ",
        "USER",
    }
)


class OpenCodeInstallError(RuntimeError):
    """The pinned OpenCode binary could not be trusted or installed."""


@dataclass(frozen=True)
class OpenCodeStatus:
    """Sanitized result of inspecting one deterministic install location."""

    state: str
    ready: bool
    version: str
    home: str
    binary: str
    reason: str
    binary_sha256: str | None = None
    observed_version: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["installed"] = self.ready
        return payload


def resolve_opencode_home(
    value: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the configured root lexically without touching the filesystem."""

    source: str | os.PathLike[str]
    if value is not None:
        source = value
    else:
        env = os.environ if environ is None else environ
        source = env.get(OPENCODE_HOME_ENV, "") or DEFAULT_OPENCODE_HOME
    text = os.fspath(source)
    if not text or "\x00" in text:
        raise OpenCodeInstallError("OpenCode home is invalid")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        raise OpenCodeInstallError(
            f"{OPENCODE_HOME_ENV} must name an absolute directory"
        )
    normalized = Path(os.path.normpath(os.fspath(candidate)))
    if normalized == Path("/"):
        raise OpenCodeInstallError("OpenCode home cannot be the filesystem root")
    return normalized


def opencode_version_dir(
    home: str | os.PathLike[str] | None = None,
    *,
    artifact: HarnessArtifact = OPENCODE_ARTIFACT,
) -> Path:
    return resolve_opencode_home(home) / "versions" / f"v{artifact.version}"


def opencode_binary_path(
    home: str | os.PathLike[str] | None = None,
    *,
    artifact: HarnessArtifact = OPENCODE_ARTIFACT,
) -> Path:
    return opencode_version_dir(home, artifact=artifact) / artifact.executable_name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_owner_file(path: Path, *, executable: bool) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise OpenCodeInstallError(f"required file is missing: {path.name}") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise OpenCodeInstallError(f"refusing non-regular or symlink file: {path.name}")
    if metadata.st_uid != os.geteuid():
        raise OpenCodeInstallError(f"refusing file not owned by this user: {path.name}")
    if metadata.st_nlink != 1:
        raise OpenCodeInstallError(f"refusing multiply-linked file: {path.name}")
    mode = stat.S_IMODE(metadata.st_mode)
    if mode & 0o077:
        raise OpenCodeInstallError(f"refusing non-private file: {path.name}")
    if executable:
        if not mode & stat.S_IXUSR:
            raise OpenCodeInstallError(
                f"OpenCode binary is not owner-executable: {path.name}"
            )
    elif mode & 0o111:
        raise OpenCodeInstallError(f"refusing executable metadata file: {path.name}")
    return metadata


def _private_directory(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise OpenCodeInstallError(
            f"required directory is missing: {path.name}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise OpenCodeInstallError(f"refusing non-directory or symlink: {path.name}")
    if metadata.st_uid != os.geteuid():
        raise OpenCodeInstallError(
            f"refusing directory not owned by this user: {path.name}"
        )
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise OpenCodeInstallError(f"refusing non-private directory: {path.name}")
    return metadata


def _assert_no_symlink_components(path: Path) -> None:
    """Reject every existing symlink component without resolving through it."""

    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise OpenCodeInstallError(
                f"OpenCode install path contains a symlink component: {component}"
            )


def _read_receipt(path: Path, artifact: HarnessArtifact) -> dict[str, object]:
    metadata = _regular_owner_file(path, executable=False)
    if metadata.st_size > 16 * 1024:
        raise OpenCodeInstallError("OpenCode install receipt is too large")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpenCodeInstallError("OpenCode install receipt is invalid") from exc
    if not isinstance(payload, dict):
        raise OpenCodeInstallError("OpenCode install receipt is invalid")
    expected = {
        "schema_version": INSTALL_RECEIPT_SCHEMA,
        "version": artifact.version,
        "archive_name": artifact.archive_name,
        "archive_sha256": artifact.archive_sha256,
        "source_url": artifact.url,
        "binary_sha256": artifact.executable_sha256,
        "binary_size": artifact.executable_size,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise OpenCodeInstallError(f"OpenCode receipt {key} does not match the pin")
    digest = payload.get("binary_sha256")
    size = payload.get("binary_size")
    if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
        raise OpenCodeInstallError("OpenCode receipt binary digest is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise OpenCodeInstallError("OpenCode receipt binary size is invalid")
    return payload


def _probe_version(
    binary: Path,
    *,
    timeout: float = VERSION_PROBE_TIMEOUT_SECONDS,
) -> str:
    # ``aeon doctor`` calls this probe.  Even a digest-pinned binary has no need
    # to receive API keys, service credentials, loader hooks, Fleet authority,
    # or caller-supplied OpenCode configuration merely to print its version.
    try:
        # Pinned OpenCode initializes all XDG directories before dispatching
        # even ``--version``.  Route that unavoidable probe activity into one
        # exact temporary home so read-only status never touches real user
        # config/data/cache/state or the process-global /tmp/opencode path.
        with tempfile.TemporaryDirectory(prefix="aeon-opencode-version-") as temporary:
            probe_root = Path(temporary)
            probe_root.chmod(0o700)
            environment = {
                key: value
                for key, value in os.environ.items()
                if key in _VERSION_PROBE_ENVIRONMENT
                and isinstance(value, str)
                and "\x00" not in value
            }
            environment.setdefault("PATH", os.defpath)
            environment.update(
                {
                    "HOME": str(probe_root / "home"),
                    "TMPDIR": str(probe_root / "tmp"),
                    "XDG_CACHE_HOME": str(probe_root / "cache"),
                    "XDG_CONFIG_HOME": str(probe_root / "config"),
                    "XDG_DATA_HOME": str(probe_root / "data"),
                    "XDG_RUNTIME_DIR": str(probe_root / "runtime"),
                    "XDG_STATE_HOME": str(probe_root / "state"),
                    "OPENCODE_CONFIG_DIR": str(probe_root / "config" / "opencode"),
                    "OPENCODE_TEST_HOME": str(probe_root / "home"),
                    "OPENCODE_DISABLE_AUTOUPDATE": "true",
                    "OPENCODE_DISABLE_MODELS_FETCH": "true",
                    "OPENCODE_DISABLE_PROJECT_CONFIG": "true",
                    "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
                }
            )
            completed = subprocess.run(
                [os.fspath(binary), "--version"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
                env=environment,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OpenCodeInstallError("OpenCode version probe failed") from exc
    if completed.returncode != 0:
        raise OpenCodeInstallError("OpenCode version probe returned an error")
    output = completed.stdout.strip()
    match = _VERSION_LINE_RE.fullmatch(output)
    if not match:
        raise OpenCodeInstallError("OpenCode returned an invalid version string")
    return match.group(1)


def _inspect_opencode_status(
    home: str | os.PathLike[str] | None = None,
    *,
    artifact: HarnessArtifact = OPENCODE_ARTIFACT,
    system: str | None = None,
    machine: str | None = None,
    version_probe: Callable[[Path], str] = _probe_version,
) -> OpenCodeStatus:
    """Inspect the pinned install without creating, repairing, or downloading."""

    try:
        root = resolve_opencode_home(home)
    except OpenCodeInstallError as exc:
        return OpenCodeStatus(
            state="invalid",
            ready=False,
            version=artifact.version,
            home=os.fspath(home or ""),
            binary="",
            reason=str(exc),
        )
    binary = opencode_binary_path(root, artifact=artifact)
    detected_system = platform.system() if system is None else system
    detected_machine = (
        platform.machine().lower() if machine is None else machine.lower()
    )
    if detected_system != artifact.system or detected_machine not in artifact.machines:
        return OpenCodeStatus(
            state="unsupported",
            ready=False,
            version=artifact.version,
            home=os.fspath(root),
            binary=os.fspath(binary),
            reason=(
                "OpenCode is pinned only for Linux x86-64; detected "
                f"{detected_system} {detected_machine}"
            ),
        )
    if not os.path.lexists(root):
        return OpenCodeStatus(
            state="missing",
            ready=False,
            version=artifact.version,
            home=os.fspath(root),
            binary=os.fspath(binary),
            reason="OpenCode home does not exist",
        )
    try:
        _assert_no_symlink_components(root)
        _private_directory(root)
        versions = root / "versions"
        _private_directory(versions)
        version_dir = opencode_version_dir(root, artifact=artifact)
        _private_directory(version_dir)
        binary_metadata = _regular_owner_file(binary, executable=True)
        receipt = _read_receipt(version_dir / INSTALL_RECEIPT_NAME, artifact)
        if (
            binary_metadata.st_size != artifact.executable_size
            or binary_metadata.st_size != receipt["binary_size"]
        ):
            raise OpenCodeInstallError("OpenCode binary size does not match the pin")
        actual_digest = _sha256_file(binary)
        if not hmac.compare_digest(actual_digest, artifact.executable_sha256):
            raise OpenCodeInstallError("OpenCode binary digest does not match the pin")
        observed_version = version_probe(binary)
        if observed_version != artifact.version:
            raise OpenCodeInstallError(
                "OpenCode binary version does not match the pinned version"
            )
    except (OpenCodeInstallError, OSError) as exc:
        return OpenCodeStatus(
            state="invalid",
            ready=False,
            version=artifact.version,
            home=os.fspath(root),
            binary=os.fspath(binary),
            reason=str(exc),
        )
    return OpenCodeStatus(
        state="ready",
        ready=True,
        version=artifact.version,
        home=os.fspath(root),
        binary=os.fspath(binary),
        reason="Pinned OpenCode binary is ready",
        binary_sha256=actual_digest,
        observed_version=observed_version,
    )


def opencode_status(
    home: str | os.PathLike[str] | None = None,
    *,
    artifact: HarnessArtifact = OPENCODE_ARTIFACT,
    system: str | None = None,
    machine: str | None = None,
    version_probe: Callable[[Path], str] = _probe_version,
) -> dict[str, object]:
    """Return JSON-safe status without creating, repairing, or downloading."""

    return _inspect_opencode_status(
        home,
        artifact=artifact,
        system=system,
        machine=machine,
        version_probe=version_probe,
    ).to_dict()


def get_opencode_status(
    home: str | os.PathLike[str] | None = None,
    *,
    artifact: HarnessArtifact = OPENCODE_ARTIFACT,
) -> dict[str, object]:
    """Compatibility spelling for control-plane callers."""

    return opencode_status(home, artifact=artifact)


def resolve_opencode_binary(
    home: str | os.PathLike[str] | None = None,
    *,
    artifact: HarnessArtifact = OPENCODE_ARTIFACT,
) -> Path:
    """Return only a fully validated pinned executable, never a PATH fallback."""

    status_result = _inspect_opencode_status(home, artifact=artifact)
    if not status_result.ready:
        raise OpenCodeInstallError(status_result.reason)
    return Path(status_result.binary)


def _create_private_directory(path: Path) -> None:
    """Create a private directory tree, then reject symlink or unsafe leaves."""

    _assert_no_symlink_components(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    _assert_no_symlink_components(path)
    _private_directory(path)


def _copy_verified_local_archive(
    source_path: Path,
    destination: Path,
    artifact: HarnessArtifact,
) -> Path:
    """Copy an owner file through no-follow descriptors while hashing it."""

    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    try:
        source_fd = os.open(source_path, source_flags)
    except OSError as exc:
        raise OpenCodeInstallError(
            "OpenCode archive could not be opened safely"
        ) from exc
    destination_fd = -1
    try:
        source_metadata = os.fstat(source_fd)
        if (
            not stat.S_ISREG(source_metadata.st_mode)
            or source_metadata.st_uid != os.geteuid()
            or source_metadata.st_nlink != 1
        ):
            raise OpenCodeInstallError(
                "OpenCode archive must be a regular, singly-linked owner file"
            )
        if source_metadata.st_size != artifact.archive_size:
            raise OpenCodeInstallError("OpenCode archive size does not match the pin")
        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            destination_flags |= os.O_NOFOLLOW
        destination_fd = os.open(destination, destination_flags, 0o600)
        size = 0
        digest = hashlib.sha256()
        with (
            os.fdopen(source_fd, "rb") as source,
            os.fdopen(destination_fd, "wb") as target,
        ):
            source_fd = -1
            destination_fd = -1
            while True:
                chunk = source.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > artifact.archive_size:
                    raise OpenCodeInstallError(
                        "OpenCode archive exceeds its pinned size"
                    )
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if size != artifact.archive_size:
            raise OpenCodeInstallError("OpenCode archive size does not match the pin")
        if not hmac.compare_digest(digest.hexdigest(), artifact.archive_sha256):
            raise OpenCodeInstallError("OpenCode archive digest does not match the pin")
        return destination
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if destination_fd >= 0:
            os.close(destination_fd)


def _download_archive(
    destination: Path,
    artifact: HarnessArtifact,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> Path:
    parsed = urlparse(artifact.url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_DOWNLOAD_HOSTS:
        raise OpenCodeInstallError(
            "OpenCode release URL is not an approved HTTPS origin"
        )
    request = urllib.request.Request(
        artifact.url,
        headers={"User-Agent": "Aeon-OpenCode-Installer/1"},
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(destination, flags, 0o600)
    size = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as target:
            fd = -1
            with opener(request, timeout=60) as response:
                final_url = getattr(response, "url", artifact.url)
                final = urlparse(final_url)
                if (
                    final.scheme != "https"
                    or final.hostname not in _ALLOWED_DOWNLOAD_HOSTS
                ):
                    raise OpenCodeInstallError(
                        "OpenCode download redirected to an unapproved origin"
                    )
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES or size > artifact.archive_size:
                        raise OpenCodeInstallError(
                            "OpenCode archive exceeds its pinned size"
                        )
                    digest.update(chunk)
                    target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    if size != artifact.archive_size:
        raise OpenCodeInstallError("OpenCode archive size does not match the pin")
    if not hmac.compare_digest(digest.hexdigest(), artifact.archive_sha256):
        raise OpenCodeInstallError("OpenCode archive digest does not match the pin")
    return destination


def _archive_binary_member(
    archive: tarfile.TarFile, artifact: HarnessArtifact
) -> tarfile.TarInfo:
    candidates = []
    for member in archive.getmembers():
        normalized = member.name.removeprefix("./")
        if normalized == artifact.executable_name:
            candidates.append(member)
    if len(candidates) != 1:
        raise OpenCodeInstallError(
            "OpenCode archive does not contain exactly one binary"
        )
    member = candidates[0]
    if (
        not member.isreg()
        or member.issym()
        or member.islnk()
        or member.size != artifact.executable_size
    ):
        raise OpenCodeInstallError("OpenCode archive binary member is unsafe")
    return member


def _extract_binary(
    archive_path: Path, destination: Path, artifact: HarnessArtifact
) -> None:
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            member = _archive_binary_member(archive, artifact)
            source = archive.extractfile(member)
            if source is None:
                raise OpenCodeInstallError("OpenCode archive binary cannot be read")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(destination, flags, 0o700)
            try:
                with os.fdopen(fd, "wb") as target:
                    fd = -1
                    shutil.copyfileobj(source, target, DOWNLOAD_CHUNK_BYTES)
                    target.flush()
                    os.fsync(target.fileno())
            finally:
                source.close()
                if fd >= 0:
                    os.close(fd)
    except (OSError, tarfile.TarError) as exc:
        raise OpenCodeInstallError("OpenCode archive could not be extracted") from exc
    _regular_owner_file(destination, executable=True)


def _write_receipt(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            fd = -1
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if fd >= 0:
            os.close(fd)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def install_opencode(
    home: str | os.PathLike[str] | None = None,
    *,
    archive_path: str | os.PathLike[str] | None = None,
    artifact: HarnessArtifact = OPENCODE_ARTIFACT,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
    system: str | None = None,
    machine: str | None = None,
    version_probe: Callable[[Path], str] = _probe_version,
) -> OpenCodeStatus:
    """Install one verified artifact, atomically and without PATH mutation."""

    detected_system = platform.system() if system is None else system
    detected_machine = (
        platform.machine().lower() if machine is None else machine.lower()
    )
    if detected_system != artifact.system or detected_machine not in artifact.machines:
        raise OpenCodeInstallError(
            "OpenCode installation is supported only on Linux x86-64"
        )
    root = resolve_opencode_home(home)
    existing = _inspect_opencode_status(
        root,
        artifact=artifact,
        system=detected_system,
        machine=detected_machine,
        version_probe=version_probe,
    )
    if existing.ready:
        return existing
    target = opencode_version_dir(root, artifact=artifact)
    if os.path.lexists(target):
        raise OpenCodeInstallError(
            "refusing to replace an existing invalid OpenCode version directory"
        )

    _create_private_directory(root)
    versions = root / "versions"
    _create_private_directory(versions)
    stage = Path(tempfile.mkdtemp(prefix=".install-", dir=versions))
    os.chmod(stage, 0o700)
    published = False
    try:
        if archive_path is None:
            archive = _download_archive(
                stage / artifact.archive_name, artifact, opener=opener
            )
        else:
            archive = _copy_verified_local_archive(
                Path(archive_path), stage / artifact.archive_name, artifact
            )
        binary = stage / artifact.executable_name
        _extract_binary(archive, binary, artifact)
        binary_metadata = _regular_owner_file(binary, executable=True)
        if binary_metadata.st_size != artifact.executable_size:
            raise OpenCodeInstallError("OpenCode binary size does not match the pin")
        binary_digest = _sha256_file(binary)
        if not hmac.compare_digest(binary_digest, artifact.executable_sha256):
            raise OpenCodeInstallError("OpenCode binary digest does not match the pin")
        observed_version = version_probe(binary)
        if observed_version != artifact.version:
            raise OpenCodeInstallError(
                "OpenCode binary version does not match the pinned version"
            )
        archive.unlink()
        receipt = {
            "schema_version": INSTALL_RECEIPT_SCHEMA,
            "version": artifact.version,
            "archive_name": artifact.archive_name,
            "archive_sha256": artifact.archive_sha256,
            "source_url": artifact.url,
            "binary_sha256": artifact.executable_sha256,
            "binary_size": artifact.executable_size,
        }
        _write_receipt(stage / INSTALL_RECEIPT_NAME, receipt)
        _fsync_directory(stage)
        try:
            stage.rename(target)
            published = True
        except OSError:
            if not os.path.lexists(target):
                raise
            winner = _inspect_opencode_status(
                root,
                artifact=artifact,
                system=detected_system,
                machine=detected_machine,
                version_probe=version_probe,
            )
            if winner.ready:
                return winner
            raise OpenCodeInstallError(
                "a concurrent OpenCode install published an invalid directory"
            )
        _fsync_directory(versions)
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)

    result = _inspect_opencode_status(
        root,
        artifact=artifact,
        system=detected_system,
        machine=detected_machine,
        version_probe=version_probe,
    )
    if not result.ready:
        raise OpenCodeInstallError(result.reason)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aeon.harnesses.opencode_install",
        description="Inspect or install Aeon's pinned OpenCode binary.",
    )
    parser.add_argument(
        "command",
        choices=("status", "install"),
        help="read status or install the pinned release",
    )
    parser.add_argument(
        "--home",
        help=f"installation root (default: ${OPENCODE_HOME_ENV} or {DEFAULT_OPENCODE_HOME})",
    )
    parser.add_argument(
        "--archive",
        help="use an already-downloaded archive; its size and SHA-256 are still required",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "status" and args.archive:
        parser.error("--archive is valid only with install")
    try:
        result = (
            _inspect_opencode_status(args.home)
            if args.command == "status"
            else install_opencode(args.home, archive_path=args.archive)
        )
    except OpenCodeInstallError as exc:
        if args.json:
            print(json.dumps({"ready": False, "state": "error", "reason": str(exc)}))
        else:
            print(f"OpenCode {args.command} failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result.to_dict(), sort_keys=True))
    else:
        print(f"OpenCode {result.state}: {result.reason}")
        print(result.binary)
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
