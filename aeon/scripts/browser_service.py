#!/usr/bin/python3
"""Build and provision Aeon's exact, CPU-only browser host service.

This module is an operator boundary, never a model-facing tool.  It never lists
containers, guesses a fixed name, adopts a pre-existing runtime, or removes a
container.  A build records its exact image ID and source digest.  Provisioning
then creates one randomly named container and persists the returned container ID
before it can be started.  All later lifecycle operations use only that ID.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import secrets
import stat
import subprocess
import tempfile
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from aeon.tools.command_fleet_guard import (
    require_fleet_low_priority_wrapper,
    scrubbed_fleet_command_environment,
)


DOCKER = Path("/usr/bin/docker")
DOCKER_HOST = "unix:///var/run/docker.sock"
IMAGE_TAG = "aeon_browser_service:latest"
PORT = 8030
CONTAINER_PORT = 8030
API_VERSION = "human_v6"
AUTH_VERSION = "required-v1"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "services" / "browser"
SOURCE_FILES = (
    ".dockerignore",
    "Dockerfile",
    "browser_util.py",
    "entrypoint.sh",
    "human_motion.py",
    "media_safety.py",
    "requirements.txt",
    "server.py",
)

_OWNER = pwd.getpwuid(os.geteuid())
OWNER_HOME = Path(_OWNER.pw_dir)
AEON_ROOT = OWNER_HOME / ".aeon"
STATE_ROOT = AEON_ROOT / "host-services" / "browser"
IMAGE_RELEASE_ROOT = STATE_ROOT / "image-releases"
RETIRED_SERVICE_ROOT = STATE_ROOT / "retired-services"
CURRENT_IMAGE_PATH = STATE_ROOT / "current-image.json"
SERVICE_RECEIPT_PATH = STATE_ROOT / "service.json"
CREATE_INTENT_PATH = STATE_ROOT / "create-intent.json"
PENDING_CID_PATH = STATE_ROOT / "container.cid.pending"
LOCK_PATH = STATE_ROOT / "launch.lock"
TOKEN_PATH = AEON_ROOT / "browser_api_token"
PROFILE_ROOT = AEON_ROOT / "browser_profiles"
TOKEN_CONTAINER_PATH = "/run/secrets/aeon_browser_token"
XDG_RUNTIME_CONTAINER_PATH = "/tmp/aeon-runtime"

IMAGE_RECEIPT_SCHEMA = 1
SERVICE_RECEIPT_SCHEMA = 1
INTENT_SCHEMA = 1
SERVICE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MEMORY_BYTES = 6 * 1024 * 1024 * 1024
NANO_CPUS = 3_000_000_000
PIDS_LIMIT = 512
SHM_BYTES = 2 * 1024 * 1024 * 1024
CPU_SHARES = 2
BLKIO_WEIGHT = 10
OOM_SCORE_ADJ = 1000
STARTUP_TIMEOUT_SECONDS = 90

_NO_ACCELERATOR_ENV = {
    "CUDA_VISIBLE_DEVICES": "void",
    "GPU_DEVICE_ORDINAL": "-1",
    "HIP_VISIBLE_DEVICES": "-1",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "ROCR_VISIBLE_DEVICES": "-1",
}


class BrowserServiceError(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _secure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise BrowserServiceError(f"unsafe browser service directory: {path}")


def _secure_file(path: Path, *, expected_mode: int = 0o600) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != expected_mode
        or metadata.st_nlink != 1
    ):
        raise BrowserServiceError(f"unsafe browser service file: {path}")


def _atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".tmp-browser-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_json(path: Path, *, label: str) -> dict | None:
    if not (path.exists() or path.is_symlink()):
        return None
    _secure_file(path)
    try:
        if path.stat().st_size > 32 * 1024:
            raise BrowserServiceError(f"{label} is oversized")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BrowserServiceError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise BrowserServiceError(f"{label} is malformed")
    return value


def source_digest() -> str:
    """Hash exactly the files copied into, or controlling, the browser image."""
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        path = SOURCE_ROOT / relative
        try:
            metadata = path.lstat()
            body = path.read_bytes()
        except OSError as exc:
            raise BrowserServiceError(
                f"browser image source is unavailable: {relative}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise BrowserServiceError(
                f"browser image source must be a regular file: {relative}"
            )
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


def _docker_environment() -> dict[str, str]:
    environment = scrubbed_fleet_command_environment()
    for name in tuple(environment):
        upper = name.upper()
        if upper.startswith("DOCKER_") or upper in {
            "CONTAINER_HOST",
            "BUILDAH_HOST",
            "KUBECONFIG",
        }:
            environment.pop(name, None)
    environment["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    return environment


def _docker(
    arguments: list[str],
    *,
    timeout: int | None = 60,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        metadata = DOCKER.stat()
    except OSError as exc:
        raise BrowserServiceError("root-owned Docker client is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & 0o022
        or not os.access(DOCKER, os.X_OK)
    ):
        raise BrowserServiceError("root-owned Docker client failed identity checks")
    command = [
        require_fleet_low_priority_wrapper(),
        str(DOCKER),
        "--host",
        DOCKER_HOST,
        *arguments,
    ]
    return subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=_docker_environment(),
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _parse_single_inspection(result: subprocess.CompletedProcess[str], label: str) -> dict:
    if result.returncode != 0:
        raise BrowserServiceError(f"exact receipted {label} is unavailable")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BrowserServiceError(f"Docker returned malformed {label} evidence") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise BrowserServiceError(f"Docker returned ambiguous {label} evidence")
    return value[0]


def _inspect_image(image_id: str) -> dict:
    if IMAGE_ID_RE.fullmatch(image_id) is None:
        raise BrowserServiceError("browser image ID is invalid")
    return _parse_single_inspection(
        _docker(["image", "inspect", image_id]), "browser image"
    )


def _validate_image(document: dict, *, image_id: str, source_sha256: str) -> None:
    labels = (document.get("Config") or {}).get("Labels") or {}
    if not (
        document.get("Id") == image_id
        and labels.get("com.bc_aeon.browser.auth") == AUTH_VERSION
        and labels.get("com.bc_aeon.browser.api") == API_VERSION.replace("_", "-")
        and labels.get("com.bc_aeon.browser.source-sha256") == source_sha256
    ):
        raise BrowserServiceError(
            "browser image identity or immutable source labels do not match"
        )


def _image_release(image_id: str, source_sha256: str) -> dict:
    return {
        "schema": IMAGE_RECEIPT_SCHEMA,
        "image_id": image_id,
        "source_sha256": source_sha256,
        "auth_version": AUTH_VERSION,
        "api_version": API_VERSION,
    }


def _validate_image_release(value: dict, *, current_source: bool) -> dict:
    if not (
        value.get("schema") == IMAGE_RECEIPT_SCHEMA
        and IMAGE_ID_RE.fullmatch(str(value.get("image_id", "")))
        and SHA256_RE.fullmatch(str(value.get("source_sha256", "")))
        and value.get("auth_version") == AUTH_VERSION
        and value.get("api_version") == API_VERSION
    ):
        raise BrowserServiceError("browser image release receipt is invalid")
    if current_source and value["source_sha256"] != source_digest():
        raise BrowserServiceError(
            "browser source changed after the current image release was recorded; "
            "rebuild it before provisioning"
        )
    return value


def _load_current_image_release() -> dict:
    value = _load_json(CURRENT_IMAGE_PATH, label="current browser image receipt")
    if value is None:
        raise BrowserServiceError(
            "no exact browser image release is installed; run the operator setup first"
        )
    value = _validate_image_release(value, current_source=True)
    release_path = IMAGE_RELEASE_ROOT / f"{value['image_id'].removeprefix('sha256:')}.json"
    immutable = _load_json(release_path, label="immutable browser image receipt")
    if immutable != value:
        raise BrowserServiceError("current browser image receipt is not release-bound")
    return value


def record_image_release(image_id: str, expected_source_sha256: str) -> dict:
    if expected_source_sha256 != source_digest():
        raise BrowserServiceError("browser source changed during its image build")
    _secure_directory(STATE_ROOT)
    _secure_directory(IMAGE_RELEASE_ROOT)
    document = _inspect_image(image_id)
    _validate_image(
        document, image_id=image_id, source_sha256=expected_source_sha256
    )
    release = _image_release(image_id, expected_source_sha256)
    release_path = IMAGE_RELEASE_ROOT / f"{image_id.removeprefix('sha256:')}.json"
    existing = _load_json(release_path, label="immutable browser image receipt")
    if existing is None:
        _atomic_json(release_path, release)
        _secure_file(release_path)
    elif existing != release:
        raise BrowserServiceError("immutable browser image receipt conflicts")
    _atomic_json(CURRENT_IMAGE_PATH, release)
    _secure_file(CURRENT_IMAGE_PATH)
    return release


def build_image(*, no_cache: bool = False) -> dict:
    """Build through the verified owner-work wrapper and record its exact ID."""
    _secure_directory(STATE_ROOT)
    _secure_directory(IMAGE_RELEASE_ROOT)
    source_sha256 = source_digest()
    descriptor, iid_name = tempfile.mkstemp(prefix=".browser-image-", dir=STATE_ROOT)
    os.close(descriptor)
    iid_path = Path(iid_name)
    try:
        arguments = [
            "image",
            "build",
            "--network=host",
            "--build-arg",
            f"AEON_BROWSER_SOURCE_SHA256={source_sha256}",
            "--iidfile",
            str(iid_path),
            "--tag",
            IMAGE_TAG,
            "--file",
            str(SOURCE_ROOT / "Dockerfile"),
        ]
        if no_cache:
            arguments.append("--no-cache")
        arguments.append(str(SOURCE_ROOT))
        result = _docker(arguments, timeout=None, capture=False)
        if result.returncode != 0:
            raise BrowserServiceError("bounded browser image build failed")
        try:
            image_id = iid_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise BrowserServiceError("browser image build ID is unavailable") from exc
        if IMAGE_ID_RE.fullmatch(image_id) is None:
            raise BrowserServiceError("browser image build returned an invalid exact ID")
        return record_image_release(image_id, source_sha256)
    finally:
        try:
            iid_path.unlink()
        except OSError:
            pass


def _ensure_token() -> str:
    _secure_directory(AEON_ROOT)
    if TOKEN_PATH.exists() or TOKEN_PATH.is_symlink():
        _secure_file(TOKEN_PATH)
    else:
        try:
            descriptor = os.open(
                TOKEN_PATH,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
        except FileExistsError:
            _secure_file(TOKEN_PATH)
        else:
            token = secrets.token_urlsafe(48)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(token + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            _secure_file(TOKEN_PATH)
    try:
        if TOKEN_PATH.stat().st_size > 4096:
            raise BrowserServiceError("browser API token is oversized")
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise BrowserServiceError("browser API token is unreadable") from exc
    if (
        len(token.encode("utf-8")) < 32
        or len(token) > 256
        or re.fullmatch(r"[A-Za-z0-9_-]+", token) is None
    ):
        raise BrowserServiceError("browser API token is invalid")
    return token


def _service_receipt(value: dict) -> dict:
    if not (
        value.get("schema") == SERVICE_RECEIPT_SCHEMA
        and SERVICE_ID_RE.fullmatch(str(value.get("service_id", "")))
        and CONTAINER_ID_RE.fullmatch(str(value.get("container_id", "")))
        and IMAGE_ID_RE.fullmatch(str(value.get("image_id", "")))
        and SHA256_RE.fullmatch(str(value.get("source_sha256", "")))
        and value.get("auth_version") == AUTH_VERSION
        and value.get("api_version") == API_VERSION
    ):
        raise BrowserServiceError("browser service receipt identity is invalid")
    if value.get("container_name") != f"aeon-browser-{value['service_id']}":
        raise BrowserServiceError("browser service receipt name is invalid")
    return value


def _load_service_receipt() -> dict | None:
    value = _load_json(SERVICE_RECEIPT_PATH, label="browser service receipt")
    return None if value is None else _service_receipt(value)


def _intent(value: dict) -> dict:
    if value.get("schema") != INTENT_SCHEMA:
        raise BrowserServiceError("browser create intent is invalid")
    candidate = dict(value)
    candidate["schema"] = SERVICE_RECEIPT_SCHEMA
    candidate["container_id"] = "0" * 64
    _service_receipt(candidate)
    candidate.pop("container_id")
    candidate["schema"] = INTENT_SCHEMA
    return candidate


def _load_intent() -> dict | None:
    value = _load_json(CREATE_INTENT_PATH, label="browser create intent")
    return None if value is None else _intent(value)


def _inspect_container(container_id: str) -> dict:
    if CONTAINER_ID_RE.fullmatch(container_id) is None:
        raise BrowserServiceError("receipted browser container ID is invalid")
    return _parse_single_inspection(
        _docker(["container", "inspect", container_id]), "browser container"
    )


def _env_map(entries: object) -> dict[str, str]:
    result: dict[str, str] = {}
    if not isinstance(entries, list):
        return result
    for entry in entries:
        if not isinstance(entry, str) or "=" not in entry:
            continue
        name, value = entry.split("=", 1)
        result[name] = value
    return result


def _exact_tmpfs(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"/run", "/tmp"}:
        return False
    expected = {
        "/run": {"rw", "nosuid", "nodev", "noexec", "size=67108864", "mode=755"},
        "/tmp": {"rw", "nosuid", "nodev", "size=1073741824", "mode=1777"},
    }
    return all(
        isinstance(value[path], str)
        and set(value[path].split(",")) == options
        for path, options in expected.items()
    )


def _validate_container_contract(
    document: dict,
    receipt: dict,
    *,
    cuda_visible_devices: str,
) -> bool:
    if cuda_visible_devices not in {"-1", "void"}:
        raise BrowserServiceError("unsupported browser CUDA visibility contract")
    config = document.get("Config") or {}
    host = document.get("HostConfig") or {}
    state = document.get("State") or {}
    labels = config.get("Labels") or {}
    ports = host.get("PortBindings") or {}
    mounts = document.get("Mounts") or []
    environment = _env_map(config.get("Env"))
    expected_name = "/" + receipt["container_name"]
    exact_port = (
        set(ports) == {f"{CONTAINER_PORT}/tcp"}
        and len(ports[f"{CONTAINER_PORT}/tcp"] or []) == 1
        and ports[f"{CONTAINER_PORT}/tcp"][0].get("HostIp") == "127.0.0.1"
        and ports[f"{CONTAINER_PORT}/tcp"][0].get("HostPort") == str(PORT)
        and host.get("PublishAllPorts") is False
    )
    expected_mounts = {
        (str(PROFILE_ROOT), "/profiles", True),
        (str(TOKEN_PATH), TOKEN_CONTAINER_PATH, False),
    }
    actual_mounts = {
        (str(item.get("Source", "")), str(item.get("Destination", "")), bool(item.get("RW")))
        for item in mounts
        if isinstance(item, dict)
    }
    exact_limits = (
        host.get("Memory") == MEMORY_BYTES
        and host.get("MemorySwap") == MEMORY_BYTES
        and host.get("NanoCpus") == NANO_CPUS
        and host.get("PidsLimit") == PIDS_LIMIT
        and host.get("ShmSize") == SHM_BYTES
        and host.get("CpuShares") == CPU_SHARES
        and host.get("BlkioWeight") == BLKIO_WEIGHT
        and host.get("OomScoreAdj") == OOM_SCORE_ADJ
    )
    exact_isolation = (
        host.get("Privileged") is False
        and not (host.get("Devices") or host.get("DeviceRequests"))
        and set(host.get("CapDrop") or []) == {"ALL"}
        and not (host.get("CapAdd") or [])
        and "no-new-privileges" in (host.get("SecurityOpt") or [])
        and host.get("ReadonlyRootfs") is True
        and host.get("NetworkMode") == "bridge"
        and host.get("PidMode") in {"", None}
        and host.get("IpcMode") in {"private", ""}
        and host.get("AutoRemove") is False
        and host.get("Init") is True
        and (host.get("RestartPolicy") or {}).get("Name") == "unless-stopped"
        and (host.get("LogConfig") or {}).get("Type") == "json-file"
        and (host.get("LogConfig") or {}).get("Config")
        == {"max-file": "2", "max-size": "10m"}
    )
    exact_tmpfs = _exact_tmpfs(host.get("Tmpfs"))
    required_env = {
        "PORT": str(CONTAINER_PORT),
        "AEON_BROWSER_PROFILE": "/profiles/default",
        "AEON_BROWSER_PROFILE_ROOT": "/profiles",
        "AEON_BROWSER_TOKEN_FILE": TOKEN_CONTAINER_PATH,
        "AEON_BROWSER_SERVICE_ID": receipt["service_id"],
        "HOME": "/profiles/.browser-home",
        "XDG_CACHE_HOME": "/profiles/.browser-home/.cache",
        "XDG_RUNTIME_DIR": XDG_RUNTIME_CONTAINER_PATH,
        "PYTHONDONTWRITEBYTECODE": "1",
        **_NO_ACCELERATOR_ENV,
    }
    required_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    exact_env = all(environment.get(name) == value for name, value in required_env.items())
    forbidden_gpu_env = {
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_REQUIRE_CUDA",
        "GPU_AGENT_CLAIM_ID",
        "GPU_MEM_LIMIT_GB",
    }
    exact_env = exact_env and not forbidden_gpu_env.intersection(environment)
    forbidden_network_env = {
        "AEON_BROWSER_PROXY",
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
    }
    exact_env = exact_env and not forbidden_network_env.intersection(
        {name.upper() for name in environment}
    )
    if not (
        document.get("Id") == receipt["container_id"]
        and document.get("Name") == expected_name
        and document.get("Image") == receipt["image_id"]
        and config.get("Image") == receipt["image_id"]
        and config.get("User") == f"{os.geteuid()}:{os.getegid()}"
        and labels.get("owner") == _OWNER.pw_name
        and labels.get("com.bc_aeon.component") == "browser"
        and labels.get("com.bc_aeon.service-id") == receipt["service_id"]
        and labels.get("com.bc_aeon.browser.auth") == AUTH_VERSION
        and labels.get("com.bc_aeon.browser.api") == API_VERSION.replace("_", "-")
        and labels.get("com.bc_aeon.browser.source-sha256") == receipt["source_sha256"]
        and exact_port
        and actual_mounts == expected_mounts
        and exact_limits
        and exact_isolation
        and exact_tmpfs
        and exact_env
    ):
        raise BrowserServiceError("receipted browser container configuration does not match")
    return state.get("Running") is True


def _validate_container(document: dict, receipt: dict) -> bool:
    return _validate_container_contract(
        document,
        receipt,
        cuda_visible_devices="void",
    )


def _validate_legacy_cuda_container(document: dict, receipt: dict) -> bool:
    entries = (document.get("Config") or {}).get("Env")
    if not isinstance(entries, list):
        raise BrowserServiceError("legacy browser environment evidence is malformed")
    cuda_entries = [
        entry
        for entry in entries
        if isinstance(entry, str) and entry.startswith("CUDA_VISIBLE_DEVICES=")
    ]
    if cuda_entries != ["CUDA_VISIBLE_DEVICES=-1"]:
        raise BrowserServiceError(
            "browser container is not the exact legacy CUDA-sentinel configuration"
        )
    return _validate_container_contract(
        document,
        receipt,
        cuda_visible_devices="-1",
    )


def _read_pending_container_id() -> str:
    if not (PENDING_CID_PATH.exists() or PENDING_CID_PATH.is_symlink()):
        raise BrowserServiceError(
            "browser create intent exists without an exact container-ID receipt; "
            "refusing to guess whether Docker created anything"
        )
    metadata = PENDING_CID_PATH.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_nlink != 1:
        raise BrowserServiceError("pending browser container-ID file is unsafe")
    os.chmod(PENDING_CID_PATH, 0o600)
    _secure_file(PENDING_CID_PATH)
    try:
        container_id = PENDING_CID_PATH.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise BrowserServiceError("pending browser container ID is unreadable") from exc
    if CONTAINER_ID_RE.fullmatch(container_id) is None:
        raise BrowserServiceError("pending browser container ID is invalid")
    return container_id


def _create_arguments(intent: dict) -> list[str]:
    return [
        "container",
        "create",
        "--cidfile",
        str(PENDING_CID_PATH),
        "--name",
        intent["container_name"],
        "--label",
        f"owner={_OWNER.pw_name}",
        "--label",
        "com.bc_aeon.component=browser",
        "--label",
        f"com.bc_aeon.service-id={intent['service_id']}",
        "--label",
        f"com.bc_aeon.browser.auth={AUTH_VERSION}",
        "--label",
        f"com.bc_aeon.browser.api={API_VERSION.replace('_', '-')}",
        "--label",
        f"com.bc_aeon.browser.source-sha256={intent['source_sha256']}",
        "--restart",
        "unless-stopped",
        "--publish",
        f"127.0.0.1:{PORT}:{CONTAINER_PORT}",
        "--mount",
        f"type=bind,src={PROFILE_ROOT},dst=/profiles",
        "--mount",
        f"type=bind,src={TOKEN_PATH},dst={TOKEN_CONTAINER_PATH},readonly",
        "--user",
        f"{os.geteuid()}:{os.getegid()}",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=1073741824,mode=1777",
        "--tmpfs",
        "/run:rw,nosuid,nodev,noexec,size=67108864,mode=755",
        "--memory",
        str(MEMORY_BYTES),
        "--memory-swap",
        str(MEMORY_BYTES),
        "--cpus",
        "3.0",
        "--cpu-shares",
        str(CPU_SHARES),
        "--blkio-weight",
        str(BLKIO_WEIGHT),
        "--oom-score-adj",
        str(OOM_SCORE_ADJ),
        "--pids-limit",
        str(PIDS_LIMIT),
        "--shm-size",
        str(SHM_BYTES),
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--network",
        "bridge",
        "--init",
        "--log-driver",
        "json-file",
        "--log-opt",
        "max-size=10m",
        "--log-opt",
        "max-file=2",
        "--env",
        f"PORT={CONTAINER_PORT}",
        "--env",
        "AEON_BROWSER_PROFILE=/profiles/default",
        "--env",
        "AEON_BROWSER_PROFILE_ROOT=/profiles",
        "--env",
        f"AEON_BROWSER_TOKEN_FILE={TOKEN_CONTAINER_PATH}",
        "--env",
        f"AEON_BROWSER_SERVICE_ID={intent['service_id']}",
        "--env",
        "HOME=/profiles/.browser-home",
        "--env",
        "XDG_CACHE_HOME=/profiles/.browser-home/.cache",
        "--env",
        f"XDG_RUNTIME_DIR={XDG_RUNTIME_CONTAINER_PATH}",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
    ] + [item for name, value in _NO_ACCELERATOR_ENV.items() for item in ("--env", f"{name}={value}")] + [intent["image_id"]]


def _new_intent(release: dict) -> dict:
    service_id = secrets.token_hex(16)
    return {
        "schema": INTENT_SCHEMA,
        "service_id": service_id,
        "container_name": f"aeon-browser-{service_id}",
        "image_id": release["image_id"],
        "source_sha256": release["source_sha256"],
        "auth_version": AUTH_VERSION,
        "api_version": API_VERSION,
    }


def _recover_or_create_receipt() -> dict:
    intent = _load_intent()
    if intent is None:
        if PENDING_CID_PATH.exists() or PENDING_CID_PATH.is_symlink():
            raise BrowserServiceError(
                "orphaned pending browser container-ID file exists without an intent"
            )
        intent = _new_intent(_load_current_image_release())
        _atomic_json(CREATE_INTENT_PATH, intent)
        _secure_file(CREATE_INTENT_PATH)
        result = _docker(_create_arguments(intent), timeout=120)
        if result.returncode != 0:
            raise BrowserServiceError(
                "Docker refused the bounded browser container create; the exact "
                "intent was retained for operator recovery"
            )
    container_id = _read_pending_container_id()
    receipt = dict(intent)
    receipt["schema"] = SERVICE_RECEIPT_SCHEMA
    receipt["container_id"] = container_id
    receipt = _service_receipt(receipt)
    _atomic_json(SERVICE_RECEIPT_PATH, receipt)
    _secure_file(SERVICE_RECEIPT_PATH)
    document = _inspect_container(container_id)
    if _validate_container(document, receipt):
        raise BrowserServiceError("new browser container started before receipt commit")
    CREATE_INTENT_PATH.unlink()
    PENDING_CID_PATH.unlink()
    return receipt


def _settle_committed_create(receipt: dict) -> None:
    """Finish the safe tail of a create transaction after a local crash."""
    has_intent = CREATE_INTENT_PATH.exists() or CREATE_INTENT_PATH.is_symlink()
    has_pending_id = PENDING_CID_PATH.exists() or PENDING_CID_PATH.is_symlink()
    if not has_intent and not has_pending_id:
        return
    if not has_intent or not has_pending_id:
        raise BrowserServiceError(
            "committed browser receipt has incomplete create-transaction evidence"
        )
    intent = _load_intent()
    container_id = _read_pending_container_id()
    recovered = dict(intent or {})
    recovered["schema"] = SERVICE_RECEIPT_SCHEMA
    recovered["container_id"] = container_id
    if _service_receipt(recovered) != receipt:
        raise BrowserServiceError(
            "committed browser receipt conflicts with create-transaction evidence"
        )
    document = _inspect_container(container_id)
    if _validate_container(document, receipt):
        raise BrowserServiceError(
            "receipted browser container started before create recovery completed"
        )
    CREATE_INTENT_PATH.unlink()
    PENDING_CID_PATH.unlink()


def _validate_retirement_identity(document: dict, receipt: dict) -> None:
    """Prove a stopped receipt still names the exact Aeon-owned container.

    Retirement deliberately accepts an older reviewed resource configuration:
    its purpose is to preserve that immutable receipt while allowing a fixed
    replacement to be provisioned.  It therefore validates ownership and exact
    identity, mounts, loopback port, and absence of device passthrough, but does
    not claim the retired configuration still satisfies the current release.
    """
    config = document.get("Config") or {}
    host = document.get("HostConfig") or {}
    labels = config.get("Labels") or {}
    state = document.get("State") or {}
    ports = host.get("PortBindings") or {}
    bindings = ports.get(f"{CONTAINER_PORT}/tcp") or []
    mounts = document.get("Mounts") or []
    expected_mounts = {
        (str(PROFILE_ROOT), "/profiles", True),
        (str(TOKEN_PATH), TOKEN_CONTAINER_PATH, False),
    }
    actual_mounts = {
        (str(item.get("Source", "")), str(item.get("Destination", "")), bool(item.get("RW")))
        for item in mounts
        if isinstance(item, dict)
    }
    if not (
        document.get("Id") == receipt["container_id"]
        and document.get("Name") == "/" + receipt["container_name"]
        and document.get("Image") == receipt["image_id"]
        and config.get("Image") == receipt["image_id"]
        and labels.get("owner") == _OWNER.pw_name
        and labels.get("com.bc_aeon.component") == "browser"
        and labels.get("com.bc_aeon.service-id") == receipt["service_id"]
        and labels.get("com.bc_aeon.browser.auth") == AUTH_VERSION
        and labels.get("com.bc_aeon.browser.api") == API_VERSION.replace("_", "-")
        and labels.get("com.bc_aeon.browser.source-sha256") == receipt["source_sha256"]
        and set(ports) == {f"{CONTAINER_PORT}/tcp"}
        and len(bindings) == 1
        and bindings[0].get("HostIp") == "127.0.0.1"
        and bindings[0].get("HostPort") == str(PORT)
        and actual_mounts == expected_mounts
        and not (host.get("Devices") or host.get("DeviceRequests"))
        and state.get("Running") is False
    ):
        raise BrowserServiceError(
            "stopped browser container retirement identity does not match"
        )


def _retire_stopped_service_locked(*, require_legacy_cuda: bool = False) -> dict:
    if CREATE_INTENT_PATH.exists() or PENDING_CID_PATH.exists():
        raise BrowserServiceError(
            "cannot retire a browser service with an unsettled create transaction"
        )
    receipt = _load_service_receipt()
    if receipt is None:
        raise BrowserServiceError("no current browser service receipt exists")
    document = _inspect_container(receipt["container_id"])
    if require_legacy_cuda:
        if _validate_legacy_cuda_container(document, receipt):
            raise BrowserServiceError(
                "legacy browser container is still running during retirement"
            )
    else:
        _validate_retirement_identity(document, receipt)
    destination = RETIRED_SERVICE_ROOT / f"{receipt['service_id']}.json"
    if destination.exists() or destination.is_symlink():
        raise BrowserServiceError("retired browser receipt already exists")
    os.rename(SERVICE_RECEIPT_PATH, destination)
    _secure_file(destination)
    return receipt


def retire_stopped_service() -> dict:
    """Atomically archive one exact stopped receipt without deleting its container."""
    _secure_directory(STATE_ROOT)
    _secure_directory(RETIRED_SERVICE_ROOT)
    descriptor = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return _retire_stopped_service_locked()
    finally:
        os.close(descriptor)


def _healthy(service_id: str, token: str) -> bool:
    if SERVICE_ID_RE.fullmatch(service_id) is None:
        return False
    request = Request(
        f"http://127.0.0.1:{PORT}/health",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=2) as response:
            body = response.read(8193)
            content_type = response.headers.get("content-type", "")
            status = response.status
        if len(body) > 8192 or not content_type.startswith("application/json"):
            return False
        value = json.loads(body)
        return (
            status == 200
            and isinstance(value, dict)
            and value.get("status") == "ok"
            and value.get("auth_required") is True
            and value.get("api_version") == API_VERSION
            and value.get("service_id") == service_id
        )
    except Exception:
        return False


def _ensure_service_locked(token: str) -> dict:
    receipt = _load_service_receipt()
    if receipt is None:
        receipt = _recover_or_create_receipt()
        running = False
    else:
        release = _load_current_image_release()
        if not (
            receipt["image_id"] == release["image_id"]
            and receipt["source_sha256"] == release["source_sha256"]
        ):
            raise BrowserServiceError(
                "browser service receipt is not bound to the current image release; "
                "stop it exactly, retire its receipt, and provision the replacement"
            )
        _settle_committed_create(receipt)
        document = _inspect_container(receipt["container_id"])
        running = _validate_container(document, receipt)
        if running and _healthy(receipt["service_id"], token):
            return receipt
    operation = "restart" if running else "start"
    result = _docker(
        ["container", operation, receipt["container_id"]], timeout=120
    )
    if result.returncode != 0:
        raise BrowserServiceError(
            f"Docker refused the exact receipted browser container {operation}"
        )
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _healthy(receipt["service_id"], token):
            return receipt
        time.sleep(1)
    raise BrowserServiceError(
        "exact receipted browser service did not become semantically healthy"
    )


def _prepare_service_state() -> str:
    _secure_directory(AEON_ROOT)
    _secure_directory(STATE_ROOT)
    _secure_directory(IMAGE_RELEASE_ROOT)
    _secure_directory(PROFILE_ROOT)
    _secure_directory(PROFILE_ROOT / ".browser-home")
    _secure_directory(PROFILE_ROOT / ".browser-home" / ".cache")
    return _ensure_token()


def ensure_service() -> dict:
    token = _prepare_service_state()
    descriptor = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return _ensure_service_locked(token)
    finally:
        os.close(descriptor)


def migrate_legacy_cuda_sentinel() -> tuple[dict, dict]:
    """Replace only the exact running ``CUDA_VISIBLE_DEVICES=-1`` service.

    The retired container is stopped and retained. No container discovery,
    deletion, adoption, or name-based lifecycle operation is permitted.
    """

    token = _prepare_service_state()
    _secure_directory(RETIRED_SERVICE_ROOT)
    # Prove a replacement image for the current source exists before downtime.
    _load_current_image_release()
    descriptor = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if CREATE_INTENT_PATH.exists() or PENDING_CID_PATH.exists():
            raise BrowserServiceError(
                "cannot migrate a browser service with an unsettled create transaction"
            )
        receipt = _load_service_receipt()
        if receipt is None:
            raise BrowserServiceError("no current browser service receipt exists")
        document = _inspect_container(receipt["container_id"])
        if not _validate_legacy_cuda_container(document, receipt):
            raise BrowserServiceError(
                "legacy browser CUDA-sentinel migration requires a running container"
            )
        result = _docker(
            ["container", "stop", "--time", "30", receipt["container_id"]],
            timeout=60,
        )
        if result.returncode != 0:
            raise BrowserServiceError(
                "Docker refused to stop the exact legacy browser container"
            )
        stopped = _inspect_container(receipt["container_id"])
        if _validate_legacy_cuda_container(stopped, receipt):
            raise BrowserServiceError(
                "exact legacy browser container remained running after stop"
            )
        retired = _retire_stopped_service_locked(require_legacy_cuda=True)
        replacement = _ensure_service_locked(token)
        if replacement["container_id"] == retired["container_id"]:
            raise BrowserServiceError("browser migration reused the retired container")
        return retired, replacement
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    build = subparsers.add_parser("build-image", help="build and receipt exact image")
    build.add_argument("--no-cache", action="store_true")
    subparsers.add_parser("source-digest", help="print deterministic source digest")
    subparsers.add_parser("ensure", help="ensure exact receipted CPU service")
    subparsers.add_parser(
        "retire-stopped",
        help="archive an exact stopped service receipt without deleting its container",
    )
    subparsers.add_parser(
        "migrate-legacy-cuda-sentinel",
        help="replace only the exact running legacy CUDA -1 service",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    command = arguments.command or "ensure"
    try:
        if command == "source-digest":
            print(source_digest())
        elif command == "build-image":
            release = build_image(no_cache=arguments.no_cache)
            print(
                "Browser image release recorded: "
                f"source={release['source_sha256']} image={release['image_id']}"
            )
        elif command == "ensure":
            ensure_service()
            print(f"Browser service ready at http://127.0.0.1:{PORT}")
        elif command == "retire-stopped":
            receipt = retire_stopped_service()
            print(
                "Stopped browser service receipt retired: "
                f"service={receipt['service_id']}"
            )
        elif command == "migrate-legacy-cuda-sentinel":
            retired, replacement = migrate_legacy_cuda_sentinel()
            print(
                "Legacy browser CUDA sentinel migrated: "
                f"retired={retired['service_id']} "
                f"replacement={replacement['service_id']}"
            )
        else:
            raise BrowserServiceError("unsupported browser service operation")
    except Exception as exc:
        print(f"Browser service unavailable: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
