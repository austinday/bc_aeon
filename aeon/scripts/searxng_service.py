#!/usr/bin/python3
"""Provision Aeon's CPU-only SearXNG dependency with exact ownership.

This is an operator helper, never a model-facing tool. It creates at most one
randomly named, resource-capped container and records its immutable ID before
starting it. Existing containers are never listed, guessed by name, replaced,
or removed. Later starts touch only the exact ID in the owner-private receipt.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import tempfile
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from aeon.tools.command_fleet_guard import (
    require_fleet_low_priority_wrapper,
    scrubbed_fleet_command_environment,
)


DOCKER = Path("/usr/bin/docker")
DOCKER_HOST = "unix:///var/run/docker.sock"
IMAGE_REF = (
    "searxng/searxng@sha256:"
    "892cf809341915a4b7710d3c9045005b4c377d51335a089b6d4da0b28750788d"
)
IMAGE_ID = "sha256:892cf809341915a4b7710d3c9045005b4c377d51335a089b6d4da0b28750788d"
PORT = 8095
HEALTH_URL = f"http://127.0.0.1:{PORT}/healthz"
STATE_ROOT = Path.home() / ".aeon" / "host-services" / "searxng"
SETTINGS_PATH = STATE_ROOT / "settings.yml"
RECEIPT_PATH = STATE_ROOT / "service.json"
LOCK_PATH = STATE_ROOT / "launch.lock"
SERVICE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
RECEIPT_SCHEMA = 1
INSTANCE_PREFIX = "Aeon SearXNG "
ANONYMOUS_VOLUME_RE = re.compile(r"^[0-9a-f]{64}$")


class SearxngServiceError(RuntimeError):
    pass


def _secure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise SearxngServiceError(f"unsafe service-state directory: {path}")


def _secure_file(path: Path) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise SearxngServiceError(f"unsafe service-state file: {path}")


def _secure_settings_file(path: Path) -> None:
    """Keep the secret in an untraversable owner directory but container-readable.

    SearXNG's worker deliberately drops to its image UID. A read-only bind mount
    therefore needs read permission on the file itself; the 0700 parent remains
    the host-side confidentiality boundary.
    """

    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        raise SearxngServiceError(f"unsafe service settings file: {path}")
    os.chmod(path, 0o644)
    if stat.S_IMODE(path.lstat().st_mode) != 0o644:
        raise SearxngServiceError("service settings mode could not be constrained")


def _atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".tmp-service-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
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


def _atomic_text(path: Path, value: str, *, mode: int) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".tmp-settings-", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _settings_body(service_id: str, secret: str) -> str:
    if not SERVICE_ID_RE.fullmatch(service_id):
        raise SearxngServiceError("service settings identity is invalid")
    return (
        "use_default_settings: true\n"
        "general:\n"
        f"  instance_name: \"{INSTANCE_PREFIX}{service_id}\"\n"
        "server:\n"
        f"  secret_key: \"{secret}\"\n"
        "  bind_address: \"0.0.0.0\"\n"
        "  port: 8080\n"
        "  limiter: false\n"
        "  image_proxy: false\n"
        "search:\n"
        "  safe_search: 0\n"
        "  formats:\n"
        "    - html\n"
        "    - json\n"
    )


def _ensure_settings(service_id: str) -> bool:
    """Create or identity-bind settings; return whether a restart is required."""
    if SETTINGS_PATH.exists() or SETTINGS_PATH.is_symlink():
        _secure_settings_file(SETTINGS_PATH)
        try:
            current = SETTINGS_PATH.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SearxngServiceError("service settings are unreadable") from exc
        match = re.search(
            r'^\s*secret_key:\s*"([0-9a-f]{64})"\s*$', current, re.MULTILINE
        )
        if not match:
            raise SearxngServiceError("service settings secret is invalid")
        body = _settings_body(service_id, match.group(1))
        if current == body:
            return False
        _atomic_text(SETTINGS_PATH, body, mode=0o644)
        _secure_settings_file(SETTINGS_PATH)
        return True
    body = _settings_body(service_id, secrets.token_hex(32))
    _atomic_text(SETTINGS_PATH, body, mode=0o644)
    _secure_settings_file(SETTINGS_PATH)
    return False


def _docker(arguments: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    metadata = DOCKER.stat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & 0o022
        or not os.access(DOCKER, os.X_OK)
    ):
        raise SearxngServiceError("root-owned Docker client is unavailable")
    environment = scrubbed_fleet_command_environment()
    for name in tuple(environment):
        if name.upper().startswith("DOCKER_") or name.upper() in {
            "CONTAINER_HOST",
            "BUILDAH_HOST",
            "KUBECONFIG",
        }:
            environment.pop(name, None)
    environment["PATH"] = (
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    result = subprocess.run(
        [
            require_fleet_low_priority_wrapper(),
            str(DOCKER),
            "--host",
            DOCKER_HOST,
            *arguments,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return result


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _read_local(path: str, *, max_bytes: int) -> tuple[int, bytes, str]:
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    with opener.open(f"http://127.0.0.1:{PORT}{path}", timeout=2) as response:
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise SearxngServiceError("service health response is oversized")
        return response.status, data, response.headers.get("content-type", "")


def _healthy(service_id: str) -> bool:
    try:
        status, body, content_type = _read_local("/healthz", max_bytes=16)
        if status != 200 or body != b"OK" or not content_type.startswith("text/plain"):
            return False
        status, body, content_type = _read_local("/config", max_bytes=512 * 1024)
        if status != 200 or not content_type.startswith("application/json"):
            return False
        document = json.loads(body)
        return (
            isinstance(document, dict)
            and document.get("instance_name") == f"{INSTANCE_PREFIX}{service_id}"
            and isinstance(document.get("version"), str)
            and isinstance(document.get("engines"), list)
        )
    except Exception:
        return False


def _load_receipt() -> dict | None:
    if not (RECEIPT_PATH.exists() or RECEIPT_PATH.is_symlink()):
        return None
    _secure_file(RECEIPT_PATH)
    try:
        value = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SearxngServiceError("service receipt is unreadable") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != RECEIPT_SCHEMA
        or not SERVICE_ID_RE.fullmatch(str(value.get("service_id", "")))
        or not CONTAINER_ID_RE.fullmatch(str(value.get("container_id", "")))
        or value.get("image_id") != IMAGE_ID
        or value.get("image_ref") != IMAGE_REF
    ):
        raise SearxngServiceError("service receipt identity is invalid")
    expected_name = f"aeon-searxng-{value['service_id']}"
    if value.get("container_name") != expected_name:
        raise SearxngServiceError("service receipt name is invalid")
    return value


def _inspect(container_id: str) -> dict:
    result = _docker(["container", "inspect", container_id])
    if result.returncode != 0:
        raise SearxngServiceError("receipted SearXNG container is unavailable")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SearxngServiceError("Docker returned malformed inspection evidence") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise SearxngServiceError("Docker returned ambiguous inspection evidence")
    return value[0]


def _validate_container(document: dict, receipt: dict) -> bool:
    config = document.get("Config") or {}
    host = document.get("HostConfig") or {}
    state = document.get("State") or {}
    labels = config.get("Labels") or {}
    ports = host.get("PortBindings") or {}
    mounts = document.get("Mounts") or []
    environment = {}
    for entry in config.get("Env") or []:
        if isinstance(entry, str) and "=" in entry:
            name, value = entry.split("=", 1)
            environment[name] = value
    expected_name = "/" + receipt["container_name"]
    exact_port = (
        set(ports) == {"8080/tcp"}
        and len(ports["8080/tcp"] or []) == 1
        and ports["8080/tcp"][0].get("HostIp") == "127.0.0.1"
        and ports["8080/tcp"][0].get("HostPort") == str(PORT)
        and host.get("PublishAllPorts") is False
    )
    bind_mounts = [
        item for item in mounts
        if isinstance(item, dict) and item.get("Type") == "bind"
    ]
    volume_mounts = [
        item for item in mounts
        if isinstance(item, dict) and item.get("Type") == "volume"
    ]
    exact_bind_mount = len(bind_mounts) == 1 and (
        str(bind_mounts[0].get("Source", "")),
        str(bind_mounts[0].get("Destination", "")),
        bool(bind_mounts[0].get("RW")),
    ) == (str(SETTINGS_PATH), "/etc/searxng/settings.yml", False)
    # The pinned upstream image declares these two VOLUMEs. Docker materializes
    # them even though Aeon supplies only the settings-file bind. Accept exactly
    # those local anonymous volumes and no arbitrary host mount.
    exact_volume_destinations = {"/etc/searxng", "/var/cache/searxng"}
    exact_image_volumes = (
        len(volume_mounts) == 2
        and {str(item.get("Destination", "")) for item in volume_mounts}
        == exact_volume_destinations
        and all(
            item.get("Driver") == "local"
            and item.get("RW") is True
            and ANONYMOUS_VOLUME_RE.fullmatch(str(item.get("Name", "")))
            and item.get("Source")
            == f"/var/lib/docker/volumes/{item.get('Name')}/_data"
            for item in volume_mounts
        )
    )
    exact_mount = (
        len(mounts) == len(bind_mounts) + len(volume_mounts)
        and exact_bind_mount
        and exact_image_volumes
    )
    exact_devices = not (host.get("Devices") or host.get("DeviceRequests"))
    # Docker accepts capability names without the ``CAP_`` prefix but reports
    # them with the prefix on some daemon/API versions. Normalize that
    # representation only; the reviewed capability set remains exact.
    cap_add = {
        str(value).removeprefix("CAP_")
        for value in (host.get("CapAdd") or [])
    }
    exact_security = (
        host.get("Privileged") is False
        and set(host.get("CapDrop") or []) == {"ALL"}
        and cap_add == {"CHOWN", "SETGID", "SETUID"}
        and set(host.get("SecurityOpt") or [])
        in ({"no-new-privileges"}, {"no-new-privileges:true"})
        and host.get("ReadonlyRootfs") is False
        and host.get("NetworkMode") == "bridge"
        and host.get("PidMode") in {"", None}
        and host.get("IpcMode") in {"private", ""}
        and host.get("AutoRemove") is False
        and host.get("Init") in {False, None}
        and (host.get("LogConfig") or {}).get("Type") == "json-file"
        and (host.get("LogConfig") or {}).get("Config")
        in ({}, {"max-file": "2", "max-size": "10m"})
    )
    exact_limits = (
        host.get("Memory") == 512 * 1024 * 1024
        and host.get("MemorySwap") == 512 * 1024 * 1024
        and host.get("NanoCpus") == 1_000_000_000
        and host.get("PidsLimit") == 256
        and host.get("CpuShares") == 2
        and host.get("BlkioWeight") == 10
        and host.get("OomScoreAdj") == 1000
        and host.get("ShmSize") == 128 * 1024 * 1024
        and (host.get("RestartPolicy") or {}).get("Name") == "unless-stopped"
    )
    expected_compute_env = {
        "CUDA_VISIBLE_DEVICES": "void",
        "GPU_DEVICE_ORDINAL": "-1",
        "HIP_VISIBLE_DEVICES": "-1",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "ROCR_VISIBLE_DEVICES": "-1",
    }
    exact_env = all(
        environment.get(name) == value
        for name, value in expected_compute_env.items()
    )
    forbidden_authority = {
        name
        for name in environment
        if name.upper().startswith(("AEON_FLEET", "FLEET_"))
        or name.upper() in {
            "GPU_AGENT_CLAIM_ID",
            "GPU_MEM_LIMIT_GB",
            "GPU_LEASE_ID",
            "GPU_LEASE_RUN_DIR",
        }
    }
    exact_env = exact_env and not forbidden_authority
    if not (
        document.get("Id") == receipt["container_id"]
        and document.get("Name") == expected_name
        and document.get("Image") == IMAGE_ID
        and config.get("Image") == IMAGE_REF
        and labels.get("owner") == "aday"
        and labels.get("com.bc_aeon.component") == "searxng"
        and labels.get("com.bc_aeon.service-id") == receipt["service_id"]
        and exact_port
        and exact_mount
        and exact_devices
        and exact_security
        and exact_limits
        and exact_env
    ):
        raise SearxngServiceError("receipted container configuration does not match")
    return state.get("Running") is True


def _create_receipt(service_id: str) -> dict:
    if not SERVICE_ID_RE.fullmatch(service_id):
        raise SearxngServiceError("new service identity is invalid")
    name = f"aeon-searxng-{service_id}"
    arguments = [
        "container", "create",
        "--name", name,
        "--label", "owner=aday",
        "--label", "com.bc_aeon.component=searxng",
        "--label", f"com.bc_aeon.service-id={service_id}",
        "--restart", "unless-stopped",
        "--publish", f"127.0.0.1:{PORT}:8080",
        "--network", "bridge",
        "--mount", f"type=bind,src={SETTINGS_PATH},dst=/etc/searxng/settings.yml,readonly",
        "--memory", "512m",
        "--memory-swap", "512m",
        "--cpus", "1.0",
        "--cpu-shares", "2",
        "--blkio-weight", "10",
        "--oom-score-adj", "1000",
        "--pids-limit", "256",
        "--shm-size", "128m",
        "--cap-drop", "ALL",
        "--cap-add", "CHOWN",
        "--cap-add", "SETGID",
        "--cap-add", "SETUID",
        "--security-opt", "no-new-privileges",
        "--log-driver", "json-file",
        "--log-opt", "max-size=10m",
        "--log-opt", "max-file=2",
        "--env", "SEARXNG_BASE_URL=http://127.0.0.1:8095/",
        "--env", "CUDA_VISIBLE_DEVICES=void",
        "--env", "GPU_DEVICE_ORDINAL=-1",
        "--env", "HIP_VISIBLE_DEVICES=-1",
        "--env", "NVIDIA_VISIBLE_DEVICES=void",
        "--env", "ROCR_VISIBLE_DEVICES=-1",
        IMAGE_REF,
    ]
    result = _docker(arguments, timeout=60)
    container_id = result.stdout.strip()
    if result.returncode != 0 or not CONTAINER_ID_RE.fullmatch(container_id):
        raise SearxngServiceError("Docker refused the bounded SearXNG container create")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "service_id": service_id,
        "container_id": container_id,
        "container_name": name,
        "image_id": IMAGE_ID,
        "image_ref": IMAGE_REF,
    }
    document = _inspect(container_id)
    if _validate_container(document, receipt):
        raise SearxngServiceError("new container unexpectedly started before receipt commit")
    _atomic_json(RECEIPT_PATH, receipt)
    _secure_file(RECEIPT_PATH)
    return receipt


def ensure_service() -> None:
    _secure_directory(STATE_ROOT)
    lock_descriptor = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(lock_descriptor, 0o600)
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        receipt = _load_receipt()
        if receipt is None:
            if SETTINGS_PATH.exists() or SETTINGS_PATH.is_symlink():
                raise SearxngServiceError(
                    "orphaned service settings exist without an ownership receipt"
                )
            service_id = secrets.token_hex(16)
            _ensure_settings(service_id)
            receipt = _create_receipt(service_id)
            running = False
        else:
            document = _inspect(receipt["container_id"])
            running = _validate_container(document, receipt)
            settings_changed = _ensure_settings(receipt["service_id"])
            if running and settings_changed:
                result = _docker(
                    ["container", "restart", receipt["container_id"]], timeout=60
                )
                if result.returncode != 0:
                    raise SearxngServiceError(
                        "Docker refused the exact receipted service restart"
                    )
            elif running and _healthy(receipt["service_id"]):
                return
        if not running:
            result = _docker(["container", "start", receipt["container_id"]], timeout=60)
            if result.returncode != 0:
                raise SearxngServiceError("Docker refused the exact receipted service start")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if _healthy(receipt["service_id"]):
                return
            time.sleep(1)
        raise SearxngServiceError("receipted SearXNG service did not become healthy")
    finally:
        os.close(lock_descriptor)


def main() -> int:
    try:
        ensure_service()
    except Exception as exc:
        print(f"SearXNG service unavailable: {exc}", file=os.sys.stderr)
        return 1
    print(f"SearXNG ready at {HEALTH_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
