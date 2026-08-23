"""Private Fleet Compute service client and migration-safe backend selection.

The new broker is optional while it is being rolled out.  ``auto`` uses it only
when its owner-only Unix socket is healthy and an enabled Aeon service profile is
advertised.  If no broker socket exists, Aeon's proven coordinator lifecycle
remains the compatibility backend.  A present-but-unhealthy broker fails closed
so two control planes cannot accidentally manage the same workload.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import stat
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse


DEFAULT_BROKER_SOCKET = Path("/home/aday/.local/state/fleet-compute/broker.sock")
DEFAULT_QWEN_PROFILE = "aeon-qwen38-standard"
DEFAULT_TICKET_TTL_SECONDS = 300.0
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_TICKET_ID = re.compile(r"^fd-[0-9a-f]{32}$")


class FleetBackendError(RuntimeError):
    pass


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: Path, timeout: float) -> None:
        super().__init__("fleet-compute", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        transport = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        transport.settimeout(self.timeout)
        try:
            transport.connect(str(self.socket_path))
        except BaseException:
            transport.close()
            raise
        self.sock = transport


def _validated_socket(path: Path) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        metadata = absolute.lstat()
        parent = absolute.parent.lstat()
    except OSError as exc:
        raise FleetBackendError("fleet broker socket is unavailable") from exc
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise FleetBackendError("fleet broker socket is not an owner-only Unix socket")
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise FleetBackendError("fleet broker state directory is not owner-private")
    return absolute


class FleetBrokerClient:
    """Small dependency-free client for the broker's owner-only Unix API."""

    def __init__(self, socket_path: str | Path = DEFAULT_BROKER_SOCKET, *, timeout: float = 10) -> None:
        self.socket_path = Path(socket_path).expanduser()
        self.timeout = float(timeout)

    def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        socket_path = _validated_socket(self.socket_path)
        body = None if payload is None else json.dumps(
            dict(payload), separators=(",", ":")
        ).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(body))})
        connection = _UnixHTTPConnection(socket_path, self.timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(1024 * 1024 + 1)
        except (OSError, http.client.HTTPException) as exc:
            raise FleetBackendError("fleet broker request failed") from exc
        finally:
            connection.close()
        if len(raw) > 1024 * 1024:
            raise FleetBackendError("fleet broker response exceeded the safety limit")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FleetBackendError("fleet broker returned invalid JSON") from exc
        if response.status >= 400:
            detail = decoded.get("detail") if isinstance(decoded, dict) else None
            raise FleetBackendError(
                f"fleet broker rejected the request: {detail or response.status}"
            )
        if not isinstance(decoded, dict):
            raise FleetBackendError("fleet broker response is not an object")
        return decoded

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/v1/status")

    def acquire_service(
        self, *, profile: str, consumer: str, idempotency_key: str, ttl_seconds: float
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/services",
            {
                "profile": profile,
                "consumer": consumer,
                "idempotency_key": idempotency_key,
                "ttl_seconds": ttl_seconds,
                "metadata": {},
            },
        )

    def service_status(self, ticket_id: str) -> dict[str, Any]:
        if not _TICKET_ID.fullmatch(ticket_id):
            raise FleetBackendError("fleet broker ticket ID is invalid")
        return self._request("GET", f"/v1/services/{ticket_id}")

    def renew_service(self, ticket_id: str, *, ttl_seconds: float) -> dict[str, Any]:
        if not _TICKET_ID.fullmatch(ticket_id):
            raise FleetBackendError("fleet broker ticket ID is invalid")
        return self._request(
            "POST", f"/v1/services/{ticket_id}/renew", {"ttl_seconds": ttl_seconds}
        )

    def release_service(self, ticket_id: str) -> dict[str, Any]:
        if not _TICKET_ID.fullmatch(ticket_id):
            raise FleetBackendError("fleet broker ticket ID is invalid")
        return self._request("DELETE", f"/v1/services/{ticket_id}")


def _profile_from_status(status: Mapping[str, Any], profile_id: str) -> Mapping[str, Any] | None:
    profiles = status.get("profiles")
    if not isinstance(profiles, list):
        raise FleetBackendError("fleet broker status omitted its profile registry")
    matches = [item for item in profiles if isinstance(item, Mapping) and item.get("profile_id") == profile_id]
    if len(matches) > 1:
        raise FleetBackendError("fleet broker advertised a duplicate Aeon profile")
    return matches[0] if matches else None


def select_compute_backend(
    *,
    environ: Mapping[str, str] | None = None,
    client: FleetBrokerClient | None = None,
) -> tuple[str, str]:
    """Return ``(backend, reason)`` for ``broker`` or ``coordinator``."""

    values = os.environ if environ is None else environ
    requested = values.get("AEON_COMPUTE_BACKEND", "auto").strip().lower()
    if requested not in {"auto", "broker", "coordinator"}:
        raise FleetBackendError(
            "AEON_COMPUTE_BACKEND must be auto, broker, or coordinator"
        )
    if requested == "coordinator":
        return "coordinator", "explicit compatibility backend"

    socket_path = Path(values.get("AEON_FLEET_SOCKET", str(DEFAULT_BROKER_SOCKET))).expanduser()
    if not (socket_path.exists() or socket_path.is_symlink()):
        if requested == "broker":
            raise FleetBackendError("fleet broker was required but its socket is absent")
        return "coordinator", "fleet broker is not installed/running yet"

    broker = client or FleetBrokerClient(socket_path)
    status = broker.status()
    profile_id = values.get("AEON_FLEET_PROFILE", DEFAULT_QWEN_PROFILE)
    profile = _profile_from_status(status, profile_id)
    if profile is not None and (
        profile.get("enabled") is True
        and profile.get("mode") == "service"
        and profile.get("project") == "aeon"
    ):
        return "broker", f"enabled profile {profile_id}"

    active = [
        item for item in status.get("runtimes", [])
        if isinstance(item, Mapping)
        and item.get("profile_id") == profile_id
        and item.get("state") not in {"stopped", "released", "failed"}
    ]
    if active:
        raise FleetBackendError(
            "fleet broker has Aeon runtime evidence but no enabled matching profile"
        )
    if requested == "broker":
        raise FleetBackendError(f"fleet broker profile {profile_id!r} is not enabled")
    return "coordinator", f"fleet broker has no enabled {profile_id} profile"


def validate_loopback_endpoint(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 512:
        raise FleetBackendError("fleet broker returned an invalid endpoint")
    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise FleetBackendError("fleet broker returned an invalid endpoint") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port is None
        or port < 1024
        or port > 65535
        or parsed.path.rstrip("/") not in {"", "/v1"}
    ):
        raise FleetBackendError("fleet broker endpoint is not an approved loopback API")
    return value.rstrip("/") + ("/v1" if parsed.path.rstrip("/") == "" else "")


class BrokerServiceSession:
    """One expiring demand ticket held only while an Aeon process needs Qwen."""

    def __init__(
        self,
        *,
        client: FleetBrokerClient | None = None,
        profile: str | None = None,
        consumer: str | None = None,
        ttl_seconds: float = DEFAULT_TICKET_TTL_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client or FleetBrokerClient(
            os.environ.get("AEON_FLEET_SOCKET", str(DEFAULT_BROKER_SOCKET))
        )
        self.profile = profile or os.environ.get("AEON_FLEET_PROFILE", DEFAULT_QWEN_PROFILE)
        identity = consumer or f"aeon/{uuid.uuid4().hex}"
        if not _SAFE_ID.fullmatch(identity):
            raise FleetBackendError("fleet broker consumer identity is invalid")
        self.consumer = identity
        self.ttl_seconds = float(ttl_seconds)
        self.sleep = sleep
        self.ticket_id: str | None = None
        self.endpoint: str | None = None
        self._stop = threading.Event()
        self._renew_thread: threading.Thread | None = None
        self._renew_error: BaseException | None = None

    def _wait_ready(self) -> str:
        delay = 1.0
        while not self._stop.is_set():
            if self.ticket_id is None:
                raise FleetBackendError("fleet broker ticket was lost")
            status = self.client.service_status(self.ticket_id)
            state = status.get("state")
            compute_state = status.get("compute_state")
            if state != "active":
                raise FleetBackendError("fleet broker demand became inactive")
            if compute_state == "ready":
                return validate_loopback_endpoint(status.get("endpoint"))
            if compute_state != "waiting_for_compute":
                raise FleetBackendError("fleet broker returned an unknown compute state")
            self.client.renew_service(self.ticket_id, ttl_seconds=self.ttl_seconds)
            self.sleep(delay)
            delay = min(15.0, delay * 1.5)
        raise FleetBackendError("fleet broker wait was cancelled")

    def start(self) -> str:
        result = self.client.acquire_service(
            profile=self.profile,
            consumer=self.consumer,
            idempotency_key=f"{self.consumer}/primary-qwen38",
            ttl_seconds=self.ttl_seconds,
        )
        ticket = result.get("ticket_id")
        if not isinstance(ticket, str) or not _TICKET_ID.fullmatch(ticket):
            raise FleetBackendError("fleet broker returned an invalid ticket")
        self.ticket_id = ticket
        self.endpoint = (
            validate_loopback_endpoint(result.get("endpoint"))
            if result.get("compute_state") == "ready"
            else self._wait_ready()
        )
        self._renew_thread = threading.Thread(
            target=self._renew_loop,
            name="aeon-fleet-ticket-renewal",
            daemon=True,
        )
        self._renew_thread.start()
        return self.endpoint

    def _renew_loop(self) -> None:
        interval = min(120.0, max(15.0, self.ttl_seconds / 3.0))
        while not self._stop.wait(interval):
            try:
                if self.ticket_id is None:
                    raise FleetBackendError("fleet broker ticket was lost")
                self.client.renew_service(self.ticket_id, ttl_seconds=self.ttl_seconds)
            except BaseException as exc:
                self._renew_error = exc
                return

    def ensure_ready(self) -> None:
        if self._renew_error is not None:
            raise FleetBackendError("fleet broker ticket renewal failed") from self._renew_error
        endpoint = self._wait_ready()
        if self.endpoint is not None and endpoint != self.endpoint:
            raise FleetBackendError(
                "fleet broker endpoint changed; restart Aeon to bind the replacement safely"
            )

    def close(self) -> None:
        self._stop.set()
        if self._renew_thread is not None:
            self._renew_thread.join(timeout=2)
        if self.ticket_id is not None:
            self.client.release_service(self.ticket_id)
            self.ticket_id = None
