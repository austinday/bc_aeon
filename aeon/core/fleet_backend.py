"""Private Fleet Compute client and fail-closed broker-only selection."""

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

from aeon.core.model_identity import runtime_pool_summary
from aeon.core.utils.io import read_bounded_fd

DEFAULT_BROKER_SOCKET = Path("/home/aday/.local/state/fleet-compute/broker.sock")
DEFAULT_NEXUS_CAPABILITY = Path(
    "/home/aday/.local/state/fleet-compute/nexus-interactive.capability"
)
DEFAULT_QWEN_PROFILE = "aeon-qwen38-standard"
DEFAULT_TICKET_TTL_SECONDS = 300.0
BENCHMARK_COMPUTE_STATUS_FD_ENV = "AEON_BENCHMARK_COMPUTE_STATUS_FD"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}$")
_TICKET_ID = re.compile(r"^fd-[0-9a-f]{32}$")
_JOB_ID = re.compile(r"^fj-[0-9a-f]{32}$")


class FleetBackendError(RuntimeError):
    pass


class FleetBrokerUnavailable(FleetBackendError):
    """The private broker transport is temporarily unavailable."""


def _publish_benchmark_compute_state(state: str) -> None:
    """Best-effort one-way status for the benchmark parent process.

    The benchmark runner supplies an inherited anonymous pipe, never a path or
    lease identifier. Ordinary Aeon launches do not carry this descriptor.
    Requiring a FIFO keeps a caller-supplied descriptor from turning this tiny
    status channel into writes to an unrelated regular file or terminal.
    """

    raw_descriptor = os.environ.get(BENCHMARK_COMPUTE_STATUS_FD_ENV, "")
    if not raw_descriptor.isascii() or not raw_descriptor.isdecimal():
        return
    try:
        descriptor = int(raw_descriptor)
        metadata = os.fstat(descriptor)
        if descriptor <= 2 or not stat.S_ISFIFO(metadata.st_mode):
            return
        os.write(descriptor, (state + "\n").encode("ascii"))
    except (OSError, OverflowError, ValueError):
        # This channel is observational. It must never alter Fleet ownership.
        return


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
        raise FleetBrokerUnavailable("fleet broker socket is unavailable") from exc
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


def _read_nexus_capability(path: Path) -> str:
    """Read the broker capability without following links or weakening ownership."""

    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        parent = absolute.parent.lstat()
    except OSError as exc:
        raise FleetBackendError("Nexus interactive capability is unavailable") from exc
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise FleetBackendError("Nexus interactive capability is not owner-safe")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise FleetBackendError("Nexus interactive capability is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > 512
        ):
            raise FleetBackendError("Nexus interactive capability is not owner-safe")
        raw = read_bounded_fd(descriptor, 512)
    finally:
        os.close(descriptor)
    try:
        token = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise FleetBackendError("Nexus interactive capability is malformed") from exc
    # The service-side loader accepts the conventional newline written by
    # secret/capability file tooling.  Normalize exactly one line ending while
    # continuing to reject every other whitespace byte below.
    if token.endswith("\r\n"):
        token = token[:-2]
    elif token.endswith("\n"):
        token = token[:-1]
    if (
        not 32 <= len(token) <= 512
        or any(character.isspace() for character in token)
    ):
        raise FleetBackendError("Nexus interactive capability is malformed")
    return token


class FleetBrokerClient:
    """Small dependency-free client for the broker's owner-only Unix API."""

    def __init__(
        self,
        socket_path: str | Path = DEFAULT_BROKER_SOCKET,
        *,
        timeout: float = 10,
        nexus_capability_path: str | Path | None = None,
    ) -> None:
        self.socket_path = Path(socket_path).expanduser()
        self.timeout = float(timeout)
        self.nexus_capability_path = (
            Path(nexus_capability_path).expanduser()
            if nexus_capability_path is not None
            else None
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        socket_path = _validated_socket(self.socket_path)
        body = None if payload is None else json.dumps(
            dict(payload), separators=(",", ":")
        ).encode("utf-8")
        request_headers = {"Accept": "application/json"}
        request_headers.update(dict(headers or {}))
        if body is not None:
            request_headers.update(
                {"Content-Type": "application/json", "Content-Length": str(len(body))}
            )
        connection = _UnixHTTPConnection(socket_path, self.timeout)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            raw = response.read(1024 * 1024 + 1)
        except (OSError, http.client.HTTPException) as exc:
            raise FleetBrokerUnavailable("fleet broker request failed") from exc
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
            if response.status in {502, 503, 504}:
                raise FleetBrokerUnavailable(
                    f"fleet broker is temporarily unavailable: {detail or response.status}"
                )
            raise FleetBackendError(
                f"fleet broker rejected the request: {detail or response.status}"
            )
        if not isinstance(decoded, dict):
            raise FleetBackendError("fleet broker response is not an object")
        return decoded

    def _nexus_headers(self) -> dict[str, str] | None:
        if self.nexus_capability_path is None:
            return None
        token = _read_nexus_capability(self.nexus_capability_path)
        return {"Authorization": f"Bearer {token}"}

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/v1/status")

    def service_policy(self, service_id: str) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(str(service_id or "")):
            raise FleetBackendError("fleet service ID is invalid")
        return self._request("GET", f"/v1/service-policies/{service_id}")

    def set_service_policy(
        self,
        service_id: str,
        *,
        requested_replicas: int,
        actor: str,
        preferred_model: str | None = None,
    ) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(str(service_id or "")):
            raise FleetBackendError("fleet service ID is invalid")
        if isinstance(requested_replicas, bool) or not isinstance(requested_replicas, int):
            raise FleetBackendError("fleet replica count is invalid")
        if not _SAFE_ID.fullmatch(str(actor or "")):
            raise FleetBackendError("fleet policy actor is invalid")
        if preferred_model is not None and not _SAFE_ID.fullmatch(preferred_model):
            raise FleetBackendError("fleet preferred model is invalid")
        payload: dict[str, Any] = {
            "requested_replicas": requested_replicas,
            "actor": actor,
        }
        if preferred_model is not None:
            payload["preferred_model"] = preferred_model
        headers = self._nexus_headers()
        return self._request(
            "PUT",
            f"/v1/service-policies/{service_id}",
            payload,
            **({"headers": headers} if headers is not None else {}),
        )

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

    def acquire_nexus_interactive_service(
        self,
        *,
        profile: str,
        consumer: str,
        idempotency_key: str,
        ttl_seconds: float,
    ) -> dict[str, Any]:
        if self.nexus_capability_path is None:
            raise FleetBackendError("Nexus interactive capability is not configured")
        return self._request(
            "POST",
            "/v1/nexus/interactive/services",
            {
                "profile": profile,
                "consumer": consumer,
                "idempotency_key": idempotency_key,
                "ttl_seconds": ttl_seconds,
                "metadata": {"purpose": "authenticated direct Nexus turn"},
            },
            headers=self._nexus_headers(),
        )

    def service_status(self, ticket_id: str) -> dict[str, Any]:
        if not _TICKET_ID.fullmatch(ticket_id):
            raise FleetBackendError("fleet broker ticket ID is invalid")
        return self._request("GET", f"/v1/services/{ticket_id}")

    def renew_service(self, ticket_id: str, *, ttl_seconds: float) -> dict[str, Any]:
        if not _TICKET_ID.fullmatch(ticket_id):
            raise FleetBackendError("fleet broker ticket ID is invalid")
        headers = self._nexus_headers()
        return self._request(
            "POST",
            f"/v1/services/{ticket_id}/renew",
            {"ttl_seconds": ttl_seconds},
            **({"headers": headers} if headers is not None else {}),
        )

    def release_service(self, ticket_id: str) -> dict[str, Any]:
        if not _TICKET_ID.fullmatch(ticket_id):
            raise FleetBackendError("fleet broker ticket ID is invalid")
        headers = self._nexus_headers()
        return self._request(
            "DELETE",
            f"/v1/services/{ticket_id}",
            **({"headers": headers} if headers is not None else {}),
        )

    def submit_job(
        self,
        *,
        profile: str,
        project: str,
        idempotency_key: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit ordinary durable owner work through Fleet's batch API."""

        for label, value in (
            ("profile", profile),
            ("project", project),
            ("idempotency_key", idempotency_key),
        ):
            if not _SAFE_ID.fullmatch(str(value or "")):
                raise FleetBackendError(f"fleet batch {label} is invalid")
        return self._request(
            "POST",
            "/v1/jobs",
            {
                "profile": profile,
                "project": project,
                "idempotency_key": idempotency_key,
                "payload": dict(payload or {}),
            },
        )

    def job_status(self, job_id: str) -> dict[str, Any]:
        if not _JOB_ID.fullmatch(str(job_id or "")):
            raise FleetBackendError("fleet batch job ID is invalid")
        return self._request("GET", f"/v1/jobs/{job_id}")


_TERMINAL_RUNTIME_STATES = frozenset({"failed", "lost", "released", "stopped"})


def _profile_registry_from_status(
    status: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    profiles = status.get("profiles")
    if not isinstance(profiles, list):
        raise FleetBackendError("fleet broker status omitted its profile registry")
    result: dict[str, Mapping[str, Any]] = {}
    for item in profiles:
        if not isinstance(item, Mapping):
            raise FleetBackendError("fleet broker advertised a malformed profile")
        profile_id = item.get("profile_id")
        if not isinstance(profile_id, str) or not _SAFE_ID.fullmatch(profile_id):
            raise FleetBackendError("fleet broker advertised a malformed profile ID")
        if profile_id in result:
            raise FleetBackendError("fleet broker advertised a duplicate profile")
        result[profile_id] = item
    return result


def _service_registry_from_status(
    status: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]] | None:
    if "services" not in status:
        # Older brokers advertised only concrete profiles. Exact-profile
        # validation below remains the deliberately narrow compatibility path.
        return None
    services = status.get("services")
    if not isinstance(services, list):
        raise FleetBackendError("fleet broker advertised a malformed service registry")
    result: dict[str, Mapping[str, Any]] = {}
    for item in services:
        if not isinstance(item, Mapping):
            raise FleetBackendError("fleet broker advertised a malformed service")
        service_id = item.get("service_id")
        if not isinstance(service_id, str) or not _SAFE_ID.fullmatch(service_id):
            raise FleetBackendError("fleet broker advertised a malformed service ID")
        if service_id in result:
            raise FleetBackendError("fleet broker advertised a duplicate service")
        result[service_id] = item
    return result


def _runtime_registry_from_status(
    status: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    runtimes = status.get("runtimes", [])
    if not isinstance(runtimes, list):
        raise FleetBackendError("fleet broker advertised malformed runtime evidence")
    checked: list[Mapping[str, Any]] = []
    for item in runtimes:
        if not isinstance(item, Mapping):
            raise FleetBackendError("fleet broker advertised malformed runtime evidence")
        profile_id = item.get("profile_id")
        state = item.get("state")
        if (
            not isinstance(profile_id, str)
            or not _SAFE_ID.fullmatch(profile_id)
            or not isinstance(state, str)
            or not state
            or len(state) > 64
        ):
            raise FleetBackendError("fleet broker advertised malformed runtime evidence")
        checked.append(item)
    return tuple(checked)


def _validate_aeon_service_profile(
    profile: Mapping[str, Any],
    *,
    expected_service_id: str,
    logical_variant: bool,
) -> None:
    enabled = profile.get("enabled")
    if not isinstance(enabled, bool):
        raise FleetBackendError("fleet broker advertised malformed profile enablement")
    if profile.get("mode") != "service" or profile.get("project") != "aeon":
        raise FleetBackendError("fleet broker advertised an inconsistent Aeon profile")
    service_id = profile.get("service_id")
    if logical_variant:
        if service_id != expected_service_id:
            raise FleetBackendError(
                "fleet broker service variant has an inconsistent logical service ID"
            )
    elif service_id not in {None, expected_service_id}:
        raise FleetBackendError(
            "fleet broker exact profile conflicts with another logical service"
        )


def _logical_service_variants(
    *,
    service_id: str,
    service: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    if service.get("project") != "aeon":
        raise FleetBackendError("fleet broker logical service is not owned by Aeon")
    purpose = service.get("purpose")
    routing = service.get("request_routing")
    variant_count = service.get("variant_count")
    raw_variants = service.get("variants")
    if (
        not isinstance(purpose, str)
        or not purpose.strip()
        or routing not in {"least_busy", "ticket_affinity"}
        or isinstance(variant_count, bool)
        or not isinstance(variant_count, int)
        or variant_count < 1
        or not isinstance(raw_variants, list)
        or not raw_variants
    ):
        raise FleetBackendError("fleet broker advertised malformed logical service data")

    variants: list[str] = []
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, str) or not _SAFE_ID.fullmatch(raw_variant):
            raise FleetBackendError("fleet broker advertised a malformed service variant")
        if raw_variant in variants:
            raise FleetBackendError("fleet broker advertised a duplicate service variant")
        variants.append(raw_variant)
    if variant_count != len(variants):
        raise FleetBackendError("fleet broker service variant count is inconsistent")

    for variant_id in variants:
        profile = profiles.get(variant_id)
        if profile is None:
            raise FleetBackendError("fleet broker service references an unknown variant")
        _validate_aeon_service_profile(
            profile,
            expected_service_id=service_id,
            logical_variant=True,
        )
        if profile.get("enabled") is not True:
            raise FleetBackendError("fleet broker service references a disabled variant")
        if profile.get("request_routing") != routing:
            raise FleetBackendError("fleet broker service routing metadata is inconsistent")
        variant_purpose = profile.get("purpose")
        if not isinstance(variant_purpose, str) or not variant_purpose.strip():
            raise FleetBackendError(
                "fleet broker service variant has malformed purpose metadata"
            )

    advertised = set(variants)
    enabled_for_service = {
        profile_id
        for profile_id, profile in profiles.items()
        if profile.get("enabled") is True
        and (
            profile.get("service_id") == service_id
            or profile_id == service_id
        )
    }
    if enabled_for_service != advertised:
        raise FleetBackendError("fleet broker enabled service variants are inconsistent")
    return tuple(variants)


def _active_runtime_is_related(
    runtime: Mapping[str, Any],
    *,
    service_id: str,
    profiles: Mapping[str, Mapping[str, Any]],
) -> bool:
    if runtime.get("state") in _TERMINAL_RUNTIME_STATES:
        return False
    profile_id = str(runtime["profile_id"])
    if profile_id == service_id:
        return True
    profile = profiles.get(profile_id)
    return profile is not None and profile.get("service_id") == service_id


def _validate_related_active_runtimes(
    *,
    service_id: str,
    profiles: Mapping[str, Mapping[str, Any]],
    runtimes: tuple[Mapping[str, Any], ...],
) -> None:
    for runtime in runtimes:
        if not _active_runtime_is_related(
            runtime, service_id=service_id, profiles=profiles
        ):
            continue
        runtime_profile_id = str(runtime["profile_id"])
        profile = profiles.get(runtime_profile_id)
        if profile is None:
            raise FleetBackendError(
                "fleet broker has active runtime evidence without a known profile"
            )
        _validate_aeon_service_profile(
            profile,
            expected_service_id=service_id,
            logical_variant=runtime_profile_id != service_id,
        )


def select_compute_backend(
    *,
    environ: Mapping[str, str] | None = None,
    client: FleetBrokerClient | None = None,
) -> tuple[str, str]:
    """Return the single supported production backend: ``broker``."""

    values = os.environ if environ is None else environ
    requested = values.get("AEON_COMPUTE_BACKEND", "broker").strip().lower()
    if requested not in {"auto", "broker"}:
        raise FleetBackendError(
            "AEON_COMPUTE_BACKEND must be broker (auto is a broker-only alias)"
        )

    socket_path = Path(values.get("AEON_FLEET_SOCKET", str(DEFAULT_BROKER_SOCKET))).expanduser()
    if not (socket_path.exists() or socket_path.is_symlink()):
        raise FleetBackendError("fleet broker is required but its socket is absent")

    broker = client or FleetBrokerClient(socket_path)
    status = broker.status()
    if not isinstance(status, Mapping):
        raise FleetBackendError("fleet broker status is not an object")
    profile_id = values.get("AEON_FLEET_PROFILE", DEFAULT_QWEN_PROFILE)
    if not isinstance(profile_id, str) or not _SAFE_ID.fullmatch(profile_id):
        raise FleetBackendError("fleet broker profile ID is invalid")
    profiles = _profile_registry_from_status(status)
    services = _service_registry_from_status(status)
    runtimes = _runtime_registry_from_status(status)

    exact_profile = profiles.get(profile_id)
    if exact_profile is not None:
        _validate_aeon_service_profile(
            exact_profile,
            expected_service_id=profile_id,
            logical_variant=False,
        )
        if exact_profile.get("enabled") is True:
            service = services.get(profile_id) if services is not None else None
            if service is not None:
                _logical_service_variants(
                    service_id=profile_id,
                    service=service,
                    profiles=profiles,
                )
            _validate_related_active_runtimes(
                service_id=profile_id,
                profiles=profiles,
                runtimes=runtimes,
            )
            return "broker", f"enabled profile {profile_id}"

    service = services.get(profile_id) if services is not None else None
    if service is not None:
        variants = _logical_service_variants(
            service_id=profile_id,
            service=service,
            profiles=profiles,
        )
        _validate_related_active_runtimes(
            service_id=profile_id,
            profiles=profiles,
            runtimes=runtimes,
        )
        return "broker", (
            f"enabled logical service {profile_id} with {len(variants)} variant"
            f"{'s' if len(variants) != 1 else ''}"
        )

    active = any(
        _active_runtime_is_related(
            item, service_id=profile_id, profiles=profiles
        )
        for item in runtimes
    )
    if active:
        raise FleetBackendError(
            "fleet broker has Aeon runtime evidence but no enabled matching profile"
        )
    raise FleetBackendError(f"fleet broker profile {profile_id!r} is not enabled")


def validate_loopback_endpoint(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
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
        or parsed.params
        or port is None
        or port < 1024
        or port > 65535
        or parsed.path.rstrip("/") not in {"", "/v1"}
    ):
        raise FleetBackendError("fleet broker endpoint is not an approved loopback API")
    origin = (
        f"http://[::1]:{port}"
        if parsed.hostname == "::1"
        else f"http://127.0.0.1:{port}"
    )
    canonical = origin + "/v1"
    if value not in {origin, origin + "/", canonical, canonical + "/"}:
        raise FleetBackendError("fleet broker endpoint is not canonical")
    return canonical


def _validate_service_snapshot(
    value: Any,
    *,
    expected_ticket_id: str | None,
    expected_profile_id: str | None,
    expected_service_id: str | None,
    expected_consumer: str,
    expected_state: str,
) -> tuple[str, str, str, str, str | None, tuple[str, ...]]:
    """Validate one complete broker demand snapshot.

    Fleet stores service demand against the canonical logical service.  A caller
    may request a concrete deployment profile, so the returned profile is bound
    from the first response rather than compared with that request.  Every later
    response must repeat the exact ticket and canonical identity.
    """

    if not isinstance(value, Mapping):
        raise FleetBackendError("fleet broker service response is not an object")

    ticket_id = value.get("ticket_id")
    if not isinstance(ticket_id, str) or not _TICKET_ID.fullmatch(ticket_id):
        raise FleetBackendError("fleet broker returned an invalid ticket")
    if expected_ticket_id is not None and ticket_id != expected_ticket_id:
        raise FleetBackendError("fleet broker returned a different demand ticket")

    profile_id = value.get("profile_id")
    service_id = value.get("service_id")
    if not isinstance(profile_id, str) or not _SAFE_ID.fullmatch(profile_id):
        raise FleetBackendError("fleet broker returned an invalid demand profile")
    if not isinstance(service_id, str) or not _SAFE_ID.fullmatch(service_id):
        raise FleetBackendError("fleet broker returned an invalid logical service ID")
    if profile_id != service_id:
        raise FleetBackendError(
            "fleet broker demand profile is not its canonical logical service"
        )
    if expected_profile_id is not None and profile_id != expected_profile_id:
        raise FleetBackendError("fleet broker changed the demand profile identity")
    if expected_service_id is not None and service_id != expected_service_id:
        raise FleetBackendError("fleet broker changed the logical service identity")
    if value.get("consumer") != expected_consumer:
        raise FleetBackendError("fleet broker changed the demand consumer identity")

    state = value.get("state")
    if state != expected_state:
        raise FleetBackendError("fleet broker returned an unexpected demand state")
    compute_state = value.get("compute_state")
    endpoint_value = value.get("endpoint")
    runtime_profiles_present = "runtime_profiles" in value
    runtime_profiles_value = value.get("runtime_profiles", [])
    if runtime_profiles_present and (
        not isinstance(runtime_profiles_value, list)
        or len(runtime_profiles_value) > 8
        or any(
            not isinstance(item, str) or not _SAFE_ID.fullmatch(item)
            for item in runtime_profiles_value
        )
        or runtime_profiles_value != sorted(set(runtime_profiles_value))
    ):
        raise FleetBackendError(
            "fleet broker returned malformed concrete runtime profiles"
        )
    runtime_profiles = tuple(runtime_profiles_value)
    if state == "active" and compute_state == "ready":
        endpoint = validate_loopback_endpoint(endpoint_value)
        if runtime_profiles_present and not runtime_profiles:
            raise FleetBackendError(
                "fleet broker omitted the concrete ready runtime profile"
            )
    elif state == "active" and compute_state == "waiting_for_compute":
        if endpoint_value is not None or runtime_profiles:
            raise FleetBackendError(
                "fleet broker returned runtime evidence without ready compute"
            )
        endpoint = None
    elif state == "released" and compute_state == "inactive":
        if endpoint_value is not None or runtime_profiles:
            raise FleetBackendError(
                "fleet broker returned runtime evidence for released demand"
            )
        endpoint = None
    else:
        raise FleetBackendError(
            "fleet broker returned an inconsistent demand/compute state"
        )
    return (
        ticket_id,
        profile_id,
        service_id,
        compute_state,
        endpoint,
        runtime_profiles,
    )


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
        clock: Callable[[], float] = time.monotonic,
        renew_wait: Callable[[float], bool] | None = None,
    ) -> None:
        self.client = client or FleetBrokerClient(
            os.environ.get("AEON_FLEET_SOCKET", str(DEFAULT_BROKER_SOCKET))
        )
        self.profile = profile or os.environ.get("AEON_FLEET_PROFILE", DEFAULT_QWEN_PROFILE)
        identity = consumer or f"aeon/{uuid.uuid4().hex}"
        if not _SAFE_ID.fullmatch(identity):
            raise FleetBackendError("fleet broker consumer identity is invalid")
        self.consumer = identity
        # An idempotency key identifies one acquisition attempt, not the stable
        # Nexus terminal. Reusing a terminal's key after its previous ticket was
        # released would make Fleet return that terminal ticket forever.
        self._idempotency_key = f"aeon-qwen/{uuid.uuid4().hex}"
        self.ttl_seconds = float(ttl_seconds)
        self.sleep = sleep
        self._clock = clock
        self._ticket_id: str | None = None
        self._ticket_profile_id: str | None = None
        self._service_id: str | None = None
        self.endpoint: str | None = None
        self._stop = threading.Event()
        self._renew_wait = renew_wait or self._stop.wait
        self._renew_thread: threading.Thread | None = None
        self._renew_error: BaseException | None = None
        self._renew_deadline: float | None = None
        self._endpoint_lock = threading.RLock()
        self._endpoint_change_handler: Callable[[str, tuple[str, ...]], None] | None = None
        self._pending_endpoint: str | None = None
        self._runtime_profiles: tuple[str, ...] = ()
        self._pending_runtime_profiles: tuple[str, ...] = ()
        self._started = False
        self._close_lock = threading.Lock()

    def set_endpoint_change_handler(
        self, handler: Callable[[str, tuple[str, ...]], None]
    ) -> None:
        """Register the foreground callback used to bind a promoted runtime."""

        if not callable(handler):
            raise FleetBackendError("fleet endpoint change handler must be callable")
        with self._endpoint_lock:
            self._endpoint_change_handler = handler

    @property
    def logical_service_id(self) -> str | None:
        """Broker-confirmed logical service identity for this exact ticket."""

        return self._service_id

    @property
    def ticket_id(self) -> str | None:
        """Exact broker ticket acquired by this session, if still owned."""

        return self._ticket_id

    @property
    def runtime_profiles(self) -> tuple[str, ...]:
        """Broker-proven concrete profiles behind the currently bound endpoint."""

        with self._endpoint_lock:
            return self._runtime_profiles

    def _publish_compute(
        self,
        state: str,
        summary: str,
        *,
        runtime_profiles: tuple[str, ...] = (),
    ) -> None:
        """Best-effort sanitized admission state for Nexus presence telemetry."""

        _publish_benchmark_compute_state(state)
        try:
            from aeon.core.presence import get_active_presence

            presence = get_active_presence()
            if presence is not None:
                concrete_profile = (
                    ", ".join(runtime_profiles)
                    if runtime_profiles
                    else self._service_id or self.profile
                )
                presence.update_compute(
                    state=state,
                    profile=concrete_profile,
                    summary=summary,
                )
        except Exception:
            # Presence is display-only and must never alter broker ownership.
            pass

    def _wait_ready(self) -> tuple[str, tuple[str, ...]]:
        delay = 1.0
        reconnect_delay = 1.0
        while not self._stop.is_set():
            if self._ticket_id is None:
                raise FleetBackendError("fleet broker ticket was lost")
            try:
                status = self.client.service_status(self._ticket_id)
                _, _, _, compute_state, endpoint, runtime_profiles = (
                    self._validate_bound_snapshot(status, expected_state="active")
                )
                if compute_state == "ready":
                    self._publish_compute(
                        "allocated",
                        runtime_pool_summary(runtime_profiles),
                        runtime_profiles=runtime_profiles,
                    )
                    if endpoint is None:  # Kept explicit for type narrowing and auditability.
                        raise FleetBackendError("fleet broker omitted its ready endpoint")
                    return endpoint, runtime_profiles
                self._publish_compute(
                    "waiting_for_compute",
                    "Waiting for compatible compute; renters and existing work keep priority",
                )
                renewed = self.client.renew_service(
                    self._ticket_id, ttl_seconds=self.ttl_seconds
                )
                (
                    _,
                    _,
                    _,
                    renewed_compute_state,
                    renewed_endpoint,
                    renewed_runtime_profiles,
                ) = (
                    self._validate_bound_snapshot(renewed, expected_state="active")
                )
                self._renew_deadline = self._clock() + self.ttl_seconds
                if renewed_compute_state == "ready":
                    self._publish_compute(
                        "allocated",
                        runtime_pool_summary(renewed_runtime_profiles),
                        runtime_profiles=renewed_runtime_profiles,
                    )
                    if renewed_endpoint is None:
                        raise FleetBackendError("fleet broker omitted its ready endpoint")
                    return renewed_endpoint, renewed_runtime_profiles
            except FleetBrokerUnavailable as exc:
                # Retry only typed transport/502/503/504 outages. Broker refusals,
                # malformed snapshots, and identity drift remain immediate errors.
                deadline = self._renew_deadline
                now = self._clock()
                if deadline is None or now >= deadline:
                    raise FleetBackendError(
                        "fleet broker remained unavailable through the ticket lease deadline"
                    ) from exc
                self._publish_compute(
                    "waiting_for_compute",
                    "Fleet control plane is reconnecting; the current ticket remains bounded",
                )
                wait_for = min(reconnect_delay, max(0.0, deadline - now))
                if wait_for <= 0:
                    raise FleetBackendError(
                        "fleet broker remained unavailable through the ticket lease deadline"
                    ) from exc
                self.sleep(wait_for)
                reconnect_delay = min(15.0, reconnect_delay * 2.0)
                continue
            reconnect_delay = 1.0
            self.sleep(delay)
            delay = min(15.0, delay * 1.5)
        raise FleetBackendError("fleet broker wait was cancelled")

    def _validate_bound_snapshot(
        self, value: Any, *, expected_state: str
    ) -> tuple[str, str, str, str, str | None, tuple[str, ...]]:
        if self._ticket_id is None:
            raise FleetBackendError("fleet broker ticket was lost")
        if self._ticket_profile_id is None or self._service_id is None:
            raise FleetBackendError("fleet broker logical service identity was lost")
        return _validate_service_snapshot(
            value,
            expected_ticket_id=self._ticket_id,
            expected_profile_id=self._ticket_profile_id,
            expected_service_id=self._service_id,
            expected_consumer=self.consumer,
            expected_state=expected_state,
        )

    def _release_known_ticket(self, ticket_id: str) -> None:
        """Release and clear ownership only after exact terminal proof."""

        result = self.client.release_service(ticket_id)
        _validate_service_snapshot(
            result,
            expected_ticket_id=ticket_id,
            expected_profile_id=self._ticket_profile_id,
            expected_service_id=self._service_id,
            expected_consumer=self.consumer,
            expected_state="released",
        )
        self._ticket_id = None
        self._ticket_profile_id = None
        self._service_id = None

    def _bind_acquired_identity(
        self, value: Mapping[str, Any], ticket_id: str
    ) -> tuple[str, str]:
        """Bind only a ticket proven to belong to this exact acquisition.

        A syntactically valid ID alone is not ownership evidence: a malformed or
        cross-wired broker response could otherwise make this consumer delete a
        different consumer's demand.  The initial consumer and reviewed Qwen
        service identity must match before the ID is retained or released.
        """

        if value.get("consumer") != self.consumer:
            raise FleetBackendError(
                "fleet broker returned an unowned demand consumer identity"
            )
        profile_id = value.get("profile_id")
        service_id = value.get("service_id")
        if (
            not isinstance(profile_id, str)
            or not _SAFE_ID.fullmatch(profile_id)
            or not isinstance(service_id, str)
            or not _SAFE_ID.fullmatch(service_id)
            or profile_id != service_id
            or service_id not in {self.profile, DEFAULT_QWEN_PROFILE}
        ):
            raise FleetBackendError(
                "fleet broker returned an unowned or unreviewed logical service identity"
            )
        self._ticket_id = ticket_id
        self._ticket_profile_id = profile_id
        self._service_id = service_id
        return profile_id, service_id

    def start(self) -> str:
        if self._started:
            raise FleetBackendError("fleet broker session objects cannot be restarted")
        self._started = True
        result = self.client.acquire_service(
            profile=self.profile,
            consumer=self.consumer,
            idempotency_key=self._idempotency_key,
            ttl_seconds=self.ttl_seconds,
        )
        if not isinstance(result, Mapping):
            raise FleetBackendError("fleet broker service response is not an object")
        ticket = result.get("ticket_id")
        if not isinstance(ticket, str) or not _TICKET_ID.fullmatch(ticket):
            raise FleetBackendError("fleet broker returned an invalid ticket")
        try:
            profile_id, service_id = self._bind_acquired_identity(result, ticket)
            (
                _,
                _,
                _,
                compute_state,
                acquired_endpoint,
                runtime_profiles,
            ) = _validate_service_snapshot(
                result,
                expected_ticket_id=ticket,
                expected_profile_id=profile_id,
                expected_service_id=service_id,
                expected_consumer=self.consumer,
                expected_state="active",
            )
            self._renew_deadline = self._clock() + self.ttl_seconds
            if compute_state == "ready":
                if acquired_endpoint is None:
                    raise FleetBackendError("fleet broker omitted its ready endpoint")
                self.endpoint = acquired_endpoint
                self._runtime_profiles = runtime_profiles
                self._publish_compute(
                    "allocated",
                    runtime_pool_summary(runtime_profiles),
                    runtime_profiles=runtime_profiles,
                )
            else:
                self._publish_compute(
                    "waiting_for_compute",
                    "Waiting for compatible compute; renters and existing work keep priority",
                )
                self.endpoint, self._runtime_profiles = self._wait_ready()
            self._renew_thread = threading.Thread(
                target=self._renew_loop,
                name="aeon-fleet-ticket-renewal",
                daemon=True,
            )
            self._renew_thread.start()
        except BaseException as start_error:
            self._publish_compute(
                "unavailable", "Standard Qwen compute is currently unavailable"
            )
            # Release only after the response proved that this exact consumer
            # owns the ticket under a reviewed Qwen service identity. Ambiguous
            # IDs are deliberately left to their bounded broker TTL.
            self.endpoint = None
            if self._ticket_id is not None:
                try:
                    self._release_known_ticket(ticket)
                except BaseException:
                    # Keep ticket_id so SessionManager.close() can retry the release.
                    if hasattr(start_error, "add_note"):
                        start_error.add_note(
                            "Fleet release was attempted but exact terminal proof failed; "
                            "the ticket was retained for close() to retry."
                        )
            raise
        from aeon.core.process_resources import register_service_owner

        register_service_owner(self)
        return self.endpoint

    def request_stop(self) -> None:
        """Wake an admission/renewal wait before process-local cleanup."""

        self._stop.set()

    def _renew_loop(self) -> None:
        interval = min(120.0, max(15.0, self.ttl_seconds / 3.0))
        while not self._renew_wait(interval):
            retry_delay = 1.0
            while not self._stop.is_set():
                try:
                    self._renew_once()
                    break
                except FleetBrokerUnavailable as exc:
                    deadline = self._renew_deadline
                    now = self._clock()
                    if deadline is None or now >= deadline:
                        self._latch_renew_error(exc)
                        return
                    self._publish_compute(
                        "waiting_for_compute",
                        "Fleet control plane is reconnecting; the current ticket remains bounded",
                    )
                    delay = min(retry_delay, max(0.0, deadline - now))
                    if delay <= 0 or self._renew_wait(delay):
                        return
                    retry_delay = min(15.0, retry_delay * 2.0)
                except BaseException as exc:
                    self._latch_renew_error(exc)
                    return

    def _latch_renew_error(self, exc: BaseException) -> None:
        """Make a terminal or lease-expired renewal failure visible once."""

        self._renew_error = exc
        self._publish_compute(
            "unavailable",
            "Standard Qwen compute status could not be renewed",
        )

    def _renew_once(self) -> None:
        """Renew demand and refresh the exact compute evidence Nexus consumes."""

        if self._ticket_id is None:
            raise FleetBackendError("fleet broker ticket was lost")
        status = self.client.renew_service(
            self._ticket_id, ttl_seconds=self.ttl_seconds
        )
        _, _, _, compute_state, endpoint, runtime_profiles = self._validate_bound_snapshot(
            status, expected_state="active"
        )
        self._renew_deadline = self._clock() + self.ttl_seconds
        if compute_state == "ready":
            if endpoint is None:
                raise FleetBackendError("fleet broker omitted its ready endpoint")
            with self._endpoint_lock:
                binding_changed = self.endpoint is not None and (
                    endpoint != self.endpoint
                    or runtime_profiles != self._runtime_profiles
                )
                if binding_changed:
                    self._pending_endpoint = endpoint
                    self._pending_runtime_profiles = runtime_profiles
                else:
                    self._pending_endpoint = None
                    self._pending_runtime_profiles = ()
                    self._runtime_profiles = runtime_profiles
            if binding_changed:
                self._publish_compute(
                    "allocated",
                    runtime_pool_summary(runtime_profiles)
                    + "; runtime binding switch pending next turn",
                    runtime_profiles=runtime_profiles,
                )
                return
            self._publish_compute(
                "allocated",
                runtime_pool_summary(runtime_profiles),
                runtime_profiles=runtime_profiles,
            )
            return
        else:
            self._publish_compute(
                "waiting_for_compute",
                "Waiting for compatible compute; renters and existing work keep priority",
            )
            return

    def ensure_ready(self) -> None:
        if self._renew_error is not None:
            raise FleetBackendError("fleet broker ticket renewal failed") from self._renew_error
        endpoint, runtime_profiles = self._wait_ready()
        with self._endpoint_lock:
            if self.endpoint is not None and (
                endpoint != self.endpoint
                or runtime_profiles != self._runtime_profiles
            ):
                self._pending_endpoint = endpoint
                self._pending_runtime_profiles = runtime_profiles
            else:
                self._runtime_profiles = runtime_profiles

        with self._endpoint_lock:
            pending = self._pending_endpoint
            handler = self._endpoint_change_handler
            if pending is None:
                return
            if handler is None:
                raise FleetBackendError(
                    "fleet broker endpoint changed without a safe rebind handler"
                )
            try:
                handler(pending, self._pending_runtime_profiles)
            except BaseException as exc:
                raise FleetBackendError(
                    "failed to bind the promoted fleet runtime"
                ) from exc
            self.endpoint = pending
            self._runtime_profiles = self._pending_runtime_profiles
            if self._pending_endpoint == pending:
                self._pending_endpoint = None
                self._pending_runtime_profiles = ()
        self._publish_compute(
            "allocated",
            runtime_pool_summary(self._runtime_profiles),
            runtime_profiles=self._runtime_profiles,
        )

    def close(self) -> dict[str, str] | None:
        """Release this exact demand and return only sanitized terminal proof."""

        with self._close_lock:
            return self._close_locked()

    def _close_locked(self) -> dict[str, str] | None:
        self._stop.set()
        if self._renew_thread is not None:
            self._renew_thread.join(timeout=2)
        self.endpoint = None
        self._runtime_profiles = ()
        self._pending_runtime_profiles = ()
        release_proof = None
        if self._ticket_id is not None:
            ticket_id = self._ticket_id
            try:
                self._release_known_ticket(ticket_id)
            except FleetBackendError as exc:
                raise FleetBackendError(
                    "fleet broker did not prove exact ticket release"
                ) from exc
            release_proof = {"state": "released", "compute_state": "inactive"}
        self._publish_compute("idle", "")
        from aeon.core.process_resources import unregister_service_owner

        unregister_service_owner(self)
        return release_proof
