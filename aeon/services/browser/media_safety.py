"""Bounded network and child-process primitives for the browser service.

This module intentionally has no FastAPI or Patchright dependency so its exact
redirect, proxy, byte-limit, and process-group behavior is hermetically tested
outside the browser container.
"""

from __future__ import annotations

import asyncio
import http.client
import ipaddress
import os
from pathlib import Path
import signal
import socket
import ssl
import stat
from typing import Callable, Iterable, Mapping, NamedTuple
from urllib.parse import urlsplit, urlunsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)


DOWNLOAD_CHUNK_BYTES = 64 * 1024
HELPER_STDERR_TAIL_BYTES = 4096
HELPER_TERMINATION_GRACE_SECONDS = 3.0
_BLOCKED_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".lan",
    ".home.arpa",
)


class BrowserMediaSafetyError(RuntimeError):
    """A media fetch or helper could not remain inside its reviewed boundary."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _global_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    return address.is_global


class _PublicTarget(NamedTuple):
    url: str
    scheme: str
    hostname: str
    port: int
    addresses: tuple[str, ...]


def _resolve_public_target(
    value: str,
    *,
    resolver: Callable[..., Iterable[tuple]],
) -> _PublicTarget:
    """Resolve once, reject mixed/private answers, and retain the exact IP set."""

    if not isinstance(value, str) or not value or len(value) > 8192:
        raise BrowserMediaSafetyError("media URL is missing or oversized")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise BrowserMediaSafetyError("media URL contains control characters")
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise BrowserMediaSafetyError("media URL has an invalid port") from exc
    hostname = parsed.hostname
    if (
        scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not 1 <= port <= 65535
    ):
        raise BrowserMediaSafetyError(
            "media URL must be credential-free public HTTP(S) without a fragment"
        )
    try:
        ascii_host = hostname.encode("idna").decode("ascii").rstrip(".").lower()
    except UnicodeError as exc:
        raise BrowserMediaSafetyError("media URL hostname is invalid") from exc
    if (
        ascii_host == "localhost"
        or ascii_host.endswith(_BLOCKED_HOST_SUFFIXES)
        or not ascii_host
    ):
        raise BrowserMediaSafetyError("media URL hostname is not public")

    try:
        literal = ipaddress.ip_address(ascii_host.split("%", 1)[0])
    except ValueError:
        try:
            answers = list(
                resolver(
                    ascii_host,
                    port,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                    proto=socket.IPPROTO_TCP,
                )
            )
        except OSError as exc:
            raise BrowserMediaSafetyError("media URL hostname did not resolve") from exc
        addresses = tuple(
            dict.fromkeys(
                str(answer[4][0])
                for answer in answers
                if isinstance(answer, tuple)
                and len(answer) >= 5
                and isinstance(answer[4], tuple)
                and answer[4]
            )
        )
        if not addresses or any(not _global_address(address) for address in addresses):
            raise BrowserMediaSafetyError(
                "media URL hostname did not resolve exclusively to public addresses"
            )
    else:
        if not literal.is_global:
            raise BrowserMediaSafetyError("media URL address is not public")
        addresses = (str(literal),)
    return _PublicTarget(value, scheme, ascii_host, port, addresses)


def validate_public_http_url(
    value: str,
    *,
    resolver: Callable[..., Iterable[tuple]] = socket.getaddrinfo,
) -> str:
    """Require one credential-free public HTTP(S) destination.

    Redirects are separately disabled by :func:`bounded_download`; resolving
    every address here prevents the helper from turning a page-controlled media
    URL into a fresh loopback, link-local, LAN, or metadata-service request.
    """

    _resolve_public_target(value, resolver=resolver)
    return value


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, hostname: str, port: int, address: str, *, timeout: float):
        super().__init__(hostname, port, timeout=timeout)
        self._pinned_address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
            self.source_address,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname: str, port: int, address: str, *, timeout: float):
        super().__init__(
            hostname,
            port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._pinned_address = address

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
            self.source_address,
        )
        try:
            self.sock = self._context.wrap_socket(
                raw_socket,
                server_hostname=self.host,
            )
        except BaseException:
            raw_socket.close()
            raise


class _PinnedResponse:
    def __init__(self, response, connection, url: str):
        self._response = response
        self._connection = connection
        self._url = url
        self.headers = response.headers

    def geturl(self) -> str:
        return self._url

    def read(self, size: int) -> bytes:
        return self._response.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self._response.close()
        self._connection.close()
        return False


def _open_pinned_response(
    target: _PublicTarget,
    *,
    timeout: float,
    headers: Mapping[str, str],
):
    """Open against one already-validated IP while preserving Host and TLS SNI."""

    parsed = urlsplit(target.url)
    request_target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    safe_headers = {
        str(name): str(value)
        for name, value in headers.items()
        if str(name).lower() not in {"host", "proxy-authorization"}
    }
    errors: list[BaseException] = []
    for address in target.addresses:
        connection_type = (
            _PinnedHTTPSConnection if target.scheme == "https" else _PinnedHTTPConnection
        )
        connection = connection_type(
            target.hostname,
            target.port,
            address,
            timeout=timeout,
        )
        try:
            connection.request("GET", request_target, headers=safe_headers)
            response = connection.getresponse()
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            connection.close()
            errors.append(exc)
            continue
        if 300 <= response.status < 400:
            response.close()
            connection.close()
            raise BrowserMediaSafetyError("media download redirect was refused")
        if not 200 <= response.status < 300:
            status = response.status
            response.close()
            connection.close()
            raise BrowserMediaSafetyError(
                f"media download returned HTTP status {status}"
            )
        return _PinnedResponse(response, connection, target.url)
    raise BrowserMediaSafetyError("media destination could not be reached") from (
        errors[-1] if errors else None
    )


def normalize_public_navigation_url(
    value: str,
    *,
    resolver: Callable[..., Iterable[tuple]] = socket.getaddrinfo,
) -> str:
    """Normalize a user destination and admit only public HTTP(S) navigation."""

    raw = str(value or "").strip()
    if not raw:
        raise BrowserMediaSafetyError("navigation URL is required")
    candidate = raw if "://" in raw else "https://" + raw
    parsed = urlsplit(candidate)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise BrowserMediaSafetyError("browser navigation permits only public HTTP(S)")
    # Anchors are ordinary public navigation state, not a separate network hop.
    network_url = urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")
    )
    validate_public_http_url(network_url, resolver=resolver)
    return candidate


def validate_public_browser_request_url(
    value: str,
    *,
    resolver: Callable[..., Iterable[tuple]] = socket.getaddrinfo,
) -> str:
    """Validate one browser HTTP(S) or WebSocket request/redirect hop."""

    parsed = urlsplit(str(value or ""))
    scheme = parsed.scheme.lower()
    mapped = {"http": "http", "https": "https", "ws": "http", "wss": "https"}
    if scheme not in mapped:
        raise BrowserMediaSafetyError("browser network request scheme is not allowed")
    network_url = urlunsplit(
        (mapped[scheme], parsed.netloc, parsed.path, parsed.query, "")
    )
    validate_public_http_url(network_url, resolver=resolver)
    return value


def validate_http_proxy_url(value: str) -> str:
    """Validate the operator's explicit proxy for stdlib streaming downloads."""

    if not isinstance(value, str) or not value or len(value) > 4096:
        raise BrowserMediaSafetyError("browser media proxy is invalid")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise BrowserMediaSafetyError("browser media proxy contains control characters")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise BrowserMediaSafetyError("browser media proxy has an invalid port") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or port is None
        or not 1 <= port <= 65535
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise BrowserMediaSafetyError(
            "streaming downloads require an explicit HTTP(S) proxy origin"
        )
    return value


def strict_url_opener(proxy_url: str = ""):
    """Build an opener that never inherits proxy variables or follows redirects."""

    proxies: dict[str, str] = {}
    if proxy_url:
        exact = validate_http_proxy_url(proxy_url)
        proxies = {"http": exact, "https": exact}
    return build_opener(ProxyHandler(proxies), _NoRedirect())


def bounded_download(
    url: str,
    destination: str | Path,
    *,
    max_bytes: int,
    timeout: float,
    headers: Mapping[str, str] | None = None,
    proxy_url: str = "",
    resolver: Callable[..., Iterable[tuple]] = socket.getaddrinfo,
    opener=None,
) -> tuple[str, int]:
    """Stream one no-redirect public response into a new private regular file."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise BrowserMediaSafetyError("download byte limit is invalid")
    target_identity = _resolve_public_target(url, resolver=resolver)
    validated = target_identity.url
    if proxy_url and opener is None:
        raise BrowserMediaSafetyError(
            "pinned media downloads do not permit an upstream proxy"
        )
    target = Path(destination)
    descriptor = None
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            raise BrowserMediaSafetyError("download destination is not owner-safe")
        request = Request(validated, headers=dict(headers or {}), method="GET")
        response_context = (
            opener.open(request, timeout=float(timeout))
            if opener is not None
            else _open_pinned_response(
                target_identity,
                timeout=float(timeout),
                headers=dict(headers or {}),
            )
        )
        with response_context as response:
            final_url = str(response.geturl())
            if final_url != validated:
                raise BrowserMediaSafetyError("media download redirect was refused")
            advertised = response.headers.get("content-length")
            if advertised is not None:
                try:
                    advertised_size = int(advertised)
                except (TypeError, ValueError) as exc:
                    raise BrowserMediaSafetyError(
                        "media response has an invalid Content-Length"
                    ) from exc
                if not 0 <= advertised_size <= max_bytes:
                    raise BrowserMediaSafetyError("media response exceeds its byte limit")
            content_type = str(response.headers.get("content-type") or "")[:256]
            total = 0
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise BrowserMediaSafetyError("media response exceeds its byte limit")
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise BrowserMediaSafetyError("media destination write failed")
                    remaining = remaining[written:]
        if total == 0:
            raise BrowserMediaSafetyError("media response was empty")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        return content_type, total
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise


def scrub_helper_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Remove compute authority and every inherited proxy from CPU helpers."""

    environment = dict(source)
    for key in tuple(environment):
        upper = key.upper()
        if (
            upper.startswith(("AEON_FLEET", "FLEET_"))
            or upper in {
                "ALL_PROXY",
                "AEON_BROWSER_PROXY",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "NO_PROXY",
                "CUDA_VISIBLE_DEVICES",
                "GPU_AGENT_CLAIM_ID",
                "GPU_MEM_LIMIT_GB",
                "HIP_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            }
        ):
            environment.pop(key, None)
    environment.update(
        {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "CUDA_VISIBLE_DEVICES": "void",
            "GPU_DEVICE_ORDINAL": "-1",
            "HIP_VISIBLE_DEVICES": "-1",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "ROCR_VISIBLE_DEVICES": "-1",
        }
    )
    return environment


async def _stderr_tail(stream, limit: int = HELPER_STDERR_TAIL_BYTES) -> str:
    tail = bytearray()
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        tail.extend(chunk)
        if len(tail) > limit:
            del tail[:-limit]
    return bytes(tail).decode("utf-8", "replace")


async def terminate_exact_process_group(
    process,
    *,
    grace_seconds: float = HELPER_TERMINATION_GRACE_SECONDS,
) -> None:
    """Terminate then kill only the new-session group led by ``process.pid``."""

    if process.returncode is not None:
        await process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return
    except asyncio.TimeoutError:
        pass
    if process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await process.wait()


async def run_cpu_helper(
    command: list[str],
    *,
    timeout: float,
    environment: Mapping[str, str],
) -> tuple[int, str]:
    """Run a fixed CPU helper in a new process group with bounded diagnostics."""

    if not command or not all(isinstance(item, str) and item for item in command):
        raise BrowserMediaSafetyError("browser helper command is invalid")
    process = await asyncio.create_subprocess_exec(
        "/usr/bin/nice",
        "-n",
        "19",
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        env=scrub_helper_environment(environment),
        start_new_session=True,
    )
    stderr_task = asyncio.create_task(_stderr_tail(process.stderr))
    try:
        await asyncio.wait_for(process.wait(), timeout=float(timeout))
        return process.returncode, await stderr_task
    except asyncio.TimeoutError:
        await terminate_exact_process_group(process)
        return 124, f"timed out after {timeout:g}s; " + (await stderr_task)
    except BaseException:
        await terminate_exact_process_group(process)
        await stderr_task
        raise
