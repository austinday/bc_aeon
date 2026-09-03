"""Authenticated loopback gateway from OpenCode to one Fleet service ticket."""

from __future__ import annotations

import hmac
import http.client
import json
import secrets
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from aeon.core.fleet_backend import FleetBackendError, validate_loopback_endpoint
from aeon.core.model_identity import wire_model_for_runtime_profiles


MAX_REQUEST_BYTES = 48 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 128 * 1024 * 1024
class FleetModelProxy:
    """Hold no allocation itself; gate HTTP through an existing broker session.

    ``BrokerServiceSession`` remains the only owner of admission, renewal,
    endpoint promotion, and release.  This gateway merely ensures that every
    OpenCode request crosses ``ensure_ready`` immediately before transport.
    """

    def __init__(self, fleet_session: Any) -> None:
        self._session = fleet_session
        self._lock = threading.RLock()
        self._endpoint = validate_loopback_endpoint(fleet_session.endpoint)
        self._wire_model = wire_model_for_runtime_profiles(
            fleet_session.runtime_profiles
        )
        self.token = secrets.token_urlsafe(32)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._activity_lock = threading.RLock()
        self._active_handlers: set[threading.Thread] = set()
        self._active_connections: set[http.client.HTTPConnection] = set()
        self._active_upstream_sockets: set[socket.socket] = set()
        self._active_responses: set[http.client.HTTPResponse] = set()
        self._active_downstreams: set[socket.socket] = set()
        self._accepting = True
        self._permanently_closed = False
        fleet_session.set_endpoint_change_handler(self.rebind)

    @property
    def wire_model(self) -> str:
        with self._lock:
            return self._wire_model

    @property
    def base_url(self) -> str:
        server = self._server
        if server is None:
            raise RuntimeError("Fleet model gateway has not started")
        return f"http://127.0.0.1:{server.server_port}/v1"

    def rebind(self, endpoint: str, runtime_profiles: tuple[str, ...]) -> None:
        next_endpoint = validate_loopback_endpoint(endpoint)
        next_model = wire_model_for_runtime_profiles(runtime_profiles)
        with self._lock:
            self._endpoint = next_endpoint
            self._wire_model = next_model

    def _binding(self) -> tuple[str, str]:
        self._session.ensure_ready()
        with self._lock:
            return self._endpoint, self._wire_model

    def _enter_handler(self, downstream: socket.socket) -> bool:
        with self._activity_lock:
            if not self._accepting or self._permanently_closed:
                return False
            self._active_handlers.add(threading.current_thread())
            self._active_downstreams.add(downstream)
            return True

    def _leave_handler(self, downstream: socket.socket) -> None:
        with self._activity_lock:
            self._active_handlers.discard(threading.current_thread())
            self._active_downstreams.discard(downstream)

    def _track_connection(self, connection: http.client.HTTPConnection) -> None:
        with self._activity_lock:
            if not self._accepting or self._permanently_closed:
                connection.close()
                raise RuntimeError("OpenCode model turn was cancelled")
            self._active_connections.add(connection)

    def _track_response(self, response: http.client.HTTPResponse) -> None:
        with self._activity_lock:
            if not self._accepting or self._permanently_closed:
                response.close()
                raise RuntimeError("OpenCode model turn was cancelled")
            self._active_responses.add(response)

    def _track_upstream_socket(self, upstream: socket.socket) -> None:
        with self._activity_lock:
            if not self._accepting or self._permanently_closed:
                try:
                    upstream.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                upstream.close()
                raise RuntimeError("OpenCode model turn was cancelled")
            self._active_upstream_sockets.add(upstream)

    def _untrack_upstream(
        self,
        connection: http.client.HTTPConnection | None,
        response: http.client.HTTPResponse | None,
        upstream: socket.socket | None,
    ) -> None:
        with self._activity_lock:
            if response is not None:
                self._active_responses.discard(response)
            if connection is not None:
                self._active_connections.discard(connection)
            if upstream is not None:
                self._active_upstream_sockets.discard(upstream)

    @staticmethod
    def _upstream_connection(endpoint: str) -> tuple[http.client.HTTPConnection, str]:
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None:
            raise ValueError("Fleet model endpoint is not an exact loopback HTTP origin")
        base_path = parsed.path.rstrip("/")
        return (
            http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10),
            base_path + "/chat/completions",
        )

    def start(self) -> str:
        if self._server is not None:
            raise RuntimeError("Fleet model gateway is already running")
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _authorized(self) -> bool:
                expected = f"Bearer {owner.token}"
                supplied = self.headers.get("authorization", "")
                return hmac.compare_digest(supplied, expected)

            def _json_error(self, status: int, message: str) -> None:
                body = json.dumps(
                    {"error": {"message": message, "type": "nexus_gateway_error"}},
                    separators=(",", ":"),
                ).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
                if not owner._enter_handler(self.connection):
                    self._json_error(503, "OpenCode model turn is closing")
                    return
                try:
                    self._handle_get()
                finally:
                    owner._leave_handler(self.connection)

            def _handle_get(self) -> None:
                if not self._authorized():
                    self._json_error(401, "Local model capability required")
                    return
                if self.path != "/v1/models":
                    self._json_error(404, "Unknown local model route")
                    return
                try:
                    _endpoint, model = owner._binding()
                except Exception:
                    self._json_error(503, "Fleet model is not ready")
                    return
                payload = json.dumps(
                    {"object": "list", "data": [{"id": model, "object": "model"}]},
                    separators=(",", ":"),
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                self.close_connection = True

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
                if not owner._enter_handler(self.connection):
                    self._json_error(503, "OpenCode model turn is closing")
                    return
                try:
                    self._handle_post()
                finally:
                    owner._leave_handler(self.connection)

            def _handle_post(self) -> None:
                if not self._authorized():
                    self._json_error(401, "Local model capability required")
                    return
                if self.path != "/v1/chat/completions":
                    self._json_error(404, "Unknown local model route")
                    return
                try:
                    length = int(self.headers.get("content-length", ""))
                except (TypeError, ValueError):
                    self._json_error(411, "A bounded Content-Length is required")
                    return
                if not 1 <= length <= MAX_REQUEST_BYTES:
                    self._json_error(413, "Local model request exceeds its size limit")
                    return
                raw = self.rfile.read(length)
                try:
                    document = json.loads(raw.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError):
                    self._json_error(400, "Local model request is not valid JSON")
                    return
                if not isinstance(document, dict):
                    self._json_error(400, "Local model request must be an object")
                    return
                try:
                    endpoint, model = owner._binding()
                except (FleetBackendError, ValueError, RuntimeError):
                    self._json_error(503, "Fleet model is not ready")
                    return
                document["model"] = model
                body = json.dumps(document, separators=(",", ":")).encode("utf-8")
                connection: http.client.HTTPConnection | None = None
                response: http.client.HTTPResponse | None = None
                upstream_socket: socket.socket | None = None
                downstream_started = False
                try:
                    connection, route = owner._upstream_connection(endpoint)
                    owner._track_connection(connection)
                    connection.request(
                        "POST",
                        route,
                        body=body,
                        headers={
                            "Authorization": "Bearer no-key-needed",
                            "Content-Type": "application/json",
                            "Content-Length": str(len(body)),
                            "Connection": "close",
                        },
                    )
                    upstream_socket = connection.sock
                    if upstream_socket is None:
                        raise RuntimeError("Fleet model transport has no socket")
                    owner._track_upstream_socket(upstream_socket)
                    response = connection.getresponse()
                    owner._track_response(response)
                    upstream_socket.settimeout(600)
                    if 300 <= response.status < 400:
                        self._json_error(502, "Fleet model attempted a redirect")
                        return
                    response_limit = (
                        MAX_ERROR_BYTES
                        if response.status >= 400
                        else MAX_RESPONSE_BYTES
                    )
                    content_length = response.getheader("content-length")
                    if content_length:
                        try:
                            declared_length = int(content_length)
                        except ValueError:
                            self._json_error(502, "Fleet model returned invalid framing")
                            return
                        if not 0 <= declared_length <= response_limit:
                            self._json_error(502, "Fleet model response exceeded its size limit")
                            return
                    self.send_response(response.status)
                    upstream_type = str(
                        response.getheader("content-type", "application/json")
                    ).lower()
                    content_type = (
                        "text/event-stream"
                        if upstream_type.startswith("text/event-stream")
                        else "application/json"
                    )
                    self.send_header("Content-Type", content_type[:200])
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    downstream_started = True
                    sent = 0
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        remaining = response_limit - sent
                        if remaining <= 0:
                            break
                        chunk = chunk[:remaining]
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        sent += len(chunk)
                    self.close_connection = True
                except (
                    BrokenPipeError,
                    ConnectionResetError,
                    TimeoutError,
                    OSError,
                    http.client.HTTPException,
                    RuntimeError,
                ):
                    self.close_connection = True
                    if not downstream_started:
                        try:
                            self._json_error(502, "Fleet model transport failed")
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            self.close_connection = True
                finally:
                    owner._untrack_upstream(connection, response, upstream_socket)
                    if response is not None:
                        response.close()
                    if connection is not None:
                        connection.close()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="aeon-opencode-model-gateway",
            daemon=True,
        )
        self._thread.start()
        return self.base_url

    def _drain_active(self, *, reopen: bool) -> None:
        with self._activity_lock:
            self._accepting = False
            connections = list(self._active_connections)
            upstream_sockets = list(self._active_upstream_sockets)
            downstreams = list(self._active_downstreams)
            handlers = list(self._active_handlers)
        for connection in connections:
            try:
                if connection.sock is not None:
                    connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except Exception:
                pass
        for upstream in upstream_sockets:
            try:
                upstream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                upstream.close()
            except OSError:
                pass
        # Do not call HTTPResponse.close() from this thread while its handler is
        # inside BufferedReader.read(): CPython's buffered-I/O lock can make
        # close wait for the socket timeout. Shutting down the tracked transport
        # wakes that read; the joined handler then closes and unregisters its
        # own response in ``finally``.
        for downstream in downstreams:
            try:
                downstream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                downstream.close()
            except OSError:
                pass
        deadline = time.monotonic() + 10.0
        for handler in handlers:
            if handler is threading.current_thread():
                continue
            handler.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._activity_lock:
            alive = [thread for thread in self._active_handlers if thread.is_alive()]
            if (
                alive
                or self._active_connections
                or self._active_upstream_sockets
                or self._active_responses
                or self._active_downstreams
            ):
                raise RuntimeError(
                    "OpenCode model gateway could not prove active request cancellation"
                )
            if reopen and not self._permanently_closed:
                self._accepting = True

    def cancel_active_turn(self) -> None:
        """Close and join every request from the just-finished OpenCode child."""

        self._drain_active(reopen=True)

    def close(self) -> None:
        with self._activity_lock:
            self._permanently_closed = True
            self._accepting = False
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        self._drain_active(reopen=False)
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=10)
            if thread.is_alive():
                raise RuntimeError("OpenCode model gateway server thread did not stop")


__all__ = (
    "FleetModelProxy",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
)
