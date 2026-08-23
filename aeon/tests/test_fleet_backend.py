from __future__ import annotations

import os
import socket
import tempfile
import unittest
from pathlib import Path

from aeon.core.fleet_backend import (
    BrokerServiceSession,
    FleetBackendError,
    select_compute_backend,
    validate_loopback_endpoint,
)


class _FakeClient:
    def __init__(self, status=None):
        self._status = status or {}
        self.released = []
        self.renewed = []
        self.service_calls = 0

    def status(self):
        return self._status

    def acquire_service(self, **_kwargs):
        return {
            "ticket_id": "fd-0123456789abcdef0123456789abcdef",
            "state": "active",
            "compute_state": "waiting_for_compute",
            "endpoint": None,
        }

    def service_status(self, _ticket_id):
        self.service_calls += 1
        return {
            "ticket_id": "fd-test-ticket",
            "state": "active",
            "compute_state": "ready",
            "endpoint": "http://127.0.0.1:8033/v1",
        }

    def renew_service(self, ticket_id, *, ttl_seconds):
        self.renewed.append((ticket_id, ttl_seconds))
        return {}

    def release_service(self, ticket_id):
        self.released.append(ticket_id)
        return {}


class FleetBackendSelectionTests(unittest.TestCase):
    def test_auto_uses_coordinator_when_broker_is_not_installed(self):
        backend, reason = select_compute_backend(
            environ={"AEON_FLEET_SOCKET": "/definitely/absent/broker.sock"}
        )
        self.assertEqual(backend, "coordinator")
        self.assertIn("not installed", reason)

    def test_required_broker_fails_when_socket_is_absent(self):
        with self.assertRaises(FleetBackendError):
            select_compute_backend(
                environ={
                    "AEON_COMPUTE_BACKEND": "broker",
                    "AEON_FLEET_SOCKET": "/definitely/absent/broker.sock",
                }
            )

    def test_enabled_aeon_profile_selects_broker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            path = root / "broker.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(path))
                os.chmod(path, 0o600)
                client = _FakeClient(
                    {
                        "profiles": [
                            {
                                "profile_id": "aeon-qwen38-standard",
                                "enabled": True,
                                "mode": "service",
                                "project": "aeon",
                            }
                        ],
                        "runtimes": [],
                    }
                )
                backend, _reason = select_compute_backend(
                    environ={"AEON_FLEET_SOCKET": str(path)}, client=client
                )
                self.assertEqual(backend, "broker")
            finally:
                listener.close()


class BrokerServiceSessionTests(unittest.TestCase):
    def test_waits_for_ready_loopback_endpoint_and_releases_ticket(self):
        client = _FakeClient()
        session = BrokerServiceSession(client=client, consumer="aeon/test", sleep=lambda _n: None)
        self.assertEqual(session.start(), "http://127.0.0.1:8033/v1")
        session.ensure_ready()
        session.close()
        self.assertEqual(client.released, ["fd-0123456789abcdef0123456789abcdef"])

    def test_endpoint_validation_rejects_non_loopback_and_credentials(self):
        for endpoint in (
            "https://127.0.0.1:8033/v1",
            "http://192.168.0.177:8033/v1",
            "http://user@127.0.0.1:8033/v1",
            "http://127.0.0.1:8033/admin",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(FleetBackendError):
                validate_loopback_endpoint(endpoint)


if __name__ == "__main__":
    unittest.main()
