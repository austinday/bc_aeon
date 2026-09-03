from __future__ import annotations

import copy
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from aeon.core.fleet_backend import (
    BENCHMARK_COMPUTE_STATUS_FD_ENV,
    BrokerServiceSession,
    FleetBackendError,
    FleetBrokerClient,
    FleetBrokerUnavailable,
    select_compute_backend,
    validate_loopback_endpoint,
)


class _FakeClient:
    def __init__(self, status=None):
        self._status = status or {}
        self.released = []
        self.renewed = []
        self.acquired = []
        self.service_calls = 0
        self.endpoint = "http://127.0.0.1:8033/v1"
        self.ticket_id = "fd-0123456789abcdef0123456789abcdef"
        self.profile_id = "aeon-qwen38-standard"
        self.service_id = "aeon-qwen38-standard"
        self.consumer = "aeon/test"

    def _snapshot(
        self, *, state, compute_state, endpoint, runtime_profiles=None
    ):
        if runtime_profiles is None:
            runtime_profiles = (
                ["aeon-qwen38-compact-workers"]
                if compute_state == "ready"
                else []
            )
        return {
            "ticket_id": self.ticket_id,
            "profile_id": self.profile_id,
            "service_id": self.service_id,
            "consumer": self.consumer,
            "state": state,
            "compute_state": compute_state,
            "endpoint": endpoint,
            "runtime_profiles": runtime_profiles,
        }

    def status(self):
        return self._status

    def acquire_service(self, **kwargs):
        self.acquired.append(kwargs)
        self.consumer = kwargs["consumer"]
        return self._snapshot(
            state="active", compute_state="waiting_for_compute", endpoint=None
        )

    def service_status(self, _ticket_id):
        self.service_calls += 1
        return self._snapshot(
            state="active", compute_state="ready", endpoint=self.endpoint
        )

    def renew_service(self, ticket_id, *, ttl_seconds):
        self.renewed.append((ticket_id, ttl_seconds))
        response = self._snapshot(
            state="active", compute_state="ready", endpoint=self.endpoint
        )
        response["ticket_id"] = ticket_id
        return response

    def release_service(self, ticket_id):
        self.released.append(ticket_id)
        response = self._snapshot(
            state="released", compute_state="inactive", endpoint=None
        )
        response["ticket_id"] = ticket_id
        return response


class FleetBackendSelectionTests(unittest.TestCase):
    @staticmethod
    def _variant(profile_id, *, enabled=True):
        return {
            "profile_id": profile_id,
            "enabled": enabled,
            "mode": "service",
            "project": "aeon",
            "purpose": "Reviewed Qwen service",
            "service_id": "aeon-qwen38-standard",
            "request_routing": "least_busy",
        }

    @classmethod
    def _logical_status(cls, *, include_disabled_exact=False):
        variant_ids = ["aeon-qwen38-local-177", "aeon-qwen38-compact-180"]
        profiles = [cls._variant(profile_id) for profile_id in variant_ids]
        if include_disabled_exact:
            profiles.insert(
                0,
                {
                    **cls._variant("aeon-qwen38-standard", enabled=False),
                    # A retired concrete profile may retain its historical
                    # ticket-affinity routing while the current logical release
                    # uses the stable least-busy router.
                    "request_routing": "ticket_affinity",
                },
            )
        return {
            "profiles": profiles,
            "services": [
                {
                    "service_id": "aeon-qwen38-standard",
                    "project": "aeon",
                    "purpose": "Reviewed Qwen service",
                    "variant_count": len(variant_ids),
                    "request_routing": "least_busy",
                    "variants": variant_ids,
                }
            ],
            "runtimes": [
                {
                    "profile_id": "aeon-qwen38-compact-180",
                    "state": "ready",
                }
            ],
        }


    def _select(self, status):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.chmod(root, 0o700)
            path = root / "broker.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(path))
                os.chmod(path, 0o600)
                return select_compute_backend(
                    environ={"AEON_FLEET_SOCKET": str(path)},
                    client=_FakeClient(status),
                )
            finally:
                listener.close()

    def test_service_policy_client_uses_fixed_owner_only_api_contract(self):
        client = FleetBrokerClient("/unused/broker.sock")
        with patch.object(client, "_request", return_value={"requested_replicas": 2}) as request:
            self.assertEqual(
                client.service_policy("aeon-qwen38-standard"),
                {"requested_replicas": 2},
            )
            request.assert_called_once_with(
                "GET", "/v1/service-policies/aeon-qwen38-standard"
            )

        with patch.object(client, "_request", return_value={"requested_replicas": 2}) as request:
            client.set_service_policy(
                "aeon-qwen38-standard", requested_replicas=2, actor="nexus"
            )
            request.assert_called_once_with(
                "PUT",
                "/v1/service-policies/aeon-qwen38-standard",
                {"requested_replicas": 2, "actor": "nexus"},
            )

        with patch.object(client, "_request", return_value={}) as request:
            client.set_service_policy(
                "aeon-qwen38-standard",
                requested_replicas=2,
                preferred_model="qwen38-flash-next",
                actor="nexus",
            )
            request.assert_called_once_with(
                "PUT",
                "/v1/service-policies/aeon-qwen38-standard",
                {
                    "requested_replicas": 2,
                    "actor": "nexus",
                    "preferred_model": "qwen38-flash-next",
                },
            )

        with self.assertRaises(FleetBackendError):
            client.set_service_policy(
                "aeon-qwen38-standard", requested_replicas=True, actor="nexus"
            )

    def test_batch_client_uses_fixed_standard_job_api(self):
        client = FleetBrokerClient("/unused/broker.sock")
        expected = {"job_id": "fj-" + "a" * 32}
        with patch.object(client, "_request", return_value=expected) as request:
            self.assertEqual(
                client.submit_job(
                    profile="aeon-qwen38-dflash-adapt",
                    project="aeon-dflash-adapt",
                    idempotency_key="aeon-request-1",
                    payload={"run_mode": "adapt-v1"},
                ),
                expected,
            )
            request.assert_called_once_with(
                "POST",
                "/v1/jobs",
                {
                    "profile": "aeon-qwen38-dflash-adapt",
                    "project": "aeon-dflash-adapt",
                    "idempotency_key": "aeon-request-1",
                    "payload": {"run_mode": "adapt-v1"},
                },
            )

        job_id = "fj-" + "b" * 32
        with patch.object(client, "_request", return_value={"job_id": job_id}) as request:
            self.assertEqual(client.job_status(job_id), {"job_id": job_id})
            request.assert_called_once_with("GET", f"/v1/jobs/{job_id}")

        with self.assertRaises(FleetBackendError):
            client.job_status("not-a-job")

    def test_auto_fails_closed_when_broker_is_not_installed(self):
        with self.assertRaises(FleetBackendError):
            select_compute_backend(
                environ={"AEON_FLEET_SOCKET": "/definitely/absent/broker.sock"}
            )

    def test_direct_coordinator_backend_is_rejected(self):
        with self.assertRaises(FleetBackendError):
            select_compute_backend(environ={"AEON_COMPUTE_BACKEND": "coordinator"})

    def test_required_broker_fails_when_socket_is_absent(self):
        with self.assertRaises(FleetBackendError):
            select_compute_backend(
                environ={
                    "AEON_COMPUTE_BACKEND": "broker",
                    "AEON_FLEET_SOCKET": "/definitely/absent/broker.sock",
                }
            )

    def test_enabled_aeon_profile_selects_broker(self):
        backend, _reason = self._select(
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
        self.assertEqual(backend, "broker")

    def test_enabled_logical_service_selects_without_exact_profile(self):
        backend, reason = self._select(self._logical_status())

        self.assertEqual(backend, "broker")
        self.assertIn("enabled logical service aeon-qwen38-standard", reason)
        self.assertIn("2 variants", reason)

    def test_logical_service_accepts_lane_specific_purpose_descriptions(self):
        status = self._logical_status()
        status["profiles"][0]["purpose"] = "Primary local inference lane"
        status["profiles"][1]["purpose"] = "Compact worker fallback lane"

        backend, _reason = self._select(status)

        self.assertEqual(backend, "broker")

    def test_disabled_historical_exact_profile_does_not_hide_logical_service(self):
        backend, reason = self._select(
            self._logical_status(include_disabled_exact=True)
        )

        self.assertEqual(backend, "broker")
        self.assertIn("logical service", reason)

    def test_known_retired_runtime_is_compatible_with_enabled_logical_service(self):
        status = self._logical_status(include_disabled_exact=True)
        status["runtimes"].append(
            {"profile_id": "aeon-qwen38-standard", "state": "ready"}
        )

        backend, _reason = self._select(status)

        self.assertEqual(backend, "broker")

    def test_logical_service_rejects_malformed_or_inconsistent_registry(self):
        cases = {}

        duplicate_service = self._logical_status()
        duplicate_service["services"].append(
            copy.deepcopy(duplicate_service["services"][0])
        )
        cases["duplicate service"] = duplicate_service

        wrong_count = self._logical_status()
        wrong_count["services"][0]["variant_count"] = 3
        cases["wrong variant count"] = wrong_count

        duplicate_variant = self._logical_status()
        duplicate_variant["services"][0]["variants"] = [
            "aeon-qwen38-local-177",
            "aeon-qwen38-local-177",
        ]
        cases["duplicate variant"] = duplicate_variant

        unknown_variant = self._logical_status()
        unknown_variant["services"][0]["variants"][1] = "aeon-qwen38-unknown"
        cases["unknown variant"] = unknown_variant

        wrong_routing = self._logical_status()
        wrong_routing["profiles"][0]["request_routing"] = "ticket_affinity"
        cases["routing mismatch"] = wrong_routing

        empty_variant_purpose = self._logical_status()
        empty_variant_purpose["profiles"][0]["purpose"] = "  "
        cases["empty variant purpose"] = empty_variant_purpose

        extra_enabled = self._logical_status()
        extra_enabled["profiles"].append(self._variant("aeon-qwen38-extra"))
        cases["unadvertised enabled variant"] = extra_enabled

        malformed_services = self._logical_status()
        malformed_services["services"] = {"not": "a list"}
        cases["malformed services"] = malformed_services

        for label, status in cases.items():
            with self.subTest(label=label), self.assertRaises(FleetBackendError):
                self._select(status)

    def test_logical_service_rejects_unknown_active_runtime_for_service_id(self):
        status = self._logical_status()
        status["runtimes"].append(
            {"profile_id": "aeon-qwen38-standard", "state": "starting"}
        )

        with self.assertRaises(FleetBackendError):
            self._select(status)

    def test_duplicate_profile_fails_closed_before_logical_selection(self):
        status = self._logical_status()
        status["profiles"].append(copy.deepcopy(status["profiles"][0]))

        with self.assertRaises(FleetBackendError):
            self._select(status)


class BrokerServiceSessionTests(unittest.TestCase):
    @staticmethod
    def _bind_session(session, client):
        client.consumer = session.consumer
        session._ticket_id = client.ticket_id
        session._ticket_profile_id = client.profile_id
        session._service_id = client.service_id

    def test_publishes_waiting_ready_and_idle_compute_presence(self):
        class Presence:
            def __init__(self):
                self.updates = []

            def update_compute(self, **values):
                self.updates.append(values)

        presence = Presence()
        client = _FakeClient()
        session = BrokerServiceSession(
            client=client, consumer="aeon/presence", sleep=lambda _n: None
        )
        with patch(
            "aeon.core.presence.get_active_presence", return_value=presence
        ):
            session.start()
            session.close()

        self.assertEqual(
            [update["state"] for update in presence.updates],
            ["waiting_for_compute", "allocated", "idle"],
        )
        self.assertEqual(
            [update["profile"] for update in presence.updates],
            [
                "aeon-qwen38-standard",
                "aeon-qwen38-compact-workers",
                "aeon-qwen38-standard",
            ],
        )

    def test_benchmark_status_pipe_tracks_one_durable_demand(self):
        client = _FakeClient()
        session = BrokerServiceSession(
            client=client,
            consumer="aeon/benchmark-one-demand",
            sleep=lambda _delay: None,
        )
        read_descriptor, write_descriptor = os.pipe()
        try:
            with patch.dict(
                os.environ,
                {BENCHMARK_COMPUTE_STATUS_FD_ENV: str(write_descriptor)},
            ):
                session.start()
                session.close()
            os.close(write_descriptor)
            write_descriptor = -1
            states = os.read(read_descriptor, 4096).decode("ascii").splitlines()
        finally:
            os.close(read_descriptor)
            if write_descriptor >= 0:
                os.close(write_descriptor)

        self.assertEqual(states, ["waiting_for_compute", "allocated", "idle"])
        self.assertEqual(len(client.acquired), 1)
        self.assertEqual(client.released, [client.ticket_id])

    def test_ticket_renewal_refreshes_allocated_presence(self):
        class Presence:
            def __init__(self):
                self.updates = []

            def update_compute(self, **values):
                self.updates.append(values)

        presence = Presence()
        client = _FakeClient()
        session = BrokerServiceSession(client=client, consumer="aeon/presence-renewal")
        self._bind_session(session, client)
        session.endpoint = "http://127.0.0.1:8033/v1"
        session._runtime_profiles = ("aeon-qwen38-compact-workers",)

        with patch(
            "aeon.core.presence.get_active_presence", return_value=presence
        ):
            session._renew_once()

        self.assertEqual(client.renewed, [(session.ticket_id, session.ttl_seconds)])
        self.assertEqual(
            presence.updates,
            [
                {
                    "state": "allocated",
                    "profile": "aeon-qwen38-compact-workers",
                    "summary": (
                        "Ready pool: Qwen3.8-27B RTX 5000 fallback via "
                        "aeon-qwen38-compact-workers"
                    ),
                }
            ],
        )

    def test_admission_wait_renewal_refreshes_local_ticket_deadline(self):
        client = _FakeClient()
        now = [0.0]

        def service_status(_ticket_id):
            now[0] = 50.0
            return client._snapshot(
                state="active", compute_state="waiting_for_compute", endpoint=None
            )

        client.service_status = service_status
        session = BrokerServiceSession(
            client=client,
            consumer="aeon/admission-deadline",
            ttl_seconds=30,
            clock=lambda: now[0],
            sleep=lambda _delay: None,
        )

        session.start()

        self.assertEqual(session._renew_deadline, 80.0)
        session.close()

    def test_renew_loop_survives_brief_broker_restart_within_ticket_lease(self):
        class Presence:
            def __init__(self):
                self.updates = []

            def update_compute(self, **values):
                self.updates.append(values)

        presence = Presence()
        client = _FakeClient()
        now = [0.0]
        waits = []

        def renew_wait(delay):
            waits.append(delay)
            now[0] += delay
            # Stop after the recovered renewal returns to its normal interval.
            return len(waits) >= 3

        responses = [
            FleetBrokerUnavailable("broker restarting"),
            "success",
        ]

        def renew_service(_ticket_id, *, ttl_seconds):
            response = responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return client._snapshot(
                state="active", compute_state="ready", endpoint=client.endpoint
            )

        client.renew_service = renew_service
        session = BrokerServiceSession(
            client=client,
            consumer="aeon/restart-recovery",
            ttl_seconds=30,
            clock=lambda: now[0],
            renew_wait=renew_wait,
        )
        self._bind_session(session, client)
        session.endpoint = client.endpoint
        session._renew_deadline = 30.0

        with patch(
            "aeon.core.presence.get_active_presence", return_value=presence
        ):
            session._renew_loop()

        self.assertIsNone(session._renew_error)
        self.assertEqual(waits, [15.0, 1.0, 15.0])
        self.assertEqual(session._renew_deadline, 46.0)
        self.assertEqual(
            [update["state"] for update in presence.updates],
            ["waiting_for_compute", "allocated"],
        )
        self.assertIn("reconnecting", presence.updates[0]["summary"])

    def test_renew_loop_fails_closed_when_broker_outage_reaches_lease_deadline(self):
        class Presence:
            def __init__(self):
                self.updates = []

            def update_compute(self, **values):
                self.updates.append(values)

        presence = Presence()
        client = _FakeClient()
        now = [0.0]

        def renew_wait(delay):
            now[0] += delay
            return False

        client.renew_service = Mock(
            side_effect=FleetBrokerUnavailable("broker remains unavailable")
        )
        session = BrokerServiceSession(
            client=client,
            consumer="aeon/restart-deadline",
            ttl_seconds=18,
            clock=lambda: now[0],
            renew_wait=renew_wait,
        )
        self._bind_session(session, client)
        session.endpoint = client.endpoint
        session._renew_deadline = 18.0

        with patch(
            "aeon.core.presence.get_active_presence", return_value=presence
        ):
            session._renew_loop()

        self.assertIsInstance(session._renew_error, FleetBrokerUnavailable)
        self.assertEqual(now[0], 18.0)
        self.assertEqual(presence.updates[-1]["state"], "unavailable")

    def test_renew_loop_does_not_retry_identity_drift(self):
        client = _FakeClient()
        now = [0.0]
        waits = []

        def renew_wait(delay):
            waits.append(delay)
            now[0] += delay
            return False

        invalid = client._snapshot(
            state="active", compute_state="ready", endpoint=client.endpoint
        )
        invalid["consumer"] = "aeon/other"
        client.renew_service = Mock(return_value=invalid)
        session = BrokerServiceSession(
            client=client,
            consumer="aeon/identity-drift",
            ttl_seconds=30,
            clock=lambda: now[0],
            renew_wait=renew_wait,
        )
        self._bind_session(session, client)
        session.endpoint = client.endpoint
        session._renew_deadline = 30.0

        session._renew_loop()

        self.assertIsInstance(session._renew_error, FleetBackendError)
        self.assertNotIsInstance(session._renew_error, FleetBrokerUnavailable)
        self.assertEqual(client.renew_service.call_count, 1)
        self.assertEqual(waits, [15.0])

    def test_foreground_readiness_survives_brief_broker_restart_within_ticket_lease(self):
        presence = Mock()
        client = _FakeClient()
        now = [0.0]
        waits = []

        def sleep(delay):
            waits.append(delay)
            now[0] += delay

        session = BrokerServiceSession(
            client=client,
            consumer="aeon/continuous-foreground-recovery",
            ttl_seconds=30,
            clock=lambda: now[0],
            sleep=sleep,
        )
        self._bind_session(session, client)
        session.endpoint = client.endpoint
        session._runtime_profiles = ("aeon-qwen38-compact-workers",)
        session._renew_deadline = 30.0
        ready = client._snapshot(
            state="active", compute_state="ready", endpoint=client.endpoint
        )
        client.service_status = Mock(
            side_effect=[FleetBrokerUnavailable("broker restarting"), ready]
        )

        with patch(
            "aeon.core.presence.get_active_presence", return_value=presence
        ):
            session.ensure_ready()

        self.assertEqual(waits, [1.0])
        self.assertEqual(client.service_status.call_count, 2)
        self.assertEqual(client.renewed, [])
        self.assertEqual(
            [call.kwargs["state"] for call in presence.update_compute.call_args_list],
            ["waiting_for_compute", "allocated"],
        )
        self.assertIn(
            "reconnecting", presence.update_compute.call_args_list[0].kwargs["summary"]
        )

    def test_foreground_readiness_retries_transient_wait_renewal_failure(self):
        client = _FakeClient()
        now = [0.0]
        waits = []

        def sleep(delay):
            waits.append(delay)
            now[0] += delay

        session = BrokerServiceSession(
            client=client,
            consumer="aeon/continuous-renewal-recovery",
            ttl_seconds=30,
            clock=lambda: now[0],
            sleep=sleep,
        )
        self._bind_session(session, client)
        session._renew_deadline = 30.0
        waiting = client._snapshot(
            state="active", compute_state="waiting_for_compute", endpoint=None
        )
        ready = client._snapshot(
            state="active", compute_state="ready", endpoint=client.endpoint
        )
        client.service_status = Mock(side_effect=[waiting, ready])
        client.renew_service = Mock(
            side_effect=FleetBrokerUnavailable("broker restarting during renewal")
        )

        endpoint, profiles = session._wait_ready()

        self.assertEqual(endpoint, client.endpoint)
        self.assertEqual(profiles, ("aeon-qwen38-compact-workers",))
        self.assertEqual(waits, [1.0])
        self.assertEqual(client.service_status.call_count, 2)
        self.assertEqual(client.renew_service.call_count, 1)

    def test_foreground_readiness_fails_after_broker_outage_reaches_lease_deadline(self):
        client = _FakeClient()
        now = [0.0]
        waits = []
        client.service_status = Mock(
            side_effect=FleetBrokerUnavailable("broker remains unavailable")
        )

        def sleep(delay):
            waits.append(delay)
            now[0] += delay

        session = BrokerServiceSession(
            client=client,
            consumer="aeon/continuous-restart-deadline",
            ttl_seconds=3,
            clock=lambda: now[0],
            sleep=sleep,
        )
        self._bind_session(session, client)
        session.endpoint = client.endpoint
        session._renew_deadline = 3.0

        with self.assertRaisesRegex(
            FleetBackendError, "ticket lease deadline"
        ) as raised:
            session.ensure_ready()

        self.assertIs(type(raised.exception), FleetBackendError)
        self.assertEqual(waits, [1.0, 2.0])
        self.assertEqual(now[0], 3.0)
        self.assertEqual(client.service_status.call_count, 3)

    def test_foreground_readiness_does_not_retry_identity_drift(self):
        client = _FakeClient()
        invalid = client._snapshot(
            state="active", compute_state="ready", endpoint=client.endpoint
        )
        invalid["consumer"] = "aeon/other"
        client.service_status = Mock(return_value=invalid)
        sleep = Mock()
        session = BrokerServiceSession(
            client=client,
            consumer="aeon/continuous-identity-drift",
            sleep=sleep,
        )
        self._bind_session(session, client)
        session.endpoint = client.endpoint
        session._renew_deadline = session.ttl_seconds

        with self.assertRaises(FleetBackendError):
            session.ensure_ready()

        self.assertEqual(client.service_status.call_count, 1)
        sleep.assert_not_called()

    def test_pre_upgrade_broker_ready_snapshot_is_accepted_without_mislabeling(self):
        class Presence:
            def __init__(self):
                self.updates = []

            def update_compute(self, **values):
                self.updates.append(values)

        presence = Presence()
        client = _FakeClient()

        def renew_service(_ticket_id, *, ttl_seconds):
            snapshot = client._snapshot(
                state="active", compute_state="ready", endpoint=client.endpoint
            )
            snapshot.pop("runtime_profiles")
            return snapshot

        client.renew_service = renew_service
        session = BrokerServiceSession(client=client, consumer="aeon/pre-upgrade")
        self._bind_session(session, client)
        session.endpoint = client.endpoint

        with patch(
            "aeon.core.presence.get_active_presence", return_value=presence
        ):
            session._renew_once()

        self.assertEqual(
            presence.updates[0]["summary"],
            "Qwen runtime ready; concrete profile unavailable from broker",
        )
        self.assertNotIn("Flash", presence.updates[0]["summary"])

    def test_ticket_renewal_names_mixed_flash_and_fallback_pool(self):
        class Presence:
            def __init__(self):
                self.updates = []

            def update_compute(self, **values):
                self.updates.append(values)

        presence = Presence()
        client = _FakeClient()
        profiles = [
            "aeon-qwen38-compact-workers",
            "aeon-qwen38-flash-next-177",
        ]
        client.renew_service = lambda _ticket_id, *, ttl_seconds: client._snapshot(
            state="active",
            compute_state="ready",
            endpoint=client.endpoint,
            runtime_profiles=profiles,
        )
        session = BrokerServiceSession(client=client, consumer="aeon/mixed-pool")
        self._bind_session(session, client)
        session.endpoint = client.endpoint
        session._runtime_profiles = tuple(profiles)

        with patch(
            "aeon.core.presence.get_active_presence", return_value=presence
        ):
            session._renew_once()

        self.assertEqual(
            presence.updates,
            [
                {
                    "state": "allocated",
                    "profile": ", ".join(profiles),
                    "summary": (
                        "Ready pool: Qwen3.8-Flash-Next NVFP4+MTP via "
                        "aeon-qwen38-flash-next-177; Qwen3.8-27B RTX 5000 "
                        "fallback via aeon-qwen38-compact-workers"
                    ),
                }
            ],
        )

    def test_ticket_renewal_publishes_wait_when_runtime_is_lost(self):
        class Presence:
            def __init__(self):
                self.updates = []

            def update_compute(self, **values):
                self.updates.append(values)

        presence = Presence()
        client = _FakeClient()
        session = BrokerServiceSession(client=client, consumer="aeon/presence-wait")
        self._bind_session(session, client)
        client.renew_service = lambda _ticket_id, *, ttl_seconds: client._snapshot(
            state="active", compute_state="waiting_for_compute", endpoint=None
        )
        session.endpoint = "http://127.0.0.1:8033/v1"

        with patch(
            "aeon.core.presence.get_active_presence", return_value=presence
        ):
            session._renew_once()

        self.assertEqual(presence.updates[0]["state"], "waiting_for_compute")

    def test_promoted_endpoint_rebinds_at_foreground_boundary(self):
        client = _FakeClient()
        session = BrokerServiceSession(client=client, consumer="aeon/promotion")
        self._bind_session(session, client)
        session.endpoint = client.endpoint
        rebound = []
        session.set_endpoint_change_handler(
            lambda endpoint, profiles: rebound.append((endpoint, profiles))
        )

        client.endpoint = "http://127.0.0.1:18034/v1"
        session._renew_once()

        self.assertEqual(session.endpoint, "http://127.0.0.1:8033/v1")
        self.assertEqual(rebound, [])

        session.ensure_ready()

        self.assertEqual(
            rebound,
            [("http://127.0.0.1:18034/v1", ("aeon-qwen38-compact-workers",))],
        )
        self.assertEqual(session.endpoint, "http://127.0.0.1:18034/v1")

    def test_stable_router_profile_change_rebinds_at_foreground_boundary(self):
        client = _FakeClient()
        session = BrokerServiceSession(client=client, consumer="aeon/profile-promotion")
        self._bind_session(session, client)
        session.endpoint = client.endpoint
        session._runtime_profiles = ("aeon-qwen38-compact-workers",)
        rebound = []
        session.set_endpoint_change_handler(
            lambda endpoint, profiles: rebound.append((endpoint, profiles))
        )
        client.renew_service = lambda _ticket_id, *, ttl_seconds: client._snapshot(
            state="active",
            compute_state="ready",
            endpoint=client.endpoint,
            runtime_profiles=["aeon-qwen38-flash-next-vllm-177"],
        )

        session._renew_once()

        self.assertEqual(rebound, [])
        self.assertEqual(session.runtime_profiles, ("aeon-qwen38-compact-workers",))
        session.ensure_ready()
        self.assertEqual(
            rebound,
            [
                (
                    client.endpoint,
                    ("aeon-qwen38-flash-next-vllm-177",),
                )
            ],
        )
        self.assertEqual(
            session.runtime_profiles,
            ("aeon-qwen38-flash-next-vllm-177",),
        )

    def test_waits_for_ready_loopback_endpoint_and_releases_ticket(self):
        client = _FakeClient()
        session = BrokerServiceSession(client=client, consumer="aeon/test", sleep=lambda _n: None)
        self.assertEqual(session.start(), "http://127.0.0.1:8033/v1")
        session.ensure_ready()
        self.assertEqual(
            session.close(), {"state": "released", "compute_state": "inactive"}
        )
        self.assertEqual(client.released, ["fd-0123456789abcdef0123456789abcdef"])
        self.assertIsNone(session.endpoint)

    def test_concrete_request_binds_broker_canonical_logical_service(self):
        client = _FakeClient()
        session = BrokerServiceSession(
            client=client,
            profile="aeon-qwen38-local-177",
            consumer="aeon/concrete-profile",
            sleep=lambda _n: None,
        )

        session.start()

        self.assertEqual(client.acquired[0]["profile"], "aeon-qwen38-local-177")
        self.assertEqual(session.logical_service_id, "aeon-qwen38-standard")
        self.assertEqual(session._ticket_profile_id, "aeon-qwen38-standard")
        with self.assertRaises(AttributeError):
            session.ticket_id = "fd-ffffffffffffffffffffffffffffffff"
        session.close()

    def test_unowned_post_acquire_identity_is_never_released(self):
        for field, replacement in (
            ("profile_id", "wrong-profile"),
            ("service_id", "wrong-service"),
            ("consumer", "aeon/different-consumer"),
        ):
            with self.subTest(field=field):
                client = _FakeClient()
                original_acquire = client.acquire_service

                def acquire_service(_field=field, _replacement=replacement, **kwargs):
                    result = original_acquire(**kwargs)
                    result[_field] = _replacement
                    return result

                client.acquire_service = acquire_service
                session = BrokerServiceSession(
                    client=client,
                    consumer="aeon/unowned-acquire",
                    sleep=lambda _n: None,
                )

                with self.assertRaises(FleetBackendError):
                    session.start()

                self.assertEqual(client.released, [])
                self.assertIsNone(session.ticket_id)

    def test_owned_but_invalid_post_acquire_state_releases_exact_ticket(self):
        for field, replacement in (
            ("state", "released"),
            ("compute_state", "unknown"),
            ("endpoint", "http://127.0.0.1:8033/v1"),
        ):
            with self.subTest(field=field):
                client = _FakeClient()
                original_acquire = client.acquire_service

                def acquire_service(_field=field, _replacement=replacement, **kwargs):
                    result = original_acquire(**kwargs)
                    result[_field] = _replacement
                    return result

                client.acquire_service = acquire_service
                session = BrokerServiceSession(
                    client=client,
                    consumer="aeon/invalid-acquire",
                    sleep=lambda _n: None,
                )

                with self.assertRaises(FleetBackendError):
                    session.start()

                self.assertEqual(client.released, [client.ticket_id])
                self.assertIsNone(session.ticket_id)

    def test_status_drift_releases_only_the_bound_ticket(self):
        for field, replacement in (
            ("ticket_id", "fd-ffffffffffffffffffffffffffffffff"),
            ("profile_id", "aeon-qwen38-other"),
            ("service_id", "aeon-qwen38-other"),
            ("consumer", "aeon/other"),
            ("state", "released"),
            ("compute_state", "unknown"),
            ("endpoint", "http://192.168.0.177:8033/v1"),
        ):
            with self.subTest(field=field):
                client = _FakeClient()

                def service_status(_ticket_id, _field=field, _replacement=replacement):
                    result = client._snapshot(
                        state="active", compute_state="ready", endpoint=client.endpoint
                    )
                    result[_field] = _replacement
                    return result

                client.service_status = service_status
                session = BrokerServiceSession(
                    client=client,
                    consumer="aeon/status-drift",
                    sleep=lambda _n: None,
                )

                with self.assertRaises(FleetBackendError):
                    session.start()

                self.assertEqual(client.released, [client.ticket_id])
                self.assertIsNone(session.ticket_id)

    def test_close_keeps_ticket_when_release_response_is_not_terminal_proof(self):
        client = _FakeClient()
        session = BrokerServiceSession(
            client=client, consumer="aeon/release-proof", sleep=lambda _n: None
        )
        self._bind_session(session, client)
        client.release_service = lambda _ticket_id: {}

        with self.assertRaisesRegex(
            FleetBackendError, "did not prove exact ticket release"
        ):
            session.close()

        self.assertEqual(
            session.ticket_id, "fd-0123456789abcdef0123456789abcdef"
        )

    def test_release_identity_drift_retains_the_exact_ticket(self):
        for field, replacement in (
            ("ticket_id", "fd-ffffffffffffffffffffffffffffffff"),
            ("profile_id", "aeon-qwen38-other"),
            ("service_id", "aeon-qwen38-other"),
            ("consumer", "aeon/other"),
            ("state", "active"),
            ("compute_state", "ready"),
            ("endpoint", "http://127.0.0.1:8033/v1"),
        ):
            with self.subTest(field=field):
                client = _FakeClient()
                session = BrokerServiceSession(
                    client=client, consumer="aeon/release-drift"
                )
                self._bind_session(session, client)

                def release_service(
                    ticket_id, _field=field, _replacement=replacement
                ):
                    client.released.append(ticket_id)
                    result = client._snapshot(
                        state="released", compute_state="inactive", endpoint=None
                    )
                    result[_field] = _replacement
                    return result

                client.release_service = release_service

                with self.assertRaisesRegex(
                    FleetBackendError, "did not prove exact ticket release"
                ):
                    session.close()

                self.assertEqual(client.released, [client.ticket_id])
                self.assertEqual(session.ticket_id, client.ticket_id)

    def test_renew_identity_drift_retains_ticket_for_close(self):
        client = _FakeClient()
        session = BrokerServiceSession(client=client, consumer="aeon/renew-drift")
        self._bind_session(session, client)
        session.endpoint = client.endpoint

        def renew_service(ticket_id, *, ttl_seconds):
            client.renewed.append((ticket_id, ttl_seconds))
            result = client._snapshot(
                state="active", compute_state="ready", endpoint=client.endpoint
            )
            result["service_id"] = "aeon-qwen38-other"
            return result

        client.renew_service = renew_service

        with self.assertRaisesRegex(FleetBackendError, "logical service"):
            session._renew_once()

        self.assertEqual(session.ticket_id, client.ticket_id)
        session.close()

    def test_failed_start_keeps_ticket_when_release_is_unproven(self):
        client = _FakeClient()
        original_acquire = client.acquire_service

        def acquire_service(**kwargs):
            result = original_acquire(**kwargs)
            result["state"] = "released"
            return result

        client.acquire_service = acquire_service
        client.release_service = lambda _ticket_id: {}
        session = BrokerServiceSession(
            client=client,
            consumer="aeon/unproven-start-release",
            sleep=lambda _n: None,
        )

        with self.assertRaises(FleetBackendError) as raised:
            session.start()

        self.assertEqual(session.ticket_id, client.ticket_id)
        self.assertTrue(
            any("ticket was retained" in note for note in raised.exception.__notes__)
        )

    def test_endpoint_validation_rejects_non_loopback_and_credentials(self):
        for endpoint in (
            "https://127.0.0.1:8033/v1",
            "http://192.168.0.177:8033/v1",
            "http://user@127.0.0.1:8033/v1",
            "http://127.0.0.1:8033/admin",
            " http://127.0.0.1:8033/v1",
            "http://127.0.0.1:8033/v\n1",
            "http://127.0.0.1:8033/v1\r",
            "http://127.0.0.1:8033/v1\t",
            "http://127.0.0.1:8033/v1;admin",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(FleetBackendError):
                validate_loopback_endpoint(endpoint)

    def test_endpoint_validation_returns_one_canonical_loopback_url(self):
        self.assertEqual(
            validate_loopback_endpoint("http://127.0.0.1:8033/"),
            "http://127.0.0.1:8033/v1",
        )
        self.assertEqual(
            validate_loopback_endpoint("http://[::1]:8033/v1/"),
            "http://[::1]:8033/v1",
        )

    def test_new_session_for_same_terminal_gets_a_new_acquisition_key(self):
        client = _FakeClient()
        first = BrokerServiceSession(client=client, consumer="aeon/stable", sleep=lambda _n: None)
        second = BrokerServiceSession(client=client, consumer="aeon/stable", sleep=lambda _n: None)
        first.start()
        first.close()
        second.start()
        second.close()
        keys = [call["idempotency_key"] for call in client.acquired]
        self.assertEqual(len(keys), 2)
        self.assertNotEqual(keys[0], keys[1])

    def test_session_object_cannot_be_started_twice(self):
        client = _FakeClient()
        session = BrokerServiceSession(client=client, consumer="aeon/test", sleep=lambda _n: None)
        session.start()
        with self.assertRaises(FleetBackendError):
            session.start()
        session.close()

    def test_failed_readiness_releases_the_acquired_ticket(self):
        client = _FakeClient()
        session = BrokerServiceSession(
            client=client, consumer="aeon/readiness-failure", sleep=lambda _n: None
        )
        client.consumer = session.consumer
        client.service_status = lambda _ticket_id: client._snapshot(
            state="released", compute_state="inactive", endpoint=None
        )

        with self.assertRaises(FleetBackendError):
            session.start()

        self.assertEqual(
            client.released, ["fd-0123456789abcdef0123456789abcdef"]
        )
        self.assertIsNone(session.ticket_id)

    def test_session_manager_retains_failed_release_for_atexit_retry(self):
        from aeon import main

        manager = main.SessionManager()
        broker_session = Mock()
        broker_session.close.side_effect = [
            FleetBackendError("malformed release proof"),
            {"state": "released", "compute_state": "inactive"},
        ]
        manager._broker_service = broker_session

        with patch.object(main, "terminate_all_sub_agents"), patch.object(
            main, "cleanup_transient_tools"
        ):
            manager.exit()
            self.assertIs(manager._broker_service, broker_session)
            self.assertFalse(manager._cleanup_done)

            manager.exit()

        self.assertIsNone(manager._broker_service)
        self.assertTrue(manager._cleanup_done)
        self.assertEqual(broker_session.close.call_count, 2)


class NexusInteractiveFleetClientTests(unittest.TestCase):
    def test_capability_stays_in_owner_only_header_not_request_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            capability = Path(directory) / "nexus.capability"
            capability.write_text(
                "nexus-capability-" + "a" * 40 + "\n", encoding="ascii"
            )
            capability.chmod(0o600)
            client = FleetBrokerClient(nexus_capability_path=capability)
            client._request = Mock(return_value={})

            client.acquire_nexus_interactive_service(
                profile="aeon-qwen38-standard",
                consumer="nexus/direct-main-orchestrator",
                idempotency_key="nexus-direct/" + "b" * 32,
                ttl_seconds=120,
            )

            _method, path, payload = client._request.call_args.args
            headers = client._request.call_args.kwargs["headers"]
            self.assertEqual(path, "/v1/nexus/interactive/services")
            self.assertEqual(
                headers["Authorization"], "Bearer nexus-capability-" + "a" * 40
            )
            self.assertNotIn("capability", repr(payload).lower())
            self.assertNotIn("demand_class", payload)

            client.renew_service("fd-" + "c" * 32, ttl_seconds=120)
            self.assertEqual(
                client._request.call_args.kwargs["headers"], headers
            )
            client.release_service("fd-" + "c" * 32)
            self.assertEqual(
                client._request.call_args.kwargs["headers"], headers
            )
            client.set_service_policy(
                "aeon-qwen38-standard", requested_replicas=1, actor="nexus"
            )
            self.assertEqual(
                client._request.call_args.kwargs["headers"], headers
            )

    def test_unsafe_capability_file_fails_before_broker_request(self):
        with tempfile.TemporaryDirectory() as directory:
            capability = Path(directory) / "nexus.capability"
            capability.write_text("nexus-capability-" + "a" * 40, encoding="ascii")
            capability.chmod(0o644)
            client = FleetBrokerClient(nexus_capability_path=capability)
            client._request = Mock(return_value={})

            with self.assertRaises(FleetBackendError):
                client.acquire_nexus_interactive_service(
                    profile="aeon-qwen38-standard",
                    consumer="nexus/direct-main-orchestrator",
                    idempotency_key="nexus-direct/" + "b" * 32,
                    ttl_seconds=120,
                )
            client._request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
