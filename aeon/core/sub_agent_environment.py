"""Environment boundary for bounded Aeon sub-agents.

Bounded workers inherit ordinary runtime configuration from their principal, but
must never inherit a capability or identity that authorizes mutations as that
principal.  Keep the allow-deny decision in one small module so both the spawn
boundary and the child wrapper apply the same defense.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, Callable

from aeon.core.fleet_backend import (
    DEFAULT_BROKER_SOCKET,
    DEFAULT_QWEN_PROFILE,
    BrokerServiceSession,
    FleetBackendError,
    FleetBrokerClient,
    select_compute_backend,
    validate_loopback_endpoint,
)
from aeon.core.skills.manager import INSTANCE_SKILLS_DIR_ENV


PRINCIPAL_ONLY_ENV_KEYS = frozenset(
    {
        "AEON_MAIN_ORCHESTRATOR",
        "AEON_REMOTE_INSTANCE_ID",
        INSTANCE_SKILLS_DIR_ENV,
        "AEON_COLLABORATOR_MODE_PATH",
        "NEXUS_INTERNAL_ORCHESTRATOR_URL",
        "NEXUS_ORCHESTRATOR_TOKEN_FILE",
        "NEXUS_INTERNAL_SELF_SETTINGS_URL",
        "NEXUS_SELF_SETTINGS_TOKEN_FILE",
        "NEXUS_INTERNAL_MCP_URL",
        "NEXUS_MCP_DELEGATION_ID",
        "NEXUS_MCP_DELEGATION_TOKEN_FILE",
    }
)

VERIFICATION_PREBOUND_NONCE_ENV = "AEON_VERIFICATION_PREBOUND_NONCE"
VERIFICATION_PREBOUND_RECEIPT = ".verification-prebound-fleet.json"
LAUNCHER_ONLY_ENV_KEYS = frozenset(
    {"AEON_CPU_SANDBOX_SLICE", VERIFICATION_PREBOUND_NONCE_ENV}
)
CHILD_FLEET_CONFIGURATION_KEYS = frozenset(
    {"AEON_COMPUTE_BACKEND", "AEON_FLEET_SOCKET", "AEON_FLEET_PROFILE"}
)
NO_ACCELERATOR_ENV = {
    "CUDA_VISIBLE_DEVICES": "void",
    "GPU_DEVICE_ORDINAL": "-1",
    "HIP_VISIBLE_DEVICES": "-1",
    "NVIDIA_VISIBLE_DEVICES": "void",
    "ROCR_VISIBLE_DEVICES": "-1",
}
_EXACT_RESOURCE_AUTHORITY_KEYS = frozenset(
    {
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        "CUDA_MPS_LOG_DIRECTORY",
        "CUDA_MPS_PIPE_DIRECTORY",
        "GPU_AGENT_CLAIM_ID",
        "GPU_LEASE_EXCLUSIVE",
        "GPU_LEASE_ID",
        "GPU_LEASE_OWNER",
        "GPU_LEASE_RUN_DIR",
        "GPU_MEM_LIMIT_GB",
        "GPU_MEM_UTIL",
        "GPU_PLANNED_VRAM_GB",
        "GPU_RESERVE_GB",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_REQUIRE_CUDA",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
    }
)

# These are the local container-served providers recognized by Aeon's normal
# SessionManager. Subscription/API providers do not consume owner GPU compute and
# must not create Fleet demand merely because they happen to be used by a bounded
# child in an older or downstream configuration.
LOCAL_QWEN_PROVIDERS = frozenset({"llamacpp", "vllm"})

# Keep the child cleanup budget explicit and aligned with the principal's
# termination grace in sub_agent_state.py. BrokerServiceSession.close() can also
# spend up to two seconds joining its renewal thread before the release request.
SUB_AGENT_BROKER_TIMEOUT_SECONDS = 10.0
SUB_AGENT_START_CANCEL_GRACE_SECONDS = 12.0
SUB_AGENT_FLEET_CLOSE_WORST_CASE_SECONDS = (
    SUB_AGENT_START_CANCEL_GRACE_SECONDS
    + 2.0
    + SUB_AGENT_BROKER_TIMEOUT_SECONDS
)


def bounded_sub_agent_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a child environment without principal-only identity/capabilities."""

    environment = dict(os.environ if source is None else source)
    scrub_principal_capabilities(environment)
    return environment


def scrub_principal_capabilities(environment: MutableMapping[str, str]) -> None:
    """Remove inherited identity, endpoint, ticket, lease, and GPU authority.

    The bounded child may keep only enough Fleet configuration to request its
    own broker ticket. Accelerator selectors are then replaced by explicit
    no-device values; an inherited selector is never forwarded verbatim.
    ``AEON_CPU_SANDBOX_SLICE`` is launcher-only and is therefore removed here
    too. The spawn path adds its freshly UUID-derived value after this scrub.
    """

    for key in tuple(environment):
        upper = str(key).upper()
        remove = (
            upper in PRINCIPAL_ONLY_ENV_KEYS
            or upper in LAUNCHER_ONLY_ENV_KEYS
            or upper in _EXACT_RESOURCE_AUTHORITY_KEYS
            or upper in NO_ACCELERATOR_ENV
            or (
                upper.startswith("AEON_FLEET_")
                and key not in CHILD_FLEET_CONFIGURATION_KEYS
            )
            or (
                upper == "AEON_COMPUTE_BACKEND"
                and key != "AEON_COMPUTE_BACKEND"
            )
            or upper.startswith("FLEET_")
            or "TICKET" in upper
            or "LEASE" in upper
            or "CLAIM" in upper
            or "BASE_URL" in upper
            or ("VISION" in upper and ("URL" in upper or "ENDPOINT" in upper))
            or upper.endswith("_ENDPOINT")
        )
        if remove:
            environment.pop(key, None)
    # Never inherit a coordinator/direct/auto choice. ``auto`` is broker-only in
    # current Aeon, but an explicit value makes the child contract unambiguous.
    environment["AEON_COMPUTE_BACKEND"] = "broker"
    environment.update(NO_ACCELERATOR_ENV)


def model_requires_fleet_compute(model_config: Mapping[str, Any]) -> bool:
    """Return whether this child model is one of Aeon's local Qwen runtimes."""

    provider = str(model_config.get("provider") or "").strip().lower()
    return provider in LOCAL_QWEN_PROVIDERS


class SubAgentFleetCompute:
    """Own one child-specific Fleet service ticket for a bounded sub-agent.

    The principal's ``base_url`` is never accepted as compute authority. For a
    local Qwen config, ``start`` first validates the broker-only backend, creates
    a distinct durable demand ticket, and replaces the inherited URL with the
    broker-returned loopback endpoint. Non-local provider configs remain intact.

    ``close`` is retryable and serialized so normal completion, exception cleanup,
    a signal handler, and the watchdog can all converge on releasing the exact
    same ticket without racing each other.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        model_config: MutableMapping[str, Any],
        environ: Mapping[str, str] | None = None,
        broker_client: FleetBrokerClient | None = None,
        session_factory: Callable[..., BrokerServiceSession] = BrokerServiceSession,
        wait_callback: Callable[[], None] | None = None,
    ) -> None:
        self.agent_id = str(agent_id)
        self.model_config = model_config
        self.environ = dict(os.environ if environ is None else environ)
        self.broker_client = broker_client
        self.session_factory = session_factory
        self.wait_callback = wait_callback
        self.required = model_requires_fleet_compute(model_config)
        self.session: BrokerServiceSession | None = None
        self._abort_wait = threading.Event()
        self._start_done = threading.Event()
        self._close_lock = threading.Lock()
        self._start_called = False
        if not self.required:
            self._start_done.set()

    @property
    def consumer(self) -> str:
        """Stable, unambiguous identity for this one bounded child process."""

        return f"aeon/sub-agent/{self.agent_id}"

    def _wait_for_compute(self, seconds: float) -> None:
        if self.wait_callback is not None:
            self.wait_callback()
        if self._abort_wait.wait(float(seconds)):
            raise FleetBackendError("sub-agent Fleet compute wait was cancelled")

    def start(self) -> str | None:
        """Acquire child demand and replace an inherited local endpoint."""

        if not self.required:
            return None
        if self._start_called:
            raise FleetBackendError("sub-agent Fleet compute cannot be started twice")
        self._start_called = True
        try:
            socket_path = Path(
                self.environ.get("AEON_FLEET_SOCKET", str(DEFAULT_BROKER_SOCKET))
            ).expanduser()
            client = self.broker_client or FleetBrokerClient(
                socket_path, timeout=SUB_AGENT_BROKER_TIMEOUT_SECONDS
            )
            backend, _reason = select_compute_backend(
                environ=self.environ,
                client=client,
            )
            if backend != "broker":
                raise FleetBackendError("sub-agent local compute must use the Fleet broker")
            profile = self.environ.get("AEON_FLEET_PROFILE", DEFAULT_QWEN_PROFILE)
            self.session = self.session_factory(
                client=client,
                profile=profile,
                consumer=self.consumer,
                sleep=self._wait_for_compute,
            )
            endpoint = self.session.start()
            # BrokerServiceSession validates this as an approved loopback API.
            # Do not retain or fall back to the principal's inherited endpoint.
            self.model_config["base_url"] = endpoint
            return endpoint
        finally:
            self._start_done.set()

    def bind(self, *, llm_client: Any, worker: Any) -> None:
        """Attach endpoint promotion and per-turn ticket validation."""

        session = self.session
        if session is None:
            return

        def rebind(endpoint: str, runtime_profiles: tuple[str, ...]) -> None:
            from aeon.core.model_identity import wire_model_for_runtime_profiles

            wire_model = wire_model_for_runtime_profiles(runtime_profiles)
            llm_client.rebind_base_url(endpoint, api_model=wire_model)
            self.model_config["base_url"] = endpoint
            self.model_config["api_model"] = wire_model
            if self.model_config.get("multimodal"):
                os.environ["AEON_VISION_BASE_URL"] = endpoint
                os.environ["AEON_VISION_MODEL"] = wire_model

        if self.model_config.get("multimodal"):
            # Vision tools must follow this child's ticket-affine router too;
            # retaining the principal's inherited vision URL would bypass the
            # independent demand even though the control-model client is safe.
            os.environ["AEON_VISION_BASE_URL"] = str(self.model_config["base_url"])
        session.set_endpoint_change_handler(rebind)
        worker.compute_guard = session.ensure_ready

    def assert_prebound_endpoint_healthy(self, expected_endpoint: str) -> None:
        """Fail if a parent-owned verification endpoint can no longer be used.

        The parent keeps the broker session and renewal thread.  A verification
        child cannot safely rebind itself because it deliberately receives no
        broker capability, so any renewal failure or endpoint promotion stops
        that child and forces a fresh verification attempt.
        """

        if not self.required:
            raise FleetBackendError("prebound verification requires local Fleet compute")
        expected = validate_loopback_endpoint(expected_endpoint)
        session = self.session
        if session is None:
            raise FleetBackendError("prebound verification Fleet session is absent")
        renewal_error = getattr(session, "_renew_error", None)
        if renewal_error is not None:
            raise FleetBackendError(
                "prebound verification Fleet ticket renewal failed"
            ) from renewal_error
        endpoint_lock = getattr(session, "_endpoint_lock", None)
        if endpoint_lock is None:
            raise FleetBackendError("prebound verification endpoint lock is absent")
        with endpoint_lock:
            current = getattr(session, "endpoint", None)
            pending = getattr(session, "_pending_endpoint", None)
        if current != expected:
            raise FleetBackendError("prebound verification endpoint identity changed")
        if pending is not None:
            raise FleetBackendError(
                "prebound verification runtime binding was promoted; restart verification"
            )

    def request_stop(self) -> None:
        """Wake a broker admission wait without doing I/O in a signal handler."""

        self._abort_wait.set()

    def close(
        self,
        *,
        wait_for_start_seconds: float = SUB_AGENT_START_CANCEL_GRACE_SECONDS,
    ) -> dict[str, str] | None:
        """Cancel admission if needed, then release the exact child ticket.

        Broker calls have bounded ten-second I/O timeouts. The slightly larger
        startup grace lets a watchdog/signal cancel a child waiting for compute
        and allows ``BrokerServiceSession.start`` to perform its own exact release
        before this method retries closure. If that cannot finish, the demand's
        broker TTL remains the final crash-safety boundary.
        """

        self.request_stop()
        if not self._start_done.wait(timeout=max(0.0, float(wait_for_start_seconds))):
            raise FleetBackendError(
                "sub-agent Fleet startup did not stop within the cleanup grace"
            )
        with self._close_lock:
            session = self.session
            if session is None:
                return None
            proof = session.close()
            self.session = None
            return proof
