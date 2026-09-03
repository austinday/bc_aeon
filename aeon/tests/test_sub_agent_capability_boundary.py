"""Hermetic checks for the principal-to-bounded-agent capability boundary."""

from __future__ import annotations

import os
import json
import stat
import tempfile
import threading
import types
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from aeon.core.fleet_backend import FleetBackendError
from aeon.core.skills.manager import INSTANCE_SKILLS_DIR_ENV, SkillsManager
from aeon.core.sub_agent_environment import (
    CHILD_FLEET_CONFIGURATION_KEYS,
    NO_ACCELERATOR_ENV,
    PRINCIPAL_ONLY_ENV_KEYS,
    SubAgentFleetCompute,
    VERIFICATION_PREBOUND_NONCE_ENV,
    VERIFICATION_PREBOUND_RECEIPT,
    bounded_sub_agent_environment,
    model_requires_fleet_compute,
)
from aeon.core.sub_agent_state import CPU_SANDBOX_SLICE_ENV, sub_agent_systemd_units
from aeon.core.orchestrator_instructions import main_orchestrator_instruction_section
from aeon.scripts.sub_agent_wrapper import (
    _bind_private_skill_overlay,
    _consume_prebound_verification_capability,
    _scrub_inherited_principal_environment,
)
from aeon.tools.start_agent_instance import StartAgentInstanceTool
from aeon.tools.sub_agent import SpawnSubAgent


class _Process:
    pid = 4242


class _FleetSession:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.endpoint_handler = None
        self.ensure_calls = 0
        self.close_calls = 0
        self.endpoint = None
        self._endpoint_lock = threading.RLock()
        self._renew_error = None
        self._pending_endpoint = None

    def start(self):
        self.endpoint = "http://127.0.0.1:8443/v1"
        return self.endpoint

    def set_endpoint_change_handler(self, handler):
        self.endpoint_handler = handler

    def ensure_ready(self):
        self.ensure_calls += 1

    def close(self):
        self.close_calls += 1
        return {"state": "released", "compute_state": "inactive"}


class SubAgentCapabilityBoundaryTests(unittest.TestCase):
    def test_sub_agent_browser_cleanup_disables_proxies_and_redirects(self):
        source = Path(__import__(
            "aeon.scripts.sub_agent_wrapper", fromlist=["__file__"]
        ).__file__).read_text(encoding="utf-8")
        body = source.split("def _release_browser_profile", 1)[1].split(
            "\n\ndef main", 1
        )[0]
        self.assertIn("BROWSER_API_URL", body)
        self.assertIn("trust_env = False", body)
        self.assertIn("allow_redirects=False", body)

    def test_environment_copy_removes_every_principal_capability(self):
        source = {
            "PATH": "/safe/bin",
            **{key: f"secret-{index}" for index, key in enumerate(PRINCIPAL_ONLY_ENV_KEYS)},
            "AEON_BENCHMARK_GPU_CAPABILITY_RECEIPT_KEY": "benchmark-secret",
            "AEON_BENCHMARK_GPU_CAPABILITY_RECEIPT_PATH": "/private/receipt.json",
            "AEON_OPENCODE_COMPLETION_KEY_FILE": "/private/completion-key.bin",
            "AEON_OPENCODE_COMPLETION_STATE": "/private/completion.json",
            "AEON_OPENCODE_PROXY_TOKEN": "proxy-secret",
            "aeon_opencode_authority_file": "/private/authority.json",
        }

        child = bounded_sub_agent_environment(source)

        self.assertEqual(
            child,
            {
                "PATH": "/safe/bin",
                "AEON_COMPUTE_BACKEND": "broker",
                **NO_ACCELERATOR_ENV,
            },
        )
        self.assertTrue(PRINCIPAL_ONLY_ENV_KEYS.issubset(source))

    def test_wrapper_scrub_is_idempotent_defense_in_depth(self):
        environment = {
            "PATH": "/safe/bin",
            **{key: "must-not-survive" for key in PRINCIPAL_ONLY_ENV_KEYS},
        }

        with patch.dict(os.environ, environment, clear=True):
            _scrub_inherited_principal_environment()
            _scrub_inherited_principal_environment()

            self.assertEqual(
                dict(os.environ),
                {
                    "PATH": "/safe/bin",
                    "AEON_COMPUTE_BACKEND": "broker",
                    **NO_ACCELERATOR_ENV,
                },
            )
            self.assertTrue(StartAgentInstanceTool().is_internal)
            self.assertEqual(main_orchestrator_instruction_section(), "")

    def test_environment_removes_inherited_compute_authority_and_sets_no_devices(self):
        source = {
            "PATH": "/safe/bin",
            "AEON_COMPUTE_BACKEND": "coordinator",
            "AEON_FLEET_SOCKET": "/run/user/1000/fleet.sock",
            "AEON_FLEET_PROFILE": "aeon-qwen38-standard",
            "AEON_FLEET_TICKET": "fd-secret",
            "FLEET_LEASE_ID": "lease-secret",
            "GPU_AGENT_CLAIM_ID": "claim-secret",
            "GPU_MEM_LIMIT_GB": "48",
            "CUDA_VISIBLE_DEVICES": "GPU-secret",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "HIP_VISIBLE_DEVICES": "0",
            "ROCR_VISIBLE_DEVICES": "1",
            "AEON_VISION_BASE_URL": "http://127.0.0.1:9999/v1",
            "OPENAI_BASE_URL": "https://endpoint.invalid/v1",
            CPU_SANDBOX_SLICE_ENV: "attacker.slice",
        }

        child = bounded_sub_agent_environment(source)

        self.assertEqual(child["AEON_COMPUTE_BACKEND"], "broker")
        self.assertEqual(child["AEON_FLEET_SOCKET"], source["AEON_FLEET_SOCKET"])
        self.assertEqual(child["AEON_FLEET_PROFILE"], source["AEON_FLEET_PROFILE"])
        for key, value in NO_ACCELERATOR_ENV.items():
            self.assertEqual(child[key], value)
        for key in (
            "AEON_FLEET_TICKET",
            "FLEET_LEASE_ID",
            "GPU_AGENT_CLAIM_ID",
            "GPU_MEM_LIMIT_GB",
            "AEON_VISION_BASE_URL",
            "OPENAI_BASE_URL",
            CPU_SANDBOX_SLICE_ENV,
        ):
            self.assertNotIn(key, child)

    def test_child_replaces_principal_skill_overlay_with_exact_private_state(self):
        agent_id = "12345678-1234-1234-1234-123456789abc"
        inherited = "/private/principal/skills"
        child = bounded_sub_agent_environment(
            {"PATH": "/safe/bin", INSTANCE_SKILLS_DIR_ENV: inherited}
        )
        self.assertNotIn(INSTANCE_SKILLS_DIR_ENV, child)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / agent_id
            output_dir.mkdir(mode=0o700)
            with patch.dict(
                os.environ, {INSTANCE_SKILLS_DIR_ENV: inherited}, clear=True
            ):
                _scrub_inherited_principal_environment(agent_id, str(output_dir))
                self.assertNotIn(INSTANCE_SKILLS_DIR_ENV, os.environ)
                overlay = _bind_private_skill_overlay(agent_id, output_dir)
                self.assertEqual(overlay, output_dir / "skills")
                self.assertEqual(os.environ[INSTANCE_SKILLS_DIR_ENV], str(overlay))
                self.assertEqual(SkillsManager().instance_dir, overlay)
                self.assertEqual(stat.S_IMODE(overlay.stat().st_mode), 0o700)

    def test_wrapper_restores_only_its_exact_generated_slice(self):
        agent_id = "12345678-1234-1234-1234-123456789abc"
        _scope, exact_slice = sub_agent_systemd_units(agent_id)
        environment = {
            CPU_SANDBOX_SLICE_ENV: exact_slice,
            "CUDA_VISIBLE_DEVICES": "GPU-secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            _scrub_inherited_principal_environment(agent_id)
            self.assertEqual(os.environ[CPU_SANDBOX_SLICE_ENV], exact_slice)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "void")

        with patch.dict(
            os.environ, {CPU_SANDBOX_SLICE_ENV: "attacker.slice"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                _scrub_inherited_principal_environment(agent_id)
            self.assertNotIn(CPU_SANDBOX_SLICE_ENV, os.environ)

    def test_prebound_verification_consumes_exact_receipt_without_broker_authority(self):
        agent_id = "12345678-1234-1234-1234-123456789abc"
        scope_unit, slice_unit = sub_agent_systemd_units(agent_id)
        nonce = "a" * 64
        endpoint = "http://127.0.0.1:8443/v1"
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / agent_id
            output_dir.mkdir(mode=0o700)
            receipt_path = output_dir / VERIFICATION_PREBOUND_RECEIPT
            receipt_path.write_text(json.dumps({
                "schema": 1,
                "kind": "aeon-verification-prebound-fleet",
                "agent_id": agent_id,
                "scope_unit": scope_unit,
                "slice_unit": slice_unit,
                "endpoint": endpoint,
                "nonce": nonce,
            }), encoding="utf-8")
            receipt_path.chmod(0o600)
            args = types.SimpleNamespace(agent_id=agent_id, output_dir=str(output_dir))
            config = {"provider": "vllm", "base_url": endpoint}
            environment = {
                CPU_SANDBOX_SLICE_ENV: slice_unit,
                VERIFICATION_PREBOUND_NONCE_ENV: nonce,
                "AEON_COMPUTE_BACKEND": "broker",
                "AEON_FLEET_SOCKET": "/must/not/reach/candidate.sock",
                "AEON_FLEET_PROFILE": "must-not-reach-candidate",
            }
            membership = PurePosixPath(f"/fixture/{slice_unit}/{scope_unit}")
            with patch.dict(os.environ, environment, clear=True), patch(
                "aeon.scripts.sub_agent_wrapper._current_unified_cgroup",
                return_value=membership,
            ):
                self.assertEqual(
                    _consume_prebound_verification_capability(args, config),
                    endpoint,
                )
                self.assertNotIn(VERIFICATION_PREBOUND_NONCE_ENV, os.environ)
                for key in CHILD_FLEET_CONFIGURATION_KEYS:
                    self.assertNotIn(key, os.environ)
            self.assertFalse(receipt_path.exists())

    def test_prebound_verification_nonce_cannot_bless_an_inherited_url(self):
        agent_id = "12345678-1234-1234-1234-123456789abc"
        scope_unit, slice_unit = sub_agent_systemd_units(agent_id)
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(agent_id=agent_id, output_dir=temporary)
            with patch.dict(os.environ, {
                CPU_SANDBOX_SLICE_ENV: slice_unit,
                VERIFICATION_PREBOUND_NONCE_ENV: "b" * 64,
            }, clear=True), patch(
                "aeon.scripts.sub_agent_wrapper._current_unified_cgroup",
                return_value=PurePosixPath(f"/fixture/{slice_unit}/{scope_unit}"),
            ), self.assertRaisesRegex(RuntimeError, "unavailable"):
                _consume_prebound_verification_capability(
                    args,
                    {"provider": "vllm", "base_url": "http://127.0.0.1:9999/v1"},
                )

    def test_spawn_passes_only_sanitized_environment_to_popen(self):
        unsafe = {
            "AEON_MAIN_ORCHESTRATOR": "1",
            "AEON_REMOTE_INSTANCE_ID": "8dac3bf6190c53eaa221fbd0a566cc0b",
            "NEXUS_INTERNAL_ORCHESTRATOR_URL": (
                "http://127.0.0.1:8765/internal/orchestrator/agents"
            ),
            "NEXUS_ORCHESTRATOR_TOKEN_FILE": "/private/orchestrator.token",
            "NEXUS_INTERNAL_SELF_SETTINGS_URL": (
                "http://127.0.0.1:8765/internal/agent/job-role"
            ),
            "NEXUS_SELF_SETTINGS_TOKEN_FILE": "/private/self-settings.token",
            CPU_SANDBOX_SLICE_ENV: "inherited-attacker.slice",
            "CUDA_VISIBLE_DEVICES": "GPU-principal",
            "GPU_AGENT_CLAIM_ID": "claim-principal",
            "SUB_AGENT_SAFE_MARKER": "preserved",
        }
        worker = types.SimpleNamespace(
            instance_id="principal-instance",
            model_config={"model": "fixture"},
            debug_mode=False,
        )

        with tempfile.TemporaryDirectory() as temporary:
            previous = os.getcwd()
            os.chdir(temporary)
            try:
                with (
                    patch.dict(os.environ, unsafe, clear=False),
                    patch(
                        "aeon.tools.sub_agent.assert_sub_agent_systemd_units_available"
                    ) as available,
                    patch("aeon.tools.sub_agent.subprocess.Popen", return_value=_Process()) as popen,
                    patch(
                        "aeon.tools.sub_agent.capture_sub_agent_process",
                        return_value={
                            "schema": 2,
                            "agent_id": "fixture",
                            "pid": _Process.pid,
                        },
                    ) as capture,
                ):
                    result = SpawnSubAgent(worker=worker).execute(
                        "Perform a bounded, hermetic review.",
                        time_budget_minutes=1,
                        max_iterations=1,
                        stall_timeout_seconds=60,
                    )
            finally:
                os.chdir(previous)

        self.assertIn("Sub-agent spawned", result)
        available.assert_called_once()
        child = popen.call_args.kwargs["env"]
        self.assertEqual(child["SUB_AGENT_SAFE_MARKER"], "preserved")
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "void")
        self.assertNotIn("GPU_AGENT_CLAIM_ID", child)
        for key in PRINCIPAL_ONLY_ENV_KEYS:
            self.assertNotIn(key, child)
        launch_argv = popen.call_args.args[0]
        self.assertEqual(launch_argv[0], "/usr/bin/systemd-run")
        self.assertIn(f"--slice={child[CPU_SANDBOX_SLICE_ENV]}", launch_argv)
        self.assertRegex(
            child[CPU_SANDBOX_SLICE_ENV],
            r"^aeon_subagent_[0-9a-f]{32}\.slice$",
        )
        self.assertEqual(
            capture.call_args.kwargs["slice_unit"], child[CPU_SANDBOX_SLICE_ENV]
        )


class SubAgentFleetComputeTests(unittest.TestCase):
    def test_only_local_container_served_models_require_fleet(self):
        self.assertTrue(model_requires_fleet_compute({"provider": "vllm"}))
        self.assertTrue(model_requires_fleet_compute({"provider": "llamacpp"}))
        self.assertFalse(model_requires_fleet_compute({"provider": "openai"}))
        self.assertFalse(model_requires_fleet_compute({"provider": "anthropic"}))

    def test_local_child_owns_ticket_endpoint_promotion_guard_and_exact_release(self):
        config = {
            "provider": "vllm",
            "model": "fixture",
            "base_url": "http://127.0.0.1:9999/v1",
            "multimodal": True,
        }
        client = object()
        created = []

        def session_factory(**kwargs):
            session = _FleetSession(**kwargs)
            created.append(session)
            return session

        with patch(
            "aeon.core.sub_agent_environment.select_compute_backend",
            return_value=("broker", "fixture"),
        ) as select:
            compute = SubAgentFleetCompute(
                agent_id="12345678-1234-1234-1234-123456789abc",
                model_config=config,
                environ={
                    "AEON_FLEET_SOCKET": "/fixture/broker.sock",
                    "AEON_FLEET_PROFILE": "aeon-qwen38-standard",
                },
                broker_client=client,
                session_factory=session_factory,
            )
            endpoint = compute.start()

        self.assertEqual(endpoint, "http://127.0.0.1:8443/v1")
        self.assertEqual(config["base_url"], endpoint)
        select.assert_called_once()
        self.assertIs(select.call_args.kwargs["client"], client)
        self.assertEqual(len(created), 1)
        session = created[0]
        self.assertIs(session.kwargs["client"], client)
        self.assertEqual(session.kwargs["profile"], "aeon-qwen38-standard")
        self.assertEqual(
            session.kwargs["consumer"],
            "aeon/sub-agent/12345678-1234-1234-1234-123456789abc",
        )

        llm_client = types.SimpleNamespace(rebound=[])
        llm_client.rebind_base_url = lambda endpoint, *, api_model=None: (
            llm_client.rebound.append((endpoint, api_model))
        )
        worker = types.SimpleNamespace(compute_guard=None)
        with patch.dict(os.environ, {}, clear=True):
            compute.bind(llm_client=llm_client, worker=worker)
            worker.compute_guard()
            session.endpoint_handler(
                "http://127.0.0.1:8555/v1",
                ("aeon-qwen38-flash-next-vllm-177",),
            )
            self.assertEqual(
                os.environ["AEON_VISION_BASE_URL"],
                "http://127.0.0.1:8555/v1",
            )

        self.assertEqual(session.ensure_calls, 1)
        self.assertEqual(
            llm_client.rebound,
            [
                (
                    "http://127.0.0.1:8555/v1",
                    "Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP",
                )
            ],
        )
        self.assertEqual(config["base_url"], "http://127.0.0.1:8555/v1")
        self.assertEqual(
            config["api_model"],
            "Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP",
        )
        self.assertEqual(
            compute.close(), {"state": "released", "compute_state": "inactive"}
        )
        self.assertIsNone(compute.close())
        self.assertEqual(session.close_calls, 1)

    def test_parent_owned_prebound_health_refuses_endpoint_promotion(self):
        config = {"provider": "vllm", "model": "fixture", "base_url": "old"}
        sessions = []

        def session_factory(**kwargs):
            session = _FleetSession(**kwargs)
            sessions.append(session)
            return session

        with patch(
            "aeon.core.sub_agent_environment.select_compute_backend",
            return_value=("broker", "fixture"),
        ):
            compute = SubAgentFleetCompute(
                agent_id="12345678-1234-1234-1234-123456789abc",
                model_config=config,
                broker_client=object(),
                session_factory=session_factory,
            )
            endpoint = compute.start()

        compute.assert_prebound_endpoint_healthy(endpoint)
        sessions[0]._endpoint_lock = threading.RLock()
        sessions[0]._renew_error = None
        sessions[0]._pending_endpoint = "http://127.0.0.1:8555/v1"
        sessions[0].endpoint = endpoint
        with self.assertRaisesRegex(FleetBackendError, "promoted"):
            compute.assert_prebound_endpoint_healthy(endpoint)
        compute.close()

    def test_external_provider_does_not_create_fleet_demand_or_rewrite_endpoint(self):
        config = {
            "provider": "openai",
            "model": "cloud-fixture",
            "base_url": "https://provider.example/v1",
        }

        def forbidden_factory(**_kwargs):
            self.fail("external provider must not create a Fleet session")

        with patch(
            "aeon.core.sub_agent_environment.select_compute_backend"
        ) as select:
            compute = SubAgentFleetCompute(
                agent_id="cloud-child",
                model_config=config,
                session_factory=forbidden_factory,
            )
            self.assertIsNone(compute.start())
            self.assertIsNone(compute.close())

        select.assert_not_called()
        self.assertEqual(config["base_url"], "https://provider.example/v1")

    def test_local_child_refuses_any_nonbroker_selection(self):
        config = {"provider": "vllm", "model": "fixture", "base_url": "inherited"}
        with patch(
            "aeon.core.sub_agent_environment.select_compute_backend",
            return_value=("coordinator", "unsafe fixture"),
        ):
            compute = SubAgentFleetCompute(
                agent_id="local-child",
                model_config=config,
                broker_client=object(),
            )
            with self.assertRaisesRegex(FleetBackendError, "must use the Fleet broker"):
                compute.start()

        self.assertEqual(config["base_url"], "inherited")

    def test_watchdog_style_close_cancels_wait_then_releases_one_ticket(self):
        started = threading.Event()
        errors = []

        class WaitingSession(_FleetSession):
            def start(self):
                started.set()
                self.kwargs["sleep"](60)
                raise AssertionError("cancelled wait unexpectedly returned")

        sessions = []

        def session_factory(**kwargs):
            session = WaitingSession(**kwargs)
            sessions.append(session)
            return session

        config = {"provider": "vllm", "model": "fixture", "base_url": "inherited"}
        with patch(
            "aeon.core.sub_agent_environment.select_compute_backend",
            return_value=("broker", "fixture"),
        ):
            compute = SubAgentFleetCompute(
                agent_id="waiting-child",
                model_config=config,
                broker_client=object(),
                session_factory=session_factory,
            )

            def acquire():
                try:
                    compute.start()
                except Exception as exc:
                    errors.append(exc)

            thread = threading.Thread(target=acquire)
            thread.start()
            self.assertTrue(started.wait(timeout=1))
            proof = compute.close(wait_for_start_seconds=1)
            thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], FleetBackendError)
        self.assertEqual(proof, {"state": "released", "compute_state": "inactive"})
        self.assertEqual(sessions[0].close_calls, 1)
        self.assertEqual(config["base_url"], "inherited")


if __name__ == "__main__":
    unittest.main()
