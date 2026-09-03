"""Hermetic Fleet/lifecycle contract tests for self-modification verification."""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from aeon.core.sub_agent_environment import (
    CHILD_FLEET_CONFIGURATION_KEYS,
    VERIFICATION_PREBOUND_NONCE_ENV,
    VERIFICATION_PREBOUND_RECEIPT,
)
from aeon.core.sub_agent_state import CPU_SANDBOX_SLICE_ENV, sub_agent_systemd_units
from aeon.tools.verify_modification import VerifySelfModificationTool


AGENT_ID = "12345678-1234-1234-1234-123456789abc"
ENDPOINT = "http://127.0.0.1:8443/v1"


class _Process:
    pid = 4242
    returncode = 0

    def __init__(self, output_dir, events):
        self.output_dir = Path(output_dir)
        self.events = events
        self.killed = False

    def communicate(self, timeout=None):
        self.returncode = 0
        return "candidate stdout", ""

    def kill(self):
        self.killed = True
        self.events.append("launcher-kill")

    def wait(self, timeout=None):
        self.returncode = -9
        return self.returncode


class _Compute:
    required = True

    def __init__(self, events, model_config, *, close_error=None):
        self.events = events
        self.model_config = model_config
        self.close_error = close_error

    def start(self):
        self.events.append("fleet-start")
        self.model_config["base_url"] = ENDPOINT
        return ENDPOINT

    def assert_prebound_endpoint_healthy(self, endpoint):
        self.events.append("fleet-health")
        if endpoint != ENDPOINT:
            raise AssertionError("wrong endpoint")

    def close(self):
        self.events.append("fleet-close")
        if self.close_error is not None:
            raise self.close_error
        return {"state": "released", "compute_state": "inactive"}


class VerifyModificationFleetTests(unittest.TestCase):
    def _worker(self, root):
        return types.SimpleNamespace(
            instance_id="principal-instance",
            request_id="principal-request",
            debug_mode=False,
            model_config={
                "provider": "vllm",
                "model": "Qwen3.8-27B-ARA-NVFP4-MTP",
                "api_model": "Qwen3.8-27B-ARA-NVFP4-MTP",
                "base_url": "http://127.0.0.1:9999/v1",
                "context_limit": 65536,
                "multimodal": True,
                "container_name": "must-not-reach-candidate",
                "GPU_AGENT_CLAIM_ID": "must-not-reach-candidate",
            },
            sub_agent_output_dir=lambda: Path(root) / "sub_agents",
            blackboard_path=lambda: Path(root) / "blackboard.jsonl",
        )

    def test_parent_acquires_before_candidate_and_releases_after_exact_slice(self):
        events = []
        captured = {}
        with tempfile.TemporaryDirectory() as temporary:
            worker = self._worker(temporary)
            tool = VerifySelfModificationTool(worker=worker)
            compute_holder = {}

            def compute_factory(*, model_config, **_kwargs):
                compute = _Compute(events, model_config)
                compute_holder["compute"] = compute
                return compute

            def preflight(_agent_id):
                events.append("unit-preflight")

            def popen(argv, **kwargs):
                events.append("popen")
                captured["argv"] = argv
                captured["env"] = kwargs["env"]
                output_dir = Path(temporary) / "sub_agents" / AGENT_ID
                receipt = json.loads(
                    (output_dir / VERIFICATION_PREBOUND_RECEIPT).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(receipt["endpoint"], ENDPOINT)
                self.assertNotIn("ticket", json.dumps(receipt).lower())
                (output_dir / "status.txt").write_text("COMPLETED", encoding="utf-8")
                (output_dir / "output.json").write_text(
                    json.dumps({"result": "verified"}), encoding="utf-8"
                )
                (output_dir / "agent.log").write_text("done\n", encoding="utf-8")
                return _Process(output_dir, events)

            def capture(*_args, **_kwargs):
                events.append("capture")
                return {"schema": 2, "agent_id": AGENT_ID, "pid": 4242}

            def terminate(_output_dir):
                events.append("slice-retire")
                return True

            unsafe = {
                "AEON_FLEET_SOCKET": "/principal/broker.sock",
                "AEON_FLEET_PROFILE": "principal-profile",
                "AEON_FLEET_TICKET": "principal-ticket",
                "GPU_AGENT_CLAIM_ID": "principal-claim",
                "CUDA_VISIBLE_DEVICES": "GPU-principal",
            }
            with (
                patch(
                    "aeon.tools.verify_modification._model_verification_boundary_available",
                    return_value=True,
                ),
                patch.object(tool, "_run_test_gate", return_value=None),
                patch("aeon.tools.verify_modification.uuid.uuid4", return_value=uuid.UUID(AGENT_ID)),
                patch(
                    "aeon.tools.verify_modification.assert_sub_agent_systemd_units_available",
                    side_effect=preflight,
                ),
                patch(
                    "aeon.tools.verify_modification.SubAgentFleetCompute",
                    side_effect=compute_factory,
                ),
                patch("aeon.tools.verify_modification.subprocess.Popen", side_effect=popen),
                patch(
                    "aeon.tools.verify_modification.capture_sub_agent_process",
                    side_effect=capture,
                ),
                patch(
                    "aeon.tools.verify_modification.terminate_sub_agent",
                    side_effect=terminate,
                ),
                patch.dict(os.environ, unsafe, clear=False),
            ):
                result = tool.execute("Exercise the modified tool", timeout=30)

        self.assertIn("VERIFICATION SUCCESSFUL", result)
        self.assertLess(events.index("unit-preflight"), events.index("popen"))
        self.assertLess(events.index("fleet-start"), events.index("popen"))
        self.assertLess(events.index("capture"), events.index("slice-retire"))
        self.assertLess(events.index("slice-retire"), events.index("fleet-close"))
        child_env = captured["env"]
        for key in CHILD_FLEET_CONFIGURATION_KEYS:
            self.assertNotIn(key, child_env)
        self.assertNotIn("AEON_FLEET_TICKET", child_env)
        self.assertNotIn("GPU_AGENT_CLAIM_ID", child_env)
        self.assertEqual(child_env["CUDA_VISIBLE_DEVICES"], "void")
        self.assertEqual(
            child_env[CPU_SANDBOX_SLICE_ENV], sub_agent_systemd_units(AGENT_ID)[1]
        )
        self.assertRegex(child_env[VERIFICATION_PREBOUND_NONCE_ENV], r"^[0-9a-f]{64}$")
        model_index = captured["argv"].index("--model_config") + 1
        child_model = json.loads(captured["argv"][model_index])
        self.assertEqual(child_model["base_url"], ENDPOINT)
        self.assertNotIn("container_name", child_model)
        self.assertNotIn("GPU_AGENT_CLAIM_ID", child_model)

    def test_unresolved_exact_release_overrides_candidate_success(self):
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            worker = self._worker(temporary)
            tool = VerifySelfModificationTool(worker=worker)

            def compute_factory(*, model_config, **_kwargs):
                return _Compute(
                    events,
                    model_config,
                    close_error=RuntimeError("synthetic unresolved release"),
                )

            def popen(_argv, **_kwargs):
                output_dir = Path(temporary) / "sub_agents" / AGENT_ID
                (output_dir / "status.txt").write_text("COMPLETED", encoding="utf-8")
                (output_dir / "output.json").write_text(
                    json.dumps({"result": "untrusted until release"}), encoding="utf-8"
                )
                return _Process(output_dir, events)

            with (
                patch(
                    "aeon.tools.verify_modification._model_verification_boundary_available",
                    return_value=True,
                ),
                patch.object(tool, "_run_test_gate", return_value=None),
                patch("aeon.tools.verify_modification.uuid.uuid4", return_value=uuid.UUID(AGENT_ID)),
                patch(
                    "aeon.tools.verify_modification.assert_sub_agent_systemd_units_available"
                ),
                patch(
                    "aeon.tools.verify_modification.SubAgentFleetCompute",
                    side_effect=compute_factory,
                ),
                patch("aeon.tools.verify_modification.subprocess.Popen", side_effect=popen),
                patch(
                    "aeon.tools.verify_modification.capture_sub_agent_process",
                    return_value={"schema": 2, "agent_id": AGENT_ID, "pid": 4242},
                ),
                patch("aeon.tools.verify_modification.terminate_sub_agent", return_value=True),
            ):
                result = tool.execute("Exercise the modified tool", timeout=30)

        self.assertIn("exact verification Fleet ticket release remains unresolved", result)
        self.assertEqual(events.count("fleet-close"), 2)

    def test_model_verification_is_fail_closed_without_exact_destination_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool = VerifySelfModificationTool(worker=self._worker(temporary))
            with patch.object(tool, "_run_test_gate") as gate, patch(
                "aeon.tools.verify_modification.SubAgentFleetCompute"
            ) as compute, patch(
                "aeon.tools.verify_modification.subprocess.Popen"
            ) as popen:
                result = tool.execute("Exercise the modified tool")

        self.assertIn("VERIFICATION BLOCKED", result)
        self.assertIn("preconnected-FD/proxy", result)
        gate.assert_not_called()
        compute.assert_not_called()
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
