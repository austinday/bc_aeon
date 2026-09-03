"""Hermetic regressions for Aeon's coordinator-owned worker runtime adapter."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from aeon import main as aeon_main
from aeon.core import gpu_queue, qwen_capabilities, qwen_runtime
from aeon.core import qwen_fleet_runtime as fleet
from aeon.core.qwen_capabilities import qwen_runtime_capability
from aeon.core.qwen_runtime import QwenRuntimeError


def _remote_capability():
    capability, manifest = qwen_runtime_capability(
        "qwen38-compact-180-128k", require_enabled=True
    )
    return capability, manifest


def _state() -> dict:
    capability, manifest = _remote_capability()
    return {
        "schema_version": 1,
        "phase": "ready",
        "runtime_capability_key": capability.key,
        "runtime_capability_manifest_sha256": manifest,
        "runtime_adapter": capability.runtime_adapter,
        "host": capability.host,
        "expected_hostname": capability.hostname,
        "physical_gpu": 0,
        "gpu_uuid": "GPU-12345678-abcd",
        "claim_id": "gc-test-remote",
        "owner": "owner-test-remote",
        "run_dir": (
            "/home/aday/.aeon/runtime/qwen38/"
            "aeon-qwen38-vllm-owner-test-remote"
        ),
        "source_manifest_sha256": "1" * 64,
        "model_manifest_sha256": capability.model_manifest_sha256,
        "model_sha256s_sha256": capability.model_sha256s_sha256,
        "container_name": "aeon-qwen38-vllm",
        "container_id": "2" * 64,
        "container_pid": 4321,
        "remote_port": 8033,
        "local_port": 8033,
        "deploy_environment": {"AEON_SERVED_NAME": "qwen"},
        "tunnel_nonce": "3" * 64,
        "tunnel_pid": 8765,
        "tunnel_create_time": 999,
        "updated_at": 1.0,
    }


class RemoteStateTests(unittest.TestCase):
    def test_remote_worker_source_imports_under_isolated_python(self):
        package_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                (
                    "import sys; sys.modules['requests']=None; "
                    "sys.modules['fleet_compute']=None; "
                    "sys.path.insert(0, sys.argv[1]); "
                    "import aeon.scripts.qwen_remote_worker"
                ),
                str(package_root),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fleet_tree_payload_receipts_are_verified_then_canonicalized(self):
        capability, _manifest = _remote_capability()
        source_digest = "1" * 64
        image_digest = str(capability.image_id).removeprefix("sha256:")
        source = qwen_runtime.SourceIdentity(
            package_root=Path("/unused"),
            stage_dir=Path("/unused"),
            manifest_sha256=source_digest,
            manifest_bytes=b"",
            file_sha256=(),
        )

        def binding(
            artifact_id: str,
            kind: str,
            digest: str,
            *,
            size_bytes: int,
            inode_count: int,
        ) -> dict[str, object]:
            return {
                "artifact_id": artifact_id,
                "kind": kind,
                "worker_path": str(fleet._cache_entry_path(digest)),
                "digest_sha256": digest,
                "size_bytes": size_bytes,
                "inode_count": inode_count,
                "filesystem_id": "66306",
                "payload_sha256": digest,
            }

        bindings = {
            fleet.QWEN_SOURCE_CACHE_ARTIFACT_ID: binding(
                fleet.QWEN_SOURCE_CACHE_ARTIFACT_ID,
                "manifested_tree",
                source_digest,
                size_bytes=100,
                inode_count=2,
            ),
            fleet.QWEN_MODEL_CACHE_ARTIFACT_ID: binding(
                fleet.QWEN_MODEL_CACHE_ARTIFACT_ID,
                "manifested_tree",
                capability.model_sha256s_sha256,
                size_bytes=200,
                inode_count=22,
            ),
            fleet.QWEN_IMAGE_CACHE_ARTIFACT_ID: binding(
                fleet.QWEN_IMAGE_CACHE_ARTIFACT_ID,
                "oci_archive",
                image_digest,
                size_bytes=229,
                inode_count=1,
            ),
        }

        request = fleet.qwen_remote_artifact_cache(
            capability, source, bindings
        ).to_request()

        self.assertNotIn("payload_sha256", request["source"])
        self.assertNotIn("payload_sha256", request["model"])
        self.assertEqual(request["image"]["payload_sha256"], image_digest)

        bindings[fleet.QWEN_MODEL_CACHE_ARTIFACT_ID]["payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(QwenRuntimeError, "cache binding changed"):
            fleet.qwen_remote_artifact_cache(capability, source, bindings)

    def test_receipt_binds_capability_lease_runtime_and_tunnel_fields(self):
        state = _state()
        self.assertEqual(fleet._validate_remote_state(state), state)
        mutations = {
            "schema_version": 2,
            "phase": "unknown",
            "runtime_adapter": "local-docker",
            "host": "192.168.0.179",
            "expected_hostname": "wrong",
            "physical_gpu": False,
            "gpu_uuid": "0",
            "claim_id": "bad",
            "owner": "bad owner",
            "run_dir": "/tmp/foreign",
            "source_manifest_sha256": "x" * 64,
            "model_manifest_sha256": "0" * 64,
            "model_sha256s_sha256": "0" * 64,
            "container_name": "bad name",
            "container_id": "short",
            "container_pid": True,
            "remote_port": 80,
            "local_port": 9000,
            "deploy_environment": {1: "bad"},
            "tunnel_nonce": "short",
            "tunnel_pid": True,
            "tunnel_create_time": 0,
            "updated_at": float("nan"),
        }
        for field, value in mutations.items():
            with self.subTest(field=field), self.assertRaises(QwenRuntimeError):
                fleet._validate_remote_state({**state, field: value})

    def test_released_worker_state_validates_as_enabled_capacity(self):
        self.assertEqual(
            fleet._validate_remote_state(_state(), require_enabled=True), _state()
        )

    def test_runtime_liveness_requires_exact_claim_container_and_pid(self):
        state = _state()
        capability, _manifest = _remote_capability()
        active = {
            "ok": True,
            "state": "active",
            "claim_id": state["claim_id"],
            "container_id": state["container_id"],
            "container_pid": state["container_pid"],
            "run_dir": state["run_dir"],
            "physical_gpu": state["physical_gpu"],
            "gpu_uuid": state["gpu_uuid"],
            "owner": state["owner"],
        }
        with patch.object(fleet, "remote_state", return_value=state), patch.object(
            fleet, "_capability_for_state", return_value=(capability, "f" * 64)
        ), patch.object(fleet, "remote_call", return_value=active):
            self.assertEqual(fleet.remote_runtime_liveness(), "active")
            for field, value in (
                ("claim_id", "gc-other"),
                ("container_id", "4" * 64),
                ("container_pid", 4322),
            ):
                with self.subTest(field=field), patch.object(
                    fleet, "remote_call", return_value={**active, field: value}
                ):
                    self.assertEqual(fleet.remote_runtime_liveness(), "ambiguous")

    def test_pre_cache_ready_receipt_still_probes_and_stops(self):
        old_source = (
            "49ccbbb5bd4ef96f1fef48added9e7625838acb0e8e296dc8eff0b2003fd9491"
        )
        old_manifest = (
            "be70339f04d54ba9a2fc71267d1bfa9edd3fec6687dd721f27523c12c4674981"
        )
        state = {
            **_state(),
            "source_manifest_sha256": old_source,
            "runtime_capability_manifest_sha256": old_manifest,
        }
        capability, _current_manifest = _remote_capability()
        status = {
            "ok": True,
            "state": "active",
            **{
                field: state[field]
                for field in (
                    "run_dir",
                    "physical_gpu",
                    "gpu_uuid",
                    "claim_id",
                    "owner",
                    "container_id",
                    "container_pid",
                )
            },
        }
        stopped = {
            "ok": True,
            "state": "stopped",
            "scratch_cleaned": True,
            **{
                field: state[field]
                for field in (
                    "run_dir",
                    "physical_gpu",
                    "gpu_uuid",
                    "claim_id",
                    "owner",
                )
            },
        }
        calls = []

        def remote_call(_capability, source, action, _request, **_kwargs):
            calls.append((source, action))
            return {
                "status": status,
                "stop": stopped,
                "clear": {"ok": True, "state": "cleared", "receipt_absent": True},
            }[action]

        with tempfile.TemporaryDirectory() as temp:
            receipt = Path(temp) / "remote-runtime.json"
            receipt.write_text("{}", encoding="utf-8")
            receipt.chmod(0o600)
            with patch.object(fleet, "REMOTE_STATE_FILE", receipt), patch.object(
                fleet, "remote_state", return_value=state
            ), patch.object(
                fleet, "_capability_for_state", return_value=(capability, old_manifest)
            ), patch.object(fleet, "remote_call", side_effect=remote_call), patch.object(
                fleet, "stop_tunnel", return_value=True
            ), patch.object(fleet, "_private_json_write"):
                self.assertEqual(fleet.remote_runtime_liveness(), "active")
                source = qwen_runtime.SourceIdentity(
                    package_root=Path("/unused"),
                    stage_dir=Path("/unused"),
                    manifest_sha256=old_source,
                    manifest_bytes=b"",
                    file_sha256=(),
                )
                self.assertTrue(
                    fleet.stop_managed_remote_runtime(
                        capability,
                        old_manifest,
                        source,
                        release_reason="rolling compatibility test",
                        release_claim=False,
                    )
                )
        self.assertEqual(
            calls,
            [
                (old_source, "status"),
                (old_source, "status"),
                (old_source, "stop"),
                (old_source, "clear"),
            ],
        )
        command = fleet._remote_command(capability, old_source, "status", None)
        self.assertIn(str(fleet.REMOTE_RELEASE_ROOT / old_source), " ".join(command))
        self.assertNotIn("-I", command)


class RemoteTransportTests(unittest.TestCase):
    def test_ssh_and_rsync_are_fixed_nonmultiplexed_and_low_priority(self):
        capability, _manifest = _remote_capability()
        ssh = fleet._ssh_base(capability)
        joined = " ".join(ssh)
        for token in (
            "BatchMode=yes",
            "StrictHostKeyChecking=yes",
            "IdentitiesOnly=yes",
            "ControlMaster=no",
            "ControlPath=none",
            "ControlPersist=no",
            "ServerAliveInterval=5",
            "ServerAliveCountMax=6",
        ):
            self.assertIn(token, joined)
        source = Path(fleet.__file__).read_text(encoding="utf-8")
        self.assertIn('"--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync"', source)
        self.assertIn('"/usr/bin/bash",\n        str(REMOTE_WRAPPER),\n        "/usr/bin/rsync"', source)

    def test_tunnel_identity_is_full_argv_and_nonce_bound(self):
        state = _state()
        capability, _manifest = _remote_capability()
        expected = fleet._tunnel_argv(
            capability,
            state["local_port"],
            state["remote_port"],
            state["tunnel_nonce"],
        )
        self.assertIn(f"tunnel-{state['tunnel_nonce']}.sock", " ".join(expected))
        with patch.object(fleet, "_process_create_time", return_value=999), patch.object(
            fleet, "_process_argv", return_value=expected
        ):
            self.assertTrue(fleet.tunnel_is_exact(state))
        with patch.object(fleet, "_process_create_time", return_value=999), patch.object(
            fleet, "_process_argv", return_value=[*expected, "extra"]
        ):
            self.assertFalse(fleet.tunnel_is_exact(state))

    def test_tunnel_lifecycle_never_scans_or_adopts_global_processes(self):
        source = Path(fleet.__file__).read_text(encoding="utf-8")
        self.assertNotIn('Path("/proc").iterdir()', source)
        self.assertNotIn("def _tunnel_candidates", source)

    def test_tunnel_refuses_preexisting_pidless_launch_intent(self):
        state = {
            **_state(),
            "tunnel_pid": None,
            "tunnel_create_time": None,
        }
        capability, _manifest = _remote_capability()
        receipt = Path("/tmp/not-written.json")
        with patch.object(
            fleet, "_remote_state_entry", return_value=(state, receipt, True)
        ), patch.object(fleet, "_private_json_write") as write, patch.object(
            fleet.subprocess, "Popen"
        ) as popen:
            with self.assertRaisesRegex(QwenRuntimeError, "intent is ambiguous"):
                fleet.start_tunnel(capability, state)
        write.assert_not_called()
        popen.assert_not_called()

    def test_tunnel_health_response_is_streamed_and_bounded(self):
        class Response:
            headers = {"content-length": str(64 * 1024 + 1)}

            def close(self):
                self.closed = True

            def iter_content(self, **_kwargs):
                raise AssertionError("advertised oversize must fail before reading")

        response = Response()
        with self.assertRaisesRegex(QwenRuntimeError, "exceeded"):
            fleet._bounded_loopback_body(response, 64 * 1024)
        self.assertTrue(response.closed)

    def test_stop_tunnel_refuses_identity_drift_after_sigterm(self):
        state = _state()
        with patch.object(
            fleet, "tunnel_liveness", side_effect=["active", "ambiguous"]
        ), patch.object(fleet.os, "kill") as kill:
            self.assertFalse(fleet.stop_tunnel(state))
        kill.assert_called_once_with(state["tunnel_pid"], fleet.signal.SIGTERM)

    def test_stop_tunnel_accepts_only_exact_pid_absence_after_sigterm(self):
        state = _state()
        with patch.object(
            fleet, "tunnel_liveness", side_effect=["active", "gone"]
        ), patch.object(fleet.os, "kill") as kill:
            self.assertTrue(fleet.stop_tunnel(state))
        kill.assert_called_once_with(state["tunnel_pid"], fleet.signal.SIGTERM)

    def test_only_observational_remote_calls_retry(self):
        capability, _manifest = _remote_capability()
        calls = []

        def runner(*_args, **_kwargs):
            calls.append(1)
            return subprocess.CompletedProcess([], 255, "", "")

        with self.assertRaises(QwenRuntimeError):
            fleet.remote_call(
                capability, "1" * 64, "status", {}, timeout=1, command_runner=runner
            )
        self.assertEqual(len(calls), 3)
        calls.clear()
        with self.assertRaises(QwenRuntimeError):
            fleet.remote_call(
                capability, "1" * 64, "start", {}, timeout=1, command_runner=runner
            )
        self.assertEqual(len(calls), 1)


class PlacementPlanTests(unittest.TestCase):
    def test_compact_environment_is_derived_from_capability_and_uuid_lease(self):
        capability, _manifest = _remote_capability()
        capability = replace(capability, enabled=True)
        base = {
            "AEON_DEPLOY_PLAN": json.dumps(
                {
                    "tier": "solo",
                    "context_limit": 114688,
                    "nodes": [{"ctx": 114688, "devices": "old"}],
                }
            )
        }
        lease = {
            "claim_id": "gc-test-remote",
            "owner": "owner-test-remote",
            "run_dir": "/home/aday/.aeon/runtime/qwen38/test",
            "gpu_uuid": "GPU-12345678-abcd",
        }
        result = fleet.capability_deploy_environment(capability, base, lease)
        plan = json.loads(result["AEON_DEPLOY_PLAN"])
        self.assertEqual(plan["context_limit"], 131072)
        self.assertEqual(plan["nodes"][0]["ctx"], 131072)
        self.assertEqual(plan["nodes"][0]["devices"], lease["gpu_uuid"])
        self.assertEqual(result["AEON_LLM_VRAM_BUDGET_GB"], "41.25")
        self.assertEqual(result["AEON_GPU_MEM_UTIL"], "0.7")
        self.assertEqual(result["AEON_MAX_NUM_SEQS"], "8")
        self.assertEqual(result["AEON_MAX_NUM_BATCHED"], "8192")
        self.assertEqual(result["GPU_RESERVE_GB"], "6")

    def test_start_prefers_local_then_spills_to_enabled_worker(self):
        local, manifest = qwen_capabilities.active_qwen_runtime_capability()
        worker, _manifest = _remote_capability()
        worker = replace(worker, enabled=True, disabled_reason=None)
        lease = {
            "runtime_capability_key": worker.key,
            "runtime_capability_manifest_sha256": manifest,
            "runtime_adapter": worker.runtime_adapter,
            "host": worker.host,
            "physical_gpu": 0,
            "gpu_uuid": "GPU-12345678-abcd",
            "claim_id": "gc-test-remote",
            "owner": "owner-test-remote",
            "run_dir": "/home/aday/.aeon/runtime/qwen38/aeon-test-owner",
        }
        reserve_targets = []

        def reserve(**kwargs):
            reserve_targets.append((kwargs["host"], kwargs["gpu_id"]))
            if kwargs["host"] == local.host:
                raise TimeoutError("local full")
            return lease

        class Heartbeat:
            def start(self, **_kwargs):
                return self

            def raise_if_failed(self):
                return None

            def promote_to_exact_pid(self):
                return 4321

            def stop(self):
                return None

        config = {
            "model": "Qwen test",
            "container_name": "aeon-qwen38-vllm",
            "health_port": 8033,
            "_deploy_env": {
                "AEON_LOCAL_MODEL_DIR": "model",
                "AEON_SERVED_NAME": "qwen",
                "AEON_DEPLOY_PLAN": json.dumps(
                    {
                        "tier": "solo",
                        "context_limit": 114688,
                        "image": "aeon_vllm:latest",
                        "nodes": [{"ctx": 114688, "devices": "old"}],
                    }
                ),
                "AEON_LLM_VRAM_BUDGET_GB": "48.7",
            },
        }
        remote_receipt = {
            "image_id": worker.image_id,
            "model_manifest_sha256": worker.model_manifest_sha256,
            "model_sha256s_sha256": worker.model_sha256s_sha256,
        }
        with patch.object(qwen_runtime, "current_runtime_state", return_value=None), patch.object(
            fleet, "remote_state", return_value=None
        ), patch.object(
            qwen_capabilities,
            "enabled_qwen_runtime_capabilities",
            return_value=((local, worker), manifest),
        ), patch.object(
            qwen_runtime, "load_artifact_identity", return_value=object()
        ), patch.object(
            qwen_runtime, "local_image_id", return_value=local.image_id
        ), patch.object(
            qwen_runtime, "local_image_size", return_value=123
        ), patch.object(
            fleet, "remote_preflight", return_value=("1" * 64, remote_receipt)
        ), patch.object(
            gpu_queue, "reserve_named_lease", side_effect=reserve
        ), patch.object(
            qwen_runtime, "verify_coordinator_lease", side_effect=lambda item: item
        ), patch.object(
            gpu_queue, "PeriodicLeaseHeartbeat", return_value=Heartbeat()
        ), patch.object(
            fleet, "start_managed_remote_runtime", return_value={"phase": "ready"}
        ) as start_remote:
            self.assertTrue(aeon_main.start_llamacpp_server(config))
        self.assertEqual(
            reserve_targets,
            [("192.168.0.177", 0), ("192.168.0.180", 0)],
        )
        start_remote.assert_called_once()


if __name__ == "__main__":
    unittest.main()
