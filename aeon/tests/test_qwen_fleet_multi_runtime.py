"""Hermetic coverage for two exact standard Qwen worker runtimes."""

from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aeon.core import fleet_adapter as adapter_module
from aeon.core import qwen_fleet_runtime as fleet
from aeon.core.fleet_adapter import AeonQwenFleetAdapter
from aeon.core.qwen_capabilities import (
    RTX5000_178_RELEASE_CAPABILITY_KEY,
    qwen_runtime_capability,
)
from aeon.core.qwen_runtime import (
    QwenRuntimeError,
    RUNTIME_STATE_FILE,
    SourceIdentity,
)
from aeon.scripts import qwen_remote_worker as worker
from fleet_compute.models import ProbeState


def _state(runtime_id: str, physical_gpu: int) -> dict:
    capability, manifest = qwen_runtime_capability(
        "qwen38-compact-180-128k", require_enabled=True
    )
    run_dir = f"/home/aday/.local/state/fleet-compute/runs/{runtime_id}"
    resources = fleet.fleet_remote_runtime_resources(run_dir, physical_gpu)
    return {
        "schema_version": 1,
        "phase": "ready",
        "runtime_capability_key": capability.key,
        "runtime_capability_manifest_sha256": manifest,
        "runtime_adapter": capability.runtime_adapter,
        "host": capability.host,
        "expected_hostname": capability.hostname,
        "physical_gpu": physical_gpu,
        "gpu_uuid": f"GPU-12345678-0000-0000-0000-00000000000{physical_gpu}",
        "claim_id": f"gc-test-runtime-{physical_gpu}",
        "owner": f"owner-test-runtime-{physical_gpu}",
        "run_dir": run_dir,
        "source_manifest_sha256": "1" * 64,
        "model_manifest_sha256": capability.model_manifest_sha256,
        "model_sha256s_sha256": capability.model_sha256s_sha256,
        "container_name": resources["container_name"],
        "container_id": str(physical_gpu + 2) * 64,
        "container_pid": 4300 + physical_gpu,
        "remote_port": resources["remote_port"],
        "local_port": resources["local_port"],
        "deploy_environment": {"AEON_SERVED_NAME": "qwen"},
        "tunnel_nonce": None,
        "tunnel_pid": None,
        "tunnel_create_time": None,
        "updated_at": time.time(),
    }


def _runtime_record(state: dict) -> dict:
    return {
        **{
            key: state[key]
            for key in (
                "claim_id",
                "owner",
                "host",
                "physical_gpu",
                "gpu_uuid",
                "run_dir",
            )
        },
        "pid": state["container_pid"],
        "runtime_id": Path(state["run_dir"]).name,
        "process_absent": 1,
        "state": "stopped",
        "process_identity": (
            f"{state['runtime_capability_key']}:{state['container_id']}"
        ),
    }


def _precontainer_state(runtime_id: str, physical_gpu: int) -> dict:
    return {
        **_state(runtime_id, physical_gpu),
        "phase": "starting",
        "container_id": None,
        "container_pid": None,
        "tunnel_nonce": None,
        "tunnel_pid": None,
        "tunnel_create_time": None,
        "updated_at": (
            time.time() - fleet.REMOTE_STARTUP_TIMEOUT_SECONDS - 1
        ),
    }


def _precontainer_runtime(state: dict) -> dict:
    return {
        **_runtime_record(state),
        "pid": None,
        "process_identity": None,
        "endpoint": None,
        "process_absent": 0,
        "state": "quarantined",
    }


def _uncommitted_container_runtime(state: dict) -> dict:
    return {
        **_runtime_record(state),
        "pid": None,
        "process_identity": None,
        "endpoint": None,
        "process_absent": 0,
        "state": "quarantined",
    }


def _atomic_worker_recovery(state: dict, *, receipt_absent: bool = False) -> dict:
    return {
        "ok": True,
        "state": "recovered",
        "controller_protocol": 1,
        "process_absent": True,
        "worker_receipt_absent": receipt_absent,
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


def _atomic_uncommitted_recovery(
    state: dict, container_id: str, *, receipt_absent: bool = False
) -> dict:
    return {
        "ok": True,
        "state": "recovered",
        "controller_protocol": 2,
        "process_absent": True,
        "worker_receipt_absent": receipt_absent,
        "container_id": container_id,
        "container_pid": state["container_pid"],
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


def _solo_deploy_environment() -> dict[str, str]:
    return {
        "AEON_DEPLOY_PLAN": json.dumps(
            {
                "entry_name": "Qwen3.8-27B-ARA-NVFP4-MTP",
                "tier": "solo",
                "image": "sha256:" + "a" * 64,
                "container_name": "aeon_qwen_base",
                "all_containers": ["aeon_qwen_base"],
                "health_port": 8033,
                "lb_port": 8033,
                "nodes": [
                    {
                        "container": "aeon_qwen_base",
                        "devices": "GPU-12345678-0000-0000-0000-000000000000",
                        "port": 8033,
                        "ctx": 131072,
                        "cpu_offload_gib": 0.0,
                    }
                ],
            },
            sort_keys=True,
        ),
        "AEON_MAX_NUM_SEQS": "8",
    }


def _worker_request(state: dict) -> tuple[dict, object, str]:
    capability, manifest = qwen_runtime_capability(
        "qwen38-compact-180-128k", require_enabled=True
    )
    resources = fleet.fleet_remote_runtime_resources(
        state["run_dir"], state["physical_gpu"]
    )
    request = fleet._runtime_request_base(
        capability,
        manifest,
        state["source_manifest_sha256"],
        state,
        resources["orchestrator_state_path"],
    )
    return request, capability, manifest


class MultiRuntimeReceiptTests(unittest.TestCase):
    def test_remote_hosts_have_distinct_orchestrator_tunnel_ports(self):
        run_dir = "/home/aday/.local/state/fleet-compute/runs/fr-" + "9" * 32
        on_180 = fleet.fleet_remote_runtime_resources(
            run_dir, 0, host="192.168.0.180"
        )
        on_178 = fleet.fleet_remote_runtime_resources(
            run_dir, 0, host="192.168.0.178"
        )

        self.assertEqual(on_180["remote_port"], on_178["remote_port"])
        self.assertNotEqual(on_180["local_port"], on_178["local_port"])

    def test_178_future_key_uses_shared_remote_contract_after_registry_proof(self):
        """Exercise transport shape only; the production registry stays disabled."""

        current, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        qualified_178 = replace(
            current,
            key=RTX5000_178_RELEASE_CAPABILITY_KEY,
            host="192.168.0.178",
            hostname="DAY2XRTX5000",
        )
        state = _state("fr-" + "7" * 32, 0)
        resources = fleet.fleet_remote_runtime_resources(
            state["run_dir"], state["physical_gpu"], host=qualified_178.host
        )
        state.update(
            {
                "runtime_capability_key": qualified_178.key,
                "host": qualified_178.host,
                "expected_hostname": qualified_178.hostname,
                "local_port": resources["local_port"],
            }
        )
        with patch.object(
            fleet,
            "qwen_runtime_capability",
            return_value=(qualified_178, manifest),
        ):
            checked = fleet._validate_remote_state(
                state, require_enabled=True, legacy_binding=False
            )
        self.assertEqual(checked["runtime_capability_key"], qualified_178.key)

    def test_retired_remote_manifest_is_teardown_only_and_exactly_key_scoped(self):
        state = _state("fr-" + "8" * 32, 0)
        state["runtime_capability_manifest_sha256"] = (
            "52e2d54b70c14eefac3d5cae796b1f1ce40ececb95961a42d1c8ec6457254b6a"
        )

        checked = fleet._validate_remote_state(
            state, require_enabled=False, legacy_binding=False
        )
        self.assertEqual(
            checked["runtime_capability_key"], "qwen38-compact-180-128k"
        )
        with self.assertRaises(QwenRuntimeError):
            fleet._validate_remote_state(
                state, require_enabled=True, legacy_binding=False
            )
        with self.assertRaises(QwenRuntimeError):
            fleet._validate_remote_state(
                {
                    **state,
                    "runtime_capability_manifest_sha256": "0" * 64,
                },
                require_enabled=False,
                legacy_binding=False,
            )

    def test_fleet_plan_binding_is_exact_without_mutating_the_caller(self):
        state = _state("fr-" + "e" * 32, 1)
        resources = fleet.fleet_remote_runtime_resources(
            state["run_dir"], state["physical_gpu"]
        )
        environment = _solo_deploy_environment()
        original = dict(environment)

        bound = fleet._bind_fleet_runtime_deploy_environment(
            environment,
            container_name=resources["container_name"],
            port=resources["remote_port"],
        )

        self.assertEqual(environment, original)
        self.assertIsNot(bound, environment)
        self.assertEqual(bound["AEON_MAX_NUM_SEQS"], "8")
        plan = json.loads(bound["AEON_DEPLOY_PLAN"])
        self.assertEqual(plan["container_name"], resources["container_name"])
        self.assertEqual(plan["all_containers"], [resources["container_name"]])
        self.assertEqual(plan["health_port"], resources["remote_port"])
        self.assertEqual(plan["lb_port"], resources["remote_port"])
        self.assertEqual(plan["nodes"][0]["container"], resources["container_name"])
        self.assertEqual(plan["nodes"][0]["port"], resources["remote_port"])

    def test_fleet_plan_binding_rejects_incoherent_base_aliases(self):
        environment = _solo_deploy_environment()
        plan = json.loads(environment["AEON_DEPLOY_PLAN"])
        plan["nodes"][0]["port"] += 1
        environment["AEON_DEPLOY_PLAN"] = json.dumps(plan)

        with self.assertRaisesRegex(QwenRuntimeError, "coherent one-node release"):
            fleet._bind_fleet_runtime_deploy_environment(
                environment,
                container_name="aeon-qwen38-standard-fr-" + "e" * 32 + "-gpu0",
                port=fleet.FLEET_REMOTE_PORT_BASE,
            )

    def test_two_gpu_slots_have_distinct_exact_resources(self):
        first = _state("fr-" + "a" * 32, 0)
        second = _state("fr-" + "b" * 32, 1)

        self.assertEqual(
            fleet._validate_remote_state(first, legacy_binding=False), first
        )
        self.assertEqual(
            fleet._validate_remote_state(second, legacy_binding=False), second
        )
        for field in (
            "run_dir",
            "container_name",
            "remote_port",
            "local_port",
        ):
            self.assertNotEqual(first[field], second[field])
        with self.assertRaises(QwenRuntimeError):
            fleet._validate_remote_state(
                {**first, "local_port": second["local_port"]},
                legacy_binding=False,
            )
        with self.assertRaises(QwenRuntimeError):
            fleet._validate_remote_state(
                {**first, "container_name": second["container_name"]},
                legacy_binding=False,
            )

    def test_per_runtime_receipts_and_worker_paths_are_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(fleet, "REMOTE_STATE_ROOT", root / "orchestrator"), patch.object(
                fleet, "WORKER_STATE_ROOT", root / "worker"
            ), patch.object(fleet, "REMOTE_STATE_FILE", root / "legacy.json"):
                first = _state("fr-" + "c" * 32, 0)
                second = _state("fr-" + "d" * 32, 1)
                first_resources = fleet.fleet_remote_runtime_resources(
                    first["run_dir"], first["physical_gpu"]
                )
                second_resources = fleet.fleet_remote_runtime_resources(
                    second["run_dir"], second["physical_gpu"]
                )
                fleet._private_json_write(
                    first_resources["orchestrator_state_path"], first
                )
                fleet._private_json_write(
                    second_resources["orchestrator_state_path"], second
                )

                self.assertEqual(fleet.remote_state(first["run_dir"]), first)
                self.assertEqual(fleet.remote_state(second["run_dir"]), second)
                first_request = fleet._worker_request_binding(
                    first, first_resources["orchestrator_state_path"]
                )
                second_request = fleet._worker_request_binding(
                    second, second_resources["orchestrator_state_path"]
                )
                self.assertEqual(
                    worker._runtime_binding(first_request)[1],
                    first_resources["worker_state_path"],
                )
                self.assertEqual(
                    worker._runtime_binding(second_request)[1],
                    second_resources["worker_state_path"],
                )
                self.assertNotEqual(
                    first_request["worker_state_path"],
                    second_request["worker_state_path"],
                )

    def test_gone_tunnel_recovery_is_scoped_to_the_exact_runtime_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(fleet, "REMOTE_STATE_ROOT", root / "orchestrator"), patch.object(
                fleet, "WORKER_STATE_ROOT", root / "worker"
            ), patch.object(fleet, "REMOTE_STATE_FILE", root / "legacy.json"):
                first = {
                    **_state("fr-" + "1" * 32, 0),
                    "tunnel_nonce": "e" * 64,
                    "tunnel_pid": 9999,
                    "tunnel_create_time": 1234,
                }
                second = _state("fr-" + "2" * 32, 1)
                first_path = fleet.fleet_remote_runtime_resources(
                    first["run_dir"], first["physical_gpu"]
                )["orchestrator_state_path"]
                second_path = fleet.fleet_remote_runtime_resources(
                    second["run_dir"], second["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(first_path, first)
                fleet._private_json_write(second_path, second)

                def restart(_capability, observed, *, receipt_path):
                    self.assertEqual(receipt_path, first_path)
                    self.assertIsNone(observed["tunnel_nonce"])
                    self.assertIsNone(observed["tunnel_pid"])
                    self.assertIsNone(observed["tunnel_create_time"])
                    self.assertEqual(fleet.remote_state(second["run_dir"]), second)
                    recovered = {
                        **observed,
                        "tunnel_nonce": "f" * 64,
                        "tunnel_pid": 5555,
                        "tunnel_create_time": 5678,
                    }
                    fleet._private_json_write(receipt_path, recovered)
                    return recovered

                with patch.object(
                    fleet, "remote_runtime_liveness", return_value="active"
                ) as runtime_liveness, patch.object(
                    fleet, "tunnel_liveness", return_value="gone"
                ), patch.object(
                    fleet, "start_tunnel", side_effect=restart
                ) as start:
                    recovered = fleet.restore_managed_remote_tunnel(
                        first["run_dir"]
                    )

                self.assertEqual(recovered["tunnel_pid"], 5555)
                self.assertEqual(fleet.remote_state(first["run_dir"]), recovered)
                self.assertEqual(fleet.remote_state(second["run_dir"]), second)
                runtime_liveness.assert_called_once_with(first["run_dir"])
                start.assert_called_once()

    def test_ambiguous_tunnel_identity_is_never_recreated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(fleet, "REMOTE_STATE_ROOT", root / "orchestrator"), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = {
                    **_state("fr-" + "3" * 32, 0),
                    "tunnel_nonce": "d" * 64,
                    "tunnel_pid": 8888,
                    "tunnel_create_time": 4321,
                }
                receipt_path = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt_path, state)

                with patch.object(
                    fleet, "remote_runtime_liveness", return_value="active"
                ), patch.object(
                    fleet, "tunnel_liveness", return_value="ambiguous"
                ), patch.object(
                    fleet, "start_tunnel"
                ) as start:
                    with self.assertRaisesRegex(
                        QwenRuntimeError, "tunnel identity is ambiguous"
                    ):
                        fleet.restore_managed_remote_tunnel(state["run_dir"])

                start.assert_not_called()
                self.assertEqual(fleet.remote_state(state["run_dir"]), state)

    def test_precontainer_recovery_accepts_only_exact_atomic_worker_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = _precontainer_state("fr-" + "4" * 32, 0)
                resources = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )
                fleet._private_json_write(
                    resources["orchestrator_state_path"], state
                )
                before = resources["orchestrator_state_path"].read_bytes()
                with patch.object(
                    fleet,
                    "remote_call",
                    return_value=_atomic_worker_recovery(state),
                ) as recover:
                    self.assertTrue(
                        fleet.recover_remote_precontainer_intent(state["run_dir"])
                    )

                self.assertEqual(
                    resources["orchestrator_state_path"].read_bytes(), before
                )
                self.assertEqual(recover.call_args.args[2], "recover-precontainer")
                request = recover.call_args.args[3]
                self.assertEqual(
                    request["worker_state_path"],
                    str(resources["worker_state_path"]),
                )
                self.assertEqual(request["claim_id"], state["claim_id"])
                command = fleet._remote_command(
                    qwen_runtime_capability(
                        "qwen38-compact-180-128k", require_enabled=True
                    )[0],
                    state["source_manifest_sha256"],
                    "recover-precontainer",
                )
                self.assertEqual(command[-1], "recover-precontainer")

    def test_precontainer_recovery_rejects_old_or_mismatched_protocol_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = _precontainer_state("fr-" + "6" * 32, 0)
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                exact = _atomic_worker_recovery(state)
                refused = (
                    {
                        "ok": True,
                        "state": "gone",
                        "phase": "preparing",
                        "container_pid": None,
                        "container_id": None,
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
                        "scratch_cleaned": False,
                    },
                    {**exact, "claim_id": "gc-another-lease"},
                    {**exact, "controller_protocol": 0},
                    {**exact, "controller_protocol": True},
                    {**exact, "process_absent": False},
                    {key: value for key, value in exact.items() if key != "owner"},
                )
                for response in refused:
                    with self.subTest(response=response), patch.object(
                        fleet, "remote_call", return_value=response
                    ):
                        self.assertFalse(
                            fleet.recover_remote_precontainer_intent(
                                state["run_dir"]
                            )
                        )
                before = receipt.read_bytes()
                with patch.object(
                    fleet,
                    "remote_call",
                    side_effect=QwenRuntimeError("invalid_action"),
                ):
                    self.assertFalse(
                        fleet.recover_remote_precontainer_intent(
                            state["run_dir"]
                        )
                    )
                self.assertEqual(receipt.read_bytes(), before)

    def test_uncommitted_recovery_binds_id_then_uses_atomic_worker_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = {
                    **_state("fr-" + "5" * 32, 0),
                    "phase": "starting",
                    "container_id": None,
                    "tunnel_nonce": None,
                    "tunnel_pid": None,
                    "tunnel_create_time": None,
                }
                container_id = "a" * 64
                resources = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )
                receipt = resources["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                status = {
                    "ok": True,
                    "state": "active",
                    "phase": "launching",
                    "container_pid": state["container_pid"],
                    "container_id": container_id,
                    "scratch_cleaned": False,
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
                calls: list[str] = []

                def remote_call(_capability, _source, action, request, **_kwargs):
                    calls.append(action)
                    if action == "status":
                        return status
                    self.assertEqual(action, "recover-uncommitted")
                    self.assertEqual(request["expected_container_id"], container_id)
                    self.assertEqual(
                        request["expected_container_pid"], state["container_pid"]
                    )
                    return _atomic_uncommitted_recovery(state, container_id)

                with patch.object(
                    fleet, "remote_call", side_effect=remote_call
                ), patch.object(
                    fleet, "stop_tunnel", return_value=True
                ), patch.object(
                    fleet, "remote_runtime_liveness", return_value="gone"
                ):
                    self.assertTrue(
                        fleet.recover_remote_uncommitted_intent(state["run_dir"])
                    )

                self.assertEqual(calls, ["status", "recover-uncommitted"])
                self.assertFalse(receipt.exists())

    def test_uncommitted_recovery_binds_exited_pidless_worker_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = {
                    **_state("fr-" + "4" * 32, 1),
                    "phase": "starting",
                    "container_id": None,
                    "tunnel_nonce": None,
                    "tunnel_pid": None,
                    "tunnel_create_time": None,
                }
                container_id = "d" * 64
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                status = {
                    "ok": True,
                    "state": "exited",
                    "phase": "launching",
                    "container_pid": None,
                    "container_id": container_id,
                    "scratch_cleaned": False,
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
                calls: list[str] = []

                def remote_call(_capability, _source, action, request, **_kwargs):
                    calls.append(action)
                    if action == "status":
                        return status
                    self.assertEqual(action, "recover-uncommitted")
                    self.assertEqual(request["expected_container_id"], container_id)
                    self.assertEqual(
                        request["expected_container_pid"], state["container_pid"]
                    )
                    return _atomic_uncommitted_recovery(state, container_id)

                with patch.object(
                    fleet, "remote_call", side_effect=remote_call
                ), patch.object(
                    fleet, "stop_tunnel", return_value=True
                ), patch.object(
                    fleet, "remote_runtime_liveness", return_value="gone"
                ):
                    self.assertTrue(
                        fleet.recover_remote_uncommitted_intent(state["run_dir"])
                    )

                self.assertEqual(calls, ["status", "recover-uncommitted"])
                self.assertFalse(receipt.exists())

    def test_pidless_exited_old_worker_uses_exact_stop_clear_compatibility(self):
        container_id = "e" * 64
        for initially_bound in (False, True):
            with self.subTest(initially_bound=initially_bound), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with patch.object(
                    fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
                ), patch.object(
                    fleet, "REMOTE_STATE_FILE", root / "legacy.json"
                ):
                    state = {
                        **_state("fr-" + ("8" if initially_bound else "7") * 32, 0),
                        "phase": "starting",
                        "source_manifest_sha256": (
                            fleet._PID_CLEARED_STOP_CLEAR_SOURCE_SHA256
                        ),
                        "container_id": container_id if initially_bound else None,
                        "tunnel_nonce": None,
                        "tunnel_pid": None,
                        "tunnel_create_time": None,
                    }
                    receipt = fleet.fleet_remote_runtime_resources(
                        state["run_dir"], state["physical_gpu"]
                    )["orchestrator_state_path"]
                    fleet._private_json_write(receipt, state)
                    identity = {
                        field: state[field]
                        for field in (
                            "run_dir",
                            "physical_gpu",
                            "gpu_uuid",
                            "claim_id",
                            "owner",
                        )
                    }
                    status = {
                        "ok": True,
                        "state": "exited",
                        "phase": "launching",
                        "container_pid": None,
                        "container_id": container_id,
                        "scratch_cleaned": False,
                        **identity,
                    }
                    calls: list[tuple[str, str]] = []

                    def remote_call(_capability, source, action, _request, **_kwargs):
                        calls.append((action, source))
                        if action == "status":
                            return status
                        if action == "stop":
                            return {
                                "ok": True,
                                "state": "stopped",
                                "scratch_cleaned": True,
                                **identity,
                            }
                        if action == "clear":
                            return {
                                "ok": True,
                                "state": "cleared",
                                "receipt_absent": False,
                                **identity,
                            }
                        raise AssertionError(action)

                    with patch.object(
                        fleet, "remote_call", side_effect=remote_call
                    ):
                        self.assertTrue(
                            fleet.recover_remote_uncommitted_intent(
                                state["run_dir"]
                            )
                        )

                    self.assertEqual(
                        [action for action, _source in calls],
                        ["status", "stop", "clear"],
                    )
                    self.assertTrue(
                        all(
                            source
                            == fleet._PID_CLEARED_STOP_CLEAR_SOURCE_SHA256
                            for _action, source in calls
                        )
                    )
                    self.assertFalse(receipt.exists())

    def test_old_worker_compatibility_requires_exact_source_and_capability(self):
        state = {
            **_state("fr-" + "6" * 32, 0),
            "source_manifest_sha256": (
                fleet._PID_CLEARED_STOP_CLEAR_SOURCE_SHA256
            ),
        }
        self.assertTrue(
            fleet._uses_pid_cleared_stop_clear_compatibility(state)
        )
        self.assertFalse(
            fleet._uses_pid_cleared_stop_clear_compatibility(
                {**state, "source_manifest_sha256": "0" * 64}
            )
        )
        self.assertFalse(
            fleet._uses_pid_cleared_stop_clear_compatibility(
                {**state, "runtime_capability_key": "another-capability"}
            )
        )

    def test_bound_old_worker_status_cannot_authorize_changed_outer_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = {
                    **_state("fr-" + "b" * 32, 0),
                    "phase": "starting",
                    "source_manifest_sha256": (
                        fleet._PID_CLEARED_STOP_CLEAR_SOURCE_SHA256
                    ),
                    "tunnel_nonce": None,
                    "tunnel_pid": None,
                    "tunnel_create_time": None,
                }
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                status = {
                    "ok": True,
                    "state": "exited",
                    "phase": "launching",
                    "container_pid": None,
                    "container_id": state["container_id"],
                    "scratch_cleaned": False,
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

                def mutate_outer(*_args, **_kwargs):
                    fleet._private_json_write(
                        receipt,
                        {**state, "phase": "ready", "updated_at": time.time()},
                    )
                    return status

                with patch.object(
                    fleet, "remote_call", side_effect=mutate_outer
                ), patch.object(
                    fleet, "stop_managed_remote_runtime"
                ) as stop:
                    self.assertFalse(
                        fleet.recover_remote_uncommitted_intent(state["run_dir"])
                    )
                stop.assert_not_called()
                self.assertEqual(fleet.remote_state(state["run_dir"])["phase"], "ready")

    def test_old_worker_stop_clear_interruption_retries_atomic_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = {
                    **_state("fr-" + "a" * 32, 0),
                    "phase": "starting",
                    "source_manifest_sha256": (
                        fleet._PID_CLEARED_STOP_CLEAR_SOURCE_SHA256
                    ),
                    "container_id": None,
                    "tunnel_nonce": None,
                    "tunnel_pid": None,
                    "tunnel_create_time": None,
                }
                container_id = "a" * 64
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                identity = {
                    field: state[field]
                    for field in (
                        "run_dir",
                        "physical_gpu",
                        "gpu_uuid",
                        "claim_id",
                        "owner",
                    )
                }
                status = {
                    "ok": True,
                    "state": "exited",
                    "phase": "launching",
                    "container_pid": None,
                    "container_id": container_id,
                    "scratch_cleaned": False,
                    **identity,
                }
                calls: list[str] = []

                def interrupted(_capability, _source, action, _request, **_kwargs):
                    calls.append(action)
                    if action == "status":
                        return status
                    if action == "stop":
                        return {
                            "ok": True,
                            "state": "stopped",
                            "scratch_cleaned": True,
                            **identity,
                        }
                    self.assertEqual(action, "clear")
                    raise QwenRuntimeError("simulated clear transport loss")

                with patch.object(
                    fleet, "remote_call", side_effect=interrupted
                ):
                    self.assertFalse(
                        fleet.recover_remote_uncommitted_intent(state["run_dir"])
                    )
                releasing = fleet.remote_state(state["run_dir"])
                self.assertEqual(releasing["phase"], "releasing")
                self.assertEqual(releasing["container_id"], container_id)

                with patch.object(
                    fleet,
                    "remote_call",
                    return_value=_atomic_uncommitted_recovery(
                        state, container_id, receipt_absent=True
                    ),
                ) as recover, patch.object(
                    fleet, "stop_tunnel", return_value=True
                ), patch.object(
                    fleet, "remote_runtime_liveness", return_value="gone"
                ):
                    self.assertTrue(
                        fleet.recover_remote_uncommitted_intent(state["run_dir"])
                    )

                self.assertEqual(calls, ["status", "stop", "clear"])
                self.assertEqual(
                    recover.call_args.args[2], "recover-uncommitted"
                )
                self.assertFalse(receipt.exists())

    def test_old_worker_stop_failure_retries_exact_dirty_releasing_receipt(self):
        container_id = "c" * 64
        for dirty_liveness in ("exited", "gone"):
            with self.subTest(
                dirty_liveness=dirty_liveness
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with patch.object(
                    fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
                ), patch.object(
                    fleet, "REMOTE_STATE_FILE", root / "legacy.json"
                ):
                    state = {
                        **_state(
                            "fr-"
                            + ("c" if dirty_liveness == "exited" else "d") * 32,
                            0,
                        ),
                        "phase": "starting",
                        "source_manifest_sha256": (
                            fleet._PID_CLEARED_STOP_CLEAR_SOURCE_SHA256
                        ),
                        "container_id": None,
                        "tunnel_nonce": None,
                        "tunnel_pid": None,
                        "tunnel_create_time": None,
                    }
                    receipt = fleet.fleet_remote_runtime_resources(
                        state["run_dir"], state["physical_gpu"]
                    )["orchestrator_state_path"]
                    fleet._private_json_write(receipt, state)
                    identity = {
                        field: state[field]
                        for field in (
                            "run_dir",
                            "physical_gpu",
                            "gpu_uuid",
                            "claim_id",
                            "owner",
                        )
                    }
                    launching_status = {
                        "ok": True,
                        "state": "exited",
                        "phase": "launching",
                        "container_pid": None,
                        "container_id": container_id,
                        "scratch_cleaned": False,
                        **identity,
                    }
                    dirty_status = {
                        **launching_status,
                        "state": dirty_liveness,
                        "phase": "releasing",
                    }
                    calls: list[str] = []
                    stop_attempts = 0

                    def remote_call(
                        _capability, _source, action, _request, **_kwargs
                    ):
                        nonlocal stop_attempts
                        calls.append(action)
                        if action == "status":
                            return (
                                launching_status
                                if stop_attempts == 0
                                else dirty_status
                            )
                        if action == "stop":
                            stop_attempts += 1
                            if stop_attempts == 1:
                                # The old worker has already journaled phase=releasing,
                                # PID=None, scratch_cleaned=False before reporting this.
                                return {
                                    "ok": True,
                                    "state": "ambiguous",
                                    "scratch_cleaned": False,
                                    **identity,
                                }
                            return {
                                "ok": True,
                                "state": "stopped",
                                "scratch_cleaned": True,
                                **identity,
                            }
                        if action == "clear":
                            return {
                                "ok": True,
                                "state": "cleared",
                                "receipt_absent": False,
                                **identity,
                            }
                        raise AssertionError(action)

                    with patch.object(
                        fleet, "remote_call", side_effect=remote_call
                    ):
                        self.assertFalse(
                            fleet.recover_remote_uncommitted_intent(
                                state["run_dir"]
                            )
                        )
                        bound = fleet.remote_state(state["run_dir"])
                        self.assertEqual(bound["phase"], "starting")
                        self.assertEqual(bound["container_id"], container_id)
                        self.assertTrue(
                            fleet.recover_remote_uncommitted_intent(
                                state["run_dir"]
                            )
                        )

                    self.assertEqual(
                        calls, ["status", "stop", "status", "stop", "clear"]
                    )
                    self.assertFalse(receipt.exists())

    def test_old_worker_dirty_releasing_status_refuses_changed_proof(self):
        state = {
            **_state("fr-" + "e" * 32, 0),
            "phase": "starting",
            "source_manifest_sha256": (
                fleet._PID_CLEARED_STOP_CLEAR_SOURCE_SHA256
            ),
            "container_id": "d" * 64,
            "tunnel_nonce": None,
            "tunnel_pid": None,
            "tunnel_create_time": None,
        }
        receipt = fleet.fleet_remote_runtime_resources(
            state["run_dir"], state["physical_gpu"]
        )["orchestrator_state_path"]
        exact = {
            "ok": True,
            "state": "exited",
            "phase": "releasing",
            "container_pid": None,
            "container_id": state["container_id"],
            "scratch_cleaned": False,
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
        refused = (
            (state, {**exact, "state": "active"}),
            (state, {**exact, "state": "ambiguous"}),
            (state, {**exact, "container_id": "e" * 64}),
            (state, {**exact, "container_pid": state["container_pid"]}),
            (state, {**exact, "scratch_cleaned": True}),
            (
                {**state, "source_manifest_sha256": "0" * 64},
                exact,
            ),
        )
        for outer, response in refused:
            with self.subTest(
                outer_source=outer["source_manifest_sha256"], response=response
            ), patch.object(fleet, "remote_call", return_value=response):
                self.assertIsNone(
                    fleet._remote_uncommitted_worker_status(
                        outer, receipt, legacy=False
                    )
                )

    def test_pidless_worker_status_binding_is_narrow_and_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = {
                    **_state("fr-" + "9" * 32, 1),
                    "phase": "starting",
                    "container_id": None,
                    "tunnel_nonce": None,
                    "tunnel_pid": None,
                    "tunnel_create_time": None,
                }
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                exact = {
                    "ok": True,
                    "state": "exited",
                    "phase": "launching",
                    "container_pid": None,
                    "container_id": "f" * 64,
                    "scratch_cleaned": False,
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
                refused = (
                    {**exact, "state": "active"},
                    {**exact, "phase": "ready"},
                    {**exact, "container_pid": state["container_pid"]},
                    {**exact, "scratch_cleaned": True},
                    {**exact, "claim_id": "gc-another-lease"},
                )
                before = receipt.read_bytes()
                for response in refused:
                    with self.subTest(response=response), patch.object(
                        fleet, "remote_call", return_value=response
                    ):
                        self.assertFalse(
                            fleet.recover_remote_uncommitted_intent(
                                state["run_dir"]
                            )
                        )
                    self.assertEqual(receipt.read_bytes(), before)

    def test_uncommitted_recovery_refuses_unbound_or_changed_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = {
                    **_state("fr-" + "6" * 32, 1),
                    "phase": "starting",
                    "container_id": None,
                    "tunnel_nonce": None,
                    "tunnel_pid": None,
                    "tunnel_create_time": None,
                }
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                malformed_status = {
                    "ok": True,
                    "state": "active",
                    "phase": "launching",
                    "container_pid": state["container_pid"] + 1,
                    "container_id": "b" * 64,
                    "scratch_cleaned": False,
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
                before = receipt.read_bytes()
                with patch.object(
                    fleet, "remote_call", return_value=malformed_status
                ) as remote_call:
                    self.assertFalse(
                        fleet.recover_remote_uncommitted_intent(state["run_dir"])
                    )
                remote_call.assert_called_once()
                self.assertEqual(receipt.read_bytes(), before)

                changed = {**state, "tunnel_pid": 9999}
                fleet._private_json_write(receipt, changed)
                with patch.object(fleet, "remote_call") as remote_call:
                    self.assertFalse(
                        fleet.recover_remote_uncommitted_intent(state["run_dir"])
                    )
                remote_call.assert_not_called()

    def test_uncommitted_recovery_retries_releasing_outer_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = {
                    **_state("fr-" + "7" * 32, 0),
                    "phase": "releasing",
                    "tunnel_nonce": None,
                    "tunnel_pid": None,
                    "tunnel_create_time": None,
                }
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                with patch.object(
                    fleet,
                    "remote_call",
                    return_value=_atomic_uncommitted_recovery(
                        state, state["container_id"], receipt_absent=True
                    ),
                ) as recover, patch.object(
                    fleet, "stop_tunnel", return_value=True
                ), patch.object(
                    fleet, "remote_runtime_liveness", return_value="gone"
                ):
                    self.assertTrue(
                        fleet.recover_remote_uncommitted_intent(state["run_dir"])
                    )
                self.assertEqual(
                    recover.call_args.args[2], "recover-uncommitted"
                )
                self.assertFalse(receipt.exists())

                starting = {**state, "phase": "starting", "updated_at": time.time()}
                fleet._private_json_write(receipt, starting)
                with patch.object(
                    fleet,
                    "remote_call",
                    return_value=_atomic_uncommitted_recovery(
                        starting,
                        starting["container_id"],
                        receipt_absent=True,
                    ),
                ), patch.object(
                    fleet, "stop_tunnel", return_value=True
                ), patch.object(
                    fleet, "remote_runtime_liveness", return_value="gone"
                ):
                    self.assertTrue(
                        fleet.recover_remote_uncommitted_intent(
                            starting["run_dir"]
                        )
                    )
                self.assertFalse(receipt.exists())

    def test_uncommitted_recovery_rejects_response_or_outer_receipt_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = {
                    **_state("fr-" + "8" * 32, 1),
                    "phase": "starting",
                    "tunnel_nonce": None,
                    "tunnel_pid": None,
                    "tunnel_create_time": None,
                }
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                invalid = {
                    **_atomic_uncommitted_recovery(state, state["container_id"]),
                    "controller_protocol": 2.0,
                }
                before = receipt.read_bytes()
                with patch.object(fleet, "remote_call", return_value=invalid):
                    self.assertFalse(
                        fleet.recover_remote_uncommitted_intent(state["run_dir"])
                    )
                self.assertEqual(receipt.read_bytes(), before)

                exact = _atomic_uncommitted_recovery(
                    state, state["container_id"]
                )

                def mutate_outer(_capability, _source, action, _request, **_kwargs):
                    self.assertEqual(action, "recover-uncommitted")
                    current = fleet.remote_state(state["run_dir"])
                    fleet._private_json_write(
                        receipt,
                        {**current, "phase": "ready", "updated_at": time.time()},
                    )
                    return exact

                with patch.object(
                    fleet, "remote_call", side_effect=mutate_outer
                ), patch.object(
                    fleet, "stop_tunnel", return_value=True
                ), patch.object(
                    fleet, "remote_runtime_liveness", return_value="gone"
                ):
                    self.assertFalse(
                        fleet.recover_remote_uncommitted_intent(state["run_dir"])
                    )
                self.assertTrue(receipt.exists())
                self.assertEqual(
                    fleet.remote_state(state["run_dir"])["phase"], "ready"
                )

    def test_unlaunched_stop_atomically_recovers_then_clears_orchestrator_receipt(self):
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = _precontainer_state("fr-" + "7" * 32, 1)
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                calls: list[str] = []

                def remote_call(_capability, _source, action, request, **_kwargs):
                    calls.append(action)
                    self.assertEqual(action, "recover-precontainer")
                    return _atomic_worker_recovery(state)

                with patch.object(
                    fleet, "remote_call", side_effect=remote_call
                ):
                    self.assertTrue(
                        fleet.stop_managed_remote_runtime(
                            capability,
                            manifest,
                            state["source_manifest_sha256"],
                            release_reason="test pre-container cleanup",
                            release_claim=False,
                            run_dir=state["run_dir"],
                        )
                    )

                self.assertEqual(calls, ["recover-precontainer"])
                self.assertFalse(receipt.exists())

    def test_unlaunched_stop_refuses_old_unlocked_worker_response(self):
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = _precontainer_state("fr-" + "8" * 32, 0)
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                response = {
                    "ok": True,
                    "state": "gone",
                    "container_pid": None,
                    "container_id": None,
                }
                with patch.object(
                    fleet, "remote_call", return_value=response
                ) as remote_call:
                    self.assertFalse(
                        fleet.stop_managed_remote_runtime(
                            capability,
                            manifest,
                            state["source_manifest_sha256"],
                            release_reason="test refusal",
                            release_claim=False,
                            run_dir=state["run_dir"],
                        )
                    )
                self.assertEqual(
                    [call.args[2] for call in remote_call.call_args_list],
                    ["recover-precontainer", "status"],
                )
                self.assertTrue(receipt.exists())

    def test_unlaunched_stop_uses_exact_old_worker_stop_for_exited_create(self):
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = _precontainer_state("fr-" + "e" * 32, 1)
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                identity = {
                    field: state[field]
                    for field in (
                        "run_dir",
                        "physical_gpu",
                        "gpu_uuid",
                        "claim_id",
                        "owner",
                    )
                }
                calls: list[str] = []

                def remote_call(_capability, _source, action, _request, **_kwargs):
                    calls.append(action)
                    if action == "recover-precontainer":
                        return {"state": "old-worker-refusal"}
                    if action == "stop":
                        return {
                            "state": "stopped",
                            "scratch_cleaned": True,
                            **identity,
                        }
                    if action == "clear":
                        return {
                            "state": "cleared",
                            "receipt_absent": False,
                            **identity,
                        }
                    raise AssertionError(action)

                with patch.object(
                    fleet, "remote_call", side_effect=remote_call
                ), patch.object(
                    fleet, "remote_runtime_liveness", return_value="exited"
                ), patch.object(fleet, "stop_tunnel", return_value=True):
                    self.assertTrue(
                        fleet.stop_managed_remote_runtime(
                            capability,
                            manifest,
                            state["source_manifest_sha256"],
                            release_reason="recover old exited create",
                            release_claim=False,
                            run_dir=state["run_dir"],
                            require_unlaunched=True,
                        )
                    )

                self.assertEqual(
                    calls, ["recover-precontainer", "stop", "clear"]
                )
                self.assertFalse(receipt.exists())

    def test_unlaunched_stop_recovers_after_outer_receipt_reached_releasing(self):
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                fleet, "REMOTE_STATE_ROOT", root / "orchestrator"
            ), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                state = {
                    **_precontainer_state("fr-" + "f" * 32, 1),
                    "phase": "releasing",
                }
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                with patch.object(
                    fleet,
                    "remote_call",
                    return_value=_atomic_worker_recovery(
                        state, receipt_absent=True
                    ),
                ) as recover:
                    self.assertTrue(
                        fleet.stop_managed_remote_runtime(
                            capability,
                            manifest,
                            state["source_manifest_sha256"],
                            release_reason="test idempotent outer cleanup",
                            release_claim=False,
                            run_dir=state["run_dir"],
                            require_unlaunched=True,
                        )
                    )
                self.assertEqual(
                    recover.call_args.args[2], "recover-precontainer"
                )
                self.assertFalse(receipt.exists())

    def test_two_managed_starts_do_not_share_a_lifecycle_receipt(self):
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        source = SourceIdentity(
            package_root=Path("/source"),
            stage_dir=Path("/stage"),
            manifest_sha256="1" * 64,
            manifest_bytes=b"source",
            file_sha256=(),
        )
        first = _state("fr-" + "5" * 32, 0)
        second = _state("fr-" + "6" * 32, 1)

        def lease_for(state: dict) -> dict:
            return {
                key: state[key]
                for key in (
                    "runtime_capability_key",
                    "runtime_capability_manifest_sha256",
                    "runtime_adapter",
                    "host",
                    "physical_gpu",
                    "gpu_uuid",
                    "claim_id",
                    "owner",
                    "run_dir",
                )
            }

        def start_remote(*args, heartbeat_pid, **kwargs):
            gpu = int(kwargs["port"]) - fleet.FLEET_REMOTE_PORT_BASE
            pid = 4400 + gpu
            heartbeat_pid(pid)
            lease = args[3]
            return {
                "container_id": str(gpu + 7) * 64,
                "container_pid": pid,
                **{
                    field: lease[field]
                    for field in (
                        "run_dir",
                        "physical_gpu",
                        "gpu_uuid",
                        "claim_id",
                        "owner",
                    )
                },
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(fleet, "REMOTE_STATE_ROOT", root / "orchestrator"), patch.object(
                fleet, "WORKER_STATE_ROOT", root / "worker"
            ), patch.object(fleet, "REMOTE_STATE_FILE", root / "legacy.json"), patch(
                "aeon.core.qwen_runtime.verify_coordinator_lease",
                side_effect=lambda lease: dict(lease),
            ), patch.object(
                fleet, "start_remote_runtime", side_effect=start_remote
            ), patch.object(
                fleet, "start_tunnel", side_effect=lambda _capability, state, **_kwargs: state
            ):
                results = []
                for state in (first, second):
                    resources = fleet.fleet_remote_runtime_resources(
                        state["run_dir"], state["physical_gpu"]
                    )
                    results.append(
                        fleet.start_managed_remote_runtime(
                            capability,
                            manifest,
                            source,
                            lease_for(state),
                            _solo_deploy_environment(),
                            container_name=resources["container_name"],
                            port=resources["remote_port"],
                            heartbeat_pid=lambda _pid: None,
                        )
                    )

                self.assertEqual([item["phase"] for item in results], ["ready", "ready"])
                self.assertIsNotNone(fleet.remote_state(first["run_dir"]))
                self.assertIsNotNone(fleet.remote_state(second["run_dir"]))

    def test_legacy_managed_start_preserves_the_original_plan(self):
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        source = SourceIdentity(
            package_root=Path("/source"),
            stage_dir=Path("/stage"),
            manifest_sha256="1" * 64,
            manifest_bytes=b"source",
            file_sha256=(),
        )
        run_dir = "/home/aday/.aeon/runtime/qwen38/aeon-qwen38-vllm-legacy-test"
        lease = {
            "runtime_capability_key": capability.key,
            "runtime_capability_manifest_sha256": manifest,
            "runtime_adapter": capability.runtime_adapter,
            "host": capability.host,
            "run_dir": run_dir,
            "physical_gpu": 0,
            "gpu_uuid": "GPU-12345678-0000-0000-0000-000000000000",
            "claim_id": "gc-test-legacy-runtime",
            "owner": "legacy-test",
        }
        environment = _solo_deploy_environment()
        original = dict(environment)

        def start_remote(*args, **_kwargs):
            observed_lease = args[3]
            return {
                "container_id": "9" * 64,
                "container_pid": 4400,
                **{
                    field: observed_lease[field]
                    for field in (
                        "run_dir",
                        "physical_gpu",
                        "gpu_uuid",
                        "claim_id",
                        "owner",
                    )
                },
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(fleet, "REMOTE_STATE_FILE", root / "legacy.json"), patch(
                "aeon.core.qwen_runtime.verify_coordinator_lease",
                side_effect=lambda value: dict(value),
            ), patch.object(
                fleet, "start_remote_runtime", side_effect=start_remote
            ) as start, patch.object(
                fleet,
                "start_tunnel",
                side_effect=lambda _capability, state, **_kwargs: state,
            ):
                result = fleet.start_managed_remote_runtime(
                    capability,
                    manifest,
                    source,
                    lease,
                    environment,
                    container_name="aeon_qwen_base",
                    port=8033,
                    heartbeat_pid=lambda _pid: None,
                )

        self.assertEqual(result["phase"], "ready")
        self.assertEqual(environment, original)
        self.assertEqual(start.call_args.args[4], original)
        self.assertEqual(
            start.call_args.args[4]["AEON_DEPLOY_PLAN"],
            original["AEON_DEPLOY_PLAN"],
        )

    def test_remote_start_timeout_terminates_and_reaps_exact_ssh_child(self):
        state = _precontainer_state("fr-" + "d" * 32, 0)
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        source = SourceIdentity(
            package_root=Path("/source"),
            stage_dir=Path("/stage"),
            manifest_sha256=state["source_manifest_sha256"],
            manifest_bytes=b"source",
            file_sha256=(),
        )
        resources = fleet.fleet_remote_runtime_resources(
            state["run_dir"], state["physical_gpu"]
        )

        class StartProcess:
            def __init__(self):
                self.args = ["ssh", "exact-start"]
                self.stdin = io.StringIO()
                self.returncode = None
                self.terminated = False
                self.killed = False
                self.communicated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def kill(self):
                self.killed = True
                self.returncode = -9

            def communicate(self, timeout=None):
                self.communicated = True
                return "", ""

        process = StartProcess()
        lease = {
            field: state[field]
            for field in (
                "run_dir",
                "physical_gpu",
                "gpu_uuid",
                "claim_id",
                "owner",
            )
        }
        with patch.object(
            fleet.subprocess, "Popen", return_value=process
        ), patch.object(
            fleet,
            "remote_call",
            side_effect=QwenRuntimeError("worker is still entering controller"),
        ), patch.object(fleet.time, "monotonic", return_value=100.0):
            with self.assertRaisesRegex(QwenRuntimeError, "bounded timeout"):
                fleet.start_remote_runtime(
                    capability,
                    manifest,
                    source,
                    lease,
                    {},
                    receipt_path=resources["orchestrator_state_path"],
                    container_name=resources["container_name"],
                    port=resources["remote_port"],
                    heartbeat_pid=lambda _pid: None,
                    timeout=0,
                )

        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        self.assertTrue(process.communicated)

    def test_remote_start_controller_cleanup_escalates_only_its_exact_child(self):
        class StubbornProcess:
            def __init__(self):
                self.stdin = io.StringIO()
                self.returncode = None
                self.terminated = 0
                self.killed = 0
                self.communicated = 0

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated += 1

            def kill(self):
                self.killed += 1
                self.returncode = -9

            def communicate(self, timeout=None):
                self.communicated += 1
                if self.communicated == 1:
                    raise fleet.subprocess.TimeoutExpired("exact-ssh", timeout)
                return "", ""

        process = StubbornProcess()
        fleet._terminate_remote_start_controller(process)
        self.assertEqual(process.terminated, 1)
        self.assertEqual(process.killed, 1)
        self.assertEqual(process.communicated, 2)
        self.assertEqual(process.poll(), -9)

    def test_stale_per_runtime_receipt_blocks_the_same_gpu_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(fleet, "REMOTE_STATE_ROOT", root / "orchestrator"), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                stale = _state("fr-" + "8" * 32, 0)
                stale_path = fleet.fleet_remote_runtime_resources(
                    stale["run_dir"], 0
                )["orchestrator_state_path"]
                fleet._private_json_write(stale_path, stale)
                with self.assertRaises(QwenRuntimeError):
                    fleet._assert_remote_gpu_slot_available(
                        host=stale["host"],
                        physical_gpu=0,
                        run_dir=(
                            "/home/aday/.local/state/fleet-compute/runs/fr-"
                            + "9" * 32
                        ),
                    )
                fleet._assert_remote_gpu_slot_available(
                    host=stale["host"],
                    physical_gpu=1,
                    run_dir=(
                        "/home/aday/.local/state/fleet-compute/runs/fr-"
                        + "a" * 32
                    ),
                )

    def test_matching_legacy_singleton_is_recoverable_but_never_ambiguous(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(fleet, "REMOTE_STATE_ROOT", root / "orchestrator"), patch.object(
                fleet, "WORKER_STATE_ROOT", root / "worker"
            ), patch.object(fleet, "REMOTE_STATE_FILE", root / "legacy.json"):
                state = _state("fr-" + "e" * 32, 0)
                legacy = {
                    **state,
                    "container_name": "aeon-qwen38-vllm",
                    "remote_port": 8033,
                    "local_port": 8033,
                }
                fleet._private_json_write(fleet.REMOTE_STATE_FILE, legacy)
                self.assertEqual(fleet.remote_state(state["run_dir"]), legacy)
                binding = fleet._worker_request_binding(
                    legacy, fleet.REMOTE_STATE_FILE
                )
                self.assertEqual(worker._runtime_binding(binding)[1], RUNTIME_STATE_FILE)

                resources = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], state["physical_gpu"]
                )
                fleet._private_json_write(
                    resources["orchestrator_state_path"], state
                )
                with self.assertRaises(QwenRuntimeError):
                    fleet.remote_state(state["run_dir"])

    def test_malformed_legacy_receipt_blocks_unrelated_runtime_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(fleet, "REMOTE_STATE_ROOT", root / "orchestrator"), patch.object(
                fleet, "REMOTE_STATE_FILE", root / "legacy.json"
            ):
                fleet._private_json_write(fleet.REMOTE_STATE_FILE, {"bad": "receipt"})
                with self.assertRaises(QwenRuntimeError):
                    fleet.remote_state(
                        "/home/aday/.local/state/fleet-compute/runs/fr-"
                        + "f" * 32
                    )

    def test_stop_clears_only_the_exact_runtime_receipt(self):
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        calls: list[tuple[str, dict]] = []

        def remote_call(_capability, _source, action, request, **_kwargs):
            calls.append((action, request))
            identity = {
                field: request[field]
                for field in (
                    "run_dir",
                    "physical_gpu",
                    "gpu_uuid",
                    "claim_id",
                    "owner",
                )
            }
            if action == "stop":
                return {
                    "state": "stopped",
                    "scratch_cleaned": True,
                    **identity,
                }
            if action == "clear":
                return {
                    "state": "cleared",
                    "receipt_absent": False,
                    **identity,
                }
            raise AssertionError(action)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(fleet, "REMOTE_STATE_ROOT", root / "orchestrator"), patch.object(
                fleet, "WORKER_STATE_ROOT", root / "worker"
            ), patch.object(fleet, "REMOTE_STATE_FILE", root / "legacy.json"), patch.object(
                fleet, "remote_call", side_effect=remote_call
            ):
                first = _state("fr-" + "1" * 32, 0)
                second = _state("fr-" + "2" * 32, 1)
                first_path = fleet.fleet_remote_runtime_resources(
                    first["run_dir"], 0
                )["orchestrator_state_path"]
                second_path = fleet.fleet_remote_runtime_resources(
                    second["run_dir"], 1
                )["orchestrator_state_path"]
                fleet._private_json_write(first_path, first)
                fleet._private_json_write(second_path, second)

                self.assertTrue(
                    fleet.stop_managed_remote_runtime(
                        capability,
                        manifest,
                        first["source_manifest_sha256"],
                        release_reason="test exact stop",
                        release_claim=False,
                        run_dir=first["run_dir"],
                    )
                )
                self.assertFalse(first_path.exists())
                self.assertEqual(fleet.remote_state(second["run_dir"]), second)
                self.assertTrue(second_path.exists())
                self.assertEqual([action for action, _request in calls], ["stop", "clear"])
                self.assertTrue(
                    all(
                        request["runtime_id"] == Path(first["run_dir"]).name
                        and request["worker_state_path"].endswith(
                            f"{Path(first['run_dir']).name}.json"
                        )
                        for _action, request in calls
                    )
                )

    def test_stop_refuses_a_worker_response_from_another_lease(self):
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(fleet, "REMOTE_STATE_ROOT", root / "orchestrator"), patch.object(
                fleet, "WORKER_STATE_ROOT", root / "worker"
            ), patch.object(fleet, "REMOTE_STATE_FILE", root / "legacy.json"):
                state = _state("fr-" + "b" * 32, 0)
                receipt = fleet.fleet_remote_runtime_resources(
                    state["run_dir"], 0
                )["orchestrator_state_path"]
                fleet._private_json_write(receipt, state)
                response = {
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
                    "claim_id": "gc-another-lease",
                }
                with patch.object(fleet, "remote_call", return_value=response):
                    self.assertFalse(
                        fleet.stop_managed_remote_runtime(
                            capability,
                            manifest,
                            state["source_manifest_sha256"],
                            release_reason="test mismatch",
                            release_claim=False,
                            run_dir=state["run_dir"],
                        )
                    )
                self.assertTrue(receipt.exists())

    def test_worker_stop_uses_only_the_selected_runtime_journal(self):
        state = _state("fr-" + "7" * 32, 0)
        resources = fleet.fleet_remote_runtime_resources(state["run_dir"], 0)
        request = fleet._worker_request_binding(
            state, resources["orchestrator_state_path"]
        )
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        with patch.object(
            worker, "_capability", return_value=(capability, manifest)
        ), patch.object(worker, "_state_for_request", return_value=state), patch.object(
            worker, "stop_qwen_runtime", return_value=True
        ) as stop, patch.object(
            worker, "current_runtime_state", return_value=state
        ) as current:
            self.assertEqual(worker._stop(request)["state"], "stopped")

        self.assertEqual(
            stop.call_args.kwargs["state_path"], resources["worker_state_path"]
        )
        current.assert_called_once_with(resources["worker_state_path"])

    def test_worker_recovery_refuses_while_exact_start_controller_is_held(self):
        state = _precontainer_state("fr-" + "9" * 32, 0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_root = root / "worker"
            with patch.object(
                fleet, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_CONTROLLER_ROOT", root / "controllers"
            ):
                request, capability, manifest = _worker_request(state)
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), worker._hold_controller_lock(
                    request, action="start", create=True
                ):
                    with patch.object(worker, "_state_for_request") as saved:
                        with self.assertRaisesRegex(
                            QwenRuntimeError, "controller is already active"
                        ):
                            worker._dispatch("recover-precontainer", request)
                    saved.assert_not_called()
                    receipt = worker._controller_lock_path(request)
                    self.assertEqual(
                        json.loads(receipt.read_text(encoding="utf-8"))["status"],
                        "active",
                    )

                self.assertEqual(
                    json.loads(receipt.read_text(encoding="utf-8"))["status"],
                    "quiescent",
                )

    def test_worker_uncommitted_recovery_refuses_an_active_controller(self):
        state = _state("fr-" + "3" * 32, 0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_root = root / "worker"
            with patch.object(
                fleet, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_CONTROLLER_ROOT", root / "controllers"
            ):
                request, capability, manifest = _worker_request(state)
                request.update(
                    {
                        "expected_container_id": state["container_id"],
                        "expected_container_pid": state["container_pid"],
                    }
                )
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), worker._hold_controller_lock(
                    request, action="start", create=True
                ):
                    with patch.object(worker, "_state_for_request") as saved:
                        with self.assertRaisesRegex(
                            QwenRuntimeError, "controller is already active"
                        ):
                            worker._dispatch("recover-uncommitted", request)
                    saved.assert_not_called()

    def test_worker_recovery_refuses_empty_malformed_or_mismatched_controller(self):
        state = _precontainer_state("fr-" + "a" * 32, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_root = root / "worker"
            with patch.object(
                fleet, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_CONTROLLER_ROOT", root / "controllers"
            ):
                request, capability, manifest = _worker_request(state)
                controller_root = worker.WORKER_CONTROLLER_ROOT
                controller_root.mkdir(mode=0o700)
                receipt = worker._controller_lock_path(request)
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(worker, "_state_for_request") as saved:
                    with self.assertRaisesRegex(
                        QwenRuntimeError, "controller receipt is absent"
                    ):
                        worker._dispatch("recover-precontainer", request)
                saved.assert_not_called()

                receipt.touch(mode=0o600)
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(worker, "_state_for_request") as saved:
                    with self.assertRaisesRegex(
                        QwenRuntimeError, "receipt is incomplete"
                    ):
                        worker._dispatch("recover-precontainer", request)
                saved.assert_not_called()

                receipt.write_text("{}", encoding="utf-8")
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(worker, "_state_for_request") as saved:
                    with self.assertRaisesRegex(
                        QwenRuntimeError, "receipt is malformed"
                    ):
                        worker._dispatch("recover-precontainer", request)
                saved.assert_not_called()

                receipt.unlink()
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), worker._hold_controller_lock(
                    request, action="start", create=True
                ):
                    pass
                changed = {**request, "claim_id": "gc-another-lease"}
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(worker, "_state_for_request") as saved:
                    with self.assertRaisesRegex(
                        QwenRuntimeError, "receipt identity changed"
                    ):
                        worker._dispatch("recover-precontainer", changed)
                saved.assert_not_called()

    def test_worker_missing_receipt_recovery_requires_zero_exact_claim_candidates(self):
        state = _precontainer_state("fr-" + "b" * 32, 0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_root = root / "worker"
            with patch.object(
                fleet, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_CONTROLLER_ROOT", root / "controllers"
            ):
                request, capability, manifest = _worker_request(state)
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), worker._hold_controller_lock(
                    request, action="start", create=True
                ):
                    pass

                refused = (
                    ("a" * 64,),
                    ("a" * 64, "b" * 64),
                    QwenRuntimeError("test Docker ambiguity"),
                )
                for candidates in refused:
                    with self.subTest(candidates=candidates), patch.object(
                        worker, "_capability", return_value=(capability, manifest)
                    ), patch.object(
                        worker, "_state_for_request", return_value=None
                    ), patch.object(
                        worker,
                        "_claim_container_candidates",
                        side_effect=(
                            candidates
                            if isinstance(candidates, BaseException)
                            else None
                        ),
                        return_value=(
                            ()
                            if isinstance(candidates, BaseException)
                            else candidates
                        ),
                    ):
                        with self.assertRaises(QwenRuntimeError):
                            worker._dispatch("recover-precontainer", request)
                    controller = json.loads(
                        worker._controller_lock_path(request).read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(controller["status"], "quiescent")

                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(
                    worker, "_state_for_request", return_value=None
                ), patch.object(
                    worker, "_claim_container_candidates", return_value=()
                ):
                    result = worker._dispatch("recover-precontainer", request)
                self.assertTrue(result["process_absent"])
                self.assertTrue(result["worker_receipt_absent"])
                self.assertEqual(result["claim_id"], state["claim_id"])

    def test_worker_exact_claim_candidate_query_is_narrow_and_fail_closed(self):
        state = _precontainer_state("fr-" + "e" * 32, 0)
        request, capability, _manifest = _worker_request(state)
        exact_id = "a" * 64
        with patch.object(
            worker.subprocess,
            "run",
            return_value=worker.subprocess.CompletedProcess(
                [], 0, exact_id + "\n", ""
            ),
        ) as run:
            self.assertEqual(
                worker._claim_container_candidates(request, capability),
                (exact_id,),
            )
        command = run.call_args.args[0]
        self.assertIn("label=com.bc_aeon.component=qwen38-vllm", command)
        self.assertIn(f"label=com.bc_aeon.claim={state['claim_id']}", command)
        self.assertIn(
            f"label=com.bc_aeon.runtime-capability={capability.key}", command
        )
        self.assertNotIn("nvidia-smi", command)

        refused = (
            worker.subprocess.CompletedProcess([], 1, "", ""),
            worker.subprocess.CompletedProcess([], 0, "", "warning"),
            worker.subprocess.CompletedProcess([], 0, "short\n", ""),
            worker.subprocess.CompletedProcess(
                [], 0, exact_id + "\n" + exact_id + "\n", ""
            ),
            worker.subprocess.CompletedProcess([], 0, "x" * 8193, ""),
        )
        for completed in refused:
            with self.subTest(completed=completed), patch.object(
                worker.subprocess, "run", return_value=completed
            ):
                with self.assertRaises(QwenRuntimeError):
                    worker._claim_container_candidates(request, capability)

    def test_worker_bound_recovery_is_exact_and_idempotent(self):
        state = _precontainer_state("fr-" + "c" * 32, 1)
        before = {
            **state,
            "phase": "preparing",
            "scratch_cleaned": False,
        }
        releasing = {
            **before,
            "phase": "releasing",
            "scratch_cleaned": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_root = root / "worker"
            with patch.object(
                fleet, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_CONTROLLER_ROOT", root / "controllers"
            ):
                request, capability, manifest = _worker_request(state)
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), worker._hold_controller_lock(
                    request, action="start", create=True
                ):
                    pass

                for refused, liveness in (
                    ({**before, "phase": "ready"}, "gone"),
                    (before, "active"),
                ):
                    with self.subTest(
                        phase=refused["phase"], liveness=liveness
                    ), patch.object(
                        worker, "_capability", return_value=(capability, manifest)
                    ), patch.object(
                        worker, "_state_for_request", return_value=refused
                    ), patch.object(
                        worker, "qwen_runtime_liveness", return_value=liveness
                    ), patch.object(worker, "stop_qwen_runtime") as stop:
                        with self.assertRaises(QwenRuntimeError):
                            worker._dispatch("recover-precontainer", request)
                    stop.assert_not_called()

                dirty_releasing = {
                    **before,
                    "phase": "releasing",
                    "scratch_cleaned": False,
                }
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(
                    worker, "_state_for_request", return_value=dirty_releasing
                ), patch.object(
                    worker, "qwen_runtime_liveness", return_value="gone"
                ), patch.object(
                    worker, "stop_qwen_runtime", return_value=True
                ) as stop, patch.object(
                    worker,
                    "current_runtime_state",
                    side_effect=[releasing, None],
                ), patch.object(worker, "clear_runtime_state") as clear:
                    recovered_dirty = worker._dispatch(
                        "recover-precontainer", request
                    )
                self.assertFalse(recovered_dirty["worker_receipt_absent"])
                stop.assert_called_once()
                clear.assert_called_once()

                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(
                    worker, "_state_for_request", return_value=releasing
                ), patch.object(
                    worker, "qwen_runtime_liveness", return_value="gone"
                ), patch.object(worker, "stop_qwen_runtime") as stop, patch.object(
                    worker, "current_runtime_state", return_value=None
                ), patch.object(worker, "clear_runtime_state") as clear:
                    recovered_clean = worker._dispatch(
                        "recover-precontainer", request
                    )
                self.assertFalse(recovered_clean["worker_receipt_absent"])
                stop.assert_not_called()
                clear.assert_called_once()

                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(
                    worker, "_state_for_request", return_value=before
                ), patch.object(
                    worker, "qwen_runtime_liveness", return_value="gone"
                ), patch.object(
                    worker, "stop_qwen_runtime", return_value=True
                ) as stop, patch.object(
                    worker,
                    "current_runtime_state",
                    side_effect=[releasing, None],
                ), patch.object(worker, "clear_runtime_state") as clear:
                    result = worker._dispatch("recover-precontainer", request)
                self.assertFalse(result["worker_receipt_absent"])
                stop.assert_called_once()
                clear.assert_called_once()

                exited_id = "d" * 64
                exited = {**before, "container_id": exited_id}
                exited_releasing = {
                    **releasing,
                    "container_id": exited_id,
                }
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(
                    worker, "_state_for_request", return_value=exited
                ), patch.object(
                    worker,
                    "qwen_runtime_liveness",
                    side_effect=["exited", "gone"],
                ), patch.object(
                    worker, "stop_qwen_runtime", return_value=True
                ) as stop, patch.object(
                    worker,
                    "current_runtime_state",
                    side_effect=[exited_releasing, None],
                ), patch.object(worker, "clear_runtime_state") as clear:
                    recovered_exited = worker._dispatch(
                        "recover-precontainer", request
                    )
                self.assertFalse(recovered_exited["worker_receipt_absent"])
                stop.assert_called_once()
                clear.assert_called_once()

                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(
                    worker, "_state_for_request", return_value=None
                ), patch.object(
                    worker, "_claim_container_candidates", return_value=()
                ):
                    retried = worker._dispatch("recover-precontainer", request)
                self.assertTrue(retried["worker_receipt_absent"])
                controller = json.loads(
                    worker._controller_lock_path(request).read_text(encoding="utf-8")
                )
                self.assertEqual(controller["status"], "recovered")

    def test_worker_uncommitted_recovery_is_atomic_exact_and_idempotent(self):
        state = _state("fr-" + "1" * 32, 0)
        container_id = state["container_id"]
        before = {
            **state,
            "phase": "launching",
            "scratch_cleaned": False,
        }
        releasing = {
            **before,
            "phase": "releasing",
            "container_pid": None,
            "scratch_cleaned": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_root = root / "worker"
            with patch.object(
                fleet, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_CONTROLLER_ROOT", root / "controllers"
            ):
                request, capability, manifest = _worker_request(state)
                request.update(
                    {
                        "expected_container_id": container_id,
                        "expected_container_pid": state["container_pid"],
                    }
                )
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), worker._hold_controller_lock(
                    request, action="start", create=True
                ):
                    pass

                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(
                    worker,
                    "_state_for_request",
                    side_effect=[releasing, releasing],
                ), patch.object(
                    worker,
                    "qwen_runtime_liveness",
                    side_effect=["gone", "gone"],
                ), patch.object(
                    worker,
                    "_claim_container_candidates",
                    side_effect=[(), (), ()],
                ), patch.object(worker, "stop_qwen_runtime") as stop, patch.object(
                    worker, "clear_runtime_state"
                ) as clear, patch.object(
                    worker, "current_runtime_state", return_value=None
                ):
                    clean_retry = worker._dispatch(
                        "recover-uncommitted", request
                    )
                self.assertTrue(clean_retry["process_absent"])
                stop.assert_not_called()
                clear.assert_called_once()

                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(
                    worker,
                    "_state_for_request",
                    side_effect=[before, releasing],
                ), patch.object(
                    worker,
                    "qwen_runtime_liveness",
                    side_effect=["active", "gone"],
                ), patch.object(
                    worker,
                    "_claim_container_candidates",
                    side_effect=[(container_id,), (), ()],
                ), patch.object(
                    worker, "stop_qwen_runtime", return_value=True
                ) as stop, patch.object(
                    worker, "clear_runtime_state"
                ) as clear, patch.object(
                    worker, "current_runtime_state", return_value=None
                ):
                    result = worker._dispatch("recover-uncommitted", request)

                self.assertTrue(result["process_absent"])
                self.assertFalse(result["worker_receipt_absent"])
                self.assertEqual(result["container_id"], container_id)
                stop.assert_called_once_with(
                    state_path=fleet.fleet_remote_runtime_resources(
                        state["run_dir"], state["physical_gpu"]
                    )["worker_state_path"],
                    allow_lost_lease=True,
                )
                clear.assert_called_once()

                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(
                    worker, "_state_for_request", return_value=None
                ), patch.object(
                    worker, "_claim_container_candidates", return_value=()
                ):
                    retried = worker._dispatch("recover-uncommitted", request)
                self.assertTrue(retried["worker_receipt_absent"])
                controller = json.loads(
                    worker._controller_lock_path(request).read_text(encoding="utf-8")
                )
                self.assertEqual(controller["status"], "recovered")

    def test_worker_status_returns_nonadopting_coherent_snapshot(self):
        state = {
            **_state("fr-" + "4" * 32, 0),
            "phase": "launching",
            "container_id": None,
        }
        resolved = {
            **state,
            "container_id": "4" * 64,
            "container_pid": None,
        }
        request, capability, manifest = _worker_request(state)
        resources = fleet.fleet_remote_runtime_resources(
            state["run_dir"], state["physical_gpu"]
        )
        before = dict(state)
        with patch.object(
            worker, "_capability", return_value=(capability, manifest)
        ), patch.object(
            worker, "_state_for_request", return_value=state
        ), patch.object(
            worker,
            "_resolve_container",
            return_value=("exited", None, resolved),
        ) as resolve:
            result = worker._status(request)

        self.assertEqual(state, before)
        self.assertEqual(result["state"], "exited")
        self.assertEqual(result["phase"], "launching")
        self.assertIsNone(result["container_pid"])
        self.assertEqual(result["container_id"], resolved["container_id"])
        resolve.assert_called_once_with(
            state,
            adopt=False,
            state_path=resources["worker_state_path"],
        )

    def test_worker_recovers_only_exact_exited_pidless_uncommitted_startup(self):
        state = _state("fr-" + "3" * 32, 0)
        container_id = state["container_id"]
        before = {
            **state,
            "phase": "launching",
            "container_pid": None,
            "scratch_cleaned": False,
        }
        releasing = {
            **before,
            "phase": "releasing",
            "scratch_cleaned": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_root = root / "worker"
            with patch.object(
                fleet, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_CONTROLLER_ROOT", root / "controllers"
            ):
                request, capability, manifest = _worker_request(state)
                request.update(
                    {
                        "expected_container_id": container_id,
                        "expected_container_pid": state["container_pid"],
                    }
                )
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), worker._hold_controller_lock(
                    request, action="start", create=True
                ):
                    pass

                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(
                    worker,
                    "_state_for_request",
                    side_effect=[before, releasing],
                ), patch.object(
                    worker,
                    "qwen_runtime_liveness",
                    side_effect=["exited", "gone"],
                ), patch.object(
                    worker,
                    "_claim_container_candidates",
                    side_effect=[(container_id,), (), ()],
                ), patch.object(
                    worker, "stop_qwen_runtime", return_value=True
                ) as stop, patch.object(
                    worker, "clear_runtime_state"
                ) as clear, patch.object(
                    worker, "current_runtime_state", return_value=None
                ):
                    result = worker._dispatch("recover-uncommitted", request)

                self.assertTrue(result["process_absent"])
                self.assertEqual(result["container_id"], container_id)
                self.assertEqual(
                    result["container_pid"], state["container_pid"]
                )
                stop.assert_called_once()
                clear.assert_called_once()

    def test_worker_retries_exact_dirty_pidless_release(self):
        state = _state("fr-" + "5" * 32, 1)
        container_id = state["container_id"]
        before = {
            **state,
            "phase": "releasing",
            "container_pid": None,
            "scratch_cleaned": False,
        }
        releasing = {
            **before,
            "scratch_cleaned": True,
        }
        for liveness, first_candidates in (
            ("exited", (container_id,)),
            ("gone", ()),
        ):
            with self.subTest(liveness=liveness), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                worker_root = root / "worker"
                with patch.object(
                    fleet, "WORKER_STATE_ROOT", worker_root
                ), patch.object(
                    worker, "WORKER_STATE_ROOT", worker_root
                ), patch.object(
                    worker, "WORKER_CONTROLLER_ROOT", root / "controllers"
                ):
                    request, capability, manifest = _worker_request(state)
                    request.update(
                        {
                            "expected_container_id": container_id,
                            "expected_container_pid": state["container_pid"],
                        }
                    )
                    with patch.object(
                        worker, "_capability", return_value=(capability, manifest)
                    ), worker._hold_controller_lock(
                        request, action="start", create=True
                    ):
                        pass

                    with patch.object(
                        worker,
                        "_capability",
                        return_value=(capability, manifest),
                    ), patch.object(
                        worker,
                        "_state_for_request",
                        side_effect=[before, releasing],
                    ), patch.object(
                        worker,
                        "qwen_runtime_liveness",
                        side_effect=[liveness, "gone"],
                    ), patch.object(
                        worker,
                        "_claim_container_candidates",
                        side_effect=[first_candidates, (), ()],
                    ), patch.object(
                        worker, "stop_qwen_runtime", return_value=True
                    ) as stop, patch.object(
                        worker, "clear_runtime_state"
                    ) as clear, patch.object(
                        worker, "current_runtime_state", return_value=None
                    ):
                        result = worker._dispatch(
                            "recover-uncommitted", request
                        )

                    self.assertTrue(result["process_absent"])
                    self.assertEqual(
                        result["container_pid"], state["container_pid"]
                    )
                    stop.assert_called_once()
                    clear.assert_called_once()

    def test_worker_uncommitted_recovery_refuses_identity_divergence(self):
        state = _state("fr-" + "2" * 32, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_root = root / "worker"
            with patch.object(
                fleet, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_STATE_ROOT", worker_root
            ), patch.object(
                worker, "WORKER_CONTROLLER_ROOT", root / "controllers"
            ):
                request, capability, manifest = _worker_request(state)
                request.update(
                    {
                        "expected_container_id": state["container_id"],
                        "expected_container_pid": state["container_pid"],
                    }
                )
                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), worker._hold_controller_lock(
                    request, action="start", create=True
                ):
                    pass

                refused = (
                    {**state, "phase": "launching", "scratch_cleaned": False,
                     "container_id": "f" * 64},
                    {**state, "phase": "launching", "scratch_cleaned": False,
                     "container_pid": state["container_pid"] + 1},
                    {**state, "phase": "ready", "scratch_cleaned": False,
                     "container_pid": None},
                    {**state, "phase": "launching", "scratch_cleaned": True,
                     "container_pid": None},
                )
                for changed in refused:
                    with self.subTest(changed=changed), patch.object(
                        worker, "_capability", return_value=(capability, manifest)
                    ), patch.object(
                        worker, "_state_for_request", return_value=changed
                    ), patch.object(
                        worker, "stop_qwen_runtime"
                    ) as stop, patch.object(
                        worker, "_claim_container_candidates"
                    ) as candidates:
                        with self.assertRaises(QwenRuntimeError):
                            worker._dispatch("recover-uncommitted", request)
                    stop.assert_not_called()
                    candidates.assert_not_called()

                pidless = {
                    **state,
                    "phase": "launching",
                    "scratch_cleaned": False,
                    "container_pid": None,
                }
                for liveness, candidates_result in (
                    ("active", (state["container_id"],)),
                    ("gone", ()),
                    ("exited", ()),
                    ("exited", ("e" * 64,)),
                    ("exited", (state["container_id"], "e" * 64)),
                ):
                    with self.subTest(
                        liveness=liveness, candidates=candidates_result
                    ), patch.object(
                        worker, "_capability", return_value=(capability, manifest)
                    ), patch.object(
                        worker, "_state_for_request", return_value=pidless
                    ), patch.object(
                        worker, "qwen_runtime_liveness", return_value=liveness
                    ), patch.object(
                        worker,
                        "_claim_container_candidates",
                        return_value=candidates_result,
                    ), patch.object(worker, "stop_qwen_runtime") as stop:
                        with self.assertRaises(QwenRuntimeError):
                            worker._dispatch("recover-uncommitted", request)
                    stop.assert_not_called()

                with patch.object(
                    worker, "_capability", return_value=(capability, manifest)
                ), patch.object(
                    worker, "_state_for_request", return_value=None
                ), patch.object(
                    worker,
                    "_claim_container_candidates",
                    return_value=(state["container_id"],),
                ), patch.object(worker, "stop_qwen_runtime") as stop:
                    with self.assertRaises(QwenRuntimeError):
                        worker._dispatch("recover-uncommitted", request)
                stop.assert_not_called()

    def test_worker_start_refuses_an_orphaned_same_gpu_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_root = root / "worker"
            stale_path = worker_root / ("fr-" + "c" * 32 + ".json")
            fleet._private_json_write(stale_path, {"placeholder": True})
            selected = worker_root / ("fr-" + "d" * 32 + ".json")

            def current(path):
                if path == stale_path:
                    return {"physical_gpu": 0}
                return None

            with patch.object(worker, "WORKER_STATE_ROOT", worker_root), patch.object(
                worker, "RUNTIME_STATE_FILE", root / "legacy.json"
            ), patch.object(worker, "current_runtime_state", side_effect=current):
                with self.assertRaises(QwenRuntimeError):
                    worker._assert_worker_gpu_slot_available(
                        {"physical_gpu": 0}, selected
                    )

    def test_preflight_receipts_are_source_content_addressed(self):
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        first = fleet._request_base(capability, manifest, "d" * 64)
        second = fleet._request_base(capability, manifest, "e" * 64)
        first_path = worker._release_paths(first, capability)[2]
        second_path = worker._release_paths(second, capability)[2]
        self.assertNotEqual(first_path, second_path)
        self.assertIn("d" * 64, first_path.name)
        self.assertIn("e" * 64, second_path.name)

    def test_worker_teardown_accepts_a_disabled_unchanged_capability(self):
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        disabled = replace(
            capability, enabled=False, disabled_reason="test rollout"
        )
        request = fleet._request_base(capability, manifest, "f" * 64)
        request["capability_manifest_sha256"] = (
            "52e2d54b70c14eefac3d5cae796b1f1ce40ececb95961a42d1c8ec6457254b6a"
        )
        with patch.object(
            worker, "qwen_runtime_capability", return_value=(disabled, "1" * 64)
        ), patch.object(worker.socket, "gethostname", return_value=capability.hostname):
            observed, _current = worker._capability(
                request, require_enabled=False
            )
            self.assertEqual(observed.key, capability.key)
            with self.assertRaises(QwenRuntimeError):
                worker._capability(request, require_enabled=True)

            request["capability_manifest_sha256"] = "0" * 64
            with self.assertRaises(QwenRuntimeError):
                worker._capability(request, require_enabled=False)


class FleetAdapterRuntimeScopeTests(unittest.TestCase):
    def test_remote_profile_must_match_the_staged_runtime_source(self):
        digest = "a" * 64
        source = SourceIdentity(
            package_root=Path("/source"),
            stage_dir=Path("/stage"),
            manifest_sha256=digest,
            manifest_bytes=b"source",
            file_sha256=(),
        )
        context = SimpleNamespace(
            profile=SimpleNamespace(artifact_identity={"runtime_source": digest})
        )
        AeonQwenFleetAdapter._require_runtime_source(context, source)
        context.profile.artifact_identity["runtime_source"] = "b" * 64
        with self.assertRaises(QwenRuntimeError):
            AeonQwenFleetAdapter._require_runtime_source(context, source)

    def test_launch_returns_the_runtime_specific_loopback_endpoint(self):
        state = _state("fr-" + "4" * 32, 1)
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        resources = fleet.fleet_remote_runtime_resources(state["run_dir"], 1)
        adapter = AeonQwenFleetAdapter()
        adapter._prepared[resources["runtime_id"]] = {
            "capability": capability,
            "manifest_sha256": manifest,
            "source": state["source_manifest_sha256"],
            "lease": {"run_dir": state["run_dir"]},
            "environment": {"AEON_MAX_NUM_SEQS": "8"},
            "remote_resources": resources,
            "artifact_cache": {},
        }
        context = SimpleNamespace(
            runtime_id=resources["runtime_id"],
            lease=SimpleNamespace(run_dir=state["run_dir"]),
            heartbeat=lambda *_args: None,
            startup_check=lambda: None,
        )
        with patch.object(adapter_module, "remote_state", return_value=None), patch.object(
            adapter_module, "start_managed_remote_runtime", return_value=state
        ) as start:
            result = adapter.launch(context)

        self.assertEqual(
            result.endpoint, f"http://127.0.0.1:{state['local_port']}/v1"
        )
        self.assertEqual(start.call_args.kwargs["container_name"], resources["container_name"])
        self.assertEqual(start.call_args.kwargs["port"], resources["remote_port"])

    def test_probe_and_stop_target_the_runtime_run_dir(self):
        state = _state("fr-" + "3" * 32, 0)
        runtime = _runtime_record(state)
        adapter = AeonQwenFleetAdapter()
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        with patch.object(adapter_module, "remote_state", return_value=state) as receipt, patch.object(
            adapter_module, "remote_runtime_liveness", return_value="active"
        ) as liveness, patch.object(
            adapter_module, "tunnel_liveness", return_value="active"
        ), patch.object(
            adapter, "_endpoint_ready", return_value=True
        ):
            result = adapter.probe(runtime)
        self.assertEqual(result.state, ProbeState.READY)
        receipt.assert_called_once_with(state["run_dir"])
        liveness.assert_called_once_with(state["run_dir"])

        with patch.object(adapter_module, "remote_state", return_value=state), patch.object(
            adapter_module,
            "qwen_runtime_capability",
            return_value=(capability, manifest),
        ), patch.object(
            adapter_module, "stop_managed_remote_runtime", return_value=True
        ) as stop:
            result = adapter.stop(runtime, reason="test")
        self.assertTrue(result.process_absent)
        self.assertEqual(stop.call_args.kwargs["run_dir"], state["run_dir"])
        self.assertFalse(stop.call_args.kwargs["release_claim"])

    def test_probe_quarantines_a_missing_runtime_receipt(self):
        remote_state = _state("fr-" + "a" * 32, 0)
        remote_runtime = _runtime_record(remote_state)
        local_runtime = {
            **remote_runtime,
            "host": adapter_module.LOCAL_HOST,
        }
        adapter = AeonQwenFleetAdapter()

        with self.subTest(host="remote"), patch.object(
            adapter_module, "remote_state", return_value=None
        ), patch.object(
            adapter_module, "remote_runtime_liveness"
        ) as liveness:
            result = adapter.probe(remote_runtime)
            self.assertEqual(result.state, ProbeState.UNKNOWN)
            self.assertFalse(result.process_identity_verified)
            self.assertFalse(result.process_absent)
            self.assertIn("exact process absence is unproven", result.note)
            liveness.assert_not_called()

        with self.subTest(host="local"), patch.object(
            adapter_module, "current_runtime_state", return_value=None
        ), patch.object(adapter_module, "qwen_runtime_liveness") as liveness:
            result = adapter.probe(local_runtime)
            self.assertEqual(result.state, ProbeState.UNKNOWN)
            self.assertFalse(result.process_identity_verified)
            self.assertFalse(result.process_absent)
            self.assertIn("exact process absence is unproven", result.note)
            liveness.assert_not_called()

    def test_stop_refuses_a_missing_runtime_receipt(self):
        remote_state = _state("fr-" + "b" * 32, 1)
        remote_runtime = _runtime_record(remote_state)
        local_runtime = {
            **remote_runtime,
            "host": adapter_module.LOCAL_HOST,
        }
        adapter = AeonQwenFleetAdapter()

        with self.subTest(host="remote"), patch.object(
            adapter_module, "remote_state", return_value=None
        ), patch.object(adapter_module, "stop_managed_remote_runtime") as stop:
            result = adapter.stop(remote_runtime, reason="missing receipt")
            self.assertFalse(result.process_absent)
            self.assertFalse(result.identity_matched)
            self.assertIn("exact process absence is unproven", result.note)
            stop.assert_not_called()

        with self.subTest(host="local"), patch.object(
            adapter_module, "current_runtime_state", return_value=None
        ), patch.object(adapter_module, "stop_qwen_runtime") as stop:
            result = adapter.stop(local_runtime, reason="missing receipt")
            self.assertFalse(result.process_absent)
            self.assertFalse(result.identity_matched)
            self.assertIn("exact process absence is unproven", result.note)
            stop.assert_not_called()

    def test_probe_marks_only_exact_atomic_precontainer_recovery_absent(self):
        state = _precontainer_state("fr-" + "0" * 32, 0)
        runtime = _precontainer_runtime(state)
        adapter = AeonQwenFleetAdapter()
        with patch.object(
            adapter_module, "remote_state", return_value=state
        ), patch.object(
            adapter_module, "qwen_runtime_capability"
        ) as capability, patch.object(
            adapter_module, "stop_managed_remote_runtime", return_value=True
        ) as recover, patch.object(
            adapter_module, "remote_runtime_liveness"
        ) as liveness:
            capability.return_value = (
                type("Capability", (), {"key": state["runtime_capability_key"]})(),
                state["runtime_capability_manifest_sha256"],
            )
            result = adapter.probe(runtime)

        self.assertEqual(result.state, ProbeState.ABSENT)
        self.assertFalse(result.process_identity_verified)
        self.assertTrue(result.process_absent)
        self.assertTrue(result.prelaunch_cleanup_verified)
        recover.assert_called_once_with(
            capability.return_value[0],
            state["runtime_capability_manifest_sha256"],
            state["source_manifest_sha256"],
            release_reason="recover exact Fleet pre-container runtime",
            release_claim=False,
            run_dir=state["run_dir"],
            require_unlaunched=True,
        )
        liveness.assert_not_called()

    def test_probe_quarantines_unproven_or_mismatched_precontainer_intent(self):
        state = _precontainer_state("fr-" + "1" * 32, 1)
        runtime = _precontainer_runtime(state)
        adapter = AeonQwenFleetAdapter()
        with patch.object(
            adapter_module, "remote_state", return_value=state
        ), patch.object(
            adapter_module, "stop_managed_remote_runtime", return_value=False
        ) as recover:
            result = adapter.probe(runtime)
        self.assertEqual(result.state, ProbeState.UNKNOWN)
        self.assertFalse(result.process_absent)
        recover.assert_called_once()

        with patch.object(
            adapter_module, "remote_state", return_value=state
        ), patch.object(
            adapter_module, "stop_managed_remote_runtime"
        ) as recover:
            result = adapter.probe(
                {**runtime, "claim_id": "gc-another-lease"}
            )
        self.assertEqual(result.state, ProbeState.UNKNOWN)
        self.assertFalse(result.process_absent)
        recover.assert_not_called()

    def test_probe_recovers_only_an_exact_uncommitted_container_startup(self):
        state = {
            **_state("fr-" + "c" * 32, 0),
            "phase": "starting",
            "container_id": None,
            "tunnel_nonce": None,
            "tunnel_pid": None,
            "tunnel_create_time": None,
        }
        runtime = _uncommitted_container_runtime(state)
        adapter = AeonQwenFleetAdapter()
        with patch.object(
            adapter_module, "remote_state", return_value=state
        ), patch.object(
            adapter_module,
            "recover_remote_uncommitted_intent",
            return_value=True,
        ) as recover:
            result = adapter.probe(runtime)

        self.assertEqual(result.state, ProbeState.ABSENT)
        self.assertFalse(result.process_identity_verified)
        self.assertTrue(result.process_absent)
        self.assertTrue(result.prelaunch_cleanup_verified)
        recover.assert_called_once_with(state["run_dir"])

        with patch.object(
            adapter_module, "remote_state", return_value=state
        ), patch.object(
            adapter_module, "recover_remote_uncommitted_intent"
        ) as recover:
            result = adapter.probe({**runtime, "claim_id": "gc-another-lease"})
        self.assertEqual(result.state, ProbeState.UNKNOWN)
        self.assertFalse(result.process_absent)
        recover.assert_not_called()

    def test_probe_recovers_exact_local_uncommitted_startup(self):
        state = {
            "phase": "launching",
            "host": "192.168.0.177",
            "physical_gpu": 0,
            "gpu_uuid": "GPU-12345678-0000-0000-0000-000000000000",
            "claim_id": "gc-test-local-runtime",
            "owner": "owner-test-local-runtime",
            "run_dir": "/home/aday/.local/state/fleet-compute/runs/fr-"
            + "a" * 32,
            "container_id": "b" * 64,
            "container_pid": 4321,
        }
        runtime = {
            **{
                key: state[key]
                for key in (
                    "host",
                    "physical_gpu",
                    "gpu_uuid",
                    "claim_id",
                    "owner",
                    "run_dir",
                )
            },
            "runtime_id": "fr-" + "a" * 32,
            "state": "quarantined",
            "pid": None,
            "process_identity": None,
            "endpoint": None,
        }
        adapter = AeonQwenFleetAdapter()
        with patch.object(
            adapter_module, "current_runtime_state", return_value=state
        ), patch.object(
            adapter,
            "_recover_local_uncommitted_intent",
            return_value=True,
        ) as recover:
            result = adapter.probe(runtime)

        self.assertEqual(result.state, ProbeState.ABSENT)
        self.assertTrue(result.process_absent)
        self.assertTrue(result.prelaunch_cleanup_verified)
        recover.assert_called_once_with(runtime, state)

    def test_probe_keeps_an_incomplete_uncommitted_cleanup_quarantined(self):
        state = {
            **_state("fr-" + "d" * 32, 1),
            "phase": "starting",
            "tunnel_nonce": None,
            "tunnel_pid": None,
            "tunnel_create_time": None,
        }
        runtime = _uncommitted_container_runtime(state)
        adapter = AeonQwenFleetAdapter()
        with patch.object(
            adapter_module, "remote_state", return_value=state
        ), patch.object(
            adapter_module,
            "recover_remote_uncommitted_intent",
            return_value=False,
        ):
            result = adapter.probe(runtime)

        self.assertEqual(result.state, ProbeState.UNKNOWN)
        self.assertFalse(result.process_identity_verified)
        self.assertFalse(result.process_absent)

        with patch.object(
            adapter_module, "remote_state", return_value=state
        ), patch.object(
            adapter_module,
            "recover_remote_uncommitted_intent",
            side_effect=QwenRuntimeError("test exact stop proof failed"),
        ):
            result = adapter.probe(runtime)
        self.assertEqual(result.state, ProbeState.UNKNOWN)
        self.assertFalse(result.process_absent)
        self.assertIn("incomplete", result.note)

    def test_probe_never_recovers_a_nonquarantined_or_changed_launch(self):
        state = {
            **_state("fr-" + "f" * 32, 0),
            "phase": "starting",
            "container_id": None,
            "tunnel_nonce": None,
            "tunnel_pid": None,
            "tunnel_create_time": None,
        }
        runtime = _uncommitted_container_runtime(state)
        cases = (
            ({**runtime, "state": "starting"}, state),
            ({**runtime, "process_absent": 1}, state),
            (runtime, {**state, "phase": "ready"}),
            (runtime, {**state, "container_id": "short"}),
            (runtime, {**state, "tunnel_nonce": "a" * 64}),
            ({**runtime, "gpu_uuid": "GPU-87654321-wrong"}, state),
            ({**runtime, "run_dir": state["run_dir"] + "-changed"}, state),
        )
        adapter = AeonQwenFleetAdapter()
        for changed_runtime, changed_state in cases:
            with self.subTest(
                runtime_state=changed_runtime.get("state"),
                receipt_phase=changed_state.get("phase"),
            ), patch.object(
                adapter_module, "remote_state", return_value=changed_state
            ), patch.object(
                adapter_module, "recover_remote_uncommitted_intent"
            ) as recover:
                result = adapter.probe(changed_runtime)
            self.assertEqual(result.state, ProbeState.UNKNOWN)
            self.assertFalse(result.process_absent)
            recover.assert_not_called()

    def test_stop_refuses_ready_but_uncommitted_container_identity(self):
        state = _state("fr-" + "e" * 32, 0)
        runtime = _uncommitted_container_runtime(state)
        adapter = AeonQwenFleetAdapter()
        with patch.object(
            adapter_module, "remote_state", return_value=state
        ), patch.object(
            adapter_module, "stop_managed_remote_runtime", return_value=True
        ) as stop:
            result = adapter.stop(runtime, reason="test uncommitted cleanup")

        self.assertFalse(result.identity_matched)
        self.assertFalse(result.process_absent)
        stop.assert_not_called()

    def test_stop_accepts_only_the_same_exact_precontainer_binding(self):
        state = _precontainer_state("fr-" + "2" * 32, 0)
        runtime = _precontainer_runtime(state)
        adapter = AeonQwenFleetAdapter()
        capability, manifest = qwen_runtime_capability(
            "qwen38-compact-180-128k", require_enabled=True
        )
        with patch.object(
            adapter_module, "remote_state", return_value=state
        ), patch.object(
            adapter_module,
            "qwen_runtime_capability",
            return_value=(capability, manifest),
        ), patch.object(
            adapter_module, "stop_managed_remote_runtime", return_value=True
        ) as stop:
            result = adapter.stop(runtime, reason="test pre-container cleanup")
        self.assertTrue(result.identity_matched)
        self.assertTrue(result.process_absent)
        stop.assert_called_once()
        self.assertTrue(stop.call_args.kwargs["require_unlaunched"])

        with patch.object(
            adapter_module, "remote_state", return_value=state
        ), patch.object(
            adapter_module, "stop_managed_remote_runtime"
        ) as stop:
            result = adapter.stop(
                {**runtime, "gpu_uuid": "GPU-87654321-wrong"},
                reason="test mismatch",
            )
        self.assertFalse(result.identity_matched)
        self.assertFalse(result.process_absent)
        stop.assert_not_called()

    def test_probe_restores_a_provably_gone_exact_tunnel(self):
        state = _state("fr-" + "5" * 32, 0)
        runtime = _runtime_record(state)
        adapter = AeonQwenFleetAdapter()
        with patch.object(
            adapter_module, "remote_state", return_value=state
        ), patch.object(
            adapter_module, "remote_runtime_liveness", return_value="active"
        ), patch.object(
            adapter_module, "tunnel_liveness", return_value="gone"
        ), patch.object(
            adapter_module,
            "restore_managed_remote_tunnel",
            return_value=state,
        ) as restore, patch.object(
            adapter, "_endpoint_ready", return_value=True
        ):
            result = adapter.probe(runtime)

        self.assertEqual(result.state, ProbeState.READY)
        self.assertTrue(result.process_identity_verified)
        self.assertFalse(result.process_absent)
        restore.assert_called_once_with(state["run_dir"])

    def test_probe_does_not_recreate_an_ambiguous_tunnel(self):
        state = _state("fr-" + "6" * 32, 1)
        runtime = _runtime_record(state)
        adapter = AeonQwenFleetAdapter()
        with patch.object(
            adapter_module, "remote_state", return_value=state
        ), patch.object(
            adapter_module, "remote_runtime_liveness", return_value="active"
        ), patch.object(
            adapter_module, "tunnel_liveness", return_value="ambiguous"
        ), patch.object(
            adapter_module, "restore_managed_remote_tunnel"
        ) as restore, patch.object(
            adapter, "_endpoint_ready"
        ) as endpoint_ready:
            result = adapter.probe(runtime)

        self.assertEqual(result.state, ProbeState.UNKNOWN)
        self.assertFalse(result.process_identity_verified)
        self.assertFalse(result.process_absent)
        restore.assert_not_called()
        endpoint_ready.assert_not_called()

    def test_probe_retries_a_still_gone_tunnel_after_recovery_failure(self):
        state = _state("fr-" + "7" * 32, 0)
        runtime = _runtime_record(state)
        adapter = AeonQwenFleetAdapter()
        with patch.object(
            adapter_module, "remote_state", side_effect=[state, state]
        ) as receipt, patch.object(
            adapter_module,
            "remote_runtime_liveness",
            side_effect=["active", "active"],
        ) as runtime_liveness, patch.object(
            adapter_module, "tunnel_liveness", side_effect=["gone", "gone"]
        ), patch.object(
            adapter_module,
            "restore_managed_remote_tunnel",
            side_effect=QwenRuntimeError("test tunnel health timeout"),
        ), patch.object(
            adapter, "_endpoint_ready"
        ) as endpoint_ready:
            result = adapter.probe(runtime)

        self.assertEqual(result.state, ProbeState.STARTING)
        self.assertTrue(result.process_identity_verified)
        self.assertFalse(result.process_absent)
        self.assertIn("will be retried", result.note)
        self.assertEqual(receipt.call_count, 2)
        self.assertEqual(runtime_liveness.call_count, 2)
        endpoint_ready.assert_not_called()

    def test_probe_waits_for_an_exact_restored_tunnel_endpoint(self):
        state = _state("fr-" + "8" * 32, 1)
        runtime = _runtime_record(state)
        adapter = AeonQwenFleetAdapter()
        with patch.object(
            adapter_module, "remote_state", side_effect=[state, state]
        ), patch.object(
            adapter_module,
            "remote_runtime_liveness",
            side_effect=["active", "active"],
        ), patch.object(
            adapter_module, "tunnel_liveness", side_effect=["gone", "active"]
        ), patch.object(
            adapter_module,
            "restore_managed_remote_tunnel",
            side_effect=QwenRuntimeError("test tunnel health timeout"),
        ), patch.object(
            adapter, "_endpoint_ready", return_value=False
        ) as endpoint_ready:
            result = adapter.probe(runtime)

        self.assertEqual(result.state, ProbeState.STARTING)
        self.assertTrue(result.process_identity_verified)
        self.assertFalse(result.process_absent)
        self.assertIn("still becoming ready", result.note)
        endpoint_ready.assert_called_once_with(state)

    def test_probe_quarantines_ambiguous_state_after_recovery_failure(self):
        state = _state("fr-" + "9" * 32, 0)
        runtime = _runtime_record(state)
        adapter = AeonQwenFleetAdapter()
        with patch.object(
            adapter_module, "remote_state", side_effect=[state, state]
        ), patch.object(
            adapter_module,
            "remote_runtime_liveness",
            side_effect=["active", "active"],
        ), patch.object(
            adapter_module,
            "tunnel_liveness",
            side_effect=["gone", "ambiguous"],
        ), patch.object(
            adapter_module,
            "restore_managed_remote_tunnel",
            side_effect=QwenRuntimeError("test tunnel health timeout"),
        ), patch.object(
            adapter, "_endpoint_ready"
        ) as endpoint_ready:
            result = adapter.probe(runtime)

        self.assertEqual(result.state, ProbeState.UNKNOWN)
        self.assertFalse(result.process_identity_verified)
        self.assertFalse(result.process_absent)
        self.assertIn("identity changed", result.note)
        endpoint_ready.assert_not_called()

    def test_probe_quarantines_an_unreadable_post_recovery_receipt(self):
        state = _state("fr-" + "a" * 32, 1)
        runtime = _runtime_record(state)
        adapter = AeonQwenFleetAdapter()
        with patch.object(
            adapter_module,
            "remote_state",
            side_effect=[state, QwenRuntimeError("test unreadable receipt")],
        ), patch.object(
            adapter_module, "remote_runtime_liveness", return_value="active"
        ), patch.object(
            adapter_module, "tunnel_liveness", return_value="gone"
        ), patch.object(
            adapter_module,
            "restore_managed_remote_tunnel",
            side_effect=QwenRuntimeError("test tunnel health timeout"),
        ):
            result = adapter.probe(runtime)

        self.assertEqual(result.state, ProbeState.UNKNOWN)
        self.assertFalse(result.process_identity_verified)
        self.assertFalse(result.process_absent)
        self.assertIn("state is ambiguous", result.note)

    def test_finalize_rejects_non_fleet_or_non_absent_runtime_metadata(self):
        state = _state("fr-" + "e" * 32, 0)
        runtime = _runtime_record(state)
        adapter = AeonQwenFleetAdapter()
        with self.assertRaises(QwenRuntimeError):
            adapter.finalize_storage(
                {**runtime, "run_dir": "/tmp/not-fleet"},
                {"scratch_path": "/tmp/not-fleet"},
            )
        with self.assertRaises(QwenRuntimeError):
            adapter.finalize_storage(
                {**runtime, "process_absent": 0},
                {"scratch_path": runtime["run_dir"]},
            )
        with patch.object(adapter_module, "remote_state", return_value=state):
            with self.assertRaises(QwenRuntimeError):
                adapter.finalize_storage(
                    runtime, {"scratch_path": runtime["run_dir"]}
                )


if __name__ == "__main__":
    unittest.main()
