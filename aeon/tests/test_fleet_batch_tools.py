from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from aeon.core.benchmark_receipt import (
    CAPABILITY_RECEIPT_KEY_ENV,
    CAPABILITY_RECEIPT_PATH_ENV,
    decode_capability_receipts,
)
from aeon.core.agent_protocol import ToolStatus
from aeon.core.worker import Worker
from aeon.tools.command_fleet_guard import scrubbed_fleet_command_environment
from aeon.tools.fleet_batch import (
    FleetBatchCapabilitiesTool,
    FleetBatchJobStatusTool,
    FleetSubmitBatchJobTool,
)


JOB_ID = "fj-" + "a" * 32


class _Worker:
    def __init__(self, root: Path, objective: str, instance_id: str = "agent-one"):
        self.root = root
        self.current_objective = objective
        self.instance_id = instance_id

    def _instance_state_dir(self) -> Path:
        path = self.root / self.instance_id
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)
        return path


class _Client:
    def __init__(self):
        self.submissions = []

    def status(self):
        return {
            "profiles": [
                {
                    "profile_id": "aeon-qwen38-dflash-adapt",
                    "enabled": True,
                    "mode": "batch",
                    "project": "aeon-dflash-adapt",
                    "purpose": "exact adaptation",
                },
                {
                    "profile_id": "aeon-qwen38-full-gdn-quant",
                    "enabled": True,
                    "mode": "batch",
                    "project": "aeon-qwen38-full-gdn-quant",
                    "purpose": "exact conversion",
                },
            ]
        }

    def submit_job(self, **kwargs):
        self.submissions.append(kwargs)
        return {
            "job_id": JOB_ID,
            "profile_id": kwargs["profile"],
            "project": kwargs["project"],
            "demand_class": "standard",
            "state": "waiting_for_compute",
            "attempts": 0,
            "runtime_state": None,
            "wait_reason": "no compatible capacity",
            "retry_at": 123.0,
            "result": None,
            "last_error": None,
        }

    def job_status(self, job_id):
        assert job_id == JOB_ID
        return {
            "job_id": JOB_ID,
            "profile_id": "aeon-qwen38-dflash-adapt",
            "project": "aeon-dflash-adapt",
            "demand_class": "standard",
            "state": "running",
            "attempts": 1,
            "runtime_state": "running",
            "wait_reason": None,
            "retry_at": None,
            "result": None,
            "last_error": None,
        }


def test_general_huggingface_goal_gets_truthful_empty_recipe_catalog(tmp_path) -> None:
    worker = _Worker(
        tmp_path,
        "Find a useful Hugging Face model, make it on our GPUs, and upload it.",
    )
    with patch("aeon.tools.fleet_batch._client", return_value=_Client()):
        result = FleetBatchCapabilitiesTool(worker).execute()

    assert result.status is ToolStatus.OK
    assert result.raw["recipes"] == []
    assert result.raw["general_model_build_available"] is False
    assert result.raw["unavailable_compute_is_durable_wait"] is True


def test_capabilities_tool_emits_typed_benchmark_receipt(tmp_path) -> None:
    worker = _Worker(tmp_path, "Inspect available Fleet batch capabilities.")
    receipt_path = tmp_path / "capability.receipt"
    receipt_path.touch(mode=0o600)
    key = "b" * 64
    with (
        patch("aeon.tools.fleet_batch._client", return_value=_Client()),
        patch.dict(
            os.environ,
            {
                CAPABILITY_RECEIPT_PATH_ENV: str(receipt_path),
                CAPABILITY_RECEIPT_KEY_ENV: key,
            },
        ),
    ):
        result = FleetBatchCapabilitiesTool(worker).execute()

    receipts = decode_capability_receipts(receipt_path.read_bytes(), key=key)
    assert result.status is ToolStatus.OK
    assert len(receipts) == 1
    assert receipts[0].submission_boundary == "reviewed_recipe_only"
    assert receipts[0].unavailable_compute_is_durable_wait is True
    assert receipts[0].general_model_build_available is False


def test_benchmark_receipt_authority_is_hidden_from_model_commands() -> None:
    environment = scrubbed_fleet_command_environment(
        {
            "PATH": "/usr/bin",
            CAPABILITY_RECEIPT_PATH_ENV: "/private/receipt",
            CAPABILITY_RECEIPT_KEY_ENV: "a" * 64,
        }
    )
    assert CAPABILITY_RECEIPT_PATH_ENV not in environment
    assert CAPABILITY_RECEIPT_KEY_ENV not in environment


def test_exact_recipe_submits_once_and_status_is_agent_owned(tmp_path) -> None:
    worker = _Worker(
        tmp_path,
        "Build the exact Qwen3.8 DFlash adaptation through Fleet Compute.",
    )
    client = _Client()
    with patch("aeon.tools.fleet_batch._client", return_value=client):
        catalog = FleetBatchCapabilitiesTool(worker).execute()
        submitted = FleetSubmitBatchJobTool(worker).execute(
            "qwen38-dflash-adapt-v1", "candidate-one"
        )
        status = FleetBatchJobStatusTool(worker).execute(JOB_ID)

    assert [item["recipe_id"] for item in catalog.raw["recipes"]] == [
        "qwen38-dflash-adapt-v1"
    ]
    assert submitted.status is ToolStatus.PENDING
    assert submitted.raw["owned_by_agent"] is True
    assert status.status is ToolStatus.PENDING
    assert status.raw["state"] == "running"
    assert client.submissions[0]["payload"] == {"run_mode": "adapt-v1"}
    assert client.submissions[0]["profile"] == "aeon-qwen38-dflash-adapt"
    assert client.submissions[0]["project"] == "aeon-dflash-adapt"


def test_another_agent_cannot_read_owned_job(tmp_path) -> None:
    owner = _Worker(
        tmp_path,
        "Build the exact Qwen3.8 DFlash adaptation through Fleet Compute.",
        "owner-agent",
    )
    other = _Worker(
        tmp_path,
        "Build the exact Qwen3.8 DFlash adaptation through Fleet Compute.",
        "other-agent",
    )
    client = _Client()
    with patch("aeon.tools.fleet_batch._client", return_value=client):
        assert FleetSubmitBatchJobTool(owner).execute(
            "qwen38-dflash-adapt-v1", "candidate-one"
        ).status is ToolStatus.PENDING
        refused = FleetBatchJobStatusTool(other).execute(JOB_ID)

    assert refused.status is ToolStatus.BLOCKED
    assert refused.error_code == "fleet_batch_capability_unavailable"


def test_batch_receipt_is_the_only_new_wait_authority(tmp_path) -> None:
    worker = _Worker(
        tmp_path,
        "Build the exact Qwen3.8 DFlash adaptation through Fleet Compute.",
    )
    client = _Client()
    with patch("aeon.tools.fleet_batch._client", return_value=client):
        pending = FleetSubmitBatchJobTool(worker).execute(
            "qwen38-dflash-adapt-v1", "candidate-one"
        )

    assert Worker._has_verified_compute_wait(pending)
    pending.raw["owned_by_agent"] = False
    assert not Worker._has_verified_compute_wait(pending)
