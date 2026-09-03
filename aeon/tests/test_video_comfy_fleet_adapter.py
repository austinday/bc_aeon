"""Hermetic contracts for Aeon's dedicated video-rendering Fleet lane."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from fleet_compute.artifact_cache import ArtifactCacheSafetyError
from fleet_compute.models import ProbeState

from aeon.core.video_artifact_cache import VideoArtifactCacheBackend
from aeon.core.video_comfy_fleet_adapter import (
    AeonVideoComfyFleetAdapter,
    VideoComfyFleetError,
)
from aeon.core.video_comfy_release import (
    VIDEO_ADAPTER_ID,
    VIDEO_IMAGE_ID,
    VIDEO_LOCAL_PROFILE_IDENTITIES,
    VIDEO_PROFILE_IDENTITIES,
    VIDEO_SERVICE_ID,
    VIDEO_WORKER_HOSTNAMES,
    VIDEO_WORKER_SCRATCH_ROOT,
)


WORKSPACE = Path(__file__).resolve().parents[3]
PROFILES = WORKSPACE / "fleet_compute" / "profiles.d"
RUNTIME_ID = "fr-" + "1" * 32
CONTAINER_ID = "2" * 64
PROCESS_IDENTITY = "video-comfy:" + "3" * 64


def _profile(name: str) -> dict:
    value = json.loads((PROFILES / name).read_text(encoding="utf-8"))
    expected = value.pop("manifest_sha256")
    actual = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert expected == actual
    return value


def _runtime_and_receipt() -> tuple[dict, dict]:
    runtime = {
        "runtime_id": RUNTIME_ID,
        "profile_id": VIDEO_SERVICE_ID,
        "host": "192.168.0.178",
        "physical_gpu": 1,
        "claim_id": "gc-video-test",
        "pid": 4321,
        "process_identity": PROCESS_IDENTITY,
    }
    receipt = {
        "schema_version": 1,
        "runtime_id": RUNTIME_ID,
        "profile_id": VIDEO_SERVICE_ID,
        "host": "192.168.0.178",
        "claim_id": "gc-video-test",
        "container_name": "aeon_video_comfyui_fr_" + "1" * 32,
        "container_id": CONTAINER_ID,
        "container_pid": 4321,
        "remote_port": 28189,
        "local_port": 24567,
        "tunnel_pid": 9876,
        "tunnel_create_time": 1234.5,
        "image_id": VIDEO_IMAGE_ID,
        "process_identity": PROCESS_IDENTITY,
    }
    return runtime, receipt


def _prelaunch_runtime() -> dict:
    return {
        "runtime_id": RUNTIME_ID,
        "profile_id": VIDEO_SERVICE_ID,
        "adapter": VIDEO_ADAPTER_ID,
        "mode": "service",
        "state": "starting",
        "host": "192.168.0.178",
        "physical_gpu": 1,
        "claim_id": "gc-video-test",
        "run_dir": str(VIDEO_WORKER_SCRATCH_ROOT / RUNTIME_ID),
        "pid": None,
        "process_identity": None,
        "endpoint": None,
    }


def test_video_profiles_form_one_local_first_service() -> None:
    local = _profile("aeon-video-comfyui-local-177.json")
    worker = _profile("aeon-video-comfyui.json")

    assert local["service_id"] == worker["service_id"] == VIDEO_SERVICE_ID
    assert local["adapter"] == worker["adapter"] == VIDEO_ADAPTER_ID
    assert local["variant_priority"] < worker["variant_priority"]
    assert local["artifact_identity"] == VIDEO_LOCAL_PROFILE_IDENTITIES
    assert worker["artifact_identity"] == VIDEO_PROFILE_IDENTITIES
    assert worker["min_physical_vram_gb"] == 42
    assert {
        placement["host"] for placement in worker["placements"]
    } == set(VIDEO_WORKER_HOSTNAMES)


def test_video_cache_accepts_only_docker_capable_release_workers() -> None:
    backend = VideoArtifactCacheBackend()
    for host, hostname in VIDEO_WORKER_HOSTNAMES.items():
        assert backend._host(host) == hostname
    with pytest.raises(ArtifactCacheSafetyError):
        backend._host("192.168.0.179")


def test_video_cache_implements_atomic_promotion_contract() -> None:
    parameters = set(inspect.signature(VideoArtifactCacheBackend.promote).parameters)
    assert {
        "host",
        "temporary_path",
        "final_path",
        "descriptor",
        "identity_token",
        "expected_filesystem_id",
        "owner_uid",
    } <= parameters


def test_video_cache_staging_forces_private_regular_file_permissions() -> None:
    source = inspect.getsource(VideoArtifactCacheBackend.stage)
    assert '"--chmod=Fu=rw,Fgo="' in source


def test_worker_cleanup_accepts_only_receipt_scoped_root_regular_outputs() -> None:
    source = inspect.getsource(AeonVideoComfyFleetAdapter._cleanup_worker_scratch)

    assert "item_meta.st_uid in {0, os.geteuid()}" in source
    assert "item_meta.st_nlink==1" in source
    assert "item_meta.st_uid==os.geteuid(); item.rmdir()" in source
    assert "not stat.S_ISLNK(item_meta.st_mode)" in source
    assert "not os.path.ismount(item)" in source


def test_worker_receipt_is_bound_to_exact_host_gpu_and_container() -> None:
    runtime, receipt = _runtime_and_receipt()
    assert AeonVideoComfyFleetAdapter._runtime_from_receipt(runtime, receipt) == receipt

    for field, replacement in (
        ("host", "192.168.0.179"),
        ("container_name", "another-container"),
        ("remote_port", 28188),
        ("container_id", "short"),
        ("image_id", "sha256:" + "f" * 64),
    ):
        changed = {**receipt, field: replacement}
        with pytest.raises(VideoComfyFleetError):
            AeonVideoComfyFleetAdapter._runtime_from_receipt(runtime, changed)


def test_container_match_requires_all_owner_labels_and_exact_image() -> None:
    runtime, receipt = _runtime_and_receipt()
    container = {
        "container_id": CONTAINER_ID,
        "image_id": VIDEO_IMAGE_ID,
        "labels": {
            "com.bc_aeon.component": "video-comfyui",
            "com.bc_aeon.claim": runtime["claim_id"],
            "com.bc_aeon.runtime": runtime["runtime_id"],
        },
    }
    assert AeonVideoComfyFleetAdapter._container_matches(container, receipt)
    container["labels"] = {**container["labels"], "com.bc_aeon.claim": "gc-other"}
    assert not AeonVideoComfyFleetAdapter._container_matches(container, receipt)


def test_prelaunch_cache_failure_cleans_only_exact_absent_scratch() -> None:
    adapter = AeonVideoComfyFleetAdapter()
    runtime = _prelaunch_runtime()
    storage = {
        "scratch_path": f"/home/aday/.local/state/fleet-compute/runs/{RUNTIME_ID}",
        "filesystem_id": None,
    }
    with (
        patch.object(adapter, "_remote_container", return_value=None),
        patch.object(
            adapter,
            "_remote_storage_metrics",
            side_effect=VideoComfyFleetError("scratch absent"),
        ),
        patch.object(
            adapter, "_cleanup_worker_scratch", return_value=(True, 0)
        ) as cleanup,
    ):
        result = adapter.finalize_storage(runtime, storage)

    assert result.output_settled is True
    assert result.cleanup_complete is True
    cleanup.assert_called_once_with(
        "192.168.0.178",
        storage["scratch_path"],
        RUNTIME_ID,
        "prelaunch-absent",
    )


def test_probe_recovers_only_a_provably_absent_prelaunch_worker() -> None:
    adapter = AeonVideoComfyFleetAdapter()
    runtime = _prelaunch_runtime()
    with patch.object(AeonVideoComfyFleetAdapter, "_remote_container", return_value=None):
        result = adapter.probe(runtime)

    assert result.state is ProbeState.ABSENT
    assert result.process_identity_verified is False
    assert result.process_absent is True


def test_probe_quarantines_a_receiptless_attempt_with_an_exact_container() -> None:
    adapter = AeonVideoComfyFleetAdapter()
    runtime = _prelaunch_runtime()
    with patch.object(
        AeonVideoComfyFleetAdapter,
        "_remote_container",
        return_value={"running": True},
    ):
        result = adapter.probe(runtime)

    assert result.state is ProbeState.UNKNOWN
    assert result.process_absent is False


def test_stop_settles_only_a_provably_absent_prelaunch_worker() -> None:
    adapter = AeonVideoComfyFleetAdapter()
    runtime = _prelaunch_runtime()
    with patch.object(AeonVideoComfyFleetAdapter, "_remote_container", return_value=None):
        result = adapter.stop(runtime, reason="restart recovery")

    assert result.process_absent is True
    assert result.identity_matched is True


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("adapter", "another-adapter"),
        ("mode", "batch"),
        ("state", "ready"),
        ("claim_id", "not-a-claim"),
        ("run_dir", "/home/aday/.local/state/fleet-compute/runs/another"),
        ("endpoint", "http://127.0.0.1:1234"),
    ),
)
def test_prelaunch_absence_requires_the_full_durable_identity(
    field: str, replacement: object
) -> None:
    runtime = {**_prelaunch_runtime(), field: replacement}
    with patch.object(AeonVideoComfyFleetAdapter, "_remote_container") as inspect:
        assert not AeonVideoComfyFleetAdapter._prelaunch_absent(runtime)
    inspect.assert_not_called()


def test_container_absence_requires_a_successful_exact_name_census() -> None:
    missing = subprocess.CompletedProcess([], 1, stdout="", stderr="missing")
    absent = subprocess.CompletedProcess(
        [], 0, stdout='"another-container"\n', stderr=""
    )
    exact_present = subprocess.CompletedProcess(
        [],
        0,
        stdout=json.dumps(AeonVideoComfyFleetAdapter._container_name(RUNTIME_ID))
        + "\n",
        stderr="",
    )
    transport_failure = subprocess.CompletedProcess(
        [], 255, stdout="", stderr="connection lost"
    )

    with patch.object(
        AeonVideoComfyFleetAdapter,
        "_remote_run",
        side_effect=(missing, absent),
    ):
        assert (
            AeonVideoComfyFleetAdapter._remote_container(
                "192.168.0.178",
                AeonVideoComfyFleetAdapter._container_name(RUNTIME_ID),
            )
            is None
        )
    for census in (exact_present, transport_failure):
        with patch.object(
            AeonVideoComfyFleetAdapter,
            "_remote_run",
            side_effect=(missing, census),
        ), pytest.raises(VideoComfyFleetError):
            AeonVideoComfyFleetAdapter._remote_container(
                "192.168.0.178",
                AeonVideoComfyFleetAdapter._container_name(RUNTIME_ID),
            )
