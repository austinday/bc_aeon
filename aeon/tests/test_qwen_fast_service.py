from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from aeon.core import qwen_fast_service_adapter as fast
from aeon.core import qwen_speed_lab_adapter as speed


def test_fast_release_evidence_and_profile_are_hash_bound() -> None:
    fast.AeonQwenFastServiceAdapter._verify_evidence()
    sources = speed._source_manifest()
    prompt, _identities = speed._prompt_bundle()
    profile_path = (
        Path(__file__).resolve().parents[3]
        / "fleet_compute/profiles.d/aeon-qwen38-fast-180.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert "aeon/core/qwen_fast_service_adapter.py" in sources
    assert profile["profile_id"] == fast.PROFILE_ID
    assert profile["adapter"] == "aeon-qwen38-fast-service-v1"
    assert profile["artifact_identity"] == (
        fast.AeonQwenFastServiceAdapter._expected_artifacts(sources, prompt)
    )


def test_fast_release_uses_only_the_gated_winner() -> None:
    variant = speed.VARIANT_CONFIGS[fast.WINNER_VARIANT]

    assert variant["model_id"] == "fullgdn"
    assert variant["draft_id"] == "bf16"
    assert variant["speculative_method"] == "dflash"
    assert variant["speculative_tokens"] == 7
    assert variant["attention_backend"] == "FLASHINFER"
    assert variant["kv_cache_dtype"] == "fp8"
    assert variant["compilation_profile"] == "piecewise"
    assert fast.LOCAL_ENDPOINT == "http://127.0.0.1:18034/v1"


def _tunnel_receipt(*, state: str = "active") -> dict:
    return {
        "schema_version": 1,
        "runtime_id": "fr-" + "a" * 32,
        "request_sha256": "b" * 64,
        "state": state,
        "pid": 4321,
        "start_ticks": 99,
        "created_at": 1.0,
    }


def test_fast_tunnel_lifecycle_never_scans_or_adopts_processes() -> None:
    source = Path(fast.__file__).read_text(encoding="utf-8")
    assert 'Path("/proc").iterdir()' not in source
    assert "def _tunnel_candidates" not in source

    with tempfile.TemporaryDirectory() as temporary:
        assert fast._stop_tunnel(
            Path(temporary), "fr-" + "a" * 32, "b" * 64
        ) is False


def test_fast_tunnel_publishes_intent_before_process_launch() -> None:
    receipt = _tunnel_receipt(state="starting")
    fake_socket = MagicMock()
    with tempfile.TemporaryDirectory() as temporary, patch.object(
        fast.socket, "socket", return_value=fake_socket
    ), patch.object(fast.subprocess, "Popen", side_effect=OSError("blocked")):
        with pytest.raises(OSError, match="blocked"):
            fast._start_tunnel(
                Path(temporary), receipt["runtime_id"], receipt["request_sha256"]
            )
        persisted = fast._private_json(Path(temporary) / fast._TUNNEL_RECEIPT)
    assert persisted["state"] == "starting"
    assert persisted["pid"] is None
    fake_socket.bind.assert_called_once_with(("127.0.0.1", fast.LOCAL_PORT))


def test_fast_tunnel_refuses_preexisting_starting_intent() -> None:
    receipt = _tunnel_receipt(state="starting")
    receipt["pid"] = None
    receipt["start_ticks"] = None
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / fast._TUNNEL_RECEIPT
        fast._atomic_json(path, receipt)
        with patch.object(fast.subprocess, "Popen") as popen:
            with pytest.raises(fast.QwenSpeedLabError, match="already exists"):
                fast._start_tunnel(
                    Path(temporary), receipt["runtime_id"], receipt["request_sha256"]
                )
        popen.assert_not_called()


def test_fast_endpoint_body_closes_on_advertised_oversize() -> None:
    response = MagicMock()
    response.headers = {"content-length": str(256 * 1024 + 1)}
    with pytest.raises(fast.QwenSpeedLabError, match="exceeded"):
        fast._bounded_loopback_body(response, 256 * 1024)
    response.iter_content.assert_not_called()
    response.close.assert_called_once_with()


def test_fast_stop_refuses_pid_reuse_or_identity_drift() -> None:
    receipt = _tunnel_receipt()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / fast._TUNNEL_RECEIPT
        fast._atomic_json(path, receipt)
        with (
            patch.object(fast, "_tunnel_exact", side_effect=[True, False]),
            patch.object(fast, "_pid_slot_absent", return_value=False),
            patch.object(fast.os, "kill") as kill_process,
        ):
            assert fast._stop_tunnel(
                Path(temporary), receipt["runtime_id"], receipt["request_sha256"]
            ) is False
        kill_process.assert_called_once_with(receipt["pid"], fast.signal.SIGTERM)
        assert fast._private_json(path)["state"] == "active"


def test_fast_stop_requires_and_records_exact_pid_absence() -> None:
    receipt = _tunnel_receipt()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / fast._TUNNEL_RECEIPT
        fast._atomic_json(path, receipt)
        with (
            patch.object(fast, "_tunnel_exact", return_value=True),
            patch.object(fast, "_pid_slot_absent", return_value=True),
            patch.object(fast.os, "kill"),
            patch.object(fast.os, "waitpid", side_effect=ChildProcessError),
        ):
            assert fast._stop_tunnel(
                Path(temporary), receipt["runtime_id"], receipt["request_sha256"]
            ) is True
        assert fast._private_json(path)["state"] == "stopped"
