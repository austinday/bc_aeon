"""Fake-backed coverage for durable benchmark matrices and current comparison."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from aeon.benchmarks import executor as executor_module
from aeon.benchmarks import service as service_module
from aeon.benchmarks.runner import run_benchmark
from aeon.benchmarks.service import (
    BenchmarkExecutionUnavailable,
    BenchmarkService,
)


def _service(tmp_path: Path) -> tuple[BenchmarkService, list[list[str]]]:
    launches: list[list[str]] = []
    service = BenchmarkService(
        tmp_path / "benchmarks", launcher=lambda argv: launches.append(list(argv))
    )
    return service, launches


def _matrix_request(
    serial: str,
    *,
    harness: str = "all",
    missing_only: bool = True,
) -> dict[str, object]:
    return {
        "request_id": f"bm-{serial * 32}",
        "harness_id": harness,
        "model_id": "local/qwen",
        "tool_profile_id": "fleet-local",
        "repetitions": 1,
        "missing_only": missing_only,
    }


def _single_request(serial: str, *, harness: str = "opencode") -> dict[str, object]:
    return {
        "request_id": f"br-{serial * 32}",
        "harness_id": harness,
        "model_id": "local/qwen",
        "tool_profile_id": "fleet-local",
        "repetitions": 1,
    }


def _runtime_status(harness_id: str | None = None) -> dict[str, object]:
    return {"supported": harness_id in {"opencode", "legacy-aeon"}, "reason": ""}


class _PassingExecutor:
    def execute(self, _request):
        return {
            "status": "passed",
            "score": 1.0,
            "wall_ms": 10.0,
            "active_wall_ms": 10.0,
            "compute_wait_ms": 0.0,
            "model_turn_count": 1,
            "tool_call_count": 0,
        }


def test_matrix_is_idempotent_bounded_and_uses_explicit_catalog_order(
    tmp_path: Path,
) -> None:
    service, launches = _service(tmp_path)
    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        created = service.submit_matrix(_matrix_request("a", missing_only=False))
        replay = service.submit_matrix(_matrix_request("a", missing_only=False))

    assert created["id"] == replay["id"]
    assert created["selected_count"] == 2
    assert created["created_count"] == 2
    assert created["skipped_count"] == 0
    assert [item["combination_id"] for item in created["items"]] == [
        "opencode:local-qwen:fleet-local",
        "legacy-aeon:local-qwen:fleet-local",
    ]
    assert [item["status"] for item in created["items"]] == ["queued", "pending"]
    assert len(launches) == 1
    with pytest.raises(ValueError, match="different request"):
        service.submit_matrix(_matrix_request("a", missing_only=True))


def test_matrix_all_skips_unavailable_harness_and_explicit_selection_rejects_it(
    tmp_path: Path,
) -> None:
    service, launches = _service(tmp_path)

    def only_opencode(harness_id: str | None = None) -> dict[str, object]:
        return {
            "supported": harness_id == "opencode",
            "reason": "runtime unavailable" if harness_id != "opencode" else "",
        }

    with patch.object(executor_module, "runtime_execution_status", only_opencode):
        batch = service.submit_matrix(_matrix_request("b", missing_only=False))
        with pytest.raises(
            BenchmarkExecutionUnavailable, match="harness is unavailable"
        ):
            service.submit_matrix(
                _matrix_request("c", harness="legacy-aeon", missing_only=False)
            )

    assert batch["selected_count"] == 1
    assert batch["items"][0]["combination_id"].startswith("opencode:")
    assert len(launches) == 1


def test_lost_response_replays_survive_runtime_availability_change(
    tmp_path: Path,
) -> None:
    service, _launches = _service(tmp_path)
    exact_request = _single_request("e")
    exact = service.submit(exact_request)
    service._launcher = service_module.default_launcher

    unavailable = {"supported": False, "reason": "synthetic outage"}
    with (
        patch.object(service, "_existing_submission_replay", return_value=None),
        patch.object(
            executor_module,
            "runtime_execution_status",
            return_value=unavailable,
        ) as runtime_status,
    ):
        # Force the fast replay lookup to miss, as it can in a concurrent
        # request that started immediately before the original commit. The
        # duplicate lookup under BEGIN IMMEDIATE remains authoritative.
        exact_replay = service.submit(exact_request)
    assert exact_replay["id"] == exact["id"]
    runtime_status.assert_not_called()

    service._launcher = lambda _argv: None
    matrix_request = _matrix_request("f", harness="opencode", missing_only=False)
    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        matrix = service.submit_matrix(matrix_request)
    with patch.object(
        executor_module,
        "runtime_execution_status",
        return_value=unavailable,
    ) as runtime_status:
        matrix_replay = service.submit_matrix(matrix_request)
    assert matrix_replay["id"] == matrix["id"]
    runtime_status.assert_not_called()


def test_only_one_matrix_child_is_active_globally_and_restart_kicks_pending(
    tmp_path: Path,
) -> None:
    service, launches = _service(tmp_path)
    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        first = service.submit_matrix(
            _matrix_request("d", harness="opencode", missing_only=False)
        )
        second = service.submit_matrix(
            _matrix_request("e", harness="legacy-aeon", missing_only=False)
        )
    assert len(launches) == 1
    assert first["items"][0]["status"] == "queued"
    assert second["items"][0]["status"] == "pending"

    # Model a service crash after the active child became terminal but before
    # its normal completion callback could kick the global matrix queue.
    with sqlite3.connect(service.database_path) as connection:
        connection.execute(
            "UPDATE benchmark_runs SET status = 'failed', error_code = 'worker_lost' "
            "WHERE id = ?",
            (first["run_ids"][0],),
        )
    resumed_launches: list[list[str]] = []
    reopened = BenchmarkService(
        service.root, launcher=lambda argv: resumed_launches.append(list(argv))
    )
    assert len(resumed_launches) == 1
    assert reopened.get_batch(str(second["id"]))["items"][0]["status"] == "queued"


def test_terminal_child_advances_global_queue_across_batches(tmp_path: Path) -> None:
    service, launches = _service(tmp_path)
    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        first = service.submit_matrix(
            _matrix_request("1", harness="opencode", missing_only=False)
        )
        second = service.submit_matrix(
            _matrix_request("2", harness="legacy-aeon", missing_only=False)
        )
    service._mark_failed(str(first["run_ids"][0]), error_code="launcher_failed")
    assert len(launches) == 2
    assert service.get_batch(str(second["id"]))["items"][0]["status"] == "queued"


def test_concurrent_run_all_missing_commits_one_child_set_and_one_launch(
    tmp_path: Path,
) -> None:
    first_launches: list[list[str]] = []
    second_launches: list[list[str]] = []
    root = tmp_path / "benchmarks"
    first = BenchmarkService(root, launcher=lambda argv: first_launches.append(list(argv)))
    second = BenchmarkService(
        root, launcher=lambda argv: second_launches.append(list(argv))
    )
    barrier = threading.Barrier(2)
    results: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def submit(service: BenchmarkService, serial: str) -> None:
        try:
            barrier.wait(timeout=2.0)
            results.append(service.submit_matrix(_matrix_request(serial)))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        threads = [
            threading.Thread(target=submit, args=(first, "c")),
            threading.Thread(target=submit, args=(second, "d")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    assert sorted(int(result["created_count"]) for result in results) == [0, 2]
    assert len(first_launches) + len(second_launches) == 1


def test_cancel_batch_cancels_owned_work_without_touching_reused_runs(
    tmp_path: Path,
) -> None:
    service, launches = _service(tmp_path)
    existing = service.submit(_single_request("3"))
    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        reused = service.submit_matrix(
            _matrix_request("4", harness="opencode", missing_only=True)
        )
        owned = service.submit_matrix(_matrix_request("5", missing_only=False))
    launches_before_cancel = len(launches)
    cancelled = service.cancel_batch(str(owned["id"]))
    assert {item["status"] for item in cancelled["items"]} == {"cancelled"}
    assert service.get_run(str(existing["id"]))["status"] == "queued"
    assert service.get_batch(str(reused["id"]))["items"][0]["created"] is False
    assert len(launches) == launches_before_cancel == 2


def test_run_all_missing_reuses_current_active_and_verified_success(
    tmp_path: Path,
) -> None:
    service, launches = _service(tmp_path)
    existing = service.submit(_single_request("6"))
    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        active_batch = service.submit_matrix(
            _matrix_request("7", harness="opencode", missing_only=True)
        )
    assert active_batch["created_count"] == 0
    assert active_batch["items"][0]["run_id"] == existing["id"]
    assert len(launches) == 1

    finished = run_benchmark(service, str(existing["id"]), executor=_PassingExecutor())
    assert finished["status"] == "succeeded"
    assert finished["evidence_verified"] is True
    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        success_batch = service.submit_matrix(
            _matrix_request("8", harness="opencode", missing_only=True)
        )
    assert success_batch["created_count"] == 0
    assert success_batch["items"][0]["run_id"] == existing["id"]


@pytest.mark.parametrize(
    ("status_value", "cancel_requested", "expected_terminal"),
    [
        ("queued", 0, "failed"),
        ("running", 0, "failed"),
        ("waiting_for_compute", 0, "failed"),
        ("cancelling", 1, "cancelled"),
    ],
)
def test_run_all_missing_replaces_a_stale_unregistered_active_row(
    tmp_path: Path,
    status_value: str,
    cancel_requested: int,
    expected_terminal: str,
) -> None:
    service, launches = _service(tmp_path)
    stale = service.submit(_single_request("a"))
    with sqlite3.connect(service.database_path) as connection:
        connection.execute(
            "UPDATE benchmark_runs SET status = ?, cancel_requested = ?, "
            "created_at = 0, worker_pid = NULL, worker_starttime = NULL WHERE id = ?",
            (status_value, cancel_requested, stale["id"]),
        )

    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        batch = service.submit_matrix(
            _matrix_request("b", harness="opencode", missing_only=True)
        )

    assert service.get_run(str(stale["id"]))["status"] == expected_terminal
    assert batch["created_count"] == 1
    assert batch["items"][0]["created"] is True
    assert batch["items"][0]["run_id"] != stale["id"]
    assert len(launches) == 2


def test_comparison_is_current_provenance_and_evidence_authority(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    existing = service.submit(_single_request("9"))
    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        active = service.comparison()
    entry = next(
        item
        for item in active["combinations"]
        if item["combination"]["harness_id"] == "opencode"
    )
    assert entry["state"] == "active"
    assert entry["needs_run"] is False

    finished = run_benchmark(service, str(existing["id"]), executor=_PassingExecutor())
    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        succeeded = service.comparison()
    entry = next(
        item
        for item in succeeded["combinations"]
        if item["combination"]["harness_id"] == "opencode"
    )
    assert entry["state"] == "succeeded"
    assert entry["evidence_verified"] is True
    assert entry["run"]["summary"]["overall_score"] == pytest.approx(100.0)

    evidence = service.evidence_root / str(finished["id"]) / "results.json"
    evidence.write_bytes(b"{}\n")
    evidence.chmod(0o600)
    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        tampered = service.comparison()
    entry = next(
        item
        for item in tampered["combinations"]
        if item["combination"]["harness_id"] == "opencode"
    )
    assert entry["state"] == "missing"
    assert entry["needs_run"] is True
    assert entry["run"] is None


def test_matrix_replay_survives_catalog_drift_after_commit(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    request = _matrix_request("a", harness="opencode", missing_only=False)
    with patch.object(executor_module, "runtime_execution_status", _runtime_status):
        created = service.submit_matrix(request)
    with patch.object(service_module, "BENCHMARK_CATALOG_SHA256", "f" * 64):
        replay = service.submit_matrix(request)
    assert replay["id"] == created["id"]
    assert replay["catalog_sha256"] == created["catalog_sha256"]


def test_historical_partial_request_can_only_be_replayed(tmp_path: Path) -> None:
    service, launches = _service(tmp_path)
    created = service.submit(_single_request("b"))
    with sqlite3.connect(service.database_path) as connection:
        connection.execute(
            "UPDATE benchmark_runs SET suite_id = 'tools', suite_label = 'Legacy tools' "
            "WHERE id = ?",
            (created["id"],),
        )
    replay_request = _single_request("b")
    replay_request["suite_id"] = "tools"
    assert service.submit(replay_request)["id"] == created["id"]
    fresh_partial = _single_request("c")
    fresh_partial["suite_id"] = "tools"
    with pytest.raises(ValueError, match="partial benchmark suites"):
        service.submit(fresh_partial)
    assert len(launches) == 1
