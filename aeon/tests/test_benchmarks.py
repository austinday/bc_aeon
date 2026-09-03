"""Hermetic durability and sanitization tests for the benchmark domain."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import signal
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import pytest

from aeon.core.fleet_backend import BENCHMARK_COMPUTE_STATUS_FD_ENV
from aeon.benchmarks import executor as executor_module
from aeon.benchmarks import runner as runner_module
from aeon.benchmarks import service as service_module
from aeon.benchmarks import worker as worker_module
from aeon.benchmarks.catalog import (
    BENCHMARK_CATALOG_SHA256,
    EXECUTOR_PROTOCOL_SHA256,
    EXECUTOR_PROTOCOL_VERSION,
    HARNESS_SOURCE_SHA256,
    RUNNER_PROTOCOL_SHA256,
    RUNNER_SOURCE_SHA256,
    SUITES,
    TOOL_SOURCE_SHA256,
    ToolProfileSpec,
    public_catalog,
)
from aeon.core.benchmark_receipt import (
    CAPABILITY_RECEIPT_KEY_ENV,
    CAPABILITY_RECEIPT_PATH_ENV,
    decode_capability_receipts,
    emit_fleet_wait_capability_receipt,
)
from aeon.benchmarks.protocol import (
    executor_source_sha256,
    harness_source_sha256,
    runner_source_sha256,
    tool_source_sha256,
)
from aeon.benchmarks.executor import (
    BENCHMARK_BROWSER_PROFILE,
    FleetHarnessExecutor,
    ProcessResult,
    run_bounded_process,
)
from aeon.harnesses import opencode_runtime
from aeon.benchmarks.runner import (
    ExecutionCancelled,
    ExecutionRequest,
    ExecutorUnavailable,
    ExecutorUnresolved,
    _execute_with_deadline,
    run_benchmark,
)
from aeon.benchmarks.service import (
    FLEET_LOW_PRIORITY,
    BenchmarkError,
    BenchmarkService,
)


def _request(
    serial: str = "1",
    *,
    suite: str = "smoke",
    harness: str = "opencode",
    repetitions: int = 1,
) -> dict[str, object]:
    return {
        "request_id": f"br-{serial * 32}",
        "suite_id": suite,
        "harness_id": harness,
        "model_id": "local/qwen",
        "repetitions": repetitions,
    }


def _service(tmp_path: Path):
    launches: list[list[str]] = []
    service = BenchmarkService(
        tmp_path / "benchmarks",
        launcher=lambda argv: launches.append(list(argv)),
    )
    return service, launches


def test_catalog_is_immutable_complete_json_safe_and_hashed() -> None:
    catalog = public_catalog()
    assert {item["id"] for item in catalog["suites"]} == {
        "smoke",
        "tools",
        "browser",
        "vision",
        "context",
        "comprehensive",
    }
    assert [item["id"] for item in catalog["harnesses"]] == [
        "opencode",
        "legacy-aeon",
    ]
    assert catalog["models"][0]["harnesses"] == ["opencode", "legacy-aeon"]
    required_by_suite = {
        item["id"]: item["required_capabilities"] for item in catalog["suites"]
    }
    assert required_by_suite == {
        "smoke": [],
        "tools": ["fleet-tools", "local-tools"],
        "browser": ["browser"],
        "vision": ["browser", "vision"],
        "context": [],
        "comprehensive": ["browser", "fleet-tools", "local-tools", "vision"],
    }
    assert catalog["models"][0]["identity_scope"] == "logical_service"
    assert catalog["models"][0]["service_id"] == "aeon-qwen38-standard"
    assert catalog["models"][0]["selection_semantics"] == "fleet_policy_routed"
    assert len(catalog["combinations"]) == 2
    assert re.fullmatch(r"[0-9a-f]{64}", BENCHMARK_CATALOG_SHA256)
    assert re.fullmatch(r"[0-9a-f]{64}", RUNNER_PROTOCOL_SHA256)
    assert catalog["executor_protocol_version"] == EXECUTOR_PROTOCOL_VERSION
    assert catalog["executor_protocol_sha256"] == EXECUTOR_PROTOCOL_SHA256
    assert re.fullmatch(r"[0-9a-f]{64}", EXECUTOR_PROTOCOL_SHA256)
    assert catalog["runner_source_sha256"] == RUNNER_SOURCE_SHA256
    assert catalog["harness_source_sha256"] == HARNESS_SOURCE_SHA256
    assert catalog["tool_source_sha256"] == TOOL_SOURCE_SHA256
    for digest in (RUNNER_SOURCE_SHA256, HARNESS_SOURCE_SHA256, TOOL_SOURCE_SHA256):
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
    serialized = json.dumps(catalog, allow_nan=False)
    assert "http://" not in serialized
    assert "https://" not in serialized
    assert "prompt" not in serialized.lower()

    catalog["models"][0]["harnesses"].append("tampered")
    catalog["suites"].clear()
    fresh = public_catalog()
    assert fresh["models"][0]["harnesses"] == ["opencode", "legacy-aeon"]
    assert len(fresh["suites"]) == 6
    with pytest.raises(FrozenInstanceError):
        SUITES["smoke"].version = "changed"


def test_private_separate_database_and_evidence_roots(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    assert stat.S_IMODE(service.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(service.evidence_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(service.database_path.stat().st_mode) == 0o600
    assert service.database_path.parent == service.root
    assert service.evidence_root.parent == service.root


def test_unsafe_or_relative_state_roots_are_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="absolute"):
        BenchmarkService("relative/benchmarks", launcher=lambda _argv: None)

    public = tmp_path / "public"
    public.mkdir(mode=0o700)
    public.chmod(0o755)
    with pytest.raises(BenchmarkError, match="owner-private"):
        BenchmarkService(public, launcher=lambda _argv: None)

    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(BenchmarkError, match="symbolic link"):
        BenchmarkService(linked, launcher=lambda _argv: None)


def test_submit_is_durable_idempotent_and_uses_only_fixed_worker_argv(
    tmp_path: Path,
) -> None:
    service, launches = _service(tmp_path)
    created = service.submit(_request())
    assert re.fullmatch(r"run-[0-9a-f]{32}", str(created["id"]))
    assert created["request_id"] == "br-" + "1" * 32
    assert created["status"] == "queued"
    assert created["tool_profile_id"] == "fleet-local"
    assert created["suite_version"] == "1"
    assert launches == [[
        FLEET_LOW_PRIORITY,
        os.sys.executable,
        "-m",
        "aeon.benchmarks.worker",
        "--root",
        str(service.root),
        "--run-id",
        created["id"],
    ]]
    duplicate = service.submit(_request())
    assert duplicate["id"] == created["id"]
    assert len(launches) == 1

    reopened = BenchmarkService(service.root, launcher=lambda _argv: None)
    assert reopened.get_run(str(created["id"]))["request_id"] == created["request_id"]
    assert reopened.list_runs(limit=1)["runs"][0]["id"] == created["id"]


def test_submit_rejects_request_id_reuse_for_a_different_request(
    tmp_path: Path,
) -> None:
    service, launches = _service(tmp_path)
    created = service.submit(_request())

    with pytest.raises(ValueError, match="already bound to a different request"):
        service.submit(_request(suite="tools"))

    assert len(launches) == 1
    assert service.get_run(str(created["id"]))["suite_id"] == "smoke"
    assert [run["id"] for run in service.list_runs()["runs"]] == [created["id"]]


def test_submit_rejects_tool_profile_missing_suite_capabilities(
    tmp_path: Path,
) -> None:
    service, launches = _service(tmp_path)
    incomplete = ToolProfileSpec(
        profile_id="fleet-local",
        label="Incomplete profile",
        version="test",
        capabilities=("local-tools",),
    )

    with patch.object(service_module, "TOOL_PROFILES", (incomplete,)):
        with pytest.raises(ValueError, match="required capabilities"):
            service.submit(_request(suite="vision"))

    assert launches == []
    assert service.list_runs()["runs"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"request_id": "bad"},
        {"request_id": "br-" + "A" * 32},
        {"suite_id": "unknown"},
        {"harness_id": "other"},
        {"model_id": "remote/model"},
        {"tool_profile_id": "direct-gpu"},
        {"repetitions": 0},
        {"repetitions": 21},
        {"repetitions": True},
        {"prompt": "secret raw prompt"},
    ],
)
def test_submit_rejects_unreviewed_shapes(tmp_path: Path, mutation: dict[str, object]) -> None:
    service, launches = _service(tmp_path)
    request = _request()
    request.update(mutation)
    with pytest.raises((TypeError, ValueError)):
        service.submit(request)
    assert launches == []
    assert service.list_runs()["runs"] == []


def test_launcher_failure_is_truthful_and_does_not_persist_diagnostics(
    tmp_path: Path,
) -> None:
    secret = "token=never-store http://127.0.0.1:9999 GPU-deadbeef"

    def fail(_argv):
        raise RuntimeError(secret)

    service = BenchmarkService(tmp_path / "benchmarks", launcher=fail)
    run = service.submit(_request())
    assert run["status"] == "failed"
    assert run["error_code"] == "launcher_failed"
    serialized = json.dumps(run)
    assert secret not in serialized
    assert "127.0.0.1" not in serialized
    assert secret.encode() not in service.database_path.read_bytes()


class _PassingExecutor:
    def execute(self, request):
        return {
            "status": "passed",
            "score": 0.8,
            "wall_ms": 25,
            "active_wall_ms": 20,
            "compute_wait_ms": 5,
            "tool_success": True,
            "output": "raw prompt token=must-not-survive",
            "endpoint": "http://127.0.0.1:9999/v1",
            "gpu_uuid": "GPU-must-not-survive",
        }


def test_runner_executes_deterministic_specs_and_publishes_sanitized_evidence(
    tmp_path: Path,
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request(suite="tools", repetitions=2))
    finished = run_benchmark(
        service, str(queued["id"]), executor=_PassingExecutor()
    )
    assert finished["status"] == "succeeded"
    assert finished["evidence_verified"] is True
    assert len(finished["cases"]) == 6
    assert finished["summary"] == {
        "score": pytest.approx(0.8),
        "completion_rate": pytest.approx(1.0),
        "median_wall_ms": pytest.approx(25.0),
        "median_active_wall_ms": pytest.approx(20.0),
        "median_compute_wait_ms": pytest.approx(5.0),
        "stuck_rate": pytest.approx(0.0),
        "unsupported_rate": pytest.approx(0.0),
        "tool_success_rate": pytest.approx(1.0),
        "browser_success_rate": pytest.approx(0.0),
        "vision_score": pytest.approx(0.0),
        "case_count": 6,
        "passed_cases": 6,
    }
    serialized = json.dumps(finished)
    for forbidden in ("raw prompt", "must-not-survive", "127.0.0.1", "GPU-"):
        assert forbidden not in serialized
    evidence_dir = service.evidence_root / str(finished["id"])
    evidence = evidence_dir / "results.json"
    assert stat.S_IMODE(evidence_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o600
    provenance = finished["provenance"]
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["suite_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["combination_sha256"])
    assert provenance["executor_protocol_version"] == EXECUTOR_PROTOCOL_VERSION
    assert provenance["executor_protocol_sha256"] == EXECUTOR_PROTOCOL_SHA256
    assert provenance["runner_source_sha256"] == RUNNER_SOURCE_SHA256
    assert provenance["harness_source_sha256"] == HARNESS_SOURCE_SHA256
    assert provenance["tool_source_sha256"] == TOOL_SOURCE_SHA256
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["evidence_sha256"])
    evidence_payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert evidence_payload["executor_protocol_version"] == EXECUTOR_PROTOCOL_VERSION
    assert evidence_payload["executor_protocol_sha256"] == EXECUTOR_PROTOCOL_SHA256
    assert evidence_payload["runner_source_sha256"] == RUNNER_SOURCE_SHA256
    assert evidence_payload["harness_source_sha256"] == HARNESS_SOURCE_SHA256
    assert evidence_payload["tool_source_sha256"] == TOOL_SOURCE_SHA256


@pytest.mark.parametrize(
    "digest_function",
    [
        executor_source_sha256,
        runner_source_sha256,
        harness_source_sha256,
        tool_source_sha256,
    ],
)
def test_execution_provenance_digest_changes_with_one_source_byte(
    tmp_path: Path, digest_function
) -> None:
    source = tmp_path / "trusted-executor.py"
    source.write_bytes(b"trusted-source-a")
    first = digest_function([source])
    source.write_bytes(b"trusted-source-b")
    second = digest_function([source])
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert re.fullmatch(r"[0-9a-f]{64}", second)
    assert first != second


@pytest.mark.parametrize(
    "column",
    ["runner_source_sha256", "harness_source_sha256", "tool_source_sha256"],
)
def test_runner_refuses_queued_source_provenance_drift(
    tmp_path: Path, column: str
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    with sqlite3.connect(service.database_path) as connection:
        connection.execute(
            f"UPDATE benchmark_runs SET {column} = ? WHERE id = ?",
            ("0" * 64, queued["id"]),
        )

    finished = run_benchmark(
        service, str(queued["id"]), executor=_PassingExecutor()
    )
    assert finished["status"] == "failed"
    assert finished["error_code"] == "runner_failed"
    assert finished["cases"] == []


def test_missing_real_executor_fails_instead_of_manufacturing_passes(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    finished = run_benchmark(service, str(queued["id"]))
    assert finished["status"] == "failed"
    assert finished["error_code"] == "executor_unavailable"
    assert finished["summary"]["completion_rate"] == 0.0
    assert finished["cases"][0]["status"] == "failed"
    assert all(case["status"] != "passed" for case in finished["cases"])


def test_cancel_is_idempotent_and_running_cancel_preserves_partial_evidence(
    tmp_path: Path,
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    cancelled = service.cancel(str(queued["id"]))
    assert cancelled["status"] == "cancelled"
    assert service.cancel(str(queued["id"]))["status"] == "cancelled"
    assert run_benchmark(
        service, str(queued["id"]), executor=_PassingExecutor()
    )["status"] == "cancelled"

    second = service.submit(_request("2", suite="tools"))

    class CancelAfterFirst:
        calls = 0

        def execute(self, request):
            self.calls += 1
            service.cancel(request.run_id)
            return {"status": "passed", "score": 1.0, "wall_ms": 1.0}

    executor = CancelAfterFirst()
    partial = run_benchmark(service, str(second["id"]), executor=executor)
    assert partial["status"] == "cancelled"
    assert executor.calls == 1
    assert len(partial["cases"]) == 1
    assert partial["evidence_verified"] is True


def test_evidence_tamper_fails_closed_without_returning_untrusted_cases(
    tmp_path: Path,
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    finished = run_benchmark(
        service, str(queued["id"]), executor=_PassingExecutor()
    )
    evidence = service.evidence_root / str(finished["id"]) / "results.json"
    evidence.write_text('{"run_id":"run-bad","cases":[{"label":"secret"}]}')
    evidence.chmod(0o600)
    readback = service.get_run(str(finished["id"]))
    assert readback["evidence_verified"] is False
    assert readback["cases"] == []
    assert "secret" not in json.dumps(readback)


def test_failed_evidence_publish_retains_owned_staging_for_manual_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(service_module.os, "replace", fail_replace)
    with pytest.raises(BenchmarkError, match="could not be published"):
        run_benchmark(service, str(queued["id"]), executor=_PassingExecutor())
    run_directory = service.evidence_root / str(queued["id"])
    staging = list(run_directory.glob(".results.json.tmp-*"))
    assert len(staging) == 1
    assert stat.S_IMODE(staging[0].stat().st_mode) == 0o600


def test_database_schema_mismatch_fails_closed(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    with sqlite3.connect(service.database_path) as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(BenchmarkError, match="schema"):
        BenchmarkService(service.root, launcher=lambda _argv: None)


def test_database_v1_migrates_worker_and_executor_identity_columns(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    with sqlite3.connect(service.database_path) as connection:
        connection.execute("PRAGMA user_version = 1")
    reopened = BenchmarkService(service.root, launcher=lambda _argv: None)
    with sqlite3.connect(reopened.database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(benchmark_runs)").fetchall()
        }
    assert {
        "worker_pid",
        "worker_starttime",
        "worker_registered_at",
        "executor_protocol_version",
        "executor_protocol_sha256",
        "runner_source_sha256",
        "harness_source_sha256",
        "tool_source_sha256",
    } <= columns


def test_database_v3_migration_marks_old_source_provenance_unbound(
    tmp_path: Path,
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    with sqlite3.connect(service.database_path) as connection:
        for column in (
            "runner_source_sha256",
            "harness_source_sha256",
            "tool_source_sha256",
        ):
            connection.execute(f"ALTER TABLE benchmark_runs DROP COLUMN {column}")
        connection.execute("PRAGMA user_version = 3")

    reopened = BenchmarkService(service.root, launcher=lambda _argv: None)
    with sqlite3.connect(reopened.database_path) as connection:
        row = connection.execute(
            """
            SELECT runner_source_sha256, harness_source_sha256, tool_source_sha256
            FROM benchmark_runs WHERE id = ?
            """,
            (queued["id"],),
        ).fetchone()
    assert row == ("0" * 64, "0" * 64, "0" * 64)
    finished = run_benchmark(
        reopened, str(queued["id"]), executor=_PassingExecutor()
    )
    assert finished["status"] == "failed"
    assert finished["error_code"] == "runner_failed"


def test_stale_worker_identity_is_reconciled_without_exposing_pid(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    with sqlite3.connect(service.database_path) as connection:
        connection.execute(
            """
            UPDATE benchmark_runs
            SET status = 'running', worker_pid = 2147483647,
                worker_starttime = 1, worker_registered_at = ?
            WHERE id = ?
            """,
            (time.time(), queued["id"]),
        )
    reconciled = service.get_run(str(queued["id"]))
    assert reconciled["status"] == "failed"
    assert reconciled["error_code"] == "worker_lost"
    serialized = json.dumps(reconciled)
    assert "2147483647" not in serialized
    assert "worker_pid" not in serialized


def test_unregistered_stale_queue_is_truthfully_failed(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    with sqlite3.connect(service.database_path) as connection:
        connection.execute(
            "UPDATE benchmark_runs SET created_at = 0 WHERE id = ?", (queued["id"],)
        )
    reconciled = service.list_runs()["runs"][0]
    assert reconciled["status"] == "failed"
    assert reconciled["error_code"] == "worker_lost"


def test_unknown_run_ids_and_bad_limits_are_rejected(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    with pytest.raises(KeyError):
        service.get_run("not-a-run")
    with pytest.raises(KeyError):
        service.cancel("run-" + "f" * 32)
    with pytest.raises(ValueError):
        service.list_runs(limit=0)
    with pytest.raises(ValueError):
        service.list_runs(limit=True)


def test_service_catalog_reports_fail_closed_runtime_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _launches = _service(tmp_path)
    monkeypatch.setattr(
        "aeon.benchmarks.executor.runtime_execution_status",
        lambda harness_id=None: {
            "supported": harness_id == "legacy-aeon",
            "reason": "Pinned runtime unavailable." if harness_id == "opencode" else "",
        },
    )
    catalog = service.catalog()
    assert catalog["submission_supported"] is True
    assert catalog["submission_unavailable_reason"] == ""
    assert {item["id"]: item["available"] for item in catalog["harnesses"]} == {
        "opencode": False,
        "legacy-aeon": True,
    }


def test_real_executor_builds_only_reviewed_harness_commands_and_safe_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    observed: list[list[str]] = []
    environments: list[dict[str, str]] = []
    working_directories: list[Path] = []

    def fake_process(
        argv, cwd, environment, timeout, cancel_requested, compute_state_changed
    ):
        del timeout, cancel_requested, compute_state_changed
        command = list(argv)
        observed.append(command)
        environments.append(dict(environment))
        working_directories.append(Path(cwd))
        prompt = command[command.index("--start") + 1]
        marker = "BENCH_SMOKE_DIRECT_7Q2" if "DIRECT" in prompt else "BENCH_SMOKE_323"
        return ProcessResult("exited", 0, marker.encode(), 3.0)

    with FleetHarnessExecutor(
        service,
        str(queued["id"]),
        process_runner=fake_process,
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        retained_workspace = executor.workspace
        finished = run_benchmark(service, str(queued["id"]), executor=executor)
    assert finished["status"] == "succeeded"
    assert finished["summary"]["completion_rate"] == pytest.approx(1.0)
    assert len(observed) == 2
    assert all(command[:3] == [sys.executable, "-m", "aeon.harnesses.opencode_runtime"] for command in observed)
    assert all("--non-interactive" in command for command in observed)
    assert retained_workspace.is_dir()
    assert stat.S_IMODE(retained_workspace.stat().st_mode) == 0o700
    assert len(environments) == len(working_directories) == 2
    for environment, workspace in zip(environments, working_directories, strict=True):
        configured_state = Path(environment["AEON_STATE_DIR"])
        assert not configured_state.is_relative_to(workspace)
        assert not workspace.is_relative_to(configured_state)
        assert stat.S_IMODE(configured_state.stat().st_mode) == 0o700

    # Exercise the OpenCode runtime's own state derivation and constructor
    # guard against the exact environment emitted by the benchmark executor.
    monkeypatch.chdir(working_directories[0])
    with patch.dict(os.environ, environments[0], clear=True):
        runtime_state = opencode_runtime._state_root()
        turn_runner = opencode_runtime.OpenCodeTurnRunner(
            binary=Path("/bin/true"),
            root=runtime_state,
            proxy=object(),
            logical_model="benchmark-test",
            max_steps=1,
            resume=False,
        )
    assert turn_runner.root == runtime_state
    assert not runtime_state.is_relative_to(working_directories[0])
    assert not working_directories[0].is_relative_to(runtime_state)


def test_mutation_repetitions_use_distinct_paths_and_markers(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request(suite="tools", repetitions=2))
    prompts: list[str] = []

    def fake_process(
        argv, cwd, environment, timeout, cancel_requested, compute_state_changed
    ):
        del cwd, environment, timeout, cancel_requested, compute_state_changed
        prompt = list(argv)[list(argv).index("--start") + 1]
        prompts.append(prompt)
        match = re.search(
            r"create (?P<path>/\S+) containing exactly (?P<marker>\S+) followed",
            prompt,
        )
        assert match is not None
        Path(match.group("path")).write_text(
            match.group("marker") + "\n", encoding="utf-8"
        )
        return ProcessResult(
            "exited",
            0,
            b"OpenCode \xc2\xb7 aeon_write_file \xc2\xb7 completed",
            4.0,
        )

    scenario = next(
        item for item in SUITES["tools"].cases if item.case_id == "tools.mutate_verify"
    )
    with FleetHarnessExecutor(
        service,
        str(queued["id"]),
        process_runner=fake_process,
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        results = []
        for repetition in (1, 2):
            request = ExecutionRequest(
                run_id=str(queued["id"]),
                suite_id="tools",
                scenario=scenario,
                repetition=repetition,
                harness_id="opencode",
                model_id="local/qwen",
                tool_profile_id="fleet-local",
                timeout_seconds=scenario.timeout_seconds,
            )
            results.append(executor.execute(request))
        first = executor.workspace / "results" / "mutation-repetition-1.txt"
        second = executor.workspace / "results" / "mutation-repetition-2.txt"

    assert [item["status"] for item in results] == ["passed", "passed"]
    assert first.read_text(encoding="utf-8") == "BENCH_WRITE_FINCH_4821_R1\n"
    assert second.read_text(encoding="utf-8") == "BENCH_WRITE_FINCH_4821_R2\n"
    assert prompts[0] != prompts[1]


def _fleet_capability_document() -> dict[str, object]:
    return {
        "status": "ok",
        "recipes": [],
        "general_model_build_available": False,
        "submission_boundary": "reviewed_recipe_only",
        "unavailable_compute_is_durable_wait": True,
        "guidance": "Submit only a listed recipe.",
    }


def test_fleet_wait_requires_authenticated_typed_tool_receipt_not_model_prose(
    tmp_path: Path,
) -> None:
    service, _launches = _service(tmp_path)
    scenario = next(
        item for item in SUITES["tools"].cases if item.case_id == "tools.fleet_wait"
    )

    def request_for(run_id: str) -> ExecutionRequest:
        return ExecutionRequest(
            run_id=run_id,
            suite_id="tools",
            scenario=scenario,
            repetition=1,
            harness_id="opencode",
            model_id="local/qwen",
            tool_profile_id="fleet-local",
            timeout_seconds=scenario.timeout_seconds,
        )

    prose_run = service.submit(_request("c", suite="tools"))

    def prose_only(
        _argv, _cwd, _environment, _timeout, _cancel, _compute_state_changed
    ):
        return ProcessResult(
            "exited",
            0,
            (
                "BENCH_FLEET_BOUNDARY_OK reviewed_recipe_only durable waiting "
                "OpenCode \u00b7 aeon_fleet_batch_capabilities \u00b7 completed"
            ).encode("utf-8"),
            3.0,
        )

    with FleetHarnessExecutor(
        service,
        str(prose_run["id"]),
        process_runner=prose_only,
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        prose_result = executor.execute(request_for(str(prose_run["id"])))
    assert prose_result["status"] == "failed"
    assert prose_result["tool_success"] is False

    receipt_run = service.submit(_request("d", suite="tools"))

    def typed_receipt(
        _argv, _cwd, environment, _timeout, _cancel, _compute_state_changed
    ):
        with patch.dict(
            os.environ,
            {
                CAPABILITY_RECEIPT_PATH_ENV: environment[CAPABILITY_RECEIPT_PATH_ENV],
                CAPABILITY_RECEIPT_KEY_ENV: environment[CAPABILITY_RECEIPT_KEY_ENV],
            },
        ):
            emit_fleet_wait_capability_receipt(_fleet_capability_document())
        return ProcessResult("exited", 0, b"ordinary completion", 3.0)

    with FleetHarnessExecutor(
        service,
        str(receipt_run["id"]),
        process_runner=typed_receipt,
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        receipt_result = executor.execute(request_for(str(receipt_run["id"])))
    assert receipt_result["status"] == "passed"
    assert receipt_result["tool_success"] is True


def test_capability_receipt_rejects_one_byte_tamper(tmp_path: Path) -> None:
    path = tmp_path / "receipt"
    path.touch(mode=0o600)
    key = "a" * 64
    with patch.dict(
        os.environ,
        {
            CAPABILITY_RECEIPT_PATH_ENV: str(path),
            CAPABILITY_RECEIPT_KEY_ENV: key,
        },
    ):
        emit_fleet_wait_capability_receipt(_fleet_capability_document())
    payload = path.read_bytes()
    assert len(decode_capability_receipts(payload, key=key)) == 1
    changed = bytearray(payload)
    changed[20] = changed[20] ^ 1
    assert decode_capability_receipts(bytes(changed), key=key) == ()

    # A valid authenticator over the wrong typed shape also fails; HMAC alone
    # is not treated as proof of the Fleet capability semantics.
    malformed = json.loads(payload)
    malformed["payload"]["capability_payload_sha256"] = 7
    encoded = json.dumps(
        malformed["payload"],
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    malformed["hmac_sha256"] = hmac.new(
        bytes.fromhex(key), encoded, hashlib.sha256
    ).hexdigest()
    assert decode_capability_receipts(
        (json.dumps(malformed, sort_keys=True) + "\n").encode("utf-8"), key=key
    ) == ()


def test_browser_scenarios_are_truthfully_unsupported_without_public_fixture(
    tmp_path: Path,
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request(suite="browser"))

    def must_not_launch(*_args):
        raise AssertionError("controlled browser cases must not use an external site")

    with FleetHarnessExecutor(
        service,
        str(queued["id"]),
        process_runner=must_not_launch,
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        finished = run_benchmark(service, str(queued["id"]), executor=executor)
    assert finished["status"] == "succeeded"
    assert finished["summary"]["unsupported_rate"] == pytest.approx(1.0)
    assert {case["status"] for case in finished["cases"]} == {"unsupported"}
    assert {case["error_code"] for case in finished["cases"]} == {
        "case_unsupported"
    }


def test_browser_suite_uses_closed_fixture_client_and_actual_harness_commands(
    tmp_path: Path,
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request(suite="browser"))
    fixture_calls: list[tuple[str, str]] = []
    environments: list[dict[str, str]] = []
    commands: list[list[str]] = []
    session_turns = 0

    def fixture_client(fixture_id: str, operation: str) -> bool:
        fixture_calls.append((fixture_id, operation))
        if fixture_id == "session-v1" and operation == "reopen":
            assert session_turns == 1
        if fixture_id == "session-v1" and operation == "verify":
            assert session_turns == 2
        return True

    def fake_process(
        argv, cwd, environment, timeout, cancel_requested, compute_state_changed
    ):
        nonlocal session_turns
        del cwd, timeout, cancel_requested, compute_state_changed
        command = list(argv)
        commands.append(command)
        prompt = command[command.index("--start") + 1]
        if "benchmark tab" in prompt or "tab_id benchmark" in prompt:
            if "Sign in" in prompt or "closed and reopened" in prompt:
                session_turns += 1
        environments.append(dict(environment))
        return ProcessResult(
            "exited",
            0,
            (
                "ORBIT-5521 Session preserved "
                "OpenCode \u00b7 aeon_browser_read \u00b7 completed "
                "OpenCode \u00b7 aeon_browser_interact \u00b7 completed"
            ).encode("utf-8"),
            5.0,
        )

    with FleetHarnessExecutor(
        service,
        str(queued["id"]),
        process_runner=fake_process,
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=fixture_client,
    ) as executor:
        finished = run_benchmark(service, str(queued["id"]), executor=executor)
    assert finished["status"] == "succeeded"
    assert finished["summary"]["browser_success_rate"] == pytest.approx(1.0)
    assert fixture_calls == [
        ("observe-v1", "seed"),
        ("form-v1", "seed"),
        ("form-v1", "verify"),
        ("session-v1", "seed"),
        ("session-v1", "reopen"),
        ("session-v1", "verify"),
        ("session-v1", "cleanup"),
    ]
    assert len(environments) == 4
    assert "--resume-unfinished" not in commands[-2]
    assert "--resume-unfinished" in commands[-1]
    assert all(
        env["AEON_BROWSER_SESSION_ID"] == f"oc-{str(queued['id'])[4:]}"
        for env in environments
    )
    assert all(
        env["AEON_BROWSER_PROFILE"] == BENCHMARK_BROWSER_PROFILE
        for env in environments
    )
    assert BENCHMARK_BROWSER_PROFILE == "benchmark-000000000000"


def test_shared_benchmark_profile_keeps_interleaved_runs_session_isolated(
    tmp_path: Path,
) -> None:
    service, _launches = _service(tmp_path)
    first_run = service.submit(_request("a", suite="browser"))
    second_run = service.submit(_request("b", suite="browser"))
    state: dict[tuple[str, str], str] = {}

    def client_for(run_id: str):
        session_id = f"oc-{run_id[4:]}"

        def fixture_client(fixture_id: str, operation: str) -> bool:
            key = (session_id, fixture_id)
            if operation == "seed":
                state[key] = "seeded"
                return True
            if operation == "reopen":
                if state.get(key) != "signed-in":
                    return False
                state[key] = "reopened"
                return True
            if operation == "cleanup":
                state.pop(key, None)
                return True
            return state.get(key) == "complete"

        return fixture_client

    first = FleetHarnessExecutor(
        service,
        str(first_run["id"]),
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=client_for(str(first_run["id"])),
    )
    second = FleetHarnessExecutor(
        service,
        str(second_run["id"]),
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=client_for(str(second_run["id"])),
    )
    assert first.browser_profile == second.browser_profile == BENCHMARK_BROWSER_PROFILE
    assert first.browser_session_id != second.browser_session_id
    assert first._fixture_operation("session-v1", "seed") is True
    state[(first.browser_session_id, "session-v1")] = "signed-in"
    assert first._fixture_operation("session-v1", "reopen") is True
    assert second._fixture_operation("session-v1", "reopen") is False
    assert second._fixture_operation("session-v1", "seed") is True
    state[(second.browser_session_id, "session-v1")] = "signed-in"
    assert second._fixture_operation("session-v1", "reopen") is True
    state[(first.browser_session_id, "session-v1")] = "complete"
    state[(second.browser_session_id, "session-v1")] = "complete"
    assert first._fixture_operation("session-v1", "verify") is True
    first.close()
    assert (first.browser_session_id, "session-v1") not in state
    assert second._fixture_operation("session-v1", "verify") is True
    second.close()


def test_browser_fixture_request_finishes_inside_cancel_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request(suite="browser"))
    observed: dict[str, object] = {}

    from aeon.tools import browser as browser_module

    monkeypatch.setattr(browser_module, "ensure_browser_running", lambda: True)
    monkeypatch.setattr(browser_module, "_browser_service_identity", lambda: "owned")
    monkeypatch.setattr(browser_module, "browser_auth_headers", lambda: {"Authorization": "Bearer test"})

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"fixture_id": "observe-v1", "status": "seeded"}

        @staticmethod
        def close() -> None:
            return None

    def post(_url, **kwargs):
        observed.update(kwargs)
        return Response()

    monkeypatch.setattr(executor_module.requests, "post", post)
    executor = FleetHarnessExecutor(
        service,
        str(queued["id"]),
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        # Keep close() hermetic; call the production fixture method directly.
        browser_fixture_client=lambda _fixture, _operation: False,
    )
    try:
        assert executor._browser_fixture("observe-v1", "seed") is True
    finally:
        executor.close()

    timeout = observed["timeout"]
    assert isinstance(timeout, (int, float))
    assert 0 < timeout < runner_module.EXECUTOR_CANCEL_GRACE_SECONDS


def test_runner_deadline_bounds_noncooperative_injected_executor(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    request = ExecutionRequest(
        run_id=str(queued["id"]),
        suite_id="smoke",
        scenario=SUITES["smoke"].cases[0],
        repetition=1,
        harness_id="opencode",
        model_id="local/qwen",
        tool_profile_id="fleet-local",
        timeout_seconds=0.05,
    )

    class BlockingExecutor:
        def execute(self, _request):
            time.sleep(0.25)
            return {"status": "passed"}

    started = time.monotonic()
    raw, _wall, cancelled = _execute_with_deadline(
        service, BlockingExecutor(), request
    )
    assert time.monotonic() - started < 0.2
    assert raw["status"] == "stuck"
    assert cancelled is False


def test_unresolved_cooperative_cancel_fails_instead_of_claiming_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    monkeypatch.setattr(
        "aeon.benchmarks.runner.EXECUTOR_CANCEL_GRACE_SECONDS", 0.05
    )

    class UnresolvedExecutor:
        calls = 0

        def execute(self, request):
            self.calls += 1
            service.cancel(request.run_id)
            time.sleep(0.3)
            return {"status": "passed"}

        def cancel(self, _request):
            return None

    executor = UnresolvedExecutor()
    finished = run_benchmark(service, str(queued["id"]), executor=executor)
    assert finished["status"] == "failed"
    assert finished["error_code"] == "executor_stuck"
    assert executor.calls == 1


def test_stuck_case_aborts_run_without_starting_another_case(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())

    class StuckExecutor:
        calls = 0

        def execute(self, _request):
            self.calls += 1
            return {"status": "stuck", "score": 0.0}

    executor = StuckExecutor()
    finished = run_benchmark(service, str(queued["id"]), executor=executor)
    assert finished["status"] == "failed"
    assert finished["error_code"] == "executor_stuck"
    assert executor.calls == 1
    assert len(finished["cases"]) == 1
    assert finished["cases"][0]["status"] == "stuck"


def test_bounded_process_terminates_nested_new_session_on_timeout(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "nested.pid"
    script = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
        "start_new_session=True); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid)); "
        "time.sleep(60)"
    )
    result = run_bounded_process(
        [sys.executable, "-c", script],
        tmp_path,
        os.environ.copy(),
        1.0,
        lambda: False,
    )
    assert result.state == "timeout"
    nested_pid = int(child_pid_file.read_text())
    deadline = time.monotonic() + 2.0
    while Path(f"/proc/{nested_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not Path(f"/proc/{nested_pid}").exists()


def test_bounded_process_excludes_proven_fleet_wait_from_case_deadline(
    tmp_path: Path,
) -> None:
    script = (
        "import os,time; "
        f"fd=int(os.environ[{BENCHMARK_COMPUTE_STATUS_FD_ENV!r}]); "
        "os.write(fd,b'waiting_for_compute\\n'); "
        "time.sleep(1.15); "
        "os.write(fd,b'allocated\\n'); "
        "time.sleep(0.05); "
        "print('ready-result')"
    )
    transitions: list[str] = []
    result = run_bounded_process(
        [sys.executable, "-c", script],
        tmp_path,
        os.environ.copy(),
        1.0,
        lambda: False,
        transitions.append,
    )
    assert result.state == "exited"
    assert result.returncode == 0
    assert result.wall_ms >= 1_100
    assert result.compute_wait_ms >= 1_000
    assert result.active_wall_ms < 500
    assert result.wall_ms == pytest.approx(
        result.active_wall_ms + result.compute_wait_ms
    )
    assert result.output.strip() == b"ready-result"
    assert transitions == ["waiting_for_compute", "allocated"]


def test_bounded_process_cancellation_remains_live_during_compute_wait(
    tmp_path: Path,
) -> None:
    script = (
        "import os,time; "
        f"fd=int(os.environ[{BENCHMARK_COMPUTE_STATUS_FD_ENV!r}]); "
        "os.write(fd,b'waiting_for_compute\\n'); "
        "time.sleep(60)"
    )
    observed_wait = threading.Event()
    cancel_after = time.monotonic() + 0.2

    def transition(state: str) -> None:
        if state == "waiting_for_compute":
            observed_wait.set()

    result = run_bounded_process(
        [sys.executable, "-c", script],
        tmp_path,
        os.environ.copy(),
        1.0,
        lambda: observed_wait.is_set() and time.monotonic() >= cancel_after,
        transition,
    )
    assert observed_wait.is_set()
    assert result.state == "cancelled"
    assert result.wall_ms < 5_000


def test_outer_deadline_remains_cancellable_but_pauses_for_compute_wait(
    tmp_path: Path,
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    request = ExecutionRequest(
        run_id=str(queued["id"]),
        suite_id="smoke",
        scenario=SUITES["smoke"].cases[0],
        repetition=1,
        harness_id="opencode",
        model_id="local/qwen",
        tool_profile_id="fleet-local",
        timeout_seconds=0.05,
    )

    class WaitingExecutor:
        def __init__(self):
            self.waiting = threading.Event()

        def prepare(self, _request):
            self.waiting.set()

        def deadline_paused(self, _request):
            return self.waiting.is_set()

        def execute(self, _request):
            time.sleep(0.2)
            self.waiting.clear()
            time.sleep(0.01)
            return {"status": "passed", "score": 1.0}

    raw, wall_ms, cancelled = _execute_with_deadline(
        service, WaitingExecutor(), request
    )
    assert raw["status"] == "passed"
    assert wall_ms >= 190
    assert raw["compute_wait_ms"] >= 150
    assert raw["active_wall_ms"] < raw["wall_ms"]
    assert cancelled is False


def test_wait_transition_is_public_status_not_a_second_demand(
    tmp_path: Path,
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    transitions: list[str] = []
    launches = 0

    def fake_process(
        argv, cwd, environment, timeout, cancel_requested, compute_state_changed
    ):
        nonlocal launches
        del argv, cwd, environment, timeout, cancel_requested
        launches += 1
        compute_state_changed("waiting_for_compute")
        transitions.append(service.get_run(str(queued["id"]))["status"])
        compute_state_changed("allocated")
        transitions.append(service.get_run(str(queued["id"]))["status"])
        marker = (
            "BENCH_SMOKE_DIRECT_7Q2" if launches == 1 else "BENCH_SMOKE_323"
        )
        return ProcessResult("exited", 0, marker.encode("ascii"), 1.0)

    with FleetHarnessExecutor(
        service,
        str(queued["id"]),
        process_runner=fake_process,
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        finished = run_benchmark(service, str(queued["id"]), executor=executor)

    assert finished["status"] == "succeeded"
    assert transitions == [
        "waiting_for_compute",
        "running",
        "waiting_for_compute",
        "running",
    ]
    assert launches == 2  # exactly one real harness process per planned case


def test_bounded_process_cleans_up_when_selector_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    children: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(executor_module.subprocess, "Popen", recording_popen)

    def broken_selector():
        raise RuntimeError("selector setup failed")

    monkeypatch.setattr(executor_module.selectors, "DefaultSelector", broken_selector)
    with pytest.raises(RuntimeError, match="selector setup"):
        run_bounded_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path,
            os.environ.copy(),
            30.0,
            lambda: False,
        )
    assert len(children) == 1
    assert children[0].poll() is not None


def test_prepared_cancel_cannot_be_cleared_by_delayed_execute(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    request = ExecutionRequest(
        run_id=str(queued["id"]),
        suite_id="smoke",
        scenario=SUITES["smoke"].cases[0],
        repetition=1,
        harness_id="opencode",
        model_id="local/qwen",
        tool_profile_id="fleet-local",
        timeout_seconds=1,
    )

    def cancelled_runner(
        _argv,
        _cwd,
        _environment,
        _timeout,
        cancel_requested,
        _compute_state_changed,
    ):
        assert cancel_requested() is True
        return ProcessResult("cancelled", -15, b"", 1.0)

    executor = FleetHarnessExecutor(
        service,
        str(queued["id"]),
        process_runner=cancelled_runner,
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=lambda _fixture, _operation: False,
    )
    executor.prepare(request)
    executor.cancel(request)
    with pytest.raises(ExecutionCancelled):
        executor.execute(request)
    executor.close()


def test_harness_environment_does_not_forward_unreviewed_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNREVIEWED_BENCHMARK_SECRET", "must-not-reach-harness")
    monkeypatch.setenv(BENCHMARK_COMPUTE_STATUS_FD_ENV, "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-must-not-reach-harness")
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    request = ExecutionRequest(
        run_id=str(queued["id"]),
        suite_id="smoke",
        scenario=SUITES["smoke"].cases[0],
        repetition=1,
        harness_id="opencode",
        model_id="local/qwen",
        tool_profile_id="fleet-local",
        timeout_seconds=1,
    )
    with FleetHarnessExecutor(
        service,
        str(queued["id"]),
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        environment = executor._environment(request)
    assert "UNREVIEWED_BENCHMARK_SECRET" not in environment
    assert BENCHMARK_COMPUTE_STATUS_FD_ENV not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "void"
    assert environment["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert environment["HIP_VISIBLE_DEVICES"] == "-1"
    assert environment["ROCR_VISIBLE_DEVICES"] == "-1"
    assert environment["PYTHONPATH"] == str(Path(executor_module.__file__).resolve().parents[2])
    assert environment["AEON_COMPUTE_BACKEND"] == "broker"


def test_executor_never_substitutes_an_unmapped_model(tmp_path: Path) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    request = ExecutionRequest(
        run_id=str(queued["id"]),
        suite_id="smoke",
        scenario=SUITES["smoke"].cases[0],
        repetition=1,
        harness_id="opencode",
        model_id="future/unmapped-model",
        tool_profile_id="fleet-local",
        timeout_seconds=1,
    )
    with FleetHarnessExecutor(
        service,
        str(queued["id"]),
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=lambda _fixture, _operation: False,
    ) as executor:
        with pytest.raises(ExecutorUnavailable, match="no reviewed runtime mapping"):
            executor._command(request, "bounded prompt")


def test_close_refuses_to_claim_success_while_executor_is_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _launches = _service(tmp_path)
    queued = service.submit(_request())
    request = ExecutionRequest(
        run_id=str(queued["id"]),
        suite_id="smoke",
        scenario=SUITES["smoke"].cases[0],
        repetition=1,
        harness_id="opencode",
        model_id="local/qwen",
        tool_profile_id="fleet-local",
        timeout_seconds=1,
    )
    started = threading.Event()

    def unresolved_runner(
        _argv,
        _cwd,
        _environment,
        _timeout,
        _cancel_requested,
        _compute_state_changed,
    ):
        started.set()
        time.sleep(0.3)
        return ProcessResult("exited", 0, b"BENCH_SMOKE_DIRECT_7Q2", 1.0)

    monkeypatch.setattr(executor_module, "EXECUTOR_CLOSE_GRACE_SECONDS", 0.05)
    executor = FleetHarnessExecutor(
        service,
        str(queued["id"]),
        process_runner=unresolved_runner,
        readiness_checker=lambda _harness: {"supported": True, "reason": ""},
        browser_fixture_client=lambda _fixture, _operation: False,
    )
    executor.prepare(request)
    thread = threading.Thread(target=executor.execute, args=(request,), daemon=True)
    thread.start()
    assert started.wait(1.0)
    with pytest.raises(ExecutorUnresolved):
        executor.close()
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_external_worker_wires_real_executor_into_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-" + "a" * 32
    observed: dict[str, object] = {}

    class FakeService:
        def __init__(self, root):
            observed["root"] = Path(root)

        def _mark_failed(self, *_args, **_kwargs):
            raise AssertionError("successful worker must not mark failure")

    class FakeExecutor:
        def __init__(self, service, supplied_run_id):
            observed["executor_service"] = service
            observed["executor_run_id"] = supplied_run_id

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            observed["closed"] = True

    def fake_run(service, supplied_run_id, *, executor):
        observed["runner_service"] = service
        observed["runner_run_id"] = supplied_run_id
        observed["runner_executor"] = executor
        return {"status": "succeeded"}

    monkeypatch.setattr(worker_module, "BenchmarkService", FakeService)
    monkeypatch.setattr(worker_module, "FleetHarnessExecutor", FakeExecutor)
    monkeypatch.setattr(worker_module, "run_benchmark", fake_run)
    assert worker_module.main(["--root", str(tmp_path), "--run-id", run_id]) == 0
    assert observed["root"] == tmp_path
    assert observed["executor_run_id"] == run_id
    assert observed["runner_run_id"] == run_id
    assert observed["runner_executor"] is not None
    assert observed["closed"] is True


def test_external_worker_signal_unwinds_executor_and_restores_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "run-" + "c" * 32
    observed: dict[str, object] = {}
    original = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }

    class FakeService:
        def __init__(self, _root):
            pass

    class FakeExecutor:
        def __init__(self, _service, _run_id):
            pass

        def __enter__(self):
            return self

        def __exit__(self, kind, _value, _traceback):
            observed["exit_kind"] = kind
            observed["closed"] = True

    def interrupted_run(_service, _run_id, *, executor):
        assert executor is not None
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        raise AssertionError("termination handler must raise")

    monkeypatch.setattr(worker_module, "BenchmarkService", FakeService)
    monkeypatch.setattr(worker_module, "FleetHarnessExecutor", FakeExecutor)
    monkeypatch.setattr(worker_module, "run_benchmark", interrupted_run)
    with pytest.raises(SystemExit) as stopped:
        worker_module.main(["--root", str(tmp_path), "--run-id", run_id])
    assert stopped.value.code == 128 + signal.SIGTERM
    assert observed["closed"] is True
    assert observed["exit_kind"] is SystemExit
    assert {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    } == original
