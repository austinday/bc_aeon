"""Hermetic regressions for compact Qwen release-evidence recomputation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aeon.scripts import benchmark_qwen38_mtp as mtp_benchmark
from aeon.scripts import validate_qwen38_release_evidence as validator


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _mtp_report() -> dict:
    records = []
    for repeat in range(validator.EXPECTED_REPEATS):
        for case in mtp_benchmark.CASES:
            completion = 120
            elapsed = 0.6
            ttft = 0.1
            decode_seconds = elapsed - ttft
            records.append(
                {
                    "action_sha256": validator._expected_action_sha256(case),
                    "case": case["name"],
                    "completion_tokens": completion,
                    "decode_seconds": decode_seconds,
                    "decode_tps": completion / decode_seconds,
                    "elapsed_seconds": elapsed,
                    "final_sha256": _digest(f"final:{case['name']}"),
                    "finish_reason": "stop",
                    "prompt_tokens": 100,
                    "repeat": repeat,
                    "response_sha256": _digest(f"response:{case['name']}"),
                    "schema_valid": True,
                    "semantic_valid": True,
                    "total_tps": completion / elapsed,
                    "ttft_seconds": ttft,
                }
            )
    decode = [record["decode_tps"] for record in records]
    total = [record["total_tps"] for record in records]
    elapsed = [record["elapsed_seconds"] for record in records]
    return {
        "benchmark_script_sha256": hashlib.sha256(
            Path(mtp_benchmark.__file__).read_bytes()
        ).hexdigest(),
        "entry_name": validator.EXPECTED_MODEL,
        "errors": [],
        "k": validator.EXPECTED_MTP_K,
        "median_decode_tps": sorted(decode)[len(decode) // 2],
        "median_total_tps": sorted(total)[len(total) // 2],
        "model": validator.EXPECTED_MODEL,
        "p95_latency_seconds": mtp_benchmark._percentile(elapsed, 0.95),
        "passed": True,
        "records": records,
        "repeats": validator.EXPECTED_REPEATS,
        "request_count": validator.EXPECTED_REQUESTS,
        "runtime_profile": {
            "attention_backend": validator.EXPECTED_ATTENTION_BACKEND,
            "image_id": validator.EXPECTED_IMAGE_ID,
            "kv_cache_dtype": validator.EXPECTED_KV_CACHE_DTYPE,
        },
        "sampling": dict(
            mtp_benchmark.QWEN_SPEED_LAB_SAMPLING_PROFILES["aeon-greedy-medium"]
        ),
        "sampling_profile": "aeon-greedy-medium",
        "schema_valid": True,
        "schema_version": mtp_benchmark.PROBE_SCHEMA_VERSION,
        "semantic_valid": True,
        "server_version": {"version": "0.23.0"},
        "speculative_metric_delta": {
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"}': 100.0,
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"}': 70.0,
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="2"}': 40.0,
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"}': 210.0,
            'vllm:spec_decode_num_draft_tokens_total{engine="0"}': 300.0,
        },
        "started_completed_at": "2026-08-25T20:27:16+00:00",
        "successful_requests": validator.EXPECTED_REQUESTS,
        "suite_sha256": mtp_benchmark._suite_sha256(),
        "suite_version": mtp_benchmark.SUITE_VERSION,
    }


def _long_batch_report() -> dict:
    levels = []
    for concurrency, completion, wall in (
        (1, 100, 1.0),
        (2, 220, 1.0),
        (4, 480, 1.0),
        (8, 960, 1.0),
    ):
        levels.append(
            {
                "aggregate_decode_tps": completion / wall,
                "completion_tokens": completion,
                "concurrency": concurrency,
                "max_request_seconds": 0.9,
                "median_request_seconds": 0.8,
                "wall_seconds": wall,
            }
        )
    serial = levels[0]["aggregate_decode_tps"]
    concurrent = levels[-1]["aggregate_decode_tps"]
    return {
        "base_url": "http://127.0.0.1:18037",
        "batch": {
            "best_aggregate_decode_tps": concurrent,
            "best_concurrency": 8,
            "concurrency_8_aggregate_decode_tps": concurrent,
            "concurrency_8_scale_vs_serial": concurrent / serial,
            "levels": levels,
            "passed": True,
            "serial_aggregate_decode_tps": serial,
            "throughput_scale_vs_serial": concurrent / serial,
        },
        "created_at": "2026-08-25T20:28:51+00:00",
        "long_context": {
            "answer_tail": "AEON-128K-NEEDLE-7F3C91B2",
            "completion_tokens": 10,
            "contains_answer": True,
            "elapsed_seconds": 10.0,
            "exact_answer": True,
            "passed": True,
            "prompt_tokens_measured": 125_985,
            "prompt_tokens_reported": 125_985,
        },
        "model": validator.EXPECTED_MODEL,
        "passed": True,
        "schema_version": 1,
    }


def test_validator_recomputes_every_release_gate() -> None:
    mtp = validator.validate_mtp_report(_mtp_report())
    long_batch = validator.validate_long_batch_report(_long_batch_report())

    assert mtp["successful_requests"] == 15
    assert mtp["median_decode_tps"] == 240.0
    assert mtp["native_draft_tokens_delta"] == 300.0
    assert mtp["native_accepted_tokens_delta"] == 210.0
    assert long_batch["prompt_tokens"] == 125_985
    assert long_batch["concurrency_8_scale_vs_serial"] == 9.6
    assert long_batch["benchmark_script_provenance_bound"] is False


def test_future_long_report_binds_exact_benchmark_script() -> None:
    report = _long_batch_report()
    report["schema_version"] = 2
    report["benchmark_script_sha256"] = hashlib.sha256(
        Path(validator.long_batch_benchmark.__file__).read_bytes()
    ).hexdigest()

    result = validator.validate_long_batch_report(report)

    assert result["benchmark_script_provenance_bound"] is True
    assert result["benchmark_script_sha256"] == report["benchmark_script_sha256"]

    report["benchmark_script_sha256"] = "0" * 64
    with pytest.raises(
        validator.QwenReleaseEvidenceError, match="script identity changed"
    ):
        validator.validate_long_batch_report(report)


@pytest.mark.parametrize(
    "mutate,match",
    (
        (
            lambda report: report["records"][0].update(action_sha256="0" * 64),
            "action digest",
        ),
        (
            lambda report: report["records"][0].update(final_sha256="0" * 64),
            "nondeterministic",
        ),
        (
            lambda report: report["speculative_metric_delta"].update(
                {'vllm:spec_decode_num_draft_tokens_total{engine="0"}': 0.0}
            ),
            "positive and finite",
        ),
        (
            lambda report: report["speculative_metric_delta"].update(
                {'vllm:spec_decode_num_accepted_tokens_total{engine="0"}': 211.0}
            ),
            "disagrees with positions",
        ),
    ),
)
def test_mtp_validator_fails_closed_on_mutated_evidence(mutate, match) -> None:
    report = _mtp_report()
    mutate(report)

    with pytest.raises(validator.QwenReleaseEvidenceError, match=match):
        validator.validate_mtp_report(report)


@pytest.mark.parametrize(
    "mutate,match",
    (
        (
            lambda report: report["long_context"].update(
                prompt_tokens_measured=119_999
            ),
            "measured long-context tokens",
        ),
        (
            lambda report: report["long_context"].update(
                prompt_tokens_reported=125_984
            ),
            "tokens disagree",
        ),
        (
            lambda report: report["long_context"].update(answer_tail="wrong"),
            "exact long-context recall",
        ),
    ),
)
def test_long_validator_fails_closed_on_recall_mutations(mutate, match) -> None:
    report = _long_batch_report()
    mutate(report)

    with pytest.raises(validator.QwenReleaseEvidenceError, match=match):
        validator.validate_long_batch_report(report)


def test_long_validator_recomputes_concurrency_eight_from_levels() -> None:
    report = _long_batch_report()
    level = next(item for item in report["batch"]["levels"] if item["concurrency"] == 8)
    level["completion_tokens"] = 50
    level["aggregate_decode_tps"] = 50.0
    report["batch"].update(
        best_aggregate_decode_tps=480.0,
        best_concurrency=4,
        concurrency_8_aggregate_decode_tps=50.0,
        concurrency_8_scale_vs_serial=0.5,
        throughput_scale_vs_serial=4.8,
        passed=False,
    )
    report["passed"] = False

    with pytest.raises(validator.QwenReleaseEvidenceError, match="did not beat serial"):
        validator.validate_long_batch_report(report)


def test_file_loader_rejects_duplicate_fields(tmp_path: Path) -> None:
    mtp_path = tmp_path / "mtp.json"
    long_path = tmp_path / "long.json"
    mtp_path.write_text('{"passed":true,"passed":true}', encoding="utf-8")
    long_path.write_text(json.dumps(_long_batch_report()), encoding="utf-8")

    with pytest.raises(validator.QwenReleaseEvidenceError, match="duplicate JSON field"):
        validator.validate_release_evidence_files(mtp_path, long_path)
