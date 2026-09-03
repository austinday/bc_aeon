#!/usr/bin/env python3
"""Fail-closed, read-only validation for compact Qwen release reports.

This validator consumes the bounded JSON emitted by
``benchmark_qwen38_mtp`` and ``benchmark_qwen38_long_batch``.  It recomputes
the release-sensitive summaries from request-level data; a report's own
``passed`` flags are never sufficient evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from aeon.scripts import benchmark_qwen38_long_batch as long_batch_benchmark
from aeon.scripts import benchmark_qwen38_mtp as mtp_benchmark


SCHEMA_VERSION = "aeon-qwen38-release-evidence-validation-v1"
MAX_MTP_REPORT_BYTES = 256 * 1024
MAX_LONG_BATCH_REPORT_BYTES = 64 * 1024
MIN_MEDIAN_DECODE_TPS = 100.0
MIN_LONG_PROMPT_TOKENS = 120_000
EXPECTED_MTP_K = 3
EXPECTED_REPEATS = 3
EXPECTED_REQUESTS = len(mtp_benchmark.CASES) * EXPECTED_REPEATS
EXPECTED_MODEL = mtp_benchmark.ENTRY_NAME
EXPECTED_IMAGE_ID = (
    "sha256:d57400972ab0ae46baac64d4bfcc49cb136c07d8b0c50a76c7e2d81bd8a9fe47"
)
EXPECTED_ATTENTION_BACKEND = "TRITON_ATTN"
EXPECTED_KV_CACHE_DTYPE = "fp8_per_token_head"
LONG_CONTEXT_NEEDLE = "AEON-128K-NEEDLE-7F3C91B2"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POSITION_RE = re.compile(r'(?:^|,)position="([0-9]+)"(?:,|$)')
_MTP_REPORT_FIELDS = {
    "benchmark_script_sha256",
    "entry_name",
    "errors",
    "k",
    "median_decode_tps",
    "median_total_tps",
    "model",
    "p95_latency_seconds",
    "passed",
    "records",
    "repeats",
    "request_count",
    "runtime_profile",
    "sampling",
    "sampling_profile",
    "schema_valid",
    "schema_version",
    "semantic_valid",
    "server_version",
    "speculative_metric_delta",
    "started_completed_at",
    "successful_requests",
    "suite_sha256",
    "suite_version",
}
_MTP_RECORD_FIELDS = {
    "action_sha256",
    "case",
    "completion_tokens",
    "decode_seconds",
    "decode_tps",
    "elapsed_seconds",
    "final_sha256",
    "finish_reason",
    "prompt_tokens",
    "repeat",
    "response_sha256",
    "schema_valid",
    "semantic_valid",
    "total_tps",
    "ttft_seconds",
}
_LONG_BATCH_FIELDS = {
    "base_url",
    "batch",
    "created_at",
    "long_context",
    "model",
    "passed",
    "schema_version",
}
_LONG_BATCH_V2_FIELDS = _LONG_BATCH_FIELDS | {"benchmark_script_sha256"}
_LONG_FIELDS = {
    "answer_tail",
    "completion_tokens",
    "contains_answer",
    "elapsed_seconds",
    "exact_answer",
    "passed",
    "prompt_tokens_measured",
    "prompt_tokens_reported",
}
_BATCH_FIELDS = {
    "best_aggregate_decode_tps",
    "best_concurrency",
    "concurrency_8_aggregate_decode_tps",
    "concurrency_8_scale_vs_serial",
    "levels",
    "passed",
    "serial_aggregate_decode_tps",
    "throughput_scale_vs_serial",
}
_BATCH_LEVEL_FIELDS = {
    "aggregate_decode_tps",
    "completion_tokens",
    "concurrency",
    "max_request_seconds",
    "median_request_seconds",
    "wall_seconds",
}


class QwenReleaseEvidenceError(ValueError):
    """A persisted release report is incomplete or internally inconsistent."""


def _reject_constant(value: str) -> None:
    raise QwenReleaseEvidenceError(f"non-finite JSON number {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QwenReleaseEvidenceError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _load_bounded_json(path: Path, maximum: int) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise QwenReleaseEvidenceError(f"release report is unavailable: {path}") from exc
    if not payload or len(payload) > maximum:
        raise QwenReleaseEvidenceError(
            f"release report size is outside 1..{maximum} bytes: {path}"
        )
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QwenReleaseEvidenceError(f"release report is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise QwenReleaseEvidenceError(f"release report is not an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _exact_fields(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise QwenReleaseEvidenceError(f"{label} fields changed")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QwenReleaseEvidenceError(f"{label} is not an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QwenReleaseEvidenceError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise QwenReleaseEvidenceError(f"{label} is not {qualifier}")
    return result


def _same_float(recorded: Any, recomputed: float, label: str) -> None:
    value = _number(recorded, label)
    if not math.isclose(value, recomputed, rel_tol=1e-10, abs_tol=1e-10):
        raise QwenReleaseEvidenceError(f"{label} disagrees with request-level data")


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise QwenReleaseEvidenceError(f"{label} is not a lowercase SHA-256")
    return value


def _timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise QwenReleaseEvidenceError(f"{label} is not a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise QwenReleaseEvidenceError(f"{label} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise QwenReleaseEvidenceError(f"{label} has no timezone")


def _expected_action_sha256(case: dict[str, Any]) -> str:
    payload = {
        "intent": case["marker"],
        "actions": [
            {
                "tool_name": case["expected_tool"],
                "parameters": case["expected_parameters"],
            }
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _native_mtp_deltas(value: Any) -> tuple[float, float]:
    if not isinstance(value, dict):
        raise QwenReleaseEvidenceError("native MTP metric deltas are malformed")
    metrics = value
    if not metrics:
        raise QwenReleaseEvidenceError("native MTP metric deltas are absent")
    totals: dict[str, float] = {}
    per_position: dict[int, float] = {}
    allowed = {
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens",
        "vllm:spec_decode_num_accepted_tokens_per_pos",
    }
    for raw_key, raw_value in metrics.items():
        if not isinstance(raw_key, str):
            raise QwenReleaseEvidenceError("native MTP metric name is malformed")
        metric_name, _, labels = raw_key.partition("{")
        canonical = metric_name[:-6] if metric_name.endswith("_total") else metric_name
        if canonical not in allowed:
            raise QwenReleaseEvidenceError(f"unexpected native MTP metric {raw_key!r}")
        number = _number(raw_value, f"native MTP metric {raw_key}", positive=True)
        if canonical == "vllm:spec_decode_num_accepted_tokens_per_pos":
            if not labels.endswith("}"):
                raise QwenReleaseEvidenceError("MTP per-position labels are malformed")
            match = _POSITION_RE.search(labels[:-1])
            if match is None:
                raise QwenReleaseEvidenceError("MTP per-position metric has no position")
            position = int(match.group(1))
            if position in per_position:
                raise QwenReleaseEvidenceError("duplicate MTP accepted-token position")
            per_position[position] = number
        else:
            if canonical in totals:
                raise QwenReleaseEvidenceError(f"duplicate aggregate MTP metric {canonical}")
            totals[canonical] = number
    draft = totals.get("vllm:spec_decode_num_draft_tokens")
    accepted = totals.get("vllm:spec_decode_num_accepted_tokens")
    if draft is None or accepted is None:
        raise QwenReleaseEvidenceError("positive native MTP draft/accepted totals are required")
    if set(per_position) != set(range(EXPECTED_MTP_K)):
        raise QwenReleaseEvidenceError("native K=3 accepted-token positions are incomplete")
    if not math.isclose(sum(per_position.values()), accepted, rel_tol=1e-10, abs_tol=1e-10):
        raise QwenReleaseEvidenceError("native MTP accepted total disagrees with positions")
    if accepted > draft:
        raise QwenReleaseEvidenceError("native MTP accepted tokens exceed drafted tokens")
    return draft, accepted


def validate_mtp_report(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute the strict 15/15 K=3 semantic, speed, and native-MTP gates."""

    _exact_fields(report, _MTP_REPORT_FIELDS, "MTP report")
    script_hash = hashlib.sha256(Path(mtp_benchmark.__file__).read_bytes()).hexdigest()
    try:
        stats = mtp_benchmark._validated_probe_stats(
            report,
            expected_k=EXPECTED_MTP_K,
            expected_entry=EXPECTED_MODEL,
            script_hash=script_hash,
        )
    except (TypeError, ValueError) as exc:
        raise QwenReleaseEvidenceError(str(exc)) from exc
    if (
        stats["repeats"] != EXPECTED_REPEATS
        or stats["request_count"] != EXPECTED_REQUESTS
        or stats["successful_requests"] != EXPECTED_REQUESTS
        or not stats["passed"]
    ):
        raise QwenReleaseEvidenceError("MTP release gate requires exactly 15/15 requests")
    if report.get("model") != EXPECTED_MODEL:
        raise QwenReleaseEvidenceError("MTP served-model identity changed")
    if report.get("sampling_profile") != "aeon-greedy-medium" or report.get(
        "sampling"
    ) != mtp_benchmark.QWEN_SPEED_LAB_SAMPLING_PROFILES["aeon-greedy-medium"]:
        raise QwenReleaseEvidenceError("MTP deterministic sampling profile changed")
    if report.get("runtime_profile") != {
        "attention_backend": EXPECTED_ATTENTION_BACKEND,
        "kv_cache_dtype": EXPECTED_KV_CACHE_DTYPE,
        "image_id": EXPECTED_IMAGE_ID,
    }:
        raise QwenReleaseEvidenceError("MTP runtime profile changed")
    _timestamp(report.get("started_completed_at"), "MTP completion timestamp")

    cases = {case["name"]: case for case in mtp_benchmark.CASES}
    digests: dict[str, dict[str, str]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in cases}
    for raw_record in report["records"]:
        record = _exact_fields(raw_record, _MTP_RECORD_FIELDS, "MTP request record")
        case_name = record.get("case")
        if case_name not in cases:
            raise QwenReleaseEvidenceError("MTP request references an unknown case")
        grouped[case_name].append(record)
        _integer(record.get("repeat"), "MTP repeat", minimum=0)
        completion = _integer(
            record.get("completion_tokens"), "MTP completion tokens", minimum=1
        )
        _integer(record.get("prompt_tokens"), "MTP prompt tokens", minimum=1)
        elapsed = _number(record.get("elapsed_seconds"), "MTP elapsed", positive=True)
        ttft = _number(record.get("ttft_seconds"), "MTP TTFT")
        decode_seconds = _number(
            record.get("decode_seconds"), "MTP decode seconds", positive=True
        )
        if ttft < 0 or ttft > elapsed:
            raise QwenReleaseEvidenceError("MTP TTFT is outside the request duration")
        if not math.isclose(
            decode_seconds, elapsed - ttft, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise QwenReleaseEvidenceError("MTP decode duration is inconsistent")
        _same_float(record.get("decode_tps"), completion / decode_seconds, "MTP decode TPS")
        _same_float(record.get("total_tps"), completion / elapsed, "MTP total TPS")
        if record.get("finish_reason") != "stop":
            raise QwenReleaseEvidenceError("MTP request did not finish normally")
        for field in ("response_sha256", "final_sha256", "action_sha256"):
            _sha256(record.get(field), f"MTP {field}")

    for case_name, records in grouped.items():
        if sorted(record["repeat"] for record in records) != list(
            range(EXPECTED_REPEATS)
        ):
            raise QwenReleaseEvidenceError(f"MTP repeats are incomplete for {case_name}")
        expected_action = _expected_action_sha256(cases[case_name])
        if {record["action_sha256"] for record in records} != {expected_action}:
            raise QwenReleaseEvidenceError(
                f"MTP action digest is wrong or nondeterministic for {case_name}"
            )
        for field in ("response_sha256", "final_sha256"):
            values = {record[field] for record in records}
            if len(values) != 1:
                raise QwenReleaseEvidenceError(
                    f"MTP {field} is nondeterministic for {case_name}"
                )
        digests[case_name] = {
            "action_sha256": expected_action,
            "final_sha256": records[0]["final_sha256"],
            "response_sha256": records[0]["response_sha256"],
        }

    median_decode = _number(stats["median_decode_tps"], "MTP median decode TPS")
    if median_decode < MIN_MEDIAN_DECODE_TPS:
        raise QwenReleaseEvidenceError(
            f"MTP median decode TPS {median_decode:.6f} is below {MIN_MEDIAN_DECODE_TPS}"
        )
    drafted, accepted = _native_mtp_deltas(report.get("speculative_metric_delta"))
    return {
        "request_count": EXPECTED_REQUESTS,
        "successful_requests": EXPECTED_REQUESTS,
        "median_decode_tps": median_decode,
        "native_draft_tokens_delta": drafted,
        "native_accepted_tokens_delta": accepted,
        "deterministic_digests": digests,
        "benchmark_script_sha256": script_hash,
        "suite_sha256": report["suite_sha256"],
    }


def _validate_loopback_url(value: Any) -> None:
    if not isinstance(value, str):
        raise QwenReleaseEvidenceError("long/batch base URL is malformed")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise QwenReleaseEvidenceError("long/batch base URL port is malformed") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise QwenReleaseEvidenceError("long/batch evidence did not use loopback transport")


def validate_long_batch_report(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute exact >=120k recall and concurrency-eight throughput gates."""

    schema_version = report.get("schema_version")
    if schema_version == 1:
        _exact_fields(report, _LONG_BATCH_FIELDS, "long/batch report")
        script_provenance_bound = False
        script_sha256 = None
    elif schema_version == long_batch_benchmark.REPORT_SCHEMA_VERSION:
        _exact_fields(report, _LONG_BATCH_V2_FIELDS, "long/batch report")
        script_sha256 = hashlib.sha256(
            Path(long_batch_benchmark.__file__).read_bytes()
        ).hexdigest()
        if report.get("benchmark_script_sha256") != script_sha256:
            raise QwenReleaseEvidenceError(
                "long/batch benchmark script identity changed"
            )
        script_provenance_bound = True
    else:
        raise QwenReleaseEvidenceError("long/batch report schema is unsupported")
    if report.get("model") != EXPECTED_MODEL:
        raise QwenReleaseEvidenceError("long/batch report identity changed")
    _timestamp(report.get("created_at"), "long/batch creation timestamp")
    _validate_loopback_url(report.get("base_url"))

    long_context = _exact_fields(report.get("long_context"), _LONG_FIELDS, "long context")
    measured = _integer(
        long_context.get("prompt_tokens_measured"),
        "measured long-context tokens",
        minimum=MIN_LONG_PROMPT_TOKENS,
    )
    reported = _integer(
        long_context.get("prompt_tokens_reported"),
        "reported long-context tokens",
        minimum=MIN_LONG_PROMPT_TOKENS,
    )
    if measured != reported:
        raise QwenReleaseEvidenceError("measured and reported long-context tokens disagree")
    _integer(long_context.get("completion_tokens"), "long completion tokens", minimum=1)
    _number(long_context.get("elapsed_seconds"), "long elapsed seconds", positive=True)
    exact_recall = (
        long_context.get("exact_answer") is True
        and long_context.get("contains_answer") is True
        and long_context.get("answer_tail") == LONG_CONTEXT_NEEDLE
    )
    if not exact_recall or long_context.get("passed") is not True:
        raise QwenReleaseEvidenceError("exact long-context recall did not pass")

    batch = _exact_fields(report.get("batch"), _BATCH_FIELDS, "batch summary")
    levels = batch.get("levels")
    if not isinstance(levels, list) or len(levels) != 4:
        raise QwenReleaseEvidenceError("batch evidence must contain levels 1,2,4,8")
    by_concurrency: dict[int, dict[str, float | int]] = {}
    for raw_level in levels:
        level = _exact_fields(raw_level, _BATCH_LEVEL_FIELDS, "batch level")
        concurrency = _integer(level.get("concurrency"), "batch concurrency", minimum=1)
        if concurrency in by_concurrency:
            raise QwenReleaseEvidenceError("duplicate batch concurrency")
        completion = _integer(
            level.get("completion_tokens"), "batch completion tokens", minimum=1
        )
        wall = _number(level.get("wall_seconds"), "batch wall seconds", positive=True)
        median_request = _number(
            level.get("median_request_seconds"),
            "batch median request seconds",
            positive=True,
        )
        max_request = _number(
            level.get("max_request_seconds"), "batch max request seconds", positive=True
        )
        if median_request > max_request or max_request > wall:
            raise QwenReleaseEvidenceError("batch request timing is inconsistent")
        aggregate = completion / wall
        _same_float(
            level.get("aggregate_decode_tps"), aggregate, "batch aggregate decode TPS"
        )
        by_concurrency[concurrency] = {
            "aggregate_decode_tps": aggregate,
            "completion_tokens": completion,
        }
    if set(by_concurrency) != {1, 2, 4, 8}:
        raise QwenReleaseEvidenceError("batch evidence must contain levels 1,2,4,8 exactly")

    serial = float(by_concurrency[1]["aggregate_decode_tps"])
    concurrency_8 = float(by_concurrency[8]["aggregate_decode_tps"])
    best_concurrency, best_level = max(
        by_concurrency.items(), key=lambda item: float(item[1]["aggregate_decode_tps"])
    )
    best = float(best_level["aggregate_decode_tps"])
    _same_float(batch.get("serial_aggregate_decode_tps"), serial, "serial aggregate TPS")
    _same_float(
        batch.get("concurrency_8_aggregate_decode_tps"),
        concurrency_8,
        "concurrency-eight aggregate TPS",
    )
    _same_float(batch.get("best_aggregate_decode_tps"), best, "best aggregate TPS")
    _same_float(
        batch.get("throughput_scale_vs_serial"), best / serial, "best scale vs serial"
    )
    _same_float(
        batch.get("concurrency_8_scale_vs_serial"),
        concurrency_8 / serial,
        "concurrency-eight scale vs serial",
    )
    if batch.get("best_concurrency") != best_concurrency:
        raise QwenReleaseEvidenceError("best batch concurrency is internally inconsistent")
    batch_passed = concurrency_8 > serial
    if batch.get("passed") is not batch_passed or not batch_passed:
        raise QwenReleaseEvidenceError("concurrency eight did not beat serial throughput")
    if report.get("passed") is not (exact_recall and batch_passed):
        raise QwenReleaseEvidenceError("long/batch passed flag is internally inconsistent")
    return {
        "prompt_tokens": measured,
        "exact_recall": True,
        "serial_aggregate_decode_tps": serial,
        "concurrency_8_aggregate_decode_tps": concurrency_8,
        "concurrency_8_scale_vs_serial": concurrency_8 / serial,
        "benchmark_script_provenance_bound": script_provenance_bound,
        "benchmark_script_sha256": script_sha256,
    }


def validate_release_evidence_files(
    mtp_path: Path, long_batch_path: Path
) -> dict[str, Any]:
    """Load two bounded reports and return only independently validated facts."""

    mtp_report, mtp_sha256 = _load_bounded_json(mtp_path, MAX_MTP_REPORT_BYTES)
    long_batch_report, long_batch_sha256 = _load_bounded_json(
        long_batch_path, MAX_LONG_BATCH_REPORT_BYTES
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "mtp_report_sha256": mtp_sha256,
        "long_batch_report_sha256": long_batch_sha256,
        "mtp": validate_mtp_report(mtp_report),
        "long_batch": validate_long_batch_report(long_batch_report),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtp-report", type=Path, required=True)
    parser.add_argument("--long-batch-report", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_release_evidence_files(args.mtp_report, args.long_batch_report)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
