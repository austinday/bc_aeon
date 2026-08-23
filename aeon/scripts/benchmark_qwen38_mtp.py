#!/usr/bin/env python3
"""Benchmark Qwen3.8 native MTP depths and build a fail-closed selection.

``probe`` targets one already-running vLLM server. ``select`` combines the five
release reports, requires semantically correct deterministic Aeon actions, and
emits the manifest consumed by Aeon's deployment planner/launcher.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from aeon.core.mtp_tuning import (
    MIN_RELEASE_REQUESTS_PER_K,
    MIN_SELECTED_DECODE_TPS,
    SCHEMA_VERSION,
    SELECTION_POLICY,
    expected_winner,
    sha256_file,
    validate_selection_manifest,
)
from aeon.core.action_schema import TURN_FIELDS_REQUIRED, build_turn_schema
from aeon.core.sampling import (
    QWEN_CONTROL_TEMPERATURE,
    QWEN_CONTROL_TOP_K,
    QWEN_CONTROL_TOP_P,
)


ENTRY_NAME = "Qwen3.8-27B-ARA-NVFP4-MTP"
PROBE_SCHEMA_VERSION = "aeon-qwen38-mtp-probe-v2"
SUITE_VERSION = "aeon-agent-mtp-suite-v5-deterministic-control"
RELEASE_DEPTHS = tuple(range(5))
MIN_RELEASE_REPEATS = 3
TOOL_NAMES = ("browser_interact", "open_file", "run_command", "task_complete")
RESPONSE_SCHEMA = build_turn_schema(list(TOOL_NAMES))
CASES = [
    {
        "name": "code_failure_replan",
        "expected_tool": "open_file",
        "expected_parameters": {"file_path": "/srv/app/session.py"},
        "marker": "AEON_CODE_OPEN",
        "prompt": (
            "A Python service fails only after a refactor. Evidence: the focused unit test says "
            "KeyError 'session_id', and the last changed file was /srv/app/session.py. The next "
            "grounded step is to inspect that exact file before editing. Return exactly one "
            "open_file action with file_path /srv/app/session.py and set intent exactly to "
            "AEON_CODE_OPEN. Keep every required metadata string concise."
        ),
    },
    {
        "name": "browser_grounding",
        "expected_tool": "browser_interact",
        "expected_parameters": {
            "action": "select", "element_id": 17, "value": "United States",
        },
        "marker": "AEON_BROWSER_SELECT",
        "vision": True,
        "prompt": (
            "A browser observation shows URL unchanged, FORM VALIDATION says Country is required, "
            "and the screenshot shows the closed Country control [17]. Return exactly one "
            "browser_interact action: action select, element_id 17, value United States. Set "
            "intent exactly to AEON_BROWSER_SELECT. Do not guess coordinates or submit the form."
        ),
    },
    {
        "name": "safe_system_diagnosis",
        "expected_tool": "run_command",
        "expected_parameters": {
            "command": "systemctl status aeon-web.service --no-pager",
            "timeout": 30,
        },
        "marker": "AEON_SYSTEM_STATUS",
        "prompt": (
            "A deployment health check fails while an unrelated production process is running. "
            "Inspect only the named service before considering a restart. Return exactly one "
            "run_command action with command 'systemctl status aeon-web.service --no-pager' and "
            "timeout 30. Set intent exactly to AEON_SYSTEM_STATUS. Never emit killall or deletion."
        ),
    },
    {
        "name": "verified_completion",
        "expected_tool": "task_complete",
        "expected_parameters": {"reason": "Verified completion marker AEON_DONE"},
        "marker": "AEON_TASK_COMPLETE",
        "prompt": (
            "The latest observation explicitly confirms all requested checks passed and the output "
            "artifact exists. Return exactly one task_complete action with reason exactly "
            "'Verified completion marker AEON_DONE'. Its parameters object must contain exactly "
            "that one reason key and no other keys. Set intent exactly to AEON_TASK_COMPLETE and "
            "keep every required metadata string concise."
        ),
    },
]


def _vision_probe_png() -> bytes:
    """Deterministic, legible browser-like evidence for the multimodal case."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (960, 540), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
    except OSError:
        font = small = ImageFont.load_default()
    draw.rounded_rectangle((100, 85, 860, 445), radius=20, fill=(246, 248, 251),
                           outline=(50, 70, 95), width=5)
    draw.text((155, 140), "FORM VALIDATION", font=font, fill=(25, 35, 50))
    draw.text((155, 245), "Country is required", font=small, fill=(180, 25, 35))
    draw.text((155, 325), "[17] Country (closed)", font=small, fill=(15, 90, 45))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _suite_sha256():
    payload = {
        "version": SUITE_VERSION,
        "schema": RESPONSE_SCHEMA,
        "cases": CASES,
        "vision_probe_sha256": hashlib.sha256(_vision_probe_png()).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _metric_snapshot(base_url):
    """Best-effort Prometheus counter snapshot, including per-position labels."""
    result = {}
    try:
        response = requests.get(base_url.rstrip("/") + "/metrics", timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return result
    wanted = {
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens",
        "vllm:spec_decode_num_accepted_tokens_per_pos",
    }
    for raw in response.text.splitlines():
        if not raw or raw.startswith("#") or " " not in raw:
            continue
        metric, value = raw.rsplit(None, 1)
        name = metric.split("{", 1)[0]
        # Prometheus counters are exported with ``_total`` by vLLM 0.23;
        # older releases used the unsuffixed spelling. Preserve the exact raw
        # key in the report while accepting either API spelling.
        canonical = name[:-6] if name.endswith("_total") else name
        if canonical not in wanted:
            continue
        try:
            numeric = float(value)
        except ValueError:
            continue
        result[metric] = result.get(metric, 0.0) + numeric
    return result


def _metric_delta(before, after):
    keys = set(before) | set(after)
    return {key: max(0.0, after.get(key, 0.0) - before.get(key, 0.0))
            for key in sorted(keys) if after.get(key, 0.0) - before.get(key, 0.0) > 0}


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _validated_probe_stats(report, *, expected_k, expected_entry, script_hash):
    """Recompute the security/correctness-sensitive fields of one probe report.

    Selection must not trust a hand-edited ``passed`` flag or throughput summary.
    The exact benchmark script is already bound by SHA-256; this additionally
    checks that its persisted request-level evidence is complete and internally
    consistent before that evidence can influence production serving.
    """
    if not isinstance(report, dict):
        raise ValueError(f"K={expected_k} report is not an object")
    if report.get("schema_version") != PROBE_SCHEMA_VERSION:
        raise ValueError(f"K={expected_k} uses an unsupported probe schema")
    if report.get("suite_version") != SUITE_VERSION or report.get("suite_sha256") != _suite_sha256():
        raise ValueError(f"K={expected_k} was not produced by the current benchmark suite")
    if report.get("benchmark_script_sha256") != script_hash:
        raise ValueError(f"K={expected_k} was not produced by this exact benchmark script")
    if report.get("entry_name") != expected_entry:
        raise ValueError(f"K={expected_k} is for a different catalog entry")
    if int(report.get("k", -1)) != expected_k:
        raise ValueError(f"report identity does not match K={expected_k}")
    if not isinstance(report.get("model"), str) or not report["model"].strip():
        raise ValueError(f"K={expected_k} has no served-model identity")

    try:
        repeats = int(report.get("repeats"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"K={expected_k} has an invalid repeat count") from exc
    if repeats < MIN_RELEASE_REPEATS:
        raise ValueError(
            f"K={expected_k} has {repeats} repeats; release selection requires "
            f"at least {MIN_RELEASE_REPEATS}")
    expected_keys = {
        (case["name"], repeat) for repeat in range(repeats) for case in CASES
    }
    if int(report.get("request_count", -1)) != len(expected_keys):
        raise ValueError(
            f"K={expected_k} did not declare all {len(expected_keys)} required requests")
    records = report.get("records")
    errors = report.get("errors")
    if not isinstance(records, list) or not isinstance(errors, list):
        raise ValueError(f"K={expected_k} records/errors are malformed")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"K={expected_k} contains a non-object request record")
        key = (record.get("case"), record.get("repeat"))
        if key not in expected_keys or key in seen:
            raise ValueError(f"K={expected_k} has an invalid or duplicate request key {key!r}")
        seen.add(key)
        if record.get("schema_valid") is not True:
            raise ValueError(f"K={expected_k} persisted a successful record that failed its schema gate")
        if record.get("semantic_valid") is not True:
            raise ValueError(f"K={expected_k} persisted a successful record that failed its semantic gate")
        if (not _is_sha256(record.get("response_sha256"))
                or not _is_sha256(record.get("final_sha256"))
                or not _is_sha256(record.get("action_sha256"))):
            raise ValueError(f"K={expected_k} request {key!r} has an invalid output digest")
        for field in ("decode_tps", "total_tps", "elapsed_seconds"):
            try:
                value = float(record.get(field))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"K={expected_k} request {key!r} has invalid {field}") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"K={expected_k} request {key!r} has non-positive {field}")

    if int(report.get("successful_requests", -1)) != len(records):
        raise ValueError(f"K={expected_k} successful_requests disagrees with its records")
    recomputed_schema_valid = seen == expected_keys
    recomputed_semantic_valid = (
        recomputed_schema_valid and all(record["semantic_valid"] for record in records)
    )
    if report.get("schema_valid") is not recomputed_schema_valid:
        raise ValueError(f"K={expected_k} schema_valid is internally inconsistent")
    if report.get("semantic_valid") is not recomputed_semantic_valid:
        raise ValueError(f"K={expected_k} semantic_valid is internally inconsistent")
    recomputed_passed = recomputed_schema_valid and recomputed_semantic_valid and not errors
    if report.get("passed") is not recomputed_passed:
        raise ValueError(f"K={expected_k} passed is internally inconsistent")

    def recompute(field):
        values = [float(record[field]) for record in records]
        return statistics.median(values) if values else 0.0

    median_decode = recompute("decode_tps")
    median_total = recompute("total_tps")
    p95_latency = _percentile([float(record["elapsed_seconds"]) for record in records], 0.95)
    for field, expected in (("median_decode_tps", median_decode),
                            ("median_total_tps", median_total),
                            ("p95_latency_seconds", p95_latency)):
        try:
            recorded = float(report.get(field))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"K={expected_k} has invalid {field}") from exc
        if not math.isfinite(recorded) or not math.isclose(
                recorded, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"K={expected_k} {field} disagrees with request records")
    return {
        "passed": recomputed_passed,
        "schema_valid": recomputed_schema_valid,
        "semantic_valid": recomputed_semantic_valid,
        "repeats": repeats,
        "successful_requests": len(records),
        "request_count": len(expected_keys),
        "median_decode_tps": median_decode,
        "median_total_tps": median_total,
        "p95_latency_seconds": p95_latency,
    }


def _stream_request(base_url, model, case, repeat):
    user_content = case["prompt"]
    if case.get("vision"):
        encoded = base64.b64encode(_vision_probe_png()).decode("ascii")
        user_content = [
            {"type": "text", "text": case["prompt"]},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + encoded,
            }},
        ]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "You are Aeon's local evidence reasoner. Analyze privately, then return only the "
                "schema-constrained final object. Do not invent observations."
            )},
            {"role": "user", "content": user_content},
        ],
        # Import the exact control-plane profile used by LLMClient so the release
        # gate can never silently benchmark a faster/different sampler.
        "temperature": QWEN_CONTROL_TEMPERATURE,
        "top_p": QWEN_CONTROL_TOP_P,
        "top_k": QWEN_CONTROL_TOP_K,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "reasoning_effort": "medium",
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "aeon_mtp_benchmark", "strict": True,
                            "schema": RESPONSE_SCHEMA},
        },
        "seed": 1701 + repeat,
        "max_tokens": 4096,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first = None
    content_parts, reasoning_parts = [], []
    usage = {}
    finish_reason = None
    response = requests.post(
        base_url.rstrip("/") + "/v1/chat/completions",
        json=payload, stream=True, timeout=(30, 300),
    )
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        chunk = json.loads(body)
        choices = chunk.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            content = delta.get("content") or ""
            if (reasoning or content) and first is None:
                first = time.perf_counter()
            if reasoning:
                reasoning_parts.append(str(reasoning))
            if content:
                content_parts.append(str(content))
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        if chunk.get("usage"):
            usage = chunk["usage"]
    end = time.perf_counter()
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    if not content.strip():
        raise ValueError(
            "server returned no post-reasoning content "
            f"(reasoning_chars={len(reasoning)}, finish_reason={finish_reason!r}); "
            "structured decoding must begin only after </think>"
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid post-reasoning JSON for {case['name']}: "
            f"content_chars={len(content)}, reasoning_chars={len(reasoning)}, "
            f"finish_reason={finish_reason!r}, prefix={content[:240]!r}"
        ) from exc
    allowed_fields = set(RESPONSE_SCHEMA["properties"])
    required_fields = set(TURN_FIELDS_REQUIRED)
    metadata_fields = required_fields - {"actions"}
    schema_valid = (
        isinstance(parsed, dict)
        and required_fields <= set(parsed) <= allowed_fields
        and all(isinstance(parsed.get(field), str) for field in metadata_fields)
        and ("updated_plan" not in parsed or isinstance(parsed["updated_plan"], str))
        and isinstance(parsed.get("actions"), list)
        and len(parsed["actions"]) == 1
        and isinstance(parsed["actions"][0], dict)
        and set(parsed["actions"][0]) == {"tool_name", "parameters"}
        and parsed["actions"][0].get("tool_name") in TOOL_NAMES
        and isinstance(parsed["actions"][0].get("parameters"), dict)
    )
    if not schema_valid:
        raise ValueError(f"turn-schema gate failed for {case['name']}: {content[:500]}")
    action = parsed["actions"][0]
    semantic_valid = (
        parsed["intent"] == case["marker"]
        and action["tool_name"] == case["expected_tool"]
        and action["parameters"] == case["expected_parameters"]
    )
    if not semantic_valid:
        raise ValueError(f"agent-action gate failed for {case['name']}: {content[:500]}")
    elapsed = end - start
    ttft = (first - start) if first is not None else elapsed
    decode_seconds = max(1e-9, end - (first if first is not None else start))
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if completion_tokens <= 0:
        raise ValueError("server did not return completion token usage")
    combined = reasoning + "\n<FINAL>\n" + content
    action_contract = {
        "intent": parsed["intent"],
        "actions": parsed["actions"],
    }
    action_bytes = json.dumps(
        action_contract, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "case": case["name"],
        "repeat": repeat,
        "schema_valid": True,
        "semantic_valid": True,
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "elapsed_seconds": elapsed,
        "ttft_seconds": ttft,
        "decode_seconds": decode_seconds,
        "decode_tps": completion_tokens / decode_seconds,
        "total_tps": completion_tokens / max(elapsed, 1e-9),
        "response_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        "final_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "action_sha256": hashlib.sha256(action_bytes).hexdigest(),
    }


def run_probe(args):
    base = args.base_url.rstrip("/")
    health = requests.get(base + "/health", timeout=15)
    health.raise_for_status()
    version = {}
    try:
        version = requests.get(base + "/version", timeout=15).json()
    except (requests.RequestException, ValueError):
        pass

    # One excluded request warms tokenizer, grammar, attention, and MTP kernels on
    # every fresh K server before timing begins.  A semantically malformed warmup
    # is correctness evidence for this K, but it must not prevent later K values
    # from being measured.  Transport/server failures still propagate so the
    # runner does not mistake a broken endpoint for a completed benchmark.
    errors = []
    try:
        _stream_request(base, args.model, CASES[0], 999)
    except ValueError as exc:
        errors.append(f"excluded warmup: {type(exc).__name__}: {exc}")
    metrics_before = _metric_snapshot(base)
    records = []
    for repeat in range(args.repeats):
        for case in CASES:
            try:
                records.append(_stream_request(base, args.model, case, repeat))
            except Exception as exc:
                errors.append(f"{case['name']} repeat {repeat}: {type(exc).__name__}: {exc}")
    metrics_after = _metric_snapshot(base)
    decode = [item["decode_tps"] for item in records]
    total = [item["total_tps"] for item in records]
    latencies = [item["elapsed_seconds"] for item in records]
    expected_count = args.repeats * len(CASES)
    report = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "suite_sha256": _suite_sha256(),
        "benchmark_script_sha256": sha256_file(Path(__file__)),
        "entry_name": args.entry_name,
        "model": args.model,
        "k": args.k,
        "repeats": args.repeats,
        "runtime_profile": {
            "attention_backend": getattr(args, "attention_backend", "TRITON_ATTN"),
            "kv_cache_dtype": getattr(args, "kv_cache_dtype", "fp8_per_token_head"),
            "image_id": args.runtime_image_id,
        },
        "started_completed_at": _utc_now(),
        "server_version": version,
        "request_count": expected_count,
        "successful_requests": len(records),
        "schema_valid": len(records) == expected_count and all(r["schema_valid"] for r in records),
        "semantic_valid": (
            len(records) == expected_count and all(r["semantic_valid"] for r in records)
        ),
        "passed": (
            len(records) == expected_count
            and all(r["schema_valid"] and r["semantic_valid"] for r in records)
            and not errors
        ),
        "errors": errors,
        "median_decode_tps": statistics.median(decode) if decode else 0.0,
        "median_total_tps": statistics.median(total) if total else 0.0,
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "records": records,
        "speculative_metric_delta": _metric_delta(metrics_before, metrics_after),
    }
    _atomic_json(Path(args.output), report)
    print(json.dumps({key: report[key] for key in (
        "k", "passed", "successful_requests", "median_decode_tps",
        "median_total_tps", "p95_latency_seconds")}, sort_keys=True))
    return 0 if report["passed"] else 1


def run_select(args):
    reports = []
    report_paths = {}
    for path in args.reports:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        reports.append(report)
        report_paths[int(report["k"])] = path
    by_k = {int(report["k"]): report for report in reports}
    release_depths = set(RELEASE_DEPTHS)
    if set(by_k) != release_depths:
        raise ValueError(
            f"reports must contain K={','.join(map(str, RELEASE_DEPTHS))} exactly; "
            f"got {sorted(by_k)}")
    script_hash = sha256_file(Path(__file__))
    validated_stats = {
        key: _validated_probe_stats(
            by_k[key], expected_k=key, expected_entry=args.entry_name,
            script_hash=script_hash)
        for key in RELEASE_DEPTHS
    }
    suite_hashes = {report.get("suite_sha256") for report in reports}
    entries = {report.get("entry_name") for report in reports}
    models = {report.get("model") for report in reports}
    runtime_profiles = {
        (
            (report.get("runtime_profile") or {}).get("attention_backend"),
            (report.get("runtime_profile") or {}).get("kv_cache_dtype"),
            (report.get("runtime_profile") or {}).get("image_id"),
        )
        for report in reports
    }
    if (suite_hashes != {_suite_sha256()} or entries != {args.entry_name}
            or len(models) != 1 or len(runtime_profiles) != 1):
        raise ValueError("probe reports do not bind the same suite/entry/model/runtime")
    runtime_profile = next(iter(runtime_profiles))
    expected_runtime_profile = (
        args.attention_backend, args.kv_cache_dtype, args.image_id,
    )
    if runtime_profile != expected_runtime_profile:
        raise ValueError(
            "probe runtime profile does not match the requested selection profile: "
            f"measured={runtime_profile!r}, requested={expected_runtime_profile!r}")

    candidates = []
    for key in RELEASE_DEPTHS:
        report = by_k[key]
        stats = validated_stats[key]
        action_hashes_by_case = {
            case["name"]: [
                record["action_sha256"]
                for record in report.get("records") or []
                if record["case"] == case["name"]
            ]
            for case in CASES
        }
        deterministic = all(
            len(hashes) == stats["repeats"] and len(set(hashes)) == 1
            for hashes in action_hashes_by_case.values()
        )
        semantic_equivalent = stats["semantic_valid"]
        probe_passed = stats["passed"]
        passed = probe_passed and semantic_equivalent and deterministic
        reasons = []
        if not probe_passed:
            reasons.append("probe did not complete all structured/correct requests")
        if not semantic_equivalent:
            reasons.append("one or more responses chose the wrong Aeon action")
        if not deterministic:
            reasons.append("tool/argument decisions varied across production-sampler seeds")
        candidates.append({
            "k": key,
            "passed": passed,
            "probe_passed": probe_passed,
            "disqualified_reason": "; ".join(reasons),
            "schema_valid": stats["schema_valid"],
            "semantic_equivalent": semantic_equivalent,
            "deterministic": deterministic,
            "request_count": stats["request_count"],
            "successful_requests": stats["successful_requests"],
            "median_decode_tps": stats["median_decode_tps"],
            "median_total_tps": stats["median_total_tps"],
            "p95_latency_seconds": stats["p95_latency_seconds"],
            "speculative_metric_delta": report.get("speculative_metric_delta") or {},
            "report_sha256": sha256_file(report_paths[key]),
        })
    if not candidates[0]["passed"]:
        raise ValueError("non-speculative K=0 baseline failed; no selection is trustworthy")
    selected = expected_winner({item["k"]: item for item in candidates})

    model_dir = Path(args.model_dir)
    build_manifest = model_dir / "BUILD_MANIFEST.json"
    sha_sums = model_dir / "SHA256SUMS"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "complete": True,
        "completed_at": _utc_now(),
        "entry_name": args.entry_name,
        "served_model": next(iter(models)),
        "suite_version": SUITE_VERSION,
        "suite_sha256": next(iter(suite_hashes)),
        "selection_policy": SELECTION_POLICY,
        "selected_k": selected,
        "release_gate": {
            "minimum_requests_per_k": MIN_RELEASE_REQUESTS_PER_K,
            "minimum_selected_decode_tps": MIN_SELECTED_DECODE_TPS,
        },
        "artifact": {
            "path_name": model_dir.name,
            "build_manifest_sha256": sha256_file(build_manifest),
            "sha256s_sha256": sha256_file(sha_sums),
        },
        "runtime": {
            "image_ref": args.image_ref,
            "image_id": args.image_id,
            "attention_backend": args.attention_backend,
            "kv_cache_dtype": args.kv_cache_dtype,
            "server_versions": {
                str(k): by_k[k].get("server_version") or {} for k in RELEASE_DEPTHS
            },
        },
        "benchmark_script_sha256": script_hash,
        "candidates": candidates,
    }
    validate_selection_manifest(
        manifest, expected_entry=args.entry_name,
        expected_model_build_sha256=manifest["artifact"]["build_manifest_sha256"],
        expected_sha256s_sha256=manifest["artifact"]["sha256s_sha256"],
        expected_image_id=args.image_id,
        expected_attention_backend=args.attention_backend,
        expected_kv_cache_dtype=args.kv_cache_dtype,
    )
    _atomic_json(Path(args.output), manifest)
    print(json.dumps({"selected_k": selected,
                      "median_decode_tps": {str(c['k']): c['median_decode_tps'] for c in candidates}},
                     sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    probe = subs.add_parser("probe")
    probe.add_argument("--base-url", required=True, help="vLLM root, e.g. http://127.0.0.1:18033")
    probe.add_argument("--model", required=True)
    probe.add_argument("--entry-name", default=ENTRY_NAME)
    # K=0..4 are the release-selection matrix. K=5..6 remain available for
    # exploratory sweeps; both regressed on the measured Blackwell profile.
    probe.add_argument("--k", type=int, choices=range(7), required=True)
    probe.add_argument("--repeats", type=int, default=2)
    probe.add_argument("--attention-backend", default="TRITON_ATTN")
    probe.add_argument("--kv-cache-dtype", default="fp8_per_token_head")
    probe.add_argument("--runtime-image-id", required=True)
    probe.add_argument("--output", required=True)
    probe.set_defaults(func=run_probe)

    select = subs.add_parser("select")
    select.add_argument("--reports", nargs=5, required=True)
    select.add_argument("--model-dir", required=True)
    select.add_argument("--entry-name", default=ENTRY_NAME)
    select.add_argument("--image-ref", default="aeon_vllm:latest")
    select.add_argument("--image-id", required=True)
    select.add_argument("--attention-backend", default="TRITON_ATTN")
    select.add_argument("--kv-cache-dtype", default="fp8_per_token_head")
    select.add_argument("--output", required=True)
    select.set_defaults(func=run_select)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "repeats", 2) < 2:
        raise SystemExit("--repeats must be >=2")
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MTP_BENCHMARK_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
