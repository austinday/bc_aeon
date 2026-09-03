#!/usr/bin/env python3
"""Measure interactive Qwen3.8 decode speed and warm-prefix TTFT.

This is an experimental speed-lab benchmark, not a release selector.  It compares
only reviewed sampling profiles, records request-level evidence, and hashes the
unchanged prompt-source bundle instead of persisting that prompt in the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from aeon.core.sampling import QWEN_SPEED_LAB_SAMPLING_PROFILES


SCHEMA_VERSION = "aeon-qwen38-speed-lab-v3"
_LOCAL_HTTP_KWARGS = {
    "allow_redirects": False,
    "proxies": {"http": "", "https": ""},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = max(
        0,
        min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))),
    )
    return ordered[position]


def _metric_snapshot(base_url: str) -> dict[str, float]:
    result: dict[str, float] = {}
    try:
        response = requests.get(
            base_url.rstrip("/") + "/metrics",
            timeout=15,
            **_LOCAL_HTTP_KWARGS,
        )
        response.raise_for_status()
    except requests.RequestException:
        return result
    wanted_fragments = (
        "spec_decode_num_draft_tokens",
        "spec_decode_num_accepted_tokens",
        "spec_decode_num_accepted_tokens_per_pos",
        "prefix_cache",
    )
    for raw in response.text.splitlines():
        if not raw or raw.startswith("#") or " " not in raw:
            continue
        metric, raw_value = raw.rsplit(None, 1)
        if not any(fragment in metric for fragment in wanted_fragments):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            result[metric] = result.get(metric, 0.0) + value
    return result


def _metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        key: delta
        for key in sorted(set(before) | set(after))
        if (delta := after.get(key, 0.0) - before.get(key, 0.0)) > 0
    }


def _required_per_request_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise RuntimeError("final usage chunk omitted per-request metrics")
    result: dict[str, float] = {}
    bounds = {
        "time_to_first_token_ms": False,
        "generation_time_ms": True,
        "queue_time_ms": False,
        "mean_itl_ms": False,
        "tokens_per_second": True,
    }
    for name, strictly_positive in bounds.items():
        raw = value.get(name)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0
            or (strictly_positive and float(raw) <= 0)
        ):
            raise RuntimeError(f"per-request metric is missing or invalid: {name}")
        result[name] = float(raw)
    return result


def _stream_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    seed: int,
    sampling: dict[str, Any],
    require_per_request_metrics: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "reasoning_effort": sampling["reasoning_effort"],
        "chat_template_kwargs": {
            "enable_thinking": sampling["thinking"],
            "preserve_thinking": sampling["thinking"],
        },
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    reasoning_chars = 0
    content_chars = 0
    chunks_with_text = 0
    usage: dict[str, Any] = {}
    per_request_metrics: dict[str, float] | None = None
    finish_reason: str | None = None
    response = requests.post(
        base_url.rstrip("/") + "/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=(30, 900),
        **_LOCAL_HTTP_KWARGS,
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
            reasoning = str(
                delta.get("reasoning_content") or delta.get("reasoning") or ""
            )
            content = str(delta.get("content") or "")
            if reasoning or content:
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                last_token_at = now
                chunks_with_text += 1
                reasoning_chars += len(reasoning)
                content_chars += len(content)
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
            if isinstance(chunk.get("metrics"), dict):
                per_request_metrics = _required_per_request_metrics(chunk["metrics"])
    ended = time.perf_counter()
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens <= 0
        or first_token_at is None
    ):
        raise RuntimeError("stream returned no measured completion tokens")
    client_after_first_text_seconds = max(1e-9, ended - first_token_at)
    if require_per_request_metrics and per_request_metrics is None:
        raise RuntimeError("enabled canary returned incomplete per-request metrics")
    record = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": (
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if isinstance(usage.get("prompt_tokens_details"), dict)
            else None
        ),
        "client_elapsed_seconds": ended - started,
        "client_first_text_seconds": first_token_at - started,
        "client_final_chunk_after_first_text_seconds": client_after_first_text_seconds,
        "client_observed_completion_tps": (
            completion_tokens / client_after_first_text_seconds
        ),
        "chunks_with_text": chunks_with_text,
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "finish_reason": finish_reason,
        "client_last_text_chunk_offset_seconds": (
            None if last_token_at is None else last_token_at - started
        ),
    }
    if per_request_metrics is not None:
        record["per_request_metrics"] = per_request_metrics
        record["server_generation_tps_from_exact_time"] = (
            completion_tokens * 1000.0 / per_request_metrics["generation_time_ms"]
        )
    return record


def _decode_prompt(repeat: int) -> str:
    return (
        "Analyze a fault-tolerant distributed job queue in depth. Work through "
        "lease expiry, exactly-once effects, checkpoint recovery, backpressure, "
        "and split-brain prevention. Continue the technical reasoning until the "
        f"token budget ends. This is deterministic benchmark run {repeat}."
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    sampling = dict(QWEN_SPEED_LAB_SAMPLING_PROFILES[args.sampling_profile])
    prefix_path = args.system_prefix.resolve(strict=True)
    prefix = prefix_path.read_text(encoding="utf-8")
    if not 4096 <= len(prefix.encode("utf-8")) <= 2 * 1024 * 1024:
        raise RuntimeError("system-prefix fixture is outside its bounded size")
    base = args.base_url.rstrip("/")
    health = requests.get(base + "/health", timeout=30, **_LOCAL_HTTP_KWARGS)
    health.raise_for_status()
    version: dict[str, Any] = {}
    try:
        value = requests.get(
            base + "/version", timeout=15, **_LOCAL_HTTP_KWARGS
        ).json()
        if isinstance(value, dict):
            version = value
    except (requests.RequestException, ValueError):
        pass

    # Excluded warmup compiles all target/draft decode paths before timing.
    _stream_chat(
        base,
        args.model,
        [
            {"role": "system", "content": prefix},
            {"role": "user", "content": "Warm every decode path; answer briefly."},
        ],
        max_tokens=96,
        seed=9100,
        sampling=sampling,
        require_per_request_metrics=args.require_per_request_metrics,
    )
    before = _metric_snapshot(base)

    decode_records = []
    for repeat in range(args.repeats):
        decode_records.append(
            _stream_chat(
                base,
                args.model,
                [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": _decode_prompt(repeat)},
                ],
                max_tokens=args.max_tokens,
                seed=1701 + repeat,
                sampling=sampling,
                require_per_request_metrics=args.require_per_request_metrics,
            )
        )

    # The first short call establishes the full static system-prefix cache.  All
    # timed calls preserve that byte-identical prefix and vary only the tail.
    _stream_chat(
        base,
        args.model,
        [
            {"role": "system", "content": prefix},
            {"role": "user", "content": "Populate the static-prefix cache."},
        ],
        max_tokens=8,
        seed=7000,
        sampling=sampling,
        require_per_request_metrics=args.require_per_request_metrics,
    )
    prefix_records = []
    for repeat in range(args.repeats):
        prefix_records.append(
            _stream_chat(
                base,
                args.model,
                [
                    {"role": "system", "content": prefix},
                    {
                        "role": "user",
                        "content": f"Warm-prefix TTFT probe {repeat}; answer in one word.",
                    },
                ],
                max_tokens=8,
                seed=8100 + repeat,
                sampling=sampling,
                require_per_request_metrics=args.require_per_request_metrics,
            )
        )
    after = _metric_snapshot(base)

    client_tps_values = [
        float(item["client_observed_completion_tps"]) for item in decode_records
    ]
    client_ttft_values = [
        float(item["client_first_text_seconds"]) for item in prefix_records
    ]
    if args.require_per_request_metrics:
        server_generation_tps = [
            float(item["server_generation_tps_from_exact_time"])
            for item in decode_records
        ]
        server_inference_tps = [
            float(item["per_request_metrics"]["tokens_per_second"])
            for item in decode_records
        ]
        server_generation_ms = [
            float(item["per_request_metrics"]["generation_time_ms"])
            for item in decode_records
        ]
        server_ttft_seconds = [
            float(item["per_request_metrics"]["time_to_first_token_ms"]) / 1000.0
            for item in prefix_records
        ]
        server_queue_ms = [
            float(item["per_request_metrics"]["queue_time_ms"])
            for item in decode_records
        ]
        server_mean_itl_ms = [
            float(item["per_request_metrics"]["mean_itl_ms"])
            for item in decode_records
        ]
        qualification_tps = server_generation_tps
        qualification_ttft = server_ttft_seconds
        throughput_basis = "server_generation_time_ms"
        ttft_basis = "server_time_to_first_token_ms"
        target_tps = 150.0
    else:
        server_generation_tps = []
        server_inference_tps = []
        server_generation_ms = []
        server_queue_ms = []
        server_mean_itl_ms = []
        qualification_tps = client_tps_values
        qualification_ttft = client_ttft_values
        throughput_basis = "client_stream_observation"
        ttft_basis = "client_first_text_observation"
        target_tps = 200.0
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base,
        "model": args.model,
        "server_version": version,
        "sampling_profile": args.sampling_profile,
        "sampling": sampling,
        "prompt_fixture": {
            "sha256": _sha256(prefix_path),
            "bytes": len(prefix.encode("utf-8")),
        },
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "require_per_request_metrics": args.require_per_request_metrics,
        "decode_records": decode_records,
        "warm_prefix_records": prefix_records,
        "throughput_basis": throughput_basis,
        "ttft_basis": ttft_basis,
        "inference_target_tps": target_tps,
        "median_inference_tps": statistics.median(qualification_tps),
        "min_inference_tps": min(qualification_tps),
        "p95_inference_tps": _percentile(qualification_tps, 0.95),
        "median_client_observed_completion_tps": statistics.median(
            client_tps_values
        ),
        "median_warm_prefix_ttft_seconds": statistics.median(qualification_ttft),
        "p95_warm_prefix_ttft_seconds": _percentile(qualification_ttft, 0.95),
        "speculative_and_cache_metric_delta": _metric_delta(before, after),
    }
    if args.require_per_request_metrics:
        report.update(
            median_server_generation_time_ms=statistics.median(
                server_generation_ms
            ),
            median_server_generation_tps=statistics.median(server_generation_tps),
            median_server_inference_tps=statistics.median(server_inference_tps),
            median_server_queue_time_ms=statistics.median(server_queue_ms),
            median_server_mean_itl_ms=statistics.median(server_mean_itl_ms),
        )
    report["inference_target_met"] = report["median_inference_tps"] >= target_tps
    report["ttft_target_met"] = report["p95_warm_prefix_ttft_seconds"] <= 1.0
    report["benchmark_complete"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prefix", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--sampling-profile",
        choices=tuple(QWEN_SPEED_LAB_SAMPLING_PROFILES),
        default="aeon-greedy-medium",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-per-request-metrics", action="store_true")
    args = parser.parse_args()
    if not 3 <= args.repeats <= 9:
        raise SystemExit("--repeats must stay within 3..9")
    if not 256 <= args.max_tokens <= 2048:
        raise SystemExit("--max-tokens must stay within 256..2048")
    report = run(args)
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "median_inference_tps": report["median_inference_tps"],
                "p95_warm_prefix_ttft_seconds": report[
                    "p95_warm_prefix_ttft_seconds"
                ],
                "inference_target_met": report["inference_target_met"],
                "ttft_target_met": report["ttft_target_met"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
