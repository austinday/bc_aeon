#!/usr/bin/env python3
"""Release gates for Qwen 128k recall and continuous batch throughput."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


MODEL = "Qwen3.8-27B-ARA-NVFP4-MTP"
NEEDLE = "AEON-128K-NEEDLE-7F3C91B2"
FILLER = (
    "Archive record: amber cedar delta ember falcon granite harbor iris "
    "juniper kinetic lunar meadow. This line is irrelevant filler.\n"
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _post(base_url: str, route: str, payload: dict[str, Any], timeout: float):
    response = requests.post(
        base_url.rstrip("/") + route, json=payload, timeout=(15, timeout)
    )
    response.raise_for_status()
    return response.json()


def _token_count(base_url: str, messages: list[dict[str, str]]) -> int:
    value = _post(
        base_url,
        "/tokenize",
        {"model": MODEL, "messages": messages},
        120,
    )
    count = value.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RuntimeError("tokenizer returned no exact count")
    return count


def _long_messages(base_url: str, target_tokens: int) -> tuple[list[dict[str, str]], int]:
    prefix = (
        "Read the archive. One line contains a retrieval key. At the end, return "
        "only that exact key and no other text.\n"
    )
    suffix = "\nQuestion: What is the exact retrieval key? Return only the key."
    sample_messages = [{"role": "user", "content": prefix + FILLER * 1000 + suffix}]
    sample = _token_count(base_url, sample_messages)
    base = _token_count(base_url, [{"role": "user", "content": prefix + suffix}])
    per_block = max(1.0, (sample - base) / 1000.0)
    blocks = max(1, int((target_tokens - base) / per_block))
    for _ in range(8):
        lines = [FILLER] * blocks
        lines[(blocks * 3) // 4] = f"The exact retrieval key is {NEEDLE}.\n"
        messages = [{"role": "user", "content": prefix + "".join(lines) + suffix}]
        count = _token_count(base_url, messages)
        error = target_tokens - count
        if abs(error) <= 64:
            return messages, count
        blocks = max(1, blocks + int(error / per_block))
    return messages, count


def run_long(base_url: str, target_tokens: int) -> dict[str, Any]:
    messages, prompt_tokens = _long_messages(base_url, target_tokens)
    started = time.perf_counter()
    value = _post(
        base_url,
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 256,
        },
        900,
    )
    elapsed = time.perf_counter() - started
    choice = value["choices"][0]["message"]
    text = "\n".join(
        str(choice.get(key) or "") for key in ("reasoning_content", "content")
    ).strip()
    usage = value.get("usage") or {}
    exact = text.strip() == NEEDLE
    contains = NEEDLE in text
    return {
        "prompt_tokens_measured": prompt_tokens,
        "prompt_tokens_reported": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_seconds": elapsed,
        "exact_answer": exact,
        "contains_answer": contains,
        "answer_tail": text[-256:],
        "passed": prompt_tokens >= 120000 and contains,
    }


def _batch_request(base_url: str, request_id: int, max_tokens: int) -> dict[str, Any]:
    started = time.perf_counter()
    value = _post(
        base_url,
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Batch request {request_id}. Write a compact technical note "
                        "about deterministic distributed job queues."
                    ),
                }
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": max_tokens,
        },
        600,
    )
    elapsed = time.perf_counter() - started
    usage = value.get("usage") or {}
    completion = usage.get("completion_tokens")
    if isinstance(completion, bool) or not isinstance(completion, int) or completion <= 0:
        raise RuntimeError("batch request returned no completion-token count")
    return {"elapsed_seconds": elapsed, "completion_tokens": completion}


def run_batch(base_url: str, levels: list[int], max_tokens: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for concurrency in levels:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            requests_out = list(
                pool.map(
                    lambda item: _batch_request(base_url, item, max_tokens),
                    range(concurrency),
                )
            )
        wall = time.perf_counter() - started
        tokens = sum(item["completion_tokens"] for item in requests_out)
        results.append(
            {
                "concurrency": concurrency,
                "wall_seconds": wall,
                "completion_tokens": tokens,
                "aggregate_decode_tps": tokens / wall,
                "median_request_seconds": statistics.median(
                    item["elapsed_seconds"] for item in requests_out
                ),
                "max_request_seconds": max(
                    item["elapsed_seconds"] for item in requests_out
                ),
            }
        )
    serial = next(item for item in results if item["concurrency"] == 1)
    best = max(results, key=lambda item: item["aggregate_decode_tps"])
    return {
        "levels": results,
        "best_concurrency": best["concurrency"],
        "best_aggregate_decode_tps": best["aggregate_decode_tps"],
        "throughput_scale_vs_serial": (
            best["aggregate_decode_tps"] / serial["aggregate_decode_tps"]
        ),
        "passed": math.isfinite(best["aggregate_decode_tps"])
        and best["aggregate_decode_tps"] > serial["aggregate_decode_tps"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8033")
    parser.add_argument("--target-prompt-tokens", type=int, default=126000)
    parser.add_argument("--batch-levels", default="1,2,4,8")
    parser.add_argument("--batch-max-tokens", type=int, default=384)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    levels = [int(item) for item in args.batch_levels.split(",")]
    if not levels or 1 not in levels or any(item < 1 or item > 8 for item in levels):
        raise SystemExit("batch levels must include 1 and stay within 1..8")
    if not 120000 <= args.target_prompt_tokens <= 130000:
        raise SystemExit("target prompt tokens must stay within 120000..130000")
    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "base_url": args.base_url,
        "long_context": run_long(args.base_url, args.target_prompt_tokens),
        "batch": run_batch(args.base_url, levels, args.batch_max_tokens),
    }
    receipt["passed"] = bool(
        receipt["long_context"]["passed"] and receipt["batch"]["passed"]
    )
    _atomic_json(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
