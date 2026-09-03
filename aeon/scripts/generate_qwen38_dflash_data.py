#!/usr/bin/env python3
"""Generate exact-target, exact-prefix chat rows for DFlash adaptation.

The generator talks only to an already-running, Fleet-managed Qwen service.  It
does not allocate compute or launch a model.  Rows are appended durably so an
interrupted run can resume without regenerating completed samples, while the
sidecar manifest records identities without printing prompts or model output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable

import requests

from aeon.core.qwen_speed_lab_adapter import _prompt_bundle
from aeon.core.sampling import QWEN_SPEED_LAB_SAMPLING_PROFILES


SCHEMA_VERSION = "aeon-qwen38-dflash-data-v1"
DEFAULT_MODEL = "Qwen3.8-27B-ARA-NVFP4-MTP"
EXPECTED_PREFIX_SHA256 = (
    "530efefa21e13c593cf4c9b5e8cbba807f093f94078b8366676f9acb89a9ae96"
)
SAMPLING_PROFILE = "aeon-greedy-medium"
_LOCAL_HTTP_KWARGS = {
    "allow_redirects": False,
    "proxies": {"http": "", "https": ""},
}


TASKS = (
    "Analyze a fault-tolerant distributed job queue in depth. Work through lease expiry, exactly-once effects, checkpoint recovery, backpressure, and split-brain prevention.",
    "Design a durable GPU workload broker. Explain admission control, resource leases, preemption, heartbeats, idempotent retries, and artifact settlement.",
    "Diagnose an asynchronous Python service whose requests occasionally hang after cancellation. Build a concrete evidence-driven debugging plan and likely fix.",
    "Review a FastAPI control plane for race conditions around concurrent job creation, cancellation, restart recovery, and status publication.",
    "Design a SQLite-backed queue that remains correct across crashes and duplicate submissions. Include schema invariants and transaction boundaries.",
    "Explain how to make an agent tool executor safe under retries, partial failures, process death, and stale state without losing useful work.",
    "Develop a test strategy for a distributed scheduler with adversarial timing, including property tests, fault injection, and deterministic replay.",
    "Analyze a prefix-cached language-model service. Separate prefill, decode, cache lookup, scheduler, and network latency and propose measurements for each.",
    "Design a content-addressed artifact store with atomic publication, integrity receipts, garbage collection, and protection against concurrent writers.",
    "Debug a Linux service that is healthy locally but intermittently unavailable through a reverse proxy. Lay out the highest-information checks in order.",
    "Architect a multi-project agent dashboard that presents durable status without interrupting active agents. Include event flow and failure recovery.",
    "Explain how speculative decoding remains lossless, what determines acceptance length, and how a target-specific draft should be evaluated.",
    "Compare block-parallel speculative decoding, autoregressive draft models, prompt lookup, and tree verification for a long-context coding agent.",
    "Propose a benchmark that accurately measures warm-prefix TTFT and single-stream decode throughput without accidentally timing compilation or cache misses.",
    "Investigate a CUDA inference workload that is memory-safe but slower than expected. Cover kernels, graph capture, quantization, synchronization, and CPU launch overhead.",
    "Design a safe rollout process for replacing a model-serving runtime while preserving identity checks, rollback, health probes, and active-user continuity.",
    "Refactor a large Python module into testable components while preserving behavior. Explain boundaries, migration order, and regression evidence.",
    "Find the root cause of duplicate background workers after service restarts. Reason about locking, process identity, parentage, and stale PID files.",
    "Design an append-only agent event log and materialized status view. Handle ordering, deduplication, snapshots, and schema evolution.",
    "Explain a rigorous approach to optimizing a hot Python and CUDA path using profiles, controlled experiments, and statistical benchmarks.",
    "Review a secret-storage feature for an agent platform. Cover encryption boundaries, browser behavior, redaction, access grants, and auditability.",
    "Plan a migration from three independent source trees into one workspace while preserving repository history, service paths, and compatibility links.",
    "Design a project status reporting system where agents emit structured milestones and users can inspect progress without prompting the agent.",
    "Diagnose a web dashboard whose summary cards resize as live metrics arrive. Explain the likely layout mechanics and a robust CSS solution.",
    "Create a detailed incident response for a worker that disappears during a checkpointed GPU job and may have been preempted by a higher-priority tenant.",
    "Explain how to verify that a quantized model artifact is the intended model and not merely a responding endpoint. Include cryptographic and semantic evidence.",
    "Design a constrained command adapter that can launch only reviewed profiles and cannot be turned into arbitrary remote command execution.",
    "Analyze host RAM, commit, shared memory, disk, and VRAM admission as one resource-allocation problem and propose truthful failure states.",
    "Write a technical plan for adapting a speculative draft to a slightly modified target model while guaranteeing the target model's output distribution remains unchanged.",
    "Compare FlashAttention, FlashInfer, Triton attention, CUDA graphs, and compiler fusion for batch-one long-context inference on Blackwell.",
    "Investigate why an apparently successful prefix cache has a low hit rate. Cover token identity, chat templates, block boundaries, eviction, and metrics.",
    "Design a benchmark receipt format that proves model, code, prompt, sampler, hardware, memory, quality, and performance identities without storing secrets.",
    "Reason through an idempotent deployment transaction that updates an origin, tunnel routing, and DNS while allowing exact rollback after ambiguous failures.",
    "Debug a tool-calling agent that sometimes emits malformed JSON after long reasoning. Separate model behavior, parser behavior, retries, and prompt state.",
    "Design context clearing for a persistent agent so transient conversation and memory are removed but system instructions and identity remain intact.",
    "Explain how to serve many project agents in a tabbed dashboard without allowing stale sessions, orphaned runtimes, or confusing implicit agent creation.",
    "Review a worker scratch-cleanup design. Require exact ownership proofs, bounded deletion, durable result retrieval, and protection of unfamiliar files.",
    "Develop a failure matrix for a streaming chat client, including connection failure, partial SSE frames, missing usage, server cancellation, and retries.",
    "Analyze a model server where throughput improves with longer generation but TTFT regresses. Identify scheduler and graph-capture tradeoffs.",
    "Propose a custom kernel optimization workflow for a recurrent linear-attention decode path, with correctness oracles and rollback gates.",
    "Explain how activation quantization and weight quantization interact in NVFP4 inference, including scale fusion, outliers, and accuracy validation.",
    "Design an on-policy dataset generator for speculative-draft training that is resumable, private, deterministic, and bound to an exact target identity.",
    "Compare selective fine-tuning of a draft projection and selector against full draft retraining after a low-KL target modification.",
    "Reason about the best sequence length and anchor sampling strategy for training a block-parallel drafter on a nine-thousand-token static system prefix.",
    "Design a semantic quality gate for an autonomous coding agent that catches unsafe actions, fabricated evidence, broken tool calls, and incomplete work.",
    "Diagnose a service that reports healthy while its pricing or allocation loop is silently failing. Explain why process existence is weak evidence.",
    "Architect a local credential setup UI that supports OAuth and tokens while preventing accidental browser password prompts and secret disclosure.",
    "Explain a safe way to preserve live production inference while running aggressive canary benchmarks on opportunistic fleet capacity.",
    "Develop a plan to quantify speculative-draft acceptance by token position and connect those measurements to block size and expected speedup.",
    "Review a CUDA graph capture configuration for a hybrid attention model and identify where full-graph assumptions can silently be invalid.",
    "Compare FP8 KV cache with BF16 KV cache for a long static prefix, including memory, bandwidth, scaling, and semantic validation.",
    "Design a reproducible experiment to decide whether a faster draft with lower acceptance beats a slower draft with higher acceptance.",
    "Analyze why a draft trained for the base model loses acceptance on a minimally modified uncensored target despite low output KL divergence.",
    "Propose an iterative optimization ladder for reaching two hundred tokens per second while keeping a single 48 GB GPU and exact target sampling.",
    "Explain how target hidden-state layer selection affects speculative-draft quality when only upper target layers were modified.",
    "Design a warm-start loader that verifies every draft tensor name, shape, dtype, and hash before adapting a published checkpoint.",
    "Plan a low-risk training smoke test that proves target load, dataset formatting, gradients, checkpointing, and memory bounds before a full run.",
    "Analyze whether tree-structured verification can exploit a top-k path selector better than verifying only one linear speculative path.",
    "Compare DFlash 2, EAGLE-style drafting, JetSpec, and dynamic speculative token counts for this exact single-user agent workload.",
    "Design a promotion gate that rejects a speed improvement unless it passes repeated throughput, p95 TTFT, memory, and semantic checks.",
    "Investigate how a 40 KB constant system message should be represented in training and inference so prefix caching and draft adaptation agree.",
    "Explain how to audit an inference engine patch for partial-rotary-position models before enabling fused QK normalization and RoPE.",
    "Develop a final technical status report for a long optimization effort, distinguishing measured wins, rejected hypotheses, and remaining risks.",
    "Design a deterministic end-to-end acceptance benchmark that compares a published speculative drafter with one adapted to an exact target delta.",
)

TASK_VARIANTS = (
    "",
    (
        "Assume a plausible first fix already failed. Identify the hidden assumptions, "
        "use evidence to isolate the cause, and propose the smallest robust correction "
        "with regression tests."
    ),
    (
        "Treat this as a live production incident. Give the observation order, safe "
        "commands or probes, decision points, rollback boundary, and durable follow-up."
    ),
    (
        "Turn the analysis into an implementation-ready design with interfaces, state "
        "transitions, invariants, pseudocode where useful, and adversarial tests."
    ),
    (
        "Compare at least three viable approaches, quantify their tradeoffs, select one, "
        "and define the evidence that would make you reverse that decision."
    ),
    (
        "Review a deliberately flawed implementation of this idea. Enumerate subtle "
        "failure modes, explain their mechanisms, then lay out a safe migration path."
    ),
    (
        "Produce an operator-facing status and decision report: current evidence, open "
        "uncertainties, highest-value next experiment, success gate, and stop conditions."
    ),
    (
        "Analyze this under adversarial timing, malformed inputs, partial failures, stale "
        "state, and resource exhaustion while preserving useful work and auditability."
    ),
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def _load_completed(path: Path, *, prefix_sha256: str) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise RuntimeError("existing dataset is not a regular file owned by this user")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise RuntimeError(f"dataset row {line_number} has no sample_id")
            if row.get("prefix_sha256") != prefix_sha256:
                raise RuntimeError(f"dataset row {line_number} belongs to another prefix")
            if sample_id in completed:
                raise RuntimeError(f"duplicate sample_id in dataset: {sample_id}")
            completed[sample_id] = row
    return completed


def _stream_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    sampling = QWEN_SPEED_LAB_SAMPLING_PROFILES[SAMPLING_PROFILE]
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
    response = requests.post(
        base_url.rstrip("/") + "/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=(30, 900),
        **_LOCAL_HTTP_KWARGS,
    )
    response.raise_for_status()
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
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
            if delta.get("reasoning_content") or delta.get("reasoning"):
                reasoning.append(str(delta.get("reasoning_content") or delta.get("reasoning")))
            if delta.get("content"):
                content.append(str(delta["content"]))
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
    result = {
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "usage": usage,
        "finish_reason": finish_reason,
    }
    completion_tokens = usage.get("completion_tokens")
    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens <= 0
        or not (result["content"] or result["reasoning_content"])
    ):
        raise RuntimeError("target returned no usable completion")
    return result


def _iter_tasks(count: int) -> Iterable[tuple[int, str]]:
    maximum = len(TASKS) * len(TASK_VARIANTS)
    if not 1 <= count <= maximum:
        raise ValueError(f"count must be within 1..{maximum}")
    for index in range(count):
        task = TASKS[index % len(TASKS)]
        variant = TASK_VARIANTS[index // len(TASKS)]
        yield index, (
            task
            + (" " + variant if variant else "")
            + " Reason carefully, stay concrete, and continue the technical analysis "
            + "until the response budget is nearly exhausted."
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output.parent.chmod(0o700)
    prefix_payload, prompt_sources = _prompt_bundle()
    prefix_sha256 = _sha256_bytes(prefix_payload)
    if prefix_sha256 != args.expected_prefix_sha256:
        raise RuntimeError(
            f"prompt identity changed: expected {args.expected_prefix_sha256}, got {prefix_sha256}"
        )
    prefix = prefix_payload.decode("utf-8")

    health = requests.get(
        args.base_url.rstrip("/") + "/health",
        timeout=10,
        **_LOCAL_HTTP_KWARGS,
    )
    health.raise_for_status()
    models = requests.get(
        args.base_url.rstrip("/") + "/v1/models",
        timeout=10,
        **_LOCAL_HTTP_KWARGS,
    )
    models.raise_for_status()
    model_ids = {
        item.get("id")
        for item in models.json().get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if args.model not in model_ids:
        raise RuntimeError("the exact requested target model is not served")

    completed = _load_completed(output, prefix_sha256=prefix_sha256)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for index, task in _iter_tasks(args.count):
        task_sha256 = _sha256_bytes(task.encode("utf-8"))
        sample_id = f"aeon-{index:03d}-{task_sha256[:16]}"
        if sample_id in completed:
            usage = completed[sample_id].get("generation", {}).get("usage", {})
            total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            continue
        messages = [
            {"role": "system", "content": prefix},
            {"role": "user", "content": task},
        ]
        generated = _stream_completion(
            base_url=args.base_url,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            seed=args.seed + index,
        )
        assistant = {
            "role": "assistant",
            "content": generated["content"],
            "reasoning_content": generated["reasoning_content"],
        }
        row = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "prefix_sha256": prefix_sha256,
            "task_sha256": task_sha256,
            "messages": [*messages, assistant],
            "generation": {
                "model": args.model,
                "sampling_profile": SAMPLING_PROFILE,
                "seed": args.seed + index,
                "max_tokens": args.max_tokens,
                "finish_reason": generated["finish_reason"],
                "usage": generated["usage"],
            },
        }
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        completed[sample_id] = row
        usage = generated["usage"]
        total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "path_basename": output.name,
            "sha256": _sha256_file(output),
            "bytes": output.stat().st_size,
            "rows": len(completed),
        },
        "target": {
            "base_url": args.base_url,
            "model": args.model,
            "sampling_profile": SAMPLING_PROFILE,
        },
        "prompt": {
            "sha256": prefix_sha256,
            "bytes": len(prefix_payload),
            "source_sha256": prompt_sources,
        },
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        },
        "generation": {
            "count": args.count,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    }
    _atomic_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8033")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=len(TASKS))
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42800)
    parser.add_argument("--expected-prefix-sha256", default=EXPECTED_PREFIX_SHA256)
    args = parser.parse_args()
    if not 256 <= args.max_tokens <= 2048:
        raise SystemExit("--max-tokens must be within 256..2048")
    manifest = run(args)
    print(
        json.dumps(
            {
                "dataset_sha256": manifest["dataset"]["sha256"],
                "rows": manifest["dataset"]["rows"],
                "prompt_sha256": manifest["prompt"]["sha256"],
                "completion_tokens": manifest["usage"]["completion_tokens"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
