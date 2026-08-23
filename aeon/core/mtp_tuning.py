"""Validation helpers for Qwen3.8 MTP draft-depth selection.

The catalog's MTP depth is a serving decision, not a model capability claim. A
selection is trusted only when K=0..4 were run against the exact model build and
runtime profile. Candidates must produce the intended Aeon tool call under the
real turn schema, remain deterministic across repeated requests, and complete a
large enough sample. The selected candidate must also clear Aeon's measured
single-stream decode-throughput floor.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Optional


SCHEMA_VERSION = "aeon-qwen38-mtp-selection-v2"
SELECTION_POLICY = "max_median_decode_tps_among_semantic_deterministic_prefer_lower_k_within_1pct"
MAX_RELEASE_K = 4
MIN_RELEASE_REQUESTS_PER_K = 12
MIN_SELECTED_DECODE_TPS = 100.0


class MtpSelectionError(ValueError):
    """A tuning artifact is missing, stale, internally inconsistent, or failed."""


def sha256_file(path) -> str:
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_map(data: Dict) -> Dict[int, Dict]:
    result = {}
    for item in data.get("candidates") or []:
        if not isinstance(item, dict):
            raise MtpSelectionError("candidate entry is not an object")
        try:
            key = int(item.get("k"))
        except (TypeError, ValueError) as exc:
            raise MtpSelectionError("candidate has invalid k") from exc
        if key in result:
            raise MtpSelectionError(f"duplicate K={key} candidate")
        result[key] = item
    return result


def expected_winner(candidates: Dict[int, Dict], tolerance: float = 0.01) -> int:
    """Recompute the policy winner among passed semantic candidates."""
    eligible = {
        key: item for key, item in candidates.items()
        if item.get("passed") is True
    }
    if not eligible:
        raise MtpSelectionError("no semantic/deterministic benchmark candidate passed")
    scores = {}
    for key, item in eligible.items():
        try:
            score = float(item.get("median_decode_tps"))
        except (TypeError, ValueError) as exc:
            raise MtpSelectionError(f"K={key} has no numeric median_decode_tps") from exc
        if not math.isfinite(score) or score <= 0:
            raise MtpSelectionError(f"K={key} has non-positive median_decode_tps")
        scores[key] = score
    best_score = max(scores.values())
    near_best = [key for key, score in scores.items()
                 if score >= best_score * (1.0 - tolerance)]
    return min(near_best)


def validate_selection_manifest(data: Dict, *, expected_entry: str,
                                expected_model_build_sha256: Optional[str] = None,
                                expected_sha256s_sha256: Optional[str] = None,
                                expected_image_id: Optional[str] = None,
                                expected_attention_backend: Optional[str] = None,
                                expected_kv_cache_dtype: Optional[str] = None,
                                max_k: int = MAX_RELEASE_K) -> int:
    """Validate and return selected K, raising ``MtpSelectionError`` on doubt."""
    if not isinstance(data, dict):
        raise MtpSelectionError("selection manifest is not an object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise MtpSelectionError("unsupported MTP selection schema")
    if data.get("status") != "validated" or data.get("complete") is not True:
        raise MtpSelectionError("MTP selection did not complete validation")
    if data.get("entry_name") != expected_entry:
        raise MtpSelectionError("MTP selection is for a different catalog entry")
    if data.get("selection_policy") != SELECTION_POLICY:
        raise MtpSelectionError("unexpected MTP selection policy")

    artifact = data.get("artifact") or {}
    if expected_model_build_sha256 and (
            artifact.get("build_manifest_sha256") != expected_model_build_sha256):
        raise MtpSelectionError("MTP selection is stale for this model BUILD_MANIFEST")
    if expected_sha256s_sha256 and (
            artifact.get("sha256s_sha256") != expected_sha256s_sha256):
        raise MtpSelectionError("MTP selection is stale for this model SHA256SUMS")
    runtime = data.get("runtime") or {}
    if expected_image_id and runtime.get("image_id") != expected_image_id:
        raise MtpSelectionError("MTP selection was measured with a different runtime image")
    attention_backend = runtime.get("attention_backend")
    kv_cache_dtype = runtime.get("kv_cache_dtype")
    if not isinstance(attention_backend, str) or not attention_backend:
        raise MtpSelectionError("MTP selection has no attention-backend identity")
    if not isinstance(kv_cache_dtype, str) or not kv_cache_dtype:
        raise MtpSelectionError("MTP selection has no KV-cache dtype identity")
    if (expected_attention_backend is not None
            and attention_backend != expected_attention_backend):
        raise MtpSelectionError("MTP selection used a different attention backend")
    if (expected_kv_cache_dtype is not None
            and kv_cache_dtype != expected_kv_cache_dtype):
        raise MtpSelectionError("MTP selection used a different KV-cache dtype")

    release_gate = data.get("release_gate") or {}
    if release_gate.get("minimum_requests_per_k") != MIN_RELEASE_REQUESTS_PER_K:
        raise MtpSelectionError("unexpected MTP request-count release gate")
    if release_gate.get("minimum_selected_decode_tps") != MIN_SELECTED_DECODE_TPS:
        raise MtpSelectionError("unexpected MTP throughput release gate")

    candidates = _candidate_map(data)
    expected_keys = set(range(max_k + 1))
    if set(candidates) != expected_keys:
        raise MtpSelectionError(
            f"expected benchmark candidates {sorted(expected_keys)}, got {sorted(candidates)}")
    for key, item in candidates.items():
        if not isinstance(item.get("passed"), bool):
            raise MtpSelectionError(f"K={key} has no boolean pass/disqualification result")
        for field in ("probe_passed", "schema_valid", "semantic_equivalent",
                      "deterministic"):
            if not isinstance(item.get(field), bool):
                raise MtpSelectionError(f"K={key} has no boolean {field} result")
        try:
            requests_ok = int(item.get("successful_requests"))
            requests_total = int(item.get("request_count"))
        except (TypeError, ValueError) as exc:
            raise MtpSelectionError(f"K={key} request counts are invalid") from exc
        if (requests_total < MIN_RELEASE_REQUESTS_PER_K or requests_ok < 0
                or requests_ok > requests_total):
            raise MtpSelectionError(f"K={key} request counts are not credible")
        expected_pass = (
            item["probe_passed"]
            and item["schema_valid"]
            and item["semantic_equivalent"]
            and item["deterministic"]
            and requests_ok == requests_total
        )
        if item["passed"] is not expected_pass:
            raise MtpSelectionError(f"K={key} pass result is internally inconsistent")

    if candidates[0].get("passed") is not True:
        raise MtpSelectionError("non-speculative K=0 baseline did not pass")

    try:
        selected = int(data.get("selected_k"))
    except (TypeError, ValueError) as exc:
        raise MtpSelectionError("selected_k is invalid") from exc
    if selected not in expected_keys:
        raise MtpSelectionError(f"selected_k={selected} is outside the measured range")
    if candidates[selected].get("passed") is not True:
        raise MtpSelectionError(f"selected_k={selected} was disqualified")
    try:
        selected_tps = float(candidates[selected].get("median_decode_tps"))
    except (TypeError, ValueError) as exc:
        raise MtpSelectionError("selected candidate has invalid throughput") from exc
    if not math.isfinite(selected_tps) or selected_tps < MIN_SELECTED_DECODE_TPS:
        raise MtpSelectionError(
            f"selected K={selected} measured {selected_tps:.3f} decode tok/s; "
            f"minimum is {MIN_SELECTED_DECODE_TPS:.1f}")
    recomputed = expected_winner(candidates)
    if selected != recomputed:
        raise MtpSelectionError(
            f"recorded selected_k={selected} disagrees with policy winner K={recomputed}")
    return selected


def load_selection(path, **validation_kwargs) -> tuple[int, Dict]:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MtpSelectionError(f"could not read MTP selection {p}: {exc}") from exc
    return validate_selection_manifest(data, **validation_kwargs), data
