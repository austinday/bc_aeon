#!/usr/bin/env python3
"""Extract hash-bound Qwen3.8 target features through a patched vLLM server.

The server-side capture hook writes the five DFlash target-layer activations.
This client deliberately sends one complete, unpadded conversation at a time so
each document has the same isolation that packed DFlash training enforces.
Prompt text and token IDs never enter logs or receipts.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any

import requests


SCHEMA_VERSION = "aeon-qwen38-dflash-feature-index-v1"
CAPTURE_SCHEMA_VERSION = "aeon-qwen38-dflash-feature-v1"
EXPECTED_DATASET_SHA256 = (
    "61b8e150651ecc14c47e1068ce36fc130bb56e18117b3b68e098390defea92f5"
)
EXPECTED_DATASET_ROWS = 256
EXPECTED_LAYER_IDS = [6, 20, 34, 48, 62]
EXPECTED_HIDDEN_SIZE = 5120
MAX_SEQUENCE_TOKENS = 10240
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class FeatureExtractionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        payload = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _private_file(path: Path, *, maximum: int) -> bytes:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= maximum
    ):
        raise FeatureExtractionError(f"private file identity changed: {path}")
    return path.read_bytes()


def _token_hash(token_ids: list[int]) -> str:
    if sys.byteorder != "little":
        raise FeatureExtractionError("feature extraction requires little-endian x86")
    values = array("I", token_ids)
    return hashlib.sha256(
        len(token_ids).to_bytes(8, "little") + values.tobytes()
    ).hexdigest()


def _normalized_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise FeatureExtractionError("dataset row has no messages")
    result: list[dict[str, Any]] = []
    for message in value:
        if not isinstance(message, dict):
            raise FeatureExtractionError("dataset message is not an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise FeatureExtractionError("dataset message role is unsupported")
        if not isinstance(content, str):
            raise FeatureExtractionError("dataset message content is not text")
        allowed = {"role", "content"}
        normalized = {"role": role, "content": content}
        if role == "assistant" and "reasoning_content" in message:
            reasoning = message["reasoning_content"]
            if not isinstance(reasoning, str):
                raise FeatureExtractionError("assistant reasoning is not text")
            allowed.add("reasoning_content")
            normalized["reasoning_content"] = reasoning
        # The exact adaptation corpus has no tool calls. Refuse to silently
        # discard structured fields if that identity ever changes.
        if set(message) != allowed:
            raise FeatureExtractionError("dataset message structure changed")
        result.append(normalized)
    return result


def _post_json(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    *,
    timeout: tuple[float, float],
) -> dict[str, Any]:
    response = session.post(
        url,
        json=payload,
        timeout=timeout,
        allow_redirects=False,
    )
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise FeatureExtractionError("server response exceeded its bound")
    if response.status_code != 200:
        raise FeatureExtractionError(
            f"server refused feature request with HTTP {response.status_code}"
        )
    try:
        value = response.json()
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FeatureExtractionError("server response is not JSON") from exc
    if not isinstance(value, dict):
        raise FeatureExtractionError("server response is not an object")
    return value


def _tokenize(
    session: requests.Session,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
) -> list[int]:
    value = _post_json(
        session,
        base_url + "/tokenize",
        {
            "model": model,
            "messages": messages,
            "add_generation_prompt": False,
        },
        timeout=(5, 120),
    )
    tokens = value.get("tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or len(tokens) > MAX_SEQUENCE_TOKENS
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < 248320
            for token in tokens
        )
    ):
        raise FeatureExtractionError("tokenizer returned an invalid sequence")
    count = value.get("count")
    if count is not None and count != len(tokens):
        raise FeatureExtractionError("tokenizer count is inconsistent")
    return tokens


def _capture_one(
    session: requests.Session,
    base_url: str,
    model: str,
    tokens: list[int],
) -> None:
    value = _post_json(
        session,
        base_url + "/v1/completions",
        {
            "model": model,
            "prompt": tokens,
            "max_tokens": 1,
            "temperature": 0,
            "seed": 0,
        },
        timeout=(5, 900),
    )
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise FeatureExtractionError("capture completion is malformed")


def _capture_receipt(feature_dir: Path, token_hash: str) -> tuple[dict[str, Any], Path]:
    receipt_path = feature_dir / f"{token_hash}.json"
    feature_path = feature_dir / f"{token_hash}.safetensors"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if receipt_path.is_file() and feature_path.is_file():
            break
        time.sleep(0.1)
    payload = _private_file(receipt_path, maximum=64 * 1024)
    try:
        receipt = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FeatureExtractionError("capture receipt is malformed") from exc
    if not isinstance(receipt, dict):
        raise FeatureExtractionError("capture receipt is not an object")
    feature_metadata = feature_path.lstat()
    if (
        not stat.S_ISREG(feature_metadata.st_mode)
        or feature_metadata.st_uid != os.geteuid()
        or feature_metadata.st_nlink != 1
        or feature_metadata.st_mode & 0o077
        or feature_metadata.st_size <= 0
    ):
        raise FeatureExtractionError("captured tensor identity changed")
    return receipt, feature_path


def extract(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(args.dataset)
    feature_dir = Path(args.feature_dir)
    output = Path(args.output)
    if _sha256(dataset) != EXPECTED_DATASET_SHA256:
        raise FeatureExtractionError("adaptation dataset digest changed")
    feature_metadata = feature_dir.lstat()
    if (
        not stat.S_ISDIR(feature_metadata.st_mode)
        or feature_metadata.st_uid != os.geteuid()
        or feature_metadata.st_mode & 0o077
        or any(feature_dir.iterdir())
    ):
        raise FeatureExtractionError("feature directory is unsafe or nonempty")
    rows = dataset.read_text(encoding="utf-8").splitlines()
    if len(rows) != EXPECTED_DATASET_ROWS:
        raise FeatureExtractionError("adaptation dataset row count changed")

    records: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    total_tokens = 0
    started = time.time()
    with requests.Session() as session:
        # This client is invoked inside a Fleet-owned worker and talks only to
        # that worker's exact loopback vLLM endpoint.  Host proxy settings are
        # never part of the workload contract.
        session.trust_env = False
        for row_index, raw in enumerate(rows):
            try:
                row = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise FeatureExtractionError("dataset JSON is malformed") from exc
            if not isinstance(row, dict):
                raise FeatureExtractionError("dataset row is not an object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise FeatureExtractionError("dataset sample identity changed")
            messages = _normalized_messages(row.get("messages"))
            tokens = _tokenize(session, args.base_url, args.model, messages)
            token_hash = _token_hash(tokens)
            total_tokens += len(tokens)
            if token_hash not in seen:
                _capture_one(session, args.base_url, args.model, tokens)
                receipt, feature_path = _capture_receipt(feature_dir, token_hash)
                expected_receipt = {
                    "schema_version": CAPTURE_SCHEMA_VERSION,
                    "token_hash": token_hash,
                    "token_count": len(tokens),
                    "layer_ids": EXPECTED_LAYER_IDS,
                    "hidden_size": EXPECTED_HIDDEN_SIZE,
                    "feature_width": len(EXPECTED_LAYER_IDS) * EXPECTED_HIDDEN_SIZE,
                    "dtype": "bfloat16",
                    "model_sha256s": args.model_sha256s,
                    "dataset_sha256": EXPECTED_DATASET_SHA256,
                }
                if receipt != expected_receipt:
                    raise FeatureExtractionError("capture receipt identity changed")
                seen[token_hash] = {
                    "token_hash": token_hash,
                    "token_count": len(tokens),
                    "feature_file": feature_path.name,
                    "feature_bytes": feature_path.stat().st_size,
                    "feature_sha256": _sha256(feature_path),
                    "receipt_file": f"{token_hash}.json",
                }
            records.append(
                {
                    "row_index": row_index,
                    "sample_id": sample_id,
                    "token_hash": token_hash,
                    "token_count": len(tokens),
                }
            )

    result = {
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "dataset_rows": len(rows),
        "model_sha256s": args.model_sha256s,
        "draft_sha256": args.draft_sha256,
        "layer_ids": EXPECTED_LAYER_IDS,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "feature_width": len(EXPECTED_LAYER_IDS) * EXPECTED_HIDDEN_SIZE,
        "dtype": "bfloat16",
        "total_tokens": total_tokens,
        "unique_features": len(seen),
        "feature_bytes": sum(item["feature_bytes"] for item in seen.values()),
        "records": records,
        "features": [seen[key] for key in sorted(seen)],
        "started_at": started,
        "completed_at": time.time(),
    }
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--model-sha256s", required=True)
    parser.add_argument("--draft-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    result = extract(_parser().parse_args())
    print(
        json.dumps(
            {
                "dataset_rows": result["dataset_rows"],
                "feature_bytes": result["feature_bytes"],
                "total_tokens": result["total_tokens"],
                "unique_features": result["unique_features"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
