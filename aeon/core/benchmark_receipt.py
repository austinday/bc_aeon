"""Authenticated, typed evidence emitted by benchmarked Aeon tools.

The model's response and harness transcript are untrusted for scoring.  During
an owner benchmark only, a reviewed tool can append a small HMAC-bound receipt
to an executor-created private file.  The executor validates that receipt
independently after the harness exits.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


CAPABILITY_RECEIPT_PATH_ENV = "AEON_BENCHMARK_GPU_CAPABILITY_RECEIPT_PATH"
CAPABILITY_RECEIPT_KEY_ENV = "AEON_BENCHMARK_GPU_CAPABILITY_RECEIPT_KEY"
CAPABILITY_RECEIPT_SCHEMA_VERSION = 1
CAPABILITY_RECEIPT_TYPE = "fleet_wait_capability"
MAX_CAPABILITY_RECEIPT_BYTES = 16_384
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FleetWaitCapabilityReceipt:
    schema_version: int
    receipt_type: str
    tool_name: str
    status: str
    submission_boundary: str
    unavailable_compute_is_durable_wait: bool
    general_model_build_available: bool
    eligible_recipe_count: int
    capability_payload_sha256: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _typed_receipt(document: Mapping[str, object]) -> FleetWaitCapabilityReceipt:
    if not isinstance(document, Mapping):
        raise ValueError("Fleet capability result is not structured")
    recipes = document.get("recipes")
    if not isinstance(recipes, list) or len(recipes) > 64:
        raise ValueError("Fleet capability recipes are malformed")
    for recipe in recipes:
        if not isinstance(recipe, Mapping) or set(recipe) != {
            "recipe_id",
            "profile_id",
            "purpose",
            "durable_wait",
        }:
            raise ValueError("Fleet capability recipe is malformed")
        if (
            not all(
                isinstance(recipe.get(field), str) and bool(recipe.get(field))
                for field in ("recipe_id", "profile_id", "purpose")
            )
            or recipe.get("durable_wait") is not True
        ):
            raise ValueError("Fleet capability recipe is not durable and typed")
    if (
        document.get("status") != "ok"
        or document.get("general_model_build_available") is not False
        or document.get("submission_boundary") != "reviewed_recipe_only"
        or document.get("unavailable_compute_is_durable_wait") is not True
    ):
        raise ValueError("Fleet capability boundary is not the reviewed durable path")
    return FleetWaitCapabilityReceipt(
        schema_version=CAPABILITY_RECEIPT_SCHEMA_VERSION,
        receipt_type=CAPABILITY_RECEIPT_TYPE,
        tool_name="fleet_batch_capabilities",
        status="ok",
        submission_boundary="reviewed_recipe_only",
        unavailable_compute_is_durable_wait=True,
        general_model_build_available=False,
        eligible_recipe_count=len(recipes),
        capability_payload_sha256=hashlib.sha256(
            _canonical_json(dict(document))
        ).hexdigest(),
    )


def emit_fleet_wait_capability_receipt(document: Mapping[str, object]) -> None:
    """Append one authenticated receipt when a benchmark executor requested it.

    This observational side channel grants no Fleet authority.  Missing or
    malformed benchmark configuration is ignored so normal tool behavior never
    depends on benchmark instrumentation.
    """

    raw_path = os.environ.get(CAPABILITY_RECEIPT_PATH_ENV, "")
    raw_key = os.environ.get(CAPABILITY_RECEIPT_KEY_ENV, "")
    if not raw_path or not _KEY_RE.fullmatch(raw_key):
        return
    path = Path(raw_path)
    if not path.is_absolute():
        return
    try:
        receipt = _typed_receipt(document)
        payload = asdict(receipt)
        encoded = _canonical_json(payload)
        envelope = _canonical_json(
            {
                "payload": payload,
                "hmac_sha256": hmac.new(
                    bytes.fromhex(raw_key), encoded, hashlib.sha256
                ).hexdigest(),
            }
        ) + b"\n"
        if len(envelope) > 4096:
            return
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size + len(envelope) > MAX_CAPABILITY_RECEIPT_BYTES
            ):
                return
            written = 0
            while written < len(envelope):
                written += os.write(descriptor, envelope[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except (OSError, TypeError, ValueError):
        return


def decode_capability_receipts(
    payload: bytes,
    *,
    key: str,
) -> tuple[FleetWaitCapabilityReceipt, ...]:
    """Validate a bounded receipt stream and return only typed receipts."""

    if not _KEY_RE.fullmatch(str(key or "")):
        return ()
    if not payload or len(payload) > MAX_CAPABILITY_RECEIPT_BYTES:
        return ()
    result: list[FleetWaitCapabilityReceipt] = []
    for line in payload.splitlines():
        if not line or len(line) > 4096:
            return ()
        try:
            envelope = json.loads(line)
        except (UnicodeError, json.JSONDecodeError):
            return ()
        if not isinstance(envelope, dict) or set(envelope) != {
            "payload",
            "hmac_sha256",
        }:
            return ()
        item = envelope.get("payload")
        supplied_mac = envelope.get("hmac_sha256")
        if not isinstance(item, dict) or not isinstance(supplied_mac, str):
            return ()
        encoded = _canonical_json(item)
        expected_mac = hmac.new(
            bytes.fromhex(key), encoded, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(supplied_mac, expected_mac):
            return ()
        try:
            receipt = FleetWaitCapabilityReceipt(**item)
        except TypeError:
            return ()
        if (
            asdict(receipt) != item
            or isinstance(receipt.schema_version, bool)
            or not isinstance(receipt.schema_version, int)
            or receipt.schema_version != CAPABILITY_RECEIPT_SCHEMA_VERSION
            or receipt.receipt_type != CAPABILITY_RECEIPT_TYPE
            or receipt.tool_name != "fleet_batch_capabilities"
            or receipt.status != "ok"
            or receipt.submission_boundary != "reviewed_recipe_only"
            or receipt.unavailable_compute_is_durable_wait is not True
            or receipt.general_model_build_available is not False
            or isinstance(receipt.eligible_recipe_count, bool)
            or not isinstance(receipt.eligible_recipe_count, int)
            or not 0 <= receipt.eligible_recipe_count <= 64
            or not isinstance(receipt.capability_payload_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", receipt.capability_payload_sha256)
            is None
        ):
            return ()
        result.append(receipt)
    return tuple(result)


__all__ = (
    "CAPABILITY_RECEIPT_KEY_ENV",
    "CAPABILITY_RECEIPT_PATH_ENV",
    "FleetWaitCapabilityReceipt",
    "decode_capability_receipts",
    "emit_fleet_wait_capability_receipt",
)
