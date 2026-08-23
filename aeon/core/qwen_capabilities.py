"""Immutable, release-bound placement capabilities for Aeon's Qwen runtime.

This is an authorization registry, not an inventory. A host appearing here as
disabled is deliberately not compute capacity. Live availability still comes
only from the fleet coordinator after one enabled capability is selected.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CAPABILITY_SCHEMA_VERSION = "aeon-qwen-runtime-capabilities-v1"
CAPABILITY_MANIFEST_FILE = (
    Path(__file__).resolve().parent / "data/qwen_runtime_capabilities.json"
)
STANDARD_RELEASE_PROFILE = "qwen38-standard-114688-k3"
STANDARD_CONTEXT_TOKENS = 114688
STANDARD_VRAM_BUDGET_GB = 48.7
STANDARD_MIN_PHYSICAL_VRAM_GB = 90.0
STANDARD_IMAGE_ID = (
    "sha256:d57400972ab0ae46baac64d4bfcc49cb136c07d8b0c50a76c7e2d81bd8a9fe47"
)
STANDARD_MODEL_MANIFEST_SHA256 = (
    "1a3ba1eb88d0507bdef3798a6db59830dc076199b7db7d111201f6997588220e"
)
STANDARD_MODEL_SHA256S_SHA256 = (
    "e7eca7ebee03c4f27482d4fe421ca1fac9f1d9986663a51fd7614361010c1237"
)
LOCAL_DOCKER_CAPABILITY_KEY = "qwen38-standard-177-local-docker"
RTX5000_RELEASE_CANDIDATE_KEY = "qwen38-compact-180-disabled"
RTX5000_RELEASE_CAPABILITY_KEY = "qwen38-compact-180-128k"
RTX5000_RELEASE_RECEIPT_FILE = (
    Path(__file__).resolve().parent
    / "data/qwen38_rtx5000_128k_release_receipt.json"
)

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
_HOST_RE = re.compile(r"^192[.]168[.]0[.][0-9]{1,3}$")
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,62}$")
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CAPABILITY_FIELDS = {
    "allowed_physical_gpus",
    "compute_profile",
    "context_tokens",
    "disabled_reason",
    "enabled",
    "exclusive",
    "host",
    "hostname",
    "image_id",
    "gpu_memory_utilization",
    "key",
    "max_batched_tokens",
    "max_num_seqs",
    "model_manifest_sha256",
    "model_sha256s_sha256",
    "min_physical_vram_gb",
    "release_profile",
    "runtime_adapter",
    "release_receipt_sha256",
    "vram_budget_gb",
}


class QwenCapabilityError(ValueError):
    """The static capability authorization is missing, changed, or disabled."""


@dataclass(frozen=True)
class QwenRuntimeCapability:
    key: str
    enabled: bool
    release_profile: str
    host: str
    hostname: str
    runtime_adapter: str
    allowed_physical_gpus: tuple[int, ...]
    min_physical_vram_gb: float
    vram_budget_gb: float | None
    exclusive: bool
    context_tokens: int
    image_id: str | None
    gpu_memory_utilization: float | None
    max_num_seqs: int | None
    max_batched_tokens: int | None
    model_manifest_sha256: str
    model_sha256s_sha256: str
    release_receipt_sha256: str | None
    compute_profile: str
    disabled_reason: str | None

    @property
    def coordinator_gpu(self) -> int:
        if not self.enabled or len(self.allowed_physical_gpus) != 1:
            raise QwenCapabilityError(
                "the active runtime capability has no single safe GPU selector"
            )
        return self.allowed_physical_gpus[0]


@dataclass(frozen=True)
class QwenRuntimeCapabilityRegistry:
    schema_version: str
    manifest_sha256: str
    capabilities: tuple[QwenRuntimeCapability, ...]

    @property
    def active(self) -> QwenRuntimeCapability:
        """Return the preferred enabled capability for legacy local callers."""

        enabled = self.enabled
        if not enabled:
            raise QwenCapabilityError("no Qwen runtime capability is enabled")
        return enabled[0]

    @property
    def enabled(self) -> tuple[QwenRuntimeCapability, ...]:
        """Enabled releases in deterministic placement-preference order."""

        return tuple(item for item in self.capabilities if item.enabled)


def _capability(
    *,
    key: str,
    enabled: bool,
    release_profile: str,
    host: str,
    hostname: str,
    runtime_adapter: str,
    allowed_physical_gpus: tuple[int, ...],
    min_physical_vram_gb: float,
    vram_budget_gb: float | None,
    exclusive: bool,
    context_tokens: int,
    image_id: str | None,
    gpu_memory_utilization: float | None,
    max_num_seqs: int | None,
    max_batched_tokens: int | None,
    model_manifest_sha256: str,
    model_sha256s_sha256: str,
    release_receipt_sha256: str | None,
    compute_profile: str,
    disabled_reason: str | None,
) -> QwenRuntimeCapability:
    return QwenRuntimeCapability(
        key=key,
        enabled=enabled,
        release_profile=release_profile,
        host=host,
        hostname=hostname,
        runtime_adapter=runtime_adapter,
        allowed_physical_gpus=allowed_physical_gpus,
        min_physical_vram_gb=min_physical_vram_gb,
        vram_budget_gb=vram_budget_gb,
        exclusive=exclusive,
        context_tokens=context_tokens,
        image_id=image_id,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_batched_tokens=max_batched_tokens,
        model_manifest_sha256=model_manifest_sha256,
        model_sha256s_sha256=model_sha256s_sha256,
        release_receipt_sha256=release_receipt_sha256,
        compute_profile=compute_profile,
        disabled_reason=disabled_reason,
    )


_EXPECTED_CAPABILITIES = (
    _capability(
        key=LOCAL_DOCKER_CAPABILITY_KEY,
        enabled=True,
        release_profile=STANDARD_RELEASE_PROFILE,
        host="192.168.0.177",
        hostname="DAY2RTX6000PRO",
        runtime_adapter="local-docker",
        allowed_physical_gpus=(0,),
        min_physical_vram_gb=STANDARD_MIN_PHYSICAL_VRAM_GB,
        vram_budget_gb=STANDARD_VRAM_BUDGET_GB,
        exclusive=True,
        context_tokens=STANDARD_CONTEXT_TOKENS,
        image_id=STANDARD_IMAGE_ID,
        gpu_memory_utilization=0.415,
        max_num_seqs=1,
        max_batched_tokens=32768,
        model_manifest_sha256=STANDARD_MODEL_MANIFEST_SHA256,
        model_sha256s_sha256=STANDARD_MODEL_SHA256S_SHA256,
        release_receipt_sha256=None,
        compute_profile="qwen38-vllm",
        disabled_reason=None,
    ),
    _capability(
        key="qwen38-standard-179-awaiting-runtime",
        enabled=False,
        release_profile=STANDARD_RELEASE_PROFILE,
        host="192.168.0.179",
        hostname="DAY2XRTX6000-2",
        runtime_adapter="remote-bare-unreleased",
        allowed_physical_gpus=(0, 1),
        min_physical_vram_gb=STANDARD_MIN_PHYSICAL_VRAM_GB,
        vram_budget_gb=STANDARD_VRAM_BUDGET_GB,
        exclusive=True,
        context_tokens=STANDARD_CONTEXT_TOKENS,
        image_id=None,
        gpu_memory_utilization=0.415,
        max_num_seqs=1,
        max_batched_tokens=32768,
        model_manifest_sha256=STANDARD_MODEL_MANIFEST_SHA256,
        model_sha256s_sha256=STANDARD_MODEL_SHA256S_SHA256,
        release_receipt_sha256=None,
        compute_profile="qwen38-vllm",
        disabled_reason=(
            "No reviewed aday-owned runtime, transport, or host release receipt "
            "exists on .179."
        ),
    ),
    _capability(
        key="qwen38-compact-178-disabled",
        enabled=False,
        release_profile="qwen38-compact-64k-unreleased",
        host="192.168.0.178",
        hostname="DAY2XRTX5000",
        runtime_adapter="remote-docker",
        allowed_physical_gpus=(0, 1),
        min_physical_vram_gb=47.0,
        vram_budget_gb=41.7,
        exclusive=True,
        context_tokens=65536,
        image_id=STANDARD_IMAGE_ID,
        gpu_memory_utilization=0.674,
        max_num_seqs=1,
        max_batched_tokens=32768,
        model_manifest_sha256=STANDARD_MODEL_MANIFEST_SHA256,
        model_sha256s_sha256=STANDARD_MODEL_SHA256S_SHA256,
        release_receipt_sha256=None,
        compute_profile="qwen38-vllm",
        disabled_reason=(
            "The 48 GB compact profile has no truthful <=41.7 GiB release receipt."
        ),
    ),
    _capability(
        key=RTX5000_RELEASE_CAPABILITY_KEY,
        enabled=True,
        release_profile="qwen38-compact-128k-k3",
        host="192.168.0.180",
        hostname="DAY2XRTX5000PRO-2",
        runtime_adapter="remote-docker",
        allowed_physical_gpus=(0, 1),
        min_physical_vram_gb=47.0,
        vram_budget_gb=41.25,
        exclusive=True,
        context_tokens=131072,
        image_id=STANDARD_IMAGE_ID,
        gpu_memory_utilization=0.70,
        max_num_seqs=8,
        max_batched_tokens=8192,
        model_manifest_sha256=STANDARD_MODEL_MANIFEST_SHA256,
        model_sha256s_sha256=STANDARD_MODEL_SHA256S_SHA256,
        release_receipt_sha256=(
            "5eda720a4e168733fd0881cedb144a69f892c707439a8e64304e24ec6a04a91a"
        ),
        compute_profile="qwen38-vllm",
        disabled_reason=None,
    ),
)


def _parse_capability(raw: Any) -> QwenRuntimeCapability:
    if not isinstance(raw, dict) or set(raw) != _CAPABILITY_FIELDS:
        raise QwenCapabilityError("Qwen runtime capability fields changed")
    key = raw["key"]
    host = raw["host"]
    hostname = raw["hostname"]
    enabled = raw["enabled"]
    exclusive = raw["exclusive"]
    gpus = raw["allowed_physical_gpus"]
    minimum = raw["min_physical_vram_gb"]
    budget = raw["vram_budget_gb"]
    context = raw["context_tokens"]
    image_id = raw["image_id"]
    utility = raw["gpu_memory_utilization"]
    max_num_seqs = raw["max_num_seqs"]
    max_batched_tokens = raw["max_batched_tokens"]
    model_manifest_sha256 = raw["model_manifest_sha256"]
    model_sha256s_sha256 = raw["model_sha256s_sha256"]
    release_receipt_sha256 = raw["release_receipt_sha256"]
    disabled_reason = raw["disabled_reason"]
    if (
        not isinstance(key, str)
        or _KEY_RE.fullmatch(key) is None
        or not isinstance(host, str)
        or _HOST_RE.fullmatch(host) is None
        or not isinstance(hostname, str)
        or _HOSTNAME_RE.fullmatch(hostname) is None
        or not isinstance(enabled, bool)
        or not isinstance(exclusive, bool)
        or not isinstance(gpus, list)
        or not gpus
        or any(isinstance(value, bool) or not isinstance(value, int) for value in gpus)
        or len(gpus) != len(set(gpus))
        or any(value not in {0, 1} for value in gpus)
        or isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or float(minimum) <= 0
        or (
            budget is not None
            and (
                isinstance(budget, bool)
                or not isinstance(budget, (int, float))
                or not math.isfinite(float(budget))
                or float(budget) <= 0
            )
        )
        or isinstance(context, bool)
        or not isinstance(context, int)
        or context <= 0
        or (image_id is not None and (not isinstance(image_id, str) or _IMAGE_ID_RE.fullmatch(image_id) is None))
        or (
            utility is not None
            and (
                isinstance(utility, bool)
                or not isinstance(utility, (int, float))
                or not math.isfinite(float(utility))
                or not 0 < float(utility) < 1
            )
        )
        or (
            max_num_seqs is not None
            and (isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int) or max_num_seqs <= 0)
        )
        or (
            max_batched_tokens is not None
            and (
                isinstance(max_batched_tokens, bool)
                or not isinstance(max_batched_tokens, int)
                or max_batched_tokens <= 0
            )
        )
        or (
            release_receipt_sha256 is not None
            and (
                not isinstance(release_receipt_sha256, str)
                or _SHA256_RE.fullmatch(release_receipt_sha256) is None
            )
        )
        or not isinstance(model_manifest_sha256, str)
        or _SHA256_RE.fullmatch(model_manifest_sha256) is None
        or not isinstance(model_sha256s_sha256, str)
        or _SHA256_RE.fullmatch(model_sha256s_sha256) is None
        or not isinstance(raw["release_profile"], str)
        or not raw["release_profile"]
        or not isinstance(raw["runtime_adapter"], str)
        or not raw["runtime_adapter"]
        or not isinstance(raw["compute_profile"], str)
        or not raw["compute_profile"]
        or (disabled_reason is not None and not isinstance(disabled_reason, str))
    ):
        raise QwenCapabilityError("Qwen runtime capability is malformed")
    return _capability(
        key=key,
        enabled=enabled,
        release_profile=raw["release_profile"],
        host=host,
        hostname=hostname,
        runtime_adapter=raw["runtime_adapter"],
        allowed_physical_gpus=tuple(gpus),
        min_physical_vram_gb=float(minimum),
        vram_budget_gb=None if budget is None else float(budget),
        exclusive=exclusive,
        context_tokens=context,
        image_id=image_id,
        gpu_memory_utilization=None if utility is None else float(utility),
        max_num_seqs=max_num_seqs,
        max_batched_tokens=max_batched_tokens,
        model_manifest_sha256=model_manifest_sha256,
        model_sha256s_sha256=model_sha256s_sha256,
        release_receipt_sha256=release_receipt_sha256,
        compute_profile=raw["compute_profile"],
        disabled_reason=disabled_reason,
    )


def load_qwen_runtime_capabilities(
    path: Path = CAPABILITY_MANIFEST_FILE,
) -> QwenRuntimeCapabilityRegistry:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise QwenCapabilityError("Qwen runtime capability manifest is unavailable") from exc
    if not payload or len(payload) > 64 * 1024:
        raise QwenCapabilityError("Qwen runtime capability manifest size is invalid")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenCapabilityError("Qwen runtime capability manifest is malformed") from exc
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema_version", "capabilities"}
        or raw.get("schema_version") != CAPABILITY_SCHEMA_VERSION
        or not isinstance(raw.get("capabilities"), list)
    ):
        raise QwenCapabilityError("Qwen runtime capability manifest schema changed")
    capabilities = tuple(_parse_capability(item) for item in raw["capabilities"])
    if capabilities != _EXPECTED_CAPABILITIES:
        raise QwenCapabilityError(
            "Qwen runtime targets changed without a reviewed capability code update"
        )
    released_5000 = tuple(
        item for item in capabilities if item.key == RTX5000_RELEASE_CAPABILITY_KEY
    )
    try:
        receipt_payload = RTX5000_RELEASE_RECEIPT_FILE.read_bytes()
    except OSError as exc:
        raise QwenCapabilityError("RTX 5000 release receipt is unavailable") from exc
    if (
        len(released_5000) != 1
        or not receipt_payload
        or len(receipt_payload) > 64 * 1024
        or hashlib.sha256(receipt_payload).hexdigest()
        != released_5000[0].release_receipt_sha256
    ):
        raise QwenCapabilityError("RTX 5000 release receipt identity changed")
    if len({item.key for item in capabilities}) != len(capabilities) or len(
        {item.host for item in capabilities}
    ) != len(capabilities):
        raise QwenCapabilityError("Qwen runtime capabilities are duplicated")
    manifest_sha256 = hashlib.sha256(payload).hexdigest()
    return QwenRuntimeCapabilityRegistry(
        schema_version=CAPABILITY_SCHEMA_VERSION,
        manifest_sha256=manifest_sha256,
        capabilities=capabilities,
    )


def active_qwen_runtime_capability() -> tuple[QwenRuntimeCapability, str]:
    registry = load_qwen_runtime_capabilities()
    return registry.active, registry.manifest_sha256


def enabled_qwen_runtime_capabilities() -> tuple[
    tuple[QwenRuntimeCapability, ...], str
]:
    registry = load_qwen_runtime_capabilities()
    if not registry.enabled:
        raise QwenCapabilityError("no Qwen runtime capability is enabled")
    return registry.enabled, registry.manifest_sha256


def qwen_runtime_capability(
    key: str, *, require_enabled: bool = True
) -> tuple[QwenRuntimeCapability, str]:
    if not isinstance(key, str) or _KEY_RE.fullmatch(key) is None:
        raise QwenCapabilityError("Qwen capability key is malformed")
    registry = load_qwen_runtime_capabilities()
    matches = tuple(item for item in registry.capabilities if item.key == key)
    if len(matches) != 1:
        raise QwenCapabilityError("Qwen capability key is unknown")
    capability = matches[0]
    if require_enabled and not capability.enabled:
        raise QwenCapabilityError(
            capability.disabled_reason or "Qwen target capability is disabled"
        )
    return capability, registry.manifest_sha256


def qwen_release_candidate_capability(
    key: str,
) -> tuple[QwenRuntimeCapability, str]:
    """Return the one exact disabled target authorized only for release gating.

    Normal placement deliberately uses :func:`enabled_qwen_runtime_capabilities`
    and can never see this entry. Keeping this accessor exact prevents a generic
    ``allow_disabled`` switch from turning future placeholder hosts into compute.
    """

    if key != RTX5000_RELEASE_CANDIDATE_KEY:
        raise QwenCapabilityError("Qwen release-candidate key is not authorized")
    capability, manifest_sha256 = qwen_runtime_capability(
        key, require_enabled=False
    )
    if (
        capability.enabled
        or capability.runtime_adapter != "remote-docker"
        or capability.host != "192.168.0.180"
        or capability.release_receipt_sha256 is not None
        or not capability.disabled_reason
    ):
        raise QwenCapabilityError("Qwen release-candidate contract changed")
    return capability, manifest_sha256


def require_qwen_release_candidate_target(
    key: str, host: str, physical_gpu: int
) -> tuple[QwenRuntimeCapability, str]:
    capability, manifest_sha256 = qwen_release_candidate_capability(key)
    if host != capability.host or physical_gpu not in capability.allowed_physical_gpus:
        raise QwenCapabilityError("Qwen release-candidate selector is unauthorized")
    return capability, manifest_sha256


def require_enabled_qwen_target(
    host: str, physical_gpu: int
) -> tuple[QwenRuntimeCapability, str]:
    if (
        not isinstance(host, str)
        or isinstance(physical_gpu, bool)
        or not isinstance(physical_gpu, int)
    ):
        raise QwenCapabilityError("Qwen target selector is malformed")
    registry = load_qwen_runtime_capabilities()
    matches = tuple(item for item in registry.capabilities if item.host == host)
    if len(matches) != 1:
        raise QwenCapabilityError("Qwen target has no exact capability entry")
    capability = matches[0]
    if not capability.enabled:
        raise QwenCapabilityError(
            capability.disabled_reason or "Qwen target capability is disabled"
        )
    if physical_gpu not in capability.allowed_physical_gpus:
        raise QwenCapabilityError("Qwen target GPU is outside the capability receipt")
    return capability, registry.manifest_sha256


def validate_qwen_capability_receipt(
    *,
    key: Any,
    manifest_sha256: Any,
    runtime_adapter: Any,
    host: Any,
    physical_gpu: Any,
    release_gate: Any = False,
) -> QwenRuntimeCapability:
    if (
        not isinstance(key, str)
        or not isinstance(manifest_sha256, str)
        or _SHA256_RE.fullmatch(manifest_sha256) is None
        or not isinstance(runtime_adapter, str)
        or not isinstance(host, str)
        or isinstance(physical_gpu, bool)
        or not isinstance(physical_gpu, int)
    ):
        raise QwenCapabilityError("Qwen capability receipt is malformed")
    if release_gate is True:
        capability, current_hash = require_qwen_release_candidate_target(
            key, host, physical_gpu
        )
    elif release_gate is False:
        capability, current_hash = require_enabled_qwen_target(host, physical_gpu)
    else:
        raise QwenCapabilityError("Qwen release-gate receipt is malformed")
    if (
        key != capability.key
        or manifest_sha256 != current_hash
        or runtime_adapter != capability.runtime_adapter
    ):
        raise QwenCapabilityError("Qwen capability receipt changed")
    return capability
