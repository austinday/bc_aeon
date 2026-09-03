#!/usr/bin/env python3
"""Receipt-bound local .177 worker for sequential Flash-Next qualification boots."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import requests

from aeon.core import qwen_flash_next_runtime_contract as runtime_contract
from aeon.scripts import build_qwen38_flash_next_nvfp4 as builder
from aeon.scripts import qualify_qwen38_flash_next_endpoint as qualify
from aeon.scripts import release_qwen38_flash_next as release_tool


SCHEMA = "aeon-qwen38-flash-next-qualification-job-v2"
RESULT_SCHEMA = "aeon-qwen38-flash-next-qualification-result-v2"
RUNTIME_COMMAND_SCHEMA = "aeon-qwen38-flash-next-runtime-commands-v1"
HOST = "192.168.0.177"
HOSTNAME = "DAY2RTX6000PRO"
SCRATCH_ROOT = PurePosixPath("/home/aday/.local/state/fleet-compute/runs")
CANONICAL_OUTPUT_ROOT = PurePosixPath(
    "/home/aday/.local/state/fleet-compute/artifacts/"
    "aeon-qwen38-flash-next-qualification"
)
CHECKPOINT_ROOT = PurePosixPath(
    "/home/aday/.local/state/fleet-compute/artifacts/aeon-qwen38-flash-next-build"
)
LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
DOCKER = "/usr/bin/docker"
IMAGE_REFERENCE = release_tool.SGLANG_IMAGE_REFERENCE
IMAGE_DIGEST = release_tool.SGLANG_IMAGE_DIGEST
IMAGE_CONFIG_DIGEST = release_tool.SGLANG_IMAGE_CONFIG_DIGEST
IMAGE_ID = release_tool.SGLANG_IMAGE_ID
IMAGE_ARCHIVE_SHA256 = release_tool.SGLANG_IMAGE_ARCHIVE_SHA256
IMAGE_ARCHIVE_PATH = Path(
    "/home/aday/.local/state/aeon-flash-next/runtime-images/"
    "qwen38-flash-next-sm120-headroom-a6c61-424e.oci.tar"
)
IMAGE_ARCHIVE_SIZE_BYTES = 13_951_062_528
SGLANG_COMMIT = release_tool.SGLANG_COMMIT
SERVED_ALIAS = runtime_contract.WIRE_SERVED_ALIAS
DISPLAY_NAME = runtime_contract.DISPLAY_NAME
ARTIFACT_NAME = runtime_contract.ARTIFACT_NAME
VRAM_BUDGET_GB = 88.0
TASK_MEMORY_GB = 160
TASK_MEMORY_BYTES = TASK_MEMORY_GB * 1024**3
HOST_PORT = 18039
CONTAINER_PORT = 30000
SCHEDULER_PAGE_SIZE = 64
MAX_MAMBA_CACHE_SIZE = 20
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
SELECTION_DOCKER_LOG_TAIL_LINES = 256
TUNED_CHECKPOINT_NAME = "model"
UNTUNED_CHECKPOINT_NAME = "official-untuned-model"
SIBLING_MANIFEST_NAME = "BUILD_SIBLING_MANIFEST.json"
EMPTY_SOURCE_FILES = frozenset(
    {
        "aeon/core/__init__.py",
        "aeon/scripts/__init__.py",
    }
)
MATERIALIZED_MODEL_PLACEHOLDER = "@AEON_MATERIALIZED_MODEL_PATH@"
SIBLING_MANIFEST_SCHEMA = "aeon-qwen38-flash-next-official-untuned-sibling-v1"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME = re.compile(r"^fr-[0-9a-f]{32}$")
_JOB = re.compile(r"^fj-[0-9a-f]{32}$")
_GPU = re.compile(r"^GPU-[0-9a-fA-F-]{32,64}$")
_CONTAINER = re.compile(r"^[0-9a-f]{64}$")
LEGAL_NEXTN = {(1, 2), (2, 3), (3, 4)}
ARM_OFFICIAL_UNTUNED = "official_untuned"
ARM_TUNED_MTP_OFF = "tuned_mtp_off"
ARM_SELECTION = "selection_candidate"
ARM_TUNED_MTP_ON = "tuned_mtp_on_winner"
ARMS = {
    ARM_OFFICIAL_UNTUNED,
    ARM_TUNED_MTP_OFF,
    ARM_SELECTION,
    ARM_TUNED_MTP_ON,
}
_SLUG = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,63})$")
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_RAW_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F-]{32,64}")
_RAW_CLAIM = re.compile(r"(?<![A-Za-z0-9_])gc-[A-Za-z0-9_.:-]+")
_RAW_HF_TOKEN = re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{12,}")
_RAW_BEARER = re.compile(r"(?i)(bearer\s+)[^\s\"']+")
_RAW_HOME_PATH = re.compile(r"/home/aday(?:/[^\s\"']*)?")
_active: tuple[str, dict[str, Any], list[str], "CandidateIdentity | None"] | None = None
WORKLOAD_ORDER = (
    "b1_512_512",
    "c4_512_512",
    "prefill_8192_256",
    "prefill_65152_256",
    "needle_32768_128",
    "needle_65280_128",
)
WORKLOAD_SPECS: dict[str, tuple[int, int, int]] = {
    "b1_512_512": (1, 512, 512),
    "c4_512_512": (4, 512, 512),
    "prefill_8192_256": (1, 8192, 256),
    "prefill_65152_256": (1, 65_152, 256),
    "needle_32768_128": (1, 32_768, 128),
    "needle_65280_128": (1, 65_280, 128),
}
PHASE_WORKLOADS: dict[str, frozenset[str]] = {
    # Backend selection exercises batch decode, ordinary prefill, the exact
    # 65,536-token scheduler boundary with its required page reserve, and a
    # long-context semantic needle before any later tuning can inherit the
    # selected MoE implementation.
    "moe_backend": frozenset(
        (
            "b1_512_512",
            "c4_512_512",
            "prefill_8192_256",
            "prefill_65152_256",
            "needle_65280_128",
        )
    ),
    "graph": frozenset(("b1_512_512", "c4_512_512")),
    "gdn_fp32": frozenset(("b1_512_512", "c4_512_512", "prefill_8192_256")),
    "state_dtype": frozenset(
        (
            "b1_512_512",
            "c4_512_512",
            "prefill_8192_256",
            "needle_32768_128",
            "needle_65280_128",
        )
    ),
    "mtp_prelim": frozenset(("b1_512_512", "c4_512_512")),
    "mtp_finalist": frozenset(("b1_512_512", "c4_512_512", "prefill_8192_256")),
    "replay": frozenset(
        (
            "b1_512_512",
            "c4_512_512",
            "prefill_8192_256",
            "needle_65280_128",
        )
    ),
    "chunk": frozenset(
        (
            "b1_512_512",
            "c4_512_512",
            "prefill_8192_256",
            "prefill_65152_256",
        )
    ),
    "memory": frozenset(("c4_512_512", "needle_65280_128")),
}
FINAL_WORKLOADS = frozenset(WORKLOAD_SPECS)
MIN_SELECTOR_TRIALS = 3
MIN_FINAL_TRIALS = 7
MAX_ENDPOINT_JSON_BYTES = 8 * 1024 * 1024
MAX_STREAM_BYTES = 8 * 1024 * 1024
# Exponential prompt fitting may overshoot once before it can binary-search. This
# fixed ceiling applies only to those intermediate tokenizer responses.
PROMPT_SEARCH_CONTEXT_MULTIPLIER = 4
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
)
C4_TOPICS = (
    "immutable release receipts",
    "multimodal validation",
    "paired speculative decoding",
    "GPU lease isolation",
)
CONSTANT_RUNTIME_ENV = {
    "HF_HUB_OFFLINE": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "USE_FLAX": "0",
    "USE_TF": "0",
}


@dataclass(frozen=True)
class CandidateIdentity:
    candidate_id: str
    phase: str
    ordinal: int
    parent_candidate_id: str | None
    parent_config_sha256: str | None

    def __post_init__(self) -> None:
        if (
            _SLUG.fullmatch(self.candidate_id) is None
            or _SLUG.fullmatch(self.phase) is None
            or not 0 <= self.ordinal < 64
            or (self.parent_candidate_id is None)
            is not (self.parent_config_sha256 is None)
            or (
                self.parent_candidate_id is not None
                and _SLUG.fullmatch(self.parent_candidate_id) is None
            )
            or (
                self.parent_config_sha256 is not None
                and _SHA.fullmatch(self.parent_config_sha256) is None
            )
            or (self.parent_candidate_id is None and self.candidate_id != "moe_cutlass")
            or (
                self.candidate_id == "moe_cutlass"
                and self.parent_candidate_id is not None
            )
        ):
            raise QualificationWorkerError("selection candidate identity is unsafe")

    @property
    def key(self) -> str:
        return f"{self.ordinal:02d}-{self.phase}-{self.candidate_id}"

    def receipt(self) -> dict[str, str | None]:
        return {
            "candidate_id": self.candidate_id,
            "phase": self.phase,
            "parent_candidate_id": self.parent_candidate_id,
            "parent_config_sha256": self.parent_config_sha256,
        }


@dataclass(frozen=True)
class RuntimeTuning:
    moe_runner_backend: str
    cuda_graph: str
    linear_decode_backend: str
    linear_prefill_backend: str
    replay_ssm: bool
    mamba_ssm_dtype: str
    nextn: tuple[int, int] | None
    chunked_prefill_size: int
    mem_fraction_static: str

    def __post_init__(self) -> None:
        if (
            self.moe_runner_backend
            not in runtime_contract.QUALIFICATION_MOE_RUNNER_BACKENDS
            or self.cuda_graph not in {"full", "disabled"}
            or self.linear_decode_backend not in {"triton", "cutedsl", "flashinfer"}
            or self.linear_prefill_backend not in {"triton", "cutedsl", "flashinfer"}
            or self.mamba_ssm_dtype not in {"float32", "bfloat16"}
            or self.nextn not in {None, *LEGAL_NEXTN}
            or self.chunked_prefill_size not in {4096, 8192}
            or self.mem_fraction_static not in {"0.84", "0.86", "0.88", "0.92"}
            or (
                self.replay_ssm
                and (
                    self.linear_decode_backend != "triton"
                    or self.mamba_ssm_dtype != "float32"
                )
            )
            or self.replay_ssm
            and self.linear_prefill_backend not in {"triton", "cutedsl"}
            or self.mamba_ssm_dtype == "bfloat16"
            and self.replay_ssm
            or self.linear_decode_backend == "flashinfer"
            and self.mamba_ssm_dtype != "bfloat16"
        ):
            raise QualificationWorkerError(
                "runtime tuning is outside pinned SM120 bounds"
            )

    @property
    def graph_json(self) -> str:
        if self.cuda_graph == "full":
            return (
                '{"decode":{"backend":"full","max_bs":4,"bs":[1,2,4]},'
                '"prefill":{"backend":"disabled"}}'
            )
        return '{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}'

    def receipt(self) -> dict[str, Any]:
        steps, drafts = self.nextn or (None, None)
        return {
            "served_alias": SERVED_ALIAS,
            "display_name": DISPLAY_NAME,
            "artifact_name": ARTIFACT_NAME,
            "model_architecture": runtime_contract.MODEL_ARCHITECTURE,
            "sglang_source_stack_sha256": runtime_contract.SOURCE_STACK_SHA256,
            "tp_size": 1,
            "ple_offload_embedding": True,
            "cpu_offload_gb": 0,
            "offload_group_size": -1,
            "moe_a2a_backend": "none",
            "moe_runner_backend": self.moe_runner_backend,
            "fp4_gemm_backend": runtime_contract.FP4_GEMM_BACKEND,
            "reasoning_parser": runtime_contract.REASONING_PARSER,
            "prefill_attention_backend": (runtime_contract.PREFILL_ATTENTION_BACKEND),
            "decode_attention_backend": (runtime_contract.DECODE_ATTENTION_BACKEND),
            "requested_speculative_draft_model_quantization": (
                runtime_contract.MTP_DRAFT_QUANTIZATION
            ),
            # SGLang resolves the explicit `unquant` spelling to None before
            # exposing ServerArgs.  Both requested argv and resolved state are
            # retained so draft quantization cannot silently inherit NVFP4.
            "speculative_draft_model_quantization": None,
            "speculative_moe_a2a_backend": "none",
            "speculative_moe_runner_backend": self.moe_runner_backend,
            "max_running_requests": 4,
            "max_total_tokens": runtime_contract.SM120_VALIDATED_CONTEXT_LENGTH,
            "page_size": SCHEDULER_PAGE_SIZE,
            "max_mamba_cache_size": MAX_MAMBA_CACHE_SIZE,
            "cuda_graph_config": json.loads(self.graph_json),
            "linear_attn_backend": "triton",
            "linear_attn_decode_backend": self.linear_decode_backend,
            "linear_attn_prefill_backend": self.linear_prefill_backend,
            "linear_attn_verify_backend": (
                "flashinfer" if self.linear_decode_backend == "flashinfer" else "triton"
            ),
            "enable_linear_replayssm_spec": self.replay_ssm,
            "mamba_radix_cache_strategy": ("extra_buffer" if self.replay_ssm else None),
            "ragged_verify_mode": "static",
            "runtime_environment": dict(CONSTANT_RUNTIME_ENV),
            "mamba_ssm_dtype": self.mamba_ssm_dtype,
            "chunked_prefill_size": self.chunked_prefill_size,
            "mem_fraction_static": float(self.mem_fraction_static),
            "requested_speculative_algorithm": (
                "NEXTN" if self.nextn is not None else None
            ),
            "speculative_algorithm": "EAGLE" if self.nextn is not None else None,
            "speculative_num_steps": steps,
            "speculative_eagle_topk": 1 if self.nextn is not None else None,
            "speculative_num_draft_tokens": drafts,
        }

    @classmethod
    def safe_baseline(cls) -> "RuntimeTuning":
        """Pinned SM120 baseline used before every measured single-field sweep."""

        return cls(
            moe_runner_backend=(runtime_contract.PREFERRED_MOE_RUNNER_BACKEND),
            cuda_graph="disabled",
            linear_decode_backend="triton",
            linear_prefill_backend="triton",
            replay_ssm=False,
            mamba_ssm_dtype="bfloat16",
            nextn=None,
            chunked_prefill_size=4096,
            mem_fraction_static="0.92",
        )


class QualificationWorkerError(RuntimeError):
    pass


def _validate_workload_scheduler_budgets() -> None:
    """Reject requests SGLang would silently clip below their measured cap."""

    context_length = runtime_contract.SM120_VALIDATED_CONTEXT_LENGTH
    for workload_id, (
        _concurrency,
        prompt_tokens,
        completion_tokens,
    ) in WORKLOAD_SPECS.items():
        paged_prompt_tokens = (
            -(-prompt_tokens // SCHEDULER_PAGE_SIZE) * SCHEDULER_PAGE_SIZE
        )
        completion_capacity = min(
            context_length - prompt_tokens - 2,
            context_length - paged_prompt_tokens - SCHEDULER_PAGE_SIZE - 1,
        )
        if completion_tokens > completion_capacity:
            raise QualificationWorkerError(
                f"selector workload exceeds the SGLang scheduler budget: {workload_id}"
            )


class SelectionBootFailure(QualificationWorkerError):
    """Sanitized, reviewable selector failure before a live arm identity exists."""

    def __init__(
        self,
        *,
        stage: str,
        code: str,
        detail_sha256: str,
        container_config_sha256: str | None,
    ) -> None:
        if (
            stage
            not in {
                "container_create",
                "container_start",
                "server_readiness",
                "runtime_identity_binding",
                "candidate_probe",
            }
            or _SLUG.fullmatch(code) is None
            or _SHA.fullmatch(detail_sha256) is None
            or (stage == "container_create") is not (container_config_sha256 is None)
            or (
                container_config_sha256 is not None
                and _SHA.fullmatch(container_config_sha256) is None
            )
        ):
            raise QualificationWorkerError(
                "selector boot-failure classification is malformed"
            )
        super().__init__(f"selection candidate failed during {stage}: {code}")
        self.stage = stage
        self.code = code
        self.detail_sha256 = detail_sha256
        self.container_config_sha256 = container_config_sha256


def _selection_failure(
    *,
    stage: str,
    code: str,
    error: BaseException,
    container_config_sha256: str | None,
) -> SelectionBootFailure:
    """Reduce a task-local failure to a secret-free immutable classification."""

    return SelectionBootFailure(
        stage=stage,
        code=code,
        detail_sha256=_canonical_sha(
            {
                "stage": stage,
                "code": code,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
        ),
        container_config_sha256=container_config_sha256,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _private_file(
    path: Path,
    *,
    maximum: int = 16 * 1024 * 1024,
    allow_empty: bool = False,
) -> os.stat_result:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or metadata.st_size > maximum
        or (metadata.st_size == 0 and not allow_empty)
    ):
        raise QualificationWorkerError(f"unsafe private file: {path}")
    return metadata


def _private_dir(path: Path) -> Path:
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise QualificationWorkerError(f"unsafe private directory: {path}")
    return path


def _read_json(path: Path, *, maximum: int = 16 * 1024 * 1024) -> dict[str, Any]:
    _private_file(path, maximum=maximum)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationWorkerError(f"malformed JSON: {path}") from exc
    if not isinstance(value, dict):
        raise QualificationWorkerError(f"JSON is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    _private_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        payload = (
            json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise QualificationWorkerError("receipt write was incomplete")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def validate_sibling_artifact(
    parent: Path,
    *,
    expected_tuned_tree_sha256: str,
    expected_untuned_tree_sha256: str,
    expected_manifest_sha256: str,
    verify_hashes: bool,
    require_hardlinks: bool,
) -> dict[str, Any]:
    """Validate both closed trees and their single-lm-head sibling proof."""

    parent = _private_dir(parent.resolve(strict=True))
    tuned = _private_dir(parent / TUNED_CHECKPOINT_NAME)
    untuned = _private_dir(parent / UNTUNED_CHECKPOINT_NAME)
    manifest_path = parent / SIBLING_MANIFEST_NAME
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise QualificationWorkerError("builder sibling manifest digest changed")
    manifest = _read_json(manifest_path, maximum=2 * 1024 * 1024)
    expected_manifest_fields = {
        "schema_version",
        "complete",
        "tuned_checkpoint_tree_sha256",
        "official_untuned_checkpoint_tree_sha256",
        "tuned_lm_head_tensor_sha256",
        "official_untuned_lm_head_tensor_sha256",
        "non_lm_head_tensor_inventory_sha256",
        "non_lm_head_tensors_byte_identical",
        "hardlink_identity",
    }
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("schema_version") != SIBLING_MANIFEST_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("tuned_checkpoint_tree_sha256") != expected_tuned_tree_sha256
        or manifest.get("official_untuned_checkpoint_tree_sha256")
        != expected_untuned_tree_sha256
        or manifest.get("non_lm_head_tensors_byte_identical") is not True
    ):
        raise QualificationWorkerError("builder sibling manifest contract changed")
    try:
        tuned_receipts, tuned_tree = release_tool._parse_sha256sums(
            tuned, verify=verify_hashes
        )
        untuned_receipts, untuned_tree = release_tool._parse_sha256sums(
            untuned, verify=verify_hashes
        )
    except release_tool.ReleaseError as exc:
        raise QualificationWorkerError("builder sibling tree is not closed") from exc
    if (
        tuned_tree != expected_tuned_tree_sha256
        or untuned_tree != expected_untuned_tree_sha256
    ):
        raise QualificationWorkerError("builder sibling tree digest changed")

    tuned_head = builder.TUNED_LM_HEAD_FILENAME
    untuned_head = builder.UNTUNED_LM_HEAD_FILENAME
    index_name = "model.safetensors.index.json"
    rewritten = [
        "BUILD_MANIFEST.json",
        "SHA256SUMS",
        "VALIDATION_REPORT.json",
        index_name,
        tuned_head,
        untuned_head,
    ]
    rewritten_common = {
        "BUILD_MANIFEST.json",
        "VALIDATION_REPORT.json",
        index_name,
    }
    expected_shared = set(tuned_receipts) - rewritten_common - {tuned_head}
    if (
        tuned_head not in tuned_receipts
        or untuned_head not in untuned_receipts
        or set(untuned_receipts) != expected_shared | rewritten_common | {untuned_head}
        or any(
            tuned_receipts[name] != untuned_receipts[name] for name in expected_shared
        )
    ):
        raise QualificationWorkerError("sibling file closures differ outside lm_head")

    tuned_build = _read_json(tuned / "BUILD_MANIFEST.json")
    untuned_build = _read_json(untuned / "BUILD_MANIFEST.json")
    expected_untuned_build = json.loads(
        json.dumps(tuned_build, sort_keys=True, separators=(",", ":"))
    )
    try:
        if expected_untuned_build["build"].get("checkpoint_role") != "tuned":
            raise QualificationWorkerError("tuned build checkpoint role changed")
        expected_untuned_build["status"] = (
            "official-untuned-baseline-unvalidated-canary"
        )
        expected_untuned_build["build"]["checkpoint_role"] = "official_untuned"
        behavior_source = expected_untuned_build["sources"]["behavior_adapter"]
        behavior_source["target_modules"] = []
        behavior_source["gate_status"] = "not-applied-official-untuned-baseline"
        preserved = expected_untuned_build["quantization"]["preserved"]
        if (
            not isinstance(preserved, list)
            or preserved.count("BF16 lm_head after bounded LoRA merge") != 1
        ):
            raise QualificationWorkerError("tuned build lm_head preservation changed")
        expected_untuned_build["quantization"]["preserved"] = [
            "official BF16 lm_head without adapter"
            if item == "BF16 lm_head after bounded LoRA merge"
            else item
            for item in preserved
        ]
    except (KeyError, TypeError, AttributeError) as exc:
        raise QualificationWorkerError(
            "builder sibling build manifest is malformed"
        ) from exc
    if untuned_build != expected_untuned_build:
        raise QualificationWorkerError(
            "untuned build manifest changed outside allowlist"
        )

    tuned_validation = _read_json(tuned / "VALIDATION_REPORT.json")
    untuned_validation = _read_json(untuned / "VALIDATION_REPORT.json")
    expected_untuned_validation = json.loads(
        json.dumps(tuned_validation, sort_keys=True, separators=(",", ":"))
    )
    expected_untuned_validation.update(
        {
            "lm_head_lora_merged_before_quantization": False,
            "lm_head_lora_relative_frobenius_norm": 0.0,
            "runtime_validation_status": (
                "official-untuned-baseline-unvalidated-canary"
            ),
        }
    )
    if untuned_validation != expected_untuned_validation:
        raise QualificationWorkerError(
            "untuned validation report changed outside allowlist"
        )

    tuned_index = _read_json(tuned / index_name, maximum=256 * 1024 * 1024)
    untuned_index = _read_json(untuned / index_name, maximum=256 * 1024 * 1024)
    tuned_map = tuned_index.get("weight_map")
    untuned_map = untuned_index.get("weight_map")
    if (
        not isinstance(tuned_map, Mapping)
        or not isinstance(untuned_map, Mapping)
        or set(tuned_map) != set(untuned_map)
        or tuned_map.get(builder.LM_HEAD) != tuned_head
        or untuned_map.get(builder.LM_HEAD) != untuned_head
        or any(
            tuned_map[name] != untuned_map[name]
            for name in tuned_map
            if name != builder.LM_HEAD
        )
    ):
        raise QualificationWorkerError("sibling model indexes differ outside lm_head")
    non_head_inventory = {
        name: {
            "shard": shard,
            "shard_sha256": tuned_receipts[str(shard)][0],
        }
        for name, shard in sorted(tuned_map.items())
        if name != builder.LM_HEAD
    }
    if hashlib.sha256(
        builder._canonical_json(non_head_inventory)
    ).hexdigest() != manifest.get("non_lm_head_tensor_inventory_sha256"):
        raise QualificationWorkerError("non-lm-head tensor inventory proof changed")

    def tensor_sha(path: Path) -> str:
        try:
            _metadata, records = builder._read_safetensors_header(path)
            if set(records) != {builder.LM_HEAD}:
                raise QualificationWorkerError("isolated lm_head shard changed")
            return builder._tensor_sha256(
                builder.TensorLocation(path, records[builder.LM_HEAD])
            )
        except builder.BuildError as exc:
            raise QualificationWorkerError("lm_head tensor receipt is invalid") from exc

    if (
        tensor_sha(tuned / tuned_head) != manifest.get("tuned_lm_head_tensor_sha256")
        or tensor_sha(untuned / untuned_head)
        != manifest.get("official_untuned_lm_head_tensor_sha256")
        or manifest.get("tuned_lm_head_tensor_sha256")
        == manifest.get("official_untuned_lm_head_tensor_sha256")
    ):
        raise QualificationWorkerError("tuned/official lm_head tensor proof changed")

    hardlinks = manifest.get("hardlink_identity")
    expected_hardlink_fields = {
        "shared_regular_file_count",
        "shared_unique_bytes",
        "shared_paths_sha256",
        "same_device_and_inode",
        "rewritten_allowlist",
    }
    shared_paths = sorted(expected_shared)
    shared_inodes: dict[tuple[int, int], int] = {}
    for name in shared_paths:
        tuned_metadata = (tuned / name).lstat()
        untuned_metadata = (untuned / name).lstat()
        if require_hardlinks and (
            tuned_metadata.st_dev != untuned_metadata.st_dev
            or tuned_metadata.st_ino != untuned_metadata.st_ino
        ):
            raise QualificationWorkerError("rsync did not preserve sibling hardlinks")
        shared_inodes.setdefault(
            (tuned_metadata.st_dev, tuned_metadata.st_ino), tuned_metadata.st_size
        )
    if (
        not isinstance(hardlinks, Mapping)
        or set(hardlinks) != expected_hardlink_fields
        or hardlinks.get("shared_regular_file_count") != len(shared_paths)
        or hardlinks.get("shared_unique_bytes") != sum(shared_inodes.values())
        or hardlinks.get("shared_paths_sha256")
        != hashlib.sha256(builder._canonical_json(shared_paths)).hexdigest()
        or hardlinks.get("same_device_and_inode") is not True
        or hardlinks.get("rewritten_allowlist") != rewritten
    ):
        raise QualificationWorkerError("builder hardlink identity proof changed")
    return manifest


def _paths(request: Mapping[str, Any]) -> dict[str, Path]:
    scratch = Path(str(request["scratch_path"]))
    checkpoint = Path(str(request["checkpoint_path"]))
    untuned_checkpoint = Path(str(request["official_untuned_checkpoint_path"]))
    sibling_manifest = Path(str(request["build_sibling_manifest_path"]))
    return {
        "scratch": scratch,
        "source": scratch / "source",
        "build": checkpoint.parent,
        "checkpoint": checkpoint,
        "untuned_checkpoint": untuned_checkpoint,
        "sibling_manifest": sibling_manifest,
        "assets": scratch / "assets",
        "output": scratch / "output",
        "status": scratch / "qualification-status.json",
        "spawn": scratch / "qualification-supervisor.json",
        "preflight": scratch / "qualification-preflight.json",
        "settled": scratch / "qualification-settled.json",
        "manifest": scratch / "output/MANIFEST.sha256",
    }


def _validate_request(
    path: Path, digest: str, *, require_environment: bool = False
) -> dict[str, Any]:
    _validate_workload_scheduler_budgets()
    if _SHA.fullmatch(digest) is None or _sha256(path) != digest:
        raise QualificationWorkerError("request digest changed")
    request = _read_json(path, maximum=2 * 1024 * 1024)
    required = {
        "schema_version",
        "runtime_id",
        "job_id",
        "host",
        "hostname",
        "physical_gpu",
        "gpu_uuid",
        "claim_id",
        "owner",
        "vram_budget_gb",
        "exclusive",
        "scratch_path",
        "checkpoint_path",
        "official_untuned_checkpoint_path",
        "build_sibling_manifest_path",
        "checkpoint_tree_sha256",
        "official_untuned_checkpoint_tree_sha256",
        "build_sibling_manifest_sha256",
        "builder_sha256",
        "repo_id",
        "source_files",
        "asset_files",
        "sglang_commit",
        "sglang_image_digest",
        "sglang_image_config_digest",
        "sglang_image_id",
        "sglang_image_archive_sha256",
        "task_memory_gb",
        "max_accounted_vram_gb",
        "preferred_moe_runner_backend",
        "qualification_moe_runner_backends",
        "cutlass_nvfp4_scale_duplication_bytes",
        "cutlass_min_cuda_reserve_bytes",
        "cutlass_min_geometric_mean_speedup",
    }
    scratch = PurePosixPath(str(request.get("scratch_path") or ""))
    checkpoint = PurePosixPath(str(request.get("checkpoint_path") or ""))
    untuned_checkpoint = PurePosixPath(
        str(request.get("official_untuned_checkpoint_path") or "")
    )
    sibling_manifest = PurePosixPath(
        str(request.get("build_sibling_manifest_path") or "")
    )
    build_parent = checkpoint.parent
    if (
        set(request) != required
        or request.get("schema_version") != SCHEMA
        or _RUNTIME.fullmatch(str(request.get("runtime_id") or "")) is None
        or _JOB.fullmatch(str(request.get("job_id") or "")) is None
        or request.get("host") != HOST
        or request.get("hostname") != HOSTNAME
        or os.uname().nodename != HOSTNAME
        or request.get("physical_gpu") != 0
        or _GPU.fullmatch(str(request.get("gpu_uuid") or "")) is None
        or request.get("vram_budget_gb") != VRAM_BUDGET_GB
        or request.get("exclusive") is not True
        or scratch != CANONICAL_OUTPUT_ROOT / str(request.get("runtime_id") or "")
        or not checkpoint.is_absolute()
        or not untuned_checkpoint.is_absolute()
        or not sibling_manifest.is_absolute()
        or ".." in checkpoint.parts
        or ".." in untuned_checkpoint.parts
        or ".." in sibling_manifest.parts
        or not build_parent.is_relative_to(CHECKPOINT_ROOT)
        or build_parent == CHECKPOINT_ROOT
        or checkpoint != build_parent / TUNED_CHECKPOINT_NAME
        or untuned_checkpoint != build_parent / UNTUNED_CHECKPOINT_NAME
        or sibling_manifest != build_parent / SIBLING_MANIFEST_NAME
        or request.get("sglang_commit") != SGLANG_COMMIT
        or request.get("sglang_image_digest") != IMAGE_DIGEST
        or request.get("sglang_image_config_digest") != IMAGE_CONFIG_DIGEST
        or request.get("sglang_image_id") != IMAGE_ID
        or request.get("sglang_image_archive_sha256") != IMAGE_ARCHIVE_SHA256
        or request.get("task_memory_gb") != TASK_MEMORY_GB
        or request.get("max_accounted_vram_gb") != VRAM_BUDGET_GB
        or request.get("preferred_moe_runner_backend")
        != runtime_contract.PREFERRED_MOE_RUNNER_BACKEND
        or request.get("qualification_moe_runner_backends")
        != list(runtime_contract.QUALIFICATION_MOE_RUNNER_BACKENDS)
        or request.get("cutlass_nvfp4_scale_duplication_bytes")
        != runtime_contract.CUTLASS_NVFP4_SCALE_DUPLICATION_BYTES
        or request.get("cutlass_min_cuda_reserve_bytes")
        != runtime_contract.CUTLASS_MIN_CUDA_RESERVE_BYTES
        or request.get("cutlass_min_geometric_mean_speedup")
        != runtime_contract.CUTLASS_MIN_GEOMETRIC_MEAN_SPEEDUP
        or _SHA.fullmatch(str(request.get("checkpoint_tree_sha256") or "")) is None
        or _SHA.fullmatch(
            str(request.get("official_untuned_checkpoint_tree_sha256") or "")
        )
        is None
        or _SHA.fullmatch(str(request.get("build_sibling_manifest_sha256") or ""))
        is None
        or _SHA.fullmatch(str(request.get("builder_sha256") or "")) is None
    ):
        raise QualificationWorkerError("qualification request contract changed")
    release_tool._validate_repo_id(str(request["repo_id"]))
    expected_environment = {
        "GPU_AGENT_CLAIM_ID": request["claim_id"],
        "CUDA_VISIBLE_DEVICES": request["gpu_uuid"],
        "GPU_MEM_LIMIT_GB": "88",
    }
    if require_environment and any(
        os.environ.get(key) != value for key, value in expected_environment.items()
    ):
        raise QualificationWorkerError("Fleet lease environment is not exact")
    for group, root, allowed_empty in (
        (
            request.get("source_files"),
            _paths(request)["source"],
            EMPTY_SOURCE_FILES,
        ),
        (request.get("asset_files"), _paths(request)["assets"], frozenset()),
    ):
        if not isinstance(group, Mapping) or not group:
            raise QualificationWorkerError("staged input receipt map is malformed")
        for relative, receipt in group.items():
            target = root / str(relative)
            if (
                not isinstance(relative, str)
                or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts
                or not isinstance(receipt, Mapping)
                or _private_file(
                    target,
                    allow_empty=relative in allowed_empty,
                ).st_size
                != receipt.get("size")
                or _sha256(target) != receipt.get("sha256")
            ):
                raise QualificationWorkerError("staged input identity changed")
    return request


def _docker(
    arguments: Sequence[str], *, timeout: float = 120
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [LOW_PRIORITY, DOCKER, *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            "HOME": "/home/aday",
            "PATH": "/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
        },
    )


def _docker_absent(result: subprocess.CompletedProcess[str], identity: str) -> bool:
    return (
        result.returncode == 1
        and re.search(
            rf"(?:No such object|No such container):\s*{re.escape(identity)}(?:\s|$)",
            result.stderr,
        )
        is not None
    )


def _inspect(identity: str) -> dict[str, Any] | None:
    result = _docker(["container", "inspect", identity], timeout=30)
    if _docker_absent(result, identity):
        return None
    if result.returncode != 0:
        raise QualificationWorkerError("exact container inspection failed")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise QualificationWorkerError("Docker inspection is malformed") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise QualificationWorkerError("Docker did not return one container")
    return value[0]


def _image_preflight() -> None:
    if (
        not runtime_contract.image_digest_is_settled(IMAGE_DIGEST)
        or not runtime_contract.image_config_digest_is_settled(IMAGE_CONFIG_DIGEST)
        or not runtime_contract.local_docker_image_id_is_settled(IMAGE_ID)
    ):
        raise QualificationWorkerError(
            "patched SM120 SGLang image manifest/config identities are not settled"
        )
    archive_before = IMAGE_ARCHIVE_PATH.lstat()
    if (
        not stat.S_ISREG(archive_before.st_mode)
        or archive_before.st_uid != os.geteuid()
        or archive_before.st_mode & 0o022
        or archive_before.st_nlink != 1
        or archive_before.st_size != IMAGE_ARCHIVE_SIZE_BYTES
        or _sha256(IMAGE_ARCHIVE_PATH) != IMAGE_ARCHIVE_SHA256
    ):
        raise QualificationWorkerError("pinned SGLang OCI archive changed")
    archive_after = IMAGE_ARCHIVE_PATH.lstat()
    if (
        archive_after.st_dev,
        archive_after.st_ino,
        archive_after.st_mode,
        archive_after.st_uid,
        archive_after.st_nlink,
        archive_after.st_size,
        archive_after.st_mtime_ns,
        archive_after.st_ctime_ns,
    ) != (
        archive_before.st_dev,
        archive_before.st_ino,
        archive_before.st_mode,
        archive_before.st_uid,
        archive_before.st_nlink,
        archive_before.st_size,
        archive_before.st_mtime_ns,
        archive_before.st_ctime_ns,
    ):
        raise QualificationWorkerError("pinned SGLang OCI archive changed while read")
    def load_exact_archive() -> None:
        loaded = _docker(
            ["image", "load", "--input", str(IMAGE_ARCHIVE_PATH)], timeout=3600
        )
        if loaded.returncode != 0:
            raise QualificationWorkerError("pinned SGLang OCI archive load failed")

    def inspected_image(result: subprocess.CompletedProcess[str]) -> Mapping[str, Any]:
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise QualificationWorkerError("image inspection is malformed") from exc
        image = value[0] if isinstance(value, list) and len(value) == 1 else None
        if not isinstance(image, Mapping):
            raise QualificationWorkerError("image inspection is malformed")
        return image

    def exact_manifest_identity(image: Mapping[str, Any]) -> bool:
        descriptor = image.get("Descriptor")
        repo_digests = image.get("RepoDigests")
        return (
            image.get("Id") == IMAGE_ID
            and isinstance(descriptor, Mapping)
            and descriptor.get("digest") == IMAGE_DIGEST
            and isinstance(repo_digests, list)
            and runtime_contract.QUALIFIED_IMAGE_REPO_DIGEST in repo_digests
        )

    result = _docker(["image", "inspect", IMAGE_REFERENCE], timeout=30)
    if result.returncode != 0:
        absent = (
            result.returncode == 1
            and (result.stdout or "").strip() in {"", "[]"}
            and re.search(
                rf"(?:No such image|No such object):\s*{re.escape(IMAGE_REFERENCE)}(?:\s|$)",
                result.stderr or "",
            )
            is not None
        )
        if not absent:
            raise QualificationWorkerError(
                "pinned SGLang image availability is ambiguous"
            )
        load_exact_archive()
        result = _docker(["image", "inspect", IMAGE_REFERENCE], timeout=30)
        if result.returncode != 0:
            raise QualificationWorkerError(
                "pinned SGLang image remained absent after exact OCI load"
            )
    image = inspected_image(result)
    if not exact_manifest_identity(image):
        raise QualificationWorkerError("preloaded image digest changed")
    # Docker 29's containerd image store can retain the exact OCI manifest/tag
    # across daemon or service lifecycle changes while lazily dropping its
    # platform/config projection.  In that state inspect reports the exact
    # immutable digest but Config/RootFS are empty and the image cannot be
    # qualified until its already-pinned archive is reloaded.  Reload only this
    # cryptographically exact manifest-only state; all other inconsistencies
    # remain fail-closed.
    config = image.get("Config")
    if (
        config == {}
        and image.get("RootFS") == {}
        and image.get("Architecture") in {None, ""}
        and image.get("Os") in {None, ""}
    ):
        load_exact_archive()
        reloaded = _docker(["image", "inspect", IMAGE_REFERENCE], timeout=30)
        if reloaded.returncode != 0:
            raise QualificationWorkerError(
                "pinned SGLang image disappeared after exact OCI metadata reload"
            )
        image = inspected_image(reloaded)
        if not exact_manifest_identity(image):
            raise QualificationWorkerError(
                "preloaded image digest changed after exact OCI metadata reload"
            )
    config = image.get("Config")
    mismatches = runtime_contract.validate_image_labels(
        config.get("Labels") if isinstance(config, Mapping) else None
    )
    if mismatches:
        raise QualificationWorkerError(
            "preloaded image lacks exact SM120 source/provenance labels: "
            + "; ".join(mismatches)
        )


def _validate_arm_tuning(arm: str, tuning: RuntimeTuning) -> None:
    if arm not in ARMS:
        raise QualificationWorkerError("unknown qualification arm")
    if arm in {ARM_OFFICIAL_UNTUNED, ARM_TUNED_MTP_OFF} and tuning.nextn is not None:
        raise QualificationWorkerError("canonical MTP-off arm enables NEXTN")
    if arm == ARM_TUNED_MTP_ON and tuning.nextn is None:
        raise QualificationWorkerError("canonical MTP-on arm omits NEXTN")


def _server_command(arm: str, *, model_path: str, tuning: RuntimeTuning) -> list[str]:
    _validate_arm_tuning(arm, tuning)
    command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--served-model-name",
        SERVED_ALIAS,
        "--host",
        "0.0.0.0",
        "--port",
        str(CONTAINER_PORT),
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--mamba-ssm-dtype",
        tuning.mamba_ssm_dtype,
        "--quantization",
        runtime_contract.QUANTIZATION,
        "--reasoning-parser",
        runtime_contract.REASONING_PARSER,
        "--prefill-attention-backend",
        runtime_contract.PREFILL_ATTENTION_BACKEND,
        "--decode-attention-backend",
        runtime_contract.DECODE_ATTENTION_BACKEND,
        "--speculative-draft-model-quantization",
        runtime_contract.MTP_DRAFT_QUANTIZATION,
        "--ple-offload-embedding",
        "--cpu-offload-gb",
        "0",
        "--context-length",
        str(runtime_contract.SM120_VALIDATED_CONTEXT_LENGTH),
        "--max-total-tokens",
        str(runtime_contract.SM120_VALIDATED_CONTEXT_LENGTH),
        "--page-size",
        str(SCHEDULER_PAGE_SIZE),
        "--max-mamba-cache-size",
        str(MAX_MAMBA_CACHE_SIZE),
        "--max-running-requests",
        "4",
        "--cuda-graph-config",
        tuning.graph_json,
        "--linear-attn-backend",
        "triton",
        "--linear-attn-decode-backend",
        tuning.linear_decode_backend,
        "--linear-attn-prefill-backend",
        tuning.linear_prefill_backend,
        "--linear-attn-verify-backend",
        "flashinfer" if tuning.linear_decode_backend == "flashinfer" else "triton",
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        tuning.moe_runner_backend,
        "--fp4-gemm-backend",
        runtime_contract.FP4_GEMM_BACKEND,
        "--speculative-moe-a2a-backend",
        "none",
        "--speculative-moe-runner-backend",
        tuning.moe_runner_backend,
        "--chunked-prefill-size",
        str(tuning.chunked_prefill_size),
        "--mem-fraction-static",
        tuning.mem_fraction_static,
    ]
    if tuning.replay_ssm:
        command.extend(
            [
                "--enable-linear-replayssm-spec",
                "--mamba-radix-cache-strategy",
                "extra_buffer",
            ]
        )
    if tuning.nextn is not None:
        steps, drafts = tuning.nextn
        command.extend(
            [
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                str(steps),
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                str(drafts),
            ]
        )
    return command


def _runtime_config(arm: str, tuning: RuntimeTuning) -> dict[str, Any]:
    _validate_arm_tuning(arm, tuning)
    return tuning.receipt()


def _checkpoint_for_arm(request: Mapping[str, Any], arm: str) -> tuple[Path, str]:
    if arm == ARM_OFFICIAL_UNTUNED:
        return (
            _paths(request)["untuned_checkpoint"],
            str(request["official_untuned_checkpoint_tree_sha256"]),
        )
    if arm in {ARM_TUNED_MTP_OFF, ARM_SELECTION, ARM_TUNED_MTP_ON}:
        return _paths(request)["checkpoint"], str(request["checkpoint_tree_sha256"])
    raise QualificationWorkerError("unknown qualification arm")


def _labels(
    request: Mapping[str, Any],
    arm: str,
    command: Sequence[str],
    candidate: CandidateIdentity | None = None,
) -> dict[str, str]:
    _checkpoint, checkpoint_tree = _checkpoint_for_arm(request, arm)
    labels = {
        "aeon.fleet.profile": "aeon-qwen38-flash-next-qualification",
        "aeon.fleet.runtime": str(request["runtime_id"]),
        "aeon.fleet.request": _sha256(
            Path(str(request["scratch_path"])) / "qualification-request.json"
        ),
        "aeon.fleet.arm": arm,
        "aeon.fleet.checkpoint": checkpoint_tree,
        "aeon.fleet.command": _canonical_sha(list(command)),
    }
    if candidate is not None:
        labels["aeon.fleet.selection"] = _canonical_sha(candidate.receipt())
    return labels


def _container_name(
    request: Mapping[str, Any], arm: str, candidate: CandidateIdentity | None = None
) -> str:
    suffix = candidate.key if candidate is not None else arm.replace("_", "-")
    return f"aeon-flash-qual-{request['runtime_id']}-{suffix}"


def _supervisor_command(
    request: Mapping[str, Any], arm: str, command: Sequence[str], checkpoint_tree: str
) -> list[str]:
    return [
        "python3",
        "/qualification/supervisor.py",
        "--output",
        "/evidence/cuda-memory.json",
        "--freeze",
        "/evidence/freeze",
        "--context",
        "/evidence/runtime-context.json",
        "--runtime-id",
        str(request["runtime_id"]),
        "--arm",
        arm,
        "--claim-sha256",
        hashlib.sha256(str(request["claim_id"]).encode("utf-8")).hexdigest(),
        "--gpu-uuid",
        str(request["gpu_uuid"]),
        "--checkpoint-tree-sha256",
        checkpoint_tree,
        "--",
        *command,
    ]


def _create(
    request: Mapping[str, Any],
    arm: str,
    tuning: RuntimeTuning,
    candidate: CandidateIdentity | None = None,
) -> tuple[str, list[str], dict[str, Path]]:
    if (arm == ARM_SELECTION) is not (candidate is not None):
        raise QualificationWorkerError("selection arm/candidate identity mismatch")
    name = _container_name(request, arm, candidate)
    if _inspect(name) is not None:
        raise QualificationWorkerError("qualification container name already exists")
    command = _server_command(arm, model_path="/model", tuning=tuning)
    checkpoint, checkpoint_tree = _checkpoint_for_arm(request, arm)
    evidence_name = candidate.key if candidate is not None else arm
    evidence = _paths(request)["scratch"] / "arm-evidence" / evidence_name
    evidence.mkdir(mode=0o700, parents=True, exist_ok=False)
    evidence_paths = {
        "root": evidence,
        "attestation": evidence / "cuda-memory.json",
        "freeze": evidence / "freeze",
        "context": evidence / "runtime-context.json",
    }
    supervisor = _paths(request)["source"] / (
        "aeon/scripts/qwen_flash_next_container_supervisor.py"
    )
    supervisor_command = _supervisor_command(request, arm, command, checkpoint_tree)
    labels = _labels(request, arm, command, candidate)
    arguments = [
        "container",
        "create",
        "--pull=never",
        "--name",
        name,
        "--user",
        f"{os.geteuid()}:{os.getegid()}",
        "--gpus",
        f"device={request['gpu_uuid']}",
        "--memory",
        f"{TASK_MEMORY_BYTES}b",
        "--memory-swap",
        f"{TASK_MEMORY_BYTES}b",
        "--shm-size",
        "32g",
        "--pids-limit",
        "4096",
        "--ulimit",
        "memlock=-1:-1",
        "--security-opt",
        "no-new-privileges=true",
        "--publish",
        f"127.0.0.1:{HOST_PORT}:{CONTAINER_PORT}",
        "--mount",
        f"type=bind,src={checkpoint},dst=/model,readonly",
        "--mount",
        f"type=bind,src={supervisor},dst=/qualification/supervisor.py,readonly",
        "--mount",
        f"type=bind,src={evidence},dst=/evidence",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,exec,size=8g",
    ]
    for key, value in sorted(labels.items()):
        arguments.extend(("--label", f"{key}={value}"))
    runtime_env = {
        **CONSTANT_RUNTIME_ENV,
        "CUDA_VISIBLE_DEVICES": str(request["gpu_uuid"]),
        "GPU_AGENT_CLAIM_ID": str(request["claim_id"]),
        "GPU_MEM_LIMIT_GB": "88",
        "HOME": "/tmp",
        "HF_HOME": "/tmp/huggingface",
    }
    for key, value in sorted(runtime_env.items()):
        arguments.extend(("--env", f"{key}={value}"))
    arguments.extend((IMAGE_REFERENCE, *supervisor_command))
    result = _docker(arguments, timeout=180)
    container_id = result.stdout.strip()
    if result.returncode != 0 or _CONTAINER.fullmatch(container_id) is None:
        if _inspect(name) is None:
            error = QualificationWorkerError("Docker create failed with exact absence")
            if arm == ARM_SELECTION:
                raise _selection_failure(
                    stage="container_create",
                    code="docker_create_exact_absence",
                    error=error,
                    container_config_sha256=None,
                ) from error
            raise error
        raise QualificationWorkerError("Docker create result is ambiguous")
    return container_id, command, evidence_paths


def _verify_container(
    request: Mapping[str, Any],
    arm: str,
    container_id: str,
    command: Sequence[str],
    evidence: Mapping[str, Path],
    tuning: RuntimeTuning,
    candidate: CandidateIdentity | None = None,
    *,
    require_running_network: bool = True,
) -> dict[str, Any]:
    item = _inspect(container_id)
    if item is None:
        raise QualificationWorkerError("qualification container disappeared")
    checkpoint, checkpoint_tree = _checkpoint_for_arm(request, arm)
    config = item.get("Config")
    host_config = item.get("HostConfig")
    state = item.get("State")
    network = item.get("NetworkSettings")
    if not all(
        isinstance(value, Mapping) for value in (config, host_config, state, network)
    ):
        raise QualificationWorkerError("container inspect contract is malformed")
    observed_labels = config.get("Labels")
    expected_labels = _labels(request, arm, command, candidate)
    if (
        item.get("Id") != container_id
        or item.get("Name") != f"/{_container_name(request, arm, candidate)}"
        or config.get("Image") != IMAGE_REFERENCE
        or config.get("User") != f"{os.geteuid()}:{os.getegid()}"
        or config.get("Cmd")
        != _supervisor_command(request, arm, command, checkpoint_tree)
        or not isinstance(observed_labels, Mapping)
        or any(
            observed_labels.get(key) != value for key, value in expected_labels.items()
        )
        or host_config.get("Memory") != TASK_MEMORY_BYTES
        or host_config.get("MemorySwap") != TASK_MEMORY_BYTES
        or host_config.get("ShmSize") != 32 * 1024**3
        or host_config.get("PidsLimit") != 4096
        or host_config.get("Ulimits")
        != [{"Name": "memlock", "Hard": -1, "Soft": -1}]
        or host_config.get("SecurityOpt") != ["no-new-privileges=true"]
        or host_config.get("PortBindings")
        != {
            f"{CONTAINER_PORT}/tcp": [
                {"HostIp": "127.0.0.1", "HostPort": str(HOST_PORT)}
            ]
        }
    ):
        raise QualificationWorkerError("container runtime identity changed")
    env = config.get("Env")
    if not isinstance(env, list) or not all(isinstance(value, str) for value in env):
        raise QualificationWorkerError("container environment is malformed")
    parsed_env: dict[str, list[str]] = {}
    for value in env:
        key, separator, field = value.partition("=")
        if separator:
            parsed_env.setdefault(key, []).append(field)
    expected_env = {
        **CONSTANT_RUNTIME_ENV,
        "CUDA_VISIBLE_DEVICES": str(request["gpu_uuid"]),
        "GPU_AGENT_CLAIM_ID": str(request["claim_id"]),
        "GPU_MEM_LIMIT_GB": "88",
        "HOME": "/tmp",
        "HF_HOME": "/tmp/huggingface",
    }
    if any(parsed_env.get(key) != [value] for key, value in expected_env.items()):
        raise QualificationWorkerError("container lease environment changed")
    device_requests = host_config.get("DeviceRequests")
    if (
        not isinstance(device_requests, list)
        or len(device_requests) != 1
        or not isinstance(device_requests[0], Mapping)
        or device_requests[0].get("DeviceIDs") != [request["gpu_uuid"]]
        or device_requests[0].get("Capabilities") != [["gpu"]]
    ):
        raise QualificationWorkerError("container GPU UUID binding changed")
    mounts = item.get("Mounts")
    if not isinstance(mounts, list):
        raise QualificationWorkerError("container mount identity is malformed")
    observed_mounts = {
        mount.get("Destination"): (mount.get("Source"), mount.get("RW"))
        for mount in mounts
        if isinstance(mount, Mapping)
    }
    expected_mounts = {
        "/model": (str(checkpoint), False),
        "/qualification/supervisor.py": (
            str(
                _paths(request)["source"]
                / "aeon/scripts/qwen_flash_next_container_supervisor.py"
            ),
            False,
        ),
        "/evidence": (str(evidence["root"]), True),
    }
    if observed_mounts != expected_mounts:
        raise QualificationWorkerError("container checkpoint/evidence mounts changed")
    ports = network.get("Ports")
    bindings = (
        ports.get(f"{CONTAINER_PORT}/tcp") if isinstance(ports, Mapping) else None
    )
    if require_running_network and bindings != [
        {"HostIp": "127.0.0.1", "HostPort": str(HOST_PORT)}
    ]:
        raise QualificationWorkerError("container loopback endpoint binding changed")
    return item


def _container_config_projection(
    request: Mapping[str, Any],
    arm: str,
    command: Sequence[str],
    evidence: Mapping[str, Path],
    item: Mapping[str, Any],
    candidate: CandidateIdentity | None = None,
) -> dict[str, Any]:
    """Return the secret-free exact Docker projection available after create."""

    config = item.get("Config")
    host_config = item.get("HostConfig")
    network = item.get("NetworkSettings")
    if not all(isinstance(value, Mapping) for value in (config, host_config, network)):
        raise QualificationWorkerError("container binding projection is malformed")
    checkpoint, checkpoint_tree = _checkpoint_for_arm(request, arm)
    supervisor = (
        _paths(request)["source"]
        / "aeon/scripts/qwen_flash_next_container_supervisor.py"
    )
    expected_labels = _labels(request, arm, command, candidate)
    device_requests = host_config.get("DeviceRequests")
    mounts = item.get("Mounts")
    if not isinstance(device_requests, list) or not isinstance(mounts, list):
        raise QualificationWorkerError("container binding inputs are malformed")
    bindings = host_config.get("PortBindings")
    return {
        "schema_version": "aeon-qwen38-flash-next-container-config-v1",
        "image": config.get("Image"),
        "user": config.get("User"),
        "sglang_command_sha256": _canonical_sha(list(command)),
        "supervisor_command_sha256": _canonical_sha(config.get("Cmd")),
        "runtime_environment": dict(CONSTANT_RUNTIME_ENV),
        "lease_environment_sha256": _canonical_sha(
            {
                "CUDA_VISIBLE_DEVICES": request["gpu_uuid"],
                "GPU_AGENT_CLAIM_ID": request["claim_id"],
                "GPU_MEM_LIMIT_GB": "88",
            }
        ),
        "lease_claim_id_sha256": hashlib.sha256(
            str(request["claim_id"]).encode("utf-8")
        ).hexdigest(),
        "leased_gpu_uuid_sha256": hashlib.sha256(
            str(request["gpu_uuid"]).encode("utf-8")
        ).hexdigest(),
        "gpu_device_request_sha256": _canonical_sha(device_requests),
        "task_cgroup": {
            "memory_bytes": host_config.get("Memory"),
            "memory_swap_bytes": host_config.get("MemorySwap"),
            "shm_bytes": host_config.get("ShmSize"),
            "pids_limit": host_config.get("PidsLimit"),
            "security_opt": host_config.get("SecurityOpt"),
        },
        "mounts": {
            "checkpoint": {
                "destination": "/model",
                "read_only": True,
                "source_path_sha256": hashlib.sha256(
                    str(checkpoint).encode("utf-8")
                ).hexdigest(),
                "checkpoint_tree_sha256": checkpoint_tree,
            },
            "supervisor": {
                "destination": "/qualification/supervisor.py",
                "read_only": True,
                "source_path_sha256": hashlib.sha256(
                    str(supervisor).encode("utf-8")
                ).hexdigest(),
                "source_sha256": _sha256(supervisor),
            },
            "evidence": {
                "destination": "/evidence",
                "read_only": False,
                "source_path_sha256": hashlib.sha256(
                    str(evidence["root"]).encode("utf-8")
                ).hexdigest(),
            },
        },
        "loopback_endpoint": {
            "host": "127.0.0.1",
            "host_port": HOST_PORT,
            "container_port": CONTAINER_PORT,
            "binding_sha256": _canonical_sha(bindings),
        },
        "labels_sha256": _canonical_sha(expected_labels),
    }


def _selection_failure_sensitive_values(
    request: Mapping[str, Any],
    container_id: str,
) -> tuple[str, ...]:
    values = {
        str(request.get(field) or "")
        for field in (
            "claim_id",
            "gpu_uuid",
            "scratch_path",
            "checkpoint_path",
            "official_untuned_checkpoint_path",
            "build_sibling_manifest_path",
        )
    }
    values.add(container_id)
    return tuple(sorted((value for value in values if value), key=len, reverse=True))


def _sanitize_selection_failure_tail(
    value: str,
    *,
    request: Mapping[str, Any],
    container_id: str,
    maximum_bytes: int,
) -> dict[str, Any]:
    """Return a bounded UTF-8 tail with task-private identifiers removed."""

    sanitized = _ANSI_ESCAPE.sub("", value)
    for sensitive in _selection_failure_sensitive_values(request, container_id):
        sanitized = sanitized.replace(sensitive, "[redacted-task-identity]")
    sanitized = _RAW_GPU_UUID.sub("[redacted-gpu]", sanitized)
    sanitized = _RAW_CLAIM.sub("[redacted-claim]", sanitized)
    sanitized = _RAW_HF_TOKEN.sub("[redacted-hf-token]", sanitized)
    sanitized = _RAW_BEARER.sub("[redacted-bearer]", sanitized)
    sanitized = _RAW_HOME_PATH.sub("[redacted-host-path]", sanitized)
    sanitized = "".join(
        character
        if character in {"\n", "\t"} or (ord(character) >= 32 and ord(character) != 127)
        else "\ufffd"
        for character in sanitized
    )
    encoded = sanitized.encode("utf-8")
    truncated = len(encoded) > maximum_bytes
    if truncated:
        sanitized = encoded[-maximum_bytes:].decode("utf-8", errors="ignore")
        encoded = sanitized.encode("utf-8")
    return {
        "tail": sanitized,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "utf8_bytes": len(encoded),
        "truncated": truncated,
    }


def _selection_failure_tail_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sha256": value["sha256"],
        "utf8_bytes": value["utf8_bytes"],
        "truncated": value["truncated"],
    }


def _selection_failure_docker_state(
    state: Any,
    *,
    request: Mapping[str, Any],
    container_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(state, Mapping):
        raise QualificationWorkerError("selector Docker state is malformed")
    error = state.get("Error")
    if not isinstance(error, str):
        raise QualificationWorkerError("selector Docker state error is malformed")
    error_tail = _sanitize_selection_failure_tail(
        error,
        request=request,
        container_id=container_id,
        maximum_bytes=qualify.MAX_SELECTION_DOCKER_STATE_ERROR_BYTES,
    )
    summary = {
        "status": state.get("Status"),
        "running": state.get("Running"),
        "paused": state.get("Paused"),
        "restarting": state.get("Restarting"),
        "oom_killed": state.get("OOMKilled"),
        "dead": state.get("Dead"),
        "pid": state.get("Pid"),
        "exit_code": state.get("ExitCode"),
        "error": _selection_failure_tail_summary(error_tail),
        "started_at": state.get("StartedAt"),
        "finished_at": state.get("FinishedAt"),
    }
    qualify._validate_selection_docker_state_summary(summary)
    return summary, error_tail


def _assert_selection_failure_tail_is_sanitized(
    value: str, *, request: Mapping[str, Any], container_id: str
) -> None:
    if any(
        sensitive in value
        for sensitive in _selection_failure_sensitive_values(request, container_id)
    ) or any(
        pattern.search(value) is not None
        for pattern in (
            _RAW_GPU_UUID,
            _RAW_CLAIM,
            _RAW_HF_TOKEN,
            _RAW_BEARER,
            _RAW_HOME_PATH,
        )
    ):
        raise QualificationWorkerError(
            "selector Docker diagnostic contains unsanitized private data"
        )


def _selection_docker_failure_sidecar_path(
    request: Mapping[str, Any], candidate: CandidateIdentity
) -> Path:
    return _paths(request)["output"] / f"{candidate.key}.docker-failure.json"


def _validate_selection_docker_failure_sidecar(
    path: Path,
    *,
    request: Mapping[str, Any],
    candidate: CandidateIdentity,
    failure_stage: str,
    failure_code: str,
    failure_detail_sha256: str,
    command_sha256: str,
    container_config_sha256: str,
) -> dict[str, Any]:
    metadata = _private_file(
        path, maximum=qualify.MAX_SELECTION_DOCKER_FAILURE_SIDECAR_BYTES
    )
    value = _read_json(path, maximum=qualify.MAX_SELECTION_DOCKER_FAILURE_SIDECAR_BYTES)
    expected = {
        "schema_version",
        "runtime_id",
        "ordered_index",
        "selection_candidate",
        "failure_stage",
        "failure_code",
        "failure_detail_sha256",
        "container_id_sha256",
        "command_sha256",
        "container_config_sha256",
        "captured_at",
        "docker_logs_exit_code",
        "docker_state",
        "docker_state_sha256",
        "docker_state_error",
        "stdout",
        "stderr",
    }
    if (
        set(value) != expected
        or value.get("schema_version")
        != qualify.SELECTION_DOCKER_FAILURE_SIDECAR_SCHEMA_VERSION
        or value.get("runtime_id") != request["runtime_id"]
        or value.get("ordered_index") != candidate.ordinal
        or value.get("selection_candidate") != candidate.receipt()
        or value.get("failure_stage") != failure_stage
        or value.get("failure_code") != failure_code
        or value.get("failure_detail_sha256") != failure_detail_sha256
        or value.get("command_sha256") != command_sha256
        or value.get("container_config_sha256") != container_config_sha256
        or value.get("docker_logs_exit_code") != 0
    ):
        raise QualificationWorkerError("selector Docker diagnostic identity changed")
    container_id_sha256 = value.get("container_id_sha256")
    if (
        not isinstance(container_id_sha256, str)
        or _SHA.fullmatch(container_id_sha256) is None
    ):
        raise QualificationWorkerError(
            "selector Docker diagnostic container identity is malformed"
        )
    streams: dict[str, dict[str, Any]] = {}
    for name, maximum in (
        ("stdout", qualify.MAX_SELECTION_DOCKER_LOG_TAIL_BYTES),
        ("stderr", qualify.MAX_SELECTION_DOCKER_LOG_TAIL_BYTES),
        ("docker_state_error", qualify.MAX_SELECTION_DOCKER_STATE_ERROR_BYTES),
    ):
        stream = value.get(name)
        if not isinstance(stream, Mapping) or set(stream) != {
            "tail",
            "sha256",
            "utf8_bytes",
            "truncated",
        }:
            raise QualificationWorkerError(
                f"selector Docker diagnostic {name} is malformed"
            )
        tail = stream.get("tail")
        if not isinstance(tail, str):
            raise QualificationWorkerError(
                f"selector Docker diagnostic {name} is malformed"
            )
        encoded = tail.encode("utf-8")
        if (
            len(encoded) > maximum
            or stream.get("utf8_bytes") != len(encoded)
            or stream.get("sha256") != hashlib.sha256(encoded).hexdigest()
            or not isinstance(stream.get("truncated"), bool)
        ):
            raise QualificationWorkerError(
                f"selector Docker diagnostic {name} digest changed"
            )
        _assert_selection_failure_tail_is_sanitized(
            tail, request=request, container_id=""
        )
        streams[name] = dict(stream)
    state = value.get("docker_state")
    if not isinstance(state, Mapping):
        raise QualificationWorkerError("selector Docker diagnostic state is malformed")
    state = dict(state)
    if state.get("error") != _selection_failure_tail_summary(
        streams["docker_state_error"]
    ):
        raise QualificationWorkerError("selector Docker state error digest changed")
    qualify._validate_selection_docker_state_summary(state)
    if value.get("docker_state_sha256") != _canonical_sha(state):
        raise QualificationWorkerError("selector Docker state digest changed")
    summary = {
        "schema_version": (qualify.SELECTION_DOCKER_FAILURE_SUMMARY_SCHEMA_VERSION),
        "sidecar_name": path.name,
        "sidecar_sha256": _sha256(path),
        "sidecar_size_bytes": metadata.st_size,
        "failure_stage": failure_stage,
        "failure_code": failure_code,
        "failure_detail_sha256": failure_detail_sha256,
        "container_id_sha256": container_id_sha256,
        "command_sha256": command_sha256,
        "container_config_sha256": container_config_sha256,
        "captured_at": value["captured_at"],
        "docker_logs_exit_code": 0,
        "docker_state": state,
        "docker_state_sha256": value["docker_state_sha256"],
        "stdout": _selection_failure_tail_summary(streams["stdout"]),
        "stderr": _selection_failure_tail_summary(streams["stderr"]),
    }
    qualify._validate_selection_docker_failure_summary(
        summary,
        sidecar_stem=candidate.key,
        failure_stage=failure_stage,
        failure_code=failure_code,
        failure_detail_sha256=failure_detail_sha256,
        command_sha256=command_sha256,
        container_config_sha256=container_config_sha256,
    )
    return summary


def _persist_selection_docker_failure(
    request: Mapping[str, Any],
    *,
    container_id: str,
    command: Sequence[str],
    evidence: Mapping[str, Path],
    tuning: RuntimeTuning,
    candidate: CandidateIdentity,
    failure: SelectionBootFailure,
) -> dict[str, Any]:
    if failure.container_config_sha256 is None:
        raise QualificationWorkerError(
            "post-create selector failure lacks its container binding"
        )
    path = _selection_docker_failure_sidecar_path(request, candidate)
    if path.exists() or path.is_symlink():
        raise QualificationWorkerError(
            "selector Docker diagnostic sidecar already exists"
        )
    before = _verify_container(
        request,
        ARM_SELECTION,
        container_id,
        command,
        evidence,
        tuning,
        candidate,
        require_running_network=False,
    )
    before_config_sha256 = _canonical_sha(
        _container_config_projection(
            request,
            ARM_SELECTION,
            command,
            evidence,
            before,
            candidate,
        )
    )
    if (
        before.get("Id") != container_id
        or before_config_sha256 != failure.container_config_sha256
    ):
        raise QualificationWorkerError(
            "selector Docker diagnostic container binding changed"
        )
    logs = _docker(
        [
            "container",
            "logs",
            "--tail",
            str(SELECTION_DOCKER_LOG_TAIL_LINES),
            container_id,
        ],
        timeout=30,
    )
    if logs.returncode != 0:
        raise QualificationWorkerError("selector Docker log capture failed")
    after = _verify_container(
        request,
        ARM_SELECTION,
        container_id,
        command,
        evidence,
        tuning,
        candidate,
        require_running_network=False,
    )
    after_config_sha256 = _canonical_sha(
        _container_config_projection(
            request,
            ARM_SELECTION,
            command,
            evidence,
            after,
            candidate,
        )
    )
    if (
        after.get("Id") != container_id
        or after_config_sha256 != failure.container_config_sha256
    ):
        raise QualificationWorkerError(
            "selector Docker diagnostic container changed during capture"
        )
    docker_state, docker_state_error = _selection_failure_docker_state(
        after.get("State"), request=request, container_id=container_id
    )
    stdout = _sanitize_selection_failure_tail(
        logs.stdout or "",
        request=request,
        container_id=container_id,
        maximum_bytes=qualify.MAX_SELECTION_DOCKER_LOG_TAIL_BYTES,
    )
    stderr = _sanitize_selection_failure_tail(
        logs.stderr or "",
        request=request,
        container_id=container_id,
        maximum_bytes=qualify.MAX_SELECTION_DOCKER_LOG_TAIL_BYTES,
    )
    command_sha256 = _canonical_sha(list(command))
    sidecar = {
        "schema_version": qualify.SELECTION_DOCKER_FAILURE_SIDECAR_SCHEMA_VERSION,
        "runtime_id": request["runtime_id"],
        "ordered_index": candidate.ordinal,
        "selection_candidate": candidate.receipt(),
        "failure_stage": failure.stage,
        "failure_code": failure.code,
        "failure_detail_sha256": failure.detail_sha256,
        "container_id_sha256": hashlib.sha256(container_id.encode("ascii")).hexdigest(),
        "command_sha256": command_sha256,
        "container_config_sha256": failure.container_config_sha256,
        "captured_at": _now(),
        "docker_logs_exit_code": logs.returncode,
        "docker_state": docker_state,
        "docker_state_sha256": _canonical_sha(docker_state),
        "docker_state_error": docker_state_error,
        "stdout": stdout,
        "stderr": stderr,
    }
    _atomic_json(path, sidecar)
    return _validate_selection_docker_failure_sidecar(
        path,
        request=request,
        candidate=candidate,
        failure_stage=failure.stage,
        failure_code=failure.code,
        failure_detail_sha256=failure.detail_sha256,
        command_sha256=command_sha256,
        container_config_sha256=failure.container_config_sha256,
    )


def _runtime_config_binding(
    request: Mapping[str, Any],
    arm: str,
    command: Sequence[str],
    evidence: Mapping[str, Path],
    tuning: RuntimeTuning,
    item: Mapping[str, Any],
    candidate: CandidateIdentity | None = None,
) -> dict[str, Any]:
    """Bind the exact argv/container configuration and live SGLang readback.

    Raw lease claims, GPU UUIDs, and host paths stay inside this worker.  The
    persisted projection contains only their SHA-256 digests, so qualification
    and release evidence can prove exact placement without publishing hardware
    identifiers or private worker paths.
    """

    projection = _container_config_projection(
        request,
        arm,
        command,
        evidence,
        item,
        candidate,
    )
    client = qualify.EndpointClient(
        f"http://127.0.0.1:{HOST_PORT}", api_key=None, timeout_seconds=30
    )
    live_raw, _live_sha = client.get_json("/server_info")
    live = qualify._sanitize_server_info(
        live_raw,
        arm=arm,
        mtp_settings=tuning.nextn,
        mamba_ssm_dtype=tuning.mamba_ssm_dtype,
    )
    live_fields = sorted(qualify.RUNTIME_CONFIG_FIELDS & set(live))
    return {
        "command_sha256": _canonical_sha(list(command)),
        "container_config_sha256": _canonical_sha(projection),
        "live_server_info_fields": live_fields,
        "unexposed_server_info_fields": sorted(
            qualify.RUNTIME_CONFIG_FIELDS - set(live_fields)
        ),
    }


def _cgroup(pid: int, container_id: str) -> Path:
    lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii").splitlines()
    unified = [line.split(":", 2)[2] for line in lines if line.startswith("0::")]
    if len(unified) != 1:
        raise QualificationWorkerError("container has no exact cgroup-v2 path")
    relative = PurePosixPath(unified[0])
    if (
        not relative.is_absolute()
        or ".." in relative.parts
        or not any(
            container_id in part or container_id[:12] in part for part in relative.parts
        )
    ):
        raise QualificationWorkerError("container task cgroup identity changed")
    path = Path("/sys/fs/cgroup").joinpath(*relative.parts[1:]).resolve(strict=True)
    if int((path / "memory.max").read_text(encoding="ascii")) != TASK_MEMORY_BYTES:
        raise QualificationWorkerError("container task cgroup memory limit changed")
    return path


def _memory_events(cgroup: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in (cgroup / "memory.events").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[1].isdigit() or fields[0] in result:
            raise QualificationWorkerError("task cgroup memory.events is malformed")
        result[fields[0]] = int(fields[1])
    if any(result.get(name) != 0 for name in ("max", "oom", "oom_kill")):
        raise QualificationWorkerError("task cgroup is not fresh and event-free")
    return result


def _wait_ready(container_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        item = _inspect(container_id)
        if item is None or item.get("State", {}).get("Running") is not True:
            raise QualificationWorkerError("qualification container exited during load")
        try:
            health = requests.get(
                f"http://127.0.0.1:{HOST_PORT}/health", timeout=(2, 10)
            )
            models = requests.get(
                f"http://127.0.0.1:{HOST_PORT}/v1/models", timeout=(2, 10)
            )
            value = models.json()
            identities = {
                row.get("id")
                for row in value.get("data", [])
                if isinstance(row, Mapping)
            }
            if (
                health.status_code == 200
                and models.status_code == 200
                and identities == {SERVED_ALIAS}
            ):
                return item
        except (requests.RequestException, ValueError, TypeError):
            pass
        time.sleep(5)
    raise QualificationWorkerError("qualification container did not become ready")


def _endpoint_json(
    path: str, payload: Mapping[str, Any], *, timeout: float = 900.0
) -> dict[str, Any]:
    try:
        response = requests.post(
            f"http://127.0.0.1:{HOST_PORT}{path}",
            json=dict(payload),
            headers={"Accept": "application/json"},
            timeout=(10, timeout),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise QualificationWorkerError(f"endpoint {path} request failed") from exc
    if not 0 < len(response.content) <= MAX_ENDPOINT_JSON_BYTES:
        raise QualificationWorkerError(f"endpoint {path} response exceeds its bound")
    try:
        value = response.json()
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QualificationWorkerError(
            f"endpoint {path} response is malformed"
        ) from exc
    if not isinstance(value, dict):
        raise QualificationWorkerError(f"endpoint {path} response is not an object")
    return value


def _tokenize_messages(
    messages: list[dict[str, str]], *, allow_oversized_search: bool = False
) -> tuple[list[int], int]:
    context_length = runtime_contract.SM120_VALIDATED_CONTEXT_LENGTH
    token_limit = context_length * (
        PROMPT_SEARCH_CONTEXT_MULTIPLIER if allow_oversized_search else 1
    )
    value = _endpoint_json(
        "/v1/tokenize",
        {
            "model": SERVED_ALIAS,
            "messages": messages,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    tokens = value.get("tokens")
    count = value.get("count")
    max_model_len = value.get("max_model_len")
    if (
        not isinstance(tokens, list)
        or not tokens
        or isinstance(count, bool)
        or not isinstance(count, int)
        or len(tokens) != count
        or len(tokens) > token_limit
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < 2**31
            for token in tokens
        )
        or isinstance(max_model_len, bool)
        or not isinstance(max_model_len, int)
        or max_model_len < context_length
    ):
        raise QualificationWorkerError("live chat tokenizer response is invalid")
    return tokens, max_model_len


def _detokenize(tokens: Sequence[int]) -> str:
    value = _endpoint_json(
        "/v1/detokenize",
        {
            "model": SERVED_ALIAS,
            "tokens": list(tokens),
            "skip_special_tokens": False,
        },
    )
    text = value.get("text")
    if (
        not isinstance(text, str)
        or not text
        or len(text.encode("utf-8")) > MAX_ENDPOINT_JSON_BYTES
    ):
        raise QualificationWorkerError("live detokenized prompt is malformed")
    return text


def _needle_nonce(workload_id: str) -> str:
    suffix = hashlib.sha256(workload_id.encode("ascii")).hexdigest()[:16].upper()
    return f"AEON_NEEDLE_{suffix}"


def _prompt_text(
    workload_id: str,
    request_index: int,
    *,
    record_count: int,
    padding_count: int,
    padding_unit: str,
) -> str:
    if record_count < 0 or padding_count < 0:
        raise QualificationWorkerError("prompt filler geometry is negative")
    if workload_id == "b1_512_512":
        topic = (
            "immutable checksums, atomic release receipts, and reversible publication"
        )
    elif workload_id == "c4_512_512":
        topic = C4_TOPICS[request_index]
    else:
        topic = "deterministic engineering validation records"
    prefix = (
        f"This is the reviewed Aeon workload {workload_id}, request {request_index}. "
        f"Read the numbered neutral records about {topic}.\n"
    )
    suffix = (
        "\nWrite a continuous technical narrative grounded in the records and continue "
        "until the response limit."
    )
    needle: str | None = None
    needle_index = -1
    if workload_id.startswith("needle_"):
        needle = _needle_nonce(workload_id)
        offsets = (0.25, 0.50, 0.75)
        offset = offsets[
            int(hashlib.sha256(workload_id.encode("ascii")).hexdigest()[:2], 16) % 3
        ]
        needle_index = min(max(0, int(record_count * offset)), max(0, record_count - 1))
        suffix = (
            "\nWhat is the exact hidden verification value? Reply with that value only, "
            "with no punctuation or explanation."
        )
    records: list[str] = []
    for index in range(record_count):
        if needle is not None and index == needle_index:
            records.append(
                f"Record {index:06d}: hidden verification value is {needle}."
            )
        else:
            records.append(f"Record {index:06d}: neutral validated datum for {topic}.")
    padding = padding_unit * padding_count
    return prefix + "\n".join(records) + padding + suffix


@dataclass(frozen=True)
class PromptMaterial:
    messages: tuple[dict[str, str], ...]
    tokens: tuple[int, ...]
    input_ids_sha256: str
    rendered_prompt_sha256: str
    max_model_len: int


def _fit_prompt(
    workload_id: str, request_index: int, target_tokens: int
) -> PromptMaterial:
    """Build a deterministic chat prompt and prove its exact live-template length."""

    context_length = runtime_contract.SM120_VALIDATED_CONTEXT_LENGTH
    if (
        isinstance(target_tokens, bool)
        or not isinstance(target_tokens, int)
        or not 0 < target_tokens <= context_length
    ):
        raise QualificationWorkerError(
            "prompt target is outside the validated context length"
        )

    def material(
        record_count: int,
        padding_count: int,
        unit: str,
        *,
        allow_oversized_search: bool = False,
    ) -> tuple[str, list[int], int]:
        text = _prompt_text(
            workload_id,
            request_index,
            record_count=record_count,
            padding_count=padding_count,
            padding_unit=unit,
        )
        tokens, max_model_len = _tokenize_messages(
            [{"role": "user", "content": text}],
            allow_oversized_search=allow_oversized_search,
        )
        return text, tokens, max_model_len

    _text, base_tokens, _maximum = material(0, 0, "")
    if len(base_tokens) > target_tokens:
        raise QualificationWorkerError("reviewed prompt envelope exceeds target length")
    low = 0
    high = max(1, target_tokens // 8)
    while True:
        _candidate_text, candidate_tokens, _maximum = material(
            high, 0, "", allow_oversized_search=True
        )
        if len(candidate_tokens) >= target_tokens:
            break
        low = high
        high *= 2
        if high > target_tokens:
            raise QualificationWorkerError("prompt record search did not converge")
    while low + 1 < high:
        middle = (low + high) // 2
        _candidate_text, candidate_tokens, _maximum = material(
            middle, 0, "", allow_oversized_search=True
        )
        if len(candidate_tokens) <= target_tokens:
            low = middle
        else:
            high = middle
    exact: tuple[str, list[int], int] | None = None
    for records in dict.fromkeys((low, max(0, low - 1), max(0, low - 2))):
        for unit in (" neutral", " x", " 0", "."):
            for padding_count in range(0, 257):
                candidate = material(
                    records,
                    padding_count,
                    unit,
                    allow_oversized_search=True,
                )
                count = len(candidate[1])
                if count == target_tokens:
                    exact = candidate
                    break
                if count > target_tokens + 8:
                    break
            if exact is not None:
                break
        if exact is not None:
            break
    if exact is None:
        raise QualificationWorkerError(
            f"live tokenizer could not build exact {target_tokens}-token prompt"
        )
    text, tokens, max_model_len = exact
    if len(tokens) != target_tokens or len(tokens) > context_length:
        raise QualificationWorkerError(
            "live tokenizer did not prove the exact validated prompt target"
        )
    rendered = _detokenize(tokens)
    messages = ({"role": "user", "content": text},)
    # One final tokenization closes time-of-check/time-of-use drift before inference.
    repeated, repeated_maximum = _tokenize_messages(list(messages))
    if repeated != tokens or repeated_maximum != max_model_len:
        raise QualificationWorkerError(
            "live tokenizer changed during prompt construction"
        )
    return PromptMaterial(
        messages=messages,
        tokens=tuple(tokens),
        input_ids_sha256=_canonical_sha(tokens),
        rendered_prompt_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        max_model_len=max_model_len,
    )


def _tokenizer_identity(checkpoint: Path) -> tuple[str, str]:
    inventory = {
        name: _sha256(checkpoint / name)
        for name in TOKENIZER_FILES
        if (checkpoint / name).is_file()
    }
    if "tokenizer_config.json" not in inventory or len(inventory) < 2:
        raise QualificationWorkerError("checkpoint tokenizer closure is incomplete")
    try:
        config = json.loads(
            (checkpoint / "tokenizer_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationWorkerError("tokenizer config is malformed") from exc
    template_inventory: dict[str, Any] = {}
    if isinstance(config, Mapping) and config.get("chat_template") is not None:
        template_inventory["tokenizer_config.chat_template"] = config["chat_template"]
    for name in ("chat_template.jinja", "chat_template.json"):
        path = checkpoint / name
        if path.is_file():
            template_inventory[name] = _sha256(path)
    if not template_inventory:
        raise QualificationWorkerError("checkpoint has no pinned chat template")
    return _canonical_sha(inventory), _canonical_sha(template_inventory)


def _stream_workload_request(
    material: PromptMaterial,
    *,
    workload_id: str,
    request_index: int,
    max_completion_tokens: int,
    barrier: threading.Barrier,
    request_id: str,
) -> dict[str, Any]:
    payload = {
        "model": SERVED_ALIAS,
        "messages": list(material.messages),
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 7,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "ignore_eos": not workload_id.startswith("needle_"),
        "stream": True,
        "stream_options": {"include_usage": True},
        "rid": request_id,
    }
    barrier.wait(timeout=60)
    started = time.perf_counter()
    try:
        response = requests.post(
            f"http://127.0.0.1:{HOST_PORT}/v1/chat/completions",
            json=payload,
            headers={"Accept": "text/event-stream"},
            timeout=(10, 1_800),
            stream=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise QualificationWorkerError("streaming selector request failed") from exc
    content: list[str] = []
    reasoning: list[str] = []
    first_token: float | None = None
    finished: float | None = None
    finish_reason: str | None = None
    usage: Mapping[str, Any] | None = None
    byte_count = 0
    event_count = 0
    try:
        for raw in response.iter_lines(decode_unicode=False):
            now = time.perf_counter()
            if not raw:
                continue
            byte_count += len(raw)
            if byte_count > MAX_STREAM_BYTES:
                raise QualificationWorkerError(
                    "streaming selector response is too large"
                )
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                finished = now
                break
            try:
                event = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise QualificationWorkerError(
                    "selector stream event is malformed"
                ) from exc
            if not isinstance(event, Mapping):
                raise QualificationWorkerError("selector stream event is not an object")
            event_count += 1
            if event.get("model") not in (None, SERVED_ALIAS):
                raise QualificationWorkerError("selector stream changed served alias")
            if event.get("usage") is not None:
                if not isinstance(event["usage"], Mapping):
                    raise QualificationWorkerError("selector stream usage is malformed")
                usage = event["usage"]
            choices = event.get("choices")
            if choices:
                if (
                    not isinstance(choices, list)
                    or len(choices) != 1
                    or not isinstance(choices[0], Mapping)
                ):
                    raise QualificationWorkerError(
                        "selector stream choices are malformed"
                    )
                choice = choices[0]
                delta = choice.get("delta") or {}
                if not isinstance(delta, Mapping):
                    raise QualificationWorkerError("selector stream delta is malformed")
                for field, destination in (
                    ("content", content),
                    ("reasoning_content", reasoning),
                ):
                    part = delta.get(field)
                    if part is not None:
                        if not isinstance(part, str):
                            raise QualificationWorkerError(
                                "selector stream text is malformed"
                            )
                        if part:
                            first_token = first_token or now
                            destination.append(part)
                            finished = now
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                    finished = now
    finally:
        response.close()
    if event_count <= 0 or first_token is None or finished is None or usage is None:
        raise QualificationWorkerError("selector stream is incomplete")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if (
        prompt_tokens != len(material.tokens)
        or isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens <= 0
    ):
        raise QualificationWorkerError("selector stream token accounting changed")
    response_text = "".join(content)
    reasoning_text = "".join(reasoning)
    is_needle = workload_id.startswith("needle_")
    if is_needle:
        nonce = _needle_nonce(workload_id)
        needle_passed: bool | None = response_text == nonce
        if finish_reason != "stop" or completion_tokens > max_completion_tokens:
            raise QualificationWorkerError("needle request did not stop within its cap")
        recorded_text: str | None = response_text
        needle_sha: str | None = hashlib.sha256(nonce.encode("ascii")).hexdigest()
    else:
        if finish_reason != "length" or completion_tokens != max_completion_tokens:
            raise QualificationWorkerError("fixed selector work did not reach its cap")
        needle_passed = None
        recorded_text = None
        needle_sha = None
    elapsed = finished - started
    ttft = first_token - started
    if elapsed <= 0 or ttft <= 0 or ttft > elapsed:
        raise QualificationWorkerError("selector stream timing is inconsistent")
    response_sha = _canonical_sha(
        {"content": response_text, "reasoning_content": reasoning_text}
    )
    if is_needle:
        response_sha = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
    return {
        "request_index": request_index,
        "input_ids_sha256": material.input_ids_sha256,
        "rendered_prompt_sha256": material.rendered_prompt_sha256,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_seconds": elapsed,
        "ttft_seconds": ttft,
        "completion_tps": completion_tokens / elapsed,
        "effective_prefill_tps": prompt_tokens / ttft,
        "response_text": recorded_text,
        "response_sha256": response_sha,
        "needle_expected_sha256": needle_sha,
        "needle_passed": needle_passed,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _run_workload(
    workload_id: str,
    *,
    materials: Sequence[PromptMaterial],
    trials: int,
    boot_id: str,
) -> dict[str, Any]:
    concurrency, prompt_tokens, max_completion_tokens = WORKLOAD_SPECS[workload_id]
    if len(materials) != concurrency:
        raise QualificationWorkerError("workload prompt concurrency changed")
    rows: list[dict[str, Any]] = []
    for trial in range(trials):
        barrier = threading.Barrier(concurrency + 1)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _stream_workload_request,
                    material,
                    workload_id=workload_id,
                    request_index=request_index,
                    max_completion_tokens=max_completion_tokens,
                    barrier=barrier,
                    request_id=(f"{boot_id}-{workload_id}-{trial}-{request_index}"),
                )
                for request_index, material in enumerate(materials)
            ]
            barrier.wait(timeout=60)
            requests_evidence = [future.result() for future in futures]
        wall = time.perf_counter() - started
        requests_evidence.sort(key=lambda value: int(value["request_index"]))
        wall = max(
            wall,
            max(float(value["elapsed_seconds"]) for value in requests_evidence),
        )
        completion = sum(int(value["completion_tokens"]) for value in requests_evidence)
        rows.append(
            {
                "trial": trial,
                "wall_elapsed_seconds": wall,
                "requests": requests_evidence,
                "completion_tokens": completion,
                "aggregate_completion_tps": completion / wall,
            }
        )
    all_requests = [request for row in rows for request in row["requests"]]
    total_completion = sum(int(row["completion_tokens"]) for row in rows)
    total_wall = sum(float(row["wall_elapsed_seconds"]) for row in rows)
    ttfts = [float(value["ttft_seconds"]) for value in all_requests]
    return {
        "workload_id": workload_id,
        "concurrency": concurrency,
        "requested_prompt_tokens": prompt_tokens,
        "max_completion_tokens": max_completion_tokens,
        "trials": rows,
        "metrics": {
            "trial_count": trials,
            "completion_tps": total_completion / total_wall,
            "effective_prefill_tps": sum(
                int(value["prompt_tokens"]) for value in all_requests
            )
            / sum(float(value["ttft_seconds"]) for value in all_requests),
            "ttft_p50_seconds": _percentile(ttfts, 0.50),
            "ttft_p95_seconds": _percentile(ttfts, 0.95),
        },
    }


def _state_semantic_evidence(
    request: Mapping[str, Any], *, client: qualify.EndpointClient
) -> dict[str, Any]:
    assets = _paths(request)["assets"]
    image_url, _image_asset = qualify._media_url(
        str(assets / "candy.JPG"), kind="image"
    )
    video_url, _video_asset = qualify._media_url(
        str(assets / "jobs_presenting_ipod.mp4"), kind="video"
    )
    image = client.chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": "What human body part is holding the colorful candies? Reply with one noun.",
                    },
                ],
            }
        ],
        max_tokens=192,
    )
    video = client.chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {
                        "type": "text",
                        "text": "What device does the presenter reveal from the small jeans pocket?",
                    },
                ],
            }
        ],
        max_tokens=256,
    )
    behavior = qualify._behavioral_probe(
        client, _paths(request)["source"] / "aeon/behavioral_sft/data/eval.jsonl"
    )

    def media_record(result: Any, term: str) -> dict[str, Any]:
        response = str(result.content)
        passed = (
            result.finish_reason == "stop" and term.casefold() in response.casefold()
        )
        return {
            "expected_term": term,
            "response_text": response,
            "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            "passed": passed,
        }

    image_record = media_record(image, "hand")
    video_record = media_record(video, "iPod")
    return {
        "image": image_record,
        "video": video_record,
        "behavioral_gate": behavior,
        "passed": bool(
            image_record["passed"]
            and video_record["passed"]
            and behavior.get("passed") is True
        ),
    }


def _build_workload_evidence(
    request: Mapping[str, Any],
    *,
    arm: str,
    identity: Mapping[str, Any],
    candidate: CandidateIdentity | None,
    checkpoint: Path,
) -> dict[str, Any]:
    selection_phase = candidate.phase if candidate is not None else None
    required = (
        PHASE_WORKLOADS[selection_phase]
        if selection_phase is not None
        else FINAL_WORKLOADS
    )
    trials = (
        MIN_FINAL_TRIALS
        if candidate is None or selection_phase in {"mtp_finalist", "replay"}
        else MIN_SELECTOR_TRIALS
    )
    started_at = _now()
    prompt_materials: dict[str, tuple[PromptMaterial, ...]] = {}
    max_model_len: int | None = None
    for workload_id in WORKLOAD_ORDER:
        if workload_id not in required:
            continue
        concurrency, target, _completion = WORKLOAD_SPECS[workload_id]
        materials = tuple(
            _fit_prompt(workload_id, request_index, target)
            for request_index in range(concurrency)
        )
        observed_maxima = {material.max_model_len for material in materials}
        if len(observed_maxima) != 1:
            raise QualificationWorkerError("live tokenizer max length changed")
        observed = observed_maxima.pop()
        if max_model_len is None:
            max_model_len = observed
        elif max_model_len != observed:
            raise QualificationWorkerError("live tokenizer identity changed")
        prompt_materials[workload_id] = materials
    workloads = [
        _run_workload(
            workload_id,
            materials=prompt_materials[workload_id],
            trials=trials,
            boot_id=str(identity["boot_id"]),
        )
        for workload_id in WORKLOAD_ORDER
        if workload_id in required
    ]
    prompt_inventory = [
        {
            "workload_id": workload["workload_id"],
            "trial": row["trial"],
            "request_index": item["request_index"],
            "input_ids_sha256": item["input_ids_sha256"],
            "rendered_prompt_sha256": item["rendered_prompt_sha256"],
            "prompt_tokens": item["prompt_tokens"],
        }
        for workload in workloads
        for row in workload["trials"]
        for item in row["requests"]
    ]
    client = qualify.EndpointClient(
        f"http://127.0.0.1:{HOST_PORT}", api_key=None, timeout_seconds=900
    )
    client.bind_served_alias(SERVED_ALIAS)
    if selection_phase == "state_dtype":
        semantic = _state_semantic_evidence(request, client=client)
    else:
        needle_passed = all(
            item["needle_passed"] is True
            for workload in workloads
            if str(workload["workload_id"]).startswith("needle_")
            for row in workload["trials"]
            for item in row["requests"]
        )
        semantic = {
            "image": None,
            "video": None,
            "behavioral_gate": None,
            "passed": needle_passed,
        }
    tokenizer_sha, chat_template_sha = _tokenizer_identity(checkpoint)
    return {
        "schema_version": qualify.WORKLOAD_EVIDENCE_SCHEMA_VERSION,
        "complete": True,
        "runtime_id": identity["runtime_id"],
        "arm": arm,
        "candidate_id": candidate.candidate_id if candidate is not None else None,
        "phase": selection_phase,
        "served_alias": SERVED_ALIAS,
        "runtime_config_sha256": identity["config_sha256"],
        "prompt_suite_sha256": qualify._sha256_json(prompt_inventory),
        "tokenizer_sha256": tokenizer_sha,
        "chat_template_sha256": chat_template_sha,
        "max_model_len": max_model_len,
        "started_at": started_at,
        "completed_at": _now(),
        "workloads": workloads,
        "semantic_equivalence": semantic,
    }


def _remove_container(
    request: Mapping[str, Any],
    arm: str,
    container_id: str,
    command: Sequence[str],
    candidate: CandidateIdentity | None = None,
) -> None:
    item = _inspect(container_id)
    if item is None:
        return
    expected = _labels(request, arm, command, candidate)
    observed = item.get("Config", {}).get("Labels")
    if not isinstance(observed, Mapping) or any(
        observed.get(key) != value for key, value in expected.items()
    ):
        raise QualificationWorkerError(
            "refusing to stop a container with changed labels"
        )
    if item.get("State", {}).get("Running") is True:
        result = _docker(
            ["container", "stop", "--time", "30", container_id], timeout=60
        )
        if result.returncode != 0:
            raise QualificationWorkerError("qualification container did not stop")
    item = _inspect(container_id)
    if item is not None and item.get("State", {}).get("Running") is not True:
        result = _docker(["container", "rm", container_id], timeout=30)
        if result.returncode != 0:
            raise QualificationWorkerError("stopped qualification container remains")
    if _inspect(container_id) is not None:
        raise QualificationWorkerError("qualification container absence is unproven")


def _probe_args(
    request: Mapping[str, Any],
    arm: str,
    identity: Path,
    workload_evidence: Path,
    cgroup: Path,
    output: Path,
) -> argparse.Namespace:
    assets = _paths(request)["assets"]
    source = _paths(request)["source"]
    return argparse.Namespace(
        arm=arm,
        base_url=f"http://127.0.0.1:{HOST_PORT}",
        served_alias=SERVED_ALIAS,
        runtime_identity=identity,
        workload_evidence=workload_evidence,
        cgroup=cgroup,
        cgroup_root=Path("/sys/fs/cgroup"),
        proc_root=Path("/proc"),
        api_key_file=None,
        image=str(assets / "candy.JPG"),
        image_question="What human body part is holding the colorful candies? Reply with one noun.",
        image_expected_term="hand",
        video=str(assets / "jobs_presenting_ipod.mp4"),
        video_question="What device does the presenter reveal from the small jeans pocket?",
        video_expected_term="iPod",
        behavior_eval=source / "aeon/behavioral_sft/data/eval.jsonl",
        trials=qualify.MIN_FINAL_TRIALS,
        benchmark_tokens=qualify.DEFAULT_BENCHMARK_TOKENS,
        max_accounted_vram_gb=VRAM_BUDGET_GB,
        max_cgroup_memory_gb=TASK_MEMORY_GB,
        max_boot_age_seconds=qualify.DEFAULT_MAX_BOOT_AGE_SECONDS,
        process_start_tolerance_seconds=qualify.DEFAULT_PROCESS_START_TOLERANCE_SECONDS,
        cuda_attestation_timeout_seconds=(
            qualify.DEFAULT_CUDA_ATTESTATION_TIMEOUT_SECONDS
        ),
        timeout_seconds=300.0,
        output=output,
    )


def _run_arm(
    request: Mapping[str, Any],
    arm: str,
    tuning: RuntimeTuning,
    candidate: CandidateIdentity | None = None,
) -> tuple[dict[str, Any], list[str]]:
    global _active
    container_id, command, evidence = _create(request, arm, tuning, candidate)
    _active = (arm, dict(request, container_id=container_id), command, candidate)
    try:
        created_item = _verify_container(
            request,
            arm,
            container_id,
            command,
            evidence,
            tuning,
            candidate,
            require_running_network=False,
        )
        container_config_sha256 = _canonical_sha(
            _container_config_projection(
                request,
                arm,
                command,
                evidence,
                created_item,
                candidate,
            )
        )
        result = _docker(["container", "start", container_id], timeout=120)
        if result.returncode != 0 or result.stdout.strip() != container_id:
            error = QualificationWorkerError("qualification container failed to start")
            if arm == ARM_SELECTION:
                raise _selection_failure(
                    stage="container_start",
                    code="docker_start_failed",
                    error=error,
                    container_config_sha256=container_config_sha256,
                ) from error
            raise error
        item = _verify_container(
            request, arm, container_id, command, evidence, tuning, candidate
        )
        if item is None or item.get("State", {}).get("Running") is not True:
            error = QualificationWorkerError("qualification container is not running")
            if arm == ARM_SELECTION:
                raise _selection_failure(
                    stage="server_readiness",
                    code="container_exited_after_start",
                    error=error,
                    container_config_sha256=container_config_sha256,
                ) from error
            raise error
        try:
            pid = item.get("State", {}).get("Pid")
            started_at = item.get("State", {}).get("StartedAt")
            if (
                isinstance(pid, bool)
                or not isinstance(pid, int)
                or pid <= 1
                or not isinstance(started_at, str)
            ):
                raise QualificationWorkerError(
                    "container process identity is malformed"
                )
            cgroup = _cgroup(pid, container_id)
            cgroup_procs = {
                int(value)
                for value in (cgroup / "cgroup.procs")
                .read_text(encoding="ascii")
                .split()
            }
            if pid not in cgroup_procs:
                raise QualificationWorkerError(
                    "container PID is absent from its task cgroup"
                )
            _memory_events(cgroup)
        except (OSError, ValueError, QualificationWorkerError) as error:
            if arm == ARM_SELECTION:
                raise _selection_failure(
                    stage="runtime_identity_binding",
                    code="task_cgroup_binding_failed",
                    error=error,
                    container_config_sha256=container_config_sha256,
                ) from error
            raise
        _atomic_json(
            evidence["context"],
            {
                "container_id": container_id,
                "container_pid": pid,
                "cgroup_path": str(cgroup),
                "container_pid_in_cgroup": True,
            },
        )
        try:
            _wait_ready(container_id)
        except QualificationWorkerError as error:
            if arm == ARM_SELECTION and str(error) in {
                "qualification container exited during load",
                "qualification container did not become ready",
            }:
                raise _selection_failure(
                    stage="server_readiness",
                    code=(
                        "container_exited_during_load"
                        if "exited" in str(error)
                        else "server_readiness_timeout"
                    ),
                    error=error,
                    container_config_sha256=container_config_sha256,
                ) from error
            raise
        ready_item = _verify_container(
            request, arm, container_id, command, evidence, tuning, candidate
        )
        try:
            _memory_events(cgroup)
        except (OSError, ValueError, QualificationWorkerError) as error:
            if arm == ARM_SELECTION:
                raise _selection_failure(
                    stage="runtime_identity_binding",
                    code="task_cgroup_freshness_failed",
                    error=error,
                    container_config_sha256=container_config_sha256,
                ) from error
            raise
        config = _runtime_config(arm, tuning)
        try:
            runtime_config_binding = _runtime_config_binding(
                request,
                arm,
                command,
                evidence,
                tuning,
                ready_item,
                candidate,
            )
        except (
            OSError,
            ValueError,
            TypeError,
            requests.RequestException,
            QualificationWorkerError,
            qualify.QualificationError,
        ) as error:
            if arm == ARM_SELECTION:
                raise _selection_failure(
                    stage="runtime_identity_binding",
                    code="server_info_binding_failed",
                    error=error,
                    container_config_sha256=container_config_sha256,
                ) from error
            raise
        _checkpoint, checkpoint_tree = _checkpoint_for_arm(request, arm)
        sibling = _read_json(
            _paths(request)["sibling_manifest"], maximum=2 * 1024 * 1024
        )
        checkpoint_role = "official_untuned" if arm == "official_untuned" else "tuned"
        lm_head_sha = sibling[
            "official_untuned_lm_head_tensor_sha256"
            if checkpoint_role == "official_untuned"
            else "tuned_lm_head_tensor_sha256"
        ]
        identity = {
            "schema_version": qualify.RUNTIME_IDENTITY_SCHEMA_VERSION,
            "arm": arm,
            "selection_candidate": (
                candidate.receipt() if candidate is not None else None
            ),
            "served_alias": SERVED_ALIAS,
            "checkpoint_tree_sha256": checkpoint_tree,
            "tuned_checkpoint_tree_sha256": request["checkpoint_tree_sha256"],
            "official_untuned_checkpoint_tree_sha256": request[
                "official_untuned_checkpoint_tree_sha256"
            ],
            "sibling_manifest_sha256": request["build_sibling_manifest_sha256"],
            "checkpoint_role": checkpoint_role,
            "lm_head_tensor_sha256": lm_head_sha,
            "non_lm_head_tensor_inventory_sha256": sibling[
                "non_lm_head_tensor_inventory_sha256"
            ],
            "boot_id": f"{request['runtime_id']}-{arm}-{secrets.token_hex(8)}",
            "runtime_id": request["runtime_id"],
            "config_sha256": qualify._sha256_json(config),
            "runtime_config": config,
            "runtime_config_binding": runtime_config_binding,
            "sglang_commit": SGLANG_COMMIT,
            "oci_image_digest": IMAGE_DIGEST,
            "started_at": started_at,
            "mtp_enabled": tuning.nextn is not None,
            "ple_offload_embedding": True,
            "transformer_weight_cpu_offload": False,
            "cgroup_path": str(cgroup),
            "task_scoped_cgroup": True,
            "lease_claim_id_sha256": hashlib.sha256(
                str(request["claim_id"]).encode("utf-8")
            ).hexdigest(),
            "leased_gpu_uuid_sha256": hashlib.sha256(
                str(request["gpu_uuid"]).encode("utf-8")
            ).hexdigest(),
            "container_id": container_id,
            "container_pid": pid,
            "container_start_ticks": _process_start_ticks(pid),
            "container_pid_in_cgroup": True,
            "checkpoint_mount_path": "/model",
            "checkpoint_mount_read_only": True,
            "endpoint_host": "127.0.0.1",
            "endpoint_port": HOST_PORT,
            "model_info_model_path": "/model",
            "cuda_memory_attestation_path": str(evidence["attestation"]),
            "cuda_memory_freeze_path": str(evidence["freeze"]),
            "cuda_memory_sampler_sha256": _sha256(
                _paths(request)["source"]
                / "aeon/scripts/qwen_flash_next_container_supervisor.py"
            ),
        }
        output = _paths(request)["output"]
        report_name = candidate.key if candidate is not None else arm
        identity_path = output / f"{report_name}.runtime-identity.json"
        workload_path = output / f"{report_name}.workloads.json"
        report_path = output / f"{report_name}.arm.json"
        _atomic_json(identity_path, identity)
        try:
            workload = _build_workload_evidence(
                request,
                arm=arm,
                identity=identity,
                candidate=candidate,
                checkpoint=_checkpoint,
            )
            _atomic_json(workload_path, workload)
            report = qualify.probe_arm(
                _probe_args(
                    request,
                    arm,
                    identity_path,
                    workload_path,
                    cgroup,
                    report_path,
                )
            )
        except (QualificationWorkerError, qualify.QualificationError) as error:
            if arm == ARM_SELECTION:
                diagnostic = _sanitize_selection_failure_tail(
                    str(error),
                    request=request,
                    container_id=container_id,
                    maximum_bytes=4096,
                )
                print(
                    "selection candidate probe failed: "
                    f"{type(error).__name__}: {diagnostic['tail']}",
                    file=sys.stderr,
                    flush=True,
                )
                raise _selection_failure(
                    stage="candidate_probe",
                    code="candidate_probe_failed",
                    error=error,
                    container_config_sha256=container_config_sha256,
                ) from error
            raise
        if report.get("passed") is not True and arm != ARM_SELECTION:
            raise QualificationWorkerError(f"{arm} qualification did not pass")
        return report, command
    except SelectionBootFailure as failure:
        if (
            arm == ARM_SELECTION
            and candidate is not None
            and failure.stage != "container_create"
        ):
            _persist_selection_docker_failure(
                request,
                container_id=container_id,
                command=command,
                evidence=evidence,
                tuning=tuning,
                candidate=candidate,
                failure=failure,
            )
        raise
    finally:
        try:
            _remove_container(request, arm, container_id, command, candidate)
        finally:
            _active = None


@dataclass(frozen=True)
class CandidateOutcome:
    identity: CandidateIdentity
    tuning: RuntimeTuning
    report: Mapping[str, Any]
    report_path: Path

    @property
    def config_sha256(self) -> str:
        value = self.report.get("runtime_identity")
        if (
            not isinstance(value, Mapping)
            or _SHA.fullmatch(str(value.get("config_sha256") or "")) is None
        ):
            raise QualificationWorkerError(
                "candidate report config identity is malformed"
            )
        return str(value["config_sha256"])


@dataclass(frozen=True)
class CandidateAttempt:
    """One sanitized selector boot failure with no live arm report."""

    identity: CandidateIdentity
    tuning: RuntimeTuning
    report_path: Path

    @property
    def config_sha256(self) -> str:
        return qualify._sha256_json(_runtime_config(ARM_SELECTION, self.tuning))


CandidateEvidence = CandidateOutcome | CandidateAttempt


def _revalidate_candidate_attempt_diagnostic(
    request: Mapping[str, Any], attempt: CandidateAttempt
) -> None:
    """Rebind task-local Docker logs immediately before final comparison."""

    receipt = _read_json(attempt.report_path, maximum=qualify.MAX_JSON_BYTES)
    failure_stage = receipt.get("failure_stage")
    path = _selection_docker_failure_sidecar_path(request, attempt.identity)
    if failure_stage == "container_create":
        if receipt.get("docker_failure_diagnostic") is not None:
            raise QualificationWorkerError(
                "container-create attempt unexpectedly embeds Docker diagnostics"
            )
        if path.exists() or path.is_symlink():
            raise QualificationWorkerError(
                "container-create attempt unexpectedly has Docker diagnostics"
            )
        return
    container_config_sha256 = receipt.get("container_config_sha256")
    fields = {
        name: receipt.get(name)
        for name in ("failure_code", "failure_detail_sha256", "command_sha256")
    }
    if (
        not isinstance(failure_stage, str)
        or not isinstance(container_config_sha256, str)
        or _SHA.fullmatch(container_config_sha256) is None
        or any(not isinstance(value, str) for value in fields.values())
    ):
        raise QualificationWorkerError(
            "selector attempt Docker diagnostic binding is malformed"
        )
    summary = _validate_selection_docker_failure_sidecar(
        path,
        request=request,
        candidate=attempt.identity,
        failure_stage=failure_stage,
        failure_code=str(fields["failure_code"]),
        failure_detail_sha256=str(fields["failure_detail_sha256"]),
        command_sha256=str(fields["command_sha256"]),
        container_config_sha256=container_config_sha256,
    )
    if receipt.get("docker_failure_diagnostic") != summary:
        raise QualificationWorkerError(
            "selector attempt Docker diagnostic summary changed"
        )
    sidecars = receipt.get("diagnostic_sidecars")
    if (
        not isinstance(sidecars, Mapping)
        or sidecars.get(path.name) != summary["sidecar_sha256"]
    ):
        raise QualificationWorkerError(
            "selector attempt Docker diagnostic digest changed"
        )


def _candidate_vector(report: Mapping[str, Any]) -> dict[str, float]:
    evidence = report.get("workload_evidence")
    workloads = evidence.get("workloads") if isinstance(evidence, Mapping) else None
    if not isinstance(workloads, list):
        raise QualificationWorkerError("candidate report has no measured workloads")
    result: dict[str, float] = {}
    for workload in workloads:
        if not isinstance(workload, Mapping) or not isinstance(
            workload.get("metrics"), Mapping
        ):
            raise QualificationWorkerError("candidate workload metrics are malformed")
        workload_id = str(workload.get("workload_id"))
        metrics = workload["metrics"]
        field = (
            "completion_tps"
            if workload_id in {"b1_512_512", "c4_512_512"}
            else "effective_prefill_tps"
        )
        value = metrics.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise QualificationWorkerError("candidate throughput metric is invalid")
        result[f"{workload_id}.{field}"] = float(value)
    if not result:
        raise QualificationWorkerError("candidate metric vector is empty")
    return result


def _rank_phase(
    outcomes: Sequence[CandidateOutcome], *, memory_phase: bool = False
) -> list[CandidateOutcome]:
    if not outcomes:
        raise QualificationWorkerError("selection phase produced no candidate evidence")
    baseline = _candidate_vector(outcomes[0].report)
    scored: list[tuple[float, float, str, CandidateOutcome]] = []
    for outcome in outcomes:
        vector = _candidate_vector(outcome.report)
        if set(vector) != set(baseline):
            raise QualificationWorkerError(
                "phase candidates measured different workloads"
            )
        ratios = [vector[key] / baseline[key] for key in sorted(vector)]
        semantic = outcome.report.get("workload_validation")
        valid = (
            outcome.report.get("passed") is True
            and isinstance(semantic, Mapping)
            and semantic.get("passed") is True
        )
        eligible = valid and (memory_phase or min(ratios) >= 0.95)
        if eligible:
            scored.append(
                (
                    min(ratios),
                    math.exp(sum(math.log(value) for value in ratios) / len(ratios)),
                    outcome.identity.candidate_id,
                    outcome,
                )
            )
    if not scored:
        raise QualificationWorkerError(
            "selection phase has no admissible safe candidate"
        )
    if memory_phase:
        eligible_ids = {item[3].identity.candidate_id for item in scored}
        return [
            outcome
            for outcome in outcomes
            if outcome.identity.candidate_id in eligible_ids
        ]
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in scored]


def _candidate_resource_receipt(
    outcome: CandidateOutcome,
) -> tuple[Mapping[str, Any], int]:
    """Return exact selector resource evidence or reject the candidate."""

    resources = outcome.report.get("resources")
    if not isinstance(resources, Mapping) or any(
        resources.get(field) is not True
        for field in (
            "memory_limit_and_oom_events_zero_before_and_after",
            "vram_budget_passed",
            "ram_budget_passed",
            "physical_cuda_reserve_passed",
        )
    ):
        raise QualificationWorkerError(
            "MoE backend candidate lacks complete passing resource gates"
        )
    physical = resources.get("physical_cuda_memory")
    reserve = (
        physical.get("min_reserve_bytes") if isinstance(physical, Mapping) else None
    )
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve <= 0:
        raise QualificationWorkerError(
            "MoE backend candidate CUDA reserve is malformed"
        )
    return resources, reserve


def _require_moe_cutlass(cutlass: CandidateOutcome) -> CandidateOutcome:
    """Require the sole SM120 ModelOpt-FP4 MoE path and comfortable reserve."""

    if (
        cutlass.identity.candidate_id != "moe_cutlass"
        or cutlass.tuning.moe_runner_backend
        != runtime_contract.CUTLASS_MOE_RUNNER_BACKEND
    ):
        raise QualificationWorkerError("required CUTLASS MoE identity changed")
    if _rank_phase((cutlass,)) != [cutlass]:
        raise QualificationWorkerError("required CUTLASS MoE baseline did not pass")
    _resources, reserve = _candidate_resource_receipt(cutlass)
    if reserve < runtime_contract.CUTLASS_MIN_CUDA_RESERVE_BYTES:
        raise QualificationWorkerError(
            "required CUTLASS MoE baseline lacks comfortable CUDA reserve"
        )
    return cutlass


def _rank_mtp_finalists(
    reference: CandidateOutcome,
    finalists: Sequence[CandidateOutcome],
) -> CandidateOutcome:
    """Select the MTP setting by the qualifier's in-phase paired CI rule."""

    eligible = {
        item.identity.candidate_id for item in _rank_phase((reference, *finalists))
    }
    if reference.identity.candidate_id not in eligible:
        raise QualificationWorkerError("MTP finalist off reference is not admissible")
    off_rows = qualify._completion_speed_rows(reference.report, "b1_512_512")
    by_id = {item.identity.candidate_id: item for item in finalists}
    bases = {
        candidate_id.removesuffix("_forward").removesuffix("_reverse")
        for candidate_id in by_id
    }
    scores: list[tuple[float, float, int, str]] = []
    for base in sorted(bases):
        forward_id = f"{base}_forward"
        reverse_id = f"{base}_reverse"
        if not {forward_id, reverse_id} <= eligible:
            continue
        on_rows = [
            row
            for candidate_id in (forward_id, reverse_id)
            for row in qualify._completion_speed_rows(
                by_id[candidate_id].report, "b1_512_512"
            )
        ]
        paired_off = [*off_rows, *off_rows]
        if len(on_rows) != len(paired_off):
            raise QualificationWorkerError(
                "MTP finalist counterbalanced trials are not paired"
            )
        point = (
            sum(row["completion_tokens"] for row in on_rows)
            / sum(row["elapsed_seconds"] for row in on_rows)
        ) / (
            sum(row["completion_tokens"] for row in paired_off)
            / sum(row["elapsed_seconds"] for row in paired_off)
        )
        ci_lower, _ci_upper = qualify._paired_bootstrap_ci(
            paired_off,
            on_rows,
            samples=qualify.DEFAULT_BOOTSTRAP_SAMPLES,
        )
        match = re.fullmatch(r"mtp_s([123])_d[234]", base)
        if match is None:
            raise QualificationWorkerError("MTP finalist base ID is malformed")
        scores.append((ci_lower, point, int(match.group(1)), base))
    if not scores:
        raise QualificationWorkerError(
            "MTP finalist phase has no admissible counterbalanced setting"
        )
    scores.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    return by_id[f"{scores[0][3]}_forward"]


def _state_dtype_admissible_candidates(
    references: Sequence[CandidateOutcome],
    bf16_candidates: Sequence[CandidateOutcome],
) -> list[CandidateOutcome]:
    """Keep safe FP32 references when an otherwise valid BF16 peer regresses."""

    by_id = {item.identity.candidate_id: item for item in references}
    admitted = list(references)
    for candidate in bf16_candidates:
        peer_id = candidate.identity.candidate_id.removesuffix("_bf16") + "_fp32_ref"
        peer = by_id.get(peer_id)
        if peer is None:
            raise QualificationWorkerError("state BF16 candidate has no FP32 reference")
        try:
            qualify._validate_one_state_dtype_peer_equivalence(
                peer.report, candidate.report
            )
        except qualify.StateDtypePeerRegression:
            continue
        admitted.append(candidate)
    return admitted


def _run_candidate(
    request: Mapping[str, Any],
    *,
    ordinal: int,
    candidate_id: str,
    phase: str,
    tuning: RuntimeTuning,
    parent: CandidateOutcome | None,
) -> CandidateEvidence:
    identity = CandidateIdentity(
        candidate_id=candidate_id,
        phase=phase,
        ordinal=ordinal,
        parent_candidate_id=(
            parent.identity.candidate_id if parent is not None else None
        ),
        parent_config_sha256=(parent.config_sha256 if parent is not None else None),
    )
    started_at = _now()
    try:
        report, _command = _run_arm(request, ARM_SELECTION, tuning, candidate=identity)
    except SelectionBootFailure as failure:
        config = _runtime_config(ARM_SELECTION, tuning)
        command_sha256 = _canonical_sha(
            _server_command(ARM_SELECTION, model_path="/model", tuning=tuning)
        )
        diagnostic_sidecars: dict[str, str] = {}
        for suffix in ("runtime-identity.json", "workloads.json"):
            sidecar = _paths(request)["output"] / f"{identity.key}.{suffix}"
            if sidecar.is_file() and not sidecar.is_symlink():
                diagnostic_sidecars[sidecar.name] = _sha256(sidecar)
        if failure.stage == "candidate_probe" and not any(
            name.endswith(".runtime-identity.json") for name in diagnostic_sidecars
        ):
            raise QualificationWorkerError(
                "candidate probe failure lacks its runtime identity sidecar"
            ) from failure
        docker_sidecar = _selection_docker_failure_sidecar_path(request, identity)
        docker_diagnostic: dict[str, Any] | None = None
        if failure.stage == "container_create":
            if docker_sidecar.exists() or docker_sidecar.is_symlink():
                raise QualificationWorkerError(
                    "container-create failure unexpectedly has Docker diagnostics"
                ) from failure
        else:
            if failure.container_config_sha256 is None:
                raise QualificationWorkerError(
                    "post-create selector failure lacks its container binding"
                ) from failure
            docker_diagnostic = _validate_selection_docker_failure_sidecar(
                docker_sidecar,
                request=request,
                candidate=identity,
                failure_stage=failure.stage,
                failure_code=failure.code,
                failure_detail_sha256=failure.detail_sha256,
                command_sha256=command_sha256,
                container_config_sha256=failure.container_config_sha256,
            )
            diagnostic_sidecars[docker_sidecar.name] = docker_diagnostic[
                "sidecar_sha256"
            ]
        sibling = _read_json(
            _paths(request)["sibling_manifest"], maximum=2 * 1024 * 1024
        )
        receipt = {
            "schema_version": qualify.SELECTION_ATTEMPT_SCHEMA_VERSION,
            "complete": True,
            "passed": False,
            "ordered_index": identity.ordinal,
            "runtime_id": request["runtime_id"],
            "served_alias": SERVED_ALIAS,
            "candidate_id": identity.candidate_id,
            "phase": identity.phase,
            "parent_candidate_id": identity.parent_candidate_id,
            "parent_config_sha256": identity.parent_config_sha256,
            "resolved_config": config,
            "resolved_config_sha256": qualify._sha256_json(config),
            "lease_claim_id_sha256": hashlib.sha256(
                str(request["claim_id"]).encode("utf-8")
            ).hexdigest(),
            "leased_gpu_uuid_sha256": hashlib.sha256(
                str(request["gpu_uuid"]).encode("utf-8")
            ).hexdigest(),
            "sglang_commit": SGLANG_COMMIT,
            "oci_image_digest": IMAGE_DIGEST,
            "checkpoint_tree_sha256": request["checkpoint_tree_sha256"],
            "sibling_manifest_sha256": request["build_sibling_manifest_sha256"],
            "lm_head_tensor_sha256": sibling["tuned_lm_head_tensor_sha256"],
            "non_lm_head_tensor_inventory_sha256": sibling[
                "non_lm_head_tensor_inventory_sha256"
            ],
            "started_at": started_at,
            "completed_at": _now(),
            "failure_stage": failure.stage,
            "failure_code": failure.code,
            "failure_detail_sha256": failure.detail_sha256,
            "command_sha256": command_sha256,
            "container_config_sha256": failure.container_config_sha256,
            "diagnostic_sidecars": diagnostic_sidecars,
            "docker_failure_diagnostic": docker_diagnostic,
        }
        path = _paths(request)["output"] / f"{identity.key}.attempt.json"
        _atomic_json(path, receipt)
        qualify._selection_candidate_record(
            path, expected_ordered_index=identity.ordinal
        )
        return CandidateAttempt(identity, tuning, path)
    path = _paths(request)["output"] / f"{identity.key}.arm.json"
    if not path.is_file():
        raise QualificationWorkerError("candidate arm report was not persisted")
    return CandidateOutcome(identity, tuning, report, path)


def _runtime_release_config(
    request: Mapping[str, Any],
    off: Mapping[str, Any],
    on: Mapping[str, Any],
    *,
    off_tuning: RuntimeTuning,
    on_tuning: RuntimeTuning,
) -> dict[str, Any]:
    repo_id = str(request["repo_id"])
    arms: dict[str, Any] = {}
    for arm, report, qualification_arm, tuning in (
        ("tuned_mtp_off", off, ARM_TUNED_MTP_OFF, off_tuning),
        (
            "tuned_mtp_on_winner",
            on,
            ARM_TUNED_MTP_ON,
            on_tuning,
        ),
    ):
        identity = report["runtime_identity"]
        arms[arm] = {
            "config_sha256": identity["config_sha256"],
            "runtime_config": identity["runtime_config"],
            "environment": dict(CONSTANT_RUNTIME_ENV),
            "command": [
                DOCKER,
                "run",
                *[
                    item
                    for key, value in sorted(CONSTANT_RUNTIME_ENV.items())
                    for item in ("--env", f"{key}={value}")
                ],
                "--mount",
                (f"type=bind,src={MATERIALIZED_MODEL_PLACEHOLDER},dst=/model,readonly"),
                IMAGE_REFERENCE,
                *_server_command(qualification_arm, model_path="/model", tuning=tuning),
            ],
        }
    return {
        "schema_version": release_tool.RUNTIME_CONFIG_SCHEMA,
        "repo_id": repo_id,
        "model_reference": repo_id,
        "checkpoint_tree_sha256": request["checkpoint_tree_sha256"],
        "served_alias": SERVED_ALIAS,
        "display_name": DISPLAY_NAME,
        "artifact_name": ARTIFACT_NAME,
        "model_architecture": runtime_contract.MODEL_ARCHITECTURE,
        "toolchain": {
            "transformers": {
                "version": release_tool.TRANSFORMERS_VERSION,
                "wheel_sha256": release_tool.TRANSFORMERS_WHEEL_SHA256,
            },
            "modelopt": {
                "version": release_tool.MODELOPT_VERSION,
                "commit": release_tool.MODELOPT_COMMIT,
                "wheel_sha256": release_tool.MODELOPT_WHEEL_SHA256,
            },
            "sglang": {
                "commit": SGLANG_COMMIT,
                "source_stack_sha256": runtime_contract.SOURCE_STACK_SHA256,
                "oci_image": release_tool.SGLANG_IMAGE,
                "oci_image_digest": IMAGE_DIGEST,
                "oci_config_digest": IMAGE_CONFIG_DIGEST,
                "oci_archive_sha256": IMAGE_ARCHIVE_SHA256,
                "local_docker_image_id": IMAGE_ID,
                "required_image_labels": dict(runtime_contract.EXPECTED_IMAGE_LABELS),
            },
        },
        "hardware": {
            "gpu": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "gpu_count": 1,
            "vram_gb": 96,
        },
        "placement": {
            "ple_offload_embedding": True,
            "transformer_weight_cpu_offload": False,
        },
        "model_path_contract": {
            "checkpoint_tree_sha256": request["checkpoint_tree_sha256"],
            "host_path_placeholder": MATERIALIZED_MODEL_PLACEHOLDER,
            "container_path": "/model",
            "mount_read_only": True,
            "source_role": "offline-materialized-canonical-checkpoint",
        },
        "launch_contract": dict(release_tool.LAUNCH_CONTRACT),
        "arms": arms,
    }


def _final_mtp_off_tuning(winner: RuntimeTuning) -> RuntimeTuning:
    """Disable only native speculative decoding for the causal final pair."""

    if winner.nextn is None:
        raise QualificationWorkerError("selected winner does not enable native MTP")
    return replace(winner, nextn=None)


def _replay_off_reference_tuning(winner: RuntimeTuning) -> RuntimeTuning:
    """Preserve the selected state/decode path for the causal replay reference."""

    if winner.replay_ssm:
        raise QualificationWorkerError("replay reference already enables replay")
    return winner


def _require_behavioral_improvement_possible(
    official: Mapping[str, Any], tuned: Mapping[str, Any] | None = None
) -> None:
    """Fail before another expensive boot when the final behavior gate is impossible."""

    official_behavior = official.get("behavioral_gate")
    if not isinstance(official_behavior, Mapping):
        raise QualificationWorkerError("official behavior evidence is malformed")
    baseline = qualify._validated_behavior_report(
        official_behavior, label=ARM_OFFICIAL_UNTUNED
    )["summary"]
    baseline_total = baseline["non_harmful_unnecessary_refusals"]
    if baseline_total == 0:
        raise QualificationWorkerError(
            "official baseline has zero unnecessary refusals; strict reduction is impossible"
        )
    if tuned is None:
        return
    tuned_behavior = tuned.get("behavioral_gate")
    if not isinstance(tuned_behavior, Mapping):
        raise QualificationWorkerError("tuned behavior evidence is malformed")
    final = qualify._validated_behavior_report(
        tuned_behavior, label=ARM_TUNED_MTP_OFF
    )["summary"]
    baseline_counts = baseline["counts"]
    final_counts = final["counts"]
    categories = sorted(qualify.behavior_training.NON_HARMFUL_CATEGORIES)
    if not (
        final["non_harmful_unnecessary_refusals"] < baseline_total
        and all(
            final_counts[category]["unnecessary_refusals"]
            <= baseline_counts[category]["unnecessary_refusals"]
            for category in categories
        )
    ):
        raise QualificationWorkerError(
            "tuned checkpoint did not strictly reduce unnecessary refusals"
        )


def _write_manifest(output: Path) -> str:
    names = sorted(
        path.name for path in output.iterdir() if path.name != "MANIFEST.sha256"
    )
    lines: list[str] = []
    total = 0
    for name in names:
        path = output / name
        metadata = _private_file(path, maximum=16 * 1024 * 1024)
        total += metadata.st_size
        if total > MAX_OUTPUT_BYTES:
            raise QualificationWorkerError("qualification evidence exceeds its bound")
        lines.append(f"{_sha256(path)}  {name}\n")
    manifest = output / "MANIFEST.sha256"
    if manifest.exists() or manifest.is_symlink():
        raise QualificationWorkerError("qualification manifest already exists")
    descriptor = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, "".join(lines).encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _sha256(manifest)


def _progress(request: Mapping[str, Any], phase: str, candidate_id: str) -> None:
    _atomic_json(
        _paths(request)["status"],
        {
            "state": "running",
            "phase": phase,
            "candidate_id": candidate_id,
            "updated_at": _now(),
        },
    )


def _selection_pipeline(
    request: Mapping[str, Any],
) -> tuple[list[CandidateEvidence], CandidateOutcome]:
    evidence: list[CandidateEvidence] = []

    def run(
        candidate_id: str,
        phase: str,
        tuning: RuntimeTuning,
        parent: CandidateOutcome | None,
    ) -> CandidateEvidence:
        _progress(request, phase, candidate_id)
        outcome = _run_candidate(
            request,
            ordinal=len(evidence),
            candidate_id=candidate_id,
            phase=phase,
            tuning=tuning,
            parent=parent,
        )
        evidence.append(outcome)
        return outcome

    def required(item: CandidateEvidence, *, label: str) -> CandidateOutcome:
        if isinstance(item, CandidateAttempt):
            raise QualificationWorkerError(
                f"required selector baseline failed before identity: {label}"
            )
        return item

    baseline = RuntimeTuning.safe_baseline()
    moe_cutlass = required(
        run(
            "moe_cutlass",
            "moe_backend",
            baseline,
            None,
        ),
        label="moe_cutlass",
    )
    moe_winner = _require_moe_cutlass(moe_cutlass)
    graph_eager = required(
        run(
            "graph_eager",
            "graph",
            replace(moe_winner.tuning, cuda_graph="disabled"),
            moe_winner,
        ),
        label="graph_eager",
    )
    graph_full_evidence = run(
        "graph_full",
        "graph",
        replace(moe_winner.tuning, cuda_graph="full"),
        graph_eager,
    )
    graph_complete = [graph_eager]
    if isinstance(graph_full_evidence, CandidateOutcome):
        graph_complete.append(graph_full_evidence)
    ranked_graph = _rank_phase(graph_complete)
    if graph_eager not in ranked_graph:
        raise QualificationWorkerError("graph eager safe baseline did not survive")
    graph_winner = ranked_graph[0]

    gdn: list[CandidateOutcome] = []
    for code, decode, prefill in (
        ("tt", "triton", "triton"),
        ("ct", "cutedsl", "triton"),
        ("tc", "triton", "cutedsl"),
        ("cc", "cutedsl", "cutedsl"),
    ):
        item = run(
            f"gdn_{code}_fp32",
            "gdn_fp32",
            replace(
                graph_winner.tuning,
                linear_decode_backend=decode,
                linear_prefill_backend=prefill,
                mamba_ssm_dtype="float32",
            ),
            graph_winner,
        )
        if isinstance(item, CandidateOutcome):
            gdn.append(item)
    ranked_gdn = _rank_phase(gdn)
    if len(ranked_gdn) < 2:
        raise QualificationWorkerError("GDN phase did not retain two safe finalists")

    state_refs: list[CandidateOutcome] = []
    for gdn_outcome in ranked_gdn[:2]:
        code = gdn_outcome.identity.candidate_id.split("_")[1]
        state_refs.append(
            required(
                run(
                    f"state_{code}_fp32_ref",
                    "state_dtype",
                    replace(gdn_outcome.tuning, mamba_ssm_dtype="float32"),
                    gdn_outcome,
                ),
                label=f"state_{code}_fp32_ref",
            )
        )
    # PR 36556's exact-card evidence identifies FlashInfer decode/verify with
    # BF16 state and Triton prefill as the fast linear-attention path.  Give it
    # a dedicated Triton/FP32 causal reference so it competes in the same
    # multimodal, behavior, long-context, and resource-qualified state sweep.
    state_refs.append(
        required(
            run(
                "state_ft_fp32_ref",
                "state_dtype",
                replace(
                    graph_winner.tuning,
                    linear_decode_backend="triton",
                    linear_prefill_backend="triton",
                    mamba_ssm_dtype="float32",
                ),
                graph_winner,
            ),
            label="state_ft_fp32_ref",
        )
    )
    state_bf16: list[CandidateOutcome] = []
    for state_ref in state_refs:
        code = state_ref.identity.candidate_id.split("_")[1]
        changes: dict[str, str] = {"mamba_ssm_dtype": "bfloat16"}
        if code == "ft":
            changes.update(
                linear_decode_backend="flashinfer",
                linear_prefill_backend="triton",
            )
        item = run(
            f"state_{code}_bf16",
            "state_dtype",
            replace(state_ref.tuning, **changes),
            state_ref,
        )
        if isinstance(item, CandidateOutcome):
            state_bf16.append(item)
    equivalent_state = _state_dtype_admissible_candidates(state_refs, state_bf16)
    ranked_state = _rank_phase(equivalent_state)
    if not {item.identity.candidate_id for item in state_refs} <= {
        item.identity.candidate_id for item in ranked_state
    }:
        raise QualificationWorkerError("state FP32 reference baseline did not survive")
    state_winner = ranked_state[0]

    mtp_prelim: list[CandidateOutcome] = []
    for steps, drafts in sorted(LEGAL_NEXTN):
        item = run(
            f"mtp_s{steps}_d{drafts}",
            "mtp_prelim",
            replace(state_winner.tuning, nextn=(steps, drafts)),
            state_winner,
        )
        if isinstance(item, CandidateOutcome):
            mtp_prelim.append(item)
    ranked_mtp = _rank_phase(mtp_prelim)
    if len(ranked_mtp) < 2:
        raise QualificationWorkerError("MTP preliminary phase lacks two safe finalists")
    mtp_finalist_reference = required(
        run(
            "mtp_none_finalist_ref",
            "mtp_finalist",
            replace(ranked_mtp[0].tuning, nextn=None),
            ranked_mtp[0],
        ),
        label="mtp_none_finalist_ref",
    )
    finalist_by_id: dict[str, CandidateOutcome] = {}
    for outcome in ranked_mtp[:2]:
        item = run(
            f"{outcome.identity.candidate_id}_forward",
            "mtp_finalist",
            outcome.tuning,
            outcome,
        )
        if isinstance(item, CandidateOutcome):
            finalist_by_id[item.identity.candidate_id] = item
    for outcome in reversed(ranked_mtp[:2]):
        item = run(
            f"{outcome.identity.candidate_id}_reverse",
            "mtp_finalist",
            outcome.tuning,
            outcome,
        )
        if isinstance(item, CandidateOutcome):
            finalist_by_id[item.identity.candidate_id] = item
    complete_finalists = [
        finalist_by_id[f"{base}_{direction}"]
        for base in sorted(outcome.identity.candidate_id for outcome in ranked_mtp[:2])
        if all(
            f"{base}_{direction}" in finalist_by_id
            for direction in ("forward", "reverse")
        )
        for direction in ("forward", "reverse")
    ]
    if not complete_finalists:
        raise QualificationWorkerError(
            "MTP finalist phase has no complete counterbalanced setting"
        )
    finalist_winner = _rank_mtp_finalists(mtp_finalist_reference, complete_finalists)

    replay_reference = required(
        run(
            "replay_none_ref",
            "replay",
            _replay_off_reference_tuning(finalist_winner.tuning),
            finalist_winner,
        ),
        label="replay_none_ref",
    )
    replay_complete = [replay_reference]
    for candidate_id, prefill in (
        ("replay_tt_fp32", "triton"),
        ("replay_tc_fp32", "cutedsl"),
    ):
        item = run(
            candidate_id,
            "replay",
            replace(
                replay_reference.tuning,
                replay_ssm=True,
                linear_decode_backend="triton",
                linear_prefill_backend=prefill,
                mamba_ssm_dtype="float32",
            ),
            replay_reference,
        )
        if isinstance(item, CandidateOutcome):
            replay_complete.append(item)
    ranked_replay = _rank_phase(replay_complete)
    if replay_reference not in ranked_replay:
        raise QualificationWorkerError("Replay-off reference did not survive")
    replay_winner = ranked_replay[0]

    chunk_4096 = required(
        run(
            "chunk_4096",
            "chunk",
            replace(replay_winner.tuning, chunked_prefill_size=4096),
            replay_winner,
        ),
        label="chunk_4096",
    )
    chunk_complete = [chunk_4096]
    chunk_8192_evidence = run(
        "chunk_8192",
        "chunk",
        replace(replay_winner.tuning, chunked_prefill_size=8192),
        replay_winner,
    )
    if isinstance(chunk_8192_evidence, CandidateOutcome):
        chunk_complete.append(chunk_8192_evidence)
    ranked_chunk = _rank_phase(chunk_complete)
    if chunk_4096 not in ranked_chunk:
        raise QualificationWorkerError("chunk 4096 safe reference did not survive")
    chunk_winner = ranked_chunk[0]

    memory: list[CandidateOutcome] = []
    for suffix, fraction in (("084", "0.84"), ("086", "0.86"), ("088", "0.88")):
        item = run(
            f"mem_{suffix}",
            "memory",
            replace(chunk_winner.tuning, mem_fraction_static=fraction),
            chunk_winner,
        )
        if isinstance(item, CandidateOutcome):
            memory.append(item)
            if item.report.get("passed") is True:
                break
    memory_winner = _rank_phase(memory, memory_phase=True)[0]
    return evidence, memory_winner


def _pipeline(request: Mapping[str, Any]) -> dict[str, Any]:
    paths = _paths(request)
    paths["output"].mkdir(mode=0o700, exist_ok=False)
    checkpoint = release_tool.validate_checkpoint(
        paths["checkpoint"],
        expected_builder_sha256=str(request["builder_sha256"]),
        verify_hashes=True,
    )
    if checkpoint.checkpoint_tree_sha256 != request["checkpoint_tree_sha256"]:
        raise QualificationWorkerError("staged builder checkpoint identity changed")
    validate_sibling_artifact(
        paths["build"],
        expected_tuned_tree_sha256=str(request["checkpoint_tree_sha256"]),
        expected_untuned_tree_sha256=str(
            request["official_untuned_checkpoint_tree_sha256"]
        ),
        expected_manifest_sha256=str(request["build_sibling_manifest_sha256"]),
        verify_hashes=True,
        require_hardlinks=True,
    )
    _image_preflight()
    candidates, winner = _selection_pipeline(request)
    # The final pair differs only in the speculative fields.  Replay/GDN/graph,
    # chunk, memory, and state-dtype winner settings stay identical so the MTP
    # speed gate is causal rather than a confounded configuration comparison.
    off_tuning = _final_mtp_off_tuning(winner.tuning)
    _progress(request, "final", ARM_OFFICIAL_UNTUNED)
    official, _official_command = _run_arm(request, ARM_OFFICIAL_UNTUNED, off_tuning)
    _require_behavioral_improvement_possible(official)
    _progress(request, "final", ARM_TUNED_MTP_OFF)
    off, _off_command = _run_arm(request, ARM_TUNED_MTP_OFF, off_tuning)
    _require_behavioral_improvement_possible(official, off)
    _progress(request, "final", ARM_TUNED_MTP_ON)
    on, _on_command = _run_arm(request, ARM_TUNED_MTP_ON, winner.tuning)
    for candidate in candidates:
        if isinstance(candidate, CandidateAttempt):
            _revalidate_candidate_attempt_diagnostic(request, candidate)
    comparison_path = paths["output"] / "final.comparison.json"
    comparison = qualify.compare_arms(
        argparse.Namespace(
            official_untuned_report=(
                paths["output"] / f"{ARM_OFFICIAL_UNTUNED}.arm.json"
            ),
            tuned_mtp_off_report=(paths["output"] / f"{ARM_TUNED_MTP_OFF}.arm.json"),
            selection_candidate_report=[item.report_path for item in candidates],
            tuned_mtp_on_winner_report=(
                paths["output"] / f"{ARM_TUNED_MTP_ON}.arm.json"
            ),
            bootstrap_samples=qualify.DEFAULT_BOOTSTRAP_SAMPLES,
            output=comparison_path,
        )
    )
    if comparison.get("passed") is not True:
        raise QualificationWorkerError("MTP comparison did not pass")
    _atomic_json(
        paths["output"] / "release-runtime.json",
        _runtime_release_config(
            request,
            off,
            on,
            off_tuning=off_tuning,
            on_tuning=winner.tuning,
        ),
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "success": True,
        "runtime_id": request["runtime_id"],
        "job_id": request["job_id"],
        "checkpoint_tree_sha256": request["checkpoint_tree_sha256"],
        "official_untuned_checkpoint_tree_sha256": request[
            "official_untuned_checkpoint_tree_sha256"
        ],
        "build_sibling_manifest_sha256": request["build_sibling_manifest_sha256"],
        "official_untuned_report_sha256": _sha256(
            paths["output"] / f"{ARM_OFFICIAL_UNTUNED}.arm.json"
        ),
        "selection_candidate_count": len(candidates),
        "selected_candidate_id": winner.identity.candidate_id,
        "selected_config_sha256": winner.config_sha256,
        "selected_moe_runner_backend": winner.tuning.moe_runner_backend,
        "preferred_moe_runner_backend": (runtime_contract.PREFERRED_MOE_RUNNER_BACKEND),
        "cutlass_nvfp4_scale_duplication_bytes": (
            runtime_contract.CUTLASS_NVFP4_SCALE_DUPLICATION_BYTES
        ),
        "mtp_off_decode_tps": comparison["throughput"]["tuned_mtp_off"],
        "mtp_on_decode_tps": comparison["throughput"]["tuned_mtp_on_winner"],
        "mtp_speedup": comparison["throughput"]["speedup"],
        "mtp_ci_lower": comparison["throughput"]["ci_lower"],
        "max_accounted_vram_gb": VRAM_BUDGET_GB,
        "max_cgroup_memory_gb": TASK_MEMORY_GB,
        "completed_at": _now(),
    }
    _atomic_json(paths["output"] / "result.json", result)
    result["manifest_sha256"] = _write_manifest(paths["output"])
    return result


def _process_start_ticks(pid: int) -> int:
    payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    end = payload.rfind(")")
    return int(payload[end + 2 :].split()[19])


def _process_exact(receipt: Mapping[str, Any]) -> bool:
    pid = receipt.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if argv and argv[-1] == b"":
            argv.pop()
        return (
            Path(f"/proc/{pid}").stat().st_uid == os.geteuid()
            and _process_start_ticks(pid) == receipt.get("start_ticks")
            and [item.decode("utf-8") for item in argv] == receipt.get("argv")
        )
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
        return False


def _pid_absent(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    try:
        Path(f"/proc/{pid}").stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _preflight(request: Mapping[str, Any], digest: str) -> dict[str, Any]:
    paths = _paths(request)
    if paths["preflight"].exists() or paths["preflight"].is_symlink():
        value = _read_json(paths["preflight"])
        if value.get("request_sha256") == digest:
            return value
        raise QualificationWorkerError("preflight receipt changed")
    checkpoint = release_tool.validate_checkpoint(
        paths["checkpoint"],
        expected_builder_sha256=str(request["builder_sha256"]),
        verify_hashes=True,
    )
    if checkpoint.checkpoint_tree_sha256 != request["checkpoint_tree_sha256"]:
        raise QualificationWorkerError("checkpoint tree differs from job payload")
    validate_sibling_artifact(
        paths["build"],
        expected_tuned_tree_sha256=str(request["checkpoint_tree_sha256"]),
        expected_untuned_tree_sha256=str(
            request["official_untuned_checkpoint_tree_sha256"]
        ),
        expected_manifest_sha256=str(request["build_sibling_manifest_sha256"]),
        verify_hashes=True,
        require_hardlinks=True,
    )
    _image_preflight()
    value = {
        "schema_version": SCHEMA,
        "request_sha256": digest,
        "checkpoint_tree_sha256": checkpoint.checkpoint_tree_sha256,
        "official_untuned_checkpoint_tree_sha256": request[
            "official_untuned_checkpoint_tree_sha256"
        ],
        "build_sibling_manifest_sha256": request["build_sibling_manifest_sha256"],
        "sglang_commit": SGLANG_COMMIT,
        "sglang_image_digest": IMAGE_DIGEST,
        "sglang_image_config_digest": IMAGE_CONFIG_DIGEST,
        "sglang_image_id": IMAGE_ID,
        "sglang_image_archive_sha256": IMAGE_ARCHIVE_SHA256,
        "max_accounted_vram_gb": VRAM_BUDGET_GB,
        "max_cgroup_memory_gb": TASK_MEMORY_GB,
        "preferred_moe_runner_backend": (runtime_contract.PREFERRED_MOE_RUNNER_BACKEND),
        "qualification_moe_runner_backends": list(
            runtime_contract.QUALIFICATION_MOE_RUNNER_BACKENDS
        ),
        "cutlass_nvfp4_scale_duplication_bytes": (
            runtime_contract.CUTLASS_NVFP4_SCALE_DUPLICATION_BYTES
        ),
        "cutlass_min_cuda_reserve_bytes": (
            runtime_contract.CUTLASS_MIN_CUDA_RESERVE_BYTES
        ),
        "cutlass_min_geometric_mean_speedup": (
            runtime_contract.CUTLASS_MIN_GEOMETRIC_MEAN_SPEEDUP
        ),
        "scratch_device": paths["scratch"].lstat().st_dev,
    }
    _atomic_json(paths["preflight"], value)
    return value


def _spawn(
    request: Mapping[str, Any], request_path: Path, digest: str
) -> dict[str, Any]:
    paths = _paths(request)
    if paths["spawn"].exists() or paths["spawn"].is_symlink():
        receipt = _read_json(paths["spawn"])
        if _process_exact(receipt):
            return {"pid": receipt["pid"], "state": "running"}
        raise QualificationWorkerError("qualification spawn receipt already exists")
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "pipeline",
        str(request_path),
        digest,
    ]
    stdout = os.open(
        paths["scratch"] / "qualification.stdout.log",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    stderr = os.open(
        paths["scratch"] / "qualification.stderr.log",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        process = subprocess.Popen(
            [LOW_PRIORITY, *argv],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
            env={
                "HOME": "/home/aday",
                "PATH": "/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONPATH": str(paths["source"]),
                "PYTHONDONTWRITEBYTECODE": "1",
                "GPU_AGENT_CLAIM_ID": str(request["claim_id"]),
                "CUDA_VISIBLE_DEVICES": str(request["gpu_uuid"]),
                "GPU_MEM_LIMIT_GB": "88",
            },
        )
    finally:
        os.close(stdout)
        os.close(stderr)
    receipt = {
        "schema_version": SCHEMA,
        "runtime_id": request["runtime_id"],
        "request_sha256": digest,
        "pid": process.pid,
        "start_ticks": _process_start_ticks(process.pid),
        "argv": argv,
    }
    _atomic_json(paths["spawn"], receipt)
    return {"pid": process.pid, "state": "running"}


def _status(request: Mapping[str, Any]) -> dict[str, Any]:
    paths = _paths(request)
    receipt = _read_json(paths["spawn"])
    if _process_exact(receipt):
        status = _read_json(paths["status"]) if paths["status"].is_file() else {}
        return {"state": "running", "pid": receipt["pid"], "status": status}
    if not _pid_absent(receipt.get("pid")):
        return {"state": "unknown", "pid": receipt.get("pid")}
    status = _read_json(paths["status"])
    return {"state": status.get("state"), "pid": None, "status": status}


def _stop(request: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _read_json(_paths(request)["spawn"])
    pid = receipt.get("pid")
    if _pid_absent(pid):
        return {"process_absent": True}
    if not _process_exact(receipt):
        raise QualificationWorkerError("supervisor process identity changed")
    os.kill(int(pid), signal.SIGTERM)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if _pid_absent(pid):
            return {"process_absent": True}
        if not _process_exact(receipt):
            raise QualificationWorkerError("supervisor PID identity changed")
        time.sleep(0.25)
    return {"process_absent": False}


def _safe_tree_bytes(root: Path) -> int:
    metadata = _private_dir(root).lstat()
    seen = {(metadata.st_dev, metadata.st_ino)}
    total = metadata.st_blocks * 512
    for item in root.rglob("*"):
        observed = item.lstat()
        if (
            observed.st_uid != os.geteuid()
            or observed.st_dev != metadata.st_dev
            or stat.S_ISLNK(observed.st_mode)
            or os.path.ismount(item)
            or not (stat.S_ISREG(observed.st_mode) or stat.S_ISDIR(observed.st_mode))
        ):
            raise QualificationWorkerError(
                "canonical qualification tree contains an unsafe inode"
            )
        key = (observed.st_dev, observed.st_ino)
        if key not in seen:
            total += observed.st_blocks * 512
            seen.add(key)
    return total


def _settle_status(request: Mapping[str, Any]) -> dict[str, Any]:
    status = _status(request)
    if status.get("state") != "completed":
        raise QualificationWorkerError("qualification is not complete")
    manifest = _paths(request)["manifest"]
    return {"state": "settle_ready", "manifest_sha256": _sha256(manifest)}


def _mark_settled(request: Mapping[str, Any], digest: str) -> dict[str, Any]:
    if _settle_status(request)["manifest_sha256"] != digest:
        raise QualificationWorkerError("settled output manifest changed")
    _atomic_json(
        _paths(request)["settled"],
        {"runtime_id": request["runtime_id"], "manifest_sha256": digest},
    )
    return {"state": "settled", "manifest_sha256": digest}


def _cleanup(request: Mapping[str, Any], digest: str) -> dict[str, Any]:
    marker = _read_json(_paths(request)["settled"])
    if (
        marker.get("runtime_id") != request["runtime_id"]
        or marker.get("manifest_sha256") != digest
        or _sha256(_paths(request)["manifest"]) != digest
        or _status(request).get("state") != "completed"
    ):
        raise QualificationWorkerError("worker output is not durably settled")
    # Qualification is local-only on canonical .177. Current fleet policy never
    # grants automatic deletion authority there. Validate the retained canonical
    # run before closing storage, but report no reclaimed bytes and leave every
    # inode in place for separate operator review.
    _safe_tree_bytes(_paths(request)["scratch"])
    return {"state": "retained", "reclaimed_bytes": 0}


def _cleanup_prelaunch(request: Mapping[str, Any]) -> dict[str, Any]:
    paths = _paths(request)
    if paths["spawn"].exists() or paths["status"].exists() or paths["output"].exists():
        raise QualificationWorkerError("prelaunch cleanup found lifecycle output")
    _safe_tree_bytes(paths["scratch"])
    return {"state": "retained", "reclaimed_bytes": 0}


def _signal_handler(_signum: int, _frame: Any) -> None:
    if _active is not None:
        arm, request, command, candidate = _active
        try:
            _remove_container(
                request, arm, str(request["container_id"]), command, candidate
            )
        except BaseException:
            pass
    raise KeyboardInterrupt("qualification supervisor terminated")


def _pipeline_entry(request: Mapping[str, Any]) -> int:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    status_path = _paths(request)["status"]
    _atomic_json(status_path, {"state": "running", "started_at": _now()})
    try:
        result = _pipeline(request)
    except BaseException as exc:
        _atomic_json(
            status_path,
            {
                "state": "failed",
                "failure_type": type(exc).__name__,
                "failure": str(exc)[:1000],
                "completed_at": _now(),
            },
        )
        return 1
    _atomic_json(
        status_path,
        {"state": "completed", "result": result, "completed_at": _now()},
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "preflight",
            "spawn",
            "status",
            "stop",
            "settle-status",
            "mark-settled",
            "cleanup",
            "cleanup-prelaunch",
            "pipeline",
        ),
    )
    parser.add_argument("request", type=Path)
    parser.add_argument("digest")
    parser.add_argument("extra", nargs="?")
    args = parser.parse_args(argv)
    try:
        request = _validate_request(
            args.request,
            args.digest,
            require_environment=args.action == "pipeline",
        )
        if args.action == "pipeline":
            return _pipeline_entry(request)
        if args.action == "preflight":
            result = _preflight(request, args.digest)
        elif args.action == "spawn":
            result = _spawn(request, args.request, args.digest)
        elif args.action == "status":
            result = _status(request)
        elif args.action == "stop":
            result = _stop(request)
        elif args.action == "settle-status":
            result = _settle_status(request)
        elif args.action == "mark-settled" and args.extra is not None:
            result = _mark_settled(request, args.extra)
        elif args.action == "cleanup" and args.extra is not None:
            result = _cleanup(request, args.extra)
        elif args.action == "cleanup-prelaunch":
            result = _cleanup_prelaunch(request)
        else:
            raise QualificationWorkerError("action requires an exact manifest digest")
    except BaseException as exc:
        print(
            json.dumps(
                {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:500]},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"ok": True, "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
