"""Fail-closed Fleet service adapter for the qualified Flash-Next release.

The checked-in profile is disabled and contains zero-digest release placeholders.
Only the explicit promotion finalizer may create the private production binding and
replace those placeholders.  This adapter never pulls an image, downloads weights,
or searches Docker state.  It operates on one receipt-bound container name/ID,
one immutable thin release, and one independently verified offline-materialized
canonical checkpoint on the .177 orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence

import requests

from fleet_compute.adapters import AdapterLaunchError, RuntimeContext
from fleet_compute.models import (
    ComputeProfile,
    LaunchResult,
    Lease,
    ProbeResult,
    ProbeState,
    StopResult,
    StorageFinalizationResult,
    StoragePreparationResult,
)

from aeon.core import qwen_flash_next_runtime_contract as runtime_contract
from aeon.scripts import materialize_qwen38_flash_next_ple as ple_materializer
from aeon.scripts import release_qwen38_flash_next as release_tool


PROFILE_ID = "aeon-qwen38-flash-next-177"
SERVICE_ID = "aeon-qwen38-standard"
SERVING_POOL_ID = "aeon-qwen38-standard-multimodal-v2"
FLASH_LANE_MAX_REPLICAS = 1
SERVED_ALIAS = runtime_contract.WIRE_SERVED_ALIAS
DISPLAY_NAME = runtime_contract.DISPLAY_NAME
ARTIFACT_NAME = runtime_contract.ARTIFACT_NAME
HOST = "192.168.0.177"
HOSTNAME = "DAY2RTX6000PRO"
PHYSICAL_GPU = 0
VRAM_BUDGET_GB = 88.0
MIN_PHYSICAL_VRAM_GB = 94.0
MAX_TASK_MEMORY_GB = 200.0
HOST_PORT = 18038
CONTAINER_PORT = 30000

LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
DOCKER = "/usr/bin/docker"
RELEASE_DIR = Path(
    "/home/aday/.local/state/aeon-flash-next/releases/"
    "Aeon-Qwen3.8-Flash-Next-NVFP4-MTP"
)
MATERIALIZED_MODEL_DIR = Path(
    "/home/aday/.local/state/aeon-flash-next/materialized/"
    "Aeon-Qwen3.8-Flash-Next-NVFP4-MTP"
)
MATERIALIZATION_RECEIPT = Path(
    "/home/aday/.local/state/aeon-flash-next/materialized/"
    "Aeon-Qwen3.8-Flash-Next-NVFP4-MTP.materialization-receipt.json"
)
BINDING_PATH = Path(
    "/home/aday/.local/state/aeon-flash-next/releases/production-service-binding.json"
)
PUBLICATION_ROOT = Path("/home/aday/.local/state/aeon-flash-next/releases")
QUALIFICATION_ASSET_ROOT = Path(
    "/home/aday/.local/state/aeon-flash-next/qualification-assets"
)
QUALIFICATION_ASSET_MANIFEST = QUALIFICATION_ASSET_ROOT / "manifest.json"
QUALIFICATION_ASSET_MANIFEST_SHA256 = (
    "dd8a1138007e0f17ba2ad50f045fd327a0b7bb1714c45d1e1d648434d835547f"
)
QUALIFICATION_ASSETS = {
    "image": (
        "candy.JPG",
        "fc417c899e94f8df465b7541c5a70f0eebb85c414d06345f0b290c061eccc84c",
        2_289_891,
    ),
    "video": (
        "jobs_presenting_ipod.mp4",
        "7e89e814848b25f65161e8bf988b2aaadbe707b15b2e8e55e095e3b851e63041",
        1_114_800,
    ),
}

SGLANG_COMMIT = release_tool.SGLANG_COMMIT
SGLANG_IMAGE_DIGEST = release_tool.SGLANG_IMAGE_DIGEST
SGLANG_IMAGE_REFERENCE = release_tool.SGLANG_IMAGE_REFERENCE
SGLANG_IMAGE_CONFIG_DIGEST = runtime_contract.QUALIFIED_IMAGE_CONFIG_DIGEST
SGLANG_IMAGE_ID = runtime_contract.QUALIFIED_LOCAL_DOCKER_IMAGE_ID
SGLANG_IMAGE_ARCHIVE_SHA256 = runtime_contract.QUALIFIED_IMAGE_ARCHIVE_SHA256
SGLANG_SOURCE_COMMIT_SHA256 = hashlib.sha256(SGLANG_COMMIT.encode("ascii")).hexdigest()
SGLANG_SOURCE_STACK_SHA256 = runtime_contract.SOURCE_STACK_SHA256
BINDING_SCHEMA = "aeon-qwen38-flash-next-service-binding-v2"
PROCESS_PREFIX = "aeon-flash-next-service"
RECEIPT_NAME = "flash-next-service-container.json"
ZERO_SHA256 = "0" * 64
CONSTANT_RUNTIME_ENV = {
    "HF_HUB_OFFLINE": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "USE_FLAX": "0",
    "USE_TF": "0",
}
MATERIALIZED_MODEL_PLACEHOLDER = "@AEON_MATERIALIZED_MODEL_PATH@"
MATERIALIZED_MODEL_MOUNT = (
    f"type=bind,src={MATERIALIZED_MODEL_PLACEHOLDER},dst=/model,readonly"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUNTIME_ID_RE = re.compile(r"^fr-[0-9a-f]{32}$")
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_PROCESS_RE = re.compile(
    r"^aeon-flash-next-service:(fr-[0-9a-f]{32}):([0-9a-f]{64}):"
    r"([0-9a-f]{64}):([0-9]+):([0-9]+)$"
)


class FlashNextServiceError(RuntimeError):
    """The production release or exact container identity failed closed."""


@dataclass(frozen=True)
class ProductionBinding:
    binding_path: Path
    binding_sha256: str
    release_dir: Path
    repo_id: str
    publication_receipt: Path
    release_tree_sha256: str
    release_manifest_sha256: str
    checkpoint_tree_sha256: str
    materialized_model_dir: Path
    materialized_checkpoint_tree_sha256: str
    ple_materialization_manifest_sha256: str
    ple_materializer_sha256: str
    materialization_receipt: Path
    materialization_receipt_sha256: str
    materialized_model_size_bytes: int
    materialized_model_inode_count: int
    release_size_bytes: int
    release_inode_count: int
    runtime_config_sha256: str
    qualification_sha256: str
    qualification_mtp_off_sha256: str
    qualification_mtp_on_sha256: str
    task_memory_bytes: int
    container_command: tuple[str, ...]
    runtime_environment: tuple[tuple[str, str], ...] = tuple(
        sorted(CONSTANT_RUNTIME_ENV.items())
    )

    @property
    def command_sha256(self) -> str:
        return _canonical_sha(
            {
                "command": list(self.container_command),
                "environment": dict(self.runtime_environment),
            }
        )

    @property
    def artifact_identity(self) -> dict[str, str]:
        return {
            "adapter_source": _sha256(Path(__file__)),
            "binding": self.binding_sha256,
            "checkpoint_tree": self.checkpoint_tree_sha256,
            "image": SGLANG_IMAGE_DIGEST.removeprefix("sha256:"),
            "image_archive": SGLANG_IMAGE_ARCHIVE_SHA256,
            "image_config": SGLANG_IMAGE_CONFIG_DIGEST.removeprefix("sha256:"),
            "image_local_id": SGLANG_IMAGE_ID.removeprefix("sha256:"),
            "materialized_checkpoint_tree": (
                self.materialized_checkpoint_tree_sha256
            ),
            "materialization_receipt": self.materialization_receipt_sha256,
            "ple_materialization_manifest": (
                self.ple_materialization_manifest_sha256
            ),
            "ple_materializer": self.ple_materializer_sha256,
            "publication_receipt": _sha256(self.publication_receipt),
            "qualification": self.qualification_sha256,
            "qualification_assets_manifest": QUALIFICATION_ASSET_MANIFEST_SHA256,
            "qualification_mtp_off": self.qualification_mtp_off_sha256,
            "qualification_mtp_on": self.qualification_mtp_on_sha256,
            "release_manifest": self.release_manifest_sha256,
            "release_tree": self.release_tree_sha256,
            "runtime_config": self.runtime_config_sha256,
            "sglang_source_commit": SGLANG_SOURCE_COMMIT_SHA256,
        }


@dataclass(frozen=True)
class MaterializedCheckpointEvidence:
    checkpoint: release_tool.CheckpointEvidence
    receipt_sha256: str
    ple_manifest_sha256: str
    materializer_sha256: str
    model_size_bytes: int
    model_inode_count: int


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


def binding_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _safe_json(path: Path, *, maximum: int = 8 * 1024 * 1024) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FlashNextServiceError(f"required private receipt is absent: {path}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= maximum
    ):
        raise FlashNextServiceError(f"private receipt is unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FlashNextServiceError(f"private receipt is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise FlashNextServiceError(f"private receipt is not an object: {path}")
    return value


def _validate_publication_receipt(
    publication: Mapping[str, Any],
    *,
    repo_id: str,
    release: Mapping[str, Any],
) -> None:
    """Validate the exact private-Hub receipt emitted by the release tool."""

    try:
        owner = release_tool._validate_repo_id(repo_id)
    except release_tool.ReleaseError as exc:
        raise FlashNextServiceError("publication repository identity is invalid") from exc
    if set(publication) != release_tool.PUBLICATION_RECEIPT_FIELDS:
        raise FlashNextServiceError("private publication receipt fields changed")
    created_at = publication.get("created_at")
    try:
        created = datetime.fromisoformat(str(created_at))
    except ValueError as exc:
        raise FlashNextServiceError(
            "private publication timestamp is malformed"
        ) from exc
    if created.tzinfo is None or created.utcoffset() is None:
        raise FlashNextServiceError("private publication timestamp lacks a timezone")
    files = release.get("files")
    root = release.get("root")
    if not isinstance(files, Mapping) or not isinstance(root, Path):
        raise FlashNextServiceError("validated release upload inventory is malformed")
    try:
        sums_size = release_tool._safe_file(
            root / "SHA256SUMS", maximum=16 * 1024 * 1024
        ).st_size
        expected_upload_bytes = sums_size + sum(
            size for _digest_value, size in files.values()
        )
    except (OSError, TypeError, ValueError, release_tool.ReleaseError) as exc:
        raise FlashNextServiceError(
            "validated release upload accounting is malformed"
        ) from exc
    expected_wheels = {
        "requests": {
            "version": release_tool.REQUESTS_VERSION,
            "sha256": release_tool.REQUESTS_WHEEL_SHA256,
        },
        "charset_normalizer": {
            "version": release_tool.CHARSET_NORMALIZER_VERSION,
            "sha256": release_tool.CHARSET_NORMALIZER_WHEEL_SHA256,
        },
        "urllib3": {
            "version": release_tool.URLLIB3_VERSION,
            "sha256": release_tool.URLLIB3_WHEEL_SHA256,
        },
    }
    if (
        publication.get("schema_version")
        != release_tool.PUBLICATION_RECEIPT_SCHEMA
        or publication.get("complete") is not True
        or publication.get("repo_id") != repo_id
        or publication.get("repo_type") != "model"
        or publication.get("visibility") != "private"
        or publication.get("authenticated_username") != owner
        or publication.get("huggingface_hub_version")
        != release_tool.HF_HUB_VERSION
        or publication.get("hf_xet_version") != release_tool.HF_XET_VERSION
        or publication.get("huggingface_hub_wheel_sha256")
        != release_tool.HF_HUB_WHEEL_SHA256
        or publication.get("hf_xet_wheel_sha256")
        != release_tool.HF_XET_WHEEL_SHA256
        or publication.get("release_validator_wheels") != expected_wheels
        or publication.get("upload_wheel_files_rehashed") is not True
        or publication.get("hf_xet_high_performance") is not True
        or publication.get("upload_bytes") != expected_upload_bytes
        or publication.get("verified_private_quota_bytes")
        != release_tool.FREE_PRIVATE_STORAGE_BYTES
        or not isinstance(publication.get("commit"), str)
        or _COMMIT_RE.fullmatch(str(publication.get("commit"))) is None
        or publication.get("remote_files") != len(files) + 1
        or publication.get("release_tree_sha256")
        != release.get("release_tree_sha256")
        or publication.get("release_manifest_sha256")
        != release.get("manifest_sha256")
        or publication.get("verification")
        != dict(release_tool.PUBLICATION_VERIFICATION)
    ):
        raise FlashNextServiceError("private Hugging Face publication is not verified")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    parent = path.parent
    metadata = parent.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise FlashNextServiceError("service receipt directory is unsafe")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        payload = binding_bytes(value)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FlashNextServiceError("service receipt write was incomplete")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FlashNextServiceError(f"{label} is not a SHA-256 digest")
    return value


def _verify_qualification_assets() -> None:
    if _sha256(QUALIFICATION_ASSET_MANIFEST) != QUALIFICATION_ASSET_MANIFEST_SHA256:
        raise FlashNextServiceError("qualification asset manifest changed")
    manifest = _safe_json(QUALIFICATION_ASSET_MANIFEST, maximum=64 * 1024)
    if manifest.get("schema_version") != (
        "aeon-qwen38-flash-next-qualification-assets-v1"
    ):
        raise FlashNextServiceError("qualification asset schema changed")
    for kind, (name, digest, size) in QUALIFICATION_ASSETS.items():
        record = manifest.get(kind)
        path = QUALIFICATION_ASSET_ROOT / name
        metadata = path.lstat()
        if (
            not isinstance(record, Mapping)
            or record.get("file") != name
            or record.get("sha256") != digest
            or record.get("size") != size
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
            or metadata.st_size != size
            or _sha256(path) != digest
        ):
            raise FlashNextServiceError(f"qualified {kind} asset changed")


def _validate_materialized_checkpoint(
    release: Mapping[str, Any],
    *,
    expected_builder_sha256: str,
    verify_hashes: bool,
) -> MaterializedCheckpointEvidence:
    """Bind the thin release to one independently verified canonical model tree."""

    try:
        release_tool._safe_directory(MATERIALIZED_MODEL_DIR.parent)
        release_tool._safe_directory(MATERIALIZED_MODEL_DIR)
    except release_tool.ReleaseError as exc:
        raise FlashNextServiceError(
            f"offline-materialized model directory is unsafe: {exc}"
        ) from exc
    if MATERIALIZED_MODEL_DIR.is_symlink() or (
        MATERIALIZATION_RECEIPT.parent != MATERIALIZED_MODEL_DIR.parent
    ):
        raise FlashNextServiceError("materialized model path contract changed")
    if (MATERIALIZED_MODEL_DIR / ".cache").exists() or (
        MATERIALIZED_MODEL_DIR / ".cache"
    ).is_symlink():
        raise FlashNextServiceError("materialized model contains unreceipted cache state")

    manifest = release.get("manifest")
    files = release.get("files")
    if not isinstance(manifest, Mapping) or not isinstance(files, Mapping):
        raise FlashNextServiceError("thin release evidence is malformed")
    packaging = manifest.get("packaging")
    if not isinstance(packaging, Mapping):
        raise FlashNextServiceError("thin release packaging receipt is absent")
    ple_path = RELEASE_DIR / release_tool.PLE_MATERIALIZATION_FILENAME
    ple_manifest = _safe_json(ple_path, maximum=8 * 1024 * 1024)
    ple_manifest_sha256 = _sha256(ple_path)
    materializer_sha256 = _sha256(
        RELEASE_DIR / release_tool.PLE_MATERIALIZER_FILENAME
    )
    if (
        ple_manifest_sha256
        != packaging.get("ple_materialization_manifest_sha256")
        or materializer_sha256 != packaging.get("ple_materializer_sha256")
        or ple_manifest.get("materializer_sha256") != materializer_sha256
    ):
        raise FlashNextServiceError("thin release materialization identities changed")

    try:
        checkpoint = release_tool.validate_checkpoint(
            MATERIALIZED_MODEL_DIR,
            expected_builder_sha256=expected_builder_sha256,
            verify_hashes=verify_hashes,
        )
    except (release_tool.ReleaseError, OSError) as exc:
        raise FlashNextServiceError(
            f"offline-materialized canonical checkpoint is invalid: {exc}"
        ) from exc
    canonical = ple_manifest.get("canonical_files")
    if not isinstance(canonical, Mapping):
        raise FlashNextServiceError("PLE materialization canonical inventory is absent")
    try:
        canonical_files = {
            str(name): (
                _require_digest(receipt.get("sha256"), f"canonical {name}"),
                int(receipt.get("size")),
            )
            for name, receipt in canonical.items()
            if isinstance(receipt, Mapping)
        }
    except (TypeError, ValueError) as exc:
        raise FlashNextServiceError(
            "PLE materialization canonical inventory is malformed"
        ) from exc
    if (
        len(canonical_files) != len(canonical)
        or any(size <= 0 for _digest, size in canonical_files.values())
        or checkpoint.files != canonical_files
        or checkpoint.checkpoint_tree_sha256
        != packaging.get("canonical_checkpoint_tree_sha256")
        or checkpoint.checkpoint_tree_sha256
        != ple_manifest.get("checkpoint_tree_sha256")
    ):
        raise FlashNextServiceError(
            "materialized checkpoint does not close against the thin release"
        )
    try:
        passthrough = release_tool._validate_passthrough_audit(
            RELEASE_DIR / release_tool.PASSTHROUGH_AUDIT_FILENAME,
            checkpoint_root=MATERIALIZED_MODEL_DIR,
            canonical_files=checkpoint.files,
            build_manifest=checkpoint.build_manifest,
            current_index_receipt=checkpoint.files.get(
                "model.safetensors.index.json"
            ),
        )
    except (release_tool.ReleaseError, OSError) as exc:
        raise FlashNextServiceError(
            f"materialized pass-through audit is invalid: {exc}"
        ) from exc
    if manifest.get("passthrough_audit") != release_tool._passthrough_audit_manifest(
        passthrough
    ):
        raise FlashNextServiceError(
            "materialized pass-through audit differs from the release binding"
        )

    receipt = _safe_json(MATERIALIZATION_RECEIPT, maximum=128 * 1024)
    expected_receipt_fields = {
        "schema_version",
        "complete",
        "completed_at",
        "materialized_model_dir_sha256",
        "materialized_checkpoint_tree_sha256",
        "ple_materialization_manifest_sha256",
        "ple_materializer_sha256",
        "official_fp8",
        "canonical_file_count",
        "ple_shard_count",
    }
    completed_at = receipt.get("completed_at")
    try:
        completed = datetime.fromisoformat(str(completed_at))
    except ValueError as exc:
        raise FlashNextServiceError(
            "materialization completion timestamp is malformed"
        ) from exc
    official_fp8 = {
        "repo": ple_materializer.OFFICIAL_REPO,
        "revision": ple_materializer.OFFICIAL_REVISION,
        "files_manifest_sha256": ple_materializer.OFFICIAL_FILES_MANIFEST_SHA256,
        "index_sha256": ple_materializer.OFFICIAL_INDEX_SHA256,
    }
    ple_shards = ple_manifest.get("ple_shards")
    if (
        set(receipt) != expected_receipt_fields
        or receipt.get("schema_version")
        != ple_materializer.COMPLETION_SCHEMA_VERSION
        or receipt.get("complete") is not True
        or not isinstance(completed_at, str)
        or completed.tzinfo is None
        or receipt.get("materialized_model_dir_sha256")
        != hashlib.sha256(str(MATERIALIZED_MODEL_DIR).encode("utf-8")).hexdigest()
        or receipt.get("materialized_checkpoint_tree_sha256")
        != checkpoint.checkpoint_tree_sha256
        or receipt.get("ple_materialization_manifest_sha256")
        != ple_manifest_sha256
        or receipt.get("ple_materializer_sha256") != materializer_sha256
        or receipt.get("official_fp8") != official_fp8
        or receipt.get("canonical_file_count") != len(canonical_files)
        or not isinstance(ple_shards, list)
        or receipt.get("ple_shard_count") != len(ple_shards)
    ):
        raise FlashNextServiceError("materialization completion receipt changed")

    sums_size = (MATERIALIZED_MODEL_DIR / "SHA256SUMS").lstat().st_size
    return MaterializedCheckpointEvidence(
        checkpoint=checkpoint,
        receipt_sha256=_sha256(MATERIALIZATION_RECEIPT),
        ple_manifest_sha256=ple_manifest_sha256,
        materializer_sha256=materializer_sha256,
        model_size_bytes=sum(size for _digest, size in checkpoint.files.values())
        + sums_size,
        model_inode_count=len(checkpoint.files) + 1,
    )


def _flag_values(command: Sequence[str], flag: str) -> list[str | None]:
    values: list[str | None] = []
    for index, item in enumerate(command):
        if item == flag:
            values.append(command[index + 1] if index + 1 < len(command) else None)
        elif item.startswith(flag + "="):
            values.append(item.split("=", 1)[1])
    return values


def _one_flag(command: Sequence[str], flag: str, expected: str) -> None:
    if _flag_values(command, flag) != [expected]:
        raise FlashNextServiceError(
            f"qualified MTP-on command must set {flag}={expected} exactly once"
        )


def _qualified_moe_runner_backend(runtime_config: Any) -> str:
    """Return one reviewed main/speculative backend from qualified evidence."""

    if not isinstance(runtime_config, Mapping):
        raise FlashNextServiceError(
            "qualified MTP-on runtime_config is not an object"
        )
    main = runtime_config.get("moe_runner_backend")
    speculative = runtime_config.get("speculative_moe_runner_backend")
    if (
        main not in runtime_contract.QUALIFICATION_MOE_RUNNER_BACKENDS
        or speculative != main
    ):
        raise FlashNextServiceError(
            "qualified main/speculative MoE backend pair is not reviewed"
        )
    return str(main)


def _production_container_command(
    command: Any,
    *,
    repo_id: str,
    runtime_config: Mapping[str, Any],
    expected_command_sha256: str,
) -> tuple[str, ...]:
    release_tool._validate_repo_id(repo_id)
    qualified_moe_backend = _qualified_moe_runner_backend(runtime_config)
    if (
        not isinstance(command, list)
        or not 2 <= len(command) <= 256
        or not all(isinstance(item, str) and 0 < len(item) <= 4096 for item in command)
        or any(any(char in item for char in "\x00\r\n") for item in command)
    ):
        raise FlashNextServiceError("qualified MTP-on command is malformed")
    image_positions = [
        index for index, item in enumerate(command) if item == SGLANG_IMAGE_REFERENCE
    ]
    if (
        command[:2] != [DOCKER, "run"]
        or len(image_positions) != 1
        or image_positions[0] == len(command) - 1
        or SGLANG_IMAGE_ID in command
        or SGLANG_IMAGE_CONFIG_DIGEST in command
    ):
        raise FlashNextServiceError(
            "qualified command must contain one exact repo@manifest reference, "
            "never a raw OCI config digest or daemon-only image ID"
        )
    image_index = image_positions[0]
    host_prefix = list(command[:image_index])
    expected_environment = {
        f"{key}={value}" for key, value in CONSTANT_RUNTIME_ENV.items()
    }
    observed_environment = _flag_values(host_prefix, "--env")
    if (
        len(observed_environment) != len(expected_environment)
        or set(observed_environment) != expected_environment
    ):
        raise FlashNextServiceError(
            "qualified command omits its exact offline runtime environment"
        )
    if _flag_values(host_prefix, "--mount") != [MATERIALIZED_MODEL_MOUNT]:
        raise FlashNextServiceError(
            "qualified command omits the portable materialized-model mount contract"
        )
    container = list(command[image_index + 1 :])
    if (
        _SHA256_RE.fullmatch(str(expected_command_sha256 or "")) is None
        or _canonical_sha(container) != expected_command_sha256
    ):
        raise FlashNextServiceError(
            "qualified SGLang argv differs from its measured command receipt"
        )
    required = {
        "--model-path": "/model",
        "--served-model-name": SERVED_ALIAS,
        "--host": "0.0.0.0",
        "--port": str(CONTAINER_PORT),
        "--tp-size": "1",
        "--dtype": "bfloat16",
        "--quantization": runtime_contract.QUANTIZATION,
        "--reasoning-parser": runtime_contract.REASONING_PARSER,
        "--prefill-attention-backend": (
            runtime_contract.PREFILL_ATTENTION_BACKEND
        ),
        "--decode-attention-backend": (
            runtime_contract.DECODE_ATTENTION_BACKEND
        ),
        "--context-length": str(
            runtime_contract.SM120_VALIDATED_CONTEXT_LENGTH
        ),
        "--max-total-tokens": str(
            runtime_contract.SM120_VALIDATED_CONTEXT_LENGTH
        ),
        "--page-size": "64",
        "--speculative-draft-model-quantization": (
            runtime_contract.MTP_DRAFT_QUANTIZATION
        ),
        "--speculative-algorithm": "NEXTN",
        "--speculative-eagle-topk": "1",
        "--max-running-requests": "4",
        "--linear-attn-backend": "triton",
        "--moe-a2a-backend": "none",
        "--moe-runner-backend": qualified_moe_backend,
        "--fp4-gemm-backend": runtime_contract.FP4_GEMM_BACKEND,
        "--speculative-moe-a2a-backend": "none",
        "--speculative-moe-runner-backend": qualified_moe_backend,
    }
    for flag, expected in required.items():
        _one_flag(container, flag, expected)
    mamba = _flag_values(container, "--mamba-ssm-dtype")
    decode = _flag_values(container, "--linear-attn-decode-backend")
    prefill = _flag_values(container, "--linear-attn-prefill-backend")
    verify = _flag_values(container, "--linear-attn-verify-backend")
    graph = _flag_values(container, "--cuda-graph-config")
    chunk = _flag_values(container, "--chunked-prefill-size")
    steps = _flag_values(container, "--speculative-num-steps")
    drafts = _flag_values(container, "--speculative-num-draft-tokens")
    allowed_graphs = {
        '{"decode":{"backend":"full","max_bs":4,"bs":[1,2,4]},'
        '"prefill":{"backend":"disabled"}}',
        '{"decode":{"backend":"disabled"},'
        '"prefill":{"backend":"disabled"}}',
    }
    try:
        nextn = (int(str(steps[0])), int(str(drafts[0])))
    except (IndexError, TypeError, ValueError) as exc:
        raise FlashNextServiceError("qualified NEXTN geometry is malformed") from exc
    if (
        len(mamba) != 1
        or mamba[0] not in {"float32", "bfloat16"}
        or len(decode) != 1
        or decode[0] not in {"triton", "cutedsl", "flashinfer"}
        or len(prefill) != 1
        or prefill[0] not in {"triton", "cutedsl"}
        or verify
        != (["flashinfer"] if decode == ["flashinfer"] else ["triton"])
        or len(graph) != 1
        or graph[0] not in allowed_graphs
        or len(chunk) != 1
        or chunk[0] not in {"4096", "8192"}
        or len(steps) != 1
        or len(drafts) != 1
        or nextn not in {(1, 2), (2, 3), (3, 4)}
        or decode == ["flashinfer"]
        and mamba != ["bfloat16"]
    ):
        raise FlashNextServiceError("qualified SM120 winner settings are outside bounds")
    replay = container.count("--enable-linear-replayssm-spec")
    radix = _flag_values(container, "--mamba-radix-cache-strategy")
    if replay not in {0, 1} or (
        replay == 1
        and (
            radix != ["extra_buffer"]
            or mamba != ["float32"]
            or decode != ["triton"]
        )
    ) or (replay == 0 and radix):
        raise FlashNextServiceError("qualified ReplaySSM settings are inconsistent")
    if _flag_values(container, "--cpu-offload-gb") not in ([], ["0"], ["0.0"]):
        raise FlashNextServiceError("qualified command enables transformer CPU offload")
    if _flag_values(container, "--offload-group-size") not in ([], ["0"]):
        raise FlashNextServiceError("qualified command enables layer-group CPU offload")
    if _flag_values(container, "--offload-num-in-group") not in ([], ["0"]):
        raise FlashNextServiceError("qualified command enables grouped CPU offload")
    forbidden_offload_flags = {
        "--enable-hierarchical-cache",
        "--enable-hicache",
        "--no-ple-offload-embedding",
        "--offload-prefetch-step",
    }
    if any(
        item in forbidden_offload_flags
        or any(item.startswith(flag + "=") for flag in forbidden_offload_flags)
        for item in container
    ):
        raise FlashNextServiceError("qualified command enables unqualified offload")
    if container.count("--ple-offload-embedding") != 1:
        raise FlashNextServiceError("qualified command does not enable PLE host offload")
    fraction_values = _flag_values(container, "--mem-fraction-static")
    if len(fraction_values) != 1:
        raise FlashNextServiceError("qualified command omits its static-memory cap")
    try:
        fraction = float(str(fraction_values[0]))
    except ValueError as exc:
        raise FlashNextServiceError("qualified static-memory cap is malformed") from exc
    if (
        not math.isfinite(fraction)
        or str(fraction_values[0]) not in {"0.84", "0.86", "0.88"}
    ):
        raise FlashNextServiceError(
            "qualified static-memory cap is outside the reviewed selector"
        )
    if any(
        marker in item.casefold()
        for item in container
        for marker in ("--api-key", "hf_token", "password", "secret")
    ):
        raise FlashNextServiceError("qualified command contains a secret-bearing field")
    return tuple(container)


def _media_matches(report: Mapping[str, Any]) -> bool:
    media = report.get("media")
    if not isinstance(media, Mapping):
        return False
    for kind, (_name, digest, size) in QUALIFICATION_ASSETS.items():
        item = media.get(kind)
        if not isinstance(item, Mapping) or item != {
            "source": "local_data_uri",
            "bytes": size,
            "mime_type": "image/jpeg" if kind == "image" else "video/mp4",
            "sha256": digest,
        }:
            return False
    return True


def _validate_qualification_release(
    release_dir: Path,
    manifest: Mapping[str, Any],
    checkpoint: release_tool.CheckpointEvidence,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    release_tool.QualificationEvidence,
]:
    comparison = _safe_json(release_dir / "QUALIFICATION_REPORT.json")
    official_untuned = _safe_json(
        release_dir / "QUALIFICATION_OFFICIAL_UNTUNED.json"
    )
    mtp_off = _safe_json(release_dir / "QUALIFICATION_TUNED_MTP_OFF.json")
    mtp_on = _safe_json(
        release_dir / "QUALIFICATION_TUNED_MTP_ON_WINNER.json"
    )
    selection_paths = sorted(release_dir.glob("QUALIFICATION_SELECTION_*.json"))
    if not 1 <= len(selection_paths) <= release_tool.qualification_harness.MAX_SELECTION_CANDIDATES:
        raise FlashNextServiceError("release selector evidence is absent or unbounded")
    expected_selection_names = [
        f"QUALIFICATION_SELECTION_{index:03d}.json"
        for index in range(len(selection_paths))
    ]
    if [path.name for path in selection_paths] != expected_selection_names:
        raise FlashNextServiceError("release selector evidence names are not contiguous")
    reports = manifest.get("qualification", {}).get("reports")
    expected: dict[str, str] = {
        "comparison": _sha256(release_dir / "QUALIFICATION_REPORT.json"),
        "official_untuned": _sha256(
            release_dir / "QUALIFICATION_OFFICIAL_UNTUNED.json"
        ),
        "tuned_mtp_off": _sha256(
            release_dir / "QUALIFICATION_TUNED_MTP_OFF.json"
        ),
        "tuned_mtp_on_winner": _sha256(
            release_dir / "QUALIFICATION_TUNED_MTP_ON_WINNER.json"
        ),
        "sibling_manifest": _sha256(
            release_dir / release_tool.SIBLING_MANIFEST_FILENAME
        ),
        **{
            f"selection_candidate_{index:03d}": _sha256(path)
            for index, path in enumerate(selection_paths)
        },
    }
    if reports != expected:
        raise FlashNextServiceError("release qualification receipts changed")
    checkpoint_record = manifest.get("qualified_checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise FlashNextServiceError("release omits its qualified checkpoint receipt")
    checkpoint_tree = _require_digest(
        checkpoint_record.get("checkpoint_tree_sha256"),
        "qualified checkpoint tree",
    )
    try:
        qualification = release_tool.validate_qualification(
            comparison_path=release_dir / "QUALIFICATION_REPORT.json",
            official_untuned_path=(
                release_dir / "QUALIFICATION_OFFICIAL_UNTUNED.json"
            ),
            tuned_mtp_off_path=(
                release_dir / "QUALIFICATION_TUNED_MTP_OFF.json"
            ),
            selection_candidate_paths=selection_paths,
            tuned_mtp_on_winner_path=(
                release_dir / "QUALIFICATION_TUNED_MTP_ON_WINNER.json"
            ),
            checkpoint_tree_sha256=checkpoint_tree,
            sibling_manifest_path=(
                release_dir / release_tool.SIBLING_MANIFEST_FILENAME
            ),
            official_baseline_spec=checkpoint.behavior_baseline_spec,
        )
    except (release_tool.ReleaseError, OSError) as exc:
        raise FlashNextServiceError(
            f"release qualification is not production exact: {exc}"
        ) from exc
    if (
        checkpoint.checkpoint_tree_sha256 != checkpoint_tree
        or qualification.report_sha256 != expected
        or qualification.comparison != comparison
        or qualification.official_untuned != official_untuned
        or qualification.tuned_mtp_off != mtp_off
        or qualification.tuned_mtp_on_winner != mtp_on
    ):
        raise FlashNextServiceError("release qualification revalidation changed")
    return comparison, mtp_off, mtp_on, qualification


def load_production_binding(
    path: Path = BINDING_PATH,
    *,
    verify_release_hashes: bool = True,
    binding_payload: Mapping[str, Any] | None = None,
) -> ProductionBinding:
    if binding_payload is None:
        binding = _safe_json(path, maximum=64 * 1024)
        binding_sha256 = _sha256(path)
    else:
        binding = dict(binding_payload)
        binding_sha256 = hashlib.sha256(binding_bytes(binding)).hexdigest()
    expected_fields = {
        "schema_version",
        "complete",
        "profile_id",
        "service_id",
        "served_alias",
        "release_dir",
        "materialized_model_dir",
        "materialized_checkpoint_tree_sha256",
        "ple_materialization_manifest_sha256",
        "ple_materializer_sha256",
        "materialization_receipt_sha256",
        "repo_id",
        "publication_receipt",
        "host_port",
        "container_port",
    }
    if (
        set(binding) != expected_fields
        or binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("complete") is not True
        or binding.get("profile_id") != PROFILE_ID
        or binding.get("service_id") != SERVICE_ID
        or binding.get("served_alias") != SERVED_ALIAS
        or binding.get("release_dir") != str(RELEASE_DIR)
        or binding.get("materialized_model_dir") != str(MATERIALIZED_MODEL_DIR)
        or binding.get("host_port") != HOST_PORT
        or binding.get("container_port") != CONTAINER_PORT
    ):
        raise FlashNextServiceError("production service binding is incomplete or changed")
    repo_id = binding.get("repo_id")
    if not isinstance(repo_id, str):
        raise FlashNextServiceError("production binding has no repository identity")
    try:
        release_tool._validate_repo_id(repo_id)
        release = release_tool._validate_release_tree(
            RELEASE_DIR, repo_id=repo_id, verify_hashes=verify_release_hashes
        )
    except (release_tool.ReleaseError, OSError) as exc:
        raise FlashNextServiceError(f"qualified release tree is invalid: {exc}") from exc
    manifest = release["manifest"]
    if (
        manifest.get("runtime", {}).get("sglang_commit") != SGLANG_COMMIT
        or manifest.get("runtime", {}).get("sglang_source_stack_sha256")
        != SGLANG_SOURCE_STACK_SHA256
        or manifest.get("runtime", {}).get("oci_image") != SGLANG_IMAGE_REFERENCE
        or manifest.get("runtime", {}).get("oci_manifest_digest")
        != SGLANG_IMAGE_DIGEST
        or manifest.get("runtime", {}).get("oci_config_digest")
        != SGLANG_IMAGE_CONFIG_DIGEST
        or manifest.get("runtime", {}).get("oci_archive_sha256")
        != SGLANG_IMAGE_ARCHIVE_SHA256
        or manifest.get("runtime", {}).get("local_docker_image_id")
        != SGLANG_IMAGE_ID
        or manifest.get("runtime", {}).get("required_image_labels")
        != dict(runtime_contract.EXPECTED_IMAGE_LABELS)
        or manifest.get("runtime", {}).get("wire_served_alias") != SERVED_ALIAS
        or manifest.get("runtime", {}).get("display_name") != DISPLAY_NAME
        or manifest.get("runtime", {}).get("artifact_name") != ARTIFACT_NAME
        or manifest.get("preservation", {}).get("vision_image_video_bf16") is not True
        or manifest.get("preservation", {}).get("mtp_bf16") is not True
        or manifest.get("preservation", {}).get("ple_fp8_host_offload_contract") is not True
        or manifest.get("preservation", {}).get("ordinary_transformer_weight_cpu_offload")
        is not False
    ):
        raise FlashNextServiceError("release preservation/runtime contract changed")
    checkpoint_record = manifest.get("qualified_checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise FlashNextServiceError("release omits its qualified checkpoint receipt")
    builder_sha = _require_digest(
        checkpoint_record.get("builder_sha256"), "qualified builder"
    )
    materialized = _validate_materialized_checkpoint(
        release,
        expected_builder_sha256=builder_sha,
        verify_hashes=verify_release_hashes,
    )
    packaging = manifest.get("packaging")
    if not isinstance(packaging, Mapping):
        raise FlashNextServiceError("release packaging receipt is absent")
    for field, expected in {
        "materialized_checkpoint_tree_sha256": (
            materialized.checkpoint.checkpoint_tree_sha256
        ),
        "ple_materialization_manifest_sha256": materialized.ple_manifest_sha256,
        "ple_materializer_sha256": materialized.materializer_sha256,
        "materialization_receipt_sha256": materialized.receipt_sha256,
    }.items():
        if binding.get(field) != expected:
            raise FlashNextServiceError(
                f"production binding {field} does not match materialized evidence"
            )
    comparison, mtp_off, mtp_on, qualification = _validate_qualification_release(
        RELEASE_DIR, manifest, materialized.checkpoint
    )
    runtime_path = RELEASE_DIR / "RUNTIME_CONFIG.json"
    runtime = _safe_json(runtime_path, maximum=2 * 1024 * 1024)
    runtime_digest = _sha256(runtime_path)
    if (
        runtime.get("schema_version") != release_tool.RUNTIME_CONFIG_SCHEMA
        or runtime.get("repo_id") != repo_id
        or runtime.get("model_reference") != repo_id
        or runtime.get("checkpoint_tree_sha256")
        != comparison.get("checkpoint_tree_sha256")
        or runtime.get("model_path_contract")
        != {
            "checkpoint_tree_sha256": (
                materialized.checkpoint.checkpoint_tree_sha256
            ),
            "host_path_placeholder": MATERIALIZED_MODEL_PLACEHOLDER,
            "container_path": "/model",
            "mount_read_only": True,
            "source_role": "offline-materialized-canonical-checkpoint",
        }
        or runtime.get("served_alias") != SERVED_ALIAS
        or runtime.get("display_name") != DISPLAY_NAME
        or runtime.get("artifact_name") != ARTIFACT_NAME
        or runtime.get("model_architecture")
        != runtime_contract.MODEL_ARCHITECTURE
        or runtime.get("placement")
        != {
            "ple_offload_embedding": True,
            "transformer_weight_cpu_offload": False,
        }
        or runtime.get("launch_contract") != release_tool.LAUNCH_CONTRACT
        or runtime_digest != manifest.get("runtime", {}).get("config_sha256")
    ):
        raise FlashNextServiceError("release runtime config changed")
    try:
        runtime_evidence = release_tool.validate_runtime_config(
            runtime_path,
            repo_id=repo_id,
            checkpoint_tree_sha256=str(comparison["checkpoint_tree_sha256"]),
            qualification=qualification,
        )
    except (release_tool.ReleaseError, OSError) as exc:
        raise FlashNextServiceError(
            f"release runtime is not production exact: {exc}"
        ) from exc
    if runtime_evidence.config != runtime or runtime_evidence.config_sha256 != runtime_digest:
        raise FlashNextServiceError("release runtime revalidation changed")
    mtp_arm = runtime.get("arms", {}).get("tuned_mtp_on_winner")
    if not isinstance(mtp_arm, Mapping):
        raise FlashNextServiceError("release runtime config omits MTP-on")
    identity_config = mtp_on.get("runtime_identity", {}).get("runtime_config")
    if (
        mtp_arm.get("runtime_config") != identity_config
        or mtp_arm.get("config_sha256")
        != mtp_on.get("runtime_identity", {}).get("config_sha256")
        or mtp_arm.get("environment") != CONSTANT_RUNTIME_ENV
        or not isinstance(identity_config, Mapping)
        or identity_config.get("runtime_environment") != CONSTANT_RUNTIME_ENV
    ):
        raise FlashNextServiceError("production runtime differs from qualified MTP-on")
    container_command = _production_container_command(
        mtp_arm.get("command"),
        repo_id=repo_id,
        runtime_config=identity_config,
        expected_command_sha256=str(
            mtp_on.get("runtime_identity", {})
            .get("runtime_config_binding", {})
            .get("command_sha256", "")
        ),
    )
    memory_gb = float(mtp_on["resources"]["max_cgroup_memory_gb"])
    if not math.isfinite(memory_gb) or not 1 <= memory_gb <= MAX_TASK_MEMORY_GB:
        raise FlashNextServiceError("qualified task cgroup ceiling is invalid")
    task_memory_bytes = int(memory_gb * 1024**3)
    publication_path = Path(str(binding.get("publication_receipt"))).resolve(strict=True)
    try:
        publication_path.relative_to(PUBLICATION_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise FlashNextServiceError("publication receipt is outside its private root") from exc
    publication = _safe_json(publication_path, maximum=128 * 1024)
    _validate_publication_receipt(publication, repo_id=repo_id, release=release)
    _verify_qualification_assets()
    return ProductionBinding(
        binding_path=path,
        binding_sha256=binding_sha256,
        release_dir=RELEASE_DIR,
        repo_id=repo_id,
        publication_receipt=publication_path,
        release_tree_sha256=release["release_tree_sha256"],
        release_manifest_sha256=release["manifest_sha256"],
        checkpoint_tree_sha256=_require_digest(
            manifest.get("qualified_checkpoint", {}).get("checkpoint_tree_sha256"),
            "qualified checkpoint tree",
        ),
        materialized_model_dir=MATERIALIZED_MODEL_DIR,
        materialized_checkpoint_tree_sha256=(
            materialized.checkpoint.checkpoint_tree_sha256
        ),
        ple_materialization_manifest_sha256=materialized.ple_manifest_sha256,
        ple_materializer_sha256=materialized.materializer_sha256,
        materialization_receipt=MATERIALIZATION_RECEIPT,
        materialization_receipt_sha256=materialized.receipt_sha256,
        materialized_model_size_bytes=materialized.model_size_bytes,
        materialized_model_inode_count=materialized.model_inode_count,
        release_size_bytes=sum(size for _digest, size in release["files"].values())
        + (RELEASE_DIR / "SHA256SUMS").lstat().st_size,
        release_inode_count=len(release["files"]) + 1,
        runtime_config_sha256=runtime_digest,
        qualification_sha256=_sha256(RELEASE_DIR / "QUALIFICATION_REPORT.json"),
        qualification_mtp_off_sha256=_sha256(
            RELEASE_DIR / "QUALIFICATION_TUNED_MTP_OFF.json"
        ),
        qualification_mtp_on_sha256=_sha256(
            RELEASE_DIR / "QUALIFICATION_TUNED_MTP_ON_WINNER.json"
        ),
        task_memory_bytes=task_memory_bytes,
        container_command=container_command,
        runtime_environment=tuple(sorted(CONSTANT_RUNTIME_ENV.items())),
    )


def production_binding_payload(
    *, repo_id: str, publication_receipt: Path
) -> dict[str, Any]:
    """Return the only binding shape accepted by the runtime and finalizer."""

    release_tool._validate_repo_id(repo_id)
    try:
        release = release_tool._validate_release_tree(
            RELEASE_DIR, repo_id=repo_id, verify_hashes=True
        )
    except (release_tool.ReleaseError, OSError) as exc:
        raise FlashNextServiceError(
            f"cannot bind an invalid thin release: {exc}"
        ) from exc
    packaging = release["manifest"].get("packaging")
    if not isinstance(packaging, Mapping):
        raise FlashNextServiceError("thin release packaging receipt is absent")
    checkpoint_tree = _require_digest(
        packaging.get("canonical_checkpoint_tree_sha256"),
        "materialized checkpoint tree",
    )
    ple_manifest_sha = _require_digest(
        packaging.get("ple_materialization_manifest_sha256"),
        "PLE materialization manifest",
    )
    materializer_sha = _require_digest(
        packaging.get("ple_materializer_sha256"), "PLE materializer"
    )
    receipt_sha = _sha256(MATERIALIZATION_RECEIPT)
    return {
        "schema_version": BINDING_SCHEMA,
        "complete": True,
        "profile_id": PROFILE_ID,
        "service_id": SERVICE_ID,
        "served_alias": SERVED_ALIAS,
        "release_dir": str(RELEASE_DIR),
        "materialized_model_dir": str(MATERIALIZED_MODEL_DIR),
        "materialized_checkpoint_tree_sha256": checkpoint_tree,
        "ple_materialization_manifest_sha256": ple_manifest_sha,
        "ple_materializer_sha256": materializer_sha,
        "materialization_receipt_sha256": receipt_sha,
        "repo_id": repo_id,
        "publication_receipt": str(publication_receipt.absolute()),
        "host_port": HOST_PORT,
        "container_port": CONTAINER_PORT,
    }


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
    escaped = re.escape(identity)
    return result.returncode == 1 and re.search(
        rf"(?:No such object|No such container):\s*{escaped}(?:\s|$)",
        result.stderr,
    ) is not None


def _inspect(identity: str) -> dict[str, Any] | None:
    result = _docker(["container", "inspect", identity], timeout=30)
    if _docker_absent(result, identity):
        return None
    if result.returncode != 0:
        raise FlashNextServiceError("exact Docker container inspection failed")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FlashNextServiceError("Docker inspection response is malformed") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise FlashNextServiceError("Docker inspection did not return one container")
    return value[0]


def _image_preflight() -> None:
    if (
        not runtime_contract.image_digest_is_settled(SGLANG_IMAGE_DIGEST)
        or not runtime_contract.image_config_digest_is_settled(
            SGLANG_IMAGE_CONFIG_DIGEST
        )
        or not runtime_contract.local_docker_image_id_is_settled(
            SGLANG_IMAGE_ID
        )
    ):
        raise FlashNextServiceError(
            "patched SM120 SGLang image manifest/config identities are not settled"
        )
    result = _docker(["image", "inspect", SGLANG_IMAGE_REFERENCE], timeout=30)
    if result.returncode != 0:
        raise FlashNextServiceError(
            "qualified SGLang image is not preloaded; runtime never pulls images"
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FlashNextServiceError("Docker image inspection is malformed") from exc
    image = value[0] if isinstance(value, list) and len(value) == 1 else None
    if (
        not isinstance(image, Mapping)
        or image.get("Id") != SGLANG_IMAGE_ID
        or not isinstance(image.get("Descriptor"), Mapping)
        or image["Descriptor"].get("digest") != SGLANG_IMAGE_DIGEST
        or (
            image.get("RepoDigests")
            and runtime_contract.QUALIFIED_IMAGE_REPO_DIGEST
            not in image.get("RepoDigests", [])
        )
    ):
        raise FlashNextServiceError(
            "preloaded image does not expose the pinned config/manifest identity"
        )
    config = image.get("Config")
    mismatches = runtime_contract.validate_image_labels(
        config.get("Labels") if isinstance(config, Mapping) else None
    )
    if mismatches:
        raise FlashNextServiceError(
            "preloaded image lacks exact SM120 source/provenance labels: "
            + "; ".join(mismatches)
        )


def _container_name(runtime_id: str) -> str:
    return f"aeon-qwen38-flash-next-{runtime_id}"


def _labels(context: RuntimeContext, binding: ProductionBinding) -> dict[str, str]:
    return {
        "aeon.fleet.profile": PROFILE_ID,
        "aeon.fleet.runtime": context.runtime_id,
        "aeon.fleet.claim_sha256": hashlib.sha256(
            context.lease.claim_id.encode("utf-8")
        ).hexdigest(),
        "aeon.fleet.binding": binding.binding_sha256,
        "aeon.fleet.release": binding.release_tree_sha256,
        "aeon.fleet.command": binding.command_sha256,
        "aeon.model.artifact": ARTIFACT_NAME,
        "aeon.model.display_name": DISPLAY_NAME,
        "aeon.model.wire_alias": SERVED_ALIAS,
    }


def _docker_create_argv(
    context: RuntimeContext, binding: ProductionBinding
) -> tuple[str, ...]:
    if any(
        key in context.lease.required_environment
        and context.lease.required_environment[key] != value
        for key, value in binding.runtime_environment
    ):
        raise FlashNextServiceError("lease environment conflicts with qualified runtime")
    required_env = {
        **dict(binding.runtime_environment),
        **context.lease.required_environment,
    }
    arguments = [
        "container",
        "create",
        "--pull=never",
        "--name",
        _container_name(context.runtime_id),
        "--user",
        f"{os.geteuid()}:{os.getegid()}",
        "--gpus",
        f"device={context.lease.gpu_uuid}",
        "--memory",
        f"{binding.task_memory_bytes}b",
        "--memory-swap",
        f"{binding.task_memory_bytes}b",
        "--shm-size",
        f"{int(context.profile.min_shm_free_gb)}g",
        "--pids-limit",
        "4096",
        "--ulimit",
        "memlock=-1:-1",
        "--security-opt",
        "no-new-privileges=true",
        "--publish",
        f"127.0.0.1:{HOST_PORT}:{CONTAINER_PORT}",
        "--mount",
        f"type=bind,src={binding.materialized_model_dir},dst=/model,readonly",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,exec,size=8g",
    ]
    for key, value in sorted(_labels(context, binding).items()):
        arguments.extend(("--label", f"{key}={value}"))
    for key, value in sorted(required_env.items()):
        arguments.extend(("--env", f"{key}={value}"))
    arguments.extend((SGLANG_IMAGE_REFERENCE, *binding.container_command))
    return tuple(arguments)


def _process_start_ticks(pid: int) -> int:
    payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    end = payload.rfind(")")
    if end < 0:
        raise FlashNextServiceError("container process stat is malformed")
    return int(payload[end + 2 :].split()[19])


def _container_identity(
    item: Mapping[str, Any],
    *,
    context: RuntimeContext,
    binding: ProductionBinding,
    require_running: bool,
) -> tuple[str, int]:
    container_id = str(item.get("Id") or "")
    config = item.get("Config")
    host_config = item.get("HostConfig")
    state = item.get("State")
    mounts = item.get("Mounts")
    network = item.get("NetworkSettings")
    if (
        _CONTAINER_ID_RE.fullmatch(container_id) is None
        or item.get("Name") != "/" + _container_name(context.runtime_id)
        or not isinstance(config, Mapping)
        or not isinstance(host_config, Mapping)
        or not isinstance(state, Mapping)
        or not isinstance(mounts, list)
        or not isinstance(network, Mapping)
        or config.get("Image") != SGLANG_IMAGE_REFERENCE
        or config.get("User") != f"{os.geteuid()}:{os.getegid()}"
        or tuple(config.get("Cmd") or ()) != binding.container_command
        or host_config.get("Memory") != binding.task_memory_bytes
        or host_config.get("MemorySwap") != binding.task_memory_bytes
        or host_config.get("ShmSize") != int(context.profile.min_shm_free_gb * 1024**3)
        or host_config.get("PidsLimit") != 4096
        or host_config.get("Ulimits")
        != [{"Name": "memlock", "Hard": -1, "Soft": -1}]
        or host_config.get("SecurityOpt") != ["no-new-privileges=true"]
    ):
        raise FlashNextServiceError("exact container configuration changed")
    expected_labels = _labels(context, binding)
    observed_labels = config.get("Labels")
    if not isinstance(observed_labels, Mapping) or any(
        observed_labels.get(key) != value for key, value in expected_labels.items()
    ):
        raise FlashNextServiceError("exact container labels changed")
    required_env = {
        f"{key}={value}"
        for key, value in {
            **dict(binding.runtime_environment),
            **context.lease.required_environment,
        }.items()
    }
    if not required_env <= set(config.get("Env") or []):
        raise FlashNextServiceError(
            "lease/offline runtime environment was not injected unchanged"
        )
    model_mounts = [mount for mount in mounts if mount.get("Destination") == "/model"]
    if len(model_mounts) != 1 or (
        model_mounts[0].get("Source") != str(binding.materialized_model_dir)
        or model_mounts[0].get("RW") is not False
    ):
        raise FlashNextServiceError(
            "materialized checkpoint mount is not exact and read-only"
        )
    device_requests = host_config.get("DeviceRequests") or []
    if (
        len(device_requests) != 1
        or not isinstance(device_requests[0], Mapping)
        or device_requests[0].get("DeviceIDs") != [context.lease.gpu_uuid]
        or device_requests[0].get("Capabilities") != [["gpu"]]
    ):
        raise FlashNextServiceError("container GPU UUID binding changed")
    ports = network.get("Ports")
    bindings = ports.get(f"{CONTAINER_PORT}/tcp") if isinstance(ports, Mapping) else None
    if bindings != [{"HostIp": "127.0.0.1", "HostPort": str(HOST_PORT)}]:
        raise FlashNextServiceError("container endpoint is not loopback-exact")
    pid = state.get("Pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 0:
        raise FlashNextServiceError("container PID is malformed")
    running = state.get("Running") is True and state.get("Status") == "running"
    if require_running and (not running or pid <= 1):
        raise FlashNextServiceError("exact container is not running")
    return container_id, pid


def _cgroup_exact(pid: int, container_id: str, memory_bytes: int) -> bool:
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii").splitlines()
        unified = [line.split(":", 2)[2] for line in lines if line.startswith("0::")]
        if len(unified) != 1:
            return False
        relative = PurePosixPath(unified[0])
        if not relative.is_absolute() or ".." in relative.parts:
            return False
        if not any(container_id in part or container_id[:12] in part for part in relative.parts):
            return False
        root = Path("/sys/fs/cgroup")
        cgroup = root.joinpath(*relative.parts[1:])
        maximum = (cgroup / "memory.max").read_text(encoding="ascii").strip()
        return int(maximum) == memory_bytes
    except (OSError, ValueError, IndexError):
        return False


def _bounded_body(response: requests.Response, maximum: int) -> bytes:
    payload = bytearray()
    try:
        advertised = response.headers.get("content-length")
        if advertised is not None and int(advertised) > maximum:
            raise FlashNextServiceError("loopback response exceeded its bound")
        for chunk in response.iter_content(chunk_size=min(64 * 1024, maximum + 1)):
            payload.extend(chunk)
            if len(payload) > maximum:
                raise FlashNextServiceError("loopback response exceeded its bound")
    finally:
        response.close()
    return bytes(payload)


def _endpoint_ready(*, semantic: bool) -> bool:
    base = f"http://127.0.0.1:{HOST_PORT}"
    request_options = {
        "timeout": (2, 20),
        "allow_redirects": False,
        "proxies": {"http": "", "https": ""},
        "stream": True,
    }
    try:
        health = requests.get(f"{base}/health", **request_options)
        health_status = health.status_code
        _bounded_body(health, 64 * 1024)
        models = requests.get(f"{base}/v1/models", **request_options)
        models_status = models.status_code
        model_payload = json.loads(_bounded_body(models, 256 * 1024))
        identities = {
            item.get("id")
            for item in model_payload.get("data", [])
            if isinstance(item, Mapping)
        }
        if health_status != 200 or models_status != 200 or identities != {SERVED_ALIAS}:
            return False
        if not semantic:
            return True
        response = requests.post(
            f"{base}/v1/chat/completions",
            json={
                "model": SERVED_ALIAS,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly AEON_READY and nothing else.",
                    }
                ],
                "temperature": 0,
                "max_tokens": 16,
            },
            **request_options,
        )
        status = response.status_code
        value = json.loads(_bounded_body(response, 256 * 1024))
        choices = value.get("choices") if isinstance(value, Mapping) else None
        if status != 200 or value.get("model") != SERVED_ALIAS or not isinstance(choices, list):
            return False
        text = " ".join(
            str((choice.get("message") or {}).get(field) or "")
            for choice in choices
            if isinstance(choice, Mapping)
            for field in ("content", "reasoning_content")
        )
        return "AEON_READY" in text
    except (
        requests.RequestException,
        FlashNextServiceError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


class _PrepareHeartbeat:
    def __init__(self, context: RuntimeContext) -> None:
        self.context = context
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_PrepareHeartbeat":
        self.context.heartbeat(None, "Validating immutable Flash-Next release receipts")
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.error is not None:
            raise FlashNextServiceError("release-validation heartbeat failed") from self.error

    def _run(self) -> None:
        while not self.stop_event.wait(120):
            try:
                self.context.heartbeat(
                    None, "Immutable Flash-Next release validation remains active"
                )
            except BaseException as exc:
                self.error = exc
                return


class AeonQwenFlashNextServiceAdapter:
    """Serve one qualified NVFP4+MTP multimodal release on .177 GPU0."""

    def __init__(self) -> None:
        self._prepared: dict[str, ProductionBinding] = {}
        self._contexts: dict[str, RuntimeContext] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_context(context: RuntimeContext, binding: ProductionBinding) -> None:
        profile = context.profile
        lease = context.lease
        if (
            profile.profile_id != PROFILE_ID
            or profile.service_id != SERVICE_ID
            or profile.adapter != "aeon-qwen38-flash-next-service-v1"
            or profile.vram_budget_gb != VRAM_BUDGET_GB
            or profile.min_physical_vram_gb != MIN_PHYSICAL_VRAM_GB
            or profile.exclusive is not True
            # This is the authenticated logical-service ceiling.  The explicit
            # pool lane plus exact .177 GPU0 placement and exclusive lease still
            # admit at most one local Flash container.
            or profile.max_replicas != 2
            or profile.serving_pool_id != SERVING_POOL_ID
            or profile.lane_max_replicas != FLASH_LANE_MAX_REPLICAS
            or profile.artifact_identity != binding.artifact_identity
        ):
            raise FlashNextServiceError("Fleet profile is not the finalized release profile")
        if any(value == ZERO_SHA256 for value in profile.artifact_identity.values()):
            raise FlashNextServiceError("release profile still contains placeholders")
        model = str(lease.model or "").casefold()
        if (
            lease.host != HOST
            or lease.physical_gpu != PHYSICAL_GPU
            or lease.memory_total_mib is None
            or lease.memory_total_mib < int(MIN_PHYSICAL_VRAM_GB * 1024)
            or "rtx pro 6000" not in model
            or "blackwell" not in model
            or abs(lease.vram_budget_gb - VRAM_BUDGET_GB) > 1e-9
            or lease.exclusive is not True
            or context.job_id is not None
            or context.scratch_path is not None
            or str(context.run_dir) != lease.run_dir
        ):
            raise FlashNextServiceError("lease is not exact .177 RTX PRO 6000 GPU0")
        if lease.memory_total_mib / 1024 - VRAM_BUDGET_GB < 6:
            raise FlashNextServiceError("lease does not preserve six GiB of physical VRAM")

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        if _RUNTIME_ID_RE.fullmatch(context.runtime_id) is None or context.payload:
            raise FlashNextServiceError("service runtime identity/payload is not reviewed")
        with _PrepareHeartbeat(context):
            binding = load_production_binding(verify_release_hashes=True)
            self._validate_context(context, binding)
            _image_preflight()
        existing = _inspect(_container_name(context.runtime_id))
        if existing is not None:
            raise FlashNextServiceError("exact service container name already exists")
        metadata = context.run_dir.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise FlashNextServiceError("Fleet run directory is unsafe")
        filesystem = os.statvfs(context.run_dir)
        with self._lock:
            self._prepared[context.runtime_id] = binding
            self._contexts[context.runtime_id] = context
        return StoragePreparationResult(
            scratch_path=context.scratch_path,
            filesystem_id=str(metadata.st_dev),
            free_bytes_after_stage=filesystem.f_bavail * filesystem.f_frsize,
            free_inodes_after_stage=filesystem.f_favail,
            staged_bytes=0,
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        with self._lock:
            binding = self._prepared.get(context.runtime_id)
        if binding is None:
            raise AdapterLaunchError(
                "production binding preflight is absent", process_absent=True
            )
        receipt_path = context.run_dir / RECEIPT_NAME
        if receipt_path.exists() or receipt_path.is_symlink():
            raise FlashNextServiceError("service container intent already exists")
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "runtime_id": context.runtime_id,
            "profile_id": PROFILE_ID,
            "binding_sha256": binding.binding_sha256,
            "release_tree_sha256": binding.release_tree_sha256,
            "materialized_checkpoint_tree_sha256": (
                binding.materialized_checkpoint_tree_sha256
            ),
            "materialization_receipt_sha256": (
                binding.materialization_receipt_sha256
            ),
            "command_sha256": binding.command_sha256,
            "container_name": _container_name(context.runtime_id),
            "container_id": None,
            "state": "creating",
            "pid": None,
            "start_ticks": None,
        }
        _atomic_json(receipt_path, receipt)
        created_id: str | None = None
        try:
            result = _docker(_docker_create_argv(context, binding), timeout=180)
            created_id = result.stdout.strip()
            if result.returncode != 0 or _CONTAINER_ID_RE.fullmatch(created_id) is None:
                existing = _inspect(_container_name(context.runtime_id))
                if existing is None:
                    raise AdapterLaunchError(
                        "Docker create failed with exact container absence",
                        process_absent=True,
                    )
                raise FlashNextServiceError("Docker create result is ambiguous")
            receipt.update(container_id=created_id, state="created")
            _atomic_json(receipt_path, receipt)
            item = _inspect(created_id)
            if item is None:
                raise AdapterLaunchError(
                    "created container disappeared", process_absent=True
                )
            _container_identity(
                item, context=context, binding=binding, require_running=False
            )
            started = _docker(["container", "start", created_id], timeout=120)
            if started.returncode != 0 or started.stdout.strip() != created_id:
                raise FlashNextServiceError("exact container did not start")
            item = _inspect(created_id)
            if item is None:
                raise AdapterLaunchError(
                    "started container disappeared", process_absent=True
                )
            _container_id, pid = _container_identity(
                item, context=context, binding=binding, require_running=True
            )
            start_ticks = _process_start_ticks(pid)
            receipt.update(
                state="running", pid=pid, start_ticks=start_ticks
            )
            _atomic_json(receipt_path, receipt)
            deadline = time.monotonic() + context.profile.startup_timeout_seconds
            last_heartbeat = 0.0
            while time.monotonic() < deadline:
                context.startup_check()
                now = time.monotonic()
                if now - last_heartbeat >= 30:
                    context.heartbeat(pid, "Qualified Flash-Next SGLang is loading")
                    last_heartbeat = now
                item = _inspect(created_id)
                if item is None:
                    raise AdapterLaunchError(
                        "service container exited and disappeared", process_absent=True
                    )
                _container_identity(
                    item, context=context, binding=binding, require_running=True
                )
                if (
                    _cgroup_exact(pid, created_id, binding.task_memory_bytes)
                    and _endpoint_ready(semantic=True)
                ):
                    identity = (
                        f"{PROCESS_PREFIX}:{context.runtime_id}:{created_id}:"
                        f"{binding.binding_sha256}:{pid}:{start_ticks}"
                    )
                    return LaunchResult(
                        pid=pid,
                        process_identity=identity,
                        endpoint=f"http://127.0.0.1:{HOST_PORT}/v1",
                    )
                time.sleep(5)
            raise FlashNextServiceError("qualified service startup exceeded its bound")
        except AdapterLaunchError:
            raise
        except BaseException as exc:
            if created_id is None:
                raise AdapterLaunchError(
                    f"service launch failed before container creation: {exc}",
                    process_absent=True,
                ) from exc
            item = _inspect(created_id)
            if item is None:
                raise AdapterLaunchError(
                    f"service launch failed with exact process absence: {exc}",
                    process_absent=True,
                ) from exc
            raise

    @staticmethod
    def _runtime_parts(runtime: Mapping[str, Any]) -> tuple[str, str, str, int, int]:
        match = _PROCESS_RE.fullmatch(str(runtime.get("process_identity") or ""))
        if (
            match is None
            or match.group(1) != runtime.get("runtime_id")
            or runtime.get("profile_id") != PROFILE_ID
            or runtime.get("host") != HOST
            or int(match.group(4)) != runtime.get("pid")
        ):
            raise FlashNextServiceError("saved service runtime identity changed")
        return (
            match.group(1),
            match.group(2),
            match.group(3),
            int(match.group(4)),
            int(match.group(5)),
        )

    def _saved_context_binding(
        self, runtime: Mapping[str, Any]
    ) -> tuple[RuntimeContext, ProductionBinding]:
        runtime_id, _container, binding_sha, _pid, _ticks = self._runtime_parts(runtime)
        with self._lock:
            context = self._contexts.get(runtime_id)
        if context is None:
            context = self._context_from_runtime(runtime)
        binding = load_production_binding(verify_release_hashes=False)
        self._validate_context(context, binding)
        if binding.binding_sha256 != binding_sha:
            raise FlashNextServiceError("saved production binding changed")
        return context, binding

    @staticmethod
    def _context_from_runtime(runtime: Mapping[str, Any]) -> RuntimeContext:
        """Rebuild exact adapter identity solely from Fleet's durable record."""

        try:
            raw_profile = json.loads(str(runtime["profile_json"]))
            raw_payload = json.loads(str(runtime["payload_json"]))
            if not isinstance(raw_profile, dict) or not isinstance(raw_payload, dict):
                raise ValueError("runtime profile/payload snapshot is malformed")
            profile = ComputeProfile.from_dict(raw_profile)
            lease = Lease(
                claim_id=str(runtime["claim_id"]),
                owner=str(runtime["owner"]),
                host=str(runtime["host"]),
                physical_gpu=int(runtime["physical_gpu"]),
                gpu_uuid=str(runtime["gpu_uuid"]),
                vram_budget_gb=float(runtime["vram_budget_gb"]),
                exclusive=bool(runtime["exclusive"]),
                run_dir=str(runtime["run_dir"]),
                model=(
                    str(runtime["gpu_model"])
                    if runtime.get("gpu_model") is not None
                    else None
                ),
                memory_total_mib=(
                    int(runtime["memory_total_mib"])
                    if runtime.get("memory_total_mib") is not None
                    else None
                ),
            )
            canonical = Path(str(runtime["canonical_output_path"]))
            scratch = runtime.get("scratch_path")
            if scratch is not None:
                scratch = str(scratch)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FlashNextServiceError(
                f"durable runtime context is incomplete: {exc}"
            ) from exc
        return RuntimeContext(
            runtime_id=str(runtime["runtime_id"]),
            profile=profile,
            lease=lease,
            run_dir=Path(str(runtime["run_dir"])),
            payload=raw_payload,
            job_id=(str(runtime["job_id"]) if runtime.get("job_id") else None),
            scratch_path=scratch,
            canonical_output_path=canonical,
            heartbeat=lambda _pid, _note: None,
            startup_check=lambda: None,
        )

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        try:
            runtime_id, container_id, _binding_sha, pid, ticks = self._runtime_parts(runtime)
            context, binding = self._saved_context_binding(runtime)
            receipt = _safe_json(Path(str(runtime["run_dir"])) / RECEIPT_NAME, maximum=64 * 1024)
            if (
                receipt.get("runtime_id") != runtime_id
                or receipt.get("container_id") != container_id
                or receipt.get("binding_sha256") != binding.binding_sha256
                or receipt.get("materialized_checkpoint_tree_sha256")
                != binding.materialized_checkpoint_tree_sha256
                or receipt.get("materialization_receipt_sha256")
                != binding.materialization_receipt_sha256
                or receipt.get("pid") != pid
                or receipt.get("start_ticks") != ticks
            ):
                raise FlashNextServiceError("container receipt changed")
            item = _inspect(container_id)
            if item is None:
                return ProbeResult(
                    ProbeState.ABSENT, False, True, "exact Flash-Next container is absent"
                )
            _identity, observed_pid = _container_identity(
                item, context=context, binding=binding, require_running=True
            )
            if (
                observed_pid != pid
                or _process_start_ticks(pid) != ticks
                or not _cgroup_exact(pid, container_id, binding.task_memory_bytes)
                or not _endpoint_ready(semantic=True)
            ):
                raise FlashNextServiceError("live process/cgroup/semantic identity changed")
            return ProbeResult(
                ProbeState.READY,
                True,
                False,
                f"{DISPLAY_NAME} multimodal primary is ready "
                f"(compatibility wire alias {SERVED_ALIAS})",
            )
        except FlashNextServiceError as exc:
            return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))
        except (OSError, ValueError) as exc:
            return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        try:
            _runtime_id, container_id, _binding_sha, _pid, _ticks = self._runtime_parts(runtime)
            context, binding = self._saved_context_binding(runtime)
            item = _inspect(container_id)
            if item is None:
                return StopResult(True, True, reason)
            _container_identity(
                item, context=context, binding=binding, require_running=False
            )
            if item.get("State", {}).get("Running") is True:
                stopped = _docker(
                    ["container", "stop", "--time", "30", container_id], timeout=60
                )
                if stopped.returncode != 0 or stopped.stdout.strip() != container_id:
                    return StopResult(False, True, "exact container is still stopping")
            item = _inspect(container_id)
            if item is not None:
                _container_identity(
                    item, context=context, binding=binding, require_running=False
                )
                if item.get("State", {}).get("Running") is True:
                    return StopResult(False, True, "exact container remains live")
                removed = _docker(["container", "rm", container_id], timeout=30)
                if removed.returncode != 0 or removed.stdout.strip() != container_id:
                    return StopResult(False, True, "stopped container remains present")
            absent = _inspect(container_id) is None
            return StopResult(absent, True, reason if absent else "container absence unproven")
        except (FlashNextServiceError, OSError, ValueError) as exc:
            return StopResult(False, False, str(exc))

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        scratch = storage.get("scratch_path")
        if (
            runtime.get("host") != HOST
            or scratch is not None
            or storage.get("canonical_output_path")
            != runtime.get("canonical_output_path")
        ):
            raise FlashNextServiceError("service storage identity changed")
        # .177 is canonical.  The release and its qualification/publication
        # receipts are never cleanup targets; the Fleet run receipt is retained.
        if runtime.get("process_identity") is not None:
            _runtime_id, container_id, _binding, _pid, _ticks = self._runtime_parts(runtime)
            if _inspect(container_id) is not None:
                return StorageFinalizationResult(
                    True, False, 0, "exact service container remains present"
                )
        return StorageFinalizationResult(
            True,
            True,
            0,
            "canonical .177 release retained; no model or receipt cleanup performed",
        )


def create_fleet_adapter() -> AeonQwenFlashNextServiceAdapter:
    return AeonQwenFlashNextServiceAdapter()
