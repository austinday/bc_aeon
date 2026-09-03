#!/usr/bin/env python3
"""Create the immutable production binding from one passing vLLM canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

from aeon.core import qwen_flash_next_vllm_canary_adapter as canary_adapter
from aeon.core import qwen_flash_next_vllm_contract as contract
from aeon.core import qwen_flash_next_vllm_service_binding as binding_module


PROFILE_PATH = Path(
    "/home/aday/NexusAgentDashboard/fleet_compute/profiles.d/"
    "aeon-qwen38-flash-next-vllm-canary.json"
)
ACKNOWLEDGEMENT = "create-exact-vllm-production-binding"
_RUNTIME = re.compile(r"^fr-[0-9a-f]{32}$")


class PromotionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_json(path: Path, *, maximum: int) -> Mapping[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= maximum
    ):
        raise PromotionError(f"promotion input is unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"promotion input is malformed: {path}") from exc
    if not isinstance(value, Mapping):
        raise PromotionError(f"promotion input is not an object: {path}")
    return value


def build_binding(runtime_id: str) -> dict[str, Any]:
    if _RUNTIME.fullmatch(runtime_id) is None:
        raise PromotionError("runtime ID is malformed")
    root = binding_module.CANARY_OUTPUT_ROOT / runtime_id
    try:
        resolved = root.resolve(strict=True)
        resolved.relative_to(binding_module.CANARY_OUTPUT_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise PromotionError("canary runtime is outside its canonical root") from exc
    request = _private_json(resolved / "canary-request.json", maximum=2 * 1024 * 1024)
    # Fleet settlement keeps the immutable task result beneath the adapter's
    # canonical ``output`` directory; evidence beside it remains diagnostic.
    qualification_path = resolved / "output" / "qualification.json"
    qualification = _private_json(qualification_path, maximum=64 * 1024 * 1024)
    failures = contract.validate_qualification_receipt(qualification)
    if failures:
        raise PromotionError("canary is not promotable: " + "; ".join(failures))

    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    manifest = profile.pop("manifest_sha256", None)
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    if (
        profile.get("enabled") is not True
        or profile.get("profile_id") != contract.PROFILE_ID
        or hashlib.sha256(canonical).hexdigest() != manifest
    ):
        raise PromotionError("reviewed canary profile changed")
    payload = {
        key: str(request[key])
        for key in (
            "checkpoint_path",
            "checkpoint_manifest_path",
            "checkpoint_manifest_sha256",
            "derived_image_digest",
            "derived_image_config_digest",
            "derived_image_archive_path",
            "derived_image_archive_sha256",
        )
    }
    identity = canary_adapter.expected_artifact_identity(payload)
    if profile.get("artifact_identity") != identity:
        raise PromotionError("canary source/artifact identity changed")
    if (
        request.get("runtime_id") != runtime_id
        or request.get("host") != contract.HOST
        or request.get("physical_gpu") != contract.PHYSICAL_GPU
        or request.get("exclusive") is not True
        or request.get("vram_cap_gib") != contract.VRAM_CAP_GIB
        or request.get("runtime") != contract.expected_runtime()
    ):
        raise PromotionError("canary request contract changed")
    return {
        "schema_version": binding_module.BINDING_SCHEMA,
        "complete": True,
        "profile_id": binding_module.PROFILE_ID,
        "service_id": binding_module.SERVICE_ID,
        "host": contract.HOST,
        "physical_gpu": contract.PHYSICAL_GPU,
        "vram_cap_gib": contract.VRAM_CAP_GIB,
        "runtime": contract.expected_runtime(),
        "qualification_receipt": str(qualification_path),
        "qualification_receipt_sha256": _sha256(qualification_path),
        **payload,
        "canary_artifact_identity": identity,
    }


def _write_binding(value: Mapping[str, Any]) -> None:
    path = binding_module.BINDING_PATH
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or parent.st_mode & 0o077
    ):
        raise PromotionError("binding directory is unsafe")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    binding_module.load_binding(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge")
    arguments = parser.parse_args(argv)
    try:
        binding = build_binding(arguments.runtime_id)
        if arguments.execute:
            if arguments.acknowledge != ACKNOWLEDGEMENT:
                raise PromotionError("exact acknowledgement is required")
            _write_binding(binding)
            loaded = binding_module.load_binding()
            print(json.dumps({"binding": str(loaded.path), "sha256": loaded.sha256}))
        else:
            print(json.dumps({"valid": True, "execute": False}))
    except (OSError, KeyError, TypeError, ValueError, PromotionError) as exc:
        print(f"promotion failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
