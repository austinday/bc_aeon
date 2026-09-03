#!/usr/bin/env python3
"""Build a metadata-only NVFP4 W4A16 canary from Aeon's speed checkpoint.

The packed NVFP4 weights and their scales remain byte-identical.  Only the
compressed-tensors activation contract changes from W4A4 to W4A16 so vLLM can
select its weight-only NVFP4 kernel.  Safetensors are hard-linked into an
atomic owner-local artifact; this script never mutates a linked file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any


EXPECTED_BUILD_SHA256 = (
    "b9cd6f0791fe08817ec1a5e7e739ddc80230fcceb94dbcfc06fc444b94c2e624"
)
EXPECTED_SHA256SUMS_SHA256 = (
    "dd5a88636198a00e02ae10df0d95d7d07987b91299d2ebf56474e6d2ef5c421b"
)
NVFP4_GROUP = "group_0"


class BuildError(RuntimeError):
    """Raised when an immutable source or output invariant fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_owner_file(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise BuildError(f"source file is not immutable and owner-controlled: {path}")
    return metadata


def _read_json(path: Path) -> dict[str, Any]:
    _regular_owner_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"invalid JSON source: {path}") from exc
    if not isinstance(value, dict):
        raise BuildError(f"JSON source is not an object: {path}")
    return value


def _parse_sha256s(path: Path) -> dict[str, str]:
    _regular_owner_file(path)
    expected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2:
            raise BuildError("source SHA256SUMS has malformed rows")
        digest, name = fields
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or Path(name).name != name
            or name in expected
        ):
            raise BuildError("source SHA256SUMS has unsafe identities")
        expected[name] = digest
    if not expected:
        raise BuildError("source SHA256SUMS is empty")
    return expected


def _validate_source(source: Path) -> tuple[dict[str, str], dict[str, Any]]:
    if source.is_symlink() or not source.is_dir():
        raise BuildError("source must be one real directory")
    build_path = source / "BUILD_MANIFEST.json"
    sums_path = source / "SHA256SUMS"
    if _sha256(build_path) != EXPECTED_BUILD_SHA256:
        raise BuildError("source build manifest identity changed")
    if _sha256(sums_path) != EXPECTED_SHA256SUMS_SHA256:
        raise BuildError("source SHA256SUMS identity changed")
    build = _read_json(build_path)
    if (
        build.get("schema_version") != "aeon-qwen38-speed-variant-v3"
        or build.get("status") != "canary_unvalidated"
        or build.get("quantization", {}).get("body") != "unchanged NVFP4 W4A4 group-16"
    ):
        raise BuildError("source build contract changed")

    expected = _parse_sha256s(sums_path)
    actual_names = {
        path.name
        for path in source.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(expected) != actual_names:
        raise BuildError("source file set does not match SHA256SUMS")
    for name, digest in expected.items():
        path = source / name
        _regular_owner_file(path)
        if _sha256(path) != digest:
            raise BuildError(f"source digest changed: {name}")
    return expected, build


def _w4a16_config(source_config: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(source_config))
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise BuildError("source config has no quantization contract")
    groups = quantization.get("config_groups")
    if not isinstance(groups, dict) or NVFP4_GROUP not in groups:
        raise BuildError("source config has no canonical NVFP4 group")
    group = groups[NVFP4_GROUP]
    if (
        not isinstance(group, dict)
        or group.get("format") != "nvfp4-pack-quantized"
        or group.get("targets") != ["Linear"]
        or not isinstance(group.get("input_activations"), dict)
        or group.get("weights", {}).get("num_bits") != 4
        or group.get("weights", {}).get("group_size") != 16
        or group.get("weights", {}).get("type") != "float"
    ):
        raise BuildError("source NVFP4 group is not the reviewed W4A4 layout")
    group["input_activations"] = None
    return config


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build(source: Path, output: Path) -> None:
    source = source.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise BuildError(f"output already exists: {output}")
    if output.parent.resolve(strict=True) == source:
        raise BuildError("output cannot be nested directly in the source")

    source_hashes, source_build = _validate_source(source)
    source_config = _read_json(source / "config.json")
    config = _w4a16_config(source_config)
    index = _read_json(source / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise BuildError("source tensor index is malformed")
    shards = sorted(set(weight_map.values()))
    if any(
        not isinstance(name, str) or not name.endswith(".safetensors")
        for name in shards
    ):
        raise BuildError("source tensor index has unsafe shard names")

    partial = output.with_name(
        f".{output.name}.partial-{os.getpid()}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if partial.exists() or partial.is_symlink():
        raise BuildError(f"partial output already exists: {partial}")
    partial.mkdir(mode=0o700)
    try:
        for name in sorted(source_hashes):
            if name in {"BUILD_MANIFEST.json", "README.md", "config.json"}:
                continue
            source_path = source / name
            target_path = partial / name
            if name.endswith(".safetensors"):
                os.link(source_path, target_path, follow_symlinks=False)
            else:
                shutil.copy2(source_path, target_path, follow_symlinks=False)

        _write_json(partial / "config.json", config)
        partial.joinpath("README.md").write_text(
            """# Aeon Qwen3.8 NVFP4 W4A16 speed canary

This canary keeps every packed model tensor and weight scale byte-identical to
the hash-bound Aeon speed-v3 checkpoint. Only the compressed-tensors activation
contract changes from NVFP4 W4A4 to NVFP4 W4A16, selecting a weight-only kernel
with BF16 activations. It is not production-authorized until the speed, semantic,
multimodal, and artifact-integrity gates pass.
""",
            encoding="utf-8",
        )
        partial.joinpath("README.md").chmod(0o600)
        manifest = {
            "schema_version": "aeon-qwen38-nvfp4-w4a16-variant-v1",
            "status": "canary_unvalidated",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "path": str(source),
                "build_manifest_sha256": EXPECTED_BUILD_SHA256,
                "sha256s_sha256": EXPECTED_SHA256SUMS_SHA256,
                "schema_version": source_build["schema_version"],
            },
            "transformation": {
                "weights": "byte-identical NVFP4 group-16 hard links",
                "input_activations": "NVFP4 W4A4 dynamic-local to BF16 W4A16",
                "changed_files": [
                    "config.json",
                    "README.md",
                    "BUILD_MANIFEST.json",
                    "SHA256SUMS",
                ],
                "required_runtime_behavior": "ignore unexpected source input_global_scale tensors for W4A16 layers",
            },
            "validation": {
                "source_files_verified": len(source_hashes),
                "tensor_count": len(weight_map),
                "safetensors_bytes": sum(
                    (source / name).stat().st_size for name in shards
                ),
            },
        }
        _write_json(partial / "BUILD_MANIFEST.json", manifest)

        hash_lines = []
        for path in sorted(
            item
            for item in partial.iterdir()
            if item.is_file() and item.name != "SHA256SUMS"
        ):
            _regular_owner_file(path)
            hash_lines.append(f"{_sha256(path)}  {path.name}")
        partial.joinpath("SHA256SUMS").write_text(
            "\n".join(hash_lines) + "\n", encoding="utf-8"
        )
        partial.joinpath("SHA256SUMS").chmod(0o600)
        for path in partial.iterdir():
            if path.is_file() and not path.name.endswith(".safetensors"):
                _fsync_file(path)
        descriptor = os.open(partial, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        partial.rename(output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        if partial.is_dir() and not partial.is_symlink():
            for path in partial.iterdir():
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            partial.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
    print(f"Published unvalidated NVFP4 W4A16 canary: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
