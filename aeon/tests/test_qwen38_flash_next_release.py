"""CPU-hermetic release, thin-package, and uploader regressions."""

from __future__ import annotations

from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace
import urllib.parse

import pytest

from aeon.scripts import materialize_qwen38_flash_next_ple as materializer
from aeon.scripts import release_qwen38_flash_next as release


REPO_ID = "aday777/Aeon-Qwen3.8-Flash-Next-NVFP4-MTP"


def _sha(character: str) -> str:
    return character * 64


def _write(path: Path, payload: bytes) -> tuple[str, int]:
    path.write_bytes(payload)
    path.chmod(0o600)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _write_json(path: Path, value: object) -> tuple[str, int]:
    return _write(
        path,
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )


def _runtime_environment() -> dict[str, str]:
    return {
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "USE_TF": "0",
        "USE_FLAX": "0",
    }


def _runtime_config(*, mtp: bool) -> dict[str, object]:
    return {
        "served_alias": "served-model",
        "tp_size": 1,
        "ple_offload_embedding": True,
        "cpu_offload_gb": 0,
        "offload_group_size": -1,
        "moe_a2a_backend": "none",
        "moe_runner_backend": "flashinfer_cutlass",
        "fp4_gemm_backend": "flashinfer_cutlass",
        "speculative_moe_a2a_backend": "none",
        "speculative_moe_runner_backend": "flashinfer_cutlass",
        "max_running_requests": 4,
        "cuda_graph_config": {
            "decode": {"backend": "full", "max_bs": 4, "bs": [1, 2, 4]},
            "prefill": {"backend": "disabled"},
        },
        "linear_attn_backend": "triton",
        "linear_attn_decode_backend": "triton",
        "linear_attn_prefill_backend": "cutedsl",
        "linear_attn_verify_backend": "triton",
        "enable_linear_replayssm_spec": False,
        "mamba_radix_cache_strategy": None,
        "ragged_verify_mode": "static",
        "runtime_environment": _runtime_environment(),
        "mamba_ssm_dtype": "bfloat16",
        "chunked_prefill_size": 4096,
        "mem_fraction_static": 0.86,
        "requested_speculative_algorithm": "NEXTN" if mtp else None,
        "speculative_algorithm": "EAGLE" if mtp else None,
        "speculative_num_steps": 2 if mtp else None,
        "speculative_eagle_topk": 1 if mtp else None,
        "speculative_num_draft_tokens": 3 if mtp else None,
    }


def _command(*, mtp: bool) -> list[str]:
    environment = [
        item
        for key, value in sorted(_runtime_environment().items())
        for item in ("--env", f"{key}={value}")
    ]
    command = [
        "/usr/bin/docker",
        "run",
        *environment,
        "--mount",
        "type=bind,src=@AEON_MATERIALIZED_MODEL_PATH@,dst=/model,readonly",
        release.SGLANG_IMAGE_REFERENCE,
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        "/model",
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--quantization",
        release.runtime_contract.QUANTIZATION,
        "--reasoning-parser",
        release.runtime_contract.REASONING_PARSER,
        "--prefill-attention-backend",
        release.runtime_contract.PREFILL_ATTENTION_BACKEND,
        "--decode-attention-backend",
        release.runtime_contract.DECODE_ATTENTION_BACKEND,
        "--context-length",
        str(release.runtime_contract.SM120_VALIDATED_CONTEXT_LENGTH),
        "--max-total-tokens",
        str(release.runtime_contract.SM120_VALIDATED_CONTEXT_LENGTH),
        "--page-size",
        "64",
        "--speculative-draft-model-quantization",
        release.runtime_contract.MTP_DRAFT_QUANTIZATION,
        "--served-model-name",
        "served-model",
        "--ple-offload-embedding",
        "--cpu-offload-gb",
        "0",
    ]
    if mtp:
        command.extend(
            [
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                "2",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "3",
            ]
        )
    return command


def _inner_command_sha256(command: list[str]) -> str:
    image_index = command.index(release.SGLANG_IMAGE_REFERENCE)
    return release._sha256_bytes(release._canonical_json(command[image_index + 1 :]))


def test_runtime_command_binds_offline_environment_and_materialized_mount() -> None:
    on = release._validate_command(
        _command(mtp=True),
        arm="tuned_mtp_on_winner",
        served_alias="served-model",
        runtime_config=_runtime_config(mtp=True),
        expected_inner_command_sha256=_inner_command_sha256(_command(mtp=True)),
    )
    off = release._validate_command(
        _command(mtp=False),
        arm="tuned_mtp_off",
        served_alias="served-model",
        runtime_config=_runtime_config(mtp=False),
        expected_inner_command_sha256=_inner_command_sha256(_command(mtp=False)),
    )

    assert "--model-path /model" in on
    assert on.startswith("python3 -m sglang.launch_server")
    assert "@AEON_MATERIALIZED_MODEL_PATH@" not in on
    assert "--speculative-algorithm NEXTN" in on
    assert "--speculative-algorithm" not in off

    changed = _command(mtp=True)
    changed[changed.index("type=bind,src=@AEON_MATERIALIZED_MODEL_PATH@,dst=/model,readonly")] = (
        "type=bind,src=/private/model,dst=/model,readonly"
    )
    with pytest.raises(release.ReleaseError, match="materialized mount"):
        release._validate_command(
            changed,
            arm="tuned_mtp_on_winner",
            served_alias="served-model",
            runtime_config=_runtime_config(mtp=True),
            expected_inner_command_sha256=_inner_command_sha256(
                _command(mtp=True)
            ),
        )

    changed = _command(mtp=True)
    changed[changed.index("--mount"):changed.index("--mount")] = [
        "--env",
        "UNREVIEWED_AMBIENT_STATE=1",
    ]
    with pytest.raises(release.ReleaseError, match="environment contract"):
        release._validate_command(
            changed,
            arm="tuned_mtp_on_winner",
            served_alias="served-model",
            runtime_config=_runtime_config(mtp=True),
            expected_inner_command_sha256=_inner_command_sha256(
                _command(mtp=True)
            ),
        )

    for injected in (
        ["-eHF_HUB_OFFLINE=0"],
        ["--env-file", "/private/unreviewed.env"],
    ):
        changed = _command(mtp=True)
        changed[changed.index("--mount"):changed.index("--mount")] = injected
        with pytest.raises(release.ReleaseError, match="environment spelling"):
            release._validate_command(
                changed,
                arm="tuned_mtp_on_winner",
                served_alias="served-model",
                runtime_config=_runtime_config(mtp=True),
                expected_inner_command_sha256=_inner_command_sha256(
                    _command(mtp=True)
                ),
            )

    changed = _command(mtp=True)
    changed.extend(("--disable-radix-cache",))
    with pytest.raises(release.ReleaseError, match="measured command receipt"):
        release._validate_command(
            changed,
            arm="tuned_mtp_on_winner",
            served_alias="served-model",
            runtime_config=_runtime_config(mtp=True),
            expected_inner_command_sha256=_inner_command_sha256(
                _command(mtp=True)
            ),
        )


def _tiny_safetensors(name: str) -> bytes:
    header = {
        "__metadata__": {},
        name: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
    }
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return struct.pack("<Q", len(raw)) + raw + b"\x00"


def _synthetic_passthrough_receipt(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, tuple[str, int]], dict[str, object], dict]:
    checkpoint = tmp_path / "checkpoint"
    audit_root = tmp_path / "audit"
    checkpoint.mkdir(mode=0o700)
    audit_root.mkdir(mode=0o700)
    index_receipt = _write_json(
        checkpoint / "model.safetensors.index.json",
        {"metadata": {"total_size": 333}, "weight_map": {}},
    )
    source_index_sha256 = _sha("1")
    hybrid = {
        "schema_version": "aeon-qwen38-flash-next-hybrid-v1",
        "complete": True,
        "artifact": "qwen38-flash-next-tensor-hybrid",
        "sources": {
            "bf16": {"repo": release.BF16_REPO, "revision": release.BF16_REVISION},
            "fp8_ple": {"repo": release.FP8_REPO, "revision": release.FP8_REVISION},
        },
        "upstream_metadata": {
            "bf16_config_sha256": _sha("2"),
            "bf16_index_sha256": _sha("3"),
        },
        "topology": {
            "tensor_count": 1_659,
            "bf16_source_expert_tensor_count": 96,
            "bf16_mtp_tensor_count": 31,
            "bf16_vision_tensor_count": 333,
            "fp8_ple_table_tensor_count": 128,
            "bf16_ple_scale_tensor_count": 1,
            "non_expert_non_mtp_tensor_count": 1_532,
        },
        "files": {
            "model.safetensors.index.json": {
                "sha256": source_index_sha256,
                "size": 123,
            },
            "source-a.safetensors": {"sha256": _sha("4"), "size": 300},
            "source-b.safetensors": {"sha256": _sha("5"), "size": 400},
        },
    }
    hybrid_receipt = _write_json(checkpoint / "HYBRID_MANIFEST.json", hybrid)
    canonical_files = {
        "model.safetensors.index.json": index_receipt,
        "HYBRID_MANIFEST.json": hybrid_receipt,
        "final-a.safetensors": (_sha("6"), 111),
        "final-b.safetensors": (_sha("7"), 222),
    }
    build_manifest: dict[str, object] = {
        "sources": {
            "hybrid": {
                "manifest": "HYBRID_MANIFEST.json",
                "manifest_sha256": hybrid_receipt[0],
            }
        }
    }
    audit = {
        "auditor": {
            "file": "audit_qwen38_flash_next_passthrough.py",
            "sha256": release.PASSTHROUGH_AUDITOR_SHA256,
        },
        "checkpoint": {
            "file_bytes": 333,
            "index_sha256": index_receipt[0],
            "safetensors_file_count": 2,
            "tensor_bytes": 135_156_121_594,
            "tensor_count": 296_475,
            "topology_sha256": _sha("8"),
        },
        "complete": True,
        "contract": json.loads(json.dumps(release.PASSTHROUGH_CONTRACT)),
        "created_at": "2026-08-26T20:00:00+00:00",
        "passed": True,
        "passthrough": {
            "canonical_name_set_sha256": release.PASSTHROUGH_NAME_SET_SHA256,
            "category_counts": {"other": 1_069, "ple": 129, "vision": 333},
            "category_payload_sha256": {
                "other": _sha("9"),
                "ple": _sha("a"),
                "vision": _sha("b"),
            },
            "dtype_bytes": {
                "BF16": 9_521_860_834,
                "F8_E4M3": 51_200_245_760,
                "I64": 280,
            },
            "dtype_counts": {"BF16": 1_400, "F8_E4M3": 128, "I64": 3},
            "exact_raw_payload_identity": True,
            "payload_inventory_sha256": _sha("c"),
            "shard_mapping_preserved": True,
            "tensor_bytes": release.PASSTHROUGH_TENSOR_BYTES,
            "tensor_count": release.PASSTHROUGH_TENSOR_COUNT,
        },
        "schema_version": release.PASSTHROUGH_AUDIT_SCHEMA,
        "source_hybrid": {
            "file_bytes": 700,
            "index_sha256": source_index_sha256,
            "safetensors_file_count": 2,
            "tensor_bytes": 308_799_717_370,
            "tensor_count": 1_659,
            "topology_sha256": _sha("d"),
        },
    }
    receipt_path = audit_root / "passthrough.json"
    _write_json(receipt_path, audit)
    return receipt_path, checkpoint, canonical_files, build_manifest, audit


def test_release_accepts_exact_standalone_passthrough_auditor_contract(
    tmp_path: Path,
) -> None:
    receipt, checkpoint, canonical_files, build_manifest, _audit = (
        _synthetic_passthrough_receipt(tmp_path)
    )

    evidence = release._validate_passthrough_audit(
        receipt,
        checkpoint_root=checkpoint,
        canonical_files=canonical_files,
        build_manifest=build_manifest,
        current_index_receipt=canonical_files["model.safetensors.index.json"],
        require_external=True,
    )

    assert evidence.receipt_sha256 == release._sha256(receipt)
    assert evidence.source_hybrid_index_sha256 == _sha("1")
    assert evidence.payload_inventory_sha256 == _sha("c")
    assert release._sha256(Path(release.passthrough_auditor.__file__)) == (
        release.PASSTHROUGH_AUDITOR_SHA256
    )
    assert release.passthrough_auditor._validate_contract(
        release.passthrough_auditor.PRODUCTION_CONTRACT
    ) == release.PASSTHROUGH_CONTRACT


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("raw_identity", "raw identity"),
        ("shard_mapping", "raw identity"),
        ("tensor_count", "raw identity"),
        ("uppercase_inventory", "lowercase SHA-256"),
        ("source_index", "inventory binding"),
        ("checkpoint_index", "inventory binding"),
        ("contract", "identity or completion"),
        ("extra_top_level", "fields changed"),
    ],
)
def test_release_rejects_passthrough_receipt_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    receipt, checkpoint, canonical_files, build_manifest, audit = (
        _synthetic_passthrough_receipt(tmp_path)
    )
    if mutation == "raw_identity":
        audit["passthrough"]["exact_raw_payload_identity"] = False
    elif mutation == "shard_mapping":
        audit["passthrough"]["shard_mapping_preserved"] = False
    elif mutation == "tensor_count":
        audit["passthrough"]["tensor_count"] = 1_530
    elif mutation == "uppercase_inventory":
        audit["passthrough"]["payload_inventory_sha256"] = _sha("A")
    elif mutation == "source_index":
        audit["source_hybrid"]["index_sha256"] = _sha("e")
    elif mutation == "checkpoint_index":
        audit["checkpoint"]["index_sha256"] = _sha("f")
    elif mutation == "contract":
        audit["contract"]["details"]["name"] = "changed"
    else:
        audit["unexpected"] = True
    changed = receipt.with_name(f"{mutation}.json")
    _write_json(changed, audit)

    with pytest.raises(release.ReleaseError, match=message):
        release._validate_passthrough_audit(
            changed,
            checkpoint_root=checkpoint,
            canonical_files=canonical_files,
            build_manifest=build_manifest,
            current_index_receipt=canonical_files[
                "model.safetensors.index.json"
            ],
            require_external=True,
        )


def _synthetic_checkpoint_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[release.CheckpointEvidence, Path]:
    checkpoint_root = tmp_path / "checkpoint"
    source_root = tmp_path / "official-fp8"
    checkpoint_root.mkdir(mode=0o700)
    source_root.mkdir(mode=0o700)
    files: dict[str, tuple[str, int]] = {}
    official_files: dict[str, dict[str, object]] = {}
    official_map: dict[str, str] = {}
    final_map: dict[str, str] = {}
    for index in range(33):
        tensor = release.PLE_PREFIX + f"shard_{index}.weight"
        source_name = f"model-{index + 1:05d}-of-00033.safetensors"
        target_name = "fp8-ple-" + source_name
        payload = _tiny_safetensors(tensor)
        source_receipt = _write(source_root / source_name, payload)
        target_receipt = _write(checkpoint_root / target_name, payload)
        official_files[source_name] = {
            "sha256": source_receipt[0],
            "size": source_receipt[1],
            "git_blob_sha1": "1" * 40,
        }
        files[target_name] = target_receipt
        official_map[tensor] = source_name
        final_map[tensor] = target_name
    official_index = {"metadata": {"total_size": 33}, "weight_map": official_map}
    official_index_receipt = _write_json(
        source_root / "model.safetensors.index.json", official_index
    )
    official_files["model.safetensors.index.json"] = {
        "sha256": official_index_receipt[0],
        "size": official_index_receipt[1],
        "git_blob_sha1": "2" * 40,
    }
    final_index = {"metadata": {"total_size": 33}, "weight_map": final_map}
    files["model.safetensors.index.json"] = _write_json(
        checkpoint_root / "model.safetensors.index.json", final_index
    )
    files["README.md"] = _write(checkpoint_root / "README.md", b"canonical\n")
    files["config.json"] = _write(checkpoint_root / "config.json", b"{}\n")
    sums_payload = "".join(
        f"{digest}  {name}\n" for name, (digest, _size) in sorted(files.items())
    ).encode()
    checkpoint_tree = hashlib.sha256(sums_payload).hexdigest()
    source_manifest = {
        "schema_version": "aeon-pinned-hf-files-v1",
        "repo": release.FP8_REPO,
        "revision": release.FP8_REVISION,
        "total_bytes": sum(item["size"] for item in official_files.values()),
        "files": official_files,
    }
    source_manifest_path = tmp_path / "qwen-fp8-files.json"
    source_manifest_sha, _ = _write_json(source_manifest_path, source_manifest)
    monkeypatch.setattr(release, "FP8_FILES_MANIFEST", source_manifest_path)
    monkeypatch.setattr(release, "FP8_FILES_MANIFEST_SHA256", source_manifest_sha)
    monkeypatch.setattr(release, "FP8_INDEX_SHA256", official_index_receipt[0])
    monkeypatch.setattr(
        materializer, "OFFICIAL_FILES_MANIFEST_SHA256", source_manifest_sha
    )
    monkeypatch.setattr(
        materializer, "OFFICIAL_INDEX_SHA256", official_index_receipt[0]
    )
    monkeypatch.setattr(materializer, "OFFICIAL_INDEX_SIZE", official_index_receipt[1])
    checkpoint = release.CheckpointEvidence(
        root=checkpoint_root,
        checkpoint_tree_sha256=checkpoint_tree,
        files=files,
        build_manifest={},
        build_manifest_sha256=_sha("a"),
        builder_sha256=_sha("b"),
        validation={},
        config={},
        tensor_summary={},
        behavior_baseline_spec={},
        behavior_baseline_spec_sha256=_sha("c"),
    )
    return checkpoint, source_root


def test_thin_package_materializes_exact_canonical_checkpoint_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, source_root = _synthetic_checkpoint_and_source(tmp_path, monkeypatch)
    manifest, ple_targets = release._ple_materialization_manifest(checkpoint)
    thin = tmp_path / "thin"
    thin.mkdir(mode=0o700)
    for canonical_name, thin_name in manifest["thin_file_map"].items():
        os.link(checkpoint.root / canonical_name, thin / thin_name)
    materialization_receipt = _write_json(
        thin / release.PLE_MATERIALIZATION_FILENAME,
        manifest,
    )
    os.link(
        Path(materializer.__file__),
        thin / release.PLE_MATERIALIZER_FILENAME,
    )
    release_manifest = release._release_manifest(
        repo_id=REPO_ID,
        checkpoint=checkpoint,
        qualification=SimpleNamespace(report_sha256={}, summary={}),
        runtime=SimpleNamespace(
            config_sha256=_sha("d"),
            config={"model_path_contract": {}},
            commands={},
        ),
        ple_materialization=manifest,
        ple_materialization_sha256=materialization_receipt[0],
        passthrough_audit=release.PassthroughAuditEvidence(
            receipt={
                "passthrough": {
                    "exact_raw_payload_identity": True,
                    "shard_mapping_preserved": True,
                }
            },
            receipt_sha256=_sha("4"),
            source_hybrid_manifest_sha256=_sha("5"),
            source_hybrid_index_sha256=_sha("6"),
            checkpoint_index_sha256=_sha("7"),
            payload_inventory_sha256=_sha("8"),
        ),
        generated_receipts={},
        source_receipts={},
    )
    thin_files = {
        path.name: (release._sha256(path), path.stat().st_size)
        for path in thin.iterdir()
    }
    assert release._validate_release_ple_materialization(
        thin,
        thin_files,
        release_manifest,
    ) == manifest

    completion_receipt = tmp_path / "materialized.materialization-receipt.json"
    result = materializer.materialize(
        thin,
        source_root,
        tmp_path / "materialized",
        receipt=completion_receipt,
    )

    assert len(ple_targets) == 33
    assert result["checkpoint_tree_sha256"] == checkpoint.checkpoint_tree_sha256
    assert result["ple_shard_count"] == 33
    completion = json.loads(completion_receipt.read_text())
    assert completion["schema_version"] == materializer.COMPLETION_SCHEMA_VERSION
    assert completion["materialized_checkpoint_tree_sha256"] == (
        checkpoint.checkpoint_tree_sha256
    )
    assert completion_receipt.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "materialized" / "README.md").read_bytes() == b"canonical\n"
    for name, (digest, size) in checkpoint.files.items():
        path = tmp_path / "materialized" / name
        assert path.stat().st_size == size
        assert release._sha256(path) == digest


class _DownloadResponse:
    def __init__(self, request, payload: bytes):
        range_header = request.get_header("Range")
        start = int(range_header.removeprefix("bytes=").removesuffix("-")) if range_header else 0
        self._payload = payload[start:]
        self._offset = 0
        self.status = 206 if start else 200
        self.headers = {"Content-Length": str(len(self._payload))}
        if start:
            self.headers["Content-Range"] = f"bytes {start}-{len(payload) - 1}/{len(payload)}"
        self._url = request.full_url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self) -> str:
        return self._url

    def read(self, size: int) -> bytes:
        block = self._payload[self._offset : self._offset + size]
        self._offset += len(block)
        return block


def test_thin_package_auto_downloads_exact_pinned_sources_resumably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, source_root = _synthetic_checkpoint_and_source(tmp_path, monkeypatch)
    manifest, _ple_targets = release._ple_materialization_manifest(checkpoint)
    thin = tmp_path / "thin"
    thin.mkdir(mode=0o700)
    for canonical_name, thin_name in manifest["thin_file_map"].items():
        os.link(checkpoint.root / canonical_name, thin / thin_name)
    _write_json(thin / release.PLE_MATERIALIZATION_FILENAME, manifest)

    payloads = {path.name: path.read_bytes() for path in source_root.iterdir()}

    def fake_urlopen(request, timeout):
        assert timeout == 120
        name = urllib.parse.unquote(urllib.parse.urlparse(request.full_url).path.rsplit("/", 1)[-1])
        return _DownloadResponse(request, payloads[name])

    monkeypatch.setattr(materializer.urllib.request, "urlopen", fake_urlopen)
    cache = tmp_path / "official-cache"
    cache.mkdir(mode=0o700)
    resumed_name = sorted(name for name in payloads if name.endswith(".safetensors"))[0]
    partial = cache / f".{resumed_name}.partial"
    partial.write_bytes(payloads[resumed_name][:7])
    partial.chmod(0o600)

    downloaded = materializer.download_official_sources(thin, cache)
    assert downloaded["revision"] == release.FP8_REVISION
    assert downloaded["files"] == 34
    assert downloaded["downloaded_bytes"] == sum(map(len, payloads.values())) - 7
    for name, payload in payloads.items():
        assert (cache / name).read_bytes() == payload

    result = materializer.materialize(
        thin,
        cache,
        tmp_path / "auto-materialized",
        receipt=tmp_path / "auto-materialized.receipt.json",
    )
    assert result["checkpoint_tree_sha256"] == checkpoint.checkpoint_tree_sha256


def test_model_card_exposes_auto_resolution_and_low_refusal_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, _source_root = _synthetic_checkpoint_and_source(tmp_path, monkeypatch)
    manifest, _ple_targets = release._ple_materialization_manifest(checkpoint)
    qualification = SimpleNamespace(
        summary={
            "task_scoped_ram_peak_bytes": 128 * 1024**3,
            "physical_cuda_peak_used_bytes": 80 * 1024**3,
            "physical_cuda_min_reserve_bytes": 16 * 1024**3,
            "mtp_speedup_ci_95": [1.04, 1.20],
            "mtp_off_completion_tps": 100.0,
            "mtp_on_completion_tps": 112.0,
            "mtp_speedup": 1.12,
        }
    )
    runtime = SimpleNamespace(
        commands={
            "tuned_mtp_off": "python3 -m sglang.launch_server --mtp off",
            "tuned_mtp_on_winner": (
                "python3 -m sglang.launch_server --mtp on"
            ),
        }
    )

    readme = release._render_readme(
        repo_id=REPO_ID,
        checkpoint=checkpoint,
        qualification=qualification,
        runtime=runtime,
        ple_materialization=manifest,
    ).decode()

    assert "--download-official-to" in readme
    assert f"hf download {release.FP8_REPO}" not in readme
    assert "low-refusal" in readme
    assert "clearly harmful" in readme
    assert "uncensored" not in readme.lower()
    assert release.DISPLAY_NAME in readme
    assert "Exact qualified SGLang argv and Fleet launch" in readme
    assert "BrokerServiceSession" in readme
    assert 'profile="aeon-qwen38-standard"' in readme
    assert "unlimited memlock" in readme
    assert "/usr/bin/docker run" not in readme
    assert "@AEON_MATERIALIZED_MODEL_PATH@" not in readme
    assert "GPU-" not in readme and "gc-" not in readme
    assert "HF_TOKEN=" not in readme


def _labeling_evidence(tmp_path: Path):
    root = tmp_path / "labeling-release"
    root.mkdir(mode=0o700)
    _write(
        root / "README.md",
        (
            b"This is a private derivative with a low-refusal behavioral tune. "
            b"It retains safeguards for clearly harmful requests. Passing this "
            b"bounded suite is not a general safety guarantee.\n"
        ),
    )
    _write(root / "NOTICE", b"Bounded derivative notice.\n")
    _write(
        root / release.CANONICAL_README_FILENAME,
        b"Pinned official checkpoint documentation.\n",
    )
    baseline_sha = _sha("a")
    files = {
        release.BEHAVIOR_BASELINE_FILENAME: (baseline_sha, 100),
    }
    manifest = {
        "behavioral_tuning": {
            "scope": ["lm_head"],
            "precision": "bfloat16",
            "merged_before_nvfp4": True,
            "held_out_gate": "passed",
            "official_untuned_baseline_spec_sha256": baseline_sha,
            "strictly_fewer_unnecessary_refusals": True,
            "all_clearly_harmful_cases_refused": True,
            "cross_entropy_used_as_improvement_evidence": False,
            "intent": release.BEHAVIORAL_TUNING_INTENT,
        }
    }
    qualification = SimpleNamespace(
        summary={
            "baseline_non_harmful_unnecessary_refusals": 4,
            "final_non_harmful_unnecessary_refusals": 1,
            "strictly_fewer_unnecessary_refusals": True,
            "all_clearly_harmful_cases_refused": True,
        }
    )
    return root, files, manifest, qualification


def test_release_labeling_is_low_refusal_and_rejects_uncensored_metadata(
    tmp_path: Path,
) -> None:
    root, files, manifest, qualification = _labeling_evidence(tmp_path)
    release._validate_release_labeling(
        root,
        repo_id=REPO_ID,
        manifest=manifest,
        files=files,
        qualification=qualification,
    )

    readme = root / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "This model is uncensored.\n",
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseError, match="uncensored"):
        release._validate_release_labeling(
            root,
            repo_id=REPO_ID,
            manifest=manifest,
            files=files,
            qualification=qualification,
        )


def test_release_labeling_requires_exact_behavioral_tuning_disclosure(
    tmp_path: Path,
) -> None:
    root, files, manifest, qualification = _labeling_evidence(tmp_path)
    manifest["behavioral_tuning"] = {
        **manifest["behavioral_tuning"],
        "all_clearly_harmful_cases_refused": False,
    }
    with pytest.raises(release.ReleaseError, match="behavioral-tuning disclosure"):
        release._validate_release_labeling(
            root,
            repo_id=REPO_ID,
            manifest=manifest,
            files=files,
            qualification=qualification,
        )


class _RemoteEntry:
    def __init__(self, path: str, size: int):
        self.path = path
        self.size = size


class _FakeApi:
    def __init__(self, files: dict[str, int], manifest: Path):
        self.files = files
        self.manifest = manifest
        self.created: list[dict[str, object]] = []
        self.uploaded: list[dict[str, object]] = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def repo_info(self, **_kwargs):
        return SimpleNamespace(private=True, sha="d" * 40)

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)

    def list_repo_tree(self, **_kwargs):
        return [_RemoteEntry(name, size) for name, size in self.files.items()]


class _FakeHub:
    __version__ = release.HF_HUB_VERSION

    class HfApi:
        upload_folder = object()

    def __init__(self, root: Path):
        self.root = root
        self.downloaded: list[str] = []

    def hf_hub_download(self, **kwargs):
        filename = kwargs["filename"]
        self.downloaded.append(filename)
        return str(self.root / filename)


def test_pinned_hub_attributes_are_content_addressed() -> None:
    assert len(release.GITATTRIBUTES_PAYLOAD) == 1_635
    assert hashlib.sha256(release.GITATTRIBUTES_PAYLOAD).hexdigest() == (
        release.GITATTRIBUTES_SHA256
    )
    assert b"*.safetensors filter=lfs" in release.GITATTRIBUTES_PAYLOAD


def test_upload_uses_current_upload_folder_and_verifies_private_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir(mode=0o700)
    manifest = release_dir / "RELEASE_MANIFEST.json"
    receipt = _write(manifest, b"release manifest\n")
    attributes_receipt = _write(
        release_dir / release.GITATTRIBUTES_FILENAME,
        release.GITATTRIBUTES_PAYLOAD,
    )
    sums_receipt = _write(release_dir / "SHA256SUMS", b"release tree receipt\n")
    files = {
        release.GITATTRIBUTES_FILENAME: attributes_receipt,
        "RELEASE_MANIFEST.json": receipt,
    }
    upload_files = {**files, "SHA256SUMS": sums_receipt}
    local = {
        "root": release_dir,
        "manifest": {"publication": {"repo_id": REPO_ID, "visibility": "private"}},
        "manifest_sha256": receipt[0],
        "files": files,
        "release_tree_sha256": sums_receipt[0],
    }
    api = _FakeApi(
        {name: size for name, (_digest, size) in upload_files.items()}, manifest
    )
    hub = _FakeHub(release_dir)
    monkeypatch.setattr(release, "_validate_release_tree", lambda *args, **kwargs: local)
    monkeypatch.setattr(
        release,
        "_load_huggingface_hub",
        lambda **_kwargs: (hub, release.HF_XET_VERSION),
    )
    monkeypatch.setattr(
        release,
        "_authenticated_hub",
        lambda _hub, repo_id: (api, "hf_private_test_token", "aday777"),
    )
    publication = tmp_path / "publication.json"
    args = Namespace(
        repo_id=REPO_ID,
        release_dir=release_dir,
        receipt=publication,
        execute=False,
    )
    dry = release.upload_release(args)
    assert dry["status"] == "dry-run-authenticated-and-validated"
    assert dry["files"] == 3
    assert dry["upload_bytes"] == sum(size for _digest, size in upload_files.values())
    assert not api.created and not api.uploaded

    args.execute = True
    complete = release.upload_release(args)
    assert complete["status"] == "complete"
    assert api.created[0]["private"] is True
    assert api.uploaded[0]["folder_path"] == release_dir
    assert set(hub.downloaded) == {
        release.GITATTRIBUTES_FILENAME,
        "RELEASE_MANIFEST.json",
        "SHA256SUMS",
    }
    assert json.loads(publication.read_text())["huggingface_hub_version"] == (
        release.HF_HUB_VERSION
    )
    publication_value = json.loads(publication.read_text())
    assert set(publication_value) == release.PUBLICATION_RECEIPT_FIELDS
    assert publication_value["verification"] == release.PUBLICATION_VERIFICATION


def test_upload_fails_before_any_hub_action_when_private_quota_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir(mode=0o700)
    sums_receipt = _write(release_dir / "SHA256SUMS", b"release tree receipt\n")
    local = {
        "root": release_dir,
        "manifest": {},
        "manifest_sha256": _sha("1"),
        "files": {"huge.safetensors": (_sha("2"), release.FREE_PRIVATE_STORAGE_BYTES + 1)},
        "release_tree_sha256": sums_receipt[0],
    }
    monkeypatch.setattr(release, "_validate_release_tree", lambda *args, **kwargs: local)
    monkeypatch.setattr(
        release,
        "_load_huggingface_hub",
        lambda **_kwargs: pytest.fail("Hub must not be touched before the quota gate"),
    )
    with pytest.raises(release.ReleaseError, match="free 100 GB"):
        release.upload_release(
            Namespace(
                repo_id=REPO_ID,
                release_dir=release_dir,
                receipt=tmp_path / "receipt.json",
                execute=True,
            )
        )


def test_upload_execution_rehashes_exact_pinned_wheel_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_wheel = tmp_path / "huggingface_hub.whl"
    xet_wheel = tmp_path / "hf_xet.whl"
    requests_wheel = tmp_path / "requests.whl"
    charset_wheel = tmp_path / "charset_normalizer.whl"
    urllib3_wheel = tmp_path / "urllib3.whl"
    hub_digest, _hub_size = _write(hub_wheel, b"pinned huggingface hub wheel")
    xet_digest, _xet_size = _write(xet_wheel, b"pinned hf xet wheel")
    requests_digest, _ = _write(requests_wheel, b"pinned requests wheel")
    charset_digest, _ = _write(charset_wheel, b"pinned charset wheel")
    urllib3_digest, _ = _write(urllib3_wheel, b"pinned urllib3 wheel")
    monkeypatch.setattr(release, "HF_HUB_WHEEL", hub_wheel)
    monkeypatch.setattr(release, "HF_XET_WHEEL", xet_wheel)
    monkeypatch.setattr(release, "REQUESTS_WHEEL", requests_wheel)
    monkeypatch.setattr(release, "CHARSET_NORMALIZER_WHEEL", charset_wheel)
    monkeypatch.setattr(release, "URLLIB3_WHEEL", urllib3_wheel)
    monkeypatch.setattr(release, "HF_HUB_WHEEL_SHA256", hub_digest)
    monkeypatch.setattr(release, "HF_XET_WHEEL_SHA256", xet_digest)
    monkeypatch.setattr(release, "REQUESTS_WHEEL_SHA256", requests_digest)
    monkeypatch.setattr(release, "CHARSET_NORMALIZER_WHEEL_SHA256", charset_digest)
    monkeypatch.setattr(release, "URLLIB3_WHEEL_SHA256", urllib3_digest)

    assert release._rehash_pinned_upload_wheels() == {
        "charset_normalizer": charset_digest,
        "huggingface_hub": hub_digest,
        "hf_xet": xet_digest,
        "requests": requests_digest,
        "urllib3": urllib3_digest,
    }
    xet_wheel.write_bytes(b"changed")
    with pytest.raises(release.ReleaseError, match="wheel digest changed"):
        release._rehash_pinned_upload_wheels()


def test_repo_owner_must_match_authenticated_user_and_has_no_public_mode() -> None:
    assert release._validate_repo_id(REPO_ID, authenticated_user="aday777") == "aday777"
    with pytest.raises(release.ReleaseError, match="authenticated username"):
        release._validate_repo_id(REPO_ID, authenticated_user="someone-else")
    with pytest.raises(release.ReleaseError, match="not an uncensored model"):
        release._validate_repo_id(
            "aday777/Aeon-Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP",
            authenticated_user="aday777",
        )
    upload_actions = release._parser().parse_args(
        ["upload", "--release-dir", "/tmp/model", "--repo-id", REPO_ID]
    )
    assert not hasattr(upload_actions, "public")
    finalize_argv = [
        "finalize",
        "--checkpoint",
        "/tmp/checkpoint",
        "--release-dir",
        "/tmp/release",
        "--repo-id",
        REPO_ID,
        "--builder-sha256",
        _sha("1"),
        "--passthrough-audit",
        "/tmp/passthrough.json",
        "--qualification-report",
        "/tmp/qualification.json",
        "--official-untuned-report",
        "/tmp/official.json",
        "--tuned-mtp-off-report",
        "/tmp/off.json",
        "--selection-candidate-report",
        "/tmp/selection.json",
        "--tuned-mtp-on-winner-report",
        "/tmp/on.json",
        "--sibling-manifest",
        "/tmp/sibling.json",
        "--runtime-config",
        "/tmp/runtime.json",
    ]
    finalize_actions = release._parser().parse_args(finalize_argv)
    assert finalize_actions.passthrough_audit == Path("/tmp/passthrough.json")
    audit_index = finalize_argv.index("--passthrough-audit")
    without_audit = finalize_argv[:audit_index] + finalize_argv[audit_index + 2 :]
    with pytest.raises(SystemExit):
        release._parser().parse_args(without_audit)


def test_upload_revalidates_release_before_any_hub_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalid_release(*_args, **_kwargs):
        raise release.ReleaseError("qualification evidence failed")

    monkeypatch.setattr(release, "_validate_release_tree", invalid_release)
    monkeypatch.setattr(
        release,
        "_load_huggingface_hub",
        lambda **_kwargs: pytest.fail("Hub access preceded release revalidation"),
    )
    with pytest.raises(release.ReleaseError, match="qualification evidence"):
        release.upload_release(
            Namespace(
                repo_id=REPO_ID,
                release_dir=tmp_path / "invalid-release",
                receipt=None,
                execute=False,
            )
        )
