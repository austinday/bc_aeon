from __future__ import annotations

import errno
import json
import os
from pathlib import Path, PurePosixPath
import stat
import struct
from types import SimpleNamespace
import urllib.error
import zipfile

import pytest

from aeon.core import qwen_flash_next_build_adapter as adapter
from aeon.scripts import assemble_qwen38_flash_next_hybrid as assembler
from aeon.scripts import qwen_flash_next_build_worker as worker
from fleet_compute.models import ProbeResult, ProbeState


def _private(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True)
    path.chmod(0o700)
    return path


def _safetensors(path: Path) -> None:
    header = {
        "__metadata__": {"format": "pt"},
        "drop": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
        "keep": {"dtype": "U8", "shape": [3], "data_offsets": [2, 5]},
    }
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"abXYZ")
    path.chmod(0o600)


def test_filtered_safetensors_closes_inventory(tmp_path: Path) -> None:
    root = _private(tmp_path / "private")
    source = root / "source.safetensors"
    output = root / "output.safetensors"
    _safetensors(source)

    assembler._filtered_safetensors(source, output, ["keep"])

    _metadata, tensors, start = assembler._read_header(output)
    assert set(tensors) == {"keep"}
    assert output.read_bytes()[start:] == b"XYZ"


def test_interrupted_download_resumes_exact_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _private(tmp_path / "downloads")
    destination = root / "shard.safetensors"
    payload = b"reviewed-payload"
    receipt = (len(payload), worker.hashlib.sha256(payload).hexdigest())
    calls: list[str | None] = []

    class Response:
        def __init__(self, body: bytes, *, start: int, fail: bool) -> None:
            self.body = body
            self.start = start
            self.fail = fail
            self.used = False
            self.status = 206 if start else 200
            self.headers = (
                {"Content-Range": f"bytes {start}-{len(payload) - 1}/{len(payload)}"}
                if start
                else {}
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self) -> str:
            return "https://huggingface.co/pinned"

        def read(self, _size: int) -> bytes:
            if self.used:
                if self.fail:
                    raise urllib.error.URLError("interrupted")
                return b""
            self.used = True
            return self.body

    def fake_open(request, timeout):
        assert timeout == 120
        raw_range = request.headers.get("Range")
        calls.append(raw_range)
        if len(calls) == 1:
            return Response(payload[:5], start=0, fail=True)
        return Response(payload[5:], start=5, fail=False)

    monkeypatch.setattr(assembler.urllib.request, "urlopen", fake_open)
    with pytest.raises(urllib.error.URLError):
        assembler._download("https://huggingface.co/pinned", destination, receipt)
    assert (root / ".shard.safetensors.partial").read_bytes() == payload[:5]

    assembler._download("https://huggingface.co/pinned", destination, receipt)

    assert destination.read_bytes() == payload
    assert calls == [None, "bytes=5-"]
    assert not (root / ".shard.safetensors.partial").exists()


def test_durable_receipt_restarts_an_interrupted_partial(tmp_path: Path) -> None:
    root = _private(tmp_path / "receipts")
    destination = root / "source-manifest.json"
    partial = root / ".source-manifest.json.partial"
    partial.write_bytes(b"interrupted")
    partial.chmod(0o600)
    payload = b'{"complete":true}\n'

    assembler._durable_write_once(destination, payload)
    assembler._durable_write_once(destination, payload)

    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_worker_environment_requires_exact_reviewed_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewed = dict(worker.REVIEWED_DISTRIBUTIONS)
    receipt = {
        "versions": reviewed,
        "nvfp4_qtensor": worker.NVFP4_QTENSOR_IMPORT,
    }
    monkeypatch.setattr(worker, "_extract_overlay", lambda _request: None)
    monkeypatch.setattr(worker, "_pythonpath", lambda _request: "/reviewed")
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(receipt)
        ),
    )
    assert worker._verify_environment({}) == receipt

    reviewed["tokenizers"] = "0.22.2"
    with pytest.raises(worker.FlashBuildWorkerError):
        worker._verify_environment({})


def test_modelopt_runtime_wheel_closure_matches_every_pinned_receipt() -> None:
    source_manifest = (
        adapter.PACKAGE_ROOT / "aeon/core/data/qwen38_modelopt_runtime_wheels.json"
    )
    canonical_manifest = worker.LOCAL_MODELOPT_RUNTIME_MANIFEST
    assert worker._sha256(source_manifest) == worker.MODELOPT_RUNTIME_MANIFEST_SHA256
    assert source_manifest.read_bytes() == canonical_manifest.read_bytes()
    manifest = json.loads(canonical_manifest.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert len(manifest["runtime_wheels"]) == 12

    for name, receipt in manifest["runtime_wheels"].items():
        path = worker.LOCAL_MODELOPT_RUNTIME / name
        assert worker._verify_regular(path, receipt, 64 * 1024**2) == receipt["size"]
        worker._wheel_files(path, receipt)

    antlr = manifest["antlr_build"]
    assert antlr["byte_identical"] is True
    assert antlr["independent_builds"] == 2
    assert antlr["record_verified"] is True
    derived = manifest["runtime_wheels"][antlr["derived_wheel"]]
    assert (antlr["derived_sha256"], antlr["derived_size"]) == (
        derived["sha256"],
        derived["size"],
    )
    for name, receipt in {
        antlr["official_pypi_sdist"]["filename"]: antlr["official_pypi_sdist"],
        **antlr["toolchain"],
    }.items():
        path = worker.LOCAL_MODELOPT_RUNTIME / name
        assert path.stat().st_size == receipt["size"]
        assert worker._sha256(path) == receipt["sha256"]


def test_wheel_validation_rejects_special_inode_member(tmp_path: Path) -> None:
    wheel = tmp_path / "unsafe.whl"
    member = zipfile.ZipInfo("payload")
    member.external_attr = stat.S_IFIFO << 16
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, b"unsafe")

    with pytest.raises(worker.FlashBuildWorkerError, match="unsafe member"):
        worker._wheel_files(wheel, None)


def test_resume_checks_reject_live_historical_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = Path(worker.RESUME_ROOT).with_name("fr-" + "9" * 32)
    request = {
        "recipe": worker.RESUME_RECIPE,
        "host": worker.LOCAL_HOST,
        "resume_source_manifest_sha256": worker.RESUME_MANIFEST_SHA256,
        "scratch_path": str(fresh),
        "source_root": str(worker.LOCAL_SOURCE_ROOT),
    }
    monkeypatch.setattr(worker.os, "killpg", lambda _pid, signal_number: None)
    with pytest.raises(worker.FlashBuildWorkerError, match="group is still live"):
        worker._validate_resume_source(request)

    monkeypatch.setattr(adapter.os, "killpg", lambda _pid, signal_number: None)
    with pytest.raises(adapter.FlashNextBuildError, match="group is still live"):
        adapter._verify_resume_operational_state()


def test_worker_resume_accepts_only_esrch_for_historical_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = Path(worker.RESUME_ROOT).with_name("fr-" + "8" * 32)
    request = {
        "recipe": worker.RESUME_RECIPE,
        "host": worker.LOCAL_HOST,
        "resume_source_manifest_sha256": worker.RESUME_MANIFEST_SHA256,
        "scratch_path": str(fresh),
        "source_root": str(worker.LOCAL_SOURCE_ROOT),
    }
    calls: list[tuple[int, int]] = []

    def absent(pgid: int, signal_number: int) -> None:
        calls.append((pgid, signal_number))
        raise OSError(errno.ESRCH, "absent")

    def after_group_check(_path: Path, *, create: bool = False) -> Path:
        raise worker.FlashBuildWorkerError("reached source-tree validation")

    monkeypatch.setattr(worker.os, "killpg", absent)
    monkeypatch.setattr(worker, "_private_dir", after_group_check)
    with pytest.raises(
        worker.FlashBuildWorkerError, match="reached source-tree validation"
    ):
        worker._validate_resume_source(request)

    receipt = json.loads(worker.LOCAL_RESUME_MANIFEST.read_text(encoding="utf-8"))
    assert calls == [(receipt["source_runtime"]["pid"], 0)]


def _write_private_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _synthetic_resume_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resume_receipts: list[dict[str, str]],
) -> tuple[dict[str, Path], list[list[str]], dict[str, bytes]]:
    root = _private(tmp_path / "resume-pipeline")
    source = _private(root / "source")
    scratch = _private(root / "new-runtime")
    output = _private(scratch / "output")
    old = _private(root / "old-runtime")
    hybrid = _private(old / "hybrid")
    behavior = _private(old / "behavior-adapter")
    hybrid_marker = hybrid / "source-marker.bin"
    hybrid_marker.write_bytes(b"immutable-hybrid-source")
    hybrid_marker.chmod(0o600)
    behavior_manifest = behavior / "aeon_behavior_manifest.json"
    _write_private_json(behavior_manifest, {"complete": True})
    behavior_adapter = behavior / "adapter_model.safetensors"
    behavior_adapter.write_bytes(b"immutable-behavior-adapter")
    behavior_adapter.chmod(0o600)
    _write_private_json(
        old / "behavior-receipt.json",
        {
            "status": "completed",
            "manifest_sha256": worker._sha256(behavior_manifest),
        },
    )
    monkeypatch.setattr(worker, "RESUME_ROOT", old)

    request = {
        "recipe": worker.RESUME_RECIPE,
        "runtime_id": "fr-" + "7" * 32,
        "host": worker.LOCAL_HOST,
        "scratch_path": str(scratch),
        "source_root": str(source),
        "claim_id": "gc-synthetic-resume-1234",
        "owner": "synthetic-resume-owner",
        "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
        "fixture_files": {
            "mtp-bf16.manifest.json": {"sha256": "1" * 64},
            "expert-scales.manifest.json": {"sha256": "2" * 64},
        },
    }
    request_path = scratch / "qwen-flash-next-build-request.json"
    _write_private_json(request_path, request)
    first = resume_receipts[0]
    _write_private_json(
        output / "preflight.json",
        {
            "request_sha256": worker._sha256(request_path),
            "recipe": worker.RESUME_RECIPE,
            "environment": {"nvfp4_qtensor": worker.NVFP4_QTENSOR_IMPORT},
            "source_stage": {"resume_closure_sha256": first["closure_sha256"]},
            "resume_source": {"closure_sha256": first["closure_sha256"]},
        },
    )
    (output / "build.log").write_bytes(b"")
    (output / "build.log").chmod(0o600)
    source_before = {
        item.relative_to(old).as_posix(): item.read_bytes()
        for item in old.rglob("*")
        if item.is_file()
    }

    validation_calls: list[int] = []

    def validate(_request: dict) -> dict[str, str]:
        index = len(validation_calls)
        validation_calls.append(index)
        return dict(resume_receipts[index])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("resume must not run behavior training or assembly")

    commands: list[list[str]] = []

    def successful_builder(command, **_kwargs):
        commands.append(list(command))
        model = _private(output / "model")
        official = _private(output / "official-untuned-model")
        _write_private_json(model / "BUILD_MANIFEST.json", {"complete": True})
        (model / "SHA256SUMS").write_bytes(b"tuned\n")
        (model / "SHA256SUMS").chmod(0o600)
        (official / "SHA256SUMS").write_bytes(b"official\n")
        (official / "SHA256SUMS").chmod(0o600)
        _write_private_json(
            output / "BUILD_SIBLING_MANIFEST.json",
            {
                "schema_version": (
                    "aeon-qwen38-flash-next-official-untuned-sibling-v1"
                ),
                "complete": True,
                "tuned_checkpoint_tree_sha256": worker._sha256(model / "SHA256SUMS"),
                "official_untuned_checkpoint_tree_sha256": worker._sha256(
                    official / "SHA256SUMS"
                ),
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker, "_validate_resume_source", validate)
    monkeypatch.setattr(worker, "_run_behavior_trainer", forbidden)
    monkeypatch.setattr(worker.assembler, "assemble", forbidden)
    monkeypatch.setattr(worker.subprocess, "run", successful_builder)
    return worker._paths(request), commands, source_before


def test_quant_only_resume_skips_training_and_preserves_old_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume = {"closure_sha256": "3" * 64, "hybrid_manifest_sha256": "4" * 64}
    paths, commands, source_before = _synthetic_resume_pipeline(
        tmp_path, monkeypatch, [resume, resume]
    )

    worker._pipeline(
        {
            "recipe": worker.RESUME_RECIPE,
            "runtime_id": "fr-" + "7" * 32,
            "host": worker.LOCAL_HOST,
            "scratch_path": str(paths["scratch"]),
            "source_root": str(paths["source"]),
            "claim_id": "gc-synthetic-resume-1234",
            "owner": "synthetic-resume-owner",
            "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
            "fixture_files": {
                "mtp-bf16.manifest.json": {"sha256": "1" * 64},
                "expert-scales.manifest.json": {"sha256": "2" * 64},
            },
        }
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--hybrid") + 1] == str(paths["hybrid"])
    assert command[command.index("--adapter") + 1] == str(
        paths["behavior"] / "adapter_model.safetensors"
    )
    assert command[command.index("--output") + 1] == str(paths["model"])
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert result["recipe"] == worker.RESUME_RECIPE
    assert result["resume_source_closure_sha256"] == resume["closure_sha256"]
    assert {
        item.relative_to(Path(worker.RESUME_ROOT)).as_posix(): item.read_bytes()
        for item in Path(worker.RESUME_ROOT).rglob("*")
        if item.is_file()
    } == source_before


def test_quant_only_resume_fails_when_source_changes_during_quantization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {"closure_sha256": "5" * 64, "hybrid_manifest_sha256": "6" * 64}
    after = {"closure_sha256": "7" * 64, "hybrid_manifest_sha256": "6" * 64}
    paths, commands, source_before = _synthetic_resume_pipeline(
        tmp_path, monkeypatch, [before, after]
    )

    with pytest.raises(worker.FlashBuildWorkerError, match="changed during"):
        worker._pipeline(
            {
                "recipe": worker.RESUME_RECIPE,
                "runtime_id": "fr-" + "7" * 32,
                "host": worker.LOCAL_HOST,
                "scratch_path": str(paths["scratch"]),
                "source_root": str(paths["source"]),
                "claim_id": "gc-synthetic-resume-1234",
                "owner": "synthetic-resume-owner",
                "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
                "fixture_files": {
                    "mtp-bf16.manifest.json": {"sha256": "1" * 64},
                    "expert-scales.manifest.json": {"sha256": "2" * 64},
                },
            }
        )

    assert len(commands) == 1
    assert not paths["result"].exists()
    assert {
        item.relative_to(Path(worker.RESUME_ROOT)).as_posix(): item.read_bytes()
        for item in Path(worker.RESUME_ROOT).rglob("*")
        if item.is_file()
    } == source_before


def test_failed_behavior_trainer_removes_exact_offload_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = _private(tmp_path / ".behavior-work-test")
    offload = _private(work / "offload")
    shard = offload / "model.weight.dat"
    shard.write_bytes(b"reproducible-offload-scratch")
    shard.chmod(0o600)
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )

    with pytest.raises(worker.FlashBuildWorkerError, match="behavior trainer failed"):
        worker._run_behavior_trainer(
            ["reviewed-trainer"],
            environment={},
            source=tmp_path,
            log=None,
            work=work,
        )

    assert not work.exists()


def test_terminal_settlement_removes_abandoned_behavior_work(tmp_path: Path) -> None:
    scratch = _private(tmp_path / "runtime")
    _private(scratch / "output")
    source = _private(scratch / "source")
    runtime_id = "fr-" + "e" * 32
    work = _private(scratch / f".behavior-work-{runtime_id}")
    offload = _private(work / "offload")
    shard = offload / "model.weight.dat"
    shard.write_bytes(b"abandoned-reproducible-scratch")
    shard.chmod(0o600)

    result = worker._terminal(
        {
            "runtime_id": runtime_id,
            "scratch_path": str(scratch),
            "source_root": str(source),
        },
        "trainer was stopped",
    )

    assert result["success"] is False
    assert not work.exists()


def test_behavior_work_cleanup_refuses_hardlinked_content(tmp_path: Path) -> None:
    work = _private(tmp_path / ".behavior-work-test")
    outside = tmp_path / "preserve.bin"
    outside.write_bytes(b"not-exclusively-owned-by-work-tree")
    outside.chmod(0o600)
    linked = work / "linked.bin"
    os.link(outside, linked)

    with pytest.raises(worker.FlashBuildWorkerError, match="unsafe inode"):
        worker._remove_exact_work(work)

    assert work.is_dir()
    assert linked.is_file()
    assert outside.read_bytes() == b"not-exclusively-owned-by-work-tree"


def test_spawn_log_descriptor_is_append_only_after_creation() -> None:
    assert worker.SPAWN_LOG_FLAGS & os.O_APPEND
    assert worker.SPAWN_LOG_FLAGS & os.O_EXCL
    assert worker.SPAWN_LOG_FLAGS & os.O_CLOEXEC


def test_hybrid_contract_matches_builder_exact_keys() -> None:
    from aeon.scripts.build_qwen38_flash_next_nvfp4 import HYBRID_MANIFEST_CONTRACT

    assert set(HYBRID_MANIFEST_CONTRACT) == {
        "schema_version",
        "complete",
        "artifact",
        "sources",
        "upstream_metadata",
        "topology",
        "files",
    }
    assert HYBRID_MANIFEST_CONTRACT["topology"]["tensor_count"] == 1_659
    assert assembler.BF16_CONFIG_SHA256 == (
        "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
    )


def test_hybrid_config_copy_is_byte_exact(tmp_path: Path) -> None:
    source = _private(tmp_path / "source")
    destination = _private(tmp_path / "destination")
    payload = b'{ "text_config": {"split_ngram_parts": 128} }\n'
    (source / "config.json").write_bytes(payload)
    (source / "config.json").chmod(0o600)

    assembler._copy_metadata(source, destination, "config.json")

    assert (destination / "config.json").read_bytes() == payload
    assert "ple_embedding_dtype" not in Path(assembler.__file__).read_text(
        encoding="utf-8"
    )


def test_canonical_source_copy_is_independent_and_preserved(tmp_path: Path) -> None:
    root = _private(tmp_path / "canonical-copy")
    source = root / "source.safetensors"
    target = root / "target.safetensors"
    source.write_bytes(b"immutable-official-weights")
    source.chmod(0o600)

    assembler._copy_regular(source, target)

    assert source.read_bytes() == target.read_bytes()
    assert source.stat().st_ino != target.stat().st_ino
    assert source.stat().st_nlink == target.stat().st_nlink == 1


def test_canonical_repo_source_may_be_owner_readable(tmp_path: Path) -> None:
    source = tmp_path / "reviewed.py"
    source.write_bytes(b"reviewed = True\n")
    source.chmod(0o644)
    receipt = {
        "sha256": worker._sha256(source),
        "size": source.stat().st_size,
    }

    assert worker._verify_regular(source, receipt, 1024, private=False) == len(
        b"reviewed = True\n"
    )
    source.chmod(0o664)
    with pytest.raises(worker.FlashBuildWorkerError):
        worker._verify_regular(source, receipt, 1024, private=False)


def test_adapter_payload_is_closed() -> None:
    assert (
        adapter.AeonQwenFlashNextBuildAdapter._payload({"recipe": worker.RESUME_RECIPE})
        == worker.RESUME_RECIPE
    )

    for payload in (
        {},
        {"recipe": worker.FULL_RECIPE},
        {"recipe": "unknown"},
        {"recipe": worker.RESUME_RECIPE, "host": worker.LOCAL_HOST},
        {"host": worker.LOCAL_HOST},
        None,
    ):
        with pytest.raises(adapter.FlashNextBuildError):
            adapter.AeonQwenFlashNextBuildAdapter._payload(payload)  # type: ignore[arg-type]


def test_probe_delegates_exact_legacy_recovery_before_current_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = {"profile_id": "aeon-qwen38-flash-next-build"}
    expected = ProbeResult(
        ProbeState.ABSENT,
        process_identity_verified=False,
        process_absent=True,
        note="exact audited legacy absence",
        prelaunch_cleanup_verified=True,
    )
    seen: list[dict[str, str]] = []

    def legacy_probe(value):
        seen.append(value)
        return expected

    def current_identity_must_not_run(_runtime):
        raise AssertionError("current identity path ran for an exact legacy result")

    instance = adapter.AeonQwenFlashNextBuildAdapter()
    monkeypatch.setattr(
        adapter.legacy_recovery, "probe_legacy_pidless_build", legacy_probe
    )
    monkeypatch.setattr(instance, "_identity", current_identity_must_not_run)

    assert instance.probe(runtime) is expected
    assert seen == [runtime]


def test_probe_falls_through_for_current_v2_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = {"profile_id": adapter.PROFILE_ID}
    pid = 12_345
    calls: list[tuple[str, object]] = []
    instance = adapter.AeonQwenFlashNextBuildAdapter()

    def legacy_probe(value):
        calls.append(("legacy", value))
        return None

    def current_identity(value):
        calls.append(("identity", value))
        return "fr-" + "a" * 32, "b" * 64, pid

    def current_action(value, action, timeout):
        calls.append(("action", (value, action, timeout)))
        return {"state": "running", "pid": pid}

    monkeypatch.setattr(
        adapter.legacy_recovery, "probe_legacy_pidless_build", legacy_probe
    )
    monkeypatch.setattr(instance, "_identity", current_identity)
    monkeypatch.setattr(instance, "_action", current_action)

    result = instance.probe(runtime)

    assert result.state is ProbeState.RUNNING
    assert result.process_identity_verified is True
    assert result.process_absent is False
    assert calls == [
        ("legacy", runtime),
        ("identity", runtime),
        ("action", (runtime, "status", 90)),
    ]


def test_authorized_build_profile_binds_complete_input_and_disk_closure() -> None:
    profile_path = (
        adapter.PACKAGE_ROOT.parent
        / "fleet_compute/profiles.d/aeon-qwen38-flash-next-build.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    manifest = profile.pop("manifest_sha256")

    assert profile["enabled"] is True
    assert profile["version"] == 3
    assert profile["placements"] == [
        {"host": worker.LOCAL_HOST, "physical_gpu": 0, "enabled": True}
    ]
    assert profile["min_host_memory_gb"] == 170.0
    assert profile["min_host_commit_gb"] == 162.0
    assert profile["min_disk_free_gb"] == 191.0
    assert profile["stage_bytes_max"] == 1_000_000_000
    assert profile["runtime_growth_bytes_max"] == 170_000_000_000
    assert profile["worker_free_reserve_bytes"] == 20_000_000_000
    assert worker.RESUME_RECIPE in profile["purpose"]
    assert "full tune/assembly requests are refused" in profile["hard_cap_mechanism"]
    assert "no automatic cleanup" in profile["hard_cap_mechanism"]
    assert profile["personal_priority"] == 0
    assert profile["max_attempts"] == 3
    expected_identity = adapter.expected_artifact_identity()
    assert profile["artifact_identity"] == expected_identity
    assert adapter._canonical_sha(profile) == manifest


def test_worker_request_is_exact_and_uuid_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _private(tmp_path / "runs")
    runtime = "fr-" + "a" * 32
    scratch = _private(root / runtime)
    source = _private(scratch / "source")
    request_path = scratch / "qwen-flash-next-build-request.json"
    request = {
        "schema_version": worker.SCHEMA_VERSION,
        "runtime_id": runtime,
        "job_id": "job-test",
        "host": worker.HOST,
        "hostname": worker.HOSTNAME,
        "claim_id": "gc-test-claim-1234",
        "owner": "test-owner",
        "physical_gpu": 0,
        "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
        "vram_budget_gb": 88.0,
        "exclusive": True,
        "min_host_memory_gb": 170.0,
        "min_host_commit_gb": 162.0,
        "post_stage_disk_floor_bytes": 190_000_000_000,
        "min_shm_free_gb": 16.0,
        "scratch_path": str(scratch),
        "source_root": str(source),
        "source_files": {},
        "input_files": {},
        "fixture_files": {},
        "recipe": worker.FULL_RECIPE,
        "resume_source_manifest_sha256": None,
        "sglang_commit": worker.SGLANG_COMMIT,
        "sglang_image_digest": worker.SGLANG_IMAGE_DIGEST,
    }
    raw = (json.dumps(request, indent=2, sort_keys=True) + "\n").encode()
    request_path.write_bytes(raw)
    request_path.chmod(0o600)
    monkeypatch.setattr(worker, "SCRATCH_ROOT", PurePosixPath(root))
    monkeypatch.setattr(worker.socket, "gethostname", lambda: worker.HOSTNAME)

    assert (
        worker._validate_request(request_path, worker.hashlib.sha256(raw).hexdigest())
        == request
    )
    request["gpu_uuid"] = "0"
    changed = (json.dumps(request, indent=2, sort_keys=True) + "\n").encode()
    request_path.write_bytes(changed)
    request_path.chmod(0o600)
    with pytest.raises(worker.FlashBuildWorkerError):
        worker._validate_request(
            request_path, worker.hashlib.sha256(changed).hexdigest()
        )


def test_worker_request_accepts_only_gpu0_on_canonical_177(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical_root = _private(tmp_path / "canonical")
    source = _private(tmp_path / "source")
    runtime = "fr-" + "d" * 32
    scratch = _private(canonical_root / runtime)
    request_path = scratch / "qwen-flash-next-build-request.json"
    request = {
        "schema_version": worker.SCHEMA_VERSION,
        "runtime_id": runtime,
        "job_id": "job-local-test",
        "host": worker.LOCAL_HOST,
        "hostname": worker.LOCAL_HOSTNAME,
        "claim_id": "gc-local-test-1234",
        "owner": "local-test-owner",
        "physical_gpu": 0,
        "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
        "vram_budget_gb": 88.0,
        "exclusive": True,
        "min_host_memory_gb": 170.0,
        "min_host_commit_gb": 162.0,
        "post_stage_disk_floor_bytes": worker.FULL_LOCAL_POST_STAGE_DISK_FLOOR_BYTES,
        "min_shm_free_gb": 16.0,
        "scratch_path": str(scratch),
        "source_root": str(source),
        "source_files": {},
        "input_files": {},
        "fixture_files": {},
        "recipe": worker.FULL_RECIPE,
        "resume_source_manifest_sha256": None,
        "sglang_commit": worker.SGLANG_COMMIT,
        "sglang_image_digest": worker.SGLANG_IMAGE_DIGEST,
    }
    monkeypatch.setattr(worker, "LOCAL_CANONICAL_ROOT", PurePosixPath(canonical_root))
    monkeypatch.setattr(worker, "LOCAL_SOURCE_ROOT", source)
    monkeypatch.setattr(worker.socket, "gethostname", lambda: worker.LOCAL_HOSTNAME)

    raw = (json.dumps(request, indent=2, sort_keys=True) + "\n").encode()
    request_path.write_bytes(raw)
    request_path.chmod(0o600)
    assert (
        worker._validate_request(request_path, worker.hashlib.sha256(raw).hexdigest())
        == request
    )

    request["recipe"] = worker.RESUME_RECIPE
    request["resume_source_manifest_sha256"] = worker.RESUME_MANIFEST_SHA256
    request["post_stage_disk_floor_bytes"] = worker.RESUME_POST_STAGE_DISK_FLOOR_BYTES
    resume = (json.dumps(request, indent=2, sort_keys=True) + "\n").encode()
    request_path.write_bytes(resume)
    request_path.chmod(0o600)
    assert (
        worker._validate_request(
            request_path, worker.hashlib.sha256(resume).hexdigest()
        )
        == request
    )

    request["post_stage_disk_floor_bytes"] = (
        worker.FULL_LOCAL_POST_STAGE_DISK_FLOOR_BYTES
    )
    wrong_floor = (json.dumps(request, indent=2, sort_keys=True) + "\n").encode()
    request_path.write_bytes(wrong_floor)
    request_path.chmod(0o600)
    with pytest.raises(worker.FlashBuildWorkerError):
        worker._validate_request(
            request_path, worker.hashlib.sha256(wrong_floor).hexdigest()
        )
    request["post_stage_disk_floor_bytes"] = worker.RESUME_POST_STAGE_DISK_FLOOR_BYTES

    request["physical_gpu"] = 1
    changed = (json.dumps(request, indent=2, sort_keys=True) + "\n").encode()
    request_path.write_bytes(changed)
    request_path.chmod(0o600)
    with pytest.raises(worker.FlashBuildWorkerError):
        worker._validate_request(
            request_path, worker.hashlib.sha256(changed).hexdigest()
        )


def test_worker_refuses_automatic_cleanup_on_canonical_177() -> None:
    with pytest.raises(
        worker.FlashBuildWorkerError,
        match="never auto-cleaned",
    ):
        worker._cleanup({"host": worker.LOCAL_HOST}, "a" * 64, prelaunch=True)


def test_cleanup_scan_refuses_symlink(tmp_path: Path) -> None:
    root = _private(tmp_path / "scratch")
    (root / "safe").write_bytes(b"x")
    (root / "safe").chmod(0o600)
    os.symlink(root / "safe", root / "link")
    with pytest.raises(worker.FlashBuildWorkerError):
        worker._safe_tree_bytes(root)


def test_worker_preflight_writes_initial_receipt_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _private(tmp_path / "runs")
    runtime = "fr-" + "b" * 32
    scratch = _private(root / runtime)
    source = _private(scratch / "source")
    request = {
        "scratch_path": str(scratch),
        "source_root": str(source),
        "source_files": {},
        "input_files": {},
        "fixture_files": {},
        "recipe": worker.FULL_RECIPE,
    }
    monkeypatch.setattr(worker, "_verify_manifested", lambda *_args: 0)
    monkeypatch.setattr(worker, "_verify_acl", lambda _request: None)
    monkeypatch.setattr(
        worker.assembler, "stage_sources", lambda **_kwargs: {"complete": True}
    )
    monkeypatch.setattr(
        worker,
        "_resources",
        lambda _request: {"scratch_device": scratch.lstat().st_dev},
    )
    monkeypatch.setattr(worker, "_verify_environment", lambda _request: {})

    result = worker._preflight(request, "a" * 64)

    receipt = json.loads((scratch / "output/preflight.json").read_text())
    assert result["state"] == "preflight_ready"
    assert receipt["request_sha256"] == "a" * 64


def _incomplete_staging_request(root: Path) -> tuple[dict, str, Path]:
    runtime = "fr-" + "c" * 32
    scratch = _private(root / runtime)
    source = _private(scratch / "source")
    source_file = source / "aeon/scripts/worker.py"
    source_file.parent.mkdir(mode=0o700, parents=True)
    source_file.parent.parent.chmod(0o700)
    source_file.parent.chmod(0o700)
    source_file.write_bytes(b"reviewed worker")
    source_file.chmod(0o600)
    bf16 = scratch / "inputs/bf16/config.json"
    bf16.parent.mkdir(mode=0o700, parents=True)
    bf16.parent.parent.chmod(0o700)
    bf16.parent.chmod(0o700)
    bf16.write_bytes(b"{}\n")
    bf16.chmod(0o600)
    fp8 = scratch / "inputs/fp8-ple/config.json"
    fp8.parent.mkdir(mode=0o700, parents=True)
    fp8.parent.chmod(0o700)
    fp8.write_bytes(b"{}\n")
    fp8.chmod(0o600)
    fixture = scratch / "fixtures/receipt.json"
    fixture.parent.mkdir(mode=0o700)
    fixture.parent.chmod(0o700)
    fixture.write_bytes(b"{}\n")
    fixture.chmod(0o600)
    _private(scratch / "output")

    def receipt(path: Path) -> dict[str, int | str]:
        return {
            "sha256": worker._sha256(path),
            "size": path.stat().st_size,
        }

    request = {
        "runtime_id": runtime,
        "scratch_path": str(scratch),
        "source_root": str(source),
        "source_files": {"aeon/scripts/worker.py": receipt(source_file)},
        "input_files": {
            "bf16/config.json": receipt(bf16),
            "fp8-ple/config.json": receipt(fp8),
        },
        "fixture_files": {"receipt.json": receipt(fixture)},
    }
    request_path = scratch / "qwen-flash-next-build-request.json"
    raw = (json.dumps(request, indent=2, sort_keys=True) + "\n").encode()
    request_path.write_bytes(raw)
    request_path.chmod(0o600)
    return request, worker.hashlib.sha256(raw).hexdigest(), scratch


def test_missing_preflight_cleanup_requires_exact_request_owned_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _private(tmp_path / "runs")
    request, digest, scratch = _incomplete_staging_request(root)
    monkeypatch.setattr(worker, "SCRATCH_ROOT", PurePosixPath(root))

    result = worker._cleanup(request, digest, prelaunch=True)

    assert result["state"] == "cleaned"
    assert result["reclaimed_bytes"] > 0
    assert not scratch.exists()


@pytest.mark.parametrize(
    "changed", ("request", "output", "spawn", "undeclared", "symlink")
)
def test_missing_preflight_cleanup_refuses_changed_or_ambiguous_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    root = _private(tmp_path / changed / "runs")
    request, digest, scratch = _incomplete_staging_request(root)
    monkeypatch.setattr(worker, "SCRATCH_ROOT", PurePosixPath(root))
    if changed == "request":
        path = scratch / "qwen-flash-next-build-request.json"
        path.write_bytes(path.read_bytes() + b" ")
        path.chmod(0o600)
    elif changed == "output":
        path = scratch / "output/result.json"
        path.write_bytes(b"{}\n")
        path.chmod(0o600)
    elif changed == "spawn":
        path = scratch / "spawn.json"
        path.write_bytes(b"{}\n")
        path.chmod(0o600)
    elif changed == "symlink":
        os.symlink(scratch / "source", scratch / "linked-source")
    else:
        path = scratch / "unowned.bin"
        path.write_bytes(b"x")
        path.chmod(0o600)

    with pytest.raises(worker.FlashBuildWorkerError):
        worker._cleanup(request, digest, prelaunch=True)
    assert scratch.is_dir()
