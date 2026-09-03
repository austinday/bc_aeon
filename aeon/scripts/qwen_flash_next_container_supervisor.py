#!/usr/bin/env python3
"""Container PID1: CUDA-memory sampler followed by the exact SGLang child."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time
from typing import Any, Sequence


SCHEMA = "aeon-qwen38-flash-next-cuda-memory-v1"
INTERVAL_SECONDS = 0.1
# ``torch.cuda.mem_get_info`` can block behind a long SM120 kernel even though
# this sampler is a separate PID.  Four exact-card qualification receipts
# measured otherwise-continuous gaps of 1.112--1.848 seconds across hundreds to
# thousands of samples.  Two seconds is therefore the smallest round bound that
# admits the observed hardware behavior while still rejecting a stalled sampler.
MAX_GAP_SECONDS = 2.0
MIN_SAMPLE_DENSITY = 0.9
RESERVE_BYTES = 6 * 1024**3
_SHA = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME = re.compile(r"^fr-[0-9a-f]{32}$")
_GPU = re.compile(r"^GPU-[0-9a-fA-F-]{32,64}$")
_CONTAINER = re.compile(r"^[0-9a-f]{64}$")
_child: subprocess.Popen[bytes] | None = None


class SamplerError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SamplerError("attestation write was incomplete")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _forward(signum: int, _frame: Any) -> None:
    if _child is not None and _child.poll() is None:
        _child.send_signal(signum)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument(
        "--arm",
        choices=(
            "official_untuned",
            "tuned_mtp_off",
            "selection_candidate",
            "tuned_mtp_on_winner",
        ),
        required=True,
    )
    parser.add_argument("--claim-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--checkpoint-tree-sha256", required=True)
    parser.add_argument("server", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global _child
    args = _parser().parse_args(argv)
    server = list(args.server)
    if server[:1] == ["--"]:
        server.pop(0)
    if (
        not server
        or _RUNTIME.fullmatch(args.runtime_id) is None
        or _SHA.fullmatch(args.claim_sha256) is None
        or _SHA.fullmatch(args.checkpoint_tree_sha256) is None
        or _GPU.fullmatch(args.gpu_uuid) is None
        or any(not value or any(character in value for character in "\x00\r\n") for value in server)
    ):
        raise SamplerError("SGLang child command is absent")
    try:
        evidence_parent = args.output.parent.resolve(strict=True)
    except OSError as exc:
        raise SamplerError("CUDA evidence parent is absent") from exc
    parent_metadata = evidence_parent.lstat()
    if (
        args.output != evidence_parent / "cuda-memory.json"
        or args.freeze != evidence_parent / "freeze"
        or args.context != evidence_parent / "runtime-context.json"
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or parent_metadata.st_mode & 0o077
    ):
        raise SamplerError("CUDA evidence paths are not exact and private")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise SamplerError("visible CUDA device differs from the leased UUID")
    if hashlib.sha256(
        os.environ.get("GPU_AGENT_CLAIM_ID", "").encode("utf-8")
    ).hexdigest() != args.claim_sha256:
        raise SamplerError("container claim identity changed")
    if args.output.exists() or args.output.is_symlink() or args.freeze.exists() or args.freeze.is_symlink():
        raise SamplerError("CUDA sampler output/freeze path already exists")
    try:
        import torch
    except ImportError as exc:
        raise SamplerError("pinned SGLang image lacks torch") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SamplerError("container does not expose exactly one CUDA device")
    torch.cuda.set_device(0)
    free, total = (int(value) for value in torch.cuda.mem_get_info(0))
    if total <= RESERVE_BYTES or not 0 <= free <= total:
        raise SamplerError("initial CUDA memory accounting is malformed")
    started_at = _now()
    started_monotonic = time.monotonic()
    last_sample = started_monotonic
    first_sample_at = started_at
    last_sample_at = started_at
    min_free = free
    max_used = total - free
    min_reserve_at = started_at
    max_used_at = started_at
    max_gap = 0.0
    sample_count = 0
    digest = hashlib.sha256()

    def record(sample_free: int, sample_total: int, sampled: float) -> None:
        nonlocal min_free, max_used, min_reserve_at, max_used_at
        nonlocal max_gap, sample_count, last_sample, last_sample_at
        gap = sampled - last_sample
        max_gap = max(max_gap, gap)
        last_sample = sampled
        sample_at = _now()
        observed_used = sample_total - sample_free
        if sample_free < min_free:
            min_free = sample_free
            min_reserve_at = sample_at
        if observed_used > max_used:
            max_used = observed_used
            max_used_at = sample_at
        digest.update(
            _canonical(
                {
                    "sample": sample_count,
                    "monotonic_offset_seconds": sampled - started_monotonic,
                    "free_bytes": sample_free,
                    "total_bytes": sample_total,
                }
            )
        )
        last_sample_at = sample_at
        sample_count += 1

    def publish(*, complete: bool) -> dict[str, Any]:
        try:
            context = json.loads(args.context.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            context = {}
        if complete and (
            set(context)
            != {
                "container_id",
                "container_pid",
                "cgroup_path",
                "container_pid_in_cgroup",
            }
            or _CONTAINER.fullmatch(str(context.get("container_id") or "")) is None
            or type(context.get("container_pid")) is not int
            or int(context["container_pid"]) <= 1
            or not str(context.get("cgroup_path") or "").startswith("/sys/fs/cgroup/")
            or context.get("container_pid_in_cgroup") is not True
        ):
            raise SamplerError("final sampler context is not exact")
        value = {
            "schema_version": SCHEMA,
            "complete": complete,
            "runtime_id": args.runtime_id,
            "arm": args.arm,
            "lease_claim_id_sha256": args.claim_sha256,
            "leased_gpu_uuid_sha256": hashlib.sha256(
                args.gpu_uuid.encode("utf-8")
            ).hexdigest(),
            "container_id": context.get("container_id"),
            "container_pid": context.get("container_pid"),
            "cgroup_path": context.get("cgroup_path"),
            "started_at": started_at,
            "completed_at": _now() if complete else None,
            "first_sample_at": first_sample_at,
            "last_sample_at": last_sample_at,
            "sample_interval_seconds": INTERVAL_SECONDS,
            "max_sample_gap_seconds": max_gap,
            "sample_count": sample_count,
            "total_bytes": total,
            "min_free_bytes": min_free,
            "max_used_bytes": max_used,
            "max_used_at": max_used_at,
            "min_reserve_bytes": min_free,
            "min_reserve_at": min_reserve_at,
            "reserve_required_bytes": RESERVE_BYTES,
            "reserve_passed": min_free >= RESERVE_BYTES,
            "samples_sha256": digest.hexdigest(),
        }
        _atomic(args.output, value)
        return value

    record(free, total, started_monotonic)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _forward)
    _child = subprocess.Popen(server, stdin=subprocess.DEVNULL, close_fds=True)
    frozen = False
    final: dict[str, Any] | None = None
    while _child.poll() is None:
        if not frozen:
            sampled = time.monotonic()
            free, observed_total = (int(value) for value in torch.cuda.mem_get_info(0))
            if observed_total != total or not 0 <= free <= total:
                raise SamplerError("CUDA memory geometry changed")
            record(free, total, sampled)
            if args.context.is_file() and not args.context.is_symlink():
                final = publish(complete=False)
            if args.freeze.is_file() and not args.freeze.is_symlink():
                freeze_metadata = args.freeze.lstat()
                if (
                    not stat.S_ISREG(freeze_metadata.st_mode)
                    or freeze_metadata.st_uid != os.geteuid()
                    or freeze_metadata.st_mode & 0o077
                    or freeze_metadata.st_size > 1024
                ):
                    raise SamplerError("CUDA sampler freeze marker is unsafe")
                frozen = True
                final = publish(complete=True)
        time.sleep(INTERVAL_SECONDS)
    if not frozen:
        final = publish(complete=True)
    if final is None:
        raise SamplerError("CUDA sampler produced no final attestation")
    if (
        final["sample_count"] < 10
        or final["sample_count"]
        < MIN_SAMPLE_DENSITY
        * ((last_sample - started_monotonic) / INTERVAL_SECONDS + 1)
        or not math.isfinite(float(final["max_sample_gap_seconds"]))
        or final["max_sample_gap_seconds"] > MAX_GAP_SECONDS
        or final["reserve_passed"] is not True
    ):
        return 1
    return int(_child.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
