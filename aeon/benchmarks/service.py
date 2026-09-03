"""Owner-private durable benchmark queue and sanitized query surface."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .catalog import (
    BENCHMARK_CATALOG_SHA256,
    BENCHMARK_CATALOG_VERSION,
    BENCHMARK_SCHEMA_VERSION,
    EXECUTOR_PROTOCOL_SHA256,
    EXECUTOR_PROTOCOL_VERSION,
    HARNESS_SOURCE_SHA256,
    RUNNER_PROTOCOL_SHA256,
    RUNNER_PROTOCOL_VERSION,
    RUNNER_SOURCE_SHA256,
    SUITES,
    TOOL_PROFILES,
    TOOL_SOURCE_SHA256,
    COMPONENTS,
    DEFAULT_SUITE_ID,
    combination_for,
    combination_sha256,
    public_catalog,
    valid_combinations,
)


FLEET_LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
DATABASE_FILENAME = "benchmarks.sqlite3"
EVIDENCE_DIRECTORY_NAME = "evidence"
DATABASE_SCHEMA_VERSION = 5
WORKER_START_GRACE_SECONDS = 120.0
MAX_REPETITIONS = 20
MAX_RUN_LIST = 500
MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
REQUEST_ID_RE = re.compile(r"^br-[0-9a-f]{32}$")
RUN_ID_RE = re.compile(r"^run-[0-9a-f]{32}$")
BATCH_REQUEST_ID_RE = re.compile(r"^bm-[0-9a-f]{32}$")
BATCH_ID_RE = re.compile(r"^batch-[0-9a-f]{32}$")
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
ACTIVE_STATUSES = frozenset(
    {"pending", "queued", "waiting_for_compute", "starting", "running", "cancelling"}
)
_SUBMIT_KEYS = frozenset(
    {
        "request_id",
        "suite_id",
        "harness_id",
        "model_id",
        "tool_profile_id",
        "repetitions",
    }
)
_MATRIX_KEYS = frozenset(
    {
        "request_id",
        "suite_id",
        "harness_id",
        "model_id",
        "tool_profile_id",
        "repetitions",
        "missing_only",
    }
)
MAX_MATRIX_COMBINATIONS = 64
MAX_MATRIX_PLANNED_CASES = 4096
_SAFE_ENVIRONMENT = frozenset(
    {
        "AEON_COMPUTE_BACKEND",
        "AEON_OPENCODE_HOME",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "PATH",
        "PYTHONPATH",
        "TMPDIR",
        "TZ",
        "USER",
        "VIRTUAL_ENV",
    }
)


class BenchmarkError(RuntimeError):
    """Benchmark state or execution could not be handled safely."""


class BenchmarkExecutionUnavailable(BenchmarkError):
    """A fresh request cannot run with the currently available harnesses."""


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _owned_process_starttime(pid: int) -> int | None:
    """Return one owner process's kernel start time, without process discovery."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return None
    try:
        proc = Path("/proc") / str(pid)
        if proc.stat().st_uid != os.geteuid():
            return None
        raw = (proc / "stat").read_text(encoding="ascii")
        fields = raw[raw.rfind(")") + 2 :].split()
        if fields[0] == "Z":
            return None
        return int(fields[19])
    except (OSError, UnicodeError, ValueError, IndexError):
        return None


def _assert_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute() or expanded == Path("/"):
        raise BenchmarkError("benchmark state root must be an absolute private directory")
    return Path(os.path.normpath(os.fspath(expanded)))


def _assert_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise BenchmarkError("benchmark state path contains a symbolic link")


def _private_directory(path: Path, *, create: bool) -> Path:
    _assert_no_symlink_components(path)
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    _assert_no_symlink_components(path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BenchmarkError("benchmark state directory is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise BenchmarkError("benchmark state directory must be owner-private")
    return path


def _private_regular_file(path: Path, *, maximum_bytes: int | None = None) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BenchmarkError("benchmark state file is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or (maximum_bytes is not None and metadata.st_size > maximum_bytes)
    ):
        raise BenchmarkError("benchmark state file is not an owner-private regular file")
    return metadata


def _create_private_file(path: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    os.close(descriptor)


def _atomic_private_json(directory: Path, name: str, value: object) -> tuple[Path, str]:
    if not name or "/" in name or name in {".", ".."}:
        raise BenchmarkError("benchmark evidence filename is invalid")
    payload = _canonical_json(value)
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise BenchmarkError("benchmark evidence exceeds its size limit")
    directory = _private_directory(directory, create=True)
    temp_name = f".{name}.tmp-{secrets.token_hex(12)}"
    directory_fd = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor: int | None = None
    try:
        try:
            existing = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode)
            or existing.st_uid != os.geteuid()
            or existing.st_nlink != 1
            or stat.S_IMODE(existing.st_mode) != 0o600
        ):
            raise BenchmarkError("benchmark evidence target is unsafe")
        descriptor = os.open(
            temp_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temp_name, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as exc:
        raise BenchmarkError("benchmark evidence could not be published") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)
    return directory / name, _sha256(payload)


def _default_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in _SAFE_ENVIRONMENT
        and isinstance(key, str)
        and isinstance(value, str)
        and "\x00" not in key
        and "\x00" not in value
    }
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def default_launcher(argv: Sequence[str]) -> subprocess.Popen[bytes]:
    """Launch only the fixed low-priority worker command, without a shell."""

    command = [str(part) for part in argv]
    if (
        len(command) != 8
        or command[0] != FLEET_LOW_PRIORITY
        or not Path(command[1]).is_absolute()
        or command[2:4] != ["-m", "aeon.benchmarks.worker"]
        or command[4] != "--root"
        or not Path(command[5]).is_absolute()
        or command[6] != "--run-id"
        or not RUN_ID_RE.fullmatch(command[7])
    ):
        raise BenchmarkError("benchmark worker must use fleet-low-priority")
    try:
        wrapper = Path(FLEET_LOW_PRIORITY).lstat()
    except OSError as exc:
        raise BenchmarkError("fleet-low-priority is unavailable") from exc
    if (
        not stat.S_ISREG(wrapper.st_mode)
        or stat.S_ISLNK(wrapper.st_mode)
        or wrapper.st_uid != os.geteuid()
        or wrapper.st_nlink != 1
        or not wrapper.st_mode & stat.S_IXUSR
        or stat.S_IMODE(wrapper.st_mode) & 0o022
    ):
        raise BenchmarkError("fleet-low-priority identity is unsafe")
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
        shell=False,
        env=_default_environment(),
    )


class BenchmarkService:
    """Durable benchmark CRUD and launch service used by Nexus."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        launcher: Callable[[Sequence[str]], object] | None = None,
    ) -> None:
        self.root = _private_directory(_assert_absolute(Path(root)), create=True)
        self.evidence_root = _private_directory(
            self.root / EVIDENCE_DIRECTORY_NAME, create=True
        )
        self.database_path = self.root / DATABASE_FILENAME
        if not os.path.lexists(self.database_path):
            _create_private_file(self.database_path)
        _private_regular_file(self.database_path)
        self._launcher = default_launcher if launcher is None else launcher
        self._lock = threading.RLock()
        self._initialize_database()
        # A matrix is a durable queue, not a browser-session loop.  Reconcile
        # exact dead workers and atomically claim one pending matrix child on
        # service construction so a restart cannot strand the batch.
        self._reconcile_stale_runs()

    def _connect(self) -> sqlite3.Connection:
        _private_regular_file(self.database_path)
        connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize_database(self) -> None:
        with self._lock, self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, 1, 2, 3, 4, DATABASE_SCHEMA_VERSION}:
                raise BenchmarkError("benchmark database schema is unsupported")
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id TEXT PRIMARY KEY NOT NULL,
                    request_id TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    suite_id TEXT NOT NULL,
                    suite_label TEXT NOT NULL,
                    suite_version TEXT NOT NULL,
                    suite_sha256 TEXT NOT NULL,
                    harness_id TEXT NOT NULL,
                    harness_label TEXT NOT NULL,
                    harness_version TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_label TEXT NOT NULL,
                    model_revision TEXT NOT NULL,
                    tool_profile_id TEXT NOT NULL,
                    tool_profile_label TEXT NOT NULL,
                    tool_profile_version TEXT NOT NULL,
                    repetitions INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    request_sha256 TEXT NOT NULL,
                    catalog_version TEXT NOT NULL,
                    catalog_sha256 TEXT NOT NULL,
                    runner_protocol_version TEXT NOT NULL,
                    runner_protocol_sha256 TEXT NOT NULL,
                    executor_protocol_version TEXT NOT NULL,
                    executor_protocol_sha256 TEXT NOT NULL,
                    runner_source_sha256 TEXT NOT NULL,
                    harness_source_sha256 TEXT NOT NULL,
                    tool_source_sha256 TEXT NOT NULL,
                    combination_sha256 TEXT NOT NULL,
                    evidence_sha256 TEXT,
                    worker_pid INTEGER,
                    worker_starttime INTEGER,
                    worker_registered_at REAL
                );
                CREATE INDEX IF NOT EXISTS benchmark_runs_created
                    ON benchmark_runs(created_at DESC, id DESC);
                CREATE TABLE IF NOT EXISTS benchmark_batches (
                    id TEXT PRIMARY KEY NOT NULL,
                    request_id TEXT UNIQUE NOT NULL,
                    created_at REAL NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    catalog_version TEXT NOT NULL,
                    catalog_sha256 TEXT NOT NULL,
                    missing_only INTEGER NOT NULL,
                    repetitions INTEGER NOT NULL,
                    harness_selection TEXT NOT NULL,
                    model_selection TEXT NOT NULL,
                    tool_profile_selection TEXT NOT NULL,
                    selected_count INTEGER NOT NULL,
                    created_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS benchmark_batch_runs (
                    batch_id TEXT NOT NULL REFERENCES benchmark_batches(id),
                    ordinal INTEGER NOT NULL,
                    combination_id TEXT NOT NULL,
                    run_id TEXT NOT NULL REFERENCES benchmark_runs(id),
                    created_child INTEGER NOT NULL,
                    PRIMARY KEY (batch_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS benchmark_batch_runs_run
                    ON benchmark_batch_runs(run_id, batch_id);
                COMMIT;
                """
            )
            if version in {1, 2, 3}:
                columns = {
                    str(row[1])
                    for row in connection.execute(
                        "PRAGMA table_info(benchmark_runs)"
                    ).fetchall()
                }
                connection.execute("BEGIN IMMEDIATE")
                for name, kind in (
                    ("worker_pid", "INTEGER"),
                    ("worker_starttime", "INTEGER"),
                    ("worker_registered_at", "REAL"),
                    ("executor_protocol_version", "TEXT"),
                    ("executor_protocol_sha256", "TEXT"),
                    ("runner_source_sha256", "TEXT"),
                    ("harness_source_sha256", "TEXT"),
                    ("tool_source_sha256", "TEXT"),
                ):
                    if name not in columns:
                        connection.execute(
                            f"ALTER TABLE benchmark_runs ADD COLUMN {name} {kind}"
                        )
                connection.execute(
                    """
                    UPDATE benchmark_runs
                    SET executor_protocol_version = ?, executor_protocol_sha256 = ?
                    WHERE executor_protocol_version IS NULL
                       OR executor_protocol_sha256 IS NULL
                    """,
                    ("legacy-unbound", "0" * 64),
                )
                connection.execute(
                    """
                    UPDATE benchmark_runs
                    SET runner_source_sha256 = COALESCE(runner_source_sha256, ?),
                        harness_source_sha256 = COALESCE(harness_source_sha256, ?),
                        tool_source_sha256 = COALESCE(tool_source_sha256, ?)
                    """,
                    ("0" * 64, "0" * 64, "0" * 64),
                )
                connection.execute("COMMIT")
            connection.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}")
        os.chmod(self.database_path, 0o600, follow_symlinks=False)
        _private_regular_file(self.database_path)

    @staticmethod
    def _string(value: object, *, field: str) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"invalid benchmark {field}")
        return value

    def _normalize_request(self, request: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(request, Mapping):
            raise TypeError("benchmark request must be a mapping")
        unknown = set(request) - _SUBMIT_KEYS
        if unknown:
            raise ValueError("benchmark request contains unsupported fields")
        request_id = self._string(request.get("request_id"), field="request ID")
        if not REQUEST_ID_RE.fullmatch(request_id):
            raise ValueError("invalid benchmark request ID")
        suite_value = request.get("suite_id")
        suite_id = (
            DEFAULT_SUITE_ID
            if suite_value is None
            else self._string(suite_value, field="suite")
        )
        if suite_id != DEFAULT_SUITE_ID:
            raise ValueError("partial benchmark suites cannot be submitted")
        harness_id = self._string(request.get("harness_id"), field="harness")
        model_id = self._string(request.get("model_id"), field="model")
        suite = SUITES.get(suite_id)
        if suite is None:
            raise ValueError("unknown benchmark suite")
        tool_value = request.get("tool_profile_id")
        tool_profile_id = (
            suite.default_tool_profile_id
            if tool_value is None
            else self._string(tool_value, field="tool profile")
        )
        combination = combination_for(harness_id, model_id, tool_profile_id)
        if combination is None:
            raise ValueError("unsupported benchmark combination")
        tool_spec = next(
            (item for item in TOOL_PROFILES if item.profile_id == tool_profile_id),
            None,
        )
        if tool_spec is None or not tool_spec.supports(suite.required_capabilities):
            raise ValueError(
                "benchmark tool profile does not provide the suite's required capabilities"
            )
        repetitions = request.get("repetitions", 1)
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or not 1 <= repetitions <= MAX_REPETITIONS
        ):
            raise ValueError("benchmark repetitions must be between 1 and 20")
        catalog = public_catalog()
        harness = next(
            item for item in catalog["harnesses"] if item["id"] == harness_id
        )
        model = next(item for item in catalog["models"] if item["id"] == model_id)
        tool = next(
            item
            for item in catalog["tool_profiles"]
            if item["id"] == tool_profile_id
        )
        normalized = {
            "request_id": request_id,
            "suite_id": suite_id,
            "suite_label": suite.label,
            "suite_version": suite.version,
            "suite_sha256": suite.sha256,
            "harness_id": harness_id,
            "harness_label": str(harness["label"]),
            "harness_version": str(combination["harness_version"]),
            "model_id": model_id,
            "model_label": str(model["label"]),
            "model_revision": str(combination["model_revision"]),
            "tool_profile_id": tool_profile_id,
            "tool_profile_label": str(tool["label"]),
            "tool_profile_version": str(combination["tool_profile_version"]),
            "repetitions": repetitions,
            "catalog_version": BENCHMARK_CATALOG_VERSION,
            "catalog_sha256": BENCHMARK_CATALOG_SHA256,
            "runner_protocol_version": RUNNER_PROTOCOL_VERSION,
            "runner_protocol_sha256": RUNNER_PROTOCOL_SHA256,
            "executor_protocol_version": EXECUTOR_PROTOCOL_VERSION,
            "executor_protocol_sha256": EXECUTOR_PROTOCOL_SHA256,
            "runner_source_sha256": RUNNER_SOURCE_SHA256,
            "harness_source_sha256": HARNESS_SOURCE_SHA256,
            "tool_source_sha256": TOOL_SOURCE_SHA256,
            "combination_sha256": combination_sha256(combination),
        }
        normalized["request_sha256"] = _sha256(_canonical_json(normalized))
        return normalized

    @staticmethod
    def _run_id(value: object) -> str:
        if not isinstance(value, str) or not RUN_ID_RE.fullmatch(value):
            raise KeyError("invalid benchmark run ID")
        return value

    @staticmethod
    def _safe_summary(value: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            return {}
        allowed = {
            "score",
            "overall_score",
            "quality_score",
            "completion_rate",
            "median_wall_ms",
            "median_active_wall_ms",
            "median_compute_wait_ms",
            "stuck_rate",
            "unsupported_rate",
            "tool_success_rate",
            "browser_success_rate",
            "vision_score",
            "case_count",
            "passed_cases",
            "total_wall_ms",
            "total_active_wall_ms",
            "total_compute_wait_ms",
            "model_turn_count",
            "model_call_count",
            "tool_call_count",
            "prompt_tokens",
            "peak_prompt_tokens",
            "context_tokens",
            "completion_tokens",
            "context_pressure_bytes",
            "context_pressure_turns",
            "highest_verified_context_pressure_bytes",
            "fleet_compute_judgment_score",
            "preemption_recovery_score",
            "useful_wait_work_score",
            "checkpoint_reacquire_score",
            "duplicate_submission_count",
            "useful_overlap_ratio",
            "idle_wait_ratio",
            "max_parallelism",
            "integration_score",
        }
        nullable = {
            "model_turn_count",
            "model_call_count",
            "tool_call_count",
            "prompt_tokens",
            "peak_prompt_tokens",
            "context_tokens",
            "completion_tokens",
            "fleet_compute_judgment_score",
            "preemption_recovery_score",
            "useful_wait_work_score",
            "checkpoint_reacquire_score",
            "useful_overlap_ratio",
            "idle_wait_ratio",
            "integration_score",
        }
        result: dict[str, object] = {}
        for key in allowed:
            if key not in value:
                continue
            item = value.get(key)
            if item is None and key in nullable:
                result[key] = None
                continue
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
            ):
                continue
            result[key] = item
        component_value = value.get("component_scores")
        if isinstance(component_value, Mapping):
            known = {component.component_id for component in COMPONENTS}
            components: dict[str, float] = {}
            for key, item in component_value.items():
                if (
                    key in known
                    and not isinstance(item, bool)
                    and isinstance(item, (int, float))
                    and math.isfinite(float(item))
                    and 0.0 <= float(item) <= 100.0
                ):
                    components[str(key)] = float(item)
            result["component_scores"] = components
        token_complete = value.get("token_metrics_complete")
        if isinstance(token_complete, bool):
            result["token_metrics_complete"] = token_complete
        return result

    @staticmethod
    def _safe_cases(
        value: Sequence[Mapping[str, object]],
        *,
        suite_id: str,
        repetitions: int,
    ) -> list[dict[str, object]]:
        suite = SUITES.get(suite_id)
        if suite is None:
            raise BenchmarkError("benchmark suite provenance is unavailable")
        scenarios = {item.case_id: item for item in suite.cases}
        results: list[dict[str, object]] = []
        seen: set[tuple[str, int]] = set()
        for item in value:
            if not isinstance(item, Mapping):
                raise BenchmarkError("benchmark case evidence is invalid")
            case_id = item.get("case_id")
            repetition = item.get("repetition")
            status_value = item.get("status")
            scenario = scenarios.get(case_id) if isinstance(case_id, str) else None
            if (
                scenario is None
                or isinstance(repetition, bool)
                or not isinstance(repetition, int)
                or not 1 <= repetition <= repetitions
                or status_value
                not in {"passed", "failed", "timeout", "stuck", "unsupported"}
                or (case_id, repetition) in seen
            ):
                raise BenchmarkError("benchmark case evidence is invalid")
            seen.add((case_id, repetition))
            score = item.get("score")
            wall_ms = item.get("wall_ms")
            active_wall_ms = item.get("active_wall_ms")
            compute_wait_ms = item.get("compute_wait_ms")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
                or isinstance(wall_ms, bool)
                or not isinstance(wall_ms, (int, float))
                or not math.isfinite(float(wall_ms))
                or not 0.0 <= float(wall_ms) <= 86_400_000.0
                or isinstance(active_wall_ms, bool)
                or not isinstance(active_wall_ms, (int, float))
                or not math.isfinite(float(active_wall_ms))
                or not 0.0 <= float(active_wall_ms) <= float(wall_ms)
                or isinstance(compute_wait_ms, bool)
                or not isinstance(compute_wait_ms, (int, float))
                or not math.isfinite(float(compute_wait_ms))
                or not 0.0 <= float(compute_wait_ms) <= float(wall_ms)
                or abs(
                    float(active_wall_ms)
                    + float(compute_wait_ms)
                    - float(wall_ms)
                ) > 1.0
            ):
                raise BenchmarkError("benchmark case metrics are invalid")
            record: dict[str, object] = {
                "id": f"{case_id}:{repetition}",
                "case_id": case_id,
                "label": scenario.label,
                "category": scenario.category,
                "component_id": scenario.component_id,
                "repetition": repetition,
                "status": status_value,
                "score": float(score),
                "wall_ms": float(wall_ms),
                "active_wall_ms": float(active_wall_ms),
                "compute_wait_ms": float(compute_wait_ms),
            }
            for field in ("tool_success", "browser_success"):
                metric = item.get(field)
                if isinstance(metric, bool):
                    record[field] = metric
            vision = item.get("vision_score")
            if (
                not isinstance(vision, bool)
                and isinstance(vision, (int, float))
                and math.isfinite(float(vision))
                and 0.0 <= float(vision) <= 1.0
            ):
                record["vision_score"] = float(vision)
            if status_value != "passed":
                record["error_code"] = {
                    "timeout": "case_timeout",
                    "stuck": "case_stuck",
                    "unsupported": "case_unsupported",
                }.get(str(status_value), "case_failed")
            for field in (
                "model_turn_count",
                "model_call_count",
                "tool_call_count",
                "prompt_tokens",
                "peak_prompt_tokens",
                "context_tokens",
                "completion_tokens",
            ):
                metric = item.get(field)
                if metric is None:
                    record[field] = None
                elif (
                    isinstance(metric, int)
                    and not isinstance(metric, bool)
                    and 0 <= metric <= 1_000_000_000
                ):
                    record[field] = metric
            for field in (
                "context_pressure_bytes",
                "context_pressure_turns",
                "highest_verified_context_pressure_bytes",
                "duplicate_submission_count",
                "max_parallelism",
            ):
                metric = item.get(field)
                if (
                    isinstance(metric, int)
                    and not isinstance(metric, bool)
                    and 0 <= metric <= 1_000_000_000
                ):
                    record[field] = metric
            for field, high in (
                ("fleet_compute_judgment_score", 100.0),
                ("preemption_recovery_score", 100.0),
                ("useful_wait_work_score", 100.0),
                ("checkpoint_reacquire_score", 100.0),
                ("useful_overlap_ratio", 1.0),
                ("idle_wait_ratio", 1.0),
                ("integration_score", 100.0),
            ):
                metric = item.get(field)
                if (
                    not isinstance(metric, bool)
                    and isinstance(metric, (int, float))
                    and math.isfinite(float(metric))
                    and 0.0 <= float(metric) <= high
                ):
                    record[field] = float(metric)
            results.append(record)
        return results

    def _evidence_payload(self, row: sqlite3.Row) -> dict[str, object] | None:
        digest = row["evidence_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return None
        path = self.evidence_root / str(row["id"]) / "results.json"
        try:
            metadata = _private_regular_file(path, maximum_bytes=MAX_EVIDENCE_BYTES)
            payload_bytes = path.read_bytes()
            if len(payload_bytes) != metadata.st_size or not hmac.compare_digest(
                _sha256(payload_bytes), digest
            ):
                return None
            payload = json.loads(payload_bytes)
        except (BenchmarkError, OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("run_id") != row["id"]:
            return None
        return payload

    def _public_run(self, row: sqlite3.Row, *, include_cases: bool) -> dict[str, object]:
        try:
            summary = self._safe_summary(json.loads(row["summary_json"]))
        except (TypeError, json.JSONDecodeError):
            summary = {}
        result: dict[str, object] = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "id": row["id"],
            "request_id": row["request_id"],
            "status": row["status"],
            "suite_id": row["suite_id"],
            "suite_label": row["suite_label"],
            "suite_version": row["suite_version"],
            "harness_id": row["harness_id"],
            "harness_label": row["harness_label"],
            "harness_version": row["harness_version"],
            "model_id": row["model_id"],
            "model_label": row["model_label"],
            "model_revision": row["model_revision"],
            "tool_profile_id": row["tool_profile_id"],
            "tool_profile_label": row["tool_profile_label"],
            "tool_profile_version": row["tool_profile_version"],
            "repetitions": row["repetitions"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "summary": summary,
            "provenance": {
                "catalog_version": row["catalog_version"],
                "catalog_sha256": row["catalog_sha256"],
                "suite_sha256": row["suite_sha256"],
                "request_sha256": row["request_sha256"],
                "runner_protocol_version": row["runner_protocol_version"],
                "runner_protocol_sha256": row["runner_protocol_sha256"],
                "executor_protocol_version": row["executor_protocol_version"],
                "executor_protocol_sha256": row["executor_protocol_sha256"],
                "runner_source_sha256": row["runner_source_sha256"],
                "harness_source_sha256": row["harness_source_sha256"],
                "tool_source_sha256": row["tool_source_sha256"],
                "combination_sha256": row["combination_sha256"],
                "evidence_sha256": row["evidence_sha256"],
            },
        }
        if row["error_code"]:
            result["error_code"] = row["error_code"]
        if include_cases:
            evidence = self._evidence_payload(row)
            cases = evidence.get("cases", []) if evidence else []
            result["cases"] = cases if isinstance(cases, list) else []
            result["evidence_verified"] = evidence is not None
        return result

    def _fetch_row(self, connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM benchmark_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return row

    def catalog(self) -> dict[str, object]:
        # Import lazily to keep service construction side-effect free and avoid
        # the executor's intentional type-only dependency on this class.
        from .executor import runtime_execution_status

        catalog = public_catalog()
        statuses: dict[str, dict[str, object]] = {}
        for harness in catalog["harnesses"]:
            harness_id = str(harness["id"])
            status = runtime_execution_status(harness_id)
            statuses[harness_id] = status
            harness["available"] = status.get("supported") is True
            harness["unavailable_reason"] = (
                ""
                if status.get("supported") is True
                else str(status.get("reason") or "Harness unavailable.")[:200]
            )
        supported = any(
            status.get("supported") is True for status in statuses.values()
        )
        catalog["submission_supported"] = supported
        catalog["submission_unavailable_reason"] = (
            ""
            if supported
            else next(
                (
                    str(status.get("reason"))[:200]
                    for status in statuses.values()
                    if status.get("supported") is not True
                ),
                "Benchmark execution is unavailable.",
            )
        )
        return catalog

    def _reconcile_stale_runs(self, *, now: float | None = None) -> None:
        """Truthfully terminalize active rows whose exact worker disappeared."""

        observed_at = time.time() if now is None else float(now)
        stale_run_ids: list[str] = []
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT id, status, created_at, cancel_requested,
                       worker_pid, worker_starttime
                FROM benchmark_runs
                WHERE status IN ('queued', 'waiting_for_compute', 'starting',
                                 'running', 'cancelling')
                """
            ).fetchall()
            for row in rows:
                pid = row["worker_pid"]
                starttime = row["worker_starttime"]
                if isinstance(pid, int) and isinstance(starttime, int):
                    stale = _owned_process_starttime(pid) != starttime
                else:
                    # Every launched active state must eventually bind the
                    # worker's exact PID/start time.  Older schema rows and a
                    # crash between state transitions can otherwise leave a
                    # running/waiting/cancelling row that suppresses Run all
                    # missing forever.  Keep the same grace that protects a
                    # genuinely new queued child while it starts and registers.
                    stale = (
                        observed_at - float(row["created_at"])
                        >= WORKER_START_GRACE_SECONDS
                    )
                if not stale:
                    continue
                stale_run_ids.append(str(row["id"]))
                cancelled = bool(row["cancel_requested"])
                connection.execute(
                    """
                    UPDATE benchmark_runs
                    SET status = ?, finished_at = ?, error_code = ?
                    WHERE id = ? AND status NOT IN ('succeeded', 'failed', 'cancelled')
                    """,
                    (
                        "cancelled" if cancelled else "failed",
                        observed_at,
                        None if cancelled else "worker_lost",
                        row["id"],
                    ),
                )
            connection.execute("COMMIT")
        for stale_run_id in stale_run_ids:
            self._advance_batches_for_run(stale_run_id)
        # This is also the restart/reconciliation kick for a matrix committed
        # before its first launcher invocation.  The claim is a transactional
        # pending -> queued transition, so concurrent service instances cannot
        # launch the same child twice.
        self._launch_next_matrix_child()

    def list_runs(self, *, limit: int = 100) -> dict[str, object]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_RUN_LIST:
            raise ValueError("benchmark run limit is invalid")
        self._reconcile_stale_runs()
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM benchmark_runs ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {"runs": [self._public_run(row, include_cases=False) for row in rows]}

    def get_run(self, run_id: str) -> dict[str, object]:
        normalized = self._run_id(run_id)
        self._reconcile_stale_runs()
        with self._lock, self._connect() as connection:
            row = self._fetch_row(connection, normalized)
        return self._public_run(row, include_cases=True)

    def _worker_command(self, run_id: str) -> list[str]:
        return [
            FLEET_LOW_PRIORITY,
            sys.executable,
            "-m",
            "aeon.benchmarks.worker",
            "--root",
            os.fspath(self.root),
            "--run-id",
            run_id,
        ]

    @staticmethod
    def _insert_normalized_run(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        normalized: Mapping[str, object],
        status_value: str,
        created_at: float,
    ) -> None:
        columns = ["id", "status", "created_at", "summary_json", "cancel_requested"]
        values: list[object] = [run_id, status_value, created_at, "{}", 0]
        for key in (
            "request_id",
            "suite_id",
            "suite_label",
            "suite_version",
            "suite_sha256",
            "harness_id",
            "harness_label",
            "harness_version",
            "model_id",
            "model_label",
            "model_revision",
            "tool_profile_id",
            "tool_profile_label",
            "tool_profile_version",
            "repetitions",
            "request_sha256",
            "catalog_version",
            "catalog_sha256",
            "runner_protocol_version",
            "runner_protocol_sha256",
            "executor_protocol_version",
            "executor_protocol_sha256",
            "runner_source_sha256",
            "harness_source_sha256",
            "tool_source_sha256",
            "combination_sha256",
        ):
            columns.append(key)
            values.append(normalized[key])
        placeholders = ",".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO benchmark_runs ({','.join(columns)}) VALUES ({placeholders})",
            values,
        )

    def submit(self, request: Mapping[str, object]) -> dict[str, object]:
        replay = self._existing_submission_replay(request)
        if replay is not None:
            return replay
        normalized = self._normalize_request(request)
        run_id = f"run-{uuid.uuid4().hex}"
        created_at = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            duplicate = connection.execute(
                "SELECT * FROM benchmark_runs WHERE request_id = ?",
                (normalized["request_id"],),
            ).fetchone()
            if duplicate is not None:
                connection.execute("COMMIT")
                if not hmac.compare_digest(
                    str(duplicate["request_sha256"]),
                    str(normalized["request_sha256"]),
                ):
                    raise ValueError(
                        "benchmark request ID is already bound to a different request"
                    )
                return self._public_run(duplicate, include_cases=False)
            try:
                # Availability is intentionally checked only after the
                # transaction's authoritative duplicate lookup. A retry racing
                # the original commit must replay that row even if the harness
                # becomes unavailable between the two requests.
                if self._launcher is default_launcher:
                    from .executor import runtime_execution_status

                    execution = runtime_execution_status(
                        str(normalized["harness_id"])
                    )
                    if execution.get("supported") is not True:
                        raise BenchmarkExecutionUnavailable(
                            "benchmark execution is unavailable"
                        )
                self._insert_normalized_run(
                    connection,
                    run_id=run_id,
                    normalized=normalized,
                    status_value="queued",
                    created_at=created_at,
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        try:
            self._launcher(self._worker_command(run_id))
        except Exception:
            self._mark_failed(run_id, error_code="launcher_failed")
        return self.get_run(run_id)

    def _existing_submission_replay(
        self, request: Mapping[str, object]
    ) -> dict[str, object] | None:
        """Return an exact old/new idempotent replay before new-suite validation.

        This preserves lost-response retries from clients that submitted a now
        historical partial suite.  It never creates another partial run and it
        compares only the original caller-controlled fields; provenance remains
        whatever was durably bound when that row was created.
        """

        if not isinstance(request, Mapping):
            raise TypeError("benchmark request must be a mapping")
        if set(request) - _SUBMIT_KEYS:
            raise ValueError("benchmark request contains unsupported fields")
        request_id = self._string(request.get("request_id"), field="request ID")
        if not REQUEST_ID_RE.fullmatch(request_id):
            raise ValueError("invalid benchmark request ID")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM benchmark_runs WHERE request_id = ?", (request_id,)
            ).fetchone()
        if row is None:
            return None
        supplied_suite = request.get("suite_id")
        if supplied_suite is None:
            # Omission has always meant the one current benchmark; it must not
            # accidentally replay an old partial run with a reused ID.
            if row["suite_id"] != DEFAULT_SUITE_ID:
                raise ValueError(
                    "benchmark request ID is already bound to a different request"
                )
        elif supplied_suite != row["suite_id"]:
            raise ValueError(
                "benchmark request ID is already bound to a different request"
            )
        supplied_tool = request.get("tool_profile_id")
        comparisons = {
            "harness_id": request.get("harness_id"),
            "model_id": request.get("model_id"),
            "tool_profile_id": (
                row["tool_profile_id"] if supplied_tool is None else supplied_tool
            ),
            "repetitions": request.get("repetitions", 1),
        }
        if any(row[key] != value for key, value in comparisons.items()):
            raise ValueError(
                "benchmark request ID is already bound to a different request"
            )
        return self._public_run(row, include_cases=False)

    @staticmethod
    def _matrix_selector(value: object, *, field: str) -> str:
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or len(value) > 128
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
        ):
            raise ValueError(f"invalid benchmark matrix {field}")
        return value

    def _normalize_matrix_client_request(
        self, request: Mapping[str, object]
    ) -> tuple[dict[str, object], str]:
        if not isinstance(request, Mapping):
            raise TypeError("benchmark matrix request must be a mapping")
        if set(request) - _MATRIX_KEYS:
            raise ValueError("benchmark matrix request contains unsupported fields")
        request_id = self._string(request.get("request_id"), field="matrix request ID")
        if BATCH_REQUEST_ID_RE.fullmatch(request_id) is None:
            raise ValueError("invalid benchmark matrix request ID")
        suite_value = request.get("suite_id")
        if suite_value is not None and suite_value != DEFAULT_SUITE_ID:
            raise ValueError("partial benchmark suites cannot be submitted")
        repetitions = request.get("repetitions", 1)
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or not 1 <= repetitions <= MAX_REPETITIONS
        ):
            raise ValueError("benchmark repetitions must be between 1 and 20")
        missing_only = request.get("missing_only", True)
        if not isinstance(missing_only, bool):
            raise ValueError("benchmark matrix missing_only must be boolean")
        normalized: dict[str, object] = {
            "request_id": request_id,
            "suite_id": DEFAULT_SUITE_ID,
            "harness_id": self._matrix_selector(
                request.get("harness_id", "all"), field="harness"
            ),
            "model_id": self._matrix_selector(
                request.get("model_id", "all"), field="model"
            ),
            "tool_profile_id": self._matrix_selector(
                request.get("tool_profile_id", "all"), field="tool profile"
            ),
            "repetitions": repetitions,
            "missing_only": missing_only,
        }
        return normalized, _sha256(_canonical_json(normalized))

    @staticmethod
    def _batch_id(value: object) -> str:
        if not isinstance(value, str) or BATCH_ID_RE.fullmatch(value) is None:
            raise KeyError("invalid benchmark batch ID")
        return value

    def _current_run_for_combination(
        self,
        connection: sqlite3.Connection,
        combination: Mapping[str, object],
        *,
        repetitions: int,
    ) -> sqlite3.Row | None:
        rows = connection.execute(
            """
            SELECT * FROM benchmark_runs
            WHERE suite_id = ? AND harness_id = ? AND model_id = ?
              AND tool_profile_id = ? AND repetitions = ?
            ORDER BY created_at DESC, id DESC
            """,
            (
                DEFAULT_SUITE_ID,
                combination["harness_id"],
                combination["model_id"],
                combination["tool_profile_id"],
                repetitions,
            ),
        ).fetchall()
        eligible: list[sqlite3.Row] = []
        for row in rows:
            if not self._row_has_current_provenance(row, combination):
                continue
            eligible.append(row)
        active = next(
            (row for row in eligible if str(row["status"]) in ACTIVE_STATUSES),
            None,
        )
        if active is not None:
            return active
        return next(
            (
                row
                for row in eligible
                if row["status"] == "succeeded"
                and self._evidence_payload(row) is not None
            ),
            None,
        )

    @staticmethod
    def _row_has_current_provenance(
        row: Mapping[str, object], combination: Mapping[str, object]
    ) -> bool:
        suite = SUITES[DEFAULT_SUITE_ID]
        return (
            row["suite_id"] == DEFAULT_SUITE_ID
            and row["suite_version"] == suite.version
            and row["suite_sha256"] == suite.sha256
            and row["catalog_version"] == BENCHMARK_CATALOG_VERSION
            and row["catalog_sha256"] == BENCHMARK_CATALOG_SHA256
            and row["runner_protocol_version"] == RUNNER_PROTOCOL_VERSION
            and row["runner_protocol_sha256"] == RUNNER_PROTOCOL_SHA256
            and row["executor_protocol_version"] == EXECUTOR_PROTOCOL_VERSION
            and row["executor_protocol_sha256"] == EXECUTOR_PROTOCOL_SHA256
            and row["runner_source_sha256"] == RUNNER_SOURCE_SHA256
            and row["harness_source_sha256"] == HARNESS_SOURCE_SHA256
            and row["tool_source_sha256"] == TOOL_SOURCE_SHA256
            and row["combination_sha256"] == combination_sha256(combination)
            and all(
                row[key] == value
                for key, value in combination.items()
                if key != "id"
            )
        )
    def _public_batch(self, row: sqlite3.Row) -> dict[str, object]:
        with self._lock, self._connect() as connection:
            mappings = connection.execute(
                """
                SELECT m.ordinal, m.combination_id, m.created_child, r.*
                FROM benchmark_batch_runs AS m
                JOIN benchmark_runs AS r ON r.id = m.run_id
                WHERE m.batch_id = ?
                ORDER BY m.ordinal ASC
                """,
                (row["id"],),
            ).fetchall()
        runs = [self._public_run(item, include_cases=False) for item in mappings]
        statuses = [str(item["status"]) for item in mappings]
        if any(status in ACTIVE_STATUSES for status in statuses):
            status_value = "active"
        elif statuses and all(status == "succeeded" for status in statuses):
            status_value = "succeeded"
        elif statuses and all(status == "cancelled" for status in statuses):
            status_value = "cancelled"
        else:
            status_value = "failed"
        created_run_ids = [
            str(item["id"]) for item in mappings if bool(item["created_child"])
        ]
        run_ids = [str(item["id"]) for item in mappings]
        items = [
            {
                "ordinal": int(item["ordinal"]),
                "combination_id": str(item["combination_id"]),
                "run_id": str(item["id"]),
                "created": bool(item["created_child"]),
                "status": str(item["status"]),
            }
            for item in mappings
        ]
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "id": row["id"],
            "request_id": row["request_id"],
            "status": status_value,
            "created_at": row["created_at"],
            "catalog_version": row["catalog_version"],
            "catalog_sha256": row["catalog_sha256"],
            "missing_only": bool(row["missing_only"]),
            "repetitions": row["repetitions"],
            "harness_selection": row["harness_selection"],
            "model_selection": row["model_selection"],
            "tool_profile_selection": row["tool_profile_selection"],
            "selected_count": row["selected_count"],
            "created_count": row["created_count"],
            "skipped_count": row["selected_count"] - row["created_count"],
            "run_ids": run_ids,
            "created_run_ids": created_run_ids,
            "items": items,
            "runs": runs,
        }

    def get_batch(self, batch_id: str) -> dict[str, object]:
        normalized = self._batch_id(batch_id)
        self._reconcile_stale_runs()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM benchmark_batches WHERE id = ?", (normalized,)
            ).fetchone()
        if row is None:
            raise KeyError(normalized)
        return self._public_batch(row)

    def comparison(self, *, repetitions: int = 1) -> dict[str, object]:
        """Return server-authoritative current coverage for every reviewed row.

        A result is comparable only when every scoring/source identity matches
        the current catalog.  A succeeded row additionally needs intact,
        hash-verified evidence; old or corrupt results remain historical but do
        not suppress ``Run all missing``.
        """

        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or not 1 <= repetitions <= MAX_REPETITIONS
        ):
            raise ValueError("benchmark repetitions must be between 1 and 20")
        self._reconcile_stale_runs()
        runtime_catalog = self.catalog()
        available_harnesses = {
            str(item["id"])
            for item in runtime_catalog["harnesses"]
            if item.get("available") is True
        }
        entries: list[dict[str, object]] = []
        counts = {"succeeded": 0, "active": 0, "failed": 0, "missing": 0}
        with self._lock, self._connect() as connection:
            for combination in valid_combinations():
                row = self._current_run_for_combination(
                    connection, combination, repetitions=repetitions
                )
                state = "missing"
                evidence_verified = False
                if row is not None:
                    status_value = str(row["status"])
                    if status_value in ACTIVE_STATUSES:
                        state = "active"
                    elif status_value == "succeeded":
                        state = "succeeded"
                        evidence_verified = True
                if row is None:
                    # Surface the latest current-provenance terminal failure for
                    # diagnosis without treating it as coverage.
                    candidates = connection.execute(
                        """
                        SELECT * FROM benchmark_runs
                        WHERE suite_id = ? AND harness_id = ? AND model_id = ?
                          AND tool_profile_id = ? AND repetitions = ?
                          AND catalog_sha256 = ? AND suite_sha256 = ?
                          AND runner_protocol_sha256 = ?
                          AND executor_protocol_sha256 = ?
                          AND combination_sha256 = ?
                        ORDER BY created_at DESC, id DESC
                        """,
                        (
                            DEFAULT_SUITE_ID,
                            combination["harness_id"],
                            combination["model_id"],
                            combination["tool_profile_id"],
                            repetitions,
                            BENCHMARK_CATALOG_SHA256,
                            SUITES[DEFAULT_SUITE_ID].sha256,
                            RUNNER_PROTOCOL_SHA256,
                            EXECUTOR_PROTOCOL_SHA256,
                            combination_sha256(combination),
                        ),
                    ).fetchall()
                    failed = next(
                        (
                            candidate
                            for candidate in candidates
                            if str(candidate["status"]) in {"failed", "cancelled"}
                            and self._row_has_current_provenance(
                                candidate, combination
                            )
                        ),
                        None,
                    )
                    if failed is not None:
                        row = failed
                        state = "failed"
                submission_available = (
                    str(combination["harness_id"]) in available_harnesses
                )
                counts[state] += 1
                entries.append(
                    {
                        "combination": dict(combination),
                        "state": state,
                        "submission_available": submission_available,
                        "needs_run": submission_available
                        and state not in {"active", "succeeded"},
                        "evidence_verified": evidence_verified,
                        "run": (
                            self._public_run(row, include_cases=False)
                            if row is not None
                            else None
                        ),
                    }
                )
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "catalog_version": BENCHMARK_CATALOG_VERSION,
            "catalog_sha256": BENCHMARK_CATALOG_SHA256,
            "suite_id": DEFAULT_SUITE_ID,
            "repetitions": repetitions,
            "counts": counts,
            "combinations": entries,
        }

    def _matrix_combinations(
        self, client_request: Mapping[str, object]
    ) -> tuple[dict[str, object], ...]:
        catalog = self.catalog()
        available_harnesses = {
            str(item["id"])
            for item in catalog["harnesses"]
            if item.get("available") is True
        }
        if not available_harnesses:
            raise BenchmarkExecutionUnavailable(
                "benchmark execution is unavailable"
            )
        selector_specs = (
            ("harness_id", "harnesses"),
            ("model_id", "models"),
            ("tool_profile_id", "tool_profiles"),
        )
        filters: dict[str, tuple[str, ...] | None] = {}
        for field, catalog_field in selector_specs:
            selected = str(client_request[field])
            known = {str(item["id"]) for item in catalog[catalog_field]}
            if selected == "all":
                filters[field] = (
                    tuple(sorted(available_harnesses))
                    if field == "harness_id"
                    else None
                )
            elif selected in known:
                if field == "harness_id" and selected not in available_harnesses:
                    raise BenchmarkExecutionUnavailable(
                        "selected benchmark harness is unavailable"
                    )
                filters[field] = (selected,)
            else:
                raise ValueError(f"unknown benchmark matrix {field}")
        combinations = valid_combinations(
            harness_ids=filters["harness_id"],
            model_ids=filters["model_id"],
            tool_profile_ids=filters["tool_profile_id"],
        )
        if not combinations:
            raise ValueError("benchmark matrix selects no reviewed combinations")
        suite = SUITES[DEFAULT_SUITE_ID]
        repetitions = int(client_request["repetitions"])
        if (
            len(combinations) > MAX_MATRIX_COMBINATIONS
            or len(combinations) * len(suite.cases) * repetitions
            > MAX_MATRIX_PLANNED_CASES
        ):
            raise ValueError("benchmark matrix exceeds its bounded workload")
        return combinations

    def submit_matrix(self, request: Mapping[str, object]) -> dict[str, object]:
        # Resolve dead workers before deciding that an active row satisfies
        # missing coverage.  Otherwise the first matrix submitted after a
        # quiet period could bind itself to a stale queued/running row and
        # finish failed instead of creating the replacement it requested.
        self._reconcile_stale_runs()
        client_request, client_sha = self._normalize_matrix_client_request(request)
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM benchmark_batches WHERE request_id = ?",
                (client_request["request_id"],),
            ).fetchone()
        if existing is not None:
            if not hmac.compare_digest(str(existing["request_sha256"]), client_sha):
                raise ValueError(
                    "benchmark matrix request ID is already bound to a different request"
                )
            return self._public_batch(existing)

        repetitions = int(client_request["repetitions"])
        batch_id = f"batch-{uuid.uuid4().hex}"
        created_at = time.time()
        created_count = 0
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            duplicate = connection.execute(
                "SELECT * FROM benchmark_batches WHERE request_id = ?",
                (client_request["request_id"],),
            ).fetchone()
            if duplicate is not None:
                connection.execute("COMMIT")
                if not hmac.compare_digest(str(duplicate["request_sha256"]), client_sha):
                    raise ValueError(
                        "benchmark matrix request ID is already bound to a different request"
                    )
                return self._public_batch(duplicate)
            # As with single submissions, the transaction's duplicate lookup
            # must precede dynamic availability. This makes a concurrent
            # lost-response retry deterministic across an availability change.
            combinations = self._matrix_combinations(client_request)
            connection.execute(
                """
                INSERT INTO benchmark_batches (
                    id, request_id, created_at, request_sha256,
                    catalog_version, catalog_sha256, missing_only, repetitions,
                    harness_selection, model_selection, tool_profile_selection,
                    selected_count, created_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    batch_id,
                    client_request["request_id"],
                    created_at,
                    client_sha,
                    BENCHMARK_CATALOG_VERSION,
                    BENCHMARK_CATALOG_SHA256,
                    int(bool(client_request["missing_only"])),
                    repetitions,
                    client_request["harness_id"],
                    client_request["model_id"],
                    client_request["tool_profile_id"],
                    len(combinations),
                ),
            )
            for ordinal, combination in enumerate(combinations):
                current = (
                    self._current_run_for_combination(
                        connection, combination, repetitions=repetitions
                    )
                    if client_request["missing_only"]
                    else None
                )
                created_child = current is None
                if current is None:
                    child_request_id = "br-" + hashlib.sha256(
                        f"{batch_id}\0{combination['id']}".encode("utf-8")
                    ).hexdigest()[:32]
                    normalized = self._normalize_request(
                        {
                            "request_id": child_request_id,
                            "harness_id": combination["harness_id"],
                            "model_id": combination["model_id"],
                            "tool_profile_id": combination["tool_profile_id"],
                            "repetitions": repetitions,
                        }
                    )
                    run_id = f"run-{uuid.uuid4().hex}"
                    self._insert_normalized_run(
                        connection,
                        run_id=run_id,
                        normalized=normalized,
                        status_value="pending",
                        created_at=created_at + ordinal * 1e-6,
                    )
                    created_count += 1
                else:
                    run_id = str(current["id"])
                connection.execute(
                    """
                    INSERT INTO benchmark_batch_runs (
                        batch_id, ordinal, combination_id, run_id, created_child
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        ordinal,
                        combination["id"],
                        run_id,
                        int(created_child),
                    ),
                )
            connection.execute(
                "UPDATE benchmark_batches SET created_count = ? WHERE id = ?",
                (created_count, batch_id),
            )
            connection.execute("COMMIT")
        self._launch_next_matrix_child()
        return self.get_batch(batch_id)

    def _launch_next_matrix_child(self) -> None:
        """Launch at most one matrix-owned child globally.

        Exact one-off benchmark runs are independent.  Matrix work is globally
        serialized so two tabs or two batches cannot amplify local/Fleet demand.
        """

        run_id: str | None = None
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1
                FROM benchmark_batch_runs AS m
                JOIN benchmark_runs AS r ON r.id = m.run_id
                WHERE m.created_child = 1
                  AND r.status IN ('queued', 'waiting_for_compute', 'starting',
                                   'running', 'cancelling')
                LIMIT 1
                """,
            ).fetchone()
            if active is not None:
                connection.execute("COMMIT")
                return
            pending = connection.execute(
                """
                SELECT r.id
                FROM benchmark_batch_runs AS m
                JOIN benchmark_batches AS b ON b.id = m.batch_id
                JOIN benchmark_runs AS r ON r.id = m.run_id
                WHERE m.created_child = 1 AND r.status = 'pending'
                ORDER BY b.created_at ASC, b.id ASC, m.ordinal ASC
                LIMIT 1
                """
            ).fetchone()
            if pending is not None:
                run_id = str(pending["id"])
                connection.execute(
                    """
                    UPDATE benchmark_runs SET status = 'queued'
                    WHERE id = ? AND status = 'pending'
                    """,
                    (run_id,),
                )
            connection.execute("COMMIT")
        if run_id is None:
            return
        try:
            self._launcher(self._worker_command(run_id))
        except Exception:
            self._mark_failed(run_id, error_code="launcher_failed")

    def _advance_batches_for_run(self, run_id: str) -> None:
        with self._lock, self._connect() as connection:
            owned = connection.execute(
                """
                SELECT 1 FROM benchmark_batch_runs
                WHERE run_id = ? AND created_child = 1 LIMIT 1
                """,
                (self._run_id(run_id),),
            ).fetchone()
        if owned is not None:
            self._launch_next_matrix_child()

    def cancel_batch(self, batch_id: str) -> dict[str, object]:
        """Cancel only children created by one matrix, preserving reused runs."""

        normalized = self._batch_id(batch_id)
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            batch = connection.execute(
                "SELECT * FROM benchmark_batches WHERE id = ?", (normalized,)
            ).fetchone()
            if batch is None:
                connection.execute("ROLLBACK")
                raise KeyError(normalized)
            rows = connection.execute(
                """
                SELECT r.id, r.status
                FROM benchmark_batch_runs AS m
                JOIN benchmark_runs AS r ON r.id = m.run_id
                WHERE m.batch_id = ? AND m.created_child = 1
                """,
                (normalized,),
            ).fetchall()
            for row in rows:
                status_value = str(row["status"])
                if status_value in TERMINAL_STATUSES:
                    continue
                if status_value in {"pending", "queued"}:
                    connection.execute(
                        """
                        UPDATE benchmark_runs
                        SET cancel_requested = 1, status = 'cancelled',
                            finished_at = ?
                        WHERE id = ?
                        """,
                        (now, row["id"]),
                    )
                else:
                    connection.execute(
                        """
                        UPDATE benchmark_runs
                        SET cancel_requested = 1, status = 'cancelling'
                        WHERE id = ?
                        """,
                        (row["id"],),
                    )
            connection.execute("COMMIT")
        self._launch_next_matrix_child()
        return self.get_batch(normalized)

    def cancel(self, run_id: str) -> dict[str, object]:
        normalized = self._run_id(run_id)
        self._reconcile_stale_runs()
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._fetch_row(connection, normalized)
            status_value = str(row["status"])
            if status_value in TERMINAL_STATUSES:
                connection.execute("COMMIT")
                return self._public_run(row, include_cases=True)
            next_status = (
                "cancelled"
                if status_value in {"pending", "queued"}
                else "cancelling"
            )
            finished_at = now if next_status == "cancelled" else None
            connection.execute(
                """
                UPDATE benchmark_runs
                SET cancel_requested = 1, status = ?, finished_at = ?
                WHERE id = ?
                """,
                (next_status, finished_at, normalized),
            )
            row = self._fetch_row(connection, normalized)
            connection.execute("COMMIT")
        result = self._public_run(row, include_cases=True)
        if next_status == "cancelled":
            self._advance_batches_for_run(normalized)
        return result

    # Runner-only methods.  They accept only generated run IDs and fixed states;
    # no caller-provided prompt, endpoint, claim, or diagnostic is persisted.
    def _register_worker(self, run_id: str) -> bool:
        normalized = self._run_id(run_id)
        pid = os.getpid()
        starttime = _owned_process_starttime(pid)
        if starttime is None:
            raise BenchmarkError("benchmark worker identity is unavailable")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._fetch_row(connection, normalized)
            if row["status"] != "queued" or row["cancel_requested"]:
                connection.execute("COMMIT")
                return False
            connection.execute(
                """
                UPDATE benchmark_runs
                SET worker_pid = ?, worker_starttime = ?, worker_registered_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (pid, starttime, time.time(), normalized),
            )
            connection.execute("COMMIT")
        return True

    def _claim_run(self, run_id: str) -> dict[str, object] | None:
        normalized = self._run_id(run_id)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._fetch_row(connection, normalized)
            if row["status"] == "cancelled" or row["cancel_requested"]:
                if row["status"] != "cancelled":
                    connection.execute(
                        "UPDATE benchmark_runs SET status = 'cancelled', finished_at = ? WHERE id = ?",
                        (time.time(), normalized),
                    )
                connection.execute("COMMIT")
                return None
            if row["status"] != "queued":
                connection.execute("COMMIT")
                return None
            connection.execute(
                "UPDATE benchmark_runs SET status = 'running', started_at = ? WHERE id = ?",
                (time.time(), normalized),
            )
            row = self._fetch_row(connection, normalized)
            connection.execute("COMMIT")
        return dict(row)

    def _cancel_requested(self, run_id: str) -> bool:
        normalized = self._run_id(run_id)
        with self._lock, self._connect() as connection:
            row = self._fetch_row(connection, normalized)
        return bool(row["cancel_requested"])

    def _set_active_state(self, run_id: str, status_value: str) -> None:
        """Persist only a sanitized state transition for the active worker.

        Compute ownership remains entirely inside the harness's one Fleet
        service session. This method is observational and cannot create, renew,
        route, or release demand.
        """

        if status_value not in {"running", "waiting_for_compute"}:
            raise BenchmarkError("invalid active benchmark status")
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE benchmark_runs
                SET status = ?
                WHERE id = ?
                  AND status IN ('running', 'waiting_for_compute')
                  AND cancel_requested = 0
                """,
                (status_value, self._run_id(run_id)),
            )

    def _finish_run(
        self,
        run_id: str,
        *,
        status_value: str,
        summary: Mapping[str, object],
        cases: Sequence[Mapping[str, object]],
        error_code: str | None = None,
        honor_cancel_requested: bool = True,
    ) -> dict[str, object]:
        normalized = self._run_id(run_id)
        if status_value not in TERMINAL_STATUSES:
            raise BenchmarkError("invalid terminal benchmark status")
        safe_summary = self._safe_summary(summary)
        with self._lock, self._connect() as connection:
            row = self._fetch_row(connection, normalized)
        safe_cases = self._safe_cases(
            cases,
            suite_id=str(row["suite_id"]),
            repetitions=int(row["repetitions"]),
        )
        evidence = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "run_id": normalized,
            "suite_id": row["suite_id"],
            "suite_version": row["suite_version"],
            "suite_sha256": row["suite_sha256"],
            "runner_protocol_version": row["runner_protocol_version"],
            "runner_protocol_sha256": row["runner_protocol_sha256"],
            "executor_protocol_version": row["executor_protocol_version"],
            "executor_protocol_sha256": row["executor_protocol_sha256"],
            "runner_source_sha256": row["runner_source_sha256"],
            "harness_source_sha256": row["harness_source_sha256"],
            "tool_source_sha256": row["tool_source_sha256"],
            "combination_sha256": row["combination_sha256"],
            "summary": safe_summary,
            "cases": safe_cases,
        }
        run_directory = self.evidence_root / normalized
        _path, evidence_sha256 = _atomic_private_json(
            run_directory, "results.json", evidence
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._fetch_row(connection, normalized)
            final_status = (
                "cancelled"
                if honor_cancel_requested and current["cancel_requested"]
                else status_value
            )
            connection.execute(
                """
                UPDATE benchmark_runs
                SET status = ?, finished_at = ?, summary_json = ?, error_code = ?,
                    evidence_sha256 = ?
                WHERE id = ?
                """,
                (
                    final_status,
                    time.time(),
                    _canonical_json(safe_summary).decode("ascii").strip(),
                    error_code if final_status == "failed" else None,
                    evidence_sha256,
                    normalized,
                ),
            )
            final = self._fetch_row(connection, normalized)
            connection.execute("COMMIT")
        result = self._public_run(final, include_cases=True)
        self._advance_batches_for_run(normalized)
        return result

    def _mark_failed(self, run_id: str, *, error_code: str) -> None:
        safe_code = error_code if error_code in {
            "launcher_failed",
            "executor_unavailable",
            "executor_stuck",
            "harness_unavailable",
            "runner_failed",
            "worker_lost",
        } else "runner_failed"
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE benchmark_runs
                SET status = 'failed', finished_at = ?, error_code = ?
                WHERE id = ? AND status NOT IN ('succeeded', 'failed', 'cancelled')
                """,
                (time.time(), safe_code, self._run_id(run_id)),
            )
        self._advance_batches_for_run(self._run_id(run_id))


__all__ = (
    "ACTIVE_STATUSES",
    "BenchmarkError",
    "BenchmarkExecutionUnavailable",
    "BenchmarkService",
    "FLEET_LOW_PRIORITY",
    "REQUEST_ID_RE",
    "RUN_ID_RE",
    "TERMINAL_STATUSES",
    "default_launcher",
)
