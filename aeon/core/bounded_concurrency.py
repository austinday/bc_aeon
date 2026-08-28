"""Hermetic bounded concurrency for already-classified read-only work.

This module contains no tool policy. Callers must prove that every supplied
callable is read-only before invoking it; mutations must remain serialized by
the owning harness. The helper only provides bounded execution, stable result
ordering, exception capture, and cooperative stop behavior.
"""

from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Iterable, TypeVar


ResultT = TypeVar("ResultT")
MAX_READ_ONLY_WORKERS = 4


class CallStatus(str, Enum):
    """Terminal state of one proposed callable."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NOT_STARTED = "not_started"


@dataclass(frozen=True)
class IndexedCallable(Generic[ResultT]):
    """One callable keyed by its position in the model's proposal."""

    proposal_index: int
    call: Callable[[], ResultT]

    def __post_init__(self) -> None:
        if isinstance(self.proposal_index, bool) or not isinstance(
            self.proposal_index, int
        ):
            raise TypeError("proposal_index must be an integer")
        if self.proposal_index < 0:
            raise ValueError("proposal_index must be non-negative")
        if not callable(self.call):
            raise TypeError("call must be callable")


@dataclass(frozen=True)
class IndexedCallResult(Generic[ResultT]):
    """Captured outcome for one proposal, including failures as values."""

    proposal_index: int
    status: CallStatus
    value: ResultT | None = None
    exception: Exception | None = None

    @property
    def started(self) -> bool:
        return self.status is not CallStatus.NOT_STARTED

    @property
    def succeeded(self) -> bool:
        return self.status is CallStatus.SUCCEEDED


def _invoke(
    item: IndexedCallable[ResultT],
    *,
    should_stop: Callable[[], bool],
    start_gate: threading.Lock,
) -> IndexedCallResult[ResultT]:
    # Serialize only the final start decision, never the supplied work. This
    # prevents a queued callable from crossing a cooperative-stop boundary.
    with start_gate:
        if should_stop():
            return IndexedCallResult(
                proposal_index=item.proposal_index,
                status=CallStatus.NOT_STARTED,
            )
    try:
        return IndexedCallResult(
            proposal_index=item.proposal_index,
            status=CallStatus.SUCCEEDED,
            value=item.call(),
        )
    except Exception as exc:
        return IndexedCallResult(
            proposal_index=item.proposal_index,
            status=CallStatus.FAILED,
            exception=exc,
        )


def run_read_only_batch(
    calls: Iterable[IndexedCallable[ResultT]],
    *,
    max_workers: int = MAX_READ_ONLY_WORKERS,
    should_stop: Callable[[], bool] | None = None,
) -> list[IndexedCallResult[ResultT]]:
    """Run pre-classified read-only callables with at most four threads.

    Results are returned by ascending proposal index, independently of
    completion order. Exceptions raised by supplied callables are captured in
    result records. Once the cooperative callback reports true, no additional
    supplied callable is started; already-running work is allowed to finish.

    The stop callback is part of the harness control plane and must not raise.
    """

    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError("max_workers must be an integer")
    if not 1 <= max_workers <= MAX_READ_ONLY_WORKERS:
        raise ValueError(
            f"max_workers must be between 1 and {MAX_READ_ONLY_WORKERS}"
        )
    stop_requested = should_stop or (lambda: False)
    if not callable(stop_requested):
        raise TypeError("should_stop must be callable")

    ordered = sorted(list(calls), key=lambda item: item.proposal_index)
    indexes = [item.proposal_index for item in ordered]
    if len(indexes) != len(set(indexes)):
        raise ValueError("proposal indexes must be unique")
    if not ordered:
        return []

    results: dict[int, IndexedCallResult[ResultT]] = {}
    pending: dict[Future[IndexedCallResult[ResultT]], IndexedCallable[ResultT]] = {}
    cursor = 0
    start_gate = threading.Lock()

    def submit_available(executor: ThreadPoolExecutor) -> None:
        nonlocal cursor
        while cursor < len(ordered) and len(pending) < max_workers:
            if stop_requested():
                return
            item = ordered[cursor]
            cursor += 1
            future = executor.submit(
                _invoke,
                item,
                should_stop=stop_requested,
                start_gate=start_gate,
            )
            pending[future] = item

    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="aeon-read"
    ) as executor:
        submit_available(executor)
        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                item = pending.pop(future)
                # _invoke captures task exceptions. A failure here is therefore
                # a control-plane programming error and should remain visible.
                result = future.result()
                results[item.proposal_index] = result
            submit_available(executor)

    # Calls never submitted because of a stop request receive explicit records.
    for item in ordered[cursor:]:
        results[item.proposal_index] = IndexedCallResult(
            proposal_index=item.proposal_index,
            status=CallStatus.NOT_STARTED,
        )

    return [results[item.proposal_index] for item in ordered]


__all__ = [
    "CallStatus",
    "IndexedCallable",
    "IndexedCallResult",
    "MAX_READ_ONLY_WORKERS",
    "run_read_only_batch",
]
