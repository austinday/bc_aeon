"""Hermetic tests for the read-only batch concurrency primitive."""

from __future__ import annotations

import threading

import pytest

from aeon.core.bounded_concurrency import (
    CallStatus,
    IndexedCallable,
    MAX_READ_ONLY_WORKERS,
    run_read_only_batch,
)


def test_results_follow_proposal_order_and_never_exceed_four_threads() -> None:
    active = 0
    peak_active = 0
    lock = threading.Lock()
    barrier = threading.Barrier(MAX_READ_ONLY_WORKERS)

    def operation(index: int) -> int:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        barrier.wait(timeout=2)
        with lock:
            active -= 1
        return index * 10

    # Deliberately supply calls out of order. The index is the proposal order.
    proposals = [
        IndexedCallable(index, lambda index=index: operation(index))
        for index in (3, 1, 2, 0)
    ]

    results = run_read_only_batch(proposals, max_workers=4)

    assert [result.proposal_index for result in results] == [0, 1, 2, 3]
    assert [result.value for result in results] == [0, 10, 20, 30]
    assert all(result.status is CallStatus.SUCCEEDED for result in results)
    assert peak_active == MAX_READ_ONLY_WORKERS


def test_callable_exceptions_are_captured_without_aborting_siblings() -> None:
    failure = ValueError("read failed")

    def raise_failure() -> None:
        raise failure

    results = run_read_only_batch(
        [
            IndexedCallable(0, raise_failure),
            IndexedCallable(1, lambda: "still ran"),
        ],
        max_workers=2,
    )

    assert results[0].status is CallStatus.FAILED
    assert results[0].exception is failure
    assert results[0].value is None
    assert results[0].started
    assert results[1].status is CallStatus.SUCCEEDED
    assert results[1].value == "still ran"
    assert results[1].exception is None


def test_cooperative_stop_prevents_later_proposals_from_starting() -> None:
    stop = threading.Event()
    first_wave = threading.Barrier(2)
    started: list[int] = []
    lock = threading.Lock()

    def operation(index: int) -> int:
        with lock:
            started.append(index)
        if index < 2:
            first_wave.wait(timeout=2)
        if index == 1:
            stop.set()
        return index

    proposals = [
        IndexedCallable(index, lambda index=index: operation(index))
        for index in range(7)
    ]

    results = run_read_only_batch(
        proposals,
        max_workers=2,
        should_stop=stop.is_set,
    )

    assert sorted(started) == [0, 1]
    assert [result.status for result in results[:2]] == [
        CallStatus.SUCCEEDED,
        CallStatus.SUCCEEDED,
    ]
    assert all(
        result.status is CallStatus.NOT_STARTED for result in results[2:]
    )
    assert all(not result.started for result in results[2:])


def test_initial_stop_returns_records_without_invoking_any_callable() -> None:
    invoked: list[int] = []
    proposals = [
        IndexedCallable(index, lambda index=index: invoked.append(index))
        for index in range(3)
    ]

    results = run_read_only_batch(proposals, should_stop=lambda: True)

    assert invoked == []
    assert [result.proposal_index for result in results] == [0, 1, 2]
    assert all(result.status is CallStatus.NOT_STARTED for result in results)


def test_inputs_are_not_reordered_or_modified() -> None:
    proposals = [
        IndexedCallable(2, lambda: "two"),
        IndexedCallable(0, lambda: "zero"),
        IndexedCallable(1, lambda: "one"),
    ]
    original_indexes = [item.proposal_index for item in proposals]

    run_read_only_batch(proposals, max_workers=1)

    assert [item.proposal_index for item in proposals] == original_indexes


@pytest.mark.parametrize("workers", [0, 5, -1])
def test_worker_bound_is_enforced(workers: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 4"):
        run_read_only_batch([], max_workers=workers)


def test_duplicate_proposal_indexes_are_rejected() -> None:
    calls = [
        IndexedCallable(1, lambda: "first"),
        IndexedCallable(1, lambda: "duplicate"),
    ]

    with pytest.raises(ValueError, match="indexes must be unique"):
        run_read_only_batch(calls)
