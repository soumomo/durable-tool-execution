import asyncio
import sqlite3
import time
from contextlib import closing, suppress
from typing import Any

import pytest
import pytest_asyncio

from runtime.engine import DurableToolRuntime
from runtime.models import (
    BackpressureError,
    IdempotencyConflictError,
    JobRecord,
    JobStatus,
    RuntimeConfig,
    ToolFatalError,
    ToolRetryableError,
)
from runtime.registry import ToolRegistry
from runtime.store import SQLiteJobStore

pytestmark = pytest.mark.asyncio
WAIT_SECONDS = 5
TOOL_TIMEOUT_MS = 15_000


@pytest_asyncio.fixture
async def runtime_factory(tmp_path):
    runtimes = []

    def make_runtime(registry, **overrides):
        options = {
            "db_path": str(tmp_path / f"runtime-{len(runtimes)}.sqlite3"),
            "max_workers": 2,
            "max_outstanding_jobs": 16,
            "lease_duration_ms": 1000,
            "heartbeat_interval_ms": 50,
            "poll_interval_ms": 10,
            "recovery_interval_ms": 10,
        }
        options.update(overrides)
        runtime = DurableToolRuntime(RuntimeConfig(**options), registry)
        runtimes.append(runtime)
        return runtime

    yield make_runtime

    for runtime in reversed(runtimes):
        # Supervision tests may leave a recorded error for stop() to surface.
        with suppress(RuntimeError):
            await asyncio.wait_for(runtime.stop(graceful=False), WAIT_SECONDS)


async def wait_event(event):
    await asyncio.wait_for(event.wait(), WAIT_SECONDS)


async def terminal(runtime, job_id):
    record = await asyncio.wait_for(runtime.wait_for_terminal(job_id), WAIT_SECONDS)
    assert isinstance(record, JobRecord)
    assert record.status.terminal
    return record


async def test_submit_acknowledges_committed_job_and_event(runtime_factory):
    registry = ToolRegistry()
    entered, release = asyncio.Event(), asyncio.Event()

    @registry.tool(idempotent=True, timeout_ms=TOOL_TIMEOUT_MS)
    async def echo(value):
        entered.set()
        await release.wait()
        return {"echo": value}

    runtime = runtime_factory(registry)
    await runtime.start()
    assert isinstance(runtime.store, SQLiteJobStore)
    assert runtime.fenced_stale_completions == runtime.lost_leases == 0
    try:
        job_id = await runtime.submit("echo", {"value": ["雪", None, False]}, "key")
        assert isinstance(job_id, str) and job_id
        # An independent connection cannot see an uncommitted admission.
        with closing(sqlite3.connect(runtime.store.config.db_path)) as reader:
            assert reader.execute(
                "SELECT job_id, args_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone() == (job_id, '{"value":["雪",null,false]}')
            assert reader.execute(
                "SELECT to_status FROM job_events WHERE job_id = ? ORDER BY event_id",
                (job_id,),
            ).fetchall()[0] == ("PENDING",)
        await wait_event(entered)
    finally:
        release.set()
    result = await terminal(runtime, job_id)
    assert result.status is JobStatus.SUCCEEDED
    assert result.result == {"echo": ["雪", None, False]}
    assert result.attempt_count == 1
    assert await runtime.wait_for_terminal(job_id, timeout=None) == result


async def test_admission_errors_do_not_persist_or_execute_jobs(runtime_factory):
    registry = ToolRegistry()
    calls = []

    @registry.tool(idempotent=True)
    async def echo(value):
        calls.append(value)
        return value

    runtime = runtime_factory(registry)
    await runtime.start()
    with pytest.raises(KeyError):
        await runtime.submit("missing", {}, "unknown")
    for args in ({}, {"value": 1, "extra": 2}):
        with pytest.raises(TypeError):
            await runtime.submit("echo", args, "bad-signature")
    for args in ({"value": float("nan")}, {"value": (1, 2)}):
        with pytest.raises((TypeError, ValueError)):
            await runtime.submit("echo", args, "bad-json")
    with pytest.raises(KeyError):
        await runtime.wait_for_terminal("missing", timeout=1)
    assert await runtime.store.list_jobs() == []
    assert calls == []
    # Rejected submissions must not reserve their idempotency key.
    job_id = await runtime.submit("echo", {"value": 1}, "bad-signature")
    assert (await terminal(runtime, job_id)).result == 1
    assert calls == [1]


async def test_registry_membership_is_snapshotted_at_start_not_construction(
    runtime_factory,
):
    registry = ToolRegistry()
    runtime = runtime_factory(registry)

    @registry.tool(idempotent=True)
    async def before_start():
        return "visible"

    await runtime.start()

    @registry.tool(idempotent=True)
    async def after_start():
        raise AssertionError("late registration must not be executed")

    await runtime.start()
    with pytest.raises(KeyError):
        await runtime.submit("after_start", {}, "late")
    job_id = await runtime.submit("before_start", {}, "early")
    assert (await terminal(runtime, job_id)).result == "visible"
    assert len(await runtime.store.list_jobs()) == 1


@pytest.mark.acceptance
async def test_fixed_worker_bound_backlog_and_duplicate_admission_at_capacity(
    runtime_factory,
):
    registry = ToolRegistry()
    release, saturated = asyncio.Event(), asyncio.Event()
    calls, active, peak = [], 0, 0

    @registry.tool(idempotent=True, timeout_ms=TOOL_TIMEOUT_MS)
    async def gated(value):
        nonlocal active, peak
        calls.append(value)
        active += 1
        peak = max(peak, active)
        if active == 2:
            saturated.set()
        try:
            await release.wait()
            return value
        finally:
            active -= 1

    runtime = runtime_factory(registry, max_workers=2, max_outstanding_jobs=5)
    await runtime.start()
    workers = tuple(runtime._workers)
    assert len(workers) == 2
    try:
        ids = [
            await runtime.submit("gated", {"value": i}, f"key-{i}") for i in range(5)
        ]
        await wait_event(saturated)
        records = await runtime.store.list_jobs()
        assert sum(job.status is JobStatus.LEASED for job in records) == 2
        assert sum(job.status is JobStatus.PENDING for job in records) == 3
        assert await runtime.store.count_outstanding_jobs() == 5
        assert len(calls) == active == peak == 2
        duplicates = await asyncio.gather(
            *(runtime.submit("gated", {"value": 0}, "key-0") for _ in range(12))
        )
        assert duplicates == [ids[0]] * 12
        with pytest.raises(BackpressureError):
            await runtime.submit("gated", {"value": 5}, "overflow")
        with pytest.raises(IdempotencyConflictError):
            await runtime.submit("gated", {"value": 99}, "key-0")
        assert tuple(runtime._workers) == workers
        assert all(not worker.done() for worker in workers)
    finally:
        release.set()
    results = await asyncio.gather(*(terminal(runtime, job_id) for job_id in ids))
    assert all(job.status is JobStatus.SUCCEEDED for job in results)
    assert sorted(calls) == list(range(5))
    assert peak == 2 and active == 0
    assert await runtime.submit("gated", {"value": 0}, "key-0") == ids[0]
    assert len(await runtime.store.get_events(ids[0])) == 3
    extra = await runtime.submit("gated", {"value": 5}, "overflow")
    assert (await terminal(runtime, extra)).result == 5


async def test_canonical_duplicate_requests_execute_once_and_conflicts_stay_conflicts(
    runtime_factory,
):
    registry = ToolRegistry()
    release, entered = asyncio.Event(), asyncio.Event()
    calls = []

    @registry.tool(idempotent=True, timeout_ms=TOOL_TIMEOUT_MS)
    async def echo(value):
        calls.append(value)
        entered.set()
        await release.wait()
        return value

    @registry.tool(idempotent=True)
    async def other(value):
        raise AssertionError("a conflicting request must not execute")

    runtime = runtime_factory(registry)
    await runtime.start()
    left = {"value": {"b": [1, None], "a": "雪"}}
    right = {"value": {"a": "雪", "b": [1, None]}}
    try:
        ids = await asyncio.gather(
            *(runtime.submit("echo", args, "same") for args in [left, right] * 6)
        )
        await wait_event(entered)
        assert len(set(ids)) == 1
        with pytest.raises(IdempotencyConflictError):
            await runtime.submit("other", left, "same")
        with pytest.raises(IdempotencyConflictError):
            await runtime.submit("echo", {"value": None}, "same")
        assert len(await runtime.store.list_jobs()) == 1
    finally:
        release.set()
    assert (await terminal(runtime, ids[0])).result == left["value"]
    assert calls == [left["value"]]
    assert await runtime.submit("echo", right, "same") == ids[0]
    assert (await terminal(runtime, ids[0])).attempt_count == 1


@pytest.mark.parametrize("admission", ["submit", "store"])
async def test_empty_claim_race_and_direct_store_admission_are_eventually_observed(
    runtime_factory, monkeypatch, admission
):
    registry = ToolRegistry()

    @registry.tool(idempotent=True)
    async def echo(value):
        return value

    runtime = runtime_factory(registry, max_workers=1)
    original_claim = runtime.store.claim_next_job
    empty_claim, release_claim = asyncio.Event(), asyncio.Event()
    gated = False

    async def claim(*args, **kwargs):
        nonlocal gated
        job = await original_claim(*args, **kwargs)
        if job is None and not gated:
            gated = True
            empty_claim.set()
            # Admission occurs after the empty DB snapshot but before it returns.
            await release_claim.wait()
        return job

    monkeypatch.setattr(runtime.store, "claim_next_job", claim)
    try:
        await asyncio.wait_for(runtime.start(), WAIT_SECONDS)
        await wait_event(empty_claim)
        if admission == "submit":
            job_id = await runtime.submit("echo", {"value": "wakeup"}, "key")
        else:
            job, _ = await runtime.store.admit_job(
                registry.get("echo").policy, {"value": "poll"}, "key"
            )
            job_id = job.job_id
    finally:
        release_claim.set()
    result = await terminal(runtime, job_id)
    assert result.status is JobStatus.SUCCEEDED
    assert result.attempt_count == 1


@pytest.mark.parametrize("idempotent", [True, False])
@pytest.mark.parametrize(
    "failure,retryable,outcome_unknown",
    [
        ("retryable", True, True),
        pytest.param("timeout", True, True, marks=pytest.mark.acceptance),
        ("self-cancel", True, True),
        ("fatal", False, False),
        ("exception", False, True),
        ("invalid-json", False, True),
    ],
)
async def test_tool_outcome_classification_and_safe_replay(
    runtime_factory, monkeypatch, idempotent, failure, retryable, outcome_unknown
):
    registry = ToolRegistry()
    calls = 0
    failures = []

    @registry.tool(
        idempotent=idempotent,
        timeout_ms=75 if failure == "timeout" else TOOL_TIMEOUT_MS,
        max_retries=1,
        base_backoff_ms=0,
    )
    async def unstable() -> Any:
        nonlocal calls
        calls += 1
        if calls > 1:
            return "replayed safely"
        if failure == "retryable":
            raise ToolRetryableError("transient tool failure")
        if failure == "fatal":
            raise ToolFatalError("definitive tool failure")
        if failure == "exception":
            raise ValueError("unexpected tool failure")
        if failure == "invalid-json":
            return {"not_json": {1, 2}}
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "self-cancel":
            child = asyncio.current_task()
            assert child is not None
            child.cancel()
            await asyncio.sleep(0)
        raise AssertionError("failure path should not return")

    @registry.tool(idempotent=True)
    async def healthy():
        return "worker survived"

    runtime = runtime_factory(registry, max_workers=1)
    original_fail = runtime.store.fail_or_retry_job

    async def record_failure(*args, **kwargs):
        failures.append((kwargs["retryable"], kwargs.get("outcome_unknown", True)))
        return await original_fail(*args, **kwargs)

    monkeypatch.setattr(runtime.store, "fail_or_retry_job", record_failure)
    await runtime.start()
    job_id = await runtime.submit("unstable", {}, "unstable")
    result = await terminal(runtime, job_id)
    replayed = idempotent and retryable
    expected = (
        JobStatus.SUCCEEDED
        if replayed
        else JobStatus.UNKNOWN_OUTCOME
        if not idempotent and outcome_unknown
        else JobStatus.FAILED
    )
    assert result.status is expected
    assert result.attempt_count == calls == (2 if replayed else 1)
    assert failures == [(retryable, outcome_unknown)]
    if replayed:
        assert result.result == "replayed safely"
    else:
        assert result.error_text
        assert result.result_json is None
    events = await runtime.store.get_events(job_id)
    assert [event.to_status for event in events] == (
        [
            JobStatus.PENDING,
            JobStatus.LEASED,
            JobStatus.RETRY_WAIT,
            JobStatus.LEASED,
            JobStatus.SUCCEEDED,
        ]
        if replayed
        else [JobStatus.PENDING, JobStatus.LEASED, expected]
    )
    assert [event.from_status for event in events] == [None] + [
        event.to_status for event in events[:-1]
    ]
    assert [event.attempt for event in events] == (
        [0, 1, 1, 2, 2] if replayed else [0, 1, 1]
    )
    assert all(event.event_type == "STATE_TRANSITION" for event in events)
    assert [event.event_id for event in events] == sorted(
        {event.event_id for event in events}
    )
    healthy_id = await runtime.submit("healthy", {}, "healthy")
    assert (await terminal(runtime, healthy_id)).result == "worker survived"
    assert all(not worker.done() for worker in runtime._workers)


async def test_retry_budget_and_audit_count_attempts_not_retries(runtime_factory):
    registry = ToolRegistry()
    calls = 0

    @registry.tool(idempotent=True, max_retries=2, base_backoff_ms=20)
    async def always_retry():
        nonlocal calls
        calls += 1
        raise ToolRetryableError("still unavailable")

    runtime = runtime_factory(registry)
    await runtime.start()
    job_id = await runtime.submit("always_retry", {}, "key")
    result = await terminal(runtime, job_id)
    assert result.status is JobStatus.FAILED
    assert result.attempt_count == calls == 3
    assert result.error_text is not None
    assert "still unavailable" in result.error_text
    events = await runtime.store.get_events(job_id)
    assert [event.to_status for event in events] == [
        JobStatus.PENDING,
        JobStatus.LEASED,
        JobStatus.RETRY_WAIT,
        JobStatus.LEASED,
        JobStatus.RETRY_WAIT,
        JobStatus.LEASED,
        JobStatus.FAILED,
    ]
    assert [event.attempt for event in events] == [0, 1, 1, 2, 2, 3, 3]
    assert all(event.event_type == "STATE_TRANSITION" for event in events)
    assert [event.event_id for event in events] == sorted(
        {event.event_id for event in events}
    )


@pytest.mark.parametrize("observation", ["timeout", "cancel"])
async def test_waiter_timeout_or_cancellation_does_not_cancel_execution_or_other_waiters(
    runtime_factory, observation
):
    registry = ToolRegistry()
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @registry.tool(idempotent=False, timeout_ms=TOOL_TIMEOUT_MS)
    async def gated():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "completed"

    runtime = runtime_factory(registry)
    await runtime.start()
    job_id = await runtime.submit("gated", {}, "key")
    await wait_event(entered)
    survivor = asyncio.create_task(runtime.wait_for_terminal(job_id))
    observer = None
    try:
        if observation == "timeout":
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    runtime.wait_for_terminal(job_id, timeout=0.02), WAIT_SECONDS
                )
        else:
            observing = asyncio.Event()

            async def observe():
                observing.set()
                return await runtime.wait_for_terminal(job_id)

            observer = asyncio.create_task(observe())
            # observe() enters wait_for_terminal before yielding back to this task.
            await wait_event(observing)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
        assert not cancelled.is_set()
        assert not survivor.done()
        assert (await runtime.store.get_job(job_id)).status is JobStatus.LEASED
        release.set()
        result = await asyncio.wait_for(survivor, WAIT_SECONDS)
        assert result.status is JobStatus.SUCCEEDED
        assert result.result == "completed"
        assert result.attempt_count == 1
        assert not cancelled.is_set()
    finally:
        release.set()
        for task in (survivor, observer):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.wait_for(
            asyncio.gather(
                *(task for task in (survivor, observer) if task is not None),
                return_exceptions=True,
            ),
            WAIT_SECONDS,
        )


@pytest.mark.parametrize("stage", ["execution", "result-persistence"])
async def test_heartbeat_renews_past_original_deadline_through_result_persistence(
    runtime_factory, monkeypatch, stage
):
    registry = ToolRegistry()
    entered, release, renewed_past_deadline = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    deadline = None

    @registry.tool(idempotent=True, timeout_ms=TOOL_TIMEOUT_MS)
    async def slow():
        if stage == "execution":
            entered.set()
            await release.wait()
        return "result"

    runtime = runtime_factory(
        registry, max_workers=1, lease_duration_ms=400, heartbeat_interval_ms=40
    )
    original_heartbeat = runtime.store.heartbeat_lease
    original_complete = runtime.store.complete_job

    async def heartbeat(job_id, attempt_count, **kwargs):
        renewed = await original_heartbeat(job_id, attempt_count, **kwargs)
        if renewed and deadline is not None:
            record = await runtime.store.get_job(job_id)
            if record.updated_at > deadline:
                renewed_past_deadline.set()
        return renewed

    async def complete(*args, **kwargs):
        if stage == "result-persistence":
            entered.set()
            await release.wait()
        return await original_complete(*args, **kwargs)

    monkeypatch.setattr(runtime.store, "heartbeat_lease", heartbeat)
    monkeypatch.setattr(runtime.store, "complete_job", complete)
    await runtime.start()
    try:
        job_id = await runtime.submit("slow", {}, "key")
        await wait_event(entered)
        leased = await runtime.store.get_job(job_id)
        deadline = leased.lease_expires_at
        assert deadline is not None
        await wait_event(renewed_past_deadline)
        current = await runtime.store.get_job(job_id)
        assert time.time() > deadline
        assert current.status is JobStatus.LEASED
        assert current.attempt_count == 1
        assert current.lease_expires_at > deadline
        assert runtime.lost_leases == runtime.fenced_stale_completions == 0
    finally:
        release.set()
    result = await terminal(runtime, job_id)
    assert result.status is JobStatus.SUCCEEDED
    assert result.result == "result"
    assert result.attempt_count == 1


async def test_lost_heartbeat_lease_cancels_child_without_committing_a_tool_failure(
    runtime_factory, monkeypatch
):
    registry = ToolRegistry()
    entered, cancelled = asyncio.Event(), asyncio.Event()
    failures = []

    @registry.tool(idempotent=False, timeout_ms=TOOL_TIMEOUT_MS)
    async def gated():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    @registry.tool(idempotent=True)
    async def healthy():
        return "alive"

    runtime = runtime_factory(
        registry, max_workers=1, lease_duration_ms=300, heartbeat_interval_ms=30
    )
    original_heartbeat = runtime.store.heartbeat_lease
    original_fail = runtime.store.fail_or_retry_job

    async def lose_lease(job_id, attempt_count, **kwargs):
        if entered.is_set() and not cancelled.is_set():
            return False
        return await original_heartbeat(job_id, attempt_count, **kwargs)

    async def record_failure(job_id, attempt_count, error_text, **kwargs):
        failures.append(job_id)
        return await original_fail(job_id, attempt_count, error_text, **kwargs)

    monkeypatch.setattr(runtime.store, "heartbeat_lease", lose_lease)
    monkeypatch.setattr(runtime.store, "fail_or_retry_job", record_failure)
    await runtime.start()
    job_id = await runtime.submit("gated", {}, "key")
    await wait_event(entered)
    await wait_event(cancelled)
    result = await terminal(runtime, job_id)
    assert result.status is JobStatus.UNKNOWN_OUTCOME
    assert result.attempt_count == 1
    assert failures == []
    assert runtime.lost_leases == 1
    healthy_id = await runtime.submit("healthy", {}, "healthy")
    assert (await terminal(runtime, healthy_id)).result == "alive"


async def test_stale_result_is_audited_and_counted_without_killing_worker(
    runtime_factory, monkeypatch
):
    registry = ToolRegistry()
    fenced = asyncio.Event()
    calls = []

    @registry.tool(idempotent=False)
    async def echo(value):
        calls.append(value)
        return value

    runtime = runtime_factory(
        registry, max_workers=1, lease_duration_ms=300, heartbeat_interval_ms=30
    )
    original_complete = runtime.store.complete_job

    async def complete(job_id, attempt_count, result, **kwargs):
        if result == "stale":
            try:
                # Use the real store's fencing path, independent of clock races.
                return await original_complete(
                    job_id, attempt_count + 1, result, **kwargs
                )
            finally:
                fenced.set()
        return await original_complete(job_id, attempt_count, result, **kwargs)

    monkeypatch.setattr(runtime.store, "complete_job", complete)
    await runtime.start()
    stale_id = await runtime.submit("echo", {"value": "stale"}, "stale")
    await wait_event(fenced)
    good_id = await runtime.submit("echo", {"value": "good"}, "good")
    assert (await terminal(runtime, good_id)).status is JobStatus.SUCCEEDED
    result = await terminal(runtime, stale_id)
    assert result.status is JobStatus.UNKNOWN_OUTCOME
    assert result.attempt_count == 1
    assert runtime.fenced_stale_completions == 1
    assert calls == ["stale", "good"]
    events = await runtime.store.get_events(stale_id)
    assert sum(event.event_type == "STALE_COMPLETION_FENCED" for event in events) == 1
    assert all(not worker.done() for worker in runtime._workers)
