import asyncio
import time
from contextlib import suppress

import pytest
import pytest_asyncio

from runtime.engine import DurableToolRuntime
from runtime.models import JobStatus, RuntimeConfig, ToolPolicy, ToolRetryableError
from runtime.registry import ToolRegistry
from runtime.store import SQLiteJobStore

pytestmark = pytest.mark.asyncio
WAIT_SECONDS = 5
TOOL_TIMEOUT_MS = 15_000


@pytest_asyncio.fixture
async def runtime_factory(tmp_path):
    runtimes = []

    def make_runtime(registry, *, config=None, **overrides):
        if config is None:
            options = {
                "db_path": str(tmp_path / f"runtime-{len(runtimes)}.sqlite3"),
                "max_workers": 1,
                "max_outstanding_jobs": 16,
                "lease_duration_ms": 1000,
                "heartbeat_interval_ms": 50,
                "poll_interval_ms": 10,
                "recovery_interval_ms": 10,
            }
            options.update(overrides)
            config = RuntimeConfig(**options)
        runtime = DurableToolRuntime(config, registry)
        runtimes.append(runtime)
        return runtime

    yield make_runtime

    for runtime in reversed(runtimes):
        with suppress(RuntimeError):
            await asyncio.wait_for(runtime.stop(graceful=False), WAIT_SECONDS)


async def wait_event(event):
    await asyncio.wait_for(event.wait(), WAIT_SECONDS)


async def terminal(runtime, job_id):
    return await asyncio.wait_for(runtime.wait_for_terminal(job_id), WAIT_SECONDS)


async def settle(*tasks):
    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=WAIT_SECONDS)
        for task in done:
            if not task.cancelled():
                task.exception()
        assert not pending, "test left a task blocked after releasing its gates"


async def assert_closed(runtime):
    with pytest.raises(RuntimeError):
        await runtime.store.list_jobs()
    assert all(worker.done() for worker in runtime._workers)
    assert runtime._reaper is None or runtime._reaper.done()


async def wait_until_rejecting(runtime, tool_name, args, key):
    # Probe with an existing key: observing shutdown must not admit extra work.
    async with asyncio.timeout(WAIT_SECONDS):
        while True:
            try:
                await runtime.submit(tool_name, args, key)
            except RuntimeError:
                return
            await asyncio.sleep(0)


async def seed_job(config, policy, status, *, args=None, now=None):
    when = time.time() - 10 if now is None else now
    async with SQLiteJobStore(config, random_source=lambda: 1.0) as store:
        job, _ = await store.admit_job(
            policy, {} if args is None else args, "persisted", now=when
        )
        if status is not JobStatus.PENDING:
            job = await store.claim_next_job("previous-process", now=when)
            assert job is not None
        if status is JobStatus.SUCCEEDED:
            job = await store.complete_job(job.job_id, 1, "historical", now=when + 0.01)
        elif status in {
            JobStatus.RETRY_WAIT,
            JobStatus.FAILED,
            JobStatus.UNKNOWN_OUTCOME,
        }:
            job = await store.fail_or_retry_job(
                job.job_id,
                1,
                "previous failure",
                retryable=status is JobStatus.RETRY_WAIT,
                outcome_unknown=status is not JobStatus.FAILED,
                now=when + 0.01,
            )
        assert job.status is status
        return job


async def test_start_and_stop_are_idempotent_but_closed_runtime_is_one_shot(
    runtime_factory,
):
    registry = ToolRegistry()

    @registry.tool(idempotent=True)
    async def healthy():
        return "ok"

    runtime = runtime_factory(registry, max_workers=2)
    with pytest.raises(RuntimeError):
        await runtime.submit("healthy", {}, "before-start")
    with pytest.raises(RuntimeError):
        await runtime.wait_for_terminal("missing")
    await runtime.start()
    workers, reaper = tuple(runtime._workers), runtime._reaper
    assert len(workers) == 2
    assert reaper is not None
    await runtime.start()
    assert tuple(runtime._workers) == workers
    assert runtime._reaper is reaper
    job_id = await runtime.submit("healthy", {}, "running")
    assert (await terminal(runtime, job_id)).status is JobStatus.SUCCEEDED
    await asyncio.wait_for(runtime.stop(), WAIT_SECONDS)
    await runtime.stop()
    await runtime.stop(graceful=False)
    await assert_closed(runtime)
    assert all(worker.done() for worker in workers)
    assert reaper.done()
    with pytest.raises(RuntimeError):
        await runtime.start()
    with pytest.raises(RuntimeError):
        await runtime.submit("healthy", {}, "after-stop")
    with pytest.raises(RuntimeError):
        await runtime.wait_for_terminal(job_id)


async def test_graceful_stop_rejects_admission_and_drains_active_pending_and_retry_jobs(
    runtime_factory, monkeypatch
):
    registry = ToolRegistry()
    active, release = asyncio.Event(), asyncio.Event()
    retry_entered, allow_failure, retry_written = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    calls = {"retry": 0, "active": 0, "pending": 0}

    @registry.tool(
        idempotent=True, timeout_ms=TOOL_TIMEOUT_MS, max_retries=1, base_backoff_ms=100
    )
    async def work(kind):
        calls[kind] += 1
        if kind == "retry" and calls[kind] == 1:
            retry_entered.set()
            await allow_failure.wait()
            raise ToolRetryableError("retry during drain")
        if kind == "active":
            active.set()
            await release.wait()
        return kind

    runtime = runtime_factory(registry)
    # Queue the active job before the retry is scheduled, so its eligibility is
    # earlier regardless of scheduler delays. Its gate then holds RETRY_WAIT.
    monkeypatch.setattr(runtime.store, "_random", lambda: 1.0)
    original_fail = runtime.store.fail_or_retry_job

    async def fail(*args, **kwargs):
        job = await original_fail(*args, **kwargs)
        if job.status is JobStatus.RETRY_WAIT:
            retry_written.set()
        return job

    monkeypatch.setattr(runtime.store, "fail_or_retry_job", fail)
    await runtime.start()
    stopping = None
    try:
        retry_id = await runtime.submit("work", {"kind": "retry"}, "retry")
        await wait_event(retry_entered)
        active_id = await runtime.submit("work", {"kind": "active"}, "active")
        pending_id = await runtime.submit("work", {"kind": "pending"}, "pending")
        allow_failure.set()
        await wait_event(retry_written)
        await wait_event(active)
        assert (await runtime.store.get_job(retry_id)).status is JobStatus.RETRY_WAIT
        assert (await runtime.store.get_job(pending_id)).status is JobStatus.PENDING
        stopping = asyncio.create_task(runtime.stop())
        await wait_until_rejecting(runtime, "work", {"kind": "active"}, "active")
        assert not stopping.done()
        assert await runtime.store.count_outstanding_jobs() == 3
        release.set()
        await asyncio.wait_for(asyncio.shield(stopping), WAIT_SECONDS)
    finally:
        allow_failure.set()
        release.set()
        if stopping is not None:
            await settle(stopping)
    await assert_closed(runtime)
    async with SQLiteJobStore(runtime.store.config) as reader:
        records = [
            await reader.get_job(job_id) for job_id in (retry_id, active_id, pending_id)
        ]
        assert all(job.status is JobStatus.SUCCEEDED for job in records)
        assert [job.attempt_count for job in records] == [2, 1, 1]
    assert calls == {"retry": 2, "active": 1, "pending": 1}


async def test_normal_context_exit_starts_runtime_and_waits_for_accepted_work(
    runtime_factory,
):
    registry = ToolRegistry()
    entered, release, body_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    job_ids = []

    @registry.tool(idempotent=True, timeout_ms=TOOL_TIMEOUT_MS)
    async def gated():
        entered.set()
        await release.wait()
        return "drained"

    runtime = runtime_factory(registry)

    async def owner():
        async with runtime as opened:
            assert opened is runtime
            job_ids.append(await runtime.submit("gated", {}, "key"))
            await wait_event(entered)
            body_finished.set()

    task = asyncio.create_task(owner())
    try:
        await wait_event(body_finished)
        await wait_until_rejecting(runtime, "gated", {}, "key")
        assert not task.done()
        release.set()
        await asyncio.wait_for(asyncio.shield(task), WAIT_SECONDS)
    finally:
        release.set()
        await settle(task)
    await assert_closed(runtime)
    async with SQLiteJobStore(runtime.store.config) as reader:
        assert (await reader.get_job(job_ids[0])).result == "drained"


@pytest.mark.parametrize("shutdown", ["immediate", "body-exception"])
@pytest.mark.parametrize("idempotent", [True, False])
async def test_immediate_shutdown_preserves_lease_for_restart_without_unsafe_replay(
    runtime_factory, shutdown, idempotent
):
    registry = ToolRegistry()
    entered, cancelled = asyncio.Event(), asyncio.Event()
    calls = 0

    @registry.tool(
        idempotent=idempotent,
        timeout_ms=TOOL_TIMEOUT_MS,
        max_retries=1,
        base_backoff_ms=0,
    )
    async def interrupted():
        nonlocal calls
        calls += 1
        if calls > 1:
            return "recovered"
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    runtime = runtime_factory(registry, lease_duration_ms=400, heartbeat_interval_ms=40)

    class BodyError(Exception):
        pass

    job_ids = []
    if shutdown == "body-exception":
        error = BodyError("body failed")

        async def failing_body():
            async with runtime:
                job_ids.append(await runtime.submit("interrupted", {}, "key"))
                await wait_event(entered)
                raise error

        owner = asyncio.create_task(failing_body())
        try:
            with pytest.raises(BodyError) as raised:
                await asyncio.wait_for(asyncio.shield(owner), WAIT_SECONDS)
            assert raised.value is error
        finally:
            if not owner.done():
                owner.cancel()
            await settle(owner)
    else:
        await asyncio.wait_for(runtime.start(), WAIT_SECONDS)
        job_ids.append(await runtime.submit("interrupted", {}, "key"))
        await wait_event(entered)
        await asyncio.wait_for(runtime.stop(graceful=False), WAIT_SECONDS)
    job_id = job_ids[0]
    assert cancelled.is_set()
    await assert_closed(runtime)
    async with SQLiteJobStore(runtime.store.config) as reader:
        abandoned = await reader.get_job(job_id)
        assert abandoned.status is JobStatus.LEASED
        assert abandoned.attempt_count == 1
        assert [event.to_status for event in await reader.get_events(job_id)] == [
            JobStatus.PENDING,
            JobStatus.LEASED,
        ]
    restarted = runtime_factory(registry, config=runtime.store.config)
    await restarted.start()
    result = await terminal(restarted, job_id)
    assert result.status is (
        JobStatus.SUCCEEDED if idempotent else JobStatus.UNKNOWN_OUTCOME
    )
    assert result.attempt_count == calls == (2 if idempotent else 1)


@pytest.mark.parametrize(
    "status,idempotent",
    [
        pytest.param(JobStatus.PENDING, True, marks=pytest.mark.acceptance),
        pytest.param(JobStatus.PENDING, False, marks=pytest.mark.acceptance),
        pytest.param(JobStatus.LEASED, True, marks=pytest.mark.acceptance),
        pytest.param(JobStatus.LEASED, False, marks=pytest.mark.acceptance),
        (JobStatus.RETRY_WAIT, True),
    ],
)
async def test_restart_runs_persisted_work_and_recovers_expired_leases(
    runtime_factory, status, idempotent
):
    registry = ToolRegistry()
    calls = []

    @registry.tool(idempotent=idempotent, max_retries=2, base_backoff_ms=0)
    async def work(value):
        calls.append(value)
        return value

    runtime = runtime_factory(registry)
    seeded = await seed_job(
        runtime.store.config,
        registry.get("work").policy,
        status,
        args={"value": "saved"},
    )
    await runtime.start()
    result = await terminal(runtime, seeded.job_id)
    unsafe_expiry = status is JobStatus.LEASED and not idempotent
    assert result.status is (
        JobStatus.UNKNOWN_OUTCOME if unsafe_expiry else JobStatus.SUCCEEDED
    )
    assert calls == ([] if unsafe_expiry else ["saved"])
    assert result.result == (None if unsafe_expiry else "saved")
    assert result.attempt_count == (
        1 if status is JobStatus.PENDING or unsafe_expiry else 2
    )
    if status is JobStatus.LEASED:
        events = await runtime.store.get_events(seeded.job_id)
        assert events[2].from_status is JobStatus.LEASED
        assert events[2].to_status is (
            JobStatus.UNKNOWN_OUTCOME if unsafe_expiry else JobStatus.RETRY_WAIT
        )


async def test_startup_recovery_finishes_before_any_worker_claims(
    runtime_factory, monkeypatch
):
    registry = ToolRegistry()

    @registry.tool(idempotent=True, max_retries=1, base_backoff_ms=0)
    async def work():
        return "recovered"

    runtime = runtime_factory(registry)
    seeded = await seed_job(
        runtime.store.config, registry.get("work").policy, JobStatus.LEASED
    )
    recovery_entered, release_recovery = asyncio.Event(), asyncio.Event()
    claims = []
    recovered = False
    original_recover = runtime.store.recover_expired_leases
    original_claim = runtime.store.claim_next_job

    async def recover(*args, **kwargs):
        nonlocal recovered
        if not recovered:
            recovery_entered.set()
            await release_recovery.wait()
        jobs = await original_recover(*args, **kwargs)
        recovered = True
        return jobs

    async def claim(*args, **kwargs):
        claims.append(recovered)
        assert recovered, "worker claimed before startup recovery finished"
        return await original_claim(*args, **kwargs)

    monkeypatch.setattr(runtime.store, "recover_expired_leases", recover)
    monkeypatch.setattr(runtime.store, "claim_next_job", claim)
    starting = asyncio.create_task(runtime.start())
    try:
        await wait_event(recovery_entered)
        assert not starting.done()
        assert claims == []
        release_recovery.set()
        await asyncio.wait_for(asyncio.shield(starting), WAIT_SECONDS)
        result = await terminal(runtime, seeded.job_id)
        assert result.status is JobStatus.SUCCEEDED
        assert result.attempt_count == 2
        assert claims and all(claims)
    finally:
        release_recovery.set()
        await settle(starting)


async def test_periodic_recovery_handles_a_lease_that_was_not_expired_at_startup(
    runtime_factory, monkeypatch
):
    registry = ToolRegistry()

    @registry.tool(idempotent=False)
    async def external_effect():
        raise AssertionError("an abandoned non-idempotent job must never replay")

    runtime = runtime_factory(registry, lease_duration_ms=400, heartbeat_interval_ms=40)
    seeded = await seed_job(
        runtime.store.config,
        registry.get("external_effect").policy,
        JobStatus.LEASED,
        now=time.time(),
    )
    original_recover = runtime.store.recover_expired_leases
    periodic_entered, release_periodic = asyncio.Event(), asyncio.Event()
    recovery_calls = 0

    async def recover(**kwargs):
        nonlocal recovery_calls
        recovery_calls += 1
        if recovery_calls == 1:
            # Fix the startup observation to the seed's live-lease time. Periodic
            # recovery below uses wall time, even if the test runner was paused.
            return await original_recover(now=seeded.started_at)
        periodic_entered.set()
        await release_periodic.wait()
        return await original_recover(**kwargs)

    monkeypatch.setattr(runtime.store, "recover_expired_leases", recover)
    try:
        await runtime.start()
        await wait_event(periodic_entered)
        assert (await runtime.store.get_job(seeded.job_id)).status is JobStatus.LEASED
        release_periodic.set()
        result = await terminal(runtime, seeded.job_id)
        assert result.status is JobStatus.UNKNOWN_OUTCOME
        assert result.attempt_count == 1
        assert recovery_calls >= 2
    finally:
        release_periodic.set()


@pytest.mark.parametrize(
    "status", [JobStatus.PENDING, JobStatus.LEASED, JobStatus.RETRY_WAIT]
)
@pytest.mark.parametrize("mismatch", ["missing", "idempotency", "arguments"])
async def test_incompatible_unfinished_jobs_fail_startup_and_close_store(
    runtime_factory, monkeypatch, status, mismatch
):
    registry = ToolRegistry()
    calls = []

    async def current_tool(value):
        calls.append(value)
        return value

    async def changed_signature(required):
        calls.append(required)
        return required

    persisted_policy = ToolPolicy("saved", True, max_retries=2, base_backoff_ms=0)
    if mismatch != "missing":
        registry.register(
            ToolPolicy("saved", mismatch != "idempotency"),
            changed_signature if mismatch == "arguments" else current_tool,
        )
    runtime = runtime_factory(registry)
    await seed_job(runtime.store.config, persisted_policy, status, args={"value": 1})
    recoveries = []
    original_recover = runtime.store.recover_expired_leases

    async def recover(*args, **kwargs):
        recoveries.append(True)
        return await original_recover(*args, **kwargs)

    monkeypatch.setattr(runtime.store, "recover_expired_leases", recover)
    with pytest.raises(KeyError if mismatch == "missing" else ValueError):
        await asyncio.wait_for(runtime.start(), WAIT_SECONDS)
    assert recoveries == []
    assert calls == []
    await assert_closed(runtime)
    with pytest.raises(RuntimeError):
        await runtime.start()


@pytest.mark.parametrize(
    "status", [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.UNKNOWN_OUTCOME]
)
async def test_startup_does_not_require_registration_for_terminal_history(
    runtime_factory, status
):
    runtime = runtime_factory(ToolRegistry())
    policy = ToolPolicy("removed-tool", status is not JobStatus.UNKNOWN_OUTCOME)
    seeded = await seed_job(runtime.store.config, policy, status)
    await runtime.start()
    assert await terminal(runtime, seeded.job_id) == seeded
    assert len(await runtime.store.list_jobs()) == 1


async def test_persisted_timeout_retry_budget_and_backoff_override_registry_defaults(
    runtime_factory,
):
    registry = ToolRegistry()
    calls = 0

    @registry.tool(
        name="saved",
        idempotent=True,
        timeout_ms=TOOL_TIMEOUT_MS,
        max_retries=0,
        base_backoff_ms=10_000,
    )
    async def current_tool():
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.Event().wait()
        return "used frozen policy"

    runtime = runtime_factory(registry)
    persisted_policy = ToolPolicy(
        "saved", True, timeout_ms=75, max_retries=1, base_backoff_ms=0
    )
    seeded = await seed_job(runtime.store.config, persisted_policy, JobStatus.PENDING)
    await runtime.start()
    result = await terminal(runtime, seeded.job_id)
    assert result.status is JobStatus.SUCCEEDED
    assert result.result == "used frozen policy"
    assert result.attempt_count == calls == 2
    assert result.policy == persisted_policy
    events = await runtime.store.get_events(seeded.job_id)
    retry_event = next(
        event for event in events if event.to_status is JobStatus.RETRY_WAIT
    )
    assert result.next_eligible_at == retry_event.timestamp


async def repeat_cancellation_while_gated(task):
    for _ in range(3):
        assert task.cancel()
        # Advance ready tasks, not wall time, while cleanup is held at a gate.
        for _ in range(3):
            await asyncio.sleep(0)
        assert not task.done(), "cancellation escaped before cleanup finished"


async def test_cancelled_graceful_stop_switches_to_immediate_and_finishes_cleanup(
    runtime_factory, monkeypatch
):
    registry = ToolRegistry()
    tool_entered, child_cleanup, release_child = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    child_finished, close_entered, release_close = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    close_finished = asyncio.Event()

    @registry.tool(idempotent=False, timeout_ms=TOOL_TIMEOUT_MS)
    async def gated():
        tool_entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            child_cleanup.set()
            await release_child.wait()
            child_finished.set()

    runtime = runtime_factory(registry)
    original_close = runtime.store.close

    async def close():
        close_entered.set()
        await release_close.wait()
        await original_close()
        close_finished.set()

    monkeypatch.setattr(runtime.store, "close", close)
    await runtime.start()
    job_id = await runtime.submit("gated", {}, "key")
    await wait_event(tool_entered)
    stopping = asyncio.create_task(runtime.stop())
    try:
        await wait_until_rejecting(runtime, "gated", {}, "key")
        assert stopping.cancel()
        await wait_event(child_cleanup)
        await repeat_cancellation_while_gated(stopping)
        assert not child_finished.is_set()
        assert not close_finished.is_set()
        release_child.set()
        await wait_event(close_entered)
        await repeat_cancellation_while_gated(stopping)
        assert child_finished.is_set()
        assert not close_finished.is_set()
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(stopping), WAIT_SECONDS)
        assert close_finished.is_set()
        await assert_closed(runtime)
        async with SQLiteJobStore(runtime.store.config) as reader:
            assert (await reader.get_job(job_id)).status is JobStatus.LEASED
    finally:
        release_child.set()
        release_close.set()
        if not stopping.done():
            stopping.cancel()
        await settle(stopping)


async def test_cancelled_start_closes_open_store_despite_repeated_cancellation(
    runtime_factory, monkeypatch
):
    runtime = runtime_factory(ToolRegistry())
    opened, release_open, close_entered, release_close = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    close_finished = asyncio.Event()
    original_start, original_close = runtime.store.start, runtime.store.close

    async def start_store():
        await original_start()
        opened.set()
        await release_open.wait()

    async def close_store():
        close_entered.set()
        await release_close.wait()
        await original_close()
        close_finished.set()

    monkeypatch.setattr(runtime.store, "start", start_store)
    monkeypatch.setattr(runtime.store, "close", close_store)
    starting = asyncio.create_task(runtime.start())
    try:
        await wait_event(opened)
        assert starting.cancel()
        await wait_event(close_entered)
        await repeat_cancellation_while_gated(starting)
        assert not close_finished.is_set()
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(starting), WAIT_SECONDS)
        assert close_finished.is_set()
        await assert_closed(runtime)
        with pytest.raises(RuntimeError):
            await runtime.start()
    finally:
        release_open.set()
        release_close.set()
        if not starting.done():
            starting.cancel()
        await settle(starting)


@pytest.mark.parametrize(
    "fault_method",
    [
        "claim_next_job",
        "recover_expired_leases",
        "complete_job",
        "fail_or_retry_job",
        "heartbeat_lease",
    ],
)
async def test_background_infrastructure_failure_is_supervised_not_a_tool_outcome(
    runtime_factory, monkeypatch, fault_method
):
    registry = ToolRegistry()
    release_tool, fault_entered, release_fault = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    armed = False

    @registry.tool(idempotent=True, timeout_ms=TOOL_TIMEOUT_MS, base_backoff_ms=0)
    async def work():
        if fault_method == "complete_job":
            return "result whose commit fails"
        if fault_method == "fail_or_retry_job":
            raise ToolRetryableError("real tool failure, followed by a DB failure")
        await release_tool.wait()
        return "released"

    # A long lease keeps recovery from muddying the injected persistence failure.
    runtime = runtime_factory(
        registry, lease_duration_ms=15_000, heartbeat_interval_ms=20
    )
    original = getattr(runtime.store, fault_method)

    async def broken(*args, **kwargs):
        if not armed:
            return await original(*args, **kwargs)
        fault_entered.set()
        await release_fault.wait()
        raise OSError(f"injected infrastructure failure: {fault_method}")

    monkeypatch.setattr(runtime.store, fault_method, broken)
    await runtime.start()
    armed = True
    try:
        job_id = await runtime.submit("work", {}, "accepted")
        await wait_event(fault_entered)
        release_fault.set()
        # Observe supervision through the public API rather than an engine-private
        # error field or assumptions about when a background task exits.
        async with asyncio.timeout(WAIT_SECONDS):
            while True:
                try:
                    await runtime.wait_for_terminal(job_id, timeout=0.01)
                except TimeoutError:
                    pass
                except RuntimeError:
                    break
                await asyncio.sleep(0)
        with pytest.raises(RuntimeError):
            await runtime.submit("work", {}, "after-failure")
        with pytest.raises(RuntimeError):
            await runtime.wait_for_terminal(job_id)
        if fault_method in {"complete_job", "fail_or_retry_job", "heartbeat_lease"}:
            assert (await runtime.store.get_job(job_id)).status is JobStatus.LEASED
            assert not any(
                event.to_status.terminal
                for event in await runtime.store.get_events(job_id)
            )
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(runtime.stop(graceful=False), WAIT_SECONDS)
        await assert_closed(runtime)
    finally:
        release_fault.set()
        release_tool.set()
