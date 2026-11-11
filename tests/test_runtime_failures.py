import asyncio
from contextlib import suppress

import pytest
import pytest_asyncio

from runtime.engine import DurableToolRuntime
from runtime.models import JobStatus, JSONValue, RuntimeConfig
from runtime.registry import ToolRegistry

pytestmark = pytest.mark.asyncio
WAIT_SECONDS = 5


@pytest_asyncio.fixture
async def runtime_factory(tmp_path):
    runtimes = []

    def make_runtime(registry, **overrides):
        options = {
            "db_path": str(tmp_path / f"failures-{len(runtimes)}.sqlite3"),
            "max_workers": 2,
            "max_outstanding_jobs": 8,
            "lease_duration_ms": 15_000,
            "heartbeat_interval_ms": 20,
            "poll_interval_ms": 5,
            "recovery_interval_ms": 10,
        }
        options.update(overrides)
        runtime = DurableToolRuntime(RuntimeConfig(**options), registry)
        runtimes.append(runtime)
        return runtime

    yield make_runtime

    for runtime in reversed(runtimes):
        with suppress(RuntimeError):
            await asyncio.wait_for(runtime.stop(graceful=False), WAIT_SECONDS)


async def wait_event(event):
    await asyncio.wait_for(event.wait(), WAIT_SECONDS)


async def loop_turn():
    ready = asyncio.Event()
    asyncio.get_running_loop().call_soon(ready.set)
    await wait_event(ready)


async def assert_failure(awaitable, cause):
    with pytest.raises(RuntimeError) as raised:
        await asyncio.wait_for(awaitable, WAIT_SECONDS)
    assert raised.value.__cause__ is cause


async def settle(*tasks):
    if not tasks:
        return
    done, pending = await asyncio.wait(tasks, timeout=WAIT_SECONDS)
    for task in done:
        if not task.cancelled():
            task.exception()
    for task in pending:
        task.cancel()
    if pending:
        finished, pending = await asyncio.wait(pending, timeout=WAIT_SECONDS)
        for task in finished:
            if not task.cancelled():
                task.exception()
    assert not pending, "test left background work blocked after releasing its gates"


async def assert_closed(runtime):
    with pytest.raises(RuntimeError):
        await runtime.store.list_jobs()
    assert all(worker.done() for worker in runtime._workers)
    assert runtime._reaper is None or runtime._reaper.done()


@pytest.mark.parametrize("cancel_stop_caller", [False, True])
async def test_heartbeat_error_is_visible_before_tool_cleanup_and_survives_shutdown(
    runtime_factory, monkeypatch, cancel_stop_caller
):
    registry = ToolRegistry()
    fault_started, peer_started, peer_cancelled = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    allow_fault, cleanup_entered, release_cleanup = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    cleanup_finished, cleanup_interrupted = asyncio.Event(), asyncio.Event()
    fault = OSError("heartbeat database unavailable")

    @registry.tool(idempotent=False, timeout_ms=15_000)
    async def work(role):
        if role == "peer":
            peer_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                peer_cancelled.set()
        else:
            fault_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_entered.set()
                try:
                    await release_cleanup.wait()
                except asyncio.CancelledError:
                    cleanup_interrupted.set()
                    raise
                cleanup_finished.set()

    runtime = runtime_factory(registry)
    original_heartbeat = runtime.store.heartbeat_lease

    async def heartbeat(job_id, attempt_count, **kwargs):
        job = await runtime.store.get_job(job_id)
        if job.args["role"] == "fault":
            await allow_fault.wait()
            raise fault
        return await original_heartbeat(job_id, attempt_count, **kwargs)

    monkeypatch.setattr(runtime.store, "heartbeat_lease", heartbeat)
    await asyncio.wait_for(runtime.start(), WAIT_SECONDS)
    tasks = []
    try:
        fault_id = await runtime.submit("work", {"role": "fault"}, "fault")
        peer_id = await runtime.submit("work", {"role": "peer"}, "peer")
        await wait_event(fault_started)
        await wait_event(peer_started)
        observing = asyncio.Event()

        async def observe():
            observing.set()
            return await runtime.wait_for_terminal(fault_id)

        observer = asyncio.create_task(observe())
        tasks.append(observer)
        await wait_event(observing)
        allow_fault.set()
        await wait_event(cleanup_entered)

        # Cleanup is deliberately unfinished: a done callback cannot yet be the
        # mechanism that makes the heartbeat failure visible to these callers.
        await assert_failure(asyncio.shield(observer), fault)
        await assert_failure(
            runtime.submit("work", {"role": "peer"}, "must-not-be-admitted"), fault
        )
        await assert_failure(runtime.wait_for_terminal(peer_id), fault)
        assert runtime._error is fault
        assert len(await runtime.store.list_jobs()) == 2
        await wait_event(peer_cancelled)
        assert not cleanup_finished.is_set()
        assert not cleanup_interrupted.is_set()

        stop_entered = asyncio.Event()

        async def stop():
            stop_entered.set()
            await runtime.stop(graceful=False)

        stopping = asyncio.create_task(stop())
        tasks.append(stopping)
        await wait_event(stop_entered)
        assert not stopping.done()
        assert runtime._error is fault
        if cancel_stop_caller:
            assert stopping.cancel()
            await loop_turn()
            assert not stopping.done()
        assert not cleanup_interrupted.is_set()
        assert not cleanup_finished.is_set()
        # Neither error reporting nor caller cancellation may close the DB before
        # the trusted tool's finally block has completed.
        assert (await runtime.store.get_job(fault_id)).status is JobStatus.LEASED
        release_cleanup.set()
        if cancel_stop_caller:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(stopping), WAIT_SECONDS)
        else:
            await assert_failure(asyncio.shield(stopping), fault)
        assert cleanup_finished.is_set()
        assert not cleanup_interrupted.is_set()
        assert runtime._error is fault
        await assert_closed(runtime)
    finally:
        allow_fault.set()
        release_cleanup.set()
        await settle(*tasks)


async def test_completion_error_survives_heartbeat_false_during_operation_cleanup(
    runtime_factory, monkeypatch
):
    registry = ToolRegistry()
    heartbeat_entered, release_heartbeat, heartbeat_finished = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    heartbeat_false, heartbeat_cancelled = asyncio.Event(), asyncio.Event()
    cleanup_entered, release_cleanup = asyncio.Event(), asyncio.Event()
    fault = OSError("completion transaction failed before cleanup")
    tool_failures = []

    @registry.tool(idempotent=False)
    async def work() -> JSONValue:
        return {"valid": ["JSON", None]}

    runtime = runtime_factory(registry, max_workers=1)
    original_perform = runtime._perform
    original_fail = runtime.store.fail_or_retry_job

    async def complete(*args, **kwargs):
        # Ensure the heartbeat is in flight when completion fails.
        await heartbeat_entered.wait()
        raise fault

    async def perform(job):
        try:
            return await original_perform(job)
        finally:
            # This models operation cleanup after persistence has raised. A lease
            # loss can cancel this await and replace the escaping DB exception.
            cleanup_entered.set()
            await release_cleanup.wait()

    async def heartbeat(*args, **kwargs):
        heartbeat_entered.set()
        try:
            await release_heartbeat.wait()
            heartbeat_false.set()
            return False
        except asyncio.CancelledError:
            heartbeat_cancelled.set()
            raise
        finally:
            heartbeat_finished.set()

    async def fail(*args, **kwargs):
        tool_failures.append(args)
        return await original_fail(*args, **kwargs)

    monkeypatch.setattr(runtime.store, "complete_job", complete)
    monkeypatch.setattr(runtime, "_perform", perform)
    monkeypatch.setattr(runtime.store, "heartbeat_lease", heartbeat)
    monkeypatch.setattr(runtime.store, "fail_or_retry_job", fail)
    await asyncio.wait_for(runtime.start(), WAIT_SECONDS)
    stopping = None
    try:
        job_id = await runtime.submit("work", {}, "accepted")
        await wait_event(cleanup_entered)
        release_heartbeat.set()
        await wait_event(heartbeat_finished)
        # An engine that cancels peers immediately may cancel the heartbeat rather
        # than observe False. Either route must retain the already-raised DB error.
        assert heartbeat_false.is_set() or heartbeat_cancelled.is_set()
        await assert_failure(runtime.wait_for_terminal(job_id), fault)
        await assert_failure(runtime.submit("work", {}, "rejected"), fault)
        assert runtime._error is fault
        assert tool_failures == []
        assert (await runtime.store.get_job(job_id)).status is JobStatus.LEASED
        stopping = asyncio.create_task(runtime.stop(graceful=False))
        release_cleanup.set()
        await assert_failure(asyncio.shield(stopping), fault)
        assert runtime._error is fault
        await assert_closed(runtime)
    finally:
        release_heartbeat.set()
        release_cleanup.set()
        if stopping is not None:
            await settle(stopping)


@pytest.mark.parametrize("exception_type", [TypeError, ValueError])
@pytest.mark.parametrize("idempotent", [True, False])
async def test_completion_type_and_value_errors_are_infrastructure_not_invalid_json(
    runtime_factory, monkeypatch, exception_type, idempotent
):
    registry = ToolRegistry()
    fault = exception_type("store failed after the valid result was prevalidated")
    completion_entered = asyncio.Event()
    tool_failures = []

    @registry.tool(idempotent=idempotent)
    async def work() -> JSONValue:
        return {"valid": [1, "雪", None, False]}

    runtime = runtime_factory(registry, max_workers=1)
    original_fail = runtime.store.fail_or_retry_job

    async def complete(*args, **kwargs):
        completion_entered.set()
        raise fault

    async def fail(*args, **kwargs):
        tool_failures.append(args)
        return await original_fail(*args, **kwargs)

    monkeypatch.setattr(runtime.store, "complete_job", complete)
    monkeypatch.setattr(runtime.store, "fail_or_retry_job", fail)
    await asyncio.wait_for(runtime.start(), WAIT_SECONDS)
    job_id = await runtime.submit("work", {}, "accepted")
    await wait_event(completion_entered)
    await assert_failure(runtime.wait_for_terminal(job_id), fault)
    await assert_failure(runtime.submit("work", {}, "rejected"), fault)
    assert runtime._error is fault
    assert tool_failures == []
    job = await runtime.store.get_job(job_id)
    assert job.status is JobStatus.LEASED
    assert job.attempt_count == 1
    assert [event.to_status for event in await runtime.store.get_events(job_id)] == [
        JobStatus.PENDING,
        JobStatus.LEASED,
    ]
    await assert_failure(runtime.stop(graceful=False), fault)
    await assert_closed(runtime)
