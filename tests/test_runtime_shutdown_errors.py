import asyncio
from contextlib import suppress

import pytest
import pytest_asyncio

from runtime.engine import DurableToolRuntime
from runtime.models import JobStatus, RuntimeConfig, ToolPolicy
from runtime.registry import ToolRegistry
from runtime.store import SQLiteJobStore

pytestmark = pytest.mark.asyncio
WAIT_SECONDS = 5


@pytest_asyncio.fixture
async def runtime_factory(tmp_path):
    runtimes = []

    def make_runtime(registry):
        runtime = DurableToolRuntime(
            RuntimeConfig(
                db_path=str(tmp_path / f"shutdown-errors-{len(runtimes)}.sqlite3"),
                max_workers=2,
                max_outstanding_jobs=8,
                lease_duration_ms=15_000,
                heartbeat_interval_ms=20,
                poll_interval_ms=5,
                recovery_interval_ms=10,
            ),
            registry,
        )
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
        done, pending = await asyncio.wait(pending, timeout=WAIT_SECONDS)
        for task in done:
            if not task.cancelled():
                task.exception()
    assert not pending, "test left a task blocked after releasing its cleanup gate"


def background_tasks(runtime):
    tasks = list(runtime._workers)
    if runtime._reaper is not None:
        tasks.append(runtime._reaper)
    return tasks


async def assert_closed(runtime, tasks):
    with pytest.raises(RuntimeError):
        await runtime.store.list_jobs()
    assert all(task.done() for task in tasks)
    assert all(task.done() for task in background_tasks(runtime))


async def assert_retained_failure(runtime, fault, job_id):
    assert runtime._error is fault
    for method, args in (
        (runtime.submit, ("gated", {}, "after-shutdown")),
        (runtime.wait_for_terminal, (job_id,)),
    ):
        with pytest.raises(RuntimeError) as raised:
            await asyncio.wait_for(method(*args), WAIT_SECONDS)
        assert raised.value.__cause__ is fault


class CleanupGate:
    def __init__(self):
        self.started = asyncio.Event()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.interrupted = asyncio.Event()

    async def tool(self):
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.interrupted.set()
                raise
            self.finished.set()

    def assert_pending(self, stopping):
        assert self.entered.is_set()
        assert not self.finished.is_set()
        assert not self.interrupted.is_set()
        assert not stopping.done(), "stop returned before tool cleanup finished"


def registry_with_gate(gate):
    registry = ToolRegistry()
    registry.register(
        ToolPolicy("gated", False, timeout_ms=15_000, max_retries=0), gate.tool
    )
    return registry


@pytest.mark.parametrize(
    "source", ["heartbeat_lease", "recover_expired_leases", "claim_next_job"]
)
@pytest.mark.parametrize("cancel_stop_caller", [False, True])
async def test_store_failure_survives_immediate_stop_ahead_of_background_observation(
    runtime_factory, monkeypatch, source, cancel_stop_caller
):
    gate = CleanupGate()
    runtime = runtime_factory(registry_with_gate(gate))
    fault = OSError(f"injected {source} failure at shutdown")
    allow_fault, about_to_raise, stop_listening = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    original = getattr(runtime.store, source)
    armed = False

    async def fail_at_source(*args, **kwargs):
        if not armed or (source == "claim_next_job" and not gate.started.is_set()):
            return await original(*args, **kwargs)
        await allow_fault.wait()
        # Wake the independent stop caller BEFORE task completion can wake a
        # worker's heartbeat observer or schedule the source's done callback.
        about_to_raise.set()
        raise fault

    async def stop_on_failure():
        stop_listening.set()
        await about_to_raise.wait()
        await runtime.stop(graceful=False)

    monkeypatch.setattr(runtime.store, source, fail_at_source)
    stopping = None
    try:
        await asyncio.wait_for(runtime.start(), WAIT_SECONDS)
        armed = True
        job_id = await runtime.submit("gated", {}, "accepted")
        await wait_event(gate.started)
        tasks = background_tasks(runtime)
        stopping = asyncio.create_task(stop_on_failure())
        await wait_event(stop_listening)
        allow_fault.set()
        await wait_event(gate.entered)
        gate.assert_pending(stopping)
        assert (await runtime.store.get_job(job_id)).status is JobStatus.LEASED
        if cancel_stop_caller:
            assert stopping.cancel()
            await loop_turn()
            gate.assert_pending(stopping)
        gate.release.set()
        if cancel_stop_caller:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(stopping), WAIT_SECONDS)
        else:
            with pytest.raises(RuntimeError) as raised:
                await asyncio.wait_for(asyncio.shield(stopping), WAIT_SECONDS)
            assert raised.value.__cause__ is fault
        assert gate.finished.is_set()
        assert not gate.interrupted.is_set()
        await assert_closed(runtime, tasks)
        await assert_retained_failure(runtime, fault, job_id)
        async with SQLiteJobStore(runtime.store.config) as reader:
            job = await reader.get_job(job_id)
            assert job.status is JobStatus.LEASED
            assert job.attempt_count == 1
            assert [event.to_status for event in await reader.get_events(job_id)] == [
                JobStatus.PENDING,
                JobStatus.LEASED,
            ]
    finally:
        allow_fault.set()
        gate.release.set()
        if stopping is not None:
            await settle(stopping)


@pytest.mark.parametrize("late_done_callback", [False, True])
async def test_cancelled_stop_harvests_already_failed_reaper_after_cleanup(
    runtime_factory, monkeypatch, late_done_callback
):
    gate = CleanupGate()
    runtime = runtime_factory(registry_with_gate(gate))
    fault = OSError("completed reaper failure must survive deferred cancellation")
    injected_reaper = None
    stopping = None
    try:
        await asyncio.wait_for(runtime.start(), WAIT_SECONDS)
        job_id = await runtime.submit("gated", {}, "accepted")
        await wait_event(gate.started)
        original_reaper = runtime._reaper
        assert original_reaper is not None
        # Isolate the fallback from source-level reporting. Removing this callback
        # also prevents the intentional replacement from being reported as a fault.
        original_reaper.remove_done_callback(runtime._background_done)
        original_reaper.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(original_reaper), WAIT_SECONDS)

        async def failed_recovery(*args, **kwargs):
            raise fault

        monkeypatch.setattr(runtime.store, "recover_expired_leases", failed_recovery)
        injected_reaper = asyncio.create_task(
            runtime.store.recover_expired_leases(), name="already-failed-test-reaper"
        )
        done, pending = await asyncio.wait({injected_reaper}, timeout=WAIT_SECONDS)
        assert not pending
        assert injected_reaper in done
        assert injected_reaper.exception() is fault
        assert runtime._error is None
        monkeypatch.setattr(runtime, "_reaper", injected_reaper)
        tasks = [*background_tasks(runtime), original_reaper]
        stopping = asyncio.create_task(runtime.stop(graceful=False))
        await wait_event(gate.entered)
        gate.assert_pending(stopping)

        if late_done_callback:
            callback_finished = asyncio.Event()

            def report_done(task):
                try:
                    runtime._background_done(task)
                finally:
                    callback_finished.set()

            # Adding a callback to a completed task schedules it on the loop;
            # tool cleanup holds stop in its stopping phase until it runs.
            injected_reaper.add_done_callback(report_done)
            await wait_event(callback_finished)
            assert runtime._error is fault

        assert stopping.cancel()
        await loop_turn()
        gate.assert_pending(stopping)
        gate.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(stopping), WAIT_SECONDS)
        assert gate.finished.is_set()
        assert not gate.interrupted.is_set()
        await assert_closed(runtime, tasks)
        # With no done callback, retaining this error specifically requires
        # inspecting gather results even when _finish_cleanup rethrows cancellation.
        await assert_retained_failure(runtime, fault, job_id)
    finally:
        gate.release.set()
        await settle(
            *(task for task in (stopping, injected_reaper) if task is not None)
        )


async def test_late_reaper_done_callback_retains_real_error_after_store_is_closed(
    runtime_factory, monkeypatch
):
    runtime = runtime_factory(ToolRegistry())
    fault = OSError("reaper exception delivered after shutdown completed")
    await asyncio.wait_for(runtime.start(), WAIT_SECONDS)
    tasks = background_tasks(runtime)
    await asyncio.wait_for(runtime.stop(graceful=False), WAIT_SECONDS)
    await assert_closed(runtime, tasks)
    assert runtime._error is None

    async def failed_recovery(*args, **kwargs):
        raise fault

    monkeypatch.setattr(runtime.store, "recover_expired_leases", failed_recovery)
    reaper = asyncio.create_task(
        runtime.store.recover_expired_leases(), name="late-failed-test-reaper"
    )
    try:
        done, pending = await asyncio.wait({reaper}, timeout=WAIT_SECONDS)
        assert not pending
        assert reaper in done
        assert reaper.exception() is fault
        callback_finished = asyncio.Event()

        def report_done(task):
            try:
                runtime._background_done(task)
            finally:
                callback_finished.set()

        reaper.add_done_callback(report_done)
        await wait_event(callback_finished)
        await assert_retained_failure(runtime, fault, "historical-job")
        await assert_closed(runtime, [*tasks, reaper])
    finally:
        await settle(reaper)
