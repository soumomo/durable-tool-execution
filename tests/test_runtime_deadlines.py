import asyncio
from contextlib import suppress

import pytest
import pytest_asyncio

from runtime.engine import DurableToolRuntime
from runtime.models import JobStatus, RuntimeConfig, ToolPolicy
from runtime.registry import ToolRegistry
from runtime.store import SQLiteJobStore

WAIT_SECONDS = 5
TOOL_TIMEOUT_MS = 250


@pytest_asyncio.fixture
async def runtime_factory(tmp_path):
    runtimes = []

    def make_runtime(registry, **overrides):
        options = {
            "db_path": str(tmp_path / f"deadlines-{len(runtimes)}.sqlite3"),
            "max_workers": 1,
            "max_outstanding_jobs": 8,
            "lease_duration_ms": 2000,
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


async def settle(task):
    done, pending = await asyncio.wait({task}, timeout=WAIT_SECONDS)
    if pending:
        task.cancel()
        done, pending = await asyncio.wait({task}, timeout=WAIT_SECONDS)
    for finished in done:
        if not finished.cancelled():
            finished.exception()
    assert not pending, "task did not finish after its cleanup gate was released"


async def assert_closed(runtime):
    with pytest.raises(RuntimeError):
        await runtime.store.list_jobs()
    assert all(worker.done() for worker in runtime._workers)
    assert runtime._reaper is None or runtime._reaper.done()


class FinallyGate:
    def __init__(self):
        self.started = asyncio.Event()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.interrupted = asyncio.Event()
        self.deadline_passed = asyncio.Event()
        self.cancellation_counts = []
        self.deadline_marker = None

    async def tool(self):
        task = asyncio.current_task()
        assert task is not None
        # The runtime arms its deadline before entering the user tool. This marker
        # fires strictly later, even if the loop processes both timers in one turn.
        self.deadline_marker = asyncio.get_running_loop().call_later(
            TOOL_TIMEOUT_MS / 1000 + 0.05, self.deadline_passed.set
        )
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancellation_counts.append(task.cancelling())
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.interrupted.set()
                raise
            self.cancellation_counts.append(task.cancelling())
            self.finished.set()

    def dispose(self):
        self.release.set()
        if self.deadline_marker is not None:
            self.deadline_marker.cancel()

    def assert_pending(self, task):
        assert self.entered.is_set()
        assert not self.interrupted.is_set(), (
            "runtime cancelled the tool's finally block twice"
        )
        assert not self.finished.is_set()
        assert self.cancellation_counts == [1]
        assert not task.done(), "caller returned before the tool finished cleanup"


@pytest.mark.parametrize("shutdown", ["immediate", "cancelled-graceful"])
async def test_shutdown_disarms_original_deadline_before_joining_tool_cleanup(
    runtime_factory, monkeypatch, shutdown
):
    registry = ToolRegistry()
    gate = FinallyGate()
    registry.register(
        ToolPolicy("gated", False, timeout_ms=TOOL_TIMEOUT_MS, max_retries=0), gate.tool
    )
    runtime = runtime_factory(registry)
    drain_entered, release_drain = asyncio.Event(), asyncio.Event()
    original_count = runtime.store.count_outstanding_jobs

    async def count():
        drain_entered.set()
        await release_drain.wait()
        return await original_count()

    if shutdown == "cancelled-graceful":
        # Observe stop's drain phase directly before cancelling its caller.
        monkeypatch.setattr(runtime.store, "count_outstanding_jobs", count)
    stopping = None
    try:
        await asyncio.wait_for(runtime.start(), WAIT_SECONDS)
        job_id = await runtime.submit("gated", {}, "accepted")
        await wait_event(gate.started)
        if shutdown == "immediate":
            stopping = asyncio.create_task(runtime.stop(graceful=False))
        else:
            stopping = asyncio.create_task(runtime.stop())
            await wait_event(drain_entered)
            assert stopping.cancel()
        await wait_event(gate.entered)
        gate.assert_pending(stopping)
        await wait_event(gate.deadline_passed)
        await loop_turn()
        # No time-based ordering assertion: the deadline marker has fired while
        # the finally gate is still closed, and shutdown must still be joining it.
        gate.assert_pending(stopping)
        assert (await runtime.store.get_job(job_id)).status is JobStatus.LEASED
        gate.release.set()
        if shutdown == "cancelled-graceful":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(stopping), WAIT_SECONDS)
        else:
            assert (
                await asyncio.wait_for(asyncio.shield(stopping), WAIT_SECONDS) is None
            )
        assert gate.finished.is_set()
        assert not gate.interrupted.is_set()
        assert gate.cancellation_counts == [1, 1]
        assert runtime._error is None
        await assert_closed(runtime)
        async with SQLiteJobStore(runtime.store.config) as reader:
            abandoned = await reader.get_job(job_id)
            assert abandoned.status is JobStatus.LEASED
            assert abandoned.attempt_count == 1
            assert [event.to_status for event in await reader.get_events(job_id)] == [
                JobStatus.PENDING,
                JobStatus.LEASED,
            ]
    finally:
        gate.dispose()
        release_drain.set()
        if stopping is not None:
            await settle(stopping)


@pytest.mark.parametrize("idempotent", [True, False])
async def test_normal_timeout_joins_finally_before_recording_outcome(
    runtime_factory, monkeypatch, idempotent
):
    registry = ToolRegistry()
    gate = FinallyGate()
    registry.register(
        ToolPolicy("gated", idempotent, timeout_ms=TOOL_TIMEOUT_MS, max_retries=0),
        gate.tool,
    )
    runtime = runtime_factory(registry)
    heartbeat_in_cleanup = asyncio.Event()
    original_heartbeat = runtime.store.heartbeat_lease

    async def heartbeat(job_id, attempt_count, **kwargs):
        renewed = await original_heartbeat(job_id, attempt_count, **kwargs)
        if renewed and gate.entered.is_set():
            heartbeat_in_cleanup.set()
        return renewed

    monkeypatch.setattr(runtime.store, "heartbeat_lease", heartbeat)
    observer = None
    try:
        await asyncio.wait_for(runtime.start(), WAIT_SECONDS)
        job_id = await runtime.submit("gated", {}, "accepted")
        observer = asyncio.create_task(runtime.wait_for_terminal(job_id))
        await wait_event(gate.entered)
        await wait_event(gate.deadline_passed)
        await wait_event(heartbeat_in_cleanup)
        await loop_turn()
        gate.assert_pending(observer)
        assert (await runtime.store.get_job(job_id)).status is JobStatus.LEASED
        assert [
            event.to_status for event in await runtime.store.get_events(job_id)
        ] == [JobStatus.PENDING, JobStatus.LEASED]
        gate.release.set()
        result = await asyncio.wait_for(asyncio.shield(observer), WAIT_SECONDS)
        assert result.status is (
            JobStatus.FAILED if idempotent else JobStatus.UNKNOWN_OUTCOME
        )
        assert result.attempt_count == 1
        assert result.error_text
        assert gate.finished.is_set()
        assert not gate.interrupted.is_set()
        assert gate.cancellation_counts == [1, 1]
        assert runtime._error is None
        assert runtime.lost_leases == runtime.fenced_stale_completions == 0
        await asyncio.wait_for(runtime.stop(), WAIT_SECONDS)
        await assert_closed(runtime)
    finally:
        gate.dispose()
        if observer is not None:
            await settle(observer)


@pytest.mark.parametrize("field", ["poll_interval_ms", "recovery_interval_ms"])
@pytest.mark.parametrize("value", [0, -1, True, 1.0, "10", None])
def test_poll_and_recovery_intervals_require_positive_integers(tmp_path, field, value):
    with pytest.raises(ValueError):
        RuntimeConfig(db_path=str(tmp_path / "invalid.sqlite3"), **{field: value})


def test_poll_and_recovery_interval_defaults_and_short_positive_values(tmp_path):
    default = RuntimeConfig(db_path=str(tmp_path / "default.sqlite3"))
    assert default.poll_interval_ms == 50
    assert default.recovery_interval_ms == 100
    short = RuntimeConfig(
        db_path=str(tmp_path / "short.sqlite3"),
        poll_interval_ms=5,
        recovery_interval_ms=20,
    )
    assert short.poll_interval_ms == 5
    assert short.recovery_interval_ms == 20


@pytest.mark.parametrize("graceful", [True, False])
async def test_stop_before_start_closes_one_shot_runtime_without_background_tasks(
    runtime_factory, graceful
):
    runtime = runtime_factory(ToolRegistry())
    await asyncio.wait_for(runtime.stop(graceful=graceful), WAIT_SECONDS)
    await runtime.stop(graceful=not graceful)
    await assert_closed(runtime)
    assert runtime._workers == []
    assert runtime._reaper is None
    with pytest.raises(RuntimeError):
        await runtime.start()
    with pytest.raises(RuntimeError):
        await runtime.submit("missing", {}, "after-stop")
    with pytest.raises(RuntimeError):
        await runtime.wait_for_terminal("missing")


@pytest.mark.parametrize(
    "timeout", [-1, float("nan"), float("inf"), -float("inf"), True, False, "1", 1j]
)
async def test_invalid_waiter_timeout_does_not_poison_runtime(runtime_factory, timeout):
    registry = ToolRegistry()

    @registry.tool(idempotent=True)
    async def healthy():
        return "healthy"

    runtime = runtime_factory(registry)
    await asyncio.wait_for(runtime.start(), WAIT_SECONDS)
    job_id = await runtime.submit("healthy", {}, "accepted")
    result = await asyncio.wait_for(runtime.wait_for_terminal(job_id), WAIT_SECONDS)
    assert result.status is JobStatus.SUCCEEDED
    with pytest.raises(ValueError):
        await runtime.wait_for_terminal(job_id, timeout=timeout)
    assert await runtime.wait_for_terminal(job_id, timeout=1) == result
    assert runtime._error is None
    assert all(not worker.done() for worker in runtime._workers)
