import asyncio
import sqlite3
import threading
from contextlib import closing

import pytest

from runtime import store as store_module
from runtime.models import JobStatus, StaleLeaseError, ToolPolicy

pytestmark = pytest.mark.asyncio
WAIT_SECONDS = 5
NOW = 1000.0
POLICY = ToolPolicy("fetch", True)


class ThreadGate:
    def __init__(self, name):
        self.name = name
        self.entered = threading.Event()
        self.released = threading.Event()
        self.threads = []

    def block(self):
        self.threads.append(threading.get_ident())
        self.entered.set()
        if not self.released.wait(timeout=WAIT_SECONDS):
            raise AssertionError(f"test did not release {self.name}")

    async def wait_until_entered(self):
        await wait_for_event(self.entered, self.name)

    def release(self):
        self.released.set()


async def wait_for_event(event, name):
    observed = await asyncio.wait_for(
        asyncio.to_thread(event.wait, WAIT_SECONDS), timeout=WAIT_SECONDS + 1
    )
    assert observed, f"timed out waiting for {name}"


async def loop_turns():
    # Yield to ready tasks without using elapsed time to order the race.
    for _ in range(3):
        await asyncio.sleep(0)


async def cancel_repeatedly(task, count=3):
    for _ in range(count):
        assert task.cancel(), "cleanup finished before its gate was released"
        await loop_turns()
        assert not task.done(), "cancellation escaped before cleanup completed"


async def task_result(task):
    done, pending = await asyncio.wait({task}, timeout=WAIT_SECONDS)
    assert not pending, "task did not finish after its gates were released"
    assert task in done
    return task.result()


async def drain_tasks(*tasks):
    if not tasks:
        return
    done, pending = await asyncio.wait(tasks, timeout=WAIT_SECONDS)
    for task in done:
        if not task.cancelled():
            task.exception()
    for task in pending:
        task.cancel()
    assert not pending, "test left a task blocked after releasing all gates"


def assert_closed_on_owner(store, executor, close_gate, close_finished, owner_thread):
    assert close_finished.is_set(), "the actual connection close did not finish"
    assert close_gate.threads == [owner_thread]
    assert owner_thread != threading.get_ident()
    assert store._connection is None
    assert store._state == "closed"
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(lambda: None)


async def test_repeated_start_cancellation_waits_for_blocked_open_and_blocked_close(
    store_factory, monkeypatch
):
    store = store_factory()
    open_gate = ThreadGate("open")
    close_gate = ThreadGate("start cleanup close")
    close_finished = threading.Event()
    original_open, original_close = store._open, store._close

    def blocked_open():
        original_open()
        open_gate.block()

    def blocked_close():
        close_gate.block()
        original_close()
        close_finished.set()

    monkeypatch.setattr(store, "_open", blocked_open)
    monkeypatch.setattr(store, "_close", blocked_close)
    start = asyncio.create_task(store.start())
    tasks = [start]
    try:
        await open_gate.wait_until_entered()
        executor = store._executor
        assert executor is not None
        assert store._connection is not None
        await cancel_repeatedly(start)
        assert not close_gate.entered.is_set()
        open_gate.release()
        await close_gate.wait_until_entered()
        await cancel_repeatedly(start)
        assert store._connection is not None
        assert not close_finished.is_set()

        # The lifecycle lock must stay held throughout cancelled-start cleanup.
        concurrent_close = asyncio.create_task(store.close())
        tasks.append(concurrent_close)
        await loop_turns()
        assert not concurrent_close.done()
        close_gate.release()
        with pytest.raises(asyncio.CancelledError):
            await task_result(start)
        assert await task_result(concurrent_close) is None
        assert_closed_on_owner(
            store, executor, close_gate, close_finished, open_gate.threads[0]
        )
        with pytest.raises(RuntimeError):
            await store.list_jobs()
        await store.close()
        assert len(close_gate.threads) == 1
    finally:
        open_gate.release()
        close_gate.release()
        await drain_tasks(*tasks)


async def test_repeated_close_cancellation_behind_busy_reader_still_closes_on_owner(
    store, monkeypatch
):
    admitted, _ = await store.admit_job(POLICY, {}, "key", now=NOW)
    read_gate = ThreadGate("read occupying executor")
    close_gate = ThreadGate("close behind read")
    close_finished = threading.Event()
    original_read, original_close = store._get_job, store._close
    executor = store._executor
    order = []

    def blocked_read(job_id):
        order.append("read entered")
        read_gate.block()
        result = original_read(job_id)
        order.append("read finished")
        return result

    def blocked_close():
        order.append("close entered")
        close_gate.block()
        original_close()
        order.append("close finished")
        close_finished.set()

    monkeypatch.setattr(store, "_get_job", blocked_read)
    monkeypatch.setattr(store, "_close", blocked_close)
    read = asyncio.create_task(store.get_job(admitted.job_id))
    tasks = [read]
    try:
        await read_gate.wait_until_entered()
        close = asyncio.create_task(store.close())
        tasks.append(close)
        await loop_turns()
        assert store._state == "closing"
        await cancel_repeatedly(close)
        assert not close_gate.entered.is_set()
        with pytest.raises(RuntimeError):
            await store.count_outstanding_jobs()

        read_gate.release()
        await close_gate.wait_until_entered()
        await cancel_repeatedly(close)
        assert not close_finished.is_set()
        close_gate.release()
        assert await task_result(read) == admitted
        with pytest.raises(asyncio.CancelledError):
            await task_result(close)
        assert order == [
            "read entered",
            "read finished",
            "close entered",
            "close finished",
        ]
        assert_closed_on_owner(
            store, executor, close_gate, close_finished, read_gate.threads[0]
        )
        await store.close()
        assert len(close_gate.threads) == 1
    finally:
        read_gate.release()
        close_gate.release()
        await drain_tasks(*tasks)


@pytest.mark.parametrize("cancel_first", [False, True])
async def test_concurrent_close_calls_close_connection_exactly_once(
    store, monkeypatch, cancel_first
):
    admitted, _ = await store.admit_job(POLICY, {}, "key", now=NOW)
    original_read, original_close = store._get_job, store._close
    owner_threads = []
    close_gate = ThreadGate("concurrent close")
    close_finished = threading.Event()
    executor = store._executor

    def owner_read(job_id):
        owner_threads.append(threading.get_ident())
        return original_read(job_id)

    def blocked_close():
        close_gate.block()
        original_close()
        close_finished.set()

    monkeypatch.setattr(store, "_get_job", owner_read)
    monkeypatch.setattr(store, "_close", blocked_close)
    await store.get_job(admitted.job_id)
    first = asyncio.create_task(store.close())
    tasks = [first]
    try:
        await close_gate.wait_until_entered()
        followers = [asyncio.create_task(store.close()) for _ in range(4)]
        tasks.extend(followers)
        await loop_turns()
        assert all(not task.done() for task in tasks)
        followers[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_result(followers[0])
        if cancel_first:
            await cancel_repeatedly(first)
        assert not close_finished.is_set()
        assert len(close_gate.threads) == 1
        close_gate.release()
        if cancel_first:
            with pytest.raises(asyncio.CancelledError):
                await task_result(first)
        else:
            assert await task_result(first) is None
        for follower in followers[1:]:
            assert await task_result(follower) is None
        assert_closed_on_owner(
            store, executor, close_gate, close_finished, owner_threads[0]
        )
    finally:
        close_gate.release()
        await drain_tasks(*tasks)


@pytest.mark.parametrize("opened_connection", [False, True])
@pytest.mark.parametrize("cancel_cleanup", [False, True])
async def test_failed_start_cleans_up_on_owner_even_if_cleanup_is_cancelled_repeatedly(
    store_factory, monkeypatch, opened_connection, cancel_cleanup
):
    store = store_factory()
    close_gate = ThreadGate("failed-start cleanup")
    close_finished = threading.Event()
    original_open, original_close = store._open, store._close
    owner_threads = []

    class OpenFailure(Exception):
        pass

    def failed_open():
        owner_threads.append(threading.get_ident())
        if opened_connection:
            original_open()
        raise OpenFailure("injected initialization failure")

    def blocked_close():
        close_gate.block()
        original_close()
        close_finished.set()

    monkeypatch.setattr(store, "_open", failed_open)
    monkeypatch.setattr(store, "_close", blocked_close)
    start = asyncio.create_task(store.start())
    try:
        await close_gate.wait_until_entered()
        executor = store._executor
        assert executor is not None
        assert (store._connection is not None) is opened_connection
        if cancel_cleanup:
            await cancel_repeatedly(start)
        assert not start.done()
        assert not close_finished.is_set()
        close_gate.release()
        if cancel_cleanup:
            # Either error precedence is acceptable; leaking cleanup is not.
            with pytest.raises((OpenFailure, asyncio.CancelledError)):
                await task_result(start)
        else:
            with pytest.raises(OpenFailure, match="injected initialization failure"):
                await task_result(start)
        assert_closed_on_owner(
            store, executor, close_gate, close_finished, owner_threads[0]
        )
        await store.close()
        assert len(close_gate.threads) == 1
        with pytest.raises(RuntimeError):
            await store.start()
        with pytest.raises(RuntimeError):
            await store.list_jobs()
    finally:
        close_gate.release()
        await drain_tasks(start)


class ControlledClock:
    def __init__(self, value):
        self.value = value
        self.samples = []

    def __call__(self):
        self.samples.append((threading.get_ident(), self.value))
        return self.value


async def default_clock_operation(store, operation, claimed):
    if operation == "heartbeat":
        return await store.heartbeat_lease(claimed.job_id, claimed.attempt_count)
    return await store.complete_job(
        claimed.job_id, claimed.attempt_count, {"late": True}
    )


async def assert_expired_operation(
    store, operation, task, claimed, events_before, timestamp
):
    if operation == "heartbeat":
        assert await task_result(task) is False
        assert await store.get_events(claimed.job_id) == events_before
    else:
        with pytest.raises(StaleLeaseError):
            await task_result(task)
        events = await store.get_events(claimed.job_id)
        assert events[:-1] == events_before
        assert events[-1].event_type == "STALE_COMPLETION_FENCED"
        assert events[-1].timestamp == timestamp
        assert events[-1].attempt == claimed.attempt_count
        assert events[-1].from_status is JobStatus.LEASED
        assert events[-1].to_status is JobStatus.LEASED
    assert await store.get_job(claimed.job_id) == claimed


@pytest.mark.parametrize("operation", ["heartbeat", "complete"])
@pytest.mark.parametrize(
    "expiry_offset", [0.0, 0.5], ids=["exact-expiry", "past-expiry"]
)
async def test_default_clock_is_sampled_on_owner_after_executor_queue_delay(
    store, monkeypatch, operation, expiry_offset
):
    admitted, _ = await store.admit_job(POLICY, {}, "key", now=NOW)
    claimed = await store.claim_next_job("worker", now=NOW)
    assert claimed.job_id == admitted.job_id
    events_before = await store.get_events(claimed.job_id)
    read_gate = ThreadGate("read before clock-sensitive operation")
    original_read = store._get_job

    def blocked_read(job_id):
        read_gate.block()
        return original_read(job_id)

    monkeypatch.setattr(store, "_get_job", blocked_read)
    clock = ControlledClock(NOW + 0.25)
    monkeypatch.setattr(store_module.time, "time", clock)
    read = asyncio.create_task(store.get_job(claimed.job_id))
    tasks = [read]
    try:
        await read_gate.wait_until_entered()
        operation_task = asyncio.create_task(
            default_clock_operation(store, operation, claimed)
        )
        tasks.append(operation_task)
        await loop_turns()
        assert not operation_task.done()
        assert clock.samples == [], (
            "the default timestamp was sampled before reaching the DB thread"
        )
        clock.value = claimed.lease_expires_at + expiry_offset
        read_gate.release()
        assert await task_result(read) == claimed
        # Restore the read wrapper before the assertions read the record again.
        monkeypatch.setattr(store, "_get_job", original_read)
        await assert_expired_operation(
            store, operation, operation_task, claimed, events_before, clock.value
        )
        assert clock.samples
        assert all(thread == read_gate.threads[0] for thread, _ in clock.samples)
        assert all(value == clock.value for _, value in clock.samples)
        assert read_gate.threads[0] != threading.get_ident()
    finally:
        read_gate.release()
        await drain_tasks(*tasks)


@pytest.mark.parametrize("operation", ["heartbeat", "complete"])
@pytest.mark.parametrize(
    "expiry_offset", [0.0, 0.5], ids=["exact-expiry", "past-expiry"]
)
async def test_default_clock_is_sampled_after_begin_immediate_acquires_writer_lock(
    store_factory, db_path, monkeypatch, operation, expiry_offset
):
    real_connect = sqlite3.connect
    begin_attempted = threading.Event()
    begin_acquired = threading.Event()
    owner_threads = []
    armed = False

    class ObservedConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if armed and sql.strip().upper() == "BEGIN IMMEDIATE":
                owner_threads.append(threading.get_ident())
                begin_attempted.set()
                result = super().execute(sql, *args, **kwargs)
                begin_acquired.set()
                return result
            return super().execute(sql, *args, **kwargs)

    def connect(*args, **kwargs):
        kwargs["factory"] = ObservedConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(store_module.sqlite3, "connect", connect)
    store = store_factory()
    await store.start()
    await store.admit_job(POLICY, {}, "key", now=NOW)
    claimed = await store.claim_next_job("worker", now=NOW)
    events_before = await store.get_events(claimed.job_id)
    clock = ControlledClock(NOW + 0.25)
    monkeypatch.setattr(store_module.time, "time", clock)

    # This independent connection holds the write lock while the owner enters BEGIN.
    with closing(
        real_connect(db_path, isolation_level=None, timeout=WAIT_SECONDS)
    ) as writer:
        writer.execute("BEGIN IMMEDIATE")
        armed = True
        operation_task = asyncio.create_task(
            default_clock_operation(store, operation, claimed)
        )
        try:
            await wait_for_event(begin_attempted, "owner attempting BEGIN IMMEDIATE")
            assert not begin_acquired.is_set()
            assert not operation_task.done()
            assert clock.samples == [], (
                "the timestamp was sampled before obtaining the write lock"
            )
            clock.value = claimed.lease_expires_at + expiry_offset
            writer.execute("ROLLBACK")
            await assert_expired_operation(
                store, operation, operation_task, claimed, events_before, clock.value
            )
            assert begin_acquired.is_set()
            assert clock.samples
            assert len(owner_threads) == 1
            assert owner_threads[0] != threading.get_ident()
            assert all(thread == owner_threads[0] for thread, _ in clock.samples)
            assert all(value == clock.value for _, value in clock.samples)
        finally:
            if writer.in_transaction:
                writer.execute("ROLLBACK")
            await drain_tasks(operation_task)
