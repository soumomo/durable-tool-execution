import asyncio
import sqlite3
import threading
from contextlib import closing

import pytest

from runtime.models import JobStatus, StaleLeaseError, ToolPolicy

pytestmark = pytest.mark.asyncio
POLICY = ToolPolicy("fetch", True)
NOW = 1000.0


PUBLIC_OPERATIONS = [
    pytest.param("admit_job", (POLICY, {}, "key"), {"now": NOW}, id="admit"),
    pytest.param("claim_next_job", ("worker",), {"now": NOW}, id="claim"),
    pytest.param("heartbeat_lease", ("missing", 1), {"now": NOW}, id="heartbeat"),
    pytest.param("complete_job", ("missing", 1, None), {"now": NOW}, id="complete"),
    pytest.param(
        "fail_or_retry_job",
        ("missing", 1, "error"),
        {"retryable": True, "now": NOW},
        id="fail",
    ),
    pytest.param("recover_expired_leases", (), {"now": NOW}, id="recover"),
    pytest.param("get_job", ("missing",), {}, id="get-job"),
    pytest.param("get_events", ("missing",), {}, id="get-events"),
    pytest.param("list_jobs", (), {}, id="list"),
    pytest.param("count_outstanding_jobs", (), {}, id="count"),
]


@pytest.mark.parametrize("method,args,kwargs", PUBLIC_OPERATIONS)
@pytest.mark.parametrize("lifecycle", ["before-start", "after-close"])
async def test_public_operations_require_an_open_store(
    store_factory, method, args, kwargs, lifecycle
):
    store = store_factory()
    if lifecycle == "after-close":
        await store.start()
        await store.close()
    with pytest.raises(RuntimeError):
        await getattr(store, method)(*args, **kwargs)


async def test_start_and_close_are_idempotent_and_empty_store_queries_are_safe(
    store_factory,
):
    store = store_factory()
    await store.start()
    await store.start()
    assert await store.list_jobs() == []
    assert await store.count_outstanding_jobs() == 0
    assert await store.claim_next_job("worker", now=NOW) is None
    assert await store.recover_expired_leases(now=NOW) == []
    job, _ = await store.admit_job(POLICY, {}, "key", now=NOW)
    await store.start()
    assert await store.get_job(job.job_id) == job
    assert len(await store.get_events(job.job_id)) == 1
    await store.close()
    await store.close()
    with pytest.raises(RuntimeError):
        await store.list_jobs()


async def test_close_before_start_is_safe_and_idempotent(store_factory):
    store = store_factory()
    await store.close()
    await store.close()
    with pytest.raises(RuntimeError):
        await store.list_jobs()


@pytest.mark.parametrize("exceptional_exit", [False, True])
async def test_async_context_manager_starts_and_closes_even_on_exception(
    store_factory, exceptional_exit
):
    store = store_factory()

    class BodyError(Exception):
        pass

    try:
        async with store as opened:
            assert opened is store
            await store.admit_job(POLICY, {}, "key", now=NOW)
            if exceptional_exit:
                raise BodyError("body failed")
    except BodyError:
        assert exceptional_exit
    with pytest.raises(RuntimeError):
        await store.list_jobs()
    reopened = store_factory()
    await reopened.start()
    assert len(await reopened.list_jobs()) == 1


async def test_acknowledged_admission_and_event_are_committed_before_return(
    store, db_path, store_factory
):
    job, created = await store.admit_job(POLICY, {"x": 1}, "key", now=NOW)
    assert created
    with closing(sqlite3.connect(db_path)) as reader:
        assert reader.execute("SELECT job_id FROM jobs").fetchall() == [(job.job_id,)]
        assert reader.execute("SELECT job_id FROM job_events").fetchall() == [
            (job.job_id,)
        ]
    independent_store = store_factory()
    await independent_store.start()
    assert await independent_store.get_job(job.job_id) == job
    assert await independent_store.get_events(job.job_id) == await store.get_events(
        job.job_id
    )


@pytest.mark.parametrize("status", list(JobStatus))
async def test_all_job_states_policy_fields_and_events_survive_restart(
    store_factory, status
):
    store = store_factory()
    await store.start()
    policy = ToolPolicy(
        "snapshotted-tool",
        status is not JobStatus.UNKNOWN_OUTCOME,
        timeout_ms=731,
        max_retries=2,
        base_backoff_ms=375,
    )
    job, _ = await store.admit_job(
        policy, {"雪": [False, None, {"n": 3}]}, "key", now=NOW
    )
    if status is not JobStatus.PENDING:
        job = await store.claim_next_job("original-worker", now=NOW)
    if status is JobStatus.SUCCEEDED:
        job = await store.complete_job(
            job.job_id, 1, {"result": [0, "雪", None]}, now=NOW + 0.1
        )
    elif status in {JobStatus.RETRY_WAIT, JobStatus.FAILED, JobStatus.UNKNOWN_OUTCOME}:
        job = await store.fail_or_retry_job(
            job.job_id,
            1,
            "persist this error",
            retryable=status is JobStatus.RETRY_WAIT,
            now=NOW + 0.1,
        )
    assert job.status is status
    events = await store.get_events(job.job_id)
    await store.close()

    restarted = store_factory()
    await restarted.start()
    assert await restarted.get_job(job.job_id) == job
    assert await restarted.list_jobs() == [job]
    assert await restarted.get_events(job.job_id) == events
    assert await restarted.count_outstanding_jobs() == (0 if status.terminal else 1)
    changed_policy = ToolPolicy(
        policy.name,
        not policy.idempotent,
        timeout_ms=1,
        max_retries=0,
        base_backoff_ms=0,
    )
    duplicate, created = await restarted.admit_job(
        changed_policy, job.args, "key", now=NOW + 100
    )
    assert created is False
    assert duplicate == job
    assert duplicate.policy == policy
    assert await restarted.get_events(job.job_id) == events


@pytest.mark.parametrize("idempotent", [True, False])
async def test_restart_recovery_uses_frozen_policy_not_new_submission_policy(
    store_factory, idempotent
):
    store = store_factory()
    await store.start()
    policy = ToolPolicy(
        "tool", idempotent, timeout_ms=917, max_retries=2, base_backoff_ms=750
    )
    admitted, _ = await store.admit_job(policy, {}, "key", now=NOW)
    await store.claim_next_job("crashed-worker", now=NOW)
    await store.close()
    restarted = store_factory()
    await restarted.start()
    changed = ToolPolicy(
        "tool", not idempotent, timeout_ms=1, max_retries=0, base_backoff_ms=0
    )
    original, created = await restarted.admit_job(changed, {}, "key", now=NOW + 1)
    assert created is False
    assert original.policy == policy
    recovered = await restarted.recover_expired_leases(now=NOW + 1)
    assert len(recovered) == 1
    assert recovered[0].job_id == admitted.job_id
    assert recovered[0].policy == policy
    assert recovered[0].attempt_count == 1
    if idempotent:
        assert recovered[0].status is JobStatus.RETRY_WAIT
        assert recovered[0].next_eligible_at == NOW + 1.75
        assert await restarted.claim_next_job("early", now=NOW + 1.749) is None
        second = await restarted.claim_next_job("new-worker", now=NOW + 1.75)
        assert second.attempt_count == 2
    else:
        assert recovered[0].status is JobStatus.UNKNOWN_OUTCOME
        assert await restarted.claim_next_job("never", now=NOW + 100) is None


async def test_fenced_event_survives_exception_and_restart_without_modifying_job(
    store_factory,
):
    store = store_factory()
    await store.start()
    await store.admit_job(POLICY, {}, "key", now=NOW)
    claimed = await store.claim_next_job("worker", now=NOW)
    with pytest.raises(StaleLeaseError):
        await store.complete_job(claimed.job_id, 1, "expired result", now=NOW + 1)
    before_restart = await store.get_events(claimed.job_id)
    assert before_restart[-1].event_type == "STALE_COMPLETION_FENCED"
    await store.close()
    restarted = store_factory()
    await restarted.start()
    assert await restarted.get_job(claimed.job_id) == claimed
    assert await restarted.get_events(claimed.job_id) == before_restart


def install_event_failure(db_path, target):
    # Abort only the event insert, so a partial job mutation would be observable.
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            "CREATE TRIGGER reject_test_event BEFORE INSERT ON job_events "
            f"WHEN NEW.event_type = 'STATE_TRANSITION' AND NEW.to_status = '{target}' "
            "BEGIN SELECT RAISE(ABORT, 'injected event failure'); END"
        )


def remove_event_failure(db_path):
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute("DROP TRIGGER reject_test_event")


async def test_admission_and_initial_event_rollback_as_one_transaction(store, db_path):
    install_event_failure(db_path, "PENDING")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="injected event failure"):
            await store.admit_job(POLICY, {}, "key", now=NOW)
        assert await store.list_jobs() == []
        assert await store.count_outstanding_jobs() == 0
        with closing(sqlite3.connect(db_path)) as reader:
            assert reader.execute("SELECT count(*) FROM job_events").fetchone()[0] == 0
    finally:
        remove_event_failure(db_path)
    admitted, created = await store.admit_job(POLICY, {}, "key", now=NOW)
    assert created
    assert len(await store.get_events(admitted.job_id)) == 1


@pytest.mark.parametrize("operation", ["claim", "complete", "fail", "recover"])
async def test_state_change_and_event_are_atomic_and_connection_recovers_after_rollback(
    store, db_path, operation
):
    before, _ = await store.admit_job(POLICY, {}, "key", now=NOW)
    if operation != "claim":
        before = await store.claim_next_job("worker", now=NOW)
    events = await store.get_events(before.job_id)
    target = {
        "claim": "LEASED",
        "complete": "SUCCEEDED",
        "fail": "RETRY_WAIT",
        "recover": "RETRY_WAIT",
    }[operation]

    async def transition():
        if operation == "claim":
            return await store.claim_next_job("worker", now=NOW)
        if operation == "complete":
            return await store.complete_job(
                before.job_id, 1, {"ok": True}, now=NOW + 0.1
            )
        if operation == "fail":
            return await store.fail_or_retry_job(
                before.job_id, 1, "retry", retryable=True, now=NOW + 0.1
            )
        return await store.recover_expired_leases(now=NOW + 1)

    install_event_failure(db_path, target)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="injected event failure"):
            await transition()
        assert await store.get_job(before.job_id) == before
        assert await store.get_events(before.job_id) == events
    finally:
        remove_event_failure(db_path)
    await transition()
    assert (await store.get_job(before.job_id)).status.value == target
    assert len(await store.get_events(before.job_id)) == len(events) + 1


@pytest.mark.parametrize("mode,expected_sync", [("FULL", 2), ("NORMAL", 1)])
async def test_connection_pragmas_and_entire_lifecycle_use_one_non_loop_thread(
    store_factory, monkeypatch, mode, expected_sync
):
    loop_thread = threading.get_ident()
    real_connect = sqlite3.connect
    connections = []
    thread_ids = []

    class TrackedConnection(sqlite3.Connection):
        def execute(self, *args, **kwargs):
            thread_ids.append(threading.get_ident())
            return super().execute(*args, **kwargs)

        def close(self):
            thread_ids.append(threading.get_ident())
            return super().close()

    def connect(*args, **kwargs):
        thread_ids.append(threading.get_ident())
        kwargs["factory"] = TrackedConnection
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    store = store_factory(synchronous_mode=mode)
    await store.start()
    assert len(connections) == 1
    original_get_job = store._get_job
    settings = {}

    def inspect_on_owner_thread(job_id):
        thread_ids.append(threading.get_ident())
        connection = connections[0]
        for pragma in ("journal_mode", "synchronous", "foreign_keys"):
            settings[pragma] = connection.execute(f"PRAGMA {pragma}").fetchone()[0]
        return original_get_job(job_id)

    monkeypatch.setattr(store, "_get_job", inspect_on_owner_thread)
    admitted, _ = await store.admit_job(POLICY, {}, "key", now=NOW)
    await asyncio.gather(*[store.get_job(admitted.job_id) for _ in range(8)])
    claimed = await store.claim_next_job("worker", now=NOW)
    await store.heartbeat_lease(claimed.job_id, 1, now=NOW + 0.1)
    await store.complete_job(claimed.job_id, 1, None, now=NOW + 0.2)
    await store.get_events(claimed.job_id)
    await store.list_jobs()
    await store.count_outstanding_jobs()
    await store.close()
    assert settings == {
        "journal_mode": "wal",
        "synchronous": expected_sync,
        "foreign_keys": 1,
    }
    assert len(set(thread_ids)) == 1
    assert loop_thread not in thread_ids
    assert len(connections) == 1


async def test_slow_sync_read_does_not_block_event_loop(store, monkeypatch):
    admitted, _ = await store.admit_job(POLICY, {}, "key", now=NOW)
    original_get_job = store._get_job
    entered = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()
    reader_threads = []

    def blocked_read(job_id):
        reader_threads.append(threading.get_ident())
        entered.set()
        # A bounded gate avoids timing assertions and real sleeps on the loop.
        if not release.wait(timeout=5):
            raise AssertionError("event loop did not release the database reader")
        return original_get_job(job_id)

    monkeypatch.setattr(store, "_get_job", blocked_read)
    read = asyncio.create_task(store.get_job(admitted.job_id))
    ticks = []
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), timeout=6)
        assert len(reader_threads) == 1
        assert reader_threads[0] != loop_thread
        for i in range(20):
            asyncio.get_running_loop().call_soon(ticks.append, i)
            await asyncio.sleep(0)
        assert ticks == list(range(20))
        assert not read.done()
    finally:
        release.set()
        result = await asyncio.wait_for(read, timeout=6)
    assert result == admitted


async def test_database_schema_enforces_core_checks_uniqueness_and_foreign_key(
    store, db_path
):
    job, _ = await store.admit_job(POLICY, {}, "key", now=NOW)
    with closing(sqlite3.connect(db_path)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(job_events)"
        ).fetchall()
        assert any(
            row[2] == "jobs" and row[3] == "job_id" and row[4] == "job_id"
            for row in foreign_keys
        )
        for assignment in (
            "status = 'INVALID'",
            "attempt_count = -1",
            "max_retries = -1",
            "idempotent = 2",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    f"UPDATE jobs SET {assignment} WHERE job_id = ?", (job.job_id,)
                )
            connection.rollback()
        another, _ = await store.admit_job(POLICY, {}, "another", now=NOW)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE jobs SET idempotency_key = ? WHERE job_id = ?",
                (job.idempotency_key, another.job_id),
            )
        connection.rollback()
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE job_events SET job_id = 'missing' WHERE job_id = ?",
                (job.job_id,),
            )
        connection.rollback()
    assert await store.get_job(job.job_id) == job
