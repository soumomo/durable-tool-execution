import asyncio

import pytest

from runtime import (
    DurableToolRuntime,
    JobStatus,
    RuntimeConfig,
    StaleLeaseError,
    ToolRegistry,
)
from runtime import store as store_module

pytestmark = [pytest.mark.asyncio, pytest.mark.acceptance]
WAIT_SECONDS = 5


async def wait_event(event):
    await asyncio.wait_for(event.wait(), WAIT_SECONDS)


async def wait_for_fence(runtime):
    async with asyncio.timeout(WAIT_SECONDS):
        while runtime.fenced_stale_completions == 0:
            await asyncio.sleep(0)


@pytest.mark.parametrize("synchronous_mode", ["FULL", "NORMAL"])
async def test_fenced_lease_prevents_stale_worker_overwrite(
    tmp_path, monkeypatch, synchronous_mode
):
    clock = 1000.0
    original_timestamp = store_module._timestamp

    def timestamp(now):
        return original_timestamp(clock if now is None else now)

    monkeypatch.setattr(store_module, "_timestamp", timestamp)
    first_started, heartbeat_stalled, recovered = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    release_first, release_heartbeat, stale_write = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    first_returned = asyncio.Event()
    calls, recovery_records, winning_claims, stale_errors = [], [], [], []
    registry = ToolRegistry()
    job_id = None

    @registry.tool(idempotent=True, timeout_ms=15_000, max_retries=1, base_backoff_ms=0)
    async def slow_tool():
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            first_started.set()
            await release_first.wait()
            first_returned.set()
            return "worker_a_late_result"
        assert len(calls) == 2, "unexpected third execution"
        assert job_id is not None
        winning_claims.append(await runtime.store.get_job(job_id))
        return "worker_b_result"

    runtime = DurableToolRuntime(
        RuntimeConfig(
            db_path=str(tmp_path / "acceptance.sqlite3"),
            max_workers=2,
            lease_duration_ms=1000,
            heartbeat_interval_ms=10,
            poll_interval_ms=5,
            recovery_interval_ms=5,
            synchronous_mode=synchronous_mode,
        ),
        registry,
    )
    original_heartbeat = runtime.store.heartbeat_lease
    original_recover = runtime.store.recover_expired_leases
    original_complete = runtime.store.complete_job

    async def heartbeat(job_id, attempt_count, **kwargs):
        if attempt_count == 1:
            # Pause only A's heartbeat, leaving the DB and second worker usable.
            heartbeat_stalled.set()
            await release_heartbeat.wait()
        return await original_heartbeat(job_id, attempt_count, **kwargs)

    async def recover(**kwargs):
        records = await original_recover(**kwargs)
        for record in records:
            if record.attempt_count == 1 and record.status is JobStatus.RETRY_WAIT:
                recovery_records.append(record)
                recovered.set()
        return records

    async def complete(job_id, attempt_count, result, **kwargs):
        try:
            # Forward the genuine old epoch, never manufacture an invalid token.
            return await original_complete(job_id, attempt_count, result, **kwargs)
        except StaleLeaseError as exc:
            if attempt_count == 1:
                stale_errors.append(exc)
                stale_write.set()
            raise

    monkeypatch.setattr(runtime.store, "heartbeat_lease", heartbeat)
    monkeypatch.setattr(runtime.store, "recover_expired_leases", recover)
    monkeypatch.setattr(runtime.store, "complete_job", complete)

    async with runtime:
        try:
            job_id = await runtime.submit("slow_tool", {}, "same-logical-job")
            await wait_event(first_started)
            await wait_event(heartbeat_stalled)
            first_claim = await runtime.store.get_job(job_id)
            assert first_claim.status is JobStatus.LEASED
            assert first_claim.attempt_count == 1
            assert first_claim.lease_expires_at is not None

            # Expiration is clock-driven, not ordered by arbitrary test sleeps.
            clock = first_claim.lease_expires_at + 1
            await wait_event(recovered)
            winner = await runtime.wait_for_terminal(job_id, timeout=WAIT_SECONDS)
            assert winner.status is JobStatus.SUCCEEDED
            assert winner.result == "worker_b_result"
            assert winner.attempt_count == 2
            assert not first_returned.is_set()
            assert len(recovery_records) == len(winning_claims) == 1
            assert recovery_records[0].attempt_count == 1
            assert winning_claims[0].attempt_count == 2
            assert winning_claims[0].lease_owner != first_claim.lease_owner

            release_first.set()
            await wait_event(stale_write)
            await wait_for_fence(runtime)
            assert first_returned.is_set()
            assert len(stale_errors) == 1
            assert calls == [1, 2]
            assert runtime.fenced_stale_completions == 1
            assert runtime.lost_leases == 0
            assert await runtime.store.get_job(job_id) == winner
            assert await runtime.store.count_outstanding_jobs() == 0

            events = await runtime.store.get_events(job_id)
            assert [
                (event.from_status, event.to_status, event.attempt)
                for event in events
                if event.event_type == "STATE_TRANSITION"
            ] == [
                (None, JobStatus.PENDING, 0),
                (JobStatus.PENDING, JobStatus.LEASED, 1),
                (JobStatus.LEASED, JobStatus.RETRY_WAIT, 1),
                (JobStatus.RETRY_WAIT, JobStatus.LEASED, 2),
                (JobStatus.LEASED, JobStatus.SUCCEEDED, 2),
            ]
            assert len(events) == 6
            fenced_event = events[-1]
            assert fenced_event.event_type == "STALE_COMPLETION_FENCED"
            assert fenced_event.attempt == 1
            assert fenced_event.from_status is JobStatus.SUCCEEDED
            assert fenced_event.to_status is JobStatus.SUCCEEDED
            assert all(not worker.done() for worker in runtime._workers)
        finally:
            release_first.set()
            release_heartbeat.set()
