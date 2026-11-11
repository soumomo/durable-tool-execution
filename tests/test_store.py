import asyncio
import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from runtime.models import (
    BackpressureError,
    IdempotencyConflictError,
    JobEvent,
    JobRecord,
    JobStatus,
    StaleLeaseError,
    ToolPolicy,
)

pytestmark = pytest.mark.asyncio
POLICY = ToolPolicy("fetch", True)
NOW = 1000.0


def transitions(events):
    return [
        (event.from_status, event.to_status)
        for event in events
        if event.event_type == "STATE_TRANSITION"
    ]


async def leased_job(store, policy=POLICY, *, key="request", now=NOW):
    admitted, created = await store.admit_job(
        policy, {"input": [1, "雪", None]}, key, now=now
    )
    assert created
    claimed = await store.claim_next_job("worker-a", now=now)
    assert claimed is not None
    assert claimed.job_id == admitted.job_id
    return claimed


async def write_outcome(store, operation, job_id, attempt, *, now):
    if operation == "complete":
        return await store.complete_job(job_id, attempt, {"winner": "late"}, now=now)
    return await store.fail_or_retry_job(
        job_id, attempt, "late failure", retryable=True, now=now
    )


async def test_admission_creates_a_decoded_pending_record_and_initial_event(store):
    args = {"z": [True, None, 1.25], "a": {"雪": "value", "b": 2}}
    job, created = await store.admit_job(POLICY, args, "request", now=NOW)
    assert created is True
    assert isinstance(job, JobRecord)
    assert job.job_id
    assert job.idempotency_key == "request"
    assert job.policy == POLICY
    assert job.tool_name == POLICY.name
    assert job.args == args
    canonical_forms = {
        json.dumps(
            args,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ascii_only,
            allow_nan=False,
        )
        for ascii_only in (True, False)
    }
    assert job.args_json in canonical_forms
    assert len(job.args_hash) == 64
    assert len(bytes.fromhex(job.args_hash)) == hashlib.sha256().digest_size
    assert job.status is JobStatus.PENDING
    assert job.attempt_count == 0
    assert job.created_at == job.updated_at == job.next_eligible_at == NOW
    assert job.started_at is job.completed_at is None
    assert job.lease_owner is job.lease_expires_at is None
    assert job.result_json is job.result is job.error_text is None
    assert await store.get_job(job.job_id) == job
    assert await store.list_jobs() == [job]
    assert await store.count_outstanding_jobs() == 1
    events = await store.get_events(job.job_id)
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, JobEvent)
    assert event.event_id > 0
    assert (event.job_id, event.attempt, event.timestamp) == (job.job_id, 0, NOW)
    assert event.event_type == "STATE_TRANSITION"
    assert event.from_status is None
    assert event.to_status is JobStatus.PENDING


async def test_canonical_nested_key_order_deduplicates_without_changing_original_job(
    store,
):
    first_args = {"b": [1, {"d": 2, "c": "雪"}], "a": {"y": False, "x": None}}
    reordered = {"a": {"x": None, "y": False}, "b": [1, {"c": "雪", "d": 2}]}
    first, _ = await store.admit_job(POLICY, first_args, "same", now=NOW)
    duplicate, created = await store.admit_job(POLICY, reordered, "same", now=NOW + 10)
    assert created is False
    assert duplicate == first
    assert len(await store.get_events(first.job_id)) == 1
    separate, created = await store.admit_job(
        POLICY, reordered, "another-key", now=NOW + 20
    )
    assert created is True
    assert separate.job_id != first.job_id
    assert separate.args_hash == first.args_hash


async def test_fingerprint_includes_tool_name_even_with_identical_args(store):
    one, _ = await store.admit_job(POLICY, {"x": 1}, "one", now=NOW)
    other_policy = ToolPolicy("different-tool", True)
    two, _ = await store.admit_job(other_policy, {"x": 1}, "two", now=NOW)
    assert one.args_hash != two.args_hash
    with pytest.raises(IdempotencyConflictError):
        await store.admit_job(other_policy, {"x": 1}, "one", now=NOW)
    assert await store.get_job(one.job_id) == one
    assert len(await store.get_events(one.job_id)) == 1


@pytest.mark.parametrize(
    "changed", [{"x": 2}, {"x": True}, {"x": "1"}, {"x": [1, 2]}, {}]
)
async def test_different_args_for_existing_key_conflict_without_writes(store, changed):
    original, _ = await store.admit_job(POLICY, {"x": 1}, "same", now=NOW)
    with pytest.raises(IdempotencyConflictError):
        await store.admit_job(POLICY, changed, "same", now=NOW + 1)
    assert await store.list_jobs() == [original]
    assert len(await store.get_events(original.job_id)) == 1


async def test_dedup_and_conflict_precede_capacity_and_keep_original_policy(
    store_factory,
):
    store = store_factory(max_outstanding_jobs=1)
    await store.start()
    original_policy = ToolPolicy(
        "fetch", True, timeout_ms=731, max_retries=2, base_backoff_ms=19
    )
    original, _ = await store.admit_job(original_policy, {"x": 1}, "same", now=NOW)
    replacement = ToolPolicy(
        "fetch", False, timeout_ms=1, max_retries=0, base_backoff_ms=0
    )
    duplicate, created = await store.admit_job(
        replacement, {"x": 1}, "same", now=NOW + 1
    )
    assert created is False
    assert duplicate == original
    assert duplicate.policy == original_policy
    with pytest.raises(IdempotencyConflictError):
        await store.admit_job(replacement, {"x": 2}, "same", now=NOW + 1)
    with pytest.raises(IdempotencyConflictError):
        await store.admit_job(ToolPolicy("other", True), {"x": 1}, "same", now=NOW + 1)
    with pytest.raises(BackpressureError):
        await store.admit_job(POLICY, {}, "new-key", now=NOW + 1)
    assert await store.list_jobs() == [original]
    assert len(await store.get_events(original.job_id)) == 1


@pytest.mark.parametrize("connection_count", [1, 2])
async def test_concurrent_admission_deduplicates_atomically(
    store_factory, connection_count
):
    stores = [store_factory(max_outstanding_jobs=1) for _ in range(connection_count)]
    for store in stores:
        await store.start()
    results = await asyncio.gather(
        *[
            stores[i % connection_count].admit_job(POLICY, {"x": 1}, "same", now=NOW)
            for i in range(24)
        ]
    )
    assert sum(created for _, created in results) == 1
    assert len({job.job_id for job, _ in results}) == 1
    assert await stores[0].count_outstanding_jobs() == 1
    assert len(await stores[0].list_jobs()) == 1
    assert len(await stores[0].get_events(results[0][0].job_id)) == 1


@pytest.mark.parametrize("connection_count", [1, 2])
async def test_concurrent_admission_never_exceeds_capacity(
    store_factory, connection_count
):
    stores = [store_factory(max_outstanding_jobs=5) for _ in range(connection_count)]
    for store in stores:
        await store.start()
    results = await asyncio.gather(
        *[
            stores[i % connection_count].admit_job(
                POLICY, {"i": i}, f"key-{i}", now=NOW
            )
            for i in range(24)
        ],
        return_exceptions=True,
    )
    accepted = [result for result in results if isinstance(result, tuple)]
    rejected = [result for result in results if isinstance(result, BackpressureError)]
    assert len(accepted) == 5
    assert len(rejected) == 19
    assert len({job.job_id for job, created in accepted if created}) == 5
    assert await stores[0].count_outstanding_jobs() == 5
    jobs = await stores[0].list_jobs()
    assert len(jobs) == 5
    for job in jobs:
        assert len(await stores[0].get_events(job.job_id)) == 1


async def test_concurrent_conflicting_submissions_have_one_winner(store):
    results = await asyncio.gather(
        store.admit_job(POLICY, {"x": 1}, "same", now=NOW),
        store.admit_job(POLICY, {"x": 2}, "same", now=NOW),
        return_exceptions=True,
    )
    assert sum(isinstance(result, IdempotencyConflictError) for result in results) == 1
    winners = [result for result in results if isinstance(result, tuple)]
    assert len(winners) == 1
    job, created = winners[0]
    assert created is True
    assert await store.list_jobs() == [job]
    assert len(await store.get_events(job.job_id)) == 1


@pytest.mark.parametrize(
    "state", [JobStatus.PENDING, JobStatus.LEASED, JobStatus.RETRY_WAIT]
)
async def test_all_unfinished_states_consume_capacity(store_factory, state):
    store = store_factory(max_outstanding_jobs=1)
    await store.start()
    job, _ = await store.admit_job(POLICY, {}, "first", now=NOW)
    if state is not JobStatus.PENDING:
        job = await store.claim_next_job("worker", now=NOW)
    if state is JobStatus.RETRY_WAIT:
        job = await store.fail_or_retry_job(
            job.job_id, 1, "transient", retryable=True, now=NOW
        )
    assert job.status is state
    assert await store.count_outstanding_jobs() == 1
    with pytest.raises(BackpressureError):
        await store.admit_job(POLICY, {}, "second", now=NOW)
    assert len(await store.list_jobs()) == 1


@pytest.mark.parametrize(
    "terminal", [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.UNKNOWN_OUTCOME]
)
async def test_terminal_jobs_release_capacity_but_remain_deduplicable(
    store_factory, terminal
):
    store = store_factory(max_outstanding_jobs=1)
    await store.start()
    policy = ToolPolicy("tool", terminal is not JobStatus.UNKNOWN_OUTCOME)
    claimed = await leased_job(store, policy)
    if terminal is JobStatus.SUCCEEDED:
        finished = await store.complete_job(
            claimed.job_id, 1, {"cached": True}, now=NOW + 0.1
        )
    else:
        finished = await store.fail_or_retry_job(
            claimed.job_id, 1, "failure", retryable=False, now=NOW + 0.1
        )
    assert finished.status is terminal
    assert finished.completed_at == NOW + 0.1
    assert finished.lease_owner is finished.lease_expires_at is None
    assert await store.count_outstanding_jobs() == 0
    await store.admit_job(POLICY, {}, "second", now=NOW + 1)
    duplicate, created = await store.admit_job(
        policy, claimed.args, "request", now=NOW + 2
    )
    assert created is False
    assert duplicate == finished
    assert len(await store.list_jobs()) == 2
    assert await store.count_outstanding_jobs() == 1


INVALID_JSON = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param((1, 2), id="tuple"),
    pytest.param({1, 2}, id="set"),
    pytest.param(b"bytes", id="bytes"),
    pytest.param(object(), id="object"),
    pytest.param({1: "integer-key"}, id="integer-key"),
    pytest.param({None: "null-key"}, id="null-key"),
    pytest.param({False: "boolean-key"}, id="boolean-key"),
    pytest.param({("tuple",): "key"}, id="tuple-key"),
    pytest.param({"nested": [float("nan")]}, id="nested-nan"),
    pytest.param({"nested": [{2: "not-string"}]}, id="deep-nonstring-key"),
]


@pytest.mark.parametrize("invalid", INVALID_JSON)
async def test_admission_rejects_non_json_values_atomically(store, invalid):
    with pytest.raises((TypeError, ValueError)):
        await store.admit_job(POLICY, {"invalid": invalid}, "key", now=NOW)
    assert await store.list_jobs() == []
    assert await store.count_outstanding_jobs() == 0
    valid, created = await store.admit_job(POLICY, {}, "key", now=NOW)
    assert created is True
    assert len(await store.get_events(valid.job_id)) == 1


@pytest.mark.parametrize("args", [None, [], [1], "text", 42, True, (1, 2)])
async def test_admission_requires_a_json_object_at_the_root(store, args):
    with pytest.raises((TypeError, ValueError)):
        await store.admit_job(POLICY, args, "key", now=NOW)
    assert await store.list_jobs() == []


async def test_cyclic_args_and_results_are_rejected_without_partial_writes(store):
    cyclic_args = {}
    cyclic_args["self"] = cyclic_args
    with pytest.raises((TypeError, ValueError)):
        await store.admit_job(POLICY, cyclic_args, "key", now=NOW)
    assert await store.list_jobs() == []
    claimed = await leased_job(store)
    events = await store.get_events(claimed.job_id)
    cyclic_result = []
    cyclic_result.append(cyclic_result)
    with pytest.raises((TypeError, ValueError)):
        await store.complete_job(claimed.job_id, 1, cyclic_result, now=NOW + 0.1)
    assert await store.get_job(claimed.job_id) == claimed
    assert await store.get_events(claimed.job_id) == events


async def test_claim_populates_lease_and_does_not_claim_leased_work_twice(store):
    admitted, _ = await store.admit_job(POLICY, {}, "key", now=NOW)
    assert await store.claim_next_job("early", now=NOW - 0.001) is None
    claimed = await store.claim_next_job("worker-a", now=NOW)
    assert claimed.job_id == admitted.job_id
    assert claimed.status is JobStatus.LEASED
    assert claimed.attempt_count == 1
    assert claimed.lease_owner == "worker-a"
    assert claimed.lease_expires_at == NOW + 1
    assert claimed.started_at == claimed.updated_at == NOW
    assert claimed.completed_at is None
    assert await store.claim_next_job("worker-b", now=NOW + 0.5) is None
    assert await store.claim_next_job("worker-b", now=NOW + 2) is None
    assert await store.get_job(claimed.job_id) == claimed
    assert transitions(await store.get_events(claimed.job_id)) == [
        (None, JobStatus.PENDING),
        (JobStatus.PENDING, JobStatus.LEASED),
    ]


async def test_claim_uses_configured_lease_duration(store_factory):
    store = store_factory(lease_duration_ms=2500, heartbeat_interval_ms=500)
    await store.start()
    claimed = await leased_job(store)
    assert claimed.lease_expires_at == NOW + 2.5
    assert await store.heartbeat_lease(claimed.job_id, 1, now=NOW + 0.5) is True
    assert (await store.get_job(claimed.job_id)).lease_expires_at == NOW + 3


async def test_claim_orders_by_eligibility_before_creation_time(store):
    older = await leased_job(store, ToolPolicy("slow", True, base_backoff_ms=5000))
    retry = await store.fail_or_retry_job(
        older.job_id, 1, "retry", retryable=True, now=NOW
    )
    newer, _ = await store.admit_job(POLICY, {}, "newer", now=NOW + 1)
    assert retry.next_eligible_at == NOW + 5
    first = await store.claim_next_job("first", now=NOW + 5)
    second = await store.claim_next_job("second", now=NOW + 5)
    assert first.job_id == newer.job_id
    assert second.job_id == older.job_id
    assert second.attempt_count == 2


async def test_claim_breaks_equal_eligibility_ties_by_creation_time(store):
    older = await leased_job(store, ToolPolicy("retry", True, base_backoff_ms=2000))
    await store.fail_or_retry_job(older.job_id, 1, "retry", retryable=True, now=NOW)
    newer, _ = await store.admit_job(POLICY, {}, "newer", now=NOW + 2)
    first = await store.claim_next_job("first", now=NOW + 2)
    second = await store.claim_next_job("second", now=NOW + 2)
    assert first.job_id == older.job_id
    assert second.job_id == newer.job_id


@pytest.mark.parametrize("connection_count", [1, 2])
async def test_concurrent_claims_are_unique_atomic_and_never_claim_terminal_jobs(
    store_factory, connection_count
):
    stores = [store_factory() for _ in range(connection_count)]
    for store in stores:
        await store.start()
    finished = await leased_job(stores[0], key="finished")
    await stores[0].complete_job(finished.job_id, 1, "done", now=NOW)
    pending = [
        (await stores[0].admit_job(POLICY, {"i": i}, f"key-{i}", now=NOW))[0]
        for i in range(12)
    ]
    claims = await asyncio.gather(
        *[
            stores[i % connection_count].claim_next_job(f"worker-{i}", now=NOW)
            for i in range(30)
        ]
    )
    claimed = [job for job in claims if job is not None]
    assert len(claimed) == 12
    assert {job.job_id for job in claimed} == {job.job_id for job in pending}
    assert all(
        job.attempt_count == 1 and job.status is JobStatus.LEASED for job in claimed
    )
    assert len({job.lease_owner for job in claimed}) == 12
    assert await stores[0].count_outstanding_jobs() == 12
    for job in claimed:
        assert len(await stores[0].get_events(job.job_id)) == 2


async def test_heartbeat_renews_only_live_matching_leases_without_state_events(store):
    claimed = await leased_job(store)
    events = await store.get_events(claimed.job_id)
    assert await store.heartbeat_lease(claimed.job_id, 0, now=NOW + 0.25) is False
    assert await store.heartbeat_lease(claimed.job_id, 2, now=NOW + 0.25) is False
    assert await store.get_job(claimed.job_id) == claimed
    assert await store.heartbeat_lease(claimed.job_id, 1, now=NOW + 0.5) is True
    renewed = await store.get_job(claimed.job_id)
    assert renewed.lease_expires_at == NOW + 1.5
    assert renewed.status is JobStatus.LEASED
    assert renewed.attempt_count == 1
    assert renewed.lease_owner == claimed.lease_owner
    assert await store.recover_expired_leases(now=NOW + 1) == []
    assert await store.get_events(claimed.job_id) == events


@pytest.mark.parametrize("offset", [1.0, 1.001])
async def test_expired_lease_cannot_be_resurrected_by_heartbeat(store, offset):
    claimed = await leased_job(store)
    events = await store.get_events(claimed.job_id)
    assert await store.heartbeat_lease(claimed.job_id, 1, now=NOW + offset) is False
    assert await store.get_job(claimed.job_id) == claimed
    assert await store.get_events(claimed.job_id) == events


@pytest.mark.parametrize(
    "result", [None, False, True, 0, 1.25, "雪", [], {}, [1, {"nested": None}]]
)
async def test_completion_persists_all_json_value_types_and_clears_lease(store, result):
    claimed = await leased_job(store)
    completed = await store.complete_job(claimed.job_id, 1, result, now=NOW + 0.25)
    assert completed.status is JobStatus.SUCCEEDED
    assert completed.result == result
    assert type(completed.result) is type(result)
    assert completed.result_json is not None
    assert json.loads(completed.result_json) == result
    assert completed.lease_owner is completed.lease_expires_at is None
    assert completed.completed_at == completed.updated_at == NOW + 0.25
    assert completed.attempt_count == 1
    assert completed.created_at == NOW
    assert await store.get_job(claimed.job_id) == completed
    assert await store.count_outstanding_jobs() == 0
    assert await store.claim_next_job("other", now=NOW + 10) is None
    assert await store.heartbeat_lease(completed.job_id, 1, now=NOW + 0.5) is False


@pytest.mark.parametrize("invalid", INVALID_JSON)
async def test_invalid_result_keeps_live_lease_and_audit_unchanged(store, invalid):
    claimed = await leased_job(store)
    events = await store.get_events(claimed.job_id)
    with pytest.raises((TypeError, ValueError)):
        await store.complete_job(claimed.job_id, 1, invalid, now=NOW + 0.1)
    assert await store.get_job(claimed.job_id) == claimed
    assert await store.get_events(claimed.job_id) == events
    assert (
        await store.complete_job(claimed.job_id, 1, "valid", now=NOW + 0.2)
    ).result == "valid"


@pytest.mark.parametrize("operation", ["complete", "fail"])
@pytest.mark.parametrize("offset", [1.0, 1.001])
async def test_expired_outcomes_are_fenced_before_any_reaper_runs(
    store, operation, offset
):
    claimed = await leased_job(store)
    before = await store.get_events(claimed.job_id)
    with pytest.raises(StaleLeaseError):
        await write_outcome(store, operation, claimed.job_id, 1, now=NOW + offset)
    assert await store.get_job(claimed.job_id) == claimed
    events = await store.get_events(claimed.job_id)
    assert events[:-1] == before
    assert events[-1].event_type == "STALE_COMPLETION_FENCED"
    assert events[-1].job_id == claimed.job_id
    assert events[-1].attempt == 1
    assert events[-1].timestamp == NOW + offset
    assert transitions(events) == transitions(before)


@pytest.mark.parametrize("operation", ["complete", "fail"])
@pytest.mark.parametrize("attempt", [0, 2])
async def test_wrong_epoch_cannot_write_a_live_lease(store, operation, attempt):
    claimed = await leased_job(store)
    with pytest.raises(StaleLeaseError):
        await write_outcome(store, operation, claimed.job_id, attempt, now=NOW + 0.1)
    assert await store.get_job(claimed.job_id) == claimed
    assert (await store.get_events(claimed.job_id))[
        -1
    ].event_type == "STALE_COMPLETION_FENCED"


@pytest.mark.parametrize("operation", ["complete", "fail"])
async def test_pending_work_cannot_be_completed_or_failed_without_a_lease(
    store, operation
):
    pending, _ = await store.admit_job(POLICY, {}, "pending", now=NOW)
    assert await store.heartbeat_lease(pending.job_id, 0, now=NOW) is False
    with pytest.raises(StaleLeaseError):
        await write_outcome(store, operation, pending.job_id, 0, now=NOW)
    assert await store.get_job(pending.job_id) == pending
    assert (await store.get_events(pending.job_id))[
        -1
    ].event_type == "STALE_COMPLETION_FENCED"


@pytest.mark.parametrize("operation", ["complete", "fail"])
async def test_unknown_outcome_write_and_lookup_raise_key_error(store, operation):
    with pytest.raises(KeyError):
        await write_outcome(store, operation, "missing", 1, now=NOW)
    with pytest.raises(KeyError):
        await store.get_job("missing")
    assert await store.heartbeat_lease("missing", 1, now=NOW) is False
    assert await store.list_jobs() == []


@pytest.mark.parametrize("operation", ["complete", "fail"])
async def test_stale_attempt_is_fenced_while_new_attempt_runs_and_after_it_finishes(
    store, operation
):
    first = await leased_job(store)
    recovered = await store.recover_expired_leases(now=NOW + 1)
    assert len(recovered) == 1
    assert await store.heartbeat_lease(first.job_id, 1, now=NOW + 1) is False
    with pytest.raises(StaleLeaseError):
        await write_outcome(store, operation, first.job_id, 1, now=NOW + 1)
    assert await store.get_job(first.job_id) == recovered[0]
    second = await store.claim_next_job("worker-b", now=recovered[0].next_eligible_at)
    assert second.attempt_count == 2
    assert await store.heartbeat_lease(first.job_id, 1, now=second.started_at) is False
    with pytest.raises(StaleLeaseError):
        await write_outcome(store, operation, first.job_id, 1, now=second.started_at)
    assert await store.get_job(first.job_id) == second
    winner = await store.complete_job(
        first.job_id, 2, {"winner": "worker-b"}, now=second.started_at + 0.1
    )
    for attempt in (1, 2):
        with pytest.raises(StaleLeaseError):
            await write_outcome(
                store, operation, first.job_id, attempt, now=second.started_at + 0.2
            )
        assert await store.get_job(first.job_id) == winner
    events = await store.get_events(first.job_id)
    assert sum(event.event_type == "STALE_COMPLETION_FENCED" for event in events) == 4
    assert await store.count_outstanding_jobs() == 0


@pytest.mark.parametrize("max_retries", [0, 2])
async def test_max_retries_counts_additional_attempts_with_exponential_backoff(
    store, max_retries
):
    policy = ToolPolicy("retry", True, max_retries=max_retries, base_backoff_ms=250)
    job, _ = await store.admit_job(policy, {}, "retry", now=NOW)
    now = NOW
    for attempt in range(1, policy.max_attempts + 1):
        claimed = await store.claim_next_job("worker", now=now)
        assert claimed.job_id == job.job_id
        assert claimed.attempt_count == attempt
        failed_at = now + 0.1
        outcome = await store.fail_or_retry_job(
            job.job_id, attempt, f"error-{attempt}", retryable=True, now=failed_at
        )
        assert outcome.error_text == f"error-{attempt}"
        assert outcome.attempt_count == attempt
        assert outcome.lease_owner is outcome.lease_expires_at is None
        if attempt < policy.max_attempts:
            assert outcome.status is JobStatus.RETRY_WAIT
            assert outcome.completed_at is None
            delay = min(250 * 2 ** (attempt - 1), 60000) / 1000
            assert outcome.next_eligible_at == pytest.approx(failed_at + delay)
            assert (
                await store.claim_next_job(
                    "early", now=outcome.next_eligible_at - 0.001
                )
                is None
            )
            now = outcome.next_eligible_at
        else:
            assert outcome.status is JobStatus.FAILED
            assert outcome.completed_at == failed_at
    assert await store.claim_next_job("no-more", now=now + 1000) is None
    assert await store.count_outstanding_jobs() == 0


@pytest.mark.parametrize("random_value", [0.0, 0.25, 1.0])
async def test_full_jitter_is_not_equal_jitter_and_is_capped_before_scaling(
    store_factory, random_value
):
    store = store_factory(random_source=lambda: random_value)
    await store.start()
    claimed = await leased_job(store, ToolPolicy("retry", True, base_backoff_ms=40000))
    retry = await store.fail_or_retry_job(
        claimed.job_id, 1, "one", retryable=True, now=NOW
    )
    assert retry.next_eligible_at == pytest.approx(NOW + 40 * random_value)
    second = await store.claim_next_job("worker-b", now=retry.next_eligible_at)
    assert second.attempt_count == 2
    capped = await store.fail_or_retry_job(
        claimed.job_id, 2, "two", retryable=True, now=retry.next_eligible_at
    )
    assert capped.next_eligible_at == pytest.approx(
        retry.next_eligible_at + 60 * random_value
    )


@pytest.mark.parametrize(
    ("idempotent", "retryable", "outcome_unknown", "max_retries", "expected"),
    [
        (False, True, True, 0, JobStatus.UNKNOWN_OUTCOME),
        (False, False, True, 5, JobStatus.UNKNOWN_OUTCOME),
        (False, True, True, 5, JobStatus.UNKNOWN_OUTCOME),
        (False, True, False, 5, JobStatus.FAILED),
        (False, False, False, 5, JobStatus.FAILED),
        (True, False, True, 5, JobStatus.FAILED),
        (True, False, False, 5, JobStatus.FAILED),
        (True, True, True, 0, JobStatus.FAILED),
        (True, True, False, 0, JobStatus.FAILED),
        (True, True, True, 2, JobStatus.RETRY_WAIT),
        (True, True, False, 2, JobStatus.RETRY_WAIT),
    ],
)
async def test_failure_policy_branching(
    store, idempotent, retryable, outcome_unknown, max_retries, expected
):
    policy = ToolPolicy("tool", idempotent, max_retries=max_retries)
    claimed = await leased_job(store, policy)
    outcome = await store.fail_or_retry_job(
        claimed.job_id,
        1,
        "explicit failure",
        retryable=retryable,
        outcome_unknown=outcome_unknown,
        now=NOW + 0.1,
    )
    assert outcome.status is expected
    assert outcome.error_text == "explicit failure"
    assert outcome.lease_owner is outcome.lease_expires_at is None
    assert outcome.completed_at == (NOW + 0.1 if expected.terminal else None)
    assert await store.count_outstanding_jobs() == (0 if expected.terminal else 1)
    assert transitions(await store.get_events(claimed.job_id))[-1] == (
        JobStatus.LEASED,
        expected,
    )


async def test_omitted_outcome_unknown_is_conservative_for_non_idempotent_tools(store):
    claimed = await leased_job(store, ToolPolicy("charge", False, max_retries=10))
    outcome = await store.fail_or_retry_job(
        claimed.job_id, 1, "timeout", retryable=True, now=NOW
    )
    assert outcome.status is JobStatus.UNKNOWN_OUTCOME
    assert await store.claim_next_job("never-retry", now=NOW + 1000) is None


@pytest.mark.parametrize(
    ("idempotent", "max_retries", "expected"),
    [
        (True, 2, JobStatus.RETRY_WAIT),
        (True, 0, JobStatus.FAILED),
        (False, 0, JobStatus.UNKNOWN_OUTCOME),
        (False, 5, JobStatus.UNKNOWN_OUTCOME),
    ],
)
async def test_recovery_uses_inclusive_expiry_policy_budget_and_is_repeatable(
    store, idempotent, max_retries, expected
):
    policy = ToolPolicy(
        "tool", idempotent, max_retries=max_retries, base_backoff_ms=250
    )
    claimed = await leased_job(store, policy)
    events_before = await store.get_events(claimed.job_id)
    assert await store.recover_expired_leases(now=NOW + 0.999) == []
    assert await store.get_job(claimed.job_id) == claimed
    recovered = await store.recover_expired_leases(now=NOW + 1)
    assert len(recovered) == 1
    outcome = recovered[0]
    assert outcome.status is expected
    assert outcome.attempt_count == 1
    assert outcome.policy == policy
    assert outcome.lease_owner is outcome.lease_expires_at is None
    assert outcome.error_text
    assert outcome.completed_at == (NOW + 1 if expected.terminal else None)
    if expected is JobStatus.RETRY_WAIT:
        assert outcome.next_eligible_at == pytest.approx(NOW + 1.25)
    events = await store.get_events(claimed.job_id)
    assert events[:-1] == events_before
    assert events[-1].event_type == "STATE_TRANSITION"
    assert events[-1].from_status is JobStatus.LEASED
    assert events[-1].to_status is expected
    assert events[-1].attempt == 1
    assert events[-1].timestamp == NOW + 1
    assert await store.recover_expired_leases(now=NOW + 100) == []
    assert await store.get_job(claimed.job_id) == outcome
    assert await store.get_events(claimed.job_id) == events


async def test_recovery_does_not_reset_retry_budget_or_increment_attempts(store):
    policy = ToolPolicy("tool", True, max_retries=2, base_backoff_ms=0)
    await store.admit_job(policy, {}, "key", now=NOW)
    for attempt in (1, 2, 3):
        now = NOW + 2 * (attempt - 1)
        claimed = await store.claim_next_job("worker", now=now)
        assert claimed.attempt_count == attempt
        recovered = await store.recover_expired_leases(now=now + 1)
        assert len(recovered) == 1
        assert recovered[0].attempt_count == attempt
        assert recovered[0].status is (
            JobStatus.FAILED if attempt == 3 else JobStatus.RETRY_WAIT
        )
    assert await store.claim_next_job("done", now=NOW + 100) is None


@pytest.mark.parametrize("random_value", [0.0, 0.25, 1.0])
async def test_recovery_uses_attempt_based_full_jitter_and_the_same_cap(
    store_factory, random_value
):
    store = store_factory(random_source=lambda: random_value)
    await store.start()
    policy = ToolPolicy("retry", True, max_retries=2, base_backoff_ms=40000)
    first = await leased_job(store, policy)
    recovered = (await store.recover_expired_leases(now=first.lease_expires_at))[0]
    assert recovered.next_eligible_at == pytest.approx(
        first.lease_expires_at + 40 * random_value
    )
    second = await store.claim_next_job("worker-b", now=recovered.next_eligible_at)
    assert second.attempt_count == 2
    recovered_again = (await store.recover_expired_leases(now=second.lease_expires_at))[
        0
    ]
    assert recovered_again.attempt_count == 2
    assert recovered_again.next_eligible_at == pytest.approx(
        second.lease_expires_at + 60 * random_value
    )


async def test_concurrent_reapers_do_not_duplicate_transitions(store_factory):
    first_store, second_store = store_factory(), store_factory()
    await first_store.start()
    await second_store.start()
    claimed = await leased_job(first_store)
    results = await asyncio.gather(
        first_store.recover_expired_leases(now=NOW + 1),
        second_store.recover_expired_leases(now=NOW + 1),
    )
    assert sum(len(result) for result in results) == 1
    job = await first_store.get_job(claimed.job_id)
    assert job.status is JobStatus.RETRY_WAIT
    assert job.attempt_count == 1
    assert len(await first_store.get_events(claimed.job_id)) == 3


async def test_recovery_sweeps_only_expired_leased_jobs(store):
    expired = await leased_job(store, key="expired")
    live, _ = await store.admit_job(POLICY, {}, "live", now=NOW + 0.5)
    live = await store.claim_next_job("live-worker", now=NOW + 0.5)
    pending, _ = await store.admit_job(POLICY, {}, "pending", now=NOW + 0.5)
    recovered = await store.recover_expired_leases(now=NOW + 1)
    assert {job.job_id for job in recovered} == {expired.job_id}
    assert await store.get_job(live.job_id) == live
    assert await store.get_job(pending.job_id) == pending


@pytest.mark.acceptance
async def test_audit_log_exact_chronology_includes_attempts_but_not_heartbeats(store):
    claimed = await leased_job(store)
    assert await store.heartbeat_lease(claimed.job_id, 1, now=NOW + 0.1)
    retry = await store.fail_or_retry_job(
        claimed.job_id, 1, "transient", retryable=True, now=NOW + 0.2
    )
    second = await store.claim_next_job("worker-b", now=retry.next_eligible_at)
    await store.complete_job(
        claimed.job_id, 2, {"done": True}, now=second.started_at + 0.1
    )
    events = await store.get_events(claimed.job_id)
    assert transitions(events) == [
        (None, JobStatus.PENDING),
        (JobStatus.PENDING, JobStatus.LEASED),
        (JobStatus.LEASED, JobStatus.RETRY_WAIT),
        (JobStatus.RETRY_WAIT, JobStatus.LEASED),
        (JobStatus.LEASED, JobStatus.SUCCEEDED),
    ]
    assert all(event.event_type == "STATE_TRANSITION" for event in events)
    assert [event.attempt for event in events] == [0, 1, 1, 2, 2]
    assert [event.timestamp for event in events] == [
        NOW,
        NOW,
        NOW + 0.2,
        retry.next_eligible_at,
        second.started_at + 0.1,
    ]
    assert all(event.job_id == claimed.job_id for event in events)
    ids = [event.event_id for event in events]
    assert ids == sorted(set(ids))
    assert len(ids) == 5


async def test_event_queries_filter_interleaved_jobs_and_order_by_event_id(store):
    first, _ = await store.admit_job(POLICY, {}, "first", now=NOW)
    second, _ = await store.admit_job(POLICY, {}, "second", now=NOW + 0.1)
    claimed_first = await store.claim_next_job("worker-a", now=NOW + 0.2)
    claimed_second = await store.claim_next_job("worker-b", now=NOW + 0.3)
    assert claimed_first.job_id == first.job_id
    assert claimed_second.job_id == second.job_id
    await store.complete_job(second.job_id, 1, "second", now=NOW + 0.4)
    await store.complete_job(first.job_id, 1, "first", now=NOW + 0.5)
    all_ids = []
    for job in (first, second):
        events = await store.get_events(job.job_id)
        assert len(events) == 3
        assert all(event.job_id == job.job_id for event in events)
        ids = [event.event_id for event in events]
        assert ids == sorted(ids)
        all_ids.extend(ids)
    assert len(set(all_ids)) == 6


@pytest.mark.parametrize("operation", ["complete", "fail"])
@pytest.mark.parametrize("terminal", [JobStatus.FAILED, JobStatus.UNKNOWN_OUTCOME])
async def test_terminal_failure_cannot_be_overwritten_by_the_same_attempt(
    store, operation, terminal
):
    policy = ToolPolicy("tool", terminal is JobStatus.FAILED)
    claimed = await leased_job(store, policy)
    finished = await store.fail_or_retry_job(
        claimed.job_id, 1, "final", retryable=False, now=NOW + 0.1
    )
    assert finished.status is terminal
    assert await store.heartbeat_lease(claimed.job_id, 1, now=NOW + 0.2) is False
    with pytest.raises(StaleLeaseError):
        await write_outcome(store, operation, claimed.job_id, 1, now=NOW + 0.2)
    assert await store.get_job(claimed.job_id) == finished
    assert (await store.get_events(claimed.job_id))[
        -1
    ].event_type == "STALE_COMPLETION_FENCED"


async def test_returned_args_results_and_policy_cannot_mutate_persisted_state(store):
    args = {"items": [{"value": 1}]}
    admitted, _ = await store.admit_job(POLICY, args, "key", now=NOW)
    args["items"][0]["value"] = 99
    admitted.args["items"].append("local")
    assert (await store.get_job(admitted.job_id)).args == {"items": [{"value": 1}]}
    with pytest.raises(FrozenInstanceError):
        admitted.policy.max_retries = 99
    claimed = await store.claim_next_job("worker", now=NOW)
    result = {"items": [1]}
    completed = await store.complete_job(claimed.job_id, 1, result, now=NOW)
    result["items"].append(2)
    completed.result["items"].append(3)
    assert (await store.get_job(claimed.job_id)).result == {"items": [1]}
