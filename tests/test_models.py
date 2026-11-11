import json
from dataclasses import FrozenInstanceError, replace

import pytest

from runtime.models import (
    BackpressureError,
    IdempotencyConflictError,
    JobEvent,
    JobRecord,
    JobStatus,
    RuntimeConfig,
    StaleLeaseError,
    ToolFatalError,
    ToolPolicy,
    ToolRetryableError,
)


@pytest.mark.parametrize("status", list(JobStatus))
def test_status_values_and_terminal_classification(status: JobStatus):
    assert isinstance(status, str)
    assert status.value == status.name
    assert JobStatus(status.value) is status
    assert status.terminal is (
        status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.UNKNOWN_OUTCOME}
    )


def test_status_set_is_the_phase_one_state_machine():
    assert {status.value for status in JobStatus} == {
        "PENDING",
        "LEASED",
        "RETRY_WAIT",
        "SUCCEEDED",
        "FAILED",
        "UNKNOWN_OUTCOME",
    }


def test_policy_defaults_and_additional_retry_budget():
    policy = ToolPolicy(name="fetch", idempotent=True)
    assert (policy.timeout_ms, policy.max_retries, policy.base_backoff_ms) == (
        2000,
        3,
        50,
    )
    assert policy.max_attempts == 4
    assert replace(policy, max_retries=0).max_attempts == 1
    assert replace(policy, max_retries=2).max_attempts == 3
    assert replace(policy, base_backoff_ms=0).base_backoff_ms == 0


@pytest.mark.parametrize("name", ["", " ", "\t\n", None, 123])
def test_policy_rejects_invalid_names(name):
    with pytest.raises(ValueError):
        ToolPolicy(name=name, idempotent=True)


@pytest.mark.parametrize("idempotent", [None, 0, 1, "true", [], {}])
def test_policy_requires_an_actual_boolean(idempotent):
    with pytest.raises(ValueError):
        ToolPolicy(name="fetch", idempotent=idempotent)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout_ms", 0),
        ("timeout_ms", -1),
        ("max_retries", -1),
        ("base_backoff_ms", -1),
        *[
            (field, value)
            for field in ("timeout_ms", "max_retries", "base_backoff_ms")
            for value in (True, False, 1.5, "1", None)
        ],
    ],
)
def test_policy_rejects_invalid_integer_settings(field, value):
    with pytest.raises(ValueError):
        ToolPolicy(name="fetch", idempotent=True, **{field: value})


def test_config_defaults_and_supported_durability_modes(tmp_path):
    config = RuntimeConfig(db_path=str(tmp_path / "jobs.sqlite3"))
    assert config.max_workers == 8
    assert config.max_outstanding_jobs == 500
    assert config.lease_duration_ms == 1000
    assert config.heartbeat_interval_ms == 300
    assert config.synchronous_mode == "FULL"
    assert replace(config, synchronous_mode="NORMAL").synchronous_mode == "NORMAL"
    assert replace(config, lease_duration_ms=2, heartbeat_interval_ms=1)


@pytest.mark.parametrize("db_path", ["", " ", "\n", ":memory:", None, 123])
def test_config_requires_a_nonempty_file_path(db_path):
    with pytest.raises(ValueError):
        RuntimeConfig(db_path=db_path)


@pytest.mark.parametrize(
    "field",
    [
        "max_workers",
        "max_outstanding_jobs",
        "lease_duration_ms",
        "heartbeat_interval_ms",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "1", None])
def test_config_rejects_invalid_positive_integer_settings(tmp_path, field, value):
    with pytest.raises(ValueError):
        RuntimeConfig(db_path=str(tmp_path / "jobs.sqlite3"), **{field: value})


@pytest.mark.parametrize("heartbeat", [1000, 1001])
def test_config_requires_heartbeat_strictly_shorter_than_lease(tmp_path, heartbeat):
    with pytest.raises(ValueError):
        RuntimeConfig(
            db_path=str(tmp_path / "jobs.sqlite3"),
            lease_duration_ms=1000,
            heartbeat_interval_ms=heartbeat,
        )


@pytest.mark.parametrize("mode", ["full", "normal", "OFF", "EXTRA", "", None, 1])
def test_config_rejects_unsupported_synchronous_modes(tmp_path, mode):
    with pytest.raises(ValueError):
        RuntimeConfig(db_path=str(tmp_path / "jobs.sqlite3"), synchronous_mode=mode)


@pytest.fixture
def record():
    return JobRecord(
        job_id="job-1",
        idempotency_key="request-1",
        policy=ToolPolicy(
            "fetch", True, timeout_ms=321, max_retries=2, base_backoff_ms=17
        ),
        args_json='{"nested":{"items":[1,true,null,"雪"]}}',
        args_hash="fingerprint",
        status=JobStatus.PENDING,
        attempt_count=0,
        lease_owner=None,
        lease_expires_at=None,
        next_eligible_at=1000.0,
        result_json=None,
        error_text=None,
        created_at=1000.0,
        updated_at=1000.0,
        started_at=None,
        completed_at=None,
    )


def test_record_exposes_decoded_args_and_policy_name(record):
    assert record.tool_name == "fetch"
    assert record.args == {"nested": {"items": [1, True, None, "雪"]}}
    assert record.result is None
    decoded = record.args
    decoded["nested"]["items"].append("local mutation")
    assert record.args == {"nested": {"items": [1, True, None, "雪"]}}


@pytest.mark.parametrize(
    "value", [None, False, True, 0, 1.5, "雪", [], {}, [1, {"x": None}]]
)
def test_record_decodes_any_json_result_without_losing_falsy_values(record, value):
    completed = replace(record, result_json=json.dumps(value))
    assert completed.result == value
    assert type(completed.result) is type(value)


def test_result_decode_does_not_expose_shared_mutable_state(record):
    completed = replace(record, result_json='{"items":[1]}')
    completed.result["items"].append(2)
    assert completed.result == {"items": [1]}


def test_models_and_embedded_policy_are_frozen(tmp_path, record):
    event = JobEvent(
        1, record.job_id, 0, "STATE_TRANSITION", None, JobStatus.PENDING, 1000.0, None
    )
    objects_and_fields = [
        (record.policy, "name", "changed"),
        (RuntimeConfig(str(tmp_path / "jobs.sqlite3")), "max_workers", 99),
        (record, "status", JobStatus.FAILED),
        (event, "attempt", 9),
    ]
    for obj, field, value in objects_and_fields:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, value)


@pytest.mark.parametrize(
    ("exception", "base"),
    [
        (IdempotencyConflictError, ValueError),
        (BackpressureError, RuntimeError),
        (StaleLeaseError, RuntimeError),
        (ToolRetryableError, RuntimeError),
        (ToolFatalError, RuntimeError),
    ],
)
def test_domain_errors_preserve_messages_and_expected_base_class(exception, base):
    error = exception("specific failure")
    assert isinstance(error, base)
    assert str(error) == "specific failure"
